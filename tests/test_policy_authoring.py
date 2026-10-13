"""Pure guarded Inspector owner edits use the real persisted policy contract."""
import copy
import json
import subprocess
import sys
import unittest
from app.core import GraphError, default_model
from app.policy_authoring import prepare, build, reset_block, block_summary, TOKEN
from app.policy_graph import owner_key, plan
from policy_runtime_fixture import models as fixture


class PolicyAuthoringTests(unittest.TestCase):
    def test_plain_model_default_noop_freeze_and_reset(self):
        model = default_model(); model['id'] = 'm'; models = {'m': model}
        info = prepare(models, 'm', 'linear1'); token = {k: info[k] for k in TOKEN}
        self.assertEqual(len(info['owners']), 2)
        weight = next(row for row in info['owners'] if row['selector'] == owner_key(['linear1', 'weight']))
        self.assertEqual(build(models, token, weight['selector'], '{"trainable":true}'), model)
        changed = build(models, token, weight['selector'], '{"trainable":false}'); models['m'] = changed
        self.assertEqual(plan(models, 'm')['resolved']['counts'], {'stored': 3, 'trainable': 1, 'frozen': 2})
        refreshed = prepare(models, 'm', 'linear1'); token = {k: refreshed[k] for k in TOKEN}
        reset = build(models, token, weight['selector'], '{"trainable":null,"initializer":null}')
        models['m'] = reset; self.assertEqual(plan(models, 'm')['resolved']['counts']['trainable'], 3)

    def test_initializer_shape_validation_replay_wrong_owner_and_closed_json_refuse(self):
        models, mid = fixture(); before = copy.deepcopy(models)
        info = prepare(models, mid, 'A'); token = {k: info[k] for k in TOKEN}; selector = info['owners'][0]['selector']
        for text in ('{"trainable":0}', '{"trainable":true,"trainable":false}', 'NaN',
                     '{"initializer":{"kind":"values","dtype":"float32","values":[1]}}',
                     '{"optimizer_resume":true}', 'x' * 65537):
            with self.subTest(text=text[:80]), self.assertRaises(GraphError): build(models, token, selector, text)
            self.assertEqual(models, before)
        with self.assertRaisesRegex(GraphError, 'not owned'):
            build(models, token, owner_key(['base', 'weight']), '{"trainable":true}')
        changed = build(models, token, selector, '{"initializer":{"kind":"zeros"}}'); models[mid] = changed
        with self.assertRaisesRegex(GraphError, 'changed'): build(models, token, selector, '{"trainable":false}')

    def test_shared_owners_and_nested_definition_links(self):
        from test_shared_parameters import serial
        models = serial(kind='Module'); info = prepare(models, 'shared', 'use2')
        self.assertEqual(len(info['owners']), 2)
        self.assertTrue(all(row['shared'] for row in info['owners']))
        self.assertTrue(all(row['definition'] == {'model': 'inner', 'node': 'use0'} for row in info['owners']))
        token = {k: info[k] for k in TOKEN}; selector = next(row['selector'] for row in info['owners'] if row['selector'].endswith('"weight"]'))
        changed = build(models, token, selector, '{"trainable":false}'); models['shared'] = changed
        summary = block_summary(models, 'shared')
        self.assertEqual(summary['use2']['frozen'], 4); self.assertEqual(summary['use2']['owner'], 'use0')
        self.assertEqual(summary['use0']['trainable'], 2)

    def test_ref_conflicting_policy_is_not_silently_overridden(self):
        from test_shared_parameters import serial
        models = serial(kind='Module')
        models['inner']['graph']['parameter_policy'] = {'version': 1, 'parameters': {owner_key(['use0', 'weight']): {'trainable': False}}}
        info = prepare(models, 'shared', 'use0'); token = {k: info[k] for k in TOKEN}
        selector = next(row['selector'] for row in info['owners'] if row['selector'].endswith('"weight"]'))
        with self.assertRaisesRegex(GraphError, 'conflicting explicit'): build(models, token, selector, '{"trainable":true}')

    def test_large_exact_values_are_summarized_not_sent_to_Inspector(self):
        model = default_model(inputs=256, outputs=256); model['id'] = 'large'
        model['graph']['parameter_policy'] = {'version': 1, 'parameters': {owner_key(['linear1', 'weight']):
            {'initializer': {'kind': 'values', 'dtype': 'float32', 'values': [[0.] * 256 for _ in range(256)]}}}}
        info = prepare({'large': model}, 'large', 'linear1')
        weight = next(row for row in info['owners'] if row['initializer_kind'] == 'values')
        self.assertIsNone(weight['initializer_json']); self.assertEqual(weight['shape'], [256, 256])

    def test_valid_tensor_graph_with_outdated_initializer_can_be_repaired(self):
        model = default_model(); model['id'] = 'm'
        model['graph']['parameter_policy'] = {'version': 1, 'parameters': {owner_key(['linear1', 'weight']):
            {'initializer': {'kind': 'values', 'dtype': 'float32', 'values': [[1., 2.]]}}}}
        model['graph']['nodes'][1]['features'] = 3
        models = {'m': model}; info = prepare(models, 'm', 'linear1')
        self.assertIn('shape mismatch', info['policy_error'])
        self.assertTrue(all(not row['effective_available'] for row in info['owners']))
        token = {k: info[k] for k in TOKEN}
        repaired = build(models, token, owner_key(['linear1', 'weight']), '{"initializer":{"kind":"zeros"}}')
        models['m'] = repaired
        self.assertIsNone(prepare(models, 'm', 'linear1')['policy_error'])

    def test_fresh_queries_and_edits_import_no_numerical_or_Qt_packages(self):
        script = ('import sys,json;from app.core import default_model;from app.policy_authoring import prepare,build,TOKEN;'
                  'm=default_model();ms={m["id"]:m};p=prepare(ms,m["id"],"linear1");'
                  'build(ms,{k:p[k] for k in TOKEN},p["owners"][0]["selector"],\'{"trainable":false}\');'
                  'print(json.dumps([n for n in sys.modules if n.split(".")[0] in ("torch","numpy","PyQt6")]))')
        result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr); self.assertEqual(json.loads(result.stdout), [])

    def test_alias_declarations_edit_and_reset_through_either_use(self):
        from test_shared_parameters import serial
        for nid in ('use0', 'use2'):
            models = serial()
            models['shared']['graph']['parameter_policy'] = {'version': 1, 'parameters': {
                owner_key(['use2', 'weight']): {'trainable': False},
                owner_key(['use0', 'weight']): {'trainable': False}}}
            info = prepare(models, 'shared', nid); token = {k: info[k] for k in TOKEN}
            selected = next(row for row in info['owners'] if row['selector'].endswith('"weight"]'))
            changed = build(models, token, selected['selector'], '{"trainable":true}')
            self.assertTrue(all(entry['trainable'] for entry in changed['graph']['parameter_policy']['parameters'].values()))
            models['shared'] = changed
            self.assertEqual(block_summary(models, 'shared')['use2']['frozen'], 0)
            token = {k: prepare(models, 'shared', nid)[k] for k in TOKEN}
            reset = build(models, token, selected['selector'], '{"trainable":null}')
            self.assertEqual(reset['graph']['parameter_policy']['parameters'], {})

    def test_atomic_block_reset_repairs_multiple_outdated_initializers(self):
        m = default_model(); m['id'] = 'm'
        m['graph']['parameter_policy'] = {'version': 1, 'parameters': {
            owner_key(['linear1', 'weight']): {'initializer': {'kind': 'values', 'dtype': 'float32', 'values': [[1., 2.]]}},
            owner_key(['linear1', 'bias']): {'initializer': {'kind': 'values', 'dtype': 'float32', 'values': [0.]}}}}
        m['graph']['nodes'][1]['features'] = 3; models = {'m': m}; before = copy.deepcopy(models)
        info = prepare(models, 'm', 'linear1'); token = {k: info[k] for k in TOKEN}
        for row in info['owners']:
            with self.assertRaises(GraphError): build(models, token, row['selector'], '{"initializer":null}')
        self.assertEqual(models, before)
        changed = reset_block(models, token); models['m'] = changed
        self.assertIsNone(prepare(models, 'm', 'linear1')['policy_error'])
        self.assertEqual(changed['graph']['parameter_policy']['parameters'], {})

    def test_orphaned_bias_and_alias_are_explicitly_reset(self):
        from test_shared_parameters import serial
        models = serial()
        m = models['shared']; m['graph']['parameter_policy'] = {'version': 1, 'parameters': {
            owner_key(['use2', 'bias']): {'trainable': False},
            owner_key(['use0', 'weight']): {'trainable': False}}}
        for n in m['graph']['nodes']:
            if n['type'] == 'Linear': n['bias'] = False
        info = prepare(models, 'shared', 'use2'); self.assertIn('unknown', info['policy_error'])
        self.assertTrue(info['has_local_policy'])
        token = {k: info[k] for k in TOKEN}; changed = reset_block(models, token)
        models['shared'] = changed
        self.assertIsNone(prepare(models, 'shared', 'use2')['policy_error'])

    def test_block_reset_preserves_other_blocks_and_descendant_declarations(self):
        models, mid = fixture(); before = copy.deepcopy(models[mid]['graph']['parameter_policy']['parameters'])
        info = prepare(models, mid, 'A'); token = {k: info[k] for k in TOKEN}
        changed = reset_block(models, token)
        self.assertEqual(changed['graph']['parameter_policy']['parameters'],
            {key: entry for key, entry in before.items() if json.loads(key)[0] != 'A'})
        with self.assertRaisesRegex(GraphError, 'changed'): reset_block(models, {**token, 'architecture': 'wrong'})


if __name__ == '__main__': unittest.main()
