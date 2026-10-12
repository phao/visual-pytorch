"""Pure new persisted policy admission, identities, scoped ownership and refusal."""
import copy
import hashlib
import json
import sqlite3
import subprocess
import sys
import tempfile
import unittest
import io
from contextlib import redirect_stdout
from pathlib import Path
from app.core import GraphError, Workspace, VERSION, SUPPORTED_VERSIONS, check_document, safety_report
from app.evidence import identity, parameter_schema
from app.policy_graph import declaration, owner_key, plan
from app.revisions import model_document, validate_model_document
from policy_runtime_fixture import models as fixture


class PolicyGraphTests(unittest.TestCase):
    def test_new_counts_component_ids_and_complete_frozen_version(self):
        models, mid = fixture(); policy = plan(models, mid)
        self.assertEqual(policy['resolved']['counts'], {'stored': 320, 'trainable': 64, 'frozen': 256})
        self.assertEqual(set(o['id'] for o in policy['schema'].values()),
                         {owner_key(['lora', n, 'weight']) for n in ('base', 'A', 'B')})
        frozen = model_document(models, mid); self.assertEqual(frozen['version'], 6)
        for version in range(1, 6):
            bad = dict(frozen, version=version)
            with self.subTest(version=version), self.assertRaises(GraphError): validate_model_document(bad)

    def test_change_policy_is_scientific_change_but_presentation_is_not(self):
        models, mid = fixture(); before = identity(models, mid)
        models[mid]['graph']['nodes'][1].update(x=20, collapsed=True)
        self.assertEqual(before, identity(models, mid))
        models[mid]['graph']['parameter_policy']['parameters'][owner_key(['base', 'weight'])]['trainable'] = True
        self.assertNotEqual(before['architecture'], identity(models, mid)['architecture'])

    def test_nested_scopes_combine_initializer_and_freezing_and_shared_aliases(self):
        from test_shared_parameters import serial
        models = serial(kind='Module')
        models['inner']['graph']['parameter_policy'] = {'version': 1, 'parameters':
            {owner_key(['use0', 'weight']): {'initializer': {'kind': 'zeros'}}}}
        models['shared']['graph']['parameter_policy'] = {'version': 1, 'parameters':
            {owner_key(['use2', 'use1', 'weight']): {'trainable': False}}}
        resolved = plan(models, 'shared')['resolved']
        owner = resolved['owners'][owner_key(['shared', 'use0', 'use0', 'weight'])]
        self.assertFalse(owner['trainable']); self.assertEqual(owner['initializer'], {'kind': 'zeros'})
        self.assertEqual(resolved['counts'], {'stored': 6, 'trainable': 2, 'frozen': 4})
        models['inner']['graph']['parameter_policy']['parameters'][owner_key(['use0', 'weight'])]['trainable'] = True
        with self.assertRaisesRegex(GraphError, 'conflicting explicit'): plan(models, 'shared')

    def test_component_ids_remove_slash_collisions_without_reinterpreting_old_models(self):
        from test_shared_parameters import serial
        models = serial(depth=1, shared=False, kind='Module')
        root = models['shared']; child = models['inner']; root['graph']['nodes'][1]['id'] = 'a'
        root['graph']['edges'] = [{'from': 'input', 'to': 'a'}, {'from': 'a', 'to': 'output'}]
        child['graph']['nodes'][1]['id'] = 'b'; child['graph']['nodes'][2]['share_with'] = 'b'
        child['graph']['edges'] = [{'from': 'input', 'to': 'b'}, {'from': 'b', 'to': 'use1'}, {'from': 'use1', 'to': 'output'}]
        root['graph']['nodes'].insert(1, {'id': 'a/b', 'type': 'Linear', 'features': 2})
        root['graph']['edges'] = [{'from': 'input', 'to': 'a/b'}, {'from': 'a/b', 'to': 'a'}, {'from': 'a', 'to': 'output'}]
        legacy = parameter_schema(models, 'shared')
        self.assertEqual(len({e['id'] for e in legacy.values()}), 2)
        root['graph']['parameter_policy'] = {'version': 1, 'parameters': {owner_key(['a/b', 'weight']): {'trainable': False}}}
        resolved = plan(models, 'shared')['resolved']; self.assertEqual(len(resolved['owners']), 4)
        self.assertNotEqual(owner_key(['shared', 'a/b', 'weight']), owner_key(['shared', 'a', 'b', 'weight']))

    def test_bad_closed_declarations_and_unknown_shapes_fail_before_numerics(self):
        bad = [None, {'version': True, 'parameters': {}}, {'version': 2, 'parameters': {}},
               {'version': 1, 'parameters': {'base/weight': {}}},
               {'version': 1, 'parameters': {'["base", "weight"]': {}}},
               {'version': 1, 'parameters': {owner_key(['base', 'weight']): {'trainable': 0}}},
               {'version': 1, 'parameters': {}, 'code': 'print(1)'}]
        for value in bad:
            with self.subTest(value=value), self.assertRaises(GraphError): declaration(value)
        models, mid = fixture()
        models[mid]['graph']['parameter_policy']['parameters'][owner_key(['missing', 'weight'])] = {}
        with self.assertRaisesRegex(GraphError, 'unknown'): safety_report(models, mid)

    def test_container24_and_older_policy_containers_refuse_without_disk_write(self):
        self.assertEqual(VERSION, 24); self.assertNotIn(21, SUPPORTED_VERSIONS); self.assertNotIn(22, SUPPORTED_VERSIONS)
        models, mid = fixture()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'lora.vtw'; workspace = Workspace()
            try: workspace.put(models[mid]); workspace.freeze_model(mid); workspace.save(path)
            finally: workspace.db.close()
            reopened = Workspace.load(path)
            try: self.assertEqual(reopened.models()[mid]['graph']['parameter_policy'], models[mid]['graph']['parameter_policy'])
            finally: reopened.db.close()
            for version in (20, 21, 22, 23):
                with sqlite3.connect(path) as db: db.execute('PRAGMA user_version=' + str(version))
                before = hashlib.sha256(path.read_bytes()).hexdigest()
                with self.subTest(version=version), self.assertRaises(GraphError): Workspace.load(path)
                self.assertEqual(before, hashlib.sha256(path.read_bytes()).hexdigest())

    def test_fresh_policy_discovery_imports_no_torch_numpy_or_Qt(self):
        source = ('import sys,json;from policy_runtime_fixture import models;from app.policy_graph import plan;'
                  'm,k=models();p=plan(m,k);print(json.dumps([n for n in sys.modules if n.split(".")[0] in ("torch","numpy","PyQt6")]))')
        process = subprocess.run([sys.executable, '-c', source], capture_output=True, text=True, timeout=20)
        self.assertEqual(process.returncode, 0, process.stderr); self.assertEqual(json.loads(process.stdout), [])

    def test_CLI_policy_edits_discovery_and_undo_preserve_scientific_history(self):
        import vpw_cli
        models, mid = fixture(); workspace = Workspace()
        try:
            workspace.put(models[mid]); before = copy.deepcopy(workspace.models()[mid])
            policy = copy.deepcopy(before['graph']['parameter_policy'])
            policy['parameters'][owner_key(['base', 'weight'])]['trainable'] = True
            snapshot = workspace.history.capture(workspace.db)
            vpw_cli.apply_edits(workspace, [{'op': 'set-parameter-policy', 'id': mid, 'policy': policy}])
            workspace.history.record(snapshot, workspace.db)
            self.assertTrue(plan(workspace.models(), mid)['resolved']['counts']['frozen'] == 0)
            # Workspace history records the actual owned edit, not fake tensor execution.
            self.assertTrue(workspace.history.summary()['undo'])
            workspace.history.restore(workspace)
            self.assertEqual(workspace.models()[mid]['graph']['parameter_policy'], before['graph']['parameter_policy'])
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'policy.vtw'; workspace.save(path)
                output = io.StringIO()
                with redirect_stdout(output): status = vpw_cli.main([str(path), 'parameter-policy', mid])
                self.assertEqual(status, 0); self.assertEqual(json.loads(output.getvalue())['value']['counts']['trainable'], 64)
        finally: workspace.db.close()


if __name__ == '__main__': unittest.main()
