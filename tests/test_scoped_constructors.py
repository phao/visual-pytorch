"""Allocation-free local constructor contexts, immutable evidence and agent edits."""
import copy
import json
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from app.core import GraphError, VERSION, Workspace, generate_code
from app.families import preview, specialize, validate_family
from app.family_records import retain
from app.revisions import validate_all, frozen_document
from app.authoring_sessions import Sessions
from app.experiment_objects import object_reference, preview_experiment, candidate_bundle
from test_family_axes import family_experiment

ROOT = Path(__file__).resolve().parents[1]


def scoped():
    return json.loads((ROOT / 'examples/families/scoped-branches.json').read_text())


def scoped_experiment():
    w, e = family_experiment(); source = scoped()
    for mid in ('parent', 'child'): source['models'][mid]['graph']['nodes'][0]['features'] = 1
    ref = w.put_spec('family', source)
    e.pop('revision'); e.update(version=3, id='scoped-experiment', models=[{
        'id': 'scoped', 'family': ref, 'bindings': {}, 'axes': [
            {'symbol': 'left-width', 'values': [2, 3]}, {'symbol': 'right-width', 'values': [3, 5]}]}])
    ref = w.put_spec('experiment', e)
    return w, w.get_spec('experiment', ref['id'], ref['revision'])


def modules(resolved):
    return [n for n in resolved['models'][resolved['root']]['graph']['nodes'] if n['type'] == 'Module']


def same_context(source):
    for c in source['constructors']:
        if c['node'] == 'right': c['value'] = copy.deepcopy(next(x['value'] for x in source['constructors']
                                                              if x['node'] == 'left' and x['symbol'] == c['symbol']))
    # Keep the unused right-width symbol out of this deliberately equal family.
    source['symbols'] = [s for s in source['symbols'] if s['id'] != 'right-width']
    return source


class ScopedConstructorTests(unittest.TestCase):
    def workspace(self):
        w = Workspace(); self.addCleanup(w.db.close)
        self.addCleanup(lambda: w.history.directory.cleanup() if w.history.directory else None)
        return w

    def test_required_child_arguments_are_satisfied_only_by_calls(self):
        source = scoped(); before = copy.deepcopy(source); r = specialize(source)
        self.assertEqual(preview(source)['status'], 'verified'); self.assertEqual(source, before)
        self.assertEqual(r['version'], 2); self.assertEqual(r['bindings'], {'left-width': 3, 'right-width': 5})
        self.assertEqual(r['contract']['parameters'], 30); self.assertEqual(len(r['models']), 3)
        children = [i for i in r['instances'] if i['template'] == 'child']
        self.assertEqual([i['bindings'] for i in children], [
            {'child-width': 3, 'child-activation': 'Tanh', 'child-bias': False},
            {'child-width': 5, 'child-activation': 'ReLU', 'child-bias': True}])
        self.assertEqual(r, specialize(copy.deepcopy(source)))

    def test_same_name_does_not_bind_across_scopes_and_missing_path_is_actionable(self):
        source = scoped(); source['constructors'] = [c for c in source['constructors']
                                                   if not (c['node'] == 'left' and c['symbol'] == 'child-width')]
        source['assignments'].append({'model': 'parent', 'node': 'head', 'field': 'features', 'value': {'symbol': 'left-width'}})
        result = preview(source); self.assertEqual(result['status'], 'conditional')
        self.assertEqual(result['missing'], [{'id': 'child-width', 'scope': 'child', 'name': 'width', 'path': 'parent/left'}])
        self.assertEqual(preview(source, {'child-width': 4})['status'], 'verified')
        with self.assertRaisesRegex(GraphError, 'parent/left'): specialize(source)

    def test_explicit_call_overrides_supplied_fallback_and_defaults(self):
        source = scoped(); source['symbols'][2]['default'] = 2
        r = specialize(source, {'child-width': 7, 'child-bias': True, 'child-activation': 'Sigmoid'})
        self.assertEqual(r['bindings']['child-width'], 7)
        children = [i for i in r['instances'] if i['template'] == 'child']
        self.assertEqual([i['bindings']['child-width'] for i in children], [3, 5])
        self.assertEqual(r['contract']['parameters'], 30)
        self.assertEqual(preview(source, {'child-width': 0})['status'], 'incompatible')

    def test_nested_local_expression_and_boolean_enum_forwarding(self):
        source = scoped(); parent = source['models']['parent']
        wrapper = {'id': 'wrapper', 'name': 'Wrapper', 'graph': {
            'nodes': [{'id': 'input', 'type': 'Input', 'features': 2},
                      {'id': 'inside', 'type': 'Module', 'ref': 'child'}, {'id': 'output', 'type': 'Output'}],
            'edges': [{'from': 'input', 'to': 'inside'}, {'from': 'inside', 'to': 'output'}]}}
        source['models']['wrapper'] = wrapper
        for n in parent['graph']['nodes']:
            if n['type'] == 'Module': n['ref'] = 'wrapper'
        for s in list(source['symbols']):
            if s['scope'] == 'child':
                new = copy.deepcopy(s); new.update(id=s['id'].replace('child-', 'wrapper-'), scope='wrapper')
                source['symbols'].append(new)
        for c in source['constructors']: c['symbol'] = c['symbol'].replace('child-', 'wrapper-')
        for sid in ('width', 'activation', 'bias'):
            value = {'symbol': 'wrapper-' + sid}
            if sid == 'width': value = {'op': 'add', 'args': [value, 1]}
            source['constructors'].append({'model': 'wrapper', 'node': 'inside', 'symbol': 'child-' + sid, 'value': value})
        r = specialize(source); self.assertEqual(len(r['models']), 5)
        self.assertEqual([i['bindings']['child-width'] for i in r['instances'] if i['template'] == 'child'], [4, 6])
        self.assertEqual(r['contract']['parameters'], 37)

    def test_scope_type_duplicate_targets_and_unknown_fields_are_rejected(self):
        for mutation in ('parent-target', 'not-module', 'unknown-node', 'unknown-symbol', 'cross-scope', 'duplicate',
                         'boolean-integer', 'enum-integer', 'python', 'extra', 'unknown-key'):
            source = scoped(); c = source['constructors'][0]
            if mutation == 'parent-target': c['symbol'] = 'left-width'
            if mutation == 'not-module': c['node'] = 'head'
            if mutation == 'unknown-node': c['node'] = 'absent'
            if mutation == 'unknown-symbol': c['symbol'] = 'absent'
            if mutation == 'cross-scope': c['value'] = {'symbol': 'child-width'}
            if mutation == 'duplicate': source['constructors'].append(copy.deepcopy(c))
            if mutation == 'boolean-integer': c['value'] = True
            if mutation == 'enum-integer': c['value'] = {'symbol': 'child-activation'}
            if mutation == 'python': c['value'] = 'eval(width)'
            if mutation == 'extra': c['exec'] = 'not-code'
            if mutation == 'unknown-key': source['constructor'] = source['constructors']
            with self.subTest(mutation=mutation), self.assertRaises(GraphError): validate_family(source)

    def test_old_grammars_cannot_silently_ignore_scoped_calls(self):
        for version in (1, 2, 3, True, 5, 4.):
            source = scoped(); source['version'] = version
            with self.subTest(version=version), self.assertRaises(GraphError): validate_family(source)
        source = scoped(); source.pop('constructors')
        source['symbols'] = [s for s in source['symbols'] if s['scope'] == 'child']
        self.assertEqual(preview(source)['status'], 'conditional')
        source['constructors'] = None
        with self.assertRaises(GraphError): validate_family(source)

    def test_call_value_bounds_and_division_errors_name_the_call_site(self):
        for value in (17, {'op': 'div', 'args': [3, 2]}, {'op': 'div', 'args': [3, 0]}):
            source = scoped(); source['constructors'][0]['value'] = value
            source['assignments'].append({'model': 'parent', 'node': 'head', 'field': 'features', 'value': {'symbol': 'left-width'}})
            result = preview(source); self.assertEqual(result['status'], 'incompatible')
            self.assertIn('parent/left', result['reason'])

    def test_equal_contexts_reuse_specs_not_parameter_identity(self):
        r = specialize(same_context(scoped())); left, right = modules(r)
        self.assertEqual(left['ref'], right['ref']); self.assertNotIn('share_with', right)
        self.assertEqual(len(r['models']), 2); self.assertEqual(r['contract']['parameters'], 19)
        self.assertEqual(len(r['model_ids']['child']), 1)
        self.assertEqual(len(r['instances']), 2)

    def test_explicit_share_requires_identical_resolved_context(self):
        source = same_context(scoped()); source['models']['parent']['graph']['nodes'][2]['share_with'] = 'left'
        r = specialize(source); self.assertEqual(r['contract']['parameters'], 13)
        source['constructors'][3]['value'] = 4
        result = preview(source); self.assertEqual(result['status'], 'incompatible')
        self.assertIn('share', result['reason'].lower())

    def test_repeated_module_calls_keep_their_template_arguments(self):
        source = same_context(scoped()); parent = source['models']['parent']
        parent['graph']['nodes'] = [n for n in parent['graph']['nodes'] if n['id'] in ('input', 'left', 'output')]
        parent['graph']['edges'] = [{'from': 'input', 'to': 'left'}, {'from': 'left', 'to': 'output'}]
        source['constructors'] = [c for c in source['constructors'] if c['node'] == 'left']
        source['symbols'][0]['default'] = 2
        for mode, params in [('independent', 12), ('tied', 4)]:
            source['repeats'] = [{'id': 'calls', 'model': 'parent', 'nodes': ['left'], 'count': 3, 'mode': mode}]
            r = specialize(source); self.assertEqual(r['contract']['parameters'], params)
            self.assertEqual(len(r['occurrences']), 3)
            self.assertTrue(all(o['model'] == r['root'] and o['template'] == 'parent' for o in r['occurrences']))
            self.assertEqual(len(set(n['ref'] for n in modules(r))), 1)

    def test_variant_budget_is_bounded_before_copying_the_next_model(self):
        source = scoped()
        with patch('app.families.copy.deepcopy', wraps=copy.deepcopy) as copies:
            # The normal bounded closure copies exactly the actual three contexts,
            # not the template closure once per arbitrary parameter occurrence.
            r = specialize(source)
            calls = [c for c in copies.call_args_list if c.args[0] in source['models'].values()]
            self.assertEqual(len(calls), 3); self.assertEqual(len(r['models']), 3)
        source['constructors'] *= 86
        with self.assertRaisesRegex(GraphError, '512'): validate_family(source)

    def test_nested_variant_explosion_stops_before_context_sixty_five_is_copied(self):
        source = scoped(); source['symbols'] = [s for s in source['symbols'] if s['scope'] == 'child']
        source['symbols'][0]['maximum'] = 128
        source['symbols'].append({'id': 'offset', 'scope': 'wrapper', 'name': 'offset', 'minimum': 0})
        source['constructors'] = []
        for mid, child in [('parent', 'wrapper'), ('wrapper', 'child')]:
            nodes = [{'id': 'input', 'type': 'Input', 'features': 2}]
            nodes += [{'id': 'use'+str(i), 'type': 'Module', 'ref': child} for i in range(8)]
            nodes += [{'id': 'join', 'type': 'Concat', 'sources': ['use'+str(i) for i in range(8)], 'flatten': False},
                      {'id': 'output', 'type': 'Output'}]
            edges = [{'from': 'input', 'to': 'use'+str(i)} for i in range(8)]
            edges += [{'from': 'use'+str(i), 'to': 'join'} for i in range(8)] + [{'from': 'join', 'to': 'output'}]
            source['models'][mid] = {'id': mid, 'name': mid, 'graph': {'nodes': nodes, 'edges': edges}}
            for i in range(8):
                pairs = [('offset', i*8)] if mid == 'parent' else [
                    ('child-width', {'op': 'add', 'args': [{'symbol': 'offset'}, i+1]}),
                    ('child-activation', 'Tanh'), ('child-bias', False)]
                source['constructors'] += [{'model': mid, 'node': 'use'+str(i), 'symbol': sid, 'value': value}
                                           for sid, value in pairs]
        # Seed Module shapes must agree before constructor annotations are applied.
        source['models']['parent']['graph']['nodes'][0]['features'] = 2
        with patch('app.families.copy.deepcopy', wraps=copy.deepcopy) as copies:
            with self.assertRaisesRegex(GraphError, '64 distinct constructor contexts'): specialize(source)
            models = list(source['models'].values())
            self.assertEqual(sum(c.args[0] in models for c in copies.call_args_list), 64)

    def test_retained_provenance_reproduces_every_context_and_rejects_forgery(self):
        w = self.workspace(); r = retain(w, family=scoped(), materialize=False)
        record = w.get_spec('specialization', r['specialization']['id']); self.assertEqual(record['version'], 3)
        self.assertEqual(record['instances'], r['resolved']['instances']); validate_all(w.db)
        for mutation in ('omit', 'change', 'template', 'fallback', 'downgrade'):
            forged = copy.deepcopy(record); forged.pop('revision'); forged['id'] = 'forged-' + mutation
            if mutation == 'omit': forged['instances'].pop()
            if mutation == 'change': forged['instances'][0]['bindings']['child-width'] = 4
            if mutation == 'template': forged['instances'][0]['template'] = 'parent'
            if mutation == 'fallback': forged['bindings'].pop('left-width')
            if mutation == 'downgrade': forged['version'] = 2; forged.pop('instances')
            before = w.db.serialize()
            with self.subTest(mutation=mutation), self.assertRaises(GraphError): w.put_spec('specialization', forged)
            self.assertEqual(w.db.serialize(), before)

    def test_materialization_closure_and_atomic_failure(self):
        w = self.workspace(); before = w.db.serialize()
        with patch.object(w, 'put', side_effect=GraphError('injected drawing failure')):
            with self.assertRaisesRegex(GraphError, 'injected'): retain(w, family=scoped())
        self.assertEqual(w.db.serialize(), before)
        r = retain(w, family=scoped()); self.assertEqual(set(w.models()), set(r['resolved']['models']))
        validate_all(w.db)

    def test_owned_optional_cards_explicit_upgrade_and_no_sql_preview(self):
        w = self.workspace(); source = scoped(); source.pop('constructors'); source['version'] = 3
        source['symbols'] = [s for s in source['symbols'] if s['scope'] == 'child']
        ref = w.put_spec('family', source); s = Sessions(); owner = object(); self.addCleanup(s.close_owner, owner)
        h = s.open(w, owner, 'family', base=ref); before = w.db.serialize()
        page = s.family_page(h['session'], owner, 0, 'constructors')
        self.assertEqual(page['total'], 0); self.assertIsNone(page['path']); self.assertEqual(w.db.serialize(), before)
        op = {'op': 'insert', 'collection': 'constructors', 'index': 0, 'value': scoped()['constructors'][0]}
        with self.assertRaisesRegex(GraphError, 'version-4'): s.family_edit(w, h['session'], owner, 0, [op])
        operations = [{'op': 'property', 'target': {'kind': 'document'}, 'key': 'version', 'text': '4'}]
        operations += [{'op': 'insert', 'collection': 'symbols', 'index': 3+i, 'value': symbol}
                       for i, symbol in enumerate(scoped()['symbols'][:2])]
        operations += [dict(op, index=i, value=c) for i, c in enumerate(scoped()['constructors'])]
        h = s.family_edit(w, h['session'], owner, 0, operations)
        result = s.family_preview(w, h['session'], owner, h['sequence'])
        self.assertEqual(result['parameters'], 30); self.assertEqual(result['constructor_contexts'], 3)
        self.assertEqual(w.db.serialize(), before)
        with self.assertRaises(GraphError): s.family_edit(w, h['session'], owner, h['sequence'], [operations[0] | {'text': '3'}])
        saved = s.commit(w, h['session'], owner, h['sequence']); self.assertEqual(w.get_spec('family', saved['id'])['version'], 4)

    def test_version_eleven_gate_and_nonmutating_old_read(self):
        w = self.workspace(); retain(w, family=scoped(), materialize=False)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'scoped.vtw'; w.save(path); original = path.read_bytes(); reopened = Workspace.load(path)
            try: validate_all(reopened.db); self.assertEqual(reopened.db.execute('PRAGMA user_version').fetchone()[0], VERSION)
            finally: reopened.db.close()
            self.assertEqual(path.read_bytes(), original)
            with closing(sqlite3.connect(path)) as db: db.execute('PRAGMA user_version=10'); db.commit()
            downgraded = path.read_bytes()
            with self.assertRaisesRegex(GraphError, 'version-11'): Workspace.load(path)
            self.assertEqual(path.read_bytes(), downgraded)

    def test_agent_atomic_batch_and_identical_specialization(self):
        import vpw_cli
        w = self.workspace(); result = vpw_cli.apply_edits(w, [{'op': 'specialize-family', 'family': scoped(), 'bindings': {}}])[0]
        self.assertEqual(result['resolved'], specialize(scoped())); validate_all(w.db)

    def test_experiment_axes_select_parent_scopes_and_keep_invalid_points(self):
        w, e = scoped_experiment(); self.addCleanup(w.db.close)
        self.addCleanup(lambda: w.history.directory.cleanup() if w.history.directory else None)
        result = preview_experiment(e, w.get_spec); self.assertEqual(len(result['candidates']), 16)
        self.assertEqual({r['status'] for r in result['candidates']}, {'compatible'})
        e.pop('revision'); e['models'][0]['axes'][0]['values'] = [3, 17]
        ref = w.put_spec('experiment', e); e = w.get_spec('experiment', ref['id'], ref['revision'])
        rows = preview_experiment(e, w.get_spec)['candidates']
        self.assertEqual([rows[i]['status'] for i in (0, 8)], ['compatible', 'incompatible'])
        self.assertIn('parent/left', rows[8]['reason'])
        bundle = candidate_bundle(e, rows[0]['coordinates'], w.get_spec)
        self.assertEqual(bundle['model']['models'][bundle['model']['root']]['graph']['input_order'], ['input'])

    def test_fresh_preview_and_export_import_no_numerical_or_qt_runtime(self):
        code = 'import json,sys;from app.families import specialize;from app.core import generate_code;f=json.load(open("examples/families/scoped-branches.json"));r=specialize(f);generate_code(r["models"],r["root"]);assert not any(m in sys.modules for m in ("torch","numpy","PyQt6"))'
        ran = subprocess.run([sys.executable, '-c', code], cwd=ROOT, capture_output=True, text=True, timeout=10)
        self.assertEqual(ran.returncode, 0, ran.stderr)


if __name__ == '__main__': unittest.main()
