"""Pure shared ownership/storage/version/CLI checks; no numerical imports."""
import copy
import json
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from app.core import GraphError, VERSION, Workspace, check_document, validate_model, generate_code
from app.evidence import identity, parameter_schema
from app.families import preview, specialize
from app.family_records import retain
from app.graph import parameter_owners
from app.presentation import block_parameter_counts
from app.revisions import model_document, validate_model_document, validate_all

ROOT = Path(__file__).resolve().parents[1]


def serial(depth=3, shared=True, kind='Linear'):
    nodes = [{'id': 'input', 'type': 'Input', 'features': 2}]
    for i in range(depth):
        node = {'id': 'use' + str(i), 'type': kind}
        if kind == 'Linear': node.update(features=2, bias=True)
        else: node['ref'] = 'inner'
        if shared and i: node['share_with'] = 'use0'
        nodes.append(node)
    nodes.append({'id': 'output', 'type': 'Output'})
    for i, node in enumerate(nodes): node.update(x=60 + i * 220, y=180)
    doc = {'id': 'shared', 'name': 'Shared network', 'graph': {'nodes': nodes,
        'edges': [{'from': a['id'], 'to': b['id']} for a, b in zip(nodes, nodes[1:])]}}
    models = {doc['id']: doc}
    if kind == 'Module':
        inner = serial(2, True)['shared']; inner['id'] = 'inner'; inner['name'] = 'Inner'
        models['inner'] = inner
    return models


def tied_family(depth=3):
    family = json.loads((ROOT / 'examples/families/width-depth-tied.json').read_text())
    family['symbols'][0]['default'] = 3; family['symbols'][1]['default'] = depth
    return family


class SharedParametersTests(unittest.TestCase):
    def test_counts_plans_and_schema_store_each_parameter_once(self):
        models = serial(); contract = validate_model(models, 'shared')
        self.assertEqual(contract['parameters'], 6); self.assertEqual(contract['operations'], 3)
        plans = [p for p in contract['plan'] if p['type'] == 'Linear']
        self.assertEqual([p['layer'] for p in plans], ['layer_0'] * 3)
        schema = parameter_schema(models, 'shared')
        self.assertEqual(set(schema), {'layer_0.weight', 'layer_0.bias'})
        self.assertEqual(schema['layer_0.weight']['id'], 'shared/use0/weight')
        counts = block_parameter_counts(models['shared'], contract, {})
        self.assertEqual(counts, {'input': '0', 'use0': '6', 'use1': '0', 'use2': '0', 'output': '0'})
        self.assertGreater(contract['activation_elements'], validate_model(serial(1), 'shared')['activation_elements'])

    def test_no_field_means_independent_not_equal_initial_values(self):
        models = serial(shared=False); contract = validate_model(models, 'shared')
        self.assertEqual(contract['parameters'], 18); self.assertEqual(len(parameter_schema(models, 'shared')), 6)
        self.assertNotEqual(identity(models, 'shared')['architecture'], identity(serial(), 'shared')['architecture'])

    def test_agent_node_edits_and_freeze_preserve_explicit_sharing(self):
        import vpw_cli
        w = Workspace(); self.addCleanup(w.db.close); w.put(serial(shared=False)['shared'])
        result = vpw_cli.apply_edits(w, [{'op': 'set-node', 'id': 'shared', 'node': 'use1', 'values': {'share_with': 'use0'}},
            {'op': 'set-node', 'id': 'shared', 'node': 'use2', 'values': {'share_with': 'use1'}}, {'op': 'freeze-model', 'id': 'shared'}])
        ref = result[-1]['reference']; frozen = w.get_spec('model', ref['id'], ref['revision'])
        self.assertEqual(frozen['version'], 2); self.assertEqual(validate_model(frozen['models'], frozen['root'])['parameters'], 6)
        vpw_cli.apply_edits(w, [{'op': 'set-node', 'id': 'shared', 'node': 'use1', 'values': {}, 'remove_fields': ['share_with']}])
        self.assertEqual(validate_model(w.models(), 'shared')['parameters'], 12)
        self.assertEqual(w.get_spec('model', ref['id'], ref['revision']), frozen)

    def test_chains_and_owner_after_use_resolve_without_wire_dependencies(self):
        models = serial(); nodes = models['shared']['graph']['nodes']
        nodes[1]['share_with'] = 'use2'; nodes[2]['share_with'] = 'use0'; del nodes[3]['share_with']
        owners = parameter_owners(models['shared'])
        self.assertEqual([owners['use' + str(i)] for i in range(3)], ['use2'] * 3)
        c = validate_model(models, 'shared'); self.assertEqual(c['parameters'], 6)
        self.assertEqual(set(parameter_schema(models, 'shared')), {'layer_2.weight', 'layer_2.bias'})
        self.assertEqual([p['layer'] for p in c['plan'] if p['type'] == 'Linear'], ['layer_2'] * 3)

    def test_bad_owner_self_cycles_missing_types_and_unsupported_blocks_fail(self):
        for target in ('use1', 'missing', 'input', '', True, 2, None):
            models = serial(); models['shared']['graph']['nodes'][2]['share_with'] = target
            with self.subTest(target=target), self.assertRaises(GraphError): check_document(models['shared'])
        models = serial(); models['shared']['graph']['nodes'][1]['share_with'] = 'use1'
        with self.assertRaisesRegex(GraphError, 'cycle'): validate_model(models, 'shared')
        models = serial(); models['shared']['graph']['nodes'][0]['share_with'] = 'use0'
        with self.assertRaises(GraphError): check_document(models['shared'])

    def test_shape_bias_and_module_reference_disagreement_fail(self):
        for field, value in [('features', 3), ('bias', False)]:
            models = serial(); models['shared']['graph']['nodes'][2][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(GraphError, 'same constructor'): validate_model(models, 'shared')
        models = serial(kind='Module'); other = copy.deepcopy(models['inner']); other['id'] = 'other'; models['other'] = other
        models['shared']['graph']['nodes'][2]['ref'] = 'other'
        with self.assertRaisesRegex(GraphError, 'same constructor/reference'): validate_model(models, 'shared')

    def test_nested_module_counts_and_canonical_stable_paths(self):
        models = serial(kind='Module'); c = validate_model(models, 'shared')
        self.assertEqual(c['parameters'], 6); self.assertEqual(c['operations'], 6)
        schema = parameter_schema(models, 'shared')
        self.assertEqual(set(schema), {'layer_0.layer_0.weight', 'layer_0.layer_0.bias'})
        self.assertEqual(schema['layer_0.layer_0.weight']['id'], 'shared/use0/use0/weight')
        counts = block_parameter_counts(models['shared'], c, {n['id']: validate_model(models, n['ref']) for n in models['shared']['graph']['nodes'] if n['type'] == 'Module'})
        self.assertEqual(counts['use0'], '6'); self.assertEqual(counts['use1'], '0')

    def test_model_v2_preserves_identity_and_v1_cannot_ignore_sharing(self):
        models = serial(); frozen = model_document(models, 'shared'); self.assertEqual(frozen['version'], 2)
        validate_model_document(frozen)
        bad = copy.deepcopy(frozen); bad['version'] = 1
        with self.assertRaisesRegex(GraphError, 'version 2'): validate_model_document(bad)
        plain = model_document(serial(shared=False), 'shared'); self.assertEqual(plain['version'], 1)
        w = Workspace(); self.addCleanup(w.db.close)
        ref = w.put_spec('model', frozen); self.assertEqual(w.get_spec('model', ref['id'], ref['revision'])['version'], 2); validate_all(w.db)

    def test_model_structure_is_validated_before_scanning_sharing_fields(self):
        frozen = model_document(serial(shared=False), 'shared')
        for content in (None, [], {}, {'id': 'shared', 'name': 'Shared network', 'graph': {}}):
            bad = copy.deepcopy(frozen); bad['models']['shared'] = content
            with self.subTest(content=content), self.assertRaises(GraphError): validate_model_document(bad)

    def test_tied_family_distinguishes_uses_from_parameters(self):
        for depth in (1, 2, 4):
            result = specialize(tied_family(depth)); contract = result['contract']
            self.assertEqual(contract['parameters'], 25)  # 2→3 + one3→3 +3→1.
            self.assertEqual(contract['operations'], 2 + 2 * depth)
            self.assertTrue(all(o['mode'] == 'tied' for o in result['occurrences']))
        source = tied_family(); source['version'] = 1
        with self.assertRaisesRegex(GraphError, 'version 2'): specialize(source)
        source = tied_family(); source['repeats'][0]['mode'] = 'recurrent'
        with self.assertRaisesRegex(GraphError, 'not copies'): specialize(source)

    def test_sharing_template_links_are_remapped_inside_and_after_repeats(self):
        models = serial(); model = models['shared']
        family = {'version': 2, 'id': 'repeat-links', 'name': 'Links', 'root': 'shared', 'models': models,
            'symbols': [], 'assignments': [], 'repeats': [{'id': 'r', 'model': 'shared', 'nodes': ['use0', 'use1'], 'count': 2, 'mode': 'independent'}]}
        result = specialize(family); root = result['models'][result['root']]
        byid = {n['id']: n for n in root['graph']['nodes']}
        self.assertEqual(byid['r/0/use1']['share_with'], 'r/0/use0')
        self.assertEqual(byid['r/1/use1']['share_with'], 'r/1/use0')
        self.assertEqual(byid['use2']['share_with'], 'r/0/use0')
        self.assertEqual(result['contract']['parameters'], 12)
        family['repeats'][0]['mode'] = 'tied'; self.assertEqual(specialize(family)['contract']['parameters'], 6)

    def test_retention_roundtrip_and_old_header_rejects_semantics_without_rewrite(self):
        w = Workspace(); self.addCleanup(w.db.close); saved = retain(w, family=tied_family(), materialize=True)
        self.assertEqual(w.get_spec('model', saved['model']['id'])['version'], 2); validate_all(w.db)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'shared.vtw'; w.save(path)
            reopened = Workspace.load(path); self.addCleanup(reopened.db.close)
            self.assertEqual(reopened.get_spec('specialization', saved['specialization']['id']), w.get_spec('specialization', saved['specialization']['id']))
            self.assertEqual(reopened.db.execute('PRAGMA user_version').fetchone()[0], VERSION)
            with closing(sqlite3.connect(path)) as db: db.execute('PRAGMA user_version=8'); db.commit()
            before = path.read_bytes()
            with self.assertRaisesRegex(GraphError, 'version-9'): Workspace.load(path)
            self.assertEqual(path.read_bytes(), before)

    def test_v8_independent_container_migrates_in_memory_with_identical_hashes(self):
        w = Workspace(); self.addCleanup(w.db.close); ref = w.put_spec('model', model_document(serial(shared=False), 'shared'))
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'v8.vtw'; w.save(path)
            with closing(sqlite3.connect(path)) as db: db.execute('PRAGMA user_version=8'); db.commit()
            before = path.read_bytes(); reopened = Workspace.load(path); self.addCleanup(reopened.db.close)
            self.assertEqual(reopened.get_spec('model', ref['id'], ref['revision']), w.get_spec('model', ref['id'], ref['revision']))
            self.assertEqual(path.read_bytes(), before)

    def test_drawing_seed_and_owned_upgrade_preserve_semantic_version(self):
        from app import authoring
        from app.authoring_sessions import Sessions
        w = Workspace(); self.addCleanup(w.db.close); w.put(serial()['shared'])
        draft = authoring.begin(w, 'family', sid='seeded', model='shared')
        self.assertEqual(draft['document']['version'], 2)
        owner = object(); store = Sessions(); self.addCleanup(store.close_owner, owner)
        ref = w.put_spec('family', tied_family()); h = store.open(w, owner, 'family', base=ref)
        h = store.family_edit(w, h['session'], owner, h['sequence'], [{'op': 'property', 'target': {'kind': 'document'}, 'key': 'name', 'text': '"Renamed"'}])
        with self.assertRaises(GraphError): store.family_edit(w, h['session'], owner, h['sequence'], [{'op': 'property', 'target': {'kind': 'document'}, 'key': 'version', 'text': '1'}])

    def test_v1_owned_family_upgrade_is_atomic_and_mismatch_stays_visible(self):
        from app.authoring_sessions import Sessions
        from app import authoring
        w = Workspace(); self.addCleanup(w.db.close); family = tied_family(); family['version'] = 1; family['repeats'][0]['mode'] = 'independent'
        ref = w.put_spec('family', family); owner = object(); store = Sessions(); self.addCleanup(store.close_owner, owner)
        h = store.open(w, owner, 'family', base=ref); before = w.db.serialize()
        h = store.family_edit(w, h['session'], owner, h['sequence'], [
            {'op': 'property', 'target': {'kind': 'document'}, 'key': 'version', 'text': '2'},
            {'op': 'property', 'target': {'kind': 'entry', 'collection': 'repeats', 'ordinal': 0}, 'key': 'mode', 'text': '"tied"'}])
        projected = store.family_preview(w, h['session'], owner, h['sequence'])
        self.assertEqual(projected['parameters_exact'], '25'); self.assertEqual(w.db.serialize(), before)
        stored = store.commit(w, h['session'], owner, h['sequence']); self.assertEqual(w.get_spec('family', stored['id'])['version'], 2)
        # A non-square repeated Linear cannot become shared by silently changing
        # its constructor to fit later occurrences.
        model = serial(1, False)['shared']; model['graph']['nodes'][1]['features'] = 3
        wrong = {'version': 2, 'id': 'bad-repeat', 'name': 'Mismatch', 'root': 'shared', 'models': {'shared': model},
            'symbols': [], 'assignments': [], 'repeats': [{'id': 'r', 'model': 'shared', 'nodes': ['use0'], 'count': 2, 'mode': 'tied'}]}
        result = preview(wrong); self.assertEqual(result['status'], 'incompatible'); self.assertIn('same constructor', result['reason'])

    def test_pure_source_generation_and_resolution_import_no_numerical_libraries(self):
        code = 'from tests.test_shared_parameters import serial,tied_family; from app.core import generate_code; from app.families import specialize; import sys; generate_code(serial(),"shared"); specialize(tied_family()); assert not any(x in sys.modules for x in ("torch","numpy","PyQt6"))'
        # tests is not a package; explicit sys.path keeps this fixture self-contained.
        code = 'import sys; sys.path.insert(0,"tests");' + code.replace('from tests.test_shared_parameters', 'from test_shared_parameters')
        result = subprocess.run([sys.executable, '-c', code], cwd=ROOT, text=True, capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__': unittest.main()
