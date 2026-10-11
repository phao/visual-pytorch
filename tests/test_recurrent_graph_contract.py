"""Pure multiport graph/reference, immutable revision and .vtw admission tests."""
import copy
import io
import json
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from app.core import GraphError, Workspace, VERSION, default_model, validate_model, check_document, generate_code
from app.evidence import identity, parameter_schema
from app.operators import normalized
from app.revisions import model_document, validate_model_document
from test_recurrent_contract import fixture
import vpw_cli


def graph_fixture(*, cnn=False, steps=4, controls=True):
    models, cell, spec = fixture(cnn=cnn)
    root = default_model('Sequence architecture'); root['id'] = 'sequence-model'
    node = {'id': 'scan', 'type': 'Recurrent', 'ref': cell,
            'sequence_inputs': [{'input': 'frame' if cnn else 'x', 'source': 'sequence'}],
            'states': [{'input': 'h', 'output': 'next', 'port': 'hidden_final', 'source': 'initial'}],
            'sequence_outputs': [{'output': 'next', 'port': 'trajectory'}]}
    nodes = [{'id': 'sequence', 'type': 'Input', 'shape': [steps, 1, 4, 5] if cnn else [steps, 2]},
             {'id': 'initial', 'type': 'Input', 'shape': [2]}, node,
             {'id': 'prediction', 'type': 'Output'}, {'id': 'final', 'type': 'Output'}]
    if cnn: nodes[0]['axes'] = ['T', 'C', 'H', 'W']
    edges = [{'from': 'sequence', 'to': 'scan'}, {'from': 'initial', 'to': 'scan'},
             {'from': 'scan', 'from_port': 'trajectory', 'to': 'prediction'},
             {'from': 'scan', 'from_port': 'hidden_final', 'to': 'final'}]
    if controls:
        node['controls'] = {}
        for role, shape in (('lengths', [1]), ('mask', [steps]), ('reset', [steps])):
            nodes.append({'id': role, 'type': 'Input', 'shape': shape})
            edges.append({'from': role, 'to': 'scan'})
            node['controls'][role] = {'source': role, 'adapter': 'exact-int64-v1' if role == 'lengths' else 'exact-bool-v1'}
    root['graph'] = {'nodes': nodes, 'edges': edges, 'input_order': [n['id'] for n in nodes if n['type'] == 'Input']}
    models[root['id']] = root
    return models, root['id']


class RecurrentGraphContractTests(unittest.TestCase):
    def test_baseline_shapes_axes_ports_packing_and_once_only_parameters(self):
        models, mid = graph_fixture(); c = validate_model(models, mid)
        self.assertEqual(c['parameters'], 10); self.assertFalse(c['affine']); self.assertFalse(c['matrix_affine'])
        self.assertEqual(c['shapes']['scan']['ports'], {'trajectory': [4, 2], 'hidden_final': [2]})
        self.assertEqual(c['shapes']['prediction']['output_axes'], ['T', 'D0'])
        self.assertEqual(c['outputs'], 10); self.assertEqual(c['output_mode'], 'branches')
        self.assertEqual(set(parameter_schema(models, mid)), {'layer_0.transition.layer_0.weight', 'layer_0.transition.layer_0.bias'})
        self.assertEqual(parameter_schema(models, mid)['layer_0.transition.layer_0.weight']['id'], mid + '/scan/mix/weight')

    def test_unroll_work_and_activation_estimate_scale_while_parameters_do_not(self):
        a = validate_model(*graph_fixture(steps=2)); b = validate_model(*graph_fixture(steps=20))
        self.assertEqual(a['parameters'], b['parameters'])
        self.assertGreater(b['operations'], a['operations']); self.assertGreater(b['activation_elements'], a['activation_elements'])

    def test_cnn_transition_and_semantic_time_axis(self):
        models, mid = graph_fixture(cnn=True); c = validate_model(models, mid)
        self.assertEqual(c['parameters'], 14); self.assertGreater(c['convolution_workspace_elements'], 0)
        models[mid]['graph']['nodes'][0]['axes'] = ['C', 'T', 'H', 'W']
        with self.assertRaisesRegex(GraphError, 'leading.*T'): validate_model(models, mid)

    def test_normalized_closed_bindings_and_explicit_control_adapters(self):
        models, mid = graph_fixture(); n = models[mid]['graph']['nodes'][2]
        normalized(n)
        for key, value in (('op_version', True), ('extra', 1), ('ref', None)):
            bad = dict(n, **{key: value})
            with self.subTest(key=key), self.assertRaises(GraphError): normalized(bad)
        for role in n['controls']:
            bad = copy.deepcopy(n); bad['controls'][role]['adapter'] = 'implicit'
            with self.subTest(role=role), self.assertRaisesRegex(GraphError, 'explicit adapter'): normalized(bad)

    def test_duplicate_body_inputs_or_graph_output_port_names_are_rejected(self):
        models, mid = graph_fixture(); node = models[mid]['graph']['nodes'][2]
        node['sequence_inputs'].append(dict(node['sequence_inputs'][0]))
        with self.assertRaises(GraphError): check_document(models[mid])
        models, mid = graph_fixture(); node = models[mid]['graph']['nodes'][2]
        node['sequence_outputs'][0]['port'] = 'hidden_final'
        with self.assertRaisesRegex(GraphError, 'unique'): check_document(models[mid])

    def test_bindings_must_cover_exact_incoming_edges_and_valid_output_ports(self):
        for change in ('missing-edge', 'extra-edge', 'wrong-port', 'unknown-child-input'):
            models, mid = graph_fixture(); g = models[mid]['graph']
            if change == 'missing-edge': g['edges'].pop(0)
            elif change == 'extra-edge':
                g['nodes'].append({'id': 'extra', 'type': 'Input', 'features': 2}); g['input_order'].append('extra')
                g['edges'].append({'from': 'extra', 'to': 'scan'})
            elif change == 'wrong-port': g['edges'][2]['from_port'] = 'missing'
            else: g['nodes'][2]['sequence_inputs'][0]['input'] = 'observation'
            with self.subTest(change=change), self.assertRaises(GraphError): validate_model(models, mid)

    def test_bad_sequence_state_or_control_shapes_are_not_inferred(self):
        for nid, shape in (('sequence', [4, 3]), ('initial', [1]), ('mask', [1, 4]), ('lengths', [4]), ('reset', [3])):
            models, mid = graph_fixture(); next(n for n in models[mid]['graph']['nodes'] if n['id'] == nid)['shape'] = shape
            with self.subTest(nid=nid), self.assertRaises(GraphError): validate_model(models, mid)

    def test_upstream_router_port_is_a_stable_bound_source_not_a_packed_tensor(self):
        models, mid = graph_fixture(); graph = models[mid]['graph']
        graph['nodes'][0]['shape'] = [4, 4]
        graph['nodes'].append({'id': 'split', 'type': 'Splitter', 'parts': [
            {'output': 'frames', 'features': 2}, {'output': 'unused', 'features': 2}]})
        graph['nodes'][2]['sequence_inputs'][0]['source'] = 'split@frames'
        graph['edges'][0].update(**{'from': 'split', 'from_port': 'frames'})
        graph['edges'].append({'from': 'sequence', 'to': 'split'})
        contract = validate_model(models, mid)
        self.assertEqual(contract['shapes']['scan']['ports']['trajectory'], [4, 2])
        plan = next(row for row in contract['plan'] if row['id'] == 'scan')
        self.assertEqual(plan['sources'][0], {'node': 'split', 'port': 'frames'})

    def test_closure_and_cycles_include_recurrent_references(self):
        models, mid = graph_fixture(); doc = model_document(models, mid)
        self.assertEqual(doc['version'], 3); self.assertEqual(set(doc['models']), {mid, 'transition'})
        for version in (1, 2):
            bad = dict(doc, version=version)
            with self.subTest(version=version), self.assertRaisesRegex(GraphError, 'version 3'): validate_model_document(bad)
        models['transition']['graph'] = copy.deepcopy(models[mid]['graph'])
        with self.assertRaisesRegex(GraphError, 'reference cycle'): validate_model(models, mid)

    def test_cell_changes_invalidate_root_identity_while_geometry_and_labels_do_not(self):
        models, mid = graph_fixture(); before = identity(models, mid)['architecture']
        models['transition']['graph']['nodes'][3]['x'] = 900
        self.assertEqual(identity(models, mid)['architecture'], before)
        models['transition']['graph']['nodes'][4]['activation'] = 'ReLU'
        self.assertNotEqual(identity(models, mid)['architecture'], before)

    def test_workspace_indexes_references_freezes_v3_and_saves_container14(self):
        w = Workspace(); self.addCleanup(w.db.close); models, mid = graph_fixture()
        w.put(models['transition']); w.put(models[mid]); frozen = w.freeze_model(mid)
        self.assertEqual(w.get_spec('model', frozen['id'], frozen['revision'])['version'], 3)
        self.assertEqual(w.db.execute('SELECT target_id FROM architecture_refs WHERE owner_id=?', (mid,)).fetchone(), ('transition',))
        with self.assertRaises(GraphError): w.delete('transition')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'sequence.vtw'; w.save(path)
            with sqlite3.connect(path) as db: self.assertEqual(db.execute('PRAGMA user_version').fetchone()[0], VERSION)
            opened = Workspace.load(path); self.addCleanup(opened.db.close)
            self.assertEqual(validate_model(opened.models(), mid)['parameters'], 10)
            with sqlite3.connect(path) as db: db.execute('PRAGMA user_version=13')
            with self.assertRaisesRegex(GraphError, 'version-14'): Workspace.load(path)

    def test_family1_to5_cannot_silently_acquire_recurrent_nodes(self):
        from app.families import validate_family
        models, mid = graph_fixture(); f = {'id': 'f', 'name': 'Recurrent family', 'root': mid,
                                           'models': {k: {field: m[field] for field in ('id', 'name', 'graph')} for k, m in models.items()},
                                           'symbols': [], 'assignments': [], 'repeats': []}
        for version in range(1, 6):
            with self.subTest(version=version), self.assertRaisesRegex(GraphError, 'recurrent family grammar'):
                validate_family(dict(f, version=version))

    def test_container13_legacy_frozen_models_still_open_without_disk_migration(self):
        w = Workspace(); self.addCleanup(w.db.close)
        model = default_model('Legacy'); w.put(model); frozen = w.freeze_model(model['id'])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'legacy.vtw'; w.save(path)
            with sqlite3.connect(path) as db: db.execute('PRAGMA user_version=13')
            opened = Workspace.load(path); self.addCleanup(opened.db.close)
            self.assertEqual(opened.get_spec('model', frozen['id'], frozen['revision'])['version'], 1)
            with sqlite3.connect(path) as db: self.assertEqual(db.execute('PRAGMA user_version').fetchone()[0], 13)

    def test_cli_ordered_create_freeze_export_roundtrip_and_parameter_counts(self):
        models, mid = graph_fixture()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'cli.vtw'; w = Workspace(); w.save(path); w.db.close()
            edits = [{'op': 'create', 'document': models[k]} for k in ('transition', mid)] + [{'op': 'freeze-model', 'id': mid}]
            with redirect_stdout(io.StringIO()), unittest.mock.patch('sys.stdin', io.StringIO(json.dumps(edits))):
                self.assertEqual(vpw_cli.main([str(path), 'apply', '-']), 0)
            stdout = io.StringIO()
            with redirect_stdout(stdout): self.assertEqual(vpw_cli.main([str(path), 'parameter-counts', mid]), 0)
            self.assertEqual(json.loads(stdout.getvalue())['value']['counts']['scan'], '10')
            opened = Workspace.load(path); self.addCleanup(opened.db.close)
            code = generate_code(opened.models(), mid); compile(code, '<standalone-recurrent>', 'exec')

    def test_pure_import_revision_and_source_generation_have_no_torch_or_qt(self):
        code = ('import sys;from test_recurrent_graph_contract import graph_fixture; '
                'from app.revisions import model_document;from app.core import generate_code; '
                'm,i=graph_fixture();model_document(m,i);generate_code(m,i); '
                'assert not any(n in sys.modules for n in ("torch","PyQt6"));print("PASS")')
        result = subprocess.run([sys.executable, '-c', code], text=True, capture_output=True, check=True)
        self.assertEqual(result.stdout.strip(), 'PASS')


if __name__ == '__main__': unittest.main()
