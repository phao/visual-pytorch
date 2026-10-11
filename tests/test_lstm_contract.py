"""Pure proof, closed grammar, state schema and versioned .vtw/CLI checks."""
import copy
import json
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.core import GraphError, Workspace, VERSION, check_document, compile_model, validate_model, generate_code
from app.evidence import identity, parameter_schema
from app.lstm_nodes import normalized, sources
from app.revisions import model_document, validate_model_document
from examples.projected_lstm import models as fixture


class LSTMContractTests(unittest.TestCase):
    def test_projected_bidirectional_contract_and_parameter_shapes(self):
        models, mid = fixture(); c = validate_model(models, mid)
        s = c['shapes']['lstm']['lstm']
        self.assertEqual(s['ports'], {'sequence': [5, 8], 'hidden': [4, 4], 'cell': [4, 7]})
        self.assertEqual(c['parameters'], 1400); self.assertEqual(c['outputs'], 84)
        self.assertFalse(c['affine']); self.assertFalse(c['matrix_affine'])
        self.assertEqual(s['kwargs']['dropout'], 0.); self.assertTrue(s['kwargs']['batch_first'])
        schema = parameter_schema(models, mid)
        self.assertEqual(schema['layer_0.weight_hr_l1_reverse']['shape'], [4, 7])
        self.assertEqual(schema['layer_0.weight_ih_l1']['shape'], [28, 8])
        self.assertEqual(schema['layer_0.weight_hh_l0']['shape'], [28, 4])
        self.assertEqual(schema['layer_0.weight_hr_l0']['id'], mid + '/lstm/weight_hr_l0')

    def test_plain_unidirectional_and_biasless_schema(self):
        models, mid = fixture(projection=0, bidirectional=False, bias=False)
        c = validate_model(models, mid)
        self.assertEqual(c['shapes']['lstm']['ports']['hidden'], [2, 7])
        self.assertEqual(c['parameters'], 672)
        self.assertFalse(any('bias' in key or 'weight_hr' in key for key in parameter_schema(models, mid)))

    def test_work_scales_with_time_not_parameter_count(self):
        a = validate_model(*fixture(steps=2)); b = validate_model(*fixture(steps=25))
        self.assertEqual(a['parameters'], b['parameters'])
        self.assertGreater(b['activation_elements'], a['activation_elements'])
        self.assertGreater(b['operations'], a['operations'])

    def test_normalized_defaults_and_binding_order_not_json_key_order(self):
        models, mid = fixture(); n = models[mid]['graph']['nodes'][3]
        n['inputs'] = {'cell': 'cell', 'hidden': 'hidden', 'sequence': 'sequence'}
        for key in ('bias', 'bidirectional', 'num_layers', 'proj_size'): n.pop(key)
        v = normalized(n)
        self.assertEqual((v['bias'], v['bidirectional'], v['num_layers'], v['proj_size']), (True, False, 1, 0))
        self.assertEqual(sources(n), ['sequence', 'hidden', 'cell', 'lengths'])
        self.assertEqual(normalized(v), v)

    def test_bad_constructor_and_unsupported_fields_fail_without_torch(self):
        for key, value in (('op_version', True), ('op_version', 2), ('hidden_size', 0), ('hidden_size', True),
                           ('num_layers', 0), ('num_layers', 65), ('proj_size', 7), ('proj_size', -1),
                           ('bias', 1), ('bidirectional', 'yes'), ('dropout', .2), ('batch_first', False)):
            models, mid = fixture(); models[mid]['graph']['nodes'][3][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(GraphError): check_document(models[mid])

    def test_missing_states_or_unknown_binding_roles_fail(self):
        for bound in ({'sequence': 'sequence'}, {'sequence': 'sequence', 'hidden': 'hidden', 'cell': ''},
                      {'sequence': 'sequence', 'hidden': 'hidden', 'cell': 'cell', 'reset': 'cell'}, None):
            models, mid = fixture(); models[mid]['graph']['nodes'][3]['inputs'] = bound
            with self.subTest(bound=bound), self.assertRaises(GraphError): validate_model(models, mid)

    def test_state_shapes_are_not_confused_with_native_batch_axis(self):
        for nid, shape in (('sequence', [3, 5, 1]), ('hidden', [4, 7]), ('cell', [4, 4]), ('hidden', [4]), ('lengths', [5])):
            models, mid = fixture(); next(n for n in models[mid]['graph']['nodes'] if n['id'] == nid)['shape'] = shape
            with self.subTest(nid=nid, shape=shape), self.assertRaises(GraphError): validate_model(models, mid)

    def test_axes_and_length_adapter_are_explicit(self):
        for nid, axes in (('sequence', ['F', 'T']), ('hidden', ['Hout', 'LD']), ('cell', ['Hcell', 'LD'])):
            models, mid = fixture(); next(n for n in models[mid]['graph']['nodes'] if n['id'] == nid)['axes'] = axes
            with self.subTest(nid=nid), self.assertRaises(GraphError): validate_model(models, mid)
        for value in ({'source': 'lengths'}, {'source': 'lengths', 'adapter': 'cast'}, {'source': 'lengths', 'adapter': 'exact-int64-v1', 'extra': 1}):
            models, mid = fixture(); models[mid]['graph']['nodes'][3]['lengths'] = value
            with self.subTest(value=value), self.assertRaises(GraphError): validate_model(models, mid)
        with self.assertRaisesRegex(GraphError, r'2\*\*24'): validate_model(*fixture(steps=2**24 + 1))

    def test_incoming_wire_set_is_exact_and_output_port_mandatory(self):
        for change in ('missing', 'duplicate', 'wrong_port', 'no_port'):
            models, mid = fixture(); g = models[mid]['graph']
            if change == 'missing': g['edges'].pop()
            elif change == 'duplicate': g['edges'].append(copy.deepcopy(g['edges'][0]))
            elif change == 'wrong_port': g['edges'][3]['from_port'] = 'unknown'
            else: g['edges'][3].pop('from_port')
            with self.subTest(change=change), self.assertRaises(GraphError): validate_model(models, mid)

    def test_same_source_can_bind_hidden_and_cell_when_shapes_match(self):
        models, mid = fixture(projection=0); g = models[mid]['graph']
        g['nodes'][3]['inputs']['cell'] = 'hidden'
        g['nodes'] = [n for n in g['nodes'] if n['id'] != 'cell']
        g['edges'] = [e for e in g['edges'] if e['from'] != 'cell']
        g['input_order'].remove('cell')
        c = validate_model(models, mid)
        plan = next(p for p in c['plan'] if p['id'] == 'lstm')['lstm']
        self.assertEqual(plan['inputs']['hidden'], plan['inputs']['cell'])

    def test_retained_revision_v4_and_legacy_version_refusal(self):
        models, mid = fixture(); doc = model_document(models, mid)
        self.assertEqual(doc['version'], 4); validate_model_document(doc)
        for version in (1, 2, 3):
            with self.subTest(version=version), self.assertRaisesRegex(GraphError, 'version 4'):
                validate_model_document(dict(doc, version=version))

    def test_identity_ignores_layout_but_not_projection_or_bindings(self):
        models, mid = fixture(); before = identity(models, mid)['architecture']
        n = models[mid]['graph']['nodes'][3]; n.update(x=22, collapsed=True)
        self.assertEqual(identity(models, mid)['architecture'], before)
        n['inputs']['hidden'], n['inputs']['cell'] = n['inputs']['cell'], n['inputs']['hidden']
        self.assertNotEqual(identity(models, mid)['architecture'], before)

    def test_workspace_roundtrip_and_container17_fence(self):
        models, mid = fixture(); w = Workspace(); self.addCleanup(w.db.close)
        w.put(models[mid]); ref = w.freeze_model(mid)
        self.assertEqual(w.get_spec('model', ref['id'], ref['revision'])['version'], 4)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'lstm.vtw'; w.save(path)
            before = path.read_bytes(); opened = Workspace.load(path); self.addCleanup(opened.db.close)
            self.assertEqual(validate_model(opened.models(), mid)['parameters'], 1400)
            self.assertEqual(path.read_bytes(), before)
            with sqlite3.connect(path) as db: self.assertEqual(db.execute('PRAGMA user_version').fetchone()[0], VERSION)
            with sqlite3.connect(path) as db: db.execute('PRAGMA user_version=16')
            downgraded = path.read_bytes()
            with self.assertRaisesRegex(GraphError, 'version-17'): Workspace.load(path)
            self.assertEqual(path.read_bytes(), downgraded)

    def test_old_family_and_runtime_policies_do_not_acquire_new_grammar(self):
        from app.families import validate_family
        from app.temporal_shapes import specification
        models, mid = fixture(); root = models[mid]
        f = {'id': 'f', 'name': 'LSTM family', 'root': mid,
             'models': {mid: {k: root[k] for k in ('id', 'name', 'graph')}}, 'symbols': [], 'assignments': [], 'repeats': []}
        for version in range(1, 7):
            candidate = dict(f, version=version)
            if version >= 5: candidate['runtime_inputs'] = []
            if version == 6: candidate['runtime_policy'] = None
            with self.subTest(version=version), self.assertRaises(GraphError): validate_family(candidate)
        with self.assertRaisesRegex(GraphError, 'do not support LSTM'): specification(models, mid, {'sequence': [0]})

    def test_incompatible_graph_fails_before_constructor_allocation(self):
        models, mid = fixture(); models[mid]['graph']['nodes'][1]['shape'] = [4, 7]
        with patch.dict(sys.modules, {'torch': None}), self.assertRaisesRegex(GraphError, 'initial hidden'):
            compile_model(models, mid, constrained=False)

    def test_example_uses_pure_host_and_cli_validation_and_export(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'lstm.vtw'
            result = subprocess.run(['/usr/bin/python3', 'examples/projected_lstm.py', str(path)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr); self.assertIn('1400', result.stdout)
            again = subprocess.run(['/usr/bin/python3', 'examples/projected_lstm.py', str(path)], capture_output=True, text=True)
            self.assertNotEqual(again.returncode, 0)
            result = subprocess.run(['/usr/bin/python3', 'vpw_cli.py', str(path), 'validate', 'lstm-example'], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            result = subprocess.run(['/usr/bin/python3', 'vpw_cli.py', str(path), 'export', 'lstm-example'], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr); self.assertIn('nn.LSTM', result.stdout)


if __name__ == '__main__': unittest.main()
