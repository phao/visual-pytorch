"""Family resolution: independent tensor/gradient oracles and atomic agent edits."""
import copy
import io
import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import torch

from app.core import GraphError, MAX_INTEGER, Workspace, compile_model, default_cnn, default_model, generate_code
from app.evidence import digest, parameter_schema
from app.families import evaluate, preview, specialize, validate_family
from app.revisions import model_document
import vpw_cli

torch.set_num_threads(1)
ROOT = Path(__file__).resolve().parents[1]


def mlp():
    return json.loads((ROOT / 'examples/families/width-depth.json').read_text())


def concrete(doc):
    model = {k: copy.deepcopy(doc[k]) for k in ('id', 'name', 'graph')}
    return {'version': 1, 'id': 'family', 'name': 'Concrete family', 'root': doc['id'],
            'models': {doc['id']: model}, 'symbols': [], 'assignments': [], 'repeats': []}


def cnn():
    doc = default_cnn()
    doc['id'] = 'cnn'
    lookup = {n['id']: n for n in doc['graph']['nodes']}
    lookup['input']['shape'] = [4, 5, 6]
    lookup['conv'].update(out_channels=8, kernel_size=[3, 3], padding=[1, 1], groups=2)
    lookup['head']['features'] = 3
    family = concrete(doc)
    family['symbols'] = [{'id': s, 'scope': 'cnn', 'name': s, 'default': v}
                         for s, v in [('C', 4), ('H', 5), ('W', 6), ('d', 8), ('K', 3), ('g', 2)]]
    family['assignments'] = [
        {'model': 'cnn', 'node': 'input', 'field': 'shape', 'value': [{'symbol': s} for s in ('C', 'H', 'W')]},
        {'model': 'cnn', 'node': 'conv', 'field': 'out_channels', 'value': {'symbol': 'd'}},
        {'model': 'cnn', 'node': 'conv', 'field': 'groups', 'value': {'symbol': 'g'}},
        {'model': 'cnn', 'node': 'head', 'field': 'features', 'value': {'symbol': 'K'}},
    ]
    return family


class FamilyTests(unittest.TestCase):
    def test_integer_only_identity_and_boundary_order_survive(self):
        doc = default_model(); family = concrete(doc)
        before = copy.deepcopy(family)
        resolved = specialize(family)
        self.assertEqual(family, before)
        graph = resolved['models'][resolved['root']]['graph']
        self.assertEqual(graph['input_order'], ['input'])
        self.assertEqual(graph['nodes'], doc['graph']['nodes'])
        self.assertEqual(resolved['contract']['parameters'], 3)

    def test_defaults_bindings_hashes_and_names_do_not_alias(self):
        family = mlp(); one = specialize(family); again = specialize(copy.deepcopy(family))
        self.assertEqual(one, again)
        self.assertEqual(one['contract']['parameters'], 37)
        two = specialize(family, {'hidden-width': 4, 'hidden-depth': 3})
        self.assertEqual(two['contract']['parameters'], 77)
        self.assertNotEqual(one['root'], two['root'])
        family['symbols'][0]['name'] = 'renamed width'
        renamed = specialize(family, {'hidden-width': 4})
        self.assertEqual(renamed['bindings']['hidden-width'], 4)
        self.assertEqual(renamed['family']['revision'], digest(family))

    def test_unresolved_draft_and_bad_binding_are_not_executable(self):
        family = mlp(); del family['symbols'][0]['default']
        result = preview(family)
        self.assertEqual(result['status'], 'conditional')
        self.assertEqual(result['missing'], [{'id': 'hidden-width', 'scope': 'mlp', 'name': 'width'}])
        with self.assertRaisesRegex(GraphError, 'Unresolved symbol hidden-width'):
            specialize(family)
        self.assertEqual(preview(family, {'hidden-width': 0})['status'], 'incompatible')
        for supplied in ({'hidden-width': True}, {'unknown': 3}, [], {'hidden-width': 1.5}):
            with self.subTest(supplied=supplied), self.assertRaises(GraphError):
                preview(family, supplied)

    def test_bounds_and_unused_symbols_are_explicit(self):
        family = mlp(); family['symbols'][0].update(minimum=2, maximum=5)
        self.assertEqual(preview(family, {'hidden-width': 5})['status'], 'verified')
        with self.assertRaisesRegex(GraphError, 'outside declared bounds'):
            specialize(family, {'hidden-width': 6})
        family['symbols'].append({'id': 'unused', 'scope': 'mlp', 'name': 'unused'})
        with self.assertRaisesRegex(GraphError, 'dependent'):
            validate_family(family)

    def test_typed_arithmetic_exactness_and_limits(self):
        family = mlp()
        for op, args, wanted in [('add', [2, 3, 4], 9), ('mul', [2, 3], 6),
                                 ('sub', [8, 3], 5), ('div', [8, 2], 4), ('floordiv', [7, 2], 3)]:
            expr = {'op': op, 'args': args}
            family['assignments'][0]['value'] = expr
            validate_family(family)
            self.assertEqual(evaluate(expr, {}, 'probe'), wanted)
        for expr in [{'op': 'div', 'args': [7, 2]}, {'op': 'floordiv', 'args': [7, 0]},
                     {'op': 'mul', 'args': [MAX_INTEGER, MAX_INTEGER]},
                     {'op': 'add', 'args': [MAX_INTEGER, 1, -1]}]:
            family['assignments'][0]['value'] = expr
            self.assertEqual(preview(family)['status'], 'incompatible')

    def test_no_python_text_calls_extra_fields_or_deep_expression(self):
        for value in ['__import__("os").system("touch forbidden")', True, 3.2,
                      {'op': 'pow', 'args': [2, 3]}, {'symbol': 'hidden-width', 'exec': 'bad'},
                      {'op': 'add', 'args': [1]}, {'symbol': 'absent'}]:
            family = mlp(); family['assignments'][0]['value'] = value
            with self.subTest(value=value), self.assertRaises(GraphError):
                validate_family(family)
        value = 1
        for _ in range(30):
            value = {'op': 'add', 'args': [value, 1]}
        family = mlp(); family['assignments'][0]['value'] = value
        with self.assertRaisesRegex(GraphError, 'depth/node'):
            validate_family(family)
        family = mlp(); family['models']['mlp']['graph']['nodes'][1]['parameter_scope'] = 'secret-tied'
        with self.assertRaisesRegex(GraphError, 'unsupported family template'):
            validate_family(family)

    def test_duplicate_assignments_and_noninteger_fields_refused(self):
        family = mlp(); family['assignments'].append(copy.deepcopy(family['assignments'][0]))
        with self.assertRaisesRegex(GraphError, 'duplicate assignment'):
            validate_family(family)
        for field in ['id', 'bias', 'label', 'x', 'ref', 'unknown']:
            family = mlp(); family['assignments'][0]['field'] = field
            with self.subTest(field=field), self.assertRaisesRegex(GraphError, 'supported integer field'):
                validate_family(family)

    def test_ignored_fields_and_unknown_operator_versions_are_not_family_semantics(self):
        family = mlp(); family['models']['mlp']['graph']['nodes'][0]['shape'] = [2]
        family['assignments'][0].update(node='input')
        with self.assertRaisesRegex(GraphError, 'ignored features'):
            validate_family(family)
        family = mlp(); family['models']['mlp']['graph']['nodes'][3]['dim'] = 0
        with self.assertRaisesRegex(GraphError, 'only to Softmax'):
            validate_family(family)
        family = mlp(); family['models']['mlp']['graph']['nodes'][1]['op_version'] = 2
        with self.assertRaisesRegex(GraphError, 'operator version'):
            validate_family(family)

    def test_softmax_sample_axis_is_a_checked_family_assignment(self):
        family = mlp(); activation = family['models']['mlp']['graph']['nodes'][3]
        activation.update(activation='Softmax', dim=-1)
        family['symbols'].append({'id': 'normalization-axis', 'scope': 'mlp', 'name': 'axis', 'default': -1,
                                  'minimum': -8, 'maximum': 7})
        family['assignments'].append({'model': 'mlp', 'node': 'activation', 'field': 'dim',
                                      'value': {'symbol': 'normalization-axis'}})
        rr = specialize(family)
        self.assertIn('nn.Softmax(dim=1)', generate_code(rr['models'], rr['root']))
        invalid = preview(family, {'normalization-axis': 1})
        self.assertEqual(invalid['status'], 'incompatible'); self.assertIn('axis', invalid['reason'])

    def test_independent_repeat_paths_stable_and_graph_bounded(self):
        one = specialize(mlp(), {'hidden-depth': 2})
        two = specialize(mlp(), {'hidden-depth': 3})
        ids = lambda r: [n['id'] for n in r['models'][r['root']]['graph']['nodes']]
        self.assertTrue(set(ids(one)) <= set(ids(two)))
        self.assertIn('stack/0/hidden', ids(one)); self.assertIn('stack/1/activation', ids(one))
        self.assertEqual(len(one['occurrences']), 4)
        family = mlp()
        for count, fragment in [(0, 'count must be positive'), (MAX_INTEGER, '64-block')]:
            result = preview(family, {'hidden-depth': count}) if count else None
            if count:
                self.assertEqual(result['status'], 'incompatible'); self.assertIn(fragment, result['reason'])
            else:
                family['symbols'][1]['minimum'] = 0
                result = preview(family, {'hidden-depth': 0})
                self.assertIn(fragment, result['reason'])

    def test_tied_and_recurrent_requests_never_become_independent(self):
        for mode in ['tied', 'recurrent', 'shared', None]:
            family = mlp(); family['repeats'][0]['mode'] = mode
            with self.subTest(mode=mode), self.assertRaisesRegex(GraphError, 'not copies'):
                specialize(family)

    def test_repeat_overlap_collision_and_branch_crossings_rejected(self):
        family = mlp(); family['repeats'].append(dict(family['repeats'][0], id='other'))
        with self.assertRaisesRegex(GraphError, 'overlap'):
            specialize(family)
        family = mlp(); family['models']['mlp']['graph']['nodes'][1]['id'] = 'stack/0/hidden'
        for e in family['models']['mlp']['graph']['edges']:
            for k in ('from', 'to'):
                if e[k] == 'stem': e[k] = 'stack/0/hidden'
        family['assignments'][0]['node'] = 'stack/0/hidden'
        self.assertIn('collides', preview(family)['reason'])
        family = mlp(); family['repeats'][0]['nodes'] = ['input']
        with self.assertRaisesRegex(GraphError, 'not graph boundaries'):
            validate_family(family)

    def test_tensor_and_parameter_gradients_against_manual_formula_and_export(self):
        resolved = specialize(mlp(), {'hidden-width': 4, 'hidden-depth': 3})
        native = compile_model(resolved['models'], resolved['root'], constrained=False).double()
        scope = {}; exec(generate_code(resolved['models'], resolved['root']), scope)
        exported = scope['model'].double(); exported.load_state_dict(native.state_dict())
        self.assertEqual(sum(p.numel() for p in native.parameters()), 77)
        self.assertIsNot(native.layer_1.weight, native.layer_3.weight)
        self.assertIsNot(native.layer_3.weight, native.layer_5.weight)
        # Independent functional oracle uses copied leaf tensors, not graph plan/executor.
        leaves = {k: v.detach().clone().requires_grad_() for k, v in native.named_parameters()}
        for batch in (1, 5):
            x = torch.linspace(-.7, .9, batch * 2, dtype=torch.float64).reshape(batch, 2).requires_grad_()
            xx = x.detach().clone().requires_grad_()
            yy = xx @ leaves['layer_0.weight'].T + leaves['layer_0.bias']
            for layer in (1, 3, 5):
                yy = torch.tanh(yy @ leaves[f'layer_{layer}.weight'].T + leaves[f'layer_{layer}.bias'])
            yy = yy @ leaves['layer_7.weight'].T + leaves['layer_7.bias']
            actual = native(x)
            torch.testing.assert_close(actual, yy, rtol=1e-12, atol=1e-12)
            torch.testing.assert_close(exported(x), yy, rtol=1e-12, atol=1e-12)
            expected_grad = torch.autograd.grad(yy.square().sum(), [xx, *leaves.values()])
            actual_grad = torch.autograd.grad(actual.square().sum(), [x, *native.parameters()])
            for a, b in zip(actual_grad, expected_grad):
                torch.testing.assert_close(a, b, rtol=1e-11, atol=1e-11)

    def test_grouped_cnn_179_parameters_and_invalid_binding_origin(self):
        family = cnn(); resolved = specialize(family)
        self.assertEqual(resolved['contract']['parameters'], 179)
        native = compile_model(resolved['models'], resolved['root'], constrained=False)
        self.assertEqual(sum(p.numel() for p in native.parameters()), 179)
        for h, w in ((5, 6), (9, 7)):
            rr = specialize(family, {'H': h, 'W': w})
            net = compile_model(rr['models'], rr['root'], constrained=False)
            net.load_state_dict(native.state_dict())
            x = torch.randn(2, 4, h, w)
            expected = torch.nn.functional.linear(torch.tanh(torch.nn.functional.conv2d(
                x, native.layer_0.weight, native.layer_0.bias, padding=1, groups=2)).mean((2, 3)),
                native.layer_4.weight, native.layer_4.bias)
            torch.testing.assert_close(net(x), expected)
        invalid = preview(family, {'g': 3})
        self.assertEqual(invalid['status'], 'incompatible')
        self.assertIn('conv.groups', invalid['reason']); self.assertIn('divisible', invalid['reason'])

    def test_nested_equal_symbol_labels_keep_distinct_scopes(self):
        child = default_model('Child', 2, 2); child['id'] = 'child'
        parent = default_model('Parent', 2, 1); parent['id'] = 'parent'
        parent['graph']['nodes'].insert(1, {'id': 'module', 'type': 'Module', 'ref': 'child'})
        parent['graph']['edges'][0:1] = [{'from': 'input', 'to': 'module'}, {'from': 'module', 'to': 'linear1'}]
        family = concrete(parent); family['models']['child'] = concrete(child)['models']['child']
        family['symbols'] = [{'id': sid, 'scope': scope, 'name': 'width', 'default': val}
                             for sid, scope, val in [('local', 'parent', 3), ('nested', 'child', 4)]]
        family['assignments'] = [{'model': scope, 'node': 'linear1', 'field': 'features', 'value': {'symbol': sid}}
                                 for sid, scope in [('local', 'parent'), ('nested', 'child')]]
        rr = specialize(family)
        self.assertEqual(rr['contract']['parameters'], 27)  # child 12 + parent 15
        self.assertNotEqual(rr['model_ids']['parent'], rr['model_ids']['child'])
        self.assertEqual(len(parameter_schema(rr['models'], rr['root'])), 4)
        workspace = Workspace()
        try:
            # Parent appears first in JSON, but SQLite refs require children first.
            vpw_cli.apply_edits(workspace, [{'op': 'specialize-family', 'family': family}])
            self.assertEqual(len(workspace.models()), 2)
            workspace.freeze_model(rr['root'])
        finally:
            workspace.db.close()
        family['assignments'][0]['value'] = {'symbol': 'nested'}
        with self.assertRaisesRegex(GraphError, 'outside this model scope'):
            specialize(family)

    def test_repeat_egress_to_concat_preserves_declared_source_order(self):
        family = mlp(); graph = family['models']['mlp']['graph']
        graph['nodes'].insert(1, {'id': 'other', 'type': 'Input', 'features': 2})
        graph['nodes'].insert(-2, {'id': 'join', 'type': 'Concat', 'sources': ['activation', 'other']})
        graph['edges'].remove({'from': 'activation', 'to': 'head'})
        graph['edges'].extend([{'from': 'activation', 'to': 'join'}, {'from': 'other', 'to': 'join'},
                               {'from': 'join', 'to': 'head'}])
        rr = specialize(family)
        join = next(n for n in rr['models'][rr['root']]['graph']['nodes'] if n['id'] == 'join')
        self.assertEqual(join['sources'], ['stack/1/activation', 'other'])
        self.assertEqual(rr['contract']['input_groups'][0]['id'], 'input')
        self.assertEqual(rr['contract']['input_groups'][1]['id'], 'other')
        graph['nodes'].append({'id': 'extra', 'type': 'Output'})
        graph['edges'].append({'from': 'hidden', 'to': 'extra'})
        with self.assertRaisesRegex(GraphError, 'no branch crossings'):
            validate_family(family)

    def test_repeat_router_ingress_keeps_explicit_source_port(self):
        family = mlp(); graph = family['models']['mlp']['graph']
        graph['nodes'].insert(2, {'id': 'copies', 'type': 'Replicator', 'outputs': ['a', 'b']})
        graph['nodes'].append({'id': 'extra', 'type': 'Output'})
        graph['edges'].remove({'from': 'stem', 'to': 'hidden'})
        graph['edges'].extend([{'from': 'stem', 'to': 'copies'},
                               {'from': 'copies', 'from_port': 'a', 'to': 'hidden'},
                               {'from': 'copies', 'from_port': 'b', 'to': 'extra'}])
        rr = specialize(family); edges = rr['models'][rr['root']]['graph']['edges']
        self.assertIn({'from': 'copies', 'from_port': 'a', 'to': 'stack/0/hidden'}, edges)
        self.assertEqual(rr['contract']['output_mode'], 'branches')

    def test_adjacent_repeats_keep_stable_connections_and_independent_parameters(self):
        family = mlp(); family['repeats'].append({'id': 'head-stack', 'model': 'mlp',
            'nodes': ['head'], 'count': 2, 'mode': 'independent'})
        rr = specialize(family); graph = rr['models'][rr['root']]['graph']
        self.assertIn({'from': 'stack/1/activation', 'to': 'head-stack/0/head'}, graph['edges'])
        self.assertIn({'from': 'head-stack/0/head', 'to': 'head-stack/1/head'}, graph['edges'])
        self.assertEqual(rr['contract']['parameters'], 39)
        self.assertEqual(len({(n['x'], n['y']) for n in graph['nodes']}), len(graph['nodes']))

    def test_derived_reshape_and_output_constraint_failure(self):
        doc = default_model(inputs=6, outputs=2); doc['id'] = 'shaped'
        doc['graph']['nodes'].insert(1, {'id': 'reshape', 'type': 'Reshape', 'shape': [2, 3]})
        doc['graph']['nodes'][-1]['shape'] = [2, 2]
        doc['graph']['edges'][0:1] = [{'from': 'input', 'to': 'reshape'}, {'from': 'reshape', 'to': 'linear1'}]
        family = concrete(doc)
        family['symbols'] = [{'id': 'channels', 'scope': 'shaped', 'name': 'C', 'default': 2}]
        family['assignments'] = [{'model': 'shaped', 'node': 'reshape', 'field': 'shape',
            'value': [{'symbol': 'channels'}, {'op': 'div', 'args': [6, {'symbol': 'channels'}]}]}]
        self.assertEqual(specialize(family)['contract']['output_shape'], [2, 2])
        bad = preview(family, {'channels': 4})
        self.assertEqual(bad['status'], 'incompatible'); self.assertIn('reshape.shape', bad['reason'])
        bad = preview(family, {'channels': 3})
        self.assertEqual(bad['status'], 'incompatible'); self.assertIn('Output output declares', bad['reason'])

    def test_model_closure_extra_models_cycles_and_format_version_rejected(self):
        family = mlp(); family['models']['unused'] = concrete(default_model())['models'].popitem()[1]
        family['models']['unused']['id'] = 'unused'
        with self.assertRaisesRegex(GraphError, 'unrelated'):
            validate_family(family)
        for version in [True, 3, 1.0]:
            family = mlp(); family['version'] = version
            with self.subTest(version=version), self.assertRaises(GraphError):
                validate_family(family)

    def test_resolved_concrete_spec_uses_existing_training_and_checkpoint_pipeline(self):
        from app.core import generate_dataset, train_model
        from app.evidence import check_checkpoint, binding_for
        family = concrete(default_model())
        rr = specialize(family); doc = rr['models'][rr['root']]
        doc['training'] = {'epochs': 140, 'lr': .05, 'seed': 4}
        doc['dataset'] = generate_dataset(rr['models'], rr['root'], samples=40, seed=8)
        doc['dataset_binding'] = binding_for(rr['contract'])
        result = train_model(rr['models'], rr['root'], constrained=False)
        self.assertLess(result['mse'], .001)
        doc['result'] = result
        check_checkpoint(rr['models'], rr['root'], result)

    def test_export_runs_in_fresh_process_without_app_imports(self):
        rr = specialize(mlp())
        source = generate_code(rr['models'], rr['root'])
        result = subprocess.run([sys.executable, '-c', source +
            '\nassert model(torch.ones(4, 2)).shape == (4,1)\n'
            'assert sum(p.numel() for p in model.parameters()) == 37\n'],
            cwd=ROOT, capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def cli(self, *args):
        stream = io.StringIO()
        with redirect_stdout(stream):
            code = vpw_cli.main([str(a) for a in args])
        return code, json.loads(stream.getvalue())

    def test_cli_preview_is_readonly_and_import_is_atomic_freezable(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'family.vtw'; workspace = Workspace(); workspace.save(path); workspace.db.close()
            before = path.read_bytes()
            with patch('vpw_cli.read_json', return_value=mlp()):
                code, result = self.cli(path, 'preview-family', '-')
            self.assertEqual(code, 0, result); self.assertEqual(result['value']['status'], 'verified')
            self.assertEqual(before, path.read_bytes())
            edit = {'op': 'specialize-family', 'family': mlp(), 'bindings': {'hidden-width': 4}}
            with patch('vpw_cli.read_json', return_value=[edit]):
                code, imported = self.cli(path, 'apply', '-')
            self.assertEqual(code, 0, imported)
            workspace = Workspace.load(path)
            rr = imported['value']['edits'][0]['resolved']
            self.assertEqual(model_document(workspace.models(), rr['root'])['architecture_hash'], rr['architecture_hash'])
            self.assertEqual(workspace.get_spec('model', workspace.freeze_model(rr['root'])['id'])['root'], rr['root'])
            workspace.db.close()
            saved = path.read_bytes()
            with patch('vpw_cli.read_json', return_value=[edit]):
                self.assertEqual(self.cli(path, 'apply', '-')[0], 1)  # no hidden replacement
            self.assertEqual(path.read_bytes(), saved)
            with patch('vpw_cli.read_json', return_value=[{'op': 'create'}, edit]):
                self.assertEqual(self.cli(path, 'apply', '-')[0], 1)
            self.assertEqual(path.read_bytes(), saved)

    def test_stdlib_family_import_does_not_import_torch_or_qt(self):
        result = subprocess.run([sys.executable, '-c',
            'import sys,json; from app.families import specialize; '
            'r=specialize(json.load(open("examples/families/width-depth.json"))); '
            'assert r["contract"]["parameters"]==37; '
            'assert not any(k=="torch" or k.startswith("PyQt") for k in sys.modules)'],
            cwd=ROOT, capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()
