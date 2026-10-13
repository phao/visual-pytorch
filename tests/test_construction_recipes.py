"""Tiny pure graph checks for inspectable recipes, without Torch or Qt."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

from app.construction_recipes import build, catalogue, show
from app.core import GraphError, check_document, validate_model


ROOT = Path(__file__).resolve().parents[1]


def checked(name, options=None):
    edits = build(name, model_id='explicit', options=options)
    models, contracts = {}, {}
    for edit in edits:
        assert set(edit) == {'op', 'document'} and edit['op'] == 'create'
        doc = edit['document']
        for node in doc['graph']['nodes']:
            if 'ref' in node:
                assert node['ref'] in models  # consumer admitted after dependency
        check_document(doc)
        models[doc['id']] = doc
        contracts[doc['id']] = validate_model(models, doc['id'])
    return edits, models, contracts


class ConstructionRecipesTests(unittest.TestCase):
    def test_catalogue_is_closed_json_discloses_semantics_and_returns_fresh_values(self):
        entries = catalogue()
        self.assertEqual(len(entries), 5)
        self.assertEqual([x['name'] for x in entries], sorted(x['name'] for x in entries))
        self.assertEqual(json.loads(json.dumps(entries, allow_nan=False)), entries)
        self.assertEqual({x['kind'] for x in entries},
                         {'exact-model', 'parameter-component', 'approximate-estimator'})
        for entry in entries:
            self.assertFalse(entry['options']['additionalProperties'])
            self.assertTrue(entry['description'])
            self.assertTrue(entry['limitations'])
            for spec in entry['options']['properties'].values():
                self.assertIn(spec['type'], ('integer', 'boolean'))
        entries[0]['options']['properties'].clear()
        self.assertTrue(catalogue()[0]['options']['properties'])
        definition = show('neural-msvar')
        definition['options']['properties']['K']['default'] = 123
        self.assertEqual(show('neural-msvar')['options']['properties']['K']['default'], 5)

    def test_every_recipe_native_validation_order_determinism_and_no_side_effect_declarations(self):
        forbidden = {'dataset', 'records', 'training', 'result', 'code', 'script', 'expression', 'eval', 'exec'}
        def walk(value):
            if isinstance(value, dict):
                self.assertFalse(set(value) & forbidden)
                for item in value.values():
                    walk(item)
            elif isinstance(value, list):
                for item in value:
                    walk(item)
        for entry in catalogue():
            name = entry['name']
            with self.subTest(name=name):
                edits, models, contracts = checked(name)
                self.assertEqual(edits, build(name, model_id='explicit'))
                self.assertEqual(list(models), ['explicit' + suffix for suffix in entry['model_suffixes']])
                self.assertEqual([len(doc['graph']['nodes']) for doc in models.values()], entry['blocks'])
                self.assertGreater(contracts['explicit']['parameters'], 0)
                walk(edits)
                self.assertEqual(json.loads(json.dumps(edits, allow_nan=False)), edits)
                edits[-1]['document']['graph']['nodes'].clear()
                self.assertTrue(build(name, model_id='explicit')[-1]['document']['graph']['nodes'])

    def test_branching_nonlinear_independent_paths_and_ordered_concat(self):
        _, models, contracts = checked('branching-mlp', {'left': 2, 'right': 4, 'hidden': 3, 'outputs': 2})
        graph = models['explicit']['graph']
        nodes = {n['id']: n for n in graph['nodes']}
        self.assertEqual(nodes['join']['sources'], ['left_nonlinearity', 'right_nonlinearity'])
        self.assertEqual(nodes['left_nonlinearity']['activation'], 'Tanh')
        self.assertEqual(nodes['right_nonlinearity']['activation'], 'Tanh')
        self.assertFalse(any('share_with' in n for n in graph['nodes']))
        self.assertEqual(contracts['explicit']['shapes']['join']['output_shape'], [6])
        self.assertEqual(contracts['explicit']['parameters'], 9 + 15 + 14)
        self.assertEqual(graph['input_order'], ['left', 'right'])

    def test_shared_encoders_have_actual_single_parameter_owner(self):
        _, models, contracts = checked('shared-encoders', {'D': 4, 'hidden': 3})
        nodes = {n['id']: n for n in models['explicit']['graph']['nodes']}
        self.assertEqual(nodes['right_encoder']['share_with'], 'left_encoder')
        contract = contracts['explicit']
        encoders = [p for p in contract['plan'] if p['type'] == 'Linear']
        self.assertEqual(encoders[0]['layer'], encoders[1]['layer'])
        self.assertEqual(contract['parameters'], 15)
        self.assertEqual(contract['output_shape'], [6])

    def test_recurrent_dependency_state_and_prefix_control_adapters(self):
        for mask, reset in ((False, False), (True, False), (False, True), (True, True)):
            with self.subTest(mask=mask, reset=reset):
                edits, models, contracts = checked('recurrent-cell',
                    {'D': 2, 'hidden': 3, 'T': 4, 'mask': mask, 'reset': reset})
                self.assertEqual([e['document']['id'] for e in edits], ['explicit/cell', 'explicit'])
                node = next(n for n in models['explicit']['graph']['nodes'] if n['type'] == 'Recurrent')
                self.assertEqual(set(node['controls']), {'lengths'} | ({'mask'} if mask else set()) | ({'reset'} if reset else set()))
                self.assertEqual(node['controls']['lengths']['adapter'], 'exact-int64-v1')
                for role in ('mask', 'reset'):
                    if role in node['controls']:
                        self.assertEqual(node['controls'][role]['adapter'], 'exact-bool-v1')
                self.assertEqual(contracts['explicit']['parameters'], (2 + 3 + 1) * 3)
                self.assertEqual(contracts['explicit']['shapes']['out_sequence']['output_shape'], [4, 3])
                self.assertEqual(contracts['explicit']['shapes']['out_state']['output_shape'], [3])

    def test_finite_window_is_supervised_approximation_with_only_observation_input(self):
        _, models, contracts = checked('finite-window-regime-estimator',
                                      {'window': 3, 'D': 2, 'K': 4, 'hidden': 5})
        c = contracts['explicit']
        self.assertEqual(c['input_groups'][0]['id'], 'window')
        self.assertEqual(c['input_groups'][0]['shape'], [3, 2])
        self.assertEqual(c['output_shape'], [4])
        self.assertEqual(c['parameters'], 7 * 5 + 6 * 4)
        self.assertEqual(c['output_groups'][0]['representation'], 'probabilities')
        self.assertFalse(any(n['type'] == 'GaussianFilter' for n in models['explicit']['graph']['nodes']))
        self.assertIn('genuine regime targets', show('finite-window-regime-estimator')['limitations'])

    def test_msvar_recomputes_shapes_counts_and_uses_actual_constructors(self):
        from app import neural_msvar
        for k, p, d, h, t in ((2, 1, 2, 3, 4), (3, 2, 1, 4, 3), (5, 3, 3, 32, 12)):
            for controlled in (False, True):
                with self.subTest(K=k, p=p, D=d, controlled=controlled):
                    constructor = neural_msvar.controlled_drawing if controlled else neural_msvar.drawing
                    expected = constructor(states=k, lags=p, dimensions=d, hidden=h, steps=t)
                    expected['id'] = 'explicit'
                    _, models, contracts = checked('neural-msvar',
                        {'K': k, 'p': p, 'D': d, 'hidden': h, 'T': t, 'controlled': controlled})
                    self.assertEqual(models['explicit'], expected)
                    c = contracts['explicit']
                    self.assertEqual(c['parameters'], (p*d+1)*h + (h+1)*k*k + (p*d+1)*k*d + k + k*d)
                    self.assertEqual(c['shapes']['lags']['output_shape'], [t, p*d])
                    self.assertEqual(c['shapes']['transition']['output_shape'], [t, k, k])
                    self.assertEqual(c['shapes']['means']['output_shape'], [t, k, d])
                    self.assertEqual(c['shapes']['variances']['output_shape'], [k, d])
                    self.assertEqual(c['shapes']['out_filtered']['output_shape'], [t, k])
                    self.assertEqual(len(c['input_groups']), 7 if controlled else 3)
                    variance = next(n for n in models['explicit']['graph']['nodes'] if n['id'] == 'variances')
                    if controlled:
                        self.assertEqual(variance['transform'], 'exp')
                        self.assertEqual(variance['positive_floor'], 0.)

    def test_strict_integer_boolean_unknown_and_invalid_options(self):
        for entry in catalogue():
            name = entry['name']
            for options in ([], 'anything', 1, False, {'unknown': 2}, {'code': 'execute()'}, {1: 3}):
                with self.subTest(name=name, options=options), self.assertRaises(GraphError):
                    build(name, model_id='explicit', options=options)
            for key, spec in entry['options']['properties'].items():
                invalid = (0, 1, 'true', None, [], {}) if spec['type'] == 'boolean' else (True, False, 0, -1, 4097, 1.0, '2', None, [], {}, float('nan'))
                for value in invalid:
                    with self.subTest(name=name, key=key, value=value), self.assertRaises(GraphError):
                        build(name, model_id='explicit', options={key: value})
        with self.assertRaisesRegex(GraphError, 'p >= 1.*p=0 HMM'):
            build('neural-msvar', model_id='explicit', options={'p': 0})
        for name in ('unknown', None, [], True):
            with self.assertRaises(GraphError):
                show(name)
        for mid in ('', ' ', '\x00id', 'id\n', 'id\x7f', 'x' * 201, None, 1, True):
            with self.assertRaises(GraphError):
                build('branching-mlp', model_id=mid)

    def test_product_bounds_refuse_before_constructor_or_document_allocation(self):
        oversized = {
            'branching-mlp': {'left': 4096, 'right': 4096, 'hidden': 4096},
            'shared-encoders': {'D': 4096, 'hidden': 4096},
            'recurrent-cell': {'T': 4096, 'hidden': 4096},
            'finite-window-regime-estimator': {'window': 4096, 'D': 4096},
            'neural-msvar': {'K': 4096, 'D': 4096, 'p': 4096}}
        for name, options in oversized.items():
            with self.subTest(name=name), patch('app.construction_recipes._document', side_effect=AssertionError('allocated')), patch('app.neural_msvar.drawing', side_effect=AssertionError('allocated')):
                with self.assertRaisesRegex(GraphError, 'before construction'):
                    build(name, model_id='explicit', options=options)
        # Individually small dimensions can still exceed unroll/control work.
        with patch('app.neural_msvar.controlled_drawing', side_effect=AssertionError('allocated')):
            with self.assertRaisesRegex(GraphError, 'before construction'):
                build('neural-msvar', model_id='explicit', options={'T': 4096, 'K': 8, 'controlled': True})

    def test_options_are_not_mutated(self):
        options = {'K': 2, 'D': 2, 'p': 1, 'controlled': True}
        before = copy.deepcopy(options)
        build('neural-msvar', model_id='explicit', options=options)
        self.assertEqual(options, before)

    def test_fresh_no_site_import_and_all_recipes_have_no_numerical_or_config_import(self):
        script = """
import sys
from app.construction_recipes import catalogue, build
for recipe in catalogue():
    build(recipe['name'], model_id='fresh')
build('neural-msvar', model_id='fresh-controlled', options={'controlled': True})
build('recurrent-cell', model_id='fresh-recurrent', options={'mask': True, 'reset': True})
for name in sys.modules:
    assert name.split('.')[0] not in {'torch', 'numpy', 'PySide6', 'PyQt5', 'PyQt6'}
    assert name not in {'app.config', 'userconfig'}
print('pure recipes checked without site packages')
"""
        result = subprocess.run([sys.executable, '-S', '-B', '-c', script], cwd=ROOT,
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('pure recipes checked', result.stdout)


if __name__ == '__main__':
    unittest.main()
