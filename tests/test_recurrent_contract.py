"""Pure explicit state-interface admission; no Torch dependency."""
import copy
import subprocess
import sys
import unittest
from unittest.mock import patch

from app.core import GraphError, default_model
from app.recurrent import recurrent_contract, recurrent_memory_report


def fixture(*, cnn=False):
    cell = default_model('Transition'); cell['id'] = 'transition'
    nodes = [{'id': 'x', 'type': 'Input', 'features': 2, 'label': 'observation'},
             {'id': 'h', 'type': 'Input', 'features': 2, 'label': 'previous state'},
             {'id': 'join', 'type': 'Concat', 'sources': ['x', 'h'], 'flatten': True},
             {'id': 'mix', 'type': 'Linear', 'features': 2, 'bias': True},
             {'id': 'tanh', 'type': 'Activation', 'activation': 'Tanh'},
             {'id': 'next', 'type': 'Output', 'label': 'next hidden'}]
    edges = [{'from': a, 'to': b} for a, b in
             [('x', 'join'), ('h', 'join'), ('join', 'mix'), ('mix', 'tanh'), ('tanh', 'next')]]
    if cnn:
        nodes[0] = {'id': 'frame', 'type': 'Input', 'shape': [1, 4, 5], 'axes': ['C', 'H', 'W']}
        nodes[2]['sources'] = ['flatten', 'h']
        nodes.extend([{'id': 'conv', 'type': 'Conv2d', 'out_channels': 2, 'kernel_size': [1, 1]},
                      {'id': 'pool', 'type': 'AdaptiveAvgPool2d', 'output_size': [1, 1]},
                      {'id': 'flatten', 'type': 'Flatten'}])
        edges[0]['from'] = 'flatten'
        edges.extend({'from': a, 'to': b} for a, b in [('frame', 'conv'), ('conv', 'pool'), ('pool', 'flatten')])
    cell['graph'] = {'nodes': nodes, 'edges': edges, 'input_order': ['frame' if cnn else 'x', 'h']}
    spec = {'version': 1, 'sequence_inputs': ['frame' if cnn else 'x'],
            'states': [{'input': 'h', 'output': 'next', 'port': 'hidden'}],
            'sequence_outputs': [{'output': 'next', 'port': 'trajectory'}]}
    return {cell['id']: cell}, cell['id'], spec


class RecurrentContractTests(unittest.TestCase):
    def contract(self, **kwargs):
        return recurrent_contract(*fixture(**kwargs))

    def test_closed_specification_without_mutating_drafts(self):
        models, mid, spec = fixture(); before = copy.deepcopy((models, spec))
        c = recurrent_contract(models, mid, spec)
        self.assertEqual((models, spec), before)
        self.assertEqual(c['parameters'], 10)
        self.assertEqual(c['inputs']['x']['shape'], [2])
        self.assertEqual(set(c['parameter_schema']), {'transition.layer_0.weight', 'transition.layer_0.bias'})
        spec['states'][0]['port'] = 'changed'
        self.assertEqual(c['specification']['states'][0]['port'], 'hidden')

    def test_cnn_body_is_a_real_graph_not_a_different_recurrent_catalogue(self):
        c = self.contract(cnn=True)
        self.assertEqual(c['parameters'], 14)
        self.assertEqual(c['inputs']['frame']['shape'], [1, 4, 5])
        self.assertGreater(c['convolution_workspace_elements'], 0)

    def test_reject_unknown_or_missing_fields_and_noninteger_versions(self):
        for change in ({'extra': True}, {'version': True}, {'version': 2}, {'version': 1.0}):
            models, mid, spec = fixture(); spec.update(change)
            with self.subTest(change=change), self.assertRaises(GraphError): recurrent_contract(models, mid, spec)
        for key in ('version', 'sequence_inputs', 'states', 'sequence_outputs'):
            models, mid, spec = fixture(); spec.pop(key)
            with self.subTest(key=key), self.assertRaises(GraphError): recurrent_contract(models, mid, spec)

    def test_inputs_are_bound_exactly_once_by_id_not_label(self):
        for inputs in ([], ['observation'], ['x', 'h'], ['x', 'x'], ['missing'], 'x', [False], ['']):
            models, mid, spec = fixture(); spec['sequence_inputs'] = inputs
            with self.subTest(inputs=inputs), self.assertRaises(GraphError): recurrent_contract(models, mid, spec)

    def test_states_are_explicit_and_have_closed_fields(self):
        for states in ([], None, [{'input': 'h', 'output': 'next', 'port': ''}],
                       [{'input': 'h', 'output': 'next', 'port': 'h', 'zero': True}],
                       [{'input': 'h', 'output': 'next'}],
                       [{'input': 'h', 'output': 'missing', 'port': 'h'}]):
            models, mid, spec = fixture(); spec['states'] = states
            with self.subTest(states=states), self.assertRaises(GraphError): recurrent_contract(models, mid, spec)

    def test_duplicate_state_bindings_and_ports_are_rejected(self):
        for key in ('input', 'output', 'port'):
            models, mid, spec = fixture(); spec['states'].append(dict(spec['states'][0]))
            with self.subTest(key=key), self.assertRaises(GraphError): recurrent_contract(models, mid, spec)

    def test_unknown_outputs_invalid_lists_and_duplicate_output_ports(self):
        for outputs in (None, [{'output': 'next hidden', 'port': 'y'}],
                        [{'output': 'next', 'port': 'y', 'alias': True}],
                        [{'output': 'next', 'port': 'y'}, {'output': 'next', 'port': 'y'}]):
            models, mid, spec = fixture(); spec['sequence_outputs'] = outputs
            with self.subTest(outputs=outputs), self.assertRaises(GraphError): recurrent_contract(models, mid, spec)

    def test_only_final_state_is_allowed_and_output_aliases_are_explicit(self):
        models, mid, spec = fixture(); spec['sequence_outputs'] = []
        self.assertEqual(recurrent_contract(models, mid, spec)['specification']['sequence_outputs'], [])
        spec['sequence_outputs'] = [{'output': 'next', 'port': 'a'}, {'output': 'next', 'port': 'b'}]
        self.assertEqual(len(recurrent_contract(models, mid, spec)['specification']['sequence_outputs']), 2)

    def test_next_state_shape_mismatch_is_actionable(self):
        models, mid, spec = fixture(); models[mid]['graph']['nodes'][3]['features'] = 3
        with self.assertRaisesRegex(GraphError, 'hidden.*preserve.*shape'): recurrent_contract(models, mid, spec)

    def test_state_axes_mismatch_is_rejected_even_at_equal_shape(self):
        models, mid, spec = fixture(); models[mid]['graph']['nodes'][1]['axes'] = ['hidden']
        with self.assertRaisesRegex(GraphError, 'axes'): recurrent_contract(models, mid, spec)

    def test_architecture_identity_excludes_geometry_and_labels_but_includes_cell_changes(self):
        models, mid, spec = fixture(); initial_contract = recurrent_contract(models, mid, spec)
        initial = initial_contract['transition_identity']
        for node in models[mid]['graph']['nodes']: node.update(x=500, collapsed=True, label=node['id'] + ' renamed')
        self.assertEqual(recurrent_contract(models, mid, spec)['transition_identity'], initial)
        self.assertEqual(recurrent_contract(models, mid, spec)['identity'], initial_contract['identity'])
        spec['states'][0]['port'] = 'other'
        self.assertNotEqual(recurrent_contract(models, mid, spec)['identity'], initial_contract['identity'])
        models[mid]['graph']['nodes'][4]['activation'] = 'ReLU'
        self.assertNotEqual(recurrent_contract(models, mid, spec)['transition_identity'], initial)

    def test_memory_scales_with_batch_and_full_unroll_not_just_inputs(self):
        c = self.contract(cnn=True)
        with patch('app.recurrent.available_memory', return_value=10**10):
            small = recurrent_memory_report(c, 2, 3)
            large = recurrent_memory_report(c, 2, 13)
            self.assertGreater(large['estimated_bytes'], small['estimated_bytes'])
            self.assertEqual(large['components']['parameters_optimizer_evidence'], small['components']['parameters_optimizer_evidence'])
            self.assertGreater(large['components']['activations_backward'], small['components']['activations_backward'])
            self.assertGreater(large['components']['convolution_workspace'], small['components']['convolution_workspace'])
            self.assertTrue(small['allowed'])

    def test_ram_budget_and_unknown_free_ram_fail_closed_but_can_be_explicitly_disabled(self):
        c = self.contract()
        for free in (None, 0, 1):
            with self.subTest(free=free), patch('app.recurrent.available_memory', return_value=free):
                self.assertFalse(recurrent_memory_report(c, 2, 3)['allowed'])
                self.assertTrue(recurrent_memory_report(c, 2, 3, False)['allowed'])
        with patch('app.recurrent.available_memory', return_value=10**9):
            self.assertFalse(recurrent_memory_report(c, 1, 1, memory_limit_percent=1.)['allowed'])
            self.assertTrue(recurrent_memory_report(c, 1, 1, memory_limit_percent=50.)['allowed'])

    def test_invalid_safety_arguments_are_not_coerced(self):
        c = self.contract()
        for batch, steps in ((True, 2), (1, 0), (0, 2), (1, 1.5)):
            with self.subTest(batch=batch, steps=steps), self.assertRaises(GraphError): recurrent_memory_report(c, batch, steps)
        for options in ({'constrained': 1}, {'training': 'yes'}, {'memory_limit_percent': 0}, {'memory_limit_percent': True}):
            with self.subTest(options=options), self.assertRaises(ValueError): recurrent_memory_report(c, 1, 1, **options)

    def test_import_contract_and_ram_estimate_do_not_import_torch(self):
        code = ('import sys; from test_recurrent_contract import fixture; '
                'from app.recurrent import recurrent_contract,recurrent_memory_report; '
                'c=recurrent_contract(*fixture()); recurrent_memory_report(c,2,3,False); '
                'assert "torch" not in sys.modules; print("pure-contract-no-torch")')
        result = subprocess.run([sys.executable, '-c', code], text=True, capture_output=True, check=True)
        self.assertEqual(result.stdout.strip(), 'pure-contract-no-torch')


if __name__ == '__main__': unittest.main()
