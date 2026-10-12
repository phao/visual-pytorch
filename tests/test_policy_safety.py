"""Bounded reproductions: never allocate the review's oversized inventories."""
import unittest
from unittest.mock import patch
from app.core import GraphError, safety_report, validate_model
from app.policy_graph import owner_key, plan
from app.temporal_shapes_v2 import compile_temporal_model
from app.runtime_shapes import compile_runtime_model
from app.families import specialize


def expanding_dag():
    models = {'M0': {'id': 'M0', 'name': 'Scalar leaf', 'graph': {'nodes': [
        {'id': 'x', 'type': 'Input', 'features': 1}, {'id': 'w', 'type': 'Linear', 'features': 1, 'bias': False},
        {'id': 'y', 'type': 'Output'}], 'edges': [{'from': 'x', 'to': 'w'}, {'from': 'w', 'to': 'y'}]}}}
    for i in range(1, 31):
        mid = 'M' + str(i)
        models[mid] = {'id': mid, 'name': 'Level ' + str(i), 'graph': {'nodes': [
            {'id': 'x', 'type': 'Input', 'features': 1},
            {'id': 'a', 'type': 'Module', 'ref': 'M' + str(i - 1)},
            {'id': 'b', 'type': 'Module', 'ref': 'M' + str(i - 1)}, {'id': 'y', 'type': 'Output'}],
            'edges': [{'from': 'x', 'to': 'a'}, {'from': 'a', 'to': 'b'}, {'from': 'b', 'to': 'y'}]}}
    models['M30']['graph']['parameter_policy'] = {'version': 1, 'parameters': {}}
    return models


class PolicySafetyTests(unittest.TestCase):
    def test_billion_owner_reference_DAG_refuses_before_enumeration(self):
        models = expanding_dag()
        with patch('app.evidence.parameter_schema', side_effect=AssertionError('owner enumeration must not start')):
            for query in (lambda: validate_model(models, 'M30'), lambda: plan(models, 'M30'),
                          lambda: safety_report(models, 'M30')):
                with self.assertRaisesRegex(GraphError, 'structural limits'): query()

    def test_direct_policy_schema_refuses_before_recursive_enumeration(self):
        from app.evidence import parameter_schema
        models = expanding_dag()
        # Keep the public root callable but prohibit its recursive enumerator.
        with patch('app.evidence.parameter_schema', side_effect=AssertionError('recursive enumeration must not start')):
            with self.assertRaisesRegex(GraphError, 'structural limits'): parameter_schema(models, 'M30')

    def test_direct_alias_query_bounds_logical_occurrences_even_with_one_owner(self):
        from app.parameter_policy import graph_aliases
        models = expanding_dag()
        for i in range(1, 31): models['M' + str(i)]['graph']['nodes'][2]['share_with'] = 'a'
        with patch('app.evidence.parameter_schema', side_effect=AssertionError('alias enumeration must not start')):
            with self.assertRaisesRegex(GraphError, 'structural limits'): graph_aliases(models, 'M30')

    def test_whole_RAM_refusal_precedes_small_owner_inventory(self):
        models = {'m': {'id': 'm', 'name': 'Big linear', 'graph': {'nodes': [
            {'id': 'x', 'type': 'Input', 'features': 1024},
            {'id': 'w', 'type': 'Linear', 'features': 1024, 'bias': False}, {'id': 'y', 'type': 'Output'}],
            'edges': [{'from': 'x', 'to': 'w'}, {'from': 'w', 'to': 'y'}],
            'parameter_policy': {'version': 1, 'parameters': {owner_key(['w', 'weight']): {'initializer': {'kind': 'zeros'}}}}}}}
        with (patch('app.core.available_memory', return_value=32 * 1024**2),
              patch('app.evidence.parameter_schema', side_effect=AssertionError('RAM refusal must precede inventory'))):
            report = safety_report(models, 'm', constrained=True)
        self.assertFalse(report['allowed']); self.assertNotIn('parameter_counts', report['memory'])

    def test_actual_time_policy_initialization_budget_refuses_before_native_constructor(self):
        models = {'m': {'id': 'm', 'name': 'Runtime time', 'graph': {'nodes': [
            {'id': 'x', 'type': 'Input', 'shape': [3, 1024], 'axes': ['T', 'F']},
            {'id': 'w', 'type': 'Linear', 'features': 1024, 'bias': False}, {'id': 'y', 'type': 'Output'}],
            'edges': [{'from': 'x', 'to': 'w'}, {'from': 'w', 'to': 'y'}],
            'parameter_policy': {'version': 1, 'parameters': {owner_key(['w', 'weight']): {'initializer': {'kind': 'zeros'}}}}}}}
        with (patch('app.runtime_shapes.available_memory', return_value=2048 * 1024**2),
              patch('app.core.compile_model', side_effect=AssertionError('native allocation must not start'))):
            with self.assertRaisesRegex(GraphError, 'parameter-policy initialization'):
                compile_temporal_model(models, 'm', {'x': [0]}, constrained=True, initial_shapes={'x': [1, 1024]})

    def test_actual_spatial_policy_initialization_budget_refuses_before_native_constructor(self):
        models = {'m': {'id': 'm', 'name': 'Runtime height', 'graph': {'nodes': [
            {'id': 'x', 'type': 'Input', 'shape': [3, 16, 16], 'axes': ['C', 'H', 'W']},
            {'id': 'w', 'type': 'Linear', 'features': 1024, 'bias': False}, {'id': 'y', 'type': 'Output'}],
            'edges': [{'from': 'x', 'to': 'w'}, {'from': 'w', 'to': 'y'}],
            'parameter_policy': {'version': 1, 'parameters': {owner_key(['w', 'weight']): {'initializer': {'kind': 'zeros'}}}}}}}
        with (patch('app.runtime_shapes.available_memory', return_value=670 * 1024**2),
              patch('app.core.compile_model', side_effect=AssertionError('native allocation must not start'))):
            with self.assertRaisesRegex(GraphError, 'parameter-policy initialization'):
                compile_runtime_model(models, 'm', {'x': [1]}, constrained=True, initial_shapes={'x': [3, 1, 16]})

    def test_ancestor_selector_tracks_descendant_independent_and_tied_repeats(self):
        child = {'id': 'child', 'name': 'Child', 'graph': {'nodes': [
            {'id': 'x', 'type': 'Input', 'features': 1}, {'id': 'w', 'type': 'Linear', 'features': 1, 'bias': False},
            {'id': 'y', 'type': 'Output'}], 'edges': [{'from': 'x', 'to': 'w'}, {'from': 'w', 'to': 'y'}]}}
        root = {'id': 'root', 'name': 'Parent selection', 'graph': {'nodes': [
            {'id': 'x', 'type': 'Input', 'features': 1}, {'id': 'm', 'type': 'Module', 'ref': 'child'}, {'id': 'y', 'type': 'Output'}],
            'edges': [{'from': 'x', 'to': 'm'}, {'from': 'm', 'to': 'y'}],
            'parameter_policy': {'version': 1, 'parameters': {owner_key(['m', 'w', 'weight']): {'trainable': False}}}}}
        for mode in ('independent', 'tied'):
            family = {'version': 10, 'id': 'f', 'name': 'Nested repeat', 'root': 'root', 'models': {'root': root, 'child': child},
                      'symbols': [], 'assignments': [], 'constructors': [], 'runtime_policy': None,
                      'repeats': [{'id': 'r', 'model': 'child', 'nodes': ['w'], 'count': 2, 'mode': mode}]}
            result = specialize(family)
            counts = plan(result['models'], result['root'])['resolved']['counts']
            self.assertEqual(counts, {'stored': 2 if mode == 'independent' else 1, 'trainable': 0, 'frozen': 2 if mode == 'independent' else 1})
            selectors = result['models'][result['root']]['graph']['parameter_policy']['parameters']
            self.assertEqual(set(selectors), {owner_key(['m', 'r/' + str(i) + '/w', 'weight']) for i in range(2)})


if __name__ == '__main__': unittest.main()
