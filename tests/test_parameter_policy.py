"""Internal policy foundation; NOT proof of persisted/runtime route support."""
import copy
import json
import subprocess
import sys
import unittest
from app.core import GraphError, default_model
from app.evidence import parameter_schema
from app.parameter_policy import graph_aliases, resolve


class ParameterPolicyTests(unittest.TestCase):
    def setUp(self):
        self.schema = {'layer_0.weight': {'id': 'net/base/weight', 'shape': [16, 16]},
                       'layer_1.weight': {'id': 'net/A/weight', 'shape': [2, 16]},
                       'layer_2.weight': {'id': 'net/B/weight', 'shape': [16, 2]}}

    def policy(self, parameters):
        return {'version': 1, 'parameters': parameters}

    def test_omitted_policy_defaults_all_owners_constructor_trainable(self):
        result = resolve(self.schema)
        self.assertEqual(result['counts'], {'stored': 320, 'trainable': 320, 'frozen': 0})
        self.assertTrue(all(e['trainable'] and e['initializer'] == {'kind': 'constructor'} for e in result['owners'].values()))

    def test_lora_counts_and_nonzero_A_zero_B_declarations(self):
        policy = self.policy({'net/base/weight': {'trainable': False, 'initializer': {'kind': 'values', 'dtype': 'float32', 'values': [[float(i == j) for j in range(16)] for i in range(16)]}},
                              'net/A/weight': {'initializer': {'kind': 'normal', 'seed': 7, 'mean': 0, 'std': .05}},
                              'net/B/weight': {'initializer': {'kind': 'zeros'}}})
        before = copy.deepcopy((self.schema, policy)); result = resolve(self.schema, policy)
        self.assertEqual(result['counts'], {'stored': 320, 'trainable': 64, 'frozen': 256})
        self.assertEqual((self.schema, policy), before)
        result['owners']['net/base/weight']['initializer']['values'][0][0] = 123
        self.assertEqual((self.schema, policy), before)

    def test_exact_default_and_empty_alias_declarations_have_same_identity(self):
        a = resolve(self.schema)
        b = resolve(self.schema, self.policy({'net/A/weight': {}}))
        self.assertEqual(a, b)

    def test_identical_aliases_count_one_owner_not_each_occurrence(self):
        aliases = {'net/again/weight': 'net/A/weight'}
        a = resolve(self.schema, self.policy({'net/A/weight': {'trainable': False}}))
        b = resolve(self.schema, self.policy({'net/A/weight': {'trainable': False}, 'net/again/weight': {'trainable': False}}), aliases=aliases)
        self.assertEqual(a, b)

    def test_conflicting_aliases_fail_in_both_orders(self):
        entries = [('net/A/weight', {'trainable': False}), ('net/again/weight', {'trainable': True})]
        for items in (entries, list(reversed(entries))):
            with self.assertRaisesRegex(GraphError, 'conflicting'):
                resolve(self.schema, self.policy(dict(items)), aliases={'net/again/weight': 'net/A/weight'})

    def test_equal_integer_float_alias_values_do_not_make_identity_order_dependent(self):
        entries = [('net/A/weight', {'initializer': {'kind': 'values', 'dtype': 'float32', 'values': [[1]*16]*2}}),
                   ('net/again/weight', {'initializer': {'kind': 'values', 'dtype': 'float32', 'values': [[1.0]*16]*2}})]
        aliases = {'net/again/weight': 'net/A/weight'}
        a = resolve(self.schema, self.policy(dict(entries)), aliases=aliases)
        b = resolve(self.schema, self.policy(dict(reversed(entries))), aliases=aliases)
        self.assertEqual(a, b)

    def test_equal_integer_float_random_declarations_have_same_normalized_identity(self):
        for kind, fields in (('normal', {'mean': 0, 'std': 1}), ('uniform', {'low': -1, 'high': 1})):
            a = {'kind': kind, 'seed': 7, **fields}
            b = {'kind': kind, 'seed': 7, **{k: float(v) for k, v in fields.items()}}
            first = resolve(self.schema, self.policy({'net/A/weight': {'initializer': a}}))
            second = resolve(self.schema, self.policy({'net/A/weight': {'initializer': b}}))
            self.assertEqual(first, second)

    def test_all_frozen_counts_and_bias_can_be_independently_selected(self):
        model = default_model(); schema = parameter_schema({model['id']: model}, model['id'])
        weight = next(e['id'] for p, e in schema.items() if p.endswith('.weight'))
        result = resolve(schema, self.policy({weight: {'trainable': False}}))
        self.assertEqual(result['counts'], {'stored': 3, 'trainable': 1, 'frozen': 2})
        all_frozen = resolve(schema, self.policy({e['id']: {'trainable': False} for e in schema.values()}))
        self.assertEqual(all_frozen['counts'], {'stored': 3, 'trainable': 0, 'frozen': 3})

    def test_initializers_shapes_values_and_seeds_reject_malformed_data(self):
        bad = [{'kind': 'values', 'dtype': 'float64', 'values': [[0]*16]*2},
               {'kind': 'values', 'dtype': 'float32', 'values': [0]*32},
               {'kind': 'values', 'dtype': 'float32', 'values': [[True]*16]*2},
               {'kind': 'values', 'dtype': 'float32', 'values': [[float('nan')]*16]*2},
               {'kind': 'values', 'dtype': 'float32', 'values': [[10**1000]*16]*2},
               {'kind': 'normal', 'seed': True, 'mean': 0, 'std': .1},
               {'kind': 'normal', 'seed': -1, 'mean': 0, 'std': .1},
               {'kind': 'normal', 'seed': 7, 'mean': 0, 'std': 0},
               {'kind': 'normal', 'seed': 7, 'mean': float('inf'), 'std': .1},
               {'kind': 'uniform', 'seed': 7, 'low': 1, 'high': 1},
               {'kind': 'uniform', 'seed': 7, 'low': -3e38, 'high': 3e38},
               {'kind': 'zeros', 'seed': 7}, {'kind': 'eval', 'code': '1+1'}]
        for init in bad:
            with self.subTest(init=str(init)[:80]), self.assertRaises(GraphError):
                resolve(self.schema, self.policy({'net/A/weight': {'initializer': init}}))

    def test_unknown_owners_fields_versions_and_nonboolean_trainable_refuse(self):
        bad = [self.policy({'missing': {}}), self.policy({'net/A/weight': {'trainable': 0}}),
               self.policy({'net/A/weight': {'trainable': 'false'}}),
               self.policy({'net/A/weight': {'requires_grad': False}}),
               {'version': True, 'parameters': {}}, {'version': 2, 'parameters': {}},
               {'version': 1, 'parameters': {}, 'resume': True}]
        for policy in bad:
            with self.subTest(policy=policy), self.assertRaises(GraphError): resolve(self.schema, policy)

    def test_alias_cycles_missing_targets_and_owner_shadowing_refuse(self):
        for aliases in ({'x': 'y', 'y': 'x'}, {'x': 'missing'}, {'net/A/weight': 'net/B/weight'}):
            with self.assertRaises(GraphError): resolve(self.schema, aliases=aliases)

    def test_ambiguous_or_malformed_schema_refuses_instead_of_guessing(self):
        bad = [dict(self.schema, other={'id': 'net/A/weight', 'shape': [2, 16]}),
               {'w': {'id': 'x', 'shape': [True]}}, {'w': {'id': 'x', 'shape': [0]}},
               {'w': {'id': 'x', 'shape': [], 'dtype': 'float32'}}]
        for schema in bad:
            with self.assertRaises(GraphError): resolve(schema)

    def test_paths_are_addresses_but_owner_shape_values_and_trainability_are_meaning(self):
        a = resolve(self.schema)
        moved = {'new.'+path: entry for path, entry in self.schema.items()}
        self.assertEqual(a['identity'], resolve(moved)['identity'])
        frozen = resolve(self.schema, self.policy({'net/A/weight': {'trainable': False}}))
        zero = resolve(self.schema, self.policy({'net/A/weight': {'initializer': {'kind': 'zeros'}}}))
        self.assertNotEqual(a['identity'], frozen['identity']); self.assertNotEqual(a['identity'], zero['identity'])

    def test_existing_schema_works_without_widening_old_graph_behavior(self):
        model = default_model(); schema = parameter_schema({model['id']: model}, model['id'])
        self.assertEqual(resolve(schema)['counts']['stored'], 3)
        model['graph']['nodes'][1]['trainable'] = False  # Old ignored metadata is not reinterpreted.
        self.assertEqual(resolve(parameter_schema({model['id']: model}, model['id']))['counts']['trainable'], 3)

    def test_fresh_pure_query_imports_no_torch_numpy_or_Qt(self):
        source = 'import sys,json;from app.parameter_policy import resolve;resolve({});print(json.dumps([n for n in sys.modules if n.split(".")[0] in ("torch","numpy","PyQt6")]))'
        process = subprocess.run([sys.executable, '-c', source], capture_output=True, text=True, timeout=20)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(json.loads(process.stdout), [])

    def test_graph_alias_inventory_traverses_nested_scopes_and_sharing_chains(self):
        from test_shared_parameters import serial
        for kind in ('Linear', 'Module'):
            models = serial(kind=kind); before = copy.deepcopy(models)
            schema = parameter_schema(models, 'shared'); aliases = graph_aliases(models, 'shared')
            self.assertEqual(models, before)
            self.assertEqual(len(aliases), 4 if kind == 'Linear' else 10)
            target = 'shared/use0/weight' if kind == 'Linear' else 'shared/use0/use0/weight'
            occurrence = 'shared/use2/weight' if kind == 'Linear' else 'shared/use2/use1/weight'
            self.assertEqual(aliases[occurrence], target)
            result = resolve(schema, self.policy({occurrence: {'trainable': False}}), aliases=aliases)
            self.assertFalse(result['owners'][target]['trainable'])
            self.assertEqual(result['counts']['stored'], 6)

    def test_slash_component_collision_refuses_instead_of_inferring_shared_owner(self):
        inner = {'id': 'inner', 'name': 'Inner', 'graph': {'nodes': [
            {'id': 'x', 'type': 'Input', 'features': 2}, {'id': 'b', 'type': 'Linear', 'features': 2},
            {'id': 'y', 'type': 'Output'}], 'edges': [{'from': 'x', 'to': 'b'}, {'from': 'b', 'to': 'y'}]}}
        root = {'id': 'root', 'name': 'Root', 'graph': {'nodes': [
            {'id': 'x', 'type': 'Input', 'features': 2}, {'id': 'a/b', 'type': 'Linear', 'features': 2},
            {'id': 'a', 'type': 'Module', 'ref': 'inner'}, {'id': 'y', 'type': 'Output'}],
            'edges': [{'from': 'x', 'to': 'a/b'}, {'from': 'a/b', 'to': 'a'}, {'from': 'a', 'to': 'y'}]}}
        with self.assertRaisesRegex(GraphError, 'unambiguous'): graph_aliases({'root': root, 'inner': inner}, 'root')
