"""Numerical internal policy proof, not persistence/export/worker admission."""
import copy
import json
import unittest
import torch
from torch import nn
from app.core import GraphError, compile_model, default_model
from app.evidence import parameter_schema
from app.parameter_execution import configure, trainable_parameters
from app.parameter_policy import graph_aliases

torch.set_num_threads(1)


class ParameterExecutionTests(unittest.TestCase):
    def linear(self):
        doc = default_model(); models = {doc['id']: doc}
        return compile_model(models, doc['id']), parameter_schema(models, doc['id'])

    def policy(self, entries):
        return {'version': 1, 'parameters': entries}

    def assert_unchanged(self, model, before, flags):
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, before[key]), key)
        self.assertEqual([p.requires_grad for p in model.parameters()], flags)

    def test_existing_native_schema_constructor_defaults_and_no_global_rng_use(self):
        model, schema = self.linear()
        before = copy.deepcopy(model.state_dict()); rng = torch.get_rng_state().clone()
        receipt = configure(model, schema)
        self.assert_unchanged(model, before, [True, True])
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(receipt['counts'], {'stored': 3, 'trainable': 3, 'frozen': 0})

    def test_exact_values_round_to_float32_and_freeze_does_not_cut_input_gradient(self):
        model, schema = self.linear()
        weight, bias = list(schema.values())
        configure(model, schema, self.policy({weight['id']: {'trainable': False,
            'initializer': {'kind': 'values', 'dtype': 'float32', 'values': [[2., -3.]]}},
            bias['id']: {'initializer': {'kind': 'zeros'}}}))
        x = torch.tensor([[1., 4.]], requires_grad=True)
        model(x).sum().backward()
        self.assertTrue(torch.equal(x.grad, torch.tensor([[2., -3.]])))
        self.assertIsNone(next(model.parameters()).grad)
        self.assertEqual(len(trainable_parameters(model)), 1)
        optimizer = torch.optim.Adam(trainable_parameters(model), lr=.1, foreach=False, fused=False)
        frozen = next(model.parameters()).detach().clone(); optimizer.step()
        self.assertTrue(torch.equal(next(model.parameters()), frozen))
        self.assertEqual(len(optimizer.state), 1)

    def test_private_random_generators_path_order_independence_and_repeatable_hash(self):
        first, schema = self.linear(); second, _ = self.linear()
        sid = next(iter(schema.values()))['id']
        policy = self.policy({sid: {'initializer': {'kind': 'normal', 'seed': 13, 'mean': 0., 'std': .2}}})
        # Constructors differ; explicitly zero the bias too.
        policy['parameters'][list(schema.values())[1]['id']] = {'initializer': {'kind': 'zeros'}}
        rng = torch.get_rng_state().clone()
        a = configure(first, schema, policy)
        b = configure(second, dict(reversed(list(schema.items()))), policy)
        self.assertEqual(a['initial_values_identity'], b['initial_values_identity'])
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        expected = torch.empty(1, 2).normal_(0., .2, generator=torch.Generator().manual_seed(13))
        self.assertTrue(torch.equal(next(first.parameters()), expected))

    def test_bad_late_initializer_leaves_earlier_values_and_flags_unchanged(self):
        model, schema = self.linear(); before = copy.deepcopy(model.state_dict())
        owners = list(schema.values())
        policy = self.policy({owners[0]['id']: {'trainable': False, 'initializer': {'kind': 'zeros'}},
            owners[1]['id']: {'initializer': {'kind': 'normal', 'seed': 0, 'mean': 3e38, 'std': 3e38}}})
        # The prepared seeded draw overflows float32; fail before any mutation.
        with self.assertRaisesRegex(GraphError, 'materialized'):
            configure(model, schema, policy)
        self.assert_unchanged(model, before, [True, True])

    def test_missing_extra_shape_dtype_and_existing_gradients_refuse_without_changes(self):
        for mutation in ('missing', 'extra', 'shape', 'dtype', 'gradient'):
            model, schema = self.linear()
            if mutation == 'missing': schema.pop(next(iter(schema)))
            elif mutation == 'extra': schema['missing'] = {'id': 'extra', 'shape': [1]}
            elif mutation == 'shape': next(iter(schema.values()))['shape'] = [2, 1]
            elif mutation == 'dtype': model.double()
            else: next(model.parameters()).grad = torch.ones_like(next(model.parameters()))
            before = copy.deepcopy(model.state_dict()); flags = [p.requires_grad for p in model.parameters()]
            with self.subTest(mutation=mutation), self.assertRaises(GraphError): configure(model, schema)
            self.assert_unchanged(model, before, flags)

    def test_native_alias_object_initialized_once_unique_optimizer_owner(self):
        model = nn.Module(); model.a = nn.Linear(2, 1, bias=False); model.b = model.a
        schema = {'a.weight': {'id': 'a/weight', 'shape': [1, 2]}}
        receipt = configure(model, schema, self.policy({'again/weight': {'initializer': {'kind': 'zeros'}}}),
                            aliases={'again/weight': 'a/weight'})
        self.assertEqual(receipt['addresses']['a/weight']['native_alias_paths'], ['a.weight', 'b.weight'])
        self.assertEqual(receipt['counts']['stored'], 2)
        self.assertEqual(len(trainable_parameters(model)), 1)
        self.assertTrue(torch.equal(model.a.weight, torch.zeros(1, 2)))

    def test_two_canonical_owners_for_one_object_and_storage_aliases_refuse(self):
        for same_object in (True, False):
            model = nn.Module(); model.a = nn.Parameter(torch.ones(2))
            model.b = model.a if same_object else nn.Parameter(model.a.detach())
            schema = {k: {'id': k, 'shape': [2]} for k in ('a', 'b')}
            with self.subTest(same_object=same_object), self.assertRaises(GraphError): configure(model, schema)

    def test_all_frozen_is_inference_valid_training_selection_refuses(self):
        model, schema = self.linear()
        receipt = configure(model, schema, self.policy({v['id']: {'trainable': False} for v in schema.values()}))
        self.assertEqual(receipt['counts']['trainable'], 0)
        self.assertTrue(torch.isfinite(model(torch.ones(2, 2))).all())
        with self.assertRaisesRegex(GraphError, 'no trainable'): trainable_parameters(model)

    def test_uniform_exact_seed_and_bounds(self):
        model, schema = self.linear(); sid = next(iter(schema.values()))['id']
        configure(model, schema, self.policy({sid: {'initializer': {'kind': 'uniform', 'seed': 5, 'low': -.2, 'high': .4}}}))
        expected = torch.empty(1, 2).uniform_(-.2, .4, generator=torch.Generator().manual_seed(5))
        self.assertTrue(torch.equal(next(model.parameters()), expected))

    def test_collapsed_float32_random_distribution_refuses_atomically(self):
        initializers = [{'kind': 'uniform', 'seed': 3, 'low': .1, 'high': .100000000000001},
                        {'kind': 'normal', 'seed': 3, 'mean': 0., 'std': 1e-100}]
        for init in initializers:
            model, schema = self.linear(); before = copy.deepcopy(model.state_dict())
            sid = next(iter(schema.values()))['id']
            with self.subTest(init=init), self.assertRaisesRegex(GraphError, 'float32'):
                configure(model, schema, self.policy({sid: {'trainable': False, 'initializer': init}}))
            self.assert_unchanged(model, before, [True, True])

    def test_nested_shared_graph_occurrence_declaration_freezes_one_owner(self):
        from test_shared_parameters import serial
        models = serial(kind='Module'); schema = parameter_schema(models, 'shared')
        model = compile_model(models, 'shared')
        receipt = configure(model, schema, self.policy({'shared/use2/use1/weight': {'trainable': False}}),
                            aliases=graph_aliases(models, 'shared'))
        self.assertEqual(receipt['counts'], {'stored': 6, 'trainable': 2, 'frozen': 4})
        x = torch.ones(2, 2, requires_grad=True); model(x).sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertEqual(len(trainable_parameters(model)), 1)

    def test_recurrent_and_projected_lstm_native_schema_binding(self):
        from test_recurrent_graph_contract import graph_fixture
        from examples.projected_lstm import models as lstm_fixture
        for models, mid in (graph_fixture(controls=False), lstm_fixture(steps=2)):
            with self.subTest(mid=mid):
                schema = parameter_schema(models, mid); model = compile_model(models, mid)
                receipt = configure(model, schema, self.policy({entry['id']: {'trainable': False}
                                                                for entry in schema.values()}))
                self.assertEqual(receipt['counts']['stored'], sum(p.numel() for p in model.parameters()))
                self.assertEqual(receipt['counts']['trainable'], 0)
                self.assertEqual(set(receipt['addresses']), {entry['id'] for entry in schema.values()})

    def lora(self, rank=2, temporal=False):
        nodes = [{'id': 'x', 'type': 'Input', 'shape': [3, 16] if temporal else [16]},
                 {'id': 'base', 'type': 'Linear', 'features': 16, 'bias': False},
                 {'id': 'A', 'type': 'Linear', 'features': rank, 'bias': False},
                 {'id': 'B', 'type': 'Linear', 'features': 16, 'bias': False},
                 {'id': 'scale', 'type': 'Scale', 'coefficient': 4. / rank, 'op_version': 1},
                 {'id': 'add', 'type': 'Add', 'sources': ['base', 'scale'], 'op_version': 1},
                 {'id': 'y', 'type': 'Output'}]
        pairs = [('x', 'base'), ('x', 'A'), ('A', 'B'), ('B', 'scale'),
                 ('base', 'add'), ('scale', 'add'), ('add', 'y')]
        doc = {'id': 'lora', 'name': 'Internal LoRA proof', 'graph': {'nodes': nodes,
               'edges': [{'from': a, 'to': b} for a, b in pairs]}}
        models = {'lora': doc}; schema = parameter_schema(models, 'lora')
        model = compile_model(models, 'lora')
        base = (.6 * torch.eye(16)).tolist()
        receipt = configure(model, schema, self.policy({
            'lora/base/weight': {'trainable': False, 'initializer': {'kind': 'values', 'dtype': 'float32', 'values': base}},
            'lora/A/weight': {'initializer': {'kind': 'normal', 'seed': 17, 'mean': 0., 'std': .05}},
            'lora/B/weight': {'initializer': {'kind': 'zeros'}}}))
        parameters = {entry['id']: dict(model.named_parameters())[path] for path, entry in schema.items()}
        return model, receipt, parameters

    def test_lora_initial_values_independent_analytic_gradients_and_input_path(self):
        model, receipt, parameters = self.lora()
        self.assertEqual(receipt['counts'], {'stored': 320, 'trainable': 64, 'frozen': 256})
        W, A, B = (parameters['lora/' + name + '/weight'] for name in ('base', 'A', 'B'))
        x = torch.linspace(-1., 1., 48).reshape(3, 16).requires_grad_()
        target = torch.linspace(.3, -.4, 48).reshape(3, 16)
        self.assertTrue(torch.equal(model(x), x @ W.T))
        loss = ((model(x) - target) ** 2).mean(); loss.backward()
        G = 2. * (x.detach() @ W.T - target) / 48
        torch.testing.assert_close(B.grad, 2. * G.T @ (x.detach() @ A.detach().T))
        self.assertTrue(torch.equal(A.grad, torch.zeros_like(A)))
        torch.testing.assert_close(x.grad, G @ W)
        self.assertIsNone(W.grad)

    def test_bounded_rank_and_temporal_lora_fits_match_independent_pytorch_formula(self):
        for rank in (1, 2, 4):
            for temporal in (False, True):
                with self.subTest(rank=rank, temporal=temporal):
                    model, _, parameters = self.lora(rank, temporal)
                    W, A, B = (parameters['lora/' + name + '/weight'] for name in ('base', 'A', 'B'))
                    frozen = W.detach().clone()
                    refA = nn.Parameter(A.detach().clone()); refB = nn.Parameter(B.detach().clone())
                    rng = torch.Generator().manual_seed(31)
                    x = torch.randn((32, 3, 16) if temporal else (32, 16), generator=rng)
                    targetA = torch.randn(rank, 16, generator=rng) * .15
                    targetB = torch.randn(16, rank, generator=rng) * .15
                    y = x @ frozen.T + (4. / rank) * (x @ targetA.T) @ targetB.T
                    optimizer = torch.optim.Adam(trainable_parameters(model), lr=.03, foreach=False, fused=False)
                    reference = torch.optim.Adam([refA, refB], lr=.03, foreach=False, fused=False)
                    initial = ((model(x) - y) ** 2).mean().item()
                    for _ in range(200):
                        optimizer.zero_grad(set_to_none=True); reference.zero_grad(set_to_none=True)
                        output = model(x)
                        expected = x @ frozen.T + (4. / rank) * (x @ refA.T) @ refB.T
                        loss = ((output - y) ** 2).mean(); ref_loss = ((expected - y) ** 2).mean()
                        loss.backward(); ref_loss.backward()
                        torch.testing.assert_close(A.grad, refA.grad, rtol=1e-5, atol=1e-7)
                        torch.testing.assert_close(B.grad, refB.grad, rtol=1e-5, atol=1e-7)
                        optimizer.step(); reference.step()
                    final = ((model(x) - y) ** 2).mean().item()
                    self.assertLess(final, initial * .01)
                    self.assertTrue(torch.equal(W, frozen)); self.assertIsNone(W.grad)
                    self.assertEqual(set(optimizer.state), {A, B})
                    torch.testing.assert_close(A, refA, rtol=1e-5, atol=1e-7)
                    torch.testing.assert_close(B, refB, rtol=1e-5, atol=1e-7)
                    print(json.dumps({'test': 'bounded-internal-LoRA-fit', 'rank': rank,
                        'shape': list(x.shape), 'scale': 4. / rank, 'updates': 200,
                        'initial_mse': initial, 'final_mse': final, 'frozen_base_bitwise': True,
                        'independent_formula_parameters_and_gradients': True,
                        'trainable_elements': A.numel() + B.numel()}), flush=True)


if __name__ == '__main__': unittest.main()
