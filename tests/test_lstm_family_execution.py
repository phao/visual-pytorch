"""Checked LSTM family constructors versus an independent gate/gradient oracle.

The oracle scans time and direction with explicit matrix gate equations. It
does not call LSTM/LSTMCell or any application execution helper.
"""
import copy
import itertools
import json
import unittest
from unittest.mock import patch

import torch

from app.core import GraphError, Workspace, generate_code
from app.family_export import family_source
from app.family_records import compile_specialization, retain
from app.families import specialize
from app.revisions import model_document
from app.task_execution import task_bundle, check_task_checkpoint
from app.task_training import train_task
from app.transport import run_json_worker
from examples.lstm_family import definition
from test_lstm_execution import data, named, oracle, returned
from test_lstm_task_execution import lstm_bundle

torch.set_num_threads(1)


class LSTMFamilyExecutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        namespace = {}
        cls.source = family_source(definition(), constrained=False)
        exec(cls.source['code'], namespace)
        cls.Family = namespace['ArchitectureFamily']

    def setUp(self):
        self.w = Workspace(); self.addCleanup(self.w.db.close)
        self.addCleanup(lambda: self.w.history.directory.cleanup() if self.w.history.directory else None)

    def check_oracle(self, model, *, lengths=None, nan_padding=True):
        plan = model.contract['shapes']['lstm']['lstm']; kw = plan['kwargs']
        values = data(model.contract, grad=True, lengths=lengths, nan_padding=nan_padding)
        keys = ('sequence', 'hidden', 'cell')
        independent = {key: values[key].detach().clone().requires_grad_() for key in keys}
        layer = (model.network if isinstance(model, self.Family) else model).layer_0
        weights = {key: value.detach().clone().requires_grad_() for key, value in layer.named_parameters()}
        expected = oracle(*(independent[key] for key in keys), weights, lengths=values['lengths'][:, 0].long(),
                          layers=kw['num_layers'], bidirectional=kw['bidirectional'], projection=kw['proj_size'])
        actual = returned(model.contract, model(named(model.contract, values)))
        for key in keys: torch.testing.assert_close(actual[key], expected[key], rtol=3e-6, atol=2e-6)
        def loss(result): return sum((i + .5) * result[key].square().sum() for i, key in enumerate(keys))
        aa = torch.autograd.grad(loss(actual), [*(values[key] for key in keys), *layer.parameters()])
        bb = torch.autograd.grad(loss(expected), [*(independent[key] for key in keys), *weights.values()])
        self.assertEqual(len(aa), len(bb))
        for a, b in zip(aa, bb): torch.testing.assert_close(a, b, rtol=8e-6, atol=3e-6)

    def test_sixteen_constructor_combinations_native_family_plain_export_and_gradients(self):
        for layers, bidirectional, projection, bias in itertools.product((1, 2), (False, True), (0, 2), (False, True)):
            bindings = {'time': 4, 'width': 2, 'cell-width': 3, 'layers': layers,
                        'projection': projection, 'directions': bidirectional, 'bias': bias}
            with self.subTest(bindings=bindings):
                retained = retain(self.w, family=definition(), bindings=bindings, materialize=False)
                native = compile_specialization(self.w, retained['specialization'], constrained=False)
                family = self.Family(bindings=bindings)
                family.network.load_state_dict(native.state_dict(), strict=True)
                namespace = {}; resolved = retained['resolved']
                exec(generate_code(resolved['models'], resolved['root']), namespace)
                plain = namespace['model']; plain.load_state_dict(native.state_dict(), strict=True)
                for route in (native, family, plain):
                    self.assertEqual(sum(p.numel() for p in route.parameters()), native.contract['parameters'])
                    self.check_oracle(route)

    def test_zero_lengths_preserve_explicit_state_and_zeros_do_not_cut_gradient_contract(self):
        model = self.Family(bindings={'time': 3, 'width': 1, 'cell-width': 3, 'projection': 2, 'layers': 1})
        self.check_oracle(model, lengths=[0] * 4)
        values = data(model.contract, lengths=[0] * 4, grad=True, nan_padding=True)
        result = returned(model.contract, model(named(model.contract, values)))
        self.assertEqual(result['sequence'].count_nonzero(), 0)
        for key in ('hidden', 'cell'): torch.testing.assert_close(result[key], values[key], rtol=0, atol=0)
        result['sequence'].sum().backward()
        for p in model.parameters(): self.assertIsNotNone(p.grad); self.assertEqual(p.grad.count_nonzero(), 0)

    def test_semantic_ids_are_stable_across_time_and_name_gates_projection_reverse_layers(self):
        a = self.Family(bindings={'time': 3}); b = self.Family(bindings={'time': 7})
        self.assertEqual(a.semantic_parameter_schema(), b.semantic_parameter_schema())
        ids = [json.loads(v['id']) for v in a.semantic_parameter_schema().values()]
        self.assertEqual(len(ids), len({json.dumps(v) for v in ids}))
        suffixes = {row[-1] for row in ids}
        self.assertIn('weight_hr_l0', suffixes); self.assertIn('weight_hr_l1_reverse', suffixes)
        self.assertIn('bias_hh_l1_reverse', suffixes)
        for row in ids:
            self.assertEqual(row[:-1], ['family', 'lstm-family', 'template', 'lstm-example', 'owner', 'lstm', 'parameter'])

    def test_strict_warm_start_with_changed_time_reproduces_same_prefix_outputs(self):
        short = self.Family(bindings={'time': 3}); long = self.Family(bindings={'time': 7})
        receipt = long.warm_start(short.semantic_state(), strict=True)
        self.assertFalse(receipt['optimizer_resume']); self.assertEqual(receipt['skipped'], [])
        self.assertEqual(len(receipt['loaded']), len(short.state_dict()))
        values = data(short.contract, lengths=[0, 3, 1, 2])
        extended = {k: v.clone() for k, v in values.items()}
        extended['sequence'] = torch.cat([values['sequence'], torch.full((4, 4, 3), float('nan'))], 1)
        a = returned(short.contract, short(named(short.contract, values)))
        b = returned(long.contract, long(named(long.contract, extended)))
        torch.testing.assert_close(a['sequence'], b['sequence'][:, :3], rtol=0, atol=0)
        for key in ('hidden', 'cell'): torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
        self.assertEqual(b['sequence'][:, 3:].count_nonzero(), 0)

    def test_constructor_change_requires_explicit_partial_loading_and_strict_is_atomic(self):
        source = self.Family()
        for bindings in ({'projection': 0}, {'directions': False}, {'bias': False}, {'layers': 1}):
            with self.subTest(bindings=bindings):
                target = self.Family(bindings=bindings)
                before = {k: v.clone() for k, v in target.state_dict().items()}
                with self.assertRaisesRegex(ValueError, 'Strict warm-start'): target.warm_start(source.semantic_state(), strict=True)
                for key, value in before.items(): torch.testing.assert_close(target.state_dict()[key], value, rtol=0, atol=0)
                receipt = target.warm_start(source.semantic_state())
                self.assertFalse(receipt['optimizer_resume'])
                self.assertTrue(receipt['skipped'] or receipt['missing'] or receipt['unused'])

    def test_tampered_source_schema_and_nonfinite_tensor_refuse_before_parameter_write(self):
        model = self.Family(); state = model.semantic_state()
        for mutation in ('runtime', 'family', 'binding', 'schema', 'nan'):
            bad = copy.deepcopy(state)
            if mutation == 'runtime': bad['runtime_sha256'] = '0' * 64
            elif mutation == 'family': bad['family']['revision'] = '0' * 64
            elif mutation == 'binding': bad['bindings']['time'] = 3
            elif mutation == 'schema': next(iter(bad['schema'].values()))['id'] = 'forged'
            else: next(iter(bad['weights'].values())).flatten()[0] = float('nan')
            # T alone is not a parameter identity change. The source binding
            # may legitimately change time with the same checked schema.
            if mutation == 'binding':
                bad['bindings']['cell-width'] = 8
            before = {k: v.clone() for k, v in model.state_dict().items()}
            with self.subTest(mutation=mutation), self.assertRaises((ValueError, GraphError)):
                model.warm_start(bad, strict=True)
            for key, value in before.items(): torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)

    def test_invalid_constructor_and_runtime_overrides_refuse_before_nn_lstm_allocation(self):
        with patch('torch.nn.LSTM', side_effect=AssertionError('must not allocate')):
            for kwargs in ({'bindings': {'projection': 7}}, {'runtime_temporal': {'sequence': [0]}},
                           {'runtime_inputs': {'sequence': [1]}}):
                with self.subTest(kwargs=kwargs), self.assertRaises((ValueError, GraphError)): self.Family(**kwargs)

    def test_thirty_five_update_fit_against_independent_teacher(self):
        bindings = {'time': 3, 'width': 1, 'cell-width': 3, 'projection': 2, 'layers': 1}
        torch.manual_seed(43); teacher = self.Family(bindings=bindings)
        values = data(teacher.contract, lengths=[0, 3, 1, 2])
        with torch.no_grad():
            targets = oracle(values['sequence'], values['hidden'], values['cell'],
                             dict(teacher.network.layer_0.named_parameters()), lengths=values['lengths'][:, 0].long(),
                             layers=1, bidirectional=True, projection=2)
        torch.manual_seed(91); model = self.Family(bindings=bindings); optimizer = torch.optim.Adam(model.parameters(), lr=.03)
        losses = []
        for _ in range(35):
            optimizer.zero_grad(); out = returned(model.contract, model(named(model.contract, values)))
            loss = sum((out[key] - targets[key]).square().mean() for key in targets)
            self.assertTrue(torch.isfinite(loss)); losses.append(loss.item()); loss.backward(); optimizer.step()
        self.assertLess(losses[-1], .35 * losses[0])
        restored = self.Family(bindings=bindings); restored.warm_start(model.semantic_state(), strict=True)
        for key, expected in model(named(model.contract, values)).items():
            torch.testing.assert_close(restored(named(restored.contract, values))[key], expected, rtol=0, atol=0)

    def test_retained_family_model_runs_real_typed_worker_with_exact_checkpoint(self):
        import sys
        bindings = {'time': 3, 'width': 1, 'cell-width': 3, 'projection': 2, 'layers': 1}
        resolved = specialize(definition(), bindings); base = lstm_bundle(2)
        bundle = task_bundle(model_document(resolved['models'], resolved['root']), base['dataset'], base['task'],
                             base['manifest'], base['protocol'])
        result = train_task(bundle, constrained=False)
        self.assertEqual(result['status'], 'completed', result['failure'])
        check_task_checkpoint(bundle, result['checkpoint'], constrained=False)
        worker = run_json_worker([sys.executable, '-u', '-m', 'app.worker'],
                                {'kind': 'task-v1', 'bundle': bundle, 'constrained': False}, timeout=30)
        self.assertEqual(worker['checkpoint'], result['checkpoint']); self.assertEqual(worker['history'], result['history'])


if __name__ == '__main__': unittest.main()
