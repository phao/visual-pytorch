"""Independent scalar loss/gradient oracles and real prepared graph batches."""
import copy
import math
import random
import unittest

import torch
from app.core import GraphError, compile_model
from app.datasets import prepare
from app.task_runtime import MetricAccumulator, forward_ports, head_terms, tensor_batch
from test_datasets import data, task, contract, manifest

torch.set_num_threads(1)


class TaskRuntimeTests(unittest.TestCase):
    def test_masked_weighted_mse_scalar_value_and_analytic_gradient(self):
        x = torch.tensor([[1., 3.], [2., 5.], [-1., 4.]], dtype=torch.float64, requires_grad=True)
        y = torch.tensor([[0., 1.], [1., 2.], [-3., 3.]], dtype=torch.float64)
        mask = torch.tensor([[True, False], [True, True], [False, True]])
        weights = torch.tensor([1., 2., .5], dtype=torch.float64)
        terms = head_terms(x, y, mask, weights, 'mse')
        denominator = 1 + 2 + 2 + .5
        numerator = 1 + 2 * 1 + 2 * 9 + .5 * 1
        self.assertEqual(float(terms['numerator'].detach()), numerator)
        self.assertEqual(float(terms['denominator']), denominator)
        terms['mean'].backward()
        expected = [[2 / denominator, 0.], [4 / denominator, 12 / denominator], [0., 1 / denominator]]
        self.assertTrue(torch.allclose(x.grad, torch.tensor(expected, dtype=torch.float64), atol=1e-14, rtol=0))
        self.assertEqual(terms['elements'], 6); self.assertEqual(terms['included_elements'], 4)

    def test_mae_gradients_and_zero_subgradient(self):
        x = torch.tensor([[1., -2., 0.]], requires_grad=True); y = torch.zeros_like(x)
        terms = head_terms(x, y, torch.ones(1), torch.ones(1), 'mae')
        self.assertEqual(float(terms['mean'].detach()), 1.)
        terms['mean'].backward()
        self.assertTrue(torch.allclose(x.grad, torch.tensor([[1/3, -1/3, 0.]]), atol=1e-7, rtol=0))

    def test_population_metric_is_not_average_of_unequal_batch_means(self):
        x = torch.tensor([[1.], [2.], [8.], [4.], [9.]])
        y = torch.zeros_like(x); mask = torch.tensor([1., 1., 0., 1., 1.]); weights = torch.tensor([1., 2., 5., 3., .5])
        acc = MetricAccumulator('mse', 'final', 'validation')
        means = []
        for start, stop in ((0, 2), (2, 5)):
            terms = head_terms(x[start:stop], y[start:stop], mask[start:stop], weights[start:stop], 'mse')
            means.append(float(terms['mean']))
            acc.add(terms, ['r' + str(i) for i in range(start, stop)], checkpoint='final', population='validation')
        report = acc.report(); expected = (1 + 2 * 4 + 3 * 16 + .5 * 81) / (1 + 2 + 3 + .5)
        self.assertEqual(report['value'], expected); self.assertNotEqual(expected, sum(means) / 2)
        self.assertEqual(report['denominator'], 6.5); self.assertEqual(report['included_elements'], 4)
        self.assertEqual(report['evaluated_records'], 5); self.assertEqual(report['unique_observations'], 5)

    def test_24_masked_dense_configurations_and_partitions_match_scalar_oracles(self):
        rng = random.Random(123)
        for kind in ('mse', 'mae'):
            for config in range(12):
                n = 3 + config; width = 1 + config % 5
                xs = [[rng.uniform(-3., 3.) for _ in range(width)] for _ in range(n)]
                ys = [[rng.uniform(-3., 3.) for _ in range(width)] for _ in range(n)]
                ms = [[float(rng.random() > .3) for _ in range(width)] for _ in range(n)]
                ws = [[rng.choice((0., .2, 1., 3.)) for _ in range(width)] for _ in range(n)]
                ms[0][0] = ws[0][0] = 1.
                numerator = 0.; denominator = 0.; gradient = [[0.] * width for _ in range(n)]
                for i in range(n):
                    for j in range(width):
                        difference = xs[i][j] - ys[i][j]; effective = ms[i][j] * ws[i][j]
                        numerator += effective * (difference**2 if kind == 'mse' else abs(difference))
                        denominator += effective
                        derivative = 2 * difference if kind == 'mse' else float(difference > 0) - float(difference < 0)
                        gradient[i][j] = effective * derivative
                x = torch.tensor(xs, dtype=torch.float64, requires_grad=True); y = torch.tensor(ys, dtype=torch.float64)
                mask = torch.tensor(ms, dtype=torch.float64); weight = torch.tensor(ws, dtype=torch.float64)
                terms = head_terms(x, y, mask, weight, kind); terms['mean'].backward()
                expected = torch.tensor(gradient, dtype=torch.float64) / denominator
                self.assertTrue(torch.allclose(x.grad, expected, atol=1e-14, rtol=0))
                self.assertAlmostEqual(float(terms['mean'].detach()), numerator / denominator, places=13)
                for batch_size in (1, 2, 4):
                    acc = MetricAccumulator(kind, 'same-checkpoint', 'fixed-population')
                    for start in range(0, n, batch_size):
                        stop = min(n, start + batch_size)
                        part = head_terms(x.detach()[start:stop], y[start:stop], mask[start:stop], weight[start:stop], kind)
                        acc.add(part, ['r' + str(i) for i in range(start, stop)], checkpoint='same-checkpoint', population='fixed-population')
                    self.assertAlmostEqual(acc.report()['value'], numerator / denominator, places=13)

    def test_zero_weight_population_is_unavailable_not_nan_or_silent_mean(self):
        x = torch.tensor([[3.], [4.]], requires_grad=True)
        terms = head_terms(x, torch.zeros_like(x), torch.zeros(2), torch.ones(2), 'mse')
        self.assertIsNone(terms['mean']); self.assertEqual(terms['status'], 'unavailable')
        acc = MetricAccumulator('mse', 'c0', 'test'); acc.add(terms, ['a', 'b'], checkpoint='c0', population='test')
        report = acc.report(); self.assertEqual(report['denominator'], 0); self.assertIsNone(report['value'])
        self.assertIsNone(x.grad)
        terms2 = head_terms(x.detach(), torch.zeros_like(x), torch.ones(2), torch.zeros(2), 'mse')
        self.assertEqual(terms2['included_elements'], 0)

    def test_cross_entropy_spatial_weighted_scalar_oracle_and_gradient(self):
        logits = torch.tensor([[[.2, .4], [1.1, -.3], [-.7, .8]], [[1., -.2], [0., .5], [-1., .3]]], dtype=torch.float64, requires_grad=True)
        target = torch.tensor([[1, 2], [0, 1]], dtype=torch.int64)
        mask = torch.tensor([[1., 0.], [1., 1.]], dtype=torch.float64); weights = torch.tensor([2., 3.], dtype=torch.float64)
        terms = head_terms(logits, target, mask, weights, 'cross-entropy')
        denominator = 8.; numerator = 0.; expected_grad = torch.zeros_like(logits)
        correct = 0.
        for b in range(2):
            for site in range(2):
                numbers = [float(logits[b, c, site].detach()) for c in range(3)]; z = math.fsum(math.exp(v) for v in numbers)
                weight = float(mask[b, site]) * float(weights[b]); label = int(target[b, site])
                numerator += weight * (math.log(z) - numbers[label])
                if numbers.index(max(numbers)) == label: correct += weight
                for c in range(3): expected_grad[b, c, site] = weight * (math.exp(numbers[c]) / z - float(c == label)) / denominator
        self.assertAlmostEqual(float(terms['numerator'].detach()), numerator, places=13)
        self.assertEqual(float(terms['correct_numerator']), correct)
        terms['mean'].backward(); self.assertTrue(torch.allclose(logits.grad, expected_grad, atol=1e-14, rtol=0))

    def test_classification_accumulator_uses_same_masked_weighted_denominator(self):
        pred = torch.tensor([[0., 3., 1.], [2., 0., -1.], [4., 0., 1.]])
        target = torch.tensor([1, 1, 0]); mask = torch.tensor([1., 1., 0.]); weights = torch.tensor([1., 2., 10.])
        terms = head_terms(pred, target, mask, weights, 'cross-entropy')
        acc = MetricAccumulator('cross-entropy', 'c', 'test'); acc.add(terms, ['r0', 'r1', 'r2'], checkpoint='c', population='test')
        report = acc.report(); self.assertEqual(report['accuracy'], 1 / 3); self.assertEqual(report['accuracy_numerator'], 1.)

    def test_broadcast_dtype_ontology_nonfinite_and_weight_errors_fail_closed(self):
        prediction = torch.ones((2, 2), requires_grad=True); target = torch.zeros((2, 2))
        cases = [(prediction, target[:, :1], torch.ones(2), torch.ones(2), 'mse'),
                 (prediction, target.double(), torch.ones(2), torch.ones(2), 'mse'),
                 (prediction, target, torch.ones((1, 2)), torch.ones(2), 'mse'),
                 (prediction, target, torch.tensor([.5, 1.]), torch.ones(2), 'mse'),
                 (prediction, target, torch.ones(2), torch.tensor([-1., 1.]), 'mse'),
                 (prediction, target, torch.ones(2, requires_grad=True), torch.ones(2), 'mse'),
                 (prediction, target, torch.ones(2), torch.tensor([float('nan'), 1.]), 'mse'),
                 (prediction, target, torch.ones(2), torch.ones(2, dtype=torch.complex64), 'mse'),
                 (prediction, torch.tensor([0, -100]), torch.ones(2), torch.ones(2), 'cross-entropy'),
                 (prediction, torch.tensor([0, 2]), torch.ones(2), torch.ones(2), 'cross-entropy'),
                 (prediction, torch.tensor([0., 1.]), torch.ones(2), torch.ones(2), 'cross-entropy'),
                 (prediction, target, torch.ones(2), torch.ones(2), 'mystery-loss'),
                 (prediction * float('nan'), target, torch.zeros(2), torch.ones(2), 'mse')]
        for args in cases:
            with self.subTest(kind=args[-1]), self.assertRaises(GraphError): head_terms(*args)
        overflow = torch.tensor([[2e30], [2e30]])
        with self.assertRaisesRegex(GraphError, 'arithmetic'): head_terms(overflow, torch.zeros_like(overflow), torch.ones(2), torch.ones(2), 'mse')

    def test_metric_checkpoint_population_and_actual_record_counts_are_enforced(self):
        terms = head_terms(torch.ones((2, 1)), torch.zeros((2, 1)), torch.ones(2), torch.ones(2), 'mse')
        acc = MetricAccumulator('mse', 'c0', 'test')
        for checkpoint, population, ids in (('c1', 'test', ['a','b']), ('c0', 'train', ['a','b']), ('c0', 'test', ['a'])):
            with self.assertRaises(GraphError): acc.add(terms, ids, checkpoint=checkpoint, population=population)
        self.assertEqual(acc.report()['evaluated_records'], 0)
        acc.add(terms, ['a', 'a'], checkpoint='c0', population='test')
        self.assertEqual(acc.report()['unique_observations'], 1); self.assertEqual(acc.report()['evaluated_records'], 2)

    def test_real_prepared_model_batch_order_empty_batch_and_global_dtype(self):
        d = data(); t = task(); model, c = contract(); p = prepare(d, t, c, manifest(d), constrained=False)
        before = copy.deepcopy(p); original = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float64)
            batch = tensor_batch(p, t, 'test', [3, 0, 3])
        finally: torch.set_default_dtype(original)
        self.assertEqual(batch['inputs']['input'].dtype, torch.float32)
        self.assertEqual(batch['targets']['prediction'].dtype, torch.float32)
        self.assertEqual(batch['observation_ids'], ['r6', 'r3', 'r6'])
        self.assertEqual(p, before)
        module = compile_model({model['id']: model}, model['id'])
        with torch.no_grad(): module.layer_0.weight.fill_(2.); module.layer_0.bias.fill_(1.)
        predictions = forward_ports(module, batch['inputs'], c)
        self.assertTrue(torch.equal(predictions['output'], batch['targets']['prediction']))
        empty = tensor_batch(p, t, 'test', []); self.assertEqual(tuple(empty['inputs']['input'].shape), (0, 1))
        for indices in ([True], [-1], [999], (0, 1)):
            with self.assertRaises(GraphError): tensor_batch(p, t, 'test', indices)
        with self.assertRaises(GraphError): forward_ports(module, {'renamed-field': batch['inputs']['input']}, c)

    def test_actual_minibatch_optimizer_improves_prepared_regression(self):
        d = data(); t = task([{'kind': 'standardize'}]); model, c = contract()
        m = manifest(d, train=['r0', 'r1', 'r2', 'r3'], validation=['r4'], test=['r5', 'r6', 'r7'])
        p = prepare(d, t, c, m, constrained=False)
        with torch.random.fork_rng():
            torch.manual_seed(8); module = compile_model({model['id']: model}, model['id'])
        opt = torch.optim.Adam(module.parameters(), lr=.08)
        eval_batch = tensor_batch(p, t, 'test', [0, 1, 2])
        def evaluation():
            predictions = forward_ports(module, eval_batch['inputs'], c)
            return float(head_terms(predictions['output'], eval_batch['targets']['prediction'], eval_batch['masks']['prediction'], eval_batch['sample_weights']['prediction'], 'mse')['mean'].detach())
        initial = evaluation()
        for update in range(160):
            batch = tensor_batch(p, t, 'train', [0, 2] if update % 2 == 0 else [1, 3])
            pred = forward_ports(module, batch['inputs'], c)['output']
            terms = head_terms(pred, batch['targets']['prediction'], batch['masks']['prediction'], batch['sample_weights']['prediction'], 'mse')
            opt.zero_grad(); terms['mean'].backward(); opt.step()
        final = evaluation(); self.assertLess(final, .0001); self.assertLess(final, initial / 10000)


if __name__ == '__main__':
    unittest.main()
