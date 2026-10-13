"""Bounded helper/activation checks, not graph, GUI or whole-model admission."""
import copy
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import unittest
from unittest import mock

import torch
from torch import nn

from app import activations as acts
from app import positive_variance as pv
from app.core import GraphError


def activation(name='LogSoftmax', **fields):
    return {'id': 'a', 'type': 'Activation', 'activation': name, **fields}


def variance(**fields):
    return {'id': 'v', 'type': 'PositiveVariance', **fields}


def activation_module(node, rank):
    # Evaluate only the trusted constructor string, without app runtime imports.
    return eval(acts.constructor_code(node, rank), {'nn': nn})


def isolated(script):
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1')
    result = subprocess.run([sys.executable, '-I', '-c', script], env=env,
                            capture_output=True, text=True, timeout=30)
    if result.returncode:
        raise AssertionError(result.stderr)
    return result.stdout.strip()


class ActivationContractTests(unittest.TestCase):
    def test_closed_fields_immutable_softplus_and_axis_defaults(self):
        raw = activation(label='Logs', x=1, collapsed=False)
        before = copy.deepcopy(raw)
        self.assertEqual(acts.normalize_activation(raw), dict(raw, dim=-1))
        self.assertEqual(raw, before)
        self.assertEqual(acts.constructor_code(raw, 3), 'nn.LogSoftmax(dim=3)')
        self.assertEqual(acts.constructor_code(activation('Softplus'), 3),
                         'nn.Softplus(beta=1, threshold=20)')
        for name in ('LogSoftmax', 'Softplus'):
            for key in ('trainable', 'share_with', 'beta', 'threshold', 'floor',
                        'positive_floor', 'dtype', 'op_version', 'unknown'):
                with self.subTest(name=name, key=key), self.assertRaises(GraphError):
                    acts.constructor_code(activation(name, **{key: 1}), 2)
        with self.assertRaises(GraphError):
            acts.normalize_activation(activation('Softplus', dim=-1))

    def test_all_sample_axes_and_closed_integer_grammar(self):
        for name in ('Softmax', 'LogSoftmax'):
            for rank in range(1, 9):
                for dim in range(-rank, rank):
                    self.assertEqual(acts.softmax_axis(activation(name, dim=dim), rank), dim % rank)
                    self.assertEqual(acts.constructor_code(activation(name, dim=dim), rank),
                                     f'nn.{name}(dim={dim % rank + 1})')
            for dim in (True, False, 1., '1', None, -9, 8, float('nan')):
                with self.subTest(name=name, dim=dim), self.assertRaises(GraphError):
                    acts.normalize_activation(activation(name, dim=dim))
            for dim, rank in ((1, 1), (-2, 1), (0, 0), (0, 9), (0, True), (0, 1.)):
                with self.assertRaises(GraphError):
                    acts.constructor_code(activation(name, dim=dim), rank)

    def test_legacy_elementwise_documents_are_preserved(self):
        for name in ('ReLU', 'Tanh', 'Sigmoid', 'GELU'):
            raw = activation(name, legacy='opaque')
            self.assertEqual(acts.normalize_activation(raw), raw)


class ActivationNumericsTests(unittest.TestCase):
    def test_logsoftmax_ranks_one_to_eight_normalization_and_batch_independence(self):
        for rank in range(1, 9):
            shape = [2] * rank
            for dim in range(-rank, rank):
                with self.subTest(rank=rank, dim=dim):
                    axis = dim % rank + 1
                    x = torch.linspace(-3, 4, 2 * math.prod(shape)).reshape(2, *shape)
                    module = activation_module(activation(dim=dim), rank)
                    y = module(x)
                    self.assertEqual(y.shape, x.shape)
                    torch.testing.assert_close(y.exp().sum(axis), torch.ones_like(y.sum(axis)),
                                               atol=2e-6, rtol=2e-6)
                    torch.testing.assert_close(module(x[:1]), y[:1])
                    torch.testing.assert_close(module(x[1:]), y[1:])
                    self.assertEqual(list(module.parameters()), [])

    def test_logsoftmax_independent_scalar_values_and_analytic_vjp(self):
        for shape in ([5], [2, 3], [2, 2, 3, 2]):
            for dim in range(len(shape)):
                x = torch.linspace(-3, 4, 2 * math.prod(shape), dtype=torch.float64).reshape(2, *shape).requires_grad_()
                u = torch.linspace(-.7, 1.2, x.numel(), dtype=x.dtype).reshape_as(x)
                axis = dim + 1
                moved = x.detach().movedim(axis, -1)
                rows = moved.reshape(-1, shape[dim]).tolist()
                weights = u.movedim(axis, -1).reshape(-1, shape[dim]).tolist()
                values, grads = [], []
                for row, weight in zip(rows, weights):
                    shift = max(row)
                    denominator = math.fsum(math.exp(v - shift) for v in row)
                    values.append([v - shift - math.log(denominator) for v in row])
                    grads.append([w - math.exp(v - shift) / denominator * math.fsum(weight)
                                  for v, w in zip(row, weight)])
                expected = torch.tensor(values, dtype=x.dtype).reshape(moved.shape).movedim(-1, axis)
                gradient = torch.tensor(grads, dtype=x.dtype).reshape(moved.shape).movedim(-1, axis)
                y = activation_module(activation(dim=dim), len(shape))(x)
                (y * u).sum().backward()
                torch.testing.assert_close(y, expected, atol=1e-12, rtol=1e-12)
                torch.testing.assert_close(x.grad, gradient, atol=1e-12, rtol=1e-12)

    def test_extreme_logits_are_stable_with_finite_gradients(self):
        x = torch.tensor([[10000., 10001., -10000.], [-10000., -9999., -30000.]], requires_grad=True)
        module = activation_module(activation(), 1)
        y = module(x)
        self.assertTrue(bool(torch.isfinite(y).all()))
        torch.testing.assert_close(y[0], y[1])
        torch.testing.assert_close(y.exp().sum(1), torch.ones(2))
        self.assertEqual(y[0, 2].exp().item(), 0.)
        self.assertTrue(math.isfinite(y[0, 2].item()))
        y.sum().backward()
        self.assertTrue(bool(torch.isfinite(x.grad).all()))
        singleton = torch.tensor([[1.], [-3.]], requires_grad=True)
        result = module(singleton)
        self.assertTrue(torch.equal(result, torch.zeros_like(result)))
        result.sum().backward()
        self.assertTrue(torch.equal(singleton.grad, torch.zeros_like(singleton)))

    def test_softplus_stable_scalar_values_derivatives_and_gradcheck(self):
        points = [-1000., -100., -5., 0., 5., 19., 20., 21., 1000.]
        x = torch.tensor(points, dtype=torch.float64, requires_grad=True)
        module = activation_module(activation('Softplus'), 1)
        y = module(x)
        expected = [v if v > 20 else math.log1p(math.exp(v)) for v in points]
        gradient = [1. if v > 20 else (math.exp(v) / (1 + math.exp(v))) for v in points]
        torch.testing.assert_close(y, torch.tensor(expected, dtype=x.dtype), atol=1e-14, rtol=1e-14)
        y.sum().backward()
        torch.testing.assert_close(x.grad, torch.tensor(gradient, dtype=x.dtype), atol=1e-14, rtol=1e-14)
        self.assertEqual(y[0].item(), 0.)
        self.assertTrue(torch.autograd.gradcheck(module, (torch.tensor([-2., 0., 3.], dtype=torch.float64, requires_grad=True),)))


class PositiveVarianceContractTests(unittest.TestCase):
    def test_required_floor_closed_version_and_presentation_copy(self):
        raw = variance(positive_floor=.1, label=['presentation'], x=3, collapsed=True)
        before = copy.deepcopy(raw)
        result = pv.normalized(raw)
        self.assertEqual(result['op_version'], 1)
        self.assertEqual(result['positive_floor'], struct.unpack('!f', struct.pack('!f', .1))[0])
        result['label'].append('changed')
        self.assertEqual(raw, before)
        self.assertEqual(pv.constructor_args(raw), {'positive_floor': result['positive_floor']})
        malformed = [None, [], {}, variance(), dict(raw, type='Other')]
        malformed += [dict(raw, op_version=v) for v in (True, 1., '1', None, 0, 2)]
        malformed += [dict(raw, **{key: 1}) for key in
                      ('trainable', 'share_with', 'shape', 'axes', 'beta', 'threshold', 'dtype', 'device', 'unknown')]
        for node in malformed:
            with self.subTest(node=node), self.assertRaises(pv.PositiveVarianceError):
                pv.normalized(node)

    def test_floor_rejects_underflow_nonfinite_and_no_silent_clamp(self):
        for floor in (True, False, None, '0.1', [], 0., -1., 1e-100, 1e-46,
                      float('nan'), float('inf'), -float('inf'), 3.5e38, 10**1000):
            with self.subTest(floor=repr(floor)), self.assertRaises(pv.PositiveVarianceError):
                pv.normalized(variance(positive_floor=floor))
            with self.assertRaises(pv.PositiveVarianceError):
                pv.runtime_class(nn)(positive_floor=floor)
        smallest = 2**-149
        self.assertEqual(pv.constructor_args(variance(positive_floor=smallest)), {'positive_floor': smallest})
        module = pv.build(variance(positive_floor=smallest))
        self.assertEqual(module(torch.tensor([-1000.])).item(), smallest)
        with self.assertRaises(TypeError):
            pv.runtime_class(nn)()

    def test_pure_normalization_emission_and_invalid_build_before_torch_import(self):
        root = str(Path(__file__).resolve().parents[1])
        script = f'''import sys, importlib.abc
sys.path.insert(0, {root!r})
class BlockNumerics(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {{'torch', 'numpy', 'PySide6'}}:
            raise AssertionError('forbidden import: ' + fullname)
sys.meta_path.insert(0, BlockNumerics())
from app import positive_variance as pv, activations as acts
node = {{'type': 'PositiveVariance', 'positive_floor': .25}}
assert pv.constructor_args(node) == {{'positive_floor': .25}}
assert 'from app' not in pv.runtime_source()
assert acts.constructor_code({{'activation': 'LogSoftmax', 'dim': 0}}, 2) == 'nn.LogSoftmax(dim=1)'
for kwargs in ({{'max_elements': True}}, {{'admitted_elements': -1}}):
    try: pv.build(node, **kwargs)
    except pv.PositiveVarianceError: pass
    else: raise AssertionError('accepted invalid budget')
assert 'torch' not in sys.modules
print('pure')
'''
        self.assertEqual(isolated(script), 'pure')


class PositiveVarianceNumericsTests(unittest.TestCase):
    def test_independent_scalar_values_derivatives_and_primitive_parity(self):
        from app.gaussian_filter import variances_from_raw
        for dtype in (torch.float32, torch.float64):
            module = pv.build(variance(positive_floor=.25))
            points = [-1000., -100., -5., 0., 5., 20., 21., 1000.]
            x = torch.tensor(points, dtype=dtype).reshape(2, 2, 2).requires_grad_()
            y = module(x)
            expected = [.25 + (v if v > 20 else math.log1p(math.exp(v))) for v in points]
            gradients = [1. if v > 20 else math.exp(v) / (1 + math.exp(v)) for v in points]
            torch.testing.assert_close(y, torch.tensor(expected, dtype=dtype).reshape_as(x))
            y.sum().backward()
            torch.testing.assert_close(x.grad, torch.tensor(gradients, dtype=dtype).reshape_as(x))
            reference = x.detach().clone().requires_grad_()
            primitive = variances_from_raw(reference, positive_floor=.25)
            primitive.sum().backward()
            self.assertTrue(torch.equal(y, primitive))
            self.assertTrue(torch.equal(x.grad, reference.grad))
            self.assertEqual(y.dtype, dtype)
            self.assertEqual(y.device.type, 'cpu')
            self.assertEqual(y.shape, x.shape)
            self.assertEqual(list(module.parameters()), [])
            self.assertEqual(list(module.buffers()), [])
            self.assertEqual(dict(module.state_dict()), {})

    def test_float64_gradcheck_noncontiguous_empty_and_scalar(self):
        module = pv.build(variance(positive_floor=.125))
        x = torch.tensor([[-2., .5, 3.], [1., -1., 4.]], dtype=torch.float64).t().requires_grad_()
        self.assertFalse(x.is_contiguous())
        self.assertTrue(torch.autograd.gradcheck(module, (x,)))
        self.assertEqual(module(torch.empty(0, 2)).shape, (0, 2))
        self.assertEqual(module(torch.tensor(0.)).shape, ())

    def test_invalid_dense_cpu_inputs_and_overflow_are_refused(self):
        module = pv.build(variance(positive_floor=.25))
        bad = [None, [0.], torch.tensor([1]), torch.tensor([True]), torch.tensor([1.], dtype=torch.float16),
               torch.tensor([1j]), torch.tensor([float('nan')]), torch.tensor([float('inf')]),
               torch.tensor([-float('inf')]), torch.tensor([1.]).to_sparse(), torch.empty(1, device='meta')]
        for raw in bad:
            with self.subTest(raw=raw), self.assertRaises(pv.PositiveVarianceError):
                module(raw)
        huge = pv.build(variance(positive_floor=pv.FLOAT32_MAX))
        with self.assertRaises(FloatingPointError):
            huge(torch.tensor([pv.FLOAT32_MAX]))
        raw = torch.tensor([1000., -1000.], requires_grad=True)
        y = module(raw)
        self.assertTrue(torch.equal(y, torch.tensor([1000.25, .25])))
        y.sum().backward()
        self.assertTrue(torch.equal(raw.grad, torch.tensor([1., 0.])))

    def test_default_guard_before_allocations_and_explicit_external_budget(self):
        module = pv.build(variance(positive_floor=.25))
        raw = torch.tensor(0.).expand(pv.DEFAULT_MAX_ELEMENTS + 1)
        with (mock.patch.object(torch, 'isfinite', side_effect=AssertionError('mask allocated')) as finite,
              mock.patch.object(torch.nn.functional, 'softplus', side_effect=AssertionError('output allocated')) as softplus):
            with self.assertRaisesRegex(pv.PositiveVarianceError, 'allocation budget'):
                module(raw)
            self.assertEqual(finite.call_count, 0)
            self.assertEqual(softplus.call_count, 0)
        with self.assertRaises(pv.PositiveVarianceError):
            pv.build(variance(positive_floor=.25), max_elements=3)(torch.zeros(4))
        admitted = pv.build(variance(positive_floor=.25), max_elements=3, admitted_elements=4)
        self.assertEqual(admitted(torch.zeros(4)).numel(), 4)
        for kwargs in ({'max_elements': True}, {'max_elements': -1}, {'admitted_elements': 1.}):
            with self.assertRaises(pv.PositiveVarianceError):
                pv.runtime_class(nn)(positive_floor=.25, **kwargs)
        with mock.patch.object(torch, 'tensor', side_effect=AssertionError('constructor allocated')):
            pv.build(variance(positive_floor=.25))

    def test_app_free_export_value_gradient_guard_and_thread_parity(self):
        emitted = pv.runtime_source()
        ctor = acts.constructor_code(activation(dim=0), 2)
        softplus_ctor = acts.constructor_code(activation('Softplus'), 2)
        script = '''import sys, importlib.abc
import torch
from torch import nn
threads = (torch.get_num_threads(), torch.get_num_interop_threads())
class BlockApp(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] == 'app': raise AssertionError('app dependency')
sys.meta_path.insert(0, BlockApp())
''' + emitted + f'''
assert threads == (torch.get_num_threads(), torch.get_num_interop_threads())
module = runtime_class(nn)(**{pv.constructor_args(variance(positive_floor=.25))!r})
x = torch.tensor([-5., 0., 5., 21.], dtype=torch.float64, requires_grad=True)
y = module(x)
ref = torch.nn.functional.softplus(x) + .25
assert torch.equal(y, ref)
assert torch.equal(torch.autograd.grad(y.sum(), x, retain_graph=True)[0], torch.autograd.grad(ref.sum(), x)[0])
assert not list(module.parameters()) and not dict(module.state_dict())
try: module(torch.tensor(0.).expand(DEFAULT_MAX_ELEMENTS + 1))
except PositiveVarianceError: pass
else: raise AssertionError('missing guard')
try: runtime_class(nn)(positive_floor=1e-100)
except PositiveVarianceError: pass
else: raise AssertionError('clamped floor')
logsoftmax = {ctor}
z = torch.tensor([[[1000., -1000.], [1001., -999.]], [[-1000., 1000.], [-999., 1001.]]], requires_grad=True)
logs = logsoftmax(z)
assert torch.isfinite(logs).all()
torch.testing.assert_close(logs.exp().sum(1), torch.ones(2, 2))
torch.testing.assert_close(logsoftmax(z[:1]), logs[:1])
logs.sum().backward()
assert torch.isfinite(z.grad).all()
softplus = {softplus_ctor}
assert softplus.beta == 1 and softplus.threshold == 20
assert softplus(torch.tensor([-1000.])).item() == 0
assert not any(name == 'app' or name.startswith('app.') for name in sys.modules)
print('app-free')
'''
        self.assertEqual(isolated(script), 'app-free')


if __name__ == '__main__':
    unittest.main()
