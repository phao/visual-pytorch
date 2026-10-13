"""Independent bounded kernel checks; no graph integration, fits or broad suite."""
import copy
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest import mock

import torch
from torch import nn

from app import causal_lags as cl

torch.set_num_threads(1)


def node(lags=2, **fields):
    return {'type': 'CausalLagWindow', 'lags': lags, **fields}


def scalar_oracle(observations, history, lengths, p):
    """List/scalar indexing only: no cat, unfold, reshape or vectorized masking."""
    b, t, d = observations.shape
    result = torch.zeros((b, t, p * d), dtype=observations.dtype)
    for batch in range(b):
        for time in range(int(lengths[batch])):
            for slot in range(p):
                source_time = time - p + slot
                for feature in range(d):
                    value = (history[batch, source_time + p, feature]
                             if source_time < 0 else observations[batch, source_time, feature])
                    result[batch, time, slot * d + feature] = value
    return result


def isolated(script):
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    result = subprocess.run([sys.executable, '-I', '-c', script], env=env,
                            capture_output=True, text=True, timeout=30)
    if result.returncode:
        raise AssertionError(result.stderr)
    return result.stdout.strip()


class LagDeclarationTests(unittest.TestCase):
    def test_closed_version_presentation_copy_and_required_exact_integer(self):
        raw = node(label=['Lags'], id='lag', x=1, y=2, width=3, height=4, collapsed=True)
        before = copy.deepcopy(raw)
        normalized = cl.normalized(raw)
        self.assertEqual(normalized['op_version'], 1)
        normalized['label'].append('edit')
        self.assertEqual(raw, before)
        self.assertEqual(cl.constructor_args(raw), {'lags': 2})
        self.assertIn('oldest to newest', cl.describe(raw))
        self.assertEqual(cl.normalized(node(10**100))['lags'], 10**100)
        invalid = [None, [], {}, {'type': 'CausalLagWindow'}, node(type='Other')]
        invalid += [node(p) for p in (None, True, False, 0, -1, 2., '2', [2])]
        invalid += [node(op_version=v) for v in (True, False, 1., '1', None, 0, 3)]
        invalid += [node(**{key: None}) for key in
                    ('shape', 'dtype', 'device', 'trainable', 'share_with', 'padding',
                     'order', 'reset', 'reset_mask', 'execution_mask', 'emission_mask',
                     'missing', 'lengths', 'initial_history', 'max_elements', 'code')]
        for raw in invalid:
            with self.subTest(raw=raw), self.assertRaises(cl.CausalLagWindowError):
                cl.normalized(raw)
        for p in (True, 0, 1., None):
            with self.assertRaises(cl.CausalLagWindowError):
                cl.runtime_class(nn)(lags=p)

    def test_pure_import_normalize_describe_emission_and_bad_build(self):
        root = str(Path(__file__).resolve().parents[1])
        script = f'''import sys, importlib.abc
sys.path.insert(0, {root!r})
class BlockNumerics(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {{'torch', 'numpy', 'PySide6', 'PyQt6'}}:
            raise AssertionError('forbidden import: ' + fullname)
sys.meta_path.insert(0, BlockNumerics())
from app import causal_lags as cl
assert cl.normalized({{'type': 'CausalLagWindow', 'lags': 10**100}})['op_version'] == 1
assert cl.constructor_args({{'type': 'CausalLagWindow', 'lags': 2}}) == {{'lags': 2}}
assert 'oldest to newest' in cl.describe({{'type': 'CausalLagWindow', 'lags': 2}})
assert 'from app' not in cl.runtime_source()
for declaration, kwargs in (({{'type': 'CausalLagWindow', 'lags': True}}, {{}}),
                             ({{'type': 'CausalLagWindow', 'lags': 2}}, {{'max_elements': True}})):
    try: cl.build(declaration, **kwargs)
    except cl.CausalLagWindowError: pass
    else: raise AssertionError('invalid constructor accepted')
assert not any(k.split('.')[0] in {{'torch', 'numpy', 'PySide6', 'PyQt6'}} for k in sys.modules)
print('pure')
'''
        self.assertEqual(isolated(script), 'pure')

    def test_allocation_free_stateless_construction_and_closed_call(self):
        with mock.patch.object(torch, 'tensor', side_effect=AssertionError('allocation')):
            module = cl.build(node())
        self.assertEqual(list(module.parameters()), [])
        self.assertEqual(list(module.buffers()), [])
        self.assertEqual(dict(module.state_dict()), {})
        x, h, lengths = torch.zeros(1, 1, 1), torch.zeros(1, 2, 1), torch.tensor([1])
        for field in ('reset', 'execution_mask', 'emission_mask', 'missing'):
            with self.subTest(field=field), self.assertRaises(TypeError):
                module(x, h, lengths, **{field: None})
        with self.assertRaises(TypeError):
            module(x, h)


class LagNumericsTests(unittest.TestCase):
    def test_exact_scalar_oracle_history_order_multivariate_and_p_greater_than_t(self):
        for dtype in (torch.float32, torch.float64):
            for b, t, d, p in ((2, 5, 3, 2), (3, 2, 2, 5), (1, 4, 1, 1)):
                x = torch.arange(b * t * d, dtype=dtype).reshape(b, t, d)
                h = -(torch.arange(b * p * d, dtype=dtype).reshape(b, p, d) + 1)
                lengths = torch.tensor([max(0, t - i) for i in range(b)], dtype=torch.int64)
                output = cl.build(node(p))(x, h, lengths)
                self.assertTrue(torch.equal(output, scalar_oracle(x, h, lengths, p)))
                self.assertTrue(torch.equal(output[0, 0], h[0].flatten()))
                self.assertEqual(output.dtype, dtype)
                self.assertEqual(tuple(output.shape), (b, t, p * d))

    def test_current_observation_mutations_never_leak_into_own_time_or_past(self):
        x = torch.arange(14., dtype=torch.float64).reshape(1, 7, 2)
        h, lengths = torch.tensor([[[-2., -1.], [-4., -3.]]]), torch.tensor([7])
        h = h.to(x.dtype)
        module = cl.build(node(2))
        baseline = module(x, h, lengths)
        for time in range(7):
            changed = x.clone()
            changed[0, time] += 1000
            output = module(changed, h, lengths)
            self.assertTrue(torch.equal(output[:, :time + 1], baseline[:, :time + 1]))
            if time + 1 < 7:
                self.assertFalse(torch.equal(output[:, time + 1], baseline[:, time + 1]))
            self.assertTrue(torch.equal(output[:, time + 3:], baseline[:, time + 3:]))

    def test_zero_unequal_prefixes_nan_padding_sanitized_before_windowing(self):
        for dtype in (torch.float32, torch.float64):
            x = torch.tensor([[[1., 2.], [3., 4.], [float('nan'), float('inf')]],
                              [[float('nan'), -float('inf')]] * 3,
                              [[5., 6.], [float('nan'), float('nan')], [float('inf'), 0.]]], dtype=dtype)
            h = torch.tensor([[[7., 8.], [9., 10.]], [[float('nan'), float('inf')]] * 2,
                              [[11., 12.], [13., 14.]]], dtype=dtype)
            x.requires_grad_()
            h.requires_grad_()
            lengths = torch.tensor([2, 0, 1])
            original_x, original_h = x.detach().clone(), h.detach().clone()
            original_unfold = torch.Tensor.unfold

            def checked_unfold(value, *args, **kwargs):
                self.assertTrue(bool(torch.isfinite(value).all()), 'windowed unsanitized input')
                return original_unfold(value, *args, **kwargs)

            with mock.patch.object(torch.Tensor, 'unfold', checked_unfold):
                output = cl.build(node())(x, h, lengths)
            self.assertTrue(torch.equal(output, scalar_oracle(x.detach(), h.detach(), lengths, 2)))
            self.assertTrue(bool(torch.isfinite(output).all()))
            output.sum().backward()
            self.assertTrue(bool(torch.isfinite(x.grad).all()))
            self.assertTrue(bool(torch.isfinite(h.grad).all()))
            self.assertEqual(x.grad[1].abs().sum().item(), 0)
            self.assertEqual(h.grad[1].abs().sum().item(), 0)
            self.assertEqual(x.grad[0, 2].abs().sum().item(), 0)
            torch.testing.assert_close(x.detach(), original_x, equal_nan=True)
            torch.testing.assert_close(h.detach(), original_h, equal_nan=True)

    def test_all_zero_lengths_allow_nonfinite_history_and_have_zero_gradients(self):
        x = torch.full((2, 3, 2), float('nan'), requires_grad=True)
        h = torch.full((2, 5, 2), float('nan'), requires_grad=True)
        result = cl.build(node(5))(x, h, torch.zeros(2, dtype=torch.int64))
        self.assertTrue(torch.equal(result, torch.zeros(2, 3, 10)))
        result.sum().backward()
        self.assertTrue(torch.equal(x.grad, torch.zeros_like(x)))
        self.assertTrue(torch.equal(h.grad, torch.zeros_like(h)))

    def test_exact_independent_gradient_counts_and_per_time_causality(self):
        b, t, d, p = 2, 4, 2, 3
        x = torch.arange(b * t * d, dtype=torch.float64).reshape(b, t, d).requires_grad_()
        h = torch.arange(b * p * d, dtype=torch.float64).reshape(b, p, d).requires_grad_()
        lengths = torch.tensor([4, 2])
        output = cl.build(node(p))(x, h, lengths)
        gx, gh = torch.autograd.grad(output.sum(), (x, h), retain_graph=True)
        expected_x, expected_h = torch.zeros_like(x), torch.zeros_like(h)
        for batch in range(b):
            for time in range(int(lengths[batch])):
                for previous in range(time - p, time):
                    if previous < 0:
                        expected_h[batch, previous + p] += 1
                    else:
                        expected_x[batch, previous] += 1
        self.assertTrue(torch.equal(gx, expected_x))
        self.assertTrue(torch.equal(gh, expected_h))
        for time in range(t):
            gradient = torch.autograd.grad(output[:, time].sum(), x, retain_graph=True)[0]
            self.assertEqual(gradient[:, time:].abs().sum().item(), 0)
            self.assertEqual(gradient[:, :max(0, time - p)].abs().sum().item(), 0)
        self.assertTrue(bool(torch.isfinite(gx).all() & torch.isfinite(gh).all()))

    def test_noncontiguous_inputs_and_float64_gradcheck(self):
        x = torch.arange(12., dtype=torch.float64).reshape(1, 2, 6).transpose(1, 2).requires_grad_()
        h = torch.arange(8., dtype=torch.float64).reshape(1, 2, 4).transpose(1, 2).requires_grad_()
        lengths = torch.tensor([5])
        self.assertFalse(x.is_contiguous())
        self.assertFalse(h.is_contiguous())
        module = cl.build(node(4))
        self.assertTrue(torch.equal(module(x, h, lengths), scalar_oracle(x.detach(), h.detach(), lengths, 4)))
        self.assertTrue(torch.autograd.gradcheck(lambda a, z: module(a, z, lengths), (x, h)))

    def test_private_empty_batch_and_time_shapes(self):
        for b, t in ((0, 3), (2, 0), (0, 0)):
            for dtype in (torch.float32, torch.float64):
                x = torch.empty(b, t, 2, dtype=dtype, requires_grad=True)
                h = torch.full((b, 4, 2), float('nan'), dtype=dtype, requires_grad=True)
                output = cl.build(node(4))(x, h, torch.zeros(b, dtype=torch.int64))
                self.assertEqual(tuple(output.shape), (b, t, 8))
                self.assertEqual(output.dtype, dtype)
                output.sum().backward()
                self.assertTrue(torch.equal(x.grad, torch.zeros_like(x)))
                self.assertTrue(torch.equal(h.grad, torch.zeros_like(h)))


class LagGuardTests(unittest.TestCase):
    def test_structural_shape_dtype_layout_device_and_prefix_rejections(self):
        x, h, lengths = torch.zeros(2, 3, 2), torch.zeros(2, 2, 2), torch.tensor([3, 2])
        invalid = [(None, h, lengths), (x, None, lengths), (x, h, None),
                   (x[0], h, lengths), (x.unsqueeze(0), h, lengths),
                   (x, h[:1], lengths), (x, h[:, :1], lengths), (x, h[:, :, :1], lengths),
                   (x, h, lengths[:1]), (x, h, lengths[:, None]),
                   (x, h, lengths.to(torch.int32)), (x, h, lengths.float()),
                   (x, h, lengths.bool()), (x, h, torch.tensor([-1, 2])),
                   (x, h, torch.tensor([4, 2])), (x, h.double(), lengths),
                   (x.half(), h.half(), lengths), (x.long(), h.long(), lengths),
                   (x.bool(), h.bool(), lengths), (x.to(torch.complex64), h.to(torch.complex64), lengths),
                   (x.to_sparse(), h, lengths), (x, h.to_sparse(), lengths),
                   (x, h, lengths.to_sparse()), (x.to('meta'), h, lengths),
                   (x, h.to('meta'), lengths), (x, h, lengths.to('meta')),
                   (torch.empty(2, 3, 0), torch.empty(2, 2, 0), lengths)]
        for args in invalid:
            with self.subTest(shapes=[getattr(v, 'shape', None) for v in args]), self.assertRaises(cl.CausalLagWindowError):
                cl.build(node())(*args)

    def test_active_observation_and_history_nonfinite_rejected(self):
        for bad in (float('nan'), float('inf'), -float('inf')):
            for slot in ('observations', 'history'):
                x, h = torch.zeros(1, 2, 1), torch.zeros(1, 4, 1)
                if slot == 'observations':
                    x[0, 1, 0] = bad  # Last active x is still required finite.
                else:
                    h[0, 0, 0] = bad
                with self.subTest(slot=slot, bad=bad), self.assertRaisesRegex(cl.CausalLagWindowError, 'active'):
                    cl.build(node(4))(x, h, torch.tensor([2]))

    def test_budget_and_dimension_guards_precede_any_allocation(self):
        x = torch.tensor(0.).expand(1, 1001, 1)
        h = torch.tensor(0.).expand(1, 1001, 1)
        lengths = torch.tensor([1001])
        with (mock.patch.object(torch, 'isfinite', side_effect=AssertionError('finite allocation')),
              mock.patch.object(torch, 'arange', side_effect=AssertionError('time allocation')),
              mock.patch.object(torch, 'where', side_effect=AssertionError('mask allocation')),
              mock.patch.object(torch, 'cat', side_effect=AssertionError('stream allocation')),
              mock.patch.object(torch.Tensor, '__ge__', side_effect=AssertionError('length allocation'))):
            with self.assertRaisesRegex(cl.CausalLagWindowError, 'allocation budget'):
                cl.build(node(1001))(x, h, lengths)
            huge_p = cl.MAX_DIMENSION + 1
            with self.assertRaisesRegex(cl.CausalLagWindowError, 'dimensions'):
                cl.build(node(huge_p))(torch.empty(0, 0, 1), torch.empty(0, huge_p, 1), torch.empty(0, dtype=torch.int64))
            product_p = cl.MAX_DIMENSION // 2 + 1
            with self.assertRaisesRegex(cl.CausalLagWindowError, 'dimensions'):
                cl.build(node(product_p))(torch.empty(0, 0, 2), torch.empty(0, product_p, 2), torch.empty(0, dtype=torch.int64))

    def test_exact_work_limit_and_only_external_admission_can_raise_default(self):
        # B=1,T=2,D=1,p=2: 1+2+6+16+8=33 work elements.
        x, h, lengths = torch.ones(1, 2, 1), torch.zeros(1, 2, 1), torch.tensor([2])
        with self.assertRaisesRegex(cl.CausalLagWindowError, '33 work elements'):
            cl.build(node(), max_elements=32)(x, h, lengths)
        self.assertTrue(torch.equal(cl.build(node(), max_elements=33)(x, h, lengths),
                                    cl.build(node(), max_elements=0, admitted_elements=33)(x, h, lengths)))
        with self.assertRaises(cl.CausalLagWindowError):
            cl.build(node(), max_elements=cl.DEFAULT_MAX_ELEMENTS + 1)
        for kwargs in ({'max_elements': True}, {'max_elements': -1}, {'max_elements': 1.},
                       {'admitted_elements': True}, {'admitted_elements': -1}, {'admitted_elements': 1.}):
            with self.subTest(kwargs=kwargs), self.assertRaises(cl.CausalLagWindowError):
                cl.runtime_class(nn)(lags=2, **kwargs)
        # A real above-default request succeeds only with an explicit admission.
        t, p = 710, 710
        work = 1 + t + 3 * t + 4 * (t + p) + 2 * t * p
        x, h, lengths = torch.ones(1, t, 1), torch.zeros(1, p, 1), torch.tensor([t])
        with self.assertRaises(cl.CausalLagWindowError):
            cl.build(node(p))(x, h, lengths)
        result = cl.build(node(p), admitted_elements=work)(x, h, lengths)
        self.assertEqual(result.numel(), t * p)


class LagStandaloneTests(unittest.TestCase):
    def test_actual_function_source_and_fresh_standalone_value_gradient_guard_parity(self):
        source = cl.runtime_source()
        for definition in (cl.CausalLagWindowError, cl._lags, cl._limit, cl._budget, cl.runtime_class):
            self.assertIn(inspect.getsource(definition), source)
        self.assertNotIn('from app', source)
        script = '''import sys, json, importlib.abc
import torch
from torch import nn
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
threads = (torch.get_num_threads(), torch.get_num_interop_threads())
class BlockApp(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] == 'app': raise AssertionError('app dependency')
sys.meta_path.insert(0, BlockApp())
''' + source + '''
results = []
for dtype in (torch.float32, torch.float64):
    x = torch.tensor([[[1., 2.], [3., 4.], [float('nan'), float('inf')]],
                      [[float('nan'), float('nan')]] * 3], dtype=dtype, requires_grad=True)
    h = torch.tensor([[[7., 8.], [9., 10.], [11., 12.], [13., 14.]],
                      [[float('nan'), float('inf')]] * 4], dtype=dtype, requires_grad=True)
    lengths = torch.tensor([2, 0])
    module = runtime_class(nn)(lags=4)
    y = module(x, h, lengths)
    gx, gh = torch.autograd.grad(y.sum(), (x, h))
    assert torch.isfinite(y).all() and torch.isfinite(gx).all() and torch.isfinite(gh).all()
    assert not list(module.parameters()) and not dict(module.state_dict())
    results.append([y.tolist(), gx.tolist(), gh.tolist()])
    try: runtime_class(nn)(lags=4, max_elements=0)(x, h, lengths)
    except CausalLagWindowError: pass
    else: raise AssertionError('missing budget guard')
    try: module(x, h, lengths.float())
    except CausalLagWindowError: pass
    else: raise AssertionError('rounded lengths')
    try: module(x, h, lengths, reset=None)
    except TypeError: pass
    else: raise AssertionError('accepted reset')
assert threads == (torch.get_num_threads(), torch.get_num_interop_threads())
assert not any(k == 'app' or k.startswith('app.') for k in sys.modules)
print(json.dumps(results))
'''
        standalone = json.loads(isolated(script))
        for i, dtype in enumerate((torch.float32, torch.float64)):
            x = torch.tensor([[[1., 2.], [3., 4.], [float('nan'), float('inf')]],
                              [[float('nan'), float('nan')]] * 3], dtype=dtype, requires_grad=True)
            h = torch.tensor([[[7., 8.], [9., 10.], [11., 12.], [13., 14.]],
                              [[float('nan'), float('inf')]] * 4], dtype=dtype, requires_grad=True)
            y = cl.build(node(4))(x, h, torch.tensor([2, 0]))
            gx, gh = torch.autograd.grad(y.sum(), (x, h))
            self.assertEqual(standalone[i], [y.tolist(), gx.tolist(), gh.tolist()])


if __name__ == '__main__':
    unittest.main()
