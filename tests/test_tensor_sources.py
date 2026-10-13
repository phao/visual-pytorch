"""Bounded source-kernel proof only: no graph admission, fits or UI changes."""
import ast
import copy
import inspect
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import unittest
from contextlib import ExitStack
from unittest import mock

import torch
from torch import nn

from app import tensor_sources as sources

torch.set_num_threads(1)
# Do not reset the process-wide interop pool during test-module import: another
# suite may already have started work. Fresh subprocesses configure their own
# pool explicitly; these bounded eager tests use one numerical thread.


def node(kind='Constant', shape=Ellipsis, **fields):
    return {'id': 'source', 'type': kind, 'shape': [2, 3] if shape is Ellipsis else shape, **fields}


def float_bytes(value):
    return b''.join(struct.pack('!f', v) for v in value.detach().reshape(-1).tolist()).hex()


class SourceNormalizationTests(unittest.TestCase):
    def test_defaults_copies_axes_and_presentation(self):
        raw = node(values=[[1, 2, 3], [4, 5, 6]], axes=['T', 'C'], label='Fixed',
                   x=1, y=2, width=3, height=4, collapsed=False)
        before = copy.deepcopy(raw)
        normalized = sources.normalized(raw)
        self.assertEqual(raw, before)
        self.assertEqual(normalized['op_version'], 1)
        self.assertEqual(normalized['values'], [[1., 2., 3.], [4., 5., 6.]])
        normalized['shape'][0] = 1
        normalized['axes'][0] = 'other'
        normalized['values'][0][0] = 7
        self.assertEqual(raw, before)
        self.assertEqual(sources.constructor_args(raw),
                         {'kind': 'Constant', 'shape': [2, 3], 'values': before['values']})

    def test_closed_fields_types_and_versions(self):
        malformed = [None, [], {}, {'type': []}, {'type': 'Other', 'shape': [1]}]
        malformed += [node(fill=0, op_version=v) for v in (True, False, 0, 2, 1., '1', None)]
        for field in ('trainable', 'initializer', 'parameter_policy', 'dtype', 'device', 'input', 'sources', 'unknown'):
            malformed += [node(fill=0, **{field: True}), node('TensorParameter', **{field: True})]
        malformed += [node(share_with='a', fill=0), node('TensorParameter', fill=0),
                      node('TensorParameter', values=[[0]*3]*2), node('TensorParameter', share_with=1)]
        for raw in malformed:
            with self.subTest(raw=raw), self.assertRaises(sources.TensorSourceError):
                sources.normalized(raw)

    def test_shape_contract_no_64_feature_limit(self):
        malformed = [None, [], [0], [-1], [True], [1.], ['2'], [1]*9,
                     (2, 3), [2**53], [float('inf')], [1, False]]
        for shape in malformed:
            with self.subTest(shape=shape), self.assertRaises(sources.TensorSourceError):
                sources.normalized(node(fill=0, shape=shape))
        for shape in ([65], [128, 256], [1]*8, [2**53-1]*8):
            self.assertEqual(sources.normalized(node(fill=0, shape=shape))['shape'], shape)

    def test_axes_closed_optional_contract(self):
        malformed = [None, [], ['T'], ['T', 'T'], ['', 'C'], [True, 'C'], ['x'*33, 'C'], ('T', 'C')]
        for axes in malformed:
            with self.subTest(axes=axes), self.assertRaises(sources.TensorSourceError):
                sources.normalized(node(fill=0, axes=axes))
        self.assertNotIn('axes', sources.normalized(node(fill=0)))

    def test_exact_nested_shape_not_reshape_or_broadcast(self):
        malformed = [None, 0, [1]*6, [[1, 2], [3, 4]], [[1, 2, 3]],
                     [[1, 2, 3], [4, 5]], [[1, 2, 3], [4, 5, [6]]], ((1, 2, 3), (4, 5, 6))]
        for values in malformed:
            with self.subTest(values=values), self.assertRaises(sources.TensorSourceError):
                sources.normalized(node(values=values))
        for raw in (node(), node(fill=0, values=[[0]*3]*2)):
            with self.assertRaises(sources.TensorSourceError):
                sources.normalized(raw)
        self.assertEqual(sources.normalized(node(shape=[1]*8, values=[[[[[[[[2]]]]]]]]))['shape'], [1]*8)

    def test_float32_finiteness_boolean_and_signed_zero(self):
        for bad in (True, False, None, '0', [], complex(1), float('nan'), float('inf'),
                    -float('inf'), 3.5e38, -3.5e38, 10**1000):
            for raw in (node(shape=[1], fill=bad), node(shape=[1], values=[bad])):
                with self.subTest(bad=repr(bad)), self.assertRaises(sources.TensorSourceError):
                    sources.normalized(raw)
        for value in (-0., -1e-100, 0., 1e-100, .1, sources.FLOAT32_MAX):
            normalized = sources.normalized(node(shape=[1], fill=value))['fill']
            self.assertEqual(struct.pack('!f', normalized), struct.pack('!f', value))
        self.assertEqual(math.copysign(1., sources.normalized(node(fill=-0.))['fill']), -1.)

    def test_huge_compact_fill_and_parameter_editing_remain_pure(self):
        for kind in sources.SOURCE_TYPES:
            raw = node(kind, [2**53-1]*8, **({'fill': -0.} if kind == 'Constant' else {'share_with': 'owner'}))
            result = sources.normalized(raw)
            self.assertEqual(len(result['shape']), 8)
            self.assertNotIn('values', result)
            self.assertLess(len(repr(result)), 1000)
            if 'share_with' in result:
                with self.assertRaisesRegex(sources.TensorSourceError, 'share_with'):
                    sources.constructor_args(result)
                result.pop('share_with')
            self.assertLess(len(repr(sources.constructor_args(result))), 1000)

    def test_import_and_normalization_isolation_in_fresh_python(self):
        root = str(Path(__file__).resolve().parents[1])
        script = '''import sys, importlib.abc
class BlockNumerics(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'numpy', 'PySide6', 'PyQt6', 'PyQt5'}:
            raise AssertionError('forbidden import: ' + fullname)
sys.meta_path.insert(0, BlockNumerics())
sys.path.insert(0, ROOT)
from app import tensor_sources as sources
sources.normalized({'type': 'Constant', 'shape': [2**53-1]*8, 'fill': -0.0})
sources.constructor_args({'type': 'TensorParameter', 'shape': [2**53-1]*8})
sources.runtime_source()
assert 'app.core' not in sys.modules
assert not any(k.split('.')[0] in {'torch', 'numpy', 'PySide6', 'PyQt6', 'PyQt5'} for k in sys.modules)
print('isolated')
'''.replace('ROOT', repr(root))
        result = subprocess.run([sys.executable, '-I', '-c', script], capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), 'isolated')


class SourceRuntimeTests(unittest.TestCase):
    def test_constant_buffer_signed_zero_no_weights_or_gradients(self):
        for raw in (node(fill=-0.), node(values=[[-0., 0., 2.], [1., -2., -0.]])):
            module = sources.build(raw)
            self.assertEqual(list(module.named_parameters()), [])
            self.assertEqual(list(module.named_buffers()), [('value', module.value)])
            self.assertEqual(dict(module.state_dict()), {})
            self.assertFalse(module.value.requires_grad)
            self.assertIsNone(module.value.grad)
            expected = (torch.full((2, 3), -0.) if 'fill' in raw else torch.tensor(raw['values']))
            self.assertEqual(float_bytes(module.value), float_bytes(expected))
            self.assertFalse(module(5).requires_grad)
            self.assertEqual(float_bytes(module(1)[0]), float_bytes(expected))

    def test_parameter_default_zero_native_owner_state_and_rng(self):
        rng = torch.get_rng_state().clone()
        module = sources.build(node('TensorParameter', [65]))
        self.assertEqual(list(dict(module.named_parameters())), ['value'])
        self.assertEqual(list(module.state_dict()), ['value'])
        self.assertEqual(list(module.buffers()), [])
        self.assertIs(type(module.value), nn.Parameter)
        self.assertTrue(module.value.is_leaf and module.value.requires_grad)
        self.assertTrue(module.value.is_contiguous())
        self.assertEqual(float_bytes(module.value), '00000000'*65)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))

    def test_explicit_batch_expansion_sample_T_and_zero_batch(self):
        for raw in (node(fill=.25, axes=['T', 'C']), node('TensorParameter', axes=['T', 'C'])):
            module = sources.build(raw)
            for batch in (0, 1, 5):
                output = module(batch)
                self.assertEqual(list(output.shape), [batch, 2, 3])
                self.assertEqual(output.dtype, torch.float32)
                self.assertEqual(output.device.type, 'cpu')
                self.assertEqual(output.untyped_storage().data_ptr(), module.value.untyped_storage().data_ptr())
                self.assertEqual(output.untyped_storage().nbytes(), 24)
                if batch > 1:
                    self.assertEqual(output.stride()[0], 0)
            for bad in (True, False, -1, 1., '2', None, torch.ones(2, 3)):
                with self.subTest(batch=bad), self.assertRaises(sources.TensorSourceError):
                    module(bad)
            with self.assertRaises(TypeError):
                module(torch.ones(2, 3), 2)

    def test_source_independence_and_parameter_gradient_sums_all_batch_axes(self):
        module = sources.build(node('TensorParameter', axes=['T', 'C']))
        with torch.no_grad():
            module.value.copy_(torch.tensor([[1., -2., 3.], [4., -5., 6.]]))
        unrelated = torch.randn(4, 2, 3, requires_grad=True)
        output = module(unrelated.shape[0])
        unrelated_other = unrelated.detach()*100
        self.assertEqual(float_bytes(output), float_bytes(module(unrelated_other.shape[0])))
        weights = torch.arange(24, dtype=torch.float32).reshape(4, 2, 3)
        (output * weights).sum().backward()
        # Independent scalar sum, not a second expand implementation.
        expected = [[sum(float(weights[b, t, c]) for b in range(4)) for c in range(3)] for t in range(2)]
        self.assertEqual(module.value.grad.tolist(), expected)
        self.assertIsNone(unrelated.grad)
        module.value.grad = None
        module(0).sum().backward()
        self.assertEqual(module.value.grad.tolist(), [[0.]*3]*2)

    def test_no_frozen_or_alias_guess_from_labels_share_or_equal_values(self):
        raw = node('TensorParameter', label='Frozen', share_with='unresolved/path')
        self.assertEqual(sources.normalized(raw)['share_with'], 'unresolved/path')
        for share_with in ('unresolved/path', ''):
            raw['share_with'] = share_with
            for construct in (sources.constructor_args, sources.build):
                with self.assertRaisesRegex(sources.TensorSourceError, 'share_with'):
                    construct(raw)
        raw.pop('share_with')  # Parent resolves references, then reuses the owner module.
        self.assertEqual(sources.constructor_args(raw), {'kind': 'TensorParameter', 'shape': [2, 3]})
        first, second = sources.build(raw), sources.build(raw)
        self.assertTrue(first.value.requires_grad and second.value.requires_grad)
        self.assertIsNot(first.value, second.value)
        self.assertNotEqual(first.value.untyped_storage().data_ptr(), second.value.untyped_storage().data_ptr())

    def test_cpu_float32_and_declared_shape_validation_after_conversion(self):
        for kind in sources.SOURCE_TYPES:
            for mutation in ('double', 'meta', 'shape'):
                module = sources.build(node(kind, **({'fill': 0} if kind == 'Constant' else {})))
                if mutation == 'double':
                    module.double()
                elif mutation == 'meta':
                    module.to('meta')
                else:
                    module.value = (nn.Parameter(torch.zeros(6)) if kind == 'TensorParameter' else torch.zeros(6))
                with self.subTest(kind=kind, mutation=mutation), self.assertRaises(sources.TensorSourceError):
                    module(2)
        module = sources.build(node(fill=0))
        module.value.requires_grad_(True)
        with self.assertRaises(sources.TensorSourceError):
            module(1)

    def test_constructor_forces_cpu_float32_under_changed_defaults(self):
        dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float64)
            with torch.device('meta'):
                for raw in (node(fill=.1), node(values=[[1]*3]*2), node('TensorParameter')):
                    module = sources.build(raw)
                    self.assertEqual(module.value.dtype, torch.float32)
                    self.assertEqual(module.value.device.type, 'cpu')
        finally:
            torch.set_default_dtype(dtype)

    def test_numerical_budget_default_explicit_and_before_all_tensor_allocations(self):
        cls = sources.runtime_class(nn)
        with ExitStack() as stack:
            mocks = [stack.enter_context(mock.patch.object(torch, name, side_effect=AssertionError('allocation before guard')))
                     for name in ('zeros', 'full', 'tensor', 'empty')]
            for args in ({'kind': 'TensorParameter', 'shape': [1001, 1000]},
                         {'kind': 'Constant', 'shape': [1001, 1000], 'fill': 0},
                         {'kind': 'Constant', 'shape': [1001, 1000], 'values': []},
                         {'kind': 'Constant', 'shape': [2**53-1]*8, 'fill': -0.}):
                with self.subTest(args=args), self.assertRaisesRegex(sources.TensorSourceError, 'budget'):
                    cls(**args)
            for raw in (node('TensorParameter', [1001, 1000]), node(shape=[1001, 1000], fill=0)):
                with self.assertRaisesRegex(sources.TensorSourceError, 'budget'):
                    sources.build(raw)
            for bad in (True, -1, 1., '5'):
                for field in ('max_elements', 'admitted_elements'):
                    with self.subTest(field=field, bad=bad), self.assertRaises(sources.TensorSourceError):
                        cls('TensorParameter', [2, 3], **{field: bad})
            for limits in ({'max_elements': 5}, {'admitted_elements': 5}, {'max_elements': 0}):
                with self.assertRaises(sources.TensorSourceError):
                    cls('TensorParameter', [2, 3], **limits)
            for allocation in mocks:
                allocation.assert_not_called()
        for limits in ({'max_elements': 6}, {'max_elements': 1, 'admitted_elements': 6}):
            self.assertEqual(sources.build(node('TensorParameter'), **limits).value.numel(), 6)

    def test_explicit_admission_overrides_prototype_budget_without_large_allocation(self):
        # Prove admission reaches the constructor without allocating a >1e6
        # fixture. The injected exception is after the guard, at materialization.
        for raw, allocator in ((node(shape=[1001, 1000], fill=0), 'full'),
                               (node('TensorParameter', [1001, 1000]), 'zeros')):
            with mock.patch.object(torch, allocator, side_effect=RuntimeError('admitted allocation boundary')) as allocation:
                with self.assertRaisesRegex(RuntimeError, 'admitted allocation boundary'):
                    sources.build(raw, admitted_elements=1_001_000)
                allocation.assert_called_once()
                self.assertEqual(allocation.call_args.args[0], [1001, 1000])
                self.assertEqual(allocation.call_args.kwargs, {'dtype': torch.float32, 'device': 'cpu'})

    def test_direct_class_closed_literals_and_policy_fields(self):
        cls = sources.runtime_class(nn)
        for kind, kwargs in [('Constant', {}), ('Constant', {'fill': None}),
                             ('Constant', {'fill': 0, 'values': [[0]*3]*2}),
                             ('Constant', {'values': [[True]*3]*2}),
                             ('TensorParameter', {'fill': None}), ('TensorParameter', {'values': []}),
                             ('Other', {})]:
            with self.subTest(kind=kind, kwargs=kwargs), self.assertRaises(sources.TensorSourceError):
                cls(kind, [2, 3], **kwargs)
        for extra in ('trainable', 'initializer', 'share_with', 'op_version'):
            with self.assertRaises(TypeError):
                cls('TensorParameter', [2, 3], **{extra: True})

    def test_existing_external_policy_nonzero_freeze_alias_initialize_and_hash_once(self):
        from app.evidence import digest
        from app.parameter_execution import configure
        from app.parameter_policy import resolve
        model = nn.Module()
        model.a = sources.build(node('TensorParameter'))
        model.b = model.a  # Sharing is an explicit graph-level action, external to this kernel.
        schema = {'a.value': {'id': 'root/source/value', 'shape': [2, 3]}}
        values = [[1., -2., 3.], [4., -5., 6.]]
        entry = {'trainable': False, 'initializer': {'kind': 'values', 'dtype': 'float32', 'values': values}}
        policy = {'version': 1, 'parameters': {'root/source/value': entry, 'root/reuse/value': entry}}
        aliases = {'root/reuse/value': 'root/source/value'}
        owner = model.a.value
        with mock.patch.object(torch, 'tensor', wraps=torch.tensor) as allocation, \
             mock.patch('app.parameter_execution.digest', wraps=digest) as state_hash:
            receipt = configure(model, schema, policy, aliases=aliases)
        self.assertIs(model.a.value, owner)
        self.assertIs(model.b.value, owner)
        self.assertEqual(allocation.call_count, 1)
        state_hash.assert_called_once_with({'dtype': 'float32', 'owners': {'root/source/value': values}})
        self.assertFalse(owner.requires_grad)
        self.assertEqual(owner.tolist(), values)
        self.assertEqual(receipt['counts'], {'stored': 6, 'trainable': 0, 'frozen': 6})
        self.assertEqual(receipt['addresses']['root/source/value']['native_alias_paths'], ['a.value', 'b.value'])
        self.assertEqual(receipt['initial_values_identity'], digest({'dtype': 'float32', 'owners': {'root/source/value': values}}))
        self.assertEqual(len(resolve(schema, policy, aliases=aliases)['owners']), 1)
        data = torch.ones(4, 2, 3, requires_grad=True)
        (data + model.a(4)).sum().backward()
        self.assertEqual(data.grad.tolist(), [[[1.]*3]*2]*4)
        self.assertIsNone(owner.grad)
        constant = sources.build(node(fill=2))
        empty = configure(constant, {})
        self.assertEqual(empty['counts'], {'stored': 0, 'trainable': 0, 'frozen': 0})


class SourceExportTests(unittest.TestCase):
    def test_emission_is_inspected_known_source_and_checked_import_closure(self):
        emitted = sources.runtime_source()
        self.assertIn(inspect.getsource(sources.runtime_class), emitted)
        tree = ast.parse(emitted)
        imports = [n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]
        self.assertFalse(any(isinstance(n, ast.ImportFrom) for n in imports))
        self.assertEqual({alias.name for n in imports for alias in n.names}, {'math', 'struct', 'torch'})
        calls = [n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
        self.assertNotIn('exec', calls)
        self.assertNotIn('eval', calls)

    def test_fresh_app_free_export_bytes_gradient_state_and_guard_parity(self):
        # Subprocess contains only inspected helper definitions and data kwargs.
        # -I excludes this checkout; app imports are actively blocked as well.
        script = '''import sys, importlib.abc, json, struct
class BlockApp(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'app', 'PySide6', 'PyQt6'}:
            raise AssertionError('app dependency: ' + fullname)
sys.meta_path.insert(0, BlockApp())
import torch
from torch import nn
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
'''+sources.runtime_source()+'''
def bytes_of(t):
    return b''.join(struct.pack('!f', v) for v in t.detach().reshape(-1).tolist()).hex()
rows = []
for args in json.loads(sys.stdin.read()):
    m = runtime_class(nn)(**args, admitted_elements=6)
    out = m(4)
    grad = None
    if m.kind == 'TensorParameter':
        with torch.no_grad():
            m.value.copy_(torch.arange(6, dtype=torch.float32).reshape(2, 3))
        out = m(4)
        (out * torch.arange(24, dtype=torch.float32).reshape(4, 2, 3)).sum().backward()
        grad = bytes_of(m.value.grad)
    rows.append({'value': bytes_of(m.value), 'output': bytes_of(out), 'gradient': grad,
                 'state': list(m.state_dict()), 'parameters': list(dict(m.named_parameters())),
                 'zero_shape': list(m(0).shape)})
from unittest import mock
from contextlib import ExitStack
with ExitStack() as stack:
    guards = [stack.enter_context(mock.patch.object(torch, name, side_effect=AssertionError('early allocation')))
              for name in ('zeros', 'full', 'tensor', 'empty')]
    for args in ({'kind': 'TensorParameter', 'shape': [2**53-1]*8},
                 {'kind': 'Constant', 'shape': [1001, 1000], 'fill': -0.0},
                 {'kind': 'Constant', 'shape': [1001, 1000], 'values': []}):
        try:
            runtime_class(nn)(**args)
        except TensorSourceError as error:
            assert 'budget' in str(error)
        else:
            raise AssertionError('budget bypass')
    assert all(g.call_count == 0 for g in guards)
assert not any(k == 'app' or k.startswith('app.') for k in sys.modules)
print(json.dumps(rows))
'''
        declarations = [node(fill=-0.), node(values=[[-0., 0., .1], [1., -2., -0.]]), node('TensorParameter')]
        expected = []
        for raw in declarations:
            module = sources.build(raw, admitted_elements=6)
            if module.kind == 'TensorParameter':
                with torch.no_grad():
                    module.value.copy_(torch.arange(6, dtype=torch.float32).reshape(2, 3))
            output = module(4)
            gradient = None
            if module.kind == 'TensorParameter':
                (output * torch.arange(24, dtype=torch.float32).reshape(4, 2, 3)).sum().backward()
                gradient = float_bytes(module.value.grad)
            expected.append({'value': float_bytes(module.value), 'output': float_bytes(output),
                             'gradient': gradient, 'state': list(module.state_dict()),
                             'parameters': list(dict(module.named_parameters())), 'zero_shape': list(module(0).shape)})
        result = subprocess.run([sys.executable, '-I', '-c', script],
                                input=json.dumps([sources.constructor_args(raw) for raw in declarations]),
                                capture_output=True, text=True, timeout=45,
                                env={**os.environ, 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
                                     'OPENBLAS_NUM_THREADS': '1', 'NUMEXPR_NUM_THREADS': '1'})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), expected)


if __name__ == '__main__':
    unittest.main()
