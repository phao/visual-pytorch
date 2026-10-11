"""Actual graph/.vtw/export recurrence, with independent tensor/gradient oracles."""
import copy
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F
from app.core import GraphError, Workspace, compile_model, default_model, generate_code, validate_model
from app.evidence import parameter_schema
from test_recurrent_graph_contract import graph_fixture
from test_recurrent_execution import oracle

torch.set_num_threads(1)


def named(contract, values):
    return {g['name']: values[g['id']] for g in contract['input_groups']}


def output(contract, result, node):
    return result[next(g['name'] for g in contract['output_groups'] if g['id'] == node)]


def inputs(steps=4, *, cnn=False, grad=False):
    torch.manual_seed(23)
    return {'sequence': torch.randn(3, steps, *([1, 4, 5] if cnn else [2]), requires_grad=grad),
            'initial': torch.randn(3, 2, requires_grad=grad),
            'lengths': torch.tensor([[0.], [float(steps-1)], [float(steps)]]),
            'mask': torch.tensor([[1.]*steps, [float(i % 2 == 0) for i in range(steps)], [1.]*steps]),
            'reset': torch.tensor([[1.]*steps, [float(i == 2) for i in range(steps)], [float(i == 1) for i in range(steps)]])}


class RecurrentGraphExecutionTests(unittest.TestCase):
    def pair(self, models=None, mid=None, **kw):
        if models is None: models, mid = graph_fixture(**kw)
        torch.manual_seed(17)
        native = compile_model(models, mid, constrained=False)
        scope = {}; exec(generate_code(models, mid), scope)
        exported = scope['model']; exported.load_state_dict(native.state_dict(), strict=True)
        self.assertEqual(set(native.state_dict()), set(parameter_schema(models, mid)))
        return native, exported

    def test_real_forward_and_all_gradients_match_independent_row_recurrence(self):
        native, exported = self.pair()
        for model in (native, exported):
            data = inputs(grad=True); x, h = data['sequence'], data['initial']
            p = list(model.parameters()); q = [v.detach().clone().requires_grad_() for v in p]
            xi, hi = x.detach().clone().requires_grad_(), h.detach().clone().requires_grad_()
            y, end = oracle(xi, hi, *q, lengths=data['lengths'][:, 0].long(), mask=data['mask'].bool(), reset=data['reset'].bool())
            result = model(named(model.contract, data))
            a, b = output(model.contract, result, 'prediction'), output(model.contract, result, 'final')
            torch.testing.assert_close(a, y); torch.testing.assert_close(b, end)
            actual = torch.autograd.grad(a.square().sum() + .7*b.square().sum(), [x, h, *p])
            expected = torch.autograd.grad(y.square().sum() + .7*end.square().sum(), [xi, hi, *q])
            for aa, bb in zip(actual, expected): torch.testing.assert_close(aa, bb, rtol=3e-6, atol=2e-6)

    def test_named_and_packed_boundaries_produce_identical_port_outputs(self):
        for model in self.pair():
            data = named(model.contract, inputs())
            packed = torch.cat([data[g['name']].flatten(1) for g in model.contract['input_groups']], 1)
            a, b = model(data), model(packed)
            for key in a: torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
            tensor = model.forward_tensor(data)
            self.assertEqual(list(tensor.shape), [3, 10])

    def test_nan_padding_is_not_evaluated_and_all_masked_backward_is_zero(self):
        for model in self.pair():
            data = inputs(grad=True)
            with torch.no_grad():
                for row, length in enumerate(data['lengths'][:, 0].long()): data['sequence'][row, int(length):] = float('nan')
            result = model(named(model.contract, data)); y = output(model.contract, result, 'prediction')
            self.assertTrue(torch.isfinite(y).all()); y.sum().backward()
            self.assertTrue(torch.isfinite(data['sequence'].grad).all())
            model.zero_grad(); data = inputs(grad=True)
            data['lengths'].zero_()
            with torch.no_grad(): data['sequence'].fill_(float('nan'))
            result = model(named(model.contract, data))
            torch.testing.assert_close(output(model.contract, result, 'final'), data['initial'], rtol=0, atol=0)
            output(model.contract, result, 'prediction').sum().backward()
            for value in (data['sequence'], data['initial'], *model.parameters()): self.assertEqual(value.grad.count_nonzero(), 0)

    def test_explicit_control_adapters_reject_fractional_nonfinite_and_out_of_range(self):
        for model in self.pair():
            for role, value in [('lengths', -.1), ('lengths', 1.5), ('lengths', 5.), ('lengths', float('nan')),
                                ('mask', .5), ('reset', -1.), ('mask', float('inf'))]:
                data = inputs(); data[role].fill_(value)
                with self.subTest(exported=type(model).__name__, role=role, value=value), self.assertRaisesRegex(ValueError, 'exact-'):
                    model(named(model.contract, data))

    def test_actual_cell_parameters_are_reused_and_no_state_is_hidden_between_calls(self):
        for model in self.pair():
            before = [p.data_ptr() for p in model.parameters()]; data = named(model.contract, inputs())
            a, b = model(data), model(data)
            for key in a: torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
            self.assertEqual(before, [p.data_ptr() for p in model.parameters()]); self.assertEqual(sum(p.numel() for p in model.parameters()), 10)

    def test_cnn_cell_graph_outputs_and_gradients_match_independent_conv_pool_linear(self):
        for model in self.pair(cnn=True):
            data = inputs(cnn=True, grad=True); x, h = data['sequence'], data['initial']
            params = list(model.parameters()); copied = [p.detach().clone().requires_grad_() for p in params]
            xi, hi = x.detach().clone().requires_grad_(), h.detach().clone().requires_grad_()
            frames = F.adaptive_avg_pool2d(F.conv2d(xi.flatten(0, 1), *copied[:2]), (1, 1)).reshape(3, 4, 2)
            y, end = oracle(frames, hi, *copied[2:], lengths=data['lengths'][:, 0].long(), mask=data['mask'].bool(), reset=data['reset'].bool())
            result = model(named(model.contract, data)); actual = output(model.contract, result, 'prediction'); final = output(model.contract, result, 'final')
            torch.testing.assert_close(actual, y); torch.testing.assert_close(final, end)
            aa = torch.autograd.grad(actual.square().sum() + final.sum(), [x, h, *params])
            bb = torch.autograd.grad(y.square().sum() + end.sum(), [xi, hi, *copied])
            for a, b in zip(aa, bb): torch.testing.assert_close(a, b, rtol=3e-6, atol=2e-6)

    def test_two_recurrent_nodes_with_explicit_share_use_one_cell_parameter_owner(self):
        models, mid = graph_fixture(); g = models[mid]['graph']; node = copy.deepcopy(g['nodes'][2]); node.update(id='scan2', share_with='scan')
        g['nodes'].extend([node, {'id': 'second', 'type': 'Output'}])
        g['edges'].extend([dict(e, to='scan2') for e in list(g['edges']) if e['to'] == 'scan'])
        g['edges'].append({'from': 'scan2', 'from_port': 'trajectory', 'to': 'second'})
        self.assertEqual(validate_model(models, mid)['parameters'], 10)
        for model in self.pair(models, mid):
            self.assertEqual(len(list(model.parameters())), 2)
            result = model(named(model.contract, inputs()))
            torch.testing.assert_close(output(model.contract, result, 'prediction'), output(model.contract, result, 'second'), rtol=0, atol=0)

    def test_constrained_native_graph_rechecks_actual_unroll_before_any_cell_call(self):
        native = compile_model(*graph_fixture(), constrained=False)
        layer = native.layer_0; layer.constrained = True
        calls = []; hook = layer.transition.register_forward_pre_hook(lambda *_: calls.append(1)); self.addCleanup(hook.remove)
        with patch('app.recurrent.available_memory', return_value=1), self.assertRaisesRegex(GraphError, 'RAM|memory'):
            native(named(native.contract, inputs()))
        self.assertEqual(calls, [])

    def test_sqlite_reload_preserves_native_strict_state_and_export_prediction(self):
        models, mid = graph_fixture(); a, _ = self.pair(models, mid)
        w = Workspace(); self.addCleanup(w.db.close)
        w.put(models['transition']); w.put(models[mid])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'sequence.vtw'; w.save(path)
            opened = Workspace.load(path); self.addCleanup(opened.db.close)
            b, c = self.pair(opened.models(), mid)
            for model in (b, c):
                model.load_state_dict(a.state_dict(), strict=True)
                expected, actual = a(named(a.contract, inputs())), model(named(model.contract, inputs()))
                for key in expected: torch.testing.assert_close(expected[key], actual[key], rtol=0, atol=0)

    def test_nested_module_wrappers_preserve_current_call_recurrent_safety_and_percentage(self):
        models, mid = graph_fixture(); c = validate_model(models, mid)
        # Two ordinary references surround the new block; the policy must survive
        # the entire path, not only a direct Recurrent constructor.
        for wrapper_id in ('inner-wrapper', 'outer-wrapper'):
            wrapper = default_model(wrapper_id, inputs=c['inputs'], outputs=c['outputs']); wrapper['id'] = wrapper_id
            layer = next(n for n in wrapper['graph']['nodes'] if n['type'] == 'Linear')
            layer.pop('features'); layer.pop('bias'); layer.update(type='Module', ref=mid)
            models[wrapper_id] = wrapper; mid = wrapper_id
        with patch('app.core.available_memory', return_value=2**40), patch('app.recurrent.available_memory', return_value=2**40):
            model = compile_model(models, mid, constrained=True, memory_limit_percent=13.)
        layer = model.layer_0.layer_0.layer_0
        self.assertTrue(layer.constrained); self.assertEqual(layer.memory_limit_percent, 13.)
        calls = []; hook = layer.transition.register_forward_pre_hook(lambda *_: calls.append(1)); self.addCleanup(hook.remove)
        packed = torch.cat([v.flatten(1) for v in inputs().values()], 1)
        with patch('app.recurrent.available_memory', return_value=1), self.assertRaisesRegex(GraphError, 'RAM|memory'):
            model(packed)
        self.assertEqual(calls, [])
        unconstrained = compile_model(models, mid, constrained=False)
        with patch('app.recurrent.available_memory', return_value=1): self.assertEqual(list(unconstrained(packed).shape), [3, 10])

    def test_fresh_export_has_no_application_dependency_and_runs_forward_backward(self):
        code = generate_code(*graph_fixture())
        probe = '\n'.join(['assert not any(n == "app" or n.startswith("app.") for n in __import__("sys").modules)',
                           'ports = {g["name"]: torch.zeros(2, *g["shape"]) for g in model.contract["input_groups"]}',
                           'for g in model.contract["input_groups"]:',
                           '    if g["id"] == "lengths": ports[g["name"]].fill_(4)',
                           '    if g["id"] == "mask": ports[g["name"]].fill_(1)',
                           'result=model(ports); sum(v.square().sum() for v in result.values()).backward()',
                           'assert all(p.grad is not None for p in model.parameters()); print("PASS")'])
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([sys.executable, '-c', code + '\n' + probe], cwd=directory, capture_output=True, text=True, check=True)
        self.assertEqual(result.stdout.strip(), 'PASS')


if __name__ == '__main__': unittest.main()
