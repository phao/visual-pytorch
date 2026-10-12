"""LSTM gates, projection, direction/state order and gradients: independent oracle.

The oracle never calls nn.LSTM, LSTMCell, packed-sequence helpers or application
execution helpers. Rows and directions are scanned explicitly with gate equations.
"""
import copy
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F

from app.core import GraphError, Workspace, compile_model, generate_code, validate_model
from app.evidence import parameter_schema
from app.lstm_nodes import lstm_memory_report
from examples.projected_lstm import models as fixture

torch.set_num_threads(1)


def oracle(x, initial_h, initial_c, weights, *, lengths, layers, bidirectional, projection):
    """Independent scalar-time scan of i,f,g,o gates; row-specific reverse order."""
    directions = 2 if bidirectional else 1
    batch, steps = x.shape[:2]
    anchor = sum((p.flatten()[:0].sum() for p in weights.values()), x.flatten()[:0].sum())
    anchor = anchor + initial_h.flatten()[:0].sum() + initial_c.flatten()[:0].sum()
    row_outputs, row_hidden, row_cell = [], [], []
    for row in range(batch):
        length = int(lengths[row])
        frames = x[row, :length]
        end_h, end_c = [], []
        for layer in range(layers):
            streams = []
            for direction in range(directions):
                suffix = '_l' + str(layer) + ('_reverse' if direction else '')
                h = initial_h[row, layer * directions + direction]
                c = initial_c[row, layer * directions + direction]
                values = {}
                times = range(length) if direction == 0 else range(length - 1, -1, -1)
                for time in times:
                    gates = F.linear(frames[time], weights['weight_ih' + suffix], weights.get('bias_ih' + suffix))
                    gates = gates + F.linear(h, weights['weight_hh' + suffix], weights.get('bias_hh' + suffix))
                    i, f, g, o = gates.chunk(4)
                    c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
                    h = torch.sigmoid(o) * torch.tanh(c)
                    if projection:
                        h = F.linear(h, weights['weight_hr' + suffix])
                    values[time] = h
                end_h.append(h + anchor); end_c.append(c + anchor)
                if length:
                    streams.append(torch.stack([values[time] for time in range(length)]))
            width = directions * initial_h.shape[-1]
            frames = torch.cat(streams, -1) if length else x.new_zeros((0, width)) + anchor
        padding = x.new_zeros(steps - length, directions * initial_h.shape[-1]) + anchor
        row_outputs.append(torch.cat([frames, padding], 0))
        row_hidden.append(torch.stack(end_h)); row_cell.append(torch.stack(end_c))
    return dict(sequence=torch.stack(row_outputs), hidden=torch.stack(row_hidden), cell=torch.stack(row_cell))


def data(contract, *, grad=False, lengths=None, nan_padding=False):
    torch.manual_seed(31)
    result = {g['id']: torch.randn(4, *g['shape'], requires_grad=grad) for g in contract['input_groups']}
    steps = contract['shapes']['lstm']['lstm']['steps']
    lengths = lengths if lengths is not None else [0, steps, 1, max(1, steps - 1)]
    if 'lengths' in result:
        result['lengths'] = torch.tensor(lengths, dtype=torch.float32).reshape(4, 1)
    if nan_padding:
        with torch.no_grad():
            for row, length in enumerate(lengths): result['sequence'][row, int(length):] = float('nan')
    return result


def named(contract, values):
    return {g['name']: values[g['id']] for g in contract['input_groups']}


def returned(contract, result):
    return {port: result[next(g['name'] for g in contract['output_groups'] if g['id'] == 'out_' + port)]
            for port in ('sequence', 'hidden', 'cell')}


class LSTMExecutionTests(unittest.TestCase):
    def pair(self, **options):
        models, mid = fixture(**options)
        torch.manual_seed(19)
        native = compile_model(models, mid, constrained=False)
        namespace = {}; exec(generate_code(models, mid), namespace)
        exported = namespace['model']; exported.load_state_dict(native.state_dict(), strict=True)
        self.assertEqual(set(native.state_dict()), set(parameter_schema(models, mid)))
        self.assertEqual(sum(p.numel() for p in native.parameters()), validate_model(models, mid)['parameters'])
        return native, exported

    def check_oracle(self, model, *, nan_padding=False):
        plan = model.contract['shapes']['lstm']['lstm']; kw = plan['kwargs']
        values = data(model.contract, grad=True, nan_padding=nan_padding)
        keys = ('sequence', 'hidden', 'cell')
        independent = {key: values[key].detach().clone().requires_grad_() for key in keys}
        layer = model.layer_0
        weights = {key: value.detach().clone().requires_grad_() for key, value in layer.named_parameters()}
        lengths = values['lengths'][:, 0].long() if 'lengths' in values else torch.full((4,), plan['steps'])
        expected = oracle(*(independent[key] for key in keys), weights, lengths=lengths,
                          layers=kw['num_layers'], bidirectional=kw['bidirectional'], projection=kw['proj_size'])
        actual = returned(model.contract, model(named(model.contract, values)))
        for key in keys: torch.testing.assert_close(actual[key], expected[key], rtol=3e-6, atol=2e-6)
        loss = lambda outputs: sum((index + .5) * outputs[key].square().sum() for index, key in enumerate(keys))
        aa = torch.autograd.grad(loss(actual), [*(values[key] for key in keys), *layer.parameters()])
        bb = torch.autograd.grad(loss(expected), [*(independent[key] for key in keys), *weights.values()])
        for a, b in zip(aa, bb): torch.testing.assert_close(a, b, rtol=8e-6, atol=3e-6)

    def test_all_gates_states_and_gradients_match_independent_scan(self):
        for layers in (1, 2):
            for bidirectional in (False, True):
                for projection in (0, 2):
                    for bias in (False, True):
                        for model in self.pair(steps=4, width=2, hidden=3, projection=projection,
                                               layers=layers, bidirectional=bidirectional, bias=bias):
                            with self.subTest(layers=layers, bidirectional=bidirectional, projection=projection,
                                              bias=bias, model=type(model).__name__):
                                self.check_oracle(model, nan_padding=True)

    def test_full_length_without_control_and_single_time_step(self):
        for steps in (1, 5):
            for model in self.pair(steps=steps, lengths=False): self.check_oracle(model)

    def test_reverse_final_hidden_is_not_last_reverse_sequence(self):
        for model in self.pair(layers=1):
            values = data(model.contract, lengths=[5] * 4)
            out = returned(model.contract, model(named(model.contract, values)))
            torch.testing.assert_close(out['hidden'][:, 0], out['sequence'][:, -1, :4])
            torch.testing.assert_close(out['hidden'][:, 1], out['sequence'][:, 0, 4:])
            self.assertGreater((out['hidden'][:, 1] - out['sequence'][:, -1, 4:]).abs().max().item(), 1e-4)

    def test_all_zero_lengths_return_supplied_state_with_exact_zero_gradients(self):
        for model in self.pair():
            values = data(model.contract, grad=True, lengths=[0] * 4, nan_padding=True)
            out = returned(model.contract, model(named(model.contract, values)))
            self.assertEqual(out['sequence'].count_nonzero(), 0)
            for role in ('hidden', 'cell'): torch.testing.assert_close(out[role], values[role], rtol=0, atol=0)
            out['sequence'].sum().backward(retain_graph=True)
            for value in [values[k] for k in ('sequence', 'hidden', 'cell')] + list(model.parameters()):
                self.assertIsNotNone(value.grad); self.assertEqual(value.grad.count_nonzero(), 0)
            gradients = torch.autograd.grad(out['hidden'].sum() + out['cell'].sum(),
                                            [values['hidden'], values['cell']])
            for gradient in gradients: torch.testing.assert_close(gradient, torch.ones_like(gradient), rtol=0, atol=0)

    def test_packed_and_named_boundary_exact_and_no_hidden_between_call_state(self):
        for model in self.pair():
            values = named(model.contract, data(model.contract))
            packed = torch.cat([values[g['name']].flatten(1) for g in model.contract['input_groups']], 1)
            first, second, third = model(values), model(packed), model(values)
            for key in first:
                torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)
                torch.testing.assert_close(first[key], third[key], rtol=0, atol=0)
            expected = torch.cat([first[g['name']].flatten(1) for g in model.contract['output_groups']], 1)
            torch.testing.assert_close(model.forward_tensor(values), expected, rtol=0, atol=0)

    def test_explicit_lengths_and_active_state_refusals(self):
        for model in self.pair():
            for role, invalid in [('lengths', -1.), ('lengths', .5), ('lengths', 6.), ('lengths', float('inf')),
                                  ('hidden', float('nan')), ('cell', float('inf')), ('sequence', float('nan'))]:
                values = data(model.contract); values[role].fill_(invalid)
                with self.subTest(role=role, value=invalid), self.assertRaises((GraphError, ValueError)):
                    model(named(model.contract, values))

    def test_dtype_state_shape_and_batch_refusals(self):
        for model in self.pair():
            for role in ('sequence', 'hidden', 'cell', 'lengths'):
                values = data(model.contract); values[role] = values[role].double()
                with self.subTest(role=role), self.assertRaises((GraphError, ValueError)):
                    model(named(model.contract, values))
            # B=LD=4: dimensions alone cannot expose native orientation. Values
            # still undergo the documented graph adapter; use unequal B here.
            values = {k: v[:3] for k, v in data(model.contract).items()}
            values['hidden'] = values['hidden'].transpose(0, 1)
            with self.assertRaises((GraphError, ValueError)): model(named(model.contract, values))

    def test_memory_refuses_before_native_lstm_and_scales_with_actual_batch(self):
        native, _ = self.pair(); plan = native.contract['shapes']['lstm']['lstm']
        with patch('app.lstm_nodes.available_memory', return_value=10**10):
            a = lstm_memory_report(plan, 1); b = lstm_memory_report(plan, 12)
            self.assertTrue(a['allowed']); self.assertGreater(b['estimated_bytes'], a['estimated_bytes'])
        native.layer_0.constrained = True
        with patch('app.lstm_nodes.available_memory', return_value=1), patch.object(native.layer_0, 'forward', side_effect=AssertionError('No run')):
            with self.assertRaisesRegex(GraphError, 'Constrained LSTM'): native(named(native.contract, data(native.contract)))

    def test_constructor_ignores_process_default_dtype_and_device(self):
        old_dtype, old_device = torch.get_default_dtype(), torch.get_default_device()
        try:
            torch.set_default_dtype(torch.float64); torch.set_default_device('meta')
            for model in self.pair():
                self.assertTrue(all(p.dtype == torch.float32 and p.device.type == 'cpu' for p in model.parameters()))
        finally:
            torch.set_default_dtype(old_dtype); torch.set_default_device(old_device)

    def test_workspace_freeze_reload_and_trained_app_free_checkpoint(self):
        models, mid = fixture(steps=3, width=1, hidden=3, projection=2, layers=1)
        student = compile_model(models, mid, constrained=False); teacher = compile_model(models, mid, constrained=False)
        values = named(student.contract, data(student.contract))
        with torch.no_grad(): targets = teacher(values)
        optimizer = torch.optim.Adam(student.parameters(), lr=.025)
        def loss(): return sum(F.mse_loss(v, targets[k]) for k, v in student(values).items())
        before = loss().item()
        for _ in range(70): optimizer.zero_grad(); current = loss(); current.backward(); optimizer.step()
        self.assertLess(loss().item(), before * .15)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); w = Workspace(); w.put(models[mid]); w.freeze_model(mid); w.save(root / 'model.vtw'); w.db.close()
            opened = Workspace.load(root / 'model.vtw')
            try: source = generate_code(opened.models(), mid)
            finally: opened.db.close()
            (root / 'predict.py').write_text(source + '\nimport sys\npayload=torch.load(sys.argv[1], weights_only=True)\nmodel.load_state_dict(payload["weights"],strict=True)\ntorch.save(model(payload["inputs"]),sys.argv[2])\n')
            torch.save({'weights': student.state_dict(), 'inputs': values}, root / 'payload.pt')
            result = subprocess.run([sys.executable, str(root / 'predict.py'), str(root / 'payload.pt'), str(root / 'result.pt')],
                                    cwd=root, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            reloaded = torch.load(root / 'result.pt', weights_only=True)
            for key, expected in student(values).items(): torch.testing.assert_close(reloaded[key], expected, rtol=0, atol=0)

    def test_tied_lstm_parameters_count_update_once_and_calls_both_contribute(self):
        models, mid = fixture(steps=3, hidden=3, projection=2, layers=1)
        graph = models[mid]['graph']; other = copy.deepcopy(graph['nodes'][3]); other.update(id='shared', share_with='lstm')
        graph['nodes'].append(other)
        graph['edges'] += [{'from': key, 'to': 'shared'} for key in ('sequence', 'hidden', 'cell', 'lengths')]
        for port in ('sequence', 'hidden', 'cell'):
            graph['nodes'].append(dict(id='shared_' + port, type='Output', label='Shared ' + port))
            graph['edges'].append(dict(from_port=port, **{'from': 'shared', 'to': 'shared_' + port}))
        contract = validate_model(models, mid); original_count = validate_model(*fixture(steps=3, hidden=3, projection=2, layers=1))['parameters']
        self.assertEqual(contract['parameters'], original_count)
        native = compile_model(models, mid, constrained=False); scope = {}; exec(generate_code(models, mid), scope)
        for model in (native, scope['model']):
            self.assertEqual(sum(p.numel() for p in model.parameters()), original_count)
            self.assertFalse(any('layer_1.' in key for key in model.state_dict()))
            values = named(contract, data(contract)); result = model(values)
            first = result['Returned sequence']; second = result['Shared sequence']
            torch.testing.assert_close(first, second, rtol=0, atol=0)
            params = list(model.parameters()); gradients = torch.autograd.grad(first.sum() + second.sum(), params, retain_graph=True)
            single = torch.autograd.grad(first.sum(), params)
            for both, one in zip(gradients, single): torch.testing.assert_close(both, 2 * one)
            previous = [p.detach().clone() for p in params]
            optimizer = torch.optim.SGD(params, lr=.01); optimizer.zero_grad()
            sum(v.sum() for v in model(values).values()).backward()
            applied = [p.grad.detach().clone() for p in params]; optimizer.step()
            for p, old, grad in zip(params, previous, applied): torch.testing.assert_close(p, old - .01 * grad)

    def test_module_wrapper_keeps_lstm_adapters_and_named_port_packing(self):
        from app.core import default_model
        models, child = fixture(steps=3)
        contract = validate_model(models, child)
        parent = default_model('Wrapper', contract['inputs'], contract['outputs']); parent['id'] = 'parent'
        parent['graph']['nodes'][1] = dict(id='linear1', type='Module', ref=child)
        models['parent'] = parent
        native = compile_model(models, 'parent', constrained=False); scope = {}; exec(generate_code(models, 'parent'), scope)
        exported = scope['model']; exported.load_state_dict(native.state_dict(), strict=True)
        values = data(contract); packed = torch.cat([values[g['id']].flatten(1) for g in contract['input_groups']], 1)
        expected = native.layer_0.forward_tensor(packed)
        torch.testing.assert_close(native(packed), expected, rtol=0, atol=0)
        torch.testing.assert_close(exported(packed), expected, rtol=0, atol=0)

    def test_lstm_and_generic_recurrent_export_helpers_coexist(self):
        from test_recurrent_graph_contract import graph_fixture
        models, mid = graph_fixture(steps=3, controls=False)
        extra, eid = fixture(steps=3, hidden=3, projection=2, layers=1, lengths=False)
        additions = extra[eid]['graph']
        for n in additions['nodes']:
            n['id'] = 'lstm_' + n['id']
            if n['type'] == 'LSTM': n['inputs'] = {key: 'lstm_' + value for key, value in n['inputs'].items()}
        for edge in additions['edges']:
            edge['from'] = 'lstm_' + edge['from']; edge['to'] = 'lstm_' + edge['to']
        models[mid]['graph']['nodes'].extend(additions['nodes']); models[mid]['graph']['edges'].extend(additions['edges'])
        models[mid]['graph']['input_order'].extend('lstm_' + key for key in additions['input_order'])
        native = compile_model(models, mid, constrained=False); scope = {}; exec(generate_code(models, mid), scope)
        exported = scope['model']; exported.load_state_dict(native.state_dict(), strict=True)
        torch.manual_seed(27)
        values = {g['name']: torch.randn(2, *g['shape']) for g in native.contract['input_groups']}
        expected, actual = native(values), exported(values)
        for key in expected: torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)


if __name__ == '__main__': unittest.main()
