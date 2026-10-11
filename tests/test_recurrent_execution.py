"""Independent eager recurrence oracles, gradients, fits and portable source."""
import copy
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F

from app.core import GraphError
from app.recurrent import compile_recurrent_transition, generate_recurrent_code
from test_recurrent_contract import fixture

torch.set_num_threads(1)


def oracle(x, initial, weight, bias, lengths=None, mask=None, reset=None):
    """One independent Python recurrence per observation (no runtime helper)."""
    trajectories, final = [], []
    for row in range(x.shape[0]):
        h = initial[row]
        outputs = []
        zero = x[row, :0].sum() + initial[row, :0].sum() + weight[:0].sum() + bias[:0].sum()
        for t in range(x.shape[1]):
            active = (lengths is None or t < int(lengths[row])) and (mask is None or bool(mask[row, t]))
            if active:
                if reset is not None and bool(reset[row, t]): h = initial[row]
                h = torch.tanh(F.linear(torch.cat((x[row, t], h)), weight, bias))
                outputs.append(h)
            else:
                outputs.append(torch.zeros_like(h) + zero)
        trajectories.append(torch.stack(outputs)); final.append(h + zero)
    return torch.stack(trajectories), torch.stack(final)


def two_state_fixture():
    models, mid, spec = fixture()
    graph = models[mid]['graph']
    cloned = copy.deepcopy(graph['nodes'])
    for n in cloned:
        n['id'] += '2'
        if 'label' in n: n['label'] += ' second'
        if n['type'] == 'Concat': n['sources'] = [s + '2' for s in n['sources']]
    graph['nodes'].extend(cloned)
    graph['edges'].extend({'from': e['from'] + '2', 'to': e['to'] + '2'} for e in list(graph['edges']))
    graph['input_order'].extend(['x2', 'h2'])
    spec['sequence_inputs'].append('x2')
    spec['states'].append({'input': 'h2', 'output': 'next2', 'port': 'second'})
    spec['sequence_outputs'].append({'output': 'next2', 'port': 'second_trajectory'})
    return models, mid, spec


class RecurrentExecutionTests(unittest.TestCase):
    def model(self, **kwargs):
        torch.manual_seed(17)
        return compile_recurrent_transition(*fixture(**kwargs), constrained=False)

    def inputs(self, batch=3, steps=5, *, cnn=False, grad=False):
        shape = (1, 4, 5) if cnn else (2,)
        torch.manual_seed(19)
        return torch.randn(batch, steps, *shape, requires_grad=grad), torch.randn(batch, 2, requires_grad=grad)

    def test_runtime_lengths_reuse_exact_parameters_and_no_hidden_cross_call_state(self):
        model = self.model(); pointers = [p.data_ptr() for p in model.parameters()]
        for steps in (1, 7, 3):
            x, h = self.inputs(steps=steps); expected, final = oracle(x, h, *model.parameters())
            a = model({'x': x}, {'hidden': h}); b = model({'x': x}, {'hidden': h})
            torch.testing.assert_close(a['sequences']['trajectory'], expected)
            torch.testing.assert_close(a['state']['hidden'], final)
            torch.testing.assert_close(a['state']['hidden'], b['state']['hidden'], rtol=0, atol=0)
            self.assertEqual(pointers, [p.data_ptr() for p in model.parameters()])
        self.assertEqual(sum(p.numel() for p in model.parameters()), 10)

    def test_lengths_masks_holes_and_active_only_resets_match_per_row_oracle(self):
        model = self.model(); x, h = self.inputs()
        lengths = torch.tensor([0, 3, 5])
        mask = torch.tensor([[True, False, True, True, True], [True, False, True, True, True], [False, True, True, False, True]])
        reset = torch.tensor([[True, True, True, True, True], [False, True, True, True, True], [True, False, False, True, True]])
        for controls in ({}, {'lengths': lengths}, {'mask': mask}, {'reset': reset},
                         {'lengths': lengths, 'mask': mask, 'reset': reset}):
            with self.subTest(controls=list(controls)):
                result = model({'x': x}, {'hidden': h}, **controls)
                expected, state = oracle(x, h, *model.parameters(), **controls)
                torch.testing.assert_close(result['sequences']['trajectory'], expected)
                torch.testing.assert_close(result['state']['hidden'], state)
        self.assertEqual(model({'x': x}, {'hidden': h}, lengths=lengths)['sequences']['trajectory'][0].count_nonzero(), 0)

    def test_real_outputs_input_state_and_parameter_gradients_match_independent_oracle(self):
        controls = {'lengths': torch.tensor([0, 4, 5]),
                    'mask': torch.tensor([[True]*5, [True, False, True, True, True], [False, True, True, False, True]]),
                    'reset': torch.tensor([[False]*5, [False, True, False, False, True], [False, False, True, True, False]])}
        model = self.model(); x, h = self.inputs(grad=True)
        params = list(model.parameters()); independent = [p.detach().clone().requires_grad_() for p in params]
        xi, hi = x.detach().clone().requires_grad_(), h.detach().clone().requires_grad_()
        result = model({'x': x}, {'hidden': h}, **controls)
        y, final = oracle(xi, hi, *independent, **controls)
        actual_loss = result['sequences']['trajectory'].square().sum() + .7 * result['state']['hidden'].square().sum()
        wanted_loss = y.square().sum() + .7 * final.square().sum()
        actual = torch.autograd.grad(actual_loss, [x, h, *params])
        wanted = torch.autograd.grad(wanted_loss, [xi, hi, *independent])
        for a, b in zip(actual, wanted): torch.testing.assert_close(a, b, rtol=3e-6, atol=2e-6)

    def test_nan_padding_is_never_evaluated_and_inactive_rows_have_zero_input_gradient(self):
        model = self.model(); x, h = self.inputs(grad=True)
        lengths = torch.tensor([0, 2, 4])
        with torch.no_grad():
            for row, length in enumerate(lengths): x[row, int(length):] = float('nan')
        sizes = []; handle = model.transition.register_forward_pre_hook(lambda _, args: sizes.append(next(iter(args[0].values())).shape[0]))
        self.addCleanup(handle.remove)
        result = model({'x': x}, {'hidden': h}, lengths=lengths)
        self.assertEqual(sizes, [2, 2, 1, 1])
        self.assertTrue(torch.isfinite(result['sequences']['trajectory']).all())
        loss = result['sequences']['trajectory'].sum() + result['state']['hidden'].sum(); loss.backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        for row, length in enumerate(lengths): self.assertEqual(x.grad[row, int(length):].count_nonzero(), 0)

    def test_all_masked_has_no_cell_calls_preserves_state_and_backward_is_defined(self):
        model = self.model(); x, h = self.inputs(grad=True); x.data.fill_(float('nan'))
        calls = []; handle = model.transition.register_forward_pre_hook(lambda *_: calls.append(1)); self.addCleanup(handle.remove)
        result = model({'x': x}, {'hidden': h}, lengths=torch.zeros(3, dtype=torch.int64), reset=torch.ones(3, 5, dtype=torch.bool))
        self.assertEqual(calls, [])
        torch.testing.assert_close(result['state']['hidden'], h, rtol=0, atol=0)
        self.assertEqual(result['sequences']['trajectory'].count_nonzero(), 0)
        result['sequences']['trajectory'].sum().backward()
        for value in [x, h, *model.parameters()]: self.assertEqual(value.grad.count_nonzero(), 0)

    def test_inputs_states_controls_are_not_modified_in_place(self):
        model = self.model(); x, h = self.inputs(); mask = torch.ones(3, 5, dtype=torch.bool)
        reset = mask.clone(); lengths = torch.tensor([0, 3, 5]); before = [v.clone() for v in (x, h, mask, reset, lengths)]
        model({'x': x}, {'hidden': h}, lengths=lengths, mask=mask, reset=reset)
        for a, b in zip((x, h, mask, reset, lengths), before): torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_chunked_explicit_state_carry_preserves_full_bptt_without_implicit_detach(self):
        model = self.model(); x, h = self.inputs(grad=True)
        whole = model({'x': x}, {'hidden': h})
        first = model({'x': x[:, :2]}, {'hidden': h})
        second = model({'x': x[:, 2:]}, first['state'])
        torch.testing.assert_close(torch.cat([first['sequences']['trajectory'], second['sequences']['trajectory']], 1), whole['sequences']['trajectory'])
        wanted = torch.autograd.grad(whole['state']['hidden'].sum(), [x, h, *model.parameters()], retain_graph=True)
        actual = torch.autograd.grad(second['state']['hidden'].sum(), [x, h, *model.parameters()])
        for a, b in zip(actual, wanted): torch.testing.assert_close(a, b, rtol=2e-6, atol=2e-6)

    def test_two_independent_states_and_input_streams_use_stable_ports(self):
        model = compile_recurrent_transition(*two_state_fixture(), constrained=False)
        x, h = self.inputs(); x2 = x + .6; h2 = h - .4
        result = model({'x2': x2, 'x': x}, {'second': h2, 'hidden': h})
        parameters = list(model.parameters())
        for stream, initial, port, output, pair in ((x, h, 'hidden', 'trajectory', parameters[:2]),
                                                   (x2, h2, 'second', 'second_trajectory', parameters[2:])):
            y, final = oracle(stream, initial, *pair)
            torch.testing.assert_close(result['sequences'][output], y)
            torch.testing.assert_close(result['state'][port], final)
        with self.assertRaisesRegex(ValueError, 'same B and T'):
            model({'x': x, 'x2': x2[:, :3]}, {'hidden': h, 'second': h2})

    def test_explicitly_tied_parameters_inside_transition_remain_tied_across_all_steps(self):
        models, mid, spec = two_state_fixture()
        next(n for n in models[mid]['graph']['nodes'] if n['id'] == 'mix2')['share_with'] = 'mix'
        model = compile_recurrent_transition(models, mid, spec, constrained=False)
        self.assertEqual(model.contract['parameters'], 10); self.assertEqual(len(list(model.parameters())), 2)
        x, h = self.inputs(); result = model({'x': x, 'x2': x + 1}, {'hidden': h, 'second': h - 1})
        a, _ = oracle(x, h, *model.parameters()); b, _ = oracle(x + 1, h - 1, *model.parameters())
        torch.testing.assert_close(result['sequences']['trajectory'], a)
        torch.testing.assert_close(result['sequences']['second_trajectory'], b)

    def test_spatial_state_and_broadcast_reset_preserve_CHW_shape_and_gradients(self):
        models, mid, spec = fixture()
        models[mid]['graph']['nodes'] = [
            {'id': 'x', 'type': 'Input', 'shape': [1, 2, 3], 'axes': ['C', 'H', 'W']},
            {'id': 'h', 'type': 'Input', 'shape': [1, 2, 3], 'axes': ['C', 'H', 'W']},
            {'id': 'join', 'type': 'Concat', 'sources': ['x', 'h'], 'flatten': False},
            {'id': 'mix', 'type': 'Conv2d', 'out_channels': 1, 'kernel_size': [1, 4]},
            {'id': 'tanh', 'type': 'Activation', 'activation': 'Tanh'}, {'id': 'next', 'type': 'Output'}]
        model = compile_recurrent_transition(models, mid, spec, constrained=False)
        x = torch.randn(2, 3, 1, 2, 3, requires_grad=True); h = torch.randn(2, 1, 2, 3, requires_grad=True)
        reset = torch.tensor([[False, True, False], [False, False, True]])
        independent = [p.detach().clone().requires_grad_() for p in model.parameters()]
        xi, hi = x.detach().clone().requires_grad_(), h.detach().clone().requires_grad_()
        ys = []; current = hi
        for t in range(3):
            current = torch.where(reset[:, t, None, None, None], hi, current)
            current = torch.tanh(F.conv2d(torch.cat([xi[:, t], current], -1), *independent))
            ys.append(current)
        expected = torch.stack(ys, 1); result = model({'x': x}, {'hidden': h}, reset=reset)
        torch.testing.assert_close(result['sequences']['trajectory'], expected)
        ga = torch.autograd.grad(result['sequences']['trajectory'].sum(), [x, h, *model.parameters()])
        gb = torch.autograd.grad(expected.sum(), [xi, hi, *independent])
        for a, b in zip(ga, gb): torch.testing.assert_close(a, b)

    def test_nested_module_transition_preserves_parameter_paths_and_export_parity(self):
        from app.core import default_model
        models, mid, spec = fixture(); child = default_model('Projection', inputs=4, outputs=2); child['id'] = 'projection'
        models[child['id']] = child
        mix = models[mid]['graph']['nodes'][3]; mix.pop('features'); mix.pop('bias')
        mix.update(type='Module', ref=child['id'])
        model = compile_recurrent_transition(models, mid, spec, constrained=False)
        self.assertEqual(set(model.state_dict()), set(model.contract['parameter_schema']))
        namespace = {}; exec(generate_recurrent_code(models, mid, spec, constrained=False), namespace)
        exported = namespace['model']; exported.load_state_dict(model.state_dict(), strict=True)
        x, h = self.inputs()
        torch.testing.assert_close(exported({'x': x}, {'hidden': h})['state']['hidden'], model({'x': x}, {'hidden': h})['state']['hidden'], rtol=0, atol=0)

    def test_cnn_per_frame_and_recurrent_mlp_compose_with_independent_outputs_and_gradients(self):
        model = self.model(cnn=True); frames, initial = self.inputs(cnn=True, grad=True)
        params = list(model.parameters()); copied = [p.detach().clone().requires_grad_() for p in params]
        f2, h2 = frames.detach().clone().requires_grad_(), initial.detach().clone().requires_grad_()
        encoded = F.adaptive_avg_pool2d(F.conv2d(f2.flatten(0, 1), copied[0], copied[1]), (1, 1)).reshape(3, 5, 2)
        y, final = oracle(encoded, h2, *copied[2:])
        actual = model({'frame': frames}, {'hidden': initial})
        torch.testing.assert_close(actual['sequences']['trajectory'], y)
        torch.testing.assert_close(actual['state']['hidden'], final)
        a = torch.autograd.grad(actual['sequences']['trajectory'].square().sum(), [frames, initial, *params])
        b = torch.autograd.grad(y.square().sum(), [f2, h2, *copied])
        for left, right in zip(a, b): torch.testing.assert_close(left, right, rtol=5e-6, atol=3e-6)

    def test_actual_bounded_adam_fit_learns_sequence_targets_and_matches_functional_training(self):
        model = self.model(); x, h = self.inputs()
        teacher_weight = torch.tensor([[.5, -.2, .35, .1], [-.1, .4, .15, -.3]])
        teacher_bias = torch.tensor([.15, -.2]); target, _ = oracle(x, h, teacher_weight, teacher_bias)
        params = [p.detach().clone().requires_grad_() for p in model.parameters()]
        opt = torch.optim.Adam(model.parameters(), lr=.04, foreach=False, fused=False)
        other = torch.optim.Adam(params, lr=.04, foreach=False, fused=False)
        losses = []
        for _ in range(40):
            opt.zero_grad(); other.zero_grad()
            y = model({'x': x}, {'hidden': h})['sequences']['trajectory']; wanted, _ = oracle(x, h, *params)
            loss, independent = F.mse_loss(y, target), F.mse_loss(wanted, target)
            torch.testing.assert_close(loss, independent, rtol=1e-5, atol=2e-6)
            loss.backward(); independent.backward(); opt.step(); other.step(); losses.append(float(loss.detach()))
        self.assertLess(losses[-1], losses[0] * .03)
        for left, right in zip(model.parameters(), params): torch.testing.assert_close(left, right, rtol=2e-5, atol=3e-6)

    def test_real_cnn_recurrent_fit_reduces_loss(self):
        model = self.model(cnn=True); x, h = self.inputs(batch=4, steps=3, cnn=True)
        teacher = self.model(cnn=True)
        with torch.no_grad():
            for p in teacher.parameters(): p.mul_(.35)
            target = teacher({'frame': x}, {'hidden': h})['sequences']['trajectory']
        opt = torch.optim.Adam(model.parameters(), lr=.03, foreach=False, fused=False); losses = []
        for _ in range(30):
            opt.zero_grad(); loss = F.mse_loss(model({'frame': x}, {'hidden': h})['sequences']['trajectory'], target)
            loss.backward(); opt.step(); losses.append(float(loss.detach()))
        self.assertLess(losses[-1], losses[0] * .08)

    def test_bad_inputs_and_controls_fail_before_transition_execution(self):
        model = self.model(); x, h = self.inputs(); calls = []
        handle = model.transition.register_forward_pre_hook(lambda *_: calls.append(1)); self.addCleanup(handle.remove)
        for sequences, states, controls in (
            ({'observation': x}, {'hidden': h}, {}), ({'x': x.double()}, {'hidden': h}, {}),
            ({'x': x[:, :0]}, {'hidden': h}, {}), ({'x': x[:, :, :1]}, {'hidden': h}, {}),
            ({'x': x}, {'h': h}, {}), ({'x': x}, {'hidden': h[:2]}, {}),
            ({'x': x}, {'hidden': h}, {'lengths': torch.tensor([0., 1., 2.])}),
            ({'x': x}, {'hidden': h}, {'lengths': torch.tensor([0, 1, 6])}),
            ({'x': x}, {'hidden': h}, {'lengths': torch.tensor([0, -1, 2])}),
            ({'x': x}, {'hidden': h}, {'lengths': torch.tensor([[0], [1], [2]])}),
            ({'x': x}, {'hidden': h}, {'mask': torch.ones(3, 5)}),
            ({'x': x}, {'hidden': h}, {'mask': torch.ones(3, 1, dtype=torch.bool)}),
            ({'x': x}, {'hidden': h}, {'reset': True})):
            with self.subTest(controls=controls.keys()), self.assertRaises(ValueError): model(sequences, states, **controls)
        self.assertEqual(calls, [])

    def test_ram_refusal_precedes_weight_allocation_and_each_unroll(self):
        with patch('app.recurrent.available_memory', return_value=0), patch('torch.nn.Linear') as constructor:
            with self.assertRaisesRegex(GraphError, 'RAM'): compile_recurrent_transition(*fixture())
            constructor.assert_not_called()
        model = self.model(); model.constrained = True; x, h = self.inputs()
        with patch('app.recurrent.available_memory', return_value=0), patch('app.recurrent._recurrent_run') as run:
            with self.assertRaisesRegex(ValueError, 'RAM'): model({'x': x}, {'hidden': h})
            run.assert_not_called()

    def test_changed_parameter_dtype_is_rejected_before_transition_execution(self):
        model = self.model().double(); x, h = self.inputs(); calls = []
        handle = model.transition.register_forward_pre_hook(lambda *_: calls.append(1)); self.addCleanup(handle.remove)
        with self.assertRaisesRegex(ValueError, 'parameters.*CPU float32'): model({'x': x}, {'hidden': h})
        self.assertEqual(calls, [])

    def test_compilation_is_a_snapshot_not_a_reference_to_mutable_graph_or_interface(self):
        models, mid, spec = fixture(); model = compile_recurrent_transition(models, mid, spec, constrained=False)
        x, h = self.inputs(); expected = model({'x': x}, {'hidden': h})
        spec['states'][0]['port'] = 'changed'; models[mid]['graph']['nodes'][4]['activation'] = 'ReLU'
        torch.testing.assert_close(model({'x': x}, {'hidden': h})['sequences']['trajectory'], expected['sequences']['trajectory'], rtol=0, atol=0)

    def test_compile_and_export_capture_only_runnable_closure_not_data_results_or_unrelated_models(self):
        class NotGraphPayload:
            def __deepcopy__(self, memo): raise AssertionError('Must not copy unrelated data/result payloads.')
            def __repr__(self): raise AssertionError('Must not export unrelated data/result payloads.')
        models, mid, spec = fixture()
        models[mid].update(dataset=NotGraphPayload(), result=NotGraphPayload(), training=NotGraphPayload())
        models['unrelated'] = {'id': 'unrelated', 'name': 'Broken unrelated draft', 'graph': NotGraphPayload()}
        model = compile_recurrent_transition(models, mid, spec, constrained=False)
        namespace = {}; exec(generate_recurrent_code(models, mid, spec, constrained=False), namespace)
        namespace['model'].load_state_dict(model.state_dict(), strict=True)
        x, h = self.inputs()
        torch.testing.assert_close(model({'x': x}, {'hidden': h})['state']['hidden'], namespace['model']({'x': x}, {'hidden': h})['state']['hidden'], rtol=0, atol=0)

    def test_only_final_state_and_explicit_sequence_aliases(self):
        models, mid, spec = fixture(); spec['sequence_outputs'] = []
        model = compile_recurrent_transition(models, mid, spec, constrained=False); x, h = self.inputs()
        self.assertEqual(model({'x': x}, {'hidden': h})['sequences'], {})
        spec['sequence_outputs'] = [{'output': 'next', 'port': 'a'}, {'output': 'next', 'port': 'b'}]
        model = compile_recurrent_transition(models, mid, spec, constrained=False); result = model({'x': x}, {'hidden': h})
        torch.testing.assert_close(result['sequences']['a'], result['sequences']['b'], rtol=0, atol=0)

    def test_app_free_export_matches_native_control_semantics_and_gradients(self):
        models, mid, spec = fixture(cnn=True); native = compile_recurrent_transition(models, mid, spec, constrained=False)
        source = generate_recurrent_code(models, mid, spec, constrained=False); namespace = {}; exec(source, namespace)
        exported = namespace['model']; exported.load_state_dict(native.state_dict(), strict=True)
        x, h = self.inputs(cnn=True, grad=True); xi, hi = x.detach().clone().requires_grad_(), h.detach().clone().requires_grad_()
        controls = {'lengths': torch.tensor([0, 3, 5]), 'reset': torch.ones(3, 5, dtype=torch.bool)}
        a, b = native({'frame': x}, {'hidden': h}, **controls), exported({'frame': xi}, {'hidden': hi}, **controls)
        torch.testing.assert_close(a['sequences']['trajectory'], b['sequences']['trajectory'], rtol=0, atol=0)
        ga = torch.autograd.grad(a['sequences']['trajectory'].sum(), [x, h, *native.parameters()])
        gb = torch.autograd.grad(b['sequences']['trajectory'].sum(), [xi, hi, *exported.parameters()])
        for left, right in zip(ga, gb): torch.testing.assert_close(left, right, rtol=0, atol=0)

    def test_standalone_source_runs_new_T_and_training_in_a_fresh_process_without_app(self):
        source = generate_recurrent_code(*fixture(), constrained=False)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'recurrent.py'; path.write_text(source, encoding='utf-8')
            code = ('import runpy,sys,torch; torch.set_num_threads(1); '
                    'm=runpy.run_path(sys.argv[1])["model"]; ids=[id(p) for p in m.parameters()]; '
                    'x=torch.ones(2,7,2); h=torch.zeros(2,2); '
                    'r=m({"x":x},{"hidden":h},lengths=torch.tensor([0,7])); '
                    'assert r["sequences"]["trajectory"].shape==(2,7,2); '
                    'opt=torch.optim.Adam(m.parameters(),lr=.01); '
                    'opt.zero_grad(); r["sequences"]["trajectory"].square().mean().backward(); opt.step(); '
                    'm({"x":x[:,:2]},{"hidden":h}); '
                    'assert ids==[id(p) for p in m.parameters()]; '
                    'assert not any(n=="app" or n.startswith("app.") for n in sys.modules); '
                    'print("standalone-recurrence-pass")')
            environment = dict(os.environ, PYTHONPATH='')
            result = subprocess.run([sys.executable, '-c', code, str(path)], cwd=directory, env=environment,
                                    text=True, capture_output=True, check=True, timeout=30)
            self.assertEqual(result.stdout.strip(), 'standalone-recurrence-pass')


if __name__ == '__main__': unittest.main()
