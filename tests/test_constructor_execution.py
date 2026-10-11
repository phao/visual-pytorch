"""Independent functional oracles plus real typed-family worker/kernel execution."""
import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F
from app.core import compile_model, generate_code
from app.families import specialize
from app.experiment_objects import candidate_bundle, execute_candidate, object_reference, preview_experiment
from app.revisions import validate_all
from test_constructor_choices import choices, choice_experiment

torch.set_num_threads(1)


class ConstructorExecutionTests(unittest.TestCase):
    def workspace(self):
        w, e = choice_experiment(); self.addCleanup(w.db.close)
        self.addCleanup(lambda: w.history.directory.cleanup() if w.history.directory else None)
        return w, e

    def test_all_activation_bias_choices_match_outputs_input_parameter_gradients_and_export(self):
        operations = {'ReLU': F.relu, 'GELU': F.gelu, 'Sigmoid': torch.sigmoid,
                      'Tanh': torch.tanh, 'Softmax': lambda v: F.softmax(v, dim=-1)}
        for activation, operation in operations.items():
            for bias in (False, True):
                resolved = specialize(choices(), {'activation-kind': activation, 'learn-bias': bias})
                net = compile_model(resolved['models'], resolved['root'], constrained=False).double()
                self.assertEqual(sum(p.numel() for p in net.parameters()), 10 if bias else 6)
                layers = [module for module in net.modules() if isinstance(module, torch.nn.Linear)]
                self.assertEqual(len(layers), 2); self.assertTrue(all((l.bias is not None) == bias for l in layers))
                leaves = [v.detach().clone().requires_grad_() for v in net.parameters()]
                for batch in (1, 4):
                    x = torch.linspace(-.8, .9, batch, dtype=torch.float64).reshape(batch, 1).requires_grad_()
                    xx = x.detach().clone().requires_grad_()
                    w1, b1, w2, b2 = leaves if bias else (leaves[0], None, leaves[1], None)
                    expected = F.linear(operation(F.linear(xx, w1, b1)), w2, b2)
                    actual = net(x); torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
                    observed = torch.autograd.grad(actual.square().sum(), [x, *net.parameters()])
                    oracle = torch.autograd.grad(expected.square().sum(), [xx, *leaves])
                    for a, b in zip(observed, oracle): torch.testing.assert_close(a, b, rtol=1e-11, atol=1e-11)
                    scope = {}; exec(generate_code(resolved['models'], resolved['root']), scope)
                    exported = scope['model'].double(); exported.load_state_dict(net.state_dict(), strict=True)
                    torch.testing.assert_close(exported(x), expected, rtol=1e-12, atol=1e-12)

    def test_convolution_and_pool_boolean_constructor_fields_are_actual_computation(self):
        nodes = [{'id': 'input', 'type': 'Input', 'shape': [1, 3, 3]},
            {'id': 'conv', 'type': 'Conv2d', 'out_channels': 2, 'kernel_size': 1, 'bias': True},
            {'id': 'pool', 'type': 'AvgPool2d', 'kernel_size': 3, 'stride': 1, 'padding': 1, 'count_include_pad': True},
            {'id': 'flat', 'type': 'Flatten'}, {'id': 'head', 'type': 'Linear', 'features': 1, 'bias': False},
            {'id': 'output', 'type': 'Output'}]
        model = {'id': 'conv', 'name': 'Constructor convolution', 'graph': {'nodes': nodes,
            'edges': [{'from': a['id'], 'to': b['id']} for a, b in zip(nodes, nodes[1:])]}}
        source = {'version': 3, 'id': 'pool-family', 'name': 'Pooling choices', 'root': 'conv', 'models': {'conv': model},
            'symbols': [{'id': 'boolean-choice', 'scope': 'conv', 'name': 'include', 'type': 'boolean', 'default': True}],
            'assignments': [{'model': 'conv', 'node': 'conv', 'field': 'bias', 'value': {'symbol': 'boolean-choice'}},
                {'model': 'conv', 'node': 'pool', 'field': 'count_include_pad', 'value': {'symbol': 'boolean-choice'}}], 'repeats': []}
        for bias in (False, True):
            resolved = specialize(source, {'boolean-choice': bias}); net = compile_model(resolved['models'], resolved['root'], constrained=False).double()
            leaves = [p.detach().clone().requires_grad_() for p in net.parameters()]
            x = torch.linspace(-.7, .8, 18, dtype=torch.float64).reshape(2, 1, 3, 3).requires_grad_(); xx = x.detach().clone().requires_grad_()
            conv = F.conv2d(xx, leaves[0], leaves[1] if bias else None)
            pool = F.avg_pool2d(conv, kernel_size=3, stride=1, padding=1, count_include_pad=bias)
            expected = F.linear(pool.flatten(1), leaves[2] if bias else leaves[1])
            actual = net(x); torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
            for a, b in zip(torch.autograd.grad(actual.square().sum(), [x, *net.parameters()]),
                            torch.autograd.grad(expected.square().sum(), [xx, *leaves])):
                torch.testing.assert_close(a, b, rtol=1e-11, atol=1e-11)

    def test_real_typed_worker_retry_checkpoint_and_fresh_predictor(self):
        from app.task_execution import restore_task_predictor, check_task_checkpoint
        from app.task_result import validate_result
        from app.run_export import predictor_source, write_new_predictor
        w, e = self.workspace(); ref = object_reference('experiment', e); rows = preview_experiment(e, w.get_spec)['candidates']
        for i in (0, 12, 20):
            result = execute_candidate(w, ref, rows[i]['coordinates'], 'typed-' + str(i), 'Typed choice', python=sys.executable, timeout=20)
            record = result['record']; self.assertEqual(record['version'], 3)
            attempt = record['attempts'][0]; self.assertEqual(attempt['status'], 'completed', attempt['error'])
            self.assertEqual(attempt['result']['counts']['completed_updates'], 3)
            bundle = candidate_bundle(e, rows[i]['coordinates'], w.get_spec); validate_result(bundle, attempt['result'])
            checkpoint = attempt['result']['checkpoint']; check_task_checkpoint(bundle, checkpoint, constrained=False)
            spec_ref = record['resolved']['references']['specialization']; spec = w.get_spec('specialization', spec_ref['id'], spec_ref['revision'])
            self.assertEqual(spec['version'], 2); self.assertIs(type(spec['bindings']['learn-bias']), bool)
            self.assertIs(type(spec['bindings']['activation-kind']), str)
            retry = execute_candidate(w, ref, rows[i]['coordinates'], record['id'], 'Retry typed', retry=True, python=sys.executable, timeout=20)
            self.assertEqual(retry['record']['attempts'][1]['result']['checkpoint'], checkpoint)
            records = [{'x': [.2]}, {'x': [-.7]}]
            expected = {k: v.tolist() for k, v in restore_task_predictor(bundle, checkpoint, constrained=False)(records).items()}
            with tempfile.TemporaryDirectory() as td:
                write_new_predictor(predictor_source(w, result['reference']), Path(td) / 'predictor.py')
                code = 'import predictor,json,sys;assert "app" not in sys.modules;print(json.dumps({k:v.tolist() for k,v in predictor.predict_records('+repr(records)+').items()}))'
                ran = subprocess.run([sys.executable, '-c', code], cwd=td, text=True, capture_output=True, timeout=20)
                self.assertEqual(ran.returncode, 0, ran.stderr); self.assertEqual(json.loads(ran.stdout), expected)
        validate_all(w.db)

    def test_actual_typed_family_jupyter_kernel_sources_curves_and_predictors(self):
        from app.notebooks import create_bundle, execute_bundle, verify_bundle
        w, e = self.workspace(); e.pop('revision'); e['models'][0]['axes'] = [
            {'symbol': 'activation-kind', 'values': ['GELU', 'Softmax']}]
        e['models'][0]['bindings'] = {'learn-bias': False}; e['data_tasks'] = e['data_tasks'][:1]; e['initialization_seeds'] = [7]
        ref = w.put_spec('experiment', e)
        recipe = w.put_spec('report', {'version': 1, 'id': 'constructor-rerun', 'name': 'Constructor choices', 'question': 'Do typed choices preserve meaning?',
            'mode': 'rerun', 'experiment': ref, 'runs': [], 'sections': ['models', 'training', 'evaluation', 'checkpoints', 'failures'],
            'analysis': {'kind': 'descriptive', 'inferential_unit': 'run'}})
        with tempfile.TemporaryDirectory() as td:
            before = w.db.serialize(); export = create_bundle(w, recipe, Path(td) / 'bundle'); self.assertEqual(before, w.db.serialize())
            result = execute_bundle(Path(export['path']), timeout=60)
            self.assertEqual(result['execution']['status'], 'completed', result['execution']['error'])
            output = Path(result['path']); view = json.loads((output / 'report-data.json').read_text())
            self.assertEqual(len(view['runs']), 2); self.assertTrue(all(r['status'] == 'completed' for r in view['runs']))
            self.assertEqual(len(list(output.glob('architecture-*.py'))), 2); self.assertEqual(len(list(output.glob('predictor-*.py'))), 2)
            self.assertEqual(len(list(output.glob('curve-*.svg'))), 2); verify_bundle(Path(export['path'])); self.assertEqual(before, w.db.serialize())


if __name__ == '__main__': unittest.main()
