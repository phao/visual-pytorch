"""Independent sharing oracles, actual workers, checkpoint/export/kernel checks."""
import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F
from app.core import compile_model, generate_code, validate_model, affine_summary
from app.evidence import parameter_schema
from app.experiment_objects import candidate_bundle, execute_candidate, object_reference, preview_experiment
from app.revisions import validate_all
from test_shared_parameters import serial, tied_family
from test_family_axes import family_experiment

torch.set_num_threads(1)


def exported(models, mid, network):
    scope = {}; exec(generate_code(models, mid), scope)
    result = scope['model'].double(); result.load_state_dict(network.state_dict(), strict=True)
    return result


class SharedExecutionTests(unittest.TestCase):
    def test_serial_and_nested_module_input_and_accumulated_parameter_gradients(self):
        for kind in ('Linear', 'Module'):
            for depth in (1, 2, 3, 5):
                models = serial(depth, kind=kind); net = compile_model(models, 'shared', constrained=False).double()
                self.assertEqual(len(list(net.parameters())), 2); self.assertEqual(sum(p.numel() for p in net.parameters()), 6)
                self.assertEqual(set(net.state_dict()), set(parameter_schema(models, 'shared')))
                leaves = [v.detach().clone().requires_grad_() for v in net.parameters()]
                for batch in (1, 4):
                    x = torch.linspace(-.7, .8, batch * 2, dtype=torch.float64).reshape(batch, 2).requires_grad_()
                    xx = x.detach().clone().requires_grad_(); expected = xx
                    for _ in range(depth * (2 if kind == 'Module' else 1)): expected = F.linear(expected, *leaves)
                    actual = net(x); torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
                    observed = torch.autograd.grad(actual.square().sum(), [x, *net.parameters()])
                    oracle = torch.autograd.grad(expected.square().sum(), [xx, *leaves])
                    for a, b in zip(observed, oracle): torch.testing.assert_close(a, b, rtol=1e-11, atol=1e-11)
                    torch.testing.assert_close(exported(models, 'shared', net)(x), expected, rtol=1e-12, atol=1e-12)

    def test_owner_after_first_execution_and_affine_composition_are_exact(self):
        models = serial(); nodes = models['shared']['graph']['nodes']
        nodes[1]['share_with'] = 'use2'; del nodes[3]['share_with']; nodes[2]['share_with'] = 'use2'
        net = compile_model(models, 'shared', constrained=False).double()
        self.assertFalse(hasattr(net, 'layer_0')); self.assertFalse(hasattr(net, 'layer_1'))
        analysis = affine_summary(net, net.contract); self.assertEqual(analysis['status'], 'available')
        x = torch.tensor([[.3, -.6], [.8, .1]], dtype=torch.float64)
        oracle = x
        for _ in range(3): oracle = F.linear(oracle, net.layer_2.weight, net.layer_2.bias)
        torch.testing.assert_close(net(x), oracle)
        torch.testing.assert_close(x @ torch.tensor(analysis['A'], dtype=torch.float64).T + torch.tensor(analysis['b'], dtype=torch.float64), oracle)

    def test_grouped_convolution_branches_share_across_different_spatial_shapes(self):
        nodes = [{'id': 'x1', 'type': 'Input', 'shape': [2, 4, 4], 'axes': ['C', 'H', 'W']},
                 {'id': 'x2', 'type': 'Input', 'shape': [2, 6, 6], 'axes': ['C', 'H', 'W']},
                 {'id': 'c1', 'type': 'Conv2d', 'out_channels': 2, 'kernel_size': 1, 'groups': 2, 'bias': True},
                 {'id': 'c2', 'type': 'Conv2d', 'out_channels': 2, 'kernel_size': 1, 'groups': 2, 'bias': True, 'share_with': 'c1'},
                 {'id': 'join', 'type': 'Concat', 'sources': ['c1', 'c2']}, {'id': 'output', 'type': 'Output'}]
        models = {'conv': {'id': 'conv', 'name': 'Two images, shared convolution', 'graph': {'nodes': nodes,
            'input_order': ['x1', 'x2'], 'edges': [{'from': a, 'to': b} for a, b in [('x1', 'c1'), ('x2', 'c2'), ('c1', 'join'), ('c2', 'join'), ('join', 'output')]]}}}
        net = compile_model(models, 'conv', constrained=False).double(); self.assertEqual(net.contract['parameters'], 4)
        self.assertEqual(sum(p.numel() for p in net.parameters()), 4)
        a = torch.linspace(-1., 1., 64, dtype=torch.float64).reshape(2, 2, 4, 4).requires_grad_()
        b = torch.linspace(-.4, .9, 144, dtype=torch.float64).reshape(2, 2, 6, 6).requires_grad_()
        aa, bb = [v.detach().clone().requires_grad_() for v in (a, b)]
        leaves = [v.detach().clone().requires_grad_() for v in net.parameters()]
        expected = torch.cat([F.conv2d(v, *leaves, groups=2).flatten(1) for v in (aa, bb)], dim=1)
        actual = net({'x1': a, 'x2': b}); torch.testing.assert_close(actual, expected)
        observed = torch.autograd.grad(actual.square().sum(), [a, b, *net.parameters()])
        oracle = torch.autograd.grad(expected.square().sum(), [aa, bb, *leaves])
        for x, y in zip(observed, oracle): torch.testing.assert_close(x, y, rtol=1e-11, atol=1e-11)
        torch.testing.assert_close(exported(models, 'conv', net)({'x1': a, 'x2': b}), expected)
        nodes[3]['groups'] = 1
        with self.assertRaisesRegex(ValueError, 'same constructor'): validate_model(models, 'conv')

    def test_adam_step_updates_shared_parameters_once_with_summed_gradients(self):
        models = serial(); net = compile_model(models, 'shared', constrained=False).double()
        leaves = [v.detach().clone().requires_grad_() for v in net.parameters()]
        actual_optimizer = torch.optim.Adam(net.parameters(), lr=.003)
        oracle_optimizer = torch.optim.Adam(leaves, lr=.003)
        x = torch.tensor([[.2, -.3], [.8, .1]], dtype=torch.float64)
        for _ in range(4):
            actual_optimizer.zero_grad(); oracle_optimizer.zero_grad(); expected = x
            for _ in range(3): expected = F.linear(expected, *leaves)
            net(x).square().mean().backward(); expected.square().mean().backward()
            actual_optimizer.step(); oracle_optimizer.step()
            for a, b in zip(net.parameters(), leaves): torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-12)
        self.assertEqual(len(actual_optimizer.state), 2)

    def test_legacy_trainer_and_affine_evidence_preserve_repeated_uses(self):
        from app.core import Workspace, generate_dataset, train_model
        from app.evidence import check_checkpoint
        models = serial(); root = models['shared']; root['training'] = {'epochs': 3, 'lr': .003, 'seed': 7}
        root['dataset'] = generate_dataset(models, 'shared', samples=12, constrained=False)
        w = Workspace(); self.addCleanup(w.db.close); w.put(root); models = w.models()
        result = train_model(models, 'shared', constrained=False)
        self.assertEqual(result['completed_updates'], 3); self.assertEqual(set(result['weights']), {'layer_0.weight', 'layer_0.bias'})
        check_checkpoint(models, 'shared', result)
        net = compile_model(models, 'shared', constrained=False)
        net.load_state_dict({k: torch.tensor(v, dtype=torch.float32) for k, v in result['weights'].items()})
        x = torch.tensor([[.4, -.1], [.8, .7]], dtype=torch.float32)
        torch.testing.assert_close(net(x), x @ torch.tensor(result['A']).T + torch.tensor(result['b']), rtol=1e-5, atol=1e-6)

    def tied_experiment(self):
        w, e = family_experiment(); self.addCleanup(w.db.close)
        family = w.get_spec('family', e['models'][0]['family']['id']); family.pop('revision')
        family['version'] = 2; family['repeats'][0]['mode'] = 'tied'
        e = copy.deepcopy(e); e.pop('revision'); e['models'][0]['family'] = w.put_spec('family', family)
        e['models'][0]['axes'][1]['values'] = [2, 3]; e['data_tasks'] = e['data_tasks'][:1]; e['initialization_seeds'] = [7]
        ref = w.put_spec('experiment', e); return w, w.get_spec('experiment', ref['id']), ref

    def test_real_tied_worker_retry_checkpoint_and_standalone_predictor(self):
        from app.run_export import predictor_source, write_new_predictor
        from app.task_execution import restore_task_predictor
        w, e, ref = self.tied_experiment(); rows = preview_experiment(e, w.get_spec)['candidates']
        self.assertEqual([r['parameters_exact'] for r in rows], ['13', '13', '22', '22'])
        for i in (0, 3):
            result = execute_candidate(w, ref, rows[i]['coordinates'], 'tied-' + str(i), 'Tied run', python=sys.executable, timeout=20)
            record = result['record']; attempt = record['attempts'][0]
            self.assertEqual(attempt['status'], 'completed', attempt['error']); self.assertEqual(attempt['result']['counts']['completed_updates'], 3)
            frozen = w.get_spec('model', record['resolved']['references']['model']['id']); self.assertEqual(frozen['version'], 2)
            bundle = candidate_bundle(e, rows[i]['coordinates'], w.get_spec, saved_manifest=record['resolved']['manifest'])
            schema = parameter_schema(frozen['models'], frozen['root'])
            self.assertEqual(set(attempt['result']['checkpoint']['weights']), set(schema))
            from app.task_execution import check_task_checkpoint
            forged = copy.deepcopy(attempt['result']['checkpoint'])
            key = next(iter(schema)); forged['weights']['invented_alias.weight'] = forged['weights'][key]
            with self.assertRaisesRegex(ValueError, 'parameter keys'): check_task_checkpoint(bundle, forged, constrained=False)
            retry = execute_candidate(w, ref, rows[i]['coordinates'], record['id'], 'Retry tied', retry=True, python=sys.executable, timeout=20)
            self.assertEqual(retry['record']['attempts'][1]['result']['checkpoint'], attempt['result']['checkpoint'])
            records = [{'x': [.2]}, {'x': [-.7]}]
            expected = {k: v.tolist() for k, v in restore_task_predictor(bundle, attempt['result']['checkpoint'], constrained=False)(records).items()}
            with tempfile.TemporaryDirectory() as td:
                write_new_predictor(predictor_source(w, result['reference']), Path(td) / 'predictor.py')
                code = 'import predictor,json,sys;assert "app" not in sys.modules;print(json.dumps({k:v.tolist() for k,v in predictor.predict_records('+repr(records)+').items()}))'
                ran = subprocess.run([sys.executable, '-c', code], cwd=td, text=True, capture_output=True, timeout=20)
                self.assertEqual(ran.returncode, 0, ran.stderr); self.assertEqual(json.loads(ran.stdout), expected)
        validate_all(w.db)

    def test_actual_tied_family_kernel_exports_unique_checkpoints_and_sources(self):
        from app.notebooks import create_bundle, execute_bundle, verify_bundle
        w, e, ref = self.tied_experiment()
        e.pop('revision'); e['models'][0]['axes'][0]['values'] = [3]; ref = w.put_spec('experiment', e)
        recipe = w.put_spec('report', {'version': 1, 'id': 'tied-rerun', 'name': 'Shared finite uses', 'question': 'Are parameters shared across uses?',
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
