"""Independent numerical, real worker and fresh Jupyter family-axis checks."""
import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from app.core import compile_model, generate_code
from app.experiment_objects import candidate_bundle, execute_candidate, object_reference, preview_experiment
from app.revisions import validate_all
from test_family_axes import family_experiment


class FamilyAxisExecutionTests(unittest.TestCase):
    def setUp(self):
        self.w, self.e = family_experiment()
        self.addCleanup(self.w.db.close)

    def test_new_runtime_missing_axis_dependency_is_rejected_before_kernel_launch(self):
        from app.core import GraphError
        from app.notebooks import create_bundle, verify_bundle
        recipe = self.w.put_spec('report', {'version': 1, 'id': 'dependency-guard', 'name': 'Guard', 'question': 'Is the source closure complete?',
            'mode': 'rerun', 'experiment': object_reference('experiment', self.e), 'runs': [], 'sections': ['models'],
            'analysis': {'kind': 'descriptive', 'inferential_unit': 'run'}})
        with tempfile.TemporaryDirectory() as td:
            root = Path(create_bundle(self.w, recipe, Path(td) / 'bundle')['path'])
            path = root / 'manifest.json'; manifest = json.loads(path.read_text())
            manifest['files'].pop('runtime/app/family_axes.py'); path.write_text(json.dumps(manifest))
            (root / 'runtime/app/family_axes.py').unlink()
            with self.assertRaisesRegex(GraphError, 'requires its captured family_axes.py'): verify_bundle(root)
            self.assertFalse((root / 'executions').exists())

    def test_all_four_model_coordinates_match_independent_forward_gradients_and_export(self):
        import torch
        torch.set_num_threads(1)
        rows = preview_experiment(self.e, self.w.get_spec)['candidates']
        for index in (0, 4, 8, 12):
            row = rows[index]; doc = candidate_bundle(self.e, row['coordinates'], self.w.get_spec)['model']
            network = compile_model(doc['models'], doc['root'], constrained=False).double()
            bindings = row['coordinates']['model']['bindings']; width, depth = bindings['hidden-width'], bindings['hidden-depth']
            self.assertEqual(sum(p.numel() for p in network.parameters()), (width * 2) + depth * (width**2 + width) + width + 1)
            x = torch.tensor([[-.3], [.7], [1.1]], dtype=torch.float64, requires_grad=True)
            xx = x.detach().clone().requires_grad_(); leaves = {k: v.detach().clone().requires_grad_() for k, v in network.named_parameters()}
            y = xx @ leaves['layer_0.weight'].T + leaves['layer_0.bias']
            for layer in range(1, depth * 2, 2): y = torch.tanh(y @ leaves[f'layer_{layer}.weight'].T + leaves[f'layer_{layer}.bias'])
            layer = depth * 2 + 1; y = y @ leaves[f'layer_{layer}.weight'].T + leaves[f'layer_{layer}.bias']
            actual = network(x); torch.testing.assert_close(actual, y, rtol=1e-12, atol=1e-12)
            expected = torch.autograd.grad(y.square().sum(), [xx, *leaves.values()])
            observed = torch.autograd.grad(actual.square().sum(), [x, *network.parameters()])
            for a, b in zip(observed, expected): torch.testing.assert_close(a, b, rtol=1e-11, atol=1e-11)
            scope = {}; exec(generate_code(doc['models'], doc['root']), scope)
            exported = scope['model'].double(); exported.load_state_dict(network.state_dict())
            torch.testing.assert_close(exported(x), y, rtol=1e-12, atol=1e-12)

    def test_eight_real_family_trials_retry_checkpoint_and_fresh_standalone_predictor(self):
        import torch
        from app.run_export import predictor_source, write_new_predictor
        from app.task_execution import restore_task_predictor
        torch.set_num_threads(1)
        rows = preview_experiment(self.e, self.w.get_spec)['candidates']
        completed = []
        # Four architectures × two independent initializations, fixed dataset.
        for index in (0, 1, 4, 5, 8, 9, 12, 13):
            row = rows[index]
            produced = execute_candidate(self.w, object_reference('experiment', self.e), row['coordinates'], 'trial-' + str(index),
                                         'Family trial ' + str(index), python=sys.executable, timeout=20)
            attempt = produced['record']['attempts'][0]
            self.assertEqual(attempt['status'], 'completed', attempt['error'])
            self.assertEqual(attempt['result']['counts']['completed_updates'], 3)
            self.assertEqual(produced['record']['resolved']['references'], row['references']); completed.append(produced)
        self.assertEqual(len(self.w.list_specs('specialization')), 4)
        first = completed[0]; coords = first['record']['selection']['coordinates']
        retry = execute_candidate(self.w, object_reference('experiment', self.e), coords, first['record']['id'], 'Retry family trial',
                                  retry=True, python=sys.executable, timeout=20)
        self.assertEqual(retry['record']['attempts'][0], first['record']['attempts'][0])
        self.assertEqual(retry['record']['attempts'][1]['result']['checkpoint'], first['record']['attempts'][0]['result']['checkpoint'])
        bundle = candidate_bundle(self.e, coords, self.w.get_spec, saved_manifest=first['record']['resolved']['manifest'])
        records = [{'x': [.1]}, {'x': [-1.]}]
        expected = {k: v.tolist() for k, v in restore_task_predictor(bundle, first['record']['attempts'][0]['result']['checkpoint'], constrained=False)(records).items()}
        source = predictor_source(self.w, first['reference'])
        with tempfile.TemporaryDirectory() as td:
            write_new_predictor(source, Path(td) / 'predictor.py')
            script = 'import predictor,sys,json; assert "app" not in sys.modules; print(json.dumps({k:v.tolist() for k,v in predictor.predict_records('+repr(records)+').items()}))'
            result = subprocess.run([sys.executable, '-c', script], cwd=td, capture_output=True, text=True, timeout=20)
            self.assertEqual(result.returncode, 0, result.stderr); self.assertEqual(json.loads(result.stdout), expected)
            self.w.save(Path(td) / 'runs.vtw')
        validate_all(self.w.db)

    def test_real_family_rerun_notebook_keeps_closure_provenance_figures_and_predictors(self):
        from app.notebooks import create_bundle, execute_bundle, verify_bundle
        from app.report_runtime import resolve_payload
        experiment = copy.deepcopy(self.e); experiment.pop('revision'); experiment['data_tasks'] = experiment['data_tasks'][:1]; experiment['initialization_seeds'] = [7]
        ref = self.w.put_spec('experiment', experiment)
        recipe = self.w.put_spec('report', {'version': 1, 'id': 'family-rerun', 'name': 'Width depth experiment', 'question': 'Which explicit concrete family instances did we execute?',
            'mode': 'rerun', 'experiment': ref, 'runs': [], 'sections': ['models', 'training', 'evaluation', 'checkpoints', 'failures'],
            'analysis': {'kind': 'descriptive', 'inferential_unit': 'run'}})
        with tempfile.TemporaryDirectory() as td:
            before = self.w.db.serialize(); export = create_bundle(self.w, recipe, Path(td) / 'bundle')
            self.assertEqual(before, self.w.db.serialize())
            root = Path(export['path']); self.assertTrue((root / 'runtime/app/family_axes.py').is_file())
            payload = json.loads((root / 'payload.json').read_text()); resolve_payload(payload)
            self.assertEqual(payload['preview']['total_combinations'], 4)
            self.assertFalse(any(d['kind'] == 'specialization' for d in payload['documents']))
            result = execute_bundle(root, timeout=60)
            self.assertEqual(result['execution']['status'], 'completed', result['execution']['error'])
            output = Path(result['path']); view = json.loads((output / 'report-data.json').read_text())
            self.assertEqual(len(view['runs']), 4); self.assertEqual({r['status'] for r in view['runs']}, {'completed'})
            self.assertEqual(len(list(output.glob('architecture-*.py'))), 4)
            self.assertEqual(len(list(output.glob('predictor-*.py'))), 4)
            self.assertEqual(len(list(output.glob('curve-*.svg'))), 4)
            self.assertEqual({r['selection']['coordinates']['model']['bindings']['hidden-depth'] for r in view['runs']}, {1, 2})
            self.assertTrue(all('specialization' in r['resolved']['references'] for r in view['runs']))
            verify_bundle(root); self.assertEqual(before, self.w.db.serialize())
