"""Actual persisted LoRA policy: native, fresh export, legacy/task worker routes."""
import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from app.core import GraphError, compile_model, generate_code, train_model, restore_predictor, generate_predictor
from app.evidence import parameter_schema, check_checkpoint, normalize_models
from app.policy_graph import owner_key
from app.task_execution import check_task_checkpoint, compile_task_model, restore_task_predictor
from app.task_training import train_task
from app.transport import run_json_worker
from policy_runtime_fixture import models as fixture, bundle

torch.set_num_threads(1)


class PolicyGraphExecutionTests(unittest.TestCase):
    def test_native_and_plain_export_ranks_shapes_values_all_gradients_and_receipts(self):
        for rank in (1, 2, 4):
            for temporal in (False, True):
                models, mid = fixture(rank, temporal)
                native = compile_model(models, mid); namespace = {}; exec(generate_code(models, mid), namespace)
                exported = namespace['model']
                self.assertEqual(native.parameter_policy_receipt, exported.parameter_policy_receipt)
                for runner in (native, exported):
                    x = torch.linspace(-1., 1., 64 * (3 if temporal else 1)).reshape((4, 3, 16) if temporal else (4, 16)).requires_grad_()
                    p = dict(runner.named_parameters()); W = p['layer_0.weight']; A = p['layer_1.weight']; B = p['layer_2.weight']
                    with torch.no_grad(): B.copy_(torch.linspace(-.02, .03, 16 * rank).reshape(16, rank))
                    a = A.detach().clone().requires_grad_(); b = B.detach().clone().requires_grad_(); reference_x = x.detach().clone().requires_grad_()
                    reference = reference_x @ W.detach().T + (4. / rank) * (reference_x @ a.T) @ b.T
                    output = runner(x); torch.testing.assert_close(output, reference)
                    actual = torch.autograd.grad(output.square().sum(), [x, A, B])
                    expected = torch.autograd.grad(reference.square().sum(), [reference_x, a, b])
                    for aa, bb in zip(actual, expected): torch.testing.assert_close(aa, bb)
                    self.assertFalse(W.requires_grad); self.assertIsNone(W.grad)

    def test_real_task_worker_initialization_freezing_checkpoint_restore_and_identity_refusal(self):
        for temporal in (False, True):
            b = bundle(30, temporal); initial = compile_task_model(b, constrained=False)
            base = initial.layer_0.weight.detach().clone()
            local = train_task(b, constrained=False)
            self.assertEqual(local['status'], 'completed', local['failure'])
            checkpoint = local['checkpoint']; self.assertEqual(checkpoint['version'], 2)
            check_task_checkpoint(b, checkpoint, constrained=False)
            torch.testing.assert_close(torch.tensor(checkpoint['weights']['layer_0.weight']), base, rtol=0, atol=0)
            worker = run_json_worker([sys.executable, '-u', '-m', 'app.worker'],
                {'kind': 'task-v1', 'bundle': b, 'constrained': False}, cwd=str(Path(__file__).parents[1]))
            self.assertEqual(worker['checkpoint'], checkpoint); self.assertEqual(worker['history'], local['history'])
            captured = []
            def capture(*args, **kwargs):
                model = compile_task_model(*args, **kwargs); captured.append(model); return model
            with patch('app.task_execution.compile_task_model', side_effect=capture):
                restored = restore_task_predictor(b, checkpoint, constrained=False)
            self.assertFalse(captured[0].layer_0.weight.requires_grad)
            self.assertEqual(captured[0].parameter_policy_receipt['policy_identity'], checkpoint['parameter_policy_receipt']['policy_identity'])
            self.assertIn('y', restored([r['values'] for r in b['dataset']['records'][:2]]))
            bad = copy.deepcopy(checkpoint); bad['parameter_policy_receipt']['counts']['trainable'] = 320
            with self.assertRaisesRegex(GraphError, 'receipt'): check_task_checkpoint(b, bad, constrained=False)

    def test_legacy_fit_checkpoint_and_fresh_trained_export_keep_frozen_base(self):
        models, mid = fixture(); b = bundle(20); records = b['dataset']['records']
        models[mid].update(dataset={'x': [r['values']['x'] for r in records], 'y': [r['values']['y'] for r in records]},
                           training={'epochs': 40, 'lr': .02, 'seed': 7})
        models = normalize_models(models, mid)
        result = train_model(models, mid, constrained=False); self.assertEqual(result['checkpoint_version'], 2)
        check_checkpoint(models, mid, result)
        expected = compile_model(models, mid).layer_0.weight
        self.assertTrue(torch.equal(expected, torch.tensor(result['weights']['layer_0.weight'])))
        restored = restore_predictor(models, mid, result, constrained=False)
        namespace = {}; exec(generate_predictor(models, mid, result), namespace)
        x = torch.ones(3, 16); torch.testing.assert_close(restored(x), namespace['model'](x), rtol=0, atol=0)
        self.assertFalse(namespace['model'].layer_0.weight.requires_grad)

    def test_no_app_fresh_architecture_export_and_no_missing_captured_policy_dependencies(self):
        models, mid = fixture(); source = generate_code(models, mid)
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / 'export.py'; file.write_text(source + '\nprint(model.parameter_policy_receipt["counts"])\n', encoding='utf-8')
            process = subprocess.run([sys.executable, '-I', str(file)], cwd=directory, capture_output=True, text=True, timeout=30)
            self.assertEqual(process.returncode, 0, process.stderr)
            self.assertIn("'trainable': 64", process.stdout)

    def test_all_frozen_inference_is_valid_and_training_refuses_before_optimizer(self):
        models, mid = fixture()
        for entry in models[mid]['graph']['parameter_policy']['parameters'].values(): entry['trainable'] = False
        self.assertTrue(torch.isfinite(compile_model(models, mid)(torch.ones(2, 16))).all())
        models[mid].update(dataset={'x': [[0.] * 16] * 2, 'y': [[0.] * 16] * 2}, training={'epochs': 1})
        with self.assertRaisesRegex(GraphError, 'No trainable'): train_model(models, mid)


if __name__ == '__main__': unittest.main()
