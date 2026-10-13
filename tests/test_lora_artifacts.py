import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from app.core import GraphError, Workspace
from app.lora import populate
from app.lora_execution import merge_selections
from app.experiment_objects import execute_candidate, preview_experiment
from app.lora_artifacts import prepare, write_new_package, validate_package


class LoRAArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.w = Workspace(); refs = populate(cls.w, updates=3)
        ref = refs['lora-linear-comparison']
        exp = cls.w.get_spec('experiment', ref['id'], ref['revision'])
        coordinate = next(row['coordinates'] for row in preview_experiment(exp, cls.w.get_spec)['candidates']
                          if row['coordinates']['model']['id'] == 'lora-r2')
        result = execute_candidate(cls.w, ref, coordinate, 'package-source', 'Package source', python=sys.executable, timeout=30)
        assert result['record']['attempts'][0]['status'] == 'completed', result
        cls.selections = merge_selections(cls.w, result['reference'], attempt=1)
        cls.prepared = prepare(cls.w, cls.selections)

    @classmethod
    def tearDownClass(cls):
        if cls.w.history.directory: cls.w.history.directory.cleanup()
        cls.w.db.close()

    def test_exact_package_pure_read_and_existing_directory_refusal(self):
        raw = self.w.db.serialize(); package = prepare(self.w, self.selections)
        self.assertEqual(raw, self.w.db.serialize())
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary) / 'package'; receipt = write_new_package(package, folder)
            checked = validate_package(folder, expected_manifest=receipt['manifest_sha256'])
            self.assertFalse(checked['manifest']['optimizer_resume'])
            self.assertEqual(set(p.name for p in folder.iterdir()), {'base.f32', 'A.f32', 'B.f32', 'manifest.json', 'predictor.py'})
            before = {p.name: p.read_bytes() for p in folder.iterdir()}
            with self.assertRaises(FileExistsError): write_new_package(package, folder)
            self.assertEqual(before, {p.name: p.read_bytes() for p in folder.iterdir()})

    def test_missing_corrupt_swapped_base_and_manifest_or_source_tampering_refuse(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary) / 'package'; write_new_package(self.prepared, folder)
            base = folder / 'base.f32'; original = base.read_bytes(); base.unlink()
            with self.assertRaises(FileNotFoundError): validate_package(folder)
            base.write_bytes(original[:-1])
            with self.assertRaises(GraphError): validate_package(folder)
            base.write_bytes(b'\x00' * len(original))
            with self.assertRaises(GraphError): validate_package(folder)
            base.write_bytes(original)
            manifest = folder / 'manifest.json'; raw = manifest.read_text()
            changed = json.loads(raw); changed['operation']['scale'] = 99.
            manifest.write_text(json.dumps(changed))
            with self.assertRaises(GraphError): validate_package(folder, expected_manifest=self.prepared['manifest_sha256'])
            with self.assertRaises(GraphError): validate_package(folder)
            manifest.write_text(raw)
            predictor = folder / 'predictor.py'; predictor.write_text(predictor.read_text() + '# changed\n')
            with self.assertRaises(GraphError): validate_package(folder)

    def test_payload_tampering_refuses_before_any_directory_creation(self):
        package = copy.deepcopy(self.prepared); package['payloads']['base.f32'] = b'bad'
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary) / 'package'
            with self.assertRaises(GraphError): write_new_package(package, folder)
            self.assertFalse(folder.exists())

    def test_fresh_portable_predictor_and_merged_oracle_with_input_gradients(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary) / 'package'; write_new_package(self.prepared, folder)
            script = '''import torch, predictor
torch.set_num_threads(1)
model = predictor.load()
x = torch.arange(48, dtype=torch.float32).reshape(3,16).div(50).requires_grad_()
y = model({'x': x})['y']
oracle = torch.nn.functional.linear(x, model.base) + 2 * torch.nn.functional.linear(torch.nn.functional.linear(x, model.A), model.B)
torch.testing.assert_close(y, oracle, rtol=0, atol=0)
merged = model.base + 2 * model.B @ model.A
torch.testing.assert_close(y, torch.nn.functional.linear(x, merged), rtol=1e-5, atol=2e-6)
y.sum().backward()
assert x.grad is not None and torch.isfinite(x.grad).all()
assert not list(model.parameters()) and all(not v.requires_grad for v in model.buffers())
for wrong in ({'wrong': x}, {'x': x.double()}, {'x': x.reshape(1,3,16)}):
    try: model(wrong)
    except ValueError: pass
    else: raise AssertionError('Invalid port/dtype/sample shape admitted')
print('PASS portable exact-base inference and merged oracle')
'''
            environment = {**os.environ, 'PYTHONPATH': str(folder)}
            process = subprocess.run([sys.executable, '-c', script], cwd=folder, env=environment,
                                     capture_output=True, text=True, timeout=30)
            self.assertEqual(process.returncode, 0, process.stderr)
            self.assertIn('PASS portable', process.stdout)
            base = folder / 'base.f32'; original = base.read_bytes(); base.write_bytes(b'\x00' * len(original))
            refused = subprocess.run([sys.executable, '-c', 'import predictor; predictor.load()'], cwd=folder,
                env=environment, capture_output=True, text=True, timeout=30)
            self.assertNotEqual(refused.returncode, 0); self.assertIn('identity mismatch', refused.stderr)

    def test_fresh_verification_never_imports_numerical_packages(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary) / 'package'; write_new_package(self.prepared, folder)
            script = ('import sys; from app.lora_artifacts import validate_package; validate_package(sys.argv[1]);'
                      'assert not any(x in sys.modules for x in ("torch", "numpy", "PyQt6"))')
            process = subprocess.run([sys.executable, '-c', script, str(folder)], capture_output=True, text=True, timeout=30)
            self.assertEqual(process.returncode, 0, process.stderr)

    def test_cli_exports_exact_package_without_workspace_changes_or_numerical_import(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary); workspace = folder / 'source.vtw'; self.w.save(workspace)
            before = workspace.read_bytes()
            selection = folder / 'sources.json'; selection.write_text(json.dumps(self.selections))
            output = folder / 'package'
            script = ('import sys,json,vpw_cli; result=vpw_cli.main(sys.argv[1:]);'
                      'assert not any(x in sys.modules for x in ("torch","numpy","PyQt6"));sys.exit(result)')
            command = [sys.executable, '-c', script, str(workspace), 'export-lora-package', str(selection), str(output)]
            process = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=30)
            self.assertEqual(process.returncode, 0, process.stderr + process.stdout)
            value = json.loads(process.stdout); self.assertTrue(value['ok'])
            validate_package(output, expected_manifest=value['value']['manifest_sha256'])
            self.assertEqual(before, workspace.read_bytes())
            refused = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=30)
            self.assertEqual(refused.returncode, 1); self.assertFalse(json.loads(refused.stdout)['ok'])
            self.assertEqual(before, workspace.read_bytes())


if __name__ == '__main__': unittest.main()
