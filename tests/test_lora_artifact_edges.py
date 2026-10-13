"""Actual temporal LoRA checkpoint, signed-zero base and nonexact alpha/r."""
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from app.core import Workspace, compile_model
from app.lora import linear, dataset, task
from app.parameter_sources import inventory
from app.lora_artifacts import prepare, write_new_package, validate_package
from app.experiment_objects import execute_candidate, preview_experiment
from evidence_fixture import experiment


class LoRAArtifactEdgeTests(unittest.TestCase):
    def test_real_temporal_checkpoint_signed_zero_nonexact_scale_and_upstream_gradients(self):
        import torch
        torch.set_num_threads(1)
        w = Workspace(); self.addCleanup(w.db.close)
        self.addCleanup(lambda: w.history.directory.cleanup() if w.history.directory else None)
        model = linear('temporal-adapter', rank=3)
        for node in model['graph']['nodes']:
            if node['type'] in ('Input', 'Output'): node['shape'] = [3,16]
        values = model['graph']['parameter_policy']['parameters']['["base","weight"]']['initializer']['values']
        for i, row in enumerate(values):
            for j in range(len(row)):
                if i != j: row[j] = -0.
        data = dataset(samples=16)
        for field in data['fields']: field['shape'] = [3,16]
        for row in data['records']:
            row['values'] = {key: [[scale * v for v in vector] for scale in (1., .5, -.3)]
                             for key, vector in row['values'].items()}
        ref = experiment(w, model, data, task(), 'temporal-package', updates=3)
        exp = w.get_spec('experiment', ref['id'], ref['revision'])
        coordinates = preview_experiment(exp, w.get_spec)['candidates'][0]['coordinates']
        executed = execute_candidate(w, ref, coordinates, 'temporal-package-run', 'Temporal adapter', python=sys.executable, timeout=30)
        attempt = executed['record']['attempts'][0]; self.assertEqual(attempt['status'], 'completed', attempt['error'])
        listing = inventory(w, executed['reference'], attempt=1)
        selections = {}
        for role in ('base', 'A', 'B'):
            row = next(p for p in listing['parameters'] if json.loads(p['id'])[-2:] == [role, 'weight'])
            selections[role] = {'run': executed['reference'], 'attempt': 1,
                'checkpoint_id': listing['checkpoint_id'], 'parameter': {key: row[key] for key in ('path', 'id')}}
        prepared = prepare(w, selections)
        native = compile_model(w.models(), model['id'])
        native.load_state_dict({key: torch.tensor(value, dtype=torch.float32)
                               for key, value in attempt['result']['checkpoint']['weights'].items()})
        x = torch.arange(144, dtype=torch.float32).reshape(3,3,16).div(100).requires_grad_()
        expected = native(x); expected.sum().backward()
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary) / 'package'; write_new_package(prepared, folder)
            checked = validate_package(folder)
            self.assertEqual(checked['manifest']['operation']['scale'], 4. / 3)
            self.assertEqual(checked['manifest']['ports']['input'][0]['shape'], [3,16])
            self.assertEqual(math.copysign(1., checked['values']['base'][0][1]), -1.)
            payload = {'input': x.detach().tolist(), 'expected': expected.detach().tolist(), 'gradient': x.grad.tolist()}
            script = '''import json,sys,torch,predictor
torch.set_num_threads(1)
p=json.loads(sys.argv[1]); model=predictor.load()
x=torch.tensor(p['input'],dtype=torch.float32).requires_grad_()
y=model({'x':x})['y']
torch.testing.assert_close(y,torch.tensor(p['expected']),rtol=0,atol=0)
y.sum().backward();torch.testing.assert_close(x.grad,torch.tensor(p['gradient']),rtol=0,atol=0)
assert torch.signbit(model.base[0,1]) and not list(model.parameters())
merged=model.base+(4./3)*(model.B@model.A)
torch.testing.assert_close(y,torch.nn.functional.linear(x,merged),rtol=1e-5,atol=2e-6)
print('PASS actual native temporal checkpoint, signed-zero base, alpha/r and input gradients')
'''
            process = subprocess.run([sys.executable, '-c', script, json.dumps(payload)], cwd=folder,
                env={**os.environ, 'PYTHONPATH': str(folder)}, capture_output=True, text=True, timeout=30)
            self.assertEqual(process.returncode, 0, process.stderr)
            self.assertIn('PASS actual native temporal checkpoint', process.stdout)


if __name__ == '__main__': unittest.main()
