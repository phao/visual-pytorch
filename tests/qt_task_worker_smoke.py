"""Actual Qt thread/signals -> task worker; no Torch imported in GUI process."""
import json
import sys
from PyQt6.QtCore import QCoreApplication, QTimer
from app.core import default_model
from app.datasets import make_split
from app.revisions import model_document
from app.task_execution import task_bundle, training_protocol
from app.desktop import TaskTrainingJob, find_torch_python

app = QCoreApplication([])
model = default_model(inputs=1, outputs=2)
dataset = {'version': 1, 'id': 'classification', 'name': 'Qt worker test',
    'storage': {'kind': 'inline-records', 'version': 1},
    'fields': [{'id': 'signal', 'dtype': 'float32', 'shape': [1]},
               {'id': 'class', 'dtype': 'category', 'shape': [], 'vocabulary': {'version': 1, 'values': ['negative', 'positive']}}],
    'records': [{'id': 'r' + str(i), 'values': {'signal': [(i - 15.5) / 5.],
        'class': 'negative' if i < 16 else 'positive'}} for i in range(32)]}
task = {'version': 1, 'id': 'classification', 'name': 'Fixed vocabulary',
    'inputs': [{'port': 'input', 'field': 'signal', 'transforms': [{'kind': 'standardize'}]}],
    'heads': [{'id': 'classification', 'port': 'output', 'field': 'class', 'loss': 'cross-entropy', 'weight': 1., 'transforms': []}]}
split = make_split(dataset, {'method': 'row', 'seed': 31, 'fractions': [.65, .15, .2]})
bundle = task_bundle(model_document({model['id']: model}, model['id']), dataset, task, split,
                     training_protocol(epochs=60, batch_size=5, lr=.05, evaluate_every=40))
python = find_torch_python()
if python is None:
    raise SystemExit('No numerical Python found.')
job = TaskTrainingJob(bundle, app, python)
output = {'points': []}
job.point.connect(lambda point: output['points'].append(json.loads(point)))
job.completed.connect(lambda result: output.update(result=json.loads(result)))
job.failed.connect(lambda error: output.update(error=error))
job.finished.connect(app.quit)
QTimer.singleShot(30000, job.stop)
job.start(); app.exec(); job.wait()
if 'result' not in output:
    raise SystemExit(str(output))
result = output['result']; metric = result['final_metrics']['test']['heads']['classification']
assert result['status'] == 'completed', result['failure']
assert metric['accuracy'] == 1., metric
assert output['points'] == result['history']
assert all(set(point['metrics']) == {'train', 'validation'} for point in output['points'])
assert 'torch' not in sys.modules
print(json.dumps({'engine': 'PyQt6 task worker', 'status': result['status'],
    'updates': result['counts']['completed_updates'], 'test_accuracy': metric['accuracy'],
    'test_cross_entropy': metric['value'], 'points': len(output['points']), 'host_torch': False}))
