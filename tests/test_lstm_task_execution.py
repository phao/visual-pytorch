"""Concrete LSTM tasks, real worker/checkpoints and captured notebook kernels."""
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from app.core import Workspace, compile_model, validate_model
from app.datasets import make_split
from app.revisions import model_document
from app.task_execution import task_bundle, training_protocol, check_task_checkpoint, restore_task_predictor, generate_task_predictor
from app.task_training import train_task
from app.transport import run_json_worker
from examples.projected_lstm import models as fixture
from test_datasets import data, task
from test_lstm_execution import oracle

torch.set_num_threads(1)


def lstm_bundle(updates=40):
    models, mid = fixture(steps=3, width=1, hidden=3, projection=2, layers=1)
    contract = validate_model(models, mid)
    torch.manual_seed(41); teacher = compile_model(models, mid, constrained=False)
    values = {g['id']: torch.randn(24, *g['shape']) * .3 for g in contract['input_groups']}
    lengths = torch.tensor([i % 4 for i in range(24)])
    values['lengths'] = lengths.float()[:, None]
    with torch.no_grad():
        targets = oracle(values['sequence'], values['hidden'], values['cell'], dict(teacher.layer_0.named_parameters()),
                         lengths=lengths, layers=1, bidirectional=True, projection=2)
    fields = [dict(id=g['id'], dtype='float32', shape=g['shape'], **({'axes': g['axes']} if g.get('axes') else {}))
              for g in contract['input_groups']]
    fields += [dict(id='target_' + role, dtype='float32', shape=contract['shapes']['lstm']['ports'][role],
                    axes=contract['shapes']['lstm']['port_axes'][role]) for role in ('sequence', 'hidden', 'cell')]
    fields.append(dict(id='valid', dtype='bool', shape=[3, 4]))
    rows = []
    for row in range(24):
        record = {key: value[row].tolist() for key, value in values.items()}
        record.update({'target_' + key: value[row].tolist() for key, value in targets.items()})
        record['valid'] = ((torch.arange(3) < lengths[row])[:, None].expand(3, 4)).tolist()
        rows.append(record)
    dataset = data(fields, rows); binding = task()
    binding['inputs'] = [dict(port=g['id'], field=g['id'], transforms=[]) for g in contract['input_groups']]
    binding['heads'] = [dict(id=role, port='out_' + role, field='target_' + role, loss='mse',
                             weight=1. if role == 'sequence' else .2, transforms=[], **({'mask': 'valid'} if role == 'sequence' else {}))
                        for role in ('sequence', 'hidden', 'cell')]
    split = make_split(dataset, {'method': 'external', 'roles': {'train': ['r'+str(i) for i in range(16)],
                        'validation': ['r'+str(i) for i in range(16, 20)], 'test': ['r'+str(i) for i in range(20, 24)]}, 'excluded': []})
    return task_bundle(model_document(models, mid), dataset, binding, split,
                       training_protocol(epochs=None, updates=updates, batch_size=16, lr=.03, evaluate_every=20))


class LSTMTaskExecutionTests(unittest.TestCase):
    def test_real_fit_worker_identity_and_exact_checkpoint(self):
        bundle = lstm_bundle(60); before = copy.deepcopy(bundle)
        local = train_task(bundle, constrained=False)
        self.assertEqual(local['status'], 'completed', local['failure'])
        self.assertEqual(local['counts']['completed_updates'], 60)
        self.assertLess(local['final_metrics']['train']['objective'], .2 * local['history'][0]['metrics']['train']['objective'])
        check_task_checkpoint(bundle, local['checkpoint'], constrained=False)
        points = []; worker = run_json_worker([sys.executable, '-u', '-m', 'app.worker'],
                    {'kind': 'task-v1', 'bundle': bundle, 'constrained': False}, points.append, timeout=30)
        self.assertEqual(worker['checkpoint'], local['checkpoint']); self.assertEqual(worker['history'], local['history'])
        self.assertEqual(points, worker['history']); self.assertEqual(bundle, before)

    def test_execution_lengths_do_not_infer_loss_masks(self):
        bundle = lstm_bundle(1)
        # All-false explicit supervision means no updates even though execution
        # lengths still contain positive prefixes. Every Output retains a head.
        dataset = {k: copy.deepcopy(v) for k, v in bundle['dataset'].items() if k != 'revision'}
        binding = {k: copy.deepcopy(v) for k, v in bundle['task'].items() if k != 'revision'}
        dataset['fields'].append(dict(id='valid_state', dtype='bool', shape=[]))
        for head in binding['heads'][1:]: head['mask'] = 'valid_state'
        for row in dataset['records']:
            row['values']['valid'] = [[False] * 4 for _ in range(3)]
            row['values']['valid_state'] = False
        # Rebuild, rather than tamper with a captured payload's hash.
        bundle = task_bundle(bundle['model'], dataset, binding,
                             make_split(dataset, {'method': 'row', 'seed': 7, 'fractions': [.7, .2, .1]}), bundle['protocol'])
        result = train_task(bundle, constrained=False)
        self.assertEqual(result['status'], 'unavailable'); self.assertEqual(result['counts']['completed_updates'], 0)

    def test_saved_predictor_and_app_free_export_preserve_all_three_ports(self):
        bundle = lstm_bundle(4); result = train_task(bundle, constrained=False)
        records = [r['values'] for r in bundle['dataset']['records'][-4:]]
        expected = restore_task_predictor(bundle, result['checkpoint'], constrained=False)(records)
        source = generate_task_predictor(bundle, result['checkpoint'], constrained=False)
        import subprocess
        with tempfile.TemporaryDirectory() as directory:
            probe = source + '\nimport json,sys\nout=predict_records(' + repr(records) + ')\nassert not any(n=="app" or n.startswith("app.") for n in sys.modules)\nprint(json.dumps({k:v.tolist() for k,v in out.items()}))'
            run = subprocess.run([sys.executable, '-c', probe], cwd=directory, capture_output=True, text=True, timeout=30)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(json.loads(run.stdout), {k: v.tolist() for k, v in expected.items()})

    def test_actual_saved_and_rerun_kernels_keep_model_v4_and_lstm_closure(self):
        from app.experiment_objects import execute_candidate, object_reference, preview_experiment
        from app.notebooks import create_bundle, execute_bundle, verify_bundle
        bundle = lstm_bundle(2); w = Workspace(); self.addCleanup(w.db.close)
        w.set_constrained_mode(False)
        refs = {kind: w.put_spec(kind, bundle[kind]) for kind in ('model', 'dataset', 'task')}
        experiment = {'version': 1, 'id': 'lstm', 'name': 'Projected bidirectional trial', 'question': 'Can state ports replay?',
                      'models': [refs['model']], 'data_tasks': [{'id': 'pair', 'dataset': refs['dataset'], 'task': refs['task']}],
                      'splits': [{'id': 'split', 'protocol': {'method': 'row', 'seed': 31, 'fractions': [.65, .15, .2]}}],
                      'initialization_seeds': [7], 'protocols': [{'id': 'adam', 'protocol': bundle['protocol']}],
                      'selection': {'kind': 'crossed', 'pairs': [], 'exclusions': []}}
        ref = w.put_spec('experiment', experiment); frozen = w.get_spec('experiment', ref['id'], ref['revision'])
        selected = preview_experiment(frozen, w.get_spec)['candidates'][0]; self.assertEqual(selected['status'], 'compatible')
        run = execute_candidate(w, object_reference('experiment', frozen), selected['coordinates'], 'run', 'Actual LSTM run', timeout=30)
        attempt = run['record']['attempts'][-1]; self.assertEqual(attempt['status'], 'completed', attempt['error'])
        with tempfile.TemporaryDirectory() as directory:
            views = []
            for mode in ('result-report', 'rerun'):
                recipe = {'version': 1, 'id': mode, 'name': 'LSTM evidence', 'question': 'Can this run be reproduced?',
                          'mode': mode, 'runs': [run['reference']] if mode == 'result-report' else [],
                          'experiment': object_reference('experiment', frozen) if mode == 'rerun' else None,
                          'sections': ['models', 'training', 'evaluation', 'checkpoints', 'failures'],
                          'analysis': {'kind': 'descriptive', 'inferential_unit': 'run'}}
                report = w.put_spec('report', recipe); exported = create_bundle(w, report, Path(directory) / mode)
                verify_bundle(exported['path']); self.assertTrue((Path(exported['path']) / 'runtime/app/lstm_nodes.py').is_file())
                result = execute_bundle(exported['path'], timeout=60)
                self.assertEqual(result['execution']['status'], 'completed', result['execution']['error'])
                view = json.loads((Path(result['path']) / 'report-data.json').read_text())
                self.assertEqual(view['runs'][0]['status'], 'completed')
                self.assertEqual(view['runs'][0]['checkpoint_id'], attempt['result']['checkpoint_id'])
                self.assertEqual(view['notebook_execution']['training_requested'], mode == 'rerun')
                views.append(view)
            self.assertEqual(views[0]['runs'][0]['metrics'], views[1]['runs'][0]['metrics'])


if __name__ == '__main__': unittest.main()
