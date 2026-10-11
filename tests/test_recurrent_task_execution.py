"""Real bounded recurrent graph training, worker, predictors and notebook kernel."""
import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from app.core import Workspace
from app.datasets import make_split
from app.revisions import model_document
from app.task_execution import (task_bundle, training_protocol, check_task_checkpoint,
                                restore_task_predictor, generate_task_predictor)
from app.task_training import train_task
from app.transport import run_json_worker
from test_datasets import data, task
from test_recurrent_graph_contract import graph_fixture
from test_recurrent_execution import oracle

torch.set_num_threads(1)


def recurrent_bundle(*, updates=40, unavailable=False):
    models, mid = graph_fixture(steps=4)
    fields = [{'id': 'sequence', 'dtype': 'float32', 'shape': [4, 2]},
              {'id': 'initial', 'dtype': 'float32', 'shape': [2]},
              {'id': 'lengths', 'dtype': 'float32', 'shape': [1]},
              {'id': 'mask', 'dtype': 'float32', 'shape': [4]},
              {'id': 'reset', 'dtype': 'float32', 'shape': [4]},
              {'id': 'target', 'dtype': 'float32', 'shape': [4, 2], 'axes': ['T', 'D0']},
              {'id': 'valid', 'dtype': 'bool', 'shape': [4, 2]},
              {'id': 'target_final', 'dtype': 'float32', 'shape': [2]},
              {'id': 'valid_final', 'dtype': 'bool', 'shape': []}]
    generator = torch.Generator().manual_seed(31)
    x = torch.randn(24, 4, 2, generator=generator)
    h = torch.randn(24, 2, generator=generator) * .2
    lengths = torch.tensor([i % 5 for i in range(24)])
    mask = torch.tensor([[not (i % 3 == 0 and t == 1) for t in range(4)] for i in range(24)])
    reset = torch.tensor([[t == 2 and i % 2 == 0 for t in range(4)] for i in range(24)])
    target, final = oracle(x, h, torch.tensor([[.4, -.2, .25, .1], [-.1, .5, -.2, .3]]), torch.tensor([.1, -.1]), lengths=lengths, mask=mask, reset=reset)
    valid = ((torch.arange(4)[None, :] < lengths[:, None]) & mask)[:, :, None].expand(24, 4, 2)
    rows = [{'sequence': x[i].tolist(), 'initial': h[i].tolist(), 'lengths': [float(lengths[i])],
             'mask': mask[i].float().tolist(), 'reset': reset[i].float().tolist(),
             'target': target[i].tolist(), 'valid': (torch.zeros_like(valid[i]) if unavailable else valid[i]).tolist(),
             'target_final': final[i].tolist(), 'valid_final': not unavailable and bool(lengths[i] > 0)} for i in range(24)]
    dataset = data(fields, rows); binding = task()
    binding['inputs'] = [{'port': key, 'field': key, 'transforms': []} for key in ('sequence', 'initial', 'lengths', 'mask', 'reset')]
    binding['heads'][0].update(port='prediction', field='target', mask='valid')
    binding['heads'].append({'id': 'final', 'port': 'final', 'field': 'target_final', 'loss': 'mse',
                            'weight': .2, 'transforms': [], 'mask': 'valid_final'})
    split = make_split(dataset, {'method': 'external', 'roles': {'train': ['r'+str(i) for i in range(16)],
                      'validation': ['r'+str(i) for i in range(16, 20)], 'test': ['r'+str(i) for i in range(20, 24)]}, 'excluded': []})
    return task_bundle(model_document(models, mid), dataset, binding, split,
                       training_protocol(epochs=None, updates=updates, batch_size=16, lr=.04, evaluate_every=20))


class RecurrentTaskExecutionTests(unittest.TestCase):
    def test_bounded_real_fit_and_worker_checkpoint_metrics_match_exactly(self):
        bundle = recurrent_bundle(); before = copy.deepcopy(bundle)
        local = train_task(bundle, constrained=False)
        self.assertEqual(local['status'], 'completed', local['failure'])
        self.assertEqual(local['counts']['completed_updates'], 40)
        self.assertLess(local['final_metrics']['train']['objective'], local['history'][0]['metrics']['train']['objective']*.15)
        self.assertEqual(len(local['checkpoint']['weights']), 2)
        self.assertEqual(set(local['checkpoint']['parameter_ids'].values()), {'sequence-model/scan/mix/weight', 'sequence-model/scan/mix/bias'})
        check_task_checkpoint(bundle, local['checkpoint'], constrained=False)
        points = []; worker = run_json_worker([sys.executable, '-u', '-m', 'app.worker'],
                 {'kind': 'task-v1', 'bundle': bundle, 'constrained': False}, points.append, timeout=30)
        self.assertEqual(worker['checkpoint'], local['checkpoint']); self.assertEqual(worker['history'], local['history'])
        self.assertEqual(points, worker['history']); self.assertEqual(worker['final_metrics'], local['final_metrics'])
        self.assertEqual(bundle, before)

    def test_empty_supervision_skips_optimizer_without_mistaking_padding_for_loss(self):
        bundle = recurrent_bundle(unavailable=True); result = train_task(bundle, constrained=False)
        self.assertEqual(result['status'], 'unavailable'); self.assertEqual(result['stop_reason'], 'no_training_supervision')
        self.assertEqual(result['counts']['completed_updates'], 0)
        self.assertIsNone(result['final_metrics']['train']['objective'])
        check_task_checkpoint(bundle, result['checkpoint'], constrained=False)

    def test_checkpoint_predictor_fresh_export_matches_independent_recurrence(self):
        bundle = recurrent_bundle(updates=8); result = train_task(bundle, constrained=False)
        records = [r['values'] for r in bundle['dataset']['records'][-4:]]
        predictor = restore_task_predictor(bundle, result['checkpoint'], constrained=False)
        expected = predictor(records)
        params = [torch.tensor(result['checkpoint']['weights'][k]) for k in ('layer_0.transition.layer_0.weight', 'layer_0.transition.layer_0.bias')]
        y, final = oracle(torch.tensor([r['sequence'] for r in records]), torch.tensor([r['initial'] for r in records]), *params,
                    lengths=torch.tensor([r['lengths'][0] for r in records]).long(),
                    mask=torch.tensor([r['mask'] for r in records]).bool(), reset=torch.tensor([r['reset'] for r in records]).bool())
        torch.testing.assert_close(expected['prediction'], y, rtol=3e-6, atol=2e-6)
        torch.testing.assert_close(expected['final'], final, rtol=3e-6, atol=2e-6)
        source = generate_task_predictor(bundle, result['checkpoint'], constrained=False)
        with tempfile.TemporaryDirectory() as directory:
            probe = '\n'.join([source, 'import json,sys', 'out=predict_records(' + repr(records) + ')',
                               'assert not any(n=="app" or n.startswith("app.") for n in sys.modules)',
                               'print(json.dumps({k:v.tolist() for k,v in out.items()}))'])
            run = subprocess.run([sys.executable, '-c', probe], cwd=directory, capture_output=True, text=True, timeout=30)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(json.loads(run.stdout), {k: v.tolist() for k, v in expected.items()})

    def test_actual_saved_and_rerun_report_kernels_resolve_v3_model_and_captured_helpers(self):
        from app.experiment_objects import execute_candidate, object_reference, preview_experiment
        from app.notebooks import create_bundle, execute_bundle, verify_bundle
        bundle = recurrent_bundle(updates=2); w = Workspace(); self.addCleanup(w.db.close)
        self.addCleanup(lambda: w.history.directory.cleanup() if w.history.directory else None)
        w.set_constrained_mode(False)
        refs = {kind: w.put_spec(kind, bundle[kind]) for kind in ('model', 'dataset', 'task')}
        experiment = {'version': 1, 'id': 'recurrent', 'name': 'Recurrent trial', 'question': 'Does the concrete recurrent graph replay?',
                      'models': [refs['model']], 'data_tasks': [{'id': 'pair', 'dataset': refs['dataset'], 'task': refs['task']}],
                      'splits': [{'id': 'split', 'protocol': {'method': 'row', 'seed': 31, 'fractions': [.65, .15, .2]}}],
                      'initialization_seeds': [7], 'protocols': [{'id': 'adam', 'protocol': bundle['protocol']}],
                      'selection': {'kind': 'crossed', 'pairs': [], 'exclusions': []}}
        ref = w.put_spec('experiment', experiment); frozen = w.get_spec('experiment', ref['id'], ref['revision'])
        selected = preview_experiment(frozen, w.get_spec)['candidates'][0]
        self.assertEqual(selected['status'], 'compatible')
        run = execute_candidate(w, object_reference('experiment', frozen), selected['coordinates'], 'run', 'Actual recurrent run', timeout=30)
        attempt = run['record']['attempts'][-1]
        self.assertEqual(attempt['status'], 'completed', attempt['error'])
        with tempfile.TemporaryDirectory() as directory:
            views = []
            for mode in ('result-report', 'rerun'):
                recipe = {'version': 1, 'id': mode, 'name': 'Recurrent evidence', 'question': 'Can this run be reproduced?',
                          'mode': mode, 'runs': [run['reference']] if mode == 'result-report' else [],
                          'experiment': object_reference('experiment', frozen) if mode == 'rerun' else None,
                          'sections': ['models', 'training', 'evaluation', 'checkpoints', 'failures'],
                          'analysis': {'kind': 'descriptive', 'inferential_unit': 'run'}}
                report = w.put_spec('report', recipe)
                exported = create_bundle(w, report, Path(directory) / mode)
                verify_bundle(exported['path'])
                result = execute_bundle(exported['path'], timeout=60)
                self.assertEqual(result['execution']['status'], 'completed', result['execution']['error'])
                view = json.loads((Path(result['path']) / 'report-data.json').read_text())
                self.assertEqual(len(view['runs']), 1)
                self.assertEqual(view['runs'][0]['status'], 'completed')
                self.assertEqual(view['runs'][0]['checkpoint_id'], attempt['result']['checkpoint_id'])
                self.assertEqual(view['notebook_execution']['training_requested'], mode == 'rerun')
                views.append(view)
            self.assertEqual(views[0]['runs'][0]['metrics'], views[1]['runs'][0]['metrics'])


if __name__ == '__main__': unittest.main()
