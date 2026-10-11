"""Real task worker, independent optimizer oracle and frozen inference checks."""
import copy
import io
import json
import math
import random
import re
import subprocess
import sys
import tempfile
import threading
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import torch

from app.core import GraphError, Workspace, default_model, default_cnn, compile_model
from app.datasets import make_split, prepare
from app.evidence import digest
from app.revisions import model_document
from app.task_execution import (check_task_checkpoint, fitted_inputs, restore_task_predictor,
                                generate_task_predictor, resolve_run, task_bundle, training_protocol, validate_protocol)
from app.task_training import train_task
from app.transport import run_json_worker, WorkerError
from test_datasets import data, task, manifest
import vpw_cli

torch.set_num_threads(1)


def bundle(model=None, dataset=None, binding=None, protocol=None, split=None):
    m = model or default_model(inputs=1, outputs=1); d = dataset or data(); t = binding or task()
    return task_bundle(model_document({m['id']: m}, m['id']), d, t, split or manifest(d),
                       protocol or training_protocol(epochs=2, batch_size=2))


def classification():
    model = default_model(inputs=1, outputs=2)
    fields = [{'id': 'x', 'dtype': 'float32', 'shape': [1]},
              {'id': 'label', 'dtype': 'category', 'shape': [], 'vocabulary': {'version': 1, 'values': ['negative', 'positive']}}]
    values = [{'x': [(i - 11.5) / 5.], 'label': 'negative' if i < 12 else 'positive'} for i in range(24)]
    d = data(fields, values); t = task(); t['heads'][0].update(field='label', loss='cross-entropy')
    s = make_split(d, {'method': 'row', 'seed': 31, 'fractions': [.65, .15, .2]})
    return bundle(model, d, t, training_protocol(epochs=60, batch_size=5, lr=.05, evaluate_every=30), s)


class TaskTrainingTests(unittest.TestCase):
    def test_documented_protocol_has_exact_executable_defaults(self):
        text = (Path(__file__).resolve().parents[1] / 'docs/task-training-contract.md').read_text()
        example = json.loads(re.findall(r'```json\n(.*?)```', text, re.S)[0])
        self.assertEqual(validate_protocol(example), training_protocol())

    def test_closed_protocol_rejects_unknown_and_invalid_before_torch_allocation(self):
        p = training_protocol()
        for name in ('boolean', 'seed', 'batch', 'lr', 'budget', 'eps', 'betas', 'backend', 'augment', 'selection', 'unknown'):
            bad = copy.deepcopy(p)
            if name == 'boolean': bad['version'] = True
            if name == 'seed': bad['seeds']['initialization'] = -1
            if name == 'batch': bad['batch_size'] = 0
            if name == 'lr': bad['optimizer']['lr'] = float('nan')
            if name == 'budget': bad['budget'] = {k: None for k in bad['budget']}
            if name == 'eps': bad['optimizer']['eps'] = 0
            if name == 'betas': bad['optimizer']['betas'] = [.9, 1.]
            if name == 'backend': bad['backend'] = 'invented-cuda'
            if name == 'augment': bad['augmentation'] = 'guess'
            if name == 'selection': bad['checkpoint_selection'] = 'best-test'
            if name == 'unknown': bad['extra'] = 1
            with self.subTest(name=name), self.assertRaises(GraphError): validate_protocol(bad)

    def test_exact_epochs_updates_examples_and_partial_batches(self):
        d = data(); s = manifest(d, train=['r0', 'r1', 'r2', 'r3', 'r4'], validation=['r5'], test=['r6', 'r7'])
        for p, expected, reason in ((training_protocol(epochs=2, batch_size=3), (4, 10, 2), 'epochs_budget'),
                (training_protocol(epochs=None, updates=3, batch_size=3), (3, 8, 1), 'updates_budget'),
                (training_protocol(epochs=None, examples=7, batch_size=3), (3, 7, 1), 'examples_budget')):
            r = train_task(bundle(dataset=d, protocol=p, split=s), constrained=False)
            self.assertEqual(r['status'], 'completed', r['failure']); self.assertEqual(r['stop_reason'], reason)
            self.assertEqual(tuple(r['counts'][k] for k in ('completed_updates', 'examples_processed', 'completed_epochs')), expected)
            self.assertEqual(r['counts']['updated_examples'], expected[1]); self.assertEqual(r['counts']['skipped_batches'], 0)
            self.assertEqual(r['history'][-1]['update'], expected[0])
            self.assertEqual(r['history'][-1]['checkpoint_id'], r['checkpoint_id'])
            self.assertNotIn('test', r['history'][-1]['metrics']); self.assertIn('test', r['final_metrics'])

    def test_time_budget_uses_observed_clock_and_keeps_zero_update_checkpoint(self):
        b = bundle(protocol=training_protocol(epochs=None, seconds=.5))
        calls = iter((0., 1., 1.5))
        r = train_task(b, constrained=False, clock=lambda: next(calls))
        self.assertEqual(r['stop_reason'], 'time_budget'); self.assertEqual(r['counts']['completed_updates'], 0)
        check_task_checkpoint(b, r['checkpoint'], constrained=False)
        self.assertEqual(r['elapsed_seconds'], 1.5)

    def test_cancel_after_progress_preserves_actual_updates_and_does_not_inspect_test(self):
        cancel = threading.Event(); points = []
        def progress(point):
            points.append(point); cancel.set()
        b = bundle(protocol=training_protocol(epochs=100, batch_size=1))
        r = train_task(b, progress=progress, cancel=cancel, constrained=False)
        self.assertEqual(r['status'], 'cancelled'); self.assertEqual(r['counts']['completed_updates'], 1)
        self.assertEqual(r['counts']['examples_processed'], 1); self.assertEqual(len(points), 1)
        self.assertNotIn('test', r['final_metrics']); check_task_checkpoint(b, r['checkpoint'], constrained=False)

    def test_cancel_before_update_has_valid_initial_checkpoint(self):
        c = threading.Event(); c.set(); b = bundle()
        r = train_task(b, cancel=c, constrained=False)
        self.assertEqual(r['status'], 'cancelled'); self.assertEqual(r['counts']['attempted_batches'], 0)
        self.assertEqual(r['history'][0]['update'], 0); self.assertNotIn('test', r['final_metrics'])

    def test_unavailable_training_supervision_is_bounded_and_not_zero_loss_success(self):
        d = data(); d['fields'].append({'id': 'mask', 'dtype': 'bool', 'shape': []})
        for row in d['records']: row['values']['mask'] = False
        t = task(); t['heads'][0]['mask'] = 'mask'
        r = train_task(bundle(dataset=d, binding=t, protocol=training_protocol(epochs=None, updates=100, batch_size=1)), constrained=False)
        self.assertEqual(r['status'], 'unavailable'); self.assertEqual(r['stop_reason'], 'no_training_supervision')
        self.assertEqual(r['counts']['completed_updates'], 0); self.assertEqual(r['counts']['skipped_batches'], 2)
        self.assertIsNone(r['final_metrics']['train']['objective']); self.assertNotIn('test', r['final_metrics'])

    def test_skipped_batches_do_not_count_as_updates_but_count_seen_examples(self):
        d = data(); d['fields'].append({'id': 'mask', 'dtype': 'bool', 'shape': []})
        for i, row in enumerate(d['records']): row['values']['mask'] = i != 0
        t = task(); t['heads'][0]['mask'] = 'mask'
        p = training_protocol(epochs=2, batch_size=1, shuffle=False)
        r = train_task(bundle(dataset=d, binding=t, protocol=p), constrained=False)
        self.assertEqual(r['counts'], {'completed_updates': 2, 'attempted_batches': 4, 'examples_processed': 4,
                                     'updated_examples': 2, 'skipped_batches': 2, 'completed_epochs': 2})
        self.assertEqual(r['history'][-1]['attempted_batches'], 4)

    def test_nonfinite_optimizer_returns_failed_evidence_and_restores_last_finite_weights(self):
        b = bundle(protocol=training_protocol(epochs=3, batch_size=1))
        initial = train_task(b, cancel=type('Cancel', (), {'is_set': lambda _: True})(), constrained=False)['checkpoint']
        def corrupt(opt):
            with torch.no_grad(): opt.param_groups[0]['params'][0].fill_(float('nan'))
        with patch('torch.optim.Adam.step', corrupt): r = train_task(b, constrained=False)
        self.assertEqual(r['status'], 'failed'); self.assertIn('nonfinite', r['failure'])
        self.assertEqual(r['counts']['completed_updates'], 0); self.assertEqual(r['counts']['attempted_batches'], 1)
        self.assertEqual(r['checkpoint']['weights'], initial['weights']); self.assertNotIn('test', r['final_metrics'])
        check_task_checkpoint(b, r['checkpoint'], constrained=False); json.dumps(r, allow_nan=False)

    def test_bad_loss_after_one_update_keeps_successful_checkpoint_and_actual_counts(self):
        from app.task_runtime import head_terms
        calls = 0
        def fail_later(*args):
            nonlocal calls
            calls += 1
            if calls == 4: raise GraphError('injected later loss failure')
            return head_terms(*args)
        b = bundle(protocol=training_protocol(epochs=20, batch_size=2))
        with patch('app.task_training.head_terms', side_effect=fail_later): r = train_task(b, constrained=False)
        self.assertEqual(r['status'], 'failed'); self.assertEqual(r['counts']['completed_updates'], 1)
        check_task_checkpoint(b, r['checkpoint'], constrained=False)

    def test_seed_streams_and_input_bundle_are_not_mutated(self):
        b = bundle(protocol=training_protocol(epochs=3, batch_size=1)); before = copy.deepcopy(b)
        py_state = random.getstate(); torch_state = torch.get_rng_state().clone()
        one = train_task(b, constrained=False); two = train_task(b, constrained=False)
        self.assertEqual(b, before); self.assertEqual(random.getstate(), py_state); self.assertTrue(torch.equal(torch.get_rng_state(), torch_state))
        self.assertEqual(one['checkpoint'], two['checkpoint']); self.assertEqual(one['history'], two['history'])
        changed = copy.deepcopy(b); changed['protocol']['seeds']['shuffle'] = 1
        # Seeds 11 and 12 happen to give the SAME three two-row permutations;
        # a different seed does not mathematically imply different samples.
        # Use a verified distinct order, and separately prove init independence.
        orders = []
        for seed in (11, 1):
            rng = random.Random(seed); epochs = []
            for _ in range(3):
                order = [0, 1]; rng.shuffle(order); epochs.append(order)
            orders.append(epochs)
        self.assertNotEqual(*orders)
        three = train_task(changed, constrained=False)
        self.assertNotEqual(one['identity'], three['identity']); self.assertNotEqual(one['checkpoint']['weights'], three['checkpoint']['weights'])
        cancel = threading.Event(); cancel.set()
        self.assertEqual(train_task(b, cancel=cancel, constrained=False)['checkpoint']['weights'],
                         train_task(changed, cancel=cancel, constrained=False)['checkpoint']['weights'])

    def test_training_float32_does_not_follow_global_default_dtype(self):
        b = bundle(); old = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float64); r = train_task(b, constrained=False)
            self.assertEqual(r['status'], 'completed', r['failure'])
            predict = restore_task_predictor(b, r['checkpoint'], constrained=False)
            self.assertEqual(predict([{'x': [1.]}])['output'].dtype, torch.float32)
        finally: torch.set_default_dtype(old)

    def test_combined_ram_admission_includes_optimizer_model_and_collation_before_compile(self):
        b = bundle()
        with patch('app.task_training.compile_task_model', side_effect=AssertionError('must not allocate')):
            with self.assertRaisesRegex(GraphError, 'RAM'): train_task(b, memory_limit_percent=.00001)
        with self.assertRaisesRegex(GraphError, 'RAM'): resolve_run(b, available_bytes=128 * 1024**2)
        _, m = resolve_run(b, constrained=False, available_bytes=128 * 1024**2)
        self.assertIn('typed_preparation', m['components']); self.assertIn('parameters_optimizer_evidence', m['components'])
        self.assertGreaterEqual(m['batch'], m['actual_batch_size'])

    def test_test_population_never_appears_in_training_curves(self):
        b = bundle(protocol=training_protocol(epochs=3, batch_size=1, evaluate_every=1))
        r = train_task(b, constrained=False)
        for point in r['history']:
            self.assertEqual(set(point['metrics']), {'train', 'validation'})
            for role, population in point['metrics'].items():
                metric = population['heads']['prediction']
                self.assertEqual(metric['observation_ids'], b['manifest']['roles'][role])
                self.assertEqual(metric['checkpoint_id'], point['checkpoint_id'])
        self.assertEqual(r['final_metrics']['test']['heads']['prediction']['observation_ids'], b['manifest']['roles']['test'])

    def test_fitted_inputs_restore_new_records_without_refitting_and_match_saved_metrics(self):
        d = data(); d['fields'][0]['nullable'] = True; d['records'][0]['values']['x'] = [None]
        transforms = [{'kind': 'impute-mean'}, {'kind': 'standardize'}, {'kind': 'affine', 'scale': 2., 'offset': 1.}]
        b = bundle(dataset=d, binding=task(transforms), protocol=training_protocol(epochs=4, batch_size=1))
        r = train_task(b, constrained=False); fitted = r['checkpoint']['fitted']; before = copy.deepcopy(fitted)
        # Only r1 is available when fitting imputation; both training values then
        # equal 1, so standardized constant feature -> 0 and affine -> 1.
        self.assertEqual(fitted_inputs(b, fitted, [{'x': [None]}, {'x': [2.]}])['input'], [[1.], [3.]])
        predictor = restore_task_predictor(b, r['checkpoint'], constrained=False)
        rows = {row['id']: row['values'] for row in d['records']}; ids = b['manifest']['roles']['test']
        predictions = predictor([rows[rid] for rid in ids])['output'].reshape(-1).tolist()
        expected = math.fsum((v - rows[rid]['y'][0])**2 for v, rid in zip(predictions, ids)) / len(ids)
        self.assertAlmostEqual(expected, r['final_metrics']['test']['heads']['prediction']['value'], places=4)
        self.assertEqual(fitted, before)
        with self.assertRaises(GraphError): predictor([{'unknown': [1.]}])

    def test_checkpoint_rejects_wrong_identity_shapes_keys_ids_fitted_and_nonfinite_weights(self):
        b = bundle(); r = train_task(b, constrained=False)
        for change in ('identity', 'fitted', 'keys', 'shape', 'ids', 'nonfinite', 'version', 'kind'):
            ck = copy.deepcopy(r['checkpoint']); key = next(iter(ck['weights']))
            if change == 'identity': ck['identity']['task']['revision'] = '0' * 64
            if change == 'fitted': ck['fitted']['fit_observation_ids'].reverse()
            if change == 'keys': ck['weights']['phantom'] = [1.]
            if change == 'shape': ck['weights'][key] = []
            if change == 'ids': ck['parameter_ids'][key] = 'wrong'
            if change == 'nonfinite': ck['weights'][key][0][0] = float('inf')
            if change == 'version': ck['version'] = True
            if change == 'kind': ck['kind'] = 'legacy-identity'
            with self.subTest(change=change), self.assertRaises(GraphError): check_task_checkpoint(b, ck, constrained=False)

    def test_classification_learns_and_returns_logits_in_fixed_vocabulary_order(self):
        b = classification(); r = train_task(b, constrained=False)
        self.assertEqual(r['status'], 'completed', r['failure'])
        metric = r['final_metrics']['test']['heads']['prediction']
        self.assertEqual(metric['accuracy'], 1.); self.assertLess(metric['value'], .3)
        predictor = restore_task_predictor(b, r['checkpoint'], constrained=False)
        logits = predictor([{'x': [-2.]}, {'x': [2.]}])['output']
        self.assertEqual(logits.argmax(dim=1).tolist(), [0, 1]); self.assertEqual(list(logits.shape), [2, 2])
        self.assertEqual(r['checkpoint']['fitted']['heads']['prediction']['vocabulary']['values'], ['negative', 'positive'])

    def test_multihead_masked_objective_and_adam_update_match_independent_scalar_oracle(self):
        m = default_model(inputs=1, outputs=1); m['graph']['nodes'] += [{'id': 'out2', 'type': 'Output', 'label': 'second'}]
        m['graph']['edges'] += [{'from': 'linear1', 'to': 'out2'}]
        d = data(); d['fields'] += [{'id': 'z', 'dtype': 'float32', 'shape': [1]}, {'id': 'mask', 'dtype': 'bool', 'shape': []},
                                   {'id': 'weight', 'dtype': 'float32', 'shape': []}]
        for i, row in enumerate(d['records']): row['values'].update(z=[float(i - 2)], mask=i != 0, weight=float(i + 1))
        t = task(); t['heads'][0].update(mask='mask', sample_weight='weight', weight=.25)
        t['heads'].append({'id': 'absolute', 'port': 'out2', 'field': 'z', 'loss': 'mae', 'weight': 2., 'transforms': []})
        b = bundle(m, d, t, training_protocol(epochs=2, batch_size=2, shuffle=False, lr=.01))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(7); native = compile_model(b['model']['models'], m['id'], constrained=False)
        w = float(native.layer_0.weight.detach()[0, 0]); bias = float(native.layer_0.bias.detach()[0])
        moments = [0., 0.]; variances = [0., 0.]
        # Two full-batch updates, independent scalar losses and Adam equations.
        for step in (1, 2):
            p0 = w * 0 + bias; p1 = w + bias
            # MSE head includes ONLY r1 with mask/weight=2. MAE includes both.
            sign0 = float(p0 > -2) - float(p0 < -2); sign1 = float(p1 > -1) - float(p1 < -1)
            gradients = [.25 * 2 * (p1 - 3) + sign1, .25 * 2 * (p1 - 3) + sign0 + sign1]
            updated = []
            for i, value in enumerate((w, bias)):
                moments[i] = .9 * moments[i] + .1 * gradients[i]
                variances[i] = .999 * variances[i] + .001 * gradients[i]**2
                updated.append(value - .01 * (moments[i] / (1 - .9**step)) / (math.sqrt(variances[i] / (1 - .999**step)) + 1e-8))
            w, bias = updated
        r = train_task(b, constrained=False)
        self.assertEqual(r['status'], 'completed', r['failure'])
        self.assertAlmostEqual(r['checkpoint']['weights']['layer_0.weight'][0][0], w, places=6)
        self.assertAlmostEqual(r['checkpoint']['weights']['layer_0.bias'][0], bias, places=6)
        metrics = r['final_metrics']['train']
        self.assertAlmostEqual(metrics['objective'], .25 * metrics['heads']['prediction']['value'] + 2 * metrics['heads']['absolute']['value'])

    def test_unavailable_one_head_does_not_renormalize_other_head_weight(self):
        m = default_model(inputs=1); m['graph']['nodes'] += [{'id': 'out2', 'type': 'Output', 'label': 'second'}]
        m['graph']['edges'] += [{'from': 'linear1', 'to': 'out2'}]
        d = data(); d['fields'].append({'id': 'mask', 'dtype': 'bool', 'shape': []})
        for row in d['records']: row['values']['mask'] = False
        t = task(); t['heads'][0].update(mask='mask', weight=5.)
        t['heads'].append({'id': 'other', 'port': 'out2', 'field': 'y', 'loss': 'mse', 'weight': .125, 'transforms': []})
        r = train_task(bundle(m, d, t), constrained=False)
        metrics = r['final_metrics']['train']; self.assertEqual(metrics['heads']['prediction']['status'], 'unavailable')
        self.assertAlmostEqual(metrics['objective'], .125 * metrics['heads']['other']['value'])

    def test_spatial_cnn_cross_entropy_masks_each_site_and_preserves_axes(self):
        from test_cnn import chain
        m = chain([1, 2, 2], {'type': 'Conv2d', 'out_channels': 2, 'kernel_size': 1}, axes=['C', 'H', 'W'])
        fields = [{'id': 'x', 'dtype': 'float32', 'shape': [1, 2, 2], 'axes': ['C', 'H', 'W']},
                  {'id': 'label', 'dtype': 'category', 'shape': [2, 2], 'vocabulary': {'version': 1, 'values': ['a', 'b']}},
                  {'id': 'mask', 'dtype': 'bool', 'shape': [2, 2]}]
        rows = [{'x': [[[-1., 1.], [-2., 2.]]], 'label': [['a', 'b'], ['a', 'b']], 'mask': [[True, False], [True, True]]} for _ in range(8)]
        d = data(fields, rows); t = task(); t['heads'][0].update(field='label', loss='cross-entropy', mask='mask')
        b = bundle(m, d, t, training_protocol(epochs=60, batch_size=2, lr=.05, evaluate_every=60))
        r = train_task(b, constrained=False); self.assertEqual(r['status'], 'completed', r['failure'])
        metric = r['final_metrics']['test']['heads']['prediction']
        self.assertEqual(metric['accuracy'], 1.); self.assertEqual(metric['denominator'], 3 * len(b['manifest']['roles']['test']))
        self.assertEqual(r['checkpoint']['parameter_ids']['layer_0.weight'], m['id'] + '/op0/weight')

    def test_real_worker_classification_roundtrip_matches_inprocess_without_torch_in_host(self):
        b = classification(); b['protocol']['budget']['epochs'] = 5
        local = train_task(b, constrained=False); points = []
        worker = run_json_worker([sys.executable, '-u', '-m', 'app.worker'],
                                {'kind': 'task-v1', 'bundle': b, 'constrained': False}, points.append, timeout=30)
        self.assertEqual(worker['checkpoint'], local['checkpoint']); self.assertEqual(worker['history'], local['history'])
        self.assertEqual(points, worker['history']); self.assertEqual(worker['final_metrics'], local['final_metrics'])
        self.assertEqual(worker['environment']['numerical_threads'], 1)

    def test_real_worker_rejects_tampered_revision_and_unknown_dispatch(self):
        b = bundle(); b['dataset']['records'][0]['values']['x'] = [99.]
        for request in ({'kind': 'task-v1', 'bundle': b}, {'kind': 'made-up-v2', 'models': {}, 'id': 'absent'}):
            with self.assertRaises(WorkerError): run_json_worker([sys.executable, '-u', '-m', 'app.worker'], request, timeout=15)

    def test_real_worker_cancellation_preserves_terminal_result_and_reaps_child(self):
        b = bundle(protocol=training_protocol(epochs=100000, batch_size=1, evaluate_every=1, history_limit=2))
        cancel = threading.Event(); processes = []
        r = run_json_worker([sys.executable, '-u', '-m', 'app.worker'],
                           {'kind': 'task-v1', 'bundle': b, 'constrained': False},
                           lambda _: cancel.set(), cancel, timeout=20, on_process=processes.append)
        self.assertEqual(r['status'], 'cancelled'); self.assertGreaterEqual(r['counts']['completed_updates'], 1)
        self.assertLess(r['counts']['completed_updates'], 200000); self.assertNotIn('test', r['final_metrics'])
        self.assertEqual(processes[0].poll(), 0); check_task_checkpoint(b, r['checkpoint'], constrained=False)

    def test_cli_exact_stored_train_uses_worker_leaves_workspace_bytes_and_no_host_torch(self):
        from app.config import Config
        with tempfile.TemporaryDirectory() as td:
            directory = Path(td); path = directory / 'training.vtw'; w = Workspace(); m = default_model(inputs=1)
            w.put(m); d = data(); t = task()
            refs = {'model': w.freeze_model(m['id']), 'dataset': w.put_spec('dataset', d), 'task': w.put_spec('task', t)}
            w.set_constrained_mode(False); w.save(path); w.db.close(); before = path.read_bytes()
            protocol = directory / 'protocol.json'; protocol.write_text(json.dumps(training_protocol(epochs=2)))
            split = directory / 'split.json'; split.write_text(json.dumps({'method': 'row', 'seed': 4, 'fractions': [.5, .25, .25]}))
            args = [str(path), 'train-stored', '--protocol', str(protocol), '--split', str(split), '--memory-limit-percent', '10']
            for kind, ref in refs.items(): args += ['--' + kind, ref['id'], ref['revision']]
            code = "import sys,json,vpw_cli; code=vpw_cli.main(sys.argv[1:]); print(json.dumps({'host_torch': 'torch' in sys.modules, 'host_qt': any(k.startswith('PyQt') for k in sys.modules)})); sys.exit(code)"
            result = subprocess.run([sys.executable, '-c', code, *args], capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            lines = result.stdout.splitlines(); value = json.loads(lines[0])['value']
            self.assertEqual(value['status'], 'completed'); self.assertEqual(json.loads(lines[1]), {'host_torch': False, 'host_qt': False})
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(value['identity']['model'], refs['model'])

    def test_export_component_imports_and_validates_without_torch_qt_numpy(self):
        code = "import sys,json; import app.task_execution; print(json.dumps([k for k in sys.modules if k=='torch' or k=='numpy' or k.startswith('PyQt')]))"
        r = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=10)
        self.assertEqual(r.returncode, 0, r.stderr); self.assertEqual(json.loads(r.stdout), [])

    def test_standalone_export_runs_from_unrelated_directory_with_no_app_import_or_refit(self):
        b = bundle(binding=task([{'kind': 'standardize'}]), protocol=training_protocol(epochs=3))
        r = train_task(b, constrained=False); source = generate_task_predictor(b, r['checkpoint'], constrained=False)
        records = [{'x': [100.]}, {'x': [-2.]}]
        expected = restore_task_predictor(b, r['checkpoint'], constrained=False)(records)['output'].tolist()
        self.assertNotIn("from app", source); self.assertNotIn("'records':", source)
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / 'predictor.py'; p.write_text(source)
            code = "import predictor,json,sys; out=predictor.predict_records(" + repr(records) + "); print(json.dumps({'prediction':out['output'].tolist(), 'app_imported':any(k=='app' or k.startswith('app.') for k in sys.modules)}))"
            process = subprocess.run([sys.executable, '-c', code], cwd=td, capture_output=True, text=True, timeout=20)
            self.assertEqual(process.returncode, 0, process.stderr)
            self.assertEqual(json.loads(process.stdout), {'prediction': expected, 'app_imported': False})

    def test_standalone_classification_export_agrees_bitwise_and_keeps_logit_semantics(self):
        b = classification(); b['protocol']['budget']['epochs'] = 5; r = train_task(b, constrained=False)
        namespace = {}; exec(generate_task_predictor(b, r['checkpoint'], constrained=False), namespace)
        records = [{'x': [-3.]}, {'x': [4.]}]
        actual = namespace['predict_records'](records)['output']
        expected = restore_task_predictor(b, r['checkpoint'], constrained=False)(records)['output']
        self.assertTrue(torch.equal(actual, expected)); self.assertFalse(torch.allclose(actual.sum(1), torch.ones(2)))

    def test_ragged_pad_derived_mask_and_onehot_new_inference_values_match_preparation(self):
        m = default_model(inputs=1)
        m['graph'] = {'nodes': [{'id': 'seq', 'type': 'Input', 'shape': [3]},
            {'id': 'mask', 'type': 'Input', 'shape': [3]}, {'id': 'category', 'type': 'Input', 'shape': [2]},
            {'id': 'join', 'type': 'Concat', 'sources': ['seq', 'mask', 'category'], 'flatten': True},
            {'id': 'linear', 'type': 'Linear', 'features': 1}, {'id': 'output', 'type': 'Output'}],
            'edges': [{'from': a, 'to': b} for a, b in [('seq', 'join'), ('mask', 'join'), ('category', 'join'), ('join', 'linear'), ('linear', 'output')]]}
        fields = [{'id': 's', 'dtype': 'float32', 'shape': [-1]}, {'id': 'c', 'dtype': 'category', 'shape': [],
            'vocabulary': {'version': 1, 'values': ['left', 'right']}}, {'id': 'y', 'dtype': 'float32', 'shape': [1]}]
        d = data(fields, [{'s': [float(i)] * (i % 3 + 1), 'c': 'left' if i % 2 else 'right', 'y': [float(i)]} for i in range(8)])
        t = task(); t['inputs'] = [{'port': 'seq', 'field': 's', 'transforms': [{'kind': 'pad', 'length': 3, 'value': 0.}]},
            {'port': 'mask', 'field': 's', 'derived': 'mask', 'length': 3, 'transforms': []},
            {'port': 'category', 'field': 'c', 'transforms': [{'kind': 'onehot'}]}]
        b = bundle(m, d, t); r = train_task(b, constrained=False)
        rows = [{'s': [8.], 'c': 'left'}, {'s': [], 'c': 'right'}]
        columns = fitted_inputs(b, r['checkpoint']['fitted'], rows)
        self.assertEqual(columns, {'seq': [[8., 0., 0.], [0., 0., 0.]], 'mask': [[1., 0., 0.], [0., 0., 0.]], 'category': [[1., 0.], [0., 1.]]})
        native = restore_task_predictor(b, r['checkpoint'], constrained=False); namespace = {}
        exec(generate_task_predictor(b, r['checkpoint'], constrained=False), namespace)
        self.assertTrue(torch.equal(native(rows)['output'], namespace['predict_records'](rows)['output']))
        for bad in ({'s': [1., 2., 3., 4.], 'c': 'left'}, {'s': [1.], 'c': 'unknown'}, {'s': None, 'c': 'right'}):
            with self.assertRaises(GraphError): native([bad])

    def test_cli_task_export_is_readonly_and_emits_real_standalone_source(self):
        with tempfile.TemporaryDirectory() as td:
            directory = Path(td); path = directory / 'lab.vtw'; w = Workspace(); w.set_constrained_mode(False); w.save(path); w.db.close()
            before = path.read_bytes(); b = bundle(); r = train_task(b, constrained=False)
            result = directory / 'result.json'; result.write_text(json.dumps(r)); stream = io.StringIO()
            with redirect_stdout(stream): code = vpw_cli.main([str(path), 'export-task-trained', str(result)])
            self.assertEqual(code, 0); source = json.loads(stream.getvalue())['value']['code']; namespace = {}; exec(source, namespace)
            self.assertIn('output', namespace['predict_records']([{'x': [1.]}])); self.assertEqual(path.read_bytes(), before)

    def test_curve_retention_is_bounded_explicit_and_preserves_final_checkpoint(self):
        b = bundle(protocol=training_protocol(epochs=4, batch_size=1, evaluate_every=1, history_limit=2))
        points = []; r = train_task(b, progress=points.append, constrained=False)
        self.assertEqual(len(points), 8); self.assertEqual(r['history'], points[-2:])
        self.assertEqual(r['history_retention'], {'kind': 'latest-window', 'max_points': 2, 'observed_points': 8, 'discarded_points': 6})
        self.assertEqual(r['history'][-1]['checkpoint_id'], r['checkpoint_id'])
        _, small = resolve_run(b, constrained=False)
        b['protocol']['history']['max_points'] = 200
        _, large = resolve_run(b, constrained=False)
        self.assertGreater(large['estimated_bytes'], small['estimated_bytes'])
        b['protocol']['history']['max_points'] = 0
        with self.assertRaises(GraphError): resolve_run(b, constrained=False)

    def test_initial_checkpoint_timing_and_empty_validation_are_explicit(self):
        d = data(); s = manifest(d, validation=[], test=['r2', 'r3', 'r4', 'r5', 'r6', 'r7'])
        b = bundle(dataset=d, split=s); cancel = threading.Event(); cancel.set()
        r = train_task(b, cancel=cancel, constrained=False)
        self.assertEqual(r['history'][0]['timing'], 'initial')
        metric = r['final_metrics']['validation']['heads']['prediction']
        self.assertEqual(metric['evaluated_records'], 0); self.assertEqual(metric['status'], 'unavailable')
        self.assertIsNone(metric['value']); json.dumps(r, allow_nan=False)

    def test_task_checkpoint_does_not_accept_boolean_alias_of_numeric_identity_version(self):
        b = bundle(); r = train_task(b, constrained=False)
        r['checkpoint']['identity']['version'] = True
        with self.assertRaisesRegex(GraphError, 'identity'): check_task_checkpoint(b, r['checkpoint'], constrained=False)

    def test_prediction_arithmetic_overflow_is_an_error_not_silent_nonfinite_output(self):
        b = bundle(); r = train_task(b, constrained=False)
        r['checkpoint']['weights']['layer_0.weight'] = [[3.e38]]
        native = restore_task_predictor(b, r['checkpoint'], constrained=False)
        with self.assertRaisesRegex(GraphError, 'nonfinite'): native([{'x': [3.e38]}])

    def test_multi_input_independent_branches_join_and_export_preserve_stable_port_meaning(self):
        m = default_model(); m['graph'] = {'nodes': [
            {'id': 'a', 'type': 'Input', 'features': 1, 'label': 'first'},
            {'id': 'b', 'type': 'Input', 'features': 1, 'label': 'second'},
            {'id': 't', 'type': 'Linear', 'features': 2}, {'id': 'ta', 'type': 'Activation', 'activation': 'Tanh'},
            {'id': 'r', 'type': 'Linear', 'features': 2}, {'id': 'ra', 'type': 'Activation', 'activation': 'ReLU'},
            {'id': 'join', 'type': 'Concat', 'sources': ['ta', 'ra'], 'flatten': True},
            {'id': 'end', 'type': 'Linear', 'features': 1}, {'id': 'output', 'type': 'Output'}],
            'edges': [{'from': a, 'to': b} for a, b in [('a', 't'), ('t', 'ta'), ('b', 'r'), ('r', 'ra'),
                ('ta', 'join'), ('ra', 'join'), ('join', 'end'), ('end', 'output')]]}
        fields = [{'id': 'u', 'dtype': 'float32', 'shape': [1]}, {'id': 'v', 'dtype': 'float32', 'shape': [1]},
                  {'id': 'y', 'dtype': 'float32', 'shape': [1]}]
        d = data(fields, [{'u': [i / 8.], 'v': [1. - i / 5.], 'y': [i / 3.]} for i in range(8)])
        t = task(); t['inputs'] = [{'port': 'a', 'field': 'u', 'transforms': []}, {'port': 'b', 'field': 'v', 'transforms': []}]
        b = bundle(m, d, t, training_protocol(epochs=3, batch_size=2)); r = train_task(b, constrained=False)
        self.assertEqual(r['status'], 'completed', r['failure'])
        rows = [{'u': [0.], 'v': [1.]}, {'u': [1.], 'v': [-1.]}]
        namespace = {}; exec(generate_task_predictor(b, r['checkpoint'], constrained=False), namespace)
        expected = restore_task_predictor(b, r['checkpoint'], constrained=False)(rows)['output']
        self.assertTrue(torch.equal(expected, namespace['predict_records'](rows)['output']))
        self.assertEqual(set(r['checkpoint']['fitted']['inputs']), {'a', 'b'})


if __name__ == '__main__':
    unittest.main()
