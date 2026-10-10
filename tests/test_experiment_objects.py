"""Frozen selection and terminal storage against real task-worker evidence."""
import copy
import io
import json
import random
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from unittest.mock import patch

from app.core import GraphError, Workspace, VERSION, default_model, default_mlp
from app.evidence import digest
from app.experiment_objects import (candidate, candidate_bundle, execute_candidate, object_reference,
                                    preview_experiment, report_data, validate_experiment, validate_report)
from app.revisions import frozen_document, validate_all
from app.task_execution import training_protocol
from app.task_result import validate_result
from test_datasets import data, task
import vpw_cli


def experiment_workspace():
    w = Workspace(); w.set_constrained_mode(False)
    a = default_model('Linear', inputs=1); b = default_mlp('Tanh MLP', inputs=1, hidden=(3,))
    w.put(a); w.put(b)
    models = [w.freeze_model(m['id']) for m in (a, b)]
    pairs = []
    for i in range(2):
        d = data(); d['id'] = 'dataset' + str(i); d['name'] = 'Observations ' + str(i)
        for row in d['records']: row['values']['y'][0] += i
        t = task(); t['id'] = 'task' + str(i)
        pairs.append({'id': 'pair' + str(i), 'dataset': w.put_spec('dataset', d), 'task': w.put_spec('task', t)})
    e = {'version': 1, 'id': 'experiment', 'name': 'Linear versus Tanh', 'question': 'How do these families compare under a fixed update budget?',
        'models': models, 'data_tasks': pairs, 'splits': [{'id': 'split-a', 'protocol': {'method': 'row', 'seed': 4, 'fractions': [.5, .25, .25]}}],
        'initialization_seeds': [7, 13], 'protocols': [{'id': 'adam', 'protocol': training_protocol(epochs=None, updates=3, batch_size=3, evaluate_every=2)}],
        'selection': {'kind': 'crossed', 'pairs': [], 'exclusions': []}}
    ref = w.put_spec('experiment', e)
    return w, w.get_spec('experiment', ref['id'], ref['revision'])


class ExperimentObjectTests(unittest.TestCase):
    def setUp(self):
        self.w, self.e = experiment_workspace()
        self.addCleanup(self.w.db.close)
        self.addCleanup(lambda: self.w.history.directory.cleanup() if self.w.history.directory else None)

    def first(self):
        return preview_experiment(self.e, self.w.get_spec)['candidates'][0]

    def execute(self, selected=None, run_id='run', **kwargs):
        row = selected or self.first()
        return execute_candidate(self.w, object_reference('experiment', self.e), row['coordinates'], run_id, 'Trial', timeout=20, **kwargs)

    def test_two_models_two_datasets_two_replicates_have_exact_reproducible_coordinates(self):
        before = random.getstate(); page = preview_experiment(self.e, self.w.get_spec)
        self.assertEqual(page, preview_experiment(self.e, self.w.get_spec)); self.assertEqual(random.getstate(), before)
        self.assertEqual(page['total_combinations'], 8); self.assertEqual(len(page['candidates']), 8)
        self.assertEqual(len({r['id'] for r in page['candidates']}), 8)
        self.assertEqual({r['status'] for r in page['candidates']}, {'compatible'})
        for row in page['candidates']:
            self.assertEqual(row['protocol']['seeds']['initialization'], row['coordinates']['initialization'])
            self.assertEqual(row['protocol']['seeds']['shuffle'], 11)
            dataset = self.w.get_spec('dataset', row['references']['dataset']['id'], row['references']['dataset']['revision'])
            self.assertEqual(row['manifest']['dataset_revision'], dataset['revision'])
        a = page['candidates'][0]; b = page['candidates'][4]
        self.assertEqual(a['manifest'], b['manifest'])  # same dataset/split across models

    def test_preview_is_paged_without_fitting_training_or_global_rng(self):
        with patch('app.datasets._prepare_column', side_effect=AssertionError('preview must not fit')):
            a = preview_experiment(self.e, self.w.get_spec, limit=3)
            b = preview_experiment(self.e, self.w.get_spec, offset=a['next_offset'], limit=3)
        self.assertEqual(a['next_offset'], 3); self.assertEqual(b['next_offset'], 6)
        self.assertEqual(a['candidates'] + b['candidates'], preview_experiment(self.e, self.w.get_spec)['candidates'][:6])
        self.assertEqual(preview_experiment(self.e, self.w.get_spec, offset=10**10)['candidates'], [])
        for args in ({'offset': -1}, {'limit': True}, {'limit': 1001}):
            with self.assertRaises(GraphError): preview_experiment(self.e, self.w.get_spec, **args)

    def test_exclusions_and_declared_pairings_remain_visible(self):
        e = copy.deepcopy(self.e); e.pop('revision')
        e['selection'] = {'kind': 'paired', 'pairs': [{'model': e['models'][0], 'data_task': 'pair0'}],
                          'exclusions': [{'model': e['models'][0], 'data_task': 'pair0', 'reason': 'Deliberately excluded negative control'}]}
        ref = self.w.put_spec('experiment', e); e = self.w.get_spec('experiment', *[ref[k] for k in ('id', 'revision')])
        rows = preview_experiment(e, self.w.get_spec)['candidates']
        self.assertEqual({r['status'] for r in rows}, {'excluded'}); self.assertIn('negative control', rows[0]['reason'])
        self.assertIn('pairing', rows[-1]['reason'])

    def test_incompatible_shape_and_split_are_visible_not_silently_filtered(self):
        e = copy.deepcopy(self.e); e.pop('revision')
        bad = data(); bad['id'] = 'wrongshape'; bad['fields'][0]['shape'] = [2]
        for row in bad['records']: row['values']['x'] += [0.]
        e['data_tasks'][0]['dataset'] = self.w.put_spec('dataset', bad)
        e['splits'].append({'id': 'absent-group-field', 'protocol': {'method': 'group', 'field': 'nonexistent', 'seed': 4, 'fractions': [.5, .25, .25]}})
        ref = self.w.put_spec('experiment', e); e = self.w.get_spec('experiment', ref['id'], ref['revision'])
        rows = preview_experiment(e, self.w.get_spec)['candidates']
        self.assertEqual(len(rows), 16); self.assertIn('compatible', {r['status'] for r in rows}); self.assertIn('incompatible', {r['status'] for r in rows})
        self.assertIn('shape', rows[0]['reason']); self.assertTrue(all(r['reason'] for r in rows if r['status'] != 'compatible'))

    def test_intrinsic_experiment_schema_is_closed_and_references_are_exact(self):
        for change in ('duplicate', 'missing-ref', 'bad-seed', 'bad-pair', 'split', 'config', 'version', 'question'):
            bad = copy.deepcopy(self.e); bad.pop('revision')
            if change == 'duplicate': bad['models'] *= 2
            if change == 'missing-ref': bad['models'][0]['revision'] = '0' * 64
            if change == 'bad-seed': bad['initialization_seeds'][0] = True
            if change == 'bad-pair': bad['selection']['exclusions'] = [{'model': bad['models'][0], 'data_task': 'unknown', 'reason': 'reason'}]
            if change == 'split': bad['splits'][0]['protocol']['fractions'] = [.6, .2, .3]
            if change == 'config': bad['protocols'][0]['protocol']['optimizer']['lr'] = -1.
            if change == 'version': bad['version'] = True
            if change == 'question': bad['question'] = ''
            with self.subTest(change=change), self.assertRaises(GraphError): self.w.put_spec('experiment', bad)

    def test_reordering_browsing_heads_does_not_replace_frozen_selections(self):
        before = self.first(); pair = self.e['data_tasks'][0]; d = self.w.get_spec('dataset', pair['dataset']['id'])
        d.pop('revision'); d['records'][0]['values']['x'] = [99.]; self.w.put_spec('dataset', d)
        self.assertEqual(self.first(), before)
        self.w.archive_spec('dataset', pair['dataset']['id'])
        self.assertEqual(self.first(), before)

    def test_actual_worker_run_stores_independent_validated_result_without_raw_duplicate_bundle(self):
        saved = self.execute(); record = saved['record']; attempt = record['attempts'][0]
        self.assertEqual(attempt['status'], 'completed', attempt['error']); self.assertEqual(attempt['stage'], 'execution')
        self.assertNotIn('bundle', attempt['result']); self.assertEqual(attempt['result']['counts']['completed_updates'], 3)
        validate_all(self.w.db)
        bundle = candidate_bundle(self.e, record['selection']['coordinates'], self.w.get_spec)
        validate_result(bundle, attempt['result'])
        self.w.delete(self.e['models'][0]['id']); validate_all(self.w.db)
        self.assertEqual(self.w.get_spec('run', record['id'], record['revision']), record)

    def test_explicit_retry_appends_immutable_attempts_and_old_run_revision_remains_readable(self):
        first = self.execute(); old = copy.deepcopy(first['record'])
        with self.assertRaisesRegex(GraphError, 'retry'): self.execute()
        second = self.execute(retry=True)
        self.assertEqual(second['record']['attempts'][0], old['attempts'][0]); self.assertEqual(len(second['record']['attempts']), 2)
        self.assertEqual(self.w.get_spec('run', 'run', first['reference']['revision']), old)
        self.assertEqual(second['record']['attempts'][1]['result']['checkpoint'], old['attempts'][0]['result']['checkpoint'])
        changed = copy.deepcopy(second['record']); changed.pop('revision'); changed['attempts'][0]['finished_at'] = changed['attempts'][1]['finished_at']
        with self.assertRaises(GraphError): self.w.put_spec('run', changed)

    def test_run_cannot_be_reused_for_another_coordinate_even_if_browsing_head_is_old(self):
        self.execute(); rows = preview_experiment(self.e, self.w.get_spec)['candidates']
        with self.assertRaisesRegex(GraphError, 'substitute'): self.execute(rows[1], retry=True)

    def test_admission_and_missing_worker_failures_are_terminal_records_not_fabricated_metrics(self):
        from app.transport import WorkerError
        with patch('app.experiment_objects.resolve_run', side_effect=GraphError('RAM admission rejected')):
            a = self.execute(run_id='admission')['record']['attempts'][0]
        self.assertEqual((a['stage'], a['status'], a['result']), ('admission', 'failed', None))
        self.assertIn('RAM', a['error']['message'])
        b = self.execute(run_id='missingworker', python='/nonexistent/private-test-python')['record']['attempts'][0]
        self.assertEqual((b['stage'], b['status'], b['result']), ('worker', 'failed', None)); validate_all(self.w.db)

    def test_bad_worker_artifact_is_retained_as_worker_failure_not_success(self):
        with patch('app.transport.run_json_worker', return_value={'bundle': {}, 'status': 'completed'}):
            r = self.execute()['record']['attempts'][0]
        self.assertEqual(r['status'], 'failed'); self.assertEqual(r['stage'], 'worker'); self.assertIsNone(r['result'])
        self.assertIn('bundle', r['error']['message'])

    def test_result_mutations_fail_without_importing_torch_in_validator(self):
        saved = self.execute(); coords = saved['record']['selection']['coordinates']; b = candidate_bundle(self.e, coords, self.w.get_spec)
        result = saved['record']['attempts'][0]['result']
        for change in ('identity', 'checkpoint', 'counts', 'population', 'denominator', 'quotient', 'timing', 'test-curve', 'environment', 'sampler', 'budget'):
            bad = copy.deepcopy(result)
            metric = bad['final_metrics']['test']['heads']['prediction']
            if change == 'identity': bad['identity']['manifest'] = '0' * 64
            if change == 'checkpoint': bad['checkpoint_id'] = '0' * 64
            if change == 'counts': bad['counts']['completed_updates'] = True
            if change == 'population': metric['observation_ids'].reverse()
            if change == 'denominator': metric['denominator'] *= 2
            if change == 'quotient': metric['value'] += 1.
            if change == 'timing': bad['history'][0]['timing'] = 'before-update'
            if change == 'test-curve': bad['history'][0]['metrics']['test'] = bad['final_metrics']['test']
            if change == 'environment': bad['environment']['optimizer_resume'] = True
            if change == 'sampler': bad['sampler']['seed'] += 1
            if change == 'budget': bad['stop_reason'] = 'epochs_budget'
            with self.subTest(change=change), self.assertRaises(GraphError): validate_result(b, bad)

    def test_reports_require_exact_runs_and_explicit_result_vs_rerun_modes(self):
        saved = self.execute()
        result_recipe = {'version': 1, 'id': 'report', 'name': 'Measured results', 'question': 'What actually happened?',
            'mode': 'result-report', 'runs': [saved['reference']], 'experiment': None,
            'sections': ['question', 'protocol', 'evaluation', 'failures'], 'analysis': {'kind': 'descriptive', 'inferential_unit': 'run'}}
        ref = self.w.put_spec('report', result_recipe); self.assertEqual(ref['kind'], 'report')
        rerun = copy.deepcopy(result_recipe); rerun.update(id='rerun', mode='rerun', runs=[], experiment=object_reference('experiment', self.e))
        self.w.put_spec('report', rerun); validate_all(self.w.db)
        for change in ('mixed', 'missing', 'analysis', 'section'):
            bad = copy.deepcopy(result_recipe)
            if change == 'mixed': bad['experiment'] = object_reference('experiment', self.e)
            if change == 'missing': bad['runs'][0]['revision'] = '0' * 64
            if change == 'analysis': bad['analysis']['confidence'] = .95
            if change == 'section': bad['sections'].append('invented-heavy-analysis')
            with self.assertRaises(GraphError): self.w.put_spec('report', bad)

    def test_run_attempt_time_order_and_checkpoint_free_success_are_rejected(self):
        record = self.execute()['record']
        for change in ('number', 'time', 'timezone', 'success-no-result', 'extra'):
            bad = copy.deepcopy(record); bad.pop('revision')
            a = bad['attempts'][0]
            if change == 'number': a['number'] = True
            if change == 'time': a['finished_at'] = '2000-01-01T00:00:00Z'
            if change == 'timezone': a['started_at'] = '2026-10-10T20:00:00'
            if change == 'success-no-result': a['result'] = None; a['stage'] = 'worker'; a['error'] = {'type': 'Error', 'message': 'fake'}
            if change == 'extra': a['invented'] = True
            with self.assertRaises(GraphError): self.w.put_spec('run', bad)

    def test_save_reload_version6_closes_exact_dependencies_and_retains_old_v5_bytes(self):
        saved = self.execute()
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'records.vtw'; self.w.save(path); loaded = Workspace.load(path)
            try:
                self.assertEqual(loaded.get_spec('run', saved['reference']['id'], saved['reference']['revision']), saved['record'])
                self.assertEqual(loaded.db.execute('PRAGMA user_version').fetchone()[0], 6)
            finally: loaded.db.close()
            legacy = Path(td) / 'v5.vtw'; old = Workspace(); old.put(default_model()); old.save(legacy); old.db.close()
            db = sqlite3.connect(legacy); db.execute('PRAGMA user_version=5'); db.commit(); db.close(); before = legacy.read_bytes()
            migrated = Workspace.load(legacy)
            try:
                self.assertEqual(migrated.db.execute('PRAGMA user_version').fetchone()[0], 6)
                self.assertEqual(legacy.read_bytes(), before)
                migrated.save(legacy); self.assertNotEqual(legacy.read_bytes(), before)
            finally: migrated.db.close()

    def test_old_header_cannot_hide_version6_object_semantics(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'fake-v5.vtw'; self.w.save(path)
            db = sqlite3.connect(path); db.execute('PRAGMA user_version=5'); db.commit(); db.close()
            with self.assertRaisesRegex(GraphError, 'version-6'): Workspace.load(path)

    def test_all_eight_candidates_execute_and_independent_retries_reproduce_checkpoints(self):
        rows = preview_experiment(self.e, self.w.get_spec)['candidates']
        for i, row in enumerate(rows):
            first = self.execute(row, run_id='grid' + str(i))
            repeated = self.execute(row, run_id='grid' + str(i), retry=True)
            a, b = repeated['record']['attempts']
            self.assertEqual(a['status'], 'completed', a['error']); self.assertEqual(b['status'], 'completed', b['error'])
            self.assertEqual(a['result']['checkpoint'], b['result']['checkpoint'])
            self.assertEqual(a['result']['final_metrics'], b['result']['final_metrics'])
            self.assertEqual(a['result']['identity']['manifest'], digest(row['manifest']))
            self.assertEqual(a['result']['identity']['model'], row['references']['model'])
        self.assertEqual(len(self.w.list_specs('run')), 8)

    def test_actual_ram_admission_failure_is_saved_before_worker_launch(self):
        self.w.set_constrained_mode(True)
        with patch('app.task_execution.available_memory', return_value=128 * 1024**2), patch('app.transport.run_json_worker', side_effect=AssertionError('must not launch')):
            record = self.execute(run_id='ram')['record']
        attempt = record['attempts'][0]
        self.assertEqual((attempt['stage'], attempt['status']), ('admission', 'failed'))
        self.assertIn('RAM', attempt['error']['message']); self.assertIsNone(attempt['result'])

    def test_incompatible_attempt_is_saved_as_unavailable_without_running_or_inventing_loss(self):
        e = copy.deepcopy(self.e); e.pop('revision')
        e['selection']['exclusions'] = [{'model': e['models'][0], 'data_task': 'pair0', 'reason': 'Not part of this comparison'}]
        ref = self.w.put_spec('experiment', e); self.e = self.w.get_spec('experiment', ref['id'], ref['revision'])
        with patch('app.transport.run_json_worker', side_effect=AssertionError('must not train excluded selection')):
            record = self.execute()['record']
        a = record['attempts'][0]; self.assertEqual((a['stage'], a['status'], a['result']), ('compatibility', 'unavailable', None))
        self.assertIn('comparison', a['error']['message']); validate_all(self.w.db)

    def test_weighted_metric_validator_uses_actual_float32_sample_weights(self):
        pair = self.e['data_tasks'][0]; d = self.w.get_spec('dataset', pair['dataset']['id']); d.pop('revision')
        d['id'] = 'weighted'; d['fields'] += [{'id': 'w', 'dtype': 'float32', 'shape': []}, {'id': 'm', 'dtype': 'bool', 'shape': []}]
        for i, row in enumerate(d['records']): row['values'].update(w=.2 if i % 2 else .7, m=i != 0)
        t = task(); t['id'] = 'weighted-task'; t['heads'][0].update(mask='m', sample_weight='w')
        e = copy.deepcopy(self.e); e.pop('revision'); e['data_tasks'][0].update(dataset=self.w.put_spec('dataset', d), task=self.w.put_spec('task', t))
        ref = self.w.put_spec('experiment', e); self.e = self.w.get_spec('experiment', ref['id'], ref['revision'])
        record = self.execute()['record']; self.assertEqual(record['attempts'][0]['status'], 'completed', record['attempts'][0]['error'])
        validate_all(self.w.db)

    def test_run_history_undo_preserves_immutable_attempts_and_dependencies(self):
        before = self.w.history.capture(self.w.db); first = self.execute(); self.w.history.record(before, self.w.db)
        self.assertTrue(self.w.history.restore(self.w))
        self.assertEqual(self.w.list_specs('run'), [])
        self.assertEqual(self.w.get_spec('run', 'run', first['reference']['revision']), first['record'])
        second = self.execute(retry=True); self.assertEqual(len(second['record']['attempts']), 2)
        shortened = copy.deepcopy(second['record']); shortened.pop('revision'); shortened['attempts'] = shortened['attempts'][:1]; shortened['name'] = 'Truncated new record'
        with self.assertRaisesRegex(GraphError, 'truncate'): self.w.put_spec('run', shortened)

    def test_load_and_save_detect_sql_inserted_dangling_dependencies(self):
        bad = copy.deepcopy(self.e); bad.pop('revision'); bad['id'] = 'dangling'; bad['models'][0]['revision'] = '0' * 64
        value = frozen_document('experiment', bad)
        with self.w.db:
            self.w.db.execute('INSERT INTO spec_revisions VALUES (?,?,?,?,?,?)',
                ('experiment', value['id'], value['revision'], value['name'], json.dumps(value), '2026-10-10T20:00:00Z'))
        with self.assertRaisesRegex(GraphError, 'missing immutable'): validate_all(self.w.db)
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(GraphError): self.w.save(Path(td) / 'invalid.vtw')

    def test_cli_preview_and_actual_run_roundtrip_are_pure_host_and_atomic_file_writes(self):
        with tempfile.TemporaryDirectory() as td:
            directory = Path(td); path = directory / 'experiment.vtw'; self.w.save(path); before = path.read_bytes()
            e = object_reference('experiment', self.e)
            preview = subprocess.run([sys.executable, 'vpw_cli.py', str(path), 'preview-experiment', e['id'], e['revision'], '--limit', '2'], capture_output=True, text=True, timeout=10)
            self.assertEqual(preview.returncode, 0, preview.stdout + preview.stderr); rows = json.loads(preview.stdout)['value']['candidates']
            self.assertEqual(len(rows), 2); self.assertEqual(path.read_bytes(), before)
            coords = directory / 'coords.json'; coords.write_text(json.dumps(rows[0]['coordinates']))
            args = [str(path), 'run-candidate', '--experiment', e['id'], e['revision'], '--coordinates', str(coords), '--run-id', 'cli-run', '--name', 'CLI trial', '--memory-limit-percent', '10']
            code = "import sys,json,vpw_cli; result=vpw_cli.main(sys.argv[1:]); print(json.dumps({'torch': 'torch' in sys.modules,'qt': any(k.startswith('PyQt') for k in sys.modules)})); sys.exit(result)"
            run = subprocess.run([sys.executable, '-c', code, *args], capture_output=True, text=True, timeout=25)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr); lines = run.stdout.splitlines()
            value = json.loads(lines[0])['value']; self.assertTrue(value['persisted']); self.assertEqual(value['record']['attempts'][0]['status'], 'completed')
            self.assertEqual(json.loads(lines[1]), {'torch': False, 'qt': False}); self.assertNotEqual(path.read_bytes(), before)
            loaded = Workspace.load(path)
            try: self.assertEqual(loaded.get_spec('run', 'cli-run', value['reference']['revision']), value['record'])
            finally: loaded.db.close()

    def test_cli_save_conflict_returns_recoverable_evidence_without_overwriting_concurrent_edits(self):
        with tempfile.TemporaryDirectory() as td:
            directory = Path(td); path = directory / 'conflict.vtw'; self.w.save(path)
            coords = directory / 'coords.json'; coords.write_text(json.dumps(self.first()['coordinates']))
            from app.transport import run_json_worker
            disk_after = []
            def edit_after_run(*args, **kwargs):
                result = run_json_worker(*args, **kwargs)
                other = Workspace.load(path)
                try:
                    doc = next(iter(other.models().values())); doc['name'] = 'Concurrent user rename'; other.put(doc); other.save(path)
                    disk_after.append(path.read_bytes())
                finally: other.db.close()
                return result
            stream = io.StringIO()
            with patch('app.transport.run_json_worker', side_effect=edit_after_run), redirect_stdout(stream):
                code = vpw_cli.main([str(path), 'run-candidate', '--experiment', self.e['id'], self.e['revision'], '--coordinates', str(coords), '--run-id', 'conflict-run', '--name', 'Conflict trial', '--memory-limit-percent', '10'])
            self.assertEqual(code, 0); value = json.loads(stream.getvalue())['value']; self.assertFalse(value['persisted'])
            self.assertIn('changed on disk', value['save_error'].lower()); self.assertEqual(value['record']['attempts'][0]['status'], 'completed')
            self.assertEqual(path.read_bytes(), disk_after[0])
            loaded = Workspace.load(path)
            try:
                self.assertEqual(loaded.list_specs('run'), [])
                loaded.put_spec('run', value['record']); loaded.save(path)  # explicit recovery, no retraining
            finally: loaded.db.close()

    def test_cli_failed_atomic_batch_cannot_leave_a_partial_experiment_on_disk(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'atomic.vtw'; self.w.save(path); before = path.read_bytes()
            bad = copy.deepcopy(self.e); bad.pop('revision'); bad['id'] = 'will-not-persist'; missing = copy.deepcopy(bad); missing['id'] = 'invalid'; missing['models'][0]['revision'] = '0' * 64
            edits = Path(td) / 'edits.json'; edits.write_text(json.dumps([{'op': 'put-spec', 'kind': 'experiment', 'document': bad}, {'op': 'put-spec', 'kind': 'experiment', 'document': missing}]))
            stream = io.StringIO()
            with redirect_stdout(stream): code = vpw_cli.main([str(path), 'apply', str(edits)])
            self.assertEqual(code, 1); self.assertEqual(path.read_bytes(), before)

    def test_stored_records_load_in_fresh_stdlib_only_python(self):
        self.execute()
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'pure.vtw'; self.w.save(path)
            code = "from app.core import Workspace; import json,sys; w=Workspace.load(sys.argv[1]); print(json.dumps({'run_count':len(w.list_specs('run')),'imports':[k for k in sys.modules if k=='torch' or k=='numpy' or k.startswith('PyQt')]})); w.db.close()"
            p = subprocess.run([sys.executable, '-c', code, str(path)], capture_output=True, text=True, timeout=10)
            self.assertEqual(p.returncode, 0, p.stderr); self.assertEqual(json.loads(p.stdout), {'run_count': 1, 'imports': []})

    def test_record_reads_retries_and_reload_use_saved_manifest_not_split_regeneration(self):
        first = self.execute(); captured = first['record']['resolved']['manifest']
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'captured.vtw'; self.w.save(path)
            with patch('app.experiment_objects.make_split', side_effect=AssertionError('must not regenerate historical split')):
                self.assertEqual(self.w.get_spec('run', 'run')['resolved']['manifest'], captured)
                loaded = Workspace.load(path)
                try:
                    retry = execute_candidate(loaded, object_reference('experiment', self.e), first['record']['selection']['coordinates'], 'run', 'Trial', retry=True, timeout=20)
                    self.assertEqual(retry['record']['resolved']['manifest'], captured)
                    self.assertEqual(retry['record']['attempts'][1]['result']['checkpoint'], first['record']['attempts'][0]['result']['checkpoint'])
                finally: loaded.db.close()

    def test_report_materialization_is_readonly_preserves_failures_and_never_retrains(self):
        completed = self.execute()
        failed = self.execute(run_id='failed', python='/nonexistent/private-test-python')
        recipe = {'version': 1, 'id': 'report', 'name': 'Include unsuccessful trials', 'question': 'What evidence is available?',
            'mode': 'result-report', 'runs': [completed['reference'], failed['reference']], 'experiment': None,
            'sections': ['question', 'evaluation', 'failures'], 'analysis': {'kind': 'descriptive', 'inferential_unit': 'run'}}
        ref = self.w.put_spec('report', recipe)
        with patch('app.transport.run_json_worker', side_effect=AssertionError('report must not train')):
            result = report_data(self.w.get_spec('report', ref['id'], ref['revision']), self.w.get_spec)
        self.assertEqual(result['execution_behavior'], 'read-saved-evidence-only'); self.assertEqual(result['notebook_status'], 'not-generated')
        self.assertEqual([r['status'] for r in result['runs']], ['completed', 'failed'])
        self.assertIsNone(result['runs'][1]['counts']); self.assertEqual(result['runs'][1]['metrics'], {})
        self.execute(retry=True)
        self.assertEqual(report_data(self.w.get_spec('report', ref['id'], ref['revision']), self.w.get_spec), result)
        rerun = copy.deepcopy(recipe); rerun.update(id='rerun-report', mode='rerun', runs=[], experiment=object_reference('experiment', self.e))
        rref = self.w.put_spec('report', rerun)
        with patch('app.transport.run_json_worker', side_effect=AssertionError('viewing rerun recipe must not launch')):
            data = report_data(self.w.get_spec('report', rref['id'], rref['revision']), self.w.get_spec)
        self.assertEqual(data['execution_behavior'], 'explicit-execution-required'); self.assertEqual(data['runs'], [])
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'reports.vtw'; self.w.save(path); before = path.read_bytes(); stream = io.StringIO()
            with redirect_stdout(stream): code = vpw_cli.main([str(path), 'report-data', ref['id'], ref['revision']])
            self.assertEqual(code, 0); self.assertEqual(json.loads(stream.getvalue())['value'], result); self.assertEqual(path.read_bytes(), before)

    def test_descriptive_run_unit_does_not_double_count_two_revisions_of_same_run(self):
        first = self.execute(); second = self.execute(retry=True)
        recipe = {'version': 1, 'id': 'duplicate', 'name': 'Not independent units', 'question': 'Should not count twice',
            'mode': 'result-report', 'runs': [first['reference'], second['reference']], 'experiment': None,
            'sections': ['evaluation'], 'analysis': {'kind': 'descriptive', 'inferential_unit': 'run'}}
        with self.assertRaisesRegex(GraphError, 'logical run'): self.w.put_spec('report', recipe)

    def test_classification_terminal_metrics_are_validated_and_tampered_accuracy_is_rejected(self):
        from test_task_training import classification
        frozen = classification(); frozen['protocol']['budget']['epochs'] = 3
        e = copy.deepcopy(self.e); e.pop('revision'); e['id'] = 'classification-experiment'
        e['models'] = [self.w.put_spec('model', frozen['model'])]
        e['data_tasks'] = [{'id': 'classification', 'dataset': self.w.put_spec('dataset', frozen['dataset']), 'task': self.w.put_spec('task', frozen['task'])}]
        e['protocols'] = [{'id': 'adam', 'protocol': frozen['protocol']}]
        ref = self.w.put_spec('experiment', e); self.e = self.w.get_spec('experiment', ref['id'], ref['revision'])
        row = self.first(); record = self.execute(row)['record']; attempt = record['attempts'][0]
        self.assertEqual(attempt['status'], 'completed', attempt['error'])
        b = candidate_bundle(self.e, row['coordinates'], self.w.get_spec, saved_manifest=record['resolved']['manifest'])
        bad = copy.deepcopy(attempt['result']); bad['final_metrics']['test']['heads']['prediction']['accuracy'] = 2.
        with self.assertRaisesRegex(GraphError, 'accuracy'): validate_result(b, bad)

    def test_invented_ram_components_do_not_pass_as_runtime_evidence(self):
        record = self.execute()['record']; bad = copy.deepcopy(record['attempts'][0]['result'])
        bad['memory_admission']['components'] = {'fake': 0}; bad['memory_admission']['estimated_bytes'] = 0
        b = candidate_bundle(self.e, record['selection']['coordinates'], self.w.get_spec, saved_manifest=record['resolved']['manifest'])
        with self.assertRaisesRegex(GraphError, 'RAM component'): validate_result(b, bad)

    def test_public_example_saves_preview_without_training_and_refuses_overwrite(self):
        from examples.experiment_workspace import main
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'example.vtw'; output = io.StringIO()
            with patch('app.transport.run_json_worker', side_effect=AssertionError('default example must not train')):
                with redirect_stdout(output):
                    main([str(path)])
            artifact = json.loads(output.getvalue()); before = path.read_bytes()
            self.assertIsNone(artifact['run']); self.assertEqual(artifact['preview']['total_combinations'], 8)
            self.assertEqual({r['status'] for r in artifact['preview']['candidates']}, {'compatible'})
            loaded = Workspace.load(path)
            try:
                self.assertEqual(len(loaded.models()), 2); self.assertEqual(loaded.list_specs('run'), [])
                recipe = loaded.get_spec('report', 'rerun-comparison')
                self.assertEqual(report_data(recipe, loaded.get_spec)['execution_behavior'], 'explicit-execution-required')
                for kind in ('dataset', 'model', 'task'): self.assertEqual(len(loaded.list_specs(kind)), 2)
            finally:
                loaded.db.close()
            with redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit): main([str(path)])
            self.assertEqual(path.read_bytes(), before)

    def test_public_example_explicit_fit_stores_actual_completed_evidence(self):
        from examples.experiment_workspace import main
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'trained-example.vtw'; output = io.StringIO()
            with redirect_stdout(output): main([str(path), '--run-first'])
            artifact = json.loads(output.getvalue()); loaded = Workspace.load(path)
            try:
                record = loaded.get_spec('run', artifact['run']['id'], artifact['run']['revision'])
                attempt = record['attempts'][0]; self.assertEqual(attempt['status'], 'completed', attempt['error'])
                self.assertEqual(attempt['result']['counts']['completed_updates'], 100)
                self.assertLess(attempt['result']['final_metrics']['test']['heads']['prediction']['value'], .03)
                recipe = loaded.get_spec('report', 'first-trial-report')
                view = report_data(recipe, loaded.get_spec)
                self.assertEqual(view['runs'][0]['checkpoint_id'], attempt['result']['checkpoint_id'])
            finally:
                loaded.db.close()


if __name__ == '__main__':
    unittest.main()
