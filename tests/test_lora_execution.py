"""Six bounded actual fits; native-only frozen baselines and explicit merge seam."""
import copy
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

from app.core import GraphError, Workspace
from app.evidence import digest
from app.lora import populate
from app.lora_execution import evaluate_baseline, merge_selections, run_comparisons
from app.revisions import validate_all
from examples.lora_workspace import MISSING_MERGE, complete_merge, main


ROOT = Path(__file__).resolve().parents[1]


def close(workspace):
    workspace.db.close()
    if workspace.history.directory: workspace.history.directory.cleanup()


class LoRAPreparationTests(unittest.TestCase):
    def test_fresh_default_generator_is_preparation_only_and_has_no_numerical_imports(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'prepared.vtw'
            script = ('import sys,json,random,io;from contextlib import redirect_stdout;'
                'before=random.getstate();from examples.lora_workspace import main;'
                'from app.core import Workspace;'
                'out=io.StringIO();\nwith redirect_stdout(out): answer=main([sys.argv[1]])\n'
                'w=Workspace.load(sys.argv[1]);'
                'print(json.dumps({"answer":answer,"runs":w.list_specs("run"),'
                '"merged":"lora-merged" in w.models(),"rng":random.getstate()==before,'
                '"imports":[n for n in sys.modules if n.split(".")[0] in ("torch","numpy","PyQt6")]}));w.db.close()')
            proc = subprocess.run([sys.executable, '-c', script, str(path)], cwd=ROOT,
                                  capture_output=True, text=True, timeout=30)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            value = json.loads(proc.stdout)
            self.assertEqual(value['imports'], []); self.assertTrue(value['rng'])
            self.assertEqual(value['runs'], []); self.assertFalse(value['merged'])
            self.assertEqual(value['answer']['phase'], 'preparation-only')
            self.assertIn(MISSING_MERGE, value['answer']['merge']['warning'])
            self.assertEqual(value['answer']['artifacts'], {})
            self.assertFalse(list(Path(directory).glob('*.json')))

    def test_generator_never_overwrites_workspace(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'existing.vtw'; path.write_bytes(b'preserved user bytes')
            with redirect_stdout(io.StringIO()), self.assertRaises(SystemExit): main([str(path)])
            self.assertEqual(path.read_bytes(), b'preserved user bytes')

    def test_trainable_baseline_refuses_before_native_compilation(self):
        w = Workspace(); self.addCleanup(close, w); refs = populate(w, updates=1)
        expref = refs['lora-linear-comparison']; exp = w.get_spec('experiment', expref['id'], expref['revision'])
        modelref = next(r for r in exp['models'] if r['id'] == 'full-finetune')
        model = w.get_spec('model', modelref['id'], modelref['revision'])
        with patch('app.lora_execution._native_initial', side_effect=AssertionError('No compilation')):
            with self.assertRaisesRegex(GraphError, 'all parameter owners frozen'):
                evaluate_baseline(model, w.get_spec('dataset', exp['data_tasks'][0]['dataset']['id']),
                                  w.get_spec('task', exp['data_tasks'][0]['task']['id']), {})


class LoRAExecutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.w = Workspace()
        try:
            cls.experiments = populate(cls.w, updates=200)
            cls.evidence = run_comparisons(cls.w, cls.experiments, timeout=120.)
            # Explicit private evidence destination; no public default output.
            if os.environ.get('LORA_COMPARISON_TEST_EVIDENCE'):
                folder = Path(os.environ['LORA_COMPARISON_TEST_EVIDENCE']).resolve()
                folder.mkdir(parents=True, exist_ok=True)
                cls.w.save(folder / 'actual.vtw', require_new=True)
                with (folder / 'comparisons.json').open('x', encoding='utf-8') as stream:
                    json.dump(cls.evidence, stream, allow_nan=False, sort_keys=True, indent=2)
        except BaseException:
            close(cls.w); raise

    @classmethod
    def tearDownClass(cls): close(cls.w)

    def test_six_actual_completed_worker_fits_and_reports(self):
        self.assertTrue(self.evidence['complete'])
        self.assertEqual(len(self.w.list_specs('run')), 6)
        expected = {'full-finetune': 256, 'lora-r1': 32, 'lora-r2': 64, 'lora-r4': 128,
                    'mlp-full': 512, 'mlp-adapter': 128}
        for group in self.evidence['groups'].values():
            report = group['report']; saved = self.w.get_spec('report', report['id'], report['revision'])
            self.assertEqual(saved['runs'], [r['run'] for r in group['runs']])
            for trial in group['runs']:
                run = self.w.get_spec('run', trial['run']['id'], trial['run']['revision'])
                actual = run['attempts'][0]
                self.assertEqual(actual['stage'], 'execution'); self.assertEqual(actual['status'], 'completed')
                self.assertEqual(actual['result']['counts']['completed_updates'], 200)
                self.assertEqual(actual['result']['environment']['numerical_threads'], 1)
                self.assertEqual(trial['initial_snapshot']['parameter_policy_receipt']['counts']['trainable'],
                                 expected[trial['model']['id']])
                self.assertEqual(trial['checkpoint_id'], digest(actual['result']['checkpoint']))
        validate_all(self.w.db)

    def test_native_baselines_are_separate_zero_update_evidence_with_matched_populations(self):
        for group in self.evidence['groups'].values():
            baseline = group['baseline']; snapshot = baseline['initial_snapshot']
            self.assertEqual(baseline['status'], 'evaluated'); self.assertFalse(baseline['training'])
            self.assertFalse(baseline['optimizer_created']); self.assertFalse(baseline['optimizer_resume'])
            self.assertEqual(baseline['completed_updates'], 0); self.assertEqual(baseline['parameter_counts']['trainable'], 0)
            self.assertEqual(snapshot['sha256'], digest({k: v for k, v in snapshot.items() if k != 'sha256'}))
            self.assertEqual(baseline['manifest_sha256'], digest(baseline['manifest']))
            self.assertEqual(baseline['fitted_sha256'], digest(baseline['fitted']))
            # The analytic teacher is the TARGET, not a fabricated native baseline prediction.
            self.assertGreater(baseline['metrics']['test']['heads']['prediction']['value'], 0)
            self.assertNotIn(baseline['model']['id'], [r['model']['id'] for r in group['runs']])
            for trial in group['runs']:
                result = self.w.get_spec('run', trial['run']['id'], trial['run']['revision'])['attempts'][0]['result']
                self.assertEqual(digest(result['checkpoint']['fitted']), baseline['fitted_sha256'])
                for role in ('train', 'validation', 'test'):
                    a = trial['metrics'][role]['heads']['prediction']; b = baseline['metrics'][role]['heads']['prediction']
                    for key in ('observation_ids', 'denominator', 'elements', 'included_elements', 'reduction', 'accumulation_dtype'):
                        self.assertEqual(a[key], b[key])
                    self.assertNotEqual(a['checkpoint_id'], b['checkpoint_id'])
            self.assertEqual(baseline['metrics']['test']['heads']['prediction']['denominator'], 24 * 16)

    def test_native_mlp_baseline_matches_independent_matrix_relu_oracle_without_optimizer(self):
        import torch
        baseline = self.evidence['groups']['mlp']['baseline']; snapshot = baseline['initial_snapshot']
        shapes = {json.loads(row['id'])[-3]: snapshot['weights'][path]
                  for path, row in snapshot['schema'].items()}
        records = {r['id']: r['values'] for r in baseline['dataset_snapshot']['records']}
        ids = baseline['manifest']['roles']['test']
        x = torch.tensor([records[r]['x'] for r in ids], dtype=torch.float32)
        y = torch.tensor([records[r]['y'] for r in ids], dtype=torch.float32)
        pred = torch.relu(x @ torch.tensor(shapes['layer-1'], dtype=torch.float32).T)
        pred = pred @ torch.tensor(shapes['layer-2'], dtype=torch.float32).T
        expected = float((pred - y).square().double().sum()) / y.numel()
        self.assertAlmostEqual(baseline['metrics']['test']['heads']['prediction']['value'], expected, places=12)
        # Re-evaluation uses the saved manifest, never a fresh split or optimizer.
        with patch('app.experiment_objects.make_split', side_effect=AssertionError('No new split')), \
             patch('torch.optim.Adam', side_effect=AssertionError('No optimizer')):
            rerun = evaluate_baseline(snapshot['model_snapshot'], baseline['dataset_snapshot'],
                                     baseline['task_snapshot'], baseline['manifest'])
        self.assertEqual(rerun['metrics'], baseline['metrics'])

    def test_actual_fits_improve_this_fixed_synthetic_test_population(self):
        for label, group in self.evidence['groups'].items():
            before = group['baseline']['metrics']['test']['heads']['prediction']['value']
            for trial in group['runs']:
                with self.subTest(group=label, model=trial['model']['id']):
                    self.assertLess(trial['metrics']['test']['heads']['prediction']['value'], before)

    def test_exact_rank2_merge_boundary_is_mocked_not_a_placeholder_implementation(self):
        rank2 = self.evidence['rank2']; selections = merge_selections(self.w, rank2['run'], attempt=1)
        self.assertEqual(set(selections), {'base', 'A', 'B'})
        self.assertEqual(len({s['checkpoint_id'] for s in selections.values()}), 1)
        for role, item in selections.items():
            self.assertEqual(item['run'], rank2['run']); self.assertEqual(item['attempt'], 1)
            self.assertEqual(json.loads(item['parameter']['id']), ['lora-r2', role, 'weight'])
        boundary = Mock(return_value={'reference': {'kind': 'mock-only', 'id': 'not-a-real-derivation'},
                                     'model': {'id': 'lora-merged'}, 'changed': True})
        answer = complete_merge(self.w, self.evidence, admit=boundary)
        self.assertEqual(answer['status'], 'admitted')
        boundary.assert_called_once_with(self.w, selections, alpha=4., model_id='lora-merged',
                                        name='LoRA merged · actual retained rank2 checkpoint')
        self.assertNotIn('lora-merged', self.w.models())  # Mock means NO integration claim.

    def test_artifact_json_and_actual_registry_roundtrip_without_rebuilding_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'actual.vtw'; self.w.save(path)
            reopened = Workspace.load(path)
            try:
                self.assertEqual(len(reopened.list_specs('run')), 6)
                with patch('app.experiment_objects.make_split', side_effect=AssertionError('No split regeneration')):
                    for group in self.evidence['groups'].values():
                        report = group['report']; reopened.get_spec('report', report['id'], report['revision'])
            finally: close(reopened)
            artifact = Path(directory) / 'comparison.json'
            artifact.write_text(json.dumps(self.evidence, allow_nan=False, sort_keys=True), encoding='utf-8')
            restored = json.loads(artifact.read_text(encoding='utf-8'))
            self.assertEqual(restored, self.evidence)

    def test_generator_complete_path_writes_actual_artifacts_but_never_claims_mock_merge(self):
        cloned = Workspace(); self.w.db.backup(cloned.db)
        # Reuse the six real fits above; mock orchestration, not numerical evidence.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'generated.vtw'
            with patch('examples.lora_workspace.Workspace', return_value=cloned), \
                 patch('examples.lora_workspace.run_comparisons', return_value=copy.deepcopy(self.evidence)) as driver, \
                 patch('examples.lora_workspace.complete_merge', return_value={
                     'status': 'unavailable', 'warning': 'Explicitly mocked merge boundary; no integration claim'}), \
                 redirect_stdout(io.StringIO()):
                output = main([str(path), '--run-comparisons'])
            driver.assert_called_once()
            self.assertTrue(output['comparisons_complete']); self.assertFalse(output['complete'])
            self.assertEqual(len(output['runs']), 6)
            for label in ('linear', 'mlp'):
                artifact = json.loads(Path(output['artifacts'][label]).read_text(encoding='utf-8'))
                self.assertEqual(artifact, self.evidence['groups'][label]['baseline'])
            saved = Workspace.load(path)
            try:
                self.assertEqual(len(saved.list_specs('run')), 6)
                self.assertNotIn('lora-merged', saved.models())
            finally: close(saved)

    def test_source_identity_and_attempt_refuse_implicit_or_wrong_merge_selection(self):
        rank2 = self.evidence['rank2']; raw = self.w.db.serialize()
        for attempt in (0, True, 2):
            with self.assertRaises(GraphError): merge_selections(self.w, rank2['run'], attempt=attempt)
        first = self.evidence['groups']['linear']['runs'][0]
        with self.assertRaises(GraphError): merge_selections(self.w, first['run'], attempt=1)
        self.assertEqual(raw, self.w.db.serialize())


if __name__ == '__main__': unittest.main()
