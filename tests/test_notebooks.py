"""Real Jupyter kernels, portable closures, honest modes and failed-cell evidence."""
import copy
import csv
import io
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import patch

import nbformat
from app.core import GraphError
from app.evidence import digest
from app.experiment_objects import execute_candidate, object_reference, preview_experiment
from app.notebooks import build_notebook, create_bundle, execute_bundle, payload_for, verify_bundle
from app.report_runtime import curve_svg, resolve_payload
from test_experiment_objects import experiment_workspace


class NotebookTests(unittest.TestCase):
    def setUp(self):
        self.w, self.e = experiment_workspace()
        self.addCleanup(self.w.db.close)
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def recipe(self, mode='rerun', runs=None, sections=None):
        return self.w.put_spec('report', {'version': 1, 'id': mode, 'name': 'Model comparison',
            'question': 'What do these bounded trials show?', 'mode': mode, 'runs': runs or [],
            'experiment': object_reference('experiment', self.e) if mode == 'rerun' else None,
            'sections': sections or ['question', 'models', 'datasets', 'protocol', 'training', 'evaluation', 'failures', 'environment', 'checkpoints'],
            'analysis': {'kind': 'descriptive', 'inferential_unit': 'run'}})

    def trial(self, run_id='first', **kwargs):
        row = preview_experiment(self.e, self.w.get_spec)['candidates'][0]
        return execute_candidate(self.w, object_reference('experiment', self.e), row['coordinates'],
                                 run_id, 'Selected trial', timeout=20, **kwargs)

    def execute(self, ref, destination='bundle', **options):
        exported = create_bundle(self.w, ref, self.root / destination, **options)
        result = execute_bundle(exported['path'], timeout=60)
        self.assertEqual(result['execution']['status'], 'completed', result['execution']['error'])
        return Path(exported['path']), Path(result['path']), result

    @staticmethod
    def no_training_kernel(payload, index):
        """Arm probes in the actual fresh kernel, not only the authoring host."""
        notebook = build_notebook(payload, index)
        bootstrap = next(c for c in notebook.cells if c.metadata.visual_pytorch.section == 'bootstrap')
        bootstrap.source += """
import builtins
import app.report_runtime as audited_report_runtime
import app.experiment_objects as audited_experiment_objects
import app.task_training as audited_task_training
import app.transport as audited_transport
no_training_audit = {'phase': 'armed', 'violations': [], 'kernel_pid': os.getpid(),
                     'runtime': str(Path(audited_report_runtime.__file__).resolve())}
audit_path = execution_dir / 'no-training-audit.json'
def write_no_training_audit():
    audit_path.write_text(json.dumps(no_training_audit, sort_keys=True), encoding='utf-8')
def no_training_guard(label):
    def forbidden(*args, **kwargs):
        no_training_audit['phase'] = 'violation'
        no_training_audit['violations'].append(label)
        write_no_training_audit()
        raise AssertionError('Saved-result training fence: ' + label)
    return forbidden
audited_report_runtime.execute_candidate = no_training_guard('report_runtime.execute_candidate')
audited_experiment_objects.execute_candidate = no_training_guard('experiment_objects.execute_candidate')
audited_task_training.train_task = no_training_guard('task_training.train_task')
audited_transport.run_json_worker = no_training_guard('transport.run_json_worker')
original_import = builtins.__import__
def no_torch_import(name, *args, _original=original_import,
                    _forbid=no_training_guard('torch import'), **kwargs):
    if name == 'torch' or name.startswith('torch.'):
        return _forbid(name)
    return _original(name, *args, **kwargs)
builtins.__import__ = no_torch_import
assert 'torch' not in sys.modules, 'Fresh report kernel unexpectedly loaded Torch'
write_no_training_audit()
"""
        final = next(c for c in reversed(notebook.cells) if c.cell_type == 'code')
        final.source += """
assert no_training_audit['violations'] == []
assert 'torch' not in sys.modules
builtins.__import__ = original_import
no_training_audit.update(phase='completed', torch_loaded=False)
write_no_training_audit()
"""
        return notebook

    def test_actual_saved_report_kernel_forbids_training_worker_and_torch_import(self):
        successful = self.trial()
        failed = self.trial('failed', python='/nonexistent/private-report-fence-python')
        ref = self.recipe('result-report', [successful['reference'], failed['reference']])
        before = self.w.db.serialize()
        with patch('app.notebooks.build_notebook', side_effect=self.no_training_kernel):
            directory, attempt, terminal = self.execute(ref, destination='guarded-saved-report')
        audit = json.loads((attempt / 'no-training-audit.json').read_text())
        self.assertEqual(audit['phase'], 'completed'); self.assertEqual(audit['violations'], [])
        self.assertFalse(audit['torch_loaded']); self.assertGreater(audit['kernel_pid'], 0)
        self.assertTrue(Path(audit['runtime']).is_relative_to(directory / 'runtime'))
        view = json.loads((attempt / 'report-data.json').read_text())
        self.assertFalse(view['notebook_execution']['training_requested'])
        self.assertEqual([r['reference'] for r in view['runs']], [successful['reference'], failed['reference']])
        self.assertEqual([r['status'] for r in view['runs']], ['completed', 'failed'])
        self.assertEqual(view['runs'][0]['checkpoint_id'], successful['record']['attempts'][0]['result']['checkpoint_id'])
        self.assertFalse(list(attempt.glob('run-*.json')))
        self.assertIn('no-training-audit.json', terminal['execution']['artifacts'])
        self.assertEqual(self.w.db.serialize(), before)

    def test_actual_rerun_kernel_is_rejected_by_same_training_fence_negative_control(self):
        ref = self.recipe(); before = self.w.db.serialize()
        with patch('app.notebooks.build_notebook', side_effect=self.no_training_kernel):
            exported = create_bundle(self.w, ref, self.root / 'guarded-rerun')
        terminal = execute_bundle(exported['path'], timeout=60)
        self.assertEqual(terminal['execution']['status'], 'failed')
        self.assertEqual(terminal['execution']['error']['type'], 'CellExecutionError')
        attempt = Path(terminal['path']); audit = json.loads((attempt / 'no-training-audit.json').read_text())
        self.assertEqual(audit['phase'], 'violation')
        self.assertEqual(audit['violations'], ['report_runtime.execute_candidate'])
        notebook = nbformat.read(attempt / 'executed.ipynb', as_version=4)
        failure = next(c for c in notebook.cells if c.metadata.visual_pytorch.section == 'resolution')
        self.assertIn('Saved-result training fence', json.dumps(failure.outputs))
        later = notebook.cells[notebook.cells.index(failure) + 1:]
        self.assertTrue(all(c.execution_count is None for c in later if c.cell_type == 'code'))
        self.assertFalse((attempt / 'report-data.json').exists()); self.assertFalse(list(attempt.glob('run-*.json')))
        self.assertIn('no-training-audit.json', terminal['execution']['artifacts'])
        self.assertEqual(self.w.db.serialize(), before)

    def test_generation_validates_format_stable_ids_no_training_or_execution(self):
        ref = self.recipe()
        with patch('app.transport.run_json_worker', side_effect=AssertionError('generation must not train')):
            a = create_bundle(self.w, ref, self.root / 'a')
            b = create_bundle(self.w, ref, self.root / 'b')
        self.assertEqual(a['status'], 'generated-not-executed')
        x = nbformat.read(Path(a['path']) / 'requested.ipynb', as_version=4)
        y = nbformat.read(Path(b['path']) / 'requested.ipynb', as_version=4)
        nbformat.validate(x); self.assertEqual(x, y)
        self.assertEqual(len({c.id for c in x.cells}), len(x.cells))
        self.assertTrue(all(c.execution_count is None and c.outputs == [] for c in x.cells if c.cell_type == 'code'))
        self.assertIn('explicitly launches', x.cells[0].source)
        self.assertFalse((Path(a['path']) / 'executions').exists())
        self.assertEqual(self.w.list_specs('run'), [])
        _, manifest, _ = verify_bundle(a['path'])
        self.assertIn('runtime/app/task_training.py', manifest['files'])
        self.assertFalse(any('desktop' in p or '/web/' in p or 'config' in p for p in manifest['files']))

    def test_partial_selection_is_explicit_and_coverage_is_visible(self):
        ref = self.recipe()
        with self.assertRaisesRegex(GraphError, 'partial'): create_bundle(self.w, ref, self.root / 'no', limit=2)
        self.assertFalse((self.root / 'no').exists())
        result = create_bundle(self.w, ref, self.root / 'partial', offset=2, limit=2, allow_partial=True)
        nb = nbformat.read(Path(result['path']) / 'requested.ipynb', as_version=4)
        self.assertIn('2–4', nb.cells[0].source); self.assertIn('of 8', nb.cells[0].source)
        payload = json.loads((Path(result['path']) / 'payload.json').read_text())
        self.assertEqual(len(payload['preview']['candidates']), 2)
        with self.assertRaisesRegex(GraphError, 'empty'): create_bundle(self.w, ref, self.root / 'empty', offset=8, allow_partial=True)

    def test_export_never_overwrites_existing_files(self):
        ref = self.recipe(); path = self.root / 'bundle'; create_bundle(self.w, ref, path)
        before = {str(p.relative_to(path)): p.read_bytes() for p in path.rglob('*') if p.is_file()}
        with self.assertRaisesRegex(GraphError, 'overwrite'): create_bundle(self.w, ref, path)
        self.assertEqual(before, {str(p.relative_to(path)): p.read_bytes() for p in path.rglob('*') if p.is_file()})

    def test_captured_payload_does_not_regenerate_split_and_rejects_substitution(self):
        payload = payload_for(self.w, self.recipe())
        with patch('app.experiment_objects.make_split', side_effect=AssertionError('do not regenerate captured splits')):
            resolve_payload(payload)
            selected = payload['preview']['candidates'][0]
            record = execute_candidate(self.w, object_reference('experiment', self.e), selected['coordinates'],
                'captured', 'Captured manifest', saved_manifest=selected['manifest'], timeout=20)
            self.assertEqual(record['record']['attempts'][0]['status'], 'completed')
        bad = copy.deepcopy(payload); bad['preview']['candidates'][0]['manifest']['roles']['train'] = ['r7']
        with self.assertRaises(GraphError): resolve_payload(bad)
        bad = copy.deepcopy(payload); bad['documents'].append(copy.deepcopy(bad['documents'][0]))
        with self.assertRaisesRegex(GraphError, 'closure'): resolve_payload(bad)

    def test_retry_rejects_substituted_captured_manifest_before_worker(self):
        selected = preview_experiment(self.e, self.w.get_spec)['candidates'][0]
        original = self.trial()
        manifest = copy.deepcopy(selected['manifest']); manifest['roles']['train'].reverse()
        with patch('app.transport.run_json_worker', side_effect=AssertionError('retry substitution must not execute')):
            with self.assertRaisesRegex(GraphError, 'substitute'):
                execute_candidate(self.w, object_reference('experiment', self.e), selected['coordinates'],
                    'first', 'Selected trial', retry=True, saved_manifest=manifest)
        self.assertEqual(self.w.get_spec('run', 'first'), original['record'])

    def test_actual_rerun_preserves_excluded_rows_and_displays_preview_before_training(self):
        e = copy.deepcopy(self.e); e.pop('revision')
        e['selection']['exclusions'] = [{'model': e['models'][0], 'data_task': 'pair0', 'reason': 'Excluded control'}]
        ref = self.w.put_spec('experiment', e)
        self.e = self.w.get_spec('experiment', ref['id'], ref['revision'])
        directory, attempt, result = self.execute(self.recipe(), destination='excluded', limit=3, allow_partial=True)
        view = json.loads((attempt / 'report-data.json').read_text())
        self.assertEqual([r['status'] for r in view['runs']], ['unavailable', 'unavailable', 'completed'])
        self.assertEqual([r['metrics'] for r in view['runs'][:2]], [{}, {}])
        nb = nbformat.read(attempt / 'executed.ipynb', as_version=4)
        preview = next(c for c in nb.cells if c.metadata.visual_pytorch.section == 'selection-preview')
        resolution = next(c for c in nb.cells if c.metadata.visual_pytorch.section == 'resolution')
        self.assertLess(preview.execution_count, resolution.execution_count)
        self.assertIn('Excluded control', json.dumps(preview.outputs))
        rows = list(csv.DictReader(io.StringIO((attempt / 'metrics.csv').read_text())))
        self.assertEqual([r['value'] for r in rows[:2]], ['', ''])

    def test_tampered_inputs_or_notebook_fail_before_kernel_launch(self):
        for relative in ('payload.json', 'runtime/app/task_training.py', 'requested.ipynb'):
            directory = self.root / digest(relative)[:12]; create_bundle(self.w, self.recipe(), directory)
            path = directory / relative; path.write_bytes(path.read_bytes() + b' ')
            with patch('app.notebooks._jupyter', side_effect=AssertionError('must reject bytes before kernel')):
                with self.assertRaisesRegex(GraphError, 'integrity|bytes'): execute_bundle(directory)
            self.assertFalse((directory / 'executions').exists())

    def test_symlink_and_incomplete_source_closure_are_rejected(self):
        directory = self.root / 'bundle'; create_bundle(self.w, self.recipe(), directory)
        path = directory / 'runtime/app/core.py'; original = path.read_bytes()
        elsewhere = self.root / 'outside.py'; elsewhere.write_bytes(original); path.unlink(); path.symlink_to(elsewhere)
        with self.assertRaisesRegex(GraphError, 'symlink|escapes'): verify_bundle(directory)
        path.unlink(); path.write_bytes(original)
        manifest_path = directory / 'manifest.json'; manifest = json.loads(manifest_path.read_text())
        manifest['files'].pop('runtime/app/task_runtime.py'); manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(GraphError, 'closure'): verify_bundle(directory)

    def test_actual_result_kernel_reads_exact_evidence_preserves_failure_and_never_trains(self):
        successful = self.trial(); failed = self.trial('failed', python='/nonexistent/private-notebook-python')
        ref = self.recipe('result-report', [successful['reference'], failed['reference']])
        with patch('app.transport.run_json_worker', side_effect=AssertionError('result report must not train')):
            directory, attempt, result = self.execute(ref)
        view = json.loads((attempt / 'report-data.json').read_text())
        self.assertFalse(view['notebook_execution']['training_requested'])
        self.assertEqual([r['status'] for r in view['runs']], ['completed', 'failed'])
        self.assertEqual(view['runs'][0]['checkpoint_id'], successful['record']['attempts'][0]['result']['checkpoint_id'])
        self.assertEqual(view['runs'][1]['metrics'], {})
        self.assertEqual((directory / 'requested.ipynb').read_bytes(), (attempt / 'requested.ipynb').read_bytes())
        executed = nbformat.read(attempt / 'executed.ipynb', as_version=4)
        self.assertTrue(all(c.execution_count is not None for c in executed.cells if c.cell_type == 'code'))
        self.assertTrue(list(attempt.glob('curve-*.svg'))); self.assertEqual(len(list(attempt.glob('predictor-*.py'))), 1)
        curve = next(attempt.glob('curve-*.svg')); tree = ET.fromstring(curve.read_text())
        metadata = json.loads(tree.find('{http://www.w3.org/2000/svg}metadata').text)
        self.assertEqual(metadata['final_checkpoint'], view['runs'][0]['checkpoint_id'])
        raw_rows = list(csv.DictReader(io.StringIO(curve.with_suffix('.csv').read_text())))
        history = successful['record']['attempts'][0]['result']['history']
        self.assertEqual(len(raw_rows), 2 * len(history))
        for point, raw in zip(history, raw_rows[::2]):
            self.assertEqual(raw['checkpoint'], point['checkpoint_id'])
            self.assertEqual(float(raw['value']), point['metrics']['train']['heads'][raw['head']]['value'])
        self.assertEqual(result['execution']['engine'], 'nbclient-real-ipykernel')
        rows = list(csv.DictReader(io.StringIO((attempt / 'metrics.csv').read_text())))
        self.assertEqual(len(rows), 4); self.assertEqual(rows[-1]['value'], '')
        self.assertEqual(self.w.list_specs('run')[0]['id'], 'failed')

    def test_actual_rerun_kernel_portable_eight_candidates_and_repeated_attempts(self):
        directory = self.root / 'source'; ref = self.recipe()
        create_bundle(self.w, ref, directory)
        unrelated = self.root / 'unrelated' / 'portable'; shutil.copytree(directory, unrelated)
        original = (directory / 'requested.ipynb').read_bytes()
        a = execute_bundle(unrelated, timeout=60); b = execute_bundle(unrelated, timeout=60)
        for result in (a, b): self.assertEqual(result['execution']['status'], 'completed', result['execution']['error'])
        av = json.loads((Path(a['path']) / 'report-data.json').read_text())
        bv = json.loads((Path(b['path']) / 'report-data.json').read_text())
        self.assertEqual(len(av['runs']), 8); self.assertTrue(av['notebook_execution']['training_requested'])
        self.assertEqual([r['checkpoint_id'] for r in av['runs']], [r['checkpoint_id'] for r in bv['runs']])
        self.assertEqual([r['metrics'] for r in av['runs']], [r['metrics'] for r in bv['runs']])
        for run, selected in zip(av['runs'], json.loads((unrelated / 'payload.json').read_text())['preview']['candidates']):
            self.assertEqual(run['resolved']['manifest'], selected['manifest'])
            self.assertEqual(run['status'], 'completed')
        self.assertEqual(len(list(Path(a['path']).glob('predictor-*.py'))), 8)
        self.assertEqual(len(list(Path(a['path']).glob('run-*.json'))), 8)
        self.assertFalse((directory / 'executions').exists()); self.assertEqual((unrelated / 'requested.ipynb').read_bytes(), original)
        self.assertEqual(self.w.list_specs('run'), [])

    def test_actual_cell_error_preserves_prior_evidence_and_unexecuted_later_cells(self):
        ref = self.recipe('result-report', [self.trial()['reference']])
        def faulty(payload, index):
            nb = build_notebook(payload, index)
            next(c for c in nb.cells if c.metadata.visual_pytorch.section == 'question').source = "raise RuntimeError('deliberate private analysis failure')"
            return nb
        with patch('app.notebooks.build_notebook', side_effect=faulty):
            exported = create_bundle(self.w, ref, self.root / 'failure')
        result = execute_bundle(exported['path'], timeout=60)
        self.assertEqual(result['execution']['status'], 'failed')
        self.assertEqual(result['execution']['error']['type'], 'CellExecutionError')
        attempt = Path(result['path']); self.assertTrue((attempt / 'report-data.json').is_file())
        nb = nbformat.read(attempt / 'executed.ipynb', as_version=4)
        errors = [o for c in nb.cells if c.cell_type == 'code' for o in c.outputs if o.output_type == 'error']
        self.assertEqual(len(errors), 1); self.assertEqual(errors[0].ename, 'RuntimeError')
        question = next(c for c in nb.cells if c.metadata.visual_pytorch.section == 'question')
        after = nb.cells[nb.cells.index(question) + 1:]
        self.assertTrue(all(c.execution_count is None for c in after if c.cell_type == 'code'))
        self.assertEqual((attempt / 'requested.ipynb').read_bytes(), (Path(exported['path']) / 'requested.ipynb').read_bytes())

    def test_actual_softmax_rerun_kernel_contains_complete_runtime_and_probability_predictor(self):
        from app.core import default_model
        from evidence_fixture import experiment
        from test_authoring import measurements
        m=default_model('Explicit probabilities',inputs=1,outputs=3)
        m['graph']['nodes'].insert(2,{'id':'probabilities','type':'Activation','activation':'Softmax','dim':-1})
        m['graph']['edges']=[{'from':'input','to':'linear1'},{'from':'linear1','to':'probabilities'},
                             {'from':'probabilities','to':'output'}]
        data=measurements();data['fields'][1]['shape']=[3]
        for row in data['records']:row['values']['y']=[.2,.3,.5]
        task={'version':1,'id':'probability-task','name':'Probability regression',
              'inputs':[{'port':'input','field':'x','transforms':[]}],
              'heads':[{'id':'probability','port':'output','field':'y','loss':'mse','weight':1.,'transforms':[]}]}
        ref=experiment(self.w,m,data,task,'softmax-notebook',updates=3,history_limit=4)
        e=self.w.get_spec('experiment',ref['id'],ref['revision']);e.pop('revision');e['initialization_seeds']=[7]
        ref=self.w.put_spec('experiment',e);self.e=self.w.get_spec('experiment',ref['id'],ref['revision'])
        recipe=self.recipe();before=self.w.db.serialize()
        directory,attempt,result=self.execute(recipe,destination='softmax-rerun')
        self.assertTrue((directory/'runtime/app/activations.py').is_file())
        view=json.loads((attempt/'report-data.json').read_text());self.assertEqual(len(view['runs']),1)
        self.assertEqual(view['runs'][0]['status'],'completed')
        predictor=next(attempt.glob('predictor-*.py'));self.assertIn('nn.Softmax(dim=1)',predictor.read_text())
        probe='import runpy,sys,torch;ns=runpy.run_path(sys.argv[1]);y=ns["model"](torch.tensor([[1.],[2.]]));assert tuple(y.shape)==(2,3);assert torch.allclose(y.sum(1),torch.ones(2));assert not any(n=="app" or n.startswith("app.") for n in sys.modules);print("PASS")'
        check=subprocess.run([sys.executable,'-c',probe,str(predictor)],text=True,capture_output=True,cwd=self.root,timeout=20)
        self.assertEqual(check.returncode,0,check.stderr);self.assertEqual(check.stdout.strip(),'PASS')
        self.assertEqual(self.w.db.serialize(),before);self.assertEqual(self.w.list_specs('run'),[])

    def test_actual_pre_softmax_bundle_keeps_its_original_runtime_and_executes(self):
        import hashlib
        import zipfile
        from app.notebooks import RUNTIME
        fixture=Path(__file__).parent/'fixtures/legacy-report-runtime-v1.zip'
        self.assertEqual(hashlib.sha256(fixture.read_bytes()).hexdigest(),
                         '7114ff088dd8acfb72bc92704cadaf45473ee1b0b61c222a4d62615a269e1c8a')
        legacy=self.root/'legacy';legacy.mkdir()
        with zipfile.ZipFile(fixture) as archive:
            self.assertTrue(all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in archive.namelist()))
            archive.extractall(legacy)
        run=self.trial('legacy-failed',python='/missing/legacy-notebook-python')
        recipe=self.recipe('result-report',[run['reference']]);before=self.w.db.serialize()
        old_runtime=tuple(n for n in RUNTIME if n not in ('activations.py', 'families.py', 'family_records.py', 'family_axes.py', 'runtime_shapes.py','runtime_tasks.py','family_export.py','recurrent.py','recurrent_nodes.py'))
        with patch('app.notebooks.RUNTIME',old_runtime),patch('app.notebooks.__file__',str(legacy/'app/notebooks.py')):
            exported=create_bundle(self.w,recipe,self.root/'legacy-bundle')
        root,manifest,notebook=verify_bundle(exported['path'])
        self.assertNotIn('runtime/app/activations.py',manifest['files'])
        original=(root/'requested.ipynb').read_bytes()
        result=execute_bundle(root,timeout=60)
        self.assertEqual(result['execution']['status'],'completed',result['execution']['error'])
        self.assertEqual((root/'requested.ipynb').read_bytes(),original)
        self.assertFalse((root/'runtime/app/activations.py').exists())
        view=json.loads((Path(result['path'])/'report-data.json').read_text())
        self.assertFalse(view['notebook_execution']['training_requested'])
        self.assertEqual(self.w.db.serialize(),before)

    def test_actual_pre_family_bundle_keeps_its_original_runtime_and_executes(self):
        import hashlib
        import zipfile
        from app.notebooks import RUNTIME
        fixture = Path(__file__).parent / 'fixtures/pre-family-report-runtime-v1.zip'
        self.assertEqual(hashlib.sha256(fixture.read_bytes()).hexdigest(),
                         'd169eccc0681f13750e22479f6e3192d4a9f5f16abdc6344110984e5224764a0')
        legacy = self.root / 'pre-family'; legacy.mkdir()
        with zipfile.ZipFile(fixture) as archive:
            self.assertTrue(all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in archive.namelist()))
            archive.extractall(legacy)
        run = self.trial('pre-family-failed', python='/missing/pre-family-python')
        recipe = self.recipe('result-report', [run['reference']]); before = self.w.db.serialize()
        old_runtime = tuple(n for n in RUNTIME if n not in ('families.py', 'family_records.py', 'family_axes.py', 'runtime_shapes.py','runtime_tasks.py','family_export.py','recurrent.py','recurrent_nodes.py'))
        with patch('app.notebooks.RUNTIME', old_runtime), patch('app.notebooks.__file__', str(legacy / 'app/notebooks.py')):
            exported = create_bundle(self.w, recipe, self.root / 'pre-family-bundle')
        root, manifest, _ = verify_bundle(exported['path'])
        self.assertEqual(len([n for n in manifest['files'] if n.startswith('runtime/app/')]), 21)
        self.assertIn('runtime/app/activations.py', manifest['files'])
        original = (root / 'requested.ipynb').read_bytes()
        result = execute_bundle(root, timeout=60)
        self.assertEqual(result['execution']['status'], 'completed', result['execution']['error'])
        self.assertEqual((root / 'requested.ipynb').read_bytes(), original)
        self.assertFalse((root / 'runtime/app/families.py').exists())
        view = json.loads((Path(result['path']) / 'report-data.json').read_text())
        self.assertFalse(view['notebook_execution']['training_requested'])
        self.assertEqual(self.w.db.serialize(), before)

    def test_new_runtime_cannot_omit_both_family_dependencies(self):
        exported = create_bundle(self.w, self.recipe(), self.root / 'incomplete-family')
        root = Path(exported['path']); manifest = json.loads((root / 'manifest.json').read_text())
        for name in ('families.py', 'family_records.py'):
            relative = 'runtime/app/' + name
            manifest['files'].pop(relative); (root / relative).unlink()
        (root / 'manifest.json').write_text(json.dumps(manifest))
        with patch('app.notebooks._jupyter', side_effect=AssertionError('must reject before kernel')):
            with self.assertRaisesRegex(GraphError, 'requires its captured family'):
                execute_bundle(root)
        self.assertFalse((root / 'executions').exists())

    def test_new_runtime_cannot_omit_runtime_shapes_even_if_fixed_execution_requested(self):
        exported = create_bundle(self.w, self.recipe(), self.root / 'incomplete-runtime-shapes')
        root = Path(exported['path']); manifest = json.loads((root / 'manifest.json').read_text())
        relative = 'runtime/app/runtime_shapes.py'
        self.assertIn(relative, manifest['files'])
        manifest['files'].pop(relative); (root / relative).unlink()
        (root / 'manifest.json').write_text(json.dumps(manifest))
        with patch('app.notebooks._jupyter', side_effect=AssertionError('must reject before kernel')):
            with self.assertRaisesRegex(GraphError, 'requires its captured runtime_shapes.py'):
                execute_bundle(root)
        self.assertFalse((root / 'executions').exists())

    def test_partial_family_runtime_is_not_a_supported_legacy_closure(self):
        for name in ('families.py', 'family_records.py'):
            with self.subTest(name=name):
                exported = create_bundle(self.w, self.recipe(), self.root / name)
                root = Path(exported['path']); manifest = json.loads((root / 'manifest.json').read_text())
                relative = 'runtime/app/' + name
                manifest['files'].pop(relative); (root / relative).unlink()
                (root / 'manifest.json').write_text(json.dumps(manifest))
                with self.assertRaisesRegex(GraphError, 'closure'):
                    verify_bundle(root)

    def test_new_runtime_cannot_omit_task_or_export_dependency_before_kernel(self):
        for name in ('runtime_tasks.py','family_export.py','recurrent.py','recurrent_nodes.py'):
            with self.subTest(name=name):
                exported=create_bundle(self.w,self.recipe(),self.root/('incomplete-'+name))
                root=Path(exported['path']);manifest=json.loads((root/'manifest.json').read_text())
                relative='runtime/app/'+name;manifest['files'].pop(relative);(root/relative).unlink()
                (root/'manifest.json').write_text(json.dumps(manifest))
                with patch('app.notebooks._jupyter',side_effect=AssertionError('No kernel')):
                    with self.assertRaisesRegex(GraphError,'requires its captured '+name):execute_bundle(root)
                self.assertFalse((root/'executions').exists())

    def test_new_runtime_cannot_masquerade_as_old_bundle_to_omit_activation_dependency(self):
        ref=self.recipe();exported=create_bundle(self.w,ref,self.root/'incomplete-activation')
        root=Path(exported['path']);manifest=json.loads((root/'manifest.json').read_text())
        manifest['files'].pop('runtime/app/activations.py')
        (root/'runtime/app/activations.py').unlink()
        (root/'manifest.json').write_text(json.dumps(manifest))
        with self.assertRaisesRegex(GraphError,'requires its captured activations.py'):
            verify_bundle(root)

    def test_actual_cell_timeout_retains_terminal_failure(self):
        def slow(payload, index):
            nb = build_notebook(payload, index)
            next(c for c in nb.cells if c.metadata.visual_pytorch.section == 'resolution').source = 'import time; time.sleep(4)'
            return nb
        with patch('app.notebooks.build_notebook', side_effect=slow):
            exported = create_bundle(self.w, self.recipe(), self.root / 'timeout', limit=1, allow_partial=True)
        result = execute_bundle(exported['path'], timeout=1)
        self.assertEqual(result['execution']['status'], 'failed')
        self.assertEqual(result['execution']['error']['type'], 'CellTimeoutError', result['execution']['error'])
        self.assertTrue((Path(result['path']) / 'executed.ipynb').exists())
        self.assertFalse((Path(result['path']) / 'report-data.json').exists())

    def test_actual_cli_generation_and_kernel_execution_keep_workspace_readonly_host_torch_free(self):
        ref = self.recipe('result-report', [self.trial()['reference']])
        path = self.root / 'lab.vtw'; self.w.save(path); before = path.read_bytes()
        source = """import contextlib, io, json, sys
import vpw_cli
with contextlib.redirect_stdout(io.StringIO()) as out:
    code = vpw_cli.main(sys.argv[1:])
print(json.dumps({'code': code, 'output': json.loads(out.getvalue()), 'torch_host': 'torch' in sys.modules}))
"""
        for args in ([str(path), 'generate-report', ref['id'], ref['revision'], '--output', str(self.root / 'cli')],
                     [str(path), 'execute-report', str(self.root / 'cli'), '--cell-timeout', '60']):
            process = subprocess.run([sys.executable, '-c', source, *args], capture_output=True, text=True, timeout=90)
            self.assertEqual(process.returncode, 0, process.stderr)
            result = json.loads(process.stdout); self.assertEqual(result['code'], 0, result); self.assertFalse(result['torch_host'])
        self.assertEqual(result['output']['value']['execution']['status'], 'completed')
        self.assertEqual(path.read_bytes(), before)

    def test_notebook_predictor_runs_outside_bundle_with_independent_scalar_output(self):
        run = self.trial(); ref = self.recipe('result-report', [run['reference']])
        _, attempt, _ = self.execute(ref)
        predictor = next(attempt.glob('predictor-*.py'))
        target = self.root / 'only-predictor.py'; shutil.copyfile(predictor, target)
        source = """import importlib.util, json, sys
spec=importlib.util.spec_from_file_location('standalone', sys.argv[1]); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
print(json.dumps({'prediction':m.predict_records([{'x':[1.25]}])['output'].item(), 'app_imported':'app' in sys.modules}))
"""
        child = subprocess.run([sys.executable, '-c', source, str(target)], cwd=self.root, capture_output=True, text=True, timeout=20)
        self.assertEqual(child.returncode, 0, child.stderr)
        value = json.loads(child.stdout); weights = run['record']['attempts'][0]['result']['checkpoint']['weights']
        expected = weights['layer_0.weight'][0][0] * 1.25 + weights['layer_0.bias'][0]
        self.assertAlmostEqual(value['prediction'], expected, places=6); self.assertFalse(value['app_imported'])


class ReportCurveTests(unittest.TestCase):
    def test_coordinates_missingness_redundant_encoding_and_full_identity(self):
        summary = {'reference': {'kind': 'run', 'id': 'long-run-' + 'a' * 80, 'revision': 'b' * 64}}
        def point(update, value):
            return {'update': update, 'metrics': {r: {'heads': {'head': {'kind': 'mse', 'value': value if r == 'train' else 2.}}}
                                                 for r in ('train', 'validation')}}
        result = {'history': [point(0, 4.), point(1, None), point(2, 1.), point(3, 0.)],
            'checkpoint_id': 'c' * 64, 'history_retention': {'observed_points': 21, 'discarded_points': 17}}
        context = {'model': 'Model <script>', 'dataset': 'Recorded observations', 'unit': 'm', 'transforms': [{'op': 'standardize'}]}
        svg = curve_svg(summary, result, 'head', context); tree = ET.fromstring(svg)
        ns = {'s': 'http://www.w3.org/2000/svg'}
        meta = json.loads(tree.find('s:metadata', ns).text)
        self.assertEqual(meta['series']['train'], [[0, 4.], [1, None], [2, 1.], [3, 0.]])
        self.assertEqual(meta['run'], summary['reference']); self.assertEqual(meta['context'], context)
        traces = tree.findall('.//s:polyline', ns)
        train = [p for p in traces if p.attrib['data-population'] == 'train']
        validation = [p for p in traces if p.attrib['data-population'] == 'validation']
        self.assertEqual(len(train), 2, 'Unavailable update must break, not interpolate, the train trace')
        self.assertEqual(len(validation), 1); self.assertEqual(validation[0].attrib['stroke-dasharray'], '6 4')
        xy = [list(map(float, p.split(','))) for p in train[-1].attrib['points'].split()]
        self.assertAlmostEqual(xy[0][0], 512., places=5); self.assertAlmostEqual(xy[-1][0], 732., places=5)
        self.assertAlmostEqual(xy[0][1], 270. - 224. / 4.2, places=5); self.assertEqual(xy[-1][1], 270.)
        self.assertTrue(tree.findall('.//s:circle', ns)); self.assertIn('Validation: dashed / squares', svg)
        self.assertIn('discarded 17', svg); self.assertIn('no smoothing or confidence intervals', svg)
        self.assertIn('transformed target space', svg); self.assertNotIn('<script>', svg)
        height = int(tree.attrib['height'])
        self.assertTrue(all(float(t.attrib['y']) < height for t in tree.findall('.//s:text', ns)))

    def test_zero_and_entirely_unavailable_histories_are_not_log_filtered(self):
        summary = {'reference': {'kind': 'run', 'id': 'zero', 'revision': 'b' * 64}}
        context = {'model': 'Zero', 'dataset': 'Empty mask', 'unit': '', 'transforms': []}
        result = {'history': [{'update': 0, 'metrics': {r: {'heads': {'h': {'kind': 'mae', 'value': 0.}}}
                                                     for r in ('train', 'validation')}}],
                  'checkpoint_id': 'c' * 64, 'history_retention': {'observed_points': 1, 'discarded_points': 0}}
        svg = curve_svg(summary, result, 'h', context)
        self.assertIn('scale', svg); self.assertIn('72.000000,270.000000', svg)
        for role in ('train', 'validation'): result['history'][0]['metrics'][role]['heads']['h']['value'] = None
        self.assertIsNone(curve_svg(summary, result, 'h', context))


if __name__ == '__main__':
    unittest.main()
