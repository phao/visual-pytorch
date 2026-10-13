"""Real selected-environment generation, pure Qt/CLI host and byte admission."""
import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.core import GraphError
from app.experiment_objects import object_reference, execute_candidate, preview_experiment
from app.notebooks import capture_bundle, publish_bundle, create_bundle, create_bundle_isolated, verify_bundle, RUNTIME
from app.transport import run_json_worker
from test_experiment_objects import experiment_workspace


class NotebookGenerationTests(unittest.TestCase):
    def setUp(self):
        self.w, self.e = experiment_workspace(); self.addCleanup(self.w.db.close)
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.ref = self.w.put_spec('report', {'version': 1, 'id': 'generation', 'name': 'Explicit generation',
            'question': 'What will run?', 'mode': 'rerun', 'runs': [],
            'experiment': object_reference('experiment', self.e), 'sections': ['question', 'protocol', 'failures'],
            'analysis': {'kind': 'descriptive', 'inferential_unit': 'run'}})

    def generate(self, name='bundle', **kw):
        return create_bundle_isolated(self.w, self.ref, self.root / name, sys.executable, **kw)

    def test_capture_requires_no_notebook_dependencies_and_changes_no_sql(self):
        before = self.w.db.serialize()
        with patch('app.notebooks._jupyter', side_effect=AssertionError('Host notebook dependency')):
            captured = capture_bundle(self.w, self.ref)
        self.assertEqual(self.w.db.serialize(), before)
        self.assertEqual(len(captured['payload']['preview']['candidates']), 8)
        self.assertEqual(set(captured['files']), {'payload.json', 'LICENSE', 'README.txt', 'requirements.txt'} |
                         {'runtime/app/' + n for n in RUNTIME})
        self.assertTrue({'parameter_binding.py', 'parameter_sources.py', 'policy_authoring.py'} <= set(RUNTIME))
        self.assertFalse(list(self.root.iterdir()))

    def test_actual_isolated_generation_uses_captured_bytes_and_never_runs_cells(self):
        before = self.w.db.serialize()
        with patch('app.notebooks._jupyter', side_effect=AssertionError('Host notebook dependency')):
            result = self.generate()
        self.assertEqual(result['status'], 'generated-not-executed'); self.assertEqual(result['mode'], 'rerun')
        self.assertFalse((self.root / 'bundle/executions').exists())
        _, _, notebook = verify_bundle(result['path'])
        self.assertTrue(all(c.get('execution_count') is None and not c.get('outputs') for c in notebook.cells))
        self.assertEqual(self.w.db.serialize(), before); self.assertFalse(self.w.list_specs('run'))

    def test_direct_and_isolated_authoring_have_identical_notebook_and_index(self):
        a = create_bundle(self.w, self.ref, self.root / 'direct'); b = self.generate('isolated')
        self.assertEqual(a['input_index_hash'], b['input_index_hash'])
        self.assertEqual(a['requested_sha256'], b['requested_sha256'])
        self.assertEqual((self.root / 'direct/requested.ipynb').read_bytes(), (self.root / 'isolated/requested.ipynb').read_bytes())

    def test_generation_child_import_fences_forbid_training_and_kernel_dependencies(self):
        capture = capture_bundle(self.w, self.ref)
        source = """import sys,importlib.abc
class Forbidden(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in {'torch','PyQt6','nbclient','jupyter_client','ipykernel'}:
            raise AssertionError('Generation imported execution dependency: '+fullname)
sys.meta_path.insert(0,Forbidden())
from app.notebook_worker import main
raise SystemExit(main())
"""
        result = run_json_worker([sys.executable, '-c', source],
            {'operation': 'generate', 'capture': capture, 'directory': str(self.root / 'fenced')}, timeout=20)
        self.assertEqual(result['generated']['status'], 'generated-not-executed')
        self.assertFalse((self.root / 'fenced/executions').exists())

    def test_real_stdlib_only_host_generates_through_selected_environment(self):
        source = """import sys,json,importlib.abc
class Forbidden(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in {'torch','PyQt6','nbformat','nbclient','jupyter_client','ipykernel'}:
            raise AssertionError('Host imported optional dependency: '+fullname)
sys.meta_path.insert(0,Forbidden())
from app.core import Workspace
from app.notebooks import create_bundle_isolated
w=Workspace.load(sys.argv[1]);before=w.db.serialize()
try:
    result=create_bundle_isolated(w,json.loads(sys.argv[2]),sys.argv[3],sys.argv[4])
    assert w.db.serialize()==before
    assert not any(name in sys.modules for name in ('nbformat','torch','PyQt6'))
    print(json.dumps(result))
finally:w.db.close()
"""
        path = self.root / 'source.vtw'; self.w.save(path); before = path.read_bytes()
        run = subprocess.run(['/usr/bin/python3', '-c', source, str(path), json.dumps(self.ref),
                              str(self.root / 'pure-host'), sys.executable], capture_output=True, text=True, timeout=30)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(json.loads(run.stdout)['status'], 'generated-not-executed')
        self.assertEqual(path.read_bytes(), before)

    def test_capture_survives_new_heads_without_substituting_dependencies(self):
        captured = capture_bundle(self.w, self.ref); original = copy.deepcopy(captured)
        doc = self.w.get_spec('dataset', self.e['data_tasks'][0]['dataset']['id']); doc.pop('revision')
        doc['name'] = 'New browsing head'; self.w.put_spec('dataset', doc)
        result = publish_bundle(captured, self.root / 'old-capture')
        self.assertEqual(captured, original)
        self.assertEqual(json.loads((Path(result['path']) / 'payload.json').read_text()), original['payload'])

    def test_closed_capture_refuses_unsupported_shapes_versions_and_paths_before_writes(self):
        for change in ('version', 'extra', 'missing-source', 'traversal', 'non-text', 'payload-mismatch'):
            captured = capture_bundle(self.w, self.ref)
            if change == 'version': captured['version'] = True
            elif change == 'extra': captured['extra'] = 1
            elif change == 'missing-source': captured['files'].pop('runtime/app/graph.py')
            elif change == 'traversal': captured['files']['../outside.py'] = 'bad'
            elif change == 'non-text': captured['files']['LICENSE'] = 7
            else: captured['files']['payload.json'] = '{}'
            with self.subTest(change=change), patch('app.notebooks.build_notebook', side_effect=AssertionError('must not author')):
                with self.assertRaises(GraphError): publish_bundle(captured, self.root / change)
            self.assertFalse((self.root / change).exists())

    def test_payload_with_duplicate_keys_fails_before_output_creation(self):
        captured = capture_bundle(self.w, self.ref)
        captured['files']['payload.json'] = '{"version":1,"version":1}'
        with self.assertRaises(GraphError): publish_bundle(captured, self.root / 'duplicate')
        self.assertFalse((self.root / 'duplicate').exists())

    def test_missing_or_inadequate_report_environment_does_not_create_bundle(self):
        for python in ('/no-such-report-python', '/usr/bin/python3'):
            with self.subTest(python=python), self.assertRaises(ValueError):
                create_bundle_isolated(self.w, self.ref, self.root / 'missing', python)
            self.assertFalse((self.root / 'missing').exists())

    def test_existing_destination_refuses_before_launch_and_bytes_are_unchanged(self):
        result = self.generate(); root = Path(result['path'])
        before = {str(p.relative_to(root)): p.read_bytes() for p in root.rglob('*') if p.is_file()}
        with patch('app.transport.run_json_worker', side_effect=AssertionError('Must not launch')):
            with self.assertRaisesRegex(GraphError, 'overwrite'): self.generate()
        self.assertEqual(before, {str(p.relative_to(root)): p.read_bytes() for p in root.rglob('*') if p.is_file()})

    def test_partial_selection_requires_permission_and_remains_visible(self):
        with self.assertRaisesRegex(GraphError, 'partial'): self.generate('refused', limit=2)
        self.assertFalse((self.root / 'refused').exists())
        result = self.generate('partial', offset=2, limit=2, allow_partial=True)
        payload = json.loads((Path(result['path']) / 'payload.json').read_text())
        self.assertEqual(len(payload['preview']['candidates']), 2)
        self.assertEqual(payload['preview']['total_combinations'], 8)

    def test_saved_result_generation_keeps_failed_evidence_without_launching_training(self):
        row = preview_experiment(self.e, self.w.get_spec)['candidates'][0]
        saved = execute_candidate(self.w, object_reference('experiment', self.e), row['coordinates'],
                                  'failed-generation-input', 'Failure retained', python='/no-such-trainer')
        ref = self.w.put_spec('report', {'version': 1, 'id': 'saved-generation', 'name': 'Saved failure',
            'question': 'What failed?', 'mode': 'result-report', 'runs': [saved['reference']], 'experiment': None,
            'sections': ['question', 'failures'], 'analysis': {'kind': 'descriptive', 'inferential_unit': 'run'}})
        before = self.w.db.serialize()
        result = create_bundle_isolated(self.w, ref, self.root / 'saved', sys.executable)
        self.assertEqual(result['mode'], 'result-report'); self.assertEqual(self.w.db.serialize(), before)
        self.assertFalse((self.root / 'saved/executions').exists())

    def test_corrupt_worker_identity_or_source_cannot_be_admitted(self):
        for change in ('mode', 'selection', 'source', 'manifest', 'executions'):
            def run(*args, **kwargs):
                value = run_json_worker(*args, **kwargs); root = Path(value['generated']['path'])
                if change == 'mode': value['generated']['mode'] = 'result-report'
                elif change == 'selection': value['selection']['input_index_hash'] = '0' * 64
                elif change == 'source': (root / 'runtime/app/graph.py').write_text('changed source')
                elif change == 'manifest': (root / 'manifest.json').write_text('{}')
                else: (root / 'executions').mkdir()
                return value
            before = self.w.db.serialize()
            with self.subTest(change=change), patch('app.transport.run_json_worker', side_effect=run):
                with self.assertRaises(GraphError): self.generate(change)
            self.assertEqual(self.w.db.serialize(), before)

    def test_worker_deadline_and_captured_request_are_explicit(self):
        with patch('app.transport.run_json_worker', side_effect=ValueError('deliberate timeout')) as run:
            with self.assertRaisesRegex(ValueError, 'timeout'): self.generate('timed')
        self.assertEqual(run.call_args.kwargs['timeout'], 30)
        self.assertEqual(run.call_args.kwargs['max_line_bytes'], 1024**2)
        self.assertEqual(run.call_args.args[1]['operation'], 'generate')
        self.assertEqual(set(run.call_args.args[1]), {'operation', 'capture', 'directory'})
        self.assertFalse((self.root / 'timed').exists())

    def test_actual_cli_selected_generation_keeps_source_file_readonly(self):
        path = self.root / 'cli.vtw'; self.w.save(path); before = path.read_bytes()
        source = """import sys,importlib.abc
class Forbidden(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in {'torch','PyQt6','nbformat','nbclient','jupyter_client','ipykernel'}:
            raise AssertionError('CLI host imported optional dependency: '+fullname)
sys.meta_path.insert(0,Forbidden())
import vpw_cli
raise SystemExit(vpw_cli.main(sys.argv[1:]))
"""
        run = subprocess.run(['/usr/bin/python3', '-c', source, str(path), 'generate-report',
                              self.ref['id'], self.ref['revision'], '--output', str(self.root / 'cli-bundle'),
                              '--report-python', sys.executable, '--memory-limit-percent', '10'],
                             capture_output=True, text=True, timeout=30)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(json.loads(run.stdout)['value']['status'], 'generated-not-executed')
        self.assertEqual(path.read_bytes(), before); self.assertFalse((self.root / 'cli-bundle/executions').exists())


if __name__ == '__main__': unittest.main()
