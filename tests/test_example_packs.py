"""Bounded checks of the actual committed MS-VAR bytes and saved evidence."""
import copy
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.core import GraphError, Workspace
from app.evidence import digest
from app.example_packs import check, replay
from app.experiment_objects import execute_candidate
from app.transport import WorkerError, run_json_worker


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'examples/neural-msvar.vtw'
MANIFEST = ROOT / 'examples/neural-msvar-pack.json'


def pack():
    return json.loads(MANIFEST.read_text())


def state(workspace):
    """Evidence of SQL, session/history, disk bytes and directory immutability."""
    history = workspace.history
    source = Path(workspace.path)
    return (tuple(workspace.db.iterdump()), workspace.db.total_changes,
            workspace.db.in_transaction, workspace.dirty, workspace.path,
            workspace.disk_revision, workspace.disk_stamp,
            tuple(history.undo_stack), tuple(history.redo_stack), history.notice,
            history.directory, copy.deepcopy(vars(history.registry_validation)),
            source.read_bytes(), source.stat().st_mtime_ns,
            tuple(sorted(p.name for p in source.parent.iterdir())))


class ExamplePackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workspace = Workspace.load(SOURCE)

    @classmethod
    def tearDownClass(cls):
        cls.workspace.db.close()

    def reject(self, document, where):
        before = state(self.workspace)
        with patch('app.example_packs.execute_candidate', side_effect=AssertionError('automatic execution')):
            with self.assertRaises(GraphError) as error:
                check(self.workspace, document)
        self.assertIn(where, str(error.exception))
        self.assertEqual(state(self.workspace), before)

    def test_check_actual_saved_pack_is_read_only_and_never_executes(self):
        before = state(self.workspace)
        with (patch('app.example_packs.execute_candidate', side_effect=AssertionError('automatic execution')),
             patch.object(Workspace, 'save', side_effect=AssertionError('Save')),
             patch.object(self.workspace.history, 'capture', side_effect=AssertionError('history capture'))):
            result = check(self.workspace, MANIFEST)
        self.assertEqual(result['status'], 'checked')
        self.assertIs(result['executed'], False)
        self.assertEqual(result['counts']['completed_updates'], 12)
        self.assertIn('Exact conditional', result['claim'])
        self.assertIn('estimated parameters', result['claim'])
        self.assertEqual(state(self.workspace), before)

    def test_path_check_uses_canonical_workspace_loader(self):
        with patch.object(Workspace, 'load', wraps=Workspace.load) as loader:
            result = check(SOURCE, MANIFEST)
        loader.assert_called_once_with(SOURCE)
        self.assertEqual(result['workspace_sha256'], hashlib.sha256(SOURCE.read_bytes()).hexdigest())

    def test_existing_history_files_and_stacks_are_preserved(self):
        workspace = Workspace.load(SOURCE)
        try:
            snapshot = workspace.history.capture(workspace.db)
            workspace.history.undo_stack.append(snapshot)
            before, saved_history = state(workspace), snapshot.read_bytes()
            check(workspace, MANIFEST)
            self.assertEqual(state(workspace), before)
            self.assertEqual(snapshot.read_bytes(), saved_history)
        finally:
            workspace.db.close()
            if workspace.history.directory:
                workspace.history.directory.cleanup()

    def test_no_site_standard_library_imports_and_check(self):
        script = '''
import importlib.abc, pathlib, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'app.config' or fullname.split('.')[0] in ('torch', 'numpy', 'pandas', 'PySide6'):
            raise AssertionError('forbidden import: ' + fullname)
sys.meta_path.insert(0, Block())
from app.example_packs import check
from app.core import Workspace
root = pathlib.Path.cwd()
source = root / 'examples/neural-msvar.vtw'
before = source.read_bytes(), source.stat().st_mtime_ns
w = Workspace.load(source)
sql = tuple(w.db.iterdump()), w.db.total_changes
history = w.history.summary(), w.history.directory, w.dirty
result = check(w, root / 'examples/neural-msvar-pack.json')
assert result['executed'] is False and result['status'] == 'checked'
assert sql == (tuple(w.db.iterdump()), w.db.total_changes)
assert history == (w.history.summary(), w.history.directory, w.dirty)
assert before == (source.read_bytes(), source.stat().st_mtime_ns)
assert not any(n == 'app.config' or n.split('.')[0] in ('torch','numpy','pandas','PySide6') for n in sys.modules)
w.db.close()
'''
        result = subprocess.run(['/usr/bin/python3', '-S', '-c', script], cwd=ROOT,
            env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1',
                     MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1'),
            capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_wrong_hash_or_missing_exact_reference(self):
        document = pack(); document['workspace_sha256'] = '0' * 64
        self.reject(document, '$.workspace_sha256')
        for kind in document['references']:
            with self.subTest(kind=kind):
                document = pack(); document['references'][kind]['revision'] = '0' * 64
                self.reject(document, '$.references.' + kind)

    def test_existing_wrong_reference_is_rejected_by_meaning(self):
        workspace = Workspace.load(SOURCE)
        try:
            document = pack()
            ref = document['references']['task']
            task = workspace.get_spec('task', ref['id'], ref['revision'])
            task.pop('revision'); task['id'] = 'other-density-task'
            task['heads'][0]['port'] = 'out_step_evidence'
            document['references']['task'] = workspace.put_spec('task', task)
            before = state(workspace)
            with self.assertRaisesRegex(GraphError, 'saved run execution'):
                check(workspace, document)
            self.assertEqual(state(workspace), before)
        finally:
            workspace.db.close()

    def test_unsaved_graph_semantics_are_checked_despite_valid_disk_hash_and_refs(self):
        workspace = Workspace.load(SOURCE)
        try:
            model = workspace.models()['neural-msvar']
            next(n for n in model['graph']['nodes'] if n['id'] == 'variances')['positive_floor'] = .001
            workspace.put(model)
            before = state(workspace)
            with self.assertRaisesRegex(GraphError, 'graph meaning differs'):
                check(workspace, MANIFEST)
            self.assertEqual(state(workspace), before)
        finally:
            workspace.db.close()

    def test_wrong_recipe_bindings_and_scalar_types(self):
        for key, value in (('K', 4), ('p', 2), ('D', 2), ('hidden', 31), ('T', 11),
                           ('K', True), ('p', 3.0), ('D', '3'), ('controlled', True)):
            with self.subTest(key=key, value=value):
                document = pack(); document['architectures'][0]['recipe']['bindings'][key]['value'] = value
                self.reject(document, '$.architectures[0].recipe')
        document = pack(); document['architectures'][0]['recipe']['bindings']['K']['type'] = 'float'
        self.reject(document, '$.architectures[0].recipe.bindings.K')

    def test_classification_and_assumptions_cannot_overstate_evidence(self):
        for kind in ('approximate-estimator', 'parameter-component', 'exact-fit'):
            document = pack(); document['architectures'][0]['intended_use'] = kind
            self.reject(document, '$.architectures[0].intended_use')
        for key in pack()['architectures'][0]['assumptions']:
            with self.subTest(key=key):
                document = pack(); document['architectures'][0]['assumptions'][key] = 'unconditional-exact-fit'
                self.reject(document, '$.architectures[0].assumptions.' + key)

    def test_closed_schema_and_original_source_provenance(self):
        for key, value, where in (('version', True, '$.version'), ('version', 2, '$.version'),
                                 ('producer_source_commit', '0' * 40, '$.producer_source_commit'),
                                 ('id', 'other-pack', '$.id'), ('python', '/usr/bin/python3', '$')):
            document = pack(); document[key] = value
            self.reject(document, where)
        document = pack(); document['verification']['command'] = 'print("no")'
        self.reject(document, '$.verification')
        document = pack(); document['references']['run']['revision'] = 'latest'
        self.reject(document, '$.references.run')

    def test_saved_checkpoint_metrics_and_coordinates_are_verified(self):
        for key in ('checkpoint_sha256', 'metrics_sha256'):
            document = pack(); document['verification']['expected'][key] = '0' * 64
            self.reject(document, '$.verification.expected.' + key)
        document = pack(); document['verification']['coordinates']['initialization'] = 8
        self.reject(document, '$.verification.coordinates')
        document = pack(); document['verification']['coordinates']['initialization'] = 7.0
        self.reject(document, '$.verification.coordinates')

    def test_duplicate_nonfinite_and_malformed_json_fail_clearly(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'manifest.json'
            for content in ('{"version":1,"version":1}', '{"version":NaN}',
                            '{"version":1e999}', '{', '[]', '[' * 2000 + ']' * 2000):
                with self.subTest(content=content[:40]):
                    path.write_text(content)
                    self.reject(path, '$')
        document = pack(); document['command'] = 'x' * (1024 * 1024)
        self.reject(document, '$')

    def test_excess_or_unsupported_budgets_refused_before_execution(self):
        for updates in (0, True, 12.0, 24, 25, 1_000_000):
            document = pack(); document['verification']['budget']['updates'] = updates
            self.reject(document, '$.verification.budget')
            with patch('app.example_packs.execute_candidate', side_effect=AssertionError('execution')):
                with self.assertRaisesRegex(GraphError, 'verification.budget'):
                    replay(self.workspace, document, python=sys.executable, timeout=60)
        document = pack(); document['verification']['budget']['max_updates'] = 25
        self.reject(document, '$.verification.budget')
        document = pack(); document['verification']['budget']['epochs'] = 1
        self.reject(document, '$.verification.budget')

    def test_replay_requires_explicit_bounded_execution_settings(self):
        for timeout in (None, True, 0, float('inf'), 61):
            with patch('app.example_packs.execute_candidate', side_effect=AssertionError('execution')):
                with self.assertRaisesRegex(GraphError, 'timeout'):
                    replay(self.workspace, MANIFEST, python=sys.executable, timeout=timeout)
        with self.assertRaisesRegex(GraphError, 'memory_limit_percent'):
            replay(self.workspace, MANIFEST, python=sys.executable, timeout=60,
                   memory_limit_percent=11)

    def test_transport_failure_is_reported_without_fake_evidence_or_source_changes(self):
        before = state(self.workspace)
        with patch('app.transport.run_json_worker', side_effect=WorkerError('bounded worker failure')):
            result = replay(self.workspace, MANIFEST, python=sys.executable, timeout=60)
        self.assertEqual(result['status'], 'failed')
        self.assertIs(result['verified'], False)
        self.assertIsNone(result['result'])
        self.assertEqual(result['error']['message'], 'bounded worker failure')
        self.assertFalse(any(result['comparisons'].values()))
        self.assertEqual(state(self.workspace), before)


@unittest.skipUnless(importlib.util.find_spec('torch') is not None, 'Torch runtime needed only for explicit replay')
class ExamplePackReplayTests(unittest.TestCase):
    def test_real_native_worker_reproduces_saved_checkpoint_and_metrics_without_source_changes(self):
        workspace = Workspace.load(SOURCE)
        try:
            before = state(workspace)
            with (patch.object(Workspace, 'save', side_effect=AssertionError('Save')),
                 patch.object(workspace.history, 'capture', side_effect=AssertionError('source history')),
                 patch('app.example_packs.execute_candidate', wraps=execute_candidate) as execute,
                 patch('app.transport.run_json_worker', wraps=run_json_worker) as worker):
                result = replay(workspace, MANIFEST, python=sys.executable, timeout=60)
            execute.assert_called_once()
            worker.assert_called_once()
            self.assertIsNot(execute.call_args.args[0], workspace)
            self.assertEqual(worker.call_args.args[0], [sys.executable, '-u', '-m', 'app.worker'])
            self.assertEqual(worker.call_args.args[1]['bundle']['protocol']['budget'],
                             {'epochs': None, 'updates': 12, 'examples': None, 'seconds': None})
            self.assertEqual(result['status'], 'matched', result['error'])
            self.assertTrue(result['verified'])
            self.assertTrue(all(result['comparisons'].values()))
            self.assertEqual(result['result']['counts']['completed_updates'], 12)
            self.assertEqual(digest(result['result']['checkpoint']), pack()['verification']['expected']['checkpoint_sha256'])
            self.assertEqual(digest(result['result']['final_metrics']), pack()['verification']['expected']['metrics_sha256'])
            self.assertEqual(state(workspace), before)
        finally:
            workspace.db.close()
