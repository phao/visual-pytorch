"""Actual kernels, async Stop/cleanup, exact bundle identity and independent runner."""
import copy
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import nbformat
from app.core import GraphError
from app.experiment_objects import object_reference, preview_experiment, execute_candidate
from app.notebooks import build_notebook, create_bundle, execute_bundle
from app.notebook_execution import inspect_bundle, inspect_isolated, run_isolated, validate_result
from test_experiment_objects import experiment_workspace


class NotebookExecutionTests(unittest.TestCase):
    def setUp(self):
        self.w, self.e = experiment_workspace(); self.addCleanup(self.w.db.close)
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def bundle(self, name='bundle', mode='rerun', build=None):
        runs = []
        if mode == 'result-report':
            row = preview_experiment(self.e, self.w.get_spec)['candidates'][0]
            runs = [execute_candidate(self.w, object_reference('experiment', self.e), row['coordinates'],
                'saved', 'Saved failure', python='/no-such-notebook-python')['reference']]
        ref = self.w.put_spec('report', {'version': 1, 'id': name, 'name': 'Explicit notebook <svg>',
            'question': 'What is actually retained?', 'mode': mode, 'runs': runs,
            'experiment': object_reference('experiment', self.e) if mode == 'rerun' else None,
            'sections': ['question', 'training', 'failures'],
            'analysis': {'kind': 'descriptive', 'inferential_unit': 'run'}})
        options = {'limit': 1, 'allow_partial': True} if mode == 'rerun' else {}
        with patch('app.notebooks.build_notebook', side_effect=build or build_notebook):
            value = create_bundle(self.w, ref, self.root / name, **options)
        return inspect_bundle(value['path'])

    def test_isolated_inspection_is_readonly_and_does_not_execute(self):
        selection = self.bundle(); before = self.w.db.serialize()
        result = inspect_isolated(selection['path'], sys.executable)
        self.assertEqual(result['selection'], selection)
        self.assertFalse((Path(selection['path'])/'executions').exists())
        self.assertEqual(self.w.db.serialize(), before)

    def test_actual_isolated_modes_progress_receipts_and_no_source_workspace_changes(self):
        for mode in ('result-report', 'rerun'):
            selection = self.bundle(mode, mode); before = self.w.db.serialize(); events = []
            result = run_isolated(selection, sys.executable, timeout=60, progress=events.append)
            self.assertEqual(result['execution']['status'], 'completed', result)
            self.assertEqual(validate_result(result, selection), result)
            self.assertEqual(self.w.db.serialize(), before)
            self.assertTrue(any(p['stage'] == 'cell-running' for p in events))
            self.assertTrue(any(p['stage'] == 'cell-replied' for p in events))
            self.assertEqual({p['attempt_id'] for p in events}, {result['execution']['attempt_id']})
            view = json.loads((Path(result['path'])/'report-data.json').read_text())
            self.assertEqual(view['notebook_execution']['training_requested'], mode == 'rerun')
            self.assertEqual(view['runs'][0]['status'], 'failed' if mode == 'result-report' else 'completed')

    def test_prelaunch_stop_keeps_real_zero_cell_receipt_without_launching_kernel(self):
        selection = self.bundle(); stop = threading.Event(); stop.set()
        with patch('jupyter_client.manager.AsyncKernelManager', side_effect=AssertionError('no kernel launch')):
            result = execute_bundle(selection['path'], cancel=stop)
        self.assertEqual(result['execution']['status'], 'cancelled')
        self.assertEqual(result['execution']['executed_code_cells'], 0)
        self.assertEqual(result['execution']['error']['type'], 'NotebookCancelled')
        validate_result(result, selection)

    def test_actual_stop_during_long_cell_stops_kernel_and_retains_prior_evidence(self):
        pid_file = self.root/'kernel.pid'
        def slow(payload, index):
            nb = build_notebook(payload, index)
            cell = next(c for c in nb.cells if c.metadata.visual_pytorch.section == 'question')
            cell.source = "import os,time\nfrom pathlib import Path\nPath(" + repr(str(pid_file)) + ").write_text(str(os.getpid()))\nprint('before-stop',flush=True)\ntime.sleep(60)"
            return nb
        selection = self.bundle(mode='result-report', build=slow); stop = threading.Event(); events = []
        def cancel_when_running():
            deadline = time.monotonic()+30
            while not pid_file.exists() and time.monotonic()<deadline:
                time.sleep(.025)
            stop.set()
        helper = threading.Thread(target=cancel_when_running); helper.start()
        started = time.monotonic()
        try: result = run_isolated(selection, sys.executable, timeout=90, cancel=stop, progress=events.append)
        finally: helper.join(35)
        self.assertFalse(helper.is_alive()); self.assertTrue(pid_file.exists())
        self.assertLess(time.monotonic()-started, 30)
        self.assertEqual(result['execution']['status'], 'cancelled', result)
        self.assertEqual(result['execution']['error']['type'], 'NotebookCancelled')
        path = Path(result['path']); self.assertTrue((path/'report-data.json').is_file())
        nb = nbformat.read(path/'executed.ipynb', as_version=4)
        self.assertIn('before-stop', json.dumps(nb.cells))
        self.assertTrue(any(c.cell_type=='code' and c.execution_count is None for c in nb.cells))
        with self.assertRaises(ProcessLookupError): os.kill(int(pid_file.read_text()), 0)

    def test_changed_selection_rejects_before_attempt_or_kernel(self):
        selection = self.bundle(); bad = copy.deepcopy(selection); bad['mode'] = 'result-report'
        with self.assertRaisesRegex(GraphError, 'changed after inspection'):
            execute_bundle(selection['path'], expected=bad)
        self.assertFalse((Path(selection['path'])/'executions').exists())

    def test_requested_bytes_changed_after_validation_reject_before_attempt(self):
        from app.notebooks import verify_bundle
        selection=self.bundle();path=Path(selection['path'])/'requested.ipynb'
        def change(directory):
            retained=verify_bundle(directory);path.write_bytes(path.read_bytes()+b' ');return retained
        with patch('app.notebooks.verify_bundle',side_effect=change):
            with self.assertRaisesRegex(GraphError,'changed during preparation'):
                execute_bundle(selection['path'],expected=selection)
        self.assertFalse((Path(selection['path'])/'executions').exists())

    def test_bad_timeout_and_interpreter_cannot_create_attempt(self):
        selection = self.bundle()
        for timeout in (True, 0, -1, 1.5):
            with self.assertRaisesRegex(GraphError, 'timeout'): run_isolated(selection, sys.executable, timeout=timeout)
        with self.assertRaisesRegex(GraphError, 'executable'): run_isolated(selection, '/no-such-python')
        self.assertFalse((Path(selection['path'])/'executions').exists())

    def test_host_without_torch_or_jupyter_can_inspect_and_execute(self):
        selection = self.bundle(mode='result-report'); request = self.root/'selection.json'
        request.write_text(json.dumps(selection))
        script = """import json,sys
from app.notebook_execution import inspect_isolated,run_isolated
s=json.load(open(sys.argv[1]));assert inspect_isolated(s['path'],sys.argv[2])['selection']==s
r=run_isolated(s,sys.argv[2],timeout=60)
print(json.dumps({'status':r['execution']['status'],'loaded':[x for x in ('torch','nbformat','nbclient','jupyter_client','PyQt6') if x in sys.modules]}))
"""
        child = subprocess.run(['/usr/bin/python3','-c',script,str(request),sys.executable],
                               capture_output=True,text=True,timeout=90)
        self.assertEqual(child.returncode,0,child.stderr)
        self.assertEqual(json.loads(child.stdout), {'status':'completed','loaded':[]})

    def test_retained_receipt_and_artifact_mutation_are_not_accepted_as_success(self):
        selection = self.bundle(); stop = threading.Event(); stop.set()
        result = execute_bundle(selection['path'], cancel=stop); path = Path(result['path'])
        wrong = copy.deepcopy(result); wrong['execution']['recipe']['revision'] = '0'*64
        with self.assertRaisesRegex(GraphError,'identity'): validate_result(wrong, selection)
        wrong = copy.deepcopy(result); wrong['path'] = str(self.root/'elsewhere')
        with self.assertRaisesRegex(GraphError,'path'): validate_result(wrong, selection)
        artifact = path/'executed.ipynb'; artifact.write_bytes(artifact.read_bytes()+b' ')
        with self.assertRaisesRegex(GraphError,'bytes changed'): validate_result(result, selection)

    def test_retained_receipt_symlink_and_malformed_scalar_diagnostics_are_rejected(self):
        selection=self.bundle();stop=threading.Event();stop.set()
        result=execute_bundle(selection['path'],cancel=stop);path=Path(result['path'])
        for field,value in [('elapsed_seconds',float('nan')),('elapsed_seconds',True),('error','no structure'),
                            ('executed_code_cells',True),('cell_timeout_seconds',0)]:
            bad=copy.deepcopy(result);bad['execution'][field]=value
            with self.subTest(field=field,value=value),self.assertRaises(GraphError):validate_result(bad,selection)
        raw=(path/'execution.json').read_bytes();other=self.root/'elsewhere.json';other.write_bytes(raw)
        (path/'execution.json').unlink();(path/'execution.json').symlink_to(other)
        with self.assertRaisesRegex(GraphError,'receipt path'):validate_result(result,selection)

    def test_independent_kernel_death_is_failure_not_a_stop_request(self):
        def die(payload,index):
            nb=build_notebook(payload,index)
            next(c for c in nb.cells if c.metadata.visual_pytorch.section=='question').source='import os; os._exit(17)'
            return nb
        selection=self.bundle(mode='result-report',build=die)
        result=run_isolated(selection,sys.executable,timeout=5)
        self.assertEqual(result['execution']['status'],'failed')
        self.assertEqual(result['execution']['error']['type'],'DeadKernelError')
        self.assertTrue((Path(result['path'])/'report-data.json').is_file())

    @unittest.skipUnless(sys.platform.startswith('linux'),'Owned Linux kernel/worker process-group probe')
    def test_stop_during_actual_rerun_fit_stops_kernel_and_numerical_child(self):
        e=copy.deepcopy(self.e);e.pop('revision')
        protocol=e['protocols'][0]['protocol'];protocol['budget']['updates']=100000
        protocol['budget']['seconds']=10.;protocol['evaluate_every']=1
        ref=self.w.put_spec('experiment',e);self.e=self.w.get_spec('experiment',ref['id'],ref['revision'])
        marker=self.root/'training-point.json';pids=self.root/'processes.json'
        prefix="""import os,json
from pathlib import Path
import app.transport as transport, app.report_runtime as report_runtime
real_worker=transport.run_json_worker
def monitored_worker(*args,**kwargs):
    original=kwargs.get('on_process')
    def launched(process):
        Path(PIDS).write_text(json.dumps({'kernel':os.getpid(),'worker':process.pid}))
        if original: original(process)
    kwargs['on_process']=launched
    return real_worker(*args,**kwargs)
transport.run_json_worker=monitored_worker
real_candidate=report_runtime.execute_candidate
def monitored_candidate(*args,**kwargs):
    kwargs['progress']=lambda point:Path(MARKER).write_text(json.dumps(point))
    return real_candidate(*args,**kwargs)
report_runtime.execute_candidate=monitored_candidate
""".replace('PIDS',repr(str(pids))).replace('MARKER',repr(str(marker)))
        def instrument(payload,index):
            nb=build_notebook(payload,index)
            cell=next(c for c in nb.cells if c.metadata.visual_pytorch.section=='resolution')
            cell.source=prefix+cell.source;return nb
        selection=self.bundle(build=instrument);stop=threading.Event();observed=[]
        def when_training():
            deadline=time.monotonic()+25
            while time.monotonic()<deadline:
                try:
                    point=json.loads(marker.read_text())
                    if point.get('update',0)>=1:
                        observed.append(point);break
                except (OSError,ValueError):pass
                time.sleep(.025)
            stop.set()
        waiter=threading.Thread(target=when_training);waiter.start()
        try:result=run_isolated(selection,sys.executable,timeout=90,cancel=stop)
        finally:waiter.join(30)
        self.assertFalse(waiter.is_alive());self.assertTrue(observed,'Never observed an actual gradient update')
        self.assertEqual(result['execution']['status'],'cancelled',result)
        owners=json.loads(pids.read_text())
        def live(pid):
            stat=Path('/proc')/str(pid)/'stat'
            try:return stat.read_text().split(') ')[1].split()[0] not in ('Z','X')
            except FileNotFoundError:return False
        deadline=time.monotonic()+3
        while any(live(pid) for pid in owners.values()) and time.monotonic()<deadline:time.sleep(.025)
        self.assertFalse(any(live(pid) for pid in owners.values()),owners)
        self.assertFalse((Path(result['path'])/'report-data.json').exists(),'Interrupted resolution must not claim a complete report')


if __name__ == '__main__': unittest.main()
