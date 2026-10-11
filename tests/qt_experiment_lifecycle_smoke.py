"""Native Qt lifecycle tests, including a real blocked GUI delivery interval.

Run using the GUI Python; no WebEngine, numerical host imports or user state.
"""
import copy
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from PyQt6.QtCore import QCoreApplication, QObject, QEventLoop, QTimer
from app.config import Config
from app.core import Workspace
from app.desktop import Window, find_torch_python
from app.desktop_experiments import Delivery, complete, dispatch, recovery_rows
from app.experiment_ui import capture
from app.experiment_objects import execute_candidate, object_reference, preview_experiment
from app.job_state import job_busy
from examples.experiment_workspace import populate

app=QCoreApplication([])


class Harness(QObject):
    state=Window.state; describe=Window.describe; dispatch=Window.dispatch
    can_replace=Window.can_replace
    def __init__(self, folder):
        super().__init__();self.workspace=Workspace();self.workspace.set_constrained_mode(False)
        self.experiment=populate(self.workspace);self.experiment.pop('revision')
        self.experiment['protocols'][0]['protocol']['budget']['updates']=3
        self.experiment['protocols'][0]['protocol']['evaluate_every']=2
        ref=self.workspace.put_spec('experiment',self.experiment)
        self.experiment=self.workspace.get_spec('experiment',ref['id'],ref['revision'])
        self.config=Config(folder/'config.json');self.job=None;self.events=[]
        self.torch_python='/definitely/missing/python';self.rendering={}
    def recent_files(self): return []
    def broadcast(self, kind='state', payload=None, **kw):
        self.events.append((kind, self.state() if kind=='state' else payload))


def pump_until(predicate, seconds=20):
    deadline=time.monotonic()+seconds
    while time.monotonic()<deadline:
        if predicate(): return
        loop=QEventLoop();QTimer.singleShot(10,loop.quit);loop.exec()
    raise AssertionError('Qt condition timeout')


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        self.directory=tempfile.TemporaryDirectory();self.folder=Path(self.directory.name)
        self.h=Harness(self.folder);self.workspaces=[self.h.workspace]
        self.ref=object_reference('experiment',self.h.experiment)
        self.coords=preview_experiment(self.h.experiment,self.h.workspace.get_spec)['candidates'][0]['coordinates']
        self.snapshot=capture(self.h.workspace,self.ref,self.coords,'trial','Trial')
    def tearDown(self):
        if self.h.job and isinstance(self.h.job,QObject):
            self.h.job.stop();self.h.job.wait(20000)
            QCoreApplication.processEvents()
        for w in self.workspaces:
            if w.history.directory:w.history.directory.cleanup()
            w.db.close()
        self.h.deleteLater();QCoreApplication.processEvents();self.directory.cleanup()
    def fake(self):
        job=SimpleNamespace(snapshot=copy.deepcopy(self.snapshot),awaiting_terminal=True,
            terminal_delivered=False,isRunning=lambda:False,stop=lambda:None,mid=None)
        self.h.job=job;return job
    def result(self):
        w=Workspace();self.workspaces.append(w)
        for row in self.snapshot['documents']:w.put_spec(row['kind'],row['document'],select=False)
        return execute_candidate(w,self.ref,self.coords,'trial','Trial',python='/missing/python',
                                 saved_manifest=self.snapshot['manifest'])
    def train(self, **kw):
        return self.h.dispatch({'action':'scienceTrain','reference':self.ref,
                               'coordinates':self.coords,'run_id':'trial','name':'Trial',**kw},None)
    def test_pending_delivery_blocks_edits_policy_undo_and_both_training_paths(self):
        job=self.fake();before=self.h.workspace.db.serialize()
        requests=[{'action':'scienceFreeze','id':self.ref['id']}, {'action':'scienceTrain'},
                  {'action':'undo'}, {'action':'setConstrainedMode','enabled':True},
                  {'action':'newModel'}, {'action':'updateModel'}, {'action':'generateData'},
                  {'action':'deleteModel'}, {'action':'train'}, {'action':'setLayoutPreferences'}]
        for req in requests:
            with self.subTest(action=req['action']),self.assertRaisesRegex(ValueError,'Stop'):
                self.h.dispatch(req,None)
            self.assertEqual(self.h.workspace.db.serialize(),before)
        self.assertEqual(self.h.state()['experiment_job'],'trial')
        self.assertTrue(job_busy(job))
        self.assertTrue(self.h.dispatch({'action':'scienceIndex'},None))
    def test_pending_delivery_blocks_workspace_replacement_and_close_without_wait(self):
        job=self.fake()
        with patch('app.desktop.QMessageBox.information'):
            self.assertFalse(self.h.can_replace())
        event=SimpleNamespace(ignore=lambda:setattr(job,'close_ignored',True))
        Window.closeEvent(self.h,event)
        self.assertTrue(job.close_ignored);self.assertTrue(job.awaiting_terminal)
    def test_admission_runs_once_and_retains_original_revision_after_head_change(self):
        job=self.fake();result=self.result()
        changed=copy.deepcopy(self.h.experiment);changed.pop('revision');changed['initialization_seeds']=[99]
        self.h.workspace.put_spec('experiment',changed)
        complete(self.h,self.h.workspace,job,json.dumps(result));before=self.h.workspace.db.serialize()
        complete(self.h,self.h.workspace,job,json.dumps(result))
        self.assertEqual(self.h.workspace.db.serialize(),before)
        self.assertFalse(job.awaiting_terminal);self.assertTrue(job.terminal_delivered)
        record=self.h.workspace.get_spec('run','trial')
        self.assertEqual(record['selection']['experiment'],self.ref)
        self.assertEqual(sum(k=='scienceComplete' for k,v in self.h.events),1)
    def test_wrong_reference_cannot_insert_even_valid_record(self):
        job=self.fake();result=self.result();result['reference']['revision']='0'*64
        before=self.h.workspace.db.serialize()
        complete(self.h,self.h.workspace,job,json.dumps(result))
        self.assertEqual(self.h.workspace.db.serialize(),before)
        self.assertEqual(len(recovery_rows(self.h)),1)
        self.assertFalse(job.awaiting_terminal)
    def test_workspace_change_never_admits_to_either_workspace(self):
        job=self.fake();original=self.h.workspace;result=self.result()
        self.h.workspace=Workspace();self.workspaces.append(self.h.workspace)
        old=original.db.serialize();new=self.h.workspace.db.serialize()
        complete(self.h,original,job,json.dumps(result))
        self.assertEqual(original.db.serialize(),old);self.assertEqual(self.h.workspace.db.serialize(),new)
        value=next(iter(self.h.experiment_recoveries.values()))
        self.assertEqual(value['result'],result);self.assertEqual(value['snapshot'],self.snapshot)
    def test_superseded_job_cannot_replace_active_job_or_admit_late_results(self):
        old=self.fake();result=self.result();current=self.fake()
        before=self.h.workspace.db.serialize()
        complete(self.h,self.h.workspace,old,json.dumps(result))
        self.assertIs(self.h.job,current);self.assertTrue(current.awaiting_terminal)
        self.assertEqual(self.h.workspace.db.serialize(),before)
        receiver=Delivery(self.h,self.h.workspace,old);before=len(self.h.events)
        receiver.point(json.dumps({'run_id':'trial','point':{'update':1}}))
        self.assertEqual(len(self.h.events),before)
    def test_admission_failure_survives_later_job_and_export_no_overwrite(self):
        job=self.fake();result=self.result()
        with patch.object(self.h.workspace,'put_spec',side_effect=ValueError('Forced admission failure')):
            complete(self.h,self.h.workspace,job,json.dumps(result))
        sid=recovery_rows(self.h)[0]['id'];self.h.job=None
        shown=dispatch(self.h,{'action':'scienceRecoveryShow','id':sid})
        shown['snapshot']['run_id']='altered'
        self.assertEqual(dispatch(self.h,{'action':'scienceRecoveryShow','id':sid})['snapshot'],self.snapshot)
        output=self.folder/'recovery.json'
        dispatch(self.h,{'action':'scienceExportRecovery','id':sid,'output':str(output)})
        evidence=json.loads(output.read_text());self.assertEqual(evidence['result'],result)
        self.assertEqual(evidence['snapshot'],self.snapshot)
        before=output.read_bytes()
        with self.assertRaises(FileExistsError):dispatch(self.h,{'action':'scienceExportRecovery','id':sid,'output':str(output)})
        self.assertEqual(output.read_bytes(),before)
        self.assertEqual(recovery_rows(self.h)[0]['exported_to'],str(output))
    def test_failed_delivery_and_missing_terminal_keep_snapshot_without_fabricated_run(self):
        for failure in ('error','missing'):
            job=self.fake();delivery=Delivery(self.h,self.h.workspace,job)
            if failure=='error':delivery.failed('Forced pre-execution error')
            else:delivery.finished()
            self.assertFalse(job.awaiting_terminal);self.assertTrue(job.terminal_delivered)
            self.assertEqual(self.h.workspace.list_specs('run'),[])
        self.assertEqual(len(recovery_rows(self.h)),2)
        self.assertTrue(all(r['result'] is None for r in self.h.experiment_recoveries.values()))
    def test_close_cannot_silently_discard_unexported_recovery(self):
        from PyQt6.QtWidgets import QMessageBox
        job=self.fake();Delivery(self.h,self.h.workspace,job).failed('admission unavailable')
        event=SimpleNamespace(ignore=lambda:setattr(job,'close_ignored',True))
        with patch('app.desktop.QMessageBox.warning',return_value=QMessageBox.StandardButton.Cancel):
            Window.closeEvent(self.h,event)
        self.assertTrue(job.close_ignored);self.assertEqual(len(recovery_rows(self.h)),1)
    def test_real_thread_stops_before_queued_admission_but_busy_stays_true(self):
        self.train();job=self.h.job
        # Intentionally don't run the GUI loop: thread finishes, delivery is queued.
        self.assertTrue(job.wait(20000));self.assertFalse(job.isRunning())
        self.assertTrue(job_busy(job));self.assertEqual(self.h.workspace.list_specs('run'),[])
        with self.assertRaisesRegex(ValueError,'Stop'):self.h.dispatch({'action':'undo'},None)
        pump_until(lambda:job.terminal_delivered)
        self.assertFalse(job_busy(job))
        self.assertEqual(self.h.workspace.get_spec('run','trial')['attempts'][-1]['stage'],'worker')
        self.assertEqual(recovery_rows(self.h),[])
        self.assertNotIn('torch',sys.modules)
    def test_real_worker_cancellation_retains_partial_counts_and_never_evaluates_test(self):
        python=find_torch_python();self.assertIsNotNone(python)
        e=copy.deepcopy(self.h.experiment);e.pop('revision')
        e['protocols'][0]['protocol']['budget']['updates']=100000
        e['protocols'][0]['protocol']['evaluate_every']=1
        self.ref=self.h.workspace.put_spec('experiment',e);self.h.torch_python=python
        self.train();job=self.h.job
        def stop_after_point(payload):job.stop()
        job.point.connect(stop_after_point)
        pump_until(lambda:job.terminal_delivered,30);self.assertTrue(job.wait(10000))
        record=self.h.workspace.get_spec('run','trial');result=record['attempts'][-1]['result']
        self.assertIsNotNone(result,record['attempts'][-1]['error'])
        self.assertEqual(result['status'],'cancelled');self.assertEqual(result['stop_reason'],'cancelled')
        self.assertGreaterEqual(result['counts']['completed_updates'],1)
        self.assertLess(result['counts']['completed_updates'],100000)
        self.assertNotIn('test',result['final_metrics'])
        self.assertEqual(result['history'][-1]['update'],result['counts']['completed_updates'])
        self.assertNotIn('torch',sys.modules)

    def test_actual_state_and_recovery_events_are_bounded_without_discarding_full_evidence(self):
        from app.recovery_views import header
        job=self.fake();job.snapshot['documents'].append({'hidden':'λ😀'*100000})
        Delivery(self.h,self.h.workspace,job).failed('λ😀'*10000)
        state=self.h.state()['experiment_recoveries']
        self.assertEqual(state['scope'],'experiment-recovery-header')
        self.assertEqual(state['total'],1);self.assertLess(len(json.dumps(state)),400)
        notice=[v for k,v in self.h.events if k=='scienceRecovery'][-1]
        self.assertEqual(notice['scope'],'experiment-recovery-notification')
        self.assertLess(len(json.dumps(notice)),4000);self.assertNotIn('snapshot',notice)
        sid=notice['recovery']['id']
        self.assertEqual(self.h.experiment_recoveries[sid]['snapshot'],job.snapshot)
        request={'action':'scienceRecoveryPage','scope_id':state['scope_id'],'sequence':state['sequence']}
        page=dispatch(self.h,request);self.assertEqual(page['rows'][0]['id'],sid)
        old=self.h.workspace;self.h.workspace=Workspace();self.workspaces.append(self.h.workspace)
        self.assertEqual(dispatch(self.h,request),page)
        for changes in ({'extra':True},{'sequence':True},{'scope_id':'foreign'},{'limit':26}):
            with self.subTest(changes=changes),self.assertRaises(ValueError):
                dispatch(self.h,{**request,**changes})
        self.assertEqual(header(self.h),state);self.assertIsNot(self.h.workspace,old)

    def test_guarded_export_rechecks_sequence_after_modal_and_failed_export_preserves_scope(self):
        from app.recovery_views import header,changed
        job=self.fake();Delivery(self.h,self.h.workspace,job).failed('retained')
        h=header(self.h);sid=recovery_rows(self.h)[0]['id']
        output=self.folder/'guarded.json'
        def choose(*args):
            changed(self.h);return str(output),''
        with patch('app.desktop_experiments.QFileDialog.getSaveFileName',side_effect=choose):
            with self.assertRaisesRegex(ValueError,'Refresh'):
                dispatch(self.h,{'action':'scienceExportRecovery','id':sid,'scope_id':h['scope_id'],'sequence':h['sequence']})
        self.assertFalse(output.exists());self.assertIsNone(recovery_rows(self.h)[0]['exported_to'])
        h=header(self.h)
        request={'action':'scienceExportRecovery','id':sid,'scope_id':h['scope_id'],'sequence':h['sequence'],'output':str(output)}
        dispatch(self.h,request);self.assertEqual(header(self.h)['sequence'],h['sequence']+1)
        before=output.read_bytes();current=header(self.h)
        with self.assertRaises(FileExistsError):
            dispatch(self.h,{**request,'sequence':current['sequence']})
        self.assertEqual(output.read_bytes(),before);self.assertEqual(header(self.h),current)

    def test_terminal_notice_has_exact_identity_and_duplicate_delivery_keeps_one_notice(self):
        job=self.fake();result=self.result()
        complete(self.h,self.h.workspace,job,json.dumps(result));complete(self.h,self.h.workspace,job,json.dumps(result))
        notices=[v for k,v in self.h.events if k=='scienceComplete'];self.assertEqual(len(notices),1)
        notice=notices[0]
        self.assertEqual(notice['reference'],result['reference'])
        self.assertEqual(notice['attempt'],{k:result['record']['attempts'][-1][k] for k in ('number','status','stage')})
        self.assertNotIn('record',notice);self.assertEqual(notice['workspace_id'],self.h.state()['workspace_id'])


if __name__=='__main__':unittest.main(verbosity=2)
