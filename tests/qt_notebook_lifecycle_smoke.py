"""Native pending-delivery/origin/exact-artifact tests; no GUI-host Jupyter/Torch."""
import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PyQt6.QtCore import QCoreApplication,QObject,QEventLoop,QTimer
from app.authoring import workspace_id
from app.config import Config
from app.core import Workspace
from app.desktop import Window,find_torch_python
from app.desktop_notebooks import Delivery
from app.job_state import job_busy

app=QCoreApplication([])
class Harness(QObject):
    state=Window.state;describe=Window.describe;dispatch=Window.dispatch;can_replace=Window.can_replace
    def __init__(self,folder):
        super().__init__();self.workspace=Workspace.load(folder/'workspace.vtw');self.config=Config(folder/'config.json')
        self.job=None;self.events=[];self.rendering={};self.torch_python=find_torch_python()
    def recent_files(self):return []
    def broadcast(self,kind='state',payload=None,**kw):self.events.append((kind,self.state() if kind=='state' else payload))


class NotebookLifecycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();cls.folder=Path(cls.temp.name)
        python=find_torch_python();assert python
        p=subprocess.run([python,'tests/notebook_fixture.py',str(cls.folder),'--receipt'],capture_output=True,text=True,timeout=30)
        assert p.returncode==0,p.stderr
        cls.fixture=json.loads((cls.folder/'fixture.json').read_text())
    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()
    def setUp(self):
        self.h=Harness(self.folder);self.workspaces=[self.h.workspace]
        self.selection=copy.deepcopy(self.fixture['selections']['saved'])
        self.snapshot={'id':'j'*32,'workspace_id':workspace_id(self.h.workspace),'selection':self.selection,
                       'python':str(Path(self.h.torch_python).resolve()),'timeout':600}
    def tearDown(self):
        if isinstance(self.h.job,QObject):
            self.h.job.stop();self.h.job.wait(20000);QCoreApplication.processEvents()
        for w in self.workspaces:w.db.close()
        self.h.deleteLater();QCoreApplication.processEvents()
    def fake(self):
        self.h.job=SimpleNamespace(snapshot=copy.deepcopy(self.snapshot),mid=None,awaiting_terminal=True,
            terminal_delivered=False,isRunning=lambda:False,stop=lambda:None)
        return self.h.job
    def req(self,**kw):
        return {'action':'scienceExecuteNotebook','workspace_id':workspace_id(self.h.workspace),'selection':self.selection,
                'python':self.h.torch_python,'timeout':600,**kw}
    def test_pending_notebook_blocks_edits_runs_replacement_and_close_without_blocking(self):
        job=self.fake();before=self.h.workspace.db.serialize()
        for req in [{'action':'newModel'},{'action':'undo'},{'action':'scienceTrain'},self.req(),{'action':'scienceNotebookInspect'}]:
            with self.assertRaisesRegex(ValueError,'Stop'):self.h.dispatch(req,None)
        self.assertEqual(self.h.workspace.db.serialize(),before)
        self.assertEqual(self.h.state()['notebook_job']['id'],self.snapshot['id']);self.assertIsNone(self.h.state()['experiment_job'])
        with patch('app.desktop.QMessageBox.information'):self.assertFalse(self.h.can_replace())
        event=SimpleNamespace(ignore=lambda:setattr(job,'ignored',True))
        Window.closeEvent(self.h,event);self.assertTrue(job.ignored);self.assertTrue(job_busy(job))
        self.assertEqual(self.h.dispatch({'action':'scienceNotebookHistory'},None),[])
    def test_exact_terminal_is_delivered_once_without_sql_or_domain_history_changes(self):
        job=self.fake();delivery=Delivery(self.h,self.h.workspace,job);before=self.h.workspace.db.serialize()
        payload=json.dumps(self.fixture['terminal']);delivery.completed(payload);delivery.completed(payload)
        self.assertEqual(self.h.workspace.db.serialize(),before);self.assertFalse(job.awaiting_terminal)
        self.assertEqual(len(self.h.notebook_executions),1)
        row=self.h.dispatch({'action':'scienceNotebookEvidence','id':job.snapshot['id']},None)
        self.assertEqual(row['status'],'cancelled');self.assertEqual(row['result'],self.fixture['terminal'])
        row['selection']['path']='other';self.assertEqual(self.h.notebook_executions[job.snapshot['id']]['selection'],self.selection)
        self.assertEqual(sum(k=='scienceNotebookComplete' for k,v in self.h.events),1)
    def test_origin_replacement_keeps_artifacts_but_cannot_insert_into_new_workspace(self):
        origin=self.h.workspace;job=self.fake();delivery=Delivery(self.h,origin,job)
        self.h.workspace=Workspace();self.workspaces.append(self.h.workspace);before=self.h.workspace.db.serialize()
        delivery.point(json.dumps({'stage':'cell-running'}));self.assertFalse(any(k=='scienceNotebookProgress' for k,v in self.h.events))
        delivery.completed(json.dumps(self.fixture['terminal']))
        self.assertEqual(self.h.workspace.db.serialize(),before)
        row=self.h.notebook_executions[job.snapshot['id']];self.assertEqual(row['workspace_id'],workspace_id(origin))
        self.assertNotEqual(row['workspace_id'],workspace_id(self.h.workspace))
    def test_wrong_terminal_identity_is_not_a_completed_or_cancelled_receipt(self):
        job=self.fake();delivery=Delivery(self.h,self.h.workspace,job);wrong=copy.deepcopy(self.fixture['terminal'])
        wrong['execution']['input_index_hash']='0'*64;before=self.h.workspace.db.serialize()
        delivery.completed(json.dumps(wrong));row=self.h.notebook_executions[job.snapshot['id']]
        self.assertEqual(row['status'],'runner-error-no-verified-receipt');self.assertIsNone(row['result'])
        self.assertIn('identity',row['error']);self.assertEqual(self.h.workspace.db.serialize(),before)
    def test_cancel_confirmation_and_foreign_origin_do_not_start_threads(self):
        with patch('app.desktop_notebooks.confirm_execution',return_value=False),patch('app.desktop_notebooks.NotebookJob') as launch:
            self.assertIsNone(self.h.dispatch(self.req(),None));launch.assert_not_called()
        with self.assertRaisesRegex(ValueError,'another workspace'):self.h.dispatch(self.req(workspace_id='foreign'),None)
        self.assertIsNone(self.h.job)
    def test_modal_origin_and_new_busy_job_rechecked_before_launch(self):
        before=self.h.workspace.db.serialize()
        def replace(*args):
            self.h.workspace=Workspace();self.workspaces.append(self.h.workspace);return True
        with patch('app.desktop_notebooks.confirm_execution',side_effect=replace),patch('app.desktop_notebooks.NotebookJob') as launch:
            with self.assertRaisesRegex(ValueError,'changed during'):self.h.dispatch(self.req(),None)
            launch.assert_not_called()
        self.h.workspace=self.workspaces[0];self.assertEqual(self.h.workspace.db.serialize(),before)
        def busy(*args):self.fake();return True
        with patch('app.desktop_notebooks.confirm_execution',side_effect=busy),patch('app.desktop_notebooks.NotebookJob') as launch:
            with self.assertRaisesRegex(ValueError,'changed during'):self.h.dispatch(self.req(),None)
            launch.assert_not_called()
    def test_thread_stop_is_not_terminal_delivery_and_native_edits_stay_blocked(self):
        with patch('app.desktop_notebooks.confirm_execution',return_value=True),patch('app.desktop_notebooks.run_isolated',return_value=self.fixture['terminal']):
            result=self.h.dispatch(self.req(),None);job=self.h.job
            self.assertTrue(job.wait(20000));self.assertFalse(job.isRunning());self.assertTrue(job_busy(job))
            with self.assertRaisesRegex(ValueError,'Stop'):self.h.dispatch({'action':'newModel'},None)
            self.assertEqual(result['status'],'started-not-completed')
            QCoreApplication.processEvents()
            self.assertFalse(job_busy(job));self.assertEqual(len(self.h.notebook_executions),1)

    def test_actual_async_inspection_has_no_kernel_or_receipt_and_pending_delivery_fence(self):
        before=self.h.workspace.db.serialize();directory=self.selection['path']
        attempts=sorted((Path(directory)/'executions').iterdir())
        result=self.h.dispatch({'action':'scienceNotebookInspect','directory':directory,'python':self.h.torch_python,
            'workspace_id':workspace_id(self.h.workspace),'request_id':'ticket'},None)
        job=self.h.job;self.assertEqual(result['operation'],'inspect')
        self.assertTrue(job.wait(20000));self.assertTrue(job_busy(job))
        with self.assertRaisesRegex(ValueError,'Stop'):self.h.dispatch({'action':'newModel'},None)
        QCoreApplication.processEvents();self.assertFalse(job_busy(job))
        row=next(v for k,v in self.h.events if k=='scienceNotebookInspected')
        self.assertEqual(row['request_id'],'ticket');self.assertEqual(row['status'],'inspected-not-executed')
        self.assertEqual(row['inspection']['selection'],self.selection)
        self.assertFalse(getattr(self.h,'notebook_executions',{}))
        self.assertEqual(sorted((Path(directory)/'executions').iterdir()),attempts)
        self.assertEqual(self.h.workspace.db.serialize(),before)
        self.assertFalse(any(x in sys.modules for x in ('torch','nbformat','nbclient','jupyter_client')))

    def test_inspection_replacement_with_same_uuid_cannot_select_old_origin(self):
        snapshot={**self.snapshot,'operation':'inspect','timeout':None,'selection':None,
                  'directory':self.selection['path'],'request_id':'old'}
        job=SimpleNamespace(snapshot=snapshot,mid=None,awaiting_terminal=True,terminal_delivered=False,
                            cancel=SimpleNamespace(is_set=lambda:False),isRunning=lambda:False)
        self.h.job=job;origin=self.h.workspace;delivery=Delivery(self.h,origin,job)
        # Load a distinct connection carrying the same persisted UUID.
        self.h.workspace=Workspace.load(self.folder/'workspace.vtw');self.workspaces.append(self.h.workspace)
        self.assertEqual(workspace_id(origin),workspace_id(self.h.workspace))
        delivery.completed(json.dumps({'selection':self.selection,'python':self.snapshot['python']}))
        row=next(v for k,v in self.h.events if k=='scienceNotebookInspected')
        self.assertIsNone(row['inspection']);self.assertIn('Origin changed',row['error'])
        self.assertFalse(job.awaiting_terminal)

    def test_inspection_chooser_cancel_and_modal_busy_recheck_do_not_start_threads(self):
        req={'action':'scienceNotebookInspect','workspace_id':workspace_id(self.h.workspace),'python':self.h.torch_python}
        with patch('app.desktop_notebooks.QFileDialog.getExistingDirectory',return_value=''),patch('app.desktop_notebooks.NotebookJob') as launch:
            self.assertIsNone(self.h.dispatch(req,None));launch.assert_not_called()
        def busy(*args):self.fake();return self.selection['path']
        with patch('app.desktop_notebooks.QFileDialog.getExistingDirectory',side_effect=busy),patch('app.desktop_notebooks.NotebookJob') as launch:
            with self.assertRaisesRegex(ValueError,'changed during'):self.h.dispatch(req,None)
            launch.assert_not_called()


if __name__=='__main__':
    unittest.main()
