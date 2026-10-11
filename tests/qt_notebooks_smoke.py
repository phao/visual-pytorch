"""Actual sandboxed GUI notebook generation/inspection/execution/Stop/artifacts."""
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PyQt6.QtCore import QObject,QEventLoop,QTimer,QUrl,QCoreApplication,QEvent
from PyQt6.QtWidgets import QApplication,QMessageBox,QWidget
from PyQt6.QtWebChannel import QWebChannel
from PyQt6.QtWebEngineCore import QWebEnginePage,QWebEngineProfile
from PyQt6.QtWebEngineWidgets import QWebEngineView
from app.authoring import workspace_id
from app.config import Config
from app.core import Workspace
from app.desktop import Bridge,Window,find_torch_python
from app.desktop_notebooks import confirm_execution
from app.notebook_execution import validate_result
from app.notebook_execution import inspect_isolated
from app.revisions import model_document

assert not os.environ.get('QTWEBENGINE_DISABLE_SANDBOX')
assert '--no-sandbox' not in os.environ.get('QTWEBENGINE_CHROMIUM_FLAGS','')
app=QApplication(['notebook-ui-smoke']);view=QWebEngineView();view.resize(1500,1100)
profile=QWebEngineProfile(view);errors=[]
class Page(QWebEnginePage):
    def javaScriptConsoleMessage(self,level,message,line,source):
        if level==self.JavaScriptConsoleMessageLevel.ErrorMessageLevel:errors.append(f'{source}:{line}: {message}')
class Harness(QObject):
    state=Window.state;describe=Window.describe;dispatch=Window.dispatch
    def __init__(self,folder):
        super().__init__();self.workspace=Workspace.load(folder/'workspace.vtw');self.config=Config(folder/'config.json')
        self.job=None;self.torch_python=find_torch_python();self.rendering={};self.bridge=Bridge(self,view);self.events=[]
    def recent_files(self):return []
    def broadcast(self,kind='state',payload=None,**kw):
        self.events.append((kind,payload));self.bridge.send(kind,self.state() if kind=='state' else payload)
def js(script):
    loop=QEventLoop();timer=QTimer();timer.setSingleShot(True);timer.timeout.connect(loop.quit);values=[]
    view.page().runJavaScript(script,lambda value:(values.append(value),loop.quit()));timer.start(3000);loop.exec();timer.stop()
    assert values,'JavaScript timeout';return values[0]
def until(predicate,timeout=40):
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        if (predicate() if callable(predicate) else js(predicate)):return
        loop=QEventLoop();QTimer.singleShot(20,loop.quit);loop.exec()
    raise AssertionError(f'UI timeout: {predicate}; errors: {errors[:5]}')
def click(sid):
    until('!document.getElementById('+json.dumps(sid)+').disabled')
    js('document.getElementById('+json.dumps(sid)+').click()')
def text(sid):return js('document.getElementById('+json.dumps(sid)+').textContent')
def inspect(folder,name):
    with patch('app.desktop_notebooks.QFileDialog.getExistingDirectory',return_value=str(folder/name)) as chooser:
        click('science-notebook-inspect')
        until('!document.getElementById("science-notebook-execute").disabled && document.getElementById("science-notebook-selection").textContent.includes('+json.dumps(str(folder/name))+')')
        assert chooser.call_args.args[1]=='Inspect notebook bundle directory'
def run(h,status):
    mark=len(getattr(h,'notebook_executions',{}))
    with patch('app.desktop_notebooks.confirm_execution',return_value=True) as confirm:
        click('science-notebook-execute');until(lambda:len(getattr(h,'notebook_executions',{}))==mark+1,80)
        assert confirm.call_count==1
    until('!window.loom.state.notebook_job && !document.getElementById("science-notebook-execute").disabled')
    until('document.getElementById("science-notebook-evidence").textContent.includes('+json.dumps('Notebook execution · '+status)+')')
    row=list(h.notebook_executions.values())[-1];assert row['status']==status,row
    validate_result(row['result'],row['selection']);return row

with tempfile.TemporaryDirectory() as directory:
    folder=Path(directory);python=find_torch_python();assert python
    producer=subprocess.run([python,'tests/notebook_fixture.py',str(folder)],capture_output=True,text=True,timeout=40)
    assert producer.returncode==0,producer.stderr
    fixture=json.loads((folder/'fixture.json').read_text());h=Harness(folder);workspaces=[h.workspace]
    page=Page(profile,view);view.setPage(page);channel=QWebChannel(page);channel.registerObject('host',h.bridge);page.setWebChannel(channel)
    url=QUrl.fromLocalFile(str(Path(__file__).resolve().parents[1]/'app/web/index.html'))
    mid=next(iter(h.workspace.models()));frozen=model_document(h.workspace.models(),mid)
    url.setFragment('model='+mid);view.setUrl(url);view.show()
    try:
        until('Boolean(window.loom?.document.view.fitted)')
        until('Boolean(window.loom?.document?.view.fitted)')  # Camera is session-only.
        assert model_document(h.workspace.models(),mid)==frozen
        before=h.workspace.db.serialize();source=(folder/'workspace.vtw').read_bytes()
        js('document.querySelector("[data-panel=science]").click();Object.assign(document.getElementById("experiment-dock").style,{height:"880px",width:"960px",top:"110px",left:"330px",bottom:"auto"});document.getElementById("science-notebook").open=true;')
        until('document.getElementById("science-notebook-python").value.length>0')
        assert js('document.getElementById("science-notebook-timeout").value')=='600'
        assert js('document.getElementById("science-notebook-execute").disabled')
        inspect(folder,'saved');assert 'no training' in text('science-notebook-selection')
        assert not (folder/'saved/executions').exists();assert h.workspace.db.serialize()==before
        inspected_job=h.job
        with patch('app.desktop_notebooks.confirm_execution',return_value=False):
            click('science-notebook-execute');until('!document.getElementById("science-notebook-execute").disabled')
        assert h.job is inspected_job and not h.job.awaiting_terminal and not h.job.isRunning()
        assert h.job.snapshot['operation']=='inspect' and not (folder/'saved/executions').exists()
        row=run(h,'completed');attempt=Path(row['result']['path'])
        data=json.loads((attempt/'report-data.json').read_text());assert not data['notebook_execution']['training_requested']
        assert len(data['runs'])==1 and data['runs'][0]['status']=='completed'
        assert any(p['stage']=='cell-running' for k,payload in h.events if k=='scienceNotebookProgress' for p in [payload['progress']])
        assert h.workspace.db.serialize()==before and (folder/'workspace.vtw').read_bytes()==source
        assert js('document.querySelectorAll("#science-notebook-evidence img,#science-notebook-evidence svg,#science-notebook-evidence script").length')==0
        for theme,font,spacing in [('dark',14,1),('github',18,4),('math',28,10)]:
            h.config.values.update(theme=theme,font_size=font,spacing=spacing);h.broadcast()
            until('getComputedStyle(document.body).getPropertyValue("--ui-font-size").trim()==='+json.dumps(str(font)+'px'))
            assert js('parseFloat(getComputedStyle(document.getElementById("science-notebook-evidence")).fontSize)')>=font-2
            assert js('document.getElementById("science-notebook-evidence").getBoundingClientRect().width<=document.getElementById("science-panel").getBoundingClientRect().width')
        h.config.values.update(theme='github',font_size=18,spacing=4);h.broadcast()
        if os.environ.get('VTW_NOTEBOOK_SCREENSHOT'):
            js('document.getElementById("science-notebook-evidence").scrollIntoView({block:"start"});document.getElementById("toast-close").click();')
            loop=QEventLoop();QTimer.singleShot(300,loop.quit);loop.exec();assert view.grab().save(os.environ['VTW_NOTEBOOK_SCREENSHOT'])
        inspect(folder,'rerun');assert 'fresh CPU fits' in text('science-notebook-selection')
        assert 'trains' in text('science-notebook-execute')
        rerun=run(h,'completed');viewdata=json.loads((Path(rerun['result']['path'])/'report-data.json').read_text())
        assert viewdata['notebook_execution']['training_requested'] and len(viewdata['runs'])==8
        assert all(r['status']=='completed' for r in viewdata['runs'])
        assert h.workspace.db.serialize()==before and (folder/'workspace.vtw').read_bytes()==source
        inspect(folder,'failure');failed=run(h,'failed');attempt=Path(failed['result']['path'])
        assert failed['result']['execution']['error']['type']=='CellExecutionError'
        assert (attempt/'report-data.json').is_file()
        executed=json.loads((attempt/'executed.ipynb').read_text());assert any(c['cell_type']=='code' and c['execution_count'] is None for c in executed['cells'])
        assert '<img>' in text('science-notebook-evidence') and not js('Boolean(window.notebookInjected)')
        inspect(folder,'slow');js('document.getElementById("science-notebook-timeout").value="1"')
        timed=run(h,'failed');assert timed['result']['execution']['error']['type']=='CellTimeoutError'
        # Reset the private PID marker left by the timed-out kernel only.
        pid=folder/'kernel.pid';pid.unlink()
        js('document.getElementById("science-notebook-timeout").value="90"')
        mark=len(h.notebook_executions)
        with patch('app.desktop_notebooks.confirm_execution',return_value=True):
            click('science-notebook-execute');until(lambda:pid.exists())
            until('Boolean(window.loom.state.notebook_job) && !document.getElementById("science-notebook-stop").hidden')
            assert js('document.getElementById("science-freeze").disabled')
            for req in [{'action':'newModel'},{'action':'undo'},{'action':'scienceTrain'},{'action':'scienceExecuteNotebook'}]:
                try:h.dispatch(req,None)
                except ValueError as error:assert 'Stop' in str(error)
                else:raise AssertionError('Edit/run admitted during actual notebook execution')
            click('science-notebook-stop');until(lambda:len(h.notebook_executions)==mark+1)
        until('!window.loom.state.notebook_job && document.getElementById("science-notebook-evidence").textContent.includes("Notebook execution · cancelled")')
        stopped=list(h.notebook_executions.values())[-1];assert stopped['status']=='cancelled'
        validate_result(stopped['result'],stopped['selection'])
        try:os.kill(int(pid.read_text()),0)
        except ProcessLookupError:pass
        else:raise AssertionError('Cancelled private notebook kernel still running')
        assert (Path(stopped['result']['path'])/'report-data.json').is_file()
        assert h.workspace.db.serialize()==before
        # Mutation of retained artifact bytes cannot leave an old verified card.
        bad_path=Path(failed['result']['path'])/'executed.ipynb';original=bad_path.read_bytes()
        bad_path.write_bytes(original+b' ')
        js('document.getElementById("science-notebook-history").value='+json.dumps(failed['id'])+';document.getElementById("science-notebook-history").dispatchEvent(new Event("change"))')
        until('document.getElementById("toast-message").textContent.includes("bytes changed")')
        assert text('science-notebook-evidence')=='' and h.workspace.db.serialize()==before
        bad_path.write_bytes(original)
        # Every read remains external, explicit and read-only.
        click('science-notebook-refresh');until('document.getElementById("science-notebook-history").options.length===5')
        assert h.workspace.db.serialize()==before
        # Real generation button: new directory, no execution; inspected automatically.
        js('document.getElementById("science-kind").value="report";document.getElementById("science-kind").dispatchEvent(new Event("change"))')
        saved=fixture['reports']['result-report'];value=json.dumps(saved,separators=(',',':'))
        until('[...document.getElementById("science-object").options].some(o=>o.value==='+json.dumps(value)+')')
        js('document.getElementById("science-object").value='+json.dumps(value)+';document.getElementById("science-object").dispatchEvent(new Event("change"))')
        until('!document.getElementById("science-report").hidden')
        with patch('app.desktop_experiments.QFileDialog.getSaveFileName',return_value=(str(folder/'generated'),'') ):
            click('science-generate');until('document.getElementById("science-notebook-selection").textContent.includes('+json.dumps(str(folder/'generated'))+')')
        assert (folder/'generated/requested.ipynb').is_file() and not (folder/'generated/executions').exists()
        assert h.workspace.db.serialize()==before
        # Native confirmation is plaintext, default Cancel, full title and exact Details.
        owner=QWidget();observed=[]
        def close_confirmation():
            dialog=app.activeModalWidget();assert isinstance(dialog,QMessageBox)
            observed.append((dialog.windowTitle(),dialog.textFormat(),dialog.defaultButton()==dialog.button(QMessageBox.StandardButton.Cancel),dialog.detailedText()))
            dialog.reject()
        QTimer.singleShot(30,close_confirmation)
        assert not confirm_execution(owner,fixture['selections']['saved'],python,60)
        assert observed[0][0]=='Visual PyTorch — Execute notebook' and observed[0][1]==observed[0][1].PlainText
        assert observed[0][2] and fixture['selections']['saved']['requested_sha256'] in observed[0][3]
        owner.deleteLater()
        # Real inspection with deliberately delayed delivery to a replaced UUID.
        gate=threading.Event();prepared=threading.Event()
        def delayed(*args,**kwargs):
            result=inspect_isolated(*args,**kwargs);prepared.set()
            assert gate.wait(15),'Private inspection release timeout';return result
        origin=h.workspace
        with patch('app.desktop_notebooks.inspect_isolated',side_effect=delayed),patch('app.desktop_notebooks.QFileDialog.getExistingDirectory',return_value=str(folder/'saved')):
            click('science-notebook-inspect');until(lambda:prepared.is_set())
            h.workspace=Workspace();workspaces.append(h.workspace);h.broadcast()
            until('window.loom.state.workspace_id==='+json.dumps(workspace_id(h.workspace)))
            gate.set();until(lambda:not h.job.awaiting_terminal and not h.job.isRunning())
            until('document.getElementById("science-notebook-selection").textContent.startsWith("No bundle inspected")')
            assert js('document.getElementById("science-notebook-execute").disabled')
        h.workspace=origin;h.broadcast();until('window.loom.state.workspace_id==='+json.dumps(workspace_id(origin)))
        assert h.workspace.db.serialize()==before
        assert not any(x in sys.modules for x in ('torch','nbclient','jupyter_client'))
        assert not errors,errors
        print(json.dumps({'saved_and_rerun_real_kernel':'PASS','eight_rerun_fits':'PASS','failure_timeout_stop_artifacts':'PASS',
            'kernel_shutdown_and_sql_isolation':'PASS','generation_inspection_confirmation':'PASS','themes_fonts_safe_dom':'PASS',
            'late_inspection_and_corrupt_receipt_reads':'PASS',
            'host_torch':False,'host_nbclient':False,'console_errors':errors}))
    finally:
        if h.job:h.job.stop();h.job.wait(30000);QCoreApplication.processEvents()
        view.close();view.setPage(QWebEnginePage(view));page.deleteLater()
        QCoreApplication.sendPostedEvents(None,QEvent.Type.DeferredDelete);QCoreApplication.processEvents()
        profile.deleteLater();QCoreApplication.sendPostedEvents(None,QEvent.Type.DeferredDelete)
        for w in workspaces:w.db.close()
