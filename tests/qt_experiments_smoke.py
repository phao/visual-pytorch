"""Real sandboxed Qt/WebEngine scientific browsing/run/retry/report contracts."""
import json
import os
import sys
import tempfile
import time
from unittest.mock import patch
from pathlib import Path
from PyQt6.QtCore import QObject,QEvent,QEventLoop,QTimer,QUrl,QCoreApplication
from PyQt6.QtWidgets import QApplication
from PyQt6.QtWebChannel import QWebChannel
from PyQt6.QtWebEngineCore import QWebEnginePage,QWebEngineProfile
from PyQt6.QtWebEngineWidgets import QWebEngineView
from app.config import Config
from app.core import Workspace
from app.desktop import Bridge,Window,find_torch_python
from examples.experiment_workspace import populate

assert not os.environ.get('QTWEBENGINE_DISABLE_SANDBOX')
assert '--no-sandbox' not in os.environ.get('QTWEBENGINE_CHROMIUM_FLAGS','')
app=QApplication(['experiment-ui-smoke']);view=QWebEngineView();view.resize(1440,1000)
profile=QWebEngineProfile(view);errors=[]
class Page(QWebEnginePage):
    def javaScriptConsoleMessage(self,level,message,line,source):
        if level==self.JavaScriptConsoleMessageLevel.ErrorMessageLevel:errors.append(f'{source}:{line}: {message}')
class Harness(QObject):
    state=Window.state;describe=Window.describe;dispatch=Window.dispatch
    def __init__(self,path):
        super().__init__();self.config=Config(path/'config.json');self.workspace=Workspace()
        self.workspace.set_constrained_mode(False);self.e=populate(self.workspace)
        # Use three-update fixtures, not the longer teaching example budget.
        self.e.pop('revision');self.e['protocols'][0]['protocol']['budget']['updates']=3
        self.e['protocols'][0]['protocol']['evaluate_every']=2
        self.eref=self.workspace.put_spec('experiment',self.e)
        self.job=None;self.torch_python=find_torch_python();assert self.torch_python
        self.rendering={};self.bridge=Bridge(self,view);self.events=[]
    def recent_files(self):return []
    def broadcast(self,kind='state',payload=None,exclude=None):
        self.events.append((kind,payload));self.bridge.send(kind,self.state() if kind=='state' else payload)
def js(script):
    loop=QEventLoop();timer=QTimer();timer.setSingleShot(True);timer.timeout.connect(loop.quit);values=[]
    def receive(value):values.append(value);loop.quit()
    view.page().runJavaScript(script,receive);timer.start(3000);loop.exec();timer.stop()
    assert values,'JavaScript timeout';return values[0]
def until(script,timeout=25):
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        if js(script):return
        loop=QEventLoop();QTimer.singleShot(25,loop.quit);loop.exec()
    raise AssertionError(f'UI timeout: {script}; console: {errors[:5]}')
with tempfile.TemporaryDirectory() as directory:
    path=Path(directory);h=Harness(path)
    page=Page(profile,view);view.setPage(page);channel=QWebChannel(page);channel.registerObject('host',h.bridge);page.setWebChannel(channel)
    url=QUrl.fromLocalFile(str(Path(__file__).resolve().parents[1]/'app/web/index.html'))
    url.setFragment('model='+next(iter(h.workspace.models())));view.setUrl(url);view.show()
    try:
        until('Boolean(window.loom?.document)')
        before=h.workspace.db.serialize()
        try:h.dispatch({'action':'scienceShow','reference':{'kind':'experiment','id':h.eref['id'],'revision':None}},None)
        except ValueError as error:assert 'SHA-256' in str(error)
        else:raise AssertionError('Scientific browser silently followed a null revision')
        assert h.workspace.db.serialize()==before
        js('document.querySelector("[data-panel=science]").click();document.getElementById("experiment-dock").style.height="650px";')
        until('document.querySelectorAll(".science-candidate").length===8')
        assert js('document.getElementById("experiment-panel").hidden && !document.getElementById("science-panel").hidden')
        assert h.workspace.list_specs('run')==[]
        assert js('document.querySelectorAll("[data-science-run]:not(:disabled)").length')==8
        js('document.querySelector("[data-science-run]").click()')
        until('Boolean(window.loom.state.experiment_job)')
        assert js('document.getElementById("science-import").disabled')
        try:h.dispatch({'action':'scienceFreeze','id':next(iter(h.workspace.models()))},None)
        except ValueError as error:assert 'Stop the current run' in str(error)
        else:raise AssertionError('Scientific mutation admitted during a run')
        until('!window.loom.state.experiment_job && document.getElementById("toast-message").textContent.includes("completed")')
        runs=h.workspace.list_specs('run');assert len(runs)==1
        record=h.workspace.get_spec('run',runs[0]['id'],runs[0]['revision']);assert record['attempts'][0]['result']['counts']['completed_updates']==3
        js('document.getElementById("science-kind").value="run";document.getElementById("science-kind").dispatchEvent(new Event("change"));')
        until('!document.getElementById("science-run").hidden')
        assert js('document.getElementById("science-status").textContent.includes("completed")')
        js('document.getElementById("science-retry").click()')
        until('Boolean(window.loom.state.experiment_job)')
        assert js('document.getElementById("science-stop").getClientRects().length>0'), 'Stop must remain visible in run history'
        until('!window.loom.state.experiment_job && document.getElementById("toast-message").textContent.includes("attempt 2")')
        run=h.workspace.get_spec('run',runs[0]['id']);assert len(run['attempts'])==2
        until('document.getElementById("science-object").options.length===2 && JSON.parse(document.getElementById("science-document").textContent||"{}").attempt_count===1')
        assert js('JSON.parse(document.getElementById("science-document").textContent).attempt_count')==1
        js('const choices=document.getElementById("science-object");choices.value=[...choices.options].find(o=>o.textContent.includes("current head")).value;choices.dispatchEvent(new Event("change"));')
        until('JSON.parse(document.getElementById("science-document").textContent||"{}").attempt_count===2')
        js('document.getElementById("science-make-report").click()')
        until('!document.getElementById("science-report").hidden')
        assert js('document.getElementById("science-report-mode").textContent.includes("no training")')
        before=len(h.workspace.list_specs('run',True));reports=h.workspace.list_specs('report')
        saved=[r for r in reports if r['name']=='Saved run evidence'][0]
        result=h.dispatch({'action':'scienceGenerateNotebook','reference':{k:saved[k] for k in ('kind','id','revision')},'output':str(path/'notebook')},None)
        assert result['status']=='generated-not-executed';assert len(h.workspace.list_specs('run',True))==before
        assert (path/'notebook/requested.ipynb').is_file()
        h.workspace.save(path/'science.vtw');loaded=Workspace.load(path/'science.vtw')
        assert len(loaded.get_spec('run',runs[0]['id'])['attempts'])==2;loaded.db.close()
        # Force terminal SQL admission failure through the actual worker/bridge.
        # UI recovery remains inspectable after replacing the QThread object.
        js('document.getElementById("science-kind").value="experiment";document.getElementById("science-kind").dispatchEvent(new Event("change"));')
        until('document.querySelectorAll(".science-candidate").length===8')
        put=h.workspace.put_spec
        def reject_run(kind, document, **kw):
            if kind=='run':raise ValueError('Forced GUI admission failure')
            return put(kind,document,**kw)
        with patch.object(h.workspace,'put_spec',side_effect=reject_run):
            js('document.querySelector("[data-science-run]").click()')
            until('Boolean(window.loom.state.experiment_job)')
            until('!window.loom.state.experiment_job && !document.getElementById("science-recovery").hidden')
        assert len(h.workspace.list_specs('run',True))==before
        assert len(h.experiment_recoveries)==1
        job=h.job;assert job.wait(15000);h.job=None;job.deleteLater();h.broadcast()
        js('document.getElementById("science-recovery-show").click()')
        until('document.getElementById("science-tree-status").textContent.includes("session recovery evidence")')
        selected=json.loads(js('document.getElementById("science-document").textContent'))
        assert selected['scope']=='session-recovery-summary'
        assert js('document.getElementById("science-tree-edit").disabled')
        # The real view projects the native evidence instead of copying it all.
        evidence=h.experiment_recoveries[selected['id']]
        assert js('[...document.querySelectorAll("#science-tree-rows tr")].some(r=>r.textContent.includes("visual-pytorch-experiment-recovery"))')
        assert evidence['snapshot']['origin']['workspace_id']==h.state()['workspace_id']
        assert evidence['result']['record']['attempts'][-1]['status']=='completed'
        assert evidence['snapshot']['documents']
        destination=path/'recovery.json'
        with patch('app.desktop_experiments.QFileDialog.getSaveFileName',return_value=(str(destination),'')):
            js('document.getElementById("science-recovery-export").click()')
            until('document.getElementById("toast-message").textContent.includes("Recovery evidence exported")')
        assert json.loads(destination.read_text())['result']==evidence['result']
        assert len(h.workspace.list_specs('run',True))==before
        assert not errors,errors;assert 'torch' not in sys.modules
        notices=[payload for kind,payload in h.events if kind in ('scienceComplete','scienceRecovery')]
        assert notices and all(len(json.dumps(value,ensure_ascii=True,separators=(',',':')).encode('ascii'))<=65536 for value in notices)
        assert all('record' not in value and 'snapshot' not in value and 'result' not in value for value in notices)
        if os.environ.get('VTW_SMOKE_SCREENSHOT'):
            js('document.getElementById("science-kind").value="run";document.getElementById("science-kind").dispatchEvent(new Event("change"));')
            until('!document.getElementById("science-run").hidden')
            loop=QEventLoop();QTimer.singleShot(300,loop.quit);loop.exec();assert view.grab().save(os.environ['VTW_SMOKE_SCREENSHOT'])
        print(json.dumps({'experiment_browser':'PASS','exact_revisions':'PASS','actual_worker_fit_retry':'PASS','independent_run_history':'PASS','saved_report_no_training':'PASS','notebook_generation':'PASS','save_reload':'PASS','failed_admission_recovery_export':'PASS','host_torch':False,'console_errors':errors}))
    finally:
        if h.job and h.job.isRunning():h.job.stop();h.job.wait(15000)
        if h.job:h.job.wait(15000)
        h.workspace.history.directory.cleanup();h.workspace.db.close()
        view.close();view.setPage(QWebEnginePage(view));page.deleteLater();QCoreApplication.sendPostedEvents(None,QEvent.Type.DeferredDelete)
        profile.deleteLater();QCoreApplication.sendPostedEvents(None,QEvent.Type.DeferredDelete)
