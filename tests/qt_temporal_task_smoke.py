"""Actual sandboxed Qt explicit temporal experiment edit and worker delivery."""
import copy
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from PyQt6.QtCore import QObject, QCoreApplication, QEvent, QEventLoop, QTimer, QUrl
from PyQt6.QtWidgets import QApplication
from PyQt6.QtWebChannel import QWebChannel
from PyQt6.QtWebEngineCore import QWebEnginePage, QWebEngineProfile
from PyQt6.QtWebEngineWidgets import QWebEngineView
from app.config import Config
from app.core import Workspace, VERSION, default_model
from app.revisions import validate_all
from test_temporal_tasks import temporal_experiment

assert not os.environ.get('QTWEBENGINE_DISABLE_SANDBOX')
assert '--no-sandbox' not in os.environ.get('QTWEBENGINE_CHROMIUM_FLAGS','')
app=QApplication(['temporal-task-smoke']);view=QWebEngineView();view.resize(1440,1000)
profile=QWebEngineProfile(view);errors=[]
class Page(QWebEnginePage):
    def javaScriptConsoleMessage(self,level,message,line,source):
        if level==self.JavaScriptConsoleMessageLevel.ErrorMessageLevel:errors.append(f'{source}:{line}: {message}')
class Harness(QObject):
    state=WindowState=None
    def __init__(self,folder):
        from app.desktop import Bridge, Window
        super().__init__();self.config=Config(folder/'config.json')
        self.workspace,self.e,self.ref=temporal_experiment(times=(7,),version=3,execution=None,updates=3)
        self.workspace.put(default_model());self.workspace.save(folder/'original.vtw')
        self.job=None;self.torch_python='/home/pedro/miniforge3/bin/python3.12'
        self.rendering={};self.requests=[];self.bridge=Bridge(self,view)
    def recent_files(self):return []
    def dispatch(self,req,source=None):
        from app.desktop import Window
        self.requests.append(copy.deepcopy(req));return Window.dispatch(self,req,source)
    def broadcast(self,kind='state',payload=None,**kwargs):self.bridge.send(kind,self.state() if kind=='state' else payload)
from app.desktop import Window
Harness.state=Window.state;Harness.describe=Window.describe
def js(source):
    loop=QEventLoop();timer=QTimer();timer.setSingleShot(True);timer.timeout.connect(loop.quit);values=[]
    view.page().runJavaScript(source,lambda value:(values.append(value),loop.quit()));timer.start(4000);loop.exec();timer.stop()
    assert values,'JS callback timeout';return values[0]
def until(source,seconds=25):
    deadline=time.monotonic()+seconds
    while time.monotonic()<deadline:
        if js(source):return
        loop=QEventLoop();QTimer.singleShot(20,loop.quit);loop.exec()
    raise AssertionError(f'UI timeout: {source}; errors: {errors}; requests: {h.requests[-3:]}')
def click(identifier):js('document.getElementById('+json.dumps(identifier)+').click()')
def change(source,value):js('(()=>{const n='+source+';n.value='+json.dumps(value)+';n.dispatchEvent(new Event("change"));})()')

with tempfile.TemporaryDirectory() as td:
    folder=Path(td);h=Harness(folder);page=Page(profile,view);view.setPage(page)
    channel=QWebChannel(page);channel.registerObject('host',h.bridge);page.setWebChannel(channel)
    url=QUrl.fromLocalFile(str(Path(__file__).resolve().parents[1]/'app/web/index.html'))
    url.setFragment('model='+next(iter(h.workspace.models())));view.setUrl(url);view.show()
    try:
        until('Boolean(window.loom?.document?.view.fitted)')
        js('document.querySelector("[data-panel=science]").click();document.getElementById("experiment-dock").style.height="850px";document.getElementById("experiment-dock").style.width="1080px"')
        until('!document.getElementById("science-panel").hidden')
        change('document.getElementById("science-kind")','experiment')
        until('document.getElementById("science-object").options.length===1')
        js('(()=>{const n=document.getElementById("science-object");n.value=n.options[0].value;n.dispatchEvent(new Event("change"));})()')
        until('document.querySelectorAll("#science-candidates [data-science-run]").length===1')
        before=h.workspace.db.serialize();click('science-author-edit')
        until('!document.getElementById("science-author-check").disabled')
        field='([...document.querySelectorAll(\'[data-experiment-axis="model"] label\')].find(n=>n.firstChild.textContent==="Family execution").querySelector("select"))'
        assert js(field+'.value')=='concrete-v1';change(field,'runtime-temporal-v1');click('science-experiment-preview')
        until('document.querySelector("#science-experiment-preview-rows .science-candidate")?.textContent.includes("compatible")')
        assert h.workspace.db.serialize()==before
        request=next(r for r in h.requests if r['action']=='scienceDraftExperimentEdit')
        assert request['operations'][0]['key']=='version' and request['operations'][0]['text']=='5'
        assert request['operations'][1]['text']=='"runtime-temporal-v1"'
        version='([...document.querySelectorAll("#science-experiment-bindings label")].find(n=>n.firstChild.textContent==="Selection version · 4 enables spatial, 5 temporal runtime").querySelector("input"))'
        assert js(version+'.value')=='5'
        if os.environ.get('VTW_TEMPORAL_TASK_CAPTURE'):
            loop=QEventLoop();QTimer.singleShot(150,loop.quit);loop.exec();assert view.grab().save(os.environ['VTW_TEMPORAL_TASK_CAPTURE'])
        click('science-author-save');until('document.getElementById("science-author").hidden && !document.querySelector("#science-candidates [data-science-run]").disabled')
        saved=h.workspace.get_spec('experiment',h.e['id']);assert saved['version']==5
        assert saved['models'][0]['execution']=='runtime-temporal-v1'
        js('document.querySelector("#science-candidates [data-science-run]").click()')
        until('Boolean(window.loom.state.experiment_job)')
        until('!window.loom.state.experiment_job && document.getElementById("toast-message").textContent.includes("completed")',40)
        run=h.workspace.get_spec('run',h.workspace.list_specs('run')[0]['id']);attempt=run['attempts'][0]
        assert run['version']==5 and attempt['status']=='completed'
        assert attempt['result']['counts']['completed_updates']==3
        assert attempt['result']['identity']['compiler']=='native-runtime-temporal-v1'
        validate_all(h.workspace.db);h.workspace.save(folder/'final.vtw');loaded=Workspace.load(folder/'final.vtw')
        try:assert loaded.get_spec('run',run['id'],run['revision'])==run
        finally:loaded.db.close()
        assert not errors,errors;assert 'torch' not in sys.modules
        print(json.dumps({'ok':True,'explicit_execution_upgrade':5,'no_preview_sql_writes':True,
            'runtime_T':7,'constructor_baseline_T':4,'worker_updates':3,
            'save_reload_version':VERSION,'host_torch':False,'console_errors':errors}))
    finally:
        if h.job:assert h.job.wait(15000)
        view.setPage(QWebEnginePage(view));page.deleteLater();channel.deleteLater()
        QCoreApplication.sendPostedEvents(None,QEvent.Type.DeferredDelete);app.processEvents();profile.deleteLater()
        QCoreApplication.sendPostedEvents(None,QEvent.Type.DeferredDelete);app.processEvents();view.close();h.workspace.db.close()
        if h.workspace.history.directory:h.workspace.history.directory.cleanup()
