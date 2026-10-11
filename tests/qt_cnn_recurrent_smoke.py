"""Real sandboxed Qt drawing and padded CNN/recurrent worker."""
import copy
import json
import os
import tempfile
import time
from pathlib import Path

from PyQt6.QtCore import QObject, QCoreApplication, QEvent, QEventLoop, QTimer, QUrl
from PyQt6.QtWidgets import QApplication
from PyQt6.QtWebChannel import QWebChannel
from PyQt6.QtWebEngineCore import QWebEnginePage, QWebEngineProfile
from PyQt6.QtWebEngineWidgets import QWebEngineView

from app.config import Config
from app.core import Workspace, VERSION, validate_model
from app.desktop import Bridge, Window
from examples.cnn_recurrent_experiment import populate

assert not os.environ.get('QTWEBENGINE_DISABLE_SANDBOX')
assert '--no-sandbox' not in os.environ.get('QTWEBENGINE_CHROMIUM_FLAGS','')
app=QApplication(['cnn-recurrent-smoke']);view=QWebEngineView();view.resize(1440,1000)
profile=QWebEngineProfile(view);errors=[]
class Page(QWebEnginePage):
    def javaScriptConsoleMessage(self,level,message,line,source):
        if level==self.JavaScriptConsoleMessageLevel.ErrorMessageLevel:errors.append(f'{source}:{line}: {message}')
class Harness(QObject):
    state=Window.state;describe=Window.describe
    def __init__(self,folder):
        super().__init__();self.config=Config(folder/'config.json');self.workspace=Workspace()
        self.workspace.set_constrained_mode(False);self.info=populate(self.workspace,updates=3)
        self.mid=next(k for k,m in self.workspace.models().items() if any(n['type']=='Recurrent' for n in m['graph']['nodes']))
        self.workspace.save(folder/'initial.vtw');self.job=None;self.torch_python='/home/pedro/miniforge3/bin/python3.12'
        self.rendering={};self.requests=[];self.bridge=Bridge(self,view)
    def recent_files(self):return []
    def dispatch(self,req,source=None):
        self.requests.append(copy.deepcopy(req));return Window.dispatch(self,req,source)
    def broadcast(self,kind='state',payload=None,**kwargs):self.bridge.send(kind,self.state() if kind=='state' else payload)
def js(source):
    loop=QEventLoop();timer=QTimer();timer.setSingleShot(True);timer.timeout.connect(loop.quit);values=[]
    view.page().runJavaScript(source,lambda value:(values.append(value),loop.quit()));timer.start(4000);loop.exec();timer.stop()
    assert values,'JavaScript callback timeout';return values[0]
def until(source,seconds=25):
    end=time.monotonic()+seconds
    while time.monotonic()<end:
        if js(source):return
        loop=QEventLoop();QTimer.singleShot(20,loop.quit);loop.exec()
    raise AssertionError(f'UI timeout: {source}; {errors}; {h.requests[-3:]}')
def change(identifier,value):
    js('(()=>{const n=document.getElementById('+json.dumps(identifier)+');n.value='+json.dumps(value)+';n.dispatchEvent(new Event("change"));})()')

with tempfile.TemporaryDirectory() as td:
    folder=Path(td);h=Harness(folder);page=Page(profile,view);view.setPage(page)
    channel=QWebChannel(page);channel.registerObject('host',h.bridge);page.setWebChannel(channel)
    url=QUrl.fromLocalFile(str(Path(__file__).resolve().parents[1]/'app/web/index.html'));url.setFragment('model='+h.mid)
    view.setUrl(url);view.show()
    try:
        until('Boolean(window.loom?.document?.view.fitted)')
        c=validate_model(h.workspace.models(),h.mid);assert c['parameters']==14 and len(c['input_groups'])==5
        if os.environ.get('VTW_CNN_RECURRENT_CAPTURE'):
            loop=QEventLoop();QTimer.singleShot(150,loop.quit);loop.exec();assert view.grab().save(os.environ['VTW_CNN_RECURRENT_CAPTURE'])
        js('document.querySelector("[data-panel=science]").click();document.getElementById("experiment-dock").style.height="850px";document.getElementById("experiment-dock").style.width="1080px"')
        until('!document.getElementById("science-panel").hidden');change('science-kind','experiment')
        until('document.getElementById("science-object").options.length===1')
        js('(()=>{const n=document.getElementById("science-object");n.value=n.options[0].value;n.dispatchEvent(new Event("change"));})()')
        until('document.querySelector("#science-candidates [data-science-run]")?.disabled===false')
        js('document.querySelector("#science-candidates [data-science-run]").click()')
        until('Boolean(window.loom.state.experiment_job)')
        until('!window.loom.state.experiment_job && document.getElementById("toast-message").textContent.includes("completed")',40)
        record=h.workspace.get_spec('run',h.workspace.list_specs('run')[0]['id']);a=record['attempts'][0]
        assert record['version']==6 and a['status']=='completed',a
        assert a['result']['counts']['completed_updates']==3
        assert a['result']['identity']['compiler']=='native-runtime-temporal-v2'
        h.workspace.save(folder/'final.vtw');loaded=Workspace.load(folder/'final.vtw')
        try:assert loaded.get_spec('run',record['id'],record['revision'])==record
        finally:loaded.db.close()
        import sys
        assert 'torch' not in sys.modules and not errors,errors
        print(json.dumps({'ok':True,'materialized_models':2,'parameters':14,'named_inputs':5,
            'scope':'CNN in referenced recurrent cell; explicit length/mask/reset; temporal timestamp groups',
            'worker_updates':3,'container_version':VERSION,'host_torch':False,'console_errors':errors}))
    finally:
        if h.job:assert h.job.wait(15000)
        view.setPage(QWebEnginePage(view));page.deleteLater();channel.deleteLater()
        QCoreApplication.sendPostedEvents(None,QEvent.Type.DeferredDelete);app.processEvents();profile.deleteLater()
        QCoreApplication.sendPostedEvents(None,QEvent.Type.DeferredDelete);app.processEvents();view.close();h.workspace.db.close()
        if h.workspace.history.directory:h.workspace.history.directory.cleanup()
