"""Actual sandboxed Qt runtime root editor, save/reload and checked export."""
import copy
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch
from PyQt6.QtCore import QObject, QCoreApplication, QEvent, QEventLoop, QTimer, QUrl
from PyQt6.QtWidgets import QApplication
from PyQt6.QtWebChannel import QWebChannel
from PyQt6.QtWebEngineCore import QWebEnginePage, QWebEngineProfile
from PyQt6.QtWebEngineWidgets import QWebEngineView
from app.config import Config
from app.core import Workspace, default_model
from app.desktop import Bridge, Window
from app.family_export import stored_source
from test_retained_runtime import source

assert not os.environ.get('QTWEBENGINE_DISABLE_SANDBOX')
assert '--no-sandbox' not in os.environ.get('QTWEBENGINE_CHROMIUM_FLAGS','')
app=QApplication(['runtime-family-smoke']);view=QWebEngineView();view.resize(1440,1000)
profile=QWebEngineProfile(view);errors=[]
class Page(QWebEnginePage):
    def javaScriptConsoleMessage(self,level,message,line,source):
        if level==self.JavaScriptConsoleMessageLevel.ErrorMessageLevel:errors.append(f'{source}:{line}: {message}')
class Harness(QObject):
    state=Window.state;describe=Window.describe
    def __init__(self,folder):
        super().__init__();self.config=Config(folder/'config.json');self.workspace=Workspace();self.workspace.put(default_model())
        doc=source();doc['version']=4;doc.pop('runtime_inputs');self.family_ref=self.workspace.put_spec('family',doc)
        self.workspace.save(folder/'original.vtw');self.job=None;self.torch_python=None
        self.rendering={};self.requests=[];self.bridge=Bridge(self,view)
    def recent_files(self):return []
    def dispatch(self,req,source=None):
        self.requests.append(copy.deepcopy(req));assert req['action'] not in ('train','scienceTrain','scienceRunCandidate','scienceNotebookExecute')
        return Window.dispatch(self,req,source)
    def broadcast(self,kind='state',payload=None,**kwargs):self.bridge.send(kind,self.state() if kind=='state' else payload)
def js(source):
    loop=QEventLoop();timer=QTimer();timer.setSingleShot(True);timer.timeout.connect(loop.quit);values=[]
    view.page().runJavaScript(source,lambda value:(values.append(value),loop.quit()));timer.start(4000);loop.exec();timer.stop()
    assert values,'JS callback timeout';return values[0]
def until(source):
    deadline=time.monotonic()+25
    while time.monotonic()<deadline:
        if js(source):return
        loop=QEventLoop();QTimer.singleShot(20,loop.quit);loop.exec()
    raise AssertionError(f'UI timeout: {source}; errors: {errors}; requests: {h.requests[-3:]}')
def click(identifier):js('document.getElementById('+json.dumps(identifier)+').click()')
def change(value):
    js('(()=>{const n=document.querySelector("#science-family-roots [data-key=runtime_inputs]");n.value='+json.dumps(value)+';n.dispatchEvent(new Event("change"));})()')

with tempfile.TemporaryDirectory() as td:
    folder=Path(td);h=Harness(folder);page=Page(profile,view);view.setPage(page)
    channel=QWebChannel(page);channel.registerObject('host',h.bridge);page.setWebChannel(channel)
    url=QUrl.fromLocalFile(str(Path(__file__).resolve().parents[1]/'app/web/index.html'));url.setFragment('model='+next(iter(h.workspace.models())));view.setUrl(url);view.show()
    try:
        until('Boolean(window.loom?.document?.view.fitted)')
        js('document.querySelector("[data-panel=science]").click();document.getElementById("experiment-dock").style.height="850px";document.getElementById("experiment-dock").style.width="1080px"')
        until('!document.getElementById("science-panel").hidden')
        js('(()=>{const n=document.getElementById("science-kind");n.value="family";n.dispatchEvent(new Event("change"));})()')
        until('document.getElementById("science-object").options.length===1')
        js('(()=>{const n=document.getElementById("science-object");n.value=n.options[0].value;n.dispatchEvent(new Event("change"));})()')
        click('science-author-edit');until('!document.getElementById("science-family-check").disabled')
        before=h.workspace.db.serialize();change('not JSON');click('science-family-preview')
        assert js('document.querySelector("#science-family-roots [data-key=runtime_inputs]").value==="not JSON"')
        assert h.workspace.db.serialize()==before
        change('{"input":[1,2]}');click('science-family-preview')
        until('JSON.parse(document.getElementById("science-family-preview-result").textContent||"{}").runtime_status==="conditional-per-call-guards"')
        assert js('JSON.parse(document.getElementById("science-family-preview-result").textContent).parameters===179')
        assert js('document.querySelector("#science-family-roots [data-key=version]").value==="5"')
        assert h.workspace.db.serialize()==before
        edit=next(r for r in h.requests if r['action']=='scienceDraftFamilyEdit')
        assert edit['operations'][0]['key']=='version' and edit['operations'][0]['text']=='5'
        assert edit['operations'][1]['key']=='runtime_inputs'
        if os.environ.get('VTW_RUNTIME_CAPTURE'):
            click('toast-close')
            loop=QEventLoop();QTimer.singleShot(150,loop.quit);loop.exec();assert view.grab().save(os.environ['VTW_RUNTIME_CAPTURE'])
        click('science-family-save');until('document.getElementById("science-family-editor").hidden')
        saved=h.workspace.get_spec('family',h.family_ref['id']);assert saved['version']==5 and saved['runtime_inputs']=={'input':[1,2]}
        until('!document.getElementById("science-family-export").disabled && JSON.parse(document.getElementById("science-object").value).revision==='+json.dumps(saved['revision']))
        exported=folder/'family.py';before=h.workspace.db.serialize()
        with patch('app.desktop_experiments.QFileDialog.getSaveFileName',return_value=(str(exported),'')):
            click('science-family-export');until('document.getElementById("toast").textContent.includes("Checked family source exported")')
        request=next(r for r in reversed(h.requests) if r['action']=='scienceExportFamily')
        expected=stored_source(h.workspace,request['family'],memory_limit_percent=h.config.values['memory_limit_percent'])
        assert expected['provenance']['runtime_inputs']=={'input':[1,2]};assert exported.read_text()==expected['code'];assert h.workspace.db.serialize()==before
        click('science-family-create');until('document.getElementById("science-family-instance-status").textContent.includes("179 parameters")')
        instance=h.workspace.list_specs('specialization')[0];record=h.workspace.get_spec('specialization',instance['id'])
        assert record['version']==4 and record['runtime']['inputs']=={'input':[1,2]}
        h.workspace.save(folder/'final.vtw');loaded=Workspace.load(folder/'final.vtw')
        try:assert loaded.get_spec('specialization',instance['id'],record['revision'])==record
        finally:loaded.db.close()
        assert not errors,errors;assert 'torch' not in sys.modules
        print(json.dumps({'ok':True,'runtime_root_editor':True,'explicit_v5_upgrade':True,'invalid_text_retained':True,'parameters':179,
                          'no_preview_sql_writes':True,'checked_export_exact_bytes':True,'save_reload_v12':True,'specialization_version':4,'host_torch':False,'console_errors':errors}))
    finally:
        view.setPage(QWebEnginePage(view));page.deleteLater();channel.deleteLater()
        QCoreApplication.sendPostedEvents(None,QEvent.Type.DeferredDelete);app.processEvents();profile.deleteLater()
        QCoreApplication.sendPostedEvents(None,QEvent.Type.DeferredDelete);app.processEvents();view.close();h.workspace.db.close()
        if h.workspace.history.directory:h.workspace.history.directory.cleanup()
