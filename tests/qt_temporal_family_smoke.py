"""Real sandboxed Qt temporal-policy card, provenance, export and vtw15 reload."""
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
from app.core import Workspace, default_model, VERSION
from app.desktop import Bridge, Window
from app.family_export import stored_source
from test_temporal_families import family

assert not os.environ.get('QTWEBENGINE_DISABLE_SANDBOX')
assert '--no-sandbox' not in os.environ.get('QTWEBENGINE_CHROMIUM_FLAGS','')
app=QApplication(['temporal-family-smoke']);view=QWebEngineView();view.resize(1440,1000)
profile=QWebEngineProfile(view);errors=[]


class Page(QWebEnginePage):
    def javaScriptConsoleMessage(self,level,message,line,source):
        if level==self.JavaScriptConsoleMessageLevel.ErrorMessageLevel:errors.append(f'{source}:{line}: {message}')


class Harness(QObject):
    state=Window.state;describe=Window.describe
    def __init__(self,folder):
        super().__init__();self.config=Config(folder/'config.json');self.workspace=Workspace();self.workspace.put(default_model())
        self.family_ref=self.workspace.put_spec('family',family(dynamic=False));self.workspace.save(folder/'original.vtw')
        self.job=None;self.torch_python=None;self.rendering={};self.requests=[];self.bridge=Bridge(self,view)
    def recent_files(self):return []
    def dispatch(self,request,source=None):
        self.requests.append(copy.deepcopy(request))
        assert request['action'] not in ('train','scienceTrain','scienceRunCandidate','scienceNotebookExecute')
        return Window.dispatch(self,request,source)
    def broadcast(self,kind='state',payload=None,**kwargs):self.bridge.send(kind,self.state() if kind=='state' else payload)


def js(source):
    loop=QEventLoop();timer=QTimer();timer.setSingleShot(True);timer.timeout.connect(loop.quit);values=[]
    view.page().runJavaScript(source,lambda value:(values.append(value),loop.quit()));timer.start(4000);loop.exec();timer.stop()
    assert values,'JavaScript callback timeout';return values[0]


def until(source):
    deadline=time.monotonic()+25
    while time.monotonic()<deadline:
        if js(source):return
        loop=QEventLoop();QTimer.singleShot(20,loop.quit);loop.exec()
    status=js('document.getElementById("science-family-instance-status")?.textContent')
    raise AssertionError(f'UI timeout: {source}; instance status: {status}; errors: {errors}; requests: {h.requests[-3:]}')


def click(identifier):js('document.getElementById('+json.dumps(identifier)+').click()')
def change(value):
    js('(()=>{const n=document.querySelector("#science-family-roots [data-key=runtime_policy]");n.value='+json.dumps(value)+';n.dispatchEvent(new Event("change"));})()')


with tempfile.TemporaryDirectory() as td:
    folder=Path(td);h=Harness(folder);page=Page(profile,view);view.setPage(page)
    channel=QWebChannel(page);channel.registerObject('host',h.bridge);page.setWebChannel(channel)
    url=QUrl.fromLocalFile(str(Path(__file__).resolve().parents[1]/'app/web/index.html'))
    url.setFragment('model='+next(iter(h.workspace.models())));view.setUrl(url);view.show()
    try:
        until('Boolean(window.loom?.document?.view.fitted)')
        js('document.querySelector("[data-panel=science]").click();document.getElementById("experiment-dock").style.height="850px";document.getElementById("experiment-dock").style.width="1080px"')
        until('!document.getElementById("science-panel").hidden')
        js('(()=>{const n=document.getElementById("science-kind");n.value="family";n.dispatchEvent(new Event("change"));})()')
        until('document.getElementById("science-object").options.length===1')
        js('(()=>{const n=document.getElementById("science-object");n.value=n.options[0].value;n.dispatchEvent(new Event("change"));})()')
        click('science-author-edit');until('!document.getElementById("science-family-check").disabled')
        assert js('document.querySelector("#science-family-roots [data-key=runtime_policy]").tagName==="TEXTAREA"')
        assert js('document.querySelector("#science-family-roots [data-key=version]").value==="6"')
        before=h.workspace.db.serialize();change('not JSON');click('science-family-preview')
        assert js('document.querySelector("#science-family-roots [data-key=runtime_policy]").value==="not JSON"')
        assert h.workspace.db.serialize()==before
        change(json.dumps(family()['runtime_policy']));click('science-family-preview')
        until('JSON.parse(document.getElementById("science-family-preview-result").textContent||"{}").runtime_status==="conditional-per-call-guards"')
        assert js('JSON.parse(document.getElementById("science-family-preview-result").textContent).parameters===10')
        assert h.workspace.db.serialize()==before
        if os.environ.get('VTW_TEMPORAL_FAMILY_CAPTURE'):
            click('toast-close');loop=QEventLoop();QTimer.singleShot(150,loop.quit);loop.exec()
            assert view.grab().save(os.environ['VTW_TEMPORAL_FAMILY_CAPTURE'])
        click('science-family-save');until('document.getElementById("science-family-editor").hidden')
        saved=h.workspace.get_spec('family',h.family_ref['id']);assert saved['version']==6
        assert saved['runtime_policy']==family()['runtime_policy']
        until('!document.getElementById("science-family-export").disabled && JSON.parse(document.getElementById("science-object").value).revision==='+json.dumps(saved['revision']))
        exported=folder/'family.py';before=h.workspace.db.serialize()
        with patch('app.desktop_experiments.QFileDialog.getSaveFileName',return_value=(str(exported),'')):
            click('science-family-export');until('document.getElementById("toast").textContent.includes("Checked family source exported")')
        request=next(r for r in reversed(h.requests) if r['action']=='scienceExportFamily')
        expected=stored_source(h.workspace,request['family'],memory_limit_percent=h.config.values['memory_limit_percent'])
        assert expected['provenance']['runtime_policy']==family()['runtime_policy']
        assert exported.read_text()==expected['code'];assert h.workspace.db.serialize()==before
        click('science-family-create');until('document.getElementById("science-family-instance-status").textContent.includes("10 parameters")')
        instance=h.workspace.list_specs('specialization')[0];record=h.workspace.get_spec('specialization',instance['id'])
        assert record['version']==5 and record['runtime']['kind']=='runtime-temporal-v1'
        h.workspace.save(folder/'final.vtw');opened=Workspace.load(folder/'final.vtw')
        try:
            assert opened.get_spec('specialization',instance['id'],record['revision'])==record
            assert opened.db.execute('PRAGMA user_version').fetchone()[0]==15==VERSION
        finally:opened.db.close()
        assert not errors,errors;assert 'torch' not in sys.modules
        print(json.dumps({'ok':True,'temporal_card':True,'explicit_family6':True,'invalid_text_retained':True,
                          'parameters':10,'preview_sql_unchanged':True,'checked_export_exact':True,'vtw15_reload':True,
                          'specialization5_witness':True,'host_torch':False,'console_errors':errors}))
    finally:
        view.setPage(QWebEnginePage(view));page.deleteLater();channel.deleteLater()
        QCoreApplication.sendPostedEvents(None,QEvent.Type.DeferredDelete);app.processEvents();profile.deleteLater()
        QCoreApplication.sendPostedEvents(None,QEvent.Type.DeferredDelete);app.processEvents();view.close();h.workspace.db.close()
        if h.workspace.history.directory:h.workspace.history.directory.cleanup()
