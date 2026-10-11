"""Actual sandboxed Qt family editing, specialization, undo and .vtw reload."""
import copy
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from PyQt6.QtCore import QObject, QEventLoop, QTimer, QUrl, QEvent, QCoreApplication
from PyQt6.QtWidgets import QApplication
from PyQt6.QtWebChannel import QWebChannel
from PyQt6.QtWebEngineCore import QWebEnginePage, QWebEngineProfile
from PyQt6.QtWebEngineWidgets import QWebEngineView
from app.config import Config
from app.core import Workspace, default_model, validate_model
from app.desktop import Bridge, Window
from app.science_browser import encoded

assert not os.environ.get('QTWEBENGINE_DISABLE_SANDBOX')
assert '--no-sandbox' not in os.environ.get('QTWEBENGINE_CHROMIUM_FLAGS','')
ROOT=Path(__file__).resolve().parents[1]
app=QApplication(['family-authoring-smoke']);view=QWebEngineView();view.resize(1440,1000)
profile=QWebEngineProfile(view);errors=[]


class Page(QWebEnginePage):
    def javaScriptConsoleMessage(self,level,message,line,source):
        if level==self.JavaScriptConsoleMessageLevel.ErrorMessageLevel:errors.append(f'{source}:{line}: {message}')


class Harness(QObject):
    state=Window.state;describe=Window.describe
    def __init__(self,folder):
        super().__init__();self.config=Config(folder/'config.json');self.workspace=Workspace()
        seed=default_model();self.mid=seed['id'];self.workspace.put(seed);self.workspace.save(folder/'draft.vtw')
        self.job=None;self.torch_python=None;self.rendering={};self.bridge=Bridge(self,view);self.requests=[]
    def recent_files(self):return []
    def broadcast(self,kind='state',payload=None,**kwargs):self.bridge.send(kind,self.state() if kind=='state' else payload)
    def dispatch(self,req,source=None):
        self.requests.append(copy.deepcopy(req))
        assert req['action'] not in ('train','scienceTrain','scienceNotebookExecute')
        result=Window.dispatch(self,req,source)
        if req['action'] in ('scienceDraftFamilyPage','scienceDraftFamilyPreview'):
            assert len(encoded(result))<=65536
        return result


def js(source):
    loop=QEventLoop();timer=QTimer();timer.setSingleShot(True);timer.timeout.connect(loop.quit);values=[]
    view.page().runJavaScript(source,lambda value:(values.append(value),loop.quit()));timer.start(4000);loop.exec();timer.stop()
    assert values,'JS callback timeout';return values[0]


def until(source):
    deadline=time.monotonic()+20
    while time.monotonic()<deadline:
        if js(source):return
        loop=QEventLoop();QTimer.singleShot(20,loop.quit);loop.exec()
    raise AssertionError(f'UI timeout: {source}; errors: {errors}; last requests: {h.requests[-3:]}')


def click(identifier):js('document.getElementById('+json.dumps(identifier)+').click()')


def change(collection,key,value):
    selector='[data-family-card="'+collection+'"] [data-key="'+key+'"]'
    assert js('(()=>{const n=document.querySelector('+json.dumps(selector)+');if(!n)return false;n.value='+json.dumps(value)+';n.dispatchEvent(new Event("change"));return true;})()'),selector


def add(collection):
    js('(()=>{const s=[...document.querySelectorAll("#science-family-cards details")].find(s=>s.querySelector("summary").textContent==='+
       json.dumps({'symbols':'Scoped integer symbols','assignments':'Derived field assignments','repeats':'Independent finite repeats'}[collection])+
       ');[...s.querySelectorAll("button")].find(b=>b.textContent.startsWith("Add ")).click();})()')
    until('document.querySelectorAll('+json.dumps('[data-family-card="'+collection+'"]')+').length===1')


with tempfile.TemporaryDirectory() as directory:
    h=Harness(Path(directory));page=Page(profile,view);view.setPage(page)
    channel=QWebChannel(page);channel.registerObject('host',h.bridge);page.setWebChannel(channel)
    url=QUrl.fromLocalFile(str(ROOT/'app/web/index.html'));url.setFragment('model='+h.mid);view.setUrl(url);view.show()
    try:
        until('Boolean(window.loom?.document?.view.fitted)')
        click_selector='document.querySelector("[data-panel=science]").click();document.getElementById("experiment-dock").style.height="850px";document.getElementById("experiment-dock").style.width="1080px"'
        js(click_selector);until('!document.getElementById("science-panel").hidden')
        before=h.workspace.db.serialize()
        click('science-author-new-family')
        until('!document.getElementById("science-family-editor").hidden && document.querySelectorAll('+json.dumps('[data-family-card="templates"]')+').length===1 && !document.getElementById("science-family-check").disabled')
        assert not h.workspace.dirty;assert before==h.workspace.db.serialize()
        add('symbols');sid=js('document.querySelector('+json.dumps('[data-family-card="symbols"] [data-key="id"]')+').value')
        change('symbols','default','3')
        add('assignments');change('assignments','node','linear1')
        change('assignments','value',json.dumps({'symbol':sid}))
        click('science-family-preview')
        until('JSON.parse(document.getElementById("science-family-preview-result").textContent||"{}").status==="verified"')
        assert js('JSON.parse(document.getElementById("science-family-preview-result").textContent).parameters')==9
        add('repeats');change('repeats','nodes','["linear1"]');change('repeats','count','2')
        click('science-family-preview')
        until('JSON.parse(document.getElementById("science-family-preview-result").textContent||"{}").parameters===21')
        until('document.querySelectorAll('+json.dumps('[data-family-card="repeats"]')+').length===1 && !document.getElementById("science-family-preview").disabled')
        assert js('getComputedStyle(document.querySelector("#science-family-editor label")).display')=='flex'
        assert js('document.getElementById("science-family-editor").scrollWidth<=document.getElementById("science-family-editor").clientWidth+1')
        if os.environ.get('VTW_FAMILY_EDITOR_CAPTURE'):
            loop=QEventLoop();QTimer.singleShot(150,loop.quit);loop.exec()
            assert view.grab().save(os.environ['VTW_FAMILY_EDITOR_CAPTURE'])
        assert before==h.workspace.db.serialize();assert not h.workspace.dirty
        # Bad text stays visible and cannot publish a silently repaired family.
        change('repeats','count','not JSON');click('science-family-save')
        until('document.getElementById("science-family-status").textContent.includes("invalid")')
        assert not h.workspace.list_specs('family');change('repeats','count','2')
        click('science-family-save')
        until('document.getElementById("science-family-editor").hidden && document.getElementById("science-kind").value==="family" && !document.getElementById("science-family-create").disabled')
        ref=json.loads(js('document.getElementById("science-object").value'))
        family=h.workspace.get_spec('family',ref['id'],ref['revision']);assert len(family['symbols'])==1
        click('science-family-create')
        until('document.getElementById("science-family-instance-status").textContent.includes("21 parameters")')
        assert len(h.workspace.models())==2
        specialization=h.workspace.get_spec('specialization',h.workspace.list_specs('specialization')[0]['id'])
        root=specialization['model']['id'];assert validate_model(h.workspace.models(),root)['parameters']==21
        assert specialization['bindings']=={sid:3}
        h.dispatch({'action':'undo'});assert root not in h.workspace.models()
        assert h.workspace.get_spec('specialization',specialization['id'],specialization['revision'])==specialization
        h.dispatch({'action':'redo'});assert root in h.workspace.models()
        h.workspace.save(Path(directory)/'final.vtw');opened=Workspace.load(Path(directory)/'final.vtw')
        try:assert opened.get_spec('specialization',specialization['id'])==specialization
        finally:opened.db.close()
        assert not errors,errors;assert 'torch' not in sys.modules
        if os.environ.get('VTW_FAMILY_AUTHOR_CAPTURE'):assert view.grab().save(os.environ['VTW_FAMILY_AUTHOR_CAPTURE'])
        print(json.dumps({'ok':True,'typed_family_editor':True,'scoped_width':3,'independent_repeat':2,
                          'parameters':21,'exact_retained_specialization':True,'undo_redo_reload':True,
                          'invalid_text_retained':True,'host_torch':False,'console_errors':errors}))
    finally:
        view.setPage(QWebEnginePage(view));page.deleteLater();channel.deleteLater()
        QCoreApplication.sendPostedEvents(None,QEvent.Type.DeferredDelete);app.processEvents();profile.deleteLater()
        QCoreApplication.sendPostedEvents(None,QEvent.Type.DeferredDelete);app.processEvents();view.close()
        h.workspace.db.close()
        if h.workspace.history.directory:h.workspace.history.directory.cleanup()
