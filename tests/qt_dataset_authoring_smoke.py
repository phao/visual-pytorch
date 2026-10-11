"""Real sandboxed WebEngine typed dataset authoring, history and conflict gates.

Use the GUI Python with PYTHONPATH=.; never touches user data/preferences or
imports Torch. Registry starts empty, rather than loading scientific fixtures.
"""
import copy
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from PyQt6.QtCore import QObject,QEvent,QEventLoop,QTimer,QUrl,QCoreApplication
from PyQt6.QtWidgets import QApplication
from PyQt6.QtWebChannel import QWebChannel
from PyQt6.QtWebEngineCore import QWebEnginePage,QWebEngineProfile
from PyQt6.QtWebEngineWidgets import QWebEngineView
from app.config import Config
from app.core import Workspace,default_model
from app.desktop import Bridge,Window

assert not os.environ.get('QTWEBENGINE_DISABLE_SANDBOX')
assert '--no-sandbox' not in os.environ.get('QTWEBENGINE_CHROMIUM_FLAGS','')
app=QApplication(['dataset-authoring-smoke']);view=QWebEngineView();view.resize(1440,1000)
profile=QWebEngineProfile(view);errors=[]
class Page(QWebEnginePage):
    def javaScriptConsoleMessage(self,level,message,line,source):
        if level==self.JavaScriptConsoleMessageLevel.ErrorMessageLevel:errors.append(f'{source}:{line}: {message}')
class Harness(QObject):
    state=Window.state;describe=Window.describe;dispatch=Window.dispatch
    def __init__(self,folder):
        super().__init__();self.config=Config(folder/'config.json');self.workspace=Workspace()
        self.model=default_model('Author measurements');self.workspace.put(self.model)
        assert self.workspace.list_specs(all_revisions=True)==[]
        self.job=None;self.torch_python=None;self.rendering={};self.bridge=Bridge(self,view)
    def recent_files(self):return []
    def broadcast(self,kind='state',payload=None,**kw):self.bridge.send(kind,self.state() if kind=='state' else payload)
def js(script):
    loop=QEventLoop();timer=QTimer();timer.setSingleShot(True);timer.timeout.connect(loop.quit);values=[]
    view.page().runJavaScript(script,lambda value:(values.append(value),loop.quit()));timer.start(3000);loop.exec();timer.stop()
    assert values,'JavaScript timeout';return values[0]
def until(script,timeout=12):
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        if js(script):return
        loop=QEventLoop();QTimer.singleShot(15,loop.quit);loop.exec()
    raise AssertionError(f'UI timeout: {script}; console: {errors[:5]}')
def click(sid):
    until('!document.getElementById('+json.dumps(sid)+').disabled')
    js('document.getElementById('+json.dumps(sid)+').click()')
    if sid in ('science-author-add-field','science-author-add-record','science-author-fields-prev','science-author-fields-next','science-author-rows-prev','science-author-rows-next'):
        until('!document.getElementById("science-author-check").disabled')
def set_value(sid,value):js('setAuthorValue(document.getElementById('+json.dumps(sid)+'),'+json.dumps(value)+')')
def set_field(fid,label,value):
    assert js('setAuthorValue(authorField('+json.dumps(fid)+','+json.dumps(label)+'),'+json.dumps(value)+')'), (fid,label,js('document.getElementById("science-author-status").textContent'),errors)
    if label=='Dtype':until('!document.getElementById("science-author-check").disabled')
def set_cell(rid,fid,value):
    label=rid+' / '+fid
    js('setAuthorValue([...document.querySelectorAll("#science-author-records textarea")].find(el=>el.getAttribute("aria-label")==='+json.dumps(label)+'),'+json.dumps(json.dumps(value))+')')

with tempfile.TemporaryDirectory() as directory:
    folder=Path(directory);h=Harness(folder);workspaces=[h.workspace]
    page=Page(profile,view);view.setPage(page);channel=QWebChannel(page);channel.registerObject('host',h.bridge);page.setWebChannel(channel)
    url=QUrl.fromLocalFile(str(Path(__file__).resolve().parents[1]/'app/web/index.html'));url.setFragment('model='+h.model['id']);view.setUrl(url);view.show()
    try:
        until('Boolean(window.loom?.document)')
        js('window.setAuthorValue=(control,value)=>{if(!control)throw Error("Missing author control");control.value=value;control.dispatchEvent(new Event("change"));return true;};window.authorField=(fid,label)=>{const card=[...document.querySelectorAll("#science-author-fields fieldset")].find(el=>el.dataset.fieldId===fid);if(!card)throw Error("Missing field card "+fid);const target=[...card.querySelectorAll("label")].find(el=>el.firstChild.textContent===label);if(!target)throw Error("Missing label "+fid+" / "+JSON.stringify(label)+"; labels: "+[...card.querySelectorAll("label")].map(el=>JSON.stringify(el.firstChild.textContent)).join(","));return target.querySelector("input,select,textarea");};')
        js('window.authorCell=(rid,fid)=>[...document.querySelectorAll("#science-author-records textarea")].find(el=>el.getAttribute("aria-label")===rid+" / "+fid);')
        js('document.querySelector("[data-panel=science]").click();document.getElementById("experiment-dock").style.height="750px";')
        click('science-author-new');until('!document.getElementById("science-author").hidden')
        before=h.workspace.db.serialize();assert h.workspace.list_specs(all_revisions=True)==[]
        set_value('science-author-name','Measurements <img src=x onerror=alert(1)>')
        for fid in ['x','y','subject','time','sequence','label','flag']:
            set_value('science-author-field-id',fid);click('science-author-add-field')
        assert js('document.querySelectorAll("#science-author-fields fieldset").length')==1
        click('science-author-fields-prev')
        set_field('x','Sample shape','1');set_field('x','Axis names','F');set_field('y','Sample shape','1')
        set_field('subject','Dtype','string');set_field('time','Dtype','int64')
        set_field('sequence','Sample shape','-1, 2')
        js('authorField("sequence","Nullable ").click()')
        set_field('sequence','unit','"K"')
        set_field('label','Dtype','category');set_field('label','Ordered vocabulary','["one",1,"unused"]')
        for i in range(8):set_value('science-author-record-id','r'+str(i));click('science-author-add-record')
        for i in range(8):
            rid='r'+str(i)
            for fid,value in [('x',[i]),('y',[2*i+1]),('subject','s'+str(i//2)),('time',i),('sequence',[[None,1]]*(i+1)),('label',1 if i%2 else 'one')]:set_cell(rid,fid,value)
        click('science-author-fields-next');set_field('flag','Dtype','bool')
        for i in range(8):set_cell('r'+str(i),'flag',bool(i%2))
        for theme,font,spacing in [('dark',14,1),('github',18,4),('math',28,10)]:
            h.config.values.update(theme=theme,font_size=font,spacing=spacing);h.broadcast()
            until('getComputedStyle(document.body).getPropertyValue("--ui-font-size").trim()==='+json.dumps(str(font)+'px'))
            assert js('parseFloat(getComputedStyle(document.querySelector("#science-author textarea")).fontSize)')>=font-2
            assert js('getComputedStyle(document.querySelector("#science-author textarea")).backgroundColor!=="rgba(0, 0, 0, 0)"')
            assert js('document.getElementById("science-author-records").getBoundingClientRect().width<=document.getElementById("science-panel").getBoundingClientRect().width')
        h.config.values.update(theme='github',font_size=18,spacing=4);h.broadcast()
        if os.environ.get('VTW_SMOKE_SCREENSHOT'):
            click('science-author-fields-prev')
            loop=QEventLoop();QTimer.singleShot(150,loop.quit);loop.exec()
            assert view.grab().save(os.environ['VTW_SMOKE_SCREENSHOT'])
        assert h.workspace.db.serialize()==before
        click('science-author-check');until('document.getElementById("science-author-status").textContent.includes("Valid DatasetSpec")')
        assert h.workspace.db.serialize()==before
        click('science-author-save');until('document.getElementById("science-author").hidden && document.getElementById("science-kind").value==="dataset"')
        rows=h.workspace.list_specs('dataset');assert len(rows)==1;ref={k:rows[0][k] for k in ('kind','id','revision')}
        d=h.workspace.get_spec('dataset',ref['id']);assert d['fields'][4]['shape']==[-1,2];assert d['fields'][4]['unit']=='K'
        assert d['fields'][5]['vocabulary']['values']==['one',1,'unused'];assert d['records'][3]['values']['sequence']==[[None,1]]*4
        assert d['records'][0]['values']['flag'] is False
        assert not js('Boolean(document.querySelector("#science-author img"))')
        click('science-author-edit');until('!document.getElementById("science-author").hidden')
        # Local parse failures retain text across attempted paging; backend shape
        # validation cannot silently coerce or admit a malformed record.
        js('setAuthorValue(authorCell("r0","x"),"1e400")')
        assert js('authorCell("r0","x").value')=='1e400', errors
        click('science-author-fields-next');assert js('document.getElementById("science-author-field-page").textContent.startsWith("Fields 1")')
        set_cell('r0','x',[999]);set_field('x','Sample shape','2')
        before=h.workspace.db.serialize();click('science-author-save')
        until('document.getElementById("science-author-status").textContent.includes("does not match field shape")')
        assert h.workspace.db.serialize()==before;assert not js('document.getElementById("science-author").hidden')
        set_field('x','Sample shape','1');set_value('science-author-name','Edited measurements')
        click('science-author-save');until('document.getElementById("science-author").hidden')
        edited=h.workspace.get_spec('dataset',ref['id']);assert edited['records'][0]['values']['x']==[999]
        assert h.dispatch({'action':'undo'},None);assert h.workspace.get_spec('dataset',ref['id'])['revision']==ref['revision']
        assert h.workspace.get_spec('dataset',ref['id'],edited['revision'])['name']=='Edited measurements'
        assert h.dispatch({'action':'redo'},None);until('document.getElementById("science-object").options.length===2')
        # No-op retention must not add history or change DB bytes.
        click('science-author-edit');until('!document.getElementById("science-author").hidden')
        before=h.workspace.db.serialize();history=copy.deepcopy(h.workspace.history.summary())
        click('science-author-save');until('document.getElementById("science-author").hidden')
        assert h.workspace.db.serialize()==before;assert h.workspace.history.summary()==history
        # Concurrent browsing-head change is rejected, draft name remains.
        click('science-author-edit');until('!document.getElementById("science-author").hidden')
        set_value('science-author-name','Retain this unsaved text')
        other=copy.deepcopy(edited);other.pop('revision');other['name']='External concurrent edit';h.workspace.put_spec('dataset',other)
        before=h.workspace.db.serialize();click('science-author-save')
        until('document.getElementById("science-author-status").textContent.includes("head changed")')
        assert h.workspace.db.serialize()==before;assert js('document.getElementById("science-author-name").value')=='Retain this unsaved text'
        # Busy guards run before malformed scientific edit payloads; draft stays.
        h.job=SimpleNamespace(isRunning=lambda:False,awaiting_terminal=True,mid=None,snapshot={'run_id':'busy'});h.broadcast()
        until('document.getElementById("science-author-save").disabled')
        try:h.dispatch({'action':'scienceAuthorSave'},None)
        except ValueError as error:assert 'Stop' in str(error)
        else:raise AssertionError('Draft save bypassed pending terminal guard')
        h.job=None;h.broadcast();until('!document.getElementById("science-author-save").disabled')
        # Workspace switch disables retained draft, no insertion into either DB.
        original=h.workspace;h.workspace=Workspace();workspaces.append(h.workspace);h.workspace.put(h.model);h.broadcast()
        until('document.getElementById("science-author-status").textContent.includes("Workspace changed")')
        assert js('document.getElementById("science-author-save").disabled');assert h.workspace.list_specs(all_revisions=True)==[]
        assert original.db.serialize()==before
        click('science-author-cancel');assert js('document.getElementById("science-author").hidden')
        # Save/reload is exact and independent of GUI state.
        original.save(folder/'dataset.vtw');loaded=Workspace.load(folder/'dataset.vtw')
        try:assert len(loaded.list_specs('dataset',True))==3
        finally:loaded.db.close()
        assert not errors,errors;assert 'torch' not in sys.modules
        print(json.dumps({'typed_creation_empty_registry':'PASS','rich_shapes_categories_missing':'PASS','readonly_validation':'PASS','invalid_text_retention':'PASS','atomic_invalid_shape':'PASS','history_noop':'PASS','head_workspace_conflicts':'PASS','pending_job_guard':'PASS','save_reload':'PASS','console_errors':errors,'host_torch':False}))
    finally:
        for workspace in workspaces:
            if workspace.history.directory:workspace.history.directory.cleanup()
            workspace.db.close()
        view.close();view.setPage(QWebEnginePage(view));page.deleteLater();QCoreApplication.sendPostedEvents(None,QEvent.Type.DeferredDelete)
        profile.deleteLater();QCoreApplication.sendPostedEvents(None,QEvent.Type.DeferredDelete)
