"""Actual large typed dataset pages and linked raw editor, no user state/Torch."""
import copy
import json
import os
import sys
import tempfile
import time
from pathlib import Path

from PyQt6.QtCore import QObject,QCoreApplication,QEvent,QEventLoop,QTimer,QUrl
from PyQt6.QtWidgets import QApplication
from PyQt6.QtWebChannel import QWebChannel
from PyQt6.QtWebEngineCore import QWebEnginePage,QWebEngineProfile
from PyQt6.QtWebEngineWidgets import QWebEngineView
from app.config import Config
from app.core import Workspace,default_model,GraphError
from app.desktop import Bridge,Window
from app.science_browser import encoded
from test_authoring import measurements

assert not os.environ.get('QTWEBENGINE_DISABLE_SANDBOX')
assert '--no-sandbox' not in os.environ.get('QTWEBENGINE_CHROMIUM_FLAGS','')
app=QApplication(['dataset-session-smoke']);view=QWebEngineView();view.resize(1440,1050)
profile=QWebEngineProfile(view);errors=[]
class Page(QWebEnginePage):
    def javaScriptConsoleMessage(self,level,message,line,source):
        if level==self.JavaScriptConsoleMessageLevel.ErrorMessageLevel:errors.append(f'{source}:{line}: {message}')
class Harness(QObject):
    state=Window.state;describe=Window.describe
    def __init__(self,folder):
        super().__init__();self.config=Config(folder/'config.json');self.workspace=Workspace()
        self.model=default_model('Typed dataset session fixture');self.workspace.put(self.model)
        self.job=None;self.torch_python=None;self.rendering={};self.bridge=Bridge(self,view);self.actions=[];self.pages=[];self.fail_pages=0
        self.original=measurements();d=self.original;d['fields'][0]['shape']=[512]
        d['fields'] += [{'id':'counter','dtype':'int64','shape':[]},{'id':'flag','dtype':'bool','shape':[]}]+[
            {'id':'extra'+str(i),'dtype':'float32','shape':[1]} for i in range(3)]
        d['records']=[{'id':'row'+str(i),'values':{'x':[i+j/8. for j in range(512)],'y':[float(i)],
            'counter':9007199254740993,'flag':bool(i%2),**{'extra'+str(j):[float(i+j)] for j in range(3)}}} for i in range(200)]
        d['usage']={'edition':'old','long':'λ😀'*10000};self.old=self.workspace.put_spec('dataset',d)
        newer=copy.deepcopy(d);newer['usage']['edition']='new';self.head=self.workspace.put_spec('dataset',newer)
    def recent_files(self):return []
    def broadcast(self,kind='state',payload=None,**kw):self.bridge.send(kind,self.state() if kind=='state' else payload)
    def dispatch(self,request,source=None):
        self.actions.append(request['action'])
        if request['action']=='scienceDraftDatasetPage' and self.fail_pages:
            self.fail_pages-=1;raise GraphError('Fixture: transient page read failure')
        value=Window.dispatch(self,request,source)
        if request['action'].startswith('scienceDraft'):
            assert len(encoded(value))<=65536
            self.pages.append({'action':request['action'],'bytes':len(encoded(value))})
        return value
def js(script):
    loop=QEventLoop();timer=QTimer();timer.setSingleShot(True);timer.timeout.connect(loop.quit);values=[]
    view.page().runJavaScript(script,lambda value:(values.append(value),loop.quit()));timer.start(5000);loop.exec();timer.stop()
    assert values,'JavaScript timeout';return values[0]
def until(condition,timeout=20):
    end=time.monotonic()+timeout
    while time.monotonic()<end:
        if condition() if callable(condition) else js(condition):return
        loop=QEventLoop();QTimer.singleShot(15,loop.quit);loop.exec()
    raise AssertionError(f'UI timeout: {condition}; errors {errors}; status '+str(js('[document.getElementById("science-author-status").textContent,document.getElementById("science-value-status").textContent]')))
def click(sid):
    until('!document.getElementById('+json.dumps(sid)+').disabled');js('document.getElementById('+json.dumps(sid)+').click()')
def ready():until('!document.getElementById("science-author-check").disabled')
def cell(rid,fid):return 'document.querySelector("#science-author-records textarea[aria-label=\\"'+rid+' / '+fid+'\\"]")'
def set_control(expression,value,event='change'):
    js('(()=>{const n='+expression+';if(!n)throw Error("Missing control");n.value='+json.dumps(value)+';n.dispatchEvent(new Event('+json.dumps(event)+'));})()')

with tempfile.TemporaryDirectory() as directory:
    folder=Path(directory);h=Harness(folder);page=Page(profile,view);view.setPage(page)
    channel=QWebChannel(page);channel.registerObject('host',h.bridge);page.setWebChannel(channel)
    url=QUrl.fromLocalFile(str(Path(__file__).resolve().parents[1]/'app/web/index.html'));url.setFragment('model='+h.model['id']);view.setUrl(url);view.show()
    try:
        until('Boolean(window.loom?.document?.view.fitted)')  # Camera is session-only.
        before=h.workspace.db.serialize();steps=len(h.workspace.history.undo_stack)
        js('document.querySelector("[data-panel=science]").click();document.getElementById("experiment-dock").style.height="850px";document.getElementById("experiment-dock").style.width="1000px";')
        js('document.getElementById("science-kind").value="dataset";document.getElementById("science-kind").dispatchEvent(new Event("change"))')
        until('document.getElementById("science-object").options.length===2')
        js('document.getElementById("science-object").value=JSON.stringify('+json.dumps(h.old)+');document.getElementById("science-object").dispatchEvent(new Event("change"))')
        h.fail_pages=1;click('science-author-edit')
        until('!document.getElementById("science-author-retry-page").hidden')
        assert js('!document.getElementById("science-author").hidden && document.getElementById("science-author-save").disabled && document.getElementById("science-author-add-field").disabled')
        assert len(h.authoring_sessions._sessions)==1 and h.workspace.db.serialize()==before
        click('science-author-retry-page');ready();assert js('document.querySelectorAll("#science-author-fields fieldset").length')<=6
        assert js('document.querySelectorAll("#science-author-records tbody tr").length')<=20
        assert js(cell('row0','counter')+'.value')=='9007199254740993'
        assert js(cell('row0','x')+'.dataset.large')=='true';assert js(cell('row0','x')+'.value')==''
        set_control('document.getElementById("science-author-name")','Keep this typed name')
        set_control(cell('row0','counter'),'9007199254740995')
        # Open the tensor's raw editor; queued name/int64 edits must flush into
        # the same parent session before resolving the current stable target.
        js(cell('row0','x')+'.parentElement.querySelector("button").click()')
        until('!document.getElementById("science-value-finish").disabled')
        assert len(h.authoring_sessions._sessions)==1
        native=next(iter(h.authoring_sessions._sessions.values()))
        assert native['draft']['document']['name']=='Keep this typed name'
        assert native['draft']['document']['records'][0]['values']['counter']==9007199254740995
        assert js('document.getElementById("science-value-save").disabled') and js('document.getElementById("science-author-save").disabled')
        first=js('document.getElementById("science-value-text").value');assert first.startswith('[0.0,0.125,')
        set_control('document.getElementById("science-value-text")',first.replace('0.0','9.0',1),'input')
        click('science-value-apply');until('!document.getElementById("science-value-finish").disabled')
        click('science-value-finish');until('document.getElementById("science-value-editor").hidden');ready()
        assert len(h.authoring_sessions._sessions)==1 and h.workspace.db.serialize()==before
        assert js('document.getElementById("science-author-name").value')=='Keep this typed name'
        # Explicit child discard releases just its buffer, not the parent draft.
        js(cell('row0','x')+'.parentElement.querySelector("button").click()');until('!document.getElementById("science-value-finish").disabled')
        h.fail_pages=1;click('science-value-discard');until('document.getElementById("science-value-editor").hidden')
        until('!document.getElementById("science-author-retry-page").hidden')
        assert js('document.getElementById("science-author-name").value')=='Keep this typed name'
        click('science-author-retry-page');ready()
        assert len(h.authoring_sessions._sessions)==1
        click('science-author-fields-next');ready();assert js('document.getElementById("science-author-field-page").textContent').startswith('Fields 7')
        click('science-author-fields-prev');ready();click('science-author-rows-next');ready()
        assert js('document.getElementById("science-author-row-page").textContent').startswith('Observations 21')
        click('science-author-rows-prev');ready()
        for theme,font,spacing in [('dark',14,1),('github',18,4),('math',28,10)]:
            h.config.values.update(theme=theme,font_size=font,spacing=spacing);h.broadcast()
            until('getComputedStyle(document.body).getPropertyValue("--ui-font-size").trim()==='+json.dumps(str(font)+'px'))
            assert js('parseFloat(getComputedStyle('+cell('row0','counter')+').fontSize)')>=font-2
            assert js('document.getElementById("science-author-records").getBoundingClientRect().width<=document.getElementById("science-panel").getBoundingClientRect().width')
            assert js('(()=>{const n=document.querySelector("#science-author-records thead th");return n.getBoundingClientRect().height<=3*parseFloat(getComputedStyle(n).fontSize)})()'), 'Observation heading squeezed vertically'
            assert js('(()=>{const n=document.querySelector("#science-author-records tbody th"),s=getComputedStyle(n),c=document.createElement("canvas").getContext("2d");c.font=s.fontWeight+" "+s.fontSize+" "+s.fontFamily;return n.getBoundingClientRect().width>=12*c.measureText("0").width-0.5})()'), 'Stable IDs need the specified 12ch column'
        if os.environ.get('VTW_SMOKE_SCREENSHOT'):
            h.config.values.update(theme='github',font_size=18,spacing=4);h.broadcast()
            until('getComputedStyle(document.body).getPropertyValue("--ui-font-size").trim()==="18px"')
            js('document.getElementById("science-author-records").scrollIntoView({block:"start"});document.getElementById("toast-close").click()')
            loop=QEventLoop();QTimer.singleShot(350,loop.quit);loop.exec();assert view.grab().save(os.environ['VTW_SMOKE_SCREENSHOT'])
        click('science-author-check');until('document.getElementById("science-author-status").textContent.includes("Valid DatasetSpec")')
        assert h.workspace.db.serialize()==before;click('science-author-save');until('document.getElementById("science-author").hidden')
        expected=copy.deepcopy(h.original);expected['name']='Keep this typed name';expected['records'][0]['values']['counter']=9007199254740995;expected['records'][0]['values']['x'][0]=9.
        actual=h.workspace.get_spec('dataset',h.old['id']);assert {k:v for k,v in actual.items() if k!='revision'}==expected
        assert h.workspace.get_spec('dataset',h.old['id'],h.old['revision'])['records']==h.original['records']
        assert len(h.workspace.history.undo_stack)==steps+1
        h.dispatch({'action':'undo'});assert h.workspace.get_spec('dataset',h.old['id'])['revision']==h.head['revision']
        h.dispatch({'action':'redo'});assert h.workspace.get_spec('dataset',h.old['id'])==actual
        h.workspace.save(folder/'typed.vtw');loaded=Workspace.load(folder/'typed.vtw')
        try:assert loaded.get_spec('dataset',h.old['id'])==actual
        finally:loaded.db.close()
        assert not any(a in h.actions for a in ('scienceAuthorBegin','scienceAuthorCheck','scienceAuthorSave','scienceIndex','scienceShow','scienceTrain'))
        assert h.actions.count('scienceDraftOpen')==1 and h.actions.count('scienceDraftClose')==0
        assert 'torch' not in sys.modules and 'nbclient' not in sys.modules and not errors,errors
        print(json.dumps({'typed_bounded_pages':'PASS','linked_shared_session':'PASS','typed_int64_precision':'PASS','large_tensor_staging':'PASS',
            'hidden_values_historical_metadata':'PASS','child_discard_retains_parent':'PASS','page_failure_retry':'PASS','single_history_reload':'PASS','font28':'PASS',
            'projections':len(h.pages),'max_bytes':max(row['bytes'] for row in h.pages),'console_errors':errors,'host_torch':False}))
    finally:
        if hasattr(h,'authoring_sessions'):h.authoring_sessions.close_owner(h.bridge)
        if h.workspace.history.directory:h.workspace.history.directory.cleanup()
        h.workspace.db.close();view.close();view.setPage(QWebEnginePage(view));page.deleteLater();QCoreApplication.sendPostedEvents(None,QEvent.Type.DeferredDelete)
        profile.deleteLater();QCoreApplication.sendPostedEvents(None,QEvent.Type.DeferredDelete)
