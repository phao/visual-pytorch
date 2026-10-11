"""Actual bounded task form, large ordered pipelines and linked staging; no fits."""
import copy,json,os,sys,tempfile,time
from pathlib import Path
from PyQt6.QtCore import QObject,QCoreApplication,QEvent,QEventLoop,QTimer,QUrl
from PyQt6.QtWidgets import QApplication
from PyQt6.QtWebChannel import QWebChannel
from PyQt6.QtWebEngineCore import QWebEnginePage,QWebEngineProfile
from PyQt6.QtWebEngineWidgets import QWebEngineView
from app.config import Config
from app.core import Workspace
from app.desktop import Bridge,Window
from app.science_browser import encoded
from test_task_authoring import multi_fixture

assert not os.environ.get('QTWEBENGINE_DISABLE_SANDBOX')
assert '--no-sandbox' not in os.environ.get('QTWEBENGINE_CHROMIUM_FLAGS','')
app=QApplication(['task-session-smoke']);view=QWebEngineView();view.resize(1440,1050)
profile=QWebEngineProfile(view);errors=[]
class Page(QWebEnginePage):
    def javaScriptConsoleMessage(self,level,message,line,source):
        if level==self.JavaScriptConsoleMessageLevel.ErrorMessageLevel:errors.append(f'{source}:{line}: {message}')
class Harness(QObject):
    state=Window.state;describe=Window.describe
    def __init__(self,folder):
        super().__init__();self.workspace=Workspace();self.config=Config(folder/'config.json')
        self.job=None;self.torch_python=None;self.rendering={};self.actions=[];self.projections=[];self.fail=None
        self.model,d,t=multi_fixture();self.workspace.put(self.model);self.workspace.freeze_model(self.model['id']);self.workspace.put_spec('dataset',d)
        self.original={'description':{'edition':'old','long':'λ😀'*10000},**{k:v for k,v in t.items() if k!='description'}}
        self.original['inputs']=[{'port':'p'+str(i),'field':'a','transforms':[]} for i in range(80)]
        self.original['inputs'][0]['transforms']=[{'kind':'affine','scale':1,'offset':i} for i in range(1000)]
        self.original['heads']=[{'id':'h'+str(i),'port':'o'+str(i),'field':'y','loss':'mse','weight':1,'mask':'valid','transforms':[]} for i in range(20)]
        self.old=self.workspace.put_spec('task',self.original);head=copy.deepcopy(self.original);head['description']['edition']='new'
        self.head=self.workspace.put_spec('task',head);self.bridge=Bridge(self,view)
    def recent_files(self):return []
    def broadcast(self,kind='state',payload=None,**kw):self.bridge.send(kind,self.state() if kind=='state' else payload)
    def dispatch(self,req,source=None):
        self.actions.append(copy.deepcopy(req))
        if req['action']==self.fail:self.fail=None;raise ValueError('Fixture: transient task page failure')
        value=Window.dispatch(self,req,source)
        if req['action'] in ('scienceDraftTaskPage','scienceDraftTaskTransforms'):
            assert len(encoded(value))<=65536 and len(value['rows'])<=6;self.projections.append(value)
        return value
def js(script):
    loop=QEventLoop();timer=QTimer();timer.setSingleShot(True);timer.timeout.connect(loop.quit);values=[]
    view.page().runJavaScript(script,lambda value:(values.append(value),loop.quit()));timer.start(5000);loop.exec();timer.stop();assert values,'JS timeout';return values[0]
def until(condition):
    end=time.monotonic()+30
    while time.monotonic()<end:
        if condition() if callable(condition) else js(condition):return
        loop=QEventLoop();QTimer.singleShot(20,loop.quit);loop.exec()
    raise AssertionError(str(condition)+'; '+str(errors)+'; '+str(js('["science-author-status","science-value-status"].map(id=>document.getElementById(id)?.textContent)')))
def ready():until('!document.getElementById("science-author-check").disabled')
def click(sid):until('!document.getElementById('+json.dumps(sid)+').disabled');js('document.getElementById('+json.dumps(sid)+').click()')
def named(name,within='#science-task-bindings'):
    return '[...document.querySelectorAll('+json.dumps(within+' button')+')].find(n=>n.textContent==='+json.dumps(name)+')'
def button(name,within='#science-task-bindings'):
    expression=named(name,within);until('!'+expression+'.disabled');js(expression+'.click()');ready()
def field(label,within='[data-task-input="0"]'):
    return '[...document.querySelectorAll('+json.dumps(within+' label')+')].find(n=>n.firstChild.textContent==='+json.dumps(label)+').querySelector("input,textarea")'
def set_field(expression,value,event='change'):
    js('(()=>{const n='+expression+';n.value='+json.dumps(value)+';n.dispatchEvent(new Event('+json.dumps(event)+'));})()')

with tempfile.TemporaryDirectory() as directory:
    folder=Path(directory);h=Harness(folder);page=Page(profile,view);view.setPage(page)
    channel=QWebChannel(page);channel.registerObject('host',h.bridge);page.setWebChannel(channel)
    url=QUrl.fromLocalFile(str(Path(__file__).resolve().parents[1]/'app/web/index.html'));url.setFragment('model='+h.model['id']);view.setUrl(url);view.show()
    try:
        until('Boolean(window.loom?.document)');until(lambda:h.workspace.models()[h.model['id']]['view'].get('fitted') is True)
        before=h.workspace.db.serialize();steps=len(h.workspace.history.undo_stack)
        js('document.querySelector("[data-panel=science]").click();Object.assign(document.getElementById("experiment-dock").style,{width:"1000px",height:"850px",top:"100px",left:"240px",bottom:"auto"});document.getElementById("science-kind").value="task";document.getElementById("science-kind").dispatchEvent(new Event("change"))')
        until('document.getElementById("science-object").options.length===2')
        js('document.getElementById("science-object").value=JSON.stringify('+json.dumps(h.old)+');document.getElementById("science-object").dispatchEvent(new Event("change"))')
        h.fail='scienceDraftTaskPage';click('science-author-edit');until('!document.getElementById("science-author-retry-page").hidden')
        assert js('document.getElementById("science-author-save").disabled')
        click('science-author-retry-page');ready();assert len(h.authoring_sessions._sessions)==1
        assert js('document.querySelectorAll("[data-task-input]").length')==6
        assert js('document.querySelectorAll("[data-task-head]").length')==6
        assert js('document.querySelectorAll("[data-task-input=\\"0\\"] [data-transform-index]").length')==6
        for _ in range(3):button('Transforms ›','[data-task-input="0"]')
        assert js('document.querySelector("[data-task-input=\\"0\\"] [data-transform-index]").dataset.transformIndex')=='18'
        h.fail='scienceDraftTaskTransforms';button('‹ Transforms','[data-task-input="0"]')
        assert js('!'+named('Retry transforms','[data-task-input="0"]')+'.hidden')
        button('‹ Transforms','[data-task-input="0"]');assert js('document.querySelector("[data-task-input=\\"0\\"] [data-transform-index]").dataset.transformIndex')=='12'
        for _ in range(2):button('‹ Transforms','[data-task-input="0"]')
        assert js(named('‹ Transforms','[data-task-input="0"]')+'.disabled')
        # Capture a focused literal edit before paging; do not round via Number.
        scale=field('scale','[data-task-input="0"] [data-transform-index="0"]')
        set_field(scale,'9007199254740993','input');js(scale+'.focus()');button('Bindings ›')
        native=next(iter(h.authoring_sessions._sessions.values()));assert native['draft']['document']['inputs'][0]['transforms'][0]['scale']==9007199254740993
        assert js('document.querySelector("[data-task-input]").dataset.taskInput')=='6'
        button('‹ Bindings');assert js(scale+'.value')=='9007199254740993'
        # Optional property removal must not shift a later queued weight target.
        set_field(field('Mask field ID (optional)','[data-task-head="0"]'),'')
        set_field(field('Objective weight','[data-task-head="0"]'),'9007199254740993')
        set_field('document.getElementById("science-author-name")','Bounded task editing')
        description=field('Task description','#science-task-bindings');assert js(description+'.dataset.large')=='true'
        js(description+'.parentElement.querySelector("button").click()');until('!document.getElementById("science-value-finish").disabled')
        native=next(iter(h.authoring_sessions._sessions.values()));assert 'mask' not in native['draft']['document']['heads'][0]
        assert native['draft']['document']['heads'][0]['weight']==9007199254740993
        assert native['draft']['document']['name']=='Bounded task editing' and len(h.authoring_sessions._sessions)==1
        assert js('document.getElementById("science-author-save").disabled && document.getElementById("science-value-save").disabled')
        text=js('document.getElementById("science-value-text").value');assert '"edition":"old"' in text
        fragment=text.replace('"edition":"old"','"edition":"fix"',1);assert len(fragment)<=2000
        set_field('document.getElementById("science-value-text")',fragment,'input')
        click('science-value-apply');until('!document.getElementById("science-value-finish").disabled');click('science-value-finish')
        until('document.getElementById("science-value-editor").hidden');ready();assert h.workspace.db.serialize()==before
        js(description+'.parentElement.querySelector("button").click()');until('!document.getElementById("science-value-finish").disabled')
        h.fail='scienceDraftTaskPage';click('science-value-discard');until('document.getElementById("science-value-editor").hidden')
        until('!document.getElementById("science-author-retry-page").hidden');click('science-author-retry-page');ready()
        assert len(h.authoring_sessions._sessions)==1 and h.workspace.db.serialize()==before
        for theme,font in [('dark',14),('github',18),('math',28)]:
            h.config.values.update(theme=theme,font_size=font);h.broadcast()
            until('getComputedStyle(document.body).getPropertyValue("--ui-font-size").trim()==='+json.dumps(str(font)+'px'))
            assert js('parseFloat(getComputedStyle('+field('Input port ID')+').fontSize)')>=font-2
            assert js('document.querySelector("[data-task-input]").getBoundingClientRect().width<=document.getElementById("science-panel").getBoundingClientRect().width')
        if os.environ.get('VTW_SMOKE_SCREENSHOT'):
            h.config.values.update(theme='github',font_size=18);h.broadcast();until('getComputedStyle(document.body).getPropertyValue("--ui-font-size").trim()==="18px"')
            js('document.querySelector("[data-task-input]").scrollIntoView({block:"start"});document.getElementById("toast-close").click()')
            loop=QEventLoop();QTimer.singleShot(350,loop.quit);loop.exec();assert view.grab().save(os.environ['VTW_SMOKE_SCREENSHOT'])
        click('science-author-check');until('document.getElementById("science-author-status").textContent.includes("Valid TaskSpec")')
        assert h.workspace.db.serialize()==before;click('science-author-save');until('document.getElementById("science-author").hidden')
        expected=copy.deepcopy(h.original);expected['name']='Bounded task editing';expected['description']['edition']='fix'
        expected['inputs'][0]['transforms'][0]['scale']=9007199254740993;expected['heads'][0]['weight']=9007199254740993;del expected['heads'][0]['mask']
        actual=h.workspace.get_spec('task',h.old['id']);assert {k:v for k,v in actual.items() if k!='revision'}==expected
        assert h.workspace.get_spec('task',h.old['id'],h.old['revision'])['description']['edition']=='old'
        assert len(h.workspace.history.undo_stack)==steps+1
        h.dispatch({'action':'undo'});assert h.workspace.get_spec('task',h.old['id'])['revision']==h.head['revision']
        h.dispatch({'action':'redo'});assert h.workspace.get_spec('task',h.old['id'])==actual
        h.workspace.save(folder/'tasks.vtw');loaded=Workspace.load(folder/'tasks.vtw')
        try:assert loaded.get_spec('task',h.old['id'])==actual
        finally:loaded.db.close()
        assert not any(r['action'] in ('scienceAuthorBegin','scienceAuthorCheck','scienceAuthorSave','scienceIndex','scienceTrain') for r in h.actions)
        assert sum(r['action']=='scienceDraftOpen' for r in h.actions)==1
        assert not errors and 'torch' not in sys.modules and 'nbclient' not in sys.modules,errors
        print(json.dumps({'actual_bounded_task':'PASS','bindings':80,'pipeline':1000,'historical_base':'PASS','literal_precision':'PASS',
            'named_optional_edits':'PASS','child_finish_discard':'PASS','read_retry':'PASS','single_history_reload':'PASS','fonts':'PASS',
            'projections':len(h.projections),'max_bytes':max(len(encoded(v)) for v in h.projections),'host_torch':False,'console_errors':errors}))
    finally:
        h.authoring_sessions.close_owner(h.bridge) if hasattr(h,'authoring_sessions') else None
        if h.workspace.history.directory:h.workspace.history.directory.cleanup()
        h.workspace.db.close();view.close();view.setPage(QWebEnginePage(view));page.deleteLater();QCoreApplication.sendPostedEvents(None,QEvent.Type.DeferredDelete)
        profile.deleteLater();QCoreApplication.sendPostedEvents(None,QEvent.Type.DeferredDelete)
