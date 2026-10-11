"""Actual bounded experiment form, independent axes/local pins and shared staging.

Large reusable selection fixture, not billion-fit execution or host memory proof.
All numerical dispatches are forbidden; sandbox and private files remain in use.
"""
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
from test_experiment_authoring import fixture

assert not os.environ.get('QTWEBENGINE_DISABLE_SANDBOX')
assert '--no-sandbox' not in os.environ.get('QTWEBENGINE_CHROMIUM_FLAGS','')
app=QApplication(['experiment-session-smoke']);view=QWebEngineView();view.resize(1480,1080)
profile=QWebEngineProfile(view);errors=[]
class Page(QWebEnginePage):
    def javaScriptConsoleMessage(self,level,message,line,source):
        if level==self.JavaScriptConsoleMessageLevel.ErrorMessageLevel:errors.append(f'{source}:{line}: {message}')
class Harness(QObject):
    state=Window.state;describe=Window.describe
    def __init__(self,folder):
        super().__init__();self.workspace=Workspace();self.config=Config(folder/'config.json');self.job=None;self.torch_python=None
        self.rendering={};self.actions=[];self.projections=[];self.fail=None
        e=fixture(self.workspace);pair=e['data_tasks'];protocol=e['protocols'][0]
        e['question']='old '+('λ😀'*9998)
        e['data_tasks']=[{**copy.deepcopy(pair[i%2]),'id':'p'+str(i).zfill(3)} for i in range(80)]
        e['splits']=[{'id':'s'+str(i).zfill(3),'protocol':{'method':'row','seed':i,'fractions':[.5,.25,.25]}} for i in range(80)]
        e['protocols']=[{**copy.deepcopy(protocol),'id':'adam-'+str(i).zfill(3)} for i in range(80)]
        e['initialization_seeds']=list(range(1000))
        e['selection']={'kind':'paired','pairs':[{'model':e['models'][i%2],'data_task':'p'+str(i).zfill(3)} for i in range(80)],
            'exclusions':[{'model':e['models'][i%2],'data_task':'p'+str(i).zfill(3),'reason':'old '+('λ'*5000)} for i in range(80)]}
        self.original=e;self.old=self.workspace.put_spec('experiment',e);head=copy.deepcopy(e);head['question']='new '+('λ😀'*9998)
        self.head=self.workspace.put_spec('experiment',head);self.model=e['models'][0]['id'];self.bridge=Bridge(self,view)
    def recent_files(self):return []
    def broadcast(self,kind='state',payload=None,**kw):self.bridge.send(kind,self.state() if kind=='state' else payload)
    def dispatch(self,req,source=None):
        self.actions.append(copy.deepcopy(req));action=req['action']
        assert action not in ('scienceTrain','scienceNotebookExecute'), 'Fixture forbids execution'
        if (action,req.get('collection'))==self.fail:self.fail=None;raise ValueError('Fixture: transient experiment page failure')
        value=Window.dispatch(self,req,source)
        if action in ('scienceDraftExperimentPage','scienceDraftExperimentChoices','scienceDraftExperimentValue','scienceDraftExperimentPreview'):
            assert len(encoded(value))<=65536
            if action=='scienceDraftExperimentPage':assert len(value['rows'])<=6
            if action=='scienceDraftExperimentChoices':assert len(value['rows'])<=25
            self.projections.append(value)
        return value
def js(script):
    loop=QEventLoop();timer=QTimer();timer.setSingleShot(True);timer.timeout.connect(loop.quit);values=[]
    view.page().runJavaScript(script,lambda value:(values.append(value),loop.quit()));timer.start(5000);loop.exec();timer.stop();assert values,'JS timeout';return values[0]
def until(condition):
    end=time.monotonic()+40
    while time.monotonic()<end:
        if condition() if callable(condition) else js(condition):return
        loop=QEventLoop();QTimer.singleShot(20,loop.quit);loop.exec()
    raise AssertionError(str(condition)+'; '+str(errors)+'; '+str(js('["science-author-status","science-value-status"].map(id=>document.getElementById(id)?.textContent)')))
def ready():until('!document.getElementById("science-author-check").disabled')
def click(sid):until('!document.getElementById('+json.dumps(sid)+').disabled');js('document.getElementById('+json.dumps(sid)+').click()')
def named(name,within):return '[...document.querySelectorAll('+json.dumps(within+' button')+')].find(n=>n.textContent==='+json.dumps(name)+')'
def button(name,within):
    n=named(name,within);until('!'+n+'.disabled');js(n+'.click()');ready()
def field(label,within):return '[...document.querySelectorAll('+json.dumps(within+' label')+')].find(n=>n.firstChild.textContent==='+json.dumps(label)+').querySelector("input,textarea,select")'
def set_field(expression,value,event='change'):
    js('(()=>{const n='+expression+';n.value='+json.dumps(value)+';n.dispatchEvent(new Event('+json.dumps(event)+'));})()')
def section(collection):return '[data-experiment-collection="'+collection+'"]'

with tempfile.TemporaryDirectory() as directory:
    folder=Path(directory);h=Harness(folder);page=Page(profile,view);view.setPage(page)
    channel=QWebChannel(page);channel.registerObject('host',h.bridge);page.setWebChannel(channel)
    url=QUrl.fromLocalFile(str(Path(__file__).resolve().parents[1]/'app/web/index.html'));url.setFragment('model='+h.model);view.setUrl(url);view.show()
    try:
        until('Boolean(window.loom?.document)');until(lambda:h.workspace.models()[h.model]['view'].get('fitted') is True)
        before=h.workspace.db.serialize();steps=len(h.workspace.history.undo_stack)
        js('document.querySelector("[data-panel=science]").click();Object.assign(document.getElementById("experiment-dock").style,{width:"1040px",height:"890px",top:"90px",left:"240px",bottom:"auto"});document.getElementById("science-kind").value="experiment";document.getElementById("science-kind").dispatchEvent(new Event("change"))')
        until('document.getElementById("science-object").options.length===2')
        js('document.getElementById("science-object").value=JSON.stringify('+json.dumps(h.old)+');document.getElementById("science-object").dispatchEvent(new Event("change"))')
        h.fail=('scienceDraftExperimentPage','models');click('science-author-edit');until('!document.getElementById("science-author-retry-page").hidden && !document.getElementById("science-author-cancel").disabled')
        assert js('document.getElementById("science-author-save").disabled');click('science-author-retry-page');ready();assert len(h.authoring_sessions._sessions)==1
        for collection in ('data_tasks','splits','initialization_seeds','protocols','pairs','exclusions'):
            assert js('document.querySelectorAll('+json.dumps(section(collection)+' [data-experiment-axis]')+').length')==6
        for _ in range(3):button('Entries ›',section('initialization_seeds'))
        assert js('document.querySelector('+json.dumps(section('initialization_seeds')+' [data-experiment-axis]')+').dataset.axisIndex')=='18'
        assert js('document.querySelector('+json.dumps(section('data_tasks')+' [data-experiment-axis]')+').dataset.axisIndex')=='0'
        h.fail=('scienceDraftExperimentPage','initialization_seeds');js(named('‹ Entries',section('initialization_seeds'))+'.click()')
        until('document.getElementById("science-author-status").textContent.includes("transient experiment page failure") && !'+named('‹ Entries',section('initialization_seeds'))+'.disabled')
        assert js('!'+named('‹ Entries',section('initialization_seeds'))+'.disabled');button('‹ Entries',section('initialization_seeds'))
        assert js('document.querySelector('+json.dumps(section('initialization_seeds')+' [data-experiment-axis]')+').dataset.axisIndex')=='12'
        # A local pairing search pages independently and pins the out-of-page value.
        selector='[data-experiment-axis="pairs"][data-axis-index="0"] [data-local-choice="data_tasks"]'
        js('document.querySelector('+json.dumps(selector+' input')+').value="p079";document.querySelector('+json.dumps(selector+' [data-choice-action=search]')+').click()');ready()
        until('document.querySelector('+json.dumps(selector)+').textContent.includes("of 1")')
        selected=field('Dataset/task pairing ID','[data-experiment-axis="pairs"][data-axis-index="0"]')
        assert js(selected+'.value')=='p000';assert js(selected+'.options.length')==3
        # Literal large integers are not rounded by Number on their way to native.
        updates=field('Maximum updates','[data-experiment-axis="protocol"][data-axis-index="0"]')
        set_field(updates,'9007199254740993','input');js(updates+'.focus()');button('Entries ›',section('protocols'))
        native=next(iter(h.authoring_sessions._sessions.values()));assert native['draft']['document']['protocols'][0]['protocol']['budget']['updates']==9007199254740993
        button('‹ Entries',section('protocols'));assert js(updates+'.value')=='9007199254740993';set_field(updates,'3')
        set_field('document.getElementById("science-author-name")','Bounded experiment editing')
        question=field('Research question','#science-experiment-bindings');assert js(question+'.dataset.large')=='true'
        js(question+'.parentElement.querySelector("button").click()');until('!document.getElementById("science-value-finish").disabled')
        native=next(iter(h.authoring_sessions._sessions.values()));assert native['draft']['document']['name']=='Bounded experiment editing'
        assert native['draft']['document']['protocols'][0]['protocol']['budget']['updates']==3
        assert js('document.getElementById("science-author-save").disabled && document.getElementById("science-value-save").disabled')
        text=js('document.getElementById("science-value-text").value');assert text.startswith('"old ')
        fragment=text.replace('"old ','"fix ',1);assert len(fragment)<=2000;set_field('document.getElementById("science-value-text")',fragment,'input')
        click('science-value-apply');until('!document.getElementById("science-value-finish").disabled');click('science-value-finish')
        until('document.getElementById("science-value-editor").hidden');ready();assert h.workspace.db.serialize()==before
        js(question+'.parentElement.querySelector("button").click()');until('!document.getElementById("science-value-finish").disabled')
        h.fail=('scienceDraftExperimentPage','protocols');click('science-value-discard');until('document.getElementById("science-value-editor").hidden')
        until('!document.getElementById("science-author-retry-page").hidden');assert js('document.getElementById("science-author-save").disabled')
        click('science-author-retry-page');ready();assert len(h.authoring_sessions._sessions)==1
        click('science-experiment-preview');until('document.querySelectorAll("[data-draft-candidate]").length===25')
        assert js('document.getElementById("science-experiment-preview-status").textContent.includes("of 1024000000")')
        assert js('document.querySelectorAll("#science-experiment-preview-rows button").length')==0
        assert h.workspace.db.serialize()==before and h.workspace.list_specs('run',True)==[]
        for theme,font in [('dark',14),('github',18),('math',28)]:
            h.config.values.update(theme=theme,font_size=font);h.broadcast()
            until('getComputedStyle(document.body).getPropertyValue("--ui-font-size").trim()==='+json.dumps(str(font)+'px'))
            assert js('parseFloat(getComputedStyle('+updates+').fontSize)')>=font-2
            assert js('document.querySelector("[data-experiment-axis=protocol]").getBoundingClientRect().width<=document.getElementById("science-panel").getBoundingClientRect().width')
        if os.environ.get('VTW_SMOKE_SCREENSHOT'):
            h.config.values.update(theme='github',font_size=18);h.broadcast();until('getComputedStyle(document.body).getPropertyValue("--ui-font-size").trim()==="18px"')
            js('document.querySelector("[data-experiment-axis=protocol]").scrollIntoView({block:"start"});document.getElementById("toast-close").click()')
            loop=QEventLoop();QTimer.singleShot(350,loop.quit);loop.exec();assert view.grab().save(os.environ['VTW_SMOKE_SCREENSHOT'])
        click('science-author-check');until('document.getElementById("science-author-status").textContent.includes("Valid ExperimentSpec")')
        assert h.workspace.db.serialize()==before;click('science-author-save');until('document.getElementById("science-author").hidden')
        expected=copy.deepcopy(h.original);expected['name']='Bounded experiment editing';expected['question']='fix '+('λ😀'*9998)
        actual=h.workspace.get_spec('experiment',h.old['id']);assert {k:v for k,v in actual.items() if k!='revision'}==expected
        assert h.workspace.get_spec('experiment',h.old['id'],h.old['revision'])['question'].startswith('old ')
        assert len(h.workspace.history.undo_stack)==steps+1
        h.dispatch({'action':'undo'});assert h.workspace.get_spec('experiment',h.old['id'])['revision']==h.head['revision']
        h.dispatch({'action':'redo'});assert h.workspace.get_spec('experiment',h.old['id'])==actual
        h.workspace.save(folder/'experiments.vtw');loaded=Workspace.load(folder/'experiments.vtw')
        try:assert loaded.get_spec('experiment',h.old['id'])==actual
        finally:loaded.db.close()
        assert not any(r['action'] in ('scienceAuthorBegin','scienceAuthorCheck','scienceAuthorSave','scienceAuthorPreview','scienceIndex') for r in h.actions)
        assert all('draft' not in r for r in h.actions);assert sum(r['action']=='scienceDraftOpen' for r in h.actions)==1
        assert not errors and 'torch' not in sys.modules and 'nbclient' not in sys.modules,errors
        print(json.dumps({'actual_bounded_experiment':'PASS','axis_entries':80,'seeds':1000,'historical_base':'PASS','literal_precision':'PASS','local_outside_pin':'PASS',
            'child_finish_discard':'PASS','read_retry':'PASS','preview_coordinates':1024000000,'single_history_reload':'PASS','fonts':'PASS',
            'projections':len(h.projections),'max_bytes':max(len(encoded(v)) for v in h.projections),'host_torch':False,'console_errors':errors}))
    finally:
        h.authoring_sessions.close_owner(h.bridge) if hasattr(h,'authoring_sessions') else None
        if h.workspace.history.directory:h.workspace.history.directory.cleanup()
        h.workspace.db.close();view.close();view.setPage(QWebEnginePage(view));page.deleteLater();QCoreApplication.sendPostedEvents(None,QEvent.Type.DeferredDelete)
        profile.deleteLater();QCoreApplication.sendPostedEvents(None,QEvent.Type.DeferredDelete)
