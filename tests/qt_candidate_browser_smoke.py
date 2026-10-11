"""Actual stored candidate UI: byte-shortened pages, historical pins, no fits."""
import copy,json,os,random,sys,tempfile,time
from pathlib import Path
from unittest.mock import patch
from PyQt6.QtCore import QObject,QCoreApplication,QEvent,QEventLoop,QTimer,QUrl
from PyQt6.QtWidgets import QApplication
from PyQt6.QtWebChannel import QWebChannel
from PyQt6.QtWebEngineCore import QWebEnginePage,QWebEngineProfile
from PyQt6.QtWebEngineWidgets import QWebEngineView
from app.config import Config
from app.core import Workspace,default_model
from app.desktop import Bridge,Window
from app import candidate_views,science_browser
from app.authoring import workspace_id
from app.experiment_objects import preview_experiment
from test_experiment_authoring import fixture

assert not os.environ.get('QTWEBENGINE_DISABLE_SANDBOX')
assert '--no-sandbox' not in os.environ.get('QTWEBENGINE_CHROMIUM_FLAGS','')
app=QApplication(['candidate-browser-smoke']);view=QWebEngineView();view.resize(1440,1080)
profile=QWebEngineProfile(view);errors=[]
class Page(QWebEnginePage):
    def javaScriptConsoleMessage(self,level,message,line,source):
        if level==self.JavaScriptConsoleMessageLevel.ErrorMessageLevel:errors.append(f'{source}:{line}: {message}')
class Harness(QObject):
    state=Window.state;describe=Window.describe
    def __init__(self,folder):
        super().__init__();self.workspace=Workspace();self.config=Config(folder/'config.json');e=fixture(self.workspace)
        for pair in e['data_tasks']:
            d=self.workspace.get_spec('dataset',pair['dataset']['id']);d.pop('revision');old=d['records'];d['records']=[]
            for i in range(200):
                r=copy.deepcopy(old[i%len(old)]);r['id']=f'row-{i:03d}-'+'x'*140;r['values'].update(subject='s'+str(i//4),time=i);d['records'].append(r)
            pair['dataset']=self.workspace.put_spec('dataset',d)
        bad=default_model('Negative width',inputs=2);self.workspace.put(bad);e['models'].append(self.workspace.freeze_model(bad['id']))
        e['splits'] += [{'id':'group','protocol':{'method':'group','field':'subject','seed':5,'fractions':[.5,.25,.25]}},
            {'id':'temporal','protocol':{'method':'temporal','field':'time','fractions':[.5,.25,.25]}},
            {'id':'external','protocol':{'method':'external','roles':{'train':[r['id'] for r in d['records'][:100]],
                'validation':[r['id'] for r in d['records'][100:150]],'test':[r['id'] for r in d['records'][150:]]},'excluded':[]}}]
        p=copy.deepcopy(e['protocols'][0]);p['id']='second-adam';e['protocols'].append(p)
        e['selection']['exclusions']=[{'model':e['models'][0],'data_task':'pair-1','reason':'Planned negative control '+'λ😀'*300}]
        self.ref=self.workspace.put_spec('experiment',e);self.original=copy.deepcopy(e)
        newer=copy.deepcopy(e);newer['name']='A newer experiment head';newer['initialization_seeds']=[99];self.workspace.put_spec('experiment',newer)
        self.model=e['models'][0]['id'];self.job=None;self.torch_python=None;self.rendering={};self.actions=[];self.projections=[];self.fail=None
        self.bridge=Bridge(self,view)
    def recent_files(self):return []
    def broadcast(self,kind='state',payload=None,**kw):self.bridge.send(kind,self.state() if kind=='state' else payload)
    def dispatch(self,req,source=None):
        self.actions.append(copy.deepcopy(req));action=req['action']
        assert action not in ('scienceTrain','scienceExecuteNotebook','sciencePreview')
        if (action,req.get('offset'))==self.fail:self.fail=None;raise ValueError('Fixture candidate read failure')
        if action=='sciencePreviewPage':
            with patch.object(science_browser,'MAX_BYTES',6000):value=Window.dispatch(self,req,source)
            assert len(science_browser.encoded(value))<=6000 and len(value['candidates'])<=25
            self.projections.append(value);return value
        return Window.dispatch(self,req,source)
def js(script):
    loop=QEventLoop();timer=QTimer();timer.setSingleShot(True);timer.timeout.connect(loop.quit);values=[]
    view.page().runJavaScript(script,lambda v:(values.append(v),loop.quit()));timer.start(5000);loop.exec();timer.stop();assert values,'JS timeout';return values[0]
def until(condition):
    end=time.monotonic()+35
    while time.monotonic()<end:
        if condition() if callable(condition) else js(condition):return
        loop=QEventLoop();QTimer.singleShot(20,loop.quit);loop.exec()
    raise AssertionError(str(condition)+'; '+str(errors)+'; '+str(js('document.getElementById("science-page").textContent')))
def click(sid):
    until('!document.getElementById('+json.dumps(sid)+').disabled');js('document.getElementById('+json.dumps(sid)+').click()')
with tempfile.TemporaryDirectory() as directory:
    folder=Path(directory);h=Harness(folder);page=Page(profile,view);view.setPage(page)
    channel=QWebChannel(page);channel.registerObject('host',h.bridge);page.setWebChannel(channel)
    url=QUrl.fromLocalFile(str(Path(__file__).resolve().parents[1]/'app/web/index.html'));url.setFragment('model='+h.model);view.setUrl(url);view.show()
    try:
        until('Boolean(window.loom?.document?.view.fitted)')  # Camera is session-only.
        before=h.workspace.db.serialize();history=copy.deepcopy(h.workspace.history.summary());rng=random.getstate()
        full=preview_experiment(h.workspace.get_spec('experiment',h.ref['id'],h.ref['revision']),h.workspace.get_spec,limit=100)
        assert full['total_combinations']==96 and len(science_browser.encoded(full))>65536
        req={'action':'sciencePreviewPage','workspace_id':workspace_id(h.workspace),'reference':h.ref}
        for bad in ({**req,'workspace_id':'foreign'},{**req,'unknown':True},{**req,'limit':26},{**req,'offset':True}):
            try:Window.dispatch(h,bad,None)
            except (ValueError,KeyError):pass
            else:raise AssertionError('Closed candidate read should fail')
        with patch('app.experiment_ui.capture',side_effect=AssertionError('Must not capture foreign selection')):
            try:Window.dispatch(h,{'action':'scienceTrain','workspace_id':'foreign','reference':h.ref,'coordinates':{}},None)
            except ValueError:pass
            else:raise AssertionError('Foreign selection must fail before capture')
        js('document.querySelector("[data-panel=science]").click();Object.assign(document.getElementById("experiment-dock").style,{width:"1000px",height:"880px",top:"90px",left:"200px",bottom:"auto"});')
        until('!document.getElementById("science-refresh").disabled')
        js('document.getElementById("science-kind").value="experiment";document.getElementById("science-kind").dispatchEvent(new Event("change"));')
        until('document.getElementById("science-object").options.length===2')
        # Select immutable historical identity explicitly, not its newer head.
        js('document.getElementById("science-object").value='+json.dumps(json.dumps(h.ref,separators=(',',':')))+';document.getElementById("science-object").dispatchEvent(new Event("change"));')
        until('document.querySelectorAll(".science-candidate").length>0 && document.getElementById("science-page").textContent.includes("of 96") && !document.getElementById("science-candidate-refresh").disabled')
        first=h.projections[-1];assert first['experiment']==h.ref and first['next_offset']<25
        seen=list(first['candidates']);next_offset=first['next_offset']
        while next_offset is not None:
            click('science-next');until('document.getElementById("science-page").textContent.startsWith('+json.dumps(str(next_offset)+'–')+') && !document.getElementById("science-candidate-refresh").disabled')
            latest=h.projections[-1];assert latest['experiment']==h.ref;seen.extend(latest['candidates']);next_offset=latest['next_offset']
        assert [r['id'] for r in seen]==[r['id'] for r in full['candidates']]
        assert {r['status'] for r in seen}=={'compatible','incompatible','excluded'}
        previous=h.projections[-2]['offset'];current=h.projections[-1]['offset'];h.fail=('sciencePreviewPage',previous)
        click('science-previous');until('document.getElementById("science-page").textContent.includes("Fixture candidate read failure")')
        click('science-candidate-refresh');until('document.getElementById("science-page").textContent.startsWith('+json.dumps(str(current)+'–')+')')
        click('science-previous');until('document.getElementById("science-page").textContent.startsWith('+json.dumps(str(previous)+'–')+')')
        js('document.getElementById("science-object").dispatchEvent(new Event("change"))')
        until('document.getElementById("science-page").textContent.startsWith("0–") && !document.getElementById("science-candidate-refresh").disabled')
        for font in (14,18,28):
            h.config.save(font_size=font,theme='github',ui_zoom=1.15,spacing=1);h.broadcast();until('window.loom.state.display.font_size==='+str(font))
            assert js('document.querySelectorAll(".science-candidate").length')<=25
            assert js('[...document.querySelectorAll(".science-candidate")].every(c=>c.getBoundingClientRect().width<=document.getElementById("science-panel").getBoundingClientRect().width)')
        if os.environ.get('VTW_SMOKE_SCREENSHOT'):
            h.config.save(font_size=18);h.broadcast();until('window.loom.state.display.font_size===18')
            js('document.getElementById("toast-close").click();document.getElementById("science-selection").scrollIntoView({block:"start"})')
            loop=QEventLoop();QTimer.singleShot(250,loop.quit);loop.exec();assert view.grab().save(os.environ['VTW_SMOKE_SCREENSHOT'])
        assert h.workspace.db.serialize()==before and h.workspace.history.summary()==history and random.getstate()==rng
        assert h.workspace.list_specs('run',True)==[];assert not errors,errors;assert 'torch' not in sys.modules
        print(json.dumps({'actual_candidate_browser':'PASS','exact_historical_selection':'PASS','all_96_statuses':'PASS',
            'failed_back_retry':'PASS','read_only_sql_history_rng':'PASS','fonts':'PASS','complete_preview_bytes':len(science_browser.encoded(full)),
            'projections':len(h.projections),'max_bytes':max(len(science_browser.encoded(p)) for p in h.projections),'host_torch':False,'console_errors':errors}))
    finally:
        if h.workspace.history.directory:h.workspace.history.directory.cleanup()
        h.workspace.db.close();page.deleteLater();view.deleteLater()
        QCoreApplication.sendPostedEvents(None,QEvent.Type.DeferredDelete);QCoreApplication.processEvents()
