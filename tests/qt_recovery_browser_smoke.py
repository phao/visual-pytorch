"""Real bounded recovery browser, no numerical execution or user state."""
import copy,json,os,sys,tempfile,time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from PyQt6.QtCore import QObject,QCoreApplication,QEvent,QEventLoop,QTimer,QUrl
from PyQt6.QtWidgets import QApplication
from PyQt6.QtWebChannel import QWebChannel
from PyQt6.QtWebEngineCore import QWebEnginePage,QWebEngineProfile
from PyQt6.QtWebEngineWidgets import QWebEngineView
from app.config import Config
from app.core import Workspace,default_model
from app.desktop import Bridge,Window
from app.desktop_experiments import retain_recovery
from app.recovery_views import changed,header
from app.science_browser import encoded

assert not os.environ.get('QTWEBENGINE_DISABLE_SANDBOX')
assert '--no-sandbox' not in os.environ.get('QTWEBENGINE_CHROMIUM_FLAGS','')
app=QApplication(['recovery-browser-smoke']);view=QWebEngineView();view.resize(1440,1080)
profile=QWebEngineProfile(view);errors=[]
class Page(QWebEnginePage):
    def javaScriptConsoleMessage(self,level,message,line,source):
        if level==self.JavaScriptConsoleMessageLevel.ErrorMessageLevel:errors.append(f'{source}:{line}: {message}')
class Harness(QObject):
    state=Window.state;describe=Window.describe
    def __init__(self,folder):
        super().__init__();self.workspace=Workspace();self.config=Config(folder/'config.json')
        self.workspace.put(default_model('Recovery inspector'));self.model=next(iter(self.workspace.models()))
        self.job=None;self.torch_python=None;self.rendering={};self.actions=[];self.projections=[];self.fail=None
        self.bridge=Bridge(self,view);self.experiment_recoveries={}
        for i in range(80):
            sid=f'{i:032x}'
            self.experiment_recoveries[sid]={'version':1,'format':'visual-pytorch-experiment-recovery','id':sid,
                'snapshot':{'run_id':'run-'+str(i).zfill(3),'documents':[{'tensor':[list(range(256)) for _ in range(128)],
                    'literal':9007199254740993,'λ😀'*1500:'<script>literal only</script>'+'λ😀'*10000}]},
                'result':None,'error':'Failure '+str(i)+' '+'λ😀'*10000,'exported_to':None}
        changed(self)
    def recent_files(self):return []
    def broadcast(self,kind='state',payload=None,**kw):
        value=self.state() if kind=='state' else payload
        if kind=='scienceRecovery':self.projections.append(value)
        self.bridge.send(kind,value)
    def dispatch(self,req,source=None):
        self.actions.append(copy.deepcopy(req));action=req['action']
        assert action not in ('scienceTrain','scienceExecuteNotebook','scienceRecoveries','scienceRecoveryShow')
        if (action,req.get('offset'))==self.fail:self.fail=None;raise ValueError('Fixture recovery read failure')
        value=Window.dispatch(self,req,source)
        if action in ('scienceRecoveryPage','scienceRecoveryInspect'):
            assert len(encoded(value))<=65536;self.projections.append(value)
            if action=='scienceRecoveryPage':assert len(value['rows'])<=25
        return value
def js(script):
    loop=QEventLoop();timer=QTimer();timer.setSingleShot(True);timer.timeout.connect(loop.quit);values=[]
    view.page().runJavaScript(script,lambda value:(values.append(value),loop.quit()));timer.start(5000);loop.exec();timer.stop();assert values,'JS timeout';return values[0]
def until(condition):
    end=time.monotonic()+35
    while time.monotonic()<end:
        if condition() if callable(condition) else js(condition):return
        loop=QEventLoop();QTimer.singleShot(20,loop.quit);loop.exec()
    raise AssertionError(str(condition)+'; '+str(errors)+'; '+str(js('["science-recovery-page","science-tree-status"].map(id=>document.getElementById(id).textContent)')))
def click(sid):
    until('!document.getElementById('+json.dumps(sid)+').disabled');js('document.getElementById('+json.dumps(sid)+').click()')
def open_member(name):
    expression='[...document.querySelectorAll("#science-tree-rows tbody tr")].find(r=>r.cells[0].textContent==='+json.dumps(name)+')?.querySelector("[data-spec-value]")'
    until('Boolean('+expression+') && !('+expression+').disabled');js('('+expression+').click()')
    until('!document.getElementById("science-tree-root").disabled')

with tempfile.TemporaryDirectory() as directory:
    folder=Path(directory);h=Harness(folder);page=Page(profile,view);view.setPage(page)
    channel=QWebChannel(page);channel.registerObject('host',h.bridge);page.setWebChannel(channel)
    url=QUrl.fromLocalFile(str(Path(__file__).resolve().parents[1]/'app/web/index.html'));url.setFragment('model='+h.model);view.setUrl(url);view.show()
    try:
        until('Boolean(window.loom?.document?.view.fitted)')  # Camera is session-only.
        before=h.workspace.db.serialize();history=len(h.workspace.history.undo_stack);evidence=copy.deepcopy(h.experiment_recoveries)
        js('document.querySelector("[data-panel=science]").click();Object.assign(document.getElementById("experiment-dock").style,{width:"1000px",height:"880px",top:"90px",left:"200px",bottom:"auto"});')
        until('document.getElementById("science-recovery-object").options.length===25 && !document.getElementById("science-recovery-next").disabled')
        assert js('Array.isArray(window.loom.state.experiment_recoveries)') is False
        click('science-recovery-next');until('document.getElementById("science-recovery-page").textContent.startsWith("25–50")')
        h.fail=('scienceRecoveryPage',0);click('science-recovery-prev');until('document.getElementById("science-recovery-page").textContent.includes("Fixture recovery read failure")')
        click('science-recovery-refresh');until('document.getElementById("science-recovery-page").textContent.startsWith("25–50")')
        click('science-recovery-prev');until('document.getElementById("science-recovery-page").textContent.startsWith("0–25")')
        js('document.getElementById("science-recovery-query").value="run-079"');click('science-recovery-search')
        until('document.getElementById("science-recovery-object").options.length===2 && !document.getElementById("science-recovery-show").disabled')
        assert js('document.getElementById("science-recovery-object").value')==f'{0:032x}'
        click('science-recovery-show');until('document.getElementById("science-tree-status").textContent.includes("session recovery evidence")')
        assert js('document.getElementById("science-tree-edit").disabled')
        open_member('snapshot');open_member('documents');open_member('[0]');open_member('literal')
        until('document.querySelector("#science-tree-rows pre")?.textContent==="9007199254740993"')
        click('science-tree-root');open_member('snapshot');open_member('documents');open_member('[0]');open_member('tensor')
        click('science-tree-next');until('document.getElementById("science-tree-status").textContent.includes("25–50")')
        h.fail=('scienceRecoveryInspect',0);click('science-tree-prev');until('document.getElementById("science-tree-status").textContent.includes("Fixture recovery read failure")')
        click('science-tree-retry');until('document.getElementById("science-tree-status").textContent.includes("25–50")')
        click('science-tree-prev');until('document.getElementById("science-tree-status").textContent.includes("0–25")')
        assert h.workspace.db.serialize()==before and len(h.workspace.history.undo_stack)==history
        assert h.experiment_recoveries==evidence
        destination=folder/'recovery.json'
        with patch('app.desktop_experiments.QFileDialog.getSaveFileName',return_value=(str(destination),'')):
            click('science-recovery-export');until(lambda:destination.is_file())
            until('document.getElementById("toast-message").textContent.includes("Recovery evidence exported")')
        assert json.loads(destination.read_text())==evidence[f'{0:032x}']
        assert h.experiment_recoveries[f'{0:032x}']['exported_to']==str(destination)
        assert h.workspace.db.serialize()==before and len(h.workspace.history.undo_stack)==history
        assert header(h)['sequence']==2
        job=SimpleNamespace(snapshot={'run_id':'another','documents':[{'hidden':'λ😀'*10000}]})
        retained=retain_recovery(h,job,None,'another diagnostic '+'λ😀'*10000);h.broadcast()
        until('window.loom.state.experiment_recoveries.total===81')
        assert retained['snapshot']==job.snapshot
        for font in (14,18,28):
            h.config.save(font_size=font,theme='github',ui_zoom=1.15,spacing=1);h.broadcast()
            until('window.loom.state.display.font_size==='+str(font))
            assert js('document.getElementById("science-recovery-object").options.length')<=26
        if os.environ.get('VTW_SMOKE_SCREENSHOT'):
            h.config.save(font_size=18);h.broadcast();until('window.loom.state.display.font_size===18')
            click('science-recovery-show');until('document.getElementById("science-tree-status").textContent.includes("session recovery evidence")')
            js('document.getElementById("toast-close").click();document.getElementById("science-tree-rows").scrollIntoView({block:"end"})')
            loop=QEventLoop();QTimer.singleShot(250,loop.quit);loop.exec();assert view.grab().save(os.environ['VTW_SMOKE_SCREENSHOT'])
        assert not errors,errors;assert 'torch' not in sys.modules
        print(json.dumps({'actual_recovery_browser':'PASS','large_hidden_evidence':'PASS','exact_integer':'PASS',
            'outside_pin':'PASS','failed_back_retry':'PASS','read_only_sql_history':'PASS','exact_full_export':'PASS',
            'small_notifications':'PASS','fonts':'PASS','projections':len(h.projections),'max_bytes':max(map(lambda v:len(encoded(v)),h.projections)),
            'host_torch':False,'console_errors':errors}))
    finally:
        if h.workspace.history.directory:h.workspace.history.directory.cleanup()
        h.workspace.db.close()
        page.deleteLater();view.deleteLater();QCoreApplication.sendPostedEvents(None,QEvent.Type.DeferredDelete);QCoreApplication.processEvents()
