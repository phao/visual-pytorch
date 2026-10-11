"""Actual mixed parameter layout, history, sessions and read-only camera behavior."""
import copy,json,os,sys,tempfile,time
from pathlib import Path
from PyQt6.QtCore import QObject,QEvent,QEventLoop,QTimer,QUrl,QCoreApplication,Qt,QPoint
from PyQt6.QtTest import QTest
from PyQt6.QtGui import QMouseEvent
from PyQt6.QtWidgets import QApplication
from PyQt6.QtWebChannel import QWebChannel
from PyQt6.QtWebEngineCore import QWebEnginePage,QWebEngineProfile
from PyQt6.QtWebEngineWidgets import QWebEngineView
from app.config import Config
from app.core import Workspace,default_mlp,generate_code
from app.desktop import Bridge,Window
from app.evidence import identity

assert not os.environ.get('QTWEBENGINE_DISABLE_SANDBOX')
assert '--no-sandbox' not in os.environ.get('QTWEBENGINE_CHROMIUM_FLAGS','')
app=QApplication(['parameter-layout-smoke']);view=QWebEngineView();view.resize(1440,980)
profile=QWebEngineProfile(view);errors=[]
class Page(QWebEnginePage):
    def javaScriptConsoleMessage(self,level,message,line,source):
        if level==self.JavaScriptConsoleMessageLevel.ErrorMessageLevel:errors.append(f'{source}:{line}: {message}')
class Harness(QObject):
    state=Window.state;describe=Window.describe;dispatch=Window.dispatch
    def __init__(self,folder):
        super().__init__();self.config=Config(folder/'config.json');self.workspace=Workspace()
        self.model=default_mlp(inputs=2,outputs=1,hidden=[3,5]);self.model['view']={'x':9999,'y':-9999,'zoom':.2,'fitted':True}
        self.workspace.put(self.model);self.workspace.save(folder/'model.vtw');self.job=None;self.torch_python=None;self.rendering={};self.bridge=Bridge(self,view)
    def recent_files(self):return []
    def broadcast(self,kind='state',payload=None,**kw):self.bridge.send(kind,self.state() if kind=='state' else payload)
def js(script):
    loop=QEventLoop();timer=QTimer();timer.setSingleShot(True);timer.timeout.connect(loop.quit);values=[]
    view.page().runJavaScript(script,lambda value:(values.append(value),loop.quit()));timer.start(3000);loop.exec();timer.stop()
    assert values,'JavaScript timeout';return values[0]
def until(script,timeout=15):
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        if js(script):return
        QTest.qWait(20)
    raise AssertionError(f'UI timeout: {script}; console: {errors}')
def pan(dx,dy):
    r=js('(() => {const r=document.getElementById("canvas").getBoundingClientRect();return {x:r.right-80,y:r.bottom-80};})()')
    start=QPoint(round(r['x']),round(r['y']));finish=start+QPoint(dx,dy);widget=view.focusProxy() or view
    QTest.mousePress(widget,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier,start)
    app.sendEvent(widget,QMouseEvent(QEvent.Type.MouseMove,finish.toPointF(),widget.mapToGlobal(finish).toPointF(),Qt.MouseButton.NoButton,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier))
    QTest.qWait(30);QTest.mouseRelease(widget,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier,finish);QTest.qWait(80)

with tempfile.TemporaryDirectory() as directory:
    folder=Path(directory);h=Harness(folder);mid=h.model['id'];page=Page(profile,view);view.setPage(page)
    channel=QWebChannel(page);channel.registerObject('host',h.bridge);page.setWebChannel(channel)
    url=QUrl.fromLocalFile(str(Path(__file__).resolve().parents[1]/'app/web/index.html'));url.setFragment('model='+mid);view.setUrl(url);view.show()
    try:
        until('Boolean(window.loom?.document.view.fitted)')
        assert js('window.loom.document.view.x')!=9999 and not h.workspace.dirty
        raw=h.workspace.db.serialize();disk=Path(h.workspace.path).read_bytes();history=copy.deepcopy(h.workspace.history.summary())
        first=js('window.loom.document.view');pan(-67,-43);after=js('window.loom.document.view')
        assert after['x']!=first['x'] and after['y']!=first['y']
        js('document.getElementById("zoom-in").click()');QTest.qWait(60)
        assert h.workspace.db.serialize()==raw and Path(h.workspace.path).read_bytes()==disk and not h.workspace.dirty
        assert h.workspace.history.summary()==history and js('!window.loom.state.dirty')
        # Other acknowledgements do not snap an actively used viewport back to disk.
        camera=js('window.loom.document.view');h.broadcast();QTest.qWait(50)
        assert js('window.loom.document.view')==camera
        js('window.loom.fit()');QTest.qWait(50);assert h.workspace.db.serialize()==raw and not h.workspace.dirty
        # Native count read is exact, guarded and presentation-only.
        graph=js('window.loom.document.graph');state=h.state()
        req={'action':'layoutParameterCounts','id':mid,'graph':graph,'workspace_id':state['workspace_id'],'panel_scope':state['panel_scope']}
        result=Window.dispatch(h,req,h.bridge);assert [result['counts'][n['id']] for n in graph['nodes'] if n['type']=='Linear']==['9','20','6']
        assert h.workspace.db.serialize()==raw and h.workspace.history.summary()==history
        for bad in (dict(req,workspace_id='wrong'),dict(req,panel_scope='wrong'),dict(req,extra=True),dict(req,id='missing')):
            try:Window.dispatch(h,bad,h.bridge)
            except ValueError:pass
            else:raise AssertionError('Stale/malformed count request admitted')
            assert h.workspace.db.serialize()==raw
        before=js('window.loom.document.graph');before_text=js('JSON.stringify(window.loom.document.graph)');signature=identity(h.workspace.models(),mid);code=generate_code(h.workspace.models(),mid);steps=len(h.workspace.history.undo_stack)
        js('window.loom.perform("layout");document.getElementById("layout-expanded-section").open=true;document.getElementById("layout-expanded-parameter_sizing").checked=true;document.getElementById("expand-all").click()')
        until('window.loom.document.graph.nodes.filter(n=>n.type!=="Linear").every(n=>n.collapsed) && window.loom.document.graph.nodes.filter(n=>n.type==="Linear").every(n=>!n.collapsed&&n.height)')
        until('window.loom.state.history.undo');QTest.qWait(100)
        layout=js('window.loom.document.graph');layout_text=js('JSON.stringify(window.loom.document.graph)');assert len(h.workspace.history.undo_stack)==steps+1
        sizes=js('window.loom.document.graph.nodes.filter(n=>n.type==="Linear").map(n=>({id:n.id,height:n.height,min:LoomGraph.minimumSize(n).height,width:n.width??null}))')
        by={n['id']:n for n in sizes};assert by['readout']['height']==by['readout']['min']
        assert by['linear2']['height']==4*by['linear2']['min']
        assert by['readout']['height']<by['linear1']['height']<by['linear2']['height']
        assert all(n['width'] is None for n in sizes)
        assert identity(h.workspace.models(),mid)==signature and generate_code(h.workspace.models(),mid)==code
        assert h.workspace.models()[mid]['view']=={'x':0,'y':0,'zoom':1}
        js('window.loom.perform("undo")');until('JSON.stringify(window.loom.document.graph)==='+json.dumps(before_text))
        js('window.loom.perform("redo")');until('JSON.stringify(window.loom.document.graph)==='+json.dumps(layout_text))
        js('document.getElementById("save-layout-preferences").click()');until('window.loom.state.layout_preferences.expanded.parameter_sizing===true')
        h.workspace.save(h.workspace.path);loaded=Workspace.load(h.workspace.path)
        try:assert loaded.layout_preferences['expanded']['parameter_sizing'];assert loaded.models()[mid]['graph']==layout
        finally:loaded.db.close()
        # Ordinary expand remains all-expanded and no special sizing by default.
        js('document.getElementById("layout-expanded-parameter_sizing").checked=false;document.getElementById("expand-all").click()')
        until('window.loom.document.graph.nodes.every(n=>!n.collapsed)');QTest.qWait(100)
        # Opening again ignores any previous camera, without rewriting disk.
        h.workspace.save(h.workspace.path);baseline=h.workspace.db.serialize();pan(-50,-30);js('window.loom.call("openModel",{id:window.loom.document.id})')
        until('Boolean(window.loom.document.view.fitted)');QTest.qWait(100)
        assert h.workspace.db.serialize()==baseline and not h.workspace.dirty
        if os.environ.get('VTW_LAYOUT_SCREENSHOT'):
            js('document.getElementById("layout-expanded-parameter_sizing").checked=true;document.getElementById("expand-all").click()')
            until('window.loom.document.graph.nodes.filter(n=>n.type!=="Linear").every(n=>n.collapsed)');QTest.qWait(150)
            assert view.grab().save(os.environ['VTW_LAYOUT_SCREENSHOT'])
        assert not errors,errors;assert 'torch' not in sys.modules
        print(json.dumps({'real_parameter_layout_mixed_folding_relative_heights':'PASS','single_history_and_identity_code_preserved':'PASS',
                         'workspace_option_save_reload':'PASS','pan_zoom_fit_no_dirty_sql_disk_history':'PASS','legacy_camera_ignored':'PASS',
                         'native_exact_guarded_count_read':'PASS','console_errors':errors,'host_torch':False}))
    finally:
        if h.workspace.history.directory:h.workspace.history.directory.cleanup()
        h.workspace.db.close();view.close();view.setPage(QWebEnginePage(view));page.deleteLater();QCoreApplication.sendPostedEvents(None,QEvent.Type.DeferredDelete)
        profile.deleteLater();QCoreApplication.sendPostedEvents(None,QEvent.Type.DeferredDelete)
