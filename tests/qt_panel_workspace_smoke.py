"""Actual sandboxed Chromium preset save/reload and default/inverted snap gestures."""
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
from app.core import Workspace,default_model
from app.desktop import Bridge,Window

assert not os.environ.get('QTWEBENGINE_DISABLE_SANDBOX')
assert '--no-sandbox' not in os.environ.get('QTWEBENGINE_CHROMIUM_FLAGS','')
app=QApplication(['panel-workspace-smoke']);errors=[];views=[]
class Page(QWebEnginePage):
    def javaScriptConsoleMessage(self,level,message,line,source):
        if level==self.JavaScriptConsoleMessageLevel.ErrorMessageLevel:errors.append(f'{source}:{line}: {message}')
class Harness(QObject):
    state=Window.state;describe=Window.describe;dispatch=Window.dispatch
    def __init__(self,folder):
        super().__init__();self.config=Config(folder/'config.json');self.config.save(ui_zoom=1)
        self.workspace=Workspace();self.model=default_model();self.model['view']['fitted']=True
        self.workspace.put(self.model);self.workspace.dirty=False
        self.job=None;self.torch_python=None;self.rendering={};self.bridges=[]
    def recent_files(self):return []
    def open_tab(self,mid,*,clear_menus=False):
        assert mid==self.model['id'];return open_view(self,clear_menus=clear_menus,wait=False)
    def broadcast(self,kind='state',payload=None,**kw):
        for bridge in self.bridges:bridge.send(kind,self.state() if kind=='state' else payload)
def js(view,script):
    loop=QEventLoop();timer=QTimer();timer.setSingleShot(True);timer.timeout.connect(loop.quit);values=[]
    view.page().runJavaScript(script,lambda value:(values.append(value),loop.quit()));timer.start(3000);loop.exec();timer.stop()
    assert values,'JavaScript timeout';return values[0]
def until(view,script,timeout=15):
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        if js(view,script):return
        QTest.qWait(20)
    raise AssertionError(f'UI timeout: {script}; console: {errors}')
def open_view(h,clear_menus=False,wait=True):
    view=QWebEngineView();view.resize(1440,980);profile=QWebEngineProfile(view);page=Page(profile,view);view.setPage(page)
    bridge=Bridge(h,view);h.bridges.append(bridge);channel=QWebChannel(page);channel.registerObject('host',bridge);page.setWebChannel(channel)
    url=QUrl.fromLocalFile(str(Path(__file__).resolve().parents[1]/'app/web/index.html'))
    if clear_menus:url.setQuery('menus=hidden')
    url.setFragment('model='+h.model['id']);view.setUrl(url);view.show()
    views.append((view,page,profile,channel))
    if wait:until(view,'Boolean(window.loom?.document)')
    return view
def drag(view,selector,dx,dy,ctrl=False):
    point=js(view,'(() => {const r=document.querySelector('+json.dumps(selector)+').getBoundingClientRect();return {x:r.left+r.width/2,y:r.top+r.height/2};})()')
    start=QPoint(round(point['x']),round(point['y']));finish=start+QPoint(dx,dy);widget=view.focusProxy() or view
    mod=Qt.KeyboardModifier.ControlModifier if ctrl else Qt.KeyboardModifier.NoModifier
    if ctrl:QTest.keyPress(widget,Qt.Key.Key_Control)
    QTest.mousePress(widget,Qt.MouseButton.LeftButton,mod,start)
    app.sendEvent(widget,QMouseEvent(QEvent.Type.MouseMove,finish.toPointF(),widget.mapToGlobal(finish).toPointF(),Qt.MouseButton.NoButton,Qt.MouseButton.LeftButton,mod))
    QTest.qWait(30);QTest.mouseRelease(widget,Qt.MouseButton.LeftButton,mod,finish)
    if ctrl:QTest.keyRelease(widget,Qt.Key.Key_Control)
    QTest.qWait(80)
def rect(view,id):
    return js(view,'(() => {const p=document.getElementById('+json.dumps(id)+'),r=p.getBoundingClientRect(),b=document.getElementById("editor").getBoundingClientRect();return {x:r.left-b.left,y:r.top-b.top,width:r.width,height:r.height};})()')
def aligned(value):return abs(value/24-round(value/24))<.0001

with tempfile.TemporaryDirectory() as directory:
    folder=Path(directory);h=Harness(folder);view=open_view(h);config_before=h.config.path.read_bytes()
    try:
        assert js(view,'!document.getElementById("find")')
        # Real mouse events, not calls to a coordinate helper.
        for enabled in (False,True):
            h.config.values['grid_snap']=enabled;h.broadcast()
            until(view,'document.documentElement.dataset.gridSnap==='+json.dumps(str(enabled).lower()))
            for ctrl in (False,True):
                js(view,'window.loom.perform("reset-panels");document.querySelector("[data-open-panel=library-panel]").click()')
                p0=rect(view,'library-panel');drag(view,'#library-panel [data-drag-handle]',53,67,ctrl);p=rect(view,'library-panel')
                if enabled!=ctrl:assert aligned(p['x']) and aligned(p['y']),(enabled,ctrl,p)
                else:assert abs(p['x']-p0['x']-53)<1 and abs(p['y']-p0['y']-67)<1,(p0,p)
                drag(view,'#library-panel .panel-resize',17,13,ctrl);r=rect(view,'library-panel')
                if enabled!=ctrl:assert aligned(r['x']+r['width']) and aligned(r['y']+r['height']),r
                else:assert abs(r['width']-p['width']-17)<1 and abs(r['height']-p['height']-13)<1,(p,r)
                js(view,'window.loom.perform("reset-panels");window.loom.fit()');QTest.qWait(100)
                before=js(view,'window.loom.document.graph.nodes.find(n=>n.type==="Linear")')
                drag(view,'.node.Linear .node-title',53,67,ctrl)
                after=js(view,'window.loom.document.graph.nodes.find(n=>n.type==="Linear")')
                if enabled!=ctrl:assert aligned(after['x']) and aligned(after['y']),after
                else:assert after['x']!=before['x'] and after['y']!=before['y'],(before,after)
                drag(view,'.node.Linear .node-resize',17,13,ctrl)
                resized=js(view,'window.loom.document.graph.nodes.find(n=>n.type==="Linear")')
                if enabled!=ctrl:assert aligned(resized['x']+resized['width']) and aligned(resized['y']+resized['height']),resized
        # Wait for acknowledged graph edits before measuring presentation-only work.
        until(view,'!window.loom.state.history.notice');QTest.qWait(100)
        js(view,'window.loom.perform("reset-panels");document.querySelector("[data-open-panel=library-panel]").click()')
        drag(view,'#library-panel [data-drag-handle]',149,97);drag(view,'#library-panel .panel-resize',60,35)
        normal=rect(view,'library-panel');models=copy.deepcopy(h.workspace.models());history=copy.deepcopy(h.workspace.history.summary())
        js(view,'document.querySelector("#library-panel [data-expand-panel]").click();document.querySelector("[data-open-panel=layout-panel]").click();document.getElementById("panel-arrangement-section").open=true;document.getElementById("save-panel-arrangement").click()')
        until(view,'Boolean(window.loom.state.panel_layout)');saved=copy.deepcopy(h.workspace.panel_layout)
        assert saved['panels']['library-panel']['expanded']
        for key in normal:assert abs(saved['panels']['library-panel'][key]-normal[key])<.01,(normal,saved)
        assert h.workspace.models()==models and h.workspace.history.summary()==history
        assert h.config.path.read_bytes()==config_before
        # A second actual webview initializes the saved preset, independent of tab-local moves.
        second=open_view(h);until(second,'document.getElementById("library-panel").classList.contains("expanded")')
        js(second,'document.querySelector("#library-panel [data-expand-panel]").click()')
        restored=rect(second,'library-panel')
        for key in normal:assert abs(restored[key]-normal[key])<.01,(normal,restored)
        path=folder/'saved.vtw';h.workspace.save(path);old=h.workspace;h.workspace=Workspace.load(path);old.db.close();h.broadcast()
        until(view,'window.loom.state.panel_scope==='+json.dumps(h.workspace.panel_scope))
        until(view,'document.getElementById("library-panel").classList.contains("expanded")')
        assert h.workspace.panel_layout==saved
        # Menu-specific opens clear selection and hide the layer, without changing its preset.
        raw=h.workspace.db.serialize()
        js(view,'if(document.getElementById("architectures-panel").hidden)document.querySelector("[data-open-panel=architectures-panel]").click();document.querySelector("#model-list .model-row").click()')
        until(view,'document.getElementById("menu-pack").classList.contains("collapsed") && !document.querySelector(".node.selected")')
        assert js(view,'[...document.querySelectorAll(".floating-panel")].every(p=>p.hidden)')
        assert h.workspace.panel_layout==saved and h.workspace.db.serialize()==raw
        # New-tab opens use the same behavior even when the saved preset has expanded menus.
        js(second,'document.querySelector("[data-open-panel=architectures-panel]").click();document.querySelector("#model-list .tab-hint").click()')
        until(second,'document.getElementById("menu-pack").classList.contains("collapsed")')
        assert len(views)==3;third=views[-1][0]
        until(third,'document.getElementById("menu-pack")?.classList.contains("collapsed") && !document.querySelector(".node.selected")')
        assert h.workspace.panel_layout==saved and h.workspace.db.serialize()==raw
        # Font/viewport adaptation keeps the stored preset immutable and a recoverable slice.
        samples=0
        for font in (14,18,28):
            for width,height in ((1440,980),(900,700)):
                h.config.values['font_size']=font;h.broadcast();second.resize(width,height);QTest.qWait(100)
                js(second,'document.getElementById("restore-panel-arrangement").click();document.querySelector("#library-panel [data-expand-panel]").click()')
                r=rect(second,'library-panel');b=js(second,'({width:document.getElementById("editor").clientWidth,height:document.getElementById("editor").clientHeight})')
                assert r['x']+r['width']>=24 and r['y']+r['height']>=24 and r['x']<=b['width']-24 and r['y']<=b['height']-24,(r,b)
                assert h.workspace.panel_layout==saved;samples+=1
        assert not errors,errors;assert 'torch' not in sys.modules
        print(json.dumps({'real_mouse_snap_defaults_ctrl_inverse':'PASS','real_panel_save_reload_second_tab':'PASS','architecture_menu_same_tab_new_tab_hide':'PASS',
          'ordinary_expanded_geometry':'PASS','font_viewport_samples':samples,'global_config_science_history_unchanged':'PASS','console_errors':errors,'host_torch':False}))
    finally:
        h.bridges.clear()
        for v,page,profile,channel in views:
            v.close();v.setPage(QWebEnginePage(v));page.deleteLater();QCoreApplication.sendPostedEvents(None,QEvent.Type.DeferredDelete)
            profile.deleteLater();QCoreApplication.sendPostedEvents(None,QEvent.Type.DeferredDelete)
        if h.workspace.history.directory:h.workspace.history.directory.cleanup()
        h.workspace.db.close()
