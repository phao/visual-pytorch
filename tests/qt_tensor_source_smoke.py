"""Actual sandboxed Qt source palette, sockets, Inspector/history and .vtw."""
import json
import os
from pathlib import Path
import tempfile
import time
from PyQt6.QtCore import QObject,QEvent,QCoreApplication,QEventLoop,QTimer,QUrl,QPoint,Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication
from PyQt6.QtWebChannel import QWebChannel
from PyQt6.QtWebEngineCore import QWebEnginePage,QWebEngineProfile
from PyQt6.QtWebEngineWidgets import QWebEngineView
from app.core import Workspace,validate_model,VERSION
from app.config import Config
from app.desktop import Window,Bridge
from test_tensor_source_graph import source_graph

assert not os.environ.get('QTWEBENGINE_DISABLE_SANDBOX')
assert '--no-sandbox' not in os.environ.get('QTWEBENGINE_CHROMIUM_FLAGS','')
app=QApplication(['source-smoke']);view=QWebEngineView();view.resize(1500,1000)
profile=QWebEngineProfile(view);errors=[]
class Page(QWebEnginePage):
    def javaScriptConsoleMessage(self,level,message,line,source):
        if level==self.JavaScriptConsoleMessageLevel.ErrorMessageLevel:errors.append(f'{source}:{line}:{message}')
class Harness(QObject):
    state=Window.state;describe=Window.describe;dispatch=Window.dispatch
    def __init__(self,folder):
        super().__init__();self.config=Config(folder/'config.json');self.workspace=Workspace()
        doc=source_graph();positions={'input':(40,80),'tensor':(40,240),'join':(300,160),'output':(550,160)}
        for n in doc['graph']['nodes']:n.update(x=positions[n['id']][0],y=positions[n['id']][1])
        self.workspace.put(doc);self.workspace.save(folder/'initial.vtw');self.job=None;self.rendering={};self.torch_python='';self.bridge=Bridge(self,view)
    def recent_files(self):return []
    def broadcast(self,kind='state',payload=None,**kwargs):self.bridge.send(kind,self.state() if kind=='state' else payload)
def js(source):
    values=[];loop=QEventLoop();timer=QTimer();timer.setSingleShot(True);timer.timeout.connect(loop.quit)
    view.page().runJavaScript(source,lambda v:(values.append(v),loop.quit()));deadline=time.monotonic()+20
    while not values and time.monotonic()<deadline:timer.start(2000);loop.exec();timer.stop()
    assert values,'JS callback timeout';return values[0]
def until(source):
    deadline=time.monotonic()+20
    while time.monotonic()<deadline:
        assert not errors,errors
        if js(source):return
        loop=QEventLoop();QTimer.singleShot(25,loop.quit);loop.exec()
    raise AssertionError('UI timeout: '+source)
def inspect(nid):
    js('document.querySelector(".node[data-id='+nid+']").dispatchEvent(new MouseEvent("dblclick",{bubbles:true}))')
def click(selector):
    point=js('(()=>{const e=document.querySelector('+json.dumps(selector)+');e.scrollIntoView({block:"center"});const r=e.getBoundingClientRect();return [r.x+r.width/2,r.y+r.height/2]})()')
    QTest.mouseClick(view.focusProxy() or view,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier,QPoint(round(point[0]),round(point[1])))
    QTest.qWait(70)

with tempfile.TemporaryDirectory() as temporary:
    folder=Path(temporary);h=Harness(folder);page=Page(profile,view);view.setPage(page)
    channel=QWebChannel(page);channel.registerObject('host',h.bridge);page.setWebChannel(channel)
    url=QUrl.fromLocalFile(str(Path(__file__).parents[1]/'app/web/index.html'));url.setFragment('model=sources');view.setUrl(url);view.show()
    try:
        until('Boolean(window.loom?.document?.view.fitted)')
        until('document.getElementById("validation-message").textContent.includes("Valid")')
        assert js('document.querySelectorAll(".node[data-id=tensor] .port-in").length')==0
        assert js('document.querySelectorAll(".node[data-id=tensor] .port-out").length')==1
        inspect('tensor');until('Boolean(document.getElementById("parameter-policy-editor")?.textContent.includes("value"))')
        assert js('document.getElementById("sample-shape").value')=='2'
        assert js('window.LoomBlockHelp.describe("add-tensor-parameter").paragraphs.join(" ").includes("Every batch observation sees the same owner")')  # prose moved to About (R2)
        js('document.activeElement?.blur();document.body.dispatchEvent(new KeyboardEvent("keydown",{key:"b",bubbles:true}))')
        until('!document.getElementById("library-panel").hidden')
        click('#add-constant')
        until('window.loom.document.graph.nodes.some(n=>n.type==="Constant")')
        until('document.getElementById("validation-message").textContent.includes("Valid")')
        nid=js('window.loom.document.graph.nodes.find(n=>n.type==="Constant").id')
        assert js('document.querySelectorAll(".node[data-id='+nid+'] .port-in").length')==0
        inspect(nid);until('Boolean(document.getElementById("constant-fill"))')
        js('(()=>{const e=document.getElementById("constant-fill");e.value="-0";e.dispatchEvent(new Event("change"));})()')
        until('Object.is(window.loom.document.graph.nodes.find(n=>n.type==="Constant").fill,-0)')
        import math
        until('document.getElementById("validation-message").textContent.includes("Valid")')
        assert math.copysign(1,next(n for n in h.workspace.models()['sources']['graph']['nodes'] if n['type']=='Constant')['fill'])==-1
        h.dispatch({'action':'undo'},h.bridge)
        until('!Object.is(window.loom.document.graph.nodes.find(n=>n.type==="Constant").fill,-0)')
        h.dispatch({'action':'redo'},h.bridge)
        until('Object.is(window.loom.document.graph.nodes.find(n=>n.type==="Constant").fill,-0)')
        # Switching literal editor mode must not persist an invalid empty array.
        inspect(nid);before=h.workspace.db.serialize()
        js('(()=>{const e=document.getElementById("constant-mode");e.value="values";e.dispatchEvent(new Event("change"));})()')
        QTest.qWait(100);assert before==h.workspace.db.serialize()
        js('(()=>{const e=document.getElementById("constant-values");e.value="[-0.0,2.0]";e.dispatchEvent(new Event("change"));})()')
        until('window.loom.document.graph.nodes.find(n=>n.type==="Constant").values?.[1]===2')
        until('document.getElementById("validation-message").textContent.includes("Valid")')
        js('document.activeElement?.blur();document.body.dispatchEvent(new KeyboardEvent("keydown",{key:"Escape",bubbles:true}));document.body.dispatchEvent(new KeyboardEvent("keydown",{key:"Escape",bubbles:true}));window.loom.perform("fit")')
        click('.node[data-id='+nid+'] .node-type-icon .icon-hit')
        until('window.loom.document.graph.nodes.find(n=>n.type==="Constant").collapsed===true')
        point=js('(()=>{const e=document.querySelector(".node[data-id='+nid+'] .node-type-icon .icon-hit"),r=e.getBoundingClientRect();return [r.x+r.width/2,r.y+r.height/2]})()')
        old=next(n for n in h.workspace.models()['sources']['graph']['nodes'] if n['id']==nid).copy()
        target=view.focusProxy() or view;start=QPoint(round(point[0]),round(point[1]))
        QTest.mousePress(target,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier,start)
        QTest.mouseMove(target,start+QPoint(45,20),70);QTest.mouseRelease(target,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier,start+QPoint(45,20))
        until('window.loom.document.graph.nodes.find(n=>n.type==="Constant").x!=='+str(old['x']))
        h.workspace.save(folder/'final.vtw');loaded=Workspace.load(folder/'final.vtw')
        try:
            assert loaded.models()['sources']['graph']==h.workspace.models()['sources']['graph']
            assert validate_model(loaded.models(),'sources')['buffer_elements']==2
            assert loaded.db.execute('PRAGMA user_version').fetchone()[0]==VERSION
        finally:loaded.db.close()
        import sys
        assert not any(n.split('.')[0] in ('torch','numpy') for n in sys.modules)
        print(json.dumps({'status':'PASS','native_palette_click':True,'source_socket_roles':True,
            'parameter_inspector':True,'constant_signed_zero_history':True,'literal_mode_no_invalid_write':True,
            'compact_icon_native_drag':True,'save_reload':True,'container':VERSION,'host_torch':False,'console_errors':errors}))
    finally:
        view.close();view.setPage(QWebEnginePage(view));page.deleteLater();channel.deleteLater()
        QCoreApplication.sendPostedEvents(None,QEvent.Type.DeferredDelete);profile.deleteLater();QCoreApplication.sendPostedEvents(None,QEvent.Type.DeferredDelete)
        h.workspace.db.close()
        if h.workspace.history.directory:h.workspace.history.directory.cleanup()
