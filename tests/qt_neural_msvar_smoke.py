"""Real sandboxed Qt full-model diagram; host has no numerical imports."""
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from PyQt6.QtCore import QObject,QEventLoop,QTimer,QUrl,Qt,QPoint
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication
from PyQt6.QtWebChannel import QWebChannel
from PyQt6.QtWebEngineCore import QWebEnginePage,QWebEngineProfile
from PyQt6.QtWebEngineWidgets import QWebEngineView
from app.core import Workspace,validate_model,VERSION
from app.config import Config
from app.desktop import Window,Bridge
from app.neural_msvar import drawing,controlled_drawing

assert not os.environ.get('QTWEBENGINE_DISABLE_SANDBOX')
assert '--no-sandbox' not in os.environ.get('QTWEBENGINE_CHROMIUM_FLAGS','')
controlled=os.environ.get('VTW_TEST_CONTROLLED')=='1'
app=QApplication(['full-msvar-smoke']);view=QWebEngineView();view.resize(1700,1100)
profile=QWebEngineProfile(view);errors=[]
class Page(QWebEnginePage):
    def javaScriptConsoleMessage(self,level,message,line,source):
        if level==self.JavaScriptConsoleMessageLevel.ErrorMessageLevel:errors.append(f'{source}:{line}:{message}')
class Harness(QObject):
    state=Window.state;describe=Window.describe;dispatch=Window.dispatch
    def __init__(self,path):
        super().__init__();self.config=Config(path/'config.json');self.workspace=Workspace()
        self.workspace.put(controlled_drawing() if controlled else drawing());self.workspace.save(path/'start.vtw')
        self.job=None;self.rendering={};self.torch_python='';self.bridge=Bridge(self,view)
    def recent_files(self):return []
    def broadcast(self,kind='state',payload=None,**kwargs):self.bridge.send(kind,self.state() if kind=='state' else payload)
def js(source):
    values=[];loop=QEventLoop();timer=QTimer();timer.setSingleShot(True);timer.timeout.connect(loop.quit)
    view.page().runJavaScript(source,lambda value:(values.append(value),loop.quit()))
    end=time.monotonic()+20
    while not values and time.monotonic()<end:timer.start(2000);loop.exec();timer.stop()
    assert values,'JS callback timed out';return values[0]
def until(source):
    end=time.monotonic()+20
    while time.monotonic()<end:
        assert not errors,errors
        if js(source):return
        QTest.qWait(25)
    raise AssertionError('UI timeout: '+source)
def click(selector):
    point=js('(()=>{const r=document.querySelector('+json.dumps(selector)+').getBoundingClientRect();return [r.x+r.width/2,r.y+r.height/2]})()')
    QTest.mouseClick(view.focusProxy() or view,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier,QPoint(round(point[0]),round(point[1])))
    QTest.qWait(70)

with tempfile.TemporaryDirectory() as temporary:
    folder=Path(temporary);h=Harness(folder);page=Page(profile,view);view.setPage(page)
    channel=QWebChannel(page);channel.registerObject('host',h.bridge);page.setWebChannel(channel)
    url=QUrl.fromLocalFile(str(Path(__file__).parents[1]/'app/web/index.html'));url.setFragment('model=neural-msvar');view.setUrl(url);view.show()
    try:
        until('Boolean(window.loom?.document?.view.fitted)')
        until('document.getElementById("validation-message").textContent.includes("Valid")')
        assert js('document.querySelectorAll(".node[data-id=lags] .port-in").length')==(6 if controlled else 3)
        assert js('document.querySelectorAll(".node[data-id=lags] .port-out").length')==1
        assert js('document.querySelectorAll(".node[data-id=filter] .port-in").length')==(9 if controlled else 6)
        assert js('document.querySelectorAll(".node[data-id=filter] .port-out").length')==4
        assert js('getComputedStyle(document.querySelector(".node[data-id=filter] .node-header")).fill')!='rgb(0, 0, 0)'
        if controlled:
            js('document.querySelector(".node[data-id=variances]").dispatchEvent(new MouseEvent("dblclick",{bubbles:true}))')
            until('Boolean(document.getElementById("variance-transform"))')
            assert js('document.getElementById("variance-transform").value')=='exp'
            assert js('document.getElementById("variance-floor").value')=='0'
        js('document.querySelector(".node[data-id=lags]").dispatchEvent(new MouseEvent("dblclick",{bubbles:true}))')
        until('document.getElementById("inspector-content").textContent.includes("current observation is excluded")')
        js('document.activeElement?.blur();document.body.dispatchEvent(new KeyboardEvent("keydown",{key:"Escape",bubbles:true}));document.body.dispatchEvent(new KeyboardEvent("keydown",{key:"Escape",bubbles:true}));window.loom.perform("fit")')
        click('.node[data-id=lags] .node-type-icon .icon-hit')
        until('window.loom.document.graph.nodes.find(n=>n.id==="lags").collapsed===true')
        point=js('(()=>{const r=document.querySelector(".node[data-id=lags] .node-type-icon .icon-hit").getBoundingClientRect();return [r.x+r.width/2,r.y+r.height/2]})()')
        old=next(n for n in h.workspace.models()['neural-msvar']['graph']['nodes'] if n['id']=='lags').copy()
        target=view.focusProxy() or view;start=QPoint(round(point[0]),round(point[1]))
        QTest.mousePress(target,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier,start)
        QTest.mouseMove(target,start+QPoint(35,20),70);QTest.mouseRelease(target,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier,start+QPoint(35,20))
        until('window.loom.document.graph.nodes.find(n=>n.id==="lags").x!=='+str(old['x']))
        h.dispatch({'action':'undo'},h.bridge)
        until('window.loom.document.graph.nodes.find(n=>n.id==="lags").x==='+str(old['x']))
        h.dispatch({'action':'redo'},h.bridge)
        until('window.loom.document.graph.nodes.find(n=>n.id==="lags").x!=='+str(old['x']))
        h.workspace.save(folder/'final.vtw');loaded=Workspace.load(folder/'final.vtw')
        assert loaded.models()['neural-msvar']['graph']==h.workspace.models()['neural-msvar']['graph']
        assert validate_model(loaded.models(),'neural-msvar')['parameters']==1315;loaded.db.close()
        assert not any(name.split('.')[0] in ('torch','numpy') for name in sys.modules)
        output=Path(os.environ.get('VTW_TEST_CAPTURE',str(folder/'msvar.png')));view.grab().save(str(output))
        print(json.dumps({'status':'PASS','model':'K5/p3/D3','controlled':controlled,'parameters':1315,'causal_sockets':[6 if controlled else 3,1],
            'filter_sockets':[9 if controlled else 6,4],'native_compact_drag':True,'undo_redo':True,'save_reload':True,
            'host_torch':False,'sandbox':True,'container':VERSION,'console_errors':errors}),flush=True)
    finally:
        h.workspace.db.close();view.close();page.deleteLater();profile.deleteLater();QTest.qWait(100)
