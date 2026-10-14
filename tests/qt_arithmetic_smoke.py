"""Actual sandboxed Add/Scale editing, sockets, history and VTW23 save/reload."""
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
from test_arithmetic import model

assert not os.environ.get('QTWEBENGINE_DISABLE_SANDBOX')
assert '--no-sandbox' not in os.environ.get('QTWEBENGINE_CHROMIUM_FLAGS','')
app=QApplication(['arithmetic-smoke']);view=QWebEngineView();view.resize(1500,1000)
profile=QWebEngineProfile(view);errors=[]
class Page(QWebEnginePage):
    def javaScriptConsoleMessage(self,level,message,line,source):
        if level==self.JavaScriptConsoleMessageLevel.ErrorMessageLevel:errors.append(f'{source}:{line}:{message}')
class Harness(QObject):
    state=Window.state;describe=Window.describe;dispatch=Window.dispatch
    def __init__(self,folder):
        super().__init__();self.config=Config(folder/'config.json');self.workspace=Workspace()
        m=model(learned=True);positions={'x':(40,60),'base':(265,60),'other':(40,250),'scale':(265,250),'sum':(490,145),'y':(715,145)}
        for n in m['graph']['nodes']:n.update(x=positions[n['id']][0],y=positions[n['id']][1])
        self.workspace.put(m);self.workspace.save(folder/'initial.vtw');self.job=None;self.rendering={};self.torch_python='';self.bridge=Bridge(self,view)
    def recent_files(self):return []
    def broadcast(self,kind='state',payload=None,**kwargs):self.bridge.send(kind,self.state() if kind=='state' else payload)
def js(source):
    values=[];loop=QEventLoop();timer=QTimer();timer.setSingleShot(True);timer.timeout.connect(loop.quit)
    view.page().runJavaScript(source,lambda v:(values.append(v),loop.quit()));deadline=time.monotonic()+30
    while not values and time.monotonic()<deadline:timer.start(4000);loop.exec();timer.stop()
    assert values,'JS callback timeout';return values[0]
def until(source):
    deadline=time.monotonic()+30
    while time.monotonic()<deadline:
        assert not errors,errors
        if js(source):return
        loop=QEventLoop();QTimer.singleShot(25,loop.quit);loop.exec()
    raise AssertionError('UI timeout: '+source)
def click(selector,modifier=Qt.KeyboardModifier.NoModifier):
    point=js('(()=>{const e=document.querySelector('+json.dumps(selector)+');const r=e.getBoundingClientRect();const x=r.x+r.width/2,y=r.y+r.height/2;return [x,y,document.elementFromPoint(x,y)===e]})()')
    assert point[2],('Socket is occluded',selector,point)
    QTest.mouseClick(view.focusProxy() or view,Qt.MouseButton.LeftButton,modifier,QPoint(round(point[0]),round(point[1])))
    QTest.qWait(70)
def inspect(nid):js('document.querySelector(".node[data-id='+nid+']").dispatchEvent(new MouseEvent("dblclick",{bubbles:true}))')

with tempfile.TemporaryDirectory() as td:
    folder=Path(td);h=Harness(folder);page=Page(profile,view);view.setPage(page);channel=QWebChannel(page);channel.registerObject('host',h.bridge);page.setWebChannel(channel)
    url=QUrl.fromLocalFile(str(Path(__file__).parents[1]/'app/web/index.html'));url.setFragment('model=arithmetic');view.setUrl(url);view.show()
    try:
        until('Boolean(window.loom?.document?.view.fitted)')
        inspect('scale');until('Boolean(document.getElementById("op-coefficient"))')
        js('(()=>{const n=document.getElementById("op-coefficient");n.value="-1.5";n.dispatchEvent(new Event("change"));})()')
        until('window.loom.document.graph.nodes.find(n=>n.id==="scale").coefficient===-1.5')
        until('document.getElementById("validation-message").textContent.includes("Valid")')
        assert next(n for n in h.workspace.models()['arithmetic']['graph']['nodes'] if n['id']=='scale')['coefficient']==-1.5
        h.dispatch({'action':'undo'},h.bridge)
        until('window.loom.document.graph.nodes.find(n=>n.id==="scale").coefficient===2')
        h.dispatch({'action':'redo'},h.bridge)
        until('window.loom.document.graph.nodes.find(n=>n.id==="scale").coefficient===-1.5')
        inspect('base')
        # Panel keyboard shortcut is the same real event path as a user.
        js('document.activeElement?.blur();document.body.dispatchEvent(new KeyboardEvent("keydown",{key:"b",bubbles:true}))')
        until('!document.getElementById("library-panel").hidden')
        js('document.getElementById("add-add").click()')
        until('window.loom.document.graph.nodes.filter(n=>n.type==="Add").length===2')
        add=js('window.loom.document.graph.nodes.find(n=>n.type==="Add"&&n.id!=="sum").id')
        js('document.activeElement?.blur();document.body.dispatchEvent(new KeyboardEvent("keydown",{key:"Escape",bubbles:true}));document.body.dispatchEvent(new KeyboardEvent("keydown",{key:"Escape",bubbles:true}))')
        click('.node[data-id=other] .port-out')
        click('.node[data-id='+add+'] .port-in:not([data-peer])')
        until('window.loom.document.graph.nodes.find(n=>n.id==='+json.dumps(add)+').sources.length===2')
        until('document.getElementById("validation-message").textContent.includes("Valid")')
        c=validate_model(h.workspace.models(),'arithmetic');assert c['parameters']==6
        inspect(add);assert js('window.LoomBlockHelp.describe("add-add").paragraphs.join(" ").includes("No broadcasting")')  # prose moved to About (R2)
        assert not js('Boolean(document.getElementById("flatten-concat"))')
        assert js('document.querySelector(".node[data-id=sum] .node-params").textContent.includes("0 params")')
        h.workspace.save(folder/'final.vtw');loaded=Workspace.load(folder/'final.vtw')
        try:assert loaded.models()['arithmetic']['graph']==h.workspace.models()['arithmetic']['graph'];assert loaded.db.execute('PRAGMA user_version').fetchone()[0]==VERSION
        finally:loaded.db.close()
        if os.environ.get('VTW_ARITHMETIC_CAPTURE'):
            loop=QEventLoop();QTimer.singleShot(150,loop.quit);loop.exec();view.grab().save(os.environ['VTW_ARITHMETIC_CAPTURE'])
        # A separate private multioutput fixture proves Ctrl-click is selection,
        # not wiring or a document edit, and insertion touches only that branch.
        split={'id':'arithmetic','name':'Choose an output','graph':{'nodes':[
            {'id':'x','type':'Input','shape':[4],'x':50,'y':100},
            {'id':'split','type':'Splitter','parts':[{'output':'left','features':2},{'output':'right','features':2}],'x':300,'y':100},
            {'id':'left','type':'Output','x':550,'y':50},{'id':'right','type':'Output','x':550,'y':250}],
            'edges':[{'from':'x','to':'split'},{'from':'split','to':'left','from_port':'left'},{'from':'split','to':'right','from_port':'right'}]}}
        h.workspace.put(split);h.broadcast();until('Boolean(document.querySelector(".node[data-id=split]"))')
        js('document.activeElement?.blur();document.body.dispatchEvent(new KeyboardEvent("keydown",{key:"Escape",bubbles:true}));document.body.dispatchEvent(new KeyboardEvent("keydown",{key:"Escape",bubbles:true}));window.loom.perform("fit")')
        before=h.workspace.db.serialize()
        click('.node[data-id=split] .port-out[data-peer=right]',Qt.KeyboardModifier.ControlModifier)
        until('document.querySelectorAll(".port.append-target").length===1')
        assert js('document.querySelector(".port.append-target").dataset.peer==="right"')
        QTest.mouseMove(view.focusProxy() or view,QPoint(900,300));QTest.qWait(70)
        assert js('document.getElementById("pending-wire").hasAttribute("hidden")')
        assert h.workspace.db.serialize()==before
        js('document.getElementById("add-linear").click()')
        until('window.loom.document.graph.nodes.some(n=>n.type==="Linear")')
        until('document.getElementById("validation-message").textContent.includes("Valid")')
        chosen=js('window.loom.document.graph.nodes.find(n=>n.type==="Linear").id')
        graph=h.workspace.models()['arithmetic']['graph']
        assert {'from':'split','to':'left','from_port':'left'} in graph['edges']
        assert {'from':'split','to':chosen,'from_port':'right'} in graph['edges']
        assert {'from':chosen,'to':'right'} in graph['edges']
        assert next(n for n in graph['nodes'] if n['id']==chosen)['features']==2
        assert js('document.querySelectorAll(".port.append-target").length===0')
        click('.node[data-id=split] .port-out[data-peer=left]')
        until('Boolean(document.querySelector(".node[data-id=split] .port-out.armed[data-peer=left]"))')
        QTest.mouseMove(view.focusProxy() or view,QPoint(900,310));QTest.qWait(70)
        until('!document.getElementById("pending-wire").hasAttribute("hidden")')
        import sys
        assert not any(n.split('.')[0] in ('torch','numpy') for n in sys.modules)
        print(json.dumps({'ok':True,'actual_add_palette_and_pointer_sockets':True,'scale_edit_undo_redo':True,'ctrl_output_selection_no_write_or_wire':True,'selected_branch_inserted_other_branch_unchanged':True,'normal_click_still_wires':True,'container_version':VERSION,'parameters':c['parameters'],'host_torch':False,'console_errors':errors}))
    finally:
        view.close();view.setPage(QWebEnginePage(view));page.deleteLater();channel.deleteLater();QCoreApplication.sendPostedEvents(None,QEvent.Type.DeferredDelete);profile.deleteLater();QCoreApplication.sendPostedEvents(None,QEvent.Type.DeferredDelete);h.workspace.db.close()
        if h.workspace.history.directory:h.workspace.history.directory.cleanup()
