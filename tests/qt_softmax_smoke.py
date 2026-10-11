"""Real palette/inspector axis edits, folding, history, export and .vtw reload."""
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from PyQt6.QtCore import QObject,QEvent,QEventLoop,QTimer,QUrl,QCoreApplication
from PyQt6.QtWidgets import QApplication
from PyQt6.QtWebChannel import QWebChannel
from PyQt6.QtWebEngineCore import QWebEnginePage,QWebEngineProfile
from PyQt6.QtWebEngineWidgets import QWebEngineView
from app.config import Config
from app.core import Workspace,default_model,generate_code,validate_model
from app.desktop import Bridge,Window

assert not os.environ.get('QTWEBENGINE_DISABLE_SANDBOX')
assert '--no-sandbox' not in os.environ.get('QTWEBENGINE_CHROMIUM_FLAGS','')
app=QApplication(['softmax-smoke']);view=QWebEngineView();view.resize(1440,980)
profile=QWebEngineProfile(view);errors=[]
class Page(QWebEnginePage):
    def javaScriptConsoleMessage(self,level,message,line,source):
        if level==self.JavaScriptConsoleMessageLevel.ErrorMessageLevel:errors.append(f'{source}:{line}: {message}')
class Harness(QObject):
    state=Window.state;describe=Window.describe;dispatch=Window.dispatch
    def __init__(self,folder):
        super().__init__();self.config=Config(folder/'config.json');self.workspace=Workspace()
        self.model=default_model('Three-way probabilities',inputs=3,outputs=3)
        self.model['graph']['nodes'][0].update(shape=[2,3],axes=['Position','Class'])
        self.workspace.put(self.model);self.job=None;self.torch_python=None;self.rendering={};self.bridge=Bridge(self,view)
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
        loop=QEventLoop();QTimer.singleShot(20,loop.quit);loop.exec()
    raise AssertionError(f'UI timeout: {script}; console: {errors}')
def inspect():
    js('document.querySelector(".node.Activation").dispatchEvent(new MouseEvent("dblclick",{bubbles:true}))')
    until('Boolean(document.getElementById("node-activation"))')
def native_axis(dim):
    deadline=time.monotonic()+10
    while True:
        n=next((n for n in h.workspace.models()[mid]['graph']['nodes'] if n['type']=='Activation'),None)
        if n and n.get('dim')==dim:return n
        assert time.monotonic()<deadline,'Native axis edit not acknowledged'
        loop=QEventLoop();QTimer.singleShot(20,loop.quit);loop.exec()

with tempfile.TemporaryDirectory() as directory:
    folder=Path(directory);h=Harness(folder);mid=h.model['id'];page=Page(profile,view);view.setPage(page)
    channel=QWebChannel(page);channel.registerObject('host',h.bridge);page.setWebChannel(channel)
    url=QUrl.fromLocalFile(str(Path(__file__).resolve().parents[1]/'app/web/index.html'));url.setFragment('model='+mid);view.setUrl(url);view.show()
    try:
        until('Boolean(window.loom?.document.view.fitted)')
        assert js('[...document.getElementById("activation-select").options].map(o=>o.value)')==['GELU','ReLU','Sigmoid','Softmax','Tanh']
        js('document.querySelector(".node.Linear").dispatchEvent(new MouseEvent("dblclick",{bubbles:true}));document.getElementById("activation-select").value="Softmax";document.getElementById("add-activation").click()')
        until('window.loom.document.graph.nodes.some(n=>n.activation==="Softmax")');inspect()
        assert js('document.getElementById("softmax-dim").value')=='-1';native_axis(-1)
        assert js('document.getElementById("inspector-content").textContent.includes("PyTorch dim = 2 (Class)")')
        assert js('document.getElementById("inspector-content").textContent.includes("Cross-entropy expects raw logits")')
        js('document.getElementById("softmax-dim").value="0";document.getElementById("softmax-dim").dispatchEvent(new Event("change"))')
        native_axis(0);until('document.getElementById("inspector-content").textContent.includes("PyTorch dim = 1 (Position)")')
        c=validate_model(h.workspace.models(),mid);assert c['output_shape']==[2,3];assert c['parameters']==12
        assert 'nn.Softmax(dim=1)' in generate_code(h.workspace.models(),mid)
        for bad in ('2','-3','1.5',''):
            js('document.getElementById("softmax-dim").value='+json.dumps(bad)+';document.getElementById("softmax-dim").dispatchEvent(new Event("change"))')
            assert js('document.getElementById("softmax-dim").value')=='0';native_axis(0)
        js('window.loom.perform("undo")');until('window.loom.document.graph.nodes.some(n=>n.activation==="Softmax"&&n.dim===-1)');native_axis(-1)
        js('window.loom.perform("redo")');until('window.loom.document.graph.nodes.some(n=>n.activation==="Softmax"&&n.dim===0)');native_axis(0)
        inspect();js('document.getElementById("node-activation").value="Tanh";document.getElementById("node-activation").dispatchEvent(new Event("change"))')
        until('window.loom.document.graph.nodes.some(n=>n.activation==="Tanh"&&!Object.hasOwn(n,"dim"))')
        assert not js('Boolean(document.getElementById("softmax-dim"))')
        js('window.loom.perform("undo")');until('window.loom.document.graph.nodes.some(n=>n.activation==="Softmax"&&n.dim===0)');native_axis(0)
        js('document.querySelector(".node.Activation .node-type-icon").dispatchEvent(new MouseEvent("click",{bubbles:true}))')
        until('document.querySelector(".node.Activation.collapsed .node-compact-label")?.firstChild.textContent==="Softmax 0"')
        deadline=time.monotonic()+10
        while not native_axis(0).get('collapsed'):
            assert time.monotonic()<deadline
            loop=QEventLoop();QTimer.singleShot(20,loop.quit);loop.exec()
        path=folder/'probabilities.vtw';h.workspace.save(path);loaded=Workspace.load(path)
        try:
            assert 'nn.Softmax(dim=1)' in generate_code(loaded.models(),mid)
            assert validate_model(loaded.models(),mid)['output_shape']==[2,3]
        finally:loaded.db.close()
        if os.environ.get('VTW_SOFTMAX_SCREENSHOT'):
            js('document.querySelector(".node.Activation .node-type-icon").dispatchEvent(new MouseEvent("dblclick",{bubbles:true}))');inspect()
            js('if(document.getElementById("library-panel").hidden)document.querySelector("[data-open-panel=library-panel]").click()')
            loop=QEventLoop();QTimer.singleShot(300,loop.quit);loop.exec();assert view.grab().save(os.environ['VTW_SOFTMAX_SCREENSHOT'])
        assert not errors,errors;assert 'torch' not in sys.modules
        print(json.dumps({'actual_softmax_palette_inspector':'PASS','sample_axis_batch_exclusion':'PASS','invalid_edits_preserved':'PASS',
                         'history_folding_and_switch':'PASS','native_export_vtw_reload':'PASS','console_errors':errors,'host_torch':False}))
    finally:
        if h.workspace.history.directory:h.workspace.history.directory.cleanup()
        h.workspace.db.close();view.close();view.setPage(QWebEnginePage(view));page.deleteLater();QCoreApplication.sendPostedEvents(None,QEvent.Type.DeferredDelete)
        profile.deleteLater();QCoreApplication.sendPostedEvents(None,QEvent.Type.DeferredDelete)
