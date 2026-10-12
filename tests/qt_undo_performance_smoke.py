"""Actual sandboxed Ctrl-Z/Shift-Z with large retained data and typing guard."""
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch
from PyQt6.QtCore import QObject,QCoreApplication,QEvent,QEventLoop,QTimer,QUrl,Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication
from PyQt6.QtWebChannel import QWebChannel
from PyQt6.QtWebEngineCore import QWebEnginePage,QWebEngineProfile
from PyQt6.QtWebEngineWidgets import QWebEngineView
from app.core import Workspace,default_model
from app.config import Config
from app.desktop import Window,Bridge
from app.revisions import frozen_document

assert not os.environ.get('QTWEBENGINE_DISABLE_SANDBOX')
assert '--no-sandbox' not in os.environ.get('QTWEBENGINE_CHROMIUM_FLAGS','')
app=QApplication(['undo-performance-smoke']);view=QWebEngineView();view.resize(1500,1000)
profile=QWebEngineProfile(view);errors=[]
class Page(QWebEnginePage):
    def javaScriptConsoleMessage(self,level,message,line,source):
        if level==self.JavaScriptConsoleMessageLevel.ErrorMessageLevel:errors.append(f'{source}:{line}:{message}')
class Tabs:
    def count(self):return 1
    def widget(self,index):return view
    def setTabText(self,index,name):assert name=='Undo fixture'
class Harness(QObject):
    state=Window.state;describe=Window.describe;dispatch=Window.dispatch;broadcast=Window.broadcast;tab_name=Window.tab_name
    def __init__(self,folder):
        super().__init__();self.config=Config(folder/'config.json');self.workspace=Workspace();m=default_model('Undo fixture');m['id']='diagram';self.workspace.put(m)
        for j in range(4):
            self.workspace.put_spec('dataset',{'version':1,'id':'d'+str(j),'name':'Private retained data '+str(j),
                'storage':{'version':1,'kind':'inline-records'},'fields':[{'id':'x','dtype':'float32','shape':[2]}],
                'records':[{'id':'r'+str(i),'values':{'x':[i/1000.,-i/2000.]}} for i in range(10000)]})
        self.workspace.save(folder/'initial.vtw');self.job=None;self.rendering={};self.torch_python='';self.tabs=Tabs();self.bridge=Bridge(self,view)
    def recent_files(self):return []
    def update_title(self):pass
    def update_history_actions(self):pass
def js(source):
    values=[];loop=QEventLoop();timer=QTimer();timer.setSingleShot(True);timer.timeout.connect(loop.quit)
    view.page().runJavaScript(source,lambda value:(values.append(value),loop.quit()));deadline=time.monotonic()+30
    while not values and time.monotonic()<deadline:timer.start(4000);loop.exec();timer.stop()
    assert values,'Same callback exceeded deadline';return values[0]
def until(source):
    deadline=time.monotonic()+30
    while time.monotonic()<deadline:
        assert not errors,errors
        if js(source):return
        loop=QEventLoop();QTimer.singleShot(20,loop.quit);loop.exec()
    details=js('JSON.stringify({graph:window.loom?.document?.graph,validation:document.getElementById("validation-message")?.textContent,status:document.getElementById("status")?.textContent,toast:document.getElementById("toast")?.textContent})')
    raise AssertionError('UI timeout: '+source+' · '+str(details))

with tempfile.TemporaryDirectory() as td:
    folder=Path(td);h=Harness(folder);page=Page(profile,view);view.mid='diagram';view.bridge=h.bridge;view.setPage(page)
    channel=QWebChannel(page);channel.registerObject('host',h.bridge);page.setWebChannel(channel)
    url=QUrl.fromLocalFile(str(Path(__file__).parents[1]/'app/web/index.html'));url.setFragment('model=diagram');view.setUrl(url);view.show()
    try:
        until('Boolean(window.loom?.document?.view.fitted)')
        samples=[]
        for index in range(3):
            js('document.querySelector(".node.Linear").dispatchEvent(new MouseEvent("dblclick",{bubbles:true}))')
            until('Boolean(document.getElementById("features"))')
            js('(()=>{const f=document.getElementById("features");f.value="3";f.dispatchEvent(new Event("change"));})()')
            until('window.loom.document.graph.nodes.find(n=>n.id==="linear1").features===3&&document.getElementById("validation-message").textContent.includes("B, 3")')
            js('document.activeElement?.blur();document.getElementById("canvas").focus()')
            with patch('app.revisions.frozen_document',wraps=frozen_document) as proof:
                start=time.perf_counter();QTest.keyClick(view.focusProxy() or view,Qt.Key.Key_Z,Qt.KeyboardModifier.ControlModifier)
                until('window.loom.document.graph.nodes.find(n=>n.id==="linear1").features===1&&document.getElementById("validation-message").textContent.includes("B, 1")')
                samples.append({'keyboard_to_valid_render_ms':(time.perf_counter()-start)*1000,'full_frozen_checks':proof.call_count})
                assert proof.call_count==0
        js('document.activeElement?.blur();document.getElementById("canvas").focus()')
        QTest.keyClick(view.focusProxy() or view,Qt.Key.Key_Z,Qt.KeyboardModifier.ControlModifier|Qt.KeyboardModifier.ShiftModifier)
        until('window.loom.document.graph.nodes.find(n=>n.id==="linear1").features===3')
        # Text controls retain native editing undo instead of undoing the graph.
        js('document.querySelector(".node.Input").dispatchEvent(new MouseEvent("dblclick",{bubbles:true}))')
        until('Boolean(document.getElementById("port-label"))');before=h.workspace.db.serialize()
        js('document.getElementById("port-label").focus()')
        QTest.keyClick(view.focusProxy() or view,Qt.Key.Key_Z,Qt.KeyboardModifier.ControlModifier);QTest.qWait(80)
        assert h.workspace.db.serialize()==before
        assert next(n for n in h.workspace.models()['diagram']['graph']['nodes'] if n['id']=='linear1')['features']==3
        h.workspace.save(folder/'final.vtw');loaded=Workspace.load(folder/'final.vtw')
        try:assert loaded.models()==h.workspace.models();assert len(loaded.list_specs('dataset',all_revisions=True))==4
        finally:loaded.db.close()
        assert not any(n.split('.')[0] in ('torch','numpy') for n in sys.modules)
        print(json.dumps({'ok':True,'actual_ctrl_z_and_shift_z':True,'typing_undo_preserved':True,'saved_data_unchanged':True,
                          'samples':samples,'host_torch':False,'console_errors':errors}))
    finally:
        view.close();view.setPage(QWebEnginePage(view));page.deleteLater();channel.deleteLater();QCoreApplication.sendPostedEvents(None,QEvent.Type.DeferredDelete);profile.deleteLater();QCoreApplication.sendPostedEvents(None,QEvent.Type.DeferredDelete);h.workspace.db.close()
        if h.workspace.history.directory:h.workspace.history.directory.cleanup()
