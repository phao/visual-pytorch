"""Actual sandboxed WebEngine LSTM forms: creation, editing, refusal/history."""
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from PyQt6.QtCore import QObject,QEventLoop,QTimer,QUrl,QPoint,Qt,QCoreApplication,QEvent
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication
from PyQt6.QtWebEngineCore import QWebEnginePage,QWebEngineProfile
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWebChannel import QWebChannel
from app.config import Config
from app.core import Workspace,default_model,validate_model
from app.desktop import Window,Bridge

assert not os.environ.get('QTWEBENGINE_DISABLE_SANDBOX')
assert '--no-sandbox' not in os.environ.get('QTWEBENGINE_CHROMIUM_FLAGS','')
app=QApplication(['lstm-authoring-smoke']);errors=[]


class Page(QWebEnginePage):
    def javaScriptConsoleMessage(self,level,message,line,source):
        if level==self.JavaScriptConsoleMessageLevel.ErrorMessageLevel:errors.append(f'{source}:{line}: {message}')


class Harness(QObject):
    state=Window.state;describe=Window.describe;dispatch=Window.dispatch
    def __init__(self,root):
        super().__init__();self.config=Config(root/'config.json');self.config.save(ui_zoom=1)
        self.workspace=Workspace();parent=default_model();parent['id']='parent';self.workspace.put(parent);self.workspace.dirty=False
        self.job=None;self.torch_python=None;self.rendering={};self.bridges=[]
    def recent_files(self):return []
    def broadcast(self,kind='state',payload=None,**kwargs):
        for bridge in self.bridges:bridge.send(kind,self.state() if kind=='state' else payload)


def js(view,script):
    loop=QEventLoop();values=[];timer=QTimer();timer.setSingleShot(True);timer.timeout.connect(loop.quit)
    view.page().runJavaScript(script,lambda value:(values.append(value),loop.quit()));timer.start(4000);loop.exec();timer.stop()
    assert values,'JS timeout';return values[0]


def until(view,script,seconds=15):
    deadline=time.monotonic()+seconds
    while time.monotonic()<deadline:
        if js(view,script):return
        QTest.qWait(25)
    diagnostic=js(view,'({error:document.getElementById("lstm-error").textContent,shapes:document.getElementById("lstm-shapes").textContent,bidirectional:document.getElementById("lstm-bidirectional").checked,layers:document.getElementById("lstm-layers").value,hidden:document.getElementById("lstm-hidden-size").value,projection:document.getElementById("lstm-projection").value})')
    raise AssertionError(f'UI timeout: {script}; errors={errors}; form={diagnostic}')


def click(view,selector):
    js(view,'document.querySelector('+json.dumps(selector)+').scrollIntoView({block:"center"})');QTest.qWait(30)
    r=js(view,'(()=>{const r=document.querySelector('+json.dumps(selector)+').getBoundingClientRect();return {x:r.left+r.width/2,y:r.top+r.height/2};})()')
    QTest.mouseClick(view.focusProxy() or view,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier,QPoint(round(r['x']),round(r['y'])))


def capture(view,name):
    evidence=os.environ.get('LSTM_EVIDENCE_DIR')
    if evidence:
        destination=Path(evidence)
        assert destination.is_dir(),'LSTM_EVIDENCE_DIR must be an existing private evidence directory'
        # The DOM's open flag precedes WebEngine's composited frame. Capture
        # painted evidence rather than the graph frame from before the modal.
        QTest.qWait(500)
        assert view.grab().save(str(destination/name))


with tempfile.TemporaryDirectory() as folder:
    root=Path(folder);h=Harness(root);view=QWebEngineView();view.resize(1500,1000)
    profile=QWebEngineProfile(view);page=Page(profile,view);view.setPage(page);bridge=Bridge(h,view);h.bridges=[bridge]
    channel=QWebChannel(page);channel.registerObject('host',bridge);page.setWebChannel(channel)
    url=QUrl.fromLocalFile(str(Path(__file__).resolve().parents[1]/'app/web/index.html'));url.setFragment('model=parent');view.setUrl(url);view.show()
    try:
        until(view,'Boolean(window.loom?.document)');config=h.config.path.read_bytes()
        js(view,'window.loom.perform("reset-panels");window.loom.perform("library")')
        # Open the existing palette through an actual mouse click.
        click(view,'#add-lstm');until(view,'document.getElementById("lstm-dialog").open')
        raw=h.workspace.db.serialize();count=len(h.workspace.history.undo_stack)
        click(view,'#lstm-cancel');until(view,'!document.getElementById("lstm-dialog").open')
        assert h.workspace.db.serialize()==raw and len(h.workspace.history.undo_stack)==count
        click(view,'#add-lstm');until(view,'document.getElementById("lstm-dialog").open')
        js(view,'document.getElementById("lstm-label").value="Authored projected LSTM";document.getElementById("lstm-hidden-size").value="7";document.getElementById("lstm-layers").value="2";document.getElementById("lstm-projection").value="4"')
        click(view,'#lstm-bidirectional');click(view,'#lstm-lengths')
        until(view,'document.getElementById("lstm-shapes").textContent.includes("[B, 4, 7]")')
        capture(view,'lstm-create-form.png')
        click(view,'#lstm-apply');until(view,'!document.getElementById("lstm-dialog").open&&Boolean(document.querySelector(".node.LSTM"))')
        doc=h.workspace.models()['parent'];node=next(n for n in doc['graph']['nodes'] if n['type']=='LSTM');nid=node['id'];contract=validate_model(h.workspace.models(),'parent')
        assert contract['shapes'][nid]['ports']=={'sequence':[4,8],'hidden':[4,4],'cell':[4,7]}
        assert len(h.workspace.history.undo_stack)==count+1
        until(view,'Boolean(document.getElementById("edit-lstm-interface"))');click(view,'#edit-lstm-interface')
        until(view,'document.getElementById("lstm-dialog").open&&!document.getElementById("lstm-edit-fields").hidden')
        assert js(view,'document.getElementById("lstm-dialog").scrollTop')==0
        capture(view,'lstm-edit-form.png')
        raw=h.workspace.db.serialize();count=len(h.workspace.history.undo_stack);click(view,'#lstm-apply')
        until(view,'!document.getElementById("lstm-dialog").open');assert h.workspace.db.serialize()==raw and len(h.workspace.history.undo_stack)==count
        click(view,'#edit-lstm-interface');until(view,'document.getElementById("lstm-dialog").open')
        js(view,'document.getElementById("lstm-hidden-size").value="8"');click(view,'#lstm-apply')
        until(view,'document.getElementById("lstm-error").textContent.includes("initial cell")')
        assert h.workspace.db.serialize()==raw and len(h.workspace.history.undo_stack)==count
        assert js(view,'document.getElementById("lstm-hidden-size").value')=='8'
        js(view,'document.getElementById("lstm-hidden-size").value="7";document.getElementById("lstm-label").value="Edited through Qt"')
        click(view,'#lstm-apply');until(view,'!document.getElementById("lstm-dialog").open')
        after=h.workspace.models();assert next(n for n in after['parent']['graph']['nodes'] if n['id']==nid)['label']=='Edited through Qt'
        assert len(h.workspace.history.undo_stack)==count+1
        js(view,'window.loom.call("undo")');until(view,'window.loom.document.graph.nodes.find(n=>n.type==="LSTM").label==="Authored projected LSTM"')
        js(view,'window.loom.call("redo")');until(view,'window.loom.document.graph.nodes.find(n=>n.type==="LSTM").label==="Edited through Qt"')
        click(view,'#edit-lstm-interface');until(view,'document.getElementById("lstm-dialog").open')
        raw=h.workspace.db.serialize();doc=h.workspace.models()['parent'];doc['name']='External model edit';h.workspace.put(doc);changed=h.workspace.db.serialize()
        js(view,'document.getElementById("lstm-label").value="Stale edit"');click(view,'#lstm-apply')
        until(view,'document.getElementById("lstm-error").textContent.includes("changed")')
        assert h.workspace.db.serialize()==changed and changed!=raw
        click(view,'#lstm-cancel');until(view,'!document.getElementById("lstm-dialog").open')
        path=root/'saved.vtw';h.workspace.save(path);loaded=Workspace.load(path)
        try:assert loaded.models()==h.workspace.models()
        finally:loaded.db.close()
        assert h.config.path.read_bytes()==config and not errors and 'torch' not in sys.modules
        capture(view,'lstm-authored-graph.png')
        print(json.dumps({'actual_palette_form_create_edit':'PASS','cancel_noop_refusal_stale':'PASS',
            'atomic_undo_redo_save_reload':'PASS','initial_state_ports':'PASS','console_errors':errors,'host_torch':False}))
    finally:
        h.workspace.db.close();view.close();view.setPage(QWebEnginePage(view));page.deleteLater()
        QCoreApplication.sendPostedEvents(None,QEvent.Type.DeferredDelete)
        profile.deleteLater();QCoreApplication.sendPostedEvents(None,QEvent.Type.DeferredDelete);app.processEvents()
