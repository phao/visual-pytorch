"""Real sandboxed WebEngine recurrent form, sockets, rewiring and compact drag."""
import copy,json,os,tempfile,time,sys
from pathlib import Path
from PyQt6.QtCore import QObject,QEvent,QEventLoop,QTimer,QUrl,QPoint,Qt,QCoreApplication
from PyQt6.QtGui import QMouseEvent
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication
from PyQt6.QtWebEngineCore import QWebEnginePage,QWebEngineProfile
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWebChannel import QWebChannel
from app.config import Config
from app.core import Workspace,validate_model
from app.desktop import Window,Bridge
from test_recurrent_authoring import authored

assert not os.environ.get('QTWEBENGINE_DISABLE_SANDBOX')
assert '--no-sandbox' not in os.environ.get('QTWEBENGINE_CHROMIUM_FLAGS','')
app=QApplication(['recurrent-authoring-smoke']);errors=[]
class Page(QWebEnginePage):
    def javaScriptConsoleMessage(self,level,message,line,source):
        if level==self.JavaScriptConsoleMessageLevel.ErrorMessageLevel:errors.append(f'{source}:{line}: {message}')
class Harness(QObject):
    state=Window.state;describe=Window.describe;dispatch=Window.dispatch
    def __init__(self,path):
        super().__init__();self.config=Config(path/'config.json');self.config.save(ui_zoom=1)
        self.workspace=Workspace();models,_,_=authored(cnn='--cnn' in sys.argv)
        if '--multi-state' in sys.argv:
            graph=models['transition']['graph'];cloned=copy.deepcopy(graph['nodes'])
            for n in cloned:
                n['id']+='2'
                if 'label' in n:n['label']+=' second'
                if n['type']=='Concat':n['sources']=[s+'2' for s in n['sources']]
            graph['nodes'].extend(cloned)
            graph['edges'].extend({'from':e['from']+'2','to':e['to']+'2'} for e in list(graph['edges']))
            graph['input_order'].extend(['x2','h2'])
        for model in models.values():self.workspace.put(model)
        self.workspace.dirty=False;self.job=None;self.torch_python=None;self.rendering={};self.bridges=[]
    def recent_files(self):return []
    def broadcast(self,kind='state',payload=None,**kwargs):
        for bridge in self.bridges:bridge.send(kind,self.state() if kind=='state' else payload)
def js(view,script):
    loop=QEventLoop();values=[];timer=QTimer();timer.setSingleShot(True);timer.timeout.connect(loop.quit)
    view.page().runJavaScript(script,lambda value:(values.append(value),loop.quit()));timer.start(4000);loop.exec();timer.stop()
    assert values,'JavaScript timeout';return values[0]
def until(view,script,seconds=15):
    deadline=time.monotonic()+seconds
    while time.monotonic()<deadline:
        if js(view,script):return
        QTest.qWait(25)
    diagnostic=js(view,'({error:document.getElementById("recurrent-error")?.textContent,invalid:[...document.querySelectorAll("#recurrent-form :invalid")].map(el=>({id:el.id,value:el.value})),open:document.getElementById("recurrent-dialog")?.open})')
    raise AssertionError(f'UI timeout: {script}; console: {errors}; dialog: {diagnostic}')
def point(view,selector):
    p=js(view,'(() => {const r=document.querySelector('+json.dumps(selector)+').getBoundingClientRect();return {x:r.left+r.width/2,y:r.top+r.height/2};})()')
    return QPoint(round(p['x']),round(p['y']))
def click(view,selector):
    js(view,'document.querySelector('+json.dumps(selector)+').scrollIntoView({block:"nearest"})');QTest.qWait(40)
    QTest.mouseClick(view.focusProxy() or view,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier,point(view,selector));QTest.qWait(70)
def drag(view,selector,dx,dy):
    widget=view.focusProxy() or view;start=point(view,selector);finish=start+QPoint(dx,dy)
    QTest.mousePress(widget,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier,start)
    app.sendEvent(widget,QMouseEvent(QEvent.Type.MouseMove,finish.toPointF(),widget.mapToGlobal(finish).toPointF(),Qt.MouseButton.NoButton,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier))
    QTest.qWait(30);QTest.mouseRelease(widget,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier,finish);QTest.qWait(100)
def choices(view,controls=True):
    js(view,'document.getElementById("recurrent-steps").value="4";[...document.querySelectorAll(".recurrent-input-row")].forEach(row=>{const role=row.querySelector("[data-role]"),state=row.dataset.input.startsWith("h");role.value=state?"state":"sequence";role.dispatchEvent(new Event("change"));if(state)row.querySelector("[data-next]").value="next"+row.dataset.input.slice(1);});')
    if controls:js(view,'document.querySelectorAll("[data-recurrent-control]").forEach(input=>input.checked=true)')

with tempfile.TemporaryDirectory() as tmp:
    h=Harness(Path(tmp));view=QWebEngineView();view.resize(1500,980)
    profile=QWebEngineProfile(view);page=Page(profile,view);view.setPage(page)
    bridge=Bridge(h,view);h.bridges=[bridge];channel=QWebChannel(page);channel.registerObject('host',bridge);page.setWebChannel(channel)
    url=QUrl.fromLocalFile(str(Path(__file__).resolve().parents[1]/'app/web/index.html'));url.setFragment('model=parent')
    view.setUrl(url);view.show();until(view,'Boolean(window.loom?.document)');config=h.config.path.read_bytes()
    try:
        before=copy.deepcopy(h.workspace.models()['parent']);history=len(h.workspace.history.undo_stack)
        js(view,'document.querySelector("[data-open-panel=library-panel]").click();document.getElementById("reference-select").value="transition";document.getElementById("add-recurrent").click()')
        until(view,'document.getElementById("recurrent-dialog").open');choices(view)
        click(view,'#recurrent-create')
        until(view,'!document.getElementById("recurrent-dialog").open && Boolean(document.querySelector(".node.Recurrent"))')
        until(view,'document.getElementById("inspector-content").textContent.includes("exact-int64-v1")')
        model=h.workspace.models()['parent'];r=next(n for n in model['graph']['nodes'] if n['type']=='Recurrent');nid=r['id']
        assert model['graph']['nodes'][:3]==before['graph']['nodes'] and model['graph']['edges'][:2]==before['graph']['edges']
        assert len(h.workspace.history.undo_stack)==history+1
        multi='--multi-state' in sys.argv;input_count=7 if multi else 5;output_count=4 if multi else 2
        parameters=23 if multi else 17 if '--cnn' in sys.argv else 13
        expected={'sequence/next':[4,2],'state/h':[2]}
        if multi:expected.update({'sequence/next2':[4,2],'state/h2':[2]})
        assert validate_model(h.workspace.models(),'parent')['shapes'][nid]['ports']==expected
        assert js(view,'document.querySelectorAll(".node.Recurrent .port-in").length')==input_count
        assert js(view,'document.querySelectorAll(".node.Recurrent .port-out").length')==output_count
        assert js(view,'new Set([...document.querySelectorAll(".node.Recurrent .port-in")].map(p=>p.getAttribute("cy"))).size')==input_count
        assert js(view,'[...document.querySelectorAll(".node.Recurrent .port-out title")].every(t=>t.textContent.includes("[B,"))')
        # Replacing exactly one named initial-state socket with native clicks.
        js(view,'window.loom.perform("reset-panels");window.loom.fit()');QTest.qWait(100)
        click(view,'.node.Input[data-id="input"] .port-out')
        target='.node.Recurrent .port-in[data-peer='+json.dumps(r['states'][0]['source'])+']'
        click(view,target)
        until(view,'window.loom.document.graph.nodes.find(n=>n.type==="Recurrent").states[0].source==="input"')
        changed=h.workspace.models()['parent'];changed_r=next(n for n in changed['graph']['nodes'] if n['id']==nid)
        assert changed_r['sequence_inputs']==r['sequence_inputs'] and changed_r['controls']==r['controls']
        assert changed_r['states'][1:]==r['states'][1:]
        assert len([e for e in changed['graph']['edges'] if e['to']==nid])==input_count
        assert not Window.describe(h,'parent')['valid'] # Original initial Input is visibly disconnected, not deleted.
        js(view,'window.loom.call("undo")');until(view,'window.loom.document.graph.nodes.find(n=>n.type==="Recurrent").states[0].source!=="input"')
        assert validate_model(h.workspace.models(),'parent')['parameters']==parameters
        # New layers after a selected state Output use its width 2, not aggregate 10.
        js(view,'window.loom.perform("reset-panels");window.loom.fit()');QTest.qWait(80)
        click(view,'.node.Recurrent .port-out[data-peer="state/h"]')
        js(view,'document.getElementById("add-linear").click()')
        until(view,'window.loom.document.graph.nodes.filter(n=>n.type==="Linear").length===2')
        model=h.workspace.models()['parent'];added=next(n for n in model['graph']['nodes'] if n['type']=='Linear' and n['id']!='linear1')
        assert added['features']==2 and added['bias'] is True
        validate_model(h.workspace.models(),'parent')
        js(view,'window.loom.call("undo")');until(view,'window.loom.document.graph.nodes.filter(n=>n.type==="Linear").length===1')
        # Folded recurrent icon is a real drag surface; sockets remain distinct.
        js(view,'window.loom.perform("reset-panels");window.loom.fit()');QTest.qWait(80)
        click(view,'.node.Recurrent .node-type-icon');until(view,'document.querySelector(".node.Recurrent.collapsed")!==null')
        compact=next(n for n in h.workspace.models()['parent']['graph']['nodes'] if n['id']==nid)
        drag(view,'.node.Recurrent .node-type-icon',45,37)
        until(view,'window.loom.document.graph.nodes.find(n=>n.type==="Recurrent").x!=='+str(compact['x']))
        assert js(view,'document.querySelectorAll(".node.Recurrent .port-in").length')==input_count
        assert js(view,'new Set([...document.querySelectorAll(".node.Recurrent .port-in")].map(p=>p.getAttribute("cy"))).size')==input_count
        # A changed parent graph while the modal is open refuses, preserving the edit.
        js(view,'document.getElementById("add-recurrent").click()');until(view,'document.getElementById("recurrent-dialog").open');choices(view)
        altered=h.workspace.models()['parent'];altered['graph']['nodes'][0]['x']+=24;h.workspace.put(altered);h.broadcast();QTest.qWait(100)
        sql=h.workspace.db.serialize();click(view,'#recurrent-create')
        until(view,'document.getElementById("recurrent-error").textContent.includes("changed")')
        assert h.workspace.db.serialize()==sql
        click(view,'#recurrent-cancel')
        path=Path(tmp)/'ui.vtw';h.workspace.save(path);loaded=Workspace.load(path)
        try:validate_model(loaded.models(),'parent')
        finally:loaded.db.close()
        assert h.config.path.read_bytes()==config and not errors,errors
        assert 'torch' not in sys.modules
        print(json.dumps({'real_recurrent_form':'PASS','explicit_multiport_controls':'PASS','native_socket_replace_preserves_other_bindings':'PASS',
          'selected_port_layer_shape':'PASS','native_compact_icon_drag':'PASS','stale_form_sql_unchanged':'PASS','atomic_undo_vtw_reload':'PASS',
          'transition':'multi-state' if multi else 'CNN' if '--cnn' in sys.argv else 'MLP','input_sockets':input_count,'output_sockets':output_count,'console_errors':errors,'host_torch':False}))
    finally:
        h.bridges=[];view.close();view.setPage(QWebEnginePage(view));page.deleteLater();QCoreApplication.sendPostedEvents(None,QEvent.Type.DeferredDelete)
        profile.deleteLater();QCoreApplication.sendPostedEvents(None,QEvent.Type.DeferredDelete)
        if h.workspace.history.directory:h.workspace.history.directory.cleanup()
        h.workspace.db.close()
