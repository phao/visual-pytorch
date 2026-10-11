"""Bounded real WebEngine UI/bridge check using a private in-memory workspace.

Run with QT_QPA_PLATFORM=offscreen QTWEBENGINE_CHROMIUM_FLAGS=--disable-gpu
/usr/bin/python3 -m tests.qt_ui_smoke. Chromium sandboxing remains enabled.
"""
import json
import os
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

from PyQt6.QtCore import QObject, QEventLoop, QTimer, QUrl, QEvent, QCoreApplication, Qt, QPoint, QPointF
from PyQt6.QtGui import QMouseEvent
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication
from PyQt6.QtWebChannel import QWebChannel
from PyQt6.QtWebEngineCore import QWebEnginePage, QWebEngineProfile
from PyQt6.QtWebEngineWidgets import QWebEngineView

from app.config import Config
from app.core import Workspace, default_model
from app.desktop import Bridge, Window, find_torch_python


assert not os.environ.get('QTWEBENGINE_DISABLE_SANDBOX')
assert '--no-sandbox' not in os.environ.get('QTWEBENGINE_CHROMIUM_FLAGS', '')
app = QApplication(['qt-ui-smoke'])
view = QWebEngineView()
view.resize(1100, 700)
profile = QWebEngineProfile(view)  # Off the record; never touches user browser storage.
errors = []


class Page(QWebEnginePage):
    def javaScriptConsoleMessage(self, level, message, line, source):
        if level == QWebEnginePage.JavaScriptConsoleMessageLevel.ErrorMessageLevel:
            errors.append(f'{source}:{line}: {message}')


class Harness(QObject):
    state = Window.state
    describe = Window.describe
    dispatch = Window.dispatch

    def __init__(self, path):
        super().__init__()
        self.config = Config(path)
        self.workspace = Workspace()
        self.model = default_model()
        self.model['graph']['nodes'][0].update(shape=[2,3], features=6)
        self.model['graph']['nodes'][1]['features']=4
        self.workspace.put(self.model)
        self.model=self.workspace.models()[self.model['id']]
        self.workspace.put(default_model('zeta'))
        self.workspace.put(default_model('Alpha'))
        self.job = None
        self.torch_python = None
        self.rendering = {'mode': 'software'}
        self.bridge = Bridge(self, view)

    def recent_files(self):
        return []

    def broadcast(self, kind='state', payload=None, exclude=None):
        if self.bridge != exclude:
            self.bridge.send(kind, self.state() if kind == 'state' else payload)


def js(script):
    loop = QEventLoop()
    timer = QTimer()
    timer.setSingleShot(True)
    timer.timeout.connect(loop.quit)
    values = []

    def result(value):
        values.append(value)
        loop.quit()

    view.page().runJavaScript(script, result)
    timer.start(3000)
    loop.exec()
    timer.stop()
    assert values, 'JavaScript callback timed out'
    return values[0]


def until(script, timeout=8):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if js(script):
            return
        loop = QEventLoop()
        QTimer.singleShot(20, loop.quit)
        loop.exec()
    raise AssertionError(f'UI condition timed out: {script}; errors: {errors}')


with tempfile.TemporaryDirectory() as td:
    h = Harness(Path(td)/'config.json')
    page = Page(profile, view)
    view.setPage(page)
    channel = QWebChannel(page)
    channel.registerObject('host', h.bridge)
    page.setWebChannel(channel)
    url = QUrl.fromLocalFile(str(Path(__file__).resolve().parents[1]/'app/web/index.html'))
    url.setFragment('model='+h.model['id'])
    view.setUrl(url)
    view.show()
    until('Boolean(window.loom && window.loom.document && document.querySelector(".node-title"))')
    assert js('document.title.endsWith(" · Visual PyTorch")')
    assert js('document.getElementById("home").textContent.includes("Visual PyTorch")')
    assert js('!document.body.textContent.includes("Tensor Loom")')
    assert js('JSON.stringify([...document.querySelectorAll("#model-list b")].map(e=>e.textContent))') == json.dumps(['Alpha','Linear regression','zeta'],separators=(',',':'))
    assert js('JSON.stringify([...document.getElementById("reference-select").options].map(e=>e.textContent))') == '["Alpha","zeta"]'
    assert js('JSON.stringify([...document.getElementById("activation-select").options].map(e=>e.textContent))') == '["GELU","ReLU","Sigmoid","Softmax","Tanh"]'
    assert js('document.getElementById("activation-select").value') == 'ReLU'
    assert js('JSON.stringify([...document.querySelectorAll("#pack-menus button")].map(b=>[...b.childNodes].filter(n=>n.nodeType===Node.TEXT_NODE).map(n=>n.textContent).join("").trim()))') == '["Architectures","Building blocks","Experiment","Inspector","Layout"]'
    assert js('JSON.stringify([...document.querySelectorAll("#library-scroll .palette b")].map(e=>e.textContent))') == '["Activation","AdaptiveAvgPool2d","Architecture reference","AvgPool2d","Concat","Conv2d","Flatten","Linear","MaxPool2d","Permute","Reshape","Input part","Output","Replicator","Splitter"]'
    samples = []
    for theme in ('github', 'math', 'metallic'):
        for font, spacing in ((14, 1), (18, 4), (28, 10)):
            h.config.values.update(theme=theme, font_size=font, spacing=spacing)
            h.broadcast()
            until(f'document.documentElement.dataset.theme=={json.dumps(theme)} && getComputedStyle(document.body).fontSize=="{font}px"')
            metrics = js("""(() => {
                const editor=document.getElementById('editor').getBoundingClientRect();
                const pack=document.getElementById('menu-pack'),rect=pack.getBoundingClientRect();
                const menus=document.getElementById('pack-menus'),style=getComputedStyle(menus);
                return {x:rect.left-editor.left,y:rect.top-editor.top,
                    gap:parseFloat(style.rowGap),left:parseFloat(style.paddingLeft),
                    bottom:parseFloat(style.paddingBottom),top:parseFloat(style.paddingTop),
                    fonts:{body:getComputedStyle(document.body).fontFamily,
                        code:getComputedStyle(document.getElementById('code')).fontFamily,
                        shape:getComputedStyle(document.querySelector('.node-shape')).fontFamily,
                        title:getComputedStyle(document.querySelector('.node-title')).fontFamily},
                    background:getComputedStyle(pack).backgroundImage,
                    overflow:menus.scrollHeight>menus.clientHeight || menus.scrollWidth>menus.clientWidth,
                    hiddenMenus:[...document.querySelectorAll('.floating-panel')].every(p=>p.hidden),
                    noArchitectureDeletion:!document.getElementById('delete-model')};
            })()""")
            assert metrics['x'] == metrics['y'] == 0, metrics
            assert metrics['gap'] == metrics['top'] == spacing, metrics
            assert metrics['left'] == metrics['bottom'] == 2*spacing, metrics
            assert not metrics['overflow'] and metrics['hiddenMenus'], metrics
            assert metrics['noArchitectureDeletion']
            assert 'monospace' in metrics['fonts']['code'] and 'monospace' in metrics['fonts']['shape']
            assert ('STIX' in metrics['fonts']['body']) == (theme == 'math'), metrics
            assert metrics['fonts']['title'] == metrics['fonts']['body'], metrics
            if theme == 'metallic':
                assert metrics['background'].startswith('linear-gradient'), metrics
            samples.append({'theme':theme,'font':font,'spacing':spacing})
            js('window.loom.perform("about-building-block","add-linear");')
            assert js('getComputedStyle(document.querySelector("#block-about-content p")).fontSize')==str(font)+'px'
            assert js('(() => {const r=document.getElementById("block-about-dialog").getBoundingClientRect();return r.width<=innerWidth && r.height<=innerHeight;})()')
            js('document.getElementById("block-about-dialog").close();')
            screenshot = os.environ.get('VTW_SMOKE_SCREENSHOT')
            if screenshot and font == 18:
                js('window.loom.fit();')
                loop = QEventLoop()
                QTimer.singleShot(100, loop.quit)
                loop.exec()
                target = Path(screenshot).with_name(Path(screenshot).stem+'-'+theme+'.png')
                assert view.grab().save(str(target))

    # Reset restores CSS origin even after moving the launcher.
    assert js("""(() => {
        const p=document.getElementById('menu-pack');p.style.left='90px';p.style.top='75px';
        window.loom.perform('reset-panels');
        return getComputedStyle(p).left==='0px' && getComputedStyle(p).top==='0px';
    })()""")
    h.config.values.update(theme='math',font_size=18,spacing=4)
    h.broadcast()
    until('document.documentElement.dataset.theme=="math"')
    # Palette context help is read-only, targets nested icons/text, and covers
    # every actual button rather than only a detached descriptor unit test.
    before_help=h.workspace.db.serialize()
    help_graph=js('JSON.stringify(window.loom.document.graph)')
    js('window.loom.perform("library");')
    palette_buttons=js('[...document.querySelectorAll("#library-panel .palette")].map(b=>({id:b.id,name:b.querySelector("b").textContent.trim()}))')
    for button in palette_buttons:
        js('document.getElementById('+json.dumps(button['id'])+').scrollIntoView({block:"center"});')
        context=js('(() => {const b=document.getElementById('+json.dumps(button['id'])+');const r=b.querySelector("b").getBoundingClientRect();return window.loom.contextAt(r.left+r.width/2,r.top+r.height/2);})()')
        assert context=={'palette':button},context
        js('window.loom.perform("about-building-block",'+json.dumps(button['id'])+');')
        assert js('document.getElementById("block-about-dialog").open')
        assert js('document.getElementById("block-about-title").textContent')=='About '+button['name']
        assert js('document.querySelectorAll("#block-about-content p").length')>=4
        assert js('JSON.stringify(window.loom.document.graph)')==help_graph
        js('document.getElementById("block-about-dialog").close();')
    js('document.getElementById("activation-select").value="Sigmoid";window.loom.perform("about-building-block","add-activation");')
    assert js('document.getElementById("block-about-content").textContent.includes("Sigmoid") && document.getElementById("block-about-content").textContent.includes("does not choose a classification loss")')
    QTest.keyClick(view.focusProxy() or view,Qt.Key.Key_Escape)
    until('!document.getElementById("block-about-dialog").open')
    assert h.workspace.db.serialize()==before_help
    js('document.getElementById("activation-select").value="ReLU";window.loom.perform("about-building-block","missing");')
    assert not js('document.getElementById("block-about-dialog").open')
    js('window.loom.perform("reset-panels");')
    js('window.loom.fit();')
    linear = next(n['id'] for n in h.model['graph']['nodes'] if n['type']=='Linear')
    input_id = next(n['id'] for n in h.model['graph']['nodes'] if n['type']=='Input')
    QTest.qWait(100)  # Let Chromium composite the selected font before mouse input.
    assert js('document.querySelector(".node.Linear .node-params").textContent.startsWith("Bias on")')
    assert js('document.querySelector(".node.Linear .node-params title").textContent.includes("16 params")')
    assert js('document.querySelectorAll(".node-resize").length===window.loom.document.graph.nodes.length')
    # Real Qt mouse events flow through Chromium's pointer-capture resize path.
    grip=js("""(() => {
        const r=document.querySelector('.node.Linear .node-resize').getBoundingClientRect();
        return {x:r.left+r.width/2,y:r.top+r.height/2};
    })()""")
    widget=view.focusProxy() or view
    start=QPoint(round(grip['x']),round(grip['y']))
    finish=start+QPoint(60,35)
    QTest.mousePress(widget,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier,start)
    QTest.mouseMove(widget,finish,20)
    QTest.mouseRelease(widget,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier,finish)
    until('window.loom.document.graph.nodes.find(n=>n.type==="Linear").width>180')
    until('window.loom.state.history.undo')
    geom=js("""(() => {
        const n=window.loom.document.graph.nodes.find(n=>n.type==='Linear'),g=document.querySelector('.node.Linear');
        return {width:n.width,height:n.height,renderWidth:g.querySelector('rect').width.baseVal.value,
            out:g.querySelector('.port-out').cx.baseVal.value,
            y:g.querySelector('.port-out').cy.baseVal.value,dirtyCursor:document.documentElement.classList.contains('loom-resizing')};
    })()""")
    assert abs(geom['width']-geom['renderWidth'])<.001 and abs(geom['width']-geom['out'])<.001,geom
    assert abs(geom['height']/2-geom['y'])<.001 and not geom['dirtyCursor'],geom
    assert h.workspace.models()[h.model['id']]['graph']['nodes'][1]['width']==geom['width']
    screenshot = os.environ.get('VTW_SMOKE_SCREENSHOT')
    if screenshot:
        target = Path(screenshot).with_name(Path(screenshot).stem+'-resized.png')
        QTest.qWait(100)
        assert view.grab().save(str(target))
    js('window.loom.perform("undo");')
    until('!("width" in window.loom.document.graph.nodes.find(n=>n.type==="Linear"))')
    js('window.loom.perform("redo");')
    until('window.loom.document.graph.nodes.find(n=>n.type==="Linear").width>180')
    linear_id=js('window.loom.document.graph.nodes.find(n=>n.type==="Linear").id')
    model_id=h.model['id']
    js('window.loom.perform("inspector");')
    assert js('!["remove-node","block-width","block-height","auto-block-size"].some(id=>document.getElementById(id))')
    sized_graph=js('JSON.stringify(window.loom.document.graph)')
    from app.evidence import identity
    before_size_reset=identity(h.workspace.models(),model_id)
    js('window.loom.perform("automatic-block-size",'+json.dumps(linear_id)+',"wrong-model");window.loom.perform("automatic-block-size","missing",'+json.dumps(model_id)+');')
    assert js('JSON.stringify(window.loom.document.graph)')==sized_graph
    js('window.loom.perform("automatic-block-size",'+json.dumps(linear_id)+','+json.dumps(model_id)+');')
    until('!("width" in window.loom.document.graph.nodes.find(n=>n.type==="Linear"))')
    assert identity(h.workspace.models(),model_id)==before_size_reset
    js('window.loom.perform("undo");')
    until('JSON.stringify(window.loom.document.graph)==='+json.dumps(sized_graph))
    js('window.loom.perform("redo");')
    until('!("width" in window.loom.document.graph.nodes.find(n=>n.type==="Linear"))')
    history_count=len(h.workspace.history.undo_stack)
    js('window.loom.perform("automatic-block-size",'+json.dumps(linear_id)+','+json.dumps(model_id)+');')
    assert len(h.workspace.history.undo_stack)==history_count
    js("document.querySelector('.node.Linear .node-resize').dispatchEvent(new KeyboardEvent('keydown',{key:'ArrowRight',ctrlKey:true,bubbles:true}));")
    until('"width" in window.loom.document.graph.nodes.find(n=>n.type==="Linear")')
    corner=js("""(() => {const n=window.loom.document.graph.nodes.find(n=>n.type==='Linear');return {x:n.x+n.width,y:n.y+n.height};})()""")
    assert corner['x']%24==corner['y']%24==0,corner
    js('window.loom.perform("automatic-block-size",'+json.dumps(linear_id)+','+json.dumps(model_id)+');')
    until('!("width" in window.loom.document.graph.nodes.find(n=>n.type==="Linear"))')
    js('document.getElementById("bias").click();')
    until('document.querySelector(".node.Linear .node-params").textContent.startsWith("Bias off")')
    assert js('document.querySelector(".contract-card").textContent.includes("Bias off")')
    assert js('document.querySelector(".contract-card").textContent.includes("A [4, 3]")')
    assert js('document.querySelector(".node.Linear .node-params title").textContent.includes("12 params")')
    assert js('document.getElementById("contract-explanation").textContent.includes("Bias is off")')
    js('window.loom.perform("undo");')
    until('document.querySelector(".node.Linear .node-params").textContent.startsWith("Bias on")')
    js('window.loom.perform("reset-panels");')
    # Folding controls are real mouse targets. A compact icon is also its
    # block's drag surface, not a button which swallows all pointer presses.
    def block_center(selector):
        point=js('(() => {const r=document.querySelector('+json.dumps(selector)+').getBoundingClientRect();return {x:r.left+r.width/2,y:r.top+r.height/2};})()')
        return QPoint(round(point['x']),round(point['y']))

    graph_before_fold=js('JSON.stringify(window.loom.document.graph)')
    from app.evidence import identity
    signature_before_fold=identity(h.workspace.models(),h.model['id'])
    icon_selector='.node.Linear .node-type-icon'
    QTest.mouseClick(widget,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier,block_center(icon_selector))
    until('window.loom.document.graph.nodes.find(n=>n.type==="Linear").collapsed===true')
    assert js('document.querySelector(".node.Linear .node-title")===null')
    assert js('document.querySelector(".node.Linear .node-compact-label").firstChild.nodeValue==="3 → 4"')
    assert js('document.querySelector(".node.Linear .node-compact-bias").textContent==="Bias on"')
    assert js('document.querySelector(".node.Linear .port-in")!==null && document.querySelector(".node.Linear .port-out")!==null')
    assert js('[...document.querySelectorAll(".node.Linear title")].some(t=>t.textContent.includes("Bias on"))')
    before_move=js('window.loom.document.graph.nodes.find(n=>n.type==="Linear").x')
    start=block_center(icon_selector);finish=start+QPoint(65,30)
    QTest.mousePress(widget,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier,start)
    QTest.mouseMove(widget,finish,20)
    QTest.mouseRelease(widget,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier,finish)
    until('window.loom.document.graph.nodes.find(n=>n.type==="Linear").x>'+str(before_move+10))
    until('window.loom.state.history.undo')
    assert js('window.loom.document.graph.nodes.find(n=>n.type==="Linear").collapsed===true')
    assert not js('document.documentElement.classList.contains("loom-dragging")')
    assert js('document.getSelection().toString()')==''
    assert h.workspace.models()[h.model['id']]['graph']['nodes'][1]['x']>before_move+10
    assert identity(h.workspace.models(),h.model['id'])==signature_before_fold
    moved_graph=js('JSON.stringify(window.loom.document.graph)')
    js('window.loom.perform("undo");')
    until('window.loom.document.graph.nodes.find(n=>n.type==="Linear").x==='+str(before_move))
    js('window.loom.perform("redo");')
    until('JSON.stringify(window.loom.document.graph)==='+json.dumps(moved_graph))
    js('window.loom.perform("undo");')
    until('window.loom.document.graph.nodes.find(n=>n.type==="Linear").x==='+str(before_move))
    start=block_center(icon_selector);finish=start+QPoint(39,27)
    QTest.keyPress(widget,Qt.Key.Key_Control)
    QTest.mousePress(widget,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.ControlModifier,start)
    QCoreApplication.sendEvent(widget,QMouseEvent(QEvent.Type.MouseMove,QPointF(finish),
        QPointF(widget.mapToGlobal(finish)),Qt.MouseButton.NoButton,
        Qt.MouseButton.LeftButton,Qt.KeyboardModifier.ControlModifier))
    QTest.qWait(20)
    QTest.mouseRelease(widget,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.ControlModifier,finish)
    QTest.keyRelease(widget,Qt.Key.Key_Control)
    until('window.loom.document.graph.nodes.find(n=>n.type==="Linear").x!=='+str(before_move))
    position=js('(() => {const n=window.loom.document.graph.nodes.find(n=>n.type==="Linear");return [n.x,n.y,n.collapsed];})()')
    assert position[0]%24==position[1]%24==0 and position[2] is True,position
    js('window.loom.perform("undo");')
    until('window.loom.document.graph.nodes.find(n=>n.type==="Linear").x==='+str(before_move))
    compact_center=block_center(icon_selector)
    native_window=view.windowHandle()
    QTest.mouseDClick(native_window,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier,
        native_window.mapFromGlobal(widget.mapToGlobal(compact_center)))
    until('window.loom.document.graph.nodes.find(n=>n.type==="Linear").collapsed===false')
    assert js('document.querySelector(".node.Linear .node-title")!==null')
    assert identity(h.workspace.models(),h.model['id'])==signature_before_fold
    js('window.loom.perform("undo");')
    until('window.loom.document.graph.nodes.find(n=>n.type==="Linear").collapsed===true')
    js('window.loom.perform("undo");')
    until('JSON.stringify(window.loom.document.graph)==='+json.dumps(graph_before_fold))
    # Bulk layout is a single visual/history edit; independent profiles do not
    # change shapes, bindings, code or checkpoint identity.
    before_bulk=js('JSON.stringify(window.loom.document.graph)')
    view_before_bulk=js('JSON.stringify(window.loom.document.view)')
    history_before=len(h.workspace.history.undo_stack)
    js('window.loom.perform("layout");')
    until('!document.getElementById("layout-panel").hidden')
    assert js('document.getElementById("layout-collapsed-section").open && !document.getElementById("layout-expanded-section").open')
    js('document.getElementById("layout-expanded-layer_gap").value="7";')
    QTest.mouseClick(widget,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier,block_center('#minimize-all'))
    until('window.loom.document.graph.nodes.every(n=>n.collapsed)')
    assert js('document.querySelector(".node.Input .node-compact-label").firstChild.nodeValue==="[B, 2, 3]"')
    assert js('document.querySelector(".node.Output .node-compact-label").firstChild.nodeValue==="[B, 2, 4]"')
    QTest.qWait(100)
    assert len(h.workspace.history.undo_stack)==history_before+1
    assert identity(h.workspace.models(),h.model['id'])==signature_before_fold
    small_doc=js('JSON.stringify(window.loom.document.graph)')
    small_extent=js('Math.max(...window.loom.document.graph.nodes.map(n=>n.x+LoomGraph.width(n)))')
    screenshot=os.environ.get('VTW_SMOKE_SCREENSHOT')
    if screenshot:
        QTest.qWait(100)
        assert view.grab().save(str(Path(screenshot).with_name(Path(screenshot).stem+'-collapsed.png')))
    js('document.getElementById("layout-collapsed-section").open=false;document.getElementById("layout-expanded-section").open=true;')
    js('document.getElementById("layout-expanded-layer_gap").value="80";')
    js('document.getElementById("expand-all").scrollIntoView({block:"nearest"})')
    QTest.mouseClick(widget,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier,block_center('#expand-all'))
    QTest.qWait(100)
    assert js('(() => {const p=document.getElementById("expand-all"),r=p.getBoundingClientRect();return document.elementFromPoint(r.left+r.width/2,r.top+r.height/2)===p;})()'), 'Notification text must not intercept a panel button'
    until('window.loom.document.graph.nodes.every(n=>n.collapsed===false)')
    QTest.qWait(100)
    assert len(h.workspace.history.undo_stack)==history_before+2
    assert identity(h.workspace.models(),h.model['id'])==signature_before_fold
    assert js('Math.max(...window.loom.document.graph.nodes.map(n=>n.x+LoomGraph.width(n)))')>small_extent
    if screenshot:
        QTest.qWait(100)
        assert view.grab().save(str(Path(screenshot).with_name(Path(screenshot).stem+'-expanded.png')))
    js('window.loom.perform("undo");')
    until('JSON.stringify(window.loom.document.graph)==='+json.dumps(small_doc))
    js('window.loom.perform("undo");')
    until('JSON.stringify(window.loom.document.graph)==='+json.dumps(before_bulk))
    # Viewport is now session-only: graph undo must not restore a stored camera.
    assert h.workspace.models()[h.model['id']]['view']=={'x':0,'y':0,'zoom':1}
    config_before=h.config.path.read_bytes() if h.config.path.exists() else None
    js('document.getElementById("layout-expanded-direction").value="TB";document.getElementById("layout-expanded-layer_gap").value="123";document.getElementById("save-layout-preferences").click();')
    until('window.loom.state.layout_preferences.expanded.layer_gap===123')
    assert h.workspace.layout_preferences['expanded']['direction']=='TB'
    assert h.workspace.layout_preferences['collapsed']['layer_gap']==48
    assert (h.config.path.read_bytes() if h.config.path.exists() else None)==config_before
    assert len(h.workspace.history.undo_stack)==history_before+1
    layout_path=h.config.path.parent/'layout.vtw';h.workspace.save(layout_path)
    loaded=Workspace.load(layout_path)
    assert loaded.layout_preferences==h.workspace.layout_preferences;loaded.db.close()
    js('window.loom.perform("undo");')
    until('window.loom.state.layout_preferences.expanded.layer_gap===80')
    assert js('document.getElementById("layout-expanded-direction").value')=='LR'
    js('window.loom.perform("redo");')
    until('window.loom.state.layout_preferences.expanded.layer_gap===123')
    js('window.loom.perform("reset-panels");')
    context_script = """(() => {
        const group=[...document.querySelectorAll('.node')].find(n=>n.dataset.id===%s);
        const r=group.querySelector('.node-title').getBoundingClientRect();
        return window.loom.contextAt(r.left+r.width/2,r.top+r.height/2);
    })()"""
    context = js(context_script % json.dumps(linear))
    assert context == {'id':linear,'modelId':h.model['id'],'canDelete':True,'canResize':True}, context
    # Empty canvas clicks/pan do not discard selection. Escape is two-stage:
    # clear selection, then hide panels and collapse the launcher, not reset geometry.
    QTest.mouseClick(widget,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier,QPoint(900,120))
    assert js('document.querySelector(".node.Linear").classList.contains("selected")')
    js('window.loom.perform("inspector");document.getElementById("canvas").dispatchEvent(new KeyboardEvent("keydown",{key:"Escape",bubbles:true}));')
    assert not js('document.querySelector(".node.Linear").classList.contains("selected")')
    assert js('!document.getElementById("inspector-panel").hidden')
    js('document.getElementById("canvas").dispatchEvent(new KeyboardEvent("keydown",{key:"Escape",bubbles:true}));')
    assert js('document.getElementById("inspector-panel").hidden')
    assert js('document.getElementById("menu-pack").classList.contains("collapsed")')
    js('window.loom.perform("reset-panels");')
    # Inspection is explicit and read-only, including when selection changes.
    before_inspection=js('JSON.stringify(window.loom.document.graph)')
    assert js(context_script % json.dumps(input_id))['id']==input_id
    js('window.loom.perform("inspect-block",'+json.dumps(linear)+','+json.dumps(h.model['id'])+');')
    assert js('document.querySelector(".node.Linear").classList.contains("selected")')
    assert js('!document.getElementById("inspector-panel").hidden')
    assert js('JSON.stringify(window.loom.document.graph)')==before_inspection
    js('window.loom.perform("reset-panels");window.loom.perform("inspect-block",'+json.dumps(linear)+',"other-model");')
    assert js('document.getElementById("inspector-panel").hidden')
    h.job=SimpleNamespace(mid=h.model['id'],isRunning=lambda:True)
    h.broadcast()
    until('window.loom.state.training_id !== null')
    assert js('document.getElementById("minimize-all").disabled && document.getElementById("expand-all").disabled')
    js('window.loom.perform("compact-all");window.loom.perform("expand-all");')
    assert js('document.querySelector(".node.Linear .node-resize").getAttribute("aria-disabled")==="true"')
    js("document.querySelector('.node.Linear .node-resize').dispatchEvent(new KeyboardEvent('keydown',{key:'ArrowRight',bubbles:true}));")
    assert not any('width' in n for n in js('window.loom.document.graph.nodes'))
    assert js(context_script % json.dumps(linear))['canDelete'] is False
    assert js(context_script % json.dumps(linear))['canResize'] is False
    js('window.loom.perform("about-building-block","add-linear");')
    assert js('document.getElementById("block-about-dialog").open')
    js('document.getElementById("block-about-dialog").close();')
    js('window.loom.perform("automatic-block-size",'+json.dumps(linear)+','+json.dumps(h.model['id'])+');')
    assert js('JSON.stringify(window.loom.document.graph)')==before_inspection
    js('window.loom.perform("inspect-block",'+json.dumps(linear)+','+json.dumps(h.model['id'])+');')
    assert js('!document.getElementById("inspector-panel").hidden')
    assert js('JSON.stringify(window.loom.document.graph)')==before_inspection
    js('window.loom.perform("reset-panels");')
    js('window.loom.perform("delete-block",'+json.dumps(linear)+','+json.dumps(h.model['id'])+');')
    assert any(n['id']==linear for n in h.workspace.models()[h.model['id']]['graph']['nodes'])
    h.job=None
    h.broadcast()
    until('window.loom.state.training_id === null')
    assert js('!document.getElementById("minimize-all").disabled && !document.getElementById("expand-all").disabled')
    # Later selection does not redirect the explicit deletion target.
    assert js(context_script % json.dumps(input_id))['id'] == input_id
    js('window.loom.perform("delete-block",'+json.dumps(linear)+',"other-model");')
    assert any(n['id']==linear for n in js('window.loom.document.graph.nodes'))
    js('window.loom.perform("delete-block",'+json.dumps(context['id'])+','+json.dumps(context['modelId'])+');')
    until('!window.loom.document.graph.nodes.some(n=>n.id==='+json.dumps(linear)+')')
    until('window.loom.state.history.undo')
    deleted=h.workspace.models()[h.model['id']]['graph']['nodes']
    assert not any(n['id']==linear for n in deleted)
    assert any(n['id']==input_id for n in deleted)
    js('window.loom.perform("undo");')
    until('window.loom.document.graph.nodes.some(n=>n.id==='+json.dumps(linear)+')')
    assert h.workspace.models()[h.model['id']]['graph']==h.model['graph']
    js('window.loom.perform("redo");')
    until('!window.loom.document.graph.nodes.some(n=>n.id==='+json.dumps(linear)+')')
    assert js(context_script % json.dumps(input_id))['canDelete'] is False
    js('window.loom.perform("undo");')
    until('window.loom.document.graph.nodes.some(n=>n.id==='+json.dumps(linear)+')')
    output_id = next(n['id'] for n in h.model['graph']['nodes'] if n['type']=='Output')
    js('window.loom.perform("delete-block",'+json.dumps(output_id)+','+json.dumps(h.model['id'])+');')
    until('window.loom.state.history.undo')
    until('document.getElementById("validation-message").classList.contains("error")')
    assert not any(n['id']==output_id for n in h.workspace.models()[h.model['id']]['graph']['nodes'])
    js('window.loom.perform("undo");')
    until('window.loom.document.graph.nodes.some(n=>n.id==='+json.dumps(output_id)+')')
    # Private draft with an empty destination: exercise real Chromium pointer
    # capture, not merely synthetic DOM events or a graph-helper unit test.
    fixture=json.loads(json.dumps(h.model))
    fixture['graph']['nodes'].append({'id':'reroute','type':'Activation','activation':'Tanh','x':570,'y':340})
    h.workspace.put(fixture);h.broadcast()
    until('window.loom.document.graph.nodes.some(n=>n.id==="reroute")')
    js('window.loom.perform("reset-panels");window.loom.fit();')
    QTest.qWait(100)

    def center(selector):
        p=js('(() => {const r=document.querySelector('+json.dumps(selector)+').getBoundingClientRect();return {x:r.left+r.width/2,y:r.top+r.height/2};})()')
        return QPoint(round(p['x']),round(p['y']))

    endpoint='.wire-end[data-from="'+linear+'"][data-to="'+output_id+'"]'
    target='.node[data-id="reroute"] .port-in'
    original=js('JSON.stringify(window.loom.document.graph)')
    start=center(endpoint);finish=center(target)
    QTest.mousePress(widget,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier,start)
    QTest.mouseMove(widget,finish,20)
    # No persisted edit until a valid pointerup.
    assert js('JSON.stringify(window.loom.document.graph)')==original
    QTest.mouseRelease(widget,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier,finish)
    moved='window.loom.document.graph.edges.some(e=>e.from==='+json.dumps(linear)+' && e.to==="reroute")'
    until(moved)
    until('window.loom.state.history.undo')
    assert any(e['from']==linear and e['to']=='reroute' for e in h.workspace.models()[h.model['id']]['graph']['edges'])
    js('window.loom.perform("undo");')
    until('JSON.stringify(window.loom.document.graph)==='+json.dumps(original))
    js('window.loom.perform("redo");');until(moved)
    js('window.loom.perform("undo");');until('JSON.stringify(window.loom.document.graph)==='+json.dumps(original))
    # Invalid/blank release and Escape cancel leave bytes and history unchanged.
    for cancel in [False,True]:
        start=center(endpoint);finish=QPoint(650,130)
        QTest.mousePress(widget,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier,start)
        QTest.mouseMove(widget,finish,20)
        if cancel:QTest.keyClick(widget,Qt.Key.Key_Escape)
        QTest.mouseRelease(widget,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier,finish)
        assert js('JSON.stringify(window.loom.document.graph)')==original
        assert js('!document.documentElement.classList.contains("loom-dragging") && document.getElementById("pending-wire").hasAttribute("hidden")')
    # Click selects a precise wire; D deletes it, and history restores it.
    QTest.mouseClick(widget,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier,center(endpoint))
    QTest.keyClick(widget,Qt.Key.Key_D)
    until('!window.loom.document.graph.edges.some(e=>e.from==='+json.dumps(linear)+' && e.to==='+json.dumps(output_id)+')')
    js('window.loom.perform("undo");');until('JSON.stringify(window.loom.document.graph)==='+json.dumps(original))
    # Select a box; typing D in an input is harmless, then D on canvas deletes.
    # Keep wire hit elements alive between clicks (legacy double-click removal).
    wire_center=js('''(() => {
        const path=document.querySelector('.wire[data-from="'''+linear+'''"][data-to="'''+output_id+'''"]');
        const p=path.getPointAtLength(path.getTotalLength()/2),q=new DOMPoint(p.x,p.y).matrixTransform(path.getScreenCTM());
        return {x:q.x,y:q.y};
    })()''')
    wire_point=QPoint(round(wire_center['x']),round(wire_center['y']))
    assert js('document.elementFromPoint('+str(wire_point.x())+','+str(wire_point.y())+').classList.contains("wire")'),wire_center
    # QWidget QTest clicks deliberately separate their timestamps. The QWindow
    # double-click overload emits two real QPA press/release pairs together,
    # preserving Chromium's own double-click detection (no DOM synthesis).
    native_window=view.windowHandle()
    QTest.mouseDClick(native_window,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier,
        native_window.mapFromGlobal(widget.mapToGlobal(wire_point)))
    until('!window.loom.document.graph.edges.some(e=>e.from==='+json.dumps(linear)+' && e.to==='+json.dumps(output_id)+')')
    js('window.loom.perform("undo");');until('JSON.stringify(window.loom.document.graph)==='+json.dumps(original))
    QTest.qWait(60)  # Composite the history-restored hit regions before clicking.
    QTest.mouseClick(widget,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier,center('.node[data-id="reroute"] .node-title'))
    until('document.querySelector(".node[data-id=reroute]").classList.contains("selected")')
    js('window.loom.perform("inspector");document.getElementById("node-activation").dispatchEvent(new KeyboardEvent("keydown",{key:"d",bubbles:true}));')
    assert js('window.loom.document.graph.nodes.some(n=>n.id==="reroute")')
    js('''(() => {
        const c=document.getElementById('canvas');
        for(const modifier of ['ctrlKey','metaKey','altKey'])c.dispatchEvent(new KeyboardEvent('keydown',{key:'d',bubbles:true,[modifier]:true}));
        c.dispatchEvent(new KeyboardEvent('keydown',{key:'d',bubbles:true,repeat:true}));
        const dialog=document.getElementById('confirm-dialog');dialog.showModal();
        c.dispatchEvent(new KeyboardEvent('keydown',{key:'d',bubbles:true}));dialog.close('cancel');
    })()''')
    assert js('window.loom.document.graph.nodes.some(n=>n.id==="reroute")')
    h.job=SimpleNamespace(mid=h.model['id'],isRunning=lambda:True);h.broadcast()
    until('window.loom.state.training_id !== null')
    js('document.getElementById("canvas").focus();')
    QTest.keyClick(widget,Qt.Key.Key_D)
    assert js('window.loom.document.graph.nodes.some(n=>n.id==="reroute")')
    # Busy graph cannot start or commit a connection drag either.
    start=center(endpoint);finish=center(target)
    QTest.mousePress(widget,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier,start)
    QTest.mouseMove(widget,finish,20)
    QTest.mouseRelease(widget,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier,finish)
    assert js('JSON.stringify(window.loom.document.graph)')==original
    h.job=None;h.broadcast();until('window.loom.state.training_id === null')
    QTest.keyClick(widget,Qt.Key.Key_D)
    until('!window.loom.document.graph.nodes.some(n=>n.id==="reroute")')
    js('window.loom.perform("undo");');until('JSON.stringify(window.loom.document.graph)==='+json.dumps(original))
    js('window.loom.perform("redo");');until('!window.loom.document.graph.nodes.some(n=>n.id==="reroute")')
    # Real DOM + bridge checks: graph validity is not dataset admission, and
    # legacy metrics cannot claim a verified current checkpoint.
    admitted=default_model()
    admitted['id']=h.model['id'];admitted['training']['epochs']=2
    admitted['dataset']={'x':[[i/10.,i/8.] for i in range(10)],'y':[[i/7.] for i in range(10)]}
    h.workspace.put(admitted);h.broadcast()
    until('window.loom.document.graph.nodes.length===3 && !document.getElementById("train").disabled')
    assert js('document.getElementById("binding-status").textContent==="Bound"')
    bad=h.workspace.models()[admitted['id']]
    bad['dataset_binding']['inputs'][0]['id']='previous-port';h.workspace.put(bad);h.broadcast()
    until('document.getElementById("binding-status").textContent==="Needs rebind"')
    assert js('document.getElementById("train").disabled && !document.getElementById("rebind-data").hidden && document.getElementById("dataset-bindings").open')
    assert js('document.getElementById("binding-columns").textContent.includes("previous-port") && document.getElementById("binding-columns").textContent.includes("input")')
    from unittest.mock import patch
    from PyQt6.QtWidgets import QMessageBox
    with patch('app.desktop.QMessageBox.question',return_value=QMessageBox.StandardButton.Yes):
        js('document.getElementById("rebind-data").click();')
        until('document.getElementById("binding-status").textContent==="Bound" && !document.getElementById("train").disabled')
    js('window.loom.perform("undo");')
    until('document.getElementById("binding-status").textContent==="Needs rebind"')
    js('window.loom.perform("redo");')
    until('document.getElementById("binding-status").textContent==="Bound"')
    legacy=h.workspace.models()[admitted['id']];legacy['result']={'mse':.25};h.workspace.put(legacy);h.broadcast()
    until('document.getElementById("training-state").textContent==="Unverified legacy result"')
    assert js('document.getElementById("loss-value").textContent.includes("Unverified") && document.getElementById("results").textContent.includes("Unverified legacy")')
    assert js('document.getElementById("code-panel").textContent.includes("fresh weights, not the trained checkpoint")')
    # Produce the trained fixture in the separate interpreter; the Qt host must
    # still import no Torch and uses only pure checkpoint compatibility checks.
    import subprocess,sys
    trained=h.workspace.models()[admitted['id']]
    request=json.dumps({'models':h.workspace.models(),'id':admitted['id']})
    source=('import json,sys,torch;torch.set_num_threads(1);from app.core import train_model;'
            'r=json.load(sys.stdin);print(json.dumps(train_model(r["models"],r["id"])))')
    torch_python=find_torch_python()
    assert torch_python,'The trained UI fixture needs a PyTorch interpreter.'
    proc=subprocess.run([torch_python,'-c',source],
                        input=request,text=True,capture_output=True,timeout=15)
    assert proc.returncode==0,proc.stderr
    trained['result']=json.loads(proc.stdout);h.workspace.put(trained);h.broadcast()
    until('document.getElementById("training-state").textContent==="Complete · current checkpoint"')
    assert js('document.getElementById("results").textContent.includes("post-update / eval")')
    assert 'torch' not in sys.modules
    assert not errors, errors
    screenshot = os.environ.get('VTW_SMOKE_SCREENSHOT')
    if screenshot:
        QTest.qWait(100)
        assert view.grab().save(screenshot)
    h.workspace.history.directory.cleanup()
    h.workspace.db.close()
    view.close()
    view.setPage(QWebEnginePage(view))
    page.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    profile.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
print(json.dumps({'real_webengine':'PASS','layouts':samples,'reset_origin':'PASS','mouse_resize_undo_redo':'PASS','mouse_compact_icon_drag_snap_restore':'PASS','bulk_layout_single_undo':'PASS','layout_preferences_persistence':'PASS','compact_dimensions_bias':'PASS','mouse_destination_rewire_undo_redo':'PASS','rewire_cancel':'PASS','D_box_and_wire_deletion':'PASS','D_typing_training_guards':'PASS','bias_state':'PASS','alphabetical_lists':'PASS',
    'clicked_block_delete_undo_redo':'PASS','training_guard':'PASS','binding_admission_rebind_undo_redo':'PASS','current_legacy_status':'PASS','console_errors':errors}))
