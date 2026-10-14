"""Real sandboxed editor: a CNN built only from simple pieces trains and exports; a ready-made
Conv2d/pooling copy expands in place into the same shapes and parameter count (vpcc checks 4-5).

Private Qt profile, config, in-memory workspace and bounded one-thread worker.
No user app/files/preferences are opened or changed.
"""
import json
import os
import tempfile
import time
from pathlib import Path

from PyQt6.QtCore import QObject, QEvent, QEventLoop, QTimer, QUrl, QCoreApplication
from PyQt6.QtWidgets import QApplication
from PyQt6.QtTest import QTest
from PyQt6.QtWebChannel import QWebChannel
from PyQt6.QtWebEngineCore import QWebEnginePage, QWebEngineProfile
from PyQt6.QtWebEngineWidgets import QWebEngineView

from app.config import Config
from app.core import Workspace, default_model, validate_model, generate_code, generate_predictor
from app.desktop import Bridge, Window, find_torch_python
from app.evidence import check_checkpoint, normalize_models

assert not os.environ.get('QTWEBENGINE_DISABLE_SANDBOX')
assert '--no-sandbox' not in os.environ.get('QTWEBENGINE_CHROMIUM_FLAGS', '')
app = QApplication(['qt-cnn-pieces-smoke'])
view = QWebEngineView(); view.resize(1200, 800)
profile = QWebEngineProfile(view)
errors = []


class Page(QWebEnginePage):
    def javaScriptConsoleMessage(self, level, message, line, source):
        if level == self.JavaScriptConsoleMessageLevel.ErrorMessageLevel:
            errors.append(f'{source}:{line}: {message}')


class Harness(QObject):
    state = Window.state
    describe = Window.describe
    dispatch = Window.dispatch
    training_complete = Window.training_complete

    def __init__(self, path):
        super().__init__()
        self.config = Config(path); self.workspace = Workspace()
        self.model = default_model('CNN from pieces')
        self.model['graph']['nodes'][0].update(shape=[1, 6, 5], axes=['C', 'H', 'W'])
        self.model['training'].update(epochs=450, lr=.03)
        self.workspace.put(self.model); self.job = None
        self.torch_python = find_torch_python()
        assert self.torch_python, 'A Torch interpreter is required for the actual worker gate.'
        self.rendering = {'mode': 'software'}; self.bridge = Bridge(self, view)
        self.events = []

    def recent_files(self): return []

    def broadcast(self, kind='state', payload=None, exclude=None):
        self.events.append(kind)
        if self.bridge != exclude:
            self.bridge.send(kind, self.state() if kind == 'state' else payload)


def js(script):
    loop = QEventLoop(); timer = QTimer(); timer.setSingleShot(True)
    timer.timeout.connect(loop.quit); values = []
    def receive(value):
        values.append(value); loop.quit()
    view.page().runJavaScript(script, receive); timer.start(3000); loop.exec(); timer.stop()
    assert values, 'JavaScript callback timeout'
    return values[0]


def until(script, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if js(script): return
        loop = QEventLoop(); QTimer.singleShot(20, loop.quit); loop.exec()
    raise AssertionError(f'UI condition timeout: {script}; errors: {errors}')


def inspect(nid):
    js(f'window.loom.perform("inspect-block",{json.dumps(nid)},window.loom.document.id)')


def last(kind):
    return js(f'window.loom.document.graph.nodes.filter(n=>n.type==="{kind}").at(-1).id')


def add(selector, kind):
    count = js(f'window.loom.document.graph.nodes.filter(n=>n.type==="{kind}").length')
    js(f'document.querySelector({json.dumps(selector)}).click()')
    until(f'window.loom.document.graph.nodes.filter(n=>n.type==="{kind}").length==={count + 1}')
    return last(kind)


def field(identifier, value):
    js(f'(()=>{{const n=document.getElementById({json.dumps(identifier)});n.value={json.dumps(value)};n.dispatchEvent(new Event("change"))}})()')


def chain(models, mid):
    """Block types from Input to Output along the single path."""
    graph = models[mid]['graph']; at, out = 'input', []
    while at != 'output':
        at = next(e['to'] for e in graph['edges'] if e['from'] == at)
        out.append(next(n['type'] for n in graph['nodes'] if n['id'] == at))
    return out


with tempfile.TemporaryDirectory() as folder:
    h = Harness(Path(folder) / 'config.json')
    page = Page(profile, view); view.setPage(page); channel = QWebChannel(page)
    channel.registerObject('host', h.bridge); page.setWebChannel(channel)
    url = QUrl.fromLocalFile(str(Path(__file__).resolve().parents[1] / 'app/web/index.html'))
    url.setFragment('model=' + h.model['id']); view.setUrl(url); view.show()
    mid = h.model['id']
    try:
        until('Boolean(window.loom?.document && document.querySelector(".node"))')
        js('document.querySelector("[data-open-panel=library-panel]").click()')
        assert js('document.getElementById("ready-made-heading").textContent') == 'READY-MADE · EXPAND INTO PIECES'
        headings = js('JSON.stringify([...document.querySelector("#library-scroll .sidebar-section").children].filter(e=>e.classList.contains("eyebrow")||e.matches(".palette,.palette-group")).map(e=>e.classList.contains("eyebrow")?"#"+e.textContent:e.querySelector("b").textContent))')
        headings = json.loads(headings); section = lambda h: headings[headings.index(h) + 1:next(i for i, x in enumerate(headings) if x.startswith('#') and i > headings.index(h))]
        assert section('#BUILDING BLOCKS')[:3] == ['Activation', 'Add', 'Architecture reference'] and 'Patches' in section('#BUILDING BLOCKS'), headings
        assert section('#READY-MADE · EXPAND INTO PIECES') == ['AdaptiveAvgPool2d', 'AvgPool2d', 'Conv2d', 'MaxPool2d', 'MLP'], headings
        assert section('#TENSOR SOURCES') == ['Constant', 'Tensor parameter'] and section('#PROBABILITY COMPONENTS') == ['Gaussian filter'], headings
        # Pieces: Input [1,6,5] -> Patches 3x3 -> Linear 3 -> Tanh -> Reshape 12,3 -> Permute 1,0 -> Reduce mean -> linear1 -> Output.
        inspect('input')
        add('[data-tensor-type=Patches]', 'Patches')
        until('document.querySelector(".node.Patches .node-shape")?.textContent.includes("[B, 4, 3, 9]")')
        add('#add-linear', 'Linear'); field('features', '3')
        until('document.querySelector(".node.Patches + .node.Linear, .node.Linear")!==null')
        js('document.getElementById("activation-select").value="Tanh";document.getElementById("add-activation").click()')
        until('window.loom.document.graph.nodes.some(n=>n.type==="Activation")')
        add('#add-reshape', 'Reshape'); field('sample-shape', '12, 3')
        until('JSON.stringify(window.loom.document.graph.nodes.find(n=>n.type==="Reshape").shape)==="[12,3]"')
        add('[data-tensor-type=Permute]', 'Permute'); field('op-dims', '1, 0')
        until('JSON.stringify(window.loom.document.graph.nodes.find(n=>n.type==="Permute").dims)==="[1,0]"')
        add('[data-tensor-type=Reduce]', 'Reduce')
        until('document.getElementById("status").textContent.startsWith("Graph valid")')
        assert js('document.getElementById("op-mode").tagName') == 'SELECT'
        models = h.workspace.models()
        assert chain(models, mid) == ['Patches', 'Linear', 'Activation', 'Reshape', 'Permute', 'Reduce', 'Linear', 'Output'], chain(models, mid)
        contract = validate_model(models, mid)
        assert contract['output_shape'] == [1] and contract['parameters'] == 9 * 3 + 3 + 3 + 1, contract['parameters']
        # Train with the Experiment panel, then export.
        js('document.querySelector("[data-open-panel=experiment-dock]").click()')
        js('document.getElementById("sample-count").value="128";document.getElementById("target-kind").value="spatial";document.getElementById("generate-data").click()')
        until('window.loom.document.dataset?.truth?.kind==="spatial"')
        until('!document.getElementById("train").disabled')
        js('document.getElementById("train").click()')
        until('Boolean(window.loom.document.result?.checkpoint_version)', timeout=60)
        until('!window.loom.state.training_id', timeout=10)
        result = h.workspace.models()[mid]['result']
        check_checkpoint(normalize_models(h.workspace.models(), mid), mid, result)
        assert 'trainingError' not in h.events and result['mse'] == result['mse']
        code = generate_code(h.workspace.models(), mid)
        assert 'class _Patches' in code and 'class _Reduce' in code and 'nn.Conv2d' not in code
        assert 'class _Patches' in generate_predictor(h.workspace.models(), mid, result)

        # Ready-made copy: Input -> Conv2d 3x3 (3 channels) -> Tanh -> AdaptiveAvgPool2d 1,1 -> Flatten -> linear1 -> Output.
        ready = default_model('CNN ready-made'); ready['graph']['nodes'][0].update(shape=[1, 6, 5], axes=['C', 'H', 'W'])
        h.workspace.put(ready); rid = ready['id']; h.broadcast()
        js(f'location.hash="model={rid}"'); until(f'window.loom?.document?.id==={json.dumps(rid)}')
        inspect('input')
        conv = add('[data-tensor-type=Conv2d]', 'Conv2d'); field('op-out_channels', '3')
        until(f'window.loom.document.graph.nodes.find(n=>n.id==={json.dumps(conv)}).out_channels===3')
        js('document.getElementById("activation-select").value="Tanh";document.getElementById("add-activation").click()')
        until('window.loom.document.graph.nodes.some(n=>n.type==="Activation")')
        pool = add('[data-tensor-type=AdaptiveAvgPool2d]', 'AdaptiveAvgPool2d')
        add('[data-tensor-type=Flatten]', 'Flatten')
        until('document.getElementById("status").textContent.startsWith("Graph valid")')
        before = validate_model(h.workspace.models(), rid)
        assert before['parameters'] == contract['parameters'], (before['parameters'], contract['parameters'])
        # Expand Conv2d with the Inspector button (one undo step), then the pooling with the context-menu command.
        inspect(conv); until('Boolean(document.getElementById("expand-block"))')
        js('document.getElementById("expand-block").click()')
        until('!window.loom.document.graph.nodes.some(n=>n.type==="Conv2d")')
        until('document.getElementById("status").textContent.startsWith("Graph valid")')
        js('window.loom.perform("undo")'); until(f'window.loom.document.graph.nodes.some(n=>n.id==={json.dumps(conv)})')
        assert not js('window.loom.document.graph.nodes.some(n=>n.type==="Patches")')
        js('window.loom.perform("redo")'); until('window.loom.document.graph.nodes.some(n=>n.type==="Patches")')
        inspect(pool); js('window.loom.perform("reset-panels");document.querySelectorAll(".panel").forEach(p=>p.hidden=true);window.loom.fit()'); QTest.qWait(200)
        at = js(f'(()=>{{const b=document.querySelector(\'.node[data-id="{pool}"] rect\').getBoundingClientRect();return window.loom.contextAt(b.x+b.width/2,b.y+b.height/2)}})()')
        assert at['id'] == pool and at['canExpand'] is True, at
        js(f'window.loom.perform("expand-block",{json.dumps(pool)},window.loom.document.id)')
        until('!window.loom.document.graph.nodes.some(n=>n.type==="AdaptiveAvgPool2d")')
        until('document.getElementById("status").textContent.startsWith("Graph valid")')
        models = h.workspace.models(); after = validate_model(models, rid)
        assert chain(models, rid) == ['Patches', 'Linear', 'Permute', 'Activation', 'Reshape', 'Reduce', 'Reshape', 'Flatten', 'Linear', 'Output'], chain(models, rid)
        assert (after['output_shape'], after['parameters']) == (before['output_shape'], before['parameters'])
        last_shapes = lambda c: [c['shapes'][n['id']]['output_shape'] for n in models[rid]['graph']['nodes'] if n['type'] == 'Flatten']
        assert last_shapes(after) == last_shapes(before) == [[3]], (last_shapes(after), last_shapes(before))
        # Expansion invalidates results like other structural edits: train the original first, expand, result gone.
        js('window.loom.perform("undo")'); until('window.loom.document.graph.nodes.some(n=>n.type==="AdaptiveAvgPool2d")')
        js('document.getElementById("generate-data").click()'); until('Boolean(window.loom.document.dataset)')
        until('!document.getElementById("train").disabled'); js('document.getElementById("train").click()')
        until('Boolean(window.loom.document.result?.checkpoint_version)', timeout=60); until('!window.loom.state.training_id', timeout=10)
        js(f'window.loom.perform("expand-block",{json.dumps(pool)},window.loom.document.id)')
        until('!window.loom.document.graph.nodes.some(n=>n.type==="AdaptiveAvgPool2d")')
        until('!window.loom.document.result')
        path = Path(folder) / 'pieces.vtw'; h.workspace.save(path); loaded = Workspace.load(path)
        try: assert loaded.models() == h.workspace.models()
        finally: loaded.db.close()
        import sys
        assert 'torch' not in sys.modules and not errors, errors
        if os.environ.get('VTW_CAPTURE_DIR'):
            js('window.loom.perform("reset-panels");window.loom.fit()'); QTest.qWait(400)
            assert view.grab().save(str(Path(os.environ['VTW_CAPTURE_DIR']) / 'cnn-pieces-expanded.png'))
        print(json.dumps({'ok': True, 'pieces_parameters': contract['parameters'], 'held_out_mse': result['mse'],
                          'expanded_equals_ready_made': True, 'console_errors': errors}))
    finally:
        if h.job and h.job.isRunning(): h.job.stop()
        if h.job: h.job.wait(10000)
        h.workspace.history.directory.cleanup()
        h.workspace.db.close()
        view.close(); view.setPage(QWebEnginePage(view)); page.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        profile.deleteLater(); QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
