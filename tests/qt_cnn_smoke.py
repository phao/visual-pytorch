"""Real sandboxed editor constructs a CNN, edits it, generates data and trains.

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
from PyQt6.QtWebChannel import QWebChannel
from PyQt6.QtWebEngineCore import QWebEnginePage, QWebEngineProfile
from PyQt6.QtWebEngineWidgets import QWebEngineView

from app.config import Config
from app.core import Workspace, default_model, validate_model, generate_predictor
from app.desktop import Bridge, Window, find_torch_python
from app.evidence import check_checkpoint, normalize_models

assert not os.environ.get('QTWEBENGINE_DISABLE_SANDBOX')
assert '--no-sandbox' not in os.environ.get('QTWEBENGINE_CHROMIUM_FLAGS', '')
app = QApplication(['qt-cnn-smoke'])
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
        self.model = default_model('CNN built in editor')
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


with tempfile.TemporaryDirectory() as folder:
    h = Harness(Path(folder) / 'config.json')
    page = Page(profile, view); view.setPage(page); channel = QWebChannel(page)
    channel.registerObject('host', h.bridge); page.setWebChannel(channel)
    url = QUrl.fromLocalFile(str(Path(__file__).resolve().parents[1] / 'app/web/index.html'))
    url.setFragment('model=' + h.model['id']); view.setUrl(url); view.show()
    try:
        until('Boolean(window.loom?.document && document.querySelector(".node"))')
        js('document.querySelector("[data-open-panel=library-panel]").click()')
        js('document.querySelector(".node.Input").dispatchEvent(new MouseEvent("dblclick",{bubbles:true}))')
        until('document.getElementById("input-axes")?.value === "C, H, W"')
        js('document.querySelector("[data-tensor-type=Conv2d]").click()')
        until('window.loom.document.graph.nodes.some(n=>n.type==="Conv2d")')
        js('document.querySelector(".node.Conv2d").dispatchEvent(new MouseEvent("dblclick",{bubbles:true}))')
        until('Boolean(document.getElementById("op-kernel_size"))')
        js('document.getElementById("op-kernel_size").value="3, 2"; document.getElementById("op-kernel_size").dispatchEvent(new Event("change"))')
        until('document.querySelector(".node.Conv2d .node-shape")?.textContent.includes("4, 4")')
        conv = next(n for n in h.workspace.models()[h.model['id']]['graph']['nodes'] if n['type'] == 'Conv2d')
        assert conv['kernel_size'] == [3, 2]
        assert js('document.getElementById("inspector-content").textContent.includes("Cross-correlation")')
        js('document.getElementById("activation-select").value="Tanh";document.getElementById("add-activation").click()')
        until('window.loom.document.graph.nodes.some(n=>n.type==="Activation")')
        js('document.querySelector("[data-tensor-type=AdaptiveAvgPool2d]").click()')
        until('window.loom.document.graph.nodes.some(n=>n.type==="AdaptiveAvgPool2d")')
        js('document.querySelector("[data-tensor-type=Flatten]").click()')
        until('document.querySelector(".node.Linear .node-shape")?.textContent.includes("[B, 2]")')
        mid = h.model['id']; contract = validate_model(h.workspace.models(), mid)
        assert contract['output_shape'] == [1]; assert contract['parameters'] == 17
        js('window.loom.perform("undo")')
        until('!window.loom.document.graph.nodes.some(n=>n.type==="Flatten")')
        js('window.loom.perform("redo")')
        until('window.loom.document.graph.nodes.some(n=>n.type==="Flatten")')
        js('document.querySelector(".node.Conv2d .node-type-icon").dispatchEvent(new MouseEvent("click",{bubbles:true}))')
        until('Boolean(document.querySelector(".node.Conv2d.collapsed .node-compact-bias"))')
        assert js('document.querySelector(".node.Conv2d .node-compact-bias").textContent') == 'Bias on'
        js('document.querySelector("[data-open-panel=experiment-dock]").click()')
        js('document.getElementById("sample-count").value="128";document.getElementById("target-kind").value="spatial";document.getElementById("generate-data").click()')
        until('window.loom.document.dataset?.truth?.kind==="spatial"')
        assert h.workspace.models()[mid]['dataset']['adapter']['sample_shape'] == [1, 6, 5]
        until('!document.getElementById("train").disabled')
        js('document.getElementById("train").click()')
        until('Boolean(window.loom.document.result?.checkpoint_version)', timeout=30)
        until('!window.loom.state.training_id', timeout=10)
        result = h.workspace.models()[mid]['result']
        check_checkpoint(normalize_models(h.workspace.models(), mid), mid, result)
        data = h.workspace.models()[mid]['dataset']; train = result['split_indices']['train']; held = result['split_indices']['held_out']
        mean = sum(data['y'][i][0] for i in train) / len(train)
        baseline = sum((data['y'][i][0] - mean) ** 2 for i in held) / len(held)
        assert result['mse'] < baseline * .1, (result['mse'], baseline)
        js('document.querySelector("[data-panel=results]").click()')
        until('document.getElementById("results").textContent.includes("Conv2d")')
        assert 'trainingError' not in h.events
        assert h.workspace.db.execute('SELECT count(*) FROM runs').fetchone()[0] == 1
        path = Path(folder) / 'editor-cnn.vtw'; h.workspace.save(path); loaded = Workspace.load(path)
        try:
            assert loaded.models() == h.workspace.models()
            assert 'nn.Conv2d' in generate_predictor(loaded.models(), mid, loaded.models()[mid]['result'])
        finally: loaded.db.close()
        assert not errors, errors
        if os.environ.get('VTW_SMOKE_SCREENSHOT'):
            assert view.grab().save(os.environ['VTW_SMOKE_SCREENSHOT'])
        print(json.dumps({'real_webengine_cnn': 'PASS', 'palette_construct_edit_undo_redo': 'PASS',
                          'compact_bias': 'PASS', 'spatial_adapter': 'PASS', 'actual_worker_training': 'PASS',
                          'held_out_mse': result['mse'], 'train_mean_baseline': baseline,
                          'checkpoint_reload_trained_export': 'PASS', 'console_errors': errors}))
    finally:
        if h.job and h.job.isRunning(): h.job.stop()
        if h.job: h.job.wait(10000)
        h.workspace.history.directory.cleanup()
        h.workspace.db.close()
        view.close(); view.setPage(QWebEnginePage(view)); page.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        profile.deleteLater(); QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
