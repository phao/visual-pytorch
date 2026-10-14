"""Actual desktop window: one grouped activation list and the MLP building block.

Private config/settings and temporary workspaces only; sandbox on; no host Torch.
Set VTW_CAPTURE_DIR to save PNG captures.
"""
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch
from PyQt6.QtCore import QCoreApplication, QEvent, QEventLoop, QPoint, QTimer, Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication
from PyQt6.QtWebEngineCore import QWebEnginePage
from app.config import Config
from app.core import Workspace
from app.desktop import Window
from app.core import default_model

assert not os.environ.get('QTWEBENGINE_DISABLE_SANDBOX')
assert '--no-sandbox' not in os.environ.get('QTWEBENGINE_CHROMIUM_FLAGS', '')
app = QApplication(['mlp-block-smoke'])
errors = []
captures = Path(os.environ['VTW_CAPTURE_DIR']) if os.environ.get('VTW_CAPTURE_DIR') else None


class Page(QWebEnginePage):
    def javaScriptConsoleMessage(self, level, message, line, source):
        if level == self.JavaScriptConsoleMessageLevel.ErrorMessageLevel:
            errors.append(f'{source}:{line}:{message}')


class Settings:
    def __init__(self, *args): self.data = {}
    def value(self, key, default=None, **kwargs): return self.data.get(key, default)
    def setValue(self, key, value): self.data[key] = value


def js(source):
    values = []; loop = QEventLoop(); timer = QTimer(); timer.setSingleShot(True)
    timer.timeout.connect(loop.quit)
    view.page().runJavaScript(source, lambda value: (values.append(value), loop.quit()))
    timer.start(6000); loop.exec(); timer.stop()
    assert values, 'JavaScript callback timeout: ' + source[:80]
    return values[0]


def until(source, timeout=20):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        assert not errors, errors
        if js(source): return
        QTest.qWait(25)
    raise AssertionError('UI timeout: ' + source + ' · pill: ' + str(js('document.getElementById("validation-message").textContent')) +
                         ' · edges: ' + str(js('JSON.stringify(window.loom.document.graph.edges)')))


def grab(name):
    if captures:
        captures.mkdir(parents=True, exist_ok=True); QTest.qWait(400)
        view.grab().save(str(captures / (name + '.png')))


def native_click(selector, fx=0.5, fy=0.5):
    rect = js(f'(()=>{{const b=document.querySelector({json.dumps(selector)}).getBoundingClientRect();return [b.x+b.width*{fx},b.y+b.height*{fy}]}})()')
    zoom = view.page().zoomFactor()
    point = view.mapTo(window, QPoint(int(rect[0] * zoom), int(rect[1] * zoom)))
    handle, left, none = window.windowHandle(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier
    QTest.mousePress(handle, left, none, point, 20); QTest.mouseRelease(handle, left, none, point, 30)
    QTest.qWait(450)  # also outlasts the double-click interval between separate gestures


def shape_text(nid):
    return js(f'document.querySelector(\'.node[data-id="{nid}"] .node-shape\')?.firstChild?.nodeValue || ""')


def state():
    return js('JSON.stringify([window.loom.state.dirty, window.loom.state.history, window.loom.document.graph.nodes.map(n=>n.id)])')


def open_model(doc):
    window.workspace = Workspace(); window.workspace.put(doc); window.workspace.save(folder / (doc['id'] + '.vtw'))
    window.broadcast(); view.bridge.send('navigate', doc['id'])
    until(f'window.loom?.document?.id==={json.dumps(doc["id"])} && Boolean(window.loom.document.view.fitted)')
    js('window.loom.perform("reset-panels");window.loom.fit()'); QTest.qWait(300)



def status():
    return js('document.getElementById("status").textContent')


with tempfile.TemporaryDirectory() as directory:
    folder = Path(directory); config = Config(folder / 'preferences.json'); config.save()
    with patch('app.desktop.Config', return_value=config), patch('app.desktop.QSettings', Settings), \
         patch('app.desktop.QWebEnginePage', Page):
        window = Window(torch_python='private-test-no-worker')
        window.file_timer.stop(); window.resize(1440, 900); window.show()
        view = window.current_view(); until('Boolean(window.loom)')
        try:
            doc = default_model('MLP block', inputs=4, outputs=4)
            open_model(doc)
            js('window.loom.perform("library")'); until('!document.getElementById("library-panel").hidden')
            groups = 'JSON.stringify([...document.querySelectorAll("#SEL optgroup")].map(g=>[g.label,[...g.querySelectorAll("option")].map(o=>o.value)]))'
            palette = js(groups.replace('SEL', 'activation-select'))
            assert json.loads(palette) == [['Elementwise', ['GELU', 'ReLU', 'Sigmoid', 'Softplus', 'Tanh']], ['Normalizing along an axis', ['LogSoftmax', 'Softmax']]], palette
            assert js(groups.replace('SEL', 'mlp-activation')) == palette
            # Check 2: each of the seven, added from Building blocks after linear1, validates; check 1 on its Inspector.
            for name in ('GELU', 'ReLU', 'Sigmoid', 'Softplus', 'Tanh', 'LogSoftmax', 'Softmax'):
                js('window.loom.perform("inspect-block","linear1",window.loom.document.id)')
                js(f'(()=>{{const s=document.getElementById("activation-select");s.value="{name}";s.dispatchEvent(new Event("change"));document.getElementById("add-activation").click();}})()')
                until(f'window.loom.document.graph.nodes.some(n=>n.activation==="{name}")')
                until('document.getElementById("status").textContent.startsWith("Graph valid")')
                act = js(f'window.loom.document.graph.nodes.find(n=>n.activation==="{name}").id')
                js(f'window.loom.perform("inspect-block",{json.dumps(act)},window.loom.document.id)')
                until('Boolean(document.getElementById("node-activation"))')
                assert js(groups.replace('SEL', 'node-activation')) == palette, name
                window.undo_action.trigger(); until('window.loom.document.graph.nodes.length===3')
            # Check 4: MLP after a selected Linear (width 4), typed with native keys.
            js('window.loom.perform("inspect-block","linear1",window.loom.document.id)')
            js('document.getElementById("add-mlp").click()'); until('!document.getElementById("mlp-form").hidden')
            js('document.getElementById("mlp-activation").value="GELU";document.getElementById("mlp-widths").focus()')
            QTest.keyClicks(view.focusProxy() or view, '4, 3*8, 2'); QTest.keyClick(view.focusProxy() or view, Qt.Key.Key_Return)
            until('window.loom.document.graph.nodes.length===10')
            until('document.getElementById("status").textContent.startsWith("Graph valid")')
            chain = js('(()=>{const d=window.loom.document,out=[];let at="linear1";while(at!=="output"){at=d.graph.edges.find(e=>e.from===at).to;const n=d.graph.nodes.find(m=>m.id===at);out.push(n.type==="Linear"?"L"+n.features:n.type==="Activation"?n.activation:n.type)}return out.join(",")})()')
            assert chain == 'L8,GELU,L8,GELU,L8,GELU,L2,Output', chain
            assert status().startswith('Graph valid · ' + str(5 * 4 + 5 * 8 + 2 * 9 * 8 + 9 * 2) + ' parameters'), status()
            grab('mlp-inserted')
            window.undo_action.trigger(); until('window.loom.document.graph.nodes.length===3')
            # Check 6: mismatch and limit give clear errors and change nothing.
            before = js('JSON.stringify(window.loom.document.graph)')
            js('window.loom.perform("inspect-block","linear1",window.loom.document.id)')
            js('document.getElementById("add-mlp").click();document.getElementById("mlp-widths").value="3, 2";document.querySelector("#mlp-form [type=submit]").click()')
            assert 'outputs 4, not 3' in js('document.getElementById("mlp-error").textContent')
            js('document.getElementById("mlp-widths").value="4, 40*2";document.querySelector("#mlp-form [type=submit]").click()')
            assert 'the limit is 64' in js('document.getElementById("mlp-error").textContent')
            assert js('JSON.stringify(window.loom.document.graph)') == before
            js('document.getElementById("mlp-cancel").click()')
            # Check 5: nothing selected -> new Input [n0] -> chain -> Output branch.
            js('document.activeElement?.blur();document.getElementById("canvas").focus()')
            QTest.keyClick(view.focusProxy() or view, Qt.Key.Key_Escape); QTest.qWait(150)
            js('document.getElementById("add-mlp").click();document.getElementById("mlp-widths").value="2, 10*3, 2";document.getElementById("mlp-activation").value="Tanh";document.querySelector("#mlp-form [type=submit]").click()')
            until('window.loom.document.graph.nodes.some(n=>n.label==="MLP input 1")')
            until('document.getElementById("status").textContent.startsWith("Graph valid")')
            widths = [2] + [3] * 10 + [2]
            expected = 5 * 4 + sum((a + 1) * b for a, b in zip(widths, widths[1:]))
            assert status().startswith(f'Graph valid · {expected} parameters'), status()
            js('window.loom.fit()'); grab('mlp-branch')
            # About page with its PyTorch links.
            js('window.loom.perform("about-building-block","add-mlp")'); until('document.getElementById("block-about-dialog").open')
            assert js('[...document.querySelectorAll("#block-about-content [data-help-resource]")].map(a=>a.textContent).join()') == 'torch.nn.Sequential,torch.nn.Linear'
            js('document.getElementById("block-about-dialog").close()')
            assert not errors, errors
            print(json.dumps({'ok': True, 'console_errors': errors, 'host_torch': 'torch' in sys.modules}))
        finally:
            window.workspace.dirty = False; window.close()
            for i in range(window.tabs.count()):
                current = window.tabs.widget(i); page = current.page()
                current.setPage(QWebEnginePage(current)); page.deleteLater()
            QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
            window.deleteLater()
            QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
