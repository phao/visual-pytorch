"""Actual desktop window: comfortable incomplete drafts (partial shapes, located problems,
loose blocks, selected box to front), with native mouse input.

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
from test_incomplete_drafts import reported_draft

assert not os.environ.get('QTWEBENGINE_DISABLE_SANDBOX')
assert '--no-sandbox' not in os.environ.get('QTWEBENGINE_CHROMIUM_FLAGS', '')
app = QApplication(['incomplete-drafts-smoke'])
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


with tempfile.TemporaryDirectory() as directory:
    folder = Path(directory); config = Config(folder / 'preferences.json'); config.save()
    with patch('app.desktop.Config', return_value=config), patch('app.desktop.QSettings', Settings), \
         patch('app.desktop.QWebEnginePage', Page):
        window = Window(torch_python='private-test-no-worker')
        window.file_timer.stop(); window.resize(1440, 900); window.show()
        view = window.current_view(); until('Boolean(window.loom)')
        try:
            # 1-2: the reported draft keeps every determinable shape and locates both problems.
            models, mid = reported_draft(); draft = models[mid]
            open_model(draft)
            until('document.getElementById("validation-message").textContent.includes("has no incoming wire")')
            for nid in ('h', 't', 'c', 'h2', 't2', 'r'):
                assert shape_text(nid) and '?' not in shape_text(nid), (nid, shape_text(nid))
            assert shape_text('h2') == '[B, 18] → [B, 16]', shape_text('h2')
            pill = js('document.getElementById("validation-message").textContent')
            assert pill == "⚠ Output 'output' has no incoming wire. · +1 more", pill
            assert js('[...document.querySelectorAll(".node.problem")].map(g=>g.dataset.id).sort().join()') == 'output,r'
            assert js('getComputedStyle(document.querySelector(\'.node[data-id="output"] rect\')).strokeDasharray') == '6px, 3px'  # outline visible over theme colours
            assert js('document.getElementById("status").textContent') == "Output 'output' has no incoming wire."
            grab('draft-located')
            native_click('#validation-message')
            until('document.querySelector(".node.selected")?.dataset.id==="output" && !document.getElementById("inspector-panel").hidden')
            # 6: running and exporting stay disabled while invalid.
            assert js('document.getElementById("generate-data").disabled && document.getElementById("export-copy").disabled')

            # 3: a block with two incoming wires, named by its display name.
            two = json.loads(json.dumps(draft)); two['id'] = 'two_inputs'
            two['graph']['edges'] += [{'from': 'r', 'to': 'output'}, {'from': 'h', 'to': 'h2'}]
            open_model(two)
            until('document.getElementById("validation-message").textContent.includes("incoming wires")')
            pill = js('document.getElementById("validation-message").textContent')
            assert pill.startswith('⚠ Linear') and pill.endswith('has 2 incoming wires; use Concat or Add to join branches.'), pill
            assert js('[...document.querySelectorAll(".node.problem")].map(g=>g.dataset.id).join()') == 'h2'
            assert '?' not in shape_text('c') and '?' in shape_text('h2'), (shape_text('c'), shape_text('h2'))  # upstream kept, downstream undetermined

            # 4-5: loose blocks with nothing selected; ordinary insertion with a selection.
            valid = json.loads(json.dumps(draft)); valid['id'] = 'valid'
            valid['graph']['edges'].append({'from': 'r', 'to': 'output'})
            open_model(valid)
            until('document.getElementById("validation-message").textContent.startsWith("✓ Valid")')
            js('window.loom.perform("inspect-block","h",window.loom.document.id)')
            js('(()=>{const s=document.getElementById("activation-select");s.value="ReLU";s.dispatchEvent(new Event("change"));document.getElementById("add-activation").click();})()')
            relu = js('window.loom.document.graph.nodes.at(-1).id')
            edges = js('JSON.stringify(window.loom.document.graph.edges)')
            assert json.dumps({'from': 'h', 'to': relu}, separators=(',', ':')) in edges and json.dumps({'from': relu, 'to': 't'}, separators=(',', ':')) in edges
            window.undo_action.trigger(); until('window.loom.document.graph.nodes.length===8')
            js('document.activeElement?.blur();document.getElementById("canvas").focus()')
            QTest.keyClick(view.focusProxy() or view, Qt.Key.Key_Escape); QTest.qWait(200)
            assert js('!document.querySelector(".node.selected")')
            js('document.getElementById("add-linear").click()')
            until('window.loom.document.graph.nodes.length===9')
            loose = js('window.loom.document.graph.nodes.at(-1).id')
            assert js(f'window.loom.document.graph.edges.every(e=>e.from!=={json.dumps(loose)}&&e.to!=={json.dumps(loose)})')
            until('document.getElementById("validation-message").textContent.includes("Linear has no incoming wire")')
            assert js(f'(()=>{{const b=document.querySelector(\'.node[data-id="{loose}"]\').getBoundingClientRect(),c=document.getElementById("canvas").getBoundingClientRect();return b.left>=c.left&&b.right<=c.right&&b.top>=c.top&&b.bottom<=c.bottom}})()')
            grab('loose-block')
            # Wire it in with native clicks: r -> loose, then loose -> output (replaces r -> output).
            js('window.loom.perform("reset-panels")'); QTest.qWait(200)  # keep sockets uncovered
            native_click('.node[data-id="r"] .port-out')
            native_click(f'.node[data-id="{loose}"] .port-in')
            until(f'window.loom.document.graph.edges.some(e=>e.from==="r"&&e.to==={json.dumps(loose)})')
            native_click(f'.node[data-id="{loose}"] .port-out'); native_click('.node[data-id="output"] .port-in')
            until('document.getElementById("validation-message").textContent.startsWith("✓ Valid")')
            assert js('document.getElementById("status").textContent').startswith('Graph valid')
            history = js('window.loom.state.history')
            window.undo_action.trigger(); window.undo_action.trigger()
            until('!document.getElementById("validation-message").textContent.startsWith("✓ Valid")')
            window.undo_action.trigger(); until('window.loom.document.graph.nodes.length===8')
            assert history

            # 7: clicking a partly covered box draws it on top; order and history unchanged.
            covered = json.loads(json.dumps(valid)); covered['id'] = 'covered'
            for n in covered['graph']['nodes']:
                n['x'], n['y'] = {'h': (300, 200), 't': (340, 220)}.get(n['id'], (90 + 260 * len(n['id']), 420))
            open_model(covered)
            js('window.loom.perform("inspect-block","t",window.loom.document.id);window.loom.perform("reset-panels")')
            until('[...document.getElementById("nodes").children].at(-1).dataset.id==="t"')  # t covers h
            before = state()
            native_click('.node[data-id="h"] rect', 0.12, 0.2)  # the top-left corner, not covered by t
            until('[...document.getElementById("nodes").children].at(-1).dataset.id==="h"')
            assert state() == before, (state(), before)
            grab('raised')
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
