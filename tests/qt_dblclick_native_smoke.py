"""Actual desktop window: a native double-click on a block opens the Inspector.

Synthetic DOM dblclick events bypass pointer capture; the real gesture is
retargeted to the canvas by the block-drag capture, which once left the
Inspector closed. Native input goes through the QWindow so Qt's own
double-click detection applies. Expanded and compact blocks run in separate
windows so click counting cannot leak between them.
Private config/settings only; sandbox on, no running-user app changes.
"""
import json
import os
import tempfile
import time
from pathlib import Path
from unittest.mock import patch
from PyQt6.QtCore import QCoreApplication, QEvent, QEventLoop, QPoint, QTimer, Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication
from PyQt6.QtWebEngineCore import QWebEnginePage
from app.config import Config
from app.core import Workspace, default_model
from app.desktop import Window

assert not os.environ.get('QTWEBENGINE_DISABLE_SANDBOX')
assert '--no-sandbox' not in os.environ.get('QTWEBENGINE_CHROMIUM_FLAGS', '')
app = QApplication(['dblclick-native-smoke'])
errors = []


class Page(QWebEnginePage):
    def javaScriptConsoleMessage(self, level, message, line, source):
        if level == self.JavaScriptConsoleMessageLevel.ErrorMessageLevel:
            errors.append(f'{source}:{line}:{message}')


class Settings:
    def __init__(self, *args): self.data = {}
    def value(self, key, default=None, **kwargs): return self.data.get(key, default)
    def setValue(self, key, value): self.data[key] = value


def js(view, source):
    values = []; loop = QEventLoop(); timer = QTimer(); timer.setSingleShot(True)
    timer.timeout.connect(loop.quit)
    view.page().runJavaScript(source, lambda value: (values.append(value), loop.quit()))
    timer.start(4000); loop.exec(); timer.stop()
    assert values, 'JavaScript callback timeout'
    return values[0]


def until(view, source):
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        assert not errors, errors
        if js(view, source): return
        QTest.qWait(20)
    raise AssertionError('UI timeout: ' + source)


def native_double_click(window, view, node_id):
    rect = js(view, f'(()=>{{const b=document.querySelector(".node[data-id={node_id}]").getBoundingClientRect();'
                    'return [b.x+b.width/2, b.y+b.height*0.75]})()')
    zoom = view.page().zoomFactor()
    point = view.mapTo(window, QPoint(int(rect[0] * zoom), int(rect[1] * zoom)))
    handle, left, none = window.windowHandle(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier
    for _ in range(2):
        QTest.mousePress(handle, left, none, point, 30); QTest.mouseRelease(handle, left, none, point, 30)


def check(compact):
    with tempfile.TemporaryDirectory() as directory:
        folder = Path(directory); config = Config(folder / 'preferences.json'); config.save()
        with patch('app.desktop.Config', return_value=config), patch('app.desktop.QSettings', Settings), patch('app.desktop.QWebEnginePage', Page):
            window = Window(torch_python='private-test-no-worker')
            window.file_timer.stop(); window.resize(1400, 900); window.show()
            view = window.current_view(); until(view, 'Boolean(window.loom)')
            window.workspace = Workspace(); model = default_model('Double click')
            window.workspace.put(model); window.workspace.save(folder / 'fixture.vtw')
            window.broadcast(); view.bridge.send('navigate', model['id'])
            until(view, 'Boolean(window.loom?.document?.view.fitted)')
        try:
            js(view, 'window.loom.perform("reset-panels")')
            if compact:
                js(view, 'window.loom.perform("compact-all")')
                until(view, 'window.loom.document.graph.nodes.every(n=>n.collapsed)')
            until(view, 'document.getElementById("inspector-panel").hidden')
            QTest.qWait(600)
            native_double_click(window, view, 'linear1')
            until(view, '!document.getElementById("inspector-panel").hidden && document.querySelector(".node.selected")?.dataset.id==="linear1"')
            until(view, 'document.getElementById("node-kind").textContent==="Linear"')
        finally:
            window.workspace.dirty = False; window.close()
            for i in range(window.tabs.count()):
                current = window.tabs.widget(i); page = current.page()
                current.setPage(QWebEnginePage(current)); page.deleteLater()
            QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
            window.deleteLater()
            QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


check(compact=False)
check(compact=True)
assert not errors, errors
print(json.dumps({'expanded_block_native_dblclick': 'PASS', 'compact_block_native_dblclick': 'PASS', 'console_errors': errors}))
