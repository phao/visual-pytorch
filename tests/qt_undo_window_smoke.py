"""Actual desktop window: all tabs refresh, native menu and typing undo survive.

Private config/recent-file store only; sandbox on, no running-user app changes.
The large-data keyboard timing fixture is qt_undo_performance_smoke.py.
"""
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch
from PyQt6.QtCore import QEvent, QEventLoop, QTimer, QCoreApplication, Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication
from PyQt6.QtWebEngineCore import QWebEnginePage
from app.config import Config
from app.core import Workspace, default_model
from app.desktop import Window

assert not os.environ.get('QTWEBENGINE_DISABLE_SANDBOX')
assert '--no-sandbox' not in os.environ.get('QTWEBENGINE_CHROMIUM_FLAGS', '')
app = QApplication(['undo-window-smoke'])
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


with tempfile.TemporaryDirectory() as directory:
    folder = Path(directory); config = Config(folder / 'preferences.json'); config.save()
    with patch('app.desktop.Config', return_value=config), patch('app.desktop.QSettings', Settings), patch('app.desktop.QWebEnginePage', Page):
        window = Window(torch_python='private-test-no-worker')
        window.file_timer.stop(); window.show()
        view = window.current_view(); until(view, 'Boolean(window.loom)')
        window.workspace = Workspace(); model = default_model('Undo window')
        window.workspace.put(model); window.workspace.save(folder / 'fixture.vtw')
        window.broadcast(); view.bridge.send('navigate', model['id'])
        until(view, 'Boolean(window.loom?.document?.view.fitted)')
        for _ in range(2):
            window.open_tab(model['id'])
            until(window.current_view(), 'Boolean(window.loom?.document?.view.fitted)')
        view = window.current_view()
    try:
        js(view, 'document.querySelector(".node.Linear").dispatchEvent(new MouseEvent("dblclick",{bubbles:true}))')
        until(view, 'Boolean(document.getElementById("features"))')
        js(view, '(()=>{const f=document.getElementById("features");f.value="3";f.dispatchEvent(new Event("change"));})()')
        changed = 'window.loom.document.graph.nodes.find(n=>n.id==="linear1").features===3'
        original = 'window.loom.document.graph.nodes.find(n=>n.id==="linear1").features===1'
        for i in range(3): until(window.tabs.widget(i), changed)
        js(view, 'document.activeElement?.blur();document.getElementById("canvas").focus()')
        QTest.keyClick(view.focusProxy() or view, Qt.Key.Key_Z, Qt.KeyboardModifier.ControlModifier)
        for i in range(3): until(window.tabs.widget(i), original)
        assert window.redo_action.isEnabled()
        window.redo_action.trigger()
        for i in range(3): until(window.tabs.widget(i), changed)
        window.undo_action.trigger()
        for i in range(3): until(window.tabs.widget(i), original)
        js(view, 'document.querySelector(".node.Input").dispatchEvent(new MouseEvent("dblclick",{bubbles:true}))')
        until(view, 'Boolean(document.getElementById("port-label"))')
        before = window.workspace.db.serialize()
        js(view, 'document.getElementById("port-label").focus()')
        QTest.keyClicks(view.focusProxy() or view, 'typing')
        QTest.keyClick(view.focusProxy() or view, Qt.Key.Key_Z, Qt.KeyboardModifier.ControlModifier)
        QTest.qWait(100)
        assert window.workspace.db.serialize() == before
        assert [window.tabs.tabText(i) for i in range(3)] == ['Undo window'] * 3
        assert not errors, errors
        assert not any(n.split('.')[0] in ('torch', 'numpy') for n in sys.modules)
        print(json.dumps({'ok': True, 'actual_window': True, 'tabs': 3, 'keyboard_undo': True,
                          'native_menu_undo_redo': True, 'typing_undo_preserved': True,
                          'exact_tab_names': True, 'host_torch': False, 'console_errors': errors}))
    finally:
        window.workspace.dirty = False; window.close()
        for i in range(window.tabs.count()):
            current = window.tabs.widget(i); page = current.page()
            current.setPage(QWebEnginePage(current)); page.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
