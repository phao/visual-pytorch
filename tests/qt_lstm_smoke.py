"""Real sandboxed Qt LSTM inspection, sockets, compact dragging and undo/reload.

Imported example is a drawing fixture, not evidence of a creation form. All
configuration/workspaces/profiles are private; the Qt host never imports Torch.
"""
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

from PyQt6.QtCore import QObject, QEvent, QEventLoop, QTimer, QUrl, QPoint, Qt, QCoreApplication
from PyQt6.QtGui import QMouseEvent
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication
from PyQt6.QtWebEngineCore import QWebEnginePage, QWebEngineProfile
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWebChannel import QWebChannel

from app.config import Config
from app.core import Workspace, validate_model
from app.desktop import Window, Bridge
from examples.projected_lstm import models

assert not os.environ.get('QTWEBENGINE_DISABLE_SANDBOX')
assert '--no-sandbox' not in os.environ.get('QTWEBENGINE_CHROMIUM_FLAGS', '')
app = QApplication(['lstm-smoke']); errors = []


class Page(QWebEnginePage):
    def javaScriptConsoleMessage(self, level, message, line, source):
        if level == self.JavaScriptConsoleMessageLevel.ErrorMessageLevel: errors.append(f'{source}:{line}: {message}')


class Harness(QObject):
    state = Window.state; describe = Window.describe; dispatch = Window.dispatch

    def __init__(self, path):
        super().__init__(); self.config = Config(path / 'config.json'); self.config.save(ui_zoom=1)
        self.workspace = Workspace(); collection, self.mid = models()
        self.workspace.put(collection[self.mid]); self.workspace.dirty = False
        self.job = None; self.torch_python = None; self.rendering = {}; self.bridges = []

    def recent_files(self): return []

    def broadcast(self, kind='state', payload=None, **kwargs):
        for bridge in self.bridges: bridge.send(kind, self.state() if kind == 'state' else payload)


def js(view, script):
    loop = QEventLoop(); values = []; timer = QTimer(); timer.setSingleShot(True); timer.timeout.connect(loop.quit)
    view.page().runJavaScript(script, lambda value: (values.append(value), loop.quit()))
    timer.start(4000); loop.exec(); timer.stop(); assert values, 'JS timeout'; return values[0]


def until(view, script, seconds=15):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if js(view, script): return
        QTest.qWait(25)
    diagnostic = js(view, '({node:window.loom.document.graph.nodes.find(n=>n.id==="lstm"),icon:document.querySelector(".node.LSTM .node-type-icon")?.outerHTML})')
    raise AssertionError(f'UI timeout: {script}; console={errors}; diagnostic={diagnostic}')


def point(view, selector):
    value = js(view, '(() => {const r=document.querySelector(' + json.dumps(selector) + ').getBoundingClientRect();return {x:r.left+r.width/2,y:r.top+r.height/2};})()')
    return QPoint(round(value['x']), round(value['y']))


with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary); host = Harness(root); view = QWebEngineView(); view.resize(1500, 980)
    profile = QWebEngineProfile(view); page = Page(profile, view); view.setPage(page)
    bridge = Bridge(host, view); host.bridges = [bridge]; channel = QWebChannel(page)
    channel.registerObject('host', bridge); page.setWebChannel(channel)
    url = QUrl.fromLocalFile(str(Path(__file__).resolve().parents[1] / 'app/web/index.html')); url.setFragment('model=' + host.mid)
    view.setUrl(url); view.show(); until(view, 'Boolean(window.loom?.document)')
    config = host.config.path.read_bytes()
    try:
        until(view, 'document.querySelector(".node.LSTM") !== null')
        js(view, 'window.loom.perform("reset-panels");window.loom.fit()'); QTest.qWait(80)
        widget = view.focusProxy() or view
        QTest.mouseDClick(widget, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, point(view, '.node.LSTM rect'))
        # QWidget mouseDClick emits the second press; complete its release so
        # the real canvas does not retain pointer capture across the next gesture.
        QTest.mouseRelease(widget, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, point(view, '.node.LSTM rect'))
        until(view, 'document.getElementById("inspector-content").textContent.includes("Projection narrows hidden, not cell")')
        text = js(view, 'document.getElementById("inspector-content").textContent')
        for required in ('Initial hidden', 'Initial cell', 'exact-int64-v1', '[B, 4, 4]', '[B, 4, 7]', 'Bias on'):
            assert required in text, (required, text)
        assert js(view, 'document.querySelectorAll(".node.LSTM .port-in").length') == 4
        assert js(view, 'document.querySelectorAll(".node.LSTM .port-out").length') == 3
        assert js(view, '[...document.querySelectorAll(".node.LSTM .port-out title")].map(t=>t.textContent).join(";")').count('Final') == 2
        js(view, 'window.loom.perform("reset-panels");window.loom.fit()'); QTest.qWait(250)
        location = point(view, '.node.LSTM .node-type-icon')
        hit = js(view, 'document.elementFromPoint(' + str(location.x()) + ',' + str(location.y()) + ')?.outerHTML')
        print(json.dumps(dict(icon_point=[location.x(), location.y()], icon_hit=hit)), flush=True)
        QTest.mouseClick(widget, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, point(view, '.node.LSTM .node-type-icon'))
        until(view, 'document.querySelector(".node.LSTM.collapsed") !== null')
        before = next(n for n in host.workspace.models()[host.mid]['graph']['nodes'] if n['id'] == 'lstm')
        start = point(view, '.node.LSTM .node-type-icon'); finish = start + QPoint(42, 33)
        QTest.mousePress(widget, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, start)
        app.sendEvent(widget, QMouseEvent(QEvent.Type.MouseMove, finish.toPointF(), widget.mapToGlobal(finish).toPointF(),
                                          Qt.MouseButton.NoButton, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier))
        QTest.qWait(40); QTest.mouseRelease(widget, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, finish)
        until(view, 'window.loom.document.graph.nodes.find(n=>n.id==="lstm").x!==' + str(before['x']))
        assert js(view, 'new Set([...document.querySelectorAll(".node.LSTM .port-in")].map(p=>p.getAttribute("cy"))).size') == 4
        assert js(view, 'document.querySelector(".node.LSTM .node-compact-bias").textContent') == 'Bias on'
        js(view, 'window.loom.call("undo")'); until(view, 'window.loom.document.graph.nodes.find(n=>n.id==="lstm").x===' + str(before['x']))
        # Space acts on the selection even when its icon is not focused.
        js(view, 'document.getElementById("canvas").focus()')
        QTest.keyClick(widget, Qt.Key.Key_Space)
        until(view, 'document.querySelector(".node.LSTM.collapsed") === null')
        QTest.keyClick(widget, Qt.Key.Key_Space)
        until(view, 'document.querySelector(".node.LSTM.collapsed") !== null')
        history = len(host.workspace.history.undo_stack)
        js(view, 'const typingProbe=document.createElement("textarea");typingProbe.id="space-typing-probe";document.body.append(typingProbe);typingProbe.focus()')
        QTest.keyClick(widget, Qt.Key.Key_Space); QTest.qWait(80)
        assert js(view, 'document.getElementById("space-typing-probe").value') == ' '
        QTest.keyClick(widget, Qt.Key.Key_H); QTest.qWait(50)
        assert js(view, 'document.getElementById("space-typing-probe").value') == ' h'
        assert js(view, 'document.getElementById("help-panel").hidden')
        assert js(view, 'Boolean(document.querySelector(".node.LSTM.collapsed"))')
        assert len(host.workspace.history.undo_stack) == history
        # Native Help toggling, control guards and resource dispatch. Never open
        # the user's browser for a test; intercept the OS launcher, not the bridge.
        js(view, 'document.getElementById("canvas").focus()')
        raw = host.workspace.db.serialize()
        QTest.keyClick(widget, Qt.Key.Key_H)
        until(view, '!document.getElementById("help-panel").hidden')
        assert js(view, 'document.querySelectorAll("#help-shortcuts tbody tr").length') == 24
        assert js(view, '!document.querySelector(".canvas-note")')
        assert js(view, 'document.getElementById("help-scroll").scrollHeight > document.getElementById("help-scroll").clientHeight')
        assert js(view, 'Object.keys(LoomPanels.validatePreset(window.loom.state.panel_layout) || {}).length') == 0
        route = js(view, 'location.hash')
        QTest.mouseClick(widget, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
                        point(view, '.help-sections a[href="#help-shortcuts"]'))
        QTest.qWait(80)
        assert js(view, 'location.hash') == route and js(view, '!document.getElementById("editor").hidden')
        assert js(view, 'document.getElementById("help-scroll").scrollTop > 0')
        # Small width / larger font must wrap, not hide table cells horizontally.
        js(view, 'document.documentElement.style.setProperty("--ui-font-size","18px");document.getElementById("help-panel").style.width="320px"')
        QTest.qWait(80)
        assert js(view, 'document.getElementById("help-scroll").scrollWidth <= document.getElementById("help-scroll").clientWidth + 1')
        start = point(view, '#help-panel [data-drag-handle]'); finish = start + QPoint(33, 27)
        QTest.mousePress(widget, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, start)
        app.sendEvent(widget, QMouseEvent(QEvent.Type.MouseMove, finish.toPointF(), widget.mapToGlobal(finish).toPointF(),
                                          Qt.MouseButton.NoButton, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier))
        QTest.mouseRelease(widget, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, finish)
        assert js(view, 'Boolean(document.getElementById("help-panel").style.left)')
        width = js(view, 'document.getElementById("help-panel").getBoundingClientRect().width')
        start = point(view, '#help-panel .panel-resize'); finish = start + QPoint(45, 25)
        QTest.mousePress(widget, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, start)
        app.sendEvent(widget, QMouseEvent(QEvent.Type.MouseMove, finish.toPointF(), widget.mapToGlobal(finish).toPointF(),
                                          Qt.MouseButton.NoButton, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier))
        QTest.mouseRelease(widget, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, finish)
        assert js(view, 'document.getElementById("help-panel").getBoundingClientRect().width') > width
        with patch('app.desktop.QDesktopServices.openUrl', return_value=True) as opener:
            js(view, 'document.querySelector("[data-help-resource=pytorch-basics]").click()')
            deadline=time.monotonic()+3
            while not opener.called and time.monotonic()<deadline: QTest.qWait(20)
            opener.assert_called_once()
            assert opener.call_args.args[0].toString() == 'https://docs.pytorch.org/tutorials/beginner/basics/intro.html'
            for bad in ({'action':'openHelpResource','id':'https://untrusted.example'},
                        {'action':'openHelpResource','id':'pytorch-basics','url':'file:///tmp/a'}):
                try: host.dispatch(bad, bridge)
                except ValueError: pass
                else: raise AssertionError('Untrusted resource was admitted')
            opener.assert_called_once()
        js(view, 'document.querySelector("#help-panel [data-expand-panel]").focus()')
        QTest.keyClick(widget, Qt.Key.Key_H); QTest.qWait(60)
        assert js(view, '!document.getElementById("help-panel").hidden')
        js(view, 'document.getElementById("canvas").focus()')
        QTest.keyClick(widget, Qt.Key.Key_H)
        until(view, 'document.getElementById("help-panel").hidden')
        QTest.keyClick(widget, Qt.Key.Key_H)
        until(view, '!document.getElementById("help-panel").hidden')
        js(view, 'document.querySelector("#help-panel [data-fold-panel]").click()')
        assert js(view, 'document.getElementById("help-panel").hidden')
        assert host.workspace.db.serialize() == raw and len(host.workspace.history.undo_stack) == history
        js(view, 'document.documentElement.style.removeProperty("--ui-font-size")')
        js(view, 'document.getElementById("space-typing-probe").remove();document.getElementById("validate").focus()')
        QTest.keyClick(widget, Qt.Key.Key_Space); QTest.qWait(80)
        assert js(view, 'Boolean(document.querySelector(".node.LSTM.collapsed"))')
        assert len(host.workspace.history.undo_stack) == history
        path = root / 'lstm.vtw'; host.workspace.save(path); opened = Workspace.load(path)
        try: assert validate_model(opened.models(), host.mid)['parameters'] == 1400
        finally: opened.db.close()
        assert host.config.path.read_bytes() == config and not errors, errors
        assert 'torch' not in sys.modules
        print(json.dumps(dict(lstm_inspection='PASS', native_compact_icon_drag='PASS', space_selection_toggle='PASS', space_typing_control_guards='PASS',
                              help_native_toggle_drag_table_wrap='PASS', help_control_guards='PASS', help_allowlisted_browser_dispatch='PASS', distinct_ports='PASS',
                              atomic_undo_vtw_reload='PASS', console_errors=errors, host_torch=False)))
    finally:
        host.bridges = []; view.close(); view.setPage(QWebEnginePage(view)); page.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete); profile.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        if host.workspace.history.directory: host.workspace.history.directory.cleanup()
        host.workspace.db.close()
