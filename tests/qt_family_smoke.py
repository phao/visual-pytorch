"""Render a CLI-imported family instance in the real sandboxed editor.

Uses private configuration/workspace/profile; no user app or file is touched.
This verifies CONCRETE instance integration, not a family authoring GUI.
"""
import json
import os
import sys
import tempfile
import time
from pathlib import Path

from PyQt6.QtCore import QObject, QEventLoop, QTimer, QUrl, QEvent, QCoreApplication
from PyQt6.QtWidgets import QApplication
from PyQt6.QtWebChannel import QWebChannel
from PyQt6.QtWebEngineCore import QWebEnginePage, QWebEngineProfile
from PyQt6.QtWebEngineWidgets import QWebEngineView

from app.config import Config
from app.core import Workspace, validate_model
from app.desktop import Bridge, Window
from vpw_cli import apply_edits

assert not os.environ.get('QTWEBENGINE_DISABLE_SANDBOX')
assert '--no-sandbox' not in os.environ.get('QTWEBENGINE_CHROMIUM_FLAGS', '')
ROOT = Path(__file__).resolve().parents[1]
app = QApplication(['family-instance-smoke'])
view = QWebEngineView(); view.resize(1440, 900)
profile = QWebEngineProfile(view); errors = []


class Page(QWebEnginePage):
    def javaScriptConsoleMessage(self, level, message, line, source):
        if level == self.JavaScriptConsoleMessageLevel.ErrorMessageLevel:
            errors.append(f'{source}:{line}: {message}')


class Harness(QObject):
    state = Window.state
    describe = Window.describe
    dispatch = Window.dispatch

    def __init__(self, folder):
        super().__init__()
        self.config = Config(folder / 'config.json'); self.workspace = Workspace()
        family = json.loads((ROOT / 'examples/families/width-depth.json').read_text())
        result = apply_edits(self.workspace, [{'op': 'specialize-family', 'family': family,
                                              'bindings': {'hidden-width': 4, 'hidden-depth': 3}}])[0]
        self.mid = result['resolved']['root']; self.family = result['family']; self.specialization = result['specialization']
        self.workspace.save(folder / 'instance.vtw')
        self.workspace.db.close(); self.workspace = Workspace.load(folder / 'instance.vtw')
        self.job = None; self.torch_python = None; self.rendering = {}; self.bridge = Bridge(self, view)

    def recent_files(self): return []

    def broadcast(self, kind='state', payload=None, **kwargs):
        self.bridge.send(kind, self.state() if kind == 'state' else payload)


def js(source):
    loop = QEventLoop(); timer = QTimer(); timer.setSingleShot(True); timer.timeout.connect(loop.quit)
    values = []
    view.page().runJavaScript(source, lambda value: (values.append(value), loop.quit()))
    timer.start(3000); loop.exec(); timer.stop()
    assert values, 'JS callback timeout'
    return values[0]


def until(source):
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if js(source): return
        loop = QEventLoop(); QTimer.singleShot(20, loop.quit); loop.exec()
    raise AssertionError(f'UI timeout: {source}; errors: {errors}')


with tempfile.TemporaryDirectory() as directory:
    h = Harness(Path(directory))
    page = Page(profile, view); view.setPage(page)
    channel = QWebChannel(page); channel.registerObject('host', h.bridge); page.setWebChannel(channel)
    url = QUrl.fromLocalFile(str(ROOT / 'app/web/index.html')); url.setFragment('model=' + h.mid)
    view.setUrl(url); view.show()
    try:
        until('Boolean(window.loom?.document?.view.fitted && document.querySelectorAll(".node").length===10)')
        assert validate_model(h.workspace.models(), h.mid)['parameters'] == 77
        assert js('document.querySelectorAll(".node.Linear").length') == 5
        assert js('document.querySelectorAll(".node.Activation").length') == 3
        assert js('new Set(window.loom.document.graph.nodes.map(n=>n.x+","+n.y)).size') == 10
        js('document.querySelector(".node.Linear").dispatchEvent(new MouseEvent("dblclick",{bubbles:true}))')
        until('!document.getElementById("inspector-panel").hidden && document.getElementById("inspector-content")?.textContent.includes("bias")')
        assert h.describe(h.mid)['valid']
        assert 'nn.Linear(4, 4' in h.describe(h.mid)['code']
        assert not h.workspace.dirty, 'View fit/inspect must not dirty saved instance'
        before = h.workspace.db.serialize()
        js('document.querySelector("[data-panel=science]").click()')
        until('!document.getElementById("science-panel").hidden')
        assert js('document.getElementById("science-kind").value') == 'experiment'
        labels = js('[...document.getElementById("science-kind").options].map(o=>o.textContent)')
        assert labels == sorted(labels, key=str.casefold), labels
        for kind, ref in (('family', h.family), ('specialization', h.specialization)):
            js('document.getElementById("science-kind").value=' + json.dumps(kind) + ';document.getElementById("science-kind").dispatchEvent(new Event("change"))')
            until('JSON.parse(document.getElementById("science-document").textContent||"{}").reference?.kind===' + json.dumps(kind))
            displayed = js('JSON.parse(document.getElementById("science-document").textContent)')
            assert displayed['reference'] == ref
            if kind == 'family':
                assert displayed['counts']['symbols'] == 2
                assert displayed['capabilities']['tied_weights'] is False
            else:
                assert displayed['family'] == h.family
                assert displayed['counts']['resolved_bindings'] == 2
        assert h.workspace.db.serialize() == before, 'Exact revision browsing must be read-only'
        assert not errors, errors
        assert 'torch' not in sys.modules
        if os.environ.get('VTW_FAMILY_CAPTURE'):
            assert view.grab().save(os.environ['VTW_FAMILY_CAPTURE'])
        print(json.dumps({'ok': True, 'concrete_family_instance': True, 'blocks': 10,
                          'parameters': 77, 'console_errors': errors, 'host_torch': False,
                          'workspace_dirty': h.workspace.dirty, 'retained_family_browser': True,
                          'retained_specialization_browser': True, 'family_authoring_gui': False}))
    finally:
        view.setPage(QWebEnginePage(view)); page.deleteLater(); channel.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        app.processEvents(); profile.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        app.processEvents(); view.close()
        h.workspace.db.close()
        if h.workspace.history.directory is not None:
            h.workspace.history.directory.cleanup()
