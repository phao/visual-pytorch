"""Actual sandboxed Qt Inspector sharing and owned tied-family editing."""
import copy
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from PyQt6.QtCore import QObject, QCoreApplication, QEvent, QEventLoop, QTimer, QUrl
from PyQt6.QtWidgets import QApplication
from PyQt6.QtWebChannel import QWebChannel
from PyQt6.QtWebEngineCore import QWebEnginePage, QWebEngineProfile
from PyQt6.QtWebEngineWidgets import QWebEngineView
from app.config import Config
from app.core import Workspace, validate_model
from app.desktop import Bridge, Window
from test_shared_parameters import serial, tied_family

assert not os.environ.get('QTWEBENGINE_DISABLE_SANDBOX')
assert '--no-sandbox' not in os.environ.get('QTWEBENGINE_CHROMIUM_FLAGS', '')
app = QApplication(['shared-parameters-smoke']); view = QWebEngineView(); view.resize(1440, 1000)
profile = QWebEngineProfile(view); errors = []


class Page(QWebEnginePage):
    def javaScriptConsoleMessage(self, level, message, line, source):
        if level == self.JavaScriptConsoleMessageLevel.ErrorMessageLevel: errors.append(f'{source}:{line}: {message}')


class Harness(QObject):
    state = Window.state; describe = Window.describe
    def __init__(self, folder):
        super().__init__(); self.config = Config(folder / 'config.json'); self.workspace = Workspace()
        self.workspace.put(serial(shared=False)['shared'])
        family = tied_family(2); family['version'] = 1; family['repeats'][0]['mode'] = 'independent'
        self.family_ref = self.workspace.put_spec('family', family); self.workspace.save(folder / 'original.vtw')
        self.job = None; self.torch_python = None; self.rendering = {}; self.requests = []; self.bridge = Bridge(self, view)
    def recent_files(self): return []
    def dispatch(self, req, source=None):
        self.requests.append(copy.deepcopy(req)); assert req['action'] not in ('train', 'scienceTrain', 'scienceRunCandidate', 'scienceNotebookExecute')
        return Window.dispatch(self, req, source)
    def broadcast(self, kind='state', payload=None, **kwargs): self.bridge.send(kind, self.state() if kind == 'state' else payload)


def js(source):
    loop = QEventLoop(); timer = QTimer(); timer.setSingleShot(True); timer.timeout.connect(loop.quit); values = []
    view.page().runJavaScript(source, lambda value: (values.append(value), loop.quit())); timer.start(4000); loop.exec(); timer.stop()
    assert values, 'JS callback timeout'; return values[0]


def until(source):
    deadline = time.monotonic() + 25
    while time.monotonic() < deadline:
        if js(source): return
        loop = QEventLoop(); QTimer.singleShot(20, loop.quit); loop.exec()
    raise AssertionError(f'UI timeout: {source}; errors: {errors}; requests: {h.requests[-3:]}')


def click(identifier): js('document.getElementById(' + json.dumps(identifier) + ').click()')


with tempfile.TemporaryDirectory() as td:
    folder = Path(td); h = Harness(folder); page = Page(profile, view); view.setPage(page)
    channel = QWebChannel(page); channel.registerObject('host', h.bridge); page.setWebChannel(channel)
    url = QUrl.fromLocalFile(str(Path(__file__).resolve().parents[1] / 'app/web/index.html')); url.setFragment('model=shared'); view.setUrl(url); view.show()
    try:
        until('Boolean(window.loom?.document?.view.fitted)')
        js('document.querySelector(".node[data-id=use1]").dispatchEvent(new MouseEvent("dblclick",{bubbles:true}))')
        until('Boolean(document.getElementById("parameter-owner"))')
        assert js('!document.getElementById("parameter-count")')
        assert js('!document.querySelector("#inspector-panel .parameter-total")')
        assert js('document.getElementById("inspector-content").textContent.includes("A [")')
        assert js('Boolean(document.getElementById("equation").textContent)')
        js('(()=>{const n=document.getElementById("parameter-owner");n.value="use0";n.dispatchEvent(new Event("change"));})()')
        until('window.loom.document.graph.nodes.find(n=>n.id==="use1").share_with==="use0" && document.getElementById("inspector-content").textContent.includes("Canonical owner: use0")')
        assert validate_model(h.workspace.models(), 'shared')['parameters'] == 12
        assert js('document.getElementById("status").textContent.includes("12 parameters")')
        assert h.workspace.models()['shared']['graph']['nodes'][2]['share_with'] == 'use0'
        assert js('document.querySelector(".node[data-id=use1] .node-params title").textContent.includes("Shared: use0")')
        if os.environ.get('VTW_SHARED_CAPTURE'):
            until('document.getElementById("parameter-owner").value==="use0"')
            loop = QEventLoop(); QTimer.singleShot(150, loop.quit); loop.exec()
            assert view.grab().save(os.environ['VTW_SHARED_CAPTURE'])
        h.dispatch({'action': 'undo'}); assert 'share_with' not in h.workspace.models()['shared']['graph']['nodes'][2]
        h.dispatch({'action': 'redo'}); assert h.workspace.models()['shared']['graph']['nodes'][2]['share_with'] == 'use0'
        # Open exact v1 family; changing the mode must explicitly upgrade it.
        js('document.querySelector("[data-panel=science]").click();document.getElementById("experiment-dock").style.height="850px";document.getElementById("experiment-dock").style.width="1080px"')
        until('!document.getElementById("science-panel").hidden')
        js('(()=>{const n=document.getElementById("science-kind");n.value="family";n.dispatchEvent(new Event("change"));})()')
        until('document.getElementById("science-object").options.length===1')
        js('(()=>{const n=document.getElementById("science-object");n.value=n.options[0].value;n.dispatchEvent(new Event("change"));})()')
        click('science-author-edit')
        until('Boolean(document.querySelector(\'[data-family-card="repeats"] [data-key="mode"]\')) && !document.getElementById("science-family-check").disabled')
        before = h.workspace.db.serialize()
        js('(()=>{const n=document.querySelector(\'[data-family-card="repeats"] [data-key="mode"]\');n.value="tied";n.dispatchEvent(new Event("change"));})()')
        click('science-family-preview')
        until('JSON.parse(document.getElementById("science-family-preview-result").textContent||"{}").parameters===25')
        assert before == h.workspace.db.serialize()
        click('science-family-save'); until('document.getElementById("science-family-editor").hidden')
        assert h.workspace.get_spec('family', h.family_ref['id'])['version'] == 2
        assert h.workspace.get_spec('family', h.family_ref['id'])['repeats'][0]['mode'] == 'tied'
        until('!document.getElementById("science-family-create").disabled && JSON.parse(document.getElementById("science-object").value).revision==='+json.dumps(h.workspace.get_spec('family', h.family_ref['id'])['revision']))
        click('science-family-create'); until('document.getElementById("science-family-instance-status").textContent.includes("25 parameters")')
        h.workspace.save(folder / 'final.vtw'); reopened = Workspace.load(folder / 'final.vtw')
        try:
            assert reopened.get_spec('family', h.family_ref['id'])['version'] == 2
            assert any('share_with' in n for m in reopened.models().values() for n in m['graph']['nodes'])
        finally: reopened.db.close()
        assert not errors, errors; assert 'torch' not in sys.modules
        print(json.dumps({'ok': True, 'actual_inspector_sharing': True, 'owned_tied_family_upgrade': True,
            'undo_redo': True, 'parameters': 25, 'no_preview_sql_writes': True, 'save_reload': True,
            'host_torch': False, 'console_errors': errors}))
    finally:
        view.setPage(QWebEnginePage(view)); page.deleteLater(); channel.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete); app.processEvents(); profile.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete); app.processEvents(); view.close(); h.workspace.db.close()
        if h.workspace.history.directory: h.workspace.history.directory.cleanup()
