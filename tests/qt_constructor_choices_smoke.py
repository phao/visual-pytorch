"""Actual sandboxed Qt typed constructor defaults, preview, retention and reload."""
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
from app.core import Workspace, default_model, generate_code
from app.desktop import Bridge, Window
from test_constructor_choices import choices

assert not os.environ.get('QTWEBENGINE_DISABLE_SANDBOX')
assert '--no-sandbox' not in os.environ.get('QTWEBENGINE_CHROMIUM_FLAGS', '')
app = QApplication(['constructor-choices-smoke']); view = QWebEngineView(); view.resize(1440, 1000)
profile = QWebEngineProfile(view); errors = []


class Page(QWebEnginePage):
    def javaScriptConsoleMessage(self, level, message, line, source):
        if level == self.JavaScriptConsoleMessageLevel.ErrorMessageLevel: errors.append(f'{source}:{line}: {message}')


class Harness(QObject):
    state = Window.state; describe = Window.describe
    def __init__(self, folder):
        super().__init__(); self.config = Config(folder / 'config.json'); self.workspace = Workspace()
        self.workspace.put(default_model()); self.family_ref = self.workspace.put_spec('family', choices())
        self.workspace.save(folder / 'original.vtw'); self.job = None; self.torch_python = None
        self.rendering = {}; self.requests = []; self.bridge = Bridge(self, view)
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


def change(ordinal, key, value):
    selector = f'[data-family-card="symbols"][data-ordinal="{ordinal}"] [data-key="{key}"]'
    js('(()=>{const n=document.querySelector('+json.dumps(selector)+');n.value='+json.dumps(value)+';n.dispatchEvent(new Event("change"));})()')


with tempfile.TemporaryDirectory() as td:
    folder = Path(td); h = Harness(folder); page = Page(profile, view); view.setPage(page)
    channel = QWebChannel(page); channel.registerObject('host', h.bridge); page.setWebChannel(channel)
    url = QUrl.fromLocalFile(str(Path(__file__).resolve().parents[1] / 'app/web/index.html')); url.setFragment('model=' + next(iter(h.workspace.models()))); view.setUrl(url); view.show()
    try:
        until('Boolean(window.loom?.document?.view.fitted)')
        js('document.querySelector("[data-panel=science]").click();document.getElementById("experiment-dock").style.height="850px";document.getElementById("experiment-dock").style.width="1080px"')
        until('!document.getElementById("science-panel").hidden')
        js('(()=>{const n=document.getElementById("science-kind");n.value="family";n.dispatchEvent(new Event("change"));})()')
        until('document.getElementById("science-object").options.length===1')
        js('(()=>{const n=document.getElementById("science-object");n.value=n.options[0].value;n.dispatchEvent(new Event("change"));})()')
        click('science-author-edit')
        until('document.querySelectorAll(\'[data-family-card="symbols"]\').length===2 && !document.getElementById("science-family-check").disabled')
        assert js('document.querySelector(\'[data-family-card="symbols"][data-ordinal="0"] [data-key="type"]\').value==="enum"')
        assert js('document.querySelector(\'[data-family-card="symbols"][data-ordinal="1"] [data-key="type"]\').value==="boolean"')
        before = h.workspace.db.serialize(); change(0, 'default', '"GELU"'); change(1, 'default', '0')
        click('science-family-preview')
        until('document.getElementById("science-family-status").textContent.includes("boolean") && !document.getElementById("science-family-preview").disabled')
        assert js('document.querySelector(\'[data-family-card="symbols"][data-ordinal="1"] [data-key="default"]\').value==="0"')
        assert before == h.workspace.db.serialize()
        change(1, 'default', 'false'); click('science-family-preview')
        until('JSON.parse(document.getElementById("science-family-preview-result").textContent||"{}").parameters===6')
        assert before == h.workspace.db.serialize()
        assert js('document.getElementById("science-family-preview-result").textContent.includes("GELU")')
        if os.environ.get('VTW_CHOICES_CAPTURE'):
            js('document.getElementById("toast-close").click();[...document.getElementById("science-family-cards").children].forEach(n=>{n.open=n.querySelector("summary").textContent.includes("symbols")})')
            loop = QEventLoop(); QTimer.singleShot(150, loop.quit); loop.exec()
            assert view.grab().save(os.environ['VTW_CHOICES_CAPTURE'])
        click('science-family-save'); until('document.getElementById("science-family-editor").hidden')
        saved = h.workspace.get_spec('family', h.family_ref['id'])
        assert saved['version'] == 3 and saved['symbols'][0]['default'] == 'GELU' and saved['symbols'][1]['default'] is False
        until('!document.getElementById("science-family-create").disabled && JSON.parse(document.getElementById("science-object").value).revision==='+json.dumps(saved['revision']))
        click('science-family-create'); until('document.getElementById("science-family-instance-status").textContent.includes("6 parameters")')
        instance = h.workspace.list_specs('specialization')[0]
        assert h.workspace.get_spec('specialization', instance['id'])['version'] == 2
        h.workspace.save(folder / 'final.vtw'); reopened = Workspace.load(folder / 'final.vtw')
        try:
            assert reopened.get_spec('family', h.family_ref['id'])['version'] == 3
            specialization = reopened.get_spec('specialization', instance['id']); model = reopened.get_spec('model', specialization['model']['id'])
            assert 'nn.GELU(' in generate_code(model['models'], model['root'])
            assert specialization['bindings']['learn-bias'] is False
        finally: reopened.db.close()
        assert not errors, errors; assert 'torch' not in sys.modules
        print(json.dumps({'ok': True, 'actual_typed_enum_boolean_defaults': True, 'invalid_value_retained': True,
            'parameters': 6, 'no_preview_sql_writes': True, 'save_reload': True, 'host_torch': False, 'console_errors': errors}))
    finally:
        view.setPage(QWebEnginePage(view)); page.deleteLater(); channel.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete); app.processEvents(); profile.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete); app.processEvents(); view.close(); h.workspace.db.close()
        if h.workspace.history.directory: h.workspace.history.directory.cleanup()
