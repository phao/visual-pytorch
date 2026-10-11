"""Actual sandboxed Qt scoped call cards, explicit upgrade and retained reload."""
import copy
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch
from PyQt6.QtCore import QObject, QCoreApplication, QEvent, QEventLoop, QTimer, QUrl
from PyQt6.QtWidgets import QApplication
from PyQt6.QtWebChannel import QWebChannel
from PyQt6.QtWebEngineCore import QWebEnginePage, QWebEngineProfile
from PyQt6.QtWebEngineWidgets import QWebEngineView
from app.config import Config
from app.core import Workspace, default_model, generate_code
from app.desktop import Bridge, Window
from test_scoped_constructors import scoped

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
        self.workspace.put(default_model()); source = scoped()
        if os.environ.get('VTW_SCOPED_UPGRADE'):
            source.pop('constructors'); source['version'] = 3
            source['symbols'] = [s for s in source['symbols'] if s['scope'] == 'child']
            for s, default in zip(source['symbols'], [3, 'Tanh', True]): s['default'] = default
        self.family_ref = self.workspace.put_spec('family', source)
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
    selector = f'[data-family-card="constructors"][data-ordinal="{ordinal}"] [data-key="{key}"]'
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
        until('!document.getElementById("science-family-check").disabled')
        before = h.workspace.db.serialize()
        if os.environ.get('VTW_SCOPED_UPGRADE'):
            js('([...document.getElementById("science-family-cards").querySelectorAll("button")].find(n=>n.textContent==="Add constructor")).click()')
            until('document.querySelectorAll(\'[data-family-card="constructors"]\').length===1 && !document.getElementById("science-family-check").disabled')
            for key, value in [('model', 'parent'), ('node', 'left'), ('symbol', 'child-width'), ('value', '4')]:
                change(0, key, value)
            expected = 29
        else:
            until('document.querySelectorAll(\'[data-family-card="constructors"]\').length===6')
            change(0, 'value', 'not JSON'); click('science-family-preview')
            assert js('document.querySelector(\'[data-family-card="constructors"][data-ordinal="0"] [data-key="value"]\').value==="not JSON"')
            assert before == h.workspace.db.serialize()
            change(0, 'value', '{"op":"add","args":[{"symbol":"left-width"},1]}')
            change(5, 'value', 'false'); expected = 28
        click('science-family-preview')
        until('JSON.parse(document.getElementById("science-family-preview-result").textContent||"{}").parameters==='+str(expected))
        assert before == h.workspace.db.serialize()
        assert js('JSON.parse(document.getElementById("science-family-preview-result").textContent).constructor_contexts===3')
        if os.environ.get('VTW_SCOPED_CAPTURE'):
            js('document.getElementById("toast-close").click();[...document.getElementById("science-family-cards").children].forEach(n=>{n.open=n.querySelector("summary").textContent.includes("Module")})')
            loop = QEventLoop(); QTimer.singleShot(150, loop.quit); loop.exec()
            assert view.grab().save(os.environ['VTW_SCOPED_CAPTURE'])
        click('science-family-save'); until('document.getElementById("science-family-editor").hidden')
        saved = h.workspace.get_spec('family', h.family_ref['id'])
        assert saved['version'] == 4 and len(saved['constructors']) == (1 if os.environ.get('VTW_SCOPED_UPGRADE') else 6)
        until('!document.getElementById("science-family-create").disabled && JSON.parse(document.getElementById("science-object").value).revision==='+json.dumps(saved['revision']))
        if os.environ.get('VTW_FAMILY_EXPORT'):
            from app.family_export import stored_source
            exported = folder / 'architecture-family.py'; before_export = h.workspace.db.serialize()
            with patch('app.desktop_experiments.QFileDialog.getSaveFileName', return_value=(str(exported), '')):
                click('science-family-export')
                until('!document.getElementById("science-family-export").disabled && document.getElementById("toast").textContent.includes("Checked family source exported")')
            export_request = next(r for r in reversed(h.requests) if r['action'] == 'scienceExportFamily')
            assert export_request['family']['revision'] == saved['revision']
            expected_source = stored_source(h.workspace, export_request['family'], memory_limit_percent=h.config.values['memory_limit_percent'])
            assert exported.read_text() == expected_source['code']
            assert h.workspace.db.serialize() == before_export
            if os.environ.get('VTW_FAMILY_EXPORT_CAPTURE'):
                loop = QEventLoop(); QTimer.singleShot(150, loop.quit); loop.exec()
                assert view.grab().save(os.environ['VTW_FAMILY_EXPORT_CAPTURE'])
        click('science-family-create'); until('document.getElementById("science-family-instance-status").textContent.includes("'+str(expected)+' parameters")')
        instance = h.workspace.list_specs('specialization')[0]
        assert h.workspace.get_spec('specialization', instance['id'])['version'] == 3
        h.workspace.save(folder / 'final.vtw'); reopened = Workspace.load(folder / 'final.vtw')
        try:
            assert reopened.get_spec('family', h.family_ref['id'])['version'] == 4
            specialization = reopened.get_spec('specialization', instance['id']); model = reopened.get_spec('model', specialization['model']['id'])
            assert 'nn.Tanh(' in generate_code(model['models'], model['root'])
            assert len(specialization['instances']) == 3
            assert specialization['instances'][0]['bindings']['child-width'] == 4
        finally: reopened.db.close()
        assert not errors, errors; assert 'torch' not in sys.modules
        print(json.dumps({'ok': True, 'actual_scoped_constructor_cards': True, 'explicit_upgrade': bool(os.environ.get('VTW_SCOPED_UPGRADE')),
            'checked_family_export': bool(os.environ.get('VTW_FAMILY_EXPORT')),
            'invalid_value_retained': not bool(os.environ.get('VTW_SCOPED_UPGRADE')),
            'parameters': expected, 'no_preview_sql_writes': True, 'save_reload': True, 'host_torch': False, 'console_errors': errors}))
    finally:
        view.setPage(QWebEnginePage(view)); page.deleteLater(); channel.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete); app.processEvents(); profile.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete); app.processEvents(); view.close(); h.workspace.db.close()
        if h.workspace.history.directory: h.workspace.history.directory.cleanup()
