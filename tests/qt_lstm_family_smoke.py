"""Actual sandboxed Qt family7 seed/cards/preview/export/history/.vtw18.

All workspaces/configuration are temporary; no numerical runtime in the host.
"""
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
from app.core import Workspace, VERSION, validate_model
from app.desktop import Bridge, Window
from app.family_export import stored_source
from examples.lstm_family import definition
from examples.runtime_lstm_family import definition as runtime_definition

assert not os.environ.get('QTWEBENGINE_DISABLE_SANDBOX')
assert '--no-sandbox' not in os.environ.get('QTWEBENGINE_CHROMIUM_FLAGS', '')
ROOT = Path(__file__).resolve().parents[1]
RUNTIME_V2 = os.environ.get('VTW_LSTM_FAMILY_VARIANT') == 'runtime-v2'
app = QApplication(['lstm-family-smoke']); view = QWebEngineView(); view.resize(1440, 1000)
profile = QWebEngineProfile(view); errors = []


class Page(QWebEnginePage):
    def javaScriptConsoleMessage(self, level, message, line, source):
        if level == self.JavaScriptConsoleMessageLevel.ErrorMessageLevel:
            errors.append(f'{source}:{line}: {message}')


class Harness(QObject):
    state = Window.state; describe = Window.describe
    def __init__(self, folder):
        super().__init__(); self.config = Config(folder/'config.json'); self.workspace = Workspace()
        doc = runtime_definition() if RUNTIME_V2 else definition(); self.mid = doc['root']
        if RUNTIME_V2: doc.update(version=7, runtime_policy=None)
        for model in doc['models'].values(): self.workspace.put(model)
        self.family_ref = self.workspace.put_spec('family', doc); self.workspace.save(folder/'initial.vtw')
        self.job = None; self.torch_python = None; self.rendering = {}; self.bridge = Bridge(self, view); self.requests = []
    def recent_files(self): return []
    def broadcast(self, kind='state', payload=None, **kwargs):
        self.bridge.send(kind, self.state() if kind == 'state' else payload)
    def dispatch(self, req, source=None):
        self.requests.append(copy.deepcopy(req))
        assert req['action'] not in ('train', 'scienceTrain', 'scienceNotebookExecute')
        return Window.dispatch(self, req, source)


def js(source):
    loop = QEventLoop(); timer = QTimer(); timer.setSingleShot(True); timer.timeout.connect(loop.quit); values = []
    view.page().runJavaScript(source, lambda value: (values.append(value), loop.quit()))
    timer.start(4000); loop.exec(); timer.stop(); assert values, 'JavaScript timeout'; return values[0]


def until(source):
    deadline = time.monotonic() + 25
    while time.monotonic() < deadline:
        if js(source): return
        loop = QEventLoop(); QTimer.singleShot(20, loop.quit); loop.exec()
    raise AssertionError(f'UI timeout: {source}; errors: {errors}; last requests: {h.requests[-3:]}')


def click(identifier): js('document.getElementById('+json.dumps(identifier)+').click()')
def change(selector, value):
    assert js('(()=>{const n=document.querySelector('+json.dumps(selector)+');if(!n)return false;'
              'n.value='+json.dumps(value)+';n.dispatchEvent(new Event("change"));return true;})()'), selector


with tempfile.TemporaryDirectory() as directory:
    folder = Path(directory); h = Harness(folder); page = Page(profile, view); view.setPage(page)
    channel = QWebChannel(page); channel.registerObject('host', h.bridge); page.setWebChannel(channel)
    url = QUrl.fromLocalFile(str(ROOT/'app/web/index.html')); url.setFragment('model='+h.mid)
    view.setUrl(url); view.show()
    try:
        until('Boolean(window.loom?.document?.view.fitted)')
        js('document.querySelector("[data-panel=science]").click();'
           'document.getElementById("experiment-dock").style.height="850px";'
           'document.getElementById("experiment-dock").style.width="1080px"')
        until('!document.getElementById("science-panel").hidden')
        before = h.workspace.db.serialize(); click('science-author-new-family')
        until('!document.getElementById("science-family-check").disabled')
        assert js('document.querySelector("#science-family-roots [data-key=version]").value==="7"')
        assert js('document.querySelector("#science-family-roots [data-key=runtime_policy]").value==="null"')
        click('science-family-check'); until('document.getElementById("science-family-status").textContent.includes("Valid")')
        assert h.workspace.db.serialize() == before
        click('science-family-cancel'); until('document.getElementById("science-family-editor").hidden')
        change('#science-kind', 'family'); until('document.getElementById("science-object").options.length===1')
        change('#science-object', json.dumps(h.family_ref, separators=(',', ':')))
        click('science-author-edit'); until('!document.getElementById("science-family-check").disabled')
        assert js('document.querySelector("#science-family-roots [data-key=version]").value==="7"')
        if RUNTIME_V2:
            change('#science-family-roots [data-key=runtime_policy]', json.dumps(runtime_definition()['runtime_policy']))
        # Card zero is sequence.shape. A v7 typed selection is edited through
        # the real textarea, not dispatched directly to the host by the test.
        value = [{'select': {'symbol': 'directions'}, 'then': {'symbol': 'time'}, 'else': {'symbol': 'time'}}, {'symbol': 'width'}]
        selector = '[data-family-card="assignments"] [data-key="value"]'
        change(selector, 'not JSON'); click('science-family-preview')
        assert h.workspace.db.serialize() == before; assert js('document.querySelector('+json.dumps(selector)+').value') == 'not JSON'
        change(selector, json.dumps(value))
        symbol = '(id)=>[...document.querySelectorAll("[data-family-card=symbols]")].find(c=>c.querySelector("[data-key=id]").value===id)'
        for sid, val in (('projection', '0'), ('directions', 'false')):
            assert js('(()=>{const c=('+symbol+')('+json.dumps(sid)+');if(!c)return false;const n=c.querySelector("[data-key=default]");'
                      'n.value='+json.dumps(val)+';n.dispatchEvent(new Event("change"));return true;})()'), sid
        click('science-family-preview')
        until('JSON.parse(document.getElementById("science-family-preview-result").textContent||"{}").parameters===784')
        assert h.workspace.db.serialize() == before
        if os.environ.get('VTW_LSTM_FAMILY_CAPTURE'):
            click('toast-close')
            loop = QEventLoop(); QTimer.singleShot(150, loop.quit); loop.exec()
            assert view.grab().save(os.environ['VTW_LSTM_FAMILY_CAPTURE'])
        click('science-family-save'); until('document.getElementById("science-family-editor").hidden')
        saved = h.workspace.get_spec('family', h.family_ref['id']); assert saved['version'] == (8 if RUNTIME_V2 else 7)
        until('!document.getElementById("science-family-export").disabled && JSON.parse(document.getElementById("science-object").value).revision==='+json.dumps(saved['revision']))
        output = folder/'family.py'; before_export = h.workspace.db.serialize()
        with patch('app.desktop_experiments.QFileDialog.getSaveFileName', return_value=(str(output), '')):
            click('science-family-export'); until('document.getElementById("toast").textContent.includes("Checked family source exported")')
        req = next(r for r in reversed(h.requests) if r['action'] == 'scienceExportFamily')
        expected = stored_source(h.workspace, req['family'], memory_limit_percent=h.config.values['memory_limit_percent'])
        assert output.read_text() == expected['code']; assert h.workspace.db.serialize() == before_export
        click('science-family-create'); until('document.getElementById("science-family-instance-status").textContent.includes("784 parameters")')
        ref = h.workspace.list_specs('specialization')[0]; record = h.workspace.get_spec('specialization', ref['id'])
        assert record['version'] == (7 if RUNTIME_V2 else 6)
        if RUNTIME_V2:
            assert record['runtime']['version'] == 3 and record['runtime']['kind'] == 'runtime-temporal-v2'
        else: assert record['runtime'] is None
        root = record['model']['id']; assert validate_model(h.workspace.models(), root)['parameters'] == 784
        h.dispatch({'action': 'undo'}); assert root not in h.workspace.models()
        assert h.workspace.get_spec('specialization', ref['id'], record['revision']) == record
        h.dispatch({'action': 'redo'}); assert root in h.workspace.models()
        h.workspace.save(folder/'final.vtw'); opened = Workspace.load(folder/'final.vtw')
        try:
            assert opened.get_spec('specialization', ref['id']) == record
            assert opened.db.execute('PRAGMA user_version').fetchone()[0] == VERSION == 19
        finally: opened.db.close()
        assert not errors, errors; assert 'torch' not in sys.modules
        print(json.dumps({'ok': True, 'drawing_seed_family7': True, 'typed_selection_card': True,
                          'projection_zero_forward_only': True, 'parameters': 784, 'preview_sql_unchanged': True,
                          'invalid_text_retained': True, 'exact_export': True, 'specialization_version': record['version'],
                          'runtime_v2_explicit_upgrade': RUNTIME_V2,
                          'undo_redo_reload_vtw19': True, 'host_torch': False, 'console_errors': errors}))
    finally:
        view.setPage(QWebEnginePage(view)); page.deleteLater(); channel.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete); app.processEvents(); profile.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete); app.processEvents(); view.close()
        h.workspace.db.close()
        if h.workspace.history.directory: h.workspace.history.directory.cleanup()
