"""Actual app HTML/QWebChannel chunked editor, private profiles and SQLite.

No Torch/Jupyter/worker or user workspace/config. Raw JSON tokens deliberately
exercise int64 values that JavaScript Number cannot represent exactly.
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
from app.core import Workspace, default_model
from app.desktop import Bridge, Window
from app.science_browser import encoded
from test_authoring import measurements

assert not os.environ.get('QTWEBENGINE_DISABLE_SANDBOX')
assert '--no-sandbox' not in os.environ.get('QTWEBENGINE_CHROMIUM_FLAGS', '')
app = QApplication(['chunked-draft-values-smoke'])
view = QWebEngineView(); view.resize(1440, 1050)
profile = QWebEngineProfile(view); errors = []


class Page(QWebEnginePage):
    def javaScriptConsoleMessage(self, level, message, line, source):
        if level == self.JavaScriptConsoleMessageLevel.ErrorMessageLevel:
            errors.append(f'{source}:{line}: {message}')


class Harness(QObject):
    state = Window.state; describe = Window.describe

    def __init__(self, folder):
        super().__init__(); self.config = Config(folder/'config.json')
        self.workspace = Workspace(); self.model = default_model('Value editor fixture')
        self.workspace.put(self.model); self.job = None; self.torch_python = None
        self.rendering = {}; self.bridge = Bridge(self, view); self.actions = []; self.projections = []
        self.original = measurements()
        self.original['usage'] = {'edition':'historical', 'large':[9007199254740993, *range(1, 20000)],
            'literal':'<script>not HTML</script>'+'λ😀'*5000}
        self.old = self.workspace.put_spec('dataset', self.original)
        newer = copy.deepcopy(self.original); newer['usage']['edition'] = 'current'
        self.head = self.workspace.put_spec('dataset', newer)

    def recent_files(self): return []

    def broadcast(self, kind='state', payload=None, **kw):
        self.bridge.send(kind, self.state() if kind == 'state' else payload)

    def dispatch(self, request, source=None):
        self.actions.append(request['action']); result = Window.dispatch(self, request, source)
        if request['action'].startswith('scienceDraft'):
            if request['action']=='scienceDraftOpen': assert 'document' not in result
            size = len(encoded(result)); assert size <= 65536
            self.projections.append({'action':request['action'], 'bytes':size})
        return result


def js(script):
    loop = QEventLoop(); timer = QTimer(); timer.setSingleShot(True); values = []
    timer.timeout.connect(loop.quit)
    view.page().runJavaScript(script, lambda value:(values.append(value),loop.quit()))
    timer.start(5000); loop.exec(); timer.stop(); assert values, 'JavaScript timeout'; return values[0]


def until(condition, timeout=15):
    end = time.monotonic()+timeout
    while time.monotonic() < end:
        if condition() if callable(condition) else js(condition): return
        loop = QEventLoop(); QTimer.singleShot(15,loop.quit); loop.exec()
    raise AssertionError(f'UI timeout: {condition}; console {errors[:5]}; status '+str(js('document.getElementById("science-value-status").textContent')))


def click(sid):
    until('!document.getElementById('+json.dumps(sid)+').disabled')
    js('document.getElementById('+json.dumps(sid)+').click()')


def open_key(name):
    count = h.actions.count('scienceInspect')
    assert js('(()=>{const r=[...document.querySelectorAll("#science-tree-rows tbody tr")].find(r=>r.cells[0].textContent===' + json.dumps(name) + ');if(!r)return false;r.querySelector("[data-spec-value]").click();return true;})()'), name
    until(lambda:h.actions.count('scienceInspect')>count)
    until('!document.getElementById("science-tree-status").textContent.startsWith("Reading")')


def edit_text(text):
    js('(()=>{const t=document.getElementById("science-value-text");t.value='+json.dumps(text)+';t.dispatchEvent(new Event("input"));})()')


with tempfile.TemporaryDirectory() as directory:
    folder = Path(directory); h = Harness(folder)
    page = Page(profile,view); view.setPage(page)
    channel = QWebChannel(page); channel.registerObject('host',h.bridge); page.setWebChannel(channel)
    url = QUrl.fromLocalFile(str(Path(__file__).resolve().parents[1]/'app/web/index.html'))
    url.setFragment('model='+h.model['id']); view.setUrl(url); view.show()
    try:
        until('Boolean(window.loom?.document)')
        until('Boolean(window.loom?.document?.view.fitted)')  # Camera is session-only.
        before = h.workspace.db.serialize(); history = copy.deepcopy(h.workspace.history.summary())
        history_steps = len(h.workspace.history.undo_stack)
        js('document.querySelector("[data-panel=science]").click();document.getElementById("experiment-dock").style.height="850px";document.getElementById("experiment-dock").style.width="900px";')
        js('document.getElementById("science-kind").value="dataset";document.getElementById("science-kind").dispatchEvent(new Event("change"))')
        until('document.getElementById("science-object").options.length===2')
        js('document.getElementById("science-object").value=JSON.stringify('+json.dumps(h.old)+');document.getElementById("science-object").dispatchEvent(new Event("change"));document.getElementById("science-details").open=true;')
        until('document.getElementById("science-tree-status").textContent.includes("exact revision")')
        open_key('usage'); open_key('large'); click('science-tree-edit')
        until('!document.getElementById("science-value-finish").disabled')
        first = js('document.getElementById("science-value-text").value')
        assert first.startswith('[9007199254740993,1,2,') and len(first)==2000
        assert h.workspace.db.serialize()==before and h.workspace.history.summary()==history
        click('science-value-next')
        until('document.getElementById("science-value-status").textContent.includes("bytes 2000–")')
        click('science-value-prev')
        until('document.getElementById("science-value-text").value.startsWith("[9007199254740993")')
        edit_text('{'+first[1:]); click('science-value-apply')
        until('!document.getElementById("science-value-finish").disabled')
        click('science-value-finish'); until('document.getElementById("science-value-status").textContent.includes("Invalid revision JSON") && !document.getElementById("science-value-finish").disabled')
        assert js('document.getElementById("science-value-text").value').startswith('{9007199254740993')
        assert js('document.getElementById("science-value-save").disabled')
        assert h.workspace.db.serialize()==before and h.workspace.history.summary()==history
        corrected = first.replace('9007199254740993','9007199254740995',1)
        edit_text(corrected); click('science-value-apply'); until('!document.getElementById("science-value-finish").disabled')
        # Guard must reject before whole-value parsing and before SQL history.
        h.authoring_sessions.available_fn = lambda:0
        from app.authoring_sessions import _parse
        def guarded_parse(text):
            if len(text)>65536: raise AssertionError('Budget must precede full staged JSON parsing')
            return _parse(text)
        with patch('app.authoring_sessions._parse',side_effect=guarded_parse):
            click('science-value-finish'); until('document.getElementById("science-value-status").textContent.includes("available-RAM") && !document.getElementById("science-value-finish").disabled')
        assert h.workspace.db.serialize()==before and h.workspace.history.summary()==history
        assert js('document.getElementById("science-value-text").value')==corrected
        h.authoring_sessions.available_fn = lambda:2*1024**3
        for theme,font,spacing in [('dark',14,1),('github',18,4),('math',28,10)]:
            h.config.values.update(theme=theme,font_size=font,spacing=spacing); h.broadcast()
            until('getComputedStyle(document.body).getPropertyValue("--ui-font-size").trim()==='+json.dumps(str(font)+'px'))
            assert js('parseFloat(getComputedStyle(document.getElementById("science-value-text")).fontSize)')>=font-2
            assert js('document.getElementById("science-value-text").getBoundingClientRect().width<=document.getElementById("science-panel").getBoundingClientRect().width')
        h.config.values.update(theme='github',font_size=18,spacing=4); h.broadcast()
        until('getComputedStyle(document.body).getPropertyValue("--ui-font-size").trim()==="18px"')
        if os.environ.get('VTW_SMOKE_SCREENSHOT'):
            js('document.getElementById("science-value-editor").scrollIntoView({block:"start"});document.getElementById("toast-close").click()')
            loop = QEventLoop(); QTimer.singleShot(350,loop.quit); loop.exec()
            assert view.grab().save(os.environ['VTW_SMOKE_SCREENSHOT'])
        click('science-value-finish'); until('!document.getElementById("science-value-check").disabled')
        assert h.workspace.db.serialize()==before and h.workspace.history.summary()==history
        click('science-value-check'); until('document.getElementById("science-value-status").textContent.startsWith("Valid draft")')
        assert h.workspace.db.serialize()==before
        # The native shell must perform the same admission before undo capture.
        h.authoring_sessions.available_fn = lambda:0
        with patch.object(h.workspace.history,'capture',side_effect=AssertionError('Admission must precede undo snapshot')):
            click('science-value-save'); until('document.getElementById("science-value-status").textContent.includes("available-RAM") && !document.getElementById("science-value-save").disabled')
        assert h.workspace.db.serialize()==before and h.workspace.history.summary()==history
        h.authoring_sessions.available_fn = lambda:2*1024**3
        click('science-value-save'); until('document.getElementById("science-value-editor").hidden')
        saved = h.workspace.get_spec('dataset',h.old['id']); expected = copy.deepcopy(h.original)
        expected['usage']['large'][0] = 9007199254740995
        assert {k:v for k,v in saved.items() if k!='revision'}==expected
        assert h.workspace.get_spec('dataset',h.old['id'],h.old['revision'])['usage']==h.original['usage']
        assert len(h.workspace.history.undo_stack)==history_steps+1
        h.dispatch({'action':'undo'}); assert h.workspace.get_spec('dataset',h.old['id'])['revision']==h.head['revision']
        h.dispatch({'action':'redo'}); assert h.workspace.get_spec('dataset',h.old['id'])['revision']==saved['revision']
        h.workspace.save(folder/'values.vtw'); loaded = Workspace.load(folder/'values.vtw')
        try: assert loaded.get_spec('dataset',h.old['id'])==saved
        finally: loaded.db.close()
        assert not any(action in h.actions for action in ('scienceAuthorBegin','scienceAuthorCheck','scienceAuthorSave','scienceShow','scienceIndex','scienceTrain'))
        assert 'torch' not in sys.modules and 'nbclient' not in sys.modules and not errors, errors
        print(json.dumps({'actual_editor':'PASS','int64_precision':'PASS','historical_exact_base':'PASS','malformed_fragment_retained':'PASS',
            'ram_before_parse_and_history':'PASS','themes_font28':'PASS','single_history_undo_redo_reload':'PASS',
            'projections':len(h.projections),'max_bytes':max(row['bytes'] for row in h.projections),'console_errors':errors,'host_torch':False}))
    finally:
        if hasattr(h,'authoring_sessions'): h.authoring_sessions.close_owner(h.bridge)
        if h.workspace.history.directory: h.workspace.history.directory.cleanup()
        h.workspace.db.close(); view.close(); view.setPage(QWebEnginePage(view)); page.deleteLater()
        QCoreApplication.sendPostedEvents(None,QEvent.Type.DeferredDelete)
        profile.deleteLater(); QCoreApplication.sendPostedEvents(None,QEvent.Type.DeferredDelete)
