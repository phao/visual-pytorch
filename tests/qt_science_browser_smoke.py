"""Actual sandboxed WebEngine, large inline data and bounded read projections.

Private workspace/config/profile only. No numerical worker, fitting or Torch.
Limits are response projections, not registry/SQLite cost or lazy-storage claims.
"""
import copy
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from PyQt6.QtCore import QObject, QEvent, QEventLoop, QTimer, QUrl, QCoreApplication
from PyQt6.QtWidgets import QApplication
from PyQt6.QtWebChannel import QWebChannel
from PyQt6.QtWebEngineCore import QWebEnginePage, QWebEngineProfile
from PyQt6.QtWebEngineWidgets import QWebEngineView
from app.config import Config
from app.core import Workspace
from app.desktop import Bridge, Window
from app.science_browser import encoded, MAX_BYTES
from examples.experiment_workspace import populate

assert not os.environ.get('QTWEBENGINE_DISABLE_SANDBOX')
assert '--no-sandbox' not in os.environ.get('QTWEBENGINE_CHROMIUM_FLAGS', '')
app = QApplication(['scientific-read-browser-smoke'])
view = QWebEngineView(); view.resize(1440, 1000)
profile = QWebEngineProfile(view); errors = []


class Page(QWebEnginePage):
    def javaScriptConsoleMessage(self, level, message, line, source):
        if level == self.JavaScriptConsoleMessageLevel.ErrorMessageLevel:
            errors.append(f'{source}:{line}: {message}')


class Harness(QObject):
    state = Window.state; describe = Window.describe

    def __init__(self, folder):
        super().__init__(); self.config = Config(folder / 'config.json')
        self.workspace = Workspace(); populate(self.workspace)
        self.job = None; self.torch_python = None; self.rendering = {}
        self.bridge = Bridge(self, view); self.reads = []; self.actions = []
        template = self.workspace.get_spec('dataset', 'affine'); template.pop('revision')
        for i in range(58):
            doc = copy.deepcopy(template); doc.update(id=f'small-{i:02}', name=f'Small measurements {i:02}')
            self.workspace.put_spec('dataset', doc)
        large = copy.deepcopy(template)
        large.update(id='large', name='Large image tensor <script>not HTML</script>')
        large['fields'][0]['shape'] = [512]
        large['records'] = [{'id': f'row-{i}', 'values': {'signal': [float(i) + j / 512. for j in range(512)], 'target': [float(i)]}} for i in range(1250)]
        self.long_key = '🔬' * 10001
        self.long_value = '<script>not HTML</script>' + 'λ' * 8000
        large['usage'] = {'notes': {self.long_key: self.long_value}, 'fixture': 'historical'}
        self.old = self.workspace.put_spec('dataset', large)
        large['usage']['fixture'] = 'current'
        self.head = self.workspace.put_spec('dataset', large)

    def recent_files(self): return []

    def broadcast(self, kind='state', payload=None, **kw):
        self.bridge.send(kind, self.state() if kind == 'state' else payload)

    def dispatch(self, request, source=None):
        self.actions.append(request['action'])
        result = Window.dispatch(self, request, source)
        if request['action'] in ('scienceBrowse', 'scienceSummary', 'scienceInspect'):
            size = len(encoded(result)); assert size <= MAX_BYTES
            self.reads.append({'action': request['action'], 'bytes': size})
        return result


def js(script):
    loop = QEventLoop(); timer = QTimer(); timer.setSingleShot(True); timer.timeout.connect(loop.quit); values = []
    view.page().runJavaScript(script, lambda value: (values.append(value), loop.quit()))
    timer.start(5000); loop.exec(); timer.stop()
    assert values, 'JavaScript timeout'; return values[0]


def until(script, timeout=20):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if script() if callable(script) else js(script): return
        loop = QEventLoop(); QTimer.singleShot(20, loop.quit); loop.exec()
    raise AssertionError(f'UI timeout: {script}; console: {errors[:5]}')


def click(sid):
    until('!document.getElementById(' + json.dumps(sid) + ').disabled')
    js('document.getElementById(' + json.dumps(sid) + ').click()')


def tree_ready():
    until('document.getElementById("science-tree-status").textContent.includes("exact revision")')


def open_key(name):
    tree_ready()
    assert js('(()=>{const r=[...document.querySelectorAll("#science-tree-rows tbody tr")].find(r=>r.cells[0].textContent===' + json.dumps(name) + ');if(!r)return false;r.querySelector("[data-spec-value]").click();return true;})()'), name
    # Bridge delivery is asynchronous; ensure the new read actually replaced it.
    until('document.getElementById("science-tree-rows").children.length>0 && document.getElementById("science-tree-status").textContent.includes("exact revision")')


with tempfile.TemporaryDirectory() as directory:
    folder = Path(directory); h = Harness(folder)
    page = Page(profile, view); view.setPage(page)
    channel = QWebChannel(page); channel.registerObject('host', h.bridge); page.setWebChannel(channel)
    url = QUrl.fromLocalFile(str(Path(__file__).resolve().parents[1] / 'app/web/index.html'))
    url.setFragment('model=' + next(iter(h.workspace.models()))); view.setUrl(url); view.show()
    try:
        until('Boolean(window.loom?.document)')
        # Await the local animation-frame Fit, not a persisted camera write.
        until('Boolean(window.loom?.document?.view.fitted)')  # Camera is session-only.
        before = h.workspace.db.serialize(); initial_runs = len(h.workspace.list_specs('run', True)); writes = []
        h.workspace.db.set_trace_callback(lambda sql: writes.append(sql[:200]) if sql.lstrip().split(' ', 1)[0].upper() in ('INSERT', 'UPDATE', 'DELETE', 'REPLACE', 'CREATE', 'DROP', 'ALTER') else None)
        for action, args in [('scienceBrowse', {'kind': 'dataset'}), ('scienceSummary', {'reference': h.old}), ('scienceInspect', {'reference': h.old})]:
            try: h.dispatch({'action': action, 'workspace_id': 'foreign', **args})
            except ValueError as error: assert 'workspace' in str(error).lower()
            else: raise AssertionError('Read admitted a foreign workspace identity')
        js('document.querySelector("[data-panel=science]").click();document.getElementById("experiment-dock").style.height="750px";document.getElementById("experiment-dock").style.width="900px";')
        until('document.querySelectorAll(".science-candidate").length===8')
        js('document.getElementById("science-kind").value="dataset";document.getElementById("science-kind").dispatchEvent(new Event("change"))')
        until('document.getElementById("science-object").options.length===25 && JSON.parse(document.getElementById("science-document").textContent||"{}").reference?.kind==="dataset"')
        pinned = js('document.getElementById("science-object").value')
        click('science-object-next')
        until('document.getElementById("science-object-page").textContent.startsWith("25–50")')
        assert js('document.getElementById("science-object").value') == pinned
        assert js('document.getElementById("science-object").options.length') == 26
        click('science-object-prev'); until('document.getElementById("science-object-page").textContent.startsWith("0–25")')
        js('document.getElementById("science-object-query").value="Large image tensor"'); click('science-object-search')
        until('document.getElementById("science-object-page").textContent.startsWith("0–2")')
        assert js('document.getElementById("science-object").value') == pinned
        js('document.getElementById("science-object").value=JSON.stringify(' + json.dumps(h.old) + ');document.getElementById("science-object").dispatchEvent(new Event("change"))')
        until('JSON.parse(document.getElementById("science-document").textContent||"{}").counts?.observations===1250')
        assert js('JSON.parse(document.getElementById("science-document").textContent).reference.revision') == h.old['revision']
        assert js('document.getElementById("science-document").textContent.length') < 2000
        click('science-refresh'); until('JSON.parse(document.getElementById("science-document").textContent||"{}").counts?.observations===1250')
        assert js('JSON.parse(document.getElementById("science-object").value).revision') == h.old['revision']
        js('document.getElementById("science-details").open=true'); tree_ready()
        open_key('records'); until('document.getElementById("science-tree-status").textContent.includes("0–25 of 1250")')
        click('science-tree-next'); until('document.getElementById("science-tree-status").textContent.includes("25–50 of 1250")')
        click('science-tree-prev'); until('document.getElementById("science-tree-status").textContent.includes("0–25 of 1250")')
        open_key('[0]'); open_key('values'); open_key('signal')
        until('document.getElementById("science-tree-status").textContent.includes("0–25 of 512")')
        click('science-tree-next'); until('document.getElementById("science-tree-status").textContent.includes("25–50 of 512")')
        open_key('[25]'); until('document.getElementById("science-tree-rows").textContent==="0.048828125"')
        click('science-tree-root'); open_key('usage'); open_key('notes')
        until('document.querySelector("#science-tree-rows tbody tr")?.cells[0].textContent.includes("partial key")')
        assert js('document.getElementById("science-tree-rows").textContent.length') < 1000
        js('[...document.querySelectorAll("#science-tree-rows button")].find(b=>b.textContent==="Read full key").click()')
        until('document.getElementById("science-tree-status").textContent.includes("0–2000 of 10001")')
        assert js('Array.from(document.querySelector("#science-tree-rows pre").textContent).length') == 2000
        click('science-tree-next'); until('document.getElementById("science-tree-status").textContent.includes("2000–4000 of 10001")')
        click('science-tree-parent'); until('document.querySelector("#science-tree-rows pre")?.textContent.startsWith("<script>not HTML</script>")')
        assert js('document.getElementById("science-tree-rows").querySelectorAll("script,img,iframe").length') == 0
        for theme, font, spacing in [('dark', 14, 1), ('github', 18, 4), ('math', 28, 10)]:
            h.config.values.update(theme=theme, font_size=font, spacing=spacing); h.broadcast()
            until('getComputedStyle(document.body).getPropertyValue("--ui-font-size").trim()===' + json.dumps(str(font) + 'px'))
            assert js('parseFloat(getComputedStyle(document.querySelector("#science-tree-rows pre")).fontSize)') >= font - 2
            assert js('document.getElementById("science-document").getBoundingClientRect().height') <= font * 11
            assert js('document.getElementById("science-tree-rows").getBoundingClientRect().width<=document.getElementById("science-panel").getBoundingClientRect().width')
        # Actual same-UUID external refresh: invalidate the old tree and retain
        # the historical exact choice, without silently following its newer head.
        h.broadcast('externalReload', 'Private test reload')
        until('document.getElementById("science-tree-status").textContent.includes("0–") && document.getElementById("science-tree-path").textContent.includes("[]")')
        assert js('JSON.parse(document.getElementById("science-object").value).revision') == h.old['revision']
        click('science-tree-root'); open_key('records')
        if os.environ.get('VTW_SMOKE_SCREENSHOT'):
            h.config.values.update(theme='github', font_size=18, spacing=4); h.broadcast()
            until('getComputedStyle(document.body).getPropertyValue("--ui-font-size").trim()==="18px"')
            js('document.getElementById("toast-close").click();document.getElementById("science-details").scrollIntoView({block:"start"})')
            loop = QEventLoop(); QTimer.singleShot(350, loop.quit); loop.exec()
            assert view.grab().save(os.environ['VTW_SMOKE_SCREENSHOT'])
        assert h.workspace.db.serialize() == before, writes
        assert len(h.workspace.list_specs('run', True)) == initial_runs == 0
        assert not any(a in h.actions for a in ('scienceIndex', 'scienceShow', 'scienceReport', 'scienceTrain', 'scienceGenerateNotebook'))
        assert 'torch' not in sys.modules and 'nbclient' not in sys.modules
        assert not errors, errors
        print(json.dumps({'large_inline_shape': [1250, 512], 'revision_paging': 'PASS', 'historical_identity': 'PASS', 'scalar_key_value_paging': 'PASS', 'same_uuid_reload': 'PASS', 'themes_font_28': 'PASS', 'sql_read_only': 'PASS', 'projection_max_bytes': max(r['bytes'] for r in h.reads), 'read_calls': len(h.reads), 'host_torch': False, 'console_errors': errors}))
    finally:
        if h.workspace.history.directory:h.workspace.history.directory.cleanup()
        h.workspace.db.close()
        view.close(); view.setPage(QWebEnginePage(view)); page.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        profile.deleteLater(); QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
