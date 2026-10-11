"""Real sandboxed Qt family choice editing, candidate preview and worker delivery."""
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
from app.core import Workspace
from app.desktop import Bridge, Window
from app.experiment_objects import object_reference
from app.revisions import validate_all
from test_family_axes import family_experiment

assert not os.environ.get('QTWEBENGINE_DISABLE_SANDBOX')
assert '--no-sandbox' not in os.environ.get('QTWEBENGINE_CHROMIUM_FLAGS', '')
app = QApplication(['family-axes-smoke']); view = QWebEngineView(); view.resize(1440, 1000)
profile = QWebEngineProfile(view); errors = []


class Page(QWebEnginePage):
    def javaScriptConsoleMessage(self, level, message, line, source):
        if level == self.JavaScriptConsoleMessageLevel.ErrorMessageLevel: errors.append(f'{source}:{line}: {message}')


class Harness(QObject):
    state = Window.state; describe = Window.describe
    def __init__(self, folder):
        super().__init__(); self.config = Config(folder / 'config.json'); self.workspace, self.e = family_experiment()
        self.ref = object_reference('experiment', self.e); self.workspace.save(folder / 'original.vtw')
        self.job = None; self.torch_python = '/home/pedro/miniforge3/bin/python3.12'; self.rendering = {}; self.requests = []
        self.bridge = Bridge(self, view)
    def recent_files(self): return []
    def dispatch(self, req, source=None):
        self.requests.append(copy.deepcopy(req)); return Window.dispatch(self, req, source)
    def broadcast(self, kind='state', payload=None, **kwargs): self.bridge.send(kind, self.state() if kind == 'state' else payload)


def js(source):
    loop = QEventLoop(); timer = QTimer(); timer.setSingleShot(True); timer.timeout.connect(loop.quit); values = []
    view.page().runJavaScript(source, lambda value: (values.append(value), loop.quit())); timer.start(4000); loop.exec(); timer.stop()
    assert values, 'JS callback timeout'; return values[0]


def until(source, seconds=25):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if js(source): return
        loop = QEventLoop(); QTimer.singleShot(20, loop.quit); loop.exec()
    status = js('document.getElementById("science-author-status").textContent')
    raise AssertionError(f'UI timeout: {source}; status: {status}; errors: {errors}; last requests: {h.requests[-3:]}')


def click(identifier): js('document.getElementById(' + json.dumps(identifier) + ').click()')


def field(label, ordinal=0):
    return '([...document.querySelectorAll(\'[data-experiment-axis="model"][data-axis-index="' + str(ordinal) + '"] label\')].find(n=>n.firstChild.textContent===' + json.dumps(label) + ').querySelector("input,textarea,select"))'


def change(source, value): js('(()=>{const n=' + source + ';n.value=' + json.dumps(value) + ';n.dispatchEvent(new Event("change"));})()')


with tempfile.TemporaryDirectory() as directory:
    folder = Path(directory); h = Harness(folder); page = Page(profile, view); view.setPage(page)
    channel = QWebChannel(page); channel.registerObject('host', h.bridge); page.setWebChannel(channel)
    url = QUrl.fromLocalFile(str(Path(__file__).resolve().parents[1] / 'app/web/index.html'))
    url.setFragment('model=' + next(iter(h.workspace.models()))); view.setUrl(url); view.show()
    try:
        until('Boolean(window.loom?.document?.view.fitted)')
        js('document.querySelector("[data-panel=science]").click();document.getElementById("experiment-dock").style.height="850px";document.getElementById("experiment-dock").style.width="1080px"')
        until('!document.getElementById("science-panel").hidden')
        change('document.getElementById("science-kind")', 'experiment')
        until('document.getElementById("science-object").options.length===2')
        # Browser option serialization need not use the caller's key order.
        js('(()=>{const n=document.getElementById("science-object");n.value=[...n.options].find(o=>JSON.parse(o.value).revision===' + json.dumps(h.ref['revision']) + ').value;n.dispatchEvent(new Event("change"));})()')
        until('document.querySelectorAll("#science-candidates [data-science-run]").length===16')
        before = h.workspace.db.serialize(); click('science-author-edit')
        until('!document.getElementById("science-author-check").disabled && Boolean(document.querySelector(\'[data-experiment-axis="model"] textarea\'))')
        axes = field('Ordered binding axes JSON'); changed_axes = json.dumps([
            {'symbol': 'hidden-width', 'values': [3, 2]}, {'symbol': 'hidden-depth', 'values': [2, 1]}])
        change(axes, '['); click('science-author-check')
        until('document.getElementById("science-author-status").textContent.includes("invalid")')
        assert js(axes + '.value') == '['; assert before == h.workspace.db.serialize()
        change(axes, changed_axes); click('science-experiment-preview')
        until('document.querySelectorAll("#science-experiment-preview-rows .science-candidate").length===16')
        assert js('document.querySelector("#science-experiment-preview-rows .science-candidate").textContent.includes("34 trainable parameters")')
        assert before == h.workspace.db.serialize(); assert not h.workspace.dirty
        # Create a second family choice through the real version-upgrading button
        # and exact revision selector, not a fabricated complete document.
        js('[...document.querySelectorAll(\'[data-experiment-collection="models"] button\')].find(b=>b.textContent==="Add family selection").click()')
        until('document.querySelectorAll(\'[data-experiment-axis="model"]\').length===2 && !document.getElementById("science-author-check").disabled')
        family_ref = h.e['models'][0]['family']
        exact = field('Exact family', 1)
        js('(()=>{const n=' + exact + ';n.value=[...n.options].find(o=>o.value && JSON.parse(o.value).revision===' + json.dumps(family_ref['revision']) + ').value;n.dispatchEvent(new Event("change"));})()')
        click('science-author-save')
        until('document.getElementById("science-author").hidden && document.querySelectorAll("#science-candidates [data-science-run]").length===20')
        retained = h.workspace.get_spec('experiment', h.e['id']); assert retained['version'] == 2
        assert retained['models'][0]['axes'][0]['values'] == [3, 2]; assert len(retained['models']) == 2
        assert h.workspace.list_specs('run') == []; assert h.workspace.list_specs('specialization') == []
        if os.environ.get('VTW_FAMILY_AXES_CAPTURE'):
            loop = QEventLoop(); QTimer.singleShot(150, loop.quit); loop.exec(); assert view.grab().save(os.environ['VTW_FAMILY_AXES_CAPTURE'])
        js('document.querySelector("#science-candidates [data-science-run]").click()')
        until('Boolean(window.loom.state.experiment_job)')
        until('!window.loom.state.experiment_job && document.getElementById("toast-message").textContent.includes("completed")', 40)
        run = h.workspace.get_spec('run', h.workspace.list_specs('run')[0]['id'])
        assert run['version'] == 2; assert run['attempts'][0]['result']['counts']['completed_updates'] == 3
        refs = run['resolved']['references']; specialization = h.workspace.get_spec('specialization', refs['specialization']['id'], refs['specialization']['revision'])
        assert specialization['bindings'] == {'hidden-width': 3, 'hidden-depth': 2}
        change('document.getElementById("science-kind")', 'run')
        until('!document.getElementById("science-run").hidden && Boolean(document.querySelector("[data-run-evidence]"))')
        evidence = h.dispatch({'action': 'scienceRunEvidence', 'reference': object_reference('run', run)})
        assert evidence['objects']['model']['parameters_exact'] == '34'
        assert evidence['objects']['specialization']['reference'] == refs['specialization']
        assert evidence['execution']['counts']['completed_updates'] == 3
        assert not getattr(h, 'experiment_recoveries', {}); validate_all(h.workspace.db)
        h.workspace.save(folder / 'final.vtw'); loaded = Workspace.load(folder / 'final.vtw')
        try: assert loaded.get_spec('run', run['id'], run['revision']) == run
        finally: loaded.db.close()
        assert 'torch' not in sys.modules; assert not errors, errors
        print(json.dumps({'ok': True, 'typed_family_axes': True, 'invalid_json_preserved': True,
                          'ordered_axes': True, 'add_family_selection': True, 'candidates': 20,
                          'worker_updates': 3, 'exact_specialization_delivered': True,
                          'save_reload': True, 'host_torch': False, 'console_errors': errors}))
    finally:
        if h.job: assert h.job.wait(15000)
        view.setPage(QWebEnginePage(view)); page.deleteLater(); channel.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete); app.processEvents(); profile.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete); app.processEvents(); view.close(); h.workspace.db.close()
        if h.workspace.history.directory: h.workspace.history.directory.cleanup()
