"""Actual sandboxed task/experiment pickers with >100 revisions per kind.

Metadata paging is not a bound on drafts, axis cards or field/port contexts.
Private working copies only; no fits, registry admission or host Torch.
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
from app.core import Workspace, default_model
from app.desktop import Bridge, Window
from app.science_browser import encoded, MAX_BYTES
from examples.experiment_workspace import populate

assert not os.environ.get('QTWEBENGINE_DISABLE_SANDBOX')
assert '--no-sandbox' not in os.environ.get('QTWEBENGINE_CHROMIUM_FLAGS', '')
app = QApplication(['authoring-revision-choices-smoke']); view = QWebEngineView(); view.resize(1500, 1100)
profile = QWebEngineProfile(view); errors = []


class Page(QWebEnginePage):
    def javaScriptConsoleMessage(self, level, message, line, source):
        if level == self.JavaScriptConsoleMessageLevel.ErrorMessageLevel: errors.append(f'{source}:{line}: {message}')


class Harness(QObject):
    state = Window.state; describe = Window.describe

    def __init__(self, folder):
        super().__init__(); self.config = Config(folder / 'config.json'); self.workspace = Workspace()
        example = populate(self.workspace); self.refs = {}; self.calls = []; self.reads = []
        for kind, source in [('dataset', example['data_tasks'][0]['dataset']), ('task', example['data_tasks'][0]['task']), ('model', example['models'][0])]:
            template = self.workspace.get_spec(kind, source['id'], source['revision']); template.pop('revision')
            for i in range(55):
                doc = copy.deepcopy(template); doc.update(id=f'choices-{kind}-{i:02}', name=f'Choices {kind} {i:02} · old')
                if kind=='model':
                    model=default_model(doc['name'],inputs=1);model['id']=doc['id'];self.workspace.put(model);old=self.workspace.freeze_model(model['id'])
                    model['name']=f'Choices {kind} {i:02} · current';self.workspace.put(model);head=self.workspace.freeze_model(model['id'])
                else:
                    old = self.workspace.put_spec(kind, doc)
                    doc['name'] = f'Choices {kind} {i:02} · current'; head = self.workspace.put_spec(kind, doc)
                if i == 54: self.refs[kind] = (old, head)
        self.job = None; self.torch_python = None; self.rendering = {}; self.bridge = Bridge(self, view)

    def recent_files(self): return []
    def broadcast(self, kind='state', payload=None, **kw): self.bridge.send(kind, self.state() if kind == 'state' else payload)
    def dispatch(self, request, source=None):
        self.calls.append(request['action']); result = Window.dispatch(self, request, source)
        if request['action'] == 'scienceBrowse':
            size = len(encoded(result)); assert size <= MAX_BYTES and len(result['rows']) <= 25
            self.reads.append({'kind': result['kind'], 'bytes': size, 'rows': len(result['rows']), 'offset': result['offset']})
        return result


def js(script):
    loop = QEventLoop(); timer = QTimer(); timer.setSingleShot(True); timer.timeout.connect(loop.quit); values = []
    view.page().runJavaScript(script, lambda value: (values.append(value), loop.quit()))
    timer.start(3000); loop.exec(); timer.stop(); assert values, 'JavaScript timeout'; return values[0]


def until(script, timeout=20):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if script() if callable(script) else js(script): return
        loop = QEventLoop(); QTimer.singleShot(20, loop.quit); loop.exec()
    raise AssertionError(f'UI timeout: {script}; console: {errors[:5]}')


def click(sid):
    until('!document.getElementById(' + json.dumps(sid) + ').disabled'); js('document.getElementById(' + json.dumps(sid) + ').click()')


def picker_action(selector, action):
    until('!(' + selector + ').closest(".revision-choice-field").querySelector("[data-choice-action=' + action + ']").disabled')
    js('(' + selector + ').closest(".revision-choice-field").querySelector("[data-choice-action=' + action + ']").click()')


def search(selector, query):
    js('(' + selector + ').closest(".revision-choice-field").querySelector("input[type=search]").value=' + json.dumps(query))
    picker_action(selector, 'search')
    until('!(' + selector + ').disabled')


def choose(selector, reference):
    value = json.dumps(reference, separators=(',', ':'))
    until('[...(' + selector + ').options].some(o=>o.value===' + json.dumps(value) + ') && !(' + selector + ').disabled')
    js('(' + selector + ').value=' + json.dumps(value) + ';(' + selector + ').dispatchEvent(new Event("change"))')

def experiment_reference(axis,label):
    return '[...document.querySelectorAll("[data-experiment-axis='+axis+'] label")].find(l=>l.firstChild.textContent==='+json.dumps(label)+')?.querySelector("select")'


with tempfile.TemporaryDirectory() as directory:
    folder = Path(directory); h = Harness(folder)
    page = Page(profile, view); view.setPage(page); channel = QWebChannel(page); channel.registerObject('host', h.bridge); page.setWebChannel(channel)
    url = QUrl.fromLocalFile(str(Path(__file__).resolve().parents[1] / 'app/web/index.html')); url.setFragment('model=' + next(iter(h.workspace.models()))); view.setUrl(url); view.show()
    try:
        until('Boolean(window.loom?.document?.view.fitted)')  # Camera is session-only.
        before = h.workspace.db.serialize()
        js('document.querySelector("[data-panel=science]").click();document.getElementById("experiment-dock").style.height="820px";document.getElementById("experiment-dock").style.width="1000px"')
        click('science-author-new-task'); until('!document.getElementById("science-task-model").disabled && !document.getElementById("science-task-dataset").disabled')
        model = 'document.getElementById("science-task-model")'; dataset = 'document.getElementById("science-task-dataset")'
        assert js('(' + dataset + ').options.length') == 26
        pinned = js('(' + dataset + ').value'); picker_action(dataset, 'next')
        until('(' + dataset + ').closest(".revision-choice-field").querySelector("[role=status]").textContent.startsWith("25–50")')
        assert js('(' + dataset + ').value') == pinned; assert js('(' + dataset + ').options.length') == 27
        search(dataset, 'Choices dataset 54'); choose(dataset, h.refs['dataset'][0])
        until('document.getElementById("science-task-context-status").textContent.startsWith("Exact context")')
        search(model, 'Choices model 54'); choose(model, h.refs['model'][0])
        old_model = js('(' + model + ').value'); old_data = js('(' + dataset + ').value')
        # Model/dataset suggestion changes must not discard unfinished task text.
        js('[...document.querySelectorAll("#science-task-bindings label")].find(l=>l.firstChild.textContent==="Task description").querySelector("textarea").value="unfinished <img> description"')
        click('science-task-context-refresh'); until('!(' + model + ').disabled && !(' + dataset + ').disabled')
        assert js('(' + model + ').value') == old_model and js('(' + dataset + ').value') == old_data
        assert js('[...document.querySelectorAll("#science-task-bindings label")].find(l=>l.firstChild.textContent==="Task description").querySelector("textarea").value') == 'unfinished <img> description'
        click('science-author-cancel'); click('science-author-new-experiment'); until('!document.getElementById("science-author-name").disabled')
        js('[...document.querySelectorAll("#science-experiment-bindings button")].find(b=>b.textContent==="Add architecture selection").click()')
        exp_model = experiment_reference('model','Exact architecture')
        until('(' + exp_model + ') && !(' + exp_model + ').disabled')
        assert js('(' + exp_model + ').value') == '', 'Experiment selections must not be guessed'
        search(exp_model, 'Choices model 54'); choose(exp_model, h.refs['model'][0]); until('!(' + exp_model + ').disabled')
        js('[...document.querySelectorAll("#science-experiment-bindings button")].find(b=>b.textContent==="Add dataset/task pairing").click()')
        exp_data = experiment_reference('data-task','Exact dataset')
        exp_task = experiment_reference('data-task','Exact task')
        until('(' + exp_data + ') && !(' + exp_data + ').disabled && !(' + exp_task + ').disabled')
        search(exp_data, 'Choices dataset 54'); choose(exp_data, h.refs['dataset'][0]); until('!(' + exp_data + ').disabled')
        search(exp_task, 'Choices task 54'); choose(exp_task, h.refs['task'][0]); until('!(' + exp_task + ').disabled')
        values = [js('(' + s + ').value') for s in [exp_model, exp_data, exp_task]]
        js('[...document.querySelectorAll("#science-experiment-bindings textarea")][0].value="unfinished research question"')
        click('science-experiment-refresh'); until('!(' + exp_model + ').disabled && !(' + exp_data + ').disabled && !(' + exp_task + ').disabled')
        assert [js('(' + s + ').value') for s in [exp_model, exp_data, exp_task]] == values
        assert js('[...document.querySelectorAll("#science-experiment-bindings textarea")][0].value') == 'unfinished research question'
        h.broadcast('externalReload', 'Private authoring reload')
        # Owned drafts retain unfinished work but freeze on actual reload; they
        # must not silently admit a revision against the replacement origin.
        until('document.getElementById("science-author-status").textContent.includes("Workspace reloaded")')
        assert js('document.getElementById("science-author-save").disabled && document.getElementById("science-author-check").disabled')
        assert js('(' + exp_model + ').disabled && (' + exp_data + ').disabled && (' + exp_task + ').disabled')
        assert [js('(' + s + ').value') for s in [exp_model, exp_data, exp_task]] == values
        assert js('[...document.querySelectorAll("#science-experiment-bindings textarea")][0].value') == 'unfinished research question'
        for theme, font, spacing in [('dark', 14, 1), ('github', 18, 4), ('math', 28, 10)]:
            h.config.values.update(theme=theme, font_size=font, spacing=spacing); h.broadcast()
            until('getComputedStyle(document.body).getPropertyValue("--ui-font-size").trim()===' + json.dumps(str(font) + 'px'))
            assert js('[...document.querySelectorAll(".revision-choice-field")].every(e=>e.getBoundingClientRect().width<=document.getElementById("science-panel").getBoundingClientRect().width)')
            assert js('[...document.querySelectorAll(".revision-choice-field>label")].every(l=>l.querySelectorAll("input,select,button,textarea").length===1)')
            assert js('parseFloat(getComputedStyle(document.querySelector(".revision-choice input")).fontSize)') >= font - 2
        if os.environ.get('VTW_SMOKE_SCREENSHOT'):
            h.config.values.update(theme='github', font_size=18, spacing=4); h.broadcast()
            until('getComputedStyle(document.body).getPropertyValue("--ui-font-size").trim()==="18px"')
            js('document.getElementById("science-experiment-bindings").scrollIntoView({block:"start"});document.getElementById("toast-close").click()')
            loop = QEventLoop(); QTimer.singleShot(350, loop.quit); loop.exec(); assert view.grab().save(os.environ['VTW_SMOKE_SCREENSHOT'])
        assert h.workspace.db.serialize() == before
        assert 'scienceIndex' not in h.calls and 'scienceTrain' not in h.calls
        assert h.workspace.list_specs('run', True) == [] and 'torch' not in sys.modules
        assert not errors, errors
        print(json.dumps({'revisions_per_kind': 112, 'task_paging_search_historical': 'PASS', 'experiment_explicit_pickers': 'PASS', 'unfinished_text_retained': 'PASS', 'reload_freezes_owned_draft': 'PASS', 'label_structure_font28': 'PASS', 'sql_readonly': 'PASS', 'no_full_metadata_index': 'PASS', 'page_count': len(h.reads), 'projection_max_bytes': max(r['bytes'] for r in h.reads), 'host_torch': False, 'console_errors': errors}))
    finally:
        if h.workspace.history.directory:h.workspace.history.directory.cleanup()
        h.workspace.db.close(); view.close(); view.setPage(QWebEnginePage(view)); page.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete); profile.deleteLater(); QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
