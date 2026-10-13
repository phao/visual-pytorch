"""Sandboxed real LoRA example: native controls, worker, undo and persistence.

Usage: system python tests/qt_lora_example_smoke.py EXISTING_LORA.vtw NEW_OUTPUT
Private in-memory copy and config; never overwrite the source or restart the app.
"""
import copy
import json
import os
import sys
import time
from pathlib import Path
from unittest.mock import Mock
from PyQt6.QtCore import QObject, QEvent, QCoreApplication, QEventLoop, QTimer, QUrl, QPoint, Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication
from PyQt6.QtWebChannel import QWebChannel
from PyQt6.QtWebEngineCore import QWebEnginePage, QWebEngineProfile
from PyQt6.QtWebEngineWidgets import QWebEngineView
from app.core import Workspace
from app.config import Config
from app.desktop import Window, Bridge, find_torch_python

assert not os.environ.get('QTWEBENGINE_DISABLE_SANDBOX')
assert '--no-sandbox' not in os.environ.get('QTWEBENGINE_CHROMIUM_FLAGS', '')
source = Path(sys.argv[1]).resolve(); output = Path(sys.argv[2]).resolve(); output.mkdir(exist_ok=False)
original = source.read_bytes()
app = QApplication(['lora-example-acceptance']); view = QWebEngineView(); view.resize(1500,1000)
profile = QWebEngineProfile(view); errors = []


class Page(QWebEnginePage):
    def javaScriptConsoleMessage(self, level, message, line, url):
        if level == self.JavaScriptConsoleMessageLevel.ErrorMessageLevel: errors.append(f'{url}:{line}:{message}')


class Harness(QObject):
    state = Window.state; describe = Window.describe; dispatch = Window.dispatch
    def __init__(self):
        super().__init__(); self.workspace = Workspace.load(source)
        self.config = Config(output / 'config.json'); self.config.values['ui_zoom'] = 1.
        self.config.save = Mock(side_effect=AssertionError('No configuration writes'))
        self.job = None; self.rendering = {}; self.torch_python = find_torch_python(); assert self.torch_python
        self.bridge = Bridge(self, view); self.events = []
    def recent_files(self): return []
    def broadcast(self, kind='state', payload=None, **kw):
        self.events.append((kind, payload)); self.bridge.send(kind, self.state() if kind == 'state' else payload)


def js(code):
    values = []; loop = QEventLoop(); timer = QTimer(); timer.setSingleShot(True); timer.timeout.connect(loop.quit)
    view.page().runJavaScript(code, lambda value: (values.append(value), loop.quit()))
    timer.start(5000); loop.exec(); timer.stop(); assert values, 'JS timeout'; return values[0]


def until(code, timeout=45):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        assert not errors, errors
        if js(code): return
        QTest.qWait(25)
    raise AssertionError('UI timeout: ' + code)


def point(selector):
    js('document.querySelector(' + json.dumps(selector) + ').scrollIntoView({block:"nearest"})')
    QTest.qWait(100)
    value = js('(()=>{const e=document.querySelector(' + json.dumps(selector) + '),r=e.getBoundingClientRect();'
               'return [r.x+r.width/2,r.y+r.height/2,Boolean(e===document.elementFromPoint(r.x+r.width/2,r.y+r.height/2)||e.contains(document.elementFromPoint(r.x+r.width/2,r.y+r.height/2)))]})()')
    assert value[2], ('Control occluded', selector, value)
    return QPoint(round(value[0]), round(value[1]))


def click(selector):
    QTest.mouseClick(view.focusProxy() or view, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, point(selector))


def inspect(nid):
    js('document.querySelector(".node[data-id=' + nid + ']").dispatchEvent(new MouseEvent("dblclick",{bubbles:true}))')


h = Harness(); page = Page(profile, view); view.setPage(page)
channel = QWebChannel(page); channel.registerObject('host', h.bridge); page.setWebChannel(channel)
url = QUrl.fromLocalFile(str(Path(__file__).resolve().parents[1] / 'app/web/index.html'))
url.setFragment('model=lora-r2'); view.setUrl(url); view.show()
try:
    until('Boolean(window.loom?.document?.view.fitted)')
    until('document.querySelector(".node[data-id=base]")?.dataset.policyStatus==="Frozen"')
    assert js('document.querySelectorAll(".node[data-id=sum] .port-in").length') >= 2
    for nid in ('A', 'B'):
        assert js('document.querySelector(".node[data-id=' + nid + ']").dataset.policyStatus') == 'Trainable'
    for nid, role in (('base', 'Frozen'), ('A', 'Trainable'), ('B', 'Trainable')):
        inspect(nid); until('Boolean(document.querySelector(".policy-owner"))')
        assert js('document.getElementById("parameter-policy-editor").textContent').find(role) >= 0
    js('document.activeElement?.blur();document.body.dispatchEvent(new KeyboardEvent("keydown",{key:"Escape",bubbles:true}));document.body.dispatchEvent(new KeyboardEvent("keydown",{key:"Escape",bubbles:true}))')
    QTest.qWait(200); assert view.grab().save(str(output / '01-expanded-lora.png'))
    click('.node[data-id=B] .node-type-icon')
    until('window.loom.document.graph.nodes.find(n=>n.id==="B").collapsed===true')
    before = copy.deepcopy(next(n for n in h.workspace.models()['lora-r2']['graph']['nodes'] if n['id'] == 'B'))
    origin = point('.node[data-id=B] .node-type-icon'); target = origin + QPoint(55,35)
    surface = view.focusProxy() or view
    QTest.mousePress(surface, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, origin)
    QTest.mouseMove(surface, target, 150); QTest.mouseRelease(surface, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, target)
    until('window.loom.document.graph.nodes.find(n=>n.id==="B").x!==' + str(before['x']))
    until('!document.getElementById("toast-message").textContent.includes("Saving")')
    h.dispatch({'action': 'undo'}, h.bridge)
    until('window.loom.document.graph.nodes.find(n=>n.id==="B").x===' + str(before['x']))
    h.dispatch({'action': 'redo'}, h.bridge)
    until('window.loom.document.graph.nodes.find(n=>n.id==="B").x!==' + str(before['x']))
    until('(()=>{const n=window.loom.document.graph.nodes.find(n=>n.id==="B");return document.querySelector(".node[data-id=B]").getAttribute("transform")===`translate(${n.x},${n.y})`})()')
    QTest.qWait(350)
    # Qt synthetic mouseDClick produced two detail=1 browser clicks in the
    # preserved diagnostic trace. Verify the real documented keyboard path
    # instead; do not claim a native browser double-click from that injection.
    QTest.keyClick(surface, Qt.Key.Key_Space)
    until('!window.loom.document.graph.nodes.find(n=>n.id==="B").collapsed')
    baseline = h.workspace.get_spec('run', 'lora-linear-comparison-lora-r2-run')['attempts'][0]['result']
    js('document.querySelector("[data-panel=science]").click();document.getElementById("experiment-dock").style.height="700px"')
    until('document.getElementById("science-object").options.length===2')
    js('const e=document.getElementById("science-object");e.value=[...e.options].find(o=>JSON.parse(o.value).id==="lora-linear-comparison").value;e.dispatchEvent(new Event("change"))')
    until('document.querySelectorAll(".science-candidate").length===4')
    js('[...document.querySelectorAll(".science-candidate")].find(e=>e.textContent.includes("lora-r2")).querySelector("button").id="actual-lora-run"')
    click('#actual-lora-run'); until('Boolean(window.loom.state.experiment_job)')
    snapshot = copy.deepcopy(h.job.snapshot)
    until('!window.loom.state.experiment_job', timeout=90)
    assert h.job.wait(10000)
    new_ref = next(row for row in h.workspace.list_specs('run') if row['id'] == snapshot['run_id'])
    actual = h.workspace.get_spec('run', new_ref['id'], new_ref['revision'])['attempts'][0]
    assert actual['status'] == 'completed', actual['error']
    assert actual['result']['counts']['completed_updates'] == 200
    assert actual['result']['checkpoint_id'] == baseline['checkpoint_id']
    assert actual['result']['final_metrics'] == baseline['final_metrics']
    js('window.loom.perform("layout");document.getElementById("layout-expanded-section").open=true;document.getElementById("layout-expanded-parameter_sizing").checked=true;document.getElementById("expand-all").click()')
    until('window.loom.document.graph.nodes.find(n=>n.id==="base").collapsed && ["A","B"].every(id=>{const n=window.loom.document.graph.nodes.find(n=>n.id===id);return !n.collapsed&&n.height})')
    with (output / 'worker-snapshot.json').open('x') as stream: json.dump(snapshot, stream, allow_nan=False)
    with (output / 'worker-result.json').open('x') as stream: json.dump(actual, stream, allow_nan=False)
    h.workspace.save(output / 'final.vtw', require_new=True); restored = Workspace.load(output / 'final.vtw')
    try: assert restored.models()['lora-r2']['graph'] == h.workspace.models()['lora-r2']['graph']
    finally: restored.db.close()
    assert source.read_bytes() == original and 'torch' not in sys.modules and not errors
    print(json.dumps({'status': 'PASS', 'real_example': True, 'dual_paths_and_roles': True,
        'native_compact_icon_drag_undo_and_space_expand': True, 'real_gui_worker_updates': 200,
        'exact_original_checkpoint_and_metrics': True, 'learnable_layout_excludes_frozen_storage': True, 'save_reload': True,
        'untouched_source_and_config': True, 'host_torch': False, 'console_errors': errors}), flush=True)
finally:
    if h.job and h.job.isRunning(): h.job.stop(); h.job.wait(15000)
    if h.job: h.job.wait(15000)
    h.workspace.db.close()
    if h.workspace.history.directory: h.workspace.history.directory.cleanup()
    view.close(); view.setPage(QWebEnginePage(view)); page.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    profile.deleteLater(); QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
