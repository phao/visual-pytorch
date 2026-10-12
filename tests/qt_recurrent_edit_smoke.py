"""Sandboxed real WebEngine interface editing with native Apply/Cancel clicks."""
import copy
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from PyQt6.QtCore import QObject, QEventLoop, QTimer, QUrl, QPoint, Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication
from PyQt6.QtWebEngineCore import QWebEnginePage, QWebEngineProfile
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWebChannel import QWebChannel
from app.config import Config
from app.core import Workspace, validate_model
from app.desktop import Window, Bridge
from test_recurrent_reconfiguration import editable

assert not os.environ.get('QTWEBENGINE_DISABLE_SANDBOX')
assert '--no-sandbox' not in os.environ.get('QTWEBENGINE_CHROMIUM_FLAGS', '')
app = QApplication(['recurrent-edit-smoke']);errors = []
class Page(QWebEnginePage):
    def javaScriptConsoleMessage(self, level, message, line, source):
        if level == self.JavaScriptConsoleMessageLevel.ErrorMessageLevel:errors.append(f'{source}:{line}: {message}')
class Harness(QObject):
    state = Window.state;describe = Window.describe;dispatch = Window.dispatch
    def __init__(self, path):
        super().__init__();self.config = Config(path/'config.json');self.config.save(ui_zoom=1)
        self.workspace = Workspace();models, token, _ = editable('--cnn' in sys.argv);self.nid = token['node']
        for model in models.values():self.workspace.put(model)
        self.workspace.dirty = False;self.job = None;self.torch_python = None;self.rendering = {};self.bridges = []
    def recent_files(self):return []
    def broadcast(self, kind='state', payload=None, **kwargs):
        for bridge in self.bridges:bridge.send(kind, self.state() if kind == 'state' else payload)
def js(view, script):
    loop = QEventLoop();values = [];timer = QTimer();timer.setSingleShot(True);timer.timeout.connect(loop.quit)
    view.page().runJavaScript(script, lambda value: (values.append(value), loop.quit()));timer.start(4000);loop.exec();timer.stop()
    assert values, 'JavaScript timeout';return values[0]
def until(view, script, seconds=15):
    deadline = time.monotonic()+seconds
    while time.monotonic() < deadline:
        if js(view, script):return
        QTest.qWait(25)
    raise AssertionError((script, errors, js(view, 'document.getElementById("recurrent-edit-error").textContent')))
def click(view, selector):
    js(view, 'document.querySelector('+json.dumps(selector)+').scrollIntoView({block:"nearest"})');QTest.qWait(40)
    p = js(view, '(()=>{const r=document.querySelector('+json.dumps(selector)+').getBoundingClientRect();return {x:r.left+r.width/2,y:r.top+r.height/2};})()')
    QTest.mouseClick(view.focusProxy() or view, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(round(p['x']), round(p['y'])));QTest.qWait(60)
def open_edit(view):
    js(view, 'window.loom.fit()');QTest.qWait(60)
    click(view, '.node.Recurrent')
    until(view, 'Boolean(document.getElementById("edit-recurrent-interface"))')
    js(view, 'document.getElementById("edit-recurrent-interface").click()')
    until(view, 'document.getElementById("recurrent-edit-dialog").open && !document.getElementById("recurrent-edit-apply").disabled')

with tempfile.TemporaryDirectory() as tmp:
    h = Harness(Path(tmp));view = QWebEngineView();view.resize(1450,980)
    profile = QWebEngineProfile(view);page = Page(profile, view);view.setPage(page)
    bridge = Bridge(h, view);h.bridges = [bridge];channel = QWebChannel(page);channel.registerObject('host', bridge);page.setWebChannel(channel)
    url = QUrl.fromLocalFile(str(Path(__file__).resolve().parents[1]/'app/web/index.html'));url.setFragment('model=parent')
    view.setUrl(url);view.show();until(view, 'Boolean(window.loom?.document)')
    try:
        before = h.workspace.models();history = len(h.workspace.history.undo_stack);config = h.config.path.read_bytes()
        open_edit(view)
        initial = js(view, 'document.getElementById("recurrent-edit-json").value')
        options = json.loads(initial)
        r = next(n for n in before['parent']['graph']['nodes'] if n['id'] == h.nid)
        assert options['sequence_inputs'] == r['sequence_inputs'] and options['states'] == r['states'] and options['controls'] == r['controls']
        # Connected return-port removal is refused; draft is kept on screen.
        js(view, '(()=>{const e=document.getElementById("recurrent-edit-json"),o=JSON.parse(e.value);o.sequence_outputs=[];e.value=JSON.stringify(o);})()')
        click(view, '#recurrent-edit-apply')
        until(view, 'document.getElementById("recurrent-edit-error").textContent.includes("connected return port")')
        assert h.workspace.models() == before and len(h.workspace.history.undo_stack) == history
        assert js(view, 'document.getElementById("recurrent-edit-dialog").open')
        # Invalid JSON and duplicate keys must never become partial graph edits.
        for text in ['{', '{"version":1,"version":1}']:
            js(view, 'document.getElementById("recurrent-edit-json").value='+json.dumps(text))
            click(view, '#recurrent-edit-apply');until(view, 'Boolean(document.getElementById("recurrent-edit-error").textContent)')
            assert h.workspace.models() == before and len(h.workspace.history.undo_stack) == history
        options['label'] = 'Edited existing scan';options['sequence_outputs'].append({'output': 'next', 'port': 'extra'})
        js(view, 'document.getElementById("recurrent-edit-json").value='+json.dumps(json.dumps(options, indent=2)))
        QTest.qWait(100)
        capture = os.environ.get('VP_EDIT_CAPTURE')
        if capture:assert view.grab().save(capture)
        click(view, '#recurrent-edit-apply');until(view, '!document.getElementById("recurrent-edit-dialog").open')
        after = h.workspace.models();edited = next(n for n in after['parent']['graph']['nodes'] if n['id'] == h.nid)
        assert edited['label'] == 'Edited existing scan' and edited['sequence_outputs'][-1]['port'] == 'extra'
        assert after['parent']['graph']['edges'] == before['parent']['graph']['edges']
        assert [n['id'] for n in after['parent']['graph']['nodes']] == [n['id'] for n in before['parent']['graph']['nodes']]
        assert len(h.workspace.history.undo_stack) == history+1
        until(view, 'document.querySelectorAll(".node.Recurrent .port-out").length===3')
        js(view, 'window.loom.perform("undo")');until(view, 'document.querySelectorAll(".node.Recurrent .port-out").length===2')
        assert h.workspace.models() == before
        js(view, 'window.loom.perform("redo")');until(view, 'document.querySelectorAll(".node.Recurrent .port-out").length===3')
        assert h.workspace.models() == after
        # A changed transition after opening cannot accept stale bindings.
        open_edit(view);body = h.workspace.models()['transition'];body['name'] += ' changed';h.workspace.put(body)
        snapshot = h.workspace.models();history = len(h.workspace.history.undo_stack)
        click(view, '#recurrent-edit-apply');until(view, 'document.getElementById("recurrent-edit-error").textContent.includes("changed")')
        assert h.workspace.models() == snapshot and len(h.workspace.history.undo_stack) == history
        click(view, '#recurrent-edit-cancel');until(view, '!document.getElementById("recurrent-edit-dialog").open')
        saved = Path(tmp)/'edited.vtw';h.workspace.save(saved);loaded = Workspace.load(saved)
        try:
            assert loaded.models() == snapshot
            assert validate_model(loaded.models(), 'parent')['parameters'] == (17 if '--cnn' in sys.argv else 13)
        finally:loaded.db.close()
        assert h.config.path.read_bytes() == config, 'Private config changed unexpectedly'
        assert not errors, errors
        assert 'torch' not in sys.modules, 'Qt host imported Torch'
        print(json.dumps({'prefilled_existing_interface': 'PASS', 'native_apply_cancel_and_draft_errors': 'PASS',
          'stable_blocks_wires_and_new_port': 'PASS', 'one_undo_redo': 'PASS', 'stale_transition_reload': 'PASS',
          'cnn': '--cnn' in sys.argv, 'console_errors': errors, 'host_torch': False, 'sandbox': True}))
    finally:
        view.close();page.deleteLater();view.deleteLater();QTest.qWait(100);h.workspace.db.close()
