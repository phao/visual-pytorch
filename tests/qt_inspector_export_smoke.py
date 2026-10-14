"""Actual desktop window: essentials-first Inspector, port-name boxes, Export panel.

Covers every block type found in the shipped examples plus UI-built, CNN,
recurrent, LSTM and router graphs. Checks Inspector order/folding/ID-free text,
session fold memory, About pages, GUI default port names, the Export panel
(X key, highlighted source, scoped Ctrl+A, Copy, Save as .py, invalid graph,
SVG placeholder, context-menu route) and that Replicator is no longer offered
while existing Replicators still validate and inspect.
Private config/settings and temporary workspace copies only; sandbox on; no host Torch.
Set VTW_CAPTURE_DIR to also save PNG captures of the panels.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch
from PyQt6.QtCore import QCoreApplication, QEvent, QEventLoop, QTimer, Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication
from PyQt6.QtWebEngineCore import QWebEnginePage
from app.config import Config
from app.context_menu import build_context_menu
from app.core import Workspace, default_cnn, default_mlp, validate_model
from app.desktop import HELP_RESOURCES, Window
from examples.projected_lstm import models as lstm_models
from examples.recurrent_graph import graph_example

ROOT = Path(__file__).resolve().parents[1]
assert not os.environ.get('QTWEBENGINE_DISABLE_SANDBOX')
assert '--no-sandbox' not in os.environ.get('QTWEBENGINE_CHROMIUM_FLAGS', '')
app = QApplication(['inspector-export-smoke'])
errors = []
captures = Path(os.environ['VTW_CAPTURE_DIR']) if os.environ.get('VTW_CAPTURE_DIR') else None
SUB_ORDER = ['shapes', 'axes', 'ports', 'sharing', 'policy', 'advanced']
STATUS_NOTES = ('Loading parameter owners…', 'This block owns no learned parameters.',
                'Pinned tensor source', 'Shared canonical owner')


class Page(QWebEnginePage):
    def javaScriptConsoleMessage(self, level, message, line, source):
        if level == self.JavaScriptConsoleMessageLevel.ErrorMessageLevel:
            errors.append(f'{source}:{line}:{message}')


class Settings:
    def __init__(self, *args): self.data = {}
    def value(self, key, default=None, **kwargs): return self.data.get(key, default)
    def setValue(self, key, value): self.data[key] = value


def js(source):
    values = []; loop = QEventLoop(); timer = QTimer(); timer.setSingleShot(True)
    timer.timeout.connect(loop.quit)
    view.page().runJavaScript(source, lambda value: (values.append(value), loop.quit()))
    timer.start(6000); loop.exec(); timer.stop()
    assert values, 'JavaScript callback timeout: ' + source[:80]
    return values[0]


def until(source, timeout=20):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        assert not errors, errors
        if js(source): return
        QTest.qWait(25)
    raise AssertionError('UI timeout: ' + source)


def grab(name):
    if captures:
        captures.mkdir(parents=True, exist_ok=True); QTest.qWait(400)
        view.grab().save(str(captures / (name + '.png')))


def key(code, modifier=Qt.KeyboardModifier.NoModifier):
    QTest.keyClick(view.focusProxy() or view, code, modifier); QTest.qWait(150)


def navigate(mid):
    window.broadcast(); view.bridge.send('navigate', mid)
    until(f'window.loom?.document?.id==={json.dumps(mid)} && Boolean(document.querySelector(".node"))')


def inspect(nid):
    js(f'window.loom.perform("inspect-block",{json.dumps(nid)},window.loom.document.id)')
    until(f'document.querySelector(".node.selected")?.dataset.id==={json.dumps(nid)} && !document.getElementById("inspector-panel").hidden')
    QTest.qWait(80)


STRUCTURE = '''(()=>{
  const scroll=document.getElementById('inspector-scroll'),contract=document.getElementById('contract-details');
  return {children:[...scroll.children].map(e=>e.id),contractOpen:contract.open,contractHidden:contract.hidden,
    essentials:Boolean(document.querySelector('#inspector-content > .essentials')),
    subs:[...document.querySelectorAll('#inspector-content > details.sub')].map(d=>d.dataset.sub),
    open:[...document.querySelectorAll('#inspector-content > details.sub')].filter(d=>d.open).map(d=>d.dataset.sub),
    notes:[...scroll.querySelectorAll('.inspect-note')].map(p=>p.textContent),
    title:document.querySelector('#inspector-header .inspect-title')?.textContent,
    about:Boolean(document.getElementById('inspector-about')),
    text:scroll.innerText+' '+[...scroll.querySelectorAll('option')].map(o=>o.textContent).join(' ')};
})()'''


def check_structure(nid, kind, internal_ids):
    info = js(STRUCTURE)
    assert info['children'] == ['inspector-header', 'contract-details', 'inspector-content'], info['children']
    assert not info['contractHidden'], nid
    assert info['essentials'], nid
    assert info['subs'] == sorted(info['subs'], key=SUB_ORDER.index), (nid, info['subs'])
    assert all(note.startswith(STATUS_NOTES) for note in info['notes'] if note.strip()), (kind, info['notes'])
    leaked = [i for i in internal_ids if re.search(r'(?<![\w-])' + re.escape(i) + r'(?![\w-])', info['text'])]
    assert not leaked, (kind, nid, leaked)
    return info


def internal(models):
    # IDs that are not words a user could have chosen as a port name or label.
    return sorted({n['id'] for doc in models.values() for n in doc['graph']['nodes']
                   if re.fullmatch(r'n_[a-z0-9]{8}', n['id']) or re.search(r'\d', n['id'])} - {
                   n.get('label') for doc in models.values() for n in doc['graph']['nodes']})


def routed(mode):
    # Real webview graph-tools construction, as tests/test_mlp.py does, without Torch.
    source = ("let t='';process.stdin.on('data',x=>t+=x);process.stdin.on('end',()=>{const d=JSON.parse(t),"
              "g=require('./app/web/graph-tools.js');const extra=g.addInput(d);d.graph.nodes.find(n=>n.id===extra).label='context';"
              "g.addRouter(d,process.argv[1],3);process.stdout.write(JSON.stringify(d));});")
    doc = default_mlp(inputs=3, outputs=3, hidden=[8, 8])
    out = subprocess.run(['node', '-e', source, mode], input=json.dumps(doc), text=True,
                         capture_output=True, check=True, cwd=ROOT, timeout=10)
    return json.loads(out.stdout)


seen_types = set()
opened = []
with tempfile.TemporaryDirectory() as directory:
    folder = Path(directory); config = Config(folder / 'preferences.json'); config.save()
    saved = folder / 'exported.py'
    with patch('app.desktop.Config', return_value=config), patch('app.desktop.QSettings', Settings), \
         patch('app.desktop.QWebEnginePage', Page), \
         patch('app.desktop.QFileDialog.getSaveFileName', return_value=(str(saved), 'Python module (*.py)')), \
         patch('app.desktop.QDesktopServices.openUrl', side_effect=lambda url: opened.append(url.toString()) or True):
        window = Window(torch_python='private-test-no-worker')
        window.file_timer.stop(); window.resize(1440, 900); window.show()
        view = window.current_view(); until('Boolean(window.loom)')
        window.workspace = Workspace(); window.workspace.save(folder / 'built.vtw')
        window.broadcast()
        try:
            # R5: GUI-created architecture gets readable port names; boxes show the port name.
            js('window.loom.call("newModel",{name:"Built"}).then(id=>window.__mid=id)')
            until('Boolean(window.__mid) && window.loom?.document?.id===window.__mid')
            mid = js('window.__mid')
            until('document.querySelector(\'.node[data-id="output"] .node-title\')?.firstChild.nodeValue==="Prediction ŷ"')
            assert js('document.querySelector(\'.node[data-id="input"] .node-title\').firstChild.nodeValue') == 'Input x'
            js('window.loom.perform("reset-panels")')
            inspect('output')
            assert js('document.getElementById("port-label").value') == 'Prediction ŷ'
            js('(()=>{const f=document.getElementById("port-label");f.value="Price";f.dispatchEvent(new Event("change"));})()')
            until('document.querySelector(\'.node[data-id="output"] .node-title\')?.firstChild.nodeValue==="Price"')
            js('(()=>{const f=document.getElementById("port-label");f.value="Prediction ŷ";f.dispatchEvent(new Event("change"));})()')
            until('document.querySelector(\'.node[data-id="output"] .node-title\')?.firstChild.nodeValue==="Prediction ŷ"')

            # Build Input 3 -> Linear 4 -> ReLU -> Linear 2 with real palette actions.
            inspect('input'); js('(()=>{const f=document.getElementById("sample-shape");f.value="3";f.dispatchEvent(new Event("change"));})()')
            inspect('linear1'); js('(()=>{const f=document.getElementById("features");f.value="4";f.dispatchEvent(new Event("change"));})()')
            js('document.getElementById("add-linear").click()')
            second = js('window.loom.document.graph.nodes.at(-1).id')
            inspect(second); js('(()=>{const f=document.getElementById("features");f.value="2";f.dispatchEvent(new Event("change"));})()')
            inspect('linear1'); js('(()=>{const s=document.getElementById("activation-select");s.value="ReLU";s.dispatchEvent(new Event("change"));document.getElementById("add-activation").click();})()')
            until('window.loom.document.graph.nodes.length===5')
            until('document.getElementById("status").textContent.includes("26 parameters")')

            # R1/R3: Linear essentials, folded order, readable sharing names.
            inspect(second)
            info = check_structure(second, 'Linear', [second, 'linear1'])
            assert info['subs'] == ['shapes', 'sharing', 'policy'] and not info['open'] and not info['contractOpen'], info
            assert info['title'] == 'Linear' and info['about']
            assert js('document.querySelector(".inspect-summary").textContent') == '[B, 4] → [B, 2] · 10 parameters'
            assert js('[...document.querySelectorAll("#parameter-owner option")].map(o=>o.textContent).join("|")') == 'Independent parameters|Linear 3 → 4'
            js('document.querySelector(\'details.sub[data-sub="policy"]\').open=true')
            until('Boolean(document.querySelector(".policy-owner-title"))')
            owners = js('[...document.querySelectorAll(".policy-owner-title")].map(e=>e.textContent).join("|")')
            assert owners.startswith('Linear 4 → 2') and second not in owners, owners
            grab('inspector-linear')

            # R4: fold memory across selections; no dirty/undo change from folding.
            before = js('JSON.stringify([window.loom.state.dirty,window.loom.state.history])')
            js('document.querySelector(\'details.sub[data-sub="shapes"]\').open=true;document.getElementById("contract-details").open=true')
            QTest.qWait(150)
            inspect('linear1')
            assert js('document.querySelector(\'details.sub[data-sub="shapes"]\').open') is True
            assert js('document.getElementById("contract-details").open') is True
            assert js('document.querySelector(\'details.sub[data-sub="policy"]\').open') is True
            assert js('JSON.stringify([window.loom.state.dirty,window.loom.state.history])') == before
            until('document.getElementById("equation").textContent.includes("x")')
            grab('inspector-linear-expanded')
            for name in ('shapes', 'policy'):
                js(f'document.querySelector(\'details.sub[data-sub="{name}"]\').open=false')
            js('document.getElementById("contract-details").open=false'); QTest.qWait(150)

            # About button opens the block's About page (existing dialog).
            js('document.getElementById("inspector-about").click()')
            until('document.getElementById("block-about-dialog").open')
            assert js('document.getElementById("block-about-title").textContent') == 'About Linear'
            assert 'Parameter sharing (Inspector)' in js('document.getElementById("block-about-content").textContent')
            # R9: PyTorch documentation links open in the user's browser, never the webview.
            page_url = js('location.href')
            assert js('[...document.querySelectorAll("#block-about-content [data-help-resource]")].map(a=>a.textContent).join("|")') == 'torch.nn.Linear'
            js('document.querySelector("#block-about-content [data-help-resource]").click()')
            deadline = time.monotonic() + 10
            while not opened and time.monotonic() < deadline: QTest.qWait(50)
            assert opened == [HELP_RESOURCES['torch.nn.Linear']], opened
            assert js('location.href') == page_url
            grab('about-linear')
            js('document.getElementById("block-about-dialog").close()')
            names = js('JSON.stringify([...Object.values(window.LoomBlockHelp.docs),...Object.values(window.LoomBlockHelp.tensorDocs)].flat())')
            activations = js('JSON.stringify(Object.keys(window.LoomActivations.info).map(k=>"torch.nn."+k))')
            missing = set(json.loads(names)) | set(json.loads(activations))
            assert missing <= set(HELP_RESOURCES), sorted(missing - set(HELP_RESOURCES))

            # R6: Export panel via the X key, highlighted, live, scoped Ctrl+A, Copy, Save.
            js('document.activeElement?.blur();document.getElementById("canvas").focus()')
            key(Qt.Key.Key_X)
            until('!document.getElementById("export-panel").hidden')
            until('Boolean(document.querySelector("#export-source .tok-keyword"))')
            code = js('document.getElementById("export-source").textContent')
            assert 'nn.Linear(4, 2' in code and 'nn.ReLU(' in code, [l for l in code.splitlines() if 'self.layer_' in l]
            grab('export-pytorch')
            inspect(second); js('(()=>{const f=document.getElementById("features");f.value="5";f.dispatchEvent(new Event("change"));})()')
            until('document.getElementById("export-source").textContent.includes("nn.Linear(4, 5")')
            code = js('document.getElementById("export-source").textContent')
            js('document.getElementById("export-source").focus()')
            key(Qt.Key.Key_A, Qt.KeyboardModifier.ControlModifier)
            # Exactly the code (Chromium's selection text omits the final newline).
            assert js('document.getSelection().toString()') == code.rstrip('\n')
            QApplication.clipboard().setText('sentinel')
            js('document.getElementById("export-copy").click()')
            deadline = time.monotonic() + 10
            while QApplication.clipboard().text() != code and time.monotonic() < deadline: QTest.qWait(50)
            assert QApplication.clipboard().text() == code
            js('document.getElementById("export-code").click()')
            deadline = time.monotonic() + 10
            while not saved.exists() and time.monotonic() < deadline: QTest.qWait(50)
            assert saved.read_text(encoding='utf-8') == code
            # Invalid graph (Output shape declared against its wire): message, actions disabled.
            inspect('output')
            js('(()=>{const f=document.getElementById("sample-shape");f.value="7";f.dispatchEvent(new Event("change"));})()')
            until('document.getElementById("export-source").classList.contains("invalid")')
            assert js('document.getElementById("export-copy").disabled && document.getElementById("export-code").disabled')
            grab('export-invalid')
            js('(()=>{const f=document.getElementById("sample-shape");f.value="";f.dispatchEvent(new Event("change"));})()')
            until('!document.getElementById("export-source").classList.contains("invalid")')
            js('document.querySelector(\'[data-export-tab="svg"]\').click()')
            assert js('!document.getElementById("export-svg").hidden && document.getElementById("export-pytorch").hidden')
            assert 'not yet available' in js('document.getElementById("export-svg").textContent')
            js('document.querySelector(\'[data-export-tab="pytorch"]\').click()')
            # Dock has no PyTorch tab; context menu and perform routes open Export.
            assert js('!document.querySelector(\'[data-panel="code"]\') && !document.getElementById("code-panel")')
            js('window.loom.perform("reset-panels")')
            menu = build_context_menu(view, window)
            labels = [a.text() for a in menu.actions()]
            assert 'Export panel' in labels and 'PyTorch forward' not in labels, labels
            next(a for a in menu.actions() if a.text() == 'Export panel').trigger()
            until('!document.getElementById("export-panel").hidden')
            js('window.loom.perform("reset-panels")'); js('window.loom.perform("code")')
            until('!document.getElementById("export-panel").hidden')
            # Saved panel presets keep their closed grammar (Export is session-only).
            js('window.loom.perform("reset-panels")')

            # R7: Replicator is no longer offered.
            assert js('!document.getElementById("add-replicator")')
            assert 'Replicator' not in js('document.getElementById("library-panel").textContent')
            seen_types |= {'Input', 'Output', 'Linear', 'Activation'}

            # R1/R3 across every other block type, from real fixtures.
            fixtures = []
            for name in ('lora.vtw', 'neural-msvar.vtw'):
                copy = folder / name; shutil.copy(ROOT / 'examples' / name, copy)
                fixtures.append(Workspace.load(copy))
            extra = Workspace()
            for models, _ in (graph_example(), lstm_models()):
                for doc in models.values(): extra.put(doc)
            extra.put(default_cnn('CNN'))
            constant = default_mlp('Constant source', inputs=2, outputs=1, hidden=[3])
            constant['graph']['nodes'] += [{'id': 'offsets', 'type': 'Constant', 'op_version': 1, 'shape': [2], 'fill': 1.0, 'x': 90, 'y': 320},
                                           {'id': 'offsets_out', 'type': 'Output', 'label': 'Offsets', 'x': 340, 'y': 320}]
            constant['graph']['edges'].append({'from': 'offsets', 'to': 'offsets_out'})
            extra.put(constant)
            for mode in ('Splitter', 'Replicator'):
                doc = routed(mode); doc['name'] = mode; extra.put(doc)
            fixtures.append(extra)
            replicator_checked = False
            for workspace in fixtures:
                window.workspace = workspace
                models = workspace.models()
                ids = internal(models)
                for model_id, doc in sorted(models.items()):
                    try: validate_model(models, model_id)
                    except Exception: continue  # drafts stay editable; structure checks need a valid contract
                    navigate(model_id)
                    for n in doc['graph']['nodes']:
                        inspect(n['id'])
                        info = check_structure(n['id'], n['type'], ids)
                        if n['type'] in ('Input', 'Output'):
                            assert info['title'] == (n.get('label') or n['id']), info['title']
                        if n['type'] == 'Replicator':
                            replicator_checked = True; assert info['about']
                        if n['type'] not in seen_types and n['type'] in ('GaussianFilter', 'LSTM', 'Module', 'Conv2d'):
                            grab('inspector-' + n['type'])
                        seen_types.add(n['type'])
            assert replicator_checked
            js('window.loom.perform("reset-panels")')
            assert not errors, errors
            print(json.dumps({'ok': True, 'block_types': sorted(seen_types), 'console_errors': errors,
                              'host_torch': 'torch' in sys.modules}))
        finally:
            window.workspace.dirty = False; window.close()
            for i in range(window.tabs.count()):
                current = window.tabs.widget(i); page = current.page()
                current.setPage(QWebEnginePage(current)); page.deleteLater()
            QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
            window.deleteLater()
            QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
