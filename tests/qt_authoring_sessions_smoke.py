"""Actual sandboxed QWebChannel owner/session transport and SQLite history.

Private bridge fixture, not a claim that typed forms have migrated to sessions.
No Torch, numerical worker, Jupyter, user workspace or config is involved.
"""
import copy
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

from PyQt6.QtCore import QObject, QCoreApplication, QEvent, QEventLoop, QTimer
from PyQt6.QtWidgets import QApplication
from PyQt6.QtWebChannel import QWebChannel
from PyQt6.QtWebEngineCore import QWebEnginePage, QWebEngineProfile
from PyQt6.QtWebEngineWidgets import QWebEngineView
from app.core import Workspace
from app.desktop import Bridge, Window
from app.science_browser import encoded
from test_authoring import measurements

assert not os.environ.get('QTWEBENGINE_DISABLE_SANDBOX')
assert '--no-sandbox' not in os.environ.get('QTWEBENGINE_CHROMIUM_FLAGS', '')
app = QApplication(['owned-authoring-session-smoke']); view = QWebEngineView()
profile = QWebEngineProfile(view); errors = []


class Page(QWebEnginePage):
    def javaScriptConsoleMessage(self, level, message, line, source):
        if level == self.JavaScriptConsoleMessageLevel.ErrorMessageLevel: errors.append(message)


class Harness(QObject):
    dispatch = Window.dispatch

    def __init__(self):
        super().__init__(); self.workspace = Workspace(); self.job = None; self.events = []

    def broadcast(self, kind='state', payload=None, **kw): self.events.append(kind)


def js(script):
    loop = QEventLoop(); timer = QTimer(); timer.setSingleShot(True); values = []
    timer.timeout.connect(loop.quit)
    view.page().runJavaScript(script, lambda value: (values.append(value), loop.quit()))
    timer.start(5000); loop.exec(); timer.stop(); assert values, 'JavaScript timeout'; return values[0]


def until(script):
    end = time.monotonic()+15
    while time.monotonic() < end:
        if js(script): return
        loop = QEventLoop(); QTimer.singleShot(15, loop.quit); loop.exec()
    raise AssertionError('QWebChannel timeout: '+script)


def request(value, owner='host', ok=True):
    payload = value if isinstance(value, str) else json.dumps(value, allow_nan=False)
    js('window.answer=null;window.testChannel.objects['+json.dumps(owner)+'].request('+json.dumps(payload)+', reply=>window.answer=reply)')
    until('typeof window.answer === "string"')
    result = json.loads(js('window.answer')); assert result['ok'] is ok, result
    if ok: assert len(encoded(result['value'])) <= 65536
    return result.get('value') if ok else result['error']


h = Harness(); workspaces = [h.workspace]
second_view = QWebEngineView()
host = Bridge(h, view); second = Bridge(h, second_view)
page = Page(profile, view); view.setPage(page)
channel = QWebChannel(page); channel.registerObject('host', host); channel.registerObject('other', second); page.setWebChannel(channel)
view.setHtml('<!doctype html><title>Private session transport fixture</title><script src="qrc:///qtwebchannel/qwebchannel.js"></script><script>new QWebChannel(qt.webChannelTransport,c=>window.testChannel=c);</script>')
try:
    until('Boolean(window.testChannel)')
    js('window.testChannel.objects.host.event.connect(payload=>window.lastEvent=JSON.parse(payload))')
    host.send('session-test', {'literal':'notification retains its channel name'})
    until('window.lastEvent?.kind === "session-test"')
    dataset = measurements(); dataset['records'] = [dict(copy.deepcopy(dataset['records'][i%8]),id='r'+str(i)) for i in range(1600)]
    dataset['usage'] = {'literal': '<script>not HTML</script>'+'λ'*8000}
    ref = h.workspace.put_spec('dataset', dataset); h.workspace.dirty = False
    before = h.workspace.db.serialize(); prior = h.workspace.history.summary()
    draft = request({'action':'scienceDraftOpen','kind':'dataset','reference':ref})
    assert 'document' not in draft and draft['base']==ref and draft['sequence']==0
    token = draft['session']; root = request({'action':'scienceDraftRead','session':token,'sequence':0})
    paths = {entry['key']['text']:[entry['ordinal']] for entry in root['entries']}
    data = request({'action':'scienceDraftRead','session':token,'sequence':0,'path':paths['records'],'offset':500})
    assert len(data['entries'])==25 and data['entries'][0]['ordinal']==500 and data['total']==1600
    scalar = request({'action':'scienceDraftRead','session':token,'sequence':0,'path':paths['usage']+[0]})
    assert scalar['scalar']['text'].startswith('<script>') and scalar['next_offset']==2000
    assert 'Unknown draft session' in request({'action':'scienceDraftRead','session':token,'sequence':0},owner='other',ok=False)
    assert 'sequence' in request({'action':'scienceDraftRead','session':token},ok=False)
    assert 'unsupported fields' in request({'action':'scienceDraftRead','session':token,'sequence':0,'extra':True},ok=False)
    raw = '{"action":"scienceDraftPatch","session":'+json.dumps(token)+',"sequence":0,"operations":[{"op":"put","path":[],"key":"name","value":"first","value":"last"}]}'
    assert 'Duplicate JSON key' in request(raw,ok=False)
    assert h.workspace.db.serialize()==before and h.workspace.history.summary()==prior and not h.workspace.dirty
    changed = request({'action':'scienceDraftPatch','session':token,'sequence':0,
        'operations':[{'op':'put','path':[],'key':'name','value':'Changed via paged owner session'}]})
    assert changed['sequence']==1 and h.workspace.db.serialize()==before
    assert 'sequence changed' in request({'action':'scienceDraftCommit','session':token,'sequence':0},ok=False)
    h.job = SimpleNamespace(isRunning=lambda:False,awaiting_terminal=True,terminal_delivered=False)
    assert 'Stop the current run' in request({'action':'scienceDraftPatch','session':token,'sequence':1,'operations':[]},ok=False)
    request({'action':'scienceDraftRead','session':token,'sequence':1})
    h.job = None
    checked = request({'action':'scienceDraftCheck','session':token,'sequence':1})
    assert h.workspace.db.serialize()==before and h.workspace.history.summary()==prior
    saved = request({'action':'scienceDraftCommit','session':token,'sequence':1})
    assert saved==checked and h.workspace.history.summary()['undo'] and h.workspace.dirty
    assert h.workspace.get_spec('dataset',saved['id'])['records']==dataset['records']
    assert 'Unknown draft session' in request({'action':'scienceDraftDescribe','session':token},ok=False)
    request({'action':'undo'}); assert h.workspace.get_spec('dataset',ref['id'])['revision']==ref['revision']
    assert h.workspace.get_spec('dataset',saved['id'],saved['revision'])['name']==changed['name']
    request({'action':'redo'}); assert h.workspace.get_spec('dataset',ref['id'])['revision']==saved['revision']
    noop = request({'action':'scienceDraftOpen','kind':'dataset','reference':saved})
    prior = h.workspace.history.summary(); snapshot = h.workspace.db.serialize()
    request({'action':'scienceDraftCommit','session':noop['session'],'sequence':0})
    assert h.workspace.db.serialize()==snapshot and h.workspace.history.summary()==prior
    unsaved = request({'action':'scienceDraftOpen','kind':'task'})
    origin = h.workspace; replacement = Workspace(); workspaces.append(replacement)
    replacement.db.execute("UPDATE metadata SET value=? WHERE key='workspace_id'", (unsaved['workspace_id'],))
    h.workspace = replacement; before_replacement = replacement.db.serialize()
    assert 'origin changed' in request({'action':'scienceDraftCheck','session':unsaved['session'],'sequence':0},ok=False)
    request({'action':'scienceDraftRead','session':unsaved['session'],'sequence':0})
    request({'action':'scienceDraftClose','session':unsaved['session']})
    assert replacement.db.serialize()==before_replacement
    h.workspace = origin
    orphan = request({'action':'scienceDraftOpen','kind':'task'},owner='other')
    # Both direct bridge disposal and actual view/tab destruction must release
    # its owned draft. This exercises the repaired event-signal/method collision.
    second.deleteLater(); QCoreApplication.sendPostedEvents(None,QEvent.Type.DeferredDelete); app.processEvents()
    assert orphan['session'] not in h.authoring_sessions._sessions
    second_view.deleteLater(); QCoreApplication.sendPostedEvents(None,QEvent.Type.DeferredDelete); app.processEvents()
    closing_view = QWebEngineView(); closing_bridge = Bridge(h, closing_view)
    # This additional native-only tab lifecycle check needs no dynamic channel
    # registration; transport ownership was already exercised above.
    closing = h.dispatch({'action':'scienceDraftOpen','kind':'task'}, closing_bridge)
    closing_view.deleteLater(); QCoreApplication.sendPostedEvents(None,QEvent.Type.DeferredDelete); app.processEvents()
    assert closing['session'] not in h.authoring_sessions._sessions
    with tempfile.TemporaryDirectory() as folder:
        h.workspace.save(Path(folder)/'session.vtw'); loaded=Workspace.load(Path(folder)/'session.vtw')
        try: assert loaded.get_spec('dataset',saved['id'])['revision']==saved['revision']
        finally: loaded.db.close()
    assert errors==[] and 'torch' not in sys.modules
    print(json.dumps({'qwebchannel_owner_transport':'PASS','large_records':1600,'bounded_pages':'PASS',
        'duplicate_json_atomic':'PASS','exact_sequences':'PASS','same_uuid_origin_fence':'PASS',
        'history_noop_save_reload':'PASS','owner_teardown':'PASS','host_torch':False,'console_errors':errors}))
finally:
    # Drop the externally referenced page before its off-record profile owner.
    page.deleteLater(); QCoreApplication.sendPostedEvents(None,QEvent.Type.DeferredDelete); app.processEvents()
    view.close(); view.deleteLater(); QCoreApplication.sendPostedEvents(None,QEvent.Type.DeferredDelete); app.processEvents()
    for workspace in workspaces:
        if workspace.history.directory: workspace.history.directory.cleanup()
        workspace.db.close()
