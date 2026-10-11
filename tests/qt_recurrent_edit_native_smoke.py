"""Actual native dispatcher: in-place reconfiguration, refusal and SQL history."""
import copy
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from PyQt6.QtCore import QCoreApplication, QObject
from app.config import Config
from app.core import Workspace, validate_model
from app.desktop import Window
from app.recurrent_reconfiguration import TOKEN
from test_recurrent_reconfiguration import editable


class Harness(QObject):
    state = Window.state
    describe = Window.describe
    def __init__(self, path):
        super().__init__();self.workspace = Workspace();self.job = None
        self.config = Config(path/'config.json');self.config.save = Mock(side_effect=AssertionError('No config writes'))
        self.torch_python = sys.executable;self.rendering = {};self.events = []
        models, token, _ = editable();self.nid = token['node']
        for doc in models.values():self.workspace.put(doc)
        self.workspace.dirty = False
    def recent_files(self):return []
    def broadcast(self, kind='state', payload=None, **kwargs):self.events.append((kind, payload))
    def prepare(self):
        s = self.state()
        info = Window.dispatch(self, {'action': 'recurrentEditing', 'workspace_id': s['workspace_id'],
          'panel_scope': s['panel_scope'], 'id': 'parent', 'ref': 'transition', 'node': self.nid}, None)
        return {**{k: info[k] for k in (*TOKEN, 'workspace_id', 'panel_scope')},
                'action': 'editRecurrent', 'options_json': json.dumps(info['options'])}


app = QCoreApplication([])
with tempfile.TemporaryDirectory() as tmp:
    h = Harness(Path(tmp));request = h.prepare();raw = h.workspace.db.serialize();history = h.workspace.history.summary()
    for changes in ({'workspace_id': 'wrong'}, {'panel_scope': 'wrong'}, {'node': 'input'}, {'root_model': 'wrong'},
                    {'root_graph': 'wrong'}, {'transition': 'wrong'}, {'extra': 1}, {'options_json': '{'},
                    {'options_json': '{"version":1,"version":1}'}, {'options_json': 'x'*65537}, {'options_json': {}}):
        try:Window.dispatch(h, {**request, **changes}, None)
        except ValueError:pass
        else:raise AssertionError('Stale/malformed interface editing admitted')
        assert h.workspace.db.serialize() == raw and h.workspace.history.summary() == history
    assert Window.dispatch(h, request, None)['node'] == h.nid
    assert h.workspace.db.serialize() == raw and h.workspace.history.summary() == history and not h.workspace.dirty
    options = json.loads(request['options_json']);options['label'] = 'Edited through native dispatcher'
    request['options_json'] = json.dumps(options)
    h.job = SimpleNamespace(isRunning=lambda: True, mid=None)
    try:Window.dispatch(h, request, None)
    except ValueError as error:assert 'Stop' in str(error)
    else:raise AssertionError('Busy edit admitted')
    assert h.workspace.db.serialize() == raw;h.job = None
    before = h.workspace.models();Window.dispatch(h, request, None);after = h.workspace.models()
    assert validate_model(after, 'parent')['parameters'] == 13
    assert len(h.workspace.history.undo_stack) == 1
    assert Window.dispatch(h, {'action': 'undo'}, None);assert h.workspace.models() == before
    assert Window.dispatch(h, {'action': 'redo'}, None);assert h.workspace.models() == after
    try:Window.dispatch(h, request, None)
    except ValueError as error:assert 'changed' in str(error)
    else:raise AssertionError('Edit replay admitted')
    path = Path(tmp)/'saved.vtw';h.workspace.save(path);stale = h.prepare();old = h.workspace
    h.workspace = Workspace.load(path);old.db.close()
    try:Window.dispatch(h, stale, None)
    except ValueError as error:assert 'Workspace changed' in str(error)
    else:raise AssertionError('Same UUID reload admitted stale edit')
    assert h.workspace.models() == after;h.config.save.assert_not_called();h.workspace.db.close()
assert 'torch' not in sys.modules
print(json.dumps({'native_closed_request_and_duplicate_json': 'PASS', 'noop_sql_history_dirty': 'PASS',
 'one_atomic_undo_redo': 'PASS', 'busy_replay_and_same_uuid_reload': 'PASS', 'host_torch': False, 'config_untouched': True}))
