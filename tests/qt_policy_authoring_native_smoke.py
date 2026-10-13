"""Real Qt dispatcher: guarded owner edits, noops, undo/redo and private reload."""
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from PyQt6.QtCore import QCoreApplication, QObject
from app.core import Workspace
from app.desktop import Window
from app.config import Config
from app.policy_authoring import TOKEN
from policy_runtime_fixture import models as fixture


class Harness(QObject):
    state = Window.state; describe = Window.describe
    def __init__(self, root):
        super().__init__(); self.workspace = Workspace(); self.job = None; self.torch_python = None; self.rendering = {}
        self.config = Config(root / 'config.json'); self.config.save = Mock(side_effect=AssertionError('No config writes'))
        models, mid = fixture(); self.workspace.put(models[mid]); self.workspace.dirty = False
    def recent_files(self): return []
    def broadcast(self, *args, **kwargs): pass
    def prepare(self):
        state = self.state(); context = {k: state[k] for k in ('workspace_id', 'panel_scope')}
        info = Window.dispatch(self, {'action': 'parameterPolicy', 'id': 'lora', 'node': 'base', **context}, None)
        return {'action': 'editParameterPolicy', **context, **{k: info[k] for k in TOKEN},
                'selector': info['owners'][0]['selector'], 'changes_json': '{"trainable":true}'}


app = QCoreApplication([])
with tempfile.TemporaryDirectory() as folder:
    h = Harness(Path(folder)); before = h.workspace.models(); raw = h.workspace.db.serialize(); request = h.prepare()
    assert h.workspace.db.serialize() == raw and not h.workspace.dirty
    for fields in ({'workspace_id': 'wrong'}, {'panel_scope': 'wrong'}, {'architecture': 'wrong'}, {'extra': 1},
                   {'selector': '["A","weight"]'}, {'changes_json': '{"trainable":0}'},
                   {'changes_json': '{"trainable":true,"trainable":false}'}, {'changes_json': 'NaN'}, {'changes_json': 'x' * 65537}):
        try: Window.dispatch(h, {**request, **fields}, None)
        except ValueError: pass
        else: raise AssertionError('Bad/stale request admitted')
        assert h.workspace.db.serialize() == raw and not h.workspace.history.undo_stack
    h.job = SimpleNamespace(isRunning=lambda: True, mid=None)
    try: Window.dispatch(h, request, None)
    except ValueError as error: assert 'Stop' in str(error)
    else: raise AssertionError('Busy mutation admitted')
    h.job = None; result = Window.dispatch(h, request, None)
    assert result['owners'][0]['trainable'] and len(h.workspace.history.undo_stack) == 1
    after = h.workspace.models(); assert Window.dispatch(h, {'action': 'undo'}, None)
    assert h.workspace.models() == before
    assert Window.dispatch(h, {'action': 'redo'}, None) and h.workspace.models() == after
    try: Window.dispatch(h, request, None)
    except ValueError as error: assert 'changed' in str(error)
    else: raise AssertionError('Stale replay admitted')
    noop = h.prepare(); raw = h.workspace.db.serialize(); count = len(h.workspace.history.undo_stack)
    Window.dispatch(h, noop, None)
    assert h.workspace.db.serialize() == raw and len(h.workspace.history.undo_stack) == count
    path = Path(folder) / 'private.vtw'; h.workspace.save(path); stale = h.prepare(); old = h.workspace
    h.workspace = Workspace.load(path); old.db.close(); raw = h.workspace.db.serialize()
    try: Window.dispatch(h, stale, None)
    except ValueError as error: assert 'Workspace changed' in str(error)
    else: raise AssertionError('Same-path reload admitted stale policy token')
    assert h.workspace.db.serialize() == raw; h.config.save.assert_not_called()
    h.workspace.db.close()
    if h.workspace.history.directory: h.workspace.history.directory.cleanup()
assert 'torch' not in sys.modules and 'numpy' not in sys.modules
print(json.dumps({'native_policy_edit': True, 'stale_busy_closed_json_noop_undo_redo_reload': True,
                  'host_torch': False, 'user_config_untouched': True}))
