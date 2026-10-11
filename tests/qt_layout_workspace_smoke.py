"""Native workspace settings dispatch/history/guards; private files, no user state."""
import copy
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from PyQt6.QtCore import QCoreApplication, QObject
from app.config import Config
from app.core import Workspace
from app.desktop import Window
from app.presentation import LAYOUT_DEFAULTS


class Harness(QObject):
    state = Window.state
    def __init__(self, directory):
        super().__init__()
        self.workspace = Workspace.demo(); self.job = None; self.events = []
        self.config = Config(directory / 'preferences.json')
        self.config.values['layout'] = {'obsolete': True}
        self.config.save = Mock(side_effect=AssertionError('Never save global layout'))
        self.torch_python = sys.executable; self.rendering = {}
    def recent_files(self): return []
    def broadcast(self, kind='state', payload=None):
        self.events.append((kind, self.state() if kind == 'state' else payload))
    def dispatch(self, action, **args): return Window.dispatch(self, {'action': action, **args}, None)


app = QCoreApplication([])
with tempfile.TemporaryDirectory() as directory:
    h = Harness(Path(directory)); original = copy.deepcopy(h.workspace.models())
    layout = copy.deepcopy(LAYOUT_DEFAULTS); layout['collapsed']['block_gap'] = 37
    assert h.dispatch('setLayoutPreferences', layout=layout) == layout
    assert h.workspace.models() == original and h.workspace.dirty
    assert h.events[0][1]['layout_preferences'] == layout
    assert 'layout' not in h.state()['display']; assert h.state()['workspace_id']
    assert len(h.workspace.history.undo_stack) == 1
    assert h.dispatch('undo') and h.state()['layout_preferences'] == LAYOUT_DEFAULTS
    assert h.dispatch('redo') and h.state()['layout_preferences'] == layout
    before = len(h.workspace.history.undo_stack)
    h.dispatch('setLayoutPreferences', layout=layout)
    assert len(h.workspace.history.undo_stack) == before
    h.job = SimpleNamespace(isRunning=lambda: True)
    try: h.dispatch('setLayoutPreferences', layout=LAYOUT_DEFAULTS)
    except ValueError as error: assert 'Stop the current run' in str(error)
    else: raise AssertionError('Layout policy admitted during active job')
    assert h.workspace.layout_preferences == layout
    h.job = None; path = Path(directory) / 'transfer.vtw'; h.workspace.save(path)
    loaded = Workspace.load(path); assert loaded.layout_preferences == layout; loaded.db.close()
    h.workspace.history.directory.cleanup(); h.workspace.db.close(); h.workspace = None
    try: h.dispatch('setLayoutPreferences', layout=layout)
    except ValueError as error: assert 'Open a workspace' in str(error)
    else: raise AssertionError('Settings were saved without a workspace')
    h.config.save.assert_not_called()
assert 'torch' not in sys.modules
print(json.dumps({'workspace_layout_dispatch': 'PASS', 'undo_redo': 'PASS', 'save_reload': 'PASS',
                  'global_config_untouched': 'PASS', 'active_job_and_empty_guards': 'PASS', 'host_torch': False}))
