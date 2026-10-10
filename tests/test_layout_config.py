"""Workspace-owned layout; no implicit migration from per-user appearance."""
import copy
import json
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from app.config import Config, DEFAULTS
from app.core import GraphError, Workspace
from app.presentation import LAYOUT_DEFAULTS, validate_layout


class LayoutConfigTests(unittest.TestCase):
    def setUp(self):
        self.w = Workspace.demo(); self.addCleanup(self.w.db.close)
        self.addCleanup(lambda: self.w.history.directory.cleanup() if self.w.history.directory else None)
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def custom(self):
        value = copy.deepcopy(LAYOUT_DEFAULTS)
        value['expanded'].update(direction='TB', layer_gap=123, fit=False)
        return value

    def test_profiles_are_independent_of_other_workspaces_and_global_preferences(self):
        value = self.w.layout_preferences; value['collapsed']['layer_gap'] = 73
        self.assertEqual(self.w.layout_preferences, LAYOUT_DEFAULTS)
        other = Workspace(); self.addCleanup(other.db.close)
        self.w.set_layout_preferences(value)
        self.assertEqual(other.layout_preferences, LAYOUT_DEFAULTS)
        self.assertEqual(LAYOUT_DEFAULTS['collapsed']['layer_gap'], 48)
        self.assertNotIn('layout', DEFAULTS)
        config = Config(self.root / 'config.json'); config.save()
        before = config.path.read_bytes()
        with self.assertRaisesRegex(ValueError, 'workspace'): config.save(layout=value)
        self.assertEqual(config.path.read_bytes(), before)

    def test_save_reload_preserves_profiles_and_scientific_identity(self):
        from app.evidence import identity
        mid = next(iter(self.w.models())); before = identity(self.w.models(), mid)
        self.w.set_layout_preferences(self.custom())
        self.assertEqual(identity(self.w.models(), mid), before)
        path = self.root / 'lab.vtw'; self.w.save(path)
        disk = path.read_bytes(); loaded = Workspace.load(path); self.addCleanup(loaded.db.close)
        self.assertEqual(loaded.layout_preferences, self.custom())
        self.assertFalse(loaded.dirty); self.assertEqual(path.read_bytes(), disk)
        with sqlite3.connect(path) as db:
            self.assertEqual(json.loads(db.execute("SELECT value FROM metadata WHERE key='layout_preferences'").fetchone()[0]), self.custom())

    def test_old_file_uses_defaults_without_writing_or_importing_global_settings(self):
        path = self.root / 'older.vtw'; self.w.save(path)
        with sqlite3.connect(path) as db:
            db.execute("DELETE FROM metadata WHERE key='layout_preferences'")
        old = path.read_bytes()
        config_path = self.root / 'config.json'
        config_path.write_text(json.dumps({'layout': {'unsupported': 'obsolete'}, 'font_size': 18}))
        config = Config(config_path)
        self.assertIsNone(config.error); self.assertEqual(config.values['font_size'], 18)
        loaded = Workspace.load(path); self.addCleanup(loaded.db.close)
        self.assertEqual(loaded.layout_preferences, LAYOUT_DEFAULTS)
        self.assertEqual(path.read_bytes(), old)
        self.assertEqual(json.loads(config_path.read_text())['layout'], {'unsupported': 'obsolete'})

    def test_invalid_profiles_reject_atomically_and_corrupt_metadata_cannot_open(self):
        self.w.set_layout_preferences(self.custom()); before = self.w.layout_preferences
        for mode in ('collapsed', 'expanded'):
            for key, values in {'layer_gap': [0, 7, 1001, True, 1.5, '48', None],
                               'block_gap': [0, 1001, False], 'fit': [1, 'yes', None],
                               'rearrange': [0, 'no'], 'style': ['unknown', None],
                               'direction': ['RL', 'BT', None]}.items():
                for value in values:
                    bad = self.custom(); bad[mode][key] = value
                    with self.subTest(mode=mode, key=key, value=value):
                        with self.assertRaises(ValueError): self.w.set_layout_preferences(bad)
                        self.assertEqual(self.w.layout_preferences, before)
        for version in (0, 2, True, 1.0, '1', None):
            bad = self.custom(); bad['version'] = version
            with self.assertRaises(ValueError): validate_layout(bad)
        for bad in (self.custom() | {'extra': 1}, {'version': 1}, None):
            with self.assertRaises(ValueError): validate_layout(bad)
        bad = self.custom(); bad['expanded']['extra'] = 1
        with self.assertRaises(ValueError): validate_layout(bad)
        path = self.root / 'corrupt.vtw'; self.w.save(path)
        with sqlite3.connect(path) as db:
            db.execute("UPDATE metadata SET value=? WHERE key='layout_preferences'", ('{"version":2}',))
        before_disk = path.read_bytes()
        with self.assertRaisesRegex(GraphError, 'layout_preferences'): Workspace.load(path)
        self.assertEqual(path.read_bytes(), before_disk)

    def test_workspace_history_restores_layout_and_noop_does_not_dirty(self):
        before = self.w.history.capture(self.w.db)
        self.w.set_layout_preferences(self.custom()); self.w.history.record(before, self.w.db)
        self.assertTrue(self.w.history.restore(self.w))
        self.assertEqual(self.w.layout_preferences, LAYOUT_DEFAULTS)
        self.assertTrue(self.w.history.restore(self.w, redo=True))
        self.assertEqual(self.w.layout_preferences, self.custom())
        self.w.dirty = False; changes = self.w.db.total_changes
        self.w.set_layout_preferences(self.custom())
        self.assertFalse(self.w.dirty); self.assertEqual(self.w.db.total_changes, changes)

    def test_cli_layout_edit_and_readonly_inspection_with_atomic_batch_failure(self):
        path = self.root / 'cli.vtw'; self.w.save(path)
        edits = self.root / 'edits.json'
        def cli(*args):
            output = subprocess.run([sys.executable, 'vpw_cli.py', str(path), *args], capture_output=True, text=True, timeout=20)
            return output.returncode, json.loads(output.stdout)
        edits.write_text(json.dumps([{'op': 'layout', 'value': self.custom()}]))
        code, result = cli('apply', str(edits)); self.assertEqual(code, 0, result)
        saved = path.read_bytes(); code, result = cli('inspect')
        self.assertEqual(code, 0); self.assertEqual(result['value']['layout_preferences'], self.custom())
        self.assertEqual(path.read_bytes(), saved)
        bad = self.custom(); bad['expanded']['layer_gap'] = 0
        edits.write_text(json.dumps([{'op': 'layout', 'value': LAYOUT_DEFAULTS}, {'op': 'layout', 'value': bad}]))
        code, result = cli('apply', str(edits)); self.assertNotEqual(code, 0)
        self.assertEqual(path.read_bytes(), saved)
