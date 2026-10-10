import re
import tempfile
import unittest
from pathlib import Path
from app.core import Workspace,default_model
from app.history import History
from app.config import Config
from app.themes import THEMES,THEME_NAMES,resolve_theme


class HistoryThemesTests(unittest.TestCase):
    def fixture(self):
        w=Workspace();d=default_model();w.put(d);self.addCleanup(w.db.close)
        self.addCleanup(lambda:w.history.directory.cleanup() if w.history.directory else None)
        return w,d

    def edit(self,w,d):
        before=w.history.capture(w.db);w.put(d);w.history.record(before,w.db)

    def test_deleted_output_and_layout_can_be_undone_redone(self):
        w,d=self.fixture();original=w.models()
        d['graph']['nodes'].pop();d['graph']['edges'].pop();self.edit(w,d)
        edited=w.models();self.assertNotEqual(original,edited)
        self.assertTrue(w.history.restore(w));self.assertEqual(w.models(),original)
        self.assertTrue(w.dirty)
        self.assertTrue(w.history.restore(w,redo=True));self.assertEqual(w.models(),edited)
        d['graph']['nodes'][0]['x']=48;self.edit(w,d)
        self.assertFalse(w.history.summary()['redo'])

    def test_collection_references_mode_data_restore_and_training_evidence_survives(self):
        w,d=self.fixture();w.record_run(d['id'],{'mse':1},{d['id']:d})
        before=w.history.capture(w.db);w.set_constrained_mode(False)
        extra=default_model('Second');w.put(extra);w.history.record(before,w.db)
        w.record_run(d['id'],{'mse':2},w.models())
        w.history.restore(w)
        self.assertTrue(w.constrained_mode);self.assertNotIn(extra['id'],w.models())
        self.assertEqual(w.db.execute('SELECT count(*) FROM runs').fetchone()[0],2)
        self.assertFalse(w.db.execute('PRAGMA foreign_key_check').fetchall())

    def test_save_revision_is_not_rewound_by_undo(self):
        w,d=self.fixture()
        with tempfile.TemporaryDirectory() as td:
            w.save(Path(td)/'edit.vtw');d['name']='Changed';self.edit(w,d);w.save(w.path)
            revision=w.disk_revision;w.history.restore(w)
            self.assertEqual(w.disk_revision,revision);self.assertTrue(w.dirty)
            w.save(w.path)

    def test_history_is_bounded_and_oversized_edits_are_reported(self):
        w,d=self.fixture();w.history=History(max_steps=2)
        for i in range(4):d['name']=str(i);self.edit(w,d)
        self.assertEqual(len(w.history.undo_stack),2)
        w.history.directory.cleanup();w.history=History(max_bytes=1)
        self.edit(w,d)
        self.assertFalse(w.history.summary()['undo']);self.assertIn('budget',w.history.notice)

    def test_theme_selection_is_persisted_and_system_resolves_without_changing_choice(self):
        with tempfile.TemporaryDirectory() as td:
            config=Config(Path(td)/'config.json')
            for theme in THEME_NAMES:
                config.save(theme=theme);self.assertEqual(Config(config.path).values['theme'],theme)
            with self.assertRaises(ValueError):config.save(theme='unknown')
            config.save(spacing=10);self.assertEqual(Config(config.path).values['spacing'],10)
            with self.assertRaises(ValueError):config.save(spacing='tiny')
        self.assertEqual(resolve_theme('system','light'),'light')
        self.assertEqual(resolve_theme('system','dark'),'dark')
        for name,tokens in THEMES.items():
            self.assertEqual(set(tokens),set(THEMES['dark']))
            self.assertTrue(all(re.fullmatch('#[0-9a-f]{6}',v) for v in tokens.values()))


if __name__=='__main__':unittest.main()
