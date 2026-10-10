import io
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
from contextlib import redirect_stdout
from app.core import Workspace, default_model, GraphError, VERSION
from app.config import Config, DEFAULTS
from app.file_sync import revision, save_lock
import vpw_cli
import install_desktop


class CLIConfigTests(unittest.TestCase):
    def cli(self, *args):
        stream=io.StringIO()
        with redirect_stdout(stream):
            code=vpw_cli.main([str(a) for a in args])
        return code,json.loads(stream.getvalue())

    def test_destination_rewire_batch_roundtrip_and_export(self):
        # The GUI gesture uses existing edge records; no private UI-only file
        # format or CLI command is required. Agents perform an atomic edit batch.
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'rewire.vtw';w=Workspace();d=default_model()
            d['graph']['nodes'].append({'id':'branch','type':'Activation','activation':'Tanh','x':500,'y':200})
            w.put(d);w.save(path);w.db.close()
            old={'from':'linear1','to':'output'}
            edits=[{'op':'disconnect','id':d['id'],'edge':old},
                   {'op':'connect','id':d['id'],'edge':{'from':'linear1','to':'branch'}},
                   {'op':'connect','id':d['id'],'edge':{'from':'branch','to':'output'}}]
            revision_before=self.cli(path,'inspect')[1]['value']['revision']
            with mock.patch('vpw_cli.read_json',return_value=edits):
                code,result=self.cli(path,'--expect-revision',revision_before,'apply','-')
            self.assertEqual(code,0,result)
            stored=self.cli(path,'show',d['id'])[1]['value']
            self.assertNotIn(old,stored['graph']['edges'])
            self.assertIn({'from':'linear1','to':'branch'},stored['graph']['edges'])
            self.assertEqual(self.cli(path,'validate',d['id'])[0],0)
            code,export=self.cli(path,'export',d['id']);self.assertEqual(code,0,export)
            compile(export['value']['code'],'cli-rewired-export','exec')
            self.assertIn('Tanh',export['value']['code'])
            before=path.read_bytes()
            # Trying to replay the stale disconnection is atomic, not a guess.
            with mock.patch('vpw_cli.read_json',return_value=edits):
                self.assertEqual(self.cli(path,'apply','-')[0],1)
            self.assertEqual(path.read_bytes(),before)

    def test_create_inspect_replace_move_and_revision_conflict(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'lab.vtw';self.assertEqual(self.cli(path,'init')[0],0)
            self.assertEqual(self.cli(path,'init')[0],1)
            w=Workspace.load(path);d=default_model('Agent model');d['result']={'mse':0}
            w.put(d);w.save(path);w.db.close()
            before=self.cli(path,'inspect')[1]['value']['revision']
            with unittest.mock.patch('vpw_cli.read_json',return_value=[{'op':'move','id':d['id'],'node':'linear1','x':48,'y':72}]):
                code,result=self.cli(path,'--expect-revision',before,'apply','-')
            self.assertEqual(code,0,result)
            doc=self.cli(path,'show',d['id'])[1]['value'];self.assertEqual(doc['graph']['nodes'][1]['x'],48)
            self.assertEqual(doc['result'],{'mse':0})
            self.assertEqual(self.cli(path,'--expect-revision',before,'inspect')[0],1)

    def test_edit_batch_is_atomic_and_invalid_draft_is_explicit(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'lab.vtw';w=Workspace();d=default_model();w.put(d);w.save(path);w.db.close()
            before=path.read_bytes()
            edits=[{'op':'move','id':d['id'],'node':'input','x':12,'y':13},
                   {'op':'set-node','id':d['id'],'node':'linear1','values':{'features':0}}]
            with unittest.mock.patch('vpw_cli.read_json',return_value=edits):
                self.assertEqual(self.cli(path,'apply','-')[0],1)
            self.assertEqual(path.read_bytes(),before)
            edge=d['graph']['edges'][0]
            with unittest.mock.patch('vpw_cli.read_json',return_value=[{'op':'disconnect','id':d['id'],'edge':edge}]):
                self.assertEqual(self.cli(path,'apply','-')[0],1)
                self.assertEqual(self.cli(path,'apply','-','--allow-invalid')[0],0)
            self.assertEqual(self.cli(path,'validate',d['id'])[0],1)

    def test_concurrent_save_refuses_overwriting_another_revision(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'lab.vtw';w=Workspace();d=default_model();w.put(d);w.save(path)
            other=Workspace.load(path);d['name']='External';other.put(d);other.save(path)
            saved=path.read_bytes()
            with self.assertRaisesRegex(GraphError,'changed on disk'):w.save(path)
            self.assertEqual(path.read_bytes(),saved);w.db.close();other.db.close()

    def test_save_lock_is_nonblocking_and_persistent(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'lab.vtw'
            with save_lock(path),self.assertRaisesRegex(ValueError,'another process'):
                with save_lock(path):pass
            self.assertTrue(Path(str(path)+'.lock').is_file())

    def test_legacy_container_is_upgraded_only_on_save(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'legacy.vtw';w=Workspace();w.put(default_model());w.save(path);w.db.close()
            with sqlite3.connect(path) as db:db.execute('PRAGMA user_version=1')
            before=path.read_bytes();w=Workspace.load(path)
            self.assertEqual(w.db.execute('PRAGMA user_version').fetchone()[0],VERSION)
            self.assertEqual(path.read_bytes(),before);w.save(path);w.db.close()
            with sqlite3.connect(path) as db:self.assertEqual(db.execute('PRAGMA user_version').fetchone()[0],VERSION)

    def test_config_remembers_zoom_and_font_and_preserves_unknown_keys(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'config.json';c=Config(path)
            self.assertEqual(c.values,DEFAULTS)
            c.save(ui_zoom=1.5,font_size=18,memory_limit_percent=15,other_option='future')
            self.assertEqual(Config(path).values['ui_zoom'],1.5)
            self.assertEqual(Config(path).values['other_option'],'future')
            self.assertEqual(Config(path).values['memory_limit_percent'],15)
            with self.assertRaises(ValueError):c.save(ui_zoom=float('nan'))
            with self.assertRaises(ValueError):c.save(font_size=True)
            for percent in (0, -1, 101, True, '10', float('nan')):
                with self.assertRaises(ValueError):c.save(memory_limit_percent=percent)
            c.save(box_spacing=1);self.assertEqual(Config(path).values['box_spacing'],1)
            c.save(box_spacing=100);self.assertEqual(Config(path).values['box_spacing'],100)
            for value in (0,101,True,1.5,'35'):
                with self.assertRaises(ValueError):c.save(box_spacing=value)

    def test_numeric_menu_spacing_is_bounded_and_independent_of_boxes(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'config.json';c=Config(path)
            for level in range(1,11):
                c.save(spacing=level)
                loaded=Config(path)
                self.assertIsNone(loaded.error)
                self.assertEqual(loaded.values['spacing'],level)
                self.assertEqual(loaded.values['box_spacing'],35)
            c.save(box_spacing=1)
            self.assertEqual(Config(path).values['spacing'],10)
            for value in (0,11,True,1.5,'4','compact','spaced',None,[]):
                before=path.read_bytes()
                with self.assertRaises(ValueError):c.save(spacing=value)
                self.assertEqual(path.read_bytes(),before)

    def test_legacy_menu_choices_migrate_in_memory_until_explicit_save(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'config.json'
            for legacy,level in [('compact',4),('spaced',10)]:
                path.write_text(json.dumps({'spacing':legacy,'box_spacing':72,'future_key':'kept'}))
                before=path.read_bytes();c=Config(path)
                self.assertIsNone(c.error)
                self.assertEqual(c.values['spacing'],level)
                self.assertEqual(c.values['box_spacing'],72)
                self.assertEqual(path.read_bytes(),before)
                c.save(ui_zoom=1.5)
                saved=json.loads(path.read_text())
                self.assertEqual(saved['spacing'],level)
                self.assertEqual(saved['future_key'],'kept')

    def test_invalid_config_is_not_silently_overwritten(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'config.json'
            # Test fixture bytes, not a production rewrite helper.
            path.write_text('{ broken')
            c=Config(path);self.assertIsNotNone(c.error)
            with self.assertRaisesRegex(ValueError,'Repair'):c.save(ui_zoom=1.4)
            self.assertEqual(path.read_text(),'{ broken')

    def test_desktop_integration_uses_literal_paths_and_workspace_mime(self):
        with mock.patch('pathlib.Path.read_text',return_value='<svg/>'):
            artifacts=install_desktop.artifacts(Path('/tmp/example'),Path('/path with spaces'),Path('/usr/bin/python3'))
        desktop=artifacts[Path('/tmp/example/applications/tensor-loom.desktop')]
        self.assertIn('Exec="/usr/bin/python3" "/path with spaces/run.py" %f',desktop)
        self.assertIn('MimeType=application/x-visual-torch-workspace;',desktop)
        self.assertIn('Name=Visual PyTorch\n',desktop)
        self.assertIn('StartupWMClass=Visual PyTorch\n',desktop)
        self.assertIn('Icon=tensor-loom\n',desktop)
        self.assertEqual(len(artifacts),3)


if __name__=='__main__':unittest.main()
