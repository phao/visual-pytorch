"""Portable panel presets, atomic edits and presentation/evidence isolation."""
import copy
import json
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from app.core import GraphError,Workspace
from app.evidence import identity
from app.presentation import PANEL_IDS,validate_panel_layout


def preset():
    return {'version':1,'pack':{'x':12.25,'y':-11,'collapsed':False},
            'panels':{key:{'x':-70+i*80,'y':40+i*24,'width':310.5,'height':420,
                           'visible':key=='library-panel','expanded':False} for i,key in enumerate(PANEL_IDS)},
            'order':list(reversed(PANEL_IDS))}


class PanelWorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.w=Workspace.demo();self.addCleanup(self.w.db.close)
        self.addCleanup(lambda:self.w.history.directory.cleanup() if self.w.history.directory else None)
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)

    def test_independent_defaults_and_defensive_copies(self):
        other=Workspace();self.addCleanup(other.db.close)
        self.assertIsNone(self.w.panel_layout);self.assertNotEqual(self.w.panel_scope,other.panel_scope)
        value=preset();self.w.set_panel_layout(value);value['pack']['x']=999
        read=self.w.panel_layout;read['order'].reverse()
        self.assertEqual(self.w.panel_layout,preset());self.assertIsNone(other.panel_layout)

    def test_save_reload_is_readonly_and_preserves_computational_identity(self):
        mid=next(iter(self.w.models()));before=identity(self.w.models(),mid);models=self.w.models()
        self.w.set_panel_layout(preset());self.assertTrue(self.w.dirty)
        self.assertEqual(identity(self.w.models(),mid),before);self.assertEqual(self.w.models(),models)
        path=self.root/'portable.vtw';self.w.save(path);raw=path.read_bytes()
        loaded=Workspace.load(path);self.addCleanup(loaded.db.close)
        self.assertEqual(loaded.panel_layout,preset());self.assertFalse(loaded.dirty)
        self.assertNotEqual(loaded.panel_scope,self.w.panel_scope)
        self.assertEqual(path.read_bytes(),raw)
        self.assertNotIn('panel_scope',dict(loaded.db.execute('SELECT key,value FROM metadata')))

    def test_noop_clear_and_legacy_defaults_without_disk_mutation(self):
        self.w.set_panel_layout(preset());self.w.dirty=False;count=self.w.db.total_changes
        self.w.set_panel_layout(preset());self.assertFalse(self.w.dirty);self.assertEqual(count,self.w.db.total_changes)
        self.w.set_panel_layout(None);self.assertTrue(self.w.dirty);self.assertIsNone(self.w.panel_layout)
        path=self.root/'legacy.vtw';self.w.save(path);raw=path.read_bytes()
        loaded=Workspace.load(path);self.addCleanup(loaded.db.close)
        self.assertIsNone(loaded.panel_layout);self.assertFalse(loaded.dirty);self.assertEqual(path.read_bytes(),raw)

    def test_invalid_coordinate_domains_are_atomic(self):
        self.w.set_panel_layout(preset());before=self.w.db.serialize()
        for field,values in {'x':[-1000001,1000001,float('nan'),float('inf'),True,'3',None,10**1000],
                             'width':[0,-1,100001,False,'100',None],
                             'height':[0,float('-inf')], 'visible':[0,'yes'], 'expanded':[1,None]}.items():
            for value in values:
                bad=preset();bad['panels']['library-panel'][field]=value
                with self.subTest(field=field,value=str(value)[:25]):
                    with self.assertRaises(ValueError):self.w.set_panel_layout(bad)
                    self.assertEqual(self.w.db.serialize(),before)
        valid=preset();valid['panels']['library-panel'].update(x=-1000000,y=1000000,width=1,height=100000)
        self.assertEqual(validate_panel_layout(valid),valid)

    def test_closed_fields_versions_order_and_state_consistency(self):
        values=[{},[],True,'css',dict(preset(),css='body{}')]
        for version in (0,2,True,1.0,'1',None):
            values.append(dict(preset(),version=version))
        for order in ([],list(PANEL_IDS)[:-1],[PANEL_IDS[0]]*5,list(PANEL_IDS)+['unknown'],True):
            values.append(dict(preset(),order=order))
        for mutation in ('missing','unknown','hidden-expanded','collapsed-visible','extra-rect','extra-pack'):
            bad=preset()
            if mutation=='missing':bad['panels'].pop('library-panel')
            elif mutation=='unknown':bad['panels']['unknown']=bad['panels'].pop('library-panel')
            elif mutation=='hidden-expanded':bad['panels']['inspector-panel']['expanded']=True
            elif mutation=='collapsed-visible':bad['pack']['collapsed']=True
            elif mutation=='extra-rect':bad['panels']['library-panel']['zIndex']=42
            elif mutation=='extra-pack':bad['pack']['width']=42
            values.append(bad)
        for bad in values:
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):validate_panel_layout(bad)
        collapsed=preset();collapsed['pack']['collapsed']=True
        for row in collapsed['panels'].values():row['visible']=False
        self.assertEqual(validate_panel_layout(collapsed),collapsed)

    def test_bad_stored_metadata_cannot_open_and_original_bytes_are_retained(self):
        path=self.root/'bad.vtw';self.w.save(path)
        with sqlite3.connect(path) as db:db.execute("INSERT INTO metadata VALUES ('panel_layout','{\"version\":2}')")
        raw=path.read_bytes()
        with self.assertRaisesRegex(GraphError,'panel_layout'):Workspace.load(path)
        self.assertEqual(path.read_bytes(),raw)

    def test_saved_arrangement_survives_computational_undo_and_redo(self):
        mid=next(iter(self.w.models()));old=self.w.models()[mid]
        before=self.w.history.capture(self.w.db);new=copy.deepcopy(old);new['name']='Edited'
        self.w.put(new);self.w.history.record(before,self.w.db)
        self.w.set_panel_layout(preset());steps=len(self.w.history.undo_stack)
        self.assertEqual(steps,1);self.assertTrue(self.w.history.restore(self.w))
        self.assertEqual(self.w.models()[mid]['name'],old['name']);self.assertEqual(self.w.panel_layout,preset())
        self.w.set_panel_layout(None);self.assertTrue(self.w.history.restore(self.w,redo=True))
        self.assertEqual(self.w.models()[mid]['name'],'Edited');self.assertIsNone(self.w.panel_layout)

    def test_cli_inspection_and_atomic_ordered_edit_preserve_failure_bytes(self):
        path=self.root/'cli.vtw';self.w.save(path);edits=self.root/'edit.json'
        def cli(*args):
            result=subprocess.run([sys.executable,'vpw_cli.py',str(path),*args],capture_output=True,text=True,timeout=20)
            return result.returncode,json.loads(result.stdout)
        edits.write_text(json.dumps([{'op':'panel-layout','value':preset()}]))
        code,result=cli('apply',str(edits));self.assertEqual(code,0,result)
        raw=path.read_bytes();code,result=cli('inspect');self.assertEqual(code,0,result)
        self.assertEqual(result['value']['panel_layout'],preset());self.assertEqual(path.read_bytes(),raw)
        bad=preset();bad['pack']['x']=True
        edits.write_text(json.dumps([{'op':'panel-layout','value':None},{'op':'panel-layout','value':bad}]))
        code,result=cli('apply',str(edits));self.assertNotEqual(code,0);self.assertEqual(path.read_bytes(),raw)

    def test_pure_fresh_host_import_needs_neither_torch_nor_qt(self):
        result=subprocess.run([sys.executable,'-c','from app.core import Workspace; import sys; w=Workspace(); assert w.panel_layout is None; assert "torch" not in sys.modules; assert "PyQt6" not in sys.modules'],capture_output=True,text=True,timeout=20)
        self.assertEqual(result.returncode,0,result.stderr)
