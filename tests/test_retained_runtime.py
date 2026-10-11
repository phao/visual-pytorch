"""Exact persisted runtime family/provenance and owned typed authoring; no Torch."""
import copy
import json
import random
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from app.authoring_sessions import Sessions
from app.core import GraphError, Workspace, VERSION
from app.evidence import digest
from app.families import preview, specialize, validate_family
from app.family_records import retain, validate_specialization, validate_specialization_dependencies
from app.family_export import stored_source
from app.revisions import validate_all

ROOT=Path(__file__).resolve().parents[1]
def source():return json.loads((ROOT/'examples/families/spatial-adaptive.json').read_text())


class RetainedRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.w=Workspace();self.addCleanup(self.w.db.close)
        self.addCleanup(lambda:self.w.history.directory.cleanup() if self.w.history.directory else None)

    def test_version_five_resolution_keeps_stable_root_ids_and_exact_conditional_witness(self):
        doc=source();original=copy.deepcopy(doc);r=specialize(doc)
        self.assertEqual(r['version'],3);self.assertEqual(r['contract']['parameters'],179)
        self.assertEqual(r['runtime']['inputs'],{'input':[1,2]})
        self.assertEqual(r['runtime']['specification_sha256'],digest(r['runtime_specification']))
        self.assertEqual(r['runtime_specification']['status'],'conditional')
        self.assertEqual(r['runtime_specification']['output_dependencies'],[[]]);self.assertEqual(doc,original)
        self.assertEqual(specialize(doc,{'C':6,'d':12})['contract']['parameters'],375)

    def test_runtime_axes_are_not_inferred_as_constructor_symbols_or_binding_axes(self):
        doc=source();self.assertEqual(set(specialize(doc)['bindings']),{'C','d','K','g'})
        for values in ({'H':9},{'W':11},{'input': [4,7,9]}):
            with self.assertRaises(GraphError):specialize(doc,values)
        doc['runtime_inputs']={'input':[2,1]};r=specialize(doc)
        self.assertEqual(doc['runtime_inputs'],{'input':[2,1]});self.assertEqual(r['runtime']['inputs'],{'input':[1,2]})

    def test_old_grammars_reject_runtime_field_and_unknown_new_version_is_still_closed(self):
        doc=source()
        for version in (1,2,3,4,6,True,5.):
            value=copy.deepcopy(doc);value['version']=version
            with self.subTest(version=version),self.assertRaises(GraphError):validate_family(value)
        doc.pop('runtime_inputs');r=specialize(doc)
        self.assertIsNone(r['runtime']);self.assertIsNone(r['runtime_specification'])

    def test_malformed_unknown_batch_channel_axes_and_empty_maps_cannot_be_retained(self):
        before=self.w.db.serialize()
        for axes in ({},{'unknown':[1]},{'input':[0]},{'input':[3]},{'input':[True]},{'input':[1,1]},None,[]):
            value=source();value['runtime_inputs']=axes
            with self.subTest(axes=axes),self.assertRaises(GraphError):self.w.put_spec('family',value)
            self.assertEqual(self.w.db.serialize(),before)

    def test_fixed_flatten_family_remains_visible_but_specialization_is_rejected_without_mutation(self):
        doc=source();model=doc['models']['cnn']
        model['graph']['nodes']=[n for n in model['graph']['nodes'] if n['id']!='pool']
        model['graph']['edges']=[{'from':a,'to':b} for a,b in [('input','conv'),('conv','activation'),('activation','flatten'),('flatten','head'),('head','output')]]
        ref=self.w.put_spec('family',doc);before=self.w.db.serialize();p=preview(doc)
        self.assertEqual(p['status'],'incompatible');self.assertIn('head',p['reason']);self.assertIn('AdaptiveAvgPool2d',p['reason'])
        with self.assertRaisesRegex(GraphError,'Linear.*runtime'):retain(self.w,family_ref=ref,materialize=False)
        self.assertEqual(self.w.db.serialize(),before)

    def test_required_constructor_binding_has_no_automatic_tensor_inference(self):
        doc=source();doc['symbols'][0].pop('default');self.w.put_spec('family',doc)
        p=preview(doc);self.assertEqual(p['status'],'conditional');self.assertEqual(p['missing'][0]['id'],'C')
        with self.assertRaises(GraphError):specialize(doc)
        self.assertEqual(specialize(doc,{'C':4})['runtime']['inputs'],{'input':[1,2]})

    def test_atomic_retention_save_reload_exact_historical_contract_and_no_sidecar(self):
        ref=self.w.put_spec('family',source());result=retain(self.w,family_ref=ref,materialize=False)
        record=self.w.get_spec('specialization',result['specialization']['id'],result['specialization']['revision'])
        self.assertEqual(record['version'],4);self.assertEqual(record['runtime'],result['resolved']['runtime'])
        newer=source();newer['runtime_inputs']={'input':[1]};self.w.put_spec('family',newer)
        self.assertNotEqual(self.w.get_spec('family',ref['id'])['revision'],ref['revision'])
        self.assertEqual(self.w.get_spec('family',ref['id'],ref['revision'])['runtime_inputs'],{'input':[1,2]})
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'runtime.vtw';self.w.save(path);before=path.read_bytes();loaded=Workspace.load(path)
            try:
                self.assertEqual(loaded.db.execute('PRAGMA user_version').fetchone()[0],VERSION);validate_all(loaded.db)
                self.assertEqual(loaded.get_spec('specialization',record['id'],record['revision']),record)
            finally:loaded.db.close()
            self.assertEqual(path.read_bytes(),before);self.assertEqual(sorted(p.name for p in Path(td).iterdir()),['runtime.vtw','runtime.vtw.lock'])

    def test_runtime_witness_bindings_schema_version_and_axes_cannot_be_forged(self):
        result=retain(self.w,family=source(),materialize=False)
        record=self.w.get_spec('specialization',result['specialization']['id']);record.pop('revision')
        for key,value in [('specification_sha256','0'*64),('inputs',{'input':[1]}),('version',True)]:
            bad=copy.deepcopy(record);bad['runtime'][key]=value
            with self.subTest(key=key),self.assertRaises(GraphError):self.w.put_spec('specialization',bad)
        bad=copy.deepcopy(record);bad['runtime']=None
        with self.assertRaises(GraphError):self.w.put_spec('specialization',bad)
        for axes in ([2,1],[True],[1,1],[],[0]):
            bad=copy.deepcopy(record);bad['runtime']['inputs']['input']=axes
            with self.assertRaises(GraphError):validate_specialization(bad)
        bad=copy.deepcopy(record);bad['bindings']['C']=6
        with self.assertRaises(GraphError):validate_specialization_dependencies(bad,self.w.get_spec)

    def test_late_registry_failure_rolls_back_all_runtime_records_and_dirty_flag(self):
        before=self.w.db.serialize();dirty=self.w.dirty;original=self.w.put_spec
        def fail(kind,*args,**kwargs):
            value=original(kind,*args,**kwargs)
            if kind=='specialization':raise RuntimeError('late storage failure')
            return value
        with patch.object(self.w,'put_spec',side_effect=fail),self.assertRaisesRegex(RuntimeError,'late'):retain(self.w,family=source(),materialize=False)
        self.assertEqual(self.w.db.serialize(),before);self.assertEqual(self.w.dirty,dirty)

    def test_version_twelve_gate_and_nonmutating_older_container_read(self):
        retain(self.w,family=source(),materialize=False)
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'runtime.vtw';self.w.save(path)
            with closing(sqlite3.connect(path)) as db:db.execute('PRAGMA user_version=11');db.commit()
            before=path.read_bytes()
            with self.assertRaisesRegex(GraphError,'version-12'):Workspace.load(path)
            self.assertEqual(path.read_bytes(),before)
            old=Workspace();self.addCleanup(old.db.close);doc=source();doc['version']=4;doc.pop('runtime_inputs')
            retain(old,family=doc,materialize=False);older=Path(td)/'old.vtw';old.save(older)
            with closing(sqlite3.connect(older)) as db:db.execute('PRAGMA user_version=11');db.commit()
            before=older.read_bytes();loaded=Workspace.load(older)
            try:self.assertEqual(loaded.db.execute('PRAGMA user_version').fetchone()[0],12)
            finally:loaded.db.close()
            self.assertEqual(older.read_bytes(),before)

    def test_owned_root_edit_explicit_upgrade_invalid_text_preview_and_removal(self):
        doc=source();doc['version']=4;doc.pop('runtime_inputs');ref=self.w.put_spec('family',doc)
        s=Sessions();owner=object();self.addCleanup(s.close_owner,owner);h=s.open(self.w,owner,'family',base=ref);before=self.w.db.serialize()
        page=s.family_page(h['session'],owner,0,'templates');d=next(p for p in page['roots'] if p['key']=='runtime_inputs')
        self.assertFalse(d['present']);self.assertIsNone(d['path'])
        ops=[{'op':'property','target':{'kind':'document'},'key':'version','text':'5'},
             {'op':'property','target':{'kind':'document'},'key':'runtime_inputs','text':'{"input":[1,2]}'}]
        h=s.family_edit(self.w,h['session'],owner,0,ops)
        p=s.family_preview(self.w,h['session'],owner,h['sequence']);self.assertEqual(p['parameters'],179)
        self.assertEqual(p['runtime_status'],'conditional-per-call-guards');self.assertEqual(self.w.db.serialize(),before)
        with self.assertRaises(GraphError):s.family_edit(self.w,h['session'],owner,h['sequence'],[dict(ops[0],text='4')])
        bad=dict(ops[1],text='not JSON')
        with self.assertRaises(GraphError):s.family_edit(self.w,h['session'],owner,h['sequence'],[bad])
        h=s.family_edit(self.w,h['session'],owner,h['sequence'],[dict(ops[1],text=None)])
        self.assertEqual(s.family_preview(self.w,h['session'],owner,h['sequence'])['runtime_status'],'fixed-shape')
        saved=s.commit(self.w,h['session'],owner,h['sequence']);self.assertEqual(self.w.get_spec('family',saved['id'])['version'],5)

    def test_owned_runtime_property_wrong_owner_sequence_origin_and_failed_batch_remain_fenced(self):
        ref=self.w.put_spec('family',source());s=Sessions();owner=object();self.addCleanup(s.close_owner,owner)
        h=s.open(self.w,owner,'family',base=ref);op={'op':'property','target':{'kind':'document'},'key':'runtime_inputs','text':'{"input":[1]}'}
        for who,seq in ((object(),0),(owner,1)):
            with self.assertRaises(GraphError):s.family_edit(self.w,h['session'],who,seq,[op])
        before=s.read(h['session'],owner,0)
        with self.assertRaises(GraphError):s.family_edit(self.w,h['session'],owner,0,[op,dict(op,key='id',text='"wrong"')])
        self.assertEqual(s.read(h['session'],owner,0),before)

    def test_undo_redo_materialized_baseline_preserves_immutable_runtime_provenance(self):
        before=self.w.history.capture(self.w.db);r=retain(self.w,family=source());self.w.history.record(before,self.w.db)
        self.assertTrue(self.w.history.restore(self.w));record=self.w.get_spec('specialization',r['specialization']['id'],r['specialization']['revision'])
        self.assertEqual(record['runtime']['inputs'],{'input':[1,2]});self.assertEqual(self.w.models(),{})
        self.assertTrue(self.w.history.restore(self.w,redo=True));self.assertIn(r['resolved']['root'],self.w.models())

    def test_export_and_actual_cli_pin_runtime_policy_without_instantiating_or_mutating_sql(self):
        ref=self.w.put_spec('family',source());before=self.w.db.serialize();rng=random.getstate()
        generated=stored_source(self.w,ref)
        self.assertEqual(generated['provenance']['family_grammar'],5);self.assertEqual(generated['provenance']['runtime_inputs'],{'input':[1,2]})
        self.assertEqual(self.w.db.serialize(),before);self.assertEqual(random.getstate(),rng)
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'workspace.vtw';self.w.save(path);original=path.read_bytes()
            command=[sys.executable,str(ROOT/'vpw_cli.py'),str(path),'preview-stored-family',ref['id'],ref['revision']]
            done=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,check=True)
            r=json.loads(done.stdout)['value'];self.assertEqual(r['resolved']['runtime']['inputs'],{'input':[1,2]})
            out=Path(td)/'family.py';command[-3:]=['export-family',ref['id'],ref['revision']];command+=['--output',str(out)]
            done=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,check=True)
            self.assertEqual(json.loads(done.stdout)['value']['provenance']['runtime_inputs'],{'input':[1,2]});self.assertEqual(path.read_bytes(),original)

    def test_fresh_retained_preview_has_no_torch_numpy_qt_imports_or_random_mutation(self):
        code="""import json,sys,random
from pathlib import Path
from app.core import Workspace
from app.family_records import retain
doc=json.loads(Path('examples/families/spatial-adaptive.json').read_text())
w=Workspace();rng=random.getstate()
try:
 r=retain(w,family=doc,materialize=False)
 assert r['resolved']['contract']['parameters']==179
 assert random.getstate()==rng
 assert not any(k.startswith(('torch','numpy','PyQt')) for k in sys.modules)
 print('PASS')
finally:w.db.close()
"""
        done=subprocess.run([sys.executable,'-c',code],cwd=ROOT,capture_output=True,text=True,check=True)
        self.assertEqual(done.stdout.strip(),'PASS')
