"""Pure temporal family6/specialization5 provenance, vtw15 and owned authoring."""
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

from app import authoring
from app.authoring_sessions import Sessions
from app.core import GraphError, Workspace, VERSION, default_model
from app.evidence import digest
from app.families import specialize, validate_family, preview
from app.family_records import retain, validate_specialization, validate_specialization_dependencies
from app.family_export import stored_source
from app.revisions import validate_all
from test_temporal_shapes import temporal_fixture

ROOT=Path(__file__).resolve().parents[1]


def family(*, cnn=False, dynamic=True, scoped=False):
    models,mid,decl=temporal_fixture(cnn=cnn)
    doc={'version':6,'id':'temporal-family','name':'Temporal transition family','root':mid,
         'models':{k:{p:copy.deepcopy(m[p]) for p in ('id','name','graph')} for k,m in models.items()},
         'symbols':[],'assignments':[],'repeats':[], 'constructors':[],
         'runtime_policy':{'version':1,'kind':'runtime-temporal-v1','inputs':decl} if dynamic else None}
    if scoped:
        doc['symbols']=[{'id':'T0','scope':mid,'name':'Baseline T','default':4,'minimum':1,'maximum':128},
                        {'id':'choice','scope':mid,'name':'Cell activation','type':'enum','values':['ReLU','Tanh'],'default':'Tanh'},
                        {'id':'cell-choice','scope':'transition','name':'Activation','type':'enum','values':['ReLU','Tanh']}]
        doc['assignments']=[{'model':mid,'node':nid,'field':'shape','value':[{'symbol':'T0'},*tail]}
                            for nid,tail in [('sequence',[1,4,5] if cnn else [2]),('mask',[]),('reset',[])]]
        doc['assignments'].append({'model':'transition','node':'tanh','field':'activation','value':{'symbol':'cell-choice'}})
        doc['constructors']=[{'model':mid,'node':'scan','symbol':'cell-choice','value':{'symbol':'choice'}}]
    return doc


class TemporalFamilyTests(unittest.TestCase):
    def setUp(self):
        self.w=Workspace();self.addCleanup(self.w.db.close)
        self.addCleanup(lambda:self.w.history.directory.cleanup() if self.w.history.directory else None)

    def test_closed_grammar_and_allocation_free_resolution_exact_temporal_witness(self):
        doc=family();before=copy.deepcopy(doc);rng=random.getstate();r=specialize(doc)
        self.assertEqual(r['version'],4);self.assertEqual(r['contract']['parameters'],10)
        self.assertEqual(r['runtime'],{'version':2,'kind':'runtime-temporal-v1','inputs':doc['runtime_policy']['inputs'],
                                      'specification_sha256':digest(r['runtime_specification'])})
        self.assertEqual(len(r['models']),2);self.assertEqual(doc,before);self.assertEqual(random.getstate(),rng)
        scan=next(n for n in r['models'][r['root']]['graph']['nodes'] if n['type']=='Recurrent')
        self.assertIn(scan['ref'],r['models']);self.assertNotEqual(scan['ref'],'transition')
        self.assertEqual(scan['states'][0],doc['models'][doc['root']]['graph']['nodes'][2]['states'][0])

    def test_null_policy_is_explicit_fixed_recurrence_not_a_hidden_temporal_domain(self):
        doc=family(dynamic=False);r=specialize(doc)
        self.assertIsNone(r['runtime']);self.assertIsNone(r['runtime_specification'])
        retained=retain(self.w,family=doc,materialize=False);record=self.w.get_spec('specialization',retained['specialization']['id'])
        self.assertEqual(record['version'],5);self.assertIsNone(record['runtime'])
        self.assertEqual(self.w.get_spec('model',record['model']['id'])['version'],3)

    def test_unknown_fields_wrong_versions_missing_acknowledgement_and_spatial_reinterpretation_refused(self):
        for change in ('missing','old-spatial','wrong-kind','extra','bool-version','bad-axis','empty'):
            doc=family()
            if change=='missing':doc.pop('runtime_policy')
            elif change=='old-spatial':doc['runtime_inputs']={'sequence':[1]}
            elif change=='wrong-kind':doc['runtime_policy']['kind']='runtime-spatial-v1'
            elif change=='extra':doc['runtime_policy']['guess']=True
            elif change=='bool-version':doc['runtime_policy']['version']=True
            elif change=='bad-axis':doc['runtime_policy']['inputs']['sequence']=[1]
            else:doc['runtime_policy']['inputs']={}
            with self.subTest(change=change),self.assertRaises(GraphError):validate_family(doc)
        for version in range(1,6):
            doc=family();doc['version']=version;doc.pop('runtime_policy')
            if version<4:doc.pop('constructors')
            with self.subTest(version=version),self.assertRaisesRegex(GraphError,'recurrent family grammar'):validate_family(doc)

    def test_scoped_cell_choices_baseline_time_and_required_child_binding_are_retained(self):
        doc=family(scoped=True);r=specialize(doc,{'T0':9,'choice':'ReLU'})
        self.assertEqual(r['bindings'],{'T0':9,'choice':'ReLU'})
        cell=next(m for m in r['models'].values() if any(n['id']=='mix' for n in m['graph']['nodes']))
        self.assertEqual(next(n for n in cell['graph']['nodes'] if n['id']=='tanh')['activation'],'ReLU')
        self.assertEqual(next(row for row in r['instances'] if row['template']=='transition')['bindings'],{'cell-choice':'ReLU'})
        self.assertEqual(r['contract']['shapes']['scan']['recurrent']['steps'],9)
        self.assertEqual(r['contract']['parameters'],10)
        doc['constructors']=[];doc['symbols']=[s for s in doc['symbols'] if s['id']!='choice'];p=preview(doc)
        self.assertEqual(p['status'],'conditional');self.assertEqual(p['missing'][0]['id'],'cell-choice')
        self.assertTrue(p['missing'][0]['path'].endswith('/scan'))

    def test_shared_cells_keep_one_owner_and_incompatible_scoped_choices_refuse_point(self):
        doc=family(scoped=True);g=doc['models'][doc['root']]['graph'];scan=copy.deepcopy(g['nodes'][2]);scan.update(id='scan2',share_with='scan')
        g['nodes'].extend([scan,{'id':'second','type':'Output'}]);g['edges'].extend(dict(e,to='scan2') for e in list(g['edges']) if e['to']=='scan')
        g['edges'].append({'from':'scan2','from_port':'trajectory','to':'second'})
        doc['constructors'].append(dict(doc['constructors'][0],node='scan2'))
        r=specialize(doc);self.assertEqual(r['contract']['parameters'],10)
        doc['constructors'][-1]['value']='ReLU';p=preview(doc)
        self.assertEqual(p['status'],'incompatible');self.assertIn('same constructor/reference',p['reason'])

    def test_recurrent_is_not_a_unary_finite_repeat_body(self):
        doc=family();doc['repeats']=[{'id':'bad','model':doc['root'],'nodes':['scan'],'count':2,'mode':'tied'}]
        with self.assertRaisesRegex(GraphError,'unary operation'):validate_family(doc)

    def test_upstream_finite_repeat_rewrites_explicit_recurrent_sources(self):
        doc=family();g=doc['models'][doc['root']]['graph'];scan=g['nodes'][2]
        g['nodes'].append({'id':'activate','type':'Activation','activation':'Tanh'})
        scan['sequence_inputs'][0]['source']='activate'
        next(e for e in g['edges'] if e['from']=='sequence' and e['to']=='scan')['to']='activate'
        g['edges'].append({'from':'activate','to':'scan'})
        doc['repeats']=[{'id':'repeat','model':doc['root'],'nodes':['activate'],'count':3,'mode':'independent'}]
        r=specialize(doc);node=next(n for n in r['models'][r['root']]['graph']['nodes'] if n['type']=='Recurrent')
        self.assertEqual(node['sequence_inputs'][0]['source'],'repeat/2/activate')
        self.assertEqual(r['contract']['shapes']['scan']['ports']['trajectory'],[4,2])

    def test_exact_historical_policy_retention_save_reload_and_no_sidecar(self):
        doc=family(scoped=True);ref=self.w.put_spec('family',doc);r=retain(self.w,family_ref=ref,bindings={'T0':7},materialize=False)
        record=self.w.get_spec('specialization',r['specialization']['id']);self.assertEqual(record['version'],5)
        self.assertEqual(record['runtime'],r['resolved']['runtime'])
        newer=family(scoped=True);newer['runtime_policy']=None;self.w.put_spec('family',newer)
        self.assertIsNotNone(self.w.get_spec('family',ref['id'],ref['revision'])['runtime_policy'])
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'temporal.vtw';self.w.save(path);before=path.read_bytes();opened=Workspace.load(path)
            try:validate_all(opened.db);self.assertEqual(opened.get_spec('specialization',record['id']),record)
            finally:opened.db.close()
            self.assertEqual(path.read_bytes(),before);self.assertEqual(sorted(p.name for p in Path(td).iterdir()),['temporal.vtw','temporal.vtw.lock'])

    def test_witness_kind_axes_version_hash_null_and_context_forgery_refused_atomically(self):
        r=retain(self.w,family=family(scoped=True),materialize=False);record=self.w.get_spec('specialization',r['specialization']['id']);record.pop('revision')
        before=self.w.db.serialize();dirty=self.w.dirty
        for key,value in (('version',1),('version',True),('kind','runtime-spatial-v1'),('inputs',{'sequence':[1]}),
                          ('inputs',{'sequence':[True]}),('specification_sha256','0'*64)):
            bad=copy.deepcopy(record);bad['runtime'][key]=value
            with self.subTest(key=key,value=value),self.assertRaises(GraphError):self.w.put_spec('specialization',bad)
            self.assertEqual(self.w.db.serialize(),before);self.assertEqual(self.w.dirty,dirty)
        for field,value in (('runtime',None),('bindings',{'T0':5,'choice':'Tanh'}),('instances',record['instances'][:-1]),('version',4)):
            bad=copy.deepcopy(record);bad[field]=value
            with self.subTest(field=field),self.assertRaises(GraphError):self.w.put_spec('specialization',bad)
            self.assertEqual(self.w.db.serialize(),before)

    def test_late_registry_failure_restores_sql_and_dirty_flag(self):
        before=self.w.db.serialize();dirty=self.w.dirty;original=self.w.put_spec
        def fail(kind,*args,**kwargs):
            result=original(kind,*args,**kwargs)
            if kind=='specialization':raise RuntimeError('late temporal write')
            return result
        with patch.object(self.w,'put_spec',side_effect=fail),self.assertRaisesRegex(RuntimeError,'late temporal'):
            retain(self.w,family=family(),materialize=True)
        self.assertEqual(self.w.db.serialize(),before);self.assertEqual(self.w.dirty,dirty)

    def test_materialization_inserts_complete_recurrent_closure_children_first(self):
        from app.core import validate_model
        order=[];original=self.w.put
        def record(model):
            order.append(model['id']);return original(model)
        with patch.object(self.w,'put',side_effect=record):
            result=retain(self.w,family=family(scoped=True),materialize=True)
        models=self.w.models();root=result['resolved']['root']
        self.assertEqual(set(models),set(result['resolved']['models']))
        scan=next(n for n in models[root]['graph']['nodes'] if n['type']=='Recurrent')
        self.assertLess(order.index(scan['ref']),order.index(root))
        self.assertEqual(validate_model(models,root)['parameters'],10)
        self.assertEqual(self.w.get_spec('specialization',result['specialization']['id'])['version'],5)

    def test_materialization_failure_after_transition_insert_restores_everything(self):
        before=self.w.db.serialize();dirty=self.w.dirty;original=self.w.put;calls=[]
        def fail(model):
            original(model);calls.append(model['id'])
            if len(calls)==2:raise RuntimeError('late root drawing write')
        with patch.object(self.w,'put',side_effect=fail),self.assertRaisesRegex(RuntimeError,'late root drawing'):
            retain(self.w,family=family(),materialize=True)
        self.assertEqual(len(calls),2)
        self.assertEqual(self.w.db.serialize(),before);self.assertEqual(self.w.dirty,dirty)
        self.assertEqual(self.w.models(),{})

    def test_version15_gate_and_genuine_version14_read_do_not_rewrite_disk(self):
        retain(self.w,family=family(),materialize=False)
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'temporal.vtw';self.w.save(path)
            with closing(sqlite3.connect(path)) as db:db.execute('PRAGMA user_version=14');db.commit()
            before=path.read_bytes()
            with self.assertRaisesRegex(GraphError,'version-15'):Workspace.load(path)
            self.assertEqual(path.read_bytes(),before)
            old=Workspace();self.addCleanup(old.db.close);old.put(default_model());legacy=Path(td)/'legacy.vtw';old.save(legacy)
            with closing(sqlite3.connect(legacy)) as db:db.execute('PRAGMA user_version=14');db.commit()
            before=legacy.read_bytes();opened=Workspace.load(legacy)
            try:self.assertEqual(opened.db.execute('PRAGMA user_version').fetchone()[0],VERSION)
            finally:opened.db.close()
            self.assertEqual(legacy.read_bytes(),before)

    def test_owned_drawing_seed_is_fixed_family6_then_explicit_policy_edit_and_retention(self):
        models,mid,_=temporal_fixture()
        for model in models.values():self.w.put(model)
        before=self.w.db.serialize();draft=authoring.begin(self.w,'family',model=mid)
        self.assertEqual(draft['document']['version'],6);self.assertIsNone(draft['document']['runtime_policy'])
        authoring.check(self.w,draft);self.assertEqual(self.w.db.serialize(),before)
        sessions=Sessions();owner=object();self.addCleanup(sessions.close_owner,owner)
        handle=sessions.open(self.w,owner,'family',model=mid)
        page=sessions.family_page(handle['session'],owner,0,'templates')
        root=next(p for p in page['roots'] if p['key']=='runtime_policy');self.assertTrue(root['present']);self.assertEqual(root['text'],'null')
        handle=sessions.family_edit(self.w,handle['session'],owner,0,[{'op':'property','target':{'kind':'document'},'key':'runtime_policy',
                  'text':json.dumps(family()['runtime_policy'])}])
        view=sessions.family_preview(self.w,handle['session'],owner,handle['sequence']);self.assertEqual(view['runtime_status'],'conditional-per-call-guards')
        self.assertEqual(self.w.db.serialize(),before)

    def test_owned_root_changes_are_not_silent_spatial_to_temporal_upgrades(self):
        doc=family();ref=self.w.put_spec('family',doc);sessions=Sessions();owner=object();self.addCleanup(sessions.close_owner,owner)
        h=sessions.open(self.w,owner,'family',base=ref);before=self.w.db.serialize()
        h=sessions.family_edit(self.w,h['session'],owner,0,[{'op':'property','target':{'kind':'document'},'key':'runtime_policy',
                'text':'{"version":1,"kind":"runtime-spatial-v1","inputs":{"sequence":[1]}}'}])
        with self.assertRaisesRegex(GraphError,'runtime-temporal-v1'):
            sessions.family_preview(self.w,h['session'],owner,h['sequence'])
        with self.assertRaises(GraphError):sessions.family_edit(self.w,h['session'],owner,h['sequence'],[{
                'op':'property','target':{'kind':'document'},'key':'runtime_policy','text':'not JSON'}])
        self.assertEqual(self.w.db.serialize(),before)

    def test_actual_cli_preview_and_export_capture_temporal_policy_without_sql_or_torch(self):
        ref=self.w.put_spec('family',family())
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'temporal.vtw';self.w.save(path);before=path.read_bytes()
            command=[sys.executable,str(ROOT/'vpw_cli.py'),str(path),'preview-stored-family',ref['id'],ref['revision']]
            done=subprocess.run(command,cwd=td,capture_output=True,text=True,check=True);self.assertIn('runtime-temporal-v1',done.stdout)
            out=Path(td)/'family.py';command[-3:]=['export-family',ref['id'],ref['revision']];command+=['--output',str(out)]
            subprocess.run(command,cwd=td,capture_output=True,text=True,check=True)
            self.assertIn('runtime-temporal-v1',out.read_text());self.assertEqual(path.read_bytes(),before)

    def test_fresh_pure_recurrence_retention_imports_no_numerical_or_gui_packages(self):
        code='''import random,sys
from app.core import Workspace
from app.family_records import retain
from test_temporal_families import family
w=Workspace();rng=random.getstate()
try:
 r=retain(w,family=family(scoped=True),materialize=False)
 assert r['resolved']['contract']['parameters']==10
 assert r['resolved']['runtime']['kind']=='runtime-temporal-v1'
 assert random.getstate()==rng
 assert not any(k.startswith(('torch','numpy','PyQt')) for k in sys.modules)
 print('PASS')
finally:w.db.close()
'''
        result=subprocess.run([sys.executable,'-c',code],cwd=ROOT,capture_output=True,text=True,check=True)
        self.assertEqual(result.stdout.strip(),'PASS')
