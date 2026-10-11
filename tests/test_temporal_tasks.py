"""Pure explicit temporal experiment selection, dataset cases and frozen identity."""
import copy
import json
import math
import random
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from app.core import Workspace, GraphError, VERSION
from app.evidence import digest
from app.experiment_objects import candidate_bundle, preview_experiment, selection_at
from app.runtime_tasks import input_policy, record_shapes
from app.task_execution import validate_bundle, run_identity, resolve_run, training_protocol
from test_temporal_families import family

ROOT=Path(__file__).resolve().parents[1]


def temporal_experiment(*, times=(3,7), version=5, execution='runtime-temporal-v1', updates=4, unavailable=False):
    w=Workspace();w.set_constrained_mode(False);ref=w.put_spec('family',family(scoped=True));pairs=[]
    for steps in times:
        fields=[{'id':'sequence','dtype':'float32','shape':[steps,2],'axes':['T','D0']},
                {'id':'initial','dtype':'float32','shape':[2]},
                {'id':'lengths','dtype':'float32','shape':[1]},
                *[{'id':k,'dtype':'float32','shape':[steps],'axes':['T']} for k in ('mask','reset')],
                {'id':'target','dtype':'float32','shape':[steps,2],'axes':['T','D0']},
                {'id':'valid','dtype':'bool','shape':[steps,2]},
                {'id':'target_final','dtype':'float32','shape':[2]},
                {'id':'valid_final','dtype':'bool','shape':[]}]
        rows=[]
        for i in range(12):
            frames=[[math.sin((i+1)*(t+1)*.13),math.cos((i+2)*(t+1)*.11)] for t in range(steps)]
            initial=[.1*i/12,-.08*i/12];length=i%(steps+1)
            mask=[not(i%3==0 and t==1) for t in range(steps)];reset=[t==2 and i%2==0 for t in range(steps)]
            state=list(initial);target=[];valid=[]
            for t in range(steps):
                active=t<length and mask[t]
                if active:
                    if reset[t]:state=list(initial)
                    a,b=frames[t];h,k=state
                    state=[math.tanh(.4*a-.2*b+.25*h+.1*k+.1),math.tanh(-.1*a+.5*b-.2*h+.3*k-.1)]
                target.append(list(state) if active else [0.,0.]);valid.append([active and not unavailable]*2)
            rows.append({'id':'r'+str(i),'values':{'sequence':frames,'initial':initial,'lengths':[float(length)],
                'mask':[float(v) for v in mask],'reset':[float(v) for v in reset],'target':target,
                'valid':valid,'target_final':state,'valid_final':length>0 and not unavailable}})
        d={'version':1,'id':'sequence-data-'+str(steps),'name':'Sequence data T='+str(steps),
           'storage':{'version':1,'kind':'inline-records'},'fields':fields,'records':rows,
           'provenance':{'procedure':'independent Python tanh transition teacher'}}
        t={'version':1,'id':'sequence-task-'+str(steps),'name':'Explicit sequence/state objectives',
           'inputs':[{'port':k,'field':k,'transforms':[]} for k in ('sequence','initial','lengths','mask','reset')],
           'heads':[{'id':'trajectory','port':'prediction','field':'target','loss':'mse','weight':1.,'transforms':[],'mask':'valid'},
                    {'id':'final','port':'final','field':'target_final','loss':'mse','weight':.2,'transforms':[],'mask':'valid_final'}]}
        pairs.append({'id':'T'+str(steps),'dataset':w.put_spec('dataset',d),'task':w.put_spec('task',t)})
    choice={'id':'temporal','family':ref,'bindings':{},'axes':[]}
    if execution is not None:choice['execution']=execution
    e={'version':version,'id':'temporal-experiment','name':'Explicit temporal experiment',
       'question':'Do the same cell owners learn at explicitly supplied sequence lengths?',
       'models':[choice],'data_tasks':pairs,'initialization_seeds':[7],
       'splits':[{'id':'split','protocol':{'method':'external','roles':{
           'train':['r'+str(i) for i in range(8)],'validation':['r8','r9'],'test':['r10','r11']},'excluded':[]}}],
       'protocols':[{'id':'adam','protocol':training_protocol(epochs=None,updates=updates,batch_size=8,lr=.04,
                   shuffle=False,evaluate_every=1,history_limit=50)}],
       'selection':{'kind':'crossed','pairs':[],'exclusions':[]}}
    r=w.put_spec('experiment',e)
    return w,w.get_spec('experiment',r['id'],r['revision']),r


class TemporalTaskTests(unittest.TestCase):
    def make(self,**kw):
        w,e,ref=temporal_experiment(**kw);self.addCleanup(w.db.close)
        self.addCleanup(lambda:w.history.directory.cleanup() if w.history.directory else None)
        return w,e,ref

    def test_explicit_dataset_times_preserve_one_baseline_constructor_and_exact_witness(self):
        w,e,_=self.make();before=w.db.serialize();rng=random.getstate()
        rows=preview_experiment(e,w.get_spec)['candidates'];self.assertEqual([r['status'] for r in rows],['compatible']*2)
        bundles=[candidate_bundle(e,r['coordinates'],w.get_spec) for r in rows]
        self.assertEqual([validate_bundle(b)['shapes']['scan']['recurrent']['steps'] for b in bundles],[3,7])
        self.assertEqual(bundles[0]['model'],bundles[1]['model']);self.assertEqual(bundles[0]['runtime'],bundles[1]['runtime'])
        self.assertEqual([b['version'] for b in bundles],[3,3]);self.assertEqual(run_identity(bundles[0])['compiler'],'native-runtime-temporal-v1')
        self.assertNotEqual(run_identity(bundles[0]),run_identity(bundles[1]));self.assertEqual(w.db.serialize(),before)
        self.assertEqual(random.getstate(),rng)

    def test_old_grammars_deny_temporal_and_omitted_policy_never_opts_in(self):
        w,e,_=self.make();before=w.db.serialize()
        for version in (1,2,3,4):
            bad=copy.deepcopy(e);bad.pop('revision');bad['version']=version
            with self.subTest(version=version),self.assertRaises(GraphError):w.put_spec('experiment',bad)
            self.assertEqual(w.db.serialize(),before)
        for version,policy in ((3,None),(4,None),(5,None),(5,'concrete-v1')):
            w,e,_=self.make(version=version,execution=policy)
            self.assertEqual(preview_experiment(e,w.get_spec)['candidates'][0]['status'],'incompatible')

    def test_null_temporal_witness_and_spatial_policy_cannot_execute_recurrence(self):
        for mode in ('null','spatial'):
            w,e,_=self.make();e.pop('revision')
            if mode=='null':
                f=w.get_spec('family',e['models'][0]['family']['id']);f.pop('revision');f['runtime_policy']=None
                e['models'][0]['family']=w.put_spec('family',f)
            else:e['models'][0]['execution']='runtime-spatial-v1'
            ref=w.put_spec('experiment',e);e=w.get_spec('experiment',ref['id'],ref['revision'])
            self.assertEqual(preview_experiment(e,w.get_spec)['candidates'][0]['status'],'incompatible')

    def test_bundle_version_kind_and_witness_tampering_fail_closed(self):
        w,e,_=self.make();b=candidate_bundle(e,selection_at(e,0),w.get_spec)
        for key,value in (('version',2),('version',True),('version',4),('runtime',None)):
            bad=copy.deepcopy(b);bad[key]=value
            with self.subTest(key=key,value=value),self.assertRaises(GraphError):validate_bundle(bad)
        for change in ('kind','axis','hash','extra'):
            bad=copy.deepcopy(b);witness=bad['runtime']['specialization']['runtime']
            if change=='kind':witness['kind']='runtime-spatial-v1'
            elif change=='axis':witness['inputs']['sequence']=[1]
            elif change=='hash':witness['specification_sha256']='0'*64
            else:bad['runtime']['auto']=True
            with self.subTest(change=change),self.assertRaises(GraphError):validate_bundle(bad)

    def test_inference_shape_planning_is_explicit_rank_frame_state_and_common_time(self):
        w,e,_=self.make();b=candidate_bundle(e,selection_at(e,0),w.get_spec)
        spec={'dataset':b['dataset'],'task':b['task'],'runtime_policy':input_policy(b)}
        row=copy.deepcopy(b['dataset']['records'][1]['values'])
        row.update(sequence=[[.2,.3]]*11,mask=[1.]*11,reset=[0.]*11,lengths=[11.])
        actual=record_shapes(spec,[row]);self.assertEqual(actual['sequence'],[11,2]);self.assertEqual(actual['mask'],[11])
        for change in ('frame','rank','empty','mixed'):
            bad=copy.deepcopy(row)
            if change=='frame':bad['sequence']=[[.2]]*11
            elif change=='rank':bad['sequence']=[[[.2,.3]]]*11
            elif change=='empty':bad['sequence']=[]
            rows=[row,bad] if change=='mixed' else [bad]
            if change=='mixed':bad['sequence']=[[.2,.3]]*7
            with self.subTest(change=change),self.assertRaises(GraphError):record_shapes(spec,rows)

    def test_fitted_transforms_use_saved_raw_field_shape_not_dynamic_feature_statistics(self):
        w,e,_=self.make();e.pop('revision');pair=e['data_tasks'][0]
        t=w.get_spec('task',pair['task']['id']);t.pop('revision');t['inputs'][0]['transforms']=[{'kind':'standardize'}]
        pair['task']=w.put_spec('task',t);ref=w.put_spec('experiment',e);e=w.get_spec('experiment',ref['id'])
        b=candidate_bundle(e,selection_at(e,0),w.get_spec);spec={'dataset':b['dataset'],'task':b['task'],'runtime_policy':input_policy(b)}
        row=b['dataset']['records'][1]['values'];self.assertEqual(record_shapes(spec,[row])['sequence'],[3,2])
        bad=copy.deepcopy(row);bad['sequence']=[[1.,2.]]*7
        with self.assertRaises(GraphError):record_shapes(spec,[bad])

    def test_mismatched_time_controls_targets_and_initial_state_remain_incompatible(self):
        for field,shape in (('mask',[4]),('target',[4,2]),('initial',[3])):
            w,e,_=self.make();pair=e['data_tasks'][0];e.pop('revision');d=w.get_spec('dataset',pair['dataset']['id']);d.pop('revision')
            def zeros(s):return [zeros(s[1:]) for _ in range(s[0])] if s else 0.
            next(f for f in d['fields'] if f['id']==field)['shape']=shape
            for row in d['records']:row['values'][field]=zeros(shape)
            pair['dataset']=w.put_spec('dataset',d);r=w.put_spec('experiment',e);e=w.get_spec('experiment',r['id'])
            self.assertEqual(preview_experiment(e,w.get_spec)['candidates'][0]['status'],'incompatible')

    def test_current_case_memory_includes_unroll_and_refuses_before_torch(self):
        w,e,_=self.make();b=candidate_bundle(e,selection_at(e,0),w.get_spec)
        with self.assertRaisesRegex(GraphError,'RAM|constrained'):resolve_run(b,constrained=True,available_bytes=1)
        _,m=resolve_run(b,constrained=False,available_bytes=1)
        self.assertGreater(m['components']['recurrent_unrolls'],0)

    def test_owned_temporal_upgrade_is_explicit_atomic_and_cannot_downgrade(self):
        from app.authoring_sessions import Sessions
        w,e,ref=self.make(version=3,execution=None);s=Sessions();owner=object();self.addCleanup(s.close_owner,owner)
        h=s.open(w,owner,'experiment',base=ref);before=w.db.serialize()
        ops=[{'op':'property','target':{'kind':'document'},'key':'version','text':'5'},
             {'op':'property','target':{'kind':'entry','collection':'models','ordinal':0},'key':'execution','text':'"runtime-temporal-v1"'}]
        h=s.experiment_edit(w,h['session'],owner,0,ops)
        self.assertEqual(s.experiment_preview(w,h['session'],owner,h['sequence'])['candidates'][0]['status'],'compatible')
        self.assertEqual(w.db.serialize(),before)
        with self.assertRaises(GraphError):s.experiment_edit(w,h['session'],owner,h['sequence'],[dict(ops[0],text='4')])
        r=s.commit(w,h['session'],owner,h['sequence']);self.assertEqual(w.get_spec('experiment',r['id'])['version'],5)

    def test_vtw16_gate_exact_reload_and_read_only_cli_preview(self):
        w,e,ref=self.make()
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'temporal.vtw';w.save(path);before=path.read_bytes()
            done=subprocess.run([sys.executable,str(ROOT/'vpw_cli.py'),str(path),'preview-experiment',ref['id'],ref['revision']],capture_output=True,text=True)
            self.assertEqual(done.returncode,0,done.stderr);self.assertEqual(path.read_bytes(),before)
            loaded=Workspace.load(path)
            try:self.assertEqual(loaded.get_spec('experiment',ref['id']),e);self.assertEqual(loaded.db.execute('PRAGMA user_version').fetchone()[0],VERSION)
            finally:loaded.db.close()
            with closing(sqlite3.connect(path)) as db:db.execute('PRAGMA user_version=15');db.commit()
            before=path.read_bytes()
            with self.assertRaisesRegex(GraphError,'version-16'):Workspace.load(path)
            self.assertEqual(path.read_bytes(),before)

    def test_fresh_selection_and_preparation_import_no_torch_numpy_or_qt(self):
        code='''import sys
from test_temporal_tasks import temporal_experiment
from app.experiment_objects import candidate_bundle,selection_at
from app.task_execution import resolve_run
w,e,_=temporal_experiment()
try:
 b=candidate_bundle(e,selection_at(e,0),w.get_spec);resolve_run(b,constrained=False)
 assert not any(n.startswith(('torch','numpy','PyQt')) for n in sys.modules)
 print('PASS')
finally:w.db.close()
'''
        done=subprocess.run([sys.executable,'-c',code],cwd=ROOT,capture_output=True,text=True)
        self.assertEqual(done.returncode,0,done.stderr)
