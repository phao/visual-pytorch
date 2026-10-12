"""Actual temporal workers, independent optimizer oracle and saved/rerun kernels."""
import copy
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from app.core import GraphError, Workspace
from app.evidence import digest
from app.experiment_objects import candidate_bundle, selection_at, execute_candidate
from app.task_execution import (compile_task_model,resolve_run,generate_task_predictor,
                               restore_task_predictor,check_task_checkpoint)
from app.task_training import train_task
from app.task_result import validate_result
from app.transport import run_json_worker
from test_temporal_tasks import temporal_experiment
from test_recurrent_execution import oracle

torch.set_num_threads(1)


class TemporalTaskExecutionTests(unittest.TestCase):
    def make(self,**kw):
        w,e,ref=temporal_experiment(**kw);self.addCleanup(w.db.close)
        self.addCleanup(lambda:w.history.directory.cleanup() if w.history.directory else None)
        return w,e,ref

    def test_actual_temporal_fit_matches_independent_functional_adam_and_metrics(self):
        w,e,_=self.make(updates=5)
        for index in range(2):
            b=candidate_bundle(e,selection_at(e,index),w.get_spec);prepared,_=resolve_run(b,constrained=False)
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(7);seed=compile_task_model(b,constrained=False)
            params=[p.detach().clone().requires_grad_() for p in seed.parameters()]
            opt=torch.optim.Adam(params,lr=.04,foreach=False,fused=False)
            p=prepared['populations']['train'];x=torch.tensor(p['inputs']['sequence']);h=torch.tensor(p['inputs']['initial'])
            lengths=torch.tensor(p['inputs']['lengths']).flatten().long();mask=torch.tensor(p['inputs']['mask']).bool()
            reset=torch.tensor(p['inputs']['reset']).bool();y=torch.tensor(p['targets']['trajectory']);f=torch.tensor(p['targets']['final'])
            active=((torch.arange(x.shape[1])[None,:]<lengths[:,None])&mask)[:,:,None].expand_as(y)
            final_active=torch.tensor(p['masks']['final']).bool();losses=[]
            def loss():
                pred,last=oracle(x,h,*params,lengths=lengths,mask=mask,reset=reset)
                first=((pred-y).square().double()*active).sum()/active.sum()
                second=((last-f).square().double()*final_active[:,None]).sum()/(2*final_active.sum())
                return first+.2*second
            for _ in range(5):opt.zero_grad();loss().backward();opt.step();losses.append(float(loss().detach()))
            result=train_task(b,constrained=False);self.assertEqual(result['status'],'completed',result['failure'])
            self.assertEqual(result['identity']['version'],3);self.assertEqual(result['identity']['compiler'],'native-runtime-temporal-v1')
            for saved,value in zip(result['checkpoint']['weights'].values(),params):
                # The independent per-row oracle and active-row batched linear
                # kernels can differ by float32 rounding. Native/worker/export
                # checkpoint comparisons below remain bit-exact.
                torch.testing.assert_close(torch.tensor(saved),value.detach(),rtol=1e-6,atol=1e-7)
            for point,value in zip(result['history'],losses):self.assertAlmostEqual(point['metrics']['train']['objective'],value,places=6)
            validate_result(b,{k:v for k,v in result.items() if k!='bundle'})

    def test_local_and_real_worker_checkpoint_history_identity_and_unavailable_are_exact(self):
        for unavailable in (False,True):
            w,e,_=self.make(times=(7,),updates=4,unavailable=unavailable);b=candidate_bundle(e,selection_at(e,0),w.get_spec)
            local=train_task(b,constrained=False);points=[]
            worker=run_json_worker([sys.executable,'-u','-m','app.worker'],{'kind':'task-v3','bundle':b,'constrained':False},points.append,timeout=30)
            self.assertEqual(worker['checkpoint'],local['checkpoint']);self.assertEqual(worker['history'],local['history'])
            self.assertEqual(worker['final_metrics'],local['final_metrics']);self.assertEqual(points,worker['history'])
            self.assertEqual(worker['status'],'unavailable' if unavailable else 'completed')
            self.assertEqual(worker['counts']['completed_updates'],0 if unavailable else 4)
            validate_result(b,{k:v for k,v in worker.items() if k!='bundle'})

    def test_saved_native_predictor_varies_T_matches_controlled_oracle_and_fresh_export(self):
        w,e,_=self.make(times=(3,),updates=4);b=candidate_bundle(e,selection_at(e,0),w.get_spec);r=train_task(b,constrained=False)
        predictor=restore_task_predictor(b,r['checkpoint'],constrained=False)
        weights=[torch.tensor(v) for v in r['checkpoint']['weights'].values()]
        for steps in (1,7,11):
            records=[{'sequence':[[.25,-.1]]*steps,'initial':[.12,-.2],'lengths':[float(steps)],
                      'mask':[1.]*steps,'reset':[float(t==2) for t in range(steps)]}]
            expected,last=oracle(torch.tensor([records[0]['sequence']]),torch.tensor([records[0]['initial']]),*weights,
                                lengths=torch.tensor([steps]),mask=torch.ones(1,steps,dtype=torch.bool),
                                reset=torch.tensor([records[0]['reset']]).bool())
            actual=predictor(records);torch.testing.assert_close(actual['prediction'],expected,rtol=0,atol=0)
            torch.testing.assert_close(actual['final'],last,rtol=0,atol=0)
        code=generate_task_predictor(b,r['checkpoint'],constrained=False)
        self.assertNotIn('from app',code);self.assertNotIn("'records':",code)
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'predictor.py';path.write_text(code)
            command=[sys.executable,'-c','''import json,runpy,sys,torch
torch.set_num_threads(1);s=runpy.run_path(sys.argv[1]);rows=json.loads(sys.argv[2]);result=s['predict_records'](rows)
assert not any(n=='app' or n.startswith('app.') for n in sys.modules)
print(json.dumps({k:v.tolist() for k,v in result.items()}))
''',str(path),json.dumps(records)]
            done=subprocess.run(command,cwd=td,capture_output=True,text=True,timeout=20)
            self.assertEqual(done.returncode,0,done.stderr);self.assertEqual(json.loads(done.stdout),{k:v.tolist() for k,v in actual.items()})

    def test_wrong_runtime_controls_refused_before_execution_and_weights_do_not_change(self):
        w,e,_=self.make(times=(3,));b=candidate_bundle(e,selection_at(e,0),w.get_spec);m=compile_task_model(b,constrained=False)
        before={k:v.detach().clone() for k,v in m.state_dict().items()};owners=[id(p) for p in m.parameters()]
        good={'sequence':torch.zeros(2,7,2),'initial':torch.zeros(2,2),'lengths':torch.full((2,1),7.),
              'mask':torch.ones(2,7),'reset':torch.zeros(2,7)}
        for key,value in (('mask',torch.full((2,7),.5)),('lengths',torch.full((2,1),7.5)),('reset',torch.zeros(2,3))):
            with self.subTest(key=key),self.assertRaises(ValueError):m({**good,key:value})
        self.assertEqual([id(p) for p in m.parameters()],owners)
        for key,value in m.state_dict().items():torch.testing.assert_close(value,before[key],rtol=0,atol=0)

    def test_cnn_transition_dataset_actual_T_and_saved_predictor_use_the_same_owners(self):
        from test_temporal_families import family
        w,e,_=self.make(times=(3,),updates=2);e.pop('revision');e['models'][0]['family']=w.put_spec('family',family(cnn=True,scoped=True))
        pair=e['data_tasks'][0];d=w.get_spec('dataset',pair['dataset']['id']);d.pop('revision')
        d['fields'][0].update(shape=[3,1,4,5],axes=['T','C','H','W'])
        for row in d['records']:
            row['values']['sequence']=[[[[v[0]+i/50+j/100 for j in range(5)] for i in range(4)]] for v in row['values']['sequence']]
        pair['dataset']=w.put_spec('dataset',d);ref=w.put_spec('experiment',e);e=w.get_spec('experiment',ref['id'])
        b=candidate_bundle(e,selection_at(e,0),w.get_spec);r=train_task(b,constrained=False)
        self.assertEqual(r['status'],'completed',r['failure']);m=compile_task_model(b,constrained=False)
        self.assertEqual(sum(isinstance(n,torch.nn.Conv2d) for n in m.modules()),1)
        m.load_state_dict({k:torch.tensor(v) for k,v in r['checkpoint']['weights'].items()})
        rows=[{'sequence':torch.ones(7,1,4,5).tolist(),'initial':[0.,0.],'lengths':[7.],'mask':[1.]*7,'reset':[0.]*7}]
        expected=m({k:torch.tensor([v]) for k,v in rows[0].items()})
        actual=restore_task_predictor(b,r['checkpoint'],constrained=False)(rows)
        for key in expected:torch.testing.assert_close(actual[key],expected[key],rtol=0,atol=0)

    def test_explicit_pad_and_fitted_standardization_lower_without_shape_reinterpretation(self):
        for mode in ('pad','standardize'):
            w,e,_=self.make(times=(7,),updates=2);e.pop('revision');pair=e['data_tasks'][0]
            d=w.get_spec('dataset',pair['dataset']['id']);d.pop('revision');t=w.get_spec('task',pair['task']['id']);t.pop('revision')
            if mode=='pad':
                d['fields'][0]['shape']=[-1,2]
                for row in d['records']:row['values']['sequence']=row['values']['sequence'][:int(row['values']['lengths'][0])]
                t['inputs'][0]['transforms']=[{'kind':'pad','length':7,'value':0.}]
                pair['dataset']=w.put_spec('dataset',d)
            else:t['inputs'][0]['transforms']=[{'kind':'standardize'}]
            pair['task']=w.put_spec('task',t);ref=w.put_spec('experiment',e);e=w.get_spec('experiment',ref['id'])
            b=candidate_bundle(e,selection_at(e,0),w.get_spec);r=train_task(b,constrained=False)
            self.assertEqual(r['status'],'completed',r['failure'])
            rows=[record['values'] for record in b['dataset']['records'][-3:]]
            predictor=restore_task_predictor(b,r['checkpoint'],constrained=False);expected=predictor(rows)
            scope={};exec(generate_task_predictor(b,r['checkpoint'],constrained=False),scope)
            actual=scope['predict_records'](rows)
            for key in expected:torch.testing.assert_close(actual[key],expected[key],rtol=0,atol=0)
            bad=copy.deepcopy(rows);bad[0]['sequence']=[[1.,2.]]*8
            with self.assertRaises(GraphError):predictor(bad)

    def test_saved_run_reload_actual_cli_and_identity_tampering_are_guarded(self):
        w,e,ref=self.make(times=(3,),updates=3);coord=selection_at(e,0)
        done=execute_candidate(w,ref,coord,'temporal-worker','Actual temporal worker',python=sys.executable,timeout=30)
        record=done['record'];self.assertEqual(record['version'],5);attempt=record['attempts'][0]
        self.assertEqual(attempt['status'],'completed',attempt['error'])
        b=candidate_bundle(e,coord,w.get_spec);bad=copy.deepcopy(attempt['result']['checkpoint']);bad['identity']['compiler']='native-runtime-spatial-v1'
        with self.assertRaisesRegex(GraphError,'identity'):check_task_checkpoint(b,bad,constrained=False)
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'temporal.vtw';w.save(path);loaded=Workspace.load(path)
            try:self.assertEqual(loaded.get_spec('run',record['id'],record['revision']),record)
            finally:loaded.db.close()
            root=Path(__file__).resolve().parents[1];coordinates=Path(td)/'coordinates.json';coordinates.write_text(json.dumps(coord))
            command=[sys.executable,str(root/'vpw_cli.py'),str(path),'run-candidate','--experiment',ref['id'],ref['revision'],
                     '--coordinates',str(coordinates),'--run-id','cli-temporal','--name','Temporal CLI','--timeout','25']
            fitted=subprocess.run(command,capture_output=True,text=True,timeout=35)
            self.assertEqual(fitted.returncode,0,fitted.stderr);saved=json.loads(fitted.stdout)['value']
            self.assertTrue(saved['persisted']);self.assertEqual(saved['record']['version'],5)
            self.assertEqual(saved['record']['attempts'][0]['result']['checkpoint'],attempt['result']['checkpoint'])
            before=path.read_bytes()
            export=subprocess.run([sys.executable,str(root/'vpw_cli.py'),str(path),'export-run-trained',record['id'],record['revision']],capture_output=True,text=True,timeout=20)
            self.assertEqual(export.returncode,0,export.stderr);self.assertIn('runtime-temporal-v1',export.stdout);self.assertEqual(path.read_bytes(),before)

    def test_actual_jupyter_saved_and_rerun_match_exact_temporal_checkpoints_and_metrics(self):
        from app.notebooks import create_bundle,execute_bundle,verify_bundle
        w,e,ref=self.make(times=(3,),updates=3);done=execute_candidate(w,ref,selection_at(e,0),'kernel-run','Temporal evidence',python=sys.executable,timeout=30)
        attempt=done['record']['attempts'][0];self.assertEqual(attempt['status'],'completed',attempt['error'])
        with tempfile.TemporaryDirectory() as td:
            views=[]
            for mode in ('result-report','rerun'):
                recipe=w.put_spec('report',{'version':1,'id':mode,'name':'Temporal report','question':'Does exact temporal execution reproduce?',
                    'mode':mode,'experiment':ref if mode=='rerun' else None,'runs':[] if mode=='rerun' else [done['reference']],
                    'sections':['models','training','evaluation','checkpoints','failures'],
                    'analysis':{'kind':'descriptive','inferential_unit':'run'}})
                before=w.db.serialize();export=create_bundle(w,recipe,Path(td)/mode);verify_bundle(export['path'])
                result=execute_bundle(export['path'],timeout=60)
                self.assertEqual(result['execution']['status'],'completed',result['execution']['error'])
                directory=Path(result['path']);view=json.loads((directory/'report-data.json').read_text());views.append(view)
                self.assertEqual(view['runs'][0]['checkpoint_id'],attempt['result']['checkpoint_id'])
                self.assertEqual(view['notebook_execution']['training_requested'],mode=='rerun')
                self.assertEqual(w.db.serialize(),before)
                for source in directory.glob('runtime-architecture-*.py'):
                    scope={};exec(source.read_text(),scope)
                    result=scope['model']({'sequence':torch.zeros(1,9,2),'initial':torch.zeros(1,2),'lengths':torch.tensor([[9.]]),
                                          'mask':torch.ones(1,9),'reset':torch.zeros(1,9)})
                    self.assertEqual(result['prediction'].shape,(1,9,2))
            self.assertEqual(views[0]['runs'][0]['metrics'],views[1]['runs'][0]['metrics'])
