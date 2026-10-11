"""Actual v2 task workers, independent gate/Adam oracle and app-free inference."""
import copy
import itertools
import json
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from app.core import GraphError
from app.evidence import parameter_schema
from app.task_execution import (compile_task_model, resolve_run, restore_task_predictor,
                               generate_task_predictor, check_task_checkpoint)
from app.task_training import train_task
from app.task_result import validate_result
from app.transport import run_json_worker
from runtime_v2_task_fixture import bundle, rebuild
from test_lstm_execution import oracle

torch.set_num_threads(1)


def teacher_bundle(**kwargs):
    b=bundle(**kwargs);d=copy.deepcopy(b['dataset'])
    model=b['model'];schema=parameter_schema(model['models'],model['root'])
    weights={key.removeprefix('layer_0.'): .2*torch.sin(torch.arange(torch.tensor(info['shape']).prod().item()).float()+i)
             .reshape(info['shape']) for i,(key,info) in enumerate(schema.items())}
    p,_=resolve_run(b,constrained=False);c=p['contract'];kw=c['shapes']['lstm']['lstm']['kwargs']
    # Rows remain raw rectangular here; padded/fitted cases use the pure fixture.
    rows=[r['values'] for r in d['records']]
    x,h,z=[torch.tensor([r[k] for r in rows]) for k in ('sequence','hidden','cell')]
    with torch.no_grad():
        expected=oracle(x,h,z,weights,lengths=torch.tensor([r['lengths'][0] for r in rows]).long(),
                        layers=kw['num_layers'],bidirectional=kw['bidirectional'],projection=kw['proj_size'])
    for i,r in enumerate(rows):
        for k,v in expected.items():r['target_out_'+k]=v[i].tolist()
    return rebuild(b,dataset=d)


class TemporalV2TaskExecutionTests(unittest.TestCase):
    def test_sixteen_constructors_training_matches_independent_gate_Adam_and_metrics(self):
        for layers,directions,projection,bias in itertools.product((1,2),(False,True),(0,2),(False,True)):
            bindings={'layers':layers,'directions':directions,'projection':projection,'bias':bias}
            b=teacher_bundle(steps=3,updates=4,bindings=bindings);p,_=resolve_run(b,constrained=False)
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(7);seed=compile_task_model(b,constrained=False)
            weights={k:v.detach().clone().requires_grad_() for k,v in seed.layer_0.named_parameters()}
            opt=torch.optim.Adam(weights.values(),lr=.03,foreach=False,fused=False)
            pop=p['populations']['train'];x,h,c=[torch.tensor(pop['inputs'][k]) for k in ('sequence','hidden','cell')]
            lengths=torch.tensor(pop['inputs']['lengths'])[:,0].long();losses=[]
            targets={k:torch.tensor(pop['targets']['out_'+k]) for k in ('sequence','hidden','cell')}
            masks={k:torch.tensor(pop['masks']['out_'+k]) for k in targets}
            def objective():
                out=oracle(x,h,c,weights,lengths=lengths,layers=layers,bidirectional=directions,projection=projection)
                value=0.
                for k in targets:
                    valid=masks[k]
                    if valid.ndim==1:valid=valid.reshape(-1,1,1).expand_as(targets[k])
                    value+=(1. if k=='sequence' else .2)*(((out[k]-targets[k]).square().double()*valid).sum()/valid.sum())
                return value
            for _ in range(4):opt.zero_grad();objective().backward();opt.step();losses.append(float(objective().detach()))
            result=train_task(b,constrained=False)
            with self.subTest(bindings=bindings):
                self.assertEqual(result['status'],'completed',result['failure'])
                self.assertEqual(result['identity']['compiler'],'native-runtime-temporal-v2')
                for k,v in weights.items():
                    torch.testing.assert_close(torch.tensor(result['checkpoint']['weights']['layer_0.'+k]),v.detach(),rtol=5e-6,atol=2e-7)
                for point,value in zip(result['history'],losses):self.assertAlmostEqual(point['metrics']['train']['objective'],value,places=7)
                validate_result(b,{k:v for k,v in result.items() if k!='bundle'})

    def test_actual_worker_matches_every_checkpoint_metric_progress_and_unavailable(self):
        for mode,unavailable in (('rectangular',False),('pad',False),('standardize',False),('rectangular',True)):
            b=bundle(steps=7,mode=mode,unavailable=unavailable);before=copy.deepcopy(b)
            local=train_task(b,constrained=False);points=[]
            worker=run_json_worker([sys.executable,'-u','-m','app.worker'],
                {'kind':'task-v4','bundle':b,'constrained':False},points.append,timeout=30)
            with self.subTest(mode=mode,unavailable=unavailable):
                self.assertEqual(worker['checkpoint'],local['checkpoint']);self.assertEqual(worker['history'],local['history'])
                self.assertEqual(worker['final_metrics'],local['final_metrics']);self.assertEqual(points,worker['history'])
                self.assertEqual(worker['status'],'unavailable' if unavailable else 'completed')
                self.assertEqual(worker['counts']['completed_updates'],0 if unavailable else 3)
                self.assertEqual(b,before)
                validate_result(b,{k:v for k,v in worker.items() if k!='bundle'})

    def test_native_predictor_matches_gate_oracle_at_three_times(self):
        b=teacher_bundle();r=train_task(b,constrained=False);predictor=restore_task_predictor(b,r['checkpoint'],constrained=False)
        weights={k.removeprefix('layer_0.'):torch.tensor(v) for k,v in r['checkpoint']['weights'].items()}
        c=compile_task_model(b,constrained=False).contract;kw=c['shapes']['lstm']['lstm']['kwargs']
        for steps in (1,7,11):
            rows=[copy.deepcopy(b['dataset']['records'][i]['values']) for i in (0,3,5)]
            for i,row in enumerate(rows):row.update(sequence=[[.25,-.1]]*steps,lengths=[float((0,steps,1)[i])])
            expected=oracle(*(torch.tensor([r[k] for r in rows]) for k in ('sequence','hidden','cell')),weights,
                lengths=torch.tensor([r['lengths'][0] for r in rows]).long(),layers=kw['num_layers'],
                bidirectional=kw['bidirectional'],projection=kw['proj_size'])
            actual=predictor(rows)
            for k in expected:torch.testing.assert_close(actual['out_'+k],expected[k],rtol=4e-6,atol=2e-7)

    def test_fresh_app_free_source_matches_variable_time_saved_predictions(self):
        b=bundle();r=train_task(b,constrained=False);predictor=restore_task_predictor(b,r['checkpoint'],constrained=False)
        source=generate_task_predictor(b,r['checkpoint'],constrained=False)
        self.assertNotIn('from app',source);self.assertNotIn("'records':",source)
        self.assertIn('native-runtime-temporal-v2',source)
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'predictor.py';path.write_text(source)
            for steps in (1,9):
                row=copy.deepcopy(b['dataset']['records'][2]['values']);row.update(sequence=[[.2,.3]]*steps,lengths=[float(steps)])
                expected=predictor([row])
                probe='''import json,runpy,sys,torch
torch.set_num_threads(1);s=runpy.run_path(sys.argv[1]);rows=json.loads(sys.argv[2]);out=s['predict_records'](rows)
assert not any(n=='app' or n.startswith('app.') for n in sys.modules)
print(json.dumps({k:v.tolist() for k,v in out.items()}))'''
                done=subprocess.run([sys.executable,'-c',probe,str(path),json.dumps([row])],cwd=td,capture_output=True,text=True,timeout=20)
                self.assertEqual(done.returncode,0,done.stderr);self.assertEqual(json.loads(done.stdout),{k:v.tolist() for k,v in expected.items()})

    def test_padded_and_fitted_exports_reuse_captured_transform_no_new_fit(self):
        for mode in ('pad','standardize'):
            b=bundle(steps=7,mode=mode);r=train_task(b,constrained=False)
            native=restore_task_predictor(b,r['checkpoint'],constrained=False)
            scope={};exec(generate_task_predictor(b,r['checkpoint'],constrained=False),scope)
            rows=[row['values'] for row in b['dataset']['records'][-4:]]
            expected=native(rows);actual=scope['predict_records'](rows)
            for k in expected:torch.testing.assert_close(actual[k],expected[k],rtol=0,atol=0)
            bad=copy.deepcopy(rows);bad[0]['sequence']=[[.1,.2]]*8
            with self.assertRaises(GraphError):native(bad)
            with self.assertRaises(ValueError):scope['predict_records'](bad)

    def test_bad_controls_and_state_axes_fail_without_mutating_weights(self):
        b=bundle();m=compile_task_model(b,constrained=False)
        before={k:v.detach().clone() for k,v in m.state_dict().items()};owners=[id(p) for p in m.parameters()]
        row=copy.deepcopy(b['dataset']['records'][2]['values']);row.update(sequence=[[.2,.3]]*7,lengths=[7.])
        good={g['id']:torch.tensor([row[g['id']]]) for g in m.contract['input_groups']}
        for key,value in (('lengths',torch.tensor([[7.5]])),('lengths',torch.tensor([[8.]])),
                          ('hidden',torch.zeros(1,3,2)),('cell',torch.zeros(1,2,2))):
            with self.subTest(key=key,shape=value.shape),self.assertRaises(ValueError):m({**good,key:value})
        self.assertEqual([id(p) for p in m.parameters()],owners)
        for k,v in m.state_dict().items():torch.testing.assert_close(v,before[k],rtol=0,atol=0)

    def test_memory_refusal_before_model_allocation_and_checkpoint_export(self):
        b=bundle();r=train_task(b,constrained=False)
        with patch('app.task_execution.available_memory',return_value=1),patch('torch.nn.LSTM') as allocation:
            with self.assertRaisesRegex(GraphError,'RAM'):train_task(b)
            with self.assertRaisesRegex(GraphError,'RAM'):restore_task_predictor(b,r['checkpoint'])
            with self.assertRaisesRegex(GraphError,'RAM'):generate_task_predictor(b,r['checkpoint'])
            allocation.assert_not_called()
        with patch('app.runtime_shapes.available_memory',return_value=None),patch('torch.nn.LSTM') as allocation:
            with self.assertRaisesRegex(GraphError,'RAM'):compile_task_model(b)
            allocation.assert_not_called()

    def test_checkpoint_identity_and_lstm_memory_evidence_tampering_refuse(self):
        b=bundle();r=train_task(b,constrained=False)
        bad=copy.deepcopy(r['checkpoint']);bad['identity']['compiler']='native-runtime-temporal-v1'
        with self.assertRaisesRegex(GraphError,'identity'):check_task_checkpoint(b,bad,constrained=False)
        value={k:copy.deepcopy(v) for k,v in r.items() if k!='bundle'}
        value['memory_admission']['components']['lstm_unrolls']=0
        value['memory_admission']['estimated_bytes']=sum(value['memory_admission']['components'].values())
        with self.assertRaisesRegex(GraphError,'RAM'):validate_result(b,value)

    def test_actual_cancelled_task_preserves_checkpoint_and_never_evaluates_test(self):
        b=bundle();cancel=threading.Event();cancel.set();r=train_task(b,cancel=cancel,constrained=False)
        self.assertEqual(r['status'],'cancelled');self.assertEqual(r['counts']['completed_updates'],0)
        self.assertNotIn('test',r['final_metrics']);validate_result(b,{k:v for k,v in r.items() if k!='bundle'})

    def test_worker_kind_mismatch_and_old_bundle_refuse_before_torch_import(self):
        b=bundle()
        probe='''import io,json,sys
from app.worker import main
sys.stdin=io.StringIO(sys.argv[1]);code=main()
assert code==1 and 'torch' not in sys.modules
'''
        for kind,version in (('task-v3',4),('task-v4',3),('task-v4',True)):
            bad=copy.deepcopy(b);bad['version']=version
            done=subprocess.run([sys.executable,'-c',probe,json.dumps({'kind':kind,'bundle':bad,'constrained':False})],
                                capture_output=True,text=True,timeout=10)
            self.assertEqual(done.returncode,0,done.stderr);self.assertIn('exact TaskBundle version',done.stdout)


if __name__=='__main__':unittest.main()
