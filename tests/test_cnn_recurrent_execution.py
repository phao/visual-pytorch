"""Independent complete image→CNN→state scans/gradients and actual typed fits."""
import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from app.core import Workspace
from app.evidence import parameter_schema
from app.experiment_objects import candidate_bundle, execute_candidate, selection_at
from app.task_execution import compile_task_model, resolve_run, restore_task_predictor, generate_task_predictor
from app.task_runtime import forward_ports
from app.task_training import train_task
from app.notebooks import create_bundle, execute_bundle, verify_bundle
from examples.cnn_recurrent_experiment import populate, TEACHER

torch.set_num_threads(1)


def keyed(bundle,values):
    m=bundle['model'];schema=parameter_schema(m['models'],m['root'])
    return {v['id'].split('/')[-2]+'/'+v['id'].split('/')[-1]:values[k] for k,v in schema.items()}


def oracle(frames,initial,lengths,mask,reset,weights,*,activation='Tanh'):
    """Row/time gate scan; 1×1 cross-correlation and pooling via pixel means.

    No application helper, nn.Module, conv2d or recurrent/packed operator is used.
    Frame means are computed only for active rows/steps, including NaN padding.
    """
    anchor=sum(p.flatten()[:0].sum() for p in weights.values())+frames.flatten()[:0].sum()+initial.flatten()[:0].sum()
    trajectories=[];final=[]
    for b in range(frames.shape[0]):
        state=initial[b];steps=[]
        for t in range(frames.shape[1]):
            if t<int(lengths[b]) and bool(mask[b,t]):
                if bool(reset[b,t]):state=initial[b]
                mean=frames[b,t,0].mean()
                features=mean*weights['conv/weight'][:,0,0,0]+weights['conv/bias']
                before=torch.cat((features,state));state=weights['mix/weight']@before+weights['mix/bias']
                state=torch.tanh(state) if activation=='Tanh' else torch.where(state>0,state,torch.zeros_like(state))
                steps.append(state+anchor)
            else:steps.append(frames.new_zeros(initial.shape[-1])+anchor)
        trajectories.append(torch.stack(steps));final.append(state+anchor)
    return {'prediction':torch.stack(trajectories),'final':torch.stack(final)}


class CNNRecurrentExecutionTests(unittest.TestCase):
    def make(self,**kwargs):
        w=Workspace();w.set_constrained_mode(False);self.addCleanup(w.db.close)
        self.addCleanup(lambda:w.history.directory.cleanup() if w.history.directory else None)
        info=populate(w,materialize=False,**kwargs);ref=info['experiment'];e=w.get_spec('experiment',ref['id'],ref['revision'])
        b=candidate_bundle(e,selection_at(e,0),w.get_spec);return w,e,ref,b

    def test_independent_stdlib_teacher_is_real_full_CNN_state_path(self):
        w,e,ref,b=self.make();p,_=resolve_run(b,constrained=False);m=compile_task_model(b,constrained=False)
        schema=parameter_schema(b['model']['models'],b['model']['root']);values={}
        for key,info in schema.items():
            role='/'.join(info['id'].split('/')[-2:]);raw=TEACHER[role.replace('/','_')]
            values[key]=torch.tensor(raw).reshape(info['shape'])
        m.load_state_dict(values);pop=p['populations']['train'];inputs={k:torch.tensor(v) for k,v in pop['inputs'].items()}
        actual=forward_ports(m,inputs,p['contract'])
        torch.testing.assert_close(actual['prediction'],torch.tensor(pop['targets']['trajectory']),rtol=2e-6,atol=8e-8)
        torch.testing.assert_close(actual['final'],torch.tensor(pop['targets']['final']),rtol=2e-6,atol=8e-8)
        self.assertEqual(sum(param.numel() for param in m.parameters()),14)

    def test_current_T_all_outputs_input_initial_and_parameter_gradients_match_independent_scan(self):
        w,e,ref,b=self.make();model=compile_task_model(b,constrained=False)
        for steps in (1,3,9):
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(31);frames=torch.randn(4,steps,1,4,5,requires_grad=True);initial=torch.randn(4,2,requires_grad=True)*.1
            initial=initial.detach().requires_grad_();lengths=torch.tensor([0,steps,1,max(1,steps-1)])
            mask=torch.ones(4,steps);reset=torch.zeros(4,steps)
            if steps>2:mask[1,1]=0.;reset[2,2]=1.;reset[3,2]=1.
            with torch.no_grad():
                for row,length in enumerate(lengths):frames[row,int(length):]=float('nan')
            x=frames.detach().clone().requires_grad_();h=initial.detach().clone().requires_grad_()
            params={k:v.detach().clone().requires_grad_() for k,v in model.state_dict().items()}
            expected=oracle(x,h,lengths,mask,reset,keyed(b,params))
            actual=forward_ports(model,{'frames':frames,'initial':initial,'lengths':lengths.float()[:,None],
                'step_mask':mask,'step_reset':reset},model.contract)
            for k in expected:torch.testing.assert_close(actual[k],expected[k],rtol=3e-6,atol=2e-7)
            loss=lambda r:r['prediction'].square().sum()+.3*r['final'].square().sum()
            aa=torch.autograd.grad(loss(actual),[frames,initial,*model.parameters()])
            bb=torch.autograd.grad(loss(expected),[x,h,*params.values()])
            for a,c in zip(aa,bb):torch.testing.assert_close(a,c,rtol=9e-6,atol=4e-7)

    def test_real_temporal_and_subject_holdout_fits_worker_checkpoints_and_loss_improvement(self):
        for split in ('temporal','group'):
            w,e,ref,b=self.make(split=split,updates=60);local=train_task(b,constrained=False)
            self.assertEqual(local['status'],'completed',local['failure'])
            self.assertLess(local['final_metrics']['train']['objective'],.2*local['history'][0]['metrics']['train']['objective'])
            done=execute_candidate(w,ref,selection_at(e,0),'fit-'+split,'Padded CNN/recurrent fit',python=sys.executable,timeout=30)
            actual=done['record']['attempts'][0];self.assertEqual(actual['status'],'completed',actual['error'])
            self.assertEqual(actual['result']['checkpoint'],local['checkpoint']);self.assertEqual(actual['result']['history'],local['history'])
            self.assertEqual(actual['result']['final_metrics'],local['final_metrics'])
            self.assertGreater(actual['result']['memory_admission']['components']['recurrent_unrolls'],0)

    def test_saved_and_fresh_app_free_predictors_preserve_ragged_controls_and_all_outputs(self):
        w,e,ref,b=self.make(updates=5);result=train_task(b,constrained=False)
        rows=[r['values'] for r in b['dataset']['records'][::13]]
        predictor=restore_task_predictor(b,result['checkpoint'],constrained=False);expected=predictor(rows)
        source=generate_task_predictor(b,result['checkpoint'],constrained=False)
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'predictor.py';path.write_text(source)
            probe='''import json,runpy,sys,torch
torch.set_num_threads(1);n=runpy.run_path(sys.argv[1]);out=n['predict_records'](json.loads(sys.argv[2]))
assert not any(k=='app' or k.startswith('app.') for k in sys.modules)
print(json.dumps({k:v.tolist() for k,v in out.items()}))'''
            done=subprocess.run([sys.executable,'-c',probe,str(path),json.dumps(rows)],cwd=td,capture_output=True,text=True,timeout=25)
            self.assertEqual(done.returncode,0,done.stderr);self.assertEqual(json.loads(done.stdout),{k:v.tolist() for k,v in expected.items()})

    def test_scoped_executable_channel_activation_variants_independent_all_gradients(self):
        from app.family_records import retain, compile_specialization
        from examples.cnn_recurrent_experiment import family
        for width in (1,2,4):
            for activation in ('ReLU','Tanh'):
                w=Workspace();self.addCleanup(w.db.close)
                info=retain(w,family=family(),bindings={'features':width,'activation':activation},materialize=False)
                model=compile_specialization(w,info['specialization'],constrained=False)
                self.assertEqual(sum(p.numel() for p in model.parameters()),4*width+6)
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(37);x=torch.randn(3,7,1,4,5,requires_grad=True);h=torch.randn(3,2,requires_grad=True)
                lengths=torch.tensor([0,7,4]);mask=torch.ones(3,7);mask[1,1]=0;reset=torch.zeros(3,7);reset[2,2]=1
                xx=x.detach().clone().requires_grad_();hh=h.detach().clone().requires_grad_()
                params={k:v.detach().clone().requires_grad_() for k,v in model.state_dict().items()}
                schema=parameter_schema(info['resolved']['models'],info['resolved']['root'])
                weights={'/'.join(v['id'].split('/')[-2:]):params[k] for k,v in schema.items()}
                expected=oracle(xx,hh,lengths,mask,reset,weights,activation=activation)
                actual=forward_ports(model,{'frames':x,'initial':h,'lengths':lengths.float()[:,None],
                                     'step_mask':mask,'step_reset':reset},model.contract)
                with self.subTest(width=width,activation=activation):
                    for key in expected:torch.testing.assert_close(actual[key],expected[key],rtol=3e-6,atol=2e-7)
                    loss=lambda r:r['prediction'].square().sum()+.3*r['final'].square().sum()
                    aa=torch.autograd.grad(loss(actual),[x,h,*model.parameters()])
                    bb=torch.autograd.grad(loss(expected),[xx,hh,*params.values()])
                    for a,b in zip(aa,bb):torch.testing.assert_close(a,b,rtol=8e-6,atol=4e-7)

    def test_real_cli_retained_fit_save_export_uses_the_same_padded_semantics(self):
        w,e,ref,b=self.make(updates=3);expected=train_task(b,constrained=False)
        with tempfile.TemporaryDirectory() as td:
            folder=Path(td);workspace=folder/'cnn-rnn.vtw';w.save(workspace);coordinates=folder/'coordinates.json'
            coordinates.write_text(json.dumps(selection_at(e,0)));root=Path(__file__).resolve().parents[1]
            command=[sys.executable,str(root/'vpw_cli.py'),str(workspace),'run-candidate','--experiment',ref['id'],ref['revision'],
                     '--coordinates',str(coordinates),'--run-id','cli-cnn-rnn','--name','CNN/state CLI proof','--timeout','25']
            done=subprocess.run(command,capture_output=True,text=True,timeout=35)
            self.assertEqual(done.returncode,0,done.stderr);result=json.loads(done.stdout)['value']
            record=result['record'];self.assertTrue(result['persisted'])
            self.assertEqual(record['attempts'][0]['result']['checkpoint'],expected['checkpoint'])
            self.assertEqual(record['attempts'][0]['result']['final_metrics'],expected['final_metrics'])
            before=workspace.read_bytes();export=subprocess.run([sys.executable,str(root/'vpw_cli.py'),str(workspace),
                'export-run-trained',record['id'],record['revision']],capture_output=True,text=True,timeout=20)
            self.assertEqual(export.returncode,0,export.stderr);self.assertEqual(workspace.read_bytes(),before)
            code=json.loads(export.stdout)['value']['code'];scope={};exec(code,scope)
            rows=[r['values'] for r in b['dataset']['records'][::17]]
            native=restore_task_predictor(b,expected['checkpoint'],constrained=False)(rows);actual=scope['predict_records'](rows)
            for key in native:torch.testing.assert_close(native[key],actual[key],rtol=0,atol=0)

    def test_real_saved_and_rerun_kernels_preserve_complete_CNN_state_case(self):
        w,e,ref,b=self.make(updates=3);done=execute_candidate(w,ref,selection_at(e,0),'cnn-kernel','CNN/state evidence',python=sys.executable,timeout=30)
        result=done['record']['attempts'][0]['result'];self.assertEqual(done['record']['attempts'][0]['status'],'completed')
        rows=[r['values'] for r in b['dataset']['records'][::17]];expected=restore_task_predictor(b,result['checkpoint'],constrained=False)(rows)
        with tempfile.TemporaryDirectory() as td:
            views=[]
            for mode in ('result-report','rerun'):
                recipe=w.put_spec('report',{'version':1,'id':mode,'name':'CNN/state '+mode,
                   'question':'Does the complete padded CNN/state experiment reproduce?',
                   'mode':mode,'experiment':ref if mode=='rerun' else None,'runs':[] if mode=='rerun' else [done['reference']],
                   'sections':['models','training','evaluation','checkpoints','failures'],
                   'analysis':{'kind':'descriptive','inferential_unit':'run'}})
                before=w.db.serialize();created=create_bundle(w,recipe,Path(td)/mode);verify_bundle(created['path'])
                executed=execute_bundle(created['path'],timeout=60)
                self.assertEqual(executed['execution']['status'],'completed',executed['execution']['error'])
                directory=Path(executed['path']);view=json.loads((directory/'report-data.json').read_text());views.append(view)
                self.assertEqual(view['runs'][0]['checkpoint_id'],result['checkpoint_id'])
                self.assertEqual(view['notebook_execution']['training_requested'],mode=='rerun');self.assertEqual(w.db.serialize(),before)
                sources=list(directory.glob('predictor-*.py'));self.assertEqual(len(sources),1)
                scope={};exec(sources[0].read_text(),scope);actual=scope['predict_records'](rows)
                for key in expected:torch.testing.assert_close(actual[key],expected[key],rtol=0,atol=0)
            self.assertEqual(views[0]['runs'][0]['metrics'],views[1]['runs'][0]['metrics'])


if __name__=='__main__':unittest.main()
