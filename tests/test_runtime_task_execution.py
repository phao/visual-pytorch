"""Actual runtime task fits, worker admission and app-free saved inference."""
import copy
import importlib.util
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F

from app.core import GraphError, Workspace
from app.evidence import digest
from app.experiment_objects import candidate_bundle, selection_at, preview_experiment, execute_candidate
from app.task_execution import (compile_task_model, generate_task_predictor, restore_task_predictor,
                               resolve_run, check_task_checkpoint)
from app.task_training import train_task
from test_runtime_tasks import runtime_experiment

torch.set_num_threads(1)


class RuntimeTaskExecutionTests(unittest.TestCase):
    def make(self,**kw):
        w,e,ref=runtime_experiment(**kw);self.addCleanup(w.db.close)
        self.addCleanup(lambda:w.history.directory.cleanup() if w.history.directory else None)
        return w,e,ref

    def test_runtime_training_matches_independent_functional_adam_and_population_means(self):
        for shape in ((1,3,5),(1,7,2)):
            with self.subTest(shape=shape):
                w,e,_=self.make(shape=shape);b=candidate_bundle(e,selection_at(e,0),w.get_spec)
                b['protocol'].update(shuffle=False,batch_size=4,evaluate_every=1)
                prepared,_=resolve_run(b,constrained=False)
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(7);oracle=compile_task_model(b,constrained=False)
                params=[p.detach().clone().requires_grad_() for p in oracle.parameters()]
                opt=torch.optim.Adam(params,lr=.03,foreach=False,fused=False)
                p=prepared['populations']['train']
                x=torch.tensor(p['inputs']['input'],dtype=torch.float32).reshape(-1,*shape)
                y=torch.tensor(p['targets']['prediction'],dtype=torch.float32).reshape(-1,1)
                def predict(value):
                    return F.linear(F.adaptive_avg_pool2d(torch.tanh(F.conv2d(value,params[0],params[1],padding=1)),(1,1)).flatten(1),params[2],params[3])
                expected=[]
                for _ in range(3):
                    opt.zero_grad();F.mse_loss(predict(x),y).backward();opt.step()
                    expected.append(math.fsum((predict(x)-y).square().detach().flatten().tolist())/y.numel())
                result=train_task(b,constrained=False)
                self.assertEqual(result['status'],'completed',result.get('failure'))
                self.assertEqual(result['identity']['version'],2);self.assertEqual(result['counts']['completed_updates'],3)
                for saved,value in zip(result['checkpoint']['weights'].values(),params):
                    torch.testing.assert_close(torch.tensor(saved),value.detach(),rtol=0,atol=0)
                self.assertEqual([point['update'] for point in result['history']],[1,2,3])
                for point,loss in zip(result['history'],expected):
                    self.assertAlmostEqual(point['metrics']['train']['objective'],loss,places=5)

    def test_saved_predictor_accepts_new_rectangular_resolutions_without_refit_or_parameter_recreation(self):
        w,e,_=self.make();b=candidate_bundle(e,selection_at(e,0),w.get_spec);r=train_task(b,constrained=False)
        predictor=restore_task_predictor(b,r['checkpoint'],constrained=False)
        model=compile_task_model(b,constrained=False);model.load_state_dict({k:torch.tensor(v) for k,v in r['checkpoint']['weights'].items()})
        ids=[id(p) for p in model.parameters()]
        for h,width in ((2,2),(9,3),(1,11)):
            x=torch.linspace(-1,1,2*h*width).reshape(2,1,h,width)
            records=[{'x':value.tolist()} for value in x]
            with torch.no_grad():expected=model(x)
            torch.testing.assert_close(predictor(records)['output'],expected,rtol=0,atol=0)
            self.assertEqual(ids,[id(p) for p in model.parameters()])
        with self.assertRaises(GraphError):predictor([{'x':[[[[1.]]]]}])

    def test_new_resolution_with_fitted_standardization_is_rejected_not_reinterpreted(self):
        w,e,_=self.make();t=w.get_spec('task',e['data_tasks'][0]['task']['id']);t.pop('revision')
        t['inputs'][0]['transforms']=[{'kind':'standardize'}];e.pop('revision');e['data_tasks'][0]['task']=w.put_spec('task',t)
        ref=w.put_spec('experiment',e);e=w.get_spec('experiment',ref['id'],ref['revision'])
        b=candidate_bundle(e,selection_at(e,0),w.get_spec);r=train_task(b,constrained=False)
        predictor=restore_task_predictor(b,r['checkpoint'],constrained=False)
        original={'x':b['dataset']['records'][0]['values']['x']}
        self.assertEqual(tuple(predictor([original])['output'].shape),(1,1))
        with self.assertRaisesRegex(GraphError,'resolution-independent'):predictor([{'x':[[[1.,2.],[3.,4.]]]}])

    def test_runtime_preflight_uses_actual_case_before_tensor_packing_or_layer_execution(self):
        w,e,_=self.make();b=candidate_bundle(e,selection_at(e,0),w.get_spec);r=train_task(b,constrained=False)
        predictor=restore_task_predictor(b,r['checkpoint'],constrained=True)
        with patch('app.runtime_shapes.available_memory',return_value=1),patch('torch.tensor',side_effect=AssertionError('No tensor packing')):
            with self.assertRaisesRegex(GraphError,'Constrained.*runtime'):predictor([{'x':[[[1.]]]}])

    def test_initial_runtime_allocation_charges_actual_shapes_not_unused_large_baseline_activations(self):
        w,e,_=self.make();f=w.get_spec('family',e['models'][0]['family']['id']);f.pop('revision')
        f['assignments'][0]['value']=[{'symbol':'C'},10000,10000]
        f['models']['cnn']['graph']['nodes'][0]['shape']=[1,10000,10000]
        e.pop('revision');e['models'][0]['family']=w.put_spec('family',f);ref=w.put_spec('experiment',e)
        b=candidate_bundle(w.get_spec('experiment',ref['id'],ref['revision']),selection_at(e,0),w.get_spec)
        with patch('app.core.available_memory',return_value=2*1024**3),patch('app.runtime_shapes.available_memory',return_value=2*1024**3):
            model=compile_task_model(b,constrained=True)
            with torch.no_grad():self.assertEqual(tuple(model(torch.ones(1,1,3,5)).shape),(1,1))

    def test_shared_runtime_multi_input_task_keeps_separate_resolutions_and_one_encoder(self):
        from runtime_spatial_fixtures import nested
        models,root=nested(shared=True);w,e,_=self.make(shape=(4,3,5))
        f={'version':5,'id':'shared-task','name':'Shared input task','root':root,'models':models,
           'symbols':[],'assignments':[],'repeats':[],'constructors':[], 'runtime_inputs':{'a':[1,2],'b':[1,2]}}
        e.pop('revision');e['models'][0]['family']=w.put_spec('family',f);pair=e['data_tasks'][0]
        d=w.get_spec('dataset',pair['dataset']['id']);d.pop('revision')
        d['fields'][1]['shape']=[3];d['fields'].append({'id':'right','shape':[4,2,7],'dtype':'float32','axes':['C','H','W']})
        for row in d['records']:
            v=row['values']['y'][0];row['values']['y']=[v,v/2,v/3]
            row['values']['right']=[[[.2]*7 for _ in range(2)] for _ in range(4)]
        pair['dataset']=w.put_spec('dataset',d);t=w.get_spec('task',pair['task']['id']);t.pop('revision')
        t['inputs']=[{'port':'a','field':'x','transforms':[]},{'port':'b','field':'right','transforms':[]}]
        t['heads'][0]['port']='out'
        pair['task']=w.put_spec('task',t);ref=w.put_spec('experiment',e);e=w.get_spec('experiment',ref['id'],ref['revision'])
        b=candidate_bundle(e,selection_at(e,0),w.get_spec);model=compile_task_model(b,constrained=False)
        self.assertEqual(sum(isinstance(m,torch.nn.Conv2d) for m in model.modules()),1)
        r=train_task(b,constrained=False);self.assertEqual(r['status'],'completed',r['failure'])
        predictor=restore_task_predictor(b,r['checkpoint'],constrained=False)
        out=predictor([{'x':torch.ones(4,7,2).tolist(),'right':torch.zeros(4,3,9).tolist()}])
        self.assertEqual(tuple(out['out'].shape),(1,3))

    def test_exact_saved_worker_run_reload_checkpoint_and_fresh_app_free_export(self):
        w,e,ref=self.make();coord=selection_at(e,0)
        done=execute_candidate(w,ref,coord,'runtime-worker','Runtime dataset case',python=sys.executable,timeout=20)
        record=done['record'];attempt=record['attempts'][0]
        self.assertEqual(record['version'],4);self.assertEqual(attempt['status'],'completed',attempt['error'])
        b=candidate_bundle(e,coord,w.get_spec);check_task_checkpoint(b,attempt['result']['checkpoint'],constrained=False)
        self.assertEqual(attempt['result']['identity']['compiler'],'native-runtime-spatial-v1')
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'runtime.vtw';w.save(path);loaded=Workspace.load(path)
            try:self.assertEqual(loaded.get_spec('run',record['id'],record['revision']),record)
            finally:loaded.db.close()
            code=generate_task_predictor(b,attempt['result']['checkpoint'],constrained=False)
            self.assertNotIn('from app',code);self.assertNotIn("'records':",code)
            export=Path(td)/'predictor.py';export.write_text(code)
            rows=[{'x':[[[.25,.5,.75],[1.,1.25,1.5]]]}]
            expected=restore_task_predictor(b,attempt['result']['checkpoint'],constrained=False)(rows)['output'].tolist()
            probe="""import json,runpy,sys,torch
torch.set_num_threads(1)
s=runpy.run_path(sys.argv[1]);rows=json.loads(sys.argv[2]);y=s['predict_records'](rows)['output']
assert not any(n=='app' or n.startswith('app.') for n in sys.modules)
print(json.dumps(y.tolist()))
"""
            check=subprocess.run([sys.executable,'-c',probe,str(export),json.dumps(rows)],cwd=td,capture_output=True,text=True,timeout=20)
            self.assertEqual(check.returncode,0,check.stderr);self.assertEqual(json.loads(check.stdout),expected)

    def test_actual_jupyter_runtime_rerun_two_resolutions_and_portable_architectures(self):
        from app.notebooks import create_bundle,execute_bundle,verify_bundle
        w,e,ref=self.make(pairs=2)
        recipe=w.put_spec('report',{'version':1,'id':'runtime-rerun','name':'Runtime dataset cases',
            'question':'Can one frozen spatial family fit different rectangular resolutions?',
            'mode':'rerun','experiment':ref,'runs':[],'sections':['models','training','evaluation','checkpoints','failures'],
            'analysis':{'kind':'descriptive','inferential_unit':'run'}})
        with tempfile.TemporaryDirectory() as td:
            before=w.db.serialize();export=create_bundle(w,recipe,Path(td)/'bundle')
            self.assertIn('runtime/app/family_export.py',json.loads((Path(export['path'])/'manifest.json').read_text())['files'])
            result=execute_bundle(export['path'],timeout=60)
            self.assertEqual(result['execution']['status'],'completed',result['execution']['error'])
            directory=Path(result['path']);view=json.loads((directory/'report-data.json').read_text())
            self.assertEqual([r['status'] for r in view['runs']],['completed','completed'])
            self.assertEqual(len(list(directory.glob('runtime-architecture-*.py'))),2)
            self.assertEqual(len(list(directory.glob('predictor-*.py'))),2)
            for source in directory.glob('runtime-architecture-*.py'):
                scope={};exec(source.read_text(),scope)
                self.assertEqual(tuple(scope['model'](torch.ones(2,1,9,3)).shape),(2,1))
            verify_bundle(export['path']);self.assertEqual(w.db.serialize(),before)

    def test_actual_cli_runtime_run_persists_exact_case_and_export_uses_saved_identity(self):
        w,e,ref=self.make();root=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'runtime.vtw';w.save(path);coordinates=Path(td)/'coordinates.json'
            coordinates.write_text(json.dumps(selection_at(e,0)))
            command=[sys.executable,str(root/'vpw_cli.py'),str(path),'run-candidate','--experiment',ref['id'],ref['revision'],
                     '--coordinates',str(coordinates),'--run-id','cli-runtime','--name','Runtime CLI','--timeout','20']
            done=subprocess.run(command,capture_output=True,text=True,timeout=30)
            self.assertEqual(done.returncode,0,done.stderr);value=json.loads(done.stdout)['value']
            self.assertTrue(value['persisted']);record=value['record'];self.assertEqual(record['version'],4)
            self.assertEqual(record['attempts'][0]['status'],'completed');self.assertEqual(record['attempts'][0]['result']['identity']['version'],2)
            before=path.read_bytes();export=subprocess.run([sys.executable,str(root/'vpw_cli.py'),str(path),'export-run-trained',record['id'],record['revision']],
                                              capture_output=True,text=True,timeout=20)
            self.assertEqual(export.returncode,0,export.stderr);self.assertIn('ArchitectureFamily',export.stdout);self.assertEqual(path.read_bytes(),before)
