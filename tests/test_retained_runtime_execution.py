"""Persisted v5 family -> checked actual tensors, gradients and optimizer identity."""
import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F
from app.core import Workspace
from app.family_export import stored_source, family_source, write_new_family
from app.family_records import retain
from test_retained_runtime import source
from runtime_spatial_fixtures import nested

torch.set_num_threads(1)

def exported(doc):
    scope={};exec(family_source(doc,constrained=False)['code'],scope);return scope

class RetainedRuntimeExecutionTests(unittest.TestCase):
    def fixed_task_experiment(self):
        from test_family_axes import family_experiment
        w,e=family_experiment();self.addCleanup(w.db.close)
        self.addCleanup(lambda:w.history.directory.cleanup() if w.history.directory else None)
        doc=source()
        for s,value in zip(doc['symbols'],[1,2,1,1]):s['default']=value
        doc['assignments'][0]['value']=[{'symbol':'C'},2,2]
        doc['models']['cnn']['graph']['nodes'][0]['shape']=[1,2,2]
        next(n for n in doc['models']['cnn']['graph']['nodes'] if n['id']=='conv')['groups']=1
        f=w.put_spec('family',doc);e.pop('revision');e['version']=3;e['id']='runtime-fixed-task'
        e['models']=[{'id':'runtime','family':f,'bindings':{},'axes':[]}];e['initialization_seeds']=[7];e['data_tasks']=e['data_tasks'][:1]
        pair=e['data_tasks'][0];d=w.get_spec('dataset',pair['dataset']['id']);d.pop('revision');d['fields'][0]['shape']=[1,2,2]
        d['fields'][0]['axes']=['C','H','W']
        for row in d['records']:
            v=row['values']['x'][0];row['values']['x']=[[[v,v],[v,v]]]
        pair['dataset']=w.put_spec('dataset',d);ref=w.put_spec('experiment',e)
        return w,w.get_spec('experiment',ref['id'],ref['revision']),ref

    def test_actual_fixed_task_worker_preserves_runtime_witness_without_claiming_dynamic_data(self):
        from app.experiment_objects import preview_experiment,execute_candidate,candidate_bundle
        from app.task_execution import restore_task_predictor
        from app.revisions import validate_all
        w,e,ref=self.fixed_task_experiment();row=preview_experiment(e,w.get_spec)['candidates'][0]
        result=execute_candidate(w,ref,row['coordinates'],'runtime-worker','Runtime fixed task',python=sys.executable,timeout=20)
        record=result['record'];attempt=record['attempts'][0];self.assertEqual(attempt['status'],'completed',attempt['error'])
        self.assertEqual(attempt['result']['counts']['completed_updates'],3)
        sr=record['resolved']['references']['specialization'];spec=w.get_spec('specialization',sr['id'],sr['revision'])
        self.assertEqual(spec['version'],4);self.assertEqual(spec['runtime']['inputs'],{'input':[1,2]})
        bundle=candidate_bundle(e,row['coordinates'],w.get_spec);self.assertEqual(bundle['version'],1)
        predictor=restore_task_predictor(bundle,attempt['result']['checkpoint'],constrained=False)
        self.assertEqual(list(predictor([{'x':[[[1.,2.],[3.,4.]]]}])['output'].shape),[1,1])
        validate_all(w.db)

    def test_actual_jupyter_fixed_task_rerun_keeps_exact_runtime_family_provenance(self):
        from app.notebooks import create_bundle,execute_bundle,verify_bundle
        w,e,ref=self.fixed_task_experiment()
        recipe=w.put_spec('report',{'version':1,'id':'runtime-rerun','name':'Runtime provenance','question':'Does fixed-shape fitting preserve the runtime declaration?',
            'mode':'rerun','experiment':ref,'runs':[],'sections':['models','training','evaluation','checkpoints','failures'],
            'analysis':{'kind':'descriptive','inferential_unit':'run'}})
        with tempfile.TemporaryDirectory() as td:
            before=w.db.serialize();export=create_bundle(w,recipe,Path(td)/'bundle');result=execute_bundle(Path(export['path']),timeout=60)
            self.assertEqual(result['execution']['status'],'completed',result['execution']['error'])
            view=json.loads((Path(result['path'])/'report-data.json').read_text());self.assertEqual(len(view['runs']),1)
            self.assertEqual(view['runs'][0]['status'],'completed');verify_bundle(Path(export['path']));self.assertEqual(w.db.serialize(),before)

    def test_saved_family_default_domain_matches_independent_outputs_and_gradients(self):
        w=Workspace();self.addCleanup(w.db.close);ref=w.put_spec('family',source());retain(w,family_ref=ref,materialize=False)
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'case.vtw';w.save(path);loaded=Workspace.load(path)
            try:code=stored_source(loaded,ref)['code']
            finally:loaded.db.close()
        scope={};exec(code,scope);m=scope['ArchitectureFamily'](constrained=False).double()
        self.assertEqual(m.provenance()['runtime_inputs'],{'input':[1,2]});ids=[id(p) for p in m.parameters()]
        for h,width in ((1,1),(3,9),(11,4)):
            x=torch.linspace(-.6,.8,2*4*h*width,dtype=torch.float64).reshape(2,4,h,width).requires_grad_()
            xx=x.detach().clone().requires_grad_();params=[p.detach().clone().requires_grad_() for p in m.parameters()]
            cw,cb,hw,hb=params;expected=F.linear(F.adaptive_avg_pool2d(torch.tanh(F.conv2d(xx,cw,cb,padding=1,groups=2)),(1,1)).flatten(1),hw,hb)
            actual=m(x);torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-12)
            for a,b in zip(torch.autograd.grad(actual.square().sum(),[x,*m.parameters()]),torch.autograd.grad(expected.square().sum(),[xx,*params])):
                torch.testing.assert_close(a,b,rtol=1e-10,atol=1e-11)
            self.assertEqual([id(p) for p in m.parameters()],ids)
            self.assertEqual(m.preview_shapes({'input':[4,h,width]},batch_size=2)['status'],'verified-for-supplied-shapes')

    def test_retained_shared_family_default_domain_reuses_one_encoder_for_different_ports(self):
        models,root=nested(shared=True)
        doc={'version':5,'id':'shared-runtime','name':'Shared runtime','root':root,'models':models,
             'symbols':[],'assignments':[],'repeats':[],'constructors':[], 'runtime_inputs':{'a':[1,2],'b':[1,2]}}
        w=Workspace();self.addCleanup(w.db.close);r=retain(w,family=doc,materialize=False)
        self.assertEqual(w.get_spec('specialization',r['specialization']['id'])['runtime']['inputs'],doc['runtime_inputs'])
        m=exported(doc)['ArchitectureFamily']();self.assertEqual(sum(p.numel() for p in m.parameters()),203)
        self.assertEqual(sum(isinstance(p,torch.nn.Conv2d) for p in m.modules()),1)
        for h,width in ((3,9),(7,2)):
            a=torch.ones(2,4,h,width);b=torch.zeros(2,4,width,h)
            self.assertEqual(list(m({'left':a,'right':b}).shape),[2,3])
        self.assertEqual(m.runtime_specification()['runtime_inputs'],{'a':[1,2],'b':[1,2]})

    def test_invalid_groups_binding_and_actual_ram_refuse_before_layer_execution(self):
        scope=exported(source());ctor=scope['ArchitectureFamily']
        with patch('torch.nn.Conv2d',side_effect=AssertionError('No allocation')):
            with self.assertRaisesRegex(ValueError,'divisible|groups'):ctor(bindings={'g':3})
        m=ctor(constrained=True)
        # Captured names are private; the network forward globals identify its real guard module.
        available=m.network.forward.__func__.__globals__
        with patch.dict(available,{'available_memory':lambda:1}),patch.object(m.network.layer_0,'forward',side_effect=AssertionError('No Conv')):
            with self.assertRaisesRegex(ValueError,'Constrained.*runtime'):m(torch.zeros(2,4,9,11))

    def test_fixed_v5_without_declaration_and_explicit_override_remain_separate(self):
        doc=source();doc.pop('runtime_inputs');ctor=exported(doc)['ArchitectureFamily'];fixed=ctor()
        self.assertIsNone(fixed.runtime_specification())
        with self.assertRaises(ValueError):fixed(torch.zeros(2,4,9,11))
        dynamic=ctor(runtime_inputs={'input':[1,2]});self.assertEqual(list(dynamic(torch.zeros(2,4,9,11)).shape),[2,3])
        self.assertIsNone(dynamic.constructor_specification()['runtime_inputs'])
        self.assertEqual(dynamic.provenance()['runtime_inputs'],{'input':[1,2]})

    def test_fresh_app_free_default_export_optimizer_and_strict_weights_only_transfer(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'family.py';write_new_family(family_source(source(),constrained=False),path)
            code="""import importlib.util,sys,torch
torch.set_num_threads(1)
s=importlib.util.spec_from_file_location('family',sys.argv[1]);mod=importlib.util.module_from_spec(s);s.loader.exec_module(mod)
assert 'app' not in sys.modules
rng=torch.get_rng_state().clone();assert mod.preview_family()['resolved']['runtime']['inputs']=={'input':[1,2]};assert torch.equal(rng,torch.get_rng_state())
m=mod.ArchitectureFamily();ids=[id(p) for p in m.parameters()];o=torch.optim.Adam(m.parameters(),lr=.01)
for h,w in ((3,5),(7,2),(2,9)):
 o.zero_grad();m(torch.ones(2,4,h,w)).square().mean().backward();o.step();assert ids==[id(p) for p in m.parameters()]
n=mod.ArchitectureFamily();n.warm_start(m.semantic_state(),strict=True);x=torch.ones(2,4,4,7);torch.testing.assert_close(m(x),n(x),rtol=0,atol=0)
assert len(o.state)==4 and all(int(v['step'])==3 for v in o.state.values())
assert 'app' not in sys.modules
print('PASS')
"""
            done=subprocess.run([sys.executable,'-c',code,str(path)],cwd=td,capture_output=True,text=True,check=True)
            self.assertEqual(done.stdout.strip(),'PASS')
