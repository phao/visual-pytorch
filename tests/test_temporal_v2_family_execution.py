"""Retained native/checked Family8, independent runtime-T gates and gradients."""
import copy
import itertools
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from app.core import GraphError, Workspace
from app.families import specialize
from app.family_export import family_source
from app.family_records import compile_specialization, retain
from examples.runtime_lstm_family import definition
from examples.lstm_family import definition as fixed_definition
from test_lstm_execution import data, named, oracle, returned

torch.set_num_threads(1)


class TemporalV2FamilyExecutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source=family_source(definition(),constrained=False)
        cls.namespace={};exec(cls.source['code'],cls.namespace)
        cls.Family=cls.namespace['ArchitectureFamily']

    def setUp(self):
        self.w=Workspace();self.addCleanup(self.w.db.close)

    def actual(self,model,t,batch=4):
        shapes={g['id']:copy.deepcopy(g['shape']) for g in model.contract['input_groups']}
        shapes['sequence'][0]=t
        return model.preview_shapes(shapes,batch_size=batch)['contract']

    def check_oracle(self,model,t,*,empty=False):
        c=self.actual(model,t);values=data(c,grad=True,nan_padding=True,lengths=[0]*4 if empty else None)
        network=model.network if isinstance(model,self.Family) else model
        keys=('sequence','hidden','cell');kw=c['shapes']['lstm']['lstm']['kwargs']
        inputs={k:values[k].detach().clone().requires_grad_() for k in keys}
        weights={k:p.detach().clone().requires_grad_() for k,p in network.layer_0.named_parameters()}
        expected=oracle(*(inputs[k] for k in keys),weights,lengths=values['lengths'][:,0].long(),
                        layers=kw['num_layers'],bidirectional=kw['bidirectional'],projection=kw['proj_size'])
        actual=returned(c,model(named(c,values)))
        for k in keys:torch.testing.assert_close(actual[k],expected[k],rtol=4e-6,atol=2e-6)
        loss=lambda r:sum((i+.5)*r[k].square().sum() for i,k in enumerate(keys))
        aa=torch.autograd.grad(loss(actual),[*(values[k] for k in keys),*network.layer_0.parameters()])
        bb=torch.autograd.grad(loss(expected),[*(inputs[k] for k in keys),*weights.values()])
        for a,b in zip(aa,bb):torch.testing.assert_close(a,b,rtol=9e-6,atol=4e-6)

    def test_sixteen_constructors_native_and_checked_at_three_times_all_gradients(self):
        for layers,directions,projection,bias in itertools.product((1,2),(False,True),(0,2),(False,True)):
            bindings={'time':4,'width':2,'cell-width':3,'layers':layers,'directions':directions,
                      'projection':projection,'bias':bias}
            result=retain(self.w,family=definition(),bindings=bindings,materialize=False)
            native=compile_specialization(self.w,result['specialization'],constrained=False)
            exported=self.Family(bindings=bindings);exported.network.load_state_dict(native.state_dict(),strict=True)
            for route in (native,exported):
                baseline=copy.deepcopy(route.contract);owners=[id(p) for p in route.parameters()]
                for t in (1,3,7):
                    with self.subTest(bindings=bindings,t=t,route=type(route).__name__):self.check_oracle(route,t)
                self.assertEqual(route.contract,baseline);self.assertEqual([id(p) for p in route.parameters()],owners)
                self.assertEqual(sum(p.numel() for p in route.parameters()),baseline['parameters'])

    def test_zero_lengths_nan_padding_states_and_every_gradient(self):
        result=retain(self.w,family=definition(),materialize=False)
        native=compile_specialization(self.w,result['specialization'],constrained=False)
        for route in (native,self.Family()):
            for t in (1,9):self.check_oracle(route,t,empty=True)

    def test_runtime_contract_is_private_and_provenance_matches_witness(self):
        model=self.Family();proof=model.runtime_specification()
        self.assertEqual(proof['version'],2)
        self.assertEqual(model.provenance()['runtime_temporal_v2'],{'sequence':[0]})
        self.assertEqual(model.constructor_specification()['runtime_policy'],definition()['runtime_policy'])
        proof['runtime_inputs']['sequence'][0]=99
        self.assertEqual(model.runtime_specification()['runtime_inputs'],{'sequence':[0]})

    def test_35_variable_time_updates_native_and_checked_match_independent_student(self):
        bindings={'time':4,'width':1,'cell-width':3,'layers':1,'directions':False,'projection':2}
        result=retain(self.w,family=definition(),bindings=bindings,materialize=False)
        native=compile_specialization(self.w,result['specialization'],constrained=False)
        checked=self.Family(bindings=bindings);checked.network.load_state_dict(native.state_dict(),strict=True)
        keys=('sequence','hidden','cell')
        for route in (native,checked):
            network=getattr(route,'network',route);params=dict(network.layer_0.named_parameters())
            reference={k:p.detach().clone().requires_grad_() for k,p in params.items()}
            teacher={k:.4*torch.sin(torch.arange(p.numel(),dtype=torch.float32).reshape(p.shape)+i)
                     for i,(k,p) in enumerate(params.items())}
            aa=torch.optim.Adam(route.parameters(),lr=.025);bb=torch.optim.Adam(reference.values(),lr=.025)
            owners=[id(p) for p in route.parameters()]
            def expected(values,weights):return oracle(*(values[k] for k in keys),weights,
                                                       lengths=values['lengths'][:,0].long(),layers=1,bidirectional=False,projection=2)
            loss=lambda output,target:sum((output[k]-target[k]).square().mean() for k in keys)
            c=self.actual(route,7);evaluation=data(c);target=expected(evaluation,teacher)
            initial=float(loss(returned(c,route(named(c,evaluation))),target).detach())
            for step in range(35):
                c=self.actual(route,(1,3,7)[step%3]);values=data(c);target_step=expected(values,teacher)
                aa.zero_grad();bb.zero_grad();actual=returned(c,route(named(c,values)))
                la=loss(actual,target_step);lb=loss(expected(values,reference),target_step)
                torch.testing.assert_close(la,lb,rtol=5e-6,atol=2e-6);la.backward();lb.backward();aa.step();bb.step()
                for k,p in params.items():torch.testing.assert_close(p,reference[k],rtol=8e-6,atol=3e-6)
                self.assertEqual([id(p) for p in route.parameters()],owners)
            c=self.actual(route,7);final=float(loss(returned(c,route(named(c,evaluation))),target).detach())
            self.assertLess(final,initial)

    def test_shared_family_owner_independent_branch_times_and_all_gradients(self):
        doc=definition();graph=doc['models'][doc['root']]['graph'];lookup={n['id']:n for n in graph['nodes']}
        second=copy.deepcopy(lookup['lstm']);second.update(id='shared',share_with='lstm')
        second['inputs']['sequence']='sequence2';second['lengths']['source']='lengths2'
        graph['nodes'] += [dict(copy.deepcopy(lookup['sequence']),id='sequence2',label='Second frames'),
                           dict(copy.deepcopy(lookup['lengths']),id='lengths2',label='Second lengths'),second]
        graph['input_order'] += ['sequence2','lengths2'];doc['runtime_policy']['inputs']['sequence2']=[0]
        graph['edges'] += [{'from':source,'to':'shared'} for source in ('sequence2','hidden','cell','lengths2')]
        for port in ('sequence','hidden','cell'):
            graph['nodes'].append({'id':'second-'+port,'type':'Output','label':'Second '+port})
            graph['edges'].append({'from':'shared','from_port':port,'to':'second-'+port})
        doc['assignments'] += [dict(copy.deepcopy(a),node='shared' if a['node']=='lstm' else 'sequence2')
                               for a in list(doc['assignments']) if a['node'] in ('lstm','sequence')]
        bindings={'time':4,'width':2,'cell-width':3,'layers':1,'projection':2,'directions':True}
        result=retain(self.w,family=doc,bindings=bindings,materialize=False)
        native=compile_specialization(self.w,result['specialization'],constrained=False)
        namespace={};exec(family_source(doc,constrained=False)['code'],namespace)
        checked=namespace['ArchitectureFamily'](bindings=bindings);checked.network.load_state_dict(native.state_dict(),strict=True)
        for route in (native,checked):
            network=getattr(route,'network',route);owners=[id(p) for p in route.parameters()]
            self.assertEqual(sum(p.numel() for p in route.parameters()),156)
            for t1,t2 in ((1,9),(7,2),(3,3)):
                shapes={g['id']:list(g['shape']) for g in route.contract['input_groups']}
                shapes['sequence'][0]=t1;shapes['sequence2'][0]=t2
                c=route.preview_shapes(shapes,batch_size=3)['contract'];torch.manual_seed(42)
                values={g['id']:torch.randn(3,*g['shape']) for g in c['input_groups']}
                values['lengths']=torch.tensor([[0],[t1],[max(0,t1-1)]],dtype=torch.float32)
                values['lengths2']=torch.tensor([[t2],[0],[max(0,t2-1)]],dtype=torch.float32)
                for seq,lengths in (('sequence','lengths'),('sequence2','lengths2')):
                    for row in range(3):values[seq][row,int(values[lengths][row,0]):]=float('nan')
                keys=('sequence','sequence2','hidden','cell')
                for k in keys:values[k].requires_grad_()
                inputs={k:values[k].detach().clone().requires_grad_() for k in keys}
                weights={k:p.detach().clone().requires_grad_() for k,p in network.layer_0.named_parameters()}
                expected={}
                for seq,lengths,prefix in (('sequence','lengths','Returned '),('sequence2','lengths2','Second ')):
                    output=oracle(inputs[seq],inputs['hidden'],inputs['cell'],weights,
                                  lengths=values[lengths][:,0].long(),layers=1,bidirectional=True,projection=2)
                    expected.update({prefix+k:v for k,v in output.items()})
                actual=route(named(c,values))
                for k in expected:torch.testing.assert_close(actual[k],expected[k],rtol=5e-6,atol=3e-6)
                loss=lambda r:sum((i+1)*r[k].square().sum() for i,k in enumerate(sorted(expected)))
                aa=torch.autograd.grad(loss(actual),[*(values[k] for k in keys),*route.parameters()])
                bb=torch.autograd.grad(loss(expected),[*(inputs[k] for k in keys),*weights.values()])
                for a,b in zip(aa,bb):torch.testing.assert_close(a,b,rtol=2e-5,atol=6e-6)
                self.assertEqual([id(p) for p in route.parameters()],owners)

    def test_new_policy_cannot_be_overridden_or_applied_to_old_family(self):
        for options in ({'runtime_temporal_v2':{'sequence':[1]}}, {'runtime_inputs':{'sequence':[1]}},
                        {'runtime_temporal':{'sequence':[0]}}):
            with self.subTest(options=options),patch('torch.nn.LSTM') as allocated,self.assertRaises(ValueError):self.Family(**options)
            allocated.assert_not_called()
        namespace={};exec(family_source(fixed_definition(),constrained=False)['code'],namespace)
        with patch('torch.nn.LSTM') as allocated,self.assertRaisesRegex(ValueError,'family8'):
            namespace['ArchitectureFamily'](runtime_temporal_v2={'sequence':[0]})
        allocated.assert_not_called()
        doc=fixed_definition();doc['version']=8;namespace={}
        exec(family_source(doc,constrained=False)['code'],namespace)
        with patch('torch.nn.LSTM') as allocated,self.assertRaisesRegex(ValueError,'exact retained'):
            namespace['ArchitectureFamily'](runtime_temporal_v2={'sequence':[0]})
        allocated.assert_not_called()

    def test_current_ram_unknown_or_small_refuses_before_native_or_export_allocation(self):
        result=retain(self.w,family=definition(),materialize=False)
        capsule=self.namespace['_runtime']()
        for available in (None,1):
            with patch('app.runtime_shapes.available_memory',return_value=available),patch('torch.nn.LSTM') as allocated,self.assertRaises(GraphError):
                compile_specialization(self.w,result['specialization'],constrained=True)
            allocated.assert_not_called()
            with patch.object(capsule.runtime_shapes,'available_memory',return_value=available),patch('torch.nn.LSTM') as allocated,self.assertRaises(ValueError):
                self.Family(constrained=True)
            allocated.assert_not_called()

    def test_warm_start_time_only_keeps_semantic_ids_and_every_prediction(self):
        a=self.Family(bindings={'time':3});b=self.Family(bindings={'time':9})
        self.assertEqual(a.semantic_parameter_schema(),b.semantic_parameter_schema())
        receipt=b.warm_start(a.semantic_state(),strict=True)
        self.assertFalse(receipt['optimizer_resume']);self.assertEqual(receipt['skipped'],[])
        for t in (1,4,11):
            c=self.actual(a,t);x=named(c,data(c));aa=a(x);bb=b(x)
            for k in aa:torch.testing.assert_close(aa[k],bb[k],rtol=0,atol=0)

    def test_huge_baseline_initial_actual_shapes_control_construction_and_warm_start(self):
        doc=definition();next(s for s in doc['symbols'] if s['id']=='time')['maximum']=10000000
        namespace={};exec(family_source(doc,constrained=True)['code'],namespace);F=namespace['ArchitectureFamily']
        capsule=namespace['_runtime']();resolved=specialize(doc,{'time':10000000})
        initial={g['id']:copy.deepcopy(g['shape']) for g in resolved['contract']['input_groups']};initial['sequence'][0]=2
        with patch.object(capsule.runtime_shapes,'available_memory',return_value=4000000000):
            a=F(bindings={'time':10000000},initial_shapes=initial)
            b=F(bindings={'time':10000000},initial_shapes=initial)
            receipt=b.warm_start(a.semantic_state(),strict=True)
            self.assertEqual(len(receipt['loaded']),len(a.state_dict()))
            c=self.actual(a,2);x=named(c,data(c));aa=a(x);bb=b(x)
            for k in aa:torch.testing.assert_close(aa[k],bb[k],rtol=0,atol=0)
            with patch('torch.nn.LSTM') as allocated,self.assertRaises(ValueError):F(bindings={'time':10000000})
            allocated.assert_not_called()

    def test_ram_refusal_before_warm_start_copy_leaves_weights_unchanged(self):
        a=self.Family();b=self.Family(constrained=True);checkpoint=a.semantic_state()
        before={k:v.clone() for k,v in b.state_dict().items()};capsule=self.namespace['_runtime']()
        with patch.object(capsule.runtime_shapes,'available_memory',return_value=1),self.assertRaises(ValueError):b.warm_start(checkpoint,strict=True)
        for k,v in b.state_dict().items():torch.testing.assert_close(v,before[k],rtol=0,atol=0)

    def test_named_ports_required_and_fixed_state_axes_rejected_before_execution(self):
        model=self.Family();c=self.actual(model,4);x=named(c,data(c))
        with patch.object(model.network.layer_0,'forward') as ran:
            with self.assertRaises(ValueError):model(torch.randn(4,c['inputs']))
            bad=copy.deepcopy(x);name=next(g['name'] for g in c['input_groups'] if g['id']=='hidden')
            bad[name]=torch.zeros(4,1,1)
            with self.assertRaises(ValueError):model(bad)
        ran.assert_not_called()

    def test_saved_reload_native_and_fresh_app_free_export_checkpoint_predictions(self):
        result=retain(self.w,family=definition(),materialize=False)
        with tempfile.TemporaryDirectory() as td:
            folder=Path(td);workspace=folder/'family.vtw';self.w.save(workspace);before=workspace.read_bytes()
            reopened=Workspace.load(workspace)
            try:native=compile_specialization(reopened,result['specialization'],constrained=False)
            finally:reopened.db.close()
            self.assertEqual(workspace.read_bytes(),before)
            exported=self.Family();exported.network.load_state_dict(native.state_dict(),strict=True)
            expected={};inputs={}
            for t in (1,4,9):
                c=self.actual(native,t);x=named(c,data(c));inputs[str(t)]={k:v.tolist() for k,v in x.items()}
                expected[str(t)]={k:v.detach().tolist() for k,v in native(x).items()}
            (folder/'family.py').write_text(self.source['code']);torch.save(exported.state_dict(),folder/'weights.pt')
            (folder/'inputs.json').write_text(json.dumps(inputs))
            code="""
import json,runpy,sys,torch
torch.set_num_threads(1)
s=runpy.run_path(sys.argv[1]);assert s['preview_family']()['status']=='verified'
m=s['ArchitectureFamily']();m.load_state_dict(torch.load(sys.argv[2],weights_only=True),strict=True)
rows=json.load(open(sys.argv[3]));actual={}
for t,row in rows.items():
    x={k:torch.tensor(v,dtype=torch.float32) for k,v in row.items()}
    actual[t]={k:v.detach().tolist() for k,v in m(x).items()}
assert not any(k=='app' or k.startswith('app.') for k in sys.modules)
print(json.dumps(actual))
"""
            ran=subprocess.run([sys.executable,'-c',code,str(folder/'family.py'),str(folder/'weights.pt'),str(folder/'inputs.json')],
                               cwd=folder,capture_output=True,text=True,timeout=30)
            self.assertEqual(ran.returncode,0,ran.stderr);self.assertEqual(json.loads(ran.stdout),expected)
