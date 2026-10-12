"""Actual reused tensors/gradients/optimizers and standalone runtime guards."""
import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F
from app.core import GraphError, compile_model, validate_model
from app.family_export import family_source, write_new_family
from app.runtime_shapes import compile_runtime_model
from runtime_spatial_fixtures import spatial, nested, family

torch.set_num_threads(1)


def compare_gradients(case,actual,expected,inputs,oracle_inputs,params,oracle_params):
    torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-12)
    aa=torch.autograd.grad(actual.square().sum(),[*inputs,*params])
    bb=torch.autograd.grad(expected.square().sum(),[*oracle_inputs,*oracle_params])
    for a,b in zip(aa,bb):torch.testing.assert_close(a,b,rtol=1e-10,atol=1e-11)


class RuntimeExecutionTests(unittest.TestCase):
    def test_179_parameter_adaptive_head_reuses_same_weights_at_many_sizes_and_gradients(self):
        models,mid=spatial();m=compile_model(models,mid,False,runtime_inputs={'input':[1,2]}).double()
        identities=[id(p) for p in m.parameters()];baseline=copy.deepcopy(m.contract)
        self.assertEqual(sum(p.numel() for p in m.parameters()),179)
        for batch,h,w in ((1,1,1),(2,3,9),(3,11,4),(2,6,7)):
            x=torch.linspace(-.6,.8,batch*4*h*w,dtype=torch.float64).reshape(batch,4,h,w).requires_grad_()
            xx=x.detach().clone().requires_grad_();ps=[p.detach().clone().requires_grad_() for p in m.parameters()]
            cw,cb,hw,hb=ps
            expected=F.linear(F.adaptive_avg_pool2d(torch.tanh(F.conv2d(xx,cw,cb,padding=1,groups=2)),(1,1)).flatten(1),hw,hb)
            compare_gradients(self,m(x),expected,[x],[xx],list(m.parameters()),ps)
            self.assertEqual([id(p) for p in m.parameters()],identities);self.assertEqual(m.contract,baseline)
        native=compile_model(models,mid,False).double();native.load_state_dict(m.state_dict(),strict=True)
        x=torch.zeros(2,4,5,6,dtype=torch.float64);torch.testing.assert_close(m(x),native(x),rtol=0,atol=0)

    def test_bad_shapes_dtype_layout_and_memory_fail_before_first_layer_or_packing(self):
        models,mid=spatial();m=compile_runtime_model(models,mid,{'input':[1,2]},constrained=True)
        calls=[];handles=[layer.register_forward_pre_hook(lambda module,args:calls.append(module)) for layer in m.children()]
        try:
            for x in (torch.zeros(1,3,5,6),torch.zeros(0,4,5,6),torch.zeros(1,4,0,6),torch.zeros(4,5,6),
                      torch.zeros(1,4,5,6,dtype=torch.float64),torch.zeros(1,4,5,6,dtype=torch.int32),
                      torch.empty(1,4,5,6,device='meta'), {'wrong':torch.zeros(1,4,5,6)}):
                with self.subTest(shape=str(type(x))),self.assertRaises(GraphError):m(x)
                self.assertEqual(calls,[])
            with patch('app.runtime_shapes.available_memory',return_value=1),patch('app.core._pack_inputs',side_effect=AssertionError('Do not pack')):
                with self.assertRaisesRegex(GraphError,'Constrained.*runtime'):m(torch.zeros(1,4,5,6))
            self.assertEqual(calls,[])
        finally:
            for h in handles:h.remove()

    def test_nonpositive_pooling_and_output_shape_guards_precede_conv_execution(self):
        models,mid=spatial();nodes=models[mid]['graph']['nodes'];next(n for n in nodes if n['id']=='conv')['padding']=[0,0]
        m=compile_runtime_model(models,mid,{'input':[1,2]},constrained=False)
        with patch.object(m.layer_0,'forward',side_effect=AssertionError('Do not run Conv')):
            with self.assertRaisesRegex(GraphError,'nonpositive'):m(torch.zeros(1,4,2,7))
        models,mid=spatial();nodes=models[mid]['graph']['nodes']
        nodes[:]=[n for n in nodes if n['id'] not in {'pool','flatten','head'}]
        next(n for n in nodes if n['id']=='output')['shape']=[8,5,6]
        models[mid]['graph']['edges']=[{'from':a,'to':b} for a,b in [('input','conv'),('conv','activation'),('activation','output')]]
        m=compile_runtime_model(models,mid,{'input':[1,2]},constrained=False)
        with patch.object(m.layer_0,'forward',side_effect=AssertionError('No forward')):
            with self.assertRaisesRegex(GraphError,'Output.*declares'):m(torch.zeros(1,4,7,9))

    def test_non_square_pooling_permute_and_gradients_match_functional_operator_rules(self):
        for kind in ('AvgPool2d','MaxPool2d'):
            models,mid=spatial();doc=models[mid]
            doc['graph']['nodes'].insert(2,{'id':'transpose','type':'Permute','dims':[0,2,1]})
            pool={'id':'fixed','type':kind,'kernel_size':[2,3],'stride':[1,2],'padding':[0,1],
                  **({'count_include_pad':False} if kind=='AvgPool2d' else {'dilation':[2,1]})}
            doc['graph']['nodes'].insert(3,pool)
            doc['graph']['edges']=[{'from':a,'to':b} for a,b in [('input','conv'),('conv','transpose'),('transpose','fixed'),('fixed','activation'),('activation','pool'),('pool','flatten'),('flatten','head'),('head','output')]]
            with self.assertRaisesRegex(GraphError,'needs axes.*not'):compile_runtime_model(models,mid,{'input':[1,2]},constrained=False)
            doc['graph']['nodes'].insert(3,{'id':'restore','type':'Permute','dims':[0,2,1]})
            doc['graph']['edges']=[{'from':a,'to':b} for a,b in [('input','conv'),('conv','transpose'),('transpose','restore'),('restore','fixed'),('fixed','activation'),('activation','pool'),('pool','flatten'),('flatten','head'),('head','output')]]
            m=compile_runtime_model(models,mid,{'input':[1,2]},constrained=False).double()
            for h,w in ((7,11),(9,5)):
                x=torch.linspace(-.55,.81,2*4*h*w,dtype=torch.float64).reshape(2,4,h,w).requires_grad_()
                xx=x.detach().clone().requires_grad_();ps=[p.detach().clone().requires_grad_() for p in m.parameters()];cw,cb,hw,hb=ps
                value=F.conv2d(xx,cw,cb,padding=1,groups=2).permute(0,1,3,2).permute(0,1,3,2)
                value=(F.avg_pool2d(value,(2,3),stride=(1,2),padding=(0,1),count_include_pad=False) if kind=='AvgPool2d' else
                       F.max_pool2d(value,(2,3),stride=(1,2),padding=(0,1),dilation=(2,1)))
                expected=F.linear(F.adaptive_avg_pool2d(torch.tanh(value),(1,1)).flatten(1),hw,hb)
                compare_gradients(self,m(x),expected,[x],[xx],list(m.parameters()),ps)

    def test_shared_encoder_different_named_resolutions_and_independent_functional_gradients(self):
        models,mid=nested(shared=True);m=compile_model(models,mid,False,runtime_inputs={'a':[1,2],'b':[1,2]}).double()
        self.assertEqual(sum(p.numel() for p in m.parameters()),203)
        self.assertEqual(len([p for p in m.modules() if isinstance(p,torch.nn.Conv2d)]),1)
        a=torch.linspace(-.3,.7,2*4*3*9,dtype=torch.float64).reshape(2,4,3,9).requires_grad_()
        b=torch.linspace(-.5,.8,2*4*7*2,dtype=torch.float64).reshape(2,4,7,2).requires_grad_()
        aa=a.detach().clone().requires_grad_();bb=b.detach().clone().requires_grad_();ps=[p.detach().clone().requires_grad_() for p in m.parameters()]
        cw,cb,hw,hb=ps
        def branch(x):return F.adaptive_avg_pool2d(torch.tanh(F.conv2d(x,cw,cb,padding=1,groups=2)),(1,1)).flatten(1)
        expected=F.linear(torch.cat([branch(aa),branch(bb)],1),hw,hb)
        compare_gradients(self,m({'left':a,'right':b}),expected,[a,b],[aa,bb],list(m.parameters()),ps)
        values=m.intermediate_values({'left':a,'right':b})
        self.assertEqual(list(values['left'].shape),[2,8,3,9]);self.assertEqual(list(values['right'].shape),[2,8,7,2])
        torch.testing.assert_close(m.forward_tensor({'left':a,'right':b}),expected,rtol=1e-12,atol=1e-12)

    def test_different_ports_require_names_and_matching_batches_before_operations(self):
        models,mid=nested();m=compile_model(models,mid,False,runtime_inputs={'a':[1,2],'b':[1,2]})
        for x in (torch.zeros(1,240),{'left':torch.zeros(1,4,5,6)},
                  {'left':torch.zeros(2,4,3,9),'right':torch.zeros(1,4,7,2)},
                  {'left':torch.zeros(1,4,3,9),'right':torch.zeros(1,4,7,2,dtype=torch.float64)}):
            with patch('torch.cat',side_effect=AssertionError('Do not pack')):
                with self.assertRaises(GraphError):m(x)

    def test_variable_output_shapes_and_per_call_split_ports_preserve_boundaries(self):
        models,mid=spatial();doc=models[mid]
        doc['graph']['nodes']=[n for n in doc['graph']['nodes'] if n['id'] not in {'pool','flatten','head','output'}]
        doc['graph']['nodes'] += [{'id':'route','type':'Replicator','outputs':['first','second']},
                                {'id':'first','type':'Output','label':'first'},{'id':'second','type':'Output','label':'second'}]
        doc['graph']['edges']=[{'from':a,'to':b} for a,b in [('input','conv'),('conv','activation'),('activation','route')]]
        doc['graph']['edges'] += [{'from':'route','to':pid,'from_port':pid} for pid in ('first','second')]
        m=compile_runtime_model(models,mid,{'input':[1,2]},constrained=False)
        for h,w in ((3,7),(9,2)):
            x=torch.randn(2,4,h,w);out=m(x)
            self.assertEqual(list(out['first'].shape),[2,8,h,w]);self.assertIs(out['first'],out['second'])
            torch.testing.assert_close(m.forward_tensor(x),out['first'],rtol=0,atol=0)
            self.assertEqual(m.preview_shapes({'input':[4,h,w]})['contract']['output_shape'],[8,h,w])

    def test_reshape_inferred_runtime_axis_executes_actual_shape_and_preserves_layer_hook(self):
        models,mid=spatial();doc=models[mid]
        doc['graph']['nodes']=[n for n in doc['graph']['nodes'] if n['id'] not in {'pool','flatten'}]
        doc['graph']['nodes'].insert(-2,{'id':'reshape','type':'Reshape','shape':[-1,8]})
        doc['graph']['edges']=[{'from':a,'to':b} for a,b in [('input','conv'),('conv','activation'),('activation','reshape'),('reshape','head'),('head','output')]]
        m=compile_runtime_model(models,mid,{'input':[1,2]},constrained=False).double()
        calls=[];h=m.layer_2.register_forward_hook(lambda module,args,out:calls.append(list(out.shape)))
        try:
            x=torch.randn(2,4,3,7,dtype=torch.float64)
            expected=F.linear(torch.tanh(F.conv2d(x,m.layer_0.weight,m.layer_0.bias,padding=1,groups=2)).reshape(2,21,8),m.layer_3.weight,m.layer_3.bias)
            torch.testing.assert_close(m(x),expected,rtol=1e-12,atol=1e-12);self.assertEqual(calls,[[2,21,8]])
        finally:h.remove()

    def test_optimizer_identity_across_resolution_changes_and_rejected_call(self):
        models,mid=nested(shared=True);m=compile_runtime_model(models,mid,{'a':[1,2],'b':[1,2]},constrained=False)
        optimizer=torch.optim.Adam(m.parameters(),lr=.01);identities=[id(p) for p in m.parameters()]
        for h,w in ((3,5),(7,2),(2,9)):
            optimizer.zero_grad();loss=m({'left':torch.ones(2,4,h,w),'right':torch.ones(2,4,w,h)}).square().mean();loss.backward();optimizer.step()
            self.assertEqual([id(p) for p in m.parameters()],identities)
        self.assertEqual(len(optimizer.state),4);self.assertTrue(all(int(v['step'])==3 for v in optimizer.state.values()))
        before={k:v.clone() for k,v in m.state_dict().items()}
        with self.assertRaises(GraphError):m({'left':torch.zeros(2,3,3,5),'right':torch.zeros(2,4,3,5)})
        for key,value in m.state_dict().items():torch.testing.assert_close(value,before[key],rtol=0,atol=0)

    def test_source_and_spec_copies_are_private_no_view_or_rng_mutation(self):
        models,mid=spatial();declaration={'input':[1,2]};m=compile_runtime_model(models,mid,declaration,constrained=False)
        baseline=copy.deepcopy(m.contract);rng=torch.get_rng_state().clone()
        models[mid]['graph']['nodes'][0]['shape']=[99,99,99];declaration['input'].clear()
        leaked=m.runtime_specification();leaked['runtime_inputs']['input'].clear()
        first=m.preview_shapes({'input':[4,9,11]},batch_size=3);first['contract']['shapes']['conv']['output_shape'][0]=999
        self.assertEqual(m.preview_shapes({'input':[4,9,11]})['contract']['shapes']['conv']['output_shape'],[8,9,11])
        self.assertEqual(m.contract,baseline);self.assertTrue(torch.equal(torch.get_rng_state(),rng))
        self.assertEqual(list(m(torch.zeros(1,4,7,2)).shape),[1,3])

    def test_exported_family_actual_dynamic_constructor_warm_start_and_app_free_execution(self):
        source=family();scope={};exec(family_source(source,constrained=False)['code'],scope)
        a=scope['ArchitectureFamily'](runtime_inputs={'input':[1,2]}).double()
        b=scope['ArchitectureFamily'](runtime_inputs={'input':[1,2]}).double();b.warm_start(a.semantic_state(),strict=True)
        for h,w in ((3,7),(11,2)):
            x=torch.randn(2,4,h,w,dtype=torch.float64);torch.testing.assert_close(a(x),b(x),rtol=0,atol=0)
        self.assertEqual(a.runtime_specification()['parameter_shapes'],'statically-independent-of-runtime-axes')
        self.assertEqual(a.preview_shapes({'input':[4,17,19]})['contract']['parameters'],179)
        with tempfile.TemporaryDirectory() as td:
            destination=Path(td)/'family.py';write_new_family(family_source(source,constrained=False),destination)
            code="""import json,sys,torch
from family import ArchitectureFamily
torch.set_num_threads(1)
m=ArchitectureFamily(runtime_inputs={'input':[1,2]})
optimizer=torch.optim.Adam(m.parameters(),lr=.01)
ids=[id(p) for p in m.parameters()]
for h,w in [(3,7),(9,2),(2,11)]:
 optimizer.zero_grad();m(torch.ones(2,4,h,w)).square().mean().backward();optimizer.step()
 assert ids==[id(p) for p in m.parameters()]
assert not any(k=='app' or k.startswith('app.') for k in sys.modules)
print(json.dumps({'parameters':sum(p.numel() for p in m.parameters()),'steps':[int(v['step']) for v in optimizer.state.values()]}))
"""
            env={k:v for k,v in os.environ.items() if k!='PYTHONPATH'}
            done=subprocess.run([sys.executable,'-c',code],cwd=td,env=env,text=True,capture_output=True,check=True,timeout=60)
            self.assertEqual(json.loads(done.stdout),{'parameters':179,'steps':[3,3,3,3]})

    def test_runtime_declaration_and_constructor_ram_fail_before_new_layer_allocation(self):
        models,mid=spatial()
        with patch('torch.nn.Conv2d',side_effect=AssertionError('Do not allocate')):
            with self.assertRaises(GraphError):compile_runtime_model(models,mid,{'input':[0]},constrained=False)
            with patch('app.core.available_memory',return_value=1):
                with self.assertRaisesRegex(GraphError,'Constrained'):compile_runtime_model(models,mid,{'input':[1,2]},constrained=True)

    def test_no_grad_admission_cost_and_module_hook_match_normal_pytorch(self):
        models,mid=spatial();m=compile_runtime_model(models,mid,{'input':[1,2]},constrained=False)
        calls=[];h=m.register_forward_hook(lambda module,args,out:calls.append(list(out.shape)))
        try:
            with torch.no_grad():self.assertEqual(list(m(torch.zeros(2,4,9,7)).shape),[2,3])
            self.assertEqual(calls,[[2,3]])
            a=m.preview_shapes({'input':[4,9,7]},batch_size=2,gradients=False)
            b=m.preview_shapes({'input':[4,9,7]},batch_size=2,gradients=True)
            self.assertGreater(b['safety']['memory']['estimated_bytes'],a['safety']['memory']['estimated_bytes'])
        finally:h.remove()
