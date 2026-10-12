"""Allocation-free dependency proofs, shape guards and runtime admission."""
import copy
import json
import random
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from app.core import GraphError, validate_model, MAX_INTEGER
from app.runtime_shapes import declaration, specification, shape_contract, admission, compile_runtime_model
from runtime_spatial_fixtures import spatial, nested


class RuntimeShapeTests(unittest.TestCase):
    def test_adaptive_parameter_proof_and_many_non_square_cases_are_allocation_free(self):
        models,mid = spatial(); before = copy.deepcopy(models); rng = random.getstate()
        spec = specification(models,mid,{'input':[2,1]})
        self.assertEqual(spec['runtime_inputs'],{'input':[1,2]});self.assertEqual(spec['status'],'conditional')
        self.assertEqual(spec['parameter_shapes'],'statically-independent-of-runtime-axes')
        self.assertEqual(spec['output_dependencies'],[[]]);self.assertEqual(validate_model(models,mid)['parameters'],179)
        for h in (1,2,3,5,9,17):
            for w in (1,4,6,11):
                c = shape_contract(models,mid,spec,{'input':[4,h,w]})
                self.assertEqual(c['shapes']['conv']['output_shape'],[8,h,w]);self.assertEqual(c['output_shape'],[3])
                self.assertEqual(c['parameters'],179)
        self.assertEqual(models,before);self.assertEqual(rng,random.getstate())

    def test_empty_unknown_duplicate_noninteger_channels_and_batch_axes_rejected(self):
        models,mid = spatial()
        for value in (None,{},[],{'unknown':[1]}, {'input':[]},{'input':[1,1]},{'input':[True]},
                      {'input':[0]},{'input':[3]},{'input':[-1]},{'input':[1.0]},{'input':'H,W'}):
            with self.subTest(value=value),self.assertRaises(GraphError): declaration(models,mid,value)
        models[mid]['graph']['nodes'][0]['axes'] = ['channel','height','width']
        with self.assertRaisesRegex(GraphError,'C,H,W'): declaration(models,mid,{'input':[1,2]})

    def test_actual_shapes_require_exact_input_ids_ranks_channels_positive_exact_dimensions(self):
        models,mid = spatial();spec = specification(models,mid,{'input':[1]})
        for value in (None,{}, {'other':[4,5,6]}, {'input':[4,5,7]}, {'input':[3,5,6]},
                      {'input':[4,5]}, {'input':[4,0,6]}, {'input':[4,True,6]}, {'input':[4,1.0,6]},
                      {'input':[4,MAX_INTEGER,6]}):
            with self.subTest(value=value),self.assertRaises(GraphError): shape_contract(models,mid,spec,value)
        self.assertEqual(shape_contract(models,mid,spec,{'input':[4,19,6]})['parameters'],179)

    def test_fixed_flatten_head_reports_affected_linear_and_explicit_repair_before_layers(self):
        models,mid = spatial();doc=models[mid]
        doc['graph']['nodes']=[n for n in doc['graph']['nodes'] if n['id']!='pool']
        doc['graph']['edges']=[{'from':a,'to':b} for a,b in [('input','conv'),('conv','activation'),('activation','flatten'),('flatten','head'),('head','output')]]
        for n in doc['graph']['nodes']:
            if n['id']=='head': n['features']=3
        with patch('app.core.compile_model',side_effect=AssertionError('Do not construct')):
            with self.assertRaisesRegex(GraphError,'cnn/head.*Linear.*runtime.*Bind image resolution.*AdaptiveAvgPool2d'):
                compile_runtime_model(models,mid,{'input':[1,2]},constrained=False)
        self.assertEqual(validate_model(models,mid)['shapes']['head']['input'],240)

    def test_nested_shared_calls_have_independent_actual_shapes_and_one_parameter_owner(self):
        models,mid=nested(shared=True);spec=specification(models,mid,{'a':[1,2],'b':[1,2]})
        c=shape_contract(models,mid,spec,{'a':[4,3,9],'b':[4,7,2]})
        items={p['id']:p for p in c['plan']}
        self.assertEqual(items['left']['runtime_contract']['output_shape'],[8,3,9])
        self.assertEqual(items['right']['runtime_contract']['output_shape'],[8,7,2])
        self.assertEqual(items['right']['parameter_owner'],'left');self.assertEqual(c['parameters'],203)
        self.assertNotIn('runtime_contract',validate_model(models,mid)['plan'][2])

    def test_nonpositive_kernel_output_and_explicit_output_guard_reject_actual_case(self):
        models,mid=spatial();conv=next(n for n in models[mid]['graph']['nodes'] if n['id']=='conv');conv['padding']=[0,0]
        spec=specification(models,mid,{'input':[1,2]})
        with self.assertRaisesRegex(GraphError,'nonpositive'):shape_contract(models,mid,spec,{'input':[4,2,6]})
        next(n for n in models[mid]['graph']['nodes'] if n['id']=='output')['shape']=[3]
        self.assertEqual(shape_contract(models,mid,spec,{'input':[4,9,11]})['output_shape'],[3])

    def test_concat_spatial_equality_is_conditional_not_guessed_from_example(self):
        models,mid=nested();doc=models[mid]
        doc['graph']['nodes']=[n for n in doc['graph']['nodes'] if n['id'] not in {'lp','rp','head'}]
        next(n for n in doc['graph']['nodes'] if n['id']=='join').update(sources=['left','right'],flatten=False)
        doc['graph']['edges']=[{'from':a,'to':b} for a,b in [('a','left'),('b','right'),('left','join'),('right','join'),('join','out')]]
        spec=specification(models,mid,{'a':[1,2],'b':[1,2]})
        self.assertEqual(shape_contract(models,mid,spec,{'a':[4,7,2],'b':[4,7,9]})['output_shape'],[8,7,11])
        with self.assertRaisesRegex(GraphError,'Concat.*matching'):shape_contract(models,mid,spec,{'a':[4,7,2],'b':[4,8,9]})

    def test_permute_and_fixed_pool_rules_propagate_runtime_axes_without_batch(self):
        models,mid=spatial();doc=models[mid]
        nodes=doc['graph']['nodes'];nodes.insert(2,{'id':'transpose','type':'Permute','dims':[0,2,1]})
        nodes.insert(3,{'id':'fixed-pool','type':'AvgPool2d','kernel_size':[2,3],'stride':[1,2],'padding':[0,1]})
        doc['graph']['edges']=[{'from':a,'to':b} for a,b in [('input','conv'),('conv','transpose'),('transpose','fixed-pool'),('fixed-pool','activation'),('activation','pool'),('pool','flatten'),('flatten','head'),('head','output')]]
        with self.assertRaisesRegex(GraphError,'needs axes.*not'):specification(models,mid,{'input':[1,2]})
        nodes.insert(3,{'id':'restore','type':'Permute','dims':[0,2,1]})
        doc['graph']['edges']=[{'from':a,'to':b} for a,b in [('input','conv'),('conv','transpose'),('transpose','restore'),('restore','fixed-pool'),('fixed-pool','activation'),('activation','pool'),('pool','flatten'),('flatten','head'),('head','output')]]
        spec=specification(models,mid,{'input':[1,2]})
        c=shape_contract(models,mid,spec,{'input':[4,9,7]})
        self.assertEqual(c['shapes']['transpose']['output_shape'],[8,7,9])
        self.assertEqual(c['shapes']['fixed-pool']['output_shape'],[8,8,4]);self.assertEqual(c['parameters'],179)

    def test_full_batch_resolution_and_double_precision_charge_current_memory(self):
        models,mid=spatial();spec=specification(models,mid,{'input':[1,2]})
        small=shape_contract(models,mid,spec,{'input':[4,5,6]});large=shape_contract(models,mid,spec,{'input':[4,51,63]})
        with patch('app.runtime_shapes.available_memory',return_value=10**12):
            a=admission(small,1);b=admission(large,17);d=admission(large,17,dtype_bytes=8)
            self.assertGreater(b['memory']['estimated_bytes'],a['memory']['estimated_bytes'])
            self.assertEqual(d['memory']['components']['activations'],2*b['memory']['components']['activations'])
            for key in ('parameters','activations','convolution_workspace','boundary_packing'):self.assertIn(key,b['memory']['components'])
        for available in (None,0,1):
            with patch('app.runtime_shapes.available_memory',return_value=available),self.assertRaisesRegex(GraphError,'Constrained'):admission(small,1)
        with patch('app.runtime_shapes.available_memory',return_value=None):self.assertTrue(admission(small,1,constrained=False)['allowed'])

    def test_runtime_context_reservation_counts_pending_nested_calls_before_65th_plan(self):
        models,leaf=spatial();next(n for n in models[leaf]['graph']['nodes'] if n['id']=='input')['shape']=[4,1,6]
        ref=leaf
        for key in ('inner','outer'):
            models[key]={'id':key,'name':key,'graph':{'nodes':[
                {'id':'input','type':'Input','shape':[4,1,6],'axes':['C','H','W']},
                {'id':'call','type':'Module','ref':ref},{'id':'output','type':'Output'}],
                'edges':[{'from':'input','to':'call'},{'from':'call','to':'output'}]}}
            ref=key
        nodes=[{'id':'input','type':'Input','shape':[4,5,6],'axes':['C','H','W']}];edges=[];sources=[]
        for index in range(30):
            p,c='pool'+str(index),'call'+str(index)
            nodes.extend([{'id':p,'type':'AvgPool2d','kernel_size':[1,1],'stride':[5+index,1]},
                          {'id':c,'type':'Module','ref':ref}]);sources.append(c)
            edges.extend([{'from':'input','to':p},{'from':p,'to':c},{'from':c,'to':'join'}])
        nodes.extend([{'id':'join','type':'Concat','sources':sources,'flatten':True},{'id':'output','type':'Output'}])
        edges.append({'from':'join','to':'output'})
        models['root']={'id':'root','name':'Many runtime contexts','graph':{'nodes':nodes,'edges':edges}}
        spec=specification(models,'root',{'input':[1]})
        with self.assertRaisesRegex(GraphError,'64 distinct Module contexts'):
            shape_contract(models,'root',spec,{'input':[4,1000,6]})
        self.assertEqual(shape_contract(models,'root',spec,{'input':[4,5,6]})['output_shape'],[90])

    def test_memory_boundary_equality_and_explicit_policy_types(self):
        models,mid=spatial();c=validate_model(models,mid)
        with patch('app.runtime_shapes.available_memory',return_value=10**12):n=admission(c,2)['memory']['estimated_bytes']
        with patch('app.runtime_shapes.available_memory',return_value=10*n):self.assertTrue(admission(c,2)['allowed'])
        with patch('app.runtime_shapes.available_memory',return_value=10*n-1),self.assertRaises(GraphError):admission(c,2)
        for kwargs in ({'batch_size':0},{'batch_size':True},{'batch_size':1.5},{'dtype_bytes':2},{'dtype_bytes':True},
                       {'gradients':1},{'constrained':0},{'memory_limit_percent':0},{'memory_limit_percent':True}):
            kwargs={'batch_size':1,**kwargs}
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):admission(c,**kwargs)

    def test_fresh_shape_planning_does_not_import_numerical_gui_or_change_rng(self):
        code="""import json,random,sys
from app.core import default_cnn
from app.runtime_shapes import specification,shape_contract
m=default_cnn(); models={m['id']:m};state=random.getstate()
s=specification(models,m['id'],{'input':[1,2]})
c=shape_contract(models,m['id'],s,{'input':[1,17,19]})
assert random.getstate()==state
assert not any(k.startswith(('torch','numpy','PyQt')) for k in sys.modules)
print(json.dumps({'shape':c['output_shape'],'status':s['status']}))
"""
        out=subprocess.run([sys.executable,'-c',code],cwd=Path(__file__).resolve().parents[1],capture_output=True,text=True,check=True)
        self.assertEqual(json.loads(out.stdout)['status'],'conditional')
