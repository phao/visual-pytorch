"""Variable-T LSTM outputs/gradients and optimizer ownership, independent gates."""
import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from app.core import GraphError, compile_model, default_model, validate_model
from app.temporal_shapes_v2 import compile_temporal_model
from app.temporal_export_v2 import temporal_source
from test_temporal_shapes_v2 import runtime_fixture, shapes, nested_fixture, module_fixture
from test_lstm_execution import oracle, data, named, returned

torch.set_num_threads(1)


class TemporalV2ExecutionTests(unittest.TestCase):
    def make(self, *, constrained=False, **options):
        models,mid,decl = runtime_fixture(**options)
        torch.manual_seed(41)
        model = compile_model(models,mid,constrained,runtime_temporal_v2=decl)
        return model,models,mid,decl

    def check_oracle(self, model, models, mid, steps, *, empty=False):
        actual_contract = model.preview_shapes(shapes(models,mid,steps),batch_size=4)['contract']
        controlled = 'lengths' in {g['id'] for g in actual_contract['input_groups']}
        values = data(actual_contract,grad=True,nan_padding=controlled,
                      lengths=[0]*4 if empty else None)
        keys = ('sequence','hidden','cell'); plan = actual_contract['shapes']['lstm']['lstm']; kw = plan['kwargs']
        independent = {key:values[key].detach().clone().requires_grad_() for key in keys}
        weights = {key:p.detach().clone().requires_grad_() for key,p in model.layer_0.named_parameters()}
        lengths = values['lengths'][:,0].long() if 'lengths' in values else torch.full((4,),steps)
        expected = oracle(*(independent[key] for key in keys),weights,lengths=lengths,
                          layers=kw['num_layers'],bidirectional=kw['bidirectional'],projection=kw['proj_size'])
        actual = returned(actual_contract,model(named(actual_contract,values)))
        for key in keys:
            torch.testing.assert_close(actual[key],expected[key],rtol=4e-6,atol=2e-6)
        loss = lambda result:sum((i+.5)*result[key].square().sum() for i,key in enumerate(keys))
        aa = torch.autograd.grad(loss(actual),[*(values[key] for key in keys),*model.layer_0.parameters()])
        bb = torch.autograd.grad(loss(expected),[*(independent[key] for key in keys),*weights.values()])
        for a,b in zip(aa,bb):
            torch.testing.assert_close(a,b,rtol=9e-6,atol=4e-6)

    def test_16_constructor_combinations_native_and_export_at_three_times_all_gradients(self):
        for layers in (1,2):
            for directions in (False,True):
                for projection in (0,2):
                    for bias in (False,True):
                        model,models,mid,decl = self.make(hidden=3,projection=projection,layers=layers,
                                                         bidirectional=directions,bias=bias,width=2)
                        scope = {}; exec(temporal_source(models,mid,decl,constrained=False)['code'],scope)
                        exported = scope['build_model'](); exported.load_state_dict(model.state_dict(),strict=True)
                        for current in (model,exported):
                            owners = [id(p) for p in current.parameters()]; baseline = copy.deepcopy(current.contract)
                            for steps in (1,3,7):
                                with self.subTest(layers=layers,directions=directions,projection=projection,
                                                  bias=bias,steps=steps,route=type(current).__module__):
                                    self.check_oracle(current,models,mid,steps)
                            self.assertEqual([id(p) for p in current.parameters()],owners)
                            self.assertEqual(current.contract,baseline)

    def test_no_lengths_means_every_frame_active_not_a_hidden_default_state(self):
        model,models,mid,_ = self.make(hidden=3,projection=0,layers=1,lengths=False)
        for t in (1,6):
            self.check_oracle(model,models,mid,t)

    def test_all_zero_lengths_nan_padding_preserve_states_and_zero_input_parameter_gradients(self):
        model,models,mid,_ = self.make(hidden=3,projection=2,layers=2)
        for t in (1,9):
            self.check_oracle(model,models,mid,t,empty=True)
            c = model.preview_shapes(shapes(models,mid,t),batch_size=4)['contract']
            values = data(c,grad=True,lengths=[0]*4,nan_padding=True)
            out = returned(c,model(named(c,values)))
            self.assertEqual(float(out['sequence'].detach().abs().sum()),0.)
            for role in ('hidden','cell'):
                torch.testing.assert_close(out[role],values[role],rtol=0,atol=0)
            gradients = torch.autograd.grad(out['sequence'].sum(),
                                           [values[role] for role in ('sequence','hidden','cell')]+list(model.parameters()))
            for grad in gradients:
                self.assertTrue(torch.isfinite(grad).all()); self.assertEqual(float(grad.abs().sum()),0.)

    def test_no_state_carry_and_actual_time_forward_tensor_packing(self):
        model,models,mid,_ = self.make(hidden=3,projection=2,layers=1)
        c = model.preview_shapes(shapes(models,mid,7),batch_size=4)['contract']; x = named(c,data(c))
        first = model(x)
        c2 = model.preview_shapes(shapes(models,mid,2),batch_size=4)['contract']; model(named(c2,data(c2)))
        again = model(x)
        for key in first:
            torch.testing.assert_close(first[key],again[key],rtol=0,atol=0)
        self.assertEqual(list(model.forward_tensor(x).shape),[4,c['outputs']])
        self.assertEqual(list(model.intermediate_values(x)['lstm']['sequence'].shape),[4,7,4])

    def test_35_variable_time_adam_updates_match_independent_teacher_and_student(self):
        model,models,mid,_ = self.make(hidden=3,projection=2,layers=1,width=1,bidirectional=False)
        keys = ('sequence','hidden','cell')
        params = dict(model.layer_0.named_parameters())
        reference = {key:p.detach().clone().requires_grad_() for key,p in params.items()}
        teacher = {key:.4*torch.sin(torch.arange(p.numel(),dtype=torch.float32).reshape(p.shape)+i)
                   for i,(key,p) in enumerate(params.items())}
        aa = torch.optim.Adam(model.parameters(),lr=.025); bb = torch.optim.Adam(reference.values(),lr=.025)
        owners = [id(p) for p in model.parameters()]
        def expected(values, weights):
            return oracle(*(values[key] for key in keys),weights,lengths=values['lengths'][:,0].long(),
                          layers=1,bidirectional=False,projection=2)
        def loss(output,target):
            return sum((output[key]-target[key]).square().mean() for key in keys)
        evaluation = model.preview_shapes(shapes(models,mid,7),batch_size=4)['contract']
        evaluation_data = data(evaluation)
        target_eval = expected(evaluation_data,teacher)
        initial = float(loss(returned(evaluation,model(named(evaluation,evaluation_data))),target_eval).detach())
        for step in range(35):
            t = (1,3,7)[step%3]; c = model.preview_shapes(shapes(models,mid,t),batch_size=4)['contract']
            values = data(c); target = expected(values,teacher)
            aa.zero_grad(); bb.zero_grad()
            actual = returned(c,model(named(c,values))); independent = expected(values,reference)
            la,lb = loss(actual,target),loss(independent,target)
            torch.testing.assert_close(la,lb,rtol=1e-5,atol=3e-6)
            la.backward(); lb.backward(); aa.step(); bb.step()
            for key,p in params.items():
                torch.testing.assert_close(p,reference[key],rtol=2e-5,atol=5e-6)
            self.assertEqual([id(p) for p in model.parameters()],owners)
        final = float(loss(returned(evaluation,model(named(evaluation,evaluation_data))),target_eval).detach())
        self.assertLess(final,initial)
        self.assertEqual({int(s['step']) for s in aa.state.values()},{35})

    def test_shared_lstm_aliases_update_each_owner_once_and_sum_both_gradients(self):
        model,models,mid,decl = self.make(hidden=3,projection=2,layers=1)
        graph = models[mid]['graph']; other = copy.deepcopy(graph['nodes'][3]); other.update(id='shared',share_with='lstm')
        graph['nodes'].append(other); graph['nodes'].append({'id':'shared-out','type':'Output','label':'Shared output'})
        graph['edges'] += [dict(e,to='shared') for e in list(graph['edges']) if e['to']=='lstm']
        graph['edges'].append({'from':'shared','from_port':'sequence','to':'shared-out'})
        model = compile_temporal_model(models,mid,decl,constrained=False)
        self.assertEqual(sum(p.numel() for p in model.parameters()),model.contract['parameters'])
        self.assertFalse(any(key.startswith('layer_1.') for key in model.state_dict()))
        opt = torch.optim.SGD(model.parameters(),lr=.01)
        for t in (2,8,3):
            c = model.preview_shapes(shapes(models,mid,t),batch_size=4)['contract']; values = named(c,data(c))
            result = model(values); first = result['Returned sequence']; second = result['Shared output']
            torch.testing.assert_close(first,second,rtol=0,atol=0)
            both = torch.autograd.grad(first.sum()+second.sum(),model.parameters(),retain_graph=True)
            single = torch.autograd.grad(first.sum(),model.parameters())
            for a,b in zip(both,single):
                torch.testing.assert_close(a,2*b)
            opt.zero_grad(); sum(v.square().mean() for v in model(values).values()).backward(); opt.step()

    def test_current_ram_refusal_precedes_packing_lstm_and_control_adapters(self):
        with patch('app.runtime_shapes.available_memory',return_value=10**12):
            model,models,mid,_ = self.make(constrained=True)
            c = model.preview_shapes(shapes(models,mid,17),batch_size=4)['contract']
        values = named(c,data(c))
        saved = copy.deepcopy(model.state_dict())
        with patch('app.runtime_shapes.available_memory',return_value=1),\
             patch('app.core._pack_inputs',side_effect=AssertionError('No packing')),\
             patch('app.lstm_nodes.graph_forward',side_effect=AssertionError('No conversion or LSTM')):
            with self.assertRaisesRegex(GraphError,'whole current-B/T graph'): model(values)
        self.assertTrue(model.layer_0.constrained)
        for k,v in saved.items():
            torch.testing.assert_close(model.state_dict()[k],v,rtol=0,atol=0)

    def test_invalid_shapes_dtype_device_layout_and_lengths_refuse_without_lstm(self):
        model,models,mid,_ = self.make(); c = model.preview_shapes(shapes(models,mid,7),batch_size=4)['contract']
        good = named(c,data(c)); name = c['input_groups'][0]['name']
        bads = [torch.zeros(4,20),{},dict(good,extra=torch.zeros(4,1))]
        for tensor in (torch.zeros(4,7,4),torch.zeros(4,7),torch.zeros(0,7,3),torch.zeros(4,0,3),
                       torch.zeros(4,7,3,dtype=torch.float64),torch.zeros(4,7,3,dtype=torch.int64),
                       torch.empty(4,7,3,device='meta'),torch.zeros(4,7,3).to_sparse()):
            bads.append(dict(good,**{name:tensor}))
        for bad in bads:
            with self.subTest(kind=type(bad).__name__),patch('app.core._pack_inputs',side_effect=AssertionError('No packing')):
                with self.assertRaises(GraphError): model(bad)
        for value in (1.5,8.,-1.,float('nan'),float('inf')):
            values = data(c); values['lengths'].fill_(value)
            with self.subTest(length=value),patch.object(model.layer_0,'forward',side_effect=AssertionError('No LSTM')):
                with self.assertRaisesRegex(ValueError,'finite integer lengths'): model(named(c,values))
        for role in ('hidden','cell'):
            values = data(c); values[role].fill_(float('nan'))
            with self.subTest(role=role),patch.object(model.layer_0,'forward',side_effect=AssertionError('No LSTM')):
                with self.assertRaisesRegex(ValueError,'initial states must be finite'): model(named(c,values))

    def test_nested_variable_outer_time_full_bptt_matches_two_level_gate_oracle(self):
        models,mid,decl = nested_fixture(); model = compile_temporal_model(models,mid,decl,constrained=False)
        layer = model.layer_0.transition.layer_0
        for t in (1,3,7):
            c = model.preview_shapes(shapes(models,mid,t),batch_size=2)['contract']
            torch.manual_seed(21)
            values = {g['id']:torch.randn(2,*g['shape'],requires_grad=True) for g in c['input_groups']}
            independent = {key:p.detach().clone().requires_grad_() for key,p in values.items()}
            weights = {key:p.detach().clone().requires_grad_() for key,p in layer.named_parameters()}
            h,cell = independent['hidden'],independent['cell']; trajectory = []
            for time in range(t):
                result = oracle(independent['sequence'][:,time],h,cell,weights,lengths=torch.full((2,),2),
                                layers=1,bidirectional=True,projection=2)
                h,cell = result['hidden'],result['cell']; trajectory.append(h)
            expected = {'trajectory':torch.stack(trajectory,1),'hidden':h,'cell':cell}
            result = model(named(c,values))
            actual = {key:result[next(g['name'] for g in c['output_groups'] if g['id']=='out_'+key)]
                      for key in expected}
            for key in expected:
                torch.testing.assert_close(actual[key],expected[key],rtol=5e-6,atol=2e-6)
            aa = torch.autograd.grad(sum(v.square().sum() for v in actual.values()),[*values.values(),*layer.parameters()])
            bb = torch.autograd.grad(sum(v.square().sum() for v in expected.values()),[*independent.values(),*weights.values()])
            for a,b in zip(aa,bb):
                torch.testing.assert_close(a,b,rtol=2e-5,atol=5e-6)

    def test_initial_shapes_whole_graph_admission_supports_small_actual_case_huge_baseline(self):
        models,mid,decl = runtime_fixture(steps=10_000_000,hidden=3,projection=2,layers=1,lengths=False)
        actual = shapes(models,mid,2)
        with patch('app.runtime_shapes.available_memory',return_value=5*1024**3),\
             patch('app.lstm_nodes.available_memory',return_value=5*1024**3):
            model = compile_temporal_model(models,mid,decl,initial_shapes=actual)
            c = model.preview_shapes(actual,batch_size=4)['contract']
            values = named(c,data(c))
            self.assertEqual(list(model(values)['Returned sequence'].shape),[4,2,4])
        with patch('app.core.compile_model',side_effect=AssertionError('No construction')):
            with self.assertRaises(GraphError):
                compile_temporal_model(models,mid,decl,initial_shapes=dict(actual,hidden=[2,3]),constrained=False)

    def test_fixed_module_lstm_and_variable_root_coexist_in_native_and_captured_runtime(self):
        models,mid,decl = module_fixture()
        model = compile_temporal_model(models,mid,decl,constrained=False)
        scope = {}; exec(temporal_source(models,mid,decl,constrained=False)['code'],scope)
        exported = scope['build_model'](); exported.load_state_dict(model.state_dict(),strict=True)
        for t in (1,7):
            c = model.preview_shapes(shapes(models,mid,t),batch_size=2)['contract']
            values = {g['name']:torch.randn(2,*g['shape']) for g in c['input_groups']}
            actual,expected = exported(values),model(values)
            for key in expected:
                torch.testing.assert_close(actual[key],expected[key],rtol=0,atol=0)
        with patch('app.runtime_shapes.available_memory',return_value=10**12):
            guarded = compile_temporal_model(models,mid,decl)
            self.assertTrue(guarded.layer_0.layer_0.constrained)

    def test_preview_and_frozen_sources_are_private_and_do_not_change_rng(self):
        model,models,mid,decl = self.make(); baseline = copy.deepcopy(model.contract)
        actual = shapes(models,mid,7); rng = torch.get_rng_state().clone()
        first = model.preview_shapes(actual); first['contract']['shapes']['out_sequence']['output_shape'][0] = 99
        self.assertEqual(model.preview_shapes(actual)['contract']['shapes']['out_sequence']['output_shape'][0],7)
        torch.testing.assert_close(torch.get_rng_state(),rng,rtol=0,atol=0)
        models[mid]['graph']['nodes'][3]['hidden_size'] = 99; decl['sequence'].clear()
        private = model.runtime_specification(); private['runtime_inputs']['sequence'].clear()
        self.assertEqual(model.runtime_specification()['runtime_inputs']['sequence'],[0])
        self.assertEqual(model.contract,baseline)
        for training in (False,True):
            model.train(training)
            for invalid in (0,1,None,'yes'):
                with self.assertRaisesRegex(GraphError,'gradients policy must be boolean'):
                    model.preview_shapes(actual,gradients=invalid)

    def test_fresh_export_is_app_free_torch_free_until_build_and_checkpoint_predictions_exact(self):
        model,models,mid,decl = self.make(hidden=3,projection=2,layers=1)
        expected = {}; inputs = {}
        for t in (1,4,9):
            c = model.preview_shapes(shapes(models,mid,t),batch_size=4)['contract']; x = named(c,data(c,nan_padding=False))
            inputs[str(t)] = {k:v.tolist() for k,v in x.items()}
            expected[str(t)] = {k:v.detach().tolist() for k,v in model(x).items()}
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory); source = folder/'export.py'; checkpoint = folder/'weights.pt'; rows = folder/'input.json'
            source.write_text(temporal_source(models,mid,decl,constrained=False)['code'])
            torch.save(model.state_dict(),checkpoint); rows.write_text(json.dumps(inputs))
            code = """
import json,runpy,sys
s=runpy.run_path(sys.argv[1]);s['runtime_specification']()
assert not any(k.startswith(('torch','numpy','PyQt','app')) for k in sys.modules)
import torch
torch.set_num_threads(1)
m=s['build_model']();m.load_state_dict(torch.load(sys.argv[2],weights_only=True),strict=True)
rows=json.load(open(sys.argv[3]));actual={}
for t,row in rows.items():
    x={k:torch.tensor(v,dtype=torch.float32) for k,v in row.items()}
    actual[t]={k:v.detach().tolist() for k,v in m(x).items()}
assert not any(k=='app' or k.startswith('app.') for k in sys.modules)
print(json.dumps(actual))
"""
            result = subprocess.run([sys.executable,'-c',code,str(source),str(checkpoint),str(rows)],
                                    cwd=directory,capture_output=True,text=True,timeout=30)
            self.assertEqual(result.returncode,0,result.stderr); self.assertEqual(json.loads(result.stdout),expected)


if __name__ == '__main__':
    unittest.main()
