"""Independent variable-T CPU outputs/gradients, owner reuse and safety oracles."""
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

from app.core import GraphError, compile_model
from app.temporal_shapes import compile_temporal_model
from app.temporal_export import temporal_source
from test_temporal_shapes import temporal_fixture, shapes, nested_fixture
from test_recurrent_graph_execution import inputs, named, output
from test_recurrent_execution import oracle

torch.set_num_threads(1)


def cnn_oracle(x, initial, cw, cb, weight, bias, lengths, mask, reset):
    trajectories, finals = [], []
    for row in range(x.shape[0]):
        h = initial[row]; values = []
        zero = sum(p[:0].sum() for p in (x[row], initial[row], cw, cb, weight, bias))
        for t in range(x.shape[1]):
            if t < int(lengths[row]) and bool(mask[row,t]):
                if bool(reset[row,t]): h = initial[row]
                frame = F.adaptive_avg_pool2d(F.conv2d(x[row,t].unsqueeze(0), cw, cb), (1,1)).flatten()
                h = torch.tanh(F.linear(torch.cat((frame,h)), weight,bias)); values.append(h)
            else: values.append(torch.zeros_like(h)+zero)
        trajectories.append(torch.stack(values)); finals.append(h+zero)
    return torch.stack(trajectories), torch.stack(finals)


class TemporalExecutionTests(unittest.TestCase):
    def make(self, *, cnn=False, constrained=False):
        models, mid, decl = temporal_fixture(cnn=cnn)
        torch.manual_seed(17)
        return compile_model(models, mid, constrained, runtime_temporal=decl), models, mid, decl

    def test_multiple_times_outputs_all_gradients_and_parameter_identity_match_row_oracle(self):
        m, _, _, _ = self.make(); pointers = [p.data_ptr() for p in m.parameters()]; baseline = copy.deepcopy(m.contract)
        for steps in (1,2,5,11):
            data = inputs(steps, grad=True); x,h = data['sequence'], data['initial']
            xx, hh = x.detach().clone().requires_grad_(), h.detach().clone().requires_grad_()
            params = list(m.parameters()); copied = [p.detach().clone().requires_grad_() for p in params]
            y, end = oracle(xx, hh, *copied, lengths=data['lengths'][:,0].long(), mask=data['mask'].bool(), reset=data['reset'].bool())
            result = m(named(m.contract,data)); actual = output(m.contract,result,'prediction'); final = output(m.contract,result,'final')
            torch.testing.assert_close(actual,y); torch.testing.assert_close(final,end)
            aa = torch.autograd.grad(actual.square().sum()+.7*final.square().sum(), [x,h,*params])
            bb = torch.autograd.grad(y.square().sum()+.7*end.square().sum(), [xx,hh,*copied])
            for a,b in zip(aa,bb): torch.testing.assert_close(a,b,rtol=3e-6,atol=2e-6)
            self.assertEqual([p.data_ptr() for p in m.parameters()],pointers); self.assertEqual(m.contract,baseline)

    def test_cnn_frame_composition_all_gradients_at_variable_times_with_nan_padding(self):
        m, _, _, _ = self.make(cnn=True)
        for steps in (1,3,7):
            data = inputs(steps, cnn=True, grad=True)
            with torch.no_grad():
                for row in range(3):
                    for t in range(steps):
                        if t >= int(data['lengths'][row,0]) or not data['mask'][row,t]: data['sequence'][row,t].fill_(float('nan'))
            x,h = data['sequence'],data['initial']; xx,hh = x.detach().clone().requires_grad_(),h.detach().clone().requires_grad_()
            params=list(m.parameters()); copied=[p.detach().clone().requires_grad_() for p in params]
            # Parameter registration follows topological operation order: Conv, Linear.
            cw,cb,w,b = copied
            y,end=cnn_oracle(xx,hh,cw,cb,w,b,data['lengths'][:,0].long(),data['mask'].bool(),data['reset'].bool())
            result=m(named(m.contract,data)); actual=output(m.contract,result,'prediction'); final=output(m.contract,result,'final')
            torch.testing.assert_close(actual,y); torch.testing.assert_close(final,end)
            aa=torch.autograd.grad(actual.square().sum()+final.square().sum(),[x,h,*params])
            bb=torch.autograd.grad(y.square().sum()+end.square().sum(),[xx,hh,*copied])
            for a,b in zip(aa,bb): torch.testing.assert_close(a,b,rtol=5e-6,atol=3e-6)

    def test_all_masked_nan_sequences_produce_zero_outputs_finite_zero_gradients(self):
        m, _, _, _ = self.make()
        for steps in (1,9):
            data=inputs(steps,grad=True); data['mask'].zero_(); data['lengths'].zero_()
            with torch.no_grad(): data['sequence'].fill_(float('nan'))
            result=m(named(m.contract,data)); y=output(m.contract,result,'prediction'); final=output(m.contract,result,'final')
            self.assertEqual(float(y.detach().abs().sum()),0.); torch.testing.assert_close(final,data['initial'],rtol=0,atol=0)
            gradients=torch.autograd.grad(y.sum(),[data['sequence'],data['initial'],*m.parameters()])
            for g in gradients: self.assertTrue(torch.isfinite(g).all()); self.assertEqual(float(g.abs().sum()),0.)

    def test_calls_never_carry_hidden_state_and_returned_tensor_packing_uses_actual_time(self):
        m, _, _, _ = self.make(); data=inputs(7); x=named(m.contract,data)
        first=m(x); m(named(m.contract,inputs(2))); again=m(x)
        for key in first: torch.testing.assert_close(first[key],again[key],rtol=0,atol=0)
        self.assertEqual(list(m.forward_tensor(x).shape),[3,16])
        values=m.intermediate_values(x)
        self.assertEqual(list(values['scan']['trajectory'].shape),[3,7,2]); self.assertEqual(list(values['scan']['hidden_final'].shape),[3,2])

    def test_optimizer_and_shared_cell_owners_survive_time_changes_and_rejected_call(self):
        models,mid,decl=temporal_fixture(); g=models[mid]['graph']; second=copy.deepcopy(g['nodes'][2]); second.update(id='scan2',share_with='scan')
        g['nodes'].extend([second,{'id':'second','type':'Output'}]); g['edges'].extend(dict(e,to='scan2') for e in list(g['edges']) if e['to']=='scan')
        g['edges'].append({'from':'scan2','from_port':'trajectory','to':'second'})
        m=compile_temporal_model(models,mid,decl,constrained=False); self.assertEqual(len(list(m.parameters())),2)
        opt=torch.optim.Adam(m.parameters(),lr=.01); owners=[id(p) for p in m.parameters()]
        for steps in (2,7,3):
            opt.zero_grad(); result=m(named(m.contract,inputs(steps))); y=output(m.contract,result,'prediction')
            torch.testing.assert_close(y,output(m.contract,result,'second'),rtol=0,atol=0)
            y.square().mean().backward(); opt.step(); self.assertEqual([id(p) for p in m.parameters()],owners)
        self.assertEqual([int(s['step']) for s in opt.state.values()],[3,3])
        before=copy.deepcopy(m.state_dict()); data=inputs(6);data['mask']=torch.zeros(3,4)
        with self.assertRaises(GraphError): m(named(m.contract,data))
        for k,v in before.items(): torch.testing.assert_close(m.state_dict()[k],v,rtol=0,atol=0)

    def test_current_ram_refusal_precedes_packing_control_cast_or_transition(self):
        m,models,mid,_=self.make(constrained=True); data=named(m.contract,inputs(37))
        with patch('app.runtime_shapes.available_memory',return_value=1), \
             patch('app.core._pack_inputs',side_effect=AssertionError('Do not pack')), \
             patch('app.recurrent_nodes._control',side_effect=AssertionError('Do not convert')), \
             patch.object(m.layer_0.transition,'forward',side_effect=AssertionError('Do not run')):
            with self.assertRaisesRegex(GraphError,'runtime temporal'): m(data)
            with self.assertRaisesRegex(GraphError,'runtime temporal'): m.preview_shapes(shapes(models,mid,37),batch_size=3)
        self.assertTrue(m.layer_0.constrained); self.assertEqual(m.layer_0.memory_limit_percent,10.)

    def test_invalid_shapes_types_devices_layout_and_flat_packing_fail_before_work(self):
        m, _, _, _ = self.make(); good=named(m.contract,inputs(7)); key=m.contract['input_groups'][0]['name']
        candidates=[torch.zeros(3,30),{},dict(good,unknown=torch.zeros(3,1))]
        for bad in (torch.zeros(3,7,3),torch.zeros(3,7),torch.zeros(0,7,2),torch.zeros(3,0,2),
                    torch.zeros(3,7,2,dtype=torch.float64),torch.zeros(3,7,2,dtype=torch.int32),
                    torch.empty(3,7,2,device='meta'),torch.zeros(3,7,2).to_sparse()): candidates.append(dict(good,**{key:bad}))
        for bad in candidates:
            with self.subTest(kind=str(type(bad))),patch('app.core._pack_inputs',side_effect=AssertionError('Do not pack')):
                with self.assertRaises(GraphError): m(bad)

    def test_preview_rejects_nonboolean_gradient_flags_in_both_training_modes(self):
        m,models,mid,_=self.make()
        for training in (True,False):
            m.train(training)
            for value in (0,1,None,'yes',[],{}):
                with self.subTest(training=training,value=value),self.assertRaisesRegex(GraphError,'gradients policy.*boolean'):
                    m.preview_shapes(shapes(models,mid,7),gradients=value)
        a=m.preview_shapes(shapes(models,mid,7),gradients=False)['safety']['memory']
        b=m.preview_shapes(shapes(models,mid,7),gradients=True)['safety']['memory']
        self.assertEqual(a['operation'],'runtime-temporal-inference');self.assertEqual(b['operation'],'runtime-temporal-training')
        self.assertGreater(b['estimated_bytes'],a['estimated_bytes'])

    def test_initial_runtime_shapes_admit_actual_case_without_allocating_baseline_unroll(self):
        models,mid,decl=temporal_fixture(steps=10_000_000,controls=False)
        actual={'sequence':[2,2],'initial':[2]}
        with patch('app.core.available_memory',return_value=5*1024**3), \
             patch('app.runtime_shapes.available_memory',return_value=5*1024**3), \
             patch('app.recurrent.available_memory',return_value=5*1024**3):
            m=compile_temporal_model(models,mid,decl,initial_shapes=actual,constrained=True)
            self.assertEqual(sum(p.numel() for p in m.parameters()),10)
            result=m(named(m.contract,inputs(2)))
            self.assertEqual(list(output(m.contract,result,'prediction').shape),[3,2,2])
        with patch('app.core.compile_model',side_effect=AssertionError('Do not construct')):
            with self.assertRaises(GraphError):
                compile_temporal_model(models,mid,decl,initial_shapes=dict(actual,initial=[3]),constrained=False)

    def test_probabilistic_controls_noninteger_lengths_and_time_mismatch_are_refused(self):
        m, _, _, _=self.make()
        for role,value in (('mask',torch.full((3,7),.5)),('reset',torch.full((3,7),float('nan'))),
                           ('lengths',torch.tensor([[1.5],[7.],[7.]])),('lengths',torch.tensor([[8.],[7.],[7.]])),
                           ('mask',torch.ones(3,6))):
            data=inputs(7);data[role]=value
            with self.subTest(role=role),patch.object(m.layer_0.transition,'forward',side_effect=AssertionError('No transition')):
                with self.assertRaises(ValueError):m(named(m.contract,data))

    def test_source_declaration_spec_and_preview_are_private_copies(self):
        m,models,mid,decl=self.make(); saved=copy.deepcopy(m.contract)
        models['transition']['graph']['nodes'][3]['features']=999;decl['sequence'].clear()
        spec=m.runtime_specification();spec['runtime_inputs']['sequence'].clear()
        self.assertEqual(m.runtime_specification()['runtime_inputs']['sequence'],[0])
        m(named(m.contract,inputs(7)));self.assertEqual(m.contract,saved)
        # inputs() deliberately seeds Torch; previews themselves may not do so.
        state=torch.get_rng_state().clone(); groups=m.runtime_specification()['baseline_input_groups']
        actual={g['id']:[7,*g['shape'][1:]] if g.get('axes',[None])[0]=='T' else g['shape'] for g in groups}
        view=m.preview_shapes(actual);view['contract']['shapes']['prediction']['output_shape'][0]=999
        self.assertEqual(m.preview_shapes(actual)['contract']['shapes']['prediction']['output_shape'],[7,2])
        torch.testing.assert_close(torch.get_rng_state(),state,rtol=0,atol=0)

    def test_nested_unroll_outputs_and_full_bptt_match_independent_two_level_recurrence(self):
        models,mid,decl=nested_fixture(); m=compile_temporal_model(models,mid,decl,constrained=False)
        owners=[id(p) for p in m.parameters()]
        for steps in (1,3,7):
            data=inputs(steps,grad=True)
            data['sequence']=torch.linspace(-.8,.7,3*steps*2*2).reshape(3,steps,2,2).requires_grad_()
            x,h=data['sequence'],data['initial'];xx=x.detach().clone().requires_grad_();hh=h.detach().clone().requires_grad_()
            p=list(m.parameters());w,b=[v.detach().clone().requires_grad_() for v in p]; ys=[];ends=[]
            for row in range(3):
                state=hh[row]; trajectory=[]
                zero=sum(v[:0].sum() for v in (xx[row],hh[row],w,b))
                for t in range(steps):
                    if t<int(data['lengths'][row,0]) and bool(data['mask'][row,t]):
                        if bool(data['reset'][row,t]):state=hh[row]
                        for k in range(2):state=torch.tanh(F.linear(torch.cat((xx[row,t,k],state)),w,b))
                        trajectory.append(state)
                    else:trajectory.append(torch.zeros_like(state)+zero)
                ys.append(torch.stack(trajectory));ends.append(state+zero)
            y,end=torch.stack(ys),torch.stack(ends); result=m(named(m.contract,data))
            actual=output(m.contract,result,'prediction');final=output(m.contract,result,'final')
            torch.testing.assert_close(actual,y);torch.testing.assert_close(final,end)
            aa=torch.autograd.grad(actual.square().sum()+final.sum(),[x,h,*p]);bb=torch.autograd.grad(y.square().sum()+end.sum(),[xx,hh,w,b])
            for a,b in zip(aa,bb):torch.testing.assert_close(a,b,rtol=4e-6,atol=3e-6)
            self.assertEqual([id(v) for v in m.parameters()],owners)

    def test_captured_runtime_outputs_and_all_gradients_are_exact_at_changing_times(self):
        for cnn in (False,True):
            native,models,mid,decl=self.make(cnn=cnn);namespace={}
            exec(temporal_source(models,mid,decl,constrained=False)['code'],namespace)
            exported=namespace['build_model']();exported.load_state_dict(native.state_dict(),strict=True)
            self.assertEqual(set(exported.state_dict()),set(native.state_dict()))
            for steps in (1,5,9):
                a=inputs(steps,cnn=cnn,grad=True);b={k:v.detach().clone().requires_grad_(v.requires_grad) for k,v in a.items()}
                expected=native(named(native.contract,a));actual=exported(named(exported.contract,b))
                for key in expected:torch.testing.assert_close(actual[key],expected[key],rtol=0,atol=0)
                aa=torch.autograd.grad(sum(v.square().sum() for v in expected.values()),[a['sequence'],a['initial'],*native.parameters()])
                bb=torch.autograd.grad(sum(v.square().sum() for v in actual.values()),[b['sequence'],b['initial'],*exported.parameters()])
                for p,q in zip(aa,bb):torch.testing.assert_close(p,q,rtol=0,atol=0)

    def test_fresh_app_free_export_import_is_torch_free_then_predictions_match_checkpoint(self):
        native,models,mid,decl=self.make(); data=named(native.contract,inputs(7))
        expected={k:v.detach().tolist() for k,v in native(data).items()}
        with tempfile.TemporaryDirectory() as directory:
            export=Path(directory)/'temporal.py';export.write_text(temporal_source(models,mid,decl,constrained=False)['code'])
            checkpoint=Path(directory)/'weights.pt';torch.save(native.state_dict(),checkpoint)
            rows=Path(directory)/'inputs.json';rows.write_text(json.dumps({k:v.tolist() for k,v in data.items()}))
            probe='''import json,runpy,sys
s=runpy.run_path(sys.argv[1]);s['runtime_specification']()
assert not any(k.startswith(('torch','numpy','PyQt','app')) for k in sys.modules)
import torch
torch.set_num_threads(1)
m=s['build_model']();m.load_state_dict(torch.load(sys.argv[2],weights_only=True),strict=True)
data={k:torch.tensor(v,dtype=torch.float32) for k,v in json.load(open(sys.argv[3])).items()}
print(json.dumps({k:v.detach().tolist() for k,v in m(data).items()}))
assert not any(k=='app' or k.startswith('app.') for k in sys.modules)
'''
            result=subprocess.run([sys.executable,'-c',probe,str(export),str(checkpoint),str(rows)],cwd=directory,
                                  capture_output=True,text=True,timeout=20)
            self.assertEqual(result.returncode,0,result.stderr);self.assertEqual(json.loads(result.stdout),expected)
