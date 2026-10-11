"""Actual historical native/checked family owners, controls and variable-T fits."""
import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from app.core import Workspace, GraphError
from app.family_records import retain, compile_specialization
from app.family_export import family_source
from app.evidence import digest
from test_temporal_families import family
from test_recurrent_graph_execution import inputs, named, output
from test_recurrent_execution import oracle

torch.set_num_threads(1)


class TemporalFamilyExecutionTests(unittest.TestCase):
    def setUp(self):
        self.w=Workspace();self.addCleanup(self.w.db.close)
        self.addCleanup(lambda:self.w.history.directory.cleanup() if self.w.history.directory else None)

    def pair(self,*,cnn=False,bindings=None):
        doc=family(cnn=cnn,scoped=True);retained=retain(self.w,family=doc,bindings=bindings,materialize=False)
        native=compile_specialization(self.w,retained['specialization'],constrained=False)
        scope={};exec(family_source(doc,constrained=False)['code'],scope)
        exported=scope['ArchitectureFamily'](bindings=bindings);exported.network.load_state_dict(native.state_dict(),strict=True)
        return native,exported,retained,scope

    def test_retained_native_and_checked_family_outputs_all_gradients_and_paths_are_exact(self):
        for cnn in (False,True):
            native,exported,retained,_=self.pair(cnn=cnn)
            self.assertEqual(exported.runtime_specification(),retained['resolved']['runtime_specification'])
            for steps in (1,3,7):
                a=inputs(steps,cnn=cnn,grad=True);b={k:v.detach().clone().requires_grad_(v.requires_grad) for k,v in a.items()}
                x=native(named(native.contract,a));y=exported(named(native.contract,b))
                for key in x:torch.testing.assert_close(x[key],y[key],rtol=0,atol=0)
                aa=torch.autograd.grad(sum(v.square().sum() for v in x.values()),[a['sequence'],a['initial'],*native.parameters()])
                bb=torch.autograd.grad(sum(v.square().sum() for v in y.values()),[b['sequence'],b['initial'],*exported.parameters()])
                for p,q in zip(aa,bb):torch.testing.assert_close(p,q,rtol=0,atol=0)
            schema=exported.semantic_parameter_schema()
            self.assertEqual(set(schema),set(exported.state_dict()))
            self.assertTrue(all(json.loads(s['id'])[:4]==['family','temporal-family','template','sequence-model'] for s in schema.values()))
            self.assertTrue(all('recurrent' in json.loads(s['id']) for s in schema.values()))

    def test_native_compilation_uses_exact_historical_witness_and_writes_no_sql(self):
        r=retain(self.w,family=family(),materialize=False);doc=family();doc['runtime_policy']=None;self.w.put_spec('family',doc)
        before=self.w.db.serialize();model=compile_specialization(self.w,r['specialization'],constrained=False)
        self.assertEqual(model.runtime_specification()['runtime_inputs'],family()['runtime_policy']['inputs'])
        model(named(model.contract,inputs(9)));self.assertEqual(self.w.db.serialize(),before)

    def test_fixed_policy_native_family_does_not_accept_variable_time(self):
        doc=family(dynamic=False);r=retain(self.w,family=doc,materialize=False)
        native=compile_specialization(self.w,r['specialization'],constrained=False)
        scope={};exec(family_source(doc,constrained=False)['code'],scope);exported=scope['ArchitectureFamily']()
        self.assertIsNone(exported.runtime_specification())
        for model in (native,exported):
            with self.assertRaises(ValueError):model(named(native.contract,inputs(7)))

    def test_shared_cells_have_unique_semantic_owner_once_and_can_warm_start_across_baseline_times(self):
        native,a,_,scope=self.pair();b=scope['ArchitectureFamily'](bindings={'T0':9})
        identities=[row['id'] for row in a.semantic_parameter_schema().values()]
        self.assertEqual(identities,[row['id'] for row in b.semantic_parameter_schema().values()])
        transfer=b.warm_start(a.semantic_state(),strict=True);self.assertFalse(transfer['optimizer_resume']);self.assertEqual(len(transfer['loaded']),2)
        data=named(native.contract,inputs(11));x=a(data);y=b(data)
        for key in x:torch.testing.assert_close(x[key],y[key],rtol=0,atol=0)

    def test_checked_family_retains_constrained_policy_and_refuses_actual_unroll_before_packing(self):
        doc=family();scope={};exec(family_source(doc,constrained=True,memory_limit_percent=10.)['code'],scope)
        m=scope['ArchitectureFamily']();runtime=scope['_runtime']();data=named(m.network.contract,inputs(31))
        with patch.object(runtime.runtime_shapes,'available_memory',return_value=1), \
             patch.object(runtime.core,'_pack_inputs',side_effect=AssertionError('No pack')):
            with self.assertRaises(ValueError):m(data)
        self.assertTrue(m.network.layer_0.constrained)
        self.assertEqual(m.provenance()['runtime_temporal'],doc['runtime_policy']['inputs'])

    def test_retained_variable_time_training_keeps_owner_optimizer_identity_and_matches_oracle(self):
        m,exported,_,_=self.pair();p=list(m.parameters());ids=[id(v) for v in p];opt=torch.optim.Adam(p,lr=.02)
        # Teacher and functional recurrence are independent of the graph runtime.
        weight=torch.tensor([[.3,-.2,.15,.1],[-.1,.4,.05,.2]]);bias=torch.tensor([.08,-.12]);cases=[]
        for steps in (3,7):
            data=inputs(steps);data['mask'].fill_(1);data['reset'].zero_();data['lengths'].fill_(steps)
            y,h=oracle(data['sequence'],data['initial'],weight,bias);cases.append((data,y,h))
        def loss(case):
            data,y,h=case;result=m(named(m.contract,data))
            return (output(m.contract,result,'prediction')-y).square().mean()+.2*(output(m.contract,result,'final')-h).square().mean()
        start=float(sum(loss(c).detach() for c in cases)/2)
        for step in range(80):
            opt.zero_grad();loss(cases[step%2]).backward();opt.step();self.assertEqual([id(v) for v in m.parameters()],ids)
        end=float(sum(loss(c).detach() for c in cases)/2);self.assertLess(end,start*.1)
        self.assertEqual([int(v['step']) for v in opt.state.values()],[80,80])
        exported.network.load_state_dict(m.state_dict(),strict=True)
        for data,_,_ in cases:
            a=m(named(m.contract,data));b=exported(named(m.contract,data))
            for key in a:torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)

    def test_fresh_app_free_family_reproduces_retained_weights_and_variable_time(self):
        m,exported,_,scope=self.pair();data=named(m.contract,inputs(7));expected={k:v.detach().tolist() for k,v in exported(data).items()}
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'family.py';path.write_text(family_source(family(scoped=True),constrained=False)['code'])
            weights=Path(td)/'weights.pt';torch.save(exported.state_dict(),weights)
            rows=Path(td)/'rows.json';rows.write_text(json.dumps({k:v.tolist() for k,v in data.items()}))
            probe='''import json,runpy,sys,torch
torch.set_num_threads(1)
s=runpy.run_path(sys.argv[1]);m=s['ArchitectureFamily']();m.load_state_dict(torch.load(sys.argv[2],weights_only=True),strict=True)
rows={k:torch.tensor(v,dtype=torch.float32) for k,v in json.load(open(sys.argv[3])).items()}
print(json.dumps({k:v.detach().tolist() for k,v in m(rows).items()}))
assert not any(k=='app' or k.startswith('app.') for k in sys.modules)
'''
            result=subprocess.run([sys.executable,'-c',probe,str(path),str(weights),str(rows)],cwd=td,capture_output=True,text=True,timeout=20)
            self.assertEqual(result.returncode,0,result.stderr);self.assertEqual(json.loads(result.stdout),expected)

    def test_old_family_source_cannot_silently_select_new_temporal_constructor(self):
        from runtime_spatial_fixtures import family as spatial_family
        scope={};exec(family_source(spatial_family(),constrained=False)['code'],scope)
        with self.assertRaisesRegex(ValueError,'family6'):
            scope['ArchitectureFamily'](runtime_temporal={'input':[0]})
