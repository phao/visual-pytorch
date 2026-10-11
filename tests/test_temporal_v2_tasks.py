"""Closed TaskBundle4 provenance, actual shapes/RAM and immutable old routes."""
import copy
import json
import random
import subprocess
import sys
import unittest
from unittest.mock import patch

from app.core import GraphError
from app.datasets import make_split
from app.evidence import digest
from app.runtime_tasks import proof, input_policy, record_shapes
from app.task_execution import validate_bundle, run_identity, resolve_run, fitted_inputs
from runtime_v2_task_fixture import bundle, rebuild


class TemporalV2TaskTests(unittest.TestCase):
    def test_actual_T_preserves_exact_baseline_and_family_witness(self):
        a=bundle(steps=3);b=bundle(steps=7)
        self.assertEqual(a['version'],4);self.assertEqual(a['model'],b['model']);self.assertEqual(a['runtime'],b['runtime'])
        for value,t in ((a,3),(b,7)):
            c=validate_bundle(value);self.assertEqual(c['shapes']['lstm']['ports']['sequence'],[t,4])
            self.assertEqual(value['runtime']['specialization']['version'],7)
            self.assertEqual(run_identity(value)['compiler'],'native-runtime-temporal-v2')
            self.assertEqual(run_identity(value)['version'],4)
        self.assertNotEqual(run_identity(a),run_identity(b))

    def test_new_bundle_version_does_not_widen_old_contracts(self):
        b=bundle()
        for version in (1,2,3,True,5):
            bad=copy.deepcopy(b);bad['version']=version
            with self.subTest(version=version),self.assertRaises(GraphError):validate_bundle(bad)
        for policy in ('runtime-temporal-v1','runtime-spatial-v1','unknown'):
            with self.subTest(policy=policy),self.assertRaises(GraphError):
                proof(b['runtime']['family'],b['runtime']['specialization'],b['model'],execution=policy)

    def test_exact_runtime_revision_kind_axes_hash_context_and_model_required(self):
        b=bundle()
        for mutation in ('axis','kind','hash','version','binding','family','model','extra','missing'):
            bad=copy.deepcopy(b);runtime=bad['runtime'];s=runtime['specialization']
            if mutation=='axis':s['runtime']['inputs']['sequence']=[1]
            elif mutation=='kind':s['runtime']['kind']='runtime-temporal-v1'
            elif mutation=='hash':s['runtime']['specification_sha256']='0'*64
            elif mutation=='version':s['runtime']['version']=2
            elif mutation=='binding':s['bindings']['time']=9
            elif mutation=='family':runtime['family']['name']='Forged head'
            elif mutation=='model':bad['model']['models'][bad['model']['root']]['graph']['nodes'][0]['shape'][0]=9
            elif mutation=='extra':runtime['infer']=True
            else:runtime.pop('specialization')
            with self.subTest(mutation=mutation),self.assertRaises(GraphError):validate_bundle(bad)

    def test_constructor_state_and_output_widths_do_not_follow_time(self):
        b=bundle();d=copy.deepcopy(b['dataset'])
        for key in ('hidden','cell','target_out_sequence'):
            bad=copy.deepcopy(d);f=next(f for f in bad['fields'] if f['id']==key);f['shape'][-1]+=1
            def zeros(shape):return [zeros(shape[1:]) for _ in range(shape[0])] if shape else 0.
            for r in bad['records']:r['values'][key]=zeros(f['shape'])
            with self.subTest(field=key),self.assertRaises(GraphError):rebuild(b,dataset=bad)

    def test_preparation_and_policy_are_private_rng_neutral_and_no_heavy_imports(self):
        b=bundle();before=copy.deepcopy(b);rng=random.getstate()
        prepared,m=resolve_run(b,constrained=False,available_bytes=1)
        p=input_policy(b);p['runtime_temporal']['sequence'][0]=99
        self.assertEqual(b,before);self.assertEqual(random.getstate(),rng)
        self.assertEqual(prepared['contract']['shapes']['lstm']['ports']['sequence'],[3,4])
        self.assertEqual(m['actual_batch_size'],8)
        run=subprocess.run([sys.executable,'-c',
            "import sys;from runtime_v2_task_fixture import bundle;from app.task_execution import resolve_run;resolve_run(bundle(mode='pad'),constrained=False);assert not any(k in sys.modules for k in ('torch','numpy','PyQt6'))"],
            capture_output=True,text=True,timeout=10)
        self.assertEqual(run.returncode,0,run.stderr)

    def test_whole_case_memory_charges_gate_calls_and_keeps_task_data_evidence(self):
        values=[resolve_run(bundle(steps=t),constrained=False,available_bytes=1)[1] for t in (1,3,9)]
        for m in values:
            c=m['components'];self.assertGreater(c['lstm_unrolls'],0)
            self.assertIn('recurrent_unrolls',c)
            for key in ('typed_preparation','metric_history','parameters_optimizer_evidence','dataset_and_copies'):
                self.assertGreater(c[key],0)
            self.assertEqual(m['estimated_bytes'],sum(c.values()))
        self.assertLess(values[0]['components']['lstm_unrolls'],values[-1]['components']['lstm_unrolls'])
        b=bundle();m=values[1];budget=m['estimated_bytes']
        with self.assertRaisesRegex(GraphError,'RAM'):resolve_run(b,available_bytes=budget*10-10)
        _,accepted=resolve_run(b,available_bytes=budget*10)
        self.assertEqual(accepted['estimated_bytes'],budget)
        with patch('app.task_execution.available_memory',return_value=None),self.assertRaisesRegex(GraphError,'RAM'):
            resolve_run(b)

    def test_dynamic_records_allow_only_rectangular_T_and_fixed_state_axes(self):
        b=bundle();spec={'dataset':b['dataset'],'task':b['task'],'runtime_policy':input_policy(b)}
        row=copy.deepcopy(b['dataset']['records'][2]['values']);row['sequence']=[[.1,.2]]*11;row['lengths']=[11.]
        self.assertEqual(record_shapes(spec,[row])['sequence'],[11,2])
        for frames in ([],[[1.]]*11,[[[1.,2.]]]*11,[[1.,2.],[1.]]):
            bad={**row,'sequence':frames}
            with self.subTest(frames=frames),self.assertRaises(GraphError):record_shapes(spec,[bad])
        with self.assertRaises(GraphError):record_shapes(spec,[row,{**row,'sequence':[[.1,.2]]*3}])

    def test_explicit_padding_and_derived_lengths_not_loss_mask_inference(self):
        b=bundle(steps=7,mode='pad');p,_=resolve_run(b,constrained=False)
        seq=p['populations']['train']['inputs']['sequence'];lengths=p['populations']['train']['inputs']['lengths']
        for i,(x,l) in enumerate(zip(seq,lengths)):
            self.assertEqual(l,[float(i%8)]);self.assertEqual(x[int(l[0]):],[[0.,0.]]*(7-int(l[0])))
        t=copy.deepcopy(b['task']);t['heads'][0].pop('mask');a=rebuild(b,task=t)
        p2,_=resolve_run(a,constrained=False)
        self.assertNotEqual(p['populations']['train']['masks'],p2['populations']['train']['masks'])

    def test_fitted_transform_cannot_reinterpret_new_lengths(self):
        for mode in ('pad','standardize'):
            b=bundle(steps=7,mode=mode);spec={'dataset':b['dataset'],'task':b['task'],'runtime_policy':input_policy(b)}
            r=copy.deepcopy(b['dataset']['records'][2]['values'])
            self.assertEqual(record_shapes(spec,[r])['sequence'],[7,2])
            r['sequence']=[[.1,.2]]*8
            prepared,_=resolve_run(b,constrained=False)
            with self.subTest(mode=mode),self.assertRaises(GraphError):fitted_inputs(spec,prepared['fitted'],[r])

    def test_group_and_temporal_splits_preserve_record_meaning(self):
        b=bundle()
        for mode,field in (('group','subject'),('temporal','time')):
            protocol={'method':mode,'field':field,'fractions':[.5,.25,.25]}
            if mode=='group':protocol['seed']=3
            m=make_split(b['dataset'],protocol);a=rebuild(b,manifest=m);p,_=resolve_run(a,constrained=False)
            rows={r['id']:r['values'] for r in b['dataset']['records']}
            ids=[p['populations'][r]['observation_ids'] for r in ('train','validation','test')]
            self.assertEqual(len(set(sum(ids,[]))),12)
            values=[{rows[i][field] for i in group} for group in ids]
            if mode=='group':
                for i in range(3):
                    for j in range(i):self.assertFalse(values[i]&values[j])
            else:self.assertLess(max(values[0]),min(values[1]));self.assertLess(max(values[1]),min(values[2]))


if __name__=='__main__':unittest.main()
