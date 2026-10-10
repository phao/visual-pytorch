"""Exact stored checkpoints become standalone inference, never new training."""
import copy
import hashlib
import json
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from app.core import GraphError, Workspace
from app.experiment_objects import candidate_bundle, execute_candidate, preview_experiment
from app.run_export import predictor_source, write_new_predictor
from evidence_fixture import populate, experiment
from test_task_authoring import sequence_fixture


class RunExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.w=Workspace();cls.f=populate(cls.w,sys.executable)
    @classmethod
    def tearDownClass(cls):
        if cls.w.history.directory:cls.w.history.directory.cleanup()
        cls.w.db.close()
    def source(self,key='first',**kwargs):return predictor_source(self.w,self.f[key]['reference'],**kwargs)
    def bundle(self,key):
        record=self.f[key]['record'];ref=record['selection']['experiment']
        e=self.w.get_spec('experiment',ref['id'],ref['revision'])
        return candidate_bundle(e,record['selection']['coordinates'],self.w.get_spec,saved_manifest=record['resolved']['manifest'])

    def test_generation_is_sql_rng_readonly_no_worker_new_split_or_inference(self):
        before=self.w.db.serialize();rng=random.getstate()
        with (patch('app.transport.run_json_worker',side_effect=AssertionError('No training')),
             patch('app.experiment_objects.make_split',side_effect=AssertionError('No new split')),
             patch('app.task_execution.restore_task_predictor',side_effect=AssertionError('No inference'))):
            source=self.source()
        self.assertEqual(before,self.w.db.serialize());self.assertEqual(random.getstate(),rng)
        self.assertEqual(source['provenance']['reference'],self.f['first']['reference'])
        self.assertEqual(source['provenance']['completed_updates'],3)
        self.assertEqual(source['code_sha256'],hashlib.sha256(source['code'].encode()).hexdigest())
        compile(source['code'],'retained-predictor.py','exec')

    def test_latest_failed_retry_cannot_export_earlier_success_implicitly(self):
        before=self.w.db.serialize()
        with self.assertRaisesRegex(GraphError,'no inference checkpoint'):self.source('retry')
        previous=self.source('retry',attempt=1)
        self.assertEqual(previous['provenance']['attempt'],1)
        self.assertEqual(previous['provenance']['reference'],self.f['retry']['reference'])
        self.assertEqual(previous['provenance']['checkpoint_id'],self.source()['provenance']['checkpoint_id'])
        self.assertEqual(before,self.w.db.serialize())

    def test_actual_partial_and_initial_checkpoints_keep_status_and_update_count(self):
        for key in ('cancelled','zero'):
            source=self.source(key);result=self.f[key]['record']['attempts'][-1]['result']
            self.assertEqual(source['provenance']['status'],result['status'])
            self.assertEqual(source['provenance']['completed_updates'],result['counts']['completed_updates'])
            self.assertFalse(source['provenance']['optimizer_resume'])
        self.assertEqual(self.source('zero')['provenance']['completed_updates'],0)
        with self.assertRaisesRegex(GraphError,'no inference checkpoint'):self.source('excluded')

    def test_invalid_exact_reference_or_attempt_fails_without_changes(self):
        before=self.w.db.serialize()
        for attempt in (0,True,2,2**53):
            with self.subTest(attempt=attempt),self.assertRaises(GraphError):self.source(attempt=attempt)
        for ref in (dict(self.f['first']['reference'],revision=None),dict(self.f['first']['reference'],kind='model'),
                    dict(self.f['first']['reference'],revision='0'*64)):
            with self.assertRaises(GraphError):predictor_source(self.w,ref)
        self.assertEqual(before,self.w.db.serialize())

    def test_mutable_drawing_dataset_task_heads_cannot_change_frozen_export(self):
        before=self.source();bundle=self.bundle('first');mid=bundle['model']['id']
        original=self.w.models()[mid];changed=copy.deepcopy(original)
        next(n for n in changed['graph']['nodes'] if n['type']=='Linear')['features']=5
        self.w.put(changed)
        try:self.assertEqual(self.source(),before)
        finally:self.w.put(original)
        original_task=bundle['task'];changed=copy.deepcopy(original_task);changed.pop('revision');changed['name']='New task head'
        self.w.put_spec('task',changed)
        try:self.assertEqual(self.source(),before)
        finally:self.w.select_spec('task',original_task['id'],original_task['revision'])

    def test_generated_weighted_multiinput_predictor_agrees_bitwise_and_has_no_raw_records(self):
        import torch
        from app.task_execution import restore_task_predictor
        source=self.source();bundle=self.bundle('first');checkpoint=self.f['first']['record']['attempts'][0]['result']['checkpoint']
        records=[{'a':[100.],'b':None},{'a':[-2.],'b':[15.]}]
        expected=restore_task_predictor(bundle,checkpoint,constrained=False)(records)
        namespace={};exec(source['code'],namespace);actual=namespace['predict_records'](records)
        self.assertEqual(set(actual),{'category','estimate'})
        for port in actual:self.assertTrue(torch.equal(actual[port],expected[port]))
        self.assertEqual(namespace['saved_run_provenance'],source['provenance'])
        self.assertEqual(namespace['_fitted'],checkpoint['fitted'])
        self.assertFalse(namespace['model'].training)
        self.assertNotIn('from app',source['code']);self.assertNotIn("'records':",source['code'])
        self.assertNotIn('Adam(',source['code']);self.assertNotIn('optimizer.state',source['code'])
        self.assertIn('classification-logits',source['provenance']['prediction_semantics'])
        self.assertEqual(namespace['task_specification']['heads'][1]['transforms'],[{'kind':'affine','scale':.5,'offset':-1.}])
        with self.assertRaisesRegex(ValueError,'missing input field'):namespace['predict_records']([{'a':[1.]}])

    def test_standalone_export_runs_in_unrelated_directory_without_application_import(self):
        import torch
        from app.task_execution import restore_task_predictor
        records=[{'a':[3.],'b':None}]
        expected={k:v.tolist() for k,v in restore_task_predictor(self.bundle('first'),
            self.f['first']['record']['attempts'][0]['result']['checkpoint'],constrained=False)(records).items()}
        with tempfile.TemporaryDirectory() as directory:
            source=self.source();write_new_predictor(source,Path(directory)/'predictor.py')
            script="import sys,json,predictor;assert not any(m=='app' or m.startswith('app.') for m in sys.modules);print(json.dumps({k:v.tolist() for k,v in predictor.predict_records("+repr(records)+").items()}))"
            proc=subprocess.run([sys.executable,'-c',script],cwd=directory,capture_output=True,text=True,timeout=20)
            self.assertEqual(proc.returncode,0,proc.stderr);self.assertEqual(json.loads(proc.stdout),expected)

    def test_cli_source_matches_backend_without_host_torch_or_file_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'source.vtw';self.w.save(path);before=path.read_bytes();ref=self.f['retry']['reference']
            script="import sys,vpw_cli;code=vpw_cli.main(sys.argv[1:]);assert not any(m in sys.modules for m in ('torch','numpy','PyQt6','nbformat'));raise SystemExit(code)"
            proc=subprocess.run(['/usr/bin/python3','-c',script,str(path),'export-run-trained',ref['id'],ref['revision'],'--attempt','1'],capture_output=True,text=True,timeout=20)
            self.assertEqual(proc.returncode,0,proc.stderr);self.assertEqual(json.loads(proc.stdout)['value'],self.source('retry',attempt=1))
            self.assertEqual(path.read_bytes(),before)

    def test_exclusive_export_bytes_hash_no_overwrite_and_no_final_symlink_follow(self):
        source=self.source();before=self.w.db.serialize()
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'new.py';receipt=write_new_predictor(source,path)
            self.assertEqual(path.read_text(),source['code']);self.assertEqual(receipt['code_sha256'],hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(receipt['provenance'],source['provenance'])
            with self.assertRaises(FileExistsError):write_new_predictor(source,path)
            missing=Path(directory)/'missing.py';link=Path(directory)/'link.py';link.symlink_to(missing)
            with self.assertRaises(FileExistsError):write_new_predictor(source,link)
            self.assertFalse(missing.exists());self.assertEqual(path.read_text(),source['code'])
            tampered=Path(directory)/'tampered.py'
            with self.assertRaisesRegex(GraphError,'source hash'):
                write_new_predictor(dict(source,code=source['code']+'\n'),tampered)
            self.assertFalse(tampered.exists())
        self.assertEqual(before,self.w.db.serialize())

    def test_cli_failed_latest_has_no_source_and_leaves_file_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'source.vtw';self.w.save(path);before=path.read_bytes();ref=self.f['retry']['reference']
            proc=subprocess.run(['/usr/bin/python3','vpw_cli.py',str(path),'export-run-trained',ref['id'],ref['revision']],capture_output=True,text=True,timeout=20)
            self.assertEqual(proc.returncode,1);reply=json.loads(proc.stdout)
            self.assertFalse(reply['ok']);self.assertIn('no inference checkpoint',reply['error'])
            self.assertNotIn('code',reply);self.assertEqual(path.read_bytes(),before)

    def test_generation_uses_existing_validation_ram_policy_before_source_emission(self):
        self.w.set_constrained_mode(True)
        try:
            before=self.w.db.serialize()
            with patch('app.task_execution.available_memory',return_value=1):
                with self.assertRaisesRegex(GraphError,'RAM'):self.source()
            self.assertEqual(before,self.w.db.serialize())
        finally:self.w.set_constrained_mode(False)

    def test_ragged_padding_derived_lengths_masks_export_matches_retained_checkpoint(self):
        import torch
        from app.task_execution import restore_task_predictor
        m,d,t=sequence_fixture();ref=experiment(self.w,m,d,t,'export-sequence')
        doc=self.w.get_spec('experiment',ref['id'],ref['revision']);coords=preview_experiment(doc,self.w.get_spec)['candidates'][0]['coordinates']
        run=execute_candidate(self.w,ref,coords,'sequence-export-run','Sequence checkpoint',python=sys.executable,timeout=20)
        self.assertEqual(run['record']['attempts'][0]['status'],'completed')
        source=predictor_source(self.w,run['reference']);bundle=candidate_bundle(doc,coords,self.w.get_spec,saved_manifest=run['record']['resolved']['manifest'])
        records=[{'seq':[]},{'seq':[[4.,5.],[6.,7.]]}];namespace={};exec(source['code'],namespace)
        expected=restore_task_predictor(bundle,run['record']['attempts'][0]['result']['checkpoint'],constrained=False)(records)
        self.assertTrue(torch.equal(namespace['predict_records'](records)['output'],expected['output']))
        with self.assertRaisesRegex(ValueError,'truncate|overlong'):namespace['predict_records']([{'seq':[[1.,2.]]*4}])
