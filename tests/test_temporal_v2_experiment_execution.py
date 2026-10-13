"""Actual retained v2 workers/CLI/checkpoints and saved/rerun notebook kernels."""
import copy
import json
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

import torch

from app.core import GraphError, Workspace, VERSION
from app.evidence import digest
from app.experiment_objects import (admit_run, candidate_bundle, execute_candidate,
                                   preview_experiment, selection_at, validate_run)
from app.notebooks import capture_bundle, create_bundle, execute_bundle, publish_bundle, verify_bundle
from app.revisions import validate_all
from app.task_execution import generate_task_predictor, restore_task_predictor
from app.task_training import train_task
from app.task_runtime import forward_ports
from runtime_v2_experiment_fixture import experiment

torch.set_num_threads(1)
ROOT=Path(__file__).resolve().parents[1]


class TemporalV2ExperimentExecutionTests(unittest.TestCase):
    def make(self,**kwargs):
        w,e,r=experiment(**kwargs);self.addCleanup(w.db.close)
        self.addCleanup(lambda:w.history.directory.cleanup() if w.history.directory else None)
        return w,e,r

    def run_candidate(self,w,e,ref,index=0,rid='trial'):
        done=execute_candidate(w,ref,selection_at(e,index),rid,'Actual runtime-v2 trial',python=sys.executable,timeout=30)
        self.assertEqual(done['record']['version'],6)
        return done

    def test_two_actual_workers_preserve_one_retained_constructor_exact_local_fits(self):
        w,e,ref=self.make();specializations=[];models=[]
        for i in (0,1):
            b=candidate_bundle(e,selection_at(e,i),w.get_spec);local=train_task(b,constrained=False)
            done=self.run_candidate(w,e,ref,i,'trial-'+str(i));a=done['record']['attempts'][0]
            self.assertEqual(a['status'],'completed',a['error']);self.assertEqual(a['result']['checkpoint'],local['checkpoint'])
            self.assertEqual(a['result']['history'],local['history']);self.assertEqual(a['result']['final_metrics'],local['final_metrics'])
            self.assertEqual(a['result']['identity']['version'],4)
            refs=done['record']['resolved']['references'];s=w.get_spec('specialization',refs['specialization']['id'],refs['specialization']['revision'])
            self.assertEqual(s['version'],7);self.assertEqual(s['runtime'],b['runtime']['specialization']['runtime'])
            specializations.append(refs['specialization']);models.append(refs['model'])
        self.assertEqual(specializations[0],specializations[1]);self.assertEqual(models[0],models[1])
        validate_all(w.db)

    def test_real_cli_run_preview_reload_export_and_variable_T_fresh_predictor(self):
        w,e,ref=self.make(times=(3,));done=self.run_candidate(w,e,ref);r=done['record'];checkpoint=r['attempts'][0]['result']['checkpoint']
        b=candidate_bundle(e,selection_at(e,0),w.get_spec)
        with tempfile.TemporaryDirectory() as td:
            folder=Path(td);path=folder/'trial.vtw';w.save(path);coordinates=folder/'coordinates.json'
            coordinates.write_text(json.dumps(selection_at(e,0)))
            command=[sys.executable,str(ROOT/'vpw_cli.py'),str(path),'run-candidate','--experiment',ref['id'],ref['revision'],
                     '--coordinates',str(coordinates),'--run-id','cli-trial','--name','CLI runtime-v2','--timeout','25']
            actual=subprocess.run(command,capture_output=True,text=True,timeout=40)
            self.assertEqual(actual.returncode,0,actual.stderr);value=json.loads(actual.stdout)['value']
            self.assertTrue(value['persisted']);self.assertEqual(value['record']['version'],6)
            self.assertEqual(value['record']['attempts'][0]['result']['checkpoint'],checkpoint)
            loaded=Workspace.load(path)
            try:
                self.assertEqual(loaded.db.execute('PRAGMA user_version').fetchone()[0],VERSION)
                self.assertEqual(loaded.get_spec('run',r['id'],r['revision']),r);validate_all(loaded.db)
            finally:loaded.db.close()
            before=path.read_bytes();export=subprocess.run([sys.executable,str(ROOT/'vpw_cli.py'),str(path),
                  'export-run-trained',r['id'],r['revision']],capture_output=True,text=True,timeout=25)
            self.assertEqual(export.returncode,0,export.stderr);self.assertEqual(path.read_bytes(),before)
            code=json.loads(export.stdout)['value']
            # CLI emits raw code for export-run-trained (same direct exporter).
            if isinstance(code,dict):code=code['code']
            target=folder/'predictor.py';target.write_text(code)
            row=copy.deepcopy(b['dataset']['records'][2]['values']);row.update(sequence=[[.2,.3]]*9,lengths=[9.])
            expected=restore_task_predictor(b,checkpoint,constrained=False)([row])
            probe='''import json,runpy,sys,torch
torch.set_num_threads(1);s=runpy.run_path(sys.argv[1]);out=s['predict_records'](json.loads(sys.argv[2]))
assert not any(n=='app' or n.startswith('app.') for n in sys.modules)
print(json.dumps({k:v.tolist() for k,v in out.items()}))'''
            result=subprocess.run([sys.executable,'-c',probe,str(target),json.dumps([row])],cwd=folder,capture_output=True,text=True,timeout=25)
            self.assertEqual(result.returncode,0,result.stderr);self.assertEqual(json.loads(result.stdout),{k:v.tolist() for k,v in expected.items()})

    def test_exact_retry_and_historical_selection_cannot_substitute_new_heads_or_manifest(self):
        w,e,ref=self.make(times=(3,));first=self.run_candidate(w,e,ref);checkpoint=first['record']['attempts'][0]['result']['checkpoint']
        f=w.get_spec('family',e['models'][0]['family']['id']);f.pop('revision');f['name']='New head';w.put_spec('family',f)
        repeat=execute_candidate(w,ref,selection_at(e,0),'trial','Retry old exact selection',retry=True,python=sys.executable,timeout=30)
        self.assertEqual(len(repeat['record']['attempts']),2);self.assertEqual(repeat['record']['attempts'][1]['result']['checkpoint'],checkpoint)
        self.assertEqual(repeat['record']['resolved'],first['record']['resolved'])
        before=w.db.serialize();bad=copy.deepcopy(first['record']['resolved']['manifest']);bad['roles']['train'].reverse()
        with self.assertRaisesRegex(GraphError,'manifest'):
            execute_candidate(w,ref,selection_at(e,0),'trial','Forged retry',retry=True,saved_manifest=bad,python=sys.executable,timeout=30)
        self.assertEqual(w.db.serialize(),before)

    def test_unavailable_and_admission_failure_preserve_explicit_terminal_meaning(self):
        w,e,ref=self.make(times=(7,),mode='pad',unavailable=True);done=self.run_candidate(w,e,ref)
        a=done['record']['attempts'][0];self.assertEqual(a['status'],'unavailable');self.assertIsNotNone(a['result'])
        self.assertEqual(a['result']['counts']['completed_updates'],0);self.assertIsNone(a['error'])
        w,e,ref=self.make(times=(3,));w.set_constrained_mode(True)
        with patch('app.task_execution.available_memory',return_value=1),patch('app.transport.run_json_worker') as worker:
            done=self.run_candidate(w,e,ref);worker.assert_not_called()
        a=done['record']['attempts'][0]
        self.assertEqual(a['status'],'failed');self.assertEqual(a['stage'],'admission');self.assertIsNone(a['result'])
        self.assertIn('RAM',a['error']['message']);validate_all(w.db)

    def test_run_version_proof_result_or_memory_forgery_never_creates_rows(self):
        w,e,ref=self.make(times=(3,));done=self.run_candidate(w,e,ref)
        before=w.db.serialize();original=done['record']
        for mutation in ('version','compiler','specialization','memory'):
            bad=copy.deepcopy(original);bad.pop('revision')
            if mutation=='version':bad['version']=5
            elif mutation=='compiler':bad['attempts'][0]['result']['identity']['compiler']='native-runtime-temporal-v1'
            elif mutation=='specialization':bad['resolved']['references']['specialization']['revision']='0'*64
            else:
                memory=bad['attempts'][0]['result']['memory_admission'];memory['components']['lstm_unrolls']=0
                memory['estimated_bytes']=sum(memory['components'].values())
            with self.subTest(mutation=mutation),self.assertRaises(GraphError):admit_run(w,bad)
            self.assertEqual(w.db.serialize(),before)
        for version in (True,8):
            bad=copy.deepcopy(original);bad.pop('revision');bad['version']=version
            with self.assertRaises(GraphError):validate_run(bad)

    def test_late_run_retention_failure_rolls_back_model_specialization_rows_dirty_and_history(self):
        w,e,ref=self.make(times=(3,));done=self.run_candidate(w,e,ref)
        target,_,_=self.make(times=(3,));before=target.db.serialize();dirty=target.dirty;real=target.put_spec
        def fail(kind,doc,**kwargs):
            result=real(kind,doc,**kwargs)
            if kind=='run':raise RuntimeError('late run row')
            return result
        with patch.object(target,'put_spec',side_effect=fail),self.assertRaisesRegex(RuntimeError,'late run row'):
            admit_run(target,done['record'])
        self.assertEqual(target.db.serialize(),before);self.assertEqual(target.dirty,dirty)
        self.assertEqual(target.list_specs('specialization'),[]);validate_all(target.db)

    def test_vtw20_fence_also_covers_run6_with_legacy_experiment_headers(self):
        w,e,ref=self.make(times=(3,));done=self.run_candidate(w,e,ref)
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'run.vtw';w.save(path)
            with closing(sqlite3.connect(path)) as db:db.execute('PRAGMA user_version=19');db.commit()
            before=path.read_bytes()
            with self.assertRaisesRegex(GraphError,'version-20'):Workspace.load(path)
            self.assertEqual(path.read_bytes(),before)

    def test_actual_saved_and_rerun_kernels_replay_exact_v2_metrics_checkpoints_and_predictors(self):
        w,e,ref=self.make(times=(7,),mode='pad',updates=3);done=self.run_candidate(w,e,ref)
        result=done['record']['attempts'][0]['result'];b=candidate_bundle(e,selection_at(e,0),w.get_spec)
        records=[row['values'] for row in b['dataset']['records'][-3:]]
        expected=restore_task_predictor(b,result['checkpoint'],constrained=False)(records)
        with tempfile.TemporaryDirectory() as td:
            views=[]
            for mode in ('result-report','rerun'):
                recipe=w.put_spec('report',{'version':1,'id':mode,'name':'Retained runtime-v2 report',
                    'question':'Do exact retained LSTM tasks reproduce in a real kernel?',
                    'mode':mode,'experiment':ref if mode=='rerun' else None,'runs':[] if mode=='rerun' else [done['reference']],
                    'sections':['models','training','evaluation','checkpoints','failures'],
                    'analysis':{'kind':'descriptive','inferential_unit':'run'}})
                before=w.db.serialize();captured=create_bundle(w,recipe,Path(td)/mode)
                verify_bundle(captured['path']);self.assertTrue((Path(captured['path'])/'runtime/app/temporal_shapes_v2.py').is_file())
                actual=execute_bundle(captured['path'],timeout=60)
                self.assertEqual(actual['execution']['status'],'completed',actual['execution']['error'])
                directory=Path(actual['path']);view=json.loads((directory/'report-data.json').read_text());views.append(view)
                self.assertEqual(view['runs'][0]['checkpoint_id'],result['checkpoint_id'])
                self.assertEqual(view['notebook_execution']['training_requested'],mode=='rerun')
                self.assertEqual(w.db.serialize(),before)
                predictors=list(directory.glob('predictor-*.py'));self.assertEqual(len(predictors),1)
                scope={};exec(predictors[0].read_text(),scope);out=scope['predict_records'](records)
                for key in expected:torch.testing.assert_close(out[key],expected[key],rtol=0,atol=0)
                architectures=list(directory.glob('runtime-architecture-*.py'));self.assertEqual(len(architectures),1)
                scope={};exec(architectures[0].read_text(),scope)
                row=copy.deepcopy(records[0]);row.update(sequence=[[.2,.3]]*11,lengths=[11.])
                output=forward_ports(scope['model'],{g['id']:torch.tensor([row[g['id']]])
                                     for g in scope['model'].contract['input_groups']},scope['model'].contract)
                self.assertEqual(output['out_sequence'].shape,(1,11,4))
            self.assertEqual(views[0]['runs'][0]['metrics'],views[1]['runs'][0]['metrics'])

    def test_missing_v2_source_capture_refuses_before_publish_or_jupyter_import(self):
        w,e,ref=self.make(times=(7,),mode='pad')
        recipe=w.put_spec('report',{'version':1,'id':'capture','name':'Exact captured runtime-v2',
             'question':'Is the captured source closure complete?', 'mode':'rerun','experiment':ref,'runs':[],
             'sections':['models','training','evaluation','checkpoints','failures'],
             'analysis':{'kind':'descriptive','inferential_unit':'run'}})
        value=capture_bundle(w,recipe);value['files'].pop('runtime/app/temporal_shapes_v2.py')
        before=w.db.serialize()
        with tempfile.TemporaryDirectory() as td,patch('app.notebooks._jupyter') as jupyter:
            path=Path(td)/'incomplete'
            with self.assertRaisesRegex(GraphError,'closure'):publish_bundle(value,path)
            self.assertFalse(path.exists());jupyter.assert_not_called()
        self.assertEqual(w.db.serialize(),before)


if __name__=='__main__':unittest.main()
