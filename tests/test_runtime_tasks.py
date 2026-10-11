"""Explicit spatial execution policies, provenance and actual dataset contracts; no Torch."""
import copy
import json
import random
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from app.core import GraphError, Workspace, VERSION
from app.experiment_objects import candidate_bundle, preview_experiment, selection_at
from app.evidence import digest
from app.runtime_tasks import record_shapes, input_policy
from app.task_execution import validate_bundle, run_identity, resolve_run
from test_family_axes import family_experiment
from test_retained_runtime import source


def runtime_experiment(*, shape=(1,3,5), execution='runtime-spatial-v1', version=4, pairs=1):
    w,e=family_experiment()
    f=source()
    for symbol,value in zip(f['symbols'],(1,2,1,1)):symbol['default']=value
    f['assignments'][0]['value']=[{'symbol':'C'},2,2]
    f['models']['cnn']['graph']['nodes'][0]['shape']=[1,2,2]
    next(n for n in f['models']['cnn']['graph']['nodes'] if n['id']=='conv')['groups']=1
    ref=w.put_spec('family',f)
    e.pop('revision');e.update(id='runtime-task',version=version,initialization_seeds=[7],data_tasks=e['data_tasks'][:pairs])
    e['models']=[{'id':'runtime','family':ref,'bindings':{},'axes':[]}]
    if execution is not None:e['models'][0]['execution']=execution
    for i,pair in enumerate(e['data_tasks']):
        d=w.get_spec('dataset',pair['dataset']['id']);d.pop('revision')
        c,h,width=shape;h+=i
        d['fields'][0].update(shape=[c,h,width],axes=['C','H','W'])
        for row in d['records']:
            v=row['values']['x'][0]/8
            row['values']['x']=[[[v+j/100+k/1000 for k in range(width)] for j in range(h)] for _ in range(c)]
        pair['dataset']=w.put_spec('dataset',d)
    ref=w.put_spec('experiment',e)
    return w,w.get_spec('experiment',ref['id'],ref['revision']),ref


class RuntimeTaskTests(unittest.TestCase):
    def make(self,**kw):
        w,e,ref=runtime_experiment(**kw);self.addCleanup(w.db.close)
        self.addCleanup(lambda:w.history.directory.cleanup() if w.history.directory else None)
        return w,e,ref

    def test_actual_shapes_are_dataset_cases_not_new_constructor_bindings(self):
        w,e,_=self.make(pairs=2);before=w.db.serialize();rng=random.getstate()
        rows=preview_experiment(e,w.get_spec)['candidates'];self.assertEqual([r['status'] for r in rows],['compatible']*2)
        bundles=[candidate_bundle(e,r['coordinates'],w.get_spec) for r in rows]
        self.assertEqual([validate_bundle(b)['input_groups'][0]['shape'] for b in bundles],[[1,3,5],[1,4,5]])
        self.assertEqual([b['model']['models'][b['model']['root']]['graph']['nodes'][0]['shape'] for b in bundles],[[1,2,2]]*2)
        self.assertEqual(bundles[0]['model'],bundles[1]['model'])
        self.assertEqual(len({digest(run_identity(b)) for b in bundles}),2)
        self.assertEqual(bundles[0]['runtime']['specialization']['runtime']['inputs'],{'input':[1,2]})
        self.assertEqual(w.db.serialize(),before);self.assertEqual(random.getstate(),rng)

    def test_older_and_explicit_concrete_experiments_never_opt_in_implicitly(self):
        for version,execution in ((3,None),(4,None),(4,'concrete-v1')):
            with self.subTest(version=version,execution=execution):
                w,e,_=self.make(version=version,execution=execution)
                row=preview_experiment(e,w.get_spec)['candidates'][0]
                self.assertEqual(row['status'],'incompatible')
                with self.assertRaises(GraphError):candidate_bundle(e,row['coordinates'],w.get_spec)
        w,e,_=self.make(shape=(1,2,2),version=3,execution=None)
        b=candidate_bundle(e,selection_at(e,0),w.get_spec);self.assertEqual(b['version'],1)
        identity=run_identity(b);self.assertEqual(identity['version'],1);self.assertEqual(identity['compiler'],'native-graph-v1')
        self.assertNotIn('runtime',identity);self.assertNotIn('runtime',b)

    def test_policy_is_closed_and_cannot_be_added_to_old_grammar(self):
        w,e,_=self.make();before=w.db.serialize()
        for version,policy in ((3,'runtime-spatial-v1'),(4,'automatic'),(4,True),(4,{}),(6,'runtime-spatial-v1')):
            bad=copy.deepcopy(e);bad.pop('revision');bad['version']=version;bad['models'][0]['execution']=policy
            with self.subTest(version=version,policy=policy),self.assertRaises(GraphError):w.put_spec('experiment',bad)
            self.assertEqual(w.db.serialize(),before)

    def test_frozen_runtime_proof_rejects_forged_witness_model_and_historical_family(self):
        w,e,_=self.make();b=candidate_bundle(e,selection_at(e,0),w.get_spec)
        for change in ('witness','family','model','extra','version'):
            bad=copy.deepcopy(b)
            if change=='witness':bad['runtime']['specialization']['runtime']['inputs']={'input':[1]}
            if change=='family':bad['runtime']['family']['runtime_inputs']={'input':[1]}
            if change=='model':bad['model']['models'][bad['model']['root']]['graph']['nodes'][0]['shape']=[1,9,9]
            if change=='extra':bad['runtime']['inferred']=True
            if change=='version':bad['version']=True
            with self.subTest(change=change),self.assertRaises(GraphError):validate_bundle(bad)
        family=w.get_spec('family',e['models'][0]['family']['id']);family.pop('revision');family['runtime_inputs']={'input':[1]};w.put_spec('family',family)
        self.assertEqual(candidate_bundle(e,selection_at(e,0),w.get_spec),b)

    def test_wrong_channels_undeclared_width_and_target_shape_stay_incompatible(self):
        w,e,_=self.make(shape=(2,3,5));self.assertEqual(preview_experiment(e,w.get_spec)['candidates'][0]['status'],'incompatible')
        w,e,_=self.make();f=w.get_spec('family',e['models'][0]['family']['id']);f.pop('revision');f['runtime_inputs']={'input':[1]}
        e.pop('revision');e['models'][0]['family']=w.put_spec('family',f);r=w.put_spec('experiment',e)
        e=w.get_spec('experiment',r['id'],r['revision']);self.assertEqual(preview_experiment(e,w.get_spec)['candidates'][0]['status'],'incompatible')

    def test_actual_memory_admission_precedes_torch_and_preparation(self):
        w,e,_=self.make();b=candidate_bundle(e,selection_at(e,0),w.get_spec)
        with self.assertRaisesRegex(GraphError,'constrained|RAM'):
            resolve_run(b,available_bytes=1)
        prepared,memory=resolve_run(b,constrained=False,available_bytes=1)
        self.assertEqual(prepared['contract']['input_groups'][0]['shape'],[1,3,5])
        self.assertEqual(memory['actual_batch_size'],3)

    def test_runtime_inference_rectangularity_rank_fixed_axes_and_preprocessing_are_explicit(self):
        w,e,_=self.make();b=candidate_bundle(e,selection_at(e,0),w.get_spec)
        spec={'dataset':b['dataset'],'task':b['task'],'runtime_policy':input_policy(b)}
        good=[{'x':[[[1.,2.],[3.,4.]]]}];self.assertEqual(record_shapes(spec,good),{'input':[1,2,2]})
        for rows in ([],[{'x':[[[1.],[2.,3.]]]}],[{'x':[[[[1.]]]]}], [{'x':[[[1.]],[[2.]]]}],good+[{'x':[[[1.]]]}],[{}]):
            with self.subTest(rows=rows),self.assertRaises(GraphError):record_shapes(spec,rows)
        spec['task']['inputs'][0]['transforms']=[{'kind':'standardize'}]
        with self.assertRaisesRegex(GraphError,'resolution-independent'):record_shapes(spec,good)

    def test_version_thirteen_gate_is_nonmutating_and_older_v12_families_remain_readable(self):
        w,e,_=self.make()
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'runtime.vtw';w.save(path)
            with closing(sqlite3.connect(path)) as db:db.execute('PRAGMA user_version=12');db.commit()
            before=path.read_bytes()
            with self.assertRaisesRegex(GraphError,'version-13'):Workspace.load(path)
            self.assertEqual(path.read_bytes(),before)
        w,e,_=self.make(version=3,execution=None)
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'old.vtw';w.save(path)
            with closing(sqlite3.connect(path)) as db:db.execute('PRAGMA user_version=12');db.commit()
            before=path.read_bytes();loaded=Workspace.load(path)
            try:self.assertEqual(loaded.db.execute('PRAGMA user_version').fetchone()[0],VERSION)
            finally:loaded.db.close()
            self.assertEqual(path.read_bytes(),before)

    def test_owned_policy_upgrade_is_explicit_atomic_and_cannot_downgrade(self):
        from app.authoring_sessions import Sessions
        w,e,ref=self.make(version=3,execution=None);s=Sessions();owner=object();self.addCleanup(s.close_owner,owner)
        h=s.open(w,owner,'experiment',base=ref);before=w.db.serialize()
        page=s.experiment_page(h['session'],owner,0,'models');self.assertFalse(page['rows'][0]['properties'][-1]['present'])
        ops=[{'op':'property','target':{'kind':'document'},'key':'version','text':'4'},
             {'op':'property','target':{'kind':'entry','collection':'models','ordinal':0},'key':'execution','text':'"runtime-spatial-v1"'}]
        h=s.experiment_edit(w,h['session'],owner,0,ops)
        self.assertEqual(s.experiment_preview(w,h['session'],owner,h['sequence'])['candidates'][0]['status'],'compatible')
        self.assertEqual(w.db.serialize(),before)
        with self.assertRaises(GraphError):s.experiment_edit(w,h['session'],owner,h['sequence'],[dict(ops[0],text='3')])
        r=s.commit(w,h['session'],owner,h['sequence']);self.assertEqual(w.get_spec('experiment',r['id'])['version'],4)

    def test_fresh_runtime_preview_imports_no_numeric_or_qt_library(self):
        code="""import sys
from test_runtime_tasks import runtime_experiment
from app.experiment_objects import candidate_bundle,selection_at
from app.task_execution import validate_bundle
w,e,r=runtime_experiment()
try:
 b=candidate_bundle(e,selection_at(e,0),w.get_spec)
 assert validate_bundle(b)['input_groups'][0]['shape']==[1,3,5]
 assert not any(k.startswith(('torch','numpy','PyQt')) for k in sys.modules)
 print('PASS')
finally:w.db.close()
"""
        done=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True,check=True)
        self.assertEqual(done.stdout.strip(),'PASS')

    def test_actual_cli_preview_uses_explicit_runtime_case_and_never_saves(self):
        w,e,ref=self.make();root=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'runtime.vtw';w.save(path);before=path.read_bytes()
            done=subprocess.run([sys.executable,str(root/'vpw_cli.py'),str(path),'preview-experiment',ref['id'],ref['revision']],
                                capture_output=True,text=True,check=True)
            preview=json.loads(done.stdout)['value'];self.assertEqual(preview['candidates'][0]['status'],'compatible')
            self.assertEqual(path.read_bytes(),before)

    def test_bundle_closed_object_errors_are_graph_errors_not_attribute_errors(self):
        for value in (None,[],3,'runtime'):
            with self.subTest(value=value),self.assertRaises(GraphError):validate_bundle(value)

    def test_changed_loaded_helper_source_refuses_export_instead_of_emitting_wrong_function(self):
        from app.task_execution import _helper_source, fitted_inputs
        self.assertTrue(_helper_source(fitted_inputs).startswith('def fitted_inputs('))
        for raw in ('def wrong():\n    pass\n','def fitted_inputs(bundle, fitted, records):\n    return {}\n'):
            with self.subTest(raw=raw),patch('app.task_execution.inspect.getsource',return_value=raw),self.assertRaisesRegex(GraphError,'changed since import'):
                _helper_source(fitted_inputs)

    def test_worker_kind_mismatch_and_forged_proof_fail_before_numeric_import(self):
        w,e,_=self.make();b=candidate_bundle(e,selection_at(e,0),w.get_spec)
        bad=copy.deepcopy(b);bad['runtime']['specialization']['runtime']['specification_sha256']='0'*64
        for kind,value in (('task-v1',b),('task-v2',dict(b,version=True)),('task-v2',bad)):
            script='import sys;from app.worker import main;code=main();assert "torch" not in sys.modules;raise SystemExit(code)'
            done=subprocess.run([sys.executable,'-c',script],input=json.dumps({'kind':kind,'bundle':value}),capture_output=True,text=True)
            self.assertEqual(done.returncode,1,done.stderr);self.assertEqual(json.loads(done.stdout)['kind'],'error')

    def test_missing_runtime_domain_does_not_become_automatic_shape_inference(self):
        for version in (4,5):
            w,e,_=self.make();f=w.get_spec('family',e['models'][0]['family']['id']);f.pop('revision')
            f.pop('runtime_inputs');f['version']=version;e.pop('revision');e['models'][0]['family']=w.put_spec('family',f)
            ref=w.put_spec('experiment',e);retained=w.get_spec('experiment',ref['id'],ref['revision'])
            row=preview_experiment(retained,w.get_spec)['candidates'][0]
            self.assertEqual(row['status'],'incompatible');self.assertIn('Runtime task',row['reason'])
