"""Presentation projects retained facts; independent numbers are not recomputed."""
import copy
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from app.core import Workspace,GraphError
from app.evidence import digest
from app.run_views import run_view,report_view
from evidence_fixture import populate


class RunViewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.w=Workspace();cls.f=populate(cls.w,sys.executable)
    @classmethod
    def tearDownClass(cls):
        if cls.w.history.directory:cls.w.history.directory.cleanup()
        cls.w.db.close()
    def view(self,key='first',**kwargs):return run_view(self.w,self.f[key]['reference'],**kwargs)

    def test_readonly_exact_deep_owned_without_worker_or_split_regeneration(self):
        before=self.w.db.serialize();rng=random.getstate()
        with patch('app.transport.run_json_worker',side_effect=AssertionError('No training')),patch('app.experiment_objects.make_split',side_effect=AssertionError('Captured manifests only')):
            view=self.view()
        self.assertEqual(before,self.w.db.serialize());self.assertEqual(rng,random.getstate())
        self.assertEqual(view['reference'],self.f['first']['reference']);self.assertEqual(view['scope'],'read-saved-evidence-only')
        view['execution']['counts']['completed_updates']=1000
        self.assertEqual(self.view()['execution']['counts']['completed_updates'],3)

    def test_every_scalar_metric_is_the_saved_value_with_identity_and_denominator(self):
        view=self.view();result=self.f['first']['record']['attempts'][0]['result']
        self.assertEqual(len(view['metrics']['rows']),6)
        for row in view['metrics']['rows']:
            original=result['final_metrics'][row['population']]['heads'][row['head']['id']]
            self.assertEqual(row['metric'],{k:v for k,v in original.items() if k!='observation_ids'}|{'observation_ids_hash':digest(original['observation_ids'])})
            self.assertEqual(row['metric']['checkpoint_id'],result['checkpoint_id'])
            self.assertEqual(row['head']['mask'],'valid');self.assertEqual(row['head']['sample_weight'],'weight')
            self.assertIn(row['head']['weight'],(.75,1.25))
            if row['head']['id']=='classification':
                self.assertEqual(row['metric']['accuracy'],original['accuracy']);self.assertEqual(row['head']['transforms'],[])
            else:self.assertEqual(row['head']['transforms'],[{'kind':'affine','scale':.5,'offset':-1.}]);self.assertIn('<svg',row['head']['source_unit'])

    def test_failing_latest_retry_never_substitutes_earlier_counts_or_metrics(self):
        latest=self.view('retry');self.assertEqual(latest['attempt']['number'],2)
        self.assertEqual(latest['attempt']['status'],'failed');self.assertIsNone(latest['execution'])
        self.assertEqual(latest['metrics']['rows'],[]);self.assertEqual({p['status'] for p in latest['populations']},{'not-retained'})
        previous=self.view('retry',attempt=1)
        self.assertEqual(previous['execution'],self.view()['execution']);self.assertEqual(previous['reference'],self.f['retry']['reference'])
        self.assertEqual(self.view()['attempt_count'],1)

    def test_cancellation_shows_actual_partial_counts_without_test_metrics(self):
        v=self.view('cancelled');self.assertEqual(v['attempt']['status'],'cancelled')
        self.assertGreaterEqual(v['execution']['counts']['completed_updates'],1)
        self.assertLess(v['execution']['counts']['completed_updates'],5000)
        self.assertEqual({r['population'] for r in v['metrics']['rows']},{'train','validation'})
        self.assertEqual(v['populations'][-1],{'population':'test','objective':None,'status':'not-retained','rule':None})

    def test_all_masked_metrics_are_unavailable_not_fabricated_zero(self):
        v=self.view('zero');self.assertEqual(v['execution']['counts']['completed_updates'],0)
        self.assertEqual(v['attempt']['status'],'unavailable')
        for row in v['metrics']['rows']:
            self.assertEqual(row['metric']['status'],'unavailable');self.assertIsNone(row['metric']['value'])
            self.assertEqual(row['metric']['denominator'],0)

    def test_exclusion_has_declared_context_but_no_captured_split_or_result(self):
        v=self.view('excluded');self.assertEqual(v['attempt']['stage'],'compatibility')
        self.assertEqual(v['attempt']['error']['message'],'Explicit negative control')
        self.assertIsNone(v['split']);self.assertIsNone(v['protocol']);self.assertIsNone(v['execution'])
        self.assertEqual(set(v['objects']),{'model','dataset','task'})

    def test_counts_checkpoint_retention_and_resolved_protocol_are_not_cosmetic(self):
        v=self.view();run=self.f['first']['record'];e=run['attempts'][0]['result']
        for key in ('counts','checkpoint_id','elapsed_seconds','history_retention','environment'):
            self.assertEqual(v['execution'][key],e[key])
        self.assertEqual(v['execution']['fitted_state_hash'],digest(e['checkpoint']['fitted']))
        self.assertGreater(e['history_retention']['discarded_points'],0)
        self.assertEqual(v['protocol'],run['resolved']['protocol'])
        self.assertEqual(v['split']['identity'],digest(run['resolved']['manifest']))
        self.assertEqual(v['split']['roles']['train']['records'],4)

    def test_response_omits_weights_observations_and_retains_scalar_hashes(self):
        v=self.view()
        def walk(obj):
            if isinstance(obj,dict):
                self.assertFalse(set(obj)&{'weights','observation_ids'})
                if 'records' in obj:self.assertIs(type(obj['records']),int)
                for child in obj.values():walk(child)
            elif isinstance(obj,list):
                for child in obj:walk(child)
        walk(v)
        # Split records is intentionally a scalar count, never the record values.
        self.assertTrue(all(type(p['records']) is int for p in v['split']['roles'].values()))
        for row in v['metrics']['rows']:self.assertNotIn('observation_ids',row['metric'])
        self.assertNotIn('checkpoint',v['execution'])

    def test_metric_paging_preserves_order_and_all_values(self):
        a=self.view('many');b=self.view('many',offset=25)
        self.assertEqual(a['metrics']['total'],27);self.assertEqual(a['metrics']['next_offset'],25)
        self.assertEqual(len(a['metrics']['rows']),25);self.assertEqual(len(b['metrics']['rows']),2)
        rows=a['metrics']['rows']+b['metrics']['rows']
        self.assertEqual([(r['population'],r['head']['id']) for r in rows],[(role,'head'+str(i)) for role in ('train','validation','test') for i in range(9)])
        self.assertEqual(self.view('many',offset=10**6)['metrics']['rows'],[])

    def test_invalid_attempt_ref_and_bounds_are_atomic(self):
        before=self.w.db.serialize()
        for kwargs in ({'attempt':0},{'attempt':True},{'attempt':2},{'offset':True},{'offset':-1},{'offset':2**53},{'limit':26},{'limit':0}):
            with self.subTest(kwargs=kwargs),self.assertRaises(GraphError):self.view(**kwargs)
        for ref in (dict(self.f['first']['reference'],revision=None),dict(self.f['first']['reference'],kind='report'),dict(self.f['first']['reference'],revision='0'*64)):
            with self.assertRaises(GraphError):run_view(self.w,ref)
        self.assertEqual(before,self.w.db.serialize())

    def test_report_pages_exact_selected_runs_including_unsuccessful_states(self):
        before=self.w.db.serialize()
        with patch('app.transport.run_json_worker',side_effect=AssertionError('Reports never execute')):
            a=report_view(self.w,self.f['report']);b=report_view(self.w,self.f['report'],offset=5)
        self.assertEqual(before,self.w.db.serialize());self.assertEqual(a['execution_behavior'],'read-saved-evidence-only')
        self.assertEqual([r['reference'] for r in a['runs']['rows']+b['runs']['rows']],self.f['selected'])
        self.assertEqual(a['runs']['next_offset'],5);self.assertIsNone(b['runs']['next_offset'])
        self.assertEqual({r['attempt']['status'] for r in a['runs']['rows']},{'completed','cancelled','unavailable','failed'})

    def test_page_does_not_project_skipped_report_rows(self):
        import app.run_views as module
        with patch.object(module,'run_view',wraps=module.run_view) as called:
            report_view(self.w,self.f['report'],offset=5,limit=1)
        self.assertEqual(called.call_count,1);self.assertEqual(called.call_args.args[1],self.f['selected'][5])
        for args in ({'limit':11},{'limit':True},{'offset':-1}):
            with self.assertRaises(GraphError):report_view(self.w,self.f['report'],**args)

    def test_new_heads_do_not_change_a_saved_recipe_or_historical_run(self):
        before=report_view(self.w,self.f['report'])
        self.w.select_spec('run','weighted-run',self.f['retry']['reference']['revision'])
        self.assertEqual(report_view(self.w,self.f['report']),before)

    def test_rerun_view_is_labelled_without_training_or_fake_saved_rows(self):
        report=self.w.get_spec('report',self.f['report']['id'],self.f['report']['revision'])
        report.pop('revision');report.update(id='rerun-view',mode='rerun',runs=[],experiment=self.f['first']['record']['selection']['experiment'])
        ref=self.w.put_spec('report',report);before=self.w.db.serialize()
        with patch('app.transport.run_json_worker',side_effect=AssertionError('Explicit execution only')):v=report_view(self.w,ref)
        self.assertEqual(v['execution_behavior'],'explicit-execution-required');self.assertEqual(v['runs']['total'],0)
        self.assertEqual(before,self.w.db.serialize())

    def test_fresh_presentation_import_is_stdlib_only(self):
        r=subprocess.run([sys.executable,'-c',"import sys;from app.run_views import run_view,report_view;assert not any(m in sys.modules for m in ('torch','numpy','PyQt6','nbformat'))"],capture_output=True,text=True,timeout=10)
        self.assertEqual(r.returncode,0,r.stderr)

    def test_cli_exact_presentation_matches_gui_api_without_numerical_imports_or_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'evidence.vtw';self.w.save(path);before=path.read_bytes()
            for command,ref,options,expected in [
                ('run-evidence',self.f['retry']['reference'],['--attempt','1'],self.view('retry',attempt=1)),
                ('report-evidence',self.f['report'],['--offset','5'],report_view(self.w,self.f['report'],offset=5))]:
                script="import sys,vpw_cli;code=vpw_cli.main(sys.argv[1:]);assert not any(m in sys.modules for m in ('torch','numpy','PyQt6','nbformat'));raise SystemExit(code)"
                proc=subprocess.run(['/usr/bin/python3','-c',script,str(path),command,ref['id'],ref['revision'],*options],capture_output=True,text=True,timeout=20)
                self.assertEqual(proc.returncode,0,proc.stderr);self.assertEqual(__import__('json').loads(proc.stdout)['value'],expected)
                self.assertEqual(path.read_bytes(),before)

    def test_cli_bad_attempt_and_report_limits_fail_without_file_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'evidence.vtw';self.w.save(path);before=path.read_bytes()
            import vpw_cli,io,json
            from contextlib import redirect_stdout
            for command,ref,options in [('run-evidence',self.f['first']['reference'],['--attempt','2']),
                                        ('report-evidence',self.f['report'],['--limit','11'])]:
                out=io.StringIO()
                with redirect_stdout(out):code=vpw_cli.main([str(path),command,ref['id'],ref['revision'],*options])
                self.assertEqual(code,1);self.assertFalse(json.loads(out.getvalue())['ok']);self.assertEqual(path.read_bytes(),before)
