"""Read actual saved fits; curve presentation never fits or executes again."""
import copy
import json
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from app.core import Workspace, GraphError
from app.run_curves import curve_view
from evidence_fixture import populate


class RunCurveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.w = Workspace(); cls.f = populate(cls.w, sys.executable)

    @classmethod
    def tearDownClass(cls):
        if cls.w.history.directory: cls.w.history.directory.cleanup()
        cls.w.db.close()

    def view(self, key='first', **options):
        return curve_view(self.w, self.f[key]['reference'], **options)

    def test_every_plotted_metric_counter_and_checkpoint_is_exact_saved_evidence(self):
        page = self.view(head='regression')
        result = self.f['first']['record']['attempts'][0]['result']
        self.assertEqual(page['scope'], 'read-saved-curve-only')
        self.assertEqual(page['execution'], 'no-fitting-or-training')
        self.assertEqual(page['final_checkpoint'], result['checkpoint_id'])
        self.assertEqual(page['retention'], result['history_retention'])
        self.assertEqual(page['points']['total'], len(result['history']))
        self.assertEqual(page['head']['transforms'], [{'kind':'affine','scale':.5,'offset':-1.}])
        self.assertIn('<svg', page['source_unit'])
        for ordinal, (row, point) in enumerate(zip(page['points']['rows'], result['history'])):
            self.assertEqual(row['ordinal'], ordinal)
            for key in ('update','attempted_batches','examples_processed','checkpoint_id','timing','mode'):
                self.assertEqual(row[key], point[key])
            self.assertEqual(set(row['metrics']), {'train','validation'})
            for role, metric in row['metrics'].items():
                original = point['metrics'][role]['heads']['regression']
                self.assertEqual(metric, {k:v for k,v in original.items() if k != 'observation_ids'})

    def test_axes_cover_all_retained_points_and_do_not_change_between_pages(self):
        full = self.view(); first = self.view(limit=1); second = self.view(offset=1,limit=1)
        self.assertEqual(first['x_range'], second['x_range'])
        self.assertEqual(first['y_range'], second['y_range'])
        self.assertEqual(first['points']['next_offset'], 1)
        self.assertIsNone(second['points']['next_offset'])
        self.assertEqual(first['points']['rows']+second['points']['rows'], full['points']['rows'])
        self.assertGreater(full['retention']['discarded_points'], 0)
        for row in full['points']['rows']:
            for metric in row['metrics'].values():
                if metric['value'] is not None: self.assertLessEqual(metric['value'], full['y_range'][1])

    def test_heads_page_in_declared_order_and_selected_head_can_be_outside_page(self):
        rows = []; offset = 0
        while True:
            page = self.view('many',head='head8',head_offset=offset,head_limit=3)
            self.assertEqual(page['head']['id'], 'head8'); rows.extend(page['heads']['rows'])
            if page['heads']['next_offset'] is None: break
            offset = page['heads']['next_offset']
        self.assertEqual([h['id'] for h in rows], ['head'+str(i) for i in range(9)])
        self.assertEqual(self.view('many')['head']['id'], 'head0')

    def test_failed_latest_attempt_never_borrows_previous_curve_or_checkpoint(self):
        page = self.view('retry'); self.assertEqual(page['attempt']['number'], 2)
        self.assertEqual(page['attempt']['status'], 'failed')
        self.assertEqual(page['points']['total'], 0); self.assertEqual(page['points']['rows'], [])
        self.assertIsNone(page['final_checkpoint']); self.assertIsNone(page['retention'])
        previous = self.view('retry',attempt=1)
        self.assertEqual(previous['points'], self.view()['points'])
        self.assertEqual(previous['reference'], self.f['retry']['reference'])

    def test_cancelled_partial_curve_and_all_masked_unavailable_values_remain_visible(self):
        cancelled = self.view('cancelled')
        self.assertEqual(cancelled['attempt']['status'], 'cancelled')
        self.assertGreater(cancelled['points']['total'], 0)
        unavailable = self.view('zero')
        self.assertEqual(unavailable['attempt']['status'], 'unavailable')
        self.assertGreater(unavailable['points']['total'], 0)
        for row in unavailable['points']['rows']:
            for metric in row['metrics'].values():
                self.assertIsNone(metric['value']); self.assertEqual(metric['status'], 'unavailable')
                self.assertEqual(metric['denominator'], 0)
        self.assertEqual(unavailable['y_range'], [0,1.])

    def test_exclusion_keeps_task_context_but_does_not_fabricate_points(self):
        page = self.view('excluded')
        self.assertEqual(page['attempt']['stage'], 'compatibility')
        self.assertEqual(page['attempt']['error']['message'], 'Explicit negative control')
        self.assertEqual(page['points']['rows'], [])
        self.assertEqual(page['head']['id'], 'classification')

    def test_reader_is_pure_owned_and_has_no_training_or_split_regeneration(self):
        before=self.w.db.serialize(); history=copy.deepcopy(self.w.history.summary()); dirty=self.w.dirty; rng=random.getstate()
        with patch('app.transport.run_json_worker',side_effect=AssertionError('No execution')), patch('app.experiment_objects.make_split',side_effect=AssertionError('Captured splits only')):
            page=self.view(); original=copy.deepcopy(page)
        page['points']['rows'][0]['metrics']['train']['value']=900
        page['head']['transforms'].clear(); page['heads']['rows'][0]['id']='wrong'
        self.assertEqual(self.view(),original)
        self.assertEqual(self.w.db.serialize(),before);self.assertEqual(self.w.history.summary(),history)
        self.assertEqual(self.w.dirty,dirty);self.assertEqual(random.getstate(),rng)

    def test_invalid_identity_attempt_head_and_offsets_fail_without_mutation(self):
        before=self.w.db.serialize()
        for options in ({'attempt':True},{'attempt':0},{'attempt':2},{'head':'foreign'},{'head':True},
                        {'offset':True},{'offset':-1},{'offset':2**53},{'limit':101},{'limit':0},
                        {'head_offset':True},{'head_offset':-1},{'head_limit':26},{'head_limit':False}):
            with self.subTest(options=options),self.assertRaises(GraphError):self.view(**options)
        for ref in (self.f['first']['reference']|{'revision':None}, self.f['first']['reference']|{'kind':'report'},
                    self.f['first']['reference']|{'revision':'0'*64}):
            with self.assertRaises(GraphError):curve_view(self.w,ref)
        self.assertEqual(self.w.db.serialize(),before)

    def test_past_end_pages_keep_valid_identity_without_fabricating_a_zero_sample(self):
        page=self.view(offset=2**53-1,head_offset=100)
        self.assertEqual(page['points']['rows'],[]);self.assertIsNone(page['points']['next_offset'])
        self.assertEqual(page['heads']['rows'],[]);self.assertEqual(page['head']['id'],'classification')
        self.assertIsNotNone(page['final_checkpoint'])

    def test_historical_head_changes_do_not_change_curve_identity_or_values(self):
        before=self.view(); current=self.w.get_spec('run',self.f['retry']['reference']['id'])
        self.w.select_spec('run',current['id'],current['revision'])
        self.assertEqual(self.view(),before)

    def test_cli_matches_native_projection_without_changing_file_or_loading_numerical_stack(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'curves.vtw';self.w.save(path);before=path.read_bytes();mtime=path.stat().st_mtime_ns
            ref=self.f['retry']['reference']
            script='import sys,vpw_cli;code=vpw_cli.main(sys.argv[1:]);assert not any(m in sys.modules for m in ("torch","numpy","PyQt6","nbformat"));raise SystemExit(code)'
            proc=subprocess.run(['/usr/bin/python3','-c',script,str(path),'run-curve',ref['id'],ref['revision'],
                '--attempt','1','--head','classification','--limit','1','--head-limit','1'],capture_output=True,text=True,timeout=20)
            self.assertEqual(proc.returncode,0,proc.stderr)
            self.assertEqual(json.loads(proc.stdout)['value'],self.view('retry',attempt=1,head='classification',limit=1,head_limit=1))
            self.assertEqual(path.read_bytes(),before);self.assertEqual(path.stat().st_mtime_ns,mtime)

    def test_payload_has_no_weights_fitted_tensors_observation_ids_or_test_curve(self):
        page=self.view(head='classification')
        def walk(value):
            if isinstance(value,dict):
                self.assertFalse(set(value)&{'weights','observation_ids','fitted'})
                for v in value.values():walk(v)
            elif isinstance(value,list):
                for v in value:walk(v)
        walk(page)
        self.assertTrue(all(set(row['metrics'])=={'train','validation'} for row in page['points']['rows']))
        self.assertTrue(all('accuracy' in row['metrics']['train'] for row in page['points']['rows']))


if __name__=='__main__':unittest.main()
