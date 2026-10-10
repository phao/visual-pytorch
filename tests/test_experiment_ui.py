"""Pure GUI boundary contracts: capture exact science, not mutable editor state."""
import copy
import json
import unittest
from unittest.mock import patch
from app.core import GraphError, Workspace
from app.experiment_ui import capture, index, read_report, result_report
from app.experiment_objects import execute_candidate, object_reference, preview_experiment
from test_experiment_objects import experiment_workspace


class ExperimentUITests(unittest.TestCase):
    def setUp(self):
        self.w,self.e=experiment_workspace();self.addCleanup(self.w.db.close)
        self.ref=object_reference('experiment',self.e)
        self.row=preview_experiment(self.e,self.w.get_spec)['candidates'][0]
    def capture(self, **kw):return capture(self.w,self.ref,self.row['coordinates'],'trial','Trial',**kw)
    def test_index_is_alphabetical_all_exact_revisions_and_readonly(self):
        edited=copy.deepcopy(self.e);edited.pop('revision');edited['name']='Other experiment'
        self.w.put_spec('experiment',edited)
        before=self.w.db.serialize();rows=index(self.w)
        self.assertEqual(self.w.db.serialize(),before)
        experiments=[r for r in rows if r['kind']=='experiment']
        self.assertEqual(len(experiments),2);self.assertEqual(sum(r['selected'] for r in experiments),1)
        self.assertEqual(rows,sorted(rows,key=lambda r:(r['kind'],r['name'].casefold(),r['id'],r['revision'])))
    def test_capture_is_pure_exact_bounded_closure_without_unrelated_data(self):
        before=self.w.db.serialize()
        with patch('app.datasets._prepare_column',side_effect=AssertionError('No fit')):
            snapshot=self.capture()
        self.assertEqual(self.w.db.serialize(),before)
        self.assertEqual(snapshot['experiment'],self.ref)
        self.assertEqual(snapshot['manifest'],self.row['manifest'])
        # Experiment dependencies are complete, but unrelated registry objects are not copied.
        unused=copy.deepcopy(self.w.get_spec('dataset','dataset0'));unused.pop('revision');unused['id']='unused'
        self.w.put_spec('dataset',unused)
        self.assertEqual(snapshot,self.capture())
        self.assertNotIn('unused',[e['document']['id'] for e in snapshot['documents']])
        snapshot['coordinates']['initialization']=999
        self.assertEqual(self.capture()['coordinates']['initialization'],7)
    def test_captured_closure_survives_new_heads_and_reproduces_worker_result(self):
        snapshot=self.capture()
        e=copy.deepcopy(self.e);e.pop('revision');e['initialization_seeds']=[99]
        self.w.put_spec('experiment',e)
        clone=Workspace();self.addCleanup(clone.db.close)
        for entry in snapshot['documents']:clone.put_spec(entry['kind'],entry['document'],select=False)
        run=execute_candidate(clone,snapshot['experiment'],snapshot['coordinates'],'trial','Trial',
            saved_manifest=snapshot['manifest'],python='/definitely/missing/python')
        self.assertEqual(run['record']['selection']['experiment'],self.ref)
        self.assertEqual(run['record']['resolved']['manifest'],snapshot['manifest'])
        self.assertEqual(run['record']['attempts'][-1]['status'],'failed')
        self.w.put_spec('run',run['record'])
    def test_implicit_head_invalid_name_and_retry_are_rejected(self):
        for args in [(dict(self.ref,revision=None),'trial','Trial',False),
                     (self.ref,'','Trial',False),(self.ref,'trial','Trial','yes'),
                     (self.ref,'trial','Trial',True)]:
            with self.assertRaises(GraphError):capture(self.w,args[0],self.row['coordinates'],args[1],args[2],retry=args[3])
    def test_retry_freezes_original_manifest_and_longest_history_after_old_head(self):
        run=execute_candidate(self.w,self.ref,self.row['coordinates'],'trial','Trial',python='/missing/python')
        next_run=execute_candidate(self.w,self.ref,self.row['coordinates'],'trial','Trial',retry=True,python='/missing/python')
        self.w.select_spec('run','trial',run['reference']['revision'])
        with patch('app.experiment_objects.make_split',side_effect=AssertionError('Never regenerate saved split')):
            snapshot=self.capture(retry=True)
        retained=[e['document'] for e in snapshot['documents'] if e['kind']=='run']
        self.assertEqual(retained,[next_run['record']]);self.assertEqual(snapshot['manifest'],run['record']['resolved']['manifest'])
        with self.assertRaises(GraphError):self.capture()
        changed=dict(self.row['coordinates'],initialization=13)
        with self.assertRaises(GraphError):capture(self.w,self.ref,changed,'trial','Trial',retry=True)
    def test_saved_report_reads_failed_evidence_without_worker(self):
        run=execute_candidate(self.w,self.ref,self.row['coordinates'],'trial','Trial',python='/missing/python')
        ref=result_report(self.w,[run['reference']],sid='report',name='Saved trial',question='What failed?')
        before=self.w.db.serialize()
        with patch('app.transport.run_json_worker',side_effect=AssertionError('Viewing must not train')):
            report=read_report(self.w,ref)
        self.assertEqual(self.w.db.serialize(),before)
        self.assertEqual(report['execution_behavior'],'read-saved-evidence-only')
        self.assertEqual(report['runs'][0]['status'],'failed');self.assertEqual(report['runs'][0]['metrics'],{})
        self.assertEqual(report['runs'][0]['reference'],run['reference'])
    def test_excluded_candidate_is_captured_honestly_without_manifest(self):
        e=copy.deepcopy(self.e);e.pop('revision')
        e['selection']['exclusions']=[{'model':e['models'][0],'data_task':'pair0','reason':'negative control'}]
        ref=self.w.put_spec('experiment',e)
        snapshot=capture(self.w,ref,self.row['coordinates'],'trial','Trial')
        self.assertEqual(snapshot['status'],'excluded');self.assertEqual(snapshot['reason'],'negative control')
        self.assertIsNone(snapshot['manifest'])
