"""Equivalent read-scoped reuse, with direct uncached candidate as reference."""
import copy,random,unittest
from unittest.mock import patch
from app import experiment_objects as eo
from app.core import Workspace,GraphError,default_model
from app.evidence import digest
from test_experiment_authoring import fixture

class PreviewResolutionTests(unittest.TestCase):
    def setUp(self):
        self.w=Workspace();self.addCleanup(self.w.db.close)
        self.addCleanup(lambda:self.w.history.directory.cleanup() if self.w.history.directory else None)
        self.e=fixture(self.w);self.e['initialization_seeds']=list(range(10))
        p=copy.deepcopy(self.e['protocols'][0]);p['id']='second';p['protocol']['shuffle']=False;p['protocol']['seeds']['shuffle']=29;self.e['protocols'].append(p)
        self.ref=self.w.put_spec('experiment',self.e);self.e=self.w.get_spec('experiment',self.ref['id'],self.ref['revision'])
    def preview(self,**kw):return eo.preview_experiment(self.e,self.w.get_spec,**kw)
    def reference(self,rows):return [eo.candidate(self.e,row['coordinates'],self.w.get_spec) for row in rows]
    def test_all_seed_protocol_model_dataset_coordinates_equal_direct_uncached_candidates(self):
        p=self.preview(limit=100);self.assertEqual(p['total_combinations'],80)
        self.assertEqual(p['candidates'],self.reference(p['candidates']))
        self.assertEqual([r['protocol']['seeds']['initialization'] for r in p['candidates'][:4]],[0,0,1,1])
        self.assertEqual([r['coordinates']['protocol'] for r in p['candidates'][:4]],['adam','second','adam','second'])
    def test_consecutive_replicates_load_three_dependencies_and_prepare_once(self):
        with patch.object(self.w,'get_spec',wraps=self.w.get_spec) as get,patch.object(eo,'validate_model',wraps=eo.validate_model) as model,patch.object(eo,'validate_task',wraps=eo.validate_task) as task,patch.object(eo,'make_split',wraps=eo.make_split) as split,patch.object(eo,'validate_experiment',wraps=eo.validate_experiment) as experiment:
            p=self.preview(limit=20)
        self.assertEqual(len(p['candidates']),20);self.assertEqual(get.call_count,3)
        self.assertEqual(model.call_count,1);self.assertEqual(task.call_count,1);self.assertEqual(split.call_count,1);self.assertEqual(experiment.call_count,1)
    def test_public_direct_candidate_always_fully_validates_and_has_no_context_bypass(self):
        c=self.preview(limit=1)['candidates'][0]['coordinates']
        with patch.object(eo,'validate_experiment',wraps=eo.validate_experiment) as validate,patch.object(eo,'make_split',wraps=eo.make_split) as split:
            a=eo.candidate(self.e,c,self.w.get_spec);b=eo.candidate(self.e,c,self.w.get_spec)
        self.assertEqual(a,b);self.assertEqual(validate.call_count,2);self.assertEqual(split.call_count,2)
        with self.assertRaises(TypeError):eo.candidate(self.e,c,self.w.get_spec,context=object())
    def test_rows_have_independent_manifest_protocol_coordinate_reference_copies(self):
        p=self.preview(limit=3);a,b=p['candidates'][:2];before=copy.deepcopy(b)
        a['manifest']['roles']['train'].clear();a['manifest']['protocol']['seed']=900
        a['protocol']['optimizer']['lr']=900;a['coordinates']['model']['id']='changed';a['references']['dataset']['id']='changed'
        self.assertEqual(b,before);self.assertEqual(self.preview(limit=3)['candidates'][1],before)
    def test_every_split_method_and_failure_reason_equal_direct_reference(self):
        e=copy.deepcopy(self.e);e.pop('revision');e['splits'] += [
            {'id':'group','protocol':{'method':'group','field':'subject','seed':5,'fractions':[.5,.25,.25]}},
            {'id':'time','protocol':{'method':'temporal','field':'time','fractions':[.5,.25,.25]}},
            {'id':'external','protocol':{'method':'external','roles':{'train':['row3','row1','row2','row0'],'validation':['row4','row5'],'test':['row6']},'excluded':['row7']}},
            {'id':'bad-group','protocol':{'method':'group','field':'absent','seed':5,'fractions':[.5,.25,.25]}}]
        self.ref=self.w.put_spec('experiment',e);self.e=self.w.get_spec('experiment',self.ref['id'],self.ref['revision'])
        p=self.preview(limit=1000);self.assertEqual(len(p['candidates']),400)
        self.assertEqual(p['candidates'],self.reference(p['candidates']))
        self.assertEqual({r['status'] for r in p['candidates']},{'compatible','incompatible'})
    def test_bad_compatibility_is_not_memoized_or_filtered(self):
        e=copy.deepcopy(self.e);e.pop('revision');m=default_model('Wrong width',inputs=2);self.w.put(m);e['models']=[self.w.freeze_model(m['id'])]
        self.ref=self.w.put_spec('experiment',e);self.e=self.w.get_spec('experiment',self.ref['id'],self.ref['revision'])
        with patch.object(eo,'validate_task',wraps=eo.validate_task) as task,patch.object(eo,'make_split',wraps=eo.make_split) as split:p=self.preview(limit=5)
        self.assertEqual(task.call_count,5);self.assertEqual(split.call_count,0)
        self.assertEqual(p['candidates'],self.reference(p['candidates']));self.assertTrue(all(r['status']=='incompatible' for r in p['candidates']))
    def test_exclusions_do_not_load_dependencies_and_ordered_pairing_matches_reference(self):
        e=copy.deepcopy(self.e);e.pop('revision');pair={'model':e['models'][0],'data_task':'pair-0'}
        e['selection']={'kind':'paired','pairs':[pair],'exclusions':[pair|{'reason':'Planned exclusion'}]}
        self.ref=self.w.put_spec('experiment',e);self.e=self.w.get_spec('experiment',self.ref['id'],self.ref['revision'])
        with patch.object(self.w,'get_spec',side_effect=AssertionError('Excluded does not load')):p=self.preview(limit=100)
        self.assertEqual(p['candidates'],self.reference(p['candidates']));self.assertEqual({r['status'] for r in p['candidates']},{'excluded'})
    def test_substituted_revision_and_transient_resolver_failures_remain_explicit(self):
        calls=[]
        def substituted(kind,sid,rev):
            calls.append(kind);d=self.w.get_spec(kind,sid,rev);d['revision']='0'*64;return d
        p=eo.preview_experiment(self.e,substituted,limit=3)
        self.assertEqual(calls,['model']*3);self.assertTrue(all('substituted' in r['reason'] for r in p['candidates']))
        n=[0]
        def transient(kind,sid,rev):
            if kind=='dataset':
                n[0]+=1
                if n[0]==1:raise GraphError('Transient source failure')
            return self.w.get_spec(kind,sid,rev)
        p=eo.preview_experiment(self.e,transient,limit=3)
        self.assertEqual([r['status'] for r in p['candidates']],['incompatible','compatible','compatible']);self.assertEqual(n[0],2)
    def test_new_calls_revalidate_corrupted_sources_and_cannot_reuse_old_workspace_or_head(self):
        before=self.preview(limit=3);ref=self.e['data_tasks'][0]['dataset']
        d=self.w.get_spec('dataset',ref['id']);d.pop('revision');d['records'][0]['values']['x'][0]=999;self.w.put_spec('dataset',d)
        self.assertEqual(self.preview(limit=3),before)
        other=Workspace();self.addCleanup(other.db.close)
        p=eo.preview_experiment(self.e,other.get_spec,limit=3);self.assertTrue(all(r['status']=='incompatible' for r in p['candidates']))
        with patch.object(self.w,'get_spec',side_effect=GraphError('Source disappeared')):p=self.preview(limit=3)
        self.assertTrue(all(r['reason']=='Source disappeared' for r in p['candidates']))
    def test_single_slots_drop_old_documents_before_replacements_and_do_not_grow_with_page(self):
        context=eo._PreviewResolution(self.e,self.w.get_spec);visited=[]
        def resolve(kind,sid,rev):
            self.assertNotIn(kind,context.documents);visited.append((kind,sid,rev));return self.w.get_spec(kind,sid,rev)
        context.resolve=resolve
        for pair in self.e['data_tasks']*5:
            for ref in (self.e['models'][0],pair['dataset'],pair['task']):context.get(ref)
            self.assertLessEqual(len(context.documents),3)
            self.assertEqual({kind:row[0] for kind,row in context.documents.items()},
                {ref['kind']:(ref['id'],ref['revision']) for ref in (self.e['models'][0],pair['dataset'],pair['task'])})
        self.assertEqual(len(visited),21)
    def test_sql_history_dirty_rng_and_outside_page_stay_unchanged(self):
        sql=self.w.db.serialize();history=copy.deepcopy(self.w.history.summary());dirty=self.w.dirty;rng=random.getstate()
        with patch('app.datasets._prepare_column',side_effect=AssertionError('No fitting')):self.preview(limit=25)
        with patch.object(eo,'_candidate',side_effect=AssertionError('No skipped enumeration')):self.assertEqual(self.preview(offset=2**53-1)['candidates'],[])
        self.assertEqual(self.w.db.serialize(),sql);self.assertEqual(self.w.history.summary(),history)
        self.assertEqual(self.w.dirty,dirty);self.assertEqual(random.getstate(),rng)
