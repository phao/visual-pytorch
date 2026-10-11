"""Prospective experiment pages are exact, read-only and use the existing grammar."""
import copy
import random
import subprocess
import sys
import unittest
from unittest.mock import patch
from app.authoring import begin,check,save,experiment_preview
from app.core import Workspace,GraphError,default_model,default_mlp
from app.task_execution import training_protocol
from test_authoring import measurements


def fixture(workspace):
    models=[]
    for m in (default_model('Linear',inputs=1),default_mlp('Tanh MLP',inputs=1,hidden=(3,))):
        workspace.put(m);models.append(workspace.freeze_model(m['id']))
    pairs=[]
    for i in range(2):
        d=measurements('dataset-'+str(i));d['name']='Measured observations '+str(i)
        d['fields'] += [{'id':'subject','dtype':'string','shape':[]},{'id':'time','dtype':'int64','shape':[]}]
        for n,row in enumerate(d['records']):row['values'].update(subject='s'+str(n//2),time=n);row['values']['y'][0]+=i
        t={'version':1,'id':'task-'+str(i),'name':'Measured regression '+str(i),
           'inputs':[{'port':'input','field':'x','transforms':[{'kind':'standardize'}]}],
           'heads':[{'id':'prediction','port':'output','field':'y','loss':'mse','weight':1.,'transforms':[]}]}
        pairs.append({'id':'pair-'+str(i),'dataset':workspace.put_spec('dataset',d),'task':workspace.put_spec('task',t)})
    return {'version':1,'id':'comparison','name':'Measured comparison',
        'question':'How do these models compare on two independently labelled datasets?',
        'models':models,'data_tasks':pairs,
        'splits':[{'id':'row-split','protocol':{'method':'row','seed':3,'fractions':[.5,.25,.25]}}],
        'initialization_seeds':[7,13],
        'protocols':[{'id':'adam','protocol':training_protocol(epochs=None,updates=3,batch_size=2,evaluate_every=1)}],
        'selection':{'kind':'crossed','pairs':[],'exclusions':[]}}


class ExperimentAuthoringTests(unittest.TestCase):
    def setUp(self):
        self.w=Workspace();self.addCleanup(self.w.db.close)
        self.addCleanup(lambda:self.w.history.directory.cleanup() if self.w.history.directory else None)
        self.e=fixture(self.w)

    def draft(self):
        d=begin(self.w,'experiment',sid=self.e['id']);d['document']=copy.deepcopy(self.e);return d

    def test_preview_is_prospective_readonly_rng_neutral_without_fit_or_ram_admission(self):
        d=self.draft();before=self.w.db.serialize();rng=random.getstate();dirty=self.w.dirty
        with patch('app.datasets._prepare_column',side_effect=AssertionError('No fitting')),patch('app.memory.available_memory',side_effect=AssertionError('No RAM admission')):
            result=experiment_preview(self.w,d)
        page=result['preview'];self.assertEqual(result['total_combinations_exact'],'8')
        self.assertEqual(page['experiment'],check(self.w,d));self.assertEqual(len(page['candidates']),8)
        self.assertEqual({r['status'] for r in page['candidates']},{'compatible'})
        self.assertEqual(before,self.w.db.serialize());self.assertEqual(rng,random.getstate());self.assertEqual(dirty,self.w.dirty)
        self.assertEqual(self.w.list_specs('experiment',True),[]);self.assertEqual(self.w.list_specs('run',True),[])
        self.assertEqual(page['experiment'],save(self.w,d))

    def test_paging_is_bounded_and_does_not_enumerate_skipped_combinations(self):
        d=self.draft();d['document']['initialization_seeds']=list(range(20))
        first=experiment_preview(self.w,d);second=experiment_preview(self.w,d,offset=25)
        self.assertEqual(first['total_combinations_exact'],'80');self.assertEqual(len(first['preview']['candidates']),25)
        self.assertEqual(second['preview']['offset'],25);self.assertEqual(second['preview']['next_offset'],50)
        self.assertFalse({r['id'] for r in first['preview']['candidates']} & {r['id'] for r in second['preview']['candidates']})
        with patch('app.experiment_objects.candidate',side_effect=AssertionError('No skipped enumeration')):
            self.assertEqual(experiment_preview(self.w,d,offset=10**10)['preview']['candidates'],[])
        for args in ({'limit':0},{'limit':26},{'limit':True},{'offset':-1},{'offset':True},{'offset':2**53}):
            with self.subTest(args=args),self.assertRaises(GraphError):experiment_preview(self.w,d,**args)

    def test_pairs_and_exclusion_reasons_remain_visible_not_filtered(self):
        d=self.draft();e=d['document'];pair={'model':e['models'][0],'data_task':e['data_tasks'][0]['id']}
        e['selection']={'kind':'paired','pairs':[pair],'exclusions':[pair|{'reason':'Planned negative control'}]}
        rows=experiment_preview(self.w,d)['preview']['candidates']
        self.assertEqual(len(rows),8);self.assertEqual({r['status'] for r in rows},{'excluded'})
        self.assertEqual(rows[0]['reason'],'Planned negative control');self.assertIn('pairing',rows[-1]['reason'])

    def test_shape_and_group_field_mismatches_show_incompatible_candidates(self):
        d=self.draft();m=default_model('Wrong width',inputs=2);self.w.put(m);d['document']['models'].append(self.w.freeze_model(m['id']))
        d['document']['splits'].append({'id':'bad-group','protocol':{'method':'group','field':'absent','seed':3,'fractions':[.5,.25,.25]}})
        rows=experiment_preview(self.w,d)['preview']['candidates'];self.assertEqual(len(rows),24)
        self.assertEqual({r['status'] for r in rows},{'compatible','incompatible'})
        self.assertTrue(all(r['reason'] for r in rows if r['status']=='incompatible'))

    def test_all_split_methods_use_explicit_fields_and_actual_ids(self):
        d=self.draft();d['document']['splits'] += [
            {'id':'group','protocol':{'method':'group','field':'subject','seed':5,'fractions':[.5,.25,.25]}},
            {'id':'time','protocol':{'method':'temporal','field':'time','fractions':[.5,.25,.25]}},
            {'id':'external','protocol':{'method':'external','roles':{'train':['row3','row1','row2','row0'],'validation':['row4','row5'],'test':['row6']},'excluded':['row7']}}]
        # Two models x two datasets x four splits x two seeds = 32, paged.
        rows=experiment_preview(self.w,d)['preview']['candidates']+experiment_preview(self.w,d,offset=25)['preview']['candidates']
        self.assertEqual(len(rows),32);self.assertEqual({r['status'] for r in rows},{'compatible'})
        for row in rows:
            manifest=row['manifest'];self.assertEqual(manifest['dataset_revision'],row['references']['dataset']['revision'])
            if row['coordinates']['split']=='external':self.assertEqual(manifest['roles']['train'],['row3','row1','row2','row0']);self.assertEqual(manifest['excluded'],['row7'])
            if row['coordinates']['split']=='time':self.assertEqual(manifest['roles']['train'],['row0','row1','row2','row3'])

    def test_axis_order_and_independent_seeds_resolve_without_alphabetical_sort(self):
        d=self.draft();e=d['document'];e['models'].reverse();e['data_tasks'].reverse();e['initialization_seeds']=[13,7]
        p=e['protocols'][0]['protocol'];p['seeds']={'initialization':180,'shuffle':29};p['shuffle']=False
        p['budget']={'epochs':4,'updates':3,'examples':7,'seconds':2.5};p['optimizer'].update(lr=.01,betas=[.8,.95],eps=1e-7,weight_decay=.01)
        first=experiment_preview(self.w,d)['preview']['candidates'][0]
        self.assertEqual(first['coordinates']['model'],e['models'][0]);self.assertEqual(first['coordinates']['data_task'],'pair-1')
        self.assertEqual(first['coordinates']['initialization'],13);self.assertEqual(first['protocol']['seeds'],{'initialization':13,'shuffle':29})
        self.assertEqual(first['protocol']['budget'],p['budget']);self.assertEqual(first['protocol']['optimizer'],p['optimizer']);self.assertFalse(first['protocol']['shuffle'])

    def test_new_dependency_heads_and_archive_do_not_change_exact_preview(self):
        d=self.draft();before=experiment_preview(self.w,d);ref=d['document']['data_tasks'][0]['dataset']
        data=self.w.get_spec('dataset',ref['id']);data.pop('revision');data['records'][0]['values']['x'][0]=999;self.w.put_spec('dataset',data)
        self.w.archive_spec('dataset',ref['id'])
        self.assertEqual(experiment_preview(self.w,d),before)

    def test_historical_base_and_stale_head_are_not_silently_substituted(self):
        ref=save(self.w,self.draft());edit=begin(self.w,'experiment',base=ref);edit['document']['name']='Another';new=save(self.w,edit)
        old=begin(self.w,'experiment',base=ref);self.assertEqual(old['expected_head'],new['revision'])
        self.assertEqual(old['document']['name'],self.e['name']);self.assertEqual(experiment_preview(self.w,old)['preview']['experiment'],ref)
        edit=begin(self.w,'experiment',base=new);edit['document']['name']='Concurrent';save(self.w,edit)
        before=self.w.db.serialize()
        with self.assertRaisesRegex(GraphError,'head changed'):experiment_preview(self.w,old)
        self.assertEqual(before,self.w.db.serialize())

    def test_wrong_workspace_and_wrong_kind_fail_before_preview_or_writes(self):
        d=self.draft();other=Workspace();self.addCleanup(other.db.close);before=other.db.serialize()
        with self.assertRaisesRegex(GraphError,'another workspace'):experiment_preview(other,d)
        self.assertEqual(other.db.serialize(),before)
        ref=self.e['data_tasks'][0]['dataset']
        with self.assertRaisesRegex(GraphError,'experiment draft'):experiment_preview(self.w,begin(self.w,'dataset',base=ref))

    def test_invalid_selection_and_missing_dependencies_reject_atomically(self):
        for change in [lambda e:e.update(question=''),lambda e:e['models'].append(copy.deepcopy(e['models'][0])),
                lambda e:e.update(initialization_seeds=[7,7]),lambda e:e['data_tasks'][0]['task'].update(revision='0'*64),
                lambda e:e['selection'].update(kind='paired'),lambda e:e['selection']['exclusions'].append({'model':e['models'][0],'data_task':'absent','reason':'Visible reason'}),
                lambda e:e['protocols'][0]['protocol']['budget'].update(epochs=None,updates=None,examples=None,seconds=None),
                lambda e:e['protocols'][0]['protocol']['optimizer'].update(betas=[.9,1.]),
                lambda e:e['splits'][0]['protocol'].update(fractions=[.6,.2,.3]),
                lambda e:e.update(selection={'kind':'crossed','pairs':[{'model':e['models'][0],'data_task':'pair-0'}],'exclusions':[]})]:
            d=self.draft();change(d['document']);retained=copy.deepcopy(d);before=self.w.db.serialize()
            with self.subTest(document=d['document']),self.assertRaises(GraphError):experiment_preview(self.w,d)
            self.assertEqual(d,retained);self.assertEqual(before,self.w.db.serialize())

    def test_external_missing_or_duplicate_ids_fail_in_the_right_scope(self):
        d=self.draft();d['document']['splits']=[{'id':'external','protocol':{'method':'external',
            'roles':{'train':['row0'],'validation':[],'test':['absent']},'excluded':['row1','row2','row3','row4','row5','row6','row7']}}]
        rows=experiment_preview(self.w,d)['preview']['candidates'];self.assertEqual({r['status'] for r in rows},{'incompatible'})
        d['document']['splits'][0]['protocol']['roles']['test']=['row0']
        with self.assertRaisesRegex(GraphError,'Duplicate'):experiment_preview(self.w,d)

    def test_large_count_has_an_exact_decimal_transport_representation(self):
        with patch('app.authoring.preview_experiment',return_value={'total_combinations':2**53+1}):
            result=experiment_preview(self.w,self.draft())
        self.assertEqual(result['total_combinations_exact'],'9007199254740993')

    def test_fresh_preview_import_does_not_load_torch_qt_numpy(self):
        result=subprocess.run([sys.executable,'-c',"import sys;from app.authoring import experiment_preview;assert not any(k in sys.modules for k in ('torch','numpy','PyQt6'))"],capture_output=True,text=True,timeout=10)
        self.assertEqual(result.returncode,0,result.stderr)
