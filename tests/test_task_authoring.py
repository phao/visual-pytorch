"""Task authoring preserves computation order and exact context without fitting."""
import copy
import random
import subprocess
import sys
import unittest
from unittest.mock import patch

from app.authoring import begin, check, save, task_context, task_compatibility
from app.core import GraphError, Workspace, default_model


def multi_fixture():
    m = default_model('Two measured branches', inputs=1)
    m['graph'] = {'nodes': [
        {'id':'left','type':'Input','shape':[1],'axes':['F']},
        {'id':'right','type':'Input','shape':[1]},
        {'id':'classifier','type':'Linear','features':3,'bias':True},
        {'id':'regressor','type':'Linear','features':1,'bias':True},
        {'id':'category','type':'Output'}, {'id':'estimate','type':'Output'}],
        'edges':[{'from':'left','to':'classifier'}, {'from':'classifier','to':'category'},
                 {'from':'right','to':'regressor'}, {'from':'regressor','to':'estimate'}]}
    fields = [{'id':'a','dtype':'float32','shape':[1],'axes':['F']},
        {'id':'b','dtype':'float32','shape':[1],'nullable':True},
        {'id':'label','dtype':'category','shape':[], 'vocabulary':{'version':1,'values':['cold','warm','unused']}},
        {'id':'y','dtype':'float32','shape':[1]}, {'id':'valid','dtype':'bool','shape':[]},
        {'id':'weight','dtype':'float32','shape':[]}]
    d = {'version':1,'id':'measured','name':'Measured branches',
         'storage':{'kind':'inline-records','version':1}, 'fields':fields,
         'records':[{'id':'r'+str(i),'values':{'a':[i-3.5], 'b':None if i==1 else [float(i)],
                    'label':'cold' if i<4 else 'warm', 'y':[2.*i+1.], 'valid':i!=0, 'weight':.5+i}}
                    for i in range(8)]}
    t = {'version':1,'id':'two-heads','name':'Measured objectives',
         'inputs':[{'port':'left','field':'a','transforms':[{'kind':'standardize'},{'kind':'affine','scale':2.,'offset':1.}]},
                   {'port':'right','field':'b','transforms':[{'kind':'impute-mean'},{'kind':'standardize'}]}],
         'heads':[{'id':'classification','port':'category','field':'label','loss':'cross-entropy',
                   'weight':.75,'transforms':[],'mask':'valid','sample_weight':'weight'},
                  {'id':'regression','port':'estimate','field':'y','loss':'mae','weight':1.25,
                   'transforms':[{'kind':'affine','scale':.5,'offset':-1.}],'mask':'valid','sample_weight':'weight'}],
         'description':'Two independent input paths and explicit weighted loss heads.'}
    return m,d,t


def sequence_fixture():
    m = default_model('Padded observations', inputs=1)
    m['graph'] = {'nodes':[{'id':'sequence','type':'Input','shape':[3,2]},
        {'id':'length','type':'Input','shape':[1]}, {'id':'padding-mask','type':'Input','shape':[3]},
        {'id':'join','type':'Concat','sources':['sequence','length','padding-mask']},
        {'id':'linear','type':'Linear','features':1,'bias':True}, {'id':'output','type':'Output'}],
        'edges':[{'from':n,'to':'join'} for n in ('sequence','length','padding-mask')]+
                [{'from':'join','to':'linear'},{'from':'linear','to':'output'}]}
    d = {'version':1,'id':'sequences','name':'Ragged observations',
        'storage':{'kind':'inline-records','version':1},
        'fields':[{'id':'seq','dtype':'float32','shape':[-1,2]}, {'id':'y','dtype':'float32','shape':[1]}],
        'records':[{'id':'s'+str(i),'values':{'seq':[[float(i),i+1.]]*(i%4),'y':[i+1.]}} for i in range(8)]}
    t = {'version':1,'id':'sequence-task','name':'Explicit padding and derived inputs',
        'inputs':[{'port':'sequence','field':'seq','transforms':[{'kind':'pad','length':3,'value':-5.}]},
                  {'port':'length','field':'seq','transforms':[],'derived':'length','length':3},
                  {'port':'padding-mask','field':'seq','transforms':[],'derived':'mask','length':3}],
        'heads':[{'id':'prediction','port':'output','field':'y','loss':'mse','weight':1.,'transforms':[]}]}
    return m,d,t


class TaskAuthoringTests(unittest.TestCase):
    def setUp(self):
        self.w=Workspace(); self.addCleanup(self.w.db.close)
        self.addCleanup(lambda:self.w.history.directory.cleanup() if self.w.history.directory else None)
        self.model,self.dataset,self.task=multi_fixture()
        self.w.put(self.model); self.mref=self.w.freeze_model(self.model['id'])
        self.dref=self.w.put_spec('dataset',self.dataset)

    def draft(self, document=None):
        document=copy.deepcopy(document or self.task)
        draft=begin(self.w,'task',sid=document['id']); draft['document']=document; return draft

    def result(self, draft=None, **kw):
        return task_compatibility(self.w,draft or self.draft(),kw.get('model',self.mref),kw.get('dataset',self.dref))

    def test_context_and_compatibility_are_readonly_rng_neutral_and_deep_owned(self):
        draft=self.draft(); before=self.w.db.serialize(); rng=random.getstate()
        with patch('app.datasets._prepare_column',side_effect=AssertionError('No transform fitting')):
            context=task_context(self.w,self.mref,self.dref); result=self.result(draft)
        self.assertEqual(result['status'],'compatible'); self.assertIsNone(result['reason'])
        self.assertEqual(result['scope'],'interface-only-no-fitting-or-execution')
        self.assertEqual(result['task'],check(self.w,draft))
        self.assertEqual(context['record_count'],8); self.assertNotIn('records',context)
        self.assertEqual([g['id'] for g in context['input_ports']],['left','right'])
        self.assertEqual([g['id'] for g in context['output_ports']],['category','estimate'])
        context['fields'][0]['axes'].append('not-owned'); context['input_ports'][0]['shape'][0]=100
        result['model']['revision']='0'*64
        self.assertEqual(task_context(self.w,self.mref,self.dref)['fields'][0]['axes'],['F'])
        self.assertEqual(before,self.w.db.serialize()); self.assertEqual(rng,random.getstate())
        self.assertEqual(self.mref['revision'],self.w.get_spec('model',self.mref['id'])['revision'])

    def test_task_has_no_implicit_model_dataset_dependencies_and_noop_is_readonly(self):
        draft=self.draft(); ref=save(self.w,draft); frozen=self.w.get_spec('task',ref['id'])
        self.assertNotIn('model',frozen); self.assertNotIn('dataset',frozen)
        self.assertEqual(frozen['inputs'],self.task['inputs']); self.assertEqual(frozen['heads'],self.task['heads'])
        edit=begin(self.w,'task',base=ref); before=self.w.db.serialize()
        self.assertEqual(save(self.w,edit),ref); self.assertEqual(before,self.w.db.serialize())

    def test_new_heads_never_replace_exact_context_and_port_reorder_keeps_binding_meaning(self):
        changed=copy.deepcopy(self.model); changed['graph']['nodes'][:2]=changed['graph']['nodes'][:2][::-1]
        # A already pins packing independently of node presentation order.
        self.w.put(changed); preserved=self.w.freeze_model(changed['id'])
        self.assertEqual([g['id'] for g in task_context(self.w,preserved,self.dref)['input_ports']],['left','right'])
        # Also exercise an intentional contract-order change, not just drawing.
        changed['graph']['input_order']=['right','left']
        self.w.put(changed); newer=self.w.freeze_model(changed['id'])
        d=copy.deepcopy(self.dataset); d['fields'].reverse(); d['records'].reverse()
        new_data=self.w.put_spec('dataset',d)
        old=task_context(self.w,self.mref,self.dref); new=task_context(self.w,newer,new_data)
        self.assertEqual([g['id'] for g in old['input_ports']],['left','right'])
        self.assertEqual([g['id'] for g in new['input_ports']],['right','left'])
        self.assertEqual(old['fields'][0]['id'],'a'); self.assertEqual(new['fields'][0]['id'],'weight')
        self.assertEqual(self.result()['status'],'compatible')
        self.assertEqual(self.result(model=newer,dataset=new_data)['status'],'compatible')
        self.assertEqual(self.task['inputs'][0]['field'],'a')

    def test_historical_task_edit_retains_old_content_and_detects_concurrent_head(self):
        ref=save(self.w,self.draft()); edit=begin(self.w,'task',base=ref)
        edit['document']['heads'][1]['loss']='mse'; new=save(self.w,edit)
        old=begin(self.w,'task',base=ref)
        self.assertEqual(old['expected_head'],new['revision']); self.assertEqual(old['document']['heads'][1]['loss'],'mae')
        self.assertEqual(self.result(old)['status'],'compatible')
        edit=begin(self.w,'task',base=new); edit['document']['name']='Concurrent'; save(self.w,edit)
        before=self.w.db.serialize()
        with self.assertRaisesRegex(GraphError,'head changed'):self.result(old)
        self.assertEqual(before,self.w.db.serialize())

    def test_missing_wrong_kind_and_null_refs_are_errors_not_head_fallback(self):
        before=self.w.db.serialize()
        for ref in (None,{},dict(self.mref,revision=None),dict(self.mref,revision='0'*64),self.dref):
            with self.subTest(reference=ref),self.assertRaises(GraphError):task_context(self.w,ref,self.dref)
            with self.subTest(reference=ref),self.assertRaises(GraphError):self.result(model=ref)
        self.assertEqual(before,self.w.db.serialize())

    def test_compatibility_is_not_available_for_dataset_drafts_or_wrong_workspace(self):
        draft=begin(self.w,'dataset',base=self.dref)
        with self.assertRaisesRegex(GraphError,'task draft'):self.result(draft)
        other=Workspace(); self.addCleanup(other.db.close); before=other.db.serialize()
        with self.assertRaisesRegex(GraphError,'another workspace'):task_compatibility(other,self.draft(),self.mref,self.dref)
        self.assertEqual(other.db.serialize(),before)

    def test_unknown_fields_ports_and_missing_bindings_show_incompatibility(self):
        for change,match in [(lambda t:t['inputs'][0].update(port='renamed'),'port ID'),
                (lambda t:t['inputs'][0].update(field='absent'),'field ID'),
                (lambda t:t['inputs'].pop(),'every model input'),
                (lambda t:t['heads'].pop(),'Every output'),
                (lambda t:t['heads'][0].update(field='absent'),'target field')]:
            draft=self.draft(); change(draft['document']); before=self.w.db.serialize()
            result=self.result(draft); self.assertEqual(result['status'],'incompatible'); self.assertIn(match,result['reason'])
            self.assertEqual(before,self.w.db.serialize())

    def test_shape_and_axes_mismatches_are_not_silently_adapted(self):
        for change,match in [(lambda d:d['fields'][0].update(axes=['C']),'axes'),
                (lambda d:d['fields'][0].update(shape=[2]),'shape'),
                (lambda d:d['fields'][3].update(shape=[2]),'target shape')]:
            d=copy.deepcopy(self.dataset); change(d)
            for row in d['records']:
                if d['fields'][0]['shape']==[2]:row['values']['a']*=2
                if d['fields'][3]['shape']==[2]:row['values']['y']*=2
            ref=self.w.put_spec('dataset',d); result=self.result(dataset=ref)
            self.assertEqual(result['status'],'incompatible'); self.assertIn(match,result['reason'])

    def test_intrinsic_conflicts_are_invalid_not_silent_transform_deletion(self):
        for change in (lambda t:t['inputs'].append(copy.deepcopy(t['inputs'][0])),
                lambda t:t['heads'][0]['transforms'].append({'kind':'standardize'}),
                lambda t:t['inputs'][0].update(derived='mask',length=3),
                lambda t:t['heads'][1].update(weight=0),
                lambda t:t['inputs'][0]['transforms'][1].update(scale=0)):
            draft=self.draft(); change(draft['document']); retained=copy.deepcopy(draft); before=self.w.db.serialize()
            with self.assertRaises(GraphError):self.result(draft)
            self.assertEqual(draft,retained); self.assertEqual(before,self.w.db.serialize())

    def test_category_ontology_sizes_logits_instead_of_observed_class_count(self):
        changed=copy.deepcopy(self.model); changed['graph']['nodes'][2]['features']=2
        self.w.put(changed); mref=self.w.freeze_model(changed['id'])
        result=self.result(model=mref)
        self.assertEqual(result['status'],'incompatible'); self.assertIn('fixed vocabulary',result['reason'])

    def test_masks_and_weights_require_numeric_nonmissing_scalar_or_target_shape(self):
        for key,field in [('mask','label'),('sample_weight','b'),('mask','absent')]:
            draft=self.draft(); draft['document']['heads'][0][key]=field; result=self.result(draft)
            self.assertEqual(result['status'],'incompatible'); self.assertIn('Mask/weight',result['reason'])

    def test_interface_compatibility_does_not_claim_weight_value_or_ram_admission(self):
        d=copy.deepcopy(self.dataset); d['records'][0]['values']['weight']=-1.
        dref=self.w.put_spec('dataset',d)
        with patch('app.memory.available_memory',side_effect=AssertionError('No resource admission')):
            result=self.result(dataset=dref)
        self.assertEqual(result['status'],'compatible'); self.assertEqual(result['scope'],'interface-only-no-fitting-or-execution')

    def test_ragged_pad_length_and_mask_require_the_same_explicit_field_and_length(self):
        m,d,t=sequence_fixture(); self.w.put(m); mref=self.w.freeze_model(m['id']); dref=self.w.put_spec('dataset',d)
        draft=self.draft(t); self.assertEqual(self.result(draft,model=mref,dataset=dref)['status'],'compatible')
        for index,key,value in [(1,'length',2),(2,'field','y')]:
            bad=copy.deepcopy(draft); bad['document']['inputs'][index][key]=value
            result=self.result(bad,model=mref,dataset=dref); self.assertEqual(result['status'],'incompatible')
        bad=copy.deepcopy(draft); bad['document']['inputs'][1]['transforms']=[{'kind':'standardize'}]
        with self.assertRaisesRegex(GraphError,'no transforms'):self.result(bad,model=mref,dataset=dref)

    def test_fresh_process_remains_torch_numpy_qt_free(self):
        probe=subprocess.run([sys.executable,'-c',
            "import sys; import app.authoring; assert not any(k in sys.modules for k in ('torch','numpy','PyQt6'))"],capture_output=True,text=True,timeout=10)
        self.assertEqual(probe.returncode,0,probe.stderr)
