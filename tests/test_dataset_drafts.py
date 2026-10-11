"""Typed pages are projections; hidden data and exact tokens survive edits."""
import copy
import json
import random
import unittest
from unittest.mock import patch

from app import dataset_drafts, science_browser
from app.authoring_sessions import Sessions, _at
from app.core import GraphError, Workspace
from test_authoring import measurements


class DatasetDraftTests(unittest.TestCase):
    def setUp(self):
        self.w=Workspace();self.addCleanup(self.w.db.close)
        self.addCleanup(lambda:self.w.history.directory.cleanup() if self.w.history.directory else None)
        self.owner=object();self.s=Sessions(available_fn=lambda:2**31)
        self.addCleanup(lambda:self.s.close_owner(self.owner))

    def begin(self,document=None):
        ref=self.w.put_spec('dataset',document or measurements())
        header=self.s.open(self.w,self.owner,'dataset',base=ref)
        return header,ref

    def page(self,h,**cursor):return self.s.dataset_page(h['session'],self.owner,h['sequence'],**cursor)
    def edit(self,h,operations):return self.s.dataset_edit(self.w,h['session'],self.owner,h['sequence'],operations)
    def doc(self,h):return self.s._sessions[h['session']]['draft']['document']

    def test_pages_are_exact_owned_bounded_views_with_ordinals_not_persistable_docs(self):
        h,_=self.begin(); before=self.w.db.serialize();rng=random.getstate()
        page=self.page(h,row_offset=2,row_limit=3,field_limit=1)
        self.assertEqual(page['scope'],'owned-dataset-page');self.assertEqual(page['session'],h['session'])
        self.assertEqual((page['field_total'],page['row_total']),(2,8));self.assertEqual(page['next_field_offset'],1)
        self.assertEqual(page['next_row_offset'],5);self.assertEqual([r['id'] for r in page['rows']],['row2','row3','row4'])
        self.assertEqual(page['fields'][0]['id'],'x');self.assertEqual(page['rows'][0]['cells'][0]['text'],'[2.0]')
        for row in page['rows']:
            self.assertEqual(json.loads(row['cells'][0]['text']),_at(self.doc(h),row['cells'][0]['path']))
        self.assertLessEqual(len(science_browser.encoded(page)),65536)
        self.assertEqual(self.w.db.serialize(),before);self.assertEqual(random.getstate(),rng)
        with self.assertRaises(GraphError):self.w.put_spec('dataset',page)

    def test_large_values_are_explicit_targets_without_full_encoding_or_truncation(self):
        doc=measurements();doc['fields'][0]['shape']=[4000]
        for row in doc['records']:row['values']['x']=[1.]*4000
        doc['usage']={'large':'λ'*10000}
        h,_=self.begin(doc);cell=self.doc(h)['records'][0]['values']['x'];dumps=json.dumps
        def guarded(value,*args,**kw):
            self.assertIsNot(value,cell,'Do not stringify an entire hidden tensor for its preview')
            return dumps(value,*args,**kw)
        with patch('app.dataset_drafts.json.dumps',side_effect=guarded):page=self.page(h)
        value=page['rows'][0]['cells'][0];self.assertIsNone(value['text']);self.assertEqual(value['summary']['size'],4000)
        self.assertIs(_at(self.doc(h),value['path']),cell)
        usage=next(v for v in page['metadata'] if v['key']=='usage');self.assertIsNone(usage['text'])
        self.assertEqual(self.doc(h),doc)

    def test_byte_shortening_uses_actual_offsets_and_all_scientific_order_survives(self):
        names=['🔬'*198+str(i) for i in range(6)]
        doc=measurements();doc['fields']=[{'id':fid,'dtype':'string','shape':[]} for fid in names]
        doc['records']=[{'id':'🧪'*197+str(i),'values':{fid:'😀'*100 for fid in names}} for i in range(20)]
        h,_=self.begin(doc);seen=[];offset=0;calls=0
        while True:
            page=self.page(h,row_offset=offset);calls+=1
            self.assertLessEqual(len(science_browser.encoded(page)),65536);seen.extend(row['id'] for row in page['rows'])
            self.assertEqual([field['id'] for field in page['fields']],names)
            if page['next_row_offset'] is None:break
            self.assertGreater(page['next_row_offset'],offset);offset=page['next_row_offset']
        self.assertGreater(calls,1);self.assertEqual(seen,[row['id'] for row in doc['records']])
        self.assertEqual(self.doc(h),doc)

    def test_small_raw_int64_and_vocab_values_never_round_through_javascript(self):
        doc=measurements();doc['fields'].append({'id':'label','dtype':'category','shape':[],
            'vocabulary':{'version':1,'values':[9007199254740993,'text']}})
        for row in doc['records']:row['values']['label']=9007199254740993
        h,_=self.begin(doc);page=self.page(h)
        vocab=next(v for v in page['fields'][2]['properties'] if v['key']=='vocabulary')
        self.assertEqual(vocab['text'],'[9007199254740993,"text"]');self.assertEqual(_at(self.doc(h),vocab['path']),[9007199254740993,'text'])
        self.assertEqual(page['rows'][0]['cells'][2]['text'],'9007199254740993')
        newer=self.edit(h,[{'op':'cell','record':'row0','field':'label','text':'"text"'},
            {'op':'field','field':'label','key':'vocabulary','text':'[9007199254740995,"text"]'}])
        self.assertEqual(self.doc(newer)['fields'][2]['vocabulary']['values'],[9007199254740995,'text'])
        # Remaining observations now use a missing category: validation is strict,
        # while temporary draft edits are permitted and preserve other values.
        with self.assertRaises(GraphError):self.s.check(self.w,newer['session'],self.owner,newer['sequence'])

    def test_structural_edits_preserve_hidden_values_and_prototypes_are_ordinary_ids(self):
        h,_=self.begin();original=copy.deepcopy(self.doc(h));sql=self.w.db.serialize()
        h=self.edit(h,[{'op':'field-add','id':'__proto__'},{'op':'record-add','id':'__proto__'},
            {'op':'cell','record':'row7','field':'__proto__','text':'true'},
            {'op':'field','field':'__proto__','key':'dtype','text':'"bool"'}])
        doc=self.doc(h);self.assertEqual([r['id'] for r in doc['records']],['row'+str(i) for i in range(8)]+['__proto__'])
        self.assertEqual(doc['records'][7]['values']['__proto__'],True);self.assertIsNone(doc['records'][0]['values']['__proto__'])
        self.assertEqual(doc['records'][8]['values'],{'x':None,'y':None,'__proto__':None})
        h=self.edit(h,[{'op':'record-remove','record':'__proto__'},{'op':'field-remove','field':'__proto__'}])
        self.assertEqual(self.doc(h),original);self.assertEqual(self.w.db.serialize(),sql)

    def test_atomic_batch_failure_and_corrected_retry_retain_original_data_and_sequence(self):
        h,_=self.begin();before=copy.deepcopy(self.doc(h));sql=self.w.db.serialize()
        for invalid in [{'op':'cell','record':'row0','field':'x','text':'[broken'},
            {'op':'cell','record':'unknown','field':'x','text':'[1]'},
            {'op':'field','field':'x','key':'id','text':'"renamed"'},
            {'op':'metadata','key':'usage','text':'{"a":1,"\\u0061":2}'},
            {'op':'field-add','id':'x'}]:
            with self.assertRaisesRegex(GraphError,'Dataset edit #1'):
                self.edit(h,[{'op':'name','text':'"private change"'},invalid])
            self.assertEqual(self.doc(h),before);self.assertEqual(self.s.describe(h['session'],self.owner),h)
        newer=self.edit(h,[{'op':'cell','record':'row0','field':'x','text':'[999]'}])
        self.assertEqual(newer['sequence'],1);self.assertEqual(self.doc(newer)['records'][0]['values']['x'],[999])
        self.assertEqual(self.doc(newer)['records'][7],before['records'][7]);self.assertEqual(self.w.db.serialize(),sql)

    def test_dtype_transitions_metadata_remove_and_strict_numerics(self):
        h,_=self.begin();h=self.edit(h,[{'op':'field','field':'x','key':'dtype','text':'"category"'}])
        self.assertEqual(self.doc(h)['fields'][0]['vocabulary'],{'version':1,'values':[]})
        h=self.edit(h,[{'op':'field','field':'x','key':'vocabulary','text':'["low","high"]'},
            {'op':'field','field':'x','key':'unit','text':'"m"'},
            {'op':'metadata','key':'usage','text':''}])
        self.assertNotIn('usage',self.doc(h));self.assertEqual(self.doc(h)['fields'][0]['unit'],'m')
        h=self.edit(h,[{'op':'field','field':'x','key':'dtype','text':'"float32"'},
            {'op':'field','field':'x','key':'unit','text':''}])
        self.assertNotIn('vocabulary',self.doc(h)['fields'][0]);self.assertNotIn('unit',self.doc(h)['fields'][0])
        for text in ['NaN','1e9999','[Infinity]']:
            with self.assertRaises(GraphError):self.edit(h,[{'op':'cell','record':'row0','field':'x','text':text}])

    def test_owner_sequence_origin_pending_buffer_and_ram_guards(self):
        h,_=self.begin();sql=self.w.db.serialize()
        with self.assertRaises(GraphError):self.s.dataset_page(h['session'],object(),0)
        with self.assertRaises(GraphError):self.s.dataset_page(h['session'],self.owner,True)
        self.s.available_fn=lambda:0
        with self.assertRaises(GraphError):self.edit(h,[{'op':'field-add','id':'blocked'}])
        self.page(h);self.s.available_fn=lambda:2**31
        path=self.page(h)['rows'][0]['cells'][0]['path'];b=self.s.start_value(self.w,h['session'],self.owner,0,path)
        with self.assertRaisesRegex(GraphError,'staged value'):self.edit(h,[{'op':'record-add','id':'blocked'}])
        self.s.discard_value(h['session'],self.owner,0,b['value'],0)
        other=Workspace();self.addCleanup(other.db.close)
        with self.assertRaisesRegex(GraphError,'origin changed'):
            self.s.dataset_edit(other,h['session'],self.owner,0,[{'op':'record-add','id':'blocked'}])
        self.assertEqual(self.w.db.serialize(),sql)

    def test_limits_unknown_operations_and_empty_end_pages_are_explicit(self):
        h,_=self.begin()
        for cursor in [{'field_limit':7},{'row_limit':21},{'field_offset':True},{'row_offset':-1}]:
            with self.assertRaises(GraphError):self.page(h,**cursor)
        page=self.page(h,field_offset=10,row_offset=10)
        self.assertEqual(page['fields'],[]);self.assertEqual(page['rows'],[])
        self.assertIsNone(page['next_row_offset']);self.assertIsNone(page['next_field_offset'])
        for operations in [[],[{'op':'unsupported'}],[{'op':'name','text':'"x"','extra':True}],
            [{'op':'cell','record':'row0','field':'x','text':'0'*2001}]]:
            with self.assertRaises(GraphError):self.edit(h,operations)

    def test_stable_value_target_is_resolved_after_root_or_column_ordinals_change(self):
        h,_=self.begin();target={'kind':'cell','record':'row7','field':'y'}
        old=self.s.dataset_value(h['session'],self.owner,0,target)
        h=self.edit(h,[{'op':'metadata','key':'provenance','text':''},{'op':'field-remove','field':'x'}])
        actual=self.s.dataset_value(h['session'],self.owner,h['sequence'],target)
        self.assertEqual(actual['target'],target);self.assertEqual(actual['text'],'[15.0]')
        self.assertNotEqual(old['path'],actual['path']);self.assertEqual(_at(self.doc(h),actual['path']),[15.])
        with self.assertRaises(GraphError):self.s.dataset_value(h['session'],self.owner,h['sequence'],{'kind':'metadata','key':'provenance'})

    def test_full_signed_int64_boundaries_are_valid_but_float_bool_and_overflow_are_not(self):
        from app.datasets import validate_dataset
        doc=measurements();doc['fields'][0]['dtype']='int64'
        for i,row in enumerate(doc['records']):row['values']['x']=[-(2**63) if i%2 else 2**63-1]
        validate_dataset(doc);h,ref=self.begin(doc)
        self.assertEqual(self.page(h)['rows'][0]['cells'][0]['text'],'[9223372036854775807]')
        h=self.edit(h,[{'op':'cell','record':'row0','field':'x','text':'[-9223372036854775808]'}])
        saved=self.s.commit(self.w,h['session'],self.owner,h['sequence'])
        self.assertEqual(self.w.get_spec('dataset',saved['id'])['records'][0]['values']['x'],[-2**63])
        self.assertEqual(self.w.get_spec('dataset',ref['id'],ref['revision'])['records'][0]['values']['x'],[2**63-1])
        for invalid in [2**63,-(2**63)-1,float(2**53),True]:
            broken=copy.deepcopy(doc);broken['records'][0]['values']['x']=[invalid]
            with self.assertRaises(GraphError):validate_dataset(broken)


if __name__=='__main__':unittest.main()
