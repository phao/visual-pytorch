"""Owned literal task pages: order, precision, staging and admission."""
import copy
import json
import random
import unittest
from unittest.mock import patch

from app import science_browser, task_drafts
from app.authoring_sessions import Sessions, _at
from app.core import GraphError, Workspace
from test_task_authoring import multi_fixture


class TaskDraftTests(unittest.TestCase):
    def setUp(self):
        self.w = Workspace(); self.addCleanup(self.w.db.close)
        self.addCleanup(lambda: self.w.history.directory.cleanup() if self.w.history.directory else None)
        self.s = Sessions(available_fn=lambda: 2**31); self.owner = object()
        self.addCleanup(lambda: self.s.close_owner(self.owner))
        self.model, self.dataset, self.task = multi_fixture()
        self.w.put(self.model); self.mref = self.w.freeze_model(self.model['id'])
        self.dref = self.w.put_spec('dataset', self.dataset)

    def begin(self, document=None):
        ref = self.w.put_spec('task', document or self.task)
        return self.s.open(self.w, self.owner, 'task', base=ref), ref

    def page(self, h, collection='inputs', **cursor):
        return self.s.task_page(h['session'], self.owner, h['sequence'], collection, **cursor)

    def transforms(self, h, collection='inputs', binding=0, **cursor):
        return self.s.task_transforms(h['session'], self.owner, h['sequence'], collection, binding, **cursor)

    def document(self, h):
        return self.s._sessions[h['session']]['draft']['document']

    def test_closed_literal_projections_are_readonly_not_persistable_and_preserve_order(self):
        h, _ = self.begin(); sql = self.w.db.serialize(); rng = random.getstate()
        for collection, keys in task_drafts.PROPERTIES.items():
            page = self.page(h, collection)
            self.assertEqual(page['scope'], 'owned-task-page'); self.assertEqual(page['session'], h['session'])
            self.assertEqual([row['ordinal'] for row in page['rows']], [0, 1])
            self.assertEqual(page['workspace_id'], h['workspace_id']); self.assertEqual(page['id'], h['id'])
            self.assertEqual(page['rows'][0]['transforms']['total'], len(self.task[collection][0]['transforms']))
            for row in page['rows']:
                self.assertEqual([p['key'] for p in row['properties']], list(keys))
                for value in row['properties']:
                    if value['present']:
                        self.assertEqual(json.loads(value['text']), _at(self.document(h), value['path']))
            self.assertLessEqual(len(science_browser.encoded(page)), 65536)
            with self.assertRaises(GraphError): self.w.put_spec('task', page)
        steps = self.transforms(h)
        self.assertEqual([json.loads(r['properties'][0]['text']) for r in steps['rows']], ['standardize', 'affine'])
        self.assertEqual(self.w.db.serialize(), sql); self.assertEqual(random.getstate(), rng)

    def test_large_binding_and_transform_collections_are_independently_paged_not_encoded_whole(self):
        doc = copy.deepcopy(self.task)
        doc['description'] = {'long': 'λ'*20000}
        doc['inputs'] = [{'port': 'p'+str(i), 'field': 'a', 'transforms': []} for i in range(80)]
        doc['inputs'][7]['transforms'] = [{'kind': 'affine', 'scale': i+1, 'offset': i} for i in range(1000)]
        h, _ = self.begin(doc); hidden = self.document(h)['inputs'][7]['transforms']; dumps = json.dumps
        def guarded(value, *args, **kw):
            self.assertIsNot(value, hidden, 'Do not encode a complete hidden pipeline for card previews')
            self.assertIsNot(value, self.document(h)['description'], 'Do not encode hidden description metadata')
            return dumps(value, *args, **kw)
        with patch('app.dataset_drafts.json.dumps', side_effect=guarded):
            cards = self.page(h, offset=6); steps = self.transforms(h, binding=7, offset=700, limit=3)
        self.assertEqual([row['ordinal'] for row in cards['rows']], list(range(6, 12)))
        self.assertEqual(cards['next_offset'], 12); self.assertEqual(cards['total'], 80)
        self.assertIsNone(cards['description']['text'])
        self.assertEqual(cards['rows'][1]['transforms']['total'], 1000)
        self.assertEqual([row['ordinal'] for row in steps['rows']], [700, 701, 702])
        self.assertEqual(steps['next_offset'], 703); self.assertEqual(steps['total'], 1000)
        self.assertEqual(self.document(h), doc)

    def test_exact_literal_integer_tokens_and_staged_values_share_parent_without_retaining_it(self):
        h, _ = self.begin()
        path = self.transforms(h)['rows'][1]['path']
        h = self.s.patch(self.w, h['session'], self.owner, 0,
                        [{'op': 'put', 'path': path, 'key': 'scale', 'value': 9007199254740993}])
        value = next(v for v in self.transforms(h)['rows'][1]['properties'] if v['key'] == 'scale')
        self.assertEqual(value['text'], '9007199254740993')
        sql = self.w.db.serialize()
        buffer = self.s.start_value(self.w, h['session'], self.owner, 1, value['path'])
        buffer = self.s.splice_value(self.w, h['session'], self.owner, 1, buffer['value'], 0,
                                    offset=0, delete_bytes=16, text='9007199254740995')
        h = self.s.finish_value(self.w, h['session'], self.owner, 1, buffer['value'], 1)
        self.assertEqual(h['sequence'], 2); self.assertEqual(_at(self.document(h), value['path']), 9007199254740995)
        self.assertEqual(self.w.db.serialize(), sql)
        self.assertEqual(self.s.describe(h['session'], self.owner), h)

    def test_byte_shortening_includes_actual_cursor_and_covers_all_card_ordinals(self):
        doc = copy.deepcopy(self.task)
        doc['inputs'] = [{'port': '😀'*180+str(i), 'field': 'a', 'transforms': []} for i in range(40)]
        h, _ = self.begin(doc)
        budget = len(science_browser.encoded(self.page(h, limit=1))) + 100
        seen = []; offset = 0; calls = 0
        with patch.object(science_browser, 'MAX_BYTES', budget):
            while True:
                page = self.page(h, offset=offset); calls += 1
                self.assertLessEqual(len(science_browser.encoded(page)), budget)
                seen.extend(row['ordinal'] for row in page['rows'])
                if page['next_offset'] is None:
                    break
                self.assertEqual(page['next_offset'], offset+len(page['rows']))
                offset = page['next_offset']
        self.assertGreater(calls, 7); self.assertEqual(seen, list(range(40)))
        self.assertEqual(self.document(h), doc)

    def test_literal_edit_batches_preserve_large_integer_tokens_and_are_atomic_and_closed(self):
        h,_=self.begin();path=self.transforms(h)['rows'][1]['path'];before=copy.deepcopy(self.document(h))
        def edit(ops):return self.s.task_edit(self.w,h['session'],self.owner,h['sequence'],ops)
        for bad in [
            {'op':'put-json','path':path,'key':'scale','text':'NaN'},
            {'op':'put-json','path':path,'key':'scale','text':'1e9999'},
            {'op':'put-json','path':path,'key':'scale','text':'{"a":1,"a":2}'},
            {'op':'put-json','path':path,'key':'scale','text':'3','value':99},
            {'op':'put','path':path,'key':'scale','value':3},
            {'op':'replace-json','path':path,'text':'0'*2001},
        ]:
            with self.assertRaises(GraphError):
                edit([{'op':'put-json','path':[],'key':'name','text':'"private"'},bad])
            self.assertEqual(self.document(h),before);self.assertEqual(self.s.describe(h['session'],self.owner),h)
        h=edit([{'op':'put-json','path':path,'key':'scale','text':'9007199254740993'},
                {'op':'insert-json','path':path[:-1],'index':2,'text':'{"kind":"standardize"}'},
                {'op':'move','path':path[:-1],'from':2,'to':0}])
        steps=self.document(h)['inputs'][0]['transforms']
        self.assertEqual([step['kind'] for step in steps],['standardize','standardize','affine'])
        self.assertEqual(steps[2]['scale'],9007199254740993)
        self.assertEqual(h['sequence'],1)

    def test_ordered_atomic_patches_and_old_page_sequences_cannot_target_moved_bindings(self):
        h, _ = self.begin(); page = self.page(h); original = copy.deepcopy(self.document(h))
        path = page['rows'][0]['path'][:-1]
        with self.assertRaises(GraphError):
            self.s.patch(self.w, h['session'], self.owner, 0,
                         [{'op':'move','path':path,'from':0,'to':1}, {'op':'remove','path':[999]}])
        self.assertEqual(self.document(h), original); self.assertEqual(self.s.describe(h['session'],self.owner),h)
        h = self.s.patch(self.w, h['session'], self.owner, 0, [{'op':'move','path':path,'from':0,'to':1}])
        with self.assertRaisesRegex(GraphError, 'sequence changed'):
            self.s.task_page(h['session'], self.owner, 0, 'inputs')
        with self.assertRaisesRegex(GraphError, 'sequence changed'):
            self.s.patch(self.w, h['session'], self.owner, 0,
                         [{'op':'put','path':page['rows'][0]['path'],'key':'field','value':'wrong'}])
        current = self.page(h)
        self.assertEqual(json.loads(current['rows'][0]['properties'][0]['text']), 'right')
        self.assertEqual(self.document(h)['heads'], original['heads'])

    def test_owned_compatibility_is_exact_rng_neutral_no_fit_and_matches_legacy_contract(self):
        h, ref = self.begin(); sql = self.w.db.serialize(); rng = random.getstate()
        with patch('app.datasets._prepare_column', side_effect=AssertionError('No fitting')):
            result = self.s.task_compatibility(self.w,h['session'],self.owner,0,self.mref,self.dref)
        self.assertEqual(result['task'],ref); self.assertEqual(result['status'],'compatible')
        self.assertEqual(result['model'],self.mref); self.assertEqual(result['dataset'],self.dref)
        h = self.s.patch(self.w,h['session'],self.owner,0,
                        [{'op':'put','path':self.page(h)['rows'][0]['path'],'key':'field','value':'unknown'}])
        result = self.s.task_compatibility(self.w,h['session'],self.owner,1,self.mref,self.dref)
        self.assertEqual(result['status'],'incompatible'); self.assertTrue(result['reason'])
        self.assertEqual(self.w.db.serialize(),sql); self.assertEqual(random.getstate(),rng)
        with self.assertRaises(GraphError):
            self.s.task_compatibility(self.w,h['session'],self.owner,1,{**self.mref,'revision':'0'*64},self.dref)

    def test_owner_origin_buffer_and_admission_fences_preserve_inspection(self):
        h, _ = self.begin()
        with self.assertRaises(GraphError): self.s.task_page(h['session'],object(),0,'inputs')
        other = Workspace(); self.addCleanup(other.db.close)
        other.db.execute("UPDATE metadata SET value=? WHERE key='workspace_id'",(h['workspace_id'],))
        with self.assertRaisesRegex(GraphError,'origin changed'):
            self.s.task_compatibility(other,h['session'],self.owner,0,self.mref,self.dref)
        self.s.available_fn=lambda:0
        self.page(h)
        with self.assertRaisesRegex(GraphError,'RAM'):
            self.s.task_compatibility(self.w,h['session'],self.owner,0,self.mref,self.dref)
        self.s.available_fn=lambda:2**31
        path=self.page(h)['description']['path']
        b=self.s.start_value(self.w,h['session'],self.owner,0,path)
        self.page(h)
        with self.assertRaisesRegex(GraphError,'staged value'):
            self.s.task_compatibility(self.w,h['session'],self.owner,0,self.mref,self.dref)
        self.s.discard_value(h['session'],self.owner,0,b['value'],0)

    def test_cursor_grammar_empty_pages_wrong_kind_and_malformed_drafts_are_explicit(self):
        h,_=self.begin()
        for cursor in [{'offset':True},{'offset':-1},{'limit':0},{'limit':7},{'limit':True}]:
            with self.assertRaises(GraphError):self.page(h,**cursor)
        for collection in ['fields',None,[],{}]:
            with self.assertRaises(GraphError):self.page(h,collection)
        for index in [True,-1,2,'0']:
            with self.assertRaises(GraphError):self.transforms(h,binding=index)
        for value in [self.page(h,offset=100),self.transforms(h,offset=100)]:
            self.assertEqual(value['rows'],[]);self.assertIsNone(value['next_offset'])
        h=self.s.patch(self.w,h['session'],self.owner,0,
                       [{'op':'put','path':self.page(h)['rows'][0]['path'],'key':'transforms','value':None}])
        with self.assertRaisesRegex(GraphError,'raw draft'):self.page(h)
        self.s.close(h['session'],self.owner)
        dataset=self.s.open(self.w,self.owner,'dataset')
        with self.assertRaisesRegex(GraphError,'task draft'):self.page(dataset)
        with self.assertRaisesRegex(GraphError,'task draft'):self.transforms(dataset)

    def test_historical_base_head_and_strict_validation_remain_after_paged_edits(self):
        old=self.w.put_spec('task',self.task);new=copy.deepcopy(self.task);new['name']='New head'
        head=self.w.put_spec('task',new)
        h=self.s.open(self.w,self.owner,'task',base=old)
        self.assertEqual(h['base'],old);self.assertEqual(h['expected_head'],head['revision'])
        h=self.s.patch(self.w,h['session'],self.owner,0,
                       [{'op':'put','path':self.page(h,'heads')['rows'][0]['path'],'key':'weight','value':-1}])
        with self.assertRaises(GraphError):self.s.commit(self.w,h['session'],self.owner,1)
        self.assertEqual(self.s.describe(h['session'],self.owner),h)
        h=self.s.patch(self.w,h['session'],self.owner,1,
                       [{'op':'put','path':self.page(h,'heads')['rows'][0]['path'],'key':'weight','value':0.5}])
        saved=self.s.commit(self.w,h['session'],self.owner,2)
        self.assertEqual(self.w.get_spec('task',old['id'],old['revision'])['name'],self.task['name'])
        self.assertEqual(self.w.get_spec('task',saved['id'])['heads'][0]['weight'],0.5)
        self.assertEqual(self.w.get_spec('task',saved['id'])['name'],self.task['name'])

    def test_named_targets_resolve_after_root_and_binding_key_ordinals_shift(self):
        h,_=self.begin();sql=self.w.db.serialize()
        h=self.s.task_edit(self.w,h['session'],self.owner,0,[
            {'op':'property','target':{'kind':'document'},'key':'description','text':None},
            {'op':'property','target':{'kind':'binding','collection':'heads','binding':0},'key':'mask','text':None},
            {'op':'property','target':{'kind':'binding','collection':'heads','binding':0},'key':'sample_weight','text':'"valid"'},
            {'op':'property','target':{'kind':'transform','collection':'inputs','binding':0,'transform':1},'key':'scale','text':'9007199254740993'},
            {'op':'property','target':{'kind':'document'},'key':'name','text':'"Shift-safe task"'}])
        doc=self.document(h);self.assertNotIn('description',doc);self.assertNotIn('mask',doc['heads'][0])
        self.assertEqual(doc['heads'][0]['sample_weight'],'valid')
        self.assertEqual(doc['inputs'][0]['transforms'][1]['scale'],9007199254740993)
        self.assertEqual(doc['inputs'][1],self.task['inputs'][1]);self.assertEqual(self.w.db.serialize(),sql)
        self.assertEqual(_at(doc,self.page(h)['path']),doc['inputs'])
        self.assertEqual(_at(doc,self.transforms(h)['path']),doc['inputs'][0]['transforms'])

    def test_named_target_grammar_wrong_kind_index_and_midbatch_failure_are_atomic(self):
        h,_=self.begin();before=copy.deepcopy(self.document(h))
        target={'kind':'binding','collection':'inputs','binding':0}
        for change in [
            {'target':{**target,'binding':True}}, {'target':{**target,'binding':10}},
            {'target':{**target,'extra':True}}, {'target':{**target,'collection':'records'}},
            {'target':{'kind':'document'},'key':'id'}, {'key':'transforms'},
            {'text':'NaN'}, {'text':'{"x":1,"x":2}'}, {'text':False}, {'extra':'no'},
            {'target':{'kind':'transform','collection':'inputs','binding':0,'transform':True}},
        ]:
            bad={'op':'property','target':target,'key':'field','text':'"x"',**change}
            with self.assertRaisesRegex(GraphError,'Task edit #1'):
                self.s.task_edit(self.w,h['session'],self.owner,0,[
                    {'op':'property','target':{'kind':'document'},'key':'description','text':None},bad])
            self.assertEqual(self.document(h),before);self.assertEqual(self.s.describe(h['session'],self.owner),h)

    def test_origin_proof_requires_actual_owner_workspace_connection_and_exact_sequence(self):
        h,_=self.begin();sql=self.w.db.serialize()
        self.assertEqual(self.s.origin(self.w,h['session'],self.owner,0),h)
        replacement=Workspace();self.addCleanup(replacement.db.close)
        replacement.db.execute("UPDATE metadata SET value=? WHERE key='workspace_id'",(h['workspace_id'],))
        for workspace,owner,sequence in [(replacement,self.owner,0),(self.w,object(),0),(self.w,self.owner,True),(self.w,self.owner,1)]:
            with self.assertRaises(GraphError):self.s.origin(workspace,h['session'],owner,sequence)
        self.assertEqual(self.s.origin(self.w,h['session'],self.owner,0),h)
        self.assertEqual(self.w.db.serialize(),sql)


if __name__ == '__main__': unittest.main()
