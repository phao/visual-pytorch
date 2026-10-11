"""Chunked raw JSON editing, memory preflight and exact scientific admission."""
import copy
import json
import unittest
from unittest.mock import patch

from app import authoring_memory, science_browser
from app.authoring_sessions import Sessions
from app.core import GraphError, Workspace
from test_authoring import measurements


class DraftValueTests(unittest.TestCase):
    def setUp(self):
        self.w=Workspace(); self.addCleanup(self.w.db.close)
        self.addCleanup(lambda:self.w.history.directory.cleanup() if self.w.history.directory else None)
        self.owner=object(); self.other=object(); self.available=2**30
        self.s=Sessions(available_fn=lambda:self.available)
        self.addCleanup(lambda:self.s.close_owner(self.owner))

    def begin(self, value):
        doc=measurements(); doc['usage']={'data':value}
        ref=self.w.put_spec('dataset',doc); h=self.s.open(self.w,self.owner,'dataset',base=ref)
        keys=list(self.s._sessions[h['session']]['draft']['document'])
        b=self.s.start_value(self.w,h['session'],self.owner,0,[keys.index('usage'),0])
        return h,b,ref

    def splice(self,h,b,**kw):
        return self.s.splice_value(self.w,h['session'],self.owner,h['sequence'],b['value'],b['value_revision'],**kw)

    def read(self,h,b,offset=0):
        return self.s.read_value(h['session'],self.owner,h['sequence'],b['value'],b['value_revision'],offset=offset)

    def finish(self,h,b):
        return self.s.finish_value(self.w,h['session'],self.owner,h['sequence'],b['value'],b['value_revision'])

    def test_large_value_pages_splice_finish_and_precision_without_whole_wire_value(self):
        value=[i if i!=7 else 2**53+1 for i in range(20000)]
        h,b,ref=self.begin(value); before=self.w.db.serialize(); parts=[]; offset=0
        while True:
            page=self.read(h,b,offset); self.assertLessEqual(len(science_browser.encoded(page)),65536)
            self.assertLessEqual(len(page['text']),2000); parts.append(page['text'])
            if page['next_offset'] is None: break
            offset=page['next_offset']
        self.assertEqual(json.loads(''.join(parts)),value)
        self.s.discard_value(h['session'],self.owner,0,b['value'],b['value_revision'])
        with self.assertRaisesRegex(GraphError,'byte'):
            self.s.patch(self.w,h['session'],self.owner,0,[{'op':'replace','path':b['path'],'value':value}])
        b=self.s.start_value(self.w,h['session'],self.owner,0,b['path'])
        # Full JSON exceeds a small patch; one small literal change never transfers it.
        b=self.splice(h,b,offset=1,delete_bytes=1,text='42')
        accepted=self.finish(h,b); self.assertEqual(accepted['sequence'],1)
        self.assertEqual(self.w.db.serialize(),before)
        saved=self.s.commit(self.w,h['session'],self.owner,1)
        retained=self.w.get_spec('dataset',saved['id'],saved['revision'])['usage']['data']
        self.assertEqual(retained[0],42); self.assertEqual(retained[7],2**53+1); self.assertEqual(retained[1:],value[1:])
        self.assertEqual(self.w.get_spec('dataset',ref['id'],ref['revision'])['usage']['data'],value)

    def test_utf8_byte_offsets_complete_characters_empty_last_page_and_boundaries(self):
        value='🔬λ'*5000; h,b,_=self.begin(value); text=''; offset=0
        while True:
            page=self.read(h,b,offset); text+=page['text']
            self.assertEqual(len(page['text'].encode('utf-8')), (page['next_offset'] or page['total_bytes'])-offset)
            if page['next_offset'] is None: break
            offset=page['next_offset']
        self.assertEqual(json.loads(text),value)
        self.assertEqual(self.read(h,b,b['total_bytes'])['text'],'')
        for offset in [2,True,-1,b['total_bytes']+1]:
            with self.subTest(offset=offset),self.assertRaises(GraphError): self.read(h,b,offset)
        with self.assertRaisesRegex(GraphError,'UTF-8'): self.splice(h,b,offset=1,delete_bytes=1,text='x')
        b=self.splice(h,b,offset=1,delete_bytes=4,text='é')
        self.finish(h,b)
        self.assertEqual(self.s._sessions[h['session']]['draft']['document']['usage']['data'],'é'+value[1:])

    def test_invalid_duplicate_overflow_and_incomplete_json_keep_original_and_raw_buffer(self):
        for text in ['[1,','{"a":1,"\\u0061":2}','1e9999','NaN']:
            h,b,ref=self.begin([1,2]); before=self.w.db.serialize()
            b=self.splice(h,b,offset=0,delete_bytes=b['total_bytes'],text=text)
            with self.subTest(text=text),self.assertRaises(GraphError): self.finish(h,b)
            self.assertEqual(self.read(h,b)['text'],text)
            self.assertEqual(self.s._sessions[h['session']]['draft']['document']['usage']['data'],[1,2])
            self.assertEqual(self.w.db.serialize(),before)
            if text=='[1,':
                b=self.splice(h,b,offset=b['total_bytes'],delete_bytes=0,text='2]')
                self.assertEqual(self.finish(h,b)['sequence'],1)
            self.s.close(h['session'],self.owner)

    def test_pending_buffer_blocks_other_patch_check_commit_and_discard_closes_file(self):
        h,b,_=self.begin([1,2]); file=self.s._sessions[h['session']]['buffer']['file']
        for fn in [lambda:self.s.patch(self.w,h['session'],self.owner,0,[]),
                   lambda:self.s.check(self.w,h['session'],self.owner,0),lambda:self.s.commit(self.w,h['session'],self.owner,0),
                   lambda:self.s.start_value(self.w,h['session'],self.owner,0,b['path'])]:
            with self.assertRaisesRegex(GraphError,'staged value'): fn()
        result=self.s.discard_value(h['session'],self.owner,0,b['value'],0)
        self.assertEqual(result,h); self.assertTrue(file.closed)
        self.s.check(self.w,h['session'],self.owner,0)

    def test_stale_buffer_owner_sequence_bad_splice_and_identity_preserve_everything(self):
        h,b,_=self.begin('original'); before=self.w.db.serialize()
        cases=[{'offset':0,'delete_bytes':0,'text':'λ'*2001}, {'offset':0,'delete_bytes':True,'text':''},
               {'offset':0,'delete_bytes':100,'text':''}, {'offset':0,'delete_bytes':0,'text':'\ud800'}]
        for index,kw in enumerate(cases):
            with self.subTest(case=index),self.assertRaises(GraphError): self.splice(h,b,**kw)
            self.assertEqual(self.read(h,b)['text'],'"original"')
        with self.assertRaises(GraphError):self.s.read_value(h['session'],self.other,0,b['value'],0)
        with self.assertRaises(GraphError):self.s.read_value(h['session'],self.owner,1,b['value'],0)
        newer=self.splice(h,b,offset=1,delete_bytes=1,text='O')
        with self.assertRaises(GraphError):self.read(h,b)
        self.assertEqual(self.read(h,newer)['text'],'"Original"'); self.assertEqual(self.w.db.serialize(),before)

    def test_owner_close_disposes_pending_file_and_failed_finish_keeps_file_open(self):
        h,b,_=self.begin([1]); file=self.s._sessions[h['session']]['buffer']['file']
        self.available=0
        with self.assertRaisesRegex(GraphError,'RAM'):self.finish(h,b)
        self.assertFalse(file.closed); self.assertEqual(self.read(h,b)['text'],'[1]')
        self.s.close_owner(self.owner); self.assertTrue(file.closed)

    def test_base_path_translation_before_after_revision_and_no_identity_edit(self):
        doc=measurements();doc['usage']={'source':[1,2]}; ref=self.w.put_spec('dataset',doc)
        retained=self.w.get_spec('dataset',ref['id'],ref['revision']); keys=list(retained)
        for key in ['records','usage']:
            h=self.s.open(self.w,self.owner,'dataset',base=ref)
            path=[keys.index(key)]; b=self.s.start_value(self.w,h['session'],self.owner,0,path,from_base=True)
            self.assertEqual(json.loads(self.read(h,b)['text']),retained[key])
            self.assertEqual(b['path'][0],path[0]-(1 if path[0]>keys.index('revision') else 0))
            self.s.close(h['session'],self.owner)
        h=self.s.open(self.w,self.owner,'dataset',base=ref)
        with self.assertRaisesRegex(GraphError,'revision identity'):
            self.s.start_value(self.w,h['session'],self.owner,0,[keys.index('revision')],from_base=True)

    def test_open_size_preflight_happens_before_base_parse_no_source_repair(self):
        ref=self.w.put_spec('dataset',measurements()); before=self.w.db.serialize(); self.available=0
        with patch.object(self.w,'get_spec',side_effect=AssertionError('must not parse')):
            with self.assertRaisesRegex(GraphError,'RAM'):self.s.open(self.w,self.owner,'dataset',base=ref)
        self.assertEqual(self.w.db.serialize(),before);self.assertEqual(self.s._sessions,{})

    def test_percentage_unknown_ram_unconstrained_and_later_read_inspection(self):
        h,b,_=self.begin([1]); self.available=None
        with self.assertRaisesRegex(GraphError,'observe'):self.finish(h,b)
        self.assertEqual(self.read(h,b)['text'],'[1]')
        self.w.set_constrained_mode(False)
        with patch('app.memory.available_memory',side_effect=AssertionError('not constrained')):
            self.finish(h,b)
        self.w.set_constrained_mode(True); self.s.memory_limit_percent=1.; self.available=100*1024**2
        with self.assertRaisesRegex(GraphError,'RAM'):self.s.check(self.w,h['session'],self.owner,1)
        self.s.memory_limit_percent=20.;self.s.check(self.w,h['session'],self.owner,1)

    def test_finish_budget_precedes_raw_parse_and_aggregate_session_estimate(self):
        h,b,_=self.begin([1]); other=self.s.open(self.w,self.other,'task'); self.addCleanup(lambda:self.s.close_owner(self.other))
        self.available=0
        with patch('app.authoring_sessions._parse',side_effect=AssertionError('must not parse raw')):
            with self.assertRaisesRegex(GraphError,'RAM'):self.finish(h,b)
        self.assertEqual(self.read(h,b)['text'],'[1]')
        self.available=150*1024**2 # 10% budget admits one 8-MiB template, not two.
        with self.assertRaisesRegex(GraphError,'RAM'):self.finish(h,b)
        self.s.close(other['session'],self.other);self.finish(h,b)

    def test_patch_allocation_estimate_finite_json_and_new_root_are_transactional(self):
        self.assertGreater(authoring_memory.footprint([1]*10000),authoring_memory.footprint([1]))
        with self.assertRaises(GraphError):authoring_memory.footprint({'bad':float('inf')})
        h,b,_=self.begin([1]); self.s.discard_value(h['session'],self.owner,0,b['value'],0)
        self.available=0
        with self.assertRaisesRegex(GraphError,'RAM'):
            self.s.patch(self.w,h['session'],self.owner,0,[{'op':'put','path':[],'key':'name','value':'unsaved'}])
        self.assertEqual(self.s.describe(h['session'],self.owner),h)
