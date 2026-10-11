"""Exact context browsing must not change binding order or transfer data."""
import copy
import io
import json
import random
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from app.core import Workspace,GraphError,default_model
from app import task_context,science_browser
from test_authoring import measurements
import vpw_cli


class TaskContextTests(unittest.TestCase):
    def setUp(self):
        self.w=Workspace();self.addCleanup(self.w.db.close)
        self.m=default_model('Exact task context',inputs=1,outputs=1);self.w.put(self.m);self.mref=self.w.freeze_model(self.m['id'])
        self.d=measurements();self.dref=self.w.put_spec('dataset',self.d)
    def page(self,collection='fields',**args):return task_context.page(self.w,self.mref,self.dref,collection,**args)
    def update(self):self.dref=self.w.put_spec('dataset',self.d)
    def test_pages_only_describe_fields_and_ports_and_leave_sql_rng_and_input_order_unchanged(self):
        self.d['fields'][0]['name']='Zulu';self.d['fields'][1]['name']='Alpha';self.update()
        before=self.w.db.serialize();rng=random.getstate()
        with patch('app.datasets._prepare_column',side_effect=AssertionError('no fitting')):
            fields=self.page();inputs=self.page('input_ports');outputs=self.page('output_ports')
        self.assertEqual([r['id'] for r in fields['rows']],['y','x']);self.assertEqual([r['ordinal'] for r in fields['rows']],[1,0])
        self.assertEqual(self.w.get_spec('dataset',self.dref['id'])['fields'],self.d['fields'])
        self.assertEqual(fields['record_count'],8);self.assertEqual(inputs['rows'][0]['id'],'input');self.assertEqual(outputs['rows'][0]['id'],'output')
        for value in (fields,inputs,outputs):
            self.assertEqual(value['scope'],'task-context-page');self.assertEqual(value['model'],self.mref);self.assertEqual(value['dataset'],self.dref)
            self.assertLessEqual(len(science_browser.encoded(value)),65536)
            for row in value['rows']:self.assertNotIn('vocabulary',row);self.assertNotIn('unit',row)
        self.assertEqual(self.w.db.serialize(),before);self.assertEqual(random.getstate(),rng)
    def test_full_metadata_and_large_vocab_are_not_encoded_for_projection(self):
        self.d['fields'].append({'id':'label','dtype':'category','shape':[],'vocabulary':{'version':1,'values':['label'+str(i) for i in range(20000)]},'unit':{'large':'λ'*20000}})
        for record in self.d['records']:record['values']['label']='label0'
        self.update();dumps=json.dumps
        def guard(value,*args,**kwargs):
            if isinstance(value,dict):self.assertNotIn('vocabulary',value,'No complete field encoding for suggestions')
            return dumps(value,*args,**kwargs)
        with patch('app.science_browser.json.dumps',side_effect=guard):value=self.page(query='label')
        self.assertEqual(value['rows'][0]['vocabulary_count'],20000);self.assertLess(len(science_browser.encoded(value)),2000)
    def test_shape_axes_preview_is_explicit_and_does_not_mutate_full_rank(self):
        field=self.d['fields'][0];field['shape']=[1]*8;field['axes']=['a'+str(i) for i in range(8)]
        for row in self.d['records']:
            v=1.
            for _ in range(8):v=[v]
            row['values']['x']=v
        self.update();value=self.page(query='x')['rows'][0]
        self.assertEqual(value['shape_rank'],8);self.assertEqual(len(value['shape']),4);self.assertFalse(value['shape_complete']);self.assertFalse(value['axes_complete'])
        self.assertEqual(self.w.get_spec('dataset',self.dref['id'])['fields'][0]['shape'],[1]*8)
    def test_historical_refs_casefold_search_pins_and_missing_ids_are_exact(self):
        old=self.dref;self.d['fields'][0]['name']='Straße';self.update();oldpage=task_context.page(self.w,self.mref,old,'fields')
        self.assertEqual(oldpage['rows'][0]['name'],'x');value=self.page(query='STRASSE',pin='y')
        self.assertEqual([r['id'] for r in value['rows']],['x']);self.assertEqual(value['pinned']['id'],'y');self.assertIsNone(value['missing_pin'])
        value=self.page(query='no match',pin='missing');self.assertEqual(value['rows'],[]);self.assertEqual(value['missing_pin'],'missing');self.assertIsNone(value['pinned'])
        with self.assertRaises(GraphError):task_context.page(self.w,self.mref,{**old,'revision':'0'*64},'fields')
    def test_unicode_byte_shortened_pages_cover_all_ids_with_actual_cursors(self):
        self.d['fields']=[{'id':'😀'*197+str(i),'name':'λ'*197+str(i),'dtype':'float32','shape':[]} for i in range(60)]
        self.d['records']=[{'id':str(i),'values':{f['id']:float(i) for f in self.d['fields']}} for i in range(2)];self.update()
        seen=[];offset=0;calls=0
        while True:
            value=self.page(offset=offset,pin=self.d['fields'][-1]['id']);calls+=1
            self.assertLessEqual(len(science_browser.encoded(value)),65536);seen.extend(r['id'] for r in value['rows'])
            if value['next_offset'] is None:break
            self.assertGreater(value['next_offset'],offset);offset=value['next_offset']
        self.assertGreater(calls,3);self.assertEqual(seen,sorted([f['id'] for f in self.d['fields']],key=lambda sid:(next(f['name'] for f in self.d['fields'] if f['id']==sid).casefold(),sid)))
    def test_closed_limits_and_unknown_collections_reject_without_writes(self):
        before=self.w.db.serialize()
        for args in ({'limit':26},{'offset':True},{'offset':-1},{'query':'x'*201},{'query':'\0'},{'pin':''},{'limit':False}):
            with self.assertRaises(GraphError):self.page(**args)
        with self.assertRaises(GraphError):self.page('weights')
        self.assertEqual(self.page(offset=100)['rows'],[]);self.assertEqual(self.w.db.serialize(),before)
    def test_existing_arbitrary_json_and_long_field_names_are_safe_explicit_previews(self):
        self.d['fields'][0]['name']={'arbitrary':'λ'*20000};self.d['fields'][1]['name']='λ'*20000;self.update()
        value=self.page();self.assertEqual(value['rows'][0]['id'],'x');self.assertEqual(value['rows'][0]['name'],'x');self.assertEqual(value['rows'][0]['name_source'],'id')
        self.assertEqual(value['rows'][1]['name'],'λ'*128);self.assertFalse(value['rows'][1]['name_complete']);self.assertEqual(value['name_limit'],128)
        self.assertEqual(self.w.get_spec('dataset',self.dref['id'])['fields'],self.d['fields'])
    def test_actual_cli_page_is_same_backend_and_preserves_file_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'context.vtw';self.w.save(path);before=path.read_bytes();out=io.StringIO()
            with redirect_stdout(out):status=vpw_cli.main([str(path),'task-context-page','fields','--model',self.mref['id'],self.mref['revision'],
                '--dataset',self.dref['id'],self.dref['revision'],'--limit','1','--query','x','--pin','y'])
            self.assertEqual(status,0);payload=json.loads(out.getvalue());self.assertTrue(payload['ok']);self.assertEqual(payload['value'],self.page(limit=1,query='x',pin='y'))
            self.assertEqual(path.read_bytes(),before)

if __name__=='__main__':unittest.main()
