"""Exact bounded GUI/CLI projections, large scalars, SQL/RNG/no-worker reads."""
import copy
import io
import json
import random
import subprocess
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from app.core import GraphError, Workspace
from app.experiment_objects import execute_candidate, object_reference, preview_experiment
from app.science_browser import MAX_BYTES, encoded, index_page, inspect_page, summary
from examples.experiment_workspace import populate
import vpw_cli


class ScienceBrowserTests(unittest.TestCase):
    def setUp(self):
        self.w = Workspace(); self.w.set_constrained_mode(False); self.e = populate(self.w)
        self.addCleanup(self.w.db.close)
        self.ref = object_reference('experiment', self.e)

    def dataset(self): return self.w.get_spec('dataset', 'affine')

    def path(self, doc, *keys):
        path = []
        for key in keys:
            if isinstance(doc, dict):
                path.append(list(doc).index(key)); doc = doc[key]
            else: path.append(key); doc = doc[key]
        return path

    def assert_bound(self, value): self.assertLessEqual(len(encoded(value)), MAX_BYTES)

    def test_metadata_pages_do_not_read_documents_and_match_unicode_casefold_order(self):
        d = self.dataset(); d.pop('revision')
        for i, name in enumerate(('zulu', 'Alpha', 'äther', 'Straße', 'STRASSE', 'Oceanic')):
            d.update(id='name-' + str(i), name=name); self.w.put_spec('dataset', d)
        before = self.w.db.serialize(); traces = []; self.w.db.set_trace_callback(traces.append)
        with patch.object(self.w, 'get_spec', side_effect=AssertionError('metadata cannot fetch documents')):
            first = index_page(self.w, 'dataset', limit=3)
            second = index_page(self.w, 'dataset', offset=first['next_offset'], limit=3)
            full = index_page(self.w, 'dataset')
            matched = index_page(self.w, 'dataset', query='strasse')
        self.w.db.set_trace_callback(None)
        self.assertEqual(first['rows'] + second['rows'], full['rows'][:6])
        self.assertEqual(full['rows'], sorted(full['rows'], key=lambda r:(r['name'].casefold(), r['id'], r['revision'])))
        self.assertEqual({r['name'] for r in matched['rows']}, {'Straße', 'STRASSE'})
        self.assertFalse(any('document' in sql.lower() for sql in traces))
        self.assertEqual(self.w.db.serialize(), before); self.assert_bound(full)

    def test_selected_historical_identity_is_pinned_outside_query_or_page(self):
        old = self.dataset(); d = copy.deepcopy(old); d.pop('revision'); d['name'] = 'Changed head'
        new = self.w.put_spec('dataset', d); ref = object_reference('dataset', old)
        value = index_page(self.w, 'dataset', query='Changed', preferred=ref, limit=1)
        self.assertEqual(value['pinned']['revision'], ref['revision']); self.assertFalse(value['pinned']['selected'])
        self.assertEqual(value['rows'][0]['revision'], new['revision'])
        heads = index_page(self.w, 'dataset', heads_only=True)
        self.assertNotIn(ref['revision'], [r['revision'] for r in heads['rows']])
        missing = dict(ref, revision='0'*64)
        with self.assertRaisesRegex(GraphError, 'Pinned'): index_page(self.w, 'dataset', preferred=missing)

    def test_missing_pin_is_explicit_only_when_requested_and_never_replaced(self):
        missing={'kind':'dataset','id':'missing','revision':'0'*64};before=self.w.db.serialize()
        with patch.object(self.w,'get_spec',side_effect=AssertionError('metadata cannot read documents')):
            value=index_page(self.w,'dataset',preferred=missing,allow_missing_pin=True,limit=1)
        self.assertEqual(value['missing_pin'],missing);self.assertIsNone(value['pinned']);self.assertEqual(value['total'],2)
        self.assertEqual(value['rows'][0]['id'],'affine');self.assert_bound(value)
        value['missing_pin']['id']='changed';self.assertEqual(missing['id'],'missing')
        with self.assertRaises(GraphError):index_page(self.w,'dataset',preferred=missing)
        with self.assertRaises(GraphError):index_page(self.w,'dataset',allow_missing_pin=1)
        self.assertEqual(self.w.db.serialize(),before)

    def test_cli_exact_pin_and_explicit_missing_pin_flags_preserve_disk_bytes(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'lab.vtw';self.w.save(path);before=path.read_bytes();d=self.dataset()
            for sid,rev,allow,code in [('affine',d['revision'],False,0),('missing','0'*64,True,0),('missing','0'*64,False,1)]:
                stream=io.StringIO();args=[str(path),'spec-page','dataset','--query','not found','--pin',sid,rev]
                if allow:args.append('--allow-missing-pin')
                with redirect_stdout(stream):actual=vpw_cli.main(args)
                self.assertEqual(actual,code);result=json.loads(stream.getvalue())
                if code==0:
                    self.assertEqual(result['value']['rows'],[])
                    self.assertEqual((result['value']['missing_pin'] if allow else result['value']['pinned'])['revision'],rev)
                self.assertEqual(path.read_bytes(),before)

    def test_metadata_byte_limit_reduces_rows_without_truncating_any_identity(self):
        d = self.dataset(); d.pop('revision')
        for i in range(25):
            d.update(id='🌊'*196+str(i), name='🌊'*196+str(i)); self.w.put_spec('dataset', d)
        offset = 0; rows = []; counts = []
        while offset is not None:
            value = index_page(self.w, 'dataset', offset=offset)
            self.assert_bound(value); counts.append(len(value['rows'])); rows.extend(value['rows']); offset=value['next_offset']
        self.assertLess(counts[0], 25); self.assertEqual(len(rows), 27)
        self.assertEqual(len({(r['id'], r['revision']) for r in rows}), 27)
        self.assertTrue(all(len(r['id']) >= 197 for r in rows if r['id'].startswith('🌊')))

    def test_all_summaries_are_typed_not_partial_specs_and_hide_raw_records_or_weights(self):
        row = preview_experiment(self.e, self.w.get_spec)['candidates'][0]
        run = execute_candidate(self.w, self.ref, row['coordinates'], 'failed', 'Failed', python='/missing-science-browser-python')
        report = self.w.put_spec('report', {'version':1, 'id':'r', 'name':'Reading', 'question':'Compare saved failures',
            'mode':'result-report', 'runs':[run['reference']], 'experiment':None, 'sections':['failures'],
            'analysis':{'kind':'descriptive','inferential_unit':'run'}})
        before=self.w.db.serialize(); rng=random.getstate()
        with patch('app.transport.run_json_worker', side_effect=AssertionError('read cannot train')):
            for ref in [self.ref, run['reference'], report, *[p['dataset'] for p in self.e['data_tasks']],
                        *[p['task'] for p in self.e['data_tasks']], *self.e['models']]:
                value=summary(self.w,ref);self.assert_bound(value)
                self.assertEqual(value['scope'],'exact-spec-summary');self.assertEqual(value['reference'],ref)
                self.assertFalse(set(value)&{'records','fields','models','attempts','resolved','checkpoint','runs'})
                with self.assertRaises(GraphError):self.w.put_spec(ref['kind'],value)
        value=summary(self.w,run['reference']);self.assertEqual(value['attempt_count'],1)
        self.assertEqual(value['latest']['status'],'failed');self.assertEqual(value['selection']['coordinates'],row['coordinates'])
        self.assertEqual(self.w.db.serialize(),before);self.assertEqual(random.getstate(),rng)

    def test_long_question_summary_is_explicit_and_complete_text_is_chunk_recoverable(self):
        d=copy.deepcopy(self.e);d.pop('revision');d['question']='神経🌊e\u0301'*3000
        ref=self.w.put_spec('experiment',d);doc=self.w.get_spec('experiment',ref['id'],ref['revision'])
        info=summary(self.w,ref)['question'];self.assertEqual(info['total'],len(d['question']))
        self.assertEqual(info['next_offset'],1024)
        path=self.path(doc,'question');parts=[];offset=0
        while offset is not None:
            value=inspect_page(self.w,ref,path=path,offset=offset);self.assert_bound(value)
            parts.append(value['scalar']['text']);offset=value['next_offset']
        self.assertEqual(''.join(parts),d['question'])

    def test_container_pages_keep_record_and_tensor_order_owned_values_and_empty_tail(self):
        doc=self.dataset();ref=object_reference('dataset',doc);path=self.path(doc,'records')
        a=inspect_page(self.w,ref,path=path,limit=3);b=inspect_page(self.w,ref,path=path,offset=a['next_offset'],limit=3)
        self.assertEqual([r['ordinal'] for r in a['entries']+b['entries']],list(range(6)))
        leaf=self.path(doc,'records',3,'values','signal',0)
        scalar=inspect_page(self.w,ref,path=leaf);self.assertEqual(scalar['scalar']['text'],json.dumps(doc['records'][3]['values']['signal'][0]))
        self.assertEqual(scalar['representation'],'json-token');self.assertEqual(scalar['type'],'number')
        a['entries'][0]['preview']['size']=12345;self.assertNotEqual(inspect_page(self.w,ref,path=path)['entries'][0]['preview']['size'],12345)
        tail=inspect_page(self.w,ref,path=path,offset=1000);self.assertEqual(tail['entries'],[]);self.assertIsNone(tail['next_offset'])

    def test_huge_metadata_keys_and_scalar_values_are_explicit_inspectable_byte_bounded(self):
        doc=self.dataset();doc.pop('revision')
        key='🌊'*10000;doc['usage']={key+str(i):'🌊'*10000 for i in range(25)}
        ref=self.w.put_spec('dataset',doc);doc=self.w.get_spec('dataset',ref['id'],ref['revision']);path=self.path(doc,'usage')
        value=inspect_page(self.w,ref,path=path);self.assert_bound(value);self.assertLess(len(value['entries']),25)
        entry=value['entries'][0];self.assertEqual(entry['key']['total'],10001)
        self.assertEqual(entry['preview']['value']['total'],10000)
        member_path=path+[entry['ordinal']]
        parts=[];offset=0
        while offset is not None:
            page=inspect_page(self.w,ref,path=member_path,part='key',offset=offset);self.assert_bound(page)
            parts.append(page['scalar']['text']);offset=page['next_offset']
        self.assertEqual(''.join(parts),next(iter(doc['usage'])))
        page=inspect_page(self.w,ref,path=member_path);self.assertEqual(page['scalar']['total'],10000)

    def test_boolean_null_string_and_zero_never_alias_missingness(self):
        doc=self.dataset();doc.pop('revision');doc['usage']={'bool':False,'null':None,'string':'0','zero':0}
        ref=self.w.put_spec('dataset',doc);doc=self.w.get_spec('dataset',ref['id'],ref['revision'])
        for key,kind,text in [('bool','boolean','false'),('null','null','null'),('string','string','0'),('zero','number','0')]:
            value=inspect_page(self.w,ref,path=self.path(doc,'usage',key))
            self.assertEqual(value['type'],kind);self.assertEqual(value['scalar']['text'],text)

    def test_invalid_paths_page_limits_references_parts_search_and_bools_reject_readonly(self):
        before=self.w.db.serialize()
        for args in [{'offset':True},{'limit':True},{'limit':26},{'offset':-1},{'path':[True]},
                     {'path':[1000000]},{'path':'records'},{'path':[0]*1025},{'part':'guess'},{'part':'key'}]:
            with self.subTest(args=args),self.assertRaises(GraphError):inspect_page(self.w,self.ref,**args)
        for args in [{'query':'x'*201},{'query':None},{'heads_only':1},{'offset':1.5}]:
            with self.assertRaises(GraphError):index_page(self.w,'dataset',**args)
        for ref in [dict(self.ref,revision=None),dict(self.ref,revision='0'*64),dict(self.ref,kind='future')]:
            with self.assertRaises(GraphError):summary(self.w,ref)
        self.assertEqual(self.w.db.serialize(),before)

    def test_cli_equals_shared_projection_is_readonly_and_fresh_stdlib_host_loads_no_numeric_or_qt(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'lab.vtw';self.w.save(path);before=path.read_bytes()
            for args,expected in [(['spec-page','dataset','--limit','1'],index_page(self.w,'dataset',limit=1)),
                (['spec-summary','experiment',self.ref['id'],self.ref['revision']],summary(self.w,self.ref)),
                (['spec-inspect','experiment',self.ref['id'],self.ref['revision'],'--limit','2'],inspect_page(self.w,self.ref,limit=2))]:
                stream=io.StringIO()
                with redirect_stdout(stream):code=vpw_cli.main([str(path),*args])
                self.assertEqual(code,0);self.assertEqual(json.loads(stream.getvalue())['value'],expected)
            script='import sys,json;from app.core import Workspace;from app.science_browser import index_page,summary,inspect_page;w=Workspace.load(sys.argv[1]);index_page(w,"experiment");r=json.loads(sys.argv[2]);summary(w,r);inspect_page(w,r);print([x for x in ("torch","numpy","PyQt6","nbformat","nbclient") if x in sys.modules]);w.db.close()'
            child=subprocess.run(['/usr/bin/python3','-c',script,str(path),json.dumps(self.ref)],capture_output=True,text=True,timeout=15)
            self.assertEqual(child.returncode,0,child.stderr);self.assertEqual(child.stdout.strip(),'[]')
            self.assertEqual(path.read_bytes(),before)


if __name__=='__main__':unittest.main()
