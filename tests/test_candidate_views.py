"""Stored candidate wire bounds preserve exact scientific selection semantics."""
import copy,io,json,random,subprocess,sys,tempfile,unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch
from app import candidate_views,science_browser
from app.core import Workspace,GraphError,default_model
from app.experiment_objects import preview_experiment
from test_experiment_authoring import fixture
import vpw_cli

class CandidateViewsTests(unittest.TestCase):
    def setUp(self):
        self.w=Workspace();self.addCleanup(self.w.db.close)
        self.addCleanup(lambda:self.w.history.directory.cleanup() if self.w.history.directory else None)
        self.e=fixture(self.w);self.ref=self.w.put_spec('experiment',self.e)
    def page(self,**kw):return candidate_views.stored(self.w,self.ref,**kw)
    def full(self,**kw):return preview_experiment(self.w.get_spec('experiment',self.ref['id'],self.ref['revision']),self.w.get_spec,**kw)
    def test_exact_candidates_references_counts_and_order_equal_complete_preview(self):
        full=self.full();page=self.page()
        self.assertEqual(page['scope'],'stored-experiment-preview');self.assertEqual(page['experiment'],self.ref)
        self.assertEqual(page['total_combinations_exact'],'8');self.assertIsNone(page['next_offset'])
        for row,complete in zip(page['candidates'],full['candidates']):
            for key in ('id','coordinates','status','references'):self.assertEqual(row[key],complete[key])
            self.assertEqual(row['counts'],{**{r:len(v) for r,v in complete['manifest']['roles'].items()},'excluded':len(complete['manifest']['excluded'])})
            self.assertIsNone(row['reason']);self.assertNotIn('manifest',row);self.assertNotIn('protocol',row)
        self.assertLessEqual(len(science_browser.encoded(page)),65536)
    def test_readonly_sql_history_dirty_rng_no_fitting_or_execution_admission(self):
        sql=self.w.db.serialize();history=copy.deepcopy(self.w.history.summary());dirty=self.w.dirty;rng=random.getstate()
        with patch('app.datasets._prepare_column',side_effect=AssertionError('No fit')),patch('app.memory.available_memory',side_effect=AssertionError('No numerical admission')):
            self.page();self.page(offset=7,limit=1)
        self.assertEqual(self.w.db.serialize(),sql);self.assertEqual(self.w.history.summary(),history)
        self.assertEqual(self.w.dirty,dirty);self.assertEqual(random.getstate(),rng);self.assertEqual(self.w.list_specs('run',True),[])
    def test_historical_experiment_and_dependency_heads_never_replace_exact_pins(self):
        before=self.page();e=copy.deepcopy(self.e);e['name']='Newer head';self.w.put_spec('experiment',e)
        ref=self.e['data_tasks'][0]['dataset'];d=self.w.get_spec('dataset',ref['id']);d.pop('revision');d['records'][0]['values']['x'][0]=999
        self.w.put_spec('dataset',d);self.w.archive_spec('dataset',ref['id']);self.w.archive_spec('experiment',self.ref['id'])
        self.assertEqual(self.page(),before)
    def test_exclusions_and_incompatibilities_are_retained_with_partial_reason_markers(self):
        e=copy.deepcopy(self.e);m=default_model('Wrong width',inputs=2);self.w.put(m);e['models'].append(self.w.freeze_model(m['id']))
        e['selection']['exclusions']=[{'model':e['models'][0],'data_task':'pair-0','reason':'λ😀'*300}]
        self.ref=self.w.put_spec('experiment',e);full=self.full();page=self.page()
        self.assertEqual({r['status'] for r in page['candidates']},{'compatible','incompatible','excluded'})
        for row,complete in zip(page['candidates'],full['candidates']):
            self.assertEqual(row['status'],complete['status'])
            if complete['reason']:
                self.assertEqual(row['reason'],science_browser._text(complete['reason'],limit=128))
            if complete['manifest'] is None:self.assertIsNone(row['counts'])
        excluded=next(r for r in page['candidates'] if r['status']=='excluded')
        self.assertIsNotNone(excluded['reason']['next_offset'])
    def test_actual_byte_shortened_cursor_covers_every_candidate_once(self):
        e=copy.deepcopy(self.e);e['initialization_seeds']=list(range(12));self.ref=self.w.put_spec('experiment',e)
        expected=self.full(limit=100)['candidates'];one=self.page(limit=1);budget=len(science_browser.encoded(one))+40
        offset=0;seen=[];calls=0
        with patch.object(science_browser,'MAX_BYTES',budget):
            while True:
                p=self.page(offset=offset);calls+=1;seen.extend(p['candidates']);self.assertLessEqual(len(science_browser.encoded(p)),budget)
                if p['next_offset'] is None:break
                self.assertEqual(p['next_offset'],offset+len(p['candidates']));offset=p['next_offset']
        self.assertEqual([r['id'] for r in seen],[r['id'] for r in expected]);self.assertGreater(calls,5)
        self.assertEqual(self.page(offset=100)['candidates'],[])
    def test_large_hidden_manifest_is_never_encoded_as_a_wire_document(self):
        source=self.full();hidden=source['candidates'][0]['manifest'];hidden['roles']['train']=['hidden'+str(i) for i in range(20000)]
        source['candidates'][0]['protocol']['hidden']='λ😀'*10000;dumps=json.dumps
        def guarded(value,*a,**kw):
            if isinstance(value,dict):
                for row in value.get('candidates',[]):self.assertNotIn('manifest',row);self.assertNotIn('protocol',row)
            return dumps(value,*a,**kw)
        with patch('app.candidate_views.preview_experiment',return_value=source),patch('app.science_browser.json.dumps',side_effect=guarded):p=self.page()
        self.assertEqual(p['candidates'][0]['counts']['train'],20000);self.assertLessEqual(len(science_browser.encoded(p)),65536)
    def test_decimal_total_above_browser_precision_remains_exact(self):
        source=self.full();source['total_combinations']=2**60+1
        with patch('app.candidate_views.preview_experiment',return_value=source):p=self.page()
        self.assertEqual(p['total_combinations_exact'],'1152921504606846977');self.assertEqual(p['next_offset'],8)
    def test_production_byte_budget_shortens_maximum_unicode_identity_rows_without_truncation(self):
        base=self.full();rows=[]
        for i in range(25):
            row=copy.deepcopy(base['candidates'][0]);row['id']=f'{i:064x}'
            for r in row['references'].values():r['id']='😀'*200
            row['coordinates']['model']['id']='😀'*200
            for key in ('data_task','split','protocol'):row['coordinates'][key]='😀'*200
            row['reason']='λ😀'*10000;rows.append(row)
        def provider(*a,offset=0,limit=25,**kw):
            return {**base,'offset':offset,'limit':limit,'total_combinations':25,'candidates':rows[offset:offset+limit]}
        seen=[];offset=0;calls=0
        with patch('app.candidate_views.preview_experiment',side_effect=provider):
            while True:
                p=self.page(offset=offset);calls+=1;self.assertLessEqual(len(science_browser.encoded(p)),65536)
                seen.extend(p['candidates'])
                for r in p['candidates']:
                    self.assertEqual(r['references'],rows[0]['references']);self.assertEqual(r['coordinates'],rows[0]['coordinates'])
                if p['next_offset'] is None:break
                self.assertEqual(p['next_offset'],offset+len(p['candidates']));offset=p['next_offset']
        self.assertEqual([r['id'] for r in seen],[r['id'] for r in rows]);self.assertGreater(calls,1)
    def test_wrong_kinds_missing_revisions_and_closed_page_arguments_fail(self):
        for kw in ({'offset':True},{'offset':-1},{'offset':2**53},{'limit':True},{'limit':0},{'limit':26},{'limit':None}):
            with self.subTest(kw=kw),self.assertRaises(GraphError):self.page(**kw)
        for ref in ({**self.ref,'kind':'task'},{**self.ref,'revision':'0'*64},{**self.ref,'extra':1}):
            with self.subTest(ref=ref),self.assertRaises((GraphError,KeyError)):candidate_views.stored(self.w,ref)
    def test_one_oversized_candidate_fails_explicitly_instead_of_silent_empty_preview(self):
        with patch.object(science_browser,'MAX_BYTES',100),self.assertRaisesRegex(GraphError,'cannot fit'):self.page()
    def test_no_skipped_candidate_enumeration_at_large_exact_offset(self):
        with patch('app.experiment_objects.candidate',side_effect=AssertionError('No skipped enumeration')):
            self.assertEqual(self.page(offset=2**53-1)['candidates'],[])
    def test_cli_projection_parity_readonly_and_complete_compatibility_command(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'candidate.vtw';self.w.save(path);before=path.read_bytes();mtime=path.stat().st_mtime_ns
            def cli(command,*extra):
                out=io.StringIO()
                with redirect_stdout(out):code=vpw_cli.main([str(path),command,self.ref['id'],self.ref['revision'],*extra])
                return code,json.loads(out.getvalue())
            code,result=cli('preview-experiment-page');self.assertEqual(code,0,result);self.assertEqual(result['value'],self.page())
            code,result=cli('preview-experiment');self.assertEqual(code,0,result);self.assertIn('manifest',result['value']['candidates'][0]);self.assertIn('protocol',result['value']['candidates'][0])
            self.assertEqual(cli('preview-experiment-page','--limit','26')[0],1)
            self.assertEqual(path.read_bytes(),before);self.assertEqual(path.stat().st_mtime_ns,mtime)
    def test_fresh_projection_import_does_not_load_numerical_or_qt_runtimes(self):
        p=subprocess.run([sys.executable,'-c',"import sys;import app.candidate_views;assert not any(k in sys.modules for k in ('torch','numpy','PyQt6'))"],capture_output=True,text=True,timeout=10)
        self.assertEqual(p.returncode,0,p.stderr)
