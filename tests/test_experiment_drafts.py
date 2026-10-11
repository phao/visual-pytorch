"""Owned experimental axes: native literal controls, local pins and preview bounds."""
import copy,json,random,unittest
from unittest.mock import patch
from app import authoring,experiment_drafts,science_browser
from app.authoring_sessions import Sessions,_at
from app.core import Workspace,GraphError
from test_experiment_authoring import fixture

class ExperimentDraftTests(unittest.TestCase):
    def setUp(self):
        self.w=Workspace();self.addCleanup(self.w.db.close)
        self.addCleanup(lambda:self.w.history.directory.cleanup() if self.w.history.directory else None)
        self.e=fixture(self.w);self.s=Sessions(available_fn=lambda:2**31);self.owner=object()
        self.addCleanup(lambda:self.s.close_owner(self.owner))
    def begin(self,doc=None):
        ref=self.w.put_spec('experiment',doc or self.e)
        return self.s.open(self.w,self.owner,'experiment',base=ref),ref
    def doc(self,h):return self.s._sessions[h['session']]['draft']['document']
    def page(self,h,c='protocols',**kw):return self.s.experiment_page(h['session'],self.owner,h['sequence'],c,**kw)
    def edit(self,h,ops):return self.s.experiment_edit(self.w,h['session'],self.owner,h['sequence'],ops)
    def op(self,c,key,text,ordinal=0):
        return {'op':'property','target':{'kind':'entry','collection':c,'ordinal':ordinal},'key':key,'text':text}

    def test_all_collection_descriptors_are_exact_readonly_and_not_persistable(self):
        h,_=self.begin();sql=self.w.db.serialize();rng=random.getstate()
        for c in experiment_drafts.COLLECTIONS:
            result=self.page(h,c);self.assertEqual(result['scope'],'owned-experiment-page')
            self.assertEqual(_at(self.doc(h),result['path']),experiment_drafts._collection(self.doc(h),c)[0])
            self.assertEqual([r['ordinal'] for r in result['rows']],list(range(len(result['rows']))))
            for r in result['rows']:
                for v in r['properties']:
                    if v['present'] and v['text'] is not None:self.assertEqual(json.loads(v['text']),_at(self.doc(h),v['path']))
            self.assertLessEqual(len(science_browser.encoded(result)),65536)
            with self.assertRaises(GraphError):self.w.put_spec('experiment',result)
        self.assertEqual(self.w.db.serialize(),sql);self.assertEqual(random.getstate(),rng)
        result=self.page(h,'models');self.assertEqual(json.loads(result['rows'][0]['properties'][0]['text']),self.e['models'][0])

    def test_hidden_large_roles_and_questions_are_never_encoded_to_discover_preview_size(self):
        e=copy.deepcopy(self.e);e['question']='λ😀'*10000
        e['splits']=[{'id':'external','protocol':{'method':'external',
            'roles':{'train':['r'+str(i) for i in range(20000)],'validation':[],'test':['r-end']},'excluded':[]}}]
        h,_=self.begin(e);doc=self.doc(h);roles=doc['splits'][0]['protocol']['roles']['train'];dumps=json.dumps
        def guarded(value,*args,**kw):
            self.assertIsNot(value,roles);self.assertIsNot(value,doc['question']);return dumps(value,*args,**kw)
        with patch('app.dataset_drafts.json.dumps',side_effect=guarded):result=self.page(h,'splits')
        train=next(v for v in result['rows'][0]['properties'] if v['key']=='roles.train')
        self.assertIsNone(train['text']);self.assertEqual(train['summary']['size'],20000)
        self.assertIsNone(next(v for v in result['roots'] if v['key']=='question')['text'])
        located=self.s.experiment_value(h['session'],self.owner,0,{'kind':'entry','collection':'splits','ordinal':0},'roles.train')
        self.assertEqual(located['property'],train)

    def test_card_pages_cover_actual_scientific_cursors_including_byte_shortening(self):
        e=copy.deepcopy(self.e);e['initialization_seeds']=list(range(40));h,_=self.begin(e)
        expected=self.page(h,'initialization_seeds',limit=1);budget=len(science_browser.encoded(expected))+100
        seen=[];offset=0;calls=0
        with patch.object(science_browser,'MAX_BYTES',budget):
            while True:
                page=self.page(h,'initialization_seeds',offset=offset);calls+=1
                self.assertLessEqual(len(science_browser.encoded(page)),budget);seen.extend(r['ordinal'] for r in page['rows'])
                if page['next_offset'] is None:break
                self.assertEqual(page['next_offset'],offset+len(page['rows']));offset=page['next_offset']
        self.assertEqual(seen,list(range(40)));self.assertGreater(calls,7)
        self.assertEqual(self.page(h,'initialization_seeds',offset=100)['rows'],[])

    def test_named_nested_properties_resolve_after_prior_removals_and_preserve_literal_tokens(self):
        h,_=self.begin();sql=self.w.db.serialize()
        h=self.edit(h,[self.op('protocols','budget.epochs',None),self.op('protocols','budget.updates','9007199254740993'),
            self.op('protocols','budget.seconds','2.5'),self.op('protocols','optimizer.betas.1','0.98'),
            {'op':'property','target':{'kind':'document'},'key':'question','text':'"Changed question"'}])
        p=self.doc(h)['protocols'][0]['protocol'];self.assertNotIn('epochs',p['budget'])
        self.assertEqual(p['budget']['updates'],9007199254740993);self.assertEqual(p['budget']['seconds'],2.5)
        self.assertEqual(p['optimizer']['betas'],[.9,.98]);self.assertEqual(self.doc(h)['question'],'Changed question')
        self.assertEqual(self.w.db.serialize(),sql);self.assertEqual(h['sequence'],1)
        h=self.edit(h,[self.op('protocols','budget.epochs','null'),self.op('protocols','budget.updates','3')])
        self.assertIsNone(self.doc(h)['protocols'][0]['protocol']['budget']['epochs'])
        self.assertEqual(self.s.check(self.w,h['session'],self.owner,2)['kind'],'experiment')

    def test_closed_target_grammar_midbatch_failures_and_identity_changes_rollback(self):
        h,_=self.begin();before=copy.deepcopy(self.doc(h));good=self.op('protocols','budget.seconds','1')
        for bad in [
            self.op('protocols','optimizer.kind','"SGD"'),self.op('models','reference','NaN'),
            {**good,'target':{'kind':'entry','collection':'protocols','ordinal':True}},
            {**good,'target':{'kind':'entry','collection':'protocols','ordinal':99}},
            {**good,'target':{'kind':'entry','collection':'protocols','ordinal':0,'extra':0}},
            {**good,'key':'budget.epochs','text':False},{**good,'text':'{"a":1,"a":2}'},
            {**good,'target':{'kind':'document'},'key':'id','text':'"replacement"'},
            {'op':'put-json','path':[],'key':'id','text':'"replacement"'},
        ]:
            with self.assertRaises(GraphError):self.edit(h,[good,bad])
            self.assertEqual(self.doc(h),before);self.assertEqual(self.s.describe(h['session'],self.owner),h)
        for ops in [[],[good]*26]:
            with self.assertRaises(GraphError):self.edit(h,ops)

    def test_nested_locations_after_reordering_and_same_session_child_staging(self):
        h,_=self.begin();a=self.page(h,'data_tasks');second=self.e['data_tasks'][1]['id']
        h=self.edit(h,[{'op':'move','path':a['path'],'from':1,'to':0},self.op('data_tasks','id','"renamed"')])
        self.assertEqual(self.doc(h)['data_tasks'][1]['id'],self.e['data_tasks'][0]['id'])
        target={'kind':'entry','collection':'data_tasks','ordinal':0}
        value=self.s.experiment_value(h['session'],self.owner,1,target,'id')
        self.assertEqual(json.loads(value['property']['text']),'renamed')
        b=self.s.start_value(self.w,h['session'],self.owner,1,value['property']['path'])
        b=self.s.splice_value(self.w,h['session'],self.owner,1,b['value'],0,offset=0,delete_bytes=9,text=json.dumps(second))
        h=self.s.finish_value(self.w,h['session'],self.owner,1,b['value'],1)
        self.assertEqual(self.doc(h)['data_tasks'][0]['id'],second);self.assertEqual(h['sequence'],2)
        with self.assertRaisesRegex(GraphError,'sequence changed'):self.s.experiment_value(h['session'],self.owner,1,target,'id')

    def test_local_choices_sort_browsing_only_search_and_pin_exact_selected_values(self):
        h,_=self.begin();before=copy.deepcopy(self.doc(h));sql=self.w.db.serialize();ref=self.e['models'][1]
        choices=self.s.experiment_choices(h['session'],self.owner,0,'models',query='nothing',pin=ref)
        self.assertEqual(choices['rows'],[]);self.assertEqual(choices['pinned']['value'],ref);self.assertIsNone(choices['missing_pin'])
        choices=self.s.experiment_choices(h['session'],self.owner,0,'models',query=ref['revision'])
        self.assertEqual(choices['rows'][0]['value'],ref)
        missing={**ref,'revision':'0'*64}
        result=self.s.experiment_choices(h['session'],self.owner,0,'models',pin=missing)
        self.assertEqual(result['missing_pin'],missing);self.assertIsNone(result['pinned'])
        result=self.s.experiment_choices(h['session'],self.owner,0,'data_tasks',pin='pair-1')
        self.assertEqual([r['value'] for r in result['rows']],['pair-0','pair-1'])
        self.assertEqual(self.doc(h),before);self.assertEqual(self.w.db.serialize(),sql)

    def test_local_choices_preserve_incomplete_entries_and_explain_duplicate_pins(self):
        h,_=self.begin();path=self.page(h,'models')['path'];pair=self.e['data_tasks'][0]
        h=self.edit(h,[{'op':'insert-json','path':path,'index':0,'text':'{"kind":"model","id":"","revision":""}'},
            self.op('data_tasks','id',json.dumps(pair['id']),ordinal=1)])
        result=self.s.experiment_choices(h['session'],self.owner,1,'models')
        self.assertEqual(len(result['rows']),3);self.assertTrue(any(not r['valid'] and r['value'] is None for r in result['rows']))
        result=self.s.experiment_choices(h['session'],self.owner,1,'data_tasks',pin=pair['id'])
        self.assertTrue(result['ambiguous_pin']);self.assertEqual([r['ordinal'] for r in result['rows']],[0,1])
        with self.assertRaises(GraphError):self.s.check(self.w,h['session'],self.owner,1)

    def test_local_choice_byte_cursors_keep_outside_pins_and_all_original_ordinals(self):
        e=copy.deepcopy(self.e);pair=e['data_tasks'][0]
        e['data_tasks']=[{**pair,'id':'p'+str(i).zfill(3)} for i in range(39,-1,-1)]
        h,_=self.begin(e);before=copy.deepcopy(self.doc(h));pin='p039'
        one=self.s.experiment_choices(h['session'],self.owner,0,'data_tasks',limit=1,pin=pin)
        budget=len(science_browser.encoded(one))+30;seen=[];offset=0
        with patch.object(science_browser,'MAX_BYTES',budget):
            while True:
                result=self.s.experiment_choices(h['session'],self.owner,0,'data_tasks',offset=offset,pin=pin)
                self.assertLessEqual(len(science_browser.encoded(result)),budget)
                self.assertEqual(result['pinned']['value'],pin);self.assertFalse(result['ambiguous_pin'])
                seen.extend((r['value'],r['ordinal']) for r in result['rows'])
                if result['next_offset'] is None:break
                self.assertEqual(result['next_offset'],offset+len(result['rows']));offset=result['next_offset']
        self.assertEqual(seen,[('p'+str(i).zfill(3),39-i) for i in range(40)])
        result=self.s.experiment_choices(h['session'],self.owner,0,'data_tasks',offset=999,pin=pin)
        self.assertEqual(result['rows'],[]);self.assertIsNone(result['next_offset'])
        self.assertEqual(self.doc(h),before)

    def test_maximum_unicode_references_are_complete_and_never_replaced_by_browsing_heads(self):
        e=copy.deepcopy(self.e);ref={'kind':'model','id':'😀'*200,'revision':'a'*64}
        e['models']=[ref];e['selection']={'kind':'paired',
            'pairs':[{'model':ref,'data_task':e['data_tasks'][0]['id']}],
            'exclusions':[{'model':ref,'data_task':e['data_tasks'][0]['id'],'reason':'negative control'}]}
        # A valid immutable reference may be absent locally. Read controls must
        # preserve it; validation/preview, not browsing, decides admissibility.
        h,_=self.begin()
        # The document exceeds the edit-fragment budget; use the ordinary raw
        # linked staging path rather than widening transport limits for a test.
        text=json.dumps(e,ensure_ascii=False,separators=(',',':'))
        b=self.s.start_value(self.w,h['session'],self.owner,0,[])
        original=b['total_bytes'];first=True;appended=0
        for offset in range(0,len(text),1000):
            part=text[offset:offset+1000]
            b=self.s.splice_value(self.w,h['session'],self.owner,0,b['value'],b['value_revision'],
                offset=0 if first else appended,delete_bytes=original if first else 0,text=part)
            appended=(0 if first else appended)+len(part.encode('utf-8'));first=False
        h=self.s.finish_value(self.w,h['session'],self.owner,0,b['value'],b['value_revision'])
        for collection,key in [('models','reference'),('pairs','model'),('exclusions','model')]:
            result=self.page(h,collection)
            descriptor=next(p for p in result['rows'][0]['properties'] if p['key']==key)
            self.assertEqual(json.loads(descriptor['text']),ref);self.assertIsNone(descriptor['summary'])
        result=self.s.experiment_choices(h['session'],self.owner,1,'models',query='no match',pin=ref)
        self.assertEqual(result['pinned']['value'],ref);self.assertEqual(result['rows'],[])
        with self.assertRaises(GraphError):self.s.check(self.w,h['session'],self.owner,1)

    def test_preview_matches_exact_legacy_candidates_without_manifests_protocols_or_training(self):
        h,ref=self.begin();sql=self.w.db.serialize();rng=random.getstate()
        with patch('app.datasets._prepare_column',side_effect=AssertionError('No fitting')):
            page=self.s.experiment_preview(self.w,h['session'],self.owner,0)
            legacy=authoring.experiment_preview(self.w,self.s._sessions[h['session']]['draft'])['preview']
        self.assertEqual(page['experiment'],ref);self.assertEqual(page['total_combinations_exact'],'8')
        self.assertEqual([r['id'] for r in page['candidates']],[r['id'] for r in legacy['candidates']])
        for row,full in zip(page['candidates'],legacy['candidates']):
            self.assertEqual(row['coordinates'],full['coordinates']);self.assertEqual(row['status'],full['status'])
            self.assertEqual(row['counts']['train'],len(full['manifest']['roles']['train']))
            self.assertNotIn('manifest',row);self.assertNotIn('protocol',row)
        self.assertEqual(self.w.db.serialize(),sql);self.assertEqual(random.getstate(),rng)
        self.assertEqual(self.w.list_specs('run',True),[])

    def test_preview_omits_excluded_long_reasons_and_hidden_manifests_with_literal_counts(self):
        h,_=self.begin();d=self.s._sessions[h['session']]['draft'];source=authoring.experiment_preview(self.w,d)
        source['preview']['total_combinations']=2**60+1;source['preview']['candidates'][0]['reason']='λ'*20000
        hidden=source['preview']['candidates'][0]['manifest'];dumps=json.dumps
        def guarded(value,*args,**kw):
            self.assertIsNot(value,hidden);return dumps(value,*args,**kw)
        with patch('app.authoring.experiment_preview',return_value=source),patch('app.science_browser.json.dumps',side_effect=guarded):
            result=experiment_drafts.preview(self.w,d,h)
        self.assertEqual(result['total_combinations_exact'],str(2**60+1))
        self.assertEqual(result['candidates'][0]['reason'],{'offset':0,'text':'λ'*128,'total':20000,'next_offset':128})

    def test_actual_preview_byte_shortening_covers_exact_candidate_order_without_skipping(self):
        h,_=self.begin();draft=self.s._sessions[h['session']]['draft'];sql=self.w.db.serialize()
        expected=authoring.experiment_preview(self.w,draft)['preview']['candidates']
        one=experiment_drafts.preview(self.w,draft,h,limit=1)
        budget=len(science_browser.encoded(one))+50;seen=[];offset=0
        with patch.object(science_browser,'MAX_BYTES',budget):
            while True:
                result=experiment_drafts.preview(self.w,draft,h,offset=offset)
                self.assertLessEqual(len(science_browser.encoded(result)),budget)
                seen.extend(result['candidates'])
                if result['next_offset'] is None:break
                self.assertEqual(result['next_offset'],offset+len(result['candidates']));offset=result['next_offset']
        self.assertEqual([r['id'] for r in seen],[r['id'] for r in expected])
        self.assertEqual([r['coordinates'] for r in seen],[r['coordinates'] for r in expected])
        self.assertEqual(len(seen),8)
        empty=experiment_drafts.preview(self.w,draft,h,offset=99)
        self.assertEqual(empty['candidates'],[]);self.assertIsNone(empty['next_offset'])
        with patch.object(science_browser,'MAX_BYTES',100),self.assertRaisesRegex(GraphError,'cannot fit'):
            experiment_drafts.preview(self.w,draft,h)
        self.assertEqual(self.w.db.serialize(),sql)

    def test_owner_origin_kind_staging_and_ram_guards_preserve_readonly_inspection(self):
        h,_=self.begin();other=Workspace();self.addCleanup(other.db.close)
        other.db.execute("UPDATE metadata SET value=? WHERE key='workspace_id'",(h['workspace_id'],))
        for w,owner,seq in [(other,self.owner,0),(self.w,object(),0),(self.w,self.owner,1)]:
            with self.assertRaises(GraphError):self.s.experiment_preview(w,h['session'],owner,seq)
        self.s.available_fn=lambda:0;self.page(h);self.s.experiment_choices(h['session'],self.owner,0,'models')
        with self.assertRaisesRegex(GraphError,'RAM'):self.s.experiment_preview(self.w,h['session'],self.owner,0)
        with self.assertRaisesRegex(GraphError,'RAM'):self.edit(h,[self.op('protocols','budget.seconds','1')])
        self.s.available_fn=lambda:2**31;b=self.s.start_value(self.w,h['session'],self.owner,0,self.page(h)['roots'][0]['path'])
        with self.assertRaisesRegex(GraphError,'staged value'):self.s.experiment_preview(self.w,h['session'],self.owner,0)
        with self.assertRaisesRegex(GraphError,'staged value'):self.edit(h,[self.op('protocols','budget.seconds','1')])
        self.s.discard_value(h['session'],self.owner,0,b['value'],0);self.s.close(h['session'],self.owner)
        h=self.s.open(self.w,self.owner,'task')
        with self.assertRaises(GraphError):self.page(h)

    def test_closed_limits_missing_containers_and_historical_head_fences(self):
        h,ref=self.begin()
        for collection,kw in [('missing',{}),('models',{'offset':True}),('models',{'limit':7}),('models',{'offset':-1})]:
            with self.assertRaises(GraphError):self.page(h,collection,**kw)
        for kw in [{'collection':'splits'},{'collection':'models','limit':False},{'collection':'data_tasks','pin':{}},{'collection':'models','query':'x'*201}]:
            with self.assertRaises(GraphError):self.s.experiment_choices(h['session'],self.owner,0,**kw)
        for kw in [{'offset':True},{'limit':0},{'limit':26},{'offset':2**53}]:
            with self.assertRaises(GraphError):self.s.experiment_preview(self.w,h['session'],self.owner,0,**kw)
        newer=copy.deepcopy(self.e);newer['name']='New browsing head';self.w.put_spec('experiment',newer)
        with self.assertRaisesRegex(GraphError,'head changed'):self.s.experiment_preview(self.w,h['session'],self.owner,0)
        self.s.close(h['session'],self.owner);h=self.s.open(self.w,self.owner,'experiment',base=ref)
        self.assertEqual(self.s.experiment_preview(self.w,h['session'],self.owner,0)['experiment'],ref)
        h=self.edit(h,[{'op':'put-json','path':[],'key':'protocols','text':'null'}])
        with self.assertRaisesRegex(GraphError,'array'):self.page(h)

if __name__=='__main__':unittest.main()
