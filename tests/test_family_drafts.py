"""Owned typed family edits: exact source, bounded transport and no execution."""
import copy
import json
import random
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from app import authoring, family_drafts, science_browser
from app.authoring_sessions import Sessions
from app.core import GraphError, Workspace, default_model
from app.family_records import retain


def source():
    return json.loads((Path(__file__).resolve().parents[1] / 'examples/families/width-depth.json').read_text())


class FamilyDraftTests(unittest.TestCase):
    def setUp(self):
        self.w = Workspace(); self.addCleanup(self.w.db.close)
        self.s = Sessions(); self.owner = object()
        self.addCleanup(lambda: self.s.close_owner(self.owner))

    def open(self, document=None):
        ref = self.w.put_spec('family', document or source())
        return self.s.open(self.w, self.owner, 'family', base=ref)

    def edit(self, h, operations):
        return self.s.family_edit(self.w, h['session'], self.owner, h['sequence'], operations)

    def property(self, collection, ordinal, key, value):
        return {'op':'property','target':{'kind':'entry','collection':collection,'ordinal':ordinal},
                'key':key,'text':json.dumps(value)}

    def test_seed_captures_only_valid_complete_concrete_closure_readonly(self):
        doc = default_model(); doc['dataset'] = {'private':'not part of a template'}
        self.w.put(doc); before = self.w.db.serialize(); rng = random.getstate()
        h = self.s.open(self.w, self.owner, 'family', model=doc['id'])
        self.assertNotIn('document', h)
        row = self.s._get(h['session'], self.owner)['draft']['document']
        self.assertEqual(row['root'], doc['id'])
        self.assertEqual(set(row['models'][doc['id']]), {'id','name','graph'})
        self.assertEqual(row['symbols'], [])
        self.s.check(self.w,h['session'],self.owner,0)
        self.w.put(dict(doc,name='Later drawing'))
        self.assertNotEqual(row['models'][doc['id']]['name'], 'Later drawing')
        self.assertEqual(rng,random.getstate())
        self.assertNotEqual(before,self.w.db.serialize())  # Only the explicit later drawing edit.

    def test_seed_is_not_allowed_for_non_family_or_historical_base(self):
        ref=self.w.put_spec('family',source())
        for kind,base in [('dataset',None),('family',ref)]:
            with self.subTest(kind=kind),self.assertRaisesRegex(GraphError,'drawing seed'):
                self.s.open(self.w,self.owner,kind,base=base,model='not-existing')
        self.assertEqual(self.s._sessions,{})

    def test_typed_properties_keep_scientific_order_and_exact_integer_text(self):
        h=self.open();before=self.w.db.serialize()
        h=self.edit(h,[self.property('symbols',0,'default',4),self.property('symbols',1,'default',3)])
        page=self.s.family_page(h['session'],self.owner,h['sequence'],'symbols')
        self.assertEqual([p['text'] for p in page['rows'][0]['properties'] if p['key']=='default'],['4'])
        preview=self.s.family_preview(self.w,h['session'],self.owner,h['sequence'])
        self.assertEqual(preview['status'],'verified');self.assertEqual(preview['parameters'],77)
        self.assertEqual(preview['blocks'],10);self.assertEqual(preview['binding_count'],2)
        self.assertEqual(before,self.w.db.serialize())

    def test_conditional_preview_admits_unbound_family_but_not_instance(self):
        h=self.open();op=self.property('symbols',0,'default',0);op['text']=None
        h=self.edit(h,[op]);p=self.s.family_preview(self.w,h['session'],self.owner,h['sequence'])
        self.assertEqual(p['status'],'conditional');self.assertEqual(p['missing_total'],1)
        self.assertEqual(p['missing'][0]['id'],'hidden-width')
        ref=self.s.commit(self.w,h['session'],self.owner,h['sequence'])
        with self.assertRaisesRegex(GraphError,'Unresolved'):
            retain(self.w,family_ref=ref)

    def test_preview_invalid_bindings_and_shape_reason_do_not_mutate_or_fit(self):
        h=self.open();before=self.w.db.serialize();rng=random.getstate()
        with patch('app.tensor_data.fit',side_effect=AssertionError('no fitting'),create=True):
            p=self.s.family_preview(self.w,h['session'],self.owner,0,{'hidden-width':0})
        self.assertEqual(p['status'],'incompatible')
        self.assertEqual(before,self.w.db.serialize());self.assertEqual(rng,random.getstate())

    def test_owner_sequence_and_replacement_working_copy_fences(self):
        h=self.open()
        with self.assertRaises(GraphError):self.s.family_page(h['session'],object(),0,'symbols')
        h=self.edit(h,[self.property('symbols',0,'default',4)])
        with self.assertRaisesRegex(GraphError,'sequence'):self.s.family_page(h['session'],self.owner,0,'symbols')
        other=Workspace();self.addCleanup(other.db.close)
        self.w.db.backup(other.db)
        with self.assertRaisesRegex(GraphError,'origin'):self.s.family_edit(other,h['session'],self.owner,h['sequence'],[self.property('symbols',0,'default',5)])
        self.s.family_page(h['session'],self.owner,h['sequence'],'symbols')  # Read-only old origin remains inspectable.

    def test_head_changed_preserves_historical_draft_but_refuses_admission(self):
        h=self.open();new=source();new['name']='New head';self.w.put_spec('family',new)
        self.assertEqual(self.s.family_page(h['session'],self.owner,0,'symbols')['total'],2)
        with self.assertRaisesRegex(GraphError,'head changed'):self.s.commit(self.w,h['session'],self.owner,0)
        self.assertIn(h['session'],self.s._sessions)

    def test_failed_edit_batch_is_atomic_and_keeps_exact_sequence(self):
        h=self.open();before=self.s.read(h['session'],self.owner,0)
        with self.assertRaisesRegex(GraphError,'Family edit #1'):
            self.edit(h,[self.property('symbols',0,'default',4),{'op':'remove','collection':'symbols','index':999}])
        self.assertEqual(self.s.describe(h['session'],self.owner)['sequence'],0)
        self.assertEqual(before,self.s.read(h['session'],self.owner,0))

    def test_stable_root_id_and_template_id_cannot_be_forged(self):
        h=self.open()
        for target,key in [({'kind':'document'},'id'),({'kind':'entry','collection':'templates','ordinal':0},'id')]:
            with self.subTest(key=key),self.assertRaises(GraphError):
                self.edit(h,[{'op':'property','target':target,'key':key,'text':'"new-id"'}])
        self.assertEqual(self.s.describe(h['session'],self.owner)['sequence'],0)

    def test_duplicate_keys_nonfinite_and_oversized_values_are_not_coerced(self):
        h=self.open()
        for text in ['{"symbol":"hidden-width","symbol":"hidden-depth"}','NaN','"'+('x'*2001)+'"']:
            with self.subTest(text=text[:20]),self.assertRaises(GraphError):
                op=self.property('assignments',0,'value',0);op['text']=text;self.edit(h,[op])
        self.assertEqual(self.s.describe(h['session'],self.owner)['sequence'],0)

    def test_ordered_insert_remove_and_move_preserve_explicit_semantics(self):
        h=self.open();h=self.edit(h,[{'op':'move','collection':'assignments','from':0,'to':1}])
        page=self.s.family_page(h['session'],self.owner,1,'assignments')
        values=[json.loads(next(p['text'] for p in r['properties'] if p['key']=='node')) for r in page['rows']]
        self.assertEqual(values[:2],[source()['assignments'][1]['node'],source()['assignments'][0]['node']])
        h=self.edit(h,[{'op':'insert','collection':'symbols','index':2,'value':{'id':'unused','scope':'mlp','name':'new'}}])
        with self.assertRaisesRegex(GraphError,'dependent'):self.s.check(self.w,h['session'],self.owner,2)
        h=self.edit(h,[{'op':'remove','collection':'symbols','index':2}])
        self.s.check(self.w,h['session'],self.owner,3)

    def test_staged_value_blocks_typed_edit_preview_and_admission(self):
        h=self.open();value=self.s.start_value(self.w,h['session'],self.owner,0,[])
        for action in [lambda:self.edit(h,[self.property('symbols',0,'default',4)]),
                       lambda:self.s.family_preview(self.w,h['session'],self.owner,0),
                       lambda:self.s.commit(self.w,h['session'],self.owner,0)]:
            with self.assertRaisesRegex(GraphError,'staged value'):action()
        self.s.discard_value(h['session'],self.owner,0,value['value'],value['value_revision'])
        self.s.family_preview(self.w,h['session'],self.owner,0)

    def test_cards_and_long_graph_descriptor_are_bounded_without_clipping_json(self):
        doc=source();doc['models']['mlp']['graph']['nodes'][0]['label']='λ'*10000
        # Incomplete draft is deliberately inspectable; only check/retain enforce grammar.
        h=self.s.open(self.w,self.owner,'family')
        h=self.s.patch(self.w,h['session'],self.owner,0,[{'op':'replace','path':[],'value':dict(doc,id=h['id'])}])
        page=self.s.family_page(h['session'],self.owner,h['sequence'],'templates')
        graph=next(p for p in page['rows'][0]['properties'] if p['key']=='graph')
        self.assertIsNone(graph['text']);self.assertEqual(graph['summary']['type'],'object')
        self.assertLessEqual(len(science_browser.encoded(page)),65536)

    def test_integer_preview_uses_exact_default_and_rejects_boolean(self):
        h=self.open()
        with self.assertRaisesRegex(GraphError,'boolean'):
            self.s.family_preview(self.w,h['session'],self.owner,0,{'hidden-width':True})

    def test_unmodified_exact_family_commit_has_no_dirty_write(self):
        h=self.open();self.w.dirty=False;before=self.w.db.serialize()
        ref=self.s.commit(self.w,h['session'],self.owner,0)
        self.assertEqual(ref,h['base']);self.assertFalse(self.w.dirty);self.assertEqual(before,self.w.db.serialize())

    def test_closed_unknown_collection_property_and_extra_fields_preserve_draft(self):
        h=self.open()
        for op in [self.property('symbols',0,'arbitrary',1),{'op':'remove','collection':'invented','index':0},
                   {'op':'property','target':{'kind':'document'},'key':'name','text':'"x"','extra':True}]:
            with self.subTest(op=op),self.assertRaises(GraphError):self.edit(h,[op])
        self.assertEqual(self.s.describe(h['session'],self.owner)['sequence'],0)

    def test_many_long_missing_symbols_are_explicitly_partial_not_clipped_json(self):
        doc=source()
        doc['symbols']=[{'id':'é'*95+str(i),'scope':'mlp','name':'λ'*95+str(i)} for i in range(128)]
        leaves=[{'symbol':s['id']} for s in doc['symbols']]
        expr={'op':'add','args':[{'op':'add','args':leaves[i:i+8]} for i in range(0,128,8)]}
        doc['assignments']=[{'model':'mlp','node':'stem','field':'features','value':expr}]
        doc['repeats'][0]['count']=2
        h=self.open(doc)
        p=self.s.family_preview(self.w,h['session'],self.owner,0)
        self.assertEqual(p['missing_total'],128);self.assertEqual(len(p['missing']),25)
        self.assertEqual(p['missing_next_offset'],25);self.assertLessEqual(len(science_browser.encoded(p)),65536)
        bound=self.s.family_preview(self.w,h['session'],self.owner,0,{s['id']:1 for s in doc['symbols']})
        self.assertEqual(bound['status'],'verified');self.assertEqual(bound['binding_count'],128)
        self.assertIsNone(bound['resolved_bindings']['text'])
        self.assertEqual(bound['resolved_bindings']['summary']['size'],128)
        self.assertLessEqual(len(science_browser.encoded(bound)),65536)

    def test_fresh_standard_library_family_authoring_imports_no_numerical_runtime(self):
        code=('import sys; from app.core import Workspace,default_model; from app.authoring_sessions import Sessions; '
              'w=Workspace();d=default_model();w.put(d);s=Sessions();o=object();h=s.open(w,o,"family",model=d["id"]);'
              'p=s.family_preview(w,h["session"],o,0);assert p["parameters"]==3;'
              'assert not any(n=="torch" or n=="numpy" or n.startswith("PyQt") for n in sys.modules);s.close_owner(o);w.db.close()')
        result=subprocess.run([sys.executable,'-c',code],text=True,capture_output=True,timeout=15)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)

    def test_seed_never_reads_dataset_payloads_or_unrelated_architectures(self):
        doc=default_model();self.w.put(doc);other=default_model(name='Unrelated');self.w.put(other)
        # A corrupt unrelated dataset is not a seed dependency; opening does not
        # silently load it just because it happens to share the workspace.
        self.w.db.execute('INSERT INTO datasets VALUES (?,?)',(other['id'],'not JSON'));self.w.db.commit()
        reads=[];self.w.db.set_trace_callback(reads.append)
        with patch.object(self.w,'models',side_effect=AssertionError('no all-model/data parse')):
            h=self.s.open(self.w,self.owner,'family',model=doc['id'])
        self.assertEqual(self.s.family_preview(self.w,h['session'],self.owner,0)['parameters'],3)
        self.assertFalse(any('FROM datasets' in q for q in reads))
        self.assertFalse(any('SELECT document' in q and other['id'] in q for q in reads))

    def test_source_size_admission_precedes_seed_document_parsing(self):
        doc=default_model();self.w.put(doc);reads=[];self.w.db.set_trace_callback(reads.append)
        with self.assertRaisesRegex(GraphError,'budget'):
            authoring.drawing_seed(self.w,doc['id'],lambda size: (_ for _ in ()).throw(GraphError('budget')))
        self.assertTrue(any('length(' in q for q in reads))
        self.assertFalse(any('SELECT document' in q for q in reads))

    def test_seed_retains_reachable_module_closure_not_unrelated_models(self):
        child=default_model(name='Child');self.w.put(child)
        root=default_model(name='Parent');root['graph']['nodes'][1]={'id':'linear1','type':'Module','ref':child['id']}
        self.w.put(root);other=default_model(name='Unrelated');self.w.put(other)
        h=self.s.open(self.w,self.owner,'family',model=root['id'])
        doc=self.s._get(h['session'],self.owner)['draft']['document']
        self.assertEqual(set(doc['models']),{root['id'],child['id']})
        self.assertEqual(self.s.family_preview(self.w,h['session'],self.owner,0)['parameters'],3)


if __name__ == '__main__': unittest.main()
