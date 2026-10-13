"""Actual source architecture ownership, native/export execution and storage."""
import copy
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.core import GraphError, Workspace, check_document, compile_model, generate_code, validate_model, safety_report
from app.evidence import parameter_schema
from app.policy_graph import plan
from app.revisions import model_document, validate_model_document


def source_graph(kind='TensorParameter', *, input=True, shared=False):
    source = {'id':'tensor', 'type':kind, 'shape':[2]}
    if kind == 'Constant': source['values'] = [-0.0, 2.0]
    nodes = [source, {'id':'output', 'type':'Output', 'label':'result'}]
    edges = [{'from':'tensor', 'to':'output'}]
    if input:
        nodes.insert(0, {'id':'input', 'type':'Input', 'shape':[2], 'label':'x'})
        nodes.insert(-1, {'id':'join', 'type':'Add', 'sources':['input','tensor']})
        edges = [{'from':n, 'to':'join'} for n in ('input','tensor')] + [{'from':'join','to':'output'}]
    if shared:
        nodes.insert(-1, {'id':'alias', 'type':kind, 'shape':[2], 'share_with':'tensor'})
        join = next(n for n in nodes if n['id'] == 'join')
        join['sources'].append('alias')
        edges.append({'from':'alias','to':'join'})
    return {'id':'sources', 'name':'Tensor sources', 'graph':{'nodes':nodes,'edges':edges}}


class TensorSourceGraph(unittest.TestCase):
    def test_pure_declared_source_no_dataset_input(self):
        doc = source_graph(input=False)
        check_document(doc)
        c = validate_model({'sources':doc}, 'sources')
        self.assertEqual(c['inputs'], 0)
        self.assertEqual(c['input_shape'], [])
        self.assertEqual(c['input_groups'], [])
        self.assertEqual(c['parameters'], 2)
        self.assertEqual(c['plan'][0]['sources'], [])
        self.assertEqual(parameter_schema({'sources':doc}, 'sources'),
                         {'layer_0.value':{'shape':[2], 'id':'["sources","tensor","value"]'}})

    def test_constant_buffer_is_not_checkpoint_parameter(self):
        import torch
        doc = source_graph('Constant', input=False)
        model = compile_model({'sources':doc}, 'sources')
        self.assertEqual(list(model.parameters()), [])
        self.assertEqual(dict(model.state_dict()), {})
        self.assertEqual(validate_model({'sources':doc}, 'sources')['buffer_elements'], 2)
        value = model(batch_size=3)
        self.assertEqual(tuple(value.shape), (3,2))
        self.assertTrue(torch.signbit(value[0,0]))
        self.assertEqual(model.parameter_policy_receipt['counts']['stored'], 0)

    def test_source_zero_incoming_required(self):
        doc = source_graph()
        doc['graph']['edges'].append({'from':'input','to':'tensor'})
        with self.assertRaisesRegex(GraphError, 'cannot receive'):
            validate_model({'sources':doc}, 'sources')

    def test_shared_owner_gradients_and_native_generated_parity(self):
        import torch
        doc = source_graph(shared=True)
        models = {'sources':doc}
        c = validate_model(models, 'sources')
        self.assertEqual(c['parameters'], 2)
        schema = parameter_schema(models, 'sources')
        self.assertEqual(len(schema), 1)
        doc['graph']['parameter_policy'] = {'version':1,'parameters':{
            '["tensor","value"]':{'initializer':{'kind':'values','dtype':'float32','values':[1.,-2.]}}}}
        native = compile_model(models, 'sources')
        ns = {}
        exec(generate_code(models, 'sources'), ns)
        generated = ns['model']
        x = torch.tensor([[3.,4.],[5.,6.]], requires_grad=True)
        y = native(x)
        self.assertTrue(torch.equal(y, generated(x)))
        self.assertTrue(torch.equal(y, x + torch.tensor([2.,-4.])))
        y.sum().backward()
        self.assertTrue(torch.equal(native.layer_0.value.grad, torch.tensor([4.,4.])))
        self.assertEqual(len(list(native.parameters())), 1)
        self.assertEqual(plan(models,'sources')['resolved']['counts']['stored'], 2)

    def test_nullary_source_requires_explicit_batch(self):
        import torch
        doc = source_graph(input=False)
        model = compile_model({'sources':doc}, 'sources')
        for kwargs in ({}, {'batch_size':True}, {'batch_size':-1}):
            with self.assertRaises(GraphError): model(**kwargs)
        with self.assertRaises(GraphError): model(torch.ones(3,2),batch_size=3)
        self.assertEqual(tuple(model(batch_size=0).shape), (0,2))
        ns = {}
        exec(generate_code({'sources':doc},'sources'),ns)
        self.assertTrue(torch.equal(model(batch_size=4),ns['model'](batch_size=4)))

    def test_nullary_module_is_explicit_and_shared_buffers_count_once(self):
        import torch
        child = source_graph('Constant',input=False)
        root = {'id':'root','name':'Root','graph':{'nodes':[
            {'id':'a','type':'Module','ref':'sources','batch_only':True},
            {'id':'b','type':'Module','ref':'sources','batch_only':True,'share_with':'a'},
            {'id':'add','type':'Add','sources':['a','b']},
            {'id':'out','type':'Output','label':'out'}], 'edges':[
                {'from':'a','to':'add'},{'from':'b','to':'add'},{'from':'add','to':'out'}]}}
        models = {'root':root,'sources':child}
        c = validate_model(models,'root')
        self.assertEqual(c['buffer_elements'],2)
        native = compile_model(models,'root')
        ns = {}; exec(generate_code(models,'root'),ns)
        self.assertTrue(torch.equal(native(batch_size=2),ns['model'](batch_size=2)))
        root['graph']['nodes'][0].pop('batch_only')
        with self.assertRaises(GraphError): validate_model(models,'root')

    def test_mismatched_shared_shape_refuses(self):
        doc = source_graph(shared=True)
        next(n for n in doc['graph']['nodes'] if n['id']=='alias')['axes']=['features']
        with self.assertRaises(GraphError): validate_model({'sources':doc},'sources')

    def test_constant_buffers_and_batch_guard_before_allocation(self):
        doc = source_graph('Constant',input=False)
        c = validate_model({'sources':doc},'sources')
        with patch('app.core.available_memory',return_value=10**10):
            report = safety_report({'sources':doc},'sources')
        self.assertEqual(report['memory']['components']['literal_buffer_storage'],16)
        model = compile_model({'sources':doc},'sources')
        with patch('app.core.available_memory',return_value=1), patch.object(model.layer_0,'forward',side_effect=AssertionError('allocated')):
            with self.assertRaisesRegex(GraphError, 'blocked tensor-source batch'): model(batch_size=10)

    def test_storage_fences_include_archived_model(self):
        doc = source_graph()
        workspace = Workspace(); workspace.put(doc)
        frozen = model_document(workspace.models(),'sources')
        self.assertEqual(frozen['version'],7)
        ref = workspace.put_spec('model',frozen)
        workspace.archive_spec('model',ref['id'])
        old = copy.deepcopy(frozen); old['version']=6
        with self.assertRaisesRegex(GraphError,'version 7'): validate_model_document(old)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'sources.vtw'
            workspace.save(path)
            loaded = Workspace.load(path)
            self.assertEqual(loaded.get_spec('model',ref['id'],ref['revision'])['version'],7)
            with sqlite3.connect(path) as db:
                db.execute('DELETE FROM architectures')
                db.execute('PRAGMA user_version=26')
            with self.assertRaisesRegex(GraphError,'version-27'): Workspace.load(path)

    def test_actual_optimizer_updates_source_not_constant(self):
        import torch
        doc = source_graph()
        model = compile_model({'sources':doc},'sources')
        x = torch.tensor([[0.,1.],[2.,3.],[-1.,2.]])
        target = x + torch.tensor([1.5,-.5])
        optimizer = torch.optim.SGD(model.parameters(),lr=.5)
        for _ in range(30):
            optimizer.zero_grad(); loss=(model(x)-target).square().mean(); loss.backward(); optimizer.step()
        self.assertLess(loss.item(),1e-12)
        torch.testing.assert_close(model.layer_0.value,torch.tensor([1.5,-.5]))

    def test_family11_shape_symbols_retained_spec10_and_portable_nullary(self):
        import torch
        from app.families import specialize
        from app.family_records import retain, compile_specialization
        from app.family_export import family_source
        source = source_graph(input=False)
        family = {'version':11,'id':'sources-family','name':'Tensor family','root':'sources',
            'models':{'sources':source},'runtime_policy':None,
            'symbols':[{'id':'width','scope':'sources','name':'Width','default':3}],
            'assignments':[{'model':'sources','node':'tensor','field':'shape','value':[{'symbol':'width'}]}],
            'repeats':[]}
        resolved = specialize(family,{'width':5})
        self.assertEqual(resolved['contract']['parameters'],5)
        workspace = Workspace()
        records = retain(workspace,family=family,bindings={'width':5})
        ref = records['specialization']
        self.assertEqual(workspace.get_spec(ref['kind'],ref['id'],ref['revision'])['version'],10)
        native = compile_specialization(workspace,records['specialization'])
        code = family_source(family)['code'];ns={};exec(code,ns)
        portable = ns['ArchitectureFamily'](bindings={'width':5})
        self.assertTrue(torch.equal(native(batch_size=4),portable(batch_size=4)))
        self.assertTrue(all(json.loads(p['id'])[-1]=='value' for p in portable.semantic_parameter_schema().values()))
        old = copy.deepcopy(family);old['version']=10
        # Family10 requires explicit policies and cannot silently accept sources.
        old['models']['sources']['graph']['parameter_policy']={'version':1,'parameters':{}}
        with self.assertRaisesRegex(GraphError,'Family11'): specialize(old)


if __name__ == '__main__': unittest.main()
