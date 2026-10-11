"""Softmax contracts, independent scalar values/VJP and real trained export."""
import copy
import io
import json
import math
import random
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch
import torch
from app.activations import normalize_activation, softmax_axis
from app.core import (GraphError, Workspace, check_document, compile_model,
                      default_model, generate_code, generate_predictor,
                      restore_predictor, train_model, validate_model)
from app.evidence import identity, normalize_models, check_checkpoint
from app.revisions import model_document
from app.operators import normalized
import vpw_cli

torch.set_num_threads(1)


def model(shape, dim=-1, linear=False):
    d=default_model('Softmax probe',inputs=shape[-1],outputs=shape[-1])
    nodes=[{'id':'input','type':'Input','shape':list(shape)}]
    if linear:nodes.append({'id':'linear','type':'Linear','features':shape[-1],'bias':True})
    nodes.extend([{'id':'softmax','type':'Activation','activation':'Softmax','dim':dim},
                  {'id':'output','type':'Output'}])
    d['graph']={'nodes':nodes,'edges':[{'from':a['id'],'to':b['id']} for a,b in zip(nodes,nodes[1:])]}
    return d


def contract(d):return validate_model({d['id']:d},d['id'])
def compiled(d):return compile_model({d['id']:d},d['id'],constrained=False)


class SoftmaxTests(unittest.TestCase):
    def test_default_is_explicit_last_sample_axis_without_mutating_the_caller(self):
        n={'id':'s','type':'Activation','activation':'Softmax'};before=copy.deepcopy(n)
        self.assertEqual(normalized(n),dict(n,dim=-1));self.assertEqual(n,before)
        for name in ('ReLU','Tanh','Sigmoid','GELU'):
            legacy=dict(n,activation=name)
            self.assertEqual(normalized(legacy),legacy)

    def test_all_axes_for_sample_ranks_one_to_four_keep_shape_and_axis_names(self):
        for shape in ([3],[2,3],[2,3,4],[2,2,3,4]):
            for dim in range(-len(shape),len(shape)):
                with self.subTest(shape=shape,dim=dim):
                    d=model(shape,dim);d['graph']['nodes'][0]['axes']=['axis'+str(i) for i in range(len(shape))]
                    c=contract(d);self.assertEqual(c['output_shape'],list(shape))
                    self.assertEqual(c['output_axes'],d['graph']['nodes'][0]['axes'])
                    self.assertEqual(c['parameters'],0);self.assertFalse(c['affine'])
                    x=torch.linspace(-2,3,2*math.prod(shape)).reshape(2,*shape)
                    y=compiled(d)(x);axis=dim%len(shape)+1
                    self.assertEqual(y.shape,x.shape);self.assertTrue(bool((y>=0).all()))
                    torch.testing.assert_close(y.sum(axis),torch.ones_like(y.sum(axis)),rtol=1e-6,atol=1e-6)

    def test_independent_stable_scalar_values_and_analytic_vector_jacobian_products(self):
        for shape in ([5],[2,3],[2,3,4]):
            for dim in range(len(shape)):
                with self.subTest(shape=shape,dim=dim):
                    x=torch.linspace(-3,2,2*math.prod(shape),dtype=torch.float64).reshape(2,*shape).requires_grad_()
                    u=torch.linspace(-.8,1.7,x.numel(),dtype=torch.float64).reshape_as(x)
                    axis=dim+1;rows=x.detach().movedim(axis,-1).reshape(-1,shape[dim]).tolist()
                    upstream=u.movedim(axis,-1).reshape(-1,shape[dim]).tolist();values=[];gradients=[]
                    for row,weights in zip(rows,upstream):
                        exps=[math.exp(v-max(row)) for v in row];denominator=math.fsum(exps)
                        y=[v/denominator for v in exps];dot=math.fsum(a*b for a,b in zip(y,weights))
                        values.append(y);gradients.append([p*(weight-dot) for p,weight in zip(y,weights)])
                    moved=x.detach().movedim(axis,-1).shape
                    expected=torch.tensor(values,dtype=torch.float64).reshape(moved).movedim(-1,axis)
                    grad=torch.tensor(gradients,dtype=torch.float64).reshape(moved).movedim(-1,axis)
                    y=compiled(model(shape,dim)).double()(x);(y*u).sum().backward()
                    torch.testing.assert_close(y,expected,atol=1e-12,rtol=1e-12)
                    torch.testing.assert_close(x.grad,grad,atol=1e-12,rtol=1e-12)

    def test_large_logits_and_translation_do_not_overflow_or_couple_batches(self):
        m=compiled(model([3]));x=torch.tensor([[1000.,1001.,999.],[-1000.,-999.,-1001.]])
        y=m(x);self.assertTrue(bool(torch.isfinite(y).all()))
        torch.testing.assert_close(y[0],y[1],atol=1e-7,rtol=1e-7)
        torch.testing.assert_close(y,m(x+500),atol=1e-7,rtol=1e-7)
        self.assertTrue(bool(((y>=0)&(y<=1)).all()))

    def test_singleton_axis_produces_ones_and_zero_gradient(self):
        x=torch.tensor([[-2.],[8.]],requires_grad=True);y=compiled(model([1]))(x)
        self.assertTrue(torch.equal(y,torch.ones_like(y)));y.sum().backward()
        self.assertTrue(torch.equal(x.grad,torch.zeros_like(x)))

    def test_independent_input_paths_join_before_five_way_softmax(self):
        d=default_model('Joined probability vector')
        d['graph']={'nodes':[{'id':'left','type':'Input','features':2},
                            {'id':'right','type':'Input','features':3},
                            {'id':'join','type':'Concat','sources':['left','right']},
                            {'id':'s','type':'Activation','activation':'Softmax'},
                            {'id':'output','type':'Output'}],
                    'edges':[{'from':'left','to':'join'},{'from':'right','to':'join'},
                             {'from':'join','to':'s'},{'from':'s','to':'output'}]}
        self.assertEqual(contract(d)['output_shape'],[5])
        y=compiled(d)({'left':torch.tensor([[1.,2.],[3.,4.]]),
                       'right':torch.tensor([[3.,4.,5.],[5.,6.,7.]])})
        exps=[math.exp(v-5) for v in [1,2,3,4,5]];total=math.fsum(exps)
        expected=torch.tensor([[v/total for v in exps]]*2)
        torch.testing.assert_close(y,expected,atol=1e-7,rtol=1e-7)

    def test_invalid_axis_type_rank_and_unknown_settings_fail_before_layer_allocation(self):
        for dim in (True,False,1.,None,'-1',[],8,-9,3,-4):
            with self.subTest(dim=dim),patch('torch.nn.Softmax',side_effect=AssertionError('allocated')):
                d=model([2,3],dim)
                with self.assertRaises(GraphError):compiled(d)
        for key,value in [('temperature',2),('op_version',2),('dtype','float64')]:
            d=model([3]);d['graph']['nodes'][1][key]=value
            with self.subTest(key=key),self.assertRaises(GraphError):check_document(d)
            with self.assertRaises(GraphError):generate_code({d['id']:d},d['id'])

    def test_rank_eight_axis_validation_is_allocation_free(self):
        d=model([1]*7+[3],-8);self.assertEqual(contract(d)['output_shape'],[1]*7+[3])
        self.assertEqual(softmax_axis(d['graph']['nodes'][1],8),0)
        self.assertIn('nn.Softmax(dim=1)',generate_code({d['id']:d},d['id']))

    def test_generated_native_and_separate_standalone_models_have_identical_values(self):
        for dim in (0,1,-1,-2):
            d=model([2,3],dim);code=generate_code({d['id']:d},d['id'])
            self.assertIn('nn.Softmax(dim='+str(dim%2+1)+')',code)
            namespace={};exec(code,namespace)
            x=torch.arange(12,dtype=torch.float32).reshape(2,2,3)/4
            self.assertTrue(torch.equal(compiled(d)(x),namespace['model'](x)))
        d=model([3]);code=generate_code({d['id']:d},d['id'])
        with tempfile.TemporaryDirectory() as folder:
            # Send generated source over stdin: no app package required by export.
            result=subprocess.run([sys.executable,'-c',
                'import sys,json;exec(sys.stdin.read());assert not any(k=="app" or k.startswith("app.") for k in sys.modules);print(json.dumps(model(torch.tensor([[1000.,1001.,999.]])).tolist()))'],
                input=code,text=True,capture_output=True,cwd=folder,timeout=20)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual(json.loads(result.stdout),compiled(d)(torch.tensor([[1000.,1001.,999.]])).tolist())

    def test_workspace_sql_reload_frozen_default_and_history_preserve_axis(self):
        w=Workspace();self.addCleanup(w.db.close);self.addCleanup(lambda:w.history.directory.cleanup() if w.history.directory else None)
        d=model([2,3]);d['graph']['nodes'][1].pop('dim');w.put(d)
        self.assertEqual(w.models()[d['id']]['graph']['nodes'][1]['dim'],-1)
        frozen=model_document(w.models(),d['id']);self.assertEqual(frozen['models'][d['id']]['graph']['nodes'][1]['dim'],-1)
        changed=w.models()[d['id']];changed['graph']['nodes'][1]['dim']=0
        before=w.history.capture(w.db);w.put(changed);w.history.record(before,w.db)
        self.assertTrue(w.history.restore(w))
        self.assertEqual(w.models()[d['id']]['graph']['nodes'][1]['dim'],-1)
        self.assertTrue(w.history.restore(w,redo=True))
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'softmax.vtw';w.save(path);other=Workspace.load(path)
            try:self.assertEqual(other.models()[d['id']]['graph']['nodes'][1]['dim'],0)
            finally:other.db.close()

    def test_dimension_changes_computation_identity_but_folding_does_not(self):
        d=model([2,3]);models={d['id']:d};initial=identity(models,d['id'])
        d['graph']['nodes'][1]['collapsed']=True;d['graph']['nodes'][1]['x']=44
        self.assertEqual(identity(models,d['id']),initial)
        d['graph']['nodes'][1]['dim']=0;self.assertNotEqual(identity(models,d['id']),initial)

    def test_pure_contract_codegen_cli_do_not_import_torch_in_fresh_host(self):
        d=model([2,3],0)
        source='import json,sys;from app.core import validate_model,generate_code;m=json.loads(sys.stdin.read());c=validate_model({m["id"]:m},m["id"]);assert c["output_shape"]==[2,3];assert "nn.Softmax(dim=1)" in generate_code({m["id"]:m},m["id"]);assert "torch" not in sys.modules;print("PASS")'
        result=subprocess.run([sys.executable,'-c',source],input=json.dumps(d),text=True,capture_output=True,timeout=10)
        self.assertEqual(result.returncode,0,result.stderr);self.assertEqual(result.stdout.strip(),'PASS')

    def test_cli_atomic_axis_edits_validate_export_and_preserve_failed_file_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'softmax.vtw';w=Workspace();d=model([2,3]);w.put(d);w.save(path);w.db.close()
            def call(*args):
                out=io.StringIO()
                with redirect_stdout(out):status=vpw_cli.main([str(path),*args])
                return status,json.loads(out.getvalue())
            self.assertEqual(call('validate',d['id'])[0],0)
            edits=Path(folder)/'edits.json'
            with patch('vpw_cli.read_json',return_value=[{'op':'set-node','id':d['id'],'node':'softmax','values':{'dim':0}}]):
                status,result=call('apply',str(edits))
            self.assertEqual(status,0,result);self.assertIn('nn.Softmax(dim=1)',call('export',d['id'])[1]['value']['code'])
            before=path.read_bytes()
            with patch('vpw_cli.read_json',return_value=[{'op':'set-node','id':d['id'],'node':'softmax','values':{'dim':2}}]):
                self.assertEqual(call('apply',str(edits))[0],1)
            self.assertEqual(path.read_bytes(),before)

    def test_real_probability_regression_fit_checkpoint_and_export_preserve_softmax(self):
        d=model([3],linear=True);d['graph']['nodes'][0]['shape']=[1]
        xs=[[i/10.-2] for i in range(40)];ys=[]
        for (x,) in xs:
            z=[1.2*x+.3,-.7*x-.2,.1*x+.05];exp=[math.exp(v-max(z)) for v in z];total=math.fsum(exp);ys.append([v/total for v in exp])
        d['dataset']={'x':xs,'y':ys};d['training'].update(epochs=300,lr=.04,seed=7)
        models={d['id']:d};result=train_model(models,d['id'],constrained=False)
        self.assertLess(result['mse'],.001)
        saved=restore_predictor(models,d['id'],result,constrained=False)
        x=torch.tensor([[-1.7],[.4],[1.8]]);y=saved(x)
        torch.testing.assert_close(y.sum(1),torch.ones(3),atol=1e-6,rtol=1e-6)
        namespace={};exec(generate_predictor(models,d['id'],result),namespace)
        self.assertTrue(torch.equal(y,namespace['model'](x)))
        changed=copy.deepcopy(models);changed[d['id']]['graph']['nodes'][2]['dim']=0
        # For rank1, another equivalent spelling is still a different declared spec.
        with self.assertRaises(GraphError):check_checkpoint(normalize_models(changed,d['id']),d['id'],result)


if __name__=='__main__':unittest.main()
