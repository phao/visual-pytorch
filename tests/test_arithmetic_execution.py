"""Independent arithmetic values/gradients and real worker/retained export."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
import torch
from app.core import compile_model, generate_code, GraphError
from app.family_export import family_source
from app.revisions import model_document, frozen_document
from app.datasets import make_split
from app.task_execution import task_bundle, training_protocol, restore_task_predictor, generate_task_predictor
from app.task_training import train_task
from app.transport import run_json_worker
from test_arithmetic import model, family
from arithmetic_runtime_fixture import repeated_family,retained_records,make_bundle
import sys

torch.set_num_threads(1)


def arithmetic_bundle(updates=12):
    m=model(learned=True);rows=[]
    for i in range(16):
        x=[(i-8)/5.,((i*7)%13-6)/4.];other=[((i*3)%11-5)/6.,(i%5-2)/3.]
        y=[.7*x[0]-.2*x[1]+.3+2*other[0],.4*x[0]+1.1*x[1]-.5+2*other[1]]
        rows.append({'id':'r'+str(i),'values':{'x':x,'other':other,'y':y}})
    d={'version':1,'id':'data','name':'Arithmetic target','storage':{'kind':'inline-records','version':1},'fields':[{'id':k,'dtype':'float32','shape':[2]} for k in ('x','other','y')],'records':rows}
    t={'version':1,'id':'task','name':'Arithmetic MSE','inputs':[{'port':k,'field':k,'transforms':[]} for k in ('x','other')],
       'heads':[{'id':'target','port':'y','field':'y','loss':'mse','weight':1.,'transforms':[]}]}
    split=make_split(d,{'method':'row','seed':13,'fractions':[.75,.125,.125]})
    return task_bundle(model_document({m['id']:m},m['id']),d,t,split,training_protocol(epochs=None,updates=updates,batch_size=4,lr=.03,evaluate_every=3))


class ArithmeticExecutionTests(unittest.TestCase):
    def test_runtime_family9_repeated_arithmetic_values_all_gradients_and_actual_task_v4_workers(self):
        for mode in ('independent','tied'):
            for learned in (False,True):
                f=repeated_family(mode,learned);saved,fm,runtime=retained_records(f)
                declaration=runtime['specialization']['runtime']['inputs']
                native=compile_model(fm['models'],fm['root'],constrained=False,runtime_temporal_v2=declaration)
                if learned:
                    layer=next(p['layer'] for p in native.contract['plan'] if p['id']=='head')
                    with torch.no_grad():getattr(native,layer).weight.copy_(torch.eye(2))
                ns={};exec(family_source(f,constrained=False)['code'],ns);exported=ns['ArchitectureFamily']().network
                exported.load_state_dict(native.state_dict(),strict=True)
                self.assertEqual(next(n for n in fm['models'][fm['root']]['graph']['nodes'] if n['id']=='sum')['sources'],['repeat/2/m','other'])
                self.assertEqual(native.runtime_specification()['output_dependencies'],[['other.T','x.T'],[]])
                with self.assertRaises(GraphError):native.preview_shapes({'x':[5,2],'other':[1,2]})
                for steps in (1,5):
                    for runner in (native,exported):
                        with self.subTest(mode=mode,learned=learned,steps=steps,route=type(runner).__name__):
                            x=(torch.arange(4*steps,dtype=torch.float32).reshape(2,steps,2)/8).requires_grad_()
                            other=(torch.arange(4*steps,dtype=torch.float32).reshape(2,steps,2)/16).requires_grad_()
                            g=torch.arange(4*steps,dtype=torch.float32).reshape(2,steps,2)/32
                            out=runner.forward_tensor({'x':x,'other':other});expected=27*x.detach()+other.detach()
                            torch.testing.assert_close(out,expected,rtol=0,atol=0);out.backward(g)
                            torch.testing.assert_close(x.grad,27*g,rtol=0,atol=0);torch.testing.assert_close(other.grad,g,rtol=0,atol=0)
                            self.assertEqual(sum(p.numel() for p in runner.parameters()),4 if learned else 0)
                            if learned:torch.testing.assert_close(getattr(runner,layer).weight.grad,g.reshape(-1,2).T@expected.reshape(-1,2),rtol=0,atol=0)
                            runner.zero_grad(set_to_none=True)
                    if learned:
                        b=make_bundle(fm,runtime,steps);self.assertEqual(b['version'],4)
                        local=train_task(b,constrained=False)
                        actual=run_json_worker([sys.executable,'-u','-m','app.worker'],{'kind':'task-v4','bundle':b,'constrained':False,'memory_limit_percent':10.},cwd=str(Path(__file__).parents[1]))
                        self.assertEqual(actual['status'],'completed');self.assertEqual(actual['checkpoint'],local['checkpoint']);self.assertEqual(actual['history'],local['history'])

    def test_independent_values_and_input_parameter_gradients_native_and_exported(self):
        for shape, axes in (([2], None), ([3,2], ['T','F']), ([1,4,2], None)):
            for coefficient in (-1.5, 0., 2.):
                m=model(shape, coefficient, learned=True, axes=axes); native=compile_model({m['id']:m},m['id'],constrained=False)
                w=torch.tensor([[.7,-.2],[.4,1.1]]); b=torch.tensor([.3,-.5])
                with torch.no_grad(): native.layer_0.weight.copy_(w); native.layer_0.bias.copy_(b)
                ns={};exec(generate_code({m['id']:m},m['id']),ns);exported=ns['model'];exported.load_state_dict(native.state_dict(),strict=True)
                for runner in (native,exported):
                    with self.subTest(shape=shape, coefficient=coefficient, route=type(runner).__name__):
                        x=(torch.arange(2*torch.tensor(shape).prod().item(),dtype=torch.float32).reshape(2,*shape)/7).requires_grad_()
                        other=(x.detach().flip(0)*.3-.2).requires_grad_(); go=torch.linspace(.1,.8,x.numel()).reshape(x.shape)
                        actual=runner.forward_tensor({'x':x,'other':other});expected=x.detach()@w.T+b+coefficient*other.detach()
                        torch.testing.assert_close(actual,expected,rtol=1e-6,atol=1e-6)
                        actual.backward(go);torch.testing.assert_close(x.grad,go@w,rtol=1e-6,atol=1e-6)
                        torch.testing.assert_close(other.grad,go*coefficient,rtol=1e-6,atol=1e-6)
                        torch.testing.assert_close(runner.layer_0.weight.grad,go.reshape(-1,2).T@x.detach().reshape(-1,2),rtol=1e-6,atol=1e-6)
                        torch.testing.assert_close(runner.layer_0.bias.grad,go.reshape(-1,2).sum(0),rtol=1e-6,atol=1e-6)
                        self.assertEqual(sum(p.numel() for p in runner.parameters()),6);self.assertEqual(len(list(runner.parameters())),2)
                        runner.zero_grad(set_to_none=True)

    def test_declared_addition_order_is_actual_float32_order_not_edge_or_alphabetical_order(self):
        m={'id':'order','name':'Ordered addition','graph':{'nodes':[{'id':n,'type':'Input','shape':[1]} for n in ('a','b','c')]+[
            {'id':'sum','type':'Add','sources':['a','b','c']},{'id':'y','type':'Output'}],
            'edges':[{'from':n,'to':'sum'} for n in ('c','a','b')]+[{'from':'sum','to':'y'}]}}
        values={'a':torch.tensor([[1e20]]),'b':torch.tensor([[-1e20]]),'c':torch.tensor([[3.]])}
        self.assertEqual(compile_model({'order':m},'order',constrained=False).forward_tensor(values).item(),3.)
        m['graph']['nodes'][3]['sources']=['a','c','b']
        self.assertEqual(compile_model({'order':m},'order',constrained=False).forward_tensor(values).item(),0.)

    def test_dtype_and_broadcasting_cannot_silently_change_interpretation(self):
        m=model();runner=compile_model({m['id']:m},m['id'],constrained=False)
        with self.assertRaises(GraphError):runner.forward_tensor({'x':torch.ones(2,2,dtype=torch.float64),'other':torch.ones(2,2,dtype=torch.float64)})
        with self.assertRaises(GraphError):runner.forward_tensor({'x':torch.ones(2,2),'other':torch.ones(2,1)})

    def test_family9_standalone_and_nested_module_carry_arithmetic_without_application_import(self):
        f=family();source=family_source(f,constrained=False);ns={};exec(source['code'],ns)
        runner=ns['ArchitectureFamily']();native=compile_model(f['models'],f['root'],constrained=False)
        runner.network.load_state_dict(native.state_dict(),strict=True)
        values={'x':torch.tensor([[1.,2.]]),'other':torch.tensor([[-.5,.3]])}
        torch.testing.assert_close(runner.network.forward_tensor(values),native.forward_tensor(values),rtol=0,atol=0)
        child=model(learned=True);parent={'id':'parent','name':'Packed nested arithmetic','graph':{'nodes':[
            {'id':'x','type':'Input','shape':[4]},{'id':'module','type':'Module','ref':child['id']},{'id':'y','type':'Output'}],
            'edges':[{'from':'x','to':'module'},{'from':'module','to':'y'}]}}
        models={child['id']:child,'parent':parent};root=compile_model(models,'parent',constrained=False);ns={};exec(generate_code(models,'parent'),ns)
        ns['model'].load_state_dict(root.state_dict(),strict=True);packed=torch.tensor([[1.,2.,-.5,.3]])
        torch.testing.assert_close(ns['model'].forward_tensor(packed),root.forward_tensor(packed),rtol=0,atol=0)

    def test_real_worker_fit_exact_checkpoint_and_retained_export_match_local(self):
        bundle=arithmetic_bundle()
        local=train_task(bundle,constrained=False)
        actual=run_json_worker([sys.executable,'-u','-m','app.worker'],{'kind':'task-v1','bundle':bundle,'constrained':False,'memory_limit_percent':10.},cwd=str(Path(__file__).parents[1]))
        self.assertEqual(actual['status'],'completed');self.assertEqual(actual['counts']['completed_updates'],12)
        self.assertEqual(actual['checkpoint'],local['checkpoint']);self.assertEqual(actual['history'],local['history'])
        predict=restore_task_predictor(bundle,actual['checkpoint'],constrained=False);ns={};exec(generate_task_predictor(bundle,actual['checkpoint'],constrained=False),ns)
        values=[row['values'] for row in bundle['dataset']['records'][:3]];a=predict(values);b=ns['predict_records'](values)
        self.assertEqual(set(a),{'y'});torch.testing.assert_close(a['y'],b['y'],rtol=0,atol=0)


if __name__=='__main__':unittest.main()
