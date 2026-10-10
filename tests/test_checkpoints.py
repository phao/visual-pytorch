"""Checkpoint admission and independent inference, including real SQLite/CLI."""
import copy
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F
from app.core import (GraphError, Workspace, default_model, default_mlp,compile_model,
                      train_model, restore_predictor, generate_predictor)
import vpw_cli

torch.set_num_threads(1)


class CheckpointTests(unittest.TestCase):
    def fixture(self, *, bias=True):
        doc=default_model(inputs=2, outputs=2)
        doc['graph']['nodes'][1]['bias']=bias
        doc['training']['epochs']=3
        doc['dataset']={'x':[[i/4., (i-2)/3.] for i in range(12)],
                        'y':[[i/5., (i+1)/6.] for i in range(12)]}
        models={doc['id']:doc}
        return models,doc['id'],train_model(models,doc['id'])

    def reject_both(self, models, mid, result, message='Checkpoint'):
        for use in (restore_predictor, generate_predictor):
            with self.subTest(api=use.__name__), self.assertRaisesRegex(GraphError,message):
                use(models,mid,result)

    def test_linear_bias_on_off_restore_export_match_independent_tensor_algebra(self):
        for bias in (True,False):
            with self.subTest(bias=bias):
                models,mid,result=self.fixture(bias=bias)
                original=copy.deepcopy((models,result))
                x=torch.tensor([[.25,-.75], [2.,3.], [0.,0.]])
                a=torch.tensor(result['weights']['layer_0.weight'])
                b=torch.tensor(result['weights']['layer_0.bias']) if bias else None
                expected=x@a.T+(b if bias else 0)
                model=restore_predictor(models,mid,result)
                self.assertFalse(model.training)
                torch.testing.assert_close(model(x),expected,rtol=0,atol=0)
                scope={};exec(compile(generate_predictor(models,mid,result),'<trained-export>','exec'),scope)
                self.assertFalse(scope['model'].training)
                torch.testing.assert_close(scope['model'](x),expected,rtol=0,atol=0)
                self.assertEqual((models,result),original)

    def test_mlp_matches_manual_forward_from_checkpoint_not_shared_compiler(self):
        doc=default_mlp(inputs=2,outputs=1,hidden=[3],activation='Tanh')
        doc['training']['epochs']=3
        doc['dataset']={'x':[[i/6.,(i-1)/7.] for i in range(12)],'y':[[i/9.] for i in range(12)]}
        models={doc['id']:doc};result=train_model(models,doc['id']);w=result['weights']
        x=torch.tensor([[.5,-.3],[1.,2.]])
        expected=F.linear(torch.tanh(F.linear(x,torch.tensor(w['layer_0.weight']),
                                             torch.tensor(w['layer_0.bias']))),
                          torch.tensor(w['layer_2.weight']),torch.tensor(w['layer_2.bias']))
        restored=restore_predictor(models,doc['id'],result)
        torch.testing.assert_close(restored(x),expected,rtol=0,atol=0)
        scope={};exec(generate_predictor(models,doc['id'],result),scope)
        torch.testing.assert_close(scope['model'](x),expected,rtol=0,atol=0)

    def test_invalid_weight_numbers_shapes_and_keys_rejected_before_export(self):
        models,mid,result=self.fixture()
        bad=[]
        for value in (float('nan'),float('inf'),1e100,True,'1.0',None):
            r=copy.deepcopy(result);r['weights']['layer_0.weight'][0][0]=value;bad.append(r)
        for value in ([],[[1.]],[[1.,2.],[3.]],[[[1.,2.]],[[3.,4.]]],{},None):
            r=copy.deepcopy(result);r['weights']['layer_0.weight']=value;bad.append(r)
        r=copy.deepcopy(result);del r['weights']['layer_0.bias'];bad.append(r)
        r=copy.deepcopy(result);r['weights']['extra.weight']=[[1.]];bad.append(r)
        r=copy.deepcopy(result);r['weights']=None;bad.append(r)
        for i,r in enumerate(bad):
            with self.subTest(case=i):self.reject_both(models,mid,r)

    def test_version_preprocessing_and_stable_parameter_identity_are_required(self):
        models,mid,result=self.fixture()
        cases=[]
        for value in (None,True,0,2,'1'):
            r=copy.deepcopy(result);r['checkpoint_version']=value;cases.append(r)
        for field in ('parameter_ids','preprocessing'):
            r=copy.deepcopy(result);del r[field];cases.append(r)
        r=copy.deepcopy(result);r['parameter_ids']['layer_0.weight']='wrong/stable/id';cases.append(r)
        r=copy.deepcopy(result);r['preprocessing']['kind']='standardize';cases.append(r)
        r=copy.deepcopy(result);r['preprocessing']['binding']=None;cases.append(r)
        for i,r in enumerate(cases):
            with self.subTest(case=i):self.reject_both(models,mid,r)

    def test_identity_checks_graph_data_training_and_binding_changes(self):
        models,mid,result=self.fixture()
        for field in ('architecture','dataset','binding','training'):
            with self.subTest(field=field):
                r=copy.deepcopy(result);r['identity'][field]='0'*64
                self.reject_both(models,mid,r,'identity')
        r=copy.deepcopy(result);r['identity']['version']=True
        self.reject_both(models,mid,r,'identity')

    def test_sqlite_save_reload_and_real_cli_export_reproduce_trained_inference(self):
        models,mid,result=self.fixture()
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'trained.vtw'
            workspace=Workspace()
            try:
                workspace.put(models[mid]);doc=workspace.models()[mid]
                doc['result']=result;workspace.put(doc)
                workspace.record_run(mid,result,workspace.models());workspace.save(path)
            finally:workspace.db.close()
            loaded=Workspace.load(path)
            try:
                retained=loaded.models()[mid]['result'];self.assertEqual(retained,result)
                self.assertEqual(loaded.db.execute('SELECT COUNT(*) FROM runs').fetchone()[0],1)
                stream=io.StringIO()
                with redirect_stdout(stream):status=vpw_cli.main([str(path),'export-trained',mid])
                output=json.loads(stream.getvalue());self.assertEqual(status,0,output)
                scope={};exec(output['value']['code'],scope)
                x=torch.tensor([[.3,-.7]])
                expected=x@torch.tensor(result['weights']['layer_0.weight']).T+torch.tensor(result['weights']['layer_0.bias'])
                torch.testing.assert_close(scope['model'](x),expected,rtol=0,atol=0)
                torch.testing.assert_close(restore_predictor(loaded.models(),mid,retained)(x),expected,rtol=0,atol=0)
            finally:loaded.db.close()

    def test_zero_update_cancel_checkpoint_has_exact_metrics_and_is_restorable(self):
        models,mid,_=self.fixture()
        class Cancel:
            def is_set(self):return True
        result=train_model(models,mid,cancel=Cancel())
        self.assertEqual(result['completed_updates'],0)
        self.assertEqual(result['epochs'],0)
        self.assertEqual(result['examples_processed'],0)
        self.assertEqual(result['stop_reason'],'cancelled')
        model=restore_predictor(models,mid,result)
        indices=result['split_indices']['held_out']
        x=torch.tensor([models[mid]['dataset']['x'][i] for i in indices])
        y=torch.tensor([models[mid]['dataset']['y'][i] for i in indices])
        self.assertAlmostEqual(F.mse_loss(model(x),y).item(),result['mse'],places=7)
        self.assertEqual(result['history'][-1]['epoch'],0)

    def test_tensor_input_linear_uses_last_axis_in_restored_and_exported_model(self):
        doc=default_model(inputs=3,outputs=2)
        doc['graph']['nodes'][0].update(shape=[2,3]);doc['graph']['nodes'][0].pop('features')
        doc['training']['epochs']=2
        doc['dataset']={'x':[[i/7.+j/5. for j in range(6)] for i in range(10)],
                        'y':[[i/8.+j/4. for j in range(4)] for i in range(10)]}
        models={doc['id']:doc};result=train_model(models,doc['id']);w=result['weights']
        x=torch.arange(12,dtype=torch.float32).reshape(2,2,3)/10.
        expected=x@torch.tensor(w['layer_0.weight']).T+torch.tensor(w['layer_0.bias'])
        self.assertEqual(result['preprocessing']['binding']['inputs'][0]['shape'],[2,3])
        torch.testing.assert_close(restore_predictor(models,doc['id'],result)(x),expected,rtol=0,atol=0)
        scope={};exec(generate_predictor(models,doc['id'],result),scope)
        torch.testing.assert_close(scope['model'](x),expected,rtol=0,atol=0)

    def test_independent_tanh_relu_paths_join_into_sigmoid_slp(self):
        doc=default_model(inputs=3,outputs=2)
        doc['graph']={'input_order':['a','b'],'nodes':[
            {'id':'a','type':'Input','features':2,'label':'measurements'},
            {'id':'b','type':'Input','features':1,'label':'context'},
            {'id':'la','type':'Linear','features':2,'bias':False},
            {'id':'lb','type':'Linear','features':2},
            {'id':'ta','type':'Activation','activation':'Tanh'},
            {'id':'rb','type':'Activation','activation':'ReLU'},
            {'id':'join','type':'Concat','sources':['ta','rb'],'flatten':True},
            {'id':'final','type':'Linear','features':2},
            {'id':'sigmoid','type':'Activation','activation':'Sigmoid'},
            {'id':'out','type':'Output'}],
            'edges':[{'from':a,'to':b} for a,b in [('a','la'),('b','lb'),('la','ta'),('lb','rb'),
                ('ta','join'),('rb','join'),('join','final'),('final','sigmoid'),('sigmoid','out')]]}
        doc['training']['epochs']=2
        doc['dataset']={'x':[[i/10.,(i-2)/8.,i/11.] for i in range(12)],
                        'y':[[i/20.,i/24.] for i in range(12)]}
        models={doc['id']:doc};result=train_model(models,doc['id'])
        def weight(nid,name):
            key=next(k for k,v in result['parameter_ids'].items() if v.endswith('/'+nid+'/'+name))
            return torch.tensor(result['weights'][key])
        x=torch.tensor([[1.,2.,3.],[-1.,.5,-2.]])
        a=torch.tanh(F.linear(x[:,:2],weight('la','weight')))
        b=torch.relu(F.linear(x[:,2:],weight('lb','weight'),weight('lb','bias')))
        expected=torch.sigmoid(F.linear(torch.cat([a,b],dim=1),weight('final','weight'),weight('final','bias')))
        restored=restore_predictor(models,doc['id'],result)
        torch.testing.assert_close(restored.forward_tensor(x),expected,rtol=0,atol=0)
        torch.testing.assert_close(restored({'context':x[:,2:],'measurements':x[:,:2]}),expected,rtol=0,atol=0)
        scope={};exec(generate_predictor(models,doc['id'],result),scope)
        torch.testing.assert_close(scope['model'](x),expected,rtol=0,atol=0)

    def test_fresh_references_keep_distinct_parameter_ids_and_exported_instances(self):
        block=default_model('Projection',inputs=2,outputs=2)
        doc=default_model(inputs=2,outputs=1)
        doc['graph']['nodes'][1:1]=[
            {'id':'m_a','type':'Module','ref':block['id']},
            {'id':'m_b','type':'Module','ref':block['id']},
            {'id':'join','type':'Concat','sources':['m_a','m_b'],'flatten':True}]
        doc['graph']['edges']=[{'from':a,'to':b} for a,b in [('input','m_a'),('input','m_b'),
            ('m_a','join'),('m_b','join'),('join','linear1'),('linear1','output')]]
        doc['training']['epochs']=2
        doc['dataset']={'x':[[i/10.,(i-3)/7.] for i in range(10)],'y':[[i/6.] for i in range(10)]}
        models={doc['id']:doc,block['id']:block};result=train_model(models,doc['id'])
        restored=restore_predictor(models,doc['id'],result)
        self.assertEqual(len(set(result['parameter_ids'].values())),len(result['weights']))
        self.assertIsNot(restored.layer_0.layer_0.weight,restored.layer_1.layer_0.weight)
        w=result['weights'];x=torch.tensor([[.3,-.1]])
        a=F.linear(x,torch.tensor(w['layer_0.layer_0.weight']),torch.tensor(w['layer_0.layer_0.bias']))
        b=F.linear(x,torch.tensor(w['layer_1.layer_0.weight']),torch.tensor(w['layer_1.layer_0.bias']))
        expected=F.linear(torch.cat([a,b],dim=1),torch.tensor(w['layer_2.weight']),torch.tensor(w['layer_2.bias']))
        scope={};exec(generate_predictor(models,doc['id'],result),scope)
        torch.testing.assert_close(scope['model'](x),expected,rtol=0,atol=0)
        torch.testing.assert_close(restored(x),expected,rtol=0,atol=0)
        changed=copy.deepcopy(models);changed[block['id']]['graph']['nodes'][1]['bias']=False
        self.reject_both(changed,doc['id'],result,'identity')

    def test_predictor_export_admission_does_not_import_torch(self):
        models,mid,result=self.fixture()
        import subprocess,sys
        request=json.dumps({'models':models,'mid':mid,'result':result})
        source=('import json,sys;from app.core import generate_predictor;'
                'r=json.load(sys.stdin);code=generate_predictor(r["models"],r["mid"],r["result"]);'
                'assert "torch" not in sys.modules;assert "model.eval()" in code;print("PASS")')
        run=subprocess.run([sys.executable,'-c',source],input=request,text=True,
                           capture_output=True,timeout=5)
        self.assertEqual(run.returncode,0,run.stderr)
        self.assertEqual(run.stdout.strip(),'PASS')

    def test_optional_analysis_error_cannot_destroy_valid_training_checkpoint(self):
        models,mid,_=self.fixture()
        with patch('app.core.affine_summary',side_effect=ValueError('private-analysis-failure')):
            result=train_model(models,mid)
        self.assertEqual(result['affine_analysis'],{'status':'error','reason':'private-analysis-failure'})
        self.assertNotIn('A',result)
        self.assertEqual(result['completed_updates'],3)
        self.assertTrue(result['weights'])
        restore_predictor(models,mid,result)
        scope={};exec(generate_predictor(models,mid,result),scope)
        self.assertFalse(scope['model'].training)

    def test_float32_checkpoint_contract_does_not_follow_global_default_dtype(self):
        old=torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float64)
            models,mid,result=self.fixture()
            model=restore_predictor(models,mid,result)
            self.assertTrue(all(p.dtype==torch.float32 for p in model.parameters()))
            scope={};exec(generate_predictor(models,mid,result),scope)
            self.assertTrue(all(p.dtype==torch.float32 for p in scope['model'].parameters()))
            x=torch.tensor([[.3,-.7]],dtype=torch.float32)
            torch.testing.assert_close(model(x),scope['model'](x),rtol=0,atol=0)
        finally:torch.set_default_dtype(old)

    def test_split_and_replicated_outputs_restore_named_views_and_packed_loss(self):
        for kind in ('Splitter','Replicator'):
            with self.subTest(kind=kind):
                doc=default_model(inputs=2,outputs=3)
                router={'id':'route','type':kind}
                if kind=='Splitter':router['parts']=[{'output':'left','features':1},{'output':'right','features':2}]
                else:router['outputs']=['left','right']
                doc['graph']['nodes'][-1:]=[router,{'id':'left','type':'Output','label':'left'},
                                                   {'id':'right','type':'Output','label':'right'}]
                doc['graph']['edges'][-1:]=[{'from':'linear1','to':'route'},
                                           {'from':'route','from_port':'left','to':'left'},
                                           {'from':'route','from_port':'right','to':'right'}]
                doc['training']['epochs']=2
                doc['dataset']={'x':[[i/7.,i/9.] for i in range(10)],
                                'y':[[i/6.,i/10.,i/8.] for i in range(10)]}
                models={doc['id']:doc};result=train_model(models,doc['id']);w=result['weights']
                model=restore_predictor(models,doc['id'],result);scope={}
                exec(generate_predictor(models,doc['id'],result),scope)
                x=torch.tensor([[.4,-.7],[.8,.1]])
                expected=F.linear(x,torch.tensor(w['layer_0.weight']),torch.tensor(w['layer_0.bias']))
                for predictor in (model,scope['model']):
                    actual=predictor(x)
                    torch.testing.assert_close(predictor.forward_tensor(x),expected,rtol=0,atol=0)
                    if kind=='Splitter':
                        torch.testing.assert_close(actual['left'],expected[:,:1],rtol=0,atol=0)
                        torch.testing.assert_close(actual['right'],expected[:,1:],rtol=0,atol=0)
                    else:
                        self.assertIs(actual['left'],actual['right'])
                        torch.testing.assert_close(actual['left'],expected,rtol=0,atol=0)
                indices=result['split_indices']['held_out']
                testx=torch.tensor([doc['dataset']['x'][i] for i in indices])
                testy=torch.tensor([doc['dataset']['y'][i] for i in indices])
                self.assertAlmostEqual(F.mse_loss(model.forward_tensor(testx),testy).item(),result['mse'],places=7)

    def test_native_api_is_graph_module_not_fake_sequential_slicing(self):
        models,mid,_=self.fixture()
        model=compile_model(models,mid)
        self.assertIsInstance(model,torch.nn.Module)
        self.assertNotIsInstance(model,torch.nn.Sequential)
        with self.assertRaises(TypeError):model[0]
        x=torch.tensor([[.4,-.2]])
        values=model.intermediate_values(x)
        self.assertEqual(set(values),{'input','linear1','output'})
        torch.testing.assert_close(values['output'],model.forward_tensor(x),rtol=0,atol=0)


if __name__=='__main__':unittest.main()
