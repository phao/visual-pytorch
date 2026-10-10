import copy
import unittest
import torch
from app.core import (GraphError, default_model, validate_model, compile_model,
                      generate_code, generate_dataset, train_model, check_document)

torch.set_num_threads(1)


def branch_model(router='Splitter'):
    d=default_model(inputs=4,outputs=1)
    n=d['graph']['nodes']; n[1]['features']=6
    n[2:2]=[
        {'id':'router','type':router,**({'parts':[{'output':'left','features':2},{'output':'right','features':4}]} if router=='Splitter' else {'outputs':['left','right']})},
        {'id':'left','type':'Linear','features':3,'bias':True},
        {'id':'right','type':'Linear','features':2,'bias':True},
        {'id':'join','type':'Concat','sources':['left','right'],'flatten':False},
        {'id':'readout','type':'Linear','features':1,'bias':True}]
    d['graph']['edges']=[{'from':'input','to':'linear1'},{'from':'linear1','to':'router'},
        {'from':'router','to':'left','from_port':'left'},{'from':'router','to':'right','from_port':'right'},
        {'from':'left','to':'join'},{'from':'right','to':'join'},{'from':'join','to':'readout'},
        {'from':'readout','to':'output'}]
    return d


def exported(models,mid,native):
    scope={};exec(generate_code(models,mid),scope)
    result=scope['model'];result.load_state_dict(native.state_dict());return result


class DAGTests(unittest.TestCase):
    def test_user_tanh_relu_branches_join_into_sigmoid_slp(self):
        d=default_model();d['graph']={'nodes':[
            {'id':'x1','type':'Input','features':2,'label':'First'},
            {'id':'x2','type':'Input','features':1,'label':'Second'},
            {'id':'l1','type':'Linear','features':1},
            {'id':'l2','type':'Linear','features':1},
            {'id':'a1','type':'Activation','activation':'Tanh'},
            {'id':'a2','type':'Activation','activation':'ReLU'},
            {'id':'join','type':'Concat','sources':['a1','a2']},
            {'id':'l3','type':'Linear','features':2},
            {'id':'a3','type':'Activation','activation':'Sigmoid'},
            {'id':'y','type':'Output'}], 'edges':[
            {'from':a,'to':b} for a,b in [('x1','l1'),('l1','a1'),('x2','l2'),('l2','a2'),
                ('a1','join'),('a2','join'),('join','l3'),('l3','a3'),('a3','y')]]}
        models={d['id']:d};m=compile_model(models,d['id']);e=exported(models,d['id'],m)
        self.assertEqual(validate_model(models,d['id'])['parameters'],11)
        x1,x2=torch.randn(4,2),torch.randn(4,1)
        actual=m({'First':x1,'Second':x2})
        layers=[layer for layer in m.children() if isinstance(layer,torch.nn.Linear)]
        expected=torch.sigmoid(layers[2](torch.cat([torch.tanh(layers[0](x1)),torch.relu(layers[1](x2))],dim=1)))
        torch.testing.assert_close(actual,expected)
        torch.testing.assert_close(actual,e(torch.cat([x1,x2],dim=1)))
        self.assertTrue(bool(((actual>0)&(actual<1)).all()))

    def test_internal_split_branches_rejoin_and_export_gradient_match(self):
        for router in ('Splitter','Replicator'):
            d=branch_model(router); models={d['id']:d};check_document(d)
            c=validate_model(models,d['id']);m=compile_model(models,d['id']);e=exported(models,d['id'],m)
            self.assertEqual(c['outputs'],1);self.assertTrue(c['affine']);self.assertFalse(c['matrix_affine'])
            x=torch.randn(5,4,requires_grad=True)
            actual=m(x);torch.testing.assert_close(actual,e(x))
            a,b=(2,4) if router=='Splitter' else (6,6)
            self.assertEqual(c['parameters'],4*6+6+a*3+3+b*2+2+5*1+1)
            actual.sum().backward();self.assertIsNotNone(x.grad)
            self.assertTrue(all(p.grad is not None for p in m.parameters()))

    def test_many_independent_inputs_join_only_after_separate_paths(self):
        d=default_model();nodes=[];edges=[]
        for i in range(12):
            nodes += [{'id':f'i{i}','type':'Input','features':2,'label':f'port{i}'},
                      {'id':f'l{i}','type':'Linear','features':1,'bias':True}]
            edges += [{'from':f'i{i}','to':f'l{i}'},{'from':f'l{i}','to':'join'}]
        nodes += [{'id':'join','type':'Concat','sources':[f'l{i}' for i in range(12)]},
                  {'id':'output','type':'Output'}];edges += [{'from':'join','to':'output'}]
        d['graph']={'nodes':nodes,'edges':edges};models={d['id']:d};check_document(d)
        c=validate_model(models,d['id']);m=compile_model(models,d['id'])
        self.assertEqual(c['inputs'],24);self.assertEqual(c['outputs'],12)
        parts={f'port{i}':torch.randn(3,2) for i in range(12)}
        packed=torch.cat(list(parts.values()),1)
        torch.testing.assert_close(m(parts),m(packed));torch.testing.assert_close(m(parts),exported(models,d['id'],m)(parts))

    def test_multidimensional_linear_preserves_prefix_and_reshape_export(self):
        d=default_model(outputs=4);d['graph']['nodes'][0]['shape']=[2,3]
        del d['graph']['nodes'][0]['features']
        d['graph']['nodes'].insert(2,{'id':'reshape','type':'Reshape','shape':[4,-1]})
        d['graph']['nodes'][-1]['shape']=[4,2]
        d['graph']['edges']=[{'from':'input','to':'linear1'},{'from':'linear1','to':'reshape'},{'from':'reshape','to':'output'}]
        models={d['id']:d};check_document(d);c=validate_model(models,d['id'])
        self.assertEqual(c['parameters'],16);self.assertEqual(c['outputs'],8)
        self.assertEqual(c['shapes']['linear1']['output_shape'],[2,4])
        m=compile_model(models,d['id']);e=exported(models,d['id'],m)
        x=torch.randn(5,2,3,requires_grad=True)
        self.assertEqual(tuple(m(x).shape),(5,4,2));torch.testing.assert_close(m(x),e(x))
        m(x).sum().backward();self.assertEqual(tuple(x.grad.shape),(5,2,3))
        with self.assertRaises(GraphError):m(x.reshape(5,6))

    def test_shape_errors_and_batch_axis_never_inferred(self):
        for target in ([3,-1],[-1,-1],[0,2],[True,2],[-1.0],[]):
            d=default_model(outputs=4);d['graph']['nodes'].insert(2,{'id':'r','type':'Reshape','shape':target})
            d['graph']['edges']=[{'from':'input','to':'linear1'},{'from':'linear1','to':'r'},{'from':'r','to':'output'}]
            with self.subTest(target=target),self.assertRaises(GraphError):
                check_document(d);validate_model({d['id']:d},d['id'])
        d=default_model();d['graph']['nodes'][-1]['shape']=[2]
        with self.assertRaisesRegex(GraphError,'Add a Reshape'):validate_model({d['id']:d},d['id'])

    def test_named_multidimensional_independent_input_parts(self):
        d=default_model(outputs=2);d['graph']['nodes'][0]['shape']=[2,2]
        d['graph']['nodes'].insert(1,{'id':'other','type':'Input','shape':[3],'label':'context'})
        d['graph']['nodes'].insert(2,{'id':'join','type':'Concat','sources':['input','other']})
        d['graph']['edges']=[{'from':'input','to':'join'},{'from':'other','to':'join'},
                            {'from':'join','to':'linear1'},{'from':'linear1','to':'output'}]
        models={d['id']:d};m=compile_model(models,d['id']);parts={'input':torch.randn(3,2,2),'context':torch.randn(3,3)}
        torch.testing.assert_close(m(parts),m(torch.cat([parts['input'].flatten(1),parts['context']],1)))
        with self.assertRaises(GraphError):m({'input':torch.randn(2,2,2),'context':torch.randn(3,3)})

    def test_multidimensional_split_outputs_last_axis(self):
        d=default_model(outputs=4);d['graph']['nodes'][0]['shape']=[2,3]
        d['graph']['nodes'].insert(2,{'id':'split','type':'Splitter','parts':[{'output':'output','features':1},{'output':'out2','features':3}]})
        d['graph']['nodes'].append({'id':'out2','type':'Output'})
        d['graph']['edges']=[{'from':'input','to':'linear1'},{'from':'linear1','to':'split'},
                            {'from':'split','to':'output'},{'from':'split','to':'out2'}]
        models={d['id']:d};c=validate_model(models,d['id']);m=compile_model(models,d['id']);x=torch.randn(5,2,3)
        out=m(x);self.assertEqual(tuple(out['output'].shape),(5,2,1));self.assertEqual(tuple(out['out2'].shape),(5,2,3))
        torch.testing.assert_close(out,exported(models,d['id'],m)(x))
        torch.testing.assert_close(torch.cat(list(out.values()),-1),m.forward_tensor(x))
        self.assertEqual(c['outputs'],8)

    def test_direct_tensor_fanout_and_independent_outputs_are_trainable(self):
        d=branch_model('Replicator')
        d['graph']['nodes']=[n for n in d['graph']['nodes'] if n['id'] not in {'router','join','readout'}]
        d['graph']['nodes'].append({'id':'out2','type':'Output'})
        d['graph']['edges']=[{'from':'input','to':'linear1'},{'from':'linear1','to':'left'},
            {'from':'linear1','to':'right'},{'from':'left','to':'output'},{'from':'right','to':'out2'}]
        models={d['id']:d};c=validate_model(models,d['id']);self.assertEqual(c['outputs'],5)
        m=compile_model(models,d['id']);x=torch.randn(3,4)
        torch.testing.assert_close(torch.cat(list(m(x).values()),1),m.forward_tensor(x))
        d['dataset']=generate_dataset(models,d['id'],32);d['training']['epochs']=2
        result=train_model(models,d['id']);self.assertTrue(result['mse']>=0);self.assertNotIn('A',result)

    def test_multidimensional_training_flattens_loss_not_forward(self):
        d=default_model(outputs=2);d['graph']['nodes'][0]['shape']=[2,3]
        models={d['id']:d};d['dataset']=generate_dataset(models,d['id'],32)
        d['training']['epochs']=2
        result=train_model(models,d['id']);self.assertEqual(result['output_shape'],[2,2])
        self.assertEqual(len(result['preview'][0]['prediction']),4);self.assertNotIn('A',result)

    def test_bad_ports_cycles_and_incompatible_join_rejected(self):
        d=branch_model();models={d['id']:d};d['graph']['edges'][2]['from_port']='missing'
        with self.assertRaisesRegex(GraphError,'declared output port'):validate_model(models,d['id'])
        d=branch_model();d['graph']['edges'].append({'from':'readout','to':'left'})
        with self.assertRaises(GraphError):validate_model({d['id']:d},d['id'])
        d=branch_model();d['graph']['nodes'][0]['shape']=[2,4]
        right=next(n for n in d['graph']['nodes'] if n['id']=='right')
        right['type']='Reshape';right['shape']=[8]
        with self.assertRaisesRegex(GraphError,'matching other'):validate_model({d['id']:d},d['id'])

    def test_multidimensional_references_need_exact_shapes(self):
        child=default_model(inputs=3,outputs=2);child['graph']['nodes'][0]['shape']=[2,3]
        owner=default_model();owner['graph']['nodes'][0]['shape']=[2,3]
        owner['graph']['nodes'][1]={'id':'ref','type':'Module','ref':child['id']}
        owner['graph']['edges']=[{'from':'input','to':'ref'},{'from':'ref','to':'output'}]
        models={owner['id']:owner,child['id']:child};m=compile_model(models,owner['id']);x=torch.randn(4,2,3)
        torch.testing.assert_close(m(x),exported(models,owner['id'],m)(x))
        owner['graph']['nodes'][0]['shape']=[6]
        with self.assertRaisesRegex(GraphError,'Use Reshape'):validate_model(models,owner['id'])
