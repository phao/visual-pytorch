"""Authored and edited graphs: independent gate gradients, exports and fitting."""
import copy
import unittest
import torch
from app.core import compile_model, generate_code, operation_nodes, validate_model
from app.evidence import parameter_schema
from app.lstm_authoring import prepare_edit, build_edit, EDIT_TOKEN
from test_lstm_authoring import authored, by_id
from test_lstm_execution import oracle

torch.set_num_threads(1)


def inputs(contract, node, grad=False):
    torch.manual_seed(47)
    values={g['id']:torch.randn(4,*g['shape'],requires_grad=grad) for g in contract['input_groups']}
    if node['lengths']:
        lengths=[0,4,1,3];values[node['lengths']['source']]=torch.tensor(lengths,dtype=torch.float32).reshape(4,1)
        with torch.no_grad():
            for row,length in enumerate(lengths):values[node['inputs']['sequence']][row,length:]=float('nan')
    return values


def outputs(model, doc, node, values):
    result=model({g['name']:values[g['id']] for g in model.contract['input_groups']})
    groups={g['id']:g['name'] for g in model.contract['output_groups']}
    return {edge['from_port']:result[groups[edge['to']]] for edge in doc['graph']['edges'] if edge['from']==node['id']}


class LSTMAuthoredExecutionTests(unittest.TestCase):
    def pair(self, **options):
        models, token, choices=authored(steps=4,input_size=2,hidden_size=3,**options)
        # Exercise the actual in-place editing path, not only imported fixtures.
        info=prepare_edit(models,'parent',token['node']); choices={**info['options'],'label':'Executable edited branch'}
        models['parent'],nid=build_edit(models,{k:info[k] for k in EDIT_TOKEN},choices)
        torch.manual_seed(19);native=compile_model(models,'parent',constrained=False)
        namespace={};exec(generate_code(models,'parent'),namespace);exported=namespace['model']
        exported.load_state_dict(native.state_dict(),strict=True)
        self.assertEqual(set(native.state_dict()),set(parameter_schema(models,'parent')))
        return models,nid,native,exported

    def test_authored_edited_native_and_exported_forward_all_input_parameter_gradients(self):
        for options in ({'proj_size':0,'bias':False,'num_layers':1,'bidirectional':False,'lengths':False},
                        {'proj_size':2,'bias':True,'num_layers':2,'bidirectional':True,'lengths':True}):
            models,nid,native,exported=self.pair(**options);doc=models['parent'];node=by_id(doc,nid)
            index=next(i for i,n in enumerate(operation_nodes(doc)) if n['id']==nid)
            for model in (native,exported):
                with self.subTest(options=options,route=type(model).__name__):
                    values=inputs(model.contract,node,grad=True);layer=getattr(model,'layer_'+str(index))
                    own={role:values[node['inputs'][role]].detach().clone().requires_grad_() for role in ('sequence','hidden','cell')}
                    weights={key:p.detach().clone().requires_grad_() for key,p in layer.named_parameters()}
                    lengths=values[node['lengths']['source']][:,0].long() if node['lengths'] else torch.full((4,),4)
                    expected=oracle(own['sequence'],own['hidden'],own['cell'],weights,lengths=lengths,
                        layers=node['num_layers'],bidirectional=node['bidirectional'],projection=node['proj_size'])
                    actual=outputs(model,doc,node,values)
                    for role in expected:torch.testing.assert_close(actual[role],expected[role],rtol=3e-6,atol=2e-6)
                    loss=lambda result:sum((i+.5)*result[role].square().sum() for i,role in enumerate(('sequence','hidden','cell')))
                    aa=torch.autograd.grad(loss(actual),[*(values[node['inputs'][r]] for r in own),*layer.parameters()])
                    bb=torch.autograd.grad(loss(expected),[*own.values(),*weights.values()])
                    for a,b in zip(aa,bb):torch.testing.assert_close(a,b,rtol=8e-6,atol=3e-6)

    def test_authored_branch_can_fit_and_export_exact_loaded_predictions(self):
        models,nid,teacher,student=self.pair(lengths=True,proj_size=0,bias=True,num_layers=1,bidirectional=False)
        doc=models['parent'];node=by_id(doc,nid);values=inputs(student.contract,node)
        torch.manual_seed(71)
        with torch.no_grad():
            for p in student.parameters():p.uniform_(-.25,.25)
        expected={k:v.detach() for k,v in outputs(teacher,doc,node,values).items()}
        optimizer=torch.optim.Adam(student.parameters(),lr=.03,foreach=False,fused=False)
        def loss():return sum((outputs(student,doc,node,values)[k]-expected[k]).square().mean() for k in expected)
        before=loss().item()
        for _ in range(35):optimizer.zero_grad();value=loss();value.backward();optimizer.step()
        after=loss().item();self.assertLess(after,before*.1)
        restored=compile_model(models,'parent',constrained=False);restored.load_state_dict(student.state_dict(),strict=True)
        for key,value in outputs(student,doc,node,values).items():
            torch.testing.assert_close(outputs(restored,doc,node,values)[key],value,rtol=0,atol=0)

    def test_editing_bias_policy_changes_exact_schema_not_state_dimensions(self):
        models,nid,native,_=self.pair(lengths=False,proj_size=2,bias=True,num_layers=2,bidirectional=True)
        old=validate_model(models,'parent');info=prepare_edit(models,'parent',nid)
        models['parent'],_=build_edit(models,{k:info[k] for k in EDIT_TOKEN},{**info['options'],'bias':False})
        new=validate_model(models,'parent')
        self.assertEqual(old['shapes'][nid]['ports'],new['shapes'][nid]['ports'])
        self.assertEqual(old['parameters']-new['parameters'],8*3*2*2)
        schema=parameter_schema(models,'parent');self.assertFalse(any('/'+nid+'/bias_' in spec['id'] for spec in schema.values()))
        model=compile_model(models,'parent',constrained=False)
        result=outputs(model,models['parent'],by_id(models['parent'],nid),inputs(model.contract,by_id(models['parent'],nid)))
        self.assertTrue(all(bool(torch.isfinite(v).all()) for v in result.values()))


if __name__=='__main__':unittest.main()
