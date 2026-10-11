"""Independent-T shared LSTM branches: native/export values and all gradients.

Manual bounded probe, separate from unittest discovery; no workspace mutation.
"""
import copy
import json
import torch

from app.core import validate_model
from app.temporal_shapes_v2 import compile_temporal_model
from app.temporal_export_v2 import temporal_source
from test_temporal_shapes_v2 import runtime_fixture
from test_lstm_execution import oracle, named

torch.set_num_threads(1)


def main():
    models,mid,decl = runtime_fixture(hidden=3,projection=2,layers=1,width=2)
    graph = models[mid]['graph']; second = copy.deepcopy(graph['nodes'][3])
    second.update(id='shared',share_with='lstm'); second['inputs']['sequence'] = 'sequence2'
    second['lengths']['source'] = 'lengths2'
    graph['nodes'] += [{'id':'sequence2','type':'Input','shape':[5,2],'axes':['T','F'],'label':'Second frames'},
                       {'id':'lengths2','type':'Input','shape':[1],'label':'Second lengths'},second]
    graph['input_order'] += ['sequence2','lengths2']; decl['sequence2'] = [0]
    graph['edges'] += [{'from':source,'to':'shared'} for source in ('sequence2','hidden','cell','lengths2')]
    for port in ('sequence','hidden','cell'):
        graph['nodes'].append({'id':'second-'+port,'type':'Output','label':'Second '+port})
        graph['edges'].append({'from':'shared','from_port':port,'to':'second-'+port})
    native = compile_temporal_model(models,mid,decl,constrained=False)
    scope = {}; exec(temporal_source(models,mid,decl,constrained=False)['code'],scope)
    exported = scope['build_model'](); exported.load_state_dict(native.state_dict(),strict=True)
    baseline = validate_model(models,mid); tested = 0
    for model in (native,exported):
        owners = [id(p) for p in model.parameters()]
        assert sum(p.numel() for p in model.parameters()) == baseline['parameters']
        for t1,t2 in ((1,9),(7,2),(3,3)):
            shapes = {g['id']:list(g['shape']) for g in baseline['input_groups']}
            shapes['sequence'][0] = t1; shapes['sequence2'][0] = t2
            c = model.preview_shapes(shapes,batch_size=3)['contract']
            torch.manual_seed(42)
            values = {g['id']:torch.randn(3,*g['shape']) for g in c['input_groups']}
            values['lengths'] = torch.tensor([[0],[t1],[max(0,t1-1)]],dtype=torch.float32)
            values['lengths2'] = torch.tensor([[t2],[0],[max(0,t2-1)]],dtype=torch.float32)
            for sequence,lengths in (('sequence','lengths'),('sequence2','lengths2')):
                for row in range(3):
                    values[sequence][row,int(values[lengths][row,0]):] = float('nan')
            keys = ('sequence','sequence2','hidden','cell')
            for key in keys:
                values[key].requires_grad_()
            independent = {key:values[key].detach().clone().requires_grad_() for key in keys}
            weights = {key:p.detach().clone().requires_grad_() for key,p in model.layer_0.named_parameters()}
            expected = {}
            for sequence,lengths,prefix in (('sequence','lengths','Returned '),('sequence2','lengths2','Second ')):
                result = oracle(independent[sequence],independent['hidden'],independent['cell'],weights,
                                lengths=values[lengths][:,0].long(),layers=1,bidirectional=True,projection=2)
                expected.update({prefix+key:value for key,value in result.items()})
            actual = model(named(c,values))
            for key in expected:
                torch.testing.assert_close(actual[key],expected[key],rtol=5e-6,atol=3e-6)
            loss = lambda results:sum((i+1)*results[key].square().sum() for i,key in enumerate(sorted(expected)))
            aa = torch.autograd.grad(loss(actual),[*(values[key] for key in keys),*model.parameters()])
            bb = torch.autograd.grad(loss(expected),[*(independent[key] for key in keys),*weights.values()])
            for a,b in zip(aa,bb):
                torch.testing.assert_close(a,b,rtol=2e-5,atol=6e-6)
            assert [id(p) for p in model.parameters()] == owners
            assert model.contract == baseline
            tested += 1
    print(json.dumps({'ok':True,'routes':['native','captured-export'],'time_pairs':[[1,9],[7,2],[3,3]],
                      'cases':tested,'parameters':baseline['parameters'],'all_input_state_owner_gradients':True}))


if __name__ == '__main__':
    main()
