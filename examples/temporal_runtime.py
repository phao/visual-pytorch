"""Bounded variable-T graph fit; no user workspace, configuration or file writes.

Run from the repository root: python examples/temporal_runtime.py
This demonstrates the explicit backend API, not a retained family/GUI option.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from app.core import compile_model


def architectures():
    body = {'id':'cell', 'name':'Shared tanh transition', 'graph':{'nodes':[
        {'id':'x','type':'Input','features':2,'label':'frame'},
        {'id':'h','type':'Input','features':2,'label':'previous state'},
        {'id':'join','type':'Concat','sources':['x','h'],'flatten':True},
        {'id':'mix','type':'Linear','features':2,'bias':True},
        {'id':'tanh','type':'Activation','activation':'Tanh'},
        {'id':'next','type':'Output','label':'next state'}],
        'edges':[{'from':a,'to':b} for a,b in [('x','join'),('h','join'),('join','mix'),('mix','tanh'),('tanh','next')]],
        'input_order':['x','h']}}
    root = {'id':'sequence', 'name':'Explicit runtime T', 'graph':{'nodes':[
        {'id':'frames','type':'Input','shape':[4,2],'axes':['T','D0'],'label':'frames'},
        {'id':'initial','type':'Input','shape':[2],'label':'initial'},
        {'id':'scan','type':'Recurrent','ref':'cell',
         'sequence_inputs':[{'input':'x','source':'frames'}],
         'states':[{'input':'h','output':'next','port':'state','source':'initial'}],
         'sequence_outputs':[{'output':'next','port':'trajectory'}]},
        {'id':'trajectory','type':'Output','label':'trajectory'},
        {'id':'final','type':'Output','label':'final'}],
        'edges':[{'from':'frames','to':'scan'},{'from':'initial','to':'scan'},
                 {'from':'scan','from_port':'trajectory','to':'trajectory'},
                 {'from':'scan','from_port':'state','to':'final'}],
        'input_order':['frames','initial']}}
    return {'cell':body,'sequence':root}


def main():
    torch.set_num_threads(1);torch.manual_seed(7)
    model = compile_model(architectures(),'sequence',runtime_temporal={'frames':[0]})
    optimizer = torch.optim.Adam(model.parameters(),lr=.04)
    owners = [id(p) for p in model.parameters()]
    weight = torch.tensor([[.35,-.2,.25,.1],[-.1,.4,.05,.2]])
    bias = torch.tensor([.12,-.08])
    cases = []
    for steps in (4,7):
        frames = torch.randn(8,steps,2);initial = torch.randn(8,2)*.2
        state = initial; target=[]
        for t in range(steps):
            state = torch.tanh(torch.nn.functional.linear(torch.cat((frames[:,t],state),dim=1),weight,bias))
            target.append(state)
        cases.append(({'frames':frames,'initial':initial},torch.stack(target,dim=1),state))
    def objective(case):
        supplied,trajectory,final=case;result=model(supplied)
        return (result['trajectory']-trajectory).square().mean()+.2*(result['final']-final).square().mean()
    initial_loss=float(sum(objective(c).detach() for c in cases)/len(cases))
    for update in range(60):
        optimizer.zero_grad();objective(cases[update%len(cases)]).backward();optimizer.step()
        assert [id(p) for p in model.parameters()]==owners
    final_loss=float(sum(objective(c).detach() for c in cases)/len(cases))
    assert final_loss<initial_loss*.1, (initial_loss,final_loss)
    print(f'PASS: one 10-parameter owner, T=4/7, 60 updates, loss {initial_loss:.6g} -> {final_loss:.6g}')
    model.eval()
    view=model.preview_shapes({'frames':[11,2],'initial':[2]},batch_size=3,gradients=False)
    print('PASS: pure actual-case preview T=11; returned shapes',view['contract']['output_shape'])


if __name__=='__main__':
    main()
