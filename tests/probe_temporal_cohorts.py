"""Manual shared-cell, independently declared time-domain gradient probe.

From the repository root: PYTHONPATH=.:tests python tests/probe_temporal_cohorts.py
No workspace/configuration/file writes. Separate from unittest discovery counts.
"""
import copy

import torch

from app.temporal_shapes import compile_temporal_model
from test_temporal_shapes import temporal_fixture
from test_recurrent_graph_execution import named, output
from test_recurrent_execution import oracle


def main():
    torch.set_num_threads(1)
    models, mid, decl = temporal_fixture(controls=False); graph = models[mid]['graph']
    source = copy.deepcopy(graph['nodes'][0]); source['id'] = 'other-sequence'
    scan = copy.deepcopy(graph['nodes'][2]); scan.update(id='other-scan', share_with='scan')
    scan['sequence_inputs'][0]['source'] = 'other-sequence'
    graph['nodes'].extend([source, scan, {'id':'other-output', 'type':'Output'}])
    graph['input_order'].append('other-sequence')
    graph['edges'].extend([{'from':'other-sequence', 'to':'other-scan'}, {'from':'initial', 'to':'other-scan'},
                          {'from':'other-scan', 'from_port':'trajectory', 'to':'other-output'}])
    decl['other-sequence'] = [0]
    model = compile_temporal_model(models, mid, decl, constrained=False)
    owners = [id(p) for p in model.parameters()]; assert len(owners) == 2
    for left, right in ((3, 7), (9, 1)):
        torch.manual_seed(73)
        a = torch.randn(3, left, 2, requires_grad=True); b = torch.randn(3, right, 2, requires_grad=True)
        h = torch.randn(3, 2, requires_grad=True)
        aa = a.detach().clone().requires_grad_(); bb = b.detach().clone().requires_grad_(); hh = h.detach().clone().requires_grad_()
        p = list(model.parameters()); pp = [v.detach().clone().requires_grad_() for v in p]
        ya, ha = oracle(aa, hh, *pp); yb, _ = oracle(bb, hh, *pp)
        result = model(named(model.contract, {'sequence':a, 'other-sequence':b, 'initial':h}))
        x = output(model.contract, result, 'prediction'); y = output(model.contract, result, 'other-output')
        end = output(model.contract, result, 'final')
        torch.testing.assert_close(x, ya); torch.testing.assert_close(y, yb); torch.testing.assert_close(end, ha)
        ga = torch.autograd.grad(x.square().sum()+y.square().sum()+end.sum(), [a, b, h, *p])
        gb = torch.autograd.grad(ya.square().sum()+yb.square().sum()+ha.sum(), [aa, bb, hh, *pp])
        for first, second in zip(ga, gb): torch.testing.assert_close(first, second, rtol=4e-6, atol=3e-6)
        assert [id(v) for v in model.parameters()] == owners
        print('PASS: independently declared T=', (left, right), 'share one cell; all data/parameter gradients match row oracle')


if __name__ == '__main__':
    main()
