"""Gaussian multiport declaration and real numerical adapter boundary."""
import copy
import math
import unittest
from app.core import GraphError
from app import gaussian_nodes as g


def node():
    return {'id':'filter','type':'GaussianFilter','inputs':{r:r for r in g.ROLES},
        'initial_representation':'probabilities','transition_representation':'probabilities',
        'transition_mode':'sequence','mean_mode':'sequence','variance_mode':'shared'}


def incoming():
    return {'observations':([3,1],None),'initial':([2],None),
        'transition':([3,2,2],None),'means':([3,2,1],None),
        'variances':([2,1],None),'lengths':([1],None)}


class GaussianNodeTests(unittest.TestCase):
    def test_closed_roles_representations_and_shape_contract(self):
        n=node();before=copy.deepcopy(n);p=g.describe(n,incoming())
        self.assertEqual(n,before)
        self.assertEqual(p['ports'],{'filtered':[3,2],'final':[2],'step_evidence':[3],'sequence_evidence':[1]})
        self.assertEqual(p['port_axes']['filtered'],['time','state'])
        self.assertEqual(p['representations']['sequence_evidence'],'log-evidence')
        for changes in ({'trainable':False},{'share_with':'other'},{'op_version':True},
                        {'initial_representation':'logits'},{'variance_mode':'log-variance'}):
            with self.assertRaises(GraphError):g.normalized(dict(n,**changes))

    def test_wrong_shape_or_transition_axis_is_refused(self):
        for role in g.ROLES:
            shapes=incoming();shapes[role]=([9],None)
            with self.subTest(role=role),self.assertRaises(GraphError):g.describe(node(),shapes)
        shapes=incoming();shapes['transition']=([3,2,2],['time','to_state','from_state'])
        with self.assertRaisesRegex(GraphError,'from_state'):g.describe(node(),shapes)

    def test_adapter_matches_native_filter_and_has_real_gradients(self):
        import torch
        from app.gaussian_filter import filter_diagonal_gaussian_hmm
        p=g.describe(node(),incoming())
        observed=torch.tensor([[[.1],[.3],[-.2]],[[.2],[0.],[.5]]])
        initial=torch.tensor([[.8,.2],[.3,.7]])
        transition=torch.tensor([[[[.9,.1],[.2,.8]]]*3]*2,requires_grad=True)
        means=torch.tensor([[[[-.2],[.8]]]*3]*2,requires_grad=True)
        variance=torch.full((2,2,1),.4,requires_grad=True)
        lengths=torch.tensor([[3.],[2.]])
        result=g.runtime_class(torch.nn)(p)(observed,initial,transition,means,variance,lengths)
        ref=filter_diagonal_gaussian_hmm(observed,log_initial=initial.log(),log_transition=transition.log(),
            means=means,positive_variances=variance[:,None].expand(2,3,2,1),lengths=lengths[:,0].to(torch.int64))
        self.assertTrue(torch.equal(result['filtered'],ref.filtered_log_probs))
        self.assertTrue(torch.equal(result['sequence_evidence'][:,0],ref.sequence_log_likelihood))
        result['sequence_evidence'].sum().backward()
        for value in (transition,means,variance):self.assertTrue(torch.isfinite(value.grad).all())

    def test_probability_zeros_have_safe_zero_derivatives(self):
        import torch
        probability=torch.tensor([[1.,0.]],requires_grad=True)
        result=g._log_probabilities(probability)
        self.assertTrue(torch.isneginf(result[0,1]));result[0,0].backward()
        self.assertTrue(torch.equal(probability.grad,torch.tensor([[1.,0.]])))

    def test_wrong_batch_or_missing_batch_does_not_broadcast(self):
        import torch
        p=g.describe(node(),incoming())
        parts=[torch.zeros(2,3,1),torch.full((2,2),.5),torch.full((2,3,2,2),.5),
               torch.zeros(2,3,2,1),torch.ones(2,2,1),torch.full((2,1),3.)]
        for index,value in ((2,torch.full((2,2),.5)),(4,torch.ones(1,2,1))):
            bad=list(parts);bad[index]=value
            with self.assertRaisesRegex(ValueError,'exact shape'):g.forward(bad,p)

    def test_budget_precedes_probability_conversion(self):
        import torch
        from unittest.mock import patch
        n=node();shapes={'observations':([4096,1],None),'initial':([4],None),
            'transition':([4096,4,4],None),'means':([4096,4,1],None),
            'variances':([4,1],None),'lengths':([1],None)}
        p=g.describe(n,shapes)
        parts=[torch.zeros(1,4096,1),torch.full((1,4),.25),torch.full((1,4096,4,4),.25),
               torch.zeros(1,4096,4,1),torch.ones(1,4,1),torch.full((1,1),4096.)]
        with patch.object(g,'_log_probabilities',side_effect=AssertionError('allocated before admission')):
            with self.assertRaisesRegex(ValueError,'before adapter'):g.forward(parts,p)


if __name__=='__main__':unittest.main()
