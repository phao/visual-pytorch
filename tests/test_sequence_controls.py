"""Independent scalar controls and exact reused-filter segment checks."""
import copy
import itertools
import math
import unittest
from unittest.mock import patch
import torch
from app.sequence_controls import filter_run,lag_run
from app.gaussian_filter import filter_diagonal_gaussian_hmm,GaussianFilterResult

torch.set_num_threads(1)


def filtered(values):
    return filter_run(**values,kernel=filter_diagonal_gaussian_hmm,result_class=GaussianFilterResult)


def args():
    return {'observations':torch.tensor([[[.1],[.4],[-.2],[.8],[.3]]],dtype=torch.float64),
        'log_initial':torch.tensor([[.8,.2]],dtype=torch.float64).log(),
        'log_transition':torch.tensor([[[[.9,.1],[.3,.7]]]*5],dtype=torch.float64).log(),
        'means':torch.tensor([[[[-.3],[.7]]]*5],dtype=torch.float64),
        'positive_variances':torch.tensor([[[[.2],[.4]]]*5],dtype=torch.float64),
        'lengths':torch.tensor([5])}


class SequenceControlsTests(unittest.TestCase):
    def test_controlled_empty_batch_and_time_preserve_declared_output_shapes(self):
        for batch,steps in ((0,5),(1,0),(0,0)):
            with self.subTest(batch=batch,steps=steps):
                values=args()
                for key in ('observations','log_transition','means','positive_variances'):
                    values[key]=values[key][:batch,:steps].detach().requires_grad_()
                values['log_initial']=values['log_initial'][:batch].detach().requires_grad_()
                values['lengths']=torch.zeros(batch,dtype=torch.int64)
                values['execution_mask']=torch.zeros(batch,steps,dtype=torch.bool)
                values['reset_mask']=torch.ones(batch,steps,dtype=torch.bool)
                result=filtered(values)
                self.assertEqual(tuple(result.filtered_log_probs.shape),(batch,steps,2))
                self.assertEqual(tuple(result.step_log_evidence.shape),(batch,steps))
                self.assertEqual(tuple(result.final_log_probs.shape),(batch,2))
                self.assertEqual(tuple(result.sequence_log_likelihood.shape),(batch,))
                result.sequence_log_likelihood.sum().backward()
                for key in ('observations','log_initial','log_transition','means','positive_variances'):
                    self.assertIsNotNone(values[key].grad,key)
                    self.assertEqual(values[key].grad.numel(),values[key].numel())
                history=torch.zeros(batch,2,1,dtype=torch.float64,requires_grad=True)
                resets=torch.zeros(batch,steps,2,1,dtype=torch.float64,requires_grad=True)
                output=lag_run(values['observations'],history,values['lengths'],lags=2,
                    execution_mask=values['execution_mask'],reset_mask=values['reset_mask'],reset_history=resets)
                self.assertEqual(tuple(output.shape),(batch,steps,2))
                output.sum().backward();self.assertIsNotNone(history.grad);self.assertIsNotNone(resets.grad)

    def test_controls_absent_are_bitwise_original_kernel(self):
        values=args();a=filtered(values);b=filter_diagonal_gaussian_hmm(**values)
        for key in ('filtered_log_probs','final_log_probs','step_log_evidence','sequence_log_likelihood'):
            self.assertTrue(torch.equal(getattr(a,key),getattr(b,key)))

    def test_filter_skips_nan_slots_resets_only_on_execution_and_matches_path_oracle(self):
        values=args();values['execution_mask']=torch.tensor([[True,False,True,True,False]])
        values['reset_mask']=torch.tensor([[False,True,False,True,True]])
        values['emission_mask']=torch.tensor([[True,True,False,True,True]])
        for key in ('observations','log_transition','means','positive_variances'):
            values[key][:,1]=math.nan;values[key][:,4]=math.nan
        result=filtered(values)
        # First segment has emission at t0, a transition without emission at t2.
        # Second segment restarts before t3; resets at skipped t1/t4 do nothing.
        y=values['observations'][0,:,0].tolist();p=[[.9,.1],[.3,.7]];q=[.8,.2];mu=[-.3,.7];var=[.2,.4]
        def enumerated(times,emitted):
            weights=[];states=[]
            for path in itertools.product(range(2),repeat=len(times)+1):
                score=math.log(q[path[0]])
                for index,t in enumerate(times):
                    k=path[index+1];score+=math.log(p[path[index]][k])
                    if emitted[index]:score-=.5*(math.log(2*math.pi*var[k])+(y[t]-mu[k])**2/var[k])
                weights.append(score);states.append(path[-1])
            high=max(weights);evidence=high+math.log(sum(math.exp(w-high) for w in weights))
            belief=[math.log(sum(math.exp(w-evidence) for w,k in zip(weights,states) if k==state)) for state in range(2)]
            return evidence,belief
        first,b0=enumerated([0],[True]);same,b2=enumerated([0,2],[True,False]);last,b3=enumerated([3],[True])
        self.assertAlmostEqual(first,same,places=13)
        self.assertAlmostEqual(float(result.sequence_log_likelihood[0]),first+last,places=13)
        for t,expected in ((0,b0),(1,b0),(2,b2),(3,b3),(4,b3)):
            torch.testing.assert_close(result.filtered_log_probs[0,t],torch.tensor(expected,dtype=torch.float64),rtol=1e-12,atol=1e-12)
        self.assertTrue(torch.equal(result.step_log_evidence[0,[1,2,4]],torch.zeros(3,dtype=torch.float64)))

    def test_controlled_filter_gradients_zero_for_unused_values_and_segment_gradcheck(self):
        values=args();values['execution_mask']=torch.tensor([[True,False,True,True,False]])
        values['reset_mask']=torch.tensor([[False,False,False,True,False]])
        keys=('observations','log_initial','log_transition','means','positive_variances')
        for key in keys:values[key].requires_grad_()
        result=filtered(values);result.sequence_log_likelihood.sum().backward()
        for key in keys:self.assertTrue(torch.isfinite(values[key].grad).all(),key)
        for key in ('observations','log_transition','means','positive_variances'):
            self.assertEqual(float(values[key].grad[:,[1,4]].abs().sum()),0.)
        y=values['observations'].detach().requires_grad_();means=values['means'].detach().requires_grad_()
        frozen={k:v.detach() if isinstance(v,torch.Tensor) else v for k,v in values.items()}
        self.assertTrue(torch.autograd.gradcheck(lambda a,b:filtered({**frozen,'observations':a,'means':b}).sequence_log_likelihood,(y,means)))

    def test_all_skipped_filter_carries_initial_zero_evidence_and_safe_zero_gradients(self):
        values=args();values['execution_mask']=torch.zeros(1,5,dtype=torch.bool);values['reset_mask']=torch.ones(1,5,dtype=torch.bool)
        for key in ('observations','log_transition','means','positive_variances'):
            values[key].fill_(math.nan);values[key].requires_grad_()
        result=filtered(values);result.sequence_log_likelihood.sum().backward()
        torch.testing.assert_close(result.filtered_log_probs,values['log_initial'][:,None].expand(1,5,2))
        self.assertEqual(float(result.sequence_log_likelihood[0]),0.)
        for key in ('observations','log_transition','means','positive_variances'):
            self.assertTrue(torch.equal(values[key].grad,torch.zeros_like(values[key])))

    def test_lag_execution_and_reset_have_exact_oldest_to_newest_history(self):
        x=torch.tensor([[[10.],[math.nan],[30.],[40.],[math.nan]]],dtype=torch.float64,requires_grad=True)
        h=torch.tensor([[[-2.],[-1.]]],dtype=torch.float64,requires_grad=True)
        resets=torch.full((1,5,2,1),math.nan,dtype=torch.float64);resets[0,3,:,0]=torch.tensor([100.,200.])
        resets.requires_grad_();execute=torch.tensor([[True,False,True,True,False]])
        reset=torch.tensor([[False,True,False,True,True]])
        result=lag_run(x,h,torch.tensor([5]),lags=2,execution_mask=execute,reset_mask=reset,reset_history=resets)
        expected=torch.tensor([[[-2.,-1.],[0.,0.],[-1.,10.],[100.,200.],[0.,0.]]],dtype=torch.float64)
        torch.testing.assert_close(result,expected,rtol=0,atol=0);result.sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all());self.assertTrue(torch.isfinite(h.grad).all());self.assertTrue(torch.isfinite(resets.grad).all())
        self.assertEqual(float(x.grad[:,[1,3,4]].abs().sum()),0.)
        self.assertEqual(float(resets.grad[:,[0,1,2,4]].abs().sum()),0.)

    def test_leading_reset_ignores_unused_initial_and_all_skipped_lags_have_zero_gradient(self):
        x=torch.tensor([[[math.nan],[2.],[3.]]],requires_grad=True);h=torch.full((1,2,1),math.nan,requires_grad=True)
        reset_history=torch.full((1,3,2,1),math.nan);reset_history[0,1,:,0]=torch.tensor([8.,9.]);reset_history.requires_grad_()
        kwargs={'lags':2,'execution_mask':torch.tensor([[False,True,True]]),'reset_mask':torch.tensor([[True,True,False]]),'reset_history':reset_history}
        result=lag_run(x,h,torch.tensor([3]),**kwargs);result.sum().backward()
        torch.testing.assert_close(result,torch.tensor([[[0.,0.],[8.,9.],[9.,2.]]]))
        self.assertEqual(float(h.grad.abs().sum()),0.)
        one=torch.tensor([[[math.nan]]],requires_grad=True);initial=torch.full((1,2,1),math.nan,requires_grad=True)
        empty=lag_run(one,initial,torch.tensor([1]),lags=2,execution_mask=torch.zeros(1,1,dtype=torch.bool))
        empty.sum().backward();self.assertEqual(float(one.grad.abs().sum()),0.);self.assertEqual(float(initial.grad.abs().sum()),0.)

    def test_controls_budget_and_shape_refuse_before_numerical_masks(self):
        values=args();values['execution_mask']=torch.tensor([[1.,0.,1.,1.,0.]])
        with self.assertRaises(ValueError):filtered(values)
        x=torch.zeros(1,4,2);h=torch.zeros(1,2,2)
        with self.assertRaises(ValueError):lag_run(x,h,torch.tensor([4]),lags=2,reset_mask=torch.ones(1,4,dtype=torch.bool))
        with patch.object(torch,'isfinite',side_effect=AssertionError('premature allocation')):
            with self.assertRaisesRegex(ValueError,'budget'):lag_run(x,h,torch.tensor([4]),lags=2,max_elements=1)
            with self.assertRaisesRegex(ValueError,'budget'):filtered({**args(),'max_work_elements':1})


if __name__=='__main__':unittest.main()
