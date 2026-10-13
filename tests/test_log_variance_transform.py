"""Exact log-variance semantics, derivatives and checked standalone parity."""
import math
import unittest
from unittest.mock import patch
import torch
from torch import nn
from app.positive_variance import normalized,constructor_args,build,runtime_source,PositiveVarianceError


def node(**kwargs):return {'id':'variance','type':'PositiveVariance','op_version':2,'transform':'exp','positive_floor':0.,**kwargs}


class LogVarianceTests(unittest.TestCase):
    def test_exact_exp_no_hidden_floor_or_standard_deviation_and_analytic_gradient(self):
        raw=torch.tensor([-4.,-.5,0.,2.],dtype=torch.float64,requires_grad=True)
        variance=build(node())(raw)
        self.assertTrue(torch.equal(variance,raw.exp()))
        variance.sum().backward();self.assertTrue(torch.equal(raw.grad,raw.detach().exp()))
        for actual,log_variance in zip(variance.tolist(),raw.tolist()):self.assertAlmostEqual(actual,math.exp(log_variance),places=13)
        self.assertEqual(list(build(node()).parameters()),[])
        self.assertTrue(torch.autograd.gradcheck(build(node()),(raw.detach().requires_grad_(),)))

    def test_floor_is_explicit_and_does_not_change_exp_derivative(self):
        raw=torch.tensor([-.4,.2],dtype=torch.float64,requires_grad=True)
        module=build(node(positive_floor=.125));value=module(raw)
        torch.testing.assert_close(value,raw.exp()+.125,rtol=0,atol=0)
        value.sum().backward();torch.testing.assert_close(raw.grad,raw.detach().exp(),rtol=0,atol=0)

    def test_v1_grammar_and_constructor_kwargs_are_unchanged(self):
        old={'id':'v','type':'PositiveVariance','positive_floor':.125}
        self.assertEqual(normalized(old),{**old,'op_version':1})
        self.assertEqual(constructor_args(old),{'positive_floor':.125})
        raw=torch.tensor([-2.,0.,3.])
        self.assertTrue(torch.equal(build(old)(raw),torch.nn.functional.softplus(raw,beta=1,threshold=20)+.125))
        with self.assertRaises(PositiveVarianceError):normalized({**old,'transform':'exp'})
        with self.assertRaises(PositiveVarianceError):normalized({**old,'positive_floor':0})
        with self.assertRaises(PositiveVarianceError):normalized({**old,'op_version':2})
        for kwargs in ({'op_version':True},{'transform':'standard-deviation'},{'positive_floor':-1.},{'positive_floor':math.nan}):
            with self.assertRaises(PositiveVarianceError):normalized(node(**kwargs))

    def test_exp_overflow_underflow_nonfinite_and_budget_refuse_without_clamping(self):
        for dtype,values in ((torch.float32,(1000.,-1000.,math.nan,math.inf)),(torch.float64,(10000.,-10000.,math.nan,-math.inf))):
            for value in values:
                with self.subTest(dtype=dtype,value=value),self.assertRaises((ValueError,FloatingPointError)):
                    build(node())(torch.tensor([value],dtype=dtype))
        with patch.object(torch,'isfinite',side_effect=AssertionError('premature mask allocation')):
            with self.assertRaises(PositiveVarianceError):build(node(),max_elements=0)(torch.zeros(1))

    def test_checked_standalone_values_gradients_and_exact_constructor_fields(self):
        namespace={};exec(runtime_source(),namespace)
        cls=namespace['runtime_class'](nn)
        self.assertEqual(constructor_args(node()),{'positive_floor':0.,'transform':'exp'})
        for definition in (node(),node(transform='softplus',positive_floor=.1)):
            x=torch.tensor([-.5,.3],requires_grad=True);y=x.detach().clone().requires_grad_()
            native=build(definition)(x);portable=cls(**constructor_args(definition))(y)
            self.assertTrue(torch.equal(native,portable));native.sum().backward();portable.sum().backward()
            self.assertTrue(torch.equal(x.grad,y.grad))


if __name__=='__main__':unittest.main()
