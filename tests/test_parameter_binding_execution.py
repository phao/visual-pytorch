"""Copied retained values initialize native/exported frozen downstream weights."""
import copy
import unittest
import torch
import test_parameter_binding as fixture
from app.core import compile_model, generate_code
from app.parameter_binding import admit, current_bindings
from app.parameter_sources import read

torch.set_num_threads(1)


class ParameterBindingExecutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): fixture.ParameterBindingTests.setUpClass()
    @classmethod
    def tearDownClass(cls): fixture.ParameterBindingTests.tearDownClass()
    def setUp(self):
        self.fixture = fixture.ParameterBindingTests(methodName='runTest'); self.fixture.setUp()
    def tearDown(self): self.fixture.tearDown()

    def test_frozen_downstream_copy_native_export_values_and_upstream_gradients(self):
        f = self.fixture; admit(f.w, f.token, f.selector, f.selection)
        copied = torch.tensor(read(f.w, f.selection)['values'], dtype=torch.float32)
        features = copied.shape[-1]
        model = f.w.models()['destination']
        model['graph']['nodes'].insert(1, {'id': 'adapter', 'type': 'Linear', 'features': features, 'bias': False})
        model['graph']['edges'] = [{'from': 'input', 'to': 'adapter'}, {'from': 'adapter', 'to': 'linear1'}, {'from': 'linear1', 'to': 'output'}]
        f.w.put(model); self.assertIn(f.selector, current_bindings(f.w, 'destination'))
        models = f.w.models()
        with torch.random.fork_rng():
            torch.manual_seed(23); native = compile_model(models, 'destination')
            torch.manual_seed(23); namespace = {}; exec(generate_code(models, 'destination'), namespace); exported = namespace['model']
        for runner in (native, exported):
            p = dict(runner.named_parameters()); adapter = p['layer_0.weight']; weight = p['layer_1.weight']; bias = p['layer_1.bias']
            self.assertTrue(torch.equal(weight, copied)); self.assertFalse(weight.requires_grad)
            x = (torch.arange(4 * features, dtype=torch.float32).reshape(4, features) / 10).requires_grad_()
            refx = x.detach().clone().requires_grad_(); refa = adapter.detach().clone().requires_grad_(); refb = bias.detach().clone().requires_grad_()
            expected = (refx @ refa.T) @ copied.T + refb; actual = runner(x)
            torch.testing.assert_close(actual, expected)
            gradients = torch.autograd.grad(actual.square().sum(), (x, adapter, bias))
            reference = torch.autograd.grad(expected.square().sum(), (refx, refa, refb))
            for a, b in zip(gradients, reference): torch.testing.assert_close(a, b)
            self.assertGreater(float(gradients[1].abs().sum()), 0.); self.assertIsNone(weight.grad)
        torch.testing.assert_close(native(torch.zeros(2, features)), exported(torch.zeros(2, features)), rtol=0, atol=0)


if __name__ == '__main__': unittest.main()
