"""Authored mixed branches: independent tensor/gradient checks and a bounded fit."""
import unittest
import torch
import torch.nn.functional as F
from app.core import compile_model, generate_code
from test_recurrent_authoring import authored
from app.recurrent_authoring import build
from test_recurrent_execution import oracle

torch.set_num_threads(1)


class AuthoredRecurrentExecutionTests(unittest.TestCase):
    def setup_models(self):
        models, token, options = authored();doc, nid = build(models, token, options);models['parent'] = doc
        n = next(n for n in doc['graph']['nodes'] if n['id'] == nid)
        torch.manual_seed(31);native = compile_model(models, 'parent', constrained=False)
        scope = {};exec(generate_code(models, 'parent'), scope);exported = scope['model']
        exported.load_state_dict(native.state_dict(), strict=True)
        torch.manual_seed(42)
        values = {'input': torch.randn(16, 2)*.5,
                  n['sequence_inputs'][0]['source']: torch.randn(16, 4, 2)*.5,
                  n['states'][0]['source']: torch.randn(16, 2)*.2,
                  n['controls']['lengths']['source']: torch.tensor([[float(i%5)] for i in range(16)]),
                  n['controls']['mask']['source']: (torch.rand(16, 4)>.2).float(),
                  n['controls']['reset']['source']: (torch.rand(16, 4)>.9).float()}
        return native, exported, values, n

    def reference(self, values, n, parameters):
        original = F.linear(values['input'], *parameters[:2])
        trajectory, final = oracle(values[n['sequence_inputs'][0]['source']], values[n['states'][0]['source']],
                                   *parameters[2:], lengths=values[n['controls']['lengths']['source']][:, 0].long(),
                                   mask=values[n['controls']['mask']['source']].bool(), reset=values[n['controls']['reset']['source']].bool())
        return torch.cat((original, trajectory.flatten(1), final), 1)

    def test_native_appfree_forward_and_all_input_parameter_gradients_match_oracle(self):
        native, exported, values, n = self.setup_models()
        for model in (native, exported):
            parameters = list(model.parameters());self.assertEqual(sum(p.numel() for p in parameters), 13)
            data = {k: v.detach().clone().requires_grad_(k not in {r['source'] for r in n['controls'].values()}) for k, v in values.items()}
            independent = {k: v.detach().clone().requires_grad_(v.requires_grad) for k, v in data.items()}
            q = [p.detach().clone().requires_grad_() for p in parameters]
            actual = model.forward_tensor({g['name']: data[g['id']] for g in model.contract['input_groups']})
            expected = self.reference(independent, n, q)
            torch.testing.assert_close(actual, expected)
            a = torch.autograd.grad(actual.square().sum(), [v for v in data.values() if v.requires_grad]+parameters)
            b = torch.autograd.grad(expected.square().sum(), [v for v in independent.values() if v.requires_grad]+q)
            for aa, bb in zip(a, b):torch.testing.assert_close(aa, bb, rtol=4e-6, atol=3e-6)

    def test_authored_branch_trains_and_exported_parameters_reproduce_predictions(self):
        native, exported, values, n = self.setup_models()
        teacher = [p.detach().clone()*.6+.13 for p in native.parameters()]
        target = self.reference(values, n, teacher).detach()
        named = {g['name']: values[g['id']] for g in native.contract['input_groups']}
        optimizer = torch.optim.Adam(native.parameters(), lr=.035)
        start = F.mse_loss(native.forward_tensor(named), target).item()
        for _ in range(60):
            optimizer.zero_grad();loss = F.mse_loss(native.forward_tensor(named), target);loss.backward();optimizer.step()
        native.eval();end = F.mse_loss(native.forward_tensor(named), target).item()
        self.assertLess(end, start*.1)
        exported.load_state_dict(native.state_dict(), strict=True);exported.eval()
        torch.testing.assert_close(native.forward_tensor(named), exported.forward_tensor(named), rtol=0, atol=0)
