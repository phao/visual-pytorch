"""Edited roles are real PyTorch semantics, not a cosmetic diagram change."""
import unittest
import torch
import torch.nn.functional as F
from app.core import compile_model, generate_code
from app.recurrent_reconfiguration import build_edit
from test_recurrent_reconfiguration import editable
from test_recurrent_execution import oracle

torch.set_num_threads(1)


class ReconfiguredRecurrentExecutionTests(unittest.TestCase):
    def fixture(self):
        models, token, options = editable()
        options['sequence_inputs'][0]['input'] = 'h';options['states'][0]['input'] = 'x'
        models['parent'], _ = build_edit(models, token, options)
        n = next(n for n in models['parent']['graph']['nodes'] if n['id'] == token['node'])
        torch.manual_seed(91);native = compile_model(models, 'parent', constrained=False)
        scope = {};exec(generate_code(models, 'parent'), scope);exported = scope['model']
        exported.load_state_dict(native.state_dict(), strict=True)
        torch.manual_seed(92)
        values = {'input': torch.randn(8, 2)*.5,
                  n['sequence_inputs'][0]['source']: torch.randn(8, 4, 2)*.5,
                  n['states'][0]['source']: torch.randn(8, 2)*.2,
                  n['controls']['lengths']['source']: torch.tensor([[float(i%5)] for i in range(8)]),
                  n['controls']['mask']['source']: (torch.rand(8, 4)>.2).float(),
                  n['controls']['reset']['source']: (torch.rand(8, 4)>.85).float()}
        return native, exported, values, n

    def reference(self, values, n, p):
        # Original child input order remains [x, h]. The explicitly edited roles
        # now feed state into x and frames into h; permute weight COLUMN views,
        # not graph nodes, dataset columns, parameters or optimizer ownership.
        w = torch.cat((p[2][:, 2:], p[2][:, :2]), 1)
        trajectory, state = oracle(values[n['sequence_inputs'][0]['source']], values[n['states'][0]['source']],
          w, p[3], lengths=values[n['controls']['lengths']['source']][:, 0].long(),
          mask=values[n['controls']['mask']['source']].bool(), reset=values[n['controls']['reset']['source']].bool())
        return torch.cat((F.linear(values['input'], p[0], p[1]), trajectory.flatten(1), state), 1)

    def test_edited_native_export_outputs_and_all_gradients_match_independent_roles(self):
        native, exported, values, n = self.fixture()
        controls = {r['source'] for r in n['controls'].values()}
        for model in (native, exported):
            p = list(model.parameters());q = [v.detach().clone().requires_grad_() for v in p]
            data = {k: v.clone().requires_grad_(k not in controls) for k, v in values.items()}
            reference = {k: v.detach().clone().requires_grad_(v.requires_grad) for k, v in data.items()}
            actual = model.forward_tensor({g['name']: data[g['id']] for g in model.contract['input_groups']})
            expected = self.reference(reference, n, q)
            torch.testing.assert_close(actual, expected, rtol=3e-6, atol=2e-6)
            a = torch.autograd.grad(actual.square().sum(), [v for v in data.values() if v.requires_grad]+p)
            b = torch.autograd.grad(expected.square().sum(), [v for v in reference.values() if v.requires_grad]+q)
            for aa, bb in zip(a, b):torch.testing.assert_close(aa, bb, rtol=5e-6, atol=3e-6)

    def test_edited_model_fits_and_loaded_appfree_weights_reproduce_the_result(self):
        native, exported, values, n = self.fixture()
        named = {g['name']: values[g['id']] for g in native.contract['input_groups']}
        target = self.reference(values, n, [p.detach()*.7+.1 for p in native.parameters()]).detach()
        optimizer = torch.optim.Adam(native.parameters(), lr=.035)
        start = F.mse_loss(native.forward_tensor(named), target).item()
        for _ in range(40):
            optimizer.zero_grad();loss = F.mse_loss(native.forward_tensor(named), target);loss.backward();optimizer.step()
        native.eval();end = F.mse_loss(native.forward_tensor(named), target).item()
        self.assertLess(end, start*.15)
        exported.load_state_dict(native.state_dict(), strict=True);exported.eval()
        torch.testing.assert_close(native.forward_tensor(named), exported.forward_tensor(named), rtol=0, atol=0)
