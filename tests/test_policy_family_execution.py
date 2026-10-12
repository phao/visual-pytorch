"""Actual native/captured Family10 policies, repeated ownership and dynamic time."""
import unittest
import copy
import torch
from app.core import Workspace, compile_model
from app.family_export import family_source
from app.family_records import retain, compile_specialization
from policy_runtime_fixture import family
from test_policy_family import repeated

torch.set_num_threads(1)


class PolicyFamilyExecutionTests(unittest.TestCase):
    def test_actual_family10_native_and_captured_runtime_initialization_and_dynamic_time(self):
        for temporal in (False, True):
            workspace = Workspace()
            try:
                f = family(temporal); retained = retain(workspace, family=f, materialize=False)
                native = compile_specialization(workspace, retained['specialization'], constrained=False)
                namespace = {}; exec(family_source(f, constrained=False)['code'], namespace)
                exported = namespace['ArchitectureFamily']()
                self.assertEqual(native.parameter_policy_receipt, exported.network.parameter_policy_receipt)
                self.assertFalse(native.layer_0.weight.requires_grad); self.assertFalse(exported.network.layer_0.weight.requires_grad)
                for steps in (1, 5) if temporal else (1,):
                    x = torch.randn(2, steps, 16) if temporal else torch.randn(2, 16)
                    torch.testing.assert_close(native(x), exported(x), rtol=0, atol=0)
                    torch.testing.assert_close(native(x), .6 * x)
            finally: workspace.db.close()

    def test_actual_repeated_family_native_and_captured_unique_optimizer_owners(self):
        for mode in ('independent', 'tied'):
            f = repeated(mode); namespace = {}; exec(family_source(f, constrained=False)['code'], namespace)
            model = namespace['ArchitectureFamily']()
            counts = model.network.parameter_policy_receipt['counts']
            self.assertEqual(counts['trainable'], 6 if mode == 'independent' else 2)
            self.assertEqual(sum(p.numel() for p in model.parameters() if not p.requires_grad), 12 if mode == 'independent' else 4)
            x = torch.ones(3, 2, requires_grad=True); model(x).sum().backward()
            self.assertTrue(torch.isfinite(x.grad).all())
            self.assertTrue(all(p.grad is None for p in model.parameters() if not p.requires_grad))

    def test_warm_start_cannot_silently_overwrite_frozen_supplied_base(self):
        namespace = {}; exec(family_source(family(), constrained=False)['code'], namespace)
        model = namespace['ArchitectureFamily'](); checkpoint = model.semantic_state()
        model.warm_start(checkpoint, strict=True)
        bad = copy.deepcopy(checkpoint); bad['weights']['network.layer_0.weight'][0][0] += 1
        before = {k: v.clone() for k, v in model.state_dict().items()}
        with self.assertRaisesRegex(ValueError, 'frozen initial base'): model.warm_start(bad)
        for path, value in model.state_dict().items(): self.assertTrue(torch.equal(value, before[path]))


if __name__ == '__main__': unittest.main()
