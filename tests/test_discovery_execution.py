"""Every discovery witness executes identically through native/exported Torch."""
import unittest
from app.discovery import capabilities, operator_schema
from app.core import compile_model, generate_code


class DiscoveryExecutionTests(unittest.TestCase):
    def test_all_registered_example_closures_native_and_standalone_match(self):
        import torch
        torch.set_num_threads(1)
        for kind in capabilities()['operators']:
            with self.subTest(kind=kind):
                witness = operator_schema(kind)['example']; models = witness['models']; root = witness['root']
                native = compile_model(models, root, constrained=False)
                namespace = {}; source = generate_code(models, root)
                exec(compile(source, 'discovery-standalone-' + kind, 'exec'), namespace)
                exported = namespace['model']
                exported.load_state_dict(native.state_dict(), strict=True)
                values = {group['id']: torch.full((2, *group['shape']), .25, dtype=torch.float32)
                          for group in native.contract['input_groups']}
                native.eval(); exported.eval()
                with torch.no_grad():
                    if native.contract['input_groups']:
                        expected = native.forward_tensor(values); actual = exported.forward_tensor(values)
                    else:
                        expected = native.forward_tensor(batch_size=2)
                        actual = exported.forward_tensor(batch_size=2)
                self.assertTrue(torch.equal(actual, expected), kind)
                self.assertTrue(torch.isfinite(actual).all(), kind)
                self.assertIs(actual.dtype, torch.float32)


if __name__ == '__main__': unittest.main()
