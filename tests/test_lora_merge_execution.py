import unittest
import torch
import test_parameter_derivation as fixture
from app.core import compile_model, generate_code
from app.parameter_derivation import admit, merged_values
from app.parameter_sources import read

torch.set_num_threads(1)


class LoRAMergeExecutionTests(unittest.TestCase):
    def test_ranks_and_temporal_forward_match_independent_matrix_products(self):
        generator = torch.Generator().manual_seed(17)
        w = torch.eye(16) * .6
        for rank in (1, 2, 4):
            a = torch.randn((rank, 16), generator=generator) * .05
            b = torch.randn((16, rank), generator=generator) * .05
            merged = torch.tensor(merged_values(w.tolist(), a.tolist(), b.tolist(), alpha=4.), dtype=torch.float32)
            torch.testing.assert_close(merged, w + (4. / rank) * (b @ a), atol=1e-7, rtol=1e-6)
            for shape in ((5, 16), (5, 3, 16)):
                x = torch.randn(shape, generator=generator)
                expected = x @ w.T + (4. / rank) * ((x @ a.T) @ b.T)
                torch.testing.assert_close(x @ merged.T, expected, atol=2e-6, rtol=1e-5)

    def test_real_checkpoint_derived_native_and_fresh_export_preserve_predictions(self):
        fixture.ParameterDerivationTests.setUpClass()
        try:
            f = fixture.ParameterDerivationTests(methodName='runTest'); f.setUp()
            try:
                result = admit(f.w, f.selections)
                matrices = {key: torch.tensor(read(f.w, selection)['values'], dtype=torch.float32) for key, selection in f.selections.items()}
                native = compile_model(f.w.models(), result['model']['id']); namespace = {}
                exec(generate_code(f.w.models(), result['model']['id']), namespace); exported = namespace['model']
                generator = torch.Generator().manual_seed(23)
                x = torch.randn((8, 16), generator=generator)
                expected = x @ matrices['base'].T + 2 * ((x @ matrices['A'].T) @ matrices['B'].T)
                for runner in (native, exported):
                    torch.testing.assert_close(runner(x), expected, atol=2e-6, rtol=1e-5)
                    self.assertTrue(all(not p.requires_grad for p in runner.parameters()))
            finally: f.tearDown()
        finally: fixture.ParameterDerivationTests.tearDownClass()


if __name__ == '__main__': unittest.main()
