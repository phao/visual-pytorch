"""Bounded primitive checks; no application admission, fitting or registrations.

Run with CPU4,6 and OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS=1:
python -m unittest discover -s tests -p test_gaussian_filter.py -v
All fixtures are far below 10 MiB; the largest path oracle has T=8, K=2.
"""

import ast
import importlib.util
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch

from app.gaussian_filter import (
    MAX_STEPS,
    MAX_WORK_ELEMENTS,
    filter_diagonal_gaussian_hmm,
    variances_from_raw,
)
from hmm_reference import enumerate_gaussian_hmm


class GaussianFilterTests(unittest.TestCase):
    def fixture(self, *, batch=2, steps=3, dimensions=2, dtype=torch.float64):
        # No random/global seeds. Time-dependent rows are asymmetric; variances
        # have a common time scale, allowing an independent oracle change of
        # variables (with its full density Jacobian), not a filter recursion.
        y = [[[0.2 + 0.3 * t - 0.4 * b + 0.1 * d for d in range(dimensions)]
              for t in range(steps)] for b in range(batch)]
        p = [[[[0.9 - 0.04 * t, 0.1 + 0.04 * t],
                [0.3 + 0.02 * t, 0.7 - 0.02 * t]] for t in range(steps)]
             for _ in range(batch)]
        mu = [[[[0.4 * k - 0.15 * t + 0.2 * d + 0.1 * b
                 for d in range(dimensions)] for k in range(2)]
               for t in range(steps)] for b in range(batch)]
        var = [[[[((1.0 + k + 0.5 * d) * (1.0 + 0.2 * t))
                  for d in range(dimensions)] for k in range(2)]
                for t in range(steps)] for _ in range(batch)]
        return {
            "observations": torch.tensor(y, dtype=dtype).reshape(batch, steps, dimensions),
            "log_initial": torch.tensor([[0.8 - 0.1 * b, 0.2 + 0.1 * b]
                                         for b in range(batch)], dtype=dtype).reshape(batch, 2).log(),
            "log_transition": torch.tensor(p, dtype=dtype).reshape(batch, steps, 2, 2).log(),
            "means": torch.tensor(mu, dtype=dtype).reshape(batch, steps, 2, dimensions),
            "positive_variances": torch.tensor(var, dtype=dtype).reshape(batch, steps, 2, dimensions),
            "lengths": torch.full((batch,), steps, dtype=torch.int64),
        }

    def assert_reference(self, args, *, tolerance=3e-12):
        result = filter_diagonal_gaussian_hmm(**args)
        batch, steps, dimensions = args["observations"].shape
        self.assertEqual(result.filtered_log_probs.shape, (batch, steps, 2))
        self.assertEqual(result.final_log_probs.shape, (batch, 2))
        self.assertEqual(result.step_log_evidence.shape, (batch, steps))
        self.assertEqual(result.sequence_log_likelihood.shape, (batch,))
        for b in range(batch):
            length = args["lengths"][b].item()
            mask = args.get("emission_mask")
            flags = [True] * steps if mask is None else mask[b].tolist()
            # Divide each emitted time by sqrt(scale), keeping static variances
            # in the independent path enumerator. Densities acquire the explicit
            # -D/2 log(scale) Jacobian. This exercises time-varying variances.
            base = [[1.0 + k + 0.5 * d for d in range(dimensions)] for k in range(2)]
            observations, means, corrections = [], [], []
            for t in range(steps):
                if t >= length or not flags[t]:
                    observations.append(None)
                    means.append(None)
                    corrections.append(0.0)
                    continue
                scale = args["positive_variances"][b, t, 0, 0].item()
                for k in range(2):
                    for d in range(dimensions):
                        self.assertAlmostEqual(args["positive_variances"][b, t, k, d].item(),
                                               base[k][d] * scale, delta=tolerance * 10)
                observations.append([x / math.sqrt(scale) for x in args["observations"][b, t].tolist()])
                means.append([[x / math.sqrt(scale) for x in row]
                              for row in args["means"][b, t].tolist()])
                corrections.append(-0.5 * dimensions * math.log(scale))
            oracle = enumerate_gaussian_hmm(
                observations, initial_belief=args["log_initial"][b].exp().tolist(),
                transitions_by_time=args["log_transition"][b].exp().tolist(),
                means_by_time=means, variances=base, length=length, emission_mask=flags,
                probability_tolerance=1e-8 if args["observations"].dtype == torch.float32 else 1e-12,
            )
            prefix = [value + math.fsum(corrections[:t + 1])
                      for t, value in enumerate(oracle.prefix_log_likelihoods)]
            previous = 0.0
            for t in range(length):
                torch.testing.assert_close(
                    result.filtered_log_probs[b, t],
                    result.final_log_probs.new_tensor(oracle.filtered_log_probabilities[t]),
                    atol=tolerance, rtol=tolerance,
                )
                self.assertAlmostEqual(result.step_log_evidence[b, t].item(),
                                       prefix[t] - previous, delta=tolerance * 10)
                self.assertAlmostEqual(result.step_log_evidence[b, :t + 1].sum().item(),
                                       prefix[t], delta=tolerance * 10)
                previous = prefix[t]
            final = (result.final_log_probs.new_tensor(oracle.filtered_log_probabilities[-1])
                     if length else args["log_initial"][b])
            torch.testing.assert_close(result.final_log_probs[b], final, atol=tolerance, rtol=tolerance)
            for t in range(length, steps):
                torch.testing.assert_close(result.filtered_log_probs[b, t], final,
                                           atol=tolerance, rtol=tolerance)
                self.assertEqual(result.step_log_evidence[b, t].item(), 0.0)
            self.assertAlmostEqual(result.sequence_log_likelihood[b].item(), previous,
                                   delta=tolerance * 10)
        return result

    def test_every_prefix_t1_through_t8_time_dependent_parameters(self):
        for steps in range(1, 9):
            with self.subTest(steps=steps):
                self.assert_reference(self.fixture(steps=steps))

    def test_asymmetric_transition_orientation_and_initial_before_p0(self):
        args = self.fixture(batch=1, steps=1, dimensions=1)
        args["observations"].fill_(0.5)
        args["means"][0, 0, :, 0] = torch.tensor([0.0, 2.0])
        args["positive_variances"][0, 0, :, 0] = torch.tensor([1.0, 4.0])
        result = filter_diagonal_gaussian_hmm(**args)
        densities = [math.exp(-0.5 * (math.log(2 * math.pi * v) + (0.5 - mu) ** 2 / v))
                     for mu, v in ((0.0, 1.0), (2.0, 4.0))]
        weights = [0.78 * densities[0], 0.22 * densities[1]]
        torch.testing.assert_close(result.final_log_probs.exp()[0],
                                   torch.tensor([w / sum(weights) for w in weights], dtype=torch.float64))
        self.assertAlmostEqual(result.sequence_log_likelihood.item(), math.log(sum(weights)), delta=1e-14)
        wrong_initial = 0.8 * densities[0] + 0.2 * densities[1]
        self.assertGreater(abs(sum(weights) - wrong_initial), 0.001)

    def test_empty_t_and_empty_batch_shapes_neutral_scores(self):
        for batch, steps in ((2, 0), (0, 0), (0, 3)):
            args = self.fixture(batch=batch, steps=steps)
            result = self.assert_reference(args)
            torch.testing.assert_close(result.final_log_probs, args["log_initial"])
            self.assertTrue(bool((result.sequence_log_likelihood == 0).all()))

    def test_empty_time_likelihood_backward_has_zero_gradients_for_all_inputs(self):
        args = self.fixture(steps=0)
        names = ("observations", "log_initial", "log_transition", "means", "positive_variances")
        for name in names:
            args[name].requires_grad_()
        result = filter_diagonal_gaussian_hmm(**args)
        result.sequence_log_likelihood.sum().backward()
        for name in names:
            self.assertIsNotNone(args[name].grad, name)
            self.assertTrue(bool((args[name].grad == 0).all()), name)

    def test_zero_lengths_padding_nan_sanitized_before_reductions_and_gradients(self):
        args = self.fixture(batch=3, steps=3)
        args["lengths"] = torch.tensor([0, 1, 3])
        for b, length in enumerate(args["lengths"].tolist()):
            for name in ("observations", "log_transition", "means", "positive_variances"):
                args[name][b, length:] = math.nan
        for name in ("observations", "log_initial", "log_transition", "means", "positive_variances"):
            args[name].requires_grad_()
        snapshots = {name: value.detach().clone() for name, value in args.items()}
        result = self.assert_reference(args)
        objective = result.sequence_log_likelihood.sum() + result.filtered_log_probs.exp().square().sum()
        objective.backward()
        for name in ("observations", "log_initial", "log_transition", "means", "positive_variances"):
            self.assertTrue(bool(torch.isfinite(args[name].grad).all()), name)
            if name != "log_initial":
                for b, length in enumerate(args["lengths"].tolist()):
                    self.assertTrue(bool((args[name].grad[b, length:] == 0).all()), name)
        for name, value in args.items():
            torch.testing.assert_close(value, snapshots[name], equal_nan=True)

    def test_missing_emissions_advance_but_padding_does_not(self):
        args = self.fixture(batch=2, steps=3, dimensions=1)
        args["lengths"] = torch.tensor([3, 0], dtype=torch.int32)
        args["emission_mask"] = torch.zeros((2, 3), dtype=torch.bool)
        for name in ("observations", "means", "positive_variances"):
            args[name].fill_(math.nan)
        args["log_transition"][1].fill_(math.nan)
        for name in ("observations", "means", "positive_variances", "log_transition"):
            args[name].requires_grad_()
        result = self.assert_reference(args)
        torch.testing.assert_close(result.filtered_log_probs[0, 0].exp(),
                                   torch.tensor([0.78, 0.22], dtype=torch.float64))
        self.assertFalse(torch.equal(result.final_log_probs[0], args["log_initial"][0]))
        self.assertTrue(torch.equal(result.final_log_probs[1], args["log_initial"][1]))
        self.assertTrue(bool((result.step_log_evidence == 0).all()))
        (result.sequence_log_likelihood.sum() + result.final_log_probs.exp().square().sum()).backward()
        for name in ("observations", "means", "positive_variances", "log_transition"):
            self.assertTrue(bool(torch.isfinite(args[name].grad).all()))
            if name != "log_transition":
                self.assertTrue(bool((args[name].grad == 0).all()))

    def test_mixed_missingness_matches_each_oracle_prefix(self):
        args = self.fixture(steps=8)
        args["emission_mask"] = torch.tensor([[True, False, True, False, True, True, False, True],
                                               [False, True, False, True, True, False, True, False]])
        for name in ("observations", "means", "positive_variances"):
            args[name][~args["emission_mask"]] = math.nan
        result = self.assert_reference(args)
        self.assertTrue(bool((result.step_log_evidence[~args["emission_mask"]] == 0).all()))

    def test_unit_variance_full_gaussian_constant(self):
        args = self.fixture(batch=1, steps=3, dimensions=3)
        args["positive_variances"].fill_(1.0)
        args["means"].zero_()
        result = filter_diagonal_gaussian_hmm(**args)
        expected = -0.5 * (3 * math.log(2.0 * math.pi) + args["observations"].square().sum(-1))
        torch.testing.assert_close(result.step_log_evidence, expected, atol=1e-14, rtol=1e-14)

    def test_narrow_variance_negative_nll_and_no_hidden_floor(self):
        args = self.fixture(batch=1, steps=1, dimensions=1)
        args["observations"].zero_()
        args["means"].zero_()
        for variance in (1e-12, math.ulp(0.0)):
            with self.subTest(variance=variance):
                args["positive_variances"].fill_(variance)
                result = filter_diagonal_gaussian_hmm(**args)
                expected = -0.5 * (math.log(2 * math.pi) + math.log(variance))
                self.assertAlmostEqual(result.sequence_log_likelihood.item(), expected, delta=1e-12)
                self.assertLess(-result.sequence_log_likelihood.item(), 0.0)

    def test_structural_unreachable_groups_finite_gradients(self):
        for dtype in (torch.float32, torch.float64):
            args = self.fixture(batch=1, steps=3, dimensions=1, dtype=dtype)
            args["log_initial"] = torch.tensor([[0.0, -math.inf]], dtype=dtype, requires_grad=True)
            args["log_transition"] = torch.tensor([[[[0.0, -math.inf],
                                                     [-math.inf, 0.0]]] * 3], dtype=dtype, requires_grad=True)
            for name in ("observations", "means", "positive_variances"):
                args[name].requires_grad_()
            result = filter_diagonal_gaussian_hmm(**args)
            self.assertTrue(bool((result.filtered_log_probs[..., 0] == 0).all()))
            self.assertTrue(bool(torch.isneginf(result.filtered_log_probs[..., 1]).all()))
            (result.sequence_log_likelihood.sum() + result.filtered_log_probs.exp().square().sum()).backward()
            for name in ("observations", "log_initial", "log_transition", "means", "positive_variances"):
                self.assertTrue(bool(torch.isfinite(args[name].grad).all()), name)
            self.assertTrue(bool((args["means"].grad[:, :, 1] == 0).all()))
            self.assertTrue(bool((args["log_transition"].grad[:, :, :, 1] == 0).all()))

    def test_deterministic_switch_and_finite_log_underflow_tail(self):
        args = self.fixture(batch=1, steps=3, dimensions=1)
        args["log_initial"] = torch.tensor([[0.0, -math.inf]], dtype=torch.float64)
        args["log_transition"][:] = torch.tensor([[-math.inf, 0.0], [0.0, -math.inf]])
        result = filter_diagonal_gaussian_hmm(**args)
        torch.testing.assert_close(result.filtered_log_probs.exp()[0],
                                   torch.tensor([[0., 1.], [1., 0.], [0., 1.]], dtype=torch.float64))
        args = self.fixture(batch=1, steps=1, dimensions=1)
        args["observations"].fill_(1000.)
        args["means"][0, 0, :, 0] = torch.tensor([0., 1000.])
        args["positive_variances"].fill_(1.)
        result = filter_diagonal_gaussian_hmm(**args)
        self.assertEqual(result.final_log_probs.exp()[0, 0].item(), 0.0)
        self.assertTrue(bool(torch.isfinite(result.final_log_probs).all()))
        self.assertLess(result.final_log_probs[0, 0].item(), -499000.)

    def test_extreme_common_tail_preserves_nonuniform_prior(self):
        for dtype, observation in ((torch.float32, 1e15), (torch.float64, 1e150)):
            args = self.fixture(batch=1, steps=3, dimensions=1, dtype=dtype)
            args["observations"].fill_(observation)
            args["means"].zero_()
            args["positive_variances"].fill_(1.)
            args["log_transition"][:] = torch.tensor([[0., -math.inf], [-math.inf, 0.]], dtype=dtype)
            result = filter_diagonal_gaussian_hmm(**args)
            torch.testing.assert_close(result.filtered_log_probs,
                                       args["log_initial"][:, None].expand(1, 3, 2))
            self.assertTrue(bool(torch.isfinite(result.sequence_log_likelihood).all()))

    def test_unreachable_high_density_does_not_destroy_reachable_tail_priors(self):
        # Three states: two reachable identical enormous tails, one impossible
        # state at the observation. Centering over all states would lose priors.
        for dtype, value in ((torch.float32, 1e15), (torch.float64, 1e150)):
            args = {
                "observations": torch.tensor([[[value]]], dtype=dtype),
                "log_initial": torch.tensor([[.8, .2, 0.]], dtype=dtype).log(),
                "log_transition": torch.eye(3, dtype=dtype).log()[None, None],
                "means": torch.tensor([[[[0.], [0.], [value]]]], dtype=dtype),
                "positive_variances": torch.ones((1, 1, 3, 1), dtype=dtype),
                "lengths": torch.tensor([1]),
            }
            result = filter_diagonal_gaussian_hmm(**args)
            torch.testing.assert_close(result.final_log_probs, args["log_initial"])

    def test_state_permutation_preserves_likelihood_and_permutes_beliefs(self):
        args = self.fixture(steps=8)
        result = filter_diagonal_gaussian_hmm(**args)
        permuted = dict(args)
        permuted["log_initial"] = args["log_initial"].flip(-1)
        permuted["log_transition"] = args["log_transition"].flip((-1, -2))
        for name in ("means", "positive_variances"):
            permuted[name] = args[name].flip(-2)
        other = filter_diagonal_gaussian_hmm(**permuted)
        torch.testing.assert_close(other.filtered_log_probs, result.filtered_log_probs.flip(-1))
        torch.testing.assert_close(other.final_log_probs, result.final_log_probs.flip(-1))
        torch.testing.assert_close(other.step_log_evidence, result.step_log_evidence)
        torch.testing.assert_close(other.sequence_log_likelihood, result.sequence_log_likelihood)

    def test_native_float32_matches_independent_binary64_reference(self):
        # Values .75/.25 and .5/.5 are exactly representable, so conversion to
        # the oracle's 1e-8 distribution tolerance does not hide f32 rounding.
        args = self.fixture(steps=8, dtype=torch.float32)
        args["log_initial"] = torch.tensor([[.75, .25], [.5, .5]], dtype=torch.float32).log()
        args["log_transition"][:] = torch.tensor([[.75, .25], [.5, .5]], dtype=torch.float32).log()
        result = self.assert_reference(args, tolerance=4e-6)
        for field in (result.filtered_log_probs, result.final_log_probs,
                      result.step_log_evidence, result.sequence_log_likelihood):
            self.assertEqual(field.dtype, torch.float32)

    def test_gradcheck_observations_initial_transition_means_and_raw_variances(self):
        args = self.fixture(batch=1, steps=2, dimensions=1)
        inputs = (args["observations"].requires_grad_(),
                  torch.tensor([[.3, -.2]], dtype=torch.float64, requires_grad=True),
                  args["log_transition"].clone().requires_grad_(),
                  args["means"].requires_grad_(),
                  torch.tensor([[[[.2], [-.4]], [[.5], [.1]]]], dtype=torch.float64, requires_grad=True))

        def wrapper(observations, initial_logits, transition_logits, means, raw_variances):
            result = filter_diagonal_gaussian_hmm(
                observations, log_initial=initial_logits.log_softmax(-1),
                log_transition=transition_logits.log_softmax(-1), means=means,
                positive_variances=variances_from_raw(raw_variances, positive_floor=0.03),
                lengths=args["lengths"],
            )
            return (result.filtered_log_probs, result.final_log_probs,
                    result.step_log_evidence, result.sequence_log_likelihood)

        self.assertTrue(torch.autograd.gradcheck(wrapper, inputs, eps=1e-6, atol=2e-5, rtol=2e-4))

    def test_prefix_causality_and_batch_separability(self):
        args = self.fixture(steps=3)
        expected = filter_diagonal_gaussian_hmm(**args)
        changed = {key: value.clone() for key, value in args.items()}
        changed["observations"][:, 1:] += 10.
        changed["means"][:, 1:] -= 5.
        other = filter_diagonal_gaussian_hmm(**changed)
        torch.testing.assert_close(other.filtered_log_probs[:, 0], expected.filtered_log_probs[:, 0])
        torch.testing.assert_close(other.step_log_evidence[:, 0], expected.step_log_evidence[:, 0])
        for b in range(2):
            single = filter_diagonal_gaussian_hmm(**{key: value[b:b + 1] for key, value in args.items()})
            torch.testing.assert_close(single.filtered_log_probs[0], expected.filtered_log_probs[b])
            torch.testing.assert_close(single.sequence_log_likelihood[0], expected.sequence_log_likelihood[b])

    def test_noncontiguous_inputs_are_supported_without_mutation(self):
        args = self.fixture(steps=3, dimensions=2)
        other = dict(args)
        for name in ("observations", "means", "positive_variances"):
            other[name] = args[name].transpose(-1, -2).contiguous().transpose(-1, -2)
            self.assertFalse(other[name].is_contiguous())
        result = filter_diagonal_gaussian_hmm(**args)
        actual = filter_diagonal_gaussian_hmm(**other)
        torch.testing.assert_close(actual.filtered_log_probs, result.filtered_log_probs)
        torch.testing.assert_close(actual.sequence_log_likelihood, result.sequence_log_likelihood)

    def test_invalid_active_values_even_unreachable_and_missing_transitions(self):
        for name in ("observations", "means", "positive_variances", "log_initial", "log_transition"):
            bad_values = (math.nan, math.inf) if name.startswith("log_") else (math.nan, math.inf, -math.inf)
            for value in bad_values:
                args = self.fixture(batch=1, steps=1, dimensions=1)
                args[name].flatten()[-1] = value
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    filter_diagonal_gaussian_hmm(**args)
        for value in (0., -1.):
            args = self.fixture(batch=1, steps=1, dimensions=1)
            args["positive_variances"].flatten()[-1] = value
            with self.assertRaises(ValueError):
                filter_diagonal_gaussian_hmm(**args)
        args = self.fixture(batch=1, steps=1, dimensions=1)
        args["log_initial"] = torch.tensor([[0., -math.inf]], dtype=torch.float64)
        args["means"][0, 0, 1, 0] = math.nan
        with self.assertRaises(ValueError):
            filter_diagonal_gaussian_hmm(**args)
        args = self.fixture(batch=1, steps=1)
        args["emission_mask"] = torch.zeros((1, 1), dtype=torch.bool)
        args["log_transition"].flatten()[-1] = math.nan
        with self.assertRaises(ValueError):
            filter_diagonal_gaussian_hmm(**args)

    def test_invalid_normalization_refused_without_repair(self):
        for name in ("log_initial", "log_transition"):
            for value in (-math.inf, 0., -2., 1.):
                args = self.fixture(batch=1, steps=1)
                args[name].fill_(value)
                with self.subTest(name=name, value=value), self.assertRaisesRegex(ValueError, "normalized"):
                    filter_diagonal_gaussian_hmm(**args)
        for dtype, shift in ((torch.float32, 1e-4), (torch.float64, 1e-10)):
            args = self.fixture(dtype=dtype)
            args["log_initial"] += shift
            with self.assertRaisesRegex(ValueError, "normalized"):
                filter_diagonal_gaussian_hmm(**args)

    def test_shapes_dtypes_devices_and_unrelated_controls_refused(self):
        args = self.fixture()
        for name in args:
            changed = dict(args)
            changed[name] = args[name].flatten() if name != "lengths" else args[name][:1]
            with self.subTest(name=name), self.assertRaises(ValueError):
                filter_diagonal_gaussian_hmm(**changed)
        for name in ("observations", "log_initial", "log_transition", "means", "positive_variances"):
            changed = dict(args)
            changed[name] = args[name].float()
            with self.subTest(name=name), self.assertRaises(ValueError):
                filter_diagonal_gaussian_hmm(**changed)
            changed[name] = args[name].to("meta")
            with self.subTest(meta=name), self.assertRaises(ValueError):
                filter_diagonal_gaussian_hmm(**changed)
        for dtype in (torch.float16, torch.bfloat16, torch.int64):
            changed = {key: value.to(dtype) if value.is_floating_point() else value
                       for key, value in args.items()}
            with self.assertRaises(ValueError):
                filter_diagonal_gaussian_hmm(**changed)
        for lengths in (torch.tensor([1., 2.]), torch.tensor([True, False]),
                        torch.tensor([-1, 3]), torch.tensor([4, 3]), torch.tensor([1, 2], device="meta")):
            with self.subTest(lengths=lengths), self.assertRaises(ValueError):
                filter_diagonal_gaussian_hmm(**dict(args, lengths=lengths))
        for mask in (torch.ones((2, 3)), torch.ones((2, 3, 1), dtype=torch.bool),
                     torch.ones((2, 3), dtype=torch.bool, device="meta")):
            with self.assertRaises(ValueError):
                filter_diagonal_gaussian_hmm(**dict(args, emission_mask=mask))
        for control in ("reset_mask", "execution_mask", "loss_mask", "state_labels", "variance_floor"):
            with self.subTest(control=control), self.assertRaises(TypeError):
                filter_diagonal_gaussian_hmm(**dict(args, **{control: None}))
        with self.assertRaises(TypeError):
            filter_diagonal_gaussian_hmm(**dict(args, observations=args["observations"].tolist()))
        with self.assertRaises(ValueError):
            filter_diagonal_gaussian_hmm(**dict(args, observations=args["observations"].to_sparse()))
        no_states = dict(args)
        no_states["log_initial"] = args["log_initial"][:, :0]
        with self.assertRaisesRegex(ValueError, "K and D"):
            filter_diagonal_gaussian_hmm(**no_states)
        with self.assertRaisesRegex(ValueError, "K and D"):
            filter_diagonal_gaussian_hmm(**dict(args, observations=args["observations"][:, :, :0]))

    def test_work_guard_precedes_expansion_density_and_log_reductions(self):
        args = self.fixture()
        for budget in (0, -1, True, 1.5, MAX_WORK_ELEMENTS + 1):
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                filter_diagonal_gaussian_hmm(**dict(args, max_work_elements=budget))
        with patch.object(torch, "where", side_effect=AssertionError("expanded")), \
             patch.object(torch, "arange", side_effect=AssertionError("mask allocated")), \
             patch.object(torch, "logsumexp", side_effect=AssertionError("reduction")):
            with self.assertRaisesRegex(ValueError, "work budget exceeded"):
                filter_diagonal_gaussian_hmm(**dict(args, max_work_elements=1))
            # Expanded views use tiny actual storage; admission must inspect
            # shape, not stored bytes. This case exceeds the hard K*K budget.
            huge = {
                "observations": torch.zeros(1, 1, 1, dtype=torch.float64),
                "log_initial": torch.zeros(1, 1, dtype=torch.float64).expand(1, 1000),
                "log_transition": torch.zeros(1, 1, 1, 1, dtype=torch.float64).expand(1, 1, 1000, 1000),
                "means": torch.zeros(1, 1, 1, 1, dtype=torch.float64).expand(1, 1, 1000, 1),
                "positive_variances": torch.ones(1, 1, 1, 1, dtype=torch.float64).expand(1, 1, 1000, 1),
                "lengths": torch.tensor([0]),
            }
            with self.assertRaisesRegex(ValueError, "work budget exceeded"):
                filter_diagonal_gaussian_hmm(**huge)
            empty = self.fixture(batch=0)
            for name in ("observations", "log_transition", "means", "positive_variances"):
                shape = list(empty[name].shape)
                shape[1] = MAX_STEPS + 1
                empty[name] = empty[name].new_empty(shape)
            with self.assertRaisesRegex(ValueError, "work budget exceeded"):
                filter_diagonal_gaussian_hmm(**empty)

    def test_finite_input_arithmetic_overflow_is_explicit(self):
        for dtype, value in ((torch.float32, 1e30), (torch.float64, 1e200)):
            args = self.fixture(batch=1, steps=1, dimensions=1, dtype=dtype)
            args["observations"].fill_(value)
            with self.assertRaises(FloatingPointError):
                filter_diagonal_gaussian_hmm(**args)
        args = self.fixture(batch=1, steps=8, dimensions=1)
        args["observations"].fill_(1e154)
        args["means"].zero_()
        args["positive_variances"].fill_(1.)
        with self.assertRaisesRegex(FloatingPointError, "sequence log likelihood"):
            filter_diagonal_gaussian_hmm(**args)

    def test_declared_softplus_floor_transform_and_validation(self):
        raw = torch.tensor([-1000., -1., 0., 1., 1000.], dtype=torch.float64, requires_grad=True)
        transformed = variances_from_raw(raw, positive_floor=0.125)
        expected = torch.tensor([0.125 + (math.log1p(math.exp(-abs(x))) + max(x, 0.))
                                 for x in raw.detach().tolist()], dtype=torch.float64)
        torch.testing.assert_close(transformed, expected)
        transformed.sum().backward()
        torch.testing.assert_close(raw.grad, raw.detach().sigmoid())
        for floor in (0., -1., math.nan, math.inf, True, "0.1", 10**400):
            with self.subTest(floor=floor), self.assertRaises((ValueError, TypeError)):
                variances_from_raw(raw, positive_floor=floor)
        with self.assertRaises(TypeError):
            variances_from_raw(raw)
        with self.assertRaises(ValueError):
            variances_from_raw(torch.zeros(1, dtype=torch.float32), positive_floor=1e-100)
        with self.assertRaises(ValueError):
            variances_from_raw(torch.tensor([math.nan]), positive_floor=.1)

    def test_module_import_is_pure_and_torch_is_lazy(self):
        module_path = Path(__file__).resolve().parents[1] / "app" / "gaussian_filter.py"
        source = module_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        top_imports = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
        self.assertFalse(any((isinstance(node, ast.Import) and any(n.name == "torch" for n in node.names))
                             or (isinstance(node, ast.ImportFrom) and node.module == "torch")
                             for node in top_imports))
        spec = importlib.util.spec_from_file_location("_pure_gaussian_filter_probe", module_path)
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {spec.name: module, "torch": None}):
            spec.loader.exec_module(module)
        self.assertTrue(callable(module.filter_diagonal_gaussian_hmm))


if __name__ == "__main__":
    unittest.main()
