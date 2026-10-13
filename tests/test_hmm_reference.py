"""Analytic and invariance checks for the independent enumeration reference.

Run: python -m unittest discover -s tests -p test_hmm_reference.py -v
No app, Torch, NumPy, Qt, fitting, subprocess workers, or numerical threads.
"""

import copy
import math
import unittest
from unittest.mock import patch

from hmm_reference import (
    DEFAULT_PROBABILITY_TOLERANCE,
    MAX_ENUMERATED_PATHS,
    diagonal_gaussian_log_density,
    enumerate_gaussian_hmm,
)


def density(y, mean, variances):
    """Ordinary-density formula used only for moderate analytic test values."""
    return math.prod(
        math.exp(-(value - mu) ** 2 / (2.0 * variance))
        / math.sqrt(2.0 * math.pi * variance)
        for value, mu, variance in zip(y, mean, variances)
    )


class HMMReferenceTests(unittest.TestCase):
    def run_reference(self, observations, **changes):
        parameters = {
            "initial_belief": [0.8, 0.2],
            "transition": [[0.9, 0.1], [0.3, 0.7]],
            "means": [[0.0], [2.0]],
            "variances": [[1.0], [4.0]],
        }
        parameters.update(changes)
        return enumerate_gaussian_hmm(observations, **parameters)

    def assert_row_close(self, actual, expected, tolerance=2e-12):
        self.assertEqual(len(actual), len(expected))
        for value, target in zip(actual, expected):
            self.assertAlmostEqual(value, target, delta=tolerance)

    def assert_normalized(self, result):
        for probabilities, logs in zip(
            result.filtered_probabilities, result.filtered_log_probabilities
        ):
            self.assertAlmostEqual(math.fsum(probabilities), 1.0, delta=2e-14)
            for probability, log_probability in zip(probabilities, logs):
                self.assertEqual(probability, math.exp(log_probability))

    def test_density_has_full_gaussian_constant_and_variance_units(self):
        actual = diagonal_gaussian_log_density([2.0], [0.0], [4.0])
        expected = -0.5 * (math.log(2.0 * math.pi) + math.log(4.0) + 1.0)
        self.assertAlmostEqual(actual, expected, delta=1e-15)
        self.assertAlmostEqual(math.exp(actual), density([2.0], [0.0], [4.0]))

    def test_analytic_t1_nonuniform_initial_and_asymmetric_orientation(self):
        result = self.run_reference([[0.5]])
        # pi describes the state BEFORE P0: (.8,.2) @ P0 = (.78,.22).
        weights = [0.78 * density([0.5], [0.0], [1.0]),
                   0.22 * density([0.5], [2.0], [4.0])]
        likelihood = sum(weights)
        self.assertAlmostEqual(result.log_likelihood, math.log(likelihood), delta=1e-15)
        self.assert_row_close(result.filtered_probabilities[0],
                              [weight / likelihood for weight in weights])
        self.assertEqual(result.enumerated_paths, 2)
        # A mistaken initial emitting belief or transposed P gives other answers.
        wrong_initial = 0.8 * density([0.5], [0.0], [1.0]) + 0.2 * density([0.5], [2.0], [4.0])
        self.assertGreater(abs(likelihood - wrong_initial), 0.001)
        self.assert_normalized(result)

    def test_analytic_t2_four_paths_and_each_prefix(self):
        observations = [[0.5], [1.5]]
        result = self.run_reference(observations)
        b0 = [density(observations[0], [0.0], [1.0]),
              density(observations[0], [2.0], [4.0])]
        b1 = [density(observations[1], [0.0], [1.0]),
              density(observations[1], [2.0], [4.0])]
        w00 = 0.78 * b0[0] * 0.9 * b1[0]
        w01 = 0.78 * b0[0] * 0.1 * b1[1]
        w10 = 0.22 * b0[1] * 0.3 * b1[0]
        w11 = 0.22 * b0[1] * 0.7 * b1[1]
        total = w00 + w01 + w10 + w11
        self.assert_row_close(result.prefix_log_likelihoods,
                              [math.log(0.78 * b0[0] + 0.22 * b0[1]), math.log(total)])
        self.assert_row_close(result.filtered_probabilities[1],
                              [(w00 + w10) / total, (w01 + w11) / total])
        self.assertEqual(result.enumerated_paths, 6)
        self.assertEqual(result.negative_log_likelihood, -result.log_likelihood)
        self.assert_normalized(result)

    def test_time_indexed_transitions_and_means_analytic(self):
        result = self.run_reference(
            [[0.5], [1.5]], transition=None,
            transitions_by_time=[[[0.9, 0.1], [0.3, 0.7]], [[0.2, 0.8], [0.6, 0.4]]],
            means=None, means_by_time=[[[0.0], [2.0]], [[1.0], [-1.0]]],
        )
        a = 0.78 * density([0.5], [0.0], [1.0])
        b = 0.22 * density([0.5], [2.0], [4.0])
        # Explicit sums of paths ending in each state, not the oracle's helpers.
        end0 = (a * 0.2 + b * 0.6) * density([1.5], [1.0], [1.0])
        end1 = (a * 0.8 + b * 0.4) * density([1.5], [-1.0], [4.0])
        self.assertAlmostEqual(result.log_likelihood, math.log(end0 + end1), delta=1e-14)
        self.assert_row_close(result.filtered_probabilities[1],
                              [end0 / (end0 + end1), end1 / (end0 + end1)])

    def test_constant_and_repeated_time_parameters_agree(self):
        observations = [[0.1], [1.2], [-0.3]]
        constant = self.run_reference(observations)
        indexed = self.run_reference(
            observations, transition=None,
            transitions_by_time=[[[0.9, 0.1], [0.3, 0.7]]] * 3,
            means=None, means_by_time=[[[0.0], [2.0]]] * 3,
        )
        self.assertEqual(constant, indexed)

    def test_prefix_filtering_does_not_use_future_observations(self):
        a = self.run_reference([[0.5], [-20.0], [30.0]])
        b = self.run_reference([[0.5], [20.0], [-30.0]])
        prefix = self.run_reference([[0.5]])
        self.assertEqual(a.filtered_probabilities[0], b.filtered_probabilities[0])
        self.assertEqual(a.filtered_log_probabilities[0], prefix.filtered_log_probabilities[0])
        self.assertEqual(a.prefix_log_likelihoods[0], prefix.log_likelihood)
        self.assertNotEqual(a.filtered_probabilities[-1], b.filtered_probabilities[-1])

    def test_structural_zeros_keep_unreachable_state_impossible(self):
        result = self.run_reference(
            [[0.0], [0.0]], initial_belief=[1.0, 0.0],
            transition=[[1.0, 0.0], [0.0, 1.0]], means=[[10.0], [0.0]],
            variances=[[1.0], [1.0]],
        )
        self.assertEqual(result.filtered_probabilities, ((1.0, 0.0), (1.0, 0.0)))
        self.assertEqual(result.filtered_log_probabilities, ((0.0, -math.inf),) * 2)
        self.assertAlmostEqual(result.log_likelihood,
                               -100.0 - math.log(2.0 * math.pi), delta=1e-14)

    def test_deterministic_switch_includes_first_transition(self):
        result = self.run_reference(
            [[0.0]] * 3, initial_belief=[1.0, 0.0],
            transition=[[0.0, 1.0], [1.0, 0.0]],
        )
        self.assertEqual(result.filtered_probabilities, ((0.0, 1.0), (1.0, 0.0), (0.0, 1.0)))

    def test_identical_emissions_only_advance_markov_belief(self):
        result = self.run_reference([[0.1], [0.4]], means=[[0.0], [0.0]],
                                    variances=[[1.0], [1.0]])
        self.assert_row_close(result.filtered_probabilities[0], [0.78, 0.22])
        self.assert_row_close(result.filtered_probabilities[1], [0.768, 0.232])
        self.assertAlmostEqual(result.log_likelihood,
                               math.log(density([0.1], [0.0], [1.0]))
                               + math.log(density([0.4], [0.0], [1.0])))

    def test_separated_emissions_keep_finite_log_posterior_when_probability_underflows(self):
        result = self.run_reference([[1000.0]], means=[[0.0], [1000.0]],
                                    variances=[[1.0], [1.0]])
        self.assertEqual(result.filtered_probabilities[0], (0.0, 1.0))
        self.assertTrue(math.isfinite(result.filtered_log_probabilities[0][0]))
        self.assertLess(result.filtered_log_probabilities[0][0], -499_000.0)
        self.assertAlmostEqual(result.log_likelihood,
                               math.log(0.22) - 0.5 * math.log(2.0 * math.pi), delta=1e-14)

    def test_underflow_of_all_ordinary_densities_still_has_valid_likelihood(self):
        self.assertEqual(density([50.0], [0.0], [1.0]), 0.0)
        self.assertEqual(density([50.0], [1.0], [1.0]), 0.0)
        result = self.run_reference([[50.0]], means=[[0.0], [1.0]],
                                    variances=[[1.0], [1.0]])
        score0 = math.log(0.78) - 0.5 * (math.log(2.0 * math.pi) + 2500.0)
        score1 = math.log(0.22) - 0.5 * (math.log(2.0 * math.pi) + 2401.0)
        expected = score1 + math.log1p(math.exp(score0 - score1))
        self.assertAlmostEqual(result.log_likelihood, expected, delta=3e-13)
        self.assert_row_close(result.filtered_log_probabilities[0],
                              [score0 - expected, score1 - expected])
        self.assert_normalized(result)

    def test_extreme_common_tail_preserves_small_nonuniform_path_prior(self):
        result = self.run_reference(
            [[1e150]] * 2, transition=[[1.0, 0.0], [0.0, 1.0]],
            means=[[0.0], [0.0]], variances=[[1.0], [1.0]],
        )
        self.assert_row_close(result.filtered_probabilities[0], [0.8, 0.2], tolerance=1e-15)
        self.assert_row_close(result.filtered_probabilities[1], [0.8, 0.2], tolerance=1e-15)
        self.assertTrue(math.isfinite(result.log_likelihood))
        self.assert_normalized(result)

    def test_narrow_variance_can_make_negative_nll(self):
        result = self.run_reference([[0.0]], means=[[0.0], [0.0]],
                                    variances=[[1e-12], [1e-12]])
        expected = -0.5 * math.log(2.0 * math.pi * 1e-12)
        self.assertAlmostEqual(result.log_likelihood, expected, delta=1e-14)
        self.assertLess(result.negative_log_likelihood, 0.0)

    def test_subnormal_variance_has_no_floor_and_residual_is_scaled_before_squaring(self):
        variance = math.ulp(0.0)
        zero = diagonal_gaussian_log_density([0.0], [0.0], [variance])
        value = diagonal_gaussian_log_density([math.sqrt(variance)], [0.0], [variance])
        self.assertAlmostEqual(zero, -0.5 * (math.log(2.0 * math.pi) + math.log(variance)))
        self.assertAlmostEqual(zero - value, 0.5, delta=1e-14)

    def test_three_dimensions_use_diagonal_product_and_unequal_variances(self):
        observation = [0.2, -0.5, 1.0]
        means = [[0.0, 1.0, -1.0], [1.0, 0.0, 2.0]]
        variances = [[1.0, 4.0, 0.25], [0.5, 2.0, 3.0]]
        result = self.run_reference([observation], means=means, variances=variances)
        weights = [0.78 * density(observation, means[0], variances[0]),
                   0.22 * density(observation, means[1], variances[1])]
        self.assertAlmostEqual(result.log_likelihood, math.log(sum(weights)), delta=1e-14)
        self.assert_row_close(result.filtered_probabilities[0], [x / sum(weights) for x in weights])

    def test_state_permutation_preserves_all_likelihoods_and_permutes_posteriors(self):
        transitions = [[[0.9, 0.1], [0.3, 0.7]], [[0.2, 0.8], [0.6, 0.4]],
                       [[0.7, 0.3], [0.4, 0.6]]]
        means = [[[0.0], [2.0]], [[1.0], [-1.0]], [[-0.5], [1.5]]]
        options = {"transition": None, "transitions_by_time": transitions,
                   "means": None, "means_by_time": means}
        original = self.run_reference([[0.5], [1.5], [-0.2]], **options)
        permuted = self.run_reference(
            [[0.5], [1.5], [-0.2]], initial_belief=[0.2, 0.8], variances=[[4.0], [1.0]],
            transition=None,
            transitions_by_time=[[[p[j][k] for k in (1, 0)] for j in (1, 0)] for p in transitions],
            means=None, means_by_time=[list(reversed(row)) for row in means],
        )
        self.assert_row_close(permuted.prefix_log_likelihoods, original.prefix_log_likelihoods)
        for a, b in zip(permuted.filtered_log_probabilities, original.filtered_log_probabilities):
            self.assert_row_close(a, tuple(reversed(b)))

    def test_inactive_padding_is_not_evaluated(self):
        expected = self.run_reference([[0.5]])
        padded = self.run_reference(
            [[0.5], [math.nan], object()], length=1, transition=None,
            transitions_by_time=[[[0.9, 0.1], [0.3, 0.7]], [[math.inf]], None],
            means=None, means_by_time=[[[0.0], [2.0]], [[math.nan]], object()],
        )
        self.assertEqual(padded, expected)

    def test_missing_emission_advances_state_and_does_not_validate_ignored_row(self):
        result = self.run_reference(
            [[0.5], [math.nan]], emission_mask=[True, False],
            means=None, means_by_time=[[[0.0], [2.0]], object()],
        )
        first = self.run_reference([[0.5]])
        a, b = first.filtered_probabilities[0]
        self.assertAlmostEqual(result.log_likelihood, first.log_likelihood, delta=1e-15)
        self.assert_row_close(result.filtered_probabilities[1], [a * 0.9 + b * 0.3, a * 0.1 + b * 0.7])

    def test_missing_only_sequence_is_distinct_from_empty_sequence(self):
        result = self.run_reference([None, object()], emission_mask=[False, False])
        self.assertAlmostEqual(result.log_likelihood, 0.0, delta=1e-15)
        self.assert_row_close(result.filtered_probabilities[0], [0.78, 0.22])
        self.assert_row_close(result.filtered_probabilities[1], [0.768, 0.232])
        self.assertEqual(result.enumerated_paths, 6)

    def test_zero_length_explicit_neutral_likelihood_and_no_posterior(self):
        empty = self.run_reference([])
        padded = self.run_reference(
            [object(), [math.nan]], length=0, transition=None,
            transitions_by_time=[None, object()], means=None, means_by_time=[None, object()],
        )
        self.assertEqual(empty, padded)
        self.assertEqual(empty.prefix_log_likelihoods, ())
        self.assertEqual(empty.filtered_probabilities, ())
        self.assertEqual(empty.filtered_log_probabilities, ())
        self.assertEqual(empty.log_likelihood, 0.0)
        self.assertEqual(empty.negative_log_likelihood, 0.0)
        self.assertEqual(empty.enumerated_paths, 0)
        # Static model distributions/variances must still be valid.
        with self.assertRaises(ValueError):
            self.run_reference([], initial_belief=[0.4, 0.4])
        with self.assertRaises(ValueError):
            self.run_reference([], variances=[[0.0], [1.0]])

    def test_nonfinite_active_values_refused_even_in_unreachable_state(self):
        for bad in (math.nan, math.inf, -math.inf):
            for changes in ({"means": [[0.0], [bad]]}, {"variances": [[1.0], [bad]]},
                            {"transition": [[1.0, 0.0], [bad, 1.0]]},
                            {"initial_belief": [bad, 0.0]}):
                with self.subTest(bad=bad, changes=changes), self.assertRaises(ValueError):
                    self.run_reference([[0.0]], **changes)
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.run_reference([[bad]])
            with self.assertRaises(ValueError):
                self.run_reference([[0.0]], initial_belief=[1.0, 0.0],
                                   transition=[[1.0, 0.0], [0.0, 1.0]], means=[[0.0], [bad]])

    def test_missing_emission_still_validates_transition(self):
        with self.assertRaises(ValueError):
            self.run_reference([None], emission_mask=[False], transition=None,
                               transitions_by_time=[[[math.nan, 0.0], [0.0, 1.0]]])

    def test_invalid_probability_distributions_are_not_repaired(self):
        for row in ([0.0, 0.0], [0.4, 0.4], [0.6, 0.6], [-0.1, 1.1],
                    [1.0, 1.0], [True, False], ["0.8", 0.2]):
            with self.subTest(row=row), self.assertRaises(ValueError):
                self.run_reference([[0.0]], initial_belief=row)
            with self.subTest(row=row), self.assertRaises(ValueError):
                self.run_reference([[0.0]], transition=[row, [0.0, 1.0]])

    def test_tolerance_explicit_and_accepted_probabilities_not_renormalized(self):
        self.assertEqual(DEFAULT_PROBABILITY_TOLERANCE, 1e-12)
        initial = [0.8, 0.2000000000005]
        before = copy.deepcopy(initial)
        result = self.run_reference(
            [[0.0]], initial_belief=initial, transition=[[1.0, 0.0], [0.0, 1.0]],
            means=[[0.0], [0.0]], variances=[[1.0], [1.0]],
        )
        expected = -0.5 * math.log(2.0 * math.pi) + math.log(math.fsum(initial))
        self.assertAlmostEqual(result.log_likelihood, expected, delta=3e-16)
        self.assertGreater(result.log_likelihood, -0.5 * math.log(2.0 * math.pi))
        self.assertEqual(initial, before)
        with self.assertRaises(ValueError):
            self.run_reference([[0.0]], initial_belief=initial, probability_tolerance=0.0)
        for tolerance in (-1.0, 1e-7, math.nan, True):
            with self.subTest(tolerance=tolerance), self.assertRaises(ValueError):
                self.run_reference([[0.0]], probability_tolerance=tolerance)

    def test_positive_variances_required_without_floor_or_logvar_conversion(self):
        for variance in (0.0, -1.0, math.nan, math.inf, True, "1", 10**400):
            with self.subTest(variance=variance), self.assertRaises(ValueError):
                self.run_reference([[0.0]], variances=[[variance], [1.0]])
            with self.subTest(variance=variance), self.assertRaises(ValueError):
                diagonal_gaussian_log_density([0.0], [0.0], [variance])

    def test_overflow_of_finite_active_arithmetic_is_explicit_refusal(self):
        with self.assertRaises(FloatingPointError):
            self.run_reference([[1e308]], means=[[-1e308], [0.0]])
        with self.assertRaises(FloatingPointError):
            diagonal_gaussian_log_density([1e200], [0.0], [1.0])
        with self.assertRaises(FloatingPointError):
            self.run_reference([[1e154]] * 8, means=[[0.0], [0.0]], variances=[[1.0], [1.0]])

    def test_shapes_and_ambiguous_parameter_representations_refused(self):
        for changes in (
            {"means": [[0.0]]}, {"variances": [[1.0], [1.0, 2.0]]},
            {"transition": [[0.9], [0.3, 0.7]]}, {"transition": None},
            {"transitions_by_time": [[[0.9, 0.1], [0.3, 0.7]]]},
            {"means": None}, {"means_by_time": [[[0.0], [2.0]]]},
            {"transition": None, "transitions_by_time": []},
            {"means": None, "means_by_time": []},
            {"emission_mask": []}, {"emission_mask": [1]},
            {"length": True}, {"length": -1}, {"length": 2}, {"length": 0.5},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.run_reference([[0.0]], **changes)
        for observation in ([[]], [[0.0, 1.0]], [[True]], [["0"]], [None]):
            with self.subTest(observation=observation), self.assertRaises(ValueError):
                self.run_reference(observation)
        with self.assertRaises(ValueError):
            diagonal_gaussian_log_density([], [], [])

    def test_input_model_and_observations_are_not_mutated(self):
        observations = [[0.5], [1.5]]
        model = {"initial_belief": [0.8, 0.2], "transition": [[0.9, 0.1], [0.3, 0.7]],
                 "means": [[0.0], [2.0]], "variances": [[1.0], [4.0]]}
        snapshot = copy.deepcopy((observations, model))
        enumerate_gaussian_hmm(observations, **model)
        self.assertEqual((observations, model), snapshot)

    def test_t8_budget_counts_all_prefix_paths_before_enumeration(self):
        self.assertEqual(MAX_ENUMERATED_PATHS, 100_000)
        result = self.run_reference([[0.0]] * 8, max_paths=510)
        self.assertEqual(result.enumerated_paths, 510)
        self.assertEqual(len(result.prefix_log_likelihoods), 8)
        self.assert_normalized(result)
        with patch("hmm_reference.product", side_effect=AssertionError("enumeration began")):
            with self.assertRaisesRegex(ValueError, "510 paths > 509"):
                self.run_reference([[0.0]] * 8, max_paths=509)
            with self.assertRaisesRegex(ValueError, "87380 paths > 1000"):
                self.run_reference([[0.0]] * 8, initial_belief=[0.25] * 4, max_paths=1000)
        for budget in (0, -1, True, 1.5, MAX_ENUMERATED_PATHS + 1):
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                self.run_reference([[0.0]], max_paths=budget)

    def test_hard_state_time_and_dimension_bounds(self):
        for changes in ({"initial_belief": []}, {"initial_belief": [0.2] * 5},
                        {"variances": [[], []]}, {"variances": [[1.0] * 17] * 2}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.run_reference([[0.0]], **changes)
        with self.assertRaises(ValueError):
            self.run_reference([[0.0]] * 9)
        # Padding cannot evade the stored-step admission limit.
        with self.assertRaises(ValueError):
            self.run_reference([None] * 9, length=0)

    def test_small_generic_k_and_single_state_cases(self):
        for states in (1, 3, 4):
            with self.subTest(states=states):
                initial = [1.0 / states] * states
                identity = [[float(j == k) for k in range(states)] for j in range(states)]
                result = enumerate_gaussian_hmm(
                    [[0.0], [0.0]], initial_belief=initial, transition=identity,
                    means=[[0.0]] * states, variances=[[1.0]] * states,
                )
                self.assertAlmostEqual(result.log_likelihood, -math.log(2.0 * math.pi))
                self.assert_row_close(result.filtered_probabilities[1], initial)
                self.assertEqual(result.enumerated_paths, states + states**2)


if __name__ == "__main__":
    unittest.main()
