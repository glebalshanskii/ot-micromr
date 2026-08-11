import unittest

from ot_micromr.statistical_gates import (
    holm_adjust,
    independent_equivalence,
    normal_approximation_sample_size,
    one_sample_equivalence,
    one_sample_superiority,
    paired_equivalence,
)


class StatisticalGateTests(unittest.TestCase):
    def test_equivalence_requires_precision_not_only_centering(self) -> None:
        precise = one_sample_equivalence(
            [-0.006, -0.004, -0.002, 0.0, 0.002, 0.004, 0.006],
            target=0.0,
            margin=0.02,
        )
        noisy = one_sample_equivalence(
            [-0.20, -0.10, -0.05, 0.0, 0.05, 0.10, 0.20],
            target=0.0,
            margin=0.02,
        )
        self.assertEqual(precise.status, "equivalent")
        self.assertEqual(noisy.status, "inconclusive")
        self.assertAlmostEqual(noisy.estimate, 0.0)
        self.assertGreater(noisy.p_equivalence, 0.05)

    def test_equivalence_can_detect_meaningful_difference(self) -> None:
        result = one_sample_equivalence(
            [0.095, 0.10, 0.105, 0.11, 0.09, 0.102],
            target=0.0,
            margin=0.02,
        )
        self.assertEqual(result.status, "meaningfully_different")
        self.assertLess(result.p_difference, 0.05)

    def test_superiority_tests_minimum_effect(self) -> None:
        result = one_sample_superiority(
            [0.18, 0.20, 0.22, 0.19, 0.21, 0.23],
            minimum_effect=0.05,
        )
        self.assertEqual(result.status, "superior")
        self.assertGreater(result.lower_confidence_bound, 0.05)
        self.assertLess(result.p_superiority, 0.05)

    def test_superiority_distinguishes_below_minimum_from_inconclusive(self) -> None:
        below = one_sample_superiority(
            [0.008, 0.010, 0.012, 0.009, 0.011, 0.010],
            minimum_effect=0.05,
        )
        noisy = one_sample_superiority(
            [-0.10, -0.05, 0.0, 0.05, 0.10],
            minimum_effect=0.05,
        )
        self.assertEqual(below.status, "below_minimum")
        self.assertEqual(noisy.status, "inconclusive")

    def test_refinement_supports_paired_and_independent_designs(self) -> None:
        primary = [1.001, 0.999, 1.002, 0.998, 1.000]
        reference = [1.000, 1.000, 1.001, 0.999, 1.000]
        paired = paired_equivalence(primary, reference, margin=0.01)
        independent = independent_equivalence(primary, reference, margin=0.01)
        self.assertEqual(paired.status, "equivalent")
        self.assertEqual(independent.status, "equivalent")

    def test_holm_adjustment_is_monotone_in_sorted_order(self) -> None:
        adjusted = holm_adjust([0.01, 0.04, 0.03])
        self.assertEqual(adjusted, (0.03, 0.06, 0.06))

    def test_power_approximation_increases_for_smaller_effect(self) -> None:
        large_effect = normal_approximation_sample_size(
            standard_deviation=1.0,
            distance_to_null=0.5,
        )
        small_effect = normal_approximation_sample_size(
            standard_deviation=1.0,
            distance_to_null=0.25,
        )
        self.assertGreater(small_effect, large_effect)

    def test_invalid_inputs_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least two"):
            one_sample_equivalence([0.0], target=0.0, margin=0.1)
        with self.assertRaisesRegex(ValueError, "margin"):
            one_sample_equivalence([0.0, 0.1], target=0.0, margin=0.0)
        with self.assertRaisesRegex(ValueError, "p-values"):
            holm_adjust([0.1, 1.1])


if __name__ == "__main__":
    unittest.main()
