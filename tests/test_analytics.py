import math
import unittest

import numpy as np
from scipy.special import erfi

from ot_micromr.analytics import (
    dawson_foc,
    dawson_foc_derivative,
    dimensional_surrogate_rate,
    direct_optimum_ratio,
    erfi_direct_integral,
    inclusive_grid,
    kramers_threshold_ratio,
    normalized_surrogate_rate,
    optimum_rate_identity,
    rate_loss_fraction,
    solve_dawson_optimum,
)
from ot_micromr.config import load_runspec


class AnalyticalFormulaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.spec = load_runspec("cfg/experiments/ana_smoke_001.toml")
        cls.numerics = cls.spec.values["numerics"]

    def test_dawson_checkpoint_and_residual(self) -> None:
        solution = solve_dawson_optimum(0.4, self.numerics)
        self.assertAlmostEqual(solution.u_d_ratio, 1.1558728538, delta=5e-9)
        self.assertLessEqual(solution.root_abs_residual, 1e-12)
        self.assertGreater(solution.u_d_ratio, 0.4)

    def test_direct_optimizer_agrees_with_root(self) -> None:
        solution = solve_dawson_optimum(0.4, self.numerics)
        direct = direct_optimum_ratio(0.4, self.numerics)
        self.assertAlmostEqual(solution.u_d_ratio, direct, delta=1e-7)

    def test_optimum_rate_identity(self) -> None:
        solution = solve_dawson_optimum(0.4, self.numerics)
        self.assertAlmostEqual(
            solution.normalized_rate,
            optimum_rate_identity(solution.u_d_ratio),
            delta=1e-12,
        )

    def test_erfi_matches_independent_quadrature(self) -> None:
        for value in (0.1, 0.8, 2.0):
            with self.subTest(value=value):
                self.assertAlmostEqual(erfi_direct_integral(value), float(erfi(value)), delta=2e-13)

    def test_foc_is_strictly_increasing_for_positive_u(self) -> None:
        values = np.linspace(0.01, 8.0, 500)
        derivatives = dawson_foc_derivative(values)
        self.assertTrue(np.all(derivatives > 0.0))
        solution = solve_dawson_optimum(0.4, self.numerics)
        self.assertLess(dawson_foc(0.4 + 1e-12, 0.4), 0.0)
        self.assertGreater(dawson_foc(solution.u_d_ratio + 0.1, 0.4), 0.0)

    def test_large_gamma_root_has_expected_asymptotic_scale(self) -> None:
        gamma = 5.0
        solution = solve_dawson_optimum(gamma, self.numerics)
        self.assertAlmostEqual(solution.u_d_ratio, gamma + 1.0 / gamma, delta=0.03)

    def test_myopic_boundary_has_zero_rate(self) -> None:
        for gamma in (0.05, 0.4, 1.7):
            self.assertEqual(normalized_surrogate_rate(gamma, gamma), 0.0)

    def test_surrogate_rate_decays_in_far_tail(self) -> None:
        for gamma in (0.25, 0.5, 1.0):
            rate_at_grid_end = normalized_surrogate_rate(3.0, gamma)
            rate_in_far_tail = normalized_surrogate_rate(8.0, gamma)
            self.assertLess(rate_in_far_tail, rate_at_grid_end)
            self.assertLess(rate_in_far_tail, 1e-12)

    def test_dimensional_scaling(self) -> None:
        normalized = normalized_surrogate_rate(1.2, 0.4)
        dimensional = dimensional_surrogate_rate(1.2, 0.4, 2.5, 3.0)
        self.assertAlmostEqual(dimensional, 7.5 * normalized, delta=1e-15)

    def test_kramers_and_rate_loss_checkpoint(self) -> None:
        solution = solve_dawson_optimum(0.4, self.numerics)
        u_star = kramers_threshold_ratio(0.4)
        self.assertAlmostEqual(u_star, 1.2198039027, delta=5e-9)
        self.assertAlmostEqual(
            rate_loss_fraction(u_star, solution.u_d_ratio, 0.4),
            0.0030007967,
            delta=5e-8,
        )

    def test_inclusive_grid_is_exact_at_endpoints(self) -> None:
        grid = inclusive_grid(0.05, 3.0, 0.01)
        self.assertEqual(len(grid), 296)
        self.assertEqual(grid[0], 0.05)
        self.assertEqual(grid[-1], 3.0)

    def test_invalid_domain_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            normalized_surrogate_rate(0.39, 0.4)
        with self.assertRaises(ValueError):
            inclusive_grid(0.0, 1.0, 0.3)
        with self.assertRaises(ValueError):
            solve_dawson_optimum(math.inf, self.numerics)


if __name__ == "__main__":
    unittest.main()
