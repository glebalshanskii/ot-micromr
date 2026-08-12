import math
import unittest
from pathlib import Path

import torch

from ot_micromr.config import load_runspec
from ot_micromr.factorized_filter import (
    _category_conditionals,
    _histogram_density,
    causal_rolling_lognormal_parameters,
    conditional_direction_probabilities,
    lognormal_clock_terms,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class FactorizedFilterTests(unittest.TestCase):
    def test_config_loads_with_frozen_clock_contract(self) -> None:
        spec = load_runspec(
            REPOSITORY_ROOT / "cfg" / "experiments" / "emp_mark_fact_001.toml"
        )
        self.assertEqual(spec.experiment_id, "EMP-MARK-FACT-001")
        self.assertEqual(spec.values["model"]["clock_training_prior_events"], 50.0)

    def test_rolling_clock_excludes_current_duration(self) -> None:
        duration = torch.tensor((1.0, 2.0, 100.0), dtype=torch.float64)
        changed = torch.tensor((1.0, 2.0, 0.01), dtype=torch.float64)
        valid = torch.ones(3, dtype=torch.bool)
        arguments = (valid, torch.tensor(0.0), torch.tensor(1.0), 2, 1.0, 1e-4)
        mean, scale = causal_rolling_lognormal_parameters(duration, *arguments)
        changed_mean, changed_scale = causal_rolling_lognormal_parameters(
            changed, *arguments
        )
        self.assertEqual(float(mean[2]), float(changed_mean[2]))
        self.assertEqual(float(scale[2]), float(changed_scale[2]))
        self.assertEqual(float(mean[0]), 0.0)
        self.assertEqual(float(scale[0]), 1.0)

    def test_lognormal_rescaling_matches_analytic_survival(self) -> None:
        duration = torch.tensor((1.0, math.e), dtype=torch.float64)
        mean = torch.zeros(2, dtype=torch.float64)
        scale = torch.ones(2, dtype=torch.float64)
        log_density, rescaling, pit = lognormal_clock_terms(duration, mean, scale)
        self.assertTrue(torch.all(torch.isfinite(log_density)))
        self.assertAlmostEqual(float(pit[0]), 0.5, places=12)
        self.assertAlmostEqual(float(rescaling[0]), math.log(2.0), places=12)
        self.assertTrue(torch.allclose(pit, 1.0 - torch.exp(-rescaling)))

    def test_directional_tilt_is_normalized_and_does_not_change_total_clock(self) -> None:
        base = torch.tensor(((0.3, 0.2, 0.5),), dtype=torch.float32)
        gap = torch.tensor((2.0, -2.0, 0.0))
        spread = torch.zeros(3, dtype=torch.int64)
        probabilities = conditional_direction_probabilities(
            gap, spread, base, torch.tensor(1.0), 1.0
        )
        self.assertTrue(torch.allclose(probabilities.sum(dim=-1), torch.ones(3)))
        self.assertGreater(float(probabilities[0, 0]), float(probabilities[0, 2]))
        self.assertGreater(float(probabilities[1, 2]), float(probabilities[1, 0]))
        zero = conditional_direction_probabilities(
            gap, spread, base, torch.tensor(0.0), 1.0
        )
        self.assertTrue(torch.allclose(zero, base.expand_as(zero)))

    def test_category_conditionals_are_probabilities_for_every_direction(self) -> None:
        probabilities = torch.full((8, 729), 1.0 / 729.0)
        conditionals = _category_conditionals(probabilities)
        self.assertEqual(conditionals["family"].shape, (8, 3, 3))
        self.assertEqual(conditionals["midpoint_magnitude"].shape, (8, 3, 9))
        for values in conditionals.values():
            self.assertTrue(
                torch.allclose(values.sum(dim=-1), torch.ones_like(values[..., 0]))
            )

    def test_vectorized_histogram_density_integrates_to_one(self) -> None:
        values = torch.tensor((0.0, 0.5, 1.0, 1.5, 2.0), dtype=torch.float64)
        edges = torch.tensor((0.0, 1.0, 2.0), dtype=torch.float64)
        density = _histogram_density(values, edges)
        self.assertAlmostEqual(float(torch.sum(density * torch.diff(edges))), 1.0)
        self.assertTrue(
            torch.allclose(density, torch.tensor((0.4, 0.6), dtype=torch.float64))
        )


if __name__ == "__main__":
    unittest.main()
