import unittest

from ot_micromr.config import load_runspec
from ot_micromr.jump_model import (
    BookParameters,
    BookState,
    InvariantViolation,
    apply_book_event,
    generator_mid_drift,
    initial_state,
    intensities,
)


class JumpModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.balanced_spec = load_runspec("cfg/experiments/sim_moments_002.toml")
        cls.unbalanced_spec = load_runspec("cfg/experiments/sim_unbalanced_002.toml")
        cls.parameters = BookParameters.from_model(cls.balanced_spec.values["model"])

    def test_initial_state_derivations(self) -> None:
        state = initial_state(self.balanced_spec.values["model"])
        self.assertTrue(state.is_tight)
        self.assertEqual(state.mid_price(self.parameters), 0.5)
        self.assertEqual(state.bid_price(self.parameters), 0.0)
        self.assertEqual(state.spread_price(self.parameters), 1.0)
        self.assertEqual(state.gap_price(self.parameters), 0.0)

    def test_tight_intensities_and_balanced_drift(self) -> None:
        positive_gap = BookState(time_seconds=0.0, efficient_price=-0.5, mid_half_ticks=1)
        rates = intensities(positive_gap, self.parameters)
        self.assertEqual(rates, (1.0, 2.0, 0.01, 0.01, 0.0, 0.0))
        self.assertAlmostEqual(generator_mid_drift(positive_gap, self.parameters, rates), -1.0)

        negative_gap = BookState(time_seconds=0.0, efficient_price=1.5, mid_half_ticks=1)
        rates = intensities(negative_gap, self.parameters)
        self.assertEqual(rates, (2.0, 1.0, 0.01, 0.01, 0.0, 0.0))
        self.assertAlmostEqual(generator_mid_drift(negative_gap, self.parameters, rates), 1.0)

    def test_open_intensities_and_balanced_drift(self) -> None:
        state = BookState(time_seconds=0.0, efficient_price=-1.0, mid_half_ticks=0)
        rates = intensities(state, self.parameters)
        self.assertEqual(rates, (0.0, 0.0, 0.0, 0.0, 2.0, 4.0))
        self.assertAlmostEqual(generator_mid_drift(state, self.parameters, rates), -1.0)

    def test_unbalanced_open_drift(self) -> None:
        parameters = BookParameters.from_model(self.unbalanced_spec.values["model"])
        state = BookState(time_seconds=0.0, efficient_price=-1.0, mid_half_ticks=0)
        self.assertAlmostEqual(generator_mid_drift(state, parameters), -1.25)

    def test_legal_transitions_preserve_parity_contract(self) -> None:
        for channel, expected_ticks, expected_tight in (
            ("slide_up", 3, True),
            ("slide_down", -1, True),
            ("open_up", 2, False),
            ("open_down", 0, False),
        ):
            with self.subTest(channel=channel):
                state = BookState(0.0, 0.5, 1)
                apply_book_event(state, channel, self.parameters)
                self.assertEqual(state.mid_half_ticks, expected_ticks)
                self.assertEqual(state.is_tight, expected_tight)
                self.assertEqual(state.efficient_price, 0.5)
        for channel, expected_ticks in (("close_up", 1), ("close_down", -1)):
            state = BookState(0.0, 0.0, 0)
            apply_book_event(state, channel, self.parameters)
            self.assertEqual(state.mid_half_ticks, expected_ticks)
            self.assertTrue(state.is_tight)

    def test_illegal_transition_is_rejected(self) -> None:
        with self.assertRaises(InvariantViolation):
            apply_book_event(BookState(0.0, 0.0, 0), "slide_up", self.parameters)
        with self.assertRaises(InvariantViolation):
            apply_book_event(BookState(0.0, 0.5, 1), "close_up", self.parameters)


if __name__ == "__main__":
    unittest.main()
