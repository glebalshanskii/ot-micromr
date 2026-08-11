from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

from ot_micromr.errors import ExperimentError


EVENT_CHANNELS = (
    "slide_up",
    "slide_down",
    "open_up",
    "open_down",
    "close_up",
    "close_down",
)

EVENT_HALF_TICK_DELTAS = {
    "slide_up": 2,
    "slide_down": -2,
    "open_up": 1,
    "open_down": -1,
    "close_up": 1,
    "close_down": -1,
}


class InvariantViolation(ExperimentError):
    """Raised immediately when a pathwise simulator invariant is violated."""


@dataclass(frozen=True, slots=True)
class BookParameters:
    delta_price: float
    sigma_x_price_per_sqrt_second: float
    mu_s_per_second: float
    mu_o_per_second: float
    mu_c_per_second: float
    alpha_s_per_second: float
    alpha_o_per_second: float
    alpha_c_per_second: float
    tight_drift_coefficient_per_second: float
    open_drift_coefficient_per_second: float

    @classmethod
    def from_model(cls, model: Mapping[str, Any]) -> BookParameters:
        alpha_s = float(model["alpha_s_per_second"])
        alpha_o = float(model["alpha_o_per_second"])
        alpha_c = float(model["alpha_c_per_second"])
        return cls(
            delta_price=float(model["delta_price"]),
            sigma_x_price_per_sqrt_second=float(model["sigma_x_price_per_sqrt_second"]),
            mu_s_per_second=float(model["mu_s_per_second"]),
            mu_o_per_second=float(model["mu_o_per_second"]),
            mu_c_per_second=float(model["mu_c_per_second"]),
            alpha_s_per_second=alpha_s,
            alpha_o_per_second=alpha_o,
            alpha_c_per_second=alpha_c,
            tight_drift_coefficient_per_second=2.0 * alpha_s + alpha_o,
            open_drift_coefficient_per_second=alpha_c,
        )


@dataclass(slots=True)
class BookState:
    time_seconds: float
    efficient_price: float
    mid_half_ticks: int

    def mid_price(self, parameters: BookParameters) -> float:
        return self.mid_half_ticks * parameters.delta_price / 2.0

    def gap_price(self, parameters: BookParameters) -> float:
        return self.mid_price(parameters) - self.efficient_price

    @property
    def is_tight(self) -> bool:
        return self.mid_half_ticks % 2 != 0

    def spread_price(self, parameters: BookParameters) -> float:
        return parameters.delta_price if self.is_tight else 2.0 * parameters.delta_price

    def bid_price(self, parameters: BookParameters) -> float:
        return self.mid_price(parameters) - self.spread_price(parameters) / 2.0


@dataclass(frozen=True, slots=True)
class BookEventRecord:
    epsilon: float
    seed: int
    event_index: int
    time_seconds: float
    channel: str
    left_gap_price: float
    pre_event_gap_price: float
    post_event_gap_price: float
    pre_mid_half_ticks: int
    post_mid_half_ticks: int
    efficient_price: float
    delta_mid_price: float
    left_channel_intensity_per_second: float
    measured: bool


def initial_state(model: Mapping[str, Any]) -> BookState:
    initial = model["initial_state"]
    return BookState(
        time_seconds=float(initial["time_seconds"]),
        efficient_price=float(initial["efficient_price"]),
        mid_half_ticks=int(initial["mid_half_ticks"]),
    )


def intensities(
    state: BookState, parameters: BookParameters
) -> tuple[float, float, float, float, float, float]:
    gap = state.gap_price(parameters)
    positive = max(gap, 0.0)
    negative = max(-gap, 0.0)
    scale = 2.0 / parameters.delta_price
    if state.is_tight:
        values = (
            parameters.mu_s_per_second + scale * parameters.alpha_s_per_second * negative,
            parameters.mu_s_per_second + scale * parameters.alpha_s_per_second * positive,
            parameters.mu_o_per_second + scale * parameters.alpha_o_per_second * negative,
            parameters.mu_o_per_second + scale * parameters.alpha_o_per_second * positive,
            0.0,
            0.0,
        )
        inactive = values[4:]
    else:
        values = (
            0.0,
            0.0,
            0.0,
            0.0,
            parameters.mu_c_per_second + scale * parameters.alpha_c_per_second * negative,
            parameters.mu_c_per_second + scale * parameters.alpha_c_per_second * positive,
        )
        inactive = values[:4]
    if any(not math.isfinite(value) for value in values):
        raise InvariantViolation("nonfinite intensity")
    if any(value < 0.0 for value in values):
        raise InvariantViolation("negative active intensity")
    if any(value != 0.0 for value in inactive):
        raise InvariantViolation("inactive channel has nonzero intensity")
    return values


def generator_mid_drift(
    state: BookState,
    parameters: BookParameters,
    values: tuple[float, float, float, float, float, float] | None = None,
) -> float:
    rates = values if values is not None else intensities(state, parameters)
    delta = parameters.delta_price
    changes = (delta, -delta, delta / 2.0, -delta / 2.0, delta / 2.0, -delta / 2.0)
    return sum(rate * change for rate, change in zip(rates, changes, strict=True))


def expected_drift_coefficient(state: BookState, parameters: BookParameters) -> float:
    return (
        parameters.tight_drift_coefficient_per_second
        if state.is_tight
        else parameters.open_drift_coefficient_per_second
    )


def apply_book_event(state: BookState, channel: str, parameters: BookParameters) -> float:
    if channel not in EVENT_HALF_TICK_DELTAS:
        raise InvariantViolation(f"unknown event channel: {channel}")
    was_tight = state.is_tight
    if channel.startswith(("slide_", "open_")) and not was_tight:
        raise InvariantViolation(f"{channel} is inactive in open state")
    if channel.startswith("close_") and was_tight:
        raise InvariantViolation(f"{channel} is inactive in tight state")
    before_x = state.efficient_price
    before_ticks = state.mid_half_ticks
    half_tick_change = EVENT_HALF_TICK_DELTAS[channel]
    state.mid_half_ticks += half_tick_change
    if state.efficient_price != before_x:
        raise InvariantViolation("efficient price changed at book event")
    expected_tight = was_tight if channel.startswith("slide_") else not was_tight
    if state.is_tight != expected_tight:
        raise InvariantViolation("illegal parity transition")
    actual_ticks = state.mid_half_ticks - before_ticks
    if actual_ticks != half_tick_change:
        raise InvariantViolation("illegal mid-price jump")
    delta_mid = half_tick_change * parameters.delta_price / 2.0
    expected_magnitude = (
        parameters.delta_price if channel.startswith("slide_") else parameters.delta_price / 2.0
    )
    if abs(delta_mid) != expected_magnitude:
        raise InvariantViolation("book jump has an illegal magnitude")
    return delta_mid
