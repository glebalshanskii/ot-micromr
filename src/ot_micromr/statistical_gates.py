from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.stats import norm
from scipy.stats import t as student_t


@dataclass(frozen=True, slots=True)
class EquivalenceResult:
    estimate: float
    standard_error: float
    degrees_of_freedom: float
    target: float
    margin: float
    alpha: float
    tost_interval_lower: float
    tost_interval_upper: float
    compatibility_interval_lower: float
    compatibility_interval_upper: float
    p_lower: float
    p_upper: float
    p_equivalence: float
    p_difference: float
    status: str


@dataclass(frozen=True, slots=True)
class SuperiorityResult:
    estimate: float
    standard_error: float
    degrees_of_freedom: float
    minimum_effect: float
    alpha: float
    lower_confidence_bound: float
    upper_confidence_bound: float
    p_superiority: float
    p_opposite: float
    status: str


def _validate_alpha(alpha: float) -> float:
    value = float(alpha)
    if not math.isfinite(value) or not 0.0 < value < 0.5:
        raise ValueError("alpha must be finite and lie in (0, 0.5)")
    return value


def _sample(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size < 2 or not np.all(np.isfinite(array)):
        raise ValueError("statistical gate requires at least two finite observations")
    return array


def _equivalence_from_estimate(
    estimate: float,
    standard_error: float,
    degrees_of_freedom: float,
    *,
    target: float,
    margin: float,
    alpha: float,
) -> EquivalenceResult:
    alpha = _validate_alpha(alpha)
    target = float(target)
    margin = float(margin)
    if not math.isfinite(target):
        raise ValueError("target must be finite")
    if not math.isfinite(margin) or margin <= 0.0:
        raise ValueError("equivalence margin must be finite and positive")
    if not math.isfinite(standard_error) or standard_error < 0.0:
        raise ValueError("standard error must be finite and nonnegative")
    if not math.isfinite(degrees_of_freedom) or degrees_of_freedom <= 0.0:
        raise ValueError("degrees of freedom must be finite and positive")

    if standard_error == 0.0:
        inside = abs(estimate - target) < margin
        outside = abs(estimate - target) > margin
        difference = estimate != target
        p_lower = 0.0 if estimate > target - margin else 1.0
        p_upper = 0.0 if estimate < target + margin else 1.0
        p_equivalence = max(p_lower, p_upper)
        p_difference = 0.0 if difference else 1.0
        status = (
            "equivalent"
            if inside
            else "meaningfully_different"
            if outside
            else "inconclusive"
        )
        return EquivalenceResult(
            estimate=estimate,
            standard_error=standard_error,
            degrees_of_freedom=degrees_of_freedom,
            target=target,
            margin=margin,
            alpha=alpha,
            tost_interval_lower=estimate,
            tost_interval_upper=estimate,
            compatibility_interval_lower=estimate,
            compatibility_interval_upper=estimate,
            p_lower=p_lower,
            p_upper=p_upper,
            p_equivalence=p_equivalence,
            p_difference=p_difference,
            status=status,
        )

    t_lower = (estimate - (target - margin)) / standard_error
    t_upper = (estimate - (target + margin)) / standard_error
    p_lower = float(student_t.sf(t_lower, degrees_of_freedom))
    p_upper = float(student_t.cdf(t_upper, degrees_of_freedom))
    p_equivalence = max(p_lower, p_upper)
    t_difference = (estimate - target) / standard_error
    p_difference = float(2.0 * student_t.sf(abs(t_difference), degrees_of_freedom))

    tost_critical = float(student_t.ppf(1.0 - alpha, degrees_of_freedom))
    compatibility_critical = float(student_t.ppf(1.0 - alpha / 2.0, degrees_of_freedom))
    tost_lower = estimate - tost_critical * standard_error
    tost_upper = estimate + tost_critical * standard_error
    compatibility_lower = estimate - compatibility_critical * standard_error
    compatibility_upper = estimate + compatibility_critical * standard_error
    equivalent = p_equivalence < alpha
    meaningfully_different = (
        compatibility_lower > target + margin
        or compatibility_upper < target - margin
    )
    status = (
        "equivalent"
        if equivalent
        else "meaningfully_different"
        if meaningfully_different
        else "inconclusive"
    )
    return EquivalenceResult(
        estimate=estimate,
        standard_error=standard_error,
        degrees_of_freedom=degrees_of_freedom,
        target=target,
        margin=margin,
        alpha=alpha,
        tost_interval_lower=tost_lower,
        tost_interval_upper=tost_upper,
        compatibility_interval_lower=compatibility_lower,
        compatibility_interval_upper=compatibility_upper,
        p_lower=p_lower,
        p_upper=p_upper,
        p_equivalence=p_equivalence,
        p_difference=p_difference,
        status=status,
    )


def one_sample_equivalence(
    values: Sequence[float],
    *,
    target: float,
    margin: float,
    alpha: float = 0.05,
) -> EquivalenceResult:
    array = _sample(values)
    estimate = float(np.mean(array))
    standard_error = float(np.std(array, ddof=1) / math.sqrt(array.size))
    return _equivalence_from_estimate(
        estimate,
        standard_error,
        float(array.size - 1),
        target=target,
        margin=margin,
        alpha=alpha,
    )


def paired_equivalence(
    primary: Sequence[float],
    reference: Sequence[float],
    *,
    margin: float,
    alpha: float = 0.05,
) -> EquivalenceResult:
    primary_array = _sample(primary)
    reference_array = _sample(reference)
    if primary_array.shape != reference_array.shape:
        raise ValueError("paired equivalence requires equal-length inputs")
    return one_sample_equivalence(
        primary_array - reference_array,
        target=0.0,
        margin=margin,
        alpha=alpha,
    )


def independent_equivalence(
    primary: Sequence[float],
    reference: Sequence[float],
    *,
    margin: float,
    alpha: float = 0.05,
) -> EquivalenceResult:
    primary_array = _sample(primary)
    reference_array = _sample(reference)
    primary_variance = float(np.var(primary_array, ddof=1))
    reference_variance = float(np.var(reference_array, ddof=1))
    primary_component = primary_variance / primary_array.size
    reference_component = reference_variance / reference_array.size
    standard_error_squared = primary_component + reference_component
    standard_error = math.sqrt(standard_error_squared)
    if standard_error_squared == 0.0:
        degrees_of_freedom = float(primary_array.size + reference_array.size - 2)
    else:
        denominator = (
            primary_component**2 / (primary_array.size - 1)
            + reference_component**2 / (reference_array.size - 1)
        )
        degrees_of_freedom = standard_error_squared**2 / denominator
    return _equivalence_from_estimate(
        float(np.mean(primary_array) - np.mean(reference_array)),
        standard_error,
        degrees_of_freedom,
        target=0.0,
        margin=margin,
        alpha=alpha,
    )


def one_sample_superiority(
    values: Sequence[float],
    *,
    minimum_effect: float,
    alpha: float = 0.05,
) -> SuperiorityResult:
    alpha = _validate_alpha(alpha)
    minimum_effect = float(minimum_effect)
    if not math.isfinite(minimum_effect):
        raise ValueError("minimum effect must be finite")
    array = _sample(values)
    estimate = float(np.mean(array))
    standard_error = float(np.std(array, ddof=1) / math.sqrt(array.size))
    degrees_of_freedom = float(array.size - 1)
    if standard_error == 0.0:
        p_superiority = 0.0 if estimate > minimum_effect else 1.0
        p_opposite = 0.0 if estimate < minimum_effect else 1.0
        lower = estimate
        upper = estimate
    else:
        statistic = (estimate - minimum_effect) / standard_error
        p_superiority = float(student_t.sf(statistic, degrees_of_freedom))
        p_opposite = float(student_t.cdf(statistic, degrees_of_freedom))
        critical = float(student_t.ppf(1.0 - alpha, degrees_of_freedom))
        lower = estimate - critical * standard_error
        upper = estimate + critical * standard_error
    status = (
        "superior"
        if p_superiority < alpha
        else "below_minimum"
        if p_opposite < alpha
        else "inconclusive"
    )
    return SuperiorityResult(
        estimate=estimate,
        standard_error=standard_error,
        degrees_of_freedom=degrees_of_freedom,
        minimum_effect=minimum_effect,
        alpha=alpha,
        lower_confidence_bound=lower,
        upper_confidence_bound=upper,
        p_superiority=p_superiority,
        p_opposite=p_opposite,
        status=status,
    )


def holm_adjust(p_values: Sequence[float]) -> tuple[float, ...]:
    array = np.asarray(p_values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("Holm adjustment requires a non-empty finite p-value vector")
    if np.any((array < 0.0) | (array > 1.0)):
        raise ValueError("p-values must lie in [0, 1]")
    order = np.argsort(array, kind="stable")
    adjusted_sorted = np.empty(array.size, dtype=np.float64)
    running_maximum = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (array.size - rank) * float(array[index]))
        running_maximum = max(running_maximum, candidate)
        adjusted_sorted[rank] = running_maximum
    adjusted = np.empty(array.size, dtype=np.float64)
    for rank, index in enumerate(order):
        adjusted[index] = adjusted_sorted[rank]
    return tuple(float(value) for value in adjusted)


def normal_approximation_sample_size(
    *,
    standard_deviation: float,
    distance_to_null: float,
    alpha: float = 0.05,
    power: float = 0.90,
) -> int:
    alpha = _validate_alpha(alpha)
    standard_deviation = float(standard_deviation)
    distance_to_null = float(distance_to_null)
    power = float(power)
    if not math.isfinite(standard_deviation) or standard_deviation <= 0.0:
        raise ValueError("standard deviation must be finite and positive")
    if not math.isfinite(distance_to_null) or distance_to_null <= 0.0:
        raise ValueError("distance to null must be finite and positive")
    if not math.isfinite(power) or not 0.5 < power < 1.0:
        raise ValueError("power must lie in (0.5, 1)")
    critical_sum = float(norm.ppf(1.0 - alpha) + norm.ppf(power))
    return max(2, math.ceil((critical_sum * standard_deviation / distance_to_null) ** 2))
