from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq, minimize_scalar
from scipy.special import dawsn, erfi


@dataclass(frozen=True, slots=True)
class DawsonSolution:
    gamma_ratio: float
    u_d_ratio: float
    root_abs_residual: float
    normalized_rate: float
    identity_normalized_rate: float


def _finite_positive(name: str, value: float) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return result


def dawson_foc(u_ratio: float | np.ndarray, gamma_ratio: float) -> float | np.ndarray:
    """Equation B.1 in dimensionless variables."""

    gamma = _finite_positive("gamma_ratio", gamma_ratio)
    u = np.asarray(u_ratio, dtype=np.float64)
    if not np.all(np.isfinite(u)):
        raise ValueError("u_ratio must be finite")
    result = u - gamma - math.sqrt(2.0) * dawsn(u / math.sqrt(2.0))
    return float(result) if result.ndim == 0 else result


def dawson_foc_derivative(u_ratio: float | np.ndarray) -> float | np.ndarray:
    """Analytical derivative of the Dawson first-order condition for u > 0."""

    u = np.asarray(u_ratio, dtype=np.float64)
    if not np.all(np.isfinite(u)) or np.any(u <= 0.0):
        raise ValueError("u_ratio must be positive and finite")
    result = math.sqrt(2.0) * u * dawsn(u / math.sqrt(2.0))
    return float(result) if result.ndim == 0 else result


def erfi_direct_integral(value: float) -> float:
    """Independent quadrature diagnostic for the erfi passage-time term."""

    argument = float(value)
    if not math.isfinite(argument):
        raise ValueError("value must be finite")
    sign = -1.0 if argument < 0.0 else 1.0
    integral, _ = quad(
        lambda coordinate: math.exp(coordinate * coordinate),
        0.0,
        abs(argument),
        epsabs=1e-13,
        epsrel=1e-13,
        limit=200,
    )
    return sign * 2.0 * integral / math.sqrt(math.pi)


def kramers_threshold_ratio(gamma_ratio: float) -> float:
    gamma = _finite_positive("gamma_ratio", gamma_ratio)
    return 0.5 * (gamma + math.sqrt(gamma * gamma + 4.0))


def normalized_surrogate_rate(
    u_ratio: float | np.ndarray, gamma_ratio: float
) -> float | np.ndarray:
    """Return the paper's surrogate rate divided by alpha * s_G."""

    gamma = _finite_positive("gamma_ratio", gamma_ratio)
    u = np.asarray(u_ratio, dtype=np.float64)
    if not np.all(np.isfinite(u)):
        raise ValueError("u_ratio must be finite")
    if np.any(u < gamma):
        raise ValueError("u_ratio must be greater than or equal to gamma_ratio")
    result = (2.0 / math.pi) * (u - gamma) / erfi(u / math.sqrt(2.0))
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("surrogate rate produced a non-finite value")
    return float(result) if result.ndim == 0 else result


def dimensional_surrogate_rate(
    u_ratio: float | np.ndarray,
    gamma_ratio: float,
    alpha_per_second: float,
    s_g_price: float,
) -> float | np.ndarray:
    alpha = _finite_positive("alpha_per_second", alpha_per_second)
    s_g = _finite_positive("s_g_price", s_g_price)
    return alpha * s_g * normalized_surrogate_rate(u_ratio, gamma_ratio)


def optimum_rate_identity(u_d_ratio: float) -> float:
    u_d = _finite_positive("u_d_ratio", u_d_ratio)
    return math.sqrt(2.0 / math.pi) * math.exp(-0.5 * u_d * u_d)


def solve_dawson_optimum(gamma_ratio: float, numerics: Mapping[str, Any]) -> DawsonSolution:
    gamma = _finite_positive("gamma_ratio", gamma_ratio)
    lower = gamma + float(numerics["root_lower_margin_ratio"])
    upper = float(numerics["root_upper_u_ratio"])
    if upper <= lower:
        raise ValueError("root upper bound must exceed gamma plus lower margin")
    lower_value = dawson_foc(lower, gamma)
    upper_value = dawson_foc(upper, gamma)
    if not lower_value < 0.0 < upper_value:
        raise ValueError(
            f"Dawson root is not bracketed: F({lower})={lower_value}, F({upper})={upper_value}"
        )
    root = float(
        brentq(
            lambda value: dawson_foc(value, gamma),
            lower,
            upper,
            xtol=float(numerics["root_xtol"]),
            rtol=float(numerics["root_rtol"]),
            maxiter=int(numerics["root_max_iterations"]),
            disp=True,
        )
    )
    residual = abs(float(dawson_foc(root, gamma)))
    rate = float(normalized_surrogate_rate(root, gamma))
    identity = optimum_rate_identity(root)
    return DawsonSolution(
        gamma_ratio=gamma,
        u_d_ratio=root,
        root_abs_residual=residual,
        normalized_rate=rate,
        identity_normalized_rate=identity,
    )


def direct_optimum_ratio(
    gamma_ratio: float,
    numerics: Mapping[str, Any],
    *,
    lower_margin_ratio: float | None = None,
    upper_u_ratio: float | None = None,
) -> float:
    gamma = _finite_positive("gamma_ratio", gamma_ratio)
    margin = (
        float(lower_margin_ratio)
        if lower_margin_ratio is not None
        else float(numerics.get("optimizer_lower_margin_ratio", numerics["root_lower_margin_ratio"]))
    )
    upper = (
        float(upper_u_ratio)
        if upper_u_ratio is not None
        else float(numerics.get("optimizer_upper_u_ratio", numerics["root_upper_u_ratio"]))
    )
    if margin <= 0.0 or upper <= gamma + margin:
        raise ValueError("invalid direct-optimizer bounds")
    result = minimize_scalar(
        lambda value: -float(normalized_surrogate_rate(value, gamma)),
        bounds=(gamma + margin, upper),
        method="bounded",
        options={
            "xatol": float(numerics["optimizer_xatol"]),
            "maxiter": int(numerics["optimizer_max_iterations"]),
        },
    )
    if not result.success or not math.isfinite(float(result.x)):
        raise RuntimeError(f"direct optimizer failed: {result.message}")
    return float(result.x)


def inclusive_grid(start: float, stop: float, step: float) -> np.ndarray:
    start_value = float(start)
    stop_value = float(stop)
    step_value = float(step)
    if not all(math.isfinite(value) for value in (start_value, stop_value, step_value)):
        raise ValueError("grid values must be finite")
    if step_value <= 0.0 or stop_value < start_value:
        raise ValueError("grid requires positive step and stop >= start")
    intervals_float = (stop_value - start_value) / step_value
    intervals = round(intervals_float)
    if not math.isclose(intervals_float, intervals, rel_tol=0.0, abs_tol=1e-10):
        raise ValueError("grid stop must lie on the declared step")
    values = start_value + step_value * np.arange(intervals + 1, dtype=np.float64)
    values[-1] = stop_value
    return values


def rate_loss_fraction(u_candidate: float, u_optimum: float, gamma_ratio: float) -> float:
    optimum_rate = float(normalized_surrogate_rate(u_optimum, gamma_ratio))
    candidate_rate = float(normalized_surrogate_rate(u_candidate, gamma_ratio))
    if optimum_rate <= 0.0:
        raise ValueError("optimum rate must be positive")
    return 1.0 - candidate_rate / optimum_rate
