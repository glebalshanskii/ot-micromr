from __future__ import annotations

import hashlib
import json
import math
import tomllib
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from ot_micromr.errors import ConfigError


TOP_LEVEL_KEYS = {
    "schema_version",
    "experiment_id",
    "track",
    "mode",
    "objective",
    "claim_ids",
    "orders_enabled",
    "claim_eligible",
    "seed_policy",
    "units",
    "numerics",
    "inputs",
    "model",
    "simulation",
    "strategy",
    "execution",
    "evaluation",
    "acceptance",
    "artifacts",
}

ANALYTICAL_EXPERIMENTS = {"ANA-SMOKE-001", "ANA-FIG3-001"}
BALANCED_SIMULATION_EXPERIMENTS = {"SIM-MOMENTS-001", "SIM-MOMENTS-002"}
UNBALANCED_SIMULATION_EXPERIMENTS = {"SIM-UNBALANCED-001", "SIM-UNBALANCED-002"}
SIMULATION_EXPERIMENTS = BALANCED_SIMULATION_EXPERIMENTS | UNBALANCED_SIMULATION_EXPERIMENTS
FIGURE4_EXPERIMENTS = {"SIM-FIG4-PILOT-001", "SIM-FIG4-PILOT-002", "SIM-FIG4-002"}
SUPPORTED_EXPERIMENTS = ANALYTICAL_EXPERIMENTS | SIMULATION_EXPERIMENTS | FIGURE4_EXPERIMENTS


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _deep_freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_deep_freeze(item) for item in value)
    return value


def _deep_thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _deep_thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_deep_thaw(item) for item in value]
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _deep_thaw(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


@dataclass(frozen=True, slots=True)
class RunSpec:
    source_path: Path
    source_sha256: str
    repository_root: Path
    values: Mapping[str, Any]
    canonical_json: str
    sha256: str

    @property
    def experiment_id(self) -> str:
        return str(self.values["experiment_id"])

    def to_dict(self) -> dict[str, Any]:
        return _deep_thaw(self.values)


def _expect_keys(
    table: Mapping[str, Any], required: set[str], path: str, optional: set[str] | None = None
) -> None:
    allowed = required | (optional or set())
    missing = sorted(required - table.keys())
    unknown = sorted(table.keys() - allowed)
    if missing:
        raise ConfigError(f"{path}: missing required fields: {', '.join(missing)}")
    if unknown:
        raise ConfigError(f"{path}: unknown fields: {', '.join(unknown)}")


def _table(parent: Mapping[str, Any], key: str, path: str = "RunSpec") -> Mapping[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise ConfigError(f"{path}.{key}: expected table")
    return value


def _string(table: Mapping[str, Any], key: str, path: str) -> str:
    value = table.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{path}.{key}: expected non-empty string")
    return value


def _boolean(table: Mapping[str, Any], key: str, path: str) -> bool:
    value = table.get(key)
    if not isinstance(value, bool):
        raise ConfigError(f"{path}.{key}: expected boolean")
    return value


def _number(table: Mapping[str, Any], key: str, path: str, *, positive: bool = False) -> float:
    value = table.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{path}.{key}: expected number")
    result = float(value)
    if not math.isfinite(result):
        raise ConfigError(f"{path}.{key}: expected finite number")
    if positive and result <= 0.0:
        raise ConfigError(f"{path}.{key}: expected positive number")
    return result


def _integer(table: Mapping[str, Any], key: str, path: str, *, positive: bool = False) -> int:
    value = table.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{path}.{key}: expected integer")
    if positive and value <= 0:
        raise ConfigError(f"{path}.{key}: expected positive integer")
    return value


def _string_sequence(table: Mapping[str, Any], key: str, path: str) -> tuple[str, ...]:
    value = table.get(key)
    if not isinstance(value, list) or not value or any(not isinstance(item, str) or not item for item in value):
        raise ConfigError(f"{path}.{key}: expected non-empty string array")
    return tuple(value)


def _number_sequence(table: Mapping[str, Any], key: str, path: str) -> tuple[float, ...]:
    value = table.get(key)
    if not isinstance(value, list) or not value:
        raise ConfigError(f"{path}.{key}: expected non-empty number array")
    result: list[float] = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ConfigError(f"{path}.{key}[{index}]: expected number")
        number = float(item)
        if not math.isfinite(number):
            raise ConfigError(f"{path}.{key}[{index}]: expected finite number")
        result.append(number)
    return tuple(result)


def _validate_finite_tree(value: Any, path: str = "RunSpec") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            _validate_finite_tree(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _validate_finite_tree(item, f"{path}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ConfigError(f"{path}: non-finite float is forbidden")


def _validate_common(data: Mapping[str, Any]) -> None:
    _expect_keys(data, TOP_LEVEL_KEYS, "RunSpec")
    if _string(data, "schema_version", "RunSpec") != "runspec-v1":
        raise ConfigError("RunSpec.schema_version: expected 'runspec-v1'")
    experiment_id = _string(data, "experiment_id", "RunSpec")
    if experiment_id not in SUPPORTED_EXPERIMENTS:
        raise ConfigError(
            f"RunSpec.experiment_id: executable validator supports only {sorted(SUPPORTED_EXPERIMENTS)}"
        )
    if _string(data, "track", "RunSpec") != "synthetic":
        raise ConfigError("RunSpec.track: executable synthetic experiments require 'synthetic'")
    if _string(data, "mode", "RunSpec") != "paper-faithful":
        raise ConfigError("RunSpec.mode: executable synthetic experiments require 'paper-faithful'")
    _string(data, "objective", "RunSpec")
    _string_sequence(data, "claim_ids", "RunSpec")
    orders_enabled = _boolean(data, "orders_enabled", "RunSpec")
    if orders_enabled != (experiment_id in FIGURE4_EXPERIMENTS):
        expected = experiment_id in FIGURE4_EXPERIMENTS
        raise ConfigError(f"RunSpec.orders_enabled: expected {expected}")
    _boolean(data, "claim_eligible", "RunSpec")

    if experiment_id in FIGURE4_EXPERIMENTS:
        return
    if experiment_id in SIMULATION_EXPERIMENTS:
        _validate_simulation_common(data, experiment_id)
        return

    seed = _table(data, "seed_policy")
    _expect_keys(seed, {"rng_used", "rng_algorithm", "seeds", "mapping"}, "RunSpec.seed_policy")
    if _boolean(seed, "rng_used", "RunSpec.seed_policy"):
        raise ConfigError("RunSpec.seed_policy.rng_used: analytical experiments must be deterministic")
    if _string(seed, "rng_algorithm", "RunSpec.seed_policy") != "none":
        raise ConfigError("RunSpec.seed_policy.rng_algorithm: expected 'none'")
    seeds = seed.get("seeds")
    if not isinstance(seeds, list) or not seeds or any(isinstance(item, bool) or not isinstance(item, int) for item in seeds):
        raise ConfigError("RunSpec.seed_policy.seeds: expected non-empty integer array")
    _string(seed, "mapping", "RunSpec.seed_policy")

    units = _table(data, "units")
    _expect_keys(units, {"time", "price", "quantity", "cash", "timezone", "normalization"}, "RunSpec.units")
    expected_units = {
        "time": "second",
        "price": "synthetic_price_unit",
        "quantity": "lot",
        "cash": "synthetic_quote_currency",
        "timezone": "UTC",
        "normalization": "ou_dimensionless",
    }
    for key, expected in expected_units.items():
        if _string(units, key, "RunSpec.units") != expected:
            raise ConfigError(f"RunSpec.units.{key}: expected {expected!r}")

    inputs = _table(data, "inputs")
    _expect_keys(
        inputs,
        {"source_kind", "paper_version", "paper_pdf_sha256", "protocol_path", "dataset", "dataset_reason"},
        "RunSpec.inputs",
    )
    if _string(inputs, "source_kind", "RunSpec.inputs") != "paper":
        raise ConfigError("RunSpec.inputs.source_kind: expected 'paper'")
    if _string(inputs, "paper_version", "RunSpec.inputs") != "arXiv:2608.00885v1":
        raise ConfigError("RunSpec.inputs.paper_version: unexpected paper version")
    paper_hash = _string(inputs, "paper_pdf_sha256", "RunSpec.inputs")
    if len(paper_hash) != 64 or any(char not in "0123456789abcdef" for char in paper_hash):
        raise ConfigError("RunSpec.inputs.paper_pdf_sha256: expected lowercase SHA-256")
    _string(inputs, "protocol_path", "RunSpec.inputs")
    if _string(inputs, "dataset", "RunSpec.inputs") != "not_applicable":
        raise ConfigError("RunSpec.inputs.dataset: expected 'not_applicable'")
    _string(inputs, "dataset_reason", "RunSpec.inputs")

    for section in ("simulation", "strategy", "execution"):
        table = _table(data, section)
        _expect_keys(table, {"enabled", "reason"}, f"RunSpec.{section}")
        if _boolean(table, "enabled", f"RunSpec.{section}"):
            raise ConfigError(f"RunSpec.{section}.enabled: analytical experiment requires false")
        _string(table, "reason", f"RunSpec.{section}")

    artifacts = _table(data, "artifacts")
    _expect_keys(artifacts, {"output_root", "required_classes", "optional_classes"}, "RunSpec.artifacts")
    if _string(artifacts, "output_root", "RunSpec.artifacts") != "outputs":
        raise ConfigError("RunSpec.artifacts.output_root: P2 requires repository-local 'outputs'")
    _string_sequence(artifacts, "required_classes", "RunSpec.artifacts")
    optional = artifacts.get("optional_classes")
    if not isinstance(optional, list) or any(not isinstance(item, str) or not item for item in optional):
        raise ConfigError("RunSpec.artifacts.optional_classes: expected string array")


def _validate_simulation_common(data: Mapping[str, Any], experiment_id: str) -> None:
    seed = _table(data, "seed_policy")
    _expect_keys(
        seed,
        {"rng_used", "rng_algorithm", "seeds", "mapping", "bootstrap_seed", "stream_mapping_version"},
        "RunSpec.seed_policy",
    )
    if not _boolean(seed, "rng_used", "RunSpec.seed_policy"):
        raise ConfigError("RunSpec.seed_policy.rng_used: simulation requires true")
    if _string(seed, "rng_algorithm", "RunSpec.seed_policy") != "numpy.random.PCG64DXSM":
        raise ConfigError("RunSpec.seed_policy.rng_algorithm: expected 'numpy.random.PCG64DXSM'")
    seeds = seed.get("seeds")
    if (
        not isinstance(seeds, list)
        or len(seeds) != 20
        or any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ConfigError("RunSpec.seed_policy.seeds: expected 20 unique positive integer seeds")
    _string(seed, "mapping", "RunSpec.seed_policy")
    _integer(seed, "bootstrap_seed", "RunSpec.seed_policy", positive=True)
    if _string(seed, "stream_mapping_version", "RunSpec.seed_policy") != "single_policy_seedsequence_spawn_v1":
        raise ConfigError("RunSpec.seed_policy.stream_mapping_version: unexpected mapping")

    units = _table(data, "units")
    _expect_keys(units, {"time", "price", "quantity", "cash", "timezone", "normalization"}, "RunSpec.units")
    expected_units = {
        "time": "second",
        "price": "synthetic_price_unit",
        "quantity": "lot",
        "cash": "synthetic_quote_currency",
        "timezone": "UTC",
    }
    for key, expected in expected_units.items():
        if _string(units, key, "RunSpec.units") != expected:
            raise ConfigError(f"RunSpec.units.{key}: expected {expected!r}")
    expected_normalization = (
        "alpha_equals_one_reference_instance"
        if experiment_id in BALANCED_SIMULATION_EXPERIMENTS
        else "tight_response_alpha_equals_one_unbalanced_control"
    )
    if _string(units, "normalization", "RunSpec.units") != expected_normalization:
        raise ConfigError(f"RunSpec.units.normalization: expected {expected_normalization!r}")

    inputs = _table(data, "inputs")
    input_keys = {
        "source_kind",
        "paper_version",
        "paper_pdf_sha256",
        "protocol_path",
        "parameter_provenance",
        "dataset",
        "dataset_reason",
    }
    if experiment_id in UNBALANCED_SIMULATION_EXPERIMENTS:
        input_keys |= {"experiment_role", "base_experiment_id"}
    _expect_keys(inputs, input_keys, "RunSpec.inputs")
    if _string(inputs, "source_kind", "RunSpec.inputs") != "paper_model_project_parameters":
        raise ConfigError("RunSpec.inputs.source_kind: expected paper_model_project_parameters")
    if _string(inputs, "paper_version", "RunSpec.inputs") != "arXiv:2608.00885v1":
        raise ConfigError("RunSpec.inputs.paper_version: unexpected paper version")
    paper_hash = _string(inputs, "paper_pdf_sha256", "RunSpec.inputs")
    if len(paper_hash) != 64 or any(char not in "0123456789abcdef" for char in paper_hash):
        raise ConfigError("RunSpec.inputs.paper_pdf_sha256: expected lowercase SHA-256")
    for key in ("protocol_path", "parameter_provenance", "dataset_reason"):
        _string(inputs, key, "RunSpec.inputs")
    if _string(inputs, "dataset", "RunSpec.inputs") != "not_applicable":
        raise ConfigError("RunSpec.inputs.dataset: expected 'not_applicable'")
    if experiment_id in UNBALANCED_SIMULATION_EXPERIMENTS:
        if _string(inputs, "experiment_role", "RunSpec.inputs") != "extension_negative_control":
            raise ConfigError("RunSpec.inputs.experiment_role: expected extension_negative_control")
        expected_base = "SIM-MOMENTS-002" if experiment_id.endswith("002") else "SIM-MOMENTS-001"
        if _string(inputs, "base_experiment_id", "RunSpec.inputs") != expected_base:
            raise ConfigError(f"RunSpec.inputs.base_experiment_id: expected {expected_base}")

    for section in ("strategy", "execution"):
        table = _table(data, section)
        _expect_keys(table, {"enabled", "reason"}, f"RunSpec.{section}")
        if _boolean(table, "enabled", f"RunSpec.{section}"):
            raise ConfigError(f"RunSpec.{section}.enabled: P3 dynamics experiments require false")
        _string(table, "reason", f"RunSpec.{section}")

    artifacts = _table(data, "artifacts")
    _expect_keys(artifacts, {"output_root", "required_classes", "optional_classes"}, "RunSpec.artifacts")
    if _string(artifacts, "output_root", "RunSpec.artifacts") != "outputs":
        raise ConfigError("RunSpec.artifacts.output_root: expected repository-local 'outputs'")
    required = _string_sequence(artifacts, "required_classes", "RunSpec.artifacts")
    expected_required = {
        "source_config",
        "resolved_runspec",
        "manifest",
        "log",
        "metrics_summary",
        "metrics_raw",
        "table",
        "figure_data",
        "figure",
        "event_log",
    }
    if set(required) != expected_required:
        raise ConfigError("RunSpec.artifacts.required_classes: unexpected P3 artifact contract")
    optional = artifacts.get("optional_classes")
    if not isinstance(optional, list) or optional:
        raise ConfigError("RunSpec.artifacts.optional_classes: expected empty array")


def _validate_simulation_numerics(data: Mapping[str, Any], experiment_id: str) -> None:
    numerics = _table(data, "numerics")
    keys = {
        "float_dtype",
        "simulation_algorithm",
        "alpha_ref_per_second",
        "primary_resolution_epsilon",
        "refinement_epsilons",
        "max_time_step_rule",
        "max_event_probability_rule",
        "event_probability_formula",
        "event_type_probability",
        "continuous_crossing_detection",
        "crossing_time_approximation",
        "flat_entry_tree_depth_rule",
        "event_and_crossing_tie_break",
        "observation_boundary_handling",
        "invariant_suite_id",
        "diagnostic_quantile_probabilities",
    }
    _expect_keys(numerics, keys, "RunSpec.numerics")
    expected_strings = {
        "float_dtype": "float64",
        "simulation_algorithm": "adaptive_left_hazard_single_jump_v1",
        "event_probability_formula": "1 - exp(-lambda_total_left * dt_seconds)",
        "event_type_probability": "lambda_event_left / lambda_total_left",
        "event_and_crossing_tie_break": "brownian_crossing_then_book_event_at_equal_timestamp",
        "invariant_suite_id": "jump_simulator_invariants_v1",
    }
    for key, expected in expected_strings.items():
        if _string(numerics, key, "RunSpec.numerics") != expected:
            raise ConfigError(f"RunSpec.numerics.{key}: expected {expected!r}")
    for key in (
        "max_time_step_rule",
        "max_event_probability_rule",
        "continuous_crossing_detection",
        "crossing_time_approximation",
        "flat_entry_tree_depth_rule",
        "observation_boundary_handling",
    ):
        _string(numerics, key, "RunSpec.numerics")
    alpha_ref = _number(numerics, "alpha_ref_per_second", "RunSpec.numerics", positive=True)
    expected_alpha_ref = 1.0 if experiment_id in BALANCED_SIMULATION_EXPERIMENTS else 1.25
    if alpha_ref != expected_alpha_ref:
        raise ConfigError(f"RunSpec.numerics.alpha_ref_per_second: expected {expected_alpha_ref}")
    if _number(numerics, "primary_resolution_epsilon", "RunSpec.numerics", positive=True) != 0.01:
        raise ConfigError("RunSpec.numerics.primary_resolution_epsilon: expected 0.01")
    expected_epsilons = (0.01, 0.005) if experiment_id.endswith("002") else (0.02, 0.01, 0.005)
    if _number_sequence(numerics, "refinement_epsilons", "RunSpec.numerics") != expected_epsilons:
        raise ConfigError(
            f"RunSpec.numerics.refinement_epsilons: expected {list(expected_epsilons)}"
        )
    quantiles = _number_sequence(numerics, "diagnostic_quantile_probabilities", "RunSpec.numerics")
    if quantiles != (0.0, 0.25, 0.5, 0.9, 0.95, 0.99, 1.0):
        raise ConfigError("RunSpec.numerics.diagnostic_quantile_probabilities: unexpected grid")


def _validate_initial_state(model: Mapping[str, Any], delta: float) -> None:
    initial = _table(model, "initial_state", "RunSpec.model")
    keys = {
        "time_seconds",
        "mid_half_ticks",
        "efficient_price",
        "inventory_lots",
        "mid_marked_wealth_quote_currency",
        "efficient_price_marked_wealth_quote_currency",
        "state_representation",
        "derived_bid_price_assertion",
        "derived_spread_price_assertion",
        "derived_mid_price_assertion",
        "derived_gap_price_assertion",
    }
    _expect_keys(initial, keys, "RunSpec.model.initial_state")
    if _number(initial, "time_seconds", "RunSpec.model.initial_state") != 0.0:
        raise ConfigError("RunSpec.model.initial_state.time_seconds: expected 0")
    if _integer(initial, "mid_half_ticks", "RunSpec.model.initial_state") != 1:
        raise ConfigError("RunSpec.model.initial_state.mid_half_ticks: expected 1")
    efficient = _number(initial, "efficient_price", "RunSpec.model.initial_state")
    if _integer(initial, "inventory_lots", "RunSpec.model.initial_state") != 0:
        raise ConfigError("RunSpec.model.initial_state.inventory_lots: expected 0")
    for key in (
        "mid_marked_wealth_quote_currency",
        "efficient_price_marked_wealth_quote_currency",
        "derived_bid_price_assertion",
        "derived_gap_price_assertion",
    ):
        if _number(initial, key, "RunSpec.model.initial_state") != 0.0:
            raise ConfigError(f"RunSpec.model.initial_state.{key}: expected 0")
    if efficient != delta / 2.0:
        raise ConfigError("RunSpec.model.initial_state.efficient_price: expected delta/2")
    if _number(initial, "derived_spread_price_assertion", "RunSpec.model.initial_state") != delta:
        raise ConfigError("RunSpec.model.initial_state.derived_spread_price_assertion: expected delta")
    if _number(initial, "derived_mid_price_assertion", "RunSpec.model.initial_state") != delta / 2.0:
        raise ConfigError("RunSpec.model.initial_state.derived_mid_price_assertion: expected delta/2")
    _string(initial, "state_representation", "RunSpec.model.initial_state")


def _validate_simulation_model(data: Mapping[str, Any], experiment_id: str) -> None:
    model = _table(data, "model")
    balanced_keys = {
        "enabled",
        "delta_price",
        "sigma_x_price_per_sqrt_second",
        "mu_s_per_second",
        "mu_o_per_second",
        "mu_c_per_second",
        "alpha_s_per_second",
        "alpha_o_per_second",
        "alpha_c_per_second",
        "alpha_per_second",
        "balanced_response_constraint",
        "initial_state",
    }
    unbalanced_keys = {
        "enabled",
        "delta_price",
        "sigma_x_price_per_sqrt_second",
        "mu_s_per_second",
        "mu_o_per_second",
        "mu_c_per_second",
        "alpha_s_per_second",
        "alpha_o_per_second",
        "alpha_c_per_second",
        "tight_drift_coefficient_per_second",
        "open_drift_coefficient_per_second",
        "one_factor_change",
        "initial_state",
    }
    _expect_keys(
        model,
        balanced_keys if experiment_id in BALANCED_SIMULATION_EXPERIMENTS else unbalanced_keys,
        "RunSpec.model",
    )
    if not _boolean(model, "enabled", "RunSpec.model"):
        raise ConfigError("RunSpec.model.enabled: simulation model must be enabled")
    positive_keys = (
        "delta_price",
        "sigma_x_price_per_sqrt_second",
        "mu_s_per_second",
        "mu_o_per_second",
        "mu_c_per_second",
        "alpha_s_per_second",
        "alpha_c_per_second",
    )
    values = {key: _number(model, key, "RunSpec.model", positive=True) for key in positive_keys}
    alpha_o = _number(model, "alpha_o_per_second", "RunSpec.model")
    if alpha_o < 0.0:
        raise ConfigError("RunSpec.model.alpha_o_per_second: expected nonnegative number")
    delta = values["delta_price"]
    tight_alpha = 2.0 * values["alpha_s_per_second"] + alpha_o
    if experiment_id in BALANCED_SIMULATION_EXPERIMENTS:
        alpha = _number(model, "alpha_per_second", "RunSpec.model", positive=True)
        _string(model, "balanced_response_constraint", "RunSpec.model")
        if not (tight_alpha == values["alpha_c_per_second"] == alpha):
            raise ConfigError("RunSpec.model: balanced response identity is violated")
    else:
        tight_declared = _number(model, "tight_drift_coefficient_per_second", "RunSpec.model", positive=True)
        open_declared = _number(model, "open_drift_coefficient_per_second", "RunSpec.model", positive=True)
        if tight_declared != tight_alpha or open_declared != values["alpha_c_per_second"]:
            raise ConfigError("RunSpec.model: unbalanced drift coefficients disagree with primitives")
        if tight_declared != 1.0 or open_declared != 1.25:
            raise ConfigError("RunSpec.model: unexpected one-factor control coefficients")
        _string(model, "one_factor_change", "RunSpec.model")
    _validate_initial_state(model, delta)


def _validate_simulation_section(data: Mapping[str, Any], experiment_id: str) -> None:
    simulation = _table(data, "simulation")
    keys = {
        "enabled",
        "burn_in_reversion_times",
        "horizon_reversion_times",
        "replications",
        "event_log",
        "observation_interval_reversion_times",
        "simultaneous_book_events_allowed",
        "strategy_monitoring_enabled",
    }
    if experiment_id in UNBALANCED_SIMULATION_EXPERIMENTS:
        keys.add("reference_reversion_rate_per_second")
    if experiment_id.endswith("002"):
        keys |= {"cpu_workers", "deterministic_replay_seeds", "step_diagnostics"}
    _expect_keys(simulation, keys, "RunSpec.simulation")
    if not _boolean(simulation, "enabled", "RunSpec.simulation"):
        raise ConfigError("RunSpec.simulation.enabled: expected true")
    if _number(simulation, "burn_in_reversion_times", "RunSpec.simulation", positive=True) != 100.0:
        raise ConfigError("RunSpec.simulation.burn_in_reversion_times: expected 100")
    expected_horizon = (
        40000.0
        if experiment_id == "SIM-MOMENTS-002"
        else 20000.0
        if experiment_id == "SIM-UNBALANCED-002"
        else 2000.0
    )
    if _number(simulation, "horizon_reversion_times", "RunSpec.simulation", positive=True) != expected_horizon:
        raise ConfigError(
            f"RunSpec.simulation.horizon_reversion_times: expected {expected_horizon:g}"
        )
    if _integer(simulation, "replications", "RunSpec.simulation", positive=True) != 20:
        raise ConfigError("RunSpec.simulation.replications: expected 20")
    event_log = _boolean(simulation, "event_log", "RunSpec.simulation")
    if event_log != (not experiment_id.endswith("002")):
        raise ConfigError(
            f"RunSpec.simulation.event_log: expected {not experiment_id.endswith('002')}"
        )
    if _number(simulation, "observation_interval_reversion_times", "RunSpec.simulation", positive=True) != 0.01:
        raise ConfigError("RunSpec.simulation.observation_interval_reversion_times: expected 0.01")
    if _boolean(simulation, "simultaneous_book_events_allowed", "RunSpec.simulation"):
        raise ConfigError("RunSpec.simulation.simultaneous_book_events_allowed: expected false")
    if _boolean(simulation, "strategy_monitoring_enabled", "RunSpec.simulation"):
        raise ConfigError("RunSpec.simulation.strategy_monitoring_enabled: P3 dynamics requires false")
    if experiment_id in UNBALANCED_SIMULATION_EXPERIMENTS:
        if _number(simulation, "reference_reversion_rate_per_second", "RunSpec.simulation", positive=True) != 1.0:
            raise ConfigError("RunSpec.simulation.reference_reversion_rate_per_second: expected 1")
    if experiment_id.endswith("002"):
        if _boolean(simulation, "step_diagnostics", "RunSpec.simulation"):
            raise ConfigError("RunSpec.simulation.step_diagnostics: P3V requires false")
        workers = _integer(simulation, "cpu_workers", "RunSpec.simulation", positive=True)
        if workers > 20:
            raise ConfigError("RunSpec.simulation.cpu_workers: expected at most 20")
        replay_seeds = simulation.get("deterministic_replay_seeds")
        seeds = data["seed_policy"]["seeds"]
        if (
            not isinstance(replay_seeds, list)
            or len(replay_seeds) != 3
            or any(seed not in seeds for seed in replay_seeds)
            or len(set(replay_seeds)) != 3
        ):
            raise ConfigError(
                "RunSpec.simulation.deterministic_replay_seeds: expected three unique experiment seeds"
            )


def _validate_numerics(data: Mapping[str, Any], experiment_id: str) -> None:
    numerics = _table(data, "numerics")
    common = {
        "float_dtype",
        "dawson_backend",
        "erfi_backend",
        "root_algorithm",
        "root_lower_margin_ratio",
        "root_upper_u_ratio",
        "root_xtol",
        "root_rtol",
        "root_max_iterations",
        "optimizer_crosscheck_algorithm",
        "optimizer_lower_margin_ratio",
        "optimizer_upper_u_ratio",
        "optimizer_xatol",
        "optimizer_max_iterations",
    }
    _expect_keys(numerics, common, "RunSpec.numerics")
    expected_strings = {
        "float_dtype": "float64",
        "dawson_backend": "scipy.special.dawsn",
        "erfi_backend": "scipy.special.erfi",
        "root_algorithm": "scipy.optimize.brentq",
        "optimizer_crosscheck_algorithm": "scipy.optimize.minimize_scalar:bounded",
    }
    for key, expected in expected_strings.items():
        if _string(numerics, key, "RunSpec.numerics") != expected:
            raise ConfigError(f"RunSpec.numerics.{key}: expected {expected!r}")
    for key in (
        "root_lower_margin_ratio",
        "root_upper_u_ratio",
        "root_xtol",
        "root_rtol",
        "optimizer_lower_margin_ratio",
        "optimizer_upper_u_ratio",
        "optimizer_xatol",
    ):
        _number(numerics, key, "RunSpec.numerics", positive=True)
    _integer(numerics, "root_max_iterations", "RunSpec.numerics", positive=True)
    _integer(numerics, "optimizer_max_iterations", "RunSpec.numerics", positive=True)


def _validate_model(data: Mapping[str, Any], experiment_id: str) -> None:
    model = _table(data, "model")
    if experiment_id == "ANA-SMOKE-001":
        _expect_keys(model, {"enabled", "alpha_per_second", "s_g_price", "gamma_ratio", "phi_price", "constraint"}, "RunSpec.model")
    else:
        _expect_keys(model, {"enabled", "alpha_per_second", "s_g_price", "phi_rule"}, "RunSpec.model")
    if not _boolean(model, "enabled", "RunSpec.model"):
        raise ConfigError("RunSpec.model.enabled: analytical model must be enabled")
    alpha = _number(model, "alpha_per_second", "RunSpec.model", positive=True)
    s_g = _number(model, "s_g_price", "RunSpec.model", positive=True)
    if alpha != 1.0 or s_g != 1.0:
        raise ConfigError("RunSpec.model: paper-faithful P2 normalization requires alpha_per_second=s_g_price=1")
    if experiment_id == "ANA-SMOKE-001":
        gamma = _number(model, "gamma_ratio", "RunSpec.model", positive=True)
        phi = _number(model, "phi_price", "RunSpec.model", positive=True)
        _string(model, "constraint", "RunSpec.model")
        if not math.isclose(phi, gamma * s_g, rel_tol=0.0, abs_tol=1e-15):
            raise ConfigError("RunSpec.model: phi_price must equal gamma_ratio * s_g_price")
    else:
        _string(model, "phi_rule", "RunSpec.model")


def _validate_smoke(data: Mapping[str, Any]) -> None:
    evaluation = _table(data, "evaluation")
    _expect_keys(
        evaluation,
        {"primary_metric", "metrics", "aggregation", "confidence_interval", "confidence_interval_reason", "multiplicity", "multiplicity_reason"},
        "RunSpec.evaluation",
    )
    for key in ("primary_metric", "aggregation", "confidence_interval", "confidence_interval_reason", "multiplicity", "multiplicity_reason"):
        _string(evaluation, key, "RunSpec.evaluation")
    _string_sequence(evaluation, "metrics", "RunSpec.evaluation")

    acceptance = _table(data, "acceptance")
    _expect_keys(
        acceptance,
        {
            "dawson_root_abs_residual_max",
            "root_vs_direct_optimizer_abs_error_max",
            "optimum_rate_identity_abs_error_max",
            "require_u_d_strictly_greater_than_gamma",
            "require_clean_tree_for_claim",
            "stop_on_nonfinite_value",
        },
        "RunSpec.acceptance",
    )
    for key in (
        "dawson_root_abs_residual_max",
        "root_vs_direct_optimizer_abs_error_max",
        "optimum_rate_identity_abs_error_max",
    ):
        _number(acceptance, key, "RunSpec.acceptance", positive=True)
    for key in ("require_u_d_strictly_greater_than_gamma", "require_clean_tree_for_claim", "stop_on_nonfinite_value"):
        _boolean(acceptance, key, "RunSpec.acceptance")


def _validate_fig3(data: Mapping[str, Any]) -> None:
    evaluation = _table(data, "evaluation")
    _expect_keys(
        evaluation,
        {
            "primary_metric",
            "metrics",
            "aggregation",
            "confidence_interval",
            "confidence_interval_reason",
            "multiplicity",
            "gamma_grid",
            "rate_curve_grid",
            "reference_checkpoints",
            "reference_grid_summary",
        },
        "RunSpec.evaluation",
    )
    for key in ("primary_metric", "aggregation", "confidence_interval", "confidence_interval_reason", "multiplicity"):
        _string(evaluation, key, "RunSpec.evaluation")
    _string_sequence(evaluation, "metrics", "RunSpec.evaluation")

    gamma_grid = _table(evaluation, "gamma_grid", "RunSpec.evaluation")
    _expect_keys(gamma_grid, {"start_ratio", "stop_ratio", "step_ratio", "include_stop"}, "RunSpec.evaluation.gamma_grid")
    for key in ("start_ratio", "stop_ratio", "step_ratio"):
        _number(gamma_grid, key, "RunSpec.evaluation.gamma_grid", positive=True)
    _boolean(gamma_grid, "include_stop", "RunSpec.evaluation.gamma_grid")
    if float(gamma_grid["start_ratio"]) >= float(gamma_grid["stop_ratio"]):
        raise ConfigError("RunSpec.evaluation.gamma_grid: start_ratio must be below stop_ratio")

    rate_grid = _table(evaluation, "rate_curve_grid", "RunSpec.evaluation")
    _expect_keys(rate_grid, {"gamma_ratios", "u_start_at_gamma", "u_stop_ratio", "u_step_ratio", "myopic_points"}, "RunSpec.evaluation.rate_curve_grid")
    gammas = rate_grid.get("gamma_ratios")
    if not isinstance(gammas, list) or not gammas:
        raise ConfigError("RunSpec.evaluation.rate_curve_grid.gamma_ratios: expected non-empty number array")
    for index, gamma in enumerate(gammas):
        if isinstance(gamma, bool) or not isinstance(gamma, (int, float)) or not math.isfinite(float(gamma)) or float(gamma) <= 0:
            raise ConfigError(f"RunSpec.evaluation.rate_curve_grid.gamma_ratios[{index}]: expected positive finite number")
    _boolean(rate_grid, "u_start_at_gamma", "RunSpec.evaluation.rate_curve_grid")
    _number(rate_grid, "u_stop_ratio", "RunSpec.evaluation.rate_curve_grid", positive=True)
    _number(rate_grid, "u_step_ratio", "RunSpec.evaluation.rate_curve_grid", positive=True)
    _boolean(rate_grid, "myopic_points", "RunSpec.evaluation.rate_curve_grid")

    checkpoints = evaluation.get("reference_checkpoints")
    if not isinstance(checkpoints, list) or not checkpoints:
        raise ConfigError("RunSpec.evaluation.reference_checkpoints: expected non-empty table array")
    checkpoint_keys = {"gamma_ratio", "u_d_ratio", "u_star_ratio", "rate_loss_at_u_star_fraction"}
    for index, checkpoint in enumerate(checkpoints):
        if not isinstance(checkpoint, dict):
            raise ConfigError(f"RunSpec.evaluation.reference_checkpoints[{index}]: expected table")
        path = f"RunSpec.evaluation.reference_checkpoints[{index}]"
        _expect_keys(checkpoint, checkpoint_keys, path)
        for key in checkpoint_keys:
            _number(checkpoint, key, path, positive=True)

    summary = _table(evaluation, "reference_grid_summary", "RunSpec.evaluation")
    _expect_keys(
        summary,
        {"domain_gamma_min_ratio", "domain_gamma_max_ratio", "domain_gamma_step_ratio", "maximum_rate_loss_at_u_star_fraction", "argmax_gamma_ratio", "provenance"},
        "RunSpec.evaluation.reference_grid_summary",
    )
    for key in ("domain_gamma_min_ratio", "domain_gamma_max_ratio", "domain_gamma_step_ratio", "maximum_rate_loss_at_u_star_fraction", "argmax_gamma_ratio"):
        _number(summary, key, "RunSpec.evaluation.reference_grid_summary", positive=True)
    _string(summary, "provenance", "RunSpec.evaluation.reference_grid_summary")

    acceptance = _table(data, "acceptance")
    numeric_keys = {
        "dawson_root_abs_residual_max",
        "checkpoint_u_abs_error_max",
        "checkpoint_rate_loss_abs_error_max",
        "reference_grid_max_rate_loss_abs_error_max",
        "reference_grid_argmax_gamma_abs_error_max",
        "myopic_rate_abs_max",
    }
    boolean_keys = {"require_all_grid_rows_finite", "require_clean_tree_for_claim", "stop_on_nonfinite_value"}
    _expect_keys(acceptance, numeric_keys | boolean_keys, "RunSpec.acceptance")
    for key in numeric_keys:
        _number(acceptance, key, "RunSpec.acceptance", positive=True)
    for key in boolean_keys:
        _boolean(acceptance, key, "RunSpec.acceptance")


def _validate_simulation_evaluation(data: Mapping[str, Any], experiment_id: str) -> None:
    if experiment_id.endswith("002"):
        _validate_simulation_evaluation_v2(data, experiment_id)
        return
    evaluation = _table(data, "evaluation")
    common = {"primary_metric", "metrics", "aggregation", "confidence_interval", "multiplicity"}
    if experiment_id == "SIM-MOMENTS-001":
        keys = common | {
            "bootstrap_replications",
            "sampling_rule",
            "stationary_variance_estimator",
            "stationary_mean_normalization",
            "jump_variance_rate_estimator",
            "variance_identity_signed_relative_residual",
            "open_close_flow_signed_relative_residual",
            "finite_h_drift_estimator",
            "finite_h_drift_target_per_second",
            "generator_drift_check",
            "binned_drift_role",
            "drift_gap_bin_scaling",
            "acf_estimator",
            "simultaneous_interval_construction",
            "resolution_difference_se",
            "drift_gap_bin_edges_s_g",
            "acf_lags_reversion_times",
            "minimum_pooled_observations_per_drift_bin_and_parity",
            "minimum_observations_per_seed_and_parity_for_slope",
            "refinement_metrics",
        }
    else:
        keys = common | {
            "sampling_rule",
            "generator_drift_estimator",
            "finite_h_drift_estimator",
            "finite_h_contrast",
            "minimum_observations_per_seed_and_parity_for_slope",
            "resolution_difference_se",
        }
    _expect_keys(evaluation, keys, "RunSpec.evaluation")
    for key in keys:
        if key in {
            "metrics",
            "refinement_metrics",
            "drift_gap_bin_edges_s_g",
            "acf_lags_reversion_times",
            "bootstrap_replications",
            "minimum_pooled_observations_per_drift_bin_and_parity",
            "minimum_observations_per_seed_and_parity_for_slope",
        }:
            continue
        _string(evaluation, key, "RunSpec.evaluation")
    _string_sequence(evaluation, "metrics", "RunSpec.evaluation")
    _integer(
        evaluation,
        "minimum_observations_per_seed_and_parity_for_slope",
        "RunSpec.evaluation",
        positive=True,
    )
    if experiment_id == "SIM-MOMENTS-001":
        if _integer(evaluation, "bootstrap_replications", "RunSpec.evaluation", positive=True) != 10000:
            raise ConfigError("RunSpec.evaluation.bootstrap_replications: expected 10000")
        _integer(
            evaluation,
            "minimum_pooled_observations_per_drift_bin_and_parity",
            "RunSpec.evaluation",
            positive=True,
        )
        edges = _number_sequence(evaluation, "drift_gap_bin_edges_s_g", "RunSpec.evaluation")
        if any(right <= left for left, right in zip(edges, edges[1:])):
            raise ConfigError("RunSpec.evaluation.drift_gap_bin_edges_s_g: edges must increase")
        lags = _number_sequence(evaluation, "acf_lags_reversion_times", "RunSpec.evaluation")
        if any(lag <= 0.0 for lag in lags):
            raise ConfigError("RunSpec.evaluation.acf_lags_reversion_times: lags must be positive")
        _string_sequence(evaluation, "refinement_metrics", "RunSpec.evaluation")

    acceptance = _table(data, "acceptance")
    if experiment_id == "SIM-MOMENTS-001":
        numeric_keys = {
            "parity_violation_count_max",
            "illegal_transition_count_max",
            "negative_intensity_count_max",
            "nonzero_inactive_intensity_count_max",
            "invariant_violation_count_max",
            "nonfinite_value_count_max",
            "multiple_book_event_step_count_max",
            "deterministic_replay_mismatch_count_max",
            "generator_drift_abs_residual_max",
            "stationary_mean_gap_abs_over_s_g_max",
            "stationary_variance_identity_relative_error_max",
            "open_close_flow_relative_error_max",
            "conditional_drift_slope_relative_error_max",
            "refinement_each_metric_abs_difference_max",
        }
        boolean_keys = {
            "require_theoretical_drift_slopes_inside_simultaneous_95_percent_interval",
            "require_theoretical_acf_inside_simultaneous_95_percent_interval",
            "refinement_threshold_may_be_replaced_by_difference_se",
            "require_all_20_replications",
            "require_clean_tree_for_claim",
            "stop_on_invariant_violation",
            "stop_on_nonfinite_value",
        }
        string_keys = {"refinement_difference_se_rule"}
    else:
        numeric_keys = {
            "invariant_violation_count_max",
            "generator_tight_drift_slope_abs_error_max",
            "generator_open_drift_slope_abs_error_max",
            "generator_drift_abs_residual_max",
            "finite_h_parity_contrast_one_sided_95_percent_lower_bound_min_per_second",
            "refinement_primary_metric_abs_difference_max_per_second",
        }
        boolean_keys = {
            "refinement_threshold_may_be_replaced_by_difference_se",
            "require_all_20_replications",
            "require_clean_tree_for_claim",
            "stop_on_invariant_violation",
            "stop_on_nonfinite_value",
        }
        string_keys = {"refinement_difference_se_rule"}
    _expect_keys(acceptance, numeric_keys | boolean_keys | string_keys, "RunSpec.acceptance")
    for key in numeric_keys:
        value = _number(acceptance, key, "RunSpec.acceptance")
        if value < 0.0:
            raise ConfigError(f"RunSpec.acceptance.{key}: expected nonnegative number")
    for key in boolean_keys:
        _boolean(acceptance, key, "RunSpec.acceptance")
    for key in string_keys:
        _string(acceptance, key, "RunSpec.acceptance")


def _validate_simulation_evaluation_v2(data: Mapping[str, Any], experiment_id: str) -> None:
    evaluation = _table(data, "evaluation")
    common = {
        "primary_metric",
        "metrics",
        "aggregation",
        "confidence_interval",
        "multiplicity",
        "sampling_rule",
        "finite_h_drift_estimator",
        "minimum_observations_per_seed_and_parity_for_slope",
        "primary_family",
        "refinement_family",
        "familywise_alpha",
        "power_target",
    }
    if experiment_id in BALANCED_SIMULATION_EXPERIMENTS:
        keys = common | {
            "bootstrap_replications",
            "integrated_hazard_flow_estimator",
            "compensator_estimator",
            "finite_h_drift_target_per_second",
            "generator_drift_check",
            "acf_estimator",
            "drift_gap_bin_edges_s_g",
            "acf_lags_reversion_times",
            "minimum_pooled_observations_per_drift_bin_and_parity",
        }
    else:
        keys = common | {
            "generator_drift_estimator",
            "finite_h_contrast",
        }
    _expect_keys(evaluation, keys, "RunSpec.evaluation")
    sequence_keys = {
        "metrics",
        "primary_family",
        "refinement_family",
    }
    numeric_keys = {
        "bootstrap_replications",
        "minimum_observations_per_seed_and_parity_for_slope",
        "minimum_pooled_observations_per_drift_bin_and_parity",
        "familywise_alpha",
        "power_target",
    }
    for key in keys:
        if key in sequence_keys or key in numeric_keys or key in {
            "drift_gap_bin_edges_s_g",
            "acf_lags_reversion_times",
        }:
            continue
        _string(evaluation, key, "RunSpec.evaluation")
    for key in sequence_keys:
        _string_sequence(evaluation, key, "RunSpec.evaluation")
    _integer(
        evaluation,
        "minimum_observations_per_seed_and_parity_for_slope",
        "RunSpec.evaluation",
        positive=True,
    )
    alpha = _number(evaluation, "familywise_alpha", "RunSpec.evaluation", positive=True)
    power = _number(evaluation, "power_target", "RunSpec.evaluation", positive=True)
    if alpha != 0.05 or power != 0.90:
        raise ConfigError("RunSpec.evaluation: expected familywise_alpha=0.05 and power_target=0.90")
    if experiment_id in BALANCED_SIMULATION_EXPERIMENTS:
        _integer(evaluation, "bootstrap_replications", "RunSpec.evaluation", positive=True)
        _integer(
            evaluation,
            "minimum_pooled_observations_per_drift_bin_and_parity",
            "RunSpec.evaluation",
            positive=True,
        )
        edges = _number_sequence(evaluation, "drift_gap_bin_edges_s_g", "RunSpec.evaluation")
        if any(right <= left for left, right in zip(edges, edges[1:])):
            raise ConfigError("RunSpec.evaluation.drift_gap_bin_edges_s_g: edges must increase")
        lags = _number_sequence(evaluation, "acf_lags_reversion_times", "RunSpec.evaluation")
        if any(lag <= 0.0 for lag in lags):
            raise ConfigError("RunSpec.evaluation.acf_lags_reversion_times: lags must be positive")

    acceptance = _table(data, "acceptance")
    deterministic_numeric = {
        "invariant_violation_count_max",
        "parity_violation_count_max",
        "illegal_transition_count_max",
        "negative_intensity_count_max",
        "nonzero_inactive_intensity_count_max",
        "nonfinite_value_count_max",
        "multiple_book_event_step_count_max",
        "deterministic_replay_mismatch_count_max",
        "generator_drift_abs_residual_max",
    }
    scientific_numeric = (
        {
            "flow_equivalence_margin",
            "refinement_flow_equivalence_margin",
        }
        if experiment_id in BALANCED_SIMULATION_EXPERIMENTS
        else {
            "contrast_minimum_effect_per_second",
            "refinement_contrast_equivalence_margin_per_second",
        }
    )
    boolean_keys = {
        "require_all_replications",
        "require_clean_tree_for_claim",
        "stop_on_invariant_violation",
        "stop_on_nonfinite_value",
    }
    _expect_keys(
        acceptance,
        deterministic_numeric | scientific_numeric | boolean_keys,
        "RunSpec.acceptance",
    )
    for key in deterministic_numeric | scientific_numeric:
        if _number(acceptance, key, "RunSpec.acceptance") < 0.0:
            raise ConfigError(f"RunSpec.acceptance.{key}: expected nonnegative number")
    expected_scientific = (
        {
            "flow_equivalence_margin": 0.05,
            "refinement_flow_equivalence_margin": 0.05,
        }
        if experiment_id in BALANCED_SIMULATION_EXPERIMENTS
        else {
            "contrast_minimum_effect_per_second": 0.10,
            "refinement_contrast_equivalence_margin_per_second": 0.15,
        }
    )
    for key, expected in expected_scientific.items():
        if float(acceptance[key]) != expected:
            raise ConfigError(f"RunSpec.acceptance.{key}: expected preregistered {expected}")
    for key in boolean_keys:
        _boolean(acceptance, key, "RunSpec.acceptance")


def _validate_positive_unique_seed_array(
    table: Mapping[str, Any], key: str, path: str
) -> tuple[int, ...]:
    values = table.get(key)
    if (
        not isinstance(values, list)
        or not values
        or any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in values)
        or len(set(values)) != len(values)
    ):
        raise ConfigError(f"{path}.{key}: expected unique positive integer array")
    return tuple(values)


def _validate_figure4(data: Mapping[str, Any], experiment_id: str) -> None:
    pilot = experiment_id.startswith("SIM-FIG4-PILOT-")
    if _boolean(data, "claim_eligible", "RunSpec") != (not pilot):
        raise ConfigError(f"RunSpec.claim_eligible: expected {not pilot}")

    seed = _table(data, "seed_policy")
    _expect_keys(
        seed,
        {
            "rng_used",
            "rng_algorithm",
            "calibration_seeds",
            "strategy_seeds",
            "deterministic_replay_seeds",
            "mapping",
            "bootstrap_seed",
            "stream_mapping_version",
        },
        "RunSpec.seed_policy",
    )
    if not _boolean(seed, "rng_used", "RunSpec.seed_policy"):
        raise ConfigError("RunSpec.seed_policy.rng_used: expected true")
    if _string(seed, "rng_algorithm", "RunSpec.seed_policy") != "numpy.random.PCG64DXSM":
        raise ConfigError("RunSpec.seed_policy.rng_algorithm: unexpected algorithm")
    calibration_seeds = _validate_positive_unique_seed_array(
        seed, "calibration_seeds", "RunSpec.seed_policy"
    )
    strategy_seeds = _validate_positive_unique_seed_array(
        seed, "strategy_seeds", "RunSpec.seed_policy"
    )
    replay_seeds = _validate_positive_unique_seed_array(
        seed, "deterministic_replay_seeds", "RunSpec.seed_policy"
    )
    if set(calibration_seeds) & set(strategy_seeds):
        raise ConfigError("RunSpec.seed_policy: calibration and strategy seeds must be disjoint")
    if any(value not in strategy_seeds for value in replay_seeds):
        raise ConfigError("RunSpec.seed_policy.deterministic_replay_seeds: expected strategy seeds")
    if pilot and (len(calibration_seeds), len(strategy_seeds)) != (6, 6):
        raise ConfigError("RunSpec.seed_policy: pilot requires six calibration and strategy seeds")
    _string(seed, "mapping", "RunSpec.seed_policy")
    _integer(seed, "bootstrap_seed", "RunSpec.seed_policy", positive=True)
    if (
        _string(seed, "stream_mapping_version", "RunSpec.seed_policy")
        != "figure4_row_resolution_policy_bridge_v2"
    ):
        raise ConfigError("RunSpec.seed_policy.stream_mapping_version: unexpected mapping")

    units = _table(data, "units")
    _expect_keys(
        units,
        {"time", "price", "quantity", "cash", "timezone", "normalization"},
        "RunSpec.units",
    )
    expected_units = {
        "time": "second",
        "price": "synthetic_price_unit",
        "quantity": "lot",
        "cash": "synthetic_quote_currency",
        "timezone": "UTC",
        "normalization": "row_specific_alpha_and_calibrated_s_g",
    }
    for key, expected in expected_units.items():
        if _string(units, key, "RunSpec.units") != expected:
            raise ConfigError(f"RunSpec.units.{key}: expected {expected!r}")

    numerics = _table(data, "numerics")
    _expect_keys(
        numerics,
        {
            "market_float_dtype",
            "reduction_float_dtype",
            "simulation_algorithm",
            "primary_resolution_epsilon",
            "refinement_epsilons",
            "max_time_step_rule",
            "max_event_probability_rule",
            "bridge_crossing_probability",
            "bridge_probability_cutoff",
            "bridge_only_hit_time_rule",
            "omitted_probability_budget",
            "root_lower_margin_ratio",
            "root_upper_u_ratio",
            "root_xtol",
            "root_rtol",
            "root_max_iterations",
            "cpu_workers",
            "gpu_reduction_enabled",
            "gpu_compile_enabled",
            "gpu_fallback",
        },
        "RunSpec.numerics",
    )
    expected_numerics = {
        "market_float_dtype": "float64",
        "reduction_float_dtype": "float32",
        "simulation_algorithm": "adaptive_left_hazard_single_jump_v2",
        "bridge_crossing_probability": "exact_one_sided_brownian_bridge",
        "bridge_only_hit_time_rule": "step_midpoint",
        "gpu_fallback": "numpy_float64",
    }
    for key, expected in expected_numerics.items():
        if _string(numerics, key, "RunSpec.numerics") != expected:
            raise ConfigError(f"RunSpec.numerics.{key}: expected {expected!r}")
    for key in ("max_time_step_rule", "max_event_probability_rule"):
        _string(numerics, key, "RunSpec.numerics")
    for key in (
        "primary_resolution_epsilon",
        "bridge_probability_cutoff",
        "omitted_probability_budget",
        "root_lower_margin_ratio",
        "root_upper_u_ratio",
        "root_xtol",
        "root_rtol",
    ):
        _number(numerics, key, "RunSpec.numerics", positive=True)
    epsilons = _number_sequence(numerics, "refinement_epsilons", "RunSpec.numerics")
    expected_epsilons = (0.01, 0.005)
    if epsilons != expected_epsilons:
        raise ConfigError(f"RunSpec.numerics.refinement_epsilons: expected {expected_epsilons}")
    _integer(numerics, "root_max_iterations", "RunSpec.numerics", positive=True)
    workers = _integer(numerics, "cpu_workers", "RunSpec.numerics", positive=True)
    if workers > 20:
        raise ConfigError("RunSpec.numerics.cpu_workers: expected at most 20")
    _boolean(numerics, "gpu_reduction_enabled", "RunSpec.numerics")
    _boolean(numerics, "gpu_compile_enabled", "RunSpec.numerics")

    inputs = _table(data, "inputs")
    _expect_keys(
        inputs,
        {
            "source_kind",
            "paper_version",
            "paper_pdf_sha256",
            "protocol_path",
            "parameter_provenance",
            "dataset",
            "dataset_reason",
        },
        "RunSpec.inputs",
    )
    if _string(inputs, "source_kind", "RunSpec.inputs") != "paper_model_project_parameters":
        raise ConfigError("RunSpec.inputs.source_kind: unexpected source")
    if _string(inputs, "paper_version", "RunSpec.inputs") != "arXiv:2608.00885v1":
        raise ConfigError("RunSpec.inputs.paper_version: unexpected version")
    paper_hash = _string(inputs, "paper_pdf_sha256", "RunSpec.inputs")
    if len(paper_hash) != 64 or any(char not in "0123456789abcdef" for char in paper_hash):
        raise ConfigError("RunSpec.inputs.paper_pdf_sha256: expected lowercase SHA-256")
    for key in ("protocol_path", "parameter_provenance", "dataset_reason"):
        _string(inputs, key, "RunSpec.inputs")
    if _string(inputs, "dataset", "RunSpec.inputs") != "not_applicable":
        raise ConfigError("RunSpec.inputs.dataset: expected not_applicable")

    model = _table(data, "model")
    _expect_keys(
        model,
        {
            "enabled",
            "delta_price",
            "sigma_x_price_per_sqrt_second",
            "mu_s_per_second",
            "mu_o_per_second",
            "mu_c_per_second",
            "response_scale_alpha_per_second_grid",
            "alpha_s_fraction_of_alpha",
            "alpha_o_fraction_of_alpha",
            "alpha_c_fraction_of_alpha",
            "balanced_response_rule",
            "initial_state",
        },
        "RunSpec.model",
    )
    if not _boolean(model, "enabled", "RunSpec.model"):
        raise ConfigError("RunSpec.model.enabled: expected true")
    for key in (
        "delta_price",
        "sigma_x_price_per_sqrt_second",
        "mu_s_per_second",
        "mu_o_per_second",
        "mu_c_per_second",
    ):
        _number(model, key, "RunSpec.model", positive=True)
    alpha_grid = _number_sequence(model, "response_scale_alpha_per_second_grid", "RunSpec.model")
    if len(alpha_grid) < 3 or any(value <= 0.0 for value in alpha_grid):
        raise ConfigError("RunSpec.model.response_scale_alpha_per_second_grid: invalid grid")
    if any(right <= left for left, right in zip(alpha_grid, alpha_grid[1:])):
        raise ConfigError("RunSpec.model.response_scale_alpha_per_second_grid: must increase")
    fractions = tuple(
        _number(model, key, "RunSpec.model")
        for key in (
            "alpha_s_fraction_of_alpha",
            "alpha_o_fraction_of_alpha",
            "alpha_c_fraction_of_alpha",
        )
    )
    if fractions != (0.5, 0.0, 1.0):
        raise ConfigError("RunSpec.model: unexpected balanced response fractions")
    _string(model, "balanced_response_rule", "RunSpec.model")
    initial = _table(model, "initial_state", "RunSpec.model")
    _expect_keys(
        initial,
        {
            "time_seconds",
            "mid_half_ticks",
            "efficient_price",
            "inventory_lots",
            "mid_marked_wealth_quote_currency",
            "efficient_price_marked_wealth_quote_currency",
            "state_representation",
        },
        "RunSpec.model.initial_state",
    )
    if _number(initial, "time_seconds", "RunSpec.model.initial_state") != 0.0:
        raise ConfigError("RunSpec.model.initial_state.time_seconds: expected zero")
    if _integer(initial, "mid_half_ticks", "RunSpec.model.initial_state") != 1:
        raise ConfigError("RunSpec.model.initial_state.mid_half_ticks: expected one")
    delta = float(model["delta_price"])
    if _number(initial, "efficient_price", "RunSpec.model.initial_state") != delta / 2.0:
        raise ConfigError("RunSpec.model.initial_state.efficient_price: expected delta/2")
    for key in (
        "inventory_lots",
        "mid_marked_wealth_quote_currency",
        "efficient_price_marked_wealth_quote_currency",
    ):
        if _number(initial, key, "RunSpec.model.initial_state") != 0.0:
            raise ConfigError(f"RunSpec.model.initial_state.{key}: expected zero")
    _string(initial, "state_representation", "RunSpec.model.initial_state")

    simulation = _table(data, "simulation")
    _expect_keys(
        simulation,
        {
            "enabled",
            "calibration_burn_in_reversion_times",
            "calibration_sampling_reversion_times",
            "calibration_observation_interval_reversion_times",
            "market_burn_in_reversion_times",
            "strategy_burn_in_reversion_times",
            "horizon_reversion_times",
            "replications",
            "event_log",
            "fill_log",
            "simultaneous_book_events_allowed",
            "strategy_monitoring_enabled",
        },
        "RunSpec.simulation",
    )
    if not _boolean(simulation, "enabled", "RunSpec.simulation"):
        raise ConfigError("RunSpec.simulation.enabled: expected true")
    for key in (
        "calibration_burn_in_reversion_times",
        "calibration_sampling_reversion_times",
        "calibration_observation_interval_reversion_times",
        "market_burn_in_reversion_times",
        "strategy_burn_in_reversion_times",
        "horizon_reversion_times",
    ):
        _number(simulation, key, "RunSpec.simulation", positive=True)
    if _integer(simulation, "replications", "RunSpec.simulation", positive=True) != len(
        strategy_seeds
    ):
        raise ConfigError("RunSpec.simulation.replications: disagrees with strategy seeds")
    for key, expected in (
        ("event_log", False),
        ("fill_log", False),
        ("simultaneous_book_events_allowed", False),
        ("strategy_monitoring_enabled", True),
    ):
        if _boolean(simulation, key, "RunSpec.simulation") != expected:
            raise ConfigError(f"RunSpec.simulation.{key}: expected {expected}")

    strategy = _table(data, "strategy")
    _expect_keys(
        strategy,
        {
            "enabled",
            "policy",
            "threshold_multiplier_theta_over_theta_d_grid",
            "additional_thresholds",
            "initial_inventory_lots",
            "first_entry_quantity_lots",
            "subsequent_flip_quantity_lots",
            "flat_between_thresholds",
        },
        "RunSpec.strategy",
    )
    if not _boolean(strategy, "enabled", "RunSpec.strategy"):
        raise ConfigError("RunSpec.strategy.enabled: expected true")
    if _string(strategy, "policy", "RunSpec.strategy") != "symmetric_flip_band":
        raise ConfigError("RunSpec.strategy.policy: unexpected policy")
    thresholds = _number_sequence(
        strategy, "threshold_multiplier_theta_over_theta_d_grid", "RunSpec.strategy"
    )
    if any(value <= 0.0 for value in thresholds) or any(
        right <= left for left, right in zip(thresholds, thresholds[1:])
    ):
        raise ConfigError("RunSpec.strategy.threshold_multiplier_theta_over_theta_d_grid: invalid")
    if _string_sequence(strategy, "additional_thresholds", "RunSpec.strategy") != (
        "theta_star",
    ):
        raise ConfigError("RunSpec.strategy.additional_thresholds: expected theta_star")
    if (
        _integer(strategy, "initial_inventory_lots", "RunSpec.strategy") != 0
        or _integer(strategy, "first_entry_quantity_lots", "RunSpec.strategy") != 1
        or _integer(strategy, "subsequent_flip_quantity_lots", "RunSpec.strategy") != 2
        or _boolean(strategy, "flat_between_thresholds", "RunSpec.strategy")
    ):
        raise ConfigError("RunSpec.strategy: unexpected inventory transition contract")

    execution = _table(data, "execution")
    _expect_keys(
        execution,
        {
            "enabled",
            "fill_convention",
            "realized_half_spread_rule",
            "threshold_reference_phi_price",
            "contract_multiplier_quote_currency_per_price_lot",
            "latency_seconds",
            "fee_quote_currency_per_lot",
            "slippage_price",
            "market_impact_enabled",
            "partial_fills_enabled",
            "terminal_liquidation",
            "wealth_markings",
        },
        "RunSpec.execution",
    )
    if not _boolean(execution, "enabled", "RunSpec.execution"):
        raise ConfigError("RunSpec.execution.enabled: expected true")
    for key in ("fill_convention", "realized_half_spread_rule"):
        _string(execution, key, "RunSpec.execution")
    for key in (
        "threshold_reference_phi_price",
        "contract_multiplier_quote_currency_per_price_lot",
    ):
        _number(execution, key, "RunSpec.execution", positive=True)
    for key in ("latency_seconds", "fee_quote_currency_per_lot", "slippage_price"):
        if _number(execution, key, "RunSpec.execution") != 0.0:
            raise ConfigError(f"RunSpec.execution.{key}: expected zero")
    for key in ("market_impact_enabled", "partial_fills_enabled", "terminal_liquidation"):
        if _boolean(execution, key, "RunSpec.execution"):
            raise ConfigError(f"RunSpec.execution.{key}: expected false")
    if set(_string_sequence(execution, "wealth_markings", "RunSpec.execution")) != {
        "mid_marked",
        "efficient_price_marked",
    }:
        raise ConfigError("RunSpec.execution.wealth_markings: unexpected markings")

    evaluation = _table(data, "evaluation")
    _expect_keys(
        evaluation,
        {
            "primary_metric",
            "target_realized_gamma_ratios",
            "aggregation",
            "confidence_interval",
            "bootstrap_replications",
            "familywise_alpha",
            "power_target",
            "minimum_complete_interfill_intervals_per_seed_and_policy",
            "rate_normalization",
            "peak_rule",
            "rate_loss_rule",
        },
        "RunSpec.evaluation",
    )
    for key in (
        "primary_metric",
        "aggregation",
        "confidence_interval",
        "rate_normalization",
        "peak_rule",
        "rate_loss_rule",
    ):
        _string(evaluation, key, "RunSpec.evaluation")
    targets = _number_sequence(evaluation, "target_realized_gamma_ratios", "RunSpec.evaluation")
    if targets != (0.28, 0.36, 0.47):
        raise ConfigError("RunSpec.evaluation.target_realized_gamma_ratios: unexpected targets")
    _integer(evaluation, "bootstrap_replications", "RunSpec.evaluation", positive=True)
    if (
        _number(evaluation, "familywise_alpha", "RunSpec.evaluation") != 0.05
        or _number(evaluation, "power_target", "RunSpec.evaluation") != 0.90
    ):
        raise ConfigError("RunSpec.evaluation: expected alpha=0.05 and power=0.90")
    _integer(
        evaluation,
        "minimum_complete_interfill_intervals_per_seed_and_policy",
        "RunSpec.evaluation",
        positive=True,
    )

    acceptance = _table(data, "acceptance")
    _expect_keys(
        acceptance,
        {
            "require_all_replications",
            "require_all_response_rows",
            "require_all_thresholds",
            "require_nonflat_before_measurement",
            "minimum_complete_interfill_intervals_per_seed_and_policy",
            "invariant_violation_count_max",
            "nonfinite_value_count_max",
            "omitted_probability_sum_max",
            "dawson_root_abs_residual_max",
            "require_clean_tree_for_claim",
            "stop_on_invariant_violation",
            "stop_on_nonfinite_value",
            "result_label",
        },
        "RunSpec.acceptance",
    )
    for key in (
        "require_all_replications",
        "require_all_response_rows",
        "require_all_thresholds",
        "require_nonflat_before_measurement",
        "require_clean_tree_for_claim",
        "stop_on_invariant_violation",
        "stop_on_nonfinite_value",
    ):
        _boolean(acceptance, key, "RunSpec.acceptance")
    _integer(
        acceptance,
        "minimum_complete_interfill_intervals_per_seed_and_policy",
        "RunSpec.acceptance",
        positive=True,
    )
    for key in (
        "invariant_violation_count_max",
        "nonfinite_value_count_max",
        "omitted_probability_sum_max",
        "dawson_root_abs_residual_max",
    ):
        if _number(acceptance, key, "RunSpec.acceptance") < 0.0:
            raise ConfigError(f"RunSpec.acceptance.{key}: expected nonnegative")
    if _boolean(acceptance, "require_clean_tree_for_claim", "RunSpec.acceptance") != (not pilot):
        raise ConfigError("RunSpec.acceptance.require_clean_tree_for_claim: wrong pilot/target value")
    _string(acceptance, "result_label", "RunSpec.acceptance")

    artifacts = _table(data, "artifacts")
    _expect_keys(
        artifacts, {"output_root", "required_classes", "optional_classes"}, "RunSpec.artifacts"
    )
    if _string(artifacts, "output_root", "RunSpec.artifacts") != "outputs":
        raise ConfigError("RunSpec.artifacts.output_root: expected outputs")
    required = set(_string_sequence(artifacts, "required_classes", "RunSpec.artifacts"))
    expected_required = {
        "source_config",
        "resolved_runspec",
        "manifest",
        "log",
        "metrics_summary",
        "metrics_raw",
        "table",
        "figure_data",
        "figure",
        "calibration_table",
        "fill_log",
    }
    if required != expected_required:
        raise ConfigError("RunSpec.artifacts.required_classes: unexpected P4 contract")
    optional = artifacts.get("optional_classes")
    if not isinstance(optional, list) or any(not isinstance(item, str) for item in optional):
        raise ConfigError("RunSpec.artifacts.optional_classes: expected string array")


def validate_runspec(data: Mapping[str, Any]) -> None:
    _validate_finite_tree(data)
    _validate_common(data)
    experiment_id = str(data["experiment_id"])
    if experiment_id in FIGURE4_EXPERIMENTS:
        _validate_figure4(data, experiment_id)
    elif experiment_id in ANALYTICAL_EXPERIMENTS:
        _validate_numerics(data, experiment_id)
        _validate_model(data, experiment_id)
        if experiment_id == "ANA-SMOKE-001":
            _validate_smoke(data)
        else:
            _validate_fig3(data)
    else:
        _validate_simulation_numerics(data, experiment_id)
        _validate_simulation_model(data, experiment_id)
        _validate_simulation_section(data, experiment_id)
        _validate_simulation_evaluation(data, experiment_id)


def discover_repository_root(start: Path) -> Path:
    candidate = start.resolve()
    if candidate.is_file():
        candidate = candidate.parent
    for directory in (candidate, *candidate.parents):
        if (directory / "pyproject.toml").is_file() and (directory / "cfg" / "experiments").is_dir():
            return directory
    raise ConfigError(f"cannot discover repository root from {start}")


def load_runspec(path: str | Path, repository_root: str | Path | None = None) -> RunSpec:
    source_path = Path(path).resolve()
    if not source_path.is_file():
        raise ConfigError(f"config not found: {source_path}")
    root = Path(repository_root).resolve() if repository_root is not None else discover_repository_root(source_path)
    source_bytes = source_path.read_bytes()
    try:
        data = tomllib.loads(source_bytes.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise ConfigError(f"invalid TOML {source_path}: {error}") from error
    validate_runspec(data)

    protocol = root / str(data["inputs"]["protocol_path"])
    if not protocol.is_file():
        raise ConfigError(f"RunSpec.inputs.protocol_path does not exist: {protocol}")
    paper = root / "docs" / "papers" / "2608.00885v1 - Optimal Trading of Microstructure Mean Reversion.pdf"
    if not paper.is_file():
        raise ConfigError(f"paper PDF does not exist: {paper}")
    actual_paper_hash = hashlib.sha256(paper.read_bytes()).hexdigest()
    if actual_paper_hash != data["inputs"]["paper_pdf_sha256"]:
        raise ConfigError("RunSpec.inputs.paper_pdf_sha256 does not match the local paper")

    canonical = _canonical_json(data)
    return RunSpec(
        source_path=source_path,
        source_sha256=hashlib.sha256(source_bytes).hexdigest(),
        repository_root=root,
        values=_deep_freeze(data),
        canonical_json=canonical,
        sha256=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    )
