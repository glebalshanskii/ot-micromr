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
    if experiment_id not in ANALYTICAL_EXPERIMENTS:
        raise ConfigError(
            f"RunSpec.experiment_id: P2 validator supports only {sorted(ANALYTICAL_EXPERIMENTS)}; "
            "SIM contracts become executable in P3"
        )
    if _string(data, "track", "RunSpec") != "synthetic":
        raise ConfigError("RunSpec.track: analytical P2 experiments require 'synthetic'")
    if _string(data, "mode", "RunSpec") != "paper-faithful":
        raise ConfigError("RunSpec.mode: analytical P2 experiments require 'paper-faithful'")
    _string(data, "objective", "RunSpec")
    _string_sequence(data, "claim_ids", "RunSpec")
    if _boolean(data, "orders_enabled", "RunSpec"):
        raise ConfigError("RunSpec.orders_enabled: analytical experiments cannot create orders")
    _boolean(data, "claim_eligible", "RunSpec")

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


def validate_runspec(data: Mapping[str, Any]) -> None:
    _validate_finite_tree(data)
    _validate_common(data)
    experiment_id = str(data["experiment_id"])
    _validate_numerics(data, experiment_id)
    _validate_model(data, experiment_id)
    if experiment_id == "ANA-SMOKE-001":
        _validate_smoke(data)
    else:
        _validate_fig3(data)


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
