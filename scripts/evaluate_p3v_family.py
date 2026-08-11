from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ot_micromr.statistical_gates import holm_adjust


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate the joint preregistered P3V gate")
    parser.add_argument("moments_summary", type=Path)
    parser.add_argument("control_summary", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/p3v/global-gate.json"))
    return parser


def _load(path: Path, experiment_id: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("experiment_id") != experiment_id:
        raise ValueError(f"{path}: expected {experiment_id}")
    if payload.get("status") not in {"passed", "acceptance_failed"}:
        raise ValueError(f"{path}: experiment did not complete")
    return payload


def _decision(component: dict[str, Any], adjusted: float, kind: str) -> str:
    if adjusted < 0.05:
        return "supported"
    if kind == "equivalence" and component["status"] == "meaningfully_different":
        return "meaningfully_different"
    if kind == "superiority" and component["status"] == "below_minimum":
        return "meaningfully_different"
    return "inconclusive"


def main() -> int:
    args = _parser().parse_args()
    moments = _load(args.moments_summary, "SIM-MOMENTS-002")
    control = _load(args.control_summary, "SIM-UNBALANCED-002")
    flow = moments["metrics"]["scientific_components"]["flow_equivalence"]
    contrast = control["metrics"]["scientific_components"][
        "unbalanced_contrast_superiority"
    ]
    primary_raw = [float(flow["p_equivalence"]), float(contrast["p_superiority"])]
    primary_adjusted = holm_adjust(primary_raw)
    primary = [
        {
            "component": "flow_equivalence",
            "raw_p_value": primary_raw[0],
            "holm_adjusted_p_value": primary_adjusted[0],
            "decision": _decision(flow, primary_adjusted[0], "equivalence"),
            "details": flow,
        },
        {
            "component": "unbalanced_contrast_superiority",
            "raw_p_value": primary_raw[1],
            "holm_adjusted_p_value": primary_adjusted[1],
            "decision": _decision(contrast, primary_adjusted[1], "superiority"),
            "details": contrast,
        },
    ]
    flow_refinement = moments["metrics"]["refinement_components"]["flow_equivalence"]
    contrast_refinement = control["metrics"]["refinement_components"][
        "contrast_equivalence"
    ]
    refinement_raw = [
        float(flow_refinement["p_equivalence"]),
        float(contrast_refinement["p_equivalence"]),
    ]
    refinement_adjusted = holm_adjust(refinement_raw)
    refinement = [
        {
            "component": "flow_refinement_equivalence",
            "raw_p_value": refinement_raw[0],
            "holm_adjusted_p_value": refinement_adjusted[0],
            "decision": _decision(
                flow_refinement, refinement_adjusted[0], "equivalence"
            ),
            "details": flow_refinement,
        },
        {
            "component": "control_refinement_equivalence",
            "raw_p_value": refinement_raw[1],
            "holm_adjusted_p_value": refinement_adjusted[1],
            "decision": _decision(
                contrast_refinement, refinement_adjusted[1], "equivalence"
            ),
            "details": contrast_refinement,
        },
    ]
    deterministic_passed = all(moments["acceptance"].values()) and all(
        control["acceptance"].values()
    )
    primary_supported = all(row["decision"] == "supported" for row in primary)
    refinement_supported = all(row["decision"] == "supported" for row in refinement)
    p3v_passed = deterministic_passed and primary_supported and refinement_supported
    any_difference = any(
        row["decision"] == "meaningfully_different" for row in primary + refinement
    )
    status = (
        "supported"
        if p3v_passed
        else "blocked-bias"
        if any_difference or not deterministic_passed
        else "blocked-precision"
    )
    payload = {
        "schema_version": "p3v-global-gate-v1",
        "status": status,
        "p3v_passed": p3v_passed,
        "familywise_alpha": 0.05,
        "multiplicity": "Holm within primary and refinement families separately",
        "deterministic_operational_passed": deterministic_passed,
        "primary_family": primary,
        "refinement_family": refinement,
        "inputs": {
            "moments_summary": str(args.moments_summary),
            "control_summary": str(args.control_summary),
        },
        "next_step": (
            "P3V_complete_follow_docs_plan"
            if p3v_passed
            else "keep_P4_blocked_and_do_not_append_seeds"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if p3v_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
