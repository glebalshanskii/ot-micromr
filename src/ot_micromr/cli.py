from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from ot_micromr.config import load_runspec
from ot_micromr.errors import ConfigError, ExperimentError
from ot_micromr.experiments import run_experiment
from ot_micromr.p4_acceptance_review import review_p4_run


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ot-micromr",
        description="Run preregistered OT-MicroMR reproduction experiments.",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)
    validate = subcommands.add_parser("validate-config", help="strictly validate a RunSpec TOML")
    validate.add_argument("path", type=Path)
    run = subcommands.add_parser("run", help="execute one immutable RunSpec TOML")
    run.add_argument("path", type=Path)
    review = subcommands.add_parser(
        "review-p4-acceptance",
        help="reclassify frozen SIM-FIG4-002 evidence under corrected validity policy",
    )
    review.add_argument("path", type=Path, help="source SIM-FIG4-002 run directory")
    review.add_argument("--output-root", type=Path, default=Path("outputs"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(argv) if argv is not None else sys.argv[1:]
    parser = _parser()
    if not arguments:
        parser.print_help()
        return 0
    parsed = parser.parse_args(arguments)
    try:
        if parsed.command == "review-p4-acceptance":
            destination, review = review_p4_run(parsed.path, parsed.output_root)
            passed = bool(review["corrected_operational_decision"]["acceptance_passed"])
            print(
                json.dumps(
                    {
                        "review_artifact": str(destination),
                        "policy_id": review["policy_id"],
                        "source_run_id": review["source_run_id"],
                        "operational_acceptance_passed": passed,
                        "scientific_status": review["scientific_decision"]["status"],
                    },
                    sort_keys=True,
                )
            )
            return 0 if passed else 1
        spec = load_runspec(parsed.path)
        if parsed.command == "validate-config":
            print(
                json.dumps(
                    {
                        "valid": True,
                        "experiment_id": spec.experiment_id,
                        "source_sha256": spec.source_sha256,
                        "runspec_sha256": spec.sha256,
                    },
                    sort_keys=True,
                )
            )
            return 0
        command = ["ot-micromr", *arguments]
        result = run_experiment(spec, command=command)
        print(
            json.dumps(
                {
                    "experiment_id": result.experiment_id,
                    "run_id": result.run_id,
                    "run_directory": str(result.run_directory),
                    "status": result.status,
                    "acceptance_passed": result.acceptance_passed,
                },
                sort_keys=True,
            )
        )
        return 0 if result.acceptance_passed else 1
    except (ConfigError, ExperimentError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
