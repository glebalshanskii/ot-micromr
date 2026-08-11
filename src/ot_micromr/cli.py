from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from ot_micromr.config import load_runspec
from ot_micromr.errors import ConfigError, ExperimentError
from ot_micromr.experiments import run_experiment
from ot_micromr.okx_data import fetch_okx_source_list, load_okx_source_list


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
    fetch = subcommands.add_parser(
        "fetch-data", help="explicitly download a frozen empirical source list"
    )
    fetch.add_argument("path", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(argv) if argv is not None else sys.argv[1:]
    parser = _parser()
    if not arguments:
        parser.print_help()
        return 0
    parsed = parser.parse_args(arguments)
    try:
        if parsed.command == "fetch-data":
            source_list = load_okx_source_list(parsed.path)
            manifest = fetch_okx_source_list(source_list)
            print(
                json.dumps(
                    {
                        "dataset_id": manifest["dataset_id"],
                        "dataset_content_sha256": manifest["dataset_content_sha256"],
                        "manifest_path": str(source_list.manifest_path),
                        "asset_count": len(manifest["assets"]),
                    },
                    sort_keys=True,
                )
            )
            return 0
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
