from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

from ot_micromr.config import load_runspec
from ot_micromr.simulation_experiments import _ordered_simulations


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark deterministic P3V seed parallelism")
    parser.add_argument(
        "--workers",
        nargs="+",
        type=int,
        default=(1, 4, 10, 20),
        help="Process counts to benchmark",
    )
    parser.add_argument("--horizon", type=float, default=200.0)
    parser.add_argument("--epsilon", type=float, default=0.005)
    parser.add_argument("--output", type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    spec = load_runspec("cfg/experiments/sim_moments_001.toml")
    values = spec.to_dict()
    values["simulation"]["burn_in_reversion_times"] = 20.0
    values["simulation"]["horizon_reversion_times"] = args.horizon
    values["simulation"]["event_log"] = False
    values["evaluation"]["minimum_observations_per_seed_and_parity_for_slope"] = 1
    seeds = tuple(int(seed) for seed in values["seed_policy"]["seeds"])
    coordinates = [(args.epsilon, seed) for seed in seeds]
    reference_digests: list[str] | None = None
    rows: list[dict[str, object]] = []
    for workers in args.workers:
        started = time.perf_counter()
        results = _ordered_simulations(values, coordinates, workers)
        elapsed = time.perf_counter() - started
        digests = [result.replay_digest for result in results]
        if reference_digests is None:
            reference_digests = digests
        deterministic = digests == reference_digests
        rows.append(
            {
                "workers": workers,
                "elapsed_seconds": elapsed,
                "paths_per_second": len(results) / elapsed,
                "speedup_vs_one_worker": None,
                "bitwise_digest_match": deterministic,
                "total_step_count": sum(int(result.seed_metrics["step_count"]) for result in results),
            }
        )
    baseline = float(rows[0]["elapsed_seconds"])
    for row in rows:
        row["speedup_vs_one_worker"] = baseline / float(row["elapsed_seconds"])
    payload = {
        "schema_version": "p3v-compute-benchmark-v1",
        "platform": platform.platform(),
        "logical_cpu_count": os.cpu_count(),
        "native_threads_per_worker": 1,
        "pilot_only": True,
        "horizon_reversion_times": args.horizon,
        "epsilon": args.epsilon,
        "seed_count": len(seeds),
        "results": rows,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
