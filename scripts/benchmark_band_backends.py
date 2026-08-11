from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ot_micromr.vectorized_band import evaluate_discrete_band_proxy_batched_numpy


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark vectorised CPU and torch.compile GPU band evaluation"
    )
    parser.add_argument("--paths", type=int, default=20)
    parser.add_argument("--observations", type=int, default=50_000)
    parser.add_argument("--thresholds", type=int, default=21)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("outputs/p3v/band-backends.json"))
    return parser


def _median_runtime(call: Callable[[], Any], repeats: int) -> tuple[float, Any]:
    durations: list[float] = []
    result: Any = None
    for _ in range(repeats):
        started = time.perf_counter()
        result = call()
        durations.append(time.perf_counter() - started)
    return statistics.median(durations), result


def _torch_kernel(torch: Any, gaps: Any, tight: Any, thresholds: Any, interval: float) -> tuple[Any, ...]:
    path_count, observation_count = gaps.shape
    threshold_count = thresholds.numel()
    expanded = gaps.unsqueeze(-1)
    side = torch.zeros(
        (path_count, observation_count, threshold_count), dtype=torch.int8, device=gaps.device
    )
    side = torch.where(expanded >= thresholds.view(1, 1, -1), 1, side)
    side = torch.where(expanded <= -thresholds.view(1, 1, -1), -1, side)
    observed = side != 0
    time_index = torch.arange(observation_count, device=gaps.device).view(1, -1, 1)
    last_index = torch.cummax(torch.where(observed, time_index, -1), dim=1).values
    last_side = torch.gather(side, 1, torch.clamp_min(last_index, 0))
    last_side = torch.where(last_index >= 0, last_side, 0)
    previous_side = torch.cat((torch.zeros_like(last_side[:, :1]), last_side[:, :-1]), dim=1)
    first_fill = observed & (previous_side == 0)
    completed_flip = observed & (previous_side != 0) & (side != previous_side)
    fill_count = (first_fill | completed_flip).sum(dim=1, dtype=torch.int64)
    flip_count = completed_flip.sum(dim=1, dtype=torch.int64)
    half_spread = torch.where(tight, 0.5, 1.0)
    reward = 2.0 * (torch.abs(gaps) - half_spread)
    reward_sum = torch.where(completed_flip, reward.unsqueeze(-1), 0.0).sum(dim=1)
    reward_rate = reward_sum / ((observation_count - 1) * interval)
    open_count = (completed_flip & (~tight.unsqueeze(-1))).sum(dim=1, dtype=torch.int64)
    open_share = torch.where(flip_count != 0, open_count / flip_count, 0.0)
    return reward_rate, fill_count, flip_count, open_share


def _to_numpy(torch: Any, result: tuple[Any, ...]) -> tuple[np.ndarray, ...]:
    torch.cuda.synchronize()
    return tuple(value.detach().cpu().numpy() for value in result)


def _comparison(reference: Any, candidate: tuple[np.ndarray, ...]) -> dict[str, Any]:
    rate, fills, flips, open_share = candidate
    return {
        "fill_count_exact": bool(np.array_equal(reference.fill_count, fills)),
        "completed_flip_count_exact": bool(np.array_equal(reference.completed_flip_count, flips)),
        "reward_rate_max_abs_error": float(
            np.max(np.abs(reference.reward_rate_per_second - rate))
        ),
        "open_fill_share_max_abs_error": float(
            np.max(np.abs(reference.open_fill_share - open_share))
        ),
    }


def main() -> int:
    args = _parser().parse_args()
    if min(args.paths, args.thresholds, args.repeats) <= 0 or args.observations < 2:
        raise ValueError("paths, thresholds and repeats must be positive; observations >= 2")
    rng = np.random.Generator(np.random.PCG64DXSM(np.random.SeedSequence(2026081199)))
    gaps64 = rng.normal(0.0, 1.0, size=(args.paths, args.observations)).astype(np.float64)
    tight = rng.random((args.paths, args.observations)) >= 0.01
    thresholds64 = np.linspace(0.6, 1.6, args.thresholds, dtype=np.float64)
    interval = 0.01

    cpu_seconds, reference = _median_runtime(
        lambda: evaluate_discrete_band_proxy_batched_numpy(
            gaps64, tight, interval, thresholds64
        ),
        args.repeats,
    )
    payload: dict[str, Any] = {
        "schema_version": "band-backend-benchmark-v1",
        "claim_eligible": False,
        "purpose": "engineering backend selection before SIM-FIG4-002",
        "platform": platform.platform(),
        "workload": {
            "paths": args.paths,
            "observations_per_path": args.observations,
            "thresholds": args.thresholds,
            "elements": args.paths * args.observations * args.thresholds,
            "repeats": args.repeats,
        },
        "numpy_float64": {
            "median_end_to_end_seconds": cpu_seconds,
            "million_path_threshold_observations_per_second": (
                args.paths * args.observations * args.thresholds / cpu_seconds / 1e6
            ),
        },
        "torch": {"installed": False},
    }

    try:
        import torch
    except ImportError:
        pass
    else:
        payload["torch"] = {
            "installed": True,
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_runtime": torch.version.cuda,
        }
        if torch.cuda.is_available():
            payload["torch"]["device"] = torch.cuda.get_device_name(0)
            for numpy_dtype, torch_dtype, label in (
                (np.float32, torch.float32, "float32"),
                (np.float64, torch.float64, "float64"),
            ):
                host_gaps = gaps64.astype(numpy_dtype)
                host_thresholds = thresholds64.astype(numpy_dtype)

                device_gaps = torch.as_tensor(host_gaps, device="cuda")
                device_tight = torch.as_tensor(tight, device="cuda")
                device_thresholds = torch.as_tensor(host_thresholds, device="cuda")
                eager_result = _torch_kernel(
                    torch, device_gaps, device_tight, device_thresholds, interval
                )
                eager_candidate = _to_numpy(torch, eager_result)

                def eager_resident() -> tuple[np.ndarray, ...]:
                    return _to_numpy(
                        torch,
                        _torch_kernel(
                            torch, device_gaps, device_tight, device_thresholds, interval
                        ),
                    )

                eager_seconds, _ = _median_runtime(eager_resident, args.repeats)
                compiled = torch.compile(
                    lambda local_gaps, local_tight, local_thresholds: _torch_kernel(
                        torch, local_gaps, local_tight, local_thresholds, interval
                    ),
                    fullgraph=True,
                )
                torch.cuda.synchronize()
                cold_started = time.perf_counter()
                compiled_candidate = _to_numpy(
                    torch, compiled(device_gaps, device_tight, device_thresholds)
                )
                cold_seconds = time.perf_counter() - cold_started

                def compiled_resident() -> tuple[np.ndarray, ...]:
                    return _to_numpy(
                        torch, compiled(device_gaps, device_tight, device_thresholds)
                    )

                compiled_seconds, _ = _median_runtime(compiled_resident, args.repeats)

                def compiled_end_to_end() -> tuple[np.ndarray, ...]:
                    local_gaps = torch.as_tensor(host_gaps, device="cuda")
                    local_tight = torch.as_tensor(tight, device="cuda")
                    local_thresholds = torch.as_tensor(host_thresholds, device="cuda")
                    return _to_numpy(torch, compiled(local_gaps, local_tight, local_thresholds))

                end_to_end_seconds, _ = _median_runtime(compiled_end_to_end, args.repeats)
                payload["torch"][label] = {
                    "eager_resident_median_seconds": eager_seconds,
                    "compiled_cold_seconds": cold_seconds,
                    "compiled_resident_median_seconds": compiled_seconds,
                    "compiled_end_to_end_median_seconds": end_to_end_seconds,
                    "compiled_end_to_end_speedup_vs_numpy": cpu_seconds / end_to_end_seconds,
                    "comparison_eager": _comparison(reference, eager_candidate),
                    "comparison_compiled": _comparison(reference, compiled_candidate),
                }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
