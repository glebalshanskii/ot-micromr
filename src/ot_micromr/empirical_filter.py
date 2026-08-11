from __future__ import annotations

import heapq
import math
import tarfile
import time
from array import array
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import orjson
import torch

from ot_micromr.artifacts import atomic_write_json, sha256_file, write_csv
from ot_micromr.config import RunSpec
from ot_micromr.efficient_price import systematic_resample
from ot_micromr.empirical_data import (
    ORDERBOOK_KEYS,
    _apply_levels,
    _clean_heap,
    _compact_heap,
)
from ot_micromr.errors import ExperimentError
from ot_micromr.okx_data import load_okx_source_list, load_raw_manifest


@dataclass(frozen=True, slots=True)
class EmpiricalFilterResult:
    metrics: Mapping[str, Any]
    acceptance: Mapping[str, bool]
    derived_parameters: Mapping[str, Any]
    log_lines: Sequence[str]

    @property
    def passed(self) -> bool:
        return all(self.acceptance.values())


@dataclass(frozen=True, slots=True)
class DayData:
    date: str
    timestamps_ms: torch.Tensor
    bid_ticks: torch.Tensor
    ask_ticks: torch.Tensor
    snapshot_reset: torch.Tensor
    event_codes: torch.Tensor
    compatible: torch.Tensor
    reference_price: torch.Tensor
    reference_timestamp_ms: torch.Tensor
    reference_kind: str


def _extract_bbo_worker(task: Mapping[str, Any]) -> dict[str, Any]:
    torch.set_num_threads(1)
    path = Path(str(task["path"]))
    output_path = Path(str(task["output_path"]))
    expected_sha256 = str(task["sha256"])
    if path.stat().st_size != int(task["size_bytes"]):
        raise ExperimentError(f"raw size mismatch for {path.name}")
    if sha256_file(path) != expected_sha256:
        raise ExperimentError(f"raw SHA-256 mismatch for {path.name}")

    instrument = str(task["instrument"])
    timestamps = array("q")
    bids_out = array("q")
    asks_out = array("q")
    resets = array("b")
    asks: dict[int, float] = {}
    bids: dict[int, float] = {}
    ask_heap: list[int] = []
    bid_heap: list[int] = []
    seen_snapshot = False
    quarantine = False
    previous_bid: int | None = None
    previous_ask: int | None = None
    source_rows = 0
    quarantine_rows = 0
    started = time.perf_counter()

    with path.open("rb") as compressed:
        with tarfile.open(fileobj=compressed, mode="r|gz") as archive:
            member = archive.next()
            while member is not None and not member.isfile():
                member = archive.next()
            if member is None:
                raise ExperimentError(f"order-book archive has no regular member: {path}")
            stream = archive.extractfile(member)
            if stream is None:
                raise ExperimentError(f"cannot open order-book member: {path}")
            for row_index, line in enumerate(stream):
                try:
                    record = orjson.loads(line)
                    if not isinstance(record, dict) or (
                        row_index == 0 and set(record) != ORDERBOOK_KEYS
                    ):
                        raise ValueError("unexpected order-book schema")
                    if record["instId"] != instrument:
                        raise ValueError("unexpected instrument")
                    action = record["action"]
                    if action not in {"snapshot", "update"}:
                        raise ValueError("unexpected action")
                    source_rows += 1
                    if action == "snapshot":
                        asks.clear()
                        bids.clear()
                        ask_heap.clear()
                        bid_heap.clear()
                        seen_snapshot = True
                    _apply_levels(record["asks"], asks, ask_heap, 1, 10)
                    _apply_levels(record["bids"], bids, bid_heap, -1, 10)
                    if not seen_snapshot:
                        continue
                    if action == "snapshot":
                        ask_heap[:] = list(asks)
                        bid_heap[:] = [-price for price in bids]
                        heapq.heapify(ask_heap)
                        heapq.heapify(bid_heap)
                    _clean_heap(ask_heap, asks, 1)
                    _clean_heap(bid_heap, bids, -1)
                    _compact_heap(ask_heap, asks, 1)
                    _compact_heap(bid_heap, bids, -1)
                    best_ask = ask_heap[0] if ask_heap else 0
                    best_bid = -bid_heap[0] if bid_heap else 0
                    if best_bid <= 0 or best_ask <= 0 or best_bid >= best_ask:
                        quarantine = True
                        quarantine_rows += 1
                        continue
                    if quarantine and action != "snapshot":
                        quarantine_rows += 1
                        continue
                    recovered = quarantine and action == "snapshot"
                    quarantine = False
                    changed = best_bid != previous_bid or best_ask != previous_ask
                    reset = action == "snapshot" or recovered
                    if changed or reset:
                        timestamps.append(int(record["ts"]))
                        bids_out.append(best_bid)
                        asks_out.append(best_ask)
                        resets.append(int(reset))
                        previous_bid = best_bid
                        previous_ask = best_ask
                except Exception as error:
                    raise ExperimentError(
                        f"{path.name}: invalid L2 row {row_index + 1}: {error}"
                    ) from error
    if not timestamps:
        raise ExperimentError(f"no usable BBO observations in {path}")
    payload = {
        "schema_version": "bbo-events-v1",
        "asset_id": str(task["asset_id"]),
        "date": str(task["date"]),
        "instrument": instrument,
        "instrument_type": str(task["instrument_type"]),
        "timestamps_ms": torch.frombuffer(timestamps, dtype=torch.int64).clone(),
        "bid_ticks": torch.frombuffer(bids_out, dtype=torch.int64).clone(),
        "ask_ticks": torch.frombuffer(asks_out, dtype=torch.int64).clone(),
        "snapshot_reset": torch.frombuffer(resets, dtype=torch.int8).clone().to(torch.bool),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    return {
        "asset_id": str(task["asset_id"]),
        "date": str(task["date"]),
        "instrument_type": str(task["instrument_type"]),
        "source_rows": source_rows,
        "bbo_rows": len(timestamps),
        "snapshot_resets": int(sum(resets)),
        "quarantine_rows": quarantine_rows,
        "relative_path": output_path.name,
        "sha256": sha256_file(output_path),
        "elapsed_seconds": time.perf_counter() - started,
    }


def _extract_frozen_bbo(spec: RunSpec, run_directory: Path) -> list[dict[str, Any]]:
    values = spec.values
    source_list = load_okx_source_list(
        spec.repository_root / str(values["inputs"]["source_spec_path"]), spec.repository_root
    )
    raw_manifest = load_raw_manifest(
        spec.repository_root / str(values["inputs"]["raw_manifest_path"])
    )
    manifest_by_id = {str(row["asset_id"]): row for row in raw_manifest["assets"]}
    wanted_dates = set(values["evaluation"]["all_swap_dates"]) | set(
        values["evaluation"]["spot_dates"]
    )
    tasks: list[dict[str, Any]] = []
    for asset in source_list.assets:
        if asset.kind != "orderbook_l2" or asset.date not in wanted_dates:
            continue
        if asset.instrument_type == "SPOT" and asset.date not in values["evaluation"]["spot_dates"]:
            continue
        record = manifest_by_id[asset.asset_id]
        tasks.append(
            {
                "asset_id": asset.asset_id,
                "instrument": asset.instrument,
                "instrument_type": asset.instrument_type,
                "date": asset.date,
                "path": str(source_list.dataset_directory / asset.relative_path()),
                "output_path": str(run_directory / "state" / f"{asset.asset_id}.pt"),
                "size_bytes": int(record["size_bytes"]),
                "sha256": str(record["sha256"]),
            }
        )
    if len(tasks) != 10:
        raise ExperimentError(f"expected ten frozen L2 archives, found {len(tasks)}")
    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(
        max_workers=min(int(values["numerics"]["archive_workers"]), len(tasks))
    ) as executor:
        futures = {executor.submit(_extract_bbo_worker, task): task for task in tasks}
        for future in as_completed(futures):
            rows.append(future.result())
    rows.sort(key=lambda row: str(row["asset_id"]))
    atomic_write_json(
        run_directory / "state" / "extraction_manifest.json",
        {"schema_version": "p6-extraction-manifest-v1", "assets": rows},
    )
    return rows


def classify_paper_channels(
    bid_ticks: torch.Tensor,
    ask_ticks: torch.Tensor,
    snapshot_reset: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if bid_ticks.ndim != 1 or ask_ticks.shape != bid_ticks.shape:
        raise ExperimentError("BBO tensors must be equal one-dimensional arrays")
    event = torch.zeros_like(bid_ticks, dtype=torch.int64)
    if bid_ticks.numel() < 2:
        return event, torch.zeros_like(bid_ticks, dtype=torch.bool)
    previous_spread = ask_ticks[:-1] - bid_ticks[:-1]
    current_spread = ask_ticks[1:] - bid_ticks[1:]
    mid_change = (bid_ticks[1:] + ask_ticks[1:]) - (bid_ticks[:-1] + ask_ticks[:-1])
    permitted = ~snapshot_reset[1:]
    slide = permitted & (previous_spread == 1) & (current_spread == 1)
    opened = permitted & (previous_spread == 1) & (current_spread == 2)
    closed = permitted & (previous_spread == 2) & (current_spread == 1)
    current_event = torch.zeros_like(mid_change)
    current_event = torch.where(slide & (mid_change == 2), 1, current_event)
    current_event = torch.where(slide & (mid_change == -2), 2, current_event)
    current_event = torch.where(opened & (mid_change == 1), 3, current_event)
    current_event = torch.where(opened & (mid_change == -1), 4, current_event)
    current_event = torch.where(closed & (mid_change == 1), 5, current_event)
    current_event = torch.where(closed & (mid_change == -1), 6, current_event)
    event[1:] = current_event
    compatible = event != 0
    return event, compatible


def _causal_ewma(
    values: torch.Tensor, timestamps_ms: torch.Tensor, time_constant_seconds: float
) -> torch.Tensor:
    data = values.to(torch.float64)
    timestamps = timestamps_ms.to(device=data.device, dtype=torch.float64)
    prior = torch.empty_like(data)
    state = data[0]
    previous_timestamp = timestamps[0]
    # A 1024-observation block remains numerically safe in float64 for the
    # observed BBO cadence and reduces the state boundary to about 10^3 launches.
    block = 1024
    for start in range(0, data.numel(), block):
        stop = min(start + block, data.numel())
        times = timestamps[start:stop]
        deltas = torch.cat((times[:1] - previous_timestamp, times[1:] - times[:-1])) / 1000.0
        coefficients = torch.exp(-torch.clamp_min(deltas, 0.0) / time_constant_seconds)
        coefficients = torch.clamp_min(coefficients, 1e-12)
        products = torch.cumprod(coefficients, dim=0)
        posterior = products * (
            state
            + torch.cumsum((1.0 - coefficients) * data[start:stop] / products, dim=0)
        )
        prior[start:stop] = torch.cat((state.unsqueeze(0), posterior[:-1]))
        state = posterior[-1]
        previous_timestamp = times[-1]
    return prior.to(values.dtype)


def _load_day_payload(run_directory: Path, asset_id: str) -> Mapping[str, Any]:
    return torch.load(
        run_directory / "state" / f"{asset_id}.pt", map_location="cpu", weights_only=False
    )


def _prepare_days(spec: RunSpec, run_directory: Path) -> tuple[dict[str, DayData], int]:
    values = spec.values
    device = torch.device(str(values["numerics"]["compute_device"]))
    tau = float(values["model"]["basis_time_constant_seconds"])
    days: dict[str, DayData] = {}
    future_accesses = 0
    spot_dates = set(values["evaluation"]["spot_dates"])
    for date in values["evaluation"]["all_swap_dates"]:
        swap = _load_day_payload(run_directory, f"swap-l2-{date}")
        timestamps = swap["timestamps_ms"].to(device)
        bids = swap["bid_ticks"].to(device)
        asks = swap["ask_ticks"].to(device)
        resets = swap["snapshot_reset"].to(device)
        events, compatible = classify_paper_channels(bids, asks, resets)
        mid = (bids + asks).to(torch.float32) * 0.05
        if date in spot_dates:
            spot = _load_day_payload(run_directory, f"spot-l2-{date}")
            spot_timestamps = spot["timestamps_ms"].to(device)
            spot_mid = (spot["bid_ticks"].to(device) + spot["ask_ticks"].to(device)).to(
                torch.float32
            ) * 0.05
            indices = torch.searchsorted(spot_timestamps, timestamps, right=True) - 1
            valid = indices >= 0
            if not bool(torch.any(valid)):
                raise ExperimentError(f"no causal spot observation available on {date}")
            first_valid = int(torch.nonzero(valid, as_tuple=False)[0])
            if first_valid:
                timestamps = timestamps[first_valid:]
                bids = bids[first_valid:]
                asks = asks[first_valid:]
                resets = resets[first_valid:]
                mid = mid[first_valid:]
                events, compatible = classify_paper_channels(bids, asks, resets)
                indices = torch.searchsorted(spot_timestamps, timestamps, right=True) - 1
                valid = indices >= 0
            safe_indices = torch.clamp_min(indices, 0)
            selected_timestamps = spot_timestamps[safe_indices]
            future_accesses += int(torch.count_nonzero(valid & (selected_timestamps > timestamps)))
            raw_basis = mid - spot_mid[safe_indices]
            causal_basis = _causal_ewma(raw_basis, timestamps, tau)
            reference = spot_mid[safe_indices] + causal_basis
            reference = torch.where(valid, reference, mid)
            selected_timestamps = torch.where(valid, selected_timestamps, timestamps)
            reference_kind = "causal_spot_asof_plus_past_ewma_basis"
        else:
            reference = _causal_ewma(mid, timestamps, tau)
            selected_timestamps = timestamps
            reference_kind = "causal_swap_mid_ewma_descriptive_proxy"
        days[str(date)] = DayData(
            date=str(date),
            timestamps_ms=timestamps,
            bid_ticks=bids,
            ask_ticks=asks,
            snapshot_reset=resets,
            event_codes=events,
            compatible=compatible,
            reference_price=reference,
            reference_timestamp_ms=selected_timestamps,
            reference_kind=reference_kind,
        )
    return days, future_accesses


def _likelihood_inputs(day: DayData) -> tuple[torch.Tensor, ...]:
    mid = (day.bid_ticks + day.ask_ticks).to(torch.float64) * 0.05
    dt = (day.timestamps_ms[1:] - day.timestamps_ms[:-1]).to(torch.float64) / 1000.0
    mask = day.compatible[1:] & (dt > 0.0)
    gap = (mid[:-1] - day.reference_price[:-1].to(torch.float64))[mask]
    tight = ((day.ask_ticks[:-1] - day.bid_ticks[:-1]) == 1)[mask]
    return gap, tight, day.event_codes[1:][mask], dt[mask], day.timestamps_ms[1:][mask]


def _positive_parameters(raw: torch.Tensor, balanced: bool) -> torch.Tensor:
    positive = torch.nn.functional.softplus(raw) + 1e-9
    if balanced:
        return torch.stack(
            (
                positive[0],
                positive[1],
                positive[2],
                positive[3],
                positive[4],
                2.0 * positive[3] + positive[4],
            )
        )
    return positive


def _empirical_interval_nll(
    parameters: torch.Tensor,
    gap: torch.Tensor,
    tight: torch.Tensor,
    events: torch.Tensor,
    dt: torch.Tensor,
    delta: float = 0.1,
) -> torch.Tensor:
    mu_s, mu_o, mu_c, alpha_s, alpha_o, alpha_c = parameters
    positive = torch.clamp_min(gap, 0.0)
    negative = torch.clamp_min(-gap, 0.0)
    scale = 2.0 / delta
    total_tight = 2.0 * (mu_s + mu_o) + scale * (alpha_s + alpha_o) * torch.abs(gap)
    total_open = 2.0 * mu_c + scale * alpha_c * torch.abs(gap)
    total = torch.where(tight, total_tight, total_open)
    base = torch.stack(
        (mu_s, mu_s, mu_o, mu_o, mu_c, mu_c)
    )
    slope = torch.stack(
        (alpha_s, alpha_s, alpha_o, alpha_o, alpha_c, alpha_c)
    )
    directional_gap = torch.where(torch.remainder(events, 2) == 1, negative, positive)
    selected = base[events - 1] + scale * slope[events - 1] * directional_gap
    return total * dt - torch.log(selected)


def _fit_point_process(
    spec: RunSpec, day: DayData, balanced: bool
) -> tuple[dict[str, float], float, float]:
    gap, tight, events, dt, _ = _likelihood_inputs(day)
    if events.numel() < 100:
        raise ExperimentError(f"too few compatible events on {day.date}")
    dtype = torch.float64
    gap = gap.to(dtype)
    dt = dt.to(dtype)
    tight_exposure = torch.sum(dt[tight]).clamp_min(1.0)
    open_exposure = torch.sum(dt[~tight]).clamp_min(1.0)
    counts = torch.bincount(events, minlength=7).to(dtype)
    initial_positive = torch.stack(
        (
            (counts[1] + counts[2]) / (2.0 * tight_exposure),
            (counts[3] + counts[4]) / (2.0 * tight_exposure),
            (counts[5] + counts[6]) / (2.0 * open_exposure),
            gap.new_tensor(1e-3),
            gap.new_tensor(1e-3),
            gap.new_tensor(1e-3),
        )
    ).clamp_min(1e-6)
    if balanced:
        initial_positive = initial_positive[:5]
    raw = torch.nn.Parameter(torch.log(torch.expm1(initial_positive)))

    def loss_function(raw_parameters: torch.Tensor) -> torch.Tensor:
        parameters = _positive_parameters(raw_parameters, balanced)
        return _empirical_interval_nll(parameters, gap, tight, events, dt).mean()

    loss = (
        torch.compile(
            loss_function,
            mode=str(spec.values["numerics"]["compile_mode"]),
            fullgraph=True,
            dynamic=True,
        )
        if bool(spec.values["numerics"]["compile_enabled"])
        else loss_function
    )
    optimizer = torch.optim.Adam(
        (raw,), lr=float(spec.values["numerics"]["optimizer_learning_rate"])
    )
    initial_loss = float(loss(raw).detach())
    for _ in range(int(spec.values["numerics"]["optimizer_steps"])):
        optimizer.zero_grad(set_to_none=True)
        objective = loss(raw)
        objective.backward()
        optimizer.step()
    final_parameters = _positive_parameters(raw.detach(), balanced)
    final_loss = float(loss(raw).detach())
    if not bool(torch.all(torch.isfinite(final_parameters))) or not math.isfinite(final_loss):
        raise ExperimentError("non-finite point-process fit")
    names = ("mu_s", "mu_o", "mu_c", "alpha_s", "alpha_o", "alpha_c")
    return (
        {name: float(value) for name, value in zip(names, final_parameters, strict=True)},
        initial_loss,
        final_loss,
    )


def _parameter_tensor(parameters: Mapping[str, float], device: torch.device) -> torch.Tensor:
    return torch.tensor(
        [
            parameters["mu_s"],
            parameters["mu_o"],
            parameters["mu_c"],
            parameters["alpha_s"],
            parameters["alpha_o"],
            parameters["alpha_c"],
        ],
        device=device,
        dtype=torch.float64,
    )


def _selection_blocks(
    day: DayData,
    balanced_parameters: Mapping[str, float],
    unbalanced_parameters: Mapping[str, float],
    block_minutes: int,
    alpha: float,
    minimum_improvement: float,
) -> tuple[list[dict[str, Any]], float, str]:
    gap, tight, events, dt, timestamps = _likelihood_inputs(day)
    balanced = _parameter_tensor(balanced_parameters, gap.device)
    unbalanced = _parameter_tensor(unbalanced_parameters, gap.device)
    balanced_nll = _empirical_interval_nll(balanced, gap, tight, events, dt)
    unbalanced_nll = _empirical_interval_nll(unbalanced, gap, tight, events, dt)
    block_ms = block_minutes * 60 * 1000
    day_start = torch.div(timestamps.min(), 86_400_000, rounding_mode="floor") * 86_400_000
    block_index = torch.div(timestamps - day_start, block_ms, rounding_mode="floor")
    block_count = 24 * 60 // block_minutes
    counts = torch.bincount(block_index, minlength=block_count).to(torch.float64)
    improvement_sum = torch.zeros(block_count, device=gap.device, dtype=torch.float64)
    improvement_sum.scatter_add_(0, block_index, balanced_nll - unbalanced_nll)
    valid = counts > 0
    improvement = improvement_sum[valid] / counts[valid]
    mean = improvement.mean()
    standard_error = improvement.std(unbiased=True) / math.sqrt(improvement.numel())
    z = torch.distributions.Normal(0.0, 1.0).icdf(
        torch.tensor(1.0 - alpha, dtype=torch.float64)
    ).to(gap.device)
    lower = mean - z * standard_error
    rows: list[dict[str, Any]] = []
    valid_indices = torch.nonzero(valid, as_tuple=False).squeeze(-1)
    for position, index in enumerate(valid_indices):
        rows.append(
            {
                "block": int(index),
                "compatible_events": int(counts[index]),
                "unbalanced_improvement_nat_per_event": float(improvement[position]),
            }
        )
    return (
        rows,
        float(lower),
        "unbalanced" if float(lower) > minimum_improvement else "balanced",
    )


def _reduced_parameters(day: DayData) -> dict[str, float]:
    timestamps = day.timestamps_ms
    start = int(timestamps[0])
    stop = int(timestamps[-1])
    grid = torch.arange(start, stop + 1, 1000, device=timestamps.device, dtype=torch.int64)
    indices = torch.searchsorted(timestamps, grid, right=True) - 1
    valid = indices >= 0
    indices = indices[valid]
    mid = (day.bid_ticks[indices] + day.ask_ticks[indices]).to(torch.float64) * 0.05
    reference = day.reference_price[indices].to(torch.float64)
    gap = mid - reference
    centered_gap = gap - gap.mean()
    s_g = torch.sqrt(torch.mean(torch.square(centered_gap)))
    lagged = centered_gap[:-1]
    rho = torch.sum(lagged * centered_gap[1:]) / torch.sum(torch.square(lagged)).clamp_min(1e-12)
    rho = torch.clamp(rho, 1e-4, 0.9999)
    alpha = -torch.log(rho)
    increments = reference[1:] - reference[:-1]
    sigma_x = torch.sqrt(torch.mean(torch.square(increments)))
    return {
        "alpha_per_second": float(alpha),
        "s_g_usdt": float(s_g),
        "sigma_x_usdt_per_sqrt_second": float(sigma_x),
        "gap_mean_usdt": float(gap.mean()),
        "gap_acf_1s": float(rho),
        "gap_acf_5s": float(
            torch.sum(centered_gap[:-5] * centered_gap[5:])
            / torch.sum(torch.square(centered_gap[:-5])).clamp_min(1e-12)
        ),
        "gap_acf_10s": float(
            torch.sum(centered_gap[:-10] * centered_gap[10:])
            / torch.sum(torch.square(centered_gap[:-10])).clamp_min(1e-12)
        ),
        "grid_observations": int(gap.numel()),
    }


def _dawson_threshold_margin(s_g: float, half_spread: float) -> tuple[float, float, float]:
    gamma = half_spread / s_g
    upper = max(8.0, gamma + 4.0)
    u = torch.linspace(0.0, upper, 200_001, dtype=torch.float64)
    argument = u / math.sqrt(2.0)
    integrand = torch.exp(torch.square(argument))
    integral = torch.cat(
        (
            torch.zeros(1, dtype=torch.float64),
            torch.cumulative_trapezoid(integrand, argument),
        )
    )
    dawson = torch.exp(-torch.square(argument)) * integral
    foc = u - gamma - math.sqrt(2.0) * dawson
    candidates = torch.nonzero((u > gamma) & (foc >= 0.0), as_tuple=False).squeeze(-1)
    if candidates.numel() == 0:
        raise ExperimentError("Dawson threshold root is not bracketed")
    index = int(candidates[0])
    left = index - 1
    weight = -foc[left] / (foc[index] - foc[left])
    u_d = u[left] + weight * (u[index] - u[left])
    theta = float(u_d) * s_g
    return gamma, float(u_d), theta - half_spread


def _make_empirical_particle_chunk(
    parameters: Mapping[str, float],
    sigma_x: float,
    particle_count: int,
    chunk_events: int,
    compile_mode: str,
):
    parameter_tensor = torch.tensor(
        [
            parameters["mu_s"],
            parameters["mu_o"],
            parameters["mu_c"],
            parameters["alpha_s"],
            parameters["alpha_o"],
            parameters["alpha_c"],
        ],
        dtype=torch.float32,
        device="cuda",
    )

    def particle_chunk(
        particles: torch.Tensor,
        log_weights: torch.Tensor,
        prior_mid: torch.Tensor,
        prior_tight: torch.Tensor,
        events: torch.Tensor,
        dt: torch.Tensor,
        normals: torch.Tensor,
        offsets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        increments = sigma_x * torch.sqrt(dt).unsqueeze(-1) * normals
        particle_ends = particles.unsqueeze(0) + torch.cumsum(increments, dim=0)
        particle_starts = torch.cat((particles.unsqueeze(0), particle_ends[:-1]), dim=0)
        gap = prior_mid.unsqueeze(-1) - particle_starts
        positive = torch.clamp_min(gap, 0.0)
        negative = torch.clamp_min(-gap, 0.0)
        mu_s, mu_o, mu_c, alpha_s, alpha_o, alpha_c = parameter_tensor
        total_tight = 2.0 * (mu_s + mu_o) + 20.0 * (alpha_s + alpha_o) * torch.abs(gap)
        total_open = 2.0 * mu_c + 20.0 * alpha_c * torch.abs(gap)
        total = torch.where(prior_tight.unsqueeze(-1), total_tight, total_open)
        base = torch.stack((mu_s, mu_s, mu_o, mu_o, mu_c, mu_c))
        slope = torch.stack((alpha_s, alpha_s, alpha_o, alpha_o, alpha_c, alpha_c))
        safe_events = torch.clamp_min(events, 1)
        directional = torch.where(
            torch.remainder(safe_events, 2).unsqueeze(-1) == 1, negative, positive
        )
        selected = base[safe_events - 1].unsqueeze(-1) + 20.0 * slope[
            safe_events - 1
        ].unsqueeze(-1) * directional
        event_score = torch.log(selected) - total * dt.unsqueeze(-1)
        interval_score = torch.where(
            events.unsqueeze(-1) == 0, -total * dt.unsqueeze(-1), event_score
        )
        cumulative = log_weights.unsqueeze(0) + torch.cumsum(interval_score, dim=0)
        normalizers = torch.logsumexp(cumulative, dim=-1)
        normalized = cumulative - normalizers.unsqueeze(-1)
        weights = torch.exp(normalized)
        estimates = torch.sum(weights * particle_ends, dim=-1)
        variances = torch.sum(
            weights * torch.square(particle_ends - estimates.unsqueeze(-1)), dim=-1
        )
        particles, log_weights = systematic_resample(
            particle_ends[-1], normalized[-1], offsets
        )
        return particles, log_weights, estimates, variances

    return torch.compile(particle_chunk, mode=compile_mode, fullgraph=True)


def _compatible_segments(compatible_intervals: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    padded = torch.cat(
        (
            torch.zeros(1, device=compatible_intervals.device, dtype=torch.bool),
            compatible_intervals,
            torch.zeros(1, device=compatible_intervals.device, dtype=torch.bool),
        )
    )
    changes = padded[1:].to(torch.int8) - padded[:-1].to(torch.int8)
    starts = torch.nonzero(changes == 1, as_tuple=False).squeeze(-1)
    ends = torch.nonzero(changes == -1, as_tuple=False).squeeze(-1)
    return starts, ends - starts


def _filter_audit_day(
    spec: RunSpec,
    day: DayData,
    parameters: Mapping[str, float],
    reduced: Mapping[str, float],
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, str]:
    device = day.timestamps_ms.device
    particle_count = int(spec.values["numerics"]["particle_count"])
    chunk_events = int(spec.values["numerics"]["particle_chunk_events"])
    group_size = 64
    mid = (day.bid_ticks + day.ask_ticks).to(torch.float32) * 0.05
    dt_all = (day.timestamps_ms[1:] - day.timestamps_ms[:-1]).to(torch.float32) / 1000.0
    compatible_intervals = day.compatible[1:] & (dt_all > 0.0)
    starts, lengths = _compatible_segments(compatible_intervals)
    estimates = mid.clone()
    variances = torch.full_like(mid, float(reduced["s_g_usdt"]) ** 2)
    initial_generator = torch.Generator(device=device).manual_seed(seed)
    transition_generator = torch.Generator(device=device).manual_seed(seed + 1)
    resampling_generator = torch.Generator(device=device).manual_seed(seed + 2)
    chunk = _make_empirical_particle_chunk(
        parameters,
        float(reduced["sigma_x_usdt_per_sqrt_second"]),
        particle_count,
        chunk_events,
        str(spec.values["numerics"]["compile_mode"]),
    )
    arange_chunk = torch.arange(chunk_events, device=device, dtype=torch.int64).unsqueeze(1)
    for group_start in range(0, starts.numel(), group_size):
        actual_count = min(group_size, starts.numel() - group_start)
        group_starts = torch.zeros(group_size, device=device, dtype=torch.int64)
        group_lengths = torch.zeros_like(group_starts)
        group_starts[:actual_count] = starts[group_start : group_start + actual_count]
        group_lengths[:actual_count] = lengths[group_start : group_start + actual_count]
        initial_mid = mid[torch.clamp_min(group_starts, 0)]
        particles = initial_mid.unsqueeze(-1) + float(reduced["s_g_usdt"]) * torch.randn(
            (group_size, particle_count),
            device=device,
            dtype=torch.float32,
            generator=initial_generator,
        )
        log_weights = torch.full_like(particles, -math.log(particle_count))
        maximum_length = int(group_lengths.max())
        for offset in range(0, maximum_length, chunk_events):
            interval_indices = group_starts.unsqueeze(0) + offset + arange_chunk
            valid = arange_chunk < torch.clamp_min(group_lengths - offset, 0).unsqueeze(0)
            safe_indices = torch.clamp(interval_indices, 0, dt_all.numel() - 1)
            dt = torch.where(valid, dt_all[safe_indices], torch.zeros_like(dt_all[safe_indices]))
            event = torch.where(
                valid, day.event_codes[1:][safe_indices], torch.zeros_like(safe_indices)
            )
            prior_mid = mid[:-1][safe_indices]
            prior_tight = ((day.ask_ticks[:-1] - day.bid_ticks[:-1]) == 1)[safe_indices]
            normals = torch.randn(
                (chunk_events, group_size, particle_count),
                device=device,
                dtype=torch.float32,
                generator=transition_generator,
            )
            offsets = torch.rand(
                (group_size,), device=device, dtype=torch.float32, generator=resampling_generator
            )
            particles, log_weights, chunk_estimates, chunk_variances = chunk(
                particles,
                log_weights,
                prior_mid,
                prior_tight,
                event,
                dt,
                normals,
                offsets,
            )
            target = safe_indices + 1
            estimates[target[valid]] = chunk_estimates[valid]
            variances[target[valid]] = chunk_variances[valid]
            particles = particles.clone()
            log_weights = log_weights.clone()
    digest = sha256_file_tensor(estimates, variances)
    return estimates, variances, digest


def sha256_file_tensor(*values: torch.Tensor) -> str:
    import hashlib

    digest = hashlib.sha256()
    for value in values:
        tensor = value.detach().to("cpu").contiguous()
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(bytes(tensor.untyped_storage()))
    return digest.hexdigest()


def _channel_and_day_diagnostics(
    day: DayData, parameters: Mapping[str, float]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    gap, tight, events, dt, _ = _likelihood_inputs(day)
    counts = torch.bincount(events, minlength=7)
    total_exposure = dt.sum()
    parameter_tensor = _parameter_tensor(parameters, gap.device)
    nll = _empirical_interval_nll(parameter_tensor, gap, tight, events, dt)
    mu_s, mu_o, mu_c, alpha_s, alpha_o, alpha_c = parameter_tensor
    total_rate = torch.where(
        tight,
        2.0 * (mu_s + mu_o) + 20.0 * (alpha_s + alpha_o) * torch.abs(gap),
        2.0 * mu_c + 20.0 * alpha_c * torch.abs(gap),
    )
    transformed = total_rate * dt
    names = ("slide_up", "slide_down", "open_up", "open_down", "close_up", "close_down")
    rows = [
        {
            "date": day.date,
            "channel": name,
            "count": int(counts[index + 1]),
            "rate_per_second_of_total_exposure": float(counts[index + 1] / total_exposure),
        }
        for index, name in enumerate(names)
    ]
    mid = (day.bid_ticks + day.ask_ticks).to(torch.float64) * 0.05
    gap_all = mid - day.reference_price.to(torch.float64)
    split = gap_all.numel() // 2
    diagnostics = {
        "date": day.date,
        "reference_kind": day.reference_kind,
        "bbo_rows": int(day.timestamps_ms.numel()),
        "compatible_events": int(events.numel()),
        "compatible_fraction_of_bbo_transitions": float(
            events.numel() / max(day.timestamps_ms.numel() - 1, 1)
        ),
        "frozen_model_nll_nat_per_event": float(nll.mean()),
        "time_rescaling_mean": float(transformed.mean()),
        "time_rescaling_standard_deviation": float(transformed.std(unbiased=True)),
        "gap_first_half_mean_usdt": float(gap_all[:split].mean()),
        "gap_second_half_mean_usdt": float(gap_all[split:].mean()),
        "gap_first_half_standard_deviation_usdt": float(gap_all[:split].std(unbiased=True)),
        "gap_second_half_standard_deviation_usdt": float(gap_all[split:].std(unbiased=True)),
    }
    return rows, diagnostics


def evaluate_empirical_filter(spec: RunSpec, run_directory: Path) -> EmpiricalFilterResult:
    if not torch.cuda.is_available():
        raise ExperimentError("EMP-FILTER-001 requires an available CUDA device")
    started = time.perf_counter()
    values = spec.values
    evaluation = values["evaluation"]
    extraction_rows = _extract_frozen_bbo(spec, run_directory)
    days, future_accesses = _prepare_days(spec, run_directory)

    fit_day = days[str(evaluation["fit_date"])]
    selection_day = days[str(evaluation["selection_date"])]
    audit_day = days[str(evaluation["audit_date"])]
    balanced, balanced_initial_nll, balanced_final_nll = _fit_point_process(
        spec, fit_day, True
    )
    unbalanced, unbalanced_initial_nll, unbalanced_final_nll = _fit_point_process(
        spec, fit_day, False
    )
    selection_rows, selection_lower, selected_model = _selection_blocks(
        selection_day,
        balanced,
        unbalanced,
        int(evaluation["selection_block_minutes"]),
        float(evaluation["selection_alpha"]),
        float(evaluation["unbalanced_minimum_improvement_nat_per_event"]),
    )
    selected_parameters = balanced if selected_model == "balanced" else unbalanced
    parameter_freeze_sha256 = orjson.dumps(
        {"model": selected_model, "parameters": selected_parameters},
        option=orjson.OPT_SORT_KEYS,
    )
    import hashlib

    parameter_freeze_sha256 = hashlib.sha256(parameter_freeze_sha256).hexdigest()
    reduced = _reduced_parameters(fit_day)
    gamma, u_d, option_margin = _dawson_threshold_margin(
        float(reduced["s_g_usdt"]), float(values["model"]["price_tick"]) / 2.0
    )

    master_seed = int(values["seed_policy"]["seeds"][0])
    estimates, variances, first_digest = _filter_audit_day(
        spec, audit_day, selected_parameters, reduced, master_seed
    )
    replay_estimates, replay_variances, second_digest = _filter_audit_day(
        spec, audit_day, selected_parameters, reduced, master_seed
    )
    deterministic_replay = (
        first_digest == second_digest
        and torch.equal(estimates, replay_estimates)
        and torch.equal(variances, replay_variances)
    )
    parameter_freeze_after = hashlib.sha256(
        orjson.dumps(
            {"model": selected_model, "parameters": selected_parameters},
            option=orjson.OPT_SORT_KEYS,
        )
    ).hexdigest()
    model_frozen = parameter_freeze_after == parameter_freeze_sha256

    posterior_sd = torch.sqrt(torch.clamp_min(variances, 0.0))
    median_posterior_sd = float(torch.median(posterior_sd))
    reference = audit_day.reference_price.to(torch.float32)
    mid = (audit_day.bid_ticks + audit_day.ask_ticks).to(torch.float32) * 0.05
    filter_rmse_to_reference = float(torch.sqrt(torch.mean(torch.square(estimates - reference))))
    mid_rmse_to_reference = float(torch.sqrt(torch.mean(torch.square(mid - reference))))
    z90 = 1.6448536269514722
    reference_coverage = float(
        torch.mean(
            (
                (reference >= estimates - z90 * posterior_sd)
                & (reference <= estimates + z90 * posterior_sd)
            ).to(torch.float32)
        )
    )
    all_finite = all(
        bool(torch.all(torch.isfinite(tensor)))
        for tensor in (estimates, variances, reference, posterior_sd)
    )
    positive_variance = bool(torch.all(variances > 0.0))

    channel_rows: list[dict[str, Any]] = []
    day_rows: list[dict[str, Any]] = []
    for date in evaluation["all_swap_dates"]:
        rows, diagnostics = _channel_and_day_diagnostics(
            days[str(date)], selected_parameters
        )
        channel_rows.extend(rows)
        day_rows.append(diagnostics)

    elapsed = time.perf_counter() - started
    synthetic_summary_path = (
        spec.repository_root
        / "outputs"
        / str(values["inputs"]["synthetic_dependency_run"])
        / "metrics"
        / "summary.json"
    )
    synthetic_summary = orjson.loads(synthetic_summary_path.read_bytes())
    synthetic_dependency_passed = (
        synthetic_summary.get("status") == "passed"
        and synthetic_summary.get("acceptance_passed") is True
    )
    acceptance = {
        "synthetic_dependency_passed": synthetic_dependency_passed,
        "zero_future_timestamp_accesses": future_accesses == 0,
        "deterministic_replay": deterministic_replay,
        "all_filter_values_finite": all_finite,
        "positive_posterior_variance": positive_variance,
        "model_frozen_before_audit": model_frozen,
        "uncertainty_below_option_margin": median_posterior_sd < option_margin,
        "wall_time_within_limit": elapsed < float(evaluation["maximum_wall_seconds"]),
    }
    metrics = {
        "selected_model": selected_model,
        "selection_lower_95_improvement_nat_per_event": selection_lower,
        "selection_minimum_improvement_nat_per_event": float(
            evaluation["unbalanced_minimum_improvement_nat_per_event"]
        ),
        "fit_balanced_nll_initial": balanced_initial_nll,
        "fit_balanced_nll_final": balanced_final_nll,
        "fit_unbalanced_nll_initial": unbalanced_initial_nll,
        "fit_unbalanced_nll_final": unbalanced_final_nll,
        "future_timestamp_accesses": future_accesses,
        "deterministic_replay": deterministic_replay,
        "filter_digest_sha256": first_digest,
        "median_posterior_standard_deviation_usdt": median_posterior_sd,
        "optimistic_option_margin_usdt": option_margin,
        "posterior_sd_to_option_margin_ratio": median_posterior_sd / option_margin,
        "audit_filter_rmse_to_spot_reference_usdt": filter_rmse_to_reference,
        "audit_current_mid_rmse_to_spot_reference_usdt": mid_rmse_to_reference,
        "audit_reference_coverage_by_posterior_90_interval": reference_coverage,
        "dawson_gamma_ratio": gamma,
        "dawson_u_d_ratio": u_d,
        "elapsed_seconds": elapsed,
        "extracted_source_rows": sum(int(row["source_rows"]) for row in extraction_rows),
        "extracted_bbo_rows": sum(int(row["bbo_rows"]) for row in extraction_rows),
    }

    parameter_rows: list[dict[str, Any]] = []
    for model_name, parameters, initial_nll, final_nll in (
        ("balanced", balanced, balanced_initial_nll, balanced_final_nll),
        ("unbalanced", unbalanced, unbalanced_initial_nll, unbalanced_final_nll),
    ):
        for name, value in parameters.items():
            parameter_rows.append(
                {
                    "model": model_name,
                    "selected": model_name == selected_model,
                    "parameter": name,
                    "value": value,
                    "fit_initial_nll_nat_per_event": initial_nll,
                    "fit_final_nll_nat_per_event": final_nll,
                }
            )
    for name, value in reduced.items():
        parameter_rows.append(
            {
                "model": "reduced_fit_day",
                "selected": True,
                "parameter": name,
                "value": value,
                "fit_initial_nll_nat_per_event": "",
                "fit_final_nll_nat_per_event": "",
            }
        )
    write_csv(
        run_directory / "metrics" / "extraction.csv",
        list(extraction_rows[0]),
        extraction_rows,
    )
    write_csv(
        run_directory / "metrics" / "day_diagnostics.csv", list(day_rows[0]), day_rows
    )
    write_csv(
        run_directory / "tables" / "parameters.csv",
        list(parameter_rows[0]),
        parameter_rows,
    )
    write_csv(
        run_directory / "tables" / "selection_blocks.csv",
        list(selection_rows[0]),
        selection_rows,
    )
    write_csv(
        run_directory / "tables" / "channel_counts.csv",
        list(channel_rows[0]),
        channel_rows,
    )
    atomic_write_json(
        run_directory / "metrics" / "timestamp_audit.json",
        {
            "schema_version": "timestamp-audit-v1",
            "future_timestamp_accesses": future_accesses,
            "alignment": values["model"]["spot_alignment"],
            "equality_allowed": True,
            "validation_and_test_access": values["inputs"]["validation_and_test_access"],
        },
    )
    atomic_write_json(
        run_directory / "metrics" / "replay.json",
        {
            "schema_version": "empirical-filter-replay-v1",
            "first_digest_sha256": first_digest,
            "second_digest_sha256": second_digest,
            "exact_tensor_equality": deterministic_replay,
            "master_seed": master_seed,
            "seed_offsets": {"initialization": 0, "transition": 1, "resampling": 2},
        },
    )
    torch.save(
        {
            "schema_version": "audit-filter-state-v1",
            "date": str(evaluation["audit_date"]),
            "timestamps_ms": audit_day.timestamps_ms.to("cpu"),
            "filtered_efficient_price": estimates.to("cpu"),
            "posterior_variance": variances.to("cpu"),
            "causal_reference_price": reference.to("cpu"),
            "reference_timestamp_ms": audit_day.reference_timestamp_ms.to("cpu"),
        },
        run_directory / "state" / "audit_filter.pt",
    )
    return EmpiricalFilterResult(
        metrics=metrics,
        acceptance=acceptance,
        derived_parameters={
            "selected_model": selected_model,
            "selected_parameters": selected_parameters,
            "reduced_parameters": reduced,
            "parameter_freeze_sha256": parameter_freeze_sha256,
            "dawson_gamma_ratio": gamma,
            "dawson_u_d_ratio": u_d,
            "optimistic_option_margin_usdt": option_margin,
            "cuda_device": torch.cuda.get_device_name(),
        },
        log_lines=(
            f"selected_model={selected_model}; selection_lower={selection_lower:.6g} nat/event",
            f"future_timestamp_accesses={future_accesses}",
            f"posterior_sd_median={median_posterior_sd:.6g}; option_margin={option_margin:.6g}",
            f"deterministic_replay={deterministic_replay}; digest={first_digest}",
            f"acceptance_passed={all(acceptance.values())}",
        ),
    )
