from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import torch

from ot_micromr.artifacts import atomic_write_json, sha256_file, write_csv
from ot_micromr.config import load_runspec
from ot_micromr.marked_filter import (
    MARK_COUNT,
    _empirical_probability_tables,
    _load_verified_p6_payloads,
    _prepare_empirical_marked_days,
    _stack_train_intervals,
    mark_metadata,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPOSITORY_ROOT / "cfg" / "experiments" / "emp_mark_filter_001.toml"
REJECTION_ATTEMPTS = 64
CONTINUOUS_CHUNK_SECONDS = 4.0
CONTINUOUS_BRIDGE_DEPTH = 8
CONTINUOUS_MAX_CHUNKS = 8
CONTINUOUS_CROSSING_BISECTIONS = 16


@dataclass(frozen=True, slots=True)
class ExactTransitionCsr:
    order: torch.Tensor
    offsets: torch.Tensor
    counts: torch.Tensor
    delta_bid: torch.Tensor
    delta_ask: torch.Tensor
    observed_cells: torch.Tensor


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot free-running several-event marked-model forecasts from every held-out BBO point."
    )
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--start-utc", default="2024-12-15T00:00:00Z")
    parser.add_argument("--window-minutes", type=int, default=2)
    parser.add_argument("--horizon-events", type=int, default=10)
    parser.add_argument("--paths", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=202608121200)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-directory", type=Path)
    return parser.parse_args(argv)


def _load_full_fold(
    run_directory: Path, date: str, device: torch.device
) -> tuple[list[str], torch.Tensor, torch.Tensor, torch.Tensor, str]:
    with (run_directory / "tables" / "fold_parameters.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        rows = list(csv.DictReader(stream))
    row = next(item for item in rows if item["heldout_date"] == date and item["model"] == "full")
    rates = torch.tensor(
        [
            float(row[f"baseline_rate_spread_{index + 1 if index < 7 else '8plus'}"])
            for index in range(8)
        ],
        device=device,
        dtype=torch.float32,
    )
    return (
        row["train_dates"].split(";"),
        rates,
        torch.tensor(float(row["alpha_per_second"]), device=device, dtype=torch.float32),
        torch.tensor(
            float(row["sigma_x_usdt_per_sqrt_second"]), device=device, dtype=torch.float32
        ),
        row["parameter_digest_sha256"],
    )


def _exact_transition_csr(train: tuple[torch.Tensor, ...]) -> ExactTransitionCsr:
    _, spread, mark, _, delta_y, delta_d, *_ = train
    delta_bid = torch.div(delta_y - delta_d, 2, rounding_mode="floor").to(torch.int64)
    delta_ask = torch.div(delta_y + delta_d, 2, rounding_mode="floor").to(torch.int64)
    direction = mark_metadata(spread.device)[0][mark]
    mirrored_bid = -delta_ask
    mirrored_ask = -delta_bid
    mirrored_mark = torch.where(
        direction != 0,
        mark + (-2 * direction.to(torch.int64)) * 3 * 9 * 9,
        mark,
    )
    augmented_spread = torch.cat((spread, spread))
    augmented_mark = torch.cat((mark, mirrored_mark))
    augmented_bid = torch.cat((delta_bid, mirrored_bid))
    augmented_ask = torch.cat((delta_ask, mirrored_ask))
    cell = augmented_spread * MARK_COUNT + augmented_mark
    counts = torch.bincount(cell, minlength=8 * MARK_COUNT).to(torch.int64)
    offsets = torch.cumsum(counts, dim=0) - counts
    order = torch.argsort(cell, stable=True)
    return ExactTransitionCsr(
        order=order,
        offsets=offsets,
        counts=counts,
        delta_bid=augmented_bid,
        delta_ask=augmented_ask,
        observed_cells=counts.reshape(8, MARK_COUNT) > 0,
    )


def _eligible_origins(
    timestamps_ms: torch.Tensor,
    valid_interval: torch.Tensor,
    start_ms: int,
    stop_ms: int,
    horizon: int,
) -> torch.Tensor:
    candidates = torch.arange(
        0, valid_interval.numel() - horizon + 1,
        device=valid_interval.device,
        dtype=torch.int64,
    )
    invalid = (~valid_interval).to(torch.int64)
    cumulative = torch.cat(
        (torch.zeros(1, device=invalid.device, dtype=torch.int64), torch.cumsum(invalid, dim=0))
    )
    consecutive = cumulative[candidates + horizon] == cumulative[candidates]
    in_window = (timestamps_ms[candidates] >= start_ms) & (timestamps_ms[candidates] < stop_ms)
    return candidates[consecutive & in_window]


def _make_rollout_step(
    compile_mode: str, rejection_attempts: int = REJECTION_ATTEMPTS
):
    def step(
        bid_ticks: torch.Tensor,
        ask_ticks: torch.Tensor,
        efficient_price: torch.Tensor,
        mark_uniform: torch.Tensor,
        raw_uniform: torch.Tensor,
        waiting_uniform: torch.Tensor,
        brownian_normal: torch.Tensor,
        probabilities: torch.Tensor,
        correction: torch.Tensor,
        rates: torch.Tensor,
        alpha: torch.Tensor,
        sigma_x: torch.Tensor,
        direction: torch.Tensor,
        observed_cells: torch.Tensor,
        raw_order: torch.Tensor,
        raw_offsets: torch.Tensor,
        raw_counts: torch.Tensor,
        raw_delta_bid: torch.Tensor,
        raw_delta_ask: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        shape = bid_ticks.shape
        flat_bid = bid_ticks.reshape(-1)
        flat_ask = ask_ticks.reshape(-1)
        flat_x = efficient_price.reshape(-1)
        spread_ticks = flat_ask - flat_bid
        spread_bucket = torch.clamp(spread_ticks, min=1, max=8) - 1
        midpoint = (flat_bid + flat_ask).to(torch.float32) * 0.05
        gap = midpoint - flat_x
        directional_gap = torch.where(
            direction.unsqueeze(0) > 0,
            torch.clamp_min(-gap.unsqueeze(1), 0.0),
            torch.where(
                direction.unsqueeze(0) < 0,
                torch.clamp_min(gap.unsqueeze(1), 0.0),
                torch.zeros_like(gap).unsqueeze(1),
            ),
        )
        intensity = rates[spread_bucket].unsqueeze(1) * probabilities[spread_bucket]
        intensity = intensity + alpha * correction[spread_bucket] * directional_gap
        intensity = intensity * observed_cells[spread_bucket]
        cumulative = torch.cumsum(intensity, dim=1).contiguous()
        total = cumulative[:, -1]
        thresholds = (mark_uniform.reshape(rejection_attempts, -1).T * total.unsqueeze(1)).contiguous()
        candidate_mark = torch.searchsorted(cumulative, thresholds, right=False).T
        candidate_mark = torch.clamp_max(candidate_mark, MARK_COUNT - 1)
        candidate_cell = spread_bucket.unsqueeze(0) * MARK_COUNT + candidate_mark
        count = raw_counts[candidate_cell]
        rank = torch.floor(raw_uniform.reshape(rejection_attempts, -1) * count).to(torch.int64)
        rank = torch.minimum(torch.clamp_min(rank, 0), torch.clamp_min(count - 1, 0))
        transition_index = raw_order[raw_offsets[candidate_cell] + rank]
        candidate_bid_delta = raw_delta_bid[transition_index]
        candidate_ask_delta = raw_delta_ask[transition_index]
        candidate_spread = (
            spread_ticks.unsqueeze(0) + candidate_ask_delta - candidate_bid_delta
        )
        valid = candidate_spread > 0
        any_valid = torch.any(valid, dim=0)
        first_valid = torch.argmax(valid.to(torch.int64), dim=0)
        gather = first_valid.unsqueeze(0)
        selected_bid_delta = torch.gather(candidate_bid_delta, 0, gather).squeeze(0)
        selected_ask_delta = torch.gather(candidate_ask_delta, 0, gather).squeeze(0)
        selected_bid_delta = torch.where(any_valid, selected_bid_delta, torch.zeros_like(selected_bid_delta))
        selected_ask_delta = torch.where(any_valid, selected_ask_delta, torch.zeros_like(selected_ask_delta))
        waiting_time = -torch.log(torch.clamp_min(waiting_uniform.reshape(-1), 1e-7)) / total
        next_x = flat_x + sigma_x * torch.sqrt(waiting_time) * brownian_normal.reshape(-1)
        return (
            (flat_bid + selected_bid_delta).reshape(shape),
            (flat_ask + selected_ask_delta).reshape(shape),
            next_x.reshape(shape),
            waiting_time.reshape(shape),
            torch.count_nonzero(~any_valid),
        )

    return torch.compile(step, mode=compile_mode, fullgraph=True)


def _brownian_bridge_nodes(
    start_x: torch.Tensor,
    sigma_x: torch.Tensor,
    duration: float,
    depth: int,
    normals: torch.Tensor,
) -> torch.Tensor:
    """Generate a dyadic Brownian path conditional on its sampled endpoint."""
    endpoint = start_x + sigma_x * math.sqrt(duration) * normals[..., 0]
    nodes = torch.stack((start_x, endpoint), dim=-1)
    normal_offset = 1
    for level in range(depth):
        segment_count = 1 << level
        segment_duration = duration / float(segment_count)
        midpoint = 0.5 * (nodes[..., :-1] + nodes[..., 1:])
        midpoint = midpoint + sigma_x * math.sqrt(segment_duration / 4.0) * normals[
            ..., normal_offset : normal_offset + segment_count
        ]
        interleaved = torch.empty(
            (*nodes.shape[:-1], nodes.shape[-1] + midpoint.shape[-1]),
            device=nodes.device,
            dtype=nodes.dtype,
        )
        interleaved[..., 0::2] = nodes
        interleaved[..., 1::2] = midpoint
        nodes = interleaved
        normal_offset += segment_count
    return nodes


def _linear_gap_hazard(
    gap_start: torch.Tensor,
    gap_end: torch.Tensor,
    duration: torch.Tensor | float,
    base_rate: torch.Tensor,
    alpha: torch.Tensor,
    correction_down: torch.Tensor,
    correction_up: torch.Tensor,
) -> torch.Tensor:
    """Exact hazard on a linearly interpolated Brownian-bridge leaf."""
    primitive_start = 0.5 * correction_down * torch.square(torch.clamp_min(gap_start, 0.0))
    primitive_start = primitive_start - 0.5 * correction_up * torch.square(
        torch.clamp_min(-gap_start, 0.0)
    )
    primitive_end = 0.5 * correction_down * torch.square(torch.clamp_min(gap_end, 0.0))
    primitive_end = primitive_end - 0.5 * correction_up * torch.square(
        torch.clamp_min(-gap_end, 0.0)
    )
    delta = gap_end - gap_start
    nonconstant = torch.abs(delta) > 1e-7
    safe_delta = torch.where(nonconstant, delta, torch.ones_like(delta))
    average_correction = torch.where(
        nonconstant,
        (primitive_end - primitive_start) / safe_delta,
        torch.where(gap_start >= 0.0, correction_down * gap_start, -correction_up * gap_start),
    )
    return duration * (base_rate + alpha * average_correction)


def _make_continuous_wait_chunk(compile_mode: str):
    leaf_count = 1 << CONTINUOUS_BRIDGE_DEPTH
    leaf_duration = CONTINUOUS_CHUNK_SECONDS / leaf_count

    def wait_chunk(
        bid_ticks: torch.Tensor,
        ask_ticks: torch.Tensor,
        efficient_price: torch.Tensor,
        remaining_threshold: torch.Tensor,
        bridge_normals: torch.Tensor,
        rates: torch.Tensor,
        alpha: torch.Tensor,
        sigma_x: torch.Tensor,
        correction_down: torch.Tensor,
        correction_up: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        spread_bucket = torch.clamp(ask_ticks - bid_ticks, min=1, max=8) - 1
        midpoint = (bid_ticks + ask_ticks).to(torch.float32) * 0.05
        x_path = _brownian_bridge_nodes(
            efficient_price,
            sigma_x,
            CONTINUOUS_CHUNK_SECONDS,
            CONTINUOUS_BRIDGE_DEPTH,
            bridge_normals,
        )
        gap_path = midpoint.unsqueeze(-1) - x_path
        base = rates[spread_bucket].unsqueeze(-1)
        down = correction_down[spread_bucket].unsqueeze(-1)
        up = correction_up[spread_bucket].unsqueeze(-1)
        leaf_hazard = _linear_gap_hazard(
            gap_path[..., :-1],
            gap_path[..., 1:],
            leaf_duration,
            base,
            alpha,
            down,
            up,
        )
        cumulative = torch.cumsum(leaf_hazard, dim=-1).contiguous()
        hit = cumulative[..., -1] >= remaining_threshold
        leaf = torch.searchsorted(
            cumulative,
            remaining_threshold.unsqueeze(-1),
            right=False,
        ).squeeze(-1)
        leaf = torch.clamp(leaf, 0, leaf_count - 1)
        previous_leaf = torch.clamp_min(leaf - 1, 0)
        before = torch.gather(cumulative, -1, previous_leaf.unsqueeze(-1)).squeeze(-1)
        before = torch.where(leaf > 0, before, torch.zeros_like(before))
        remaining_leaf = torch.clamp_min(remaining_threshold - before, 0.0)
        gap_start = torch.gather(gap_path, -1, leaf.unsqueeze(-1)).squeeze(-1)
        gap_end = torch.gather(gap_path, -1, (leaf + 1).unsqueeze(-1)).squeeze(-1)
        x_start = torch.gather(x_path, -1, leaf.unsqueeze(-1)).squeeze(-1)
        x_end = torch.gather(x_path, -1, (leaf + 1).unsqueeze(-1)).squeeze(-1)
        fraction_low = torch.zeros_like(remaining_threshold)
        fraction_high = torch.ones_like(remaining_threshold)
        selected_base = rates[spread_bucket]
        selected_down = correction_down[spread_bucket]
        selected_up = correction_up[spread_bucket]
        for _ in range(CONTINUOUS_CROSSING_BISECTIONS):
            fraction = 0.5 * (fraction_low + fraction_high)
            partial_gap = gap_start + fraction * (gap_end - gap_start)
            partial_hazard = _linear_gap_hazard(
                gap_start,
                partial_gap,
                leaf_duration * fraction,
                selected_base,
                alpha,
                selected_down,
                selected_up,
            )
            left = partial_hazard < remaining_leaf
            fraction_low = torch.where(left, fraction, fraction_low)
            fraction_high = torch.where(left, fraction_high, fraction)
        fraction = 0.5 * (fraction_low + fraction_high)
        event_x = x_start + fraction * (x_end - x_start)
        event_time = (leaf.to(torch.float32) + fraction) * leaf_duration
        return (
            hit,
            event_x,
            event_time,
            x_path[..., -1],
            torch.clamp_min(remaining_threshold - cumulative[..., -1], 0.0),
        )

    return torch.compile(wait_chunk, mode=compile_mode, fullgraph=True)


def _make_mark_decoder_step(
    compile_mode: str, rejection_attempts: int = REJECTION_ATTEMPTS
):
    def decode(
        bid_ticks: torch.Tensor,
        ask_ticks: torch.Tensor,
        efficient_price: torch.Tensor,
        mark_uniform: torch.Tensor,
        raw_uniform: torch.Tensor,
        probabilities: torch.Tensor,
        correction: torch.Tensor,
        rates: torch.Tensor,
        alpha: torch.Tensor,
        direction: torch.Tensor,
        observed_cells: torch.Tensor,
        raw_order: torch.Tensor,
        raw_offsets: torch.Tensor,
        raw_counts: torch.Tensor,
        raw_delta_bid: torch.Tensor,
        raw_delta_ask: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        shape = bid_ticks.shape
        flat_bid = bid_ticks.reshape(-1)
        flat_ask = ask_ticks.reshape(-1)
        flat_x = efficient_price.reshape(-1)
        spread_ticks = flat_ask - flat_bid
        spread_bucket = torch.clamp(spread_ticks, min=1, max=8) - 1
        midpoint = (flat_bid + flat_ask).to(torch.float32) * 0.05
        gap = midpoint - flat_x
        directional_gap = torch.where(
            direction.unsqueeze(0) > 0,
            torch.clamp_min(-gap.unsqueeze(1), 0.0),
            torch.where(
                direction.unsqueeze(0) < 0,
                torch.clamp_min(gap.unsqueeze(1), 0.0),
                torch.zeros_like(gap).unsqueeze(1),
            ),
        )
        intensity = rates[spread_bucket].unsqueeze(1) * probabilities[spread_bucket]
        intensity = intensity + alpha * correction[spread_bucket] * directional_gap
        intensity = intensity * observed_cells[spread_bucket]
        cumulative = torch.cumsum(intensity, dim=1).contiguous()
        total = cumulative[:, -1]
        thresholds = (
            mark_uniform.reshape(rejection_attempts, -1).T * total.unsqueeze(1)
        ).contiguous()
        candidate_mark = torch.searchsorted(cumulative, thresholds, right=False).T
        candidate_mark = torch.clamp_max(candidate_mark, MARK_COUNT - 1)
        candidate_cell = spread_bucket.unsqueeze(0) * MARK_COUNT + candidate_mark
        count = raw_counts[candidate_cell]
        rank = torch.floor(raw_uniform.reshape(rejection_attempts, -1) * count).to(torch.int64)
        rank = torch.minimum(torch.clamp_min(rank, 0), torch.clamp_min(count - 1, 0))
        transition_index = raw_order[raw_offsets[candidate_cell] + rank]
        candidate_bid_delta = raw_delta_bid[transition_index]
        candidate_ask_delta = raw_delta_ask[transition_index]
        candidate_spread = spread_ticks.unsqueeze(0) + candidate_ask_delta - candidate_bid_delta
        valid = candidate_spread > 0
        any_valid = torch.any(valid, dim=0)
        first_valid = torch.argmax(valid.to(torch.int64), dim=0)
        gather = first_valid.unsqueeze(0)
        bid_delta = torch.gather(candidate_bid_delta, 0, gather).squeeze(0)
        ask_delta = torch.gather(candidate_ask_delta, 0, gather).squeeze(0)
        bid_delta = torch.where(any_valid, bid_delta, torch.zeros_like(bid_delta))
        ask_delta = torch.where(any_valid, ask_delta, torch.zeros_like(ask_delta))
        return (
            (flat_bid + bid_delta).reshape(shape),
            (flat_ask + ask_delta).reshape(shape),
            torch.count_nonzero(~any_valid),
        )

    return torch.compile(decode, mode=compile_mode, fullgraph=True)


def _rollout(
    *,
    origins: torch.Tensor,
    bid_ticks: torch.Tensor,
    ask_ticks: torch.Tensor,
    state_mean: torch.Tensor,
    state_variance: torch.Tensor,
    probabilities: torch.Tensor,
    correction: torch.Tensor,
    rates: torch.Tensor,
    alpha: torch.Tensor,
    sigma_x: torch.Tensor,
    transitions: ExactTransitionCsr,
    horizon: int,
    paths: int,
    seed: int,
    compile_mode: str,
) -> tuple[dict[str, torch.Tensor], int]:
    device = bid_ticks.device
    origin_count = origins.numel()
    generator = torch.Generator(device=device).manual_seed(seed)
    current_bid = bid_ticks[origins].unsqueeze(1).expand(-1, paths).clone()
    current_ask = ask_ticks[origins].unsqueeze(1).expand(-1, paths).clone()
    current_x = state_mean[origins].unsqueeze(1) + torch.sqrt(
        torch.clamp_min(state_variance[origins], 1e-12)
    ).unsqueeze(1) * torch.randn(
        (origin_count, paths), device=device, dtype=torch.float32, generator=generator
    )
    bids = [current_bid]
    asks = [current_ask]
    event_times = [torch.zeros((origin_count, paths), device=device, dtype=torch.float32)]
    direction = mark_metadata(device)[0]
    step = _make_rollout_step(compile_mode)
    fallback_count = 0
    for _ in range(horizon):
        torch.compiler.cudagraph_mark_step_begin()
        random_shape = (REJECTION_ATTEMPTS, origin_count, paths)
        mark_uniform = torch.rand(random_shape, device=device, generator=generator)
        raw_uniform = torch.rand(random_shape, device=device, generator=generator)
        waiting_uniform = torch.rand((origin_count, paths), device=device, generator=generator)
        brownian_normal = torch.randn(
            (origin_count, paths), device=device, dtype=torch.float32, generator=generator
        )
        current_bid, current_ask, current_x, waiting_time, fallbacks = step(
            current_bid,
            current_ask,
            current_x,
            mark_uniform,
            raw_uniform,
            waiting_uniform,
            brownian_normal,
            probabilities,
            correction,
            rates,
            alpha,
            sigma_x,
            direction,
            transitions.observed_cells,
            transitions.order,
            transitions.offsets,
            transitions.counts,
            transitions.delta_bid,
            transitions.delta_ask,
        )
        current_bid = current_bid.clone()
        current_ask = current_ask.clone()
        current_x = current_x.clone()
        waiting_time = waiting_time.clone()
        bids.append(current_bid)
        asks.append(current_ask)
        event_times.append(event_times[-1] + waiting_time)
        fallback_count += int(fallbacks)
    bid_tick_path = torch.stack(bids, dim=1)
    ask_tick_path = torch.stack(asks, dim=1)
    minimum_spread_ticks = torch.min(ask_tick_path - bid_tick_path)
    bid_path = bid_tick_path.to(torch.float32) * 0.1
    ask_path = ask_tick_path.to(torch.float32) * 0.1
    midpoint_path = 0.5 * (bid_path + ask_path)
    time_path = torch.stack(event_times, dim=1)
    return {
        "bid_mean": bid_path.mean(dim=2),
        "ask_mean": ask_path.mean(dim=2),
        "midpoint_mean": midpoint_path.mean(dim=2),
        "bid_q10": torch.quantile(bid_path, 0.1, dim=2),
        "bid_q90": torch.quantile(bid_path, 0.9, dim=2),
        "ask_q10": torch.quantile(ask_path, 0.1, dim=2),
        "ask_q90": torch.quantile(ask_path, 0.9, dim=2),
        "midpoint_q10": torch.quantile(midpoint_path, 0.1, dim=2),
        "midpoint_q90": torch.quantile(midpoint_path, 0.9, dim=2),
        "model_event_time_mean_seconds": time_path.mean(dim=2),
        "minimum_simulated_spread_ticks": minimum_spread_ticks,
    }, fallback_count


def _rollout_continuous(
    *,
    origins: torch.Tensor,
    bid_ticks: torch.Tensor,
    ask_ticks: torch.Tensor,
    state_mean: torch.Tensor,
    state_variance: torch.Tensor,
    probabilities: torch.Tensor,
    correction: torch.Tensor,
    rates: torch.Tensor,
    alpha: torch.Tensor,
    sigma_x: torch.Tensor,
    transitions: ExactTransitionCsr,
    horizon: int,
    paths: int,
    seed: int,
    compile_mode: str,
) -> tuple[dict[str, torch.Tensor], int]:
    device = bid_ticks.device
    origin_count = origins.numel()
    generator = torch.Generator(device=device).manual_seed(seed)
    current_bid = bid_ticks[origins].unsqueeze(1).expand(-1, paths).clone()
    current_ask = ask_ticks[origins].unsqueeze(1).expand(-1, paths).clone()
    current_x = state_mean[origins].unsqueeze(1) + torch.sqrt(
        torch.clamp_min(state_variance[origins], 1e-12)
    ).unsqueeze(1) * torch.randn(
        (origin_count, paths), device=device, dtype=torch.float32, generator=generator
    )
    bids = [current_bid]
    asks = [current_ask]
    event_times = [torch.zeros((origin_count, paths), device=device, dtype=torch.float32)]
    direction = mark_metadata(device)[0]
    supported = transitions.observed_cells
    clock_rates = rates * torch.sum(probabilities * supported, dim=-1)
    clock_correction_down = torch.sum(
        correction * supported * (direction < 0).unsqueeze(0), dim=-1
    )
    clock_correction_up = torch.sum(
        correction * supported * (direction > 0).unsqueeze(0), dim=-1
    )
    wait_chunk = _make_continuous_wait_chunk(compile_mode)
    decode = _make_mark_decoder_step(compile_mode)
    fallback_count = 0
    unresolved_total = 0
    bridge_normal_count = 1 << CONTINUOUS_BRIDGE_DEPTH
    for _ in range(horizon):
        threshold = -torch.log(
            torch.clamp_min(
                torch.rand(
                    (origin_count, paths), device=device, dtype=torch.float32, generator=generator
                ),
                1e-7,
            )
        )
        waiting_time = torch.zeros_like(threshold)
        active = torch.ones_like(threshold, dtype=torch.bool)
        event_x = current_x.clone()
        path_x = current_x
        for _ in range(CONTINUOUS_MAX_CHUNKS):
            torch.compiler.cudagraph_mark_step_begin()
            bridge_normals = torch.randn(
                (origin_count, paths, bridge_normal_count),
                device=device,
                dtype=torch.float32,
                generator=generator,
            )
            hit, candidate_x, candidate_time, chunk_end_x, remaining = wait_chunk(
                current_bid,
                current_ask,
                path_x,
                threshold,
                bridge_normals,
                clock_rates,
                alpha,
                sigma_x,
                clock_correction_down,
                clock_correction_up,
            )
            new_hit = active & hit
            event_x = torch.where(new_hit, candidate_x, event_x)
            waiting_time = torch.where(
                new_hit,
                waiting_time + candidate_time,
                torch.where(active, waiting_time + CONTINUOUS_CHUNK_SECONDS, waiting_time),
            )
            path_x = torch.where(active & ~hit, chunk_end_x, path_x)
            threshold = torch.where(active & ~hit, remaining, threshold)
            active = active & ~hit
            if not bool(torch.any(active)):
                break
        unresolved = int(torch.count_nonzero(active))
        unresolved_total += unresolved
        if unresolved:
            event_x = torch.where(active, path_x, event_x)
        random_shape = (REJECTION_ATTEMPTS, origin_count, paths)
        mark_uniform = torch.rand(random_shape, device=device, generator=generator)
        raw_uniform = torch.rand(random_shape, device=device, generator=generator)
        torch.compiler.cudagraph_mark_step_begin()
        current_bid, current_ask, fallbacks = decode(
            current_bid,
            current_ask,
            event_x,
            mark_uniform,
            raw_uniform,
            probabilities,
            correction,
            rates,
            alpha,
            direction,
            transitions.observed_cells,
            transitions.order,
            transitions.offsets,
            transitions.counts,
            transitions.delta_bid,
            transitions.delta_ask,
        )
        current_bid = current_bid.clone()
        current_ask = current_ask.clone()
        current_x = event_x.clone()
        waiting_time = waiting_time.clone()
        bids.append(current_bid)
        asks.append(current_ask)
        event_times.append(event_times[-1] + waiting_time)
        fallback_count += int(fallbacks)
    bid_tick_path = torch.stack(bids, dim=1)
    ask_tick_path = torch.stack(asks, dim=1)
    minimum_spread_ticks = torch.min(ask_tick_path - bid_tick_path)
    bid_path = bid_tick_path.to(torch.float32) * 0.1
    ask_path = ask_tick_path.to(torch.float32) * 0.1
    midpoint_path = 0.5 * (bid_path + ask_path)
    time_path = torch.stack(event_times, dim=1)
    return {
        "bid_mean": bid_path.mean(dim=2),
        "ask_mean": ask_path.mean(dim=2),
        "midpoint_mean": midpoint_path.mean(dim=2),
        "bid_q10": torch.quantile(bid_path, 0.1, dim=2),
        "bid_q90": torch.quantile(bid_path, 0.9, dim=2),
        "ask_q10": torch.quantile(ask_path, 0.1, dim=2),
        "ask_q90": torch.quantile(ask_path, 0.9, dim=2),
        "midpoint_q10": torch.quantile(midpoint_path, 0.1, dim=2),
        "midpoint_q90": torch.quantile(midpoint_path, 0.9, dim=2),
        "model_event_time_mean_seconds": time_path.mean(dim=2),
        "minimum_simulated_spread_ticks": minimum_spread_ticks,
        "unresolved_hazard_thresholds": unresolved_total,
    }, fallback_count


def _forecast(
    config_path: Path,
    run_directory: Path,
    start_ms: int,
    stop_ms: int,
    horizon: int,
    paths: int,
    seed: int,
) -> dict[str, Any]:
    spec = load_runspec(config_path.resolve())
    swap, spot, _ = _load_verified_p6_payloads(spec)
    days, future_accesses = _prepare_empirical_marked_days(spec, swap, spot)
    state_path = run_directory / "state" / "december_filter.pt"
    state = torch.load(
        state_path,
        map_location=spec.values["numerics"]["compute_device"],
        weights_only=True,
    )
    date = str(state["date"])
    day = days[date]
    device = day.timestamps_ms.device
    train_dates, rates, alpha, sigma_x, parameter_digest = _load_full_fold(
        run_directory, date, device
    )
    train = _stack_train_intervals(days, train_dates)
    probabilities, correction, _, _, _, _ = _empirical_probability_tables(
        train, float(spec.values["model"]["dirichlet_smoothing_beta"]), "full"
    )
    transitions = _exact_transition_csr(train)
    origins = _eligible_origins(
        day.timestamps_ms, day.valid_interval, start_ms, stop_ms, horizon
    )
    if origins.numel() == 0:
        raise RuntimeError("plot window has no origin with a complete healthy forecast horizon")
    rollout = _rollout_continuous if spec.experiment_id == "EMP-MARK-CT-001" else _rollout
    rollout_arguments: dict[str, Any] = {
        "origins": origins,
        "bid_ticks": day.bid_ticks,
        "ask_ticks": day.ask_ticks,
        "state_mean": state["filtered_efficient_price"].to(device),
        "state_variance": state["posterior_variance"].to(device),
        "probabilities": probabilities,
        "correction": correction,
        "rates": rates,
        "alpha": alpha,
        "sigma_x": sigma_x,
        "transitions": transitions,
        "horizon": horizon,
        "paths": paths,
        "seed": seed,
        "compile_mode": str(spec.values["numerics"]["compile_mode"]),
    }
    paths_out, fallback_count = rollout(**rollout_arguments)
    horizon_index = torch.arange(horizon + 1, device=device).unsqueeze(0)
    actual_index = origins.unsqueeze(1) + horizon_index
    actual_bid = day.bid_ticks[actual_index].to(torch.float32) * 0.1
    actual_ask = day.ask_ticks[actual_index].to(torch.float32) * 0.1
    actual_midpoint = 0.5 * (actual_bid + actual_ask)
    actual_timestamp = day.timestamps_ms[actual_index]
    actual_elapsed_seconds = (actual_timestamp - start_ms).to(torch.float64) / 1000.0
    actual = {"bid": actual_bid, "ask": actual_ask, "midpoint": actual_midpoint}
    metrics: dict[str, Any] = {
        "origin_count": int(origins.numel()),
        "horizon_events": horizon,
        "paths_per_origin": paths,
        "future_timestamp_accesses": future_accesses,
        "rejection_fallback_count": fallback_count,
        "minimum_simulated_spread_ticks": int(paths_out["minimum_simulated_spread_ticks"]),
        "hazard_clock": (
            "continuous_brownian_bridge_integrated"
            if spec.experiment_id == "EMP-MARK-CT-001"
            else "frozen_endpoint_exponential"
        ),
        "unresolved_hazard_thresholds": int(paths_out.get("unresolved_hazard_thresholds", 0)),
    }
    horizon_rows = []
    for step_index in range(1, horizon + 1):
        row: dict[str, Any] = {"horizon_events": step_index}
        for name in ("bid", "ask", "midpoint"):
            model_error = torch.abs(paths_out[f"{name}_mean"][:, step_index] - actual[name][:, step_index])
            persistence_error = torch.abs(actual[name][:, 0] - actual[name][:, step_index])
            row[f"{name}_model_mae_usdt"] = float(model_error.mean())
            row[f"{name}_persistence_mae_usdt"] = float(persistence_error.mean())
        row["model_mean_elapsed_seconds"] = float(
            paths_out["model_event_time_mean_seconds"][:, step_index].mean()
        )
        row["actual_mean_elapsed_seconds"] = float(
            (actual_timestamp[:, step_index] - actual_timestamp[:, 0]).to(torch.float64).mean()
            / 1000.0
        )
        horizon_rows.append(row)
    metrics.update(
        midpoint_h1_model_mae_usdt=horizon_rows[0]["midpoint_model_mae_usdt"],
        midpoint_h1_persistence_mae_usdt=horizon_rows[0]["midpoint_persistence_mae_usdt"],
        midpoint_final_horizon_model_mae_usdt=horizon_rows[-1]["midpoint_model_mae_usdt"],
        midpoint_final_horizon_persistence_mae_usdt=horizon_rows[-1]["midpoint_persistence_mae_usdt"],
        final_horizon_model_mean_elapsed_seconds=horizon_rows[-1]["model_mean_elapsed_seconds"],
        final_horizon_actual_mean_elapsed_seconds=horizon_rows[-1]["actual_mean_elapsed_seconds"],
    )
    return {
        "date": date,
        "origins": origins,
        "paths": paths_out,
        "actual": actual,
        "actual_elapsed_seconds": actual_elapsed_seconds,
        "horizon_rows": horizon_rows,
        "metrics": metrics,
        "train_dates": train_dates,
        "parameter_digest_sha256": parameter_digest,
        "state_path": state_path,
        "experiment_id": spec.experiment_id,
    }


def _trajectory_segments(x: torch.Tensor, y: torch.Tensor) -> list[torch.Tensor]:
    return [torch.stack((x[index], y[index]), dim=1) for index in range(x.shape[0])]


def _render_trajectories(result: dict[str, Any], path: Path, start_text: str) -> None:
    actual_x = result["actual_elapsed_seconds"].to("cpu")
    forecast_x = actual_x[:, :1] + result["paths"]["model_event_time_mean_seconds"].to("cpu")
    figure, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True, constrained_layout=True)
    for axis, name, title in zip(
        axes, ("bid", "ask", "midpoint"), ("Bid", "Ask", "Midpoint"), strict=True
    ):
        forecast = result["paths"][f"{name}_mean"].to("cpu")
        segments = _trajectory_segments(forecast_x, forecast)
        axis.add_collection(
            LineCollection(segments, colors="#ea580c", linewidths=0.8, alpha=0.22)
        )
        day_x = actual_x[:, 0]
        day_y = result["actual"][name][:, 0].to("cpu")
        tail_x = actual_x[-1, 1:]
        tail_y = result["actual"][name][-1, 1:].to("cpu")
        axis.plot(
            torch.cat((day_x, tail_x)),
            torch.cat((day_y, tail_y)),
            color="#172554",
            linewidth=1.5,
            label="Actual BBO",
        )
        axis.scatter(day_x, day_y, s=6, color="#172554", alpha=0.7)
        axis.autoscale()
        axis.set_title(title)
        axis.set_ylabel("USDT")
        axis.grid(alpha=0.25)
        axis.legend(
            handles=[
                plt.Line2D([], [], color="#172554", label="Actual BBO"),
                plt.Line2D(
                    [], [], color="#ea580c", alpha=0.7,
                    label=f"Conditional mean rollout ({result['metrics']['horizon_events']} events)",
                ),
            ],
            loc="upper left",
        )
    axes[-1].set_xlabel("Seconds after window start; forecast values do not use future BBO")
    metrics = result["metrics"]
    figure.suptitle(
        f"{result['experiment_id']} free-running forecasts from every held-out BBO point\n"
        f"{start_text}; {metrics['origin_count']} origins x {metrics['paths_per_origin']} paths; "
        f"midpoint MAE model/persistence: h=1 "
        f"{metrics['midpoint_h1_model_mae_usdt']:.3f}/"
        f"{metrics['midpoint_h1_persistence_mae_usdt']:.3f}, h={metrics['horizon_events']} "
        f"{metrics['midpoint_final_horizon_model_mae_usdt']:.3f}/"
        f"{metrics['midpoint_final_horizon_persistence_mae_usdt']:.3f} USDT; "
        f"h={metrics['horizon_events']} time model/actual "
        f"{metrics['final_horizon_model_mean_elapsed_seconds']:.2f}/"
        f"{metrics['final_horizon_actual_mean_elapsed_seconds']:.2f} s",
        fontsize=12,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _render_horizon_errors(
    rows: list[dict[str, Any]], path: Path, model_label: str
) -> None:
    horizon = torch.tensor([row["horizon_events"] for row in rows])
    figure, axes = plt.subplots(1, 3, figsize=(16, 5), sharex=True, constrained_layout=True)
    for axis, name, title in zip(
        axes, ("bid", "ask", "midpoint"), ("Bid", "Ask", "Midpoint"), strict=True
    ):
        model = torch.tensor([row[f"{name}_model_mae_usdt"] for row in rows])
        persistence = torch.tensor([row[f"{name}_persistence_mae_usdt"] for row in rows])
        axis.plot(horizon, model, marker="o", color="#ea580c", label=model_label)
        axis.plot(horizon, persistence, marker="o", color="#64748b", label="Persistence")
        axis.set_title(title)
        axis.set_xlabel("Forecast horizon, BBO events")
        axis.set_ylabel("MAE, USDT")
        axis.grid(alpha=0.25)
        axis.legend()
    figure.suptitle("Several-event forecast error by horizon", fontsize=13)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _arguments(argv)
    start = datetime.fromisoformat(arguments.start_utc.replace("Z", "+00:00")).astimezone(UTC)
    start_ms = int(start.timestamp() * 1000)
    stop_ms = start_ms + arguments.window_minutes * 60_000
    run_directory = arguments.run_directory.resolve()
    output_directory = arguments.output_directory
    if output_directory is None:
        output_directory = (
            REPOSITORY_ROOT
            / "outputs"
            / (
                "P6C-MULTISTEP-VIZ"
                if arguments.config.resolve().name == "emp_mark_ct_001.toml"
                else "P6M-MULTISTEP-VIZ"
            )
            / (
                f"{start.strftime('%Y%m%dT%H%M%SZ')}-{arguments.window_minutes}min-"
                f"h{arguments.horizon_events}-p{arguments.paths}"
            )
        )
    output_directory = output_directory.resolve()
    output_directory.mkdir(parents=True, exist_ok=False)
    result = _forecast(
        arguments.config,
        run_directory,
        start_ms,
        stop_ms,
        arguments.horizon_events,
        arguments.paths,
        arguments.seed,
    )
    trajectory_path = output_directory / "multistep-trajectories.png"
    error_path = output_directory / "horizon-mae.png"
    _render_trajectories(result, trajectory_path, arguments.start_utc)
    _render_horizon_errors(
        result["horizon_rows"], error_path, f"{result['experiment_id']} rollout"
    )
    write_csv(
        output_directory / "horizon-metrics.csv",
        list(result["horizon_rows"][0]),
        result["horizon_rows"],
    )
    atomic_write_json(
        output_directory / "provenance.json",
        {
            "schema_version": (
                "p6c-multistep-visualization-v1"
                if result["experiment_id"] == "EMP-MARK-CT-001"
                else "p6m-multistep-visualization-v1"
            ),
            "role": "descriptive_only_not_an_acceptance_artifact",
            "forecast": "free-running Monte Carlo event-index rollout from each causal posterior",
            "initial_latent_state": "Gaussian moment approximation from persisted causal posterior mean and variance",
            "horizontal_alignment": "each rollout starts at the real origin timestamp and advances by model-expected event times; realized future timestamps are used only for error evaluation",
            "raw_transition_decoder": "train-only exact deltas conditional on spread bucket and symmetrized mark; positive-spread rejection",
            "hazard_clock": result["metrics"]["hazard_clock"],
            "continuous_clock_numerics": (
                {
                    "chunk_seconds": CONTINUOUS_CHUNK_SECONDS,
                    "brownian_bridge_depth": CONTINUOUS_BRIDGE_DEPTH,
                    "leaf_seconds": CONTINUOUS_CHUNK_SECONDS
                    / float(1 << CONTINUOUS_BRIDGE_DEPTH),
                    "crossing_bisections": CONTINUOUS_CROSSING_BISECTIONS,
                    "maximum_chunks": CONTINUOUS_MAX_CHUNKS,
                }
                if result["experiment_id"] == "EMP-MARK-CT-001"
                else None
            ),
            "date": result["date"],
            "start_utc": arguments.start_utc,
            "window_minutes": arguments.window_minutes,
            "horizon_events": arguments.horizon_events,
            "paths_per_origin": arguments.paths,
            "seed": arguments.seed,
            "train_dates": result["train_dates"],
            "metrics": result["metrics"],
            "source_run_directory": str(run_directory.relative_to(REPOSITORY_ROOT)),
            "source_manifest_sha256": sha256_file(run_directory / "manifest.json"),
            "source_state_sha256": sha256_file(result["state_path"]),
            "parameter_digest_sha256": result["parameter_digest_sha256"],
            "config_path": str(arguments.config.resolve().relative_to(REPOSITORY_ROOT)),
            "config_sha256": sha256_file(arguments.config.resolve()),
            "visualization_script_sha256": sha256_file(Path(__file__).resolve()),
            "trajectory_plot": trajectory_path.name,
            "error_plot": error_path.name,
        },
    )
    print(json.dumps({"output_directory": str(output_directory), **result["metrics"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
