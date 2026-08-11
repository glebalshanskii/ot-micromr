from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from ot_micromr.figure4 import CalibrationRow, Figure4Replication, _policy_definition
from ot_micromr.figure4_market import Figure4MarketTrace


@dataclass(frozen=True, slots=True)
class CudaEvaluation:
    replications: tuple[Figure4Replication, ...]
    benchmark: Mapping[str, Any]


def _padded_segment(
    traces: Sequence[Figure4MarketTrace], start_times: Sequence[float], end_times: Sequence[float]
) -> dict[str, np.ndarray]:
    slices: list[slice] = []
    lengths: list[int] = []
    for trace, start, end in zip(traces, start_times, end_times, strict=True):
        first = int(np.searchsorted(trace.left_time_seconds, start, side="left"))
        last = int(np.searchsorted(trace.left_time_seconds, end, side="left"))
        slices.append(slice(first, last))
        lengths.append(last - first)
    width = max(lengths, default=0)
    batch = len(traces)
    names = (
        "left_time_seconds",
        "step_seconds",
        "left_gap_price",
        "pre_event_gap_price",
        "post_event_gap_price",
        "left_mid_price",
        "post_event_mid_price",
        "left_spread_price",
        "post_event_spread_price",
    )
    output = {
        name: np.zeros((batch, width), dtype=np.float32)
        for name in names
    }
    output["book_event"] = np.zeros((batch, width), dtype=np.bool_)
    output["valid"] = np.zeros((batch, width), dtype=np.bool_)
    for row, (trace, selected, length) in enumerate(zip(traces, slices, lengths, strict=True)):
        if length == 0:
            continue
        for name in names:
            output[name][row, :length] = getattr(trace, name)[selected].astype(
                np.float32, copy=False
            )
        output["book_event"][row, :length] = trace.book_event[selected]
        output["valid"][row, :length] = True
    return output


def _bridge_seed(
    values: Mapping[str, Any], row_index: int, epsilon: float, seeds: Sequence[int]
) -> int:
    epsilon_code = int(round(epsilon * 1_000_000))
    base_seed = int(values["seed_policy"]["bridge_seed"])
    sequence = np.random.SeedSequence([base_seed, row_index, epsilon_code, *seeds, 4])
    return int(sequence.generate_state(1, dtype=np.uint64)[0] % (2**63 - 1))


def _evaluate_row_cuda(
    values: Mapping[str, Any],
    calibration: CalibrationRow,
    traces: Sequence[Figure4MarketTrace],
    *,
    chunk_steps: int,
    deadline_monotonic: float | None,
) -> tuple[tuple[Figure4Replication, ...], dict[str, Any]]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA Figure 4 backend requested without an available CUDA device")
    if not traces:
        return (), {}
    epsilon = traces[0].epsilon
    if any(trace.row_index != calibration.row_index or trace.epsilon != epsilon for trace in traces):
        raise ValueError("CUDA row batch must share response row and epsilon")
    thresholds64, labels, multipliers = _policy_definition(values, calibration)
    thresholds = torch.as_tensor(thresholds64, dtype=torch.float32, device="cuda")
    batch = len(traces)
    policy_count = thresholds.numel()
    thresholds_batch = thresholds.unsqueeze(0).expand(batch, -1).contiguous()
    simulation = values["simulation"]
    market_burn = float(simulation["market_burn_in_reversion_times"]) / calibration.alpha_per_second
    measurement_start = market_burn + float(
        simulation["strategy_burn_in_reversion_times"]
    ) / calibration.alpha_per_second
    total_end = measurement_start + float(simulation["horizon_reversion_times"]) / calibration.alpha_per_second
    pre = _padded_segment(
        traces,
        [market_burn] * batch,
        [measurement_start] * batch,
    )
    measured = _padded_segment(
        traces,
        [measurement_start] * batch,
        [total_end + 1.0] * batch,
    )
    probability_cutoff = float(values["numerics"]["bridge_probability_cutoff"])
    delta_price = float(values["model"]["delta_price"])
    sigma2 = float(values["model"]["sigma_x_price_per_sqrt_second"]) ** 2

    def kernel(
        left_gap: Any,
        pre_gap: Any,
        post_gap: Any,
        left_mid: Any,
        post_mid: Any,
        left_spread: Any,
        post_spread: Any,
        left_time: Any,
        step: Any,
        book_event: Any,
        valid: Any,
        local_thresholds: Any,
        random_upper: Any,
        random_lower: Any,
        initial_side: Any,
        initial_count: Any,
        initial_reward: Any,
        initial_overshoot: Any,
        initial_open_count: Any,
        initial_bridge_count: Any,
        initial_jump_count: Any,
        initial_cash: Any,
        initial_first_time: Any,
        initial_last_time: Any,
    ) -> tuple[Any, ...]:
        theta = local_thresholds[:, None, :]
        lg = left_gap[:, :, None]
        pg = pre_gap[:, :, None]
        variance = torch.clamp_min(step[:, :, None] * sigma2, 1.0e-30)
        upper_deterministic = (lg >= theta) | (pg >= theta)
        lower_deterministic = (lg <= -theta) | (pg <= -theta)
        upper_valid = (lg < theta) & (pg < theta)
        lower_valid = (lg > -theta) & (pg > -theta)
        p_upper_raw = torch.where(
            upper_valid,
            torch.exp(-2.0 * (theta - lg) * (theta - pg) / variance),
            0.0,
        ).clamp_(0.0, 1.0)
        p_lower_raw = torch.where(
            lower_valid,
            torch.exp(-2.0 * (lg + theta) * (pg + theta) / variance),
            0.0,
        ).clamp_(0.0, 1.0)
        p_upper = torch.where(p_upper_raw >= probability_cutoff, p_upper_raw, 0.0)
        p_lower = torch.where(p_lower_raw >= probability_cutoff, p_lower_raw, 0.0)
        upper_hit = upper_deterministic | (random_upper < p_upper)
        lower_hit = lower_deterministic | (random_lower < p_lower)
        both = upper_hit & lower_hit
        prefer_upper = pg >= lg
        upper_selected = upper_hit & (~both | prefer_upper)
        lower_selected = lower_hit & (~both | ~prefer_upper)
        diffusion_signal = torch.zeros_like(random_upper, dtype=torch.int8)
        diffusion_signal = torch.where(upper_selected, -1, diffusion_signal)
        diffusion_signal = torch.where(lower_selected, 1, diffusion_signal)
        diffusion_signal = torch.where(valid[:, :, None], diffusion_signal, 0)

        post = post_gap[:, :, None]
        jump_signal = torch.zeros_like(diffusion_signal)
        jump_signal = torch.where(post >= theta, -1, jump_signal)
        jump_signal = torch.where(post <= -theta, 1, jump_signal)
        jump_signal = torch.where(
            valid[:, :, None] & book_event[:, :, None], jump_signal, 0
        )
        signals = torch.stack((diffusion_signal, jump_signal), dim=2).reshape(
            left_gap.shape[0], -1, local_thresholds.shape[1]
        )
        event_valid = valid[:, :, None].expand(-1, -1, policy_count)
        event_valid = torch.stack((event_valid, event_valid), dim=2).reshape_as(signals)

        prefixed = torch.cat((initial_side[:, None, :], signals), dim=1)
        observed = prefixed != 0
        event_index = torch.arange(prefixed.shape[1], device=prefixed.device).view(1, -1, 1)
        last_index = torch.cummax(torch.where(observed, event_index, -1), dim=1).values
        held = torch.gather(prefixed, 1, torch.clamp_min(last_index, 0))
        held = torch.where(last_index >= 0, held, 0)
        previous = held[:, :-1, :]
        fills = (signals != 0) & (signals != previous) & event_valid
        final_side = held[:, -1, :]

        chosen_boundary = torch.where(diffusion_signal == -1, theta, -theta)
        denominator = pg - lg
        deterministic_chosen = torch.where(
            diffusion_signal == -1, upper_deterministic, lower_deterministic
        )
        fraction = torch.where(
            deterministic_chosen & (torch.abs(denominator) > 1.0e-20),
            ((chosen_boundary - lg) / denominator).clamp(0.0, 1.0),
            0.5,
        )
        diffusion_time = left_time[:, :, None] + fraction * step[:, :, None]
        jump_time = (left_time + step)[:, :, None].expand(-1, -1, policy_count)
        times = torch.stack((diffusion_time, jump_time), dim=2).reshape_as(signals)
        diffusion_gap = torch.where(diffusion_signal != 0, chosen_boundary, 0.0)
        jump_gap = post.expand(-1, -1, policy_count)
        gaps = torch.stack((diffusion_gap, jump_gap), dim=2).reshape_as(signals)
        spread_diff = left_spread[:, :, None].expand(-1, -1, policy_count)
        spread_jump = post_spread[:, :, None].expand(-1, -1, policy_count)
        spreads = torch.stack((spread_diff, spread_jump), dim=2).reshape_as(signals)
        mid_diff = left_mid[:, :, None].expand(-1, -1, policy_count)
        mid_jump = post_mid[:, :, None].expand(-1, -1, policy_count)
        mids = torch.stack((mid_diff, mid_jump), dim=2).reshape_as(signals)
        is_bridge = torch.stack(
            (torch.ones_like(event_valid[:, ::2, :]), torch.zeros_like(event_valid[:, ::2, :])),
            dim=2,
        ).reshape_as(signals)

        local_cumulative = torch.cumsum(fills.to(torch.int64), dim=1)
        prior_boundary = initial_count[:, None, :] + local_cumulative - 1 > 0
        rewarded = fills & prior_boundary
        reward_values = 2.0 * (torch.abs(gaps) - spreads / 2.0)
        count = initial_count + fills.sum(dim=1, dtype=torch.int64)
        reward = initial_reward + torch.where(rewarded, reward_values, 0.0).sum(dim=1)
        overshoot_values = torch.clamp_min(
            torch.abs(gaps) - local_thresholds[:, None, :], 0.0
        )
        overshoot = initial_overshoot + torch.where(fills, overshoot_values, 0.0).sum(dim=1)
        open_count = initial_open_count + (
            fills & (spreads > delta_price)
        ).sum(dim=1, dtype=torch.int64)
        bridge_count = initial_bridge_count + (fills & is_bridge).sum(
            dim=1, dtype=torch.int64
        )
        jump_count = initial_jump_count + (fills & ~is_bridge).sum(
            dim=1, dtype=torch.int64
        )
        delta_q = signals.to(torch.int16) - previous.to(torch.int16)
        touch = torch.where(delta_q > 0, mids + spreads / 2.0, mids - spreads / 2.0)
        cash = initial_cash - torch.where(fills, delta_q * touch, 0.0).sum(dim=1)
        inf = torch.full_like(times, float("inf"))
        neg_inf = torch.full_like(times, -float("inf"))
        local_first = torch.where(fills, times, inf).amin(dim=1)
        local_last = torch.where(fills, times, neg_inf).amax(dim=1)
        first_time = torch.minimum(initial_first_time, local_first)
        last_time = torch.maximum(initial_last_time, local_last)
        omitted = torch.where(
            (p_upper_raw > 0.0) & (p_upper_raw < probability_cutoff),
            p_upper_raw,
            0.0,
        ).sum(dim=(1, 2)) + torch.where(
            (p_lower_raw > 0.0) & (p_lower_raw < probability_cutoff),
            p_lower_raw,
            0.0,
        ).sum(dim=(1, 2))
        competing = torch.minimum(p_upper_raw, p_lower_raw).sum(dim=(1, 2))
        return (
            final_side,
            count,
            reward,
            overshoot,
            open_count,
            bridge_count,
            jump_count,
            cash,
            first_time,
            last_time,
            omitted,
            competing,
        )

    compiled = torch.compile(kernel, fullgraph=True)
    generator = torch.Generator(device="cuda")
    generator.manual_seed(
        _bridge_seed(
            values, calibration.row_index, epsilon, [trace.seed for trace in traces]
        )
    )

    def empty_carry(side: Any | None = None) -> tuple[Any, ...]:
        shape = (batch, policy_count)
        return (
            torch.zeros(shape, dtype=torch.int8, device="cuda") if side is None else side,
            torch.zeros(shape, dtype=torch.int64, device="cuda"),
            torch.zeros(shape, dtype=torch.float32, device="cuda"),
            torch.zeros(shape, dtype=torch.float32, device="cuda"),
            torch.zeros(shape, dtype=torch.int64, device="cuda"),
            torch.zeros(shape, dtype=torch.int64, device="cuda"),
            torch.zeros(shape, dtype=torch.int64, device="cuda"),
            torch.zeros(shape, dtype=torch.float32, device="cuda"),
            torch.full(shape, float("inf"), dtype=torch.float32, device="cuda"),
            torch.full(shape, -float("inf"), dtype=torch.float32, device="cuda"),
        )

    transfer_seconds = 0.0
    kernel_seconds = 0.0
    cold_seconds = 0.0
    compiled_once = False
    def run_segment(
        segment: Mapping[str, np.ndarray], carry: tuple[Any, ...]
    ) -> tuple[tuple[Any, ...], np.ndarray, np.ndarray]:
        nonlocal transfer_seconds, kernel_seconds, cold_seconds, compiled_once
        omitted_segment = torch.zeros(batch, dtype=torch.float32, device="cuda")
        competing_segment = torch.zeros(batch, dtype=torch.float32, device="cuda")
        width = segment["valid"].shape[1]
        for offset in range(0, width, chunk_steps):
            if deadline_monotonic is not None and time.perf_counter() >= deadline_monotonic:
                raise TimeoutError("SIM-FIG4-002 exceeded its preregistered wall-clock budget")
            stop = min(offset + chunk_steps, width)
            local_width = stop - offset
            transfer_started = time.perf_counter()
            tensors: list[Any] = []
            for name in (
                "left_gap_price",
                "pre_event_gap_price",
                "post_event_gap_price",
                "left_mid_price",
                "post_event_mid_price",
                "left_spread_price",
                "post_event_spread_price",
                "left_time_seconds",
                "step_seconds",
            ):
                host = np.zeros((batch, chunk_steps), dtype=np.float32)
                host[:, :local_width] = segment[name][:, offset:stop]
                tensors.append(torch.as_tensor(host, device="cuda"))
            event_host = np.zeros((batch, chunk_steps), dtype=np.bool_)
            valid_host = np.zeros((batch, chunk_steps), dtype=np.bool_)
            event_host[:, :local_width] = segment["book_event"][:, offset:stop]
            valid_host[:, :local_width] = segment["valid"][:, offset:stop]
            book_tensor = torch.as_tensor(event_host, device="cuda")
            valid_tensor = torch.as_tensor(valid_host, device="cuda")
            random_upper = torch.rand(
                (batch, chunk_steps, policy_count),
                dtype=torch.float32,
                device="cuda",
                generator=generator,
            )
            random_lower = torch.rand(
                (batch, chunk_steps, policy_count),
                dtype=torch.float32,
                device="cuda",
                generator=generator,
            )
            torch.cuda.synchronize()
            transfer_seconds += time.perf_counter() - transfer_started
            started = time.perf_counter()
            result = compiled(
                *tensors,
                book_tensor,
                valid_tensor,
                thresholds_batch,
                random_upper,
                random_lower,
                *carry,
            )
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            kernel_seconds += elapsed
            if not compiled_once:
                cold_seconds = elapsed
                compiled_once = True
            carry = result[:10]
            omitted_segment += result[10]
            competing_segment += result[11]
        return (
            carry,
            omitted_segment.detach().cpu().numpy().astype(np.float64),
            competing_segment.detach().cpu().numpy().astype(np.float64),
        )

    pre_carry, pre_omitted, pre_competing = run_segment(pre, empty_carry())
    initial_side = pre_carry[0]
    initial_side_host = initial_side.detach().cpu().numpy()
    carry, measured_omitted, measured_competing = run_segment(
        measured, empty_carry(initial_side)
    )
    torch.cuda.synchronize()
    host = [item.detach().cpu().numpy() for item in carry]
    (
        terminal_side,
        fill_count,
        reward_sum,
        overshoot_sum,
        open_count,
        bridge_count,
        jump_count,
        cash,
        first_time,
        last_time,
    ) = host

    replications: list[Figure4Replication] = []
    for batch_index, trace in enumerate(traces):
        initial_positions = initial_side_host[batch_index]
        measurement_index = int(
            np.searchsorted(trace.left_time_seconds, measurement_start, side="left")
        )
        initial_mid = float(trace.left_mid_price[measurement_index])
        initial_gap = float(trace.left_gap_price[measurement_index])
        initial_x = initial_mid - initial_gap
        terminal_mid = float(trace.post_event_mid_price[-1])
        terminal_gap = float(trace.post_event_gap_price[-1])
        terminal_x = trace.terminal_efficient_price
        rows: list[dict[str, Any]] = []
        for policy_index in range(policy_count):
            count = int(fill_count[batch_index, policy_index])
            completed = max(count - 1, 0)
            duration = (
                float(last_time[batch_index, policy_index] - first_time[batch_index, policy_index])
                if completed > 0
                else 0.0
            )
            rate = float(reward_sum[batch_index, policy_index] / duration) if duration > 0 else 0.0
            terminal_position = int(terminal_side[batch_index, policy_index])
            initial_position = int(initial_positions[policy_index])
            local_cash = float(cash[batch_index, policy_index])
            mid_pnl = local_cash + terminal_position * terminal_mid - initial_position * initial_mid
            efficient_pnl = (
                local_cash + terminal_position * terminal_x - initial_position * initial_x
            )
            identity = abs(
                (mid_pnl - efficient_pnl)
                - (terminal_position * terminal_gap - initial_position * initial_gap)
            )
            rows.append(
                {
                    "row_index": calibration.row_index,
                    "alpha_per_second": calibration.alpha_per_second,
                    "gamma_ratio": calibration.gamma_ratio,
                    "epsilon": epsilon,
                    "seed": trace.seed,
                    "policy_index": policy_index,
                    "policy_label": labels[policy_index],
                    "threshold_multiplier_theta_d": float(multipliers[policy_index]),
                    "threshold_price": float(thresholds64[policy_index]),
                    "fill_count": count,
                    "complete_interval_count": completed,
                    "renewal_rate_per_second": rate,
                    "renewal_rate_over_alpha_s_g": rate
                    / (calibration.alpha_per_second * calibration.s_g_price),
                    "renewal_rate_over_surrogate_optimum": rate
                    / calibration.surrogate_optimum_rate_per_second,
                    "mean_interfill_seconds": duration / completed if completed else None,
                    "mean_fill_overshoot_price": float(
                        overshoot_sum[batch_index, policy_index] / count
                    )
                    if count
                    else 0.0,
                    "open_fill_share": float(open_count[batch_index, policy_index] / count)
                    if count
                    else 0.0,
                    "bridge_fill_count": int(bridge_count[batch_index, policy_index]),
                    "jump_fill_count": int(jump_count[batch_index, policy_index]),
                    "terminal_position": terminal_position,
                    "mid_marked_pnl": mid_pnl,
                    "efficient_price_marked_pnl": efficient_pnl,
                    "wealth_marking_identity_abs_residual": identity,
                }
            )
        digest = hashlib.sha256()
        digest.update(trace.replay_digest.encode("ascii"))
        digest.update(fill_count[batch_index].tobytes())
        digest.update(reward_sum[batch_index].tobytes())
        diagnostics = {
            "step_count": int(trace.left_time_seconds.size),
            "book_event_count": int(np.count_nonzero(trace.book_event)),
            "maximum_left_event_probability": trace.maximum_left_event_probability,
            "omitted_bridge_probability_sum": float(
                pre_omitted[batch_index] + measured_omitted[batch_index]
            ),
            "full_band_recrossing_probability_bound": float(
                measured_competing[batch_index]
            ),
            "flat_entry_competing_probability_sum_burn_in_only": float(
                pre_competing[batch_index]
            ),
            "nonflat_policy_count_at_measurement_start": int(
                np.count_nonzero(initial_side_host[batch_index])
            ),
            "policy_count": policy_count,
            "wealth_marking_identity_abs_residual_max": max(
                float(row["wealth_marking_identity_abs_residual"]) for row in rows
            ),
        }
        replications.append(
            Figure4Replication(
                row_index=calibration.row_index,
                alpha_per_second=calibration.alpha_per_second,
                epsilon=epsilon,
                seed=trace.seed,
                policy_rows=tuple(rows),
                diagnostics=diagnostics,
                replay_digest=digest.hexdigest(),
            )
        )
    benchmark = {
        "selected": "torch_compile_cuda_float32_continuous_crossing",
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu_device": torch.cuda.get_device_name(0),
        "batch_paths": batch,
        "policy_count": policy_count,
        "chunk_steps": chunk_steps,
        "cold_compile_and_first_kernel_seconds": cold_seconds,
        "kernel_total_seconds": kernel_seconds,
        "transfer_and_rng_total_seconds": transfer_seconds,
        "bridge_rng": "torch_cuda_philox",
        "dtype": "float32",
    }
    return tuple(replications), benchmark


def evaluate_market_traces_cuda(
    values: Mapping[str, Any],
    calibrations: Sequence[CalibrationRow],
    traces: Sequence[Figure4MarketTrace],
    *,
    chunk_steps: int = 32768,
    deadline_monotonic: float | None = None,
) -> CudaEvaluation:
    by_row = {row.row_index: row for row in calibrations}
    results: list[Figure4Replication] = []
    benchmarks: dict[str, Any] = {}
    keys = sorted({(trace.epsilon, trace.row_index) for trace in traces}, reverse=True)
    for epsilon, row_index in keys:
        if deadline_monotonic is not None and time.perf_counter() >= deadline_monotonic:
            raise TimeoutError("SIM-FIG4-002 exceeded its preregistered wall-clock budget")
        selected = sorted(
            [
                trace
                for trace in traces
                if trace.epsilon == epsilon and trace.row_index == row_index
            ],
            key=lambda item: item.seed,
        )
        local, benchmark = _evaluate_row_cuda(
            values,
            by_row[row_index],
            selected,
            chunk_steps=chunk_steps,
            deadline_monotonic=deadline_monotonic,
        )
        results.extend(local)
        benchmarks[f"epsilon_{epsilon:g}_row_{row_index}"] = benchmark
    results.sort(key=lambda item: (item.epsilon, item.row_index, item.seed), reverse=True)
    return CudaEvaluation(tuple(results), benchmarks)
