from __future__ import annotations

import csv
import heapq
import io
import math
import os
import shutil
import tarfile
import time
import zipfile
from array import array
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Mapping, Sequence

import orjson
import torch

from ot_micromr.artifacts import atomic_write_json, sha256_file, write_csv
from ot_micromr.config import RunSpec
from ot_micromr.errors import ExperimentError
from ot_micromr.okx_data import load_okx_source_list, load_raw_manifest


ORDERBOOK_KEYS = {"instId", "action", "ts", "asks", "bids"}
TRADE_FIELDS = ["instrument_name", "trade_id", "side", "price", "size", "created_time"]
FUNDING_FIELDS = ["instrument_name", "funding_rate", "funding_time"]
DAY_MILLISECONDS = 86_400_000


@dataclass(frozen=True, slots=True)
class EmpiricalEvaluationResult:
    metrics: Mapping[str, Any]
    acceptance: Mapping[str, bool]
    derived_parameters: Mapping[str, Any]
    log_lines: Sequence[str]

    @property
    def passed(self) -> bool:
        return all(self.acceptance.values())


def _utc_day_start_milliseconds(date_text: str) -> int:
    value = datetime.strptime(date_text, "%Y-%m-%d").replace(tzinfo=UTC)
    return int(value.timestamp() * 1000)


def _price_to_ticks(value: str, tick_scale: int) -> int:
    if not isinstance(value, str) or not value or value.startswith("-"):
        raise ValueError(f"invalid positive price {value!r}")
    whole, separator, fraction = value.partition(".")
    if not whole.isdigit() or (separator and not fraction.isdigit()):
        raise ValueError(f"invalid decimal price {value!r}")
    if len(fraction) > 1 and any(character != "0" for character in fraction[1:]):
        raise ValueError(f"price {value!r} is off the 0.1 tick lattice")
    fractional_tick = int(fraction[0]) if fraction else 0
    return int(whole) * tick_scale + fractional_tick


def _book_tensor_metrics(
    timestamps: torch.Tensor,
    bids: torch.Tensor,
    asks: torch.Tensor,
    bid_sizes: torch.Tensor,
    ask_sizes: torch.Tensor,
    previous_timestamp: int | None,
    previous_bid: int | None,
    previous_ask: int | None,
) -> dict[str, int | float]:
    if timestamps.device.type != "cpu" or timestamps.dtype != torch.int64:
        raise ExperimentError("book batch timestamps must be CPU int64")
    if bids.shape != timestamps.shape or asks.shape != timestamps.shape:
        raise ExperimentError("book batch price shapes disagree")
    if bid_sizes.shape != timestamps.shape or ask_sizes.shape != timestamps.shape:
        raise ExperimentError("book batch size shapes disagree")
    spread = asks - bids
    valid = (bids > 0) & (asks > 0)
    one_tick = valid & (spread == 1)
    two_tick = valid & (spread == 2)
    locked = valid & (spread == 0)
    crossed = valid & (spread < 0)
    off_support = valid & (spread > 2)

    if previous_timestamp is None:
        timestamp_differences = timestamps[1:] - timestamps[:-1]
    else:
        timestamp_differences = torch.cat(
            (
                timestamps.new_tensor([timestamps[0].item() - previous_timestamp]),
                timestamps[1:] - timestamps[:-1],
            )
        )
    if previous_bid is None or previous_ask is None:
        bbo_changes = (bids[1:] != bids[:-1]) | (asks[1:] != asks[:-1])
    else:
        previous_bids = torch.cat((bids.new_tensor([previous_bid]), bids[:-1]))
        previous_asks = torch.cat((asks.new_tensor([previous_ask]), asks[:-1]))
        bbo_changes = (bids != previous_bids) | (asks != previous_asks)

    valid_bid_sizes = bid_sizes[valid]
    valid_ask_sizes = ask_sizes[valid]
    return {
        "rows": int(timestamps.numel()),
        "valid_book_rows": int(torch.count_nonzero(valid).item()),
        "empty_book_rows": int(torch.count_nonzero(~valid).item()),
        "one_tick_rows": int(torch.count_nonzero(one_tick).item()),
        "two_tick_rows": int(torch.count_nonzero(two_tick).item()),
        "off_support_rows": int(torch.count_nonzero(off_support).item()),
        "locked_rows": int(torch.count_nonzero(locked).item()),
        "crossed_rows": int(torch.count_nonzero(crossed).item()),
        "duplicate_timestamp_rows": int(
            torch.count_nonzero(timestamp_differences == 0).item()
        ),
        "nonmonotone_timestamp_rows": int(
            torch.count_nonzero(timestamp_differences < 0).item()
        ),
        "maximum_timestamp_gap_ms": int(
            torch.max(timestamp_differences).item()
            if timestamp_differences.numel()
            else 0
        ),
        "bbo_change_rows": int(torch.count_nonzero(bbo_changes).item()),
        "bid_touch_size_sum": float(torch.sum(valid_bid_sizes, dtype=torch.float64).item()),
        "ask_touch_size_sum": float(torch.sum(valid_ask_sizes, dtype=torch.float64).item()),
        "bid_touch_size_min": float(torch.min(valid_bid_sizes).item())
        if valid_bid_sizes.numel()
        else math.inf,
        "ask_touch_size_min": float(torch.min(valid_ask_sizes).item())
        if valid_ask_sizes.numel()
        else math.inf,
    }


def _new_book_buffers() -> tuple[array, array, array, array, array]:
    return array("q"), array("q"), array("q"), array("d"), array("d")


def _apply_levels(
    levels: list[Any],
    book: dict[int, float],
    heap: list[int],
    heap_sign: int,
    tick_scale: int,
) -> None:
    for level in levels:
        price_tick = _price_to_ticks(level[0], tick_scale)
        size = float(level[1])
        order_count = int(level[2])
        if not math.isfinite(size) or size < 0.0 or order_count < 0:
            raise ValueError("invalid order-book size or order count")
        if size == 0.0:
            book.pop(price_tick, None)
        else:
            is_new = price_tick not in book
            book[price_tick] = size
            if is_new:
                heapq.heappush(heap, heap_sign * price_tick)


def _clean_heap(heap: list[int], book: Mapping[int, Any], heap_sign: int) -> None:
    while heap and heap_sign * heap[0] not in book:
        heapq.heappop(heap)


def _compact_heap(heap: list[int], book: Mapping[int, Any], heap_sign: int) -> None:
    if len(heap) > max(1024, 4 * len(book)):
        heap[:] = [heap_sign * price for price in book]
        heapq.heapify(heap)


def _audit_orderbook_worker(task: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(str(task["path"]))
    instrument = str(task["instrument"])
    date_text = str(task["date"])
    tick_scale = int(task["tick_scale"])
    batch_rows = int(task["batch_rows"])
    started = time.perf_counter()
    asks: dict[int, float] = {}
    bids: dict[int, float] = {}
    ask_heap: list[int] = []
    bid_heap: list[int] = []
    buffers = _new_book_buffers()
    accumulator: dict[str, int | float] = {
        "rows": 0,
        "valid_book_rows": 0,
        "empty_book_rows": 0,
        "one_tick_rows": 0,
        "two_tick_rows": 0,
        "off_support_rows": 0,
        "locked_rows": 0,
        "crossed_rows": 0,
        "duplicate_timestamp_rows": 0,
        "nonmonotone_timestamp_rows": 0,
        "maximum_timestamp_gap_ms": 0,
        "bbo_change_rows": 0,
        "bid_touch_size_sum": 0.0,
        "ask_touch_size_sum": 0.0,
        "bid_touch_size_min": math.inf,
        "ask_touch_size_min": math.inf,
    }
    snapshot_rows = 0
    update_rows = 0
    pre_snapshot_update_rows = 0
    first_action: str | None = None
    first_timestamp: int | None = None
    first_snapshot_timestamp: int | None = None
    last_timestamp: int | None = None
    previous_source_timestamp: int | None = None
    source_duplicate_timestamp_rows = 0
    source_nonmonotone_timestamp_rows = 0
    previous_timestamp: int | None = None
    previous_bid: int | None = None
    previous_ask: int | None = None

    def flush() -> None:
        nonlocal buffers, previous_timestamp, previous_bid, previous_ask
        timestamps_buffer, bids_buffer, asks_buffer, bid_sizes_buffer, ask_sizes_buffer = buffers
        if not timestamps_buffer:
            return
        timestamps = torch.frombuffer(timestamps_buffer, dtype=torch.int64)
        bid_values = torch.frombuffer(bids_buffer, dtype=torch.int64)
        ask_values = torch.frombuffer(asks_buffer, dtype=torch.int64)
        bid_size_values = torch.frombuffer(bid_sizes_buffer, dtype=torch.float64)
        ask_size_values = torch.frombuffer(ask_sizes_buffer, dtype=torch.float64)
        metrics = _book_tensor_metrics(
            timestamps,
            bid_values,
            ask_values,
            bid_size_values,
            ask_size_values,
            previous_timestamp,
            previous_bid,
            previous_ask,
        )
        for key, value in metrics.items():
            if key == "maximum_timestamp_gap_ms":
                accumulator[key] = max(int(accumulator[key]), int(value))
            elif key.endswith("_min"):
                accumulator[key] = min(float(accumulator[key]), float(value))
            else:
                accumulator[key] = accumulator[key] + value
        previous_timestamp = int(timestamps[-1].item())
        previous_bid = int(bid_values[-1].item())
        previous_ask = int(ask_values[-1].item())
        buffers = _new_book_buffers()

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
                        raise ValueError("unexpected order-book record schema")
                    if record["instId"] != instrument:
                        raise ValueError("unexpected order-book instrument")
                    action = record["action"]
                    if action not in {"snapshot", "update"}:
                        raise ValueError(f"unexpected order-book action {action!r}")
                    timestamp = int(record["ts"])
                    ask_levels = record["asks"]
                    bid_levels = record["bids"]
                    if not isinstance(ask_levels, list) or not isinstance(bid_levels, list):
                        raise ValueError("asks and bids must be arrays")
                    if first_action is None:
                        first_action = str(action)
                    if first_timestamp is None:
                        first_timestamp = timestamp
                    last_timestamp = timestamp
                    if previous_source_timestamp is not None:
                        source_duplicate_timestamp_rows += int(
                            timestamp == previous_source_timestamp
                        )
                        source_nonmonotone_timestamp_rows += int(
                            timestamp < previous_source_timestamp
                        )
                    previous_source_timestamp = timestamp
                    if action == "snapshot":
                        snapshot_rows += 1
                        if first_snapshot_timestamp is None:
                            first_snapshot_timestamp = timestamp
                        asks.clear()
                        bids.clear()
                        ask_heap.clear()
                        bid_heap.clear()
                    else:
                        update_rows += 1
                        if first_snapshot_timestamp is None:
                            pre_snapshot_update_rows += 1
                    _apply_levels(ask_levels, asks, ask_heap, 1, tick_scale)
                    _apply_levels(bid_levels, bids, bid_heap, -1, tick_scale)
                    if first_snapshot_timestamp is None:
                        # A daily archive can start inside the source's native snapshot
                        # cycle.  Validate the prefix, but never treat partial deltas as
                        # a reconstructible book.  The first complete snapshot clears it.
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
                    if ask_heap and bid_heap:
                        best_ask = ask_heap[0]
                        best_bid = -bid_heap[0]
                        ask_size = asks[best_ask]
                        bid_size = bids[best_bid]
                    else:
                        best_ask = 0
                        best_bid = 0
                        ask_size = 0.0
                        bid_size = 0.0
                    buffers[0].append(timestamp)
                    buffers[1].append(best_bid)
                    buffers[2].append(best_ask)
                    buffers[3].append(bid_size)
                    buffers[4].append(ask_size)
                    if len(buffers[0]) >= batch_rows:
                        flush()
                except Exception as error:
                    raise ExperimentError(
                        f"{path.name}: invalid order-book row {row_index + 1}: {error}"
                    ) from error
    flush()
    if first_timestamp is None or last_timestamp is None:
        raise ExperimentError(f"empty order-book archive: {path}")
    if first_snapshot_timestamp is None:
        raise ExperimentError(f"order-book archive contains no usable snapshot: {path}")
    valid_rows = int(accumulator["valid_book_rows"])
    support_rows = int(accumulator["one_tick_rows"]) + int(accumulator["two_tick_rows"])
    day_start = _utc_day_start_milliseconds(date_text)
    day_end = day_start + DAY_MILLISECONDS - 1
    return {
        "asset_id": str(task["asset_id"]),
        "kind": "orderbook_l2",
        "instrument_type": str(task["instrument_type"]),
        "instrument": instrument,
        "date": date_text,
        "rows": int(accumulator["rows"]),
        "source_rows": int(accumulator["rows"]) + pre_snapshot_update_rows,
        "snapshot_rows": snapshot_rows,
        "update_rows": update_rows,
        "pre_snapshot_update_rows": pre_snapshot_update_rows,
        "first_action": first_action,
        "first_timestamp_ms": first_timestamp,
        "first_snapshot_timestamp_ms": first_snapshot_timestamp,
        "last_timestamp_ms": last_timestamp,
        "start_coverage_lag_ms": first_timestamp - day_start,
        "usable_start_coverage_lag_ms": first_snapshot_timestamp - day_start,
        "end_coverage_lag_ms": day_end - last_timestamp,
        "duplicate_timestamp_rows": source_duplicate_timestamp_rows,
        "nonmonotone_timestamp_rows": source_nonmonotone_timestamp_rows,
        "maximum_timestamp_gap_ms": int(accumulator["maximum_timestamp_gap_ms"]),
        "valid_book_rows": valid_rows,
        "empty_book_rows": int(accumulator["empty_book_rows"]),
        "one_tick_rows": int(accumulator["one_tick_rows"]),
        "two_tick_rows": int(accumulator["two_tick_rows"]),
        "off_support_rows": int(accumulator["off_support_rows"]),
        "locked_rows": int(accumulator["locked_rows"]),
        "crossed_rows": int(accumulator["crossed_rows"]),
        "one_two_tick_fraction": support_rows / valid_rows if valid_rows else 0.0,
        "bbo_change_rows": int(accumulator["bbo_change_rows"]),
        "bbo_change_fraction": float(accumulator["bbo_change_rows"])
        / float(accumulator["rows"]),
        "bid_touch_size_mean": float(accumulator["bid_touch_size_sum"]) / valid_rows,
        "ask_touch_size_mean": float(accumulator["ask_touch_size_sum"]) / valid_rows,
        "bid_touch_size_min": float(accumulator["bid_touch_size_min"]),
        "ask_touch_size_min": float(accumulator["ask_touch_size_min"]),
        "elapsed_seconds": time.perf_counter() - started,
    }


def _trade_tensor_metrics(
    timestamps: torch.Tensor,
    trade_ids: torch.Tensor,
    sides: torch.Tensor,
    sizes: torch.Tensor,
    previous_timestamp: int | None,
    previous_trade_id: int | None,
) -> dict[str, int]:
    if previous_timestamp is None:
        timestamp_differences = timestamps[1:] - timestamps[:-1]
    else:
        timestamp_differences = torch.cat(
            (
                timestamps.new_tensor([timestamps[0].item() - previous_timestamp]),
                timestamps[1:] - timestamps[:-1],
            )
        )
    if previous_trade_id is None:
        id_differences = trade_ids[1:] - trade_ids[:-1]
    else:
        id_differences = torch.cat(
            (
                trade_ids.new_tensor([trade_ids[0].item() - previous_trade_id]),
                trade_ids[1:] - trade_ids[:-1],
            )
        )
    return {
        "rows": int(timestamps.numel()),
        "nonmonotone_timestamp_rows": int(
            torch.count_nonzero(timestamp_differences < 0).item()
        ),
        "nonincreasing_trade_id_rows": int(torch.count_nonzero(id_differences <= 0).item()),
        "invalid_side_rows": int(torch.count_nonzero(sides == 0).item()),
        "nonpositive_size_rows": int(torch.count_nonzero(sizes <= 0.0).item()),
    }


def _audit_trades_worker(task: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(str(task["path"]))
    instrument = str(task["instrument"])
    tick_scale = int(task["tick_scale"])
    batch_rows = int(task["batch_rows"])
    started = time.perf_counter()
    timestamps = array("q")
    trade_ids = array("q")
    sides = array("b")
    sizes = array("d")
    counts = {
        "rows": 0,
        "nonmonotone_timestamp_rows": 0,
        "nonincreasing_trade_id_rows": 0,
        "invalid_side_rows": 0,
        "nonpositive_size_rows": 0,
    }
    first_timestamp: int | None = None
    last_timestamp: int | None = None
    previous_timestamp: int | None = None
    previous_trade_id: int | None = None

    def flush() -> None:
        nonlocal timestamps, trade_ids, sides, sizes, previous_timestamp, previous_trade_id
        if not timestamps:
            return
        timestamp_tensor = torch.frombuffer(timestamps, dtype=torch.int64)
        id_tensor = torch.frombuffer(trade_ids, dtype=torch.int64)
        side_tensor = torch.frombuffer(sides, dtype=torch.int8)
        size_tensor = torch.frombuffer(sizes, dtype=torch.float64)
        metrics = _trade_tensor_metrics(
            timestamp_tensor,
            id_tensor,
            side_tensor,
            size_tensor,
            previous_timestamp,
            previous_trade_id,
        )
        for key, value in metrics.items():
            counts[key] += value
        previous_timestamp = int(timestamp_tensor[-1].item())
        previous_trade_id = int(id_tensor[-1].item())
        timestamps = array("q")
        trade_ids = array("q")
        sides = array("b")
        sizes = array("d")

    with zipfile.ZipFile(path) as archive:
        members = [name for name in archive.namelist() if not name.endswith("/")]
        if len(members) != 1:
            raise ExperimentError(f"trade archive must contain exactly one file: {path}")
        with archive.open(members[0]) as binary:
            with io.TextIOWrapper(binary, encoding="utf-8", newline="") as text:
                reader = csv.DictReader(text)
                if reader.fieldnames != TRADE_FIELDS:
                    raise ExperimentError(f"unexpected trade schema in {path}")
                for row_index, row in enumerate(reader):
                    try:
                        if row["instrument_name"] != instrument:
                            raise ValueError("unexpected trade instrument")
                        timestamp = int(row["created_time"])
                        trade_id = int(row["trade_id"])
                        price_tick = _price_to_ticks(row["price"], tick_scale)
                        size = float(row["size"])
                        if price_tick <= 0 or not math.isfinite(size):
                            raise ValueError("trade price and size must be finite and positive")
                        side = 1 if row["side"] == "buy" else -1 if row["side"] == "sell" else 0
                        if first_timestamp is None:
                            first_timestamp = timestamp
                        last_timestamp = timestamp
                        timestamps.append(timestamp)
                        trade_ids.append(trade_id)
                        sides.append(side)
                        sizes.append(size)
                        if len(timestamps) >= batch_rows:
                            flush()
                    except Exception as error:
                        raise ExperimentError(
                            f"{path.name}: invalid trade row {row_index + 2}: {error}"
                        ) from error
    flush()
    if first_timestamp is None or last_timestamp is None:
        raise ExperimentError(f"empty trade archive: {path}")
    return {
        "asset_id": str(task["asset_id"]),
        "kind": "trades",
        "instrument_type": str(task["instrument_type"]),
        "instrument": instrument,
        "date": str(task["date"]),
        **counts,
        "first_timestamp_ms": first_timestamp,
        "last_timestamp_ms": last_timestamp,
        "elapsed_seconds": time.perf_counter() - started,
    }


def _init_audit_worker() -> None:
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def _verify_raw_assets(
    source_list: Any, manifest: Mapping[str, Any], workers: int
) -> tuple[list[dict[str, Any]], dict[str, Path]]:
    manifest_records = {
        str(record["asset_id"]): record for record in manifest["assets"]
    }
    if set(manifest_records) != {asset.asset_id for asset in source_list.assets}:
        raise ExperimentError("raw manifest and source list asset IDs disagree")

    paths: dict[str, Path] = {}
    with ThreadPoolExecutor(max_workers=min(workers, len(source_list.assets))) as executor:
        futures = {}
        for asset in source_list.assets:
            record = manifest_records[asset.asset_id]
            expected_relative = asset.relative_path().as_posix()
            if record.get("relative_path") != expected_relative:
                raise ExperimentError(f"raw manifest path mismatch for {asset.asset_id}")
            path = source_list.dataset_directory / expected_relative
            paths[asset.asset_id] = path
            futures[executor.submit(sha256_file, path)] = (asset, record, path)
        verified: list[dict[str, Any]] = []
        for future in as_completed(futures):
            asset, record, path = futures[future]
            actual_size = path.stat().st_size if path.is_file() else -1
            actual_hash = future.result()
            verified.append(
                {
                    "asset_id": asset.asset_id,
                    "kind": asset.kind,
                    "instrument_type": asset.instrument_type,
                    "instrument": asset.instrument,
                    "date": asset.date,
                    "relative_path": asset.relative_path().as_posix(),
                    "size_bytes": actual_size,
                    "sha256": actual_hash,
                    "size_matches": actual_size == asset.expected_size_bytes == record["size_bytes"],
                    "sha256_matches": actual_hash == record["sha256"],
                }
            )
    verified.sort(key=lambda row: str(row["asset_id"]))
    return verified, paths


def _audit_funding(
    source_list: Any,
    paths: Mapping[str, Path],
    train_start_ms: int,
    train_end_ms: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    values: dict[int, float] = {}
    rows_by_asset: list[dict[str, Any]] = []
    conflicting_duplicates = 0
    for asset in source_list.assets:
        if asset.kind != "funding":
            continue
        rows = 0
        with zipfile.ZipFile(paths[asset.asset_id]) as archive:
            members = [name for name in archive.namelist() if not name.endswith("/")]
            if len(members) != 1:
                raise ExperimentError("funding archive must contain exactly one file")
            with archive.open(members[0]) as binary:
                with io.TextIOWrapper(binary, encoding="utf-8", newline="") as text:
                    reader = csv.DictReader(text)
                    if reader.fieldnames != FUNDING_FIELDS:
                        raise ExperimentError("unexpected funding schema")
                    for row in reader:
                        if row["instrument_name"] != "BTC-USDT-SWAP":
                            raise ExperimentError("unexpected funding instrument")
                        timestamp = int(row["funding_time"])
                        rate = float(row["funding_rate"])
                        if not math.isfinite(rate):
                            raise ExperimentError("non-finite funding rate")
                        previous = values.get(timestamp)
                        if previous is not None and previous != rate:
                            conflicting_duplicates += 1
                        values[timestamp] = rate
                        rows += 1
        rows_by_asset.append({"asset_id": asset.asset_id, "date": asset.date, "rows": rows})
    selected = sorted(
        (timestamp, rate)
        for timestamp, rate in values.items()
        if train_start_ms <= timestamp <= train_end_ms
    )
    if not selected:
        raise ExperimentError("funding data does not overlap train interval")
    timestamps = torch.tensor([row[0] for row in selected], dtype=torch.int64)
    rates = torch.tensor([row[1] for row in selected], dtype=torch.float64)
    differences = timestamps[1:] - timestamps[:-1]
    summary = {
        "raw_rows": sum(int(row["rows"]) for row in rows_by_asset),
        "unique_train_rows": int(timestamps.numel()),
        "first_timestamp_ms": int(timestamps[0].item()),
        "last_timestamp_ms": int(timestamps[-1].item()),
        "maximum_gap_ms": int(torch.max(differences).item()) if differences.numel() else 0,
        "nonpositive_gap_count": int(torch.count_nonzero(differences <= 0).item()),
        "conflicting_duplicate_count": conflicting_duplicates,
        "rate_mean": float(torch.mean(rates).item()),
        "rate_min": float(torch.min(rates).item()),
        "rate_max": float(torch.max(rates).item()),
    }
    return summary, rows_by_asset


def _bootstrap_large_tick(
    day_fractions: list[float], replications: int, seed: int
) -> dict[str, Any]:
    fractions = torch.tensor(day_fractions, dtype=torch.float64)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    indices = torch.randint(
        fractions.numel(),
        (replications, fractions.numel()),
        generator=generator,
        device="cpu",
    )
    bootstrap_means = torch.mean(fractions[indices], dim=1)
    return {
        "day_fractions": day_fractions,
        "equal_weight_day_mean": float(torch.mean(fractions).item()),
        "one_sided_95_percent_lower_bound": float(
            torch.quantile(bootstrap_means, 0.05).item()
        ),
        "bootstrap_replications": replications,
        "seed": seed,
        "independent_unit": "UTC_day",
    }


def evaluate_empirical_data(spec: RunSpec, run_directory: Path) -> EmpiricalEvaluationResult:
    values = spec.values
    inputs = values["inputs"]
    numerics = values["numerics"]
    evaluation = values["evaluation"]
    acceptance_spec = values["acceptance"]
    source_list = load_okx_source_list(
        spec.repository_root / str(inputs["source_spec_path"]), spec.repository_root
    )
    if source_list.source_sha256 != inputs["source_spec_sha256"]:
        raise ExperimentError("source-list SHA-256 does not match RunSpec")
    manifest_path = spec.repository_root / str(inputs["raw_manifest_path"])
    manifest = load_raw_manifest(manifest_path)
    if manifest["dataset_id"] != inputs["dataset"]:
        raise ExperimentError("raw manifest dataset ID does not match RunSpec")
    if manifest["dataset_content_sha256"] != inputs["dataset_sha256"]:
        raise ExperimentError("raw manifest dataset hash does not match RunSpec")
    if manifest["source_spec_sha256"] != inputs["source_spec_sha256"]:
        raise ExperimentError("raw manifest source-list hash does not match RunSpec")

    audit_workers = int(numerics["audit_workers"])
    verified_rows, paths = _verify_raw_assets(source_list, manifest, audit_workers)
    if not all(row["size_matches"] and row["sha256_matches"] for row in verified_rows):
        raise ExperimentError("one or more raw assets fail size/hash verification")
    (run_directory / "metrics").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(manifest_path, run_directory / "metrics" / "raw_manifest.json")

    tasks: list[dict[str, Any]] = []
    for asset in source_list.assets:
        if asset.kind not in {"orderbook_l2", "trades"}:
            continue
        tasks.append(
            {
                "asset_id": asset.asset_id,
                "kind": asset.kind,
                "instrument_type": asset.instrument_type,
                "instrument": asset.instrument,
                "date": asset.date,
                "path": str(paths[asset.asset_id]),
                "tick_scale": 10,
                "batch_rows": int(numerics["batch_rows"]),
            }
        )
    audit_rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(
        max_workers=audit_workers, initializer=_init_audit_worker
    ) as executor:
        futures = {
            executor.submit(
                _audit_orderbook_worker if task["kind"] == "orderbook_l2" else _audit_trades_worker,
                task,
            ): str(task["asset_id"])
            for task in tasks
        }
        for future in as_completed(futures):
            audit_rows.append(future.result())
    audit_rows.sort(key=lambda row: str(row["asset_id"]))
    orderbook_rows = [row for row in audit_rows if row["kind"] == "orderbook_l2"]
    trade_rows = [row for row in audit_rows if row["kind"] == "trades"]

    train_start_ms = int(
        datetime.fromisoformat(str(inputs["train_start_utc"]).replace("Z", "+00:00")).timestamp()
        * 1000
    )
    train_end_ms = int(
        datetime.fromisoformat(str(inputs["train_end_utc"]).replace("Z", "+00:00")).timestamp()
        * 1000
    )
    funding_summary, funding_rows = _audit_funding(
        source_list, paths, train_start_ms, train_end_ms
    )
    swap_book_rows = [
        row for row in orderbook_rows if row["instrument"] == "BTC-USDT-SWAP"
    ]
    bootstrap = _bootstrap_large_tick(
        [float(row["one_two_tick_fraction"]) for row in swap_book_rows],
        int(evaluation["bootstrap_replications"]),
        int(values["seed_policy"]["seeds"][0]),
    )

    file_rows_by_id = {str(row["asset_id"]): row for row in verified_rows}
    combined_file_rows: list[dict[str, Any]] = []
    audit_by_id = {str(row["asset_id"]): row for row in audit_rows}
    funding_by_id = {str(row["asset_id"]): row for row in funding_rows}
    for asset in source_list.assets:
        combined = dict(file_rows_by_id[asset.asset_id])
        if asset.asset_id in audit_by_id:
            combined.update(audit_by_id[asset.asset_id])
        elif asset.asset_id in funding_by_id:
            combined.update(funding_by_id[asset.asset_id])
        combined_file_rows.append(combined)
    file_fieldnames = sorted({key for row in combined_file_rows for key in row})
    write_csv(
        run_directory / "metrics" / "file_quality.csv",
        file_fieldnames,
        combined_file_rows,
    )
    day_fieldnames = sorted({key for row in audit_rows for key in row})
    write_csv(run_directory / "tables" / "day_quality.csv", day_fieldnames, audit_rows)
    write_csv(
        run_directory / "tables" / "funding_quality.csv",
        list(funding_summary.keys()),
        [funding_summary],
    )
    split_rows = [
        {
            "split": name,
            "start_utc": str(inputs[f"{name}_start_utc"]),
            "end_utc": str(inputs[f"{name}_end_utc"]),
            "calendar_days": int(inputs[f"{name}_calendar_days"]),
            "p5_sample_payload_inspected": name == "train",
        }
        for name in ("train", "validation", "test")
    ]
    write_csv(
        run_directory / "tables" / "split_freeze.csv",
        list(split_rows[0].keys()),
        split_rows,
    )
    atomic_write_json(run_directory / "metrics" / "bootstrap.json", bootstrap)

    endpoint_tolerance = int(evaluation["endpoint_tolerance_ms"])
    maximum_initial_snapshot_delay = int(
        evaluation["maximum_initial_snapshot_delay_ms"]
    )
    orderbook_structural_pass = all(
        row["nonmonotone_timestamp_rows"] == 0
        and abs(int(row["start_coverage_lag_ms"])) <= endpoint_tolerance
        and -endpoint_tolerance
        <= int(row["usable_start_coverage_lag_ms"])
        <= maximum_initial_snapshot_delay
        and abs(int(row["end_coverage_lag_ms"])) <= endpoint_tolerance
        and row["empty_book_rows"] == 0
        and row["locked_rows"] == 0
        and row["crossed_rows"] == 0
        for row in orderbook_rows
    )
    trade_structural_pass = all(
        row["nonmonotone_timestamp_rows"] == 0
        and row["nonincreasing_trade_id_rows"] == 0
        and row["invalid_side_rows"] == 0
        and row["nonpositive_size_rows"] == 0
        for row in trade_rows
    )
    spot_dates = {
        str(row["date"])
        for row in orderbook_rows
        if row["instrument"] == "BTC-USDT"
        and abs(int(row["start_coverage_lag_ms"])) <= endpoint_tolerance
        and -endpoint_tolerance
        <= int(row["usable_start_coverage_lag_ms"])
        <= maximum_initial_snapshot_delay
        and abs(int(row["end_coverage_lag_ms"])) <= endpoint_tolerance
    }
    swap_dates = {
        str(row["date"])
        for row in swap_book_rows
        if abs(int(row["start_coverage_lag_ms"])) <= endpoint_tolerance
        and -endpoint_tolerance
        <= int(row["usable_start_coverage_lag_ms"])
        <= maximum_initial_snapshot_delay
        and abs(int(row["end_coverage_lag_ms"])) <= endpoint_tolerance
    }
    expected_spot_dates = set(str(value) for value in evaluation["spot_alignment_dates"])
    funding_pass = (
        funding_summary["conflicting_duplicate_count"] == 0
        and funding_summary["nonpositive_gap_count"] == 0
        and funding_summary["maximum_gap_ms"] <= int(evaluation["funding_maximum_gap_ms"])
        and funding_summary["first_timestamp_ms"]
        == int(evaluation["funding_expected_first_timestamp_ms"])
        and funding_summary["last_timestamp_ms"]
        == int(evaluation["funding_expected_last_timestamp_ms"])
    )
    acceptance = {
        "raw_size_and_sha256": all(
            bool(row["size_matches"] and row["sha256_matches"]) for row in verified_rows
        ),
        "orderbook_structural_quality": orderbook_structural_pass,
        "trade_structural_quality": trade_structural_pass,
        "funding_train_coverage": funding_pass,
        "large_tick_day_cluster_lower_bound": float(
            bootstrap["one_sided_95_percent_lower_bound"]
        )
        > float(acceptance_spec["minimum_large_tick_lower_bound"]),
        "same_venue_spot_swap_overlap": expected_spot_dates <= spot_dates & swap_dates,
        "expected_asset_count": len(verified_rows) == int(acceptance_spec["expected_asset_count"]),
        "expected_orderbook_days": len(orderbook_rows)
        == int(acceptance_spec["expected_orderbook_days"]),
        "expected_trade_days": len(trade_rows)
        == int(acceptance_spec["expected_trade_days"]),
    }
    total_l2_rows = sum(int(row["rows"]) for row in orderbook_rows)
    total_trade_rows = sum(int(row["rows"]) for row in trade_rows)
    metrics = {
        "dataset_content_sha256": manifest["dataset_content_sha256"],
        "asset_count": len(verified_rows),
        "compressed_bytes": sum(int(row["size_bytes"]) for row in verified_rows),
        "orderbook_days": len(orderbook_rows),
        "swap_orderbook_days": len(swap_book_rows),
        "spot_orderbook_days": len(orderbook_rows) - len(swap_book_rows),
        "orderbook_rows": total_l2_rows,
        "orderbook_source_rows": sum(int(row["source_rows"]) for row in orderbook_rows),
        "pre_snapshot_update_rows": sum(
            int(row["pre_snapshot_update_rows"]) for row in orderbook_rows
        ),
        "trade_rows": total_trade_rows,
        "funding_unique_train_rows": funding_summary["unique_train_rows"],
        "swap_one_two_tick_equal_day_mean": bootstrap["equal_weight_day_mean"],
        "swap_one_two_tick_lower_95": bootstrap["one_sided_95_percent_lower_bound"],
        "maximum_orderbook_gap_ms": max(
            int(row["maximum_timestamp_gap_ms"]) for row in orderbook_rows
        ),
        "total_bbo_change_rows": sum(int(row["bbo_change_rows"]) for row in orderbook_rows),
        "total_processing_seconds_across_files": sum(
            float(row["elapsed_seconds"]) for row in audit_rows
        ),
    }
    return EmpiricalEvaluationResult(
        metrics=metrics,
        acceptance=acceptance,
        derived_parameters={
            "venue": "OKX",
            "execution_instrument": "BTC-USDT-SWAP",
            "same_venue_reference_candidate": "BTC-USDT",
            "price_tick_usdt": float(evaluation["price_tick"]),
            "train_start_utc": str(inputs["train_start_utc"]),
            "train_end_utc": str(inputs["train_end_utc"]),
            "validation_start_utc": str(inputs["validation_start_utc"]),
            "validation_end_utc": str(inputs["validation_end_utc"]),
            "test_start_utc": str(inputs["test_start_utc"]),
            "test_end_utc": str(inputs["test_end_utc"]),
        },
        log_lines=(
            f"Verified {len(verified_rows)} content-addressed OKX assets",
            f"Audited {total_l2_rows} L2 rows and {total_trade_rows} trades",
            "Swap one/two-tick day mean="
            f"{bootstrap['equal_weight_day_mean']:.12g}; lower95="
            f"{bootstrap['one_sided_95_percent_lower_bound']:.12g}",
            f"acceptance_passed={all(acceptance.values())}",
        ),
    )
