from __future__ import annotations

import hashlib
import json
import os
import tomllib
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Mapping
from urllib.parse import urlsplit

from ot_micromr.artifacts import atomic_write_json, sha256_file, utc_text
from ot_micromr.config import discover_repository_root
from ot_micromr.errors import ConfigError, ExperimentError


SOURCE_LIST_KEYS = {
    "schema_version",
    "dataset_id",
    "venue",
    "raw_root",
    "download_workers",
    "request_timeout_seconds",
    "chunk_size_bytes",
    "assets",
}
ASSET_KEYS = {
    "asset_id",
    "kind",
    "instrument_type",
    "instrument",
    "date",
    "archive_format",
    "expected_size_bytes",
    "url",
}
ALLOWED_KINDS = {"orderbook_l2", "trades", "funding"}
ALLOWED_FORMATS = {"tar_gzip_ndjson", "zip_csv"}


@dataclass(frozen=True, slots=True)
class OkxSourceAsset:
    asset_id: str
    kind: str
    instrument_type: str
    instrument: str
    date: str
    archive_format: str
    expected_size_bytes: int
    url: str

    @property
    def filename(self) -> str:
        return PurePosixPath(urlsplit(self.url).path).name

    def relative_path(self) -> Path:
        return Path("raw") / self.kind / self.filename


@dataclass(frozen=True, slots=True)
class OkxSourceList:
    source_path: Path
    source_sha256: str
    repository_root: Path
    dataset_id: str
    venue: str
    raw_root: Path
    download_workers: int
    request_timeout_seconds: int
    chunk_size_bytes: int
    assets: tuple[OkxSourceAsset, ...]

    @property
    def dataset_directory(self) -> Path:
        return self.repository_root / self.raw_root

    @property
    def manifest_path(self) -> Path:
        return self.dataset_directory / "raw_manifest.json"


def _expect_exact_keys(value: Mapping[str, Any], expected: set[str], path: str) -> None:
    missing = sorted(expected - value.keys())
    unknown = sorted(value.keys() - expected)
    if missing:
        raise ConfigError(f"{path}: missing required fields: {', '.join(missing)}")
    if unknown:
        raise ConfigError(f"{path}: unknown fields: {', '.join(unknown)}")


def _nonempty_string(value: Mapping[str, Any], key: str, path: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item.strip():
        raise ConfigError(f"{path}.{key}: expected non-empty string")
    return item


def _positive_integer(value: Mapping[str, Any], key: str, path: str) -> int:
    item = value.get(key)
    if isinstance(item, bool) or not isinstance(item, int) or item <= 0:
        raise ConfigError(f"{path}.{key}: expected positive integer")
    return item


def _validate_relative_raw_root(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or not path.parts or path.parts[0] != "data":
        raise ConfigError("OkxSourceList.raw_root: expected repository-relative path below data/")
    return path


def _validate_date(value: str, kind: str, path: str) -> None:
    try:
        if kind == "funding":
            datetime.strptime(value, "%Y-%m")
        else:
            date.fromisoformat(value)
    except ValueError as error:
        expected = "YYYY-MM" if kind == "funding" else "YYYY-MM-DD"
        raise ConfigError(f"{path}.date: expected {expected}") from error


def load_okx_source_list(
    path: str | Path, repository_root: str | Path | None = None
) -> OkxSourceList:
    source_path = Path(path).resolve()
    if not source_path.is_file():
        raise ConfigError(f"source list not found: {source_path}")
    root = (
        Path(repository_root).resolve()
        if repository_root is not None
        else discover_repository_root(source_path)
    )
    source_bytes = source_path.read_bytes()
    try:
        data = tomllib.loads(source_bytes.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise ConfigError(f"invalid TOML {source_path}: {error}") from error
    _expect_exact_keys(data, SOURCE_LIST_KEYS, "OkxSourceList")
    if _nonempty_string(data, "schema_version", "OkxSourceList") != "okx-source-list-v1":
        raise ConfigError("OkxSourceList.schema_version: expected 'okx-source-list-v1'")
    dataset_id = _nonempty_string(data, "dataset_id", "OkxSourceList")
    venue = _nonempty_string(data, "venue", "OkxSourceList")
    if venue != "OKX":
        raise ConfigError("OkxSourceList.venue: expected 'OKX'")
    raw_root = _validate_relative_raw_root(
        _nonempty_string(data, "raw_root", "OkxSourceList")
    )
    download_workers = _positive_integer(data, "download_workers", "OkxSourceList")
    if download_workers > 20:
        raise ConfigError("OkxSourceList.download_workers: expected at most 20")
    request_timeout_seconds = _positive_integer(
        data, "request_timeout_seconds", "OkxSourceList"
    )
    chunk_size_bytes = _positive_integer(data, "chunk_size_bytes", "OkxSourceList")
    assets_data = data.get("assets")
    if not isinstance(assets_data, list) or not assets_data:
        raise ConfigError("OkxSourceList.assets: expected non-empty table array")

    assets: list[OkxSourceAsset] = []
    seen_ids: set[str] = set()
    seen_paths: set[Path] = set()
    for index, item in enumerate(assets_data):
        path_label = f"OkxSourceList.assets[{index}]"
        if not isinstance(item, dict):
            raise ConfigError(f"{path_label}: expected table")
        _expect_exact_keys(item, ASSET_KEYS, path_label)
        asset_id = _nonempty_string(item, "asset_id", path_label)
        kind = _nonempty_string(item, "kind", path_label)
        instrument_type = _nonempty_string(item, "instrument_type", path_label)
        instrument = _nonempty_string(item, "instrument", path_label)
        date_text = _nonempty_string(item, "date", path_label)
        archive_format = _nonempty_string(item, "archive_format", path_label)
        expected_size = _positive_integer(item, "expected_size_bytes", path_label)
        url = _nonempty_string(item, "url", path_label)
        if asset_id in seen_ids:
            raise ConfigError(f"{path_label}.asset_id: duplicate {asset_id!r}")
        if kind not in ALLOWED_KINDS:
            raise ConfigError(f"{path_label}.kind: unsupported {kind!r}")
        if archive_format not in ALLOWED_FORMATS:
            raise ConfigError(f"{path_label}.archive_format: unsupported {archive_format!r}")
        expected_format = "tar_gzip_ndjson" if kind == "orderbook_l2" else "zip_csv"
        if archive_format != expected_format:
            raise ConfigError(
                f"{path_label}.archive_format: expected {expected_format!r} for {kind}"
            )
        if instrument_type not in {"SPOT", "SWAP"}:
            raise ConfigError(f"{path_label}.instrument_type: expected SPOT or SWAP")
        if kind == "funding" and instrument_type != "SWAP":
            raise ConfigError(f"{path_label}: funding requires SWAP")
        _validate_date(date_text, kind, path_label)
        parsed_url = urlsplit(url)
        if parsed_url.scheme != "https" or parsed_url.hostname != "static.okx.com":
            raise ConfigError(f"{path_label}.url: expected https://static.okx.com URL")
        asset = OkxSourceAsset(
            asset_id=asset_id,
            kind=kind,
            instrument_type=instrument_type,
            instrument=instrument,
            date=date_text,
            archive_format=archive_format,
            expected_size_bytes=expected_size,
            url=url,
        )
        relative_path = asset.relative_path()
        if relative_path in seen_paths:
            raise ConfigError(f"{path_label}.url: duplicate output path {relative_path}")
        seen_ids.add(asset_id)
        seen_paths.add(relative_path)
        assets.append(asset)

    return OkxSourceList(
        source_path=source_path,
        source_sha256=hashlib.sha256(source_bytes).hexdigest(),
        repository_root=root,
        dataset_id=dataset_id,
        venue=venue,
        raw_root=raw_root,
        download_workers=download_workers,
        request_timeout_seconds=request_timeout_seconds,
        chunk_size_bytes=chunk_size_bytes,
        assets=tuple(assets),
    )


def _hash_existing_prefix(path: Path, chunk_size: int) -> hashlib._Hash:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest


def _download_asset(source_list: OkxSourceList, asset: OkxSourceAsset) -> dict[str, Any]:
    destination = source_list.dataset_directory / asset.relative_path()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        size = destination.stat().st_size
        if size != asset.expected_size_bytes:
            raise ExperimentError(
                f"existing asset has wrong size: {destination} ({size} != {asset.expected_size_bytes})"
            )
        return {
            "asset_id": asset.asset_id,
            "kind": asset.kind,
            "instrument_type": asset.instrument_type,
            "instrument": asset.instrument,
            "date": asset.date,
            "url": asset.url,
            "relative_path": destination.relative_to(source_list.dataset_directory).as_posix(),
            "size_bytes": size,
            "sha256": sha256_file(destination),
            "downloaded": False,
            "etag": None,
            "last_modified": None,
            "content_md5": None,
        }

    partial = destination.with_name(destination.name + ".part")
    offset = partial.stat().st_size if partial.is_file() else 0
    if offset > asset.expected_size_bytes:
        raise ExperimentError(f"partial asset exceeds expected size: {partial}")
    headers = {"User-Agent": "ot-micromr/0.1 data-feasibility"}
    mode = "wb"
    digest = hashlib.sha256()
    if offset:
        headers["Range"] = f"bytes={offset}-"
        mode = "ab"
        digest = _hash_existing_prefix(partial, source_list.chunk_size_bytes)
    request = urllib.request.Request(asset.url, headers=headers)
    try:
        response = urllib.request.urlopen(request, timeout=source_list.request_timeout_seconds)
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as error:
        raise ExperimentError(f"download failed for {asset.asset_id}: {error}") from error
    if offset and getattr(response, "status", None) != 206:
        response.close()
        headers.pop("Range", None)
        request = urllib.request.Request(asset.url, headers=headers)
        try:
            response = urllib.request.urlopen(
                request, timeout=source_list.request_timeout_seconds
            )
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as error:
            raise ExperimentError(f"restart download failed for {asset.asset_id}: {error}") from error
        offset = 0
        mode = "wb"
        digest = hashlib.sha256()
    with response:
        status = getattr(response, "status", None)
        response_headers = response.headers
        with partial.open(mode) as output:
            while True:
                chunk = response.read(source_list.chunk_size_bytes)
                if not chunk:
                    break
                output.write(chunk)
                digest.update(chunk)
            output.flush()
            os.fsync(output.fileno())
        etag = response_headers.get("ETag")
        last_modified = response_headers.get("Last-Modified")
        content_md5 = response_headers.get("Content-MD5")

    size = partial.stat().st_size
    if size != asset.expected_size_bytes:
        raise ExperimentError(
            f"downloaded asset has wrong size: {asset.asset_id} ({size} != {asset.expected_size_bytes})"
        )
    os.replace(partial, destination)
    directory_descriptor = os.open(destination.parent, os.O_RDONLY)
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)
    return {
        "asset_id": asset.asset_id,
        "kind": asset.kind,
        "instrument_type": asset.instrument_type,
        "instrument": asset.instrument,
        "date": asset.date,
        "url": asset.url,
        "relative_path": destination.relative_to(source_list.dataset_directory).as_posix(),
        "size_bytes": size,
        "sha256": digest.hexdigest(),
        "downloaded": True,
        "etag": etag,
        "last_modified": last_modified,
        "content_md5": content_md5,
    }


def canonical_dataset_sha256(dataset_id: str, records: list[Mapping[str, Any]]) -> str:
    identity = {
        "dataset_id": dataset_id,
        "assets": [
            {
                "asset_id": str(record["asset_id"]),
                "size_bytes": int(record["size_bytes"]),
                "sha256": str(record["sha256"]),
            }
            for record in sorted(records, key=lambda item: str(item["asset_id"]))
        ],
    }
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def fetch_okx_source_list(source_list: OkxSourceList) -> dict[str, Any]:
    source_list.dataset_directory.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=source_list.download_workers) as executor:
        futures = {
            executor.submit(_download_asset, source_list, asset): asset.asset_id
            for asset in source_list.assets
        }
        for future in as_completed(futures):
            records.append(future.result())
    records.sort(key=lambda item: str(item["asset_id"]))
    manifest = {
        "schema_version": "okx-raw-manifest-v1",
        "dataset_id": source_list.dataset_id,
        "venue": source_list.venue,
        "dataset_content_sha256": canonical_dataset_sha256(source_list.dataset_id, records),
        "source_spec_path": source_list.source_path.relative_to(
            source_list.repository_root
        ).as_posix(),
        "source_spec_sha256": source_list.source_sha256,
        "retrieved_at_utc": utc_text(datetime.now(UTC)),
        "assets": records,
    }
    atomic_write_json(source_list.manifest_path, manifest)
    return manifest


def load_raw_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise ExperimentError(f"raw manifest not found: {manifest_path}")
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ExperimentError(f"invalid raw manifest {manifest_path}: {error}") from error
    if not isinstance(value, dict) or value.get("schema_version") != "okx-raw-manifest-v1":
        raise ExperimentError(f"unsupported raw manifest: {manifest_path}")
    assets = value.get("assets")
    if not isinstance(assets, list) or not assets:
        raise ExperimentError(f"raw manifest assets are missing: {manifest_path}")
    expected_hash = canonical_dataset_sha256(str(value.get("dataset_id")), assets)
    if value.get("dataset_content_sha256") != expected_hash:
        raise ExperimentError("raw manifest dataset_content_sha256 is inconsistent")
    return value
