from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import mimetypes
import os
import platform
import shutil
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary_path.unlink(missing_ok=True)


def atomic_write_json(path: Path, value: Any) -> None:
    content = json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
    atomic_write_bytes(path, (content + "\n").encode("utf-8"))


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def copy_source(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as input_stream, destination.open("xb") as output_stream:
        shutil.copyfileobj(input_stream, output_stream)
        output_stream.flush()
        os.fsync(output_stream.fileno())


def _git(repository_root: Path, *arguments: str, check: bool = True) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if check and result.returncode != 0:
        raise RuntimeError(f"git {' '.join(arguments)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def git_provenance(repository_root: Path) -> dict[str, Any]:
    status = _git(repository_root, "status", "--porcelain=v1", "--untracked-files=all")
    remote = _git(repository_root, "remote", "get-url", "origin", check=False) or None
    return {
        "remote_origin": remote,
        "commit": _git(repository_root, "rev-parse", "HEAD"),
        "branch": _git(repository_root, "branch", "--show-current") or None,
        "dirty": bool(status),
        "status_porcelain": status.splitlines(),
    }


def environment_provenance(repository_root: Path) -> dict[str, Any]:
    dependencies: dict[str, str] = {}
    for distribution in ("ot-micromr", "numpy", "scipy", "matplotlib", "torch"):
        try:
            dependencies[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            dependencies[distribution] = "not-installed"
    lock_path = repository_root / "uv.lock"
    cpu_model = platform.processor() or platform.machine()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith("model name") and ":" in line:
                cpu_model = line.split(":", 1)[1].strip()
                break
    available_memory_bytes: int | None = None
    if hasattr(os, "sysconf"):
        try:
            available_memory_bytes = int(os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
        except (OSError, ValueError):
            pass
    gpu: dict[str, Any] = {"available": False, "device": None, "cuda_runtime": None}
    try:
        import torch
    except ImportError:
        pass
    else:
        gpu["available"] = bool(torch.cuda.is_available())
        gpu["cuda_runtime"] = torch.version.cuda
        if torch.cuda.is_available():
            gpu["device"] = torch.cuda.get_device_name(0)
            gpu["compute_capability"] = list(torch.cuda.get_device_capability(0))
    return {
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
        },
        "platform": platform.platform(),
        "dependencies": dependencies,
        "uv_lock_sha256": sha256_file(lock_path) if lock_path.is_file() else None,
        "hardware": {
            "cpu_model": cpu_model,
            "machine": platform.machine(),
            "logical_cpu_count": os.cpu_count(),
            "available_memory_bytes_at_start": available_memory_bytes,
            "gpu": gpu,
        },
    }


def utc_now() -> datetime:
    return datetime.now(UTC)


def utc_text(value: datetime) -> str:
    return value.isoformat(timespec="microseconds").replace("+00:00", "Z")


def artifact_inventory(run_directory: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(run_directory.rglob("*")):
        if not path.is_file() or path.name == "manifest.json":
            continue
        media_type, _ = mimetypes.guess_type(path.name)
        records.append(
            {
                "path": path.relative_to(run_directory).as_posix(),
                "media_type": media_type or "application/octet-stream",
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return records
