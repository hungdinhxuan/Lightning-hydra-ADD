#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared helpers for training-speed gate scripts."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WDS_KEY_VERSION = "safe-no-dot-v1"


def ensure_project_root_on_path() -> None:
    import sys

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))


def parse_hydra_style(argv: Iterable[str]) -> Dict[str, Any]:
    parsed: Dict[str, Any] = {"overrides": []}
    for token in argv:
        if token.startswith("--"):
            parsed["overrides"].append(token)
            continue
        if "=" not in token:
            parsed["overrides"].append(token)
            continue
        key, value = token.split("=", 1)
        clean_key = key.lstrip("+")
        parsed["overrides"].append(token)
        if clean_key in {"experiment", "data_dir", "protocol_path"}:
            parsed[clean_key] = value
        elif clean_key == "data.data_dir":
            parsed["data_dir"] = value
        elif clean_key == "data.args.protocol_path":
            parsed["protocol_path"] = value
        elif clean_key == "data.args.wds_data_dir":
            parsed["wds_data_dir"] = value
    return parsed


def file_fingerprint(path: Path) -> Dict[str, Any]:
    stat = path.stat()
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(path.resolve()),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": digest.hexdigest(),
    }


def disk_report(path: Path) -> Dict[str, Any]:
    path.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(path)
    return {
        "path": str(path),
        "total_bytes": usage.total,
        "used_bytes": usage.used,
        "free_bytes": usage.free,
        "free_ratio": usage.free / usage.total if usage.total else 0.0,
    }


def memory_report() -> Dict[str, Any]:
    meminfo = Path("/proc/meminfo")
    if not meminfo.exists():
        return {"available": False}
    values: Dict[str, int] = {}
    for line in meminfo.read_text(encoding="utf-8").splitlines():
        key, raw = line.split(":", 1)
        values[key] = int(raw.strip().split()[0]) * 1024
    total = values.get("MemTotal", 0)
    free = values.get("MemAvailable", 0)
    return {
        "available": True,
        "total_bytes": total,
        "free_bytes": free,
        "free_ratio": free / total if total else 0.0,
    }


def gpu_report() -> Dict[str, Any]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            cwd=PROJECT_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "error": str(exc), "gpus": []}
    if result.returncode != 0:
        return {"available": False, "error": result.stderr.strip(), "gpus": []}
    gpus: List[Dict[str, Any]] = []
    for line in result.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 5:
            gpus.append(
                {
                    "index": int(parts[0]),
                    "name": parts[1],
                    "memory_total_mb": int(parts[2]),
                    "memory_used_mb": int(parts[3]),
                    "utilization_gpu_pct": int(parts[4]),
                }
            )
    return {"available": bool(gpus), "gpus": gpus}


def detect_optimized_mode(experiment: Optional[str], explicit_wds_dir: Optional[str]) -> bool:
    if explicit_wds_dir:
        return True
    if not experiment:
        return False
    if "optimized" in experiment.lower():
        return True
    cfg = PROJECT_ROOT / "configs" / "experiment" / f"{experiment}.yaml"
    if cfg.exists():
        text = cfg.read_text(encoding="utf-8")
        return "normal_MDT_datamodule_optimized" in text or "wds_data_dir" in text
    return False


def default_wds_dir(data_dir: str, explicit_wds_dir: Optional[str] = None) -> str:
    if explicit_wds_dir:
        return explicit_wds_dir
    env = os.environ.get("WDS_DATA_DIR")
    if env:
        return env
    return str(Path(data_dir) / "wds_data")


def shard_stats(wds_dir: Path) -> Dict[str, Any]:
    by_split: Dict[str, Any] = {}
    for split in ("train", "dev", "eval"):
        shards = sorted(wds_dir.glob(f"{split}-*.tar"))
        by_split[split] = {
            "shard_count": len(shards),
            "size_bytes": sum(p.stat().st_size for p in shards),
            "shards": [p.name for p in shards],
        }
    return by_split


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def machine_profile(paths: Optional[List[Path]] = None) -> Dict[str, Any]:
    paths = paths or [PROJECT_ROOT]
    return {
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "ram": memory_report(),
        "gpu": gpu_report(),
        "disk": [disk_report(p) for p in paths],
        "dev_shm": disk_report(Path("/dev/shm")) if Path("/dev/shm").exists() else {"available": False},
    }
