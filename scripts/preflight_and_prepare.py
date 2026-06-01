#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fail-fast preflight and optional WebDataset conversion for optimized training."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

from optimization_common import (
    PROJECT_ROOT,
    WDS_KEY_VERSION,
    default_wds_dir,
    detect_optimized_mode,
    file_fingerprint,
    machine_profile,
    parse_hydra_style,
    shard_stats,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preflight optimized training inputs and WDS state.")
    parser.add_argument("overrides", nargs="*", help="Hydra-style overrides")
    parser.add_argument("--experiment", default=None)
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--protocol_path", default=None)
    parser.add_argument("--wds_data_dir", default=None)
    parser.add_argument("--output_dir", default="logs/optimized_configs")
    parser.add_argument("--shard_size_mb", type=int, default=1024)
    parser.add_argument("--no_convert", action="store_true")
    parser.add_argument("--force_convert", action="store_true", help="Rebuild WebDataset shards even when manifest is current")
    return parser.parse_args()


def _is_manifest_current(manifest_path: Path, protocol_path: Path) -> bool:
    if not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    current = file_fingerprint(protocol_path)
    previous = manifest.get("protocol", {})
    converter = manifest.get("converter", {})
    return (
        previous.get("sha256") == current.get("sha256")
        and converter.get("key_version") == WDS_KEY_VERSION
    )


def _write_md(report: Dict[str, Any], path: Path) -> None:
    lines: List[str] = [
        "# Training Speed Preflight",
        "",
        f"- Status: `{report['status']}`",
        f"- Optimized mode: `{report['optimized_mode']}`",
        f"- Data dir: `{report['data_dir']}`",
        f"- Protocol: `{report['protocol_path']}`",
        f"- WDS dir: `{report['wds_data_dir']}`",
        f"- Convert action: `{report['convert_action']}`",
        "",
        "## Checks",
    ]
    for check in report["checks"]:
        lines.append(f"- `{check['status']}` {check['name']}: {check['message']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    parsed = parse_hydra_style(args.overrides)
    experiment = args.experiment or parsed.get("experiment")
    data_dir = args.data_dir or parsed.get("data_dir")
    protocol_path = args.protocol_path or parsed.get("protocol_path")
    wds_data_dir = args.wds_data_dir or parsed.get("wds_data_dir")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checks: List[Dict[str, str]] = []

    def check(name: str, ok: bool, message: str) -> None:
        checks.append({"name": name, "status": "pass" if ok else "fail", "message": message})

    check("data_dir_present", bool(data_dir), "data_dir provided" if data_dir else "data_dir missing")
    check("protocol_present", bool(protocol_path), "protocol_path provided" if protocol_path else "protocol_path missing")

    data_path = Path(data_dir) if data_dir else Path("")
    protocol = Path(protocol_path) if protocol_path else Path("")
    if data_dir:
        check("data_dir_exists", data_path.exists(), str(data_path))
    if protocol_path:
        check("protocol_exists", protocol.exists(), str(protocol))

    optimized = detect_optimized_mode(experiment, wds_data_dir)
    wds_dir = Path(default_wds_dir(str(data_path), wds_data_dir))
    profile = machine_profile([PROJECT_ROOT, data_path if data_dir else PROJECT_ROOT, wds_dir])
    ram = profile["ram"]
    if ram.get("available"):
        check("ram_headroom", ram["free_ratio"] >= 0.10, f"free={ram['free_ratio']:.1%}")
    for disk in profile["disk"]:
        check(f"disk_headroom:{disk['path']}", disk["free_ratio"] >= 0.10, f"free={disk['free_ratio']:.1%}")

    convert_action = "skip"
    manifest_current = False
    if optimized:
        stats = shard_stats(wds_dir) if wds_dir.exists() else {}
        has_train = bool(stats.get("train", {}).get("shard_count", 0))
        manifest_current = protocol.exists() and _is_manifest_current(wds_dir / "manifest.json", protocol)
        if args.force_convert:
            convert_action = "convert_forced"
        elif not has_train:
            convert_action = "convert_missing"
        elif not manifest_current:
            convert_action = "convert_stale"
        check("wds_available", has_train, f"train shards={stats.get('train', {}).get('shard_count', 0)}")
    else:
        stats = {}

    failed_before_convert = any(c["status"] == "fail" for c in checks if c["name"] not in {"wds_available"})
    if failed_before_convert:
        convert_action = "blocked"

    if optimized and convert_action.startswith("convert") and not args.no_convert and not failed_before_convert:
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "convert_to_wds.py"),
            "--input_wav_dir",
            str(data_path),
            "--protocol_path",
            str(protocol),
            "--output_dir",
            str(wds_dir),
            "--shard_size_mb",
            str(args.shard_size_mb),
        ]
        started = time.perf_counter()
        result = subprocess.run(cmd, cwd=PROJECT_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        convert_action += "_done" if result.returncode == 0 else "_failed"
        checks.append(
            {
                "name": "convert",
                "status": "pass" if result.returncode == 0 else "fail",
                "message": f"returncode={result.returncode} elapsed={time.perf_counter() - started:.1f}s",
            }
        )
        stats = shard_stats(wds_dir) if wds_dir.exists() else {}
        if result.returncode == 0:
            for item in checks:
                if item["name"] == "wds_available":
                    item["status"] = "pass"
                    item["message"] = f"train shards={stats.get('train', {}).get('shard_count', 0)}"
                    break
    elif optimized and convert_action.startswith("convert") and args.no_convert:
        check("convert_required", False, f"{convert_action}; rerun without --no_convert")

    status = "pass" if all(c["status"] == "pass" for c in checks) else "fail"
    report: Dict[str, Any] = {
        "status": status,
        "created_at_unix": time.time(),
        "experiment": experiment,
        "optimized_mode": optimized,
        "data_dir": str(data_path),
        "protocol_path": str(protocol),
        "wds_data_dir": str(wds_dir),
        "manifest_current": manifest_current,
        "convert_action": convert_action,
        "checks": checks,
        "machine_profile": profile,
        "shards": stats,
    }
    write_json(output_dir / "preflight_report.json", report)
    _write_md(report, output_dir / "preflight_report.md")
    print(json.dumps({"status": status, "output_dir": str(output_dir), "convert_action": convert_action}, indent=2))
    return 0 if status == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
