#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Rank warm-up results and write chosen optimized training config."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List

from optimization_common import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select optimized config from warm-up reports.")
    parser.add_argument("--input_dir", default="logs/optimized_configs")
    parser.add_argument("--output_dir", default="logs/optimized_configs")
    return parser.parse_args()


def _load(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required report missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _score(row: Dict[str, Any], max_workers_per_rank: int) -> float:
    if row.get("status") != "pass":
        return -1.0
    if max_workers_per_rank > 0 and int(row.get("num_workers", 0)) > max_workers_per_rank:
        return -1.0
    score = float(row.get("samples_per_sec", 0.0))
    ttfb = row.get("time_to_first_batch")
    if ttfb is not None:
        score -= min(float(ttfb), 10.0) * 0.1
    return score


def _write_md(path: Path, chosen: Dict[str, Any], preflight: Dict[str, Any], train: Dict[str, Any], rows: List[Dict[str, Any]]) -> None:
    command = [
        "python src/train.py",
        f"++data.num_workers={chosen.get('num_workers')}",
        f"++data.pin_memory={str(chosen.get('pin_memory')).lower()}",
        f"++data.args.wds_data_dir={chosen.get('wds_data_dir')}",
        f"++data.args.wds_prefetch_factor={chosen.get('prefetch_factor')}",
        f"++data.args.wds_persistent_workers={str(chosen.get('persistent_workers')).lower()}",
        "++trainer.strategy=ddp_find_unused_parameters_true",
    ]
    lines = [
        "# Optimized Training Config",
        "",
        "## Machine Profile",
        f"- CPU count: `{preflight.get('machine_profile', {}).get('cpu_count')}`",
        f"- GPU available: `{preflight.get('machine_profile', {}).get('gpu', {}).get('available')}`",
        f"- RAM free ratio: `{preflight.get('machine_profile', {}).get('ram', {}).get('free_ratio')}`",
        "",
        "## Dataset Profile",
        f"- Source path: `{preflight.get('data_dir')}`",
        f"- Protocol: `{preflight.get('protocol_path')}`",
        f"- WDS path: `{chosen.get('wds_data_dir')}`",
        "",
        "## Benchmark Matrix",
        f"- Data configs tried: {len(rows)}",
        f"- Micro-train status: `{train.get('status')}`",
        f"- Micro-train step/sec: `{train.get('global_step_per_sec', 0.0):.4f}`",
        "",
        "## Chosen Config",
        f"- num_workers: `{chosen.get('num_workers')}`",
        f"- prefetch_factor: `{chosen.get('prefetch_factor')}`",
        f"- pin_memory: `{chosen.get('pin_memory')}`",
        f"- persistent_workers: `{chosen.get('persistent_workers')}`",
        f"- cache_mode: `{chosen.get('cache_mode')}`",
        f"- hot_shard_window: `{chosen.get('hot_shard_window')}`",
        f"- samples_per_sec: `{chosen.get('samples_per_sec')}`",
        "",
        "## Reason",
        "Selected highest passing data-only throughput with no read/decode errors, after preflight and micro-train gate passed.",
        "",
        "## Rendered Train Overrides",
        "```bash",
        " \\\n  ".join(command),
        "```",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        preflight = _load(input_dir / "preflight_report.json")
        data = _load(input_dir / "warmup_data.json")
        train = _load(input_dir / "warmup_train.json")
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        payload = {"status": "fail", "error": str(exc), "created_at_unix": time.time()}
        write_json(output_dir / "chosen_config.json", payload)
        print(json.dumps(payload, indent=2))
        return 2
    rows = list(data.get("results", []))
    train_shards = int(preflight.get("shards", {}).get("train", {}).get("shard_count", 0))
    gpu_count = len(preflight.get("machine_profile", {}).get("gpu", {}).get("gpus", [])) or 1
    max_workers_per_rank = max(1, math.floor(train_shards / gpu_count)) if train_shards > 0 else 0
    ranked = sorted(rows, key=lambda r: _score(r, max_workers_per_rank), reverse=True)
    chosen = ranked[0] if ranked and _score(ranked[0], max_workers_per_rank) >= 0 else {}
    status = "pass" if preflight.get("status") == "pass" and chosen and train.get("status") in {"pass", "skipped"} else "fail"
    payload = {
        "status": status,
        "created_at_unix": time.time(),
        "chosen": chosen,
        "preflight_status": preflight.get("status"),
        "micro_train_status": train.get("status"),
        "max_workers_per_rank": max_workers_per_rank,
        "ranked": ranked,
    }
    write_json(output_dir / "chosen_config.json", payload)
    _write_md(output_dir / "optimized_config.md", chosen, preflight, train, rows)
    print(json.dumps({"status": status, "chosen": chosen}, indent=2))
    return 0 if status == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
