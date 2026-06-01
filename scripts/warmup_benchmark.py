#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run data-only and optional micro-train warm-up benchmarks."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
import math
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List

from optimization_common import (
    PROJECT_ROOT,
    default_wds_dir,
    ensure_project_root_on_path,
    machine_profile,
    parse_hydra_style,
    write_json,
)

ensure_project_root_on_path()


def parse_bool(value: str) -> bool:
    return str(value).lower() in {"1", "true", "yes", "y"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark WebDataset/DataLoader warm-up configs.")
    parser.add_argument("overrides", nargs="*", help="Hydra-style overrides for final train command")
    parser.add_argument("--experiment", default=None)
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--protocol_path", default=None)
    parser.add_argument("--wds_data_dir", default=None)
    parser.add_argument("--output_dir", default="logs/optimized_configs")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", default="2,4,8,12")
    parser.add_argument("--prefetch_factor", default="2,4,8")
    parser.add_argument("--pin_memory", default="true,false")
    parser.add_argument("--persistent_workers", default="true,false")
    parser.add_argument("--max_batches", type=int, default=50)
    parser.add_argument("--data-only-first", default="true")
    parser.add_argument("--micro-train-steps", type=int, default=300)
    parser.add_argument("--skip_micro_train", action="store_true")
    return parser.parse_args()


def _csv_ints(raw: str) -> List[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _csv_bools(raw: str) -> List[bool]:
    return [parse_bool(x.strip()) for x in raw.split(",") if x.strip()]


def _bench_loader(cfg: Dict[str, Any]) -> Dict[str, Any]:
    from src.data.dataset_optimized import get_wds_dataloader

    loader = get_wds_dataloader(
        shards_root=cfg["wds_data_dir"],
        subset="train",
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
        pin_memory=cfg["pin_memory"],
        sample_rate=16000,
        trim_length=64000,
        padding_type="repeat",
        random_start=False,
        prefetch_factor=cfg["prefetch_factor"],
        persistent_workers=cfg["persistent_workers"],
    )
    start = time.perf_counter()
    first_batch_at = None
    batches = 0
    samples = 0
    errors = 0
    for batch in loader:
        now = time.perf_counter()
        if first_batch_at is None:
            first_batch_at = now
        try:
            samples += int(batch[0].shape[0])
        except Exception:
            errors += 1
        batches += 1
        if batches >= cfg["max_batches"]:
            break
    elapsed = max(time.perf_counter() - start, 1e-9)
    steady_elapsed = max(time.perf_counter() - (first_batch_at or start), 1e-9)
    return {
        **cfg,
        "time_to_first_batch": (first_batch_at - start) if first_batch_at else None,
        "elapsed_sec": elapsed,
        "batches": batches,
        "samples": samples,
        "steady_state_batches_per_sec": batches / steady_elapsed,
        "samples_per_sec": samples / elapsed,
        "read_decode_error_count": errors,
        "status": "pass" if batches > 0 and errors == 0 else "fail",
    }


def _run_micro_train(args: argparse.Namespace, parsed: Dict[str, Any], chosen: Dict[str, Any]) -> Dict[str, Any]:
    if args.skip_micro_train or args.micro_train_steps <= 0:
        return {"status": "skipped"}
    overrides = []
    for token in parsed.get("overrides", []):
        key = token.split("=", 1)[0].lstrip("+")
        if key in {"data_dir", "protocol_path"}:
            continue
        overrides.append(token)
    overrides.extend(
        [
            f"++data.num_workers={chosen['num_workers']}",
            f"++data.pin_memory={str(chosen['pin_memory']).lower()}",
            f"++data.args.wds_data_dir={chosen['wds_data_dir']}",
            f"++data.args.wds_prefetch_factor={chosen['prefetch_factor']}",
            f"++data.args.wds_persistent_workers={str(chosen['persistent_workers']).lower()}",
            "++trainer.strategy=ddp_find_unused_parameters_true",
            f"++trainer.max_steps={args.micro_train_steps}",
            "++trainer.max_epochs=1",
            "++trainer.limit_val_batches=0",
            "++callbacks.early_stopping=null",
            "++callbacks.model_checkpoint=null",
            "++test=false",
        ]
    )
    cmd = [sys.executable, "src/train.py", *overrides]
    started = time.perf_counter()
    result = subprocess.run(cmd, cwd=PROJECT_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    elapsed = time.perf_counter() - started
    return {
        "status": "pass" if result.returncode == 0 else "fail",
        "returncode": result.returncode,
        "elapsed_sec": elapsed,
        "micro_train_steps": args.micro_train_steps,
        "global_step_per_sec": args.micro_train_steps / elapsed if result.returncode == 0 and elapsed > 0 else 0.0,
        "stdout_tail": result.stdout[-4000:],
        "stderr_tail": result.stderr[-4000:],
        "command": cmd,
    }


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    keys: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_md(output_dir: Path, data_rows: List[Dict[str, Any]], train_report: Dict[str, Any]) -> None:
    best = max(data_rows, key=lambda r: r.get("samples_per_sec", 0.0)) if data_rows else {}
    lines = [
        "# Warm-up Benchmark",
        "",
        f"- Data configs tried: {len(data_rows)}",
        f"- Best samples/sec: {best.get('samples_per_sec', 0):.2f}",
        f"- Best workers: `{best.get('num_workers')}`",
        f"- Best pin_memory: `{best.get('pin_memory')}`",
        f"- Micro-train status: `{train_report.get('status')}`",
    ]
    output_dir.joinpath("warmup_data.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    output_dir.joinpath("warmup_train.md").write_text(json.dumps(train_report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _max_workers_per_rank(wds_data_dir: str) -> int:
    train_shards = len(list(Path(wds_data_dir).glob("train-*.tar")))
    gpu_count = len(machine_profile([Path(wds_data_dir)]).get("gpu", {}).get("gpus", [])) or 1
    if train_shards <= 0:
        return 0
    return max(1, math.floor(train_shards / gpu_count))


def main() -> int:
    args = parse_args()
    parsed = parse_hydra_style(args.overrides)
    data_dir = args.data_dir or parsed.get("data_dir")
    wds_data_dir = args.wds_data_dir or parsed.get("wds_data_dir") or default_wds_dir(data_dir or "data")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for workers in _csv_ints(args.num_workers):
        for prefetch in _csv_ints(args.prefetch_factor):
            for pin in _csv_bools(args.pin_memory):
                for persistent in _csv_bools(args.persistent_workers):
                    if workers == 0 and (prefetch != 2 or persistent):
                        continue
                    cfg = {
                        "wds_data_dir": wds_data_dir,
                        "batch_size": args.batch_size,
                        "num_workers": workers,
                        "prefetch_factor": prefetch,
                        "pin_memory": pin,
                        "persistent_workers": persistent and workers > 0,
                        "cache_mode": "none",
                        "hot_shard_window": 0,
                        "max_batches": args.max_batches,
                    }
                    try:
                        rows.append(_bench_loader(cfg))
                    except Exception as exc:
                        rows.append({**cfg, "status": "fail", "error": str(exc), "samples_per_sec": 0.0})

    _write_csv(output_dir / "warmup_data.csv", rows)
    write_json(output_dir / "warmup_data.json", {"machine_profile": machine_profile([Path(wds_data_dir)]), "results": rows})
    max_workers = _max_workers_per_rank(wds_data_dir)
    passed = [r for r in rows if r.get("status") == "pass" and int(r.get("num_workers", 0)) <= max_workers]
    chosen = max(passed, key=lambda r: r.get("samples_per_sec", 0.0)) if passed else {}
    train_report = _run_micro_train(args, parsed, chosen) if chosen else {"status": "fail", "error": "no passing data config"}
    write_json(output_dir / "warmup_train.json", train_report)
    _write_md(output_dir, rows, train_report)
    print(json.dumps({"data_configs": len(rows), "data_passed": len(passed), "micro_train": train_report.get("status")}, indent=2))
    return 0 if passed and train_report.get("status") in {"pass", "skipped"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
