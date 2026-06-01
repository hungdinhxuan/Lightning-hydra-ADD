#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run preflight, warm-up, config selection, then real training."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import List

from optimization_common import PROJECT_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gate real training on optimized warm-up reports.")
    parser.add_argument("overrides", nargs="*", help="Hydra overrides passed to src/train.py")
    parser.add_argument("--optimized-config-dir", default="logs/optimized_configs")
    parser.add_argument("--skip_existing", action="store_true", help="Reuse existing passing chosen_config.json")
    parser.add_argument("--micro-train-steps", type=int, default=300)
    parser.add_argument("--max-warmup-batches", type=int, default=50)
    return parser.parse_args()


def _run(cmd: List[str]) -> None:
    print("+ " + " ".join(cmd), flush=True)
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def _load_chosen(path: Path):
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if payload.get("status") == "pass" else None


def _has_override(overrides: List[str], key: str) -> bool:
    prefixes = (f"{key}=", f"+{key}=", f"++{key}=")
    return any(override.startswith(prefixes) for override in overrides)


def main() -> int:
    args = parse_args()
    out_dir = Path(args.optimized_config_dir)
    chosen_path = out_dir / "chosen_config.json"
    chosen_payload = _load_chosen(chosen_path) if args.skip_existing else None

    if chosen_payload is None:
        _run([sys.executable, "scripts/preflight_and_prepare.py", *args.overrides, "--output_dir", str(out_dir)])
        _run(
            [
                sys.executable,
                "scripts/warmup_benchmark.py",
                *args.overrides,
                "--output_dir",
                str(out_dir),
                "--micro-train-steps",
                str(args.micro_train_steps),
                "--max_batches",
                str(args.max_warmup_batches),
            ]
        )
        _run([sys.executable, "scripts/select_optimized_config.py", "--input_dir", str(out_dir), "--output_dir", str(out_dir)])
        chosen_payload = _load_chosen(chosen_path)

    if chosen_payload is None:
        raise SystemExit("No passing chosen_config.json; real training blocked.")

    chosen = chosen_payload["chosen"]
    train_cmd = [
        sys.executable,
        "src/train.py",
        *args.overrides,
        f"++data.num_workers={chosen['num_workers']}",
        f"++data.pin_memory={str(chosen['pin_memory']).lower()}",
        f"++data.args.wds_data_dir={chosen['wds_data_dir']}",
        f"++data.args.wds_prefetch_factor={chosen['prefetch_factor']}",
        f"++data.args.wds_persistent_workers={str(chosen['persistent_workers']).lower()}",
    ]
    if not _has_override(args.overrides, "trainer.strategy"):
        train_cmd.append("++trainer.strategy=ddp")
    _run(train_cmd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
