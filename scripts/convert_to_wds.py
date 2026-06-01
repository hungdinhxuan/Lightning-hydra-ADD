#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convert protocol-based wav dataset to WebDataset shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

# Ensure `src` is importable when running this file directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import webdataset as wds
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Missing dependency 'webdataset'. Install it with `uv sync` "
        "or `pip install webdataset` and re-run this script."
    ) from exc

from src.data.dataset_optimized import read_protocol, split_entries_by_subset
from src.data.wds_keys import WDS_KEY_VERSION


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _load_bytes(path: Path) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def _file_fingerprint(path: Path) -> Dict[str, object]:
    stat = path.stat()
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(path),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": digest.hexdigest(),
    }


def _write_subset(
    subset: str,
    entries,
    input_wav_dir: Path,
    output_dir: Path,
    shard_size_bytes: int,
) -> Dict[str, int]:
    pattern = output_dir / f"{subset}-%04d.tar"
    _ensure_parent(pattern)
    n_samples = 0
    skipped = 0

    with wds.ShardWriter(str(pattern), maxsize=shard_size_bytes) as sink:
        for e in entries:
            wav_path = input_wav_dir / e.relpath
            if not wav_path.exists():
                skipped += 1
                continue

            key = e.key
            sample = {
                "__key__": key,
                "wav": _load_bytes(wav_path),
                "label": e.label_str,
                "json": json.dumps(
                    {
                        "relpath": e.relpath,
                        "subset": subset,
                        "label": e.label_str,
                    }
                ).encode("utf-8"),
            }
            sink.write(sample)
            n_samples += 1

    return {"written": n_samples, "skipped": skipped}


def build_webdataset(
    input_wav_dir: str,
    protocol_path: str,
    output_dir: str,
    shard_size_mb: int = 256,
) -> Dict[str, Dict[str, int]]:
    input_root = Path(input_wav_dir)
    protocol = Path(protocol_path)
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    entries = read_protocol(str(protocol))
    by_subset = split_entries_by_subset(entries)
    maxsize = shard_size_mb * 1024 * 1024

    summary: Dict[str, Dict[str, int]] = {}
    for subset in ("train", "dev", "eval"):
        for stale_shard in out_root.glob(f"{subset}-*.tar"):
            stale_shard.unlink()
        summary[subset] = _write_subset(
            subset=subset,
            entries=by_subset.get(subset, []),
            input_wav_dir=input_root,
            output_dir=out_root,
            shard_size_bytes=maxsize,
        )
    return summary


def build_manifest(input_wav_dir: str, protocol_path: str, output_dir: str, summary: Dict[str, Dict[str, int]]) -> Dict[str, object]:
    out_root = Path(output_dir)
    shards: List[Dict[str, object]] = []
    for shard in sorted(out_root.glob("*.tar")):
        stat = shard.stat()
        shards.append(
            {
                "path": str(shard),
                "name": shard.name,
                "size_bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    return {
        "created_at_unix": time.time(),
        "converter": {
            "key_version": WDS_KEY_VERSION,
        },
        "input_wav_dir": str(Path(input_wav_dir).resolve()),
        "protocol": _file_fingerprint(Path(protocol_path)),
        "output_dir": str(out_root.resolve()),
        "summary": summary,
        "shards": shards,
    }


def write_reports(output_dir: str, manifest: Dict[str, object]) -> None:
    out_root = Path(output_dir)
    manifest_path = out_root / "manifest.json"
    report_json = out_root / "convert_report.json"
    report_md = out_root / "convert_report.md"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    report_json.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    summary = manifest["summary"]
    shards = manifest["shards"]
    lines = [
        "# WebDataset Convert Report",
        "",
        f"- Input: `{manifest['input_wav_dir']}`",
        f"- Protocol: `{manifest['protocol']['path']}`",
        f"- Output: `{manifest['output_dir']}`",
        f"- Shards: {len(shards)}",
        "",
        "| Split | Written | Skipped |",
        "| --- | ---: | ---: |",
    ]
    for split in ("train", "dev", "eval"):
        stats = summary.get(split, {})
        lines.append(f"| {split} | {stats.get('written', 0)} | {stats.get('skipped', 0)} |")
    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert wav+protocol dataset to WebDataset shards.")
    parser.add_argument("--input_wav_dir", required=True, help="Directory containing wav files")
    parser.add_argument("--protocol_path", required=True, help="Path to protocol.txt")
    parser.add_argument("--output_dir", required=True, help="Output directory for train/dev/eval shards")
    parser.add_argument("--shard_size_mb", type=int, default=1024, help="Shard size target in MB (1GB recommended)")
    parser.add_argument(
        "--use_dev_shm",
        action="store_true",
        help="Write shards under /dev/shm/<basename(output_dir)> for faster IO when available",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    if args.use_dev_shm and os.path.isdir("/dev/shm"):
        output_dir = str(Path("/dev/shm") / Path(output_dir).name)

    summary = build_webdataset(
        input_wav_dir=args.input_wav_dir,
        protocol_path=args.protocol_path,
        output_dir=output_dir,
        shard_size_mb=args.shard_size_mb,
    )
    manifest = build_manifest(
        input_wav_dir=args.input_wav_dir,
        protocol_path=args.protocol_path,
        output_dir=output_dir,
        summary=summary,
    )
    write_reports(output_dir, manifest)
    print(f"WebDataset shards written to: {output_dir}")
    for subset, stats in summary.items():
        print(f"{subset}: written={stats['written']} skipped={stats['skipped']}")


if __name__ == "__main__":
    main()
