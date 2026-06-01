#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

DATA_DIR="${DATA_DIR:-/data/dsd_corpus_pool_13May2026}"
SOURCE_WDS_DIR="${WDS_DATA_DIR:-${PROJECT_ROOT}/data/dsd_corpus_pool_13May2026_wds}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_ROOT}/logs/benchmark_dsd_one_shard}"
RUN_NAME="${RUN_NAME:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_ROOT}/${RUN_NAME}"

TRAIN_SHARD="${TRAIN_SHARD:-${SOURCE_WDS_DIR}/train-0000.tar}"
DEV_SHARD="${DEV_SHARD:-${SOURCE_WDS_DIR}/dev-0000.tar}"
EVAL_SHARD="${EVAL_SHARD:-${SOURCE_WDS_DIR}/eval-0000.tar}"

GPUS="${GPUS:-0,1}"
BATCH_SIZE="${BATCH_SIZE:-14}"
NUM_WORKERS="${NUM_WORKERS:-1}"
MAX_EPOCHS="${MAX_EPOCHS:-1}"
SHARD_SIZE_MB="${SHARD_SIZE_MB:-1024}"

mkdir -p "$RUN_DIR"

python - "$RUN_DIR" "$TRAIN_SHARD" "$DEV_SHARD" "$EVAL_SHARD" <<'PY'
import json
import os
import sys
import tarfile
from pathlib import Path

run_dir = Path(sys.argv[1])
train_shard = Path(sys.argv[2]).resolve()
dev_shard = Path(sys.argv[3]).resolve()
eval_shard = Path(sys.argv[4]).resolve()
wds_dir = run_dir / "wds_one_shard"
wds_dir.mkdir(parents=True, exist_ok=True)

for shard in (train_shard, dev_shard):
    if not shard.exists():
        raise SystemExit(f"Missing shard: {shard}")

def link(src: Path, dst_name: str) -> None:
    dst = wds_dir / dst_name
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src)

link(train_shard, "train-0000.tar")
link(dev_shard, "dev-0000.tar")
# WebDataset shards are split at shard granularity in DDP. Two GPU ranks need
# two URLs, so both ranks receive one URL while still using one unique tar.
link(train_shard, "train-0001.tar")
link(dev_shard, "dev-0001.tar")
if eval_shard.exists():
    link(eval_shard, "eval-0000.tar")
    link(eval_shard, "eval-0001.tar")
else:
    link(dev_shard, "eval-0000.tar")
    link(dev_shard, "eval-0001.tar")

def rows_from_shard(path: Path, fallback_subset: str):
    rows = []
    with tarfile.open(path) as tf:
        for member in tf:
            if not member.name.endswith(".json"):
                continue
            f = tf.extractfile(member)
            if f is None:
                continue
            meta = json.loads(f.read().decode("utf-8"))
            rows.append((
                meta["relpath"],
                meta.get("subset", fallback_subset),
                meta["label"],
            ))
    return rows

train_rows = rows_from_shard(train_shard, "train")
dev_rows = rows_from_shard(dev_shard, "dev")
eval_rows = rows_from_shard(eval_shard, "eval") if eval_shard.exists() else []

protocol_path = run_dir / "one_shard_protocol.txt"
with protocol_path.open("w", encoding="utf-8") as f:
    for relpath, _subset, label in train_rows:
        f.write(f"{relpath} train {label}\n")
    for relpath, _subset, label in dev_rows:
        f.write(f"{relpath} dev {label}\n")
    for relpath, _subset, label in eval_rows:
        f.write(f"{relpath} eval {label}\n")

summary = {
    "protocol_path": str(protocol_path),
    "wds_dir": str(wds_dir),
    "train_shard": str(train_shard),
    "dev_shard": str(dev_shard),
    "eval_shard": str(eval_shard) if eval_shard.exists() else str(dev_shard),
    "train_samples": len(train_rows),
    "dev_samples": len(dev_rows),
    "eval_samples": len(eval_rows),
    "ddp_symlink_duplicate": True,
}
(run_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
PY

PROTOCOL_PATH="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["protocol_path"])' "${RUN_DIR}/dataset_summary.json")"
ONE_WDS_DIR="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["wds_dir"])' "${RUN_DIR}/dataset_summary.json")"
TRAIN_BATCHES="$(python -c 'import json,math,os,sys; d=json.load(open(sys.argv[1])); print(max(d["train_samples"] // int(os.environ.get("BATCH_SIZE","14")), 1))' "${RUN_DIR}/dataset_summary.json")"
VAL_BATCHES="$(python -c 'import json,math,os,sys; d=json.load(open(sys.argv[1])); print(max(math.ceil(d["dev_samples"] / int(os.environ.get("BATCH_SIZE","14"))), 1))' "${RUN_DIR}/dataset_summary.json")"

run_case() {
  local name="$1"
  shift
  local log_path="${RUN_DIR}/${name}.log"
  local metrics_path="${RUN_DIR}/${name}_metrics.jsonl"
  local result_path="${RUN_DIR}/${name}_result.json"
  local start end rc

  echo "Running ${name}..."
  start="$(date +%s)"
  set +e
  CUDA_VISIBLE_DEVICES="$GPUS" \
  SHARD_SIZE_MB="$SHARD_SIZE_MB" \
  python src/train.py "$@" \
    "++data.data_dir=${DATA_DIR}" \
    "++data.args.data_dir=${DATA_DIR}" \
    "++data.args.protocol_path=${PROTOCOL_PATH}" \
    "++data.batch_size=${BATCH_SIZE}" \
    "++data.num_workers=${NUM_WORKERS}" \
    "++data.pin_memory=true" \
    "++data.args.augmentation_methods=[none]" \
    "++data.args.is_dev_aug=false" \
    "++model.compile=false" \
    "++trainer.accelerator=gpu" \
    "++trainer.devices=2" \
    "++trainer.num_nodes=1" \
    "++trainer.strategy=ddp_find_unused_parameters_true" \
    "++trainer.max_epochs=${MAX_EPOCHS}" \
    "++trainer.num_sanity_val_steps=0" \
    "++trainer.log_every_n_steps=10" \
    "logger=json" \
    "test=false" \
    "model_averaging=false" \
    "++json_logging.enabled=true" \
    "++json_logging.log_path=${metrics_path}" \
    "hydra.run.dir=${RUN_DIR}/hydra_${name}" \
    >"$log_path" 2>&1
  rc=$?
  set -e
  end="$(date +%s)"
  python - "$result_path" "$name" "$rc" "$start" "$end" "$log_path" "$metrics_path" <<'PY'
import json
import sys
from pathlib import Path

path, name, rc, start, end, log_path, metrics_path = sys.argv[1:]
payload = {
    "name": name,
    "returncode": int(rc),
    "start_unix": int(start),
    "end_unix": int(end),
    "elapsed_seconds": int(end) - int(start),
    "log_path": log_path,
    "metrics_path": metrics_path,
}
Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
PY
  if [[ "$rc" != "0" ]]; then
    echo "${name} failed. See ${log_path}" >&2
    return "$rc"
  fi
}

run_case "normal" \
  "experiment=xlsr_conformertcm_mdt"

run_case "optimized" \
  "experiment=xlsr_conformertcm_mdt_optimized" \
  "++data.args.wds_data_dir=${ONE_WDS_DIR}" \
  "++data.args.wds_shard_shuffle=0" \
  "++data.args.wds_sample_shuffle=0" \
  "++data.args.wds_train_epoch_batches=${TRAIN_BATCHES}" \
  "++data.args.wds_val_epoch_batches=${VAL_BATCHES}" \
  "++data.args.wds_prefetch_factor=2" \
  "++data.args.wds_persistent_workers=true"

python - "$RUN_DIR" <<'PY'
import json
from pathlib import Path
import sys

run_dir = Path(sys.argv[1])
dataset = json.loads((run_dir / "dataset_summary.json").read_text(encoding="utf-8"))
normal = json.loads((run_dir / "normal_result.json").read_text(encoding="utf-8"))
optimized = json.loads((run_dir / "optimized_result.json").read_text(encoding="utf-8"))

speedup = normal["elapsed_seconds"] / optimized["elapsed_seconds"] if optimized["elapsed_seconds"] else 0.0
report = {
    "dataset": dataset,
    "normal": normal,
    "optimized": optimized,
    "speedup_wall_clock": speedup,
}
(run_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

lines = [
    "# DSD DDP One-Shard Speed Benchmark",
    "",
    "## Setup",
    f"- Train shard: `{dataset['train_shard']}`",
    f"- Dev shard: `{dataset['dev_shard']}`",
    f"- Train samples: `{dataset['train_samples']}`",
    f"- Dev samples: `{dataset['dev_samples']}`",
    "- GPUs: `0,1`",
    "- Batch size: `14` global",
    "- Num workers: `1`",
    "- Max epochs: `1`",
    "- Augmentation override: `[none]`",
    "- Model compile override: `false`",
    "- DDP one-shard workaround: second train/dev URL is a symlink to the same selected tar",
    f"- Optimized epoch batches: train=`{dataset['train_samples'] // 14}`, dev=`{(dataset['dev_samples'] + 13) // 14}`",
    "",
    "## Results",
    f"- Normal elapsed: `{normal['elapsed_seconds']}s`, rc=`{normal['returncode']}`",
    f"- Optimized elapsed: `{optimized['elapsed_seconds']}s`, rc=`{optimized['returncode']}`",
    f"- Optimized wall-clock speedup: `{speedup:.3f}x`",
    "",
    "## Artifacts",
    f"- Normal log: `{normal['log_path']}`",
    f"- Optimized log: `{optimized['log_path']}`",
    f"- Protocol: `{dataset['protocol_path']}`",
    f"- One-shard WDS dir: `{dataset['wds_dir']}`",
]
(run_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(run_dir / "summary.md")
PY
