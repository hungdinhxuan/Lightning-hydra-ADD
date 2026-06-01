#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

DATA_DIR="${DATA_DIR:-/data/dsd_corpus_pool_13May2026}"
SOURCE_WDS_DIR="${WDS_DATA_DIR:-${PROJECT_ROOT}/data/dsd_corpus_pool_13May2026_wds}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_ROOT}/logs/benchmark_dsd_shard_sweep}"
RUN_NAME="${RUN_NAME:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_ROOT}/${RUN_NAME}"
SHARD_COUNTS="${SHARD_COUNTS:-2,5,10}"

GPUS="${GPUS:-0,1}"
BATCH_SIZE="${BATCH_SIZE:-14}"
NUM_WORKERS="${NUM_WORKERS:-1}"
MAX_EPOCHS="${MAX_EPOCHS:-1}"
SHARD_SIZE_MB="${SHARD_SIZE_MB:-1024}"

mkdir -p "$RUN_DIR"

prepare_case() {
  local case_dir="$1"
  local shard_count="$2"
  python - "$case_dir" "$SOURCE_WDS_DIR" "$shard_count" <<'PY'
import json
import math
import sys
import tarfile
from pathlib import Path

case_dir = Path(sys.argv[1])
source_wds_dir = Path(sys.argv[2]).resolve()
shard_count = int(sys.argv[3])
case_dir.mkdir(parents=True, exist_ok=True)
wds_dir = case_dir / "wds_shards"
wds_dir.mkdir(parents=True, exist_ok=True)

def pick(prefix: str):
    shards = sorted(source_wds_dir.glob(f"{prefix}-*.tar"))[:shard_count]
    if len(shards) < shard_count:
        raise SystemExit(f"Need {shard_count} {prefix} shards, found {len(shards)} in {source_wds_dir}")
    return shards

train_shards = pick("train")
dev_shards = pick("dev")
eval_shards = sorted(source_wds_dir.glob("eval-*.tar"))[:shard_count]
if len(eval_shards) < shard_count:
    eval_shards = dev_shards

def link(src: Path, dst_name: str) -> None:
    dst = wds_dir / dst_name
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src.resolve())

for i, src in enumerate(train_shards):
    link(src, f"train-{i:04d}.tar")
for i, src in enumerate(dev_shards):
    link(src, f"dev-{i:04d}.tar")
for i, src in enumerate(eval_shards):
    link(src, f"eval-{i:04d}.tar")

# If shard_count=1, DDP needs two URLs. Sweep defaults avoid this, but keep safe.
ddp_symlink_duplicate = False
if shard_count == 1:
    link(train_shards[0], "train-0001.tar")
    link(dev_shards[0], "dev-0001.tar")
    link(eval_shards[0], "eval-0001.tar")
    ddp_symlink_duplicate = True

def rows_from_shards(paths, fallback_subset: str):
    rows = []
    for path in paths:
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

train_rows = rows_from_shards(train_shards, "train")
dev_rows = rows_from_shards(dev_shards, "dev")
eval_rows = rows_from_shards(eval_shards, "eval")

protocol_path = case_dir / "protocol.txt"
with protocol_path.open("w", encoding="utf-8") as f:
    for relpath, _subset, label in train_rows:
        f.write(f"{relpath} train {label}\n")
    for relpath, _subset, label in dev_rows:
        f.write(f"{relpath} dev {label}\n")
    for relpath, _subset, label in eval_rows:
        f.write(f"{relpath} eval {label}\n")

batch_size = 14
summary = {
    "shard_count": shard_count,
    "protocol_path": str(protocol_path),
    "wds_dir": str(wds_dir),
    "train_shards": [str(p) for p in train_shards],
    "dev_shards": [str(p) for p in dev_shards],
    "eval_shards": [str(p) for p in eval_shards],
    "train_samples": len(train_rows),
    "dev_samples": len(dev_rows),
    "eval_samples": len(eval_rows),
    "train_batches_at_batch14": max(len(train_rows) // batch_size, 1),
    "dev_batches_at_batch14": max(math.ceil(len(dev_rows) / batch_size), 1),
    "ddp_symlink_duplicate": ddp_symlink_duplicate,
}
(case_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
PY
}

run_case() {
  local case_dir="$1"
  local name="$2"
  shift 2
  local log_path="${case_dir}/${name}.log"
  local metrics_path="${case_dir}/${name}_metrics.jsonl"
  local result_path="${case_dir}/${name}_result.json"
  local start end rc

  echo "Running ${case_dir##*/}/${name}..."
  start="$(date +%s)"
  set +e
  CUDA_VISIBLE_DEVICES="$GPUS" \
  SHARD_SIZE_MB="$SHARD_SIZE_MB" \
  python src/train.py "$@" \
    "++data.data_dir=${DATA_DIR}" \
    "++data.args.data_dir=${DATA_DIR}" \
    "++data.args.protocol_path=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["protocol_path"])' "${case_dir}/dataset_summary.json")" \
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
    "hydra.run.dir=${case_dir}/hydra_${name}" \
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
    echo "${case_dir##*/}/${name} failed. See ${log_path}" >&2
    return "$rc"
  fi
}

IFS=',' read -ra COUNTS <<< "$SHARD_COUNTS"
for count in "${COUNTS[@]}"; do
  case_dir="${RUN_DIR}/shards_${count}"
  prepare_case "$case_dir" "$count"

  train_batches="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["train_batches_at_batch14"])' "${case_dir}/dataset_summary.json")"
  val_batches="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["dev_batches_at_batch14"])' "${case_dir}/dataset_summary.json")"
  one_wds_dir="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["wds_dir"])' "${case_dir}/dataset_summary.json")"

  run_case "$case_dir" "normal" \
    "experiment=xlsr_conformertcm_mdt"

  run_case "$case_dir" "optimized" \
    "experiment=xlsr_conformertcm_mdt_optimized" \
    "++data.args.wds_data_dir=${one_wds_dir}" \
    "++data.args.wds_shard_shuffle=0" \
    "++data.args.wds_sample_shuffle=0" \
    "++data.args.wds_train_epoch_batches=${train_batches}" \
    "++data.args.wds_val_epoch_batches=${val_batches}" \
    "++data.args.wds_prefetch_factor=2" \
    "++data.args.wds_persistent_workers=true"
done

python - "$RUN_DIR" <<'PY'
import json
import re
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
rows = []
for case_dir in sorted(run_dir.glob("shards_*"), key=lambda p: int(p.name.split("_")[1])):
    dataset = json.loads((case_dir / "dataset_summary.json").read_text(encoding="utf-8"))
    normal = json.loads((case_dir / "normal_result.json").read_text(encoding="utf-8"))
    optimized = json.loads((case_dir / "optimized_result.json").read_text(encoding="utf-8"))
    speedup = normal["elapsed_seconds"] / optimized["elapsed_seconds"] if optimized["elapsed_seconds"] else 0.0
    row = {
        "shards": dataset["shard_count"],
        "train_samples": dataset["train_samples"],
        "dev_samples": dataset["dev_samples"],
        "normal_seconds": normal["elapsed_seconds"],
        "optimized_seconds": optimized["elapsed_seconds"],
        "speedup": speedup,
        "normal_log": normal["log_path"],
        "optimized_log": optimized["log_path"],
    }
    rows.append(row)
    (case_dir / "summary.json").write_text(json.dumps({"dataset": dataset, "normal": normal, "optimized": optimized, "speedup_wall_clock": speedup}, indent=2), encoding="utf-8")

(run_dir / "summary.json").write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")

lines = [
    "# DSD DDP Shard Sweep Benchmark",
    "",
    "## Controls",
    "- Runs sequential per shard count: normal then optimized.",
    "- GPUs: `0,1`",
    "- Global batch size: `14`",
    "- Num workers: `1`",
    "- Epochs: `1`",
    "- Augmentation override: `[none]`",
    "- Model compile override: `false`",
    "",
    "## Results",
    "| Train shards | Train samples | Dev samples | Normal | Optimized | Speedup |",
    "|---:|---:|---:|---:|---:|---:|",
]
for row in rows:
    lines.append(
        f"| {row['shards']} | {row['train_samples']} | {row['dev_samples']} | "
        f"{row['normal_seconds']}s | {row['optimized_seconds']}s | {row['speedup']:.3f}x |"
    )

lines.extend([
    "",
    "## Notes",
    "- Optimized WDS epoch batch counts are capped to match the normal protocol sample count.",
    "- For shard counts >=2, no duplicate symlink workaround is needed for 2-GPU DDP.",
    "- Logs and per-case summaries live under each `shards_N/` folder.",
])
(run_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(run_dir / "report.md")
PY
