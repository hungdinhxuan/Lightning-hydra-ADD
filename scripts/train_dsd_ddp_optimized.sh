#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

DATA_DIR="${DATA_DIR:-/data/dsd_corpus_pool_13May2026}"
PROTOCOL_PATH="${PROTOCOL_PATH:-/data/dsd_corpus_pool_13May2026/15_May_full.txt}"
WDS_DATA_DIR="${WDS_DATA_DIR:-${PROJECT_ROOT}/data/dsd_corpus_pool_13May2026_wds}"
OUTPUT_DIR="${OUTPUT_DIR:-logs/optimized_configs/dsd_corpus_pool_13May2026}"

EXPERIMENT="${EXPERIMENT:-xlsr_conformertcm_mdt_optimized}"
MAX_EPOCHS="${MAX_EPOCHS:-1}"
METRICS_PATH="${METRICS_PATH:-/tmp/dsd_ddp_metrics.json}"

NUM_WORKERS="${NUM_WORKERS:-8}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-8}"
PIN_MEMORY="${PIN_MEMORY:-true}"
PERSISTENT_WORKERS="${PERSISTENT_WORKERS:-true}"

# Large datasets should use large shards. If the shard-count guard below fails,
# rerun with a smaller value, for example: SHARD_SIZE_MB=512 ./scripts/train_dsd_ddp_optimized.sh
SHARD_SIZE_MB="${SHARD_SIZE_MB:-1024}"
FORCE_CONVERT="${FORCE_CONVERT:-false}"

if [[ ! -d "$DATA_DIR" ]]; then
  echo "Missing DATA_DIR: $DATA_DIR" >&2
  exit 2
fi

if [[ ! -f "$PROTOCOL_PATH" ]]; then
  echo "Missing PROTOCOL_PATH: $PROTOCOL_PATH" >&2
  exit 2
fi

PREFLIGHT_CMD=(
  python scripts/preflight_and_prepare.py
  "experiment=${EXPERIMENT}"
  "++data.data_dir=${DATA_DIR}"
  "++data.args.protocol_path=${PROTOCOL_PATH}"
  "++data.args.wds_data_dir=${WDS_DATA_DIR}"
  --output_dir "$OUTPUT_DIR"
  --shard_size_mb "$SHARD_SIZE_MB"
)

if [[ "$FORCE_CONVERT" == "true" ]]; then
  PREFLIGHT_CMD+=(--force_convert)
fi

"${PREFLIGHT_CMD[@]}"

TRAIN_SHARDS="$(find "$WDS_DATA_DIR" -maxdepth 1 -name 'train-*.tar' | wc -l | tr -d ' ')"
if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_COUNT="$(nvidia-smi -L | wc -l | tr -d ' ')"
else
  GPU_COUNT=1
fi

MIN_TRAIN_SHARDS=$((GPU_COUNT * NUM_WORKERS))
if (( TRAIN_SHARDS < MIN_TRAIN_SHARDS )); then
  echo "Too few train shards for DDP workers: train_shards=${TRAIN_SHARDS}, required>=${MIN_TRAIN_SHARDS}" >&2
  echo "Rerun with smaller SHARD_SIZE_MB and FORCE_CONVERT=true, or lower NUM_WORKERS." >&2
  echo "Example: SHARD_SIZE_MB=512 FORCE_CONVERT=true $0" >&2
  exit 3
fi

python src/train.py \
  "experiment=${EXPERIMENT}" \
  ++model_averaging=true \
  "++data.data_dir=${DATA_DIR}" \
  "++data.args.protocol_path=${PROTOCOL_PATH}" \
  "++data.args.wds_data_dir=${WDS_DATA_DIR}" \
  "++data.num_workers=${NUM_WORKERS}" \
  "++data.pin_memory=${PIN_MEMORY}" \
  "++data.args.wds_prefetch_factor=${PREFETCH_FACTOR}" \
  "++data.args.wds_persistent_workers=${PERSISTENT_WORKERS}" \
  ++trainer.strategy=ddp_find_unused_parameters_true \
  "++trainer.max_epochs=${MAX_EPOCHS}" \
  logger=json \
  ++json_logging.enabled=true \
  "++json_logging.log_path=${METRICS_PATH}" \
  "$@"
