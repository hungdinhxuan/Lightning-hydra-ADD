#!/bin/bash
# Eval LoRA ultra-short checkpoint on test protocol (full + fixed-length matrix).
#
# Usage:
#   export LORA_CKPT=/path/to/best.ckpt
#   # or LoRA-only adapter dir saved by peft `save_pretrained`
#   export LORA_CKPT=/path/to/epoch_021
#   bash scripts/cnsl/May302026/eval_lora_ultrashort_vadshort.sh -d 0

set -euo pipefail

while getopts "d:" opt; do
  case $opt in
    d) CUDA_DEVICE="$OPTARG";;
    \?) echo "Invalid option -$OPTARG" >&2; exit 1;;
  esac
done

CUDA_DEVICE=${CUDA_DEVICE:-"0"}
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/data/add_processed_vad_short_eval10k_per_dataset_train80k_dev10k_29May2026}"
WORK_DIR="${WORK_DIR:-data/protocol_test_eval_benchmark}"
RESULTS_ROOT="${RESULTS_ROOT:-logs/results/lora_ultrashort_mdt_protocol_test_eval}"
LORA_CKPT="${LORA_CKPT:?Set LORA_CKPT to fine-tuned checkpoint .pt or .ckpt}"
BASE_CKPT="${BASE_CKPT:-/NAS1_pretrained_lab/May_21_2026_TCM_MDT.pt}"
CONFIG="${CONFIG:-xlsr_conformertcm_mdt_lora_ultrashort_vadshort}"
PREFIX="${PREFIX:-lora_ultrashort_mdt}"
MAX_PER_SOURCE="${MAX_PER_SOURCE:-}"

MODEL_PATH="$BASE_CKPT"
ADAPTER_ARGS=()
# Benchmark folders are raw/symlink audio directories, not WDS shards, so eval
# must use the raw MDT datamodule even though training uses MDT_optimized.
EXTRA_OVERRIDES=(
  "data=MDT_default"
  "++data.args.padding_type=repeat"
  "++data.args.view_lengths_samples=[8000,16000,24000,32000]"
)
if [[ -d "$LORA_CKPT" && -f "$LORA_CKPT/adapter_config.json" ]]; then
  ADAPTER_ARGS=(--adapter-paths "$LORA_CKPT")
elif [[ "$LORA_CKPT" == *.ckpt ]]; then
  EXTRA_OVERRIDES+=("ckpt_path=$LORA_CKPT")
else
  ADAPTER_ARGS=(--adapter-paths "$LORA_CKPT")
fi

PREPARE_LIMIT_ARGS=()
if [[ -n "$MAX_PER_SOURCE" ]]; then
  PREPARE_LIMIT_ARGS+=(--max-per-source "$MAX_PER_SOURCE")
fi

python scripts/benchmark_py/short_duration_eval.py prepare \
  --source-root "$DATA_ROOT" \
  --work-dir "$WORK_DIR" \
  --subsets test \
  --tracks fixed_length_test observed_short_full vad_short_full \
  --comment-prefix "$PREFIX" \
  "${PREPARE_LIMIT_ARGS[@]}"

run_full() {
  python scripts/benchmark_py/short_duration_eval.py run-single \
    --source-root "$DATA_ROOT" \
    --work-dir "$WORK_DIR" \
    --results-dir "$RESULTS_ROOT" \
    --subsets test \
    --tracks observed_short_full vad_short_full \
    --run-group full_length \
    --full-utterance \
    --gpus "$CUDA_DEVICE" \
    --batch-size 128 \
    --config "$CONFIG" \
    --model-path "$MODEL_PATH" \
    "${ADAPTER_ARGS[@]}" \
    --comment-prefix "$PREFIX" \
    "${EXTRA_OVERRIDES[@]}"
}

run_fixed() {
  local random_flag=$1
  local fixed_prefix="$PREFIX"
  if [[ "$random_flag" == "--random-start" ]]; then
    fixed_prefix="${PREFIX}_random"
  else
    fixed_prefix="${PREFIX}_norandom"
  fi
  python scripts/benchmark_py/short_duration_eval.py run-single \
    --source-root "$DATA_ROOT" \
    --work-dir "$WORK_DIR" \
    --results-dir "$RESULTS_ROOT" \
    --subsets test \
    --tracks fixed_length_test \
    --run-group fixed_length \
    --durations 0.5 1.0 1.5 2.0 \
    $random_flag \
    --gpus "$CUDA_DEVICE" \
    --batch-size 128 \
    --config "$CONFIG" \
    --model-path "$MODEL_PATH" \
    "${ADAPTER_ARGS[@]}" \
    --comment-prefix "$fixed_prefix" \
    "${EXTRA_OVERRIDES[@]}"
}

run_full
run_fixed --no-random-start
run_fixed --random-start

echo "Results under: $RESULTS_ROOT"
