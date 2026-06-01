#!/bin/bash
# LoRA fine-tune MDT on VAD-short 80k/10k with ultra-short multi-view [0.5, 1.0, 1.5, 2.0]s.
#
# Usage:
#   export XLSR_PRETRAINED_MODEL_PATH=/path/to/xlsr_ssl.pt
#   export AUG_PATH=... NOISE_PATH=... RIR_PATH=...
#   bash scripts/cnsl/May302026/train_lora_ultrashort_vadshort.sh
#   bash scripts/cnsl/May302026/train_lora_ultrashort_vadshort.sh -d 0

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
PROTOCOL="$DATA_ROOT/subsets/train_dev_90k_wds.protocol.txt"
WDS_DIR="${WDS_DATA_DIR:-$DATA_ROOT/wds_data}"
BASE_CKPT="${BASE_CKPT:-/NAS1_pretrained_lab/May_21_2026_TCM_MDT.pt}"
CONFIG="${CONFIG:-xlsr_conformertcm_mdt_lora_ultrashort_vadshort}"

export WDS_DATA_DIR="$WDS_DIR"
# export WDS_SHM_DIR="/dev/shm/add_vad_short_wds"

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE OMP_NUM_THREADS=8 python scripts/run_training_gate.py \
  experiment="$CONFIG" \
  ++data.data_dir="$DATA_ROOT" \
  ++data.args.protocol_path="$PROTOCOL" \
  ++data.args.wds_data_dir="$WDS_DIR" \
  ++model.is_base_model_path_ln=False \
  ++model.base_model_path="$BASE_CKPT" \
  logger=csv
