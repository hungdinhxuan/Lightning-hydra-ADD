# Runbook: Fine-tune MDT LoRA ultra-short (May 30)

## Mục tiêu

| Hạng mục | Giá trị |
| --- | --- |
| Baseline checkpoint | `/NAS1_pretrained_lab/May_21_2026_TCM_MDT.pt` |
| Dataset | `data/add_processed_vad_short_eval10k_per_dataset_train80k_dev10k_29May2026` |
| Training views | 0.5s / 1.0s / 1.5s / 2.0s → samples `[8000, 16000, 24000, 32000]` |
| Training | **Ưu tiên multi-GPU DDP** (LoRA + MDT) |
| Eval | `plan/baseline_mdt_protocol_inference_summary.md` |
| Báo cáo | `reports/US-MDT-LoRA-from-baseline/` |

## Đã có sẵn trong repo

- `configs/experiment/xlsr_conformertcm_mdt_lora_ultrashort_vadshort.yaml`
- `src/data/components/collate_fn.py` — `view_lengths_samples`
- `src/data/dataset_optimized.py` — `empty_check=False`
- `scripts/cnsl/May302026/train_lora_ultrashort_vadshort.sh`, `eval_lora_ultrashort_vadshort.sh`
- `scripts/run_training_gate.py` — tự gắn `ddp_find_unused_parameters_true`

**Đã chạy trước đó (có thể bỏ qua):** `train_dev_90k_wds.protocol.txt` (90k dòng), WDS 6 shards tại `data/.../wds_data/`, warmup pass tại `logs/optimized_configs/ultrashort_vadshort/`.

---

## Bước 1 — Protocol gộp (WDS)

```bash
export REPO_ROOT="/home/hungdx/code/Lightning-hydra-ADD"
export DATA_ROOT="$REPO_ROOT/data/add_processed_vad_short_eval10k_per_dataset_train80k_dev10k_29May2026"

cat "$DATA_ROOT/subsets/train_80k_balanced.protocol.txt" \
    "$DATA_ROOT/subsets/dev_10k_balanced.protocol.txt" \
  > "$DATA_ROOT/subsets/train_dev_90k_wds.protocol.txt"
```

---

## Bước 2 — WebDataset

```bash
cd "$REPO_ROOT"
export WDS_DATA_DIR="$DATA_ROOT/wds_data"

.venv/bin/python scripts/preflight_and_prepare.py \
  experiment=xlsr_conformertcm_mdt_lora_ultrashort_vadshort \
  ++data.data_dir="$DATA_ROOT" \
  ++data.args.protocol_path="$DATA_ROOT/subsets/train_dev_90k_wds.protocol.txt" \
  ++data.args.wds_data_dir="$WDS_DATA_DIR" \
  --output_dir logs/optimized_configs/ultrashort_vadshort
```

---

## Bước 3 — Training (ưu tiên multi-GPU DDP)

- Dùng `scripts/run_training_gate.py` (preflight → warmup → train).
- Gate ép `++trainer.strategy=ddp_find_unused_parameters_true`.
- Set `++trainer.devices=N` và `CUDA_VISIBLE_DEVICES=0,1,...`.
- Env: `XLSR_PRETRAINED_MODEL_PATH`, `AUG_PATH`, `NOISE_PATH`, `RIR_PATH` (từ `.env`).

### 2 GPU

```bash
export BASE_CKPT="/NAS1_pretrained_lab/May_21_2026_TCM_MDT.pt"
export CUDA_VISIBLE_DEVICES=0,1
export NUM_GPUS=2

.venv/bin/python scripts/run_training_gate.py \
  experiment=xlsr_conformertcm_mdt_lora_ultrashort_vadshort \
  ++data.data_dir="$DATA_ROOT" \
  ++data.args.protocol_path="$DATA_ROOT/subsets/train_dev_90k_wds.protocol.txt" \
  ++data.args.wds_data_dir="$WDS_DATA_DIR" \
  ++model.is_base_model_path_ln=False \
  ++model.base_model_path="$BASE_CKPT" \
  ++trainer.devices=$NUM_GPUS \
  logger=csv \
  --optimized-config-dir logs/optimized_configs/ultrashort_vadshort
```

### 4 GPU

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

.venv/bin/python scripts/run_training_gate.py \
  experiment=xlsr_conformertcm_mdt_lora_ultrashort_vadshort \
  ++data.data_dir="$DATA_ROOT" \
  ++data.args.protocol_path="$DATA_ROOT/subsets/train_dev_90k_wds.protocol.txt" \
  ++data.args.wds_data_dir="$WDS_DATA_DIR" \
  ++model.is_base_model_path_ln=False \
  ++model.base_model_path="$BASE_CKPT" \
  ++trainer.devices=4 \
  logger=csv \
  --optimized-config-dir logs/optimized_configs/ultrashort_vadshort
```

Checkpoint: `logs/train/runs/<timestamp>-ultrashort_vadshort/checkpoints/` → `export LORA_CKPT=...`

---

## Bước 4 — Eval (1 GPU)

```bash
export LORA_CKPT="/path/to/best.ckpt"
bash scripts/cnsl/May302026/eval_lora_ultrashort_vadshort.sh -d 0
```

Chi tiết: `plan/baseline_mdt_protocol_inference_summary.md`.

---

## Bước 5 — Báo cáo

Tạo `reports/US-MDT-LoRA-from-baseline/` với `README.md`, `training.md`, `metrics_summary.md`, `commands.sh`.

---

## Debug nhanh

| Triệu chứng | Xử lý |
| --- | --- |
| Fewer shards than workers | `empty_check=False` đã có; giảm `num_workers` |
| Warmup CSV lỗi `fieldnames: error` | Cập nhật `scripts/warmup_benchmark.py` |
| DDP unused params | Giữ `ddp_find_unused_parameters_true` từ gate |

## Checklist

1. [ ] Protocol + WDS (nếu chưa)
2. [ ] Gate + DDP training
3. [ ] Eval matrix
4. [ ] Report vs baseline
