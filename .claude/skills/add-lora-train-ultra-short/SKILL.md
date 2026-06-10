---
name: add-lora-train-short
description: Train an MDT LoRA ultrashort audio-deepfake-detection model with the DDP + WebDataset fix. Use when the user wants to train/fine-tune a LoRA ultrashort (vadshort) model, fix DDP/WDS epoch-sizing hangs, resume the xlsr_conformertcm_mdt_lora_ultrashort_vadshort experiment, or select a best checkpoint from a completed run. Triggers: "train lora ultrashort", "ddp wds hang", "wds_auto_epoch_batches", "fine-tune MDT lora".
---

# ADD LoRA Ultrashort Training (DDP + WDS)

Train an MDT LoRA ultrashort model on short-duration VAD audio, fine-tuning from a frozen MDT base checkpoint with DDP across 2 GPUs.

## When to use
- Launch / reproduce the `xlsr_conformertcm_mdt_lora_ultrashort_vadshort` run.
- Fix DDP training that hangs or has unstable epoch sizing with a WebDataset loader.
- Pick the best checkpoint after a run finishes.

## DDP / WDS fix (root cause + fix)
WebDataset loader length did not align with DDP, causing instability/hangs around epoch sizing.

Fix lives in `src/data/normal_datamodule_optimized.py` (~lines 88, 124): supports `wds_auto_epoch_batches`. When enabled it computes epoch batches and wraps the loader:
```python
loader.with_epoch(n_batches)
```
Enable via config:
```yaml
data:
  args:
    wds_auto_epoch_batches: true
```
Config flag observed at `configs/experiment/xlsr_conformertcm_mdt_lora_ultrashort_vadshort.yaml:70`.

## Key inputs
- Dataset root: `data/add_processed_vad_short_eval10k_per_dataset_train80k_dev10k_29May2026`
- Protocol: `<root>/subsets/train_dev_90k_wds.protocol.txt`
- WDS shards: `<root>/wds_data`
- Base MDT ckpt: `/NAS1_pretrained_lab/May_21_2026_TCM_MDT.pt`
- Experiment cfg: `configs/experiment/xlsr_conformertcm_mdt_lora_ultrashort_vadshort.yaml`
- View lengths (samples): 0.5s=8000, 1.0s=16000, 1.5s=24000, 2.0s=32000

## Reproducible command
```bash
cd /home/hungdx/code/Lightning-hydra-ADD

export DATA_ROOT="$PWD/data/add_processed_vad_short_eval10k_per_dataset_train80k_dev10k_29May2026"
export BASE_CKPT="/NAS1_pretrained_lab/May_21_2026_TCM_MDT.pt"
export CUDA_VISIBLE_DEVICES=0,1

.venv/bin/python src/train.py \
  experiment=xlsr_conformertcm_mdt_lora_ultrashort_vadshort \
  ++data.data_dir="$DATA_ROOT" \
  ++data.args.protocol_path="$DATA_ROOT/subsets/train_dev_90k_wds.protocol.txt" \
  ++data.args.wds_data_dir="$DATA_ROOT/wds_data" \
  ++model.is_base_model_path_ln=False \
  ++model.base_model_path="$BASE_CKPT" \
  ++trainer.devices=2 \
  ++trainer.max_epochs=100 \
  ++trainer.num_sanity_val_steps=0 \
  ++test=false \
  logger=csv \
  ++data.num_workers=2 \
  ++data.pin_memory=false \
  ++data.args.wds_prefetch_factor=8 \
  ++data.args.wds_persistent_workers=true \
  ++trainer.strategy=ddp
```

## Checkpoint selection
- Outputs land in `logs/train/runs/<timestamp>/checkpoints/`.
- Early stopping may end the run well before `max_epochs` (prior run: requested 100, stopped epoch 25).
- Pick the **best `val/acc_best` epoch ckpt, NOT `last.ckpt`**. Prior run selected `epoch_021.ckpt`.
- Per-view val accuracy keys to compare: `val/view_{8000,16000,24000,32000}_acc_best`.

## Verify a run
```bash
ps -eo pid,cmd | rg 'src/train.py' | rg -v 'rg ' || true
ls logs/train/runs/<timestamp>/checkpoints/
```

## Gotchas
- Early stopping ending before `max_epochs` is expected, not a failure.
- Use `epoch_NNN.ckpt`, not `last.ckpt`, for downstream inference.
- Warm-up benchmark before full training when data loading changed.

Next stage after a good ckpt: skill `add-ultra-short-eval`.
