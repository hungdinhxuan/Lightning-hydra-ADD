# LoRA Ultra-Short MDT Protocol Inference Summary

This note records how the inference and reporting were done for:

```text
logs/results/lora_ultrashort_mdt_protocol_test_eval_full_100epoch
```

The important distinction is the same as the baseline run:

- Full-length evaluation uses the processed short audio directly:
  `observed_short_full` and `vad_short_full`.
- Fixed-length evaluation does not use processed short audio as the waveform
  source. It uses the original source utterance for both observed-short and
  VAD-short samples, then applies fixed crop length at inference time.

This run evaluates the LoRA fine-tuned checkpoint from the full DDP training
job launched with `max_epochs=100`. Training stopped cleanly by early stopping
at epoch 25; the selected best checkpoint is epoch 21.

## Inputs

Dataset root:

```text
data/add_processed_vad_short_eval10k_per_dataset_train80k_dev10k_29May2026
```

Prepared benchmark work directory:

```text
data/protocol_test_eval_benchmark_lora_full
```

Prepared tracks:

```text
fixed_length_test     80,000 rows
observed_short_full    5,255 rows
vad_short_full        74,745 rows
```

Model base checkpoint:

```text
/NAS1_pretrained_lab/May_21_2026_TCM_MDT.pt
```

LoRA Lightning checkpoint:

```text
logs/train/runs/2026-05-30_19-32-38/checkpoints/epoch_021.ckpt
```

Config:

```text
xlsr_conformertcm_mdt_lora_ultrashort_vadshort
```

Score file format:

```text
<filename> <spoof_score> <bonafide_score>
```

For all metrics, `bonafide_score` is the primary score. Higher means more
bonafide.

## Benchmark Preparation

The wrapper script is:

```text
scripts/cnsl/May302026/eval_lora_ultrashort_vadshort.sh
```

It calls:

```text
scripts/benchmark_py/short_duration_eval.py
```

Preparation command used by the wrapper:

```bash
python scripts/benchmark_py/short_duration_eval.py prepare \
  --source-root data/add_processed_vad_short_eval10k_per_dataset_train80k_dev10k_29May2026 \
  --work-dir data/protocol_test_eval_benchmark_lora_full \
  --subsets test \
  --tracks fixed_length_test observed_short_full vad_short_full \
  --comment-prefix lora_ultrashort_mdt_ddp_full100_ep21
```

What this does:

- `observed_short_full`: keeps only processed observed-short samples.
- `vad_short_full`: keeps only processed VAD-short samples.
- `fixed_length_test`: keeps all protocol samples, but maps each sample to
  `source_abs_path` from the manifest.
- Fixed-length eval path is rewritten as:

```text
audio/original_utterance/<source_stem>__source.wav
```

The key point: fixed-length crops are sampled from the original utterance, not
from the already processed short segment.

## Inference Command

The full matrix was run with one GPU:

```bash
export REPO_ROOT="/home/hungdx/code/Lightning-hydra-ADD"
export BASE_CKPT="/NAS1_pretrained_lab/May_21_2026_TCM_MDT.pt"
export LORA_CKPT="$REPO_ROOT/logs/train/runs/2026-05-30_19-32-38/checkpoints/epoch_021.ckpt"
export WORK_DIR="data/protocol_test_eval_benchmark_lora_full"
export RESULTS_ROOT="logs/results/lora_ultrashort_mdt_protocol_test_eval_full_100epoch"
export PREFIX="lora_ultrashort_mdt_ddp_full100_ep21"

CUDA_VISIBLE_DEVICES=0 \
  bash scripts/cnsl/May302026/eval_lora_ultrashort_vadshort.sh -d 0 \
  2>&1 | tee logs/results/lora_ultrashort_mdt_protocol_test_eval_full_100epoch.log
```

The wrapper uses the raw MDT datamodule for evaluation because the benchmark
folders are raw audio/symlink folders, not WDS shards:

```text
data=MDT_default
++data.args.padding_type=repeat
++data.args.view_lengths_samples=[8000,16000,24000,32000]
```

Because `LORA_CKPT` is a Lightning `.ckpt`, inference loads:

```text
++model.base_model_path=/NAS1_pretrained_lab/May_21_2026_TCM_MDT.pt
ckpt_path=logs/train/runs/2026-05-30_19-32-38/checkpoints/epoch_021.ckpt
```

## Full-Length Inference

Full-length evaluation runs on the processed short audio. No fixed crop length
is used.

Benchmark settings:

```text
run_group = full_length
full_utterance = true
trim_length = 0
random_start = false
data.args.no_pad = true
```

Expected full-length score files:

```text
logs/results/lora_ultrashort_mdt_protocol_test_eval_full_100epoch/lora_ultrashort_mdt_ddp_full100_ep21_full/observed_short_full_xlsr_conformertcm_mdt_lora_ultrashort_vadshort_lora_ultrashort_mdt_ddp_full100_ep21_full.txt
logs/results/lora_ultrashort_mdt_protocol_test_eval_full_100epoch/lora_ultrashort_mdt_ddp_full100_ep21_full/vad_short_full_xlsr_conformertcm_mdt_lora_ultrashort_vadshort_lora_ultrashort_mdt_ddp_full100_ep21_full.txt
```

Full-length score count:

```text
observed_short_full     5,255
vad_short_full         74,745
merged full-length     80,000
```

Full-length metric:

| Run | EER | ROC-AUC | Accuracy | Balanced Accuracy |
| --- | ---: | ---: | ---: | ---: |
| processed short full utterance | 2.309487 | 99.753967 | 97.236250 | 97.688750 |

## Fixed-Length Inference

Fixed-length evaluation uses the original source utterance for all 80,000
samples. The model receives fixed-duration crops at inference time.

Durations:

```text
0.5s -> trim_length 8000
1.0s -> trim_length 16000
1.5s -> trim_length 24000
2.0s -> trim_length 32000
```

Modes:

```text
random_start=false
random_start=true
```

Expected fixed-length matrix:

```text
1 fixed_length_test track x 2 random_start modes x 4 durations = 8 score files
8 score files x 80,000 rows = 640,000 fixed-length inference rows
```

Combined with the 80,000 full-length rows, this is:

```text
720,000 scored inference rows
```

Filename pattern:

```text
fixed_length_test_xlsr_conformertcm_mdt_lora_ultrashort_vadshort_lora_ultrashort_mdt_ddp_full100_ep21_{norandom|random}_{duration}.txt
```

Fixed-length metrics:

| random_start | duration | trim_length | EER | ROC-AUC | Accuracy | Balanced Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| false | 0.5s | 8000 | 9.210769 | 90.092349 | 84.050000 | 89.232500 |
| false | 1.0s | 16000 | 7.503333 | 98.045664 | 92.868750 | 92.498750 |
| false | 1.5s | 24000 | 3.671282 | 99.423102 | 96.053750 | 96.330000 |
| false | 2.0s | 32000 | 2.380000 | 99.739917 | 97.478750 | 97.620000 |
| true | 0.5s | 8000 | 11.608462 | 95.925686 | 89.183750 | 88.386250 |
| true | 1.0s | 16000 | 4.635897 | 99.129558 | 94.910000 | 95.362500 |
| true | 1.5s | 24000 | 2.847179 | 99.621927 | 96.818750 | 97.152500 |
| true | 2.0s | 32000 | 2.045641 | 99.790471 | 97.693750 | 97.955000 |

Best fixed-length run:

```text
random_start=true, duration=2.0s, EER=2.045641, ROC-AUC=99.790471, Accuracy=97.693750
```

Best deterministic fixed-length run:

```text
random_start=false, duration=2.0s, EER=2.380000, ROC-AUC=99.739917, Accuracy=97.478750
```

## Result Files

Summary files:

```text
logs/results/lora_ultrashort_mdt_protocol_test_eval_full_100epoch/*/summary_results.txt
logs/results/lora_ultrashort_mdt_protocol_test_eval_full_100epoch/*/summary_results_detailed.txt
logs/results/lora_ultrashort_mdt_protocol_test_eval_full_100epoch/*/summary_results_details.jsonl
```

Inference log:

```text
logs/results/lora_ultrashort_mdt_protocol_test_eval_full_100epoch.log
```

Hydra run folders are under each run directory:

```text
logs/results/lora_ultrashort_mdt_protocol_test_eval_full_100epoch/*/hydra_runs/
```

## Notes

- `Lora Path: None` in `summary_results.txt` is expected for this run because
  the LoRA weights are restored through Lightning `ckpt_path`, while
  `Base_model_path` points to the original MDT checkpoint.
- The full training command used `max_epochs=100`, but the run stopped at epoch
  25 due to configured early stopping. The selected checkpoint for inference is
  `epoch_021.ckpt`.
