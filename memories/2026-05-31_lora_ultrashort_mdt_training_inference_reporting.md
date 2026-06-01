# Memory: LoRA Ultrashort MDT Training, Inference, Reporting

Date exported: 2026-05-31

Repository:

```text
/home/hungdx/code/Lightning-hydra-ADD
```

This memory records the reproducible state and actions for the MDT LoRA ultrashort run: DDP training fix, full 100-epoch training request, full 720k inference matrix, LoRA report generation, missing metric/report fixes, multi-window true-duration-bin fix, and final comparison report versus baseline.

## Objective

Train and evaluate an MDT LoRA ultrashort model against the existing MDT baseline using the same short-duration protocol style as:

```text
plan/baseline_mdt_protocol_inference_summary.md
```

The user requested:

- Fix DDP training.
- Run full `max_epochs=100` training.
- Run full 720k evaluation matrix.
- Generate inference summary and visualization report.
- Add missing `threshold_free_class_accuracy_summary.csv`.
- Fix multi-window true-duration-bin row mismatch.
- Generate a comparison report between LoRA and baseline, prioritizing:
  - `threshold_free_class_accuracy_summary.csv`
  - `multi_window_aggregation_by_true_duration_bin_metrics.csv`

## Key Inputs

Dataset root:

```text
data/add_processed_vad_short_eval10k_per_dataset_train80k_dev10k_29May2026
```

Train/dev WDS protocol:

```text
data/add_processed_vad_short_eval10k_per_dataset_train80k_dev10k_29May2026/subsets/train_dev_90k_wds.protocol.txt
```

WebDataset shards:

```text
data/add_processed_vad_short_eval10k_per_dataset_train80k_dev10k_29May2026/wds_data
```

Baseline MDT checkpoint:

```text
/NAS1_pretrained_lab/May_21_2026_TCM_MDT.pt
```

LoRA experiment config:

```text
configs/experiment/xlsr_conformertcm_mdt_lora_ultrashort_vadshort.yaml
```

Training view lengths:

```text
0.5s ->  8000 samples
1.0s -> 16000 samples
1.5s -> 24000 samples
2.0s -> 32000 samples
```

## DDP/WDS Training Fix

Problem seen during DDP training: the WebDataset loader length/epoch behavior did not align cleanly with DDP, causing training instability/hangs around epoch sizing.

Fix applied in:

```text
src/data/normal_datamodule_optimized.py
configs/experiment/xlsr_conformertcm_mdt_lora_ultrashort_vadshort.yaml
```

Important implementation detail:

```text
src/data/normal_datamodule_optimized.py
```

The datamodule now supports `wds_auto_epoch_batches`; when enabled it computes epoch batches and wraps the WebDataset loader with:

```python
loader.with_epoch(n_batches)
```

Relevant config flag:

```yaml
data:
  args:
    wds_auto_epoch_batches: true
```

Exact locations observed after the run:

```text
configs/experiment/xlsr_conformertcm_mdt_lora_ultrashort_vadshort.yaml:70
src/data/normal_datamodule_optimized.py:88
src/data/normal_datamodule_optimized.py:124
```

## Training Run

Training output directory:

```text
logs/train/runs/2026-05-30_19-32-38
```

Hydra overrides from the actual run:

```yaml
- experiment=xlsr_conformertcm_mdt_lora_ultrashort_vadshort
- ++data.data_dir=/home/hungdx/code/Lightning-hydra-ADD/data/add_processed_vad_short_eval10k_per_dataset_train80k_dev10k_29May2026
- ++data.args.protocol_path=/home/hungdx/code/Lightning-hydra-ADD/data/add_processed_vad_short_eval10k_per_dataset_train80k_dev10k_29May2026/subsets/train_dev_90k_wds.protocol.txt
- ++data.args.wds_data_dir=/home/hungdx/code/Lightning-hydra-ADD/data/add_processed_vad_short_eval10k_per_dataset_train80k_dev10k_29May2026/wds_data
- ++model.is_base_model_path_ln=False
- ++model.base_model_path=/NAS1_pretrained_lab/May_21_2026_TCM_MDT.pt
- ++trainer.devices=2
- ++trainer.max_epochs=100
- ++trainer.num_sanity_val_steps=0
- ++test=false
- logger=csv
- ++data.num_workers=2
- ++data.pin_memory=false
- ++data.args.wds_data_dir=data/add_processed_vad_short_eval10k_per_dataset_train80k_dev10k_29May2026/wds_data
- ++data.args.wds_prefetch_factor=8
- ++data.args.wds_persistent_workers=true
- ++trainer.strategy=ddp
```

Equivalent reproducible training command:

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

Training started:

```text
2026-05-30 19:32:52
```

Training completed:

```text
2026-05-31 04:07:49
```

The user requested `max_epochs=100`; the run stopped by early stopping after epoch 25. The selected/best checkpoint used for inference was:

```text
logs/train/runs/2026-05-30_19-32-38/checkpoints/epoch_021.ckpt
```

Available checkpoints:

```text
epoch_005.ckpt
epoch_015.ckpt
epoch_016.ckpt
epoch_021.ckpt
epoch_022.ckpt
last.ckpt
```

Training metrics around the selected checkpoint:

```text
epoch 21
step 54999
val/acc 0.978850
val/acc_best 0.978850
val/loss 0.274505
val/view_8000_acc_best 0.958036
val/view_16000_acc_best 0.981164
val/view_24000_acc_best 0.985664
val/view_32000_acc_best 0.986891
```

## Inference Preparation

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

Wrapper:

```text
scripts/cnsl/May302026/eval_lora_ultrashort_vadshort.sh
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

Important protocol distinction:

- Full-length evaluation uses processed short audio directly:
  - `observed_short_full`
  - `vad_short_full`
- Fixed-length evaluation maps all samples to original source utterances, then applies inference crop length:
  - `audio/original_utterance/<source_stem>__source.wav`

Score file format:

```text
<filename> <spoof_score> <bonafide_score>
```

Primary score for metrics:

```text
bonafide_score
```

Higher score means more bonafide.

## Full 720k Inference Matrix

Main inference output:

```text
logs/results/lora_ultrashort_mdt_protocol_test_eval_full_100epoch
```

Command:

```bash
cd /home/hungdx/code/Lightning-hydra-ADD

export REPO_ROOT="$PWD"
export BASE_CKPT="/NAS1_pretrained_lab/May_21_2026_TCM_MDT.pt"
export LORA_CKPT="$REPO_ROOT/logs/train/runs/2026-05-30_19-32-38/checkpoints/epoch_021.ckpt"
export WORK_DIR="data/protocol_test_eval_benchmark_lora_full"
export RESULTS_ROOT="logs/results/lora_ultrashort_mdt_protocol_test_eval_full_100epoch"
export PREFIX="lora_ultrashort_mdt_ddp_full100_ep21"

CUDA_VISIBLE_DEVICES=0 \
  bash scripts/cnsl/May302026/eval_lora_ultrashort_vadshort.sh -d 0 \
  2>&1 | tee logs/results/lora_ultrashort_mdt_protocol_test_eval_full_100epoch.log
```

The wrapper used raw-audio eval, not WDS:

```text
data=MDT_default
++data.args.padding_type=repeat
++data.args.view_lengths_samples=[8000,16000,24000,32000]
```

Lightning checkpoint load logic:

```text
++model.base_model_path=/NAS1_pretrained_lab/May_21_2026_TCM_MDT.pt
ckpt_path=logs/train/runs/2026-05-30_19-32-38/checkpoints/epoch_021.ckpt
```

Full matrix composition:

```text
full-length:
  observed_short_full  5,255 rows
  vad_short_full      74,745 rows
  total               80,000 rows

fixed-length:
  1 track x 2 random_start modes x 4 durations x 80,000 rows
  = 640,000 rows

total scored rows:
  720,000
```

Fixed-length matrix:

```text
random_start=false: 0.5s, 1.0s, 1.5s, 2.0s
random_start=true:  0.5s, 1.0s, 1.5s, 2.0s
```

Score filename pattern:

```text
fixed_length_test_xlsr_conformertcm_mdt_lora_ultrashort_vadshort_lora_ultrashort_mdt_ddp_full100_ep21_{norandom|random}_{duration}.txt
```

Full-length score files:

```text
logs/results/lora_ultrashort_mdt_protocol_test_eval_full_100epoch/lora_ultrashort_mdt_ddp_full100_ep21_full/observed_short_full_xlsr_conformertcm_mdt_lora_ultrashort_vadshort_lora_ultrashort_mdt_ddp_full100_ep21_full.txt
logs/results/lora_ultrashort_mdt_protocol_test_eval_full_100epoch/lora_ultrashort_mdt_ddp_full100_ep21_full/vad_short_full_xlsr_conformertcm_mdt_lora_ultrashort_vadshort_lora_ultrashort_mdt_ddp_full100_ep21_full.txt
```

## LoRA Inference Summary Report

Procedural report created:

```text
plan/lora_ultrashort_mdt_protocol_inference_summary.md
```

This report mirrors:

```text
plan/baseline_mdt_protocol_inference_summary.md
```

It documents:

- Inputs.
- Benchmark preparation.
- Full-length vs fixed-length protocol distinction.
- Inference command.
- Score file locations.
- Main metrics.
- Report artifact locations.

## LoRA Visualization Report

Visualization report directory:

```text
reports/lora_ultrashort_mdt_protocol_full_fixed_trend_report
```

Main report:

```text
reports/lora_ultrashort_mdt_protocol_full_fixed_trend_report/report.md
```

Important CSV outputs:

```text
combined_metrics.csv
fixed_length_16_scores_summary.csv
fixed_length_8_pooled_summary.csv
fixed_length_by_bin_summary.csv
full_length_by_bin_summary.csv
full_length_overall_summary.csv
best_fixed_by_dataset.csv
score_file_line_counts.csv
random_start_delta.csv
threshold_free_class_accuracy_summary.csv
multi_window_aggregation_metrics.csv
multi_window_aggregation_by_true_duration_bin_metrics.csv
multi_window_aggregation_score_file_index.csv
```

Important visualization outputs:

```text
eer_trend_fixed_vs_full_by_dataset.png
auc_accuracy_trend.png
full_length_eer_by_bin.png
random_start_delta.png
threshold_free_class_accuracy_trend.png
multi_window_aggregation_metrics.png
multi_window_class_accuracy.png
multi_window_aggregation_by_true_duration_bin_metrics.png
multi_window_aggregation_by_true_duration_bin_class_accuracy.png
score_distribution_full_vs_best_fixed.png
```

## Main LoRA Metrics

Full-length overall:

| dataset_type | samples | EER % | AUC % | threshold-free acc % | bonafide acc % | spoof acc % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| observed_short | 5,255 | 3.614907 | 99.459687 | 96.384396 | 96.937919 | 95.924765 |
| vad_short | 74,745 | 2.086123 | 99.794392 | 97.296140 | 98.763475 | 96.998181 |

Best fixed-length by dataset:

| dataset_type | random_start | inference_duration | samples | EER % | AUC % | threshold-free acc % |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| observed_short | false | 2.0s | 5,255 | 2.644889 | 99.599399 | 97.031399 |
| vad_short | true | 2.0s | 74,745 | 1.837718 | 99.829744 | 97.792494 |

Previously noted pooled headline metrics:

```text
Full utterance pooled:
  EER 2.309487
  AUC 99.753967
  Accuracy 97.236250

Best fixed pooled:
  random_start=true, 2.0s
  EER 2.045641
  AUC 99.790471
  Accuracy 97.693750
```

## Missing Threshold-Free CSV Fix

User reported that LoRA report missed:

```text
threshold_free_class_accuracy_summary.csv
```

Reference baseline file:

```text
reports/baseline_mdt_protocol_full_fixed_trend_report/threshold_free_class_accuracy_summary.csv
```

Generated LoRA file:

```text
reports/lora_ultrashort_mdt_protocol_full_fixed_trend_report/threshold_free_class_accuracy_summary.csv
```

Shape:

```text
22 rows x 15 columns
```

Schema matched baseline exactly:

```text
eval_type
dataset_type
duration_bin_eval
random_start
inference_duration
duration_sec
samples
threshold_free_accuracy_percent
threshold_free_bonafide_accuracy_percent
threshold_free_spoof_accuracy_percent
eer_percent
mdr_at_far1_percent
f1_at_eer_threshold_percent
auc_percent
score_file
```

Also generated:

```text
reports/lora_ultrashort_mdt_protocol_full_fixed_trend_report/threshold_free_class_accuracy_trend.png
```

And updated:

```text
reports/lora_ultrashort_mdt_protocol_full_fixed_trend_report/report.md
```

## Multi-Window True-Duration-Bin Fix

User noticed row mismatch:

```text
baseline multi_window_aggregation_by_true_duration_bin_metrics.csv: 48 rows
LoRA initial file: 33 rows
```

Root cause:

The initial LoRA generation grouped `observed_short` as one bin:

```text
observed_short_0p5_2p0
```

Baseline logic in:

```text
scripts/benchmark_py/multi_window_aggregation_report.py
```

re-bins `observed_short` using:

```text
total_speech_duration_sec
```

into:

```text
0.5-1.0s
1.0-1.5s
1.5-2.0s
```

For `vad_short`, baseline maps manifest `duration_bin`:

```text
vad_0p5_1p0 -> 0.5-1.0s
vad_1p0_1p5 -> 1.0-1.5s
vad_1p5_2p0 -> 1.5-2.0s
```

Expected row count:

```text
2 dataset types x 2 random_start modes x 3 true_duration_bins x 4 aggregations = 48 rows
```

Vectorized regenerated LoRA artifacts:

```text
reports/lora_ultrashort_mdt_protocol_full_fixed_trend_report/multi_window_aggregation_metrics.csv
reports/lora_ultrashort_mdt_protocol_full_fixed_trend_report/multi_window_aggregation_by_true_duration_bin_metrics.csv
reports/lora_ultrashort_mdt_protocol_full_fixed_trend_report/multi_window_aggregation_score_file_index.csv
```

Final shapes:

```text
multi_window_aggregation_metrics.csv: 16 rows x 42 columns
multi_window_aggregation_by_true_duration_bin_metrics.csv: 48 rows x 47 columns
multi_window_aggregation_score_file_index.csv: 16 rows
```

Generated aggregated score files:

```text
logs/results/lora_ultrashort_mdt_protocol_test_multi_window_aggregation
```

Filename pattern:

```text
{dataset}_random_start_{true|false}_multi_window_{min|max|mean|median}_xlsr_conformertcm_mdt_lora_ultrashort_vadshort.txt
```

Generated/updated visualizations:

```text
multi_window_aggregation_metrics.png
multi_window_class_accuracy.png
multi_window_aggregation_by_true_duration_bin_metrics.png
multi_window_aggregation_by_true_duration_bin_class_accuracy.png
```

Validation result:

```text
baseline shape: 48 x 47
LoRA shape:     48 x 47
columns_match:  true
```

Every group has 4 aggregation rows:

```text
dataset_type x random_start x true_duration_bin -> min, max, mean, median
```

## Comparison Report Versus Baseline

Comparison script added:

```text
scripts/benchmark_py/compare_lora_ultrashort_vs_baseline_report.py
```

Reproduce:

```bash
cd /home/hungdx/code/Lightning-hydra-ADD
python scripts/benchmark_py/compare_lora_ultrashort_vs_baseline_report.py
```

Comparison output directory:

```text
reports/lora_ultrashort_vs_baseline_mdt_protocol_comparison_report
```

Main report:

```text
reports/lora_ultrashort_vs_baseline_mdt_protocol_comparison_report/report.md
```

Comparison inputs:

```text
reports/baseline_mdt_protocol_full_fixed_trend_report
reports/lora_ultrashort_mdt_protocol_full_fixed_trend_report
```

Priority CSVs:

```text
threshold_free_class_accuracy_summary.csv
multi_window_aggregation_by_true_duration_bin_metrics.csv
```

Comparison CSV outputs:

```text
threshold_free_class_accuracy_comparison.csv
multi_window_aggregation_by_true_duration_bin_comparison.csv
metric_improvement_summary.csv
headline_summary.csv
multi_window_class_balance_delta.csv
```

Comparison visualization outputs:

```text
threshold_free_improvement_heatmap.png
threshold_free_bonafide_accuracy_improvement_heatmap.png
threshold_free_spoof_accuracy_improvement_heatmap.png
threshold_free_metric_improvement_bars.png
threshold_free_eer_vs_accuracy_delta.png
multi_window_eer_improvement_heatmap.png
multi_window_threshold_free_accuracy_improvement_heatmap.png
multi_window_bonafide_accuracy_improvement_heatmap.png
multi_window_spoof_accuracy_improvement_heatmap.png
multi_window_class_balance_shift_heatmap.png
multi_window_eer_vs_accuracy_delta.png
```

Comparison semantics:

- For accuracy, AUC, and F1:

```text
improvement = LoRA - baseline
```

- For EER and MDR:

```text
improvement = baseline - LoRA
```

because lower is better.

Comparison headline:

| source | metric | rows | mean improvement | improved rows | regressed rows | unchanged rows |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| threshold_free | threshold_free_accuracy_percent | 22 | 0.894956 | 12 | 10 | 0 |
| threshold_free | eer_percent | 22 | 0.935340 | 22 | 0 | 0 |
| threshold_free | auc_percent | 22 | 0.415284 | 22 | 0 | 0 |
| multi_window_true_duration_bin | threshold_free_accuracy_percent | 48 | 0.523666 | 23 | 24 | 1 |
| multi_window_true_duration_bin | eer_percent | 48 | 1.101498 | 45 | 2 | 1 |
| multi_window_true_duration_bin | auc_percent | 48 | 0.284841 | 40 | 8 | 0 |

Class-accuracy tradeoff found in comparison:

| source | class metric | mean improvement | improved rows | regressed rows |
| --- | --- | ---: | ---: | ---: |
| threshold_free | threshold_free_bonafide_accuracy_percent | 8.323262 | 22/22 | 0/22 |
| threshold_free | threshold_free_spoof_accuracy_percent | -2.176019 | 1/22 | 21/22 |
| multi_window_true_duration_bin | threshold_free_bonafide_accuracy_percent | 3.452474 | 48/48 | 0/48 |
| multi_window_true_duration_bin | threshold_free_spoof_accuracy_percent | -1.150087 | 0/48 | 48/48 |

Interpretation:

LoRA ultrashort MDT clearly improves ranking-style metrics: EER and AUC improve strongly across the priority comparisons. Threshold-free overall accuracy is mixed. The reason is a class-balance shift: bonafide threshold-free accuracy improves strongly, but spoof threshold-free accuracy often decreases. For production threshold-free decisions, inspect the class-accuracy heatmaps and consider recalibration/threshold tuning.

## Report File Map

Procedural run summary:

```text
plan/lora_ultrashort_mdt_protocol_inference_summary.md
```

LoRA report with visualizations:

```text
reports/lora_ultrashort_mdt_protocol_full_fixed_trend_report/report.md
```

Baseline-vs-LoRA comparison report:

```text
reports/lora_ultrashort_vs_baseline_mdt_protocol_comparison_report/report.md
```

Full inference result root:

```text
logs/results/lora_ultrashort_mdt_protocol_test_eval_full_100epoch
```

Multi-window aggregated score root:

```text
logs/results/lora_ultrashort_mdt_protocol_test_multi_window_aggregation
```

Best checkpoint:

```text
logs/train/runs/2026-05-30_19-32-38/checkpoints/epoch_021.ckpt
```

## Verification Commands

Check priority report schemas and shapes:

```bash
python - <<'PY'
import pandas as pd
from pathlib import Path

for root in [
    Path("reports/baseline_mdt_protocol_full_fixed_trend_report"),
    Path("reports/lora_ultrashort_mdt_protocol_full_fixed_trend_report"),
]:
    print(root)
    for name in [
        "threshold_free_class_accuracy_summary.csv",
        "multi_window_aggregation_by_true_duration_bin_metrics.csv",
    ]:
        df = pd.read_csv(root / name)
        print(name, df.shape)
PY
```

Expected:

```text
threshold_free_class_accuracy_summary.csv: 22 x 15
multi_window_aggregation_by_true_duration_bin_metrics.csv: 48 x 47
```

Check comparison outputs:

```bash
find reports/lora_ultrashort_vs_baseline_mdt_protocol_comparison_report -maxdepth 1 -type f | sort
file reports/lora_ultrashort_vs_baseline_mdt_protocol_comparison_report/*.png
```

Check for active train/eval jobs:

```bash
ps -eo pid,cmd | rg 'src/train.py|run_training_gate.py|short_duration_eval|benchmark.py|eval_lora_ultrashort|python -' | rg -v 'rg ' || true
```

## Known Caveats

- The training run requested 100 epochs but ended by early stopping at epoch 25. This is expected for this completed run.
- The best checkpoint used for inference was `epoch_021.ckpt`, not `last.ckpt`.
- Fixed-length inference uses original source utterances, not already-short processed segments.
- Threshold-free accuracy is not the same as EER/AUC quality. In this run, LoRA improves EER/AUC but shifts threshold-free class balance toward bonafide.
- `reports/` may be gitignored; use filesystem paths above even if `git status` does not show all generated report files.

