# Baseline MDT Protocol Inference Summary

This note records how the inference and reporting were done for:

```text
reports/baseline_mdt_protocol_full_fixed_trend_report
```

The important distinction is:

- Full-length evaluation uses the processed short audio directly:
  `observed_short` and `vad_short`.
- Fixed-length evaluation does not use processed short audio as the waveform source.
  It uses the original source utterance for both `observed_short` and `vad_short`,
  then applies fixed crop length at inference time.

## Inputs

Dataset root:

```text
/data/add_processed_vad_short_eval10k_per_dataset_train80k_dev10k_29May2026
```

Protocol and manifest:

```text
/data/add_processed_vad_short_eval10k_per_dataset_train80k_dev10k_29May2026/subsets/eval_10k_per_dataset.protocol.txt
/data/add_processed_vad_short_eval10k_per_dataset_train80k_dev10k_29May2026/subsets/eval_10k_per_dataset.manifest.csv
```

Model:

```text
/NAS1_pretrained_lab/May_21_2026_TCM_MDT.pt
```

Config:

```text
xlsr_conformertcm_normal
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
scripts/benchmark_py/short_duration_eval.py
```

It prepares benchmark folders under:

```text
data/protocol_test_eval_benchmark
```

Prepared tracks:

```text
fixed_length_test
observed_short_full
vad_short_full
```

Preparation command:

```bash
python scripts/benchmark_py/short_duration_eval.py prepare \
  --source-root /data/add_processed_vad_short_eval10k_per_dataset_train80k_dev10k_29May2026 \
  --work-dir data/protocol_test_eval_benchmark \
  --subsets test \
  --tracks fixed_length_test observed_short_full vad_short_full \
  --comment-prefix baseline_mdt_protocol_test
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

This is the key point: fixed-length crops are sampled from the original utterance,
not from the already processed short segment.

## Full-Length Inference

Full-length evaluation runs on the processed short audio. No fixed crop length is
used.

Benchmark settings:

```text
run_group = full_length
full_utterance = true
trim_length = 0
random_start = false
data.args.no_pad = true
```

Command pattern:

```bash
python scripts/benchmark_py/short_duration_eval.py run-single \
  --source-root /data/add_processed_vad_short_eval10k_per_dataset_train80k_dev10k_29May2026 \
  --work-dir data/protocol_test_eval_benchmark \
  --results-dir logs/results/baseline_mdt_protocol_test_eval \
  --subsets test \
  --tracks observed_short_full vad_short_full \
  --run-group full_length \
  --full-utterance \
  --gpus 0 \
  --batch-size 128 \
  --config xlsr_conformertcm_normal \
  --model-path /NAS1_pretrained_lab/May_21_2026_TCM_MDT.pt \
  --comment-prefix baseline_mdt_protocol_test
```

Expected full-length score files:

```text
logs/results/baseline_mdt_protocol_test_eval/baseline_mdt_protocol_test_full/observed_short_full_xlsr_conformertcm_normal_baseline_mdt_protocol_test_full.txt
logs/results/baseline_mdt_protocol_test_eval/baseline_mdt_protocol_test_full/vad_short_full_xlsr_conformertcm_normal_baseline_mdt_protocol_test_full.txt
```

Full-length bins:

- `observed_short`: one full observed-short group.
- `vad_short`: grouped by manifest `duration_bin`:
  `vad_0p5_1p0`, `vad_1p0_1p5`, `vad_1p5_2p0`.

## Fixed-Length Inference

Fixed-length evaluation uses the original source utterance for both datasets.
The model still receives fixed-duration crops at inference time.

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

Command pattern for deterministic fixed crop:

```bash
python scripts/benchmark_py/short_duration_eval.py run-single \
  --source-root /data/add_processed_vad_short_eval10k_per_dataset_train80k_dev10k_29May2026 \
  --work-dir data/protocol_test_eval_benchmark \
  --results-dir logs/results/baseline_mdt_protocol_test_eval \
  --subsets test \
  --tracks fixed_length_test \
  --run-group fixed_length \
  --durations 0.5 1.0 1.5 2.0 \
  --no-random-start \
  --gpus 0 \
  --batch-size 128 \
  --config xlsr_conformertcm_normal \
  --model-path /NAS1_pretrained_lab/May_21_2026_TCM_MDT.pt \
  --comment-prefix baseline_mdt_protocol_test
```

Command pattern for random-start fixed crop:

```bash
python scripts/benchmark_py/short_duration_eval.py run-single \
  --source-root /data/add_processed_vad_short_eval10k_per_dataset_train80k_dev10k_29May2026 \
  --work-dir data/protocol_test_eval_benchmark \
  --results-dir logs/results/baseline_mdt_protocol_test_random_start_eval \
  --subsets test \
  --tracks fixed_length_test \
  --run-group fixed_length \
  --durations 0.5 1.0 1.5 2.0 \
  --random-start \
  --gpus 0 \
  --batch-size 128 \
  --config xlsr_conformertcm_normal \
  --model-path /NAS1_pretrained_lab/May_21_2026_TCM_MDT.pt \
  --comment-prefix baseline_mdt_protocol_test_random_start
```

Internally this wrapper calls:

```text
scripts/benchmark_py/benchmark.py
```

That script calls `src/train.py` with Hydra overrides:

```text
++train=False
++test=True
++model.spec_eval=True
++data.args.random_start=<true|false>
++data.args.trim_length=<8000|16000|24000|32000>
++model.base_model_path=/NAS1_pretrained_lab/May_21_2026_TCM_MDT.pt
++model.is_base_model_path_ln=false
```

The resulting combined fixed-length score files are split by dataset type into a
matrix:

```text
logs/results/baseline_mdt_protocol_test_dataset_matrix
```

Expected matrix:

```text
2 datasets x 2 random_start modes x 4 durations = 16 score files
```

Filename pattern:

```text
{dataset}_random_start_{false|true}_{duration}_xlsr_conformertcm_normal.txt
```

Examples:

```text
observed_short_random_start_false_0.5s_xlsr_conformertcm_normal.txt
observed_short_random_start_true_2.0s_xlsr_conformertcm_normal.txt
vad_short_random_start_false_0.5s_xlsr_conformertcm_normal.txt
vad_short_random_start_true_2.0s_xlsr_conformertcm_normal.txt
```

Expected line counts:

```text
observed_short: 5,255 rows per fixed-length score file
vad_short: 74,745 rows per fixed-length score file
```

## Metrics

Metric implementation is in:

```text
scripts/benchmark_py/binary_eval.py
```

Labels:

```text
bonafide = 1
spoof = 0
```

Main metrics:

- EER: computed from ROC using `bonafide_score`; lower is better.
- AUC: ROC-AUC using `bonafide_score`; higher is better.
- MDR @ FAR=1%: choose the threshold whose FAR is `<= 0.01`, then report
  miss detection rate.
- F1 @ EER threshold: threshold is the EER operating point.
- Best F1: best F1 over all score thresholds.
- Threshold-Free accuracy: no external threshold, predict bonafide if
  `bonafide_score >= spoof_score`.
- Threshold-Free bonafide accuracy: class accuracy for bonafide using the same
  score comparison.
- Threshold-Free spoof accuracy: class accuracy for spoof using the same score
  comparison.

## Multi-Window Aggregation

The aggregation script is:

```text
scripts/benchmark_py/multi_window_aggregation_report.py
```

Input:

```text
logs/results/baseline_mdt_protocol_test_dataset_matrix/*.txt
```

For each tuple:

```text
dataset_type in {observed_short, vad_short}
random_start in {false, true}
aggregation in {min, max, mean, median}
```

the script loads four fixed-window score files:

```text
0.5s + 1.0s + 1.5s + 2.0s
```

Then it validates that every sample has all four windows and aggregates the two
logits separately:

```text
aggregated_bonafide_score = agg([bonafide_0.5s, bonafide_1.0s, bonafide_1.5s, bonafide_2.0s])
aggregated_spoof_score    = agg([spoof_0.5s, spoof_1.0s, spoof_1.5s, spoof_2.0s])
```

where `agg` is one of:

```text
min
max
mean
median
```

Command:

```bash
python scripts/benchmark_py/multi_window_aggregation_report.py
```

Aggregated score output:

```text
logs/results/baseline_mdt_protocol_test_multi_window_aggregation
```

Expected aggregate score files:

```text
2 datasets x 2 random_start modes x 4 aggregation methods = 16 score files
```

Expected total lines:

```text
640,000
```

## True Duration Bin Filtering

For the by-bin multi-window report, metrics are computed after filtering the
aggregated per-sample rows by true duration bin.

The true-bin metadata comes from:

```text
/data/add_processed_vad_short_eval10k_per_dataset_train80k_dev10k_29May2026/subsets/eval_10k_per_dataset.manifest.csv
```

Mapping key:

```text
score filename: audio/original_utterance/<stem>__source.wav
manifest key:  <stem>.wav
```

Observed-short binning:

```text
use total_speech_duration_sec
0.5 <= d <= 1.0 -> 0.5-1.0s
1.0 <  d <= 1.5 -> 1.0-1.5s
1.5 <  d <= 2.0 -> 1.5-2.0s
```

VAD-short binning:

```text
use manifest duration_bin
vad_0p5_1p0 -> 0.5-1.0s
vad_1p0_1p5 -> 1.0-1.5s
vad_1p5_2p0 -> 1.5-2.0s
```

Sample counts:

| dataset_type | true_duration_bin | samples |
|---|---:|---:|
| observed_short | 0.5-1.0s | 1,386 |
| observed_short | 1.0-1.5s | 1,443 |
| observed_short | 1.5-2.0s | 2,426 |
| vad_short | 0.5-1.0s | 3,986 |
| vad_short | 1.0-1.5s | 7,540 |
| vad_short | 1.5-2.0s | 63,219 |

The by-bin table has:

```text
2 datasets x 2 random_start modes x 4 aggregation methods x 3 true duration bins = 48 rows
```

## Report Outputs

Main report:

```text
reports/baseline_mdt_protocol_full_fixed_trend_report/report.md
```

Important CSVs:

```text
reports/baseline_mdt_protocol_full_fixed_trend_report/combined_metrics.csv
reports/baseline_mdt_protocol_full_fixed_trend_report/fixed_length_16_scores_summary.csv
reports/baseline_mdt_protocol_full_fixed_trend_report/full_length_overall_summary.csv
reports/baseline_mdt_protocol_full_fixed_trend_report/full_length_by_bin_summary.csv
reports/baseline_mdt_protocol_full_fixed_trend_report/threshold_free_class_accuracy_summary.csv
reports/baseline_mdt_protocol_full_fixed_trend_report/multi_window_aggregation_metrics.csv
reports/baseline_mdt_protocol_full_fixed_trend_report/multi_window_aggregation_by_true_duration_bin_metrics.csv
reports/baseline_mdt_protocol_full_fixed_trend_report/multi_window_aggregation_score_file_index.csv
```

Important figures:

```text
reports/baseline_mdt_protocol_full_fixed_trend_report/eer_trend_fixed_vs_full_by_dataset.png
reports/baseline_mdt_protocol_full_fixed_trend_report/extra_metrics_trend.png
reports/baseline_mdt_protocol_full_fixed_trend_report/threshold_free_class_accuracy_trend.png
reports/baseline_mdt_protocol_full_fixed_trend_report/multi_window_aggregation_metrics.png
reports/baseline_mdt_protocol_full_fixed_trend_report/multi_window_class_accuracy.png
reports/baseline_mdt_protocol_full_fixed_trend_report/multi_window_aggregation_by_true_duration_bin_metrics.png
reports/baseline_mdt_protocol_full_fixed_trend_report/multi_window_aggregation_by_true_duration_bin_class_accuracy.png
```

## Sanity Checks

Before trusting the report, check:

```bash
find logs/results/baseline_mdt_protocol_test_dataset_matrix -maxdepth 1 -type f -name "*.txt" | wc -l
find logs/results/baseline_mdt_protocol_test_multi_window_aggregation -maxdepth 1 -type f -name "*.txt" | wc -l
wc -l logs/results/baseline_mdt_protocol_test_multi_window_aggregation/*.txt | tail -1
```

Expected:

```text
fixed-length split score files: 16
multi-window aggregate score files: 16
multi-window aggregate score lines: 640000 total
multi_window_aggregation_metrics.csv rows: 16
multi_window_aggregation_by_true_duration_bin_metrics.csv rows: 48
```

## Current Conclusion From The Report

- Full-length `vad_short` is stronger than full-length `observed_short`.
- For single fixed-length inference, 2.0s is the best crop duration in the
  current report.
- `observed_short` best fixed result: `2.0s`, `random_start=false`.
- `vad_short` best fixed result: `2.0s`, `random_start=true`.
- Multi-window aggregation helps `vad_short` most, especially `mean` with
  `random_start=true`.
- For `observed_short`, multi-window aggregation improves stability but does not
  beat the best single fixed 2.0s result.
- In true duration bins, `observed_short` 0.5-1.0s is the hardest group.
- Mean/median aggregation is generally more stable for EER and MDR than min/max.
