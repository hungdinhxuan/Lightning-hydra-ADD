---
name: add-trend-ultra-short-report
description: Build the full/fixed trend report plus the multi-window aggregation report from ADD short-duration score files, including the threshold_free_class_accuracy_summary.csv and the 48-row true-duration-bin fix. Use when the user wants a visualization report from inference scores, is missing threshold_free CSV, sees a multi-window row-count mismatch (33 vs 48 rows), or wants per-bin/aggregation metrics + PNG trends. Triggers: "generate trend report", "full_fixed_trend_report", "multi_window aggregation", "threshold free csv missing", "48 rows true duration bin".
---

# ADD Trend + Multi-Window Report

Turn score files from `add-short-eval` into a visualization report (CSVs + PNGs + report.md).

## Two scripts
1. `scripts/benchmark_py/full_fixed_trend_report.py` — main trend report (argparse CLI).
2. `scripts/benchmark_py/multi_window_aggregation_report.py` — multi-window min/max/mean/median aggregation + true-duration-bin metrics (env-var configured, no CLI args).

## 1. Trend report
```bash
python scripts/benchmark_py/full_fixed_trend_report.py \
  --results-root logs/results/lora_ultrashort_mdt_protocol_test_eval_full_100epoch \
  --work-dir     data/protocol_test_eval_benchmark_lora_full \
  --report-dir   reports/lora_ultrashort_mdt_protocol_full_fixed_trend_report \
  --config-name  xlsr_conformertcm_mdt_lora_ultrashort_vadshort \
  --prefix       lora_ultrashort_mdt_ddp_full100_ep21 \
  --checkpoint   logs/train/runs/2026-05-30_19-32-38/checkpoints/epoch_021.ckpt
```
Optional: `--title`, `--fixed-comment-pattern {prefix}_{suffix}_{duration}`, `--fixed-random-results-root`, `--dataset-matrix-dir`.

Key CSVs produced: `combined_metrics.csv`, `full_length_overall_summary.csv`, `best_fixed_by_dataset.csv`, `fixed_length_by_bin_summary.csv`, `random_start_delta.csv`, `threshold_free_class_accuracy_summary.csv`.
Key PNGs: `eer_trend_fixed_vs_full_by_dataset.png`, `auc_accuracy_trend.png`, `threshold_free_class_accuracy_trend.png`, `score_distribution_full_vs_best_fixed.png`.

## 2. Multi-window report (env-var driven)
Constants resolve from env (defaults point at the BASELINE dirs — override for LoRA):
```bash
export MW_AGG_SCORE_DIR=logs/results/lora_ultrashort_mdt_protocol_test_multi_window_aggregation
export MW_REPORT_DIR=reports/lora_ultrashort_mdt_protocol_full_fixed_trend_report
export MW_MODEL_SUFFIX=xlsr_conformertcm_mdt_lora_ultrashort_vadshort
# also available: MW_FIXED_SCORE_DIR, MW_MANIFEST_PATH
python scripts/benchmark_py/multi_window_aggregation_report.py
```
Iterates `DATASETS(observed_short,vad_short) × RANDOM_STARTS(false,true) × AGGREGATIONS(min,max,mean,median)` over windows `0.5/1.0/1.5/2.0s`, then re-bins to `TRUE_DURATION_BINS(0.5-1.0, 1.0-1.5, 1.5-2.0s)`.
Aggregated score filename: `{dataset}_random_start_{true|false}_multi_window_{min|max|mean|median}_<MODEL_SUFFIX>.txt`.

## The 48-row true-duration-bin fix (common pitfall)
Mismatch symptom: baseline `multi_window_aggregation_by_true_duration_bin_metrics.csv` = 48 rows, LoRA = 33.
Root cause: naive grouping lumps `observed_short` into one bin (`observed_short_0p5_2p0`). Correct logic re-bins:
- `observed_short` by `total_speech_duration_sec` → 0.5-1.0 / 1.0-1.5 / 1.5-2.0s.
- `vad_short` by manifest `duration_bin`: `vad_0p5_1p0→0.5-1.0s`, `vad_1p0_1p5→1.0-1.5s`, `vad_1p5_2p0→1.5-2.0s`.
Expected: `2 datasets × 2 random_start × 3 bins × 4 aggregations = 48 rows`.

## Expected final shapes
```
threshold_free_class_accuracy_summary.csv:                 22 × 15
multi_window_aggregation_metrics.csv:                      16 × 42
multi_window_aggregation_by_true_duration_bin_metrics.csv: 48 × 47
multi_window_aggregation_score_file_index.csv:             16 rows
```

## Verify (schema/shape parity vs baseline)
```bash
python - <<'PY'
import pandas as pd
from pathlib import Path
for root in [Path("reports/baseline_mdt_protocol_full_fixed_trend_report"),
             Path("reports/lora_ultrashort_mdt_protocol_full_fixed_trend_report")]:
    print(root)
    for n in ["threshold_free_class_accuracy_summary.csv",
              "multi_window_aggregation_by_true_duration_bin_metrics.csv"]:
        print(" ", n, pd.read_csv(root/n).shape)
PY
```

## Gotchas
- `reports/` may be gitignored — verify via filesystem, not `git status`.
- Default multi-window env paths target baseline; always export `MW_*` for a LoRA run or you overwrite baseline outputs.
- Match `--config-name`/`--prefix` to the eval step exactly.

Next stage: skill `add-compare-ultra-short-report`.
