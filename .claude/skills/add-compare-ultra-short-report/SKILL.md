---
name: add-compare-ultra-short-report
description: Generate the LoRA-ultrashort vs baseline MDT comparison report (improvement CSVs, heatmaps, headline summary) from two existing trend-report directories. Use when the user wants to compare a LoRA model against baseline, quantify EER/AUC/threshold-free improvement, inspect class-balance (bonafide vs spoof) tradeoff, or produce comparison heatmaps. Triggers: "compare lora vs baseline", "improvement heatmap", "metric_improvement_summary", "class balance delta", "headline comparison".
---

# ADD LoRA-vs-Baseline Comparison Report

Diff two trend reports (`add-trend-report` outputs) into a comparison report.

## Script
`scripts/benchmark_py/compare_lora_ultrashort_vs_baseline_report.py` — env-var configured.
```bash
cd /home/hungdx/code/Lightning-hydra-ADD
export COMPARE_BASELINE_DIR=reports/baseline_mdt_protocol_full_fixed_trend_report
export COMPARE_LORA_DIR=reports/lora_ultrashort_mdt_protocol_full_fixed_trend_report
python scripts/benchmark_py/compare_lora_ultrashort_vs_baseline_report.py
```
(Defaults already point at those two dirs; override only if paths differ.)

## Priority inputs (must exist in both dirs)
```
threshold_free_class_accuracy_summary.csv
multi_window_aggregation_by_true_duration_bin_metrics.csv
```
If missing → run `add-trend-report` first.

## Output
Dir: `reports/lora_ultrashort_vs_baseline_mdt_protocol_comparison_report/` (report.md + CSVs + PNGs).
CSVs: `threshold_free_class_accuracy_comparison.csv`, `multi_window_aggregation_by_true_duration_bin_comparison.csv`, `metric_improvement_summary.csv`, `headline_summary.csv`, `multi_window_class_balance_delta.csv`.
PNGs: threshold-free + multi-window improvement heatmaps (overall / bonafide / spoof), `*_metric_improvement_bars.png`, `*_eer_vs_accuracy_delta.png`, `multi_window_class_balance_shift_heatmap.png`.

## Comparison semantics (sign convention)
- accuracy / AUC / F1: `improvement = LoRA - baseline`.
- EER / MDR (lower is better): `improvement = baseline - LoRA`.

## Interpretation guard
LoRA ultrashort typically improves ranking metrics (EER/AUC strongly), but threshold-free **overall** accuracy is mixed due to a class-balance shift: bonafide accuracy up, spoof accuracy down. Always inspect the class-accuracy heatmaps; for production threshold-free decisions consider recalibration / threshold tuning before claiming a net win.

## Verify
```bash
find reports/lora_ultrashort_vs_baseline_mdt_protocol_comparison_report -maxdepth 1 -type f | sort
file reports/lora_ultrashort_vs_baseline_mdt_protocol_comparison_report/*.png
```

## Gotchas
- `reports/` may be gitignored — check the filesystem, not `git status`.
- Both input dirs need matching schemas (22×15 threshold-free, 48×47 multi-window); a row mismatch means the LoRA trend report skipped the 48-row bin fix — fix in `add-trend-report` first.
