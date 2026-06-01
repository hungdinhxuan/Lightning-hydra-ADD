#!/usr/bin/env python3
"""Compare LoRA ultrashort MDT report artifacts against the baseline report."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINE_DIR = REPO_ROOT / os.getenv(
    "COMPARE_BASELINE_DIR", "reports/baseline_mdt_protocol_full_fixed_trend_report"
)
LORA_DIR = REPO_ROOT / os.getenv(
    "COMPARE_LORA_DIR", "reports/lora_ultrashort_mdt_protocol_full_fixed_trend_report"
)
OUT_DIR = REPO_ROOT / os.getenv(
    "COMPARE_OUT_DIR", "reports/lora_ultrashort_vs_baseline_mdt_protocol_comparison_report"
)

THRESHOLD_FILE = "threshold_free_class_accuracy_summary.csv"
MULTI_WINDOW_FILE = "multi_window_aggregation_by_true_duration_bin_metrics.csv"

POSITIVE_METRICS = {
    "threshold_free_accuracy_percent",
    "threshold_free_bonafide_accuracy_percent",
    "threshold_free_spoof_accuracy_percent",
    "accuracy_at_eer_threshold_percent",
    "f1_at_eer_threshold_percent",
    "best_f1_percent",
    "auc_percent",
}
NEGATIVE_METRICS = {
    "eer_percent",
    "mdr_at_far1_percent",
    "best_f1_far_percent",
    "best_f1_mdr_percent",
}


def read_csv(report_dir: Path, name: str) -> pd.DataFrame:
    path = report_dir / name
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def normalize_key_value(value: object) -> str:
    if pd.isna(value):
        return "<na>"
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (float, np.floating)) and value.is_integer():
        return str(int(value))
    return str(value)


def normalize_join_keys(frame: pd.DataFrame, keys: Iterable[str]) -> pd.DataFrame:
    result = frame.copy()
    for key in keys:
        result[f"__key_{key}"] = result[key].map(normalize_key_value)
    return result


def compare_frames(
    baseline: pd.DataFrame,
    lora: pd.DataFrame,
    keys: list[str],
    metrics: list[str],
) -> pd.DataFrame:
    baseline_norm = normalize_join_keys(baseline, keys)
    lora_norm = normalize_join_keys(lora, keys)
    join_keys = [f"__key_{key}" for key in keys]

    keep = join_keys + keys + [column for column in ["samples", "bonafide_samples", "spoof_samples"] if column in baseline]
    baseline_keep = baseline_norm[keep + metrics].copy()
    lora_keep = lora_norm[join_keys + metrics].copy()

    merged = baseline_keep.merge(
        lora_keep,
        on=join_keys,
        how="outer",
        suffixes=("_baseline", "_lora"),
        indicator=True,
    )
    missing = merged[merged["_merge"] != "both"]
    if not missing.empty:
        missing.to_csv(OUT_DIR / "unmatched_rows.csv", index=False)
        raise RuntimeError(f"Found {len(missing)} unmatched rows. See unmatched_rows.csv")

    merged = merged.drop(columns=join_keys + ["_merge"])
    for metric in metrics:
        baseline_col = f"{metric}_baseline"
        lora_col = f"{metric}_lora"
        delta_col = f"{metric}_delta"
        improvement_col = f"{metric}_improvement"
        merged[delta_col] = merged[lora_col] - merged[baseline_col]
        if metric in NEGATIVE_METRICS:
            merged[improvement_col] = -merged[delta_col]
        else:
            merged[improvement_col] = merged[delta_col]
    return merged


def metric_summary(frame: pd.DataFrame, source: str, metrics: list[str]) -> pd.DataFrame:
    rows = []
    for metric in metrics:
        improvement = frame[f"{metric}_improvement"].astype(float)
        delta = frame[f"{metric}_delta"].astype(float)
        rows.append(
            {
                "source": source,
                "metric": metric,
                "rows": int(improvement.notna().sum()),
                "mean_delta_lora_minus_baseline": float(delta.mean()),
                "median_delta_lora_minus_baseline": float(delta.median()),
                "min_delta_lora_minus_baseline": float(delta.min()),
                "max_delta_lora_minus_baseline": float(delta.max()),
                "mean_improvement": float(improvement.mean()),
                "median_improvement": float(improvement.median()),
                "improved_rows": int((improvement > 0).sum()),
                "regressed_rows": int((improvement < 0).sum()),
                "unchanged_rows": int((improvement == 0).sum()),
            }
        )
    return pd.DataFrame(rows)


def format_float(value: float, digits: int = 3) -> str:
    if pd.isna(value) or not math.isfinite(float(value)):
        return "n/a"
    return f"{float(value):.{digits}f}"


def md_cell(value: object) -> str:
    return str(value).replace("|", "<br>")


def heatmap(
    frame: pd.DataFrame,
    index: str,
    columns: str,
    values: str,
    title: str,
    output: Path,
    cmap_name: str = "RdYlGn",
) -> None:
    pivot = frame.pivot_table(index=index, columns=columns, values=values, aggfunc="mean")
    fig_width = max(8.0, 1.8 * len(pivot.columns))
    fig_height = max(4.5, 0.55 * len(pivot.index) + 1.8)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    data = pivot.to_numpy(dtype=float)
    vmax = np.nanmax(np.abs(data)) if np.isfinite(data).any() else 1.0
    if vmax == 0:
        vmax = 1.0
    image = ax.imshow(data, cmap=cmap_name, vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_title(title)
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for y in range(data.shape[0]):
        for x in range(data.shape[1]):
            value = data[y, x]
            if np.isfinite(value):
                ax.text(x, y, format_float(value, 2), ha="center", va="center", fontsize=8)
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Improvement points, LoRA vs baseline")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def grouped_bar(
    frame: pd.DataFrame,
    label_col: str,
    metrics: list[str],
    title: str,
    output: Path,
) -> None:
    plot_frame = frame[[label_col] + [f"{metric}_improvement" for metric in metrics]].copy()
    plot_frame = plot_frame.rename(columns={f"{metric}_improvement": metric for metric in metrics})
    labels = plot_frame[label_col].tolist()
    x = np.arange(len(labels))
    width = 0.8 / len(metrics)
    fig_width = max(10.0, 0.55 * len(labels))
    fig, ax = plt.subplots(figsize=(fig_width, 5.8))
    for idx, metric in enumerate(metrics):
        offset = (idx - (len(metrics) - 1) / 2) * width
        ax.bar(x + offset, plot_frame[metric], width, label=metric.replace("_percent", ""))
    ax.axhline(0, color="#222222", linewidth=0.9)
    ax.set_title(title)
    ax.set_ylabel("Improvement points, LoRA vs baseline")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right")
    ax.legend(ncols=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def scatter_delta(
    frame: pd.DataFrame,
    x_metric: str,
    y_metric: str,
    color_col: str,
    title: str,
    output: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    values = sorted(frame[color_col].map(str).unique())
    cmap = plt.get_cmap("tab10")
    for idx, value in enumerate(values):
        subset = frame[frame[color_col].map(str) == value]
        ax.scatter(
            subset[f"{x_metric}_improvement"],
            subset[f"{y_metric}_improvement"],
            label=value,
            s=64,
            alpha=0.82,
            color=cmap(idx % 10),
        )
    ax.axhline(0, color="#222222", linewidth=0.8)
    ax.axvline(0, color="#222222", linewidth=0.8)
    ax.set_xlabel(f"{x_metric.replace('_percent', '')} improvement points")
    ax.set_ylabel(f"{y_metric.replace('_percent', '')} improvement points")
    ax.set_title(title)
    ax.legend(title=color_col, fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def compact_label(row: pd.Series) -> str:
    parts = [
        str(row.get("eval_type", "")),
        str(row.get("dataset_type", "")),
        str(row.get("duration_bin_eval", "")),
        str(row.get("random_start", "")),
        str(row.get("inference_duration", "")),
    ]
    return " | ".join(part for part in parts if part and part != "<na>" and part != "nan")


def multi_window_label(row: pd.Series) -> str:
    return " | ".join(
        [
            str(row["dataset_type"]),
            f"rs={row['random_start']}",
            str(row["true_duration_bin"]),
            str(row["aggregation"]),
        ]
    )


def write_markdown(
    threshold_cmp: pd.DataFrame,
    multi_cmp: pd.DataFrame,
    summary: pd.DataFrame,
    out_path: Path,
) -> None:
    threshold_core = [
        "threshold_free_accuracy_percent",
        "threshold_free_bonafide_accuracy_percent",
        "threshold_free_spoof_accuracy_percent",
        "eer_percent",
        "mdr_at_far1_percent",
        "auc_percent",
    ]
    multi_core = [
        "threshold_free_accuracy_percent",
        "threshold_free_bonafide_accuracy_percent",
        "threshold_free_spoof_accuracy_percent",
        "eer_percent",
        "mdr_at_far1_percent",
        "auc_percent",
    ]

    threshold_summary = summary[summary["source"].eq("threshold_free")]
    multi_summary = summary[summary["source"].eq("multi_window_true_duration_bin")]
    tf_acc = threshold_summary[
        threshold_summary["metric"].eq("threshold_free_accuracy_percent")
    ].iloc[0]
    mw_acc = multi_summary[multi_summary["metric"].eq("threshold_free_accuracy_percent")].iloc[0]
    tf_eer = threshold_summary[threshold_summary["metric"].eq("eer_percent")].iloc[0]
    mw_eer = multi_summary[multi_summary["metric"].eq("eer_percent")].iloc[0]

    best_mw = multi_cmp.sort_values("eer_percent_improvement", ascending=False).head(8)
    worst_mw = multi_cmp.sort_values("eer_percent_improvement", ascending=True).head(8)
    best_tf = threshold_cmp.sort_values("threshold_free_accuracy_percent_improvement", ascending=False).head(8)

    lines = [
        "# LoRA Ultrashort MDT vs Baseline Comparison",
        "",
        "This report compares the generated LoRA ultrashort MDT report against the baseline report.",
        "",
        "## Source Artifacts",
        "",
        f"- Baseline: `{BASELINE_DIR.relative_to(REPO_ROOT)}`",
        f"- LoRA: `{LORA_DIR.relative_to(REPO_ROOT)}`",
        f"- Priority CSV 1: `{THRESHOLD_FILE}`",
        f"- Priority CSV 2: `{MULTI_WINDOW_FILE}`",
        "",
        "## Headline",
        "",
        "- Positive improvement means LoRA is better than baseline.",
        "- For accuracy, AUC, and F1, improvement is `LoRA - baseline`.",
        "- For EER and MDR, improvement is `baseline - LoRA` because lower is better.",
        "",
        "| Area | Metric | Mean improvement | Improved rows | Regressed rows |",
        "| --- | --- | ---: | ---: | ---: |",
        (
            "| Threshold-free summary | threshold_free_accuracy_percent | "
            f"{format_float(tf_acc['mean_improvement'])} | {int(tf_acc['improved_rows'])}/{int(tf_acc['rows'])} | "
            f"{int(tf_acc['regressed_rows'])}/{int(tf_acc['rows'])} |"
        ),
        (
            "| Threshold-free summary | eer_percent | "
            f"{format_float(tf_eer['mean_improvement'])} | {int(tf_eer['improved_rows'])}/{int(tf_eer['rows'])} | "
            f"{int(tf_eer['regressed_rows'])}/{int(tf_eer['rows'])} |"
        ),
        (
            "| Multi-window true-bin | threshold_free_accuracy_percent | "
            f"{format_float(mw_acc['mean_improvement'])} | {int(mw_acc['improved_rows'])}/{int(mw_acc['rows'])} | "
            f"{int(mw_acc['regressed_rows'])}/{int(mw_acc['rows'])} |"
        ),
        (
            "| Multi-window true-bin | eer_percent | "
            f"{format_float(mw_eer['mean_improvement'])} | {int(mw_eer['improved_rows'])}/{int(mw_eer['rows'])} | "
            f"{int(mw_eer['regressed_rows'])}/{int(mw_eer['rows'])} |"
        ),
        "",
        "## Class Accuracy Tradeoff",
        "",
        "| Area | Class metric | Mean improvement | Improved rows | Regressed rows |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for area, source, metric in [
        ("Threshold-free summary", "threshold_free", "threshold_free_bonafide_accuracy_percent"),
        ("Threshold-free summary", "threshold_free", "threshold_free_spoof_accuracy_percent"),
        (
            "Multi-window true-bin",
            "multi_window_true_duration_bin",
            "threshold_free_bonafide_accuracy_percent",
        ),
        (
            "Multi-window true-bin",
            "multi_window_true_duration_bin",
            "threshold_free_spoof_accuracy_percent",
        ),
    ]:
        row = summary[(summary["source"].eq(source)) & (summary["metric"].eq(metric))].iloc[0]
        lines.append(
            f"| {area} | {metric} | {format_float(row['mean_improvement'])} | "
            f"{int(row['improved_rows'])}/{int(row['rows'])} | {int(row['regressed_rows'])}/{int(row['rows'])} |"
        )

    lines.extend(
        [
            "",
        "## Visualizations",
        "",
        "### Threshold-Free Summary",
        "",
        "![Threshold-free improvement heatmap](threshold_free_improvement_heatmap.png)",
        "",
        "![Threshold-free bonafide accuracy improvement](threshold_free_bonafide_accuracy_improvement_heatmap.png)",
        "",
        "![Threshold-free spoof accuracy improvement](threshold_free_spoof_accuracy_improvement_heatmap.png)",
        "",
        "![Threshold-free metric deltas](threshold_free_metric_improvement_bars.png)",
        "",
        "![Threshold-free EER vs accuracy improvement](threshold_free_eer_vs_accuracy_delta.png)",
        "",
        "### Multi-Window Aggregation By True Duration Bin",
        "",
        "![Multi-window EER improvement](multi_window_eer_improvement_heatmap.png)",
        "",
        "![Multi-window threshold-free accuracy improvement](multi_window_threshold_free_accuracy_improvement_heatmap.png)",
        "",
        "![Multi-window bonafide accuracy improvement](multi_window_bonafide_accuracy_improvement_heatmap.png)",
        "",
        "![Multi-window spoof accuracy improvement](multi_window_spoof_accuracy_improvement_heatmap.png)",
        "",
        "![Multi-window class balance shift](multi_window_class_balance_shift_heatmap.png)",
        "",
        "![Multi-window EER vs accuracy improvement](multi_window_eer_vs_accuracy_delta.png)",
        "",
        "## Best Threshold-Free Accuracy Improvements",
        "",
        "| Case | Baseline acc | LoRA acc | Improvement | EER improvement | AUC improvement |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for _, row in best_tf.iterrows():
        lines.append(
            "| "
            f"{md_cell(compact_label(row))} | "
            f"{format_float(row['threshold_free_accuracy_percent_baseline'])} | "
            f"{format_float(row['threshold_free_accuracy_percent_lora'])} | "
            f"{format_float(row['threshold_free_accuracy_percent_improvement'])} | "
            f"{format_float(row['eer_percent_improvement'])} | "
            f"{format_float(row['auc_percent_improvement'])} |"
        )

    lines.extend(
        [
            "",
            "## Best Multi-Window EER Improvements",
            "",
            "| Case | Baseline EER | LoRA EER | EER improvement | Accuracy improvement | AUC improvement |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for _, row in best_mw.iterrows():
        lines.append(
            "| "
            f"{md_cell(multi_window_label(row))} | "
            f"{format_float(row['eer_percent_baseline'])} | "
            f"{format_float(row['eer_percent_lora'])} | "
            f"{format_float(row['eer_percent_improvement'])} | "
            f"{format_float(row['threshold_free_accuracy_percent_improvement'])} | "
            f"{format_float(row['auc_percent_improvement'])} |"
        )

    lines.extend(
        [
            "",
            "## Worst Multi-Window EER Regressions",
            "",
            "| Case | Baseline EER | LoRA EER | EER improvement | Accuracy improvement | AUC improvement |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for _, row in worst_mw.iterrows():
        lines.append(
            "| "
            f"{md_cell(multi_window_label(row))} | "
            f"{format_float(row['eer_percent_baseline'])} | "
            f"{format_float(row['eer_percent_lora'])} | "
            f"{format_float(row['eer_percent_improvement'])} | "
            f"{format_float(row['threshold_free_accuracy_percent_improvement'])} | "
            f"{format_float(row['auc_percent_improvement'])} |"
        )

    lines.extend(
        [
            "",
            "## CSV Outputs",
            "",
            "- `threshold_free_class_accuracy_comparison.csv`",
            "- `multi_window_aggregation_by_true_duration_bin_comparison.csv`",
            "- `metric_improvement_summary.csv`",
            "- `headline_summary.csv`",
            "- `multi_window_class_balance_delta.csv`",
            "",
            "Reproduce with: `python scripts/benchmark_py/compare_lora_ultrashort_vs_baseline_report.py`",
            "",
            "## Interpretation",
            "",
            "LoRA ultrashort MDT improves EER and AUC consistently across the priority comparisons. "
            "Threshold-free overall accuracy is mixed, mainly because bonafide accuracy improves strongly while spoof accuracy often decreases. "
            "Use the class-accuracy heatmaps before choosing a production threshold-free decision rule.",
            "",
        ]
    )
    out_path.write_text("\n".join(lines))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stale_files = [
        OUT_DIR / "multi_window_class_accuracy_improvement_heatmap.png",
    ]
    for path in stale_files:
        path.unlink(missing_ok=True)

    threshold_baseline = read_csv(BASELINE_DIR, THRESHOLD_FILE)
    threshold_lora = read_csv(LORA_DIR, THRESHOLD_FILE)
    multi_baseline = read_csv(BASELINE_DIR, MULTI_WINDOW_FILE)
    multi_lora = read_csv(LORA_DIR, MULTI_WINDOW_FILE)

    threshold_keys = [
        "eval_type",
        "dataset_type",
        "duration_bin_eval",
        "random_start",
        "inference_duration",
        "duration_sec",
    ]
    threshold_metrics = [
        "threshold_free_accuracy_percent",
        "threshold_free_bonafide_accuracy_percent",
        "threshold_free_spoof_accuracy_percent",
        "eer_percent",
        "mdr_at_far1_percent",
        "f1_at_eer_threshold_percent",
        "auc_percent",
    ]
    multi_keys = ["eval_type", "dataset_type", "random_start", "aggregation", "true_duration_bin"]
    multi_metrics = [
        "threshold_free_accuracy_percent",
        "threshold_free_bonafide_accuracy_percent",
        "threshold_free_spoof_accuracy_percent",
        "accuracy_at_eer_threshold_percent",
        "eer_percent",
        "mdr_at_far1_percent",
        "f1_at_eer_threshold_percent",
        "best_f1_percent",
        "auc_percent",
    ]

    threshold_cmp = compare_frames(threshold_baseline, threshold_lora, threshold_keys, threshold_metrics)
    multi_cmp = compare_frames(multi_baseline, multi_lora, multi_keys, multi_metrics)

    threshold_cmp["case_label"] = threshold_cmp.apply(compact_label, axis=1)
    multi_cmp["case_label"] = multi_cmp.apply(multi_window_label, axis=1)
    multi_cmp["duration_dataset_label"] = (
        multi_cmp["dataset_type"].astype(str)
        + " | rs="
        + multi_cmp["random_start"].astype(str)
        + " | "
        + multi_cmp["true_duration_bin"].astype(str)
    )
    multi_cmp["aggregation_duration_label"] = (
        multi_cmp["aggregation"].astype(str) + " | " + multi_cmp["true_duration_bin"].astype(str)
    )

    threshold_cmp.to_csv(OUT_DIR / "threshold_free_class_accuracy_comparison.csv", index=False)
    multi_cmp.to_csv(OUT_DIR / "multi_window_aggregation_by_true_duration_bin_comparison.csv", index=False)

    summary = pd.concat(
        [
            metric_summary(threshold_cmp, "threshold_free", threshold_metrics),
            metric_summary(multi_cmp, "multi_window_true_duration_bin", multi_metrics),
        ],
        ignore_index=True,
    )
    summary.to_csv(OUT_DIR / "metric_improvement_summary.csv", index=False)

    headline_metrics = summary[
        summary["metric"].isin(["threshold_free_accuracy_percent", "eer_percent", "auc_percent"])
    ].copy()
    headline_metrics.to_csv(OUT_DIR / "headline_summary.csv", index=False)

    heatmap(
        threshold_cmp,
        index="case_label",
        columns="eval_type",
        values="threshold_free_accuracy_percent_improvement",
        title="Threshold-Free Accuracy Improvement, LoRA vs Baseline",
        output=OUT_DIR / "threshold_free_improvement_heatmap.png",
    )
    heatmap(
        threshold_cmp,
        index="case_label",
        columns="eval_type",
        values="threshold_free_bonafide_accuracy_percent_improvement",
        title="Threshold-Free Bonafide Accuracy Improvement, LoRA vs Baseline",
        output=OUT_DIR / "threshold_free_bonafide_accuracy_improvement_heatmap.png",
    )
    heatmap(
        threshold_cmp,
        index="case_label",
        columns="eval_type",
        values="threshold_free_spoof_accuracy_percent_improvement",
        title="Threshold-Free Spoof Accuracy Improvement, LoRA vs Baseline",
        output=OUT_DIR / "threshold_free_spoof_accuracy_improvement_heatmap.png",
    )
    grouped_bar(
        threshold_cmp,
        label_col="case_label",
        metrics=["threshold_free_accuracy_percent", "eer_percent", "auc_percent"],
        title="Threshold-Free Summary Metric Improvements, LoRA vs Baseline",
        output=OUT_DIR / "threshold_free_metric_improvement_bars.png",
    )
    scatter_delta(
        threshold_cmp,
        x_metric="threshold_free_accuracy_percent",
        y_metric="eer_percent",
        color_col="eval_type",
        title="Threshold-Free Accuracy vs EER Improvement",
        output=OUT_DIR / "threshold_free_eer_vs_accuracy_delta.png",
    )

    heatmap(
        multi_cmp,
        index="duration_dataset_label",
        columns="aggregation",
        values="eer_percent_improvement",
        title="Multi-Window EER Improvement By True Duration Bin",
        output=OUT_DIR / "multi_window_eer_improvement_heatmap.png",
    )
    heatmap(
        multi_cmp,
        index="duration_dataset_label",
        columns="aggregation",
        values="threshold_free_accuracy_percent_improvement",
        title="Multi-Window Threshold-Free Accuracy Improvement By True Duration Bin",
        output=OUT_DIR / "multi_window_threshold_free_accuracy_improvement_heatmap.png",
    )
    heatmap(
        multi_cmp,
        index="duration_dataset_label",
        columns="aggregation",
        values="threshold_free_bonafide_accuracy_percent_improvement",
        title="Multi-Window Bonafide Accuracy Improvement By True Duration Bin",
        output=OUT_DIR / "multi_window_bonafide_accuracy_improvement_heatmap.png",
    )
    heatmap(
        multi_cmp,
        index="duration_dataset_label",
        columns="aggregation",
        values="threshold_free_spoof_accuracy_percent_improvement",
        title="Multi-Window Spoof Accuracy Improvement By True Duration Bin",
        output=OUT_DIR / "multi_window_spoof_accuracy_improvement_heatmap.png",
    )
    class_delta = multi_cmp.copy()
    class_delta["bonafide_minus_spoof_improvement_gap"] = (
        class_delta["threshold_free_bonafide_accuracy_percent_improvement"]
        - class_delta["threshold_free_spoof_accuracy_percent_improvement"]
    )
    class_delta.to_csv(OUT_DIR / "multi_window_class_balance_delta.csv", index=False)
    heatmap(
        class_delta,
        index="duration_dataset_label",
        columns="aggregation",
        values="bonafide_minus_spoof_improvement_gap",
        title="Multi-Window Class Balance Shift, Bonafide Improvement Minus Spoof Improvement",
        output=OUT_DIR / "multi_window_class_balance_shift_heatmap.png",
        cmap_name="PuOr",
    )
    scatter_delta(
        multi_cmp,
        x_metric="threshold_free_accuracy_percent",
        y_metric="eer_percent",
        color_col="aggregation",
        title="Multi-Window Threshold-Free Accuracy vs EER Improvement",
        output=OUT_DIR / "multi_window_eer_vs_accuracy_delta.png",
    )

    write_markdown(threshold_cmp, multi_cmp, summary, OUT_DIR / "report.md")

    print(f"Wrote comparison report: {OUT_DIR.relative_to(REPO_ROOT) / 'report.md'}")
    print(f"Threshold comparison rows: {len(threshold_cmp)}")
    print(f"Multi-window true-bin comparison rows: {len(multi_cmp)}")


if __name__ == "__main__":
    main()
