#!/usr/bin/env python3
"""Aggregate fixed-window score files to simulate multi-window inference."""

from __future__ import annotations

import math
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from binary_eval import (
    _find_best_f1_threshold,
    compute_auc,
    compute_eer,
    compute_score_comparison_metrics,
    compute_threshold_metrics,
    find_threshold_at_target_far,
)


FIXED_SCORE_DIR = REPO_ROOT / os.getenv(
    "MW_FIXED_SCORE_DIR", "logs/results/baseline_mdt_protocol_test_dataset_matrix"
)
AGG_SCORE_DIR = REPO_ROOT / os.getenv(
    "MW_AGG_SCORE_DIR", "logs/results/baseline_mdt_protocol_test_multi_window_aggregation"
)
REPORT_DIR = REPO_ROOT / os.getenv(
    "MW_REPORT_DIR", "reports/baseline_mdt_protocol_full_fixed_trend_report"
)
DATA_ROOT = Path(
    os.getenv(
        "MW_DATA_ROOT",
        "/data/add_processed_vad_short_eval10k_per_dataset_train80k_dev10k_29May2026",
    )
)
MANIFEST_PATH = Path(os.getenv("MW_MANIFEST_PATH", str(DATA_ROOT / "subsets/eval_10k_per_dataset.manifest.csv")))
MODEL_SUFFIX = os.getenv("MW_MODEL_SUFFIX", "xlsr_conformertcm_normal")

DATASETS = ("observed_short", "vad_short")
RANDOM_STARTS = (False, True)
DURATIONS = ("0.5s", "1.0s", "1.5s", "2.0s")
AGGREGATIONS = ("min", "max", "mean", "median")
TRUE_DURATION_BINS = ("0.5-1.0s", "1.0-1.5s", "1.5-2.0s")


def parse_label(filename: str) -> int:
    if "__bonafide__" in filename:
        return 1
    if "__spoof__" in filename:
        return 0
    raise ValueError(f"Could not parse label from filename: {filename}")


def repo_rel(path: Path) -> str:
    path = path.resolve()
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def read_score_file(path: Path, duration: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    with path.open("r") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            filename = parts[0]
            rows.append(
                {
                    "filename": filename,
                    "duration": duration,
                    "spoof_score": float(parts[1]),
                    "score": float(parts[2]),
                    "label": parse_label(filename),
                    "label_name": "bonafide" if parse_label(filename) == 1 else "spoof",
                }
            )
    return pd.DataFrame(rows)


def score_filename_to_manifest_key(filename: str) -> str:
    return Path(filename).name.replace("__source.wav", ".wav")


def vad_bin_to_true_duration_bin(duration_bin: str) -> str:
    mapping = {
        "vad_0p5_1p0": "0.5-1.0s",
        "vad_1p0_1p5": "1.0-1.5s",
        "vad_1p5_2p0": "1.5-2.0s",
    }
    return mapping.get(duration_bin, "")


def observed_duration_to_bin(value: float) -> str:
    if not np.isfinite(value):
        return ""
    if 0.5 <= value <= 1.0:
        return "0.5-1.0s"
    if 1.0 < value <= 1.5:
        return "1.0-1.5s"
    if 1.5 < value <= 2.0:
        return "1.5-2.0s"
    return ""


@lru_cache(maxsize=1)
def load_manifest_metadata() -> pd.DataFrame:
    columns = [
        "rel_path",
        "duration_bin",
        "segment_duration_sec",
        "total_speech_duration_sec",
        "is_observed_short",
        "is_vad_segment",
    ]
    manifest = pd.read_csv(MANIFEST_PATH, usecols=columns)
    manifest["manifest_key"] = manifest["rel_path"].map(lambda value: Path(str(value)).name)
    if manifest["manifest_key"].duplicated().any():
        duplicates = manifest.loc[manifest["manifest_key"].duplicated(), "manifest_key"].head(5).tolist()
        raise RuntimeError(f"Duplicate manifest keys, examples: {duplicates}")

    observed_mask = manifest["is_observed_short"].astype(bool)
    manifest["true_duration_sec"] = manifest["segment_duration_sec"].astype(float)
    manifest.loc[observed_mask, "true_duration_sec"] = manifest.loc[
        observed_mask, "total_speech_duration_sec"
    ].astype(float)
    manifest["true_duration_bin"] = manifest["duration_bin"].map(vad_bin_to_true_duration_bin)
    manifest.loc[observed_mask, "true_duration_bin"] = manifest.loc[
        observed_mask, "true_duration_sec"
    ].map(observed_duration_to_bin)
    missing = manifest["true_duration_bin"].eq("")
    if missing.any():
        examples = manifest.loc[
            missing,
            ["rel_path", "duration_bin", "segment_duration_sec", "total_speech_duration_sec"],
        ].head(10)
        raise RuntimeError(f"Could not assign true duration bin for {int(missing.sum())} rows:\n{examples}")

    return manifest[
        [
            "manifest_key",
            "duration_bin",
            "segment_duration_sec",
            "total_speech_duration_sec",
            "true_duration_sec",
            "true_duration_bin",
        ]
    ].rename(columns={"duration_bin": "manifest_duration_bin"})


def fixed_score_path(dataset: str, random_start: bool, duration: str) -> Path:
    random_text = str(random_start).lower()
    return FIXED_SCORE_DIR / f"{dataset}_random_start_{random_text}_{duration}_{MODEL_SUFFIX}.txt"


def aggregate_values(values: pd.Series, method: str) -> float:
    if method == "min":
        return float(values.min())
    if method == "max":
        return float(values.max())
    if method == "mean":
        return float(values.mean())
    if method == "median":
        return float(values.median())
    raise ValueError(f"Unsupported aggregation: {method}")


def aggregate_frame(dataset: str, random_start: bool, method: str) -> pd.DataFrame:
    pieces = [read_score_file(fixed_score_path(dataset, random_start, duration), duration) for duration in DURATIONS]
    frame = pd.concat(pieces, ignore_index=True)
    frame["manifest_key"] = frame["filename"].map(score_filename_to_manifest_key)
    frame = frame.merge(load_manifest_metadata(), on="manifest_key", how="left")
    if frame["true_duration_bin"].isna().any():
        examples = frame.loc[frame["true_duration_bin"].isna(), "filename"].head(10).tolist()
        raise RuntimeError(f"Could not map {len(examples)} score rows to manifest metadata: {examples}")

    counts = frame.groupby("filename")["duration"].nunique()
    missing = counts[counts != len(DURATIONS)]
    if not missing.empty:
        raise RuntimeError(
            f"{dataset} random_start={random_start} has {len(missing)} utterances without all {len(DURATIONS)} windows"
        )

    grouped = frame.groupby("filename", sort=False)
    records = []
    for filename, group in grouped:
        labels = group["label"].unique()
        if len(labels) != 1:
            raise RuntimeError(f"Label mismatch for {filename}")
        records.append(
            {
                "filename": filename,
                "dataset_type": dataset,
                "random_start": random_start,
                "aggregation": method,
                "label": int(labels[0]),
                "label_name": "bonafide" if int(labels[0]) == 1 else "spoof",
                "manifest_key": group["manifest_key"].iloc[0],
                "manifest_duration_bin": group["manifest_duration_bin"].iloc[0],
                "true_duration_bin": group["true_duration_bin"].iloc[0],
                "true_duration_sec": float(group["true_duration_sec"].iloc[0]),
                "segment_duration_sec": float(group["segment_duration_sec"].iloc[0]),
                "total_speech_duration_sec": float(group["total_speech_duration_sec"].iloc[0])
                if pd.notna(group["total_speech_duration_sec"].iloc[0])
                else math.nan,
                "score": aggregate_values(group["score"], method),
                "spoof_score": aggregate_values(group["spoof_score"], method),
            }
        )
    return pd.DataFrame(records)


def write_score_file(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in frame.itertuples(index=False):
            handle.write(f"{row.filename} {row.spoof_score} {row.score}\n")


def safe_percent(value: float) -> float:
    return float(value * 100.0) if np.isfinite(value) else math.nan


def metric_row(
    frame: pd.DataFrame,
    dataset: str,
    random_start: bool,
    method: str,
    score_file: Path,
    eval_type: str = "multi_window_aggregation",
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    labels = frame["label"].to_numpy(dtype=int)
    scores = frame["score"].to_numpy(dtype=float)
    eer, eer_threshold = compute_eer(labels, scores)
    auc = compute_auc(labels, scores)
    eer_metrics = compute_threshold_metrics(labels, scores, eer_threshold)
    far1 = find_threshold_at_target_far(labels, scores, target_far=0.01)
    far1_metrics = compute_threshold_metrics(labels, scores, far1["threshold"])
    best_f1 = _find_best_f1_threshold(labels, scores)
    threshold_free = compute_score_comparison_metrics(frame)
    row = {
        "eval_type": eval_type,
        "dataset_type": dataset,
        "random_start": random_start,
        "aggregation": method,
        "inference_windows": "+".join(DURATIONS),
        "samples": int(len(frame)),
        "bonafide_samples": int((frame["label"] == 1).sum()),
        "spoof_samples": int((frame["label"] == 0).sum()),
        "eer": eer,
        "eer_percent": safe_percent(eer),
        "eer_threshold": eer_threshold,
        "auc": auc,
        "auc_percent": safe_percent(auc),
        "accuracy_at_eer_threshold": eer_metrics["accuracy"],
        "accuracy_at_eer_threshold_percent": safe_percent(eer_metrics["accuracy"]),
        "f1_at_eer_threshold": eer_metrics["f1"],
        "f1_at_eer_threshold_percent": safe_percent(eer_metrics["f1"]),
        "precision_at_eer_threshold": eer_metrics["precision"],
        "recall_at_eer_threshold": eer_metrics["recall"],
        "far_at_eer_threshold": eer_metrics["far"],
        "mdr_at_eer_threshold": eer_metrics["mdr"],
        "mdr_at_far1": far1_metrics["mdr"],
        "mdr_at_far1_percent": safe_percent(far1_metrics["mdr"]),
        "tpr_at_far1": 1.0 - far1_metrics["mdr"],
        "tpr_at_far1_percent": safe_percent(1.0 - far1_metrics["mdr"]),
        "far1_threshold": far1["threshold"],
        "far_at_far1": far1_metrics["far"],
        "far_at_far1_percent": safe_percent(far1_metrics["far"]),
        "best_f1": best_f1["f1"],
        "best_f1_percent": safe_percent(best_f1["f1"]),
        "best_f1_threshold": best_f1["threshold"],
        "best_f1_far": best_f1["far"],
        "best_f1_far_percent": safe_percent(best_f1["far"]),
        "best_f1_mdr": best_f1["frr"],
        "best_f1_mdr_percent": safe_percent(best_f1["frr"]),
        "threshold_free_accuracy": threshold_free["accuracy"],
        "threshold_free_accuracy_percent": safe_percent(threshold_free["accuracy"]),
        "threshold_free_bonafide_accuracy": threshold_free["bonafide_accuracy"],
        "threshold_free_bonafide_accuracy_percent": safe_percent(threshold_free["bonafide_accuracy"]),
        "threshold_free_spoof_accuracy": threshold_free["spoof_accuracy"],
        "threshold_free_spoof_accuracy_percent": safe_percent(threshold_free["spoof_accuracy"]),
        "score_file": repo_rel(score_file),
    }
    if extra:
        row.update(extra)
    return row


def round_table(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    out = df.loc[:, list(columns)].copy()
    for column in out.columns:
        if column.endswith("_percent"):
            out[column] = out[column].map(lambda value: round(float(value), 2) if pd.notna(value) else value)
    return out


def ordered_summary(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    out["dataset_type"] = pd.Categorical(out["dataset_type"], categories=list(DATASETS), ordered=True)
    out["aggregation"] = pd.Categorical(out["aggregation"], categories=list(AGGREGATIONS), ordered=True)
    return out.sort_values(["dataset_type", "random_start", "aggregation"]).reset_index(drop=True)


def ordered_by_bin(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    out["dataset_type"] = pd.Categorical(out["dataset_type"], categories=list(DATASETS), ordered=True)
    out["true_duration_bin"] = pd.Categorical(
        out["true_duration_bin"], categories=list(TRUE_DURATION_BINS), ordered=True
    )
    out["aggregation"] = pd.Categorical(out["aggregation"], categories=list(AGGREGATIONS), ordered=True)
    return out.sort_values(["dataset_type", "random_start", "true_duration_bin", "aggregation"]).reset_index(
        drop=True
    )


def plot_multi_window_metrics(summary: pd.DataFrame, output: Path) -> None:
    metrics = [
        ("eer_percent", "EER (%)", False),
        ("mdr_at_far1_percent", "MDR @ FAR=1% (%)", False),
        ("f1_at_eer_threshold_percent", "F1 @ EER threshold (%)", True),
        ("threshold_free_accuracy_percent", "Threshold-Free accuracy (%)", True),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    for ax, (metric, title, higher_better) in zip(axes.ravel(), metrics):
        for dataset in DATASETS:
            subset = summary[summary["dataset_type"] == dataset]
            for random_start in RANDOM_STARTS:
                line = ordered_summary(subset[subset["random_start"] == random_start])
                label = f"{dataset}, random_start={str(random_start).lower()}"
                ax.plot(line["aggregation"], line[metric], marker="o", label=label)
        ax.set_title(title)
        ax.set_xlabel("aggregation")
        ax.set_ylabel(title)
        ax.grid(True, alpha=0.25)
        if higher_better:
            ax.set_ylim(bottom=max(0, summary[metric].min() - 5), top=min(100, summary[metric].max() + 3))
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Multi-window aggregation metrics from 0.5s/1.0s/1.5s/2.0s windows")
    fig.savefig(output, dpi=160)
    plt.close(fig)


def plot_class_accuracy(summary: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    for ax, dataset in zip(axes, DATASETS):
        subset = summary[summary["dataset_type"] == dataset]
        for random_start in RANDOM_STARTS:
            for metric, marker in (
                ("threshold_free_bonafide_accuracy_percent", "o"),
                ("threshold_free_spoof_accuracy_percent", "s"),
            ):
                line = ordered_summary(subset[subset["random_start"] == random_start])
                class_name = "bonafide" if "bonafide" in metric else "spoof"
                ax.plot(
                    line["aggregation"],
                    line[metric],
                    marker=marker,
                    label=f"{class_name}, random_start={str(random_start).lower()}",
                )
        ax.set_title(dataset)
        ax.set_xlabel("aggregation")
        ax.set_ylabel("Threshold-Free class accuracy (%)")
        ax.grid(True, alpha=0.25)
        ax.set_ylim(bottom=max(0, subset[["threshold_free_bonafide_accuracy_percent", "threshold_free_spoof_accuracy_percent"]].min().min() - 5), top=100)
        ax.legend(fontsize=8)
    fig.suptitle("Multi-window threshold-free per-class accuracy")
    fig.savefig(output, dpi=160)
    plt.close(fig)


def pivot_metric_by_bin(summary: pd.DataFrame, metric: str) -> pd.DataFrame:
    ordered = ordered_by_bin(summary)
    pivot = ordered.pivot_table(
        index=["dataset_type", "random_start", "true_duration_bin"],
        columns="aggregation",
        values=metric,
        observed=True,
    )
    return pivot.reindex(columns=list(AGGREGATIONS))


def draw_heatmap(ax: Any, matrix: pd.DataFrame, title: str, cmap: str) -> None:
    values = matrix.to_numpy(dtype=float)
    image = ax.imshow(values, aspect="auto", cmap=cmap)
    ax.set_title(title)
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns)
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(
        [
            f"{dataset}, rs={str(random_start).lower()}, {duration_bin}"
            for dataset, random_start, duration_bin in matrix.index
        ],
        fontsize=8,
    )
    for y in range(values.shape[0]):
        for x in range(values.shape[1]):
            ax.text(x, y, f"{values[y, x]:.1f}", ha="center", va="center", fontsize=7)
    return image


def plot_by_true_duration_bin_metrics(summary: pd.DataFrame, output: Path) -> None:
    metrics = [
        ("eer_percent", "EER (%)", "Reds"),
        ("mdr_at_far1_percent", "MDR @ FAR=1% (%)", "Oranges"),
        ("f1_at_eer_threshold_percent", "F1 @ EER threshold (%)", "Greens"),
        ("threshold_free_accuracy_percent", "Threshold-Free accuracy (%)", "Blues"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), constrained_layout=True)
    for ax, (metric, title, cmap) in zip(axes.ravel(), metrics):
        image = draw_heatmap(ax, pivot_metric_by_bin(summary, metric), title, cmap)
        fig.colorbar(image, ax=ax, fraction=0.025, pad=0.01)
    fig.suptitle("Multi-window aggregation metrics by true duration bin")
    fig.savefig(output, dpi=160)
    plt.close(fig)


def plot_by_true_duration_bin_class_accuracy(summary: pd.DataFrame, output: Path) -> None:
    metrics = [
        ("threshold_free_bonafide_accuracy_percent", "Bonafide Threshold-Free accuracy (%)", "Greens"),
        ("threshold_free_spoof_accuracy_percent", "Spoof Threshold-Free accuracy (%)", "Blues"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)
    for ax, (metric, title, cmap) in zip(axes, metrics):
        image = draw_heatmap(ax, pivot_metric_by_bin(summary, metric), title, cmap)
        fig.colorbar(image, ax=ax, fraction=0.025, pad=0.01)
    fig.suptitle("Multi-window threshold-free class accuracy by true duration bin")
    fig.savefig(output, dpi=160)
    plt.close(fig)


def update_report(summary: pd.DataFrame, by_bin: pd.DataFrame) -> None:
    report_path = REPORT_DIR / "report.md"
    report = report_path.read_text()

    insert_after = "## Threshold-Free Class Accuracy Trend\n\n![Threshold-Free class accuracy trend](threshold_free_class_accuracy_trend.png)\n"
    multi_window_section = (
        "\n## Multi-Window Aggregation Trend\n\n"
        "Aggregation pools the four fixed-window logits (`0.5s`, `1.0s`, `1.5s`, `2.0s`) per utterance, separately for bonafide and spoof logits.\n\n"
        "![Multi-window aggregation metrics](multi_window_aggregation_metrics.png)\n\n"
        "## Multi-Window Class Accuracy\n\n"
        "![Multi-window class accuracy](multi_window_class_accuracy.png)\n\n"
        "## Multi-Window By True Duration Bin\n\n"
        "Bins use the manifest metadata: observed_short uses `total_speech_duration_sec`, vad_short uses the VAD segment `duration_bin`.\n\n"
        "![Multi-window aggregation by true duration bin](multi_window_aggregation_by_true_duration_bin_metrics.png)\n\n"
        "## Multi-Window Class Accuracy By True Duration Bin\n\n"
        "![Multi-window class accuracy by true duration bin](multi_window_aggregation_by_true_duration_bin_class_accuracy.png)\n"
    )
    if "## Multi-Window Aggregation Trend" not in report:
        report = report.replace(insert_after, insert_after + multi_window_section)
    elif "## Multi-Window By True Duration Bin" not in report:
        report = report.replace(
            "## Full-Length Class Accuracy By Bin\n",
            "## Multi-Window By True Duration Bin\n\n"
            "Bins use the manifest metadata: observed_short uses `total_speech_duration_sec`, "
            "vad_short uses the VAD segment `duration_bin`.\n\n"
            "![Multi-window aggregation by true duration bin](multi_window_aggregation_by_true_duration_bin_metrics.png)\n\n"
            "## Multi-Window Class Accuracy By True Duration Bin\n\n"
            "![Multi-window class accuracy by true duration bin](multi_window_aggregation_by_true_duration_bin_class_accuracy.png)\n\n"
            "## Full-Length Class Accuracy By Bin\n",
        )

    summary_columns = [
        "dataset_type",
        "random_start",
        "aggregation",
        "samples",
        "eer_percent",
        "mdr_at_far1_percent",
        "f1_at_eer_threshold_percent",
        "best_f1_percent",
        "threshold_free_accuracy_percent",
        "threshold_free_bonafide_accuracy_percent",
        "threshold_free_spoof_accuracy_percent",
        "auc_percent",
    ]
    best_columns = [
        "dataset_type",
        "random_start",
        "aggregation",
        "eer_percent",
        "mdr_at_far1_percent",
        "f1_at_eer_threshold_percent",
        "threshold_free_accuracy_percent",
        "threshold_free_bonafide_accuracy_percent",
        "threshold_free_spoof_accuracy_percent",
    ]
    summary = ordered_summary(summary)
    by_bin = ordered_by_bin(by_bin)
    best_eer = ordered_summary(
        summary.sort_values("eer_percent").groupby("dataset_type", as_index=False, observed=True).head(1)
    )
    best_mdr = ordered_summary(
        summary.sort_values("mdr_at_far1_percent").groupby("dataset_type", as_index=False, observed=True).head(1)
    )
    best_tf = ordered_summary(
        summary.sort_values("threshold_free_accuracy_percent", ascending=False)
        .groupby("dataset_type", as_index=False, observed=True)
        .head(1)
    )
    by_bin_columns = [
        "dataset_type",
        "random_start",
        "true_duration_bin",
        "aggregation",
        "samples",
        "eer_percent",
        "mdr_at_far1_percent",
        "f1_at_eer_threshold_percent",
        "threshold_free_accuracy_percent",
        "threshold_free_bonafide_accuracy_percent",
        "threshold_free_spoof_accuracy_percent",
        "auc_percent",
    ]
    by_bin_best_columns = [
        "dataset_type",
        "random_start",
        "true_duration_bin",
        "aggregation",
        "samples",
        "eer_percent",
        "mdr_at_far1_percent",
        "f1_at_eer_threshold_percent",
        "threshold_free_accuracy_percent",
        "threshold_free_bonafide_accuracy_percent",
        "threshold_free_spoof_accuracy_percent",
    ]
    best_by_bin_eer = ordered_by_bin(
        by_bin.sort_values("eer_percent")
        .groupby(["dataset_type", "random_start", "true_duration_bin"], as_index=False, observed=True)
        .head(1)
    )

    section = (
        "\n## Multi-Window Aggregation Summary\n\n"
        + round_table(summary, summary_columns).to_markdown(index=False)
        + "\n\n## Multi-Window Aggregation By True Duration Bin\n\n"
        + round_table(by_bin, by_bin_columns).to_markdown(index=False)
        + "\n\n## Best Multi-Window By True Duration Bin EER\n\n"
        + round_table(best_by_bin_eer, by_bin_best_columns).to_markdown(index=False)
        + "\n\n## Best Multi-Window By EER\n\n"
        + round_table(best_eer, best_columns).to_markdown(index=False)
        + "\n\n## Best Multi-Window By MDR @ FAR=1%\n\n"
        + round_table(best_mdr, best_columns).to_markdown(index=False)
        + "\n\n## Best Multi-Window By Threshold-Free Accuracy\n\n"
        + round_table(best_tf, best_columns).to_markdown(index=False)
        + "\n"
    )

    marker = "\n## Conclusion\n"
    if "## Multi-Window Aggregation Summary" in report:
        before = report.split("\n## Multi-Window Aggregation Summary\n", 1)[0]
        after = report.split(marker, 1)[1]
        report = before + section + marker + after
    else:
        report = report.replace(marker, section + marker)

    observed_best = best_eer[best_eer["dataset_type"].astype(str) == "observed_short"].iloc[0]
    vad_best = best_eer[best_eer["dataset_type"].astype(str) == "vad_short"].iloc[0]
    short_bin_best = best_by_bin_eer[
        (best_by_bin_eer["dataset_type"].astype(str) == "observed_short")
        & (best_by_bin_eer["true_duration_bin"].astype(str) == "0.5-1.0s")
    ].sort_values("eer_percent").iloc[0]
    vad_bin_best = best_by_bin_eer[
        (best_by_bin_eer["dataset_type"].astype(str) == "vad_short")
        & (best_by_bin_eer["true_duration_bin"].astype(str) == "1.5-2.0s")
    ].sort_values("eer_percent").iloc[0]
    conclusion_addition = (
        "- Multi-window aggregation: vad_short được lợi rõ nhất với `mean`, `random_start=true`: "
        f"EER {vad_best['eer_percent']:.2f}%, MDR@FAR1 {vad_best['mdr_at_far1_percent']:.2f}%, "
        f"F1@EER {vad_best['f1_at_eer_threshold_percent']:.2f}%. "
        "observed_short không beat best fixed 2.0s; best multi-window là "
        f"`{observed_best['aggregation']}`, `random_start={str(observed_best['random_start']).lower()}` "
        f"với EER {observed_best['eer_percent']:.2f}%.\n"
        "- Multi-window mean/median ổn định hơn cho EER/MDR; `min/max` có thể kéo class-specific accuracy nhưng thường làm EER/MDR kém hơn.\n"
        "- Theo true duration bin: observed_short bin 0.5-1.0s vẫn là nhóm khó nhất; best EER là "
        f"{short_bin_best['eer_percent']:.2f}% với `{short_bin_best['aggregation']}`, "
        f"`random_start={str(short_bin_best['random_start']).lower()}`. "
        "Với vad_short bin 1.5-2.0s, best EER là "
        f"{vad_bin_best['eer_percent']:.2f}% với `{vad_bin_best['aggregation']}`, "
        f"`random_start={str(vad_bin_best['random_start']).lower()}`.\n"
    )
    if "- Multi-window aggregation:" in report:
        lines = []
        skip_next = False
        for line in report.splitlines():
            if line.startswith("- Multi-window aggregation:"):
                lines.extend(conclusion_addition.rstrip("\n").splitlines())
                skip_next = True
                continue
            if skip_next and line.startswith("- Multi-window mean/median"):
                continue
            if skip_next and line.startswith("- Theo true duration bin:"):
                continue
            skip_next = False
            lines.append(line)
        report = "\n".join(lines) + "\n"
    else:
        report = report.replace("## Conclusion\n\n", "## Conclusion\n\n" + conclusion_addition)

    report_path.write_text(report)


def main() -> None:
    rows = []
    by_bin_rows = []
    index_rows = []
    for dataset in DATASETS:
        for random_start in RANDOM_STARTS:
            for method in AGGREGATIONS:
                frame = aggregate_frame(dataset, random_start, method)
                random_text = str(random_start).lower()
                score_file = AGG_SCORE_DIR / f"{dataset}_random_start_{random_text}_multi_window_{method}_{MODEL_SUFFIX}.txt"
                write_score_file(frame, score_file)
                rows.append(metric_row(frame, dataset, random_start, method, score_file))
                for true_duration_bin in TRUE_DURATION_BINS:
                    bin_frame = frame[frame["true_duration_bin"] == true_duration_bin].copy()
                    by_bin_rows.append(
                        metric_row(
                            bin_frame,
                            dataset,
                            random_start,
                            method,
                            score_file,
                            eval_type="multi_window_aggregation_by_true_duration_bin",
                            extra={
                                "true_duration_bin": true_duration_bin,
                                "true_duration_sec_min": float(bin_frame["true_duration_sec"].min()),
                                "true_duration_sec_max": float(bin_frame["true_duration_sec"].max()),
                                "true_duration_sec_mean": float(bin_frame["true_duration_sec"].mean()),
                                "true_duration_source": "total_speech_duration_sec"
                                if dataset == "observed_short"
                                else "manifest_duration_bin",
                            },
                        )
                    )
                index_rows.append(
                    {
                        "dataset_type": dataset,
                        "random_start": random_start,
                        "aggregation": method,
                        "windows": "+".join(DURATIONS),
                        "score_file": repo_rel(score_file),
                        "lines": len(frame),
                    }
                )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    summary = ordered_summary(pd.DataFrame(rows))
    by_bin_summary = ordered_by_bin(pd.DataFrame(by_bin_rows))
    summary.to_csv(REPORT_DIR / "multi_window_aggregation_metrics.csv", index=False)
    by_bin_summary.to_csv(REPORT_DIR / "multi_window_aggregation_by_true_duration_bin_metrics.csv", index=False)
    pd.DataFrame(index_rows).to_csv(REPORT_DIR / "multi_window_aggregation_score_file_index.csv", index=False)
    plot_multi_window_metrics(summary, REPORT_DIR / "multi_window_aggregation_metrics.png")
    plot_class_accuracy(summary, REPORT_DIR / "multi_window_class_accuracy.png")
    plot_by_true_duration_bin_metrics(
        by_bin_summary, REPORT_DIR / "multi_window_aggregation_by_true_duration_bin_metrics.png"
    )
    plot_by_true_duration_bin_class_accuracy(
        by_bin_summary, REPORT_DIR / "multi_window_aggregation_by_true_duration_bin_class_accuracy.png"
    )
    update_report(summary, by_bin_summary)
    print(f"Wrote {len(summary)} multi-window rows to {REPORT_DIR / 'multi_window_aggregation_metrics.csv'}")
    print(
        f"Wrote {len(by_bin_summary)} true-duration-bin rows to "
        f"{REPORT_DIR / 'multi_window_aggregation_by_true_duration_bin_metrics.csv'}"
    )
    print(f"Wrote aggregated score files to {AGG_SCORE_DIR}")


if __name__ == "__main__":
    main()
