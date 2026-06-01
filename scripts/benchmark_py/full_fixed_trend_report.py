#!/usr/bin/env python3
"""Build full-length + fixed-length trend report from benchmark score files."""

from __future__ import annotations

import argparse
import math
import sys
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

from binary_eval import (  # noqa: E402
    compute_auc,
    compute_eer,
    compute_score_comparison_metrics,
    compute_threshold_metrics,
    find_threshold_at_target_far,
)


DURATIONS = ("0.5s", "1.0s", "1.5s", "2.0s")
DATASETS = ("observed_short", "vad_short")
OBSERVED_TRUE_DURATION_BINS = (
    "observed_short_0p5_1p0",
    "observed_short_1p0_1p5",
    "observed_short_1p5_2p0",
)


def observed_speech_duration_to_bin(value: float) -> str:
    if not np.isfinite(value):
        return ""
    if 0.5 <= value <= 1.0:
        return OBSERVED_TRUE_DURATION_BINS[0]
    if 1.0 < value <= 1.5:
        return OBSERVED_TRUE_DURATION_BINS[1]
    if 1.5 < value <= 2.0:
        return OBSERVED_TRUE_DURATION_BINS[2]
    return ""


def attach_observed_true_duration_bin(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "total_speech_duration_sec" not in out.columns:
        out["true_duration_bin"] = ""
        return out
    observed_mask = out["dataset_type"].eq("observed_short") | out["is_observed_short"].fillna(False).astype(bool)
    out["true_duration_bin"] = ""
    out.loc[observed_mask, "true_duration_bin"] = out.loc[observed_mask, "total_speech_duration_sec"].astype(float).map(
        observed_speech_duration_to_bin
    )
    missing = observed_mask & out["true_duration_bin"].eq("")
    if missing.any():
        examples = out.loc[missing, ["eval_rel_path", "total_speech_duration_sec"]].head(5)
        raise RuntimeError(f"Could not assign observed_short true duration bin for {int(missing.sum())} rows:\n{examples}")
    return out


def safe_percent(value: float) -> float:
    return float(value * 100.0) if np.isfinite(value) else math.nan


def repo_rel(path: Path) -> str:
    path = path.resolve()
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def metric_row(frame: pd.DataFrame, **context: Any) -> Dict[str, Any]:
    labels = frame["label"].map(lambda value: 1 if str(value).lower() == "bonafide" else 0).to_numpy(dtype=int)
    scores = frame["score"].to_numpy(dtype=float)
    eer, eer_threshold = compute_eer(labels, scores)
    auc = compute_auc(labels, scores)
    eer_metrics = compute_threshold_metrics(labels, scores, eer_threshold)
    far1 = find_threshold_at_target_far(labels, scores, target_far=0.01)
    far1_metrics = compute_threshold_metrics(labels, scores, far1["threshold"])
    threshold_frame = frame[["label", "score", "spoof_score"]].copy()
    threshold_frame["label"] = threshold_frame["label"].map(lambda value: 1 if str(value).lower() == "bonafide" else 0)
    threshold_free = compute_score_comparison_metrics(threshold_frame)
    return {
        **context,
        "samples": int(len(frame)),
        "eer": eer,
        "eer_percent": safe_percent(eer),
        "auc": auc,
        "auc_percent": safe_percent(auc),
        "eer_threshold": eer_threshold,
        "mdr_at_far1": far1_metrics["mdr"],
        "mdr_at_far1_percent": safe_percent(far1_metrics["mdr"]),
        "f1_at_eer_threshold": eer_metrics["f1"],
        "f1_at_eer_threshold_percent": safe_percent(eer_metrics["f1"]),
        "threshold_free_accuracy": threshold_free["accuracy"],
        "threshold_free_accuracy_percent": safe_percent(threshold_free["accuracy"]),
        "threshold_free_bonafide_accuracy_percent": safe_percent(threshold_free["bonafide_accuracy"]),
        "threshold_free_spoof_accuracy_percent": safe_percent(threshold_free["spoof_accuracy"]),
    }


def read_scores(path: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            parts = raw.strip().split()
            if len(parts) < 3:
                continue
            rows.append(
                {
                    "eval_rel_path": parts[0],
                    "spoof_score": float(parts[1]),
                    "score": float(parts[2]),
                }
            )
    return pd.DataFrame(rows).drop_duplicates(subset="eval_rel_path", keep="last")


def read_metadata(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    keep = [
        "eval_rel_path",
        "dataset_type",
        "dataset",
        "duration_bin",
        "label",
        "duration",
        "speech_duration",
        "total_speech_duration_sec",
        "is_observed_short",
        "is_vad_segment",
    ]
    frame = frame[keep].copy()
    if frame["dataset_type"].astype(str).eq("fixed_length").all():
        observed_mask = frame["is_observed_short"].fillna(False).astype(bool)
        frame["dataset_type"] = np.where(observed_mask, "observed_short", "vad_short")
    return frame


def load_fixed_frame(score_path: Path, metadata: pd.DataFrame) -> pd.DataFrame:
    return attach_observed_true_duration_bin(metadata.merge(read_scores(score_path), on="eval_rel_path", how="inner"))


def load_full_frame(score_path: Path, metadata: pd.DataFrame) -> pd.DataFrame:
    return attach_observed_true_duration_bin(metadata.merge(read_scores(score_path), on="eval_rel_path", how="inner"))


def fixed_run_comment(prefix: str, suffix: str, duration: str, pattern: str) -> str:
    return pattern.format(prefix=prefix, suffix=suffix, duration=duration)


def fixed_score_path(
    results_root: Path,
    prefix: str,
    suffix: str,
    duration: str,
    config_name: str,
    pattern: str,
) -> Path:
    comment = fixed_run_comment(prefix, suffix, duration, pattern)
    return results_root / comment / f"fixed_length_test_{config_name}_{comment}.txt"


def line_count(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def write_score_file(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in frame.itertuples(index=False):
            handle.write(f"{row.eval_rel_path} {row.spoof_score} {row.score}\n")


def round_table(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    out = df.loc[:, list(columns)].copy()
    for column in out.columns:
        if column.endswith("_percent"):
            out[column] = out[column].map(lambda value: round(float(value), 2) if pd.notna(value) else value)
    return out


def normalize_random_start_column(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "random_start" not in out.columns:
        return out

    def _normalize(value: object) -> object:
        if pd.isna(value):
            return ""
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if isinstance(value, (int, np.integer)):
            return bool(value)
        if isinstance(value, (float, np.floating)) and float(value).is_integer():
            return bool(int(value))
        text = str(value).strip()
        if text in {"0", "0.0", "False", "false"}:
            return False
        if text in {"1", "1.0", "True", "true"}:
            return True
        return text

    out["random_start"] = out["random_start"].map(_normalize)
    return out


def plot_eer_trend(report_dir: Path, fixed: pd.DataFrame, full: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for ax, dataset_type in zip(axes, DATASETS):
        fixed_subset = fixed[fixed["dataset_type"] == dataset_type].copy()
        full_subset = full[full["dataset_type"] == dataset_type].copy()
        for random_start, label in ((False, "random_start=false"), (True, "random_start=true")):
            line = fixed_subset[fixed_subset["random_start"] == random_start].sort_values("duration_sec")
            ax.plot(line["duration_sec"], line["eer_percent"], marker="o", label=label)
        if not full_subset.empty:
            ax.axhline(full_subset["eer_percent"].iloc[0], linestyle="--", color="#333333", label="full")
        ax.set_title(dataset_type)
        ax.set_xlabel("duration (s)")
        ax.set_ylabel("EER (%)")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle("Fixed-length vs full-length EER by dataset")
    fig.savefig(report_dir / "eer_trend_fixed_vs_full_by_dataset.png", dpi=160)
    plt.close(fig)


def plot_auc_accuracy(report_dir: Path, pooled: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for random_start, marker, color in ((False, "o", "#1f77b4"), (True, "s", "#d62728")):
        line = pooled[pooled["random_start"] == random_start].sort_values("duration_sec")
        axes[0].plot(line["duration_sec"], line["auc_percent"], marker=marker, color=color, label=f"random_start={str(random_start).lower()}")
        axes[1].plot(
            line["duration_sec"],
            line["threshold_free_accuracy_percent"],
            marker=marker,
            color=color,
            label=f"random_start={str(random_start).lower()}",
        )
    axes[0].set_title("AUC trend")
    axes[1].set_title("Threshold-free accuracy trend")
    for ax in axes:
        ax.set_xlabel("duration (s)")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("AUC (%)")
    axes[1].set_ylabel("Accuracy (%)")
    fig.savefig(report_dir / "auc_accuracy_trend.png", dpi=160)
    plt.close(fig)


def plot_full_by_bin(report_dir: Path, by_bin: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    for dataset_type, group in by_bin.groupby("dataset_type"):
        ordered = group.copy()
        ordered["sort_key"] = ordered["duration_bin_eval"].astype(str)
        ordered = ordered.sort_values("sort_key")
        ax.plot(ordered["duration_bin_eval"], ordered["eer_percent"], marker="o", label=dataset_type)
    ax.set_title("Full-length EER by duration bin")
    ax.set_xlabel("duration bin")
    ax.set_ylabel("EER (%)")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.savefig(report_dir / "full_length_eer_by_bin.png", dpi=160)
    plt.close(fig)


def plot_random_start_delta(report_dir: Path, delta: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    for dataset_type, group in delta.groupby("dataset_type"):
        ordered = group.sort_values("duration_sec")
        ax.plot(ordered["duration_sec"], ordered["eer_delta_random_minus_norandom"], marker="o", label=dataset_type)
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_title("Random-start EER delta")
    ax.set_xlabel("duration (s)")
    ax.set_ylabel("random - norandom EER (points)")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.savefig(report_dir / "random_start_delta.png", dpi=160)
    plt.close(fig)


def plot_threshold_free_trend(report_dir: Path, summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    fixed = summary[summary["eval_type"] == "fixed_length"].copy()
    full = summary[summary["eval_type"].isin(["full_length_overall", "full_length_by_bin"])].copy()
    for ax, metric, title in (
        (axes[0], "threshold_free_bonafide_accuracy_percent", "Bonafide threshold-free accuracy"),
        (axes[1], "threshold_free_spoof_accuracy_percent", "Spoof threshold-free accuracy"),
    ):
        for dataset_type in DATASETS:
            subset = fixed[fixed["dataset_type"] == dataset_type]
            for random_start in (False, True):
                line = subset[subset["random_start"] == random_start].sort_values("duration_sec")
                ax.plot(
                    line["duration_sec"],
                    line[metric],
                    marker="o",
                    label=f"{dataset_type}, rs={str(random_start).lower()}",
                )
        for dataset_type in DATASETS:
            overall = full[(full["dataset_type"] == dataset_type) & (full["eval_type"] == "full_length_overall")]
            if not overall.empty:
                ax.axhline(overall[metric].iloc[0], linestyle="--", alpha=0.5)
        ax.set_title(title)
        ax.set_xlabel("duration (s)")
        ax.set_ylabel("%")
        ax.grid(True, alpha=0.25)
    axes[0].legend(fontsize=7, ncols=2)
    fig.savefig(report_dir / "threshold_free_class_accuracy_trend.png", dpi=160)
    plt.close(fig)


def plot_score_distribution(report_dir: Path, full_frames: Dict[str, pd.DataFrame], best_fixed_frames: Dict[str, pd.DataFrame]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for row_idx, dataset_type in enumerate(DATASETS):
        full_frame = full_frames[dataset_type]
        fixed_frame = best_fixed_frames[dataset_type]
        for col_idx, label in enumerate(("bonafide", "spoof")):
            ax = axes[row_idx, col_idx]
            full_scores = full_frame.loc[full_frame["label"] == label, "score"]
            fixed_scores = fixed_frame.loc[fixed_frame["label"] == label, "score"]
            ax.hist(full_scores, bins=80, alpha=0.5, label="full")
            ax.hist(fixed_scores, bins=80, alpha=0.5, label="best_fixed")
            ax.set_title(f"{dataset_type} / {label}")
            ax.set_xlabel("score")
            ax.legend(fontsize=8)
    fig.savefig(report_dir / "score_distribution_full_vs_best_fixed.png", dpi=160)
    plt.close(fig)


def build_report(args: argparse.Namespace) -> None:
    args.results_root = args.results_root.resolve()
    args.work_dir = args.work_dir.resolve()
    args.report_dir = args.report_dir.resolve()
    report_dir = args.report_dir
    report_dir.mkdir(parents=True, exist_ok=True)
    dataset_matrix_dir = args.dataset_matrix_dir.resolve() if args.dataset_matrix_dir else None
    if dataset_matrix_dir:
        dataset_matrix_dir.mkdir(parents=True, exist_ok=True)

    fixed_metadata = read_metadata(args.work_dir / "fixed_length" / "fixed_length_test" / "metadata.csv")
    observed_metadata = read_metadata(args.work_dir / "full_length" / "observed_short_full" / "metadata.csv")
    vad_metadata = read_metadata(args.work_dir / "full_length" / "vad_short_full" / "metadata.csv")

    full_rows: List[Dict[str, Any]] = []
    full_bin_rows: List[Dict[str, Any]] = []
    fixed_rows: List[Dict[str, Any]] = []
    fixed_pooled_rows: List[Dict[str, Any]] = []
    fixed_bin_rows: List[Dict[str, Any]] = []
    line_rows: List[Dict[str, Any]] = []
    full_frames: Dict[str, pd.DataFrame] = {}

    full_specs = {
        "observed_short": (
            args.results_root / f"{args.prefix}_full" / f"observed_short_full_{args.config_name}_{args.prefix}_full.txt",
            observed_metadata,
        ),
        "vad_short": (
            args.results_root / f"{args.prefix}_full" / f"vad_short_full_{args.config_name}_{args.prefix}_full.txt",
            vad_metadata,
        ),
    }

    for dataset_type, (score_path, metadata) in full_specs.items():
        frame = load_full_frame(score_path, metadata)
        full_frames[dataset_type] = frame
        full_rows.append(
            metric_row(
                frame,
                eval_type="full_length_overall",
                dataset_type=dataset_type,
                duration_bin_eval="overall",
                random_start=np.nan,
                inference_duration="full",
                duration_sec=np.nan,
                score_file=repo_rel(score_path),
            )
        )
        line_rows.append(
            {
                "run": "full",
                "dataset_type": dataset_type,
                "random_start": np.nan,
                "inference_duration": "full",
                "score_file": repo_rel(score_path),
                "lines": line_count(score_path),
            }
        )
        if dataset_type == "observed_short":
            for true_bin, group in frame.groupby("true_duration_bin", dropna=False):
                if not true_bin:
                    continue
                full_bin_rows.append(
                    metric_row(
                        group,
                        eval_type="full_length_by_bin",
                        dataset_type=dataset_type,
                        duration_bin_eval=str(true_bin),
                        random_start=np.nan,
                        inference_duration="full",
                        duration_sec=np.nan,
                        score_file=repo_rel(score_path),
                    )
                )
        else:
            for duration_bin, group in frame.groupby("duration_bin", dropna=False):
                full_bin_rows.append(
                    metric_row(
                        group,
                        eval_type="full_length_by_bin",
                        dataset_type=dataset_type,
                        duration_bin_eval=str(duration_bin),
                        random_start=np.nan,
                        inference_duration="full",
                        duration_sec=np.nan,
                        score_file=repo_rel(score_path),
                    )
                )

    best_fixed_frames: Dict[str, pd.DataFrame] = {}
    for random_start, suffix in ((False, "norandom"), (True, "random")):
        fixed_results_root = args.results_root if not random_start else (args.fixed_random_results_root or args.results_root)
        for duration in DURATIONS:
            score_path = fixed_score_path(
                fixed_results_root,
                args.prefix,
                suffix,
                duration,
                args.config_name,
                args.fixed_comment_pattern,
            )
            frame = load_fixed_frame(score_path, fixed_metadata)
            line_rows.append(
                {
                    "run": "fixed",
                    "dataset_type": "fixed_length_test",
                    "random_start": random_start,
                    "inference_duration": duration,
                    "score_file": repo_rel(score_path),
                    "lines": line_count(score_path),
                }
            )
            fixed_pooled_rows.append(
                metric_row(
                    frame,
                    eval_type="fixed_length_pooled",
                    dataset_type="pooled",
                    duration_bin_eval=np.nan,
                    random_start=random_start,
                    inference_duration=duration,
                    duration_sec=float(duration.rstrip("s")),
                    score_file=repo_rel(score_path),
                )
            )
            for dataset_type, group in frame.groupby("dataset_type", dropna=False):
                if dataset_matrix_dir is not None:
                    split_path = dataset_matrix_dir / (
                        f"{dataset_type}_random_start_{str(random_start).lower()}_{duration}_{args.config_name}.txt"
                    )
                    write_score_file(group, split_path)
                row = metric_row(
                    group,
                    eval_type="fixed_length",
                    dataset_type=str(dataset_type),
                    duration_bin_eval=np.nan,
                    random_start=random_start,
                    inference_duration=duration,
                    duration_sec=float(duration.rstrip("s")),
                    score_file=repo_rel(score_path),
                )
                fixed_rows.append(row)
                bin_column = "true_duration_bin" if str(dataset_type) == "observed_short" else "duration_bin"
                for duration_bin, bin_group in group.groupby(bin_column, dropna=False):
                    if not duration_bin:
                        continue
                    fixed_bin_rows.append(
                        metric_row(
                            bin_group,
                            eval_type="fixed_length_by_bin",
                            dataset_type=str(dataset_type),
                            duration_bin_eval=str(duration_bin) if pd.notna(duration_bin) else np.nan,
                            random_start=random_start,
                            inference_duration=duration,
                            duration_sec=float(duration.rstrip("s")),
                            score_file=repo_rel(score_path),
                        )
                    )

    full_overall = pd.DataFrame(full_rows)
    full_by_bin = pd.DataFrame(full_bin_rows)
    fixed_split = pd.DataFrame(fixed_rows)
    fixed_pooled = pd.DataFrame(fixed_pooled_rows)
    fixed_by_bin = pd.DataFrame(fixed_bin_rows)

    combined = pd.concat([full_overall, full_by_bin, fixed_split, fixed_by_bin], ignore_index=True)
    best_fixed = fixed_split.sort_values("eer_percent").groupby("dataset_type", as_index=False).head(1).reset_index(drop=True)
    for dataset_type in DATASETS:
        best_row = best_fixed[best_fixed["dataset_type"] == dataset_type]
        if not best_row.empty:
            rs = bool(best_row["random_start"].iloc[0])
            duration = str(best_row["inference_duration"].iloc[0])
            suffix = "random" if rs else "norandom"
            score_path = fixed_score_path(
                args.results_root if not rs else (args.fixed_random_results_root or args.results_root),
                args.prefix,
                suffix,
                duration,
                args.config_name,
                args.fixed_comment_pattern,
            )
            best_fixed_frames[dataset_type] = load_fixed_frame(score_path, fixed_metadata)

    random_delta = (
        fixed_split.pivot_table(
            index=["dataset_type", "duration_sec", "inference_duration"],
            columns="random_start",
            values="eer_percent",
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )
    random_delta["eer_delta_random_minus_norandom"] = random_delta[True] - random_delta[False]
    random_delta = random_delta[["dataset_type", "duration_sec", "inference_duration", "eer_delta_random_minus_norandom"]]

    threshold_free_summary = combined[
        combined["eval_type"].isin(["fixed_length", "full_length_by_bin", "full_length_overall"])
        | (
            combined["eval_type"].eq("fixed_length_by_bin")
            & combined["dataset_type"].eq("observed_short")
        )
    ][
        [
            "eval_type",
            "dataset_type",
            "duration_bin_eval",
            "random_start",
            "inference_duration",
            "duration_sec",
            "samples",
            "threshold_free_accuracy_percent",
            "threshold_free_bonafide_accuracy_percent",
            "threshold_free_spoof_accuracy_percent",
            "eer_percent",
            "mdr_at_far1_percent",
            "f1_at_eer_threshold_percent",
            "auc_percent",
            "score_file",
        ]
    ].copy()

    full_overall = normalize_random_start_column(full_overall)
    full_by_bin = normalize_random_start_column(full_by_bin)
    fixed_split = normalize_random_start_column(fixed_split)
    fixed_pooled = normalize_random_start_column(fixed_pooled)
    fixed_by_bin = normalize_random_start_column(fixed_by_bin)
    combined = normalize_random_start_column(combined)
    best_fixed = normalize_random_start_column(best_fixed)
    threshold_free_summary = normalize_random_start_column(threshold_free_summary)

    line_counts = pd.DataFrame(line_rows)

    full_overall.to_csv(report_dir / "full_length_overall_summary.csv", index=False)
    full_by_bin.to_csv(report_dir / "full_length_by_bin_summary.csv", index=False)
    fixed_split.to_csv(report_dir / "fixed_length_16_scores_summary.csv", index=False)
    fixed_pooled.to_csv(report_dir / "fixed_length_8_pooled_summary.csv", index=False)
    fixed_by_bin.to_csv(report_dir / "fixed_length_by_bin_summary.csv", index=False)
    combined.to_csv(report_dir / "combined_metrics.csv", index=False)
    best_fixed.to_csv(report_dir / "best_fixed_by_dataset.csv", index=False)
    random_delta.to_csv(report_dir / "random_start_delta.csv", index=False)
    threshold_free_summary.to_csv(report_dir / "threshold_free_class_accuracy_summary.csv", index=False)
    line_counts.to_csv(report_dir / "score_file_line_counts.csv", index=False)

    plot_eer_trend(report_dir, fixed_split, full_overall)
    plot_auc_accuracy(report_dir, fixed_pooled)
    plot_full_by_bin(report_dir, full_by_bin)
    plot_random_start_delta(report_dir, random_delta)
    plot_threshold_free_trend(report_dir, threshold_free_summary)
    plot_score_distribution(report_dir, full_frames, best_fixed_frames)

    report_lines = [
        f"# {args.title}",
        "",
        "Generated from existing score files. No extra inference was run for this report.",
        "",
        "## Inputs",
        "",
        f"- Results root: `{repo_rel(args.results_root)}/`",
        f"- Work dir: `{repo_rel(args.work_dir)}/`",
        f"- Checkpoint: `{args.checkpoint}`" if args.checkpoint else "- Checkpoint: `n/a`",
        f"- Config: `{args.config_name}`",
        "",
        "## Sanity Check",
        "",
        f"- Full-length rows: {int(full_overall['samples'].sum()):,}",
        f"- Fixed matrix rows: {int(line_counts[line_counts['run'] == 'fixed']['lines'].sum()):,}",
        f"- Total scored inference rows: {int(line_counts['lines'].sum()):,}",
        "",
        "## Main EER Trend",
        "",
        "![EER trend fixed vs full](eer_trend_fixed_vs_full_by_dataset.png)",
        "",
        "## AUC / Accuracy Trend",
        "",
        "![AUC and accuracy trend](auc_accuracy_trend.png)",
        "",
        "## Full-Length By Bin",
        "",
        "![Full length EER by bin](full_length_eer_by_bin.png)",
        "",
        "## Random Start Delta",
        "",
        "Negative means `random_start=true` is better than `random_start=false`.",
        "",
        "![Random start delta](random_start_delta.png)",
        "",
        "## Threshold-Free Class Accuracy Trend",
        "",
        "![Threshold-Free class accuracy trend](threshold_free_class_accuracy_trend.png)",
        "",
        "## Score Distribution",
        "",
        "![Score distribution full vs best fixed](score_distribution_full_vs_best_fixed.png)",
        "",
        "## Full-Length Overall",
        "",
        round_table(
            full_overall,
            [
                "dataset_type",
                "samples",
                "eer_percent",
                "mdr_at_far1_percent",
                "f1_at_eer_threshold_percent",
                "threshold_free_accuracy_percent",
                "threshold_free_bonafide_accuracy_percent",
                "threshold_free_spoof_accuracy_percent",
                "auc_percent",
            ],
        ).to_markdown(index=False),
        "",
        "## Fixed-Length Pooled",
        "",
        round_table(
            fixed_pooled,
            [
                "random_start",
                "inference_duration",
                "samples",
                "eer_percent",
                "mdr_at_far1_percent",
                "f1_at_eer_threshold_percent",
                "threshold_free_accuracy_percent",
                "threshold_free_bonafide_accuracy_percent",
                "threshold_free_spoof_accuracy_percent",
                "auc_percent",
            ],
        ).to_markdown(index=False),
        "",
        "## Fixed-Length Split Summary",
        "",
        round_table(
            fixed_split,
            [
                "dataset_type",
                "random_start",
                "inference_duration",
                "samples",
                "eer_percent",
                "mdr_at_far1_percent",
                "f1_at_eer_threshold_percent",
                "threshold_free_accuracy_percent",
                "threshold_free_bonafide_accuracy_percent",
                "threshold_free_spoof_accuracy_percent",
                "auc_percent",
            ],
        ).to_markdown(index=False),
        "",
        "## Best Fixed-Length By EER",
        "",
        round_table(
            best_fixed,
            [
                "dataset_type",
                "random_start",
                "inference_duration",
                "samples",
                "eer_percent",
                "mdr_at_far1_percent",
                "f1_at_eer_threshold_percent",
                "threshold_free_accuracy_percent",
                "threshold_free_bonafide_accuracy_percent",
                "threshold_free_spoof_accuracy_percent",
                "auc_percent",
            ],
        ).to_markdown(index=False),
        "",
        "## Files",
        "",
        "- `combined_metrics.csv`",
        "- `fixed_length_16_scores_summary.csv`",
        "- `fixed_length_8_pooled_summary.csv`",
        "- `full_length_overall_summary.csv`",
        "- `full_length_by_bin_summary.csv`",
        "- `fixed_length_by_bin_summary.csv`",
        "- `threshold_free_class_accuracy_summary.csv`",
        "- `score_file_line_counts.csv`",
        "- `random_start_delta.csv`",
        "",
        "## Conclusion",
        "",
        f"- Best observed_short fixed row: `{best_fixed[best_fixed['dataset_type'] == 'observed_short']['inference_duration'].iloc[0]}` / `random_start={str(bool(best_fixed[best_fixed['dataset_type'] == 'observed_short']['random_start'].iloc[0])).lower()}` with EER {best_fixed[best_fixed['dataset_type'] == 'observed_short']['eer_percent'].iloc[0]:.2f}%.",
        f"- Best vad_short fixed row: `{best_fixed[best_fixed['dataset_type'] == 'vad_short']['inference_duration'].iloc[0]}` / `random_start={str(bool(best_fixed[best_fixed['dataset_type'] == 'vad_short']['random_start'].iloc[0])).lower()}` with EER {best_fixed[best_fixed['dataset_type'] == 'vad_short']['eer_percent'].iloc[0]:.2f}%.",
    ]
    (report_dir / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create full/fixed trend report from ultrashort benchmark outputs.")
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--dataset-matrix-dir", type=Path, default=None)
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--title", default="LoRA Ultra-Short MDT: full-length + fixed-length visualization report")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument(
        "--fixed-comment-pattern",
        default="{prefix}_{suffix}_{duration}",
        help="Comment/subdir name for fixed-length score folders (default: conf2/lora layout).",
    )
    parser.add_argument(
        "--fixed-random-results-root",
        type=Path,
        default=None,
        help="Optional separate results root for random_start=true fixed-length runs.",
    )
    return parser.parse_args()


def main() -> None:
    build_report(parse_args())


if __name__ == "__main__":
    main()
