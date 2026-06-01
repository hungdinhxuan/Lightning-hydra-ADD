#!/usr/bin/env python3
"""Build observed_short true-duration vs inference-length comparison tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]

# (display bin, inference mode label, manifest bin key, eval_type, inference_duration or None for full)
ROWS = [
    ("0.5–1s", "full-length", "observed_short_0p5_1p0", "full_length_by_bin", "full"),
    ("0.5–1s", "pad/crop to 1.0s", "observed_short_0p5_1p0", "fixed_length_by_bin", "1.0s"),
    ("0.5–1s", "pad to 1.5s", "observed_short_0p5_1p0", "fixed_length_by_bin", "1.5s"),
    ("0.5–1s", "pad to 2.0s", "observed_short_0p5_1p0", "fixed_length_by_bin", "2.0s"),
    ("1–1.5s", "full-length", "observed_short_1p0_1p5", "full_length_by_bin", "full"),
    ("1–1.5s", "pad/crop to 1.5s", "observed_short_1p0_1p5", "fixed_length_by_bin", "1.5s"),
    ("1–1.5s", "pad to 2.0s", "observed_short_1p0_1p5", "fixed_length_by_bin", "2.0s"),
    ("1.5–2s", "full-length", "observed_short_1p5_2p0", "full_length_by_bin", "full"),
    ("1.5–2s", "pad/crop to 2.0s", "observed_short_1p5_2p0", "fixed_length_by_bin", "2.0s"),
]

METRIC_COLS = {
    "eer_percent": "EER",
    "mdr_at_far1_percent": "MDR@FAR1",
    "auc_percent": "AUC",
    "threshold_free_accuracy_percent": "Acc",
}


def load_summary(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = frame[frame["dataset_type"].eq("observed_short")].copy()
    frame["random_start"] = frame["random_start"].map(
        lambda value: False if pd.isna(value) or value == "" else bool(value) if isinstance(value, bool) else str(value).lower() == "true"
    )
    return frame


def lookup(frame: pd.DataFrame, eval_type: str, duration_bin: str, inference_duration: str) -> pd.Series | None:
    subset = frame[
        frame["eval_type"].eq(eval_type)
        & frame["duration_bin_eval"].eq(duration_bin)
        & frame["inference_duration"].astype(str).eq(inference_duration)
    ]
    if eval_type == "fixed_length_by_bin":
        subset = subset[subset["random_start"].eq(False)]
    if subset.empty:
        return None
    if len(subset) > 1:
        raise RuntimeError(f"Ambiguous row for {eval_type} {duration_bin} {inference_duration}")
    return subset.iloc[0]


def build_table(baseline: pd.DataFrame, lora: pd.DataFrame, lora_label: str) -> pd.DataFrame:
    out_rows = []
    for true_bin_label, mode_label, manifest_bin, eval_type, inference_duration in ROWS:
        base = lookup(baseline, eval_type, manifest_bin, inference_duration)
        lor = lookup(lora, eval_type, manifest_bin, inference_duration)
        row = {
            "true_duration_bin": true_bin_label,
            "inference_mode": mode_label,
            "samples": int(lor["samples"]) if lor is not None else (int(base["samples"]) if base is not None else pd.NA),
        }
        for metric, short in METRIC_COLS.items():
            bval = base[metric] if base is not None else pd.NA
            lval = lor[metric] if lor is not None else pd.NA
            row[f"baseline_{short}"] = round(float(bval), 2) if pd.notna(bval) else pd.NA
            row[f"{lora_label}_{short}"] = round(float(lval), 2) if pd.notna(lval) else pd.NA
            if pd.notna(bval) and pd.notna(lval):
                if metric == "eer_percent" or metric == "mdr_at_far1_percent":
                    row[f"delta_{short}"] = round(float(bval) - float(lval), 2)  # positive = LoRA better
                else:
                    row[f"delta_{short}"] = round(float(lval) - float(bval), 2)
            else:
                row[f"delta_{short}"] = pd.NA
        out_rows.append(row)
    return pd.DataFrame(out_rows)


def to_markdown(table: pd.DataFrame, lora_label: str) -> str:
    lines = [
        "# Observed-short: true duration vs inference length (Baseline vs LoRA)",
        "",
        "Metrics are filtered from existing inference (`random_start=false` for fixed-length pad/crop).",
        "True-duration bins use `total_speech_duration_sec` (same as multi-window report).",
        "",
        f"LoRA model: **{lora_label}**",
        "",
        "## Combined metrics",
        "",
        "| True duration bin | Inference mode | n | "
        + " | ".join(f"Baseline {m}" for m in METRIC_COLS.values())
        + " | "
        + " | ".join(f"LoRA {m}" for m in METRIC_COLS.values())
        + " | Δ EER (B−L) |",
        "| --- | --- | ---: | " + " | ".join(["---:"] * (len(METRIC_COLS) * 2 + 1)),
    ]
    for _, row in table.iterrows():
        lines.append(
            f"| {row['true_duration_bin']} | {row['inference_mode']} | {row['samples']} | "
            + " | ".join(str(row[f"baseline_{METRIC_COLS[m]}"]) for m in METRIC_COLS)
            + " | "
            + " | ".join(str(row[f"{lora_label}_{METRIC_COLS[m]}"]) for m in METRIC_COLS)
            + f" | {row['delta_EER']} |"
        )

    lines.extend(
        [
            "",
            "## Reading guide (padding 0.5–1s → 2s)",
            "",
            "For bin **0.5–1s**, compare full-length EER to pad-to-2.0s EER on the same row group:",
            "- If **pad to longer inference window improves EER/AUC** vs full-length, padding helps ranking quality.",
            "- If **Acc (threshold-free)** drops while EER improves, the model shifts score balance (see class-accuracy heatmaps).",
            "",
            "Δ EER = baseline − LoRA (positive → LoRA lower EER).",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline-summary",
        type=Path,
        default=REPO_ROOT / "reports/baseline_mdt_protocol_full_fixed_trend_report/threshold_free_class_accuracy_summary.csv",
    )
    parser.add_argument(
        "--lora-summary",
        type=Path,
        default=REPO_ROOT / "reports/conf2_ep008_mdt_protocol_full_fixed_trend_report/threshold_free_class_accuracy_summary.csv",
    )
    parser.add_argument("--lora-label", default="conf2_ep008")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "reports/conf2_ep008_observed_short_padding_comparison",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    baseline = load_summary(args.baseline_summary)
    lora = load_summary(args.lora_summary)
    table = build_table(baseline, lora, args.lora_label)
    table.to_csv(args.out_dir / "observed_short_padding_comparison.csv", index=False)
    (args.out_dir / "report.md").write_text(to_markdown(table, args.lora_label), encoding="utf-8")
    print(f"Wrote {args.out_dir / 'observed_short_padding_comparison.csv'} ({len(table)} rows)")


if __name__ == "__main__":
    main()
