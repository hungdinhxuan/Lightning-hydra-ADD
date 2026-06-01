#!/usr/bin/env python3
"""Short-duration benchmark orchestration for May 29, 2026 plan.

This wrapper keeps model inference in the existing benchmark.py path. It adds:
- filtered benchmark folders for Observed-short and VAD-short
- dataset validation summaries
- single-duration score enrichment
- multi-duration aggregation
- metric/report generation
"""

from __future__ import annotations

import argparse
import math
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from benchmark_py.binary_eval import compute_auc, compute_eer, normalize_label
from benchmark_py.protocol import parse_protocol_line
from benchmark_py.scores import parse_score_line


DEFAULT_SOURCE_ROOT = Path("/data/add_processed_vad_short_eval10k_per_dataset_train80k_dev10k_29May2026")
DEFAULT_RESULTS_DIR = Path("logs/results/baseline_mdt_protocol_test_eval")
DEFAULT_REPORT_DIR = Path("reports/baseline_mdt_protocol_test_eval")
DEFAULT_WORK_DIR = Path("data/protocol_test_eval_benchmark")
DEFAULT_MODEL_PATH = Path("/NAS1_pretrained_lab/May_21_2026_TCM_MDT.pt")
DEFAULT_DURATIONS = (0.5, 1.0, 1.5, 2.0)
SAMPLE_RATE = 16000
DEFAULT_TRACKS = ("fixed_length_test", "observed_short_full", "vad_short_full")


@dataclass(frozen=True)
class EvalSet:
    track: str
    subset: str
    split_root: Path
    protocol_path: Path
    manifest_path: Path

    @property
    def slug(self) -> str:
        return self.track

    @property
    def group(self) -> str:
        return "fixed_length" if self.track == "fixed_length_test" else "full_length"

    @property
    def dataset_type(self) -> str:
        if self.track == "observed_short_full":
            return "observed_short"
        if self.track == "vad_short_full":
            return "vad_short"
        return "fixed_length"


def duration_slug(seconds: float) -> str:
    return f"{seconds:.1f}s"


def duration_samples(seconds: float) -> int:
    return int(round(seconds * SAMPLE_RATE))


def is_observed_path(rel_path: str) -> bool:
    return "observed_short" in rel_path


def is_vad_path(rel_path: str) -> bool:
    return "/vad_" in f"/{rel_path}"


def dataset_filter(dataset_type: str):
    if dataset_type == "fixed_length":
        return lambda rel_path: True
    if dataset_type == "observed_short":
        return is_observed_path
    if dataset_type == "vad_short":
        return is_vad_path
    raise ValueError(f"Unsupported dataset_type: {dataset_type}")


def read_protocol(protocol_path: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    with protocol_path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            parsed = parse_protocol_line(raw)
            if parsed is None:
                continue
            rel_path, subset, label = parsed
            rows.append({"rel_path": rel_path, "subset": subset, "label": label})
    return pd.DataFrame(rows, columns=["rel_path", "subset", "label"])


def load_manifest(manifest_path: Path) -> pd.DataFrame:
    if manifest_path.suffix == ".parquet":
        frame = pd.read_parquet(manifest_path)
    elif manifest_path.suffix == ".csv":
        frame = pd.read_csv(manifest_path)
    else:
        raise ValueError(f"Unsupported manifest format: {manifest_path}")

    if "rel_path" not in frame.columns:
        raise ValueError(f"Manifest missing rel_path: {manifest_path}")
    return frame.drop_duplicates(subset="rel_path", keep="last").reset_index(drop=True)


def source_eval_rel_path(rel_path: str) -> str:
    stem = Path(str(rel_path)).stem
    return f"audio/original_utterance/{stem}__source.wav"


def iter_eval_sets(source_root: Path, subsets: Sequence[str], tracks: Sequence[str]) -> List[EvalSet]:
    eval_sets: List[EvalSet] = []
    for subset in subsets:
        split_root = source_root / subset
        protocol_path = split_root / "protocol.txt"
        manifest_path = split_root / "manifest.parquet"
        if not manifest_path.exists():
            manifest_path = split_root / "manifest.csv"
        for track in tracks:
            if track not in DEFAULT_TRACKS:
                raise ValueError(f"Unsupported track: {track}")
            eval_sets.append(EvalSet(track, subset, split_root, protocol_path, manifest_path))
    return eval_sets


def build_metadata(eval_set: EvalSet) -> pd.DataFrame:
    protocol = read_protocol(eval_set.protocol_path)
    keep = protocol["rel_path"].map(dataset_filter(eval_set.dataset_type))
    protocol = protocol.loc[keep].copy()
    protocol["dataset_type"] = eval_set.dataset_type
    protocol["track"] = eval_set.track

    manifest = load_manifest(eval_set.manifest_path)
    keep_cols = [
        col
        for col in (
            "rel_path",
            "source_dataset",
            "duration_bin",
            "segment_duration_sec",
            "total_speech_duration_sec",
            "speech_duration_sec",
            "num_samples",
            "sample_rate",
            "source_abs_path",
            "source_rel_path",
            "original_duration_sec",
            "is_observed_short",
            "is_vad_segment",
        )
        if col in manifest.columns
    ]
    metadata = protocol.merge(manifest[keep_cols], on="rel_path", how="left")
    metadata["dataset"] = metadata.get("source_dataset", pd.Series(dtype=object)).fillna("unknown")
    metadata["duration"] = pd.to_numeric(metadata.get("segment_duration_sec"), errors="coerce")
    if metadata["duration"].isna().all() and "num_samples" in metadata:
        metadata["duration"] = pd.to_numeric(metadata["num_samples"], errors="coerce") / SAMPLE_RATE
    speech_col = "total_speech_duration_sec" if "total_speech_duration_sec" in metadata else "speech_duration_sec"
    metadata["speech_duration"] = pd.to_numeric(metadata.get(speech_col), errors="coerce")
    if metadata["speech_duration"].isna().all():
        metadata["speech_duration"] = metadata["duration"]
    metadata["duration_bin"] = metadata.get("duration_bin", "").fillna("").astype(str)
    metadata["utt_id"] = metadata["rel_path"].map(lambda value: Path(str(value)).stem)
    if eval_set.track == "fixed_length_test":
        metadata["eval_rel_path"] = metadata["rel_path"].map(source_eval_rel_path)
        metadata["eval_abs_path"] = metadata["source_abs_path"]
        metadata["eval_audio_kind"] = "source_utterance"
    else:
        metadata["eval_rel_path"] = metadata["rel_path"]
        metadata["eval_abs_path"] = metadata["rel_path"].map(lambda rel: str(eval_set.split_root / str(rel)))
        metadata["eval_audio_kind"] = "processed_short_full"
    return metadata


def validate_dataset(args: argparse.Namespace) -> None:
    rows: List[Dict[str, Any]] = []
    detail_rows: List[pd.DataFrame] = []
    args.report_dir.mkdir(parents=True, exist_ok=True)

    for eval_set in iter_eval_sets(args.source_root, args.subsets, args.tracks):
        metadata = build_metadata(eval_set)
        check_paths = metadata["eval_abs_path"].map(lambda path: Path(str(path)).exists())
        metadata["path_exists"] = check_paths
        detail_rows.append(metadata)

        group_cols = ["track", "dataset_type", "dataset", "subset", "duration_bin", "label"]
        grouped = metadata.groupby(group_cols, dropna=False).size().reset_index(name="samples")
        grouped["missing_audio"] = metadata.loc[~metadata["path_exists"]].groupby(group_cols, dropna=False).size().reindex(
            pd.MultiIndex.from_frame(grouped[group_cols]), fill_value=0
        ).to_numpy()
        rows.extend(grouped.to_dict("records"))

    summary = pd.DataFrame(rows)
    details = pd.concat(detail_rows, ignore_index=True) if detail_rows else pd.DataFrame()
    summary_path = args.report_dir / "dataset_validation_summary.csv"
    details_path = args.report_dir / "dataset_validation_details.csv"
    summary.to_csv(summary_path, index=False)
    details.to_csv(details_path, index=False)
    print(f"Wrote {summary_path}")
    print(f"Wrote {details_path}")


def prepare_benchmark(args: argparse.Namespace) -> None:
    args.work_dir.mkdir(parents=True, exist_ok=True)
    for eval_set in iter_eval_sets(args.source_root, args.subsets, args.tracks):
        metadata = build_metadata(eval_set)
        if args.max_per_source is not None:
            sampled = [
                group.sample(n=min(len(group), args.max_per_source), random_state=args.seed)
                for _, group in metadata.groupby("dataset", dropna=False)
            ]
            metadata = pd.concat(sampled, ignore_index=True) if sampled else metadata.iloc[0:0].copy()
        target = args.work_dir / eval_set.group / eval_set.slug
        target.mkdir(parents=True, exist_ok=True)

        if eval_set.track == "fixed_length_test":
            source_dir = target / "audio" / "original_utterance"
            source_dir.mkdir(parents=True, exist_ok=True)
            for row in metadata.itertuples(index=False):
                link_path = target / row.eval_rel_path
                source_path = Path(str(row.eval_abs_path))
                if not source_path.exists():
                    raise FileNotFoundError(source_path)
                if link_path.exists() or link_path.is_symlink():
                    if link_path.resolve() != source_path.resolve():
                        raise RuntimeError(f"{link_path} exists but points elsewhere")
                else:
                    link_path.symlink_to(source_path)
        else:
            audio_link = target / "audio"
            source_audio = eval_set.split_root / "audio"
            if audio_link.exists() or audio_link.is_symlink():
                if audio_link.resolve() != source_audio.resolve():
                    raise RuntimeError(f"{audio_link} exists but points elsewhere")
            else:
                audio_link.symlink_to(source_audio, target_is_directory=True)

        protocol_out = target / "protocol.txt"
        with protocol_out.open("w", encoding="utf-8") as handle:
            for row in metadata.itertuples(index=False):
                handle.write(f"{row.eval_rel_path} {args.benchmark_subset} {row.label}\n")
        metadata.to_csv(target / "metadata.csv", index=False)
        print(f"Prepared {target} ({len(metadata)} rows)")


def run_single(args: argparse.Namespace) -> None:
    args.results_dir.mkdir(parents=True, exist_ok=True)
    benchmark_root = args.work_dir / args.run_group if args.run_group else args.work_dir
    runs: List[tuple[str, Optional[float]]] = []
    if args.full_utterance:
        runs.append(("full", None))
    else:
        runs.extend((duration_slug(duration), duration) for duration in args.durations)

    jobs = []
    for idx, (run_label, duration) in enumerate(runs):
        gpu = args.gpus[idx % len(args.gpus)]
        comment = f"{args.comment_prefix}_{run_label}"
        trim_length = 0 if duration is None else duration_samples(duration)
        extra_overrides = list(args.extra_overrides)
        if args.full_utterance and not any("data.args.no_pad" in override for override in extra_overrides):
            extra_overrides.append("++data.args.no_pad=true")
        cmd = [
            sys.executable,
            str(SCRIPT_DIR / "benchmark.py"),
            "-g",
            gpu,
            "-c",
            args.config,
            "-b",
            str(benchmark_root),
            "-m",
            str(args.model_path),
            "-r",
            str(args.results_dir),
            "-n",
            comment,
            "-l",
            str(args.is_ln).lower(),
            "-s",
            str(args.random_start).lower(),
            "-t",
            str(trim_length),
            "-z",
            str(args.batch_size),
            "--missing-protocol-label",
            "skip",
        ]
        if args.adapter_paths:
            cmd.extend(["-a", str(args.adapter_paths)])
        cmd.extend(extra_overrides)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        env.setdefault("OMP_NUM_THREADS", str(args.omp_threads))
        jobs.append((run_label, gpu, cmd, env))

    active: List[tuple[str, str, subprocess.Popen]] = []
    failures = []
    for run_label, gpu, cmd, env in jobs:
        print("Running:", shlex.join(cmd))
        if args.dry_run:
            continue
        if args.parallel:
            while len(active) >= max(1, len(args.gpus)):
                still_active = []
                for active_label, active_gpu, process in active:
                    returncode = process.poll()
                    if returncode is None:
                        still_active.append((active_label, active_gpu, process))
                    elif returncode != 0:
                        failures.append((active_label, active_gpu, returncode))
                active = still_active
                if len(active) >= max(1, len(args.gpus)):
                    time.sleep(10)
            active.append((run_label, gpu, subprocess.Popen(cmd, cwd=REPO_ROOT, env=env)))
        else:
            subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=True)

    for active_label, active_gpu, process in active:
        returncode = process.wait()
        if returncode != 0:
            failures.append((active_label, active_gpu, returncode))
    if failures:
        raise SystemExit(f"{len(failures)} benchmark job(s) failed: {failures}")


def score_path_for_comment(results_dir: Path, comment: str, eval_slug: str, config: str) -> Path:
    normalized = config.replace("/", "_")
    return results_dir / comment / f"{eval_slug}_{normalized}_{comment}.txt"


def score_path(results_dir: Path, comment_prefix: str, duration: float, eval_slug: str, config: str) -> Path:
    return score_path_for_comment(results_dir, f"{comment_prefix}_{duration_slug(duration)}", eval_slug, config)


def read_scores(score_file: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    with score_file.open("r", encoding="utf-8") as handle:
        for raw in handle:
            parsed = parse_score_line(raw)
            if parsed is None:
                continue
            rel_path, spoof_score, bonafide_score, _ = parsed
            rows.append(
                {
                    "rel_path": rel_path,
                    "spoof_score": float(spoof_score),
                    "score": float(bonafide_score),
                }
            )
    return pd.DataFrame(rows).drop_duplicates(subset="rel_path", keep="last")


def enrich_single(args: argparse.Namespace) -> None:
    score_root = args.report_dir / "scores" / "baseline_mdt"
    score_root.mkdir(parents=True, exist_ok=True)
    for eval_set in iter_eval_sets(args.source_root, args.subsets, args.tracks):
        metadata = build_metadata(eval_set)
        out_dir = score_root / eval_set.track
        out_dir.mkdir(parents=True, exist_ok=True)
        run_items: List[tuple[str, str, Path]] = []
        if eval_set.track == "fixed_length_test":
            for duration in args.durations:
                run_items.append(
                    (
                        "fixed_length",
                        duration_slug(duration),
                        score_path(args.results_dir, args.comment_prefix, duration, eval_set.slug, args.config),
                    )
                )
        else:
            comment = f"{args.comment_prefix}_full"
            run_items.append(("full_utterance", "full", score_path_for_comment(args.results_dir, comment, eval_set.slug, args.config)))

        for inference_mode, inference_duration, source in run_items:
            if not source.exists():
                print(f"Missing score file: {source}")
                continue
            scores = read_scores(source)
            scores = scores.rename(columns={"rel_path": "eval_rel_path"})
            frame = metadata.merge(scores, on="eval_rel_path", how="inner")
            frame["inference_mode"] = inference_mode
            frame["inference_duration"] = inference_duration
            frame["aggregation_method"] = ""
            frame["model_name"] = args.config
            frame["checkpoint"] = str(args.model_path)
            cols = [
                "track",
                "dataset_type",
                "utt_id",
                "rel_path",
                "eval_rel_path",
                "eval_audio_kind",
                "dataset",
                "subset",
                "label",
                "duration_bin",
                "duration",
                "speech_duration",
                "original_duration_sec",
                "inference_mode",
                "inference_duration",
                "aggregation_method",
                "score",
                "spoof_score",
                "model_name",
                "checkpoint",
            ]
            present_cols = [col for col in cols if col in frame.columns]
            output = out_dir / f"score_{eval_set.track}_{eval_set.subset}_{inference_duration}.csv"
            frame[present_cols].to_csv(output, index=False)
            print(f"Wrote {output}")


def aggregate_multi(args: argparse.Namespace) -> None:
    score_root = args.report_dir / "scores" / "baseline_mdt"
    fixed_tracks = [track for track in args.tracks if track == "fixed_length_test"]
    for eval_set in iter_eval_sets(args.source_root, args.subsets, fixed_tracks):
        frames = []
        for duration in args.durations:
            path = score_root / eval_set.track / f"score_{eval_set.track}_{eval_set.subset}_{duration_slug(duration)}.csv"
            if not path.exists():
                print(f"Missing enriched score file: {path}")
                continue
            frame = pd.read_csv(path)
            keep = frame[
                [
                    "track",
                    "dataset_type",
                    "utt_id",
                    "rel_path",
                    "eval_rel_path",
                    "eval_audio_kind",
                    "dataset",
                    "subset",
                    "label",
                    "duration_bin",
                    "duration",
                    "speech_duration",
                    "original_duration_sec",
                    "score",
                    "spoof_score",
                    "model_name",
                    "checkpoint",
                ]
            ].copy()
            keep = keep.rename(
                columns={
                    "score": f"score_{duration_slug(duration)}",
                    "spoof_score": f"fake_score_{duration_slug(duration)}",
                }
            )
            frames.append(keep)
        if not frames:
            continue
        raw = frames[0]
        keys = [
            "track",
            "dataset_type",
            "utt_id",
            "rel_path",
            "eval_rel_path",
            "eval_audio_kind",
            "dataset",
            "subset",
            "label",
            "duration_bin",
            "duration",
            "speech_duration",
            "original_duration_sec",
            "model_name",
            "checkpoint",
        ]
        for frame in frames[1:]:
            raw = raw.merge(frame, on=keys, how="inner")

        score_cols = [f"score_{duration_slug(duration)}" for duration in args.durations if f"score_{duration_slug(duration)}" in raw]
        fake_cols = [f"fake_score_{duration_slug(duration)}" for duration in args.durations if f"fake_score_{duration_slug(duration)}" in raw]
        raw_out = score_root / eval_set.track / f"score_{eval_set.track}_{eval_set.subset}_multi_raw.csv"
        raw[keys + score_cols].to_csv(raw_out, index=False)

        values = raw[score_cols].to_numpy(dtype=float)
        fake_values = raw[fake_cols].to_numpy(dtype=float) if fake_cols else -values
        aggregated_rows = []
        methods = {
            "mean": np.nanmean(values, axis=1),
            "median": np.nanmedian(values, axis=1),
            "max_fake": -np.nanmax(fake_values, axis=1),
            "top2_mean_fake": -np.nanmean(np.sort(fake_values, axis=1)[:, -min(2, fake_values.shape[1]) :], axis=1),
        }
        for method, final_scores in methods.items():
            temp = raw[keys].copy()
            temp["inference_mode"] = "fixed_length_multi"
            temp["inference_duration"] = "multi"
            temp["aggregation_method"] = method
            temp["final_score"] = final_scores
            aggregated_rows.append(temp)
        aggregated = pd.concat(aggregated_rows, ignore_index=True)
        agg_out = score_root / eval_set.track / f"score_{eval_set.track}_{eval_set.subset}_multi_aggregated.csv"
        aggregated[
            keys[:-2] + ["inference_mode", "inference_duration", "aggregation_method", "final_score", "model_name", "checkpoint"]
        ].to_csv(agg_out, index=False)
        print(f"Wrote {raw_out}")
        print(f"Wrote {agg_out}")


def duration_bin_from_value(value: float) -> str:
    if not np.isfinite(value):
        return "unknown"
    if value <= 0.5:
        return "0.0-0.5s"
    if value <= 1.0:
        return "0.5-1.0s"
    if value <= 1.5:
        return "1.0-1.5s"
    if value <= 2.0:
        return "1.5-2.0s"
    return "2.0s+"


def fpr_tpr_points(labels: Sequence[int], scores: Sequence[float]) -> Dict[str, float]:
    labels_array = np.asarray(labels, dtype=int)
    scores_array = np.asarray(scores, dtype=float)
    if labels_array.size == 0 or np.unique(labels_array).size < 2:
        return {"fpr_at_tpr95": math.nan, "tpr_at_fpr1": math.nan}
    fpr, tpr, _ = roc_curve(labels_array, scores_array, pos_label=1, drop_intermediate=False)
    tpr95 = np.flatnonzero(tpr >= 0.95)
    fpr_at_tpr95 = float(np.min(fpr[tpr95])) if tpr95.size else math.nan
    fpr1 = np.flatnonzero(fpr <= 0.01)
    tpr_at_fpr1 = float(np.max(tpr[fpr1])) if fpr1.size else math.nan
    return {"fpr_at_tpr95": fpr_at_tpr95, "tpr_at_fpr1": tpr_at_fpr1}


def metric_record(frame: pd.DataFrame, **context: Any) -> Dict[str, Any]:
    if frame.empty:
        return {**context, "samples": 0}
    labels = frame["label"].map(normalize_label).to_numpy(dtype=int)
    scores = frame["metric_score"].to_numpy(dtype=float)
    eer, threshold = compute_eer(labels, scores)
    points = fpr_tpr_points(labels, scores)
    bonafide_scores = frame.loc[frame["label"].map(normalize_label) == 1, "metric_score"]
    spoof_scores = frame.loc[frame["label"].map(normalize_label) == 0, "metric_score"]
    return {
        **context,
        "samples": int(len(frame)),
        "bonafide_samples": int((labels == 1).sum()),
        "spoof_samples": int((labels == 0).sum()),
        "eer": eer,
        "eer_threshold": threshold,
        "auc": compute_auc(labels, scores),
        "fpr_at_tpr95": points["fpr_at_tpr95"],
        "tpr_at_fpr1": points["tpr_at_fpr1"],
        "score_mean_bonafide": float(bonafide_scores.mean()) if not bonafide_scores.empty else math.nan,
        "score_mean_spoof": float(spoof_scores.mean()) if not spoof_scores.empty else math.nan,
        "score_std_bonafide": float(bonafide_scores.std()) if len(bonafide_scores) > 1 else math.nan,
        "score_std_spoof": float(spoof_scores.std()) if len(spoof_scores) > 1 else math.nan,
    }


def report_metrics(args: argparse.Namespace) -> None:
    score_root = args.report_dir / "scores" / "baseline_mdt"
    rows: List[Dict[str, Any]] = []
    frames: List[pd.DataFrame] = []

    for track_dir in sorted(score_root.glob("*")):
        if not track_dir.is_dir():
            continue
        for path in sorted(track_dir.glob("score_*_*.csv")):
            if path.name.endswith("_multi_raw.csv"):
                continue
            frame = pd.read_csv(path)
            if frame.empty:
                continue
            if "track" not in frame.columns:
                frame["track"] = track_dir.name
            if "dataset_type" not in frame.columns:
                frame["dataset_type"] = track_dir.name
            if "duration_bin" in frame.columns:
                frame["duration_bin_eval"] = frame["duration_bin"].fillna("").astype(str)
                missing_bin = frame["duration_bin_eval"].eq("") | frame["duration_bin_eval"].eq("nan")
                frame.loc[missing_bin, "duration_bin_eval"] = frame.loc[missing_bin, "duration"].map(duration_bin_from_value)
            else:
                frame["duration_bin_eval"] = frame["duration"].map(duration_bin_from_value)
            if "final_score" in frame.columns:
                frame["metric_score"] = frame["final_score"]
                frame["inference_mode"] = frame.get("inference_mode", "fixed_length_multi")
                frame["inference_duration"] = frame.get("inference_duration", "multi")
            elif "score" in frame.columns:
                if "inference_mode" not in frame.columns:
                    frame["inference_mode"] = "fixed_length"
                frame["aggregation_method"] = ""
                frame["metric_score"] = frame["score"]
            else:
                continue
            frames.append(frame)

    if not frames:
        raise SystemExit(f"No score CSV files found under {score_root}")

    all_scores = pd.concat(frames, ignore_index=True)
    group_sets = {
        "metrics_summary.csv": ["track", "dataset_type", "subset", "inference_mode", "inference_duration", "aggregation_method"],
        "metrics_by_dataset.csv": ["track", "dataset_type", "dataset", "subset", "inference_mode", "inference_duration", "aggregation_method"],
        "metrics_by_duration_bin.csv": ["track", "dataset_type", "duration_bin_eval", "subset", "inference_mode", "inference_duration", "aggregation_method"],
        "metrics_by_inference_duration.csv": ["track", "inference_duration", "dataset_type", "subset"],
        "metrics_by_inference_mode.csv": ["track", "inference_mode", "aggregation_method", "dataset_type", "subset"],
    }
    args.report_dir.mkdir(parents=True, exist_ok=True)
    for filename, group_cols in group_sets.items():
        records = [
            metric_record(group, **dict(zip(group_cols, key if isinstance(key, tuple) else (key,))))
            for key, group in all_scores.groupby(group_cols, dropna=False)
        ]
        out = args.report_dir / filename
        pd.DataFrame(records).sort_values(group_cols).to_csv(out, index=False)
        rows.extend(records if filename == "metrics_summary.csv" else [])
        print(f"Wrote {out}")

    write_markdown_report(args.report_dir, pd.DataFrame(rows))
    write_figures(args.report_dir, all_scores, pd.DataFrame(rows))


def write_markdown_report(report_dir: Path, summary: pd.DataFrame) -> None:
    out = report_dir / "report.md"
    with out.open("w", encoding="utf-8") as handle:
        handle.write("# Baseline MDT Protocol Test Evaluation\n\n")
        handle.write("Tracks: fixed-length source utterance crops, observed-short full utterance, and VAD-short full utterance.\n\n")
        handle.write("Score direction: higher `score`/`final_score` means more bonafide. `max_fake` methods invert fake evidence for metric compatibility.\n\n")
        if not summary.empty:
            best = summary.sort_values("eer", na_position="last").head(10)
            handle.write("## Best EER Rows\n\n")
            handle.write(best.to_markdown(index=False))
            handle.write("\n")
    print(f"Wrote {out}")


def write_figures(report_dir: Path, scores: pd.DataFrame, summary: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    fig_dir = report_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    for dataset_type, group in scores.groupby("dataset_type"):
        plt.figure(figsize=(8, 4))
        for label, label_group in group.groupby("label"):
            label_group["metric_score"].plot(kind="hist", bins=80, alpha=0.5, label=str(label))
        plt.title(f"Score distribution: {dataset_type}")
        plt.xlabel("score")
        plt.legend()
        out = fig_dir / f"score_distribution_{dataset_type}.png"
        plt.tight_layout()
        plt.savefig(out)
        plt.close()
        print(f"Wrote {out}")

    if not summary.empty:
        plot_eer_by_inference_duration(fig_dir, summary)
    plot_eer_csv(
        report_dir / "metrics_by_dataset.csv",
        fig_dir / "eer_by_dataset.png",
        "Best EER by source dataset",
        "dataset",
    )
    plot_eer_csv(
        report_dir / "metrics_by_duration_bin.csv",
        fig_dir / "eer_by_duration_bin.png",
        "Best EER by duration bin",
        "duration_bin_eval",
    )


def plot_eer_by_inference_duration(fig_dir: Path, summary: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    single = summary.loc[summary["inference_mode"] == "fixed_length"].copy()
    if single.empty:
        return
    durations = [duration_slug(duration) for duration in DEFAULT_DURATIONS]
    single["series"] = single["track"].astype(str) + " " + single["subset"].astype(str)

    plt.figure(figsize=(8, 4))
    for name, group in single.groupby("series"):
        group = group.set_index("inference_duration").reindex(durations)
        plt.plot(durations, group["eer"], marker="o", label=name)
    plt.title("EER by inference duration")
    plt.xlabel("inference duration")
    plt.ylabel("EER")
    plt.legend(fontsize="small")
    out = fig_dir / "eer_by_inference_duration.png"
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    print(f"Wrote {out}")


def plot_eer_csv(csv_path: Path, out: Path, title: str, x_col: str) -> None:
    import matplotlib.pyplot as plt

    if not csv_path.exists():
        return
    frame = pd.read_csv(csv_path)
    if frame.empty or x_col not in frame.columns:
        return
    best = (
        frame.sort_values("eer", na_position="last")
        .groupby(["track", "dataset_type", "subset", x_col], dropna=False, as_index=False)
        .first()
    )
    best["series"] = best["track"].astype(str) + " " + best["subset"].astype(str)
    best[x_col] = best[x_col].fillna("unknown").astype(str)

    plt.figure(figsize=(10, 4))
    for name, group in best.groupby("series"):
        group = group.sort_values(x_col)
        plt.plot(group[x_col], group["eer"], marker="o", label=name)
    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel("best EER")
    plt.xticks(rotation=30, ha="right")
    plt.legend(fontsize="small")
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    print(f"Wrote {out}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Follow May 29 short-duration benchmark plan.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(sub):
        sub.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
        sub.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
        sub.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
        sub.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
        sub.add_argument("--subsets", nargs="+", default=["test"])
        sub.add_argument("--tracks", nargs="+", default=list(DEFAULT_TRACKS), choices=list(DEFAULT_TRACKS))
        sub.add_argument("--durations", nargs="+", type=float, default=list(DEFAULT_DURATIONS))
        sub.add_argument("--config", default="xlsr_conformertcm_normal")
        sub.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
        sub.add_argument("--adapter-paths", type=Path, default=None)
        sub.add_argument("--comment-prefix", default="baseline_mdt_protocol_test")
        sub.add_argument("--max-per-source", type=int, default=None)
        sub.add_argument("--seed", type=int, default=1234)
        sub.add_argument("--benchmark-subset", default="test")

    for name in ("validate", "prepare", "enrich-single", "aggregate", "report"):
        add_common(subparsers.add_parser(name))

    run = subparsers.add_parser("run-single")
    add_common(run)
    run.add_argument("--gpus", nargs="+", default=["0"])
    run.add_argument("--batch-size", type=int, default=int(os.getenv("DEFAULT_BATCH_SIZE", "128")))
    run.add_argument("--omp-threads", type=int, default=int(os.getenv("OMP_NUM_THREADS", "8")))
    run.add_argument("--is-ln", action=argparse.BooleanOptionalAction, default=False)
    run.add_argument("--random-start", action=argparse.BooleanOptionalAction, default=False)
    run.add_argument("--run-group", choices=["fixed_length", "full_length"], default=None)
    run.add_argument("--full-utterance", action="store_true")
    run.add_argument("--parallel", action="store_true")
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("extra_overrides", nargs=argparse.REMAINDER)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "validate":
        validate_dataset(args)
    elif args.command == "prepare":
        prepare_benchmark(args)
    elif args.command == "run-single":
        run_single(args)
    elif args.command == "enrich-single":
        enrich_single(args)
    elif args.command == "aggregate":
        aggregate_multi(args)
    elif args.command == "report":
        report_metrics(args)


if __name__ == "__main__":
    main()
