#!/usr/bin/env python3
"""Extract spoof generator (TTS/VC) counts from training and eval protocol files."""

import argparse
import csv
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark_py.protocol import parse_protocol_line


@dataclass
class ExtractionStats:
    total_lines: int = 0
    spoof_lines: int = 0
    skipped_lines: int = 0
    non_spoof_lines: int = 0
    skipped_subset_lines: int = 0

    def merge(self, other: "ExtractionStats") -> None:
        self.total_lines += other.total_lines
        self.spoof_lines += other.spoof_lines
        self.skipped_lines += other.skipped_lines
        self.non_spoof_lines += other.non_spoof_lines
        self.skipped_subset_lines += other.skipped_subset_lines


def extract_generator(rel_path: str) -> str:
    """Extract generator name from a protocol relative path using pattern rules."""
    cleaned = rel_path.strip().strip('"').strip("'")
    parts = Path(cleaned).parts

    if "fake" in parts:
        idx = parts.index("fake")
        if idx + 2 < len(parts):
            return parts[idx + 2]

    if parts and parts[0] == "tts-output" and len(parts) >= 2:
        return parts[1]

    if parts and parts[0] == "April_Synthesizers" and len(parts) >= 2:
        return parts[1]

    if len(parts) >= 3 and parts[0] == "FEB_dataset" and parts[1] == "Synthesizers":
        return parts[2]

    if len(parts) >= 2 and parts[0] == "FEB_dataset":
        return parts[1]

    if parts and parts[0] == "2026_April_Dataset_Jiwon_collected" and len(parts) >= 2:
        return parts[1]

    # Kipot / ASVspoof-style: 2025/eval/<attack_id>/<system_id>/<file>
    if parts and parts[0] == "2025" and len(parts) >= 3 and parts[1] == "eval":
        return parts[2]

    return Path(cleaned).parent.name


def infer_train_dataset(rel_path: str) -> str:
    """Infer dataset/source name from a training-protocol relative path."""
    cleaned = rel_path.strip().strip('"').strip("'")
    parts = Path(cleaned).parts
    if not parts:
        return "unknown"
    return parts[0]


def aggregate_generators(
    protocol_sources: Iterable[Tuple[Path, str]],
    *,
    skip_subsets: frozenset[str] = frozenset(),
    keep_subsets: Optional[frozenset[str]] = None,
    dataset_from_path: bool = False,
) -> Tuple[Counter, ExtractionStats]:
    """Count spoof samples per (generator, dataset, subset) across protocol files."""
    counts: Counter = Counter()
    stats = ExtractionStats()

    for protocol_path, dataset_name in protocol_sources:
        with protocol_path.open(encoding="utf-8") as handle:
            for line in handle:
                stats.total_lines += 1
                parsed = parse_protocol_line(line)
                if parsed is None:
                    stats.skipped_lines += 1
                    continue

                rel_path, subset, label = parsed
                if label != "spoof":
                    stats.non_spoof_lines += 1
                    continue

                if keep_subsets is not None and subset not in keep_subsets:
                    stats.skipped_subset_lines += 1
                    continue
                if subset in skip_subsets:
                    stats.skipped_subset_lines += 1
                    continue

                stats.spoof_lines += 1
                generator = extract_generator(rel_path)
                row_dataset = infer_train_dataset(rel_path) if dataset_from_path else dataset_name
                counts[(generator, row_dataset, subset)] += 1

    return counts, stats


def write_generator_csv(output_path: Path, counts: Counter) -> None:
    """Write generator counts to CSV sorted by dataset, generator, subset."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["generator", "dataset", "subset", "count"])
        for (generator, dataset, subset), count in sorted(counts.items()):
            writer.writerow([generator, dataset, subset, count])


def dedupe_train_dev_counts(
    counts: Counter,
) -> list[tuple[str, int, int, int, str]]:
    """Collapse train/dev rows to one entry per unique generator name."""
    totals: Counter = Counter()
    train_counts: Counter = Counter()
    dev_counts: Counter = Counter()
    datasets_by_generator: dict[str, set[str]] = {}

    for (generator, dataset, subset), count in counts.items():
        totals[generator] += count
        datasets_by_generator.setdefault(generator, set()).add(dataset)
        if subset == "train":
            train_counts[generator] += count
        elif subset == "dev":
            dev_counts[generator] += count

    rows: list[tuple[str, int, int, int, str]] = []
    for generator in sorted(totals):
        dataset_list = ";".join(sorted(datasets_by_generator[generator]))
        rows.append(
            (
                generator,
                totals[generator],
                train_counts[generator],
                dev_counts[generator],
                dataset_list,
            )
        )
    return rows


def write_deduped_train_dev_csv(output_path: Path, counts: Counter) -> None:
    """Write one row per unique generator, merging train/dev and all datasets."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = dedupe_train_dev_counts(counts)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["generator", "total_count", "train_count", "dev_count", "datasets"])
        for generator, total_count, train_count, dev_count, datasets in rows:
            writer.writerow([generator, total_count, train_count, dev_count, datasets])


def print_summary(title: str, counts: Counter, stats: ExtractionStats, output_path: Path) -> None:
    """Print extraction summary to stdout."""
    total_samples = sum(counts.values())
    unique_generators = len({generator for generator, _, _ in counts})
    unique_datasets = len({dataset for _, dataset, _ in counts})

    print(f"\n=== {title} ===")
    print(f"Output: {output_path}")
    print(f"Protocol lines read: {stats.total_lines}")
    print(f"Spoof lines counted: {stats.spoof_lines}")
    print(f"Non-spoof lines skipped: {stats.non_spoof_lines}")
    print(f"Subset-filtered lines skipped: {stats.skipped_subset_lines}")
    print(f"Malformed/skipped lines: {stats.skipped_lines}")
    print(f"Unique datasets: {unique_datasets}")
    print(f"Unique generators: {unique_generators}")
    print(f"Total spoof samples in CSV: {total_samples}")

    print("\nTop 10 generators by count:")
    generator_totals = Counter()
    for (generator, _dataset, _subset), count in counts.items():
        generator_totals[generator] += count
    for generator, count in generator_totals.most_common(10):
        print(f"  {generator}: {count}")


def discover_eval_protocols(
    eval_root: Path,
    excluded_datasets: Sequence[str] = (),
) -> list[Tuple[Path, str]]:
    """Find protocol.txt files under each dataset folder in eval_root."""
    excluded = frozenset(excluded_datasets)
    protocol_sources: list[Tuple[Path, str]] = []
    for dataset_dir in sorted(eval_root.iterdir()):
        if not dataset_dir.is_dir():
            continue
        dataset_name = dataset_dir.name
        if dataset_name in excluded:
            continue
        protocol_path = dataset_dir / "protocol.txt"
        if protocol_path.exists():
            protocol_sources.append((protocol_path, dataset_name))
    return protocol_sources


def run_train(protocol_path: Path, output_path: Path) -> None:
    counts, stats = aggregate_generators(
        [(protocol_path, protocol_path.stem)],
        skip_subsets=frozenset({"eval"}),
        dataset_from_path=True,
    )
    write_generator_csv(output_path, counts)
    deduped_path = output_path.parent / "train_dev_no_duplicated_generators.csv"
    write_deduped_train_dev_csv(deduped_path, counts)
    print_summary("Training generators (train/dev only)", counts, stats, output_path)
    print(f"Deduped generators: {deduped_path} ({len(dedupe_train_dev_counts(counts))} unique names)")


def run_eval(
    eval_root: Path,
    output_path: Path,
    train_protocol: Optional[Path] = None,
    excluded_datasets: Sequence[str] = (),
) -> None:
    protocol_sources = discover_eval_protocols(eval_root, excluded_datasets)
    if not protocol_sources and train_protocol is None:
        raise FileNotFoundError(f"No protocol.txt files found under {eval_root}")

    counts: Counter = Counter()
    stats = ExtractionStats()

    if protocol_sources:
        benchmark_counts, benchmark_stats = aggregate_generators(protocol_sources)
        counts += benchmark_counts
        stats.merge(benchmark_stats)

    if train_protocol is not None:
        train_eval_counts, train_eval_stats = aggregate_generators(
            [(train_protocol, train_protocol.stem)],
            keep_subsets=frozenset({"eval"}),
            dataset_from_path=True,
        )
        counts += train_eval_counts
        stats.merge(train_eval_stats)

    write_generator_csv(output_path, counts)
    print_summary("Eval generators (benchmark + training eval subset)", counts, stats, output_path)

    if protocol_sources:
        print("\nEval benchmark datasets scanned:")
        for _protocol_path, dataset_name in protocol_sources:
            print(f"  - {dataset_name}")
    if train_protocol is not None:
        print(f"\nTraining protocol eval subset included from: {train_protocol}")
    if excluded_datasets:
        print("\nEval datasets excluded:")
        for dataset_name in sorted(excluded_datasets):
            print(f"  - {dataset_name}")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract spoof generator counts from training/eval protocol files.",
    )
    parser.add_argument(
        "--mode",
        choices=("train", "eval", "all"),
        required=True,
        help="Which protocols to process.",
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        help="Training protocol file (required for train/all; optional for eval to include eval subset).",
    )
    parser.add_argument(
        "--eval-root",
        type=Path,
        help="Eval benchmark root containing dataset folders (required for eval/all modes).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output CSV path (for train or eval mode).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for train_generators.csv and eval_generators.csv (all mode).",
    )
    parser.add_argument(
        "--exclude-eval-dataset",
        action="append",
        default=[],
        help="Eval benchmark dataset folder name to exclude (repeatable).",
    )
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if args.mode in {"train", "all"}:
        if args.protocol is None:
            parser_error("--protocol is required for train/all modes")
        if not args.protocol.exists():
            raise FileNotFoundError(f"Training protocol not found: {args.protocol}")

    if args.mode in {"eval", "all"}:
        if args.eval_root is None:
            parser_error("--eval-root is required for eval/all modes")
        if not args.eval_root.is_dir():
            raise FileNotFoundError(f"Eval root not found: {args.eval_root}")
        if args.protocol is not None and not args.protocol.exists():
            raise FileNotFoundError(f"Training protocol not found: {args.protocol}")

    if args.mode == "all":
        if args.output_dir is None:
            parser_error("--output-dir is required for all mode")
    elif args.output is None:
        parser_error("--output is required for train/eval modes")


def parser_error(message: str) -> None:
    raise SystemExit(f"error: {message}")


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    validate_args(args)
    excluded_eval_datasets = args.exclude_eval_dataset

    if args.mode == "train":
        run_train(args.protocol, args.output)
    elif args.mode == "eval":
        run_eval(
            args.eval_root,
            args.output,
            train_protocol=args.protocol,
            excluded_datasets=excluded_eval_datasets,
        )
    else:
        output_dir = args.output_dir
        run_train(args.protocol, output_dir / "train_generators.csv")
        run_eval(
            args.eval_root,
            output_dir / "eval_generators.csv",
            train_protocol=args.protocol,
            excluded_datasets=excluded_eval_datasets,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
