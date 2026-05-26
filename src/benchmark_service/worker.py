from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from scripts.benchmark_py.benchmark import (
    calculate_pooled_eer,
    create_merged_protocol,
    get_subdirectories,
    initialize_results,
    normalize_yaml_name,
    process_dataset,
    resolve_eval_config_path,
)
from .model_runtime import LoadedBenchmarkModel
from .queue import FileJobQueue
from .schemas import BenchmarkJob


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Persistent benchmark worker")
    parser.add_argument("--queue-dir", default=".benchmark_service_queue")
    parser.add_argument("--poll-seconds", default=5.0, type=float)
    parser.add_argument("--reload-on-change", action="store_true")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--health-path", default=None)
    return parser.parse_args()


class BenchmarkWorker:
    def __init__(self, queue_dir: str | Path, health_path: Optional[str] = None, reload_on_change: bool = False) -> None:
        self.queue = FileJobQueue(queue_dir)
        self.health_path = Path(health_path) if health_path else Path(queue_dir) / "health.json"
        self.reload_on_change = reload_on_change
        self.runtime: Optional[LoadedBenchmarkModel] = None
        self.runtime_signature: Optional[Dict[str, Any]] = None
        self.load_count = 0

    def write_health(self, status: str, **extra: Any) -> None:
        payload = {
            "status": status,
            "pid": os.getpid(),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "load_count": self.load_count,
            "runtime_signature": self.runtime_signature,
            **extra,
        }
        self.health_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.health_path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp_path, self.health_path)

    def ensure_runtime(self, job: BenchmarkJob) -> LoadedBenchmarkModel:
        signature = job.runtime_signature()
        if self.runtime is not None and signature == self.runtime_signature:
            return self.runtime
        if self.runtime is not None and not self.reload_on_change:
            raise RuntimeError("Runtime signature changed; restart worker or use --reload-on-change")
        self.write_health("loading", job_id=job.job_id)
        self.runtime = LoadedBenchmarkModel(signature)
        self.runtime_signature = signature
        self.load_count += 1
        self.write_health("ready", job_id=job.job_id, load_seconds=self.runtime.load_seconds)
        return self.runtime

    def run_job(self, job: BenchmarkJob) -> bool:
        benchmark_folder = Path(job.dataset_path)
        subdirs = get_subdirectories(benchmark_folder)
        if not subdirs:
            raise RuntimeError(f"No subdirectories found in '{benchmark_folder}'")

        results_folder, summary_file, normalized_yaml = initialize_results(
            Path(job.result_dir),
            job.run_name,
            job.config_path,
            job.model_path,
            job.adapter_path,
            job.is_ln,
            job.trim_length,
            Path(job.eval_config) if job.eval_config else resolve_eval_config_path(benchmark_folder, None),
        )
        eval_config_path = Path(job.eval_config) if job.eval_config else resolve_eval_config_path(benchmark_folder, None)

        success_count = 0

        def execute_with_lazy_runtime(config):
            runtime = self.ensure_runtime(job)
            return runtime.execute_benchmark(config)

        for subfolder in subdirs:
            ok = process_dataset(
                subfolder,
                job.gpu_id,
                job.config_path,
                job.model_path,
                results_folder,
                normalized_yaml,
                job.run_name,
                job.adapter_path,
                job.is_ln,
                job.random_start,
                job.trim_length,
                job.batch_size,
                summary_file,
                job.hydra_overrides(),
                eval_config_path,
                benchmark_folder,
                job.missing_protocol_label,
                execute_benchmark_fn=execute_with_lazy_runtime,
            )
            success_count += int(ok)

        calculate_pooled_eer(results_folder, normalized_yaml, job.run_name, summary_file, subdirs, eval_config_path)
        create_merged_protocol(results_folder, normalized_yaml, job.run_name, job.config_path, job.model_path, summary_file, subdirs)
        self.write_metadata(job, results_folder, success_count, len(subdirs), True)
        return success_count == len(subdirs)

    def write_metadata(self, job: BenchmarkJob, results_folder: Path, success_count: int, total_count: int, completed: bool) -> None:
        metadata_dir = results_folder / "service_metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "job": job.to_dict(),
            "worker_pid": os.getpid(),
            "runtime_signature": self.runtime_signature,
            "load_count": self.load_count,
            "success_count": success_count,
            "total_count": total_count,
            "completed": completed,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
        (metadata_dir / f"{job.job_id}.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def process_once(self) -> bool:
        running_path = self.queue.claim_next()
        if running_path is None:
            self.write_health("idle")
            return False
        job = self.queue.read_job(running_path)
        self.write_health("running", job_id=job.job_id)
        success = False
        try:
            success = self.run_job(job)
            return success
        except Exception as exc:
            self.write_health("failed", job_id=job.job_id, error=str(exc))
            print(f"Job {job.job_id} failed: {exc}", file=sys.stderr)
            return False
        finally:
            self.queue.finish(running_path, success)
            if success:
                self.write_health("ready", job_id=job.job_id)
            else:
                self.write_health("failed", job_id=job.job_id)


def main() -> int:
    args = parse_args()
    worker = BenchmarkWorker(args.queue_dir, args.health_path, args.reload_on_change)
    if args.once:
        processed = worker.process_once()
        return 0 if processed else 2
    while True:
        worker.process_once()
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    sys.exit(main())
