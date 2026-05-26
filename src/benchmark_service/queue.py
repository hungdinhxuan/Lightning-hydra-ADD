from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from .schemas import BenchmarkJob


class FileJobQueue:
    """Small atomic file queue for one benchmark worker."""

    STATES = ("pending", "running", "done", "failed")

    def __init__(self, queue_dir: str | Path) -> None:
        self.queue_dir = Path(queue_dir)
        for state in self.STATES:
            (self.queue_dir / state).mkdir(parents=True, exist_ok=True)

    def submit(self, job: BenchmarkJob) -> Path:
        path = self.queue_dir / "pending" / f"{job.job_id}.json"
        tmp_path = path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(job.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp_path, path)
        return path

    def claim_next(self) -> Optional[Path]:
        for pending_path in sorted((self.queue_dir / "pending").glob("*.json")):
            running_path = self.queue_dir / "running" / pending_path.name
            try:
                os.replace(pending_path, running_path)
            except FileNotFoundError:
                continue
            return running_path
        return None

    def read_job(self, path: Path) -> BenchmarkJob:
        return BenchmarkJob.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def finish(self, path: Path, success: bool) -> Path:
        target_dir = self.queue_dir / ("done" if success else "failed")
        target = target_dir / path.name
        os.replace(path, target)
        return target

