from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4


@dataclass
class BenchmarkJob:
    dataset_path: str
    result_dir: str
    run_name: str
    config_path: str
    model_path: str
    gpu_id: str = "0"
    adapter_path: Optional[str] = None
    batch_size: int = 128
    precision: Optional[str] = None
    is_ln: bool = True
    random_start: bool = True
    trim_length: int = 64000
    eval_config: Optional[str] = None
    missing_protocol_label: str = "skip"
    extra_overrides: List[str] = field(default_factory=list)
    job_id: str = field(default_factory=lambda: uuid4().hex)
    submitted_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "BenchmarkJob":
        return cls(**payload)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def runtime_signature(self) -> Dict[str, Any]:
        overrides = list(self.extra_overrides)
        if self.precision and not any(
            item.startswith("+trainer.precision=")
            or item.startswith("++trainer.precision=")
            or item.startswith("trainer.precision=")
            for item in overrides
        ):
            overrides.append(f"++trainer.precision={self.precision}")

        return {
            "gpu_id": self.gpu_id,
            "config_path": self.config_path,
            "model_path": self.model_path,
            "adapter_path": self.adapter_path,
            "is_ln": self.is_ln,
            "precision": self.precision,
            "extra_overrides": overrides,
        }

    def hydra_overrides(self) -> List[str]:
        return list(self.runtime_signature()["extra_overrides"])

