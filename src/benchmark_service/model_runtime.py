from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import hydra
import lightning as L
import rootutils
import torch
from lightning import LightningDataModule, LightningModule, Trainer
from omegaconf import DictConfig, OmegaConf, open_dict

from scripts.benchmark_py.execution import BenchmarkConfig


PROJECT_ROOT = Path(__file__).resolve().parents[2]
rootutils.setup_root(PROJECT_ROOT, indicator=".project-root", pythonpath=True, dotenv=True)
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


class LoadedBenchmarkModel:
    """Hydra/Lightning runtime that keeps model weights resident across datasets."""

    def __init__(self, signature: Dict[str, Any]) -> None:
        self.signature = signature
        self.load_started_at = time.time()
        os.environ["CUDA_VISIBLE_DEVICES"] = str(signature["gpu_id"])
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
        Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
        torch.set_float32_matmul_precision("high")

        self.cfg = self._compose_base_cfg(signature)
        if self.cfg.get("seed"):
            L.seed_everything(self.cfg.seed, workers=True)

        self.model: LightningModule = hydra.utils.instantiate(self.cfg.model)
        self.model.eval()
        self.load_seconds = time.time() - self.load_started_at

    def _compose_base_cfg(self, signature: Dict[str, Any]) -> DictConfig:
        overrides = [
            f"experiment={signature['config_path']}",
            "++train=False",
            "++test=True",
            "++model.spec_eval=True",
            f"++model.base_model_path={signature['model_path']}",
            f"++model.is_base_model_path_ln={str(signature['is_ln']).lower()}",
        ]
        if signature.get("adapter_path"):
            overrides.append(f"++model.adapter_paths={signature['adapter_path']}")
        overrides.extend(signature.get("extra_overrides") or [])

        with hydra.initialize_config_dir(version_base="1.3", config_dir=str(PROJECT_ROOT / "configs")):
            cfg = hydra.compose(config_name="train.yaml", overrides=overrides)
        return cfg

    def _cfg_for_dataset(self, config: BenchmarkConfig) -> DictConfig:
        cfg = OmegaConf.create(OmegaConf.to_container(self.cfg, resolve=False))
        with open_dict(cfg):
            cfg.data.data_dir = str(config.data_dir.absolute())
            cfg.data.batch_size = config.batch_size
            cfg.data.args.protocol_path = str(config.protocol_path.absolute())
            cfg.data.args.random_start = config.is_random_start
            cfg.data.args.trim_length = config.trim_length
            cfg.model.score_save_path = str(config.score_save_path.absolute())
            cfg.model.spec_eval = True
            cfg.model.base_model_path = config.base_model_path
            cfg.model.is_base_model_path_ln = config.is_base_model_path_ln
            cfg.trainer.default_root_dir = str(config.score_save_path.parent.absolute())
            cfg.trainer.enable_checkpointing = False
            cfg.trainer.enable_model_summary = False
            cfg.trainer.enable_progress_bar = False
            cfg.trainer.num_sanity_val_steps = 0
            if "paths" in cfg:
                cfg.paths.output_dir = str(config.score_save_path.parent.absolute())
                cfg.paths.work_dir = str(PROJECT_ROOT)
            if config.adapter_paths:
                cfg.model.adapter_paths = config.adapter_paths
        return cfg

    def execute_benchmark(self, config: BenchmarkConfig) -> bool:
        if not config.protocol_path.exists() or not config.data_dir.exists():
            return False

        cfg = self._cfg_for_dataset(config)
        self.model.score_save_path = str(config.score_save_path.absolute())
        self.model.spec_eval = True
        if hasattr(self.model, "kwargs"):
            self.model.kwargs["score_save_path"] = self.model.score_save_path
            self.model.kwargs["spec_eval"] = True

        datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
        trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=[], logger=False)

        started = time.time()
        with torch.inference_mode():
            trainer.test(model=self.model, datamodule=datamodule, ckpt_path=None)
        self.last_test_seconds = time.time() - started
        return True
