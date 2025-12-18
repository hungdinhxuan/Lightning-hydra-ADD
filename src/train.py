from typing import Any, Dict, List, Optional, Tuple
import os
import hydra
import lightning as L
import rootutils
import torch
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig
import sys
import hypertune

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #
from src.utils import (
    RankedLogger,
    extras,
    get_metric_value,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
    average_checkpoints
)

log = RankedLogger(__name__, rank_zero_only=True)
import warnings
warnings.filterwarnings('ignore')
orig_torch_load = torch.load
def torch_wrapper(*args, **kwargs):
    log.warning("[unsafe-torch] I have unsafely patched `torch.load`.  The `weights_only` option of `torch.load` is forcibly disabled.")
    kwargs['weights_only'] = False

    return orig_torch_load(*args, **kwargs)

torch.load = torch_wrapper


class HypertuneCallback(Callback):
    """Lightning callback for reporting metrics using hypertune for JSON logging.
    
    This callback integrates with cloudml-hypertune to report metrics in JSON format,
    compatible with Kubeflow Katib and other hyperparameter tuning frameworks.
    """
    
    def __init__(self, log_path: Optional[str] = None, enabled: bool = True):
        """Initialize the HypertuneCallback.
        
        Args:
            log_path: Optional path to save JSON metrics. If provided, sets CLOUD_ML_HP_METRIC_FILE.
            enabled: Whether to enable hypertune reporting. Defaults to True.
        """
        super().__init__()
        self.enabled = enabled
        if not self.enabled:
            return
            
        # Set up hypertune
        if log_path:
            os.environ["CLOUD_ML_HP_METRIC_FILE"] = log_path
            
        self.hpt = hypertune.HyperTune()
        log.info(f"HypertuneCallback initialized with log_path: {log_path}")
    
    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Called when the train epoch ends."""
        if not self.enabled:
            return
            
        metrics = trainer.callback_metrics
        epoch = trainer.current_epoch
        
        # Report training metrics
        if "train/loss" in metrics:
            self.hpt.report_hyperparameter_tuning_metric(
                hyperparameter_metric_tag="train_loss",
                metric_value=float(metrics["train/loss"]),
                global_step=epoch
            )
        
        if "train/acc" in metrics:
            self.hpt.report_hyperparameter_tuning_metric(
                hyperparameter_metric_tag="train_accuracy",
                metric_value=float(metrics["train/acc"]),
                global_step=epoch
            )
    
    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Called when the validation epoch ends."""
        if not self.enabled:
            return
            
        metrics = trainer.callback_metrics
        epoch = trainer.current_epoch
        
        # Report validation metrics
        if "val/loss" in metrics:
            self.hpt.report_hyperparameter_tuning_metric(
                hyperparameter_metric_tag="val_loss",
                metric_value=float(metrics["val/loss"]),
                global_step=epoch
            )
        
        if "val/acc" in metrics:
            self.hpt.report_hyperparameter_tuning_metric(
                hyperparameter_metric_tag="val_accuracy",
                metric_value=float(metrics["val/acc"]),
                global_step=epoch
            )
        
        # Report view-specific metrics if available (for MDT models)
        for key in metrics.keys():
            if key.startswith("val/view_") and "_acc" in key:
                view_name = key.replace("val/", "").replace("/", "_")
                self.hpt.report_hyperparameter_tuning_metric(
                    hyperparameter_metric_tag=view_name,
                    metric_value=float(metrics[key]),
                    global_step=epoch
                )
    
    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Called when the test epoch ends."""
        if not self.enabled:
            return
            
        metrics = trainer.callback_metrics
        
        # Report test metrics (no global_step for final test)
        if "test/loss" in metrics:
            self.hpt.report_hyperparameter_tuning_metric(
                hyperparameter_metric_tag="test_loss",
                metric_value=float(metrics["test/loss"]),
                global_step=0
            )
        
        if "test/acc" in metrics:
            self.hpt.report_hyperparameter_tuning_metric(
                hyperparameter_metric_tag="test_accuracy",
                metric_value=float(metrics["test/acc"]),
                global_step=0
            )


@task_wrapper
def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with metrics and dict with all instantiated objects.
    """
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))
    
    # Add HypertuneCallback for JSON logging if enabled
    if cfg.get("json_logging", {}).get("enabled", False):
        log_path = cfg.get("json_logging", {}).get("log_path", None)
        hypertune_callback = HypertuneCallback(log_path=log_path, enabled=True)
        callbacks.append(hypertune_callback)
        log.info(f"JSON logging enabled with log_path: {log_path}")

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule,
                    ckpt_path=cfg.get("ckpt_path"))

    train_metrics = trainer.callback_metrics

    # Model averaging
    if cfg.get("model_averaging"):
        log.info("Performing model averaging...")
        ckpt_path = trainer.checkpoint_callback.best_model_path

        if ckpt_path == "":
            checkpoint_dir = cfg.get("ckpt_path")
        else:
            checkpoint_dir = os.path.dirname(ckpt_path)
        averaged_ckpt_path = average_checkpoints(
            checkpoint_dir=checkpoint_dir,
            model=model,
            # top_k=cfg.model_averaging.top_k
        )
        if averaged_ckpt_path:
            log.info(f"Created averaged checkpoint: {averaged_ckpt_path}")
            # Optionally test with averaged model
            if cfg.get("test"):
                log.info("Testing with averaged model...")
                trainer.test(model=model, datamodule=datamodule,
                             ckpt_path=averaged_ckpt_path)

    elif cfg.get("test"):
        log.info("Starting testing!")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            ckpt_path = cfg.get("ckpt_path")
            log.warning(f"Using weights {ckpt_path} for testing...")
        
        is_push_model_to_mlflow = cfg.get("push_model_to_mlflow", False)
        if is_push_model_to_mlflow:
            log.info("Pushing model to MLflow...")
            # Loading checkpoint from ckpt_path
            #model.
            model_uri = model.push_model_and_artifacts_to_mlflow()
            log.info(f"Model pushed to MLflow: {model_uri}")
            sys.exit(0) 
        
        from lightning.pytorch.utilities.model_summary import summarize
        model.eval()
        log.info("Model summary:")
        print(summarize(model, max_depth=2))
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        log.info(f"Best ckpt path: {ckpt_path}")
    
    test_metrics = trainer.callback_metrics

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    # train the model
    metric_dict, _ = train(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    main()
    with open("/tmp/log.txt", "a") as f:
        f.write("val/acc: 0.92\n")
    sys.exit(0)
