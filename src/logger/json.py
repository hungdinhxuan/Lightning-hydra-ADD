import json
import os
from datetime import datetime
from pathlib import Path
from lightning.pytorch.loggers import Logger
from lightning.pytorch.utilities.rank_zero import rank_zero_only


class JSONLogger(Logger):
    def __init__(self, save_dir, name="json_logs", version=None, log_by="epoch", log_on="val"):
        """
        Custom JSON Logger that writes metrics in Katib-compatible JSONL format.
        
        Katib expects line-separated JSON with timestamp field in ISO format.
        Each line: {"epoch": 0, "metric1": value1, "timestamp": "2021-12-02T14:27:51"}
        
        Args:
            save_dir: Root directory where logs will be saved
            name: Name of the logger (creates a subdirectory)
            version: Version number for the logging run
            log_by: Whether to log by "epoch" or "step" (default: "epoch")
            log_on: When to log - "val" (validation only), "train" (training only), or "all" (both) (default: "val")
        """
        super().__init__()
        self._save_dir = save_dir
        self._name = name
        self._version = version
        self._log_by = log_by if log_by in ["epoch", "step"] else "epoch"
        self._log_on = log_on if log_on in ["val", "train", "all"] else "val"
        self._metrics_file = None
        self._hyperparams = {}
        self._log_dir = None
        self._current_epoch = 0

    @property
    def save_dir(self):
        """Directory where JSON logs will be written."""
        return self._save_dir

    @property
    def name(self):
        return self._name

    @property
    def version(self):
        if self._version is None:
            self._version = self._get_next_version()
        return self._version
    
    def _get_next_version(self):
        """Get the next version number by checking existing version directories."""
        root_dir = Path(self.save_dir) / self.name
        if not root_dir.exists():
            return 0
        
        existing_versions = []
        for d in root_dir.iterdir():
            if d.is_dir() and d.name.startswith("version_"):
                try:
                    version_num = int(d.name.split("_")[1])
                    existing_versions.append(version_num)
                except (ValueError, IndexError):
                    pass
        
        return max(existing_versions) + 1 if existing_versions else 0

    @property
    def log_dir(self):
        """Get the full path to the logging directory."""
        if self._log_dir is None:
            self._log_dir = os.path.join(
                self.save_dir, 
                self.name, 
                f"version_{self.version}"
            )
            os.makedirs(self._log_dir, exist_ok=True)
        return self._log_dir

    @rank_zero_only
    def log_metrics(self, metrics, step):
        """
        Log metrics immediately in Katib-compatible JSONL format.
        Each line is a complete JSON object with timestamp.
        """
        # Determine if we should log based on metric type
        has_val_metrics = any(key.startswith("val/") or key.startswith("val_") for key in metrics.keys())
        has_train_metrics = any(key.startswith("train/") or key.startswith("train_") for key in metrics.keys())
        
        # Skip logging based on log_on setting
        if self._log_on == "val" and not has_val_metrics:
            return
        elif self._log_on == "train" and not has_train_metrics:
            return
        # If log_on == "all", we log everything (no return)
        
        if self._metrics_file is None:
            # Open metrics file on first call
            metrics_path = os.path.join(self.log_dir, "metrics.json")
            self._metrics_file = open(metrics_path, "a")
        
        # Check if we should track epoch from metrics
        if "epoch" in metrics:
            self._current_epoch = metrics["epoch"]
        
        # Create metrics entry with timestamp (required by Katib)
        metrics_entry = {
            "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")
        }
        
        # Add epoch or step based on log_by setting
        if self._log_by == "epoch":
            metrics_entry["epoch"] = self._current_epoch
        else:
            metrics_entry["step"] = step
        
        # Add all metrics, sanitizing non-serializable values
        for key, val in metrics.items():
            try:
                # Test if value is JSON serializable
                json.dumps(val)
                metrics_entry[key] = val
            except (TypeError, ValueError):
                # Convert non-serializable to string
                metrics_entry[key] = str(val)
        
        # Write as single line JSON (JSONL format)
        self._metrics_file.write(json.dumps(metrics_entry) + "\n")
        self._metrics_file.flush()  # Ensure immediate write for Katib to read

    @rank_zero_only
    def log_hyperparams(self, params):
        """Log hyperparameters to a separate JSON file."""
        self._hyperparams = self._sanitize_params(params)
        
        # Write hyperparameters to separate file
        hparams_file = os.path.join(self.log_dir, "hparams.json")
        with open(hparams_file, "w") as f:
            json.dump(self._hyperparams, f, indent=2)

    def _sanitize_params(self, params):
        """Convert parameters to JSON-serializable format."""
        sanitized = {}
        for key, val in params.items():
            if isinstance(val, (str, int, float, bool, type(None))):
                sanitized[key] = val
            elif isinstance(val, (list, tuple)):
                sanitized[key] = [self._sanitize_value(v) for v in val]
            elif isinstance(val, dict):
                sanitized[key] = self._sanitize_params(val)
            else:
                sanitized[key] = str(val)
        return sanitized
    
    def _sanitize_value(self, val):
        """Sanitize a single value for JSON serialization."""
        if isinstance(val, (str, int, float, bool, type(None))):
            return val
        elif isinstance(val, (list, tuple)):
            return [self._sanitize_value(v) for v in val]
        elif isinstance(val, dict):
            return self._sanitize_params(val)
        else:
            return str(val)

    @rank_zero_only
    def finalize(self, status):
        """Close the metrics file when training finishes."""
        if self._metrics_file is not None:
            self._metrics_file.close()
            self._metrics_file = None
