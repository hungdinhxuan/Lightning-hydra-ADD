import json
import os
from pathlib import Path
from lightning.pytorch.loggers import Logger
from lightning.pytorch.utilities.rank_zero import rank_zero_only


class JSONLogger(Logger):
    def __init__(self, save_dir, name="json_logs", version=None):
        """
        Custom JSON Logger that writes metrics and hyperparameters to JSON files.
        
        Args:
            save_dir: Root directory where logs will be saved
            name: Name of the logger (creates a subdirectory)
            version: Version number for the logging run
        """
        super().__init__()
        self._save_dir = save_dir
        self._name = name
        self._version = version
        self._metrics = []  # Store all metrics in memory
        self._hyperparams = {}
        self._log_dir = None

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
        """Log metrics to memory, will be written to file on finalize."""
        metrics_entry = {"step": step}
        metrics_entry.update(metrics)
        self._metrics.append(metrics_entry)

    @rank_zero_only
    def log_hyperparams(self, params):
        """Log hyperparameters to a JSON file."""
        # Convert params to a JSON-serializable format
        self._hyperparams = self._sanitize_params(params)
        
        # Write hyperparameters immediately
        hparams_file = os.path.join(self.log_dir, "hparams.json")
        with open(hparams_file, "w") as f:
            json.dump(self._hyperparams, f, indent=2)

    def _sanitize_params(self, params):
        """Convert parameters to JSON-serializable format."""
        sanitized = {}
        for key, val in params.items():
            # Handle common non-serializable types
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
        """Write all metrics to a JSON file when training finishes."""
        if self._metrics:
            metrics_file = os.path.join(self.log_dir, "metrics.json")
            with open(metrics_file, "w") as f:
                json.dump(self._metrics, f, indent=2)
        
        # Also create a summary file with final metrics
        if self._metrics:
            summary = {
                "status": status,
                "total_steps": len(self._metrics),
                "final_metrics": self._metrics[-1] if self._metrics else {},
                "hyperparameters": self._hyperparams
            }
            summary_file = os.path.join(self.log_dir, "summary.json")
            with open(summary_file, "w") as f:
                json.dump(summary, f, indent=2)
