"""Lightweight run logging for workflow artifacts."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class RunLogger:
    """Write workflow artifacts to a timestamped directory, with optional MLflow."""

    def __init__(
        self,
        base_dir: str | Path = "runs",
        name: str | None = None,
        *,
        mlflow: bool = False,
    ) -> None:
        """Create a filesystem run logger.

        Parameters
        ----------
        base_dir:
            Parent directory where the timestamped run folder is created.
        name:
            Optional human-readable run name included in the folder name.
        mlflow:
            Whether to mirror params, metrics, and artifacts to MLflow. MLflow
            is imported lazily only when this is ``True``.
        """

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = _slug(name) if name else "glm_run"
        self.path = Path(base_dir) / f"{timestamp}_{run_name}"
        self.path.mkdir(parents=True, exist_ok=True)
        self._mlflow = None
        if mlflow:
            try:
                import mlflow as mlflow_module
            except ModuleNotFoundError as exc:
                raise ImportError("Install glm-factor-optimizer[mlflow] to enable MLflow logging.") from exc
            self._mlflow = mlflow_module
            self._mlflow.start_run(run_name=name)

    def log_json(self, name: str, data: Any) -> Path:
        """Write JSON-serializable data under the run directory.

        Parameters
        ----------
        name:
            Relative artifact name. A ``.json`` suffix is added if missing.
        data:
            Data to serialize. Common numpy, pandas, pathlib, and dataclass
            values are converted to JSON-compatible forms.

        Returns
        -------
        pathlib.Path
            Path to the written JSON artifact.
        """

        path = self.path / _ensure_suffix(name, ".json")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(_jsonable(data), handle, indent=2, sort_keys=True)
        if self._mlflow is not None:
            self._mlflow.log_artifact(str(path))
        return path

    def log_frame(self, name: str, frame: pd.DataFrame) -> Path:
        """Write a dataframe as CSV under the run directory.

        Parameters
        ----------
        name:
            Relative artifact name. A ``.csv`` suffix is added if missing.
        frame:
            Dataframe to write.

        Returns
        -------
        pathlib.Path
            Path to the written CSV artifact.
        """

        path = self.path / _ensure_suffix(name, ".csv")
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
        if self._mlflow is not None:
            self._mlflow.log_artifact(str(path))
        return path

    def log_params(self, params: dict[str, Any]) -> Path:
        """Write parameters and mirror scalar parameters to MLflow when enabled.

        Parameters
        ----------
        params:
            Parameter dictionary to write as ``params.json``.

        Returns
        -------
        pathlib.Path
            Path to the written params artifact.
        """

        path = self.log_json("params.json", params)
        if self._mlflow is not None:
            self._mlflow.log_params({key: value for key, value in params.items() if _is_scalar(value)})
        return path

    def log_metrics(self, metrics: dict[str, float]) -> Path:
        """Write metrics and mirror them to MLflow when enabled.

        Parameters
        ----------
        metrics:
            Mapping from metric name to numeric value.

        Returns
        -------
        pathlib.Path
            Path to the written metrics artifact.
        """

        clean = {key: float(value) for key, value in metrics.items()}
        path = self.log_json("metrics.json", clean)
        if self._mlflow is not None:
            self._mlflow.log_metrics(clean)
        return path

    def log_report(self, prefix: str, report: dict[str, pd.DataFrame]) -> None:
        """Write every dataframe in a report dictionary as CSV.

        Parameters
        ----------
        prefix:
            Filename prefix applied to each report table.
        report:
            Mapping from table name to dataframe.
        """

        for name, frame in report.items():
            self.log_frame(f"{prefix}_{name}.csv", frame)

    def close(self) -> None:
        """Close the optional MLflow run.

        This is a no-op when MLflow logging was not enabled.
        """

        if self._mlflow is not None:
            self._mlflow.end_run()


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _is_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


def _slug(value: str) -> str:
    cleaned = "".join(char.lower() if char.isalnum() else "_" for char in value.strip())
    return "_".join(part for part in cleaned.split("_") if part) or "glm_run"


def _ensure_suffix(name: str, suffix: str) -> str:
    return name if name.endswith(suffix) else f"{name}{suffix}"
