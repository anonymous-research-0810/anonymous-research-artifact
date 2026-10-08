"""Normalization for the paper option-hedging pipeline."""

from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np
from option_dataset import OptionDataset
from .constants import DEFAULT_MIN_STATE_STD, EXOGENOUS_FEATURE_NAMES
from .exceptions import EnvironmentConfigurationError


def _get_finite_float(value: Any, name: str) -> float:
    """Return finite float."""
    if isinstance(value, (bool, np.bool_)):
        raise EnvironmentConfigurationError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise EnvironmentConfigurationError(f"{name} must be a finite number") from exc
    if not np.isfinite(result):
        raise EnvironmentConfigurationError(f"{name} must be a finite number")
    return result


@dataclass(frozen=True)
class StateNormalizer:
    """Training-fitted z-score statistics for the six exogenous state features; the previous holding remains unstandardized."""

    feature_names: tuple[str, ...]
    mean: tuple[float, ...]
    std: tuple[float, ...]
    scale: tuple[float, ...]
    is_scaled: tuple[bool, ...]
    min_state_std: float
    num_observation: int
    source_label: str
    source_date_start: str
    source_date_end: str

    def __post_init__(self) -> None:
        """Normalize fields and validate individual and cross-field constraints."""
        num_feature = len(EXOGENOUS_FEATURE_NAMES)
        if tuple(self.feature_names) != EXOGENOUS_FEATURE_NAMES:
            raise EnvironmentConfigurationError(
                "State normalizer feature_names differ from the environment feature order"
            )
        for name, values in (
            ("mean", self.mean),
            ("std", self.std),
            ("scale", self.scale),
            ("is_scaled", self.is_scaled),
        ):
            if len(values) != num_feature:
                raise EnvironmentConfigurationError(
                    f"state normalizer {name} length must be {num_feature}"
                )
        mean = np.asarray(self.mean, dtype=np.float64)
        std = np.asarray(self.std, dtype=np.float64)
        scale = np.asarray(self.scale, dtype=np.float64)
        if not np.isfinite(mean).all() or not np.isfinite(std).all():
            raise EnvironmentConfigurationError("State normalizer statistics must be finite")
        if not np.isfinite(scale).all() or np.any(scale <= 0):
            raise EnvironmentConfigurationError(
                "State normalizer scales must be finite and positive"
            )
        if np.any(std < 0):
            raise EnvironmentConfigurationError(
                "State normalizer standard deviations must be nonnegative"
            )
        if self.source_label != "train":
            raise EnvironmentConfigurationError(
                "Fit the state normalizer on label='train' data only"
            )
        if isinstance(self.num_observation, bool) or self.num_observation <= 0:
            raise EnvironmentConfigurationError("num_observation must be a positive integer")
        if self.min_state_std <= 0 or not np.isfinite(self.min_state_std):
            raise EnvironmentConfigurationError("min_state_std must be finite and positive")

    def transform(self, values: np.ndarray, *, dtype: np.dtype | type = np.float32) -> np.ndarray:
        """Standardize the exogenous features using stored training statistics."""
        array = np.asarray(values, dtype=np.float64)
        if array.ndim == 0 or array.shape[-1] != len(self.feature_names):
            raise EnvironmentConfigurationError(
                "The last state dimension must match the number of feature_names"
            )
        if not np.isfinite(array).all():
            raise EnvironmentConfigurationError("States to normalize contain nonfinite values")
        mean = np.asarray(self.mean, dtype=np.float64)
        scale = np.asarray(self.scale, dtype=np.float64)
        return ((array - mean) / scale).astype(dtype, copy=False)

    def get_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable configuration dictionary."""
        return {
            "feature_names": list(self.feature_names),
            "mean": list(self.mean),
            "std": list(self.std),
            "scale": list(self.scale),
            "is_scaled": list(self.is_scaled),
            "min_state_std": self.min_state_std,
            "num_observation": self.num_observation,
            "source_label": self.source_label,
            "source_date_start": self.source_date_start,
            "source_date_end": self.source_date_end,
        }

    def save(self, path: str | Path, *, is_overwrite: bool = False) -> Path:
        """Persist the object at the requested destination with explicit overwrite handling."""
        output_path = Path(path)
        if output_path.exists() and (not is_overwrite):
            raise FileExistsError(f"state normalizer already exists: {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(self.get_dict(), ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return output_path


def get_state_normalizer(
    train_dataset: OptionDataset, *, min_state_std: float = DEFAULT_MIN_STATE_STD
) -> StateNormalizer:
    """Fit exogenous-state z-scores on nonterminal decision rows from the training split only."""
    if not isinstance(train_dataset, OptionDataset):
        raise EnvironmentConfigurationError("train_dataset must be an OptionDataset")
    if train_dataset.label != "train":
        raise EnvironmentConfigurationError("Fit z-score statistics on label='train' data only")
    if train_dataset.num_episode == 0:
        raise EnvironmentConfigurationError("Cannot fit z-scores on an empty training dataset")
    min_std = _get_finite_float(min_state_std, "min_state_std")
    if min_std <= 0:
        raise EnvironmentConfigurationError("min_state_std must be greater than 0")
    decision_rows = train_dataset.episode_steps.loc[
        ~train_dataset.episode_steps["is_terminal"].astype(bool), EXOGENOUS_FEATURE_NAMES
    ]
    values = decision_rows.to_numpy(dtype=np.float64, copy=True)
    if values.size == 0 or not np.isfinite(values).all():
        raise EnvironmentConfigurationError("Training decision-time states are empty or nonfinite")
    mean = values.mean(axis=0)
    std = values.std(axis=0, ddof=0)
    is_scaled = std >= min_std
    scale = np.where(is_scaled, std, 1.0)
    return StateNormalizer(
        feature_names=EXOGENOUS_FEATURE_NAMES,
        mean=tuple((float(value) for value in mean)),
        std=tuple((float(value) for value in std)),
        scale=tuple((float(value) for value in scale)),
        is_scaled=tuple((bool(value) for value in is_scaled)),
        min_state_std=min_std,
        num_observation=len(values),
        source_label="train",
        source_date_start=train_dataset.config.date_start.strftime("%Y-%m-%d"),
        source_date_end=train_dataset.config.date_end.strftime("%Y-%m-%d"),
    )


def get_state_normalizer_from_json(path: str | Path) -> StateNormalizer:
    """Return state normalizer from JSON."""
    input_path = Path(path)
    data = json.loads(input_path.read_text(encoding="utf-8"))
    return StateNormalizer(
        feature_names=tuple(data["feature_names"]),
        mean=tuple(data["mean"]),
        std=tuple(data["std"]),
        scale=tuple(data["scale"]),
        is_scaled=tuple(data["is_scaled"]),
        min_state_std=float(data["min_state_std"]),
        num_observation=int(data["num_observation"]),
        source_label=str(data["source_label"]),
        source_date_start=str(data["source_date_start"]),
        source_date_end=str(data["source_date_end"]),
    )
