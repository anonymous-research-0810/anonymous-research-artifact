"""Config for the paper option-hedging pipeline."""

from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Any
import numpy as np

VALID_SEGMENTATION_ALGORITHMS = frozenset({"rbf"})


def _get_positive_int(value: Any, name: str, *, is_allow_zero: bool = False) -> int:
    """Return positive int."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if number != value:
        raise ValueError(f"{name} must be an integer")
    minimum = 0 if is_allow_zero else 1
    if number < minimum:
        qualifier = "nonnegative " if is_allow_zero else "positive "
        raise ValueError(f"{name} must be{qualifier} integer")
    return number


def _get_finite_float(value: Any, name: str) -> float:
    """Return finite float."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not np.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return number


@dataclass(frozen=True)
class SegmentationConfig:
    """Validated RBF-PELT settings with a slope-based penalty and moving-block bootstrap stability checks."""

    algorithm: str = "rbf"
    min_segment_length: int = 126
    penalty: float | None = None
    num_penalty_grid: int = 50
    penalty_min_ratio: float = 0.001
    penalty_min_ratio_floor: float = 1e-06
    num_penalty_expansion: int = 4
    high_complexity_fraction: float = 0.3
    min_slope_points: int = 5
    bootstrap_repetitions: int = 200
    bootstrap_block_length: int | None = None
    boundary_tolerance: int | None = None
    stability_threshold: float = 0.7
    is_select_stable_plateau: bool = True
    annualization_days: int = 252
    volatility_epsilon: float = 1e-08
    min_feature_std: float = 1e-08
    rbf_gamma: float | None = None
    random_seed: int = 2026

    def __post_init__(self) -> None:
        """Normalize fields and validate individual and cross-field constraints."""
        algorithm = str(self.algorithm).strip().lower()
        object.__setattr__(self, "algorithm", algorithm)
        if algorithm not in VALID_SEGMENTATION_ALGORITHMS:
            raise ValueError(f"algorithm must be {sorted(VALID_SEGMENTATION_ALGORITHMS)}  ")
        integer_fields = (
            "min_segment_length",
            "num_penalty_grid",
            "num_penalty_expansion",
            "min_slope_points",
            "annualization_days",
        )
        for name in integer_fields:
            object.__setattr__(self, name, _get_positive_int(getattr(self, name), name))
        object.__setattr__(
            self,
            "bootstrap_repetitions",
            _get_positive_int(
                self.bootstrap_repetitions, "bootstrap_repetitions", is_allow_zero=True
            ),
        )
        object.__setattr__(
            self,
            "random_seed",
            _get_positive_int(self.random_seed, "random_seed", is_allow_zero=True),
        )
        if self.random_seed > np.iinfo(np.uint32).max:
            raise ValueError("random_seed must be in [0, 2^32-1]")
        if self.num_penalty_grid < 2:
            raise ValueError("num_penalty_grid must be at least 2")
        if self.min_slope_points < 2:
            raise ValueError("min_slope_points must be at least 2")
        if self.bootstrap_block_length is not None:
            object.__setattr__(
                self,
                "bootstrap_block_length",
                _get_positive_int(self.bootstrap_block_length, "bootstrap_block_length"),
            )
        if self.boundary_tolerance is not None:
            object.__setattr__(
                self,
                "boundary_tolerance",
                _get_positive_int(
                    self.boundary_tolerance, "boundary_tolerance", is_allow_zero=True
                ),
            )
        if self.penalty is not None:
            penalty = _get_finite_float(self.penalty, "penalty")
            if penalty <= 0.0:
                raise ValueError("penalty must be strictly positive")
            object.__setattr__(self, "penalty", penalty)
        positive_float_fields = (
            "penalty_min_ratio",
            "penalty_min_ratio_floor",
            "high_complexity_fraction",
            "stability_threshold",
            "volatility_epsilon",
            "min_feature_std",
        )
        for name in positive_float_fields:
            number = _get_finite_float(getattr(self, name), name)
            if number <= 0.0:
                raise ValueError(f"{name} must be strictly positive")
            object.__setattr__(self, name, number)
        if not self.penalty_min_ratio_floor <= self.penalty_min_ratio < 1.0:
            raise ValueError(
                "Penalty ratios must satisfy 0 < penalty_min_ratio_floor <= penalty_min_ratio < 1"
            )
        if self.high_complexity_fraction > 1.0:
            raise ValueError("high_complexity_fraction must be in (0,1]")
        if self.stability_threshold > 1.0:
            raise ValueError("stability_threshold must be in (0,1]")
        if self.rbf_gamma is not None:
            gamma = _get_finite_float(self.rbf_gamma, "rbf_gamma")
            if gamma <= 0.0:
                raise ValueError("rbf_gamma must be strictly positive")
            object.__setattr__(self, "rbf_gamma", gamma)
        boolean_fields = ("is_select_stable_plateau",)
        for name in boolean_fields:
            value = getattr(self, name)
            if not isinstance(value, (bool, np.bool_)):
                raise ValueError(f"{name} must be a boolean")
            object.__setattr__(self, name, bool(value))

    def resolve_for_dataset(self, num_interval: int) -> dict[str, Any]:
        """Resolve for dataset."""
        num_interval = _get_positive_int(num_interval, "num_interval")
        result = self.get_dict()
        result["realized_volatility_window"] = num_interval
        result["resolved_bootstrap_block_length"] = (
            self.bootstrap_block_length if self.bootstrap_block_length is not None else num_interval
        )
        result["resolved_boundary_tolerance"] = (
            self.boundary_tolerance if self.boundary_tolerance is not None else num_interval
        )
        return result

    def get_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable configuration dictionary."""
        return asdict(self)
