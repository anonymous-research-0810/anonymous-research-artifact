"""Config for the paper option-hedging pipeline."""

from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Any
import numpy as np


def _get_int(value: Any, name: str, *, minimum: int) -> int:
    """Return int."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if result != value or result < minimum:
        raise ValueError(f"{name} must be an integer at least {minimum} ")
    return result


def _get_float(
    value: Any,
    name: str,
    *,
    lower: float | None = None,
    upper: float | None = None,
    is_lower_inclusive: bool = True,
    is_upper_inclusive: bool = True,
) -> float:
    """Return float."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    if lower is not None:
        invalid = result < lower if is_lower_inclusive else result <= lower
        if invalid:
            operator = ">=" if is_lower_inclusive else ">"
            raise ValueError(f"{name} must be {operator} {lower}")
    if upper is not None:
        invalid = result > upper if is_upper_inclusive else result >= upper
        if invalid:
            operator = "<=" if is_upper_inclusive else "<"
            raise ValueError(f"{name} must be {operator} {upper}")
    return result


def _get_float_tuple(values: Any, name: str, *, lower: float, upper: float) -> tuple[float, ...]:
    """Return float tuple."""
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a nonempty numeric sequence")
    try:
        result = tuple((float(value) for value in values))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a nonempty numeric sequence") from exc
    if not result or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a nonempty sequence of finite numbers")
    if any((value < lower or value > upper for value in result)):
        raise ValueError(f"{name} elements must be in [{lower}, {upper}]")
    return result


@dataclass(frozen=True)
class SimulationConfig:
    """Validated SABR calibration and cohort simulation settings. The augmentation ratio controls the number of complete simulated cohorts."""

    num_simulation_times: float = 1.0
    expiry_weekday_calibration: int | None = None
    num_moneyness_calibration: int = 3
    annualization_days: int = 252
    drift_shrinkage_observation: float = 126.0
    spread_weight_epsilon: float = 1e-06
    spread_weight_lower_quantile: float = 0.05
    spread_weight_upper_quantile: float = 0.95
    rho_lower: float = -0.95
    rho_upper: float = 0.95
    nu_lower: float = 0.0001
    nu_upper: float = 5.0
    rho_initial_values: tuple[float, ...] = (-0.7, -0.3, 0.0, 0.3)
    nu_initial_values: tuple[float, ...] = (0.1, 0.5, 1.0, 2.0)
    min_log_moneyness_levels: int = 3
    optimizer_max_nfev: int = 2000
    optimizer_boundary_tolerance: float = 1e-05
    sabr_z_tolerance: float = 1e-06
    max_alpha_dte_difference: int = 5
    max_path_retry: int = 100
    price_relative_tolerance: float = 1e-07
    random_seed: int = 2026
    num_environment_rollout_checks: int = 3
    fallback_task_fraction_threshold: float = 0.2

    def __post_init__(self) -> None:
        """Normalize fields and validate individual and cross-field constraints."""
        object.__setattr__(
            self,
            "num_simulation_times",
            _get_float(
                self.num_simulation_times,
                "num_simulation_times",
                lower=0.0,
                is_lower_inclusive=False,
            ),
        )
        for name, minimum in (
            ("num_moneyness_calibration", 1),
            ("annualization_days", 1),
            ("min_log_moneyness_levels", 3),
            ("optimizer_max_nfev", 1),
            ("max_alpha_dte_difference", 0),
            ("max_path_retry", 1),
            ("random_seed", 0),
            ("num_environment_rollout_checks", 0),
        ):
            object.__setattr__(self, name, _get_int(getattr(self, name), name, minimum=minimum))
        if self.expiry_weekday_calibration is not None:
            weekday = _get_int(
                self.expiry_weekday_calibration, "expiry_weekday_calibration", minimum=0
            )
            if weekday > 6:
                raise ValueError("expiry_weekday_calibration must be in [0, 6]")
            object.__setattr__(self, "expiry_weekday_calibration", weekday)
        object.__setattr__(
            self,
            "drift_shrinkage_observation",
            _get_float(self.drift_shrinkage_observation, "drift_shrinkage_observation", lower=0.0),
        )
        for name in (
            "spread_weight_epsilon",
            "optimizer_boundary_tolerance",
            "sabr_z_tolerance",
            "price_relative_tolerance",
        ):
            object.__setattr__(
                self,
                name,
                _get_float(getattr(self, name), name, lower=0.0, is_lower_inclusive=False),
            )
        lower_q = _get_float(
            self.spread_weight_lower_quantile, "spread_weight_lower_quantile", lower=0.0, upper=1.0
        )
        upper_q = _get_float(
            self.spread_weight_upper_quantile, "spread_weight_upper_quantile", lower=0.0, upper=1.0
        )
        if lower_q >= upper_q:
            raise ValueError("The lower spread-weight quantile must be below the upper quantile")
        object.__setattr__(self, "spread_weight_lower_quantile", lower_q)
        object.__setattr__(self, "spread_weight_upper_quantile", upper_q)
        rho_lower = _get_float(self.rho_lower, "rho_lower", lower=-0.999, upper=0.999)
        rho_upper = _get_float(self.rho_upper, "rho_upper", lower=-0.999, upper=0.999)
        nu_lower = _get_float(self.nu_lower, "nu_lower", lower=0.0, is_lower_inclusive=False)
        nu_upper = _get_float(self.nu_upper, "nu_upper", lower=0.0, is_lower_inclusive=False)
        if rho_lower >= rho_upper or nu_lower >= nu_upper:
            raise ValueError("SABR lower parameter bounds must be below upper bounds")
        for name, value in (
            ("rho_lower", rho_lower),
            ("rho_upper", rho_upper),
            ("nu_lower", nu_lower),
            ("nu_upper", nu_upper),
        ):
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "rho_initial_values",
            _get_float_tuple(
                self.rho_initial_values, "rho_initial_values", lower=rho_lower, upper=rho_upper
            ),
        )
        object.__setattr__(
            self,
            "nu_initial_values",
            _get_float_tuple(
                self.nu_initial_values, "nu_initial_values", lower=nu_lower, upper=nu_upper
            ),
        )
        fraction = _get_float(
            self.fallback_task_fraction_threshold,
            "fallback_task_fraction_threshold",
            lower=0.0,
            upper=1.0,
        )
        object.__setattr__(self, "fallback_task_fraction_threshold", fraction)

    def get_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable configuration dictionary."""
        result = asdict(self)
        result["rho_initial_values"] = list(self.rho_initial_values)
        result["nu_initial_values"] = list(self.nu_initial_values)
        return result
