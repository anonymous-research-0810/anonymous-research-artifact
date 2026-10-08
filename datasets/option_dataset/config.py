"""Config for the paper option-hedging pipeline."""

from __future__ import annotations
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence
import numpy as np
import pandas as pd
from .constants import VALID_CP_FLAGS, VALID_LABELS, VALID_SYMBOL_STARTS
from .exceptions import DatasetConfigurationError


def _get_finite_float(value: Any, name: str) -> float:
    """Return finite float."""
    if isinstance(value, (bool, np.bool_)):
        raise DatasetConfigurationError(f"{name} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise DatasetConfigurationError(f"{name} must be a finite number") from exc
    if not np.isfinite(number):
        raise DatasetConfigurationError(f"{name} must be a finite number")
    return number


def _get_positive_int(value: Any, name: str, *, is_allow_zero: bool = False) -> int:
    """Return positive int."""
    if isinstance(value, bool):
        raise DatasetConfigurationError(f"{name} must be an integer")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise DatasetConfigurationError(f"{name} must be an integer") from exc
    if number != value:
        raise DatasetConfigurationError(f"{name} must be an integer")
    num_minimum = 0 if is_allow_zero else 1
    if number < num_minimum:
        qualifier = "nonnegative " if is_allow_zero else "positive "
        raise DatasetConfigurationError(f"{name} must be{qualifier} integer")
    return number


def get_date_period(date_period: Sequence[Any]) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return normalized, timezone-free start and end timestamps for a closed interval."""
    try:
        num_date = len(date_period)
    except TypeError as exc:
        raise DatasetConfigurationError("date_period must contain a start and end date") from exc
    if isinstance(date_period, (str, bytes)) or num_date != 2:
        raise DatasetConfigurationError("date_period must contain a start and end date")
    try:
        date_start = pd.Timestamp(date_period[0]).normalize().tz_localize(None)
        date_end = pd.Timestamp(date_period[1]).normalize().tz_localize(None)
    except (TypeError, ValueError) as exc:
        raise DatasetConfigurationError("date_period contains unparseable dates") from exc
    if pd.isna(date_start) or pd.isna(date_end):
        raise DatasetConfigurationError("date_period must not contain missing dates")
    if date_start > date_end:
        raise DatasetConfigurationError("date_period must start on or before its end date")
    return (date_start, date_end)


def get_moneyness_range(moneyness_range: Sequence[Any]) -> tuple[float, float]:
    """Validate positive moneyness bounds containing the at-the-money value 1."""
    try:
        num_bound = len(moneyness_range)
    except TypeError as exc:
        raise DatasetConfigurationError(
            "moneyness_range must contain lower and upper bounds"
        ) from exc
    if isinstance(moneyness_range, (str, bytes)) or num_bound != 2:
        raise DatasetConfigurationError("moneyness_range must contain lower and upper bounds")
    lower = _get_finite_float(moneyness_range[0], "moneyness_range[0]")
    upper = _get_finite_float(moneyness_range[1], "moneyness_range[1]")
    if not 0 < lower <= 1.0 <= upper or lower >= upper:
        raise DatasetConfigurationError(
            "moneyness_range must satisfy 0 < lower <= 1.0 <= upper and lower < upper"
        )
    return (lower, upper)


@dataclass(frozen=True)
class OptionDatasetConfig:
    """Immutable episode configuration. Dates are closed intervals and each episode remains within its split."""

    date_start: pd.Timestamp
    date_end: pd.Timestamp
    label: str
    data_root: Path = Path("data")
    cp_flag: str = "C"
    symbol_start: str = "SPXW"
    expiry_weekday: int | None = None
    num_interval: int = 20
    num_moneyness: int = 3
    moneyness_lower: float = 0.95
    moneyness_upper: float = 1.05
    max_relative_spread: float = 0.1
    min_open_interest: float = 0.0
    min_iv: float = 1e-06
    max_iv: float = 5.0
    iv_solver_tolerance: float = 1e-08
    num_iv_solver_iterations: int = 100
    num_iv_surface_points: int = 2
    max_iv_extrapolation_log_moneyness: float = 0.1
    num_iv_forward_fill_steps: int = 1
    is_use_option_price_for_iv: bool = True
    is_use_same_strike_opposite_iv: bool = True
    is_use_surface_iv: bool = True
    is_use_past_iv: bool = True
    is_require_balanced_cohort: bool = True
    is_allow_empty: bool = False
    is_require_environment_ready: bool = True
    num_scan_batch_size: int = 131072

    def __post_init__(self) -> None:
        """Normalize fields and validate individual and cross-field constraints."""
        date_start, date_end = get_date_period((self.date_start, self.date_end))
        label = str(self.label).lower()
        cp_flag = str(self.cp_flag).upper()
        symbol_start = str(self.symbol_start).upper()
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "cp_flag", cp_flag)
        object.__setattr__(self, "symbol_start", symbol_start)
        object.__setattr__(self, "data_root", Path(self.data_root))
        object.__setattr__(self, "date_start", date_start)
        object.__setattr__(self, "date_end", date_end)
        if label not in VALID_LABELS:
            raise DatasetConfigurationError(
                f"label must be {sorted(VALID_LABELS)} ; received {label!r}"
            )
        if cp_flag not in VALID_CP_FLAGS:
            raise DatasetConfigurationError("cp_flag must be 'C' or 'P'")
        if symbol_start not in VALID_SYMBOL_STARTS:
            raise DatasetConfigurationError(f"symbol_start must be {sorted(VALID_SYMBOL_STARTS)}  ")
        if self.expiry_weekday is not None:
            num_weekday = _get_positive_int(
                self.expiry_weekday, "expiry_weekday", is_allow_zero=True
            )
            if num_weekday > 6:
                raise DatasetConfigurationError("expiry_weekday must be in [0, 6]")
            object.__setattr__(self, "expiry_weekday", num_weekday)
        object.__setattr__(
            self, "num_interval", _get_positive_int(self.num_interval, "num_interval")
        )
        object.__setattr__(
            self, "num_moneyness", _get_positive_int(self.num_moneyness, "num_moneyness")
        )
        object.__setattr__(
            self,
            "num_iv_solver_iterations",
            _get_positive_int(self.num_iv_solver_iterations, "num_iv_solver_iterations"),
        )
        object.__setattr__(
            self,
            "num_iv_surface_points",
            _get_positive_int(self.num_iv_surface_points, "num_iv_surface_points"),
        )
        if self.num_iv_surface_points < 2:
            raise DatasetConfigurationError("num_iv_surface_points must be at least 2")
        object.__setattr__(
            self,
            "num_iv_forward_fill_steps",
            _get_positive_int(
                self.num_iv_forward_fill_steps, "num_iv_forward_fill_steps", is_allow_zero=True
            ),
        )
        object.__setattr__(
            self,
            "num_scan_batch_size",
            _get_positive_int(self.num_scan_batch_size, "num_scan_batch_size"),
        )
        for name in (
            "max_relative_spread",
            "min_open_interest",
            "min_iv",
            "max_iv",
            "iv_solver_tolerance",
            "max_iv_extrapolation_log_moneyness",
        ):
            object.__setattr__(self, name, _get_finite_float(getattr(self, name), name))
        if self.max_relative_spread < 0:
            raise DatasetConfigurationError("max_relative_spread must be nonnegative")
        if self.min_open_interest < 0:
            raise DatasetConfigurationError("min_open_interest must be nonnegative")
        if not 0 < self.min_iv < self.max_iv:
            raise DatasetConfigurationError("IV bounds must satisfy 0 < min_iv < max_iv")
        if self.iv_solver_tolerance <= 0:
            raise DatasetConfigurationError("iv_solver_tolerance must be greater than 0")
        if self.max_iv_extrapolation_log_moneyness < 0:
            raise DatasetConfigurationError(
                "max_iv_extrapolation_log_moneyness must be nonnegative"
            )
        moneyness_lower = _get_finite_float(self.moneyness_lower, "moneyness_lower")
        moneyness_upper = _get_finite_float(self.moneyness_upper, "moneyness_upper")
        object.__setattr__(self, "moneyness_lower", moneyness_lower)
        object.__setattr__(self, "moneyness_upper", moneyness_upper)
        if not 0 < moneyness_lower <= 1 <= moneyness_upper:
            raise DatasetConfigurationError("The moneyness bounds must include 1.0")
        if moneyness_lower >= moneyness_upper:
            raise DatasetConfigurationError("moneyness_lower must be less than moneyness_upper")
        boolean_names = (
            "is_use_option_price_for_iv",
            "is_use_same_strike_opposite_iv",
            "is_use_surface_iv",
            "is_use_past_iv",
            "is_require_balanced_cohort",
            "is_allow_empty",
            "is_require_environment_ready",
        )
        for name in boolean_names:
            value = getattr(self, name)
            if not isinstance(value, (bool, np.bool_)):
                raise DatasetConfigurationError(f"{name} must be a boolean")
            object.__setattr__(self, name, bool(value))
        if self.is_require_environment_ready and symbol_start != "SPXW":
            raise DatasetConfigurationError(
                "Environment-ready episodes require PM settlement; other products require is_require_environment_ready=False"
            )

    @property
    def years(self) -> tuple[int, ...]:
        """Years."""
        return tuple(range(self.date_start.year, self.date_end.year + 1))

    def get_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable configuration dictionary."""
        result = asdict(self)
        result["date_start"] = self.date_start.strftime("%Y-%m-%d")
        result["date_end"] = self.date_end.strftime("%Y-%m-%d")
        result["data_root"] = str(self.data_root)
        return result


def get_option_dataset_config(
    *,
    date_period: Sequence[Any],
    label: str,
    data_root: str | Path = "data",
    cp_flag: str = "C",
    symbol_start: str = "SPXW",
    expiry_weekday: int | None = None,
    num_interval: int = 20,
    num_moneyness: int = 3,
    moneyness_range: Sequence[float] = (0.95, 1.05),
    **kwargs: Any,
) -> OptionDatasetConfig:
    """Return option dataset config."""
    date_start, date_end = get_date_period(date_period)
    moneyness_lower, moneyness_upper = get_moneyness_range(moneyness_range)
    return OptionDatasetConfig(
        date_start=date_start,
        date_end=date_end,
        label=label,
        data_root=Path(data_root),
        cp_flag=cp_flag,
        symbol_start=symbol_start,
        expiry_weekday=expiry_weekday,
        num_interval=num_interval,
        num_moneyness=num_moneyness,
        moneyness_lower=moneyness_lower,
        moneyness_upper=moneyness_upper,
        **kwargs,
    )
