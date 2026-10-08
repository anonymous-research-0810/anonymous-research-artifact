"""Features for the paper option-hedging pipeline."""

from __future__ import annotations
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from option_dataset import OptionDataset
from .config import SegmentationConfig


@dataclass(frozen=True)
class SegmentationFeatures:
    """Segmentation features."""

    daily_spot: pd.DataFrame
    feature_frame: pd.DataFrame
    signal: np.ndarray
    normalization: dict[str, dict[str, float]]
    realized_volatility_window: int

    def get_record(self) -> dict[str, Any]:
        """Return a JSON-serializable record of the result and its metadata."""
        first_feature_date = pd.Timestamp(self.feature_frame["date"].iloc[0])
        last_feature_date = pd.Timestamp(self.feature_frame["date"].iloc[-1])
        first_spot_date = pd.Timestamp(self.daily_spot["date"].iloc[0])
        last_spot_date = pd.Timestamp(self.daily_spot["date"].iloc[-1])
        date_bytes = (
            self.daily_spot["date"]
            .to_numpy(dtype="datetime64[ns]")
            .astype("<i8", copy=False)
            .tobytes()
        )
        spot_bytes = self.daily_spot["spot"].to_numpy(dtype="<f8", copy=True).tobytes()
        signal_bytes = np.ascontiguousarray(self.signal, dtype="<f8").tobytes()
        return {
            "num_daily_spot": int(len(self.daily_spot)),
            "num_feature_observation": int(len(self.feature_frame)),
            "spot_date_start": first_spot_date.strftime("%Y-%m-%d"),
            "spot_date_end": last_spot_date.strftime("%Y-%m-%d"),
            "feature_date_start": first_feature_date.strftime("%Y-%m-%d"),
            "feature_date_end": last_feature_date.strftime("%Y-%m-%d"),
            "realized_volatility_window": self.realized_volatility_window,
            "feature_columns": ["log_return_zscore", "log_realized_volatility_zscore"],
            "normalization": self.normalization,
            "daily_spot_sha256": hashlib.sha256(date_bytes + spot_bytes).hexdigest(),
            "signal_sha256": hashlib.sha256(signal_bytes).hexdigest(),
        }


def get_daily_spot(dataset: OptionDataset) -> pd.DataFrame:
    """Return daily spot."""
    if not isinstance(dataset, OptionDataset):
        raise TypeError("dataset must be an OptionDataset instance")
    if dataset.episode_steps.empty:
        raise ValueError("Cannot construct a daily spot series from an empty OptionDataset")
    raw = dataset.episode_steps.loc[:, ["date", "spot"]].copy(deep=True)
    try:
        raw["date"] = pd.to_datetime(raw["date"], errors="raise").dt.normalize()
    except (TypeError, ValueError) as exc:
        raise ValueError("episode_steps.date contains unparseable dates") from exc
    if raw["date"].isna().any():
        raise ValueError("episode_steps.date must not contain missing values")
    raw["spot"] = pd.to_numeric(raw["spot"], errors="coerce")
    if not np.isfinite(raw["spot"].to_numpy(dtype=np.float64)).all():
        raise ValueError("episode_steps.spot must contain finite values only")
    if not raw["spot"].gt(0.0).all():
        raise ValueError("episode_steps.spot must contain strictly positive values only")
    rows: list[dict[str, Any]] = []
    for date, group in raw.groupby("date", sort=True, observed=True):
        values = group["spot"].to_numpy(dtype=np.float64)
        reference = float(values[0])
        if not np.allclose(values, reference, rtol=1e-12, atol=1e-10):
            raise ValueError(
                f"The same trading date {pd.Timestamp(date):%Y-%m-%d} has conflicting spot values"
            )
        rows.append({"date": pd.Timestamp(date), "spot": reference})
    daily = pd.DataFrame(rows, columns=["date", "spot"])
    if len(daily) < 2:
        raise ValueError("At least two distinct trading days are required for spot returns")
    if not daily["date"].is_monotonic_increasing or daily["date"].duplicated().any():
        raise RuntimeError(
            "Internal error: daily spot dates are not strictly increasing and unique"
        )
    daily["log_return"] = np.log(daily["spot"]).diff()
    return daily


def get_daily_spot_from_market(
    data_root: str | Path, date_period: tuple[Any, Any] | list[Any]
) -> pd.DataFrame:
    """Read the full raw daily spot series so dates without retained option episodes still contribute to regime detection."""
    try:
        if len(date_period) != 2:
            raise ValueError("date_period must contain a start and end date")
        date_start = pd.Timestamp(date_period[0]).normalize()
        date_end = pd.Timestamp(date_period[1]).normalize()
    except (TypeError, ValueError) as exc:
        raise ValueError("date_period must be a parseable start/end date pair") from exc
    if pd.isna(date_start) or pd.isna(date_end) or date_start > date_end:
        raise ValueError("date_period start/end dates are invalid")
    root = Path(data_root)
    path = root / "spx_spot.parquet"
    if not path.is_file():
        raw_sx5e_path = root / "sx5e_spot.parquet"
        if raw_sx5e_path.is_file():
            path = raw_sx5e_path
        else:
            raise FileNotFoundError(
                f"Raw daily spot file does not exist: {root / 'spx_spot.parquet'} or {raw_sx5e_path}"
            )
    frame = pd.read_parquet(path)
    required_date = "date"
    if required_date not in frame.columns:
        raise ValueError(f"{path} is missing required columns: date")
    if "close" in frame.columns:
        spot_column = "close"
    elif "spot" in frame.columns:
        spot_column = "spot"
    else:
        raise ValueError(f"{path} is missing the daily spot column close/spot")
    raw = frame.loc[:, [required_date, spot_column]].copy()
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce").dt.normalize()
    raw["spot"] = pd.to_numeric(raw[spot_column], errors="coerce")
    if raw["date"].isna().any():
        raise ValueError(f"{path} date contains missing or unparseable values")
    raw = raw.loc[raw["date"].between(date_start, date_end, inclusive="both")]
    if raw.empty:
        raise ValueError(
            f"The raw spot file has no trading days in the requested period [{date_start:%Y-%m-%d}, {date_end:%Y-%m-%d}] with no trading days"
        )
    if not np.isfinite(raw["spot"].to_numpy(dtype=np.float64)).all():
        raise ValueError(
            f"{path} spot within the requested interval must contain finite values only"
        )
    if not raw["spot"].gt(0.0).all():
        raise ValueError(
            f"{path} spot within the requested interval must contain strictly positive values only"
        )
    rows: list[dict[str, Any]] = []
    for date, group in raw.groupby("date", sort=True, observed=True):
        values = group["spot"].to_numpy(dtype=np.float64)
        reference = float(values[0])
        if not np.allclose(values, reference, rtol=1e-12, atol=1e-10):
            raise ValueError(
                f"Raw spot file date {pd.Timestamp(date):%Y-%m-%d} has conflicting spot values"
            )
        rows.append({"date": pd.Timestamp(date), "spot": reference})
    daily = pd.DataFrame(rows, columns=["date", "spot"])
    if len(daily) < 2:
        raise ValueError(
            "The requested period requires at least two distinct trading days for spot returns"
        )
    daily = daily.sort_values("date", kind="mergesort").reset_index(drop=True)
    if daily["date"].duplicated().any():
        raise RuntimeError("Internal error: raw spot dates are not unique")
    daily["log_return"] = np.log(daily["spot"]).diff()
    return daily


def get_segmentation_features(
    dataset: OptionDataset, config: SegmentationConfig, *, daily_spot: pd.DataFrame | None = None
) -> SegmentationFeatures:
    """Standardize training log returns and log realized volatility, using the episode horizon as the volatility window."""
    if not isinstance(config, SegmentationConfig):
        raise TypeError("config must be a SegmentationConfig instance")
    window = int(dataset.config.num_interval)
    if window <= 0:
        raise ValueError("dataset.config.num_interval must be strictly positive")
    if daily_spot is None:
        daily = get_daily_spot(dataset)
    else:
        if not isinstance(daily_spot, pd.DataFrame):
            raise TypeError("daily_spot must be a pandas.DataFrame")
        if not {"date", "spot"}.issubset(daily_spot.columns):
            raise ValueError("daily_spot must contain date and spot columns")
        daily = daily_spot.loc[:, ["date", "spot"]].copy(deep=True)
        daily["date"] = pd.to_datetime(daily["date"], errors="coerce").dt.normalize()
        daily["spot"] = pd.to_numeric(daily["spot"], errors="coerce")
        if daily["date"].isna().any():
            raise ValueError("daily_spot.date contains missing or unparseable values")
        if not np.isfinite(daily["spot"].to_numpy(dtype=np.float64)).all():
            raise ValueError("daily_spot.spot must contain finite values only")
        if not daily["spot"].gt(0.0).all():
            raise ValueError("daily_spot.spot must contain strictly positive values only")
        daily = daily.sort_values("date", kind="mergesort").reset_index(drop=True)
        if daily["date"].duplicated().any():
            raise ValueError("daily_spot.date must contain exactly one row per day")
        if len(daily) < 2:
            raise ValueError("At least two distinct trading days are required for spot returns")
        daily["log_return"] = np.log(daily["spot"]).diff()
    squared_return = daily["log_return"].pow(2)
    rolling_sum = squared_return.rolling(window=window, min_periods=window).sum()
    daily["realized_volatility"] = np.sqrt(
        float(config.annualization_days) * rolling_sum / float(window)
    )
    daily["log_realized_volatility"] = np.log(
        daily["realized_volatility"] + config.volatility_epsilon
    )
    feature_frame = daily.dropna(
        subset=["log_return", "realized_volatility", "log_realized_volatility"]
    ).copy(deep=True)
    if len(feature_frame) < 2 * config.min_segment_length:
        raise ValueError(
            f"Too few feature observations to form two minimum-length segments: num_feature={len(feature_frame)}, min_segment_length={config.min_segment_length}"
        )
    normalization: dict[str, dict[str, float]] = {}
    for raw_name, zscore_name in (
        ("log_return", "log_return_zscore"),
        ("log_realized_volatility", "log_realized_volatility_zscore"),
    ):
        values = feature_frame[raw_name].to_numpy(dtype=np.float64)
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=0))
        if not np.isfinite(mean) or not np.isfinite(std):
            raise ValueError(f"{raw_name} has a nonfinite mean or standard deviation")
        if std < config.min_feature_std:
            raise ValueError(
                f"{raw_name} standard deviation {std:.6g} is below min_feature_std; stable normalization is unavailable"
            )
        feature_frame[zscore_name] = (values - mean) / std
        normalization[raw_name] = {"mean": mean, "std": std}
    signal = feature_frame.loc[:, ["log_return_zscore", "log_realized_volatility_zscore"]].to_numpy(
        dtype=np.float64, copy=True
    )
    if signal.ndim != 2 or signal.shape[1] != 2:
        raise RuntimeError(
            "Internal error: PELT signal must be a two-dimensional array with two features"
        )
    if not np.isfinite(signal).all():
        raise ValueError("The standardized PELT signal contains nonfinite values")
    return SegmentationFeatures(
        daily_spot=daily.reset_index(drop=True),
        feature_frame=feature_frame.reset_index(drop=True),
        signal=signal,
        normalization=normalization,
        realized_volatility_window=window,
    )
