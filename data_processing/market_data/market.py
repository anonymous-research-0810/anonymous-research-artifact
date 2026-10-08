"""Market for the paper option-hedging pipeline."""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
from .constants import MARKET_COLUMNS, SPX_SECID

SPOT_FILE = "spx_spot.parquet"
ZERO_CURVE_FILE = "zero_curve.parquet"
DIVIDEND_FILE = "spx_div_yield.parquet"
_PERCENT_TO_DECIMAL = 0.01
_NANOSECONDS_PER_DAY = 86400000000000


def _require_columns(frame: pd.DataFrame, required: tuple[str, ...], source: Path) -> None:
    """Require columns."""
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{source} is missing required fields: {missing}")


def _date_values(series: pd.Series, source: Path) -> np.ndarray:
    """Date values."""
    if series.isna().any():
        raise ValueError(f"{source} date field contains missing values")
    try:
        dates = pd.to_datetime(series, errors="raise").to_numpy(dtype="datetime64[ns]")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source} date field cannot be parsed as dates: {exc}") from exc
    return dates.astype(np.int64)


def _finite_values(
    series: pd.Series, source: Path, column: str, *, positive: bool = False
) -> np.ndarray:
    """Finite values."""
    try:
        values = pd.to_numeric(series, errors="raise").to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source} {column} field contains invalid numeric values: {exc}") from exc
    if not np.isfinite(values).all():
        raise ValueError(f"{source} {column} field contains missing or nonfinite values")
    if positive and np.any(values <= 0):
        raise ValueError(f"{source} {column} field values must all be positive")
    return values


def _exact_lookup(
    reference_dates: np.ndarray,
    reference_values: np.ndarray,
    query_dates: np.ndarray,
    value_name: str,
) -> np.ndarray:
    """Exact lookup."""
    positions = np.searchsorted(reference_dates, query_dates)
    within_bounds = positions < len(reference_dates)
    safe_positions = np.minimum(positions, len(reference_dates) - 1)
    matched = within_bounds & (reference_dates[safe_positions] == query_dates)
    if not matched.all():
        missing = np.unique(query_dates[~matched])
        labels = pd.to_datetime(missing[:5]).strftime("%Y-%m-%d").tolist()
        raise ValueError(f"{value_name} is missing option quote dates: {labels}")
    return reference_values[positions]


@dataclass(frozen=True)
class MarketData:
    """Aligned spot, dividend, and zero-curve tables with maturity interpolation."""

    dates: np.ndarray
    spots: np.ndarray
    dividend_rates: np.ndarray
    zero_curves: dict[int, tuple[np.ndarray, np.ndarray]]

    def enrich_batch(self, batch: pa.RecordBatch, output_schema: pa.Schema) -> pa.RecordBatch:
        """Enrich batch."""
        date_index = batch.schema.get_field_index("date")
        exdate_index = batch.schema.get_field_index("exdate")
        if date_index < 0 or exdate_index < 0:
            raise ValueError("Market enrichment requires date and exdate in the option batch")
        date_array = batch.column(date_index)
        exdate_array = batch.column(exdate_index)
        if date_array.null_count or exdate_array.null_count:
            raise ValueError("The option batch contains missing date or exdate values")
        quote_dates = (
            date_array.to_numpy(zero_copy_only=False).astype("datetime64[ns]").astype(np.int64)
        )
        expiration_dates = (
            exdate_array.to_numpy(zero_copy_only=False).astype("datetime64[ns]").astype(np.int64)
        )
        elapsed = expiration_dates - quote_dates
        if np.any(elapsed < 0) or np.any(elapsed % _NANOSECONDS_PER_DAY != 0):
            raise ValueError(
                "Option expiry must be a whole calendar date on or after the quote date"
            )
        dte = elapsed // _NANOSECONDS_PER_DAY
        spots = _exact_lookup(self.dates, self.spots, quote_dates, "spot")
        dividends = _exact_lookup(self.dates, self.dividend_rates, quote_dates, "dividend_rate")
        zero_rates = np.empty(len(batch), dtype=np.float64)
        for quote_date in np.unique(quote_dates):
            curve = self.zero_curves.get(int(quote_date))
            if curve is None:
                label = pd.Timestamp(quote_date).strftime("%Y-%m-%d")
                raise ValueError(f"zero_rate is missing option quote dates: {label}")
            mask = quote_dates == quote_date
            curve_days, curve_rates = curve
            zero_rates[mask] = np.interp(dte[mask], curve_days, curve_rates)
        raw_arrays = {name: batch.column(name) for name in batch.schema.names}
        market_arrays = {
            "spot": pa.array(spots, type=pa.float64()),
            "zero_rate": pa.array(zero_rates, type=pa.float64()),
            "dividend_rate": pa.array(dividends, type=pa.float64()),
        }
        arrays = [
            raw_arrays[column] if column in raw_arrays else market_arrays[column]
            for column in output_schema.names
        ]
        return pa.RecordBatch.from_arrays(arrays, schema=output_schema)


def load_market_data(data_root: str | Path = "data", *, secid: int = SPX_SECID) -> MarketData:
    """Load market data."""
    root = Path(data_root)
    spot_path = root / SPOT_FILE
    zero_path = root / ZERO_CURVE_FILE
    dividend_path = root / DIVIDEND_FILE
    for path in (spot_path, zero_path, dividend_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing auxiliary market data file: {path}")
    spot = pd.read_parquet(spot_path)
    dividend = pd.read_parquet(dividend_path)
    zero = pd.read_parquet(zero_path)
    _require_columns(spot, ("secid", "date", "close"), spot_path)
    _require_columns(dividend, ("secid", "date", "rate"), dividend_path)
    _require_columns(zero, ("date", "days", "rate"), zero_path)
    spot = spot.loc[spot["secid"] == int(secid), ["date", "close"]].copy()
    dividend = dividend.loc[dividend["secid"] == int(secid), ["date", "rate"]].copy()
    if spot.empty or dividend.empty:
        raise ValueError(f"No auxiliary market data were found for secid={int(secid)}")
    if spot.duplicated("date").any() or dividend.duplicated("date").any():
        raise ValueError("spot or dividend_rate contains duplicate secid/date records")
    if zero.duplicated(["date", "days"]).any():
        raise ValueError("zero_curve contains duplicate date/days records")
    spot.sort_values("date", inplace=True)
    dividend.sort_values("date", inplace=True)
    zero.sort_values(["date", "days"], inplace=True)
    spot_dates = _date_values(spot["date"], spot_path)
    dividend_dates = _date_values(dividend["date"], dividend_path)
    zero_dates = _date_values(zero["date"], zero_path)
    unique_zero_dates = np.unique(zero_dates)
    if not np.array_equal(spot_dates, dividend_dates) or not np.array_equal(
        spot_dates, unique_zero_dates
    ):
        raise ValueError("spot, zero_curve, and dividend_rate date sets differ")
    spots = _finite_values(spot["close"], spot_path, "close", positive=True)
    dividend_rates = _finite_values(dividend["rate"], dividend_path, "rate") * _PERCENT_TO_DECIMAL
    zero_days = _finite_values(zero["days"], zero_path, "days", positive=True)
    zero_rates = _finite_values(zero["rate"], zero_path, "rate") * _PERCENT_TO_DECIMAL
    zero_curves: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for quote_date in unique_zero_dates:
        mask = zero_dates == quote_date
        days = zero_days[mask]
        rates = zero_rates[mask]
        if len(days) == 0 or np.any(np.diff(days) <= 0):
            label = pd.Timestamp(quote_date).strftime("%Y-%m-%d")
            raise ValueError(f"{label} zero-curve maturity nodes are not strictly increasing")
        zero_curves[int(quote_date)] = (days, rates)
    return MarketData(
        dates=spot_dates, spots=spots, dividend_rates=dividend_rates, zero_curves=zero_curves
    )


def market_schema_fields() -> dict[str, pa.Field]:
    """Market schema fields."""
    return {column: pa.field(column, pa.float64()) for column in MARKET_COLUMNS}
