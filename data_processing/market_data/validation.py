"""Validation for the paper option-hedging pipeline."""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from .constants import NUMERIC_COLUMNS, OPTION_COLUMNS, SPX_SECID, STRING_COLUMNS
from .storage import year_from_path


@dataclass
class FileValidationReport:
    """File validation report."""

    path: Path
    year: int | None
    rows: int = 0
    row_groups: int = 0
    min_date: pd.Timestamp | None = None
    max_date: pd.Timestamp | None = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """Ok."""
        return not self.errors


def _timestamp(value: object) -> pd.Timestamp | None:
    """Timestamp."""
    if value is None:
        return None
    if isinstance(value, (date, datetime, pd.Timestamp)):
        return pd.Timestamp(value)
    return None


def _date_bounds(parquet: pq.ParquetFile) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    """Date bounds."""
    index = parquet.schema_arrow.get_field_index("date")
    if index < 0:
        return (None, None)
    minima: list[pd.Timestamp] = []
    maxima: list[pd.Timestamp] = []
    for group_index in range(parquet.metadata.num_row_groups):
        stats = parquet.metadata.row_group(group_index).column(index).statistics
        if stats is None or not stats.has_min_max:
            continue
        minimum = _timestamp(stats.min)
        maximum = _timestamp(stats.max)
        if minimum is not None:
            minima.append(minimum)
        if maximum is not None:
            maxima.append(maximum)
    return (min(minima) if minima else None, max(maxima) if maxima else None)


def _validate_batches(
    parquet: pq.ParquetFile, report: FileValidationReport, expected_secid: int
) -> None:
    """Validate batches."""
    columns = ["secid", "date", "exdate", "cp_flag", "strike_price", "best_bid", "best_offer"]
    if any((column not in parquet.schema_arrow.names for column in columns)):
        return
    counts = {
        "wrong_secid": 0,
        "bad_cp_flag": 0,
        "bad_dates": 0,
        "bad_strike": 0,
        "negative_quote": 0,
        "crossed_quote": 0,
    }
    for batch in parquet.iter_batches(columns=columns, batch_size=131072):
        frame = batch.to_pandas()
        counts["wrong_secid"] += int(frame["secid"].isna().sum())
        counts["wrong_secid"] += int((frame["secid"].dropna() != expected_secid).sum())
        counts["bad_cp_flag"] += int((~frame["cp_flag"].isin(["C", "P"])).sum())
        counts["bad_dates"] += int(
            (
                frame["date"].isna() | frame["exdate"].isna() | (frame["exdate"] < frame["date"])
            ).sum()
        )
        counts["bad_strike"] += int(
            (frame["strike_price"].isna() | (frame["strike_price"] <= 0)).sum()
        )
        counts["negative_quote"] += int(
            (
                frame["best_bid"].isna()
                | frame["best_offer"].isna()
                | (frame["best_bid"] < 0)
                | (frame["best_offer"] < 0)
            ).sum()
        )
        counts["crossed_quote"] += int((frame["best_bid"] > frame["best_offer"]).sum())
    labels = {
        "wrong_secid": f"rows whose secid is not {expected_secid}",
        "bad_cp_flag": "rows whose cp_flag is not C or P",
        "bad_dates": "rows with null dates or expiration before quote date",
        "bad_strike": "rows with null or non-positive strike",
        "negative_quote": "rows with null or negative bid/ask",
    }
    for key, label in labels.items():
        if counts[key]:
            report.errors.append(f"{counts[key]:,} {label}")
    if counts["crossed_quote"]:
        report.warnings.append(f"{counts['crossed_quote']:,} rows have bid greater than ask")


def validate_file(
    path: str | Path,
    *,
    expected_year: int | None = None,
    expected_secid: int = SPX_SECID,
    full: bool = False,
) -> FileValidationReport:
    """Validate file."""
    path = Path(path)
    if expected_year is None:
        try:
            expected_year = year_from_path(path)
        except ValueError:
            expected_year = None
    report = FileValidationReport(path=path, year=expected_year)
    try:
        parquet = pq.ParquetFile(path)
    except Exception as exc:
        report.errors.append(f"cannot open parquet: {type(exc).__name__}: {exc}")
        return report
    report.rows = parquet.metadata.num_rows
    report.row_groups = parquet.metadata.num_row_groups
    if report.rows == 0:
        report.errors.append("file contains no rows")
    schema = parquet.schema_arrow
    missing = [column for column in OPTION_COLUMNS if column not in schema.names]
    if missing:
        report.errors.append(f"missing required columns: {missing}")
    for column in NUMERIC_COLUMNS:
        if column not in schema.names:
            continue
        column_type = schema.field(column).type
        if not (pa.types.is_integer(column_type) or pa.types.is_floating(column_type)):
            report.errors.append(f"{column} has non-numeric type {column_type}")
    for column in STRING_COLUMNS:
        if column not in schema.names:
            continue
        if not pa.types.is_string(schema.field(column).type):
            report.errors.append(f"{column} has non-string type {schema.field(column).type}")
    date_is_temporal = "date" in schema.names and (
        pa.types.is_timestamp(schema.field("date").type)
        or pa.types.is_date(schema.field("date").type)
    )
    exdate_is_temporal = "exdate" in schema.names and (
        pa.types.is_timestamp(schema.field("exdate").type)
        or pa.types.is_date(schema.field("exdate").type)
    )
    if "date" in schema.names and (not date_is_temporal):
        report.errors.append(f"date has non-temporal type {schema.field('date').type}")
    if "exdate" in schema.names and (not exdate_is_temporal):
        report.errors.append(f"exdate has non-temporal type {schema.field('exdate').type}")
    report.min_date, report.max_date = _date_bounds(parquet)
    if expected_year is not None and report.min_date is not None and (report.max_date is not None):
        if report.min_date.year != expected_year or report.max_date.year != expected_year:
            report.errors.append(
                f"quote-date range {report.min_date.date()}..{report.max_date.date()} does not stay within {expected_year}"
            )
    if full and (not missing) and date_is_temporal and exdate_is_temporal:
        try:
            _validate_batches(parquet, report, int(expected_secid))
        except Exception as exc:
            report.errors.append(f"row-level validation failed: {type(exc).__name__}: {exc}")
    return report
