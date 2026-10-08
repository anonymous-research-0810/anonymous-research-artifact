"""Acquisition for the paper option-hedging pipeline."""

from __future__ import annotations
import re
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any
import pandas as pd
from .constants import OPTION_COLUMNS

_SQL_IDENTIFIER = re.compile("^[A-Za-z_][A-Za-z0-9_$]*$")
_YEARLY_TABLE = re.compile("^opprcd\\d{4}$")


def sql_identifier(value: str) -> str:
    """Sql identifier."""
    if not _SQL_IDENTIFIER.fullmatch(value):
        raise ValueError(f"unsafe SQL identifier: {value!r}")
    return value


def option_table(year: int, *, unified: bool) -> str:
    """Option table."""
    year = int(year)
    if not 1900 <= year <= 2200:
        raise ValueError(f"invalid calendar year: {year}")
    return "opprcd" if unified else f"opprcd{year}"


def build_option_query(
    library: str,
    year: int,
    secid: int,
    *,
    unified: bool,
    limit: int | None = None,
    columns: Sequence[str] = OPTION_COLUMNS,
) -> str:
    """Build option query."""
    library = sql_identifier(library)
    selected = [sql_identifier(column) for column in columns]
    if not selected:
        raise ValueError("at least one column is required")
    year = int(year)
    table = option_table(year, unified=unified)
    secid = int(secid)
    if secid <= 0:
        raise ValueError("secid must be positive")
    if limit is not None:
        limit = int(limit)
        if limit <= 0:
            raise ValueError("limit must be positive")
    sql = f"SELECT {', '.join(selected)} FROM {library}.{table} WHERE secid = {secid} AND date BETWEEN '{year}-01-01' AND '{year}-12-31' ORDER BY date, exdate, strike_price"
    return f"{sql} LIMIT {limit}" if limit is not None else sql


def discover_option_library(db: Any) -> str:
    """Discover option library."""
    libraries = sorted((lib for lib in db.list_libraries() if "option" in lib.lower()))
    if not libraries:
        raise RuntimeError("no OptionMetrics library is available for this WRDS account")
    return "optionm" if "optionm" in libraries else libraries[0]


def discover_table_layout(db: Any, library: str) -> tuple[bool, list[str]]:
    """Discover table layout."""
    library = sql_identifier(library)
    tables = sorted(
        (table for table in db.list_tables(library=library) if table.startswith("opprcd"))
    )
    if "opprcd" in tables:
        return (True, tables)
    if any((_YEARLY_TABLE.fullmatch(table) for table in tables)):
        return (False, tables)
    raise RuntimeError(f"no OptionMetrics option-price tables found in {library}")


def atomic_write_parquet(frame: pd.DataFrame, destination: str | Path) -> Path:
    """Atomic write parquet."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=f".{destination.stem}.",
            suffix=".tmp.parquet",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
        frame.to_parquet(temporary, compression="zstd", index=False)
        temporary.replace(destination)
        return destination
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
