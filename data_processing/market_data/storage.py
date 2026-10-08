"""Storage for the paper option-hedging pipeline."""

from __future__ import annotations
import re
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
from .constants import FILE_TEMPLATE, MARKET_COLUMNS
from .market import MarketData, load_market_data, market_schema_fields

_YEAR_FILE_RE = re.compile("^spx_options_(\\d{4})\\.parquet$")


def _check_year(year: int) -> int:
    """Check year."""
    year = int(year)
    if not 1900 <= year <= 2200:
        raise ValueError(f"invalid calendar year: {year}")
    return year


def year_parquet_path(data_root: str | Path, year: int) -> Path:
    """Year parquet path."""
    return Path(data_root) / FILE_TEMPLATE.format(year=_check_year(year))


def year_from_path(path: str | Path) -> int:
    """Year from path."""
    name = Path(path).name
    match = _YEAR_FILE_RE.fullmatch(name)
    if match is None:
        raise ValueError(f"not a canonical option filename: {name}")
    return int(match.group(1))


def discover_option_files(
    data_root: str | Path = "data", *, start: int | None = None, end: int | None = None
) -> dict[int, Path]:
    """Discover option files."""
    if start is not None:
        start = _check_year(start)
    if end is not None:
        end = _check_year(end)
    if start is not None and end is not None and (start > end):
        raise ValueError(f"start ({start}) is after end ({end})")
    root = Path(data_root)
    found: dict[int, Path] = {}
    if not root.is_dir():
        return found
    for path in root.glob("spx_options_*.parquet"):
        try:
            year = year_from_path(path)
        except ValueError:
            continue
        if start is not None and year < start:
            continue
        if end is not None and year > end:
            continue
        found[year] = path
    return dict(sorted(found.items()))


def missing_years(data_root: str | Path, start: int, end: int) -> list[int]:
    """Missing years."""
    start = _check_year(start)
    end = _check_year(end)
    if start > end:
        raise ValueError(f"start ({start}) is after end ({end})")
    present = discover_option_files(data_root, start=start, end=end)
    return [year for year in range(start, end + 1) if year not in present]


def years_to_pull(data_root: str | Path, start: int, end: int, *, force: bool = False) -> list[int]:
    """Years to pull."""
    start = _check_year(start)
    end = _check_year(end)
    if start > end:
        raise ValueError(f"start ({start}) is after end ({end})")
    if force:
        return list(range(start, end + 1))
    return missing_years(data_root, start, end)


def _paths_for_years(data_root: str | Path, years: Sequence[int] | None) -> list[Path]:
    """Paths for years."""
    found = discover_option_files(data_root)
    if years is None:
        paths = list(found.values())
    else:
        requested = [_check_year(year) for year in years]
        absent = [year for year in requested if year not in found]
        if absent:
            raise FileNotFoundError(f"missing annual option files for years: {absent}")
        paths = [found[year] for year in requested]
    if not paths:
        raise FileNotFoundError(f"no spx_options_YYYY.parquet files under {Path(data_root)}")
    return paths


def read_year(
    year: int,
    data_root: str | Path = "data",
    *,
    columns: Sequence[str] | None = None,
    filters: Any = None,
) -> pd.DataFrame:
    """Read year."""
    path = year_parquet_path(data_root, year)
    if not path.is_file():
        raise FileNotFoundError(path)
    selected = list(columns) if columns is not None else None
    return pd.read_parquet(path, columns=selected, filters=filters)


def iter_years(
    data_root: str | Path = "data",
    *,
    years: Sequence[int] | None = None,
    columns: Sequence[str] | None = None,
    filters: Any = None,
) -> Iterator[tuple[int, pd.DataFrame]]:
    """Iter years."""
    for path in _paths_for_years(data_root, years):
        yield (
            year_from_path(path),
            pd.read_parquet(
                path, columns=list(columns) if columns is not None else None, filters=filters
            ),
        )


def option_dataset(
    data_root: str | Path = "data", *, years: Sequence[int] | None = None
) -> ds.Dataset:
    """Option dataset."""
    paths = [str(path) for path in _paths_for_years(data_root, years)]
    return ds.dataset(paths, format="parquet")


class OptionScanner:
    """Stream yearly option tables with column projection, filtering, and optional market enrichment."""

    def __init__(
        self, scanner: ds.Scanner, output_columns: tuple[str, ...], market_data: MarketData
    ) -> None:
        """Initialize validated configuration and internal state."""
        self._scanner = scanner
        self._market_data = market_data
        raw_fields = {field.name: field for field in scanner.projected_schema}
        market_fields = market_schema_fields()
        self._projected_schema = pa.schema(
            [
                raw_fields[column] if column in raw_fields else market_fields[column]
                for column in output_columns
            ],
            metadata=scanner.projected_schema.metadata,
        )

    @property
    def projected_schema(self) -> pa.Schema:
        """Projected schema."""
        return self._projected_schema

    @property
    def dataset_schema(self) -> pa.Schema:
        """Dataset schema."""
        return self._scanner.dataset_schema

    def _enrich_table(self, table: pa.Table) -> pa.Table:
        """Enrich table."""
        batches = [
            self._market_data.enrich_batch(batch, self._projected_schema)
            for batch in table.to_batches()
        ]
        return pa.Table.from_batches(batches, schema=self._projected_schema)

    def to_batches(self) -> Iterator[pa.RecordBatch]:
        """To batches."""
        for batch in self._scanner.to_batches():
            yield self._market_data.enrich_batch(batch, self._projected_schema)

    def scan_batches(self) -> Iterator[ds.TaggedRecordBatch]:
        """Scan batches."""
        for tagged_batch in self._scanner.scan_batches():
            enriched = self._market_data.enrich_batch(
                tagged_batch.record_batch, self._projected_schema
            )
            yield ds.TaggedRecordBatch(enriched, tagged_batch.fragment)

    def to_table(self) -> pa.Table:
        """To table."""
        return pa.Table.from_batches(self.to_batches(), schema=self._projected_schema)

    def to_reader(self) -> pa.RecordBatchReader:
        """To reader."""
        return pa.RecordBatchReader.from_batches(self._projected_schema, self.to_batches())

    def count_rows(self) -> int:
        """Count rows."""
        return self._scanner.count_rows()

    def head(self, num_rows: int) -> pa.Table:
        """Head."""
        if num_rows < 0:
            raise ValueError("num_rows must not be negative")
        return self._enrich_table(self._scanner.head(num_rows))

    def take(self, indices: Any) -> pa.Table:
        """Take."""
        return self._enrich_table(self._scanner.take(indices))


def scan_options(
    data_root: str | Path = "data",
    *,
    years: Sequence[int] | None = None,
    columns: Sequence[str] | None = None,
    filter: ds.Expression | None = None,
    batch_size: int = 131072,
    enrich_market_data: bool = True,
) -> OptionScanner | ds.Scanner:
    """Scan options."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    dataset = option_dataset(data_root, years=years)
    raw_names = list(dataset.schema.names)
    requested = raw_names if columns is None else list(columns)
    if len(requested) != len(set(requested)):
        raise ValueError("columns must not contain duplicates")
    allowed = set(raw_names) | (set(MARKET_COLUMNS) if enrich_market_data else set())
    unknown = [column for column in requested if column not in allowed]
    if unknown:
        raise ValueError(f"unknown scan columns: {unknown}")
    if not enrich_market_data:
        return dataset.scanner(columns=requested, filter=filter, batch_size=batch_size)
    output_columns = tuple(
        [*requested, *(column for column in MARKET_COLUMNS if column not in requested)]
    )
    base_columns = [column for column in requested if column in raw_names]
    for helper in ("date", "exdate"):
        if helper not in base_columns:
            base_columns.append(helper)
    scanner = dataset.scanner(columns=base_columns, filter=filter, batch_size=batch_size)
    market_data = load_market_data(data_root)
    return OptionScanner(scanner, output_columns, market_data)
