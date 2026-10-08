"""Cli for the paper option-hedging pipeline."""

from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path
from .acquisition import (
    atomic_write_parquet,
    build_option_query,
    discover_option_library,
    discover_table_layout,
    option_table,
    sql_identifier,
)
from .config import wrds_connect
from .constants import DEFAULT_END_YEAR, DEFAULT_START_YEAR, OPTION_COLUMNS, SPX_SECID
from .storage import discover_option_files, missing_years, year_parquet_path, years_to_pull
from .validation import FileValidationReport, validate_file


def _add_range(parser: argparse.ArgumentParser) -> None:
    """Add range."""
    parser.add_argument("--start", type=int, default=DEFAULT_START_YEAR)
    parser.add_argument("--end", type=int, default=DEFAULT_END_YEAR)


def build_parser() -> argparse.ArgumentParser:
    """Build parser."""
    parser = argparse.ArgumentParser(
        prog="market-option-data",
        description="Acquire, inspect, and validate yearly SPX OptionMetrics parquet files.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    inspect_parser = subparsers.add_parser("inspect", help="show local files and parquet metadata")
    inspect_parser.add_argument("--data-root", type=Path, default=Path("data"))
    _add_range(inspect_parser)
    validate_parser = subparsers.add_parser("validate", help="validate schema and data invariants")
    validate_parser.add_argument("--data-root", type=Path, default=Path("data"))
    _add_range(validate_parser)
    validate_parser.add_argument(
        "--quick", action="store_true", help="check metadata only; skip the row-level scan"
    )
    validate_parser.add_argument(
        "--allow-missing", action="store_true", help="do not fail when a requested year is absent"
    )
    probe_parser = subparsers.add_parser("probe", help="run bounded WRDS entitlement/schema checks")
    probe_parser.add_argument("--year", type=int, default=DEFAULT_START_YEAR)
    probe_parser.add_argument("--secid", type=int, default=SPX_SECID)
    probe_parser.add_argument("--library")
    probe_parser.add_argument("--dotenv", type=Path)
    pull_parser = subparsers.add_parser("pull", help="pull one parquet per year from WRDS")
    pull_parser.add_argument("--data-root", type=Path, default=Path("data"))
    _add_range(pull_parser)
    pull_parser.add_argument("--secid", type=int, default=SPX_SECID)
    pull_parser.add_argument("--library")
    pull_parser.add_argument("--dotenv", type=Path)
    pull_parser.add_argument("--force", action="store_true")
    pull_parser.add_argument(
        "--pause", type=float, default=5.0, help="seconds between yearly queries"
    )
    pull_parser.add_argument(
        "--limit",
        type=int,
        help="query at most this many rows and write *.sample.parquet, never a canonical file",
    )
    return parser


def _date_text(value: object) -> str:
    """Date text."""
    return "-" if value is None else str(value)[:10]


def _print_report(report: FileValidationReport) -> None:
    """Print report."""
    status = "OK" if report.ok else "ERROR"
    print(
        f"{report.year or '-':>4}  {report.rows:>10,} rows  {_date_text(report.min_date)}..{_date_text(report.max_date)}  {status}  {report.path}"
    )
    for message in report.errors:
        print(f"      error: {message}")
    for message in report.warnings:
        print(f"      warning: {message}")


def _cmd_inspect(args: argparse.Namespace) -> int:
    """Cmd inspect."""
    files = discover_option_files(args.data_root, start=args.start, end=args.end)
    absent = missing_years(args.data_root, args.start, args.end)
    print(f"data root: {args.data_root.resolve()}")
    print(f"available years: {list(files) or 'none'}")
    print(f"missing years: {absent or 'none'}")
    failed = False
    for year, path in files.items():
        report = validate_file(path, expected_year=year, full=False)
        _print_report(report)
        failed = failed or not report.ok
    return 1 if failed else 0


def _cmd_validate(args: argparse.Namespace) -> int:
    """Cmd validate."""
    files = discover_option_files(args.data_root, start=args.start, end=args.end)
    absent = missing_years(args.data_root, args.start, args.end)
    failed = bool(absent and (not args.allow_missing))
    if absent:
        print(f"missing years: {absent}", file=sys.stderr if failed else sys.stdout)
    if not files:
        print(f"no canonical parquet files found under {args.data_root}", file=sys.stderr)
        return 1
    for year, path in files.items():
        report = validate_file(path, expected_year=year, full=not args.quick)
        _print_report(report)
        failed = failed or not report.ok
    return 1 if failed else 0


def _cmd_probe(args: argparse.Namespace) -> int:
    """Cmd probe."""
    db = wrds_connect(args.dotenv)
    try:
        library = args.library or discover_option_library(db)
        library = sql_identifier(library)
        unified, tables = discover_table_layout(db, library)
        table = option_table(args.year, unified=unified)
        print(
            f"library={library}; layout={('unified' if unified else 'per-year')}; price_tables={len(tables)}"
        )
        names = db.raw_sql(
            f"SELECT DISTINCT secid, ticker, issuer FROM {library}.secnmd WHERE secid = {int(args.secid)} LIMIT 5"
        )
        if names.empty:
            print(f"secid {args.secid} was not found", file=sys.stderr)
            return 1
        print(names.to_string(index=False))
        described = db.describe_table(library=library, table=table)
        missing = [column for column in OPTION_COLUMNS if column not in set(described["name"])]
        if missing:
            print(f"{library}.{table} is missing columns: {missing}", file=sys.stderr)
            return 1
        query = build_option_query(library, args.year, args.secid, unified=unified, limit=5)
        sample = db.raw_sql(query, date_cols=["date", "exdate"])
        print(f"sample query returned {len(sample)} rows")
        print(sample.to_string(index=False))
        return 0
    finally:
        db.close()


def _cmd_pull(args: argparse.Namespace) -> int:
    """Cmd pull."""
    if args.pause < 0:
        raise ValueError("pause must not be negative")
    years = years_to_pull(args.data_root, args.start, args.end, force=args.force)
    if not years:
        print("nothing to pull; every requested year is already present")
        return 0
    print(f"years to pull: {years}")
    db = wrds_connect(args.dotenv)
    completed: list[int] = []
    try:
        library = args.library or discover_option_library(db)
        library = sql_identifier(library)
        unified, _ = discover_table_layout(db, library)
        print(f"library={library}; layout={('unified' if unified else 'per-year')}")
        for index, year in enumerate(years):
            if index:
                time.sleep(args.pause)
            query = build_option_query(library, year, args.secid, unified=unified, limit=args.limit)
            print(f"[{year}] querying", flush=True)
            started = time.monotonic()
            frame = db.raw_sql(query, date_cols=["date", "exdate"])
            if frame.empty:
                raise RuntimeError(f"query for {year} returned no rows")
            if args.limit is None:
                output = year_parquet_path(args.data_root, year)
            else:
                output = Path(args.data_root) / f"spx_options_{year}.sample.parquet"
            atomic_write_parquet(frame, output)
            elapsed = time.monotonic() - started
            print(
                f"[{year}] {len(frame):,} rows, {output.stat().st_size / 1000000.0:.1f} MB, {elapsed:.1f}s -> {output}"
            )
            completed.append(year)
        return 0
    except Exception as exc:
        print(f"pull failed after years {completed}: {type(exc).__name__}: {exc}", file=sys.stderr)
        print("No automatic retry was attempted; rerun the command to resume.", file=sys.stderr)
        return 1
    finally:
        db.close()


def main(argv: list[str] | None = None) -> int:
    """Parse command-line arguments, run the requested operation, and report failures."""
    args = build_parser().parse_args(argv)
    commands = {
        "inspect": _cmd_inspect,
        "validate": _cmd_validate,
        "probe": _cmd_probe,
        "pull": _cmd_pull,
    }
    try:
        return commands[args.command](args)
    except (ImportError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
