"""Data source for the paper option-hedging pipeline."""

from __future__ import annotations
from pathlib import Path
from sx5e import ensure_sx5e_data_root

VALID_DATA_SOURCES = ("spx", "sx5e")


def validate_data_source(data_source: str) -> str:
    """Validate data source."""
    value = str(data_source).strip().lower()
    if value not in VALID_DATA_SOURCES:
        raise ValueError(f"data_source must be {VALID_DATA_SOURCES} ; received {data_source!r}")
    return value


def resolve_data_root(
    data_source: str = "spx", data_root: str | Path | None = None, *, is_prepare: bool = True
) -> Path:
    """Resolve an external market-data directory and prepare the SX5E adapter output when needed."""
    source = validate_data_source(data_source)
    if source == "spx":
        path = Path("data" if data_root is None else data_root)
        if not path.is_absolute():
            path = Path(__file__).resolve().parent / path
        return path.resolve()
    if data_root is None or str(data_root).replace("\\", "/").strip("/") == "data":
        raw = Path("data_sx5e")
    else:
        raw = Path(data_root)
    if not raw.is_absolute():
        raw = Path(__file__).resolve().parent / raw
    raw = raw.resolve()
    if all(
        (
            (raw / name).is_file()
            for name in ("spx_spot.parquet", "zero_curve.parquet", "spx_div_yield.parquet")
        )
    ):
        return raw
    if not is_prepare:
        return (raw / "processed_options").resolve()
    return ensure_sx5e_data_root(raw)


def get_data_source_config(data_source: str, data_root: str | Path | None = None) -> dict[str, str]:
    """Return data source config."""
    source = validate_data_source(data_source)
    resolved = resolve_data_root(source, data_root)
    return {"data_source": source, "data_root": str(resolved)}
