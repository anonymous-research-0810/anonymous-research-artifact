"""Adapter for the paper option-hedging pipeline."""

from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
import numpy as np
import pandas as pd

SX5E_CANONICAL_SECID = 108105
SX5E_SYMBOL_ROOT = "SPXW"
SX5E_CONTRACT_MULTIPLIER = 10.0
MANIFEST_FILE = "sx5e_adapter_manifest.json"
_OPTION_COLUMNS = (
    "secid",
    "date",
    "exdate",
    "cp_flag",
    "strike_price",
    "best_bid",
    "best_offer",
    "volume",
    "open_interest",
    "impl_volatility",
    "delta",
    "gamma",
    "vega",
    "theta",
    "optionid",
    "symbol",
)
_MARKET_FILES = ("spx_spot.parquet", "zero_curve.parquet", "spx_div_yield.parquet")


def _get_repository_root() -> Path:
    """Return repository root."""
    return Path(__file__).resolve().parents[2]


def _get_raw_root(raw_root: str | Path) -> Path:
    """Return raw root."""
    path = Path(raw_root)
    if not path.is_absolute():
        path = _get_repository_root() / path
    path = path.resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Raw SX5E data directory does not exist: {path}")
    return path


def _get_output_root(raw_root: Path, output_root: str | Path | None) -> Path:
    """Return output root."""
    if output_root is None:
        return raw_root / "processed_options"
    path = Path(output_root)
    if not path.is_absolute():
        path = _get_repository_root() / path
    return path.resolve()


def _json_safe(value: Any) -> Any:
    """Json safe."""
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, Path):
        return str(value)
    return value


def _get_source_fingerprint(raw_root: Path) -> dict[str, dict[str, int]]:
    """Return source fingerprint."""
    paths = sorted(raw_root.glob("sx5e_options_*.parquet"))
    paths += [
        raw_root / name
        for name in ("sx5e_spot.parquet", "sx5e_div_yield.parquet", "zero_curve.parquet")
    ]
    result: dict[str, dict[str, int]] = {}
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"The raw SX5E data file does not exist: {path}")
        stat = path.stat()
        result[path.name] = {"size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}
    if not any((name.startswith("sx5e_options_") for name in result)):
        raise FileNotFoundError(f"No sx5e_options_YYYY.parquet files were found: {raw_root}")
    return result


def _write_option_year(raw_path: Path, output_path: Path) -> dict[str, int]:
    """Write option year."""
    required = set(_OPTION_COLUMNS) | {"quote_valid"}
    frame = pd.read_parquet(raw_path)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{raw_path} is missing required SX5E fields: {missing}")
    quote_valid = frame["quote_valid"].fillna(False).astype(bool)
    bid = pd.to_numeric(frame["best_bid"], errors="coerce")
    offer = pd.to_numeric(frame["best_offer"], errors="coerce")
    valid_quote = (
        quote_valid
        & np.isfinite(bid.to_numpy(dtype=np.float64))
        & np.isfinite(offer.to_numpy(dtype=np.float64))
        & bid.ge(0.0)
        & offer.ge(0.0)
        & offer.ge(bid)
    )
    selected = frame.loc[valid_quote, list(_OPTION_COLUMNS)].copy()
    if selected.empty:
        raise ValueError(f"{raw_path} has no usable two-sided quotes")
    selected["secid"] = np.float64(SX5E_CANONICAL_SECID)
    selected["cp_flag"] = selected["cp_flag"].astype("string").str.upper()
    selected["symbol"] = (
        SX5E_SYMBOL_ROOT
        + " "
        + selected["symbol"].astype("string").str.replace("^SX5E\\s*", "", regex=True)
    )
    selected["optionid"] = pd.to_numeric(selected["optionid"], errors="raise").astype("int64")
    selected["date"] = pd.to_datetime(selected["date"], errors="raise").dt.normalize()
    selected["exdate"] = pd.to_datetime(selected["exdate"], errors="raise").dt.normalize()
    selected = selected.sort_values(["date", "exdate", "cp_flag", "strike_price", "optionid"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    selected.to_parquet(output_path, index=False, engine="pyarrow")
    return {
        "num_input_row": int(len(frame)),
        "num_quote_valid_row": int(valid_quote.sum()),
        "num_output_row": int(len(selected)),
    }


def _write_market_files(raw_root: Path, output_root: Path) -> dict[str, Any]:
    """Write market files."""
    spot_path = raw_root / "sx5e_spot.parquet"
    dividend_path = raw_root / "sx5e_div_yield.parquet"
    zero_path = raw_root / "zero_curve.parquet"
    spot = pd.read_parquet(spot_path)
    dividend = pd.read_parquet(dividend_path)
    zero = pd.read_parquet(zero_path)
    for frame, path, required in (
        (spot, spot_path, {"secid", "date", "open", "high", "low", "close", "volume", "return"}),
        (dividend, dividend_path, {"secid", "date", "rate"}),
        (zero, zero_path, {"date", "days", "rate"}),
    ):
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{path} is missing market fields: {missing}")
    spot = spot.loc[spot["secid"].eq(-1001)].copy()
    dividend = dividend.loc[dividend["secid"].eq(-1001)].copy()
    if spot.empty or dividend.empty:
        raise ValueError("No secid=-1001 rows were found in SX5E spot/dividend files")
    spot["date"] = pd.to_datetime(spot["date"], errors="raise").dt.normalize()
    dividend["date"] = pd.to_datetime(dividend["date"], errors="raise").dt.normalize()
    zero["date"] = pd.to_datetime(zero["date"], errors="raise").dt.normalize()
    if spot["date"].duplicated().any() or dividend["date"].duplicated().any():
        raise ValueError("SX5E spot or dividend data contain duplicate trading dates")
    if zero.duplicated(["date", "days"]).any():
        raise ValueError("SX5E zero curves contain duplicate date/days pairs")
    spot_dates = set(spot["date"])
    dividend_dates = set(dividend["date"])
    zero_dates = set(zero["date"])
    common_dates = sorted(spot_dates & dividend_dates & zero_dates)
    if not common_dates:
        raise ValueError("The three SX5E market tables have no common dates")
    spot = spot.loc[spot["date"].isin(common_dates)].copy()
    dividend = dividend.loc[dividend["date"].isin(common_dates)].copy()
    zero = zero.loc[zero["date"].isin(common_dates)].copy()
    spot["secid"] = np.int64(SX5E_CANONICAL_SECID)
    dividend["secid"] = np.int64(SX5E_CANONICAL_SECID)
    spot.loc[:, ["secid", "date", "open", "high", "low", "close", "volume", "return"]].to_parquet(
        output_root / "spx_spot.parquet", index=False, engine="pyarrow"
    )
    dividend.loc[:, ["secid", "date", "rate"]].to_parquet(
        output_root / "spx_div_yield.parquet", index=False, engine="pyarrow"
    )
    zero.loc[:, ["date", "days", "rate"]].sort_values(["date", "days"]).to_parquet(
        output_root / "zero_curve.parquet", index=False, engine="pyarrow"
    )
    return {
        "num_spot_date": int(len(spot)),
        "num_dividend_date": int(len(dividend)),
        "num_zero_curve_date": int(zero["date"].nunique()),
        "num_dropped_market_date": int(
            len(spot_dates | dividend_dates | zero_dates) - len(common_dates)
        ),
    }


def _is_complete_output(output_root: Path) -> bool:
    """Return whether complete output."""
    return (
        output_root.is_dir()
        and all(((output_root / name).is_file() for name in _MARKET_FILES))
        and bool(list(output_root.glob("spx_options_*.parquet")))
    )


def get_sx5e_manifest(data_root: str | Path) -> dict[str, Any]:
    """Return sx5e manifest."""
    path = Path(data_root) / MANIFEST_FILE
    if not path.is_file():
        raise FileNotFoundError(f"The SX5E adapter manifest does not exist: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_sx5e_data_root(
    raw_root: str | Path = "data_sx5e",
    *,
    output_root: str | Path | None = None,
    is_force: bool = False,
) -> Path:
    """Convert SX5E quotes to the shared schema. Keep valid bid/ask quotes, use the standard European-call payoff, and preserve index-point prices with multiplier 10."""
    raw = _get_raw_root(raw_root)
    output = _get_output_root(raw, output_root)
    fingerprint = _get_source_fingerprint(raw)
    manifest_path = output / MANIFEST_FILE
    if not is_force and _is_complete_output(output) and manifest_path.is_file():
        try:
            manifest = get_sx5e_manifest(output)
        except (OSError, ValueError, json.JSONDecodeError):
            manifest = None
        if isinstance(manifest, Mapping) and manifest.get("source_fingerprint") == fingerprint:
            return output
    output.mkdir(parents=True, exist_ok=True)
    for path in output.glob("spx_options_*.parquet"):
        path.unlink()
    for name in _MARKET_FILES:
        path = output / name
        if path.exists():
            path.unlink()
    option_statistics: dict[str, dict[str, int]] = {}
    for raw_path in sorted(raw.glob("sx5e_options_*.parquet")):
        year = raw_path.stem.rsplit("_", 1)[-1]
        option_statistics[year] = _write_option_year(
            raw_path, output / f"spx_options_{year}.parquet"
        )
    market_statistics = _write_market_files(raw, output)
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": "sx5e",
        "raw_root": str(raw),
        "output_root": str(output),
        "source_fingerprint": fingerprint,
        "canonical_secid": SX5E_CANONICAL_SECID,
        "canonical_symbol_root": SX5E_SYMBOL_ROOT,
        "contract_multiplier": SX5E_CONTRACT_MULTIPLIER,
        "currency": "EUR",
        "terminal_payoff_assumption": "C_T=max(S_T-K,0)",
        "quote_policy": "quote_valid_only; finite nonnegative bid/ask; offer>=bid",
        "invalid_quote_rows_are_dropped": True,
        "option_statistics": option_statistics,
        "market_statistics": market_statistics,
    }
    manifest_path.write_text(
        json.dumps(_json_safe(manifest), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return output
