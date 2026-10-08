"""Pipeline for the paper option-hedging pipeline."""

from __future__ import annotations
import hashlib
import json
import os
import platform
import re
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import ruptures
import scipy
from option_dataset import OptionDataset
from .config import SegmentationConfig
from .dataset import segment_option_dataset
from .features import SegmentationFeatures, get_segmentation_features
from .pelt import PeltResult, run_pelt_segmentation

SEGMENTATION_SCHEMA_VERSION = 1
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEGMENTATION_RESULT_ROOT = PACKAGE_ROOT / "seg_results"
SEGMENTATION_RESULT_NAME_PATTERN = re.compile("^seg_(?P<timestamp>\\d{8}T\\d{6}Z)\\.json$")


def _get_datetime_record() -> dict[str, str]:
    """Return the current UTC timestamp."""
    now = datetime.now(timezone.utc)
    return {"utc": now.isoformat(timespec="seconds")}


def _get_json_value(value: Any) -> Any:
    """Return JSON value."""
    if isinstance(value, Mapping):
        return {str(key): _get_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_get_json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (pd.Timestamp, datetime, np.datetime64)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, np.ndarray):
        return [_get_json_value(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _get_json_value(value.item())
    if isinstance(value, float) and (not np.isfinite(value)):
        return None
    return value


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Write JSON atomic."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(
            _get_json_value(payload), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
        ),
        encoding="utf-8",
    )
    os.replace(temporary_path, path)


def _get_file_sha256(path: Path) -> str:
    """Compute a file SHA-256 digest using bounded-memory reads."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _get_code_fingerprints() -> list[dict[str, Any]]:
    """Return code fingerprints."""
    paths = [
        PACKAGE_ROOT.parent / "run_seg.py",
        *sorted((PACKAGE_ROOT / "market_segmentation").glob("*.py")),
        PACKAGE_ROOT.parent / "datasets" / "option_dataset" / "dataset.py",
        PACKAGE_ROOT.parent / "datasets" / "option_dataset" / "config.py",
    ]
    records: list[dict[str, Any]] = []
    repository_root = PACKAGE_ROOT.parent
    for path in paths:
        if path.is_file():
            records.append(
                {
                    "path": str(path.relative_to(repository_root)),
                    "num_bytes": path.stat().st_size,
                    "sha256": _get_file_sha256(path),
                }
            )
    return records


def _get_system_info() -> dict[str, Any]:
    """Record software versions without account, hostname, or repository identity."""
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "ruptures": ruptures.__version__,
    }


def _get_dataset_record(dataset: OptionDataset) -> dict[str, Any]:
    """Return dataset record."""
    episode_ids = "\n".join(dataset.get_episode_ids()).encode("utf-8")
    return {
        "label": dataset.label,
        "config": dataset.config.get_dict(),
        "num_episode": dataset.num_episode,
        "num_step": dataset.num_step,
        "num_cohort": int(dataset.episode_manifest["cohort_id"].nunique()),
        "num_candidate_audit_row": int(len(dataset.candidate_selection_audit)),
        "is_environment_ready": dataset.is_environment_ready,
        "episode_ids_sha256": hashlib.sha256(episode_ids).hexdigest(),
        "dataset_build_report": dataset.dataset_build_report,
    }


def _get_safe_experiment_id(experiment_id: str) -> str:
    """Return safe experiment id."""
    value = str(experiment_id)
    if (
        not value
        or value in {".", ".."}
        or Path(value).name != value
        or ("/" in value)
        or ("\\" in value)
        or value.lower().endswith(".json")
    ):
        raise ValueError(
            "experiment_id must be a nonempty name without path separators or a .json suffix"
        )
    return value


def _get_seg_periods(
    dataset: OptionDataset, features: SegmentationFeatures, breakpoints: Sequence[int]
) -> tuple[list[tuple[str, str]], list[dict[str, Any]]]:
    """Return seg periods."""
    num_feature = len(features.feature_frame)
    values = [int(value) for value in breakpoints]
    if not values or values[-1] != num_feature:
        raise ValueError("breakpoints must end at the feature_frame length")
    feature_dates = pd.DatetimeIndex(features.feature_frame["date"])
    spot_dates = pd.DatetimeIndex(features.daily_spot["date"])
    internal = values[:-1]
    boundary_dates = [pd.Timestamp(feature_dates[index]) for index in internal]
    period_starts = [dataset.config.date_start, *boundary_dates]
    period_ends: list[pd.Timestamp] = []
    boundary_records: list[dict[str, Any]] = []
    for breakpoint, boundary_date in zip(internal, boundary_dates):
        previous_dates = spot_dates[spot_dates < boundary_date]
        if len(previous_dates) == 0:
            raise RuntimeError(
                "No trading day precedes the change point; a closed interval cannot be formed"
            )
        previous_date = pd.Timestamp(previous_dates[-1])
        period_ends.append(previous_date)
        boundary_records.append(
            {
                "feature_breakpoint": int(breakpoint),
                "next_segment_start": boundary_date.strftime("%Y-%m-%d"),
                "previous_segment_end": previous_date.strftime("%Y-%m-%d"),
            }
        )
    period_ends.append(dataset.config.date_end)
    periods = [
        (start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
        for start, end in zip(period_starts, period_ends)
    ]
    return (periods, boundary_records)


def _get_segment_statistics(
    dataset: OptionDataset,
    subdatasets: Sequence[OptionDataset],
    features: SegmentationFeatures,
    seg_periods: Sequence[tuple[str, str]],
    annualization_days: int,
) -> list[dict[str, Any]]:
    """Return segment statistics."""
    records: list[dict[str, Any]] = []
    daily = features.daily_spot
    feature_frame = features.feature_frame
    for index, ((start_text, end_text), subdataset) in enumerate(
        zip(seg_periods, subdatasets), start=1
    ):
        start = pd.Timestamp(start_text)
        end = pd.Timestamp(end_text)
        daily_mask = daily["date"].between(start, end, inclusive="both")
        segment_daily = daily.loc[daily_mask]
        feature_mask = feature_frame["date"].between(start, end, inclusive="both")
        segment_features = feature_frame.loc[feature_mask]
        spots = segment_daily["spot"].to_numpy(dtype=np.float64)
        returns = segment_daily["log_return"].dropna().to_numpy(dtype=np.float64)
        manifest = subdataset.episode_manifest
        if manifest.empty:
            num_cross_boundary = 0
        else:
            exdates = pd.to_datetime(manifest["exdate"]).dt.normalize()
            num_cross_boundary = int(exdates.gt(end).sum())
        if returns.size:
            mean_log_return = float(np.mean(returns))
            annualized_volatility = float(np.sqrt(annualization_days * np.mean(np.square(returns))))
        else:
            mean_log_return = None
            annualized_volatility = None
        records.append(
            {
                "segment_id": index,
                "date_start": start_text,
                "date_end": end_text,
                "num_trading_day": int(len(segment_daily)),
                "num_feature_observation": int(len(segment_features)),
                "num_episode": subdataset.num_episode,
                "num_step": subdataset.num_step,
                "num_cohort": int(manifest["cohort_id"].nunique()),
                "num_cross_boundary_episode": num_cross_boundary,
                "is_meta_task_minimum_feasible": bool(manifest["cohort_id"].nunique() >= 2),
                "spot_start": float(spots[0]) if spots.size else None,
                "spot_end": float(spots[-1]) if spots.size else None,
                "mean_daily_log_return": mean_log_return,
                "annualized_mean_log_return": (
                    annualization_days * mean_log_return if mean_log_return is not None else None
                ),
                "annualized_realized_volatility": annualized_volatility,
                "mean_rolling_realized_volatility": (
                    float(segment_features["realized_volatility"].mean())
                    if not segment_features.empty
                    else None
                ),
            }
        )
    return records


def _add_boundary_dates(
    boundary_records: list[dict[str, Any]], pelt_result: PeltResult
) -> list[dict[str, Any]]:
    """Add boundary dates."""
    bootstrap_by_breakpoint = {
        int(record["breakpoint"]): record for record in pelt_result.bootstrap["boundary_statistics"]
    }
    results = []
    for record in boundary_records:
        breakpoint = int(record["feature_breakpoint"])
        results.append({**record, "bootstrap": bootstrap_by_breakpoint.get(breakpoint)})
    return results


def run_market_segmentation(
    dataset: OptionDataset,
    config: SegmentationConfig,
    *,
    daily_spot: pd.DataFrame | None = None,
    output_root: str | Path = DEFAULT_SEGMENTATION_RESULT_ROOT,
    experiment_id: str | None = None,
) -> dict[str, Any]:
    """Run training-only regime detection and persist validated closed date intervals and input fingerprints."""
    if not isinstance(dataset, OptionDataset):
        raise TypeError("dataset must be an OptionDataset instance")
    if dataset.label != "train":
        raise ValueError("Stage 1 requires an OptionDataset with label='train'")
    if not isinstance(config, SegmentationConfig):
        raise TypeError("config must be a SegmentationConfig instance")
    started_at = _get_datetime_record()
    start_time = time.perf_counter()
    if experiment_id is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        experiment_id = f"seg_{timestamp}"
    experiment_id = _get_safe_experiment_id(experiment_id)
    output_path = Path(output_root)
    result_path = output_path / f"{experiment_id}.json"
    if result_path.exists():
        raise FileExistsError(f"The segmentation result file already exists: {result_path}")
    dataset_record = _get_dataset_record(dataset)
    running_record = {
        "schema_version": SEGMENTATION_SCHEMA_VERSION,
        "status": "running",
        "experiment_id": experiment_id,
        "started_at": started_at,
        "dataset": dataset_record,
        "segmentation_config": config.get_dict(),
        "resolved_segmentation_config": config.resolve_for_dataset(dataset.config.num_interval),
        "system": _get_system_info(),
        "code_fingerprints": _get_code_fingerprints(),
    }
    try:
        features = get_segmentation_features(dataset, config, daily_spot=daily_spot)
        pelt_result = run_pelt_segmentation(
            features.signal, config, num_interval=dataset.config.num_interval
        )
        seg_periods, boundary_records = _get_seg_periods(dataset, features, pelt_result.breakpoints)
        subdatasets = segment_option_dataset(dataset, seg_periods)
        segment_statistics = _get_segment_statistics(
            dataset, subdatasets, features, seg_periods, config.annualization_days
        )
        completed_at = _get_datetime_record()
        result = {
            **running_record,
            "status": "completed",
            "completed_at": completed_at,
            "num_elapsed_second": time.perf_counter() - start_time,
            "features": features.get_record(),
            "pelt": pelt_result.get_record(),
            "num_segment": len(seg_periods),
            "seg_periods": [list(period) for period in seg_periods],
            "boundaries": _add_boundary_dates(boundary_records, pelt_result),
            "segment_statistics": segment_statistics,
            "partition_verification": {
                "assignment_rule": "episode_t0_in_closed_period",
                "is_exact_episode_partition": True,
                "num_source_episode": dataset.num_episode,
                "num_partition_episode": int(sum((part.num_episode for part in subdatasets))),
                "episode_ids_sha256": dataset_record["episode_ids_sha256"],
            },
        }
        _write_json_atomic(result_path, result)
        return {"result_path": result_path, "result": result, "subdatasets": subdatasets}
    except Exception as exc:
        failed_record = {
            **running_record,
            "status": "failed",
            "failed_at": _get_datetime_record(),
            "num_elapsed_second": time.perf_counter() - start_time,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }
        _write_json_atomic(result_path, failed_record)
        raise


def _get_seg_periods_from_payload(result: Mapping[str, Any]) -> list[tuple[str, str]]:
    """Return seg periods from payload."""
    if not isinstance(result, Mapping):
        raise TypeError("result must be a Mapping")
    if result.get("schema_version") != SEGMENTATION_SCHEMA_VERSION:
        raise ValueError(f"segmentation schema_version must be {SEGMENTATION_SCHEMA_VERSION}")
    raw_periods = result.get("seg_periods")
    if not isinstance(raw_periods, list) or not raw_periods:
        raise ValueError("The segmentation result requires nonempty seg_periods")
    periods: list[tuple[str, str]] = []
    previous_end: pd.Timestamp | None = None
    for index, period in enumerate(raw_periods):
        if (
            not isinstance(period, list)
            or len(period) != 2
            or (not all((isinstance(value, str) for value in period)))
        ):
            raise ValueError(f"seg_periods[{index}] must contain two date strings")
        start = pd.Timestamp(period[0]).normalize()
        end = pd.Timestamp(period[1]).normalize()
        if start > end or (previous_end is not None and start <= previous_end):
            raise ValueError(
                "seg_periods must be strictly increasing, disjoint, and have valid bounds"
            )
        periods.append((start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")))
        previous_end = end
    if result.get("num_segment") != len(periods):
        raise ValueError("num_segment differs from the length of seg_periods")
    return periods


def resolve_seg_result_path(
    path: str | Path | None = None, *, result_root: str | Path = DEFAULT_SEGMENTATION_RESULT_ROOT
) -> Path:
    """Resolve seg result path."""
    root = Path(result_root)
    if path is not None:
        candidate = Path(path)
        if not candidate.is_absolute() and (not candidate.is_file()):
            candidate = root / candidate
        if not candidate.is_file():
            raise FileNotFoundError(f"The segmentation result JSON does not exist: {candidate}")
        return candidate
    candidates: list[tuple[str, Path]] = []
    if root.is_dir():
        for candidate in root.glob("seg_*.json"):
            match = SEGMENTATION_RESULT_NAME_PATTERN.fullmatch(candidate.name)
            if match is None:
                continue
            try:
                payload = json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(payload, dict) and payload.get("status") == "completed":
                candidates.append((match.group("timestamp"), candidate))
    if not candidates:
        raise FileNotFoundError(
            f"No completed segmentation result named seg_YYYYMMDDTHHMMSSZ.json was found: {root}"
        )
    return max(candidates, key=lambda item: item[0])[1]


def load_seg_result(
    path: str | Path | None = None, *, result_root: str | Path = DEFAULT_SEGMENTATION_RESULT_ROOT
) -> dict[str, Any]:
    """Load seg result."""
    result_path = resolve_seg_result_path(path, result_root=result_root)
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read the segmentation result JSON: {result_path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("The segmentation JSON root must be an object")
    if payload.get("status") != "completed":
        raise ValueError(f"The segmentation result status is not completed: {result_path}")
    _get_seg_periods_from_payload(payload)
    return payload


def get_seg_periods_from_result(
    result: Mapping[str, Any] | str | Path | None = None,
    *,
    result_root: str | Path = DEFAULT_SEGMENTATION_RESULT_ROOT,
) -> list[tuple[str, str]]:
    """Return seg periods from result."""
    if isinstance(result, Mapping):
        return _get_seg_periods_from_payload(result)
    payload = load_seg_result(result, result_root=result_root)
    return _get_seg_periods_from_payload(payload)
