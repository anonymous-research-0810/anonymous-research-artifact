"""Dataset for the paper option-hedging pipeline."""

from __future__ import annotations
from collections.abc import Sequence
from copy import deepcopy
from typing import Any
import numpy as np
import pandas as pd
from option_dataset import OptionDataset

SegPeriod = tuple[pd.Timestamp, pd.Timestamp]


def _get_normalized_date(value: Any, name: str) -> pd.Timestamp:
    """Return normalized date."""
    try:
        date = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} contains unparseable dates") from exc
    if pd.isna(date):
        raise ValueError(f"{name} must not contain missing dates")
    if date.tzinfo is not None:
        date = date.tz_localize(None)
    return date.normalize()


def normalize_seg_periods(seg_periods: Sequence[Sequence[Any]]) -> list[SegPeriod]:
    """Normalize seg periods."""
    if isinstance(seg_periods, (str, bytes)):
        raise ValueError("seg_periods must be a sequence of date pairs")
    try:
        raw_periods = list(seg_periods)
    except TypeError as exc:
        raise ValueError("seg_periods must be a sequence of date pairs") from exc
    if not raw_periods:
        raise ValueError("seg_periods must not be empty")
    normalized: list[SegPeriod] = []
    for index, period in enumerate(raw_periods):
        if isinstance(period, (str, bytes)):
            raise ValueError(f"seg_periods[{index}] must contain a start and end date")
        try:
            values = list(period)
        except TypeError as exc:
            raise ValueError(f"seg_periods[{index}] must contain a start and end date") from exc
        if len(values) != 2:
            raise ValueError(f"seg_periods[{index}] must contain a start and end date")
        start = _get_normalized_date(values[0], f"seg_periods[{index}][0]")
        end = _get_normalized_date(values[1], f"seg_periods[{index}][1]")
        if start > end:
            raise ValueError(f"seg_periods[{index}] must start on or before its end date")
        if normalized and start <= normalized[-1][1]:
            raise ValueError("seg_periods must be strictly increasing, disjoint closed intervals")
        normalized.append((start, end))
    return normalized


def _get_period_assignments(
    dates: pd.Series, periods: Sequence[SegPeriod], *, name: str
) -> np.ndarray:
    """Return period assignments."""
    parsed = pd.to_datetime(dates, errors="coerce").dt.normalize()
    if parsed.isna().any():
        raise ValueError(f"{name} contains missing or unparseable dates")
    assignments = np.full(len(parsed), -1, dtype=np.int64)
    values = parsed.to_numpy(dtype="datetime64[ns]")
    for index, (start, end) in enumerate(periods):
        mask = (values >= start.to_datetime64()) & (values <= end.to_datetime64())
        if np.any(assignments[mask] >= 0):
            raise RuntimeError("Internal error: a date belongs to multiple seg_periods")
        assignments[mask] = index
    if np.any(assignments < 0):
        missing = (
            parsed.iloc[np.flatnonzero(assignments < 0)]
            .drop_duplicates()
            .sort_values()
            .dt.strftime("%Y-%m-%d")
            .tolist()
        )
        preview = missing[:10]
        raise ValueError(
            f"{name} contains dates not covered by seg_periods: {preview}"
            + (" ..." if len(missing) > len(preview) else "")
        )
    return assignments


def _validate_exact_partition(source: OptionDataset, parts: Sequence[OptionDataset]) -> None:
    """Validate exact partition."""
    combined_episode_ids = tuple(
        (episode_id for part in parts for episode_id in part.get_episode_ids())
    )
    if combined_episode_ids != source.get_episode_ids():
        raise RuntimeError(
            "Internal error: partition episode order or identities differ from the source"
        )
    table_names = ("episode_steps", "episode_manifest", "candidate_selection_audit")
    for name in table_names:
        source_frame = getattr(source, name).reset_index(drop=True)
        frames = [getattr(part, name) for part in parts]
        combined = pd.concat(frames, ignore_index=True)
        try:
            pd.testing.assert_frame_equal(
                combined, source_frame, check_dtype=True, check_like=False, check_exact=True
            )
        except AssertionError as exc:
            raise RuntimeError(
                f"Internal error: concatenated partition {name} differs from the source dataset"
            ) from exc


def segment_option_dataset(
    dataset: OptionDataset, seg_periods: Sequence[Sequence[Any]]
) -> list[OptionDataset]:
    """Partition complete episodes by inception date without truncating lifecycles; preserve source order and table contents."""
    if not isinstance(dataset, OptionDataset):
        raise TypeError("dataset must be an OptionDataset instance")
    periods = normalize_seg_periods(seg_periods)
    if dataset.episode_manifest.empty:
        raise ValueError("Cannot partition an empty OptionDataset")
    manifest_assignments = _get_period_assignments(
        dataset.episode_manifest["t0"], periods, name="episode_manifest.t0"
    )
    if dataset.candidate_selection_audit.empty:
        audit_assignments = np.empty(0, dtype=np.int64)
    else:
        audit_assignments = _get_period_assignments(
            dataset.candidate_selection_audit["date"],
            periods,
            name="candidate_selection_audit.date",
        )
    parts: list[OptionDataset] = []
    for index, (start, end) in enumerate(periods):
        manifest_mask = manifest_assignments == index
        manifest = dataset.episode_manifest.loc[manifest_mask].copy(deep=True)
        episode_ids = set(manifest["episode_id"].astype(str))
        step_mask = dataset.episode_steps["episode_id"].astype(str).isin(episode_ids)
        steps = dataset.episode_steps.loc[step_mask].copy(deep=True)
        audit = dataset.candidate_selection_audit.loc[audit_assignments == index].copy(deep=True)
        if manifest.empty:
            num_cross_boundary = 0
        else:
            exdates = pd.to_datetime(manifest["exdate"], errors="raise").dt.normalize()
            num_cross_boundary = int(exdates.gt(end).sum())
        report = {
            "source": "segment_option_dataset",
            "source_dataset_config": dataset.config.get_dict(),
            "source_dataset_build_report": deepcopy(dataset.dataset_build_report),
            "segmentation": {
                "segment_id": index + 1,
                "date_start": start.strftime("%Y-%m-%d"),
                "date_end": end.strftime("%Y-%m-%d"),
                "assignment_rule": "episode_t0_in_closed_period",
                "num_episode": int(len(manifest)),
                "num_step": int(len(steps)),
                "num_cohort": int(manifest["cohort_id"].nunique()),
                "num_candidate_audit_row": int(len(audit)),
                "num_cross_boundary_episode": num_cross_boundary,
            },
        }
        parts.append(
            OptionDataset(
                label=dataset.label,
                config=dataset.config,
                episode_steps=steps,
                episode_manifest=manifest,
                candidate_selection_audit=audit,
                dataset_build_report=report,
            )
        )
    _validate_exact_partition(dataset, parts)
    return parts
