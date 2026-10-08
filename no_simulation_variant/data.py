"""Data for the paper option-hedging pipeline."""

from __future__ import annotations
import hashlib
from pathlib import Path
from typing import Any, Mapping
import pandas as pd
from data_source import resolve_data_root, validate_data_source
from option_dataset import OptionDataset, get_option_dataset
from market_segmentation import (
    get_seg_periods_from_result,
    load_seg_result,
    resolve_seg_result_path,
    segment_option_dataset,
)
from hedging_meta_rl import MetaTaskDataset


def _get_episode_id_sha256(dataset: OptionDataset) -> str:
    """Return episode id sha256."""
    encoded = "\n".join(dataset.get_episode_ids()).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def resolve_no_simulation_segmentation_result(
    result: str | Path | None, *, result_root: str | Path
) -> Path:
    """Resolve no simulation segmentation result."""
    return resolve_seg_result_path(result, result_root=result_root)


def _infer_result_data_source(result: Mapping[str, Any], *, requested_source: str) -> str:
    """Infer result data source."""
    if result.get("data_source") is not None:
        return validate_data_source(str(result["data_source"]))
    config = result["dataset"]["config"]
    recorded_root = str(config.get("data_root", "")).replace("\\", "/").lower()
    if "data_sx5e" in recorded_root or "processed_options" in recorded_root:
        return "sx5e"
    if requested_source == "sx5e" and (recorded_root == "data" or recorded_root.endswith("/data")):
        return "spx"
    if recorded_root not in {"", "."}:
        return validate_data_source(requested_source)
    raise ValueError(
        "The segmentation record has no data_source and an unrecognized data_root; rerun Stage 1 to record the source"
    )


def _get_dataset_from_record(
    record: Mapping[str, Any],
    *,
    label: str,
    date_period: list[str] | tuple[str, str],
    data_source: str,
    data_root: str | Path | None,
) -> OptionDataset:
    """Return dataset from record."""
    values = dict(record)
    values.pop("date_start")
    values.pop("date_end")
    values.pop("label")
    moneyness_range = [values.pop("moneyness_lower"), values.pop("moneyness_upper")]
    values["data_root"] = resolve_data_root(
        data_source, data_root if data_root is not None else values.get("data_root")
    )
    return get_option_dataset(date_period, label=label, moneyness_range=moneyness_range, **values)


def _empty_dataset_like(dataset: OptionDataset, *, segment_id: int) -> OptionDataset:
    """Empty dataset like."""
    return OptionDataset(
        label=dataset.label,
        config=dataset.config,
        episode_steps=dataset.episode_steps.iloc[0:0].copy(deep=True),
        episode_manifest=dataset.episode_manifest.iloc[0:0].copy(deep=True),
        candidate_selection_audit=dataset.candidate_selection_audit.iloc[0:0].copy(deep=True),
        dataset_build_report={
            "source": "no_simulation_variant",
            "segment_id": int(segment_id),
            "num_simulated_episode": 0,
        },
    )


def _exclude_cross_boundary(
    dataset: OptionDataset, *, date_end: pd.Timestamp, segment_id: int
) -> tuple[OptionDataset, int]:
    """Exclude cross boundary."""
    exdates = pd.to_datetime(dataset.episode_manifest["exdate"], errors="raise")
    keep = exdates.dt.normalize().le(date_end)
    ids = dataset.episode_manifest.loc[keep, "episode_id"].astype(str).tolist()
    manifest_indexed = dataset.episode_manifest.set_index("episode_id", drop=False)
    if ids:
        manifest = manifest_indexed.loc[ids].reset_index(drop=True).copy(deep=True)
        steps = pd.concat(
            [dataset.get_episode(episode_id) for episode_id in ids], ignore_index=True
        )
    else:
        manifest = dataset.episode_manifest.iloc[0:0].copy(deep=True)
        steps = dataset.episode_steps.iloc[0:0].copy(deep=True)
    subset = OptionDataset(
        label=dataset.label,
        config=dataset.config,
        episode_steps=steps,
        episode_manifest=manifest,
        candidate_selection_audit=dataset.candidate_selection_audit.iloc[0:0].copy(deep=True),
        dataset_build_report={
            "source": "no_simulation_variant",
            "segment_id": int(segment_id),
            "cross_boundary_rule": "episode_exdate_lte_segment_end",
            "num_excluded_episode": int((~keep).sum()),
        },
    )
    return (subset, int((~keep).sum()))


def load_no_simulation_tasks(
    segmentation_result: str | Path | None,
    *,
    segmentation_result_root: str | Path,
    data_source: str,
    data_root: str | Path | None,
    is_exclude_cross_boundary: bool,
) -> tuple[list[MetaTaskDataset], OptionDataset, dict[str, Any], Path]:
    """Load no simulation tasks."""
    result_path = resolve_no_simulation_segmentation_result(
        segmentation_result, result_root=segmentation_result_root
    )
    result = load_seg_result(result_path, result_root=segmentation_result_root)
    requested_source = validate_data_source(data_source)
    recorded_source = _infer_result_data_source(result, requested_source=requested_source)
    if recorded_source != requested_source:
        raise ValueError(
            f"The segmentation data source differs from the ablation configuration: {recorded_source} != {requested_source}; select the matching --data-source explicitly"
        )
    source_config = result["dataset"]["config"]
    train = _get_dataset_from_record(
        source_config,
        label="train",
        date_period=[source_config["date_start"], source_config["date_end"]],
        data_source=requested_source,
        data_root=data_root,
    )
    if _get_episode_id_sha256(train) != str(result["dataset"]["episode_ids_sha256"]):
        raise ValueError("Rebuilt training episode identities differ from the segmentation result")
    periods = get_seg_periods_from_result(result)
    real_parts = segment_option_dataset(train, periods)
    tasks: list[MetaTaskDataset] = []
    for segment_id, (period, real_part) in enumerate(zip(periods, real_parts), start=1):
        start = pd.Timestamp(period[0]).normalize()
        end = pd.Timestamp(period[1]).normalize()
        excluded = 0
        if is_exclude_cross_boundary:
            real_part, excluded = _exclude_cross_boundary(
                real_part, date_end=end, segment_id=segment_id
            )
        if real_part.num_episode == 0:
            raise ValueError(f"segment_id={segment_id} has no usable real episodes")
        if real_part.episode_manifest["cohort_id"].nunique() < 2:
            raise ValueError(
                f"segment_id={segment_id} has fewer than two real cohorts; context/query source separation is unavailable"
            )
        tasks.append(
            MetaTaskDataset(
                segment_id=segment_id,
                date_start=start,
                date_end=end,
                real_dataset=real_part,
                simulated_dataset=_empty_dataset_like(real_part, segment_id=segment_id),
                num_excluded_real_episode=excluded,
                num_excluded_simulated_episode=0,
            )
        )
    if len(tasks) < 3:
        raise ValueError("The no-simulation ablation requires at least three valid regime tasks")
    return (tasks, train, result, result_path)


def load_no_simulation_data(
    segmentation_result: str | Path | None,
    *,
    segmentation_result_root: str | Path,
    data_source: str,
    data_root: str | Path | None,
    train_date_period: list[str] | tuple[str, str],
    valid_date_period: list[str] | tuple[str, str],
    is_exclude_cross_boundary: bool,
) -> tuple[list[MetaTaskDataset], OptionDataset, OptionDataset, dict[str, Any], Path]:
    """Load no simulation data."""
    tasks, train, result, result_path = load_no_simulation_tasks(
        segmentation_result,
        segmentation_result_root=segmentation_result_root,
        data_source=data_source,
        data_root=data_root,
        is_exclude_cross_boundary=is_exclude_cross_boundary,
    )
    valid = _get_dataset_from_record(
        result["dataset"]["config"],
        label="valid",
        date_period=list(valid_date_period),
        data_source=data_source,
        data_root=data_root,
    )
    return (tasks, train, valid, result, result_path)
