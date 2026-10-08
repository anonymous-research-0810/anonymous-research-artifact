"""Data for the paper option-hedging pipeline."""

from __future__ import annotations
import hashlib
from pathlib import Path
from typing import Any, Mapping
import pandas as pd
from data_source import resolve_data_root, validate_data_source
from option_dataset import OptionDataset
from hedging_meta_rl import load_meta_task_datasets
from market_simulation import load_simulation_result, resolve_simulation_result_directory
from train_meta import _get_dataset_from_source_config


def _get_episode_id_sha256(dataset: OptionDataset) -> str:
    """Return episode id sha256."""
    encoded = "\n".join(dataset.get_episode_ids()).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def resolve_no_meta_simulation_result(
    result: str | Path | None, *, result_root: str | Path
) -> Path:
    """Resolve no meta simulation result."""
    return resolve_simulation_result_directory(result, result_root=result_root)


def _infer_result_data_source(result: Mapping[str, Any], *, requested_source: str) -> str:
    """Infer result data source."""
    if result.get("data_source") is not None:
        return validate_data_source(str(result["data_source"]))
    recorded_root = str(result["dataset"]["config"].get("data_root", ""))
    normalized = recorded_root.replace("\\", "/").lower()
    if "data_sx5e" in normalized or "processed_options" in normalized:
        return "sx5e"
    if normalized == "data" or normalized.endswith("/data"):
        return "spx"
    return validate_data_source(requested_source)


def _get_combined_train_dataset(tasks: list[Any], *, simulation_result: Path) -> OptionDataset:
    """Return combined train dataset."""
    step_frames: list[pd.DataFrame] = []
    manifest_frames: list[pd.DataFrame] = []
    for task in tasks:
        for is_simulated, dataset in ((False, task.real_dataset), (True, task.simulated_dataset)):
            if dataset.num_episode == 0:
                continue
            steps = dataset.episode_steps.copy(deep=True)
            manifest = dataset.episode_manifest.copy(deep=True)
            steps["is_simulated"] = bool(is_simulated)
            manifest["is_simulated"] = bool(is_simulated)
            manifest["segment_id"] = int(task.segment_id)
            step_frames.append(steps)
            manifest_frames.append(manifest)
    if not step_frames or not manifest_frames:
        raise ValueError(
            "The simulation result provides no usable real or simulated training episodes"
        )
    manifest = pd.concat(manifest_frames, ignore_index=True, sort=False)
    if manifest["episode_id"].duplicated().any():
        raise ValueError(
            "Real and simulated episode_id values overlap; the pooled RL dataset cannot be constructed"
        )
    steps = pd.concat(step_frames, ignore_index=True, sort=False)
    if set(steps["episode_id"].astype(str)) != set(manifest["episode_id"].astype(str)):
        raise ValueError("Merged episode_steps and episode_manifest identities differ")
    real_dataset = tasks[0].real_dataset
    return OptionDataset(
        label="train",
        config=real_dataset.config,
        episode_steps=steps,
        episode_manifest=manifest,
        candidate_selection_audit=real_dataset.candidate_selection_audit.iloc[0:0].copy(deep=True),
        dataset_build_report={
            "source": "no_meta_variant",
            "simulation_result": str(simulation_result),
            "num_real_episode": int(sum((task.real_dataset.num_episode for task in tasks))),
            "num_simulated_episode": int(
                sum((task.simulated_dataset.num_episode for task in tasks))
            ),
        },
    )


def load_no_meta_data(
    simulation_result: str | Path | None,
    *,
    simulation_result_root: str | Path,
    data_source: str,
    data_root: str | Path | None,
    valid_date_period: list[str] | tuple[str, str],
    is_exclude_cross_boundary: bool,
) -> tuple[OptionDataset, OptionDataset, list[Any], dict[str, Any], Path]:
    """Load no meta data."""
    result_path = resolve_no_meta_simulation_result(
        simulation_result, result_root=simulation_result_root
    )
    result = load_simulation_result(result_path, result_root=simulation_result_root)
    requested_source = validate_data_source(data_source)
    recorded_source = _infer_result_data_source(result, requested_source=requested_source)
    if recorded_source != requested_source:
        raise ValueError(
            f"The simulation data source differs from the no_meta configuration: {recorded_source} != {requested_source}; select the matching --data-source explicitly"
        )
    resolved_root = resolve_data_root(requested_source, data_root)
    train = _get_dataset_from_source_config(
        result["dataset"]["config"],
        label="train",
        date_period=[
            result["dataset"]["config"]["date_start"],
            result["dataset"]["config"]["date_end"],
        ],
        data_root=resolved_root,
    )
    if _get_episode_id_sha256(train) != str(result["dataset"]["episode_ids_sha256"]):
        raise ValueError("Rebuilt training episode identities differ from the simulation result")
    tasks = load_meta_task_datasets(
        result_path,
        simulation_result_root=simulation_result_root,
        data_root=resolved_root,
        real_dataset=train,
        is_exclude_cross_boundary=is_exclude_cross_boundary,
    )
    combined_train = _get_combined_train_dataset(tasks, simulation_result=result_path)
    valid = _get_dataset_from_source_config(
        result["dataset"]["config"],
        label="valid",
        date_period=list(valid_date_period),
        data_root=resolved_root,
    )
    return (combined_train, valid, tasks, result, result_path)


def load_no_meta_test_dataset(
    result: Mapping[str, Any],
    *,
    data_source: str,
    data_root: str | Path | None,
    test_date_period: list[str] | tuple[str, str],
) -> OptionDataset:
    """Load no meta test dataset."""
    resolved_root = resolve_data_root(data_source, data_root)
    return _get_dataset_from_source_config(
        result["dataset"]["config"],
        label="test",
        date_period=list(test_date_period),
        data_root=resolved_root,
    )
