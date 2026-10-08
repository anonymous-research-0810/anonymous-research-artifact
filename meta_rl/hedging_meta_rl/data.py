"""Data for the paper option-hedging pipeline."""

from __future__ import annotations
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
import pandas as pd
from option_dataset import OptionDataset, get_option_dataset
from market_segmentation import segment_option_dataset
from market_simulation import (
    load_segmented_simulated_datasets,
    load_simulation_result,
    resolve_simulation_result_directory,
)


def _get_episode_id_sha256(dataset: OptionDataset) -> str:
    """Return episode id sha256."""
    encoded = "\n".join(dataset.get_episode_ids()).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _get_real_dataset_from_record(
    record: Mapping[str, Any], *, data_root: str | Path | None
) -> OptionDataset:
    """Return real dataset from record."""
    values = dict(record)
    date_period = [values.pop("date_start"), values.pop("date_end")]
    label = values.pop("label")
    moneyness_range = [values.pop("moneyness_lower"), values.pop("moneyness_upper")]
    if data_root is not None:
        values["data_root"] = str(data_root)
    return get_option_dataset(
        date_period=date_period, label=label, moneyness_range=moneyness_range, **values
    )


def load_real_dataset_from_simulation_result(
    simulation_result: str | Path | None = None,
    *,
    simulation_result_root: str | Path = "simulation/sim_results",
    data_root: str | Path | None = None,
) -> OptionDataset:
    """Load real dataset from simulation result."""
    directory = resolve_simulation_result_directory(
        simulation_result, result_root=simulation_result_root
    )
    result = load_simulation_result(directory, result_root=simulation_result_root)
    dataset = _get_real_dataset_from_record(result["dataset"]["config"], data_root=data_root)
    if _get_episode_id_sha256(dataset) != str(result["dataset"]["episode_ids_sha256"]):
        raise ValueError("Rebuilt training episode identities differ from the Stage 2 result")
    return dataset


def _subset_dataset(
    dataset: OptionDataset, episode_ids: Sequence[str], *, report: Mapping[str, Any]
) -> OptionDataset:
    """Subset dataset."""
    ids = [str(value) for value in episode_ids]
    if len(ids) != len(set(ids)):
        raise ValueError("episode_ids must be unique")
    source_ids = set(dataset.get_episode_ids())
    missing = set(ids) - source_ids
    if missing:
        raise KeyError(f"The subset contains unknown episode_id: {sorted(missing)[:5]}")
    manifest_indexed = dataset.episode_manifest.set_index("episode_id", drop=False)
    if ids:
        manifest = manifest_indexed.loc[ids].reset_index(drop=True).copy(deep=True)
        steps = pd.concat(
            [dataset.get_episode(episode_id) for episode_id in ids], ignore_index=True
        )
    else:
        manifest = dataset.episode_manifest.iloc[0:0].copy(deep=True)
        steps = dataset.episode_steps.iloc[0:0].copy(deep=True)
    audit = dataset.candidate_selection_audit.iloc[0:0].copy(deep=True)
    return OptionDataset(
        label=dataset.label,
        config=dataset.config,
        episode_steps=steps,
        episode_manifest=manifest,
        candidate_selection_audit=audit,
        dataset_build_report=dict(report),
    )


@dataclass(frozen=True)
class MetaTaskDataset:
    """One regime with real and simulated episodes sharing the same hedging interface."""

    segment_id: int
    date_start: pd.Timestamp
    date_end: pd.Timestamp
    real_dataset: OptionDataset
    simulated_dataset: OptionDataset
    num_excluded_real_episode: int = 0
    num_excluded_simulated_episode: int = 0

    def __post_init__(self) -> None:
        """Normalize fields and validate individual and cross-field constraints."""
        if isinstance(self.segment_id, bool) or int(self.segment_id) <= 0:
            raise ValueError("segment_id must be a strictly positive integer")
        start = pd.Timestamp(self.date_start).normalize().tz_localize(None)
        end = pd.Timestamp(self.date_end).normalize().tz_localize(None)
        if start > end:
            raise ValueError("A meta task must start on or before its end date")
        if self.real_dataset.label != "train" or self.simulated_dataset.label != "train":
            raise ValueError("Build meta tasks from the training split only")
        object.__setattr__(self, "segment_id", int(self.segment_id))
        object.__setattr__(self, "date_start", start)
        object.__setattr__(self, "date_end", end)

    @property
    def num_episode(self) -> int:
        """Return the number of episode."""
        return self.real_dataset.num_episode + self.simulated_dataset.num_episode

    def get_episode_metadata(self, episode_id: str, *, is_simulated: bool) -> dict[str, Any]:
        """Return episode metadata."""
        dataset = self.simulated_dataset if is_simulated else self.real_dataset
        row = dataset.get_manifest_row(episode_id)
        anchor = row.get("anchor_cohort_id") if is_simulated else None
        if is_simulated and pd.isna(anchor):
            raise ValueError(f"Simulated episode {episode_id} is missing anchor_cohort_id")
        return {
            "episode_id": str(row["episode_id"]),
            "cohort_id": str(row["cohort_id"]),
            "is_simulated": bool(is_simulated),
            "anchor_cohort_id": None if not is_simulated else str(anchor),
        }

    def get_combined_dataset(self, schedule: Sequence[tuple[bool, str]]) -> OptionDataset:
        """Return combined dataset."""
        pairs = [(bool(source), str(episode_id)) for source, episode_id in schedule]
        if len(pairs) != self.num_episode:
            raise ValueError("Schedule length must equal the task episode count")
        expected = {
            *((False, value) for value in self.real_dataset.get_episode_ids()),
            *((True, value) for value in self.simulated_dataset.get_episode_ids()),
        }
        if len(set(pairs)) != len(pairs) or set(pairs) != expected:
            raise ValueError("The schedule must cover every task episode exactly once")
        frames = []
        manifests = []
        for is_simulated, episode_id in pairs:
            dataset = self.simulated_dataset if is_simulated else self.real_dataset
            frames.append(dataset.get_episode(episode_id))
            manifests.append(dataset.get_manifest_row(episode_id).to_frame().T)
        steps = pd.concat(frames, ignore_index=True)
        manifest = pd.concat(manifests, ignore_index=True)
        audit = self.real_dataset.candidate_selection_audit.iloc[0:0].copy(deep=True)
        return OptionDataset(
            label="train",
            config=self.real_dataset.config,
            episode_steps=steps,
            episode_manifest=manifest,
            candidate_selection_audit=audit,
            dataset_build_report={
                "source": "MetaTaskDataset.get_combined_dataset",
                "segment_id": self.segment_id,
                "num_real_episode": self.real_dataset.num_episode,
                "num_simulated_episode": self.simulated_dataset.num_episode,
            },
        )

    def get_record(self) -> dict[str, Any]:
        """Return a JSON-serializable record of the result and its metadata."""
        return {
            "segment_id": self.segment_id,
            "date_start": self.date_start.strftime("%Y-%m-%d"),
            "date_end": self.date_end.strftime("%Y-%m-%d"),
            "num_real_episode": self.real_dataset.num_episode,
            "num_simulated_episode": self.simulated_dataset.num_episode,
            "num_total_episode": self.num_episode,
            "num_real_cohort": int(self.real_dataset.episode_manifest["cohort_id"].nunique()),
            "num_simulated_cohort": int(
                self.simulated_dataset.episode_manifest["cohort_id"].nunique()
            ),
            "num_excluded_real_episode": self.num_excluded_real_episode,
            "num_excluded_simulated_episode": self.num_excluded_simulated_episode,
        }


def _exclude_cross_boundary(
    dataset: OptionDataset, *, segment_id: int, date_end: pd.Timestamp, source_name: str
) -> tuple[OptionDataset, int]:
    """Exclude cross boundary."""
    exdates = pd.to_datetime(dataset.episode_manifest["exdate"], errors="raise")
    keep = exdates.dt.normalize().le(date_end)
    kept_ids = dataset.episode_manifest.loc[keep, "episode_id"].astype(str).tolist()
    num_excluded = int((~keep).sum())
    return (
        _subset_dataset(
            dataset,
            kept_ids,
            report={
                "source": "load_meta_task_datasets",
                "segment_id": segment_id,
                "dataset_source": source_name,
                "cross_boundary_rule": "episode_exdate_lte_segment_end",
                "num_excluded_episode": num_excluded,
            },
        ),
        num_excluded,
    )


def load_meta_task_datasets(
    simulation_result: str | Path | None = None,
    *,
    simulation_result_root: str | Path = "simulation/sim_results",
    data_root: str | Path | None = None,
    real_dataset: OptionDataset | None = None,
    is_exclude_cross_boundary: bool = True,
) -> list[MetaTaskDataset]:
    """Load meta task datasets."""
    if not isinstance(is_exclude_cross_boundary, bool):
        raise ValueError("is_exclude_cross_boundary must be a boolean")
    result_directory = resolve_simulation_result_directory(
        simulation_result, result_root=simulation_result_root
    )
    result = load_simulation_result(result_directory, result_root=simulation_result_root)
    if real_dataset is None:
        real_dataset = _get_real_dataset_from_record(
            result["dataset"]["config"], data_root=data_root
        )
    if real_dataset.config.label != "train":
        raise ValueError("The real data underlying Stage 2 must use the training split")
    expected_hash = str(result["dataset"]["episode_ids_sha256"])
    if _get_episode_id_sha256(real_dataset) != expected_hash:
        raise ValueError("Rebuilt training episode identities differ from the Stage 2 result")
    periods = result["segmentation_source"]["seg_periods"]
    real_segments = segment_option_dataset(real_dataset, periods)
    simulated_segments = load_segmented_simulated_datasets(result_directory)
    if len(real_segments) != len(simulated_segments):
        raise ValueError("Real and simulated regime counts differ")
    tasks: list[MetaTaskDataset] = []
    for segment_id, (period, real_part, simulated_part) in enumerate(
        zip(periods, real_segments, simulated_segments), start=1
    ):
        start = pd.Timestamp(period[0]).normalize()
        end = pd.Timestamp(period[1]).normalize()
        if is_exclude_cross_boundary:
            real_part, num_excluded_real = _exclude_cross_boundary(
                real_part, segment_id=segment_id, date_end=end, source_name="real"
            )
            simulated_part, num_excluded_simulated = _exclude_cross_boundary(
                simulated_part, segment_id=segment_id, date_end=end, source_name="simulated"
            )
        else:
            num_excluded_real = 0
            num_excluded_simulated = 0
        if real_part.num_episode == 0 or simulated_part.num_episode == 0:
            raise ValueError(
                f"segment_id={segment_id} has no real or simulated episodes after boundary filtering"
            )
        tasks.append(
            MetaTaskDataset(
                segment_id=segment_id,
                date_start=start,
                date_end=end,
                real_dataset=real_part,
                simulated_dataset=simulated_part,
                num_excluded_real_episode=num_excluded_real,
                num_excluded_simulated_episode=num_excluded_simulated,
            )
        )
    if len(tasks) < 3:
        raise ValueError("The main meta-RL experiment requires at least three valid tasks")
    return tasks
