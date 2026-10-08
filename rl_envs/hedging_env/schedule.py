"""Schedule for the paper option-hedging pipeline."""

from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd
from option_dataset import OptionDataset
from .exceptions import EnvironmentConfigurationError


def _get_nonnegative_int(value: Any, name: str, *, is_positive: bool) -> int:
    """Return nonnegative int."""
    if isinstance(value, (bool, np.bool_)):
        raise EnvironmentConfigurationError(f"{name} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise EnvironmentConfigurationError(f"{name} must be an integer") from exc
    if result != value or result < (1 if is_positive else 0):
        qualifier = " positive integer" if is_positive else " nonnegative integer"
        raise EnvironmentConfigurationError(f"{name} must be{qualifier}")
    return result


def get_episode_schedule(
    dataset: OptionDataset,
    *,
    is_repeated: bool = False,
    num_training_episode: int | None = None,
    seed: int | None = None,
) -> pd.DataFrame:
    """Build a chronological validation/test schedule or a seeded training permutation; repetition requires an explicit training budget."""
    if not isinstance(dataset, OptionDataset):
        raise EnvironmentConfigurationError("dataset must be an OptionDataset")
    if dataset.num_episode == 0:
        raise EnvironmentConfigurationError("The environment cannot use an empty OptionDataset")
    if not isinstance(is_repeated, (bool, np.bool_)):
        raise EnvironmentConfigurationError("is_repeated must be a boolean")
    is_repeated = bool(is_repeated)
    rng = np.random.default_rng(seed)
    if dataset.label != "train":
        if is_repeated or num_training_episode is not None:
            raise EnvironmentConfigurationError(
                "Validation and testing require deterministic traversal without training repetition"
            )
        manifest = dataset.episode_manifest
        order = pd.DataFrame(
            {
                "episode_index": np.arange(dataset.num_episode, dtype=np.int64),
                "t0": pd.to_datetime(manifest["t0"]),
                "cohort_id": pd.to_datetime(manifest["cohort_id"]),
                "selection_rank": pd.to_numeric(manifest["selection_rank"], errors="raise"),
                "episode_id": manifest["episode_id"].astype(str),
            }
        ).sort_values(["t0", "cohort_id", "selection_rank", "episode_id"], kind="mergesort")
        indices = order["episode_index"].to_numpy(dtype=np.int64, copy=True)
    elif not is_repeated:
        if num_training_episode is not None:
            raise EnvironmentConfigurationError(
                "num_training_episode must be None when is_repeated=False"
            )
        indices = rng.permutation(dataset.num_episode).astype(np.int64, copy=False)
    else:
        if num_training_episode is None:
            raise EnvironmentConfigurationError(
                "num_training_episode is required when is_repeated=True"
            )
        num_draw = _get_nonnegative_int(
            num_training_episode, "num_training_episode", is_positive=True
        )
        manifest = dataset.episode_manifest
        cohorts = pd.Index(manifest["cohort_id"].drop_duplicates())
        cohort_positions = {
            cohort: positions.to_numpy(dtype=np.int64)
            for cohort, positions in manifest.groupby("cohort_id", sort=False).groups.items()
        }
        sampled_indices: list[int] = []
        for _ in range(num_draw):
            num_cohort = int(rng.integers(0, len(cohorts)))
            positions = cohort_positions[cohorts[num_cohort]]
            num_position = int(rng.integers(0, len(positions)))
            sampled_indices.append(int(positions[num_position]))
        indices = np.asarray(sampled_indices, dtype=np.int64)
    manifest_rows = dataset.episode_manifest.iloc[indices]
    schedule = pd.DataFrame(
        {
            "draw_id": np.arange(len(indices), dtype=np.int64),
            "episode_index": indices,
            "episode_id": manifest_rows["episode_id"].astype(str).to_numpy(),
            "cohort_id": pd.to_datetime(manifest_rows["cohort_id"]).to_numpy(),
        }
    )
    schedule["draw_count"] = schedule.groupby("episode_id", sort=False).cumcount() + 1
    schedule["draw_count"] = schedule["draw_count"].astype(np.int64)
    return schedule
