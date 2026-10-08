"""Test schedule for the paper option-hedging pipeline."""

from __future__ import annotations
import pytest
from hedging_env import EnvironmentConfigurationError, get_episode_schedule
from .helpers import get_synthetic_dataset


def test_non_repeated_train_schedule_is_seeded_permutation() -> None:
    """Verify non repeated train schedule is seeded permutation."""
    dataset = get_synthetic_dataset("train")
    first = get_episode_schedule(dataset, seed=123)
    second = get_episode_schedule(dataset, seed=123)
    assert first.equals(second)
    assert len(first) == dataset.num_episode
    assert first["episode_id"].nunique() == dataset.num_episode
    assert first["draw_count"].eq(1).all()


def test_repeated_train_schedule_uses_requested_rollout_count() -> None:
    """Verify repeated train schedule uses requested rollout count."""
    dataset = get_synthetic_dataset("train")
    schedule = get_episode_schedule(dataset, is_repeated=True, num_training_episode=20, seed=9)
    assert len(schedule) == 20
    assert schedule["draw_id"].tolist() == list(range(20))
    assert set(schedule["episode_id"]).issubset(set(dataset.get_episode_ids()))
    assert schedule.groupby("episode_id")["draw_count"].max().sum() == 20


def test_validation_schedule_is_deterministic_and_rejects_repetition() -> None:
    """Verify validation schedule is deterministic and rejects repetition."""
    dataset = get_synthetic_dataset("valid")
    schedule = get_episode_schedule(dataset, seed=999)
    assert schedule["episode_id"].tolist() == list(dataset.get_episode_ids())
    with pytest.raises(EnvironmentConfigurationError, match="Validation and testing"):
        get_episode_schedule(dataset, is_repeated=True, num_training_episode=3)
