"""Test features and dataset for the paper option-hedging pipeline."""

from __future__ import annotations
import pandas as pd
import pytest
from market_segmentation import (
    SegmentationConfig,
    get_segmentation_features,
    segment_option_dataset,
)
from .helpers import get_segmentation_test_dataset


def test_realized_volatility_window_always_equals_dataset_h() -> None:
    """Verify realized volatility window always equals dataset h."""
    dataset = get_segmentation_test_dataset(num_interval=3)
    config = SegmentationConfig(min_segment_length=5, bootstrap_repetitions=0)
    features = get_segmentation_features(dataset, config)
    assert features.realized_volatility_window == dataset.config.num_interval == 3
    assert features.feature_frame["date"].iloc[0] == features.daily_spot["date"].iloc[3]
    assert len(features.feature_frame) == len(features.daily_spot) - 3
    assert features.signal.shape == (len(features.feature_frame), 2)


def test_segment_option_dataset_is_an_exact_episode_partition() -> None:
    """Verify segment option dataset is an exact episode partition."""
    dataset = get_segmentation_test_dataset(num_interval=3)
    dates = pd.DatetimeIndex(dataset.episode_steps["date"].drop_duplicates())
    periods = [(dates[0], dates[11]), (dates[12], dates[-1])]
    parts = segment_option_dataset(dataset, periods)
    assert len(parts) == 2
    assert [part.num_episode for part in parts] == [3, 3]
    assert (
        tuple((episode_id for part in parts for episode_id in part.get_episode_ids()))
        == dataset.get_episode_ids()
    )
    pd.testing.assert_frame_equal(
        pd.concat([part.episode_steps for part in parts], ignore_index=True), dataset.episode_steps
    )
    assert all((part.config is dataset.config for part in parts))


def test_segment_option_dataset_rejects_uncovered_episode_start() -> None:
    """Verify segment option dataset rejects uncovered episode start."""
    dataset = get_segmentation_test_dataset(num_interval=3)
    dates = pd.DatetimeIndex(dataset.episode_steps["date"].drop_duplicates())
    periods = [(dates[0], dates[7]), (dates[12], dates[-1])]
    with pytest.raises(ValueError, match="not covered by seg_periods"):
        segment_option_dataset(dataset, periods)
