"""Test pelt and pipeline for the paper option-hedging pipeline."""

from __future__ import annotations
import json
import numpy as np
import pytest
from market_segmentation import (
    SegmentationConfig,
    get_seg_periods_from_result,
    load_seg_result,
    run_pelt_segmentation,
    run_market_segmentation,
)
from .helpers import get_segmentation_test_dataset


def _get_regime_signal() -> np.ndarray:
    """Return regime signal."""
    rng = np.random.default_rng(7)
    means = ((-2.0, -1.5), (1.8, -1.0), (-1.2, 1.7), (2.2, 1.8), (0.0, 0.0))
    return np.concatenate(
        [rng.normal(loc=mean, scale=0.15, size=(24, 2)) for mean in means], axis=0
    )


def test_rbf_pelt_return_valid_partitions() -> None:
    """Verify RBF-PELT return valid partitions."""
    signal = _get_regime_signal()
    for algorithm in ("rbf",):
        result = run_pelt_segmentation(
            signal,
            SegmentationConfig(
                algorithm=algorithm, min_segment_length=12, penalty=5.0, bootstrap_repetitions=0
            ),
            num_interval=5,
        )
        assert result.breakpoints[-1] == len(signal)
        starts = (0, *result.breakpoints[:-1])
        lengths = [end - start for start, end in zip(starts, result.breakpoints)]
        assert min(lengths) >= 12
        assert result.penalty_selection["method"] == "explicit"


def test_automatic_slope_penalty_is_positive_and_reproducible() -> None:
    """Verify automatic slope penalty is positive and reproducible."""
    signal = _get_regime_signal()
    config = SegmentationConfig(
        min_segment_length=12,
        num_penalty_grid=30,
        min_slope_points=2,
        bootstrap_repetitions=0,
        random_seed=11,
    )
    first = run_pelt_segmentation(signal, config, num_interval=5)
    second = run_pelt_segmentation(signal, config, num_interval=5)
    assert first.penalty > 0.0
    assert first.penalty_selection["method"] == "slope_heuristic"
    assert first.breakpoints == second.breakpoints
    assert first.penalty == second.penalty
    assert first.penalty_path == second.penalty_path


def test_moving_block_bootstrap_saves_every_repetition() -> None:
    """Verify moving block bootstrap saves every repetition."""
    signal = _get_regime_signal()
    result = run_pelt_segmentation(
        signal,
        SegmentationConfig(
            min_segment_length=12,
            penalty=5.0,
            bootstrap_repetitions=5,
            bootstrap_block_length=5,
            boundary_tolerance=4,
            is_select_stable_plateau=False,
            random_seed=19,
        ),
        num_interval=5,
    )
    assert result.bootstrap["status"] == "completed"
    assert result.bootstrap["num_repetition"] == 5
    assert len(result.bootstrap["bootstrap_breakpoints"]) == 5


def test_pipeline_saves_loadable_json_and_exact_partition(tmp_path) -> None:
    """Verify pipeline saves loadable JSON and exact partition."""
    dataset = get_segmentation_test_dataset(num_interval=3)
    output = run_market_segmentation(
        dataset,
        SegmentationConfig(min_segment_length=5, penalty=1.0, bootstrap_repetitions=0),
        output_root=tmp_path,
        experiment_id="seg_test",
    )
    result_path = output["result_path"]
    assert result_path.name == "seg_test.json"
    saved = load_seg_result(result_path)
    periods = get_seg_periods_from_result(saved)
    assert len(periods) == saved["num_segment"]
    assert saved["dataset"]["config"]["num_interval"] == 3
    assert saved["features"]["realized_volatility_window"] == 3
    assert saved["partition_verification"]["is_exact_episode_partition"] is True
    assert sum((part.num_episode for part in output["subdatasets"])) == len(dataset)
    assert json.loads(result_path.read_text(encoding="utf-8"))["seg_periods"]


def test_pipeline_saves_failure_record_before_reraising(tmp_path) -> None:
    """Verify pipeline saves failure record before reraising."""
    dataset = get_segmentation_test_dataset(num_interval=3)
    with pytest.raises(ValueError, match="Too few feature observations"):
        run_market_segmentation(
            dataset,
            SegmentationConfig(min_segment_length=100, bootstrap_repetitions=0),
            output_root=tmp_path,
            experiment_id="seg_failed",
        )
    failed = json.loads((tmp_path / "seg_failed.json").read_text(encoding="utf-8"))
    assert failed["status"] == "failed"
    assert failed["error_type"] == "ValueError"
    assert failed["dataset"]["num_episode"] == len(dataset)
