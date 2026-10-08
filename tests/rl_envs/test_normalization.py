"""Test normalization for the paper option-hedging pipeline."""

from __future__ import annotations
import numpy as np
import pytest
from hedging_env import (
    EnvironmentConfigurationError,
    HedgingEnv,
    get_state_normalizer,
    get_state_normalizer_from_json,
)
from .helpers import get_synthetic_dataset


def test_state_normalizer_uses_only_train_decision_rows() -> None:
    """Verify state normalizer uses only train decision rows."""
    train_dataset = get_synthetic_dataset("train")
    normalizer = get_state_normalizer(train_dataset)
    decision_rows = train_dataset.episode_steps.loc[
        ~train_dataset.episode_steps["is_terminal"], normalizer.feature_names
    ].to_numpy(dtype=np.float64)
    assert normalizer.num_observation == 4
    assert np.asarray(normalizer.mean) == pytest.approx(decision_rows.mean(axis=0))
    assert np.asarray(normalizer.std) == pytest.approx(decision_rows.std(axis=0, ddof=0))
    transformed = normalizer.transform(decision_rows, dtype=np.float64)
    assert transformed.mean(axis=0) == pytest.approx(np.zeros(6), abs=1e-12)
    assert np.all(np.asarray(normalizer.scale) > 0)


def test_state_normalizer_rejects_non_train_dataset() -> None:
    """Verify state normalizer rejects non train dataset."""
    with pytest.raises(EnvironmentConfigurationError, match="train"):
        get_state_normalizer(get_synthetic_dataset("valid"))


def test_state_normalizer_round_trip_json(tmp_path) -> None:
    """Verify state normalizer round trip JSON."""
    normalizer = get_state_normalizer(get_synthetic_dataset("train"))
    path = normalizer.save(tmp_path / "state_normalizer.json")
    restored = get_state_normalizer_from_json(path)
    values = np.asarray([[1.0, 1.0, 0.01, 0.2, 0.03, 0.0]])
    assert restored.get_dict() == normalizer.get_dict()
    assert restored.transform(values) == pytest.approx(normalizer.transform(values))
    with pytest.raises(FileExistsError):
        normalizer.save(path)


def test_environment_normalizes_only_exogenous_features() -> None:
    """Verify environment normalizes only exogenous features."""
    train_dataset = get_synthetic_dataset("train", num_episode=1)
    normalizer = get_state_normalizer(train_dataset)
    env = HedgingEnv(train_dataset, state_normalizer=normalizer, hedge_cost_rate=0.0, seed=7)
    state, _ = env.reset()
    next_state, _, terminated, _, _ = env.step(0.4)
    assert state.shape == (7,)
    assert state[2] == pytest.approx(0.0)
    assert not terminated
    assert next_state[2] == pytest.approx(0.4)
    raw_external = train_dataset.get_exogenous_state_array(0, dtype=np.float64)[1]
    expected_external = normalizer.transform(raw_external)
    assert next_state[[0, 1, 3, 4, 5, 6]] == pytest.approx(expected_external, abs=1e-07)
