"""Test environment for the paper option-hedging pipeline."""

from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
from hedging_env import (
    DEFAULT_TRAINING_REWARD_SCALES,
    EnvironmentConfigurationError,
    EnvironmentStateError,
    HedgingEnv,
    get_episode_schedule,
)
from .helpers import get_constant_dataset, get_synthetic_dataset


def test_validation_schedule_explicitly_sorts_episode_by_time() -> None:
    """Verify validation schedule explicitly sorts episode by time."""
    dataset = get_synthetic_dataset("valid")
    dataset.episode_manifest = dataset.episode_manifest.iloc[::-1].reset_index(drop=True)
    schedule = get_episode_schedule(dataset)
    rows = dataset.episode_manifest.iloc[schedule["episode_index"].to_numpy(dtype=np.int64)]
    assert rows["episode_id"].tolist() == ["synthetic_up", "synthetic_down"]
    assert pd.DatetimeIndex(rows["t0"]).is_monotonic_increasing


def test_accounting_rewards_match_hand_calculation() -> None:
    """Verify accounting rewards match hand calculation."""
    env = HedgingEnv(
        get_synthetic_dataset("train", num_episode=1),
        reward_formulation="shaped_accounting",
        reward_risk_aversion_xi=0.0,
        hedge_cost_rate=0.01,
        observation_dtype=np.float64,
        seed=1,
    )
    state, reset_info = env.reset()
    next_state, first_reward, first_terminated, first_truncated, first_info = env.step(0.5)
    terminal_state, second_reward, terminated, truncated, second_info = env.step(np.asarray([0.5]))
    assert state == pytest.approx([1.0, 1.0, 0.0, 3 / 365, 0.2, 0.03, 0.0])
    assert reset_info["episode_id"] == "synthetic_up"
    assert next_state[2] == pytest.approx(0.5)
    assert first_reward == pytest.approx(-0.5)
    assert first_info["accounting_reward"] == pytest.approx(-0.005)
    assert first_info["training_reward"] == pytest.approx(first_reward)
    assert first_info["training_reward_scale"] == pytest.approx(100.0)
    assert first_info["rebalance_cost"] == pytest.approx(0.005)
    assert not first_terminated and (not first_truncated)
    assert second_reward == pytest.approx(-0.6)
    assert second_info["accounting_reward"] == pytest.approx(-0.006)
    assert second_info["training_reward"] == pytest.approx(second_reward)
    assert second_info["liquidation_cost"] == pytest.approx(0.006)
    assert terminated and (not truncated)
    assert terminal_state == pytest.approx(np.zeros(7))
    result = env.get_completed_episode_results().iloc[0]
    assert result["accounting_wealth"] == pytest.approx(-0.011)
    assert result["cash_flow_wealth"] == pytest.approx(-0.011)
    assert result["reconciliation_error"] <= 1e-12
    assert result["net_loss"] == pytest.approx(0.011)
    assert result["replication_cost"] == pytest.approx(0.111)
    assert result["transaction_cost"] == pytest.approx(0.011)
    assert result["monetary_loss"] == pytest.approx(110.0)
    assert result["monetary_transaction_cost"] == pytest.approx(110.0)
    metrics = env.get_metrics()
    assert metrics["mean_loss"] == pytest.approx(110.0)
    assert metrics["mean_transaction_cost"] == pytest.approx(110.0)
    assert env.get_config()["metric_unit"] == "USD_per_option_contract"
    assert env.get_config()["training_reward_scale"] == pytest.approx(100.0)


def test_shaped_accounting_reward_only_changes_learning_signal() -> None:
    """Verify shaped accounting reward only changes learning signal."""
    env = HedgingEnv(
        get_synthetic_dataset("train", num_episode=1),
        reward_formulation="shaped_accounting",
        hedge_cost_rate=0.01,
        reward_risk_aversion_xi=1.5,
        training_reward_scale=100.0,
        observation_dtype=np.float64,
        seed=11,
    )
    env.reset()
    _, first_reward, _, _, first_info = env.step(0.5)
    _, second_reward, terminated, _, second_info = env.step(0.5)
    assert first_reward == pytest.approx(100.0 * (-0.005 - 1.5 * 0.005))
    assert second_reward == pytest.approx(100.0 * (-0.006 - 1.5 * 0.006))
    assert first_info["shaped_accounting_reward"] == pytest.approx(-0.0125)
    assert second_info["shaped_accounting_reward"] == pytest.approx(-0.015)
    assert first_info["training_reward"] == pytest.approx(first_reward)
    assert second_info["training_reward"] == pytest.approx(second_reward)
    assert terminated
    result = env.get_completed_episode_results().iloc[0]
    assert result["wealth"] == pytest.approx(-0.011)
    assert result["net_loss"] == pytest.approx(0.011)
    assert result["monetary_loss"] == pytest.approx(110.0)
    trajectory = env.get_trajectory()
    assert trajectory["shaped_accounting_reward"].tolist() == pytest.approx([-0.0125, -0.015])
    assert trajectory["training_reward"].tolist() == pytest.approx([first_reward, second_reward])


@pytest.mark.parametrize(
    ("reward_formulation", "expected_scale"), tuple(DEFAULT_TRAINING_REWARD_SCALES.items())
)
def test_default_training_reward_scale_is_resolved_by_formulation(
    reward_formulation: str, expected_scale: float
) -> None:
    """Verify default training reward scale is resolved by formulation."""
    env = HedgingEnv(
        get_synthetic_dataset("train", num_episode=1),
        reward_formulation=reward_formulation,
        observation_dtype=np.float64,
        seed=12,
    )
    _, reset_info = env.reset()
    _, reward, _, _, info = env.step(0.5)
    unscaled_key = {
        "accounting": "accounting_reward",
        "shaped_accounting": "shaped_accounting_reward",
        "cash_flow": "cash_flow_reward",
    }[reward_formulation]
    assert env.training_reward_scale == pytest.approx(expected_scale)
    assert env.get_config()["training_reward_scale"] == pytest.approx(expected_scale)
    assert reset_info["training_reward_scale"] == pytest.approx(expected_scale)
    assert reward == pytest.approx(expected_scale * info[unscaled_key])
    assert info["training_reward"] == pytest.approx(reward)


def test_explicit_training_reward_scale_only_changes_agent_input() -> None:
    """Verify explicit training reward scale only changes agent input."""
    environments = [
        HedgingEnv(
            get_synthetic_dataset("train", num_episode=1),
            reward_formulation="shaped_accounting",
            reward_risk_aversion_xi=0.0,
            hedge_cost_rate=0.01,
            training_reward_scale=scale,
            observation_dtype=np.float64,
            seed=13,
        )
        for scale in (1.0, 25.0)
    ]
    returned_rewards = []
    for env in environments:
        env.reset()
        returned_rewards.append([env.step(0.5)[1], env.step(0.5)[1]])
    np.testing.assert_allclose(
        returned_rewards[1], 25.0 * np.asarray(returned_rewards[0]), rtol=0.0, atol=1e-12
    )
    first_result = environments[0].get_completed_episode_results().iloc[0]
    second_result = environments[1].get_completed_episode_results().iloc[0]
    for key in (
        "accounting_wealth",
        "cash_flow_wealth",
        "wealth",
        "net_loss",
        "transaction_cost",
        "monetary_loss",
    ):
        assert second_result[key] == pytest.approx(first_result[key])


def test_cash_flow_rewards_match_hand_calculation() -> None:
    """Verify cash flow rewards match hand calculation."""
    env = HedgingEnv(
        get_synthetic_dataset("train", num_episode=1),
        reward_formulation="cash_flow",
        hedge_cost_rate=0.01,
        observation_dtype=np.float64,
        seed=2,
    )
    env.reset()
    _, first_reward, _, _, first_info = env.step(0.5)
    _, second_reward, terminated, _, second_info = env.step(0.5)
    assert first_reward == pytest.approx(-0.405)
    assert second_reward == pytest.approx(0.394)
    assert first_info["accounting_reward"] == pytest.approx(-0.005)
    assert second_info["accounting_reward"] == pytest.approx(-0.006)
    assert terminated
    result = env.get_completed_episode_results().iloc[0]
    assert result["wealth"] == pytest.approx(-0.011)
    assert result["accounting_wealth"] == pytest.approx(result["cash_flow_wealth"])


def test_constant_zero_carry_path_has_zero_terminal_wealth_without_cost() -> None:
    """Verify constant zero carry path has zero terminal wealth without cost."""
    env = HedgingEnv(
        get_constant_dataset(),
        reward_formulation="cash_flow",
        hedge_cost_rate=0.0,
        observation_dtype=np.float64,
        seed=6,
    )
    env.reset()
    _, first_reward, _, _, first_info = env.step(0.5)
    _, second_reward, terminated, _, second_info = env.step(0.5)
    assert first_reward == pytest.approx(-0.5)
    assert second_reward == pytest.approx(0.5)
    assert first_info["accounting_reward"] == pytest.approx(0.0)
    assert second_info["accounting_reward"] == pytest.approx(0.0)
    assert terminated
    assert env.get_completed_episode_results().iloc[0]["wealth"] == pytest.approx(0.0)


def test_batch_evaluation_returns_per_episode_losses_and_pooled_metrics() -> None:
    """Verify batch evaluation returns per episode losses and pooled metrics."""
    env = HedgingEnv(
        get_synthetic_dataset("valid"),
        reward_formulation="shaped_accounting",
        reward_risk_aversion_xi=0.0,
        hedge_cost_rate=0.0,
        observation_dtype=np.float64,
    )
    state, info = env.reset()
    assert state.shape == (2, 7)
    assert info["episode_id"].tolist() == ["synthetic_up", "synthetic_down"]
    next_state, reward, terminated, truncated, _ = env.step(np.zeros((2, 1)))
    assert next_state.shape == (2, 7)
    assert reward.shape == (2,)
    assert not terminated.any() and (not truncated.any())
    terminal_state, _, terminated, truncated, _ = env.step(np.zeros(2))
    assert terminal_state.shape == (2, 7)
    assert np.all(terminal_state == 0)
    assert terminated.all() and (not truncated.any())
    results = env.get_completed_episode_results()
    assert results["net_loss"].to_numpy() == pytest.approx([0.1, -0.1])
    metrics = env.get_metrics()
    assert metrics["num_episode"] == 2
    assert metrics["mean_loss"] == pytest.approx(-500.0)
    assert metrics["std_loss"] == pytest.approx(1500.0)
    assert metrics["j_lambda"] == pytest.approx(1750.0)
    assert metrics["mean_transaction_cost"] == pytest.approx(0.0)
    with pytest.raises(StopIteration):
        env.reset()


def test_chunked_evaluation_accumulates_results_across_resets() -> None:
    """Verify chunked evaluation accumulates results across resets."""
    env = HedgingEnv(get_synthetic_dataset("valid"), num_parallel_episode=1, hedge_cost_rate=0.0)
    for num_batch in range(2):
        state, _ = env.reset()
        assert state.shape == (1, 7)
        env.step(np.zeros(1))
        env.step(np.zeros(1))
        if num_batch == 0:
            with pytest.raises(EnvironmentStateError, match="is incomplete"):
                env.get_metrics()
    assert env.is_schedule_exhausted
    assert env.num_completed_episode == 2
    assert len(env.get_completed_episode_results()) == 2
    assert env.get_metrics()["std_loss"] == pytest.approx(1500.0)
    assert env.get_config()["num_scheduled_episode"] == 2


def test_action_and_call_order_are_strict_by_default() -> None:
    """Verify action and call order are strict by default."""
    env = HedgingEnv(get_synthetic_dataset("train", num_episode=1), seed=3)
    with pytest.raises(EnvironmentStateError, match="[Rr]eset"):
        env.step(0.5)
    env.reset()
    with pytest.raises(EnvironmentConfigurationError, match="\\[0,1\\]"):
        env.step(1.1)
    with pytest.raises(EnvironmentStateError, match="early reset"):
        env.reset()
    env.step(0.0)
    env.step(0.0)
    with pytest.raises(EnvironmentStateError, match="cannot continue to step"):
        env.step(0.0)
    with pytest.raises(EnvironmentConfigurationError, match="training_reward_scale"):
        HedgingEnv(get_synthetic_dataset("train", num_episode=1), training_reward_scale=0.0)


def test_bsm_delta_uses_raw_current_market_state() -> None:
    """Verify BSM delta uses raw current market state."""
    env = HedgingEnv(get_synthetic_dataset("train", num_episode=1), hedge_cost_rate=0.0, seed=4)
    env.reset()
    delta = env.get_bsm_call_delta()
    assert np.isfinite(delta)
    assert 0.0 <= delta <= 1.0
    env.step(delta)


def test_train_defaults_to_single_and_validation_defaults_to_batch() -> None:
    """Verify train defaults to single and validation defaults to batch."""
    train_env = HedgingEnv(get_synthetic_dataset("train"), seed=5)
    valid_env = HedgingEnv(get_synthetic_dataset("valid"))
    assert not train_env.is_batch_mode
    assert valid_env.is_batch_mode
    with pytest.raises(EnvironmentConfigurationError, match="Training requires"):
        HedgingEnv(get_synthetic_dataset("train"), is_batch_mode=True)
