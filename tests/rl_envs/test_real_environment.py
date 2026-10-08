"""Test real environment for the paper option-hedging pipeline."""

from __future__ import annotations
from pathlib import Path
import os
import numpy as np
import pytest
from option_dataset import OptionDataset, get_option_dataset
from hedging_env import HedgingEnv

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("MDH_TEST_DATA_ROOT", REPOSITORY_ROOT / "data"))
IS_REAL_DATA_AVAILABLE = (DATA_ROOT / "spx_options_2024.parquet").is_file()


@pytest.fixture(scope="module")
def real_dataset() -> OptionDataset:
    """Real dataset."""
    return get_option_dataset(
        date_period=["2024-05-30", "2024-06-28"], label="test", data_root=DATA_ROOT
    )


@pytest.mark.integration
@pytest.mark.skipif(not IS_REAL_DATA_AVAILABLE, reason="local SPX parquet data unavailable")
def test_real_batch_no_hedge_reconciles_accounting_and_cash_flow(
    real_dataset: OptionDataset,
) -> None:
    """Verify real batch no hedge reconciles accounting and cash flow."""
    dataset = real_dataset
    env = HedgingEnv(
        dataset,
        reward_formulation="shaped_accounting",
        reward_risk_aversion_xi=0.0,
        hedge_cost_rate=0.0005,
    )
    state, _ = env.reset()
    assert dataset.num_episode == 3
    assert state.shape == (dataset.num_episode, 7)
    for num_step in range(dataset.config.num_interval):
        state, reward, terminated, truncated, _ = env.step(np.zeros(dataset.num_episode))
        assert reward.shape == (dataset.num_episode,)
        assert not truncated.any()
        assert terminated.all() == (num_step == dataset.config.num_interval - 1)
    results = env.get_completed_episode_results()
    assert len(results) == dataset.num_episode
    assert results["reconciliation_error"].max() <= 1e-08
    expected_wealth = []
    for num_episode in range(dataset.num_episode):
        episode = dataset[num_episode]
        num_days = np.diff(episode["date"].to_numpy()).astype("timedelta64[D]").astype(float)
        funding = episode.iloc[:-1]["funding_rate"].to_numpy(dtype=float)
        discount_terminal = float(np.exp(np.sum(-funding * num_days / 365.0)))
        expected_wealth.append(
            float(episode.iloc[0]["option_mid_norm"])
            - discount_terminal * float(episode.iloc[-1]["option_mid_norm"])
        )
    assert results["wealth"].to_numpy() == pytest.approx(expected_wealth, abs=1e-12)
    assert len(env.get_trajectory()) == dataset.num_episode * dataset.config.num_interval


@pytest.mark.integration
@pytest.mark.skipif(not IS_REAL_DATA_AVAILABLE, reason="local SPX parquet data unavailable")
def test_real_varying_actions_match_between_returned_formulations(
    real_dataset: OptionDataset,
) -> None:
    """Verify real varying actions match between returned formulations."""
    accounting_env = HedgingEnv(
        real_dataset,
        reward_formulation="shaped_accounting",
        reward_risk_aversion_xi=0.0,
        observation_dtype=np.float64,
    )
    cash_flow_env = HedgingEnv(
        real_dataset, reward_formulation="cash_flow", observation_dtype=np.float64
    )
    accounting_env.reset()
    cash_flow_env.reset()
    accounting_reward_sum = np.zeros(real_dataset.num_episode)
    cash_flow_reward_sum = np.zeros(real_dataset.num_episode)
    for num_step in range(real_dataset.config.num_interval):
        base_action = 0.15 + 0.7 * num_step / (real_dataset.config.num_interval - 1)
        actions = np.clip(
            base_action + np.linspace(-0.05, 0.05, real_dataset.num_episode), 0.0, 1.0
        )
        _, accounting_reward, _, _, _ = accounting_env.step(actions)
        _, cash_flow_reward, _, _, _ = cash_flow_env.step(actions)
        accounting_reward_sum += accounting_reward
        cash_flow_reward_sum += cash_flow_reward
    accounting_results = accounting_env.get_completed_episode_results()
    cash_flow_results = cash_flow_env.get_completed_episode_results()
    assert accounting_reward_sum == pytest.approx(
        accounting_env.training_reward_scale * accounting_results["wealth"].to_numpy(), abs=1e-12
    )
    assert cash_flow_reward_sum == pytest.approx(
        cash_flow_env.training_reward_scale * cash_flow_results["wealth"].to_numpy(), abs=1e-12
    )
    assert accounting_results["wealth"].to_numpy() == pytest.approx(
        cash_flow_results["wealth"].to_numpy(), abs=1e-12
    )
    assert accounting_results["reconciliation_error"].max() <= 1e-08
    assert cash_flow_results["reconciliation_error"].max() <= 1e-08


@pytest.mark.integration
@pytest.mark.skipif(not IS_REAL_DATA_AVAILABLE, reason="local SPX parquet data unavailable")
def test_real_bsm_delta_benchmark_completes_all_episodes(real_dataset: OptionDataset) -> None:
    """Verify real BSM delta benchmark completes all episodes."""
    env = HedgingEnv(
        real_dataset,
        reward_formulation="shaped_accounting",
        reward_risk_aversion_xi=0.0,
        observation_dtype=np.float64,
    )
    env.reset()
    for _ in range(real_dataset.config.num_interval):
        actions = env.get_bsm_call_delta()
        assert np.isfinite(actions).all()
        assert ((0.0 <= actions) & (actions <= 1.0)).all()
        env.step(actions)
    results = env.get_completed_episode_results()
    assert len(results) == real_dataset.num_episode
    assert results["reconciliation_error"].max() <= 1e-08
