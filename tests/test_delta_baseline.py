"""Test delta baseline for the paper option-hedging pipeline."""

from __future__ import annotations
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import test_delta
import train_basis
from tests.rl_envs.helpers import get_synthetic_dataset


def get_small_delta_config() -> dict:
    """Return small delta config."""
    config = train_basis.get_default_basis_config()
    config["environment"]["reward_formulations"] = ["shaped_accounting", "cash_flow"]
    config["environment"]["hedge_cost_rate"] = 0.001
    config["environment"]["num_parallel_test_episode"] = 2
    config["environment"]["is_record_test_trajectory"] = True
    config["testing"]["is_save_test_trajectory"] = True
    config["testing"]["is_print_summary"] = False
    return config


def test_delta_agent_returns_exact_environment_delta() -> None:
    """Verify delta agent returns exact environment delta."""
    agent = test_delta.DeltaHedgingAgent()
    state = np.zeros(7, dtype=np.float32)
    batch_state = np.zeros((3, 7), dtype=np.float32)
    batch_delta = np.array([0.15, 0.5, 0.85], dtype=np.float64)
    assert agent.get_action(state, delta_action=0.42) == pytest.approx(0.42)
    np.testing.assert_allclose(
        agent.get_action(batch_state, delta_action=batch_delta),
        batch_delta.astype(np.float32),
        rtol=0.0,
        atol=0.0,
    )
    assert agent.is_require_delta_action
    assert not agent.is_delta_residual
    with pytest.raises(ValueError, match="delta_action"):
        agent.get_action(state)
    with pytest.raises(RuntimeError, match="requiring no training"):
        agent.update({})


def test_delta_agent_uses_environment_bsm_action_in_batch_mode() -> None:
    """Verify delta agent uses environment BSM action in batch mode."""
    dataset = get_synthetic_dataset("test", num_episode=2)
    config = get_small_delta_config()
    env = test_delta.get_delta_test_env(dataset, config, reward_formulation="shaped_accounting")
    agent = test_delta.DeltaHedgingAgent()
    state, _ = env.reset()
    delta = env.get_bsm_call_delta()
    action = agent.get_action(state, delta_action=delta)
    tau = 3.0 / 365.0
    volatility = 0.2
    d1 = (0.03 + 0.5 * volatility**2) * tau / (volatility * math.sqrt(tau))
    expected_delta = 0.5 * (1.0 + math.erf(d1 / math.sqrt(2.0)))
    np.testing.assert_allclose(delta, expected_delta, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(action, np.asarray(delta, dtype=np.float32), rtol=0.0, atol=0.0)
    assert np.asarray(action).shape == (2,)
    assert np.all((np.asarray(action) >= 0.0) & (np.asarray(action) <= 1.0))
    env.step(action)


def test_complete_delta_test_artifact_chain_and_reward_equivalence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify complete delta test artifact chain and reward equivalence."""
    config = get_small_delta_config()

    def get_testing_datasets(config, *, labels=("test",)):
        """Return testing datasets."""
        return {label: get_synthetic_dataset(label, num_episode=2) for label in labels}

    monkeypatch.setattr(test_delta, "get_basis_datasets", get_testing_datasets)
    result = test_delta.run_delta_testing(
        config, output_root=tmp_path, experiment_id="delta_unit_experiment", is_print_summary=False
    )
    experiment_dir = result["experiment_dir"]
    summary = json.loads((experiment_dir / "delta_test_summary.json").read_text(encoding="utf-8"))
    consistency = summary["cross_formulation_consistency"]
    assert summary["num_run"] == 2
    assert consistency["is_consistent"]
    assert consistency["num_episode"] == 2
    assert consistency["max_absolute_net_loss_difference"] <= float(
        config["environment"]["reconciliation_tolerance"]
    )
    episode_frames = {}
    for reward_formulation in ("shaped_accounting", "cash_flow"):
        run_dir = experiment_dir / f"bsm_delta__{reward_formulation}"
        test_record = json.loads((run_dir / "test_config.json").read_text(encoding="utf-8"))
        metrics_record = json.loads((run_dir / "test_metrics.json").read_text(encoding="utf-8"))
        episode_frames[reward_formulation] = pd.read_parquet(
            run_dir / "test_episode_results.parquet"
        )
        assert test_record["status"] == "completed"
        assert test_record["agent"]["algorithm_name"] == "bsm_delta"
        assert not test_record["agent"]["is_trainable"]
        assert test_record["test_environment"]["reward_formulation"] == reward_formulation
        assert test_record["test_environment"]["training_reward_scale"] == pytest.approx(
            config["environment"]["training_reward_scales"][reward_formulation]
        )
        assert metrics_record["metrics"]["num_episode"] == 2
        assert (run_dir / "test_trajectory.parquet").is_file()
    np.testing.assert_allclose(
        episode_frames["shaped_accounting"]["net_loss"],
        episode_frames["cash_flow"]["net_loss"],
        rtol=0.0,
        atol=config["environment"]["reconciliation_tolerance"],
    )
