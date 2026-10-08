"""Test rl utils for the paper option-hedging pipeline."""

from __future__ import annotations
import numpy as np
import pytest
import torch
import rl_utils
from tests.rl_envs.helpers import get_synthetic_dataset
from rl_agents import SACHedgingAgent, TD3HedgingAgent
from rl_utils import ReplayBuffer, evaluate_hedging_agent, train_hedging_agent
from hedging_env import HedgingEnv


def test_replay_buffer_preserves_final_action_and_both_deltas() -> None:
    """Verify replay buffer preserves final action and both deltas."""
    buffer = ReplayBuffer(num_capacity=4, num_state_feature=7, seed=1)
    buffer.add(np.zeros(7), 0.65, 0.01, np.ones(7), False, delta_action=0.6, next_delta_action=0.62)
    batch = buffer.sample(3)
    assert batch["state"].shape == (3, 7)
    assert batch["action"].shape == (3, 1)
    assert batch["action"].numpy() == pytest.approx(0.65)
    assert batch["delta_action"].numpy() == pytest.approx(0.6)
    assert batch["next_delta_action"].numpy() == pytest.approx(0.62)


@pytest.mark.parametrize(
    ("agent", "training_kwargs"),
    [
        (
            TD3HedgingAgent(
                actor_hidden_dims=(16,),
                critic_hidden_dims=(16,),
                is_delta_residual=True,
                device="cpu",
                seed=11,
            ),
            {"num_batch": 2, "warmup_fraction": 0.0},
        ),
        (
            SACHedgingAgent(
                actor_hidden_dims=(16,),
                critic_hidden_dims=(16,),
                is_delta_residual=True,
                device="cpu",
                seed=12,
            ),
            {"num_batch": 2, "warmup_fraction": 0.0},
        ),
    ],
)
def test_all_agents_train_on_single_episode_environment(
    agent: object, training_kwargs: dict[str, int]
) -> None:
    """Verify all agents train on single episode environment."""
    train_env = HedgingEnv(
        get_synthetic_dataset("train", num_episode=2), hedge_cost_rate=0.0, seed=20
    )
    result = train_hedging_agent(
        agent, train_env, is_show_progress=False, is_print_summary=False, **training_kwargs
    )
    assert result["num_episode"] == 2
    assert result["num_total_step"] == 4
    assert result["num_gradient_update"] > 0
    assert len(result["episode_results"]) == 2
    assert np.isfinite(result["update_history"].select_dtypes("number")).all().all()


def test_batch_evaluation_uses_deterministic_delta_residual_actions() -> None:
    """Verify batch evaluation uses deterministic delta residual actions."""
    agent = TD3HedgingAgent(
        actor_hidden_dims=(8,),
        critic_hidden_dims=(8,),
        is_delta_residual=True,
        device="cpu",
        seed=30,
    )
    env = HedgingEnv(get_synthetic_dataset("valid", num_episode=2), hedge_cost_rate=0.0)
    result = evaluate_hedging_agent(agent, env, is_print_summary=False)
    assert result["metrics"]["num_episode"] == 2
    assert result["num_total_step"] == 2
    assert len(result["episode_results"]) == 2
    assert np.isfinite(list(result["metrics"].values())).all()


def test_evaluation_environment_copy_preserves_training_reward_scale() -> None:
    """Verify evaluation environment copy preserves training reward scale."""
    template = HedgingEnv(
        get_synthetic_dataset("valid", num_episode=2),
        reward_formulation="cash_flow",
        training_reward_scale=3.0,
        hedge_cost_rate=0.0,
    )
    copied = rl_utils._get_evaluation_env_from_template(template)
    assert copied is not template
    assert copied.dataset is template.dataset
    assert copied.reward_formulation == "cash_flow"
    assert copied.training_reward_scale == pytest.approx(3.0)
    assert copied.get_config()["training_reward_scale"] == pytest.approx(3.0)


def test_training_progress_and_evaluation_metrics_are_printed(capsys) -> None:
    """Verify training progress and evaluation metrics are printed."""
    agent = TD3HedgingAgent(actor_hidden_dims=(8,), critic_hidden_dims=(8,), device="cpu", seed=40)
    train_env = HedgingEnv(
        get_synthetic_dataset("train", num_episode=2), hedge_cost_rate=0.0, seed=40
    )
    train_hedging_agent(agent, train_env, num_batch=2, warmup_fraction=0.0)
    captured_train = capsys.readouterr()
    assert "episode samples=2" in captured_train.out
    assert "is_repeated=False" in captured_train.out
    assert "total training steps=4" in captured_train.out
    assert "Training TD3" in captured_train.err
    assert "actor=" in captured_train.err
    assert "critic=" in captured_train.err
    assert "100%" in captured_train.err
    valid_env = HedgingEnv(get_synthetic_dataset("valid", num_episode=2), hedge_cost_rate=0.0)
    evaluate_hedging_agent(agent, valid_env)
    captured_evaluation = capsys.readouterr()
    assert "split=valid, episode samples=2" in captured_evaluation.out
    assert "Mean(L)=" in captured_evaluation.out
    assert "Std(L)=" in captured_evaluation.out
    assert "J_lambda=" in captured_evaluation.out
    assert "Mean(TC)=" in captured_evaluation.out


def test_periodic_validation_saves_all_checkpoints_and_selects_best(tmp_path) -> None:
    """Verify periodic validation saves all checkpoints and selects best."""
    agent = TD3HedgingAgent(actor_hidden_dims=(8,), critic_hidden_dims=(8,), device="cpu", seed=50)
    train_env = HedgingEnv(
        get_synthetic_dataset("train", num_episode=2), hedge_cost_rate=0.0, seed=50
    )
    num_factory_call = 0

    def get_valid_env() -> HedgingEnv:
        """Return valid env."""
        nonlocal num_factory_call
        num_factory_call += 1
        return HedgingEnv(get_synthetic_dataset("valid", num_episode=2), hedge_cost_rate=0.0)

    result = train_hedging_agent(
        agent,
        train_env,
        get_valid_env=get_valid_env,
        num_valid_episodes=1,
        num_batch=2,
        warmup_fraction=0.0,
        checkpoint_dir=tmp_path / "checkpoints",
        is_show_progress=False,
        is_print_summary=False,
    )
    assert num_factory_call == 2
    assert result["validation_history"]["num_train_episode"].tolist() == [1, 2]
    assert result["validation_history"]["num_episode"].tolist() == [2, 2]
    assert result["validation_history"]["is_best"].sum() == 1
    assert result["validation"]["metrics"]["num_episode"] == 2
    assert result["best_validation"] is result["validation"]
    assert result["last_validation"]["metrics"]["num_episode"] == 2
    assert (tmp_path / "checkpoints" / "agent_episode_00000001.pt").is_file()
    assert (tmp_path / "checkpoints" / "agent_episode_00000002.pt").is_file()
    assert (tmp_path / "checkpoints" / "best_agent.pt").is_file()


def test_periodic_validation_and_checkpoint_are_skipped_during_warmup(tmp_path) -> None:
    """Verify periodic validation and checkpoint are skipped during warmup."""
    agent = TD3HedgingAgent(actor_hidden_dims=(8,), critic_hidden_dims=(8,), device="cpu", seed=55)
    train_env = HedgingEnv(
        get_synthetic_dataset("train", num_episode=2), hedge_cost_rate=0.0, seed=55
    )
    num_factory_call = 0

    def get_valid_env() -> HedgingEnv:
        """Return valid env."""
        nonlocal num_factory_call
        num_factory_call += 1
        return HedgingEnv(get_synthetic_dataset("valid", num_episode=2), hedge_cost_rate=0.0)

    result = train_hedging_agent(
        agent,
        train_env,
        get_valid_env=get_valid_env,
        num_valid_episodes=1,
        num_batch=2,
        warmup_fraction=0.75,
        checkpoint_dir=tmp_path / "checkpoints",
        is_show_progress=False,
        is_print_summary=False,
    )
    assert result["config"]["num_warmup_step"] == 3
    assert num_factory_call == 1
    assert result["validation_history"]["num_train_episode"].tolist() == [2]
    assert not (tmp_path / "checkpoints" / "agent_episode_00000001.pt").exists()
    assert (tmp_path / "checkpoints" / "agent_episode_00000002.pt").is_file()


def test_training_restores_earlier_best_checkpoint_when_final_is_worse(
    tmp_path, monkeypatch
) -> None:
    """Verify training restores earlier best checkpoint when final is worse."""
    agent_kwargs = {
        "actor_hidden_dims": (8,),
        "critic_hidden_dims": (8,),
        "device": "cpu",
        "seed": 60,
    }
    agent = TD3HedgingAgent(**agent_kwargs)
    train_env = HedgingEnv(
        get_synthetic_dataset("train", num_episode=2), hedge_cost_rate=0.0, seed=60
    )
    num_evaluation = 0

    def get_valid_env() -> HedgingEnv:
        """Return valid env."""
        return HedgingEnv(get_synthetic_dataset("valid", num_episode=2), hedge_cost_rate=0.0)

    def get_controlled_evaluation(agent, env, **kwargs):
        """Return controlled evaluation."""
        nonlocal num_evaluation
        num_evaluation += 1
        j_lambda = float(num_evaluation)
        return {
            "metrics": {
                "num_episode": 2,
                "mean_loss": j_lambda,
                "std_loss": 0.0,
                "risk_aversion_lambda": 1.5,
                "j_lambda": j_lambda,
                "mean_transaction_cost": 0.0,
            }
        }

    monkeypatch.setattr(rl_utils, "evaluate_hedging_agent", get_controlled_evaluation)
    selected_path = tmp_path / "selected_agent.pt"
    result = train_hedging_agent(
        agent,
        train_env,
        get_valid_env=get_valid_env,
        num_valid_episodes=1,
        num_batch=2,
        warmup_fraction=0.0,
        checkpoint_path=selected_path,
        checkpoint_dir=tmp_path / "checkpoints",
        is_show_progress=False,
        is_print_summary=False,
    )
    first_checkpoint = TD3HedgingAgent(**agent_kwargs)
    first_checkpoint.load(
        tmp_path / "checkpoints" / "agent_episode_00000001.pt", is_load_optimizer=False
    )
    final_checkpoint = TD3HedgingAgent(**agent_kwargs)
    final_checkpoint.load(
        tmp_path / "checkpoints" / "agent_episode_00000002.pt", is_load_optimizer=False
    )
    selected_checkpoint = TD3HedgingAgent(**agent_kwargs)
    selected_checkpoint.load(selected_path, is_load_optimizer=False)
    assert result["num_best_validation_episode"] == 1
    assert result["validation_history"]["is_best"].tolist() == [True, False]
    assert any(
        (
            not torch.equal(first, final)
            for first, final in zip(
                first_checkpoint.actor.parameters(), final_checkpoint.actor.parameters()
            )
        )
    )
    for expected, selected, in_memory in zip(
        first_checkpoint.actor.parameters(),
        selected_checkpoint.actor.parameters(),
        agent.actor.parameters(),
    ):
        assert torch.equal(expected, selected)
        assert torch.equal(expected, in_memory)
