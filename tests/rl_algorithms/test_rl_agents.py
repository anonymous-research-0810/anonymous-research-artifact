"""Test rl agents for the paper option-hedging pipeline."""

from __future__ import annotations
import numpy as np
import pytest
import torch
from rl_agents import SACHedgingAgent, TD3HedgingAgent


def _set_module_parameters_to_zero(module: torch.nn.Module) -> None:
    """Set module parameters to zero."""
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.zero_()


@pytest.mark.parametrize("agent_class", [SACHedgingAgent, TD3HedgingAgent])
def test_zero_delta_residual_recovers_bsm_delta(agent_class: type) -> None:
    """Verify zero delta residual recovers BSM delta."""
    agent = agent_class(is_delta_residual=True, device="cpu", seed=1)
    _set_module_parameters_to_zero(agent.actor)
    state = np.zeros((3, 7), dtype=np.float32)
    delta = np.array([0.2, 0.5, 0.8], dtype=np.float32)
    action = agent.get_action(state, delta_action=delta, is_deterministic=True)
    assert action == pytest.approx(delta)
    assert np.all((action >= 0.0) & (action <= 1.0))


@pytest.mark.parametrize("agent_class", [SACHedgingAgent, TD3HedgingAgent])
def test_default_max_delta_residual_is_one_tenth(agent_class: type) -> None:
    """Verify default max delta residual is one tenth."""
    agent = agent_class(device="cpu", seed=101)
    assert agent.max_delta_residual == pytest.approx(0.1)


def test_td3_uses_minimum_target_q_and_delayed_policy_update() -> None:
    """Verify TD3 uses minimum target q and delayed policy update."""
    agent = TD3HedgingAgent(
        actor_hidden_dims=(8,),
        critic_hidden_dims=(8,),
        actor_learning_rate=1e-12,
        critic_learning_rate=1e-12,
        target_policy_noise_std=0.0,
        num_policy_delay=2,
        max_gradient_norm=0.0001,
        device="cpu",
        seed=2,
    )
    for network in (agent.q1, agent.q2, agent.target_q1, agent.target_q2):
        _set_module_parameters_to_zero(network)
    with torch.no_grad():
        agent.target_q1.network[-1].bias.fill_(2.0)
        agent.target_q2.network[-1].bias.fill_(5.0)
    batch = {
        "state": np.zeros((1, 7), dtype=np.float32),
        "action": np.zeros((1, 1), dtype=np.float32),
        "reward": np.full((1, 1), 3.0, dtype=np.float32),
        "next_state": np.zeros((1, 7), dtype=np.float32),
        "terminated": np.zeros((1, 1), dtype=np.float32),
        "delta_action": np.zeros((1, 1), dtype=np.float32),
        "next_delta_action": np.zeros((1, 1), dtype=np.float32),
    }
    first_metrics = agent.update(batch)
    second_metrics = agent.update(batch)
    assert first_metrics["q1_loss"] == pytest.approx(5.0**2)
    assert first_metrics["q2_loss"] == pytest.approx(5.0**2)
    assert first_metrics["is_actor_updated"] == pytest.approx(0.0)
    assert second_metrics["is_actor_updated"] == pytest.approx(1.0)
    metrics = second_metrics
    assert metrics["critic_gradient_norm"] > agent.max_gradient_norm
    clipped_norm = torch.sqrt(
        sum(
            (
                parameter.grad.detach().pow(2).sum()
                for network in (agent.q1, agent.q2)
                for parameter in network.parameters()
                if parameter.grad is not None
            )
        )
    )
    assert clipped_norm.item() <= agent.max_gradient_norm * 1.01


def test_network_output_layers_use_small_configurable_initialization() -> None:
    """Verify network output layers use small configurable initialization."""
    agent = TD3HedgingAgent(
        actor_hidden_dims=(8,),
        critic_hidden_dims=(8,),
        actor_output_gain=0.02,
        critic_output_gain=0.03,
        device="cpu",
        seed=21,
    )
    assert agent.actor.network[-1].weight.norm().item() == pytest.approx(0.02)
    assert agent.q1.network[-1].weight.norm().item() == pytest.approx(0.03)
    assert agent.get_config()["actor_output_gain"] == pytest.approx(0.02)
    assert agent.get_config()["critic_output_gain"] == pytest.approx(0.03)


def test_checkpoint_round_trip_preserves_deterministic_action(tmp_path) -> None:
    """Verify checkpoint round trip preserves deterministic action."""
    agent = SACHedgingAgent(
        actor_hidden_dims=(16,),
        critic_hidden_dims=(16,),
        is_delta_residual=True,
        device="cpu",
        seed=3,
    )
    state = np.linspace(-0.2, 0.2, 7, dtype=np.float32)
    expected = agent.get_action(state, delta_action=0.55)
    path = tmp_path / "sac.pt"
    agent.save(path)
    restored = SACHedgingAgent(
        actor_hidden_dims=(16,),
        critic_hidden_dims=(16,),
        is_delta_residual=True,
        device="cpu",
        seed=4,
    )
    restored.load(path)
    assert restored.get_action(state, delta_action=0.55) == pytest.approx(expected)
