"""Rl agents for the paper option-hedging pipeline."""

from __future__ import annotations
import math
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Mapping, Sequence
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from torch.nn import functional as F

DEFAULT_NUM_STATE_FEATURE = 7
DEFAULT_NUM_ACTION = 1
DEFAULT_HIDDEN_DIMS = (128, 128)
DEFAULT_OUTPUT_GAIN = 0.01
_LOG_STD_MIN = -20.0
_LOG_STD_MAX = 2.0
_LOG_PROB_EPSILON = 1e-06


class AgentConfigurationError(ValueError):
    """Agent configuration error."""


def _get_activation(activation_name: str) -> type[nn.Module]:
    """Return activation."""
    mapping: dict[str, type[nn.Module]] = {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "elu": nn.ELU,
        "leaky_relu": nn.LeakyReLU,
    }
    key = str(activation_name).lower()
    if key not in mapping:
        raise AgentConfigurationError(f"activation_name must be {sorted(mapping)}  ")
    return mapping[key]


def _get_positive_int(value: Any, name: str) -> int:
    """Return positive int."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise AgentConfigurationError(f"{name} must be a strictly positive integer")
    result = int(value)
    if result <= 0:
        raise AgentConfigurationError(f"{name} must be a strictly positive integer")
    return result


def _get_nonnegative_float(value: Any, name: str) -> float:
    """Return nonnegative float."""
    if isinstance(value, (bool, np.bool_)):
        raise AgentConfigurationError(f"{name} must be finite and nonnegative")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise AgentConfigurationError(f"{name} must be finite and nonnegative") from exc
    if not np.isfinite(result) or result < 0.0:
        raise AgentConfigurationError(f"{name} must be finite and nonnegative")
    return result


def _get_probability(value: Any, name: str, *, is_zero_allowed: bool = True) -> float:
    """Return probability."""
    result = _get_nonnegative_float(value, name)
    lower_is_valid = result >= 0.0 if is_zero_allowed else result > 0.0
    if not lower_is_valid or result > 1.0:
        interval = "[0,1]" if is_zero_allowed else "(0,1]"
        raise AgentConfigurationError(f"{name} must be in {interval}")
    return result


def _get_hidden_dims(hidden_dims: Sequence[int]) -> tuple[int, ...]:
    """Return hidden dims."""
    if isinstance(hidden_dims, (str, bytes)):
        raise AgentConfigurationError(
            "hidden_dims must be a nonempty sequence of positive integers"
        )
    result = tuple((_get_positive_int(value, "hidden_dims element") for value in hidden_dims))
    if not result:
        raise AgentConfigurationError("hidden_dims must contain at least one layer")
    return result


def _initialize_linear_layers(module: nn.Module, *, output_gain: float = 1.0) -> None:
    """Initialize linear layers."""
    linear_layers = [layer for layer in module.modules() if isinstance(layer, nn.Linear)]
    for layer in linear_layers:
        nn.init.orthogonal_(layer.weight, gain=math.sqrt(2.0))
        nn.init.zeros_(layer.bias)
    if linear_layers:
        nn.init.orthogonal_(linear_layers[-1].weight, gain=output_gain)


def _get_mlp(
    num_input: int,
    num_output: int,
    hidden_dims: Sequence[int],
    activation_name: str,
    *,
    output_gain: float = 1.0,
) -> nn.Sequential:
    """Return mlp."""
    activation = _get_activation(activation_name)
    layers: list[nn.Module] = []
    num_previous = num_input
    for num_hidden in hidden_dims:
        layers.extend((nn.Linear(num_previous, num_hidden), activation()))
        num_previous = num_hidden
    layers.append(nn.Linear(num_previous, num_output))
    network = nn.Sequential(*layers)
    _initialize_linear_layers(network, output_gain=output_gain)
    return network


class _DeterministicActor(nn.Module):
    """Bounded deterministic policy for direct holdings or Delta-residual corrections."""

    def __init__(
        self,
        num_state_feature: int,
        hidden_dims: Sequence[int],
        activation_name: str,
        *,
        is_delta_residual: bool,
        max_delta_residual: float,
        output_gain: float,
    ) -> None:
        """Initialize validated configuration and internal state."""
        super().__init__()
        self.is_delta_residual = is_delta_residual
        self.max_delta_residual = max_delta_residual
        self.network = _get_mlp(
            num_state_feature,
            DEFAULT_NUM_ACTION,
            hidden_dims,
            activation_name,
            output_gain=output_gain,
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Compute network outputs for the supplied tensor batch."""
        squashed = torch.tanh(self.network(state))
        if self.is_delta_residual:
            return self.max_delta_residual * squashed
        return 0.5 * (squashed + 1.0)


class _SquashedGaussianActor(nn.Module):
    """State-dependent Gaussian policy with tanh bounds and the action-scale Jacobian."""

    def __init__(
        self,
        num_state_feature: int,
        hidden_dims: Sequence[int],
        activation_name: str,
        *,
        is_delta_residual: bool,
        max_delta_residual: float,
        is_state_independent_log_std: bool,
        initial_log_std: float,
        output_gain: float,
    ) -> None:
        """Initialize validated configuration and internal state."""
        super().__init__()
        self.is_delta_residual = is_delta_residual
        self.max_delta_residual = max_delta_residual
        self.is_state_independent_log_std = is_state_independent_log_std
        num_output = 1 if is_state_independent_log_std else 2
        self.network = _get_mlp(
            num_state_feature, num_output, hidden_dims, activation_name, output_gain=output_gain
        )
        if is_state_independent_log_std:
            self.log_std = nn.Parameter(torch.full((DEFAULT_NUM_ACTION,), float(initial_log_std)))

    @property
    def action_scale(self) -> float:
        """Action scale."""
        return self.max_delta_residual if self.is_delta_residual else 0.5

    def get_distribution_parameters(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return distribution parameters."""
        output = self.network(state)
        if self.is_state_independent_log_std:
            mean = output
            log_std = self.log_std.expand_as(mean)
        else:
            mean, log_std = output.chunk(2, dim=-1)
        return (mean, torch.clamp(log_std, _LOG_STD_MIN, _LOG_STD_MAX))

    def get_policy_action(
        self, state: torch.Tensor, *, is_deterministic: bool
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return policy action."""
        mean, log_std = self.get_distribution_parameters(state)
        distribution = Normal(mean, log_std.exp())
        pre_tanh_action = mean if is_deterministic else distribution.rsample()
        squashed = torch.tanh(pre_tanh_action)
        if self.is_delta_residual:
            policy_action = self.max_delta_residual * squashed
        else:
            policy_action = 0.5 * (squashed + 1.0)
        log_prob = distribution.log_prob(pre_tanh_action)
        log_prob -= torch.log(1.0 - squashed.pow(2) + _LOG_PROB_EPSILON)
        log_prob -= math.log(self.action_scale)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return (policy_action, log_prob, pre_tanh_action)


class _QNetwork(nn.Module):
    """Action-value network for actor-critic updates."""

    def __init__(
        self,
        num_state_feature: int,
        hidden_dims: Sequence[int],
        activation_name: str,
        output_gain: float,
    ) -> None:
        """Initialize validated configuration and internal state."""
        super().__init__()
        self.network = _get_mlp(
            num_state_feature + DEFAULT_NUM_ACTION,
            1,
            hidden_dims,
            activation_name,
            output_gain=output_gain,
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Compute network outputs for the supplied tensor batch."""
        return self.network(torch.cat((state, action), dim=-1))


class BaseHedgingAgent(ABC):
    """Common interface for bounded holdings, optional BSM Delta corrections, and model checkpoints."""

    algorithm_name = "base"
    is_off_policy = False

    def __init__(
        self,
        *,
        num_state_feature: int = DEFAULT_NUM_STATE_FEATURE,
        is_delta_residual: bool = False,
        max_delta_residual: float = 0.1,
        device: str | torch.device | None = None,
        seed: int | None = None,
    ) -> None:
        """Initialize validated configuration and internal state."""
        self.num_state_feature = _get_positive_int(num_state_feature, "num_state_feature")
        if not isinstance(is_delta_residual, (bool, np.bool_)):
            raise AgentConfigurationError("is_delta_residual must be a boolean")
        self.is_delta_residual = bool(is_delta_residual)
        self.is_require_delta_action = self.is_delta_residual
        self.max_delta_residual = _get_nonnegative_float(max_delta_residual, "max_delta_residual")
        if self.max_delta_residual <= 0.0:
            raise AgentConfigurationError("max_delta_residual must be strictly positive")
        self.device = torch.device(
            device if device is not None else "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

    def _get_state_tensor(self, state: Any) -> tuple[torch.Tensor, bool]:
        """Return state tensor."""
        array = np.asarray(state, dtype=np.float32)
        is_single = array.ndim == 1
        if is_single:
            array = array.reshape(1, -1)
        if array.ndim != 2 or array.shape[1] != self.num_state_feature:
            raise AgentConfigurationError(
                f"state shape must be ({self.num_state_feature},) or (N,{self.num_state_feature})"
            )
        if not np.isfinite(array).all():
            raise AgentConfigurationError("state contains nonfinite values")
        return (torch.as_tensor(array, device=self.device), is_single)

    def _get_delta_tensor(
        self, delta_action: Any, *, num_batch: int, is_required: bool | None = None
    ) -> torch.Tensor:
        """Return delta tensor."""
        required = self.is_delta_residual if is_required is None else is_required
        if delta_action is None:
            if required:
                raise AgentConfigurationError("The policy requires the current BSM delta_action")
            return torch.zeros((num_batch, 1), dtype=torch.float32, device=self.device)
        array = np.asarray(delta_action, dtype=np.float32)
        if array.ndim == 0:
            array = np.full((num_batch, 1), float(array), dtype=np.float32)
        elif array.ndim == 1:
            array = array.reshape(-1, 1)
        if array.shape != (num_batch, 1):
            raise AgentConfigurationError(
                f"delta_action shape must be convertible to ({num_batch},1)"
            )
        if not np.isfinite(array).all() or np.any((array < 0.0) | (array > 1.0)):
            raise AgentConfigurationError("delta_action must be finite and in [0,1]")
        return torch.as_tensor(array, device=self.device)

    def _compose_final_action(
        self, policy_action: torch.Tensor, delta_action: torch.Tensor
    ) -> torch.Tensor:
        """Compose final action."""
        if self.is_delta_residual:
            return torch.clamp(delta_action + policy_action, 0.0, 1.0)
        return torch.clamp(policy_action, 0.0, 1.0)

    @staticmethod
    def _get_public_action(action: torch.Tensor, *, is_single: bool) -> float | np.ndarray:
        """Return public action."""
        array = action.detach().cpu().numpy().reshape(-1)
        return float(array[0]) if is_single else array

    def get_random_action(self, delta_action: Any = None) -> float | np.ndarray:
        """Return random action."""
        if self.is_delta_residual:
            delta = np.asarray(delta_action, dtype=np.float32)
            if delta.ndim == 0:
                if not np.isfinite(delta) or not 0.0 <= float(delta) <= 1.0:
                    raise AgentConfigurationError("delta_action must be in [0,1]")
                residual = self._rng.uniform(-self.max_delta_residual, self.max_delta_residual)
                return float(np.clip(delta + residual, 0.0, 1.0))
            if not np.isfinite(delta).all() or np.any((delta < 0.0) | (delta > 1.0)):
                raise AgentConfigurationError("delta_action must be in [0,1]")
            residual = self._rng.uniform(
                -self.max_delta_residual, self.max_delta_residual, size=delta.shape
            )
            return np.clip(delta + residual, 0.0, 1.0).astype(np.float32)
        return float(self._rng.uniform(0.0, 1.0))

    def _get_batch_tensor(self, batch: Mapping[str, Any], key: str) -> torch.Tensor:
        """Return batch tensor."""
        if key not in batch:
            raise AgentConfigurationError(f"Training batch is missing field {key!r}")
        value = batch[key]
        if torch.is_tensor(value):
            return value.to(device=self.device, dtype=torch.float32)
        return torch.as_tensor(value, dtype=torch.float32, device=self.device)

    @abstractmethod
    def get_action(
        self, state: Any, *, delta_action: Any = None, is_deterministic: bool = True
    ) -> float | np.ndarray:
        """Return the policy action, including the Delta correction when enabled."""

    @abstractmethod
    def update(self, batch: Mapping[str, Any]) -> dict[str, float]:
        """Update model parameters from the supplied training batch and return optimization diagnostics."""

    @abstractmethod
    def get_config(self) -> dict[str, Any]:
        """Return the configuration required to reconstruct this object."""

    @abstractmethod
    def _get_checkpoint_state(self) -> dict[str, Any]:
        """Return checkpoint state."""

    @abstractmethod
    def _load_checkpoint_state(
        self, checkpoint: Mapping[str, Any], *, is_load_optimizer: bool
    ) -> None:
        """Load checkpoint state."""

    def save(self, path: str | Path) -> None:
        """Persist the object at the requested destination with explicit overwrite handling."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "algorithm_name": self.algorithm_name,
                "config": self.get_config(),
                **self._get_checkpoint_state(),
            },
            target,
        )

    def load(self, path: str | Path, *, is_load_optimizer: bool = True) -> None:
        """Load a checkpoint and validate its configuration against the current object."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        if checkpoint.get("algorithm_name") != self.algorithm_name:
            raise AgentConfigurationError(
                f"Checkpoint algorithm is {checkpoint.get('algorithm_name')!r}; current agent algorithm is {self.algorithm_name!r}"
            )
        saved_config = checkpoint.get("config", {})
        current_config = self.get_config()
        semantic_keys = (
            "num_state_feature",
            "actor_hidden_dims",
            "critic_hidden_dims",
            "value_hidden_dims",
            "activation_name",
            "is_delta_residual",
            "max_delta_residual",
        )
        for key in semantic_keys:
            if (
                key in saved_config
                and key in current_config
                and (saved_config[key] != current_config[key])
            ):
                raise AgentConfigurationError(
                    f"Checkpoint configuration {key}={saved_config[key]!r} differs from the current model's {current_config[key]!r} differs"
                )
        self._load_checkpoint_state(checkpoint, is_load_optimizer=bool(is_load_optimizer))


class TD3HedgingAgent(BaseHedgingAgent):
    """Twin delayed deterministic policy-gradient agent with clipped target smoothing and delayed actor updates."""

    algorithm_name = "td3"
    is_off_policy = True

    def __init__(
        self,
        *,
        num_state_feature: int = DEFAULT_NUM_STATE_FEATURE,
        actor_hidden_dims: Sequence[int] = DEFAULT_HIDDEN_DIMS,
        critic_hidden_dims: Sequence[int] = DEFAULT_HIDDEN_DIMS,
        activation_name: str = "relu",
        actor_output_gain: float = DEFAULT_OUTPUT_GAIN,
        critic_output_gain: float = DEFAULT_OUTPUT_GAIN,
        actor_learning_rate: float = 0.0003,
        critic_learning_rate: float = 0.0003,
        gamma: float = 1.0,
        tau: float = 0.005,
        exploration_noise_std: float = 0.1,
        target_policy_noise_std: float = 0.2,
        target_noise_clip: float = 0.5,
        num_policy_delay: int = 2,
        max_gradient_norm: float = 1.0,
        is_delta_residual: bool = False,
        max_delta_residual: float = 0.1,
        device: str | torch.device | None = None,
        seed: int | None = None,
    ) -> None:
        """Initialize validated configuration and internal state."""
        super().__init__(
            num_state_feature=num_state_feature,
            is_delta_residual=is_delta_residual,
            max_delta_residual=max_delta_residual,
            device=device,
            seed=seed,
        )
        self.actor_hidden_dims = _get_hidden_dims(actor_hidden_dims)
        self.critic_hidden_dims = _get_hidden_dims(critic_hidden_dims)
        self.activation_name = str(activation_name).lower()
        _get_activation(self.activation_name)
        self.actor_output_gain = _get_nonnegative_float(actor_output_gain, "actor_output_gain")
        self.critic_output_gain = _get_nonnegative_float(critic_output_gain, "critic_output_gain")
        if self.actor_output_gain <= 0.0 or self.critic_output_gain <= 0.0:
            raise AgentConfigurationError("network output gain must be strictly positive")
        self.actor_learning_rate = _get_nonnegative_float(
            actor_learning_rate, "actor_learning_rate"
        )
        self.critic_learning_rate = _get_nonnegative_float(
            critic_learning_rate, "critic_learning_rate"
        )
        if self.actor_learning_rate <= 0.0 or self.critic_learning_rate <= 0.0:
            raise AgentConfigurationError("learning rate must be strictly positive")
        self.gamma = _get_probability(gamma, "gamma")
        self.tau = _get_probability(tau, "tau", is_zero_allowed=False)
        self.exploration_noise_std = _get_nonnegative_float(
            exploration_noise_std, "exploration_noise_std"
        )
        self.target_policy_noise_std = _get_nonnegative_float(
            target_policy_noise_std, "target_policy_noise_std"
        )
        self.target_noise_clip = _get_nonnegative_float(target_noise_clip, "target_noise_clip")
        self.num_policy_delay = _get_positive_int(num_policy_delay, "num_policy_delay")
        self.max_gradient_norm = _get_nonnegative_float(max_gradient_norm, "max_gradient_norm")
        if self.max_gradient_norm <= 0.0:
            raise AgentConfigurationError("max_gradient_norm must be strictly positive")
        actor_kwargs = dict(
            num_state_feature=self.num_state_feature,
            hidden_dims=self.actor_hidden_dims,
            activation_name=self.activation_name,
            is_delta_residual=self.is_delta_residual,
            max_delta_residual=self.max_delta_residual,
            output_gain=self.actor_output_gain,
        )
        self.actor = _DeterministicActor(**actor_kwargs).to(self.device)
        self.target_actor = _DeterministicActor(**actor_kwargs).to(self.device)
        critic_kwargs = dict(
            num_state_feature=self.num_state_feature,
            hidden_dims=self.critic_hidden_dims,
            activation_name=self.activation_name,
            output_gain=self.critic_output_gain,
        )
        self.q1 = _QNetwork(**critic_kwargs).to(self.device)
        self.q2 = _QNetwork(**critic_kwargs).to(self.device)
        self.target_q1 = _QNetwork(**critic_kwargs).to(self.device)
        self.target_q2 = _QNetwork(**critic_kwargs).to(self.device)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_q1.load_state_dict(self.q1.state_dict())
        self.target_q2.load_state_dict(self.q2.state_dict())
        for network in (self.target_actor, self.target_q1, self.target_q2):
            network.requires_grad_(False)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self.actor_learning_rate
        )
        self.critic_optimizer = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=self.critic_learning_rate
        )
        self.num_update = 0

    @property
    def policy_action_scale(self) -> float:
        """Policy action scale."""
        return self.max_delta_residual if self.is_delta_residual else 1.0

    def _clip_policy_action(self, policy_action: torch.Tensor) -> torch.Tensor:
        """Clip policy action."""
        if self.is_delta_residual:
            return torch.clamp(policy_action, -self.max_delta_residual, self.max_delta_residual)
        return torch.clamp(policy_action, 0.0, 1.0)

    def get_action(
        self, state: Any, *, delta_action: Any = None, is_deterministic: bool = True
    ) -> float | np.ndarray:
        """Return the policy action, including the Delta correction when enabled."""
        state_tensor, is_single = self._get_state_tensor(state)
        delta_tensor = self._get_delta_tensor(delta_action, num_batch=state_tensor.shape[0])
        with torch.no_grad():
            policy_action = self.actor(state_tensor)
            if not is_deterministic and self.exploration_noise_std > 0.0:
                noise = torch.as_tensor(
                    self._rng.normal(
                        0.0,
                        self.exploration_noise_std * self.policy_action_scale,
                        size=policy_action.shape,
                    ),
                    dtype=torch.float32,
                    device=self.device,
                )
                policy_action = self._clip_policy_action(policy_action + noise)
            action = self._compose_final_action(policy_action, delta_tensor)
        return self._get_public_action(action, is_single=is_single)

    def update(self, batch: Mapping[str, Any]) -> dict[str, float]:
        """Update model parameters from the supplied training batch and return optimization diagnostics."""
        state = self._get_batch_tensor(batch, "state")
        action = self._get_batch_tensor(batch, "action")
        reward = self._get_batch_tensor(batch, "reward")
        next_state = self._get_batch_tensor(batch, "next_state")
        terminated = self._get_batch_tensor(batch, "terminated")
        delta = self._get_batch_tensor(batch, "delta_action")
        next_delta = self._get_batch_tensor(batch, "next_delta_action")
        with torch.no_grad():
            next_policy_action = self.target_actor(next_state)
            target_noise = torch.randn_like(next_policy_action) * (
                self.target_policy_noise_std * self.policy_action_scale
            )
            noise_limit = self.target_noise_clip * self.policy_action_scale
            target_noise = torch.clamp(target_noise, -noise_limit, noise_limit)
            next_policy_action = self._clip_policy_action(next_policy_action + target_noise)
            next_action = self._compose_final_action(next_policy_action, next_delta)
            target_q = torch.minimum(
                self.target_q1(next_state, next_action), self.target_q2(next_state, next_action)
            )
            target_q = reward + self.gamma * (1.0 - terminated) * target_q
        predicted_q1 = self.q1(state, action)
        predicted_q2 = self.q2(state, action)
        q1_loss = F.mse_loss(predicted_q1, target_q)
        q2_loss = F.mse_loss(predicted_q2, target_q)
        critic_loss = q1_loss + q2_loss
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        critic_gradient_norm = nn.utils.clip_grad_norm_(
            list(self.q1.parameters()) + list(self.q2.parameters()), self.max_gradient_norm
        )
        self.critic_optimizer.step()
        is_actor_updated = (self.num_update + 1) % self.num_policy_delay == 0
        actor_loss_value = 0.0
        actor_gradient_norm_value = 0.0
        if is_actor_updated:
            for network in (self.q1, self.q2):
                network.requires_grad_(False)
            policy_action = self.actor(state)
            policy_final_action = self._compose_final_action(policy_action, delta)
            actor_loss = -self.q1(state, policy_final_action).mean()
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            actor_gradient_norm = nn.utils.clip_grad_norm_(
                self.actor.parameters(), self.max_gradient_norm
            )
            self.actor_optimizer.step()
            for network in (self.q1, self.q2):
                network.requires_grad_(True)
            self._soft_update(self.actor, self.target_actor)
            self._soft_update(self.q1, self.target_q1)
            self._soft_update(self.q2, self.target_q2)
            actor_loss_value = float(actor_loss.detach().cpu())
            actor_gradient_norm_value = float(actor_gradient_norm.detach().cpu())
        self.num_update += 1
        return {
            "critic_loss": float(critic_loss.detach().cpu()),
            "q1_loss": float(q1_loss.detach().cpu()),
            "q2_loss": float(q2_loss.detach().cpu()),
            "actor_loss": actor_loss_value,
            "actor_gradient_norm": actor_gradient_norm_value,
            "critic_gradient_norm": float(critic_gradient_norm.detach().cpu()),
            "is_actor_updated": float(is_actor_updated),
            "mean_target_q": float(target_q.mean().detach().cpu()),
        }

    def _soft_update(self, source: nn.Module, target: nn.Module) -> None:
        """Soft update."""
        with torch.no_grad():
            for source_parameter, target_parameter in zip(source.parameters(), target.parameters()):
                target_parameter.mul_(1.0 - self.tau)
                target_parameter.add_(self.tau * source_parameter)

    def get_config(self) -> dict[str, Any]:
        """Return the configuration required to reconstruct this object."""
        return {
            "algorithm_name": self.algorithm_name,
            "training_objective": "td3_expected_cumulative_reward",
            "num_state_feature": self.num_state_feature,
            "actor_hidden_dims": list(self.actor_hidden_dims),
            "critic_hidden_dims": list(self.critic_hidden_dims),
            "activation_name": self.activation_name,
            "actor_output_gain": self.actor_output_gain,
            "critic_output_gain": self.critic_output_gain,
            "actor_learning_rate": self.actor_learning_rate,
            "critic_learning_rate": self.critic_learning_rate,
            "gamma": self.gamma,
            "tau": self.tau,
            "exploration_noise_std": self.exploration_noise_std,
            "target_policy_noise_std": self.target_policy_noise_std,
            "target_noise_clip": self.target_noise_clip,
            "num_policy_delay": self.num_policy_delay,
            "max_gradient_norm": self.max_gradient_norm,
            "is_delta_residual": self.is_delta_residual,
            "max_delta_residual": self.max_delta_residual,
            "device": str(self.device),
            "seed": self.seed,
        }

    def _get_checkpoint_state(self) -> dict[str, Any]:
        """Return checkpoint state."""
        return {
            "actor": self.actor.state_dict(),
            "target_actor": self.target_actor.state_dict(),
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
            "target_q1": self.target_q1.state_dict(),
            "target_q2": self.target_q2.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "num_update": self.num_update,
        }

    def _load_checkpoint_state(
        self, checkpoint: Mapping[str, Any], *, is_load_optimizer: bool
    ) -> None:
        """Load checkpoint state."""
        for name in ("actor", "target_actor", "q1", "q2", "target_q1", "target_q2"):
            getattr(self, name).load_state_dict(checkpoint[name])
        if is_load_optimizer:
            self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
            self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        self.num_update = int(checkpoint.get("num_update", 0))


class SACHedgingAgent(BaseHedgingAgent):
    """Soft actor-critic agent with bounded Gaussian actions and optional entropy-temperature learning."""

    algorithm_name = "sac"
    is_off_policy = True

    def __init__(
        self,
        *,
        num_state_feature: int = DEFAULT_NUM_STATE_FEATURE,
        actor_hidden_dims: Sequence[int] = DEFAULT_HIDDEN_DIMS,
        critic_hidden_dims: Sequence[int] = DEFAULT_HIDDEN_DIMS,
        activation_name: str = "relu",
        actor_output_gain: float = DEFAULT_OUTPUT_GAIN,
        critic_output_gain: float = DEFAULT_OUTPUT_GAIN,
        actor_learning_rate: float = 0.0003,
        critic_learning_rate: float = 0.0003,
        alpha_learning_rate: float = 0.0003,
        gamma: float = 1.0,
        tau: float = 0.005,
        initial_alpha: float = 0.2,
        target_entropy: float | None = None,
        is_automatic_entropy_tuning: bool = True,
        max_gradient_norm: float = 1.0,
        is_delta_residual: bool = False,
        max_delta_residual: float = 0.1,
        device: str | torch.device | None = None,
        seed: int | None = None,
    ) -> None:
        """Initialize validated configuration and internal state."""
        super().__init__(
            num_state_feature=num_state_feature,
            is_delta_residual=is_delta_residual,
            max_delta_residual=max_delta_residual,
            device=device,
            seed=seed,
        )
        self.actor_hidden_dims = _get_hidden_dims(actor_hidden_dims)
        self.critic_hidden_dims = _get_hidden_dims(critic_hidden_dims)
        self.activation_name = str(activation_name).lower()
        _get_activation(self.activation_name)
        self.actor_output_gain = _get_nonnegative_float(actor_output_gain, "actor_output_gain")
        self.critic_output_gain = _get_nonnegative_float(critic_output_gain, "critic_output_gain")
        if self.actor_output_gain <= 0.0 or self.critic_output_gain <= 0.0:
            raise AgentConfigurationError("network output gain must be strictly positive")
        self.actor_learning_rate = _get_nonnegative_float(
            actor_learning_rate, "actor_learning_rate"
        )
        self.critic_learning_rate = _get_nonnegative_float(
            critic_learning_rate, "critic_learning_rate"
        )
        self.alpha_learning_rate = _get_nonnegative_float(
            alpha_learning_rate, "alpha_learning_rate"
        )
        if min(self.actor_learning_rate, self.critic_learning_rate, self.alpha_learning_rate) <= 0:
            raise AgentConfigurationError("learning rate must be strictly positive")
        self.gamma = _get_probability(gamma, "gamma")
        self.tau = _get_probability(tau, "tau", is_zero_allowed=False)
        self.initial_alpha = _get_nonnegative_float(initial_alpha, "initial_alpha")
        if self.initial_alpha <= 0:
            raise AgentConfigurationError("initial_alpha must be strictly positive")
        if not isinstance(is_automatic_entropy_tuning, (bool, np.bool_)):
            raise AgentConfigurationError("is_automatic_entropy_tuning must be a boolean")
        self.is_automatic_entropy_tuning = bool(is_automatic_entropy_tuning)
        self.max_gradient_norm = _get_nonnegative_float(max_gradient_norm, "max_gradient_norm")
        if self.max_gradient_norm <= 0.0:
            raise AgentConfigurationError("max_gradient_norm must be strictly positive")
        self.actor = _SquashedGaussianActor(
            self.num_state_feature,
            self.actor_hidden_dims,
            self.activation_name,
            is_delta_residual=self.is_delta_residual,
            max_delta_residual=self.max_delta_residual,
            is_state_independent_log_std=False,
            initial_log_std=-0.5,
            output_gain=self.actor_output_gain,
        ).to(self.device)
        if target_entropy is None:
            self.target_entropy = -1.0 + math.log(self.actor.action_scale)
        else:
            self.target_entropy = float(target_entropy)
            if not np.isfinite(self.target_entropy):
                raise AgentConfigurationError("target_entropy must be a finite number")
        self.q1 = _QNetwork(
            self.num_state_feature,
            self.critic_hidden_dims,
            self.activation_name,
            self.critic_output_gain,
        ).to(self.device)
        self.q2 = _QNetwork(
            self.num_state_feature,
            self.critic_hidden_dims,
            self.activation_name,
            self.critic_output_gain,
        ).to(self.device)
        self.target_q1 = _QNetwork(
            self.num_state_feature,
            self.critic_hidden_dims,
            self.activation_name,
            self.critic_output_gain,
        ).to(self.device)
        self.target_q2 = _QNetwork(
            self.num_state_feature,
            self.critic_hidden_dims,
            self.activation_name,
            self.critic_output_gain,
        ).to(self.device)
        self.target_q1.load_state_dict(self.q1.state_dict())
        self.target_q2.load_state_dict(self.q2.state_dict())
        self.target_q1.requires_grad_(False)
        self.target_q2.requires_grad_(False)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self.actor_learning_rate
        )
        self.critic_optimizer = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=self.critic_learning_rate
        )
        self.log_alpha = torch.tensor(
            math.log(self.initial_alpha),
            dtype=torch.float32,
            device=self.device,
            requires_grad=self.is_automatic_entropy_tuning,
        )
        self.alpha_optimizer = (
            torch.optim.Adam([self.log_alpha], lr=self.alpha_learning_rate)
            if self.is_automatic_entropy_tuning
            else None
        )
        self.num_update = 0

    @property
    def alpha(self) -> torch.Tensor:
        """Alpha."""
        return self.log_alpha.exp()

    def get_action(
        self, state: Any, *, delta_action: Any = None, is_deterministic: bool = True
    ) -> float | np.ndarray:
        """Return the policy action, including the Delta correction when enabled."""
        state_tensor, is_single = self._get_state_tensor(state)
        delta_tensor = self._get_delta_tensor(delta_action, num_batch=state_tensor.shape[0])
        with torch.no_grad():
            policy_action, _, _ = self.actor.get_policy_action(
                state_tensor, is_deterministic=is_deterministic
            )
            action = self._compose_final_action(policy_action, delta_tensor)
        return self._get_public_action(action, is_single=is_single)

    def update(self, batch: Mapping[str, Any]) -> dict[str, float]:
        """Update model parameters from the supplied training batch and return optimization diagnostics."""
        state = self._get_batch_tensor(batch, "state")
        action = self._get_batch_tensor(batch, "action")
        reward = self._get_batch_tensor(batch, "reward")
        next_state = self._get_batch_tensor(batch, "next_state")
        terminated = self._get_batch_tensor(batch, "terminated")
        delta = self._get_batch_tensor(batch, "delta_action")
        next_delta = self._get_batch_tensor(batch, "next_delta_action")
        with torch.no_grad():
            next_policy_action, next_log_prob, _ = self.actor.get_policy_action(
                next_state, is_deterministic=False
            )
            next_action = self._compose_final_action(next_policy_action, next_delta)
            target_value = (
                torch.minimum(
                    self.target_q1(next_state, next_action), self.target_q2(next_state, next_action)
                )
                - self.alpha.detach() * next_log_prob
            )
            target_q = reward + self.gamma * (1.0 - terminated) * target_value
        q1_loss = F.mse_loss(self.q1(state, action), target_q)
        q2_loss = F.mse_loss(self.q2(state, action), target_q)
        critic_loss = q1_loss + q2_loss
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        critic_gradient_norm = nn.utils.clip_grad_norm_(
            list(self.q1.parameters()) + list(self.q2.parameters()), self.max_gradient_norm
        )
        self.critic_optimizer.step()
        for network in (self.q1, self.q2):
            network.requires_grad_(False)
        policy_action, log_prob, _ = self.actor.get_policy_action(state, is_deterministic=False)
        final_action = self._compose_final_action(policy_action, delta)
        policy_q = torch.minimum(self.q1(state, final_action), self.q2(state, final_action))
        actor_loss = (self.alpha.detach() * log_prob - policy_q).mean()
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        actor_gradient_norm = nn.utils.clip_grad_norm_(
            self.actor.parameters(), self.max_gradient_norm
        )
        self.actor_optimizer.step()
        for network in (self.q1, self.q2):
            network.requires_grad_(True)
        if self.is_automatic_entropy_tuning:
            assert self.alpha_optimizer is not None
            alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()
            self.alpha_optimizer.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_optimizer.step()
        else:
            alpha_loss = torch.zeros((), device=self.device)
        self._soft_update(self.q1, self.target_q1)
        self._soft_update(self.q2, self.target_q2)
        self.num_update += 1
        return {
            "critic_loss": float(critic_loss.detach().cpu()),
            "q1_loss": float(q1_loss.detach().cpu()),
            "q2_loss": float(q2_loss.detach().cpu()),
            "actor_loss": float(actor_loss.detach().cpu()),
            "actor_gradient_norm": float(actor_gradient_norm.detach().cpu()),
            "critic_gradient_norm": float(critic_gradient_norm.detach().cpu()),
            "alpha_loss": float(alpha_loss.detach().cpu()),
            "alpha": float(self.alpha.detach().cpu()),
            "mean_log_prob": float(log_prob.mean().detach().cpu()),
        }

    def _soft_update(self, source: nn.Module, target: nn.Module) -> None:
        """Soft update."""
        with torch.no_grad():
            for source_parameter, target_parameter in zip(source.parameters(), target.parameters()):
                target_parameter.mul_(1.0 - self.tau)
                target_parameter.add_(self.tau * source_parameter)

    def get_config(self) -> dict[str, Any]:
        """Return the configuration required to reconstruct this object."""
        return {
            "algorithm_name": self.algorithm_name,
            "training_objective": "maximum_entropy_expected_wealth",
            "num_state_feature": self.num_state_feature,
            "actor_hidden_dims": list(self.actor_hidden_dims),
            "critic_hidden_dims": list(self.critic_hidden_dims),
            "activation_name": self.activation_name,
            "actor_output_gain": self.actor_output_gain,
            "critic_output_gain": self.critic_output_gain,
            "actor_learning_rate": self.actor_learning_rate,
            "critic_learning_rate": self.critic_learning_rate,
            "alpha_learning_rate": self.alpha_learning_rate,
            "gamma": self.gamma,
            "tau": self.tau,
            "initial_alpha": self.initial_alpha,
            "target_entropy": self.target_entropy,
            "is_automatic_entropy_tuning": self.is_automatic_entropy_tuning,
            "max_gradient_norm": self.max_gradient_norm,
            "is_delta_residual": self.is_delta_residual,
            "max_delta_residual": self.max_delta_residual,
            "device": str(self.device),
            "seed": self.seed,
        }

    def _get_checkpoint_state(self) -> dict[str, Any]:
        """Return checkpoint state."""
        state = {
            "actor": self.actor.state_dict(),
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
            "target_q1": self.target_q1.state_dict(),
            "target_q2": self.target_q2.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "num_update": self.num_update,
        }
        if self.alpha_optimizer is not None:
            state["alpha_optimizer"] = self.alpha_optimizer.state_dict()
        return state

    def _load_checkpoint_state(
        self, checkpoint: Mapping[str, Any], *, is_load_optimizer: bool
    ) -> None:
        """Load checkpoint state."""
        for name in ("actor", "q1", "q2", "target_q1", "target_q2"):
            getattr(self, name).load_state_dict(checkpoint[name])
        with torch.no_grad():
            self.log_alpha.copy_(checkpoint["log_alpha"].to(self.device))
        if is_load_optimizer:
            self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
            self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
            if self.alpha_optimizer is not None and "alpha_optimizer" in checkpoint:
                self.alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer"])
        self.num_update = int(checkpoint.get("num_update", 0))


__all__ = ["AgentConfigurationError", "BaseHedgingAgent", "SACHedgingAgent", "TD3HedgingAgent"]
