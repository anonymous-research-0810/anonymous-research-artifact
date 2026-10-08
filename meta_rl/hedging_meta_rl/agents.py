"""Agents for the paper option-hedging pipeline."""

from __future__ import annotations
import math
from collections.abc import Mapping, Sequence
from typing import Any
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from rl_agents import (
    DEFAULT_HIDDEN_DIMS,
    DEFAULT_NUM_STATE_FEATURE,
    DEFAULT_OUTPUT_GAIN,
    AgentConfigurationError,
    BaseHedgingAgent,
    _DeterministicActor,
    _QNetwork,
    _SquashedGaussianActor,
    _get_activation,
    _get_hidden_dims,
    _get_nonnegative_float,
    _get_positive_int,
    _get_probability,
)
from .context import ProbabilisticContextEncoder


class _PearlHedgingAgentBase(BaseHedgingAgent):
    """Common latent-conditioned actor-critic and checkpoint interface."""

    is_off_policy = True

    def __init__(
        self,
        *,
        num_state_feature: int,
        latent_dimension: int,
        context_hidden_dims: Sequence[int],
        activation_name: str,
        encoder_learning_rate: float,
        kl_coefficient: float,
        max_gradient_norm: float,
        is_delta_residual: bool,
        max_delta_residual: float,
        device: str | torch.device | None,
        seed: int | None,
    ) -> None:
        """Initialize validated configuration and internal state."""
        super().__init__(
            num_state_feature=num_state_feature,
            is_delta_residual=is_delta_residual,
            max_delta_residual=max_delta_residual,
            device=device,
            seed=seed,
        )
        self.latent_dimension = _get_positive_int(latent_dimension, "latent_dimension")
        self.context_hidden_dims = _get_hidden_dims(context_hidden_dims)
        self.activation_name = str(activation_name).lower()
        _get_activation(self.activation_name)
        self.encoder_learning_rate = _get_nonnegative_float(
            encoder_learning_rate, "encoder_learning_rate"
        )
        if self.encoder_learning_rate <= 0.0:
            raise AgentConfigurationError("encoder_learning_rate must be strictly positive")
        self.kl_coefficient = _get_nonnegative_float(kl_coefficient, "kl_coefficient")
        self.max_gradient_norm = _get_nonnegative_float(max_gradient_norm, "max_gradient_norm")
        if self.max_gradient_norm <= 0.0:
            raise AgentConfigurationError("max_gradient_norm must be strictly positive")
        self.num_context_feature = 2 * self.num_state_feature + 2
        self.augmented_state_dimension = self.num_state_feature + self.latent_dimension
        self.context_encoder = ProbabilisticContextEncoder(
            num_context_feature=self.num_context_feature,
            latent_dimension=self.latent_dimension,
            hidden_dims=self.context_hidden_dims,
            activation_name=self.activation_name,
        ).to(self.device)
        self.encoder_optimizer = torch.optim.Adam(
            self.context_encoder.parameters(), lr=self.encoder_learning_rate
        )

    def _get_latent_tensor(self, latent: Any, *, num_batch: int) -> torch.Tensor:
        """Return latent tensor."""
        if latent is None:
            return torch.zeros(
                (num_batch, self.latent_dimension), dtype=torch.float32, device=self.device
            )
        if torch.is_tensor(latent):
            tensor = latent.to(device=self.device, dtype=torch.float32)
        else:
            tensor = torch.as_tensor(latent, dtype=torch.float32, device=self.device)
        if tensor.ndim == 1:
            tensor = tensor.reshape(1, -1)
        if tensor.shape == (1, self.latent_dimension) and num_batch > 1:
            tensor = tensor.expand(num_batch, -1)
        if tensor.shape != (num_batch, self.latent_dimension):
            raise AgentConfigurationError(
                "latent shape must be (d_z,) or (N,d_z), aligned with the state batch"
            )
        if not torch.isfinite(tensor).all():
            raise AgentConfigurationError("latent contains nonfinite values")
        return tensor

    def _get_augmented_state(self, state: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        """Return augmented state."""
        if state.ndim != 2 or latent.ndim != 2 or state.shape[0] != latent.shape[0]:
            raise AgentConfigurationError(
                "state and latent must be two-dimensional tensors with aligned batches"
            )
        return torch.cat((state, latent), dim=-1)

    def _get_scalar_metrics(self, values: Mapping[str, torch.Tensor | float]) -> dict[str, float]:
        """Return scalar metrics."""
        names = list(values)
        tensors = []
        for value in values.values():
            if torch.is_tensor(value):
                tensors.append(value.detach().reshape(()).to(self.device))
            else:
                tensors.append(
                    torch.as_tensor(float(value), dtype=torch.float32, device=self.device)
                )
        scalar_values = torch.stack(tensors).cpu().tolist()
        return {name: float(value) for name, value in zip(names, scalar_values)}

    def infer_posterior(
        self, context: Any, *, is_deterministic: bool
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Infer the Gaussian task posterior from the supplied context transitions."""
        if torch.is_tensor(context):
            tensor = context.to(device=self.device, dtype=torch.float32)
        else:
            tensor = torch.as_tensor(context, dtype=torch.float32, device=self.device)
        if tensor.ndim != 2 or tensor.shape[1] != self.num_context_feature:
            raise AgentConfigurationError(
                f"context must be (L,{self.num_context_feature}) two-dimensional array"
            )
        mean, variance = self.context_encoder(tensor)
        latent = self.context_encoder.sample(mean, variance, is_deterministic=is_deterministic)
        return (latent.squeeze(0), mean.squeeze(0), variance.squeeze(0))

    def _prepare_meta_batch(
        self, task_batches: Sequence[Mapping[str, Any]], *, is_record_latent_statistics: bool
    ) -> dict[str, Any]:
        """Prepare meta batch."""
        if not task_batches:
            raise AgentConfigurationError("task_batches must not be empty")
        fields = (
            "state",
            "action",
            "reward",
            "next_state",
            "terminated",
            "delta_action",
            "next_delta_action",
        )
        collected: dict[str, list[torch.Tensor]] = {name: [] for name in fields}
        latents: list[torch.Tensor] = []
        means: list[torch.Tensor] = []
        variances: list[torch.Tensor] = []
        latent_records: list[dict[str, Any]] = []
        expected_batch_size: int | None = None
        for task_batch in task_batches:
            context = self._get_batch_tensor(task_batch, "context")
            if context.ndim != 2 or context.shape[1] != self.num_context_feature:
                raise AgentConfigurationError(
                    f"context must be (L,{self.num_context_feature}) tensor"
                )
            mean, variance = self.context_encoder(context)
            latent = self.context_encoder.sample(mean, variance, is_deterministic=False)
            task_tensors = {name: self._get_batch_tensor(task_batch, name) for name in fields}
            batch_size = int(task_tensors["state"].shape[0])
            if expected_batch_size is None:
                expected_batch_size = batch_size
            elif batch_size != expected_batch_size:
                raise AgentConfigurationError(
                    "Task RL minibatches must have equal size within a meta-update"
                )
            if task_tensors["state"].shape != (batch_size, self.num_state_feature):
                raise AgentConfigurationError("The task batch state shape is invalid")
            for name, tensor in task_tensors.items():
                if tensor.shape[0] != batch_size:
                    raise AgentConfigurationError(f"task batch {name} is not aligned with state")
                collected[name].append(tensor)
            latents.append(latent.expand(batch_size, -1))
            means.append(mean)
            variances.append(variance)
            if is_record_latent_statistics:
                kl = self.context_encoder.get_kl_divergence(mean, variance)
                record: dict[str, Any] = {
                    "segment_id": int(task_batch["segment_id"]),
                    "context_episode_id": str(task_batch["context_episode_id"]),
                    "context_source_root": str(task_batch["context_source_root"]),
                    "context_is_simulated": bool(task_batch["context_is_simulated"]),
                    "num_context": int(task_batch["num_context"]),
                    "num_real_batch": int(task_batch["num_real_batch"]),
                    "num_simulated_batch": int(task_batch["num_simulated_batch"]),
                    "kl_divergence": float(kl.detach().cpu().item()),
                }
                for index in range(self.latent_dimension):
                    record[f"posterior_mean_{index}"] = float(mean[0, index].detach().cpu())
                    record[f"posterior_std_{index}"] = float(
                        variance[0, index].sqrt().detach().cpu()
                    )
                    record[f"latent_{index}"] = float(latent[0, index].detach().cpu())
                record["posterior_mean_norm"] = float(mean.norm().detach().cpu())
                record["latent_norm"] = float(latent.norm().detach().cpu())
                latent_records.append(record)
        return {
            **{name: torch.cat(values, dim=0) for name, values in collected.items()},
            "latent": torch.cat(latents, dim=0),
            "mean": torch.cat(means, dim=0),
            "variance": torch.cat(variances, dim=0),
            "latent_records": latent_records,
            "num_task": len(task_batches),
        }

    def _get_shared_config(self) -> dict[str, Any]:
        """Return shared config."""
        return {
            "num_state_feature": self.num_state_feature,
            "num_context_feature": self.num_context_feature,
            "latent_dimension": self.latent_dimension,
            "context_hidden_dims": list(self.context_hidden_dims),
            "context_minimum_variance": self.context_encoder.minimum_variance,
            "activation_name": self.activation_name,
            "encoder_learning_rate": self.encoder_learning_rate,
            "kl_coefficient": self.kl_coefficient,
            "max_gradient_norm": self.max_gradient_norm,
            "is_delta_residual": self.is_delta_residual,
            "max_delta_residual": self.max_delta_residual,
            "device": str(self.device),
            "seed": self.seed,
        }

    def _check_checkpoint_semantics(self, checkpoint: Mapping[str, Any]) -> None:
        """Check checkpoint semantics."""
        saved = checkpoint.get("config", {})
        current = self.get_config()
        for key in (
            "latent_dimension",
            "context_hidden_dims",
            "num_context_feature",
            "context_minimum_variance",
        ):
            if saved.get(key) != current.get(key):
                raise AgentConfigurationError(
                    f"Checkpoint configuration {key} differs from the current PEARL model"
                )


class PearlTD3HedgingAgent(_PearlHedgingAgentBase):
    """Context-conditioned TD3 agent with a probabilistic task encoder and joint critic/encoder updates."""

    algorithm_name = "pearl_td3"

    def __init__(
        self,
        *,
        num_state_feature: int = DEFAULT_NUM_STATE_FEATURE,
        latent_dimension: int = 4,
        context_hidden_dims: Sequence[int] = DEFAULT_HIDDEN_DIMS,
        actor_hidden_dims: Sequence[int] = DEFAULT_HIDDEN_DIMS,
        critic_hidden_dims: Sequence[int] = DEFAULT_HIDDEN_DIMS,
        activation_name: str = "relu",
        actor_output_gain: float = DEFAULT_OUTPUT_GAIN,
        critic_output_gain: float = DEFAULT_OUTPUT_GAIN,
        actor_learning_rate: float = 0.0003,
        critic_learning_rate: float = 0.0003,
        encoder_learning_rate: float = 0.0001,
        kl_coefficient: float = 0.001,
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
            latent_dimension=latent_dimension,
            context_hidden_dims=context_hidden_dims,
            activation_name=activation_name,
            encoder_learning_rate=encoder_learning_rate,
            kl_coefficient=kl_coefficient,
            max_gradient_norm=max_gradient_norm,
            is_delta_residual=is_delta_residual,
            max_delta_residual=max_delta_residual,
            device=device,
            seed=seed,
        )
        self.actor_hidden_dims = _get_hidden_dims(actor_hidden_dims)
        self.critic_hidden_dims = _get_hidden_dims(critic_hidden_dims)
        self.actor_output_gain = _get_nonnegative_float(actor_output_gain, "actor_output_gain")
        self.critic_output_gain = _get_nonnegative_float(critic_output_gain, "critic_output_gain")
        self.actor_learning_rate = _get_nonnegative_float(
            actor_learning_rate, "actor_learning_rate"
        )
        self.critic_learning_rate = _get_nonnegative_float(
            critic_learning_rate, "critic_learning_rate"
        )
        if (
            min(
                self.actor_output_gain,
                self.critic_output_gain,
                self.actor_learning_rate,
                self.critic_learning_rate,
            )
            <= 0.0
        ):
            raise AgentConfigurationError(
                "Network gains and learning rates must be strictly positive"
            )
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
        actor_kwargs = {
            "num_state_feature": self.augmented_state_dimension,
            "hidden_dims": self.actor_hidden_dims,
            "activation_name": self.activation_name,
            "is_delta_residual": self.is_delta_residual,
            "max_delta_residual": self.max_delta_residual,
            "output_gain": self.actor_output_gain,
        }
        critic_kwargs = {
            "num_state_feature": self.augmented_state_dimension,
            "hidden_dims": self.critic_hidden_dims,
            "activation_name": self.activation_name,
            "output_gain": self.critic_output_gain,
        }
        self.actor = _DeterministicActor(**actor_kwargs).to(self.device)
        self.target_actor = _DeterministicActor(**actor_kwargs).to(self.device)
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

    def _clip_policy_action(self, action: torch.Tensor) -> torch.Tensor:
        """Clip policy action."""
        if self.is_delta_residual:
            return torch.clamp(action, -self.max_delta_residual, self.max_delta_residual)
        return torch.clamp(action, 0.0, 1.0)

    def get_action(
        self,
        state: Any,
        *,
        latent: Any = None,
        delta_action: Any = None,
        is_deterministic: bool = True,
    ) -> float | np.ndarray:
        """Return the policy action, including the Delta correction when enabled."""
        state_tensor, is_single = self._get_state_tensor(state)
        latent_tensor = self._get_latent_tensor(latent, num_batch=state_tensor.shape[0])
        delta_tensor = self._get_delta_tensor(delta_action, num_batch=state_tensor.shape[0])
        with torch.no_grad():
            policy_action = self.actor(self._get_augmented_state(state_tensor, latent_tensor))
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

    def update(
        self,
        task_batches: Sequence[Mapping[str, Any]],
        *,
        kl_coefficient: float | None = None,
        is_record_latent_statistics: bool = True,
    ) -> dict[str, Any]:
        """Update model parameters from the supplied training batch and return optimization diagnostics."""
        current_kl_coefficient = (
            self.kl_coefficient
            if kl_coefficient is None
            else _get_nonnegative_float(kl_coefficient, "kl_coefficient")
        )
        batch = self._prepare_meta_batch(
            task_batches, is_record_latent_statistics=is_record_latent_statistics
        )
        latent = batch["latent"]
        state = self._get_augmented_state(batch["state"], latent)
        next_state_target = self._get_augmented_state(batch["next_state"], latent.detach())
        with torch.no_grad():
            next_policy_action = self.target_actor(next_state_target)
            target_noise = torch.randn_like(next_policy_action) * (
                self.target_policy_noise_std * self.policy_action_scale
            )
            limit = self.target_noise_clip * self.policy_action_scale
            next_policy_action = self._clip_policy_action(
                next_policy_action + torch.clamp(target_noise, -limit, limit)
            )
            next_action = self._compose_final_action(next_policy_action, batch["next_delta_action"])
            target_q = torch.minimum(
                self.target_q1(next_state_target, next_action),
                self.target_q2(next_state_target, next_action),
            )
            target_q = batch["reward"] + self.gamma * (1.0 - batch["terminated"]) * target_q
        q1_loss = F.mse_loss(self.q1(state, batch["action"]), target_q)
        q2_loss = F.mse_loss(self.q2(state, batch["action"]), target_q)
        critic_loss = q1_loss + q2_loss
        kl_loss = self.context_encoder.get_kl_divergence(batch["mean"], batch["variance"]).mean()
        encoder_critic_loss = critic_loss + current_kl_coefficient * kl_loss
        self.critic_optimizer.zero_grad(set_to_none=True)
        self.encoder_optimizer.zero_grad(set_to_none=True)
        encoder_critic_loss.backward()
        critic_gradient_norm = nn.utils.clip_grad_norm_(
            list(self.q1.parameters()) + list(self.q2.parameters()), self.max_gradient_norm
        )
        encoder_gradient_norm = nn.utils.clip_grad_norm_(
            self.context_encoder.parameters(), self.max_gradient_norm
        )
        self.critic_optimizer.step()
        self.encoder_optimizer.step()
        is_actor_updated = (self.num_update + 1) % self.num_policy_delay == 0
        actor_loss_metric = torch.zeros((), device=self.device)
        actor_gradient_norm_metric = torch.zeros((), device=self.device)
        if is_actor_updated:
            for network in (self.q1, self.q2):
                network.requires_grad_(False)
            actor_state = self._get_augmented_state(batch["state"], latent.detach())
            policy_action = self.actor(actor_state)
            final_action = self._compose_final_action(policy_action, batch["delta_action"])
            actor_loss = -self.q1(actor_state, final_action).mean()
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
            actor_loss_metric = actor_loss
            actor_gradient_norm_metric = actor_gradient_norm
        self.num_update += 1
        scalar_metrics = self._get_scalar_metrics(
            {
                "critic_loss": critic_loss,
                "q1_loss": q1_loss,
                "q2_loss": q2_loss,
                "kl_loss": kl_loss,
                "encoder_loss": encoder_critic_loss,
                "actor_loss": actor_loss_metric,
                "critic_gradient_norm": critic_gradient_norm,
                "encoder_gradient_norm": encoder_gradient_norm,
                "actor_gradient_norm": actor_gradient_norm_metric,
                "mean_target_q": target_q.mean(),
                "kl_coefficient": current_kl_coefficient,
            }
        )
        for record in batch["latent_records"]:
            record.update(
                {
                    "algorithm": self.algorithm_name,
                    "meta_update": self.num_update,
                    "critic_loss": scalar_metrics["critic_loss"],
                    "actor_loss": scalar_metrics["actor_loss"],
                    "encoder_loss": scalar_metrics["encoder_loss"],
                    "kl_coefficient": scalar_metrics["kl_coefficient"],
                }
            )
        return {
            **scalar_metrics,
            "is_actor_updated": float(is_actor_updated),
            "num_task": int(batch["num_task"]),
            "latent_records": batch["latent_records"],
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
            "training_objective": "pearl_td3_expected_cumulative_reward",
            **self._get_shared_config(),
            "actor_hidden_dims": list(self.actor_hidden_dims),
            "critic_hidden_dims": list(self.critic_hidden_dims),
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
        }

    def _get_checkpoint_state(self) -> dict[str, Any]:
        """Return checkpoint state."""
        return {
            "context_encoder": self.context_encoder.state_dict(),
            "actor": self.actor.state_dict(),
            "target_actor": self.target_actor.state_dict(),
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
            "target_q1": self.target_q1.state_dict(),
            "target_q2": self.target_q2.state_dict(),
            "encoder_optimizer": self.encoder_optimizer.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "num_update": self.num_update,
        }

    def _load_checkpoint_state(
        self, checkpoint: Mapping[str, Any], *, is_load_optimizer: bool
    ) -> None:
        """Load checkpoint state."""
        self._check_checkpoint_semantics(checkpoint)
        for name in (
            "context_encoder",
            "actor",
            "target_actor",
            "q1",
            "q2",
            "target_q1",
            "target_q2",
        ):
            getattr(self, name).load_state_dict(checkpoint[name])
        if is_load_optimizer:
            self.encoder_optimizer.load_state_dict(checkpoint["encoder_optimizer"])
            self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
            self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        self.num_update = int(checkpoint.get("num_update", 0))


class PearlSACHedgingAgent(_PearlHedgingAgentBase):
    """Context-conditioned SAC agent with a probabilistic task encoder and joint critic/encoder updates."""

    algorithm_name = "pearl_sac"

    def __init__(
        self,
        *,
        num_state_feature: int = DEFAULT_NUM_STATE_FEATURE,
        latent_dimension: int = 4,
        context_hidden_dims: Sequence[int] = DEFAULT_HIDDEN_DIMS,
        actor_hidden_dims: Sequence[int] = DEFAULT_HIDDEN_DIMS,
        critic_hidden_dims: Sequence[int] = DEFAULT_HIDDEN_DIMS,
        activation_name: str = "relu",
        actor_output_gain: float = DEFAULT_OUTPUT_GAIN,
        critic_output_gain: float = DEFAULT_OUTPUT_GAIN,
        actor_learning_rate: float = 0.0003,
        critic_learning_rate: float = 0.0003,
        encoder_learning_rate: float = 0.0001,
        alpha_learning_rate: float = 0.0003,
        kl_coefficient: float = 0.001,
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
            latent_dimension=latent_dimension,
            context_hidden_dims=context_hidden_dims,
            activation_name=activation_name,
            encoder_learning_rate=encoder_learning_rate,
            kl_coefficient=kl_coefficient,
            max_gradient_norm=max_gradient_norm,
            is_delta_residual=is_delta_residual,
            max_delta_residual=max_delta_residual,
            device=device,
            seed=seed,
        )
        self.actor_hidden_dims = _get_hidden_dims(actor_hidden_dims)
        self.critic_hidden_dims = _get_hidden_dims(critic_hidden_dims)
        self.actor_output_gain = _get_nonnegative_float(actor_output_gain, "actor_output_gain")
        self.critic_output_gain = _get_nonnegative_float(critic_output_gain, "critic_output_gain")
        self.actor_learning_rate = _get_nonnegative_float(
            actor_learning_rate, "actor_learning_rate"
        )
        self.critic_learning_rate = _get_nonnegative_float(
            critic_learning_rate, "critic_learning_rate"
        )
        self.alpha_learning_rate = _get_nonnegative_float(
            alpha_learning_rate, "alpha_learning_rate"
        )
        if (
            min(
                self.actor_output_gain,
                self.critic_output_gain,
                self.actor_learning_rate,
                self.critic_learning_rate,
                self.alpha_learning_rate,
            )
            <= 0.0
        ):
            raise AgentConfigurationError(
                "Network gains and learning rates must be strictly positive"
            )
        self.gamma = _get_probability(gamma, "gamma")
        self.tau = _get_probability(tau, "tau", is_zero_allowed=False)
        self.initial_alpha = _get_nonnegative_float(initial_alpha, "initial_alpha")
        if self.initial_alpha <= 0.0:
            raise AgentConfigurationError("initial_alpha must be strictly positive")
        if not isinstance(is_automatic_entropy_tuning, (bool, np.bool_)):
            raise AgentConfigurationError("is_automatic_entropy_tuning must be a boolean")
        self.is_automatic_entropy_tuning = bool(is_automatic_entropy_tuning)
        self.actor = _SquashedGaussianActor(
            self.augmented_state_dimension,
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
        critic_kwargs = {
            "num_state_feature": self.augmented_state_dimension,
            "hidden_dims": self.critic_hidden_dims,
            "activation_name": self.activation_name,
            "output_gain": self.critic_output_gain,
        }
        self.q1 = _QNetwork(**critic_kwargs).to(self.device)
        self.q2 = _QNetwork(**critic_kwargs).to(self.device)
        self.target_q1 = _QNetwork(**critic_kwargs).to(self.device)
        self.target_q2 = _QNetwork(**critic_kwargs).to(self.device)
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
        self,
        state: Any,
        *,
        latent: Any = None,
        delta_action: Any = None,
        is_deterministic: bool = True,
    ) -> float | np.ndarray:
        """Return the policy action, including the Delta correction when enabled."""
        state_tensor, is_single = self._get_state_tensor(state)
        latent_tensor = self._get_latent_tensor(latent, num_batch=state_tensor.shape[0])
        delta_tensor = self._get_delta_tensor(delta_action, num_batch=state_tensor.shape[0])
        with torch.no_grad():
            policy_action, _, _ = self.actor.get_policy_action(
                self._get_augmented_state(state_tensor, latent_tensor),
                is_deterministic=is_deterministic,
            )
            action = self._compose_final_action(policy_action, delta_tensor)
        return self._get_public_action(action, is_single=is_single)

    def update(
        self,
        task_batches: Sequence[Mapping[str, Any]],
        *,
        kl_coefficient: float | None = None,
        is_record_latent_statistics: bool = True,
    ) -> dict[str, Any]:
        """Update model parameters from the supplied training batch and return optimization diagnostics."""
        current_kl_coefficient = (
            self.kl_coefficient
            if kl_coefficient is None
            else _get_nonnegative_float(kl_coefficient, "kl_coefficient")
        )
        batch = self._prepare_meta_batch(
            task_batches, is_record_latent_statistics=is_record_latent_statistics
        )
        latent = batch["latent"]
        state = self._get_augmented_state(batch["state"], latent)
        next_state_target = self._get_augmented_state(batch["next_state"], latent.detach())
        with torch.no_grad():
            next_policy_action, next_log_prob, _ = self.actor.get_policy_action(
                next_state_target, is_deterministic=False
            )
            next_action = self._compose_final_action(next_policy_action, batch["next_delta_action"])
            target_value = (
                torch.minimum(
                    self.target_q1(next_state_target, next_action),
                    self.target_q2(next_state_target, next_action),
                )
                - self.alpha.detach() * next_log_prob
            )
            target_q = batch["reward"] + self.gamma * (1.0 - batch["terminated"]) * target_value
        q1_loss = F.mse_loss(self.q1(state, batch["action"]), target_q)
        q2_loss = F.mse_loss(self.q2(state, batch["action"]), target_q)
        critic_loss = q1_loss + q2_loss
        kl_loss = self.context_encoder.get_kl_divergence(batch["mean"], batch["variance"]).mean()
        encoder_critic_loss = critic_loss + current_kl_coefficient * kl_loss
        self.critic_optimizer.zero_grad(set_to_none=True)
        self.encoder_optimizer.zero_grad(set_to_none=True)
        encoder_critic_loss.backward()
        critic_gradient_norm = nn.utils.clip_grad_norm_(
            list(self.q1.parameters()) + list(self.q2.parameters()), self.max_gradient_norm
        )
        encoder_gradient_norm = nn.utils.clip_grad_norm_(
            self.context_encoder.parameters(), self.max_gradient_norm
        )
        self.critic_optimizer.step()
        self.encoder_optimizer.step()
        for network in (self.q1, self.q2):
            network.requires_grad_(False)
        actor_state = self._get_augmented_state(batch["state"], latent.detach())
        policy_action, log_prob, _ = self.actor.get_policy_action(
            actor_state, is_deterministic=False
        )
        final_action = self._compose_final_action(policy_action, batch["delta_action"])
        policy_q = torch.minimum(
            self.q1(actor_state, final_action), self.q2(actor_state, final_action)
        )
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
        scalar_metrics = self._get_scalar_metrics(
            {
                "critic_loss": critic_loss,
                "q1_loss": q1_loss,
                "q2_loss": q2_loss,
                "kl_loss": kl_loss,
                "encoder_loss": encoder_critic_loss,
                "actor_loss": actor_loss,
                "alpha_loss": alpha_loss,
                "alpha": self.alpha,
                "mean_log_prob": log_prob.mean(),
                "critic_gradient_norm": critic_gradient_norm,
                "encoder_gradient_norm": encoder_gradient_norm,
                "actor_gradient_norm": actor_gradient_norm,
                "mean_target_q": target_q.mean(),
                "kl_coefficient": current_kl_coefficient,
            }
        )
        for record in batch["latent_records"]:
            record.update(
                {
                    "algorithm": self.algorithm_name,
                    "meta_update": self.num_update,
                    "critic_loss": scalar_metrics["critic_loss"],
                    "actor_loss": scalar_metrics["actor_loss"],
                    "encoder_loss": scalar_metrics["encoder_loss"],
                    "kl_coefficient": scalar_metrics["kl_coefficient"],
                    "alpha": scalar_metrics["alpha"],
                }
            )
        return {
            **scalar_metrics,
            "num_task": int(batch["num_task"]),
            "latent_records": batch["latent_records"],
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
            "training_objective": "pearl_sac_maximum_entropy_reward",
            **self._get_shared_config(),
            "actor_hidden_dims": list(self.actor_hidden_dims),
            "critic_hidden_dims": list(self.critic_hidden_dims),
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
        }

    def _get_checkpoint_state(self) -> dict[str, Any]:
        """Return checkpoint state."""
        state: dict[str, Any] = {
            "context_encoder": self.context_encoder.state_dict(),
            "actor": self.actor.state_dict(),
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
            "target_q1": self.target_q1.state_dict(),
            "target_q2": self.target_q2.state_dict(),
            "encoder_optimizer": self.encoder_optimizer.state_dict(),
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
        self._check_checkpoint_semantics(checkpoint)
        for name in ("context_encoder", "actor", "q1", "q2", "target_q1", "target_q2"):
            getattr(self, name).load_state_dict(checkpoint[name])
        with torch.no_grad():
            self.log_alpha.copy_(checkpoint["log_alpha"].to(self.device))
        if is_load_optimizer:
            self.encoder_optimizer.load_state_dict(checkpoint["encoder_optimizer"])
            self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
            self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
            if self.alpha_optimizer is not None and "alpha_optimizer" in checkpoint:
                self.alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer"])
        self.num_update = int(checkpoint.get("num_update", 0))
