"""Rl utils for the paper option-hedging pipeline."""

from __future__ import annotations
import math
from copy import deepcopy
from pathlib import Path
from time import perf_counter
from typing import Any, Callable
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from rl_agents import BaseHedgingAgent, SACHedgingAgent, TD3HedgingAgent


class ReplayBuffer:
    """Bounded off-policy replay storing transitions and current/next BSM anchors."""

    def __init__(
        self, *, num_capacity: int = 100000, num_state_feature: int = 7, seed: int | None = None
    ) -> None:
        """Initialize validated configuration and internal state."""
        self.num_capacity = self._get_positive_int(num_capacity, "num_capacity")
        self.num_state_feature = self._get_positive_int(num_state_feature, "num_state_feature")
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self._state = np.empty((self.num_capacity, self.num_state_feature), dtype=np.float32)
        self._action = np.empty((self.num_capacity, 1), dtype=np.float32)
        self._reward = np.empty((self.num_capacity, 1), dtype=np.float32)
        self._next_state = np.empty_like(self._state)
        self._terminated = np.empty((self.num_capacity, 1), dtype=np.float32)
        self._delta_action = np.empty((self.num_capacity, 1), dtype=np.float32)
        self._next_delta_action = np.empty((self.num_capacity, 1), dtype=np.float32)
        self._num_position = 0
        self._num_size = 0

    @staticmethod
    def _get_positive_int(value: Any, name: str) -> int:
        """Return positive int."""
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be a strictly positive integer")
        result = int(value)
        if result <= 0:
            raise ValueError(f"{name} must be a strictly positive integer")
        return result

    @property
    def num_size(self) -> int:
        """Return the number of size."""
        return self._num_size

    @property
    def is_full(self) -> bool:
        """Return whether full."""
        return self._num_size == self.num_capacity

    def __len__(self) -> int:
        """Len."""
        return self.num_size

    def _get_state_array(self, value: Any, name: str) -> np.ndarray:
        """Return state array."""
        array = np.asarray(value, dtype=np.float32)
        if array.shape != (self.num_state_feature,):
            raise ValueError(
                f"{name} shape must be ({self.num_state_feature},); received {array.shape}"
            )
        if not np.isfinite(array).all():
            raise ValueError(f"{name} contains nonfinite values")
        return array

    @staticmethod
    def _get_scalar(value: Any, name: str) -> float:
        """Return scalar."""
        array = np.asarray(value, dtype=np.float32)
        if array.size != 1:
            raise ValueError(f"{name} must be a scalar or a single-element array")
        result = float(array.reshape(-1)[0])
        if not np.isfinite(result):
            raise ValueError(f"{name} must be a finite number")
        return result

    def add(
        self,
        state: Any,
        action: Any,
        reward: Any,
        next_state: Any,
        terminated: Any,
        *,
        delta_action: Any = 0.0,
        next_delta_action: Any = 0.0,
    ) -> None:
        """Add."""
        state_array = self._get_state_array(state, "state")
        next_state_array = self._get_state_array(next_state, "next_state")
        action_scalar = self._get_scalar(action, "action")
        reward_scalar = self._get_scalar(reward, "reward")
        delta_scalar = self._get_scalar(delta_action, "delta_action")
        next_delta_scalar = self._get_scalar(next_delta_action, "next_delta_action")
        if not 0.0 <= action_scalar <= 1.0:
            raise ValueError("action must be in [0,1]")
        if not 0.0 <= delta_scalar <= 1.0:
            raise ValueError("delta_action must be in [0,1]")
        if not 0.0 <= next_delta_scalar <= 1.0:
            raise ValueError("next_delta_action must be in [0,1]")
        terminated_array = np.asarray(terminated)
        if terminated_array.size != 1:
            raise ValueError("terminated must be a boolean or 0/1")
        terminated_value = terminated_array.reshape(-1)[0]
        try:
            terminated_number = float(terminated_value)
        except (TypeError, ValueError) as exc:
            raise ValueError("terminated must be a boolean or 0/1") from exc
        if not np.isfinite(terminated_number) or terminated_number not in (0.0, 1.0):
            raise ValueError("terminated must be a boolean or 0/1")
        terminated_scalar = terminated_number
        num_index = self._num_position
        self._state[num_index] = state_array
        self._action[num_index, 0] = action_scalar
        self._reward[num_index, 0] = reward_scalar
        self._next_state[num_index] = next_state_array
        self._terminated[num_index, 0] = terminated_scalar
        self._delta_action[num_index, 0] = delta_scalar
        self._next_delta_action[num_index, 0] = next_delta_scalar
        self._num_position = (self._num_position + 1) % self.num_capacity
        self._num_size = min(self._num_size + 1, self.num_capacity)

    def sample(
        self, num_batch: int = 128, *, device: str | torch.device | None = None
    ) -> dict[str, torch.Tensor]:
        """Sample."""
        num_batch = self._get_positive_int(num_batch, "num_batch")
        if self._num_size == 0:
            raise ValueError("Cannot sample from an empty ReplayBuffer")
        indices = self._rng.integers(0, self._num_size, size=num_batch)
        target_device = torch.device(device if device is not None else "cpu")
        return {
            "state": torch.as_tensor(self._state[indices], device=target_device),
            "action": torch.as_tensor(self._action[indices], device=target_device),
            "reward": torch.as_tensor(self._reward[indices], device=target_device),
            "next_state": torch.as_tensor(self._next_state[indices], device=target_device),
            "terminated": torch.as_tensor(self._terminated[indices], device=target_device),
            "delta_action": torch.as_tensor(self._delta_action[indices], device=target_device),
            "next_delta_action": torch.as_tensor(
                self._next_delta_action[indices], device=target_device
            ),
        }

    def get_config(self) -> dict[str, Any]:
        """Return the configuration required to reconstruct this object."""
        return {
            "num_capacity": self.num_capacity,
            "num_state_feature": self.num_state_feature,
            "seed": self.seed,
            "num_size": self.num_size,
            "is_full": self.is_full,
        }


def _get_nonnegative_int(value: Any, name: str) -> int:
    """Return nonnegative int."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a nonnegative integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return result


def _get_positive_int(value: Any, name: str) -> int:
    """Return positive int."""
    result = _get_nonnegative_int(value, name)
    if result == 0:
        raise ValueError(f"{name} must be a strictly positive integer")
    return result


def _get_current_delta_action(agent: BaseHedgingAgent, env: Any) -> float | np.ndarray:
    """Return current delta action."""
    if agent.is_require_delta_action:
        return env.get_bsm_call_delta()
    return 0.0


def _validate_train_inputs(agent: BaseHedgingAgent, train_env: Any) -> None:
    """Validate train inputs."""
    if getattr(train_env, "label", None) != "train":
        raise ValueError("train_hedging_agent requires an environment with label='train'")
    if bool(getattr(train_env, "is_batch_mode", True)):
        raise ValueError("Training requires single-episode mode")
    if int(getattr(train_env, "num_state_feature", -1)) != agent.num_state_feature:
        raise ValueError("Agent and environment state dimensions differ")
    if int(getattr(train_env, "num_completed_episode", 0)) != 0:
        raise ValueError("The training environment has already been used; create a new instance")


def _get_evaluation_env_from_template(env: Any) -> Any:
    """Return evaluation env from template."""
    return type(env)(
        env.dataset,
        reward_formulation=env.reward_formulation,
        hedge_cost_rate=env.hedge_cost_rate,
        contract_multiplier=env.contract_multiplier,
        reconciliation_tolerance=env.reconciliation_tolerance,
        risk_aversion_lambda=env.risk_aversion_lambda,
        reward_risk_aversion_xi=env.reward_risk_aversion_xi,
        training_reward_scale=env.training_reward_scale,
        state_normalizer=env.state_normalizer,
        is_batch_mode=env.is_batch_mode,
        num_parallel_episode=env.num_parallel_episode,
        is_repeated=False,
        is_clip_action=env.is_clip_action,
        is_record_trajectory=env.is_record_trajectory,
        observation_dtype=env.observation_dtype,
    )


def _get_valid_env_factory(
    *, valid_env: Any | None, get_valid_env: Callable[[], Any] | None
) -> Callable[[], Any] | None:
    """Return valid env factory."""
    if valid_env is not None and get_valid_env is not None:
        raise ValueError("Provide only one of valid_env and get_valid_env")
    if get_valid_env is not None:
        if not callable(get_valid_env):
            raise TypeError("get_valid_env must be callable without arguments")
        return get_valid_env
    if valid_env is None:
        return None
    if getattr(valid_env, "label", None) != "valid":
        raise ValueError("valid_env must have label='valid'")
    return lambda: _get_evaluation_env_from_template(valid_env)


def train_hedging_agent(
    agent: BaseHedgingAgent,
    train_env: Any,
    *,
    valid_env: Any | None = None,
    get_valid_env: Callable[[], Any] | None = None,
    replay_buffer: ReplayBuffer | None = None,
    num_replay_capacity: int = 100000,
    num_batch: int = 128,
    warmup_fraction: float = 0.1,
    num_update_per_step: int = 1,
    num_valid_episodes: int = 100,
    checkpoint_path: str | Path | None = None,
    checkpoint_dir: str | Path | None = None,
    seed: int | None = 2026,
    is_show_progress: bool = True,
    is_print_summary: bool = True,
) -> dict[str, Any]:
    """Consume the configured training schedule, update the off-policy agent, and restore the validation-best checkpoint before returning."""
    if not isinstance(agent, (SACHedgingAgent, TD3HedgingAgent)):
        raise TypeError("agent must be a SAC or TD3 hedging agent")
    _validate_train_inputs(agent, train_env)
    num_batch = _get_positive_int(num_batch, "num_batch")
    if (
        isinstance(warmup_fraction, (bool, np.bool_))
        or not isinstance(warmup_fraction, (int, float, np.integer, np.floating))
        or (not np.isfinite(float(warmup_fraction)))
        or (not 0.0 <= float(warmup_fraction) < 1.0)
    ):
        raise ValueError("warmup_fraction must be a finite number in [0,1)")
    warmup_fraction = float(warmup_fraction)
    num_update_per_step = _get_positive_int(num_update_per_step, "num_update_per_step")
    num_valid_episodes = _get_positive_int(num_valid_episodes, "num_valid_episodes")
    for name, value in (
        ("is_show_progress", is_show_progress),
        ("is_print_summary", is_print_summary),
    ):
        if not isinstance(value, (bool, np.bool_)):
            raise ValueError(f"{name} must be a boolean")
    is_show_progress = bool(is_show_progress)
    is_print_summary = bool(is_print_summary)
    checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else None
    checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None
    if checkpoint_dir is not None:
        if checkpoint_dir.exists() and (not checkpoint_dir.is_dir()):
            raise ValueError(f"checkpoint_dir exists but is not a directory: {checkpoint_dir}")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    valid_env_factory = _get_valid_env_factory(valid_env=valid_env, get_valid_env=get_valid_env)
    replay_buffer = replay_buffer or ReplayBuffer(
        num_capacity=num_replay_capacity, num_state_feature=agent.num_state_feature, seed=seed
    )
    if replay_buffer.num_state_feature != agent.num_state_feature:
        raise ValueError("ReplayBuffer and agent state dimensions differ")
    num_total_step = 0
    num_episode = 0
    num_gradient_update = 0
    update_records: list[dict[str, Any]] = []
    validation_records: list[dict[str, Any]] = []
    latest_actor_loss: float | None = None
    latest_critic_loss: float | None = None
    last_validation: dict[str, Any] | None = None
    best_validation: dict[str, Any] | None = None
    best_checkpoint_state: dict[str, Any] | None = None
    best_checkpoint_path: Path | None = None
    selected_checkpoint_path: Path | None = None
    num_best_validation_episode: int | None = None
    best_j_lambda = math.inf
    last_valid_j_lambda: float | None = None
    num_unique_episode = int(train_env.dataset.num_episode)
    num_scheduled_episode = int(train_env.num_scheduled_episode)
    num_interval = int(train_env.num_interval)
    num_expected_training_step = num_scheduled_episode * num_interval
    num_warmup_step = (
        int(math.ceil(warmup_fraction * num_expected_training_step)) if agent.is_off_policy else 0
    )
    if is_print_summary:
        print(
            f"[Training started] algorithm={agent.algorithm_name.upper()}, device={agent.device}, reward={train_env.reward_formulation}"
        )
        print(
            f"[Training data] episode samples={num_unique_episode}, rollouts={num_scheduled_episode}, is_repeated={train_env.is_repeated}, intervals per episode={num_interval}, total training steps={num_expected_training_step}, warm-up steps={num_warmup_step}",
            flush=True,
        )
    num_start_time = perf_counter()
    progress_bar = tqdm(
        total=num_scheduled_episode,
        desc=f"Training {agent.algorithm_name.upper()}",
        unit="episode",
        dynamic_ncols=True,
        disable=not is_show_progress,
    )
    try:
        while True:
            try:
                state, reset_info = train_env.reset()
            except StopIteration:
                break
            while True:
                delta_action = _get_current_delta_action(agent, train_env)
                if num_total_step < num_warmup_step:
                    action = agent.get_random_action(
                        delta_action if agent.is_delta_residual else None
                    )
                else:
                    action = agent.get_action(
                        state,
                        delta_action=delta_action if agent.is_require_delta_action else None,
                        is_deterministic=False,
                    )
                next_state, reward, terminated, truncated, _ = train_env.step(action)
                is_done = bool(terminated or truncated)
                if truncated:
                    raise RuntimeError(
                        "Finite-horizon hedging must not produce truncated transitions"
                    )
                next_delta_action = 0.0 if is_done else _get_current_delta_action(agent, train_env)
                assert replay_buffer is not None
                replay_buffer.add(
                    state,
                    action,
                    reward,
                    next_state,
                    is_done,
                    delta_action=delta_action if agent.is_delta_residual else 0.0,
                    next_delta_action=next_delta_action if agent.is_delta_residual else 0.0,
                )
                num_total_step += 1
                state = next_state
                if num_total_step >= num_warmup_step:
                    assert replay_buffer is not None
                    if replay_buffer.num_size >= num_batch:
                        for _ in range(num_update_per_step):
                            metrics = agent.update(
                                replay_buffer.sample(num_batch, device=agent.device)
                            )
                            if bool(metrics.get("is_actor_updated", 1.0)):
                                latest_actor_loss = float(metrics["actor_loss"])
                            latest_critic_loss = float(metrics["critic_loss"])
                            num_gradient_update += 1
                            update_records.append(
                                {
                                    "num_gradient_update": num_gradient_update,
                                    "num_total_step": num_total_step,
                                    "num_episode": num_episode + 1,
                                    **metrics,
                                }
                            )
                if is_done:
                    break
            num_episode += 1
            is_warmup_complete = not agent.is_off_policy or num_total_step >= num_warmup_step
            is_validation_due = (
                valid_env_factory is not None
                and is_warmup_complete
                and (num_episode % num_valid_episodes == 0 or train_env.is_schedule_exhausted)
            )
            if is_validation_due:
                if is_show_progress:
                    progress_bar.write(
                        f"[Validation] Completed training episodes {num_episode}/{num_scheduled_episode}"
                    )
                validation_env = valid_env_factory()
                if getattr(validation_env, "label", None) != "valid":
                    raise ValueError(
                        "get_valid_env must return a label='valid' environment on every call"
                    )
                if int(getattr(validation_env, "num_completed_episode", -1)) != 0:
                    raise ValueError("get_valid_env returned an exhausted validation environment")
                last_validation = evaluate_hedging_agent(
                    agent, validation_env, is_print_summary=is_print_summary
                )
                validation_metrics = last_validation["metrics"]
                last_valid_j_lambda = float(validation_metrics["j_lambda"])
                if not np.isfinite(last_valid_j_lambda):
                    raise RuntimeError("validation J_lambda must be a finite number")
                periodic_checkpoint_path: Path | None = None
                if checkpoint_dir is not None:
                    periodic_checkpoint_path = (
                        checkpoint_dir / f"agent_episode_{num_episode:08d}.pt"
                    )
                    agent.save(periodic_checkpoint_path)
                is_best = last_valid_j_lambda < best_j_lambda
                if is_best:
                    for record in validation_records:
                        record["is_best"] = False
                    best_j_lambda = last_valid_j_lambda
                    best_validation = last_validation
                    best_checkpoint_state = deepcopy(agent._get_checkpoint_state())
                    num_best_validation_episode = num_episode
                    if checkpoint_dir is not None:
                        best_checkpoint_path = checkpoint_dir / "best_agent.pt"
                        agent.save(best_checkpoint_path)
                validation_records.append(
                    {
                        "num_train_episode": num_episode,
                        "num_train_step": num_total_step,
                        "num_gradient_update": num_gradient_update,
                        "is_best": is_best,
                        "checkpoint_path": (
                            str(periodic_checkpoint_path)
                            if periodic_checkpoint_path is not None
                            else None
                        ),
                        **validation_metrics,
                    }
                )
            progress_fields: dict[str, Any] = {
                "ep": f"{num_episode}/{num_scheduled_episode}",
                "upd": num_gradient_update,
            }
            if agent.is_off_policy and num_total_step < num_warmup_step:
                progress_fields["warmup"] = f"{num_total_step}/{num_warmup_step}"
            if latest_actor_loss is not None:
                progress_fields["actor"] = f"{latest_actor_loss:.3g}"
            if latest_critic_loss is not None:
                progress_fields["critic"] = f"{latest_critic_loss:.3g}"
            if last_valid_j_lambda is not None:
                progress_fields[f"valid_J({train_env.get_config().get('currency', 'USD')})"] = (
                    f"{last_valid_j_lambda:.4g}"
                )
            progress_bar.set_postfix(progress_fields, refresh=False)
            progress_bar.update(1)
    finally:
        progress_bar.close()
    if num_episode == 0:
        raise RuntimeError("The training schedule is empty or exhausted")
    num_elapsed_second = perf_counter() - num_start_time
    if best_validation is not None:
        if best_checkpoint_path is not None:
            agent.load(best_checkpoint_path, is_load_optimizer=True)
        else:
            assert best_checkpoint_state is not None
            agent._load_checkpoint_state(best_checkpoint_state, is_load_optimizer=True)
    if checkpoint_path is not None:
        agent.save(checkpoint_path)
        selected_checkpoint_path = checkpoint_path
    elif best_checkpoint_path is not None:
        selected_checkpoint_path = best_checkpoint_path
    elif checkpoint_dir is not None:
        selected_checkpoint_path = checkpoint_dir / "final_agent.pt"
        agent.save(selected_checkpoint_path)
    training_results = train_env.get_completed_episode_results()
    training_metrics = train_env.get_metrics()
    if is_print_summary:
        print(
            f"[Training completed] episode={num_episode}, step={num_total_step}, gradient_update={num_gradient_update}, elapsed time={num_elapsed_second:.2f}s",
            flush=True,
        )
    return {
        "agent": agent,
        "num_episode": num_episode,
        "num_total_step": num_total_step,
        "num_gradient_update": num_gradient_update,
        "episode_results": training_results,
        "metrics": training_metrics,
        "update_history": pd.DataFrame(update_records),
        "validation_history": pd.DataFrame(validation_records),
        "trajectory": train_env.get_trajectory(),
        "validation": best_validation,
        "best_validation": best_validation,
        "last_validation": last_validation,
        "num_best_validation_episode": num_best_validation_episode,
        "best_checkpoint_path": (
            str(best_checkpoint_path) if best_checkpoint_path is not None else None
        ),
        "selected_checkpoint_path": (
            str(selected_checkpoint_path) if selected_checkpoint_path is not None else None
        ),
        "replay_buffer": replay_buffer,
        "config": {
            "agent": agent.get_config(),
            "train_env": train_env.get_config(),
            "num_replay_capacity": num_replay_capacity,
            "num_batch": num_batch,
            "warmup_fraction": warmup_fraction,
            "num_warmup_step": num_warmup_step,
            "num_update_per_step": num_update_per_step,
            "num_valid_episodes": num_valid_episodes,
            "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
            "checkpoint_dir": str(checkpoint_dir) if checkpoint_dir is not None else None,
            "seed": seed,
            "is_show_progress": is_show_progress,
            "is_print_summary": is_print_summary,
        },
    }


def evaluate_hedging_agent(
    agent: BaseHedgingAgent,
    env: Any,
    *,
    risk_aversion_lambda: float | None = None,
    is_print_summary: bool = True,
) -> dict[str, Any]:
    """Evaluate a frozen policy across the complete environment schedule and compute pooled episode metrics."""
    if not isinstance(agent, BaseHedgingAgent):
        raise TypeError("agent must inherit from BaseHedgingAgent")
    if int(getattr(env, "num_state_feature", -1)) != agent.num_state_feature:
        raise ValueError("Agent and environment state dimensions differ")
    if int(getattr(env, "num_completed_episode", 0)) != 0:
        raise ValueError("The evaluation environment has already been used; create a new instance")
    if not isinstance(is_print_summary, (bool, np.bool_)):
        raise ValueError("is_print_summary must be a boolean")
    is_print_summary = bool(is_print_summary)
    num_evaluation_episode = int(env.num_scheduled_episode)
    if is_print_summary:
        print(
            f"[Evaluation started] split={env.label}, episode samples={num_evaluation_episode}, batch_mode={env.is_batch_mode}",
            flush=True,
        )
    num_total_step = 0
    num_reset = 0
    while True:
        try:
            state, _ = env.reset()
        except StopIteration:
            break
        num_reset += 1
        while True:
            delta_action = _get_current_delta_action(agent, env)
            action = agent.get_action(
                state,
                delta_action=delta_action if agent.is_require_delta_action else None,
                is_deterministic=True,
            )
            state, _, terminated, truncated, _ = env.step(action)
            num_total_step += 1
            terminated_array = np.asarray(terminated, dtype=bool)
            truncated_array = np.asarray(truncated, dtype=bool)
            if truncated_array.any():
                raise RuntimeError("Finite-horizon hedging must not produce truncated transitions")
            if terminated_array.all():
                break
    if num_reset == 0:
        raise RuntimeError("The evaluation schedule is empty or exhausted")
    metrics = env.get_metrics(risk_aversion_lambda=risk_aversion_lambda)
    if is_print_summary:
        metric_currency = str(env.get_config().get("currency", "USD"))
        print(
            f"[Evaluation results] Mean(L)={metric_currency} {metrics['mean_loss']:,.2f}, Std(L)={metric_currency} {metrics['std_loss']:,.2f}, J_lambda={metric_currency} {metrics['j_lambda']:,.2f}, Mean(TC)={metric_currency} {metrics['mean_transaction_cost']:,.2f} (lambda={metrics['risk_aversion_lambda']:g}, per contract)",
            flush=True,
        )
    return {
        "label": env.label,
        "num_reset": num_reset,
        "num_total_step": num_total_step,
        "episode_results": env.get_completed_episode_results(),
        "metrics": metrics,
        "trajectory": env.get_trajectory(),
        "agent_config": agent.get_config(),
        "env_config": env.get_config(),
    }


__all__ = ["ReplayBuffer", "evaluate_hedging_agent", "train_hedging_agent"]
