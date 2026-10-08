"""Evaluation for the paper option-hedging pipeline."""

from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd
import torch
from .agents import _PearlHedgingAgentBase
from .replay import TaskReplayBuffer

RECENT_COMPLETED_CONTEXT = "recent_completed_episode"
VALID_EVALUATION_CONTEXT_PROTOCOLS = (RECENT_COMPLETED_CONTEXT,)


def _get_row_arrays(value: Any, *, num_row: int, num_column: int) -> np.ndarray:
    """Return row arrays."""
    array = np.asarray(value, dtype=np.float32)
    if num_row == 1 and array.shape == (num_column,):
        array = array.reshape(1, num_column)
    if array.shape != (num_row, num_column):
        raise RuntimeError(
            f"Environment output shape must be {(num_row, num_column)}; received {array.shape}"
        )
    return array


def _get_vector(value: Any, *, num_row: int, dtype: Any) -> np.ndarray:
    """Return vector."""
    array = np.asarray(value, dtype=dtype)
    if array.ndim == 0:
        array = array.reshape(1)
    if array.shape != (num_row,):
        raise RuntimeError(f"Environment vector shape must be {(num_row,)}; received {array.shape}")
    return array


def _get_episode_identifiers(reset_info: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Return episode identifiers."""
    episode_ids = np.asarray(reset_info["episode_id"]).reshape(-1).astype(str).tolist()
    cohort_ids = np.asarray(reset_info["cohort_id"]).reshape(-1).astype(str).tolist()
    if not episode_ids or len(cohort_ids) != len(episode_ids):
        raise RuntimeError("episode_id/cohort_id are not aligned with the evaluation batch")
    return (episode_ids, cohort_ids)


def _extend_latent_records(
    records: list[dict[str, Any]],
    *,
    agent: _PearlHedgingAgentBase,
    mean: torch.Tensor,
    variance: torch.Tensor,
    episode_ids: list[str],
    cohort_ids: list[str],
    label: str,
    num_observed_transition: int,
    num_total_context_transition: int | None = None,
    context_metadata: dict[str, Any] | None = None,
) -> None:
    """Extend latent records."""
    if (
        mean.shape != variance.shape
        or mean.shape[0] != len(episode_ids)
        or len(cohort_ids) != len(episode_ids)
    ):
        raise RuntimeError("The posterior is not aligned with the evaluation episode batch")
    kl = agent.context_encoder.get_kl_divergence(mean, variance)
    statistics = (
        torch.cat((mean, variance.sqrt(), mean.norm(dim=1, keepdim=True), kl.reshape(-1, 1)), dim=1)
        .detach()
        .cpu()
        .numpy()
    )
    latent_dimension = agent.latent_dimension
    num_total_context = (
        int(num_observed_transition)
        if num_total_context_transition is None
        else int(num_total_context_transition)
    )
    metadata = dict(context_metadata or {})
    for row, episode_id, cohort_id in zip(statistics, episode_ids, cohort_ids):
        record: dict[str, Any] = {
            "algorithm": agent.algorithm_name,
            "label": label,
            "episode_id": str(episode_id),
            "cohort_id": str(cohort_id),
            "num_observed_transition": int(num_observed_transition),
            "num_total_context_transition": num_total_context,
            "posterior_mean_norm": float(row[2 * latent_dimension]),
            "kl_divergence": float(row[2 * latent_dimension + 1]),
            **metadata,
        }
        for index in range(latent_dimension):
            record[f"posterior_mean_{index}"] = float(row[index])
            record[f"posterior_std_{index}"] = float(row[latent_dimension + index])
        records.append(record)


def _evaluate_recent_completed_context(
    agent: _PearlHedgingAgentBase,
    env: Any,
    *,
    num_recent_context_episodes: int,
    context_seed: int | None,
    is_record_latent_statistics: bool,
) -> tuple[list[dict[str, Any]], int, int, dict[str, Any]]:
    """Evaluate recent completed context."""
    replay = TaskReplayBuffer(
        segment_id=1, num_state_feature=agent.num_state_feature, seed=context_seed
    )
    pending: list[tuple[pd.Timestamp, dict[str, Any]]] = []
    latent_records: list[dict[str, Any]] = []
    num_total_step = 0
    num_reset = 0
    num_prior_cohort = 0
    num_contextualized_cohort = 0
    previous_t0: pd.Timestamp | None = None
    while True:
        try:
            state, reset_info = env.reset()
        except StopIteration:
            break
        num_reset += 1
        episode_ids, cohort_ids = _get_episode_identifiers(reset_info)
        num_active = len(episode_ids)
        t0_values = pd.DatetimeIndex(np.asarray(reset_info["date"]).reshape(-1))
        cohort_values = pd.DatetimeIndex(np.asarray(reset_info["cohort_id"]).reshape(-1))
        if len(t0_values) != num_active or len(cohort_values) != num_active:
            raise RuntimeError("Evaluation dates are not aligned with the episode batch")
        batch_index = pd.DataFrame(
            {
                "position": np.arange(num_active, dtype=np.int64),
                "t0": t0_values,
                "cohort_id": cohort_values,
            }
        )
        if not pd.DatetimeIndex(batch_index["t0"]).is_monotonic_increasing:
            raise RuntimeError("Evaluation batch episodes are not sorted by inception")
        if pd.Timestamp(batch_index["cohort_id"].min()) < pd.Timestamp(batch_index["t0"].max()):
            raise RuntimeError(
                "The evaluation batch includes expiries before later inceptions; causal cross-cohort context cannot be guaranteed"
            )
        batch_latent = torch.empty(
            (num_active, agent.latent_dimension), dtype=torch.float32, device=agent.device
        )
        cohort_groups = batch_index.groupby(["t0", "cohort_id"], sort=False).groups
        for (current_t0_value, _), positions_index in cohort_groups.items():
            current_t0 = pd.Timestamp(current_t0_value)
            positions = np.asarray(positions_index, dtype=np.int64)
            if previous_t0 is not None and current_t0 < previous_t0:
                raise RuntimeError("The validation/test schedule is not ordered by inception")
            previous_t0 = current_t0
            newly_available: list[tuple[pd.Timestamp, dict[str, Any]]] = []
            still_pending: list[tuple[pd.Timestamp, dict[str, Any]]] = []
            for available_date, episode in pending:
                if available_date < current_t0:
                    newly_available.append((available_date, episode))
                else:
                    still_pending.append((available_date, episode))
            pending = still_pending
            for _, episode in sorted(newly_available, key=lambda item: item[0]):
                replay.add_episode(**episode)
            if replay.num_episode == 0:
                empty_context = torch.empty(
                    (0, agent.num_context_feature), dtype=torch.float32, device=agent.device
                )
                latent, posterior_mean, posterior_variance = agent.infer_posterior(
                    empty_context, is_deterministic=True
                )
                context_metadata = {
                    "context_protocol": RECENT_COMPLETED_CONTEXT,
                    "context_episode_id": None,
                    "context_cohort_id": None,
                    "context_source_root": None,
                    "context_is_simulated": None,
                    "num_context_candidate_episode": 0,
                }
                num_context = 0
                num_prior_cohort += 1
            else:
                context_sample = replay.sample_recent_context(
                    num_recent_context_episodes=num_recent_context_episodes, device=agent.device
                )
                context_expiry = pd.Timestamp(context_sample["context_cohort_id"])
                if not context_expiry < current_t0:
                    raise RuntimeError(
                        "Internal error: evaluation context contains information unavailable at the current inception"
                    )
                latent, posterior_mean, posterior_variance = agent.infer_posterior(
                    context_sample["context"], is_deterministic=True
                )
                num_context = int(context_sample["num_context"])
                context_metadata = {
                    "context_protocol": RECENT_COMPLETED_CONTEXT,
                    "context_episode_id": context_sample["context_episode_id"],
                    "context_cohort_id": context_sample["context_cohort_id"],
                    "context_source_root": context_sample["context_source_root"],
                    "context_is_simulated": context_sample["context_is_simulated"],
                    "num_context_candidate_episode": context_sample[
                        "num_context_candidate_episode"
                    ],
                }
                num_contextualized_cohort += 1
            batch_latent[positions] = latent
            if is_record_latent_statistics:
                num_cohort_episode = len(positions)
                _extend_latent_records(
                    latent_records,
                    agent=agent,
                    mean=posterior_mean.reshape(1, -1).expand(num_cohort_episode, -1),
                    variance=posterior_variance.reshape(1, -1).expand(num_cohort_episode, -1),
                    episode_ids=[episode_ids[index] for index in positions],
                    cohort_ids=[cohort_ids[index] for index in positions],
                    label=env.label,
                    num_observed_transition=0,
                    num_total_context_transition=num_context,
                    context_metadata=context_metadata,
                )
        state_array = _get_row_arrays(state, num_row=num_active, num_column=agent.num_state_feature)
        transition_rows: dict[str, list[np.ndarray]] = {
            name: []
            for name in (
                "state",
                "action",
                "reward",
                "next_state",
                "terminated",
                "delta_action",
                "next_delta_action",
            )
        }
        is_episode_terminated = False
        for num_step in range(int(env.num_interval)):
            delta_array = (
                _get_vector(env.get_bsm_call_delta(), num_row=num_active, dtype=np.float32)
                if agent.is_require_delta_action
                else np.zeros(num_active, dtype=np.float32)
            )
            action = agent.get_action(
                state_array[0] if num_active == 1 else state_array,
                latent=batch_latent[0] if num_active == 1 else batch_latent,
                delta_action=delta_array if agent.is_require_delta_action else None,
                is_deterministic=True,
            )
            next_state, reward, terminated, truncated, _ = env.step(action)
            action_array = _get_vector(action, num_row=num_active, dtype=np.float32)
            reward_array = _get_vector(reward, num_row=num_active, dtype=np.float32)
            terminated_array = _get_vector(terminated, num_row=num_active, dtype=bool)
            truncated_array = _get_vector(truncated, num_row=num_active, dtype=bool)
            if truncated_array.any():
                raise RuntimeError("Finite-horizon hedging must not produce truncated transitions")
            if terminated_array.any() and (not terminated_array.all()):
                raise RuntimeError("Episodes in the evaluation batch terminate at different steps")
            next_state_array = _get_row_arrays(
                next_state, num_row=num_active, num_column=agent.num_state_feature
            )
            next_delta_array = (
                np.zeros(num_active, dtype=np.float32)
                if terminated_array.all() or not agent.is_require_delta_action
                else _get_vector(env.get_bsm_call_delta(), num_row=num_active, dtype=np.float32)
            )
            transition_rows["state"].append(state_array.copy())
            transition_rows["action"].append(action_array.reshape(-1, 1))
            transition_rows["reward"].append(reward_array.reshape(-1, 1))
            transition_rows["next_state"].append(next_state_array.copy())
            transition_rows["terminated"].append(terminated_array.astype(np.float32).reshape(-1, 1))
            transition_rows["delta_action"].append(delta_array.reshape(-1, 1))
            transition_rows["next_delta_action"].append(next_delta_array.reshape(-1, 1))
            num_total_step += num_active
            state_array = next_state_array
            if terminated_array.all():
                if num_step + 1 != int(env.num_interval):
                    raise RuntimeError("The environment terminated before the expected horizon H")
                is_episode_terminated = True
                break
        if not is_episode_terminated:
            raise RuntimeError("The environment did not terminate after H intervals")
        stacked = {
            name: np.ascontiguousarray(np.stack(values, axis=1), dtype=np.float32)
            for name, values in transition_rows.items()
        }
        for index, (episode_id, cohort_id, current_expiry) in enumerate(
            zip(episode_ids, cohort_ids, cohort_values)
        ):
            pending.append(
                (
                    pd.Timestamp(current_expiry),
                    {
                        "episode_id": episode_id,
                        "cohort_id": cohort_id,
                        "is_simulated": False,
                        "anchor_cohort_id": None,
                        **{name: values[index] for name, values in stacked.items()},
                    },
                )
            )
    protocol = {
        "name": "causal_recent_completed_episode",
        "implementation_name": RECENT_COMPLETED_CONTEXT,
        "posterior_reset": "recomputed_once_at_each_cohort",
        "context_source": "one_causal_prefix_from_recent_eligible_completed_episodes",
        "context_eligibility": "context_expiry_strictly_before_query_t0",
        "num_recent_context_episodes": num_recent_context_episodes,
        "context_seed": context_seed,
        "cross_episode_context": True,
        "same_cohort_context": False,
        "current_query_transitions_in_context": False,
        "posterior_fixed_within_cohort": True,
        "batching": "maximal_fixed_date_safe_cohort_chunks",
        "evaluation_latent": "posterior_mean",
        "order_invariant_across_episodes": False,
        "num_prior_cohort": num_prior_cohort,
        "num_contextualized_cohort": num_contextualized_cohort,
    }
    return (latent_records, num_total_step, num_reset, protocol)


def evaluate_meta_hedging_agent(
    agent: _PearlHedgingAgentBase,
    env: Any,
    *,
    risk_aversion_lambda: float | None = None,
    context_protocol: str = RECENT_COMPLETED_CONTEXT,
    num_recent_context_episodes: int = 4,
    context_seed: int | None = 2026,
    is_record_latent_statistics: bool = True,
    is_print_summary: bool = True,
) -> dict[str, Any]:
    """Evaluate frozen model parameters using only context permitted by the selected protocol. Deployment context comes from already completed episodes."""
    if not isinstance(agent, _PearlHedgingAgentBase):
        raise TypeError("agent must be PearlTD3HedgingAgent or PearlSACHedgingAgent")
    if int(getattr(env, "num_state_feature", -1)) != agent.num_state_feature:
        raise ValueError("Agent and environment state dimensions differ")
    if int(getattr(env, "num_completed_episode", 0)) != 0:
        raise ValueError("The evaluation environment has already been used; create a new instance")
    for name, value in (
        ("is_record_latent_statistics", is_record_latent_statistics),
        ("is_print_summary", is_print_summary),
    ):
        if not isinstance(value, (bool, np.bool_)):
            raise ValueError(f"{name} must be a boolean")
    protocol_name = str(context_protocol).lower()
    if protocol_name not in VALID_EVALUATION_CONTEXT_PROTOCOLS:
        raise ValueError(f"context_protocol must be {VALID_EVALUATION_CONTEXT_PROTOCOLS}  ")
    if (
        isinstance(num_recent_context_episodes, (bool, np.bool_))
        or int(num_recent_context_episodes) != num_recent_context_episodes
        or int(num_recent_context_episodes) <= 0
    ):
        raise ValueError("num_recent_context_episodes must be a strictly positive integer")
    if context_seed is not None and (
        isinstance(context_seed, (bool, np.bool_)) or int(context_seed) != context_seed
    ):
        raise ValueError("context_seed must be an integer or None")
    is_record_latent_statistics = bool(is_record_latent_statistics)
    is_print_summary = bool(is_print_summary)
    num_evaluation_episode = int(env.num_scheduled_episode)
    if is_print_summary:
        print(
            f"[Evaluation started] algorithm={agent.algorithm_name.upper()}, split={env.label}, episode samples={num_evaluation_episode}, batch_mode={env.is_batch_mode}, posterior={protocol_name}",
            flush=True,
        )
    modules = [agent.context_encoder, agent.actor, agent.q1, agent.q2]
    previous_modes = [module.training for module in modules]
    for module in modules:
        module.eval()
    try:
        with torch.no_grad():
            latent_records, num_total_step, num_reset, protocol = (
                _evaluate_recent_completed_context(
                    agent,
                    env,
                    num_recent_context_episodes=int(num_recent_context_episodes),
                    context_seed=None if context_seed is None else int(context_seed),
                    is_record_latent_statistics=is_record_latent_statistics,
                )
            )
        if num_reset == 0:
            raise RuntimeError("The evaluation schedule is empty or exhausted")
        metrics = env.get_metrics(risk_aversion_lambda=risk_aversion_lambda)
    finally:
        for module, was_training in zip(modules, previous_modes):
            module.train(was_training)
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
        "latent_statistics": pd.DataFrame(latent_records),
        "context_protocol": protocol,
        "agent_config": agent.get_config(),
        "env_config": env.get_config(),
    }
