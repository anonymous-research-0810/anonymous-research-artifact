"""Training for the paper option-hedging pipeline."""

from __future__ import annotations
import hashlib
import math
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from time import perf_counter
from typing import Any
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from .agents import _PearlHedgingAgentBase
from .data import MetaTaskDataset
from .evaluation import evaluate_meta_hedging_agent
from .replay import TaskReplayBuffer


def _get_positive_int(value: Any, name: str) -> int:
    """Return positive int."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a strictly positive integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be a strictly positive integer")
    return result


def _get_source_queue(
    task: MetaTaskDataset, *, is_simulated: bool, num_warmup: int, rng: np.random.Generator
) -> tuple[list[str], list[str]]:
    """Return source queue."""
    dataset = task.simulated_dataset if is_simulated else task.real_dataset
    manifest = dataset.episode_manifest
    by_cohort = {
        str(cohort): rows["episode_id"].astype(str).tolist()
        for cohort, rows in manifest.groupby("cohort_id", sort=False)
    }
    cohorts = list(by_cohort)
    rng.shuffle(cohorts)
    for episode_ids in by_cohort.values():
        rng.shuffle(episode_ids)
    prioritized: list[str] = []
    num_level = 0
    while len(prioritized) < dataset.num_episode:
        is_added = False
        for cohort in cohorts:
            values = by_cohort[cohort]
            if num_level < len(values):
                prioritized.append(values[num_level])
                is_added = True
        if not is_added:
            break
        num_level += 1
    if len(prioritized) != dataset.num_episode:
        raise RuntimeError("Internal error: the source episode queue is incomplete")
    return (prioritized[:num_warmup], prioritized[num_warmup:])


def _interleave_sources(
    real_ids: Sequence[str], simulated_ids: Sequence[str], *, rng: np.random.Generator
) -> list[tuple[bool, str]]:
    """Interleave sources."""
    queues = {False: list(real_ids), True: list(simulated_ids)}
    positions = {False: 0, True: 0}
    result: list[tuple[bool, str]] = []
    while positions[False] < len(queues[False]) or positions[True] < len(queues[True]):
        remaining = {source: len(queues[source]) - positions[source] for source in (False, True)}
        total = remaining[False] + remaining[True]
        if remaining[False] == 0:
            source = True
        elif remaining[True] == 0:
            source = False
        else:
            source = bool(rng.random() >= remaining[False] / total)
        result.append((source, queues[source][positions[source]]))
        positions[source] += 1
    return result


def _get_task_rollout_plan(
    task: MetaTaskDataset, *, warmup_fraction: float, rng: np.random.Generator
) -> tuple[list[tuple[bool, str]], int]:
    """Return task rollout plan."""
    num_real_warmup = int(math.ceil(warmup_fraction * task.real_dataset.num_episode))
    num_sim_warmup = int(math.ceil(warmup_fraction * task.simulated_dataset.num_episode))
    real_warmup, real_joint = _get_source_queue(
        task, is_simulated=False, num_warmup=num_real_warmup, rng=rng
    )
    sim_warmup, sim_joint = _get_source_queue(
        task, is_simulated=True, num_warmup=num_sim_warmup, rng=rng
    )
    warmup = _interleave_sources(real_warmup, sim_warmup, rng=rng)
    joint = _interleave_sources(real_joint, sim_joint, rng=rng)
    plan = warmup + joint
    expected = {
        *((False, value) for value in task.real_dataset.get_episode_ids()),
        *((True, value) for value in task.simulated_dataset.get_episode_ids()),
    }
    if len(plan) != len(set(plan)) or set(plan) != expected:
        raise RuntimeError("Internal error: the task rollout plan has omissions or duplicates")
    return (plan, len(warmup))


def _get_environment_dataset_order(
    plan: Sequence[tuple[bool, str]], *, seed: int
) -> list[tuple[bool, str]]:
    """Return environment dataset order."""
    permutation = np.random.default_rng(seed).permutation(len(plan))
    dataset_order: list[tuple[bool, str] | None] = [None] * len(plan)
    for schedule_position, dataset_position in enumerate(permutation):
        dataset_order[int(dataset_position)] = plan[schedule_position]
    if any((value is None for value in dataset_order)):
        raise RuntimeError("Internal error: cannot invert the environment episode permutation")
    return [value for value in dataset_order if value is not None]


def _rollout_episode(
    *,
    agent: _PearlHedgingAgentBase,
    env: Any,
    task: MetaTaskDataset,
    replay: TaskReplayBuffer,
    expected: tuple[bool, str],
    is_warmup: bool,
) -> int:
    """Rollout episode."""
    is_simulated, expected_episode_id = expected
    metadata = task.get_episode_metadata(expected_episode_id, is_simulated=is_simulated)
    state, reset_info = env.reset()
    actual_episode_id = str(reset_info["episode_id"])
    if actual_episode_id != expected_episode_id:
        raise RuntimeError(
            f"The environment schedule differs from the fixed rollout plan: expected={expected_episode_id}, actual={actual_episode_id}"
        )
    precision_sum: torch.Tensor | None = None
    precision_weighted_mean_sum: torch.Tensor | None = None
    if not is_warmup:
        precision_sum = torch.zeros(
            (1, agent.latent_dimension), dtype=torch.float32, device=agent.device
        )
        precision_weighted_mean_sum = torch.zeros_like(precision_sum)
    records: dict[str, list[Any]] = {
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
    while True:
        delta = env.get_bsm_call_delta() if agent.is_require_delta_action else 0.0
        if is_warmup:
            action = agent.get_random_action(delta if agent.is_delta_residual else None)
        else:
            assert precision_sum is not None
            assert precision_weighted_mean_sum is not None
            with torch.no_grad():
                mean, variance = agent.context_encoder.get_posterior_from_sufficient_statistics(
                    precision_sum, precision_weighted_mean_sum
                )
                latent = agent.context_encoder.sample(mean, variance, is_deterministic=False)
            action = agent.get_action(
                state,
                latent=latent[0],
                delta_action=delta if agent.is_require_delta_action else None,
                is_deterministic=False,
            )
        next_state, reward, terminated, truncated, _ = env.step(action)
        if bool(truncated):
            raise RuntimeError("Finite-horizon hedging must not produce truncated transitions")
        is_done = bool(terminated)
        next_delta = (
            0.0 if is_done or not agent.is_require_delta_action else env.get_bsm_call_delta()
        )
        state_array = np.asarray(state, dtype=np.float32).copy()
        next_state_array = np.asarray(next_state, dtype=np.float32).copy()
        action_value = float(np.asarray(action).reshape(-1)[0])
        reward_value = float(np.asarray(reward).reshape(-1)[0])
        records["state"].append(state_array)
        records["action"].append([action_value])
        records["reward"].append([reward_value])
        records["next_state"].append(next_state_array)
        records["terminated"].append([float(is_done)])
        records["delta_action"].append([float(delta)])
        records["next_delta_action"].append([float(next_delta)])
        transition = np.concatenate(
            (
                state_array,
                np.asarray([action_value, reward_value], dtype=np.float32),
                next_state_array,
            )
        ).reshape(1, -1)
        if not is_warmup:
            transition_tensor = torch.as_tensor(
                transition, dtype=torch.float32, device=agent.device
            )
            with torch.no_grad():
                new_precision, new_weighted_mean = agent.context_encoder.get_sufficient_statistics(
                    transition_tensor
                )
            precision_sum = precision_sum + new_precision
            precision_weighted_mean_sum = precision_weighted_mean_sum + new_weighted_mean
        state = next_state
        if is_done:
            break
    replay.add_episode(
        **metadata,
        **{name: np.asarray(values, dtype=np.float32) for name, values in records.items()},
    )
    return len(records["state"])


class _ParquetRecordWriter:
    """Parquet record writer."""

    def __init__(self, path: Path | None) -> None:
        """Initialize validated configuration and internal state."""
        self.path = path
        self._writer: Any | None = None
        self._memory_records: list[dict[str, Any]] = []

    def write(self, records: list[dict[str, Any]]) -> None:
        """Write."""
        if not records:
            return
        if self.path is None:
            self._memory_records.extend(records)
            return
        import pyarrow as pa
        import pyarrow.parquet as pq

        self.path.parent.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pandas(pd.DataFrame(records), preserve_index=False)
        if self._writer is None:
            self._writer = pq.ParquetWriter(self.path, table.schema)
        self._writer.write_table(table)

    def close(self) -> None:
        """Release active resources while retaining completed result records."""
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    def get_frame(self) -> pd.DataFrame:
        """Return frame."""
        return pd.DataFrame(self._memory_records)


def _get_episode_latent_statistics(
    agent: _PearlHedgingAgentBase,
    replays: Mapping[int, TaskReplayBuffer],
    *,
    checkpoints: tuple[int, ...] = (0, 5, 10, 15, 20),
    num_batch: int = 512,
) -> pd.DataFrame:
    """Return episode latent statistics."""
    num_batch = _get_positive_int(num_batch, "num_batch")
    records: list[dict[str, Any]] = []
    agent.context_encoder.eval()
    with torch.no_grad():
        for segment_id, replay in replays.items():
            episodes = [replay.get_episode(episode_id) for episode_id in replay.get_episode_ids()]
            for batch_start in range(0, len(episodes), num_batch):
                episode_batch = episodes[batch_start : batch_start + num_batch]
                statistics_by_prefix: dict[int, dict[str, np.ndarray]] = {}
                for num_prefix in checkpoints:
                    eligible = [
                        episode for episode in episode_batch if num_prefix <= episode.num_transition
                    ]
                    if not eligible:
                        continue
                    contexts = np.stack([episode.get_context(num_prefix) for episode in eligible])
                    context_tensor = torch.as_tensor(
                        contexts, dtype=torch.float32, device=agent.device
                    )
                    mean, variance = agent.context_encoder(context_tensor)
                    kl = agent.context_encoder.get_kl_divergence(mean, variance)
                    statistics = (
                        torch.cat(
                            (
                                mean,
                                variance.sqrt(),
                                mean.norm(dim=1, keepdim=True),
                                kl.reshape(-1, 1),
                            ),
                            dim=1,
                        )
                        .cpu()
                        .numpy()
                    )
                    statistics_by_prefix[num_prefix] = {
                        episode.episode_id: row for episode, row in zip(eligible, statistics)
                    }
                latent_dimension = agent.latent_dimension
                for episode in episode_batch:
                    for num_prefix in checkpoints:
                        row = statistics_by_prefix.get(num_prefix, {}).get(episode.episode_id)
                        if row is None:
                            continue
                        record: dict[str, Any] = {
                            "algorithm": agent.algorithm_name,
                            "segment_id": int(segment_id),
                            "episode_id": episode.episode_id,
                            "cohort_id": episode.cohort_id,
                            "source_root": episode.source_root,
                            "is_simulated": episode.is_simulated,
                            "num_observed_transition": int(num_prefix),
                            "posterior_mean_norm": float(row[2 * latent_dimension]),
                            "kl_divergence": float(row[2 * latent_dimension + 1]),
                        }
                        for index in range(latent_dimension):
                            record[f"posterior_mean_{index}"] = float(row[index])
                            record[f"posterior_std_{index}"] = float(row[latent_dimension + index])
                        records.append(record)
    return pd.DataFrame(records)


def _get_training_metrics(
    episode_results: pd.DataFrame, *, risk_aversion_lambda: float
) -> dict[str, Any]:
    """Return training metrics."""
    losses = episode_results["monetary_loss"].to_numpy(dtype=np.float64)
    costs = episode_results["monetary_transaction_cost"].to_numpy(dtype=np.float64)
    mean_loss = float(losses.mean())
    std_loss = float(losses.std(ddof=0))
    return {
        "num_episode": int(len(losses)),
        "mean_loss": mean_loss,
        "std_loss": std_loss,
        "risk_aversion_lambda": float(risk_aversion_lambda),
        "j_lambda": mean_loss + float(risk_aversion_lambda) * std_loss,
        "mean_transaction_cost": float(costs.mean()),
    }


def _get_rollout_plan_sha256(values: Sequence[tuple[bool, str]]) -> str:
    """Return rollout plan sha256."""
    encoded = "\n".join(
        (f"{int(is_simulated)}|{episode_id}" for is_simulated, episode_id in values)
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def train_meta_hedging_agent(
    agent: _PearlHedgingAgentBase,
    tasks: Sequence[MetaTaskDataset],
    *,
    get_train_env: Callable[[Any, int], Any],
    get_valid_env: Callable[[], Any] | None = None,
    num_batch: int = 64,
    num_task_batch: int = 8,
    num_recent_context_episodes: int = 4,
    simulated_batch_fraction: float = 0.5,
    warmup_fraction: float = 0.1,
    num_update_per_step: int = 1,
    num_valid_episodes: int = 200,
    num_latent_log_interval: int = 1000,
    checkpoint_path: str | Path | None = None,
    checkpoint_dir: str | Path | None = None,
    update_history_path: str | Path | None = None,
    latent_history_path: str | Path | None = None,
    seed: int = 2026,
    is_show_progress: bool = True,
    is_print_summary: bool = True,
) -> dict[str, Any]:
    """Train across regime tasks with isolated context/query roots; evaluate with causal completed-episode context and retain the validation-best checkpoint."""
    if not isinstance(agent, _PearlHedgingAgentBase):
        raise TypeError("agent must be a PEARL-TD3 or PEARL-SAC hedging agent")
    task_list = list(tasks)
    if len(task_list) < 2:
        raise ValueError("Meta-training requires at least two tasks")
    segment_ids = [task.segment_id for task in task_list]
    if len(segment_ids) != len(set(segment_ids)):
        raise ValueError("Task segment_id values must be unique")
    num_batch = _get_positive_int(num_batch, "num_batch")
    num_task_batch = min(_get_positive_int(num_task_batch, "num_task_batch"), len(task_list))
    num_recent_context_episodes = _get_positive_int(
        num_recent_context_episodes, "num_recent_context_episodes"
    )
    num_update_per_step = _get_positive_int(num_update_per_step, "num_update_per_step")
    num_valid_episodes = _get_positive_int(num_valid_episodes, "num_valid_episodes")
    num_latent_log_interval = _get_positive_int(num_latent_log_interval, "num_latent_log_interval")
    if (
        isinstance(simulated_batch_fraction, (bool, np.bool_))
        or not np.isfinite(float(simulated_batch_fraction))
        or (not 0.0 <= float(simulated_batch_fraction) <= 1.0)
    ):
        raise ValueError("simulated_batch_fraction must be in [0,1]")
    simulated_batch_fraction = float(simulated_batch_fraction)
    if (
        isinstance(warmup_fraction, bool)
        or not np.isfinite(float(warmup_fraction))
        or (not 0.0 < float(warmup_fraction) < 1.0)
    ):
        raise ValueError("warmup_fraction must be in (0,1)")
    checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else None
    checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None
    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    plans: dict[int, list[tuple[bool, str]]] = {}
    warmup_counts: dict[int, int] = {}
    environments: dict[int, Any] = {}
    replays: dict[int, TaskReplayBuffer] = {}
    task_by_id = {task.segment_id: task for task in task_list}
    environment_seeds: dict[int, int] = {}
    for task in task_list:
        plan, num_warmup = _get_task_rollout_plan(
            task, warmup_fraction=float(warmup_fraction), rng=rng
        )
        env_seed = int(seed + 10000 + task.segment_id)
        dataset_order = _get_environment_dataset_order(plan, seed=env_seed)
        environment_dataset = task.get_combined_dataset(dataset_order)
        env = get_train_env(environment_dataset, env_seed)
        actual_schedule = env.get_schedule()["episode_id"].astype(str).tolist()
        if actual_schedule != [episode_id for _, episode_id in plan]:
            raise RuntimeError(
                "get_train_env did not preserve the expected training schedule without repetition"
            )
        plans[task.segment_id] = plan
        warmup_counts[task.segment_id] = num_warmup
        environments[task.segment_id] = env
        environment_seeds[task.segment_id] = env_seed
        replays[task.segment_id] = TaskReplayBuffer(
            segment_id=task.segment_id,
            num_state_feature=agent.num_state_feature,
            seed=seed + 20000 + task.segment_id,
        )
    num_total_episode = sum((len(plan) for plan in plans.values()))
    num_real_episode = sum((task.real_dataset.num_episode for task in task_list))
    num_simulated_episode = sum((task.simulated_dataset.num_episode for task in task_list))
    num_warmup_episode = sum(warmup_counts.values())
    num_intervals = {int(environment.num_interval) for environment in environments.values()}
    reward_formulations = {
        str(environment.reward_formulation) for environment in environments.values()
    }
    currencies = {
        str(environment.get_config().get("currency", "USD"))
        for environment in environments.values()
    }
    if len(num_intervals) != 1 or len(reward_formulations) != 1 or len(currencies) != 1:
        raise RuntimeError("All meta tasks must use the same horizon and reward formulation")
    num_interval = next(iter(num_intervals))
    reward_formulation = next(iter(reward_formulations))
    metric_currency = next(iter(currencies))
    num_expected_step = num_total_episode * num_interval
    num_expected_meta_update = (
        (num_total_episode - num_warmup_episode) * num_interval * num_update_per_step
    )
    if num_expected_meta_update <= 0:
        raise RuntimeError("The planned number of meta-updates must be strictly positive")
    num_total_step = 0
    num_completed_episode = 0
    num_meta_update = 0
    positions = {segment_id: 0 for segment_id in segment_ids}
    completed_pairs = {segment_id: [] for segment_id in segment_ids}
    validation_records: list[dict[str, Any]] = []
    pending_update_records: list[dict[str, Any]] = []
    pending_latent_records: list[dict[str, Any]] = []
    update_writer = _ParquetRecordWriter(
        Path(update_history_path) if update_history_path is not None else None
    )
    latent_writer = _ParquetRecordWriter(
        Path(latent_history_path) if latent_history_path is not None else None
    )
    best_validation: dict[str, Any] | None = None
    best_state: dict[str, Any] | None = None
    best_j_lambda = math.inf
    best_checkpoint_path: Path | None = None
    last_validation_episode = -1
    latest_actor_loss: float | None = None
    latest_critic_loss: float | None = None
    latest_encoder_loss: float | None = None
    latest_kl_loss: float | None = None
    latest_kl_coefficient: float | None = None
    latest_alpha: float | None = None
    start_time = perf_counter()
    if is_print_summary:
        print(
            f"[Training started] algorithm={agent.algorithm_name.upper()}, device={agent.device}, reward={reward_formulation}, latent_dim={agent.latent_dimension}",
            flush=True,
        )
        print(
            f"[Training data] tasks={len(task_list)}, real episode samples={num_real_episode}, simulated episode samples={num_simulated_episode}, rollouts={num_total_episode}, intervals per episode={num_interval}, total training steps={num_expected_step}, warm-up episodes={num_warmup_episode}",
            flush=True,
        )
    progress = tqdm(
        total=num_total_episode,
        desc=f"Training {agent.algorithm_name.upper()}",
        unit="episode",
        dynamic_ncols=True,
        disable=not is_show_progress,
    )

    def update_progress(*, is_warmup: bool) -> None:
        """Update progress."""
        progress_fields: dict[str, Any] = {
            "ep": f"{num_completed_episode}/{num_total_episode}",
            "upd": num_meta_update,
        }
        if is_warmup:
            progress_fields["warmup"] = f"{num_completed_episode}/{num_warmup_episode}"
        if latest_actor_loss is not None:
            progress_fields["actor"] = f"{latest_actor_loss:.3g}"
        if latest_critic_loss is not None:
            progress_fields["critic"] = f"{latest_critic_loss:.3g}"
        if latest_encoder_loss is not None:
            progress_fields["encoder"] = f"{latest_encoder_loss:.3g}"
        if latest_kl_loss is not None:
            progress_fields["kl"] = f"{latest_kl_loss:.3g}"
        if latest_kl_coefficient is not None:
            progress_fields["kl_beta"] = f"{latest_kl_coefficient:.3g}"
        if latest_alpha is not None:
            progress_fields["alpha"] = f"{latest_alpha:.3g}"
        if validation_records:
            progress_fields[f"valid_J({metric_currency})"] = (
                f"{validation_records[-1]['j_lambda']:.4g}"
            )
        progress.set_postfix(progress_fields, refresh=False)

    def run_validation() -> None:
        """Run validation."""
        nonlocal best_validation, best_state, best_j_lambda
        nonlocal best_checkpoint_path, last_validation_episode
        if get_valid_env is None or num_completed_episode == last_validation_episode:
            return
        if is_show_progress:
            progress.write(
                f"[Validation] Completed training episodes {num_completed_episode}/{num_total_episode}"
            )
        evaluation = evaluate_meta_hedging_agent(
            agent,
            get_valid_env(),
            num_recent_context_episodes=num_recent_context_episodes,
            context_seed=seed,
            is_record_latent_statistics=False,
            is_print_summary=is_print_summary,
        )
        metrics = evaluation["metrics"]
        j_lambda = float(metrics["j_lambda"])
        if not np.isfinite(j_lambda):
            raise RuntimeError("Validation J_lambda must be a finite number")
        periodic_path = None
        if checkpoint_dir is not None:
            periodic_path = checkpoint_dir / f"agent_episode_{num_completed_episode:08d}.pt"
            agent.save(periodic_path)
        is_best = j_lambda < best_j_lambda
        if is_best:
            for record in validation_records:
                record["is_best"] = False
            best_j_lambda = j_lambda
            best_validation = {"num_train_episode": num_completed_episode, **metrics}
            best_state = {
                "config": deepcopy(agent.get_config()),
                **deepcopy(agent._get_checkpoint_state()),
            }
            if checkpoint_dir is not None:
                best_checkpoint_path = checkpoint_dir / "best_agent.pt"
                agent.save(best_checkpoint_path)
        validation_records.append(
            {
                "num_train_episode": num_completed_episode,
                "num_train_step": num_total_step,
                "num_meta_update": num_meta_update,
                "is_best": is_best,
                "checkpoint_path": str(periodic_path) if periodic_path else None,
                **metrics,
            }
        )
        last_validation_episode = num_completed_episode

    try:
        while any(
            (positions[segment_id] < warmup_counts[segment_id] for segment_id in segment_ids)
        ):
            for segment_id in segment_ids:
                if positions[segment_id] >= warmup_counts[segment_id]:
                    continue
                position = positions[segment_id]
                pair = plans[segment_id][position]
                num_step = _rollout_episode(
                    agent=agent,
                    env=environments[segment_id],
                    task=task_by_id[segment_id],
                    replay=replays[segment_id],
                    expected=pair,
                    is_warmup=True,
                )
                positions[segment_id] += 1
                completed_pairs[segment_id].append(pair)
                num_total_step += num_step
                num_completed_episode += 1
                update_progress(is_warmup=True)
                progress.update(1)
        insufficient = [
            segment_id
            for segment_id in segment_ids
            if not replays[segment_id].can_sample(num_min_transition=num_batch)
        ]
        if insufficient:
            raise RuntimeError(
                f"The following tasks lack sufficient replay/context source roots after warm-up: {insufficient}"
            )
        while any((positions[segment_id] < len(plans[segment_id]) for segment_id in segment_ids)):
            for segment_id in segment_ids:
                if positions[segment_id] >= len(plans[segment_id]):
                    continue
                position = positions[segment_id]
                pair = plans[segment_id][position]
                num_step = _rollout_episode(
                    agent=agent,
                    env=environments[segment_id],
                    task=task_by_id[segment_id],
                    replay=replays[segment_id],
                    expected=pair,
                    is_warmup=False,
                )
                positions[segment_id] += 1
                completed_pairs[segment_id].append(pair)
                num_total_step += num_step
                num_completed_episode += 1
                ready_ids = [
                    value
                    for value in segment_ids
                    if replays[value].can_sample(num_min_transition=num_batch)
                ]
                for _ in range(num_step * num_update_per_step):
                    selected = rng.choice(
                        ready_ids, size=min(num_task_batch, len(ready_ids)), replace=False
                    )
                    task_batches = [
                        replays[int(value)].sample_context_and_batch(
                            num_batch=num_batch,
                            num_recent_context_episodes=num_recent_context_episodes,
                            simulated_batch_fraction=simulated_batch_fraction,
                            device=agent.device,
                        )
                        for value in selected
                    ]
                    is_record_latent = (
                        num_meta_update == 0 or (num_meta_update + 1) % num_latent_log_interval == 0
                    )
                    metrics = agent.update(
                        task_batches, is_record_latent_statistics=is_record_latent
                    )
                    latent_records = list(metrics.pop("latent_records"))
                    if bool(metrics.get("is_actor_updated", 1.0)):
                        latest_actor_loss = float(metrics["actor_loss"])
                    latest_critic_loss = float(metrics["critic_loss"])
                    latest_encoder_loss = float(metrics["encoder_loss"])
                    latest_kl_loss = float(metrics["kl_loss"])
                    latest_kl_coefficient = float(metrics["kl_coefficient"])
                    if "alpha" in metrics:
                        latest_alpha = float(metrics["alpha"])
                    num_meta_update += 1
                    pending_latent_records.extend(latent_records)
                    pending_update_records.append(
                        {
                            "num_meta_update": num_meta_update,
                            "num_train_episode": num_completed_episode,
                            "num_train_step": num_total_step,
                            **metrics,
                        }
                    )
                    if len(pending_update_records) >= 1000:
                        update_writer.write(pending_update_records)
                        pending_update_records.clear()
                    if latent_records:
                        latent_writer.write(pending_latent_records)
                        pending_latent_records.clear()
                is_validation_due = get_valid_env is not None and (
                    num_completed_episode % num_valid_episodes == 0
                    or num_completed_episode == num_total_episode
                )
                if is_validation_due:
                    run_validation()
                update_progress(is_warmup=False)
                progress.update(1)
        if get_valid_env is not None:
            run_validation()
    finally:
        progress.close()
        update_writer.write(pending_update_records)
        update_writer.close()
        latent_writer.write(pending_latent_records)
        latent_writer.close()
    if get_valid_env is not None and best_validation is None:
        raise RuntimeError("No validation checkpoint is available after training")
    if checkpoint_dir is not None:
        agent.save(checkpoint_dir / "training_final_agent.pt")
    if best_validation is not None:
        if best_checkpoint_path is not None:
            agent.load(best_checkpoint_path, is_load_optimizer=True)
        else:
            assert best_state is not None
            agent._load_checkpoint_state(best_state, is_load_optimizer=True)
    if checkpoint_path is not None:
        agent.save(checkpoint_path)
        selected_checkpoint_path = checkpoint_path
    elif best_checkpoint_path is not None:
        selected_checkpoint_path = best_checkpoint_path
    elif checkpoint_dir is not None:
        selected_checkpoint_path = checkpoint_dir / "selected_agent.pt"
        agent.save(selected_checkpoint_path)
    else:
        selected_checkpoint_path = None
    usage_records = []
    for segment_id in segment_ids:
        plan = plans[segment_id]
        completed = completed_pairs[segment_id]
        missing = sorted(set(plan) - set(completed))
        duplicates = len(completed) - len(set(completed))
        is_exact = (
            completed == plan
            and (not missing)
            and (duplicates == 0)
            and (environments[segment_id].num_completed_episode == len(plan))
        )
        usage_records.append(
            {
                "segment_id": segment_id,
                "num_planned_episode": len(plan),
                "num_completed_episode": len(completed),
                "num_warmup_episode": warmup_counts[segment_id],
                "num_missing_episode": len(missing),
                "num_duplicate_episode": duplicates,
                "is_exactly_once": is_exact,
                "planned_sequence_sha256": _get_rollout_plan_sha256(plan),
                "completed_sequence_sha256": _get_rollout_plan_sha256(completed),
                "environment_seed": environment_seeds[segment_id],
                **replays[segment_id].get_usage_summary(),
            }
        )
        if not is_exact:
            raise RuntimeError(f"segment_id={segment_id} episode audit failed")
    episode_frames = []
    trajectory_frames = []
    for segment_id in segment_ids:
        episode_frame = environments[segment_id].get_completed_episode_results()
        episode_frame.insert(0, "segment_id", segment_id)
        episode_frames.append(episode_frame)
        trajectory = environments[segment_id].get_trajectory()
        if not trajectory.empty:
            trajectory.insert(0, "segment_id", segment_id)
            trajectory_frames.append(trajectory)
    episode_results = pd.concat(episode_frames, ignore_index=True)
    trajectory = (
        pd.concat(trajectory_frames, ignore_index=True) if trajectory_frames else pd.DataFrame()
    )
    risk_lambda = float(environments[segment_ids[0]].risk_aversion_lambda)
    training_metrics = _get_training_metrics(episode_results, risk_aversion_lambda=risk_lambda)
    latent_episode_statistics = _get_episode_latent_statistics(agent, replays)
    elapsed = perf_counter() - start_time
    if is_print_summary:
        print(
            f"[Training completed] episode={num_completed_episode}, step={num_total_step}, meta_update={num_meta_update}, elapsed time={elapsed:.2f}s, best_valid_J({metric_currency})={(best_j_lambda if best_validation else float('nan')):.4g}",
            flush=True,
        )
    return {
        "agent": agent,
        "num_episode": num_completed_episode,
        "num_total_step": num_total_step,
        "num_meta_update": num_meta_update,
        "num_elapsed_second": elapsed,
        "episode_results": episode_results,
        "trajectory": trajectory,
        "metrics": training_metrics,
        "update_history": update_writer.get_frame(),
        "validation_history": pd.DataFrame(validation_records),
        "best_validation": best_validation,
        "selected_checkpoint_path": (
            str(selected_checkpoint_path) if selected_checkpoint_path is not None else None
        ),
        "latent_training_history": latent_writer.get_frame(),
        "latent_episode_statistics": latent_episode_statistics,
        "episode_usage_audit": usage_records,
        "task_environment_configs": [
            {"segment_id": segment_id, "environment": environments[segment_id].get_config()}
            for segment_id in segment_ids
        ],
        "replay_buffers": replays,
        "config": {
            "agent": agent.get_config(),
            "num_batch": num_batch,
            "num_task_batch": num_task_batch,
            "num_recent_context_episodes": num_recent_context_episodes,
            "simulated_batch_fraction": simulated_batch_fraction,
            "warmup_fraction": float(warmup_fraction),
            "num_update_per_step": num_update_per_step,
            "kl_coefficient": float(agent.kl_coefficient),
            "num_valid_episodes": num_valid_episodes,
            "num_latent_log_interval": num_latent_log_interval,
            "seed": seed,
        },
    }
