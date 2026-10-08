"""Train meta for the paper option-hedging pipeline."""

from __future__ import annotations
import argparse
import json
import sys
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
import numpy as np
import pandas as pd
import torch

CODES_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = CODES_ROOT
for _source_directory in (
    CODES_ROOT,
    CODES_ROOT / "meta_rl",
    CODES_ROOT / "rl_envs",
    CODES_ROOT / "datasets",
    CODES_ROOT / "data_processing",
    CODES_ROOT / "segmentation",
    CODES_ROOT / "simulation",
):
    if str(_source_directory) not in sys.path:
        sys.path.insert(0, str(_source_directory))
from option_dataset import OptionDataset, get_option_dataset
from hedging_env import HedgingEnv, StateNormalizer, get_state_normalizer
from hedging_meta_rl import (
    PearlSACHedgingAgent,
    PearlTD3HedgingAgent,
    load_meta_task_datasets,
    load_real_dataset_from_simulation_result,
    train_meta_hedging_agent,
)
from market_simulation import load_simulation_result, resolve_simulation_result_directory
from data_source import VALID_DATA_SOURCES, resolve_data_root, validate_data_source
from train_basis import (
    _merge_config_strict,
    _write_json_atomic,
    get_datetime_record,
    get_default_basis_config,
    get_dataset_record,
    get_file_sha256,
    get_contract_multiplier,
    get_currency,
    get_safe_path_component,
    get_system_info,
    prune_run_checkpoints,
    set_global_seed,
)

META_CONFIG_SCHEMA_VERSION = 5
VALID_META_ALGORITHMS = ("pearl_td3", "pearl_sac")


def get_default_meta_config() -> dict[str, Any]:
    """Return the eight paper meta-RL configurations with a four-dimensional latent and recent completed-episode context."""
    basis = get_default_basis_config()
    environment = deepcopy(basis["environment"])
    environment["reward_formulations"] = ["shaped_accounting", "cash_flow"]
    return {
        "schema_version": META_CONFIG_SCHEMA_VERSION,
        "experiment_name": "meta",
        "data_source": basis.get("data_source", "spx"),
        "simulation": {
            "result": None,
            "result_root": "simulation/sim_results",
            "is_exclude_cross_boundary": False,
        },
        "data": deepcopy(basis["data"]),
        "state_preprocessing": deepcopy(basis["state_preprocessing"]),
        "environment": environment,
        "agent": {
            "algorithm_names": ["pearl_td3", "pearl_sac"],
            "is_delta_residual_values": [True, False],
            "max_delta_residual": 0.1,
            "num_state_feature": 7,
            "latent_dimension": 4,
            "context_hidden_dims": [128, 128],
            "encoder_learning_rate": 0.0001,
            "kl_coefficient": 0.001,
            "td3": deepcopy(basis["agent"]["td3"]),
            "sac": deepcopy(basis["agent"]["sac"]),
        },
        "training": {
            "seeds": list(range(10)),
            "num_batch": 64,
            "num_task_batch": 8,
            "num_recent_context_episodes": 4,
            "simulated_batch_fraction": 0.5,
            "warmup_fraction": 0.1,
            "num_update_per_step": 1,
            "num_valid_episodes": 200,
            "num_latent_log_interval": 1000,
            "is_show_progress": True,
            "is_print_summary": True,
            "is_save_train_trajectory": True,
            "is_continue_on_error": False,
            "is_torch_deterministic": False,
            "device": None,
            "output_root": "train_results/meta",
        },
        "testing": {
            "output_root": "test_results/meta",
            "is_print_summary": True,
            "is_save_test_trajectory": True,
            "device": None,
        },
    }


def _validate_meta_config(config: Mapping[str, Any]) -> None:
    """Validate meta config."""
    if config.get("schema_version") != META_CONFIG_SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {META_CONFIG_SCHEMA_VERSION}")
    validate_data_source(config.get("data_source", "spx"))
    algorithms = list(config["agent"]["algorithm_names"])
    if not algorithms or len(algorithms) != len(set(algorithms)):
        raise ValueError("agent.algorithm_names must be a nonempty sequence without duplicates")
    if not set(algorithms).issubset(VALID_META_ALGORITHMS):
        raise ValueError(f"algorithm_names must be selected from {VALID_META_ALGORITHMS}")
    residual_values = list(config["agent"]["is_delta_residual_values"])
    if not residual_values or not all((isinstance(value, bool) for value in residual_values)):
        raise ValueError("is_delta_residual_values must be a nonempty boolean sequence")
    rewards = list(config["environment"]["reward_formulations"])
    if not rewards or not set(rewards).issubset({"shaped_accounting", "cash_flow"}):
        raise ValueError("reward_formulations contains invalid values")
    if (
        isinstance(config["agent"]["latent_dimension"], bool)
        or int(config["agent"]["latent_dimension"]) != config["agent"]["latent_dimension"]
    ):
        raise ValueError("latent_dimension must be an integer")
    if int(config["agent"]["latent_dimension"]) <= 0:
        raise ValueError("latent_dimension must be strictly positive")
    if (
        not np.isfinite(float(config["agent"]["kl_coefficient"]))
        or float(config["agent"]["kl_coefficient"]) < 0.0
    ):
        raise ValueError("agent.kl_coefficient must be finite and nonnegative")
    training = config["training"]
    for name in (
        "num_batch",
        "num_task_batch",
        "num_recent_context_episodes",
        "num_update_per_step",
        "num_valid_episodes",
        "num_latent_log_interval",
    ):
        if (
            isinstance(training[name], bool)
            or int(training[name]) != training[name]
            or int(training[name]) <= 0
        ):
            raise ValueError(f"training.{name} must be a strictly positive integer")
    if not 0.0 < float(training["warmup_fraction"]) < 1.0:
        raise ValueError("training.warmup_fraction must be in (0,1)")
    if isinstance(training["simulated_batch_fraction"], (bool, np.bool_)):
        raise ValueError("training.simulated_batch_fraction must not be a boolean")
    simulated_batch_fraction = float(training["simulated_batch_fraction"])
    if not (np.isfinite(simulated_batch_fraction) and 0.0 <= simulated_batch_fraction <= 1.0):
        raise ValueError("training.simulated_batch_fraction must be in [0,1]")
    if not training["seeds"]:
        raise ValueError("training.seeds must not be empty")
    for seed in training["seeds"]:
        if isinstance(seed, bool) or int(seed) != seed:
            raise ValueError("training.seeds must contain integers only")


def get_meta_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Return meta config."""
    config = get_default_meta_config()
    if config_path is not None:
        override = json.loads(Path(config_path).read_text(encoding="utf-8"))
        if not isinstance(override, Mapping):
            raise ValueError("The configuration root must be a JSON object")
        config = _merge_config_strict(config, override)
    _validate_meta_config(config)
    return config


def get_meta_run_specs(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Expand the meta-RL experiment matrix into deterministic run specifications."""
    scales = config["environment"]["training_reward_scales"]
    specs: list[dict[str, Any]] = []
    for seed in config["training"]["seeds"]:
        for algorithm in config["agent"]["algorithm_names"]:
            for is_residual in config["agent"]["is_delta_residual_values"]:
                for reward in config["environment"]["reward_formulations"]:
                    action_name = "delta_residual" if is_residual else "direct"
                    run_id = f"{algorithm}__{action_name}__{reward}__seed{int(seed)}"
                    specs.append(
                        {
                            "run_id": run_id,
                            "algorithm_name": algorithm,
                            "is_delta_residual": bool(is_residual),
                            "reward_formulation": reward,
                            "training_reward_scale": float(scales[reward]),
                            "seed": int(seed),
                        }
                    )
    return specs


def get_meta_agent(
    config: Mapping[str, Any],
    run_spec: Mapping[str, Any],
    *,
    device: str | torch.device | None = None,
) -> PearlTD3HedgingAgent | PearlSACHedgingAgent:
    """Return meta agent."""
    agent_config = config["agent"]
    algorithm = str(run_spec["algorithm_name"])
    backend_name = "td3" if algorithm == "pearl_td3" else "sac"
    kwargs = deepcopy(dict(agent_config[backend_name]))
    for key in ("actor_hidden_dims", "critic_hidden_dims"):
        kwargs[key] = tuple(kwargs[key])
    kwargs.update(
        {
            "num_state_feature": int(agent_config["num_state_feature"]),
            "latent_dimension": int(agent_config["latent_dimension"]),
            "context_hidden_dims": tuple(agent_config["context_hidden_dims"]),
            "encoder_learning_rate": float(agent_config["encoder_learning_rate"]),
            "kl_coefficient": float(agent_config["kl_coefficient"]),
            "is_delta_residual": bool(run_spec["is_delta_residual"]),
            "max_delta_residual": float(agent_config["max_delta_residual"]),
            "device": device if device is not None else config["training"]["device"],
            "seed": int(run_spec["seed"]),
        }
    )
    cls = PearlTD3HedgingAgent if algorithm == "pearl_td3" else PearlSACHedgingAgent
    return cls(**kwargs)


def _get_dataset_from_source_config(
    source_config: Mapping[str, Any],
    *,
    label: str,
    date_period: Sequence[str],
    data_root: str | Path | None,
) -> OptionDataset:
    """Return dataset from source config."""
    values = dict(source_config)
    values.pop("date_start")
    values.pop("date_end")
    values.pop("label")
    moneyness_range = [values.pop("moneyness_lower"), values.pop("moneyness_upper")]
    if data_root is not None:
        values["data_root"] = str(data_root)
    return get_option_dataset(date_period, label=label, moneyness_range=moneyness_range, **values)


def _get_normalizer(
    train_dataset: OptionDataset, config: Mapping[str, Any]
) -> StateNormalizer | None:
    """Return normalizer."""
    preprocessing = config["state_preprocessing"]
    if not preprocessing["is_use_zscore"]:
        return None
    return get_state_normalizer(train_dataset, min_state_std=float(preprocessing["min_state_std"]))


def _get_causal_evaluation_batch_size(
    dataset: OptionDataset, *, max_num_parallel_episode: int
) -> int:
    """Choose a whole-cohort batch size that excludes expiry/inception overlap within each batch."""
    if dataset.label not in {"valid", "test"}:
        raise ValueError("Causal batch sizing applies to validation and test datasets only")
    if (
        isinstance(max_num_parallel_episode, bool)
        or int(max_num_parallel_episode) != max_num_parallel_episode
        or int(max_num_parallel_episode) <= 0
    ):
        raise ValueError("max_num_parallel_episode must be a strictly positive integer")
    manifest = dataset.episode_manifest.copy()
    manifest["t0"] = pd.to_datetime(manifest["t0"])
    manifest["cohort_id"] = pd.to_datetime(manifest["cohort_id"])
    manifest["exdate"] = pd.to_datetime(manifest["exdate"])
    grouped = manifest.groupby(["t0", "cohort_id"], sort=False)
    cohort_table = grouped.agg(
        num_episode=("episode_id", "size"),
        exdate=("exdate", "first"),
        num_unique_exdate=("exdate", "nunique"),
    ).reset_index()
    if cohort_table.empty:
        raise ValueError("The validation/test dataset contains no evaluable cohorts")
    if not cohort_table["num_unique_exdate"].eq(1).all():
        raise ValueError("Multiple expiry dates occur within a cohort")
    if not cohort_table["cohort_id"].eq(cohort_table["exdate"]).all():
        raise ValueError("cohort_id must identify the option expiry date")
    unique_counts = cohort_table["num_episode"].unique()
    if len(unique_counts) != 1:
        raise ValueError(
            "Validation/test cohorts have different episode counts; fixed cohort batches are unavailable"
        )
    cohort_size = int(unique_counts[0])
    cohort_table = cohort_table.sort_values(["t0", "cohort_id"], kind="mergesort").reset_index(
        drop=True
    )
    max_num_cohort = max(1, min(len(cohort_table), int(max_num_parallel_episode) // cohort_size))
    for num_cohort in range(max_num_cohort, 0, -1):
        is_date_safe = True
        for start in range(0, len(cohort_table), num_cohort):
            chunk = cohort_table.iloc[start : start + num_cohort]
            if pd.Timestamp(chunk["exdate"].min()) < pd.Timestamp(chunk["t0"].max()):
                is_date_safe = False
                break
        if is_date_safe:
            return cohort_size * num_cohort
    raise RuntimeError(
        "Internal error: a single-cohort evaluation batch fails the date-safety check"
    )


def _get_environment(
    dataset: OptionDataset,
    normalizer: StateNormalizer | None,
    config: Mapping[str, Any],
    run_spec: Mapping[str, Any],
    *,
    label: str,
    seed: int | None = None,
) -> HedgingEnv:
    """Return environment."""
    environment = config["environment"]
    kwargs = {
        "reward_formulation": run_spec["reward_formulation"],
        "hedge_cost_rate": environment["hedge_cost_rate"],
        "contract_multiplier": get_contract_multiplier(config),
        "currency": get_currency(config),
        "reconciliation_tolerance": environment["reconciliation_tolerance"],
        "risk_aversion_lambda": environment["risk_aversion_lambda"],
        "reward_risk_aversion_xi": environment["reward_risk_aversion_xi"],
        "training_reward_scale": run_spec["training_reward_scale"],
        "state_normalizer": normalizer,
        "is_clip_action": environment["is_clip_action"],
        "observation_dtype": np.dtype(environment["observation_dtype"]),
    }
    if label == "train":
        return HedgingEnv(
            dataset,
            **kwargs,
            is_repeated=False,
            is_record_trajectory=environment["is_record_train_trajectory"],
            seed=seed,
        )
    if label == "valid":
        return HedgingEnv(
            dataset,
            **kwargs,
            num_parallel_episode=_get_causal_evaluation_batch_size(
                dataset, max_num_parallel_episode=int(environment["num_parallel_valid_episode"])
            ),
            is_record_trajectory=environment["is_record_valid_trajectory"],
        )
    if label == "test":
        return HedgingEnv(
            dataset,
            **kwargs,
            num_parallel_episode=_get_causal_evaluation_batch_size(
                dataset, max_num_parallel_episode=int(environment["num_parallel_test_episode"])
            ),
            is_record_trajectory=environment["is_record_test_trajectory"],
        )
    raise ValueError("label must be train,valid or test")


def _get_latent_task_summary(frame: pd.DataFrame) -> dict[str, Any]:
    """Return latent task summary."""
    if frame.empty:
        return {"num_record": 0, "groups": []}
    mean_columns = sorted(
        [
            column
            for column in frame
            if column.startswith("posterior_mean_") and column != "posterior_mean_norm"
        ],
        key=lambda value: int(value.rsplit("_", 1)[1]),
    )
    groups = []
    keys = ["segment_id", "is_simulated", "num_observed_transition"]
    for values, group in frame.groupby(keys, sort=True, dropna=False):
        array = group[mean_columns].to_numpy(dtype=np.float64)
        groups.append(
            {
                **dict(zip(keys, values)),
                "num_record": int(len(group)),
                "posterior_mean": array.mean(axis=0).tolist(),
                "posterior_std": array.std(axis=0, ddof=0).tolist(),
            }
        )
    return {"num_record": int(len(frame)), "latent_columns": mean_columns, "groups": groups}


def _get_experiment_id() -> str:
    """Return experiment id."""
    return "meta_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _get_meta_code_fingerprints() -> list[dict[str, Any]]:
    """Return meta code fingerprints."""
    paths = [
        CODES_ROOT / "train_meta.py",
        CODES_ROOT / "test_meta.py",
        *sorted((CODES_ROOT / "meta_rl" / "hedging_meta_rl").glob("*.py")),
        CODES_ROOT / "rl_agents.py",
        CODES_ROOT / "rl_envs" / "hedging_env" / "environment.py",
        CODES_ROOT / "rl_envs" / "hedging_env" / "normalization.py",
        CODES_ROOT / "datasets" / "option_dataset" / "dataset.py",
        CODES_ROOT / "segmentation" / "market_segmentation" / "dataset.py",
        CODES_ROOT / "simulation" / "market_simulation" / "pipeline.py",
        CODES_ROOT / "data_source.py",
        CODES_ROOT / "data_processing" / "sx5e" / "adapter.py",
    ]
    return [
        {
            "path": str(path.relative_to(REPOSITORY_ROOT)),
            "num_bytes": path.stat().st_size,
            "sha256": get_file_sha256(path),
        }
        for path in paths
        if path.is_file()
    ]


def _resolve_path(value: str | Path) -> Path:
    """Resolve path."""
    path = Path(value)
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def _save_training_frames(
    run_dir: Path, training_result: Mapping[str, Any], *, is_save_trajectory: bool
) -> dict[str, str]:
    """Save training frames."""
    artifacts: dict[str, str] = {}
    validation_history = training_result["validation_history"].copy(deep=True)
    if "checkpoint_path" in validation_history:
        validation_history["checkpoint_path"] = None
    frames = {
        "train_episode_results": training_result["episode_results"],
        "update_history": training_result["update_history"],
        "validation_history": validation_history,
        "latent_episode_statistics": training_result["latent_episode_statistics"],
    }
    for name, frame in frames.items():
        path = run_dir / f"{name}.parquet"
        if not path.is_file():
            frame.to_parquet(path, index=False)
        artifacts[name] = path.name
    if is_save_trajectory:
        path = run_dir / "train_trajectory.parquet"
        training_result["trajectory"].to_parquet(path, index=False)
        artifacts["train_trajectory"] = path.name
    latent_path = run_dir / "latent_training_history.parquet"
    if latent_path.is_file():
        artifacts["latent_training_history"] = latent_path.name
    return artifacts


def run_meta_training(
    config: Mapping[str, Any],
    *,
    run_specs: Sequence[Mapping[str, Any]] | None = None,
    simulation_result: str | Path | None = None,
    experiment_id: str | None = None,
) -> dict[str, Any]:
    """Train regime-conditioned meta-RL on real and simulated episodes and select checkpoints using validation risk-adjusted loss."""
    _validate_meta_config(config)
    specs = [dict(value) for value in run_specs or get_meta_run_specs(config)]
    if not specs:
        raise ValueError("run_specs must not be empty")
    simulation_value = (
        simulation_result if simulation_result is not None else config["simulation"]["result"]
    )
    simulation_root = _resolve_path(config["simulation"]["result_root"])
    simulation_dir = resolve_simulation_result_directory(
        simulation_value, result_root=simulation_root
    )
    simulation_record = load_simulation_result(simulation_dir, result_root=simulation_root)
    simulation_source = validate_data_source(simulation_record.get("data_source", "spx"))
    if simulation_source != config.get("data_source", "spx"):
        raise ValueError(
            f"The meta-RL training data source differs from the Stage 2 simulation result: {config.get('data_source', 'spx')} != {simulation_source}"
        )
    configured_data_root = resolve_data_root(
        config.get("data_source", "spx"), config["data"]["data_root"]
    )
    train_dataset = load_real_dataset_from_simulation_result(
        simulation_dir, simulation_result_root=simulation_root, data_root=configured_data_root
    )
    tasks = load_meta_task_datasets(
        simulation_dir,
        simulation_result_root=simulation_root,
        real_dataset=train_dataset,
        is_exclude_cross_boundary=bool(config["simulation"]["is_exclude_cross_boundary"]),
    )
    valid_dataset = _get_dataset_from_source_config(
        simulation_record["dataset"]["config"],
        label="valid",
        date_period=config["data"]["date_periods"]["valid"],
        data_root=configured_data_root,
    )
    normalizer = _get_normalizer(train_dataset, config)
    resolved_experiment_id = experiment_id or _get_experiment_id()
    get_safe_path_component(resolved_experiment_id, "experiment_id")
    output_root = _resolve_path(config["training"]["output_root"])
    experiment_dir = output_root / resolved_experiment_id
    if experiment_dir.exists():
        raise FileExistsError(f"The meta-RL training directory already exists: {experiment_dir}")
    experiment_dir.mkdir(parents=True)
    _write_json_atomic(experiment_dir / "requested_config.json", config)
    if normalizer is not None:
        normalizer.save(experiment_dir / "state_normalizer.json")
    dataset_directory = experiment_dir / "datasets"
    dataset_directory.mkdir()
    train_dataset.episode_manifest.to_parquet(
        dataset_directory / "train_episode_manifest.parquet", index=False
    )
    valid_dataset.episode_manifest.to_parquet(
        dataset_directory / "valid_episode_manifest.parquet", index=False
    )
    task_membership_frames = []
    for task in tasks:
        for is_simulated, dataset in ((False, task.real_dataset), (True, task.simulated_dataset)):
            frame = dataset.episode_manifest.copy(deep=True)
            frame.insert(0, "meta_segment_id", task.segment_id)
            frame.insert(1, "meta_is_simulated", is_simulated)
            task_membership_frames.append(frame)
    pd.concat(task_membership_frames, ignore_index=True).to_parquet(
        dataset_directory / "task_episode_manifest.parquet", index=False
    )
    shared_record = {
        "data_source": config.get("data_source", "spx"),
        "experiment_id": resolved_experiment_id,
        "simulation_result": str(simulation_dir),
        "simulation_experiment_id": simulation_record["experiment_id"],
        "train_dataset": get_dataset_record(train_dataset),
        "valid_dataset": get_dataset_record(valid_dataset),
        "state_normalizer": normalizer.get_dict() if normalizer else None,
        "tasks": [task.get_record() for task in tasks],
        "artifacts": {
            "train_episode_manifest": "datasets/train_episode_manifest.parquet",
            "valid_episode_manifest": "datasets/valid_episode_manifest.parquet",
            "task_episode_manifest": "datasets/task_episode_manifest.parquet",
            "state_normalizer": "state_normalizer.json" if normalizer is not None else None,
        },
    }
    _write_json_atomic(experiment_dir / "shared_data.json", shared_record)
    started_at = get_datetime_record()
    experiment_record: dict[str, Any] = {
        "schema_version": META_CONFIG_SCHEMA_VERSION,
        "status": "running",
        "data_source": config.get("data_source", "spx"),
        "experiment_id": resolved_experiment_id,
        "started_at": started_at,
        "num_run": len(specs),
        "simulation_result": str(simulation_dir),
        "system": get_system_info(),
        "code_fingerprints": _get_meta_code_fingerprints(),
    }
    _write_json_atomic(experiment_dir / "experiment.json", experiment_record)
    summaries: list[dict[str, Any]] = []
    for num_run, run_spec in enumerate(specs, start=1):
        run_id = str(run_spec["run_id"])
        get_safe_path_component(run_id, "run_id")
        run_dir = experiment_dir / run_id
        run_dir.mkdir()
        checkpoints_dir = run_dir / "checkpoints"
        checkpoints_dir.mkdir()
        run_started = get_datetime_record()
        running = {
            "schema_version": META_CONFIG_SCHEMA_VERSION,
            "status": "running",
            "data_source": config.get("data_source", "spx"),
            "experiment_id": resolved_experiment_id,
            "run_id": run_id,
            "run_started_at": run_started,
            "run_index": num_run,
            "num_run": len(specs),
            "resolved_run": run_spec,
            "requested_config": config,
            "simulation_result": str(simulation_dir),
            "state_normalizer": normalizer.get_dict() if normalizer else None,
        }
        _write_json_atomic(run_dir / "run_config.json", running)
        print(f"\n[Experiment {num_run}/{len(specs)}] {run_id}", flush=True)
        started = time.perf_counter()
        try:
            set_global_seed(
                int(run_spec["seed"]),
                is_torch_deterministic=bool(config["training"]["is_torch_deterministic"]),
            )
            agent = get_meta_agent(config, run_spec)

            def train_env_factory(dataset: OptionDataset, seed: int) -> HedgingEnv:
                """Train env factory."""
                return _get_environment(
                    dataset, normalizer, config, run_spec, label="train", seed=seed
                )

            def valid_env_factory() -> HedgingEnv:
                """Valid env factory."""
                return _get_environment(valid_dataset, normalizer, config, run_spec, label="valid")

            result = train_meta_hedging_agent(
                agent,
                tasks,
                get_train_env=train_env_factory,
                get_valid_env=valid_env_factory,
                num_batch=int(config["training"]["num_batch"]),
                num_task_batch=int(config["training"]["num_task_batch"]),
                num_recent_context_episodes=int(config["training"]["num_recent_context_episodes"]),
                simulated_batch_fraction=float(config["training"]["simulated_batch_fraction"]),
                warmup_fraction=float(config["training"]["warmup_fraction"]),
                num_update_per_step=int(config["training"]["num_update_per_step"]),
                num_valid_episodes=int(config["training"]["num_valid_episodes"]),
                num_latent_log_interval=int(config["training"]["num_latent_log_interval"]),
                checkpoint_path=run_dir / "agent.pt",
                checkpoint_dir=checkpoints_dir,
                update_history_path=run_dir / "update_history.parquet",
                latent_history_path=run_dir / "latent_training_history.parquet",
                seed=int(run_spec["seed"]),
                is_show_progress=bool(config["training"]["is_show_progress"]),
                is_print_summary=bool(config["training"]["is_print_summary"]),
            )
            artifacts = _save_training_frames(
                run_dir,
                result,
                is_save_trajectory=bool(config["training"]["is_save_train_trajectory"]),
            )
            _write_json_atomic(
                run_dir / "episode_usage_audit.json", {"tasks": result["episode_usage_audit"]}
            )
            _write_json_atomic(
                run_dir / "latent_task_summary.json",
                _get_latent_task_summary(result["latent_episode_statistics"]),
            )
            artifacts.update(
                {
                    "model": "agent.pt",
                    "episode_usage_audit": "episode_usage_audit.json",
                    "latent_task_summary": "latent_task_summary.json",
                }
            )
            model_path = run_dir / "agent.pt"
            model_sha256 = get_file_sha256(model_path)
            checkpoint_retention = prune_run_checkpoints(run_dir, checkpoints_dir, model_path)
            artifacts["checkpoint_retention"] = checkpoint_retention
            completed = {
                **running,
                "status": "completed",
                "run_completed_at": get_datetime_record(),
                "num_elapsed_second": time.perf_counter() - started,
                "agent_config": agent.get_config(),
                "train_metrics": result["metrics"],
                "best_validation": result["best_validation"],
                "num_train_episode": result["num_episode"],
                "num_train_step": result["num_total_step"],
                "num_meta_update": result["num_meta_update"],
                "model_role": "validation_best",
                "model_sha256": model_sha256,
                "train_environments": result["task_environment_configs"],
                "artifacts": artifacts,
            }
            _write_json_atomic(run_dir / "run_config.json", completed)
            summaries.append(
                {
                    "run_id": run_id,
                    "status": "completed",
                    "train_metrics": result["metrics"],
                    "best_validation": result["best_validation"],
                }
            )
        except Exception as exc:
            failed = {
                **running,
                "status": "failed",
                "run_failed_at": get_datetime_record(),
                "num_elapsed_second": time.perf_counter() - started,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }
            _write_json_atomic(run_dir / "run_config.json", failed)
            summaries.append({"run_id": run_id, "status": "failed", "error": str(exc)})
            if not config["training"]["is_continue_on_error"]:
                experiment_record.update(
                    {"status": "failed", "failed_at": get_datetime_record(), "runs": summaries}
                )
                _write_json_atomic(experiment_dir / "experiment.json", experiment_record)
                raise
    completed_count = sum((record["status"] == "completed" for record in summaries))
    experiment_record.update(
        {
            "status": "completed" if completed_count == len(summaries) else "partial",
            "completed_at": get_datetime_record(),
            "num_completed_run": completed_count,
            "runs": summaries,
        }
    )
    _write_json_atomic(experiment_dir / "experiment.json", experiment_record)
    print(f"\n[Training experiment completed] {experiment_dir}", flush=True)
    return {"experiment_dir": experiment_dir, "result": experiment_record}


def get_argument_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=None, help="JSON configuration override file"
    )
    parser.add_argument(
        "--simulation-result",
        type=Path,
        default=None,
        help="Stage 2 result directory or sim_result.json; defaults to the latest completed result",
    )
    parser.add_argument(
        "--experiment-id",
        type=str,
        default=None,
        help="Optional explicit experiment directory name",
    )
    parser.add_argument(
        "--algorithm",
        choices=VALID_META_ALGORITHMS,
        action="append",
        help="Run only the specified algorithm; may be repeated",
    )
    parser.add_argument(
        "--run-id", action="append", help="Run only the specified complete run_id; may be repeated"
    )
    parser.add_argument(
        "--data-source",
        choices=VALID_DATA_SOURCES,
        default=None,
        help="data source; sx5e triggers canonical preprocessing",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse command-line arguments, run the requested operation, and report failures."""
    parser = get_argument_parser()
    args = parser.parse_args(argv)
    config = get_meta_config(args.config)
    config["data_source"] = validate_data_source(
        args.data_source or config.get("data_source", "spx")
    )
    _validate_meta_config(config)
    print(f"[Data source] data_source={config['data_source']}", flush=True)
    specs = get_meta_run_specs(config)
    if args.algorithm:
        selected = set(args.algorithm)
        specs = [item for item in specs if item["algorithm_name"] in selected]
    if args.run_id:
        selected_ids = set(args.run_id)
        specs = [item for item in specs if item["run_id"] in selected_ids]
        missing = selected_ids - {item["run_id"] for item in specs}
        if missing:
            raise ValueError(f"Unknown or disabled run_id: {sorted(missing)}")
    if not specs:
        raise ValueError("No experiments remain after CLI filtering")
    print(json.dumps(specs, ensure_ascii=False, indent=2), flush=True)
    run_meta_training(
        config,
        run_specs=specs,
        simulation_result=args.simulation_result,
        experiment_id=args.experiment_id,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
