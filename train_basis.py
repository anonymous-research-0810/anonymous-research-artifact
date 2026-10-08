"""Train basis for the paper option-hedging pipeline."""

from __future__ import annotations
import argparse
import hashlib
import json
import os
import platform
import random
import shutil
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
    CODES_ROOT / "rl_envs",
    CODES_ROOT / "datasets",
    CODES_ROOT / "data_processing",
):
    if str(_source_directory) not in sys.path:
        sys.path.insert(0, str(_source_directory))
from rl_agents import SACHedgingAgent, TD3HedgingAgent
from rl_utils import train_hedging_agent
from option_dataset import OptionDataset, get_option_dataset
from hedging_env import (
    DEFAULT_TRAINING_REWARD_SCALES,
    HedgingEnv,
    StateNormalizer,
    get_state_normalizer,
)
from data_source import VALID_DATA_SOURCES, resolve_data_root, validate_data_source

BASIS_CONFIG_SCHEMA_VERSION = 4
VALID_ALGORITHM_NAMES = ("sac", "td3")
VALID_REWARD_FORMULATIONS = ("shaped_accounting", "cash_flow")


def get_default_basis_config() -> dict[str, Any]:
    """Return the paper baseline configuration for two rewards, two backbones, and two action parameterizations."""
    return {
        "schema_version": BASIS_CONFIG_SCHEMA_VERSION,
        "experiment_name": "basis",
        "data_source": "spx",
        "data": {
            "data_root": "data",
            "date_periods": {
                "train": ["2016-01-04", "2022-12-30"],
                "valid": ["2023-01-03", "2023-12-29"],
                "test": ["2024-01-02", "2025-08-29"],
            },
            "cp_flag": "C",
            "symbol_start": "SPXW",
            "expiry_weekday": None,
            "num_interval": 20,
            "num_moneyness": 3,
            "moneyness_range": [0.95, 1.05],
            "max_relative_spread": 0.1,
            "min_open_interest": 0.0,
            "min_iv": 1e-06,
            "max_iv": 5.0,
            "iv_solver_tolerance": 1e-08,
            "num_iv_solver_iterations": 100,
            "num_iv_surface_points": 2,
            "max_iv_extrapolation_log_moneyness": 0.1,
            "num_iv_forward_fill_steps": 1,
            "is_use_option_price_for_iv": True,
            "is_use_same_strike_opposite_iv": True,
            "is_use_surface_iv": True,
            "is_use_past_iv": True,
            "is_require_balanced_cohort": True,
            "is_allow_empty": False,
            "is_require_environment_ready": True,
            "num_scan_batch_size": 131072,
        },
        "state_preprocessing": {"is_use_zscore": True, "min_state_std": 1e-08},
        "environment": {
            "reward_formulations": ["shaped_accounting", "cash_flow"],
            "hedge_cost_rate": 0.001,
            "contract_multiplier": 100.0,
            "contract_multiplier_by_data_source": {"spx": 100.0, "sx5e": 10.0},
            "currency_by_data_source": {"spx": "USD", "sx5e": "EUR"},
            "reconciliation_tolerance": 1e-08,
            "risk_aversion_lambda": 1.5,
            "reward_risk_aversion_xi": 1.5,
            "training_reward_scales": dict(DEFAULT_TRAINING_REWARD_SCALES),
            "num_parallel_valid_episode": 256,
            "num_parallel_test_episode": 256,
            "is_clip_action": False,
            "is_record_train_trajectory": True,
            "is_record_valid_trajectory": False,
            "is_record_test_trajectory": True,
            "observation_dtype": "float32",
        },
        "agent": {
            "algorithm_names": ["sac", "td3"],
            "is_delta_residual_values": [True, False],
            "max_delta_residual": 0.1,
            "num_state_feature": 7,
            "td3": {
                "actor_hidden_dims": [128, 128],
                "critic_hidden_dims": [128, 128],
                "activation_name": "relu",
                "actor_output_gain": 0.01,
                "critic_output_gain": 0.01,
                "actor_learning_rate": 0.0003,
                "critic_learning_rate": 0.0003,
                "gamma": 1.0,
                "tau": 0.005,
                "exploration_noise_std": 0.1,
                "target_policy_noise_std": 0.2,
                "target_noise_clip": 0.5,
                "num_policy_delay": 2,
                "max_gradient_norm": 1.0,
            },
            "sac": {
                "actor_hidden_dims": [128, 128],
                "critic_hidden_dims": [128, 128],
                "activation_name": "relu",
                "actor_output_gain": 0.01,
                "critic_output_gain": 0.01,
                "actor_learning_rate": 0.0003,
                "critic_learning_rate": 0.0003,
                "alpha_learning_rate": 0.0003,
                "gamma": 1.0,
                "tau": 0.005,
                "initial_alpha": 0.2,
                "target_entropy": None,
                "is_automatic_entropy_tuning": True,
                "max_gradient_norm": 1.0,
            },
        },
        "training": {
            "seeds": list(range(10)),
            "is_repeated": False,
            "num_training_episode": None,
            "num_valid_episodes": 100,
            "num_replay_capacity": 100000,
            "num_batch": 128,
            "warmup_fraction": 0.1,
            "num_update_per_step": 1,
            "is_show_progress": True,
            "is_print_summary": True,
            "is_save_train_trajectory": True,
            "is_continue_on_error": False,
            "is_torch_deterministic": False,
            "device": None,
            "output_root": "train_results/basis",
        },
        "testing": {
            "output_root": "test_results/basis",
            "is_print_summary": True,
            "is_save_test_trajectory": True,
            "device": None,
        },
    }


def _merge_config_strict(
    default: Mapping[str, Any], override: Mapping[str, Any], *, path: str = "config"
) -> dict[str, Any]:
    """Recursively merge overrides and reject unknown fields."""
    unknown = set(override) - set(default)
    if unknown:
        raise ValueError(f"{path} contains unknown fields: {sorted(unknown)}")
    result = deepcopy(dict(default))
    for key, value in override.items():
        if isinstance(default[key], Mapping):
            if not isinstance(value, Mapping):
                raise ValueError(f"{path}.{key} must beobject")
            result[key] = _merge_config_strict(default[key], value, path=f"{path}.{key}")
        else:
            result[key] = deepcopy(value)
    return result


def get_basis_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Return basis config."""
    config = get_default_basis_config()
    if config_path is not None:
        path = Path(config_path)
        override = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(override, Mapping):
            raise ValueError("The configuration root must be a JSON object")
        config = _merge_config_strict(config, override)
    _validate_basis_config(config)
    return config


def _validate_basis_config(config: Mapping[str, Any]) -> None:
    """Validate basis config."""
    if config["schema_version"] != BASIS_CONFIG_SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {BASIS_CONFIG_SCHEMA_VERSION}")
    validate_data_source(config.get("data_source", "spx"))
    environment = config["environment"]
    agent = config["agent"]
    training = config["training"]
    testing = config["testing"]
    preprocessing = config["state_preprocessing"]
    date_periods = config["data"]["date_periods"]
    parsed_periods: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {}
    for label in ("train", "valid", "test"):
        period = date_periods[label]
        if isinstance(period, (str, bytes)) or len(period) != 2:
            raise ValueError(f"data.date_periods.{label} must contain a start and end date")
        date_start = pd.Timestamp(period[0]).normalize()
        date_end = pd.Timestamp(period[1]).normalize()
        if pd.isna(date_start) or pd.isna(date_end) or date_start > date_end:
            raise ValueError(f"data.date_periods.{label} has an invalid date range")
        parsed_periods[label] = (date_start, date_end)
    if not (
        parsed_periods["train"][1] < parsed_periods["valid"][0]
        and parsed_periods["valid"][1] < parsed_periods["test"][0]
    ):
        raise ValueError(
            "Training, validation, and test periods must be chronological and disjoint"
        )
    raw_rewards = tuple((str(value) for value in environment["reward_formulations"]))
    raw_algorithms = tuple((str(value) for value in agent["algorithm_names"]))
    if any((value != value.lower() for value in raw_rewards)):
        raise ValueError("reward_formulations must use canonical lowercase names")
    if any((value != value.lower() for value in raw_algorithms)):
        raise ValueError("algorithm_names must use canonical lowercase names")
    rewards = tuple((value.lower() for value in raw_rewards))
    algorithms = tuple((value.lower() for value in raw_algorithms))
    if not rewards or not set(rewards).issubset(VALID_REWARD_FORMULATIONS):
        raise ValueError(
            f"reward_formulations must be {VALID_REWARD_FORMULATIONS} a nonempty subset of"
        )
    if len(rewards) != len(set(rewards)):
        raise ValueError("reward_formulations must not contain duplicates")
    training_reward_scales = environment["training_reward_scales"]
    if not isinstance(training_reward_scales, Mapping):
        raise ValueError("environment.training_reward_scales must beobject")
    if set(training_reward_scales) != set(VALID_REWARD_FORMULATIONS):
        raise ValueError(
            f"environment.training_reward_scales must contain exactly {VALID_REWARD_FORMULATIONS}"
        )
    for reward_formulation, value in training_reward_scales.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or (not np.isfinite(float(value)))
            or (float(value) <= 0.0)
        ):
            raise ValueError(
                f"environment.training_reward_scales.{reward_formulation} must be finite and positive"
            )
    if not algorithms or not set(algorithms).issubset(VALID_ALGORITHM_NAMES):
        raise ValueError(f"algorithm_names must be {VALID_ALGORITHM_NAMES} a nonempty subset of")
    if len(algorithms) != len(set(algorithms)):
        raise ValueError("algorithm_names must not contain duplicates")
    residual_values = agent["is_delta_residual_values"]
    if not residual_values or any((not isinstance(value, bool) for value in residual_values)):
        raise ValueError("is_delta_residual_values must be a nonempty boolean sequence")
    if len(residual_values) != len(set(residual_values)):
        raise ValueError("is_delta_residual_values must not contain duplicates")
    max_delta_residual = float(agent["max_delta_residual"])
    if not np.isfinite(max_delta_residual) or max_delta_residual <= 0.0:
        raise ValueError("max_delta_residual must be finite and positive")
    if agent["num_state_feature"] != 7:
        raise ValueError(
            "The environment has seven state features; agent.num_state_feature must be 7"
        )
    seeds = training["seeds"]
    if not seeds or any(
        (
            isinstance(seed, bool) or int(seed) != seed or (not 0 <= int(seed) <= 2**32 - 1)
            for seed in seeds
        )
    ):
        raise ValueError("training.seeds must be a nonempty integer sequence in [0,2^32-1]")
    if len(seeds) != len(set(seeds)):
        raise ValueError("training.seeds must not contain duplicates")
    if not isinstance(training["is_repeated"], bool):
        raise ValueError("training.is_repeated must be a boolean")
    num_training_episode = training["num_training_episode"]
    if training["is_repeated"]:
        if (
            isinstance(num_training_episode, bool)
            or not isinstance(num_training_episode, int)
            or num_training_episode <= 0
        ):
            raise ValueError(
                "num_training_episode must be a strictly positive integer when is_repeated=True"
            )
    elif num_training_episode is not None:
        raise ValueError("num_training_episode must be None when is_repeated=False")
    positive_integer_paths = (
        (training, "num_valid_episodes", "training.num_valid_episodes"),
        (training, "num_replay_capacity", "training.num_replay_capacity"),
        (training, "num_batch", "training.num_batch"),
        (training, "num_update_per_step", "training.num_update_per_step"),
        (environment, "num_parallel_valid_episode", "environment.num_parallel_valid_episode"),
        (environment, "num_parallel_test_episode", "environment.num_parallel_test_episode"),
    )
    for source, key, name in positive_integer_paths:
        value = source[key]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a strictly positive integer")
    warmup_fraction = training["warmup_fraction"]
    if (
        isinstance(warmup_fraction, bool)
        or not isinstance(warmup_fraction, (int, float))
        or (not np.isfinite(float(warmup_fraction)))
        or (not 0.0 <= float(warmup_fraction) < 1.0)
    ):
        raise ValueError("training.warmup_fraction must be a finite number in [0,1)")
    for key in (
        "hedge_cost_rate",
        "contract_multiplier",
        "reconciliation_tolerance",
        "risk_aversion_lambda",
        "reward_risk_aversion_xi",
    ):
        value = environment[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or (not np.isfinite(float(value)))
            or (float(value) < 0.0)
        ):
            raise ValueError(f"environment.{key} must be finite and nonnegative")
    for key in ("contract_multiplier", "reconciliation_tolerance"):
        if float(environment[key]) <= 0.0:
            raise ValueError(f"environment.{key} must be strictly positive")
    multiplier_by_source = environment.get("contract_multiplier_by_data_source", {})
    if not isinstance(multiplier_by_source, Mapping):
        raise ValueError("environment.contract_multiplier_by_data_source must beobject")
    for source in VALID_DATA_SOURCES:
        if source in multiplier_by_source:
            value = multiplier_by_source[source]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or (not np.isfinite(float(value)))
                or (float(value) <= 0.0)
            ):
                raise ValueError(
                    f"environment.contract_multiplier_by_data_source.{source} must be a positive number"
                )
    currency_by_source = environment.get("currency_by_data_source", {})
    if not isinstance(currency_by_source, Mapping):
        raise ValueError("environment.currency_by_data_source must beobject")
    for source in VALID_DATA_SOURCES:
        if source in currency_by_source:
            value = str(currency_by_source[source]).upper()
            if len(value) != 3 or not value.isalpha():
                raise ValueError(
                    f"environment.currency_by_data_source.{source} must be a three-letter currency code"
                )
    boolean_entries = (
        (preprocessing, "is_use_zscore", "state_preprocessing.is_use_zscore"),
        (environment, "is_clip_action", "environment.is_clip_action"),
        (environment, "is_record_train_trajectory", "environment.is_record_train_trajectory"),
        (environment, "is_record_valid_trajectory", "environment.is_record_valid_trajectory"),
        (environment, "is_record_test_trajectory", "environment.is_record_test_trajectory"),
        (training, "is_show_progress", "training.is_show_progress"),
        (training, "is_print_summary", "training.is_print_summary"),
        (training, "is_save_train_trajectory", "training.is_save_train_trajectory"),
        (training, "is_continue_on_error", "training.is_continue_on_error"),
        (training, "is_torch_deterministic", "training.is_torch_deterministic"),
        (testing, "is_print_summary", "testing.is_print_summary"),
        (testing, "is_save_test_trajectory", "testing.is_save_test_trajectory"),
    )
    for source, key, name in boolean_entries:
        if not isinstance(source[key], bool):
            raise ValueError(f"{name} must be a boolean")
    if training["is_save_train_trajectory"] and (not environment["is_record_train_trajectory"]):
        raise ValueError(
            "Saving training trajectories requires environment.is_record_train_trajectory"
        )
    if testing["is_save_test_trajectory"] and (not environment["is_record_test_trajectory"]):
        raise ValueError("Saving test trajectories requires environment.is_record_test_trajectory")
    if config["data"]["cp_flag"] != "C":
        raise ValueError("The hedging framework supports short calls only; cp_flag must be 'C'")
    if config["data"]["symbol_start"] != "SPXW":
        raise ValueError("The SPX data interface requires the SPXW product")
    get_safe_path_component(str(config["experiment_name"]), "experiment_name")


def get_safe_path_component(value: str, name: str) -> str:
    """Return safe path component."""
    text = str(value)
    if (
        not text
        or text in (".", "..")
        or Path(text).name != text
        or ("/" in text)
        or ("\\" in text)
    ):
        raise ValueError(f"{name} must be a nonempty name without path separators")
    return text


def get_basis_run_specs(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Expand rewards, algorithms, action parameterizations, and seeds into deterministic run specifications."""
    specs: list[dict[str, Any]] = []
    for reward_formulation in config["environment"]["reward_formulations"]:
        for algorithm_name in config["agent"]["algorithm_names"]:
            for is_delta_residual in config["agent"]["is_delta_residual_values"]:
                for seed in config["training"]["seeds"]:
                    action_name = "delta_residual" if is_delta_residual else "direct_action"
                    specs.append(
                        {
                            "run_id": f"{reward_formulation}__{algorithm_name}__{action_name}__seed_{int(seed)}",
                            "reward_formulation": str(reward_formulation),
                            "training_reward_scale": float(
                                config["environment"]["training_reward_scales"][reward_formulation]
                            ),
                            "algorithm_name": str(algorithm_name),
                            "is_delta_residual": bool(is_delta_residual),
                            "seed": int(seed),
                        }
                    )
    return specs


def _validate_run_spec(config: Mapping[str, Any], run_spec: Mapping[str, Any]) -> None:
    """Validate run spec."""
    required = {
        "run_id",
        "reward_formulation",
        "training_reward_scale",
        "algorithm_name",
        "is_delta_residual",
        "seed",
    }
    if set(run_spec) != required:
        raise ValueError(
            f"run_spec fields must be exactly {sorted(required)}; received {sorted(run_spec)}"
        )
    if run_spec["reward_formulation"] not in config["environment"]["reward_formulations"]:
        raise ValueError("run_spec reward_formulation is outside the requested experiment matrix")
    expected_training_reward_scale = float(
        config["environment"]["training_reward_scales"][run_spec["reward_formulation"]]
    )
    if run_spec["training_reward_scale"] != expected_training_reward_scale:
        raise ValueError("run_spec training_reward_scale differs from its reward formulation")
    if run_spec["algorithm_name"] not in config["agent"]["algorithm_names"]:
        raise ValueError("run_spec algorithm_name is outside the requested experiment matrix")
    if run_spec["is_delta_residual"] not in config["agent"]["is_delta_residual_values"]:
        raise ValueError("run_spec is_delta_residual is outside the requested experiment matrix")
    if run_spec["seed"] not in config["training"]["seeds"]:
        raise ValueError("run_spec seed is outside the requested experiment matrix")
    action_name = "delta_residual" if run_spec["is_delta_residual"] else "direct_action"
    expected_run_id = f"{run_spec['reward_formulation']}__{run_spec['algorithm_name']}__{action_name}__seed_{int(run_spec['seed'])}"
    if run_spec["run_id"] != expected_run_id:
        raise ValueError(
            f"run_spec.run_id must be {expected_run_id!r}; received {run_spec['run_id']!r}"
        )
    get_safe_path_component(str(run_spec["run_id"]), "run_id")


def get_dataset_kwargs(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return dataset kwargs."""
    data = deepcopy(dict(config["data"]))
    data.pop("date_periods")
    data["data_root"] = resolve_data_root(config.get("data_source", "spx"), data.get("data_root"))
    return data


def get_contract_multiplier(config: Mapping[str, Any]) -> float:
    """Return contract multiplier."""
    source = validate_data_source(config.get("data_source", "spx"))
    environment = config["environment"]
    mapping = environment.get("contract_multiplier_by_data_source", {})
    if isinstance(mapping, Mapping) and source in mapping:
        return float(mapping[source])
    return float(environment["contract_multiplier"])


def get_currency(config: Mapping[str, Any]) -> str:
    """Return currency."""
    source = validate_data_source(config.get("data_source", "spx"))
    mapping = config["environment"].get("currency_by_data_source", {})
    if isinstance(mapping, Mapping) and source in mapping:
        return str(mapping[source]).upper()
    return "USD"


def get_basis_datasets(
    config: Mapping[str, Any], *, labels: Sequence[str] = ("train", "valid")
) -> dict[str, OptionDataset]:
    """Return basis datasets."""
    kwargs = get_dataset_kwargs(config)
    datasets: dict[str, OptionDataset] = {}
    for label in labels:
        if label not in ("train", "valid", "test"):
            raise ValueError("labels may contain only train, valid, and test")
        datasets[label] = get_option_dataset(
            config["data"]["date_periods"][label], label=label, **kwargs
        )
    return datasets


def get_basis_normalizer(
    train_dataset: OptionDataset, config: Mapping[str, Any]
) -> StateNormalizer | None:
    """Return basis normalizer."""
    preprocessing = config["state_preprocessing"]
    if not preprocessing["is_use_zscore"]:
        return None
    return get_state_normalizer(train_dataset, min_state_std=float(preprocessing["min_state_std"]))


def get_train_env(
    train_dataset: OptionDataset,
    normalizer: StateNormalizer | None,
    config: Mapping[str, Any],
    run_spec: Mapping[str, Any],
) -> HedgingEnv:
    """Return train env."""
    environment = config["environment"]
    training = config["training"]
    return HedgingEnv(
        train_dataset,
        reward_formulation=run_spec["reward_formulation"],
        hedge_cost_rate=environment["hedge_cost_rate"],
        contract_multiplier=get_contract_multiplier(config),
        currency=get_currency(config),
        reconciliation_tolerance=environment["reconciliation_tolerance"],
        risk_aversion_lambda=environment["risk_aversion_lambda"],
        reward_risk_aversion_xi=environment["reward_risk_aversion_xi"],
        training_reward_scale=run_spec["training_reward_scale"],
        state_normalizer=normalizer,
        is_repeated=training["is_repeated"],
        num_training_episode=training["num_training_episode"],
        is_clip_action=environment["is_clip_action"],
        is_record_trajectory=environment["is_record_train_trajectory"],
        observation_dtype=np.dtype(environment["observation_dtype"]),
        seed=run_spec["seed"],
    )


def get_valid_env(
    valid_dataset: OptionDataset,
    normalizer: StateNormalizer | None,
    config: Mapping[str, Any],
    run_spec: Mapping[str, Any],
) -> HedgingEnv:
    """Return valid env."""
    environment = config["environment"]
    return HedgingEnv(
        valid_dataset,
        reward_formulation=run_spec["reward_formulation"],
        hedge_cost_rate=environment["hedge_cost_rate"],
        contract_multiplier=get_contract_multiplier(config),
        currency=get_currency(config),
        reconciliation_tolerance=environment["reconciliation_tolerance"],
        risk_aversion_lambda=environment["risk_aversion_lambda"],
        reward_risk_aversion_xi=environment["reward_risk_aversion_xi"],
        training_reward_scale=run_spec["training_reward_scale"],
        state_normalizer=normalizer,
        num_parallel_episode=environment["num_parallel_valid_episode"],
        is_clip_action=environment["is_clip_action"],
        is_record_trajectory=environment["is_record_valid_trajectory"],
        observation_dtype=np.dtype(environment["observation_dtype"]),
    )


def get_hedging_agent(
    config: Mapping[str, Any],
    run_spec: Mapping[str, Any],
    *,
    device: str | torch.device | None = None,
) -> SACHedgingAgent | TD3HedgingAgent:
    """Return hedging agent."""
    agent_config = config["agent"]
    algorithm_name = str(run_spec["algorithm_name"])
    kwargs = deepcopy(dict(agent_config[algorithm_name]))
    for key in ("actor_hidden_dims", "critic_hidden_dims", "value_hidden_dims"):
        if key in kwargs:
            kwargs[key] = tuple(kwargs[key])
    kwargs.update(
        {
            "num_state_feature": agent_config["num_state_feature"],
            "is_delta_residual": run_spec["is_delta_residual"],
            "max_delta_residual": agent_config["max_delta_residual"],
            "device": device if device is not None else config["training"]["device"],
            "seed": run_spec["seed"],
        }
    )
    if algorithm_name == "sac":
        return SACHedgingAgent(**kwargs)
    if algorithm_name == "td3":
        return TD3HedgingAgent(**kwargs)
    raise ValueError(f"Unknown algorithm_name: {algorithm_name}")


def set_global_seed(seed: int, *, is_torch_deterministic: bool) -> None:
    """Seed Python, NumPy, and PyTorch, optionally enabling deterministic operators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(bool(is_torch_deterministic), warn_only=True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = not bool(is_torch_deterministic)
        torch.backends.cudnn.deterministic = bool(is_torch_deterministic)


def get_datetime_record() -> dict[str, str]:
    """Return the current UTC timestamp."""
    now = datetime.now(timezone.utc)
    return {"utc": now.isoformat(timespec="seconds")}


def get_json_value(value: Any) -> Any:
    """Return JSON value."""
    if isinstance(value, Mapping):
        return {str(key): get_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [get_json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, np.ndarray):
        return [get_json_value(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return get_json_value(value.item())
    if isinstance(value, float) and (not np.isfinite(value)):
        return None
    return value


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Write JSON atomic."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(
            get_json_value(payload), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
        ),
        encoding="utf-8",
    )
    os.replace(temporary_path, path)


def get_file_sha256(path: str | Path) -> str:
    """Compute a file SHA-256 digest using bounded-memory reads."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def prune_run_checkpoints(
    run_dir: str | Path, checkpoint_dir: str | Path, selected_model_path: str | Path
) -> dict[str, Any]:
    """Remove only verified checkpoint files after the selected model has been saved in the run directory."""
    run_path = Path(run_dir).resolve()
    checkpoint_path = Path(checkpoint_dir).resolve()
    model_path = Path(selected_model_path).resolve()
    if checkpoint_path.parent != run_path or checkpoint_path.name != "checkpoints":
        raise ValueError(
            "checkpoint_dir must be the checkpoints directory directly below the current run"
        )
    if model_path.parent != run_path or not model_path.is_file():
        raise FileNotFoundError(
            "A selected model must exist in the run directory before checkpoints are removed"
        )
    if not checkpoint_path.exists():
        return {
            "policy": "validation_best_only",
            "selected_model": model_path.name,
            "checkpoint_directory_removed": False,
            "num_removed_model_file": 0,
            "num_removed_byte": 0,
        }
    if not checkpoint_path.is_dir():
        raise ValueError("The checkpoints path exists but is not a directory")
    entries = list(checkpoint_path.iterdir())
    unexpected = [
        entry.name for entry in entries if not entry.is_file() or entry.suffix.lower() != ".pt"
    ]
    if unexpected:
        raise RuntimeError(
            f"Refusing to remove a directory containing files other than checkpoints: {sorted(unexpected)}"
        )
    num_removed_byte = sum((entry.stat().st_size for entry in entries))
    shutil.rmtree(checkpoint_path)
    return {
        "policy": "validation_best_only",
        "selected_model": model_path.name,
        "checkpoint_directory_removed": True,
        "num_removed_model_file": len(entries),
        "num_removed_byte": num_removed_byte,
    }


def get_code_fingerprints() -> list[dict[str, Any]]:
    """Return code fingerprints."""
    paths = [
        CODES_ROOT / "train_basis.py",
        CODES_ROOT / "test_basis.py",
        CODES_ROOT / "test_delta.py",
        CODES_ROOT / "rl_agents.py",
        CODES_ROOT / "rl_utils.py",
        CODES_ROOT / "rl_envs" / "hedging_env" / "environment.py",
        CODES_ROOT / "rl_envs" / "hedging_env" / "constants.py",
        CODES_ROOT / "rl_envs" / "hedging_env" / "normalization.py",
        CODES_ROOT / "datasets" / "option_dataset" / "builder.py",
        CODES_ROOT / "datasets" / "option_dataset" / "dataset.py",
        CODES_ROOT / "data_source.py",
        CODES_ROOT / "data_processing" / "sx5e" / "adapter.py",
    ]
    results = []
    for path in paths:
        if path.is_file():
            results.append(
                {
                    "path": str(path.relative_to(REPOSITORY_ROOT)),
                    "num_bytes": path.stat().st_size,
                    "sha256": get_file_sha256(path),
                }
            )
    return results


def get_system_info() -> dict[str, Any]:
    """Record software versions and compute-device availability without account, hostname, or repository identity."""
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "torch": torch.__version__,
        "cuda_is_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
    }


def get_dataset_record(dataset: OptionDataset) -> dict[str, Any]:
    """Return dataset record."""
    episode_ids = "\n".join(dataset.get_episode_ids()).encode("utf-8")
    return {
        "label": dataset.label,
        "config": dataset.config.get_dict(),
        "num_episode": dataset.num_episode,
        "num_step": dataset.num_step,
        "is_environment_ready": dataset.is_environment_ready,
        "episode_ids_sha256": hashlib.sha256(episode_ids).hexdigest(),
        "dataset_build_report": dataset.dataset_build_report,
    }


def _save_experiment_dataset_records(
    experiment_dir: Path, datasets: Mapping[str, OptionDataset], normalizer: StateNormalizer | None
) -> dict[str, Any]:
    """Save experiment dataset records."""
    dataset_dir = experiment_dir / "datasets"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, Any] = {"datasets": {}}
    for label, dataset in datasets.items():
        manifest_path = dataset_dir / f"{label}_episode_manifest.parquet"
        report_path = dataset_dir / f"{label}_dataset_build_report.json"
        dataset.episode_manifest.to_parquet(manifest_path, index=False)
        _write_json_atomic(report_path, dataset.dataset_build_report)
        artifacts["datasets"][label] = {
            "episode_manifest": str(manifest_path.relative_to(experiment_dir)),
            "dataset_build_report": str(report_path.relative_to(experiment_dir)),
        }
    if normalizer is not None:
        normalizer_path = experiment_dir / "state_normalizer.json"
        normalizer.save(normalizer_path)
        artifacts["state_normalizer"] = str(normalizer_path.relative_to(experiment_dir))
    else:
        artifacts["state_normalizer"] = None
    return artifacts


def _save_training_frames(
    run_dir: Path,
    train_env: HedgingEnv,
    result: Mapping[str, Any],
    *,
    is_save_train_trajectory: bool,
) -> dict[str, str]:
    """Save training frames."""
    validation_history = result["validation_history"].copy(deep=True)
    if "checkpoint_path" in validation_history:
        validation_history["checkpoint_path"] = None
    frames = {
        "train_schedule": train_env.get_schedule(),
        "training_episode_results": result["episode_results"],
        "training_update_history": result["update_history"],
        "validation_history": validation_history,
    }
    if is_save_train_trajectory:
        frames["training_trajectory"] = result["trajectory"]
    paths: dict[str, str] = {}
    for name, frame in frames.items():
        path = run_dir / f"{name}.parquet"
        frame.to_parquet(path, index=False)
        paths[name] = path.name
    return paths


def run_basis_training(
    config: Mapping[str, Any],
    *,
    run_specs: Sequence[Mapping[str, Any]] | None = None,
    experiment_id: str | None = None,
) -> dict[str, Any]:
    """Train conventional RL baselines and retain the checkpoint with the lowest validation risk-adjusted loss."""
    _validate_basis_config(config)
    specs = list(run_specs if run_specs is not None else get_basis_run_specs(config))
    if not specs:
        raise ValueError("run_specs must not be empty")
    for run_spec in specs:
        _validate_run_spec(config, run_spec)
    run_ids = [str(run_spec["run_id"]) for run_spec in specs]
    if len(run_ids) != len(set(run_ids)):
        raise ValueError("run_specs must contain unique run_id values")
    started_at = get_datetime_record()
    if experiment_id is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        experiment_id = f"{config['experiment_name']}_{timestamp}"
    experiment_id = get_safe_path_component(experiment_id, "experiment_id")
    output_root = Path(config["training"]["output_root"])
    if not output_root.is_absolute():
        output_root = REPOSITORY_ROOT / output_root
    experiment_dir = output_root / experiment_id
    if experiment_dir.exists() and any(experiment_dir.iterdir()):
        raise FileExistsError(
            f"The experiment directory already exists and is nonempty: {experiment_dir}"
        )
    experiment_dir.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(
        experiment_dir / "experiment_config.json",
        {
            "status": "building_datasets",
            "data_source": config.get("data_source", "spx"),
            "experiment_id": experiment_id,
            "started_at": started_at,
            "requested_config": config,
            "run_specs": specs,
            "system": get_system_info(),
            "code_fingerprints": get_code_fingerprints(),
        },
    )
    datasets = get_basis_datasets(config, labels=("train", "valid"))
    normalizer = get_basis_normalizer(datasets["train"], config)
    shared_artifacts = _save_experiment_dataset_records(experiment_dir, datasets, normalizer)
    dataset_records = {label: get_dataset_record(dataset) for label, dataset in datasets.items()}
    run_summaries: list[dict[str, Any]] = []
    for num_run, run_spec in enumerate(specs, start=1):
        run_id = str(run_spec["run_id"])
        get_safe_path_component(run_id, "run_id")
        run_dir = experiment_dir / run_id
        run_dir.mkdir(parents=False, exist_ok=False)
        run_started_at = get_datetime_record()
        num_start_time = time.perf_counter()
        print(f"\n[Experiment {num_run}/{len(specs)}] {run_id}", flush=True)
        set_global_seed(
            int(run_spec["seed"]),
            is_torch_deterministic=config["training"]["is_torch_deterministic"],
        )
        train_env = get_train_env(datasets["train"], normalizer, config, run_spec)
        agent = get_hedging_agent(config, run_spec)
        running_record = {
            "schema_version": BASIS_CONFIG_SCHEMA_VERSION,
            "status": "running",
            "data_source": config.get("data_source", "spx"),
            "experiment_id": experiment_id,
            "run_id": run_id,
            "experiment_started_at": started_at,
            "run_started_at": run_started_at,
            "requested_config": config,
            "resolved_run": run_spec,
            "datasets": dataset_records,
            "state_normalizer": normalizer.get_dict() if normalizer is not None else None,
            "train_environment": train_env.get_config(),
            "valid_environment_template": get_valid_env(
                datasets["valid"], normalizer, config, run_spec
            ).get_config(),
            "agent": agent.get_config(),
            "shared_artifacts": shared_artifacts,
            "system": get_system_info(),
            "code_fingerprints": get_code_fingerprints(),
        }
        run_config_path = run_dir / "run_config.json"
        _write_json_atomic(run_config_path, running_record)
        try:
            model_path = run_dir / "agent.pt"
            checkpoint_dir = run_dir / "checkpoints"
            result = train_hedging_agent(
                agent,
                train_env,
                get_valid_env=lambda: get_valid_env(
                    datasets["valid"], normalizer, config, run_spec
                ),
                num_replay_capacity=config["training"]["num_replay_capacity"],
                num_batch=config["training"]["num_batch"],
                warmup_fraction=config["training"]["warmup_fraction"],
                num_update_per_step=config["training"]["num_update_per_step"],
                num_valid_episodes=config["training"]["num_valid_episodes"],
                checkpoint_path=model_path,
                checkpoint_dir=checkpoint_dir,
                seed=run_spec["seed"],
                is_show_progress=config["training"]["is_show_progress"],
                is_print_summary=config["training"]["is_print_summary"],
            )
            frame_artifacts = _save_training_frames(
                run_dir,
                train_env,
                result,
                is_save_train_trajectory=config["training"]["is_save_train_trajectory"],
            )
            model_sha256 = get_file_sha256(model_path)
            checkpoint_retention = prune_run_checkpoints(run_dir, checkpoint_dir, model_path)
            completed_at = get_datetime_record()
            completed_record = {
                **running_record,
                "status": "completed",
                "run_completed_at": completed_at,
                "num_elapsed_second": time.perf_counter() - num_start_time,
                "agent": agent.get_config(),
                "training_call_config": result["config"],
                "training_metrics": result["metrics"],
                "best_validation_metrics": (
                    result["best_validation"]["metrics"]
                    if result["best_validation"] is not None
                    else None
                ),
                "final_validation_metrics": (
                    result["last_validation"]["metrics"]
                    if result["last_validation"] is not None
                    else None
                ),
                "model_selection": {
                    "metric": "j_lambda",
                    "mode": "min",
                    "num_best_validation_episode": result["num_best_validation_episode"],
                    "best_checkpoint": model_path.name,
                    "selected_model": model_path.name,
                },
                "num_train_episode": result["num_episode"],
                "num_train_step": result["num_total_step"],
                "num_gradient_update": result["num_gradient_update"],
                "model_sha256": model_sha256,
                "artifacts": {
                    "model": model_path.name,
                    "model_role": "validation_best",
                    "checkpoint_retention": checkpoint_retention,
                    **frame_artifacts,
                },
            }
            _write_json_atomic(run_config_path, completed_record)
            summary = {
                "run_id": run_id,
                "status": "completed",
                "run_completed_at": completed_at,
                "training_metrics": result["metrics"],
                "best_validation_metrics": completed_record["best_validation_metrics"],
                "final_validation_metrics": completed_record["final_validation_metrics"],
            }
            run_summaries.append(summary)
        except Exception as exc:
            failed_record = {
                **running_record,
                "status": "failed",
                "run_failed_at": get_datetime_record(),
                "num_elapsed_second": time.perf_counter() - num_start_time,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }
            _write_json_atomic(run_config_path, failed_record)
            run_summaries.append({"run_id": run_id, "status": "failed", "error": str(exc)})
            if not config["training"]["is_continue_on_error"]:
                raise
    experiment_summary = {
        "status": (
            "completed"
            if all((item["status"] == "completed" for item in run_summaries))
            else "completed_with_failures"
        ),
        "experiment_id": experiment_id,
        "data_source": config.get("data_source", "spx"),
        "started_at": started_at,
        "completed_at": get_datetime_record(),
        "num_run": len(run_summaries),
        "num_completed_run": sum((item["status"] == "completed" for item in run_summaries)),
        "runs": run_summaries,
    }
    _write_json_atomic(experiment_dir / "experiment_summary.json", experiment_summary)
    experiment_config = json.loads(
        (experiment_dir / "experiment_config.json").read_text(encoding="utf-8")
    )
    experiment_config["status"] = experiment_summary["status"]
    experiment_config["completed_at"] = experiment_summary["completed_at"]
    experiment_config["shared_artifacts"] = shared_artifacts
    experiment_config["datasets"] = dataset_records
    _write_json_atomic(experiment_dir / "experiment_config.json", experiment_config)
    print(f"\n[Training experiment completed] {experiment_dir}", flush=True)
    return {"experiment_dir": experiment_dir, "experiment_summary": experiment_summary}


def _get_cli_run_specs(
    config: Mapping[str, Any],
    *,
    reward_formulations: Sequence[str] | None,
    algorithm_names: Sequence[str] | None,
    residual_modes: Sequence[str] | None,
) -> list[dict[str, Any]]:
    """Return CLI run specs."""
    specs = get_basis_run_specs(config)
    reward_set = set(reward_formulations or ())
    algorithm_set = set(algorithm_names or ())
    residual_set = set(residual_modes or ())
    results = []
    for spec in specs:
        residual_name = "true" if spec["is_delta_residual"] else "false"
        if reward_set and spec["reward_formulation"] not in reward_set:
            continue
        if algorithm_set and spec["algorithm_name"] not in algorithm_set:
            continue
        if residual_set and residual_name not in residual_set:
            continue
        results.append(spec)
    if not results:
        raise ValueError("No experiments remain after CLI filtering")
    return results


def get_argument_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description="Run baseline option-hedging training experiments")
    parser.add_argument(
        "--config", type=Path, help="JSON file overriding the default configuration"
    )
    parser.add_argument(
        "--experiment-id", help="Explicit experiment directory name; defaults to a UTC timestamp"
    )
    parser.add_argument(
        "--reward-formulation",
        action="append",
        choices=VALID_REWARD_FORMULATIONS,
        help="Run only the specified reward; may be repeated",
    )
    parser.add_argument(
        "--algorithm",
        action="append",
        choices=VALID_ALGORITHM_NAMES,
        help="Run only the specified algorithm; may be repeated",
    )
    parser.add_argument(
        "--delta-residual",
        action="append",
        choices=("true", "false"),
        help="Run only the specified Delta-residual mode; may be repeated",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved run specifications without reading data or training",
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
    args = get_argument_parser().parse_args(argv)
    config = get_basis_config(args.config)
    config["data_source"] = validate_data_source(
        args.data_source or config.get("data_source", "spx")
    )
    _validate_basis_config(config)
    print(f"[Data source] data_source={config['data_source']}", flush=True)
    run_specs = _get_cli_run_specs(
        config,
        reward_formulations=args.reward_formulation,
        algorithm_names=args.algorithm,
        residual_modes=args.delta_residual,
    )
    print(json.dumps(run_specs, ensure_ascii=False, indent=2), flush=True)
    if args.dry_run:
        return 0
    run_basis_training(config, run_specs=run_specs, experiment_id=args.experiment_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
