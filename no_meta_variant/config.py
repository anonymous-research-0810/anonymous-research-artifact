"""Config for the paper option-hedging pipeline."""

from __future__ import annotations
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping
from data_source import validate_data_source
from train_basis import (
    BASIS_CONFIG_SCHEMA_VERSION,
    _merge_config_strict,
    _validate_basis_config,
    get_default_basis_config,
)

NO_META_CONFIG_SCHEMA_VERSION = BASIS_CONFIG_SCHEMA_VERSION


def get_default_no_meta_config() -> dict[str, Any]:
    """Return default no meta config."""
    config = deepcopy(get_default_basis_config())
    config["experiment_name"] = "no_meta"
    config["variant"] = "no_meta"
    config["agent"]["algorithm_names"] = ["sac", "td3"]
    config["simulation"] = {
        "result": None,
        "result_root": "simulation/sim_results",
        "is_exclude_cross_boundary": False,
    }
    config["training"]["output_root"] = "train_results/no_meta"
    config["testing"]["output_root"] = "test_results/no_meta"
    return config


def get_no_meta_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Return no meta config."""
    config = get_default_no_meta_config()
    if config_path is not None:
        override = json.loads(Path(config_path).read_text(encoding="utf-8"))
        if not isinstance(override, Mapping):
            raise ValueError("The configuration root must be a JSON object")
        config = _merge_config_strict(config, override)
    _validate_no_meta_config(config)
    return config


def _validate_no_meta_config(config: Mapping[str, Any]) -> None:
    """Validate no meta config."""
    _validate_basis_config(config)
    if config.get("variant") != "no_meta":
        raise ValueError("variant must be no_meta")
    algorithms = list(config["agent"]["algorithm_names"])
    if set(algorithms) != {"sac", "td3"}:
        raise ValueError("The no_meta ablation algorithm_names must be exactly sac and td3")
    validate_data_source(config.get("data_source", "spx"))
    simulation = config.get("simulation")
    if not isinstance(simulation, Mapping):
        raise ValueError("simulation must be a JSON object")
    result_root = simulation.get("result_root")
    if not isinstance(result_root, (str, Path)) or not str(result_root):
        raise ValueError("simulation.result_root must be a nonempty path")
    result = simulation.get("result")
    if result is not None and (not isinstance(result, (str, Path))):
        raise ValueError("simulation.result must be a path, filename, or null")
    if not isinstance(simulation.get("is_exclude_cross_boundary"), bool):
        raise ValueError("simulation.is_exclude_cross_boundary must be a boolean")


def get_no_meta_run_specs(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return no meta run specs."""
    _validate_no_meta_config(config)
    specs: list[dict[str, Any]] = []
    for reward in config["environment"]["reward_formulations"]:
        for algorithm in config["agent"]["algorithm_names"]:
            for is_residual in config["agent"]["is_delta_residual_values"]:
                for seed in config["training"]["seeds"]:
                    action_name = "delta_residual" if is_residual else "direct_action"
                    specs.append(
                        {
                            "run_id": f"{reward}__{algorithm}__{action_name}__seed_{int(seed)}",
                            "reward_formulation": str(reward),
                            "training_reward_scale": float(
                                config["environment"]["training_reward_scales"][reward]
                            ),
                            "algorithm_name": str(algorithm),
                            "is_delta_residual": bool(is_residual),
                            "seed": int(seed),
                        }
                    )
    return specs
