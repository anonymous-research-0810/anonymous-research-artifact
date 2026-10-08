"""Config for the paper option-hedging pipeline."""

from __future__ import annotations
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping
from data_source import validate_data_source
from train_meta import (
    META_CONFIG_SCHEMA_VERSION,
    _merge_config_strict,
    _validate_meta_config,
    get_default_meta_config,
)

NO_SIMULATION_CONFIG_SCHEMA_VERSION = META_CONFIG_SCHEMA_VERSION


def get_default_no_simulation_config() -> dict[str, Any]:
    """Return default no simulation config."""
    config = deepcopy(get_default_meta_config())
    config["experiment_name"] = "no_simulation"
    config["variant"] = "no_simulation"
    config["segmentation"] = {
        "result": None,
        "result_root": "segmentation/seg_results",
        "is_exclude_cross_boundary": bool(
            config.get("simulation", {}).get("is_exclude_cross_boundary", False)
        ),
    }
    config.pop("simulation", None)
    config["training"]["simulated_batch_fraction"] = 0.0
    config["training"]["output_root"] = "train_results/no_simulation"
    config["testing"]["output_root"] = "test_results/no_simulation"
    return config


def get_no_simulation_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Return no simulation config."""
    config = get_default_no_simulation_config()
    if config_path is not None:
        path = Path(config_path)
        import json

        override = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(override, Mapping):
            raise ValueError("The configuration root must be a JSON object")
        config = _merge_config_strict(config, override)
    _validate_no_simulation_config(config)
    return config


def _validate_no_simulation_config(config: Mapping[str, Any]) -> None:
    """Validate no simulation config."""
    _validate_meta_config(config)
    if config.get("variant") != "no_simulation":
        raise ValueError("variant must be no_simulation")
    validate_data_source(config.get("data_source", "spx"))
    segmentation = config.get("segmentation")
    if not isinstance(segmentation, Mapping):
        raise ValueError("segmentation must be a JSON object")
    result_root = segmentation.get("result_root")
    if not isinstance(result_root, (str, Path)) or not str(result_root):
        raise ValueError("segmentation.result_root must be a nonempty path")
    result = segmentation.get("result")
    if result is not None and (not isinstance(result, (str, Path))):
        raise ValueError("segmentation.result must be a path, filename, or null")
    if not isinstance(segmentation.get("is_exclude_cross_boundary"), bool):
        raise ValueError("segmentation.is_exclude_cross_boundary must be a boolean")


def get_no_simulation_run_specs(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return no simulation run specs."""
    _validate_no_simulation_config(config)
    scales = config["environment"]["training_reward_scales"]
    specs: list[dict[str, Any]] = []
    for seed in config["training"]["seeds"]:
        for algorithm in config["agent"]["algorithm_names"]:
            for is_residual in config["agent"]["is_delta_residual_values"]:
                for reward in config["environment"]["reward_formulations"]:
                    action_name = "delta_residual" if is_residual else "direct"
                    specs.append(
                        {
                            "run_id": f"{algorithm}__{action_name}__{reward}__seed{int(seed)}",
                            "algorithm_name": str(algorithm),
                            "is_delta_residual": bool(is_residual),
                            "reward_formulation": str(reward),
                            "training_reward_scale": float(scales[reward]),
                            "seed": int(seed),
                        }
                    )
    return specs
