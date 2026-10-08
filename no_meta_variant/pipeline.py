"""Pipeline for the paper option-hedging pipeline."""

from __future__ import annotations
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
import pandas as pd
from rl_utils import train_hedging_agent
from train_basis import (
    BASIS_CONFIG_SCHEMA_VERSION,
    _save_experiment_dataset_records,
    _validate_run_spec,
    _write_json_atomic,
    get_basis_normalizer,
    get_code_fingerprints,
    get_dataset_record,
    get_file_sha256,
    get_hedging_agent,
    get_safe_path_component,
    get_system_info,
    get_train_env,
    get_valid_env,
    prune_run_checkpoints,
    set_global_seed,
)
from train_meta import _resolve_path
from .config import _validate_no_meta_config, get_no_meta_run_specs
from .data import load_no_meta_data


def _get_datetime_record() -> dict[str, str]:
    """Return the current UTC timestamp."""
    now = datetime.now(timezone.utc)
    return {"utc": now.isoformat(timespec="seconds")}


def _get_variant_code_fingerprints() -> list[dict[str, Any]]:
    """Return variant code fingerprints."""
    records = get_code_fingerprints()
    root = Path(__file__).resolve().parent
    repository_root = root.parent
    for path in (
        repository_root / "train_no_meta.py",
        repository_root / "test_no_meta.py",
        *sorted(root.glob("*.py")),
    ):
        if not path.is_file():
            continue
        item = {
            "path": str(path.relative_to(repository_root)),
            "num_bytes": path.stat().st_size,
            "sha256": get_file_sha256(path),
        }
        if item["path"] not in {record["path"] for record in records}:
            records.append(item)
    return records


def _save_no_meta_training_frames(
    run_dir: Path, train_env: Any, result: Mapping[str, Any], *, is_save_train_trajectory: bool
) -> dict[str, str]:
    """Save no meta training frames."""
    validation = result["validation_history"].copy(deep=True)
    if "checkpoint_path" in validation:
        validation["checkpoint_path"] = None
    frames = {
        "train_schedule": train_env.get_schedule(),
        "training_episode_results": result["episode_results"],
        "training_update_history": result["update_history"],
        "validation_history": validation,
    }
    if is_save_train_trajectory:
        frames["training_trajectory"] = result["trajectory"]
    artifacts: dict[str, str] = {}
    for name, frame in frames.items():
        path = run_dir / f"{name}.parquet"
        frame.to_parquet(path, index=False)
        artifacts[name] = path.name
    return artifacts


def run_no_meta_training(
    config: Mapping[str, Any],
    *,
    run_specs: Sequence[Mapping[str, Any]] | None = None,
    simulation_result: str | Path | None = None,
    experiment_id: str | None = None,
) -> dict[str, Any]:
    """Run no meta training."""
    _validate_no_meta_config(config)
    specs = [
        dict(item)
        for item in (run_specs if run_specs is not None else get_no_meta_run_specs(config))
    ]
    if not specs:
        raise ValueError("run_specs must not be empty")
    for run_spec in specs:
        _validate_run_spec(config, run_spec)
    run_ids = [str(item["run_id"]) for item in specs]
    if len(run_ids) != len(set(run_ids)):
        raise ValueError("run_specs must contain unique run_id values")
    simulation = config["simulation"]
    result_value = simulation_result if simulation_result is not None else simulation["result"]
    train_dataset, valid_dataset, tasks, simulation_record, result_path = load_no_meta_data(
        result_value,
        simulation_result_root=_resolve_path(simulation["result_root"]),
        data_source=str(config.get("data_source", "spx")),
        data_root=config["data"].get("data_root"),
        valid_date_period=config["data"]["date_periods"]["valid"],
        is_exclude_cross_boundary=bool(simulation["is_exclude_cross_boundary"]),
    )
    normalizer = get_basis_normalizer(_get_real_train_for_normalizer(tasks), config)
    resolved_experiment_id = experiment_id or "no_meta_" + datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    )
    get_safe_path_component(resolved_experiment_id, "experiment_id")
    output_root = _resolve_path(config["training"]["output_root"])
    experiment_dir = output_root / resolved_experiment_id
    if experiment_dir.exists() and any(experiment_dir.iterdir()):
        raise FileExistsError(
            f"The no_meta experiment directory already exists and is nonempty: {experiment_dir}"
        )
    experiment_dir.mkdir(parents=True, exist_ok=True)
    datasets = {"train_combined": train_dataset, "valid": valid_dataset}
    shared_artifacts = _save_experiment_dataset_records(experiment_dir, datasets, normalizer)
    dataset_records = {label: get_dataset_record(dataset) for label, dataset in datasets.items()}
    _write_json_atomic(
        experiment_dir / "task_records.json",
        {"simulation_result": str(result_path), "tasks": [task.get_record() for task in tasks]},
    )
    shared_artifacts["task_records"] = "task_records.json"
    started_at = _get_datetime_record()
    experiment_record: dict[str, Any] = {
        "schema_version": BASIS_CONFIG_SCHEMA_VERSION,
        "variant": "no_meta",
        "status": "running",
        "data_source": config.get("data_source", "spx"),
        "experiment_id": resolved_experiment_id,
        "started_at": started_at,
        "num_run": len(specs),
        "requested_config": config,
        "run_specs": specs,
        "simulation_result": str(result_path),
        "simulation_experiment_id": simulation_record["experiment_id"],
        "system": get_system_info(),
        "code_fingerprints": _get_variant_code_fingerprints(),
        "datasets": dataset_records,
        "shared_artifacts": shared_artifacts,
    }
    _write_json_atomic(experiment_dir / "experiment.json", experiment_record)
    summaries: list[dict[str, Any]] = []
    for number, run_spec in enumerate(specs, start=1):
        run_id = str(run_spec["run_id"])
        get_safe_path_component(run_id, "run_id")
        run_dir = experiment_dir / run_id
        run_dir.mkdir()
        checkpoint_dir = run_dir / "checkpoints"
        checkpoint_dir.mkdir()
        run_started = _get_datetime_record()
        print(f"\n[no_meta training {number}/{len(specs)}] {run_id}", flush=True)
        start = time.perf_counter()
        set_global_seed(
            int(run_spec["seed"]),
            is_torch_deterministic=bool(config["training"]["is_torch_deterministic"]),
        )
        train_env = get_train_env(train_dataset, normalizer, config, run_spec)
        valid_env_template = get_valid_env(valid_dataset, normalizer, config, run_spec)
        agent = get_hedging_agent(config, run_spec)
        running = {
            "schema_version": BASIS_CONFIG_SCHEMA_VERSION,
            "variant": "no_meta",
            "status": "running",
            "data_source": config.get("data_source", "spx"),
            "experiment_id": resolved_experiment_id,
            "run_id": run_id,
            "run_started_at": run_started,
            "resolved_run": run_spec,
            "requested_config": config,
            "simulation_result": str(result_path),
            "datasets": dataset_records,
            "state_normalizer": normalizer.get_dict() if normalizer else None,
            "train_environment": train_env.get_config(),
            "valid_environment_template": valid_env_template.get_config(),
            "agent": agent.get_config(),
            "shared_artifacts": shared_artifacts,
        }
        run_config_path = run_dir / "run_config.json"
        _write_json_atomic(run_config_path, running)
        try:
            result = train_hedging_agent(
                agent,
                train_env,
                get_valid_env=lambda: get_valid_env(valid_dataset, normalizer, config, run_spec),
                num_replay_capacity=int(config["training"]["num_replay_capacity"]),
                num_batch=int(config["training"]["num_batch"]),
                warmup_fraction=float(config["training"]["warmup_fraction"]),
                num_update_per_step=int(config["training"]["num_update_per_step"]),
                num_valid_episodes=int(config["training"]["num_valid_episodes"]),
                checkpoint_path=run_dir / "agent.pt",
                checkpoint_dir=checkpoint_dir,
                seed=int(run_spec["seed"]),
                is_show_progress=bool(config["training"]["is_show_progress"]),
                is_print_summary=bool(config["training"]["is_print_summary"]),
            )
            frame_artifacts = _save_no_meta_training_frames(
                run_dir,
                train_env,
                result,
                is_save_train_trajectory=bool(config["training"]["is_save_train_trajectory"]),
            )
            model_path = run_dir / "agent.pt"
            model_sha256 = get_file_sha256(model_path)
            checkpoint_retention = prune_run_checkpoints(run_dir, checkpoint_dir, model_path)
            completed_at = _get_datetime_record()
            completed = {
                **running,
                "status": "completed",
                "run_completed_at": completed_at,
                "num_elapsed_second": time.perf_counter() - start,
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
            _write_json_atomic(run_config_path, completed)
            summaries.append(
                {
                    "run_id": run_id,
                    "status": "completed",
                    "run_completed_at": completed_at,
                    "training_metrics": result["metrics"],
                    "best_validation_metrics": completed["best_validation_metrics"],
                }
            )
        except Exception as exc:
            _write_json_atomic(
                run_config_path,
                {
                    **running,
                    "status": "failed",
                    "run_failed_at": _get_datetime_record(),
                    "num_elapsed_second": time.perf_counter() - start,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                },
            )
            summaries.append({"run_id": run_id, "status": "failed", "error": str(exc)})
            if not config["training"]["is_continue_on_error"]:
                experiment_record.update(
                    {"status": "failed", "failed_at": _get_datetime_record(), "runs": summaries}
                )
                _write_json_atomic(experiment_dir / "experiment.json", experiment_record)
                raise
    experiment_record.update(
        {
            "status": (
                "completed"
                if all((item["status"] == "completed" for item in summaries))
                else "completed_with_failures"
            ),
            "completed_at": _get_datetime_record(),
            "num_completed_run": sum((item["status"] == "completed" for item in summaries)),
            "runs": summaries,
        }
    )
    _write_json_atomic(experiment_dir / "experiment.json", experiment_record)
    print(f"\n[no_meta training completed] {experiment_dir}", flush=True)
    return {"experiment_dir": experiment_dir, "result": experiment_record}


def _get_real_train_for_normalizer(tasks: Sequence[Any]) -> Any:
    """Return real train for normalizer."""
    if not tasks:
        raise ValueError("tasks must not be empty")
    first = tasks[0].real_dataset
    frames = [task.real_dataset.episode_steps for task in tasks]
    manifests = [task.real_dataset.episode_manifest for task in tasks]
    steps = pd.concat(frames, ignore_index=True, sort=False)
    manifest = pd.concat(manifests, ignore_index=True, sort=False)
    return type(first)(
        label="train",
        config=first.config,
        episode_steps=steps,
        episode_manifest=manifest,
        candidate_selection_audit=first.candidate_selection_audit.iloc[0:0].copy(deep=True),
        dataset_build_report={"source": "no_meta_variant.real_only_normalizer"},
    )
