"""Pipeline for the paper option-hedging pipeline."""

from __future__ import annotations
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
import pandas as pd
from hedging_env import HedgingEnv
from hedging_meta_rl import train_meta_hedging_agent
from train_basis import (
    get_dataset_record,
    get_file_sha256,
    get_safe_path_component,
    get_system_info,
    prune_run_checkpoints,
    set_global_seed,
)
from train_meta import (
    META_CONFIG_SCHEMA_VERSION,
    _get_environment,
    _get_experiment_id,
    _get_latent_task_summary,
    _get_meta_code_fingerprints,
    _get_normalizer,
    _resolve_path,
    _save_training_frames,
    get_meta_agent,
)
from .config import _validate_no_simulation_config, get_no_simulation_run_specs
from .data import load_no_simulation_data


def _get_datetime_record() -> dict[str, str]:
    """Return the current UTC timestamp."""
    now = datetime.now(timezone.utc)
    return {"utc": now.isoformat(timespec="seconds")}


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Write JSON atomic."""
    from train_basis import _write_json_atomic as write_json

    write_json(path, payload)


def _get_variant_code_fingerprints() -> list[dict[str, Any]]:
    """Return variant code fingerprints."""
    records = _get_meta_code_fingerprints()
    root = Path(__file__).resolve().parent
    repository_root = root.parent
    for script_path in (
        repository_root / "train_no_simulation.py",
        repository_root / "test_no_simulation.py",
    ):
        if script_path.is_file():
            records.append(
                {
                    "path": str(script_path.relative_to(repository_root)),
                    "num_bytes": script_path.stat().st_size,
                    "sha256": get_file_sha256(script_path),
                }
            )
    for path in sorted(root.glob("*.py")):
        if not path.is_file():
            continue
        record = {
            "path": str(path.relative_to(root.parent)),
            "num_bytes": path.stat().st_size,
            "sha256": get_file_sha256(path),
        }
        if record["path"] not in {item["path"] for item in records}:
            records.append(record)
    return records


def _save_no_simulation_task_manifest(tasks: Sequence[Any], output_path: Path) -> None:
    """Save no simulation task manifest."""
    frames: list[pd.DataFrame] = []
    for task in tasks:
        frame = task.real_dataset.episode_manifest.copy(deep=True)
        frame.insert(0, "meta_segment_id", task.segment_id)
        frame.insert(1, "meta_is_simulated", False)
        frames.append(frame)
    pd.concat(frames, ignore_index=True).to_parquet(output_path, index=False)


def run_no_simulation_training(
    config: Mapping[str, Any],
    *,
    run_specs: Sequence[Mapping[str, Any]] | None = None,
    segmentation_result: str | Path | None = None,
    experiment_id: str | None = None,
) -> dict[str, Any]:
    """Run no simulation training."""
    _validate_no_simulation_config(config)
    specs = [dict(value) for value in run_specs or get_no_simulation_run_specs(config)]
    if not specs:
        raise ValueError("run_specs must not be empty")
    segmentation = config["segmentation"]
    result_value = (
        segmentation_result if segmentation_result is not None else segmentation.get("result")
    )
    data_source = str(config.get("data_source", "spx"))
    tasks, train_dataset, valid_dataset, segmentation_record, result_path = load_no_simulation_data(
        result_value,
        segmentation_result_root=_resolve_path(segmentation["result_root"]),
        data_source=data_source,
        data_root=config["data"].get("data_root"),
        train_date_period=config["data"]["date_periods"]["train"],
        valid_date_period=config["data"]["date_periods"]["valid"],
        is_exclude_cross_boundary=bool(segmentation["is_exclude_cross_boundary"]),
    )
    normalizer = _get_normalizer(train_dataset, config)
    resolved_experiment_id = experiment_id or _get_experiment_id().replace(
        "meta_", "no_simulation_", 1
    )
    get_safe_path_component(resolved_experiment_id, "experiment_id")
    output_root = _resolve_path(config["training"]["output_root"])
    experiment_dir = output_root / resolved_experiment_id
    if experiment_dir.exists():
        raise FileExistsError(
            f"The no-simulation training directory already exists: {experiment_dir}"
        )
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
    _save_no_simulation_task_manifest(tasks, dataset_directory / "task_episode_manifest.parquet")
    shared_record = {
        "variant": "no_simulation",
        "data_source": data_source,
        "experiment_id": resolved_experiment_id,
        "segmentation_result": str(result_path),
        "segmentation_experiment_id": segmentation_record["experiment_id"],
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
    started_at = _get_datetime_record()
    experiment_record: dict[str, Any] = {
        "schema_version": META_CONFIG_SCHEMA_VERSION,
        "variant": "no_simulation",
        "status": "running",
        "data_source": data_source,
        "experiment_id": resolved_experiment_id,
        "started_at": started_at,
        "num_run": len(specs),
        "segmentation_result": str(result_path),
        "system": get_system_info(),
        "code_fingerprints": _get_variant_code_fingerprints(),
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
        run_started = _get_datetime_record()
        running = {
            "schema_version": META_CONFIG_SCHEMA_VERSION,
            "variant": "no_simulation",
            "status": "running",
            "data_source": data_source,
            "experiment_id": resolved_experiment_id,
            "run_id": run_id,
            "run_started_at": run_started,
            "run_index": num_run,
            "num_run": len(specs),
            "resolved_run": run_spec,
            "requested_config": config,
            "segmentation_result": str(result_path),
            "state_normalizer": normalizer.get_dict() if normalizer else None,
        }
        _write_json_atomic(run_dir / "run_config.json", running)
        print(f"\n[no_simulation experiment {num_run}/{len(specs)}] {run_id}", flush=True)
        start = time.perf_counter()
        try:
            set_global_seed(
                int(run_spec["seed"]),
                is_torch_deterministic=bool(config["training"]["is_torch_deterministic"]),
            )
            agent = get_meta_agent(config, run_spec)

            def train_env_factory(dataset: Any, seed: int) -> HedgingEnv:
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
                simulated_batch_fraction=0.0,
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
            artifacts["checkpoint_retention"] = prune_run_checkpoints(
                run_dir, checkpoints_dir, model_path
            )
            completed = {
                **running,
                "status": "completed",
                "run_completed_at": _get_datetime_record(),
                "num_elapsed_second": time.perf_counter() - start,
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
                "run_failed_at": _get_datetime_record(),
                "num_elapsed_second": time.perf_counter() - start,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }
            _write_json_atomic(run_dir / "run_config.json", failed)
            summaries.append({"run_id": run_id, "status": "failed", "error": str(exc)})
            if not config["training"]["is_continue_on_error"]:
                experiment_record.update(
                    {"status": "failed", "failed_at": _get_datetime_record(), "runs": summaries}
                )
                _write_json_atomic(experiment_dir / "experiment.json", experiment_record)
                raise
    completed_count = sum((item["status"] == "completed" for item in summaries))
    experiment_record.update(
        {
            "status": "completed" if completed_count == len(summaries) else "partial",
            "completed_at": _get_datetime_record(),
            "num_completed_run": completed_count,
            "runs": summaries,
        }
    )
    _write_json_atomic(experiment_dir / "experiment.json", experiment_record)
    print(f"\n[no_simulation training completed] {experiment_dir}", flush=True)
    return {"experiment_dir": experiment_dir, "result": experiment_record}
