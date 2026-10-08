"""Test meta for the paper option-hedging pipeline."""

from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence
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
from hedging_env import get_state_normalizer_from_json
from hedging_meta_rl import evaluate_meta_hedging_agent
from market_simulation import load_simulation_result
from train_basis import (
    _write_json_atomic,
    get_datetime_record,
    get_file_sha256,
    get_safe_path_component,
    get_system_info,
)
from train_meta import (
    _get_dataset_from_source_config,
    _get_environment,
    _resolve_path,
    _validate_meta_config,
    get_meta_agent,
)
from data_source import VALID_DATA_SOURCES, resolve_data_root, validate_data_source


def get_completed_meta_train_runs(
    train_results_root: str | Path = "train_results/meta",
    *,
    experiment_id: str | None = None,
    run_ids: Sequence[str] | None = None,
) -> list[tuple[Path, dict[str, Any]]]:
    """Return completed meta train runs."""
    root = _resolve_path(train_results_root)
    if not root.is_dir():
        raise FileNotFoundError(f"The meta-RL training result directory does not exist: {root}")
    requested_ids = set(run_ids or ())
    records: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(root.glob("*/*/run_config.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("status") != "completed":
            continue
        if experiment_id is not None and record.get("experiment_id") != experiment_id:
            continue
        if requested_ids and record.get("run_id") not in requested_ids:
            continue
        model = path.parent / record.get("artifacts", {}).get("model", "")
        if record.get("model_role") != "validation_best" or not model.is_file():
            raise ValueError(f"The completed run has no validation-best model: {path}")
        if get_file_sha256(model) != record.get("model_sha256"):
            raise ValueError(f"The model SHA-256 differs from the training record: {model}")
        records.append((path, record))
    if not records:
        raise FileNotFoundError("No matching completed meta-RL training run was found")
    if experiment_id is None:
        latest_experiment = max(records, key=lambda item: item[1]["run_completed_at"]["utc"])[1][
            "experiment_id"
        ]
        records = [item for item in records if item[1]["experiment_id"] == latest_experiment]
    if requested_ids:
        missing = requested_ids - {record["run_id"] for _, record in records}
        if missing:
            raise FileNotFoundError(f"The requested completed run was not found: {sorted(missing)}")
    return records


def _get_shared_config(records: Sequence[tuple[Path, Mapping[str, Any]]]) -> dict[str, Any]:
    """Return shared config."""
    first = records[0][1]["requested_config"]
    canonical = json.dumps(first, ensure_ascii=False, sort_keys=True)
    for path, record in records[1:]:
        if json.dumps(record["requested_config"], ensure_ascii=False, sort_keys=True) != canonical:
            raise ValueError(
                f"The selected meta-RL runs have different requested_config values: {path}"
            )
    config = json.loads(canonical)
    _validate_meta_config(config)
    return config


def run_meta_testing(
    train_records: Sequence[tuple[Path, Mapping[str, Any]]],
    *,
    test_results_root: str | Path | None = None,
    device: str | torch.device | None = None,
    is_overwrite: bool = False,
    data_source: str | None = None,
) -> dict[str, Any]:
    """Run meta testing."""
    if not train_records:
        raise ValueError("train_records must not be empty")
    config = _get_shared_config(train_records)
    if data_source is not None:
        selected_source = validate_data_source(data_source)
        recorded_source = str(
            train_records[0][1].get("data_source", config.get("data_source", "spx"))
        )
        if selected_source != recorded_source:
            raise ValueError(
                f"test data_source={selected_source} differs from the training result's {recorded_source}"
            )
        config["data_source"] = selected_source
        _validate_meta_config(config)
    print(f"[Data source] data_source={config.get('data_source', 'spx')}", flush=True)
    simulation_paths = {str(record["simulation_result"]) for _, record in train_records}
    if len(simulation_paths) != 1:
        raise ValueError("The selected runs use different Stage 2 results")
    simulation_record = load_simulation_result(next(iter(simulation_paths)))
    simulation_source = validate_data_source(simulation_record.get("data_source", "spx"))
    if simulation_source != config.get("data_source", "spx"):
        raise ValueError(
            f"The meta-RL test data source differs from the Stage 2 simulation result: {config.get('data_source', 'spx')} != {simulation_source}"
        )
    configured_data_root = resolve_data_root(
        config.get("data_source", "spx"), config["data"]["data_root"]
    )
    test_dataset = _get_dataset_from_source_config(
        simulation_record["dataset"]["config"],
        label="test",
        date_period=config["data"]["date_periods"]["test"],
        data_root=configured_data_root,
    )
    experiment_id = str(train_records[0][1]["experiment_id"])
    get_safe_path_component(experiment_id, "experiment_id")
    root = _resolve_path(
        test_results_root if test_results_root is not None else config["testing"]["output_root"]
    )
    experiment_dir = root / experiment_id
    experiment_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []
    started_at = get_datetime_record()
    for num_run, (run_config_path, run_record) in enumerate(train_records, start=1):
        run_id = str(run_record["run_id"])
        get_safe_path_component(run_id, "run_id")
        output_dir = experiment_dir / run_id
        if output_dir.exists() and any(output_dir.iterdir()) and (not is_overwrite):
            raise FileExistsError(
                f"Test results already exist; use --overwrite to overwrite them: {output_dir}"
            )
        output_dir.mkdir(parents=True, exist_ok=True)
        normalizer_path = run_config_path.parent.parent / "state_normalizer.json"
        normalizer = (
            get_state_normalizer_from_json(normalizer_path) if normalizer_path.is_file() else None
        )
        if (normalizer.get_dict() if normalizer else None) != run_record.get("state_normalizer"):
            raise ValueError("The saved normalizer differs from the meta-RL training record")
        run_spec = run_record["resolved_run"]
        agent = get_meta_agent(
            config, run_spec, device=device if device is not None else config["testing"]["device"]
        )
        model_path = run_config_path.parent / run_record["artifacts"]["model"]
        agent.load(model_path, is_load_optimizer=False)
        test_env = _get_environment(test_dataset, normalizer, config, run_spec, label="test")
        if test_env.reward_formulation != run_spec["reward_formulation"]:
            raise ValueError("Test reward formulation differs from the training configuration")
        if agent.is_delta_residual != bool(run_spec["is_delta_residual"]):
            raise ValueError("Test agent Delta-residual settings differ from training")
        print(f"\n[Evaluation {num_run}/{len(train_records)}] {run_id}", flush=True)
        start = time.perf_counter()
        evaluation = evaluate_meta_hedging_agent(
            agent,
            test_env,
            num_recent_context_episodes=int(config["training"]["num_recent_context_episodes"]),
            context_seed=int(run_spec["seed"]),
            is_print_summary=bool(config["testing"]["is_print_summary"]),
        )
        episode_path = output_dir / "test_episode_results.parquet"
        latent_path = output_dir / "latent_episode_statistics.parquet"
        evaluation["episode_results"].to_parquet(episode_path, index=False)
        evaluation["latent_statistics"].to_parquet(latent_path, index=False)
        artifacts = {
            "test_episode_results": episode_path.name,
            "latent_episode_statistics": latent_path.name,
        }
        if config["testing"]["is_save_test_trajectory"]:
            trajectory_path = output_dir / "test_trajectory.parquet"
            evaluation["trajectory"].to_parquet(trajectory_path, index=False)
            artifacts["test_trajectory"] = trajectory_path.name
        result = {
            "status": "completed",
            "data_source": config.get("data_source", "spx"),
            "experiment_id": experiment_id,
            "run_id": run_id,
            "completed_at": get_datetime_record(),
            "num_elapsed_second": time.perf_counter() - start,
            "train_run_config": str(run_config_path),
            "train_model": str(model_path),
            "train_model_sha256": run_record["model_sha256"],
            "model_role": run_record["model_role"],
            "resolved_run": run_spec,
            "test_dataset_config": test_dataset.config.get_dict(),
            "test_num_episode": test_dataset.num_episode,
            "test_metrics": evaluation["metrics"],
            "context_protocol": evaluation["context_protocol"],
            "agent_config": agent.get_config(),
            "test_environment": test_env.get_config(),
            "artifacts": artifacts,
        }
        _write_json_atomic(output_dir / "test_result.json", result)
        summaries.append(
            {"run_id": run_id, "status": "completed", "test_metrics": result["test_metrics"]}
        )
    experiment_result = {
        "status": "completed",
        "data_source": config.get("data_source", "spx"),
        "experiment_id": experiment_id,
        "started_at": started_at,
        "completed_at": get_datetime_record(),
        "num_run": len(summaries),
        "test_dataset_config": test_dataset.config.get_dict(),
        "test_num_episode": test_dataset.num_episode,
        "system": get_system_info(),
        "runs": summaries,
    }
    _write_json_atomic(experiment_dir / "experiment.json", experiment_result)
    print(f"\n[Test experiment completed] {experiment_dir}", flush=True)
    return {"experiment_dir": experiment_dir, "result": experiment_result}


def get_argument_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-results-root",
        type=Path,
        default=Path("train_results/meta"),
        help="Meta-RL training result root",
    )
    parser.add_argument(
        "--experiment-id",
        type=str,
        default=None,
        help="Training experiment directory; defaults to the latest experiment",
    )
    parser.add_argument(
        "--run-id", action="append", help="Evaluate only the specified run_id; may be repeated"
    )
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
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
    records = get_completed_meta_train_runs(
        args.train_results_root, experiment_id=args.experiment_id, run_ids=args.run_id
    )
    print(
        json.dumps(
            [
                {
                    "experiment_id": record["experiment_id"],
                    "run_id": record["run_id"],
                    "config": str(path),
                }
                for path, record in records
            ],
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    run_meta_testing(
        records,
        test_results_root=args.output_root,
        device=args.device,
        is_overwrite=args.overwrite,
        data_source=args.data_source,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
