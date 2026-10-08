"""Test no meta for the paper option-hedging pipeline."""

from __future__ import annotations
import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
import torch

CODES_ROOT = Path(__file__).resolve().parent
for _source_directory in (
    CODES_ROOT,
    CODES_ROOT / "no_meta_variant",
    CODES_ROOT / "rl_envs",
    CODES_ROOT / "datasets",
    CODES_ROOT / "data_processing",
    CODES_ROOT / "simulation",
    CODES_ROOT / "segmentation",
    CODES_ROOT / "meta_rl",
):
    if str(_source_directory) not in sys.path:
        sys.path.insert(0, str(_source_directory))
from data_source import VALID_DATA_SOURCES, validate_data_source
from no_meta_variant.config import _validate_no_meta_config
from no_meta_variant.data import _infer_result_data_source, load_no_meta_test_dataset
from rl_utils import evaluate_hedging_agent
from market_simulation import load_simulation_result, resolve_simulation_result_directory
from test_basis import _validate_train_test_compatibility, get_test_env, get_test_normalizer
from train_basis import (
    BASIS_CONFIG_SCHEMA_VERSION,
    _write_json_atomic,
    get_code_fingerprints,
    get_dataset_record,
    get_file_sha256,
    get_hedging_agent,
    get_safe_path_component,
    get_system_info,
)
from train_meta import _resolve_path


def _get_datetime_record() -> dict[str, str]:
    """Return the current UTC timestamp."""
    now = datetime.now(timezone.utc)
    return {"utc": now.isoformat(timespec="seconds")}


def get_completed_no_meta_train_runs(
    train_results_root: str | Path = "train_results/no_meta",
    *,
    experiment_id: str | None = None,
    run_ids: Sequence[str] | None = None,
) -> list[tuple[Path, dict[str, Any]]]:
    """Return completed no meta train runs."""
    root = _resolve_path(train_results_root)
    if not root.is_dir():
        raise FileNotFoundError(f"The no_meta training result directory does not exist: {root}")
    requested = set(run_ids or ())
    records: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(root.glob("*/*/run_config.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("status") != "completed" or record.get("variant") != "no_meta":
            continue
        if experiment_id is not None and record.get("experiment_id") != experiment_id:
            continue
        if requested and record.get("run_id") not in requested:
            continue
        artifacts = record.get("artifacts", {})
        model_name = artifacts.get("model")
        model_path = path.parent / str(model_name or "")
        if artifacts.get("model_role") != "validation_best" or not model_path.is_file():
            raise ValueError(f"The completed no_meta run has no validation-best checkpoint: {path}")
        if get_file_sha256(model_path) != record.get("model_sha256"):
            raise ValueError(f"The model SHA-256 differs from the training record: {model_path}")
        records.append((path, record))
    if not records:
        raise FileNotFoundError("No matching completed no_meta run was found")
    if experiment_id is None:
        latest = max(records, key=lambda item: item[1]["run_completed_at"]["utc"])[1][
            "experiment_id"
        ]
        records = [item for item in records if item[1]["experiment_id"] == latest]
    if requested:
        missing = requested - {record["run_id"] for _, record in records}
        if missing:
            raise FileNotFoundError(
                f"The requested completed no_meta run was not found: {sorted(missing)}"
            )
    return records


def _get_shared_config(records: Sequence[tuple[Path, Mapping[str, Any]]]) -> dict[str, Any]:
    """Return shared config."""
    if not records:
        raise ValueError("records must not be empty")
    canonical = json.dumps(records[0][1]["requested_config"], sort_keys=True)
    for path, record in records[1:]:
        if json.dumps(record["requested_config"], sort_keys=True) != canonical:
            raise ValueError(f"The selected runs have different requested_config values: {path}")
    config = json.loads(canonical)
    _validate_no_meta_config(config)
    return config


def run_no_meta_testing(
    train_records: Sequence[tuple[Path, Mapping[str, Any]]],
    *,
    test_results_root: str | Path | None = None,
    device: str | torch.device | None = None,
    is_overwrite: bool = False,
    data_source: str | None = None,
) -> dict[str, Any]:
    """Run no meta testing."""
    if not train_records:
        raise ValueError("train_records must not be empty")
    config = _get_shared_config(train_records)
    recorded_source = validate_data_source(
        train_records[0][1].get("data_source", config.get("data_source", "spx"))
    )
    if data_source is not None and validate_data_source(data_source) != recorded_source:
        raise ValueError(
            f"test data_source={data_source} differs from the training record's {recorded_source}"
        )
    config["data_source"] = recorded_source
    simulation_paths = {str(record["simulation_result"]) for _, record in train_records}
    if len(simulation_paths) != 1:
        raise ValueError("The selected runs use different simulation_result values")
    simulation_path = resolve_simulation_result_directory(
        simulation_paths.pop(), result_root=_resolve_path(config["simulation"]["result_root"])
    )
    simulation_record = load_simulation_result(
        simulation_path, result_root=_resolve_path(config["simulation"]["result_root"])
    )
    inferred_source = _infer_result_data_source(simulation_record, requested_source=recorded_source)
    if inferred_source != recorded_source:
        raise ValueError(
            f"Simulation result data source {inferred_source} differs from training data source {recorded_source}"
        )
    test_dataset = load_no_meta_test_dataset(
        simulation_record,
        data_source=recorded_source,
        data_root=config["data"].get("data_root"),
        test_date_period=config["data"]["date_periods"]["test"],
    )
    test_dataset_record = get_dataset_record(test_dataset)
    experiment_id = str(train_records[0][1]["experiment_id"])
    get_safe_path_component(experiment_id, "experiment_id")
    output_root = _resolve_path(
        test_results_root if test_results_root is not None else config["testing"]["output_root"]
    )
    experiment_dir = output_root / experiment_id
    experiment_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []
    started_at = _get_datetime_record()
    for number, (run_config_path, run_record) in enumerate(train_records, start=1):
        run_id = str(run_record["run_id"])
        get_safe_path_component(run_id, "run_id")
        run_dir = experiment_dir / run_id
        if run_dir.exists() and any(run_dir.iterdir()) and (not is_overwrite):
            raise FileExistsError(
                f"Test run results already exist; use --overwrite to overwrite them: {run_dir}"
            )
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[no_meta evaluation {number}/{len(train_records)}] {run_id}", flush=True)
        test_started_at = _get_datetime_record()
        start = time.perf_counter()
        run_spec = run_record["resolved_run"]
        normalizer = get_test_normalizer(run_config_path, run_record)
        selected_device = device if device is not None else config["testing"]["device"]
        agent = get_hedging_agent(config, run_spec, device=selected_device)
        artifacts = run_record["artifacts"]
        model_path = run_config_path.parent / artifacts["model"]
        selection = run_record.get("model_selection")
        if (
            artifacts.get("model_role") != "validation_best"
            or not isinstance(selection, Mapping)
            or selection.get("selected_model") != model_path.name
            or (selection.get("metric") != "j_lambda")
            or (selection.get("mode") != "min")
        ):
            raise ValueError("The training result is not marked as validation-best")
        model_sha256 = get_file_sha256(model_path)
        if model_sha256 != run_record.get("model_sha256"):
            raise ValueError(f"The model SHA-256 differs from the training record: {model_path}")
        agent.load(model_path, is_load_optimizer=False)
        test_env = get_test_env(test_dataset, normalizer, config, run_spec)
        _validate_train_test_compatibility(run_record, agent, test_env)
        evaluation = evaluate_hedging_agent(
            agent, test_env, is_print_summary=bool(config["testing"]["is_print_summary"])
        )
        episode_path = run_dir / "test_episode_results.parquet"
        evaluation["episode_results"].to_parquet(episode_path, index=False)
        saved_artifacts = {"test_episode_results": episode_path.name}
        if config["testing"]["is_save_test_trajectory"]:
            trajectory_path = run_dir / "test_trajectory.parquet"
            evaluation["trajectory"].to_parquet(trajectory_path, index=False)
            saved_artifacts["test_trajectory"] = trajectory_path.name
        completed_at = _get_datetime_record()
        test_metrics = {
            "schema_version": BASIS_CONFIG_SCHEMA_VERSION,
            "variant": "no_meta",
            "data_source": recorded_source,
            "experiment_id": experiment_id,
            "run_id": run_id,
            "test_completed_at": completed_at,
            "metrics": evaluation["metrics"],
        }
        _write_json_atomic(run_dir / "test_metrics.json", test_metrics)
        test_record = {
            "schema_version": BASIS_CONFIG_SCHEMA_VERSION,
            "variant": "no_meta",
            "status": "completed",
            "data_source": recorded_source,
            "experiment_id": experiment_id,
            "run_id": run_id,
            "test_started_at": test_started_at,
            "test_completed_at": completed_at,
            "num_elapsed_second": time.perf_counter() - start,
            "training_run_config_path": str(run_config_path),
            "training_run_config_sha256": get_file_sha256(run_config_path),
            "training_completed_at": run_record["run_completed_at"],
            "training_run": run_record,
            "simulation_result": str(simulation_path),
            "simulation_experiment_id": simulation_record["experiment_id"],
            "test_dataset": test_dataset_record,
            "test_environment": test_env.get_config(),
            "agent": agent.get_config(),
            "model_sha256": model_sha256,
            "model_selection": selection,
            "metrics": evaluation["metrics"],
            "num_test_reset": evaluation["num_reset"],
            "num_test_step": evaluation["num_total_step"],
            "artifacts": {
                **saved_artifacts,
                "test_metrics": "test_metrics.json",
                "test_config": "test_config.json",
            },
            "system": get_system_info(),
            "code_fingerprints": get_code_fingerprints(),
        }
        _write_json_atomic(run_dir / "test_config.json", test_record)
        summaries.append(
            {
                "run_id": run_id,
                "status": "completed",
                "metrics": evaluation["metrics"],
                "test_completed_at": completed_at,
            }
        )
    summary = {
        "schema_version": BASIS_CONFIG_SCHEMA_VERSION,
        "variant": "no_meta",
        "status": "completed",
        "data_source": recorded_source,
        "experiment_id": experiment_id,
        "simulation_result": str(simulation_path),
        "started_at": started_at,
        "completed_at": _get_datetime_record(),
        "num_run": len(summaries),
        "test_dataset": test_dataset_record,
        "runs": summaries,
    }
    _write_json_atomic(experiment_dir / "test_experiment_summary.json", summary)
    print(f"\n[no_meta test experiment completed] {experiment_dir}", flush=True)
    return {"experiment_dir": experiment_dir, "summary": summary}


def get_argument_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-results-root", type=Path, default=Path("train_results/no_meta"))
    parser.add_argument("--test-results-root", type=Path, default=None)
    parser.add_argument("--experiment-id", type=str, default=None)
    parser.add_argument("--run-id", action="append", default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--data-source", choices=VALID_DATA_SOURCES, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse command-line arguments, run the requested operation, and report failures."""
    args = get_argument_parser().parse_args(argv)
    records = get_completed_no_meta_train_runs(
        args.train_results_root, experiment_id=args.experiment_id, run_ids=args.run_id
    )
    print(
        json.dumps(
            [
                {"experiment_id": r["experiment_id"], "run_id": r["run_id"], "config": str(p)}
                for p, r in records
            ],
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    run_no_meta_testing(
        records,
        test_results_root=args.test_results_root,
        device=args.device,
        is_overwrite=args.overwrite,
        data_source=args.data_source,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
