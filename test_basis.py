"""Test basis for the paper option-hedging pipeline."""

from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence
import numpy as np
import torch

CODES_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = CODES_ROOT
for _source_directory in (
    CODES_ROOT,
    CODES_ROOT / "rl_envs",
    CODES_ROOT / "datasets",
    CODES_ROOT / "data_processing",
):
    if str(_source_directory) not in sys.path:
        sys.path.insert(0, str(_source_directory))
from rl_utils import evaluate_hedging_agent
from option_dataset import OptionDataset
from hedging_env import HedgingEnv, StateNormalizer, get_state_normalizer_from_json
from train_basis import (
    BASIS_CONFIG_SCHEMA_VERSION,
    _validate_basis_config,
    _write_json_atomic,
    get_basis_datasets,
    get_code_fingerprints,
    get_dataset_record,
    get_contract_multiplier,
    get_currency,
    get_datetime_record,
    get_default_basis_config,
    get_file_sha256,
    get_hedging_agent,
    get_safe_path_component,
    get_system_info,
)
from data_source import VALID_DATA_SOURCES, validate_data_source


def get_completed_train_run_configs(
    train_results_root: str | Path,
    *,
    experiment_id: str | None = None,
    run_ids: Sequence[str] | None = None,
) -> list[tuple[Path, dict[str, Any]]]:
    """Return completed train run configs."""
    root = Path(train_results_root)
    if not root.is_absolute():
        root = REPOSITORY_ROOT / root
    if not root.is_dir():
        raise FileNotFoundError(f"The training result directory does not exist: {root}")
    paths = sorted(root.glob("*/*/run_config.json"))
    records: list[tuple[Path, dict[str, Any]]] = []
    run_id_set = set(run_ids or ())
    for path in paths:
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("status") != "completed":
            continue
        if experiment_id is not None and record.get("experiment_id") != experiment_id:
            continue
        if run_id_set and record.get("run_id") not in run_id_set:
            continue
        model_name = record.get("artifacts", {}).get("model")
        if not model_name or not (path.parent / model_name).is_file():
            raise FileNotFoundError(f"The training record has no valid model file: {path}")
        records.append((path, record))
    if not records:
        raise FileNotFoundError("No matching completed baseline training run was found")
    if experiment_id is None:
        latest_experiment = max(records, key=lambda item: item[1]["run_completed_at"]["utc"])[1][
            "experiment_id"
        ]
        records = [item for item in records if item[1]["experiment_id"] == latest_experiment]
    if run_id_set:
        found = {record["run_id"] for _, record in records}
        missing = run_id_set - found
        if missing:
            raise FileNotFoundError(f"The requested completed run was not found: {sorted(missing)}")
    return records


def _get_shared_requested_config(
    train_records: Sequence[tuple[Path, Mapping[str, Any]]]
) -> dict[str, Any]:
    """Return shared requested config."""
    first = train_records[0][1]["requested_config"]
    canonical = json.dumps(first, ensure_ascii=False, sort_keys=True)
    for path, record in train_records[1:]:
        candidate = json.dumps(record["requested_config"], ensure_ascii=False, sort_keys=True)
        if candidate != canonical:
            raise ValueError(f"The selected runs have different requested_config values: {path}")
    config = json.loads(canonical)
    _validate_basis_config(config)
    return config


def get_test_normalizer(
    run_config_path: Path, run_record: Mapping[str, Any]
) -> StateNormalizer | None:
    """Return test normalizer."""
    relative_path = run_record["shared_artifacts"].get("state_normalizer")
    if relative_path is None:
        if run_record.get("state_normalizer") is not None:
            raise ValueError("The training record declares a normalizer but has no saved path")
        return None
    experiment_dir = run_config_path.parent.parent
    normalizer_path = experiment_dir / relative_path
    if not normalizer_path.is_file():
        raise FileNotFoundError(f"State normalizer does not exist: {normalizer_path}")
    normalizer = get_state_normalizer_from_json(normalizer_path)
    if normalizer.get_dict() != run_record["state_normalizer"]:
        raise ValueError("The saved state normalizer differs from the training run_config")
    return normalizer


def get_test_env(
    test_dataset: OptionDataset,
    normalizer: StateNormalizer | None,
    config: Mapping[str, Any],
    run_spec: Mapping[str, Any],
) -> HedgingEnv:
    """Return test env."""
    environment = config["environment"]
    return HedgingEnv(
        test_dataset,
        reward_formulation=run_spec["reward_formulation"],
        hedge_cost_rate=environment["hedge_cost_rate"],
        contract_multiplier=get_contract_multiplier(config),
        currency=get_currency(config),
        reconciliation_tolerance=environment["reconciliation_tolerance"],
        risk_aversion_lambda=environment["risk_aversion_lambda"],
        reward_risk_aversion_xi=environment["reward_risk_aversion_xi"],
        training_reward_scale=run_spec["training_reward_scale"],
        state_normalizer=normalizer,
        num_parallel_episode=environment["num_parallel_test_episode"],
        is_clip_action=environment["is_clip_action"],
        is_record_trajectory=environment["is_record_test_trajectory"],
        observation_dtype=np.dtype(environment["observation_dtype"]),
    )


def _validate_train_test_compatibility(
    run_record: Mapping[str, Any], agent: Any, test_env: HedgingEnv
) -> None:
    """Validate train test compatibility."""
    resolved = run_record["resolved_run"]
    if agent.algorithm_name != resolved["algorithm_name"]:
        raise ValueError("The loaded agent algorithm differs from resolved_run")
    if agent.is_delta_residual != resolved["is_delta_residual"]:
        raise ValueError("The loaded agent Delta-residual settings differ from training")
    if test_env.reward_formulation != resolved["reward_formulation"]:
        raise ValueError("test reward_formulation differs from training")
    train_env = run_record["train_environment"]
    for key in (
        "hedge_cost_rate",
        "contract_multiplier",
        "currency",
        "reconciliation_tolerance",
        "risk_aversion_lambda",
        "reward_risk_aversion_xi",
        "training_reward_scale",
        "state_feature_names",
    ):
        test_value = test_env.get_config()[key]
        train_value = train_env.get(key, "USD" if key == "currency" else None)
        if test_value != train_value:
            raise ValueError(f"Test environment field {key} differs from the training environment")


def _save_test_frames(
    run_dir: Path, evaluation: Mapping[str, Any], *, is_save_test_trajectory: bool
) -> dict[str, str]:
    """Save test frames."""
    episode_path = run_dir / "test_episode_results.parquet"
    evaluation["episode_results"].to_parquet(episode_path, index=False)
    artifacts = {"test_episode_results": episode_path.name}
    if is_save_test_trajectory:
        trajectory_path = run_dir / "test_trajectory.parquet"
        evaluation["trajectory"].to_parquet(trajectory_path, index=False)
        artifacts["test_trajectory"] = trajectory_path.name
    return artifacts


def run_basis_testing(
    train_records: Sequence[tuple[Path, Mapping[str, Any]]],
    *,
    test_results_root: str | Path | None = None,
    device: str | torch.device | None = None,
    is_overwrite: bool = False,
    data_source: str | None = None,
) -> dict[str, Any]:
    """Run basis testing."""
    if not train_records:
        raise ValueError("train_records must not be empty")
    config = _get_shared_requested_config(train_records)
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
        _validate_basis_config(config)
    print(f"[Data source] data_source={config.get('data_source', 'spx')}", flush=True)
    test_dataset = get_basis_datasets(config, labels=("test",))["test"]
    test_dataset_record = get_dataset_record(test_dataset)
    experiment_id = str(train_records[0][1]["experiment_id"])
    get_safe_path_component(experiment_id, "experiment_id")
    output_root_value = (
        test_results_root if test_results_root is not None else config["testing"]["output_root"]
    )
    output_root = Path(output_root_value)
    if not output_root.is_absolute():
        output_root = REPOSITORY_ROOT / output_root
    experiment_dir = output_root / experiment_id
    experiment_dir.mkdir(parents=True, exist_ok=True)
    started_at = get_datetime_record()
    summaries: list[dict[str, Any]] = []
    for num_run, (run_config_path, run_record) in enumerate(train_records, start=1):
        run_id = str(run_record["run_id"])
        get_safe_path_component(run_id, "run_id")
        run_dir = experiment_dir / run_id
        if run_dir.exists() and any(run_dir.iterdir()) and (not is_overwrite):
            raise FileExistsError(
                f"The test run directory already exists and is nonempty: {run_dir}"
            )
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[Evaluation {num_run}/{len(train_records)}] {run_id}", flush=True)
        num_start_time = time.perf_counter()
        test_started_at = get_datetime_record()
        run_spec = run_record["resolved_run"]
        normalizer = get_test_normalizer(run_config_path, run_record)
        selected_device = device if device is not None else config["testing"]["device"]
        agent = get_hedging_agent(config, run_spec, device=selected_device)
        model_path = run_config_path.parent / run_record["artifacts"]["model"]
        model_selection = run_record.get("model_selection")
        if (
            run_record["artifacts"].get("model_role") != "validation_best"
            or not isinstance(model_selection, Mapping)
            or model_selection.get("selected_model") != model_path.name
            or (model_selection.get("metric") != "j_lambda")
            or (model_selection.get("mode") != "min")
        ):
            raise ValueError(
                "The checkpoint is not marked as validation-best; retrain with this version before testing"
            )
        model_sha256 = get_file_sha256(model_path)
        if model_sha256 != run_record.get("model_sha256"):
            raise ValueError(f"The model SHA-256 differs from the training record: {model_path}")
        agent.load(model_path, is_load_optimizer=False)
        test_env = get_test_env(test_dataset, normalizer, config, run_spec)
        _validate_train_test_compatibility(run_record, agent, test_env)
        evaluation = evaluate_hedging_agent(
            agent, test_env, is_print_summary=config["testing"]["is_print_summary"]
        )
        artifacts = _save_test_frames(
            run_dir,
            evaluation,
            is_save_test_trajectory=config["testing"]["is_save_test_trajectory"],
        )
        test_completed_at = get_datetime_record()
        metrics_record = {
            "schema_version": BASIS_CONFIG_SCHEMA_VERSION,
            "data_source": config.get("data_source", "spx"),
            "experiment_id": experiment_id,
            "run_id": run_id,
            "test_completed_at": test_completed_at,
            "metrics": evaluation["metrics"],
        }
        _write_json_atomic(run_dir / "test_metrics.json", metrics_record)
        test_record = {
            "schema_version": BASIS_CONFIG_SCHEMA_VERSION,
            "data_source": config.get("data_source", "spx"),
            "status": "completed",
            "experiment_id": experiment_id,
            "run_id": run_id,
            "test_started_at": test_started_at,
            "test_completed_at": test_completed_at,
            "num_elapsed_second": time.perf_counter() - num_start_time,
            "training_run_config_path": str(run_config_path),
            "training_run_config_sha256": get_file_sha256(run_config_path),
            "training_completed_at": run_record["run_completed_at"],
            "training_run": run_record,
            "test_dataset": test_dataset_record,
            "test_environment": test_env.get_config(),
            "agent": agent.get_config(),
            "model_sha256": model_sha256,
            "model_selection": model_selection,
            "metrics": evaluation["metrics"],
            "num_test_reset": evaluation["num_reset"],
            "num_test_step": evaluation["num_total_step"],
            "artifacts": {
                **artifacts,
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
                "test_completed_at": test_completed_at,
            }
        )
    summary = {
        "status": "completed",
        "data_source": config.get("data_source", "spx"),
        "experiment_id": experiment_id,
        "started_at": started_at,
        "completed_at": get_datetime_record(),
        "num_run": len(summaries),
        "test_dataset": test_dataset_record,
        "runs": summaries,
    }
    _write_json_atomic(experiment_dir / "test_experiment_summary.json", summary)
    print(f"\n[Test experiment completed] {experiment_dir}", flush=True)
    return {"experiment_dir": experiment_dir, "summary": summary}


def get_argument_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    defaults = get_default_basis_config()
    parser = argparse.ArgumentParser(
        description="Evaluate completed baseline option-hedging training runs"
    )
    parser.add_argument(
        "--train-results-root",
        type=Path,
        default=Path(defaults["training"]["output_root"]),
        help="Baseline training result root",
    )
    parser.add_argument(
        "--test-results-root",
        type=Path,
        default=None,
        help="Override the test output root saved in the training configuration",
    )
    parser.add_argument(
        "--experiment-id",
        help="Training experiment; defaults to the most recently completed experiment",
    )
    parser.add_argument(
        "--run-id", action="append", help="Evaluate only the specified run; may be repeated"
    )
    parser.add_argument("--device", help="Override the test device, e.g. cpu or cuda")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting test results for the same experiment/run",
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
    records = get_completed_train_run_configs(
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
    run_basis_testing(
        records,
        test_results_root=args.test_results_root,
        device=args.device,
        is_overwrite=args.overwrite,
        data_source=args.data_source,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
