"""Pipeline for the paper option-hedging pipeline."""

from __future__ import annotations
import hashlib
import json
import os
import platform
import re
import sys
import time
import uuid
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
import numpy as np
import pandas as pd
import scipy
from option_dataset import OptionDataset, get_option_dataset_config
from option_dataset.constants import CANDIDATE_AUDIT_COLUMNS
from market_segmentation import get_seg_periods_from_result, segment_option_dataset
from .calibration import calibrate_segmented_sabr, get_global_calibration_baseline
from .config import SimulationConfig
from .generation import generate_segmented_simulations
from .quality import compare_segmented_and_global_simulations

SIMULATION_SCHEMA_VERSION = 1
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SIMULATION_RESULT_ROOT = PACKAGE_ROOT / "sim_results"
SIMULATION_RESULT_NAME_PATTERN = re.compile("^sim_(?P<timestamp>\\d{8}T\\d{6}Z)$")


def _get_datetime_record() -> dict[str, str]:
    """Return the current UTC timestamp."""
    now = datetime.now(timezone.utc)
    return {"utc": now.isoformat(timespec="seconds")}


def _get_json_value(value: Any) -> Any:
    """Return JSON value."""
    if isinstance(value, Mapping):
        return {str(key): _get_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_get_json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (pd.Timestamp, datetime, np.datetime64)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, np.ndarray):
        return [_get_json_value(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _get_json_value(value.item())
    if isinstance(value, float) and (not np.isfinite(value)):
        return None
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write JSON."""
    path.write_text(
        json.dumps(
            _get_json_value(payload), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
        ),
        encoding="utf-8",
    )


def _get_file_sha256(path: Path) -> str:
    """Compute a file SHA-256 digest using bounded-memory reads."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _get_code_fingerprints() -> list[dict[str, Any]]:
    """Return code fingerprints."""
    paths = [
        PACKAGE_ROOT.parent / "run_sim.py",
        *sorted((PACKAGE_ROOT / "market_simulation").glob("*.py")),
        PACKAGE_ROOT.parent / "segmentation" / "market_segmentation" / "dataset.py",
        PACKAGE_ROOT.parent / "segmentation" / "market_segmentation" / "pipeline.py",
        PACKAGE_ROOT.parent / "datasets" / "option_dataset" / "dataset.py",
        PACKAGE_ROOT.parent / "datasets" / "option_dataset" / "pricing.py",
        PACKAGE_ROOT.parent / "rl_envs" / "hedging_env" / "environment.py",
    ]
    repository_root = PACKAGE_ROOT.parent
    records = []
    for path in paths:
        if path.is_file():
            records.append(
                {
                    "path": str(path.relative_to(repository_root)),
                    "num_bytes": path.stat().st_size,
                    "sha256": _get_file_sha256(path),
                }
            )
    return records


def _get_system_info() -> dict[str, Any]:
    """Record software versions without account, hostname, or repository identity."""
    try:
        import pyarrow

        pyarrow_version = pyarrow.__version__
    except ImportError:
        pyarrow_version = None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "pyarrow": pyarrow_version,
    }


def _get_episode_id_sha256(dataset: OptionDataset) -> str:
    """Return episode id sha256."""
    encoded = "\n".join(dataset.get_episode_ids()).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_source_dataset(
    dataset: OptionDataset, segmentation_result: Mapping[str, Any]
) -> None:
    """Validate source dataset."""
    source = segmentation_result.get("dataset")
    if not isinstance(source, Mapping):
        raise ValueError("The segmentation result is missing its dataset record")
    expected_config = source.get("config")
    if not isinstance(expected_config, Mapping):
        raise ValueError("The segmentation result is missing its dataset.config record")
    expected_semantic_config = dict(expected_config)
    actual_semantic_config = dataset.config.get_dict()
    expected_semantic_config.pop("data_root", None)
    actual_semantic_config.pop("data_root", None)
    if _get_json_value(expected_semantic_config) != _get_json_value(actual_semantic_config):
        raise ValueError("The current training dataset config differs from the Stage 1 record")
    if int(source.get("num_episode", -1)) != dataset.num_episode:
        raise ValueError("The current training episode count differs from the Stage 1 record")
    if source.get("episode_ids_sha256") != _get_episode_id_sha256(dataset):
        raise ValueError("The current training episode identities differ from the Stage 1 record")


def _validate_calibration_dataset(
    dataset: OptionDataset, calibration_dataset: OptionDataset, config: SimulationConfig
) -> None:
    """Ensure calibration data differs only in the two declared selectors."""
    if calibration_dataset.label != "train":
        raise TypeError("calibration_dataset must have label='train'")
    expected = dataset.config.get_dict()
    expected["expiry_weekday"] = config.expiry_weekday_calibration
    expected["num_moneyness"] = config.num_moneyness_calibration
    actual = calibration_dataset.config.get_dict()
    if _get_json_value(expected) != _get_json_value(actual):
        raise ValueError(
            "calibration_dataset config may differ from the generation dataset only in expiry_weekday and num_moneyness"
        )
    if calibration_dataset.num_episode == 0:
        raise ValueError("calibration_dataset must contain at least one episode")


def _get_generation_pairing_diagnostics(
    segmented_manifest: pd.DataFrame, global_manifest: pd.DataFrame
) -> dict[str, Any]:
    """Report whether both generators used identical anchors and random seeds."""
    sort_columns = ["segment_id", "simulation_id"]
    left = segmented_manifest.sort_values(sort_columns).reset_index(drop=True)
    right = global_manifest.sort_values(sort_columns).reset_index(drop=True)
    identity_columns = ["segment_id", "simulation_id"]
    anchor_columns = [*identity_columns, "anchor_episode_id", "anchor_cohort_id"]
    seed_columns = [*anchor_columns, "simulation_seed", "num_path_retry"]

    def equal(columns: list[str]) -> bool:
        if len(left) != len(right):
            return False
        left_values = left.loc[:, columns].astype(str).to_numpy()
        right_values = right.loc[:, columns].astype(str).to_numpy()
        return bool(np.array_equal(left_values, right_values))

    return {
        "num_segmented_cohort": int(len(left)),
        "num_global_baseline_cohort": int(len(right)),
        "is_identity_paired": equal(identity_columns),
        "is_anchor_paired": equal(anchor_columns),
        "is_seed_and_retry_paired": equal(seed_columns),
    }


def _get_safe_experiment_id(experiment_id: str) -> str:
    """Return safe experiment id."""
    value = str(experiment_id)
    if (
        not value
        or value in {".", ".."}
        or Path(value).name != value
        or ("/" in value)
        or ("\\" in value)
    ):
        raise ValueError("experiment_id must be a nonempty directory name without path separators")
    return value


def _get_artifact_records(directory: Path) -> list[dict[str, Any]]:
    """Return artifact records."""
    records = []
    for path in sorted(directory.iterdir(), key=lambda item: item.name):
        if path.is_file() and path.name != "sim_result.json":
            records.append(
                {
                    "filename": path.name,
                    "num_bytes": path.stat().st_size,
                    "sha256": _get_file_sha256(path),
                }
            )
    return records


def run_segmented_sabr_simulation(
    dataset: OptionDataset,
    segmentation_result: Mapping[str, Any],
    config: SimulationConfig,
    *,
    calibration_dataset: OptionDataset | None = None,
    segmentation_result_path: str | Path | None = None,
    output_root: str | Path = DEFAULT_SIMULATION_RESULT_ROOT,
    experiment_id: str | None = None,
    progress_callback: Callable[[str, Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Calibrate regime-specific and global SABR models, generate paired cohorts, evaluate fidelity, and atomically persist the completed run."""
    if not isinstance(dataset, OptionDataset) or dataset.label != "train":
        raise TypeError("dataset must be an OptionDataset with label='train'")
    if not isinstance(segmentation_result, Mapping):
        raise TypeError("segmentation_result must be a Mapping")
    if not isinstance(config, SimulationConfig):
        raise TypeError("config must be a SimulationConfig")
    seg_periods = get_seg_periods_from_result(segmentation_result)
    _validate_source_dataset(dataset, segmentation_result)
    if calibration_dataset is None:
        calibration_dataset = dataset
        calibration_dataset_source = "generation_dataset_compatibility_default"
    else:
        if not isinstance(calibration_dataset, OptionDataset):
            raise TypeError("calibration_dataset must be an OptionDataset")
        _validate_calibration_dataset(dataset, calibration_dataset, config)
        calibration_dataset_source = "explicit_calibration_overrides"
    if experiment_id is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        experiment_id = f"sim_{timestamp}"
    experiment_id = _get_safe_experiment_id(experiment_id)
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    result_directory = root / experiment_id
    if result_directory.exists():
        raise FileExistsError(f"The Stage 2 result directory already exists: {result_directory}")
    temporary_directory = root / f".{experiment_id}.{uuid.uuid4().hex}.tmp"
    temporary_directory.mkdir(parents=False, exist_ok=False)
    result_json = temporary_directory / "sim_result.json"
    start_time = time.perf_counter()
    started_at = _get_datetime_record()
    source_path = Path(segmentation_result_path) if segmentation_result_path is not None else None
    running = {
        "schema_version": SIMULATION_SCHEMA_VERSION,
        "status": "running",
        "experiment_id": experiment_id,
        "started_at": started_at,
        "simulation_config": config.get_dict(),
        "segmentation_source": {
            "experiment_id": segmentation_result.get("experiment_id"),
            "path": source_path,
            "sha256": (
                _get_file_sha256(source_path)
                if source_path is not None and source_path.is_file()
                else None
            ),
            "seg_periods": seg_periods,
        },
        "dataset": {
            "config": dataset.config.get_dict(),
            "num_episode": dataset.num_episode,
            "num_step": dataset.num_step,
            "num_cohort": int(dataset.episode_manifest["cohort_id"].nunique()),
            "episode_ids_sha256": _get_episode_id_sha256(dataset),
        },
        "calibration_dataset": {
            "source": calibration_dataset_source,
            "config": calibration_dataset.config.get_dict(),
            "num_episode": calibration_dataset.num_episode,
            "num_step": calibration_dataset.num_step,
            "num_cohort": int(calibration_dataset.episode_manifest["cohort_id"].nunique()),
            "episode_ids_sha256": _get_episode_id_sha256(calibration_dataset),
        },
        "system": _get_system_info(),
        "code_fingerprints": _get_code_fingerprints(),
    }
    _write_json(result_json, running)
    try:
        real_datasets = segment_option_dataset(dataset, seg_periods)
        if progress_callback is not None:
            progress_callback(
                "segmentation_loaded",
                {
                    "num_segment": len(real_datasets),
                    "num_episode": [item.num_episode for item in real_datasets],
                },
            )
        calibration = calibrate_segmented_sabr(calibration_dataset, seg_periods, config)
        global_calibration = get_global_calibration_baseline(calibration, config)
        if progress_callback is not None:
            progress_callback(
                "calibration_completed",
                {
                    "segment_parameters": calibration.segment_parameters.to_dict(orient="records"),
                    "global_parameters": global_calibration.segment_parameters.iloc[0].to_dict(),
                    "diagnostics": calibration.diagnostics,
                },
            )
        generation = generate_segmented_simulations(real_datasets, calibration, config)
        if progress_callback is not None:
            progress_callback("segmented_generation_completed", generation.diagnostics)
        global_generation = generate_segmented_simulations(
            real_datasets,
            global_calibration,
            config,
            simulation_plan=generation.simulation_manifest,
        )
        if progress_callback is not None:
            progress_callback("global_generation_completed", global_generation.diagnostics)
        pairing = _get_generation_pairing_diagnostics(
            generation.simulation_manifest, global_generation.simulation_manifest
        )
        if not pairing["is_identity_paired"] or not pairing["is_anchor_paired"]:
            raise RuntimeError("global baseline generation is not paired with segmented anchors")
        if progress_callback is not None:
            progress_callback("generation_pairing_completed", pairing)
        quality = compare_segmented_and_global_simulations(
            real_datasets, generation.datasets, global_generation.datasets, config
        )
        if progress_callback is not None:
            progress_callback("quality_comparison_completed", quality)
        segment_artifacts: list[dict[str, Any]] = []
        for segment_id, simulated in enumerate(generation.datasets, start=1):
            filename = f"sim_segment_{segment_id:02d}.parquet"
            path = temporary_directory / filename
            simulated.episode_steps.to_parquet(path, index=False)
            segment_artifacts.append(
                {
                    "segment_id": segment_id,
                    "filename": filename,
                    "num_episode": simulated.num_episode,
                    "num_step": simulated.num_step,
                    "num_cohort": int(simulated.episode_manifest["cohort_id"].nunique()),
                }
            )
        calibration.cross_sections.to_parquet(
            temporary_directory / "sabr_cross_sections.parquet", index=False
        )
        calibration.segment_parameters.to_csv(
            temporary_directory / "sabr_segment_parameters.csv", index=False
        )
        calibration.optimization_runs.to_csv(
            temporary_directory / "sabr_optimization_runs.csv", index=False
        )
        generated_episode_manifest = pd.concat(
            [item.episode_manifest for item in generation.datasets], ignore_index=True
        )
        generated_episode_manifest.to_parquet(
            temporary_directory / "simulation_manifest.parquet", index=False
        )
        generation.simulation_manifest.to_parquet(
            temporary_directory / "simulation_cohort_manifest.parquet", index=False
        )
        artifacts = _get_artifact_records(temporary_directory)
        completed = {
            **running,
            "status": "completed",
            "completed_at": _get_datetime_record(),
            "num_elapsed_second": time.perf_counter() - start_time,
            "num_segment": len(real_datasets),
            "calibration": {
                "segment_parameters": calibration.segment_parameters.to_dict(orient="records"),
                "global_baseline_parameters": global_calibration.segment_parameters.to_dict(
                    orient="records"
                ),
                "diagnostics": calibration.diagnostics,
            },
            "generation": generation.diagnostics,
            "global_baseline_generation": {
                **global_generation.diagnostics,
                "samples_persisted": False,
                "pairing_with_segmented": pairing,
            },
            "quality_evaluation": quality["segmented"],
            "global_baseline_quality_evaluation": quality["global_baseline"],
            "quality_comparison": quality["comparison"],
            "segment_artifacts": segment_artifacts,
            "artifacts": artifacts,
        }
        _write_json(result_json, completed)
        os.replace(temporary_directory, result_directory)
        return {
            "result_directory": result_directory,
            "result_path": result_directory / "sim_result.json",
            "result": completed,
            "real_datasets": real_datasets,
            "simulated_datasets": list(generation.datasets),
        }
    except Exception as exc:
        failed = {
            **running,
            "status": "failed",
            "failed_at": _get_datetime_record(),
            "num_elapsed_second": time.perf_counter() - start_time,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }
        _write_json(result_json, failed)
        os.replace(temporary_directory, result_directory)
        raise


def resolve_simulation_result_directory(
    path: str | Path | None = None, *, result_root: str | Path = DEFAULT_SIMULATION_RESULT_ROOT
) -> Path:
    """Resolve simulation result directory."""
    root = Path(result_root)
    if path is not None:
        candidate = Path(path)
        if not candidate.is_absolute() and (not candidate.exists()):
            candidate = root / candidate
        directory = candidate.parent if candidate.is_file() else candidate
        if not (directory / "sim_result.json").is_file():
            raise FileNotFoundError(f"The Stage 2 result does not exist: {directory}")
        return directory
    candidates: list[tuple[str, Path]] = []
    if root.is_dir():
        for directory in root.iterdir():
            match = SIMULATION_RESULT_NAME_PATTERN.fullmatch(directory.name)
            result_path = directory / "sim_result.json"
            if not directory.is_dir() or match is None or (not result_path.is_file()):
                continue
            try:
                payload = json.loads(result_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(payload, dict) and payload.get("status") == "completed":
                candidates.append((match.group("timestamp"), directory))
    if not candidates:
        raise FileNotFoundError(f"No completed standard Stage 2 result directory was found: {root}")
    return max(candidates, key=lambda item: item[0])[1]


def load_simulation_result(
    path: str | Path | None = None, *, result_root: str | Path = DEFAULT_SIMULATION_RESULT_ROOT
) -> dict[str, Any]:
    """Load simulation result."""
    directory = resolve_simulation_result_directory(path, result_root=result_root)
    result_path = directory / "sim_result.json"
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read the Stage 2 result: {result_path}") from exc
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != SIMULATION_SCHEMA_VERSION
        or payload.get("status") != "completed"
    ):
        raise ValueError("The Stage 2 schema or completion status is invalid")
    return payload


def _get_dataset_config_from_record(record: Mapping[str, Any]) -> Any:
    """Return dataset config from record."""
    values = dict(record)
    date_period = (values.pop("date_start"), values.pop("date_end"))
    label = values.pop("label")
    moneyness_range = (values.pop("moneyness_lower"), values.pop("moneyness_upper"))
    return get_option_dataset_config(
        date_period=date_period, label=label, moneyness_range=moneyness_range, **values
    )


def _get_empty_candidate_audit() -> pd.DataFrame:
    """Return empty candidate audit."""
    return pd.DataFrame({column: pd.Series(dtype="object") for column in CANDIDATE_AUDIT_COLUMNS})


def load_simulated_segment_dataset(path: str | Path, segment_id: int) -> OptionDataset:
    """Load simulated segment dataset."""
    if isinstance(segment_id, bool) or int(segment_id) != segment_id or segment_id < 1:
        raise ValueError("segment_id must be a positive integer")
    directory = resolve_simulation_result_directory(path)
    result = load_simulation_result(directory)
    artifact = next(
        (
            record
            for record in result["segment_artifacts"]
            if int(record["segment_id"]) == int(segment_id)
        ),
        None,
    )
    if artifact is None:
        raise KeyError(f"No result entry exists for segment_id={segment_id}")
    steps = pd.read_parquet(directory / artifact["filename"])
    manifest = pd.read_parquet(directory / "simulation_manifest.parquet")
    simulation_ids = set(steps["simulation_id"].drop_duplicates().astype(str).tolist())
    manifest = manifest.loc[manifest["simulation_id"].astype(str).isin(simulation_ids)].copy()
    config = _get_dataset_config_from_record(result["dataset"]["config"])
    dataset = OptionDataset(
        label=config.label,
        config=config,
        episode_steps=steps,
        episode_manifest=manifest,
        candidate_selection_audit=_get_empty_candidate_audit(),
        dataset_build_report={
            "source": "load_simulated_segment_dataset",
            "simulation_result": str(directory),
            "segment_id": int(segment_id),
        },
    )
    if dataset.num_episode != int(artifact["num_episode"]):
        raise ValueError("The loaded simulation episode count differs from the result manifest")
    return dataset


def load_segmented_simulated_datasets(path: str | Path) -> list[OptionDataset]:
    """Load segmented simulated datasets."""
    directory = resolve_simulation_result_directory(path)
    result = load_simulation_result(directory)
    return [
        load_simulated_segment_dataset(directory, int(record["segment_id"]))
        for record in result["segment_artifacts"]
    ]
