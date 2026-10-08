"""Run sim for the paper option-hedging pipeline."""

from __future__ import annotations
import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Mapping, Sequence

CODES_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = CODES_ROOT
for _source_directory in (
    CODES_ROOT,
    CODES_ROOT / "simulation",
    CODES_ROOT / "segmentation",
    CODES_ROOT / "datasets",
    CODES_ROOT / "data_processing",
    CODES_ROOT / "rl_envs",
):
    if str(_source_directory) not in sys.path:
        sys.path.insert(0, str(_source_directory))
from option_dataset import OptionDataset, get_option_dataset
from market_segmentation import (
    get_seg_periods_from_result,
    load_seg_result,
    resolve_seg_result_path,
)
from market_simulation import SimulationConfig, run_segmented_sabr_simulation
from data_source import VALID_DATA_SOURCES, resolve_data_root, validate_data_source

SIM_RUN_SCHEMA_VERSION = 1
_REQUIRED_MARKET_DATA_FILES = ("spx_spot.parquet", "zero_curve.parquet", "spx_div_yield.parquet")


def get_default_sim_run_config() -> dict[str, Any]:
    """Return default sim run config."""
    return {
        "schema_version": SIM_RUN_SCHEMA_VERSION,
        "data_source": "spx",
        "segmentation": {"result_root": "segmentation/seg_results", "result_file": None},
        "simulation": SimulationConfig().get_dict(),
        "output": {"output_root": "simulation/sim_results"},
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
                raise ValueError(f"{path}.{key} must be a JSON object")
            result[key] = _merge_config_strict(default[key], value, path=f"{path}.{key}")
        else:
            result[key] = deepcopy(value)
    return result


def _parse_override_value(text: str) -> Any:
    """Parse JSON values, falling back to a plain string."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _parse_optional_weekday(text: str) -> int | None:
    """Parse optional weekday."""
    if text.strip().lower() in {"none", "null"}:
        return None
    try:
        return int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Use an integer in [0, 6], or None/null to include all expiry weekdays"
        ) from exc


def _set_config_leaf(config: dict[str, Any], expression: str) -> None:
    """Apply a validated dotted-path configuration override."""
    if "=" not in expression:
        raise ValueError(f"--set must use SECTION.KEY=VALUE: {expression!r}")
    path_text, value_text = expression.split("=", maxsplit=1)
    keys = [key.strip() for key in path_text.split(".")]
    if not keys or any((not key for key in keys)):
        raise ValueError(f"Invalid --set path: {path_text!r}")
    target: Any = config
    for key in keys[:-1]:
        if not isinstance(target, dict) or key not in target:
            raise ValueError(f"Unknown --set configuration path: {path_text!r}")
        target = target[key]
    leaf = keys[-1]
    if not isinstance(target, dict) or leaf not in target:
        raise ValueError(f"Unknown --set configuration path: {path_text!r}")
    if isinstance(target[leaf], Mapping):
        raise ValueError("--set supports leaves only; use --config for objects")
    target[leaf] = _parse_override_value(value_text)


def _resolve_repository_path(path: str | Path) -> Path:
    """Resolve repository path."""
    result = Path(path)
    return result if result.is_absolute() else REPOSITORY_ROOT / result


def _get_persisted_path_parts(path: str | Path) -> tuple[tuple[str, ...], bool]:
    """Return persisted path parts."""
    text = str(path)
    windows_path = PureWindowsPath(text)
    posix_path = PurePosixPath(text)
    if windows_path.is_absolute():
        return (tuple(windows_path.parts[1:]), True)
    if posix_path.is_absolute():
        return (tuple(posix_path.parts[1:]), True)
    if "\\" in text:
        return (tuple(windows_path.parts), False)
    return (tuple(posix_path.parts), False)


def _is_market_data_root(path: Path) -> bool:
    """Return whether market data root."""
    return path.is_dir() and all(
        ((path / filename).is_file() for filename in _REQUIRED_MARKET_DATA_FILES)
    )


def _resolve_persisted_data_root(path: str | Path) -> Path:
    """Resolve persisted data root."""
    native_path = Path(path)
    portable_parts, is_persisted_absolute = _get_persisted_path_parts(path)
    if native_path.is_absolute() and native_path.is_dir():
        return native_path
    if not is_persisted_absolute:
        return REPOSITORY_ROOT.joinpath(*portable_parts)
    checked: list[Path] = []
    for start in range(len(portable_parts) - 1, -1, -1):
        candidate = REPOSITORY_ROOT.joinpath(*portable_parts[start:])
        checked.append(candidate)
        if _is_market_data_root(candidate):
            return candidate
    checked_text = ", ".join((str(candidate) for candidate in checked))
    raise FileNotFoundError(
        f"The saved Stage 1 data_root is unavailable and cannot be relocated within the current project: persisted={path!s}; checked=[{checked_text}]"
    )


def _validate_sim_run_config(config: Mapping[str, Any]) -> None:
    """Validate sim run config."""
    if config.get("schema_version") != SIM_RUN_SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SIM_RUN_SCHEMA_VERSION}")
    validate_data_source(config.get("data_source", "spx"))
    segmentation = config.get("segmentation")
    if not isinstance(segmentation, Mapping):
        raise ValueError("segmentation must be an object")
    result_root = segmentation.get("result_root")
    result_file = segmentation.get("result_file")
    if not isinstance(result_root, (str, Path)) or not str(result_root):
        raise ValueError("segmentation.result_root must be a nonempty path")
    if result_file is not None and (
        not isinstance(result_file, (str, Path)) or not str(result_file)
    ):
        raise ValueError("segmentation.result_file must be null or a nonempty path")
    SimulationConfig(**dict(config["simulation"]))
    output_root = config["output"]["output_root"]
    if not isinstance(output_root, (str, Path)) or not str(output_root):
        raise ValueError("output.output_root must be a nonempty path")


def get_sim_run_config(
    config_path: str | Path | None = None, *, set_overrides: Sequence[str] = ()
) -> dict[str, Any]:
    """Return sim run config."""
    config = get_default_sim_run_config()
    if config_path is not None:
        path = Path(config_path)
        try:
            override = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Cannot read the configuration JSON: {path}") from exc
        if not isinstance(override, Mapping):
            raise ValueError("The configuration root must be a JSON object")
        config = _merge_config_strict(config, override)
    for expression in set_overrides:
        _set_config_leaf(config, expression)
    _validate_sim_run_config(config)
    return config


def get_train_dataset_from_seg_result(
    segmentation_result: Mapping[str, Any],
    *,
    expiry_weekday: int | None | object = ...,
    num_moneyness: int | object = ...,
    data_source: str = "spx",
) -> OptionDataset:
    """Return train dataset from seg result."""
    dataset_record = segmentation_result.get("dataset")
    if not isinstance(dataset_record, Mapping) or not isinstance(
        dataset_record.get("config"), Mapping
    ):
        raise ValueError("The Stage 1 result is missing dataset.config")
    values = deepcopy(dict(dataset_record["config"]))
    if expiry_weekday is not ...:
        values["expiry_weekday"] = expiry_weekday
    if num_moneyness is not ...:
        values["num_moneyness"] = num_moneyness
    date_period = (values.pop("date_start"), values.pop("date_end"))
    label = values.pop("label")
    moneyness_range = (values.pop("moneyness_lower"), values.pop("moneyness_upper"))
    try:
        persisted_root = _resolve_persisted_data_root(values["data_root"])
    except FileNotFoundError:
        if validate_data_source(data_source) != "sx5e":
            raise
        persisted_root = Path(values["data_root"])
    values["data_root"] = resolve_data_root(data_source, persisted_root)
    return get_option_dataset(date_period, label=label, moneyness_range=moneyness_range, **values)


def _get_argument_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Load Stage 1 results and run regime-specific SABR calibration, simulation, and fidelity evaluation."
    )
    parser.add_argument("--config", type=Path, help="Optional partial JSON configuration override")
    parser.add_argument(
        "--set",
        dest="set_overrides",
        action="append",
        default=[],
        metavar="SECTION.KEY=VALUE",
        help="Override any configuration leaf; may be repeated",
    )
    parser.add_argument(
        "--seg-result",
        type=str,
        help="Stage 1 JSON name or path; defaults to the latest timestamped result",
    )
    parser.add_argument("--seg-result-root", type=str)
    parser.add_argument("--num-simulation-times", type=float)
    parser.add_argument(
        "--expiry-weekday-calibration",
        type=_parse_optional_weekday,
        default=argparse.SUPPRESS,
        metavar="{0,...,6,None}",
        help="Calibration expiry weekday; None/null includes all weekdays (default)",
    )
    parser.add_argument("--num-moneyness-calibration", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-path-retry", type=int)
    parser.add_argument("--environment-rollout-checks", type=int)
    parser.add_argument("--output-root", type=str)
    parser.add_argument(
        "--experiment-id",
        type=str,
        help="Optional result directory name; defaults to sim_YYYYMMDDTHHMMSSZ",
    )
    parser.add_argument(
        "--print-config-only",
        action="store_true",
        help="Print the resolved run configuration without reading data",
    )
    parser.add_argument(
        "--data-source",
        choices=VALID_DATA_SOURCES,
        default=None,
        help="data source; sx5e triggers canonical preprocessing",
    )
    return parser


def _apply_cli_overrides(overrides: Sequence[str], args: argparse.Namespace) -> list[str]:
    """Apply CLI overrides."""
    results = list(overrides)
    aliases = (
        ("seg_result", "segmentation.result_file"),
        ("seg_result_root", "segmentation.result_root"),
        ("num_simulation_times", "simulation.num_simulation_times"),
        ("expiry_weekday_calibration", "simulation.expiry_weekday_calibration"),
        ("num_moneyness_calibration", "simulation.num_moneyness_calibration"),
        ("seed", "simulation.random_seed"),
        ("max_path_retry", "simulation.max_path_retry"),
        ("environment_rollout_checks", "simulation.num_environment_rollout_checks"),
        ("output_root", "output.output_root"),
    )
    for attribute, path in aliases:
        if not hasattr(args, attribute):
            continue
        value = getattr(args, attribute)
        if value is not None:
            results.append(f"{path}={json.dumps(value)}")
        elif attribute == "expiry_weekday_calibration":
            results.append(f"{path}=null")
    return results


def _print_progress(event: str, payload: Mapping[str, Any]) -> None:
    """Report pipeline status and only the four paper fidelity features."""
    if event == "segmentation_loaded":
        print(
            f"[Stage 2] Regimes={payload['num_segment']}; real episodes={payload['num_episode']}",
            flush=True,
        )
    elif event == "calibration_completed":
        parameters = payload["global_parameters"]
        print(
            f"[Stage 2] Calibration completed; global rho={parameters['rho']:.6g}, nu={parameters['nu']:.6g}",
            flush=True,
        )
    elif event in {"segmented_generation_completed", "global_generation_completed"}:
        print(
            f"[Stage 2] {event}: episodes={payload['num_simulated_episode']}, cohorts={payload['num_simulated_cohort']}",
            flush=True,
        )
    elif event == "generation_pairing_completed":
        print(f"[Stage 2] Paired anchors={payload['is_anchor_paired']}", flush=True)
    elif event == "quality_comparison_completed":
        print("[Stage 2] Normalized Wasserstein distance (lower is better):", flush=True)
        for record in payload["comparison"]["features"]:
            print(
                f"  {record['feature']}: regime-specific={record['segmented']:.6g}, global={record['global_baseline']:.6g}",
                flush=True,
            )


def _print_result_summary(output: Mapping[str, Any]) -> None:
    """Print result summary."""
    result = output["result"]
    calibration = result["calibration"]["diagnostics"]
    print("\n[Stage 2 completed]", flush=True)
    print(
        f"Global fallback tasks={calibration['num_global_fallback_segment']}/{result['num_segment']}, Regime-specific generation requirements={calibration['is_segmented_generation_valid']}",
        flush=True,
    )
    for artifact in result["segment_artifacts"]:
        print(
            f"  [{artifact['segment_id']:02d}] {artifact['filename']} | cohort={artifact['num_cohort']}, episode={artifact['num_episode']}, step={artifact['num_step']}",
            flush=True,
        )
    print(f"Result directory={output['result_directory']}", flush=True)
    print(f"Main result={output['result_path']}", flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    """Parse command-line arguments, run the requested operation, and report failures."""
    parser = _get_argument_parser()
    args = parser.parse_args(argv)
    try:
        overrides = _apply_cli_overrides(args.set_overrides, args)
        run_config = get_sim_run_config(args.config, set_overrides=overrides)
        run_config["data_source"] = validate_data_source(
            args.data_source or run_config.get("data_source", "spx")
        )
        if args.print_config_only:
            print(json.dumps(run_config, ensure_ascii=False, indent=2, sort_keys=True))
            return 0
        segmentation_root = _resolve_repository_path(run_config["segmentation"]["result_root"])
        segmentation_path = resolve_seg_result_path(
            run_config["segmentation"]["result_file"], result_root=segmentation_root
        )
        segmentation_result = load_seg_result(segmentation_path)
        segmentation_source = validate_data_source(segmentation_result.get("data_source", "spx"))
        if segmentation_source != run_config["data_source"]:
            raise ValueError(
                f"Stage 2 data_source differs from the Stage 1 segmentation result: {run_config['data_source']} != {segmentation_source}"
            )
        periods = get_seg_periods_from_result(segmentation_result)
        print(f"[Stage 2] Stage 1 result={segmentation_path}", flush=True)
        print(f"[Stage 2] seg_periods={periods}", flush=True)
        print(
            "[Stage 2] Rebuilding the training OptionDataset from the saved configuration...",
            flush=True,
        )
        print(f"[Data source] data_source={run_config['data_source']}", flush=True)
        dataset = get_train_dataset_from_seg_result(
            segmentation_result, data_source=run_config["data_source"]
        )
        print(
            f"[Stage 2] Dataset completed: episode={dataset.num_episode}, cohort={dataset.episode_manifest['cohort_id'].nunique()}, H={dataset.config.num_interval}",
            flush=True,
        )
        simulation_config = SimulationConfig(**run_config["simulation"])
        source_config = dataset.config
        if (
            source_config.expiry_weekday == simulation_config.expiry_weekday_calibration
            and source_config.num_moneyness == simulation_config.num_moneyness_calibration
        ):
            calibration_dataset = dataset
        else:
            print("[Stage 2] Rebuilding the calibration-only dataset...", flush=True)
            calibration_dataset = get_train_dataset_from_seg_result(
                segmentation_result,
                expiry_weekday=simulation_config.expiry_weekday_calibration,
                num_moneyness=simulation_config.num_moneyness_calibration,
                data_source=run_config["data_source"],
            )
        print(
            f"[Stage 2] Calibration dataset completed: expiry_weekday={calibration_dataset.config.expiry_weekday}, num_moneyness={calibration_dataset.config.num_moneyness}, episode={calibration_dataset.num_episode}, cohort={calibration_dataset.episode_manifest['cohort_id'].nunique()}",
            flush=True,
        )
        output_root = _resolve_repository_path(run_config["output"]["output_root"])
        output = run_segmented_sabr_simulation(
            dataset,
            segmentation_result,
            simulation_config,
            calibration_dataset=calibration_dataset,
            segmentation_result_path=segmentation_path,
            output_root=output_root,
            experiment_id=args.experiment_id,
            progress_callback=_print_progress,
        )
        result_path = Path(output["result_path"])
        result_payload = json.loads(result_path.read_text(encoding="utf-8"))
        result_payload["data_source"] = run_config["data_source"]
        result_path.write_text(
            json.dumps(result_payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        _print_result_summary(output)
        return 0
    except Exception as exc:
        parser.exit(1, f"run_sim failed [{type(exc).__name__}]: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
