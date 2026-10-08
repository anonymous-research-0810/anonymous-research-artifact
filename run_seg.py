"""Run seg for the paper option-hedging pipeline."""

from __future__ import annotations
import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence
import pandas as pd

CODES_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = CODES_ROOT
for _source_directory in (
    CODES_ROOT,
    CODES_ROOT / "segmentation",
    CODES_ROOT / "datasets",
    CODES_ROOT / "data_processing",
):
    if str(_source_directory) not in sys.path:
        sys.path.insert(0, str(_source_directory))
from option_dataset import OptionDataset, get_option_dataset, get_option_dataset_config
from market_segmentation import (
    SegmentationConfig,
    get_daily_spot_from_market,
    run_market_segmentation,
)
from train_basis import get_default_basis_config
from data_source import VALID_DATA_SOURCES, resolve_data_root, validate_data_source

SEG_RUN_SCHEMA_VERSION = 1


def get_default_seg_run_config() -> dict[str, Any]:
    """Return default seg run config."""
    basis_config = get_default_basis_config()
    return {
        "schema_version": SEG_RUN_SCHEMA_VERSION,
        "data_source": basis_config.get("data_source", "spx"),
        "data": deepcopy(basis_config["data"]),
        "segmentation": SegmentationConfig().get_dict(),
        "output": {"output_root": "segmentation/seg_results"},
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


def _set_config_leaf(config: dict[str, Any], expression: str) -> None:
    """Apply a validated dotted-path configuration override."""
    if "=" not in expression:
        raise ValueError(f"--set must use SECTION.KEY=VALUE syntax: {expression!r}")
    path_text, value_text = expression.split("=", maxsplit=1)
    keys = [key.strip() for key in path_text.split(".")]
    if not keys or any((not key for key in keys)):
        raise ValueError(f"Invalid --set path: {path_text!r}")
    target: Any = config
    for key in keys[:-1]:
        if isinstance(target, dict):
            if key not in target:
                raise ValueError(f"Unknown --set configuration path: {path_text!r}")
            target = target[key]
            continue
        if isinstance(target, list):
            try:
                index = int(key)
            except ValueError as exc:
                raise ValueError(
                    f"--set list paths require integer indices: {path_text!r}"
                ) from exc
            if not 0 <= index < len(target):
                raise ValueError(f"--set list index is out of range: {path_text!r}")
            target = target[index]
            continue
        raise ValueError(f"Intermediate --set path is not an object/list: {path_text!r}")
    leaf = keys[-1]
    parsed_value = _parse_override_value(value_text)
    if isinstance(target, dict):
        if leaf not in target:
            raise ValueError(f"Unknown --set configuration path: {path_text!r}")
        if isinstance(target[leaf], Mapping):
            raise ValueError("--set supports leaves only; use --config for objects")
        target[leaf] = parsed_value
        return
    if isinstance(target, list):
        try:
            index = int(leaf)
        except ValueError as exc:
            raise ValueError(f"--set list paths require integer indices: {path_text!r}") from exc
        if not 0 <= index < len(target):
            raise ValueError(f"--set list index is out of range: {path_text!r}")
        target[index] = parsed_value
        return
    raise ValueError(f"Target --set path is not an object/list: {path_text!r}")


def _validate_seg_run_config(config: Mapping[str, Any]) -> None:
    """Validate seg run config."""
    if config["schema_version"] != SEG_RUN_SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SEG_RUN_SCHEMA_VERSION}")
    validate_data_source(config.get("data_source", "spx"))
    data = deepcopy(dict(config["data"]))
    periods = data.pop("date_periods")
    if not isinstance(periods, Mapping) or "train" not in periods:
        raise ValueError("data.date_periods must include train")
    data_root = resolve_data_root(
        config.get("data_source", "spx"), data.pop("data_root"), is_prepare=False
    )
    get_option_dataset_config(
        date_period=periods["train"], label="train", data_root=data_root, **data
    )
    SegmentationConfig(**dict(config["segmentation"]))
    output_root = config["output"]["output_root"]
    if not isinstance(output_root, (str, Path)) or not str(output_root):
        raise ValueError("output.output_root must be a nonempty path")


def get_seg_run_config(
    config_path: str | Path | None = None, *, set_overrides: Sequence[str] = ()
) -> dict[str, Any]:
    """Return seg run config."""
    config = get_default_seg_run_config()
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
    _validate_seg_run_config(config)
    return config


def get_train_dataset(config: Mapping[str, Any]) -> OptionDataset:
    """Return train dataset."""
    data = deepcopy(dict(config["data"]))
    periods = data.pop("date_periods")
    data["data_root"] = resolve_data_root(config.get("data_source", "spx"), data["data_root"])
    return get_option_dataset(periods["train"], label="train", **data)


def _apply_common_cli_overrides(
    raw_overrides: Sequence[str], args: argparse.Namespace
) -> list[str]:
    """Apply common CLI overrides."""
    overrides = list(raw_overrides)
    aliases = (
        ("algorithm", "segmentation.algorithm"),
        ("penalty", "segmentation.penalty"),
        ("min_segment_length", "segmentation.min_segment_length"),
        ("bootstrap_repetitions", "segmentation.bootstrap_repetitions"),
        ("bootstrap_block_length", "segmentation.bootstrap_block_length"),
        ("boundary_tolerance", "segmentation.boundary_tolerance"),
        ("seed", "segmentation.random_seed"),
        ("date_start", "data.date_periods.train.0"),
        ("date_end", "data.date_periods.train.1"),
        ("data_root", "data.data_root"),
        ("num_interval", "data.num_interval"),
        ("output_root", "output.output_root"),
    )
    for attribute, path in aliases:
        value = getattr(args, attribute)
        if value is None:
            continue
        if attribute in {"date_start", "date_end"}:
            continue
        overrides.append(f"{path}={json.dumps(value)}")
    return overrides


def _apply_date_cli_overrides(config: dict[str, Any], args: argparse.Namespace) -> None:
    """Apply date CLI overrides."""
    if args.date_start is not None:
        config["data"]["date_periods"]["train"][0] = args.date_start
    if args.date_end is not None:
        config["data"]["date_periods"]["train"][1] = args.date_end
    _validate_seg_run_config(config)


def _get_argument_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Build the training OptionDataset, run RBF-PELT, and save a JSON record."
    )
    parser.add_argument("--config", type=Path, help="Optional partial JSON configuration override")
    parser.add_argument(
        "--set",
        dest="set_overrides",
        action="append",
        default=[],
        metavar="SECTION.KEY=VALUE",
        help="Override a configuration leaf; repeat as needed. Values use JSON syntax, e.g. --set data.expiry_weekday=null",
    )
    parser.add_argument(
        "--algorithm", choices=("rbf",), help="Convenient alias: segmentation.algorithm"
    )
    parser.add_argument(
        "--penalty", type=float, help="Explicit positive penalty; defaults to the slope heuristic"
    )
    parser.add_argument("--min-segment-length", type=int)
    parser.add_argument("--bootstrap-repetitions", type=int)
    parser.add_argument("--bootstrap-block-length", type=int)
    parser.add_argument("--boundary-tolerance", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--date-start", type=str)
    parser.add_argument("--date-end", type=str)
    parser.add_argument("--data-root", type=str)
    parser.add_argument(
        "--num-interval",
        type=int,
        help="Episode horizon H; also sets the realized-volatility window",
    )
    parser.add_argument("--output-root", type=str)
    parser.add_argument(
        "--experiment-id", type=str, help="Optional result stem; defaults to seg_YYYYMMDDTHHMMSSZ"
    )
    parser.add_argument(
        "--print-config-only",
        action="store_true",
        help="Print resolved configuration and exit without reading market data",
    )
    parser.add_argument(
        "--data-source",
        choices=VALID_DATA_SOURCES,
        default=None,
        help="data source; sx5e triggers canonical preprocessing",
    )
    return parser


def _print_result_summary(run_output: Mapping[str, Any]) -> None:
    """Print result summary."""
    result = run_output["result"]
    pelt = result["pelt"]
    features = result["features"]
    print("\n[Stage 1 segmentation completed]", flush=True)
    print(
        f"Algorithm={pelt['algorithm'].upper()}-PELT, H/volatility window={features['realized_volatility_window']}, feature trading days={features['num_feature_observation']}, penalty={pelt['penalty']:.8g}",
        flush=True,
    )
    if pelt["rbf_gamma"] is not None:
        print(f"RBF gamma={pelt['rbf_gamma']:.8g}", flush=True)
    print(f"Number of regimes N={result['num_segment']}", flush=True)
    for stats in result["segment_statistics"]:
        print(
            f"  [{stats['segment_id']:02d}] {stats['date_start']} -> {stats['date_end']} | trading days={stats['num_trading_day']}, cohort={stats['num_cohort']}, episode={stats['num_episode']}, cross-boundary episodes={stats['num_cross_boundary_episode']}",
            flush=True,
        )
    for boundary in result["boundaries"]:
        stability = boundary["bootstrap"]
        rate_text = f"{stability['stability_rate']:.1%}" if stability is not None else "not run"
        print(
            f"  Change point {boundary['next_segment_start']} (previous regime ends at {boundary['previous_segment_end']}), stability rate={rate_text}",
            flush=True,
        )
    bootstrap = pelt["bootstrap"]
    print(
        f"Stability={bootstrap['status']}, is_stable={bootstrap['is_stable']}, bootstrap={bootstrap['num_repetition']}",
        flush=True,
    )
    if result["num_segment"] < 3:
        print(
            "Warning: fewer than three regimes; the result does not meet the main meta-RL task requirement.",
            flush=True,
        )
    print(f"Result file={run_output['result_path']}", flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    """Parse command-line arguments, run the requested operation, and report failures."""
    parser = _get_argument_parser()
    args = parser.parse_args(argv)
    try:
        overrides = _apply_common_cli_overrides(args.set_overrides, args)
        config = get_seg_run_config(args.config, set_overrides=overrides)
        config["data_source"] = validate_data_source(
            args.data_source or config.get("data_source", "spx")
        )
        _apply_date_cli_overrides(config, args)
        if args.print_config_only:
            print(json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True))
            return 0
        segmentation_config = SegmentationConfig(**config["segmentation"])
        output_root = Path(config["output"]["output_root"])
        if not output_root.is_absolute():
            output_root = REPOSITORY_ROOT / output_root
        print("[Stage 1] Building the training OptionDataset...", flush=True)
        print(f"[Data source] data_source={config['data_source']}", flush=True)
        dataset = get_train_dataset(config)
        print(
            f"[Stage 1] Dataset completed: episode={dataset.num_episode}, step={dataset.num_step}, H={dataset.config.num_interval}",
            flush=True,
        )
        print(
            f"[Stage 1] Running {segmentation_config.algorithm.upper()}-PELT: min_size={segmentation_config.min_segment_length}, bootstrap={segmentation_config.bootstrap_repetitions}...",
            flush=True,
        )
        if dataset.episode_manifest.empty:
            raise ValueError(
                "The training dataset is empty; the task coverage start cannot be determined"
            )
        first_episode_date = pd.Timestamp(dataset.episode_manifest["t0"].min()).normalize()
        spot_start = max(dataset.config.date_start, first_episode_date)
        daily_spot = get_daily_spot_from_market(
            dataset.config.data_root, (spot_start, dataset.config.date_end)
        )
        print(
            f"[Stage 1] Raw spot series within task coverage: trading_days={len(daily_spot)}, date={daily_spot['date'].iloc[0]:%Y-%m-%d}->{daily_spot['date'].iloc[-1]:%Y-%m-%d}",
            flush=True,
        )
        run_output = run_market_segmentation(
            dataset,
            segmentation_config,
            daily_spot=daily_spot,
            output_root=output_root,
            experiment_id=args.experiment_id,
        )
        result_path = Path(run_output["result_path"])
        result_payload = json.loads(result_path.read_text(encoding="utf-8"))
        result_payload["data_source"] = config["data_source"]
        result_payload["data_root"] = str(dataset.config.data_root)
        result_path.write_text(
            json.dumps(result_payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        _print_result_summary(run_output)
        return 0
    except Exception as exc:
        parser.exit(1, f"run_seg failed [{type(exc).__name__}]: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
