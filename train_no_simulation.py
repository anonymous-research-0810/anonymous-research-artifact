"""Train no simulation for the paper option-hedging pipeline."""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

CODES_ROOT = Path(__file__).resolve().parent
for _source_directory in (
    CODES_ROOT,
    CODES_ROOT / "no_simulation_variant",
    CODES_ROOT / "meta_rl",
    CODES_ROOT / "rl_envs",
    CODES_ROOT / "datasets",
    CODES_ROOT / "data_processing",
    CODES_ROOT / "segmentation",
):
    if str(_source_directory) not in sys.path:
        sys.path.insert(0, str(_source_directory))
from data_source import VALID_DATA_SOURCES, validate_data_source
from no_simulation_variant import (
    get_no_simulation_config,
    get_no_simulation_run_specs,
    run_no_simulation_training,
)
from no_simulation_variant.config import _validate_no_simulation_config


def get_argument_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional JSON override; defaults to the no-simulation variant of the meta-RL configuration",
    )
    parser.add_argument(
        "--segmentation-result",
        type=str,
        default=None,
        help="Stage 1 JSON name or path; defaults to the latest completed result in result_root",
    )
    parser.add_argument(
        "--experiment-id",
        type=str,
        default=None,
        help="Output experiment directory name; defaults to no_simulation plus a UTC timestamp",
    )
    parser.add_argument(
        "--algorithm",
        action="append",
        choices=("pearl_td3", "pearl_sac"),
        default=None,
        help="Run the specified meta agent; may be repeated. Defaults to all configured algorithms",
    )
    parser.add_argument(
        "--run-id",
        action="append",
        default=None,
        help="Run only the specified resolved run_id; may be repeated",
    )
    parser.add_argument(
        "--data-source",
        choices=VALID_DATA_SOURCES,
        default=None,
        help="Override the data source; it must match the segmentation JSON",
    )
    return parser


def _select_run_specs(
    config: dict[str, Any], algorithms: Sequence[str] | None, run_ids: Sequence[str] | None
) -> list[dict[str, Any]]:
    """Select run specs."""
    specs = get_no_simulation_run_specs(config)
    if algorithms:
        allowed = set(algorithms)
        specs = [item for item in specs if item["algorithm_name"] in allowed]
    if run_ids:
        requested = set(run_ids)
        specs = [item for item in specs if item["run_id"] in requested]
        found = {item["run_id"] for item in specs}
        missing = requested - found
        if missing:
            raise ValueError(f"The requested no-simulation run_id was not found: {sorted(missing)}")
    if not specs:
        raise ValueError("No no-simulation configurations remain after filtering")
    return specs


def main(argv: Sequence[str] | None = None) -> int:
    """Parse command-line arguments, run the requested operation, and report failures."""
    parser = get_argument_parser()
    args = parser.parse_args(argv)
    config = get_no_simulation_config(args.config)
    if args.data_source is not None:
        config["data_source"] = validate_data_source(
            args.data_source or config.get("data_source", "spx")
        )
    _validate_no_simulation_config(config)
    specs = _select_run_specs(config, args.algorithm, args.run_id)
    print(
        json.dumps(
            {
                "variant": "no_simulation",
                "data_source": config["data_source"],
                "segmentation_result": args.segmentation_result
                or config["segmentation"].get("result"),
                "num_run": len(specs),
                "run_ids": [item["run_id"] for item in specs],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    run_no_simulation_training(
        config,
        run_specs=specs,
        segmentation_result=args.segmentation_result,
        experiment_id=args.experiment_id,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
