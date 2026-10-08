"""Generate executable paper configurations for either external market dataset."""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from copy import deepcopy
from run_seg import get_default_seg_run_config
from run_sim import get_default_sim_run_config
from train_basis import get_default_basis_config
from train_meta import get_default_meta_config
from no_meta_variant.config import get_default_no_meta_config
from no_simulation_variant.config import get_default_no_simulation_config


def get_paper_configs(
    data_source: str,
    data_root: str,
    *,
    seeds=tuple(range(10)),
    augmentation_ratio: float = 1.0,
    context_window: int = 4,
) -> dict[str, dict]:
    """Use the paper's market filters and separate output roots for each market."""
    if data_source not in {"spx", "sx5e"}:
        raise ValueError("data_source must be spx or sx5e")
    if (
        not seeds
        or len(set(seeds)) != len(seeds)
        or any(
            (isinstance(s, bool) or not isinstance(s, int) or (not 0 <= s < 2**32) for s in seeds)
        )
    ):
        raise ValueError("seeds must be distinct integers in [0, 2^32-1]")
    if (
        isinstance(context_window, bool)
        or not isinstance(context_window, int)
        or context_window < 1
    ):
        raise ValueError("context_window must be a positive integer")
    from market_simulation import SimulationConfig

    SimulationConfig(num_simulation_times=augmentation_ratio)
    configs = {
        "seg": get_default_seg_run_config(),
        "sim": get_default_sim_run_config(),
        "basis": get_default_basis_config(),
        "meta": get_default_meta_config(),
        "no_meta": get_default_no_meta_config(),
        "no_simulation": get_default_no_simulation_config(),
    }
    for name, config in configs.items():
        config["data_source"] = data_source
        if "data" in config:
            config["data"]["data_root"] = data_root
            config["data"]["num_moneyness"] = 3 if data_source == "spx" else 20
            config["data"]["moneyness_range"] = [0.95, 1.05] if data_source == "spx" else [0.8, 1.2]
        if "training" in config:
            config["training"]["seeds"] = list(seeds)
            config["training"]["output_root"] = f"train_results/{name}/{data_source}"
            config["testing"]["output_root"] = f"test_results/{name}/{data_source}"
            if "num_recent_context_episodes" in config["training"]:
                config["training"]["num_recent_context_episodes"] = context_window
        if name == "seg":
            config["output"]["output_root"] = f"segmentation/seg_results/{data_source}"
        elif name == "sim":
            config["segmentation"]["result_root"] = f"segmentation/seg_results/{data_source}"
            config["output"]["output_root"] = f"simulation/sim_results/{data_source}"
            config["simulation"]["num_simulation_times"] = augmentation_ratio
            config["simulation"]["num_moneyness_calibration"] = 3 if data_source == "spx" else 20
        elif "simulation" in config:
            config["simulation"]["result_root"] = f"simulation/sim_results/{data_source}"
        elif "segmentation" in config:
            config["segmentation"]["result_root"] = f"segmentation/seg_results/{data_source}"
    return deepcopy(configs)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-source", choices=("spx", "sx5e"), required=True)
    parser.add_argument("--data-root", required=True, help="External raw market-data directory")
    parser.add_argument("--output-dir", type=Path, default=Path("configs"))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--augmentation-ratio", type=float, default=1.0)
    parser.add_argument("--context-window", type=int, default=4)
    args = parser.parse_args(argv)
    configs = get_paper_configs(
        args.data_source,
        args.data_root,
        seeds=args.seeds,
        augmentation_ratio=args.augmentation_ratio,
        context_window=args.context_window,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, config in configs.items():
        destination = args.output_dir / f"{args.data_source}_{name}.json"
        destination.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
        print(f"Configuration written: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
