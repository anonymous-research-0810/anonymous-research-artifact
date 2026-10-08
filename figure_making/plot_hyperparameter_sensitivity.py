"""Plot sensitivity results supplied as per-seed experiment measurements."""

from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="CSV columns: parameter, value, market, reward, algorithm, seed, j_lambda",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    args = parser.parse_args(argv)
    frame = pd.read_csv(args.input)
    keys = ["parameter", "value", "market", "reward", "algorithm", "seed"]
    if frame.empty or not set(keys + ["j_lambda"]).issubset(frame):
        raise ValueError("Sensitivity input is empty or missing required columns")
    if frame[keys + ["j_lambda"]].isna().any().any():
        raise ValueError("Sensitivity input must not contain missing values")
    if frame.duplicated(keys).any():
        raise ValueError("Sensitivity input contains duplicate seed measurements")
    if not set(frame["market"]).issubset({"spx", "sx5e"}):
        raise ValueError("Unknown market in sensitivity input")
    if not set(frame["reward"]).issubset({"shaped_accounting", "cash_flow"}) or not set(
        frame["algorithm"]
    ).issubset({"sac", "td3"}):
        raise ValueError("Sensitivity input must use the paper's rewards and backbones")
    valid_parameters = {"num_simulation_times", "num_recent_context_episodes"}
    if not set(frame["parameter"]).issubset(valid_parameters):
        raise ValueError("Unknown sensitivity parameter")
    for column in ("value", "j_lambda"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        if not np.isfinite(frame[column]).all():
            raise ValueError(f"{column} must be finite")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for (market, parameter), group in frame.groupby(["market", "parameter"], sort=True):
        fig, ax = plt.subplots(figsize=(6.8, 3.6), constrained_layout=True)
        for (reward, algorithm), runs in group.groupby(["reward", "algorithm"], sort=True):
            values = runs.groupby("value", sort=True)["j_lambda"].mean()
            label = ("Acct" if reward == "shaped_accounting" else "Cash") + "+" + algorithm.upper()
            ax.plot(values.index, values.values, marker="o", label=label)
        ax.set_xlabel(
            "Simulated-to-real episode ratio"
            if parameter == "num_simulation_times"
            else "Recent context window"
        )
        ax.set_ylabel(f"Risk-adjusted loss ({('USD' if market == 'spx' else 'EUR')})")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(frameon=False)
        destination = args.output_dir / f"{market}_{parameter}_sensitivity.pdf"
        fig.savefig(destination, metadata={"Creator": "", "Producer": ""})
        plt.close(fig)
        print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
