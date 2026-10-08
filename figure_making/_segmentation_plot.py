"""Reproduce and visualize the first-stage SPX/SX5E segmentation results."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REGIME_COLORS = (
    "#8dbbd6",  # blue
    "#f4b77d",  # orange
    "#a8d5a2",  # green
    "#c2ace1",  # violet
    "#ead477",  # yellow
    "#80c9c3",  # teal
    "#e5a4ad",  # rose
    "#aab9ce",  # slate
)


def _load_daily_spot(payload: dict) -> pd.DataFrame:
    config = payload["dataset"]["config"]
    data_root = Path(config["data_root"])
    if not data_root.is_absolute():
        data_root = ROOT / data_root
    source = data_root / "spx_spot.parquet"
    if not source.is_file():
        source = data_root / "sx5e_spot.parquet"
    if not source.is_file() and payload["data_source"] == "sx5e":
        # The adapter output is optional; its source spot series is unchanged.
        source = data_root.parent / "sx5e_spot.parquet"
    if not source.is_file():
        raise FileNotFoundError(f"Daily spot data not found in {data_root}")
    raw = pd.read_parquet(source)
    price_column = "close" if "close" in raw else "spot"
    daily = raw.loc[:, ["date", price_column]].rename(columns={price_column: "spot"})
    daily["date"] = pd.to_datetime(daily["date"]).dt.normalize()
    daily["spot"] = pd.to_numeric(daily["spot"], errors="raise")
    # Reproduce the exact observed span recorded by this run. The current raw
    # files may contain earlier rows than were available to the saved run.
    daily = daily.loc[
        daily["date"].between(
            payload["features"]["spot_date_start"],
            payload["features"]["spot_date_end"],
        )
    ]
    if daily.empty or daily[["date", "spot"]].isna().any().any():
        raise ValueError("Daily spot data are empty or contain missing values")
    if not np.isfinite(daily["spot"]).all() or not daily["spot"].gt(0).all():
        raise ValueError("Daily spot values must be finite and positive")
    groups = daily.groupby("date", sort=True, observed=True)["spot"]
    if any(not np.allclose(values, values.iloc[0], rtol=1e-12, atol=1e-10) for _, values in groups):
        raise ValueError("Conflicting spot values exist for the same date")
    daily = groups.first().reset_index()
    daily["log_return"] = np.log(daily["spot"]).diff()

    # The first-stage result stores a fingerprint of its exact date/spot input.
    date_bytes = daily["date"].to_numpy(dtype="datetime64[ns]").astype("<i8", copy=False).tobytes()
    spot_bytes = daily["spot"].to_numpy(dtype="<f8", copy=True).tobytes()
    actual_hash = hashlib.sha256(date_bytes + spot_bytes).hexdigest()
    if actual_hash != payload["features"]["daily_spot_sha256"]:
        raise ValueError("Daily spot does not match the saved segmentation input hash")
    return daily


def _recreate_features(daily: pd.DataFrame, payload: dict) -> pd.DataFrame:
    h = int(payload["features"]["realized_volatility_window"])
    if h != int(payload["dataset"]["config"]["num_interval"]):
        raise ValueError("Realized-volatility window and episode horizon differ")
    annualization = float(payload["segmentation_config"]["annualization_days"])
    epsilon = float(payload["segmentation_config"]["volatility_epsilon"])
    daily = daily.copy()
    daily["realized_volatility"] = np.sqrt(
        annualization * daily["log_return"].pow(2).rolling(window=h, min_periods=h).sum() / h
    )
    daily["log_realized_volatility"] = np.log(daily["realized_volatility"] + epsilon)
    features = daily.dropna(
        subset=["log_return", "realized_volatility", "log_realized_volatility"]
    ).copy()
    norm = payload["features"]["normalization"]
    for raw_name, score_name in (
        ("log_return", "log_return_zscore"),
        ("log_realized_volatility", "log_realized_volatility_zscore"),
    ):
        values = features[raw_name].to_numpy(dtype=np.float64)
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=0))
        if not np.isclose(mean, norm[raw_name]["mean"], atol=1e-12, rtol=0):
            raise ValueError(f"Saved {raw_name} mean does not match")
        if not np.isclose(std, norm[raw_name]["std"], atol=1e-12, rtol=0):
            raise ValueError(f"Saved {raw_name} standard deviation does not match")
        features[score_name] = (values - mean) / std
    signal = np.ascontiguousarray(
        features[["log_return_zscore", "log_realized_volatility_zscore"]].to_numpy(
            dtype=np.float64
        ),
        dtype="<f8",
    )
    if hashlib.sha256(signal.tobytes()).hexdigest() != payload["features"]["signal_sha256"]:
        raise ValueError("Reconstructed PELT signal does not match saved hash")
    if len(daily) != payload["features"]["num_daily_spot"]:
        raise ValueError("Daily spot count differs from saved result")
    if len(features) != payload["features"]["num_feature_observation"]:
        raise ValueError("Feature count differs from saved result")
    for stats in payload["segment_statistics"]:
        mask = daily["date"].between(stats["date_start"], stats["date_end"])
        if int(mask.sum()) != stats["num_trading_day"]:
            raise ValueError(f"Segment {stats['segment_id']} trading-day count differs")
    return daily


def _minmax(values: pd.Series) -> np.ndarray:
    array = values.to_numpy(dtype=float)
    low = np.nanmin(array)
    high = np.nanmax(array)
    if not high > low:
        raise ValueError("Cannot scale a constant time series")
    return (array - low) / (high - low)


def make_figure(
    market: str,
    result_name: str,
    *,
    background_alpha: float = 0.20,
    output_dir: str | Path | None = None,
) -> Path:
    """Generate a vector PDF after validating the original input and PELT signal."""
    if not 0 < background_alpha < 1:
        raise ValueError("background_alpha must be between zero and one")
    result_path = Path(result_name) if Path(result_name).is_absolute() else ROOT / result_name
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    if payload["status"] != "completed" or payload["data_source"] != market:
        raise ValueError("Wrong or incomplete segmentation result")
    segments = payload["segment_statistics"]
    if len(segments) != payload["num_segment"]:
        raise ValueError("Number of segments differs from saved result")
    daily = _recreate_features(_load_daily_spot(payload), payload)

    matplotlib.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.labelsize": 11.5,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "legend.fontsize": 10.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, ax = plt.subplots(figsize=(7.2, 3.55), constrained_layout=False)
    fig.subplots_adjust(left=0.095, right=0.99, bottom=0.16, top=0.81)

    first_date = pd.Timestamp(daily["date"].iloc[0])
    last_date = pd.Timestamp(daily["date"].iloc[-1])
    colors = [REGIME_COLORS[i % len(REGIME_COLORS)] for i in range(len(segments))]
    for index, (segment, color) in enumerate(zip(segments, colors)):
        left = max(pd.Timestamp(segment["date_start"]), first_date)
        right = (
            pd.Timestamp(segments[index + 1]["date_start"])
            if index + 1 < len(segments)
            else last_date + pd.Timedelta(days=1)
        )
        ax.axvspan(left, right, color=color, alpha=background_alpha, lw=0, zorder=0)
        midpoint = left + (right - left) / 2
        ax.text(
            midpoint,
            1.025,
            f"R{index + 1}",
            ha="center",
            va="bottom",
            transform=ax.get_xaxis_transform(),
            fontsize=10.5,
            fontweight="semibold",
            color="#445062",
        )
        if index:
            ax.axvline(left, color="#8d96a3", lw=0.65, alpha=0.75, zorder=1)

    ax.plot(
        daily["date"],
        _minmax(daily["log_return"]),
        color="#4b689a",
        lw=0.7,
        alpha=0.70,
        label="Daily log return",
        zorder=2,
    )
    ax.plot(
        daily["date"],
        _minmax(daily["realized_volatility"]),
        color="#008477",
        lw=1.45,
        alpha=0.98,
        label=f"{int(payload['features']['realized_volatility_window'])}-day realized volatility",
        zorder=3,
    )
    ax.plot(
        daily["date"],
        _minmax(daily["spot"]),
        color="#bf2f35",
        lw=1.45,
        alpha=0.99,
        label="Spot",
        zorder=4,
    )
    ax.set_xlim(first_date, last_date)
    ax.set_ylim(-0.025, 1.07)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_ylabel("Within-series min–max scale")
    ax.set_xlabel("Trading date")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(axis="y", color="#aeb7c2", linewidth=0.55, alpha=0.48)
    ax.set_axisbelow(True)
    ax.spines["left"].set_color("#7d8792")
    ax.spines["bottom"].set_color("#7d8792")
    ax.tick_params(axis="both", colors="#414b57", length=3)
    fig.legend(
        *ax.get_legend_handles_labels(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.977),
        ncol=3,
        frameon=False,
        columnspacing=2.2,
        handlelength=2.8,
    )

    output = (
        ROOT / "figures" if output_dir is None else Path(output_dir)
    ) / f"{market}_segmentation_overview.pdf"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, format="pdf", facecolor="white", bbox_inches="tight", pad_inches=0.035)
    plt.close(fig)
    return output
