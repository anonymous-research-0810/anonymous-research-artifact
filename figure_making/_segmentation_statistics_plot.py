"""Plot regime-level return and volatility statistics from saved PELT results."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator


ROOT = Path(__file__).resolve().parents[1]

RETURN_COLOR = "#3264A8"
VOLATILITY_COLOR = "#D35400"

# Match the exact MediaBox of the existing SPX/SX5E overview PDFs. Fixing the
# canvas rather than using a tight bounding box keeps all four paper figures
# identical in width, height, and aspect ratio.
OVERVIEW_PDF_SIZE_INCHES = (514.53925 / 72.0, 241.380825 / 72.0)


def _year_span(date_start: str, date_end: str) -> str:
    """Return a compact year label such as ``2018–19`` or ``2020``."""
    start_year = int(date_start[:4])
    end_year = int(date_end[:4])
    if start_year == end_year:
        return str(start_year)
    return f"{start_year}\N{EN DASH}{end_year % 100:02d}"


def _load_statistics(market: str, result_name: str) -> tuple[dict, list[dict]]:
    path = Path(result_name) if Path(result_name).is_absolute() else ROOT / result_name
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "completed":
        raise ValueError(f"Segmentation result is incomplete: {path}")
    if payload.get("data_source") != market:
        raise ValueError(f"Expected {market}, found {payload.get('data_source')}")
    records = payload["segment_statistics"]
    if len(records) != int(payload["num_segment"]):
        raise ValueError("Segment count does not match segment_statistics")
    expected_ids = list(range(1, len(records) + 1))
    if [int(item["segment_id"]) for item in records] != expected_ids:
        raise ValueError("Segment identifiers are not consecutive")
    for item in records:
        for name in ("annualized_mean_log_return", "mean_rolling_realized_volatility"):
            if not np.isfinite(float(item[name])):
                raise ValueError(f"Segment {item['segment_id']} has invalid {name}")
    return payload, records


def _annotation_offsets(first: np.ndarray, second: np.ndarray, index: int) -> tuple[int, int]:
    """Separate labels when two series are close at the same regime."""
    gap = float(first[index] - second[index])
    if abs(gap) < 5.0:
        return (14, -19) if gap >= 0 else (-19, 14)
    return 13, 13


def _annotate_point(
    ax: plt.Axes,
    x_value: float,
    y_value: float,
    *,
    offset: int,
    color: str,
) -> None:
    ax.annotate(
        f"{y_value:.2f}",
        xy=(x_value, y_value),
        xytext=(0, offset),
        textcoords="offset points",
        ha="center",
        va="bottom" if offset >= 0 else "top",
        color=color,
        fontsize=10.25,
        fontweight="semibold",
        zorder=6,
        bbox={
            "boxstyle": "round,pad=0.14",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.84,
        },
    )


def make_statistics_figure(
    market: str, result_name: str, *, output_dir: str | Path | None = None
) -> Path:
    """Create a vector PDF comparing adjacent PELT regimes on two statistics."""
    _, records = _load_statistics(market, result_name)
    x = np.arange(1, len(records) + 1, dtype=float)
    annualized_return = 100.0 * np.asarray(
        [item["annualized_mean_log_return"] for item in records], dtype=float
    )
    realized_volatility = 100.0 * np.asarray(
        [item["mean_rolling_realized_volatility"] for item in records], dtype=float
    )

    matplotlib.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.labelsize": 13.5,
            "xtick.labelsize": 11.5,
            "ytick.labelsize": 11.5,
            "legend.fontsize": 11.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, ax = plt.subplots(
        figsize=OVERVIEW_PDF_SIZE_INCHES,
        constrained_layout=False,
    )
    fig.subplots_adjust(left=0.11, right=0.985, bottom=0.22, top=0.80)

    # Alternating bands make each categorical regime and each adjacent transition clear.
    for index in range(len(records)):
        if index % 2 == 0:
            ax.axvspan(index + 0.5, index + 1.5, color="#DCE5EF", alpha=0.30, lw=0, zorder=0)
    for boundary in np.arange(1.5, len(records), 1.0):
        ax.axvline(boundary, color="#AAB4C0", lw=0.55, alpha=0.45, zorder=1)

    line_style = {
        "linewidth": 2.25,
        "marker": "o",
        "markersize": 7.5,
        "markeredgecolor": "white",
        "markeredgewidth": 1.15,
        "solid_capstyle": "round",
        "zorder": 4,
    }
    ax.plot(
        x,
        annualized_return,
        color=RETURN_COLOR,
        markerfacecolor=RETURN_COLOR,
        label="Annualized mean log return",
        **line_style,
    )
    ax.plot(
        x,
        realized_volatility,
        color=VOLATILITY_COLOR,
        markerfacecolor=VOLATILITY_COLOR,
        label="Mean 20-day realized volatility",
        **line_style,
    )

    for index, x_value in enumerate(x):
        return_offset, volatility_offset = _annotation_offsets(
            annualized_return, realized_volatility, index
        )
        _annotate_point(
            ax,
            x_value,
            annualized_return[index],
            offset=return_offset,
            color=RETURN_COLOR,
        )
        _annotate_point(
            ax,
            x_value,
            realized_volatility[index],
            offset=volatility_offset,
            color=VOLATILITY_COLOR,
        )

    tick_labels = [
        f"R{item['segment_id']}\n{_year_span(item['date_start'], item['date_end'])}"
        for item in records
    ]
    ax.set_xticks(x, tick_labels)
    ax.set_xlim(0.55, len(records) + 0.45)
    combined = np.concatenate([annualized_return, realized_volatility])
    lower = 10.0 * np.floor((combined.min() - 7.0) / 10.0)
    upper = 10.0 * np.ceil((combined.max() + 7.0) / 10.0)
    ax.set_ylim(lower, upper)
    ax.yaxis.set_major_locator(MultipleLocator(10.0))
    ax.axhline(0.0, color="#596675", lw=1.0, alpha=0.80, zorder=2)
    ax.grid(axis="y", color="#ABB5C1", lw=0.65, alpha=0.55)
    ax.set_axisbelow(True)
    ax.set_xlabel("Regime and calendar span")
    ax.set_ylabel("Regime statistic (%)")
    ax.spines["left"].set_color("#778391")
    ax.spines["bottom"].set_color("#778391")
    ax.tick_params(axis="both", colors="#3F4A57", length=3.5)
    fig.legend(
        *ax.get_legend_handles_labels(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.975),
        ncol=2,
        frameon=False,
        columnspacing=2.4,
        handlelength=2.7,
    )

    output = (
        ROOT / "figures" if output_dir is None else Path(output_dir)
    ) / f"{market}_segmentation_statistics.pdf"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output,
        format="pdf",
        facecolor="white",
    )
    plt.close(fig)
    return output
