"""Pelt for the paper option-hedging pipeline."""

from __future__ import annotations
from dataclasses import dataclass
from math import ceil
from typing import Any
import numpy as np
import ruptures as rpt
from scipy.spatial.distance import pdist
from scipy.stats import theilslopes
from .config import SegmentationConfig


@dataclass(frozen=True)
class PeltResult:
    """Pelt result."""

    algorithm: str
    breakpoints: tuple[int, ...]
    penalty: float
    penalty_selection: dict[str, Any]
    penalty_path: tuple[dict[str, Any], ...]
    bootstrap: dict[str, Any]
    rbf_gamma: float | None
    resolved_config: dict[str, Any]

    @property
    def num_segment(self) -> int:
        """Return the number of segment."""
        return len(self.breakpoints)

    @property
    def num_change_point(self) -> int:
        """Return the number of change point."""
        return max(len(self.breakpoints) - 1, 0)

    def get_record(self) -> dict[str, Any]:
        """Return a JSON-serializable record of the result and its metadata."""
        return {
            "algorithm": self.algorithm,
            "num_segment": self.num_segment,
            "num_change_point": self.num_change_point,
            "breakpoints": list(self.breakpoints),
            "penalty": self.penalty,
            "penalty_selection": self.penalty_selection,
            "penalty_path": list(self.penalty_path),
            "bootstrap": self.bootstrap,
            "rbf_gamma": self.rbf_gamma,
            "resolved_config": self.resolved_config,
        }


def _validate_signal(signal: np.ndarray, min_segment_length: int) -> np.ndarray:
    """Validate signal."""
    values = np.asarray(signal, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("signal must be (num_date, num_feature) two-dimensional array")
    if values.shape[1] < 1:
        raise ValueError("signal must contain at least one feature")
    if values.shape[0] < 2 * min_segment_length:
        raise ValueError(
            f"Too few signal observations to form two minimum-length segments: num_date={values.shape[0]}, min_segment_length={min_segment_length}"
        )
    if not np.isfinite(values).all():
        raise ValueError("signal contains NaN or Inf")
    return values.copy()


def _get_rbf_gamma(signal: np.ndarray, configured_gamma: float | None) -> float:
    """Return RBF gamma."""
    if configured_gamma is not None:
        return float(configured_gamma)
    squared_distances = pdist(signal, metric="sqeuclidean")
    positive = squared_distances[squared_distances > 0.0]
    if positive.size == 0:
        raise ValueError("All RBF samples are identical; kernel bandwidth is undefined")
    median_squared_distance = float(np.median(positive))
    gamma = 1.0 / median_squared_distance
    if not np.isfinite(gamma) or gamma <= 0.0:
        raise ValueError("The RBF median heuristic produced invalid gamma")
    return gamma


def _get_cost(signal: np.ndarray, config: SegmentationConfig, rbf_gamma: float | None) -> Any:
    """Return cost."""
    if rbf_gamma is None:
        raise RuntimeError("Internal error: RBF cost is missing gamma")
    return rpt.costs.CostRbf(gamma=rbf_gamma).fit(signal)


def _get_detector(signal: np.ndarray, config: SegmentationConfig, rbf_gamma: float | None) -> Any:
    """Return detector."""
    if rbf_gamma is None:
        raise RuntimeError("Internal error: RBF detector is missing gamma")
    return rpt.KernelCPD(
        kernel="rbf", min_size=config.min_segment_length, params={"gamma": rbf_gamma}
    ).fit(signal)


def _predict_breakpoints(detector: Any, penalty: float, num_sample: int) -> list[int]:
    """Predict breakpoints."""
    if not np.isfinite(penalty) or penalty <= 0.0:
        raise ValueError("PELT penalty must be finite and positive")
    breakpoints = [int(value) for value in detector.predict(pen=float(penalty))]
    if not breakpoints or breakpoints[-1] != num_sample:
        raise RuntimeError("ruptures breakpoints must end at the signal length")
    if breakpoints != sorted(set(breakpoints)):
        raise RuntimeError("ruptures breakpoints must be strictly increasing")
    return breakpoints


def _get_path_entry(cost: Any, breakpoints: list[int], penalty: float) -> dict[str, Any]:
    """Return path entry."""
    unpenalized_cost = float(cost.sum_of_costs(breakpoints))
    num_change_point = len(breakpoints) - 1
    return {
        "penalty": float(penalty),
        "num_change_point": int(num_change_point),
        "num_segment": int(len(breakpoints)),
        "unpenalized_cost": unpenalized_cost,
        "penalized_objective": unpenalized_cost + float(penalty) * num_change_point,
        "breakpoints": [int(value) for value in breakpoints],
    }


def _get_penalty_upper_bound(
    detector: Any, full_segment_cost: float, num_sample: int
) -> tuple[float, int]:
    """Return penalty upper bound."""
    scale = max(abs(float(full_segment_cost)), np.finfo(np.float64).eps)
    penalty_max = scale
    for num_expansion in range(13):
        breakpoints = _predict_breakpoints(detector, penalty_max, num_sample)
        if len(breakpoints) == 1:
            return (float(penalty_max), num_expansion)
        penalty_max *= 10.0
    raise RuntimeError("A single-segment PELT penalty bound was not found after 12 expansions")


def _get_penalty_path(
    signal: np.ndarray, config: SegmentationConfig, rbf_gamma: float | None
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return penalty path."""
    detector = _get_detector(signal, config, rbf_gamma)
    cost = _get_cost(signal, config, rbf_gamma)
    num_sample = signal.shape[0]
    full_segment_cost = float(cost.sum_of_costs([num_sample]))
    penalty_max, num_upper_expansion = _get_penalty_upper_bound(
        detector, full_segment_cost, num_sample
    )
    ratio = config.penalty_min_ratio
    final_path: list[dict[str, Any]] = []
    num_distinct_complexity = 0
    for num_expansion in range(config.num_penalty_expansion):
        penalties = np.geomspace(
            ratio * penalty_max, penalty_max, num=config.num_penalty_grid, dtype=np.float64
        )
        path = []
        for penalty in penalties:
            breakpoints = _predict_breakpoints(detector, float(penalty), num_sample)
            path.append(_get_path_entry(cost, breakpoints, float(penalty)))
        final_path = path
        num_distinct_complexity = len({entry["num_change_point"] for entry in path})
        if num_distinct_complexity >= config.min_slope_points:
            break
        next_ratio = max(ratio / 10.0, config.penalty_min_ratio_floor)
        if next_ratio == ratio:
            break
        ratio = next_ratio
    metadata = {
        "full_segment_cost": full_segment_cost,
        "penalty_max": penalty_max,
        "num_penalty_upper_expansion": num_upper_expansion,
        "resolved_penalty_min_ratio": ratio,
        "num_distinct_complexity": num_distinct_complexity,
    }
    return (final_path, metadata)


def _select_penalty_by_slope(
    path: list[dict[str, Any]], config: SegmentationConfig, metadata: dict[str, Any]
) -> tuple[float, dict[str, Any]]:
    """Select penalty by slope."""
    best_by_complexity: dict[int, dict[str, Any]] = {}
    for entry in path:
        complexity = int(entry["num_change_point"])
        current = best_by_complexity.get(complexity)
        if current is None or entry["unpenalized_cost"] < current["unpenalized_cost"]:
            best_by_complexity[complexity] = entry
    candidates = [
        entry
        for complexity, entry in sorted(
            best_by_complexity.items(), key=lambda item: item[0], reverse=True
        )
        if complexity > 0
    ]
    if len(candidates) < config.min_slope_points:
        raise RuntimeError(
            f"Too few distinct nonzero complexities on the penalty path for the slope heuristic: {len(candidates)} < {config.min_slope_points}"
        )
    num_high_complexity = max(
        config.min_slope_points, int(ceil(config.high_complexity_fraction * len(candidates)))
    )
    high_complexity = candidates[:num_high_complexity]
    x = np.asarray([entry["num_change_point"] for entry in high_complexity], dtype=np.float64)
    y = np.asarray([entry["unpenalized_cost"] for entry in high_complexity], dtype=np.float64)
    slope, intercept, slope_low, slope_high = theilslopes(y, x, alpha=0.95)
    minimum_penalty = -float(slope)
    if not np.isfinite(minimum_penalty) or minimum_penalty <= 0.0:
        raise RuntimeError(
            "High-complexity costs have no negative slope; the slope heuristic cannot determine a positive penalty"
        )
    selected_penalty = 2.0 * minimum_penalty
    record = {
        "method": "slope_heuristic",
        **metadata,
        "num_high_complexity_point": int(num_high_complexity),
        "high_complexity_points": [
            {
                "num_change_point": int(entry["num_change_point"]),
                "unpenalized_cost": float(entry["unpenalized_cost"]),
                "representative_penalty": float(entry["penalty"]),
            }
            for entry in high_complexity
        ],
        "robust_regression": {
            "slope": float(slope),
            "intercept": float(intercept),
            "slope_confidence_lower": float(slope_low),
            "slope_confidence_upper": float(slope_high),
        },
        "minimum_penalty": minimum_penalty,
        "initial_selected_penalty": selected_penalty,
    }
    return (selected_penalty, record)


def _moving_block_resample(
    segment: np.ndarray, block_length: int, rng: np.random.Generator
) -> np.ndarray:
    """Moving block resample."""
    num_sample = segment.shape[0]
    if num_sample <= 0:
        raise RuntimeError("Internal error: bootstrap segments must not be empty")
    effective_block_length = min(block_length, num_sample)
    num_block = int(ceil(num_sample / effective_block_length))
    blocks: list[np.ndarray] = []
    max_start = num_sample - effective_block_length
    for _ in range(num_block):
        start = int(rng.integers(0, max_start + 1))
        blocks.append(segment[start : start + effective_block_length])
    return np.concatenate(blocks, axis=0)[:num_sample]


def _get_bootstrap_stability(
    signal: np.ndarray,
    breakpoints: list[int],
    penalty: float,
    config: SegmentationConfig,
    rbf_gamma: float | None,
    *,
    block_length: int,
    boundary_tolerance: int,
    random_seed: int,
) -> dict[str, Any]:
    """Return bootstrap stability."""
    repetitions = config.bootstrap_repetitions
    internal = breakpoints[:-1]
    if repetitions == 0:
        return {
            "status": "skipped",
            "num_repetition": 0,
            "block_length": block_length,
            "boundary_tolerance": boundary_tolerance,
            "stability_threshold": config.stability_threshold,
            "is_stable": None,
            "boundary_statistics": [],
            "bootstrap_breakpoints": [],
        }
    if not internal:
        return {
            "status": "no_internal_boundary",
            "num_repetition": repetitions,
            "block_length": block_length,
            "boundary_tolerance": boundary_tolerance,
            "stability_threshold": config.stability_threshold,
            "is_stable": False,
            "boundary_statistics": [],
            "bootstrap_breakpoints": [],
        }
    segment_starts = [0, *internal]
    segment_ends = breakpoints
    rng = np.random.default_rng(random_seed)
    all_bootstrap_breakpoints: list[list[int]] = []
    matched_positions: dict[int, list[int]] = {int(boundary): [] for boundary in internal}
    for _ in range(repetitions):
        resampled_segments = [
            _moving_block_resample(signal[start:end], block_length, rng)
            for start, end in zip(segment_starts, segment_ends)
        ]
        bootstrap_signal = np.concatenate(resampled_segments, axis=0)
        detector = _get_detector(bootstrap_signal, config, rbf_gamma)
        bootstrap_breakpoints = _predict_breakpoints(detector, penalty, signal.shape[0])[:-1]
        all_bootstrap_breakpoints.append(bootstrap_breakpoints)
        for boundary in internal:
            if not bootstrap_breakpoints:
                continue
            nearest = min(bootstrap_breakpoints, key=lambda value: abs(int(value) - int(boundary)))
            if abs(int(nearest) - int(boundary)) <= boundary_tolerance:
                matched_positions[int(boundary)].append(int(nearest))
    boundary_statistics = []
    for boundary in internal:
        matches = matched_positions[int(boundary)]
        stability_rate = len(matches) / repetitions
        if matches:
            quantiles = np.quantile(np.asarray(matches, dtype=np.float64), [0.05, 0.5, 0.95])
            match_quantiles = {
                "q05": float(quantiles[0]),
                "q50": float(quantiles[1]),
                "q95": float(quantiles[2]),
            }
        else:
            match_quantiles = {"q05": None, "q50": None, "q95": None}
        boundary_statistics.append(
            {
                "breakpoint": int(boundary),
                "num_match": int(len(matches)),
                "stability_rate": float(stability_rate),
                "is_stable": bool(stability_rate >= config.stability_threshold),
                "matched_breakpoint_quantiles": match_quantiles,
            }
        )
    is_stable = all((bool(record["is_stable"]) for record in boundary_statistics))
    return {
        "status": "completed",
        "num_repetition": repetitions,
        "block_length": block_length,
        "boundary_tolerance": boundary_tolerance,
        "stability_threshold": config.stability_threshold,
        "is_stable": is_stable,
        "boundary_statistics": boundary_statistics,
        "bootstrap_breakpoints": all_bootstrap_breakpoints,
    }


def _get_larger_plateau_candidates(
    path: list[dict[str, Any]], selected_penalty: float, selected_breakpoints: list[int]
) -> list[dict[str, Any]]:
    """Return larger plateau candidates."""
    candidates: list[dict[str, Any]] = []
    seen = {tuple(selected_breakpoints)}
    for entry in sorted(path, key=lambda item: float(item["penalty"])):
        penalty = float(entry["penalty"])
        breakpoints = [int(value) for value in entry["breakpoints"]]
        key = tuple(breakpoints)
        if penalty <= selected_penalty or len(breakpoints) <= 1 or key in seen:
            continue
        seen.add(key)
        candidates.append(entry)
    return candidates


def _get_stability_candidate_record(
    penalty: float, breakpoints: list[int], bootstrap: dict[str, Any]
) -> dict[str, Any]:
    """Return stability candidate record."""
    return {
        "penalty": float(penalty),
        "breakpoints": [int(value) for value in breakpoints],
        "status": bootstrap["status"],
        "is_stable": bootstrap["is_stable"],
        "boundary_statistics": bootstrap["boundary_statistics"],
    }


def run_pelt_segmentation(
    signal: np.ndarray, config: SegmentationConfig, *, num_interval: int
) -> PeltResult:
    """Detect RBF-PELT regimes, select the penalty, and estimate boundary stability by moving-block bootstrap."""
    if not isinstance(config, SegmentationConfig):
        raise TypeError("config must be a SegmentationConfig instance")
    resolved_config = config.resolve_for_dataset(num_interval)
    values = _validate_signal(signal, config.min_segment_length)
    rbf_gamma = _get_rbf_gamma(values, config.rbf_gamma) if config.algorithm == "rbf" else None
    detector = _get_detector(values, config, rbf_gamma)
    cost = _get_cost(values, config, rbf_gamma)
    if config.penalty is not None:
        selected_penalty = float(config.penalty)
        selected_breakpoints = _predict_breakpoints(detector, selected_penalty, values.shape[0])
        penalty_path = [_get_path_entry(cost, selected_breakpoints, selected_penalty)]
        penalty_selection: dict[str, Any] = {
            "method": "explicit",
            "initial_selected_penalty": selected_penalty,
        }
    else:
        penalty_path, path_metadata = _get_penalty_path(values, config, rbf_gamma)
        selected_penalty, penalty_selection = _select_penalty_by_slope(
            penalty_path, config, path_metadata
        )
        selected_breakpoints = _predict_breakpoints(detector, selected_penalty, values.shape[0])
    block_length = int(resolved_config["resolved_bootstrap_block_length"])
    boundary_tolerance = int(resolved_config["resolved_boundary_tolerance"])
    bootstrap = _get_bootstrap_stability(
        values,
        selected_breakpoints,
        selected_penalty,
        config,
        rbf_gamma,
        block_length=block_length,
        boundary_tolerance=boundary_tolerance,
        random_seed=config.random_seed,
    )
    stability_candidates = [
        _get_stability_candidate_record(selected_penalty, selected_breakpoints, bootstrap)
    ]
    if (
        config.bootstrap_repetitions > 0
        and config.is_select_stable_plateau
        and (bootstrap["is_stable"] is False)
    ):
        for num_candidate, entry in enumerate(
            _get_larger_plateau_candidates(penalty_path, selected_penalty, selected_breakpoints),
            start=1,
        ):
            candidate_penalty = float(entry["penalty"])
            candidate_breakpoints = [int(value) for value in entry["breakpoints"]]
            candidate_bootstrap = _get_bootstrap_stability(
                values,
                candidate_breakpoints,
                candidate_penalty,
                config,
                rbf_gamma,
                block_length=block_length,
                boundary_tolerance=boundary_tolerance,
                random_seed=config.random_seed + num_candidate,
            )
            stability_candidates.append(
                _get_stability_candidate_record(
                    candidate_penalty, candidate_breakpoints, candidate_bootstrap
                )
            )
            if candidate_bootstrap["is_stable"] is True:
                selected_penalty = candidate_penalty
                selected_breakpoints = candidate_breakpoints
                bootstrap = candidate_bootstrap
                break
    penalty_selection["final_selected_penalty"] = selected_penalty
    penalty_selection["is_penalty_changed_for_stability"] = bool(
        selected_penalty != float(penalty_selection["initial_selected_penalty"])
    )
    penalty_selection["stability_candidates"] = stability_candidates
    return PeltResult(
        algorithm=config.algorithm,
        breakpoints=tuple(selected_breakpoints),
        penalty=float(selected_penalty),
        penalty_selection=penalty_selection,
        penalty_path=tuple(penalty_path),
        bootstrap=bootstrap,
        rbf_gamma=rbf_gamma,
        resolved_config=resolved_config,
    )
