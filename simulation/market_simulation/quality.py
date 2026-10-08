"""The four simulation-fidelity features reported in the paper."""

from __future__ import annotations
from typing import Any, Sequence
import numpy as np
from scipy.stats import wasserstein_distance
from option_dataset import OptionDataset
from hedging_env import HedgingEnv
from .config import SimulationConfig
from .exceptions import SimulationQualityError

PAPER_QUALITY_FEATURES = ("terminal_spot_ratio", "atm_iv", "iv_change", "iv_skew")


def get_dataset_quality_features(dataset: OptionDataset) -> dict[str, np.ndarray]:
    """Extract spot ratios once per cohort and pre-expiry IV features only.

    ATM is the available strike nearest the current forward price. Skew is
    the least-squares slope of IV against log strike-to-forward moneyness.
    IV changes use consecutive live observations of each individual contract.
    """
    if not isinstance(dataset, OptionDataset):
        raise TypeError("dataset must be an OptionDataset")
    steps = dataset.episode_steps
    ratios = []
    paths = steps.sort_values(["cohort_id", "date"], kind="mergesort").drop_duplicates(
        ["cohort_id", "date"]
    )
    for _, group in paths.groupby("cohort_id", sort=False):
        spot = group["spot"].to_numpy(dtype=np.float64)
        ratios.append(spot[-1] / spot[0])
    live = steps.loc[~steps["is_terminal"].astype(bool)].copy()
    forward = live["spot"].to_numpy(dtype=np.float64) * np.exp(
        (
            live["zero_rate"].to_numpy(dtype=np.float64)
            - live["dividend_rate"].to_numpy(dtype=np.float64)
        )
        * live["tau"].to_numpy(dtype=np.float64)
    )
    live["log_forward_moneyness"] = np.log(live["strike"].to_numpy(dtype=np.float64) / forward)
    atm, skew, changes = ([], [], [])
    for _, group in live.groupby(["cohort_id", "date"], sort=False):
        x = group["log_forward_moneyness"].to_numpy(dtype=np.float64)
        iv = group["resolved_iv"].to_numpy(dtype=np.float64)
        atm.append(iv[int(np.argmin(np.abs(x)))])
        if len(np.unique(x)) >= 2:
            skew.append(float(np.polyfit(x, iv, deg=1)[0]))
    for _, group in live.groupby("episode_id", sort=False):
        iv = group.sort_values("step")["resolved_iv"].to_numpy(dtype=np.float64)
        changes.extend(np.diff(iv))
    result = {}
    for name, values in zip(PAPER_QUALITY_FEATURES, (ratios, atm, changes, skew)):
        array = np.asarray(values, dtype=np.float64)
        if not array.size or not np.isfinite(array).all():
            raise SimulationQualityError(f"No complete finite observations for {name}")
        result[name] = array
    return result


def _get_distribution_summary(values: np.ndarray) -> dict[str, Any]:
    """Describe the samples used for a fidelity comparison."""
    return {
        "num_observation": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=0)),
    }


def compare_real_and_simulated(
    real_dataset: OptionDataset, simulated_dataset: OptionDataset, config: SimulationConfig
) -> dict[str, Any]:
    """Compare the four features using normalized Wasserstein distance.

    Each distance is divided by max(real population standard deviation,
    absolute real mean, 1e-12), matching the experimental implementation.
    """
    real_features = get_dataset_quality_features(real_dataset)
    simulated_features = get_dataset_quality_features(simulated_dataset)
    records = {}
    for name in PAPER_QUALITY_FEATURES:
        real, simulated = (real_features[name], simulated_features[name])
        distance = float(wasserstein_distance(real, simulated))
        scale = max(float(np.std(real, ddof=0)), abs(float(np.mean(real))), 1e-12)
        records[name] = {
            "real": _get_distribution_summary(real),
            "simulated": _get_distribution_summary(simulated),
            "wasserstein_distance": distance,
            "normalized_wasserstein_distance": distance / scale,
        }
    return records


def run_environment_rollout_checks(
    dataset: OptionDataset, config: SimulationConfig, *, seed: int
) -> dict[str, Any]:
    """Verify that generated episodes complete in the hedging environment."""
    num_check = min(config.num_environment_rollout_checks, dataset.num_episode)
    if num_check == 0:
        return {"status": "skipped", "num_checked_episode": 0}
    environment = HedgingEnv(
        dataset, is_batch_mode=False, is_repeated=False, is_record_trajectory=False, seed=seed
    )
    checked_ids = []
    try:
        for _ in range(num_check):
            _, info = environment.reset()
            checked_ids.append(str(info["episode_id"]))
            for step in range(dataset.config.num_interval):
                _, reward, terminated, truncated, _ = environment.step(0.0)
                if not np.isfinite(reward) or bool(truncated):
                    raise SimulationQualityError(
                        "Rollout returned a nonfinite reward or truncated episode"
                    )
                if bool(terminated) != (step == dataset.config.num_interval - 1):
                    raise SimulationQualityError(
                        "Rollout termination differs from the configured horizon"
                    )
    except Exception as exc:
        raise SimulationQualityError(f"Generated data failed environment rollout: {exc}") from exc
    finally:
        environment.close()
    return {
        "status": "completed",
        "num_checked_episode": num_check,
        "checked_episode_ids": checked_ids,
    }


def evaluate_segmented_simulations(
    real_datasets: Sequence[OptionDataset],
    simulated_datasets: Sequence[OptionDataset],
    config: SimulationConfig,
) -> dict[str, Any]:
    """Evaluate fidelity and environment compatibility for every regime."""
    if not real_datasets or len(real_datasets) != len(simulated_datasets):
        raise ValueError(
            "Real and simulated datasets must contain the same nonzero number of regimes"
        )
    segments = []
    for segment_id, (real, simulated) in enumerate(zip(real_datasets, simulated_datasets), 1):
        segments.append(
            {
                "segment_id": segment_id,
                "num_real_episode": real.num_episode,
                "num_simulated_episode": simulated.num_episode,
                "distribution_comparison": compare_real_and_simulated(real, simulated, config),
                "environment_rollout": run_environment_rollout_checks(
                    simulated, config, seed=config.random_seed + segment_id
                ),
            }
        )
    return {"num_segment": len(segments), "segments": segments}


def compare_segmented_and_global_simulations(
    real_datasets: Sequence[OptionDataset],
    segmented_datasets: Sequence[OptionDataset],
    global_datasets: Sequence[OptionDataset],
    config: SimulationConfig,
) -> dict[str, Any]:
    """Average each paper feature's normalized distance equally across regimes."""
    segmented = evaluate_segmented_simulations(real_datasets, segmented_datasets, config)
    baseline = evaluate_segmented_simulations(real_datasets, global_datasets, config)
    records = []
    for feature in PAPER_QUALITY_FEATURES:

        def average(result):
            return float(
                np.mean(
                    [
                        s["distribution_comparison"][feature]["normalized_wasserstein_distance"]
                        for s in result["segments"]
                    ]
                )
            )

        left, right = (average(segmented), average(baseline))
        records.append(
            {
                "feature": feature,
                "segmented": left,
                "global_baseline": right,
                "segmented_relative_improvement": None if right == 0 else (right - left) / right,
            }
        )
    return {
        "segmented": segmented,
        "global_baseline": baseline,
        "comparison": {
            "metric": "normalized_wasserstein_distance",
            "lower_is_better": True,
            "aggregation": "equal_weight_per_segment",
            "features": records,
        },
    }
