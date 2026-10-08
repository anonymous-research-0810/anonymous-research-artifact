"""Generation for the paper option-hedging pipeline."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Sequence
import numpy as np
import pandas as pd
from option_dataset import OptionDataset
from option_dataset.constants import CANDIDATE_AUDIT_COLUMNS, EPISODE_STEP_COLUMNS, MANIFEST_COLUMNS
from .calibration import SabrCalibrationResult
from .config import SimulationConfig
from .exceptions import SimulationError, SimulationQualityError
from .sabr import get_bsm_option_prices, get_lognormal_sabr_iv, simulate_lognormal_sabr_path

STEP_PROVENANCE_COLUMNS = (
    "is_simulated",
    "segment_id",
    "simulation_id",
    "anchor_episode_id",
    "anchor_cohort_id",
    "simulation_seed",
    "num_path_retry",
    "sabr_alpha",
    "calibration_fallback",
)
MANIFEST_PROVENANCE_COLUMNS = (
    "is_simulated",
    "segment_id",
    "simulation_id",
    "anchor_episode_id",
    "anchor_cohort_id",
    "simulation_seed",
    "num_path_retry",
    "calibration_fallback",
)


@dataclass(frozen=True)
class SimulationGenerationResult:
    """Simulation generation result."""

    datasets: tuple[OptionDataset, ...]
    simulation_manifest: pd.DataFrame
    diagnostics: dict[str, Any]


def _get_empty_candidate_audit() -> pd.DataFrame:
    """Return empty candidate audit."""
    return pd.DataFrame({column: pd.Series(dtype="object") for column in CANDIDATE_AUDIT_COLUMNS})


def _get_anchor_cohort_frames(dataset: OptionDataset, cohort_id: Any) -> list[pd.DataFrame]:
    """Return anchor cohort frames."""
    manifest = dataset.episode_manifest
    rows = manifest.loc[manifest["cohort_id"].eq(cohort_id)].sort_values(
        "selection_rank", kind="mergesort"
    )
    if rows.empty:
        raise RuntimeError("Internal error: the sampled anchor cohort does not exist")
    return [dataset.get_episode(str(episode_id), is_copy=True) for episode_id in rows["episode_id"]]


def _validate_anchor_cohort(frames: Sequence[pd.DataFrame]) -> None:
    """Validate anchor cohort."""
    if not frames:
        raise SimulationError("anchor cohort must not be empty")
    reference = frames[0]
    columns = (
        "date",
        "exdate",
        "step",
        "dte",
        "tau",
        "spot",
        "zero_rate",
        "dividend_rate",
        "funding_rate",
        "is_terminal",
    )
    for frame in frames:
        if len(frame) != len(reference):
            raise SimulationError("Episodes in an anchor cohort have different lengths")
        for column in columns:
            left = reference[column].to_numpy()
            right = frame[column].to_numpy()
            if np.issubdtype(left.dtype, np.number):
                is_equal = np.allclose(
                    left.astype(float), right.astype(float), rtol=0.0, atol=1e-12, equal_nan=True
                )
            else:
                is_equal = np.array_equal(left, right)
            if not is_equal:
                raise SimulationError(f"Within one anchor cohort, {column} paths differ")


def _get_initial_alpha(
    anchor: pd.DataFrame,
    segment_id: int,
    calibration: SabrCalibrationResult,
    config: SimulationConfig,
    rng: np.random.Generator,
) -> tuple[float, str, str]:
    """Return initial alpha."""
    initial = anchor.iloc[0]
    quotes = calibration.cross_sections
    exact = quotes.loc[
        quotes["segment_id"].eq(segment_id)
        & pd.to_datetime(quotes["date"]).eq(pd.Timestamp(initial["date"]))
        & pd.to_datetime(quotes["exdate"]).eq(pd.Timestamp(initial["exdate"]))
        & quotes["cp_flag"].astype(str).eq(str(initial["cp_flag"]))
        & quotes["symbol_root"].astype(str).eq(str(initial["symbol_root"]))
    ]
    if not exact.empty:
        row = exact.iloc[0]
        return (float(row["alpha_atm"]), "anchor_cross_section", str(row["cross_section_id"]))
    candidates = quotes.loc[
        quotes["segment_id"].eq(segment_id)
        & quotes["cp_flag"].astype(str).eq(str(initial["cp_flag"]))
        & quotes["symbol_root"].astype(str).eq(str(initial["symbol_root"]))
        & (
            quotes["dte"].astype(int).sub(int(initial["dte"])).abs()
            <= config.max_alpha_dte_difference
        )
    ].drop_duplicates("cross_section_id")
    if candidates.empty:
        raise SimulationQualityError(
            "The initial anchor cross-section has no alpha and no maturity-matched fallback exists in this task"
        )
    dte_difference = candidates["dte"].astype(int).sub(int(initial["dte"])).abs()
    candidates = candidates.loc[dte_difference.eq(dte_difference.min())]
    candidate_dates = candidates["date"].drop_duplicates().reset_index(drop=True)
    sampled_date = candidate_dates.iloc[int(rng.integers(0, len(candidate_dates)))]
    same_date = candidates.loc[pd.to_datetime(candidates["date"]).eq(sampled_date)]
    row = same_date.iloc[int(rng.integers(0, len(same_date)))]
    return (float(row["alpha_atm"]), "nearby_maturity_sample", str(row["cross_section_id"]))


def _get_price_bounds(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return price bounds."""
    spot = frame["spot"].to_numpy(dtype=np.float64)
    strike = frame["strike"].to_numpy(dtype=np.float64)
    tau = frame["tau"].to_numpy(dtype=np.float64)
    rate = frame["zero_rate"].to_numpy(dtype=np.float64)
    dividend = frame["dividend_rate"].to_numpy(dtype=np.float64)
    discounted_spot = spot * np.exp(-dividend * tau)
    discounted_strike = strike * np.exp(-rate * tau)
    cp_flag = str(frame["cp_flag"].iloc[0])
    if cp_flag == "C":
        return (np.maximum(discounted_spot - discounted_strike, 0.0), discounted_spot)
    return (np.maximum(discounted_strike - discounted_spot, 0.0), discounted_strike)


def _validate_cross_strike_shape(frames: Sequence[pd.DataFrame], relative_tolerance: float) -> None:
    """Validate cross strike shape."""
    if len(frames) < 2:
        return
    cp_flag = str(frames[0]["cp_flag"].iloc[0])
    num_step = len(frames[0])
    for step in range(num_step):
        strikes = np.asarray([float(frame.iloc[step]["strike"]) for frame in frames])
        prices = np.asarray([float(frame.iloc[step]["mid_price"]) for frame in frames])
        order = np.argsort(strikes, kind="mergesort")
        strikes = strikes[order]
        prices = prices[order]
        scale = max(float(np.max(strikes)), 1.0)
        tolerance = relative_tolerance * scale
        price_changes = np.diff(prices)
        if cp_flag == "C" and np.any(price_changes > tolerance):
            raise SimulationQualityError(
                "Simulated call prices are not monotonically decreasing in strike"
            )
        if cp_flag == "P" and np.any(price_changes < -tolerance):
            raise SimulationQualityError(
                "Simulated put prices are not monotonically increasing in strike"
            )
        if len(frames) >= 3:
            strike_changes = np.diff(strikes)
            if np.any(strike_changes <= 0.0):
                raise SimulationQualityError("A simulated cohort has duplicate or unsorted strikes")
            slopes = price_changes / strike_changes
            if np.any(np.diff(slopes) < -relative_tolerance):
                raise SimulationQualityError("Simulated option prices violate discrete convexity")


def _validate_candidate_cohort(
    frames: Sequence[pd.DataFrame],
    alphas: np.ndarray,
    anchor_frames: Sequence[pd.DataFrame],
    dataset: OptionDataset,
    config: SimulationConfig,
) -> None:
    """Validate candidate cohort."""
    if len(frames) != len(anchor_frames) or not frames:
        raise SimulationQualityError("Simulated and anchor cohort episode counts differ")
    if not np.isfinite(alphas).all() or np.any(alphas <= 0.0):
        raise SimulationQualityError(
            "The simulated alpha path contains nonfinite or nonpositive values"
        )
    reference_spot = frames[0]["spot"].to_numpy(dtype=np.float64)
    if not np.isfinite(reference_spot).all() or np.any(reference_spot <= 0.0):
        raise SimulationQualityError(
            "The simulated spot path contains nonfinite or nonpositive values"
        )
    for frame, anchor in zip(frames, anchor_frames):
        for column in ("date", "tau", "is_terminal"):
            if not np.array_equal(frame[column].to_numpy(), anchor[column].to_numpy()):
                raise SimulationQualityError(
                    f"Simulated {column} does not exactly reuse the anchor"
                )
        if not np.array_equal(frame["spot"].to_numpy(dtype=np.float64), reference_spot):
            raise SimulationQualityError(
                "Contracts in a simulated cohort do not share an identical spot path"
            )
        live = ~frame["is_terminal"].astype(bool)
        iv = frame.loc[live, "resolved_iv"].to_numpy(dtype=np.float64)
        price = frame["mid_price"].to_numpy(dtype=np.float64)
        if (
            not np.isfinite(iv).all()
            or np.any(iv < dataset.config.min_iv)
            or np.any(iv > dataset.config.max_iv)
            or (not np.isfinite(price).all())
            or np.any(price < 0.0)
        ):
            raise SimulationQualityError(
                "Simulated IV or option prices fail finiteness or range checks"
            )
        lower, upper = _get_price_bounds(frame)
        tolerance = config.price_relative_tolerance * max(float(frame["spot"].iloc[0]), 1.0)
        if np.any(price < lower - tolerance) or np.any(price > upper + tolerance):
            raise SimulationQualityError("Simulated option prices violate BSM no-arbitrage bounds")
        terminal = frame.iloc[-1]
        expected_payoff = (
            max(float(terminal["spot"]) - float(terminal["strike"]), 0.0)
            if str(terminal["cp_flag"]) == "C"
            else max(float(terminal["strike"]) - float(terminal["spot"]), 0.0)
        )
        if not np.isclose(float(terminal["mid_price"]), expected_payoff, rtol=0.0, atol=tolerance):
            raise SimulationQualityError("Simulated terminal prices differ from the exact payoff")
    _validate_cross_strike_shape(frames, config.price_relative_tolerance)


def _build_simulated_episode(
    anchor: pd.DataFrame,
    *,
    spots: np.ndarray,
    alphas: np.ndarray,
    rho: float,
    nu: float,
    config: SimulationConfig,
    segment_id: int,
    simulation_id: str,
    simulated_cohort_id: pd.Timestamp,
    simulated_optionid: int,
    anchor_episode_id: str,
    anchor_cohort_id: str,
    simulation_seed: int,
    num_retry: int,
    calibration_fallback: str,
) -> pd.DataFrame:
    """Build simulated episode."""
    frame = anchor.copy(deep=True)
    strike = float(anchor["strike"].iloc[0])
    cp_flag = str(anchor["cp_flag"].iloc[0])
    live = ~frame["is_terminal"].astype(bool).to_numpy()
    forward = spots[live] * np.exp(
        (
            frame.loc[live, "zero_rate"].to_numpy(dtype=np.float64)
            - frame.loc[live, "dividend_rate"].to_numpy(dtype=np.float64)
        )
        * frame.loc[live, "tau"].to_numpy(dtype=np.float64)
    )
    iv = get_lognormal_sabr_iv(
        forward,
        strike,
        frame.loc[live, "tau"].to_numpy(dtype=np.float64),
        alphas[live],
        rho,
        nu,
        z_tolerance=config.sabr_z_tolerance,
    )
    live_prices = get_bsm_option_prices(
        spot=spots[live],
        strike=strike,
        tau=frame.loc[live, "tau"].to_numpy(dtype=np.float64),
        volatility=iv,
        zero_rate=frame.loc[live, "zero_rate"].to_numpy(dtype=np.float64),
        dividend_rate=frame.loc[live, "dividend_rate"].to_numpy(dtype=np.float64),
        cp_flag=cp_flag,
    )
    terminal_payoff = (
        max(float(spots[-1]) - strike, 0.0)
        if cp_flag == "C"
        else max(strike - float(spots[-1]), 0.0)
    )
    prices = np.r_[live_prices, terminal_payoff]
    resolved_iv = np.r_[iv, np.nan]
    relative_spread = anchor["relative_spread"].to_numpy(dtype=np.float64).copy()
    relative_spread[~live] = 0.0
    best_bid = prices * (1.0 - 0.5 * relative_spread)
    best_offer = prices * (1.0 + 0.5 * relative_spread)
    frame["episode_id"] = f"{simulation_id}_rank{int(anchor['selection_rank'].iloc[0]):02d}"
    frame["cohort_id"] = simulated_cohort_id
    frame["optionid"] = int(simulated_optionid)
    frame["symbol"] = "SIM " + str(anchor["symbol"].iloc[0])
    frame["spot"] = spots
    frame["strike"] = strike
    frame["mid_price"] = prices
    frame["quoted_mid_price"] = prices
    frame["best_bid"] = best_bid
    frame["best_offer"] = best_offer
    frame["impl_volatility"] = resolved_iv
    frame["resolved_iv"] = resolved_iv
    frame["iv_source"] = [*["sabr_simulated"] * int(live.sum()), "terminal_no_action"]
    frame["relative_spread"] = relative_spread
    s0 = float(spots[0])
    frame["spot_norm"] = spots / s0
    frame["strike_norm"] = strike / s0
    frame["option_mid_norm"] = prices / s0
    frame["moneyness_initial"] = strike / s0
    frame["atm_distance"] = abs(strike / s0 - 1.0)
    frame["is_environment_ready"] = True
    frame["is_simulated"] = True
    frame["segment_id"] = int(segment_id)
    frame["simulation_id"] = simulation_id
    frame["anchor_episode_id"] = anchor_episode_id
    frame["anchor_cohort_id"] = anchor_cohort_id
    frame["simulation_seed"] = int(simulation_seed)
    frame["num_path_retry"] = int(num_retry)
    frame["sabr_alpha"] = alphas
    frame["calibration_fallback"] = calibration_fallback
    return frame.reindex(columns=[*EPISODE_STEP_COLUMNS, *STEP_PROVENANCE_COLUMNS])


def _get_simulated_manifest_row(frame: pd.DataFrame) -> dict[str, Any]:
    """Return simulated manifest row."""
    initial = frame.iloc[0]
    return {
        "episode_id": str(initial["episode_id"]),
        "cohort_id": pd.Timestamp(initial["cohort_id"]),
        "label": str(initial["label"]),
        "selection_rank": int(initial["selection_rank"]),
        "optionid": int(initial["optionid"]),
        "symbol": str(initial["symbol"]),
        "symbol_root": str(initial["symbol_root"]),
        "cp_flag": str(initial["cp_flag"]),
        "t0": pd.Timestamp(initial["date"]),
        "exdate": pd.Timestamp(frame.iloc[-1]["date"]),
        "num_interval": int(len(frame) - 1),
        "num_state": int(len(frame)),
        "s0": float(initial["spot"]),
        "c0": float(initial["mid_price"]),
        "strike": float(initial["strike"]),
        "moneyness_initial": float(initial["moneyness_initial"]),
        "atm_distance": float(initial["atm_distance"]),
        "relative_spread_initial": float(initial["relative_spread"]),
        "open_interest_initial": float(initial["open_interest"]),
        "volume_initial": float(initial["volume"]),
        "num_iv_imputed": 0,
        "is_environment_ready": True,
        "is_simulated": True,
        "segment_id": int(initial["segment_id"]),
        "simulation_id": str(initial["simulation_id"]),
        "anchor_episode_id": str(initial["anchor_episode_id"]),
        "anchor_cohort_id": str(initial["anchor_cohort_id"]),
        "simulation_seed": int(initial["simulation_seed"]),
        "num_path_retry": int(initial["num_path_retry"]),
        "calibration_fallback": str(initial["calibration_fallback"]),
    }


def _coerce_generated_tables(
    step_frames: Sequence[pd.DataFrame], manifest_rows: Sequence[dict[str, Any]]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Coerce generated tables."""
    steps = pd.concat(step_frames, ignore_index=True)
    manifest = pd.DataFrame(manifest_rows).reindex(
        columns=[*MANIFEST_COLUMNS, *MANIFEST_PROVENANCE_COLUMNS]
    )
    for column in (
        "episode_id",
        "label",
        "symbol",
        "symbol_root",
        "cp_flag",
        "iv_source",
        "simulation_id",
        "anchor_episode_id",
        "anchor_cohort_id",
        "calibration_fallback",
    ):
        if column in steps:
            steps[column] = steps[column].astype("string")
        if column in manifest:
            manifest[column] = manifest[column].astype("string")
    for column in ("cohort_id", "date", "exdate"):
        steps[column] = pd.to_datetime(steps[column], errors="raise")
    for column in ("cohort_id", "t0", "exdate"):
        manifest[column] = pd.to_datetime(manifest[column], errors="raise")
    for column in ("selection_rank", "step", "dte"):
        steps[column] = steps[column].astype(np.int16)
    steps["optionid"] = steps["optionid"].astype(np.int64)
    manifest["selection_rank"] = manifest["selection_rank"].astype(np.int16)
    manifest["optionid"] = manifest["optionid"].astype(np.int64)
    for column in ("num_interval", "num_state", "num_iv_imputed"):
        manifest[column] = manifest[column].astype(np.int16)
    for column in (
        "is_terminal",
        "is_environment_ready",
        "is_open_interest_missing",
        "is_volume_missing",
        "is_simulated",
    ):
        steps[column] = steps[column].astype(bool)
    for column in ("is_environment_ready", "is_simulated"):
        manifest[column] = manifest[column].astype(bool)
    steps = steps.sort_values(
        ["simulation_id", "selection_rank", "step"], kind="mergesort"
    ).reset_index(drop=True)
    manifest = manifest.sort_values(
        ["simulation_id", "selection_rank"], kind="mergesort"
    ).reset_index(drop=True)
    return (steps, manifest)


def generate_segment_simulation(
    real_dataset: OptionDataset,
    segment_id: int,
    calibration: SabrCalibrationResult,
    config: SimulationConfig,
    *,
    rng: np.random.Generator,
    simulation_plan: pd.DataFrame | None = None,
) -> tuple[OptionDataset, pd.DataFrame, dict[str, Any]]:
    """Generate complete shared-path option cohorts from anchors whose full lifecycles lie inside the regime."""
    if real_dataset.num_episode == 0:
        raise SimulationError(f"segment {segment_id} has no real episodes")
    manifest = real_dataset.episode_manifest
    segmentation_record = real_dataset.dataset_build_report.get("segmentation")
    if not isinstance(segmentation_record, dict) or "date_end" not in segmentation_record:
        raise SimulationError(
            "real_dataset is missing segmentation.date_end recorded by segment_option_dataset"
        )
    segment_end = pd.Timestamp(segmentation_record["date_end"]).normalize()
    eligible_manifest = manifest.loc[
        pd.to_datetime(manifest["exdate"]).dt.normalize().le(segment_end)
    ].copy()
    if eligible_manifest.empty:
        raise SimulationError(
            f"segment {segment_id} has no simulation anchors contained within the regime"
        )
    cohort_sizes = eligible_manifest.groupby("cohort_id", sort=False).size()
    if cohort_sizes.nunique() != 1:
        raise SimulationError(
            f"Exact episode counts and shared paths require balanced anchor cohorts; segment={segment_id}, sizes={sorted(cohort_sizes.unique())}"
        )
    cohort_size = int(cohort_sizes.iloc[0])
    if real_dataset.num_episode % cohort_size:
        raise SimulationError("Real episodes must form complete balanced cohorts")
    num_real_cohort = real_dataset.num_episode // cohort_size
    num_simulated_cohort = max(
        1, int(np.floor(num_real_cohort * config.num_simulation_times + 0.5))
    )
    target_episode = num_simulated_cohort * cohort_size
    segment_plan: pd.DataFrame | None = None
    if simulation_plan is not None:
        required_columns = {
            "segment_id",
            "simulation_id",
            "anchor_episode_id",
            "anchor_cohort_id",
            "simulation_seed",
        }
        missing = required_columns - set(simulation_plan.columns)
        if missing:
            raise ValueError(f"simulation_plan is missing columns: {sorted(missing)}")
        segment_plan = (
            simulation_plan.loc[simulation_plan["segment_id"].eq(segment_id)]
            .sort_values("simulation_id", kind="mergesort")
            .reset_index(drop=True)
        )
        if len(segment_plan) != num_simulated_cohort:
            raise ValueError(
                f"simulation_plan cohort count does not match generation target: segment={segment_id}, {len(segment_plan)} != {num_simulated_cohort}"
            )
    parameters = calibration.segment_parameters.loc[
        calibration.segment_parameters["segment_id"].eq(segment_id)
    ]
    if len(parameters) != 1:
        raise RuntimeError(f"segment {segment_id} has no unique SABR parameter record")
    parameter = parameters.iloc[0]
    fallback = "global" if bool(parameter["is_global_fallback"]) else "none"
    step_frames: list[pd.DataFrame] = []
    manifest_rows: list[dict[str, Any]] = []
    cohort_records: list[dict[str, Any]] = []
    failure_counts: dict[str, int] = {}
    for simulation_number in range(1, num_simulated_cohort + 1):
        planned = None if segment_plan is None else segment_plan.iloc[simulation_number - 1]
        if planned is None:
            sampled_episode_position = int(rng.integers(0, len(eligible_manifest)))
            sampled_manifest = eligible_manifest.iloc[sampled_episode_position]
        else:
            planned_episode_id = str(planned["anchor_episode_id"])
            candidates = eligible_manifest.loc[
                eligible_manifest["episode_id"].astype(str).eq(planned_episode_id)
            ]
            if len(candidates) != 1:
                raise ValueError(
                    f"simulation_plan anchor episode is not uniquely eligible: {planned_episode_id}"
                )
            sampled_manifest = candidates.iloc[0]
        anchor_episode_id = str(sampled_manifest["episode_id"])
        anchor_cohort = sampled_manifest["cohort_id"]
        anchor_frames = _get_anchor_cohort_frames(real_dataset, anchor_cohort)
        _validate_anchor_cohort(anchor_frames)
        simulation_id = f"SIM_SEG{segment_id:02d}_{simulation_number:06d}"
        if planned is not None:
            if str(planned["simulation_id"]) != simulation_id:
                raise ValueError("simulation_plan simulation_id sequence is invalid")
            if pd.Timestamp(planned["anchor_cohort_id"]) != pd.Timestamp(anchor_cohort):
                raise ValueError("simulation_plan anchor cohort does not match episode")
        simulated_cohort_id = pd.Timestamp(anchor_frames[0]["exdate"].iloc[0]) + pd.Timedelta(
            segment_id * 1000000 + simulation_number, unit="ns"
        )
        accepted_frames: list[pd.DataFrame] | None = None
        accepted_alpha_source = ""
        accepted_alpha_cross_section = ""
        accepted_seed = -1
        accepted_retry = -1
        for retry in range(config.max_path_retry):
            if planned is not None and retry == 0:
                path_seed = int(planned["simulation_seed"])
            else:
                path_seed = int(rng.integers(0, np.iinfo(np.int64).max))
            path_rng = np.random.default_rng(path_seed)
            try:
                alpha_initial, alpha_source, alpha_cross_section = _get_initial_alpha(
                    anchor_frames[0], segment_id, calibration, config, path_rng
                )
                spots, alphas = simulate_lognormal_sabr_path(
                    spot_initial=float(anchor_frames[0]["spot"].iloc[0]),
                    alpha_initial=alpha_initial,
                    annual_log_drift=float(parameter["annual_log_drift"]),
                    rho=float(parameter["rho"]),
                    nu=float(parameter["nu"]),
                    num_interval=real_dataset.config.num_interval,
                    annualization_days=config.annualization_days,
                    rng=path_rng,
                )
                candidate_frames = []
                for rank_index, anchor in enumerate(anchor_frames, start=1):
                    optionid = -(segment_id * 10**12 + simulation_number * 10**3 + rank_index)
                    candidate_frames.append(
                        _build_simulated_episode(
                            anchor,
                            spots=spots,
                            alphas=alphas,
                            rho=float(parameter["rho"]),
                            nu=float(parameter["nu"]),
                            config=config,
                            segment_id=segment_id,
                            simulation_id=simulation_id,
                            simulated_cohort_id=simulated_cohort_id,
                            simulated_optionid=optionid,
                            anchor_episode_id=anchor_episode_id,
                            anchor_cohort_id=str(pd.Timestamp(anchor_cohort)),
                            simulation_seed=path_seed,
                            num_retry=retry,
                            calibration_fallback=fallback,
                        )
                    )
                _validate_candidate_cohort(
                    candidate_frames, alphas, anchor_frames, real_dataset, config
                )
                accepted_frames = candidate_frames
                accepted_alpha_source = alpha_source
                accepted_alpha_cross_section = alpha_cross_section
                accepted_seed = path_seed
                accepted_retry = retry
                break
            except SimulationQualityError as exc:
                reason = str(exc)
                failure_counts[reason] = failure_counts.get(reason, 0) + 1
        if accepted_frames is None:
            raise SimulationError(
                f"segment {segment_id} {simulation_id} at {config.max_path_retry} attempts did not pass quality checks"
            )
        step_frames.extend(accepted_frames)
        manifest_rows.extend((_get_simulated_manifest_row(frame) for frame in accepted_frames))
        cohort_records.append(
            {
                "segment_id": segment_id,
                "simulation_id": simulation_id,
                "simulated_cohort_id": simulated_cohort_id,
                "anchor_episode_id": anchor_episode_id,
                "anchor_cohort_id": pd.Timestamp(anchor_cohort),
                "num_episode": cohort_size,
                "simulation_seed": accepted_seed,
                "num_path_retry": accepted_retry,
                "alpha_initial_source": accepted_alpha_source,
                "alpha_cross_section_id": accepted_alpha_cross_section,
                "calibration_fallback": fallback,
                "failure_before_acceptance": accepted_retry,
            }
        )
    steps, simulated_manifest = _coerce_generated_tables(step_frames, manifest_rows)
    if len(simulated_manifest) != target_episode:
        raise RuntimeError(
            f"Internal error: the simulated episode count differs from its target: {len(simulated_manifest)} != {target_episode}"
        )
    dataset = OptionDataset(
        label=real_dataset.label,
        config=real_dataset.config,
        episode_steps=steps,
        episode_manifest=simulated_manifest,
        candidate_selection_audit=_get_empty_candidate_audit(),
        dataset_build_report={
            "source": "segment_sabr_simulation",
            "segment_id": segment_id,
            "num_real_episode": real_dataset.num_episode,
            "num_simulation_times": config.num_simulation_times,
            "realized_augmentation_ratio": target_episode / real_dataset.num_episode,
            "num_simulated_episode": len(simulated_manifest),
            "num_simulated_cohort": num_simulated_cohort,
            "cohort_size": cohort_size,
            "calibration_fallback": fallback,
        },
    )
    diagnostics = {
        "segment_id": segment_id,
        "num_real_episode": real_dataset.num_episode,
        "num_real_cohort": int(manifest["cohort_id"].nunique()),
        "num_eligible_anchor_episode": int(len(eligible_manifest)),
        "num_eligible_anchor_cohort": int(eligible_manifest["cohort_id"].nunique()),
        "num_cross_boundary_anchor_excluded": int(len(manifest) - len(eligible_manifest)),
        "cohort_size": cohort_size,
        "target_simulated_episode": target_episode,
        "num_simulated_episode": dataset.num_episode,
        "num_simulated_cohort": num_simulated_cohort,
        "is_exact_episode_multiple": dataset.num_episode == target_episode,
        "num_rejected_path": int(sum(failure_counts.values())),
        "path_failure_counts": failure_counts,
    }
    return (dataset, pd.DataFrame(cohort_records), diagnostics)


def generate_segmented_simulations(
    real_datasets: Sequence[OptionDataset],
    calibration: SabrCalibrationResult,
    config: SimulationConfig,
    *,
    simulation_plan: pd.DataFrame | None = None,
) -> SimulationGenerationResult:
    """Generate regime-specific cohorts, optionally reusing a fixed anchor plan for paired comparisons."""
    if not real_datasets:
        raise ValueError("real_datasets must not be empty")
    rng = np.random.default_rng(config.random_seed)
    datasets: list[OptionDataset] = []
    manifests: list[pd.DataFrame] = []
    segment_records: list[dict[str, Any]] = []
    for segment_id, real_dataset in enumerate(real_datasets, start=1):
        simulated, manifest, diagnostics = generate_segment_simulation(
            real_dataset, segment_id, calibration, config, rng=rng, simulation_plan=simulation_plan
        )
        datasets.append(simulated)
        manifests.append(manifest)
        segment_records.append(diagnostics)
    return SimulationGenerationResult(
        datasets=tuple(datasets),
        simulation_manifest=pd.concat(manifests, ignore_index=True),
        diagnostics={
            "num_segment": len(datasets),
            "num_simulated_episode": int(sum((dataset.num_episode for dataset in datasets))),
            "num_simulated_cohort": int(
                sum((record["num_simulated_cohort"] for record in segment_records))
            ),
            "segments": segment_records,
        },
    )
