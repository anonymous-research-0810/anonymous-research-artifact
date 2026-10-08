"""Calibration for the paper option-hedging pipeline."""

from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass
from itertools import product
from typing import Any, Sequence
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from option_dataset import OptionDataset
from market_segmentation import normalize_seg_periods
from .config import SimulationConfig
from .exceptions import SabrCalibrationError
from .sabr import get_lognormal_sabr_iv

CALIBRATION_KEY_COLUMNS = ("date", "exdate", "strike", "cp_flag", "symbol_root")
CROSS_SECTION_COLUMNS = ("date", "exdate", "cp_flag", "symbol_root")
VENDOR_IV_SOURCES = frozenset({"vendor_call", "vendor_put"})


@dataclass(frozen=True)
class SabrCalibrationResult:
    """Sabr calibration result."""

    cross_sections: pd.DataFrame
    segment_parameters: pd.DataFrame
    optimization_runs: pd.DataFrame
    diagnostics: dict[str, Any]


def _assign_segment_ids(dates: pd.Series, seg_periods: Sequence[Sequence[Any]]) -> np.ndarray:
    """Assign segment ids."""
    periods = normalize_seg_periods(seg_periods)
    normalized = pd.to_datetime(dates, errors="raise").dt.normalize()
    assignments = np.zeros(len(normalized), dtype=np.int16)
    for segment_id, (start, end) in enumerate(periods, start=1):
        mask = normalized.between(start, end, inclusive="both").to_numpy()
        if np.any(assignments[mask] != 0):
            raise ValueError("seg_periods cover quote dates more than once")
        assignments[mask] = segment_id
    if np.any(assignments == 0):
        missing = normalized.iloc[np.flatnonzero(assignments == 0)[:5]]
        raise ValueError(
            "Some quote dates are not covered by seg_periods: "
            + ", ".join((value.strftime("%Y-%m-%d") for value in missing))
        )
    return assignments


def _get_cross_section_alpha(
    group: pd.DataFrame, max_atm_distance: float
) -> tuple[float | None, str, float | None]:
    """Return cross section alpha."""
    grouped = (
        group.groupby("log_forward_moneyness", sort=True, as_index=False)["resolved_iv"]
        .median()
        .sort_values("log_forward_moneyness", kind="mergesort")
    )
    x = grouped["log_forward_moneyness"].to_numpy(dtype=np.float64)
    iv = grouped["resolved_iv"].to_numpy(dtype=np.float64)
    if len(x) == 0:
        return (None, "no_valid_quote", None)
    exact = np.flatnonzero(np.isclose(x, 0.0, rtol=0.0, atol=1e-14))
    if exact.size:
        return (float(iv[exact[0]]), "exact_forward_atm", 0.0)
    left = np.flatnonzero(x < 0.0)
    right = np.flatnonzero(x > 0.0)
    if left.size and right.size:
        left_index = int(left[-1])
        right_index = int(right[0])
        weight = -x[left_index] / (x[right_index] - x[left_index])
        alpha = iv[left_index] + weight * (iv[right_index] - iv[left_index])
        distance = max(abs(x[left_index]), abs(x[right_index]))
        return (float(alpha), "interpolated_forward_atm", float(distance))
    nearest_index = int(np.argmin(np.abs(x)))
    nearest_distance = float(abs(x[nearest_index]))
    if nearest_distance <= max_atm_distance:
        return (float(iv[nearest_index]), "nearest_forward_atm", nearest_distance)
    return (None, "nearest_quote_too_far", nearest_distance)


def _get_consistent_cross_section_values(group: pd.DataFrame) -> tuple[float, float, float, float]:
    """Return consistent cross section values."""
    values = []
    for column in ("spot", "tau", "zero_rate", "dividend_rate"):
        array = group[column].to_numpy(dtype=np.float64)
        reference = float(array[0])
        if not np.allclose(array, reference, rtol=0.0, atol=1e-12):
            raise SabrCalibrationError(f"Cross-section {column} differs across strikes")
        values.append(reference)
    return (values[0], values[1], values[2], values[3])


def prepare_sabr_cross_sections(
    dataset: OptionDataset, seg_periods: Sequence[Sequence[Any]], config: SimulationConfig
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Prepare SABR cross sections."""
    if not isinstance(dataset, OptionDataset):
        raise TypeError("dataset must be an OptionDataset")
    if not isinstance(config, SimulationConfig):
        raise TypeError("config must be a SimulationConfig")
    steps = dataset.episode_steps.loc[~dataset.episode_steps["is_terminal"].astype(bool)].copy(
        deep=True
    )
    num_input = len(steps)
    if steps.empty:
        raise SabrCalibrationError("No nonterminal quotes are available for SABR calibration")
    numeric_columns = ("resolved_iv", "spot", "strike", "tau", "zero_rate", "dividend_rate")
    numeric = steps.loc[:, numeric_columns].apply(pd.to_numeric, errors="coerce")
    valid = np.isfinite(numeric.to_numpy(dtype=np.float64)).all(axis=1)
    valid &= numeric["resolved_iv"].between(
        dataset.config.min_iv, dataset.config.max_iv, inclusive="both"
    )
    valid &= numeric["spot"].gt(0.0)
    valid &= numeric["strike"].gt(0.0)
    valid &= numeric["tau"].gt(0.0)
    steps = steps.loc[valid].copy()
    num_invalid = num_input - len(steps)
    if steps.empty:
        raise SabrCalibrationError("All nonterminal quotes failed SABR input checks")
    steps["_iv_priority"] = (~steps["iv_source"].isin(VENDOR_IV_SOURCES)).astype(int)
    spread = pd.to_numeric(steps["relative_spread"], errors="coerce")
    steps["_spread_priority"] = spread.where(np.isfinite(spread) & spread.ge(0.0), np.inf)
    steps = steps.sort_values(
        [*CALIBRATION_KEY_COLUMNS, "_iv_priority", "_spread_priority", "optionid"], kind="mergesort"
    )
    duplicate_mask = steps.duplicated(CALIBRATION_KEY_COLUMNS, keep="first")
    num_duplicate = int(duplicate_mask.sum())
    steps = steps.loc[~duplicate_mask].copy()
    steps = steps.drop(columns=["_iv_priority", "_spread_priority"])
    steps["segment_id"] = _assign_segment_ids(steps["date"], seg_periods)
    cross_section_records: list[dict[str, Any]] = []
    alpha_failure_counts: dict[str, int] = {}
    grouped_sections = steps.groupby(list(CROSS_SECTION_COLUMNS), sort=True)
    for key, positions in grouped_sections.groups.items():
        group = steps.loc[positions]
        spot, tau, zero_rate, dividend_rate = _get_consistent_cross_section_values(group)
        forward = spot * np.exp((zero_rate - dividend_rate) * tau)
        log_moneyness = np.log(group["strike"].to_numpy(dtype=np.float64) / forward)
        steps.loc[positions, "forward"] = forward
        steps.loc[positions, "log_forward_moneyness"] = log_moneyness
        alpha, method, distance = _get_cross_section_alpha(
            steps.loc[positions], dataset.config.max_iv_extrapolation_log_moneyness
        )
        if alpha is None or not np.isfinite(alpha) or alpha <= 0.0:
            alpha_failure_counts[method] = alpha_failure_counts.get(method, 0) + 1
            continue
        date, exdate, cp_flag, symbol_root = key
        cross_section_id = (
            f"{pd.Timestamp(date):%Y%m%d}|{pd.Timestamp(exdate):%Y%m%d}|{symbol_root}|{cp_flag}"
        )
        steps.loc[positions, "cross_section_id"] = cross_section_id
        steps.loc[positions, "alpha_atm"] = alpha
        steps.loc[positions, "alpha_method"] = method
        steps.loc[positions, "alpha_atm_distance"] = distance
        cross_section_records.append(
            {
                "cross_section_id": cross_section_id,
                "segment_id": int(steps.loc[positions, "segment_id"].iloc[0]),
                "date": pd.Timestamp(date),
                "exdate": pd.Timestamp(exdate),
                "dte": int(group["dte"].iloc[0]),
                "cp_flag": str(cp_flag),
                "symbol_root": str(symbol_root),
                "num_quote": int(len(group)),
                "alpha_atm": float(alpha),
                "alpha_method": method,
                "alpha_atm_distance": distance,
            }
        )
    if not cross_section_records:
        raise SabrCalibrationError("No cross-section provides valid ATM alpha")
    steps = steps.loc[steps["cross_section_id"].notna()].copy()
    finite_spread = pd.to_numeric(steps["relative_spread"], errors="coerce")
    finite_spread = finite_spread.where(np.isfinite(finite_spread))
    raw_weight = 1.0 / np.maximum(
        finite_spread.to_numpy(dtype=np.float64, na_value=np.nan), config.spread_weight_epsilon
    )
    finite_weight = raw_weight[np.isfinite(raw_weight) & (raw_weight > 0.0)]
    if finite_weight.size:
        lower_clip, upper_clip = np.quantile(
            finite_weight,
            [config.spread_weight_lower_quantile, config.spread_weight_upper_quantile],
        )
    else:
        lower_clip = upper_clip = 1.0
    steps["raw_calibration_weight"] = raw_weight
    steps["clipped_calibration_weight"] = np.clip(raw_weight, lower_clip, upper_clip)
    steps["calibration_weight"] = np.nan
    for _, positions in steps.groupby("cross_section_id", sort=False).groups.items():
        weights = steps.loc[positions, "clipped_calibration_weight"].to_numpy(dtype=np.float64)
        if not np.isfinite(weights).all() or np.any(weights <= 0.0):
            weights = np.ones(len(positions), dtype=np.float64)
        weights /= weights.sum()
        steps.loc[positions, "calibration_weight"] = weights
    if not np.isfinite(steps["calibration_weight"]).all():
        raise RuntimeError("Internal error: calibration weights contain nonfinite values")
    steps = steps.sort_values(
        ["segment_id", "date", "exdate", "strike"], kind="mergesort"
    ).reset_index(drop=True)
    diagnostics = {
        "num_input_nonterminal_quote": num_input,
        "num_invalid_quote_removed": int(num_invalid),
        "num_duplicate_quote_removed": num_duplicate,
        "num_retained_quote": int(len(steps)),
        "num_candidate_cross_section": int(len(grouped_sections)),
        "num_valid_cross_section": int(len(cross_section_records)),
        "alpha_failure_counts": alpha_failure_counts,
        "weight_clip_lower": float(lower_clip),
        "weight_clip_upper": float(upper_clip),
        "alpha_method_counts": dict(
            pd.Series([record["alpha_method"] for record in cross_section_records]).value_counts()
        ),
    }
    return (steps, diagnostics)


def _validate_pooled_coverage(quotes: pd.DataFrame, config: SimulationConfig) -> str | None:
    """Validate pooled coverage."""
    x = quotes["log_forward_moneyness"].to_numpy(dtype=np.float64)
    if not np.any(x < 0.0) or not np.any(x > 0.0):
        return "pooled_quotes_do_not_cover_both_sides_of_atm"
    if len(np.unique(np.round(x, decimals=12))) < config.min_log_moneyness_levels:
        return "insufficient_distinct_log_moneyness"
    return None


def _fit_parameter_group(
    quotes: pd.DataFrame, config: SimulationConfig, *, scope: str, segment_id: int
) -> tuple[dict[str, Any] | None, list[dict[str, Any]], str | None]:
    """Fit parameter group."""
    coverage_error = _validate_pooled_coverage(quotes, config)
    if coverage_error is not None:
        return (None, [], coverage_error)
    forward = quotes["forward"].to_numpy(dtype=np.float64)
    strike = quotes["strike"].to_numpy(dtype=np.float64)
    tau = quotes["tau"].to_numpy(dtype=np.float64)
    alpha = quotes["alpha_atm"].to_numpy(dtype=np.float64)
    market = quotes["resolved_iv"].to_numpy(dtype=np.float64)
    weights = quotes["calibration_weight"].to_numpy(dtype=np.float64)
    sqrt_weights = np.sqrt(weights)

    def get_residuals(parameters: np.ndarray) -> np.ndarray:
        """Return residuals."""
        model = get_lognormal_sabr_iv(
            forward,
            strike,
            tau,
            alpha,
            float(parameters[0]),
            float(parameters[1]),
            z_tolerance=config.sabr_z_tolerance,
        )
        if not np.isfinite(model).all() or np.any(model <= 0.0):
            return np.full_like(market, 1000000.0)
        return sqrt_weights * (market - model)

    run_records: list[dict[str, Any]] = []
    successful: list[tuple[float, Any, np.ndarray]] = []
    bounds = (
        np.asarray([config.rho_lower, config.nu_lower]),
        np.asarray([config.rho_upper, config.nu_upper]),
    )
    for run_id, (rho_initial, nu_initial) in enumerate(
        product(config.rho_initial_values, config.nu_initial_values), start=1
    ):
        try:
            result = least_squares(
                get_residuals,
                x0=np.asarray([rho_initial, nu_initial], dtype=np.float64),
                bounds=bounds,
                max_nfev=config.optimizer_max_nfev,
                ftol=1e-12,
                xtol=1e-12,
                gtol=1e-12,
            )
            model = get_lognormal_sabr_iv(
                forward,
                strike,
                tau,
                alpha,
                float(result.x[0]),
                float(result.x[1]),
                z_tolerance=config.sabr_z_tolerance,
            )
            is_model_valid = bool(np.isfinite(model).all() and np.all(model > 0.0))
            objective = float(np.sum(get_residuals(result.x) ** 2))
            is_success = bool(result.success and is_model_valid)
            run_records.append(
                {
                    "scope": scope,
                    "segment_id": segment_id,
                    "run_id": run_id,
                    "rho_initial": rho_initial,
                    "nu_initial": nu_initial,
                    "rho": float(result.x[0]),
                    "nu": float(result.x[1]),
                    "objective": objective,
                    "is_success": is_success,
                    "optimizer_status": int(result.status),
                    "optimizer_message": str(result.message),
                    "num_function_evaluation": int(result.nfev),
                    "optimality": float(result.optimality),
                }
            )
            if is_success:
                successful.append((objective, result, model))
        except (FloatingPointError, ValueError) as exc:
            run_records.append(
                {
                    "scope": scope,
                    "segment_id": segment_id,
                    "run_id": run_id,
                    "rho_initial": rho_initial,
                    "nu_initial": nu_initial,
                    "rho": None,
                    "nu": None,
                    "objective": None,
                    "is_success": False,
                    "optimizer_status": None,
                    "optimizer_message": f"{type(exc).__name__}: {exc}",
                    "num_function_evaluation": 0,
                    "optimality": None,
                }
            )
    if not successful:
        return (None, run_records, "all_optimizer_runs_failed")
    objective, best, model = min(successful, key=lambda item: item[0])
    jacobian_condition = float(np.linalg.cond(best.jac))
    if not np.isfinite(jacobian_condition):
        jacobian_condition = float("inf")
    touches_boundary = bool(
        abs(best.x[0] - config.rho_lower) <= config.optimizer_boundary_tolerance
        or abs(best.x[0] - config.rho_upper) <= config.optimizer_boundary_tolerance
        or abs(best.x[1] - config.nu_lower) <= config.optimizer_boundary_tolerance
        or (abs(best.x[1] - config.nu_upper) <= config.optimizer_boundary_tolerance)
    )
    summary = {
        "rho": float(best.x[0]),
        "nu": float(best.x[1]),
        "objective": objective,
        "num_quote": int(len(quotes)),
        "num_cross_section": int(quotes["cross_section_id"].nunique()),
        "jacobian_condition_number": jacobian_condition,
        "is_parameter_on_boundary": touches_boundary,
        "best_run_id": int(
            np.argmin(
                [record["objective"] if record["is_success"] else np.inf for record in run_records]
            )
            + 1
        ),
    }
    return (summary, run_records, None)


def _evaluate_fixed_parameter_group(
    quotes: pd.DataFrame, config: SimulationConfig, *, rho: float, nu: float
) -> dict[str, float | int]:
    """Evaluate fixed SABR parameters on one quote group without refitting."""
    model = get_lognormal_sabr_iv(
        quotes["forward"].to_numpy(dtype=np.float64),
        quotes["strike"].to_numpy(dtype=np.float64),
        quotes["tau"].to_numpy(dtype=np.float64),
        quotes["alpha_atm"].to_numpy(dtype=np.float64),
        rho,
        nu,
        z_tolerance=config.sabr_z_tolerance,
    )
    market = quotes["resolved_iv"].to_numpy(dtype=np.float64)
    weights = quotes["calibration_weight"].to_numpy(dtype=np.float64)
    errors = market - model
    total_weight = float(weights.sum())
    if not np.isfinite(model).all() or np.any(model <= 0.0) or total_weight <= 0.0:
        raise SabrCalibrationError("fixed global SABR parameters produced invalid IVs")
    return {
        "objective": float(np.sum(weights * errors**2)),
        "num_quote": int(len(quotes)),
        "num_cross_section": int(quotes["cross_section_id"].nunique()),
    }


def _get_daily_spot_returns(
    dataset: OptionDataset, seg_periods: Sequence[Sequence[Any]]
) -> tuple[pd.DataFrame, float]:
    """Return daily spot returns."""
    raw = dataset.episode_steps.loc[:, ["date", "spot"]].copy()
    raw["date"] = pd.to_datetime(raw["date"], errors="raise").dt.normalize()
    raw["spot"] = pd.to_numeric(raw["spot"], errors="coerce")
    grouped = raw.groupby("date", sort=True)["spot"]
    spread = grouped.max() - grouped.min()
    if np.any(spread.to_numpy(dtype=np.float64) > 1e-10):
        raise SabrCalibrationError("Real spot values differ on the same trading day")
    daily = grouped.first().rename("spot").reset_index()
    if not np.isfinite(daily["spot"]).all() or np.any(daily["spot"] <= 0.0):
        raise SabrCalibrationError("Real daily spot values are nonfinite or nonpositive")
    daily["segment_id"] = _assign_segment_ids(daily["date"], seg_periods)
    daily["previous_segment_id"] = daily["segment_id"].shift(1)
    daily["log_return"] = np.log(daily["spot"]).diff()
    same_segment = daily["segment_id"].eq(daily["previous_segment_id"])
    daily.loc[~same_segment, "log_return"] = np.nan
    train_returns = np.log(daily["spot"]).diff().dropna()
    if train_returns.empty:
        raise SabrCalibrationError("Too few training spot observations to estimate drift")
    return (daily, float(train_returns.mean()))


def calibrate_segmented_sabr(
    dataset: OptionDataset, seg_periods: Sequence[Sequence[Any]], config: SimulationConfig
) -> SabrCalibrationResult:
    """Fit spread-weighted lognormal SABR parameters per regime with global fallback."""
    periods = normalize_seg_periods(seg_periods)
    quotes, preparation = prepare_sabr_cross_sections(dataset, periods, config)
    product_groups = quotes.loc[:, ["symbol_root", "cp_flag"]].drop_duplicates()
    if len(product_groups) != 1:
        raise SabrCalibrationError(
            f"A run must contain exactly one symbol_root/cp_flag pair; received {product_groups.to_dict(orient='records')}"
        )
    global_fit, global_runs, global_error = _fit_parameter_group(
        quotes, config, scope="global", segment_id=0
    )
    if global_fit is None:
        raise SabrCalibrationError(
            f"Global training-data SABR fallback calibration failed: {global_error}"
        )
    daily, train_mean_return = _get_daily_spot_returns(dataset, periods)
    global_fit = {
        **global_fit,
        "annual_log_drift": float(config.annualization_days * train_mean_return),
        "train_mean_daily_log_return": train_mean_return,
        "num_drift_return": int(np.log(daily["spot"]).diff().notna().sum()),
    }
    parameter_records: list[dict[str, Any]] = []
    optimization_records = list(global_runs)
    num_fallback = 0
    for segment_id, (start, end) in enumerate(periods, start=1):
        segment_quotes = quotes.loc[quotes["segment_id"].eq(segment_id)]
        fit, runs, error = _fit_parameter_group(
            segment_quotes, config, scope="segment", segment_id=segment_id
        )
        optimization_records.extend(runs)
        is_fallback = fit is None
        if is_fallback:
            num_fallback += 1
            fixed_metrics = _evaluate_fixed_parameter_group(
                segment_quotes, config, rho=float(global_fit["rho"]), nu=float(global_fit["nu"])
            )
            fit = {**global_fit, **fixed_metrics}
        segment_returns = daily.loc[daily["segment_id"].eq(segment_id), "log_return"].dropna()
        num_return = int(len(segment_returns))
        if num_return:
            segment_mean = float(segment_returns.mean())
        else:
            segment_mean = train_mean_return
        shrinkage = num_return / (num_return + config.drift_shrinkage_observation)
        annual_log_drift = config.annualization_days * (
            shrinkage * segment_mean + (1.0 - shrinkage) * train_mean_return
        )
        parameter_records.append(
            {
                "segment_id": segment_id,
                "date_start": start,
                "date_end": end,
                "rho": float(fit["rho"]),
                "nu": float(fit["nu"]),
                "annual_log_drift": float(annual_log_drift),
                "segment_mean_daily_log_return": segment_mean,
                "train_mean_daily_log_return": train_mean_return,
                "drift_shrinkage_weight": float(shrinkage),
                "num_drift_return": num_return,
                "calibration_scope": "global" if is_fallback else "segment",
                "is_global_fallback": is_fallback,
                "fallback_reason": error,
                "objective": float(fit["objective"]),
                "num_calibration_quote": int(fit["num_quote"]),
                "num_calibration_cross_section": int(fit["num_cross_section"]),
                "jacobian_condition_number": float(fit["jacobian_condition_number"]),
                "is_parameter_on_boundary": bool(fit["is_parameter_on_boundary"]),
                "best_run_id": int(fit["best_run_id"]),
            }
        )
    fallback_fraction = num_fallback / len(periods)
    diagnostics = {
        "preparation": preparation,
        "product_group": product_groups.iloc[0].to_dict(),
        "global_calibration": global_fit,
        "num_segment": len(periods),
        "num_global_fallback_segment": num_fallback,
        "global_fallback_fraction": fallback_fraction,
        "fallback_task_fraction_threshold": config.fallback_task_fraction_threshold,
        "is_segmented_generation_valid": bool(
            fallback_fraction <= config.fallback_task_fraction_threshold
        ),
    }
    return SabrCalibrationResult(
        cross_sections=quotes,
        segment_parameters=pd.DataFrame(parameter_records),
        optimization_runs=pd.DataFrame(optimization_records),
        diagnostics=diagnostics,
    )


def get_global_calibration_baseline(
    calibration: SabrCalibrationResult, config: SimulationConfig
) -> SabrCalibrationResult:
    """Build a transient all-period parameter model on segmented quote groups."""
    if not isinstance(calibration, SabrCalibrationResult):
        raise TypeError("calibration must be SabrCalibrationResult")
    global_fit = calibration.diagnostics.get("global_calibration")
    if not isinstance(global_fit, dict):
        raise ValueError("calibration diagnostics are missing global_calibration")
    records: list[dict[str, Any]] = []
    for source in calibration.segment_parameters.to_dict(orient="records"):
        segment_id = int(source["segment_id"])
        quotes = calibration.cross_sections.loc[
            calibration.cross_sections["segment_id"].eq(segment_id)
        ]
        fixed = _evaluate_fixed_parameter_group(
            quotes, config, rho=float(global_fit["rho"]), nu=float(global_fit["nu"])
        )
        record = dict(source)
        record.update(
            {
                "rho": float(global_fit["rho"]),
                "nu": float(global_fit["nu"]),
                "annual_log_drift": float(global_fit["annual_log_drift"]),
                "segment_mean_daily_log_return": float(global_fit["train_mean_daily_log_return"]),
                "drift_shrinkage_weight": 0.0,
                "num_drift_return": int(global_fit["num_drift_return"]),
                "calibration_scope": "global_baseline",
                "is_global_fallback": False,
                "fallback_reason": None,
                "objective": fixed["objective"],
                "num_calibration_quote": fixed["num_quote"],
                "num_calibration_cross_section": fixed["num_cross_section"],
                "jacobian_condition_number": float(global_fit["jacobian_condition_number"]),
                "is_parameter_on_boundary": bool(global_fit["is_parameter_on_boundary"]),
                "best_run_id": int(global_fit["best_run_id"]),
            }
        )
        records.append(record)
    diagnostics = deepcopy(calibration.diagnostics)
    diagnostics["model_scope"] = "global_baseline"
    return SabrCalibrationResult(
        cross_sections=calibration.cross_sections,
        segment_parameters=pd.DataFrame(records),
        optimization_runs=calibration.optimization_runs,
        diagnostics=diagnostics,
    )
