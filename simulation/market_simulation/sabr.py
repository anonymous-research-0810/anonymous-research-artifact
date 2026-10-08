"""Sabr for the paper option-hedging pipeline."""

from __future__ import annotations
from typing import Any
import numpy as np
from scipy.special import ndtr


def get_lognormal_sabr_iv(
    forward: Any,
    strike: Any,
    tau: Any,
    alpha: Any,
    rho: float,
    nu: float,
    *,
    z_tolerance: float = 1e-06,
) -> np.ndarray:
    """Evaluate the lognormal SABR implied-volatility approximation, including the near-ATM limit."""
    forward_array, strike_array, tau_array, alpha_array = np.broadcast_arrays(
        np.asarray(forward, dtype=np.float64),
        np.asarray(strike, dtype=np.float64),
        np.asarray(tau, dtype=np.float64),
        np.asarray(alpha, dtype=np.float64),
    )
    result = np.full(forward_array.shape, np.nan, dtype=np.float64)
    if (
        not np.isfinite(rho)
        or not -1.0 < float(rho) < 1.0
        or (not np.isfinite(nu))
        or (float(nu) <= 0.0)
        or (not np.isfinite(z_tolerance))
        or (z_tolerance <= 0.0)
    ):
        return result
    valid = (
        np.isfinite(forward_array)
        & np.isfinite(strike_array)
        & np.isfinite(tau_array)
        & np.isfinite(alpha_array)
        & (forward_array > 0.0)
        & (strike_array > 0.0)
        & (tau_array >= 0.0)
        & (alpha_array > 0.0)
    )
    if not valid.any():
        return result
    log_fk = np.log(forward_array[valid] / strike_array[valid])
    z = float(nu) * log_fk / alpha_array[valid]
    ratio = np.ones_like(z)
    regular = np.abs(z) >= z_tolerance
    if regular.any():
        regular_z = z[regular]
        radicand = 1.0 - 2.0 * float(rho) * regular_z + regular_z**2
        root = np.sqrt(radicand)
        chi = np.empty_like(regular_z)
        nonnegative = regular_z >= 0.0
        chi[nonnegative] = np.log(
            (root[nonnegative] + regular_z[nonnegative] - float(rho)) / (1.0 - float(rho))
        )
        chi[~nonnegative] = np.log(
            (1.0 + float(rho)) / (root[~nonnegative] - regular_z[~nonnegative] + float(rho))
        )
        ratio[regular] = regular_z / chi
    correction = (
        1.0
        + (
            float(rho) * float(nu) * alpha_array[valid] / 4.0
            + (2.0 - 3.0 * float(rho) ** 2) * float(nu) ** 2 / 24.0
        )
        * tau_array[valid]
    )
    values = alpha_array[valid] * ratio * correction
    values[~np.isfinite(values) | (values <= 0.0)] = np.nan
    result[valid] = values
    return result


def get_bsm_option_prices(
    *,
    spot: Any,
    strike: Any,
    tau: Any,
    volatility: Any,
    zero_rate: Any,
    dividend_rate: Any,
    cp_flag: str,
) -> np.ndarray:
    """Return BSM option prices."""
    cp = str(cp_flag).upper()
    if cp not in {"C", "P"}:
        raise ValueError("cp_flag must be 'C' or 'P'")
    arrays = np.broadcast_arrays(
        np.asarray(spot, dtype=np.float64),
        np.asarray(strike, dtype=np.float64),
        np.asarray(tau, dtype=np.float64),
        np.asarray(volatility, dtype=np.float64),
        np.asarray(zero_rate, dtype=np.float64),
        np.asarray(dividend_rate, dtype=np.float64),
    )
    spot_array, strike_array, tau_array, vol_array, rate_array, div_array = arrays
    result = np.full(spot_array.shape, np.nan, dtype=np.float64)
    valid = (
        np.isfinite(spot_array)
        & np.isfinite(strike_array)
        & np.isfinite(tau_array)
        & np.isfinite(vol_array)
        & np.isfinite(rate_array)
        & np.isfinite(div_array)
        & (spot_array > 0.0)
        & (strike_array > 0.0)
        & (tau_array >= 0.0)
        & (vol_array > 0.0)
    )
    terminal = valid & (tau_array == 0.0)
    if cp == "C":
        result[terminal] = np.maximum(spot_array[terminal] - strike_array[terminal], 0.0)
    else:
        result[terminal] = np.maximum(strike_array[terminal] - spot_array[terminal], 0.0)
    live = valid & (tau_array > 0.0)
    if live.any():
        sqrt_tau = np.sqrt(tau_array[live])
        d1 = (
            np.log(spot_array[live] / strike_array[live])
            + (rate_array[live] - div_array[live] + 0.5 * vol_array[live] ** 2) * tau_array[live]
        ) / (vol_array[live] * sqrt_tau)
        d2 = d1 - vol_array[live] * sqrt_tau
        discounted_spot = spot_array[live] * np.exp(-div_array[live] * tau_array[live])
        discounted_strike = strike_array[live] * np.exp(-rate_array[live] * tau_array[live])
        if cp == "C":
            result[live] = discounted_spot * ndtr(d1) - discounted_strike * ndtr(d2)
        else:
            result[live] = discounted_strike * ndtr(-d2) - discounted_spot * ndtr(-d1)
    return result


def simulate_lognormal_sabr_path(
    *,
    spot_initial: float,
    alpha_initial: float,
    annual_log_drift: float,
    rho: float,
    nu: float,
    num_interval: int,
    annualization_days: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate correlated lognormal spot and volatility paths with finite positive values."""
    values = np.asarray([spot_initial, alpha_initial, annual_log_drift, rho, nu], dtype=np.float64)
    if (
        not np.isfinite(values).all()
        or spot_initial <= 0.0
        or alpha_initial <= 0.0
        or (not -1.0 < rho < 1.0)
        or (nu <= 0.0)
        or isinstance(num_interval, (bool, np.bool_))
        or (int(num_interval) != num_interval)
        or (num_interval <= 0)
        or isinstance(annualization_days, (bool, np.bool_))
        or (int(annualization_days) != annualization_days)
        or (annualization_days <= 0)
        or (not isinstance(rng, np.random.Generator))
    ):
        raise ValueError("SABR path inputs violate finiteness, positivity, or parameter bounds")
    num_interval = int(num_interval)
    delta_time = 1.0 / int(annualization_days)
    sqrt_delta = np.sqrt(delta_time)
    independent = rng.standard_normal((num_interval, 2))
    spot_shocks = independent[:, 0]
    alpha_shocks = rho * independent[:, 0] + np.sqrt(1.0 - rho**2) * independent[:, 1]
    spots = np.empty(num_interval + 1, dtype=np.float64)
    alphas = np.empty(num_interval + 1, dtype=np.float64)
    spots[0] = float(spot_initial)
    alphas[0] = float(alpha_initial)
    for step in range(num_interval):
        spots[step + 1] = spots[step] * np.exp(
            annual_log_drift * delta_time + alphas[step] * sqrt_delta * spot_shocks[step]
        )
        alphas[step + 1] = alphas[step] * np.exp(
            -0.5 * nu**2 * delta_time + nu * sqrt_delta * alpha_shocks[step]
        )
    return (spots, alphas)
