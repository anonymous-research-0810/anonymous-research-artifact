"""Pricing for the paper option-hedging pipeline."""

from __future__ import annotations
from collections.abc import Sequence
import numpy as np
from scipy.optimize import brentq
from scipy.special import ndtr
from .exceptions import DatasetConfigurationError


def get_bsm_option_price(
    *,
    spot: float,
    strike: float,
    tau: float,
    volatility: float,
    zero_rate: float,
    dividend_rate: float,
    cp_flag: str,
) -> float:
    """Price a European option with continuously compounded interest and dividend yields."""
    cp_flag = str(cp_flag).upper()
    if cp_flag not in {"C", "P"}:
        raise DatasetConfigurationError("cp_flag must be 'C' or 'P'")
    values = np.asarray([spot, strike, tau, volatility, zero_rate, dividend_rate], dtype=float)
    if not np.isfinite(values).all() or spot <= 0 or strike <= 0 or (tau < 0):
        return float("nan")
    if tau == 0:
        payoff = max(spot - strike, 0.0) if cp_flag == "C" else max(strike - spot, 0.0)
        return float(payoff)
    if volatility <= 0:
        discounted_spot = spot * np.exp(-dividend_rate * tau)
        discounted_strike = strike * np.exp(-zero_rate * tau)
        lower = discounted_spot - discounted_strike
        return float(max(lower, 0.0) if cp_flag == "C" else max(-lower, 0.0))
    num_sqrt_tau = np.sqrt(tau)
    d1 = (np.log(spot / strike) + (zero_rate - dividend_rate + 0.5 * volatility**2) * tau) / (
        volatility * num_sqrt_tau
    )
    d2 = d1 - volatility * num_sqrt_tau
    discounted_spot = spot * np.exp(-dividend_rate * tau)
    discounted_strike = strike * np.exp(-zero_rate * tau)
    if cp_flag == "C":
        return float(discounted_spot * ndtr(d1) - discounted_strike * ndtr(d2))
    return float(discounted_strike * ndtr(-d2) - discounted_spot * ndtr(-d1))


def get_implied_volatility(
    *,
    option_price: float,
    spot: float,
    strike: float,
    tau: float,
    zero_rate: float,
    dividend_rate: float,
    cp_flag: str,
    min_iv: float = 1e-06,
    max_iv: float = 5.0,
    iv_solver_tolerance: float = 1e-08,
    num_iv_solver_iterations: int = 100,
) -> float:
    """Solve the BSM implied volatility within validated no-arbitrage bounds."""
    values = np.asarray([option_price, spot, strike, tau, zero_rate, dividend_rate], dtype=float)
    if (
        not np.isfinite(values).all()
        or option_price < 0
        or spot <= 0
        or (strike <= 0)
        or (tau <= 0)
        or (not 0 < min_iv < max_iv)
    ):
        return float("nan")
    discounted_spot = spot * np.exp(-dividend_rate * tau)
    discounted_strike = strike * np.exp(-zero_rate * tau)
    cp_flag = str(cp_flag).upper()
    if cp_flag == "C":
        lower = max(discounted_spot - discounted_strike, 0.0)
        upper = discounted_spot
    elif cp_flag == "P":
        lower = max(discounted_strike - discounted_spot, 0.0)
        upper = discounted_strike
    else:
        return float("nan")
    if option_price < lower - iv_solver_tolerance or option_price > upper + iv_solver_tolerance:
        return float("nan")
    price_at_min = get_bsm_option_price(
        spot=spot,
        strike=strike,
        tau=tau,
        volatility=min_iv,
        zero_rate=zero_rate,
        dividend_rate=dividend_rate,
        cp_flag=cp_flag,
    )
    price_at_max = get_bsm_option_price(
        spot=spot,
        strike=strike,
        tau=tau,
        volatility=max_iv,
        zero_rate=zero_rate,
        dividend_rate=dividend_rate,
        cp_flag=cp_flag,
    )
    if option_price <= price_at_min + iv_solver_tolerance:
        return float(min_iv)
    if option_price >= price_at_max - iv_solver_tolerance:
        return float(max_iv) if option_price <= price_at_max + iv_solver_tolerance else float("nan")

    def get_price_error(volatility: float) -> float:
        """Return price error."""
        return (
            get_bsm_option_price(
                spot=spot,
                strike=strike,
                tau=tau,
                volatility=volatility,
                zero_rate=zero_rate,
                dividend_rate=dividend_rate,
                cp_flag=cp_flag,
            )
            - option_price
        )

    try:
        return float(
            brentq(
                get_price_error,
                min_iv,
                max_iv,
                xtol=iv_solver_tolerance,
                rtol=max(iv_solver_tolerance, 4 * np.finfo(float).eps),
                maxiter=num_iv_solver_iterations,
            )
        )
    except (RuntimeError, ValueError):
        return float("nan")


def get_log_forward_moneyness(
    *, spot: float, strike: float, tau: float, zero_rate: float, dividend_rate: float
) -> float:
    """Return log forward moneyness."""
    values = np.asarray([spot, strike, tau, zero_rate, dividend_rate], dtype=float)
    if not np.isfinite(values).all() or spot <= 0 or strike <= 0 or (tau < 0):
        return float("nan")
    forward = spot * np.exp((zero_rate - dividend_rate) * tau)
    return float(np.log(strike / forward))


def get_surface_implied_volatility(
    *,
    target_log_moneyness: float,
    surface_log_moneyness: Sequence[float],
    surface_iv: Sequence[float],
    num_iv_surface_points: int = 2,
    max_iv_extrapolation_log_moneyness: float = 0.1,
) -> float:
    """Recover IV by interpolation in log forward moneyness with bounded extrapolation."""
    if isinstance(num_iv_surface_points, bool) or num_iv_surface_points < 2:
        raise DatasetConfigurationError("num_iv_surface_points must be at least 2")
    if max_iv_extrapolation_log_moneyness < 0:
        raise DatasetConfigurationError("max_iv_extrapolation_log_moneyness must be nonnegative")
    x = np.asarray(surface_log_moneyness, dtype=float)
    y = np.asarray(surface_iv, dtype=float)
    is_valid = np.isfinite(x) & np.isfinite(y) & (y > 0)
    x = x[is_valid]
    y = y[is_valid]
    if not np.isfinite(target_log_moneyness) or len(x) < num_iv_surface_points:
        return float("nan")
    order = np.argsort(x, kind="mergesort")
    x = x[order]
    y = y[order]
    unique_x, inverse = np.unique(x, return_inverse=True)
    grouped_y = np.asarray(
        [np.median(y[inverse == num_group]) for num_group in range(len(unique_x))]
    )
    if len(unique_x) < num_iv_surface_points:
        return float("nan")
    if target_log_moneyness < unique_x[0]:
        distance = unique_x[0] - target_log_moneyness
        return (
            float(grouped_y[0]) if distance <= max_iv_extrapolation_log_moneyness else float("nan")
        )
    if target_log_moneyness > unique_x[-1]:
        distance = target_log_moneyness - unique_x[-1]
        return (
            float(grouped_y[-1]) if distance <= max_iv_extrapolation_log_moneyness else float("nan")
        )
    return float(np.interp(target_log_moneyness, unique_x, grouped_y))
