"""Test pricing for the paper option-hedging pipeline."""

from __future__ import annotations
import numpy as np
import pytest
from option_dataset import (
    get_bsm_option_price,
    get_implied_volatility,
    get_log_forward_moneyness,
    get_surface_implied_volatility,
)


def test_bsm_prices_satisfy_put_call_parity() -> None:
    """Verify BSM prices satisfy put call parity."""
    parameters = {
        "spot": 100.0,
        "strike": 105.0,
        "tau": 0.5,
        "volatility": 0.25,
        "zero_rate": 0.04,
        "dividend_rate": 0.015,
    }
    call = get_bsm_option_price(**parameters, cp_flag="C")
    put = get_bsm_option_price(**parameters, cp_flag="P")
    parity = parameters["spot"] * np.exp(
        -parameters["dividend_rate"] * parameters["tau"]
    ) - parameters["strike"] * np.exp(-parameters["zero_rate"] * parameters["tau"])
    assert call - put == pytest.approx(parity, abs=1e-12)


@pytest.mark.parametrize("cp_flag", ["C", "P"])
def test_implied_volatility_recovers_bsm_input(cp_flag: str) -> None:
    """Verify implied volatility recovers BSM input."""
    parameters = {
        "spot": 5235.48,
        "strike": 5240.0,
        "tau": 29 / 365,
        "volatility": 0.123,
        "zero_rate": 0.0537,
        "dividend_rate": 0.0111,
        "cp_flag": cp_flag,
    }
    price = get_bsm_option_price(**parameters)
    result = get_implied_volatility(
        option_price=price,
        spot=parameters["spot"],
        strike=parameters["strike"],
        tau=parameters["tau"],
        zero_rate=parameters["zero_rate"],
        dividend_rate=parameters["dividend_rate"],
        cp_flag=cp_flag,
    )
    assert result == pytest.approx(parameters["volatility"], abs=1e-08)


def test_implied_volatility_rejects_arbitrage_violating_price() -> None:
    """Verify implied volatility rejects arbitrage violating price."""
    result = get_implied_volatility(
        option_price=200.0,
        spot=100.0,
        strike=100.0,
        tau=0.1,
        zero_rate=0.03,
        dividend_rate=0.01,
        cp_flag="C",
    )
    assert np.isnan(result)


def test_surface_iv_interpolates_and_limits_extrapolation() -> None:
    """Verify surface IV interpolates and limits extrapolation."""
    interpolated = get_surface_implied_volatility(
        target_log_moneyness=0.0, surface_log_moneyness=[-0.1, 0.1], surface_iv=[0.3, 0.2]
    )
    near_extrapolated = get_surface_implied_volatility(
        target_log_moneyness=0.15,
        surface_log_moneyness=[-0.1, 0.1],
        surface_iv=[0.3, 0.2],
        max_iv_extrapolation_log_moneyness=0.1,
    )
    far_extrapolated = get_surface_implied_volatility(
        target_log_moneyness=0.25,
        surface_log_moneyness=[-0.1, 0.1],
        surface_iv=[0.3, 0.2],
        max_iv_extrapolation_log_moneyness=0.1,
    )
    assert interpolated == pytest.approx(0.25)
    assert near_extrapolated == pytest.approx(0.2)
    assert np.isnan(far_extrapolated)


def test_log_forward_moneyness_is_zero_at_forward_strike() -> None:
    """Verify log forward moneyness is zero at forward strike."""
    spot = 100.0
    tau = 0.5
    zero_rate = 0.04
    dividend_rate = 0.01
    strike = spot * np.exp((zero_rate - dividend_rate) * tau)
    result = get_log_forward_moneyness(
        spot=spot, strike=strike, tau=tau, zero_rate=zero_rate, dividend_rate=dividend_rate
    )
    assert result == pytest.approx(0.0, abs=1e-15)
