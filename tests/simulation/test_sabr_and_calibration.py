"""Test sabr and calibration for the paper option-hedging pipeline."""

from __future__ import annotations
import numpy as np
import pytest
from market_simulation import (
    SimulationConfig,
    calibrate_segmented_sabr,
    get_global_calibration_baseline,
    get_lognormal_sabr_iv,
    prepare_sabr_cross_sections,
    simulate_lognormal_sabr_path,
)
from .helpers import get_simulation_test_dataset


def get_test_config(**overrides: object) -> SimulationConfig:
    """Return test config."""
    values = {
        "num_simulation_times": 1,
        "rho_initial_values": (-0.5, 0.0),
        "nu_initial_values": (0.5, 1.0),
        "num_environment_rollout_checks": 1,
    }
    values.update(overrides)
    return SimulationConfig(**values)


def test_sabr_atm_limit_and_seeded_path_are_finite() -> None:
    """Verify SABR atm limit and seeded path are finite."""
    alpha = 0.2
    rho = -0.4
    nu = 0.7
    tau = 20.0 / 365.0
    expected = alpha * (1.0 + (rho * nu * alpha / 4.0 + (2.0 - 3.0 * rho**2) * nu**2 / 24.0) * tau)
    actual = get_lognormal_sabr_iv(100.0, 100.0, tau, alpha, rho, nu)
    assert float(actual) == pytest.approx(expected)
    extreme = get_lognormal_sabr_iv(
        np.asarray([50.0, 200.0]), np.asarray([200.0, 50.0]), 0.1, 0.05, 0.9, 5.0
    )
    assert np.isfinite(extreme).all() and np.all(extreme > 0.0)
    first = simulate_lognormal_sabr_path(
        spot_initial=100.0,
        alpha_initial=alpha,
        annual_log_drift=0.05,
        rho=rho,
        nu=nu,
        num_interval=20,
        annualization_days=252,
        rng=np.random.default_rng(123),
    )
    second = simulate_lognormal_sabr_path(
        spot_initial=100.0,
        alpha_initial=alpha,
        annual_log_drift=0.05,
        rho=rho,
        nu=nu,
        num_interval=20,
        annualization_days=252,
        rng=np.random.default_rng(123),
    )
    np.testing.assert_allclose(first[0], second[0])
    np.testing.assert_allclose(first[1], second[1])
    assert np.all(first[0] > 0.0) and np.all(first[1] > 0.0)


def test_cross_sections_and_segment_calibration_are_complete() -> None:
    """Verify cross sections and segment calibration are complete."""
    dataset, periods = get_simulation_test_dataset()
    config = get_test_config()
    quotes, diagnostics = prepare_sabr_cross_sections(dataset, periods, config)
    weight_sums = quotes.groupby("cross_section_id")["calibration_weight"].sum()
    np.testing.assert_allclose(weight_sums.to_numpy(), 1.0)
    assert diagnostics["num_duplicate_quote_removed"] == 0
    assert diagnostics["num_valid_cross_section"] == 12
    calibration = calibrate_segmented_sabr(dataset, periods, config)
    assert len(calibration.segment_parameters) == 2
    assert not calibration.segment_parameters["is_global_fallback"].any()
    assert calibration.segment_parameters["rho"].between(-0.95, 0.95).all()
    assert calibration.segment_parameters["nu"].between(0.0001, 5.0).all()
    assert calibration.optimization_runs["is_success"].any()
    global_baseline = get_global_calibration_baseline(calibration, config)
    global_parameters = global_baseline.segment_parameters
    assert global_parameters["rho"].nunique() == 1
    assert global_parameters["nu"].nunique() == 1
    assert global_parameters["annual_log_drift"].nunique() == 1
    assert global_parameters["objective"].sum() == pytest.approx(
        calibration.diagnostics["global_calibration"]["objective"]
    )
