"""Check publication-scope interfaces and execution-critical input contracts."""

from __future__ import annotations
import numpy as np
import pytest
from configure_experiment import get_paper_configs
from market_simulation import (
    SimulationConfig,
    calibrate_segmented_sabr,
    generate_segmented_simulations,
    get_dataset_quality_features,
)
from market_segmentation import segment_option_dataset
from tests.simulation.helpers import get_simulation_test_dataset


@pytest.mark.parametrize("source", ["spx", "sx5e"])
def test_meta_training_cli_retains_configured_market(source, tmp_path, monkeypatch):
    """The CLI must not replace a configuration's market with a fixed default."""
    import json
    import train_meta

    config_path = tmp_path / "market.json"
    config_path.write_text(
        json.dumps({"data_source": source, "training": {"seeds": [0]}}), encoding="utf-8"
    )
    captured = {}

    def capture(config, **kwargs):
        captured.update(config)

    monkeypatch.setattr(train_meta, "run_meta_training", capture)
    assert train_meta.main(["--config", str(config_path)]) == 0
    assert captured["data_source"] == source


@pytest.mark.parametrize(
    "source,count,bounds,currency,multiplier",
    [("spx", 3, [0.95, 1.05], "USD", 100.0), ("sx5e", 20, [0.8, 1.2], "EUR", 10.0)],
)
def test_generated_configs_match_market_protocol(source, count, bounds, currency, multiplier):
    configs = get_paper_configs(source, "/external/market_data")
    for name, config in configs.items():
        assert config["data_source"] == source
        if "data" in config:
            assert config["data"]["num_moneyness"] == count
            assert config["data"]["moneyness_range"] == bounds
        if "training" in config:
            assert config["training"]["seeds"] == list(range(10))
            assert config["environment"]["currency_by_data_source"][source] == currency
            assert config["environment"]["contract_multiplier_by_data_source"][source] == multiplier
    assert configs["sim"]["simulation"]["num_moneyness_calibration"] == count
    assert configs["no_simulation"]["training"]["simulated_batch_fraction"] == 0.0


def test_quality_features_have_paper_sample_units():
    dataset, _ = get_simulation_test_dataset()
    features = get_dataset_quality_features(dataset)
    assert set(features) == {"terminal_spot_ratio", "atm_iv", "iv_change", "iv_skew"}
    assert len(features["terminal_spot_ratio"]) == dataset.episode_manifest["cohort_id"].nunique()
    assert len(features["iv_change"]) == dataset.num_episode * (dataset.config.num_interval - 1)
    expected = []
    for _, group in dataset.episode_steps.groupby("cohort_id", sort=False):
        path = group.sort_values("date").drop_duplicates("date")["spot"]
        expected.append(path.iloc[-1] / path.iloc[0])
    np.testing.assert_allclose(features["terminal_spot_ratio"], expected)


@pytest.mark.parametrize("ratio,episodes", [(0.5, 5), (1.5, 15)])
def test_fractional_augmentation_preserves_complete_shared_cohorts(ratio, episodes):
    dataset, periods = get_simulation_test_dataset()
    config = SimulationConfig(
        num_simulation_times=ratio,
        rho_initial_values=(-0.5, 0.0),
        nu_initial_values=(0.5, 1.0),
        num_environment_rollout_checks=0,
    )
    calibration = calibrate_segmented_sabr(dataset, periods, config)
    real = segment_option_dataset(dataset, periods)
    generated = generate_segmented_simulations(real, calibration, config)
    for simulated in generated.datasets:
        assert simulated.num_episode == episodes
        assert simulated.episode_manifest.groupby("cohort_id").size().eq(5).all()
        assert simulated.dataset_build_report["realized_augmentation_ratio"] == ratio
        for _, group in simulated.episode_steps.groupby(["cohort_id", "date"]):
            assert group["spot"].nunique() == 1


@pytest.mark.parametrize("ratio", [0, -1, True, float("nan"), float("inf")])
def test_simulation_rejects_invalid_augmentation_ratios(ratio):
    with pytest.raises(ValueError):
        SimulationConfig(num_simulation_times=ratio)
