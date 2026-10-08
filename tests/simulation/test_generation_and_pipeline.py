"""Test generation and pipeline for the paper option-hedging pipeline."""

from __future__ import annotations
import json
import numpy as np
import pytest
from market_segmentation import segment_option_dataset
import market_simulation.pipeline as simulation_pipeline
from market_simulation import (
    SimulationConfig,
    calibrate_segmented_sabr,
    evaluate_segmented_simulations,
    generate_segmented_simulations,
    load_segmented_simulated_datasets,
    run_environment_rollout_checks,
    run_segmented_sabr_simulation,
)
from .helpers import get_segmentation_result, get_simulation_test_dataset


def get_test_config(**overrides: object) -> SimulationConfig:
    """Return test config."""
    values = dict(
        num_simulation_times=1,
        rho_initial_values=(-0.5, 0.0),
        nu_initial_values=(0.5, 1.0),
        num_environment_rollout_checks=1,
        max_path_retry=20,
    )
    values.update(overrides)
    return SimulationConfig(**values)


def test_generation_preserves_exact_multiple_and_environment_contract() -> None:
    """Verify generation preserves exact multiple and environment contract."""
    dataset, periods = get_simulation_test_dataset()
    real_parts = segment_option_dataset(dataset, periods)
    config = get_test_config()
    calibration = calibrate_segmented_sabr(dataset, periods, config)
    generation = generate_segmented_simulations(real_parts, calibration, config)
    assert len(generation.datasets) == 2
    for real, simulated in zip(real_parts, generation.datasets):
        assert simulated.num_episode == real.num_episode
        assert simulated.episode_steps["episode_id"].str.startswith("SIM_").all()
        assert simulated.episode_steps["optionid"].lt(0).all()
        assert simulated.episode_steps["is_simulated"].all()
        live = simulated.episode_steps.loc[~simulated.episode_steps["is_terminal"]]
        assert live["relative_spread"].between(0.0, 0.1).all()
        np.testing.assert_allclose(
            (live["best_offer"] - live["best_bid"]) / live["quoted_mid_price"],
            live["relative_spread"],
            rtol=1e-12,
            atol=1e-12,
        )
        assert run_environment_rollout_checks(simulated, config, seed=7)["status"] == "completed"
    quality = evaluate_segmented_simulations(real_parts, generation.datasets, config)
    compared = quality["segments"][0]["distribution_comparison"]
    assert set(compared) == {"terminal_spot_ratio", "atm_iv", "iv_change", "iv_skew"}


def test_cross_boundary_episodes_are_not_used_as_simulation_anchors() -> None:
    """Verify cross boundary episodes are not used as simulation anchors."""
    dataset, _ = get_simulation_test_dataset()
    dates = sorted(dataset.episode_steps["date"].drop_duplicates())
    periods = [
        (dates[0].strftime("%Y-%m-%d"), dates[4].strftime("%Y-%m-%d")),
        (dates[5].strftime("%Y-%m-%d"), dates[-1].strftime("%Y-%m-%d")),
    ]
    real_parts = segment_option_dataset(dataset, periods)
    config = get_test_config()
    calibration = calibrate_segmented_sabr(dataset, periods, config)
    generation = generate_segmented_simulations(real_parts, calibration, config)
    first_diagnostics = generation.diagnostics["segments"][0]
    assert first_diagnostics["num_cross_boundary_anchor_excluded"] == 5
    first_steps = generation.datasets[0].episode_steps
    assert first_steps["date"].max() <= dates[4]
    assert generation.datasets[0].num_episode == real_parts[0].num_episode


def test_pipeline_writes_and_reloads_independent_segment_files(tmp_path) -> None:
    """Verify pipeline writes and reloads independent segment files."""
    dataset, periods = get_simulation_test_dataset()
    segmentation_result = get_segmentation_result(dataset, periods)
    segmentation_path = tmp_path / "seg_20200101T000000Z.json"
    segmentation_path.write_text(json.dumps(segmentation_result), encoding="utf-8")
    output = run_segmented_sabr_simulation(
        dataset,
        segmentation_result,
        get_test_config(
            expiry_weekday_calibration=dataset.config.expiry_weekday,
            num_moneyness_calibration=dataset.config.num_moneyness,
        ),
        calibration_dataset=dataset,
        segmentation_result_path=segmentation_path,
        output_root=tmp_path / "sim_results",
        experiment_id="sim_20200102T000000Z",
    )
    result_directory = output["result_directory"]
    assert (result_directory / "sim_segment_01.parquet").is_file()
    assert (result_directory / "sim_segment_02.parquet").is_file()
    assert (result_directory / "sabr_cross_sections.parquet").is_file()
    assert (result_directory / "simulation_manifest.parquet").is_file()
    payload = json.loads((result_directory / "sim_result.json").read_text(encoding="utf-8"))
    assert payload["status"] == "completed"
    assert payload["generation"]["num_simulated_episode"] == dataset.num_episode
    assert payload["calibration_dataset"]["num_episode"] == dataset.num_episode
    assert payload["quality_evaluation"]["num_segment"] == 2
    assert payload["global_baseline_quality_evaluation"]["num_segment"] == 2
    assert payload["quality_comparison"]["features"]
    baseline = payload["global_baseline_generation"]
    assert baseline["num_simulated_episode"] == dataset.num_episode
    assert baseline["samples_persisted"] is False
    assert baseline["pairing_with_segmented"]["is_anchor_paired"] is True
    assert baseline["pairing_with_segmented"]["is_seed_and_retry_paired"] is True
    assert not any(("global" in artifact["filename"] for artifact in payload["artifacts"]))
    reloaded = load_segmented_simulated_datasets(result_directory)
    assert [item.num_episode for item in reloaded] == [10, 10]


def test_source_dataset_validation_allows_only_data_root_relocation() -> None:
    dataset, periods = get_simulation_test_dataset()
    segmentation_result = get_segmentation_result(dataset, periods)
    segmentation_result["dataset"]["config"]["data_root"] = "C:/external_data"
    simulation_pipeline._validate_source_dataset(dataset, segmentation_result)
    segmentation_result["dataset"]["config"]["num_interval"] += 1
    with pytest.raises(ValueError, match="config differs from the Stage 1 record"):
        simulation_pipeline._validate_source_dataset(dataset, segmentation_result)
