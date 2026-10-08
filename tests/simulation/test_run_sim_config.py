"""Validate the second-stage calibration-data configuration contract."""

from __future__ import annotations
import pytest
import run_sim
from run_sim import get_default_sim_run_config, get_sim_run_config
from market_simulation import SimulationConfig
from .helpers import get_segmentation_result, get_simulation_test_dataset


def _write_required_market_files(data_root) -> None:
    data_root.mkdir(parents=True)
    for filename in run_sim._REQUIRED_MARKET_DATA_FILES:
        (data_root / filename).touch()


def test_new_calibration_and_simulation_defaults() -> None:
    config = SimulationConfig()
    assert config.num_simulation_times == 1
    assert config.expiry_weekday_calibration is None
    assert config.num_moneyness_calibration == 3
    assert get_default_sim_run_config()["simulation"] == config.get_dict()


def test_calibration_cli_style_overrides_are_validated() -> None:
    config = get_sim_run_config(
        set_overrides=(
            "simulation.expiry_weekday_calibration=null",
            "simulation.num_moneyness_calibration=40",
            "simulation.num_simulation_times=2",
        )
    )
    assert config["simulation"]["expiry_weekday_calibration"] is None
    assert config["simulation"]["num_moneyness_calibration"] == 40
    assert config["simulation"]["num_simulation_times"] == 2
    with pytest.raises(ValueError, match="expiry_weekday_calibration"):
        SimulationConfig(expiry_weekday_calibration=7)


@pytest.mark.parametrize("value", ["None", "none", "null", "NULL"])
def test_expiry_weekday_cli_accepts_all_weekdays(value: str, tmp_path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text('{"simulation": {"expiry_weekday_calibration": 4}}', encoding="utf-8")
    parser = run_sim._get_argument_parser()
    args = parser.parse_args(["--expiry-weekday-calibration", value])
    overrides = run_sim._apply_cli_overrides(args.set_overrides, args)
    assert args.expiry_weekday_calibration is None
    assert "simulation.expiry_weekday_calibration=null" in overrides
    assert (
        get_sim_run_config(config_path, set_overrides=overrides)["simulation"][
            "expiry_weekday_calibration"
        ]
        is None
    )


def test_omitted_expiry_weekday_cli_keeps_config_file_value(tmp_path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text('{"simulation": {"expiry_weekday_calibration": 4}}', encoding="utf-8")
    parser = run_sim._get_argument_parser()
    args = parser.parse_args([])
    overrides = run_sim._apply_cli_overrides(args.set_overrides, args)
    assert not hasattr(args, "expiry_weekday_calibration")
    assert (
        get_sim_run_config(config_path, set_overrides=overrides)["simulation"][
            "expiry_weekday_calibration"
        ]
        == 4
    )


def test_dataset_rebuild_overrides_only_calibration_selectors(monkeypatch) -> None:
    dataset, periods = get_simulation_test_dataset()
    segmentation_result = get_segmentation_result(dataset, periods)
    captured = {}
    sentinel = object()

    def fake_get_option_dataset(date_period, **kwargs):
        captured["date_period"] = date_period
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(run_sim, "get_option_dataset", fake_get_option_dataset)
    result = run_sim.get_train_dataset_from_seg_result(
        segmentation_result, expiry_weekday=4, num_moneyness=30
    )
    assert result is sentinel
    assert captured["expiry_weekday"] == 4
    assert captured["num_moneyness"] == 30
    assert captured["num_interval"] == dataset.config.num_interval
    assert captured["label"] == "train"


def test_dataset_rebuild_relocates_foreign_absolute_data_root(tmp_path, monkeypatch) -> None:
    dataset, periods = get_simulation_test_dataset()
    segmentation_result = get_segmentation_result(dataset, periods)
    segmentation_result["dataset"]["config"][
        "data_root"
    ] = "Z:\\archived_project\\anonymous_project\\data"
    repository_root = tmp_path / "pycharm_project_711"
    local_data_root = repository_root / "data"
    _write_required_market_files(local_data_root)
    captured = {}
    sentinel = object()

    def fake_get_option_dataset(date_period, **kwargs):
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(run_sim, "REPOSITORY_ROOT", repository_root)
    monkeypatch.setattr(run_sim, "get_option_dataset", fake_get_option_dataset)
    result = run_sim.get_train_dataset_from_seg_result(segmentation_result)
    assert result is sentinel
    assert captured["data_root"] == local_data_root


def test_windows_absolute_data_root_is_recognized_cross_platform() -> None:
    parts, is_absolute = run_sim._get_persisted_path_parts("C:/external_data")
    assert is_absolute is True
    assert parts[-1] == "external_data"


def test_persisted_data_root_preserves_available_native_absolute_path(tmp_path) -> None:
    external_data_root = tmp_path / "external_data"
    external_data_root.mkdir()
    assert run_sim._resolve_persisted_data_root(external_data_root) == external_data_root


def test_persisted_data_root_rejects_unresolvable_foreign_path(tmp_path, monkeypatch) -> None:
    repository_root = tmp_path / "pycharm_project_711"
    repository_root.mkdir()
    monkeypatch.setattr(run_sim, "REPOSITORY_ROOT", repository_root)
    with pytest.raises(FileNotFoundError, match="cannot be relocated within the current project"):
        run_sim._resolve_persisted_data_root("D:\\external\\missing_spx_data")
