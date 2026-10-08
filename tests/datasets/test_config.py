"""Test config for the paper option-hedging pipeline."""

from __future__ import annotations
import pytest
from option_dataset import DatasetConfigurationError, get_option_dataset_config


def test_get_config_normalizes_public_values() -> None:
    """Verify get config normalizes public values."""
    config = get_option_dataset_config(
        date_period=["2016-01-04", "2022-12-30"],
        label="TRAIN",
        cp_flag="c",
        symbol_start="spxw",
        moneyness_range=[0.95, 1.05],
    )
    assert config.label == "train"
    assert config.cp_flag == "C"
    assert config.symbol_start == "SPXW"
    assert config.num_interval == 20
    assert config.expiry_weekday is None
    assert config.num_moneyness == 3
    assert config.min_open_interest == 0
    assert config.years == tuple(range(2016, 2023))


@pytest.mark.parametrize("label", ["validation", "dev", ""])
def test_get_config_rejects_unknown_label(label: str) -> None:
    """Verify get config rejects unknown label."""
    with pytest.raises(DatasetConfigurationError, match="label"):
        get_option_dataset_config(date_period=["2024-01-01", "2024-12-31"], label=label)


def test_get_config_rejects_moneyness_range_not_containing_atm() -> None:
    """Verify get config rejects moneyness range not containing atm."""
    with pytest.raises(DatasetConfigurationError, match="moneyness_range"):
        get_option_dataset_config(
            date_period=["2024-01-01", "2024-12-31"], label="test", moneyness_range=[1.05, 1.2]
        )


def test_get_config_requires_explicit_opt_out_for_standard_spx() -> None:
    """Verify get config requires explicit opt out for standard spx."""
    with pytest.raises(DatasetConfigurationError, match="PM settlement"):
        get_option_dataset_config(
            date_period=["2024-01-01", "2024-12-31"], label="test", symbol_start="SPX"
        )
    config = get_option_dataset_config(
        date_period=["2024-01-01", "2024-12-31"],
        label="test",
        symbol_start="SPX",
        is_require_environment_ready=False,
    )
    assert config.symbol_start == "SPX"
    assert not config.is_require_environment_ready


def test_get_config_rejects_non_boolean_is_parameter() -> None:
    """Verify get config rejects non boolean is parameter."""
    with pytest.raises(DatasetConfigurationError, match="must be a boolean"):
        get_option_dataset_config(
            date_period=["2024-01-01", "2024-12-31"], label="test", is_use_surface_iv=1
        )
