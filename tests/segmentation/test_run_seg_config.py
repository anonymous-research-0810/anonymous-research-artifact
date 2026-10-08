"""Test run seg config for the paper option-hedging pipeline."""

from __future__ import annotations
from run_seg import get_default_seg_run_config, get_seg_run_config
from train_basis import get_default_basis_config


def test_default_dataset_config_is_shared_with_train_basis() -> None:
    """Verify default dataset config is shared with train basis."""
    assert get_default_seg_run_config()["data"] == get_default_basis_config()["data"]


def test_set_override_supports_dict_and_list_leaves() -> None:
    """Verify set override supports dict and list leaves."""
    config = get_seg_run_config(
        set_overrides=(
            "data.num_interval=7",
            "data.expiry_weekday=null",
            "data.moneyness_range.0=0.9",
            "segmentation.algorithm=rbf",
        )
    )
    assert config["data"]["num_interval"] == 7
    assert config["data"]["expiry_weekday"] is None
    assert config["data"]["moneyness_range"] == [0.9, 1.05]
    assert config["segmentation"]["algorithm"] == "rbf"
