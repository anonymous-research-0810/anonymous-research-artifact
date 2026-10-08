"""Test real dataset for the paper option-hedging pipeline."""

from __future__ import annotations
from pathlib import Path
from functools import lru_cache
import os
import numpy as np
import pandas as pd
import pytest
from option_dataset import EpisodeValidationError, get_option_dataset, get_option_dataset_config
from option_dataset.builder import OptionDatasetBuilder

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("MDH_TEST_DATA_ROOT", REPOSITORY_ROOT / "data"))
IS_REAL_DATA_AVAILABLE = (DATA_ROOT / "spx_options_2024.parquet").is_file()


@lru_cache(maxsize=1)
def _get_real_dataset():
    """Return real dataset."""
    return get_option_dataset(
        date_period=["2024-05-30", "2024-06-28"], label="test", data_root=DATA_ROOT
    )


@pytest.mark.integration
@pytest.mark.skipif(not IS_REAL_DATA_AVAILABLE, reason="local SPX parquet data unavailable")
def test_real_dataset_builds_expected_balanced_cohort() -> None:
    """Verify real dataset builds expected balanced cohort."""
    dataset = _get_real_dataset()
    assert dataset.label == "test"
    assert dataset.num_episode == 3
    assert dataset.num_step == 63
    assert dataset.is_environment_ready
    assert len(set(dataset.get_episode_ids())) == dataset.num_episode
    assert dataset.episode_steps.groupby("episode_id").size().eq(21).all()
    assert (
        dataset.episode_steps.groupby("episode_id")["step"]
        .apply(lambda steps: steps.tolist() == list(range(21)))
        .all()
    )
    assert np.isfinite(dataset.episode_manifest["c0"]).all()
    assert dataset.episode_manifest["selection_rank"].tolist() == [1, 2, 3]
    decision_steps = dataset.episode_steps.loc[~dataset.episode_steps["is_terminal"]]
    terminal_steps = dataset.episode_steps.loc[dataset.episode_steps["is_terminal"]]
    assert np.isfinite(decision_steps["mid_price"]).all()
    assert np.isfinite(decision_steps["resolved_iv"]).all()
    assert terminal_steps["resolved_iv"].isna().all()
    assert terminal_steps["tau"].eq(0).all()
    expected_payoff = np.maximum(terminal_steps["spot"] - terminal_steps["strike"], 0.0)
    assert np.allclose(terminal_steps["mid_price"], expected_payoff)
    assert decision_steps["iv_source"].notna().all()
    state_array = dataset.get_exogenous_state_array(0)
    assert state_array.shape == (20, 6)
    assert state_array.dtype == np.float32
    assert np.isfinite(state_array).all()


@pytest.mark.integration
@pytest.mark.skipif(not IS_REAL_DATA_AVAILABLE, reason="local SPX parquet data unavailable")
def test_put_dataset_uses_put_payoff_and_vendor_put_iv_source() -> None:
    """Verify put dataset uses put payoff and vendor put IV source."""
    dataset = get_option_dataset(
        date_period=["2024-05-30", "2024-06-28"], label="test", cp_flag="P", data_root=DATA_ROOT
    )
    terminal_steps = dataset.episode_steps.loc[dataset.episode_steps["is_terminal"]]
    expected_payoff = np.maximum(terminal_steps["strike"] - terminal_steps["spot"], 0.0)
    assert dataset.num_episode == 3
    assert np.allclose(terminal_steps["mid_price"], expected_payoff)
    assert set(dataset.episode_steps.loc[~dataset.episode_steps["is_terminal"], "iv_source"]) == {
        "vendor_put"
    }


@pytest.mark.integration
@pytest.mark.skipif(not IS_REAL_DATA_AVAILABLE, reason="local SPX parquet data unavailable")
def test_dataset_sequence_returns_defensive_copy_and_saves(tmp_path: Path) -> None:
    """Verify dataset sequence returns defensive copy and saves."""
    dataset = _get_real_dataset()
    episode = dataset[0]
    episode.loc[0, "spot"] = -1.0
    assert dataset[0].iloc[0]["spot"] > 0
    assert dataset.get_manifest_row(0)["episode_id"] == dataset.get_episode_ids()[0]
    paths = dataset.save(tmp_path)
    assert all((path.is_file() for path in paths.values()))
    saved_steps = pd.read_parquet(paths["episode_steps"])
    saved_manifest = pd.read_parquet(paths["episode_manifest"])
    assert len(saved_steps) == 63
    assert len(saved_manifest) == 3
    with pytest.raises(FileExistsError):
        dataset.save(tmp_path)


@pytest.mark.integration
@pytest.mark.skipif(not IS_REAL_DATA_AVAILABLE, reason="local SPX parquet data unavailable")
def test_allow_empty_dataset_preserves_schema_and_can_be_saved(tmp_path: Path) -> None:
    """Verify allow empty dataset preserves schema and can be saved."""
    dataset = get_option_dataset(
        date_period=["2024-01-02", "2024-01-03"],
        label="test",
        data_root=DATA_ROOT,
        is_allow_empty=True,
    )
    assert dataset.num_episode == 0
    assert dataset.num_step == 0
    assert str(dataset.episode_steps["episode_id"].dtype) == "string"
    assert str(dataset.episode_steps["is_terminal"].dtype) == "boolean"
    paths = dataset.save(tmp_path)
    assert all((path.is_file() for path in paths.values()))


@pytest.mark.integration
@pytest.mark.skipif(not IS_REAL_DATA_AVAILABLE, reason="local SPX parquet data unavailable")
def test_min_open_interest_zero_does_not_reject_zero_oi_candidate() -> None:
    """Verify min open interest zero does not reject zero oi candidate."""
    config = get_option_dataset_config(
        date_period=["2024-05-30", "2024-06-28"],
        label="test",
        data_root=DATA_ROOT,
        min_open_interest=0,
    )
    builder = OptionDatasetBuilder(config)
    t0 = pd.Timestamp("2024-05-30")
    expiry = pd.Timestamp("2024-06-28")
    chain = builder._get_option_chain(t0, t0, expiry)
    initial_selected, _ = builder._get_selected_candidates(chain, t0, expiry)
    target_id = int(initial_selected.iloc[0]["optionid"])
    is_target = chain["optionid"].eq(target_id)
    chain.loc[is_target, "open_interest"] = 0.0
    chain.loc[is_target, "is_open_interest_missing"] = True
    selected, audit = builder._get_selected_candidates(chain, t0, expiry)
    assert target_id in set(selected["optionid"])
    row = audit.loc[audit["optionid"].eq(target_id)].iloc[0]
    assert bool(row["is_open_interest_valid"])
    assert bool(row["is_selected"])


@pytest.mark.integration
@pytest.mark.skipif(not IS_REAL_DATA_AVAILABLE, reason="local SPX parquet data unavailable")
def test_lifecycle_failure_drops_cohort_without_candidate_backfill(monkeypatch) -> None:
    """Verify lifecycle failure drops cohort without candidate backfill."""
    config = get_option_dataset_config(
        date_period=["2024-05-30", "2024-06-28"],
        label="test",
        data_root=DATA_ROOT,
        num_moneyness=3,
        is_allow_empty=True,
    )
    builder = OptionDatasetBuilder(config)
    original_get_episode = builder._get_episode

    def get_episode_with_failure(*, selected, chain_window, t0, expiry):
        """Return episode with failure."""
        if int(selected["selection_rank"]) == 1:
            raise EpisodeValidationError("injected lifecycle failure")
        return original_get_episode(
            selected=selected, chain_window=chain_window, t0=t0, expiry=expiry
        )

    monkeypatch.setattr(builder, "_get_episode", get_episode_with_failure)
    result = builder.get_result()
    assert result.episode_manifest.empty
    selected_audit = result.candidate_selection_audit.loc[
        result.candidate_selection_audit["is_selected"].astype(bool)
    ]
    assert len(selected_audit) == 3
    assert set(selected_audit["selection_rank"]) == {1, 2, 3}
    assert selected_audit["rejection_reason"].eq("cohort_lifecycle_failed").all()
    assert not result.candidate_selection_audit["is_cohort_retained"].any()
