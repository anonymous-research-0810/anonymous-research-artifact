"""Helpers for the paper option-hedging pipeline."""

from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd
from option_dataset import OptionDataset, get_option_dataset_config
from option_dataset.constants import CANDIDATE_AUDIT_COLUMNS, EPISODE_STEP_COLUMNS, MANIFEST_COLUMNS


def get_segmentation_test_dataset(*, num_interval: int = 3, num_episode: int = 6) -> OptionDataset:
    """Return segmentation test dataset."""
    if num_interval <= 0 or num_episode <= 1:
        raise ValueError("num_interval must be positive and num_episode must be at least 2")
    num_date = num_episode * (num_interval + 1)
    dates = pd.bdate_range("2020-01-02", periods=num_date)
    num_return = num_date - 1
    log_returns = np.empty(num_return, dtype=np.float64)
    split = num_return // 2
    log_returns[:split] = 0.001
    high_pattern = np.resize(np.array([0.035, -0.025]), num_return - split)
    log_returns[split:] = high_pattern
    spots = np.empty(num_date, dtype=np.float64)
    spots[0] = 100.0
    spots[1:] = spots[0] * np.exp(np.cumsum(log_returns))
    config = get_option_dataset_config(
        date_period=(dates[0], dates[-1]),
        label="train",
        num_interval=num_interval,
        num_moneyness=1,
        is_require_balanced_cohort=False,
    )
    step_frames: list[pd.DataFrame] = []
    manifest_rows: list[dict[str, Any]] = []
    for episode_index in range(num_episode):
        start = episode_index * (num_interval + 1)
        stop = start + num_interval + 1
        episode_dates = dates[start:stop]
        episode_spots = spots[start:stop]
        episode_id = f"episode_{episode_index:02d}"
        cohort_id = f"cohort_{episode_index:02d}"
        strike = float(episode_spots[0])
        payoff = np.maximum(episode_spots - strike, 0.0)
        option_values = np.maximum(
            payoff, 0.08 * episode_spots[0] * np.linspace(1.0, 0.0, num_interval + 1)
        )
        option_values[-1] = payoff[-1]
        frame = pd.DataFrame(index=range(num_interval + 1), columns=EPISODE_STEP_COLUMNS)
        frame["episode_id"] = episode_id
        frame["cohort_id"] = cohort_id
        frame["label"] = "train"
        frame["selection_rank"] = 1
        frame["optionid"] = episode_index + 1
        frame["symbol"] = f"SPXW TEST{episode_index:02d}"
        frame["symbol_root"] = "SPXW"
        frame["cp_flag"] = "C"
        frame["step"] = np.arange(num_interval + 1)
        frame["date"] = episode_dates
        frame["exdate"] = episode_dates[-1]
        frame["dte"] = np.arange(num_interval, -1, -1)
        frame["tau"] = np.arange(num_interval, -1, -1) / 365.0
        frame["is_terminal"] = np.arange(num_interval + 1) == num_interval
        frame["is_environment_ready"] = True
        frame["strike"] = strike
        frame["best_bid"] = option_values
        frame["best_offer"] = option_values
        frame["quoted_mid_price"] = option_values
        frame["mid_price"] = option_values
        frame["impl_volatility"] = np.r_[np.full(num_interval, 0.2), np.nan]
        frame["resolved_iv"] = np.r_[np.full(num_interval, 0.2), np.nan]
        frame["iv_source"] = [*["vendor_call"] * num_interval, "terminal_no_action"]
        frame["spot"] = episode_spots
        frame["zero_rate"] = 0.02
        frame["dividend_rate"] = 0.01
        frame["funding_rate"] = 0.02
        frame["spot_norm"] = episode_spots / episode_spots[0]
        frame["strike_norm"] = strike / episode_spots[0]
        frame["option_mid_norm"] = option_values / episode_spots[0]
        frame["moneyness_initial"] = strike / episode_spots[0]
        frame["atm_distance"] = 0.0
        frame["relative_spread"] = 0.01
        frame["open_interest"] = 100.0
        frame["volume"] = 10.0
        frame["is_open_interest_missing"] = False
        frame["is_volume_missing"] = False
        step_frames.append(frame)
        manifest_rows.append(
            {
                "episode_id": episode_id,
                "cohort_id": cohort_id,
                "label": "train",
                "selection_rank": 1,
                "optionid": episode_index + 1,
                "symbol": f"SPXW TEST{episode_index:02d}",
                "symbol_root": "SPXW",
                "cp_flag": "C",
                "t0": episode_dates[0],
                "exdate": episode_dates[-1],
                "num_interval": num_interval,
                "num_state": num_interval + 1,
                "s0": float(episode_spots[0]),
                "c0": float(option_values[0]),
                "strike": strike,
                "moneyness_initial": 1.0,
                "atm_distance": 0.0,
                "relative_spread_initial": 0.01,
                "open_interest_initial": 100.0,
                "volume_initial": 10.0,
                "num_iv_imputed": 0,
                "is_environment_ready": True,
            }
        )
    return OptionDataset(
        label="train",
        config=config,
        episode_steps=pd.concat(step_frames, ignore_index=True),
        episode_manifest=pd.DataFrame(manifest_rows).reindex(columns=MANIFEST_COLUMNS),
        candidate_selection_audit=pd.DataFrame(columns=CANDIDATE_AUDIT_COLUMNS),
        dataset_build_report={"source": "segmentation_unit_test"},
    )
