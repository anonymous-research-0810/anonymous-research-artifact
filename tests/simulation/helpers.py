"""Helpers for the paper option-hedging pipeline."""

from __future__ import annotations
import hashlib
from typing import Any
import numpy as np
import pandas as pd
from option_dataset import OptionDataset, get_option_dataset_config
from option_dataset.constants import CANDIDATE_AUDIT_COLUMNS, EPISODE_STEP_COLUMNS, MANIFEST_COLUMNS
from market_simulation.sabr import get_bsm_option_prices, get_lognormal_sabr_iv


def get_simulation_test_dataset() -> tuple[OptionDataset, list[tuple[str, str]]]:
    """Return simulation test dataset."""
    num_interval = 3
    num_moneyness = 5
    dates = pd.bdate_range("2020-01-02", periods=16)
    cohort_date_groups = [dates[index : index + 4] for index in range(0, 16, 4)]
    seg_periods = [
        (dates[0].strftime("%Y-%m-%d"), dates[7].strftime("%Y-%m-%d")),
        (dates[8].strftime("%Y-%m-%d"), dates[-1].strftime("%Y-%m-%d")),
    ]
    config = get_option_dataset_config(
        date_period=(dates[0], dates[-1]),
        label="train",
        num_interval=num_interval,
        num_moneyness=num_moneyness,
        moneyness_range=(0.85, 1.15),
        is_require_balanced_cohort=True,
    )
    step_frames: list[pd.DataFrame] = []
    manifest_rows: list[dict[str, Any]] = []
    optionid = 1
    moneyness_values = np.asarray([0.9, 0.95, 1.0, 1.05, 1.1])
    for cohort_index, cohort_dates in enumerate(cohort_date_groups):
        segment_id = 1 if cohort_index < 2 else 2
        rho = -0.45 if segment_id == 1 else -0.2
        nu = 0.6 if segment_id == 1 else 0.9
        s0 = 100.0 + 4.0 * cohort_index
        spot_returns = np.asarray(
            [0.006, -0.004, 0.008] if segment_id == 1 else [-0.012, 0.018, -0.009]
        )
        spots = np.r_[s0, s0 * np.exp(np.cumsum(spot_returns))]
        exdate = cohort_dates[-1]
        dte = np.asarray([(exdate - date).days for date in cohort_dates])
        tau = dte / 365.0
        zero_rate = np.full(4, 0.02)
        dividend_rate = np.full(4, 0.01)
        alpha = 0.18 + 0.01 * cohort_index + 0.002 * np.arange(4)
        for selection_rank, moneyness in enumerate(moneyness_values, start=1):
            strike = float(s0 * moneyness)
            forward = spots[:-1] * np.exp((zero_rate[:-1] - dividend_rate[:-1]) * tau[:-1])
            iv = get_lognormal_sabr_iv(forward, strike, tau[:-1], alpha[:-1], rho, nu)
            prices = get_bsm_option_prices(
                spot=spots[:-1],
                strike=strike,
                tau=tau[:-1],
                volatility=iv,
                zero_rate=zero_rate[:-1],
                dividend_rate=dividend_rate[:-1],
                cp_flag="C",
            )
            terminal_payoff = max(float(spots[-1]) - strike, 0.0)
            all_prices = np.r_[prices, terminal_payoff]
            episode_id = f"real_{exdate:%Y%m%d}_rank{selection_rank:02d}_{optionid}"
            frame = pd.DataFrame(index=range(4), columns=EPISODE_STEP_COLUMNS)
            frame["episode_id"] = episode_id
            frame["cohort_id"] = exdate
            frame["label"] = "train"
            frame["selection_rank"] = selection_rank
            frame["optionid"] = optionid
            frame["symbol"] = f"SPXW TEST{optionid:05d}"
            frame["symbol_root"] = "SPXW"
            frame["cp_flag"] = "C"
            frame["step"] = np.arange(4)
            frame["date"] = cohort_dates
            frame["exdate"] = exdate
            frame["dte"] = dte
            frame["tau"] = tau
            frame["is_terminal"] = np.arange(4) == num_interval
            frame["is_environment_ready"] = True
            frame["strike"] = strike
            frame["best_bid"] = all_prices * 0.99
            frame["best_offer"] = all_prices * 1.01
            frame["quoted_mid_price"] = all_prices
            frame["mid_price"] = all_prices
            frame["impl_volatility"] = np.r_[iv, np.nan]
            frame["resolved_iv"] = np.r_[iv, np.nan]
            frame["iv_source"] = ["vendor_call", "vendor_call", "vendor_call", "terminal_no_action"]
            frame["spot"] = spots
            frame["zero_rate"] = zero_rate
            frame["dividend_rate"] = dividend_rate
            frame["funding_rate"] = 0.015
            frame["spot_norm"] = spots / s0
            frame["strike_norm"] = strike / s0
            frame["option_mid_norm"] = all_prices / s0
            frame["moneyness_initial"] = moneyness
            frame["atm_distance"] = abs(moneyness - 1.0)
            frame["relative_spread"] = 0.02
            frame["open_interest"] = 100.0
            frame["volume"] = 20.0
            frame["is_open_interest_missing"] = False
            frame["is_volume_missing"] = False
            step_frames.append(frame)
            manifest_rows.append(
                {
                    "episode_id": episode_id,
                    "cohort_id": exdate,
                    "label": "train",
                    "selection_rank": selection_rank,
                    "optionid": optionid,
                    "symbol": f"SPXW TEST{optionid:05d}",
                    "symbol_root": "SPXW",
                    "cp_flag": "C",
                    "t0": cohort_dates[0],
                    "exdate": exdate,
                    "num_interval": num_interval,
                    "num_state": num_interval + 1,
                    "s0": s0,
                    "c0": float(all_prices[0]),
                    "strike": strike,
                    "moneyness_initial": moneyness,
                    "atm_distance": abs(moneyness - 1.0),
                    "relative_spread_initial": 0.02,
                    "open_interest_initial": 100.0,
                    "volume_initial": 20.0,
                    "num_iv_imputed": 0,
                    "is_environment_ready": True,
                }
            )
            optionid += 1
    dataset = OptionDataset(
        label="train",
        config=config,
        episode_steps=pd.concat(step_frames, ignore_index=True),
        episode_manifest=pd.DataFrame(manifest_rows).reindex(columns=MANIFEST_COLUMNS),
        candidate_selection_audit=pd.DataFrame(columns=CANDIDATE_AUDIT_COLUMNS),
        dataset_build_report={"source": "simulation_unit_test"},
    )
    return (dataset, seg_periods)


def get_segmentation_result(
    dataset: OptionDataset, seg_periods: list[tuple[str, str]]
) -> dict[str, Any]:
    """Return segmentation result."""
    encoded = "\n".join(dataset.get_episode_ids()).encode("utf-8")
    return {
        "schema_version": 1,
        "status": "completed",
        "experiment_id": "seg_20200101T000000Z",
        "num_segment": len(seg_periods),
        "seg_periods": [list(period) for period in seg_periods],
        "dataset": {
            "config": dataset.config.get_dict(),
            "num_episode": dataset.num_episode,
            "episode_ids_sha256": hashlib.sha256(encoded).hexdigest(),
        },
    }
