"""Helpers for the paper option-hedging pipeline."""

from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd
from option_dataset import OptionDataset, get_option_dataset_config
from option_dataset.constants import CANDIDATE_AUDIT_COLUMNS, EPISODE_STEP_COLUMNS, MANIFEST_COLUMNS


def get_synthetic_dataset(label: str = "train", *, num_episode: int = 2) -> OptionDataset:
    """Return synthetic dataset."""
    if isinstance(num_episode, bool) or not 1 <= num_episode <= 2:
        raise ValueError("num_episode must be 1 or 2")
    label = str(label).lower()
    config = get_option_dataset_config(
        date_period=["2024-01-02", "2024-01-10"], label=label, num_interval=2
    )
    specifications = (
        {
            "episode_id": "synthetic_up",
            "optionid": 1,
            "dates": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-05"]),
            "spot": np.array([100.0, 110.0, 120.0]),
            "strike": 100.0,
            "option": np.array([10.0, 15.0, 20.0]),
        },
        {
            "episode_id": "synthetic_down",
            "optionid": 2,
            "dates": pd.to_datetime(["2024-01-08", "2024-01-09", "2024-01-10"]),
            "spot": np.array([200.0, 180.0, 160.0]),
            "strike": 200.0,
            "option": np.array([20.0, 10.0, 0.0]),
        },
    )
    step_frames: list[pd.DataFrame] = []
    manifest_rows: list[dict[str, Any]] = []
    for selection_rank, spec in enumerate(specifications[:num_episode], start=1):
        num_state = 3
        s0 = float(spec["spot"][0])
        strike = float(spec["strike"])
        option = np.asarray(spec["option"], dtype=np.float64)
        dates = pd.DatetimeIndex(spec["dates"])
        exdate = dates[-1]
        frame = pd.DataFrame(index=range(num_state), columns=EPISODE_STEP_COLUMNS)
        frame["episode_id"] = spec["episode_id"]
        frame["cohort_id"] = exdate
        frame["label"] = label
        frame["selection_rank"] = selection_rank
        frame["optionid"] = spec["optionid"]
        frame["symbol"] = f"SPXW TEST{selection_rank}"
        frame["symbol_root"] = "SPXW"
        frame["cp_flag"] = "C"
        frame["step"] = np.arange(num_state)
        frame["date"] = dates
        frame["exdate"] = exdate
        frame["dte"] = np.array([3, 2, 0])
        frame["tau"] = np.array([3 / 365, 2 / 365, 0.0])
        frame["is_terminal"] = np.array([False, False, True])
        frame["is_environment_ready"] = True
        frame["strike"] = strike
        frame["best_bid"] = option
        frame["best_offer"] = option
        frame["quoted_mid_price"] = option
        frame["mid_price"] = option
        frame["impl_volatility"] = np.array([0.2, 0.2, np.nan])
        frame["resolved_iv"] = np.array([0.2, 0.2, np.nan])
        frame["iv_source"] = ["vendor_call", "vendor_call", "terminal_no_action"]
        frame["spot"] = spec["spot"]
        frame["zero_rate"] = 0.03
        frame["dividend_rate"] = 0.0
        frame["funding_rate"] = 0.0
        frame["spot_norm"] = spec["spot"] / s0
        frame["strike_norm"] = strike / s0
        frame["option_mid_norm"] = option / s0
        frame["moneyness_initial"] = strike / s0
        frame["atm_distance"] = abs(strike / s0 - 1.0)
        frame["relative_spread"] = 0.0
        frame["open_interest"] = 100.0
        frame["volume"] = 10.0
        frame["is_open_interest_missing"] = False
        frame["is_volume_missing"] = False
        step_frames.append(frame)
        manifest_rows.append(
            {
                "episode_id": spec["episode_id"],
                "cohort_id": exdate,
                "label": label,
                "selection_rank": selection_rank,
                "optionid": spec["optionid"],
                "symbol": f"SPXW TEST{selection_rank}",
                "symbol_root": "SPXW",
                "cp_flag": "C",
                "t0": dates[0],
                "exdate": exdate,
                "num_interval": 2,
                "num_state": 3,
                "s0": s0,
                "c0": float(option[0]),
                "strike": strike,
                "moneyness_initial": strike / s0,
                "atm_distance": abs(strike / s0 - 1.0),
                "relative_spread_initial": 0.0,
                "open_interest_initial": 100.0,
                "volume_initial": 10.0,
                "num_iv_imputed": 0,
                "is_environment_ready": True,
            }
        )
    episode_steps = pd.concat(step_frames, ignore_index=True)
    episode_manifest = pd.DataFrame(manifest_rows).reindex(columns=MANIFEST_COLUMNS)
    candidate_audit = pd.DataFrame(columns=CANDIDATE_AUDIT_COLUMNS)
    return OptionDataset(
        label=label,
        config=config,
        episode_steps=episode_steps,
        episode_manifest=episode_manifest,
        candidate_selection_audit=candidate_audit,
        dataset_build_report={"source": "unit_test"},
    )


def get_constant_dataset(label: str = "train") -> OptionDataset:
    """Return constant dataset."""
    base = get_synthetic_dataset(label, num_episode=1)
    steps = base.episode_steps.copy(deep=True)
    steps["spot"] = 100.0
    steps["spot_norm"] = 1.0
    steps["strike"] = 120.0
    steps["strike_norm"] = 1.2
    for column in (
        "best_bid",
        "best_offer",
        "quoted_mid_price",
        "mid_price",
        "option_mid_norm",
        "zero_rate",
        "dividend_rate",
        "funding_rate",
    ):
        steps[column] = 0.0
    steps["moneyness_initial"] = 1.2
    steps["atm_distance"] = 0.2
    manifest = base.episode_manifest.copy(deep=True)
    manifest["s0"] = 100.0
    manifest["c0"] = 0.0
    manifest["strike"] = 120.0
    manifest["moneyness_initial"] = 1.2
    manifest["atm_distance"] = 0.2
    return OptionDataset(
        label=label,
        config=base.config,
        episode_steps=steps,
        episode_manifest=manifest,
        candidate_selection_audit=base.candidate_selection_audit.copy(deep=True),
        dataset_build_report={"source": "constant_unit_test"},
    )
