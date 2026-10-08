"""Constants for the paper option-hedging pipeline."""

from __future__ import annotations

VALID_LABELS = frozenset({"train", "valid", "test"})
VALID_CP_FLAGS = frozenset({"C", "P"})
VALID_SYMBOL_STARTS = frozenset({"SPXW", "SPX", "ALL"})
IV_SOURCE_VENDOR_CALL = "vendor_call"
IV_SOURCE_VENDOR_PUT = "vendor_put"
IV_SOURCE_CALL_PRICE = "call_mid_solved"
IV_SOURCE_PUT_PRICE = "put_mid_solved"
IV_SOURCE_SAME_STRIKE_PUT = "same_strike_put"
IV_SOURCE_SAME_STRIKE_CALL = "same_strike_call"
IV_SOURCE_SURFACE = "surface_interpolated"
IV_SOURCE_PAST = "past_forward_fill"
IV_SOURCE_TERMINAL = "terminal_no_action"
VENDOR_IV_SOURCES = frozenset({IV_SOURCE_VENDOR_CALL, IV_SOURCE_VENDOR_PUT})
EPISODE_ID_COLUMN = "episode_id"
EXOGENOUS_STATE_COLUMNS = (
    "spot_norm",
    "strike_norm",
    "tau",
    "resolved_iv",
    "zero_rate",
    "dividend_rate",
)
OPTION_SCAN_COLUMNS = (
    "date",
    "exdate",
    "cp_flag",
    "strike_price",
    "best_bid",
    "best_offer",
    "volume",
    "open_interest",
    "impl_volatility",
    "optionid",
    "symbol",
)
EPISODE_STEP_COLUMNS = (
    "episode_id",
    "cohort_id",
    "label",
    "selection_rank",
    "optionid",
    "symbol",
    "symbol_root",
    "cp_flag",
    "step",
    "date",
    "exdate",
    "dte",
    "tau",
    "is_terminal",
    "is_environment_ready",
    "strike",
    "best_bid",
    "best_offer",
    "quoted_mid_price",
    "mid_price",
    "impl_volatility",
    "resolved_iv",
    "iv_source",
    "spot",
    "zero_rate",
    "dividend_rate",
    "funding_rate",
    "spot_norm",
    "strike_norm",
    "option_mid_norm",
    "moneyness_initial",
    "atm_distance",
    "relative_spread",
    "open_interest",
    "volume",
    "is_open_interest_missing",
    "is_volume_missing",
)
MANIFEST_COLUMNS = (
    "episode_id",
    "cohort_id",
    "label",
    "selection_rank",
    "optionid",
    "symbol",
    "symbol_root",
    "cp_flag",
    "t0",
    "exdate",
    "num_interval",
    "num_state",
    "s0",
    "c0",
    "strike",
    "moneyness_initial",
    "atm_distance",
    "relative_spread_initial",
    "open_interest_initial",
    "volume_initial",
    "num_iv_imputed",
    "is_environment_ready",
)
CANDIDATE_AUDIT_COLUMNS = (
    "cohort_id",
    "label",
    "date",
    "exdate",
    "optionid",
    "symbol",
    "symbol_root",
    "cp_flag",
    "strike",
    "spot",
    "moneyness_initial",
    "atm_distance",
    "best_bid",
    "best_offer",
    "quoted_mid_price",
    "relative_spread",
    "open_interest",
    "volume",
    "is_open_interest_missing",
    "is_volume_missing",
    "resolved_iv",
    "iv_source",
    "is_quote_valid",
    "is_spread_valid",
    "is_open_interest_valid",
    "is_volume_valid",
    "is_moneyness_valid",
    "is_iv_valid",
    "is_optionid_unique",
    "is_candidate",
    "is_selected",
    "selection_rank",
    "is_lifecycle_valid",
    "lifecycle_error",
    "is_cohort_retained",
    "rejection_reason",
)
