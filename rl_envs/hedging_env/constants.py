"""Constants for the paper option-hedging pipeline."""

from __future__ import annotations

EXOGENOUS_FEATURE_NAMES = (
    "spot_norm",
    "strike_norm",
    "tau",
    "resolved_iv",
    "zero_rate",
    "dividend_rate",
)
STATE_FEATURE_NAMES = (
    "spot_norm",
    "strike_norm",
    "holding_norm",
    "tau",
    "resolved_iv",
    "zero_rate",
    "dividend_rate",
)
REWARD_FORMULATION_SHAPED_ACCOUNTING = "shaped_accounting"
REWARD_FORMULATION_CASH_FLOW = "cash_flow"
VALID_REWARD_FORMULATIONS = frozenset(
    {REWARD_FORMULATION_SHAPED_ACCOUNTING, REWARD_FORMULATION_CASH_FLOW}
)
DEFAULT_HEDGE_COST_RATE = 0.001
DEFAULT_CONTRACT_MULTIPLIER = 100.0
DEFAULT_RECONCILIATION_TOLERANCE = 1e-08
DEFAULT_RISK_AVERSION_LAMBDA = 1.5
DEFAULT_REWARD_RISK_AVERSION_XI = 1.5
DEFAULT_TRAINING_REWARD_SCALES = {
    REWARD_FORMULATION_SHAPED_ACCOUNTING: 100.0,
    REWARD_FORMULATION_CASH_FLOW: 1.0,
}
DEFAULT_MIN_STATE_STD = 1e-08
