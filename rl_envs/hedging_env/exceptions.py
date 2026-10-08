"""Exceptions for the paper option-hedging pipeline."""


class HedgingEnvError(Exception):
    """Hedging env error."""


class EnvironmentConfigurationError(HedgingEnvError, ValueError):
    """Environment configuration error."""


class EnvironmentStateError(HedgingEnvError, RuntimeError):
    """Environment state error."""


class RewardReconciliationError(HedgingEnvError, ArithmeticError):
    """Reward reconciliation error."""
