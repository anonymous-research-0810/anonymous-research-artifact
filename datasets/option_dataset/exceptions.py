"""Exceptions for the paper option-hedging pipeline."""


class OptionDatasetError(Exception):
    """Option dataset error."""


class DatasetConfigurationError(OptionDatasetError, ValueError):
    """Dataset configuration error."""


class DatasetBuildError(OptionDatasetError, RuntimeError):
    """Dataset build error."""


class EpisodeValidationError(OptionDatasetError, ValueError):
    """Episode validation error."""
