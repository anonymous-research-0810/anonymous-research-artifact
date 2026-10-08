"""Exceptions for the paper option-hedging pipeline."""


class SimulationError(RuntimeError):
    """Simulation error."""


class SabrCalibrationError(SimulationError):
    """Sabr calibration error."""


class SimulationQualityError(SimulationError):
    """Simulation quality error."""
