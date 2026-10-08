"""Market simulation interfaces."""

from .calibration import (
    SabrCalibrationResult,
    calibrate_segmented_sabr,
    get_global_calibration_baseline,
    prepare_sabr_cross_sections,
)
from .config import SimulationConfig
from .exceptions import SabrCalibrationError, SimulationQualityError, SimulationError
from .generation import (
    SimulationGenerationResult,
    generate_segment_simulation,
    generate_segmented_simulations,
)
from .pipeline import (
    SIMULATION_SCHEMA_VERSION,
    load_segmented_simulated_datasets,
    load_simulated_segment_dataset,
    load_simulation_result,
    resolve_simulation_result_directory,
    run_segmented_sabr_simulation,
)
from .quality import (
    compare_real_and_simulated,
    compare_segmented_and_global_simulations,
    evaluate_segmented_simulations,
    get_dataset_quality_features,
    run_environment_rollout_checks,
)
from .sabr import get_bsm_option_prices, get_lognormal_sabr_iv, simulate_lognormal_sabr_path

__all__ = [
    "SIMULATION_SCHEMA_VERSION",
    "SimulationError",
    "SabrCalibrationError",
    "SabrCalibrationResult",
    "SimulationConfig",
    "SimulationGenerationResult",
    "SimulationQualityError",
    "calibrate_segmented_sabr",
    "compare_real_and_simulated",
    "compare_segmented_and_global_simulations",
    "evaluate_segmented_simulations",
    "generate_segment_simulation",
    "generate_segmented_simulations",
    "get_bsm_option_prices",
    "get_dataset_quality_features",
    "get_global_calibration_baseline",
    "get_lognormal_sabr_iv",
    "load_segmented_simulated_datasets",
    "load_simulated_segment_dataset",
    "load_simulation_result",
    "prepare_sabr_cross_sections",
    "resolve_simulation_result_directory",
    "run_environment_rollout_checks",
    "run_segmented_sabr_simulation",
    "simulate_lognormal_sabr_path",
]
