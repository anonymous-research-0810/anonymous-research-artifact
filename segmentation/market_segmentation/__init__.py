"""Market segmentation interfaces."""

from .config import SegmentationConfig, VALID_SEGMENTATION_ALGORITHMS
from .dataset import normalize_seg_periods, segment_option_dataset
from .features import SegmentationFeatures, get_daily_spot_from_market, get_segmentation_features
from .pelt import PeltResult, run_pelt_segmentation
from .pipeline import (
    SEGMENTATION_SCHEMA_VERSION,
    get_seg_periods_from_result,
    load_seg_result,
    resolve_seg_result_path,
    run_market_segmentation,
)

__all__ = [
    "PeltResult",
    "SEGMENTATION_SCHEMA_VERSION",
    "SegmentationConfig",
    "SegmentationFeatures",
    "VALID_SEGMENTATION_ALGORITHMS",
    "get_seg_periods_from_result",
    "get_daily_spot_from_market",
    "get_segmentation_features",
    "load_seg_result",
    "normalize_seg_periods",
    "resolve_seg_result_path",
    "run_pelt_segmentation",
    "run_market_segmentation",
    "segment_option_dataset",
]
