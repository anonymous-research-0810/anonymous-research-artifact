"""Dataset for the paper option-hedging pipeline."""

from __future__ import annotations
import json
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, overload
import numpy as np
import pandas as pd
from .builder import OptionDatasetBuilder
from .config import OptionDatasetConfig, get_option_dataset_config
from .constants import (
    EPISODE_ID_COLUMN,
    EPISODE_STEP_COLUMNS,
    EXOGENOUS_STATE_COLUMNS,
    MANIFEST_COLUMNS,
)
from .exceptions import DatasetBuildError, DatasetConfigurationError


def _get_json_value(value: Any) -> Any:
    """Return JSON value."""
    if isinstance(value, dict):
        return {str(key): _get_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_get_json_value(item) for item in value]
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if not isinstance(value, (list, tuple, dict)):
        try:
            is_missing = pd.isna(value)
        except (TypeError, ValueError):
            is_missing = False
        if isinstance(is_missing, (bool, np.bool_)) and bool(is_missing):
            return None
    return value


class OptionDataset(Sequence[pd.DataFrame]):
    """Deterministic option episodes with manifests, quote audits, and split metadata."""

    def __init__(
        self,
        *,
        label: str,
        config: OptionDatasetConfig,
        episode_steps: pd.DataFrame,
        episode_manifest: pd.DataFrame,
        candidate_selection_audit: pd.DataFrame,
        dataset_build_report: dict[str, Any],
    ) -> None:
        """Initialize validated configuration and internal state."""
        if label != config.label:
            raise DatasetConfigurationError("label differs from config.label")
        missing_steps = set(EPISODE_STEP_COLUMNS) - set(episode_steps.columns)
        missing_manifest = set(MANIFEST_COLUMNS) - set(episode_manifest.columns)
        if missing_steps:
            raise DatasetBuildError(f"episode_steps is missing fields: {sorted(missing_steps)}")
        if missing_manifest:
            raise DatasetBuildError(
                f"episode_manifest is missing fields: {sorted(missing_manifest)}"
            )
        if episode_manifest[EPISODE_ID_COLUMN].duplicated().any():
            raise DatasetBuildError("episode_manifest episode_id values must be unique")
        if not episode_steps.empty:
            step_labels = set(episode_steps["label"].dropna().astype(str))
            if step_labels != {label}:
                raise DatasetBuildError("episode_steps contains a different split label")
        self.label = label
        self.config = config
        self.episode_steps = episode_steps.reset_index(drop=True)
        self.episode_manifest = episode_manifest.reset_index(drop=True)
        self.candidate_selection_audit = candidate_selection_audit.reset_index(drop=True)
        self.dataset_build_report = dataset_build_report
        self._episode_ids = tuple(self.episode_manifest[EPISODE_ID_COLUMN].astype(str).tolist())
        self._episode_positions = {
            episode_id: positions.to_numpy(dtype=np.int64)
            for episode_id, positions in self.episode_steps.groupby(
                EPISODE_ID_COLUMN, sort=False
            ).groups.items()
        }
        if set(self._episode_ids) != set(self._episode_positions):
            raise DatasetBuildError("manifest and episode_steps have different episode_id sets")

    @property
    def num_episode(self) -> int:
        """Return the number of episode."""
        return len(self._episode_ids)

    @property
    def num_step(self) -> int:
        """Return the number of step."""
        return len(self.episode_steps)

    @property
    def is_environment_ready(self) -> bool:
        """Return whether environment ready."""
        if self.episode_manifest.empty:
            return False
        return bool(self.episode_manifest["is_environment_ready"].all())

    def __len__(self) -> int:
        """Len."""
        return self.num_episode

    @overload
    def __getitem__(self, index: int) -> pd.DataFrame:
        """Getitem."""
        ...

    @overload
    def __getitem__(self, index: slice) -> list[pd.DataFrame]:
        """Getitem."""
        ...

    def __getitem__(self, index: int | slice) -> pd.DataFrame | list[pd.DataFrame]:
        """Getitem."""
        if isinstance(index, slice):
            indices = range(*index.indices(self.num_episode))
            return [self.get_episode(num_index) for num_index in indices]
        return self.get_episode(index)

    def __iter__(self) -> Iterator[pd.DataFrame]:
        """Iter."""
        for num_index in range(self.num_episode):
            yield self.get_episode(num_index)

    def get_episode_ids(self) -> tuple[str, ...]:
        """Return episode ids."""
        return self._episode_ids

    def get_episode(self, episode: int | str, *, is_copy: bool = True) -> pd.DataFrame:
        """Return episode."""
        if isinstance(episode, bool):
            raise TypeError("episode must not be a boolean")
        if isinstance(episode, (int, np.integer)):
            num_index = int(episode)
            if num_index < 0:
                num_index += self.num_episode
            if not 0 <= num_index < self.num_episode:
                raise IndexError("episode index is out of range")
            episode_id = self._episode_ids[num_index]
        else:
            episode_id = str(episode)
            if episode_id not in self._episode_positions:
                raise KeyError(f"Unknown episode_id: {episode_id}")
        positions = self._episode_positions[episode_id]
        frame = self.episode_steps.iloc[positions]
        return frame.copy(deep=True) if is_copy else frame

    def get_manifest_row(self, episode: int | str) -> pd.Series:
        """Return manifest row."""
        if isinstance(episode, (int, np.integer)) and (not isinstance(episode, bool)):
            num_index = int(episode)
            if num_index < 0:
                num_index += self.num_episode
            if not 0 <= num_index < self.num_episode:
                raise IndexError("episode index is out of range")
            return self.episode_manifest.iloc[num_index].copy(deep=True)
        episode_id = str(episode)
        rows = self.episode_manifest.loc[self.episode_manifest[EPISODE_ID_COLUMN].eq(episode_id)]
        if rows.empty:
            raise KeyError(f"Unknown episode_id: {episode_id}")
        return rows.iloc[0].copy(deep=True)

    def get_exogenous_state_array(
        self,
        episode: int | str,
        *,
        is_include_terminal: bool = False,
        dtype: np.dtype | type = np.float32,
    ) -> np.ndarray:
        """Return exogenous state array."""
        frame = self.get_episode(episode, is_copy=False)
        if not is_include_terminal:
            frame = frame.loc[~frame["is_terminal"]]
        values = frame.loc[:, EXOGENOUS_STATE_COLUMNS].to_numpy(dtype=dtype, copy=True)
        if not is_include_terminal and (not np.isfinite(values).all()):
            raise DatasetBuildError("Decision-time exogenous states contain nonfinite values")
        return values

    def save(self, output_dir: str | Path, *, is_overwrite: bool = False) -> dict[str, Path]:
        """Persist the object at the requested destination with explicit overwrite handling."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        paths = {
            "episode_steps": output_path / "episode_steps.parquet",
            "episode_manifest": output_path / "episode_manifest.parquet",
            "candidate_selection_audit": output_path / "candidate_selection_audit.parquet",
            "dataset_build_report": output_path / "dataset_build_report.json",
        }
        existing = [path for path in paths.values() if path.exists()]
        if existing and (not is_overwrite):
            raise FileExistsError(
                "The destination exists; set is_overwrite=True to overwrite it: "
                + ", ".join((str(path) for path in existing))
            )
        self.episode_steps.to_parquet(paths["episode_steps"], index=False)
        self.episode_manifest.to_parquet(paths["episode_manifest"], index=False)
        self.candidate_selection_audit.to_parquet(paths["candidate_selection_audit"], index=False)
        report = _get_json_value(self.dataset_build_report)
        paths["dataset_build_report"].write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
        )
        return paths


def get_option_dataset(
    date_period: Sequence[Any],
    *,
    label: str,
    cp_flag: str = "C",
    symbol_start: str = "SPXW",
    expiry_weekday: int | None = None,
    num_interval: int = 20,
    num_moneyness: int = 3,
    moneyness_range: Sequence[float] = (0.95, 1.05),
    max_relative_spread: float = 0.1,
    min_open_interest: float = 0.0,
    min_iv: float = 1e-06,
    max_iv: float = 5.0,
    iv_solver_tolerance: float = 1e-08,
    num_iv_solver_iterations: int = 100,
    num_iv_surface_points: int = 2,
    max_iv_extrapolation_log_moneyness: float = 0.1,
    num_iv_forward_fill_steps: int = 1,
    is_use_option_price_for_iv: bool = True,
    is_use_same_strike_opposite_iv: bool = True,
    is_use_surface_iv: bool = True,
    is_use_past_iv: bool = True,
    is_require_balanced_cohort: bool = True,
    is_allow_empty: bool = False,
    is_require_environment_ready: bool = True,
    num_scan_batch_size: int = 131072,
    data_root: str | Path = "data",
) -> OptionDataset:
    """Build deterministic complete episodes within the requested split, retaining balanced cohorts and quote-selection audits."""
    config = get_option_dataset_config(
        date_period=date_period,
        label=label,
        data_root=data_root,
        cp_flag=cp_flag,
        symbol_start=symbol_start,
        expiry_weekday=expiry_weekday,
        num_interval=num_interval,
        num_moneyness=num_moneyness,
        moneyness_range=moneyness_range,
        max_relative_spread=max_relative_spread,
        min_open_interest=min_open_interest,
        min_iv=min_iv,
        max_iv=max_iv,
        iv_solver_tolerance=iv_solver_tolerance,
        num_iv_solver_iterations=num_iv_solver_iterations,
        num_iv_surface_points=num_iv_surface_points,
        max_iv_extrapolation_log_moneyness=max_iv_extrapolation_log_moneyness,
        num_iv_forward_fill_steps=num_iv_forward_fill_steps,
        is_use_option_price_for_iv=is_use_option_price_for_iv,
        is_use_same_strike_opposite_iv=is_use_same_strike_opposite_iv,
        is_use_surface_iv=is_use_surface_iv,
        is_use_past_iv=is_use_past_iv,
        is_require_balanced_cohort=is_require_balanced_cohort,
        is_allow_empty=is_allow_empty,
        is_require_environment_ready=is_require_environment_ready,
        num_scan_batch_size=num_scan_batch_size,
    )
    result = OptionDatasetBuilder(config).get_result()
    return OptionDataset(
        label=config.label,
        config=config,
        episode_steps=result.episode_steps,
        episode_manifest=result.episode_manifest,
        candidate_selection_audit=result.candidate_selection_audit,
        dataset_build_report=result.dataset_build_report,
    )
