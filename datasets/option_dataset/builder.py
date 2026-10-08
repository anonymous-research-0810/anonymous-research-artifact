"""Builder for the paper option-hedging pipeline."""

from __future__ import annotations
import hashlib
from collections import Counter
from dataclasses import dataclass
from typing import Any
import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.dataset as ds
from market_data import discover_option_files, option_dataset
from market_data.market import MarketData, load_market_data
from .config import OptionDatasetConfig
from .constants import (
    CANDIDATE_AUDIT_COLUMNS,
    EPISODE_STEP_COLUMNS,
    IV_SOURCE_CALL_PRICE,
    IV_SOURCE_PAST,
    IV_SOURCE_PUT_PRICE,
    IV_SOURCE_SAME_STRIKE_CALL,
    IV_SOURCE_SAME_STRIKE_PUT,
    IV_SOURCE_SURFACE,
    IV_SOURCE_TERMINAL,
    IV_SOURCE_VENDOR_CALL,
    IV_SOURCE_VENDOR_PUT,
    MANIFEST_COLUMNS,
    OPTION_SCAN_COLUMNS,
    VENDOR_IV_SOURCES,
)
from .exceptions import DatasetBuildError, EpisodeValidationError
from .pricing import (
    get_implied_volatility,
    get_log_forward_moneyness,
    get_surface_implied_volatility,
)

_NANOSECONDS_PER_DAY = 86400000000000


@dataclass(frozen=True)
class OptionDatasetBuildResult:
    """Tables and audit metadata produced by the episode builder."""

    episode_steps: pd.DataFrame
    episode_manifest: pd.DataFrame
    candidate_selection_audit: pd.DataFrame
    dataset_build_report: dict[str, Any]


def _get_empty_frame(columns: tuple[str, ...]) -> pd.DataFrame:
    """Return empty frame."""
    date_columns = {"cohort_id", "date", "exdate", "t0"}
    string_columns = {
        "episode_id",
        "label",
        "symbol",
        "symbol_root",
        "cp_flag",
        "iv_source",
        "lifecycle_error",
        "rejection_reason",
    }
    boolean_columns = {column for column in columns if column.startswith("is_")}
    integer_columns = {
        "selection_rank",
        "optionid",
        "step",
        "dte",
        "num_interval",
        "num_state",
        "num_iv_imputed",
    }
    data: dict[str, pd.Series] = {}
    for column in columns:
        if column in date_columns:
            dtype = "datetime64[ns]"
        elif column in string_columns:
            dtype = "string"
        elif column in boolean_columns:
            dtype = "boolean"
        elif column in integer_columns:
            dtype = "Int64"
        else:
            dtype = "float64"
        data[column] = pd.Series(dtype=dtype)
    return pd.DataFrame(data)


def _get_symbol_root(symbol: pd.Series) -> pd.Series:
    """Return symbol root."""
    return symbol.astype("string").str.extract("^([A-Za-z0-9]+)(?:\\s|$)", expand=False).str.upper()


def _get_rejection_reason(row: pd.Series) -> str:
    """Return rejection reason."""
    reasons: list[str] = []
    checks = (
        ("is_quote_valid", "invalid_quote"),
        ("is_spread_valid", "relative_spread"),
        ("is_open_interest_valid", "open_interest"),
        ("is_volume_valid", "volume"),
        ("is_moneyness_valid", "moneyness"),
        ("is_iv_valid", "iv"),
        ("is_optionid_unique", "duplicate_optionid"),
    )
    for column, reason in checks:
        if column in row and (not bool(row[column])):
            reasons.append(reason)
    return "|".join(reasons)


class OptionDatasetBuilder:
    """Build complete call lifecycles with balanced strike cohorts and causal IV recovery."""

    def __init__(self, config: OptionDatasetConfig) -> None:
        """Initialize validated configuration and internal state."""
        self.config = config
        self._market_data: MarketData = load_market_data(config.data_root)
        self._option_dataset = option_dataset(config.data_root, years=config.years)
        self._calendar = pd.DatetimeIndex(pd.to_datetime(self._market_data.dates), name="date")
        self._date_to_market_position = {
            int(date_value): num_position
            for num_position, date_value in enumerate(self._market_data.dates)
        }
        self._reset_build_counters()

    def _reset_build_counters(self) -> None:
        """Reset build counters."""
        self._num_expiry_calendar = 0
        self._num_expiry_without_complete_window = 0
        self._num_expiry_without_quotes = 0
        self._num_expiry_with_quotes = 0
        self._num_cohort_insufficient_candidates = 0
        self._num_cohort_lifecycle_failed = 0
        self._num_cohort_retained = 0

    def get_result(self) -> OptionDatasetBuildResult:
        """Return result."""
        self._reset_build_counters()
        self._validate_calendar_coverage()
        episode_frames: list[pd.DataFrame] = []
        manifest_rows: list[dict[str, Any]] = []
        audit_frames: list[pd.DataFrame] = []
        for expiry in self._get_eligible_expiries():
            t0 = self._get_t0(expiry)
            if t0 is None or t0 < self.config.date_start:
                self._num_expiry_without_complete_window += 1
                continue
            chain_window = self._get_option_chain(t0, expiry, expiry)
            t0_chain = chain_window.loc[chain_window["date"].eq(t0)].copy()
            if t0_chain.empty:
                self._num_expiry_without_quotes += 1
                continue
            self._num_expiry_with_quotes += 1
            candidates, audit = self._get_selected_candidates(t0_chain, t0, expiry)
            if candidates.empty:
                self._num_cohort_insufficient_candidates += 1
                audit_frames.append(audit)
                continue
            cohort_episodes: list[pd.DataFrame] = []
            cohort_manifests: list[dict[str, Any]] = []
            lifecycle_errors: dict[int, str] = {}
            for selected in candidates.itertuples(index=False):
                optionid = int(selected.optionid)
                try:
                    episode = self._get_episode(
                        selected=pd.Series(selected._asdict()),
                        chain_window=chain_window,
                        t0=t0,
                        expiry=expiry,
                    )
                    cohort_episodes.append(episode)
                    cohort_manifests.append(self._get_manifest_row(episode))
                except EpisodeValidationError as exc:
                    lifecycle_errors[optionid] = str(exc)
            is_cohort_retained = not lifecycle_errors and (
                len(cohort_episodes) == self.config.num_moneyness
                or not self.config.is_require_balanced_cohort
            )
            selected_mask = audit["is_selected"].astype(bool)
            audit["is_lifecycle_valid"] = pd.Series(pd.NA, index=audit.index, dtype="boolean")
            audit.loc[selected_mask, "is_lifecycle_valid"] = ~audit.loc[
                selected_mask, "optionid"
            ].isin(lifecycle_errors)
            audit["lifecycle_error"] = audit["optionid"].map(lifecycle_errors).fillna("")
            audit["is_cohort_retained"] = bool(is_cohort_retained)
            if not is_cohort_retained:
                self._num_cohort_lifecycle_failed += 1
                audit.loc[selected_mask, "rejection_reason"] = "cohort_lifecycle_failed"
                audit_frames.append(audit)
                continue
            self._num_cohort_retained += 1
            audit.loc[selected_mask, "rejection_reason"] = ""
            audit_frames.append(audit)
            episode_frames.extend(cohort_episodes)
            manifest_rows.extend(cohort_manifests)
        episode_steps = self._get_episode_steps(episode_frames)
        episode_manifest = self._get_episode_manifest(manifest_rows)
        candidate_audit = self._get_candidate_audit(audit_frames)
        if episode_manifest.empty and (not self.config.is_allow_empty):
            raise DatasetBuildError(
                "No complete episodes remain; check the dates, product filters, and selection audit"
            )
        report = self._get_build_report(
            episode_steps=episode_steps,
            episode_manifest=episode_manifest,
            candidate_audit=candidate_audit,
        )
        return OptionDatasetBuildResult(
            episode_steps=episode_steps,
            episode_manifest=episode_manifest,
            candidate_selection_audit=candidate_audit,
            dataset_build_report=report,
        )

    def _validate_calendar_coverage(self) -> None:
        """Validate calendar coverage."""
        if self._calendar.empty:
            raise DatasetBuildError("The market trading calendar is empty")
        if self._calendar.has_duplicates or not self._calendar.is_monotonic_increasing:
            raise DatasetBuildError("The trading calendar must be strictly increasing and unique")
        if self.config.date_start < self._calendar[0] or self.config.date_end > self._calendar[-1]:
            raise DatasetBuildError(
                f"date_period exceeds market data coverage: [{self._calendar[0].date()}, {self._calendar[-1].date()}]"
            )

    def _get_calendar_expiries(self) -> pd.DatetimeIndex:
        """Return calendar expiries."""
        is_in_period = (self._calendar >= self.config.date_start) & (
            self._calendar <= self.config.date_end
        )
        expiries = self._calendar[is_in_period]
        if self.config.expiry_weekday is not None:
            expiries = expiries[expiries.weekday == self.config.expiry_weekday]
        return expiries

    def _get_eligible_expiries(self) -> pd.DatetimeIndex:
        """Return eligible expiries."""
        calendar_expiries = self._get_calendar_expiries()
        self._num_expiry_calendar = len(calendar_expiries)
        if calendar_expiries.empty:
            return calendar_expiries
        expression = (ds.field("exdate") >= self.config.date_start.to_pydatetime()) & (
            ds.field("exdate") <= self.config.date_end.to_pydatetime()
        )
        scanner = self._option_dataset.scanner(
            columns=["exdate", "symbol"],
            filter=expression,
            batch_size=self.config.num_scan_batch_size,
        )
        actual_expiries: set[pd.Timestamp] = set()
        for batch in scanner.to_batches():
            symbols = pc.ascii_upper(batch.column("symbol"))
            if self.config.symbol_start == "ALL":
                is_symbol = pc.or_(
                    pc.starts_with(symbols, pattern="SPX "),
                    pc.starts_with(symbols, pattern="SPXW "),
                )
            else:
                is_symbol = pc.starts_with(symbols, pattern=f"{self.config.symbol_start} ")
            batch_expiries = pc.filter(batch.column("exdate"), is_symbol)
            unique_expiries = pc.unique(batch_expiries).to_pylist()
            actual_expiries.update(
                (pd.Timestamp(value).normalize() for value in unique_expiries if value is not None)
            )
        eligible_expiries = calendar_expiries[calendar_expiries.isin(actual_expiries)]
        self._num_expiry_without_quotes = len(calendar_expiries) - len(eligible_expiries)
        return eligible_expiries

    def _get_t0(self, expiry: pd.Timestamp) -> pd.Timestamp | None:
        """Return t0."""
        num_position = int(self._calendar.get_indexer([expiry])[0])
        if num_position < self.config.num_interval:
            return None
        return pd.Timestamp(self._calendar[num_position - self.config.num_interval])

    def _get_option_chain(
        self, date_start: pd.Timestamp, date_end: pd.Timestamp, expiry: pd.Timestamp
    ) -> pd.DataFrame:
        """Return option chain."""
        expression = (
            (ds.field("date") >= date_start.to_pydatetime())
            & (ds.field("date") <= date_end.to_pydatetime())
            & (ds.field("exdate") == expiry.to_pydatetime())
        )
        scanner = self._option_dataset.scanner(
            columns=list(OPTION_SCAN_COLUMNS),
            filter=expression,
            batch_size=self.config.num_scan_batch_size,
        )
        table = scanner.to_table()
        if table.num_rows == 0:
            return self._get_prepared_option_frame(table.to_pandas())
        return self._get_prepared_option_frame(table.to_pandas())

    def _get_prepared_option_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return prepared option frame."""
        result = frame.copy()
        for column in OPTION_SCAN_COLUMNS:
            if column not in result:
                result[column] = pd.NA
        result["date"] = pd.to_datetime(result["date"], errors="coerce").dt.normalize()
        result["exdate"] = pd.to_datetime(result["exdate"], errors="coerce").dt.normalize()
        result["symbol"] = result["symbol"].astype("string")
        result["symbol_root"] = _get_symbol_root(result["symbol"])
        if self.config.symbol_start == "ALL":
            is_symbol = result["symbol_root"].isin(["SPX", "SPXW"])
        else:
            is_symbol = result["symbol_root"].eq(self.config.symbol_start)
        result = result.loc[is_symbol.fillna(False)].copy()
        numeric_columns = (
            "strike_price",
            "best_bid",
            "best_offer",
            "volume",
            "open_interest",
            "impl_volatility",
            "optionid",
        )
        for column in numeric_columns:
            result[column] = pd.to_numeric(result[column], errors="coerce").astype(float)
        if not result.empty:
            optionids = result["optionid"].to_numpy(dtype=float)
            is_integer_optionid = np.isfinite(optionids) & (optionids == np.floor(optionids))
            result = result.loc[is_integer_optionid].copy()
            result["optionid"] = result["optionid"].astype(np.int64)
        else:
            result["optionid"] = result["optionid"].astype(np.int64)
        result["cp_flag"] = result["cp_flag"].astype("string").str.upper()
        result["strike"] = result["strike_price"] / 1000.0
        result["quoted_mid_price"] = (result["best_bid"] + result["best_offer"]) / 2.0
        spread = result["best_offer"] - result["best_bid"]
        positive_mid = result["quoted_mid_price"].where(result["quoted_mid_price"].gt(0))
        result["relative_spread"] = spread / positive_mid
        result["is_open_interest_missing"] = result["open_interest"].isna()
        result["is_volume_missing"] = result["volume"].isna()
        result["open_interest"] = result["open_interest"].fillna(0.0)
        result["volume"] = result["volume"].fillna(0.0)
        result = self._get_market_enriched_frame(result)
        return result.reset_index(drop=True)

    def _get_market_enriched_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return market enriched frame."""
        result = frame.copy()
        if result.empty:
            for column in ("spot", "zero_rate", "dividend_rate", "funding_rate", "dte", "tau"):
                result[column] = pd.Series(dtype=float)
            return result
        market_values = [
            self._get_market_values(date, exdate)
            for date, exdate in zip(result["date"], result["exdate"])
        ]
        market = pd.DataFrame(market_values, index=result.index)
        for column in market.columns:
            result[column] = market[column]
        return result

    def _get_market_values(self, date: pd.Timestamp, exdate: pd.Timestamp) -> dict[str, Any]:
        """Return market values."""
        if pd.isna(date) or pd.isna(exdate):
            raise DatasetBuildError("Option date/exdate contain missing values")
        date_value = int(pd.Timestamp(date).value)
        num_position = self._date_to_market_position.get(date_value)
        if num_position is None:
            raise DatasetBuildError(f"Market data are missing date {pd.Timestamp(date).date()}")
        num_dte = int((pd.Timestamp(exdate) - pd.Timestamp(date)).days)
        if num_dte < 0:
            raise DatasetBuildError("exdate precedes date")
        curve_days, curve_rates = self._market_data.zero_curves[date_value]
        zero_rate = float(np.interp(num_dte, curve_days, curve_rates))
        return {
            "spot": float(self._market_data.spots[num_position]),
            "zero_rate": zero_rate,
            "dividend_rate": float(self._market_data.dividend_rates[num_position]),
            "funding_rate": float(curve_rates[0]),
            "dte": num_dte,
            "tau": num_dte / 365.0,
        }

    def _get_selected_candidates(
        self, t0_chain: pd.DataFrame, t0: pd.Timestamp, expiry: pd.Timestamp
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return selected candidates."""
        target = t0_chain.loc[t0_chain["cp_flag"].eq(self.config.cp_flag)].copy()
        if target.empty:
            return (target, self._get_empty_audit_frame())
        target["cohort_id"] = expiry
        target["label"] = self.config.label
        target["moneyness_initial"] = target["strike"] / target["spot"]
        target["atm_distance"] = (target["moneyness_initial"] - 1.0).abs()
        target["is_quote_valid"] = self._get_quote_validity(target, is_initial=True)
        target["is_spread_valid"] = (
            target["is_quote_valid"]
            & np.isfinite(target["relative_spread"])
            & target["relative_spread"].ge(0)
            & target["relative_spread"].le(self.config.max_relative_spread)
        )
        target["is_open_interest_valid"] = target["open_interest"].ge(self.config.min_open_interest)
        target["is_volume_valid"] = target["volume"].ge(0)
        target["is_moneyness_valid"] = target["moneyness_initial"].between(
            self.config.moneyness_lower, self.config.moneyness_upper, inclusive="both"
        )
        target["resolved_iv"] = np.nan
        target["iv_source"] = ""
        for num_row in target.index:
            resolved_iv, iv_source = self._get_current_iv(target.loc[num_row], t0_chain)
            target.loc[num_row, "resolved_iv"] = resolved_iv
            target.loc[num_row, "iv_source"] = iv_source
        target["is_iv_valid"] = self._get_iv_validity(target["resolved_iv"])
        is_duplicated = target["optionid"].duplicated(keep=False)
        target["is_optionid_unique"] = ~is_duplicated
        target["is_candidate"] = target[
            [
                "is_quote_valid",
                "is_spread_valid",
                "is_open_interest_valid",
                "is_volume_valid",
                "is_moneyness_valid",
                "is_iv_valid",
                "is_optionid_unique",
            ]
        ].all(axis=1)
        target["is_selected"] = False
        target["selection_rank"] = pd.Series(pd.NA, index=target.index, dtype="Int64")
        target["is_lifecycle_valid"] = pd.Series(pd.NA, index=target.index, dtype="boolean")
        target["lifecycle_error"] = ""
        target["is_cohort_retained"] = False
        target["rejection_reason"] = target.apply(_get_rejection_reason, axis=1)
        candidates = target.loc[target["is_candidate"]].copy()
        num_required = self.config.num_moneyness
        if candidates.empty or (
            self.config.is_require_balanced_cohort and len(candidates) < num_required
        ):
            target.loc[target["is_candidate"], "rejection_reason"] = "insufficient_candidate_cohort"
            return (candidates.iloc[0:0], self._get_audit_columns(target))
        candidates = candidates.sort_values(
            by=["atm_distance", "relative_spread", "open_interest", "volume", "optionid"],
            ascending=[True, True, False, False, True],
            kind="mergesort",
        ).head(num_required)
        for selection_rank, num_row in enumerate(candidates.index, start=1):
            target.loc[num_row, "is_selected"] = True
            target.loc[num_row, "selection_rank"] = selection_rank
        target.loc[target["is_candidate"] & ~target["is_selected"], "rejection_reason"] = (
            "not_nearest"
        )
        candidates = target.loc[target["is_selected"]].sort_values("selection_rank")
        return (candidates, self._get_audit_columns(target))

    def _get_quote_validity(self, frame: pd.DataFrame, *, is_initial: bool) -> pd.Series:
        """Return quote validity."""
        bid = frame["best_bid"].astype(float)
        offer = frame["best_offer"].astype(float)
        mid = frame["quoted_mid_price"].astype(float)
        is_bid_valid = bid.gt(0) if is_initial else bid.ge(0)
        return (
            np.isfinite(bid)
            & np.isfinite(offer)
            & np.isfinite(mid)
            & is_bid_valid
            & offer.ge(bid)
            & mid.gt(0)
        )

    def _get_iv_validity(self, values: pd.Series | np.ndarray) -> np.ndarray:
        """Return IV validity."""
        array = np.asarray(values, dtype=float)
        return np.isfinite(array) & (array >= self.config.min_iv) & (array <= self.config.max_iv)

    def _get_current_iv(self, target: pd.Series, chain_at_date: pd.DataFrame) -> tuple[float, str]:
        """Return current IV."""
        vendor_iv = (
            float(target["impl_volatility"]) if pd.notna(target["impl_volatility"]) else np.nan
        )
        if bool(self._get_iv_validity(np.asarray([vendor_iv]))[0]):
            iv_source = (
                IV_SOURCE_VENDOR_CALL if str(target["cp_flag"]) == "C" else IV_SOURCE_VENDOR_PUT
            )
            return (vendor_iv, iv_source)
        if self.config.is_use_option_price_for_iv:
            solved_iv = self._get_row_implied_volatility(target)
            if bool(self._get_iv_validity(np.asarray([solved_iv]))[0]):
                iv_source = (
                    IV_SOURCE_CALL_PRICE if str(target["cp_flag"]) == "C" else IV_SOURCE_PUT_PRICE
                )
                return (solved_iv, iv_source)
        if self.config.is_use_same_strike_opposite_iv:
            opposite_iv = self._get_same_strike_opposite_iv(target, chain_at_date)
            if bool(self._get_iv_validity(np.asarray([opposite_iv]))[0]):
                iv_source = (
                    IV_SOURCE_SAME_STRIKE_PUT
                    if str(target["cp_flag"]) == "C"
                    else IV_SOURCE_SAME_STRIKE_CALL
                )
                return (opposite_iv, iv_source)
        if self.config.is_use_surface_iv:
            surface_iv = self._get_surface_iv(target, chain_at_date)
            if bool(self._get_iv_validity(np.asarray([surface_iv]))[0]):
                return (surface_iv, IV_SOURCE_SURFACE)
        return (float("nan"), "")

    def _get_row_implied_volatility(self, row: pd.Series) -> float:
        """Return row implied volatility."""
        return get_implied_volatility(
            option_price=float(row["quoted_mid_price"]),
            spot=float(row["spot"]),
            strike=float(row["strike"]),
            tau=float(row["tau"]),
            zero_rate=float(row["zero_rate"]),
            dividend_rate=float(row["dividend_rate"]),
            cp_flag=str(row["cp_flag"]),
            min_iv=self.config.min_iv,
            max_iv=self.config.max_iv,
            iv_solver_tolerance=self.config.iv_solver_tolerance,
            num_iv_solver_iterations=self.config.num_iv_solver_iterations,
        )

    def _get_same_strike_opposite_iv(self, target: pd.Series, chain_at_date: pd.DataFrame) -> float:
        """Return same strike opposite IV."""
        opposite_flag = "P" if str(target["cp_flag"]) == "C" else "C"
        opposite = chain_at_date.loc[
            chain_at_date["cp_flag"].eq(opposite_flag)
            & chain_at_date["symbol_root"].eq(target["symbol_root"])
            & np.isclose(
                chain_at_date["strike"].astype(float), float(target["strike"]), rtol=0.0, atol=1e-10
            )
        ].copy()
        if opposite.empty:
            return float("nan")
        opposite = opposite.loc[self._get_iv_validity(opposite["impl_volatility"])]
        opposite = opposite.loc[self._get_quote_validity(opposite, is_initial=False)]
        if opposite.empty:
            return float("nan")
        opposite = opposite.sort_values(
            ["relative_spread", "open_interest", "volume", "optionid"],
            ascending=[True, False, False, True],
            kind="mergesort",
            na_position="last",
        )
        return float(opposite.iloc[0]["impl_volatility"])

    def _get_surface_iv(self, target: pd.Series, chain_at_date: pd.DataFrame) -> float:
        """Return surface IV."""
        surface = chain_at_date.loc[
            chain_at_date["symbol_root"].eq(target["symbol_root"])
            & chain_at_date["cp_flag"].isin(["C", "P"])
        ].copy()
        surface = surface.loc[self._get_quote_validity(surface, is_initial=False)]
        if surface.empty:
            return float("nan")
        surface["surface_iv"] = pd.to_numeric(surface["impl_volatility"], errors="coerce").astype(
            float
        )
        is_missing_iv = ~self._get_iv_validity(surface["surface_iv"])
        if self.config.is_use_option_price_for_iv and is_missing_iv.any():
            surface.loc[is_missing_iv, "surface_iv"] = surface.loc[is_missing_iv].apply(
                self._get_row_implied_volatility, axis=1
            )
        surface = surface.loc[self._get_iv_validity(surface["surface_iv"])].copy()
        if surface.empty:
            return float("nan")
        surface["log_forward_moneyness"] = surface.apply(
            lambda row: get_log_forward_moneyness(
                spot=float(row["spot"]),
                strike=float(row["strike"]),
                tau=float(row["tau"]),
                zero_rate=float(row["zero_rate"]),
                dividend_rate=float(row["dividend_rate"]),
            ),
            axis=1,
        )
        is_otm = surface["cp_flag"].eq("C") & surface["log_forward_moneyness"].ge(0) | surface[
            "cp_flag"
        ].eq("P") & surface["log_forward_moneyness"].lt(0)
        surface = surface.loc[is_otm]
        target_log_moneyness = get_log_forward_moneyness(
            spot=float(target["spot"]),
            strike=float(target["strike"]),
            tau=float(target["tau"]),
            zero_rate=float(target["zero_rate"]),
            dividend_rate=float(target["dividend_rate"]),
        )
        return get_surface_implied_volatility(
            target_log_moneyness=target_log_moneyness,
            surface_log_moneyness=surface["log_forward_moneyness"].to_numpy(),
            surface_iv=surface["surface_iv"].to_numpy(),
            num_iv_surface_points=self.config.num_iv_surface_points,
            max_iv_extrapolation_log_moneyness=self.config.max_iv_extrapolation_log_moneyness,
        )

    def _get_episode(
        self,
        *,
        selected: pd.Series,
        chain_window: pd.DataFrame,
        t0: pd.Timestamp,
        expiry: pd.Timestamp,
    ) -> pd.DataFrame:
        """Return episode."""
        optionid = int(selected["optionid"])
        num_expiry_position = int(self._calendar.get_indexer([expiry])[0])
        expected_dates = self._calendar[
            num_expiry_position - self.config.num_interval : num_expiry_position + 1
        ]
        preterminal_dates = expected_dates[:-1]
        option_rows = chain_window.loc[chain_window["optionid"].eq(optionid)].copy()
        if option_rows["date"].duplicated().any():
            raise EpisodeValidationError(f"optionid={optionid} contains duplicate dates")
        preterminal = option_rows.loc[option_rows["date"].isin(preterminal_dates)].copy()
        if len(preterminal) != self.config.num_interval:
            raise EpisodeValidationError(
                f"optionid={optionid} nonterminal quote count is {len(preterminal)}; expected {self.config.num_interval}"
            )
        if not pd.DatetimeIndex(preterminal["date"]).sort_values().equals(preterminal_dates):
            raise EpisodeValidationError(
                f"optionid={optionid} is missing expected trading-day quotes"
            )
        invariant_columns = ("optionid", "cp_flag", "symbol", "strike", "exdate")
        for column in invariant_columns:
            if preterminal[column].nunique(dropna=False) != 1:
                raise EpisodeValidationError(
                    f"optionid={optionid} {column} changes within the path"
                )
        if not self._get_quote_validity(preterminal, is_initial=False).all():
            raise EpisodeValidationError(f"optionid={optionid} contains invalid nonterminal quotes")
        preterminal = preterminal.sort_values("date", kind="mergesort").reset_index(drop=True)
        resolved_values: list[float] = []
        iv_sources: list[str] = []
        previous_iv = float("nan")
        num_steps_since_iv = self.config.num_iv_forward_fill_steps + 1
        for row in preterminal.itertuples(index=False):
            row_series = pd.Series(row._asdict())
            chain_at_date = chain_window.loc[chain_window["date"].eq(row.date)]
            resolved_iv, iv_source = self._get_current_iv(row_series, chain_at_date)
            if np.isfinite(resolved_iv):
                previous_iv = resolved_iv
                num_steps_since_iv = 0
            else:
                num_steps_since_iv += 1
                if (
                    self.config.is_use_past_iv
                    and np.isfinite(previous_iv)
                    and (num_steps_since_iv <= self.config.num_iv_forward_fill_steps)
                ):
                    resolved_iv = previous_iv
                    iv_source = IV_SOURCE_PAST
            if not bool(self._get_iv_validity(np.asarray([resolved_iv]))[0]):
                raise EpisodeValidationError(
                    f"optionid={optionid} at {pd.Timestamp(row.date).date()} has no recoverable IV"
                )
            resolved_values.append(float(resolved_iv))
            iv_sources.append(iv_source)
        preterminal["resolved_iv"] = resolved_values
        preterminal["iv_source"] = pd.Series(iv_sources, dtype="string")
        preterminal["mid_price"] = preterminal["quoted_mid_price"].astype(float)
        terminal = self._get_terminal_row(selected=selected, option_rows=option_rows, expiry=expiry)
        episode = pd.concat([preterminal, terminal], ignore_index=True, sort=False)
        episode["step"] = np.arange(len(episode), dtype=np.int16)
        episode["is_terminal"] = episode["step"].eq(self.config.num_interval)
        episode["selection_rank"] = int(selected["selection_rank"])
        episode["cohort_id"] = expiry
        episode["label"] = self.config.label
        episode_id = f"{expiry:%Y%m%d}_rank{int(selected['selection_rank']):02d}_{optionid}"
        episode["episode_id"] = episode_id
        s0 = float(episode.iloc[0]["spot"])
        if not np.isfinite(s0) or s0 <= 0:
            raise EpisodeValidationError(f"optionid={optionid} has invalid S0")
        episode["spot_norm"] = episode["spot"] / s0
        episode["strike_norm"] = episode["strike"] / s0
        episode["option_mid_norm"] = episode["mid_price"] / s0
        episode["moneyness_initial"] = float(selected["moneyness_initial"])
        episode["atm_distance"] = float(selected["atm_distance"])
        episode["is_environment_ready"] = episode["symbol_root"].eq("SPXW")
        if self.config.is_require_environment_ready and (not episode["is_environment_ready"].all()):
            raise EpisodeValidationError(f"optionid={optionid} does not use PM cash settlement")
        return self._get_typed_episode(episode)

    def _get_terminal_row(
        self, *, selected: pd.Series, option_rows: pd.DataFrame, expiry: pd.Timestamp
    ) -> pd.DataFrame:
        """Return terminal row."""
        terminal_rows = option_rows.loc[option_rows["date"].eq(expiry)].copy()
        if len(terminal_rows) > 1:
            raise EpisodeValidationError(
                f"optionid={int(selected['optionid'])} contains multiple terminal quotes"
            )
        if terminal_rows.empty:
            raw = {column: pd.NA for column in OPTION_SCAN_COLUMNS}
            for column in ("exdate", "cp_flag", "strike_price", "optionid", "symbol"):
                raw[column] = selected[column]
            raw["date"] = expiry
            terminal = self._get_prepared_option_frame(pd.DataFrame([raw]))
        else:
            terminal = terminal_rows.iloc[[0]].copy()
        terminal["resolved_iv"] = np.nan
        terminal["iv_source"] = IV_SOURCE_TERMINAL
        symbol_root = str(terminal.iloc[0]["symbol_root"])
        cp_flag = str(terminal.iloc[0]["cp_flag"])
        spot = float(terminal.iloc[0]["spot"])
        strike = float(terminal.iloc[0]["strike"])
        if symbol_root == "SPXW":
            payoff = max(spot - strike, 0.0) if cp_flag == "C" else max(strike - spot, 0.0)
            terminal["mid_price"] = float(payoff)
        else:
            terminal["mid_price"] = terminal["quoted_mid_price"]
        return terminal

    def _get_typed_episode(self, episode: pd.DataFrame) -> pd.DataFrame:
        """Return typed episode."""
        result = episode.reindex(columns=EPISODE_STEP_COLUMNS).copy()
        string_columns = ("episode_id", "label", "symbol", "symbol_root", "cp_flag", "iv_source")
        for column in string_columns:
            result[column] = result[column].astype("string")
        result["cohort_id"] = pd.to_datetime(result["cohort_id"])
        result["date"] = pd.to_datetime(result["date"])
        result["exdate"] = pd.to_datetime(result["exdate"])
        result["selection_rank"] = result["selection_rank"].astype(np.int16)
        result["optionid"] = result["optionid"].astype(np.int64)
        result["step"] = result["step"].astype(np.int16)
        result["dte"] = result["dte"].astype(np.int16)
        for column in (
            "is_terminal",
            "is_environment_ready",
            "is_open_interest_missing",
            "is_volume_missing",
        ):
            result[column] = result[column].astype(bool)
        float_columns = (
            set(EPISODE_STEP_COLUMNS)
            - set(string_columns)
            - {
                "cohort_id",
                "date",
                "exdate",
                "selection_rank",
                "optionid",
                "step",
                "dte",
                "is_terminal",
                "is_environment_ready",
                "is_open_interest_missing",
                "is_volume_missing",
            }
        )
        for column in float_columns:
            result[column] = pd.to_numeric(result[column], errors="coerce").astype(float)
        return result

    def _get_manifest_row(self, episode: pd.DataFrame) -> dict[str, Any]:
        """Return manifest row."""
        initial = episode.iloc[0]
        num_iv_imputed = int(
            (~episode.loc[~episode["is_terminal"], "iv_source"].isin(VENDOR_IV_SOURCES)).sum()
        )
        return {
            "episode_id": str(initial["episode_id"]),
            "cohort_id": pd.Timestamp(initial["cohort_id"]),
            "label": self.config.label,
            "selection_rank": int(initial["selection_rank"]),
            "optionid": int(initial["optionid"]),
            "symbol": str(initial["symbol"]),
            "symbol_root": str(initial["symbol_root"]),
            "cp_flag": str(initial["cp_flag"]),
            "t0": pd.Timestamp(initial["date"]),
            "exdate": pd.Timestamp(episode.iloc[-1]["date"]),
            "num_interval": self.config.num_interval,
            "num_state": len(episode),
            "s0": float(initial["spot"]),
            "c0": float(initial["mid_price"]),
            "strike": float(initial["strike"]),
            "moneyness_initial": float(initial["moneyness_initial"]),
            "atm_distance": float(initial["atm_distance"]),
            "relative_spread_initial": float(initial["relative_spread"]),
            "open_interest_initial": float(initial["open_interest"]),
            "volume_initial": float(initial["volume"]),
            "num_iv_imputed": num_iv_imputed,
            "is_environment_ready": bool(episode["is_environment_ready"].all()),
        }

    def _get_episode_steps(self, frames: list[pd.DataFrame]) -> pd.DataFrame:
        """Return episode steps."""
        if not frames:
            return _get_empty_frame(EPISODE_STEP_COLUMNS)
        result = pd.concat(frames, ignore_index=True)
        return result.sort_values(
            ["cohort_id", "selection_rank", "step"], kind="mergesort"
        ).reset_index(drop=True)

    def _get_episode_manifest(self, rows: list[dict[str, Any]]) -> pd.DataFrame:
        """Return episode manifest."""
        if not rows:
            return _get_empty_frame(MANIFEST_COLUMNS)
        result = pd.DataFrame(rows).reindex(columns=MANIFEST_COLUMNS)
        result = result.sort_values(["cohort_id", "selection_rank"], kind="mergesort").reset_index(
            drop=True
        )
        for column in ("episode_id", "label", "symbol", "symbol_root", "cp_flag"):
            result[column] = result[column].astype("string")
        for column in ("cohort_id", "t0", "exdate"):
            result[column] = pd.to_datetime(result[column])
        for column in ("selection_rank", "optionid", "num_interval", "num_state", "num_iv_imputed"):
            result[column] = result[column].astype(np.int64)
        result["is_environment_ready"] = result["is_environment_ready"].astype(bool)
        return result

    def _get_empty_audit_frame(self) -> pd.DataFrame:
        """Return empty audit frame."""
        return _get_empty_frame(CANDIDATE_AUDIT_COLUMNS)

    def _get_audit_columns(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return audit columns."""
        return frame.reindex(columns=CANDIDATE_AUDIT_COLUMNS).copy()

    def _get_candidate_audit(self, frames: list[pd.DataFrame]) -> pd.DataFrame:
        """Return candidate audit."""
        frames = [frame for frame in frames if not frame.empty]
        if not frames:
            return _get_empty_frame(CANDIDATE_AUDIT_COLUMNS)
        result = pd.concat(frames, ignore_index=True)
        result = result.sort_values(
            ["cohort_id", "atm_distance", "relative_spread", "optionid"],
            kind="mergesort",
            na_position="last",
        ).reset_index(drop=True)
        for column in ("cohort_id", "date", "exdate"):
            result[column] = pd.to_datetime(result[column])
        for column in (
            "label",
            "symbol",
            "symbol_root",
            "cp_flag",
            "iv_source",
            "lifecycle_error",
            "rejection_reason",
        ):
            result[column] = result[column].astype("string")
        result["optionid"] = result["optionid"].astype(np.int64)
        result["selection_rank"] = result["selection_rank"].astype("Int64")
        boolean_columns = (
            "is_open_interest_missing",
            "is_volume_missing",
            "is_quote_valid",
            "is_spread_valid",
            "is_open_interest_valid",
            "is_volume_valid",
            "is_moneyness_valid",
            "is_iv_valid",
            "is_optionid_unique",
            "is_candidate",
            "is_selected",
            "is_lifecycle_valid",
            "is_cohort_retained",
        )
        for column in boolean_columns:
            result[column] = result[column].astype("boolean")
        return result

    def _get_build_report(
        self,
        *,
        episode_steps: pd.DataFrame,
        episode_manifest: pd.DataFrame,
        candidate_audit: pd.DataFrame,
    ) -> dict[str, Any]:
        """Return build report."""
        iv_counts = (
            Counter(
                episode_steps.loc[
                    ~episode_steps.get("is_terminal", pd.Series(dtype=bool)), "iv_source"
                ].astype(str)
            )
            if not episode_steps.empty
            else Counter()
        )
        moneyness = episode_manifest.get("moneyness_initial", pd.Series(dtype=float))
        rejection_counts = Counter(
            candidate_audit.get("rejection_reason", pd.Series(dtype="string"))
            .fillna("")
            .astype(str)
        )
        rejection_counts.pop("", None)
        if len(moneyness):
            moneyness_quantiles = {
                f"q{int(probability * 100):02d}": float(moneyness.quantile(probability))
                for probability in (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)
            }
        else:
            moneyness_quantiles = {}
        if not episode_steps.empty:
            decision_spread = episode_steps.loc[~episode_steps["is_terminal"], "relative_spread"]
            decision_spread = decision_spread.loc[np.isfinite(decision_spread)]
            spread_quantiles = {
                f"q{int(probability * 100):02d}": float(decision_spread.quantile(probability))
                for probability in (0.5, 0.9, 0.95, 0.99)
            }
        else:
            decision_spread = pd.Series(dtype=float)
            spread_quantiles = {}
        return {
            "label": self.config.label,
            "config": self.config.get_dict(),
            "num_expiry_calendar": self._num_expiry_calendar,
            "num_expiry_without_complete_window": self._num_expiry_without_complete_window,
            "num_expiry_without_quotes": self._num_expiry_without_quotes,
            "num_expiry_with_quotes": self._num_expiry_with_quotes,
            "num_cohort_insufficient_candidates": self._num_cohort_insufficient_candidates,
            "num_cohort_lifecycle_failed": self._num_cohort_lifecycle_failed,
            "num_cohort_retained": self._num_cohort_retained,
            "num_episode": len(episode_manifest),
            "num_step": len(episode_steps),
            "num_candidate_audit_row": len(candidate_audit),
            "iv_source_counts": dict(sorted(iv_counts.items())),
            "rejection_reason_counts": dict(sorted(rejection_counts.items())),
            "moneyness_initial": {
                "mean": float(moneyness.mean()) if len(moneyness) else None,
                "std": float(moneyness.std(ddof=0)) if len(moneyness) else None,
                "min": float(moneyness.min()) if len(moneyness) else None,
                "max": float(moneyness.max()) if len(moneyness) else None,
                "quantiles": moneyness_quantiles,
            },
            "decision_relative_spread": {
                "num_finite": len(decision_spread),
                "max": float(decision_spread.max()) if len(decision_spread) else None,
                "quantiles": spread_quantiles,
            },
            "actual_episode_date_start": (
                episode_manifest["t0"].min().strftime("%Y-%m-%d")
                if not episode_manifest.empty
                else None
            ),
            "actual_episode_date_end": (
                episode_manifest["exdate"].max().strftime("%Y-%m-%d")
                if not episode_manifest.empty
                else None
            ),
            "input_files": self._get_input_fingerprints(),
        }

    def _get_input_fingerprints(self) -> list[dict[str, Any]]:
        """Return input fingerprints."""
        option_files = discover_option_files(
            self.config.data_root, start=self.config.date_start.year, end=self.config.date_end.year
        )
        paths = list(option_files.values()) + [
            self.config.data_root / "spx_spot.parquet",
            self.config.data_root / "zero_curve.parquet",
            self.config.data_root / "spx_div_yield.parquet",
        ]
        results: list[dict[str, Any]] = []
        for path in paths:
            stat = path.stat()
            signature = f"{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}"
            results.append(
                {
                    "path": str(path),
                    "num_bytes": stat.st_size,
                    "modified_time_ns": stat.st_mtime_ns,
                    "metadata_sha256": hashlib.sha256(signature.encode("utf-8")).hexdigest(),
                }
            )
        return results
