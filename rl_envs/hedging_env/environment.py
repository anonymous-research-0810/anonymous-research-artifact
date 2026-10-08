"""Environment for the paper option-hedging pipeline."""

from __future__ import annotations
from collections.abc import Mapping
from typing import Any
import numpy as np
import pandas as pd
from gymnasium.spaces import Box
from scipy.special import ndtr
from option_dataset import OptionDataset
from .constants import (
    DEFAULT_CONTRACT_MULTIPLIER,
    DEFAULT_HEDGE_COST_RATE,
    DEFAULT_RECONCILIATION_TOLERANCE,
    DEFAULT_REWARD_RISK_AVERSION_XI,
    DEFAULT_RISK_AVERSION_LAMBDA,
    DEFAULT_TRAINING_REWARD_SCALES,
    EXOGENOUS_FEATURE_NAMES,
    REWARD_FORMULATION_CASH_FLOW,
    REWARD_FORMULATION_SHAPED_ACCOUNTING,
    STATE_FEATURE_NAMES,
    VALID_REWARD_FORMULATIONS,
)
from .exceptions import (
    EnvironmentConfigurationError,
    EnvironmentStateError,
    RewardReconciliationError,
)
from .normalization import StateNormalizer
from .schedule import get_episode_schedule


class HedgingEnv:
    """Self-financing short-call hedging environment. Holdings lie in [0, 1]; cash flows include dividends, financing, opening costs, and terminal liquidation."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        dataset: OptionDataset,
        *,
        reward_formulation: str = REWARD_FORMULATION_SHAPED_ACCOUNTING,
        hedge_cost_rate: float = DEFAULT_HEDGE_COST_RATE,
        contract_multiplier: float = DEFAULT_CONTRACT_MULTIPLIER,
        currency: str = "USD",
        reconciliation_tolerance: float = DEFAULT_RECONCILIATION_TOLERANCE,
        risk_aversion_lambda: float = DEFAULT_RISK_AVERSION_LAMBDA,
        reward_risk_aversion_xi: float = DEFAULT_REWARD_RISK_AVERSION_XI,
        training_reward_scale: float | None = None,
        state_normalizer: StateNormalizer | None = None,
        is_batch_mode: bool | None = None,
        num_parallel_episode: int | None = None,
        is_repeated: bool = False,
        num_training_episode: int | None = None,
        is_clip_action: bool = False,
        is_record_trajectory: bool = True,
        observation_dtype: np.dtype | type = np.float32,
        seed: int | None = None,
    ) -> None:
        """Initialize validated configuration and internal state."""
        self._validate_dataset(dataset)
        self.dataset = dataset
        self.label = dataset.label
        self.reward_formulation = str(reward_formulation).lower()
        if self.reward_formulation not in VALID_REWARD_FORMULATIONS:
            raise EnvironmentConfigurationError(
                f"reward_formulation must be {sorted(VALID_REWARD_FORMULATIONS)}  "
            )
        self.hedge_cost_rate = self._get_nonnegative_float(hedge_cost_rate, "hedge_cost_rate")
        self.contract_multiplier = self._get_positive_float(
            contract_multiplier, "contract_multiplier"
        )
        currency_text = str(currency).strip().upper()
        if len(currency_text) != 3 or not currency_text.isalpha():
            raise EnvironmentConfigurationError("currency must be a three-letter currency code")
        self.currency = currency_text
        self.reconciliation_tolerance = self._get_positive_float(
            reconciliation_tolerance, "reconciliation_tolerance"
        )
        self.risk_aversion_lambda = self._get_nonnegative_float(
            risk_aversion_lambda, "risk_aversion_lambda"
        )
        self.reward_risk_aversion_xi = self._get_nonnegative_float(
            reward_risk_aversion_xi, "reward_risk_aversion_xi"
        )
        resolved_training_reward_scale = (
            DEFAULT_TRAINING_REWARD_SCALES[self.reward_formulation]
            if training_reward_scale is None
            else training_reward_scale
        )
        self.training_reward_scale = self._get_positive_float(
            resolved_training_reward_scale, "training_reward_scale"
        )
        self.state_normalizer = state_normalizer
        if state_normalizer is not None and (not isinstance(state_normalizer, StateNormalizer)):
            raise EnvironmentConfigurationError(
                "state_normalizer must be a StateNormalizer or None"
            )
        self.is_batch_mode = self.label != "train" if is_batch_mode is None else is_batch_mode
        for name, value in (
            ("is_batch_mode", self.is_batch_mode),
            ("is_repeated", is_repeated),
            ("is_clip_action", is_clip_action),
            ("is_record_trajectory", is_record_trajectory),
        ):
            if not isinstance(value, (bool, np.bool_)):
                raise EnvironmentConfigurationError(f"{name} must be a boolean")
        self.is_batch_mode = bool(self.is_batch_mode)
        self.is_repeated = bool(is_repeated)
        self.is_clip_action = bool(is_clip_action)
        self.is_record_trajectory = bool(is_record_trajectory)
        if self.label == "train" and self.is_batch_mode:
            raise EnvironmentConfigurationError(
                "Training requires single-episode interaction; is_batch_mode=True is unsupported"
            )
        if self.is_batch_mode:
            self.num_parallel_episode = (
                dataset.num_episode
                if num_parallel_episode is None
                else self._get_positive_int(num_parallel_episode, "num_parallel_episode")
            )
        else:
            if num_parallel_episode not in (None, 1):
                raise EnvironmentConfigurationError(
                    "num_parallel_episode must be None or 1 in single-episode mode"
                )
            self.num_parallel_episode = 1
        dtype = np.dtype(observation_dtype)
        if dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
            raise EnvironmentConfigurationError(
                "observation_dtype must be np.float32 or np.float64"
            )
        self.observation_dtype = dtype
        self.num_interval = int(dataset.config.num_interval)
        self.num_state_feature = len(STATE_FEATURE_NAMES)
        self.state_feature_names = STATE_FEATURE_NAMES
        self.exogenous_feature_names = EXOGENOUS_FEATURE_NAMES
        observation_low = np.full(self.num_state_feature, -np.inf, dtype=dtype)
        observation_high = np.full(self.num_state_feature, np.inf, dtype=dtype)
        observation_low[2] = 0.0
        observation_high[2] = 1.0
        self.observation_space = Box(low=observation_low, high=observation_high, dtype=dtype)
        self.action_space = Box(low=0.0, high=1.0, shape=(1,), dtype=dtype)
        self._seed = seed
        self._schedule = get_episode_schedule(
            dataset,
            is_repeated=self.is_repeated,
            num_training_episode=num_training_episode,
            seed=seed,
        )
        self._num_schedule_position = 0
        self._is_active = False
        self._is_done = False
        self._num_step = 0
        self._num_active_episode = 0
        self._completed_results: list[dict[str, Any]] = []
        self._trajectory_records: list[dict[str, Any]] = []

    @staticmethod
    def _get_nonnegative_float(value: Any, name: str) -> float:
        """Return nonnegative float."""
        if isinstance(value, (bool, np.bool_)):
            raise EnvironmentConfigurationError(f"{name} must be finite and nonnegative")
        try:
            result = float(value)
        except (TypeError, ValueError) as exc:
            raise EnvironmentConfigurationError(f"{name} must be finite and nonnegative") from exc
        if not np.isfinite(result) or result < 0:
            raise EnvironmentConfigurationError(f"{name} must be finite and nonnegative")
        return result

    @classmethod
    def _get_positive_float(cls, value: Any, name: str) -> float:
        """Return positive float."""
        result = cls._get_nonnegative_float(value, name)
        if result <= 0:
            raise EnvironmentConfigurationError(f"{name} must be greater than 0")
        return result

    @staticmethod
    def _get_positive_int(value: Any, name: str) -> int:
        """Return positive int."""
        if isinstance(value, (bool, np.bool_)):
            raise EnvironmentConfigurationError(f"{name} must be a positive integer")
        try:
            result = int(value)
        except (TypeError, ValueError) as exc:
            raise EnvironmentConfigurationError(f"{name} must be a positive integer") from exc
        if result != value or result <= 0:
            raise EnvironmentConfigurationError(f"{name} must be a positive integer")
        return result

    @staticmethod
    def _validate_dataset(dataset: OptionDataset) -> None:
        """Validate dataset."""
        if not isinstance(dataset, OptionDataset):
            raise EnvironmentConfigurationError("dataset must be an OptionDataset")
        if dataset.num_episode == 0:
            raise EnvironmentConfigurationError("The environment cannot use an empty OptionDataset")
        if not dataset.is_environment_ready:
            raise EnvironmentConfigurationError(
                "dataset contains episodes with unsupported settlement conventions"
            )
        cp_flags = set(dataset.episode_manifest["cp_flag"].astype(str))
        symbol_roots = set(dataset.episode_manifest["symbol_root"].astype(str))
        if cp_flags != {"C"}:
            raise EnvironmentConfigurationError(
                "The environment supports short European calls only"
            )
        if symbol_roots != {"SPXW"}:
            raise EnvironmentConfigurationError("The environment requires PM cash settlement")

    @property
    def is_schedule_exhausted(self) -> bool:
        """Return whether schedule exhausted."""
        return self._num_schedule_position >= len(self._schedule)

    @property
    def num_completed_episode(self) -> int:
        """Return the number of completed episode."""
        return len(self._completed_results)

    @property
    def num_scheduled_episode(self) -> int:
        """Return the number of scheduled episode."""
        return len(self._schedule)

    @property
    def num_remaining_episode(self) -> int:
        """Return the number of remaining episode."""
        return len(self._schedule) - self._num_schedule_position

    def get_schedule(self) -> pd.DataFrame:
        """Return schedule."""
        return self._schedule.copy(deep=True)

    def get_config(self) -> dict[str, Any]:
        """Return the configuration required to reconstruct this object."""
        return {
            "label": self.label,
            "reward_formulation": self.reward_formulation,
            "hedge_cost_rate": self.hedge_cost_rate,
            "contract_multiplier": self.contract_multiplier,
            "reconciliation_tolerance": self.reconciliation_tolerance,
            "risk_aversion_lambda": self.risk_aversion_lambda,
            "reward_risk_aversion_xi": self.reward_risk_aversion_xi,
            "training_reward_scale": self.training_reward_scale,
            "currency": self.currency,
            "metric_unit": f"{self.currency}_per_option_contract",
            "is_batch_mode": self.is_batch_mode,
            "num_parallel_episode": self.num_parallel_episode,
            "is_repeated": self.is_repeated,
            "is_clip_action": self.is_clip_action,
            "is_record_trajectory": self.is_record_trajectory,
            "observation_dtype": self.observation_dtype.name,
            "num_interval": self.num_interval,
            "num_scheduled_episode": self.num_scheduled_episode,
            "state_feature_names": list(self.state_feature_names),
            "state_normalizer": (
                self.state_normalizer.get_dict() if self.state_normalizer is not None else None
            ),
        }

    def reset(
        self, *, seed: int | None = None, options: Mapping[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset."""
        if options:
            raise EnvironmentConfigurationError("reset options must be None or an empty mapping")
        if self._is_active and (not self._is_done):
            raise EnvironmentStateError(
                "The current episode is incomplete; an early reset is unsupported"
            )
        if seed is not None:
            if self._num_schedule_position != 0 or self._completed_results:
                raise EnvironmentStateError(
                    "reset(seed=...) cannot reorder a schedule after it starts"
                )
            self._seed = seed
            self._schedule = get_episode_schedule(
                self.dataset,
                is_repeated=self.is_repeated,
                num_training_episode=len(self._schedule) if self.is_repeated else None,
                seed=seed,
            )
        if self.is_schedule_exhausted:
            raise StopIteration(f"{self.label} episode schedule has been exhausted")
        num_batch = min(
            self.num_parallel_episode, len(self._schedule) - self._num_schedule_position
        )
        schedule_rows = self._schedule.iloc[
            self._num_schedule_position : self._num_schedule_position + num_batch
        ].copy()
        self._num_schedule_position += num_batch
        self._load_episode_batch(schedule_rows)
        observation = self._get_observation(self._num_step, self._holding)
        info = self._get_reset_info()
        return (self._get_public_array(observation), info)

    def _load_episode_batch(self, schedule_rows: pd.DataFrame) -> None:
        """Load episode batch."""
        frames = [
            self.dataset.get_episode(int(num_index), is_copy=False)
            for num_index in schedule_rows["episode_index"]
        ]
        expected_steps = np.arange(self.num_interval + 1)
        for frame in frames:
            if len(frame) != self.num_interval + 1:
                raise EnvironmentConfigurationError("Episode row count differs from num_interval")
            if not np.array_equal(frame["step"].to_numpy(), expected_steps):
                raise EnvironmentConfigurationError(
                    "Episode steps must run consecutively from 0 to H"
                )
            if bool(frame.iloc[:-1]["is_terminal"].any()) or not bool(
                frame.iloc[-1]["is_terminal"]
            ):
                raise EnvironmentConfigurationError("Episode terminal flags are invalid")

        def get_float_matrix(column: str) -> np.ndarray:
            """Return float matrix."""
            return np.stack(
                [frame[column].to_numpy(dtype=np.float64, copy=True) for frame in frames]
            )

        self._schedule_rows = schedule_rows.reset_index(drop=True)
        self._num_active_episode = len(frames)
        self._episode_ids = self._schedule_rows["episode_id"].astype(str).to_numpy()
        self._episode_indices = self._schedule_rows["episode_index"].to_numpy(dtype=np.int64)
        self._draw_ids = self._schedule_rows["draw_id"].to_numpy(dtype=np.int64)
        self._cohort_ids = pd.to_datetime(self._schedule_rows["cohort_id"]).to_numpy(
            dtype="datetime64[ns]"
        )
        self._dates = np.stack([frame["date"].to_numpy(dtype="datetime64[ns]") for frame in frames])
        self._spot_norm = get_float_matrix("spot_norm")
        self._strike_norm = get_float_matrix("strike_norm")
        self._option_norm = get_float_matrix("option_mid_norm")
        self._tau = get_float_matrix("tau")
        self._iv = get_float_matrix("resolved_iv")
        self._zero_rate = get_float_matrix("zero_rate")
        self._dividend_rate = get_float_matrix("dividend_rate")
        self._funding_rate = get_float_matrix("funding_rate")
        self._s0 = np.asarray(
            [float(self.dataset.get_manifest_row(int(i))["s0"]) for i in self._episode_indices]
        )
        decision_arrays = (
            self._spot_norm[:, :-1],
            self._strike_norm[:, :-1],
            self._option_norm[:, :-1],
            self._tau[:, :-1],
            self._iv[:, :-1],
            self._zero_rate[:, :-1],
            self._dividend_rate[:, :-1],
            self._funding_rate[:, :-1],
        )
        if not all((np.isfinite(array).all() for array in decision_arrays)):
            raise EnvironmentConfigurationError(
                "Nonterminal episode fields contain nonfinite values"
            )
        terminal_arrays = (
            self._spot_norm[:, -1],
            self._strike_norm[:, -1],
            self._option_norm[:, -1],
            self._funding_rate[:, -1],
        )
        if not all((np.isfinite(array).all() for array in terminal_arrays)):
            raise EnvironmentConfigurationError(
                "Terminal settlement fields contain nonfinite values"
            )
        if np.any(self._spot_norm <= 0) or np.any(self._strike_norm <= 0):
            raise EnvironmentConfigurationError(
                "Normalized spot and strike must be strictly positive"
            )
        if np.any(self._option_norm < 0):
            raise EnvironmentConfigurationError("Normalized option values must be nonnegative")
        if not np.allclose(self._spot_norm[:, 0], 1.0, rtol=0.0, atol=1e-12):
            raise EnvironmentConfigurationError("Initial spot_norm must equal 1 in every episode")
        if not np.allclose(self._strike_norm, self._strike_norm[:, [0]], rtol=0.0, atol=1e-12):
            raise EnvironmentConfigurationError(
                "strike_norm must remain constant within each episode"
            )
        if not np.allclose(self._tau[:, -1], 0.0, rtol=0.0, atol=1e-12):
            raise EnvironmentConfigurationError("terminal tau must be 0")
        if np.any(np.diff(self._tau, axis=1) >= 0):
            raise EnvironmentConfigurationError("tau must strictly decrease with step")
        terminal_payoff = np.maximum(self._spot_norm[:, -1] - self._strike_norm[:, -1], 0.0)
        if not np.allclose(self._option_norm[:, -1], terminal_payoff, rtol=0.0, atol=1e-10):
            raise EnvironmentConfigurationError(
                "Terminal call option_mid_norm must equal the cash-settlement payoff"
            )
        date_deltas = np.diff(self._dates, axis=1).astype("timedelta64[D]").astype(int)
        if np.any(date_deltas <= 0):
            raise EnvironmentConfigurationError("Episode dates must be strictly increasing")
        self._num_step = 0
        self._holding = np.zeros(self._num_active_episode, dtype=np.float64)
        self._discount_factor = np.ones(self._num_active_episode, dtype=np.float64)
        self._accounting_wealth = np.zeros(self._num_active_episode, dtype=np.float64)
        self._cash_flow_wealth = np.zeros(self._num_active_episode, dtype=np.float64)
        self._discounted_transaction_cost = np.zeros(self._num_active_episode, dtype=np.float64)
        self._is_active = True
        self._is_done = False

    def _get_reset_info(self) -> dict[str, Any]:
        """Return reset info."""
        info = {
            "label": self.label,
            "draw_id": self._draw_ids.copy(),
            "episode_index": self._episode_indices.copy(),
            "episode_id": self._episode_ids.copy(),
            "cohort_id": self._cohort_ids.copy(),
            "date": self._dates[:, 0].copy(),
            "step": np.zeros(self._num_active_episode, dtype=np.int64),
            "holding_norm": self._holding.copy(),
            "is_batch_mode": self.is_batch_mode,
            "reward_formulation": self.reward_formulation,
            "training_reward_scale": self.training_reward_scale,
            "is_state_normalized": self.state_normalizer is not None,
        }
        return self._get_public_info(info)

    def _get_observation(self, num_step: int, holding: np.ndarray) -> np.ndarray:
        """Return observation."""
        external = np.column_stack(
            [
                self._spot_norm[:, num_step],
                self._strike_norm[:, num_step],
                self._tau[:, num_step],
                self._iv[:, num_step],
                self._zero_rate[:, num_step],
                self._dividend_rate[:, num_step],
            ]
        )
        if self.state_normalizer is not None:
            external = self.state_normalizer.transform(external, dtype=np.float64)
        observation = np.empty((self._num_active_episode, self.num_state_feature), dtype=np.float64)
        observation[:, 0:2] = external[:, 0:2]
        observation[:, 2] = holding
        observation[:, 3:] = external[:, 2:]
        if not np.isfinite(observation).all():
            raise EnvironmentStateError("Actor decision states contain nonfinite values")
        return observation.astype(self.observation_dtype, copy=False)

    def _get_action_array(self, action: Any) -> np.ndarray:
        """Return action array."""
        values = np.asarray(action, dtype=np.float64)
        if values.ndim == 0:
            values = values.reshape(1)
        elif values.ndim == 2 and values.shape[1] == 1:
            values = values[:, 0]
        elif values.ndim != 1:
            raise EnvironmentConfigurationError(
                "action must be a scalar or have shape (N,) or (N,1)"
            )
        if len(values) != self._num_active_episode:
            raise EnvironmentConfigurationError(
                f"action count is {len(values)}; expected {self._num_active_episode}"
            )
        if not np.isfinite(values).all():
            raise EnvironmentConfigurationError("action contains nonfinite values")
        is_outside = (values < 0.0) | (values > 1.0)
        if is_outside.any() and (not self.is_clip_action):
            raise EnvironmentConfigurationError("action must be in the closed interval [0,1]")
        return np.clip(values, 0.0, 1.0)

    def step(
        self, action: Any
    ) -> tuple[
        np.ndarray, float | np.ndarray, bool | np.ndarray, bool | np.ndarray, dict[str, Any]
    ]:
        """Step."""
        if not self._is_active or self._is_done:
            raise EnvironmentStateError(
                "Reset before stepping; a terminated episode cannot continue to step"
            )
        actions = self._get_action_array(action)
        num_step = self._num_step
        num_next_step = num_step + 1
        is_terminal_step = num_next_step == self.num_interval
        num_calendar_day = (
            (self._dates[:, num_next_step] - self._dates[:, num_step])
            .astype("timedelta64[D]")
            .astype(np.float64)
        )
        delta_time = num_calendar_day / 365.0
        discount = self._discount_factor
        next_discount = discount * np.exp(-self._funding_rate[:, num_step] * delta_time)
        spot = self._spot_norm[:, num_step]
        next_spot = self._spot_norm[:, num_next_step]
        option = self._option_norm[:, num_step]
        next_option = self._option_norm[:, num_next_step]
        dividend_cash = spot * (np.exp(self._dividend_rate[:, num_step] * delta_time) - 1.0)
        discounted_gain = next_discount * (next_spot + dividend_cash) - discount * spot
        turnover = np.abs(actions - self._holding)
        rebalance_cost = self.hedge_cost_rate * spot * turnover
        liquidation_cost = np.zeros(self._num_active_episode, dtype=np.float64)
        if is_terminal_step:
            liquidation_cost = self.hedge_cost_rate * next_spot * np.abs(actions)
        option_mark_change = next_discount * next_option - discount * option
        accounting_reward = (
            actions * discounted_gain - option_mark_change - discount * rebalance_cost
        )
        if is_terminal_step:
            accounting_reward -= next_discount * liquidation_cost
        shaped_accounting_reward = accounting_reward - self.reward_risk_aversion_xi * np.abs(
            accounting_reward
        )
        cash_flow_reward = (
            -discount * spot * (actions - self._holding)
            - discount * rebalance_cost
            + next_discount * actions * dividend_cash
        )
        if num_step == 0:
            cash_flow_reward += option
        if is_terminal_step:
            cash_flow_reward += next_discount * (
                next_spot * actions - next_option - liquidation_cost
            )
        self._accounting_wealth += accounting_reward
        self._cash_flow_wealth += cash_flow_reward
        discounted_transaction_cost = discount * rebalance_cost + next_discount * liquidation_cost
        self._discounted_transaction_cost += discounted_transaction_cost
        if self.reward_formulation == REWARD_FORMULATION_SHAPED_ACCOUNTING:
            unscaled_training_reward = shaped_accounting_reward
        else:
            unscaled_training_reward = cash_flow_reward
        training_reward = self.training_reward_scale * unscaled_training_reward
        if self.is_record_trajectory:
            self._record_transition(
                actions=actions,
                delta_time=delta_time,
                next_discount=next_discount,
                dividend_cash=dividend_cash,
                discounted_gain=discounted_gain,
                rebalance_cost=rebalance_cost,
                liquidation_cost=liquidation_cost,
                accounting_reward=accounting_reward,
                shaped_accounting_reward=shaped_accounting_reward,
                cash_flow_reward=cash_flow_reward,
                training_reward=training_reward,
                discounted_transaction_cost=discounted_transaction_cost,
            )
        previous_holding = self._holding.copy()
        self._discount_factor = next_discount
        self._num_step = num_next_step
        if is_terminal_step:
            self._holding = np.zeros_like(actions)
            reconciliation_error = np.abs(self._accounting_wealth - self._cash_flow_wealth)
            if np.any(reconciliation_error > self.reconciliation_tolerance):
                maximum_error = float(reconciliation_error.max())
                raise RewardReconciliationError(
                    f"Accounting and Cash Flow terminal wealth differ: max_error={maximum_error:.3e}, tolerance={self.reconciliation_tolerance:.3e}"
                )
            self._append_completed_results(reconciliation_error)
            next_observation = np.zeros(
                (self._num_active_episode, self.num_state_feature), dtype=self.observation_dtype
            )
            terminated = np.ones(self._num_active_episode, dtype=bool)
            self._is_done = True
            self._is_active = False
        else:
            self._holding = actions.copy()
            next_observation = self._get_observation(self._num_step, self._holding)
            terminated = np.zeros(self._num_active_episode, dtype=bool)
        truncated = np.zeros(self._num_active_episode, dtype=bool)
        info = self._get_step_info(
            actions=actions,
            previous_holding=previous_holding,
            delta_time=delta_time,
            discount=discount,
            next_discount=next_discount,
            dividend_cash=dividend_cash,
            discounted_gain=discounted_gain,
            rebalance_cost=rebalance_cost,
            liquidation_cost=liquidation_cost,
            accounting_reward=accounting_reward,
            shaped_accounting_reward=shaped_accounting_reward,
            cash_flow_reward=cash_flow_reward,
            training_reward=training_reward,
            discounted_transaction_cost=discounted_transaction_cost,
            is_terminal_step=is_terminal_step,
        )
        return (
            self._get_public_array(next_observation),
            self._get_public_array(training_reward.astype(self.observation_dtype, copy=False)),
            self._get_public_array(terminated),
            self._get_public_array(truncated),
            info,
        )

    def _record_transition(
        self,
        *,
        actions: np.ndarray,
        delta_time: np.ndarray,
        next_discount: np.ndarray,
        dividend_cash: np.ndarray,
        discounted_gain: np.ndarray,
        rebalance_cost: np.ndarray,
        liquidation_cost: np.ndarray,
        accounting_reward: np.ndarray,
        shaped_accounting_reward: np.ndarray,
        cash_flow_reward: np.ndarray,
        training_reward: np.ndarray,
        discounted_transaction_cost: np.ndarray,
    ) -> None:
        """Record transition."""
        num_step = self._num_step
        for num_episode in range(self._num_active_episode):
            self._trajectory_records.append(
                {
                    "draw_id": int(self._draw_ids[num_episode]),
                    "episode_index": int(self._episode_indices[num_episode]),
                    "episode_id": str(self._episode_ids[num_episode]),
                    "cohort_id": pd.Timestamp(self._cohort_ids[num_episode]),
                    "label": self.label,
                    "step": num_step,
                    "date": pd.Timestamp(self._dates[num_episode, num_step]),
                    "next_date": pd.Timestamp(self._dates[num_episode, num_step + 1]),
                    "delta_time": float(delta_time[num_episode]),
                    "discount_factor": float(self._discount_factor[num_episode]),
                    "next_discount_factor": float(next_discount[num_episode]),
                    "spot_norm": float(self._spot_norm[num_episode, num_step]),
                    "next_spot_norm": float(self._spot_norm[num_episode, num_step + 1]),
                    "option_norm": float(self._option_norm[num_episode, num_step]),
                    "next_option_norm": float(self._option_norm[num_episode, num_step + 1]),
                    "previous_holding_norm": float(self._holding[num_episode]),
                    "action": float(actions[num_episode]),
                    "dividend_cash": float(dividend_cash[num_episode]),
                    "discounted_gain": float(discounted_gain[num_episode]),
                    "rebalance_cost": float(rebalance_cost[num_episode]),
                    "liquidation_cost": float(liquidation_cost[num_episode]),
                    "accounting_reward": float(accounting_reward[num_episode]),
                    "shaped_accounting_reward": float(shaped_accounting_reward[num_episode]),
                    "cash_flow_reward": float(cash_flow_reward[num_episode]),
                    "training_reward_scale": self.training_reward_scale,
                    "training_reward": float(training_reward[num_episode]),
                    "discounted_transaction_cost": float(discounted_transaction_cost[num_episode]),
                    "is_terminal": num_step + 1 == self.num_interval,
                }
            )

    def _append_completed_results(self, reconciliation_error: np.ndarray) -> None:
        """Append completed results."""
        selected_wealth = (
            self._cash_flow_wealth
            if self.reward_formulation == REWARD_FORMULATION_CASH_FLOW
            else self._accounting_wealth
        )
        for num_episode in range(self._num_active_episode):
            wealth = float(selected_wealth[num_episode])
            net_loss = -wealth
            c0 = float(self._option_norm[num_episode, 0])
            monetary_scale = self.contract_multiplier * self._s0[num_episode]
            transaction_cost = float(self._discounted_transaction_cost[num_episode])
            self._completed_results.append(
                {
                    "draw_id": int(self._draw_ids[num_episode]),
                    "episode_index": int(self._episode_indices[num_episode]),
                    "episode_id": str(self._episode_ids[num_episode]),
                    "cohort_id": pd.Timestamp(self._cohort_ids[num_episode]),
                    "label": self.label,
                    "reward_formulation": self.reward_formulation,
                    "num_interval": self.num_interval,
                    "s0": float(self._s0[num_episode]),
                    "c0_norm": c0,
                    "accounting_wealth": float(self._accounting_wealth[num_episode]),
                    "cash_flow_wealth": float(self._cash_flow_wealth[num_episode]),
                    "reconciliation_error": float(reconciliation_error[num_episode]),
                    "wealth": wealth,
                    "net_loss": net_loss,
                    "replication_cost": c0 + net_loss,
                    "transaction_cost": transaction_cost,
                    "monetary_wealth": monetary_scale * wealth,
                    "monetary_loss": monetary_scale * net_loss,
                    "monetary_transaction_cost": monetary_scale * transaction_cost,
                }
            )

    def _get_step_info(
        self,
        *,
        actions: np.ndarray,
        previous_holding: np.ndarray,
        delta_time: np.ndarray,
        discount: np.ndarray,
        next_discount: np.ndarray,
        dividend_cash: np.ndarray,
        discounted_gain: np.ndarray,
        rebalance_cost: np.ndarray,
        liquidation_cost: np.ndarray,
        accounting_reward: np.ndarray,
        shaped_accounting_reward: np.ndarray,
        cash_flow_reward: np.ndarray,
        training_reward: np.ndarray,
        discounted_transaction_cost: np.ndarray,
        is_terminal_step: bool,
    ) -> dict[str, Any]:
        """Return step info."""
        info: dict[str, Any] = {
            "draw_id": self._draw_ids.copy(),
            "episode_id": self._episode_ids.copy(),
            "step": np.full(self._num_active_episode, self._num_step - 1, dtype=np.int64),
            "date": self._dates[:, self._num_step - 1].copy(),
            "next_date": self._dates[:, self._num_step].copy(),
            "delta_time": delta_time.copy(),
            "previous_holding_norm": previous_holding.copy(),
            "action": actions.copy(),
            "discount_factor": discount.copy(),
            "next_discount_factor": next_discount.copy(),
            "dividend_cash": dividend_cash.copy(),
            "discounted_gain": discounted_gain.copy(),
            "rebalance_cost": rebalance_cost.copy(),
            "liquidation_cost": liquidation_cost.copy(),
            "accounting_reward": accounting_reward.copy(),
            "shaped_accounting_reward": shaped_accounting_reward.copy(),
            "cash_flow_reward": cash_flow_reward.copy(),
            "training_reward_scale": self.training_reward_scale,
            "training_reward": training_reward.copy(),
            "discounted_transaction_cost": discounted_transaction_cost.copy(),
            "cumulative_discounted_transaction_cost": self._discounted_transaction_cost.copy(),
            "accounting_wealth": self._accounting_wealth.copy(),
            "cash_flow_wealth": self._cash_flow_wealth.copy(),
            "is_terminal_step": np.full(self._num_active_episode, is_terminal_step, dtype=bool),
        }
        if is_terminal_step:
            info["net_loss"] = -(
                self._accounting_wealth.copy()
                if self.reward_formulation != REWARD_FORMULATION_CASH_FLOW
                else self._cash_flow_wealth.copy()
            )
            info["reconciliation_error"] = np.abs(self._accounting_wealth - self._cash_flow_wealth)
        return self._get_public_info(info)

    def _get_public_array(self, value: np.ndarray) -> Any:
        """Return public array."""
        array = np.asarray(value)
        if self.is_batch_mode:
            return array
        result = array[0]
        return result.item() if np.asarray(result).ndim == 0 else result

    def _get_public_info(self, info: dict[str, Any]) -> dict[str, Any]:
        """Return public info."""
        if self.is_batch_mode:
            return info
        result: dict[str, Any] = {}
        for key, value in info.items():
            if isinstance(value, np.ndarray) and len(value) == self._num_active_episode:
                item = value[0]
                result[key] = item.item() if isinstance(item, np.generic) else item
            else:
                result[key] = value
        return result

    def get_bsm_call_delta(self) -> float | np.ndarray:
        """Return current dividend-adjusted BSM call deltas for the active decision states."""
        if not self._is_active or self._is_done:
            raise EnvironmentStateError("BSM delta requires a reset, nonterminal state")
        num_step = self._num_step
        spot = self._spot_norm[:, num_step]
        strike = self._strike_norm[:, num_step]
        tau = self._tau[:, num_step]
        volatility = self._iv[:, num_step]
        zero_rate = self._zero_rate[:, num_step]
        dividend_rate = self._dividend_rate[:, num_step]
        denominator = volatility * np.sqrt(tau)
        if np.any(denominator <= 0) or not np.isfinite(denominator).all():
            raise EnvironmentStateError("The current state has no finite BSM delta")
        d1 = (
            np.log(spot / strike) + (zero_rate - dividend_rate + 0.5 * volatility**2) * tau
        ) / denominator
        delta = np.exp(-dividend_rate * tau) * ndtr(d1)
        return self._get_public_array(np.clip(delta, 0.0, 1.0))

    def get_completed_episode_results(self) -> pd.DataFrame:
        """Return completed episode results."""
        columns = (
            "draw_id",
            "episode_index",
            "episode_id",
            "cohort_id",
            "label",
            "reward_formulation",
            "num_interval",
            "s0",
            "c0_norm",
            "accounting_wealth",
            "cash_flow_wealth",
            "reconciliation_error",
            "wealth",
            "net_loss",
            "replication_cost",
            "transaction_cost",
            "monetary_wealth",
            "monetary_loss",
            "monetary_transaction_cost",
        )
        return pd.DataFrame(self._completed_results).reindex(columns=columns).copy()

    def get_trajectory(self) -> pd.DataFrame:
        """Return trajectory."""
        return pd.DataFrame(self._trajectory_records).copy()

    def get_metrics(
        self, *, risk_aversion_lambda: float | None = None, is_require_complete: bool = True
    ) -> dict[str, float | int]:
        """Compute pooled contract-currency loss mean, population standard deviation, risk-adjusted loss, and mean discounted transaction cost over completed episodes."""
        if not isinstance(is_require_complete, (bool, np.bool_)):
            raise EnvironmentConfigurationError("is_require_complete must be a boolean")
        if bool(is_require_complete) and self.num_completed_episode != len(self._schedule):
            raise EnvironmentStateError(
                "The episode schedule is incomplete; final pooled metrics are unavailable"
            )
        results = self.get_completed_episode_results()
        if results.empty:
            raise EnvironmentStateError(
                "No episodes have completed; evaluation metrics are unavailable"
            )
        risk_lambda = (
            self.risk_aversion_lambda
            if risk_aversion_lambda is None
            else self._get_nonnegative_float(risk_aversion_lambda, "risk_aversion_lambda")
        )
        losses = results["monetary_loss"].to_numpy(dtype=np.float64)
        transaction_costs = results["monetary_transaction_cost"].to_numpy(dtype=np.float64)
        mean_loss = float(losses.mean())
        std_loss = float(losses.std(ddof=0))
        return {
            "num_episode": len(losses),
            "mean_loss": mean_loss,
            "std_loss": std_loss,
            "risk_aversion_lambda": risk_lambda,
            "j_lambda": mean_loss + risk_lambda * std_loss,
            "mean_transaction_cost": float(transaction_costs.mean()),
        }

    def close(self) -> None:
        """Release active resources while retaining completed result records."""
        self._is_active = False
        self._is_done = True
