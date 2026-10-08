"""Replay for the paper option-hedging pipeline."""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Any, Mapping
import numpy as np
import torch


@dataclass(frozen=True)
class EpisodeTransitions:
    """One complete episode and the metadata identifying its real or simulated source root."""

    episode_id: str
    cohort_id: str
    source_root: str
    is_simulated: bool
    anchor_cohort_id: str | None
    state: np.ndarray
    action: np.ndarray
    reward: np.ndarray
    next_state: np.ndarray
    terminated: np.ndarray
    delta_action: np.ndarray
    next_delta_action: np.ndarray

    @property
    def num_transition(self) -> int:
        """Return the number of transition."""
        return int(self.state.shape[0])

    def get_context(self, num_prefix: int) -> np.ndarray:
        """Return the transition prefix used as task-inference context."""
        if not 0 <= int(num_prefix) <= self.num_transition:
            raise ValueError("num_prefix exceeds the episode transition count")
        num_prefix = int(num_prefix)
        return np.concatenate(
            (
                self.state[:num_prefix],
                self.action[:num_prefix],
                self.reward[:num_prefix],
                self.next_state[:num_prefix],
            ),
            axis=1,
        ).astype(np.float32, copy=False)


class TaskReplayBuffer:
    """Store complete episodes and sample context/query batches with disjoint underlying source roots."""

    def __init__(
        self, *, segment_id: int, num_state_feature: int = 7, seed: int | None = None
    ) -> None:
        """Initialize validated configuration and internal state."""
        if isinstance(segment_id, bool) or int(segment_id) <= 0:
            raise ValueError("segment_id must be a strictly positive integer")
        if isinstance(num_state_feature, bool) or int(num_state_feature) <= 0:
            raise ValueError("num_state_feature must be a strictly positive integer")
        self.segment_id = int(segment_id)
        self.num_state_feature = int(num_state_feature)
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self._episodes: dict[str, EpisodeTransitions] = {}
        self._episode_order: list[EpisodeTransitions] = []
        self._num_transition = 0
        self._episodes_by_source_cohort: dict[bool, dict[str, list[EpisodeTransitions]]] = {
            False: {},
            True: {},
        }
        self._source_root_counts: dict[str, int] = {}
        self._source_root_counts_by_source: dict[bool, dict[str, int]] = {False: {}, True: {}}
        self._num_episode_by_source = {False: 0, True: 0}
        self._source_group_cache: dict[bool, tuple[tuple[EpisodeTransitions, ...], ...]] = {}
        self._recent_source_group_cache: dict[
            tuple[bool, int], tuple[tuple[EpisodeTransitions, ...], ...]
        ] = {}

    def __len__(self) -> int:
        """Len."""
        return len(self._episodes)

    @property
    def num_episode(self) -> int:
        """Return the number of episode."""
        return len(self)

    @property
    def num_transition(self) -> int:
        """Return the number of transition."""
        return self._num_transition

    def _rebuild_sampling_indexes(self) -> None:
        """Rebuild sampling indexes."""
        self._num_transition = 0
        self._episodes_by_source_cohort = {False: {}, True: {}}
        self._source_root_counts = {}
        self._source_root_counts_by_source = {False: {}, True: {}}
        self._num_episode_by_source = {False: 0, True: 0}
        self._source_group_cache = {}
        self._recent_source_group_cache = {}
        self._episode_order = []
        for item in self._episodes.values():
            self._episode_order.append(item)
            self._num_transition += item.num_transition
            self._episodes_by_source_cohort[item.is_simulated].setdefault(
                item.cohort_id, []
            ).append(item)
            self._source_root_counts[item.source_root] = (
                self._source_root_counts.get(item.source_root, 0) + 1
            )
            source_counts = self._source_root_counts_by_source[item.is_simulated]
            source_counts[item.source_root] = source_counts.get(item.source_root, 0) + 1
            self._num_episode_by_source[item.is_simulated] += 1

    def _index_episode(self, item: EpisodeTransitions) -> None:
        """Index episode."""
        self._episode_order.append(item)
        self._num_transition += item.num_transition
        self._episodes_by_source_cohort[item.is_simulated].setdefault(item.cohort_id, []).append(
            item
        )
        self._source_root_counts[item.source_root] = (
            self._source_root_counts.get(item.source_root, 0) + 1
        )
        source_counts = self._source_root_counts_by_source[item.is_simulated]
        source_counts[item.source_root] = source_counts.get(item.source_root, 0) + 1
        self._num_episode_by_source[item.is_simulated] += 1
        self._source_group_cache.pop(item.is_simulated, None)
        self._recent_source_group_cache.clear()

    @staticmethod
    def _get_matrix(value: Any, *, name: str, num_row: int | None, num_column: int) -> np.ndarray:
        """Return matrix."""
        array = np.asarray(value, dtype=np.float32)
        if array.ndim == 1 and num_column == 1:
            array = array.reshape(-1, 1)
        expected = (num_row, num_column) if num_row is not None else None
        if array.ndim != 2 or array.shape[1] != num_column:
            raise ValueError(f"{name} must be (*,{num_column}) two-dimensional array")
        if expected is not None and array.shape != expected:
            raise ValueError(f"{name} shape must be {expected}; received {array.shape}")
        if not np.isfinite(array).all():
            raise ValueError(f"{name} contains nonfinite values")
        return np.ascontiguousarray(array)

    def add_episode(
        self,
        *,
        episode_id: str,
        cohort_id: Any,
        is_simulated: bool,
        state: Any,
        action: Any,
        reward: Any,
        next_state: Any,
        terminated: Any,
        delta_action: Any,
        next_delta_action: Any,
        anchor_cohort_id: Any | None = None,
    ) -> None:
        """Add episode."""
        episode_id = str(episode_id)
        if not episode_id:
            raise ValueError("episode_id must not be empty")
        if episode_id in self._episodes:
            raise ValueError(f"episode_id already exists in the task replay: {episode_id}")
        if not isinstance(is_simulated, (bool, np.bool_)):
            raise ValueError("is_simulated must be a boolean")
        cohort_id = str(cohort_id)
        anchor = None if anchor_cohort_id is None else str(anchor_cohort_id)
        if bool(is_simulated) and (not anchor):
            raise ValueError("Simulated episodes must provide anchor_cohort_id")
        source_root = anchor if bool(is_simulated) else cohort_id
        assert source_root is not None
        state_array = self._get_matrix(
            state, name="state", num_row=None, num_column=self.num_state_feature
        )
        num_row = int(state_array.shape[0])
        if num_row == 0:
            raise ValueError("A complete episode must contain at least one transition")
        arrays = {
            "action": self._get_matrix(action, name="action", num_row=num_row, num_column=1),
            "reward": self._get_matrix(reward, name="reward", num_row=num_row, num_column=1),
            "next_state": self._get_matrix(
                next_state, name="next_state", num_row=num_row, num_column=self.num_state_feature
            ),
            "terminated": self._get_matrix(
                terminated, name="terminated", num_row=num_row, num_column=1
            ),
            "delta_action": self._get_matrix(
                delta_action, name="delta_action", num_row=num_row, num_column=1
            ),
            "next_delta_action": self._get_matrix(
                next_delta_action, name="next_delta_action", num_row=num_row, num_column=1
            ),
        }
        if np.any((arrays["action"] < 0.0) | (arrays["action"] > 1.0)):
            raise ValueError("action must be in [0,1]")
        for name in ("delta_action", "next_delta_action"):
            if np.any((arrays[name] < 0.0) | (arrays[name] > 1.0)):
                raise ValueError(f"{name} must be in [0,1]")
        terminal_values = arrays["terminated"].reshape(-1)
        if not np.isin(terminal_values, (0.0, 1.0)).all():
            raise ValueError("terminated must contain only 0/1")
        if terminal_values[-1] != 1.0 or terminal_values[:-1].any():
            raise ValueError("A complete episode must terminate only on its last transition")
        item = EpisodeTransitions(
            episode_id=episode_id,
            cohort_id=cohort_id,
            source_root=source_root,
            is_simulated=bool(is_simulated),
            anchor_cohort_id=anchor,
            state=state_array,
            **arrays,
        )
        self._episodes[episode_id] = item
        self._index_episode(item)

    def get_episode(self, episode_id: str) -> EpisodeTransitions:
        """Return episode."""
        try:
            return self._episodes[str(episode_id)]
        except KeyError as exc:
            raise KeyError(f"Unknown episode_id: {episode_id}") from exc

    def get_episode_ids(self) -> tuple[str, ...]:
        """Return episode ids."""
        return tuple(self._episodes)

    def _get_candidates(
        self, *, is_simulated: bool | None = None, excluded_source_root: str | None = None
    ) -> list[EpisodeTransitions]:
        """Return candidates."""
        if is_simulated is not None:
            groups = self._get_candidate_groups(is_simulated=bool(is_simulated))
            return [
                item
                for group in groups
                for item in group
                if excluded_source_root is None or item.source_root != excluded_source_root
            ]
        return [
            item
            for item in self._episodes.values()
            if excluded_source_root is None or item.source_root != excluded_source_root
        ]

    def _get_candidate_groups(
        self, *, is_simulated: bool
    ) -> tuple[tuple[EpisodeTransitions, ...], ...]:
        """Return candidate groups."""
        source = bool(is_simulated)
        cached = self._source_group_cache.get(source)
        if cached is not None:
            return cached
        result = tuple(
            (
                tuple(episodes)
                for episodes in self._episodes_by_source_cohort[source].values()
                if episodes
            )
        )
        self._source_group_cache[source] = result
        return result

    def _get_recent_candidate_groups(
        self, *, is_simulated: bool, num_recent_episode: int
    ) -> tuple[tuple[EpisodeTransitions, ...], ...]:
        """Return recent candidate groups."""
        source = bool(is_simulated)
        window = int(num_recent_episode)
        cache_key = (source, window)
        cached = self._recent_source_group_cache.get(cache_key)
        if cached is not None:
            return cached
        by_cohort: dict[str, list[EpisodeTransitions]] = {}
        for item in self._episode_order[-window:]:
            if item.is_simulated == source:
                by_cohort.setdefault(item.cohort_id, []).append(item)
        result = tuple((tuple(items) for items in by_cohort.values() if items))
        self._recent_source_group_cache[cache_key] = result
        return result

    def _has_candidate_excluding_roots(
        self, *, is_simulated: bool, excluded_source_roots: frozenset[str]
    ) -> bool:
        """Has candidate excluding roots."""
        source_counts = self._source_root_counts_by_source[bool(is_simulated)]
        num_excluded = sum(
            (source_counts.get(source_root, 0) for source_root in excluded_source_roots)
        )
        return self._num_episode_by_source[bool(is_simulated)] > num_excluded

    def can_sample(self, *, num_min_transition: int = 128) -> bool:
        """Can sample."""
        if self._num_transition < int(num_min_transition) or not self._episodes:
            return False
        return len(self._source_root_counts) >= 2

    def _sample_episode_by_cohort(
        self,
        groups: tuple[tuple[EpisodeTransitions, ...], ...],
        *,
        excluded_source_roots: frozenset[str] = frozenset(),
    ) -> EpisodeTransitions:
        """Sample episode by cohort."""
        if not groups:
            raise ValueError("No candidate episodes are available")
        while True:
            episodes = groups[int(self._rng.integers(0, len(groups)))]
            if not excluded_source_roots:
                eligible = episodes
            else:
                eligible = tuple(
                    (item for item in episodes if item.source_root not in excluded_source_roots)
                )
                if not eligible:
                    continue
            return eligible[int(self._rng.integers(0, len(eligible)))]

    def _sample_transition_rows(
        self,
        groups: tuple[tuple[EpisodeTransitions, ...], ...],
        num_sample: int,
        *,
        excluded_source_roots: frozenset[str],
    ) -> list[tuple[EpisodeTransitions, int]]:
        """Sample transition rows."""
        rows: list[tuple[EpisodeTransitions, int]] = []
        for _ in range(int(num_sample)):
            item = self._sample_episode_by_cohort(
                groups, excluded_source_roots=excluded_source_roots
            )
            step = int(self._rng.integers(0, item.num_transition))
            rows.append((item, step))
        return rows

    def sample_context_and_batch(
        self,
        *,
        num_batch: int = 64,
        num_recent_context_episodes: int = 4,
        simulated_batch_fraction: float = 0.25,
        device: str | torch.device | None = None,
    ) -> dict[str, Any]:
        """Sample a recent-episode context prefix and RL transitions from disjoint real/anchor cohort roots."""
        if isinstance(num_batch, bool) or int(num_batch) <= 0:
            raise ValueError("num_batch must be a strictly positive integer")
        if (
            isinstance(num_recent_context_episodes, bool)
            or int(num_recent_context_episodes) != num_recent_context_episodes
            or int(num_recent_context_episodes) <= 0
        ):
            raise ValueError("num_recent_context_episodes must be a strictly positive integer")
        if (
            isinstance(simulated_batch_fraction, (bool, np.bool_))
            or not np.isfinite(float(simulated_batch_fraction))
            or (not 0.0 <= float(simulated_batch_fraction) <= 1.0)
        ):
            raise ValueError("simulated_batch_fraction must be in [0,1]")
        num_recent_context_episodes = int(num_recent_context_episodes)
        simulated_batch_fraction = float(simulated_batch_fraction)
        if not self.can_sample(num_min_transition=1):
            raise ValueError("The task replay cannot yet separate context and RL source roots")
        target_device = torch.device(device if device is not None else "cpu")
        context_sample = self.sample_recent_context(
            num_recent_context_episodes=num_recent_context_episodes, device=target_device
        )
        context_roots = frozenset(context_sample["context_source_roots"])
        groups_by_source = {
            source: self._get_candidate_groups(is_simulated=source) for source in (False, True)
        }
        usable_sources = [
            source
            for source in (False, True)
            if self._has_candidate_excluding_roots(
                is_simulated=source, excluded_source_roots=context_roots
            )
        ]
        if not usable_sources:
            raise RuntimeError("No RL transitions remain after excluding context source roots")
        if len(usable_sources) == 2:
            num_simulated = int(math.floor(int(num_batch) * simulated_batch_fraction + 0.5))
            num_simulated = min(max(num_simulated, 0), int(num_batch))
            num_real = int(num_batch) - num_simulated
        elif usable_sources[0] is False:
            num_real, num_simulated = (int(num_batch), 0)
        else:
            num_real, num_simulated = (0, int(num_batch))
        sampled = self._sample_transition_rows(
            groups_by_source[False], num_real, excluded_source_roots=context_roots
        ) + self._sample_transition_rows(
            groups_by_source[True], num_simulated, excluded_source_roots=context_roots
        )
        self._rng.shuffle(sampled)
        num_state = self.num_state_feature
        row_width = 2 * num_state + 5
        row_array = np.empty((len(sampled), row_width), dtype=np.float32)
        for row_index, (item, step) in enumerate(sampled):
            row_array[row_index] = np.concatenate(
                (
                    item.state[step],
                    item.action[step],
                    item.reward[step],
                    item.next_state[step],
                    item.terminated[step],
                    item.delta_action[step],
                    item.next_delta_action[step],
                )
            )
        batch_tensor = torch.as_tensor(row_array, device=target_device)
        state_end = num_state
        action_end = state_end + 1
        reward_end = action_end + 1
        next_state_end = reward_end + num_state
        terminated_end = next_state_end + 1
        delta_end = terminated_end + 1
        result: dict[str, Any] = {
            "segment_id": self.segment_id,
            **context_sample,
            "simulated_batch_fraction": simulated_batch_fraction,
            "num_real_batch": num_real,
            "num_simulated_batch": num_simulated,
            "state": batch_tensor[:, :state_end],
            "action": batch_tensor[:, state_end:action_end],
            "reward": batch_tensor[:, action_end:reward_end],
            "next_state": batch_tensor[:, reward_end:next_state_end],
            "terminated": batch_tensor[:, next_state_end:terminated_end],
            "delta_action": batch_tensor[:, terminated_end:delta_end],
            "next_delta_action": batch_tensor[:, delta_end:],
        }
        result["batch_source_root"] = [item.source_root for item, _ in sampled]
        if context_roots.intersection(result["batch_source_root"]):
            raise RuntimeError("Internal error: context and RL minibatch source roots overlap")
        return result

    def sample_recent_context(
        self, *, num_recent_context_episodes: int = 4, device: str | torch.device | None = None
    ) -> dict[str, Any]:
        """Sample recent context."""
        if (
            isinstance(num_recent_context_episodes, bool)
            or int(num_recent_context_episodes) != num_recent_context_episodes
            or int(num_recent_context_episodes) <= 0
        ):
            raise ValueError("num_recent_context_episodes must be a strictly positive integer")
        if not self._episodes:
            raise ValueError("Replay contains no episodes available for task inference")
        window = int(num_recent_context_episodes)
        target_device = torch.device(device if device is not None else "cpu")
        groups_by_source = {
            source: self._get_recent_candidate_groups(
                is_simulated=source, num_recent_episode=window
            )
            for source in (False, True)
        }
        sources = [source for source in (False, True) if groups_by_source[source]]
        if not sources:
            raise RuntimeError("Replay contains no episodes available for task inference")
        source = sources[int(self._rng.integers(0, len(sources)))]
        episode = self._sample_episode_by_cohort(groups_by_source[source])
        num_context = int(self._rng.integers(0, episode.num_transition))
        context = np.ascontiguousarray(episode.get_context(num_context), dtype=np.float32)
        return {
            "context": torch.as_tensor(context, device=target_device),
            "context_episode_id": episode.episode_id,
            "context_cohort_id": episode.cohort_id,
            "context_source_root": episode.source_root,
            "context_source_roots": (episode.source_root,),
            "context_is_simulated": episode.is_simulated,
            "num_context": int(context.shape[0]),
            "num_recent_context_episodes": window,
            "num_context_candidate_episode": sum(
                (len(group) for groups in groups_by_source.values() for group in groups)
            ),
        }

    def get_usage_summary(self) -> dict[str, Any]:
        """Return usage summary."""
        episodes = list(self._episodes.values())
        return {
            "segment_id": self.segment_id,
            "num_episode": len(episodes),
            "num_transition": self.num_transition,
            "num_real_episode": sum((not item.is_simulated for item in episodes)),
            "num_simulated_episode": sum((item.is_simulated for item in episodes)),
            "num_real_cohort": len({item.cohort_id for item in episodes if not item.is_simulated}),
            "num_simulated_cohort": len({item.cohort_id for item in episodes if item.is_simulated}),
            "num_source_root": len({item.source_root for item in episodes}),
            "seed": self.seed,
        }

    def state_dict(self) -> dict[str, Any]:
        """State dict."""
        return {
            "segment_id": self.segment_id,
            "num_state_feature": self.num_state_feature,
            "seed": self.seed,
            "rng_state": self._rng.bit_generator.state,
            "episodes": self._episodes,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Load state dict."""
        if int(state["segment_id"]) != self.segment_id:
            raise ValueError("TaskReplayBuffer segment_id differs from the checkpoint")
        if int(state["num_state_feature"]) != self.num_state_feature:
            raise ValueError("TaskReplayBuffer state dimension differs from the checkpoint")
        episodes = state["episodes"]
        if not isinstance(episodes, dict) or not all(
            (isinstance(item, EpisodeTransitions) for item in episodes.values())
        ):
            raise ValueError("TaskReplayBuffer checkpoint episodes are invalid")
        self._episodes = dict(episodes)
        self._rebuild_sampling_indexes()
        self._rng.bit_generator.state = state["rng_state"]
