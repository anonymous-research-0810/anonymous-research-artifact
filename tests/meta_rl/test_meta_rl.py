"""Test meta rl for the paper option-hedging pipeline."""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
import test_meta
import train_meta
from tests.rl_envs.helpers import get_synthetic_dataset
from option_dataset import OptionDataset
from hedging_env import HedgingEnv
from hedging_meta_rl import (
    MetaTaskDataset,
    PearlSACHedgingAgent,
    PearlTD3HedgingAgent,
    ProbabilisticContextEncoder,
    TaskReplayBuffer,
    evaluate_meta_hedging_agent,
    train_meta_hedging_agent,
)
from train_meta import get_default_meta_config, get_meta_run_specs


def test_meta_training_wrapper_keeps_only_validation_best_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify meta training wrapper keeps only validation best model."""
    train_dataset = get_synthetic_dataset("train", num_episode=2)
    valid_dataset = get_synthetic_dataset("valid", num_episode=2)
    config = get_default_meta_config()
    config["agent"]["algorithm_names"] = ["pearl_td3"]
    config["agent"]["is_delta_residual_values"] = [False]
    config["environment"]["reward_formulations"] = ["shaped_accounting"]
    config["training"]["seeds"] = [13]
    config["training"]["output_root"] = str(tmp_path / "meta_results")
    run_spec = get_meta_run_specs(config)[0]

    class FakeTask:
        segment_id = 1
        real_dataset = train_dataset
        simulated_dataset = train_dataset

        def get_record(self):
            return {"segment_id": self.segment_id}

    class FakeAgent:

        def get_config(self):
            return {"algorithm_name": "pearl_td3"}

    def fake_training(agent, tasks, **kwargs):
        model_path = Path(kwargs["checkpoint_path"])
        checkpoint_dir = Path(kwargs["checkpoint_dir"])
        model_path.write_bytes(b"validation-best")
        for name in ("agent_episode_00000001.pt", "best_agent.pt", "training_final_agent.pt"):
            (checkpoint_dir / name).write_bytes(name.encode("ascii"))
        return {
            "episode_results": pd.DataFrame({"episode_id": ["train_1"]}),
            "update_history": pd.DataFrame({"num_meta_update": [1]}),
            "validation_history": pd.DataFrame(
                {
                    "num_train_episode": [1],
                    "is_best": [True],
                    "checkpoint_path": [str(checkpoint_dir / "agent_episode_00000001.pt")],
                }
            ),
            "latent_episode_statistics": pd.DataFrame(),
            "trajectory": pd.DataFrame(),
            "episode_usage_audit": [],
            "metrics": {"mean_loss": 1.0},
            "best_validation": {"j_lambda": 1.0},
            "num_episode": 1,
            "num_total_step": 1,
            "num_meta_update": 1,
            "task_environment_configs": [],
        }

    monkeypatch.setattr(
        train_meta,
        "resolve_simulation_result_directory",
        lambda *args, **kwargs: tmp_path / "sim_result",
    )
    monkeypatch.setattr(
        train_meta,
        "load_simulation_result",
        lambda *args, **kwargs: {
            "experiment_id": "sim_test",
            "dataset": {"config": train_dataset.config.get_dict()},
        },
    )
    monkeypatch.setattr(
        train_meta,
        "load_real_dataset_from_simulation_result",
        lambda *args, **kwargs: train_dataset,
    )
    monkeypatch.setattr(train_meta, "load_meta_task_datasets", lambda *args, **kwargs: [FakeTask()])
    monkeypatch.setattr(
        train_meta, "_get_dataset_from_source_config", lambda *args, **kwargs: valid_dataset
    )
    monkeypatch.setattr(train_meta, "_get_normalizer", lambda *args: None)
    monkeypatch.setattr(train_meta, "get_meta_agent", lambda *args: FakeAgent())
    monkeypatch.setattr(train_meta, "train_meta_hedging_agent", fake_training)
    output = train_meta.run_meta_training(config, run_specs=[run_spec], experiment_id="meta_unit")
    run_dir = output["experiment_dir"] / run_spec["run_id"]
    record = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    assert (run_dir / "agent.pt").read_bytes() == b"validation-best"
    assert not (run_dir / "checkpoints").exists()
    assert list(run_dir.rglob("*.pt")) == [run_dir / "agent.pt"]
    retention = record["artifacts"]["checkpoint_retention"]
    assert retention["checkpoint_directory_removed"] is True
    assert retention["num_removed_model_file"] == 3
    history = pd.read_parquet(run_dir / "validation_history.parquet")
    assert history["checkpoint_path"].isna().all()
    discovered = test_meta.get_completed_meta_train_runs(
        config["training"]["output_root"], experiment_id="meta_unit"
    )
    assert len(discovered) == 1


def _renamed_dataset(source: OptionDataset, *, prefix: str, is_simulated: bool) -> OptionDataset:
    """Renamed dataset."""
    steps = source.episode_steps.copy(deep=True)
    manifest = source.episode_manifest.copy(deep=True)
    mapping = {episode_id: f"{prefix}_{episode_id}" for episode_id in source.get_episode_ids()}
    steps["episode_id"] = steps["episode_id"].map(mapping)
    manifest["episode_id"] = manifest["episode_id"].map(mapping)
    offset = int(prefix.split("_")[-1]) * 30
    steps["cohort_id"] = pd.to_datetime(steps["cohort_id"]) + pd.Timedelta(days=offset)
    manifest["cohort_id"] = pd.to_datetime(manifest["cohort_id"]) + pd.Timedelta(days=offset)
    if is_simulated:
        anchors = {
            episode_id: f"anchor_{prefix}_{index}"
            for index, episode_id in enumerate(mapping.values())
        }
        steps["is_simulated"] = True
        manifest["is_simulated"] = True
        steps["anchor_cohort_id"] = steps["episode_id"].map(anchors)
        manifest["anchor_cohort_id"] = manifest["episode_id"].map(anchors)
    return OptionDataset(
        label="train",
        config=source.config,
        episode_steps=steps,
        episode_manifest=manifest,
        candidate_selection_audit=source.candidate_selection_audit.copy(deep=True),
        dataset_build_report={"source": "meta_unit_test"},
    )


def _get_tasks(num_task: int = 3) -> list[MetaTaskDataset]:
    """Return tasks."""
    base = get_synthetic_dataset("train", num_episode=2)
    tasks = []
    for segment_id in range(1, num_task + 1):
        tasks.append(
            MetaTaskDataset(
                segment_id=segment_id,
                date_start=pd.Timestamp("2024-01-01"),
                date_end=pd.Timestamp("2025-12-31"),
                real_dataset=_renamed_dataset(
                    base, prefix=f"real_{segment_id}", is_simulated=False
                ),
                simulated_dataset=_renamed_dataset(
                    base, prefix=f"sim_{segment_id}", is_simulated=True
                ),
            )
        )
    return tasks


def _add_random_episode(
    replay: TaskReplayBuffer, *, episode_id: str, root_index: int, is_simulated: bool
) -> None:
    """Add random episode."""
    rng = np.random.default_rng(root_index + 100 * int(is_simulated))
    num_step = 5
    terminated = np.zeros((num_step, 1), dtype=np.float32)
    terminated[-1] = 1.0
    replay.add_episode(
        episode_id=episode_id,
        cohort_id=f"cohort_{is_simulated}_{root_index}",
        is_simulated=is_simulated,
        anchor_cohort_id=f"anchor_{root_index}" if is_simulated else None,
        state=rng.normal(size=(num_step, 7)).astype(np.float32),
        action=rng.uniform(size=(num_step, 1)).astype(np.float32),
        reward=rng.normal(size=(num_step, 1)).astype(np.float32),
        next_state=rng.normal(size=(num_step, 7)).astype(np.float32),
        terminated=terminated,
        delta_action=rng.uniform(size=(num_step, 1)).astype(np.float32),
        next_delta_action=rng.uniform(size=(num_step, 1)).astype(np.float32),
    )


def test_context_encoder_prior_and_product_shapes() -> None:
    """Verify context encoder prior and product shapes."""
    encoder = ProbabilisticContextEncoder(latent_dimension=4)
    empty = torch.empty((0, 16))
    mean, variance = encoder(empty)
    assert torch.equal(mean, torch.zeros((1, 4)))
    assert torch.equal(variance, torch.ones((1, 4)))
    mean, variance = encoder(torch.randn(7, 16))
    assert mean.shape == variance.shape == (1, 4)
    assert torch.isfinite(mean).all()
    assert torch.all(variance > 0)


def test_context_variance_uses_smooth_positive_parameterization() -> None:
    """Verify context variance uses smooth positive parameterization."""
    encoder = ProbabilisticContextEncoder(
        latent_dimension=4, hidden_dims=(8,), minimum_variance=1e-06
    )
    context = torch.randn(2, 7, 16)
    full_mean, full_variance = encoder(context)
    precision = torch.zeros((2, 4))
    weighted_mean = torch.zeros((2, 4))
    for index in range(context.shape[1]):
        increment = encoder.get_sufficient_statistics(context[:, index : index + 1, :])
        precision = precision + increment[0]
        weighted_mean = weighted_mean + increment[1]
    recursive_mean, recursive_variance = encoder.get_posterior_from_sufficient_statistics(
        precision, weighted_mean
    )
    assert torch.allclose(full_mean, recursive_mean, atol=1e-06)
    assert torch.allclose(full_variance, recursive_variance, atol=1e-06)
    loss = full_mean.square().mean() + full_variance.mean()
    loss.backward()
    assert all(
        (
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in encoder.parameters()
        )
    )


def test_default_experiment_matrix_has_eight_configurations_per_seed() -> None:
    """Verify eight configurations for each of ten paper seeds."""
    specs = get_meta_run_specs(get_default_meta_config())
    assert len(specs) == 80
    assert len({item["run_id"] for item in specs}) == 80
    assert {item["algorithm_name"] for item in specs} == {"pearl_td3", "pearl_sac"}
    assert {item["is_delta_residual"] for item in specs} == {True, False}
    assert {item["reward_formulation"] for item in specs} == {"shaped_accounting", "cash_flow"}
    assert {item["reward_formulation"]: item["training_reward_scale"] for item in specs} == {
        "shaped_accounting": 100.0,
        "cash_flow": 1.0,
    }
    config = get_default_meta_config()
    assert config["agent"]["encoder_learning_rate"] == pytest.approx(0.0001)
    assert config["agent"]["kl_coefficient"] == pytest.approx(0.001)
    assert config["training"]["num_task_batch"] == 8
    assert config["training"]["num_batch"] == 64
    assert config["training"]["num_recent_context_episodes"] == 4
    assert config["training"]["simulated_batch_fraction"] == pytest.approx(0.5)
    assert config["training"]["warmup_fraction"] == pytest.approx(0.1)
    assert "kl_annealing_start_fraction" not in config["training"]
    assert "kl_annealing_end_fraction" not in config["training"]
    assert config["training"]["num_valid_episodes"] == 200


@pytest.mark.parametrize(
    ("agent_class", "is_expect_alpha"),
    [(PearlTD3HedgingAgent, False), (PearlSACHedgingAgent, True)],
)
def test_meta_progress_and_evaluation_output_are_aligned_with_basis(
    agent_class: type[PearlTD3HedgingAgent] | type[PearlSACHedgingAgent],
    is_expect_alpha: bool,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify meta progress and evaluation output are aligned with basis."""
    tasks = _get_tasks(num_task=2)
    agent = agent_class(
        actor_hidden_dims=(8,),
        critic_hidden_dims=(8,),
        context_hidden_dims=(8,),
        is_delta_residual=True,
        device="cpu",
        seed=41,
    )

    def get_train_env(dataset: OptionDataset, seed: int) -> HedgingEnv:
        """Return train env."""
        return HedgingEnv(
            dataset,
            reward_formulation="cash_flow",
            hedge_cost_rate=0.0,
            is_record_trajectory=False,
            seed=seed,
        )

    train_meta_hedging_agent(
        agent,
        tasks,
        get_train_env=get_train_env,
        num_batch=2,
        num_task_batch=2,
        warmup_fraction=0.05,
        num_update_per_step=1,
        seed=41,
    )
    captured_train = capsys.readouterr()
    assert "[Training started]" in captured_train.out
    assert "tasks=2" in captured_train.out
    assert "real episode samples=4" in captured_train.out
    assert "simulated episode samples=4" in captured_train.out
    assert "total training steps=16" in captured_train.out
    assert "warm-up episodes=4" in captured_train.out
    assert "Training PEARL_" in captured_train.err
    assert "actor=" in captured_train.err
    assert "critic=" in captured_train.err
    assert "encoder=" in captured_train.err
    assert "kl=" in captured_train.err
    assert ("alpha=" in captured_train.err) is is_expect_alpha
    assert "100%" in captured_train.err
    valid_env = HedgingEnv(
        get_synthetic_dataset("valid", num_episode=2),
        reward_formulation="cash_flow",
        hedge_cost_rate=0.0,
        num_parallel_episode=1,
        is_record_trajectory=False,
    )
    evaluate_meta_hedging_agent(agent, valid_env)
    captured_evaluation = capsys.readouterr()
    assert "[Evaluation started]" in captured_evaluation.out
    assert "split=valid, episode samples=2" in captured_evaluation.out
    assert "posterior=recent_completed_episode" in captured_evaluation.out
    assert "Mean(L)=" in captured_evaluation.out
    assert "Std(L)=" in captured_evaluation.out
    assert "J_lambda=" in captured_evaluation.out
    assert "Mean(TC)=" in captured_evaluation.out


def test_task_replay_balances_sources_and_separates_context_root() -> None:
    """Verify task replay balances sources and separates context root."""
    replay = TaskReplayBuffer(segment_id=1, seed=17)
    for is_simulated in (False, True):
        for index in range(5):
            _add_random_episode(
                replay,
                episode_id=f"episode_{is_simulated}_{index}",
                root_index=index,
                is_simulated=is_simulated,
            )
    batch = replay.sample_context_and_batch(num_batch=64)
    assert batch["num_real_batch"] == 48
    assert batch["num_simulated_batch"] == 16
    assert isinstance(batch["context_is_simulated"], bool)
    assert batch["context"].shape[1] == 16
    assert 0 <= batch["num_context"] < 5
    assert len(batch["context_source_roots"]) == 1
    assert not set(batch["context_source_roots"]).intersection(batch["batch_source_root"])


def test_task_replay_context_uses_only_recent_completed_episodes() -> None:
    """Verify task replay context uses only recent completed episodes."""
    replay = TaskReplayBuffer(segment_id=1, seed=23)
    for is_simulated in (False, True):
        for index in range(5):
            _add_random_episode(
                replay,
                episode_id=f"recent_{is_simulated}_{index}",
                root_index=index,
                is_simulated=is_simulated,
            )
    recent_ids = {f"recent_True_{index}" for index in range(2, 5)}
    for _ in range(20):
        batch = replay.sample_context_and_batch(num_batch=8, num_recent_context_episodes=3)
        assert batch["context_episode_id"] in recent_ids
        assert batch["num_context_candidate_episode"] == 3
        assert batch["num_recent_context_episodes"] == 3
        assert len(batch["batch_source_root"]) == 8


def test_task_replay_rebuilds_sampling_indexes_after_restore() -> None:
    """Verify task replay rebuilds sampling indexes after restore."""
    replay = TaskReplayBuffer(segment_id=1, seed=19)
    for is_simulated in (False, True):
        for index in range(4):
            _add_random_episode(
                replay,
                episode_id=f"restore_{is_simulated}_{index}",
                root_index=index,
                is_simulated=is_simulated,
            )
    state = replay.state_dict()
    restored = TaskReplayBuffer(segment_id=1, seed=99)
    restored.load_state_dict(state)
    assert restored.num_episode == replay.num_episode
    assert restored.num_transition == replay.num_transition
    batch = restored.sample_context_and_batch(num_batch=12, num_recent_context_episodes=3)
    assert batch["num_real_batch"] == 9
    assert batch["num_simulated_batch"] == 3
    assert batch["context_episode_id"] in {f"restore_True_{index}" for index in range(1, 4)}
    assert not set(batch["context_source_roots"]).intersection(batch["batch_source_root"])


@pytest.mark.parametrize("agent_class", [PearlTD3HedgingAgent, PearlSACHedgingAgent])
def test_both_pearl_agents_update_and_checkpoint(
    agent_class: type[PearlTD3HedgingAgent] | type[PearlSACHedgingAgent], tmp_path: Path
) -> None:
    """Verify both pearl agents update and checkpoint."""
    replays = []
    for segment_id in (1, 2):
        replay = TaskReplayBuffer(segment_id=segment_id, seed=segment_id)
        for is_simulated in (False, True):
            for index in range(3):
                _add_random_episode(
                    replay,
                    episode_id=f"{segment_id}_{is_simulated}_{index}",
                    root_index=index,
                    is_simulated=is_simulated,
                )
        replays.append(replay)
    agent = agent_class(is_delta_residual=True, device="cpu", seed=3)
    metrics = agent.update(
        [replay.sample_context_and_batch(num_batch=8) for replay in replays], kl_coefficient=0.003
    )
    assert np.isfinite(metrics["critic_loss"])
    assert metrics["kl_coefficient"] == pytest.approx(0.003)
    assert len(metrics["latent_records"]) == 2
    checkpoint = tmp_path / "agent.pt"
    agent.save(checkpoint)
    restored = agent_class(is_delta_residual=True, device="cpu", seed=99)
    restored.load(checkpoint, is_load_optimizer=True)
    state = np.zeros(7, dtype=np.float32)
    latent = np.zeros(4, dtype=np.float32)
    assert restored.get_action(
        state, latent=latent, delta_action=0.5, is_deterministic=True
    ) == pytest.approx(
        agent.get_action(state, latent=latent, delta_action=0.5, is_deterministic=True)
    )


def test_meta_training_exactly_once_validation_and_causal_evaluation(tmp_path: Path) -> None:
    """Verify meta training exactly once validation and causal evaluation."""
    tasks = _get_tasks()
    valid_dataset = get_synthetic_dataset("valid", num_episode=2)
    agent = PearlTD3HedgingAgent(
        actor_hidden_dims=(16, 16),
        critic_hidden_dims=(16, 16),
        context_hidden_dims=(16, 16),
        is_delta_residual=True,
        device="cpu",
        seed=11,
    )

    def get_train_env(dataset: OptionDataset, seed: int) -> HedgingEnv:
        """Return train env."""
        return HedgingEnv(
            dataset, reward_formulation="cash_flow", is_record_trajectory=False, seed=seed
        )

    def get_valid_env() -> HedgingEnv:
        """Return valid env."""
        return HedgingEnv(
            valid_dataset,
            reward_formulation="cash_flow",
            num_parallel_episode=1,
            is_record_trajectory=False,
        )

    result = train_meta_hedging_agent(
        agent,
        tasks,
        get_train_env=get_train_env,
        get_valid_env=get_valid_env,
        num_batch=2,
        num_task_batch=2,
        warmup_fraction=0.05,
        num_update_per_step=1,
        num_valid_episodes=3,
        num_latent_log_interval=2,
        checkpoint_path=tmp_path / "agent.pt",
        checkpoint_dir=tmp_path / "checkpoints",
        update_history_path=tmp_path / "update_history.parquet",
        latent_history_path=tmp_path / "latent_training_history.parquet",
        seed=13,
        is_show_progress=False,
        is_print_summary=False,
    )
    assert result["num_episode"] == 12
    assert result["num_total_step"] == 24
    assert all((item["is_exactly_once"] for item in result["episode_usage_audit"]))
    assert result["validation_history"]["is_best"].sum() == 1
    num_warmup_episode = sum((item["num_warmup_episode"] for item in result["episode_usage_audit"]))
    assert result["validation_history"]["num_train_episode"].min() > num_warmup_episode
    assert "is_checkpoint_eligible" not in result["validation_history"]
    assert "minimum_best_meta_update" not in result["validation_history"]
    assert (tmp_path / "agent.pt").is_file()
    assert (tmp_path / "checkpoints" / "best_agent.pt").is_file()
    assert (tmp_path / "update_history.parquet").is_file()
    update_history = pd.read_parquet(tmp_path / "update_history.parquet")
    assert len(update_history) == 12
    assert np.allclose(update_history["kl_coefficient"], agent.kl_coefficient)
    assert result["config"]["kl_coefficient"] == pytest.approx(agent.kl_coefficient)
    assert (tmp_path / "latent_training_history.parquet").is_file()
    latent_history = pd.read_parquet(tmp_path / "latent_training_history.parquet")
    assert set(latent_history["meta_update"]) == {1, 2, 4, 6, 8, 10, 12}
    evaluation = evaluate_meta_hedging_agent(agent, get_valid_env(), is_print_summary=False)
    assert evaluation["metrics"]["num_episode"] == 2
    prior_rows = evaluation["latent_statistics"].query("context_episode_id.isna()")
    assert len(prior_rows) == 1
    assert np.allclose(prior_rows[[f"posterior_mean_{index}" for index in range(4)]], 0.0)
    contextualized = evaluation["latent_statistics"].query("context_episode_id.notna()")
    assert len(contextualized) == 1
    assert contextualized.iloc[0]["context_episode_id"] == "synthetic_up"
    assert pd.Timestamp(contextualized.iloc[0]["context_cohort_id"]) < pd.Timestamp("2024-01-08")
    protocol = evaluation["context_protocol"]
    assert protocol["name"] == "causal_recent_completed_episode"
    assert protocol["cross_episode_context"] is True
    assert protocol["current_query_transitions_in_context"] is False


def test_recent_context_evaluation_is_reproducible_and_logging_independent() -> None:
    """Verify recent context evaluation is reproducible and logging independent."""
    dataset = get_synthetic_dataset("valid", num_episode=2)
    agent = PearlTD3HedgingAgent(
        actor_hidden_dims=(16,),
        critic_hidden_dims=(16,),
        context_hidden_dims=(16,),
        is_delta_residual=True,
        device="cpu",
        seed=23,
    )

    def get_env(num_parallel_episode: int) -> HedgingEnv:
        """Return env."""
        return HedgingEnv(
            dataset,
            reward_formulation="cash_flow",
            hedge_cost_rate=0.0,
            num_parallel_episode=num_parallel_episode,
            is_record_trajectory=False,
        )

    first = evaluate_meta_hedging_agent(
        agent, get_env(1), num_recent_context_episodes=1, context_seed=17, is_print_summary=False
    )
    repeated = evaluate_meta_hedging_agent(
        agent, get_env(1), num_recent_context_episodes=1, context_seed=17, is_print_summary=False
    )
    without_latent = evaluate_meta_hedging_agent(
        agent,
        get_env(1),
        num_recent_context_episodes=1,
        context_seed=17,
        is_record_latent_statistics=False,
        is_print_summary=False,
    )
    for key in ("mean_loss", "std_loss", "j_lambda", "mean_transaction_cost"):
        assert without_latent["metrics"][key] == pytest.approx(first["metrics"][key], abs=1e-06)
        assert repeated["metrics"][key] == pytest.approx(first["metrics"][key], abs=1e-06)
    assert without_latent["latent_statistics"].empty
    assert first["latent_statistics"]["context_episode_id"].tolist() == [None, "synthetic_up"]
    with pytest.raises(RuntimeError, match="batch"):
        evaluate_meta_hedging_agent(
            agent,
            get_env(2),
            num_recent_context_episodes=1,
            context_seed=17,
            is_print_summary=False,
        )
    with pytest.raises(ValueError, match="context_protocol"):
        evaluate_meta_hedging_agent(
            agent, get_env(2), context_protocol="unsupported_protocol", is_print_summary=False
        )


def test_recent_context_excludes_episode_that_has_not_expired() -> None:
    """Verify recent context excludes episode that has not expired."""
    dataset = get_synthetic_dataset("valid", num_episode=2)
    down_steps = dataset.episode_steps["episode_id"].eq("synthetic_down")
    replacement_dates = pd.to_datetime(["2024-01-03", "2024-01-04", "2024-01-06"])
    dataset.episode_steps.loc[down_steps, "date"] = replacement_dates
    dataset.episode_steps.loc[down_steps, "exdate"] = pd.Timestamp("2024-01-06")
    dataset.episode_steps.loc[down_steps, "cohort_id"] = pd.Timestamp("2024-01-06")
    down_manifest = dataset.episode_manifest["episode_id"].eq("synthetic_down")
    dataset.episode_manifest.loc[down_manifest, "t0"] = pd.Timestamp("2024-01-03")
    dataset.episode_manifest.loc[down_manifest, "exdate"] = pd.Timestamp("2024-01-06")
    dataset.episode_manifest.loc[down_manifest, "cohort_id"] = pd.Timestamp("2024-01-06")
    assert train_meta._get_causal_evaluation_batch_size(dataset, max_num_parallel_episode=256) == 2
    agent = PearlTD3HedgingAgent(
        actor_hidden_dims=(8,),
        critic_hidden_dims=(8,),
        context_hidden_dims=(8,),
        device="cpu",
        seed=29,
    )
    evaluation = evaluate_meta_hedging_agent(
        agent,
        HedgingEnv(
            dataset,
            reward_formulation="cash_flow",
            hedge_cost_rate=0.0,
            num_parallel_episode=2,
            is_record_trajectory=False,
        ),
        num_recent_context_episodes=1,
        context_seed=29,
        is_print_summary=False,
    )
    assert evaluation["latent_statistics"]["context_episode_id"].isna().all()
    assert evaluation["context_protocol"]["num_prior_cohort"] == 2


def test_meta_evaluation_environment_keeps_complete_cohort_in_safe_batch() -> None:
    """Verify meta evaluation environment keeps complete cohort in safe batch."""
    dataset = get_synthetic_dataset("valid", num_episode=2)
    up_steps = dataset.episode_steps["episode_id"].eq("synthetic_up")
    down_steps = dataset.episode_steps["episode_id"].eq("synthetic_down")
    up_dates = dataset.episode_steps.loc[up_steps, "date"].to_numpy()
    dataset.episode_steps.loc[down_steps, "date"] = up_dates
    dataset.episode_steps.loc[down_steps, "exdate"] = pd.Timestamp("2024-01-05")
    dataset.episode_steps.loc[down_steps, "cohort_id"] = pd.Timestamp("2024-01-05")
    down_manifest = dataset.episode_manifest["episode_id"].eq("synthetic_down")
    dataset.episode_manifest.loc[down_manifest, "t0"] = pd.Timestamp("2024-01-02")
    dataset.episode_manifest.loc[down_manifest, "exdate"] = pd.Timestamp("2024-01-05")
    dataset.episode_manifest.loc[down_manifest, "cohort_id"] = pd.Timestamp("2024-01-05")
    config = get_default_meta_config()
    run_spec = get_meta_run_specs(config)[0]
    env = train_meta._get_environment(dataset, None, config, run_spec, label="valid")
    assert env.num_parallel_episode == 2
    _, reset_info = env.reset()
    assert len(reset_info["episode_id"]) == 2
    assert len(pd.DatetimeIndex(reset_info["cohort_id"]).unique()) == 1
