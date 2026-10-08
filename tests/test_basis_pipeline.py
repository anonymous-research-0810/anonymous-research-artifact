"""Test basis pipeline for the paper option-hedging pipeline."""

from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import pytest
import test_basis
import train_basis
from tests.rl_envs.helpers import get_synthetic_dataset


def get_small_basis_config(tmp_path: Path) -> dict:
    """Return small basis config."""
    config = train_basis.get_default_basis_config()
    config["environment"]["reward_formulations"] = ["shaped_accounting"]
    config["environment"]["hedge_cost_rate"] = 0.0
    config["environment"]["num_parallel_valid_episode"] = 2
    config["environment"]["num_parallel_test_episode"] = 2
    config["agent"]["algorithm_names"] = ["td3"]
    config["agent"]["is_delta_residual_values"] = [True]
    config["agent"]["td3"]["actor_hidden_dims"] = [8]
    config["agent"]["td3"]["critic_hidden_dims"] = [8]
    config["training"]["seeds"] = [7]
    config["training"]["num_valid_episodes"] = 1
    config["training"]["num_batch"] = 2
    config["training"]["warmup_fraction"] = 0.0
    config["training"]["is_show_progress"] = False
    config["training"]["is_print_summary"] = False
    config["training"]["output_root"] = str(tmp_path / "train")
    config["testing"]["output_root"] = str(tmp_path / "test")
    config["testing"]["is_print_summary"] = False
    return config


def test_evaluation_cli_inherits_sx5e_training_market(tmp_path, monkeypatch):
    """Reload an SX5E checkpoint through the CLI without overriding its market."""
    config = get_small_basis_config(tmp_path)
    config["data_source"] = "sx5e"

    def get_datasets(config, *, labels=("train", "valid")):
        return {label: get_synthetic_dataset(label, num_episode=2) for label in labels}

    monkeypatch.setattr(train_basis, "get_basis_datasets", get_datasets)
    monkeypatch.setattr(test_basis, "get_basis_datasets", get_datasets)
    train_basis.run_basis_training(config, experiment_id="market_inheritance")
    assert (
        test_basis.main(
            [
                "--train-results-root",
                config["training"]["output_root"],
                "--experiment-id",
                "market_inheritance",
            ]
        )
        == 0
    )
    result_path = next((tmp_path / "test").rglob("test_config.json"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["test_environment"]["currency"] == "EUR"
    assert result["test_environment"]["contract_multiplier"] == 10.0


def test_default_matrix_contains_eight_configurations_per_seed() -> None:
    """Verify eight configurations for each of ten paper seeds."""
    config = train_basis.get_default_basis_config()
    specs = train_basis.get_basis_run_specs(config)
    assert len(specs) == 80
    assert len({spec["run_id"] for spec in specs}) == 80
    assert config["data"]["expiry_weekday"] is None
    assert config["data"]["num_moneyness"] == 3
    assert config["training"]["num_valid_episodes"] == 100
    assert config["agent"]["max_delta_residual"] == pytest.approx(0.1)
    assert config["agent"]["algorithm_names"] == ["sac", "td3"]
    assert config["environment"]["reward_formulations"] == ["shaped_accounting", "cash_flow"]
    assert config["environment"]["hedge_cost_rate"] == pytest.approx(0.001)
    assert config["environment"]["reward_risk_aversion_xi"] == pytest.approx(1.5)
    assert config["environment"]["risk_aversion_lambda"] == pytest.approx(1.5)
    assert config["environment"]["training_reward_scales"] == {
        "shaped_accounting": 100.0,
        "cash_flow": 1.0,
    }
    assert {spec["training_reward_scale"] for spec in specs} == {1.0, 100.0}
    assert config["training"]["warmup_fraction"] == pytest.approx(0.1)


def test_config_merge_rejects_unknown_fields(tmp_path: Path) -> None:
    """Verify config merge rejects unknown fields."""
    path = tmp_path / "bad_config.json"
    path.write_text(json.dumps({"training": {"misspelled_parameter": 1}}), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown fields"):
        train_basis.get_basis_config(path)


@pytest.mark.parametrize("invalid_scale", [0.0, -1.0, float("inf"), True])
def test_config_rejects_invalid_training_reward_scale(invalid_scale: object) -> None:
    """Verify config rejects invalid training reward scale."""
    config = train_basis.get_default_basis_config()
    config["environment"]["training_reward_scales"]["shaped_accounting"] = invalid_scale
    with pytest.raises(ValueError, match="training_reward_scales.shaped_accounting"):
        train_basis._validate_basis_config(config)


def test_checkpoint_pruning_refuses_unexpected_artifacts(tmp_path: Path) -> None:
    """Verify checkpoint pruning refuses unexpected artifacts."""
    run_dir = tmp_path / "run"
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    model_path = run_dir / "agent.pt"
    model_path.write_bytes(b"selected model")
    (checkpoint_dir / "best_agent.pt").write_bytes(b"checkpoint")
    unexpected_path = checkpoint_dir / "training_notes.json"
    unexpected_path.write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="files other than checkpoints"):
        train_basis.prune_run_checkpoints(run_dir, checkpoint_dir, model_path)
    assert checkpoint_dir.is_dir()
    assert unexpected_path.is_file()
    assert model_path.is_file()


def test_complete_basis_train_and_test_artifact_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify complete basis train and test artifact chain."""
    config = get_small_basis_config(tmp_path)

    def get_training_datasets(config, *, labels=("train", "valid")):
        """Return training datasets."""
        return {label: get_synthetic_dataset(label, num_episode=2) for label in labels}

    monkeypatch.setattr(train_basis, "get_basis_datasets", get_training_datasets)
    training = train_basis.run_basis_training(config, experiment_id="unit_experiment")
    experiment_dir = training["experiment_dir"]
    run_spec = train_basis.get_basis_run_specs(config)[0]
    run_dir = experiment_dir / run_spec["run_id"]
    run_config_path = run_dir / "run_config.json"
    run_record = json.loads(run_config_path.read_text(encoding="utf-8"))
    assert run_record["status"] == "completed"
    assert run_record["run_completed_at"]["utc"]
    assert run_record["agent"]["max_delta_residual"] == pytest.approx(0.1)
    assert run_record["resolved_run"]["training_reward_scale"] == pytest.approx(100.0)
    assert run_record["train_environment"]["training_reward_scale"] == pytest.approx(100.0)
    assert (run_dir / "agent.pt").is_file()
    assert not (run_dir / "checkpoints").exists()
    assert list(run_dir.rglob("*.pt")) == [run_dir / "agent.pt"]
    assert (run_dir / "validation_history.parquet").is_file()
    validation_history = pd.read_parquet(run_dir / "validation_history.parquet")
    assert validation_history["checkpoint_path"].isna().all()
    assert run_record["artifacts"]["model_role"] == "validation_best"
    retention = run_record["artifacts"]["checkpoint_retention"]
    assert retention["policy"] == "validation_best_only"
    assert retention["selected_model"] == "agent.pt"
    assert retention["checkpoint_directory_removed"] is True
    assert retention["num_removed_model_file"] == 3
    assert run_record["model_selection"]["metric"] == "j_lambda"
    assert run_record["model_selection"]["mode"] == "min"
    assert run_record["model_selection"]["best_checkpoint"] == "agent.pt"
    assert len(run_record["best_validation_metrics"]) == 6
    assert len(run_record["final_validation_metrics"]) == 6
    records = test_basis.get_completed_train_run_configs(
        config["training"]["output_root"], experiment_id="unit_experiment"
    )

    def get_testing_datasets(config, *, labels=("test",)):
        """Return testing datasets."""
        return {label: get_synthetic_dataset(label, num_episode=2) for label in labels}

    monkeypatch.setattr(test_basis, "get_basis_datasets", get_testing_datasets)
    testing = test_basis.run_basis_testing(records)
    test_run_dir = testing["experiment_dir"] / run_spec["run_id"]
    test_record = json.loads((test_run_dir / "test_config.json").read_text(encoding="utf-8"))
    metrics_record = json.loads((test_run_dir / "test_metrics.json").read_text(encoding="utf-8"))
    assert test_record["status"] == "completed"
    assert test_record["training_completed_at"] == run_record["run_completed_at"]
    assert test_record["model_selection"] == run_record["model_selection"]
    assert test_record["test_environment"]["training_reward_scale"] == pytest.approx(100.0)
    assert metrics_record["metrics"]["num_episode"] == 2
    assert (test_run_dir / "test_episode_results.parquet").is_file()
    assert (test_run_dir / "test_trajectory.parquet").is_file()
