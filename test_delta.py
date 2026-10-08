"""Test delta for the paper option-hedging pipeline."""

from __future__ import annotations
import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
import numpy as np
import torch

CODES_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = CODES_ROOT
for _source_directory in (
    CODES_ROOT,
    CODES_ROOT / "rl_envs",
    CODES_ROOT / "datasets",
    CODES_ROOT / "data_processing",
):
    if str(_source_directory) not in sys.path:
        sys.path.insert(0, str(_source_directory))
from rl_agents import BaseHedgingAgent
from rl_utils import evaluate_hedging_agent
from option_dataset import OptionDataset
from hedging_env import HedgingEnv
from test_basis import _save_test_frames
from train_basis import (
    BASIS_CONFIG_SCHEMA_VERSION,
    VALID_REWARD_FORMULATIONS,
    _validate_basis_config,
    _write_json_atomic,
    get_basis_config,
    get_basis_datasets,
    get_code_fingerprints,
    get_dataset_record,
    get_datetime_record,
    get_default_basis_config,
    get_contract_multiplier,
    get_currency,
    get_safe_path_component,
    get_system_info,
)
from data_source import VALID_DATA_SOURCES, validate_data_source


class DeltaHedgingAgent(BaseHedgingAgent):
    """Delta hedging agent."""

    algorithm_name = "bsm_delta"
    is_off_policy = False

    def __init__(
        self, *, num_state_feature: int = 7, device: str | torch.device | None = None
    ) -> None:
        """Initialize validated configuration and internal state."""
        super().__init__(
            num_state_feature=num_state_feature, is_delta_residual=False, device=device, seed=None
        )
        self.is_require_delta_action = True

    def get_action(
        self, state: Any, *, delta_action: Any = None, is_deterministic: bool = True
    ) -> float | np.ndarray:
        """Return the policy action, including the Delta correction when enabled."""
        if not isinstance(is_deterministic, (bool, np.bool_)):
            raise ValueError("is_deterministic must be a boolean")
        state_tensor, is_single = self._get_state_tensor(state)
        delta_tensor = self._get_delta_tensor(
            delta_action, num_batch=state_tensor.shape[0], is_required=True
        )
        return self._get_public_action(delta_tensor, is_single=is_single)

    def update(self, batch: Mapping[str, Any]) -> dict[str, float]:
        """Update model parameters from the supplied training batch and return optimization diagnostics."""
        raise RuntimeError("DeltaHedgingAgent is a deterministic baseline requiring no training")

    def get_config(self) -> dict[str, Any]:
        """Return the configuration required to reconstruct this object."""
        return {
            "algorithm_name": self.algorithm_name,
            "policy_type": "non_trainable_baseline",
            "is_trainable": False,
            "is_deterministic": True,
            "is_delta_residual": False,
            "is_require_delta_action": True,
            "num_state_feature": self.num_state_feature,
            "action_definition": "a_t = h_t / M = call_delta_t",
            "call_delta_formula": "exp(-q*tau) * Phi(d1)",
            "d1_formula": "[log(S/K) + (r-q+0.5*sigma^2)*tau] / (sigma*sqrt(tau))",
            "market_state_source": "HedgingEnv raw current S,K,tau,IV,r,q",
            "action_bounds": [0.0, 1.0],
            "device": str(self.device),
        }

    def _get_checkpoint_state(self) -> dict[str, Any]:
        """Return checkpoint state."""
        return {}

    def _load_checkpoint_state(
        self, checkpoint: Mapping[str, Any], *, is_load_optimizer: bool
    ) -> None:
        """Load checkpoint state."""
        raise RuntimeError("DeltaHedgingAgent does not accept checkpoints")


def get_delta_test_env(
    test_dataset: OptionDataset,
    config: Mapping[str, Any],
    *,
    reward_formulation: str,
    is_record_trajectory: bool | None = None,
) -> HedgingEnv:
    """Return delta test env."""
    if reward_formulation not in VALID_REWARD_FORMULATIONS:
        raise ValueError(f"reward_formulation must be {VALID_REWARD_FORMULATIONS}  ")
    environment = config["environment"]
    record_trajectory = (
        environment["is_record_test_trajectory"]
        if is_record_trajectory is None
        else is_record_trajectory
    )
    if not isinstance(record_trajectory, (bool, np.bool_)):
        raise ValueError("is_record_trajectory must be a boolean or None")
    return HedgingEnv(
        test_dataset,
        reward_formulation=reward_formulation,
        hedge_cost_rate=environment["hedge_cost_rate"],
        contract_multiplier=get_contract_multiplier(config),
        currency=get_currency(config),
        reconciliation_tolerance=environment["reconciliation_tolerance"],
        risk_aversion_lambda=environment["risk_aversion_lambda"],
        reward_risk_aversion_xi=environment["reward_risk_aversion_xi"],
        training_reward_scale=environment["training_reward_scales"][reward_formulation],
        state_normalizer=None,
        num_parallel_episode=environment["num_parallel_test_episode"],
        is_clip_action=environment["is_clip_action"],
        is_record_trajectory=bool(record_trajectory),
        observation_dtype=np.dtype(environment["observation_dtype"]),
    )


def _get_selected_reward_formulations(
    config: Mapping[str, Any], selected: Sequence[str] | None
) -> list[str]:
    """Return selected reward formulations."""
    configured = [str(value) for value in config["environment"]["reward_formulations"]]
    if selected is None:
        return configured
    selected_set = set(selected)
    results = [value for value in configured if value in selected_set]
    if not results:
        raise ValueError("No reward_formulation matches the CLI filters")
    return results


def _get_cross_formulation_consistency(
    evaluations: Mapping[str, Mapping[str, Any]], *, tolerance: float
) -> dict[str, Any] | None:
    """Return cross formulation consistency."""
    accounting_name = "shaped_accounting"
    if accounting_name not in evaluations or "cash_flow" not in evaluations:
        return None
    accounting = evaluations[accounting_name]["episode_results"]
    cash_flow = evaluations["cash_flow"]["episode_results"]
    keys = ["draw_id", "episode_index", "episode_id"]
    left = accounting.loc[:, [*keys, "net_loss"]].rename(
        columns={"net_loss": "accounting_net_loss"}
    )
    right = cash_flow.loc[:, [*keys, "net_loss"]].rename(columns={"net_loss": "cash_flow_net_loss"})
    merged = left.merge(right, on=keys, how="outer", validate="one_to_one")
    if len(merged) != len(accounting) or len(merged) != len(cash_flow):
        raise RuntimeError("The reward formulations use different test episode sets")
    differences = np.abs(
        merged["accounting_net_loss"].to_numpy(dtype=np.float64)
        - merged["cash_flow_net_loss"].to_numpy(dtype=np.float64)
    )
    if not np.isfinite(differences).all():
        raise RuntimeError("Cross-formulation loss differences contain nonfinite values")
    max_absolute_difference = float(differences.max()) if differences.size else 0.0
    if max_absolute_difference > tolerance:
        raise RuntimeError(
            f"Delta terminal losses differ between Accounting and Cash Flow: max_abs_diff={max_absolute_difference:.3e}, tolerance={tolerance:.3e}"
        )
    return {
        "is_consistent": True,
        "accounting_reference": accounting_name,
        "num_episode": len(merged),
        "reconciliation_tolerance": tolerance,
        "max_absolute_net_loss_difference": max_absolute_difference,
    }


def run_delta_testing(
    config: Mapping[str, Any],
    *,
    reward_formulations: Sequence[str] | None = None,
    output_root: str | Path = "test_results/delta",
    experiment_id: str | None = None,
    device: str | torch.device | None = None,
    is_overwrite: bool = False,
    is_print_summary: bool | None = None,
    is_save_test_trajectory: bool | None = None,
) -> dict[str, Any]:
    """Run delta testing."""
    _validate_basis_config(config)
    rewards = _get_selected_reward_formulations(config, reward_formulations)
    test_dataset = get_basis_datasets(config, labels=("test",))["test"]
    test_dataset_record = get_dataset_record(test_dataset)
    output_path = Path(output_root)
    if not output_path.is_absolute():
        output_path = REPOSITORY_ROOT / output_path
    if experiment_id is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        experiment_id = f"delta_{timestamp}"
    experiment_id = get_safe_path_component(experiment_id, "experiment_id")
    experiment_dir = output_path / experiment_id
    experiment_dir.mkdir(parents=True, exist_ok=True)
    print_summary = (
        config["testing"]["is_print_summary"] if is_print_summary is None else is_print_summary
    )
    save_trajectory = (
        config["testing"]["is_save_test_trajectory"]
        if is_save_test_trajectory is None
        else is_save_test_trajectory
    )
    for name, value in (
        ("is_overwrite", is_overwrite),
        ("is_print_summary", print_summary),
        ("is_save_test_trajectory", save_trajectory),
    ):
        if not isinstance(value, (bool, np.bool_)):
            raise ValueError(f"{name} must be a boolean")
    started_at = get_datetime_record()
    evaluations: dict[str, dict[str, Any]] = {}
    summaries: list[dict[str, Any]] = []
    for num_run, reward_formulation in enumerate(rewards, start=1):
        run_id = f"bsm_delta__{reward_formulation}"
        run_dir = experiment_dir / run_id
        if run_dir.exists() and any(run_dir.iterdir()) and (not is_overwrite):
            raise FileExistsError(
                f"The Delta test directory already exists and is nonempty: {run_dir}"
            )
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[Delta evaluation {num_run}/{len(rewards)}] {run_id}", flush=True)
        run_started_at = get_datetime_record()
        num_start_time = time.perf_counter()
        agent = DeltaHedgingAgent(
            num_state_feature=config["agent"]["num_state_feature"],
            device=device if device is not None else config["testing"]["device"],
        )
        env = get_delta_test_env(
            test_dataset,
            config,
            reward_formulation=reward_formulation,
            is_record_trajectory=bool(save_trajectory),
        )
        evaluation = evaluate_hedging_agent(agent, env, is_print_summary=bool(print_summary))
        evaluations[reward_formulation] = evaluation
        artifacts = _save_test_frames(
            run_dir, evaluation, is_save_test_trajectory=bool(save_trajectory)
        )
        completed_at = get_datetime_record()
        metrics_record = {
            "schema_version": BASIS_CONFIG_SCHEMA_VERSION,
            "data_source": config.get("data_source", "spx"),
            "experiment_id": experiment_id,
            "run_id": run_id,
            "test_completed_at": completed_at,
            "metrics": evaluation["metrics"],
        }
        _write_json_atomic(run_dir / "test_metrics.json", metrics_record)
        test_record = {
            "schema_version": BASIS_CONFIG_SCHEMA_VERSION,
            "data_source": config.get("data_source", "spx"),
            "status": "completed",
            "experiment_id": experiment_id,
            "run_id": run_id,
            "test_started_at": run_started_at,
            "test_completed_at": completed_at,
            "num_elapsed_second": time.perf_counter() - num_start_time,
            "requested_config": config,
            "resolved_baseline": {
                "policy": "bsm_call_delta",
                "reward_formulation": reward_formulation,
            },
            "test_dataset": test_dataset_record,
            "test_environment": env.get_config(),
            "agent": agent.get_config(),
            "metrics": evaluation["metrics"],
            "num_test_reset": evaluation["num_reset"],
            "num_test_step": evaluation["num_total_step"],
            "artifacts": {
                **artifacts,
                "test_metrics": "test_metrics.json",
                "test_config": "test_config.json",
            },
            "system": get_system_info(),
            "code_fingerprints": get_code_fingerprints(),
        }
        _write_json_atomic(run_dir / "test_config.json", test_record)
        summaries.append(
            {
                "run_id": run_id,
                "status": "completed",
                "test_completed_at": completed_at,
                "metrics": evaluation["metrics"],
            }
        )
    consistency = _get_cross_formulation_consistency(
        evaluations, tolerance=float(config["environment"]["reconciliation_tolerance"])
    )
    summary = {
        "status": "completed",
        "data_source": config.get("data_source", "spx"),
        "experiment_id": experiment_id,
        "started_at": started_at,
        "completed_at": get_datetime_record(),
        "num_run": len(summaries),
        "test_dataset": test_dataset_record,
        "cross_formulation_consistency": consistency,
        "runs": summaries,
    }
    _write_json_atomic(experiment_dir / "delta_test_summary.json", summary)
    print(f"\n[Delta evaluation completed] {experiment_dir}", flush=True)
    return {"experiment_dir": experiment_dir, "summary": summary, "evaluations": evaluations}


def get_argument_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    defaults = get_default_basis_config()
    parser = argparse.ArgumentParser(description="Evaluate the BSM Delta hedging baseline")
    parser.add_argument(
        "--config", type=Path, help="JSON file overriding the default baseline configuration"
    )
    parser.add_argument(
        "--reward-formulation",
        action="append",
        choices=VALID_REWARD_FORMULATIONS,
        help="Evaluate the specified reward; defaults to shaped_accounting and cash_flow",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("test_results/delta"),
        help="Delta baseline evaluation output root",
    )
    parser.add_argument(
        "--experiment-id", help="Explicit experiment directory name; defaults to a UTC timestamp"
    )
    parser.add_argument("--device", default=defaults["testing"]["device"], help="Compute device")
    parser.add_argument(
        "--overwrite", action="store_true", help="Allow overwriting standard outputs"
    )
    parser.add_argument(
        "--quiet",
        action="store_const",
        const=False,
        default=None,
        dest="is_print_summary",
        help="Disable per-run metric printing; otherwise inherit the JSON setting",
    )
    parser.add_argument(
        "--no-trajectory",
        action="store_const",
        const=False,
        default=None,
        dest="is_save_test_trajectory",
        help="Disable transition trajectory saving; otherwise inherit the JSON setting",
    )
    parser.add_argument(
        "--data-source",
        choices=VALID_DATA_SOURCES,
        default=None,
        help="data source; sx5e triggers canonical preprocessing",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse command-line arguments, run the requested operation, and report failures."""
    parser = get_argument_parser()
    args = parser.parse_args(argv)
    config = get_basis_config(args.config)
    config["data_source"] = validate_data_source(
        args.data_source or config.get("data_source", "spx")
    )
    _validate_basis_config(config)
    print(f"[Data source] data_source={config['data_source']}", flush=True)
    run_delta_testing(
        config,
        reward_formulations=args.reward_formulation,
        output_root=args.output_root,
        experiment_id=args.experiment_id,
        device=args.device,
        is_overwrite=args.overwrite,
        is_print_summary=args.is_print_summary,
        is_save_test_trajectory=args.is_save_test_trajectory,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
