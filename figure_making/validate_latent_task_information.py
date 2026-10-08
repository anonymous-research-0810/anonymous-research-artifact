"""Validate task information in Meta Deep Hedging posterior latents.

Training evaluation uses five-fold stratified cross-validation grouped only by
episode_id. Test evaluation transfers the train-fitted Extra Trees probe to
unlabelled test cohorts and reports descriptive diagnostics.
"""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

RANDOM_SEED = 2026


def _configuration(run_id: str, experiment_id: str) -> dict[str, str]:
    tokens = run_id.split("__")
    return {
        "experiment_id": experiment_id,
        "run_id": run_id,
        "algorithm": tokens[0].replace("pearl_", "").upper(),
        "action": tokens[1],
        "reward": tokens[2],
    }


def _feature_columns(frame: pd.DataFrame) -> list[str]:
    means = sorted(
        (c for c in frame if c.startswith("posterior_mean_") and c.rsplit("_", 1)[-1].isdigit())
    )
    stds = sorted(
        (c for c in frame if c.startswith("posterior_std_") and c.rsplit("_", 1)[-1].isdigit())
    )
    if not means or len(means) != len(stds):
        raise ValueError("posterior mean/std columns are incomplete")
    return means + stds


def _informative_training_rows(path: Path) -> tuple[pd.DataFrame, list[str]]:
    frame = pd.read_parquet(path)
    frame = frame.loc[frame["num_observed_transition"] > 0].reset_index(drop=True)
    columns = _feature_columns(frame)
    if frame.empty or frame[columns].isna().any().any():
        raise ValueError(f"invalid training latent data: {path}")
    return (frame, columns)


def _new_probe() -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=100,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )


def _training_validation(path: Path, experiment_id: str) -> dict[str, object]:
    frame, columns = _informative_training_rows(path)
    X = frame[columns].to_numpy(dtype=np.float64)
    y = frame["segment_id"].to_numpy(dtype=np.int64)
    groups = frame["episode_id"].astype(str).to_numpy()
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    prediction = np.empty_like(y)
    fold_accuracy: list[float] = []
    for train_index, test_index in splitter.split(X, y, groups):
        probe = _new_probe()
        probe.fit(X[train_index], y[train_index])
        prediction[test_index] = probe.predict(X[test_index])
        fold_accuracy.append(float(accuracy_score(y[test_index], prediction[test_index])))
    real = ~frame["is_simulated"].to_numpy(dtype=bool)
    simulated = ~real
    counts = pd.Series(y).value_counts()
    return {
        **_configuration(path.parent.name, experiment_id),
        "num_fold": 5,
        "num_sample": int(len(frame)),
        "num_episode": int(frame["episode_id"].nunique()),
        "accuracy": float(accuracy_score(y, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "macro_f1": float(f1_score(y, prediction, average="macro")),
        "real_accuracy": float(accuracy_score(y[real], prediction[real])),
        "simulated_accuracy": float(accuracy_score(y[simulated], prediction[simulated])),
        "majority_baseline": float(counts.max() / counts.sum()),
    }


def _test_transfer(
    train_path: Path, test_path: Path, experiment_id: str, num_task: int
) -> dict[str, object]:
    train, columns = _informative_training_rows(train_path)
    test = pd.read_parquet(test_path)
    test = (
        test.sort_values(["cohort_id", "episode_id"]).groupby("cohort_id", as_index=False).first()
    )
    test = test.loc[test["num_total_context_transition"] > 0].reset_index(drop=True)
    base = _configuration(train_path.parent.name, experiment_id)
    if test.empty:
        return {
            **base,
            "num_informative_cohort": 0,
            "last_task_share": None,
            "dominant_task": None,
            "dominant_task_share": None,
            "in_distribution_share": None,
            "median_margin": None,
        }
    X_train = train[columns].to_numpy(dtype=np.float64)
    y_train = train["segment_id"].to_numpy(dtype=np.int64)
    X_test = test[columns].to_numpy(dtype=np.float64)
    probe = _new_probe()
    probe.fit(X_train, y_train)
    prediction = probe.predict(X_test).astype(np.int64)
    probability = probe.predict_proba(X_test)
    sorted_probability = np.sort(probability, axis=1)
    margin = sorted_probability[:, -1] - sorted_probability[:, -2]
    scaler = StandardScaler().fit(X_train)
    scaled_train = scaler.transform(X_train)
    scaled_test = scaler.transform(X_test)
    is_in_distribution = np.zeros(len(test), dtype=bool)
    for task in np.unique(y_train):
        task_train = scaled_train[y_train == task]
        centroid = task_train.mean(axis=0)
        train_distance = np.linalg.norm(task_train - centroid, axis=1)
        threshold = float(np.quantile(train_distance, 0.95))
        selected = prediction == task
        test_distance = np.linalg.norm(scaled_test[selected] - centroid, axis=1)
        is_in_distribution[selected] = test_distance <= threshold
    values, counts = np.unique(prediction, return_counts=True)
    dominant_index = int(np.argmax(counts))
    return {
        **base,
        "num_informative_cohort": int(len(test)),
        "last_task_share": float(np.mean(prediction == num_task)),
        "dominant_task": int(values[dominant_index]),
        "dominant_task_share": float(counts[dominant_index] / len(test)),
        "in_distribution_share": float(np.mean(is_in_distribution)),
        "median_margin": float(np.median(margin)),
    }


def _sort_key(result: dict[str, object]) -> tuple[int, int, int]:
    reward_order = 0 if result["reward"] == "shaped_accounting" else 1
    algorithm_order = 0 if result["algorithm"] == "TD3" else 1
    action_order = 0 if result["action"] == "delta_residual" else 1
    return (reward_order, algorithm_order, action_order)


def main(argv=None) -> int:
    """Run the paper diagnostics on an explicitly selected experiment."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-experiment", type=Path, required=True)
    parser.add_argument("--test-experiment", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    training, transfer = ([], [])
    files = sorted(args.train_experiment.glob("*/latent_episode_statistics.parquet"))
    if not files:
        raise FileNotFoundError(
            "No training latent statistics were found in the selected experiment"
        )
    for train_path in files:
        training.append(_training_validation(train_path, args.train_experiment.name))
        if args.test_experiment is not None:
            test_path = (
                args.test_experiment / train_path.parent.name / "latent_episode_statistics.parquet"
            )
            if not test_path.is_file():
                raise FileNotFoundError(f"Missing test latent statistics: {test_path}")
            frame, _ = _informative_training_rows(train_path)
            transfer.append(
                _test_transfer(
                    train_path,
                    test_path,
                    args.train_experiment.name,
                    int(frame["segment_id"].max()),
                )
            )
    training.sort(key=_sort_key)
    transfer.sort(key=_sort_key)
    output = {
        "protocol": {
            "features": "posterior means and standard deviations",
            "split": "five-fold stratified grouped CV by episode_id",
            "probe": "class-balanced Extra Trees, 100 trees, minimum leaf size 2",
            "random_seed": RANDOM_SEED,
            "test_note": "test-time transfer diagnostics have no task accuracy labels",
        },
        "training_validation": training,
        "test_transfer": transfer,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Latent diagnostics written: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
