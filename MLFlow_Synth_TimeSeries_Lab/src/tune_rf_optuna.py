# src/tune_rf_optuna.py

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import mlflow
import mlflow.sklearn
import numpy as np
import optuna
from sklearn.ensemble import RandomForestRegressor

from src.signal_gen import SyntheticWeatherSignal, SignalConfig
from src.utils import (
    create_lagged_features,
    train_val_test_split_time,
    compute_regression_metrics,
)

# Ensure MLflow always writes to the root-level mlruns directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MLRUNS_PATH = PROJECT_ROOT / "mlruns"
mlflow.set_tracking_uri(MLRUNS_PATH.as_uri())

"""
Path(__file__).resolve().parents[1] → MLFlow_Synth_TimeSeries_Lab
MLRUNS_PATH → <root>/mlruns
as_uri() → file:///d:/.../MLFlow_Synth_TimeSeries_Lab/mlruns
set_tracking_uri() → MLflow writes there regardless of where the process was started (CLI or notebook).
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optuna sweeps for RandomForest on synthetic weather series."
    )
    parser.add_argument("--length", type=int, default=2000)
    parser.add_argument("--years", type=float, default=4.0)
    parser.add_argument("--noise-std", type=float, default=2.0)
    parser.add_argument("--train-frac", type=float, default=0.6)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="synthetic_weather_rf_optuna",
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default="rf_weather_study",
    )
    return parser.parse_args()


def build_objective(args: argparse.Namespace) -> Callable[[optuna.Trial], float]:
    """
    Wrap configuration so Optuna's objective can see args.
    We minimize val_rmse (but log SMAPE, etc. as well).
    """

    def objective(trial: optuna.Trial) -> float:
        # Search space for hyperparameters
        window_size = trial.suggest_int("window_size", 24, 144, step=12)
        n_estimators = trial.suggest_int("n_estimators", 150, 600, step=50)
        max_depth = trial.suggest_int("max_depth", 8, 40, step=4)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 5)

        run_name = (
            f"optuna_trial_{trial.number}_ws{window_size}"
            f"_trees{n_estimators}_depth{max_depth}_leaf{min_samples_leaf}"
        )

        # Nested run so you can have a parent "study" run if desired
        with mlflow.start_run(run_name=run_name, nested=True):
            # Log Optuna + data config
            mlflow.log_param("optuna_trial_number", trial.number)
            mlflow.log_param("length", args.length)
            mlflow.log_param("years", args.years)
            mlflow.log_param("noise_std", args.noise_std)
            mlflow.log_param("train_frac", args.train_frac)
            mlflow.log_param("val_frac", args.val_frac)
            mlflow.log_param("seed", args.seed)

            mlflow.log_param("window_size", window_size)
            mlflow.log_param("n_estimators", n_estimators)
            mlflow.log_param("max_depth", max_depth)
            mlflow.log_param("min_samples_leaf", min_samples_leaf)
            mlflow.set_tag("data.synthetic", "simplified_weather_v1")
            mlflow.set_tag("model.family", "RandomForestRegressor")
            mlflow.set_tag("tuning.algorithm", "Optuna")

            # Generate synthetic signal
            cfg = SignalConfig(
                length=args.length,
                years=args.years,
                noise_std=args.noise_std,
                seed=args.seed,
                include_components=False,
            )
            generator = SyntheticWeatherSignal(cfg)
            df = generator.generate()
            series = df["temp"]

            # Build lagged dataset and time splits
            X, y = create_lagged_features(series, window_size=window_size)
            (X_train, y_train), (X_val, y_val), (X_test, y_test) = train_val_test_split_time(
                X, y, train_frac=args.train_frac, val_frac=args.val_frac
            )

            # Model
            rf = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                random_state=args.seed,
                n_jobs=-1,
            )
            rf.fit(X_train, y_train)

            # Validation metrics drive the objective
            y_pred_val = rf.predict(X_val)
            val_metrics = compute_regression_metrics(y_val, y_pred_val)
            for k, v in val_metrics.items():
                mlflow.log_metric(f"val_{k}", float(v))

            # Also log test metrics for later inspection (not used by Optuna)
            y_pred_test = rf.predict(X_test)
            test_metrics = compute_regression_metrics(y_test, y_pred_test)
            for k, v in test_metrics.items():
                mlflow.log_metric(f"test_{k}", float(v))

            # Log the model snapshot for this trial
            mlflow.sklearn.log_model(rf, artifact_path="model")

            # Objective: minimize validation RMSE
            return float(val_metrics["rmse"])

    return objective


def main() -> None:
    args = parse_args()

    mlflow.set_experiment(args.experiment_name)

    # Optional parent run to group all trials in MLflow UI
    with mlflow.start_run(run_name=f"optuna_study_{args.study_name}"):
        objective = build_objective(args)
        study = optuna.create_study(
            study_name=args.study_name,
            direction="minimize",
        )
        study.optimize(objective, n_trials=args.n_trials)

        best = study.best_trial
        print("\n=== Optuna best trial ===")
        print(f"Trial number: {best.number}")
        print(f"Best val_rmse: {best.value:.4f}")
        print("Best params:")
        for k, v in best.params.items():
            print(f"  {k}: {v}")

        # Log best trial summary on parent run
        mlflow.log_metric("best_val_rmse", float(best.value))
        for k, v in best.params.items():
            mlflow.log_param(f"best_{k}", v)


if __name__ == "__main__":
    main()
