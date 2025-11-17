# src/train_rf.py

from pathlib import Path
import argparse



import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.ensemble import RandomForestRegressor

from src.signal_gen import SyntheticWeatherSignal, SignalConfig
from src.utils import (
    compute_regression_metrics,
    create_lagged_features,
    train_val_test_split_time,
    plot_predictions_time,
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
        description="Train RandomForest on synthetic weather series and log to MLflow."
    )
    parser.add_argument("--length", type=int, default=2000, help="Series length.")
    parser.add_argument("--years", type=float, default=4.0, help="Span in years.")
    parser.add_argument("--noise-std", type=float, default=2.0, help="Noise std.")
    parser.add_argument("--window-size", type=int, default=60, help="Lag window size.")
    parser.add_argument("--n-estimators", type=int, default=300, help="RF trees.")
    parser.add_argument("--max-depth", type=int, default=None, help="RF max depth.")
    parser.add_argument(
        "--min-samples-leaf", type=int, default=2, help="RF min samples per leaf."
    )
    parser.add_argument(
        "--train-frac", type=float, default=0.6, help="Train fraction (time split)."
    )
    parser.add_argument(
        "--val-frac", type=float, default=0.2, help="Validation fraction (time split)."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for signal + model."
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="synthetic_weather_rf",
        help="MLflow experiment name.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Set experiment
    mlflow.set_experiment(args.experiment_name)

    run_name = (
        f"rf_len{args.length}_ws{args.window_size}_trees{args.n_estimators}"
        f"_depth{args.max_depth}_noise{args.noise_std}"
    )

    with mlflow.start_run(run_name=run_name):
        # Log high-level config
        mlflow.log_param("length", args.length)
        mlflow.log_param("years", args.years)
        mlflow.log_param("noise_std", args.noise_std)
        mlflow.log_param("window_size", args.window_size)
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth)
        mlflow.log_param("min_samples_leaf", args.min_samples_leaf)
        mlflow.log_param("train_frac", args.train_frac)
        mlflow.log_param("val_frac", args.val_frac)
        mlflow.log_param("seed", args.seed)
        mlflow.set_tag("data.synthetic", "simplified_weather_v1")
        mlflow.set_tag("model.family", "RandomForestRegressor")

        # Generate signal
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

        # Lagged features and splits
        X, y = create_lagged_features(series, window_size=args.window_size)
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = train_val_test_split_time(
            X, y, train_frac=args.train_frac, val_frac=args.val_frac
        )

        # Model
        rf = RandomForestRegressor(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
            random_state=args.seed,
            n_jobs=-1,
        )

        rf.fit(X_train, y_train)

        # Evaluation on test
        y_pred_test = rf.predict(X_test)
        metrics = compute_regression_metrics(y_test, y_pred_test)

        for k, v in metrics.items():
            mlflow.log_metric(f"test_{k}", float(v))

        # Also look at validation metrics (optional)
        y_pred_val = rf.predict(X_val)
        val_metrics = compute_regression_metrics(y_val, y_pred_val)
        for k, v in val_metrics.items():
            mlflow.log_metric(f"val_{k}", float(v))

        # Log model using MLflow's sklearn flavor
        mlflow.sklearn.log_model(rf, artifact_path="model")

        # Plot predictions vs truth for test segment
        project_root = Path(__file__).resolve().parents[1]
        plots_dir = project_root / "result - export"
        plot_path = plots_dir / f"{run_name}_test_predictions.png"
        plot_predictions_time(
            y_true=y_test,
            y_pred=y_pred_test,
            title="RandomForest Test Predictions vs True",
            out_path=plot_path,
            max_points=400,
        )

        # Log the plot as MLflow artifact
        mlflow.log_artifact(str(plot_path), artifact_path="plots")

        # Basic run summary to stdout
        print("=== Test metrics ===")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()
