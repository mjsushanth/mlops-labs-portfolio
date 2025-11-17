# src/utils.py

from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error


def create_lagged_features(
    series: pd.Series,
    window_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Turn a 1D series into lagged features for one-step-ahead forecasting.

    X[i] = [y_{i-window_size}, ..., y_{i-1}]
    y[i] = y_i
    """
    values = series.to_numpy().astype(float)
    n = len(values)

    if window_size >= n:
        raise ValueError("window_size must be smaller than series length")

    X = []
    y = []
    for t in range(window_size, n):
        X.append(values[t - window_size:t])
        y.append(values[t])

    X_arr = np.asarray(X)
    y_arr = np.asarray(y)
    return X_arr, y_arr


def train_val_test_split_time(
    X: np.ndarray,
    y: np.ndarray,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
):
    """Time-ordered split into train / val / test."""
    assert 0 < train_frac < 1 and 0 <= val_frac < 1
    assert train_frac + val_frac < 1

    n = len(X)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric Mean Absolute Percentage Error (in %). Robust to zeros."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    denominator = (np.abs(y_true) + np.abs(y_pred))
    # avoid division by zero by adding small epsilon
    epsilon = 1e-8
    smape_val = np.mean(
        2.0 * np.abs(y_pred - y_true) / (denominator + epsilon)
    ) * 100.0
    return smape_val


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute regression metrics including RMSE, MAE, R2, SMAPE."""
    rmse = root_mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    smape_val = smape(y_true, y_pred)

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "smape": smape_val,
    }


def plot_predictions_time(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    out_path: Path,
    max_points: int = 400,
) -> None:
    """Plot true vs predicted over time and save to disk."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = len(y_true)
    if max_points is not None and n > max_points:
        start = n - max_points  # show last window
        y_true = y_true[start:]
        y_pred = y_pred[start:]

    x = np.arange(len(y_true))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x, y_true, label="True", linewidth=1.0)
    ax.plot(x, y_pred, label="Predicted", linewidth=1.0)
    ax.set_title(title)
    ax.set_xlabel("Time index (test segment)")
    ax.set_ylabel("Value")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
