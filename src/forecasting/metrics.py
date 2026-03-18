from __future__ import annotations

import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(eps, (np.abs(y_true) + np.abs(y_pred)) / 2.0)
    return float(np.mean(np.abs(y_true - y_pred) / denom))


def mase(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_insample: np.ndarray,
    season_length: int,
    eps: float = 1e-8,
) -> float:
    """
    Mean Absolute Scaled Error using seasonal naive scaling.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_insample = np.asarray(y_insample, dtype=float)

    if len(y_insample) <= season_length:
        scale = np.mean(np.abs(np.diff(y_insample))) if len(y_insample) > 1 else 0.0
    else:
        scale = np.mean(np.abs(y_insample[season_length:] - y_insample[:-season_length]))

    scale = max(eps, float(scale))
    return float(np.mean(np.abs(y_true - y_pred)) / scale)

