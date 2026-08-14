"""Numerically safe regression metrics with no backend fallback."""

from __future__ import annotations

import numpy as np


def _finite_pairs(
    y_true: np.ndarray, y_pred: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    truth = np.asarray(y_true, dtype=float).reshape(-1)
    prediction = np.asarray(y_pred, dtype=float).reshape(-1)
    if len(truth) != len(prediction):
        raise ValueError("metric inputs must be aligned")
    mask = np.isfinite(truth) & np.isfinite(prediction)
    return truth[mask], prediction[mask]


def _regression_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> tuple[float, float]:
    truth, prediction = _finite_pairs(y_true, y_pred)
    if not len(truth):
        return float("inf"), float("-inf")
    residual_sum = float(np.sum((truth - prediction) ** 2))
    total_sum = float(np.sum((truth - np.mean(truth)) ** 2))
    mse = residual_sum / len(truth)
    r2 = 1.0 - residual_sum / max(total_sum, 1.0e-12)
    return mse, r2


def compute_best_95_nmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    truth, prediction = _finite_pairs(y_true, y_pred)
    if not len(truth):
        return float("inf")
    return float(np.mean((truth - prediction) ** 2) / max(float(np.var(truth)), 1.0e-12))


def evaluate_predictions(
    y_true: np.ndarray, y_pred: np.ndarray,
    extra_true: np.ndarray | None, extra_pred: np.ndarray | None,
) -> dict[str, float]:
    mse, r2 = _regression_metrics(y_true, y_pred)
    metrics = {"mse_val": mse, "r2_val": r2}
    if extra_true is not None and extra_pred is not None:
        mse_extra, r2_extra = _regression_metrics(extra_true, extra_pred)
        metrics.update({"mse_extra": mse_extra, "r2_extra": r2_extra})
    return metrics


__all__ = ["compute_best_95_nmse", "evaluate_predictions"]
