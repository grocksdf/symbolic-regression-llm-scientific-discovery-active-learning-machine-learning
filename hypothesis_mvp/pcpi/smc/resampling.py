"""Weight diagnostics and unbiased systematic resampling."""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp


def normalize_log_weights(log_weights: np.ndarray) -> tuple[np.ndarray, float]:
    values = np.asarray(log_weights, dtype=float).reshape(-1)
    if not len(values) or not np.all(np.isfinite(values)):
        raise ValueError("log weights must be finite and non-empty")
    log_normalizer = float(logsumexp(values))
    normalized = values - log_normalizer
    return normalized, log_normalizer


def effective_sample_size(log_weights: np.ndarray) -> float:
    normalized, _ = normalize_log_weights(log_weights)
    weights = np.exp(normalized)
    return float(1.0 / np.square(weights).sum())


def weight_entropy(log_weights: np.ndarray) -> float:
    normalized, _ = normalize_log_weights(log_weights)
    weights = np.exp(normalized)
    positive = weights > 0
    return float(-np.sum(weights[positive] * normalized[positive]))


def conditional_effective_sample_size(
    log_weights: np.ndarray,
    log_increment: np.ndarray,
    delta: float,
) -> float:
    """Conditional ESS for a proposed likelihood-temperature increment."""

    if not 0.0 <= delta <= 1.0:
        raise ValueError("temperature increment must lie in [0, 1]")
    normalized, _ = normalize_log_weights(log_weights)
    increments = np.asarray(log_increment, dtype=float).reshape(-1)
    if len(increments) != len(normalized) or not np.all(np.isfinite(increments)):
        raise ValueError("conditional ESS increments must be finite and aligned")
    log_numerator = 2.0 * logsumexp(normalized + delta * increments)
    log_denominator = logsumexp(normalized + 2.0 * delta * increments)
    return float(len(normalized) * np.exp(log_numerator - log_denominator))


def adaptive_temperature_delta(
    log_weights: np.ndarray,
    log_increment: np.ndarray,
    remaining: float,
    target_cess: float,
    tolerance: float,
    maximum_iterations: int = 80,
) -> tuple[float, float]:
    """Largest bridge increment whose CESS does not cross the target floor."""

    if not 0.0 < remaining <= 1.0 or not 0.0 < target_cess <= len(log_weights):
        raise ValueError("invalid adaptive tempering interval or CESS target")
    if tolerance <= 0.0 or maximum_iterations < 1:
        raise ValueError("adaptive tempering search controls must be positive")
    full_cess = conditional_effective_sample_size(log_weights, log_increment, remaining)
    if full_cess >= target_cess:
        return float(remaining), full_cess
    lower = 0.0
    upper = float(remaining)
    for _ in range(maximum_iterations):
        middle = 0.5 * (lower + upper)
        middle_cess = conditional_effective_sample_size(
            log_weights,
            log_increment,
            middle,
        )
        if middle_cess >= target_cess:
            lower = middle
        else:
            upper = middle
        if upper - lower <= tolerance:
            break
    delta = max(lower, min(tolerance, remaining))
    return delta, conditional_effective_sample_size(log_weights, log_increment, delta)


def systematic_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    probabilities = np.asarray(weights, dtype=float).reshape(-1)
    if not len(probabilities) or np.any(probabilities < 0) or not np.all(np.isfinite(probabilities)):
        raise ValueError("resampling weights must be finite and non-negative")
    total = float(probabilities.sum())
    if total <= 0:
        raise ValueError("resampling weights must have positive mass")
    probabilities = probabilities / total
    cumulative = np.cumsum(probabilities)
    cumulative[-1] = 1.0
    positions = (rng.random() + np.arange(len(probabilities))) / len(probabilities)
    return np.searchsorted(cumulative, positions, side="right").astype(int)
