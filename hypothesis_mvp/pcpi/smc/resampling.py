"""Weight diagnostics and unbiased systematic resampling."""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp


def normalize_log_weights(log_weights: np.ndarray) -> tuple[np.ndarray, float]:
    values = np.asarray(log_weights, dtype=float).reshape(-1)
    if not len(values) or np.isnan(values).any() or np.all(values == -np.inf):
        raise ValueError("log weights must be non-empty and must not be NaN or all degenerate")
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


def _validated_resampling_probabilities(
    weights: np.ndarray,
    sample_count: int,
) -> np.ndarray:
    probabilities = np.asarray(weights, dtype=float).reshape(-1)
    if (
        not len(probabilities)
        or np.any(probabilities < 0)
        or not np.all(np.isfinite(probabilities))
    ):
        raise ValueError("resampling weights must be finite and non-negative")
    if sample_count < 1:
        raise ValueError("resampling sample_count must be positive")
    total = float(probabilities.sum())
    if total <= 0:
        raise ValueError("resampling weights must have positive mass")
    return probabilities / total


def systematic_resample_count(
    weights: np.ndarray,
    sample_count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw an arbitrary-size unbiased systematic population."""

    probabilities = _validated_resampling_probabilities(weights, sample_count)
    cumulative = np.cumsum(probabilities)
    cumulative[-1] = 1.0
    positions = (rng.random() + np.arange(sample_count)) / sample_count
    return np.searchsorted(cumulative, positions, side="right").astype(int)


def systematic_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return systematic_resample_count(weights, len(np.asarray(weights).reshape(-1)), rng)


def stratified_resample_count(
    weights: np.ndarray,
    sample_count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw an arbitrary-size unbiased stratified population."""

    probabilities = _validated_resampling_probabilities(weights, sample_count)
    cumulative = np.cumsum(probabilities)
    cumulative[-1] = 1.0
    positions = (np.arange(sample_count) + rng.random(sample_count)) / sample_count
    return np.searchsorted(cumulative, positions, side="right").astype(int)


def stratified_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return stratified_resample_count(weights, len(np.asarray(weights).reshape(-1)), rng)


def residual_resample_count(
    weights: np.ndarray,
    sample_count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw an arbitrary-size unbiased residual population."""

    probabilities = _validated_resampling_probabilities(weights, sample_count)
    expected = sample_count * probabilities
    deterministic = np.floor(expected).astype(int)
    residual_count = sample_count - int(deterministic.sum())
    indices = np.repeat(np.arange(len(probabilities), dtype=int), deterministic)
    if residual_count:
        residual_mass = expected - deterministic
        residual_total = float(residual_mass.sum())
        if residual_total <= 0.0:
            raise ValueError("residual resampling has invalid residual mass")
        residual_probabilities = residual_mass / residual_total
        cumulative = np.cumsum(residual_probabilities)
        cumulative[-1] = 1.0
        positions = (rng.random(residual_count) + np.arange(residual_count)) / residual_count
        residual_indices = np.searchsorted(cumulative, positions, side="right").astype(int)
        indices = np.concatenate((indices, residual_indices))
    if len(indices) != sample_count:
        raise AssertionError("residual resampling returned the wrong population size")
    return indices[rng.permutation(sample_count)]


def residual_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return residual_resample_count(weights, len(np.asarray(weights).reshape(-1)), rng)
