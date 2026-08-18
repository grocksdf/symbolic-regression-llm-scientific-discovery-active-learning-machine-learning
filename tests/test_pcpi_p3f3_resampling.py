"""Correctness-only tests for the registered unbiased resamplers."""

from __future__ import annotations

import numpy as np
import pytest

from hypothesis_mvp.pcpi.open_target import OpenTargetParticleConfig
from hypothesis_mvp.pcpi.smc import (
    residual_resample,
    stratified_resample,
    systematic_resample,
)


@pytest.mark.parametrize(
    "resampler",
    [systematic_resample, stratified_resample, residual_resample],
)
def test_registered_resampler_preserves_population_and_support(resampler) -> None:
    weights = np.asarray([0.05, 0.15, 0.30, 0.50], dtype=float)
    indices = resampler(weights, np.random.default_rng(2026081712))
    assert indices.shape == (len(weights),)
    assert np.issubdtype(indices.dtype, np.integer)
    assert np.all((indices >= 0) & (indices < len(weights)))


@pytest.mark.parametrize(
    "resampler",
    [stratified_resample, residual_resample],
)
def test_registered_resampler_is_unbiased_over_fixed_seed_replicates(resampler) -> None:
    weights = np.asarray([0.05, 0.15, 0.30, 0.50], dtype=float)
    counts = np.zeros(len(weights), dtype=float)
    replicates = 2000
    for seed in range(replicates):
        indices = resampler(weights, np.random.default_rng(seed))
        counts += np.bincount(indices, minlength=len(weights))
    frequencies = counts / (replicates * len(weights))
    np.testing.assert_allclose(frequencies, weights, atol=0.012, rtol=0.0)


def test_particle_config_rejects_unregistered_resampler() -> None:
    with pytest.raises(ValueError, match="systematic, stratified, or residual"):
        OpenTargetParticleConfig(resampling_kind="multinomial")
