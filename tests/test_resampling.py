"""Tests for resampling strategies."""

import numpy as np

from mcl_bio.core.resampling import (
    ResamplingMethod,
    effective_sample_size,
    normalize_weights,
    resample,
    systematic_resample,
)


def test_normalize_weights_sums_to_one() -> None:
    log_w = np.array([0.0, 1.0, 2.0])
    w = normalize_weights(log_w)
    assert np.isclose(np.sum(w), 1.0)


def test_systematic_resample_length() -> None:
    log_w = np.log(np.array([0.1, 0.2, 0.3, 0.4]))
    idx = systematic_resample(log_w, rng=np.random.default_rng(0))
    assert len(idx) == 4


def test_effective_sample_size_uniform() -> None:
    w = np.ones(100) / 100
    assert np.isclose(effective_sample_size(w), 100.0)


def test_soft_resample_preserves_particles() -> None:
    particles = np.array([[1.0], [2.0], [3.0]])
    log_w = np.log(np.array([0.2, 0.3, 0.5]))
    out_p, out_w = resample(particles, log_w, method=ResamplingMethod.SOFT)
    assert out_p.shape == particles.shape
    assert len(out_w) == 3
