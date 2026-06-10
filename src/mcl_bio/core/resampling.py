"""Particle resampling strategies."""

from __future__ import annotations

from enum import Enum
from typing import Literal

import numpy as np
from numpy.typing import NDArray


class ResamplingMethod(str, Enum):
    SYSTEMATIC = "systematic"
    STRATIFIED = "stratified"
    MULTINOMIAL = "multinomial"
    SOFT = "soft"


def normalize_weights(log_weights: NDArray[np.floating]) -> NDArray[np.floating]:
    """Convert log weights to normalized probabilities."""
    log_w = log_weights - np.max(log_weights)
    w = np.exp(log_w)
    total = np.sum(w)
    if total <= 0:
        return np.full_like(w, 1.0 / len(w))
    return w / total


def effective_sample_size(weights: NDArray[np.floating]) -> float:
    """Compute effective sample size."""
    w = normalize_weights(weights) if np.any(weights < 0) else weights / np.sum(weights)
    return float(1.0 / np.sum(w**2))


def systematic_resample(
    weights: NDArray[np.floating],
    rng: np.random.Generator | None = None,
) -> NDArray[np.intp]:
    """Systematic resampling (low variance)."""
    rng = rng or np.random.default_rng()
    n = len(weights)
    w = normalize_weights(weights) if np.min(weights) < 0 else weights / np.sum(weights)
    positions = (rng.random() + np.arange(n)) / n
    cumulative = np.cumsum(w)
    indices = np.searchsorted(cumulative, positions)
    return np.clip(indices, 0, n - 1).astype(np.intp)


def stratified_resample(
    weights: NDArray[np.floating],
    rng: np.random.Generator | None = None,
) -> NDArray[np.intp]:
    """Stratified resampling."""
    rng = rng or np.random.default_rng()
    n = len(weights)
    w = normalize_weights(weights) if np.min(weights) < 0 else weights / np.sum(weights)
    positions = (rng.random(n) + np.arange(n)) / n
    cumulative = np.cumsum(w)
    return np.clip(np.searchsorted(cumulative, positions), 0, n - 1).astype(np.intp)


def multinomial_resample(
    weights: NDArray[np.floating],
    rng: np.random.Generator | None = None,
) -> NDArray[np.intp]:
    """Multinomial resampling."""
    rng = rng or np.random.default_rng()
    w = normalize_weights(weights) if np.min(weights) < 0 else weights / np.sum(weights)
    return rng.choice(len(w), size=len(w), replace=True, p=w)


def soft_resample(
    log_weights: NDArray[np.floating],
    temperature: float = 1.0,
) -> NDArray[np.floating]:
    """Soft assignment weights for differentiable resampling."""
    scaled = log_weights / max(temperature, 1e-6)
    return normalize_weights(scaled)


def resample(
    particles: NDArray[np.floating],
    log_weights: NDArray[np.floating],
    method: ResamplingMethod
    | Literal["systematic", "stratified", "multinomial", "soft"] = ResamplingMethod.SYSTEMATIC,
    rng: np.random.Generator | None = None,
    temperature: float = 1.0,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Resample particles according to log weights."""
    if isinstance(method, str):
        method = ResamplingMethod(method)

    if method == ResamplingMethod.SOFT:
        soft_w = soft_resample(log_weights, temperature=temperature)
        return particles, np.log(soft_w + 1e-12)

    indices = {
        ResamplingMethod.SYSTEMATIC: systematic_resample,
        ResamplingMethod.STRATIFIED: stratified_resample,
        ResamplingMethod.MULTINOMIAL: multinomial_resample,
    }[method](log_weights, rng)

    n = len(log_weights)
    return particles[indices], np.full(n, -np.log(n))
