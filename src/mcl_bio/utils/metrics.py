"""Evaluation metrics."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def rmse(estimated: NDArray[np.floating], ground_truth: NDArray[np.floating]) -> float:
    """Root mean squared error."""
    return float(np.sqrt(np.mean((estimated - ground_truth) ** 2)))


def position_error(
    estimated: NDArray[np.floating],
    ground_truth: NDArray[np.floating],
) -> float:
    """2D position RMSE (first two state dimensions)."""
    return rmse(estimated[..., :2], ground_truth[..., :2])


def neff_from_log_weights(log_weights: NDArray[np.floating]) -> float:
    """Effective sample size from log weights."""
    from mcl_bio.core.resampling import effective_sample_size, normalize_weights

    return effective_sample_size(normalize_weights(log_weights))
