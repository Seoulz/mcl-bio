"""NumPy array backend utilities."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def ensure_float_array(x: NDArray | list | tuple) -> NDArray[np.floating]:
    """Convert input to float64 ndarray."""
    return np.asarray(x, dtype=float)


def weighted_mean(
    values: NDArray[np.floating],
    log_weights: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Compute weighted mean from log weights."""
    from mcl_bio.core.resampling import normalize_weights

    w = normalize_weights(log_weights)
    return np.average(values, axis=0, weights=w)
