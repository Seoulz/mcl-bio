"""Linear Kalman filter baseline.

On a linear-Gaussian problem the Kalman filter is the optimal (minimum
mean-squared-error) estimator, so it is the right yardstick for checking that the
particle filters are implemented correctly: a correct bootstrap PF should track
the KF closely as the particle count grows.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class KalmanFilter:
    """Standard linear Kalman filter."""

    def __init__(
        self,
        transition: NDArray[np.floating],
        observation: NDArray[np.floating],
        process_cov: NDArray[np.floating],
        observation_cov: NDArray[np.floating],
    ) -> None:
        self.f = np.asarray(transition, dtype=float)
        self.h = np.asarray(observation, dtype=float)
        self.q = np.asarray(process_cov, dtype=float)
        self.r = np.asarray(observation_cov, dtype=float)
        self.x: NDArray[np.floating] = np.zeros(self.f.shape[0])
        self.p: NDArray[np.floating] = np.eye(self.f.shape[0])

    def initialize(
        self, mean: NDArray[np.floating], cov: NDArray[np.floating] | None = None
    ) -> None:
        self.x = np.asarray(mean, dtype=float)
        self.p = np.eye(self.f.shape[0]) if cov is None else np.asarray(cov, dtype=float)

    def predict(self) -> None:
        self.x = self.f @ self.x
        self.p = self.f @ self.p @ self.f.T + self.q

    def update(self, z: NDArray[np.floating]) -> None:
        z = np.atleast_1d(np.asarray(z, dtype=float))
        y = z - self.h @ self.x
        s = self.h @ self.p @ self.h.T + self.r
        k = self.p @ self.h.T @ np.linalg.inv(s)
        self.x = self.x + k @ y
        ident = np.eye(self.f.shape[0])
        self.p = (ident - k @ self.h) @ self.p

    def step(self, z: NDArray[np.floating]) -> NDArray[np.floating]:
        self.predict()
        self.update(z)
        return self.x.copy()
