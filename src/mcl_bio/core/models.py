"""Motion and observation model abstractions."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray


class MotionModel(ABC):
    """State transition model p(x_t | x_{t-1}, u_t)."""

    @property
    @abstractmethod
    def state_dim(self) -> int:
        """Dimensionality of the state vector."""

    @abstractmethod
    def sample(
        self,
        states: NDArray[np.floating],
        control: NDArray[np.floating] | None = None,
        rng: np.random.Generator | None = None,
    ) -> NDArray[np.floating]:
        """Propagate particles through the motion model."""

    @abstractmethod
    def log_prob(
        self,
        prev_states: NDArray[np.floating],
        next_states: NDArray[np.floating],
        control: NDArray[np.floating] | None = None,
    ) -> NDArray[np.floating]:
        """Log probability of transitions."""


class ObservationModel(ABC):
    """Measurement model p(z_t | x_t)."""

    @abstractmethod
    def log_prob(
        self,
        states: NDArray[np.floating],
        observation: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Log likelihood of observation given each particle state."""

    def sample(
        self,
        states: NDArray[np.floating],
        rng: np.random.Generator | None = None,
    ) -> NDArray[np.floating]:
        """Optional forward sampling for simulation."""
        raise NotImplementedError("This observation model does not support sampling.")


class GaussianMotionModel(MotionModel):
    """Linear-Gaussian motion: x' = A x + B u + noise."""

    def __init__(
        self,
        transition: NDArray[np.floating],
        control_matrix: NDArray[np.floating] | None = None,
        process_noise: float | NDArray[np.floating] = 0.1,
    ) -> None:
        self._a = np.asarray(transition, dtype=float)
        self._b = np.asarray(control_matrix, dtype=float) if control_matrix is not None else None
        self._noise: float | None
        self._noise_vec: NDArray[np.floating] | None
        if np.ndim(process_noise) == 0:
            self._noise = float(process_noise)
            self._noise_vec = None
        else:
            self._noise = None
            self._noise_vec = np.asarray(process_noise, dtype=float)

    @property
    def state_dim(self) -> int:
        return int(self._a.shape[0])

    def _noise_scale(self) -> NDArray[np.floating]:
        if self._noise_vec is not None:
            return self._noise_vec
        return np.full(self.state_dim, self._noise or 0.1)

    def sample(
        self,
        states: NDArray[np.floating],
        control: NDArray[np.floating] | None = None,
        rng: np.random.Generator | None = None,
    ) -> NDArray[np.floating]:
        rng = rng or np.random.default_rng()
        mean = states @ self._a.T
        if control is not None and self._b is not None:
            mean = mean + control @ self._b.T
        noise = rng.normal(0.0, self._noise_scale(), size=mean.shape)
        return mean + noise

    def log_prob(
        self,
        prev_states: NDArray[np.floating],
        next_states: NDArray[np.floating],
        control: NDArray[np.floating] | None = None,
    ) -> NDArray[np.floating]:
        mean = prev_states @ self._a.T
        if control is not None and self._b is not None:
            mean = mean + control @ self._b.T
        diff = next_states - mean
        var = self._noise_scale() ** 2
        return -0.5 * np.sum(diff**2 / var + np.log(2 * np.pi * var), axis=1)


class GaussianObservationModel(ObservationModel):
    """Linear-Gaussian observation: z = H x + noise."""

    def __init__(
        self,
        observation_matrix: NDArray[np.floating],
        observation_noise: float | NDArray[np.floating] = 0.5,
    ) -> None:
        self._h = np.asarray(observation_matrix, dtype=float)
        self._noise: float | None
        self._noise_vec: NDArray[np.floating] | None
        if np.ndim(observation_noise) == 0:
            self._noise = float(observation_noise)
            self._noise_vec = None
        else:
            self._noise = None
            self._noise_vec = np.asarray(observation_noise, dtype=float)

    def _noise_scale(self) -> NDArray[np.floating]:
        if self._noise_vec is not None:
            return self._noise_vec
        obs_dim = self._h.shape[0]
        return np.full(obs_dim, self._noise or 0.5)

    def log_prob(
        self,
        states: NDArray[np.floating],
        observation: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        predicted = states @ self._h.T
        diff = observation - predicted
        var = self._noise_scale() ** 2
        return -0.5 * np.sum(diff**2 / var + np.log(2 * np.pi * var), axis=1)

    def sample(
        self,
        states: NDArray[np.floating],
        rng: np.random.Generator | None = None,
    ) -> NDArray[np.floating]:
        rng = rng or np.random.default_rng()
        mean = states @ self._h.T
        noise = rng.normal(0.0, self._noise_scale(), size=mean.shape)
        return mean + noise


def wrap_angle(angle: NDArray[np.floating]) -> NDArray[np.floating]:
    """Wrap angles to [-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


class MecanumMotionModel(MotionModel):
    """Planar mecanum drive motion model for VEX-style localization."""

    def __init__(self, process_noise: float = 0.05) -> None:
        self._process_noise = process_noise

    @property
    def state_dim(self) -> int:
        return 3  # x, y, theta

    def sample(
        self,
        states: NDArray[np.floating],
        control: NDArray[np.floating] | None = None,
        rng: np.random.Generator | None = None,
    ) -> NDArray[np.floating]:
        rng = rng or np.random.default_rng()
        if control is None:
            control = np.zeros(3)
        vx, vy, omega = control
        x, y, theta = states[:, 0], states[:, 1], states[:, 2]
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        dx = (vx * cos_t - vy * sin_t) + rng.normal(0, self._process_noise, size=states.shape[0])
        dy = (vx * sin_t + vy * cos_t) + rng.normal(0, self._process_noise, size=states.shape[0])
        dtheta = omega + rng.normal(0, self._process_noise * 0.5, size=states.shape[0])
        next_states = np.column_stack([x + dx, y + dy, wrap_angle(theta + dtheta)])
        return next_states

    def log_prob(
        self,
        prev_states: NDArray[np.floating],
        next_states: NDArray[np.floating],
        control: NDArray[np.floating] | None = None,
    ) -> NDArray[np.floating]:
        if control is None:
            control = np.zeros(3)
        vx, vy, omega = control
        x, y, theta = prev_states[:, 0], prev_states[:, 1], prev_states[:, 2]
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        mean_x = x + (vx * cos_t - vy * sin_t)
        mean_y = y + (vx * sin_t + vy * cos_t)
        mean_theta = wrap_angle(theta + omega)
        diff = np.column_stack(
            [
                next_states[:, 0] - mean_x,
                next_states[:, 1] - mean_y,
                wrap_angle(next_states[:, 2] - mean_theta),
            ]
        )
        var = self._process_noise**2
        return -0.5 * np.sum(diff**2 / var + np.log(2 * np.pi * var), axis=1)


class BeaconObservationModel(ObservationModel):
    """Range-and-bearing observations to fixed beacons."""

    def __init__(
        self,
        beacons: NDArray[np.floating],
        range_noise: float = 0.1,
        bearing_noise: float = 0.05,
    ) -> None:
        self.beacons = np.asarray(beacons, dtype=float)
        self.range_noise = range_noise
        self.bearing_noise = bearing_noise

    def _predict(self, states: NDArray[np.floating]) -> NDArray[np.floating]:
        x, y, theta = states[:, 0:1], states[:, 1:2], states[:, 2:3]
        dx = self.beacons[None, :, 0] - x
        dy = self.beacons[None, :, 1] - y
        ranges = np.hypot(dx, dy)
        bearings = wrap_angle(np.arctan2(dy, dx) - theta)
        return np.hstack([ranges, bearings])

    def log_prob(
        self,
        states: NDArray[np.floating],
        observation: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        predicted = self._predict(states)
        obs = np.broadcast_to(observation, predicted.shape)
        n_beacons = self.beacons.shape[0]
        range_diff = obs[:, :n_beacons] - predicted[:, :n_beacons]
        bearing_diff = wrap_angle(obs[:, n_beacons:] - predicted[:, n_beacons:])
        range_ll = -0.5 * (range_diff**2) / (self.range_noise**2)
        bearing_ll = -0.5 * (bearing_diff**2) / (self.bearing_noise**2)
        return np.sum(range_ll, axis=1) + np.sum(bearing_ll, axis=1)
