"""Particle filter base class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from mcl_bio.core.models import MotionModel, ObservationModel


@dataclass
class PFResult:
    """Output from a particle filter step or run."""

    mean: NDArray[np.floating]
    cov: NDArray[np.floating]
    particles: NDArray[np.floating]
    log_weights: NDArray[np.floating]
    neff: float
    metadata: dict[str, Any] = field(default_factory=dict)


class ParticleFilter(ABC):
    """Abstract particle filter interface."""

    def __init__(
        self,
        motion_model: MotionModel,
        observation_model: ObservationModel,
        num_particles: int = 500,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.motion_model = motion_model
        self.observation_model = observation_model
        self.num_particles = num_particles
        self.rng = rng or np.random.default_rng()
        self.particles: NDArray[np.floating] | None = None
        self.log_weights: NDArray[np.floating] | None = None

    def initialize(self, initial_distribution: NDArray[np.floating]) -> None:
        """Set initial particles (N, state_dim)."""
        if initial_distribution.shape[0] != self.num_particles:
            raise ValueError(
                f"Expected {self.num_particles} particles, got {initial_distribution.shape[0]}"
            )
        self.particles = initial_distribution.copy()
        self.log_weights = np.full(self.num_particles, -np.log(self.num_particles))

    @abstractmethod
    def predict(
        self,
        control: NDArray[np.floating] | None = None,
    ) -> None:
        """Propagate particles through the motion model."""

    @abstractmethod
    def update(self, observation: NDArray[np.floating]) -> None:
        """Incorporate an observation."""

    def step(
        self,
        observation: NDArray[np.floating],
        control: NDArray[np.floating] | None = None,
    ) -> PFResult:
        """Run predict-update-resample cycle."""
        self.predict(control)
        self.update(observation)
        return self.resample()

    @abstractmethod
    def resample(self) -> PFResult:
        """Resample particles and return state estimate."""

    def estimate(self) -> PFResult:
        """Compute weighted mean and covariance from current particles."""
        if self.particles is None or self.log_weights is None:
            raise RuntimeError("Filter not initialized")

        from mcl_bio.core.resampling import effective_sample_size, normalize_weights

        weights = normalize_weights(self.log_weights)
        mean = np.average(self.particles, axis=0, weights=weights)
        diff = self.particles - mean
        cov = np.einsum("ni,n,nj->ij", diff, weights, diff)
        return PFResult(
            mean=mean,
            cov=cov,
            particles=self.particles.copy(),
            log_weights=self.log_weights.copy(),
            neff=effective_sample_size(weights),
        )

    def run(
        self,
        observations: NDArray[np.floating],
        controls: NDArray[np.floating] | None = None,
        initial_particles: NDArray[np.floating] | None = None,
    ) -> list[PFResult]:
        """Run filter over a sequence of observations."""
        if initial_particles is not None:
            self.initialize(initial_particles)
        if self.particles is None:
            raise RuntimeError("Filter must be initialized before run()")

        results: list[PFResult] = []
        for t, obs in enumerate(observations):
            control = None if controls is None else controls[t]
            results.append(self.step(obs, control))
        return results
