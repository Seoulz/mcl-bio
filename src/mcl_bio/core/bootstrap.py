"""Bootstrap (SIR) particle filter."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from mcl_bio.core.filter import ParticleFilter, PFResult
from mcl_bio.core.resampling import ResamplingMethod, resample


class BootstrapPF(ParticleFilter):
    """Standard bootstrap particle filter with configurable resampling."""

    def __init__(
        self,
        *args: object,
        resample_threshold: float = 0.5,
        resampling_method: ResamplingMethod = ResamplingMethod.SYSTEMATIC,
        **kwargs: object,
    ) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self.resample_threshold = resample_threshold
        self.resampling_method = resampling_method

    def predict(self, control: NDArray[np.floating] | None = None) -> None:
        if self.particles is None:
            raise RuntimeError("Filter not initialized")
        self.particles = self.motion_model.sample(self.particles, control, self.rng)

    def update(self, observation: NDArray[np.floating]) -> None:
        if self.particles is None or self.log_weights is None:
            raise RuntimeError("Filter not initialized")
        ll = self.observation_model.log_prob(self.particles, observation)
        self.log_weights = self.log_weights + ll

    def resample(self) -> PFResult:
        if self.particles is None or self.log_weights is None:
            raise RuntimeError("Filter not initialized")

        from mcl_bio.core.resampling import effective_sample_size, normalize_weights

        weights = normalize_weights(self.log_weights)
        neff = effective_sample_size(weights)

        if neff < self.resample_threshold * self.num_particles:
            self.particles, self.log_weights = resample(
                self.particles,
                self.log_weights,
                method=self.resampling_method,
                rng=self.rng,
            )

        result = self.estimate()
        result.metadata["resampled"] = neff < self.resample_threshold * self.num_particles
        return result
