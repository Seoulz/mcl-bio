"""Auxiliary particle filter."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from mcl_bio.core.bootstrap import BootstrapPF
from mcl_bio.core.filter import PFResult
from mcl_bio.core.resampling import normalize_weights, systematic_resample


class AuxiliaryPF(BootstrapPF):
    """Auxiliary particle filter using look-ahead observations."""

    def step(
        self,
        observation: NDArray[np.floating],
        control: NDArray[np.floating] | None = None,
    ) -> PFResult:
        if self.particles is None or self.log_weights is None:
            raise RuntimeError("Filter not initialized")

        # Propagate to intermediate states
        proposed = self.motion_model.sample(self.particles, control, self.rng)
        lookahead_ll = self.observation_model.log_prob(proposed, observation)
        aux_log_w = self.log_weights + lookahead_ll
        aux_w = normalize_weights(aux_log_w)

        # Sample ancestor indices
        indices = systematic_resample(np.log(aux_w + 1e-12), self.rng)
        self.particles = proposed[indices]

        # Update weights with correction term
        ll = self.observation_model.log_prob(self.particles, observation)
        self.log_weights = ll - lookahead_ll[indices]
        self.log_weights -= np.max(self.log_weights)
        return self.resample()
