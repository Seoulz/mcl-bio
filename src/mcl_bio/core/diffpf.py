"""Differentiable particle filter (soft resampling)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from mcl_bio.core.bootstrap import BootstrapPF
from mcl_bio.core.filter import PFResult
from mcl_bio.core.resampling import ResamplingMethod, resample


class DifferentiablePF(BootstrapPF):
    """Particle filter with soft resampling for gradient-based learning.

    Requires PyTorch for training helpers; inference uses NumPy backend.
    """

    def __init__(
        self,
        *args: object,
        temperature: float = 0.5,
        always_soft_resample: bool = True,
        **kwargs: object,
    ) -> None:
        super().__init__(
            *args,
            resampling_method=ResamplingMethod.SOFT,
            **kwargs,  # type: ignore[arg-type]
        )
        self.temperature = temperature
        self.always_soft_resample = always_soft_resample

    def resample(self) -> PFResult:
        if self.particles is None or self.log_weights is None:
            raise RuntimeError("Filter not initialized")

        if self.always_soft_resample:
            self.particles, self.log_weights = resample(
                self.particles,
                self.log_weights,
                method=ResamplingMethod.SOFT,
                temperature=self.temperature,
            )

        result = self.estimate()
        result.metadata["soft_resample"] = True
        result.metadata["temperature"] = self.temperature
        return result


def train_observation_noise(
    observations: NDArray[np.floating],
    true_states: NDArray[np.floating],
    initial_noise: float = 1.0,
    steps: int = 100,
    lr: float = 0.05,
) -> float:
    """Learn observation noise scale via PyTorch (requires [diff] extra)."""
    import torch
    from torch import nn

    obs = torch.tensor(observations, dtype=torch.float32)
    states = torch.tensor(true_states, dtype=torch.float32)
    log_noise = nn.Parameter(torch.tensor(np.log(initial_noise), dtype=torch.float32))
    optimizer = torch.optim.Adam([log_noise], lr=lr)

    for _ in range(steps):
        noise = torch.exp(log_noise)
        predicted = states
        loss = 0.5 * torch.mean((obs - predicted) ** 2 / (noise**2) + 2 * log_noise)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return float(torch.exp(log_noise).detach().item())
