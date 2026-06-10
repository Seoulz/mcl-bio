"""Tests for differentiable particle filter."""

import numpy as np
import pytest

from mcl_bio.core.diffpf import DifferentiablePF, train_observation_noise
from mcl_bio.examples import make_1d_tracking_models


def test_differentiable_pf_soft_resample() -> None:
    motion, obs_model = make_1d_tracking_models()
    pf = DifferentiablePF(motion, obs_model, num_particles=50, rng=np.random.default_rng(0))
    pf.initialize(np.random.default_rng(1).normal(0, 1, size=(50, 2)))
    result = pf.step(np.array([0.5]))
    assert result.metadata.get("soft_resample") is True
    assert result.neff > 0


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("torch") is None,
    reason="torch not installed",
)
def test_train_observation_noise() -> None:
    rng = np.random.default_rng(5)
    states = rng.normal(0, 1, size=(100, 1))
    noise = 0.3
    obs = states + rng.normal(0, noise, size=(100, 1))
    learned = train_observation_noise(obs, states, initial_noise=1.0, steps=50)
    assert 0.1 < learned < 1.0
