"""Benchmark the bootstrap particle filter against the optimal Kalman filter."""

from __future__ import annotations

import numpy as np

from mcl_bio.baselines.kalman import KalmanFilter
from mcl_bio.core.bootstrap import BootstrapPF
from mcl_bio.examples import make_1d_tracking_models, simulate_1d_run


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def benchmark_pf_vs_kalman(
    num_steps: int = 80,
    num_particles: int = 500,
    seed: int = 0,
) -> dict[str, float]:
    """Run PF and KF on the same linear-Gaussian track and compare position RMSE.

    On a linear-Gaussian system the Kalman filter is optimal, so a correct PF
    should land within a few percent of it. This validates the PF implementation.
    """
    rng = np.random.default_rng(seed)
    true_states, observations = simulate_1d_run(num_steps, rng=rng)
    motion, obs_model = make_1d_tracking_models()
    dt = 0.1

    kf = KalmanFilter(
        transition=np.array([[1.0, dt], [0.0, 1.0]]),
        observation=np.array([[1.0, 0.0]]),
        process_cov=np.diag([0.01**2, 0.05**2]),
        observation_cov=np.array([[0.2**2]]),
    )
    kf.initialize(np.array([0.0, 1.0]), cov=np.eye(2) * 0.25)
    kf_est = np.array([kf.step(z) for z in observations])

    pf = BootstrapPF(motion, obs_model, num_particles=num_particles, rng=rng)
    pf.initialize(np.array([0.0, 1.0]) + rng.normal(0, 0.5, size=(num_particles, 2)))
    pf_est = np.array([pf.step(z).mean for z in observations])

    kf_rmse = _rmse(kf_est[:, 0], true_states[:, 0])
    pf_rmse = _rmse(pf_est[:, 0], true_states[:, 0])
    return {
        "kalman_rmse": kf_rmse,
        "particle_filter_rmse": pf_rmse,
        "pf_over_kf_ratio": pf_rmse / kf_rmse if kf_rmse > 0 else float("nan"),
        "num_particles": float(num_particles),
        "num_steps": float(num_steps),
    }
