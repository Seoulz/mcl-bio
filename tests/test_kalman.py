"""Kalman baseline and PF-vs-KF benchmark tests."""

import numpy as np

from mcl_bio.baselines.kalman import KalmanFilter
from mcl_bio.benchmark import benchmark_pf_vs_kalman


def test_kalman_tracks_constant_velocity() -> None:
    dt = 0.1
    kf = KalmanFilter(
        transition=np.array([[1.0, dt], [0.0, 1.0]]),
        observation=np.array([[1.0, 0.0]]),
        process_cov=np.diag([1e-4, 1e-4]),
        observation_cov=np.array([[0.04]]),
    )
    kf.initialize(np.array([0.0, 1.0]))
    rng = np.random.default_rng(0)
    pos = 0.0
    errs = []
    for _ in range(50):
        pos += dt * 1.0
        z = pos + rng.normal(0, 0.2)
        est = kf.step(np.array([z]))
        errs.append(abs(est[0] - pos))
    # Filtered estimate should be much tighter than raw measurement noise (0.2).
    assert np.mean(errs) < 0.15


def test_pf_matches_kalman() -> None:
    results = benchmark_pf_vs_kalman(num_steps=80, num_particles=600, seed=1)
    # A correct PF should be within ~30% of the optimal KF on a linear problem.
    assert results["pf_over_kf_ratio"] < 1.3
    assert results["particle_filter_rmse"] > 0
