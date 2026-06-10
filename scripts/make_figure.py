"""Generate the README figure: 1D PF-vs-Kalman tracking + 2D VEX localization.

Run with:  PYTHONPATH=src python scripts/make_figure.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mcl_bio.baselines.kalman import KalmanFilter
from mcl_bio.core.bootstrap import BootstrapPF
from mcl_bio.core.models import BeaconObservationModel, MecanumMotionModel
from mcl_bio.examples import make_1d_tracking_models, simulate_1d_run, simulate_vex_run


def main() -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # ---- 1D: particle filter vs optimal Kalman filter ----
    rng = np.random.default_rng(0)
    true_states, observations = simulate_1d_run(80, rng=rng)
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

    pf = BootstrapPF(motion, obs_model, num_particles=500, rng=rng)
    pf.initialize(np.array([0.0, 1.0]) + rng.normal(0, 0.5, size=(500, 2)))
    pf_est = np.array([pf.step(z).mean for z in observations])

    t = np.arange(len(observations))
    ax1.scatter(t, observations[:, 0], s=12, color="#bbb", label="noisy obs")
    ax1.plot(t, true_states[:, 0], "k-", lw=2, label="true position")
    ax1.plot(t, kf_est[:, 0], color="#d1495b", lw=1.5, label="Kalman (optimal)")
    ax1.plot(t, pf_est[:, 0], color="#2a9d8f", lw=1.5, ls="--", label="particle filter")
    ax1.set_xlabel("time step")
    ax1.set_ylabel("position")
    ax1.set_title("1D tracking: PF matches the optimal Kalman filter")
    ax1.legend(fontsize=8)

    # ---- 2D: nonlinear VEX beacon localization ----
    true_v, controls, obs_v = simulate_vex_run(num_steps=60, rng=np.random.default_rng(1))
    beacons = np.array([[2.0, 0.0], [0.0, 2.0], [-2.0, 0.0], [0.0, -2.0]])
    pf2 = BootstrapPF(
        MecanumMotionModel(), BeaconObservationModel(beacons), num_particles=600,
        rng=np.random.default_rng(2),
    )
    pf2.initialize(true_v[0] + np.random.default_rng(3).normal(0, 0.3, size=(600, 3)))
    est = np.array([pf2.step(obs_v[i], controls[i]).mean for i in range(len(obs_v))])

    ax2.plot(true_v[:, 0], true_v[:, 1], "k-", lw=2, label="true path")
    ax2.plot(est[:, 0], est[:, 1], color="#2a9d8f", lw=1.5, ls="--", label="PF estimate")
    ax2.scatter(beacons[:, 0], beacons[:, 1], marker="^", s=120, color="#e76f51", label="beacons")
    ax2.set_xlabel("x (m)")
    ax2.set_ylabel("y (m)")
    ax2.set_title("VEX localization: range-bearing beacons (nonlinear)")
    ax2.legend(fontsize=8)
    ax2.set_aspect("equal", adjustable="datalim")

    out = Path(__file__).resolve().parents[1] / "assets" / "tracking.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
