"""Example models and data loaders."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from mcl_bio.core.models import (
    BeaconObservationModel,
    GaussianMotionModel,
    GaussianObservationModel,
    MecanumMotionModel,
)


def load_vex_odometry(path: Path | str) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Load VEX odometry CSV with columns: timestamp, vx, vy, omega."""
    import csv

    path = Path(path)
    controls: list[list[float]] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            controls.append([float(row["vx"]), float(row["vy"]), float(row["omega"])])
    controls_arr = np.asarray(controls, dtype=float)
    return controls_arr, controls_arr


def simulate_vex_run(
    num_steps: int = 100,
    rng: np.random.Generator | None = None,
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """Simulate a VEX localization run with beacon observations."""
    rng = rng or np.random.default_rng(42)
    motion = MecanumMotionModel(process_noise=0.03)
    beacons = np.array([[2.0, 0.0], [0.0, 2.0], [-2.0, 0.0], [0.0, -2.0]])
    obs_model = BeaconObservationModel(beacons, range_noise=0.08, bearing_noise=0.04)

    state = np.array([0.0, 0.0, 0.0])
    true_states = [state.copy()]
    controls = []
    observations = []

    for _ in range(num_steps):
        control = rng.normal([0.3, 0.0, 0.05], [0.1, 0.05, 0.02])
        controls.append(control)
        state = motion.sample(state.reshape(1, -1), control, rng)[0]
        true_states.append(state.copy())
        obs = obs_model._predict(state.reshape(1, -1))[0]
        obs[:4] += rng.normal(0, 0.08, 4)
        obs[4:] += rng.normal(0, 0.04, 4)
        observations.append(obs)

    return (
        np.asarray(true_states[1:]),
        np.asarray(controls),
        np.asarray(observations),
    )


def simulate_ppg(
    duration: float = 10.0,
    sample_rate: float = 100.0,
    heart_rate_bpm: float = 72.0,
    rng: np.random.Generator | None = None,
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """Simulate noisy PPG with latent heart rate state."""
    rng = rng or np.random.default_rng(7)
    n = int(duration * sample_rate)
    t = np.linspace(0, duration, n, endpoint=False)
    hr_hz = heart_rate_bpm / 60.0
    latent_hr = heart_rate_bpm + 3 * np.sin(2 * np.pi * 0.05 * t)
    signal = np.sin(2 * np.pi * hr_hz * t)
    signal += 0.3 * np.sin(2 * np.pi * 2 * hr_hz * t)
    noise = rng.normal(0, 0.15, n)
    observed = signal + noise
    return t, observed, latent_hr


def simulate_1d_run(
    num_steps: int = 60,
    rng: np.random.Generator | None = None,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Simulate a linear-Gaussian 1D constant-velocity track.

    Returns (true_states (n, 2), observations (n, 1)) so both the particle filter
    and the Kalman filter can be evaluated on identical data.
    """
    rng = rng or np.random.default_rng(0)
    motion, obs_model = make_1d_tracking_models()
    state = np.array([0.0, 1.0])
    true_states = []
    observations = []
    for _ in range(num_steps):
        state = motion.sample(state.reshape(1, -1), rng=rng)[0]
        obs = obs_model.sample(state.reshape(1, -1), rng=rng)[0]
        true_states.append(state.copy())
        observations.append(obs)
    return np.asarray(true_states), np.asarray(observations)


def make_1d_tracking_models() -> tuple[GaussianMotionModel, GaussianObservationModel]:
    """Simple 1D constant-velocity models for quickstart."""
    dt = 0.1
    motion = GaussianMotionModel(
        transition=np.array([[1.0, dt], [0.0, 1.0]]),
        process_noise=np.array([0.01, 0.05]),
    )
    observation = GaussianObservationModel(
        observation_matrix=np.array([[1.0, 0.0]]),
        observation_noise=0.2,
    )
    return motion, observation
