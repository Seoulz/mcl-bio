"""Tests for bootstrap particle filter."""

import numpy as np

from mcl_bio.core.bootstrap import BootstrapPF
from mcl_bio.examples import make_1d_tracking_models, simulate_vex_run
from mcl_bio.utils.metrics import position_error


def test_bootstrap_pf_reduces_error() -> None:
    motion, obs_model = make_1d_tracking_models()
    pf = BootstrapPF(motion, obs_model, num_particles=300, rng=np.random.default_rng(42))

    true_state = np.array([0.0, 0.5])
    initial = true_state + np.random.default_rng(1).normal(0, 1.0, size=(300, 2))
    pf.initialize(initial)

    errors = []
    for _ in range(20):
        true_state = motion.sample(true_state.reshape(1, -1), rng=np.random.default_rng(2))[0]
        obs = obs_model.sample(true_state.reshape(1, -1), rng=np.random.default_rng(3))[0]
        result = pf.step(obs)
        errors.append(abs(result.mean[0] - true_state[0]))

    assert np.mean(errors[-5:]) < np.mean(errors[:5])


def test_vex_simulation_produces_valid_shapes() -> None:
    states, controls, obs = simulate_vex_run(num_steps=10)
    assert states.shape == (10, 3)
    assert controls.shape == (10, 3)
    assert obs.shape == (10, 8)


def test_vex_localization_reasonable_accuracy() -> None:
    from mcl_bio.core.models import BeaconObservationModel, MecanumMotionModel

    true_states, controls, observations = simulate_vex_run(
        num_steps=30, rng=np.random.default_rng(99)
    )
    beacons = np.array([[2.0, 0.0], [0.0, 2.0], [-2.0, 0.0], [0.0, -2.0]])
    pf = BootstrapPF(
        MecanumMotionModel(),
        BeaconObservationModel(beacons),
        num_particles=500,
        rng=np.random.default_rng(100),
    )
    pf.initialize(true_states[0] + np.random.default_rng(101).normal(0, 0.5, size=(500, 3)))

    estimates = []
    for obs, ctrl in zip(observations, controls, strict=True):
        estimates.append(pf.step(obs, ctrl).mean)
    err = position_error(np.asarray(estimates), true_states)
    assert err < 0.5
