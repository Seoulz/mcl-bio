"""Tests for motion and observation models."""

import numpy as np

from mcl_bio.core.models import (
    BeaconObservationModel,
    GaussianMotionModel,
    GaussianObservationModel,
    MecanumMotionModel,
    wrap_angle,
)


def test_gaussian_motion_sample_shape() -> None:
    motion = GaussianMotionModel(transition=np.eye(2), process_noise=0.1)
    states = np.zeros((10, 2))
    next_states = motion.sample(states, rng=np.random.default_rng(0))
    assert next_states.shape == (10, 2)


def test_gaussian_observation_log_prob() -> None:
    obs = GaussianObservationModel(observation_matrix=np.array([[1.0, 0.0]]))
    states = np.array([[1.0, 0.0], [2.0, 0.0]])
    ll = obs.log_prob(states, np.array([1.0]))
    assert ll[0] > ll[1]


def test_mecanum_motion_updates_position() -> None:
    motion = MecanumMotionModel()
    state = np.array([[0.0, 0.0, 0.0]])
    control = np.array([0.5, 0.0, 0.0])
    next_state = motion.sample(state, control, rng=np.random.default_rng(0))
    assert next_state[0, 0] > 0


def test_beacon_observation_higher_likelihood_at_true_state() -> None:
    beacons = np.array([[1.0, 0.0]])
    obs_model = BeaconObservationModel(beacons)
    true = np.array([[0.0, 0.0, 0.0]])
    wrong = np.array([[2.0, 2.0, 0.0]])
    measurement = obs_model._predict(true)[0]
    ll_true = obs_model.log_prob(true, measurement)
    ll_wrong = obs_model.log_prob(wrong, measurement)
    assert ll_true[0] > ll_wrong[0]


def test_wrap_angle() -> None:
    assert np.isclose(wrap_angle(np.array([3.5]))[0], wrap_angle(np.array([3.5 - 2 * np.pi]))[0])
