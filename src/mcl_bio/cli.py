"""Command-line demo entry point."""

from __future__ import annotations

import argparse

import numpy as np

from mcl_bio.core.bootstrap import BootstrapPF
from mcl_bio.examples import make_1d_tracking_models, simulate_vex_run


def run_quickstart() -> None:
    """Run a 1D tracking demo."""
    motion, obs_model = make_1d_tracking_models()
    pf = BootstrapPF(motion, obs_model, num_particles=200, rng=np.random.default_rng(0))

    true_state = np.array([0.0, 1.0])
    initial = true_state + np.random.default_rng(1).normal(0, 0.5, size=(200, 2))
    pf.initialize(initial)

    print("mcl-bio quickstart: 1D constant-velocity tracking")
    for step in range(10):
        true_state = motion.sample(true_state.reshape(1, -1))[0]
        obs = obs_model.sample(true_state.reshape(1, -1))[0]
        result = pf.step(obs)
        print(
            f"  step {step + 1:2d}: true={true_state[0]:+.3f} "
            f"est={result.mean[0]:+.3f} neff={result.neff:.0f}"
        )


def run_vex_demo() -> None:
    """Run VEX localization demo."""
    from mcl_bio.core.models import BeaconObservationModel, MecanumMotionModel

    true_states, controls, observations = simulate_vex_run(num_steps=50)
    beacons = np.array([[2.0, 0.0], [0.0, 2.0], [-2.0, 0.0], [0.0, -2.0]])
    motion = MecanumMotionModel()
    obs_model = BeaconObservationModel(beacons)
    pf = BootstrapPF(motion, obs_model, num_particles=400, rng=np.random.default_rng(3))

    initial = true_states[0] + np.random.default_rng(4).normal(0, 0.3, size=(400, 3))
    pf.initialize(initial)

    print("mcl-bio VEX localization demo")
    estimates = []
    for obs, ctrl in zip(observations, controls, strict=True):
        result = pf.step(obs, ctrl)
        estimates.append(result.mean)
    estimates_arr = np.asarray(estimates)
    from mcl_bio.utils.metrics import position_error

    err = position_error(estimates_arr, true_states)
    print(f"  position RMSE over {len(observations)} steps: {err:.4f} m")


def run_benchmark() -> None:
    """Compare the particle filter against the optimal Kalman filter."""
    import json

    from mcl_bio.benchmark import benchmark_pf_vs_kalman

    results = benchmark_pf_vs_kalman()
    print("mcl-bio benchmark: bootstrap PF vs optimal Kalman filter (linear-Gaussian)")
    print(json.dumps(results, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="mcl-bio demo runner")
    parser.add_argument(
        "demo",
        choices=["quickstart", "vex", "benchmark"],
        default="quickstart",
        nargs="?",
        help="Which demo to run",
    )
    args = parser.parse_args()
    if args.demo == "vex":
        run_vex_demo()
    elif args.demo == "benchmark":
        run_benchmark()
    else:
        run_quickstart()


if __name__ == "__main__":
    main()
