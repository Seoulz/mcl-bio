# API Reference

## Core

- `BootstrapPF` — bootstrap (SIR) particle filter
- `AuxiliaryPF` — auxiliary particle filter with look-ahead
- `DifferentiablePF` — soft-resampling filter for gradient learning
- `MotionModel` / `ObservationModel` — model abstractions
- `PFResult` — filter output dataclass

## Models

- `GaussianMotionModel`, `GaussianObservationModel` — linear-Gaussian models
- `MecanumMotionModel`, `BeaconObservationModel` — VEX localization models

## Utilities

- `mcl_bio.utils.metrics` — RMSE, position error, neff
- `mcl_bio.utils.plotting` — trajectory and PPG plots
