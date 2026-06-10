"""Particle filter state estimation for robotics and biomedical signals."""

from mcl_bio.core.auxiliary import AuxiliaryPF
from mcl_bio.core.bootstrap import BootstrapPF
from mcl_bio.core.filter import ParticleFilter, PFResult
from mcl_bio.core.models import MotionModel, ObservationModel

__all__ = [
    "AuxiliaryPF",
    "BootstrapPF",
    "MotionModel",
    "ObservationModel",
    "ParticleFilter",
    "PFResult",
]

__version__ = "1.0.0"

try:
    from mcl_bio.core.diffpf import DifferentiablePF  # noqa: F401

    __all__.append("DifferentiablePF")
except ImportError:
    pass
