"""PyTorch backend utilities for differentiable filters."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from numpy.typing import NDArray


def to_torch(x: NDArray, device: str = "cpu") -> torch.Tensor:
    """Convert NumPy array to torch tensor."""
    import torch

    return torch.tensor(x, dtype=torch.float32, device=device)


def to_numpy(x: torch.Tensor) -> NDArray:
    """Convert torch tensor to NumPy array."""
    return x.detach().cpu().numpy()
