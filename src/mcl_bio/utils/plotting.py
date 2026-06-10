"""Plotting helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import matplotlib.pyplot as plt
    from numpy.typing import NDArray


def plot_trajectory(
    true_path: NDArray,
    estimated_path: NDArray,
    title: str = "Trajectory",
) -> plt.Figure:
    """Plot 2D ground truth vs estimated trajectory."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(true_path[:, 0], true_path[:, 1], "g-", label="Ground truth", linewidth=2)
    ax.plot(estimated_path[:, 0], estimated_path[:, 1], "b--", label="Estimate", linewidth=2)
    ax.scatter(true_path[0, 0], true_path[0, 1], c="green", marker="o", s=80, label="Start")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(title)
    ax.legend()
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_ppg_tracking(
    time: NDArray,
    observed: NDArray,
    filtered: NDArray,
    title: str = "PPG Heart Rate Tracking",
) -> plt.Figure:
    """Plot observed vs filtered biomedical signal."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(time, observed, "gray", alpha=0.5, label="Observed PPG")
    ax.plot(time, filtered, "r-", linewidth=1.5, label="Filtered")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig
