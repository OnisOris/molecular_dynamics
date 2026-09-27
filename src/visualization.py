"""Matplotlib output for trajectories and observables."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np


J_TO_EV = 1.0 / 1.602_176_634e-19


def save_task1_plots(
    history: dict[str, np.ndarray],
    output_dir: Path,
    ideal_pressure_2d: float | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    time_ps = history["time"] * 1.0e12
    plots = [
        ("kinetic_energy", "Kinetic energy", "energy (10^-18 J)", 1.0e18),
        ("potential_energy", "Potential energy", "energy (10^-18 J)", 1.0e18),
        ("total_energy", "Total energy", "energy (10^-18 J)", 1.0e18),
        ("temperature", "Instantaneous temperature", "temperature (K)", 1.0),
        ("pressure_2d", "Instantaneous 2D pressure", "pressure (N/m)", 1.0),
    ]
    for key, title, ylabel, scale in plots:
        figure, axis = plt.subplots(figsize=(7, 4))
        axis.plot(time_ps, history[key] * scale, label=key.replace("_", " "))
        if key == "pressure_2d" and ideal_pressure_2d is not None:
            axis.axhline(
                ideal_pressure_2d,
                color="tab:red",
                linestyle="--",
                label="ideal-gas reference",
            )
        axis.set_title(title)
        axis.set_xlabel("time (ps)")
        axis.set_ylabel(ylabel)
        axis.legend()
        axis.grid(alpha=0.25)
        figure.tight_layout()
        figure.savefig(output_dir / f"{key}_vs_time.png", dpi=150)
        plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 4))
    for key, label in (
        ("total_energy", "total"),
        ("kinetic_energy", "kinetic"),
        ("potential_energy", "potential"),
    ):
        axis.plot(time_ps, history[key] * 1.0e18, label=label)
    axis.set_title("Energy components")
    axis.set_xlabel("time (ps)")
    axis.set_ylabel("energy (10^-18 J)")
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "energy_components_vs_time.png", dpi=150)
    plt.close(figure)

    initial_energy = history["total_energy"][0]
    relative_deviation = (
        history["total_energy"] - initial_energy
    ) / max(abs(initial_energy), 1.0e-30)
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.plot(time_ps, relative_deviation, label="(E-E0)/|E0|")
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_title("Relative total-energy deviation")
    axis.set_xlabel("time (ps)")
    axis.set_ylabel("relative energy deviation")
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "relative_energy_deviation_vs_time.png", dpi=150)
    plt.close(figure)


def save_task1_animation(
    frames: np.ndarray,
    box_length: float,
    output_path: Path,
    interval_ms: int = 40,
) -> None:
    if len(frames) == 0:
        raise ValueError("at least one frame is required")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(6, 6))
    scatter = axis.scatter(frames[0, :, 0] * 1e9, frames[0, :, 1] * 1e9, s=8)
    axis.set_xlim(0.0, box_length * 1e9)
    axis.set_ylim(0.0, box_length * 1e9)
    axis.set_aspect("equal")
    axis.set_xlabel("x (nm)")
    axis.set_ylabel("y (nm)")
    axis.set_title("2D potassium gas: periodic NVE")

    def update(frame: np.ndarray):
        scatter.set_offsets(frame * 1e9)
        return (scatter,)

    animation = FuncAnimation(
        figure, update, frames=frames, interval=interval_ms, blit=True
    )
    animation.save(output_path, writer=PillowWriter(fps=max(1, 1000 // interval_ms)))
    plt.close(figure)


def save_equilibrium_pressure_plot(
    history: dict[str, np.ndarray],
    equilibration_index: int,
    equilibrium_mean: float,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    time_ps = history["time"] * 1e12
    pressure = history["pressure_2d"]
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(time_ps, pressure, lw=0.8, label="instantaneous P")
    axis.axvline(
        time_ps[equilibration_index],
        color="tab:red",
        ls="--",
        label="equilibration point",
    )
    axis.axhline(
        equilibrium_mean,
        color="tab:green",
        ls="-",
        label="equilibrium mean",
    )
    axis.set_title("Equilibrium pressure detection")
    axis.set_xlabel("time (ps)")
    axis.set_ylabel("2D pressure (N/m)")
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def save_wall_animation(
    frames: np.ndarray,
    wall_positions: np.ndarray,
    box_length_x: float,
    box_length_y: float,
    output_path: Path,
    interval_ms: int = 40,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(6, 6))
    gas = axis.scatter(frames[0, :, 0] * 1e9, frames[0, :, 1] * 1e9, s=8, label="K")
    axis.scatter(
        wall_positions[:, 0] * 1e9,
        wall_positions[:, 1] * 1e9,
        s=16,
        color="black",
        label="fixed C wall",
    )
    axis.set_xlim(0.0, box_length_x * 1e9)
    axis.set_ylim(0.0, box_length_y * 1e9)
    axis.set_aspect("equal")
    axis.set_xlabel("x (nm)")
    axis.set_ylabel("y (nm)")
    axis.set_title("2D potassium gas with fixed carbon walls")
    axis.legend()

    def update(frame: np.ndarray):
        gas.set_offsets(frame * 1e9)
        return (gas,)

    animation = FuncAnimation(
        figure, update, frames=frames, interval=interval_ms, blit=True
    )
    animation.save(output_path, writer=PillowWriter(fps=max(1, 1000 // interval_ms)))
    plt.close(figure)


def save_phase_plot(
    temperatures: np.ndarray,
    mean_susceptibility: np.ndarray,
    transition_temperature: float,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.plot(temperatures, mean_susceptibility, "o-", label="cluster susceptibility")
    axis.axvline(
        transition_temperature,
        color="tab:red",
        ls="--",
        label=f"maximum: {transition_temperature:.1f} K",
    )
    axis.set_xlabel("temperature (K)")
    axis.set_ylabel("N Var(largest-cluster fraction)")
    axis.set_title("Finite-system gas-to-condensed crossover")
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def save_uq_plots(
    results: list[dict[str, float]],
    coefficients: dict[str, float],
    uncertainty: dict[str, float | np.ndarray],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    observed = np.asarray([row["P_eq"] for row in results])
    predicted = np.asarray(coefficients["predicted"])
    figure, axis = plt.subplots(figsize=(6, 5))
    axis.scatter(predicted, observed, label="DOE simulations")
    low = min(np.min(predicted), np.min(observed))
    high = max(np.max(predicted), np.max(observed))
    axis.plot([low, high], [low, high], "k--", label="ideal fit")
    axis.set_xlabel("surrogate P (N/m)")
    axis.set_ylabel("MD P_eq (N/m)")
    axis.set_title("Linear surrogate validation")
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "regression_comparison.png", dpi=150)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.hist(uncertainty["samples"], bins=50, density=True, alpha=0.8)
    axis.set_xlabel("surrogate pressure P (N/m)")
    axis.set_ylabel("density")
    axis.set_title("Propagated T/N uncertainty")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "uq_pressure_distribution.png", dpi=150)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 4.5))
    temperatures = np.asarray([row["T"] for row in results])
    particles = np.asarray([row["N"] for row in results])
    scatter = axis.scatter(temperatures, particles, c=observed, cmap="viridis")
    figure.colorbar(scatter, ax=axis, label="P_eq (N/m)")
    axis.set_xlabel("temperature (K)")
    axis.set_ylabel("number of particles N")
    axis.set_title("DOE inputs and simulated pressure")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "uq_input_pressure_scatter.png", dpi=150)
    plt.close(figure)
