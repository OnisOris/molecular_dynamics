"""Non-intrusive uncertainty quantification around the Task 4 model."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from .config import MDConfig
from .observables import instantaneous_temperature
from .simulation import MDEngine


def run_simulation(
    temperature: float,
    n_particles: int,
    config: MDConfig,
    equilibration_steps: int = 2000,
    production_steps: int = 4000,
    sample_interval: int = 20,
) -> dict[str, float]:
    """Run the unchanged MD engine for one UQ input pair."""
    run_config = replace(
        config, temperature=float(temperature), n_particles=int(n_particles)
    )
    engine = MDEngine(run_config)
    for step in range(equilibration_steps):
        engine.step()
        if (step + 1) % 10 == 0:
            measured = instantaneous_temperature(
                engine.state.velocities, run_config.mass, run_config.remove_com
            )
            engine.state.velocities *= np.sqrt(run_config.temperature / measured)

    pressure_samples: list[float] = []
    for step in range(1, production_steps + 1):
        observation = engine.step()
        if step % sample_interval == 0:
            pressure_samples.append(observation["pressure_wall_x"])
    samples = np.asarray(pressure_samples, dtype=np.float64)
    if len(samples) < 2:
        raise ValueError("production must contain at least two pressure samples")
    return {
        "T": float(temperature),
        "N": int(n_particles),
        "seed": int(run_config.seed),
        "P_eq": float(np.mean(samples)),
        "P_std": float(np.std(samples, ddof=1)),
        "P_sem": float(np.std(samples, ddof=1) / np.sqrt(len(samples))),
        "equilibration_time": float(equilibration_steps * run_config.dt),
    }


def fit_linear_surrogate(results: list[dict[str, float]]) -> dict[str, Any]:
    temperatures = np.asarray([row["T"] for row in results], dtype=np.float64)
    particles = np.asarray([row["N"] for row in results], dtype=np.float64)
    pressures = np.asarray([row["P_eq"] for row in results], dtype=np.float64)
    design = np.column_stack((np.ones(len(results)), temperatures, particles))
    coefficients, *_ = np.linalg.lstsq(design, pressures, rcond=None)
    predicted = design @ coefficients
    residuals = pressures - predicted
    sse = float(np.sum(residuals**2))
    sst = float(np.sum((pressures - np.mean(pressures)) ** 2))
    return {
        "a": float(coefficients[0]),
        "b_dP_dT": float(coefficients[1]),
        "c_dP_dN": float(coefficients[2]),
        "r_squared": float(1.0 - sse / sst) if sst > 0.0 else float("nan"),
        "rmse": float(np.sqrt(np.mean(residuals**2))),
        "max_abs_residual": float(np.max(np.abs(residuals))),
        "residuals": residuals,
        "predicted": predicted,
    }


def propagate_uniform_uncertainty(
    coefficients: dict[str, Any],
    nominal_temperature: float,
    nominal_particles: int,
    samples: int = 100_000,
    seed: int = 2027,
) -> dict[str, float | np.ndarray]:
    rng = np.random.default_rng(seed)
    temperatures = rng.uniform(0.9 * nominal_temperature, 1.1 * nominal_temperature, samples)
    particles = rng.uniform(0.95 * nominal_particles, 1.05 * nominal_particles, samples)
    pressures = (
        coefficients["a"]
        + coefficients["b_dP_dT"] * temperatures
        + coefficients["c_dP_dN"] * particles
    )
    return {
        "P_mean": float(np.mean(pressures)),
        "P_std": float(np.std(pressures, ddof=1)),
        "P_q025": float(np.quantile(pressures, 0.025)),
        "P_q975": float(np.quantile(pressures, 0.975)),
        "samples": pressures,
    }
