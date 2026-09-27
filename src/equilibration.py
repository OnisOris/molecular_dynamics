"""Numerical burn-in detection and block statistics."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class EquilibriumStatistics:
    start_index: int
    start_step: int
    mean: float
    std: float
    sem: float
    n_samples: int


def find_equilibration_index(
    steps: np.ndarray,
    temperatures: np.ndarray,
    pressures: np.ndarray,
    ideal_pressure: float,
    block_size: int = 2000,
    comparison_blocks: int = 5,
    temperature_tolerance: float = 0.02,
    pressure_tolerance: float = 0.05,
    consecutive: int = 3,
) -> int:
    """Return the first sample index satisfying the fixed block criterion."""
    if block_size <= 0 or comparison_blocks <= 0 or consecutive <= 0:
        raise ValueError("block parameters must be positive")
    n = len(steps)
    window_length = 2 * comparison_blocks * block_size
    required = window_length + (consecutive - 1) * block_size
    if n < required:
        raise ValueError(
            f"at least {required} samples are required for equilibration detection"
        )

    pressure_scale = max(abs(ideal_pressure), 1.0e-30)
    run_length = consecutive
    for start in range(0, n - required + 1, block_size):
        old_slice = slice(start, start + comparison_blocks * block_size)
        new_start = old_slice.stop
        new_slice = slice(new_start, new_start + comparison_blocks * block_size)
        old_temperature = float(np.mean(temperatures[old_slice]))
        new_temperature = float(np.mean(temperatures[new_slice]))
        old_pressure = float(np.mean(pressures[old_slice]))
        new_pressure = float(np.mean(pressures[new_slice]))
        temperature_ok = (
            abs(new_temperature - old_temperature) / max(abs(new_temperature), 1.0e-30)
            < temperature_tolerance
        )
        pressure_ok = (
            abs(new_pressure - old_pressure) / pressure_scale < pressure_tolerance
        )
        if temperature_ok and pressure_ok:
            run_length -= 1
            if run_length == 0:
                return new_start
        else:
            run_length = consecutive
    raise ValueError("equilibration criterion was not met in the supplied trajectory")


def block_statistics(values: np.ndarray, start_index: int, block_size: int = 2000) -> EquilibriumStatistics:
    """Compute mean, standard deviation, and block-based SEM after burn-in."""
    values = np.asarray(values, dtype=np.float64)
    post = values[start_index:]
    if len(post) < 2:
        raise ValueError("at least two post-equilibration samples are required")
    std = float(np.std(post, ddof=1))
    blocks = len(post) // max(1, block_size)
    if blocks >= 2:
        block_means = np.mean(post[: blocks * block_size].reshape(blocks, block_size), axis=1)
        sem = float(np.std(block_means, ddof=1) / np.sqrt(blocks))
    else:
        sem = float(std / np.sqrt(len(post)))
    return EquilibriumStatistics(
        start_index=start_index,
        start_step=-1,
        mean=float(np.mean(post)),
        std=std,
        sem=sem,
        n_samples=len(post),
    )
