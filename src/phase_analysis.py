"""Finite-system diagnostics for the gas-to-condensed crossover."""

from __future__ import annotations

import numpy as np

from .boundaries import minimum_image
from .config import Box


def largest_cluster_fraction(
    positions: np.ndarray,
    box: Box,
    link_distance: float,
    periodic_x: bool = False,
    periodic_y: bool = True,
) -> float:
    """Return the largest connected-component fraction by a distance link."""
    n = len(positions)
    parents = np.arange(n)

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(first: int, second: int) -> None:
        root_first, root_second = find(first), find(second)
        if root_first != root_second:
            parents[root_second] = root_first

    for i in range(n - 1):
        displacement = positions[i] - positions[i + 1 :]
        displacement = minimum_image(
            displacement, box, periodic_x=periodic_x, periodic_y=periodic_y
        )
        close = np.flatnonzero(np.linalg.norm(displacement, axis=1) < link_distance)
        for offset in close:
            union(i, i + 1 + int(offset))

    counts = np.bincount([find(index) for index in range(n)], minlength=n)
    return float(np.max(counts) / n)


def cluster_susceptibility(fractions: np.ndarray, n_particles: int) -> float:
    values = np.asarray(fractions, dtype=np.float64)
    return float(n_particles * np.var(values))


def radial_distribution_2d(
    snapshots: np.ndarray,
    box: Box,
    r_max: float,
    bins: int = 100,
    periodic_x: bool = False,
    periodic_y: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate a pair-distance histogram normalized as a bulk 2D g(r).

    For fixed-wall systems this is a diagnostic rather than a homogeneous-bulk
    thermodynamic estimator; wall inhomogeneity is reported separately.
    """
    edges = np.linspace(0.0, r_max, bins + 1)
    counts = np.zeros(bins, dtype=np.float64)
    n_snapshots = len(snapshots)
    if n_snapshots == 0:
        return 0.5 * (edges[1:] + edges[:-1]), counts
    n_particles = snapshots.shape[1]
    for positions in snapshots:
        for i in range(n_particles - 1):
            displacement = positions[i] - positions[i + 1 :]
            displacement = minimum_image(
                displacement, box, periodic_x=periodic_x, periodic_y=periodic_y
            )
            counts += np.histogram(np.linalg.norm(displacement, axis=1), edges)[0]
    radii = 0.5 * (edges[1:] + edges[:-1])
    shell_area = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
    ideal_unordered = n_snapshots * n_particles * (n_particles - 1) / (2.0 * box.area)
    g_r = counts / (ideal_unordered * shell_area)
    return radii, g_r
