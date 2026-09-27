import numpy as np

from src.config import Box
from src.phase_analysis import cluster_susceptibility, largest_cluster_fraction


def test_cluster_fraction_detects_periodic_y_cluster():
    positions = np.array([[1.0, 0.1], [1.0, 9.9], [8.0, 5.0]])
    fraction = largest_cluster_fraction(
        positions, Box(10.0, 10.0), link_distance=0.3, periodic_y=True
    )
    assert fraction == 2.0 / 3.0


def test_cluster_susceptibility_is_nonnegative():
    assert cluster_susceptibility(np.array([0.1, 0.3, 0.2]), 10) > 0.0
