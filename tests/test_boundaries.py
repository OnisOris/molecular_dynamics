import numpy as np

from src.boundaries import minimum_image, wrap_positions
from src.config import Box


def test_minimum_image_connects_opposite_periodic_edges():
    box = Box(10.0, 10.0)
    displacement = minimum_image(np.array([9.8, -9.7]), box)
    np.testing.assert_allclose(displacement, np.array([-0.2, 0.3]))


def test_minimum_image_only_wraps_periodic_axis():
    box = Box(10.0, 10.0)
    displacement = minimum_image(np.array([9.8, -9.7]), box, periodic_x=False)
    np.testing.assert_allclose(displacement, np.array([9.8, 0.3]))


def test_pbc_wrapping():
    positions = np.array([[-0.2, 10.3], [20.2, -0.1]])
    wrapped = wrap_positions(positions, Box(10.0, 10.0))
    np.testing.assert_allclose(wrapped, np.array([[9.8, 0.3], [0.2, 9.9]]))
