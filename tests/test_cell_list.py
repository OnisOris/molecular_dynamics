import numpy as np

from src.config import Box, default_config
from src.forces import compute_forces_bruteforce, compute_forces_cell_list
from src.initialization import initialize
from src.potentials import LJTS


def test_cell_list_matches_bruteforce():
    config = default_config(n_particles=80, seed=33)
    positions, _ = initialize(config)
    potential = LJTS(config.lj, config.cutoff)
    reference = compute_forces_bruteforce(positions, config.box, potential)
    accelerated = compute_forces_cell_list(positions, config.box, potential)
    np.testing.assert_allclose(accelerated[0], reference[0], rtol=1e-12, atol=1e-28)
    np.testing.assert_allclose(accelerated[1:], reference[1:], rtol=1e-12, atol=1e-28)
