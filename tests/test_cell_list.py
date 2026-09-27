import numpy as np
import pytest

from src.config import Box, default_config
from src.constants import K_LJ
from src.forces import compute_forces_bruteforce, compute_forces_cell_list
from src.initialization import initialize
from src.potentials import LJTS


def _assert_backends_match(positions: np.ndarray, box: Box) -> None:
    potential = LJTS(K_LJ, 2.5 * K_LJ.sigma)
    reference = compute_forces_bruteforce(positions, box, potential)
    accelerated = compute_forces_cell_list(positions, box, potential)
    np.testing.assert_allclose(accelerated[0], reference[0], rtol=1e-12, atol=1e-28)
    np.testing.assert_allclose(accelerated[1:], reference[1:], rtol=1e-12, atol=1e-28)


def _ordinary_lattice() -> tuple[Box, np.ndarray]:
    config = default_config(n_particles=80, seed=33)
    positions, _ = initialize(config)
    return config.box, positions


def _across_periodic_edges() -> tuple[Box, np.ndarray]:
    box = Box(6.0e-9, 6.0e-9)
    positions = np.array(
        [
            [0.12e-9, 1.00e-9],
            [5.57e-9, 1.00e-9],
            [2.00e-9, 0.10e-9],
            [2.00e-9, 5.50e-9],
            [3.00e-9, 3.00e-9],
            [3.80e-9, 3.00e-9],
        ]
    )
    return box, positions


def _near_cutoff() -> tuple[Box, np.ndarray]:
    cutoff = 2.5 * K_LJ.sigma
    box = Box(6.0e-9, 6.0e-9)
    positions = np.array(
        [
            [1.0e-9, 1.0e-9],
            [1.0e-9 + cutoff * (1.0 - 1.0e-10), 1.0e-9],
            [1.0e-9, 3.0e-9],
            [1.0e-9 + cutoff * (1.0 + 1.0e-10), 3.0e-9],
            [4.5e-9, 5.0e-9],
        ]
    )
    return box, positions


def _dense_lattice() -> tuple[Box, np.ndarray]:
    config = default_config(n_particles=64, box=Box(5.2e-9, 5.2e-9), seed=55)
    positions, _ = initialize(config)
    return config.box, positions


@pytest.mark.parametrize(
    "configuration",
    [_ordinary_lattice, _across_periodic_edges, _near_cutoff, _dense_lattice],
    ids=["ordinary", "pbc_edges", "cutoff", "dense"],
)
def test_cell_list_matches_bruteforce(configuration):
    box, positions = configuration()
    _assert_backends_match(positions, box)


def test_periodic_edge_pairs_really_interact():
    box, positions = _across_periodic_edges()
    potential = LJTS(K_LJ, 2.5 * K_LJ.sigma)
    forces, energy, virial = compute_forces_cell_list(positions, box, potential)
    assert np.linalg.norm(forces[0]) > 0.0
    assert energy != 0.0
    assert virial != 0.0
