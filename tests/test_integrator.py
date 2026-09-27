import numpy as np

from src.config import default_config
from src.forces import compute_forces_bruteforce
from src.potentials import LJTS
from src.simulation import MDEngine
from src.constants import K_LJ


def test_reproducibility_for_same_seed():
    config = default_config(n_particles=36, seed=22)
    first = MDEngine(config)
    second = MDEngine(config)
    np.testing.assert_array_equal(first.state.positions, second.state.positions)
    np.testing.assert_array_equal(first.state.velocities, second.state.velocities)
    for _ in range(5):
        first.step()
        second.step()
    np.testing.assert_array_equal(first.state.positions, second.state.positions)
    np.testing.assert_array_equal(first.state.velocities, second.state.velocities)


def test_newton_third_law_for_two_particles():
    config = default_config(n_particles=2, seed=1)
    positions = np.array([[1.0e-9, 1.0e-9], [2.0e-9, 1.0e-9]])
    velocities = np.zeros((2, 2))
    engine = MDEngine(config, positions, velocities)
    np.testing.assert_allclose(engine.state.forces.sum(axis=0), 0.0, atol=1e-20)


def test_repulsive_pair_has_positive_virial():
    config = default_config(n_particles=2, seed=1)
    positions = np.array([[1.0e-9, 1.0e-9], [1.0e-9 + K_LJ.sigma, 1.0e-9]])
    _, _, virial = compute_forces_bruteforce(
        positions, config.box, LJTS(K_LJ, 2.5 * K_LJ.sigma)
    )
    assert virial > 0.0
