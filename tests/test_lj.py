import numpy as np

from src.constants import K_LJ
from src.potentials import LJTS, lj_energy_and_radial_derivative


def test_lj_potential_is_zero_at_sigma():
    energy, _ = lj_energy_and_radial_derivative(K_LJ.sigma, K_LJ)
    assert abs(energy) < 1e-30


def test_lj_force_is_zero_at_minimum():
    potential = LJTS(K_LJ, 2.5 * K_LJ.sigma)
    displacement = np.array([K_LJ.r_min, 0.0])
    _, force = potential.force_on_i(displacement)
    assert np.linalg.norm(force) < 1e-25


def test_repulsive_force_points_away_from_the_other_particle():
    potential = LJTS(K_LJ, 2.5 * K_LJ.sigma)
    _, force_on_left_particle = potential.force_on_i(
        np.array([-K_LJ.sigma, 0.0])
    )
    assert force_on_left_particle[0] < 0.0


def test_ljts_energy_is_zero_at_cutoff():
    cutoff = 2.5 * K_LJ.sigma
    potential = LJTS(K_LJ, cutoff)
    energy, derivative = potential.energy_and_radial_derivative(cutoff)
    assert energy == 0.0
    assert derivative == 0.0
