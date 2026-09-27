import numpy as np

from src.config import default_config
from src.constants import BOLTZMANN
from src.initialization import initialize, maxwell_boltzmann_velocities
from src.observables import instantaneous_temperature


def test_initial_velocities_have_zero_total_momentum():
    config = default_config(n_particles=64, seed=101)
    _, velocities = initialize(config)
    np.testing.assert_allclose(velocities.mean(axis=0), 0.0, atol=1e-12)


def test_initial_temperature_matches_target():
    config = default_config(n_particles=64, temperature=700.0, seed=102)
    velocities = maxwell_boltzmann_velocities(config)
    measured = instantaneous_temperature(velocities, config.mass, remove_com=True)
    assert abs(measured - config.temperature) / config.temperature < 1e-12
