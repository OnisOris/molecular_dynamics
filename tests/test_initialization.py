import numpy as np

from src.config import default_config
from src.initialization import (
    _stream_rng,
    initialize,
    lattice_positions,
    maxwell_boltzmann_velocities,
)
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


def test_nominal_positions_have_no_large_underfilled_spatial_strip():
    config = default_config(n_particles=1035, seed=7)
    positions = lattice_positions(config)
    counts, _, _ = np.histogram2d(
        positions[:, 0],
        positions[:, 1],
        bins=(8, 8),
        range=((0.0, config.box.lx), (0.0, config.box.ly)),
    )
    assert np.all(counts > 0)
    assert counts.max() - counts.min() <= 8
    assert counts.std() / counts.mean() < 0.12


def test_nominal_initial_positions_are_not_too_close():
    config = default_config(n_particles=1035, seed=7)
    positions = lattice_positions(config)
    minimum = np.inf
    for index in range(len(positions) - 1):
        differences = positions[index + 1 :] - positions[index]
        differences[:, 0] -= config.box.lx * np.round(differences[:, 0] / config.box.lx)
        differences[:, 1] -= config.box.ly * np.round(differences[:, 1] / config.box.ly)
        minimum = min(minimum, np.min(np.linalg.norm(differences, axis=1)))
    assert minimum > config.lj.sigma


def test_position_and_velocity_random_streams_are_distinct():
    config = default_config(seed=7)
    position_draws = _stream_rng(config, 0).random(16)
    velocity_draws = _stream_rng(config, 1).random(16)
    assert not np.array_equal(position_draws, velocity_draws)
