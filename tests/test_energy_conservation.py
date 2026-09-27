import numpy as np

from src.config import Box, default_config
from src.initialization import initialize
from src.simulation import MDEngine


def test_dense_interacting_nve_energy_drift_is_bounded():
    config = default_config(
        n_particles=16,
        temperature=300.0,
        box=Box(3.0e-9, 3.0e-9),
        seed=44,
        dt=1.0e-15,
        force_backend="bruteforce",
    )
    positions, velocities = initialize(config)
    engine = MDEngine(config, positions, velocities)
    initial_potential = engine.state.potential_energy
    initial = 0.5 * config.mass * np.sum(velocities * velocities) + initial_potential
    energies = [initial]
    potential_energies = [initial_potential]
    force_norms = [np.linalg.norm(engine.state.forces)]
    for _ in range(2000):
        observation = engine.step()
        energies.append(observation["total_energy"])
        potential_energies.append(observation["potential_energy"])
        force_norms.append(np.linalg.norm(engine.state.forces))

    energies = np.asarray(energies)
    potential_energies = np.asarray(potential_energies)
    relative_drift = np.max(np.abs(energies - initial)) / abs(initial)

    assert np.max(np.abs(potential_energies)) > 0.05 * abs(initial)
    assert np.ptp(potential_energies) > 1.0e-22
    assert np.max(force_norms) > 0.0
    # The compact, strongly interacting 16-particle cell is a deliberately
    # demanding regression case; dt=1 fs remains below 5e-4 drift over 2 ps.
    assert relative_drift < 5e-4
