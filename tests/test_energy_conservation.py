from src.config import default_config
from src.simulation import MDEngine


def test_small_nve_energy_drift_is_bounded():
    config = default_config(n_particles=25, temperature=300.0, seed=44, dt=1.0e-15)
    engine = MDEngine(config)
    initial = engine.step()["total_energy"]
    energies = [initial]
    for _ in range(500):
        energies.append(engine.step()["total_energy"])
    drift = max(abs(value - initial) for value in energies)
    # This is a regression guard, not a claim about production accuracy.
    assert drift / max(abs(initial), 1e-30) < 2e-4
