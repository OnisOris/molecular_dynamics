import numpy as np

from src.config import default_config
from src.constants import KC_LJ
from src.simulation import MDEngine
from src.walls import diamond_wall_positions


def test_x_wall_geometry_is_periodic_only_along_y():
    config = default_config(
        n_particles=36,
        periodic_x=False,
        periodic_y=True,
        wall_mode="x",
        seed=4,
    )
    walls = diamond_wall_positions(config.box, "x")
    assert len(walls) > 2
    assert np.all((walls[:, 0] == 0.0) | (walls[:, 0] == config.box.lx))
    assert np.all(walls[:, 1] < config.box.ly)


def test_fixed_walls_do_not_integrate_and_produce_reaction_force():
    config = default_config(
        n_particles=36,
        periodic_x=False,
        periodic_y=True,
        wall_mode="x",
        seed=5,
    )
    engine = MDEngine(config)
    initial_walls = engine.wall_positions.copy()
    observation = engine.step()
    np.testing.assert_array_equal(engine.wall_positions, initial_walls)
    assert observation["pressure_wall_x"] >= 0.0
    assert engine.wall_potential.parameters == KC_LJ
