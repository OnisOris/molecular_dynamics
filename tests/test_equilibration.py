import numpy as np

from src.equilibration import block_statistics, find_equilibration_index


def test_equilibration_uses_block_means_not_last_sample():
    steps = np.arange(120)
    temperature = np.full(120, 100.0)
    pressure = np.full(120, 4.0)
    pressure[-1] = 1000.0
    index = find_equilibration_index(
        steps,
        temperature,
        pressure,
        ideal_pressure=4.0,
        block_size=10,
        comparison_blocks=2,
        consecutive=2,
    )
    stats = block_statistics(pressure, index, block_size=10)
    assert index < 80
    assert stats.mean < 100.0
    assert stats.n_samples == 120 - index
