"""Task 2: detect equilibration and estimate the equilibrium pressure."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.config import default_config
from src.constants import BOLTZMANN
from src.equilibration import block_statistics, find_equilibration_index
from src.simulation import run_nve
from src.visualization import save_equilibrium_pressure_plot


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--block-size", type=int, default=2000)
    parser.add_argument("--sample-interval", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    config = default_config(seed=args.seed, force_backend="cell_list")
    history, _ = run_nve(
        config,
        n_steps=args.steps,
        sample_interval=args.sample_interval,
        frame_interval=max(1, args.steps + 1),
    )
    ideal_pressure = config.n_particles * BOLTZMANN * config.temperature / config.box.area
    try:
        equilibration_index = find_equilibration_index(
            history["step"].astype(int),
            history["temperature"],
            history["pressure_2d"],
            ideal_pressure,
            block_size=max(1, args.block_size // args.sample_interval),
        )
    except ValueError as error:
        raise SystemExit(
            f"equilibration criterion was not reached: {error}. "
            "Increase --steps or inspect the diagnostic trajectory."
        ) from error
    stats = block_statistics(history["pressure_2d"], equilibration_index, args.block_size)
    start_step = int(history["step"][equilibration_index])
    output = Path("output")
    output.joinpath("data").mkdir(parents=True, exist_ok=True)
    np.savetxt(
        output / "data" / "task2_pressure.csv",
        np.column_stack((history["step"], history["time"], history["pressure_2d"])),
        delimiter=",",
        header="step,time_s,pressure_2d_N_per_m",
        comments="",
    )
    save_equilibrium_pressure_plot(
        history,
        equilibration_index,
        stats.mean,
        output / "plots" / "task2_equilibrium_pressure.png",
    )
    print(f"equilibration step={start_step}")
    print(f"equilibration time={history['time'][equilibration_index] * 1e12:.6f} ps")
    print(f"P_eq={stats.mean:.8e} N/m")
    print(f"P_std={stats.std:.8e} N/m")
    print(f"P_sem={stats.sem:.8e} N/m")


if __name__ == "__main__":
    main()
