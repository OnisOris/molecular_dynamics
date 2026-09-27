"""Task 1: periodic 2D potassium gas in NVE."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.config import default_config
from src.constants import BOLTZMANN, LAYER_THICKNESS
from src.simulation import run_nve
from src.visualization import save_task1_animation, save_task1_plots


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--sample-interval", type=int, default=1)
    parser.add_argument("--frame-interval", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    config = default_config(
        seed=args.seed,
        force_backend="cell_list",
        periodic_x=True,
        periodic_y=True,
    )
    history, frames = run_nve(
        config,
        n_steps=args.steps,
        sample_interval=args.sample_interval,
        frame_interval=args.frame_interval,
    )

    output = Path("output")
    data_dir = output / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    columns = [history[key] for key in history]
    np.savetxt(
        data_dir / "task1_timeseries.csv",
        np.column_stack(columns),
        delimiter=",",
        header=",".join(history),
        comments="",
    )
    ideal_pressure_2d = (
        config.n_particles * BOLTZMANN * config.temperature / config.box.area
    )
    save_task1_plots(
        history,
        output / "plots" / "task1",
        ideal_pressure_2d=ideal_pressure_2d,
    )
    save_task1_animation(
        frames,
        config.box.lx,
        output / "animations" / "task1_periodic.gif",
    )

    initial_energy = history["total_energy"][0]
    energy_difference = history["total_energy"] - initial_energy
    max_absolute_drift = float(np.max(np.abs(energy_difference)))
    max_relative_drift = max_absolute_drift / abs(initial_energy)
    mean_temperature = float(np.mean(history["temperature"]))
    std_temperature = float(np.std(history["temperature"], ddof=1))
    mean_pressure = float(np.mean(history["pressure_2d"]))
    std_pressure = float(np.std(history["pressure_2d"], ddof=1))
    mean_pressure_3d = mean_pressure / LAYER_THICKNESS
    summary = "\n".join(
        [
            f"N={config.n_particles}",
            f"steps={args.steps}",
            f"dt={config.dt * 1e15:.3f} fs",
            f"simulation_time={args.steps * config.dt * 1e12:.6f} ps",
            f"seed={config.seed}",
            f"T_target={config.temperature:.6f} K",
            f"T_initial={history['temperature'][0]:.6f} K",
            f"E0={initial_energy:.8e} J",
            f"max_absolute_energy_drift={max_absolute_drift:.8e} J",
            f"max_relative_energy_drift={max_relative_drift:.8e}",
            f"max_relative_energy_drift_percent={100.0 * max_relative_drift:.8e} %",
            f"mean_temperature={mean_temperature:.8f} K",
            f"std_temperature={std_temperature:.8f} K",
            f"mean_pressure_2d={mean_pressure:.8e} N/m",
            f"std_pressure_2d={std_pressure:.8e} N/m",
            f"ideal_pressure_2d={ideal_pressure_2d:.8e} N/m",
            f"mean_pressure_3d_equiv={mean_pressure_3d:.8e} Pa",
            f"mean_pressure_3d_equiv_mpa={mean_pressure_3d / 1e6:.8f} MPa",
            f"ideal_pressure_3d_equiv={ideal_pressure_2d / LAYER_THICKNESS:.8e} Pa",
            f"ideal_pressure_3d_equiv_mpa={ideal_pressure_2d / LAYER_THICKNESS / 1e6:.8f} MPa",
        ]
    )
    (output / "data" / "task1_summary.txt").write_text(summary + "\n", encoding="utf-8")
    print(summary)
    print(f"outputs written below {output}")


if __name__ == "__main__":
    main()
