"""Task 1: periodic 2D potassium gas in NVE."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.config import default_config
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
    save_task1_plots(history, output / "plots" / "task1")
    save_task1_animation(
        frames,
        config.box.lx,
        output / "animations" / "task1_periodic.gif",
    )

    initial = history["total_energy"][0]
    max_drift = np.max(np.abs(history["total_energy"] - initial))
    print(f"N={config.n_particles}, steps={args.steps}, seed={config.seed}")
    print(f"T0={config.temperature:.6f} K")
    print(f"maximum absolute energy drift={max_drift:.6e} J")
    print(f"outputs written below {output}")


if __name__ == "__main__":
    main()
