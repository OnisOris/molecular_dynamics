"""Task 3: fixed diamond-carbon walls on all four sides."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.config import default_config
from src.simulation import run_nve
from src.visualization import save_task1_plots, save_wall_animation
from src.walls import diamond_wall_positions


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
        periodic_x=False,
        periodic_y=False,
        wall_mode="all",
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
    np.savetxt(
        data_dir / "task3_timeseries.csv",
        np.column_stack([history[key] for key in history]),
        delimiter=",",
        header=",".join(history),
        comments="",
    )
    save_task1_plots(history, output / "plots" / "task3")
    wall_positions = diamond_wall_positions(config.box, "all")
    save_wall_animation(
        frames,
        wall_positions,
        config.box.lx,
        config.box.ly,
        output / "animations" / "task3_walls.gif",
    )
    print(f"N={config.n_particles}, wall atoms={len(wall_positions)}")
    print(f"steps={args.steps}, seed={config.seed}")
    print(f"outputs written below {output}")


if __name__ == "__main__":
    main()
