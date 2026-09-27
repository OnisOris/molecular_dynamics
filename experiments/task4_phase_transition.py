"""Task 4: x carbon walls, y periodic, and a temperature sweep."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.config import default_config
from src.constants import BOLTZMANN, K_LJ
from src.observables import instantaneous_temperature
from src.phase_analysis import cluster_susceptibility, largest_cluster_fraction
from src.simulation import MDEngine
from src.visualization import save_phase_plot


def run_temperature_point(
    temperature: float,
    seed: int,
    equilibration_steps: int,
    production_steps: int,
    sample_interval: int,
) -> dict[str, float]:
    config = default_config(
        temperature=temperature,
        seed=seed,
        force_backend="cell_list",
        periodic_x=False,
        periodic_y=True,
        wall_mode="x",
    )
    engine = MDEngine(config)
    for step in range(equilibration_steps):
        engine.step()
        if (step + 1) % 10 == 0:
            measured = instantaneous_temperature(
                engine.state.velocities, config.mass, config.remove_com
            )
            engine.state.velocities *= np.sqrt(temperature / measured)

    fractions: list[float] = []
    pressures: list[float] = []
    for step in range(1, production_steps + 1):
        observation = engine.step()
        if step % sample_interval == 0:
            fractions.append(
                largest_cluster_fraction(
                    engine.state.positions,
                    config.box,
                    link_distance=1.5 * K_LJ.sigma,
                    periodic_x=False,
                    periodic_y=True,
                )
            )
            pressures.append(observation["pressure_wall_x"])

    fraction_values = np.asarray(fractions)
    pressure_values = np.asarray(pressures)
    return {
        "T": temperature,
        "N": config.n_particles,
        "seed": seed,
        "largest_cluster_mean": float(np.mean(fraction_values)),
        "cluster_susceptibility": cluster_susceptibility(
            fraction_values, config.n_particles
        ),
        "P_eq": float(np.mean(pressure_values)),
        "P_std": float(np.std(pressure_values, ddof=1)),
        "P_sem": float(np.std(pressure_values, ddof=1) / np.sqrt(len(pressure_values))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--temperatures", type=float, nargs="+", default=list(range(900, 249, -50)))
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 17, 27])
    parser.add_argument("--equilibration-steps", type=int, default=2000)
    parser.add_argument("--production-steps", type=int, default=4000)
    parser.add_argument("--sample-interval", type=int, default=20)
    args = parser.parse_args()

    results = []
    for temperature in args.temperatures:
        for seed in args.seeds:
            result = run_temperature_point(
                temperature,
                seed,
                args.equilibration_steps,
                args.production_steps,
                args.sample_interval,
            )
            results.append(result)
            print(result)

    output = Path("output")
    output.joinpath("data").mkdir(parents=True, exist_ok=True)
    names = list(results[0])
    array = np.asarray([[result[name] for name in names] for result in results])
    np.savetxt(
        output / "data" / "task4_phase_sweep.csv",
        array,
        delimiter=",",
        header=",".join(names),
        comments="",
    )
    temperatures = np.asarray(sorted(set(args.temperatures)))
    mean_susceptibility = np.asarray(
        [
            np.mean([r["cluster_susceptibility"] for r in results if r["T"] == t])
            for t in temperatures
        ]
    )
    transition = float(temperatures[np.argmax(mean_susceptibility)])
    transition_by_seed = [
        max(
            (r for r in results if r["seed"] == seed),
            key=lambda row: row["cluster_susceptibility"],
        )["T"]
        for seed in args.seeds
    ]
    print(f"transition crossover estimate={transition:.6g} K")
    print(f"seed-to-seed range={min(transition_by_seed):.6g}..{max(transition_by_seed):.6g} K")
    save_phase_plot(
        temperatures,
        mean_susceptibility,
        transition,
        output / "plots" / "task4_phase_transition.png",
    )


if __name__ == "__main__":
    main()
