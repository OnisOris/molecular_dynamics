"""Part 2: DOE, linear surrogate, sensitivities, and uncertainty propagation."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np

from src.config import default_config
from src.constants import BOLTZMANN, LAYER_THICKNESS, TEMPERATURE_K
from src.uq import fit_linear_surrogate, propagate_uniform_uncertainty, run_simulation
from src.visualization import save_uq_plots


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--equilibration-steps", type=int, default=2000)
    parser.add_argument("--production-steps", type=int, default=4000)
    parser.add_argument("--sample-interval", type=int, default=20)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 17, 27])
    args = parser.parse_args()

    nominal_n = 1035
    temperatures = [0.9 * TEMPERATURE_K, TEMPERATURE_K, 1.1 * TEMPERATURE_K]
    particle_counts = [round(0.95 * nominal_n), nominal_n, round(1.05 * nominal_n)]
    base_config = default_config(
        force_backend="cell_list",
        periodic_x=False,
        periodic_y=True,
        wall_mode="x",
    )
    results: list[dict[str, float]] = []
    for temperature in temperatures:
        for particle_count in particle_counts:
            for seed in args.seeds:
                result = run_simulation(
                    temperature,
                    particle_count,
                    replace(base_config, seed=seed),
                    equilibration_steps=args.equilibration_steps,
                    production_steps=args.production_steps,
                    sample_interval=args.sample_interval,
                )
                results.append(result)
                print(result)

    coefficients = fit_linear_surrogate(results)
    uncertainty = propagate_uniform_uncertainty(coefficients, TEMPERATURE_K, nominal_n)
    analytical_2d = nominal_n * BOLTZMANN * TEMPERATURE_K / base_config.box.area
    analytical_3d = analytical_2d / LAYER_THICKNESS
    nominal_results = [
        row for row in results if row["T"] == TEMPERATURE_K and row["N"] == nominal_n
    ]
    md_nominal = float(np.mean([row["P_eq"] for row in nominal_results]))
    absolute_difference = abs(md_nominal - analytical_2d)
    relative_difference_percent = 100.0 * absolute_difference / abs(analytical_2d)

    output = Path("output")
    output.joinpath("data").mkdir(parents=True, exist_ok=True)
    names = list(results[0])
    np.savetxt(
        output / "data" / "task5_uq_results.csv",
        np.asarray([[row[name] for name in names] for row in results]),
        delimiter=",",
        header=",".join(names),
        comments="",
    )
    save_uq_plots(results, coefficients, uncertainty, output / "plots" / "task5_uq")
    print("surrogate coefficients:", {key: value for key, value in coefficients.items() if key not in {"residuals", "predicted"}})
    print(f"dP/dT={coefficients['b_dP_dT']:.8e} N/(m K)")
    print(f"dP/dN={coefficients['c_dP_dN']:.8e} N/m")
    print(f"analytical P_2D={analytical_2d:.8e} N/m")
    print(f"analytical equivalent P_3D={analytical_3d:.8e} Pa")
    print(f"nominal MD P_eq={md_nominal:.8e} N/m")
    print(f"absolute difference={absolute_difference:.8e} N/m")
    print(f"relative difference={relative_difference_percent:.6f}%")
    print("propagated uncertainty:", {key: value for key, value in uncertainty.items() if key != "samples"})


if __name__ == "__main__":
    main()
