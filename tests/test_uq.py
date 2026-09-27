import numpy as np

from src.uq import fit_linear_surrogate, propagate_uniform_uncertainty


def test_linear_surrogate_recovers_known_coefficients():
    results = [
        {"T": float(t), "N": float(n), "P_eq": 2.0 + 3.0 * t - 0.5 * n}
        for t in (1.0, 2.0, 3.0)
        for n in (10.0, 12.0, 14.0)
    ]
    fit = fit_linear_surrogate(results)
    assert abs(fit["a"] - 2.0) < 1e-10
    assert abs(fit["b_dP_dT"] - 3.0) < 1e-10
    assert abs(fit["c_dP_dN"] + 0.5) < 1e-10
    assert abs(fit["r_squared"] - 1.0) < 1e-12


def test_uncertainty_propagation_returns_ordered_interval():
    fit = {"a": 0.0, "b_dP_dT": 1.0, "c_dP_dN": 1.0}
    result = propagate_uniform_uncertainty(fit, 100.0, 100, samples=1000)
    assert result["P_q025"] < result["P_mean"] < result["P_q975"]
