from __future__ import annotations

import numpy as np

from experiments.benchmarks._utils import FitCell, fit_cell, rows_for_fit
from sde2d.library import make_library, polynomial_change_of_basis
from sde2d.metrics import a12_sign_accuracy, central_grid, function_l2_errors
from sde2d.systems import CholeskyDiffusion2D, CorrelatedOU2D, LogHestonSV
from sde2d.generator import fit_generator_2d


def test_backtransform_roundtrip():
    rng = np.random.default_rng(123)
    x = rng.normal(size=(50, 2))
    mean = x.mean(axis=0)
    scale = x.std(axis=0, ddof=1)
    z = (x - mean) / scale
    lib = make_library("A")
    coef_z = np.array([0.3, -0.7, 0.2, 0.1, -0.05, 0.04])
    coef_raw = polynomial_change_of_basis(mean, scale, 2) @ coef_z
    assert np.allclose(lib.transform(z) @ coef_z, lib.transform(x) @ coef_raw)


def test_oracle_separates_nondiag_selection_failure():
    cell = FitCell(
        experiment="test",
        system_key="nondiag_cholesky",
        library="C",
        regressor="lasso_stlsq",
        n_centers=64,
        n_steps=5000,
        n_trajectories=3,
        library_space="z",
        stlsq_threshold=0.18,
        seed=222,
    )
    system, x, fit, runtime = fit_cell(cell)
    rows = rows_for_fit(cell, system, x, fit, runtime)
    summary = rows["benchmark_summary"][0]
    assert summary["oracle_diffusion_rel_l2"] <= summary["diffusion_rel_l2"] + 1e-12
    assert summary["oracle_a12_cosine"] >= 0.95


def test_repro_lock_for_audited_cells():
    cells = [
        FitCell("test", "correlated_ou", "A", "stlsq", n_centers=36, n_steps=1200, seed=333),
        FitCell("test", "heston_logsv", "D", "stlsq", n_centers=36, n_steps=1500, dt=1.0 / 252.0, seed=444),
    ]
    for cell in cells:
        metrics = []
        for _ in range(2):
            system, x, fit, runtime = fit_cell(cell)
            row = rows_for_fit(cell, system, x, fit, runtime)["benchmark_summary"][0]
            metrics.append((row["drift_rel_l2"], row["diffusion_rel_l2"], row["a12_cosine"]))
        assert np.allclose(metrics[0], metrics[1], rtol=0.0, atol=1e-9)


def test_group_metadata_reaches_generator():
    system = CorrelatedOU2D()
    paths = [system.simulate(dt=0.01, M=400, seed=10 + i) for i in range(3)]
    states = np.vstack([p[:-1] for p in paths])
    inc = np.vstack([np.diff(p, axis=0) for p in paths])
    groups = np.concatenate([np.full(len(p) - 1, i) for i, p in enumerate(paths)])
    fit = fit_generator_2d(
        states,
        increments=inc,
        dt=0.01,
        library=make_library("A"),
        n_centers=25,
        center_scheme="quantile_grid",
        grid_shape=(5, 5),
        bandwidth_multiplier=1.5,
        regressor="lasso_stlsq",
        traj_ids=groups,
        seed=10,
    )
    assert fit.bandwidth_meta["n_trajectories"] == 3
    assert fit.bandwidth_meta["projected_group_folds"] >= 2


def test_zspace_lasso_beats_raw_stlsq_drift():
    raw = FitCell("test", "correlated_ou", "A", "stlsq", n_centers=36, n_steps=1500, seed=333, library_space="raw")
    cured = FitCell(
        "test",
        "correlated_ou",
        "A",
        "lasso_stlsq",
        n_centers=36,
        n_steps=1500,
        n_trajectories=3,
        seed=333,
        library_space="z",
        stlsq_threshold=0.10,
    )
    raw_system, raw_x, raw_fit, _ = fit_cell(raw)
    cured_system, cured_x, cured_fit, _ = fit_cell(cured)
    raw_err = function_l2_errors(raw_fit, raw_system, central_grid(raw_x, 13))["drift_rel_l2"]
    cured_err = function_l2_errors(cured_fit, cured_system, central_grid(cured_x, 13))["drift_rel_l2"]
    assert cured_fit.bandwidth_meta["projected_group_folds"] == 3
    assert cured_err < raw_err


def test_a12_sign_accuracy_masks_near_zero_crossings():
    a_true = np.array([-1.0, -0.02, 0.0, 0.03, 1.0])
    a_hat = np.array([-1.0, 0.02, 1.0, -0.03, 1.0])
    assert a12_sign_accuracy(a_hat, a_true, relative_floor=0.10) == 1.0
    assert a12_sign_accuracy(a_hat, a_true, relative_floor=0.0) < 1.0
