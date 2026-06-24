from __future__ import annotations

from pathlib import Path

import numpy as np

from experiments.archive.v15.common import config_for_grafts, fit_and_score
from experiments.archive.v15.v15_systems import CLUSTER
from sde2d.generator import fit_generator_2d
from sde2d.library import make_library
from sde2d.regression import solve
from sde2d.systems import CorrelatedOU2D, IndependentOU2D

ROOT = Path(__file__).resolve().parents[1]


def test_v15_defaults_are_byte_identical_to_frozen_path() -> None:
    sys = CorrelatedOU2D()
    x = sys.simulate(dt=0.01, M=420, seed=151)
    kwargs = dict(
        dt=0.01,
        library=make_library("A"),
        n_centers=18,
        center_scheme="kmeans",
        bandwidth_rule="cov",
        local_poly_order=1,
        regressor="stlsq",
        regression_kw={"threshold": 0.03},
        gls_weighting=True,
        gls_iterations=1,
    )
    before = fit_generator_2d(x, seed=152, **kwargs)
    after = fit_generator_2d(x, seed=152, **kwargs)
    assert np.array_equal(before.drift, after.drift)
    for key in before.diffusion:
        assert np.array_equal(before.diffusion[key], after.diffusion[key])


def test_v15_full_tensor_gls_reduces_to_diagonal_on_homoscedastic_independent_ou() -> None:
    sys = IndependentOU2D()
    x = sys.simulate(dt=0.01, M=640, seed=153)
    kwargs = dict(
        dt=0.01,
        library=make_library("A"),
        n_centers=20,
        center_scheme="kmeans",
        bandwidth_rule="cov",
        local_poly_order=1,
        regressor="stlsq",
        regression_kw={"threshold": 0.03},
        gls_weighting=True,
        gls_iterations=1,
    )
    diagonal = fit_generator_2d(x, seed=154, gls_mode="diagonal", **kwargs)
    full = fit_generator_2d(x, seed=154, gls_mode="full_tensor", **kwargs)
    assert np.allclose(diagonal.drift, full.drift, atol=1e-6)
    assert full.bandwidth_meta["gls_mode"] == "full_tensor"


def test_v15_estimator_path_has_no_system_name_or_low_snr_branching() -> None:
    forbidden = set(CLUSTER) | {
        "indep_ou",
        "correlated_ou",
        "rotational_ou",
        "double_well_transverse",
        "nondiag_cholesky",
        "diag_multiplicative",
        "low_snr",
        "nsr",
    }
    for rel in ["src/sde2d/generator.py", "src/sde2d/regression.py", "experiments/v15/common.py"]:
        text = (ROOT / rel).read_text()
        hits = [token for token in forbidden if token in text]
        assert not hits, (rel, hits)


def test_v15_noise_floor_keeps_small_significant_term_relative_drops() -> None:
    rng = np.random.default_rng(155)
    x = rng.normal(size=(240, 2))
    y = 10.0 * x[:, 0] + 0.15 * x[:, 1] + 0.01 * rng.normal(size=240)
    relative = solve(x, y, method="stlsq", threshold=0.12, threshold_mode="relative")
    noise_floor = solve(x, y, method="stlsq", threshold=0.12, threshold_mode="noise_floor", noise_floor_z=1.5)
    assert not relative.support[1]
    assert noise_floor.support[1]


def test_v15_psd_preserved_on_cluster_quick() -> None:
    cfg = config_for_grafts(
        ("G1", "G2"),
        "TEST_G1_G2",
        n_centers=12,
        local_poly_order=1,
        regressor="stlsq",
        diffusion_parameterization="chol",
        regression_kw={"threshold": 0.03},
    )
    for system in CLUSTER:
        row = fit_and_score(system, cfg, R=2, steps=220, seed=156, grid_n=7)
        assert float(row["psd_pct"]) >= 0.99
