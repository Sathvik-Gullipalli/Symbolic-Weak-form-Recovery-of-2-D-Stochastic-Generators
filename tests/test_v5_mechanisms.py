from __future__ import annotations

import numpy as np

from experiments.archive.v5.run_v5_campaign import graft_ablation_variants, variant_catalog
from sde2d.generator import fit_generator_2d
from sde2d.kernels import choose_centers, local_polynomial_projection_matrix
from sde2d.library import make_library
from sde2d.regression import solve
from sde2d.systems import CorrelatedOU2D


def test_local_linear_projection_preserves_constants():
    rng = np.random.default_rng(501)
    z = rng.normal(size=(80, 2))
    centers = choose_centers(z, 16, seed=502, scheme="quantile_grid", grid_shape=(4, 4))
    p = local_polynomial_projection_matrix(z, centers, 1.2, order=1)
    assert p.shape == (16, 80)
    assert np.allclose(p @ np.ones(z.shape[0]), np.ones(centers.shape[0]), atol=1e-6)


def test_cholesky_diffusion_parameterization_is_psd(quick_fit_kwargs):
    sys = CorrelatedOU2D(rho=-0.35)
    x = sys.simulate(dt=0.01, M=900, seed=503)
    fit = fit_generator_2d(
        x,
        dt=0.01,
        library=make_library("A"),
        diffusion_parameterization="chol",
        **quick_fit_kwargs,
    )
    pts = x[::80][1:8]
    _, a = fit.evaluate(pts)
    assert fit.cholesky_diffusion is not None
    assert np.min(np.linalg.eigvalsh(a)) >= -1e-9


def test_gls_weighting_marks_drift_refit(quick_fit_kwargs):
    sys = CorrelatedOU2D()
    x = sys.simulate(dt=0.01, M=850, seed=504)
    fit = fit_generator_2d(
        x,
        dt=0.01,
        library=make_library("A"),
        gls_weighting=True,
        **quick_fit_kwargs,
    )
    assert fit.bandwidth_meta["gls_weighting"]
    assert fit.selections["b1"].diagnostics["gls_weighting"]


def test_new_v5_regression_dispatches_are_deterministic():
    rng = np.random.default_rng(505)
    x = rng.normal(size=(48, 5))
    beta = np.array([0.8, 0.0, -0.5, 0.0, 0.2])
    y = x @ beta + 0.01 * rng.normal(size=48)
    methods = ["bic", "sr3", "omp", "tls", "huber", "ridge_gcv"]
    for method in methods:
        r1 = solve(x, y, method=method, seed=7)
        r2 = solve(x, y, method=method, seed=7)
        assert r1.coef.shape == beta.shape
        assert np.allclose(r1.coef, r2.coef)


def test_v5_catalog_has_no_reasonless_deferred_rows():
    catalog = variant_catalog()
    assert len(catalog) == 123
    blocked = [v for v in catalog if not v.implemented]
    assert blocked
    assert all(v.infeasible_reason and v.core_identity_preserving == "no" for v in blocked)


def test_v5_graft_ablation_variants_have_add_and_leave_one_out_pairs():
    variants = graft_ablation_variants()
    ids = {v.variant_id for v in variants}
    for name in ["LOCAL_POLY", "GLS", "CHOL", "LOCAL_COV", "SHRINKAGE", "ADAPTIVE_REG"]:
        assert f"V5ADD_{name}" in ids
        assert f"V5LOO_{name}" in ids
