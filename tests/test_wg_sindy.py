from __future__ import annotations

import numpy as np

from sde2d.library import make_library
from sde2d.metrics import central_grid, psd_validity
from sde2d.systems import CorrelatedOU2D
from sde2d.wg_sindy import WG_SINDY_DEFAULTS, fit_wg_sindy, wg_sindy_defaults


def test_wg_sindy_defaults_freeze_v6_operating_point():
    defaults = wg_sindy_defaults()
    assert defaults["center_scheme"] == "kmeans"
    assert defaults["bandwidth_rule"] == "cov"
    assert defaults["bandwidth_multiplier"] == 1.5
    assert defaults["local_poly_order"] == 2
    assert defaults["regressor"] == "adaptive_lasso"
    assert defaults["gls_weighting"] is True
    assert defaults["diffusion_parameterization"] == "chol"
    assert defaults["diffusion_shrinkage"] == 0.05
    assert WG_SINDY_DEFAULTS.gls_iterations == 1


def test_wg_sindy_psd_by_construction(quick_fit_kwargs):
    sys = CorrelatedOU2D(rho=-0.35)
    x = sys.simulate(dt=0.01, M=850, seed=8101)
    fit = fit_wg_sindy(
        x,
        dt=0.01,
        library=make_library("A"),
        n_centers=quick_fit_kwargs["n_centers"],
        center_scheme="kmeans",
        regressor="stlsq",
        regression_kw={"threshold": 0.02},
        seed=8102,
    )
    pts = central_grid(x[:-1], 7)
    assert fit.cholesky_diffusion is not None
    assert psd_validity(fit.evaluate(pts)[1])["pct_psd_valid"] == 1.0


def test_wg_sindy_gls_marks_second_pass(quick_fit_kwargs):
    sys = CorrelatedOU2D()
    x = sys.simulate(dt=0.01, M=850, seed=8103)
    fit = fit_wg_sindy(
        x,
        dt=0.01,
        library=make_library("A"),
        n_centers=quick_fit_kwargs["n_centers"],
        center_scheme="kmeans",
        regressor="stlsq",
        regression_kw={"threshold": 0.02},
        seed=8104,
    )
    assert fit.bandwidth_meta["gls_weighting"]
    assert fit.bandwidth_meta["gls_iterations"] == 1
    assert fit.selections["b1"].diagnostics["gls_weighting"]


def test_wg_sindy_reproducibility_lock(quick_fit_kwargs):
    sys = CorrelatedOU2D()
    x = sys.simulate(dt=0.01, M=820, seed=8105)
    kwargs = dict(
        dt=0.01,
        library=make_library("A"),
        n_centers=quick_fit_kwargs["n_centers"],
        center_scheme="kmeans",
        regressor="stlsq",
        regression_kw={"threshold": 0.02},
        seed=8106,
    )
    fit1 = fit_wg_sindy(x, **kwargs)
    fit2 = fit_wg_sindy(x, **kwargs)
    assert np.allclose(fit1.drift_coef, fit2.drift_coef)
    for key in fit1.diff_coef:
        assert np.allclose(fit1.diff_coef[key], fit2.diff_coef[key])

