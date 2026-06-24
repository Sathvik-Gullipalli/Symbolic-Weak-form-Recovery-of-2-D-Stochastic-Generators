from __future__ import annotations

import numpy as np

from sde2d.diffusion import lag1_noise_covariance, project_psd
from sde2d.generator import fit_generator_2d
from sde2d.library import make_library, polynomial_change_of_basis
from sde2d.metrics import central_grid, function_l2_errors, psd_validity, tensor_metrics
from sde2d.readouts.circulation import current_cosine, irreversibility_scalar
from sde2d.readouts.leverage import recover_heston_parameters
from sde2d.simulate import add_obs_noise
from sde2d.standardize import Standardizer
from sde2d.systems import CorrelatedOU2D, LogHestonSV, RotationalOU


def test_fit_generator_shapes_and_psd(quick_fit_kwargs):
    sys = CorrelatedOU2D()
    x = sys.simulate(dt=0.01, M=900, seed=2)
    fit = fit_generator_2d(x, dt=0.01, library=make_library("A"), **quick_fit_kwargs)
    pts = central_grid(x[:-1], 8)
    b, a = fit.evaluate(pts)
    assert b.shape == (64, 2)
    assert a.shape == (64, 2, 2)
    assert psd_validity(fit.evaluate(pts, psd=True)[1])["pct_psd_valid"] == 1.0


def test_covariance_bandwidth_rule_runs(quick_fit_kwargs):
    sys = CorrelatedOU2D()
    x = sys.simulate(dt=0.01, M=700, seed=12)
    fit = fit_generator_2d(x, dt=0.01, library=make_library("A"), bandwidth_rule="cov", **quick_fit_kwargs)
    assert fit.bandwidth_meta["bandwidth_rule"] == "cov"
    assert np.asarray(fit.bandwidth).shape == (2, 2)


def test_correlated_ou_a12_sign(quick_fit_kwargs):
    sys = CorrelatedOU2D(rho=-0.6)
    x = sys.simulate(dt=0.01, M=1600, seed=3)
    fit = fit_generator_2d(x, dt=0.01, library=make_library("A"), **quick_fit_kwargs)
    pts = central_grid(x[:-1], 10)
    metrics = tensor_metrics(fit, sys, pts)
    assert metrics["a12_sign_accuracy"] >= 0.9


def test_eiv_noise_covariance_is_psd():
    sys = CorrelatedOU2D()
    x = sys.simulate(dt=0.01, M=600, seed=4)
    noisy, _ = add_obs_noise(x, 0.2, seed=5)
    cov = lag1_noise_covariance(noisy)
    assert np.min(np.linalg.eigvalsh(cov)) >= -1e-12


def test_project_psd_clips_negative_eigenvalue():
    a = np.array([[1.0, 2.0], [2.0, 1.0]])
    p = project_psd(a)
    assert np.min(np.linalg.eigvalsh(p)) >= -1e-12


def test_heston_leverage_parameter_mapping(quick_fit_kwargs):
    sys = LogHestonSV(rho=-0.65)
    x = sys.simulate(dt=0.01, M=1800, seed=6)
    fit = fit_generator_2d(x, dt=0.01, library=make_library("D", ("X", "V")), **quick_fit_kwargs)
    params = recover_heston_parameters(fit)
    assert params["rho_hat"] < 0


def test_circulation_readout_positive(quick_fit_kwargs):
    sys = RotationalOU(omega=2.0)
    x = sys.simulate(dt=0.01, M=1600, seed=7)
    fit = fit_generator_2d(x, dt=0.01, library=make_library("A"), **quick_fit_kwargs)
    pts = central_grid(x[:-1], 10)
    assert irreversibility_scalar(fit) > 0.1
    assert current_cosine(fit, sys, pts) > 0.5


def test_standardizer_roundtrip_and_basis_change():
    x = np.column_stack([np.linspace(-1, 2, 20), np.linspace(0.5, 1.5, 20)])
    std = Standardizer().fit(x)
    assert np.allclose(std.inverse_state(std.transform(x)), x)
    m = polynomial_change_of_basis(std.mean, std.scale, 2)
    assert m.shape == (6, 6)
