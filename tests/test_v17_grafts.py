from __future__ import annotations

from pathlib import Path

import numpy as np

from sde2d.generator import fit_generator_2d
from sde2d.library import make_library
from sde2d.systems import REGISTRY

ROOT = Path(__file__).resolve().parents[1]


def _sample(system: str = "indep_ou", M: int = 420):
    sys = REGISTRY[system].cls()
    x = sys.simulate(dt=0.01, M=M, seed=177)
    return x[:-1], np.diff(x, axis=0)


def test_v17_flags_off_are_backward_compatible() -> None:
    cur, inc = _sample()
    kwargs = dict(dt=0.01, library=make_library("A"), n_centers=16, center_scheme="kmeans", local_poly_order=1, regressor="stlsq", regression_kw={"threshold": 0.03})
    a = fit_generator_2d(cur, inc, seed=178, **kwargs)
    b = fit_generator_2d(cur, inc, seed=178, tensor_rank="full", coverage_mode="off", domain="euclidean", library_atoms="poly", moment_order="euler", drift_lags=(1,), coord_transform="none", **kwargs)
    np.testing.assert_allclose(a.drift, b.drift, atol=0, rtol=0)
    for key in a.diffusion:
        np.testing.assert_allclose(a.diffusion[key], b.diffusion[key], atol=0, rtol=0)


def test_v17_reducing_flags_match_frozen_on_simple_system() -> None:
    cur, inc = _sample()
    kwargs = dict(dt=0.01, library=make_library("A"), n_centers=14, local_poly_order=1, regressor="stlsq", regression_kw={"threshold": 0.03})
    frozen = fit_generator_2d(cur, inc, seed=179, **kwargs)
    reduced = fit_generator_2d(cur, inc, seed=179, coord_transform="lamperti", drift_lags=(1,), moment_order="euler", tensor_rank="full", coverage_mode="off", domain="euclidean", **kwargs)
    np.testing.assert_allclose(frozen.drift, reduced.drift, atol=1e-12)


def test_v17_rank_auto_zeroes_degenerate_entries() -> None:
    cur, inc = _sample("underdamped_langevin", M=720)
    fit = fit_generator_2d(cur, inc, dt=0.01, library=make_library("B"), n_centers=18, local_poly_order=1, regressor="stlsq", regression_kw={"threshold": 0.04}, tensor_rank="auto", rank_floor=0.20, seed=180)
    points = cur[:: max(1, len(cur) // 20)]
    _, a = fit.evaluate(points)
    assert float(np.nanmedian(np.abs(a[:, 0, 0]))) < 0.1
    assert float(np.nanmedian(np.abs(a[:, 0, 1]))) < 0.1


def test_v17_enriched_library_has_atoms_and_no_estimator_branching() -> None:
    lib = make_library("POLY+TRIG")
    assert any("sin" in name or "cos" in name for name in lib.names)
    forbidden = {"gbm_2d", "two_factor_vasicek", "near_boundary_heston", "low_snr", "nsr"}
    text = (ROOT / "src/sde2d/generator.py").read_text()
    assert not [token for token in forbidden if token in text]


def test_v17_psd_preserved_with_auto_rank() -> None:
    cur, inc = _sample("near_singular", M=540)
    fit = fit_generator_2d(cur, inc, dt=0.01, library=make_library("A"), n_centers=14, local_poly_order=1, regressor="stlsq", regression_kw={"threshold": 0.04}, tensor_rank="auto", seed=181)
    _, a = fit.evaluate(cur[:: max(1, len(cur) // 30)], psd=True)
    eig = np.linalg.eigvalsh(a)
    assert np.nanmin(eig) >= -1e-10
