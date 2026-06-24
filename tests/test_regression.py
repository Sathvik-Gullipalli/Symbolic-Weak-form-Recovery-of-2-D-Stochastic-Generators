from __future__ import annotations

import numpy as np

from sde2d.regression import ColumnNormalizer, fit_lassocv_debias_stlsq, solve, stlsq


def test_column_normalizer_round_trip():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(30, 4))
    c = np.array([1.0, -2.0, 0.5, 0.0])
    y = a @ c
    norm = ColumnNormalizer().fit(a)
    xs = norm.transform(a)
    coef_norm = np.linalg.lstsq(xs, y, rcond=None)[0]
    assert np.allclose(norm.coefficients_to_raw(coef_norm), c)


def test_stlsq_prunes_small_terms():
    x = np.eye(4)
    y = np.array([1.0, 0.01, 0.4, 0.0])
    coef, iters = stlsq(x, y, y, tau=0.25, n_iter=20)
    assert iters <= 20
    assert coef[1] == 0.0
    assert coef[0] != 0.0


def test_grouped_lassocv_path_runs_when_groups_match_rows():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(18, 3))
    y = x @ np.array([1.0, 0.0, -0.5]) + 0.01 * rng.normal(size=18)
    groups = np.repeat(np.arange(6), 3)
    res = fit_lassocv_debias_stlsq(x, y, groups, stlsq_threshold=0.01)
    assert res.diagnostics["cv_folds"] >= 2
    assert res.support.shape == (3,)


def test_lassocv_single_group_uses_pseudo_blocks():
    rng = np.random.default_rng(11)
    x = rng.normal(size=(30, 4))
    y = x @ np.array([0.7, 0.0, -0.4, 0.0]) + 0.02 * rng.normal(size=30)
    groups = np.zeros(30, dtype=int)
    res = solve(x, y, groups=groups, method="lasso_stlsq", stlsq_threshold=0.01, pseudo_blocks=5)
    assert res.diagnostics["cv_pseudo_blocks"]
    assert res.diagnostics["cv_folds"] >= 2


def test_elastic_net_and_adaptive_lasso_dispatch_run():
    rng = np.random.default_rng(12)
    x = rng.normal(size=(36, 5))
    y = x @ np.array([1.0, 0.0, -0.6, 0.0, 0.2]) + 0.01 * rng.normal(size=36)
    groups = np.repeat(np.arange(6), 6)
    elastic = solve(x, y, groups=groups, method="elastic_net", stlsq_threshold=0.01)
    adaptive = solve(x, y, groups=groups, method="adaptive_lasso", stlsq_threshold=0.01, gamma=1.0)
    assert elastic.method == "elastic_net_debias_stlsq"
    assert adaptive.method == "adaptive_lasso_debias_stlsq"
    assert elastic.diagnostics["cv_folds"] >= 2
    assert adaptive.diagnostics["cv_folds"] >= 2


def test_svd_threshold_reports_rank():
    x = np.column_stack([np.ones(12), np.arange(12), np.arange(12)])
    y = np.arange(12)
    res = solve(x, y, method="svd_threshold", threshold=0.01, svd_rtol=1e-8)
    assert res.method == "svd_threshold"
    assert res.diagnostics["svd_rank"] < x.shape[1]


def test_rank_deficiency_flag():
    x = np.column_stack([np.ones(8), np.arange(8), np.arange(8)])
    y = np.arange(8)
    res = solve(x, y, method="stlsq")
    assert res.diagnostics["rank_deficient"]
