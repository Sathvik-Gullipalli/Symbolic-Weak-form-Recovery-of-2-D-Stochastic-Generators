from __future__ import annotations

import numpy as np

from experiments.archive.v5_5.run_v5_5_campaign import (
    HESTON_PRICE_DRIFT_NULL_SYSTEMS,
    IN_SCOPE_SYSTEMS,
    OUT_OF_SCOPE_SYSTEMS,
    REQUIRED_GRAFTS,
    annotate_scope_row,
    loo_variants,
    targeted_variants,
    v55_full_stack,
)
from sde2d.generator import fit_generator_2d
from sde2d.library import make_library
from sde2d.systems import CorrelatedOU2D


def test_v55_scope_is_exact_positive_class():
    assert len(IN_SCOPE_SYSTEMS) == 13
    assert {"heston_sv", "heston_logsv", "cir_pair"}.issubset(IN_SCOPE_SYSTEMS)
    assert set(IN_SCOPE_SYSTEMS).isdisjoint(OUT_OF_SCOPE_SYSTEMS)
    assert {"near_singular", "underdamped_langevin", "near_boundary_heston", "partial_observation"}.issubset(OUT_OF_SCOPE_SYSTEMS)


def test_heston_scope_objective_masks_log_price_drift():
    row = {
        "system": "heston_logsv",
        "drift_rel_l2": "9.0",
        "b1_rel_l2": "7.0",
        "b2_rel_l2": "0.2",
        "diffusion_rel_l2": "0.1",
        "psd_valid_pct": "1.0",
        "a12_cosine": "0.99",
    }
    annotated = annotate_scope_row(row)
    assert annotated["scope_status"] == "IN_SCOPE"
    assert annotated["objective_drift_rel_l2"] == 0.2
    assert annotated["log_price_drift_rel_l2"] == 7.0
    assert "b1_log_price_drift_reported_null" in annotated["scope_metric_contract"]
    assert HESTON_PRICE_DRIFT_NULL_SYSTEMS == {"heston_sv", "heston_logsv"}


def test_v55_loo_variants_cover_required_grafts():
    variants = loo_variants()
    ids = {v.variant_id for v in variants}
    for graft in REQUIRED_GRAFTS:
        assert f"V55LOO_{graft}" in ids
    full = next(v for v in variants if v.variant_id == "V55_FULL_SCOPE_GRAFT_STACK")
    assert full.gls_weighting
    assert full.gls_iterations == 2
    assert full.diffusion_parameterization == "chol"


def test_v55_targeted_combo_limit_and_mechanisms():
    variants = targeted_variants(v55_full_stack(), limit=15)
    assert len(variants) <= 15
    ids = {v.variant_id for v in variants}
    assert {"V55_GLS_ITER4", "V55_DIFFMETRIC_H125", "V55_OFFDIAG_STABILITY_2STAGE"}.issubset(ids)
    assert any(v.bandwidth_rule == "diffusion_metric" for v in variants)
    assert any(v.target_regression_kw for v in variants)


def test_diffusion_metric_kernel_and_iterated_gls_run(quick_fit_kwargs):
    sys = CorrelatedOU2D()
    x = sys.simulate(dt=0.01, M=900, seed=7101)
    fit = fit_generator_2d(
        x,
        dt=0.01,
        library=make_library("A"),
        bandwidth_rule="diffusion_metric",
        gls_weighting=True,
        gls_iterations=2,
        target_regression_kw={"a12": {"threshold": 0.01}},
        **quick_fit_kwargs,
    )
    pts = x[::100][1:6]
    _, a = fit.evaluate(pts)
    assert a.shape == (pts.shape[0], 2, 2)
    assert np.asarray(fit.bandwidth).shape[-2:] == (2, 2)
    assert fit.bandwidth_meta["gls_iterations"] == 2
    assert fit.selections["b1"].diagnostics["gls_iteration"] == 2
    assert fit.selections["a12"].diagnostics["target_regression_kw"]["threshold"] == 0.01

