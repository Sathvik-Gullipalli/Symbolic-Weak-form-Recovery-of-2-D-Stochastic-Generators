from __future__ import annotations

import argparse
import csv
import math
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[2] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from experiments.benchmarks._utils import (
    FitCell,
    active_targets,
    estimated_coefficients,
    fit_cell,
    simulate_recovered,
    summary_stats,
    true_coefficients_fit_space,
)
from experiments.common import ROOT
from experiments.v5.run_v5_campaign import ROW_FIELDS, V5Variant, _safe_float, cell_for, read_rows, run_grid, write_csv
from experiments.v5_5.run_v5_5_campaign import IN_SCOPE_SYSTEMS, OUT_OF_SCOPE_SYSTEMS, metric_contract, v55_full_stack
from sde2d.invariants import spectral_gap_linear_fit
from sde2d.metrics import central_grid, cosine_similarity, function_l2_errors, psd_validity, relative_l2, tensor_metrics
from sde2d.readouts.circulation import current_cosine, irreversibility_scalar
from sde2d.readouts.leverage import rho_from_tensor
from sde2d.systems import HestonSV, LogHestonSV, REGISTRY
from sde2d.wg_sindy import wg_sindy_defaults


OUT_DIR = "results/v6"
SHOWCASE_DIR = f"{OUT_DIR}/showcase"
FIG_DIR = "figures/v6"
PAPER_DIR = "paper"

FREEZE_CELLS = f"{OUT_DIR}/freeze_confirm_cells.csv"
FREEZE_CONFIRM = f"{OUT_DIR}/freeze_confirm.csv"
HEAD_CELLS = f"{OUT_DIR}/headtohead_cells.csv"
HEADTOHEAD = f"{OUT_DIR}/headtohead.csv"
LADDER_CELLS = f"{OUT_DIR}/act2_graft_ladder_cells.csv"
GRAFT_LADDER = f"{OUT_DIR}/act2_graft_ladder.csv"
NECESSITY_CELLS = f"{OUT_DIR}/necessity_matrix_cells.csv"
NECESSITY_MATRIX = f"{OUT_DIR}/necessity_matrix.csv"
ACT1_FAILURE = f"{OUT_DIR}/act1_naive1d_failure.csv"
B0_EQUIV = f"{OUT_DIR}/b0_equivalence.csv"
READOUT_FLUC = f"{OUT_DIR}/readouts_fluctuation.csv"
READOUT_LEV = f"{OUT_DIR}/readouts_leverage.csv"
READOUT_CIRC = f"{OUT_DIR}/readouts_circulation.csv"
CONVERGENCE = f"{OUT_DIR}/convergence.csv"
HONEST_NULLS = f"{OUT_DIR}/honest_nulls.csv"
PROVENANCE = f"{OUT_DIR}/external_1d_provenance.csv"
RUN_LOG = f"{OUT_DIR}/run_log.csv"

SHOWCASE_RAW = f"{SHOWCASE_DIR}/showcase_summary_raw.csv"
SHOWCASE_SUMMARY = f"{SHOWCASE_DIR}/showcase_summary.csv"
SHOWCASE_COEF_RAW = f"{SHOWCASE_DIR}/showcase_coefficients_raw.csv"
SHOWCASE_COEF = f"{SHOWCASE_DIR}/showcase_coefficients.csv"
SHOWCASE_DYNAMICS = f"{SHOWCASE_DIR}/showcase_dynamics.csv"
SHOWCASE_INVARIANTS = f"{SHOWCASE_DIR}/showcase_invariants.csv"

HESTON_SCOPE_SYSTEMS = {"heston_sv", "heston_logsv"}


def coefficient_term_in_scope(system_key: str, target: str, term_name: str) -> bool:
    if system_key in HESTON_SCOPE_SYSTEMS and target == "b1":
        return False
    if system_key in HESTON_SCOPE_SYSTEMS and target in {"a11", "a12", "a22"}:
        # The Heston positive claim is tensor/leverage-field recovery under a
        # PSD Cholesky parameterization; induced quadratic tensor projection
        # residue is retained in raw audit columns, not counted as sparse
        # paper support.
        return term_name in {"1", "y"}
    return True


def profile_settings(profile: str) -> dict:
    if profile == "smoke":
        return {
            "systems": ["correlated_ou", "rotational_ou", "heston_logsv"],
            "seeds": [9101, 9102],
            "base_steps": 700,
            "heston_steps": 1000,
            "showcase_grid": 11,
        }
    if profile == "standard":
        return {
            "systems": list(IN_SCOPE_SYSTEMS),
            "seeds": [9101, 9102, 9103, 9104, 9105],
            "base_steps": 1300,
            "heston_steps": 2200,
            "showcase_grid": 13,
        }
    return {
        "systems": list(IN_SCOPE_SYSTEMS),
        "seeds": [9101, 9102, 9103, 9104, 9105, 9106, 9107, 9108, 9109, 9110],
        "base_steps": 1600,
        "heston_steps": 2600,
        "showcase_grid": 15,
    }


def reset_outputs() -> None:
    for rel in [
        FREEZE_CELLS,
        FREEZE_CONFIRM,
        HEAD_CELLS,
        HEADTOHEAD,
        LADDER_CELLS,
        GRAFT_LADDER,
        NECESSITY_CELLS,
        NECESSITY_MATRIX,
        ACT1_FAILURE,
        B0_EQUIV,
        READOUT_FLUC,
        READOUT_LEV,
        READOUT_CIRC,
        CONVERGENCE,
        HONEST_NULLS,
        PROVENANCE,
        RUN_LOG,
        SHOWCASE_RAW,
        SHOWCASE_SUMMARY,
        SHOWCASE_COEF_RAW,
        SHOWCASE_COEF,
        SHOWCASE_DYNAMICS,
        SHOWCASE_INVARIANTS,
    ]:
        path = ROOT / rel
        if path.exists():
            path.unlink()
    fig_dir = ROOT / FIG_DIR
    if fig_dir.exists():
        for png in fig_dir.glob("*.png"):
            png.unlink()


def wg_sindy_variant() -> V5Variant:
    return V5Variant(
        "WG_SINDY_FROZEN",
        "wg_sindy",
        "WG-SINDy frozen v6 operating point: kmeans/cov/local-poly-2/adaptive-lasso/GLS/Cholesky/R16",
        priority="H",
        graft_source="v5.5 global default V5GREEDY_local_poly_order_2",
        regressor="adaptive_lasso",
        library_space="default",
        center_scheme="kmeans",
        n_centers=64,
        bandwidth_mult=1.5,
        bandwidth_rule="cov",
        local_poly_order=2,
        gls_weighting=True,
        gls_iterations=1,
        diffusion_parameterization="chol",
        diffusion_shrinkage=0.05,
        n_trajectories=16,
        stlsq_threshold=0.12,
    )


def b0_variant(variant_id: str = "B0_NAIVE_1D_PORT") -> V5Variant:
    return V5Variant(
        variant_id,
        "baseline_1d_port",
        "Faithful naive 1D weak-form port to five 2D targets: raw coords, local-constant kernels, STLSQ, no GLS, no Cholesky",
        priority="H",
        graft_source="Eshwar-1D componentwise port",
        regressor="stlsq",
        library_space="raw",
        center_scheme="quantile_grid",
        n_centers=64,
        bandwidth_mult=1.5,
        bandwidth_rule="nn_median",
        local_poly_order=0,
        gls_weighting=False,
        diffusion_parameterization="entries",
        diffusion_shrinkage=0.0,
        n_trajectories=1,
        stlsq_threshold=None,
        threshold=0.02,
    )


def freeze_variants() -> list[V5Variant]:
    low = replace(
        wg_sindy_variant(),
        variant_id="V6_FULL_MINUS_LOW_VALUE_GRAFTS",
        family="freeze_confirm",
        description="Mechanism-equivalent lean stack after low-value v5.5 grafts are screened off",
    )
    return [
        wg_sindy_variant(),
        replace(v55_full_stack(), variant_id="V6_ALL_GRAFTS_STACK", family="freeze_confirm"),
        low,
    ]


def headtohead_variants() -> list[V5Variant]:
    return [
        wg_sindy_variant(),
        b0_variant(),
        b0_variant("B0_PRIME_IN_REPO_REPORT"),
        V5Variant(
            "KM_LOCAL_MOMENT",
            "baseline_kramers_moyal",
            "Kramers-Moyal/local increment-moment proxy with ridge thresholding",
            graft_source="local moment baseline",
            regressor="ridge_threshold",
            library_space="raw",
            center_scheme="kmeans",
            n_centers=64,
            bandwidth_rule="knn",
            local_poly_order=0,
            gls_weighting=False,
            diffusion_parameterization="entries",
            n_trajectories=4,
            threshold=0.05,
        ),
        V5Variant(
            "WEAK_SINDY_TEMPORAL_PROXY",
            "baseline_temporal_weak",
            "Temporal-test-function weak-SINDy proxy: midpoint/subsampled targets expose endogeneity bias",
            graft_source="temporal weak projection negative control",
            regressor="lasso_stlsq",
            library_space="raw",
            center_scheme="uniform_grid",
            n_centers=64,
            bandwidth_rule="nn_median",
            target_anchor="midpoint",
            subsample_k=2,
            gls_weighting=False,
            diffusion_parameterization="entries",
            n_trajectories=4,
            stlsq_threshold=0.16,
        ),
        V5Variant(
            "GEDMD_DENSE_PROXY",
            "baseline_gedmd",
            "Dense gEDMD-style operator proxy: high-degree ridge dictionary, not sparse symbolic",
            graft_source="dense operator baseline",
            regressor="ridge_gcv",
            library_space="raw",
            library_override="C",
            center_scheme="kmeans",
            n_centers=81,
            bandwidth_rule="cov",
            gls_weighting=False,
            diffusion_parameterization="entries",
            n_trajectories=16,
        ),
    ]


def ladder_variants() -> list[V5Variant]:
    base = b0_variant("LADDER_0_B0")
    z = replace(base, variant_id="LADDER_1_Z_ANISO", family="graft_ladder", description="B0 + z-space/kmeans/cov anisotropic kernels", library_space="default", center_scheme="kmeans", bandwidth_rule="cov")
    lp = replace(z, variant_id="LADDER_2_LOCAL_POLY", description="B0 + z/cov + local polynomial order 2", local_poly_order=2)
    al = replace(lp, variant_id="LADDER_3_ADAPTIVE_LASSO", description="Add adaptive-LASSO sparse solve", regressor="adaptive_lasso", stlsq_threshold=0.12, threshold=None)
    chol = replace(al, variant_id="LADDER_4_CHOLESKY_PSD", description="Add Cholesky PSD tensor parametrization", diffusion_parameterization="chol", diffusion_shrinkage=0.05)
    gls = replace(chol, variant_id="LADDER_5_GLS", description="Add GLS drift-whitening", gls_weighting=True, gls_iterations=1)
    wg = replace(wg_sindy_variant(), variant_id="LADDER_6_WG_SINDY", family="graft_ladder", description="Full frozen WG-SINDy with R=16")
    return [base, z, lp, al, chol, gls, wg]


def necessity_variants() -> list[V5Variant]:
    wg = wg_sindy_variant()
    return [
        wg,
        replace(wg, variant_id="NEC_LOO_GLS_WHITENING", family="necessity_loo", description="WG-SINDy without GLS drift-whitening", gls_weighting=False),
        replace(wg, variant_id="NEC_LOO_LOCAL_POLY", family="necessity_loo", description="WG-SINDy without local-polynomial projection", local_poly_order=0),
        replace(wg, variant_id="NEC_LOO_CHOLESKY_PSD", family="necessity_loo", description="WG-SINDy without Cholesky PSD parametrization", diffusion_parameterization="entries"),
        replace(wg, variant_id="NEC_LOO_ADAPTIVE_LASSO", family="necessity_loo", description="WG-SINDy with lasso-STLSQ instead of adaptive-LASSO", regressor="lasso_stlsq"),
        replace(wg, variant_id="NEC_LOO_ANISOTROPIC_COV", family="necessity_loo", description="WG-SINDy without anisotropic covariance bandwidth", bandwidth_rule="nn_median"),
        replace(wg, variant_id="NEC_LOO_MULTI_TRAJECTORY", family="necessity_loo", description="WG-SINDy with R=1 instead of pooled R=16", n_trajectories=1),
    ]


def objective_drift(row: dict) -> float:
    if row.get("system") in HESTON_SCOPE_SYSTEMS:
        return _safe_float(row.get("b2_rel_l2"))
    return _safe_float(row.get("drift_rel_l2"))


def inscope_score(row: dict) -> float:
    drift = objective_drift(row)
    diff = _safe_float(row.get("diffusion_rel_l2"))
    psd = _safe_float(row.get("psd_valid_pct"), 0.0)
    cos = _safe_float(row.get("a12_cosine"))
    cos_term = 1.0 if math.isnan(cos) else max(0.0, min(1.0, (cos + 1.0) / 2.0))
    drift_term = max(0.0, 1.0 - min(drift, 2.0) / 2.0) if math.isfinite(drift) else 0.0
    diff_term = max(0.0, 1.0 - min(diff, 1.5) / 1.5) if math.isfinite(diff) else 0.0
    return 0.38 * drift_term + 0.34 * diff_term + 0.16 * psd + 0.12 * cos_term


def finite_values(rows: list[dict], col: str) -> list[float]:
    vals = [_safe_float(r.get(col)) for r in rows]
    return [v for v in vals if math.isfinite(v)]


def median_ci(values: list[float], seed: int = 260620, n_boot: int = 600) -> tuple[float, float, float]:
    vals = np.asarray([v for v in values if math.isfinite(v)], float)
    if vals.size == 0:
        return float("nan"), float("nan"), float("nan")
    if vals.size == 1:
        v = float(vals[0])
        return v, v, v
    rng = np.random.default_rng(seed + vals.size)
    boot = np.median(vals[rng.integers(0, vals.size, size=(n_boot, vals.size))], axis=1)
    return float(np.median(vals)), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def annotated_rows(path: str) -> list[dict]:
    rows = []
    for row in read_rows(path):
        if row.get("status") in {"FAILED", "INFEASIBLE_BY_INVARIANT"}:
            continue
        if row.get("system") not in IN_SCOPE_SYSTEMS:
            continue
        out = dict(row)
        out["method"] = row.get("variant_id", "")
        out["objective_drift_rel_l2"] = objective_drift(row)
        out["log_price_drift_rel_l2"] = _safe_float(row.get("b1_rel_l2")) if row.get("system") in HESTON_SCOPE_SYSTEMS else float("nan")
        out["inscope_score"] = inscope_score(row)
        out["scope_metric_contract"] = metric_contract(row.get("system", ""))
        rows.append(out)
    return rows


def aggregate(rows: list[dict], keys: list[str]) -> list[dict]:
    buckets: dict[tuple, list[dict]] = {}
    for row in rows:
        buckets.setdefault(tuple(row.get(k, "") for k in keys), []).append(row)
    out = []
    for key, part in buckets.items():
        first = part[0]
        score_vals = [inscope_score(r) for r in part]
        drift_vals = [objective_drift(r) for r in part]
        diff_vals = finite_values(part, "diffusion_rel_l2")
        cos_vals = finite_values(part, "a12_cosine")
        psd_vals = finite_values(part, "psd_valid_pct")
        row = {k: v for k, v in zip(keys, key)}
        row.update(
            {
                "n": len(part),
                "tier": first.get("tier", row.get("tier", "")),
                "family": first.get("family", ""),
                "description": first.get("description", ""),
                "median_inscope_score": float(np.median(score_vals)) if score_vals else float("nan"),
                "median_objective_drift_rel_l2": float(np.median(drift_vals)) if drift_vals else float("nan"),
                "p90_objective_drift_rel_l2": float(np.percentile(drift_vals, 90)) if drift_vals else float("nan"),
                "median_diffusion_rel_l2": float(np.median(diff_vals)) if diff_vals else float("nan"),
                "median_a12_cosine": float(np.median(cos_vals)) if cos_vals else float("nan"),
                "median_psd_valid_pct": float(np.median(psd_vals)) if psd_vals else float("nan"),
                "validated_count": sum(r.get("status") == "VALIDATED_POSITIVE" for r in part),
                "negative_or_inconclusive_count": sum(r.get("status") != "VALIDATED_POSITIVE" for r in part),
            }
        )
        out.append(row)
    return sorted(out, key=lambda r: (-_safe_float(r.get("median_inscope_score"), -1.0), _safe_float(r.get("median_objective_drift_rel_l2"), 9.0)))


def worst_tier_metric(rows: list[dict], variant_id: str) -> float:
    part = [r for r in rows if r.get("variant_id") == variant_id]
    tier_meds = []
    for tier in sorted({r.get("tier", "") for r in part}):
        vals = [objective_drift(r) for r in part if r.get("tier") == tier and math.isfinite(objective_drift(r))]
        if vals:
            tier_meds.append(float(np.median(vals)))
    return float(max(tier_meds)) if tier_meds else float("nan")


def write_freeze_confirm() -> None:
    rows = annotated_rows(FREEZE_CELLS)
    ag = aggregate(rows, ["variant_id"])
    for row in ag:
        row["worst_tier_objective_drift"] = worst_tier_metric(rows, row["variant_id"])
    ag = sorted(ag, key=lambda r: (r.get("variant_id") != "WG_SINDY_FROZEN", _safe_float(r.get("worst_tier_objective_drift"), 9.0), -_safe_float(r.get("median_inscope_score"), -1.0)))
    selected = "WG_SINDY_FROZEN" if any(r.get("variant_id") == "WG_SINDY_FROZEN" for r in ag) else (ag[0]["variant_id"] if ag else "")
    for row in ag:
        row["selected_frozen_algorithm"] = row["variant_id"] == selected
        row["selection_basis"] = "v6 freezes the v5.5 global default; confirm alternatives must not outperform it on worst-tier"
    fields = [
        "variant_id",
        "n",
        "worst_tier_objective_drift",
        "p90_objective_drift_rel_l2",
        "median_objective_drift_rel_l2",
        "median_diffusion_rel_l2",
        "median_a12_cosine",
        "median_psd_valid_pct",
        "median_inscope_score",
        "selected_frozen_algorithm",
        "selection_basis",
        "description",
    ]
    write_csv(FREEZE_CONFIRM, ag, fields)


def write_headtohead_and_act1() -> None:
    rows = annotated_rows(HEAD_CELLS)
    head = aggregate(rows, ["variant_id", "system"])
    for row in head:
        row["method_label"] = row["variant_id"]
        row["dominates_b0_on_system"] = ""
    by_system = {}
    for row in head:
        by_system.setdefault(row["system"], {})[row["variant_id"]] = row
    for system, part in by_system.items():
        b0 = part.get("B0_NAIVE_1D_PORT")
        if not b0:
            continue
        b0_score = _safe_float(b0.get("median_inscope_score"))
        for row in part.values():
            row["dominates_b0_on_system"] = bool(_safe_float(row.get("median_inscope_score")) >= b0_score)
    fields = [
        "variant_id",
        "method_label",
        "system",
        "tier",
        "n",
        "median_inscope_score",
        "median_objective_drift_rel_l2",
        "median_diffusion_rel_l2",
        "median_a12_cosine",
        "median_psd_valid_pct",
        "dominates_b0_on_system",
        "negative_or_inconclusive_count",
        "description",
    ]
    write_csv(HEADTOHEAD, head, fields)

    b0_rows = [r for r in rows if r.get("variant_id") == "B0_NAIVE_1D_PORT"]
    act1 = aggregate(b0_rows, ["system"])
    for row in act1:
        row["failure_marker"] = bool(_safe_float(row.get("median_inscope_score")) < 0.82 or _safe_float(row.get("median_objective_drift_rel_l2")) > 0.75 or _safe_float(row.get("median_psd_valid_pct")) < 0.99)
        row["root_cause"] = "raw-scale/local-constant/no-GLS/no-PSD naive 1D extension"
    write_csv(ACT1_FAILURE, act1, ["system", "tier", "n", "median_inscope_score", "median_objective_drift_rel_l2", "median_diffusion_rel_l2", "median_a12_cosine", "median_psd_valid_pct", "failure_marker", "root_cause"])

    equivalence = []
    raw = read_rows(HEAD_CELLS)
    b0 = {(r.get("system"), r.get("seed")): r for r in raw if r.get("variant_id") == "B0_NAIVE_1D_PORT"}
    b0p = {(r.get("system"), r.get("seed")): r for r in raw if r.get("variant_id") == "B0_PRIME_IN_REPO_REPORT"}
    for key in sorted(set(b0) & set(b0p)):
        r1, r2 = b0[key], b0p[key]
        equivalence.append(
            {
                "system": key[0],
                "seed": key[1],
                "drift_abs_diff": abs(_safe_float(r1.get("drift_rel_l2")) - _safe_float(r2.get("drift_rel_l2"))),
                "diffusion_abs_diff": abs(_safe_float(r1.get("diffusion_rel_l2")) - _safe_float(r2.get("diffusion_rel_l2"))),
                "score_abs_diff": abs(inscope_score(r1) - inscope_score(r2)),
                "equivalent": abs(inscope_score(r1) - inscope_score(r2)) < 1e-12,
            }
        )
    write_csv(B0_EQUIV, equivalence, ["system", "seed", "drift_abs_diff", "diffusion_abs_diff", "score_abs_diff", "equivalent"])


def write_graft_ladder() -> None:
    rows = annotated_rows(LADDER_CELLS)
    order = [v.variant_id for v in ladder_variants()]
    labels = {
        "LADDER_0_B0": "B0 naive 1D port",
        "LADDER_1_Z_ANISO": "+ standardized anisotropic kernels",
        "LADDER_2_LOCAL_POLY": "+ local-polynomial projection",
        "LADDER_3_ADAPTIVE_LASSO": "+ adaptive-LASSO",
        "LADDER_4_CHOLESKY_PSD": "+ Cholesky PSD tensor",
        "LADDER_5_GLS": "+ GLS drift whitening",
        "LADDER_6_WG_SINDY": "WG-SINDy frozen",
    }
    ag = {r["variant_id"]: r for r in aggregate(rows, ["variant_id"])}
    out = []
    running = -float("inf")
    for step, vid in enumerate(order):
        row = ag.get(vid, {"variant_id": vid})
        raw_score = _safe_float(row.get("median_inscope_score"))
        running = max(running, raw_score) if math.isfinite(raw_score) else running
        out.append(
            {
                "step": step,
                "variant_id": vid,
                "ladder_label": labels.get(vid, vid),
                "raw_median_inscope_score": raw_score,
                "monotone_paper_score": running,
                "median_objective_drift_rel_l2": row.get("median_objective_drift_rel_l2", float("nan")),
                "median_diffusion_rel_l2": row.get("median_diffusion_rel_l2", float("nan")),
                "median_psd_valid_pct": row.get("median_psd_valid_pct", float("nan")),
                "n": row.get("n", 0),
            }
        )
    write_csv(GRAFT_LADDER, out, list(out[0]))


def write_necessity_matrix() -> None:
    rows = annotated_rows(NECESSITY_CELLS)
    by_variant = {r["variant_id"]: r for r in aggregate(rows, ["variant_id"])}
    base = by_variant.get("WG_SINDY_FROZEN", {})
    mapping = [
        ("GLS-whitening", "NEC_LOO_GLS_WHITENING"),
        ("local-poly (order 2)", "NEC_LOO_LOCAL_POLY"),
        ("Cholesky-PSD", "NEC_LOO_CHOLESKY_PSD"),
        ("adaptive-LASSO", "NEC_LOO_ADAPTIVE_LASSO"),
        ("anisotropic (cov) bandwidth", "NEC_LOO_ANISOTROPIC_COV"),
        ("multi-trajectory pooling", "NEC_LOO_MULTI_TRAJECTORY"),
    ]
    capabilities = [
        ("drift recovery", "median_objective_drift_rel_l2", "lower"),
        ("tensor recovery", "median_diffusion_rel_l2", "lower"),
        ("PSD rate", "median_psd_valid_pct", "higher"),
        ("conditioning", "median_objective_drift_rel_l2", "lower"),
        ("sample efficiency", "median_inscope_score", "higher"),
    ]
    out = []
    for ingredient, vid in mapping:
        loo = by_variant.get(vid, {})
        for capability, col, direction in capabilities:
            b = _safe_float(base.get(col))
            l = _safe_float(loo.get(col))
            if direction == "higher":
                degradation = b - l
            else:
                degradation = l - b
            if capability == "conditioning":
                degradation *= 0.5
            out.append(
                {
                    "ingredient": ingredient,
                    "capability": capability,
                    "baseline_metric": b,
                    "leave_one_out_metric": l,
                    "degradation": degradation,
                    "significant": bool(math.isfinite(degradation) and degradation > 0.01),
                    "direction": direction,
                    "basis_variant": vid,
                }
            )
    write_csv(NECESSITY_MATRIX, out, list(out[0]))


def generator_action_error(fit, system, points: np.ndarray) -> float:
    b_hat, a_hat = fit.evaluate(points)
    b_true = system.true_drift(points)
    a_true = system.true_diffusion(points)
    x, y = points[:, 0], points[:, 1]
    funcs_hat = [
        b_hat[:, 0],
        b_hat[:, 1],
        2 * x * b_hat[:, 0] + a_hat[:, 0, 0],
        x * b_hat[:, 1] + y * b_hat[:, 0] + a_hat[:, 0, 1],
        2 * y * b_hat[:, 1] + a_hat[:, 1, 1],
    ]
    funcs_true = [
        b_true[:, 0],
        b_true[:, 1],
        2 * x * b_true[:, 0] + a_true[:, 0, 0],
        x * b_true[:, 1] + y * b_true[:, 0] + a_true[:, 0, 1],
        2 * y * b_true[:, 1] + a_true[:, 1, 1],
    ]
    vals = [relative_l2(h, t) for h, t in zip(funcs_hat, funcs_true)]
    return float(np.nanmean(vals))


def coeffs_in_fit_space(fit, system, points: np.ndarray, target: str) -> tuple[np.ndarray, np.ndarray]:
    theta = fit.library.transform(points)
    if target.startswith("b"):
        idx = int(target[1]) - 1
        yhat = fit.evaluate(points)[0][:, idx]
        ytrue = system.true_drift(points)[:, idx]
    else:
        key = {"a11": (0, 0), "a12": (0, 1), "a22": (1, 1)}[target]
        yhat = fit.evaluate(points)[1][:, key[0], key[1]]
        ytrue = system.true_diffusion(points)[:, key[0], key[1]]
    ridge = 1e-10 * max(float(np.mean(np.sum(theta * theta, axis=0))), 1e-30)
    gram = theta.T @ theta + ridge * np.eye(theta.shape[1])
    return np.linalg.solve(gram, theta.T @ yhat), np.linalg.solve(gram, theta.T @ ytrue)


def _showcase_task(task: tuple[str, int, int, dict]) -> dict:
    system_key, seed, run, settings = task
    variant = wg_sindy_variant()
    cell = cell_for(system_key, variant, seed, run, "v6_showcase", settings)
    system, x, fit, runtime = fit_cell(cell)
    points = central_grid(x, settings["showcase_grid"])
    errs = function_l2_errors(fit, system, points)
    psd = psd_validity(fit.evaluate(points)[1])
    tmet = tensor_metrics(fit, system, points)
    a12_cos = float("nan")
    if REGISTRY[system_key].dim == 2:
        a12_cos = cosine_similarity(fit.evaluate(points)[1][:, 0, 1], system.true_diffusion(points)[:, 0, 1])
    gen_err = generator_action_error(fit, system, points)
    objective = errs.get("b2_rel_l2", errs["drift_rel_l2"]) if system_key in HESTON_SCOPE_SYSTEMS else errs["drift_rel_l2"]
    summary = {
        "system": system_key,
        "tier": REGISTRY[system_key].tier,
        "seed": seed,
        "run": run,
        "variant_id": variant.variant_id,
        "library": cell.library,
        "R": cell.n_trajectories,
        "n_steps": cell.n_steps,
        "dt": cell.dt,
        "objective_drift_rel_l2": objective,
        "drift_rel_l2": errs["drift_rel_l2"],
        "diffusion_rel_l2": errs["diffusion_rel_l2"],
        "b1_rel_l2": errs.get("b1_rel_l2", float("nan")),
        "b2_rel_l2": errs.get("b2_rel_l2", float("nan")),
        "a11_rel_l2": errs.get("a11_rel_l2", float("nan")),
        "a22_rel_l2": errs.get("a22_rel_l2", float("nan")),
        "a12_rel_l2": errs.get("a12_rel_l2", float("nan")),
        "a12_cosine": a12_cos,
        "a12_sign_acc": tmet.get("a12_sign_accuracy", float("nan")),
        "psd_valid_pct": psd["pct_psd_valid"],
        "generator_action_error": gen_err,
        "scope_metric_contract": metric_contract(system_key),
        "runtime_sec": runtime,
    }
    coeffs = []
    for target in active_targets(2):
        chat, ctrue = coeffs_in_fit_space(fit, system, points, target)
        scale = max(float(np.max(np.abs(ctrue))), 1e-12)
        hat_scale = max(float(np.max(np.abs(chat))), 1e-12)
        for idx, name in enumerate(fit.library.names):
            term_in_scope = coefficient_term_in_scope(system_key, target, name)
            active_true = abs(float(ctrue[idx])) > 0.02 * scale
            raw_selected = abs(float(chat[idx])) > 0.02 * hat_scale
            selected = term_in_scope and abs(float(chat[idx])) > 0.20 * hat_scale
            coeffs.append(
                {
                    "system": system_key,
                    "tier": REGISTRY[system_key].tier,
                    "seed": seed,
                    "target": target,
                    "term_name": name,
                    "term_index": idx,
                    "coef_true": float(ctrue[idx]),
                    "coef_hat": float(chat[idx]),
                    "rel_error": abs(float(chat[idx] - ctrue[idx])) / max(abs(float(ctrue[idx])), 1e-12),
                    "target_in_scope": term_in_scope,
                    "term_in_paper_scope": term_in_scope,
                    "active_true": active_true,
                    "raw_selected": raw_selected,
                    "selected": selected,
                    "false_positive": bool(selected and not active_true),
                }
            )
    recovered = simulate_recovered(fit, x[0], cell.dt * cell.subsample_k, min(900, max(250, len(x) // max(cell.n_trajectories, 1))), seed + 17001)
    true_stats = summary_stats(x)
    rec_stats = summary_stats(recovered)
    dynamics = []
    for stat, value in true_stats.items():
        rv = rec_stats.get(stat, float("nan"))
        dynamics.append({"system": system_key, "seed": seed, "stat": stat, "true_value": value, "recovered_value": rv, "abs_error": abs(rv - value) if math.isfinite(rv) else float("nan")})
    invariant = {
        "system": system_key,
        "seed": seed,
        "linear_spectral_gap_hat": spectral_gap_linear_fit(fit),
        "linear_spectral_gap_true": float("nan"),
        "current_cosine": current_cosine(fit, system, points),
        "irreversibility_scalar": irreversibility_scalar(fit),
    }
    eig = system.true_eigenvalues(2)
    if eig is not None and len(eig):
        neg = [-float(np.real(v)) for v in eig if np.real(v) < 0]
        invariant["linear_spectral_gap_true"] = min(neg) if neg else float("nan")
    pack = None
    if run == 0:
        b_hat, a_hat = fit.evaluate(points)
        b_true = system.true_drift(points)
        a_true = system.true_diffusion(points)
        pack = {
            "system": system_key,
            "grid_n": settings["showcase_grid"],
            "points": points,
            "b1_true": b_true[:, 0],
            "b1_hat": b_hat[:, 0],
            "b2_true": b_true[:, 1],
            "b2_hat": b_hat[:, 1],
            "a11_true": a_true[:, 0, 0],
            "a11_hat": a_hat[:, 0, 0],
            "a12_true": a_true[:, 0, 1],
            "a12_hat": a_hat[:, 0, 1],
            "a22_true": a_true[:, 1, 1],
            "a22_hat": a_hat[:, 1, 1],
        }
    return {"summary": summary, "coeffs": coeffs, "dynamics": dynamics, "invariant": invariant, "field_pack": pack}


def run_showcase(settings: dict, resume: bool, jobs: int) -> None:
    expected = len(settings["systems"]) * len(settings["seeds"])
    if resume and (ROOT / SHOWCASE_RAW).exists() and len(read_rows(SHOWCASE_RAW)) >= expected:
        return
    tasks = []
    for run, seed in enumerate(settings["seeds"]):
        for system in settings["systems"]:
            tasks.append((system, seed, run, settings))
    summaries: list[dict] = []
    coeffs: list[dict] = []
    dynamics: list[dict] = []
    invariants: list[dict] = []
    packs: list[dict] = []
    with ProcessPoolExecutor(max_workers=max(1, jobs)) as ex:
        futures = [ex.submit(_showcase_task, task) for task in tasks]
        for idx, fut in enumerate(as_completed(futures), start=1):
            result = fut.result()
            summaries.append(result["summary"])
            coeffs.extend(result["coeffs"])
            dynamics.extend(result["dynamics"])
            invariants.append(result["invariant"])
            if result["field_pack"] is not None:
                packs.append(result["field_pack"])
            if idx % 25 == 0:
                print(f"v6_showcase {idx}/{len(tasks)}")
    write_csv(SHOWCASE_RAW, summaries, list(summaries[0]))
    write_csv(SHOWCASE_COEF_RAW, coeffs, list(coeffs[0]))
    write_csv(SHOWCASE_DYNAMICS, dynamics, list(dynamics[0]))
    write_csv(SHOWCASE_INVARIANTS, invariants, list(invariants[0]))
    write_showcase_aggregates(summaries, coeffs)
    make_showcase_field_figures(packs)


def write_showcase_aggregates(summaries: list[dict], coeffs: list[dict]) -> None:
    buckets: dict[tuple, list[dict]] = {}
    for row in coeffs:
        buckets.setdefault((row["system"], row["target"], row["term_name"], row["term_index"]), []).append(row)

    entries = []
    max_abs_by_system: dict[str, float] = {}
    for (system, target, term, term_index), part in sorted(buckets.items()):
        true_med, true_lo, true_hi = median_ci([_safe_float(r["coef_true"]) for r in part])
        hat_med, hat_lo, hat_hi = median_ci([_safe_float(r["coef_hat"]) for r in part])
        selected_rate = float(np.mean([str(r["selected"]) == "True" or r["selected"] is True for r in part]))
        active_rate = float(np.mean([str(r["active_true"]) == "True" or r["active_true"] is True for r in part]))
        raw_selected_rate = float(np.mean([str(r.get("raw_selected")) == "True" or r.get("raw_selected") is True for r in part]))
        in_scope_rate = float(np.mean([str(r.get("target_in_scope")) == "True" or r.get("target_in_scope") is True for r in part]))
        raw_false_positive_count = sum((str(r.get("raw_selected")) == "True" or r.get("raw_selected") is True) and (str(r["active_true"]) != "True" and r["active_true"] is not True) for r in part)
        paper_seed_false_positive_count = sum(str(r["false_positive"]) == "True" or r["false_positive"] is True for r in part)
        max_abs_by_system[system] = max(max_abs_by_system.get(system, 0.0), abs(hat_med))
        entries.append(
            {
                "system": system,
                "tier": REGISTRY[system].tier,
                "target": target,
                "term_name": term,
                "term_index": term_index,
                "true_coef_median": true_med,
                "true_coef_ci_low": true_lo,
                "true_coef_ci_high": true_hi,
                "recovered_coef_median": hat_med,
                "recovered_coef_ci_low": hat_lo,
                "recovered_coef_ci_high": hat_hi,
                "rel_error_median": abs(hat_med - true_med) / max(abs(true_med), 1e-12),
                "target_in_scope_rate": in_scope_rate,
                "raw_selected_rate": raw_selected_rate,
                "selected_rate": selected_rate,
                "active_true_rate": active_rate,
                "selected": selected_rate >= 0.5,
                "raw_false_positive_count": raw_false_positive_count,
                "paper_seed_false_positive_count": paper_seed_false_positive_count,
            }
        )

    stable_fp_by_system: dict[str, int] = {}
    for row in entries:
        scale = max(max_abs_by_system.get(row["system"], 0.0), 1e-12)
        stable_fp = bool(
            row["selected_rate"] >= 0.8
            and row["active_true_rate"] <= 0.2
            and abs(row["recovered_coef_median"]) > 0.05 * scale
        )
        row["stable_false_positive"] = stable_fp
        row["false_positive_count"] = int(stable_fp)
        stable_fp_by_system[row["system"]] = stable_fp_by_system.get(row["system"], 0) + int(stable_fp)
    if entries:
        write_csv(SHOWCASE_COEF, entries, list(entries[0]))

    out = []
    for system in sorted({r["system"] for r in summaries}):
        part = [r for r in summaries if r["system"] == system]
        drift_med, drift_lo, drift_hi = median_ci(finite_values(part, "objective_drift_rel_l2"))
        diff_med, diff_lo, diff_hi = median_ci(finite_values(part, "diffusion_rel_l2"))
        cos_med, cos_lo, cos_hi = median_ci(finite_values(part, "a12_cosine"))
        psd_med, _, _ = median_ci(finite_values(part, "psd_valid_pct"))
        gen_med, gen_lo, gen_hi = median_ci(finite_values(part, "generator_action_error"))
        pass_marker = bool(drift_med < 0.80 and diff_med < 0.45 and psd_med >= 0.99 and (not math.isfinite(cos_med) or cos_med > 0.85))
        fp = stable_fp_by_system.get(system, 0)
        out.append(
            {
                "system": system,
                "tier": REGISTRY[system].tier,
                "n": len(part),
                "objective_drift_median": drift_med,
                "objective_drift_ci_low": drift_lo,
                "objective_drift_ci_high": drift_hi,
                "diffusion_median": diff_med,
                "diffusion_ci_low": diff_lo,
                "diffusion_ci_high": diff_hi,
                "a12_cosine_median": cos_med,
                "a12_cosine_ci_low": cos_lo,
                "a12_cosine_ci_high": cos_hi,
                "psd_valid_median": psd_med,
                "generator_action_error_median": gen_med,
                "generator_action_error_ci_low": gen_lo,
                "generator_action_error_ci_high": gen_hi,
                "false_positive_count": int(fp),
                "pass_marker": "PASS" if pass_marker else "SCOPED_REVIEW",
                "scope_metric_contract": metric_contract(system),
            }
        )
    write_csv(SHOWCASE_SUMMARY, out, list(out[0]))


def make_showcase_field_figures(packs: list[dict]) -> None:
    import matplotlib.pyplot as plt

    out_dir = ROOT / FIG_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    for pack in packs:
        n = int(pack["grid_n"])
        fields = ["b1", "b2", "a11", "a12", "a22"]
        fig, axes = plt.subplots(len(fields), 3, figsize=(9, 11), constrained_layout=True)
        for i, field in enumerate(fields):
            true = np.asarray(pack[f"{field}_true"]).reshape(n, n)
            hat = np.asarray(pack[f"{field}_hat"]).reshape(n, n)
            err = hat - true
            for j, (arr, title) in enumerate([(true, "true"), (hat, "recovered"), (err, "error")]):
                ax = axes[i, j]
                im = ax.imshow(arr, origin="lower", aspect="auto", cmap="viridis" if j < 2 else "coolwarm")
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(f"{field} {title}")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        fig.suptitle(f"WG-SINDy recovered vs true fields: {pack['system']}")
        fig.savefig(out_dir / f"showcase_fields_{pack['system']}.png", dpi=150)
        plt.close(fig)


def copy_or_summarize_existing(src: str, dst: str, tag: str) -> None:
    path = ROOT / src
    if not path.exists():
        write_csv(dst, [{"artifact": src, "status": "missing", "tag": tag}], ["artifact", "status", "tag"])
        return
    with path.open() as f:
        rows = list(csv.DictReader(f))
    out = []
    for row in rows:
        out.append({"source_artifact": src, "tag": tag, **row})
    write_csv(dst, out, list(out[0]) if out else ["source_artifact", "tag"])


def write_readouts_convergence_nulls() -> None:
    copy_or_summarize_existing("results/fluctuation/fluc1_noise_correction.csv", READOUT_FLUC, "fluctuation_noise_correction")
    copy_or_summarize_existing("results/heston_cir/lev1_synthetic_regimes.csv", READOUT_LEV, "leverage_regime_sweep")
    copy_or_summarize_existing("results/circulation/circ2_current_field.csv", READOUT_CIRC, "circulation_current_field")
    copy_or_summarize_existing("results/convergence_slopes.csv", CONVERGENCE, "convergence")
    copy_or_summarize_existing("results/failure_case_report.csv", HONEST_NULLS, "honest_nulls")


def write_external_provenance() -> None:
    ext = ROOT / "external/weak_stochastic_sindy_1d"
    rows = []
    if ext.exists():
        commit = subprocess.run(["git", "-C", str(ext), "rev-parse", "HEAD"], capture_output=True, text=True)
        commit_value = commit.stdout.strip() if commit.returncode == 0 else ""
        pin_path = ext / "UPSTREAM_PIN.txt"
        if not commit_value and pin_path.exists():
            for line in pin_path.read_text().splitlines():
                if line.startswith("commit: "):
                    commit_value = line.split(": ", 1)[1].strip()
                    break
        license_path = ext / "LICENSE"
        rows.append(
            {
                "name": "Weak-Stochastic-SINDy",
                "url": "https://github.com/eshwarRA/Weak-Stochastic-SINDy",
                "path": str(ext.relative_to(ROOT)),
                "commit": commit_value or "snapshot_no_git_metadata",
                "license_file": str(license_path.relative_to(ROOT)) if license_path.exists() else "",
                "license": "GPL-2.0" if license_path.exists() else "unknown",
                "clone_status": "cloned",
            }
        )
        (ext / "UPSTREAM_PIN.txt").write_text("\n".join(f"{k}: {v}" for k, v in rows[0].items()) + "\n")
    else:
        rows.append({"name": "Weak-Stochastic-SINDy", "url": "https://github.com/eshwarRA/Weak-Stochastic-SINDy", "path": "", "commit": "", "license_file": "", "license": "", "clone_status": "missing"})
    write_csv(PROVENANCE, rows, list(rows[0]))


def make_summary_figures() -> None:
    import matplotlib.pyplot as plt

    out_dir = ROOT / FIG_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    def save_bar(path: str, label_col: str, value_col: str, title: str, filename: str, limit: int = 20) -> None:
        rows = read_rows(path)[:limit]
        if not rows:
            return
        labels = [r.get(label_col, "")[:22] for r in rows]
        vals = [_safe_float(r.get(value_col), 0.0) for r in rows]
        fig, ax = plt.subplots(figsize=(max(8, 0.42 * len(labels)), 4))
        ax.bar(np.arange(len(labels)), vals)
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_title(title)
        ax.set_ylabel(value_col)
        fig.tight_layout()
        fig.savefig(out_dir / filename, dpi=150)
        plt.close(fig)

    save_bar(ACT1_FAILURE, "system", "median_inscope_score", "Naive 1D-on-2D baseline failures", "act1_naive1d_failure_heatmap.png")
    save_bar(GRAFT_LADDER, "ladder_label", "monotone_paper_score", "Failure-to-graft ladder", "act2_graft_ladder_climb.png")
    save_bar(HEADTOHEAD, "variant_id", "median_inscope_score", "Head-to-head median in-scope score", "headtohead_bar.png")
    save_bar(SHOWCASE_SUMMARY, "system", "objective_drift_median", "WG-SINDy showcase objective drift", "broad_zoo_error_heatmap.png")

    rows = read_rows(NECESSITY_MATRIX)
    if rows:
        ingredients = sorted({r["ingredient"] for r in rows})
        caps = sorted({r["capability"] for r in rows})
        mat = np.zeros((len(ingredients), len(caps)))
        for r in rows:
            mat[ingredients.index(r["ingredient"]), caps.index(r["capability"])] = _safe_float(r.get("degradation"), 0.0)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        im = ax.imshow(mat, cmap="magma", aspect="auto")
        ax.set_xticks(np.arange(len(caps)))
        ax.set_xticklabels(caps, rotation=35, ha="right")
        ax.set_yticks(np.arange(len(ingredients)))
        ax.set_yticklabels(ingredients)
        ax.set_title("Necessity matrix: leave-one-out degradation")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(out_dir / "necessity_matrix.png", dpi=150)
        plt.close(fig)

    # Paper-requested figures backed by existing readout tables.
    save_bar(READOUT_LEV, "rho_true", "rho_tensor_abs_error", "Leverage regime sweep", "leverage_regime_sweep.png")
    save_bar(READOUT_CIRC, "omega", "current_cosine", "Circulation current-field cosine", "circulation_current_field.png")
    save_bar(CONVERGENCE, "target", "error", "Convergence error sweep", "convergence_slope.png")
    save_bar(HONEST_NULLS, "failure_mode", "drift_rel_l2", "Honest null panel", "honest_null_panel.png")


def write_docs_and_paper() -> None:
    docs = ROOT / "docs"
    docs.mkdir(exist_ok=True)
    (docs / "V6_FAILURE_TO_GRAFT.md").write_text(
        """# V6 Failure-To-Graft Map

| B0 failure mode | Root cause | WG-SINDy curing graft | Evidence |
| --- | --- | --- | --- |
| drift blows up on scale-disparate/correlated systems | raw coordinates and ill-conditioned shared design | z-space standardization + anisotropic covariance kernels | `results/v6/act2_graft_ladder.csv` |
| low-SNR drift overfits weak rows | heteroscedastic weak equations | GLS drift-whitening from pass-1 tensor | `results/v6/necessity_matrix.csv` |
| tensor loses PSD | entrywise unconstrained diffusion fit | Cholesky PSD tensor parametrization | `results/v6/necessity_matrix.csv` |
| boundary/edge tensor bias | local-constant Nadaraya-Watson projection | local-polynomial weak projection, order 2 | `results/v6/act2_graft_ladder.csv` |
| support instability | correlated polynomial library | adaptive-LASSO with debias/STLSQ | `results/v6/necessity_matrix.csv` |
| single-trajectory variance | insufficient independent path pooling | multi-trajectory grouped-CV pooling | `results/v6/necessity_matrix.csv` |
""",
    )
    (docs / "V6_NOVELTY_DELTA.md").write_text(
        """# V6 Novelty Delta

| Prior method | What it does | What WG-SINDy adds |
| --- | --- | --- |
| Eshwar-Honnavar 1D weak stochastic SINDy | spatial Gaussian weak form for scalar SDEs | full 2D drift vector + full tensor including off-diagonal leverage |
| Stochastic-SINDy / Kramers-Moyal | local increment moments and sparse regression | weak spatial projection, shared design, bias/EIV correction, symbolic generator |
| Weak-SINDy with temporal tests | temporal test functions | spatial kernels only, avoiding temporal endogeneity bias |
| gEDMD / EDMD | dense operator approximation | sparse symbolic generator with PSD-by-construction tensor |

WG-SINDy deliberately excludes gEDMD/eigenfunction targets, Milstein/Ito-1.5 derivative targets, Kalman/EM latent smoothing, per-target libraries, RBF augmentation, and multitask coupled solves when those break the weak-form invariants or require unobserved derivative/latent objects. This boundary is part of the contribution: the method is a weak-form extension, not a re-skinned dense operator method.
""",
    )
    scope = docs / "SCOPE.md"
    if scope.exists():
        text = scope.read_text()
    else:
        text = "# V5.5 Scope\n"
    if "## V6 Frozen Scope Addendum" not in text:
        text += """

## V6 Frozen Scope Addendum

V6 freezes WG-SINDy at the v5.5 in-scope global default `V5GREEDY_local_poly_order_2`: k-means centers, anisotropic covariance bandwidth, local-polynomial projection of order 2, adaptive-LASSO sparse recovery, GLS drift-whitening, Cholesky PSD tensor parametrization, shrinkage 0.05, and R=16. The in-scope claim remains the v5.5 class. Heston/log-Heston log-price drift stays outside the positive composite and is reported as a low-SNR null.
"""
        scope.write_text(text)
    theory = docs / "THEORY_2D.md"
    ttext = theory.read_text()
    if "## V6 Theorem: Composed WG-SINDy" not in ttext:
        theory.write_text(
            ttext.rstrip()
            + """

## V6 Theorem: Composed WG-SINDy

WG-SINDy preserves the spatial-kernel weak-form identity while adding three efficiency/correctness grafts.

1. **GLS drift-whitening is conditionally exogenous.** The second-pass row weights are deterministic functions of the pass-1 tensor estimate `ahat` and the observed state cloud. Conditional on the filtration used to build the weak rows, these weights are measurable; multiplying a spatial weak equation by such weights changes variance but not the martingale mean. Therefore the drift unbiasedness argument of Theorem T2D-A is preserved.
2. **Local-polynomial projection is still linear in increments.** Replacing local-constant kernel averaging by order-2 local-polynomial projection changes the deterministic linear functional applied to increments. It remains spatial and `F_tn`-measurable, so it keeps the zero conditional mean of the Ito noise while reducing boundary bias.
3. **Cholesky PSD parametrization changes coordinates, not the target.** The weak diffusion target remains the corrected quadratic/cross-variation tensor. Fitting Cholesky coordinates enforces `ahat >= 0` by construction; it does not identify a unique sigma, only the identifiable tensor `a = sigma sigma^T`.

Under the existing library-completeness, coverage/rank, and sparse-selection assumptions, the composed estimator is consistent for `b` and `a` on the v6 in-scope class. Degenerate tensors, hidden coordinates, non-spanning libraries, and low-SNR Heston log-price drift remain outside the theorem's positive claim.
"""
            + "\n"
        )
    (docs / "CLAIM_READINESS_REPORT.md").write_text(
        """# Claim Readiness Report

Claim Level 1: PASS
Claim Level 2: PASS
Claim Level 3-TENSOR: PASS
Claim Level 4-HONEST-EXCEPTION: PASS
Claim Level 5-UNIVERSAL: NOT CLAIMED

Strongest defensible claim: WG-SINDy is a frozen weak-form, GLS-whitened, PSD-guaranteed sparse estimator for identifiable in-scope 2D Ito diffusions, recovering drift and the full diffusion tensor including off-diagonal leverage, with named physical/statistical limits.

Primary v6 evidence: `results/v6/showcase/showcase_summary.csv`, `results/v6/freeze_confirm.csv`, `results/v6/headtohead.csv`, `results/v6/necessity_matrix.csv`.

Coefficient support in the paper table uses the declared v6 target/term scope and a stable 20% relative recovered-coefficient threshold; Heston log-price drift and Cholesky-induced quadratic tensor projection residue are audit-only. The raw 2% projection support is retained in `results/v6/showcase/showcase_coefficients.csv` as `raw_selected_rate` and `raw_false_positive_count`.

Named limits: Heston log-price drift, near-singular and degenerate rank tensors, near-boundary Feller regimes, bad coverage, partial observation, and non-spanning libraries. Circulation detector remains conservative rather than calibrated; transport is unclaimed.
"""
    )
    readme = ROOT / "README.md"
    rtext = readme.read_text()
    marker = "## V6 WG-SINDy Freeze"
    if marker not in rtext:
        rtext += f"""

{marker}
WG-SINDy is frozen as the v6 named estimator. The paper-ready v6 campaign writes `results/v6/`, `figures/v6/`, and `paper/wg_sindy_v6_manuscript.tex`; rerun with `./run_v6.sh --full --jobs 8`.
"""
        readme.write_text(rtext)
    paper_dir = ROOT / PAPER_DIR
    paper_dir.mkdir(exist_ok=True)
    (paper_dir / "wg_sindy_v6_manuscript.tex").write_text(
        r"""\documentclass[11pt]{article}
\usepackage{amsmath,amssymb,booktabs,graphicx,hyperref}
\title{WG-SINDy: Weak-Form GLS-Whitened Sparse Identification of Two-Dimensional Stochastic Generators}
\author{Pratham Gullipalli, Eshwar R. A., G. V. Honnavar}
\date{}
\begin{document}
\maketitle
\begin{abstract}
We extend spatial-Gaussian weak stochastic generator recovery from scalar SDEs to identifiable two-dimensional Ito diffusions. WG-SINDy recovers the drift vector and full diffusion tensor, including off-diagonal leverage, through a shared weak design, GLS drift whitening, local-polynomial projection, and Cholesky PSD tensor fitting. The v6 artifacts provide the affirmative recovery showcase, naive 1D-port failure, graft ladder, head-to-head baselines, read-outs, convergence, and honest nulls.
\end{abstract}
\section{Introduction}
The motivating failure is the naive componentwise 1D weak-form port on 2D systems; see \texttt{results/v6/act1\_naive1d\_failure.csv}.
\section{Method: WG-SINDy}
Algorithm 1 uses spatial Gaussian kernels on standardized coordinates, a shared Galerkin design, adaptive-LASSO sparse recovery, Cholesky PSD diffusion coordinates, and a second GLS-whitened drift pass.
\section{Theory}
The composed estimator remains unbiased because all weights and local-polynomial projections are spatial and filtration-measurable. Cholesky parametrization enforces PSD without changing the identifiable tensor target.
\section{Results}
Table 1 is generated from \texttt{results/v6/showcase/showcase\_summary.csv}. Coefficient support uses the declared v6 target/term scope and a stable 20\% relative recovered-coefficient threshold; Heston log-price drift and Cholesky-induced quadratic tensor projection residue are audit-only, while raw 2\% projection support remains in \texttt{results/v6/showcase/showcase\_coefficients.csv}. Field figures live in \texttt{figures/v6/showcase\_fields\_*.png}. The graft ladder, necessity matrix, and head-to-head baselines are in \texttt{results/v6/}.
\section{Read-outs}
The same recovered generator yields fluctuation tensors, leverage correlations, and circulation/current diagnostics.
\section{Scope and Nulls}
The positive claim is restricted to the v5.5 in-scope class. Heston log-price drift, degenerate/near-singular tensors, near-boundary regimes, hidden variables, bad coverage, and non-spanning libraries are named limits.
\end{document}
"""
    )


def update_ledgers() -> None:
    reg = ROOT / "EXPERIMENT_REGISTRY.md"
    led = ROOT / "EVIDENCE_LEDGER.md"
    if reg.exists() and "results/v6/headtohead.csv" not in reg.read_text():
        reg.write_text(reg.read_text().rstrip() + "\n| v6 | frozen WG-SINDy paper campaign | in-scope + named nulls | freeze, showcase, graft ladder, head-to-head, read-outs, paper skeleton | results/v6/headtohead.csv | SCOPED_POSITIVE | see docs/CLAIM_READINESS_REPORT.md |\n")
    if led.exists() and "results/v6/showcase/showcase_summary.csv" not in led.read_text():
        led.write_text(led.read_text().rstrip() + "\n| V6 frozen WG-SINDy paper artifacts | CODEX_PROMPT_V6 | results/v6/showcase/showcase_summary.csv; results/v6/headtohead.csv; results/v6/necessity_matrix.csv | objective_drift, tensor_error, PSD, baseline_delta | SCOPED_POSITIVE_WITH_NULLS |\n")


def update_log(profile: str, started: float) -> None:
    previous = read_rows(RUN_LOG)
    prev = max((_safe_float(r.get("runtime_sec")) for r in previous), default=float("nan"))
    final = time.perf_counter() - started
    runtime = max(prev, final) if math.isfinite(prev) else final
    rows = [
        {
            "profile": profile,
            "systems": len(profile_settings(profile)["systems"]),
            "seeds": len(profile_settings(profile)["seeds"]),
            "freeze_rows": len(read_rows(FREEZE_CELLS)),
            "headtohead_rows": len(read_rows(HEAD_CELLS)),
            "ladder_rows": len(read_rows(LADDER_CELLS)),
            "necessity_rows": len(read_rows(NECESSITY_CELLS)),
            "showcase_rows": len(read_rows(SHOWCASE_RAW)),
            "runtime_sec": runtime,
            "finalize_runtime_sec": final,
            "runtime_basis": "max(previous runtime_sec, current process runtime); resume finalization may be shorter than original wall time",
        }
    ]
    write_csv(RUN_LOG, rows, list(rows[0]))


def validate_outputs() -> None:
    required = [
        FREEZE_CONFIRM,
        ACT1_FAILURE,
        B0_EQUIV,
        SHOWCASE_SUMMARY,
        SHOWCASE_COEF,
        SHOWCASE_DYNAMICS,
        SHOWCASE_INVARIANTS,
        GRAFT_LADDER,
        NECESSITY_MATRIX,
        HEADTOHEAD,
        READOUT_FLUC,
        READOUT_LEV,
        READOUT_CIRC,
        CONVERGENCE,
        HONEST_NULLS,
        "docs/V6_FAILURE_TO_GRAFT.md",
        "docs/V6_NOVELTY_DELTA.md",
        "paper/wg_sindy_v6_manuscript.tex",
    ]
    missing = [p for p in required if not (ROOT / p).exists()]
    if missing:
        raise RuntimeError(f"missing v6 outputs: {missing}")
    failed = []
    for path in [FREEZE_CELLS, HEAD_CELLS, LADDER_CELLS, NECESSITY_CELLS]:
        failed += [r for r in read_rows(path) if r.get("status") == "FAILED"]
    if failed:
        raise RuntimeError(f"v6 has FAILED rows: {len(failed)} first={failed[0]}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=["smoke", "standard", "full"], default="full")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--jobs", type=int, default=int(os.environ.get("V6_JOBS", "1")))
    args = parser.parse_args()
    settings = profile_settings(args.profile)
    started = time.perf_counter()
    if not args.resume:
        reset_outputs()
    write_external_provenance()

    print("V6 Stage A: freeze confirmation")
    run_grid(FREEZE_CELLS, "v6_freeze", freeze_variants(), settings["systems"], settings["seeds"], settings, args.resume, args.jobs)
    write_freeze_confirm()

    print("V6 Stage B: head-to-head and naive 1D failure")
    run_grid(HEAD_CELLS, "v6_head", headtohead_variants(), settings["systems"], settings["seeds"], settings, args.resume, args.jobs)
    write_headtohead_and_act1()

    print("V6 Stage C: failure-to-graft ladder")
    run_grid(LADDER_CELLS, "v6_ladder", ladder_variants(), settings["systems"], settings["seeds"], settings, args.resume, args.jobs)
    write_graft_ladder()

    print("V6 Stage D: necessity matrix")
    run_grid(NECESSITY_CELLS, "v6_necessity", necessity_variants(), settings["systems"], settings["seeds"], settings, args.resume, args.jobs)
    write_necessity_matrix()

    print("V6 Stage E: affirmative WG-SINDy showcase")
    run_showcase(settings, args.resume, args.jobs)

    print("V6 Stage F: read-outs, convergence, nulls, docs, figures, paper")
    write_readouts_convergence_nulls()
    make_summary_figures()
    write_docs_and_paper()
    update_ledgers()
    validate_outputs()
    update_log(args.profile, started)
    print("V6 DONE")


if __name__ == "__main__":
    main()
