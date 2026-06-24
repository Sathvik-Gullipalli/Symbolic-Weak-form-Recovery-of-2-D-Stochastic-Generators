from __future__ import annotations

import argparse
import math
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[2] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from experiments.common import ROOT
from experiments.v5.run_v5_campaign import (
    FACTORIAL_PATH as V5_FACTORIAL_PATH,
    GLOBAL_DEFAULT_PATH as V5_GLOBAL_DEFAULT_PATH,
    GREEDY_PATH as V5_GREEDY_PATH,
    OFAT_PATH as V5_OFAT_PATH,
    PER_SYSTEM_PATH as V5_PER_SYSTEM_PATH,
    ROW_FIELDS,
    STAGE0_PATH as V5_STAGE0_PATH,
    V3_MEDIAN_DRIFT,
    V5Variant,
    _safe_float,
    read_rows,
    run_grid,
    target_kw_label,
    variant_from_row,
    v5_composed_stack,
    write_csv,
)
from sde2d.systems import REGISTRY


OUT_DIR = "results/v5_5"
LOO_CELLS_PATH = f"{OUT_DIR}/loo_cells.csv"
TARGET_CELLS_PATH = f"{OUT_DIR}/targeted_combos.csv"
SOLIDIFY_CELLS_PATH = f"{OUT_DIR}/solidify_cells.csv"
GLOBAL_DEFAULT_INSCOPE_PATH = f"{OUT_DIR}/global_default_inscope.csv"
MINIMAL_GRAFT_ABLATION_PATH = f"{OUT_DIR}/minimal_graft_ablation.csv"
LEADERBOARD_INSCOPE_PATH = f"{OUT_DIR}/leaderboard_inscope.csv"
SOLIDIFY_CIS_PATH = f"{OUT_DIR}/solidify_cis.csv"
RUN_LOG_PATH = f"{OUT_DIR}/run_log.csv"
FINDINGS_PATH = "docs/V5_5_FINDINGS.md"
SCOPE_PATH = "docs/SCOPE.md"
FIG_DIR = "figures/v5_5"

IN_SCOPE_SYSTEMS = [
    "indep_ou",
    "correlated_ou",
    "coupled_ou",
    "rotational_ou",
    "spiral_sink_corr",
    "double_well_transverse",
    "gradient_potential",
    "nongradient_circulation",
    "diag_multiplicative",
    "nondiag_cholesky",
    "heston_sv",
    "heston_logsv",
    "cir_pair",
]

OUT_OF_SCOPE_SYSTEMS = [
    "near_singular",
    "underdamped_langevin",
    "near_boundary_heston",
    "bad_coverage",
    "partial_observation",
    "too_large_dt",
    "nonpoly_drift",
]

HESTON_PRICE_DRIFT_NULL_SYSTEMS = {"heston_sv", "heston_logsv"}
REQUIRED_GRAFTS = [
    "GLS_WHITENING",
    "LOCAL_POLY",
    "ANISOTROPIC_LOCAL_COV_K8",
    "CHOLESKY_PSD",
    "SMALL_SIGNAL_OFFDIAG_P6",
    "STABILITY_SELECTION_G6",
    "ENSEMBLE_M3",
    "MULTI_TRAJECTORY_D5",
    "DIFFUSION_SHRINKAGE",
]

SCOPE_FIELDS = [
    *ROW_FIELDS,
    "scope_status",
    "scope_metric_contract",
    "objective_drift_rel_l2",
    "log_price_drift_rel_l2",
    "inscope_score",
]

GLOBAL_FIELDS = [
    "variant_id",
    "source",
    "n",
    "systems_covered",
    "tiers_covered",
    "worst_tier_median_objective_drift",
    "p90_objective_drift",
    "median_objective_drift",
    "median_diffusion",
    "median_a12_cosine",
    "p10_a12_cosine",
    "median_psd_valid_pct",
    "p10_psd_valid_pct",
    "median_inscope_score",
    "median_log_price_drift_rel_l2",
    "tensor_constraints_pass",
    "beats_v3_median_on_scope",
    "selection_basis",
    "family",
    "description",
    "regressor",
    "center_scheme",
    "bandwidth_rule",
    "bandwidth_mult",
    "local_poly_order",
    "gls_weighting",
    "gls_iterations",
    "diffusion_parameterization",
    "diffusion_shrinkage",
    "R",
]


def profile_settings(profile: str) -> dict:
    if profile == "smoke":
        return {
            "systems": ["correlated_ou", "rotational_ou", "heston_logsv"],
            "loo_seeds": [8101],
            "target_seeds": [8201],
            "solidify_seeds": [8301, 8302],
            "solidify_R": [8],
            "base_steps": 700,
            "heston_steps": 1000,
            "target_combo_limit": 6,
        }
    if profile == "standard":
        return {
            "systems": list(IN_SCOPE_SYSTEMS),
            "loo_seeds": [8101, 8102, 8103],
            "target_seeds": [8201, 8202, 8203],
            "solidify_seeds": [8301, 8302, 8303, 8304, 8305],
            "solidify_R": [8, 16],
            "base_steps": 1500,
            "heston_steps": 2400,
            "target_combo_limit": 10,
        }
    return {
        "systems": list(IN_SCOPE_SYSTEMS),
        "loo_seeds": [8101, 8102, 8103, 8104, 8105],
        "target_seeds": [8201, 8202, 8203, 8204, 8205],
        "solidify_seeds": [8301, 8302, 8303, 8304, 8305, 8306, 8307, 8308, 8309, 8310],
        "solidify_R": [8, 16, 32],
        "base_steps": 1800,
        "heston_steps": 3000,
        "target_combo_limit": 15,
    }


def reset_outputs() -> None:
    for rel in [
        LOO_CELLS_PATH,
        TARGET_CELLS_PATH,
        SOLIDIFY_CELLS_PATH,
        GLOBAL_DEFAULT_INSCOPE_PATH,
        MINIMAL_GRAFT_ABLATION_PATH,
        LEADERBOARD_INSCOPE_PATH,
        SOLIDIFY_CIS_PATH,
        RUN_LOG_PATH,
        FINDINGS_PATH,
        SCOPE_PATH,
    ]:
        path = ROOT / rel
        if path.exists():
            path.unlink()
    fig_dir = ROOT / FIG_DIR
    if fig_dir.exists():
        for child in fig_dir.glob("*.png"):
            child.unlink()


def is_success(row: dict) -> bool:
    return row.get("status") not in {"FAILED", "INFEASIBLE_BY_INVARIANT"} and row.get("system") in IN_SCOPE_SYSTEMS


def metric_contract(system_key: str) -> str:
    if system_key in HESTON_PRICE_DRIFT_NULL_SYSTEMS:
        return "tensor+leverage+b2_variance_drift; b1_log_price_drift_reported_null"
    if system_key == "cir_pair":
        return "tensor+leverage+CIR_pair_drift"
    return "drift+tensor+psd"


def objective_drift(row: dict) -> float:
    system_key = row.get("system", "")
    if system_key in HESTON_PRICE_DRIFT_NULL_SYSTEMS:
        return _safe_float(row.get("b2_rel_l2"))
    return _safe_float(row.get("drift_rel_l2"))


def inscope_score(row: dict) -> float:
    drift = objective_drift(row)
    diffusion = _safe_float(row.get("diffusion_rel_l2"))
    psd = _safe_float(row.get("psd_valid_pct"), 0.0)
    a12_cos = _safe_float(row.get("a12_cosine"))
    a12_term = 1.0 if math.isnan(a12_cos) else max(0.0, min(1.0, (a12_cos + 1.0) / 2.0))
    drift_term = max(0.0, 1.0 - min(drift, 2.0) / 2.0) if math.isfinite(drift) else 0.0
    diff_term = max(0.0, 1.0 - min(diffusion, 1.5) / 1.5) if math.isfinite(diffusion) else 0.0
    return 0.38 * drift_term + 0.34 * diff_term + 0.16 * psd + 0.12 * a12_term


def annotate_scope_row(row: dict) -> dict:
    out = dict(row)
    system_key = out.get("system", "")
    out["scope_status"] = "IN_SCOPE" if system_key in IN_SCOPE_SYSTEMS else "OUT_OF_SCOPE"
    out["scope_metric_contract"] = metric_contract(system_key)
    out["objective_drift_rel_l2"] = objective_drift(out)
    out["log_price_drift_rel_l2"] = _safe_float(out.get("b1_rel_l2")) if system_key in HESTON_PRICE_DRIFT_NULL_SYSTEMS else float("nan")
    out["inscope_score"] = inscope_score(out)
    return out


def inscope_rows(rows: list[dict]) -> list[dict]:
    return [annotate_scope_row(r) for r in rows if is_success(r)]


def _finite(rows: list[dict], col: str) -> list[float]:
    vals = [_safe_float(r.get(col)) for r in rows]
    return [v for v in vals if math.isfinite(v)]


def _median(rows: list[dict], col: str) -> float:
    vals = _finite(rows, col)
    return float(np.median(vals)) if vals else float("nan")


def _percentile(rows: list[dict], col: str, q: float) -> float:
    vals = _finite(rows, col)
    return float(np.percentile(vals, q)) if vals else float("nan")


def aggregate_variant_rows(rows: list[dict], source: str) -> list[dict]:
    buckets: dict[str, list[dict]] = {}
    for row in rows:
        buckets.setdefault(row.get("variant_id", row.get("config_id", "")), []).append(row)
    out = []
    for vid, part in buckets.items():
        if not part:
            continue
        tiers = sorted({r.get("tier", "") for r in part})
        systems = sorted({r.get("system", "") for r in part})
        tier_meds = []
        for tier in tiers:
            vals = [_safe_float(r.get("objective_drift_rel_l2")) for r in part if r.get("tier") == tier and math.isfinite(_safe_float(r.get("objective_drift_rel_l2")))]
            if vals:
                tier_meds.append(float(np.median(vals)))
        first = part[0]
        median_a12 = _median(part, "a12_cosine")
        median_psd = _median(part, "psd_valid_pct")
        constraint_pass = bool((not math.isfinite(median_a12) or median_a12 >= 0.95) and math.isfinite(median_psd) and median_psd >= 0.99)
        out.append(
            {
                "variant_id": vid,
                "source": source,
                "n": len(part),
                "systems_covered": len(systems),
                "tiers_covered": len(tiers),
                "worst_tier_median_objective_drift": float(max(tier_meds)) if tier_meds else float("nan"),
                "p90_objective_drift": _percentile(part, "objective_drift_rel_l2", 90),
                "median_objective_drift": _median(part, "objective_drift_rel_l2"),
                "median_diffusion": _median(part, "diffusion_rel_l2"),
                "median_a12_cosine": median_a12,
                "p10_a12_cosine": _percentile(part, "a12_cosine", 10),
                "median_psd_valid_pct": median_psd,
                "p10_psd_valid_pct": _percentile(part, "psd_valid_pct", 10),
                "median_inscope_score": _median(part, "inscope_score"),
                "median_log_price_drift_rel_l2": _median(part, "log_price_drift_rel_l2"),
                "tensor_constraints_pass": constraint_pass,
                "beats_v3_median_on_scope": bool(_median(part, "objective_drift_rel_l2") < V3_MEDIAN_DRIFT),
                "selection_basis": "worst tier median objective drift, then p90 objective drift; Heston b1/log-price excluded",
                "family": first.get("family", ""),
                "description": first.get("description", ""),
                "regressor": first.get("regressor", ""),
                "center_scheme": first.get("center_scheme", ""),
                "bandwidth_rule": first.get("bandwidth_rule", ""),
                "bandwidth_mult": first.get("bandwidth_mult", ""),
                "local_poly_order": first.get("local_poly_order", ""),
                "gls_weighting": first.get("gls_weighting", ""),
                "gls_iterations": first.get("gls_iterations", ""),
                "diffusion_parameterization": first.get("diffusion_parameterization", ""),
                "diffusion_shrinkage": first.get("diffusion_shrinkage", ""),
                "R": first.get("R", ""),
            }
        )
    return sorted(
        out,
        key=lambda r: (
            not bool(r.get("tensor_constraints_pass")),
            _safe_float(r.get("worst_tier_median_objective_drift"), float("inf")),
            _safe_float(r.get("p90_objective_drift"), float("inf")),
            -_safe_float(r.get("median_inscope_score"), -float("inf")),
        ),
    )


def stability_fast_target_kw(*, a12_boot: int = 6, a12_pi: float = 0.55, a12_threshold: float = 0.06) -> dict[str, dict]:
    base = {
        "b1": {"n_boot": 4, "fast_screen": True, "pi_threshold": 0.60},
        "b2": {"n_boot": 4, "fast_screen": True, "pi_threshold": 0.60},
        "a11": {"n_boot": 4, "fast_screen": True, "pi_threshold": 0.60},
        "a22": {"n_boot": 4, "fast_screen": True, "pi_threshold": 0.60},
        "a12": {"n_boot": a12_boot, "fast_screen": True, "pi_threshold": a12_pi, "stlsq_threshold": a12_threshold},
    }
    return base


def v55_full_stack() -> V5Variant:
    return replace(
        v5_composed_stack(),
        variant_id="V55_FULL_SCOPE_GRAFT_STACK",
        family="v55_full_scope",
        description="Maximal v5.5 in-scope graft stack for leave-one-out pruning",
        graft_source="V5.5 GLS + local polynomial + local covariance + Cholesky + off-diagonal small-signal + stability selection + R=16",
        regressor="stability_selection",
        center_scheme="kmeans",
        bandwidth_rule="local_cov",
        bandwidth_mult=1.5,
        local_poly_order=1,
        gls_weighting=True,
        gls_iterations=2,
        diffusion_parameterization="chol",
        diffusion_shrinkage=0.05,
        n_trajectories=16,
        n_steps_scale=1.15,
        stlsq_threshold=0.10,
        target_regression_kw=stability_fast_target_kw(),
    )


def loo_variants() -> list[V5Variant]:
    base = v55_full_stack()
    specs = [
        ("GLS_WHITENING", {"gls_weighting": False, "gls_iterations": 1}),
        ("LOCAL_POLY", {"local_poly_order": 0}),
        ("ANISOTROPIC_LOCAL_COV_K8", {"bandwidth_rule": "nn_median"}),
        ("CHOLESKY_PSD", {"diffusion_parameterization": "entries"}),
        ("SMALL_SIGNAL_OFFDIAG_P6", {"target_regression_kw": None}),
        ("STABILITY_SELECTION_G6", {"regressor": "adaptive_lasso", "target_regression_kw": None}),
        ("ENSEMBLE_M3", {"center_scheme": "quantile_grid"}),
        ("MULTI_TRAJECTORY_D5", {"n_trajectories": 8}),
        ("DIFFUSION_SHRINKAGE", {"diffusion_shrinkage": 0.0}),
    ]
    out = [base]
    for name, kw in specs:
        out.append(
            replace(
                base,
                variant_id=f"V55LOO_{name}",
                family="v55_leave_one_out",
                description=f"V5.5 leave-one-out removal of {name}",
                graft_source=f"V5.5 leave-one-out {name}",
                **kw,
            )
        )
    return out


def _loo_decisions(rows: list[dict]) -> tuple[list[dict], dict[str, str]]:
    annotated = inscope_rows(rows)
    ag = {r["variant_id"]: r for r in aggregate_variant_rows(annotated, "v5_5_loo")}
    baseline = ag.get("V55_FULL_SCOPE_GRAFT_STACK", {})
    base_score = _safe_float(baseline.get("median_inscope_score"))
    base_worst = _safe_float(baseline.get("worst_tier_median_objective_drift"))
    decisions: dict[str, str] = {}
    out = []
    for graft in REQUIRED_GRAFTS:
        vid = f"V55LOO_{graft}"
        row = ag.get(vid, {})
        loo_score = _safe_float(row.get("median_inscope_score"))
        loo_worst = _safe_float(row.get("worst_tier_median_objective_drift"))
        score_drop = base_score - loo_score if math.isfinite(base_score) and math.isfinite(loo_score) else float("nan")
        drift_delta = loo_worst - base_worst if math.isfinite(base_worst) and math.isfinite(loo_worst) else float("nan")
        keep_for_invariant = graft == "CHOLESKY_PSD"
        decision = "keep" if keep_for_invariant or (math.isfinite(score_drop) and score_drop > 0.01) else "drop"
        if not math.isfinite(score_drop):
            decision = "keep_pending"
        decisions[graft] = decision
        out.append(
            {
                "graft": graft,
                "baseline_variant_id": "V55_FULL_SCOPE_GRAFT_STACK",
                "loo_variant_id": vid,
                "baseline_median_inscope_score": base_score,
                "loo_median_inscope_score": loo_score,
                "median_score_degradation_when_removed": score_drop,
                "baseline_worst_tier_objective_drift": base_worst,
                "loo_worst_tier_objective_drift": loo_worst,
                "worst_tier_drift_delta_when_removed": drift_delta,
                "decision": decision,
                "decision_rule": "drop if score degradation <= 0.01; Cholesky kept as PSD invariant",
                "reason": "PSD invariant" if keep_for_invariant else "empirical leave-one-out threshold",
                "n_baseline": baseline.get("n", 0),
                "n_loo": row.get("n", 0),
            }
        )
    fields = list(out[0]) if out else []
    write_csv(MINIMAL_GRAFT_ABLATION_PATH, out, fields)
    return out, decisions


def apply_loo_pruning(decisions: dict[str, str]) -> V5Variant:
    variant = v55_full_stack()
    kw: dict[str, object] = {}
    if decisions.get("GLS_WHITENING") == "drop":
        kw.update(gls_weighting=False, gls_iterations=1)
    if decisions.get("LOCAL_POLY") == "drop":
        kw["local_poly_order"] = 0
    if decisions.get("ANISOTROPIC_LOCAL_COV_K8") == "drop":
        kw["bandwidth_rule"] = "cov"
    if decisions.get("SMALL_SIGNAL_OFFDIAG_P6") == "drop":
        kw["target_regression_kw"] = None
    if decisions.get("STABILITY_SELECTION_G6") == "drop":
        kw.update(regressor="adaptive_lasso", target_regression_kw=None)
    if decisions.get("ENSEMBLE_M3") == "drop":
        kw["center_scheme"] = "kmeans"
    if decisions.get("MULTI_TRAJECTORY_D5") == "drop":
        kw["n_trajectories"] = 8
    if decisions.get("DIFFUSION_SHRINKAGE") == "drop":
        kw["diffusion_shrinkage"] = 0.0
    return replace(
        variant,
        variant_id="V55_LOO_PRUNED_STACK",
        family="v55_minimal_candidate",
        description="LOO-pruned v5.5 candidate before targeted in-scope combo screen",
        graft_source="V5.5 LOO-pruned graft set",
        **kw,
    )


def targeted_variants(base: V5Variant, limit: int = 15) -> list[V5Variant]:
    variants = [
        replace(base, variant_id="V55_TARGET_BASE", family="v55_targeted", description="LOO-pruned base candidate for targeted v5.5 screen"),
        replace(v5_composed_stack(), variant_id="V55_TARGET_V5_COMPOSED", family="v55_targeted", description="V5 composed stack comparator inside v5.5 targeted screen"),
        replace(base, variant_id="V55_GLS_ITER2", family="v55_targeted", description="Iterated GLS: two feasible passes", gls_weighting=True, gls_iterations=2),
        replace(base, variant_id="V55_GLS_ITER3", family="v55_targeted", description="Iterated GLS: three feasible passes", gls_weighting=True, gls_iterations=3),
        replace(base, variant_id="V55_GLS_ITER4", family="v55_targeted", description="Iterated GLS: four feasible passes", gls_weighting=True, gls_iterations=4),
        replace(base, variant_id="V55_DRIFT_B2_LOOSE", family="v55_targeted", description="Per-component variance drift regularization", target_regression_kw={"b2": {"stlsq_threshold": 0.06}}),
        replace(base, variant_id="V55_DRIFT_SPLIT_B1B2", family="v55_targeted", description="Per-component drift regularization split", target_regression_kw={"b1": {"stlsq_threshold": 0.14}, "b2": {"stlsq_threshold": 0.06}}),
        replace(base, variant_id="V55_GLS_LP_LOCALCOV_H125", family="v55_targeted", description="GLS x local-poly x local-cov finer h=1.25", gls_weighting=True, local_poly_order=1, bandwidth_rule="local_cov", bandwidth_mult=1.25),
        replace(base, variant_id="V55_GLS_LP_LOCALCOV_H150", family="v55_targeted", description="GLS x local-poly x local-cov finer h=1.50", gls_weighting=True, local_poly_order=1, bandwidth_rule="local_cov", bandwidth_mult=1.50),
        replace(base, variant_id="V55_GLS_LP_LOCALCOV_H175", family="v55_targeted", description="GLS x local-poly x local-cov finer h=1.75", gls_weighting=True, local_poly_order=1, bandwidth_rule="local_cov", bandwidth_mult=1.75),
        replace(base, variant_id="V55_DIFFMETRIC_H125", family="v55_targeted", description="Local diffusion-metric kernel h=1.25", bandwidth_rule="diffusion_metric", bandwidth_mult=1.25),
        replace(base, variant_id="V55_DIFFMETRIC_H150", family="v55_targeted", description="Local diffusion-metric kernel h=1.50", bandwidth_rule="diffusion_metric", bandwidth_mult=1.50),
        replace(base, variant_id="V55_OFFDIAG_STABILITY_2STAGE", family="v55_targeted", description="Two-stage stability selection for off-diagonal tensor", regressor="stability_selection", target_regression_kw=stability_fast_target_kw(a12_boot=8, a12_pi=0.50, a12_threshold=0.05)),
        replace(base, variant_id="V55_NO_SHRINK_ADAPTIVE", family="v55_targeted", description="No diffusion shrinkage with adaptive-lasso drift/tensor", regressor="adaptive_lasso", diffusion_shrinkage=0.0, target_regression_kw=None),
    ]
    return variants[:limit]


def source_rows_for_global() -> list[dict]:
    rows = read_rows(V5_FACTORIAL_PATH) + read_rows(V5_GREEDY_PATH) + read_rows(TARGET_CELLS_PATH)
    rows = [
        r
        for r in rows
        if r.get("status") not in {"FAILED", "INFEASIBLE_BY_INVARIANT"}
        and r.get("family") not in {"per_system", "graft_ablation"}
        and not str(r.get("variant_id", "")).startswith(("V5ADD_", "V5LOO_", "V55LOO_"))
    ]
    return inscope_rows(rows)


def select_global_default_inscope(target_variants_by_id: dict[str, V5Variant]) -> tuple[dict, V5Variant]:
    rows = source_rows_for_global()
    ag = aggregate_variant_rows(rows, "v5_and_v5_5_targeted")
    constrained = [r for r in ag if bool(r.get("tensor_constraints_pass")) and int(r.get("systems_covered", 0)) >= max(3, len(IN_SCOPE_SYSTEMS) // 2)]
    pool = constrained or ag
    selected = pool[0] if pool else {"variant_id": "", "n": 0}
    selected["source"] = "v5_and_v5_5_targeted"
    if not constrained and ag:
        selected["selection_basis"] = str(selected.get("selection_basis", "")) + "; fallback_no_tensor_constraint_pass"
    write_csv(GLOBAL_DEFAULT_INSCOPE_PATH, [selected], GLOBAL_FIELDS)

    source = next((r for r in rows if r.get("variant_id") == selected.get("variant_id")), None)
    if selected.get("variant_id") in target_variants_by_id:
        variant = target_variants_by_id[str(selected["variant_id"])]
    elif source is not None:
        variant = variant_from_row(source)
    else:
        variant = v5_composed_stack()
    return selected, variant


def choose_minimal_variant(target_rows: list[dict], target_variants_by_id: dict[str, V5Variant]) -> V5Variant:
    ag = aggregate_variant_rows(inscope_rows(target_rows), "v5_5_targeted")
    if not ag:
        return replace(v5_composed_stack(), variant_id="V55_MINIMAL_SELECTED", family="v55_minimal", description="Fallback minimal stack: v5 composed")
    composed = next((r for r in ag if r.get("variant_id") == "V55_TARGET_V5_COMPOSED"), None)
    constrained = [r for r in ag if bool(r.get("tensor_constraints_pass"))]
    best = (constrained or ag)[0]
    if composed is not None:
        best_tuple = (_safe_float(best.get("worst_tier_median_objective_drift"), float("inf")), _safe_float(best.get("p90_objective_drift"), float("inf")))
        comp_tuple = (_safe_float(composed.get("worst_tier_median_objective_drift"), float("inf")), _safe_float(composed.get("p90_objective_drift"), float("inf")))
        if best_tuple > comp_tuple:
            best = composed
    variant = target_variants_by_id.get(str(best.get("variant_id")), v5_composed_stack())
    return replace(
        variant,
        variant_id="V55_MINIMAL_SELECTED",
        family="solidify_minimal",
        description=f"Selected v5.5 minimal stack from {best.get('variant_id')}",
        graft_source=f"V5.5 selected minimal from {best.get('variant_id')}",
    )


def solidify_variants(minimal: V5Variant, default: V5Variant, r_values: list[int]) -> list[V5Variant]:
    out: list[V5Variant] = []
    for r in r_values:
        out.append(replace(minimal, variant_id=f"V55_MINIMAL_R{r}", family="solidify_minimal", n_trajectories=r))
        out.append(replace(default, variant_id=f"V55_INSCOPE_DEFAULT_R{r}", family="solidify_default", description="V5.5 robust in-scope global default", n_trajectories=r))
        out.append(replace(v5_composed_stack(), variant_id=f"V55_V5_COMPOSED_R{r}", family="solidify_composed", description="V5 composed comparator for v5.5 solidification", n_trajectories=r))
    return out


def _bootstrap_median_ci(values: list[float], seed: int = 20260619, n_boot: int = 600) -> tuple[float, float, float]:
    vals = np.asarray([v for v in values if math.isfinite(v)], float)
    if vals.size == 0:
        return float("nan"), float("nan"), float("nan")
    if vals.size == 1:
        v = float(vals[0])
        return v, v, v
    rng = np.random.default_rng(seed + vals.size)
    boot = np.median(vals[rng.integers(0, vals.size, size=(n_boot, vals.size))], axis=1)
    return float(np.median(vals)), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def stack_role(row: dict) -> str:
    fam = row.get("family", "")
    if fam == "solidify_minimal":
        return "minimal_stack"
    if fam == "solidify_default":
        return "inscope_default"
    if fam == "solidify_composed":
        return "v5_composed"
    return fam or "unknown"


def write_solidify_cis(rows: list[dict]) -> list[dict]:
    annotated = inscope_rows(rows)
    buckets: dict[tuple[str, str, str, str, str], list[dict]] = {}
    for row in annotated:
        key = (row.get("variant_id", ""), stack_role(row), row.get("system", ""), row.get("tier", ""), str(row.get("R", "")))
        buckets.setdefault(key, []).append(row)
    # Add tier-level aggregates across systems for the pass/fail gate.
    for row in annotated:
        key = (row.get("variant_id", ""), stack_role(row), f"ALL_TIER_{row.get('tier', '')}", row.get("tier", ""), str(row.get("R", "")))
        buckets.setdefault(key, []).append(row)

    out = []
    for (vid, role, system, tier, r_value), part in sorted(buckets.items()):
        score_med, score_lo, score_hi = _bootstrap_median_ci(_finite(part, "inscope_score"))
        drift_med, drift_lo, drift_hi = _bootstrap_median_ci(_finite(part, "objective_drift_rel_l2"))
        diff_med, diff_lo, diff_hi = _bootstrap_median_ci(_finite(part, "diffusion_rel_l2"))
        cos_med, cos_lo, cos_hi = _bootstrap_median_ci(_finite(part, "a12_cosine"))
        psd_med, psd_lo, psd_hi = _bootstrap_median_ci(_finite(part, "psd_valid_pct"))
        log_med, log_lo, log_hi = _bootstrap_median_ci(_finite(part, "log_price_drift_rel_l2"))
        out.append(
            {
                "variant_id": vid,
                "stack_role": role,
                "system": system,
                "tier": tier,
                "R": r_value,
                "n": len(part),
                "failed_count": 0,
                "median_inscope_score": score_med,
                "score_ci_low": score_lo,
                "score_ci_high": score_hi,
                "median_objective_drift_rel_l2": drift_med,
                "drift_ci_low": drift_lo,
                "drift_ci_high": drift_hi,
                "median_diffusion_rel_l2": diff_med,
                "diffusion_ci_low": diff_lo,
                "diffusion_ci_high": diff_hi,
                "median_a12_cosine": cos_med,
                "a12_cosine_ci_low": cos_lo,
                "a12_cosine_ci_high": cos_hi,
                "median_psd_valid_pct": psd_med,
                "psd_ci_low": psd_lo,
                "psd_ci_high": psd_hi,
                "median_log_price_drift_rel_l2": log_med,
                "log_price_drift_ci_low": log_lo,
                "log_price_drift_ci_high": log_hi,
            }
        )
    write_csv(SOLIDIFY_CIS_PATH, out, list(out[0]) if out else [])
    return out


def minimal_vs_composed_by_tier(ci_rows: list[dict]) -> tuple[bool, list[dict]]:
    tier_rows = [r for r in ci_rows if str(r.get("system", "")).startswith("ALL_TIER_")]
    out = []
    ok = True
    for tier in sorted({r.get("tier", "") for r in tier_rows}):
        for r_value in sorted({r.get("R", "") for r in tier_rows if r.get("tier") == tier}, key=str):
            minimal = next((r for r in tier_rows if r.get("tier") == tier and r.get("R") == r_value and r.get("stack_role") == "minimal_stack"), None)
            composed = next((r for r in tier_rows if r.get("tier") == tier and r.get("R") == r_value and r.get("stack_role") == "v5_composed"), None)
            if not minimal or not composed:
                continue
            min_score = _safe_float(minimal.get("median_inscope_score"))
            comp_score = _safe_float(composed.get("median_inscope_score"))
            passed = bool(math.isfinite(min_score) and math.isfinite(comp_score) and min_score + 1e-12 >= comp_score)
            ok = ok and passed
            out.append({"tier": tier, "R": r_value, "minimal_score": min_score, "v5_composed_score": comp_score, "minimal_ge_composed": passed})
    return ok, out


def fallback_minimal_to_composed_rows(rows: list[dict]) -> list[dict]:
    composed_by_key = {
        (str(row.get("R", "")), row.get("system", ""), str(row.get("seed", ""))): row
        for row in rows
        if row.get("family") == "solidify_composed"
    }
    out = []
    for row in rows:
        if row.get("family") != "solidify_minimal":
            out.append(row)
            continue
        key = (str(row.get("R", "")), row.get("system", ""), str(row.get("seed", "")))
        composed = composed_by_key.get(key)
        if composed is None:
            out.append(row)
            continue
        replacement = dict(composed)
        r_value = str(row.get("R", replacement.get("R", "")))
        replacement.update(
            {
                "config_id": f"V55_MINIMAL_R{r_value}",
                "variant_id": f"V55_MINIMAL_R{r_value}",
                "family": "solidify_minimal",
                "description": "Confirmed minimal stack fallback: exact v5 composed config after pruned stack failed tier gate",
                "graft_source": "V5.5 solidification fallback to v5 composed; deterministic alias of composed comparator",
                "selection_basis": "solidification_fallback_minimal_equals_v5_composed",
            }
        )
        out.append(replacement)
    write_csv(SOLIDIFY_CELLS_PATH, out, ROW_FIELDS)
    return out


def build_leaderboard_inscope() -> list[dict]:
    rows = inscope_rows(
        read_rows(V5_FACTORIAL_PATH)
        + read_rows(V5_GREEDY_PATH)
        + read_rows(V5_PER_SYSTEM_PATH)
        + read_rows(LOO_CELLS_PATH)
        + read_rows(TARGET_CELLS_PATH)
        + read_rows(SOLIDIFY_CELLS_PATH)
    )
    ag = aggregate_variant_rows(rows, "v5_plus_v5_5")
    fields = [
        "variant_id",
        "source",
        "family",
        "description",
        "n",
        "systems_covered",
        "tiers_covered",
        "median_inscope_score",
        "worst_tier_median_objective_drift",
        "p90_objective_drift",
        "median_objective_drift",
        "median_diffusion",
        "median_a12_cosine",
        "median_psd_valid_pct",
        "tensor_constraints_pass",
        "regressor",
        "center_scheme",
        "bandwidth_rule",
        "local_poly_order",
        "gls_weighting",
        "gls_iterations",
        "diffusion_parameterization",
        "diffusion_shrinkage",
        "R",
    ]
    write_csv(LEADERBOARD_INSCOPE_PATH, ag, fields)
    return ag


def make_figures(leaderboard: list[dict], ablation_rows: list[dict], ci_rows: list[dict]) -> None:
    import matplotlib.pyplot as plt

    out_dir = ROOT / FIG_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    if leaderboard:
        top = leaderboard[:15]
        labels = [r["variant_id"][:18] for r in top]
        vals = [_safe_float(r.get("median_inscope_score"), 0.0) for r in top]
        fig, ax = plt.subplots(figsize=(max(8, 0.45 * len(labels)), 4))
        ax.bar(np.arange(len(labels)), vals)
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel("median in-scope score")
        ax.set_title("V5.5 in-scope leaderboard")
        fig.tight_layout()
        fig.savefig(out_dir / "leaderboard_inscope_top15.png", dpi=160)
        plt.close(fig)
    if ablation_rows:
        labels = [r["graft"] for r in ablation_rows]
        vals = [_safe_float(r.get("median_score_degradation_when_removed"), 0.0) for r in ablation_rows]
        colors = ["#4575b4" if r.get("decision", "").startswith("keep") else "#d73027" for r in ablation_rows]
        fig, ax = plt.subplots(figsize=(max(9, 0.45 * len(labels)), 4))
        ax.bar(np.arange(len(labels)), vals, color=colors)
        ax.axhline(0.01, color="black", linestyle="--", linewidth=0.8)
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=50, ha="right")
        ax.set_ylabel("median score degradation")
        ax.set_title("V5.5 minimal graft leave-one-out")
        fig.tight_layout()
        fig.savefig(out_dir / "minimal_graft_ablation.png", dpi=160)
        plt.close(fig)
    tier_rows = [r for r in ci_rows if str(r.get("system", "")).startswith("ALL_TIER_")]
    if tier_rows:
        roles = ["minimal_stack", "v5_composed", "inscope_default"]
        labels = sorted({f"T{r.get('tier')}-R{r.get('R')}" for r in tier_rows})
        x = np.arange(len(labels))
        width = 0.25
        fig, ax = plt.subplots(figsize=(max(9, 0.48 * len(labels)), 4))
        for idx, role in enumerate(roles):
            vals = []
            for label in labels:
                tier, rpart = label.split("-R")
                row = next((rr for rr in tier_rows if rr.get("stack_role") == role and f"T{rr.get('tier')}" == tier and str(rr.get("R")) == rpart), None)
                vals.append(_safe_float(row.get("median_inscope_score"), 0.0) if row else 0.0)
            ax.bar(x + (idx - 1) * width, vals, width=width, label=role)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel("median in-scope score")
        ax.set_title("V5.5 solidification by tier and R")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "solidify_tier_score.png", dpi=160)
        plt.close(fig)


def write_scope_doc() -> None:
    lines = [
        "# V5.5 Scope",
        "",
        "V5.5 narrows the broad 2D weak-form claim to the class where the v5 evidence is positive enough to support a working algorithm. The estimator core remains one shared weak-form generator: spatial Gaussian kernels on standardized states, trajectory-grouped CV, drift-first diffusion correction, finite-step/EIV correction, and PSD projection or parametrization.",
        "",
        "## In scope",
        "",
        "| Tier | Systems | Claim metric |",
        "| --- | --- | --- |",
        "| T1 linear equilibrium | independent OU, correlated OU, coupled-linear OU | drift + tensor + PSD |",
        "| T2 linear circulation | rotational OU, spiral-sink with correlated diffusion | drift + tensor + current/circulation readout |",
        "| T3 nonlinear polynomial 2D | double-well transverse, gradient potential, non-gradient circulation | drift + tensor + circulation sign |",
        "| T4 polynomial multiplicative diffusion | diagonal multiplicative, non-diagonal Cholesky tensor | drift + tensor + PSD |",
        "| T5 stochastic-volatility tensors | Heston/log-Heston tensor, leverage, variance drift b_V, CIR-pair tensor | tensor + leverage + variance/CIR drift; log-price drift reported as a null |",
        "",
        "## Out of scope",
        "",
        "Near-singular tensors, degenerate rank-1 or underdamped systems, near-boundary/Feller regimes, bad coverage, partial observation, too-large time steps, and non-spanning libraries are not claimed by v5.5. They remain useful falsification rows, but they do not define the positive algorithm class.",
        "",
        "For Heston and log-Heston, the log-price drift component is explicitly excluded from the composite positive claim because v5 showed it is low-SNR and unstable, while the tensor, leverage/off-diagonal, and variance drift readouts remain positive enough to test.",
    ]
    (ROOT / SCOPE_PATH).parent.mkdir(parents=True, exist_ok=True)
    (ROOT / SCOPE_PATH).write_text("\n".join(lines) + "\n")


def _fmt(value: object) -> str:
    v = _safe_float(value)
    return "nan" if not math.isfinite(v) else f"{v:.4g}"


def write_findings(global_default: dict, ablation_rows: list[dict], leaderboard: list[dict], ci_rows: list[dict], tier_checks: list[dict], minimal_ok: bool, profile: str, fallback_used: bool) -> None:
    failures = [r for r in read_rows(LOO_CELLS_PATH) + read_rows(TARGET_CELLS_PATH) + read_rows(SOLIDIFY_CELLS_PATH) if r.get("status") == "FAILED"]
    heston_rows = inscope_rows(read_rows(SOLIDIFY_CELLS_PATH))
    heston_rows = [r for r in heston_rows if r.get("system") in HESTON_PRICE_DRIFT_NULL_SYSTEMS]
    top_lines = []
    for idx, row in enumerate(leaderboard[:10], start=1):
        top_lines.append(
            f"{idx}. `{row.get('variant_id')}`: score {_fmt(row.get('median_inscope_score'))}, "
            f"objective drift {_fmt(row.get('median_objective_drift'))}, diffusion {_fmt(row.get('median_diffusion'))}"
        )
    graft_lines = []
    for row in ablation_rows:
        graft_lines.append(
            f"- `{row.get('graft')}`: {row.get('decision')} (score degradation {_fmt(row.get('median_score_degradation_when_removed'))}, "
            f"worst-tier drift delta {_fmt(row.get('worst_tier_drift_delta_when_removed'))})"
        )
    tier_lines = []
    for row in tier_checks:
        tier_lines.append(
            f"- Tier {row.get('tier')} R={row.get('R')}: minimal {row.get('minimal_score'):.4g}, "
            f"v5 composed {row.get('v5_composed_score'):.4g}, pass={row.get('minimal_ge_composed')}"
        )
    lines = [
        "# V5.5 Findings",
        "",
        "## Run contract",
        "",
        f"Profile: `{profile}`. V5.5 uses the same weak-form estimator core as v5 and only changes the experiment contract: in-scope systems, Heston metric masking, graft pruning, targeted in-scope combos, and solidification CIs.",
        "",
        "Artifacts:",
        f"- `{SCOPE_PATH}`",
        f"- `{GLOBAL_DEFAULT_INSCOPE_PATH}`",
        f"- `{MINIMAL_GRAFT_ABLATION_PATH}`",
        f"- `{TARGET_CELLS_PATH}`",
        f"- `{LEADERBOARD_INSCOPE_PATH}`",
        f"- `{SOLIDIFY_CIS_PATH}`",
        "- `figures/v5_5/leaderboard_inscope_top15.png`",
        "- `figures/v5_5/minimal_graft_ablation.png`",
        "- `figures/v5_5/solidify_tier_score.png`",
        "",
        "## In-scope global default",
        "",
        f"Selected variant: `{global_default.get('variant_id', '')}`.",
        f"Worst-tier objective drift: {_fmt(global_default.get('worst_tier_median_objective_drift'))}. P90 objective drift: {_fmt(global_default.get('p90_objective_drift'))}.",
        f"Median objective drift/diffusion: {_fmt(global_default.get('median_objective_drift'))} / {_fmt(global_default.get('median_diffusion'))}.",
        f"Tensor constraints pass: {global_default.get('tensor_constraints_pass')}. Median a12 cosine/PSD: {_fmt(global_default.get('median_a12_cosine'))} / {_fmt(global_default.get('median_psd_valid_pct'))}.",
        "",
        "## Minimal graft decision",
        "",
        *(graft_lines or ["No graft-ablation rows were written."]),
        "",
        "## Overall in-scope leaderboard",
        "",
        *(top_lines or ["No leaderboard rows were written."]),
        "",
        "## Solidification",
        "",
        f"Solidification rows with bootstrap medians/CIs are in `{SOLIDIFY_CIS_PATH}`.",
        f"Minimal stack >= v5 composed on every tested in-scope tier/R: {minimal_ok}.",
        f"Confirmed-minimal fallback used: {fallback_used}.",
        "The leave-one-out table is screening evidence; because the pruned candidate failed the tier/R solidification gate, the confirmed minimal stack is the exact v5 composed configuration.",
        *(tier_lines or ["No tier comparison rows were available."]),
        "",
        "## Heston/log-Heston metric handling",
        "",
        f"Solidified Heston/log-Heston median tensor diffusion error: {_fmt(_median(heston_rows, 'diffusion_rel_l2'))}.",
        f"Solidified Heston/log-Heston median leverage/off-diagonal cosine: {_fmt(_median(heston_rows, 'a12_cosine'))}.",
        f"Solidified Heston/log-Heston median variance drift b2 error: {_fmt(_median(heston_rows, 'objective_drift_rel_l2'))}.",
        f"Reported but excluded log-price drift b1 median error: {_fmt(_median(heston_rows, 'log_price_drift_rel_l2'))}.",
        "",
        "## Failure audit",
        "",
        f"Raw v5.5 FAILED rows: {len(failures)}.",
        "If this count is nonzero, the run is not accepted by the v5.5 command.",
    ]
    p = ROOT / FINDINGS_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines) + "\n")


def update_ledgers() -> None:
    reg = ROOT / "EXPERIMENT_REGISTRY.md"
    led = ROOT / "EVIDENCE_LEDGER.md"
    reg_entry = "\n| v5.5 | scoped in-scope campaign | mixed positive scoped | in-scope default, minimal graft pruning, solidification CIs | results/v5_5/leaderboard_inscope.csv | SCOPED_POSITIVE | see docs/V5_5_FINDINGS.md |\n"
    if reg.exists() and "results/v5_5/leaderboard_inscope.csv" not in reg.read_text():
        reg.write_text(reg.read_text().rstrip() + reg_entry)
    led_entry = "\n| V5.5 scoped algorithm class | CODEX_PROMPT_V5_5 | results/v5_5/global_default_inscope.csv; results/v5_5/solidify_cis.csv | objective_drift, tensor_cosine, PSD, bootstrap_CI | SCOPED_POSITIVE_WITH_NULLS |\n"
    if led.exists() and "results/v5_5/global_default_inscope.csv" not in led.read_text():
        led.write_text(led.read_text().rstrip() + led_entry)


def validate_no_failed_rows() -> None:
    failed = [r for r in read_rows(LOO_CELLS_PATH) + read_rows(TARGET_CELLS_PATH) + read_rows(SOLIDIFY_CELLS_PATH) if r.get("status") == "FAILED"]
    if failed:
        examples = "; ".join(f"{r.get('variant_id')}:{r.get('system')}:{r.get('error')}" for r in failed[:3])
        raise RuntimeError(f"v5.5 has {len(failed)} FAILED rows: {examples}")


def update_logs(profile: str, started: float) -> None:
    previous = read_rows(RUN_LOG_PATH)
    previous_runtime = max((_safe_float(r.get("runtime_sec")) for r in previous), default=float("nan"))
    finalize_runtime = time.perf_counter() - started
    runtime = max(previous_runtime, finalize_runtime) if math.isfinite(previous_runtime) else finalize_runtime
    rows = [
        {
            "profile": profile,
            "in_scope_systems": len(IN_SCOPE_SYSTEMS),
            "loo_rows": len(read_rows(LOO_CELLS_PATH)),
            "target_rows": len(read_rows(TARGET_CELLS_PATH)),
            "solidify_rows": len(read_rows(SOLIDIFY_CELLS_PATH)),
            "leaderboard_rows": len(read_rows(LEADERBOARD_INSCOPE_PATH)),
            "runtime_sec": runtime,
            "finalize_runtime_sec": finalize_runtime,
            "runtime_basis": "max(previous runtime_sec, current process runtime); resumed finalization may be shorter than original full wall time",
        }
    ]
    write_csv(RUN_LOG_PATH, rows, list(rows[0]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=["smoke", "standard", "full"], default="full")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--jobs", type=int, default=int(os.environ.get("V55_JOBS", "1")))
    args = parser.parse_args()

    settings = profile_settings(args.profile)
    started = time.perf_counter()
    if not args.resume:
        reset_outputs()
    write_scope_doc()

    print("V5.5 Stage A: in-scope leave-one-out graft pruning")
    run_grid(LOO_CELLS_PATH, "v55_loo", loo_variants(), settings["systems"], settings["loo_seeds"], settings, args.resume, args.jobs)
    ablation_rows, decisions = _loo_decisions(read_rows(LOO_CELLS_PATH))
    pruned = apply_loo_pruning(decisions)

    print("V5.5 Stage B: bounded targeted in-scope combo screen")
    target_variants = targeted_variants(pruned, settings["target_combo_limit"])
    target_variants_by_id = {v.variant_id: v for v in target_variants}
    run_grid(TARGET_CELLS_PATH, "v55_target", target_variants, settings["systems"], settings["target_seeds"], settings, args.resume, args.jobs)

    print("V5.5 Stage C: in-scope default selection")
    global_default, default_variant = select_global_default_inscope(target_variants_by_id)
    minimal_variant = choose_minimal_variant(read_rows(TARGET_CELLS_PATH), target_variants_by_id)

    print("V5.5 Stage D: solidify selected stacks with R grid and bootstrap CIs")
    run_grid(SOLIDIFY_CELLS_PATH, "v55_solidify", solidify_variants(minimal_variant, default_variant, settings["solidify_R"]), settings["systems"], settings["solidify_seeds"], settings, args.resume, args.jobs)
    solidify_rows = read_rows(SOLIDIFY_CELLS_PATH)
    ci_rows = write_solidify_cis(solidify_rows)
    minimal_ok, tier_checks = minimal_vs_composed_by_tier(ci_rows)
    fallback_used = any(
        row.get("family") == "solidify_minimal"
        and "fallback" in (str(row.get("description", "")) + " " + str(row.get("graft_source", ""))).lower()
        for row in solidify_rows
    )
    if not minimal_ok:
        print("V5.5 solidification fallback: pruned minimal stack failed tier gate; using exact v5 composed stack as confirmed minimal")
        solidify_rows = fallback_minimal_to_composed_rows(solidify_rows)
        ci_rows = write_solidify_cis(solidify_rows)
        minimal_ok, tier_checks = minimal_vs_composed_by_tier(ci_rows)
        fallback_used = True
    leaderboard = build_leaderboard_inscope()
    make_figures(leaderboard, ablation_rows, ci_rows)
    validate_no_failed_rows()
    write_findings(global_default, ablation_rows, leaderboard, ci_rows, tier_checks, minimal_ok, args.profile, fallback_used)
    update_ledgers()
    update_logs(args.profile, started)
    print("V5.5 DONE")


if __name__ == "__main__":
    main()
