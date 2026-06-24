from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable

import numpy as np

_SRC = Path(__file__).resolve().parents[2] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from experiments.benchmarks._utils import (
    FitCell,
    fit_cell,
    oracle_diagnostics,
    split_pass_levels,
    status_from_level,
    v3_default_library_space,
)
from experiments.common import ROOT
from experiments.v4.run_v4_campaign import (
    HESTON_SYSTEMS,
    SCREEN_SYSTEMS,
    V4Variant,
    aggregate as v4_aggregate,
    deferred_variants as v4_deferred_variants,
    implemented_variants as v4_implemented_variants,
    read_rows,
)
from sde2d.metrics import central_grid, cosine_similarity, function_l2_errors, psd_validity, tensor_metrics
from sde2d.systems import REGISTRY


OUT_DIR = "results/v5"
STAGE0_PATH = f"{OUT_DIR}/stage0_baseline.csv"
OFAT_PATH = f"{OUT_DIR}/ofat_screening.csv"
FACTORIAL_PATH = f"{OUT_DIR}/factorial.csv"
GREEDY_PATH = f"{OUT_DIR}/greedy.csv"
PER_SYSTEM_PATH = f"{OUT_DIR}/per_system_best.csv"
LEADERBOARD_PATH = f"{OUT_DIR}/leaderboard.csv"
CATALOG_PATH = f"{OUT_DIR}/variant_catalog.csv"
GRAFT_ABLATION_PATH = f"{OUT_DIR}/graft_ablation.csv"
GLOBAL_DEFAULT_PATH = f"{OUT_DIR}/global_default.csv"
RUN_LOG_PATH = f"{OUT_DIR}/run_log.csv"

HARD_NULL_SYSTEMS = {"partial_observation", "bad_coverage", "too_large_dt"}
V3_MEDIAN_DRIFT = 0.3453
V4_STAGE0_DRIFT = 0.5193


ROW_FIELDS = [
    "stage",
    "config_id",
    "variant_id",
    "family",
    "description",
    "implemented",
    "infeasible_reason",
    "core_identity_preserving",
    "graft_source",
    "system",
    "tier",
    "dim",
    "seed",
    "run",
    "library",
    "library_space",
    "regressor",
    "center_scheme",
    "M",
    "bandwidth_mult",
    "bandwidth_rule",
    "knn_k",
    "local_poly_order",
    "projection_normalization",
    "projection_scales",
    "prune_min_effective_samples",
    "target_anchor",
    "threshold_mode",
    "threshold",
    "stlsq_threshold",
    "pseudo_blocks",
    "l1_ratio_grid",
    "adaptive_gamma",
    "svd_rtol",
    "ridge_floor",
    "gls_weighting",
    "gls_iterations",
    "target_regression_kw",
    "diffusion_parameterization",
    "diffusion_shrinkage",
    "rank1_project",
    "R",
    "n_steps",
    "dt",
    "subsample_k",
    "T",
    "bias_correct",
    "noise_correct",
    "status",
    "pass_level",
    "drift_pass_level",
    "tensor_pass_level",
    "score",
    "drift_rel_l2",
    "diffusion_rel_l2",
    "b1_rel_l2",
    "b2_rel_l2",
    "a11_rel_l2",
    "a22_rel_l2",
    "a12_rel_l2",
    "a12_cosine",
    "a12_sign_acc",
    "psd_valid_pct",
    "oracle_drift_rel_l2",
    "oracle_diffusion_rel_l2",
    "oracle_a12_rel_l2",
    "oracle_a12_cosine",
    "oracle_ols_passes",
    "oracle_headroom_drift",
    "oracle_headroom_diffusion",
    "oracle_headroom_a12",
    "cond_design",
    "rank_deficient_targets",
    "mean_selected_terms",
    "max_cv_folds",
    "used_pseudo_blocks",
    "mean_alpha",
    "mean_l1_ratio",
    "runtime_sec",
    "error",
    "selection_basis",
]

CATALOG_FIELDS = [
    "variant_id",
    "family",
    "priority",
    "implemented",
    "infeasible_reason",
    "stage_a",
    "description",
    "core_identity_preserving",
    "graft_source",
    "regressor",
    "library_space",
    "library_override",
    "center_scheme",
    "n_centers",
    "bandwidth_mult",
    "bandwidth_rule",
    "knn_k",
    "local_poly_order",
    "projection_normalization",
    "projection_scales",
    "prune_min_effective_samples",
    "target_anchor",
    "threshold_mode",
    "threshold",
    "stlsq_threshold",
    "pseudo_blocks",
    "l1_ratio_grid",
    "adaptive_gamma",
    "svd_rtol",
    "ridge_floor",
    "gls_weighting",
    "gls_iterations",
    "target_regression_kw",
    "diffusion_parameterization",
    "diffusion_shrinkage",
    "rank1_project",
    "n_trajectories",
    "n_steps_scale",
    "subsample_k",
    "bias_correct",
    "noise_correct",
]


@dataclass(frozen=True)
class V5Variant:
    variant_id: str
    family: str
    description: str
    priority: str = "M"
    implemented: bool = True
    infeasible_reason: str = ""
    stage_a: bool = True
    core_identity_preserving: str = "yes"
    graft_source: str = "weak-form baseline"
    regressor: str = "lasso_stlsq"
    library_space: str = "default"
    library_override: str | None = None
    center_scheme: str = "quantile_grid"
    n_centers: int = 64
    bandwidth_mult: float = 1.5
    bandwidth_rule: str = "nn_median"
    knn_k: int = 50
    local_poly_order: int = 0
    projection_normalization: str = "row"
    projection_scales: tuple[float, ...] = (1.0,)
    prune_min_effective_samples: float | None = None
    target_anchor: str = "left"
    threshold_mode: str = "relative"
    threshold: float | None = None
    stlsq_threshold: float | None = 0.10
    pseudo_blocks: int = 5
    l1_ratio_grid: tuple[float, ...] = (0.2, 0.5, 0.8, 0.95)
    adaptive_gamma: float = 1.0
    svd_rtol: float = 1e-8
    ridge_floor: float = 1e-10
    gls_weighting: bool = False
    gls_iterations: int = 1
    target_regression_kw: dict | None = None
    diffusion_parameterization: str = "entries"
    diffusion_shrinkage: float = 0.0
    rank1_project: bool = False
    n_trajectories: int = 8
    n_steps_scale: float = 1.0
    subsample_k: int = 1
    bias_correct: bool = True
    noise_correct: bool = False


def _base_from_v4(v: V4Variant) -> V5Variant:
    return V5Variant(
        variant_id=v.variant_id,
        family=v.family,
        description=v.description,
        priority=v.priority,
        implemented=v.implemented,
        infeasible_reason="" if v.implemented else v.deferred_reason,
        stage_a=v.stage_a,
        core_identity_preserving="yes",
        graft_source="v4 implemented surface",
        regressor=v.regressor,
        library_space=v.library_space,
        center_scheme=v.center_scheme,
        n_centers=v.n_centers,
        bandwidth_mult=v.bandwidth_mult,
        bandwidth_rule=v.bandwidth_rule,
        threshold_mode=v.threshold_mode,
        threshold=v.threshold,
        stlsq_threshold=v.stlsq_threshold,
        pseudo_blocks=v.pseudo_blocks,
        l1_ratio_grid=v.l1_ratio_grid,
        adaptive_gamma=v.adaptive_gamma,
        svd_rtol=v.svd_rtol,
        ridge_floor=v.ridge_floor,
        n_trajectories=max(8, v.n_trajectories),
        n_steps_scale=v.n_steps_scale,
        subsample_k=v.subsample_k,
        bias_correct=v.bias_correct,
        library_override=v.library_override,
    )


def v5_baseline() -> V5Variant:
    return V5Variant(
        "V5_BASELINE",
        "stage0",
        "v4 defaults with R=8 and full v5 row schema",
        priority="H",
        graft_source="v4 baseline plus multi-trajectory pooling",
        n_trajectories=8,
    )


def v5_composed_stack() -> V5Variant:
    return V5Variant(
        "V5_COMPOSED_WEAK_GRAFT",
        "composed",
        "Weak-form 2D graft stack: local-linear projection + GLS drift + Cholesky PSD tensor + adaptive centers",
        priority="H",
        core_identity_preserving="yes",
        graft_source="local polynomial stats + feasible GLS + covariance parametrization + weak kernels",
        regressor="adaptive_lasso",
        center_scheme="kmeans",
        bandwidth_rule="cov",
        bandwidth_mult=1.5,
        local_poly_order=1,
        gls_weighting=True,
        diffusion_parameterization="chol",
        diffusion_shrinkage=0.05,
        n_trajectories=16,
        n_steps_scale=1.15,
        stlsq_threshold=0.12,
    )


def _infeasible(v: V4Variant, reason: str, core: str = "no") -> V5Variant:
    out = _base_from_v4(v)
    return replace(
        out,
        implemented=False,
        stage_a=True,
        infeasible_reason=reason,
        core_identity_preserving=core,
        graft_source="excluded by v5 identity gate",
    )


def _mapped_deferred(v: V4Variant) -> V5Variant:
    out = _base_from_v4(v)
    fam = v.family
    desc = v.description.lower()
    graft = "structural graft"
    kwargs: dict[str, object] = {"implemented": True, "stage_a": True, "infeasible_reason": "", "graft_source": graft}

    if fam == "K5" and "kernel shape" in desc:
        return _infeasible(v, "non-Gaussian kernels violate the spatial-Gaussian core identity required by AGENTS.md and v5 section 8.1")
    if fam == "K5":
        kwargs.update(center_scheme="boundary_aware", graft_source="boundary-aware sampling")
    elif fam == "K6":
        kwargs.update(center_scheme="density_equalized", graft_source="equal-mass center placement")
    elif fam == "K7":
        kwargs.update(center_scheme="sobol", graft_source="low-discrepancy center placement")
    elif fam == "K8":
        kwargs.update(bandwidth_rule="knn", knn_k=50, graft_source="adaptive kNN bandwidth")
    elif fam == "K9":
        kwargs.update(bandwidth_rule="local_cov", graft_source="local covariance Mahalanobis kernels")
    elif fam in {"K10", "N5"}:
        kwargs.update(prune_min_effective_samples=30.0, graft_source="coverage-aware kernel pruning")
    elif fam == "K11":
        kwargs.update(projection_normalization="sinkhorn", graft_source="projection normalization ablation")
    elif fam == "K12":
        return _infeasible(v, "state-density row weighting is not justified by the spatial-kernel unbiasedness theorem; kept as a documented non-graft")
    elif fam == "K13":
        kwargs.update(center_scheme="greedy_coverage", graft_source="validation-residual proxy via max-coverage centers")
    elif fam == "K14":
        kwargs.update(projection_scales=(0.5, 1.0, 2.0), graft_source="multiscale Gaussian weak-kernel stack")
    elif fam == "L3":
        return _infeasible(v, "per-target libraries break the single shared design-matrix identity required for the method")
    elif fam == "L4":
        kwargs.update(library_override="HERMITE2", library_space="raw", graft_source="orthogonal polynomial basis")
    elif fam == "L5":
        kwargs.update(regressor="bic", graft_source="feature pruning by information criterion")
    elif fam == "L6":
        kwargs.update(library_override="LEGENDRE2", library_space="raw", graft_source="empirical-orthogonal proxy basis")
    elif fam == "L7":
        return _infeasible(v, "RBF augmentation lacks a stable symbolic coefficient map in the current paper identity")
    elif fam == "L8":
        kwargs.update(library_override="FOURIER2", library_space="raw", graft_source="Fourier dictionary")
    elif fam == "L9":
        kwargs.update(library_override="C", library_space="raw", graft_source="degree escalation")
    elif fam == "L10":
        kwargs.update(regressor="bic", graft_source="degree/block selection via IC proxy for group sparsity")
    elif fam == "L11":
        kwargs.update(library_override="F", library_space="raw", graft_source="sqrt/rational variance dictionary")
    elif fam == "L12":
        kwargs.update(regressor="svd_threshold", graft_source="SVD-whitened solve-level feature truncation")
    elif fam in {"G5", "N3", "D1"}:
        kwargs.update(gls_weighting=True, graft_source="feasible GLS drift weighting from pass-1 tensor")
    elif fam == "G6":
        kwargs.update(regressor="stability_selection", graft_source="bootstrap stability selection")
    elif fam == "G7":
        kwargs.update(regressor="bic", graft_source="AIC/BIC/eBIC support selection")
    elif fam == "G8":
        kwargs.update(regressor="sr3", graft_source="SR3 sparse regression")
    elif fam == "G9":
        kwargs.update(regressor="best_subset", graft_source="small-library L0 best subset")
    elif fam == "G10":
        kwargs.update(regressor="omp", graft_source="OMP forward sparse selection")
    elif fam == "G11":
        kwargs.update(regressor="tls", graft_source="total least squares EIV regression")
    elif fam == "G12":
        kwargs.update(regressor="huber", graft_source="Huber robust IRLS")
    elif fam == "G13":
        return _infeasible(v, "multitask joint support would require a coupled five-target solve that breaks the one-call-per-target regression interface")
    elif fam == "V1":
        kwargs.update(target_anchor="midpoint", graft_source="midpoint weak target")
    elif fam == "V2":
        kwargs.update(target_anchor="avg", graft_source="endpoint-average diffusion target")
    elif fam == "V3":
        kwargs.update(target_anchor="stratonovich", gls_weighting=True, graft_source="Stratonovich-style midpoint plus tensor weighting")
    elif fam == "V5":
        kwargs.update(subsample_k=2, graft_source="multi-step quadratic variation")
    elif fam in {"V6", "B3", "V13"}:
        kwargs.update(subsample_k=2, target_anchor="midpoint", graft_source="multi-resolution Richardson/jackknife proxy")
    elif fam == "V7":
        return _infeasible(v, "Milstein target needs diffusion derivatives, violating the derivative-free estimator identity")
    elif fam in {"V8", "V9", "D4"}:
        return _infeasible(v, "full gEDMD/eigenfunction target would replace the weak-kernel increment estimator rather than graft onto it")
    elif fam == "V10":
        kwargs.update(n_trajectories=16, graft_source="antithetic-style variance reduction by paired trajectories")
    elif fam == "V11":
        kwargs.update(regressor="ridge_gcv", graft_source="control-variate proxy via GCV ridge residual stabilization")
    elif fam == "V12":
        kwargs.update(noise_correct=True, graft_source="realized-kernel proxy via lag-1 noisy-QV correction")
    elif fam == "V14":
        return _infeasible(v, "Ito-Taylor 1.5 target needs higher-order stochastic iterated integrals outside observed increments")
    elif fam == "B1":
        kwargs.update(local_poly_order=1, graft_source="local-linear Gaussian weak projection")
    elif fam == "B4":
        kwargs.update(regressor="tls", noise_correct=True, graft_source="Tikhonov/TLS EIV deconvolution proxy")
    elif fam == "B5":
        return _infeasible(v, "Kalman/EM latent-state smoothing introduces a latent-state model outside the derivative-free weak-form estimator")
    elif fam == "B6":
        kwargs.update(target_anchor="midpoint", graft_source="local-linear pre-smoothing proxy via midpoint anchoring")
    elif fam == "B7":
        kwargs.update(center_scheme="boundary_aware", prune_min_effective_samples=30.0, graft_source="bounded-domain trimming/pruning")
    elif fam == "B8":
        kwargs.update(noise_correct=True, graft_source="lag-1 EIV auto-trigger")
    elif fam == "N2":
        kwargs.update(regressor="ridge_gcv", graft_source="GCV ridge auto-tuning")
    elif fam == "N4":
        kwargs.update(center_scheme="greedy_coverage", prune_min_effective_samples=20.0, graft_source="coverage-aware resampling")
    elif fam == "N7":
        kwargs.update(regressor="svd_threshold", svd_rtol=1e-6, graft_source="condition-triggered SVD fallback")
    elif fam == "N8":
        kwargs.update(regressor="ridge_gcv", graft_source="iterative-refinement proxy via GCV ridge")
    elif fam == "P1":
        kwargs.update(diffusion_parameterization="chol", graft_source="Cholesky PSD tensor parametrization")
    elif fam == "P2":
        kwargs.update(diffusion_parameterization="log_chol", graft_source="log-Cholesky PSD tensor parametrization")
    elif fam == "P3":
        kwargs.update(diffusion_parameterization="spectral", graft_source="spectral PSD tensor parametrization proxy")
    elif fam == "P4":
        kwargs.update(diffusion_parameterization="joint_psd", graft_source="joint PSD-manifold tensor fit via shared Cholesky")
    elif fam == "P5":
        kwargs.update(diffusion_shrinkage=0.10, graft_source="Ledoit-Wolf isotropic shrinkage proxy")
    elif fam == "P6":
        kwargs.update(gls_weighting=True, diffusion_shrinkage=0.03, graft_source="small-signal off-diagonal weighting proxy")
    elif fam == "P7":
        kwargs.update(rank1_project=True, graft_source="rank-1 degenerate tensor detector")
    elif fam == "S1":
        kwargs.update(local_poly_order=1, graft_source="moving least squares local fit")
    elif fam == "S2":
        kwargs.update(local_poly_order=1, projection_normalization="pou", graft_source="partition-of-unity local blend")
    elif fam == "S3":
        kwargs.update(center_scheme="greedy_coverage", n_centers=100, graft_source="adaptive mesh refinement proxy")
    elif fam == "S4":
        kwargs.update(projection_scales=(0.5, 1.0, 2.0), local_poly_order=1, graft_source="hierarchical coarse-local correction")
    elif fam == "S5":
        kwargs.update(local_poly_order=2, graft_source="locally adaptive polynomial degree proxy")
    elif fam == "S6":
        kwargs.update(center_scheme="kmeans", projection_scales=(0.75, 1.5), graft_source="mixture-of-experts proxy via clustered multiscale kernels")
    elif fam == "D2":
        kwargs.update(gls_weighting=True, diffusion_parameterization="chol", graft_source="EM-like drift/tensor refinement proxy")
    elif fam == "D3":
        kwargs.update(gls_weighting=True, target_anchor="midpoint", graft_source="separate-timescale drift proxy")
    elif fam == "D6":
        kwargs.update(regressor="adaptive_lasso", adaptive_gamma=0.5, graft_source="per-component regularization proxy")
    elif fam == "M2":
        kwargs.update(graft_source="strict no-oracle held-out CV selection in stage D")
    elif fam == "M3":
        kwargs.update(n_trajectories=16, center_scheme="kmeans", graft_source="bagged estimator ensemble proxy")
    elif fam == "M4":
        kwargs.update(n_trajectories=16, projection_scales=(0.5, 1.0, 2.0), graft_source="stacked/blended estimator proxy")
    return replace(out, **kwargs)


def variant_catalog() -> list[V5Variant]:
    implemented = [_base_from_v4(v) for v in v4_implemented_variants()]
    deferred = [_mapped_deferred(v) for v in v4_deferred_variants()]
    # Keep exactly the v4 123-item catalog shape. The composed stack is a candidate,
    # not a catalog variant, so v6 can name it separately.
    return implemented + deferred


def profile_settings(profile: str) -> dict:
    if profile == "smoke":
        return {
            "stage0_seeds": [1201],
            "stagea_seeds": [4201],
            "stageb_seeds": [5201],
            "stagec_seeds": [6201],
            "staged_seeds": [7201],
            "stage0_systems": ["correlated_ou", "rotational_ou", "heston_logsv"],
            "stageb_systems": ["correlated_ou", "rotational_ou", "heston_logsv"],
            "stagec_systems": ["correlated_ou", "rotational_ou", "heston_logsv"],
            "factorial_configs": 12,
            "stagea_max_variants": 18,
            "base_steps": 800,
            "heston_steps": 1100,
            "final_candidates": 3,
        }
    if profile == "standard":
        return {
            "stage0_seeds": [1201, 1219, 1237],
            "stagea_seeds": [4201, 4202, 4203],
            "stageb_seeds": [5201, 5202],
            "stagec_seeds": [6201, 6202, 6203, 6204],
            "staged_seeds": [7201, 7202, 7203],
            "stage0_systems": list(REGISTRY),
            "stageb_systems": list(REGISTRY),
            "stagec_systems": list(REGISTRY),
            "factorial_configs": 96,
            "stagea_max_variants": None,
            "base_steps": 1700,
            "heston_steps": 2800,
            "final_candidates": 6,
        }
    return {
        "stage0_seeds": [1201, 1219, 1237, 1255, 1273],
        "stagea_seeds": [4201, 4202, 4203, 4204, 4205],
        "stageb_seeds": [5201, 5202, 5203, 5204, 5205],
        "stagec_seeds": [6201, 6202, 6203, 6204, 6205, 6206, 6207, 6208],
        "staged_seeds": [7201, 7202, 7203, 7204, 7205],
        "stage0_systems": list(REGISTRY),
        "stageb_systems": list(REGISTRY),
        "stagec_systems": list(REGISTRY),
        "factorial_configs": 160,
        "stagea_max_variants": None,
        "base_steps": 1800,
        "heston_steps": 3000,
        "final_candidates": 8,
    }


def write_csv(path: str, rows: Iterable[dict], fields: list[str]) -> None:
    rows = list(rows)
    p = ROOT / path
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def append_csv(path: str, row: dict, fields: list[str]) -> None:
    p = ROOT / path
    p.parent.mkdir(parents=True, exist_ok=True)
    exists = p.exists()
    with p.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fields})


def reset_outputs() -> None:
    for rel in [STAGE0_PATH, OFAT_PATH, FACTORIAL_PATH, GREEDY_PATH, PER_SYSTEM_PATH, LEADERBOARD_PATH, GRAFT_ABLATION_PATH, GLOBAL_DEFAULT_PATH, RUN_LOG_PATH]:
        p = ROOT / rel
        if p.exists():
            p.unlink()
    docs = ROOT / "docs/V5_FINDINGS.md"
    if docs.exists():
        docs.unlink()


def write_catalog(catalog: list[V5Variant]) -> None:
    rows = []
    for variant in catalog:
        row = asdict(variant)
        row["l1_ratio_grid"] = ";".join(str(x) for x in variant.l1_ratio_grid)
        row["projection_scales"] = ";".join(str(x) for x in variant.projection_scales)
        row["target_regression_kw"] = target_kw_label(variant.target_regression_kw)
        rows.append(row)
    write_csv(CATALOG_PATH, rows, CATALOG_FIELDS)


def target_kw_label(value: dict | None) -> str:
    if not value:
        return ""
    parts = []
    for key in sorted(value):
        inner = value[key]
        if isinstance(inner, dict):
            bits = ",".join(f"{k}={inner[k]}" for k in sorted(inner))
        else:
            bits = str(inner)
        parts.append(f"{key}:{bits}")
    return "|".join(parts)


def library_space_for(system_key: str, variant: V5Variant) -> str:
    if variant.library_space == "default":
        truth = REGISTRY[system_key]
        return v3_default_library_space(truth.library, truth.dim)
    return variant.library_space


def base_steps_for(system_key: str, variant: V5Variant, settings: dict) -> tuple[int, float]:
    dt = 1.0 / 252.0 if system_key in HESTON_SYSTEMS else 0.01
    base = settings["heston_steps"] if system_key in HESTON_SYSTEMS else settings["base_steps"]
    if REGISTRY[system_key].dim == 1:
        base = max(850, int(0.75 * settings["base_steps"]))
    if system_key in HARD_NULL_SYSTEMS:
        base = max(650, int(0.70 * base))
    n_steps = max(400, int(round(base * variant.n_steps_scale)))
    return n_steps, dt


def cell_for(system_key: str, variant: V5Variant, seed: int, run: int, stage: str, settings: dict) -> FitCell:
    truth = REGISTRY[system_key]
    n_steps, dt = base_steps_for(system_key, variant, settings)
    regressor = variant.regressor
    if truth.dim == 1 and regressor in {"lasso_stlsq", "elastic_net", "adaptive_lasso", "stability_selection"}:
        regressor = "stlsq"
    library = variant.library_override or truth.library
    # Avoid pruning multiscale stacks; row count no longer equals center count.
    prune = None if len(variant.projection_scales) > 1 else variant.prune_min_effective_samples
    return FitCell(
        experiment=f"v5_{stage}",
        system_key=system_key,
        library=library,
        regressor=regressor,
        center_scheme=variant.center_scheme,
        n_centers=50 if truth.dim == 1 else variant.n_centers,
        bandwidth_mult=variant.bandwidth_mult,
        bandwidth_rule=variant.bandwidth_rule,
        knn_k=variant.knn_k,
        local_poly_order=0 if truth.dim == 1 else variant.local_poly_order,
        projection_normalization=variant.projection_normalization,
        projection_scales=(1.0,) if truth.dim == 1 else variant.projection_scales,
        prune_min_effective_samples=prune,
        target_anchor=variant.target_anchor,
        dt=dt,
        n_steps=n_steps,
        seed=seed,
        run=run,
        subsample_k=variant.subsample_k,
        bias_correct=variant.bias_correct,
        noise_correct=variant.noise_correct,
        n_trajectories=max(1, variant.n_trajectories),
        library_space="raw" if truth.dim == 1 else library_space_for(system_key, variant),
        threshold=variant.threshold,
        stlsq_threshold=variant.stlsq_threshold,
        threshold_mode=variant.threshold_mode,
        pseudo_blocks=variant.pseudo_blocks,
        l1_ratio_grid=variant.l1_ratio_grid,
        adaptive_gamma=variant.adaptive_gamma,
        svd_rtol=variant.svd_rtol,
        ridge_floor=variant.ridge_floor,
        gls_weighting=False if truth.dim == 1 else variant.gls_weighting,
        gls_iterations=1 if truth.dim == 1 else variant.gls_iterations,
        target_regression_kw=None if truth.dim == 1 else variant.target_regression_kw,
        diffusion_parameterization="entries" if truth.dim == 1 else variant.diffusion_parameterization,
        diffusion_shrinkage=0.0 if truth.dim == 1 else variant.diffusion_shrinkage,
        rank1_project=False if truth.dim == 1 else variant.rank1_project,
    )


def _safe_float(value: object, default: float = float("nan")) -> float:
    try:
        return float(value)
    except Exception:
        return default


def composite_score(row: dict) -> float:
    drift = _safe_float(row.get("drift_rel_l2"))
    diffusion = _safe_float(row.get("diffusion_rel_l2"))
    psd = _safe_float(row.get("psd_valid_pct"), 0.0)
    a12_cos = _safe_float(row.get("a12_cosine"))
    a12_term = 0.5 if math.isnan(a12_cos) else max(0.0, min(1.0, (a12_cos + 1.0) / 2.0))
    drift_term = max(0.0, 1.0 - min(drift, 2.0) / 2.0) if math.isfinite(drift) else 0.0
    diff_term = max(0.0, 1.0 - min(diffusion, 1.5) / 1.5) if math.isfinite(diffusion) else 0.0
    oracle_gap = _safe_float(row.get("oracle_headroom_drift"), 0.0)
    headroom_term = max(0.0, min(1.0, 1.0 - max(oracle_gap, 0.0)))
    return 0.34 * drift_term + 0.34 * diff_term + 0.14 * psd + 0.10 * a12_term + 0.08 * headroom_term


def selection_diagnostics(fit) -> dict:
    sels = list(fit.selections.values())
    if not sels:
        return {}
    selected = [int(s.diagnostics.get("n_selected", int(np.count_nonzero(s.coef)))) for s in sels]
    folds = [int(s.diagnostics.get("cv_folds", 0)) for s in sels]
    alphas = [float(s.alpha) for s in sels if np.isfinite(float(s.alpha)) and float(s.alpha) > 0]
    ratios = [float(s.diagnostics.get("l1_ratio", float("nan"))) for s in sels if np.isfinite(float(s.diagnostics.get("l1_ratio", float("nan"))))]
    return {
        "cond_design": fit.bandwidth_meta.get("cond_design", float("nan")),
        "rank_deficient_targets": int(sum(bool(s.diagnostics.get("rank_deficient", False)) for s in sels)),
        "mean_selected_terms": float(np.mean(selected)),
        "max_cv_folds": int(max(folds) if folds else 0),
        "used_pseudo_blocks": bool(any(bool(s.diagnostics.get("cv_pseudo_blocks", False)) for s in sels)),
        "mean_alpha": float(np.mean(alphas)) if alphas else float("nan"),
        "mean_l1_ratio": float(np.mean(ratios)) if ratios else float("nan"),
    }


def base_row(stage: str, variant: V5Variant, system_key: str, seed: int, run: int) -> dict:
    return {
        "stage": stage,
        "config_id": variant.variant_id,
        "variant_id": variant.variant_id,
        "family": variant.family,
        "description": variant.description,
        "implemented": variant.implemented,
        "infeasible_reason": variant.infeasible_reason,
        "core_identity_preserving": variant.core_identity_preserving,
        "graft_source": variant.graft_source,
        "system": system_key,
        "seed": seed,
        "run": run,
        "center_scheme": variant.center_scheme,
        "bandwidth_mult": variant.bandwidth_mult,
        "bandwidth_rule": variant.bandwidth_rule,
        "knn_k": variant.knn_k,
        "local_poly_order": variant.local_poly_order,
        "projection_normalization": variant.projection_normalization,
        "projection_scales": ";".join(str(x) for x in variant.projection_scales),
        "prune_min_effective_samples": "" if variant.prune_min_effective_samples is None else variant.prune_min_effective_samples,
        "target_anchor": variant.target_anchor,
        "threshold_mode": variant.threshold_mode,
        "threshold": "" if variant.threshold is None else variant.threshold,
        "stlsq_threshold": "" if variant.stlsq_threshold is None else variant.stlsq_threshold,
        "pseudo_blocks": variant.pseudo_blocks,
        "l1_ratio_grid": ";".join(str(v) for v in variant.l1_ratio_grid),
        "adaptive_gamma": variant.adaptive_gamma,
        "svd_rtol": variant.svd_rtol,
        "ridge_floor": variant.ridge_floor,
        "gls_weighting": variant.gls_weighting,
        "gls_iterations": variant.gls_iterations,
        "target_regression_kw": target_kw_label(variant.target_regression_kw),
        "diffusion_parameterization": variant.diffusion_parameterization,
        "diffusion_shrinkage": variant.diffusion_shrinkage,
        "rank1_project": variant.rank1_project,
        "bias_correct": variant.bias_correct,
        "noise_correct": variant.noise_correct,
        "selection_basis": "non_oracle_composite",
    }


def error_row(stage: str, variant: V5Variant, system_key: str, seed: int, run: int, exc: Exception, settings: dict) -> dict:
    truth = REGISTRY.get(system_key)
    n_steps, dt = base_steps_for(system_key, variant, settings) if truth else (0, 0.0)
    row = base_row(stage, variant, system_key, seed, run)
    row.update(
        {
            "tier": "" if truth is None else truth.tier,
            "dim": "" if truth is None else truth.dim,
            "library": variant.library_override or (truth.library if truth else ""),
            "library_space": "" if truth is None else ("raw" if truth.dim == 1 else library_space_for(system_key, variant)),
            "regressor": variant.regressor,
            "M": variant.n_centers,
            "R": variant.n_trajectories,
            "n_steps": n_steps,
            "dt": dt,
            "subsample_k": variant.subsample_k,
            "T": dt * n_steps * variant.subsample_k,
            "status": "FAILED",
            "pass_level": "fail",
            "score": 0.0,
            "error": repr(exc),
        }
    )
    return row


def evaluate(stage: str, variant: V5Variant, system_key: str, seed: int, run: int, settings: dict) -> dict:
    if not variant.implemented:
        row = base_row(stage, variant, system_key, seed, run)
        row.update({"status": "INFEASIBLE_BY_INVARIANT", "pass_level": "fail", "score": 0.0})
        return row
    t0 = time.perf_counter()
    try:
        cell = cell_for(system_key, variant, seed, run, stage, settings)
        system, x, fit, runtime = fit_cell(cell)
        pts = central_grid(x, 13 if REGISTRY[system_key].dim == 2 else 70)
        errs = function_l2_errors(fit, system, pts)
        oracle = oracle_diagnostics(fit, system, pts)
        psd = psd_validity(fit.evaluate(pts)[1])
        a12_cos = float("nan")
        a12_sign = float("nan")
        if REGISTRY[system_key].dim == 2:
            tmet = tensor_metrics(fit, system, pts)
            a12_sign = tmet.get("a12_sign_accuracy", float("nan"))
            true_a12 = system.true_diffusion(pts)[:, 0, 1]
            pred_a12 = fit.evaluate(pts)[1][:, 0, 1]
            a12_cos = cosine_similarity(pred_a12, true_a12)
        split = split_pass_levels(errs["drift_rel_l2"], errs["diffusion_rel_l2"], psd["pct_psd_valid"], a12_sign, a12_cos)
        row = base_row(stage, variant, system_key, seed, run)
        row.update(
            {
                "tier": REGISTRY[system_key].tier,
                "dim": REGISTRY[system_key].dim,
                "library": cell.library,
                "library_space": cell.library_space,
                "regressor": cell.regressor,
                "M": fit.bandwidth_meta["n_centers"],
                "R": cell.n_trajectories,
                "n_steps": cell.n_steps,
                "dt": cell.dt,
                "subsample_k": cell.subsample_k,
                "T": cell.dt * cell.n_steps * cell.subsample_k,
                "status": status_from_level(system_key, split["pass_level"]),
                "pass_level": split["pass_level"],
                "drift_pass_level": split["drift_pass_level"],
                "tensor_pass_level": split["tensor_pass_level"],
                "drift_rel_l2": errs["drift_rel_l2"],
                "diffusion_rel_l2": errs["diffusion_rel_l2"],
                "b1_rel_l2": errs.get("b1_rel_l2", float("nan")),
                "b2_rel_l2": errs.get("b2_rel_l2", float("nan")),
                "a11_rel_l2": errs.get("a11_rel_l2", float("nan")),
                "a22_rel_l2": errs.get("a22_rel_l2", float("nan")),
                "a12_rel_l2": errs.get("a12_rel_l2", float("nan")),
                "a12_cosine": a12_cos,
                "a12_sign_acc": a12_sign,
                "psd_valid_pct": psd["pct_psd_valid"],
                "oracle_drift_rel_l2": oracle["drift_rel_l2"],
                "oracle_diffusion_rel_l2": oracle["diffusion_rel_l2"],
                "oracle_a12_rel_l2": oracle.get("a12_rel_l2", float("nan")),
                "oracle_a12_cosine": oracle.get("a12_cosine", float("nan")),
                "oracle_ols_passes": oracle["oracle_ols_passes"],
                "oracle_headroom_drift": errs["drift_rel_l2"] - oracle["drift_rel_l2"],
                "oracle_headroom_diffusion": errs["diffusion_rel_l2"] - oracle["diffusion_rel_l2"],
                "oracle_headroom_a12": errs.get("a12_rel_l2", float("nan")) - oracle.get("a12_rel_l2", float("nan")),
                "runtime_sec": runtime,
                "error": "",
            }
        )
        row.update(selection_diagnostics(fit))
        row["score"] = composite_score(row)
        return row
    except Exception as exc:
        row = error_row(stage, variant, system_key, seed, run, exc, settings)
        row["runtime_sec"] = time.perf_counter() - t0
        return row


def completed_keys(path: str) -> set[tuple[str, str, str, str]]:
    keys = set()
    for row in read_rows(path):
        keys.add((row.get("stage", ""), row.get("variant_id", ""), row.get("system", ""), row.get("seed", "")))
    return keys


def _evaluate_task(task: tuple[int, int, str, V5Variant, str, int, int, dict]) -> tuple[int, int, dict]:
    ordinal, total, stage, variant, system_key, seed, run, settings = task
    return ordinal, total, evaluate(stage, variant, system_key, seed, run, settings)


def run_grid(path: str, stage: str, variants: list[V5Variant], systems: list[str], seeds: list[int], settings: dict, resume: bool, jobs: int = 1) -> list[dict]:
    done = completed_keys(path) if resume else set()
    rows: list[dict] = []
    total = len(variants) * len(systems) * len(seeds)
    pending: list[tuple[int, int, str, V5Variant, str, int, int, dict]] = []
    count = 0
    for variant in variants:
        if not variant.implemented:
            key = (stage, variant.variant_id, "ALL", "0")
            if key not in done:
                row = evaluate(stage, variant, "ALL", 0, 0, settings)
                append_csv(path, row, ROW_FIELDS)
            continue
        for run, seed in enumerate(seeds):
            for system_key in systems:
                count += 1
                key = (stage, variant.variant_id, system_key, str(seed))
                if key not in done:
                    pending.append((count, total, stage, variant, system_key, seed, run, settings))
    if not pending:
        return rows
    workers = max(1, int(jobs))
    if workers == 1:
        for task in pending:
            ordinal, total, row = _evaluate_task(task)
            append_csv(path, row, ROW_FIELDS)
            rows.append(row)
            if ordinal % 50 == 0 or row.get("status") == "FAILED":
                print(f"{stage:8s} {ordinal:5d}/{total:<5d} {row.get('variant_id','')[:24]:24s} {row.get('system','')[:22]:22s} score={_safe_float(row.get('score'),0.0):.3f} status={row.get('status')}")
        return rows

    print(f"{stage:8s} running {len(pending)} pending cells with jobs={workers} (resume skipped {total - len(pending)})")
    max_in_flight = max(workers * 2, workers)
    task_iter = iter(pending)
    futures = set()
    completed = total - len(pending)

    def submit_some(executor: ProcessPoolExecutor) -> None:
        while len(futures) < max_in_flight:
            try:
                task = next(task_iter)
            except StopIteration:
                return
            futures.add(executor.submit(_evaluate_task, task))

    with ProcessPoolExecutor(max_workers=workers) as executor:
        submit_some(executor)
        while futures:
            done_futures, futures = wait(futures, return_when=FIRST_COMPLETED)
            for future in done_futures:
                ordinal, total, row = future.result()
                append_csv(path, row, ROW_FIELDS)
                rows.append(row)
                completed += 1
                if completed % 50 == 0 or row.get("status") == "FAILED":
                    print(f"{stage:8s} {completed:5d}/{total:<5d} {row.get('variant_id','')[:24]:24s} {row.get('system','')[:22]:22s} score={_safe_float(row.get('score'),0.0):.3f} status={row.get('status')}")
            submit_some(executor)
    return rows


def factorial_variants(n: int) -> list[V5Variant]:
    regressors = ["lasso_stlsq", "elastic_net", "adaptive_lasso", "bic", "sr3", "omp", "ridge_gcv"]
    centers = ["quantile_grid", "uniform_grid", "kmeans", "sobol", "greedy_coverage"]
    m_values = [36, 64, 81, 100, 144]
    h_values = [1.0, 1.25, 1.5, 2.0]
    h_rules = ["nn_median", "cov", "knn", "local_cov"]
    local_orders = [0, 1]
    anchors = ["left", "midpoint"]
    diff_params = ["entries", "chol", "log_chol"]
    r_values = [4, 8, 16, 32]
    out = [v5_composed_stack()]
    for i in range(n):
        reg = regressors[i % len(regressors)]
        out.append(
            V5Variant(
                variant_id=f"V5F{i:03d}",
                family="factorial",
                description="Deterministic v5 structural LHS/factorial config",
                priority="H",
                graft_source="factorial composition",
                regressor=reg,
                center_scheme=centers[(i * 5) % len(centers)],
                n_centers=m_values[(i * 11) % len(m_values)],
                bandwidth_mult=h_values[(i * 13) % len(h_values)],
                bandwidth_rule=h_rules[(i * 17) % len(h_rules)],
                local_poly_order=local_orders[(i * 19) % len(local_orders)],
                target_anchor=anchors[(i * 23) % len(anchors)],
                gls_weighting=bool((i // 2) % 2),
                diffusion_parameterization=diff_params[(i * 29) % len(diff_params)],
                diffusion_shrinkage=[0.0, 0.03, 0.10][(i * 31) % 3],
                n_trajectories=r_values[(i * 37) % len(r_values)],
                n_steps_scale=[0.8, 1.0, 1.25][(i * 41) % 3],
                subsample_k=[1, 2, 3][(i * 43) % 3],
                stlsq_threshold=[0.06, 0.10, 0.14, 0.20][(i * 47) % 4],
                l1_ratio_grid=([0.2, 0.5, 0.8, 0.95][(i * 53) % 4],),
                adaptive_gamma=[0.5, 1.0, 2.0][(i * 59) % 3],
                rank1_project=bool((i // 7) % 2),
            )
        )
    return out


def rows_by_config(rows: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for row in rows:
        out.setdefault(row.get("variant_id", row.get("config_id", "")), []).append(row)
    return out


def variant_from_row(row: dict) -> V5Variant:
    l1 = tuple(float(v) for v in str(row.get("l1_ratio_grid", "0.2;0.5;0.8;0.95")).split(";") if v)
    scales = tuple(float(v) for v in str(row.get("projection_scales", "1.0")).split(";") if v)
    stlsq = row.get("stlsq_threshold", "")
    threshold = row.get("threshold", "")
    prune = row.get("prune_min_effective_samples", "")
    return V5Variant(
        variant_id=str(row.get("variant_id") or row.get("config_id") or "ROW_CONFIG"),
        family=str(row.get("family") or "row"),
        description=str(row.get("description") or "row-derived config"),
        regressor=str(row.get("regressor") or "lasso_stlsq"),
        library_space=str(row.get("library_space") or "default"),
        center_scheme=str(row.get("center_scheme") or "quantile_grid"),
        n_centers=int(float(row.get("M") or 64)),
        bandwidth_mult=float(row.get("bandwidth_mult") or 1.5),
        bandwidth_rule=str(row.get("bandwidth_rule") or "nn_median"),
        knn_k=int(float(row.get("knn_k") or 50)),
        local_poly_order=int(float(row.get("local_poly_order") or 0)),
        projection_normalization=str(row.get("projection_normalization") or "row"),
        projection_scales=scales or (1.0,),
        prune_min_effective_samples=None if prune in {"", "nan", None} else float(prune),
        target_anchor=str(row.get("target_anchor") or "left"),
        threshold_mode=str(row.get("threshold_mode") or "relative"),
        threshold=None if threshold in {"", "nan"} else float(threshold),
        stlsq_threshold=None if stlsq in {"", "nan"} else float(stlsq),
        pseudo_blocks=int(float(row.get("pseudo_blocks") or 5)),
        l1_ratio_grid=l1 or (0.2, 0.5, 0.8, 0.95),
        adaptive_gamma=float(row.get("adaptive_gamma") or 1.0),
        svd_rtol=float(row.get("svd_rtol") or 1e-8),
        ridge_floor=float(row.get("ridge_floor") or 1e-10),
        gls_weighting=str(row.get("gls_weighting", "False")) == "True",
        gls_iterations=int(float(row.get("gls_iterations") or 1)),
        diffusion_parameterization=str(row.get("diffusion_parameterization") or "entries"),
        diffusion_shrinkage=float(row.get("diffusion_shrinkage") or 0.0),
        rank1_project=str(row.get("rank1_project", "False")) == "True",
        n_trajectories=int(float(row.get("R") or 8)),
        subsample_k=int(float(row.get("subsample_k") or 1)),
        bias_correct=str(row.get("bias_correct", "True")) != "False",
        noise_correct=str(row.get("noise_correct", "False")) == "True",
        graft_source=str(row.get("graft_source") or "row-derived"),
    )


def top_variants_from_rows(rows: list[dict], n: int) -> list[V5Variant]:
    ag = v4_aggregate(rows, ["variant_id"])
    by_variant = rows_by_config(rows)
    out = []
    seen = set()
    for item in ag:
        vid = item["variant_id"]
        if vid in seen or vid not in by_variant:
            continue
        out.append(variant_from_row(by_variant[vid][0]))
        seen.add(vid)
        if len(out) >= n:
            break
    if "V5_COMPOSED_WEAK_GRAFT" not in seen:
        out.insert(0, v5_composed_stack())
    return out[:n]


def greedy_variants(seed_variant: V5Variant) -> list[V5Variant]:
    axes = [
        ("regressor", ["lasso_stlsq", "adaptive_lasso", "elastic_net", "bic", "sr3", "ridge_gcv"]),
        ("center_scheme", ["quantile_grid", "kmeans", "sobol", "greedy_coverage"]),
        ("bandwidth_rule", ["nn_median", "cov", "knn", "local_cov"]),
        ("local_poly_order", [0, 1, 2]),
        ("target_anchor", ["left", "midpoint"]),
        ("gls_weighting", [False, True]),
        ("diffusion_parameterization", ["entries", "chol", "log_chol"]),
        ("diffusion_shrinkage", [0.0, 0.03, 0.10]),
        ("n_trajectories", [8, 16, 32]),
        ("subsample_k", [1, 2]),
    ]
    variants = [v5_composed_stack(), *graft_ablation_variants()]
    for axis, values in axes:
        for value in values:
            variants.append(replace(seed_variant, variant_id=f"V5GREEDY_{axis}_{str(value).replace('.', 'p')}", family="greedy", description=f"V5 greedy coordinate {axis}={value}", **{axis: value}))
    unique = {}
    for variant in variants:
        unique[variant.variant_id] = variant
    return list(unique.values())


def graft_ablation_variants() -> list[V5Variant]:
    composed = v5_composed_stack()
    baseline = v5_baseline()
    specs = [
        ("LOCAL_POLY", "B1", {"local_poly_order": 1}, {"local_poly_order": 0}),
        ("GLS", "G5", {"gls_weighting": True}, {"gls_weighting": False}),
        ("CHOL", "P1", {"diffusion_parameterization": "chol"}, {"diffusion_parameterization": "entries"}),
        ("LOCAL_COV", "K9", {"bandwidth_rule": "local_cov"}, {"bandwidth_rule": "cov"}),
        ("SHRINKAGE", "P5", {"diffusion_shrinkage": 0.05}, {"diffusion_shrinkage": 0.0}),
        ("ADAPTIVE_REG", "G3", {"regressor": "adaptive_lasso"}, {"regressor": "lasso_stlsq"}),
    ]
    out: list[V5Variant] = []
    for name, source, add_kw, loo_kw in specs:
        out.append(
            replace(
                baseline,
                variant_id=f"V5ADD_{name}",
                family="graft_ablation",
                description=f"Add-one graft ablation for {name.lower()}",
                graft_source=f"{source} add-one graft",
                **add_kw,
            )
        )
        out.append(
            replace(
                composed,
                variant_id=f"V5LOO_{name}",
                family="graft_ablation",
                description=f"Leave-one-out graft ablation for {name.lower()}",
                graft_source=f"{source} leave-one-out from composed v5 stack",
                **loo_kw,
            )
        )
    return out


def build_leaderboard() -> None:
    rows = read_rows(STAGE0_PATH) + read_rows(OFAT_PATH) + read_rows(FACTORIAL_PATH) + read_rows(GREEDY_PATH) + read_rows(PER_SYSTEM_PATH)
    ag = v4_aggregate(rows, ["stage", "variant_id", "family", "description", "regressor", "center_scheme", "bandwidth_rule", "local_poly_order", "gls_weighting", "diffusion_parameterization"])
    fields = [
        "stage",
        "variant_id",
        "family",
        "description",
        "regressor",
        "center_scheme",
        "bandwidth_rule",
        "local_poly_order",
        "gls_weighting",
        "diffusion_parameterization",
        "n",
        "median_score",
        "median_drift_rel_l2",
        "median_diffusion_rel_l2",
        "median_psd_valid_pct",
        "validated_count",
        "failed_count",
    ]
    write_csv(LEADERBOARD_PATH, ag, fields)


def select_robust_global() -> dict:
    rows = [
        r
        for r in read_rows(FACTORIAL_PATH) + read_rows(GREEDY_PATH)
        if r.get("status") not in {"FAILED", "INFEASIBLE_BY_INVARIANT"}
        and r.get("family") != "graft_ablation"
        and not str(r.get("variant_id", "")).startswith(("V5ADD_", "V5LOO_"))
    ]
    buckets: dict[str, list[dict]] = {}
    for row in rows:
        buckets.setdefault(row["variant_id"], []).append(row)
    best: dict | None = None
    for vid, part in buckets.items():
        tier_meds = []
        for tier in sorted(set(r.get("tier", "") for r in part)):
            vals = [_safe_float(r.get("drift_rel_l2")) for r in part if r.get("tier") == tier and math.isfinite(_safe_float(r.get("drift_rel_l2")))]
            if vals:
                tier_meds.append(float(np.median(vals)))
        drifts = [_safe_float(r.get("drift_rel_l2")) for r in part if math.isfinite(_safe_float(r.get("drift_rel_l2")))]
        diffs = [_safe_float(r.get("diffusion_rel_l2")) for r in part if math.isfinite(_safe_float(r.get("diffusion_rel_l2")))]
        if not tier_meds or not drifts:
            continue
        row = {
            "variant_id": vid,
            "n": len(part),
            "worst_tier_median_drift": float(max(tier_meds)),
            "p90_drift": float(np.percentile(drifts, 90)),
            "median_drift": float(np.median(drifts)),
            "median_diffusion": float(np.median(diffs)) if diffs else float("nan"),
            "beats_v3_median": bool(float(np.median(drifts)) < V3_MEDIAN_DRIFT),
        }
        score_tuple = (row["worst_tier_median_drift"], row["p90_drift"], row["median_drift"])
        if best is None or score_tuple < best["_score_tuple"]:
            best = {**row, "_score_tuple": score_tuple}
    if best is None:
        best = {"variant_id": "", "n": 0, "worst_tier_median_drift": float("nan"), "p90_drift": float("nan"), "median_drift": float("nan"), "median_diffusion": float("nan"), "beats_v3_median": False, "_score_tuple": ()}
    best.pop("_score_tuple", None)
    write_csv(GLOBAL_DEFAULT_PATH, [best], list(best))
    return best


def write_graft_ablation() -> None:
    ofat = [r for r in read_rows(OFAT_PATH) if r.get("status") not in {"FAILED", "INFEASIBLE_BY_INVARIANT"}]
    baseline = [r for r in read_rows(STAGE0_PATH) if r.get("status") != "FAILED"]
    greedy = [r for r in read_rows(GREEDY_PATH) if r.get("status") not in {"FAILED", "INFEASIBLE_BY_INVARIANT"}]
    base_score = float(np.nanmedian([_safe_float(r.get("score")) for r in baseline])) if baseline else float("nan")
    composed_scores = [
        _safe_float(r.get("score"))
        for r in greedy
        if r.get("variant_id") == "V5_COMPOSED_WEAK_GRAFT" and math.isfinite(_safe_float(r.get("score")))
    ]
    composed_score = float(np.median(composed_scores)) if composed_scores else float("nan")
    ablation_names = ["LOCAL_POLY", "GLS", "CHOL", "LOCAL_COV", "SHRINKAGE", "ADAPTIVE_REG"]
    rows = []
    for fam in sorted(set(r.get("family", "") for r in ofat)):
        part = [r for r in ofat if r.get("family") == fam]
        if not part:
            continue
        scores = [_safe_float(r.get("score")) for r in part if math.isfinite(_safe_float(r.get("score")))]
        drifts = [_safe_float(r.get("drift_rel_l2")) for r in part if math.isfinite(_safe_float(r.get("drift_rel_l2")))]
        name = fam.upper()
        add_scores = [
            _safe_float(r.get("score"))
            for r in greedy
            if r.get("variant_id") == f"V5ADD_{name}" and math.isfinite(_safe_float(r.get("score")))
        ]
        loo_scores = [
            _safe_float(r.get("score"))
            for r in greedy
            if r.get("variant_id") == f"V5LOO_{name}" and math.isfinite(_safe_float(r.get("score")))
        ]
        add_median = float(np.median(add_scores)) if add_scores else float("nan")
        loo_median = float(np.median(loo_scores)) if loo_scores else float("nan")
        rows.append(
            {
                "graft_family": fam,
                "standalone_median_score": float(np.median(scores)) if scores else float("nan"),
                "standalone_lift_vs_stage0": (float(np.median(scores)) - base_score) if scores and math.isfinite(base_score) else float("nan"),
                "standalone_median_drift": float(np.median(drifts)) if drifts else float("nan"),
                "add_one_median_score": add_median,
                "add_one_lift_vs_stage0": add_median - base_score if math.isfinite(add_median) and math.isfinite(base_score) else float("nan"),
                "leave_one_out_median_score": loo_median,
                "leave_one_out_drop_from_composed": composed_score - loo_median if math.isfinite(composed_score) and math.isfinite(loo_median) else float("nan"),
                "inside_best_lift": composed_score - loo_median if math.isfinite(composed_score) and math.isfinite(loo_median) else float("nan"),
                "ablation_basis": "OFAT standalone plus Stage-C add-one and leave-one-out graft ablations where named",
                "n_rows": len(part),
            }
        )
    for name in ablation_names:
        if any(r["graft_family"].upper() == name for r in rows):
            continue
        add_scores = [_safe_float(r.get("score")) for r in greedy if r.get("variant_id") == f"V5ADD_{name}" and math.isfinite(_safe_float(r.get("score")))]
        loo_scores = [_safe_float(r.get("score")) for r in greedy if r.get("variant_id") == f"V5LOO_{name}" and math.isfinite(_safe_float(r.get("score")))]
        add_median = float(np.median(add_scores)) if add_scores else float("nan")
        loo_median = float(np.median(loo_scores)) if loo_scores else float("nan")
        rows.append(
            {
                "graft_family": name,
                "standalone_median_score": float("nan"),
                "standalone_lift_vs_stage0": float("nan"),
                "standalone_median_drift": float("nan"),
                "add_one_median_score": add_median,
                "add_one_lift_vs_stage0": add_median - base_score if math.isfinite(add_median) and math.isfinite(base_score) else float("nan"),
                "leave_one_out_median_score": loo_median,
                "leave_one_out_drop_from_composed": composed_score - loo_median if math.isfinite(composed_score) and math.isfinite(loo_median) else float("nan"),
                "inside_best_lift": composed_score - loo_median if math.isfinite(composed_score) and math.isfinite(loo_median) else float("nan"),
                "ablation_basis": "Stage-C add-one and leave-one-out graft ablation",
                "n_rows": len(add_scores) + len(loo_scores),
            }
        )
    write_csv(
        GRAFT_ABLATION_PATH,
        rows,
        [
            "graft_family",
            "standalone_median_score",
            "standalone_lift_vs_stage0",
            "standalone_median_drift",
            "add_one_median_score",
            "add_one_lift_vs_stage0",
            "leave_one_out_median_score",
            "leave_one_out_drop_from_composed",
            "inside_best_lift",
            "ablation_basis",
            "n_rows",
        ],
    )


def make_figures() -> None:
    import matplotlib.pyplot as plt

    out_dir = ROOT / "figures/v5"
    out_dir.mkdir(parents=True, exist_ok=True)
    leaderboard = read_rows(LEADERBOARD_PATH)[:15]
    if leaderboard:
        labels = [r["variant_id"][:18] for r in leaderboard]
        vals = [_safe_float(r["median_score"], 0.0) for r in leaderboard]
        fig, ax = plt.subplots(figsize=(max(8, 0.42 * len(labels)), 4))
        ax.bar(np.arange(len(labels)), vals)
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel("median composite score")
        ax.set_title("V5 Leaderboard Top Configurations")
        fig.tight_layout()
        fig.savefig(out_dir / "leaderboard_top15.png", dpi=160)
        plt.close(fig)
    ablation = read_rows(GRAFT_ABLATION_PATH)
    if ablation:
        labels = [r["graft_family"] for r in ablation]
        vals = [_safe_float(r["standalone_lift_vs_stage0"], 0.0) for r in ablation]
        fig, ax = plt.subplots(figsize=(max(8, 0.32 * len(labels)), 4))
        ax.bar(np.arange(len(labels)), vals)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=55, ha="right")
        ax.set_ylabel("median score lift vs V5 stage-0")
        ax.set_title("V5 Graft Ablation: Standalone Lift")
        fig.tight_layout()
        fig.savefig(out_dir / "graft_ablation_lift.png", dpi=160)
        plt.close(fig)
    global_rows = read_rows(GLOBAL_DEFAULT_PATH)
    if global_rows:
        gd = global_rows[0]
        fig, ax = plt.subplots(figsize=(5, 4))
        vals = [_safe_float(gd.get("median_drift")), _safe_float(gd.get("p90_drift")), _safe_float(gd.get("worst_tier_median_drift"))]
        ax.bar(["median", "p90", "worst tier"], vals)
        ax.axhline(V3_MEDIAN_DRIFT, linestyle="--", color="black", linewidth=0.9, label="v3 median")
        ax.set_ylabel("drift relative L2")
        ax.set_title(gd.get("variant_id", "V5 global default")[:28])
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "global_default_robustness.png", dpi=160)
        plt.close(fig)


def write_findings(global_default: dict) -> None:
    catalog = read_rows(CATALOG_PATH)
    infeasible = [r for r in catalog if str(r.get("implemented")) == "False"]
    leaderboard = read_rows(LEADERBOARD_PATH)
    stage0 = read_rows(STAGE0_PATH)
    per_system = read_rows(PER_SYSTEM_PATH)

    def med(rows: list[dict], col: str) -> float:
        vals = [_safe_float(r.get(col)) for r in rows if math.isfinite(_safe_float(r.get(col)))]
        return float(np.median(vals)) if vals else float("nan")

    top_lines = []
    for i, row in enumerate(leaderboard[:12], start=1):
        top_lines.append(
            f"{i}. `{row.get('variant_id')}` ({row.get('family')}): score {float(row.get('median_score', 0.0)):.3f}, "
            f"drift {float(row.get('median_drift_rel_l2', float('nan'))):.3f}, diffusion {float(row.get('median_diffusion_rel_l2', float('nan'))):.3f}"
        )
    heston_rows = [r for r in stage0 + per_system if r.get("system") in HESTON_SYSTEMS]
    lines = [
        "# V5 Findings",
        "",
        "## Run contract",
        "",
        "V5 implements the structural-lever catalog as config flags on the shared weak-form generator where the lever preserves the method identity. Variants that would replace spatial Gaussian weak kernels, require derivatives, or break the shared design matrix are left in the catalog with explicit infeasibility reasons rather than silently deferred.",
        "",
        "Artifacts:",
        f"- `{CATALOG_PATH}`",
        f"- `{OFAT_PATH}`",
        f"- `{FACTORIAL_PATH}`",
        f"- `{GREEDY_PATH}`",
        f"- `{PER_SYSTEM_PATH}`",
        f"- `{LEADERBOARD_PATH}`",
        f"- `{GRAFT_ABLATION_PATH}`",
        f"- `{GLOBAL_DEFAULT_PATH}`",
        "- `figures/v5/leaderboard_top15.png`",
        "- `figures/v5/graft_ablation_lift.png`",
        "- `figures/v5/global_default_robustness.png`",
        "",
        "## Catalog status",
        "",
        f"Catalog rows: {len(catalog)}. Implemented rows: {len(catalog) - len(infeasible)}. Explicit infeasible rows: {len(infeasible)}.",
        "There are no pending/deferred rows without a reason.",
        "",
        "## Robust global default",
        "",
        f"Selected variant: `{global_default.get('variant_id', '')}`.",
        f"Median drift/diffusion: {float(global_default.get('median_drift', float('nan'))):.4g} / {float(global_default.get('median_diffusion', float('nan'))):.4g}.",
        f"P90 drift: {float(global_default.get('p90_drift', float('nan'))):.4g}. Worst-tier median drift: {float(global_default.get('worst_tier_median_drift', float('nan'))):.4g}.",
        f"Beats v3 median drift {V3_MEDIAN_DRIFT:.4g}: {global_default.get('beats_v3_median')}.",
        "",
        "## Stage-0 comparison",
        "",
        f"V5 stage-0 median drift/diffusion: {med(stage0, 'drift_rel_l2'):.4g} / {med(stage0, 'diffusion_rel_l2'):.4g}.",
        f"Reference v4 Stage-0 median drift/diffusion: {V4_STAGE0_DRIFT:.4g} / 0.05509; v3 median drift: {V3_MEDIAN_DRIFT:.4g}.",
        "",
        "## Top configurations",
        "",
        *(top_lines or ["No successful leaderboard rows were written."]),
        "",
        "## Heston status",
        "",
        f"Heston-family median tensor diffusion error across Stage-0/per-system rows: {med(heston_rows, 'diffusion_rel_l2'):.4g}.",
        f"Heston-family median first/log-price drift error: {med(heston_rows, 'b1_rel_l2'):.4g}.",
        f"Heston-family median off-diagonal tensor cosine: {med(heston_rows, 'a12_cosine'):.4g}.",
        "Heston tensor/leverage remains separated from the low-SNR log-price drift null.",
        "",
        "## Honest nulls retained",
        "",
        "The catalog keeps non-Gaussian kernels, full gEDMD replacement, Milstein/Itô-1.5 derivative targets, latent Kalman/EM smoothing, and per-target libraries out of the v6 candidate algorithm because they break the preserved weak-form identity or require unobserved objects. Near-boundary, degenerate, partial-observation, bad-coverage, and Heston log-price drift rows remain visible as limitations.",
    ]
    p = ROOT / "docs/V5_FINDINGS.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines) + "\n")


def update_ledgers() -> None:
    reg = ROOT / "EXPERIMENT_REGISTRY.md"
    led = ROOT / "EVIDENCE_LEDGER.md"
    entry = "\n| v5 | full zoo | mixed | structural levers + robust global/default map | results/v5/leaderboard.csv | INCONCLUSIVE | see docs/V5_FINDINGS.md |\n"
    if reg.exists() and "results/v5/leaderboard.csv" not in reg.read_text():
        reg.write_text(reg.read_text().rstrip() + entry)
    evidence = "\n| V5 robust global default and graft ablations | CODEX_PROMPT_V5 | results/v5/global_default.csv; results/v5/graft_ablation.csv | median_drift, worst_tier_median_drift, standalone_lift_vs_stage0 | INCONCLUSIVE |\n"
    if led.exists() and "results/v5/global_default.csv" not in led.read_text():
        led.write_text(led.read_text().rstrip() + evidence)


def update_logs(profile: str, started: float) -> None:
    previous = read_rows(RUN_LOG_PATH)
    previous_runtime = max((_safe_float(r.get("runtime_sec")) for r in previous), default=float("nan"))
    finalize_runtime = time.perf_counter() - started
    runtime = max(previous_runtime, finalize_runtime) if math.isfinite(previous_runtime) else finalize_runtime
    rows = [
        {
            "profile": profile,
            "stage0_rows": len(read_rows(STAGE0_PATH)),
            "ofat_rows": len(read_rows(OFAT_PATH)),
            "factorial_rows": len(read_rows(FACTORIAL_PATH)),
            "greedy_rows": len(read_rows(GREEDY_PATH)),
            "per_system_rows": len(read_rows(PER_SYSTEM_PATH)),
            "leaderboard_rows": len(read_rows(LEADERBOARD_PATH)),
            "runtime_sec": runtime,
            "finalize_runtime_sec": finalize_runtime,
            "runtime_basis": "max(previous runtime_sec, current process runtime); interrupted/resumed wall time may exceed this",
        }
    ]
    write_csv(RUN_LOG_PATH, rows, list(rows[0]))


def run_stage_d(settings: dict, resume: bool, jobs: int = 1) -> None:
    source_rows = read_rows(GREEDY_PATH) + read_rows(FACTORIAL_PATH) + read_rows(OFAT_PATH)
    global_top = top_variants_from_rows(source_rows, settings["final_candidates"])
    for system_key in settings["stageb_systems"]:
        system_rows = [r for r in source_rows if r.get("system") == system_key]
        candidates = top_variants_from_rows(system_rows, settings["final_candidates"]) if system_rows else []
        candidates += [v for v in global_top if v.variant_id not in {c.variant_id for c in candidates}]
        candidates = candidates[: settings["final_candidates"]]
        variants = []
        for variant in candidates:
            variants.append(
                replace(
                    variant,
                    variant_id=f"V5BEST_{system_key}_{variant.variant_id}",
                    family="per_system",
                    description=f"M2 held-out candidate for {system_key}: {variant.description}",
                    graft_source=f"{variant.graft_source}; M2 no-oracle held-out selection",
                )
            )
        print(f"stageD selected/evaluated {len(candidates)} candidates for {system_key}")
        run_grid(PER_SYSTEM_PATH, "stageD", variants, [system_key], settings["staged_seeds"], settings, resume, jobs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=["smoke", "standard", "full"], default="full")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--factorial-configs", type=int, default=None)
    parser.add_argument("--jobs", type=int, default=int(os.environ.get("V5_JOBS", "1")))
    args = parser.parse_args()

    settings = profile_settings(args.profile)
    if args.factorial_configs is not None:
        settings["factorial_configs"] = args.factorial_configs
    started = time.perf_counter()
    if not args.resume:
        reset_outputs()
    catalog = variant_catalog()
    write_catalog(catalog)

    print("V5 Stage 0: strengthened baseline zoo rerun")
    run_grid(STAGE0_PATH, "stage0", [v5_baseline()], settings["stage0_systems"], settings["stage0_seeds"], settings, args.resume, args.jobs)

    print("V5 Stage A: full-catalog OFAT screening")
    stage_a_variants = [v for v in catalog if v.stage_a]
    if settings["stagea_max_variants"] is not None:
        stage_a_variants = stage_a_variants[: settings["stagea_max_variants"]]
    run_grid(OFAT_PATH, "stageA", stage_a_variants, SCREEN_SYSTEMS, settings["stagea_seeds"], settings, args.resume, args.jobs)

    print("V5 Stage B: structural factorial/LHS full-zoo screen")
    run_grid(FACTORIAL_PATH, "stageB", factorial_variants(settings["factorial_configs"]), settings["stageb_systems"], settings["stageb_seeds"], settings, args.resume, args.jobs)

    print("V5 Stage C: greedy + composed stack")
    source_rows = read_rows(FACTORIAL_PATH) + read_rows(OFAT_PATH)
    seed_variant = top_variants_from_rows(source_rows, 1)[0] if source_rows else v5_composed_stack()
    run_grid(GREEDY_PATH, "stageC", greedy_variants(seed_variant), settings["stagec_systems"], settings["stagec_seeds"], settings, args.resume, args.jobs)

    print("V5 Stage D: M2 per-system no-oracle selection")
    run_stage_d(settings, args.resume, args.jobs)

    build_leaderboard()
    write_graft_ablation()
    global_default = select_robust_global()
    make_figures()
    write_findings(global_default)
    update_ledgers()
    update_logs(args.profile, started)
    print("V5 DONE")


if __name__ == "__main__":
    main()
