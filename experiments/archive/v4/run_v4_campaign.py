from __future__ import annotations

import argparse
import os
import csv
import math
import statistics
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable

import numpy as np

from experiments.benchmarks._utils import (
    FitCell,
    fit_cell,
    oracle_diagnostics,
    split_pass_levels,
    status_from_level,
    v3_default_library_space,
    v3_default_regressor,
)
from experiments.common import ROOT
from sde2d.metrics import a12_sign_accuracy, central_grid, cosine_similarity, function_l2_errors, psd_validity, relative_l2, tensor_metrics
from sde2d.systems import REGISTRY


STAGE0_PATH = "results/v4/stage0_baseline.csv"
OFAT_PATH = "results/v4/ofat_screening.csv"
FACTORIAL_PATH = "results/v4/factorial.csv"
GREEDY_PATH = "results/v4/greedy.csv"
PER_SYSTEM_PATH = "results/v4/per_system_best.csv"
LEADERBOARD_PATH = "results/v4/leaderboard.csv"
CATALOG_PATH = "results/v4/variant_catalog.csv"
RUN_LOG_PATH = "results/v4/run_log.csv"

SCREEN_SYSTEMS = ["correlated_ou", "coupled_ou", "rotational_ou", "gradient_potential", "nondiag_cholesky", "heston_logsv"]
HARD_NULL_SYSTEMS = {"partial_observation", "bad_coverage", "too_large_dt"}
HESTON_SYSTEMS = {"heston_logsv", "heston_sv", "cir_pair", "near_boundary_heston"}


ROW_FIELDS = [
    "stage",
    "config_id",
    "variant_id",
    "family",
    "description",
    "implemented",
    "deferred_reason",
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
    "threshold_mode",
    "threshold",
    "stlsq_threshold",
    "pseudo_blocks",
    "l1_ratio_grid",
    "adaptive_gamma",
    "svd_rtol",
    "ridge_floor",
    "R",
    "n_steps",
    "dt",
    "subsample_k",
    "T",
    "bias_correct",
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
    "stage_a",
    "description",
    "deferred_reason",
    "regressor",
    "library_space",
    "center_scheme",
    "n_centers",
    "bandwidth_mult",
    "bandwidth_rule",
    "threshold_mode",
    "threshold",
    "stlsq_threshold",
    "pseudo_blocks",
    "l1_ratio_grid",
    "adaptive_gamma",
    "svd_rtol",
    "ridge_floor",
    "n_trajectories",
    "n_steps_scale",
    "subsample_k",
    "bias_correct",
]


@dataclass(frozen=True)
class V4Variant:
    variant_id: str
    family: str
    description: str
    priority: str = "M"
    implemented: bool = True
    stage_a: bool = True
    deferred_reason: str = ""
    regressor: str = "lasso_stlsq"
    library_space: str = "default"
    center_scheme: str = "quantile_grid"
    n_centers: int = 64
    bandwidth_mult: float = 1.5
    bandwidth_rule: str = "nn_median"
    threshold_mode: str = "relative"
    threshold: float | None = None
    stlsq_threshold: float | None = 0.10
    pseudo_blocks: int = 5
    l1_ratio_grid: tuple[float, ...] = (0.2, 0.5, 0.8, 0.95)
    adaptive_gamma: float = 1.0
    svd_rtol: float = 1e-8
    ridge_floor: float = 1e-10
    n_trajectories: int = 4
    n_steps_scale: float = 1.0
    subsample_k: int = 1
    bias_correct: bool = True
    library_override: str | None = None


def v4_baseline() -> V4Variant:
    return V4Variant(
        "V4_BASELINE",
        "stage0",
        "z-space + LassoCV debias STLSQ + pseudo-block CV + bias-corrected diffusion",
        priority="H",
        stlsq_threshold=0.10,
        pseudo_blocks=5,
    )


def _v(variant_id: str, family: str, description: str, **kw: object) -> V4Variant:
    return replace(v4_baseline(), variant_id=variant_id, family=family, description=description, **kw)


def implemented_variants() -> list[V4Variant]:
    variants: list[V4Variant] = [v4_baseline()]
    variants += [
        _v("G1_ENET_L1_020", "G1", "Elastic-net selector, l1_ratio=0.20", regressor="elastic_net", l1_ratio_grid=(0.2,), stlsq_threshold=0.14),
        _v("G1_ENET_L1_050", "G1", "Elastic-net selector, l1_ratio=0.50", regressor="elastic_net", l1_ratio_grid=(0.5,), stlsq_threshold=0.14),
        _v("G1_ENET_L1_080", "G1", "Elastic-net selector, l1_ratio=0.80", regressor="elastic_net", l1_ratio_grid=(0.8,), stlsq_threshold=0.14),
        _v("G1_ENET_L1_095", "G1", "Elastic-net selector, l1_ratio=0.95", regressor="elastic_net", l1_ratio_grid=(0.95,), stlsq_threshold=0.14),
        _v("G2_ADAPT_G050", "G2", "Adaptive LASSO selector, gamma=0.5", regressor="adaptive_lasso", adaptive_gamma=0.5, stlsq_threshold=0.12),
        _v("G2_ADAPT_G100", "G2", "Adaptive LASSO selector, gamma=1.0", regressor="adaptive_lasso", adaptive_gamma=1.0, stlsq_threshold=0.12),
        _v("G2_ADAPT_G200", "G2", "Adaptive LASSO selector, gamma=2.0", regressor="adaptive_lasso", adaptive_gamma=2.0, stlsq_threshold=0.12),
        _v("G3_REL_T003", "G3", "Relative STLSQ threshold 0.03", stlsq_threshold=0.03),
        _v("G3_REL_T006", "G3", "Relative STLSQ threshold 0.06", stlsq_threshold=0.06),
        _v("G3_REL_T014", "G3", "Relative STLSQ threshold 0.14", stlsq_threshold=0.14),
        _v("G3_REL_T020", "G3", "Relative STLSQ threshold 0.20", stlsq_threshold=0.20),
        _v("G3_ABS_T001", "G3", "Absolute STLSQ threshold 0.001", threshold_mode="absolute", stlsq_threshold=0.001),
        _v("G3_ABS_T005", "G3", "Absolute STLSQ threshold 0.005", threshold_mode="absolute", stlsq_threshold=0.005),
        _v("G3_ABS_T010", "G3", "Absolute STLSQ threshold 0.010", threshold_mode="absolute", stlsq_threshold=0.010),
        _v("G4_BLOCKS_05", "G4", "Pseudo-block GroupKFold with 5 blocks", pseudo_blocks=5),
        _v("G4_BLOCKS_10", "G4", "Pseudo-block GroupKFold with 10 blocks", pseudo_blocks=10),
        _v("N1_SVD_1E10", "N1", "Truncated-SVD threshold solve, rtol=1e-10", regressor="svd_threshold", svd_rtol=1e-10, threshold=0.04, stlsq_threshold=None),
        _v("N1_SVD_1E08", "N1", "Truncated-SVD threshold solve, rtol=1e-8", regressor="svd_threshold", svd_rtol=1e-8, threshold=0.04, stlsq_threshold=None),
        _v("N1_SVD_1E06", "N1", "Truncated-SVD threshold solve, rtol=1e-6", regressor="svd_threshold", svd_rtol=1e-6, threshold=0.04, stlsq_threshold=None),
        _v("G14_RIDGE_1E12", "G14", "Ridge floor 1e-12 in debias solves", ridge_floor=1e-12),
        _v("G14_RIDGE_1E08", "G14", "Ridge floor 1e-8 in debias solves", ridge_floor=1e-8),
        _v("G14_RIDGE_1E06", "G14", "Ridge floor 1e-6 in debias solves", ridge_floor=1e-6),
        _v("K1_COV_BANDWIDTH", "K1", "Anisotropic covariance bandwidth matrix", bandwidth_rule="cov"),
        _v("K2_H100", "K2", "Bandwidth multiplier 1.00", bandwidth_mult=1.0),
        _v("K2_H125", "K2", "Bandwidth multiplier 1.25", bandwidth_mult=1.25),
        _v("K2_H200", "K2", "Bandwidth multiplier 2.00", bandwidth_mult=2.0),
        _v("K2_PAIRWISE", "K2", "Pairwise-median bandwidth rule", bandwidth_rule="pairwise_median"),
        _v("K3_KMEANS", "K3", "K-means kernel centers", center_scheme="kmeans"),
        _v("K3_UNIFORM", "K3", "Uniform-grid kernel centers", center_scheme="uniform_grid"),
        _v("K4_M036", "K4", "36 kernel centers", n_centers=36),
        _v("K4_M081", "K4", "81 kernel centers", n_centers=81),
        _v("K4_M100", "K4", "100 kernel centers", n_centers=100),
        _v("K4_M144", "K4", "144 kernel centers", n_centers=144),
        _v("L1_RAW_SPACE", "L1", "Raw-coordinate library space ablation", library_space="raw"),
        _v("L2_Z_SPACE", "L2", "Standardized-coordinate library space", library_space="z"),
        _v("N6_R001", "N6", "Single trajectory with pseudo-block CV", n_trajectories=1),
        _v("N6_R004", "N6", "Four pooled trajectories", n_trajectories=4),
        _v("N6_R008", "N6", "Eight pooled trajectories", n_trajectories=8),
        _v("N6_R016", "N6", "Sixteen pooled trajectories", n_trajectories=16, n_steps_scale=0.65),
        _v("D5_LONG_T", "D5", "Longer trajectory budget", n_steps_scale=1.75),
        _v("D5_WIDE_R", "D5", "More independent paths, shorter per-path length", n_trajectories=12, n_steps_scale=0.75),
        _v("V4_LAG2", "V4", "Two-step lag target via subsampling", subsample_k=2),
        _v("V4_LAG4", "V4", "Four-step lag target via subsampling", subsample_k=4),
        _v("B2_NO_BIAS_CORR", "B2", "No finite-step drift-square diffusion correction ablation", bias_correct=False),
        _v("R0_STLSQ", "regressor", "Plain STLSQ selector baseline", regressor="stlsq", threshold=0.02, stlsq_threshold=None),
        _v("R0_RIDGE", "regressor", "Ridge-threshold selector baseline", regressor="ridge_threshold", threshold=0.08, stlsq_threshold=None),
    ]
    return variants


def deferred_variants() -> list[V4Variant]:
    items = [
        ("K5", "boundary-aware kernel reflection/trim", "needs boundary-class API"),
        ("K6", "density-equalized kernels", "requires sampler/mesh extension"),
        ("K7", "kernel shape sweep Epanechnikov/compact", "Gaussian-only invariant retained"),
        ("K8", "adaptive local bandwidth", "requires per-center bandwidth metadata"),
        ("K9", "Mahalanobis local covariance per center", "global covariance is implemented; local covariance is next"),
        ("K10", "under-supported kernel drop/merge", "requires row-support thresholding in generator"),
        ("K11", "projection-normalization alternatives", "row-normalized weak moments remain default"),
        ("K12", "state-dependent density weighting", "no unbiasedness proof in current spec"),
        ("K13", "center CV by validation residual", "runner has config CV, not center learning"),
        ("K14", "multi-resolution kernel stack", "needs block-stacked projection design"),
        ("L3", "Legendre/Chebyshev tensor basis", "needs coefficient back-transform extension"),
        ("L4", "Hermite basis for OU-like stationary laws", "needs stationary-weighted library module"),
        ("L5", "feature pruning by VIF/correlation clustering", "requires library-selection module"),
        ("L6", "empirical Gram-Schmidt orthonormal library", "needs reversible coefficient map"),
        ("L7", "RBF feature augmentation", "requires mixed library serialization"),
        ("L8", "general Fourier dictionary beyond preset G", "preset G exists; general grid pending"),
        ("L9", "hierarchical degree escalation by BIC", "needs residual loop"),
        ("L10", "group LASSO by degree/block", "requires group-penalty dependency"),
        ("L11", "rational/sqrt variance dictionary", "needs domain-safe feature definitions"),
        ("L12", "SVD-whitened feature map", "solve-level SVD exists; feature-level map pending"),
        ("G5", "GLS heteroscedastic row weighting", "needs pass-1 tensor variance weights"),
        ("G6", "stability selection bootstrap support", "needs expensive bootstrap wrapper"),
        ("G7", "AIC/BIC/eBIC support selection", "needs support-path enumeration"),
        ("G8", "SR3 sparse regression", "requires new optimizer"),
        ("G9", "MIO/L0 best subset", "optional dependency not present"),
        ("G10", "OMP/forward-stagewise", "not yet linked to debias/STLSQ path"),
        ("G11", "total least squares EIV regression", "requires design-noise model"),
        ("G12", "Huber/IRLS robust loss", "requires robust objective path"),
        ("G13", "multitask shared support", "requires joint target solve"),
        ("V1", "midpoint-state drift target", "requires target-anchor API"),
        ("V2", "midpoint/endpoint-average diffusion target", "requires target-anchor API"),
        ("V3", "Stratonovich-corrected midpoint drift", "requires pass-1 tensor derivatives"),
        ("V5", "multi-step diffusion QV", "subsample lag exists; true multi-step QV pending"),
        ("V6", "Richardson extrapolation in dt", "requires paired native/subsample estimator blend"),
        ("V7", "Milstein-aware diffusion target", "needs derivative terms"),
        ("V8", "gEDMD operator-regression targets", "requires operator estimator path"),
        ("V9", "hybrid increment/operator targets", "depends on V8"),
        ("V10", "antithetic simulator variance reduction", "simulation-side extension pending"),
        ("V11", "control-variate drift target", "needs martingale control variates"),
        ("V12", "realized-kernel/Hayashi-Yoshida QV", "no async/noisy QV module yet"),
        ("V13", "jackknife over dt", "requires multi-resolution estimator blend"),
        ("V14", "Ito-Taylor 1.5 drift target", "needs higher-order increment features"),
        ("B1", "local-linear/local-polynomial kernel regression", "core currently local-constant NW"),
        ("B3", "Richardson finite-step bias removal", "covered as V6 pending"),
        ("B4", "Tikhonov deconvolution EIV", "needs design-noise operator"),
        ("B5", "Kalman/EM latent smoothing", "new latent-state module required"),
        ("B6", "EWMA/local-linear pre-smoothing", "readout proxy exists; generator preprocessing pending"),
        ("B7", "bounded-domain boundary correction", "needs system-domain metadata"),
        ("B8", "lag-1 EIV default for noisy configs", "available in generator; not auto-triggered by runner"),
        ("N2", "ridge L-curve/GCV auto-tune", "fixed ridge floor is implemented; auto-tune pending"),
        ("N3", "GLS weighting", "same as G5 pending"),
        ("N4", "coverage diagnostic resampling", "diagnostic exists; sampler pending"),
        ("N5", "kernel drop/merge", "same as K10 pending"),
        ("N7", "condition-triggered solver fallback", "manual SVD/ridge variants implemented; trigger pending"),
        ("N8", "iterative refinement of normal equations", "not yet necessary in tests"),
        ("P1", "Cholesky-parametrized diffusion fit", "requires nonlinear tensor target"),
        ("P2", "log-Cholesky diffusion fit", "requires nonlinear tensor target"),
        ("P3", "matrix-log diffusion fit", "requires PSD-manifold regression"),
        ("P4", "joint PSD-manifold fit", "requires coupled tensor objective"),
        ("P5", "Ledoit-Wolf isotropic shrinkage", "post-fit shrinker pending"),
        ("P6", "small-signal off-diagonal fit weighting", "metric mask exists; fit weighting pending"),
        ("P7", "rank-1 degeneracy handling", "needs structural-rank detector"),
        ("S1", "moving-least-squares local fits", "new local estimator required"),
        ("S2", "partition-of-unity local blend", "depends on S1"),
        ("S3", "adaptive mesh refinement", "runner loop pending"),
        ("S4", "coarse-global plus local correction", "new multiscale estimator"),
        ("S5", "local adaptive polynomial degree", "requires degree map"),
        ("S6", "mixture of experts", "new estimator family"),
        ("D1", "GLS drift whitening by tensor", "requires pass-1 tensor weights"),
        ("D2", "iterative EM-like drift/tensor refine", "requires iterative generator wrapper"),
        ("D3", "separate-timescale drift", "requires custom target schedule"),
        ("D4", "eigenfunction-targeted estimation", "requires spectral target path"),
        ("D6", "per-component drift regularization", "per-target kw map pending"),
        ("M2", "strict no-oracle held-out CV hyperselection", "per-system stage uses non-oracle metrics; true hold-out loss pending"),
        ("M3", "bagged estimator ensemble", "multi-config scoring implemented; prediction blend pending"),
        ("M4", "stacked/blended estimators", "depends on M3"),
    ]
    out = []
    for idx, (family, desc, reason) in enumerate(items, start=1):
        out.append(
            _v(
                f"{family}_DEFER_{idx:02d}",
                family,
                desc,
                implemented=False,
                stage_a=False,
                deferred_reason=reason,
                priority="M",
            )
        )
    return out


def variant_catalog() -> list[V4Variant]:
    return implemented_variants() + deferred_variants()


def profile_settings(profile: str) -> dict:
    if profile == "smoke":
        return {
            "stage0_seeds": [1101],
            "stagea_seeds": [4101],
            "stageb_seeds": [5101],
            "stagec_seeds": [6101],
            "staged_seeds": [7101],
            "stage0_systems": ["correlated_ou", "rotational_ou", "heston_logsv"],
            "stageb_systems": ["correlated_ou", "rotational_ou", "heston_logsv"],
            "stagec_systems": ["correlated_ou", "rotational_ou", "heston_logsv"],
            "factorial_configs": 8,
            "stagea_max_variants": 8,
            "base_steps": 900,
            "heston_steps": 1200,
            "final_candidates": 3,
        }
    if profile == "standard":
        return {
            "stage0_seeds": [1101, 1119, 1137],
            "stagea_seeds": [4101, 4102, 4103],
            "stageb_seeds": [5101, 5102],
            "stagec_seeds": [6101, 6102, 6103],
            "staged_seeds": [7101, 7102],
            "stage0_systems": list(REGISTRY),
            "stageb_systems": list(REGISTRY),
            "stagec_systems": list(REGISTRY),
            "factorial_configs": 64,
            "stagea_max_variants": None,
            "base_steps": 1600,
            "heston_steps": 2600,
            "final_candidates": 5,
        }
    return {
        "stage0_seeds": [1101, 1119, 1137, 1155, 1173],
        "stagea_seeds": [4101, 4102, 4103],
        "stageb_seeds": [5101, 5102, 5103],
        "stagec_seeds": [6101, 6102, 6103, 6104, 6105],
        "staged_seeds": [7101, 7102, 7103],
        "stage0_systems": list(REGISTRY),
        "stageb_systems": list(REGISTRY),
        "stagec_systems": list(REGISTRY),
        "factorial_configs": 256,
        "stagea_max_variants": None,
        "base_steps": 2200,
        "heston_steps": 3600,
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


def read_rows(path: str) -> list[dict]:
    p = ROOT / path
    if not p.exists():
        return []
    with p.open() as f:
        return list(csv.DictReader(f))


def reset_outputs() -> None:
    for rel in [STAGE0_PATH, OFAT_PATH, FACTORIAL_PATH, GREEDY_PATH, PER_SYSTEM_PATH, LEADERBOARD_PATH, RUN_LOG_PATH]:
        p = ROOT / rel
        if p.exists():
            p.unlink()
    docs = ROOT / "docs/V4_FINDINGS.md"
    if docs.exists():
        docs.unlink()


def write_catalog(catalog: list[V4Variant]) -> None:
    rows = []
    for variant in catalog:
        row = asdict(variant)
        row["l1_ratio_grid"] = ";".join(str(x) for x in variant.l1_ratio_grid)
        rows.append(row)
    write_csv(CATALOG_PATH, rows, CATALOG_FIELDS)


def library_space_for(system_key: str, variant: V4Variant) -> str:
    if variant.library_space == "default":
        truth = REGISTRY[system_key]
        return v3_default_library_space(truth.library, truth.dim)
    return variant.library_space


def base_steps_for(system_key: str, variant: V4Variant, settings: dict) -> tuple[int, float]:
    dt = 1.0 / 252.0 if system_key in HESTON_SYSTEMS else 0.01
    base = settings["heston_steps"] if system_key in HESTON_SYSTEMS else settings["base_steps"]
    if REGISTRY[system_key].dim == 1:
        base = max(900, int(0.75 * settings["base_steps"]))
    if system_key in HARD_NULL_SYSTEMS:
        base = max(700, int(0.75 * base))
    n_steps = max(400, int(round(base * variant.n_steps_scale)))
    return n_steps, dt


def cell_for(system_key: str, variant: V4Variant, seed: int, run: int, stage: str, settings: dict) -> FitCell:
    truth = REGISTRY[system_key]
    n_steps, dt = base_steps_for(system_key, variant, settings)
    regressor = variant.regressor
    if truth.dim == 1 and regressor in {"lasso_stlsq", "elastic_net", "adaptive_lasso"}:
        regressor = "stlsq"
    library = variant.library_override or truth.library
    return FitCell(
        experiment=f"v4_{stage}",
        system_key=system_key,
        library=library,
        regressor=regressor,
        center_scheme=variant.center_scheme,
        n_centers=50 if truth.dim == 1 else variant.n_centers,
        bandwidth_mult=variant.bandwidth_mult,
        bandwidth_rule=variant.bandwidth_rule,
        dt=dt,
        n_steps=n_steps,
        seed=seed,
        run=run,
        subsample_k=variant.subsample_k,
        bias_correct=variant.bias_correct,
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
    )


def _safe_float(value: object, default: float = float("nan")) -> float:
    try:
        out = float(value)
        return out
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
    return 0.38 * drift_term + 0.36 * diff_term + 0.16 * psd + 0.10 * a12_term


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


def error_row(stage: str, variant: V4Variant, system_key: str, seed: int, run: int, exc: Exception, settings: dict) -> dict:
    truth = REGISTRY[system_key]
    n_steps, dt = base_steps_for(system_key, variant, settings)
    row = base_row(stage, variant, system_key, seed, run)
    row.update(
        {
            "tier": truth.tier,
            "dim": truth.dim,
            "library": variant.library_override or truth.library,
            "library_space": library_space_for(system_key, variant) if truth.dim == 2 else "raw",
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


def base_row(stage: str, variant: V4Variant, system_key: str, seed: int, run: int) -> dict:
    return {
        "stage": stage,
        "config_id": variant.variant_id,
        "variant_id": variant.variant_id,
        "family": variant.family,
        "description": variant.description,
        "implemented": variant.implemented,
        "deferred_reason": variant.deferred_reason,
        "system": system_key,
        "seed": seed,
        "run": run,
        "center_scheme": variant.center_scheme,
        "bandwidth_mult": variant.bandwidth_mult,
        "bandwidth_rule": variant.bandwidth_rule,
        "threshold_mode": variant.threshold_mode,
        "threshold": "" if variant.threshold is None else variant.threshold,
        "stlsq_threshold": "" if variant.stlsq_threshold is None else variant.stlsq_threshold,
        "pseudo_blocks": variant.pseudo_blocks,
        "l1_ratio_grid": ";".join(str(v) for v in variant.l1_ratio_grid),
        "adaptive_gamma": variant.adaptive_gamma,
        "svd_rtol": variant.svd_rtol,
        "ridge_floor": variant.ridge_floor,
        "bias_correct": variant.bias_correct,
        "selection_basis": "non_oracle_composite",
    }


def evaluate(stage: str, variant: V4Variant, system_key: str, seed: int, run: int, settings: dict) -> dict:
    if not variant.implemented:
        row = base_row(stage, variant, system_key, seed, run)
        row.update({"status": "DEFERRED_NOT_IMPLEMENTED", "score": 0.0})
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


def aggregate(rows: list[dict], keys: list[str]) -> list[dict]:
    def median_finite(values: list[float]) -> float:
        finite = [v for v in values if math.isfinite(v)]
        return float(np.median(finite)) if finite else float("nan")

    buckets: dict[tuple, list[dict]] = {}
    for row in rows:
        if row.get("status") in {"FAILED", "DEFERRED_NOT_IMPLEMENTED", "INFEASIBLE_BY_INVARIANT"}:
            continue
        key = tuple(row.get(k, "") for k in keys)
        buckets.setdefault(key, []).append(row)
    out = []
    for key, part in buckets.items():
        scores = [_safe_float(r.get("score")) for r in part]
        drifts = [_safe_float(r.get("drift_rel_l2")) for r in part]
        diffs = [_safe_float(r.get("diffusion_rel_l2")) for r in part]
        psds = [_safe_float(r.get("psd_valid_pct")) for r in part]
        row = {k: v for k, v in zip(keys, key)}
        row.update(
            {
                "n": len(part),
                "median_score": median_finite(scores),
                "median_drift_rel_l2": median_finite(drifts),
                "median_diffusion_rel_l2": median_finite(diffs),
                "median_psd_valid_pct": median_finite(psds),
                "validated_count": sum(r.get("status") == "VALIDATED_POSITIVE" for r in part),
                "failed_count": sum(r.get("status") == "FAILED" for r in part),
            }
        )
        out.append(row)
    return sorted(out, key=lambda r: (-_safe_float(r["median_score"], 0.0), _safe_float(r["median_drift_rel_l2"], 9.0)))


def stage_rows(path: str) -> list[dict]:
    return read_rows(path)


def completed_keys(path: str) -> set[tuple[str, str, str, str]]:
    keys = set()
    for row in read_rows(path):
        keys.add((row.get("stage", ""), row.get("variant_id", ""), row.get("system", ""), row.get("seed", "")))
    return keys


def _evaluate_task(task: tuple[int, int, str, V4Variant, str, int, int, dict]) -> tuple[int, int, dict]:
    ordinal, total, stage, variant, system_key, seed, run, settings = task
    row = evaluate(stage, variant, system_key, seed, run, settings)
    return ordinal, total, row


def run_grid(path: str, stage: str, variants: list[V4Variant], systems: list[str], seeds: list[int], settings: dict, resume: bool, jobs: int = 1) -> list[dict]:
    done = completed_keys(path) if resume else set()
    rows: list[dict] = []
    total = len(variants) * len(systems) * len(seeds)
    pending: list[tuple[int, int, str, V4Variant, str, int, int, dict]] = []
    count = 0
    for variant in variants:
        if not variant.implemented:
            row = evaluate(stage, variant, "ALL", 0, 0, settings)
            append_csv(path, row, ROW_FIELDS)
            continue
        for run, seed in enumerate(seeds):
            for system_key in systems:
                count += 1
                key = (stage, variant.variant_id, system_key, str(seed))
                if key in done:
                    continue
                pending.append((count, total, stage, variant, system_key, seed, run, settings))
    if not pending:
        return rows

    workers = max(1, int(jobs))
    if workers == 1:
        for task in pending:
            ordinal, total, row = _evaluate_task(task)
            append_csv(path, row, ROW_FIELDS)
            rows.append(row)
            if ordinal % 25 == 0 or row.get("status") == "FAILED":
                print(
                    f"{stage:8s} {ordinal:5d}/{total:<5d} {row.get('variant_id', ''):18s} {row.get('system', ''):22s} "
                    f"score={_safe_float(row.get('score'), 0.0):.3f} status={row.get('status')}"
                )
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
                if completed % 25 == 0 or row.get("status") == "FAILED":
                    print(
                        f"{stage:8s} {completed:5d}/{total:<5d} {row.get('variant_id', ''):18s} {row.get('system', ''):22s} "
                        f"score={_safe_float(row.get('score'), 0.0):.3f} status={row.get('status')}"
                    )
            submit_some(executor)
    return rows


def factorial_variants(n: int) -> list[V4Variant]:
    regressors = ["lasso_stlsq", "elastic_net", "adaptive_lasso", "ridge_threshold", "svd_threshold"]
    centers = ["quantile_grid", "uniform_grid", "kmeans"]
    m_values = [36, 64, 81, 100]
    h_values = [1.0, 1.25, 1.5, 2.0]
    h_rules = ["nn_median", "pairwise_median", "cov"]
    threshold_modes = ["relative", "absolute"]
    stlsq_values = [0.03, 0.06, 0.10, 0.16, 0.22]
    pseudo_values = [5, 10]
    r_values = [1, 4, 8]
    lag_values = [1, 2, 4]
    ridge_values = [1e-12, 1e-10, 1e-8]
    l1_values = [0.2, 0.5, 0.8, 0.95]
    gamma_values = [0.5, 1.0, 2.0]
    svd_values = [1e-10, 1e-8, 1e-6]
    out = []
    for i in range(n):
        reg = regressors[i % len(regressors)]
        mode = threshold_modes[(i // 3) % len(threshold_modes)]
        stlsq = stlsq_values[(i * 7) % len(stlsq_values)]
        threshold = None
        if reg in {"ridge_threshold", "svd_threshold"}:
            threshold = [0.02, 0.04, 0.08, 0.12][(i * 5) % 4]
            stlsq = None
        if mode == "absolute":
            stlsq = None if stlsq is None else min(stlsq, 0.01)
            threshold = None if threshold is None else min(threshold, 0.01)
        out.append(
            V4Variant(
                variant_id=f"F{i:03d}",
                family="factorial",
                description="Deterministic LHS/fractional-factorial v4 config",
                priority="H",
                regressor=reg,
                center_scheme=centers[(i * 5) % len(centers)],
                n_centers=m_values[(i * 11) % len(m_values)],
                bandwidth_mult=h_values[(i * 13) % len(h_values)],
                bandwidth_rule=h_rules[(i * 17) % len(h_rules)],
                threshold_mode=mode,
                threshold=threshold,
                stlsq_threshold=stlsq,
                pseudo_blocks=pseudo_values[(i * 19) % len(pseudo_values)],
                l1_ratio_grid=(l1_values[(i * 23) % len(l1_values)],),
                adaptive_gamma=gamma_values[(i * 29) % len(gamma_values)],
                svd_rtol=svd_values[(i * 31) % len(svd_values)],
                ridge_floor=ridge_values[(i * 37) % len(ridge_values)],
                n_trajectories=r_values[(i * 41) % len(r_values)],
                n_steps_scale=[0.75, 1.0, 1.35][(i * 43) % 3],
                subsample_k=lag_values[(i * 47) % len(lag_values)],
                bias_correct=[True, True, False][(i * 53) % 3],
            )
        )
    return out


def rows_by_config(rows: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for row in rows:
        out.setdefault(row.get("variant_id", row.get("config_id", "")), []).append(row)
    return out


def variant_from_row(row: dict) -> V4Variant:
    l1 = tuple(float(v) for v in str(row.get("l1_ratio_grid", "0.2;0.5;0.8;0.95")).split(";") if v)
    stlsq = row.get("stlsq_threshold", "")
    threshold = row.get("threshold", "")
    return V4Variant(
        variant_id=str(row.get("variant_id") or row.get("config_id") or "ROW_CONFIG"),
        family=str(row.get("family") or "row"),
        description=str(row.get("description") or "row-derived config"),
        regressor=str(row.get("regressor") or "lasso_stlsq"),
        library_space=str(row.get("library_space") or "default"),
        center_scheme=str(row.get("center_scheme") or "quantile_grid"),
        n_centers=int(float(row.get("M") or 64)),
        bandwidth_mult=float(row.get("bandwidth_mult") or 1.5),
        bandwidth_rule=str(row.get("bandwidth_rule") or "nn_median"),
        threshold_mode=str(row.get("threshold_mode") or "relative"),
        threshold=None if threshold in {"", "nan"} else float(threshold),
        stlsq_threshold=None if stlsq in {"", "nan"} else float(stlsq),
        pseudo_blocks=int(float(row.get("pseudo_blocks") or 5)),
        l1_ratio_grid=l1 or (0.2, 0.5, 0.8, 0.95),
        adaptive_gamma=float(row.get("adaptive_gamma") or 1.0),
        svd_rtol=float(row.get("svd_rtol") or 1e-8),
        ridge_floor=float(row.get("ridge_floor") or 1e-10),
        n_trajectories=int(float(row.get("R") or 4)),
        subsample_k=int(float(row.get("subsample_k") or 1)),
        bias_correct=str(row.get("bias_correct", "True")) != "False",
    )


def top_variants_from_rows(rows: list[dict], n: int) -> list[V4Variant]:
    ag = aggregate(rows, ["variant_id"])
    by_variant = rows_by_config(rows)
    out = []
    seen = set()
    for item in ag:
        vid = item["variant_id"]
        if vid in seen or vid not in by_variant:
            continue
        first = by_variant[vid][0]
        out.append(variant_from_row(first))
        seen.add(vid)
        if len(out) >= n:
            break
    if "V4_BASELINE" not in seen:
        out.append(v4_baseline())
    return out


def greedy_variants(seed_variant: V4Variant) -> list[V4Variant]:
    variants = []
    axes = [
        ("regressor", ["lasso_stlsq", "elastic_net", "adaptive_lasso", "ridge_threshold", "svd_threshold"]),
        ("center_scheme", ["quantile_grid", "uniform_grid", "kmeans"]),
        ("n_centers", [36, 64, 81, 100, 144]),
        ("bandwidth_rule", ["nn_median", "pairwise_median", "cov"]),
        ("bandwidth_mult", [1.0, 1.25, 1.5, 2.0]),
        ("threshold_mode", ["relative", "absolute"]),
        ("stlsq_threshold", [0.03, 0.06, 0.10, 0.16, 0.22]),
        ("pseudo_blocks", [5, 10]),
        ("n_trajectories", [1, 4, 8, 16]),
        ("subsample_k", [1, 2, 4]),
        ("ridge_floor", [1e-12, 1e-10, 1e-8]),
    ]
    for axis, values in axes:
        for value in values:
            kw = {axis: value}
            variant = replace(seed_variant, variant_id=f"GREEDY_{axis}_{str(value).replace('.', 'p').replace('-', 'm')}", family="greedy", description=f"Greedy coordinate {axis}={value}", **kw)
            if axis == "regressor" and value in {"ridge_threshold", "svd_threshold"}:
                variant = replace(variant, threshold=0.04, stlsq_threshold=None)
            if axis == "threshold_mode" and value == "absolute":
                variant = replace(variant, stlsq_threshold=0.005, threshold=0.005 if variant.regressor in {"ridge_threshold", "svd_threshold"} else variant.threshold)
            variants.append(variant)
    unique = {}
    for variant in variants:
        unique[variant.variant_id] = variant
    return list(unique.values())


def build_leaderboard() -> None:
    rows = stage_rows(STAGE0_PATH) + stage_rows(OFAT_PATH) + stage_rows(FACTORIAL_PATH) + stage_rows(GREEDY_PATH) + stage_rows(PER_SYSTEM_PATH)
    ag = aggregate(rows, ["stage", "variant_id", "family", "description", "regressor", "center_scheme", "bandwidth_rule"])
    fields = [
        "stage",
        "variant_id",
        "family",
        "description",
        "regressor",
        "center_scheme",
        "bandwidth_rule",
        "n",
        "median_score",
        "median_drift_rel_l2",
        "median_diffusion_rel_l2",
        "median_psd_valid_pct",
        "validated_count",
        "failed_count",
    ]
    write_csv(LEADERBOARD_PATH, ag, fields)


def make_figures() -> None:
    import matplotlib.pyplot as plt

    out_dir = ROOT / "figures/v4"
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
        ax.set_title("V4 Leaderboard Top Configurations")
        fig.tight_layout()
        fig.savefig(out_dir / "leaderboard_top15.png", dpi=160)
        plt.close(fig)
    v3_rows = read_rows("results/benchmark_summary.csv")
    stage0 = read_rows(STAGE0_PATH)
    if v3_rows and stage0:
        systems = sorted(set(r["system"] for r in stage0 if r.get("system") in {x.get("system") for x in v3_rows}))
        v3_med = [float(np.nanmedian([_safe_float(r.get("drift_rel_l2")) for r in v3_rows if r.get("system") == s])) for s in systems]
        v4_med = [float(np.nanmedian([_safe_float(r.get("drift_rel_l2")) for r in stage0 if r.get("system") == s])) for s in systems]
        fig, ax = plt.subplots(figsize=(max(8, 0.38 * len(systems)), 4))
        idx = np.arange(len(systems))
        ax.bar(idx - 0.18, v3_med, width=0.36, label="v3")
        ax.bar(idx + 0.18, v4_med, width=0.36, label="v4 stage0")
        ax.set_xticks(idx)
        ax.set_xticklabels([s[:16] for s in systems], rotation=55, ha="right")
        ax.set_ylabel("median drift relative L2")
        ax.set_title("V3 vs V4 Stage-0 Drift")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "stage0_vs_v3_drift.png", dpi=160)
        plt.close(fig)
    ofat = read_rows(OFAT_PATH)
    if ofat:
        ag = aggregate(ofat, ["family"])
        labels = [r["family"] for r in ag]
        vals = [_safe_float(r["median_score"], 0.0) for r in ag]
        fig, ax = plt.subplots(figsize=(max(8, 0.45 * len(labels)), 4))
        ax.bar(np.arange(len(labels)), vals)
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel("median composite score")
        ax.set_title("V4 OFAT Family Scores")
        fig.tight_layout()
        fig.savefig(out_dir / "ofat_family_scores.png", dpi=160)
        plt.close(fig)


def write_findings() -> None:
    leaderboard = read_rows(LEADERBOARD_PATH)
    stage0 = read_rows(STAGE0_PATH)
    ofat = read_rows(OFAT_PATH)
    factorial = read_rows(FACTORIAL_PATH)
    greedy = read_rows(GREEDY_PATH)
    per_system = read_rows(PER_SYSTEM_PATH)
    v3 = read_rows("results/benchmark_summary.csv")
    def med(rows: list[dict], col: str) -> float:
        vals = [_safe_float(r.get(col)) for r in rows if math.isfinite(_safe_float(r.get(col)))]
        return float(np.median(vals)) if vals else float("nan")

    stage0_drift = med(stage0, "drift_rel_l2")
    stage0_diff = med(stage0, "diffusion_rel_l2")
    v3_drift = med(v3, "drift_rel_l2")
    v3_diff = med(v3, "diffusion_rel_l2")
    validated = sum(r.get("status") == "VALIDATED_POSITIVE" for r in stage0)
    top = leaderboard[:10]
    top_lines = []
    for i, row in enumerate(top, start=1):
        top_lines.append(
            f"{i}. `{row.get('variant_id')}` ({row.get('family')}, {row.get('regressor')}): "
            f"score {float(row.get('median_score', 0.0)):.3f}, "
            f"drift {float(row.get('median_drift_rel_l2', float('nan'))):.3f}, "
            f"diffusion {float(row.get('median_diffusion_rel_l2', float('nan'))):.3f}"
        )
    heston_rows = [r for r in stage0 + per_system if r.get("system") in HESTON_SYSTEMS]
    heston_tensor = med(heston_rows, "diffusion_rel_l2")
    heston_b1 = med(heston_rows, "b1_rel_l2")
    heston_a12 = med(heston_rows, "a12_cosine")
    lines = [
        "# V4 Findings",
        "",
        "## Run contract",
        "",
        "V4 is an algorithmic-improvement campaign, not a new estimator fork.  The shared weak-form generator core remains the estimator; systems, libraries, kernels, selectors, trajectory budgets, and numerical stabilizers are varied as configuration flags.",
        "",
        "Artifacts written by this run:",
        "",
        f"- `{STAGE0_PATH}`",
        f"- `{OFAT_PATH}`",
        f"- `{FACTORIAL_PATH}`",
        f"- `{GREEDY_PATH}`",
        f"- `{PER_SYSTEM_PATH}`",
        f"- `{LEADERBOARD_PATH}`",
        f"- `{CATALOG_PATH}`",
        "- `figures/v4/leaderboard_top15.png`",
        "- `figures/v4/stage0_vs_v3_drift.png`",
        "- `figures/v4/ofat_family_scores.png`",
        "",
        "## Stage-0 baseline",
        "",
        f"V3 median drift/diffusion from `results/benchmark_summary.csv`: {v3_drift:.4g} / {v3_diff:.4g}.",
        f"V4 Stage-0 median drift/diffusion: {stage0_drift:.4g} / {stage0_diff:.4g}.",
        f"V4 Stage-0 validated-positive cells: {validated} of {len(stage0)}.",
        "",
        "The Stage-0 baseline includes the v4 default flips: standardized-coordinate libraries where back-transform is valid, LassoCV debias STLSQ, pseudo-block grouped CV for single-trajectory projected rows, and bias-corrected diffusion targets.",
        "",
        "## Top configurations",
        "",
        *(top_lines or ["No successful leaderboard rows were written."]),
        "",
        "## Heston status",
        "",
        f"Heston-family median tensor diffusion error across Stage-0/per-system rows: {heston_tensor:.4g}.",
        f"Heston-family median log-price/first drift error: {heston_b1:.4g}.",
        f"Heston-family median off-diagonal tensor cosine: {heston_a12:.4g}.",
        "",
        "This preserves the v3 honesty boundary: Heston tensor/leverage recovery is a real positive result, while the low-SNR log-price drift component remains a named limitation unless long multi-trajectory budgets close it.  The v4 search logs oracle headroom per row so selection failure is separated from irreducible SNR.",
        "",
        "## What changed in code",
        "",
        "- `regression.py`: elastic-net, adaptive LASSO, truncated-SVD thresholding, and pseudo-block GroupKFold fallback.",
        "- `generator.py`: covariance-matrix bandwidth rule behind `bandwidth_rule=\"cov\"`.",
        "- `experiments/v4/run_v4_campaign.py`: staged v4 search with OFAT, fractional-factorial, greedy coordinate, per-system selection, leaderboard, plots, and findings.",
        "",
        "## Deferred but visible headroom",
        "",
        "The variant catalog includes deferred entries for local-linear kernels, PSD-manifold diffusion fitting, GLS row weighting, midpoint/Richardson/gEDMD targets, multiscale local estimators, and true blended ensembles.  These are not reported as failures of the current estimator; they are named implementation work for v5/v6.",
        "",
        "## Honest nulls retained",
        "",
        "Conservative circulation calibration remains conservative rather than a nominal 5 percent test.  Transport weighting remains unclaimed.  Nonlinear drift discovery remains dependent on the correct library.  Heston log-price drift remains separate from Heston tensor and leverage success.",
    ]
    p = ROOT / "docs/V4_FINDINGS.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines) + "\n")


def update_logs(profile: str, started: float) -> None:
    rows = [
        {
            "profile": profile,
            "stage0_rows": len(read_rows(STAGE0_PATH)),
            "ofat_rows": len(read_rows(OFAT_PATH)),
            "factorial_rows": len(read_rows(FACTORIAL_PATH)),
            "greedy_rows": len(read_rows(GREEDY_PATH)),
            "per_system_rows": len(read_rows(PER_SYSTEM_PATH)),
            "leaderboard_rows": len(read_rows(LEADERBOARD_PATH)),
            "runtime_sec": time.perf_counter() - started,
        }
    ]
    write_csv(RUN_LOG_PATH, rows, list(rows[0]))


def run_stage_d(settings: dict, resume: bool, jobs: int = 1) -> None:
    source_rows = stage_rows(GREEDY_PATH) + stage_rows(FACTORIAL_PATH) + stage_rows(OFAT_PATH)
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
                    variant_id=f"BEST_{system_key}_{variant.variant_id}",
                    family="per_system",
                    description=f"Per-system candidate for {system_key}: {variant.description}",
                )
            )
        print(f"stageD selected/evaluated {len(candidates)} candidates for {system_key}")
        run_grid(PER_SYSTEM_PATH, "stageD", variants, [system_key], settings["staged_seeds"], settings, resume, jobs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=["smoke", "standard", "full"], default="full")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--factorial-configs", type=int, default=None)
    parser.add_argument("--jobs", type=int, default=int(os.environ.get("V4_JOBS", "1")))
    args = parser.parse_args()

    settings = profile_settings(args.profile)
    if args.factorial_configs is not None:
        settings["factorial_configs"] = args.factorial_configs
    started = time.perf_counter()
    if not args.resume:
        reset_outputs()
    catalog = variant_catalog()
    write_catalog(catalog)

    print("V4 Stage 0: baseline zoo rerun")
    run_grid(STAGE0_PATH, "stage0", [v4_baseline()], settings["stage0_systems"], settings["stage0_seeds"], settings, args.resume, args.jobs)

    print("V4 Stage A: OFAT screening")
    stage_a_variants = [v for v in catalog if v.stage_a]
    if settings["stagea_max_variants"] is not None:
        stage_a_variants = stage_a_variants[: settings["stagea_max_variants"]]
    run_grid(OFAT_PATH, "stageA", stage_a_variants, SCREEN_SYSTEMS, settings["stagea_seeds"], settings, args.resume, args.jobs)

    print("V4 Stage B: fractional-factorial full-zoo screen")
    run_grid(FACTORIAL_PATH, "stageB", factorial_variants(settings["factorial_configs"]), settings["stageb_systems"], settings["stageb_seeds"], settings, args.resume, args.jobs)

    print("V4 Stage C: greedy coordinate ascent")
    source_rows = stage_rows(FACTORIAL_PATH) + stage_rows(OFAT_PATH)
    seed_variant = top_variants_from_rows(source_rows, 1)[0] if source_rows else v4_baseline()
    run_grid(GREEDY_PATH, "stageC", greedy_variants(seed_variant), settings["stagec_systems"], settings["stagec_seeds"], settings, args.resume, args.jobs)

    print("V4 Stage D: per-system best selection")
    run_stage_d(settings, args.resume, args.jobs)

    build_leaderboard()
    make_figures()
    write_findings()
    update_logs(args.profile, started)
    print("V4 DONE")


if __name__ == "__main__":
    main()
