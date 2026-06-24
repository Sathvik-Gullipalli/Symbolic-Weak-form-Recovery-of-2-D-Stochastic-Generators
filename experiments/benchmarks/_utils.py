from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace

import numpy as np

from experiments.common import ROOT, write_rows
from sde2d.generator import GeneratorFit2D, fit_generator_2d
from sde2d.library import Library, make_library
from sde2d.metrics import a12_sign_accuracy, central_grid, cosine_similarity, function_l2_errors, psd_validity, relative_l2, tensor_metrics
from sde2d.regression import fit_oracle_ols
from sde2d.systems import REGISTRY, System

Array = np.ndarray


@dataclass(frozen=True)
class FitCell:
    experiment: str
    system_key: str
    library: str
    regressor: str = "lasso_stlsq"
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
    dt: float = 0.01
    n_steps: int = 5000
    seed: int = 0
    run: int = 0
    noise_level: float = 0.0
    noise_kind: str = "none"
    subsample_k: int = 1
    bias_correct: bool = True
    noise_correct: bool = False
    n_trajectories: int = 1
    library_space: str = "z"
    threshold: float | None = None
    stlsq_threshold: float | None = None
    threshold_mode: str = "relative"
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


ZSPACE_BACKTRANSFORM_PRESETS = {"A", "POLY2", "B", "POLY3", "C", "POLY4", "D", "LOG_HESTON"}


def supports_zspace_backtransform(library: str, dim: int) -> bool:
    return dim == 2 and library.upper() in ZSPACE_BACKTRANSFORM_PRESETS


def v3_default_library_space(library: str, dim: int) -> str:
    return "z" if supports_zspace_backtransform(library, dim) else "raw"


def v3_default_regressor(dim: int) -> str:
    return "lasso_stlsq" if dim == 2 else "stlsq"


def grid_shape_for(n_centers: int, dim: int) -> tuple[int, int] | None:
    if dim < 2:
        return None
    side = int(round(math.sqrt(n_centers)))
    return (side, side) if side * side == n_centers else None


def instantiate_system(system_key: str) -> System:
    return REGISTRY[system_key].cls()


def fit_cell(cell: FitCell) -> tuple[System, Array, GeneratorFit2D, float]:
    system = instantiate_system(cell.system_key)
    t0 = time.perf_counter()
    x, inc, traj_ids = simulate_cell_data(system, cell)
    lib = make_library(cell.library)
    regression_kw = regression_kwargs(cell)
    fit = fit_generator_2d(
        x,
        increments=inc,
        dt=cell.dt * cell.subsample_k,
        library=lib,
        n_centers=cell.n_centers,
        center_scheme=cell.center_scheme,
        grid_shape=grid_shape_for(cell.n_centers, REGISTRY[cell.system_key].dim),
        bandwidth_multiplier=cell.bandwidth_mult,
        bandwidth_rule=cell.bandwidth_rule,
        knn_k=cell.knn_k,
        local_poly_order=cell.local_poly_order,
        projection_normalization=cell.projection_normalization,
        projection_scales=cell.projection_scales,
        prune_min_effective_samples=cell.prune_min_effective_samples,
        target_anchor=cell.target_anchor,
        regressor=cell.regressor,
        regression_kw=regression_kw,
        target_regression_kw=cell.target_regression_kw,
        bias_correct=cell.bias_correct,
        noise_correct=cell.noise_correct,
        library_space=cell.library_space,
        traj_ids=traj_ids,
        seed=cell.seed,
        gls_weighting=cell.gls_weighting,
        gls_iterations=cell.gls_iterations,
        diffusion_parameterization=cell.diffusion_parameterization,
        diffusion_shrinkage=cell.diffusion_shrinkage,
        rank1_project=cell.rank1_project,
    )
    return system, x, fit, time.perf_counter() - t0


def regression_kwargs(cell: FitCell) -> dict:
    if cell.regressor == "stlsq":
        return {"threshold": 0.02 if cell.threshold is None else cell.threshold, "threshold_mode": cell.threshold_mode, "ridge_floor": cell.ridge_floor}
    if cell.regressor in {"lasso_stlsq", "lassocv_debias_stlsq"}:
        return {
            "stlsq_threshold": 0.10 if cell.stlsq_threshold is None else cell.stlsq_threshold,
            "threshold_mode": cell.threshold_mode,
            "pseudo_blocks": cell.pseudo_blocks,
            "ridge_floor": cell.ridge_floor,
        }
    if cell.regressor in {"elastic_net", "elasticnet", "elastic_net_stlsq", "elastic_net_debias_stlsq"}:
        return {
            "stlsq_threshold": 0.18 if cell.stlsq_threshold is None else cell.stlsq_threshold,
            "threshold_mode": cell.threshold_mode,
            "pseudo_blocks": cell.pseudo_blocks,
            "l1_ratio_grid": cell.l1_ratio_grid,
            "ridge_floor": cell.ridge_floor,
        }
    if cell.regressor in {"adaptive_lasso", "adaptive_lasso_stlsq", "adaptive_lasso_debias_stlsq"}:
        return {
            "stlsq_threshold": 0.14 if cell.stlsq_threshold is None else cell.stlsq_threshold,
            "threshold_mode": cell.threshold_mode,
            "pseudo_blocks": cell.pseudo_blocks,
            "gamma": cell.adaptive_gamma,
            "ridge_floor": cell.ridge_floor,
        }
    if cell.regressor == "ridge_threshold":
        return {
            "threshold": 0.08 if cell.threshold is None else cell.threshold,
            "threshold_mode": cell.threshold_mode,
            "ridge_floor": cell.ridge_floor,
        }
    if cell.regressor == "svd_threshold":
        return {
            "threshold": 0.05 if cell.threshold is None else cell.threshold,
            "threshold_mode": cell.threshold_mode,
            "svd_rtol": cell.svd_rtol,
        }
    if cell.regressor in {"stability_selection", "bootstrap_lasso"}:
        return {
            "stlsq_threshold": 0.12 if cell.stlsq_threshold is None else cell.stlsq_threshold,
            "threshold_mode": cell.threshold_mode,
            "pseudo_blocks": cell.pseudo_blocks,
            "n_boot": 8,
        }
    if cell.regressor in {"bic", "aic", "ebic", "information_criterion"}:
        return {"stlsq_threshold": 0.08 if cell.stlsq_threshold is None else cell.stlsq_threshold, "threshold_mode": cell.threshold_mode}
    if cell.regressor in {"sr3"}:
        return {"threshold": 0.08 if cell.threshold is None else cell.threshold, "threshold_mode": cell.threshold_mode}
    if cell.regressor in {"best_subset", "mio_l0", "l0"}:
        return {"k_max": 4}
    if cell.regressor in {"omp", "forward_stagewise"}:
        return {"n_nonzero": 4}
    if cell.regressor in {"tls", "total_least_squares", "huber", "irls", "ridge_gcv", "ridge_auto"}:
        return {"threshold": 0.05 if cell.threshold is None else cell.threshold, "threshold_mode": cell.threshold_mode}
    return {}


def simulate_cell_data(system: System, cell: FitCell) -> tuple[Array, Array, Array]:
    states = []
    increments = []
    groups = []
    for r in range(cell.n_trajectories):
        path = system.simulate(dt=cell.dt, M=cell.n_steps, seed=cell.seed + 10007 * r)
        if cell.subsample_k > 1:
            path = path[:: cell.subsample_k]
        curr = path[:-1]
        inc = np.diff(path, axis=0)
        states.append(curr)
        increments.append(inc)
        groups.append(np.full(curr.shape[0], r, dtype=int))
    return np.vstack(states), np.vstack(increments), np.concatenate(groups)


def _target_values(system: System, points: Array, target: str) -> Array:
    if target.startswith("b"):
        idx = int(target[1]) - 1
        return system.true_drift(points)[:, idx]
    a = system.true_diffusion(points)
    if target == "a11":
        return a[:, 0, 0]
    if target == "a12":
        return a[:, 0, 1]
    if target == "a22":
        return a[:, 1, 1]
    raise ValueError(target)


def _fit_values(fit: GeneratorFit2D, points: Array, target: str, psd: bool = False) -> Array:
    b, a = fit.evaluate(points, psd=psd)
    if target.startswith("b"):
        return b[:, int(target[1]) - 1]
    if target == "a11":
        return a[:, 0, 0]
    if target == "a12":
        return a[:, 0, 1]
    if target == "a22":
        return a[:, 1, 1]
    raise ValueError(target)


def active_targets(dim: int) -> list[str]:
    return ["b1", "a11"] if dim == 1 else ["b1", "b2", "a11", "a22", "a12"]


def true_coefficients(library: Library, system: System, points: Array, target: str) -> Array:
    theta = library.transform(points)
    y = _target_values(system, points, target)
    gram = theta.T @ theta
    lam = 1e-10 * max(float(np.mean(np.diag(gram))), 1e-30)
    return np.linalg.solve(gram + lam * np.eye(theta.shape[1]), theta.T @ y)


def estimated_coefficients(fit: GeneratorFit2D, target: str) -> Array:
    if target.startswith("b"):
        return fit.drift_coef[:, int(target[1]) - 1]
    key = {"a11": (0, 0), "a12": (0, 1), "a22": (1, 1)}[target]
    return fit.diff_coef[key]


def true_coefficients_fit_space(fit: GeneratorFit2D, system: System, points: Array, target: str) -> Array:
    features = fit.standardizer.transform(points) if fit.library_space == "z" else points
    theta = fit.library.transform(features)
    y = _target_values(system, points, target)
    gram = theta.T @ theta
    lam = 1e-10 * max(float(np.mean(np.diag(gram))), 1e-30)
    return np.linalg.solve(gram + lam * np.eye(theta.shape[1]), theta.T @ y)


def _coef_to_raw_space(fit: GeneratorFit2D, coef: Array) -> Array:
    if fit.library_space == "z" and fit.basis_change is not None:
        return fit.basis_change @ coef
    return coef


def oracle_fit_for(fit: GeneratorFit2D, system: System, points: Array) -> GeneratorFit2D:
    design = fit.raw_targets.get("aggregate_design", fit.raw_targets["design"])
    dim = fit.dim
    drift = np.zeros_like(fit.drift)
    diffusion: dict[tuple[int, int], Array] = {}
    for target in active_targets(dim):
        true_coef = true_coefficients_fit_space(fit, system, points, target)
        support = np.abs(true_coef) > 1e-5
        projected_target = fit.raw_targets.get("aggregate_drift", fit.raw_targets["drift"]).get(target) if target.startswith("b") else fit.raw_targets.get("aggregate_diffusion", fit.raw_targets["diffusion"]).get(target)
        if projected_target is None:
            continue
        res = fit_oracle_ols(design, projected_target, true_support=support)
        coef = _coef_to_raw_space(fit, res.coef)
        if target.startswith("b"):
            drift[:, int(target[1]) - 1] = coef
        else:
            diffusion[{"a11": (0, 0), "a12": (0, 1), "a22": (1, 1)}[target]] = coef
    return replace(fit, drift=drift, diffusion=diffusion)


def oracle_diagnostics(fit: GeneratorFit2D, system: System, points: Array) -> dict:
    oracle_fit = oracle_fit_for(fit, system, points)
    errs = function_l2_errors(oracle_fit, system, points)
    out = {
        "fit": oracle_fit,
        "drift_rel_l2": errs["drift_rel_l2"],
        "diffusion_rel_l2": errs["diffusion_rel_l2"],
        "oracle_ols_passes": bool(errs["drift_rel_l2"] < 0.75 and errs["diffusion_rel_l2"] < 0.50),
    }
    if fit.dim == 2:
        ah = oracle_fit.evaluate(points)[1][:, 0, 1]
        at = system.true_diffusion(points)[:, 0, 1]
        out["a12_rel_l2"] = relative_l2(ah, at)
        out["a12_cosine"] = cosine_similarity(ah, at)
        out["a12_sign_acc"] = a12_sign_accuracy(ah, at)
    return out


def _level_from_thresholds(value: float, strong: float, medium: float) -> str:
    if np.isfinite(value) and value < strong:
        return "strong"
    if np.isfinite(value) and value < medium:
        return "medium"
    return "fail"


def split_pass_levels(drift: float, diffusion: float, psd_pct: float, a12_sign: float, a12_cosine: float) -> dict:
    sign_ok = np.isnan(a12_sign) or a12_sign >= 0.95
    cosine_ok = np.isnan(a12_cosine) or a12_cosine >= 0.90
    drift_level = _level_from_thresholds(drift, 0.25, 0.75)
    if diffusion < 0.25 and psd_pct >= 0.99 and sign_ok and cosine_ok:
        tensor_level = "strong"
    elif diffusion < 0.50 and psd_pct >= 0.95 and sign_ok and cosine_ok:
        tensor_level = "medium"
    else:
        tensor_level = "fail"
    combined = "strong" if drift_level == "strong" and tensor_level == "strong" else "medium" if drift_level in {"strong", "medium"} and tensor_level in {"strong", "medium"} else "fail"
    return {"drift_pass_level": drift_level, "tensor_pass_level": tensor_level, "pass_level": combined}


def status_from_level(system_key: str, level: str) -> str:
    verdict = REGISTRY[system_key].verdict
    expected_negative = "FAIL" in verdict or verdict == "FRAGILE"
    if expected_negative and level == "fail":
        return "NEGATIVE_RESULT_WORTH_REPORTING"
    if level in {"strong", "medium"}:
        return "VALIDATED_POSITIVE"
    return "INCONCLUSIVE"


def rows_for_fit(cell: FitCell, system: System, x: Array, fit: GeneratorFit2D, runtime_sec: float) -> dict[str, list[dict]]:
    dim = REGISTRY[cell.system_key].dim
    eval_points = central_grid(x, 17 if dim == 2 else 80)
    errs = function_l2_errors(fit, system, eval_points)
    tmet = tensor_metrics(fit, system, eval_points) if dim == 2 else {"a12_sign_accuracy": float("nan")}
    psd = psd_validity(fit.evaluate(eval_points)[1])
    if dim == 2:
        true_a12 = system.true_diffusion(eval_points)[:, 0, 1]
        fit_a12 = fit.evaluate(eval_points)[1][:, 0, 1]
        a12_cosine = cosine_similarity(fit_a12, true_a12)
    else:
        a12_cosine = float("nan")
    split = split_pass_levels(errs["drift_rel_l2"], errs["diffusion_rel_l2"], psd["pct_psd_valid"], tmet.get("a12_sign_accuracy", float("nan")), a12_cosine)
    status = status_from_level(cell.system_key, split["pass_level"])
    oracle = oracle_diagnostics(fit, system, eval_points)
    base = {
        "experiment": cell.experiment,
        "system": cell.system_key,
        "tier": REGISTRY[cell.system_key].tier,
        "dim": dim,
        "library": cell.library,
        "center_scheme": cell.center_scheme,
        "M": fit.bandwidth_meta["n_centers"],
        "bandwidth_mult": cell.bandwidth_mult,
        "bandwidth_rule": cell.bandwidth_rule,
        "regressor": cell.regressor,
        "library_space": cell.library_space,
        "dt": cell.dt,
        "T": cell.dt * cell.n_steps,
        "R": cell.n_trajectories,
        "n_steps": cell.n_steps,
        "seed": cell.seed,
        "run": cell.run,
        "noise_level": cell.noise_level,
        "noise_kind": cell.noise_kind,
        "subsample_k": cell.subsample_k,
        "local_poly_order": cell.local_poly_order,
        "target_anchor": cell.target_anchor,
        "gls_weighting": cell.gls_weighting,
        "gls_iterations": fit.bandwidth_meta.get("gls_iterations", 0),
        "diffusion_parameterization": cell.diffusion_parameterization,
    }
    coef_rows: list[dict] = []
    supp_rows: list[dict] = []
    fun_rows: list[dict] = []
    for target in active_targets(dim):
        c_true = true_coefficients(fit.library, system, eval_points, target)
        c_hat = estimated_coefficients(fit, target)
        active_true = np.abs(c_true) > 1e-5
        selected = np.abs(c_hat) > 1e-8
        for k, name in enumerate(fit.library.names):
            den = max(abs(float(c_true[k])), 1e-12)
            coef_rows.append(
                {
                    "experiment": cell.experiment,
                    "system": cell.system_key,
                    "library": cell.library,
                    "regressor": cell.regressor,
                    "seed": cell.seed,
                    "run": cell.run,
                    "target": target,
                    "term_name": name,
                    "term_index": k,
                    "coef_true": float(c_true[k]),
                    "coef_hat": float(c_hat[k]),
                    "abs_error": float(abs(c_hat[k] - c_true[k])),
                    "rel_error": float(abs(c_hat[k] - c_true[k]) / den),
                    "is_active_true": bool(active_true[k]),
                    "is_selected": bool(selected[k]),
                }
            )
        tp = int(np.sum(active_true & selected))
        fp = int(np.sum(~active_true & selected))
        fn = int(np.sum(active_true & ~selected))
        supp_rows.append(
            {
                "experiment": cell.experiment,
                "system": cell.system_key,
                "library": cell.library,
                "regressor": cell.regressor,
                "seed": cell.seed,
                "run": cell.run,
                "target": target,
                "n_true_active": int(active_true.sum()),
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "exact_support_match": bool(fp == 0 and fn == 0),
                "max_inactive_coef_magnitude": float(np.max(np.abs(c_hat[~active_true]))) if np.any(~active_true) else 0.0,
            }
        )
        yt = _target_values(system, eval_points, target)
        yh = _fit_values(fit, eval_points, target)
        fun_rows.append(
            {
                "experiment": cell.experiment,
                "system": cell.system_key,
                "library": cell.library,
                "regressor": cell.regressor,
                "seed": cell.seed,
                "run": cell.run,
                "field": target,
                "l2_mu_weighted": relative_l2(yh, yt),
                "l2_uniform_grid": relative_l2(yh, yt),
                "cosine": cosine_similarity(yh, yt),
                "n_grid_points": int(eval_points.shape[0]),
            }
        )
    support_exact = all(row["exact_support_match"] for row in supp_rows)
    summary_row = {
        **base,
        "b1_rel_l2": errs.get("b1_rel_l2", float("nan")),
        "b2_rel_l2": errs.get("b2_rel_l2", float("nan")),
        "drift_rel_l2": errs["drift_rel_l2"],
        "diffusion_rel_l2": errs["diffusion_rel_l2"],
        "a12_cosine": a12_cosine,
        "a12_sign_acc": tmet.get("a12_sign_accuracy", float("nan")),
        "psd_valid_pct": psd["pct_psd_valid"],
        "oracle_drift_rel_l2": oracle["drift_rel_l2"],
        "oracle_diffusion_rel_l2": oracle["diffusion_rel_l2"],
        "oracle_a12_rel_l2": oracle.get("a12_rel_l2", float("nan")),
        "oracle_a12_cosine": oracle.get("a12_cosine", float("nan")),
        "oracle_a12_sign_acc": oracle.get("a12_sign_acc", float("nan")),
        "oracle_ols_passes": oracle["oracle_ols_passes"],
        "support_exact_match": bool(support_exact),
        "drift_pass_level": split["drift_pass_level"],
        "tensor_pass_level": split["tensor_pass_level"],
        "pass_level": split["pass_level"],
        "status": status,
        "runtime_sec": runtime_sec,
    }
    tensor_rows: list[dict] = []
    psd_rows: list[dict] = []
    for estimator, ev_fit, psd_project in [("raw", fit, False), ("psd_projected", fit, True), ("oracle_ols", oracle["fit"], False)]:
        ev_a = ev_fit.evaluate(eval_points, psd=psd_project)[1]
        ev_psd = psd_validity(ev_a)
        ev_tmet = tensor_metrics(ev_fit, system, eval_points, psd=psd_project) if dim == 2 else {}
        ev_errs = function_l2_errors(ev_fit, system, eval_points)
        tensor_rows.append(
            {
                "experiment": cell.experiment,
                "system": cell.system_key,
                "library": cell.library,
                "regressor": cell.regressor,
                "seed": cell.seed,
                "run": cell.run,
                "estimator": estimator,
                "frob_pointwise_mean": ev_tmet.get("frob_avg", ev_errs["diffusion_rel_l2"]),
                "frob_avg": ev_tmet.get("frob_rel_avg", ev_errs["diffusion_rel_l2"]),
                "a11_rel_l2": ev_errs.get("a11_rel_l2", float("nan")),
                "a22_rel_l2": ev_errs.get("a22_rel_l2", float("nan")),
                "a12_rel_l2": ev_errs.get("a12_rel_l2", float("nan")),
                "a12_abs_error_mean": ev_tmet.get("a12_abs_err_avg", float("nan")),
                "a12_sign_accuracy": ev_tmet.get("a12_sign_accuracy", float("nan")),
            }
        )
        psd_rows.append(
            {
                "experiment": cell.experiment,
                "system": cell.system_key,
                "library": cell.library,
                "regressor": cell.regressor,
                "seed": cell.seed,
                "run": cell.run,
                "estimator": estimator,
                "psd_valid_pct": ev_psd["pct_psd_valid"],
                "min_eigenvalue": ev_psd["min_eigenvalue_grid"],
                "psd_violation_rate": ev_psd["psd_violation_rate"],
                "det_violation_rate": ev_psd["det_gap_violation_rate"],
                "median_condition_number": ev_psd["median_condition_number"],
                "n_grid_points": int(eval_points.shape[0]),
            }
        )
    stat_rows = simulation_stat_rows(cell, system, x, fit)
    return {
        "benchmark_summary": [summary_row],
        "coefficient_recovery": coef_rows,
        "support_recovery": supp_rows,
        "function_error": fun_rows,
        "diffusion_tensor_error": tensor_rows,
        "psd_validity": psd_rows,
        "simulation_statistics": stat_rows,
    }


def simulation_stat_rows(cell: FitCell, system: System, x: Array, fit: GeneratorFit2D) -> list[dict]:
    x_rec = simulate_recovered(fit, x[0], cell.dt * cell.subsample_k, min(1500, max(300, len(x) - 1)), cell.seed + 9001)
    true_vals = summary_stats(x)
    rec_vals = summary_stats(x_rec)
    rows = []
    for stat, value in true_vals.items():
        rv = rec_vals.get(stat, float("nan"))
        se = float(np.nanstd([value, rv])) if np.isfinite(rv) else float("nan")
        rows.append(
            {
                "experiment": cell.experiment,
                "system": cell.system_key,
                "seed": cell.seed,
                "run": cell.run,
                "stat": stat,
                "true_value": value,
                "recovered_value": rv,
                "abs_diff": float(abs(rv - value)) if np.isfinite(rv) else float("nan"),
                "within_mc_ci": bool(abs(rv - value) <= 3.0 * max(se, 1e-12)) if np.isfinite(rv) else False,
                "mc_ci_low": float(value - 3.0 * max(se, 1e-12)) if np.isfinite(rv) else float("nan"),
                "mc_ci_high": float(value + 3.0 * max(se, 1e-12)) if np.isfinite(rv) else float("nan"),
            }
        )
    return rows


def summary_stats(x: Array) -> dict[str, float]:
    vals = {
        "mean_x": float(np.mean(x[:, 0])),
        "var_x": float(np.var(x[:, 0])),
        "acf_x_lag1": _acf1(x[:, 0]),
    }
    if x.shape[1] > 1:
        vals.update(
            {
                "mean_y": float(np.mean(x[:, 1])),
                "var_y": float(np.var(x[:, 1])),
                "cov_xy": float(np.cov(x.T)[0, 1]),
                "acf_y_lag1": _acf1(x[:, 1]),
                "crosscorr_lag1": float(np.corrcoef(x[:-1, 0], x[1:, 1])[0, 1]),
            }
        )
    return vals


def simulate_recovered(fit: GeneratorFit2D, x0: Array, dt: float, n_steps: int, seed: int) -> Array:
    rng = np.random.default_rng(seed)
    x0 = np.asarray(x0, float)
    dim = x0.size
    out = np.zeros((n_steps + 1, dim))
    out[0] = x0
    lo = fit.standardizer.mean[:dim] - 8.0 * fit.standardizer.scale[:dim]
    hi = fit.standardizer.mean[:dim] + 8.0 * fit.standardizer.scale[:dim]
    for n in range(n_steps):
        b, a = fit.evaluate(out[n : n + 1], psd=True)
        mat = a[0]
        if dim == 1:
            root = np.array([[math.sqrt(max(mat[0, 0], 0.0))]])
        else:
            vals, vecs = np.linalg.eigh(0.5 * (mat + mat.T))
            root = vecs @ np.diag(np.sqrt(np.maximum(vals, 0.0)))
        out[n + 1] = out[n] + b[0] * dt + root @ rng.standard_normal(dim) * math.sqrt(dt)
        out[n + 1] = np.clip(out[n + 1], lo, hi)
    return out


def _acf1(v: Array) -> float:
    v = np.asarray(v, float)
    if len(v) < 3:
        return float("nan")
    return float(np.corrcoef(v[:-1], v[1:])[0, 1])


def write_campaign_tables(tables: dict[str, list[dict]], *, overwrite: bool) -> None:
    mapping = {
        "benchmark_summary": "results/benchmark_summary.csv",
        "coefficient_recovery": "results/coefficient_recovery.csv",
        "support_recovery": "results/support_recovery.csv",
        "function_error": "results/function_error.csv",
        "diffusion_tensor_error": "results/diffusion_tensor_error.csv",
        "psd_validity": "results/psd_validity.csv",
        "simulation_statistics": "results/simulation_statistics.csv",
    }
    for key, path in mapping.items():
        rows = tables.get(key, [])
        if rows:
            write_rows(path, rows) if overwrite else _append_rows(path, rows)


def _append_rows(path: str, rows: list[dict]) -> None:
    from experiments.common import append_rows

    append_rows(path, rows)


def save_status_figure(csv_path: str, figure_path: str, title: str) -> None:
    import csv

    import matplotlib.pyplot as plt

    p = ROOT / csv_path
    if not p.exists():
        return
    with p.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return
    labels = [r["system"][:18] for r in rows]
    drift = [float(r["drift_rel_l2"]) for r in rows]
    diff = [float(r["diffusion_rel_l2"]) for r in rows]
    fig, ax = plt.subplots(figsize=(max(8, 0.35 * len(labels)), 4))
    idx = np.arange(len(labels))
    ax.bar(idx - 0.18, drift, width=0.36, label="drift")
    ax.bar(idx + 0.18, diff, width=0.36, label="diffusion")
    ax.axhline(0.25, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(idx)
    ax.set_xticklabels(labels, rotation=55, ha="right")
    ax.set_ylabel("relative L2")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    out = ROOT / figure_path
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
