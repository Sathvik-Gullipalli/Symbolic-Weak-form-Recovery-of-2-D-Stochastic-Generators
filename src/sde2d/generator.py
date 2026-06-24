from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .diffusion import lag1_noise_covariance, project_psd
from .kernels import (
    choose_centers,
    gaussian_projection_matrix,
    knn_bandwidth,
    local_cov_bandwidth,
    local_polynomial_projection_matrix,
    median_bandwidth,
    prune_projection,
)
from .library import Library, make_library, polynomial_change_of_basis
from .regression import SelectionResult, solve
from .standardize import Standardizer

Array = np.ndarray


def _diffusion_metric_bandwidth(z: Array, increments_z: Array, centers: Array, multiplier: float, k: int) -> Array:
    z = np.asarray(z, float)
    dz = np.asarray(increments_z, float)
    centers = np.asarray(centers, float)
    dim = z.shape[1]
    out = np.zeros((centers.shape[0], dim, dim))
    k_eff = min(max(int(k), dim + 3), z.shape[0])
    for idx, center in enumerate(centers):
        dist = np.linalg.norm(z - center[None, :], axis=1)
        nn = np.argpartition(dist, k_eff - 1)[:k_eff]
        radius = max(float(np.partition(dist, k_eff - 1)[k_eff - 1]) * float(multiplier), 1e-4)
        cov = np.cov(dz[nn], rowvar=False)
        cov = np.atleast_2d(cov)
        scale = max(float(np.trace(cov) / max(dim, 1)), 1e-12)
        metric = cov / scale
        out[idx] = (radius * radius) * metric + 1e-8 * np.eye(dim)
    return out


def _bandwidth_for_rule(
    z: Array,
    centers: Array,
    multiplier: float,
    rule: str,
    knn_k: int = 50,
    increments_z: Array | None = None,
) -> float | Array:
    rule_norm = rule.lower()
    if rule_norm in {"cov", "covariance", "anisotropic_cov"}:
        base = median_bandwidth(centers, multiplier=multiplier, rule="nn_median")
        cov = np.cov(np.asarray(z, float), rowvar=False)
        cov = np.atleast_2d(cov)
        dim = cov.shape[0]
        scale = max(float(np.trace(cov) / max(dim, 1)), 1e-12)
        metric = cov / scale
        return (base * base) * metric + 1e-8 * np.eye(dim)
    if rule_norm in {"knn", "adaptive_knn", "per_center"}:
        return knn_bandwidth(z, centers, multiplier=multiplier, k=knn_k)
    if rule_norm in {"local_cov", "local_covariance", "mahalanobis_local"}:
        return local_cov_bandwidth(z, centers, multiplier=multiplier, k=knn_k)
    if rule_norm in {"diffusion_metric", "qv_metric", "local_diffusion_metric"}:
        if increments_z is None:
            return local_cov_bandwidth(z, centers, multiplier=multiplier, k=knn_k)
        return _diffusion_metric_bandwidth(z, increments_z, centers, multiplier=multiplier, k=knn_k)
    if rule_norm in {"scott", "silverman"}:
        dim = max(np.asarray(z).shape[1], 1)
        n = max(np.asarray(z).shape[0], 1)
        factor = n ** (-1.0 / (dim + 4.0))
        if rule_norm == "silverman":
            factor *= (4.0 / (dim + 2.0)) ** (1.0 / (dim + 4.0))
        return median_bandwidth(centers, multiplier=multiplier * factor * max(len(centers) ** 0.5, 1.0), rule="nn_median")
    return median_bandwidth(centers, multiplier=multiplier, rule=rule)


def _scale_bandwidth(bandwidth: float | Array, scale: float) -> float | Array:
    bw = np.asarray(bandwidth, float)
    if bw.ndim in {0, 1}:
        return bandwidth * scale  # type: ignore[operator]
    return bandwidth * (scale * scale)  # covariance bandwidths carry squared length units.


def _make_projection(
    z: Array,
    centers: Array,
    bandwidth: float | Array,
    *,
    local_poly_order: int = 0,
    normalization: str = "row",
    multiscale: tuple[float, ...] | list[float] | None = None,
) -> Array:
    scales = tuple(multiscale or (1.0,))
    rows = []
    for scale in scales:
        bw = _scale_bandwidth(bandwidth, float(scale))
        if local_poly_order > 0:
            rows.append(local_polynomial_projection_matrix(z, centers, bw, order=local_poly_order, normalization=normalization))
        else:
            rows.append(gaussian_projection_matrix(z, centers, bw, normalization=normalization))
    return rows[0] if len(rows) == 1 else np.vstack(rows)


@dataclass
class GeneratorFit2D:
    state_names: list[str]
    library: Library
    library_space: str
    drift: Array
    diffusion: dict[tuple[int, int], Array]
    centers: Array
    bandwidth: float | Array
    bandwidth_meta: dict
    standardizer: Standardizer
    dt: float
    noise_cov: Array | None
    basis_change: Array | None
    raw_targets: dict = field(default_factory=dict)
    selections: dict[str, SelectionResult] = field(default_factory=dict)
    diffusion_parameterization: str = "entries"
    cholesky_diffusion: dict[str, Array] | None = None
    diffusion_shrinkage: float = 0.0
    diffusion_shrinkage_signal_aware: bool = False
    rank1_project: bool = False

    @property
    def drift_coef(self) -> Array:
        return self.drift

    @property
    def diff_coef(self) -> dict[tuple[int, int], Array]:
        return self.diffusion

    @property
    def dim(self) -> int:
        return self.drift.shape[1]

    def evaluate(self, raw_state: Array, psd: bool = False) -> tuple[Array, Array]:
        x = np.asarray(raw_state, float)
        if x.ndim == 1:
            x = x[None, :]
        theta = self.library.transform(x)
        b = theta @ self.drift
        if self.cholesky_diffusion is not None and self.dim == 2:
            l11 = theta @ self.cholesky_diffusion["l11"]
            l21 = theta @ self.cholesky_diffusion["l21"]
            l22 = theta @ self.cholesky_diffusion["l22"]
            if self.diffusion_parameterization in {"log_chol", "log_cholesky"}:
                l11 = np.exp(np.clip(l11, -30, 30))
                l22 = np.exp(np.clip(l22, -30, 30))
            else:
                l11 = np.maximum(l11, 1e-10)
                l22 = np.maximum(l22, 1e-10)
            a = np.zeros((x.shape[0], 2, 2))
            a[:, 0, 0] = l11 * l11
            a[:, 0, 1] = a[:, 1, 0] = l11 * l21
            a[:, 1, 1] = l21 * l21 + l22 * l22
        else:
            a = np.zeros((x.shape[0], self.dim, self.dim))
            for (i, j), coef in self.diffusion.items():
                v = theta @ coef
                a[:, i, j] = v
                a[:, j, i] = v
        if self.diffusion_shrinkage > 0.0 and self.dim == 2 and not self.diffusion_shrinkage_signal_aware:
            lam = float(np.clip(self.diffusion_shrinkage, 0.0, 1.0))
            trace = 0.5 * (a[:, 0, 0] + a[:, 1, 1])
            iso = np.zeros_like(a)
            iso[:, 0, 0] = trace
            iso[:, 1, 1] = trace
            a = (1.0 - lam) * a + lam * iso
        if self.rank1_project and self.dim == 2:
            projected = np.empty_like(a)
            for n, mat in enumerate(a):
                vals, vecs = np.linalg.eigh(0.5 * (mat + mat.T))
                vals[0] = 0.0
                projected[n] = (vecs * np.maximum(vals, 0.0)) @ vecs.T
            a = projected
        if psd:
            a = project_psd(a)
        return b, a

    def diffusion_at(self, raw_state: Array, psd_project: bool = False) -> Array:
        return self.evaluate(raw_state, psd=psd_project)[1]

    def drift_at(self, raw_state: Array) -> Array:
        return self.evaluate(raw_state)[0]


def _as_curr_increments(states: Array, increments: Array | None) -> tuple[Array, Array]:
    x = np.asarray(states, float)
    if x.ndim == 1:
        x = x[:, None]
    if increments is None:
        if x.shape[0] < 2:
            raise ValueError("need at least two states when increments is None")
        return x[:-1], np.diff(x, axis=0)
    inc = np.asarray(increments, float)
    if inc.ndim == 1:
        inc = inc[:, None]
    if x.shape[0] == inc.shape[0] + 1:
        x = x[:-1]
    if x.shape[0] != inc.shape[0]:
        raise ValueError("states and increments length mismatch")
    return x, inc


def fit_generator_2d(
    states: Array,
    increments: Array | None = None,
    dt: float = 0.01,
    library: Library | None = None,
    *,
    state_names: list[str] | tuple[str, ...] = ("x", "y"),
    center_scheme: str = "kmeans",
    n_centers: int = 100,
    grid_shape: tuple[int, int] | None = None,
    bandwidth_multiplier: float = 1.0,
    bandwidth_rule: str = "nn_median",
    knn_k: int = 50,
    local_poly_order: int = 0,
    projection_normalization: str = "row",
    projection_scales: tuple[float, ...] | list[float] | None = None,
    prune_min_effective_samples: float | None = None,
    target_anchor: str = "left",
    regressor: str | Callable[..., SelectionResult] = "stlsq",
    bias_correct: bool = True,
    noise_correct: bool = False,
    noise_cov: Array | None = None,
    library_space: str = "raw",
    seed: int = 0,
    traj_ids: Array | None = None,
    regression_kw: dict | None = None,
    target_regression_kw: dict[str, dict] | None = None,
    gls_weighting: bool = False,
    gls_iterations: int = 1,
    gls_clip: tuple[float, float] = (0.05, 20.0),
    gls_mode: str = "diagonal",
    gls_cond_cap: float = 1e4,
    gls_corr_floor: float = 0.15,
    gls_converge: bool = False,
    gls_tol: float = 1e-3,
    gls_crossfit_weights: bool = False,
    selection_noise_floor: bool = False,
    noise_floor_z: float = 1.0,
    coverage_weighting: bool = False,
    coverage_weight_floor: float = 0.25,
    diffusion_parameterization: str = "entries",
    diffusion_shrinkage: float = 0.0,
    rank1_project: bool = False,
    coord_transform: str = "none",
    drift_lags: tuple[int, ...] = (1,),
    lag_bias_correct: bool = False,
    moment_order: str = "euler",
    tensor_rank: str = "full",
    rank_floor: float = 1e-3,
    coverage_mode: str = "off",
    library_atoms: str = "poly",
    domain: str = "euclidean",
    precomputed_projection: tuple | None = None,
    return_projection: bool = False,
) -> GeneratorFit2D:
    x, inc_full = _as_curr_increments(states, increments)
    dim = x.shape[1]
    is_multi_lag = inc_full.ndim == 3 and len(drift_lags) > 1
    if is_multi_lag:
        inc = inc_full[:, 0, :]
        dt_base = dt * drift_lags[0]
    else:
        inc = inc_full
        dt_base = dt
    anchor = target_anchor.lower()
    if anchor in {"mid", "midpoint", "stratonovich"}:
        x_fit = x + 0.5 * inc
    elif anchor in {"right", "endpoint"}:
        x_fit = x + inc
    elif anchor in {"avg", "endpoint_avg"}:
        x_fit = 0.5 * (x + x + inc)
    else:
        x_fit = x
    names = list(state_names)[:dim]
    if library is None:
        library = make_library("A", names if dim == 2 else ("x", "y"))
    regression_kw = {} if regression_kw is None else dict(regression_kw)
    if selection_noise_floor:
        regression_kw["threshold_mode"] = "noise_floor"
        regression_kw["noise_floor_z"] = float(noise_floor_z)
    target_regression_kw = {} if target_regression_kw is None else {str(k): dict(v) for k, v in target_regression_kw.items()}
    moment_order_norm = moment_order.lower()
    if moment_order_norm not in {"euler", "milstein", "ito15"}:
        raise ValueError("moment_order must be 'euler', 'milstein', or 'ito15'")
    tensor_rank_norm = tensor_rank.lower()
    if tensor_rank_norm not in {"full", "auto"}:
        raise ValueError("tensor_rank must be 'full' or 'auto'")
    coverage_mode_norm = coverage_mode.lower()
    if coverage_mode_norm not in {"off", "longer", "reweight", "both"}:
        raise ValueError("coverage_mode must be 'off', 'longer', 'reweight', or 'both'")
    domain_norm = domain.lower()
    if domain_norm not in {"euclidean", "positive_log"}:
        raise ValueError("domain must be 'euclidean' or 'positive_log'")
    coord_transform_norm = coord_transform.lower()
    if coord_transform_norm not in {"none", "lamperti"}:
        raise ValueError("coord_transform must be 'none' or 'lamperti'")
    if coverage_mode_norm in {"reweight", "both"}:
        coverage_weighting = True

    if precomputed_projection is not None:
        std, z, centers, bandwidth, projection = precomputed_projection
    else:
        std = Standardizer().fit(x_fit)
        z = std.transform(x_fit)
        centers = choose_centers(z, n_centers=n_centers, seed=seed, scheme=center_scheme, grid_shape=grid_shape)
        inc_z = inc / np.maximum(std.scale[:dim], 1e-12)
        bandwidth = _bandwidth_for_rule(z, centers, bandwidth_multiplier, bandwidth_rule, knn_k=knn_k, increments_z=inc_z)
        projection = _make_projection(
            z,
            centers,
            bandwidth,
            local_poly_order=local_poly_order,
            normalization=projection_normalization,
            multiscale=projection_scales,
        )
    prune_meta = {}
    if prune_min_effective_samples is not None:
        min_rows = None
        projection, centers, prune_meta = prune_projection(
            projection,
            centers,
            min_effective_samples=float(prune_min_effective_samples),
            min_rows=min_rows,
        )

    if library_space == "raw":
        theta = library.transform(x_fit)
        basis_change = None
    elif library_space == "z":
        theta = library.transform(z)
        degree = {"A": 2, "POLY2": 2, "D": 2, "LOG_HESTON": 2, "B": 3, "POLY3": 3, "C": 4, "POLY4": 4}.get(library.preset.upper())
        basis_change = polynomial_change_of_basis(std.mean, std.scale, degree) if degree is not None and dim == 2 else None
    else:
        raise ValueError("library_space must be 'raw' or 'z'")
    aggregate_design = projection @ theta
    design = aggregate_design
    fit_groups = None
    group_masks: list[Array] | None = None
    if traj_ids is not None:
        sample_groups = np.asarray(traj_ids)
        if sample_groups.shape[0] != x.shape[0]:
            raise ValueError("traj_ids must have one entry per current-state/increment row")
        unique_groups = np.unique(sample_groups)
        if unique_groups.size > 1:
            group_masks = []
            grouped_design = []
            grouped_labels = []
            for group in unique_groups:
                mask = sample_groups == group
                group_masks.append(mask)
                part = projection[:, mask]
                grouped_design.append(part @ theta[mask])
                grouped_labels.append(np.full(part.shape[0], group))
            design = np.vstack(grouped_design)
            fit_groups = np.concatenate(grouped_labels)

    row_coverage_weight: Array | None = None
    if coverage_weighting:
        weights_sq = projection * projection
        if group_masks is None:
            eff = 1.0 / np.maximum(np.sum(weights_sq, axis=1), 1e-12)
        else:
            eff = np.concatenate([1.0 / np.maximum(np.sum(weights_sq[:, mask], axis=1), 1e-12) for mask in group_masks])
        med = np.median(eff[np.isfinite(eff)]) if np.any(np.isfinite(eff)) else 1.0
        row_coverage_weight = np.sqrt(np.maximum(eff, 1e-12) / max(med, 1e-12))
        floor = float(np.clip(coverage_weight_floor, 1e-6, 1.0))
        row_coverage_weight = np.clip(row_coverage_weight, floor, 1.0 / floor)
        if coverage_mode_norm in {"reweight", "both"}:
            density = np.sum(projection, axis=1)
            if group_masks is not None:
                density = np.concatenate([density for _ in group_masks])
            density = density / max(float(np.nanmedian(density[np.isfinite(density)])) if np.any(np.isfinite(density)) else 1.0, 1e-12)
            inv_density = 1.0 / np.maximum(density, 1e-12)
            inv_density = np.clip(inv_density, floor, 1.0 / floor)
            row_coverage_weight = row_coverage_weight * inv_density

    def _project_values(values: Array) -> Array:
        values = np.asarray(values, float)
        if group_masks is None:
            return projection @ values
        return np.concatenate([projection[:, mask] @ values[mask] for mask in group_masks])

    def _project_sq_values(values: Array) -> Array:
        values = np.asarray(values, float)
        weights = projection * projection
        if group_masks is None:
            return weights @ values
        return np.concatenate([weights[:, mask] @ values[mask] for mask in group_masks])

    def _project_sq_tensor(values: Array, *, crossfit: bool = False) -> Array:
        values = np.asarray(values, float)
        weights = projection * projection
        if group_masks is None:
            return np.einsum("mn,nij->mij", weights, values)
        chunks = []
        for mask in group_masks:
            use_mask = ~mask if crossfit and np.any(~mask) else mask
            chunks.append(np.einsum("mn,nij->mij", weights[:, use_mask], values[use_mask]))
        return np.vstack(chunks)

    def _solve(target: Array, key: str, offset: int, row_weight: Array | None = None, custom_design: Array | None = None) -> SelectionResult:
        solve_design = design if custom_design is None else custom_design
        solve_target = target
        if row_coverage_weight is not None or row_weight is not None:
            if row_weight is None:
                w = np.asarray(row_coverage_weight, float)
            elif row_coverage_weight is None:
                w = np.asarray(row_weight, float)
            else:
                w = np.asarray(row_weight, float) * np.asarray(row_coverage_weight, float)
            if len(w) < len(target):
                repeats = len(target) // len(w)
                w = np.tile(w, repeats)
            solve_design = solve_design * w[:, None]
            solve_target = target * w
        kw = dict(regression_kw)
        kw.update(target_regression_kw.get(key, {}))
        if key.startswith("chol_"):
            kw.update(target_regression_kw.get("chol", {}))
        if callable(regressor):
            res = regressor(solve_design, solve_target, groups=fit_groups, seed=seed + offset, **kw)
        else:
            res = solve(solve_design, solve_target, groups=fit_groups, method=str(regressor), seed=seed + offset, **kw)
        if target_regression_kw.get(key):
            res.diagnostics["target_regression_kw"] = dict(target_regression_kw[key])
        res.diagnostics["target_key"] = key
        return res

    drift = np.zeros((theta.shape[1], dim))
    selections: dict[str, SelectionResult] = {}
    drift_targets: dict[str, Array] = {}
    aggregate_drift_targets: dict[str, Array] = {}
    
    if is_multi_lag:
        n_lags = len(drift_lags)
        tau = np.array([dt * l for l in drift_lags])
        for p in range(dim):
            stacked_t = []
            stacked_d = []
            agg_t = []
            for k in range(n_lags):
                t_k = inc_full[:, k, p] / tau[k]
                stacked_t.append(_project_values(t_k))
                agg_t.append(projection @ t_k)
                if lag_bias_correct:
                    stacked_d.append(np.hstack([design, 0.5 * tau[k] * design]))
                else:
                    stacked_d.append(design)
            target = np.concatenate(stacked_t)
            solve_d = np.vstack(stacked_d)
            if p == 0:
                drift_design = solve_d
            
            res = _solve(target, f"b{p+1}", p, custom_design=solve_d)
            if lag_bias_correct:
                n_terms = theta.shape[1]
                drift[:, p] = res.coef[:n_terms]
            else:
                drift[:, p] = res.coef
            selections[f"b{p+1}"] = res
            drift_targets[f"b{p+1}"] = target
            aggregate_drift_targets[f"b{p+1}"] = np.concatenate(agg_t)
    else:
        drift_design = design
        for p in range(dim):
            target = _project_values(inc[:, p] / dt_base)
            res = _solve(target, f"b{p+1}", p)
            drift[:, p] = res.coef
            selections[f"b{p+1}"] = res
            drift_targets[f"b{p+1}"] = target
            aggregate_drift_targets[f"b{p+1}"] = projection @ (inc[:, p] / dt_base)
    b_hat = theta @ drift

    def _solve_diffusion(current_b_hat: Array) -> tuple[dict[tuple[int, int], Array], dict[str, Array], dict[str, Array], Array]:
        nonlocal noise_cov
        outer_raw = inc[:, :, None] * inc[:, None, :]
        if noise_correct:
            if noise_cov is None:
                full = np.vstack([x[:1], x + inc])
                noise_cov = lag1_noise_covariance(full)
            outer_raw = outer_raw - 2.0 * np.asarray(noise_cov, float)[None, :, :]
        if bias_correct:
            bb = current_b_hat[:, :, None] * current_b_hat[:, None, :]
            bias_scale = 1.0
            if moment_order_norm == "milstein":
                bias_scale = 0.5
            elif moment_order_norm == "ito15":
                bias_scale = 0.25
            outer_raw = outer_raw - bias_scale * bb * (dt_base * dt_base)
        outer = outer_raw / dt_base
        out_diffusion: dict[tuple[int, int], Array] = {}
        out_targets: dict[str, Array] = {}
        out_agg_targets: dict[str, Array] = {}
        for i in range(dim):
            for j in range(i, dim):
                key = f"a{i+1}{j+1}"
                target = _project_values(outer[:, i, j])
                res = _solve(target, key, 100 + 13 * i + j)
                out_diffusion[(i, j)] = res.coef
                selections[key] = res
                out_targets[key] = target
                out_agg_targets[key] = projection @ outer[:, i, j]
        return out_diffusion, out_targets, out_agg_targets, outer

    diffusion, diff_targets, aggregate_diff_targets, outer = _solve_diffusion(b_hat)

    rank_meta: dict[str, object] = {}

    def _apply_rank_auto() -> None:
        nonlocal diffusion, rank_meta
        if tensor_rank_norm != "auto" or dim != 2:
            rank_meta = {"tensor_rank_active": False}
            return
        entry_scale = {
            key: float(np.nanmedian(np.abs(theta @ coef))) if coef.size else 0.0
            for key, coef in diffusion.items()
        }
        trace_scale = max(entry_scale.get((0, 0), 0.0) + entry_scale.get((1, 1), 0.0), 1e-12)
        threshold = float(rank_floor) * trace_scale
        zeroed: list[str] = []
        for key in list(diffusion):
            if entry_scale.get(key, 0.0) <= threshold:
                diffusion[key] = np.zeros_like(diffusion[key])
                zeroed.append(f"a{key[0]+1}{key[1]+1}")
                if f"a{key[0]+1}{key[1]+1}" in selections:
                    selections[f"a{key[0]+1}{key[1]+1}"].coef[:] = 0.0
                    selections[f"a{key[0]+1}{key[1]+1}"].support[:] = False
                    selections[f"a{key[0]+1}{key[1]+1}"].diagnostics["rank_auto_zeroed"] = True
        rank_meta = {"tensor_rank_active": True, "rank_floor": float(rank_floor), "rank_zeroed_entries": zeroed}

    _apply_rank_auto()

    gls_iters_completed = 0
    if gls_weighting and dim >= 1:
        n_gls = max(1, int(gls_iterations))
        mode = gls_mode.lower()
        if mode not in {"diagonal", "full_tensor"}:
            raise ValueError("gls_mode must be 'diagonal' or 'full_tensor'")

        def _local_gls_weights(a_train: Array) -> Array:
            lo, hi = gls_clip
            out = np.ones((design.shape[0], dim))
            for p in range(dim):
                app = np.maximum(a_train[:, p, p], 1e-10)
                row_var = _project_sq_values(app) / max(dt_base, 1e-12)
                out[:, p] = 1.0 / np.sqrt(np.maximum(row_var, 1e-12))
            for p in range(dim):
                finite = out[:, p][np.isfinite(out[:, p])]
                med = np.median(finite) if finite.size else 1.0
                out[:, p] = np.clip(out[:, p] / max(med, 1e-12), lo, hi)
            return out

        def _local_gls_whiteners(a_train: Array) -> Array:
            lo, hi = gls_clip
            cov_rows = _project_sq_tensor(a_train, crossfit=bool(gls_crossfit_weights)) / max(dt_base, 1e-12)
            cond_cap = max(float(gls_cond_cap), 1.0)
            whiteners = np.zeros((design.shape[0], dim, dim))
            use_offdiag = True
            if dim == 2 and gls_corr_floor > 0.0:
                den = np.sqrt(np.maximum(cov_rows[:, 0, 0], 0.0) * np.maximum(cov_rows[:, 1, 1], 0.0))
                corr_rows = np.divide(cov_rows[:, 0, 1], den, out=np.zeros(cov_rows.shape[0]), where=den > 1e-15)
                finite_corr = np.abs(corr_rows[np.isfinite(corr_rows)])
                use_offdiag = bool(finite_corr.size and np.median(finite_corr) >= float(gls_corr_floor))
            for row_idx, cov in enumerate(cov_rows):
                sym = 0.5 * (cov + cov.T)
                if dim == 2 and gls_corr_floor > 0.0:
                    den = np.sqrt(max(sym[0, 0], 0.0) * max(sym[1, 1], 0.0))
                    corr = sym[0, 1] / den if den > 1e-15 else 0.0
                    if (not use_offdiag) or abs(corr) < float(gls_corr_floor):
                        sym[0, 1] = sym[1, 0] = 0.0
                vals, vecs = np.linalg.eigh(sym)
                vals = np.maximum(vals, 1e-10)
                vals = np.maximum(vals, np.max(vals) / cond_cap)
                whiteners[row_idx] = (vecs * (1.0 / np.sqrt(vals))) @ vecs.T
            if row_coverage_weight is not None:
                whiteners = whiteners * np.asarray(row_coverage_weight, float)[:, None, None]
            for q in range(dim):
                norms = np.linalg.norm(whiteners[:, q, :], axis=1)
                finite = norms[np.isfinite(norms)]
                med = np.median(finite) if finite.size else 1.0
                desired = np.clip(norms / max(med, 1e-12), lo, hi)
                scale = np.divide(desired, np.maximum(norms, 1e-12), out=np.ones_like(norms), where=norms > 1e-12)
                whiteners[:, q, :] *= scale[:, None]
            return whiteners

        def _solve_joint_drift(target_matrix: Array, whitening: Array, offset: int) -> SelectionResult:
            solve_whitening = whitening
            if solve_whitening.shape[0] < target_matrix.shape[0]:
                repeats = target_matrix.shape[0] // solve_whitening.shape[0]
                solve_whitening = np.tile(solve_whitening, (repeats, 1, 1))
            n_rows, n_terms = drift_design.shape
            stacked_design = np.zeros((n_rows * dim, n_terms * dim))
            stacked_target = np.einsum("rqp,rp->rq", solve_whitening, target_matrix).reshape(n_rows * dim)
            for p in range(dim):
                block = slice(p * n_terms, (p + 1) * n_terms)
                for q in range(dim):
                    stacked_design[q::dim, block] += solve_whitening[:, q, p, None] * drift_design
            joint_groups = None if fit_groups is None else np.repeat(fit_groups, dim)
            kw = dict(regression_kw)
            kw.update(target_regression_kw.get("drift", {}))
            kw.update(target_regression_kw.get("b_joint", {}))
            if callable(regressor):
                res = regressor(stacked_design, stacked_target, groups=joint_groups, seed=seed + offset, **kw)
            else:
                res = solve(stacked_design, stacked_target, groups=joint_groups, method=str(regressor), seed=seed + offset, **kw)
            res.diagnostics["target_key"] = "b_joint"
            res.diagnostics["joint_drift"] = True
            res.diagnostics["joint_design_shape"] = list(stacked_design.shape)
            return res

        for gls_iter in range(n_gls):
            drift_prev = drift.copy()
            a_train = np.zeros((theta.shape[0], dim, dim))
            for (i, j), coef in diffusion.items():
                val = theta @ coef
                a_train[:, i, j] = val
                a_train[:, j, i] = val
            if mode == "full_tensor" and dim > 1:
                whitening = _local_gls_whiteners(a_train)
                target_matrix = np.column_stack([drift_targets[f"b{p+1}"] for p in range(dim)])
                joint_res = _solve_joint_drift(target_matrix, whitening, 300 + 31 * gls_iter)
                n_terms = theta.shape[1]
                n_terms_design = drift_design.shape[1]
                joint_coef = joint_res.coef.reshape(dim, n_terms_design).T
                drift[:, :] = joint_coef[:n_terms, :]
                for p in range(dim):
                    block = slice(p * n_terms_design, p * n_terms_design + n_terms)
                    res = SelectionResult(
                        coef=joint_res.coef[block].copy(),
                        support=joint_res.support[block].copy(),
                        alpha=joint_res.alpha,
                        coef_normalized=joint_res.coef_normalized[block].copy(),
                        column_scale=joint_res.column_scale[block].copy(),
                        method=f"{joint_res.method}_joint_drift",
                        diagnostics=dict(joint_res.diagnostics),
                    )
                    res.diagnostics["gls_weighting"] = True
                    res.diagnostics["gls_mode"] = mode
                    res.diagnostics["gls_full_tensor_joint"] = True
                    res.diagnostics["gls_crossfit_weights"] = bool(gls_crossfit_weights)
                    res.diagnostics["gls_iteration"] = gls_iter + 1
                    res.diagnostics["gls_iterations_requested"] = n_gls
                    res.diagnostics["target_key"] = f"b{p+1}"
                    selections[f"b{p+1}"] = res
            else:
                weights_by_component = _local_gls_weights(a_train)
                for p in range(dim):
                    row_weight = weights_by_component[:, p]
                    target = drift_targets[f"b{p+1}"]
                    res = _solve(target, f"b{p+1}", 300 + 31 * gls_iter + p, row_weight=row_weight, custom_design=drift_design)
                    if is_multi_lag and lag_bias_correct:
                        n_terms = theta.shape[1]
                        drift[:, p] = res.coef[:n_terms]
                    else:
                        drift[:, p] = res.coef
                    res.diagnostics["gls_weighting"] = True
                    res.diagnostics["gls_mode"] = mode
                    res.diagnostics["gls_full_tensor_joint"] = False
                    res.diagnostics["gls_crossfit_weights"] = bool(gls_crossfit_weights)
                    res.diagnostics["gls_iteration"] = gls_iter + 1
                    res.diagnostics["gls_iterations_requested"] = n_gls
                    selections[f"b{p+1}"] = res
            b_hat = theta @ drift
            diffusion, diff_targets, aggregate_diff_targets, outer = _solve_diffusion(b_hat)
            _apply_rank_auto()
            gls_iters_completed = gls_iter + 1
            delta = float(np.linalg.norm(drift - drift_prev) / max(np.linalg.norm(drift_prev), 1e-12))
            for p in range(dim):
                selections[f"b{p+1}"].diagnostics["gls_rel_change"] = delta
                selections[f"b{p+1}"].diagnostics["gls_converged"] = bool(gls_converge and delta < gls_tol)
            if gls_converge and delta < gls_tol:
                break

    cholesky_diffusion: dict[str, Array] | None = None
    param_norm = diffusion_parameterization.lower()
    if dim == 2 and param_norm in {"chol", "cholesky", "log_chol", "log_cholesky", "joint_psd", "spectral"}:
        a_train = np.zeros((theta.shape[0], 2, 2))
        for (i, j), coef in diffusion.items():
            val = theta @ coef
            a_train[:, i, j] = val
            a_train[:, j, i] = val
        a_train = project_psd(a_train, eps=1e-10)
        l11 = np.sqrt(np.maximum(a_train[:, 0, 0], 1e-10))
        l21 = a_train[:, 0, 1] / np.maximum(l11, 1e-10)
        l22 = np.sqrt(np.maximum(a_train[:, 1, 1] - l21 * l21, 1e-10))
        if param_norm in {"log_chol", "log_cholesky"}:
            targets = {"l11": np.log(l11), "l21": l21, "l22": np.log(l22)}
        else:
            targets = {"l11": l11, "l21": l21, "l22": l22}
        cholesky_diffusion = {}
        for idx, (key, values) in enumerate(targets.items()):
            target = _project_values(values)
            res = _solve(target, f"chol_{key}", 500 + idx)
            cholesky_diffusion[key] = res.coef
            res.diagnostics["diffusion_parameterization"] = param_norm
            selections[f"chol_{key}"] = res

    if library_space == "z" and basis_change is not None:
        for p in range(dim):
            drift[:, p] = basis_change @ drift[:, p]
        for key in list(diffusion):
            diffusion[key] = basis_change @ diffusion[key]
        if cholesky_diffusion is not None:
            for key in list(cholesky_diffusion):
                cholesky_diffusion[key] = basis_change @ cholesky_diffusion[key]

    meta = {
        "center_scheme": center_scheme,
        "n_centers": int(centers.shape[0]),
        "bandwidth_multiplier": bandwidth_multiplier,
        "bandwidth_rule": bandwidth_rule,
        "knn_k": int(knn_k),
        "local_poly_order": int(local_poly_order),
        "projection_normalization": projection_normalization,
        "projection_scales": list(projection_scales or [1.0]),
        "design_shape": list(design.shape),
        "cond_design": float(np.linalg.cond(design)) if design.size else float("nan"),
        "seed": seed,
        "bias_correct": bool(bias_correct),
        "noise_correct": bool(noise_correct),
        "target_anchor": target_anchor,
        "gls_weighting": bool(gls_weighting),
        "gls_iterations": int(gls_iters_completed),
        "gls_mode": gls_mode.lower(),
        "gls_cond_cap": float(gls_cond_cap),
        "gls_corr_floor": float(gls_corr_floor),
        "gls_converge": bool(gls_converge),
        "gls_tol": float(gls_tol),
        "gls_crossfit_weights": bool(gls_crossfit_weights),
        "selection_noise_floor": bool(selection_noise_floor),
        "noise_floor_z": float(noise_floor_z),
        "coverage_weighting": bool(coverage_weighting),
        "coverage_weight_floor": float(coverage_weight_floor),
        "coverage_mode": coverage_mode_norm,
        "coord_transform": coord_transform_norm,
        "drift_lags": [int(v) for v in drift_lags],
        "lag_bias_correct": bool(lag_bias_correct),
        "moment_order": moment_order_norm,
        "tensor_rank": tensor_rank_norm,
        "rank_floor": float(rank_floor),
        "library_atoms": library_atoms,
        "domain": domain_norm,
        "diffusion_parameterization": param_norm,
        "n_trajectories": int(np.unique(traj_ids).size) if traj_ids is not None else 1,
        "projected_group_folds": int(np.unique(fit_groups).size) if fit_groups is not None else 0,
        **prune_meta,
        **rank_meta,
    }
    fit = GeneratorFit2D(
        state_names=names,
        library=library,
        library_space=library_space,
        drift=drift,
        diffusion=diffusion,
        centers=centers,
        bandwidth=bandwidth,
        bandwidth_meta=meta,
        standardizer=std,
        dt=dt,
        noise_cov=None if noise_cov is None else np.asarray(noise_cov, float),
        basis_change=basis_change,
        raw_targets={
            "design": design,
            "drift": drift_targets,
            "diffusion": diff_targets,
            "outer": outer,
            "projection": projection,
            "aggregate_design": aggregate_design,
            "aggregate_drift": aggregate_drift_targets,
            "aggregate_diffusion": aggregate_diff_targets,
        },
        selections=selections,
        diffusion_parameterization=param_norm,
        cholesky_diffusion=cholesky_diffusion,
        diffusion_shrinkage=diffusion_shrinkage,
        diffusion_shrinkage_signal_aware=bool(selection_noise_floor),
        rank1_project=rank1_project,
    )
    if return_projection:
        return fit, (std, z, centers, bandwidth, projection)
    return fit
