from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import warnings

import numpy as np

Array = np.ndarray
ALPHA_GRID = np.logspace(-8.0, -0.5, 60)
L1_RATIO_GRID = (0.2, 0.5, 0.8, 0.95)


@dataclass
class SelectionResult:
    coef: Array
    support: Array
    alpha: float
    coef_normalized: Array
    column_scale: Array
    method: str
    diagnostics: dict = field(default_factory=dict)


class ColumnNormalizer:
    def fit(self, x: Array) -> "ColumnNormalizer":
        self.scale = np.linalg.norm(np.asarray(x, float), axis=0)
        self.scale = np.where(self.scale == 0, 1.0, self.scale)
        return self

    def transform(self, x: Array) -> Array:
        return np.asarray(x, float) / self.scale

    def coefficients_to_raw(self, coef_normed: Array) -> Array:
        return np.asarray(coef_normed, float) / self.scale


def _design_diag(x: Array) -> dict:
    x = np.asarray(x, float)
    rank = int(np.linalg.matrix_rank(x)) if x.size else 0
    try:
        cond = float(np.linalg.cond(x.T @ x))
    except Exception:
        cond = float("inf")
    return {"rank": rank, "condition_number": cond, "rank_deficient": bool(rank < x.shape[1])}


def _ridge_lstsq(x: Array, y: Array, ridge_floor: float = 1e-10) -> Array:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n_cols = x.shape[1]
    gram = x.T @ x
    lam = ridge_floor * max(float(np.mean(np.diag(gram))), 1e-30)
    try:
        return np.linalg.solve(gram + lam * np.eye(n_cols), x.T @ y)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(x, y, rcond=None)[0]


def _svd_lstsq(x: Array, y: Array, rtol: float = 1e-8) -> tuple[Array, dict]:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    u, s, vt = np.linalg.svd(x, full_matrices=False)
    if s.size == 0:
        return np.zeros(x.shape[1]), {"svd_rank": 0, "svd_min_kept": float("nan"), "svd_max": float("nan")}
    keep = s > float(rtol) * max(float(s[0]), 1e-30)
    if not np.any(keep):
        return np.zeros(x.shape[1]), {"svd_rank": 0, "svd_min_kept": float("nan"), "svd_max": float(s[0])}
    coef = vt[keep].T @ ((u[:, keep].T @ y) / s[keep])
    return coef, {"svd_rank": int(np.sum(keep)), "svd_min_kept": float(np.min(s[keep])), "svd_max": float(s[0]), "svd_rtol": float(rtol)}


def _pseudo_block_groups(n_rows: int, n_blocks: int) -> Array:
    n_blocks = min(max(int(n_blocks), 2), int(n_rows))
    edges = np.linspace(0, n_rows, n_blocks + 1, dtype=int)
    groups = np.empty(n_rows, dtype=int)
    for block in range(n_blocks):
        groups[edges[block] : edges[block + 1]] = block
    return groups


def _cv_groups(
    groups: Array | None,
    n_rows: int,
    *,
    n_splits: int,
    pseudo_blocks: int,
) -> tuple[Array | None, dict]:
    n_input_groups = 0
    if groups is not None and len(groups) == n_rows:
        groups = np.asarray(groups)
        n_input_groups = int(np.unique(groups).size)
        if n_input_groups >= 2:
            return groups, {"n_groups_input": n_input_groups, "cv_pseudo_blocks": False}
    if n_rows >= 4 and pseudo_blocks >= 2:
        out = _pseudo_block_groups(n_rows, min(pseudo_blocks, n_rows))
        return out, {"n_groups_input": n_input_groups, "cv_pseudo_blocks": True, "pseudo_blocks": int(np.unique(out).size)}
    return None, {"n_groups_input": n_input_groups, "cv_pseudo_blocks": False, "cv_fallback": True}


def debiased_ols_no_intercept(x: Array, y: Array, selected: Array) -> Array:
    coef = np.zeros(x.shape[1])
    selected = np.asarray(selected, bool)
    if np.any(selected):
        coef[selected] = np.linalg.lstsq(x[:, selected], y, rcond=None)[0]
    return coef


def coefficient_standard_errors(x: Array, y: Array, coef: Array, ridge_floor: float = 1e-12) -> Array:
    """OLS-style coefficient standard errors for the already-weighted design."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    coef = np.asarray(coef, float)
    resid = y - x @ coef
    n_rows, n_cols = x.shape
    dof = max(n_rows - int(np.count_nonzero(np.isfinite(coef))), 1)
    sigma2 = float(np.sum(resid * resid) / dof)
    gram = x.T @ x
    lam = ridge_floor * max(float(np.mean(np.diag(gram))) if gram.size else 1.0, 1e-30)
    cov = sigma2 * np.linalg.pinv(gram + lam * np.eye(n_cols))
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    return np.where(np.isfinite(se), np.maximum(se, 1e-15), np.inf)


def stlsq(
    x_normed: Array,
    y: Array,
    coef0: Array,
    tau: float = 0.25,
    n_iter: int = 20,
    threshold_mode: str = "relative",
    standard_errors: Array | None = None,
    noise_floor_z: float = 1.0,
) -> tuple[Array, int]:
    coef = np.asarray(coef0, float).copy()
    iters = 0
    threshold_mode = threshold_mode.lower()
    se = None if standard_errors is None else np.asarray(standard_errors, float)

    def active_mask(v: Array) -> Array:
        if threshold_mode == "absolute":
            return np.abs(v) >= tau
        if threshold_mode == "relative":
            scale = max(float(np.max(np.abs(v))), 1e-12)
            return np.abs(v) >= tau * scale
        if threshold_mode == "noise_floor":
            local_se = coefficient_standard_errors(x_normed, y, v) if se is None else se
            return np.abs(v) >= float(noise_floor_z) * np.maximum(local_se, 1e-15)
        raise ValueError("threshold_mode must be 'relative', 'absolute', or 'noise_floor'")

    for it in range(n_iter):
        iters = it + 1
        active = active_mask(coef)
        if not np.any(active):
            return np.zeros_like(coef), iters
        refined = np.zeros_like(coef)
        refined[active] = np.linalg.lstsq(x_normed[:, active], y, rcond=None)[0]
        new_active = active_mask(refined)
        coef = refined
        if np.array_equal(active, new_active):
            break
    coef[~active_mask(coef)] = 0.0
    return coef, iters


def grouped_lassocv(x_normed: Array, y: Array, groups: Array, n_splits: int = 5, seed: int = 0) -> tuple[Array, float, dict]:
    from sklearn.linear_model import LassoCV
    from sklearn.model_selection import GroupKFold
    from sklearn.exceptions import ConvergenceWarning

    groups = np.asarray(groups)
    n_groups = int(np.unique(groups).size)
    folds = min(n_splits, n_groups)
    cv = GroupKFold(n_splits=folds)
    lasso = LassoCV(
        alphas=ALPHA_GRID,
        cv=cv.split(x_normed, y, groups),
        fit_intercept=False,
        max_iter=50000,
        random_state=seed,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        lasso.fit(x_normed, y)
    diag = {
        "cv_folds": folds,
        "n_groups": n_groups,
        "cv_alpha_path": [float(a) for a in lasso.alphas_],
        "cv_mse_min": float(np.min(lasso.mse_path_.mean(axis=1))),
        "convergence_warnings_suppressed": True,
    }
    return np.abs(lasso.coef_) > 1e-8, float(lasso.alpha_), diag


def grouped_elasticnetcv(
    x_normed: Array,
    y: Array,
    groups: Array,
    n_splits: int = 5,
    seed: int = 0,
    l1_ratio_grid: tuple[float, ...] | list[float] = L1_RATIO_GRID,
) -> tuple[Array, float, dict]:
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import ElasticNetCV
    from sklearn.model_selection import GroupKFold

    groups = np.asarray(groups)
    n_groups = int(np.unique(groups).size)
    folds = min(n_splits, n_groups)
    cv = GroupKFold(n_splits=folds)
    enet = ElasticNetCV(
        l1_ratio=list(l1_ratio_grid),
        alphas=ALPHA_GRID,
        cv=cv.split(x_normed, y, groups),
        fit_intercept=False,
        max_iter=50000,
        random_state=seed,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        enet.fit(x_normed, y)
    mse_path = np.asarray(enet.mse_path_)
    diag = {
        "cv_folds": folds,
        "n_groups": n_groups,
        "cv_alpha_path": [float(a) for a in enet.alphas_],
        "cv_mse_min": float(np.min(mse_path)),
        "l1_ratio": float(enet.l1_ratio_),
        "l1_ratio_grid": [float(v) for v in l1_ratio_grid],
        "convergence_warnings_suppressed": True,
    }
    return np.abs(enet.coef_) > 1e-8, float(enet.alpha_), diag


def grouped_lassocv_coefficients(x_normed: Array, y: Array, groups: Array, n_splits: int = 5, seed: int = 0) -> tuple[Array, float, dict]:
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LassoCV
    from sklearn.model_selection import GroupKFold

    groups = np.asarray(groups)
    n_groups = int(np.unique(groups).size)
    folds = min(n_splits, n_groups)
    cv = GroupKFold(n_splits=folds)
    lasso = LassoCV(
        alphas=ALPHA_GRID,
        cv=cv.split(x_normed, y, groups),
        fit_intercept=False,
        max_iter=50000,
        random_state=seed,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        lasso.fit(x_normed, y)
    diag = {
        "cv_folds": folds,
        "n_groups": n_groups,
        "cv_alpha_path": [float(a) for a in lasso.alphas_],
        "cv_mse_min": float(np.min(lasso.mse_path_.mean(axis=1))),
        "convergence_warnings_suppressed": True,
    }
    return np.asarray(lasso.coef_, float), float(lasso.alpha_), diag


def fit_lassocv_debias_stlsq(
    design: Array,
    target: Array,
    groups: Array | None,
    *,
    seed: int = 0,
    n_splits: int = 5,
    stlsq_threshold: float = 0.25,
    stlsq_iters: int = 20,
    threshold_mode: str = "relative",
    noise_floor_z: float = 1.0,
    ridge_floor: float = 1e-10,
    pseudo_blocks: int = 5,
    **_: object,
) -> SelectionResult:
    norm = ColumnNormalizer().fit(design)
    xs = norm.transform(design)
    y = np.asarray(target, float)
    n_rows = xs.shape[0]
    cv_groups, cv_diag = _cv_groups(groups, n_rows, n_splits=n_splits, pseudo_blocks=pseudo_blocks)
    if cv_groups is not None and int(np.unique(cv_groups).size) >= 2 and n_rows >= 4:
        selected, alpha, path_diag = grouped_lassocv(xs, y, cv_groups, n_splits=n_splits, seed=seed)
        cv_diag = {**cv_diag, **path_diag}
    else:
        coef0 = _ridge_lstsq(xs, y, ridge_floor)
        selected = np.abs(coef0) > 1e-8
        alpha = 0.0
        cv_diag = {**cv_diag, "cv_folds": 0, "n_groups": 0, "cv_fallback": True}
    debiased = debiased_ols_no_intercept(xs, y, selected)
    se = coefficient_standard_errors(xs, y, debiased, ridge_floor)
    coef_norm, n_it = stlsq(xs, y, debiased, tau=stlsq_threshold, n_iter=stlsq_iters, threshold_mode=threshold_mode, standard_errors=se, noise_floor_z=noise_floor_z)
    coef_raw = norm.coefficients_to_raw(coef_norm)
    support = coef_raw != 0.0
    return SelectionResult(
        coef=coef_raw,
        support=support,
        alpha=alpha,
        coef_normalized=coef_norm,
        column_scale=norm.scale.copy(),
        method="lassocv_debias_stlsq",
        diagnostics={
            **_design_diag(xs),
            **cv_diag,
            "n_rows": n_rows,
            "n_cols": xs.shape[1],
            "n_selected": int(support.sum()),
            "n_stlsq_iters": n_it,
            "threshold_mode": threshold_mode,
            "noise_floor_z": float(noise_floor_z),
            "se_min": float(np.nanmin(se)) if se.size else float("nan"),
            "se_max": float(np.nanmax(se)) if se.size else float("nan"),
        },
    )


def fit_elastic_net_debias_stlsq(
    design: Array,
    target: Array,
    groups: Array | None,
    *,
    seed: int = 0,
    n_splits: int = 5,
    l1_ratio_grid: tuple[float, ...] | list[float] = L1_RATIO_GRID,
    stlsq_threshold: float = 0.18,
    stlsq_iters: int = 20,
    threshold_mode: str = "relative",
    noise_floor_z: float = 1.0,
    ridge_floor: float = 1e-10,
    pseudo_blocks: int = 5,
    **_: object,
) -> SelectionResult:
    norm = ColumnNormalizer().fit(design)
    xs = norm.transform(design)
    y = np.asarray(target, float)
    n_rows = xs.shape[0]
    cv_groups, cv_diag = _cv_groups(groups, n_rows, n_splits=n_splits, pseudo_blocks=pseudo_blocks)
    if cv_groups is not None and int(np.unique(cv_groups).size) >= 2 and n_rows >= 4:
        selected, alpha, path_diag = grouped_elasticnetcv(
            xs,
            y,
            cv_groups,
            n_splits=n_splits,
            seed=seed,
            l1_ratio_grid=l1_ratio_grid,
        )
        cv_diag = {**cv_diag, **path_diag}
    else:
        coef0 = _ridge_lstsq(xs, y, ridge_floor)
        selected = np.abs(coef0) > 1e-8
        alpha = 0.0
        cv_diag = {**cv_diag, "cv_folds": 0, "n_groups": 0, "cv_fallback": True, "l1_ratio": float("nan")}
    debiased = debiased_ols_no_intercept(xs, y, selected)
    se = coefficient_standard_errors(xs, y, debiased, ridge_floor)
    coef_norm, n_it = stlsq(xs, y, debiased, tau=stlsq_threshold, n_iter=stlsq_iters, threshold_mode=threshold_mode, standard_errors=se, noise_floor_z=noise_floor_z)
    coef_raw = norm.coefficients_to_raw(coef_norm)
    support = coef_raw != 0.0
    return SelectionResult(
        coef=coef_raw,
        support=support,
        alpha=alpha,
        coef_normalized=coef_norm,
        column_scale=norm.scale.copy(),
        method="elastic_net_debias_stlsq",
        diagnostics={
            **_design_diag(xs),
            **cv_diag,
            "n_rows": n_rows,
            "n_cols": xs.shape[1],
            "n_selected": int(support.sum()),
            "n_stlsq_iters": n_it,
            "threshold_mode": threshold_mode,
            "noise_floor_z": float(noise_floor_z),
            "se_min": float(np.nanmin(se)) if se.size else float("nan"),
            "se_max": float(np.nanmax(se)) if se.size else float("nan"),
        },
    )


def fit_adaptive_lasso_debias_stlsq(
    design: Array,
    target: Array,
    groups: Array | None,
    *,
    seed: int = 0,
    n_splits: int = 5,
    gamma: float = 1.0,
    weight_floor: float = 1e-3,
    stlsq_threshold: float = 0.14,
    stlsq_iters: int = 20,
    threshold_mode: str = "relative",
    noise_floor_z: float = 1.0,
    ridge_floor: float = 1e-10,
    pseudo_blocks: int = 5,
    **_: object,
) -> SelectionResult:
    norm = ColumnNormalizer().fit(design)
    xs = norm.transform(design)
    y = np.asarray(target, float)
    n_rows = xs.shape[0]
    ridge_coef = _ridge_lstsq(xs, y, ridge_floor)
    penalty_weights = 1.0 / np.maximum(np.abs(ridge_coef), weight_floor) ** float(gamma)
    scaled = xs / penalty_weights
    cv_groups, cv_diag = _cv_groups(groups, n_rows, n_splits=n_splits, pseudo_blocks=pseudo_blocks)
    if cv_groups is not None and int(np.unique(cv_groups).size) >= 2 and n_rows >= 4:
        scaled_coef, alpha, path_diag = grouped_lassocv_coefficients(scaled, y, cv_groups, n_splits=n_splits, seed=seed)
        coef0 = scaled_coef / penalty_weights
        selected = np.abs(coef0) > 1e-8
        cv_diag = {**cv_diag, **path_diag}
    else:
        coef0 = ridge_coef
        selected = np.abs(coef0) > 1e-8
        alpha = 0.0
        cv_diag = {**cv_diag, "cv_folds": 0, "n_groups": 0, "cv_fallback": True}
    debiased = debiased_ols_no_intercept(xs, y, selected)
    se = coefficient_standard_errors(xs, y, debiased, ridge_floor)
    coef_norm, n_it = stlsq(xs, y, debiased, tau=stlsq_threshold, n_iter=stlsq_iters, threshold_mode=threshold_mode, standard_errors=se, noise_floor_z=noise_floor_z)
    coef_raw = norm.coefficients_to_raw(coef_norm)
    support = coef_raw != 0.0
    return SelectionResult(
        coef=coef_raw,
        support=support,
        alpha=alpha,
        coef_normalized=coef_norm,
        column_scale=norm.scale.copy(),
        method="adaptive_lasso_debias_stlsq",
        diagnostics={
            **_design_diag(xs),
            **cv_diag,
            "adaptive_gamma": float(gamma),
            "adaptive_weight_floor": float(weight_floor),
            "n_rows": n_rows,
            "n_cols": xs.shape[1],
            "n_selected": int(support.sum()),
            "n_stlsq_iters": n_it,
            "threshold_mode": threshold_mode,
            "noise_floor_z": float(noise_floor_z),
            "se_min": float(np.nanmin(se)) if se.size else float("nan"),
            "se_max": float(np.nanmax(se)) if se.size else float("nan"),
        },
    )


def fit_svd_threshold(
    design: Array,
    target: Array,
    groups: Array | None = None,
    *,
    threshold: float = 0.05,
    n_iter: int = 12,
    threshold_mode: str = "relative",
    noise_floor_z: float = 1.0,
    svd_rtol: float = 1e-8,
    **_: object,
) -> SelectionResult:
    norm = ColumnNormalizer().fit(design)
    xs = norm.transform(design)
    coef0, svd_diag = _svd_lstsq(xs, target, rtol=svd_rtol)
    se = coefficient_standard_errors(xs, target, coef0)
    coef_norm, n_it = stlsq(xs, target, coef0, tau=threshold, n_iter=n_iter, threshold_mode=threshold_mode, standard_errors=se, noise_floor_z=noise_floor_z)
    coef_raw = norm.coefficients_to_raw(coef_norm)
    return SelectionResult(
        coef_raw,
        coef_raw != 0.0,
        0.0,
        coef_norm,
        norm.scale.copy(),
        "svd_threshold",
        {**_design_diag(xs), **svd_diag, "n_selected": int(np.count_nonzero(coef_raw)), "n_stlsq_iters": n_it, "threshold_mode": threshold_mode, "noise_floor_z": float(noise_floor_z)},
    )


def fit_stlsq(
    design: Array,
    target: Array,
    groups: Array | None = None,
    *,
    threshold: float = 0.25,
    n_iter: int = 20,
    threshold_mode: str = "relative",
    noise_floor_z: float = 1.0,
    ridge_floor: float = 1e-10,
    **_: object,
) -> SelectionResult:
    norm = ColumnNormalizer().fit(design)
    xs = norm.transform(design)
    coef0 = _ridge_lstsq(xs, target, ridge_floor)
    se = coefficient_standard_errors(xs, target, coef0, ridge_floor)
    coef_norm, n_it = stlsq(xs, target, coef0, tau=threshold, n_iter=n_iter, threshold_mode=threshold_mode, standard_errors=se, noise_floor_z=noise_floor_z)
    coef_raw = norm.coefficients_to_raw(coef_norm)
    return SelectionResult(
        coef_raw,
        coef_raw != 0.0,
        0.0,
        coef_norm,
        norm.scale.copy(),
        "stlsq",
        {**_design_diag(xs), "n_selected": int(np.count_nonzero(coef_raw)), "n_stlsq_iters": n_it, "threshold_mode": threshold_mode, "noise_floor_z": float(noise_floor_z)},
    )


def fit_ridge_threshold(
    design: Array,
    target: Array,
    groups: Array | None = None,
    *,
    seed: int = 1,
    threshold: float = 0.08,
    threshold_mode: str = "relative",
    noise_floor_z: float = 1.0,
    ridge_floor: float = 1e-10,
    **_: object,
) -> SelectionResult:
    norm = ColumnNormalizer().fit(design)
    xs = norm.transform(design)
    coef_norm = _ridge_lstsq(xs, target, ridge_floor)
    se = coefficient_standard_errors(xs, target, coef_norm, ridge_floor)
    coef_norm, n_it = stlsq(xs, target, coef_norm, tau=threshold, n_iter=8, threshold_mode=threshold_mode, standard_errors=se, noise_floor_z=noise_floor_z)
    coef_raw = norm.coefficients_to_raw(coef_norm)
    return SelectionResult(
        coef_raw,
        coef_raw != 0.0,
        0.0,
        coef_norm,
        norm.scale.copy(),
        "ridge_threshold",
        {**_design_diag(xs), "n_selected": int(np.count_nonzero(coef_raw)), "n_stlsq_iters": n_it, "threshold_mode": threshold_mode, "noise_floor_z": float(noise_floor_z)},
    )


def fit_oracle_ols(design: Array, target: Array, groups: Array | None = None, *, true_support: Optional[Array] = None, ridge_floor: float = 1e-10, **_: object) -> SelectionResult:
    if true_support is None:
        raise ValueError("method='oracle_ols' requires true_support")
    sel = np.asarray(true_support, bool)
    coef = np.zeros(design.shape[1])
    if np.any(sel):
        coef[sel] = _ridge_lstsq(np.asarray(design)[:, sel], target, ridge_floor)
    resid = float(np.linalg.norm(design @ coef - target) / max(np.linalg.norm(target), 1e-12))
    return SelectionResult(coef, sel, 0.0, coef.copy(), np.ones(design.shape[1]), "oracle_ols", {**_design_diag(design), "oracle_support": [int(i) for i in np.flatnonzero(sel)], "residual_rel_l2": resid, "n_selected": int(sel.sum())})


def _finish_from_support(
    method: str,
    norm: ColumnNormalizer,
    xs: Array,
    y: Array,
    selected: Array,
    *,
    alpha: float = 0.0,
    stlsq_threshold: float = 0.12,
    threshold_mode: str = "relative",
    noise_floor_z: float = 1.0,
    ridge_floor: float = 1e-10,
    diagnostics: dict | None = None,
) -> SelectionResult:
    if not np.any(selected):
        coef_norm = np.zeros(xs.shape[1])
        n_it = 0
    else:
        debiased = debiased_ols_no_intercept(xs, y, selected)
        se = coefficient_standard_errors(xs, y, debiased, ridge_floor)
        coef_norm, n_it = stlsq(xs, y, debiased, tau=stlsq_threshold, n_iter=20, threshold_mode=threshold_mode, standard_errors=se, noise_floor_z=noise_floor_z)
    coef_raw = norm.coefficients_to_raw(coef_norm)
    support = coef_raw != 0.0
    return SelectionResult(
        coef_raw,
        support,
        alpha,
        coef_norm,
        norm.scale.copy(),
        method,
        {
            **_design_diag(xs),
            **(diagnostics or {}),
            "n_selected": int(support.sum()),
            "n_stlsq_iters": int(n_it),
            "threshold_mode": threshold_mode,
            "noise_floor_z": float(noise_floor_z),
            "ridge_floor": ridge_floor,
        },
    )


def fit_stability_selection(
    design: Array,
    target: Array,
    groups: Array | None = None,
    *,
    seed: int = 0,
    n_boot: int = 8,
    pi_threshold: float = 0.60,
    stlsq_threshold: float = 0.12,
    threshold_mode: str = "relative",
    noise_floor_z: float = 1.0,
    ridge_floor: float = 1e-10,
    fast_screen: bool = False,
    **kw: object,
) -> SelectionResult:
    norm = ColumnNormalizer().fit(design)
    xs = norm.transform(design)
    y = np.asarray(target, float)
    rng = np.random.default_rng(seed)
    freq = np.zeros(xs.shape[1])
    n_rows = xs.shape[0]
    for b in range(max(1, int(n_boot))):
        idx = rng.choice(n_rows, size=max(4, n_rows // 2), replace=True)
        if fast_screen:
            coef0 = _ridge_lstsq(xs[idx], y[idx], ridge_floor)
            se = coefficient_standard_errors(xs[idx], y[idx], coef0, ridge_floor)
            coef_norm, _ = stlsq(xs[idx], y[idx], coef0, tau=stlsq_threshold, n_iter=12, threshold_mode=threshold_mode, standard_errors=se, noise_floor_z=noise_floor_z)
            freq += np.abs(coef_norm) > 0.0
        else:
            res = fit_lassocv_debias_stlsq(xs[idx], y[idx], None, seed=seed + b, ridge_floor=ridge_floor, **kw)
            freq += res.support
    freq /= max(1, int(n_boot))
    selected = freq >= float(pi_threshold)
    return _finish_from_support(
        "stability_selection",
        norm,
        xs,
        y,
        selected,
        stlsq_threshold=stlsq_threshold,
        threshold_mode=threshold_mode,
        noise_floor_z=noise_floor_z,
        ridge_floor=ridge_floor,
        diagnostics={"n_boot": int(n_boot), "pi_threshold": float(pi_threshold), "fast_screen": bool(fast_screen), "support_frequency_max": float(np.max(freq))},
    )


def fit_information_criterion(
    design: Array,
    target: Array,
    groups: Array | None = None,
    *,
    criterion: str = "bic",
    max_terms: int | None = None,
    stlsq_threshold: float = 0.08,
    threshold_mode: str = "relative",
    noise_floor_z: float = 1.0,
    ridge_floor: float = 1e-10,
    **_: object,
) -> SelectionResult:
    norm = ColumnNormalizer().fit(design)
    xs = norm.transform(design)
    y = np.asarray(target, float)
    coef0 = _ridge_lstsq(xs, y, ridge_floor)
    order = np.argsort(np.abs(coef0))[::-1]
    max_terms = min(max_terms or xs.shape[1], xs.shape[1])
    best_score = float("inf")
    best_sel = np.zeros(xs.shape[1], dtype=bool)
    crit = criterion.lower()
    for k in range(1, max_terms + 1):
        sel = np.zeros(xs.shape[1], dtype=bool)
        sel[order[:k]] = True
        coef = np.zeros(xs.shape[1])
        coef[sel] = _ridge_lstsq(xs[:, sel], y, ridge_floor)
        rss = float(np.sum((y - xs @ coef) ** 2))
        n = max(len(y), 1)
        if crit == "aic":
            score = n * np.log(max(rss / n, 1e-300)) + 2 * k
        elif crit == "ebic":
            score = n * np.log(max(rss / n, 1e-300)) + k * np.log(n) + 2.0 * k * np.log(max(xs.shape[1], 2))
        else:
            score = n * np.log(max(rss / n, 1e-300)) + k * np.log(n)
        if score < best_score:
            best_score, best_sel = score, sel
    return _finish_from_support(
        f"{crit}_support",
        norm,
        xs,
        y,
        best_sel,
        stlsq_threshold=stlsq_threshold,
        threshold_mode=threshold_mode,
        noise_floor_z=noise_floor_z,
        ridge_floor=ridge_floor,
        diagnostics={"criterion": crit, "criterion_score": float(best_score), "max_terms": int(max_terms)},
    )


def fit_sr3(
    design: Array,
    target: Array,
    groups: Array | None = None,
    *,
    nu: float = 1.0,
    threshold: float = 0.08,
    n_iter: int = 30,
    threshold_mode: str = "relative",
    noise_floor_z: float = 1.0,
    ridge_floor: float = 1e-10,
    **_: object,
) -> SelectionResult:
    norm = ColumnNormalizer().fit(design)
    xs = norm.transform(design)
    y = np.asarray(target, float)
    w = _ridge_lstsq(xs, y, ridge_floor)
    z = w.copy()
    gram = xs.T @ xs
    eye = np.eye(xs.shape[1])
    for _it in range(int(n_iter)):
        w = np.linalg.solve(gram + (1.0 / max(nu, 1e-12)) * eye, xs.T @ y + (1.0 / max(nu, 1e-12)) * z)
        if threshold_mode == "absolute":
            active = np.abs(w) >= threshold
        elif threshold_mode == "noise_floor":
            se = coefficient_standard_errors(xs, y, w, ridge_floor)
            active = np.abs(w) >= float(noise_floor_z) * se
        else:
            active = np.abs(w) >= threshold * max(float(np.max(np.abs(w))), 1e-12)
        z = np.where(active, w, 0.0)
    coef_raw = norm.coefficients_to_raw(z)
    return SelectionResult(coef_raw, coef_raw != 0.0, 0.0, z, norm.scale.copy(), "sr3", {**_design_diag(xs), "nu": float(nu), "threshold_mode": threshold_mode, "noise_floor_z": float(noise_floor_z), "n_selected": int(np.count_nonzero(coef_raw))})


def fit_best_subset(
    design: Array,
    target: Array,
    groups: Array | None = None,
    *,
    k_max: int = 4,
    ridge_floor: float = 1e-10,
    **_: object,
) -> SelectionResult:
    from itertools import combinations

    norm = ColumnNormalizer().fit(design)
    xs = norm.transform(design)
    y = np.asarray(target, float)
    order = np.argsort(np.abs(_ridge_lstsq(xs, y, ridge_floor)))[::-1][: min(xs.shape[1], max(8, k_max + 3))]
    best_rss = float("inf")
    best_coef = np.zeros(xs.shape[1])
    for k in range(1, min(k_max, len(order)) + 1):
        for comb in combinations(order, k):
            sel = np.zeros(xs.shape[1], dtype=bool)
            sel[list(comb)] = True
            coef = np.zeros(xs.shape[1])
            coef[sel] = _ridge_lstsq(xs[:, sel], y, ridge_floor)
            rss = float(np.sum((y - xs @ coef) ** 2))
            if rss < best_rss:
                best_rss, best_coef = rss, coef
    coef_raw = norm.coefficients_to_raw(best_coef)
    return SelectionResult(coef_raw, coef_raw != 0.0, 0.0, best_coef, norm.scale.copy(), "best_subset", {**_design_diag(xs), "k_max": int(k_max), "subset_rss": best_rss, "n_selected": int(np.count_nonzero(coef_raw))})


def fit_omp(
    design: Array,
    target: Array,
    groups: Array | None = None,
    *,
    n_nonzero: int = 4,
    ridge_floor: float = 1e-10,
    **_: object,
) -> SelectionResult:
    norm = ColumnNormalizer().fit(design)
    xs = norm.transform(design)
    y = np.asarray(target, float)
    try:
        from sklearn.linear_model import OrthogonalMatchingPursuit

        omp = OrthogonalMatchingPursuit(n_nonzero_coefs=min(int(n_nonzero), xs.shape[1]), fit_intercept=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            omp.fit(xs, y)
        coef = np.asarray(omp.coef_, float)
    except Exception:
        residual = y.copy()
        active: list[int] = []
        coef = np.zeros(xs.shape[1])
        for _ in range(min(int(n_nonzero), xs.shape[1])):
            corr = np.abs(xs.T @ residual)
            idx = int(np.argmax(corr))
            if idx in active:
                break
            active.append(idx)
            sel = np.array(active)
            coef[:] = 0.0
            coef[sel] = _ridge_lstsq(xs[:, sel], y, ridge_floor)
            residual = y - xs @ coef
    coef_raw = norm.coefficients_to_raw(coef)
    return SelectionResult(coef_raw, coef_raw != 0.0, 0.0, coef, norm.scale.copy(), "omp", {**_design_diag(xs), "n_nonzero": int(n_nonzero), "n_selected": int(np.count_nonzero(coef_raw))})


def fit_total_least_squares(
    design: Array,
    target: Array,
    groups: Array | None = None,
    *,
    threshold: float = 0.05,
    threshold_mode: str = "relative",
    noise_floor_z: float = 1.0,
    **_: object,
) -> SelectionResult:
    norm = ColumnNormalizer().fit(design)
    xs = norm.transform(design)
    y = np.asarray(target, float)
    aug = np.column_stack([xs, y])
    _, _, vt = np.linalg.svd(aug, full_matrices=False)
    v = vt[-1]
    if abs(v[-1]) < 1e-12:
        coef0 = np.linalg.lstsq(xs, y, rcond=None)[0]
    else:
        coef0 = -v[:-1] / v[-1]
    se = coefficient_standard_errors(xs, y, coef0)
    coef_norm, n_it = stlsq(xs, y, coef0, tau=threshold, n_iter=20, threshold_mode=threshold_mode, standard_errors=se, noise_floor_z=noise_floor_z)
    coef_raw = norm.coefficients_to_raw(coef_norm)
    return SelectionResult(coef_raw, coef_raw != 0.0, 0.0, coef_norm, norm.scale.copy(), "total_least_squares", {**_design_diag(xs), "n_selected": int(np.count_nonzero(coef_raw)), "n_stlsq_iters": n_it, "threshold_mode": threshold_mode, "noise_floor_z": float(noise_floor_z)})


def fit_huber(
    design: Array,
    target: Array,
    groups: Array | None = None,
    *,
    epsilon: float = 1.35,
    threshold: float = 0.05,
    threshold_mode: str = "relative",
    noise_floor_z: float = 1.0,
    ridge_floor: float = 1e-10,
    **_: object,
) -> SelectionResult:
    norm = ColumnNormalizer().fit(design)
    xs = norm.transform(design)
    y = np.asarray(target, float)
    try:
        from sklearn.exceptions import ConvergenceWarning
        from sklearn.linear_model import HuberRegressor

        huber = HuberRegressor(epsilon=epsilon, fit_intercept=False, alpha=ridge_floor, max_iter=300)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            huber.fit(xs, y)
        coef0 = np.asarray(huber.coef_, float)
    except Exception:
        coef0 = _ridge_lstsq(xs, y, ridge_floor)
    se = coefficient_standard_errors(xs, y, coef0, ridge_floor)
    coef_norm, n_it = stlsq(xs, y, coef0, tau=threshold, n_iter=20, threshold_mode=threshold_mode, standard_errors=se, noise_floor_z=noise_floor_z)
    coef_raw = norm.coefficients_to_raw(coef_norm)
    return SelectionResult(coef_raw, coef_raw != 0.0, 0.0, coef_norm, norm.scale.copy(), "huber", {**_design_diag(xs), "epsilon": float(epsilon), "n_selected": int(np.count_nonzero(coef_raw)), "n_stlsq_iters": n_it, "threshold_mode": threshold_mode, "noise_floor_z": float(noise_floor_z)})


def fit_ridge_gcv(
    design: Array,
    target: Array,
    groups: Array | None = None,
    *,
    alphas: tuple[float, ...] | list[float] | None = None,
    threshold: float = 0.04,
    threshold_mode: str = "relative",
    noise_floor_z: float = 1.0,
    **_: object,
) -> SelectionResult:
    norm = ColumnNormalizer().fit(design)
    xs = norm.transform(design)
    y = np.asarray(target, float)
    alphas = tuple(alphas or np.logspace(-12, -2, 32))
    u, s, vt = np.linalg.svd(xs, full_matrices=False)
    uy = u.T @ y
    best_alpha = float(alphas[0])
    best_score = float("inf")
    best_coef = np.zeros(xs.shape[1])
    n = len(y)
    for alpha in alphas:
        filt = s / (s * s + alpha)
        coef = vt.T @ (filt * uy)
        pred = xs @ coef
        trace = float(np.sum((s * s) / (s * s + alpha)))
        score = float(np.mean((y - pred) ** 2) / max((1.0 - trace / max(n, 1)) ** 2, 1e-12))
        if score < best_score:
            best_score, best_alpha, best_coef = score, float(alpha), coef
    se = coefficient_standard_errors(xs, y, best_coef)
    coef_norm, n_it = stlsq(xs, y, best_coef, tau=threshold, n_iter=20, threshold_mode=threshold_mode, standard_errors=se, noise_floor_z=noise_floor_z)
    coef_raw = norm.coefficients_to_raw(coef_norm)
    return SelectionResult(coef_raw, coef_raw != 0.0, best_alpha, coef_norm, norm.scale.copy(), "ridge_gcv", {**_design_diag(xs), "gcv_score": best_score, "n_selected": int(np.count_nonzero(coef_raw)), "n_stlsq_iters": n_it, "threshold_mode": threshold_mode, "noise_floor_z": float(noise_floor_z)})


def solve(design: Array, target: Array, groups: Array | None = None, method: str = "lassocv_debias_stlsq", *, true_support: Optional[Array] = None, seed: int = 0, ridge_floor: float = 1e-10, **kw: object) -> SelectionResult:
    if method in {"lasso_stlsq", "lassocv_debias_stlsq"}:
        return fit_lassocv_debias_stlsq(design, target, groups, seed=seed, ridge_floor=ridge_floor, **kw)
    if method in {"elastic_net", "elasticnet", "elastic_net_stlsq", "elastic_net_debias_stlsq"}:
        return fit_elastic_net_debias_stlsq(design, target, groups, seed=seed, ridge_floor=ridge_floor, **kw)
    if method in {"adaptive_lasso", "adaptive_lasso_stlsq", "adaptive_lasso_debias_stlsq"}:
        return fit_adaptive_lasso_debias_stlsq(design, target, groups, seed=seed, ridge_floor=ridge_floor, **kw)
    if method == "stlsq":
        return fit_stlsq(design, target, groups, ridge_floor=ridge_floor, **kw)
    if method == "ridge_threshold":
        return fit_ridge_threshold(design, target, groups, seed=seed, ridge_floor=ridge_floor, **kw)
    if method == "svd_threshold":
        return fit_svd_threshold(design, target, groups, **kw)
    if method == "oracle_ols":
        return fit_oracle_ols(design, target, groups, true_support=true_support, ridge_floor=ridge_floor, **kw)
    if method in {"stability_selection", "bootstrap_lasso"}:
        return fit_stability_selection(design, target, groups, seed=seed, ridge_floor=ridge_floor, **kw)
    if method in {"bic", "aic", "ebic", "information_criterion"}:
        criterion = method if method in {"bic", "aic", "ebic"} else str(kw.pop("criterion", "bic"))
        return fit_information_criterion(design, target, groups, criterion=criterion, ridge_floor=ridge_floor, **kw)
    if method == "sr3":
        return fit_sr3(design, target, groups, ridge_floor=ridge_floor, **kw)
    if method in {"best_subset", "mio_l0", "l0"}:
        return fit_best_subset(design, target, groups, ridge_floor=ridge_floor, **kw)
    if method in {"omp", "forward_stagewise"}:
        return fit_omp(design, target, groups, ridge_floor=ridge_floor, **kw)
    if method in {"tls", "total_least_squares"}:
        return fit_total_least_squares(design, target, groups, **kw)
    if method in {"huber", "irls"}:
        return fit_huber(design, target, groups, ridge_floor=ridge_floor, **kw)
    if method in {"ridge_gcv", "ridge_auto"}:
        return fit_ridge_gcv(design, target, groups, **kw)
    raise ValueError(f"unknown regression method {method!r}")
