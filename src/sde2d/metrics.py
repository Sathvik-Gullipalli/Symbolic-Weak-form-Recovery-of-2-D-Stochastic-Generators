from __future__ import annotations

import numpy as np

from .diffusion import project_psd

Array = np.ndarray


def relative_l2(a_hat: Array, a_true: Array, weights: Array | None = None) -> float:
    ah = np.asarray(a_hat, float)
    at = np.asarray(a_true, float)
    mask = np.isfinite(ah) & np.isfinite(at)
    if weights is None:
        num = np.sum((ah[mask] - at[mask]) ** 2)
        den = np.sum(at[mask] ** 2)
    else:
        w = np.asarray(weights, float)
        while w.ndim < ah.ndim:
            w = np.expand_dims(w, -1)
        w = np.broadcast_to(w, ah.shape)
        num = np.sum(w[mask] * (ah[mask] - at[mask]) ** 2)
        den = np.sum(w[mask] * at[mask] ** 2)
    return float(np.sqrt(num / den)) if den > 1e-20 else float("nan")


def cosine_similarity(a: Array, b: Array) -> float:
    aa = np.asarray(a, float).ravel()
    bb = np.asarray(b, float).ravel()
    den = np.linalg.norm(aa) * np.linalg.norm(bb)
    return float(aa @ bb / den) if den > 1e-20 else float("nan")


def central_grid(x: Array, grid_n: int = 25, q: tuple[float, float] = (0.02, 0.98)) -> Array:
    x = np.asarray(x, float)
    if x.shape[1] == 1:
        lo, hi = np.quantile(x[:, 0], q)
        return np.linspace(lo, hi, grid_n)[:, None]
    lo = np.quantile(x, q[0], axis=0)
    hi = np.quantile(x, q[1], axis=0)
    gx = np.linspace(lo[0], hi[0], grid_n)
    gy = np.linspace(lo[1], hi[1], grid_n)
    return np.array([(a, b) for a in gx for b in gy])


def function_l2_errors(fit, system, points: Array) -> dict:
    bh, ah = fit.evaluate(points)
    bt = system.true_drift(points)
    at = system.true_diffusion(points)
    out = {"drift_rel_l2": relative_l2(bh, bt)}
    for p in range(bh.shape[1]):
        out[f"b{p+1}_rel_l2"] = relative_l2(bh[:, p], bt[:, p])
    entries = [(0, 0, "a11")]
    if fit.dim >= 2:
        entries += [(0, 1, "a12"), (1, 1, "a22")]
    for i, j, name in entries:
        out[f"{name}_rel_l2"] = relative_l2(ah[:, i, j], at[:, i, j])
    out["diffusion_rel_l2"] = relative_l2(ah, at)
    return out


def a12_sign_accuracy(a_hat: Array, a_true: Array, relative_floor: float = 0.10, absolute_floor: float = 1e-6) -> float:
    """Sign agreement away from the near-zero band where sign is undefined."""
    ah = np.asarray(a_hat, float)
    at = np.asarray(a_true, float)
    threshold = max(float(absolute_floor), float(relative_floor) * float(np.nanmax(np.abs(at))) if at.size else 0.0)
    mask = np.isfinite(ah) & np.isfinite(at) & (np.abs(at) > threshold)
    return float(np.mean(np.sign(ah[mask]) == np.sign(at[mask]))) if np.any(mask) else float("nan")


def tensor_metrics(fit, system, points: Array, psd: bool = False, tau_sign: float = 0.10) -> dict:
    ah = fit.evaluate(points, psd=psd)[1]
    at = system.true_diffusion(points)
    d = ah - at
    frob = np.sqrt(np.sum(d * d, axis=(1, 2)))
    out = {
        "frob_avg": float(np.mean(frob)),
        "frob_rel_avg": float(np.mean(frob / np.maximum(np.sqrt(np.sum(at * at, axis=(1, 2))), 1e-12))),
    }
    if ah.shape[1] >= 2:
        out["a12_sign_accuracy"] = a12_sign_accuracy(ah[:, 0, 1], at[:, 0, 1], relative_floor=tau_sign)
        out["a12_abs_err_avg"] = float(np.mean(np.abs(d[:, 0, 1])))
    return out


def psd_validity(a_grid: Array, tol: float = 1e-10) -> dict:
    a = 0.5 * (np.asarray(a_grid, float) + np.swapaxes(np.asarray(a_grid, float), -1, -2))
    if a.shape[-1] == 1:
        lam = a[:, 0, 0]
        return {"pct_psd_valid": float(np.mean(lam >= -tol)), "min_eigenvalue_grid": float(np.min(lam)), "psd_violation_rate": float(np.mean(lam < -tol)), "det_gap_violation_rate": 0.0, "diag_violation_rate": float(np.mean(lam < -tol)), "median_condition_number": 1.0}
    eig = np.linalg.eigvalsh(a)
    det = a[:, 0, 0] * a[:, 1, 1] - a[:, 0, 1] ** 2
    return {
        "pct_psd_valid": float(np.mean(eig[:, 0] >= -tol)),
        "min_eigenvalue_grid": float(np.min(eig[:, 0])),
        "psd_violation_rate": float(np.mean(eig[:, 0] < -tol)),
        "det_gap_violation_rate": float(np.mean(det < -tol)),
        "diag_violation_rate": float(np.mean((a[:, 0, 0] < -tol) | (a[:, 1, 1] < -tol))),
        "median_condition_number": float(np.median(eig[:, -1] / np.maximum(eig[:, 0], tol))),
    }


def project_tensor_field(a: Array) -> Array:
    return project_psd(a)
