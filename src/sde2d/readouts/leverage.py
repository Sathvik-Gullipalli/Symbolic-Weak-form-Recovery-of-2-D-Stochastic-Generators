from __future__ import annotations

import numpy as np

Array = np.ndarray


def rho_from_tensor(a: Array, eps: float = 1e-12) -> Array:
    a = np.asarray(a, float)
    den = np.sqrt(np.maximum(a[..., 0, 0] * a[..., 1, 1], eps))
    return a[..., 0, 1] / den


def leverage_from_fit(fit, x: Array, psd: bool = False) -> Array:
    return rho_from_tensor(fit.evaluate(x, psd=psd)[1])


def rho_summary_from_fit(fit, x: Array, psd: bool = False, trim_abs: float = 5.0) -> dict:
    rho = leverage_from_fit(fit, x, psd=psd)
    rho = rho[np.isfinite(rho) & (np.abs(rho) <= trim_abs)]
    if rho.size == 0:
        return {"rho_tensor_mean": float("nan"), "rho_tensor_median": float("nan"), "rho_tensor_iqr": float("nan"), "n_rho_points": 0}
    q25, q75 = np.quantile(rho, [0.25, 0.75])
    return {
        "rho_tensor_mean": float(np.mean(rho)),
        "rho_tensor_median": float(np.median(rho)),
        "rho_tensor_iqr": float(q75 - q25),
        "n_rho_points": int(rho.size),
    }


def recover_heston_parameters(fit) -> dict:
    names = fit.library.names

    def coef(target: str, term: str) -> float:
        if term not in names:
            return float("nan")
        i = names.index(term)
        if target == "b2":
            return float(fit.drift[i, 1])
        key = {"a11": (0, 0), "a12": (0, 1), "a22": (1, 1)}[target]
        return float(fit.diffusion[key][i])

    v_term = "V" if "V" in names else "y"
    kappa = -coef("b2", v_term)
    const = coef("b2", "1")
    theta = const / kappa if np.isfinite(kappa) and abs(kappa) > 1e-12 else float("nan")
    xi2 = coef("a22", v_term)
    xi = float(np.sqrt(max(xi2, 0.0))) if np.isfinite(xi2) else float("nan")
    axv = coef("a12", v_term)
    rho = axv / xi if np.isfinite(xi) and xi > 1e-12 else float("nan")
    return {"kappa_hat": kappa, "theta_hat": theta, "xi_hat": xi, "rho_hat": rho}


def ewma(values: Array, span: int) -> Array:
    values = np.asarray(values, float)
    alpha = 2.0 / (span + 1.0)
    out = np.empty_like(values)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * values[i] + (1.0 - alpha) * out[i - 1]
    return np.maximum(out, 1e-12)


def shifted_ewma(values: Array, span: int) -> Array:
    sm = ewma(values, span)
    out = sm.copy()
    if len(out) > 1:
        out[1:] = sm[:-1]
    return out


def proxy_stats(proxy: Array, true_v: Array) -> tuple[float, float, int]:
    n = min(len(proxy), len(true_v))
    proxy = np.asarray(proxy[:n], float)
    true_v = np.asarray(true_v[:n], float)
    nsr = float(np.std(proxy - true_v) / max(np.std(true_v), 1e-12))
    corr = float(np.corrcoef(proxy, true_v)[0, 1]) if n > 2 else float("nan")
    max_lag = min(20, n // 5)
    best_lag, best = 0, -np.inf
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            a, b = proxy[-lag:], true_v[: n + lag]
        elif lag > 0:
            a, b = proxy[: n - lag], true_v[lag:]
        else:
            a, b = proxy, true_v
        if len(a) > 2:
            c = np.corrcoef(a, b)[0, 1]
            if c > best:
                best, best_lag = c, lag
    return nsr, corr, int(best_lag)
