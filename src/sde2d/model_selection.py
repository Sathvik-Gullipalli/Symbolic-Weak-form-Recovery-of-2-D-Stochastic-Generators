from __future__ import annotations

import math

import numpy as np

Array = np.ndarray
EPS = 1e-12


def _ols(x: Array, y: Array) -> Array:
    return np.linalg.lstsq(x, y, rcond=None)[0]


def bootstrap_quadratic_support(xw: Array, yw: Array, seed: int, n_boot: int = 30) -> float:
    rng = np.random.default_rng(seed + 4001)
    n = len(yw)
    chunk = max(n // 8, 4)
    selected = 0
    for _ in range(n_boot):
        starts = rng.integers(0, max(n - chunk, 1), size=max(n // chunk, 1))
        idx = np.concatenate([np.arange(s, min(s + chunk, n)) for s in starts])[:n]
        xx, yy = xw[idx], yw[idx]
        lin = _ols(xx[:, :2], yy)
        quad = _ols(xx, yy)
        lin_rss = float(np.sum((yy - xx[:, :2] @ lin) ** 2))
        quad_rss = float(np.sum((yy - xx @ quad) ** 2))
        lin_bic = len(yy) * math.log(max(lin_rss, EPS) / len(yy)) + 2 * math.log(len(yy))
        quad_bic = len(yy) * math.log(max(quad_rss, EPS) / len(yy)) + 3 * math.log(len(yy))
        selected += int(quad_bic < lin_bic)
    return selected / n_boot


def model_selection_stats(v: Array, dv: Array, seed: int = 0, dt: float = 1 / 252) -> dict:
    v = np.asarray(v, float)
    y = np.asarray(dv, float) / dt
    xw = np.column_stack([np.ones_like(v), v, v * v])
    cut = max(int(0.70 * len(y)), 5)
    lin = _ols(xw[:, :2], y)
    quad = _ols(xw, y)
    lin_rss = float(np.sum((y - xw[:, :2] @ lin) ** 2))
    quad_rss = float(np.sum((y - xw @ quad) ** 2))
    bic_linear = len(y) * math.log(max(lin_rss, EPS) / len(y)) + 2 * math.log(len(y))
    bic_quadratic = len(y) * math.log(max(quad_rss, EPS) / len(y)) + 3 * math.log(len(y))
    delta = bic_quadratic - bic_linear
    lin_train = _ols(xw[:cut, :2], y[:cut])
    quad_train = _ols(xw[:cut], y[:cut])
    linear_loss = float(np.mean((y[cut:] - xw[cut:, :2] @ lin_train) ** 2))
    quadratic_loss = float(np.mean((y[cut:] - xw[cut:] @ quad_train) ** 2))
    improvement = (linear_loss - quadratic_loss) / max(linear_loss, EPS)
    boot = bootstrap_quadratic_support(xw, y, seed)
    bic_pass = int(delta < 0.0)
    validation_pass = int(improvement >= 0.20)
    bootstrap_pass = int(boot >= 0.70)
    quadratic_selected = int(bic_pass and validation_pass and bootstrap_pass)
    linear_accepted = int((not bic_pass) and improvement <= 0.05 and boot <= 0.50)
    decision = "quadratic_detected" if quadratic_selected else "linear_confirmed" if linear_accepted else "inconclusive"
    return {
        "bic_linear": bic_linear,
        "bic_quadratic": bic_quadratic,
        "delta_bic_quadratic_minus_linear": delta,
        "validation_loss_linear": linear_loss,
        "validation_loss_quadratic": quadratic_loss,
        "validation_improvement_quadratic": improvement,
        "bootstrap_quadratic_support": boot,
        "bic_gate_pass": bic_pass,
        "validation_gate_pass": validation_pass,
        "bootstrap_gate_pass": bootstrap_pass,
        "quadratic_selected": quadratic_selected,
        "linear_accepted": linear_accepted,
        "decision": decision,
    }
