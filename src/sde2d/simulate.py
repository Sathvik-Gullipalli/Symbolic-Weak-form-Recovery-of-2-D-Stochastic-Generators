from __future__ import annotations

from typing import Callable

import numpy as np

Array = np.ndarray


def euler_maruyama(drift_fn: Callable[[Array], Array], sigma_fn: Callable[[Array], Array], x0: Array, dt: float, M: int, seed: int | None = None, clip: tuple[float, float] | None = None) -> Array:
    rng = np.random.default_rng(seed)
    x0 = np.asarray(x0, float)
    dim = x0.size
    out = np.zeros((M + 1, dim))
    out[0] = x0
    for n in range(M):
        x = out[n : n + 1]
        b = drift_fn(x)[0]
        sig = sigma_fn(x)[0]
        z = rng.standard_normal(sig.shape[1])
        out[n + 1] = out[n] + b * dt + sig @ z * np.sqrt(dt)
        if clip is not None:
            out[n + 1] = np.clip(out[n + 1], clip[0], clip[1])
    return out


def euler_maruyama_correlated(drift_fn: Callable[[Array], Array], a_fn: Callable[[Array], Array], x0: Array, dt: float, M: int, seed: int | None = None, clip: tuple[float, float] | None = None) -> Array:
    rng = np.random.default_rng(seed)
    x0 = np.asarray(x0, float)
    dim = x0.size
    out = np.zeros((M + 1, dim))
    out[0] = x0
    for n in range(M):
        x = out[n : n + 1]
        a = a_fn(x)[0]
        vals, vecs = np.linalg.eigh(0.5 * (a + a.T))
        root = vecs @ np.diag(np.sqrt(np.maximum(vals, 0.0)))
        out[n + 1] = out[n] + drift_fn(x)[0] * dt + root @ rng.standard_normal(dim) * np.sqrt(dt)
        if clip is not None:
            out[n + 1] = np.clip(out[n + 1], clip[0], clip[1])
    return out


def add_obs_noise(traj: Array, nu_ratio: float, seed: int) -> tuple[Array, Array]:
    traj = np.asarray(traj, float)
    dim = traj.shape[1]
    if nu_ratio <= 0:
        return traj.copy(), np.zeros((dim, dim))
    rng = np.random.default_rng(seed)
    inc_var = np.var(np.diff(traj, axis=0), axis=0, ddof=1)
    noise_var = np.maximum(nu_ratio * inc_var, 0.0)
    noise = rng.normal(scale=np.sqrt(noise_var), size=traj.shape)
    return traj + noise, np.diag(noise_var)


def add_obs_noise_laplace(traj: Array, nu_ratio: float, seed: int) -> tuple[Array, Array]:
    traj = np.asarray(traj, float)
    dim = traj.shape[1]
    if nu_ratio <= 0:
        return traj.copy(), np.zeros((dim, dim))
    rng = np.random.default_rng(seed)
    inc_var = np.var(np.diff(traj, axis=0), axis=0, ddof=1)
    noise_var = np.maximum(nu_ratio * inc_var, 0.0)
    scale = np.sqrt(noise_var / 2.0)
    noise = rng.laplace(scale=scale, size=traj.shape)
    return traj + noise, np.diag(noise_var)


def trim_boundary(x: Array, coord: int, floor: float, band_frac: float = 0.05) -> Array:
    lo = floor + band_frac * (np.quantile(x[:, coord], 0.98) - floor)
    return np.asarray(x[:-1, coord] > lo, bool)
