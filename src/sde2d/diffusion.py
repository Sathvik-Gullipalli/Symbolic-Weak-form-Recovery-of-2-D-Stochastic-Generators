from __future__ import annotations

import numpy as np

Array = np.ndarray


def symmetrize_matrix(a: Array) -> Array:
    return 0.5 * (np.asarray(a, float) + np.swapaxes(np.asarray(a, float), -1, -2))


def eigenvalues(matrix: Array) -> Array:
    return np.linalg.eigvalsh(symmetrize_matrix(matrix))


def project_psd(a: Array, eps: float = 0.0) -> Array:
    a = symmetrize_matrix(a)
    if a.ndim == 2:
        vals, vecs = np.linalg.eigh(a)
        vals = np.maximum(vals, eps)
        return (vecs * vals) @ vecs.T
    out = np.empty_like(a)
    for i, mat in enumerate(a):
        vals, vecs = np.linalg.eigh(mat)
        vals = np.maximum(vals, eps)
        out[i] = (vecs * vals) @ vecs.T
    return out


def corrected_quadratic_variation(increments: Array, drift_hat: Array, dt: float) -> Array:
    inc = np.asarray(increments, float)
    b = np.asarray(drift_hat, float)
    outer = inc[:, :, None] * inc[:, None, :]
    bb = b[:, :, None] * b[:, None, :]
    return outer / dt - bb * dt


def lag1_noise_covariance(x: Array) -> Array:
    x = np.asarray(x, float)
    dx = np.diff(x, axis=0)
    if len(dx) < 2:
        return np.zeros((x.shape[1], x.shape[1]))
    cov = -(dx[:-1].T @ dx[1:]) / max(len(dx) - 1, 1)
    cov = symmetrize_matrix(cov)
    return project_psd(cov)


def diagonal_lag1_noise_covariance(x: Array) -> Array:
    x = np.asarray(x, float)
    dx = np.diff(x, axis=0)
    if len(dx) < 2:
        return np.zeros((x.shape[1], x.shape[1]))
    diag = -np.mean(dx[:-1] * dx[1:], axis=0)
    return np.diag(np.maximum(diag, 0.0))
