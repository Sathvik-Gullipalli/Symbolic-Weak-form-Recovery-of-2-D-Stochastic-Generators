from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigs

Array = np.ndarray


def drift_jacobian_from_fit(fit) -> Array:
    names = fit.library.names
    dim = fit.dim
    jac = np.zeros((dim, dim))
    for p in range(dim):
        for q, name in enumerate(["x", "y"][:dim]):
            if name in names:
                jac[p, q] = fit.drift[names.index(name), p]
    return jac


def spectral_gap_linear_fit(fit) -> float:
    vals = np.linalg.eigvals(drift_jacobian_from_fit(fit))
    stable = -np.real(vals[np.real(vals) < 0])
    return float(np.min(stable)) if stable.size else float("nan")


def fem_generator_eigenvalues_2d(drift_fn, diff_fn, x_range: tuple[float, float], y_range: tuple[float, float], Nx: int = 35, Ny: int = 35, n_eigs: int = 6) -> Array:
    x = np.linspace(*x_range, Nx)
    y = np.linspace(*y_range, Ny)
    dx, dy = x[1] - x[0], y[1] - y[0]
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []

    def idx(i: int, j: int) -> int:
        return i * Ny + j

    for i in range(1, Nx - 1):
        for j in range(1, Ny - 1):
            p = idx(i, j)
            b = drift_fn(np.array([[x[i], y[j]]]))[0]
            a = diff_fn(np.array([[x[i], y[j]]]))[0]
            b1, b2 = b
            a11, a12, a22 = a[0, 0], a[0, 1], a[1, 1]
            entries = {
                (i, j): -a11 / dx**2 - a22 / dy**2,
                (i + 1, j): b1 / (2 * dx) + 0.5 * a11 / dx**2,
                (i - 1, j): -b1 / (2 * dx) + 0.5 * a11 / dx**2,
                (i, j + 1): b2 / (2 * dy) + 0.5 * a22 / dy**2,
                (i, j - 1): -b2 / (2 * dy) + 0.5 * a22 / dy**2,
                (i + 1, j + 1): a12 / (4 * dx * dy),
                (i - 1, j - 1): a12 / (4 * dx * dy),
                (i - 1, j + 1): -a12 / (4 * dx * dy),
                (i + 1, j - 1): -a12 / (4 * dx * dy),
            }
            for (ii, jj), val in entries.items():
                rows.append(p)
                cols.append(idx(ii, jj))
                vals.append(float(val))
    n = Nx * Ny
    mat = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
    w = eigs(mat, k=min(n_eigs + 1, n - 2), which="LR", return_eigenvectors=False)
    w = w[np.abs(w) > 1e-10]
    return w[np.argsort(w.real)[::-1]][:n_eigs]
