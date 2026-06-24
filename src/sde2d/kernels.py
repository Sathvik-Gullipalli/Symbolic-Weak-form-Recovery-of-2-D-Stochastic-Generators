from __future__ import annotations

import math
from itertools import product

import numpy as np

Array = np.ndarray


def _qmc_centers(z: Array, n_centers: int, seed: int, engine: str) -> Array:
    try:
        from scipy.stats import qmc

        sampler_cls = qmc.Sobol if engine == "sobol" else qmc.Halton
        sampler = sampler_cls(d=z.shape[1], scramble=True, seed=seed)
        if engine == "sobol":
            m = int(math.ceil(math.log2(max(n_centers, 1))))
            u = sampler.random_base2(m)[:n_centers]
        else:
            u = sampler.random(n_centers)
        lo = np.quantile(z, 0.02, axis=0)
        hi = np.quantile(z, 0.98, axis=0)
        return lo + u * (hi - lo)
    except Exception:
        rng = np.random.default_rng(seed)
        lo = np.quantile(z, 0.02, axis=0)
        hi = np.quantile(z, 0.98, axis=0)
        return lo + rng.random((n_centers, z.shape[1])) * (hi - lo)


def _greedy_coverage_centers(z: Array, n_centers: int, seed: int) -> Array:
    rng = np.random.default_rng(seed)
    n_centers = min(n_centers, z.shape[0])
    first = int(rng.integers(0, z.shape[0]))
    chosen = [first]
    dist2 = np.sum((z - z[first]) ** 2, axis=1)
    for _ in range(1, n_centers):
        nxt = int(np.argmax(dist2))
        chosen.append(nxt)
        dist2 = np.minimum(dist2, np.sum((z - z[nxt]) ** 2, axis=1))
    return z[np.array(chosen)]


def choose_centers(z: Array, n_centers: int, seed: int, scheme: str = "kmeans", grid_shape: tuple[int, int] | None = None) -> Array:
    z = np.atleast_2d(np.asarray(z, float))
    n_centers = int(min(max(n_centers, 1), z.shape[0] if scheme in {"random", "subsample"} else max(n_centers, 1)))
    rng = np.random.default_rng(seed)
    scheme = scheme.lower()
    if scheme in {"random", "subsample"}:
        idx = rng.choice(z.shape[0], size=min(n_centers, z.shape[0]), replace=False)
        return z[np.sort(idx)]
    if scheme == "kmeans":
        try:
            from sklearn.cluster import KMeans

            km = KMeans(n_clusters=n_centers, n_init=10, random_state=seed)
            km.fit(z)
            return km.cluster_centers_
        except Exception:
            idx = rng.choice(z.shape[0], size=min(n_centers, z.shape[0]), replace=False)
            return z[np.sort(idx)]
    if scheme in {"cvt", "lloyd"}:
        try:
            from sklearn.cluster import KMeans

            km = KMeans(n_clusters=n_centers, n_init=1, max_iter=100, random_state=seed)
            km.fit(z)
            return km.cluster_centers_
        except Exception:
            return _greedy_coverage_centers(z, n_centers, seed)
    if scheme in {"sobol", "halton"}:
        return _qmc_centers(z, n_centers, seed, scheme)
    if scheme in {"greedy", "greedy_coverage", "maximin"}:
        return _greedy_coverage_centers(z, n_centers, seed)
    if scheme in {"density_equalized", "boundary_aware"}:
        scheme = "quantile_grid"
    if grid_shape is None:
        side = int(round(math.sqrt(n_centers)))
        grid_shape = (max(side, 1), max(side, 1))
    if scheme == "uniform_grid":
        axes = [np.linspace(np.min(z[:, d]), np.max(z[:, d]), grid_shape[d]) for d in range(z.shape[1])]
    elif scheme == "quantile_grid":
        axes = [np.quantile(z[:, d], np.linspace(0.02, 0.98, grid_shape[d])) for d in range(z.shape[1])]
    else:
        raise ValueError(f"unknown center scheme {scheme!r}")
    return np.array(list(product(*axes)), dtype=float)


def median_bandwidth(centers: Array, multiplier: float = 1.0, rule: str = "nn_median") -> float:
    centers = np.atleast_2d(np.asarray(centers, float))
    if centers.shape[0] <= 1:
        return 1.0
    d = np.sqrt(((centers[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2))
    if rule == "pairwise_median":
        vals = d[np.triu_indices_from(d, k=1)]
        vals = vals[vals > 0]
    else:
        d = d + np.eye(d.shape[0]) * 1e300
        vals = np.min(d, axis=1)
        vals = vals[np.isfinite(vals) & (vals > 0)]
    if vals.size == 0:
        return 1.0
    return float(max(multiplier * np.median(vals), 1e-8))


def knn_bandwidth(z: Array, centers: Array, multiplier: float = 1.0, k: int = 50) -> Array:
    z = np.atleast_2d(np.asarray(z, float))
    centers = np.atleast_2d(np.asarray(centers, float))
    if z.shape[0] == 0:
        return np.full(centers.shape[0], 1.0)
    k = min(max(int(k), 1), z.shape[0])
    d = np.sqrt(((centers[:, None, :] - z[None, :, :]) ** 2).sum(axis=2))
    kth = np.partition(d, k - 1, axis=1)[:, k - 1]
    floor = median_bandwidth(centers, multiplier=1.0, rule="nn_median")
    return np.maximum(multiplier * kth, max(1e-8, 0.1 * floor))


def local_cov_bandwidth(z: Array, centers: Array, multiplier: float = 1.0, k: int = 50, shrink: float = 0.1) -> Array:
    z = np.atleast_2d(np.asarray(z, float))
    centers = np.atleast_2d(np.asarray(centers, float))
    dim = centers.shape[1]
    k = min(max(int(k), dim + 2), z.shape[0])
    base = median_bandwidth(centers, multiplier=multiplier, rule="nn_median")
    out = np.zeros((centers.shape[0], dim, dim))
    eye = np.eye(dim)
    for j, c in enumerate(centers):
        d2 = np.sum((z - c) ** 2, axis=1)
        idx = np.argpartition(d2, k - 1)[:k]
        cov = np.cov(z[idx], rowvar=False)
        cov = np.atleast_2d(cov)
        scale = max(float(np.trace(cov) / max(dim, 1)), 1e-12)
        metric = (1.0 - shrink) * (cov / scale) + shrink * eye
        out[j] = (base * base) * metric + 1e-8 * eye
    return out


def gaussian_weights_unnormalized(z: Array, centers: Array, bandwidth: float | Array) -> Array:
    z = np.atleast_2d(np.asarray(z, float))
    centers = np.atleast_2d(np.asarray(centers, float))
    bw = np.asarray(bandwidth, float)
    if bw.ndim == 3:
        diff = centers[:, None, :] - z[None, :, :]
        out = np.empty((centers.shape[0], z.shape[0]))
        for j in range(centers.shape[0]):
            inv = np.linalg.pinv(bw[j])
            out[j] = np.einsum("nd,dd,nd->n", diff[j], inv, diff[j])
        return np.exp(-0.5 * out)
    if bw.ndim == 1 and bw.size == centers.shape[0]:
        h = np.maximum(bw[:, None], 1e-8)
        d2 = ((centers[:, None, :] - z[None, :, :]) ** 2).sum(axis=2) / (h * h)
        return np.exp(-0.5 * d2)
    if bw.ndim == 2:
        inv = np.linalg.pinv(bw)
        diff = centers[:, None, :] - z[None, :, :]
        d2 = np.einsum("mnd,dd,mnd->mn", diff, inv, diff)
    else:
        h = float(bw.ravel()[0])
        d2 = ((centers[:, None, :] - z[None, :, :]) ** 2).sum(axis=2) / max(h * h, 1e-16)
    return np.exp(-0.5 * d2)


def normalize_projection(weights: Array, mode: str = "row") -> Array:
    weights = np.asarray(weights, float)
    mode = mode.lower()
    if mode in {"row", "nw", "pou"}:
        sums = np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
        return weights / sums
    if mode in {"none", "raw"}:
        return weights
    if mode == "col":
        sums = np.maximum(weights.sum(axis=0, keepdims=True), 1e-12)
        return weights / sums
    if mode == "sinkhorn":
        out = weights.copy()
        for _ in range(8):
            out /= np.maximum(out.sum(axis=1, keepdims=True), 1e-12)
            out /= np.maximum(out.sum(axis=0, keepdims=True), 1e-12)
        out /= np.maximum(out.sum(axis=1, keepdims=True), 1e-12)
        return out
    raise ValueError(f"unknown projection normalization {mode!r}")


def _local_poly_features(diff: Array, order: int) -> Array:
    if order <= 0:
        return np.ones((diff.shape[0], 1))
    cols = [np.ones(diff.shape[0]), diff[:, 0]]
    if diff.shape[1] > 1:
        cols.append(diff[:, 1])
    if order >= 2:
        cols.append(diff[:, 0] ** 2)
        if diff.shape[1] > 1:
            cols.append(diff[:, 0] * diff[:, 1])
            cols.append(diff[:, 1] ** 2)
    return np.column_stack(cols)


def local_polynomial_projection_matrix(
    z: Array,
    centers: Array,
    bandwidth: float | Array,
    order: int = 1,
    ridge: float = 1e-8,
    normalization: str = "row",
) -> Array:
    """Linear smoother matrix whose rows return local-polynomial intercepts.

    Order 0 is ordinary row-normalized Gaussian/Nadaraya-Watson projection.
    Orders 1 and 2 use the same spatial Gaussian weights but replace the
    local-constant projection by the intercept of a local polynomial fit at
    each center.
    """
    if order <= 0:
        return gaussian_projection_matrix(z, centers, bandwidth, normalization=normalization)
    z = np.atleast_2d(np.asarray(z, float))
    centers = np.atleast_2d(np.asarray(centers, float))
    weights = gaussian_weights_unnormalized(z, centers, bandwidth)
    rows = []
    for j, c in enumerate(centers):
        w = weights[j]
        diff = z - c
        phi = _local_poly_features(diff, order)
        gram = phi.T @ (phi * w[:, None])
        lam = ridge * max(float(np.trace(gram) / max(gram.shape[0], 1)), 1e-12)
        try:
            coeff = np.linalg.solve(gram + lam * np.eye(gram.shape[0]), phi.T * w)
            rows.append(coeff[0])
        except np.linalg.LinAlgError:
            rows.append(normalize_projection(w[None, :], "row")[0])
    return np.asarray(rows)


def effective_samples(weights: Array) -> Array:
    weights = np.asarray(weights, float)
    s1 = weights.sum(axis=1)
    s2 = (weights * weights).sum(axis=1)
    return (s1 * s1) / np.maximum(s2, 1e-300)


def prune_projection(
    projection: Array,
    centers: Array,
    *,
    min_effective_samples: float = 30.0,
    min_rows: int | None = None,
) -> tuple[Array, Array, dict]:
    n_eff = effective_samples(np.abs(projection))
    keep = n_eff >= float(min_effective_samples)
    if min_rows is not None and int(np.sum(keep)) < min_rows:
        order = np.argsort(n_eff)[::-1]
        keep = np.zeros_like(keep, dtype=bool)
        keep[order[: min(min_rows, len(order))]] = True
    if not np.any(keep):
        keep[np.argmax(n_eff)] = True
    meta = {
        "min_effective_samples": float(np.min(n_eff)) if n_eff.size else float("nan"),
        "median_effective_samples": float(np.median(n_eff)) if n_eff.size else float("nan"),
        "n_pruned_centers": int(np.sum(~keep)),
        "n_kept_centers": int(np.sum(keep)),
    }
    return projection[keep], centers[keep], meta


def gaussian_projection_matrix(z: Array, centers: Array, bandwidth: float | Array, normalization: str = "row") -> Array:
    weights = gaussian_weights_unnormalized(z, centers, bandwidth)
    return normalize_projection(weights, normalization)
