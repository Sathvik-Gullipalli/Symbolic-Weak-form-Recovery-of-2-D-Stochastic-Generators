from __future__ import annotations

import numpy as np

from ..diffusion import lag1_noise_covariance, project_psd
from ..kernels import gaussian_weights_unnormalized

Array = np.ndarray


def tensor_field_from_fit(fit, x: Array, psd: bool = False) -> Array:
    return fit.evaluate(x, psd=psd)[1]


def local_diffusion_tensor(states: Array, increments: Array, dt: float, eval_points: Array, bandwidth: float, noise_correct: bool = False, noise_cov: Array | None = None, psd: bool = False) -> Array:
    states = np.asarray(states, float)
    inc = np.asarray(increments, float)
    eval_points = np.asarray(eval_points, float)
    weights = gaussian_weights_unnormalized(states, eval_points, bandwidth)
    weights = weights / np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
    outer = inc[:, :, None] * inc[:, None, :]
    if noise_correct:
        if noise_cov is None:
            noise_cov = lag1_noise_covariance(np.vstack([states[:1], states + inc]))
        outer = outer - 2.0 * noise_cov[None, :, :]
    field = np.einsum("mn,nij->mij", weights, outer) / dt
    return project_psd(field) if psd else field


def corrected_vs_naive(states: Array, increments: Array, dt: float, eval_points: Array, bandwidth: float) -> dict[str, Array]:
    naive = local_diffusion_tensor(states, increments, dt, eval_points, bandwidth, noise_correct=False)
    corrected = local_diffusion_tensor(states, increments, dt, eval_points, bandwidth, noise_correct=True)
    return {"naive": naive, "corrected": corrected}
