from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Array = np.ndarray


@dataclass
class Standardizer:
    mean: Array | None = None
    scale: Array | None = None

    def fit(self, x: Array) -> "Standardizer":
        x = np.atleast_2d(np.asarray(x, float))
        self.mean = np.mean(x, axis=0)
        if x.shape[0] > 1:
            scale = np.std(x, axis=0, ddof=1)
        else:
            scale = np.ones(x.shape[1])
        self.scale = np.where(scale < 1e-12, 1.0, scale)
        return self

    @classmethod
    def from_data(cls, x: Array) -> "Standardizer":
        return cls().fit(x)

    def _check(self) -> tuple[Array, Array]:
        if self.mean is None or self.scale is None:
            raise ValueError("Standardizer is not fitted.")
        return self.mean, self.scale

    def transform(self, x: Array) -> Array:
        mean, scale = self._check()
        return (np.asarray(x, float) - mean) / scale

    def inverse_state(self, z: Array) -> Array:
        mean, scale = self._check()
        return mean + scale * np.asarray(z, float)

    def inverse_transform(self, z: Array) -> Array:
        return self.inverse_state(z)

    def drift_to_raw(self, drift_z: Array) -> Array:
        _, scale = self._check()
        return np.asarray(drift_z, float) * scale

    def diffusion_to_raw(self, diffusion_z: Array) -> Array:
        _, scale = self._check()
        a = np.asarray(diffusion_z, float)
        d = np.diag(scale)
        if a.ndim == 2:
            return d @ a @ d
        return np.einsum("ij,njk,kl->nil", d, a, d)
