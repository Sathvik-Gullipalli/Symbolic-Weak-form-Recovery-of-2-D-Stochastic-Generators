from __future__ import annotations

import numpy as np

from sde2d.invariants import spectral_gap_linear_fit


class _Library:
    names = ["1", "x", "y"]


class _Fit:
    library = _Library()
    dim = 2

    def __init__(self, jacobian: np.ndarray):
        self.drift = np.zeros((3, 2))
        self.drift[1, :] = jacobian[:, 0]
        self.drift[2, :] = jacobian[:, 1]


def test_spectral_gap_linear_fit_returns_nan_without_stable_mode() -> None:
    fit = _Fit(np.array([[0.2, 0.0], [0.0, 0.1]]))
    assert np.isnan(spectral_gap_linear_fit(fit))


def test_spectral_gap_linear_fit_uses_smallest_decay_rate() -> None:
    fit = _Fit(np.array([[-0.4, 0.0], [0.0, -1.2]]))
    assert spectral_gap_linear_fit(fit) == 0.4
