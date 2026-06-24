from __future__ import annotations

import numpy as np

from ..invariants import drift_jacobian_from_fit
from ..metrics import cosine_similarity

Array = np.ndarray


def antisymmetric_part(matrix: Array) -> Array:
    matrix = np.asarray(matrix, float)
    return 0.5 * (matrix - matrix.T)


def current_field_from_fit(fit, x: Array) -> Array:
    jac = drift_jacobian_from_fit(fit)
    anti = antisymmetric_part(jac)
    return np.asarray(x, float) @ anti.T


def current_cosine(fit, system, x: Array) -> float:
    truth = system.true_current(x)
    if truth is None:
        return float("nan")
    pred = current_field_from_fit(fit, x)
    return cosine_similarity(pred, truth)


def irreversibility_scalar(fit) -> float:
    return float(np.linalg.norm(antisymmetric_part(drift_jacobian_from_fit(fit)), ord="fro"))


def conservative_bdb_decision(stat: float, threshold: float = 0.25) -> bool:
    return bool(stat > threshold)
