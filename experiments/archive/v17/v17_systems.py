from __future__ import annotations

from experiments.v15.v15_systems import CONTROLS, DEEP_SYSTEMS, DT_BY_SYSTEM, dt_for

TARGETS = [
    "near_singular",
    "underdamped_langevin",
    "near_boundary_heston",
    "nonpoly_drift",
    "bad_coverage",
    "too_large_dt",
    "mueller_brown",
    "sabr",
    "gbm_2d",
    "two_factor_vasicek",
]

TIER = {
    "two_factor_vasicek": "A",
    "sabr": "A",
    "too_large_dt": "A",
    "near_boundary_heston": "A",
    "underdamped_langevin": "B",
    "near_singular": "B",
    "gbm_2d": "B",
    "nonpoly_drift": "C",
    "bad_coverage": "C",
    "mueller_brown": "C",
}

PILOT_SYSTEMS = TARGETS + CONTROLS

