from __future__ import annotations

from sde2d.systems import REGISTRY

CLUSTER = [
    "gbm_2d",
    "two_factor_vasicek",
    "near_boundary_heston",
    "cir_pair",
    "heston_sv",
    "heston_logsv",
]

CONTROLS = [
    "indep_ou",
    "correlated_ou",
    "rotational_ou",
    "double_well_transverse",
    "nondiag_cholesky",
    "diag_multiplicative",
]

DOCUMENTED_NULLS = [
    "sabr",
    "heston_sv:b1",
    "heston_logsv:b1",
]

PILOT_SYSTEMS = CLUSTER + CONTROLS
DEEP_SYSTEMS = sorted(REGISTRY)

DT_BY_SYSTEM = {
    "gbm_2d": 1.0 / 252.0,
    "two_factor_vasicek": 1.0 / 252.0,
    "sabr": 1.0 / 252.0,
    "mueller_brown": 0.002,
    "too_large_dt": 0.05,
}


def dt_for(system: str) -> float:
    return float(DT_BY_SYSTEM.get(system, 0.01))

