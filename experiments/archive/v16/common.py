from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from experiments.v15.common import (
    ROOT, SRC, BASE_SEEDS, DEEP_SEEDS, METRIC_FIELDS,
    V15Config, write_csv, append_csv, read_csv, ffloat,
    true_support, oracle_target_kw, false_positive_count,
    median_by, bootstrap_ci,
)

from sde2d.library import make_library
from sde2d.metrics import central_grid, cosine_similarity, function_l2_errors, psd_validity
from sde2d.systems import REGISTRY
from sde2d.wg_sindy import fit_wg_sindy, wg_sindy_defaults
from experiments.v16.v16_systems import dt_for

RESULTS = ROOT / "results" / "v16"
FIGURES = ROOT / "figures" / "v16"
CHOSEN_CONFIG = RESULTS / "chosen_config.json"

def config_for_grafts_v16(
    coord_transform: str = "none",
    drift_lags: tuple[int, ...] = (1,),
    lag_bias_correct: bool = False,
    moment_order: str = "euler",
    gls_mode: str = "diagonal",
    selection: str = "relative",
    config_id: str | None = None,
    **scalar_overrides: Any,
) -> V15Config:
    overrides: dict[str, Any] = {
        "coord_transform": coord_transform,
        "drift_lags": tuple(drift_lags),
        "lag_bias_correct": lag_bias_correct,
        "moment_order": moment_order,
    }
    grafts = []
    if coord_transform != "none": grafts.append(f"coord={coord_transform}")
    if tuple(drift_lags) != (1,): grafts.append(f"lags={drift_lags}")
    if moment_order != "euler": grafts.append(f"moment={moment_order}")
    
    if gls_mode == "full_tensor":
        overrides.update({"gls_mode": "full_tensor", "gls_iterations": 2, "gls_cond_cap": 1e4})
        grafts.append("gls=full_tensor")
    else:
        overrides.update({"gls_mode": gls_mode})
        
    regression_kw = {}
    if selection == "noise_floor":
        overrides.update({"selection_noise_floor": True, "noise_floor_z": 1.5})
        grafts.append("sel=noise_floor")
    
    if regression_kw:
        overrides.setdefault("regression_kw", {}).update(regression_kw)
    overrides.update(scalar_overrides)
    
    c_id = config_id or ("V16_" + ("_".join(grafts) if grafts else "frozen"))
    return V15Config(c_id, tuple(grafts), overrides, data_transform="left")

def simulate_pool(system: str, R: int, steps: int, seed: int, *, data_transform: str = "left", lags: tuple[int, ...] | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cls = REGISTRY[system].cls
    dt = dt_for(system)
    rng = np.random.default_rng(seed)
    states: list[np.ndarray] = []
    increments: list[np.ndarray] = []
    traj_ids: list[np.ndarray] = []
    
    max_lag = 1
    if lags is not None and len(lags) > 0:
        max_lag = max(lags)
        
    for r in range(int(R)):
        x = cls().simulate(dt=dt, M=int(steps) + max_lag, seed=int(rng.integers(1, 2**31 - 1)))
        
        if lags is not None and len(lags) > 0:
            M_eff = x.shape[0] - max_lag
            cur = x[:M_eff]
            inc_stack = np.zeros((len(lags), M_eff, x.shape[1]))
            for i, lag in enumerate(lags):
                inc_stack[i] = x[lag : lag + M_eff] - cur
            
            states.append(cur)
            increments.append(inc_stack)
            traj_ids.append(np.full(cur.shape[0], r, dtype=int))
        elif data_transform == "richardson" and x.shape[0] >= 3:
            inc1 = x[1:-1] - x[:-2]
            inc2 = x[2:] - x[:-2]
            inc = 2.0 * inc1 - 0.5 * inc2
            cur = x[:-2]
            states.append(cur)
            increments.append(inc)
            traj_ids.append(np.full(cur.shape[0], r, dtype=int))
        else:
            cur = x[:-1]
            inc = np.diff(x, axis=0)
            states.append(cur)
            increments.append(inc)
            traj_ids.append(np.full(cur.shape[0], r, dtype=int))
            
    if lags is not None and len(lags) > 0:
        states_v = np.vstack(states)
        inc_v = np.concatenate([np.transpose(inc, (1, 0, 2)) for inc in increments], axis=0)
        return states_v, inc_v, np.concatenate(traj_ids)
    
    return np.vstack(states), np.vstack(increments), np.concatenate(traj_ids)

def fit_and_score(system: str, config: V15Config, *, R: int, steps: int, seed: int, oracle: bool = False, grid_n: int = 13, precomputed_data=None, precomputed_projection=None, return_projection=False) -> dict[str, Any] | tuple[dict[str, Any], tuple]:
    dt = dt_for(system)
    
    lags = config.overrides.get("drift_lags", None)
    if lags == (1,):
        lags = None
        
    if precomputed_data is not None:
        states, increments, traj_ids = precomputed_data
        if increments.ndim == 3:
            if lags is None:
                increments = increments[:, 0, :]
            else:
                increments = increments[:, :len(lags), :]
    else:
        states, increments, traj_ids = simulate_pool(system, R, steps, seed, data_transform=config.data_transform, lags=lags)
        
    lib = make_library(REGISTRY[system].library, ("x", "y"))
    overrides = dict(config.overrides)
    if precomputed_projection is not None:
        overrides["precomputed_projection"] = precomputed_projection
    if return_projection:
        overrides["return_projection"] = True
        
    if oracle:
        defaults = wg_sindy_defaults()
        library_space = str(overrides.get("library_space", defaults["library_space"]))
        overrides["regressor"] = "oracle_ols"
        overrides["diffusion_parameterization"] = "entries"
        overrides["diffusion_shrinkage"] = 0.0
        overrides["target_regression_kw"] = oracle_target_kw(system, states, lib, library_space)
        
    fit_res = fit_wg_sindy(states, increments, dt=dt, library=lib, seed=seed, traj_ids=traj_ids, **overrides)
    if return_projection:
        fit, proj = fit_res
    else:
        fit = fit_res
        proj = None
    points = central_grid(states, grid_n=grid_n)
    metrics = function_l2_errors(fit, REGISTRY[system].cls(), points)
    _, a_hat = fit.evaluate(points)
    a_true = REGISTRY[system].cls().true_diffusion(points)
    psd = psd_validity(a_hat)
    a12_cos = cosine_similarity(a_hat[:, 0, 1], a_true[:, 0, 1]) if a_hat.shape[1] >= 2 else float("nan")
    
    sys_obj = REGISTRY[system].cls()
    b_true = sys_obj.true_drift(points)
    b_hat, _ = fit.evaluate(points)
    b_abs_err = np.linalg.norm(b_hat - b_true, axis=1)
    drift_abs_l2 = float(np.sqrt(np.mean(b_abs_err**2)))
    
    b_coef_hat = fit.drift
    sys_theta = lib.transform(points)
    b_coef_true = np.linalg.lstsq(sys_theta, b_true, rcond=None)[0]
    
    coef_err = np.abs(b_coef_hat - b_coef_true)
    max_rel_err = float(np.max(np.divide(coef_err, np.abs(b_coef_true), out=np.zeros_like(coef_err), where=np.abs(b_coef_true)>1e-8)))
    
    # Ensure composite objective metric
    # Let composite = max(drift_rel_l2, drift_abs_l2) or coefficient err based.
    # The prompt says: "robust composite (e.g. relative error floored by an absolute-scale term, or coefficient-level error)"
    composite_drift = float(max(drift_abs_l2, metrics["drift_rel_l2"]))
    
    ret = {
        "system": system,
        "seed": seed,
        "config_id": config.config_id,
        "graft": config.config_id,
        "stack": config.stack,
        "R": int(R),
        "steps": int(steps),
        "dt": dt,
        "drift_l2": metrics["drift_rel_l2"],
        "tensor_l2": metrics["diffusion_rel_l2"],
        "drift_abs_l2": drift_abs_l2,
        "coef_max_rel_err": max_rel_err,
        "composite_drift": composite_drift,
        "a12_cos": a12_cos,
        "psd_pct": psd["pct_psd_valid"],
        "n_fp": false_positive_count(fit, system, states),
        "status": "OK",
        "notes": "oracle_ols" if oracle else "",
        "b1_l2": metrics.get("b1_rel_l2", float("nan")),
        "b2_l2": metrics.get("b2_rel_l2", float("nan")),
    }
    if return_projection:
        return ret, proj
    return ret
