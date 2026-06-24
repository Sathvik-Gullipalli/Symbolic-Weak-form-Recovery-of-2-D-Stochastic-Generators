from __future__ import annotations

import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sde2d.library import make_library
from sde2d.metrics import central_grid, cosine_similarity, function_l2_errors, psd_validity
from sde2d.standardize import Standardizer
from sde2d.systems import REGISTRY
from sde2d.wg_sindy import fit_wg_sindy, wg_sindy_defaults

from experiments.v15.v15_systems import CLUSTER, CONTROLS, DEEP_SYSTEMS, DOCUMENTED_NULLS, PILOT_SYSTEMS, dt_for

RESULTS = ROOT / "results" / "v15"
FIGURES = ROOT / "figures" / "v15"
CHOSEN_CONFIG = RESULTS / "chosen_config.json"

BASE_SEEDS = [15001, 15002, 15003]
DEEP_SEEDS = [15101 + i for i in range(10)]

METRIC_FIELDS = [
    "system",
    "seed",
    "config_id",
    "graft",
    "stack",
    "R",
    "steps",
    "dt",
    "drift_l2",
    "tensor_l2",
    "a12_cos",
    "psd_pct",
    "n_fp",
    "status",
    "notes",
]


@dataclass(frozen=True)
class V15Config:
    config_id: str
    grafts: tuple[str, ...]
    overrides: dict[str, Any]
    data_transform: str = "left"

    @property
    def stack(self) -> str:
        return "+".join(self.grafts) if self.grafts else "frozen"


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def append_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def ffloat(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def config_for_grafts(grafts: tuple[str, ...] | list[str], config_id: str | None = None, **scalar_overrides: Any) -> V15Config:
    grafts = tuple(grafts)
    overrides: dict[str, Any] = {}
    data_transform = "left"
    regression_kw: dict[str, Any] = {}
    for graft in grafts:
        if graft == "G1":
            overrides.update({"gls_mode": "full_tensor", "gls_iterations": 2, "gls_cond_cap": 1e4})
        elif graft == "G2":
            overrides.update({"selection_noise_floor": True, "noise_floor_z": 1.5})
        elif graft == "G4":
            overrides.update({"gls_mode": "full_tensor", "gls_crossfit_weights": True, "gls_iterations": 2})
        elif graft == "G5":
            overrides.update({"bandwidth_rule": "diffusion_metric"})
        elif graft == "G6":
            overrides.update({"regressor": "huber"})
            regression_kw.update({"threshold": 0.06, "epsilon": 1.35})
        elif graft == "G7":
            overrides.update({"coverage_weighting": True, "coverage_weight_floor": 0.35})
        elif graft == "G8":
            data_transform = "richardson"
        elif graft == "G9":
            overrides.update({"projection_scales": (0.7, 1.0, 1.5)})
        else:
            raise ValueError(f"unknown V15 graft {graft}")
    if regression_kw:
        overrides.setdefault("regression_kw", {}).update(regression_kw)
    overrides.update(scalar_overrides)
    return V15Config(config_id or ("V15_" + ("_".join(grafts) if grafts else "frozen")), grafts, overrides, data_transform)


def ofat_configs() -> list[V15Config]:
    return [config_for_grafts((), "frozen")] + [config_for_grafts((g,), g) for g in ["G1", "G2", "G4", "G5", "G6", "G7", "G8", "G9"]]


def simulate_pool(system: str, R: int, steps: int, seed: int, *, data_transform: str = "left") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cls = REGISTRY[system].cls
    dt = dt_for(system)
    rng = np.random.default_rng(seed)
    states: list[np.ndarray] = []
    increments: list[np.ndarray] = []
    traj_ids: list[np.ndarray] = []
    for r in range(int(R)):
        x = cls().simulate(dt=dt, M=int(steps), seed=int(rng.integers(1, 2**31 - 1)))
        if data_transform == "richardson" and x.shape[0] >= 3:
            inc1 = x[1:-1] - x[:-2]
            inc2 = x[2:] - x[:-2]
            inc = 2.0 * inc1 - 0.5 * inc2
            cur = x[:-2]
        else:
            cur = x[:-1]
            inc = np.diff(x, axis=0)
        states.append(cur)
        increments.append(inc)
        traj_ids.append(np.full(cur.shape[0], r, dtype=int))
    return np.vstack(states), np.vstack(increments), np.concatenate(traj_ids)


def true_support(system: str, states: np.ndarray, library, target: str, *, library_space: str = "z") -> np.ndarray:
    x = np.asarray(states, float)
    std = Standardizer().fit(x)
    design_points = std.transform(x) if library_space == "z" else x
    theta = library.transform(design_points)
    sys_obj = REGISTRY[system].cls()
    if target in {"b1", "b2"}:
        values = sys_obj.true_drift(x)[:, 0 if target == "b1" else 1]
    else:
        i, j = {"a11": (0, 0), "a12": (0, 1), "a22": (1, 1)}[target]
        values = sys_obj.true_diffusion(x)[:, i, j]
    coef = np.linalg.lstsq(theta, values, rcond=None)[0]
    tol = max(1e-8, 1e-4 * float(np.nanmax(np.abs(coef))) if coef.size else 1e-8)
    return np.abs(coef) > tol


def oracle_target_kw(system: str, states: np.ndarray, library, library_space: str) -> dict[str, dict[str, Any]]:
    return {
        target: {"true_support": true_support(system, states, library, target, library_space=library_space)}
        for target in ["b1", "b2", "a11", "a12", "a22"]
    }


def false_positive_count(fit, system: str, states: np.ndarray) -> int:
    count = 0
    library_space = fit.library_space
    for key, selection in fit.selections.items():
        if key.startswith("chol_"):
            continue
        if key not in {"b1", "b2", "a11", "a12", "a22"}:
            continue
        truth = true_support(system, states, fit.library, key, library_space=library_space)
        support = np.asarray(selection.support, bool)
        if support.shape == truth.shape:
            count += int(np.sum(support & ~truth))
    return count


def fit_and_score(system: str, config: V15Config, *, R: int, steps: int, seed: int, oracle: bool = False, grid_n: int = 13) -> dict[str, Any]:
    dt = dt_for(system)
    states, increments, traj_ids = simulate_pool(system, R, steps, seed, data_transform=config.data_transform)
    lib = make_library(REGISTRY[system].library, ("x", "y"))
    overrides = dict(config.overrides)
    if oracle:
        defaults = wg_sindy_defaults()
        library_space = str(overrides.get("library_space", defaults["library_space"]))
        overrides["regressor"] = "oracle_ols"
        overrides["diffusion_parameterization"] = "entries"
        overrides["diffusion_shrinkage"] = 0.0
        overrides["target_regression_kw"] = oracle_target_kw(system, states, lib, library_space)
    fit = fit_wg_sindy(states, increments, dt=dt, library=lib, seed=seed, traj_ids=traj_ids, **overrides)
    points = central_grid(states, grid_n=grid_n)
    metrics = function_l2_errors(fit, REGISTRY[system].cls(), points)
    _, a_hat = fit.evaluate(points)
    a_true = REGISTRY[system].cls().true_diffusion(points)
    psd = psd_validity(a_hat)
    a12_cos = cosine_similarity(a_hat[:, 0, 1], a_true[:, 0, 1]) if a_hat.shape[1] >= 2 else float("nan")
    return {
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
        "a12_cos": a12_cos,
        "psd_pct": psd["pct_psd_valid"],
        "n_fp": false_positive_count(fit, system, states),
        "status": "OK",
        "notes": "oracle_ols" if oracle else "",
        "b1_l2": metrics.get("b1_rel_l2", float("nan")),
        "b2_l2": metrics.get("b2_rel_l2", float("nan")),
    }


def median_by(rows: list[dict[str, Any]], keys: list[str], value: str) -> dict[tuple[Any, ...], float]:
    buckets: dict[tuple[Any, ...], list[float]] = {}
    for row in rows:
        buckets.setdefault(tuple(row[k] for k in keys), []).append(ffloat(row.get(value)))
    return {k: float(np.nanmedian(v)) if v else float("nan") for k, v in buckets.items()}


def bootstrap_ci(values: list[float], seed: int = 15150, n_boot: int = 400) -> tuple[float, float, float]:
    vals = np.asarray([v for v in values if math.isfinite(v)], float)
    if vals.size == 0:
        return float("nan"), float("nan"), float("nan")
    if vals.size == 1:
        return float(vals[0]), float(vals[0]), float(vals[0])
    rng = np.random.default_rng(seed + vals.size)
    boot = np.median(vals[rng.integers(0, vals.size, size=(n_boot, vals.size))], axis=1)
    return float(np.median(vals)), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def save_config(config: V15Config, *, reason: str, profile: dict[str, Any]) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    CHOSEN_CONFIG.write_text(
        json.dumps(
            {
                "config_id": config.config_id,
                "grafts": list(config.grafts),
                "overrides": config.overrides,
                "data_transform": config.data_transform,
                "reason": reason,
                "profile": profile,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def load_config() -> V15Config:
    data = json.loads(CHOSEN_CONFIG.read_text())
    return V15Config(str(data["config_id"]), tuple(data.get("grafts", [])), dict(data.get("overrides", {})), str(data.get("data_transform", "left")))
