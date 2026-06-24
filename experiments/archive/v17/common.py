from __future__ import annotations

import csv
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from experiments.v15.common import (  # noqa: E402
    BASE_SEEDS,
    DEEP_SEEDS,
    false_positive_count,
    ffloat,
    oracle_target_kw,
    true_support,
)
from experiments.v17.v17_systems import CONTROLS, TARGETS, dt_for  # noqa: E402
from sde2d.library import make_library  # noqa: E402
from sde2d.metrics import central_grid, cosine_similarity, function_l2_errors, psd_validity  # noqa: E402
from sde2d.standardize import Standardizer  # noqa: E402
from sde2d.systems import REGISTRY  # noqa: E402
from sde2d.wg_sindy import fit_wg_sindy, wg_sindy_defaults  # noqa: E402

RESULTS = ROOT / "results" / "v17"
FIGURES = ROOT / "figures" / "v17"
CACHE = RESULTS / "_traj_cache"
CHOSEN_CONFIG = RESULTS / "chosen_config.json"

V17_FIELDS = [
    "system",
    "seed",
    "config_id",
    "stack",
    "tier",
    "R",
    "steps",
    "dt",
    "coord",
    "lags",
    "moment",
    "rank",
    "domain",
    "selection",
    "library_atoms",
    "coverage_mode",
    "drift_l2",
    "tensor_l2",
    "drift_abs_l2",
    "coef_max_rel_err",
    "composite",
    "a12_cos",
    "psd_pct",
    "n_fp",
    "status",
    "notes",
]


@dataclass(frozen=True)
class V17Config:
    config_id: str
    coord: str = "none"
    drift_lags: tuple[int, ...] = (1,)
    moment: str = "euler"
    rank: str = "full"
    domain: str = "euclidean"
    selection: str = "relative"
    library_atoms: str = "poly"
    coverage_mode: str = "off"
    overrides: dict[str, Any] | None = None

    @property
    def stack(self) -> str:
        parts: list[str] = []
        if self.coord != "none":
            parts.append(f"coord={self.coord}")
        if self.drift_lags != (1,):
            parts.append("lags=" + "-".join(str(v) for v in self.drift_lags))
        if self.moment != "euler":
            parts.append(f"moment={self.moment}")
        if self.rank != "full":
            parts.append(f"rank={self.rank}")
        if self.domain != "euclidean":
            parts.append(f"domain={self.domain}")
        if self.selection != "relative":
            parts.append(f"sel={self.selection}")
        if self.library_atoms != "poly":
            parts.append(f"lib={self.library_atoms}")
        if self.coverage_mode != "off":
            parts.append(f"coverage={self.coverage_mode}")
        return "+".join(parts) if parts else "frozen"

    def fit_overrides(self) -> dict[str, Any]:
        out = dict(self.overrides or {})
        out.update(
            {
                "coord_transform": self.coord,
                "drift_lags": self.drift_lags,
                "moment_order": self.moment,
                "tensor_rank": self.rank,
                "domain": self.domain,
                "library_atoms": self.library_atoms,
                "coverage_mode": self.coverage_mode,
            }
        )
        if self.selection == "noise_floor":
            out.update({"selection_noise_floor": True, "noise_floor_z": 1.5})
        if self.rank == "auto":
            out.setdefault("rank_floor", 0.01)
        if self.coverage_mode in {"reweight", "both"}:
            out.setdefault("coverage_weighting", True)
            out.setdefault("coverage_weight_floor", 0.25)
        return out


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


def cache_path(system: str, R: int, steps: int, seed: int, max_lag: int) -> Path:
    return CACHE / f"{system}_R{R}_M{steps}_seed{seed}_lag{max_lag}.npz"


def simulate_pool_cached(system: str, R: int, steps: int, seed: int, *, max_lag: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = cache_path(system, R, steps, seed, max_lag)
    if path.exists():
        data = np.load(path)
        return data["states"], data["path"], data["traj_ids"]
    cls = REGISTRY[system].cls
    dt = dt_for(system)
    rng = np.random.default_rng(seed)
    states: list[np.ndarray] = []
    traj_ids: list[np.ndarray] = []
    for r in range(int(R)):
        x = cls().simulate(dt=dt, M=int(steps) + int(max_lag), seed=int(rng.integers(1, 2**31 - 1)))
        states.append(x)
        traj_ids.append(np.full(x.shape[0] - max_lag, r, dtype=int))
    max_len = max(s.shape[0] for s in states)
    dim = states[0].shape[1]
    packed = np.full((len(states), max_len, dim), np.nan)
    for i, s in enumerate(states):
        packed[i, : s.shape[0], :] = s
    cur = np.vstack([s[: s.shape[0] - max_lag] for s in states])
    ids = np.concatenate(traj_ids)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, states=cur, path=packed, traj_ids=ids)
    return cur, packed, ids


def increments_for_lags(path: np.ndarray, lags: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lags = tuple(int(v) for v in lags)
    max_lag = max(lags)
    states: list[np.ndarray] = []
    incs: list[np.ndarray] = []
    ids: list[np.ndarray] = []
    for r, packed in enumerate(path):
        valid = packed[np.isfinite(packed[:, 0])]
        M_eff = valid.shape[0] - max_lag
        cur = valid[:M_eff]
        stack = np.zeros((M_eff, len(lags), valid.shape[1]))
        for j, lag in enumerate(lags):
            stack[:, j, :] = valid[lag : lag + M_eff] - cur
        states.append(cur)
        incs.append(stack)
        ids.append(np.full(M_eff, r, dtype=int))
    inc = np.vstack(incs)
    if len(lags) == 1:
        inc = inc[:, 0, :]
    return np.vstack(states), inc, np.concatenate(ids)


def config_for_grafts_v17(
    *,
    coord: str = "none",
    drift_lags: tuple[int, ...] = (1,),
    moment: str = "euler",
    rank: str = "full",
    domain: str = "euclidean",
    selection: str = "relative",
    library_atoms: str = "poly",
    coverage_mode: str = "off",
    config_id: str | None = None,
    **overrides: Any,
) -> V17Config:
    cfg = V17Config(
        config_id=config_id or "V17",
        coord=coord,
        drift_lags=tuple(drift_lags),
        moment=moment,
        rank=rank,
        domain=domain,
        selection=selection,
        library_atoms=library_atoms,
        coverage_mode=coverage_mode,
        overrides=overrides,
    )
    if config_id is None:
        object.__setattr__(cfg, "config_id", "V17_" + cfg.stack.replace("+", "_").replace("=", "-"))
    return cfg


def library_for(system: str, atoms: str):
    preset = REGISTRY[system].library
    if atoms == "poly+trig":
        preset = "POLY+TRIG"
    elif atoms == "poly+rational":
        preset = "POLY+RATIONAL"
    elif atoms == "poly+rbf":
        preset = "POLY+RBF"
    return make_library(preset, ("x", "y"))


def fit_and_score_from_data(
    system: str,
    config: V17Config,
    states: np.ndarray,
    increments: np.ndarray,
    traj_ids: np.ndarray,
    *,
    seed: int,
    oracle: bool = False,
    grid_n: int = 11,
) -> dict[str, Any]:
    dt = dt_for(system)
    lib = library_for(system, config.library_atoms)
    overrides = config.fit_overrides()
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
    b_hat, a_hat = fit.evaluate(points)
    sys_obj = REGISTRY[system].cls()
    b_true = sys_obj.true_drift(points)
    a_true = sys_obj.true_diffusion(points)
    psd = psd_validity(a_hat)
    drift_abs = float(np.sqrt(np.mean(np.sum((b_hat - b_true) ** 2, axis=1))))
    theta = lib.transform(points)
    coef_true = np.linalg.lstsq(theta, b_true, rcond=None)[0]
    coef_err = np.abs(fit.drift - coef_true)
    coef_rel = np.divide(coef_err, np.maximum(np.abs(coef_true), 1e-8))
    coef_max = float(np.nanmax(coef_rel)) if coef_rel.size else float("nan")
    drift_rel = ffloat(metrics.get("drift_rel_l2"))
    tensor_rel = ffloat(metrics.get("diffusion_rel_l2"))
    drift_scale = float(np.sqrt(np.mean(np.sum(b_true * b_true, axis=1))))
    if drift_scale < 1e-10:
        composite = tensor_rel
        drift_rel = float("nan")
        note = "zero_drift_metric_na"
    else:
        composite = max(drift_abs / max(drift_scale, 1e-3), drift_rel if math.isfinite(drift_rel) else 0.0)
        note = "oracle_ols" if oracle else ""
    a12_cos = cosine_similarity(a_hat[:, 0, 1], a_true[:, 0, 1]) if a_hat.shape[1] >= 2 else float("nan")
    return {
        "system": system,
        "seed": seed,
        "config_id": config.config_id + ("_oracle" if oracle else ""),
        "stack": config.stack,
        "tier": "control" if system in CONTROLS else "target",
        "R": int(len(np.unique(traj_ids))),
        "steps": int(states.shape[0] / max(len(np.unique(traj_ids)), 1)),
        "dt": dt,
        "coord": config.coord,
        "lags": "-".join(str(v) for v in config.drift_lags),
        "moment": config.moment,
        "rank": config.rank,
        "domain": config.domain,
        "selection": config.selection,
        "library_atoms": config.library_atoms,
        "coverage_mode": config.coverage_mode,
        "drift_l2": drift_rel,
        "tensor_l2": tensor_rel,
        "drift_abs_l2": drift_abs,
        "coef_max_rel_err": coef_max,
        "composite": composite,
        "a12_cos": a12_cos,
        "psd_pct": psd["pct_psd_valid"],
        "n_fp": false_positive_count(fit, system, states),
        "status": "OK",
        "notes": note,
    }


def save_config(config: V17Config, *, reason: str, profile: dict[str, Any]) -> None:
    CHOSEN_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(config)
    payload["stack"] = config.stack
    payload["reason"] = reason
    payload["profile"] = profile
    CHOSEN_CONFIG.write_text(json.dumps(payload, indent=2, default=str) + "\n")


def load_config() -> V17Config:
    if not CHOSEN_CONFIG.exists():
        return config_for_grafts_v17(config_id="frozen")
    data = json.loads(CHOSEN_CONFIG.read_text())
    return V17Config(
        config_id=data.get("config_id", "frozen"),
        coord=data.get("coord", "none"),
        drift_lags=tuple(data.get("drift_lags", (1,))),
        moment=data.get("moment", "euler"),
        rank=data.get("rank", "full"),
        domain=data.get("domain", "euclidean"),
        selection=data.get("selection", "relative"),
        library_atoms=data.get("library_atoms", "poly"),
        coverage_mode=data.get("coverage_mode", "off"),
        overrides=data.get("overrides") or {},
    )


def median_by(rows: list[dict[str, Any]], key: str, metric: str = "composite") -> dict[str, float]:
    out: dict[str, list[float]] = {}
    for row in rows:
        out.setdefault(str(row[key]), []).append(ffloat(row[metric]))
    return {k: float(np.nanmedian(v)) for k, v in out.items()}

