from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sde2d.library import make_library
from sde2d.systems import REGISTRY
from sde2d.wg_sindy import fit_wg_sindy


# Working systems + active true terms (drift and nonzero diffusion), with dt per system.
SPEC: dict[str, tuple[str, float, list[tuple[str, str, float]]]] = {
    "correlated_ou": ("A", 0.01, [("b1", "x", -1), ("b2", "y", -1.5), ("a11", "1", 1), ("a12", "1", -0.48), ("a22", "1", 0.64)]),
    "coupled_ou": ("A", 0.01, [("b1", "x", -1), ("b1", "y", 0.5), ("b2", "x", 0.5), ("b2", "y", -1), ("a11", "1", 1), ("a22", "1", 1)]),
    "rotational_ou": ("A", 0.01, [("b1", "x", -1), ("b1", "y", -2), ("b2", "x", 2), ("b2", "y", -1), ("a11", "1", 1), ("a22", "1", 1)]),
    "spiral_sink_corr": ("A", 0.01, [("b1", "x", -1), ("b1", "y", -1.5), ("b2", "x", 1.5), ("b2", "y", -1), ("a11", "1", 1), ("a12", "1", -0.4), ("a22", "1", 0.64)]),
    "van_der_pol": ("B", 0.01, [("b1", "y", 1), ("b2", "x", -1), ("b2", "y", 1.2), ("b2", "x^2y", -1.2), ("a11", "1", 0.1225), ("a22", "1", 0.1225)]),
    "stuart_landau": ("B", 0.01, [("b1", "x", 1), ("b1", "y", -1.5), ("b1", "x^3", -1), ("b1", "xy^2", -1), ("b2", "x", 1.5), ("b2", "y", 1), ("b2", "x^2y", -1), ("b2", "y^3", -1), ("a11", "1", 0.0784), ("a22", "1", 0.0784)]),
    "brusselator": ("B", 0.01, [("b1", "1", 1), ("b1", "x", -3.6), ("b1", "x^2y", 1), ("b2", "x", 2.6), ("b2", "x^2y", -1), ("a11", "1", 0.0144), ("a22", "1", 0.0144)]),
    "duffing": ("B", 0.01, [("b1", "y", 1), ("b2", "x", 1), ("b2", "y", -0.35), ("b2", "x^3", -1), ("a11", "1", 0.09), ("a22", "1", 0.09)]),
    "maier_stein": ("B", 0.01, [("b1", "x", 1), ("b1", "x^3", -1), ("b1", "xy^2", -0.35), ("b2", "y", -1), ("b2", "x^2y", -1), ("a11", "1", 0.1225), ("a22", "1", 0.1225)]),
    "gradient_potential": ("B", 0.01, [("b1", "x", 1), ("b1", "x^3", -1), ("b1", "xy^2", -0.5), ("b2", "y", -1), ("b2", "x^2y", -0.5), ("a11", "1", 0.245), ("a22", "1", 0.245)]),
    "diag_multiplicative": ("A", 0.01, [("b1", "x", -1), ("b2", "y", -1), ("a11", "1", 0.5), ("a11", "x^2", 0.1), ("a22", "1", 0.4), ("a22", "y^2", 0.1)]),
    "nondiag_cholesky": ("C", 0.01, [("b1", "x", -1), ("b2", "y", -1), ("a12", "xy", 0.1), ("a11", "1", 0.25), ("a22", "1", 0.16)]),
}

FIELDS = ["system", "target", "term", "true", "wg_median_when_sel", "wg_sel_rate", "km_baseline", "R", "seeds"]


def feature_names(lib: str) -> list[str]:
    return list(make_library(lib, ("x", "y")).names)


def pooled(cls, dt: float, r_count: int, seed: int, steps: int = 8000) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(1000 + seed)
    states, increments = [], []
    for _ in range(r_count):
        x = cls().simulate(dt=dt, M=steps, seed=int(rng.integers(1, 10**8)))
        states.append(x[:-1])
        increments.append(np.diff(x, axis=0))
    return np.vstack(states), np.vstack(increments)


def km_fit(states: np.ndarray, increments: np.ndarray, lib: str, dt: float) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    theta = make_library(lib, ("x", "y")).transform(states)
    drift = np.linalg.lstsq(theta, increments / dt, rcond=None)[0]
    outer = increments[:, :, None] * increments[:, None, :] / dt
    diffusion = {
        key: np.linalg.lstsq(theta, outer[:, i, j], rcond=None)[0]
        for key, (i, j) in {"a11": (0, 0), "a12": (0, 1), "a22": (1, 1)}.items()
    }
    return drift, diffusion


def get_coef(drift: np.ndarray | None, diffusion: dict[str, np.ndarray] | None, names: list[str], target: str, term: str) -> float:
    if term not in names:
        return 0.0
    idx = names.index(term)
    if target == "b1":
        return float(drift[idx, 0]) if drift is not None else 0.0
    if target == "b2":
        return float(drift[idx, 1]) if drift is not None else 0.0
    return float(diffusion[target][idx]) if diffusion is not None else 0.0


def write_rows(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def run_rerun(r_count: int = 32, seeds: int = 4, steps: int = 8000, output: Path | None = None) -> Path:
    output = output or ROOT / "results" / "coefficient_recovery" / f"coeff_recovery_R{r_count}.csv"
    rows: list[dict[str, object]] = []
    for name, (lib, dt, terms) in SPEC.items():
        names = feature_names(lib)
        wg_values: dict[tuple[str, str, float], list[float]] = {term: [] for term in terms}
        for seed in range(seeds):
            states, increments = pooled(REGISTRY[name].cls, dt, r_count, seed, steps=steps)
            fit = fit_wg_sindy(states, increments, dt=dt, library=make_library(lib, ("x", "y")), seed=seed)
            for target, term, truth in terms:
                if target in ("b1", "b2"):
                    value = get_coef(fit.drift, None, names, target, term)
                else:
                    key = {"a11": (0, 0), "a12": (0, 1), "a22": (1, 1)}[target]
                    idx = names.index(term) if term in names else None
                    value = float(fit.diffusion[key][idx]) if idx is not None else 0.0
                wg_values[(target, term, truth)].append(value)
        states, increments = pooled(REGISTRY[name].cls, dt, r_count, 0, steps=steps)
        km_drift, km_diffusion = km_fit(states, increments, lib, dt)
        for (target, term, truth), values in wg_values.items():
            values_arr = np.asarray(values, float)
            selected = np.abs(values_arr) > 1e-3
            rows.append(
                {
                    "system": name,
                    "target": target,
                    "term": term,
                    "true": truth,
                    "wg_median_when_sel": float(np.median(values_arr[selected])) if selected.any() else 0.0,
                    "wg_sel_rate": float(selected.mean()),
                    "km_baseline": get_coef(km_drift, km_diffusion, names, target, term),
                    "R": r_count,
                    "seeds": seeds,
                }
            )
        write_rows(rows, output)
        print(f"done {name} -> {output} ({len(rows)} rows)", flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="R-pooled coefficient rerun for the V10/V11 manuscript.")
    parser.add_argument("--R", type=int, default=32)
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--steps", type=int, default=8000)
    args = parser.parse_args()
    output = run_rerun(r_count=args.R, seeds=args.seeds, steps=args.steps)
    print(f"WROTE {output}", flush=True)


if __name__ == "__main__":
    main()
