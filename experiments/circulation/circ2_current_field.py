from __future__ import annotations

import argparse

import numpy as np

from experiments.common import ROOT, write_rows
from sde2d.generator import fit_generator_2d
from sde2d.library import make_library
from sde2d.metrics import central_grid
from sde2d.readouts.circulation import conservative_bdb_decision, current_cosine, irreversibility_scalar
from sde2d.systems import DoubleWellTransverse, GradientPotential2D, IndependentOU2D, NonGradientDoubleWell, RotationalOU


def fit_circulation_system(system, library: str, seed: int, quick: bool):
    side = 7 if quick else 10
    x = system.simulate(dt=0.01, M=2500 if quick else 9000, seed=seed)
    fit = fit_generator_2d(
        x,
        dt=0.01,
        library=make_library(library),
        n_centers=side * side,
        center_scheme="quantile_grid",
        grid_shape=(side, side),
        bandwidth_multiplier=1.5,
        regressor="lasso_stlsq",
        regression_kw={"stlsq_threshold": 0.10, "threshold_mode": "relative"},
        library_space="z",
        seed=seed,
    )
    return x, fit


def run_current_field(quick: bool) -> list[dict]:
    rows = []
    rot_omegas = [1.0, 2.0] if quick else [0.5, 1.0, 2.0, 4.0]
    for omega in rot_omegas:
        system = RotationalOU(omega=omega)
        seed = int(200 + omega * 100)
        x, fit = fit_circulation_system(system, "A", seed, quick)
        pts = central_grid(x[:-1], 20)
        cosine = current_cosine(fit, system, pts)
        rows.append(
            {
                "experiment": "circ2_current_field",
                "system": system.name,
                "omega": omega,
                "library": "A",
                "current_cosine": cosine,
                "irreversibility_scalar": irreversibility_scalar(fit),
                "status": "VALIDATED_POSITIVE" if cosine > 0.95 else "INCONCLUSIVE",
            }
        )
    for system, library in [(NonGradientDoubleWell(), "B")]:
        x, fit = fit_circulation_system(system, library, 8201, quick)
        pts = central_grid(x[:-1], 20)
        rows.append(
            {
                "experiment": "circ2_current_field",
                "system": system.name,
                "omega": system.omega,
                "library": library,
                "current_cosine": current_cosine(fit, system, pts),
                "irreversibility_scalar": irreversibility_scalar(fit),
                "status": "DIAGNOSTIC_NONLINEAR_CURRENT",
            }
        )
    return rows


def run_detector_nulls(quick: bool) -> list[dict]:
    threshold = 2.5
    seeds = [0, 1] if quick else [0, 1, 2, 3, 4]
    cases = [
        ("reversible_null", IndependentOU2D(), "A", False),
        ("reversible_null", DoubleWellTransverse(beta=0.0), "B", False),
        ("reversible_null", GradientPotential2D(), "B", False),
        ("power", RotationalOU(omega=2.0), "A", True),
        ("nonlinear_diagnostic", NonGradientDoubleWell(omega=1.0), "B", True),
    ]
    rows = []
    for mode, system, library, expected_fire in cases:
        for run in seeds:
            seed = 9100 + 97 * run + len(rows)
            x, fit = fit_circulation_system(system, library, seed, quick)
            stat = irreversibility_scalar(fit)
            fired = conservative_bdb_decision(stat, threshold=threshold)
            rows.append(
                {
                    "experiment": "circ3_bdb_detector",
                    "mode": mode,
                    "system": system.name,
                    "library": library,
                    "seed": seed,
                    "run": run,
                    "irreversibility_scalar": stat,
                    "conservative_threshold": threshold,
                    "detector_fires": fired,
                    "expected_fire": expected_fire,
                    "nominal_5pct_recalibration_type1_documented": 0.333,
                    "calibration_status": "conservative_not_calibrated",
                }
            )
    return rows


def save_figures(current_rows: list[dict], detector_rows: list[dict]) -> None:
    import matplotlib.pyplot as plt

    out_dir = ROOT / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    if current_rows:
        fig, ax = plt.subplots(figsize=(7, 4))
        labels = [f"{r['system']}:{r['omega']}" for r in current_rows]
        ax.bar(labels, [float(r["current_cosine"]) for r in current_rows])
        ax.axhline(0.95, color="black", linestyle="--", linewidth=1)
        ax.set_ylabel("current cosine")
        ax.set_title("Circulation Current Field")
        ax.tick_params(axis="x", rotation=35)
        fig.tight_layout()
        fig.savefig(out_dir / "circulation_current_field.png", dpi=160)
        plt.close(fig)
    if detector_rows:
        fig, ax = plt.subplots(figsize=(7, 4))
        labels = sorted({r["system"] for r in detector_rows})
        vals = [np.mean([bool(r["detector_fires"]) for r in detector_rows if r["system"] == label]) for label in labels]
        ax.bar(labels, vals)
        ax.set_ylabel("fire rate")
        ax.set_title("Conservative BDB Detector")
        ax.tick_params(axis="x", rotation=35)
        fig.tight_layout()
        fig.savefig(out_dir / "circulation_detector_nulls.png", dpi=160)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    current_rows = run_current_field(args.quick)
    detector_rows = run_detector_nulls(args.quick)
    write_rows("results/circulation/circ2_current_field.csv", current_rows)
    write_rows("results/circulation/circ3_detector_nulls.csv", detector_rows)
    save_figures(current_rows, detector_rows)
    print(f"wrote {len(current_rows)} circulation current rows")
    print(f"wrote {len(detector_rows)} detector/null rows")


if __name__ == "__main__":
    main()
