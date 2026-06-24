from __future__ import annotations

import argparse

from experiments.common import write_rows
from sde2d.generator import fit_generator_2d
from sde2d.library import make_library
from sde2d.metrics import central_grid, function_l2_errors, psd_validity, tensor_metrics
from sde2d.systems import CorrelatedOU2D, RotationalOU


def run_one(system, library_name: str, seed: int, quick: bool) -> dict:
    dt = 0.01
    x = system.simulate(dt=dt, M=2000 if quick else 8000, seed=seed)
    fit = fit_generator_2d(
        x,
        dt=dt,
        library=make_library(library_name),
        n_centers=49 if quick else 100,
        center_scheme="quantile_grid",
        grid_shape=(7, 7) if quick else (10, 10),
        bandwidth_multiplier=1.5,
        regressor="stlsq",
        regression_kw={"threshold": 0.02},
        seed=seed,
    )
    pts = central_grid(x[:-1], 15 if quick else 25)
    errs = function_l2_errors(fit, system, pts)
    tmet = tensor_metrics(fit, system, pts)
    pmet = psd_validity(fit.evaluate(pts)[1])
    status = "VALIDATED_POSITIVE" if errs["drift_rel_l2"] < 0.50 and errs["diffusion_rel_l2"] < 0.15 and pmet["pct_psd_valid"] > 0.99 else "INCONCLUSIVE"
    return {
        "experiment": "core2",
        "system": system.name,
        "library": library_name,
        "seed": seed,
        "drift_rel_l2": errs["drift_rel_l2"],
        "diffusion_rel_l2": errs["diffusion_rel_l2"],
        "a12_sign_acc": tmet.get("a12_sign_accuracy", ""),
        "psd_valid_pct": pmet["pct_psd_valid"],
        "cond_design": fit.bandwidth_meta["cond_design"],
        "status": status,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    systems = [CorrelatedOU2D(), RotationalOU()]
    rows = [run_one(sys, "A", 10 + i, args.quick) for i, sys in enumerate(systems)]
    write_rows("results/2d_benchmarks/core2_drift_diffusion_recovery.csv", rows)
    write_rows("results/benchmark_summary.csv", rows)
    print(f"wrote {len(rows)} recovery rows")


if __name__ == "__main__":
    main()
