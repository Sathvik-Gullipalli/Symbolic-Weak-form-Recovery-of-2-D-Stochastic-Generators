from __future__ import annotations

import argparse

from experiments.common import write_rows
from sde2d.generator import fit_generator_2d
from sde2d.library import make_library
from sde2d.metrics import central_grid, function_l2_errors
from sde2d.systems import DiagonalMultiplicative2D


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    sys = DiagonalMultiplicative2D()
    dt = 0.02
    x = sys.simulate(dt=dt, M=2500 if args.quick else 10000, seed=31)
    rows = []
    for corr in [False, True]:
        fit = fit_generator_2d(x, dt=dt, library=make_library("A"), n_centers=49 if args.quick else 100, center_scheme="quantile_grid", grid_shape=(7, 7) if args.quick else (10, 10), bandwidth_multiplier=1.5, regressor="stlsq", regression_kw={"threshold": 0.02}, bias_correct=corr, seed=31)
        pts = central_grid(x[:-1], 15)
        errs = function_l2_errors(fit, sys, pts)
        rows.append({"experiment": "core3_bias", "system": sys.name, "bias_correct": corr, "diffusion_rel_l2": errs["diffusion_rel_l2"], "drift_rel_l2": errs["drift_rel_l2"]})
    write_rows("results/scalar_reproduction/core3_bias_correction.csv", rows)
    print("wrote bias correction rows")


if __name__ == "__main__":
    main()
