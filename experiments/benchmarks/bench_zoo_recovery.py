from __future__ import annotations

import argparse
from collections import defaultdict

from experiments.benchmarks._utils import FitCell, fit_cell, rows_for_fit, save_status_figure, v3_default_library_space, v3_default_regressor, write_campaign_tables
from experiments.common import write_rows
from sde2d.systems import REGISTRY


QUICK_KEYS = ["ou", "correlated_ou", "coupled_ou", "rotational_ou", "diag_multiplicative", "heston_logsv", "nonpoly_drift"]


def cell_for(system_key: str, seed: int, run: int, quick: bool) -> FitCell:
    truth = REGISTRY[system_key]
    n_steps = 1800 if quick else 5000
    n_centers = 36 if quick else 81
    dt = 0.01
    regressor = v3_default_regressor(truth.dim)
    library_space = v3_default_library_space(truth.library, truth.dim)
    n_trajectories = 1 if quick else 4
    threshold = None
    stlsq_threshold = None
    if truth.dim == 1:
        n_centers = 30 if quick else 50
        regressor = "stlsq"
        library_space = "raw"
    if system_key in {"heston_logsv", "heston_sv", "cir_pair", "near_boundary_heston"}:
        dt = 1.0 / 252.0
        n_steps = 2500 if quick else 20000
        n_centers = 36 if quick else 100
    if system_key == "heston_sv":
        n_steps = 2000 if quick else 8000
    if system_key == "nondiag_cholesky":
        n_steps = 2500 if quick else 20000
        n_centers = 36 if quick else 144
        regressor = "lasso_stlsq"
        library_space = "z"
        stlsq_threshold = 0.18
    if system_key in {"bad_coverage", "too_large_dt"}:
        n_steps = 1200 if quick else 3000
        n_trajectories = 1
    if system_key == "too_large_dt":
        dt = 0.05
    return FitCell(
        experiment="bench_zoo",
        system_key=system_key,
        library=truth.library,
        regressor=regressor,
        n_centers=n_centers,
        dt=dt,
        n_steps=n_steps,
        seed=seed,
        run=run,
        n_trajectories=n_trajectories,
        library_space=library_space,
        threshold=threshold,
        stlsq_threshold=stlsq_threshold,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    keys = QUICK_KEYS if args.quick else list(REGISTRY)
    seeds = [1101] if args.quick else [1101, 1102, 1103, 1104, 1105]
    tables: dict[str, list[dict]] = defaultdict(list)
    phase_rows = []
    for run, seed in enumerate(seeds):
        for system_key in keys:
            cell = cell_for(system_key, seed + run * 17, run, args.quick)
            try:
                system, x, fit, runtime = fit_cell(cell)
                rows = rows_for_fit(cell, system, x, fit, runtime)
                for name, part in rows.items():
                    tables[name].extend(part)
                status = rows["benchmark_summary"][0]["status"]
                phase_rows.append({"phase": "zoo", "system": system_key, "seed": cell.seed, "status": status, "runtime_sec": runtime})
                print(f"{system_key:24s} seed={cell.seed} {status}")
            except Exception as exc:
                truth = REGISTRY[system_key]
                tables["benchmark_summary"].append(
                    {
                        "experiment": "bench_zoo",
                        "system": system_key,
                        "tier": truth.tier,
                        "dim": truth.dim,
                        "library": truth.library,
                        "center_scheme": "quantile_grid",
                        "M": cell.n_centers,
                        "bandwidth_mult": cell.bandwidth_mult,
                        "regressor": cell.regressor,
                        "library_space": cell.library_space,
                        "dt": cell.dt,
                        "T": cell.dt * cell.n_steps,
                        "R": cell.n_trajectories,
                        "n_steps": cell.n_steps,
                        "seed": cell.seed,
                        "run": cell.run,
                        "noise_level": 0.0,
                        "noise_kind": "none",
                        "subsample_k": 1,
                        "b1_rel_l2": float("nan"),
                        "b2_rel_l2": float("nan"),
                        "drift_rel_l2": float("nan"),
                        "diffusion_rel_l2": float("nan"),
                        "a12_cosine": float("nan"),
                        "a12_sign_acc": float("nan"),
                        "psd_valid_pct": float("nan"),
                        "oracle_drift_rel_l2": float("nan"),
                        "oracle_diffusion_rel_l2": float("nan"),
                        "oracle_a12_rel_l2": float("nan"),
                        "oracle_a12_cosine": float("nan"),
                        "oracle_a12_sign_acc": float("nan"),
                        "oracle_ols_passes": False,
                        "support_exact_match": False,
                        "drift_pass_level": "fail",
                        "tensor_pass_level": "fail",
                        "pass_level": "fail",
                        "status": "FAILED",
                        "runtime_sec": 0.0,
                    }
                )
                phase_rows.append({"phase": "zoo", "system": system_key, "seed": cell.seed, "status": "FAILED", "runtime_sec": 0.0, "error": repr(exc)})
                print(f"{system_key:24s} seed={cell.seed} FAILED {exc!r}")
    write_campaign_tables(tables, overwrite=True)
    write_rows("results/phase_run_log.csv", phase_rows)
    save_status_figure("results/benchmark_summary.csv", "figures/benchmark_zoo_errors.png", "Full Zoo Recovery Errors")
    print(f"wrote zoo tables for {len(tables['benchmark_summary'])} cells")


if __name__ == "__main__":
    main()
