from __future__ import annotations

import argparse
from itertools import product

import numpy as np

from experiments.benchmarks._utils import FitCell, fit_cell, oracle_diagnostics
from experiments.common import ROOT, write_rows
from sde2d.metrics import central_grid, function_l2_errors, psd_validity, tensor_metrics
from sde2d.systems import REGISTRY


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    systems = ["correlated_ou", "rotational_ou"] if args.quick else ["correlated_ou", "rotational_ou", "diag_multiplicative", "nondiag_cholesky", "heston_logsv", "nonpoly_drift"]
    centers = [36] if args.quick else [36, 64, 100]
    bandwidths = [1.5] if args.quick else [1.0, 1.5, 2.0]
    schemes = ["quantile_grid"] if args.quick else ["quantile_grid", "uniform_grid"]
    regressors = ["stlsq"] if args.quick else ["stlsq", "lasso_stlsq", "ridge_threshold"]
    seeds = [3101] if args.quick else [3101, 3102]
    rows = []
    for run, (system_key, n_centers, h, scheme, regressor, seed) in enumerate(product(systems, centers, bandwidths, schemes, regressors, seeds)):
        truth = REGISTRY[system_key]
        library_space = "z" if system_key == "nondiag_cholesky" else "raw"
        n_steps = 1500 if args.quick else (9000 if system_key == "nondiag_cholesky" else 9000 if "heston" in system_key else 3500)
        n_traj = 1 if args.quick else (3 if system_key in {"nondiag_cholesky", "heston_logsv"} else 1)
        cell = FitCell(
            experiment="robustness_grid",
            system_key=system_key,
            library=truth.library,
            regressor=regressor,
            center_scheme=scheme,
            n_centers=n_centers,
            bandwidth_mult=h,
            dt=1.0 / 252.0 if "heston" in system_key else 0.01,
            n_steps=n_steps,
            seed=seed,
            run=run,
            n_trajectories=n_traj,
            library_space=library_space,
            stlsq_threshold=0.18 if system_key == "nondiag_cholesky" and regressor == "lasso_stlsq" else None,
        )
        try:
            system, x, fit, _ = fit_cell(cell)
            pts = central_grid(x, 13)
            errs = function_l2_errors(fit, system, pts)
            oracle = oracle_diagnostics(fit, system, pts)
            tmet = tensor_metrics(fit, system, pts) if truth.dim == 2 else {}
            psd = psd_validity(fit.evaluate(pts)[1])
            conditioned_ok = bool(np.isfinite(fit.bandwidth_meta["cond_design"]) and fit.bandwidth_meta["cond_design"] < 1e10)
            row = {
                "experiment": "robustness_grid",
                "system": system_key,
                "center_scheme": scheme,
                "M": fit.bandwidth_meta["n_centers"],
                "bandwidth_mult": h,
                "regressor": regressor,
                "library_space": library_space,
                "library": truth.library,
                "seed": seed,
                "run": run,
                "R": n_traj,
                "drift_rel_l2": errs["drift_rel_l2"],
                "diffusion_rel_l2": errs["diffusion_rel_l2"],
                "a12_cosine": float("nan"),
                "oracle_diffusion_rel_l2": oracle["diffusion_rel_l2"],
                "oracle_a12_cosine": oracle.get("a12_cosine", float("nan")),
                "oracle_ols_passes": oracle["oracle_ols_passes"],
                "psd_valid_pct": psd["pct_psd_valid"],
                "conditioned_ok": conditioned_ok,
            }
            if truth.dim == 2:
                true_a12 = system.true_diffusion(pts)[:, 0, 1]
                pred_a12 = fit.evaluate(pts)[1][:, 0, 1]
                den = np.linalg.norm(true_a12) * np.linalg.norm(pred_a12)
                row["a12_cosine"] = float(true_a12 @ pred_a12 / den) if den > 1e-12 else float("nan")
                row["a12_sign_acc"] = tmet.get("a12_sign_accuracy", float("nan"))
            rows.append(row)
            print(f"{system_key:20s} M={n_centers:<3d} h={h:<3.1f} {scheme:13s} {regressor:15s} drift={errs['drift_rel_l2']:.3f}")
        except Exception as exc:
            rows.append(
                {
                    "experiment": "robustness_grid",
                    "system": system_key,
                    "center_scheme": scheme,
                    "M": n_centers,
                    "bandwidth_mult": h,
                    "regressor": regressor,
                    "library_space": library_space,
                    "library": truth.library,
                    "seed": seed,
                    "run": run,
                    "R": n_traj,
                    "drift_rel_l2": float("nan"),
                    "diffusion_rel_l2": float("nan"),
                    "a12_cosine": float("nan"),
                    "oracle_diffusion_rel_l2": float("nan"),
                    "oracle_a12_cosine": float("nan"),
                    "oracle_ols_passes": False,
                    "psd_valid_pct": float("nan"),
                    "conditioned_ok": False,
                    "a12_sign_acc": float("nan"),
                    "error": repr(exc),
                }
            )
            print(f"{system_key:20s} FAILED {exc!r}")
    write_rows("results/robustness_grid.csv", rows)
    save_figure(rows)
    print(f"wrote {len(rows)} robustness rows")


def save_figure(rows: list[dict]) -> None:
    import matplotlib.pyplot as plt

    if not rows:
        return
    labels = sorted({r["system"] for r in rows})
    data = [np.nanmedian([float(r["diffusion_rel_l2"]) for r in rows if r["system"] == label]) for label in labels]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(labels, data)
    ax.set_ylabel("median diffusion relative L2")
    ax.set_title("Robustness Grid Median Error")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    out = ROOT / "figures/robustness_grid.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
