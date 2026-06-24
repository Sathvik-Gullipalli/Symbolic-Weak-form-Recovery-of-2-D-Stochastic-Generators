from __future__ import annotations

import argparse

import numpy as np

from experiments.benchmarks._utils import FitCell, fit_cell, v3_default_library_space, v3_default_regressor
from experiments.common import ROOT, write_rows
from sde2d.metrics import central_grid, function_l2_errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    steps_grid = [1000, 1800] if args.quick else [1200, 2400, 4800, 7200]
    systems = ["correlated_ou"] if args.quick else ["correlated_ou", "diag_multiplicative"]
    rows = []
    for system_key in systems:
        errors_by_target: dict[str, list[tuple[float, float]]] = {"drift": [], "diffusion": []}
        interim = []
        for run, n_steps in enumerate(steps_grid):
            cell = FitCell(
                experiment="convergence",
                system_key=system_key,
                library="A",
                regressor=v3_default_regressor(2),
                n_centers=36 if args.quick else 64,
                dt=0.01,
                n_steps=n_steps,
                seed=7101,
                run=run,
                library_space=v3_default_library_space("A", 2),
            )
            system, x, fit, _ = fit_cell(cell)
            pts = central_grid(x[:-1], 13)
            errs = function_l2_errors(fit, system, pts)
            T_eff = cell.dt * n_steps
            errors_by_target["drift"].append((T_eff, errs["drift_rel_l2"]))
            errors_by_target["diffusion"].append((T_eff, errs["diffusion_rel_l2"]))
            interim.append((cell, T_eff, errs))
            print(f"{system_key:20s} T={T_eff:6.1f} drift={errs['drift_rel_l2']:.3f} diff={errs['diffusion_rel_l2']:.3f}")
        slopes = {target: slope(vals) for target, vals in errors_by_target.items()}
        floors = {target: min(err for _, err in vals) for target, vals in errors_by_target.items()}
        for cell, T_eff, errs in interim:
            for target, err_key in [("drift", "drift_rel_l2"), ("diffusion", "diffusion_rel_l2")]:
                rows.append(
                    {
                        "experiment": "convergence",
                        "system": system_key,
                        "library": cell.library,
                        "regressor": cell.regressor,
                        "varying": "T",
                        "dt": cell.dt,
                        "T_eff": T_eff,
                        "R": 1,
                        "seed": cell.seed,
                        "run": cell.run,
                        "target": target,
                        "error": errs[err_key],
                        "log_slope": slopes[target],
                        "floor_error": floors[target],
                    }
                )
    write_rows("results/convergence_slopes.csv", rows)
    save_figure(rows)
    print(f"wrote {len(rows)} convergence rows")


def slope(vals: list[tuple[float, float]]) -> float:
    clean = [(t, e) for t, e in vals if t > 0 and e > 0 and np.isfinite(e)]
    if len(clean) < 2:
        return float("nan")
    x = np.log([v[0] for v in clean])
    y = np.log([v[1] for v in clean])
    return float(np.polyfit(x, y, 1)[0])


def save_figure(rows: list[dict]) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    for system in sorted({r["system"] for r in rows}):
        for target in ["drift", "diffusion"]:
            pts = [r for r in rows if r["system"] == system and r["target"] == target]
            ax.plot([float(r["T_eff"]) for r in pts], [float(r["error"]) for r in pts], marker="o", label=f"{system} {target}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("T_eff")
    ax.set_ylabel("relative L2")
    ax.set_title("Convergence Sweep")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = ROOT / "figures/convergence_slopes.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
