from __future__ import annotations

import argparse

import numpy as np

from experiments.common import ROOT, write_rows
from sde2d.generator import fit_generator_2d
from sde2d.library import make_library
from sde2d.metrics import central_grid, function_l2_errors
from sde2d.readouts.leverage import proxy_stats, recover_heston_parameters, rho_summary_from_fit, shifted_ewma
from sde2d.systems import LogHestonSV


DT = 1.0 / 252.0


def simulate_panel(system: LogHestonSV, n_steps: int, seed: int, n_trajectories: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    states = []
    increments = []
    groups = []
    for r in range(n_trajectories):
        path = system.simulate(dt=DT, M=n_steps, seed=seed + 1009 * r)
        states.append(path[:-1])
        increments.append(np.diff(path, axis=0))
        groups.append(np.full(n_steps, r, dtype=int))
    return np.vstack(states), np.vstack(increments), np.concatenate(groups)


def fit_logheston(states: np.ndarray, increments: np.ndarray | None, groups: np.ndarray | None, seed: int, quick: bool):
    side = 7 if quick else 10
    return fit_generator_2d(
        states,
        increments=increments,
        dt=DT,
        library=make_library("D", ("X", "V")),
        n_centers=side * side,
        center_scheme="quantile_grid",
        grid_shape=(side, side),
        bandwidth_multiplier=1.5,
        regressor="lasso_stlsq",
        regression_kw={"stlsq_threshold": 0.10, "threshold_mode": "relative"},
        library_space="z",
        traj_ids=groups,
        seed=seed,
    )


def tensor_rho_row(fit, system: LogHestonSV, eval_points: np.ndarray, prefix: str = "") -> dict:
    raw = rho_summary_from_fit(fit, eval_points, psd=False)
    psd = rho_summary_from_fit(fit, eval_points, psd=True)
    params = recover_heston_parameters(fit)
    return {
        f"{prefix}rho_tensor_mean": raw["rho_tensor_mean"],
        f"{prefix}rho_tensor_median": raw["rho_tensor_median"],
        f"{prefix}rho_tensor_iqr": raw["rho_tensor_iqr"],
        f"{prefix}rho_tensor_psd_median": psd["rho_tensor_median"],
        f"{prefix}rho_parametric_hat": params["rho_hat"],
        f"{prefix}kappa_hat": params["kappa_hat"],
        f"{prefix}theta_hat": params["theta_hat"],
        f"{prefix}xi_hat": params["xi_hat"],
        f"{prefix}rho_tensor_abs_error": abs(raw["rho_tensor_median"] - system.rho) if np.isfinite(raw["rho_tensor_median"]) else float("nan"),
        f"{prefix}rho_sign_correct": bool(raw["rho_tensor_median"] * system.rho > 0) if np.isfinite(raw["rho_tensor_median"]) else False,
        f"{prefix}n_rho_points": raw["n_rho_points"],
    }


def run_regime_sweep(quick: bool) -> list[dict]:
    rhos = [-0.2, -0.65] if quick else [-0.2, -0.5, -0.65, -0.85]
    seeds = list(range(3 if quick else 30))
    rows = []
    for rho in rhos:
        system = LogHestonSV(rho=rho)
        for run, offset in enumerate(seeds):
            seed = 1000 + 37 * offset + int(abs(rho) * 100)
            states, increments, groups = simulate_panel(system, 1200 if quick else 3000, seed, 2 if quick else 4)
            fit = fit_logheston(states, increments, groups, seed, quick)
            pts = central_grid(states, 13 if quick else 17)
            errs = function_l2_errors(fit, system, pts)
            rows.append(
                {
                    "experiment": "lev1_tensor_regimes",
                    "system": system.name,
                    "rho_true": rho,
                    "seed": seed,
                    "run": run,
                    "dt": DT,
                    "R": 2 if quick else 4,
                    "n_steps_per_trajectory": 1200 if quick else 3000,
                    "diffusion_rel_l2": errs["diffusion_rel_l2"],
                    "a12_rel_l2": errs.get("a12_rel_l2", float("nan")),
                    "b1_rel_l2": errs.get("b1_rel_l2", float("nan")),
                    "b2_rel_l2": errs.get("b2_rel_l2", float("nan")),
                    **tensor_rho_row(fit, system, pts),
                }
            )
    return rows


def run_eiv_sweep(quick: bool) -> list[dict]:
    system = LogHestonSV(rho=-0.65)
    nsrs = [0.0, 0.20, 0.32, 0.50] if quick else [0.0, 0.10, 0.20, 0.32, 0.40, 0.50]
    rows = []
    base = system.simulate(dt=DT, M=2500 if quick else 9000, seed=4242)
    scale = np.std(base[:-1], axis=0, ddof=1)
    for run, nsr in enumerate(nsrs):
        rng = np.random.default_rng(5200 + run)
        noise = rng.normal(scale=nsr * scale, size=base.shape)
        noise[0] = 0.0
        observed = base + noise
        observed[:, 1] = np.maximum(observed[:, 1], system.floor)
        for corrected in [False, True]:
            fit = fit_generator_2d(
                observed,
                dt=DT,
                library=make_library("D", ("X", "V")),
                n_centers=49 if quick else 100,
                center_scheme="quantile_grid",
                grid_shape=(7, 7) if quick else (10, 10),
                bandwidth_multiplier=1.5,
                regressor="lasso_stlsq",
                regression_kw={"stlsq_threshold": 0.10, "threshold_mode": "relative"},
                library_space="z",
                noise_correct=corrected,
                seed=6100 + run,
            )
            pts = central_grid(base[:-1], 13 if quick else 17)
            errs = function_l2_errors(fit, system, pts)
            rho = tensor_rho_row(fit, system, pts)
            rows.append(
                {
                    "experiment": "lev2_eiv_phase_transition",
                    "system": system.name,
                    "rho_true": system.rho,
                    "noise_to_signal_ratio": nsr,
                    "noise_correct": corrected,
                    "seed": 5200 + run,
                    "diffusion_rel_l2": errs["diffusion_rel_l2"],
                    "a11_rel_l2": errs.get("a11_rel_l2", float("nan")),
                    "a22_rel_l2": errs.get("a22_rel_l2", float("nan")),
                    "a12_rel_l2": errs.get("a12_rel_l2", float("nan")),
                    "diagonal_break_expected": bool(nsr >= 0.32),
                    **rho,
                }
            )
    return rows


def run_ewma_proxy(quick: bool) -> list[dict]:
    system = LogHestonSV(rho=-0.65)
    path = system.simulate(dt=DT, M=3000 if quick else 10000, seed=7301)
    returns = np.diff(path[:, 0])
    raw_proxy = np.maximum((returns * returns) / DT, system.floor)
    ewma_proxy = shifted_ewma(raw_proxy, span=14)
    true_v = path[:-1, 1]
    rows = []
    for name, proxy in [("raw_realized", raw_proxy), ("shifted_ewma_span14", ewma_proxy)]:
        obs = np.column_stack([path[:-1, 0], np.maximum(proxy, system.floor)])
        fit = fit_logheston(obs, None, None, 7401 if name == "raw_realized" else 7402, quick)
        pts = central_grid(obs[:-1], 13 if quick else 17)
        nsr, corr, lag = proxy_stats(proxy, true_v)
        rows.append(
            {
                "experiment": "lev3_ewma_proxy",
                "system": system.name,
                "proxy": name,
                "rho_true": system.rho,
                "proxy_nsr": nsr,
                "proxy_corr": corr,
                "best_corr_lag": lag,
                **tensor_rho_row(fit, system, pts),
            }
        )
    return rows


def save_figures(regime_rows: list[dict], eiv_rows: list[dict]) -> None:
    import matplotlib.pyplot as plt

    out_dir = ROOT / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    if regime_rows:
        by_rho = {}
        for row in regime_rows:
            by_rho.setdefault(row["rho_true"], []).append(float(row["rho_tensor_median"]))
        fig, ax = plt.subplots(figsize=(6, 4))
        xs = sorted(by_rho)
        ax.errorbar(xs, [np.nanmedian(by_rho[x]) for x in xs], yerr=[np.nanstd(by_rho[x]) for x in xs], marker="o", capsize=4)
        ax.plot(xs, xs, color="black", linewidth=1, linestyle="--")
        ax.set_xlabel("true rho")
        ax.set_ylabel("tensor median rho_hat")
        ax.set_title("Tensor-Derived Leverage Regimes")
        fig.tight_layout()
        fig.savefig(out_dir / "leverage_tensor_regimes.png", dpi=160)
        plt.close(fig)
    if eiv_rows:
        fig, ax = plt.subplots(figsize=(6, 4))
        for corrected in [False, True]:
            rows = [r for r in eiv_rows if bool(r["noise_correct"]) is corrected]
            ax.plot([float(r["noise_to_signal_ratio"]) for r in rows], [float(r["a11_rel_l2"]) for r in rows], marker="o", label=f"a11 corrected={corrected}")
        ax.axvline(0.32, color="black", linestyle="--", linewidth=1)
        ax.set_xlabel("noise-to-signal ratio")
        ax.set_ylabel("a11 relative L2")
        ax.set_title("EIV Phase Transition")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / "leverage_eiv_phase_transition.png", dpi=160)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    regime_rows = run_regime_sweep(args.quick)
    eiv_rows = run_eiv_sweep(args.quick)
    ewma_rows = run_ewma_proxy(args.quick)
    write_rows("results/heston_cir/lev1_synthetic_regimes.csv", regime_rows)
    write_rows("results/heston_cir/lev2_eiv_phase_transition.csv", eiv_rows)
    write_rows("results/heston_cir/lev3_ewma_proxy.csv", ewma_rows)
    save_figures(regime_rows, eiv_rows)
    print(f"wrote {len(regime_rows)} leverage regime rows")
    print(f"wrote {len(eiv_rows)} EIV rows")
    print(f"wrote {len(ewma_rows)} EWMA proxy rows")


if __name__ == "__main__":
    main()
