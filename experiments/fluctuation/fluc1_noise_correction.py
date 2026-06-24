from __future__ import annotations

import argparse

import numpy as np

from experiments.common import ROOT, write_rows
from sde2d.generator import fit_generator_2d
from sde2d.library import make_library
from sde2d.metrics import central_grid, cosine_similarity, function_l2_errors
from sde2d.systems import DiagonalMultiplicative2D, LogHestonSV


def add_noise(path: np.ndarray, nsr: float, kind: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    scale = nsr * np.std(path[:-1], axis=0, ddof=1)
    if kind == "laplace":
        noise = rng.laplace(scale=scale / np.sqrt(2.0), size=path.shape)
    else:
        noise = rng.normal(scale=scale, size=path.shape)
    noise[0] = 0.0
    out = path + noise
    if out.shape[1] > 1:
        out[:, 1] = np.maximum(out[:, 1], 1e-10)
    return out


def fit_tensor(path: np.ndarray, library: str, dt: float, seed: int, quick: bool, noise_correct: bool):
    side = 7 if quick else 10
    return fit_generator_2d(
        path,
        dt=dt,
        library=make_library(library, ("X", "V") if library == "D" else ("x", "y")),
        n_centers=side * side,
        center_scheme="quantile_grid",
        grid_shape=(side, side),
        bandwidth_multiplier=1.5,
        regressor="lasso_stlsq",
        regression_kw={"stlsq_threshold": 0.10, "threshold_mode": "relative"},
        library_space="z",
        noise_correct=noise_correct,
        seed=seed,
    )


def run_noise_correction(quick: bool) -> list[dict]:
    configs = [
        ("diag_multiplicative", DiagonalMultiplicative2D(), "A", 0.01, 3500 if quick else 9000),
        ("heston_logsv", LogHestonSV(), "D", 1.0 / 252.0, 3000 if quick else 9000),
    ]
    nsrs = [0.10] if quick else [0.05, 0.10]
    kinds = ["gaussian"] if quick else ["gaussian", "laplace"]
    rows = []
    for system_key, system, library, dt, n_steps in configs:
        clean = system.simulate(dt=dt, M=n_steps, seed=1201)
        pts = central_grid(clean[:-1], 13 if quick else 17)
        for nsr in nsrs:
            for kind in kinds:
                observed = add_noise(clean, nsr, kind, seed=1300 + len(rows))
                pair = {}
                for corrected in [False, True]:
                    fit = fit_tensor(observed, library, dt, seed=1400 + len(rows), quick=quick, noise_correct=corrected)
                    errs = function_l2_errors(fit, system, pts)
                    a_hat = fit.evaluate(pts)[1][:, 0, 1]
                    a_true = system.true_diffusion(pts)[:, 0, 1]
                    row = {
                        "experiment": "fluc1_noise_correction",
                        "system": system_key,
                        "library": library,
                        "noise_kind": kind,
                        "noise_to_signal_ratio": nsr,
                        "noise_correct": corrected,
                        "seed": 1400 + len(rows),
                        "diffusion_rel_l2": errs["diffusion_rel_l2"],
                        "a11_rel_l2": errs.get("a11_rel_l2", float("nan")),
                        "a22_rel_l2": errs.get("a22_rel_l2", float("nan")),
                        "a12_rel_l2": errs.get("a12_rel_l2", float("nan")),
                        "a12_cosine": cosine_similarity(a_hat, a_true),
                    }
                    rows.append(row)
                    pair[corrected] = row["diffusion_rel_l2"]
                for row in rows[-2:]:
                    row["corrected_beats_naive"] = bool(pair.get(True, np.inf) < pair.get(False, -np.inf))
    return rows


def save_figure(rows: list[dict]) -> None:
    import matplotlib.pyplot as plt

    if not rows:
        return
    out_dir = ROOT / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = []
    naive = []
    corrected = []
    for system in sorted({r["system"] for r in rows}):
        for kind in sorted({r["noise_kind"] for r in rows if r["system"] == system}):
            subset = [r for r in rows if r["system"] == system and r["noise_kind"] == kind]
            labels.append(f"{system}\n{kind}")
            naive.append(np.nanmedian([float(r["diffusion_rel_l2"]) for r in subset if not bool(r["noise_correct"])]))
            corrected.append(np.nanmedian([float(r["diffusion_rel_l2"]) for r in subset if bool(r["noise_correct"])]))
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - 0.18, naive, width=0.36, label="naive")
    ax.bar(x + 0.18, corrected, width=0.36, label="lag-1 corrected")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("diffusion relative L2")
    ax.set_title("Fluctuation Noise Correction")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "fluctuation_noise_correction.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    rows = run_noise_correction(args.quick)
    write_rows("results/fluctuation/fluc1_noise_correction.csv", rows)
    save_figure(rows)
    print(f"wrote {len(rows)} fluctuation noise-correction rows")


if __name__ == "__main__":
    main()
