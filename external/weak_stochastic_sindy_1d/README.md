# Sparse Weak-Form Discovery of Stochastic Generators

Reproducibility package for the paper:

> **Sparse Weak-Form Discovery of Stochastic Generators**  
> Eshwar R A and Gajanan V. Honnavar  
> PES University (EC Campus), Bengaluru  
> arXiv: https://arxiv.org/abs/2603.20904

---

## What this reproduces

| Fig | Output | Description |
|-----|--------|-------------|
| 1 | `results/fig1_recovered_vs_true.pdf` | Recovered vs. true drift and diffusion functions |
| 2 | `results/fig2_lasso_paths.pdf` | LassoCV regularisation paths (CV MSE vs α) |
| 3 | `results/fig3_stationary_density.pdf` | Analytical stationary densities — true vs recovered |
| 4 | `results/fig4_autocorr.pdf` | Autocorrelation functions — true vs recovered SDE |
| 5 | `results/fig5_noise_scaling.pdf` | Theoretical noise scaling: Weak Form vs Kramers–Moyal |
| 6 | `results/fig6_endogeneity_bias.pdf` | Empirical endogeneity bias of temporal vs spatial test functions vs T *(new)* |
| 7 | `results/fig7_convergence_rates.pdf` | CLT convergence rates of the spatial-kernel OLS estimator *(new)* |
| 8 | `results/fig8_hyperparameter_robustness.pdf` | Hyperparameter robustness heatmap over (h, M) *(new)* |
| 9 | `results/fig9_bias_correction.pdf` | Finite-step bias correction — OLS uncorrected vs LASSO corrected *(new)* |
| — | `results/table1.txt` | Complete coefficient recovery table (Table 1) |

Place your own plot exports in `results/` — the folder is intentionally left empty.

---

## Installation

```bash
pip install -r requirements.txt
```

Python 3.9+ required. Tested on 3.10 and 3.12.

---

## Primary interface — the notebook

The notebook **`weak_sindy_figures.ipynb`** is the single-file pipeline that
generates all nine figures and Table 1 in one top-to-bottom run.

```bash
jupyter notebook weak_sindy_figures.ipynb
# or
jupyter lab weak_sindy_figures.ipynb
```

Run all cells in order. Figures are saved as PDFs to `results/`.

**Expected runtime:** 35–55 minutes on a CPU laptop (dominated by the 120 × 3
Euler–Maruyama simulations, three grouped LassoCV fits, and the long
trajectories needed for the autocorrelation and convergence panels).

---

## Script-based runner (Figures 1–5 only)

```bash
python reproduce_all.py
```

This runs the three SDE experiments, prints Table 1, and generates Figures 1–5.
Results are cached to `results/run_cache.pkl` for fast re-runs.

```bash
python reproduce_all.py --cached     # skip simulation, regenerate figures only
```

Figures 6–9 are currently generated only by the notebook (see above).

---

## Running individual figure scripts (Figures 1–5)

After `reproduce_all.py` has written `results/run_cache.pkl`, each figure
can be regenerated independently in seconds:

```bash
python scripts/fig1_functions.py
python scripts/fig2_lasso_paths.py
python scripts/fig3_stationary_density.py
python scripts/fig4_autocorr.py
python scripts/fig5_noise_scaling.py   # purely analytical — no cache needed
```

---

## Project layout

```
Weak-Stochastic-SINDy/
├── weak_sindy_figures.ipynb  # ← primary: all 9 figures + Table 1
├── reproduce_all.py          # script-based runner (Figs 1–5)
├── requirements.txt
├── results/                  # output directory (figures + table + cache)
├── src/
│   ├── config.py             # global hyperparameters
│   ├── sde.py                # SDE definitions + Euler–Maruyama integrator
│   ├── library.py            # polynomial library + spatial Gaussian kernels
│   ├── weak_matrices.py      # weak-form matrix construction (Algorithm 1)
│   └── regression.py         # LassoCV + OLS debias + STLSQ + bias correction
└── scripts/
    ├── fig1_functions.py
    ├── fig2_lasso_paths.py
    ├── fig3_stationary_density.py
    ├── fig4_autocorr.py
    └── fig5_noise_scaling.py
```

---

## Hyperparameters (matching the paper exactly)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `DT` | 0.002 | Euler–Maruyama time step |
| `T` | 100.0 | Trajectory horizon |
| `R` | 120 | Independent trajectories per system |
| `M` | 50 | Spatial kernel centres |
| `h` (OU, DW) | 0.22 | Gaussian kernel bandwidth |
| `h` (Mult) | 0.27 | Wider bandwidth for heavier-tailed system |
| `STLS_THR` (OU, DW) | 0.25 | STLSQ relative threshold |
| `STLS_THR` (Mult) | 0.30 | STLSQ threshold for multiplicative system |
| Seeds | 42 / 123 / 7 | Random seeds (OU / double-well / multiplicative) |
| `DEG` | 4 | Polynomial library degree (monomials 1 through x⁴) |
| `ALPHA_GRID` | 10⁻⁸ … 10⁻⁰·⁵ (60 pts) | LassoCV regularisation grid |

---

## Notes

- **Figure 1** — diffusions are shown in the top row, drifts in the bottom row.
  The x-axis is clipped to [−2.5, 2.5] to avoid x³ blow-up at the edges.
- **Figure 3** — all stationary densities are computed analytically via the
  Fokker–Planck formula; no Monte Carlo variance contaminates the comparison.
- **Figure 4** — the double-well lag window is extended to 30 s to capture the
  full Kramers-time decay (~23 s). FFT-based ACF is used throughout.
- **Figure 6** — compares spatial kernel test functions (Theorem 4, unbiased)
  against temporal window test functions (Theorem 2, biased) as T grows.
- **Figure 7** — OLS error floors at large R due to finite-T Euler–Maruyama
  discretisation bias; the −½ CLT slope holds in the variance-dominated regime.
- **Figure 8** — the heatmap sweeps h ∈ {0.08 … 0.43} × M ∈ {10 … 100} on the
  double-well system; the paper default (h = 0.22, M = 50) sits near the optimum.
- **Figure 9** — the OLS-uncorrected estimator on the multiplicative diffusion
  reveals the ~13% finite-step bias; LASSO with drift-squared correction removes it.
- The code uses `np.trapz` (compatible with NumPy < 2.0 and ≥ 2.0).
