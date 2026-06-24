# 2D Weak-Form Stochastic Generator Recovery

Reproducibility package for:

> **Recovering two-dimensional Itô generators from trajectory data**
> [Sai Sathvik Gullipalli - Eshwar R A - Gajanan V. Honnavar] — [ arXiv link (will update once link is avaliable)]

Extends the 1D spatial-Gaussian-kernel weak-form estimator of Eshwar & Honnavar (arXiv:2603.20904)
to two-dimensional Itô diffusions, recovering drift, the full diffusion tensor (including
off-diagonal leverage), and circulation from discrete trajectory data.


## Quickstart

```bash
pip install -e .
bash reproduce.sh
```

`reproduce.sh` runs four steps in order and takes roughly 5–10 minutes on a modern laptop:

| Step | Script | Output |
|------|--------|--------|
| Coefficient recovery | `experiments/coefficient_recovery.py` | `results/coefficient_recovery/` |
| Paper package (figures + tables) | `experiments/build_paper.py` | `figures/`, `paper/figures/` |
| Test suite | `pytest` | pass/fail |
| PDF compilation *(optional)* | `latexmk` | `paper/main.pdf` |

If you don't have `latexmk`, upload the `paper/` directory to Overleaf to compile.


## Repository layout

```
src/sde2d/                   library — estimator, systems, kernels, regression, metrics
experiments/
  coefficient_recovery.py    main experiment: pooled R=32 coefficient recovery
  build_paper.py             assembles all figures, tables, and datasheets
  gen_datasheets.py          per-system LaTeX datasheet generator
  datasheet_figures.py       per-system field-plot generator
  baselines/                 head-to-head baseline campaign scripts
  core/                      core estimator experiments
  benchmarks/                convergence and robustness benchmarks
  circulation/               probability-current readout
  fluctuation/               noise-correction readout
  leverage/                  leverage-regime experiments
  archive/                   earlier exploration runs (v4–v17) referenced by tests
data/
  baselines/                 pre-computed head-to-head baseline results (checked in)
  system_index/              pre-computed system index and clean coefficients (checked in)
paper/                       LaTeX source — upload to Overleaf or compile with latexmk
tests/                       pytest suite
external/
  weak_stochastic_sindy_1d/  reference 1D estimator (Eshwar & Honnavar)
```

Generated outputs (not checked in, created by `reproduce.sh`):

```
results/coefficient_recovery/    coefficient recovery table
results/paper/                   method comparison CSV
figures/datasheets/              per-system field plots
figures/paper/                   method comparison heatmap
```


## Requirements

- Python 3.11 (exact version; see `pyproject.toml`)
- Dependencies pinned in `requirements.txt`
- `latexmk` + a TeX distribution for PDF compilation (optional)


## Validated systems

The method is validated across 29 two-dimensional systems:

| Family | Systems |
|--------|---------|
| Linear / coupled OU | correlated OU, coupled OU, rotational OU, spiral sink |
| Non-gradient / rotational | non-gradient circulation |
| Bistable | gradient potential, double-well + transverse, Maier–Stein, Duffing |
| Limit cycles | van der Pol, Stuart–Landau, Brusselator, FitzHugh–Nagumo |
| Multiplicative diffusion | diagonal multiplicative, non-diagonal Cholesky |
| Stochastic volatility | Heston (log-price), Heston (S,V), CIR pair, SABR |
| Limit cases | near-singular tensor, near-boundary Heston, bad coverage, degenerate |

Key metrics across the identifiable class:

| Metric | Value |
|--------|-------|
| Median drift relative L² error | ≈ 0.23 |
| Off-diagonal diffusion cosine | ≈ 0.99 |
| PSD-valid tensors (every grid point) | 100% |
| False positives under support rule | 0 |


## Honest limits

The paper names where recovery fails or is unreliable:

- Low-SNR log-price drift in stochastic-volatility models
- Near-singular or degenerate diffusion tensors
- Trajectories that violate the Feller boundary condition
- Under-covered state spaces

These are documented in `results/paper/` after running `reproduce.sh` and discussed
in the paper's limitations section.
