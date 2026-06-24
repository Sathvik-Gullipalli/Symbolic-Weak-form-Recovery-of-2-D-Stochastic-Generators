# V8 WRITING STATUS — continuous handoff (resume-safe)

**Purpose.** Opus is writing the WG-SINDy paper + per-SDE datasheets directly (Codex hit its limit). This file
is the **single source of truth for resuming** if any model (Opus or Gemini) stops mid-way. It carries: the
plan, the COMPLETE per-system data (so no model needs to re-derive), the file checklist, and resume rules.
**Update the checklist after every file you finish.**

## How to resume (read first)
1. Read this whole file. The data tables below are authoritative (from `results/v7/system_index.csv` +
   `src/sde2d/systems.py`). Do NOT re-run code; use these numbers verbatim.
2. Find the first `[ ]` (todo) item in the checklist (§Checklist). Do it. Mark `[x]`. Repeat.
3. Output target = a single Overleaf folder `paper_overleaf/` (see §Folder). Prose = publishable academic
   English. Match the arXiv-style template of `pre_print_weak_sindy_v5.pdf`.
4. Each datasheet must be **≥2 pages** (~900–1100 words + the coefficient table + the figure). Use the
   per-system template in §Datasheet-template, filled from the data tables.

## Folder (deliverable)
```
paper_overleaf/
  main.tex            # full body: frontmatter + Background + Methodology + Results + Discussion + Conclusion + \input{datasheets}
  arxiv.sty           # vendored style  [x]
  references.bib      # all cites        [x]
  datasheets.tex      # all per-SDE datasheets (\input from main.tex)
  figures/            # vector PDFs copied from ../figures/v6/*.pdf  (codex/Gemini: `cp ../figures/v6/*.pdf figures/`)
  README.md           # "open main.tex in Overleaf; latexmk -pdf main.tex"
  V8_WRITING_STATUS.md# this file
```
Figures referenced as `figures/showcase_fields_<system>.pdf` (copy them in; do not use ../ paths in \includegraphics).

## Algorithm being written up: WG-SINDy (frozen, do not change)
Frozen config (= v5.5 in-scope default `V5GREEDY_local_poly_order_2`):
standardized-z coords · k-means centers · anisotropic cov-bandwidth ×1.5 · **local-polynomial order-2** weak
projection · drift pass-1 then **GLS-whitening by pass-1 diffusion tensor â** · **Cholesky-parametrized PSD**
diffusion fit · adaptive-LASSO selection · R=16 trajectories · finite-step + lag-1 EIV correction · symbolic
sparse `L̂` output. Three read-outs: a12→leverage, (a11,a22)→fluctuation/relaxation, antisym-drift→circulation.
Headline: in-scope worst-tier drift ≈0.23, a12 cosine ≈0.99–1.0, zero false positives across all systems.
Theorem: GLS weights are F_tn-measurable functions of pass-1 â ⇒ preserves the 1D spatial-kernel unbiasedness
(Thm 4), only improves efficiency. Provenance: extends Eshwar & Honnavar arXiv 2603.20904v5.

## MASTER DATA TABLE (Table 1 — every 2D system; numbers from system_index.csv, n=10 seeds)
fmt: system | label | tier/family | lib | driftL2 | tensorL2 | a12cos | PSD | FP | verdict
- indep_ou | Independent OU | 1/linear | A | 0.098 | 0.033 | – | 1.0 | 0 | PASS
- correlated_ou | Correlated OU | 1/linear | A | 0.136 | 0.051 | 1.000 | 1.0 | 0 | PASS
- coupled_ou | Coupled OU | 1/linear | A | 0.214 | 0.028 | – | 1.0 | 0 | PASS
- two_factor_vasicek | Two-factor Vasicek | 9/linear | A | 0.350 | 1.000* | – | 1.0 | 0 | SCOPED_REVIEW (*tiny constant tensor ⇒ rel-L2 degenerate; report abs error)
- rotational_ou | Rotational OU | 2/rotational | A | 0.058 | 0.038 | – | 1.0 | 0 | PASS
- spiral_sink_corr | Spiral sink + corr noise | 2/rotational | A | 0.078 | 0.043 | 1.000 | 1.0 | 0 | PASS
- nongradient_circulation | Non-gradient circulation | 3/rotational | B | 0.239 | 0.024 | – | 1.0 | 0 | PASS
- double_well_transverse | Double well + transverse | 3/bistable | B | 0.226 | 0.024 | – | 1.0 | 0 | PASS
- gradient_potential | Gradient potential | 3/bistable | B | 0.246 | 0.024 | – | 1.0 | 0 | PASS
- maier_stein | Maier–Stein | 8/bistable | B | 0.235 | 0.038 | – | 1.0 | 0 | PASS
- duffing | Duffing oscillator | 8/bistable | B | 0.240 | 0.031 | – | 1.0 | 0 | PASS
- mueller_brown | Müller–Brown | 8/bistable | C | 1.826 | 0.075 | – | 1.0 | 0 | NAMED_NULL (stiff multi-well, mobility 0.004 ⇒ drift signal tiny)
- diag_multiplicative | Diagonal multiplicative | 4/multiplicative | A | 0.200 | 0.112 | – | 1.0 | 0 | PASS
- nondiag_cholesky | Non-diagonal Cholesky | 4/multiplicative | C | 0.204 | 0.079 | 0.988 | 1.0 | 0 | PASS
- near_singular | Near-singular tensor | 4/multiplicative | A | 0.870 | 1.505 | 0.724 | 1.0 | 0 | NAMED_NULL (a12→0.95√(a11a22); PSD-boundary stress)
- heston_sv | Heston (S,V) | 5/financial | E | 0.273 | 0.089 | 0.993 | 1.0 | 0 | PASS (b1 log-price drift = reported null)
- heston_logsv | Log-Heston (logS,V) | 5/financial | D | 0.143 | 0.054 | 0.999 | 1.0 | 0 | PASS (b1 log-price drift = reported null)
- cir_pair | CIR pair | 5/financial | F | 0.393 | 0.116 | 0.995 | 1.0 | 0 | PASS
- sabr | SABR | 9/financial | F | nan† | 0.195 | 0.990 | 1.0 | 0 | NAMED_NULL (†true drift ≡0; tensor+leverage strong)
- gbm_2d | Correlated 2D GBM | 9/financial | A | 2.49 | 0.096 | 0.997 | 1.0 | 0 | SCOPED_REVIEW (drift μS low-SNR like log-price; tensor+leverage strong)
- van_der_pol | Van der Pol | 7/limit-cycle | B | 0.066 | 0.040 | – | 1.0 | 0 | PASS
- fitzhugh_nagumo | FitzHugh–Nagumo | 7/limit-cycle | B | 0.205 | 0.040 | – | 1.0 | 0 | PASS
- stuart_landau | Stuart–Landau | 7/limit-cycle | B | 0.126 | 0.018 | – | 1.0 | 0 | PASS
- brusselator | Brusselator | 7/limit-cycle | B | 0.013 | 0.067 | – | 1.0 | 0 | PASS
- underdamped_langevin | Underdamped Langevin | 6/limits | B | 0.157 | 0.563 | – | 1.0 | 0 | NAMED_NULL (rank-1 degenerate a: a11=a12=0)
- near_boundary_heston | Near-boundary Heston | 6/limits | D | 1.413 | 0.267 | 0.982 | 1.0 | 0 | NAMED_NULL (Feller-violating, mass at v→0)
- nonpoly_drift | Non-polynomial drift | 6/limits | G | 0.210 | 0.037 | – | 1.0 | 0 | NAMED_NULL (needs trig lib G; representability)
- bad_coverage | Bad coverage | 6/limits | A | 0.815 | 0.784 | – | 1.0 | 0 | NAMED_NULL (clustered IC ⇒ rank-deficient design)
- too_large_dt | Too-large Δt | 6/limits | A | 0.098 | 0.119 | – | 1.0 | 0 | NAMED_NULL (finite-step bias at Δt=0.05)

Counts: 30 2D systems. PASS = 19. SCOPED_REVIEW = 2 (gbm_2d, two_factor_vasicek). NAMED_NULL = 8
(mueller_brown, near_singular, sabr, underdamped_langevin, near_boundary_heston, nonpoly_drift, bad_coverage,
too_large_dt). a12 cosine reported where off-diagonal nontrivial (financial/correlated/Cholesky); "–" where
diffusion is diagonal (a12≡0). Zero false positives everywhere.

## GROUND-TRUTH (drift b, tensor a) per system — from src/sde2d/systems.py (use for datasheet "analytic generator")
- indep_ou: b=(−θ1 x, −θ2 y), θ1=1,θ2=2; a=diag(σ1²,σ2²)=diag(1,0.49). Gaussian stationary.
- correlated_ou: b=(−x, −1.5y); a=[[1,ρσ1σ2],[·,0.64]], ρ=−0.6,σ1=1,σ2=0.8 ⇒ a12=−0.48. Gaussian, off-diag noise.
- coupled_ou: b=(−a x + c y, d x − b y), a=b=1,c=d=0.5; a=I. cross-state drift.
- two_factor_vasicek: b=κ1(θ1−x)+c12(y−θ2), κ2(θ2−y)+c21(x−θ1); a const tiny [[σ1²,ρσ1σ2],[·,σ2²]] σ≈0.02.
- rotational_ou: b=(−αx−ωy, ωx−αy) α=1,ω=2; a=σ²I=I; true current J=ω(−y,x). Non-reversible.
- spiral_sink_corr: b=(−γx−ωy, ωx−γy) γ=1,ω=1.5; a=[[1,ρσ1σ2],[·,0.64]] ρ=−0.5. Non-rev + off-diag.
- nongradient_circulation: b=−∇V+ωJ∇V, V=¼(x²−1)²+½λy²+ηx²y², λ=1,η=0.25,ω=1; a=σ²I σ=0.7. current=ωJ∇V.
- double_well_transverse: b=(x−x³−βy, −λy+βx) β=0.5,λ=1; a=σ²I σ=0.7. bistable+linear coupling.
- gradient_potential: b=−∇V, V=¼(x²−1)²+½λy²+ηx²y², λ=1,η=0.25; a=σ²I σ=0.7. metastable.
- maier_stein: b=(x−x³−βxy², −(1+x²)y) β=0.35; a=σ²I σ=0.35. canonical non-gradient escape.
- duffing: b=(y, −δy+x−x³) δ=0.35; a=σ²I σ=0.30. noisy bistable oscillator.
- mueller_brown: b=−mobility·∇V_MB (4-Gaussian potential), mobility=0.004; a=σ²I σ=0.35. stiff multi-well.
- diag_multiplicative: b=−θx θ=1; a=diag(a0+ax x²+ay y², b0+bx x²+by y²), a0=0.5,ax=ay=0.1,b0=0.4,bx=by=0.1.
- nondiag_cholesky: b=−x; a=LLᵀ, L=[[0.5+0.1x²,0],[0.2xy,0.4+0.1y²]] ⇒ a11=(0.5+0.1x²)², a12=(0.5+0.1x²)(0.2xy), a22=(0.2xy)²+(0.4+0.1y²)².
- near_singular: b=−x; diag-mult a but a12=0.95√(a11a22). near-singular PSD.
- heston_sv: b=(μS, κ(θ−v)) μ=0.05,κ=2,θ=0.04; a=[[S²v, ρξSv],[·, ξ²v]] ξ=0.3,ρ=−0.65. raw S²V scale.
- heston_logsv: X=logS; b=(μ−½v, κ(θ−v)); a=[[v, ρξv],[·, ξ²v]]. cleaner scale. leverage ρξv.
- cir_pair: b=(κ(θ−x), κ(θ−y)); a=[[ξ²x, ρξ²√(xy)],[·, ξ²y]]. sqrt off-diagonal (needs lib F).
- sabr: b=(0,0); a=[[v²F^{2β}, ρν v² F^β],[·, ν²v²]] β=0.5,ν=0.45,ρ=−0.55. drift≡0 (driftless martingale).
- gbm_2d: b=(μ1 S1, μ2 S2) μ1=0.04,μ2=0.02; a=[[σ1²S1², ρσ1σ2 S1 S2],[·, σ2²S2²]] σ1=0.22,σ2=0.30,ρ=0.45.
- van_der_pol: b=(y, μ(1−x²)y−x) μ=1.2; a=σ²I σ=0.35. limit cycle.
- fitzhugh_nagumo: b=(x−x³/3−y+I, ε(x+a−by)) ε=0.08,a=0.7,b=0.8,I=0.5; a=σ²I σ=0.25. excitable.
- stuart_landau: b=((λ−r²)x−ωy, ωx+(λ−r²)y), r²=x²+y², λ=1,ω=1.5; a=σ²I σ=0.28; current ω(−y,x). Hopf.
- brusselator: b=(A−(B+1)x+x²y, Bx−x²y) A=1,B=2.6; a=σ²I σ=0.12. chemical oscillator.
- underdamped_langevin: b=(p, −γp−(q³−q)) γ=1; a=[[0,0],[0,σ²]] σ=1. RANK-1 degenerate diffusion.
- near_boundary_heston: log-Heston with κ=1,θ=0.02,ξ=0.4 ⇒ 2κθ<ξ² Feller-violated.
- nonpoly_drift: b=(−x+sin y, −y+cos x); a=σ²I σ=0.7. needs trig library G.
- bad_coverage: rotational OU but trajectories confined near (2,2), short horizon ⇒ poor state coverage.
- too_large_dt: diagonal multiplicative sampled at Δt=0.05 ⇒ finite-step bias.

## Checklist (update as you go)
- [x] arxiv.sty vendored
- [x] references.bib
- [x] main.tex — frontmatter (title/authors/abstract/keywords)
- [x] main.tex — 1 Introduction (1.1 Scope, 1.2 Code&Data)
- [x] main.tex — 2 Background
- [x] main.tex — 3 Methodology (Algorithm 1 + theorem)
- [x] main.tex — 4 Results (4.1 showcase+Table1 → 4.2 1D-fails → 4.3 ablation → 4.4 head-to-head → 4.5 read-outs → 4.6 convergence → 4.7 nulls)
- [x] main.tex — 5 Discussion & Limitations, 6 Conclusion, Acks, \bibliography
- [x] figures/ populated (cp ../figures/v6/*.pdf)
- [x] datasheets.tex — linear_ou family (indep_ou, correlated_ou, coupled_ou, two_factor_vasicek)
- [x] datasheets.tex — rotational (rotational_ou, spiral_sink_corr, nongradient_circulation)
- [x] datasheets.tex — bistable (double_well_transverse, gradient_potential, maier_stein, duffing, mueller_brown)
- [x] datasheets.tex — multiplicative (diag_multiplicative, nondiag_cholesky, near_singular)
- [x] datasheets.tex — financial (heston_sv, heston_logsv, cir_pair, sabr, gbm_2d)
- [x] datasheets.tex — limit_cycle (van_der_pol, fitzhugh_nagumo, stuart_landau, brusselator)
- [x] datasheets.tex — honest_limits (underdamped_langevin, near_boundary_heston, nonpoly_drift, bad_coverage, too_large_dt)
- [x] final: latexmk -pdf main.tex compiles clean; mirror into v7-stage/paper_overleaf/

## Datasheet template (fill from data tables; ≥2 pages each)
```
\subsection{<Label>}\label{<paper_subsection_label>}
\paragraph{System.} <SDE eqns in display math> with parameters <...>; <physical/financial/chemical meaning>.
\paragraph{Analytic generator.} True drift b(x,y)=<...>; true tensor a(x,y)=<...>; <stationary density / current /
eigenvalues if applicable>. <what this system tests: coupling / nonlinearity / off-diagonal / state-dependence /
circulation / limit-cycle / SNR / boundary>.
\paragraph{Recovered generator.} <coefficient table: true vs recovered per active term from showcase_coefficients;
abs-error for true-zero terms>. Recovered \hat L written out beside true L.
<figure: \includegraphics figures/showcase_fields_<system>.pdf, caption>
\paragraph{Quantitative verdict.} drift L2µ=<>, tensor rel-L2=<>, a12 cosine=<>, PSD=<>, FP=<>, n=10 seeds; <PASS/null>.
\paragraph{Read-out.} <which of leverage/fluctuation/circulation this feeds + result>.
\paragraph{Discussion (how well, why).} 2–4 paragraphs tying success/limit to GLS / local-poly / Cholesky / SNR /
representability; for nulls, the honest physical reason + oracle headroom.
```
