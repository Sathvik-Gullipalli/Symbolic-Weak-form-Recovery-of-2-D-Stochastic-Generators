from __future__ import annotations

import math
import textwrap
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Colormap, TwoSlopeNorm


TITLE_SIZE = 13
AXIS_SIZE = 11
TICK_SIZE = 9
ANNOTATION_SIZE = 8
PALETTE = "tab10"
HEATMAP = "viridis"
DIVERGING = "coolwarm"


LABELS = {
    "objective_drift_median": r"Objective drift error  $\|\hat b-b\|_{L^2(\mu)} / \|b\|$",
    "objective_drift_rel_l2": r"Objective drift error  $\|\hat b-b\|_{L^2(\mu)} / \|b\|$",
    "drift_rel_l2": r"Drift error  $\|\hat b-b\|_{L^2(\mu)} / \|b\|$",
    "median_objective_drift_rel_l2": r"Median objective drift error",
    "median_diffusion_rel_l2": r"Median tensor error",
    "diffusion_median": r"Diffusion tensor error  $\|\hat a-a\|_F / \|a\|_F$",
    "diffusion_rel_l2": r"Diffusion tensor error  $\|\hat a-a\|_F / \|a\|_F$",
    "median_inscope_score": "In-scope score (higher is better)",
    "raw_median_inscope_score": "Median in-scope score",
    "monotone_paper_score": "Cumulative paper score",
    "rho_tensor_abs_error": r"Leverage error  $|\hat\rho-\rho|$",
    "rho_tensor_median": r"Recovered leverage  $\hat\rho$",
    "rho_true": r"True leverage  $\rho$",
    "current_cosine": "Current-field cosine",
    "irreversibility_scalar": "Irreversibility statistic",
    "error": "Error",
    "T_eff": r"Effective horizon  $T_{\mathrm{eff}}$",
    "R": "Number of trajectories",
    "psd_valid_pct": "PSD-valid fraction",
    "median_psd_valid_pct": "Median PSD-valid fraction",
    "a12_cosine": r"Off-diagonal tensor cosine",
    "median_a12_cosine": r"Median off-diagonal tensor cosine",
    "generator_action_error": "Generator-action error",
    "false_positive_count": "Paper false-positive count",
    "degradation": "Leave-one-out degradation",
    "drift recovery": "Drift",
    "tensor recovery": "Tensor",
    "PSD rate": "PSD",
    "conditioning": "Conditioning",
    "sample efficiency": "Sample efficiency",
}


METHOD_LABELS = {
    "WG_SINDY_FROZEN": "WG-SINDy (ours)",
    "B0_NAIVE_1D_PORT": "1D-naive (B0, Eshwar repo)",
    "B0_PRIME_IN_REPO_REPORT": "B0' in-repo tuned port",
    "KM_LOCAL_MOMENT": "Stochastic-SINDy / Kramers-Moyal",
    "WEAK_SINDY_TEMPORAL_PROXY": "Weak-SINDy (temporal)",
    "GEDMD_DENSE_PROXY": "gEDMD",
}


SYSTEM_LABELS = {
    "cir_pair": "CIR pair",
    "correlated_ou": "Correlated OU",
    "coupled_ou": "Coupled OU",
    "diag_multiplicative": "Diagonal multiplicative",
    "double_well_transverse": "Double well + transverse",
    "gradient_potential": "Gradient potential",
    "heston_logsv": "Log-Heston",
    "heston_sv": "Heston",
    "indep_ou": "Independent OU",
    "nondiag_cholesky": "Non-diagonal Cholesky",
    "nongradient_circulation": "Non-gradient circulation",
    "rotational_ou": "Rotational OU",
    "spiral_sink_corr": "Spiral sink + correlated noise",
    "partial_observation": "Partial observation",
    "bad_coverage": "Bad coverage",
    "too_large_dt": "Too-large time step",
    "nonpoly_drift": "Non-polynomial drift",
    "underdamped_langevin": "Underdamped Langevin",
    "near_boundary_heston": "Near-boundary Heston",
    "near_singular": "Near-singular tensor",
    "van_der_pol": "Van der Pol",
    "fitzhugh_nagumo": "FitzHugh-Nagumo",
    "stuart_landau": "Stuart-Landau",
    "brusselator": "Brusselator",
    "maier_stein": "Maier-Stein",
    "duffing": "Duffing oscillator",
    "mueller_brown": "Mueller-Brown",
    "sabr": "SABR",
    "gbm_2d": "Correlated 2D GBM",
    "two_factor_vasicek": "Two-factor Vasicek",
}


FIELD_LABELS = {
    "b1": r"Drift component $b_1$",
    "b2": r"Drift component $b_2$",
    "a11": r"Tensor entry $a_{11}$",
    "a12": r"Tensor entry $a_{12}$",
    "a22": r"Tensor entry $a_{22}$",
}


def apply_house_style() -> None:
    plt.rcParams.update(
        {
            "font.size": AXIS_SIZE,
            "axes.titlesize": TITLE_SIZE,
            "axes.labelsize": AXIS_SIZE,
            "xtick.labelsize": TICK_SIZE,
            "ytick.labelsize": TICK_SIZE,
            "legend.fontsize": TICK_SIZE,
            "figure.titlesize": TITLE_SIZE,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def label(key: str) -> str:
    if key in LABELS:
        return LABELS[key]
    return key.replace("_", " ").replace("-", " ").strip().title()


def system_label(key: str) -> str:
    return SYSTEM_LABELS.get(key, label(key))


def method_label(key: str) -> str:
    return METHOD_LABELS.get(key, label(key))


def wrap_label(text: str, width: int = 18) -> str:
    return "\n".join(textwrap.wrap(text, width=width, break_long_words=False)) or text


def fmt_value(value: float, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(v):
        return "n/a"
    if abs(v) >= 100:
        return f"{v:.0f}"
    if abs(v) >= 10:
        return f"{v:.1f}"
    if abs(v) >= 1:
        return f"{v:.2f}"
    if abs(v) >= 0.01:
        return f"{v:.3f}"
    if abs(v) == 0:
        return "0"
    return f"{v:.1e}"


def annotate_bars(ax, bars: Iterable, values: Iterable[float], *, dy: float = 3.0, digits: int = 3) -> None:
    for bar, value in zip(bars, values):
        if value is None or not np.isfinite(value):
            continue
        ax.annotate(
            fmt_value(float(value), digits=digits),
            xy=(bar.get_x() + bar.get_width() / 2.0, bar.get_height()),
            xytext=(0, dy),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=ANNOTATION_SIZE,
        )


def annotate_heatmap(ax, raw_values: np.ndarray, *, digits: int = 3, text_colors=("black", "white")) -> None:
    arr = np.asarray(raw_values, float)
    finite = arr[np.isfinite(arr)]
    midpoint = float(np.nanmedian(finite)) if finite.size else 0.0
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            value = arr[i, j]
            color = text_colors[1] if np.isfinite(value) and value > midpoint else text_colors[0]
            ax.text(j, i, fmt_value(value, digits=digits), ha="center", va="center", color=color, fontsize=ANNOTATION_SIZE)


def annotate_heatmap_text(ax, text_values: np.ndarray) -> None:
    for i in range(text_values.shape[0]):
        for j in range(text_values.shape[1]):
            ax.text(j, i, str(text_values[i, j]), ha="center", va="center", fontsize=ANNOTATION_SIZE)


def centered_norm(values: np.ndarray) -> TwoSlopeNorm:
    finite = np.asarray(values, float)
    finite = finite[np.isfinite(finite)]
    scale = float(np.nanmax(np.abs(finite))) if finite.size else 1.0
    scale = max(scale, 1e-12)
    return TwoSlopeNorm(vmin=-scale, vcenter=0.0, vmax=scale)


def robust_limits(
    values: np.ndarray,
    *,
    lower: float = 2.0,
    upper: float = 98.0,
    pad_fraction: float = 0.02,
) -> tuple[float, float]:
    """Percentile limits for plots whose raw extrema are dominated by one row."""
    finite = np.asarray(values, float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return -1.0, 1.0
    lo, hi = np.nanpercentile(finite, [lower, upper])
    lo, hi = float(lo), float(hi)
    if math.isclose(lo, hi):
        delta = max(abs(lo) * 0.05, 1e-3)
        return lo - delta, hi + delta
    pad = pad_fraction * (hi - lo)
    return lo - pad, hi + pad


def robust_centered_norm(values: np.ndarray, *, percentile: float = 98.0) -> TwoSlopeNorm:
    """Diverging norm with symmetric percentile clipping around zero."""
    finite = np.asarray(values, float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        scale = 1.0
    else:
        scale = float(np.nanpercentile(np.abs(finite), percentile))
    scale = max(scale, 1e-12)
    return TwoSlopeNorm(vmin=-scale, vcenter=0.0, vmax=scale)


def softened_cmap(name: str, *, under: str = "#f7fbff", over: str = "#fff7bc") -> Colormap:
    """Return a copy of a Matplotlib colormap with readable over/under colors."""
    cmap = plt.get_cmap(name).copy()
    cmap.set_under(under)
    cmap.set_over(over)
    return cmap


def expanded_limits(values: np.ndarray, pad_fraction: float = 0.05) -> tuple[float, float]:
    finite = np.asarray(values, float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return -1.0, 1.0
    lo, hi = float(np.nanmin(finite)), float(np.nanmax(finite))
    if math.isclose(lo, hi):
        delta = max(abs(lo) * 0.05, 1e-3)
        return lo - delta, hi + delta
    pad = pad_fraction * (hi - lo)
    return lo - pad, hi + pad


def save_figure(fig, out_dir: str | Path, stem: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    png = out / f"{stem}.png"
    pdf = out / f"{stem}.pdf"
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)


def no_raw_key_text(text: str) -> bool:
    bad = (
        "objective_drift_median",
        "monotone_paper_score",
        "median_inscope_score",
        "rho_tensor_abs_error",
        "current_cosine",
        "drift_rel_l2",
        "diffusion_rel_l2",
    )
    return not any(item in text for item in bad)


apply_house_style()
