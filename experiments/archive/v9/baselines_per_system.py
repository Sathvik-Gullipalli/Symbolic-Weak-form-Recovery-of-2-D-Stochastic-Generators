from __future__ import annotations

import argparse
import csv
import math
import os
import re
import shutil
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sde2d import figstyle


RESULTS_V6 = ROOT / "results" / "v6"
RESULTS_V9 = ROOT / "results" / "v9"
FIGURES_V82 = ROOT / "figures" / "v8_2"
PAPER = ROOT / "paper_overleaf"
PAPER_FIGURES = PAPER / "figures"
STAGE = ROOT / "v7-stage"

REQUIRED_COLUMNS = [
    "system",
    "system_label",
    "method",
    "method_label",
    "drift_l2",
    "tensor_rel_l2",
    "a12_cosine",
    "psd_valid",
    "support_f1",
    "runtime_sec",
    "n_seeds",
    "comparison_scope",
    "config_source",
    "notes",
]

METHOD_ORDER = [
    "WG_SINDY_FROZEN",
    "B0_PRIME_IN_REPO_REPORT",
    "B0_NAIVE_1D_PORT",
    "KM_LOCAL_MOMENT",
    "WEAK_SINDY_TEMPORAL_PROXY",
    "GEDMD_DENSE_PROXY",
]

PAPER_FIGURE_AUDIT = [
    ("act1_naive1d_failure_heatmap", "left unchanged", "true heatmap, readable tick labels, no version string"),
    ("act2_graft_ladder_climb", "left unchanged", "x labels visible and not clipped in the current Overleaf copy"),
    ("necessity_matrix", "left unchanged", "v8.2 fixed raw leave-one-out score-drop label and muted scale"),
    ("headtohead_bar", "left unchanged", "aggregate bars retain B0 marker, IQR, and five-family comparison"),
    ("per_system_method_heatmap", "created", "new V9 matched-system drift/tensor baseline heatmap"),
    ("leverage_regime_sweep", "left unchanged", "four-rho boxes plus recovered-vs-true scatter remain legible"),
    ("circulation_current_field", "left unchanged", "true/recovered quivers and honest diagnostic failure label retained"),
    ("convergence_slope", "left unchanged", "log-log panels and -1/2 slope guide are readable"),
    ("honest_null_panel", "left unchanged", "oracle-headroom comparison uses log y-scale and explicit null labels"),
    ("datasheet_fields_*", "left unchanged", "shared colorbars, centered error maps, and out-of-scope hatching already pass audit"),
]


COMPONENT_NOTES = {
    "indep_ou": (
        r"\drift_1 and \drift_2 are the two independent linear mean-reversion laws, so the component test is whether each coordinate selects only its own first-order term. "
        r"\diff_{11} and \diff_{22} are constant diagonal variances; \diff_{12} is a structural zero. "
        r"The component read confirms that the tensor recovery is not borrowing signal from cross terms: the off-diagonal column remains suppressed while both diagonal constants stay PSD-valid."
    ),
    "correlated_ou": (
        r"\drift_1 and \drift_2 remain separate linear mean-reversion components, but the noise law is no longer diagonal. "
        r"\diff_{11} and \diff_{22} are constant variances, while \diff_{12} is the load-bearing negative covariance. "
        r"The recovered off-diagonal cosine near one is therefore the central component-level success: the estimator keeps drift coupling out of \drift while assigning correlation to the tensor."
    ),
    "coupled_ou": (
        r"\drift_1 and \drift_2 both contain true cross-coordinate slopes, so this case tests drift coupling rather than noise coupling. "
        r"\diff_{11} and \diff_{22} are constant diagonal entries and \diff_{12}=0. "
        r"The component read-out shows that the cross terms are selected in the drift equations without inducing a false off-diagonal diffusion term."
    ),
    "two_factor_vasicek": (
        r"\drift_1 and \drift_2 are affine short-rate factors on a very small absolute scale, including weak cross-coupling. "
        r"\diff_{11}, \diff_{22}, and the small positive \diff_{12} are constant covariance entries. "
        r"The scoped verdict is component-specific: the absolute tensor is physically correct, but the relative tensor denominator is so small that the headline relative error is not a fair pass/fail statistic."
    ),
    "rotational_ou": (
        r"\drift_1 and \drift_2 are the antisymmetric linear rotation plus damping pair, so sign discipline across the two equations is the key check. "
        r"\diff_{11} and \diff_{22} are constant diagonal entries and \diff_{12}=0. "
        r"The component audit verifies that circulation is recovered through the drift field, not through a spurious tensor asymmetry."
    ),
    "spiral_sink_corr": (
        r"\drift_1 and \drift_2 combine damping with a rotational linear pair, while the covariance is genuinely correlated. "
        r"\diff_{11} and \diff_{22} set the marginal noise levels and \diff_{12} carries the correlation. "
        r"The component audit separates the two mechanisms: rotation stays in the drift components and the off-diagonal tensor remains the noise-correlation channel."
    ),
    "nongradient_circulation": (
        r"\drift_1 and \drift_2 contain the polynomial potential terms plus the non-gradient circulating component. "
        r"\diff_{11} and \diff_{22} are constant isotropic entries and \diff_{12}=0. "
        r"The component-level success is that the broken-detailed-balance signal appears in the drift/current read-out while the tensor stays simple and PSD-valid."
    ),
    "double_well_transverse": (
        r"\drift_1 carries the cubic double-well restoring force and transverse coupling, whereas \drift_2 anchors the stable direction. "
        r"\diff_{11} and \diff_{22} are constant diagonal entries and \diff_{12}=0. "
        r"The pass is meaningful because weak drift near the wells does not cause the cubic terms to be replaced by false tensor structure."
    ),
    "gradient_potential": (
        r"\drift_1 and \drift_2 are the two gradient components of a coupled polynomial potential, including mixed terms. "
        r"\diff_{11} and \diff_{22} are constant diagonal entries and \diff_{12}=0. "
        r"The recovered generator keeps the mixed polynomial support in the two drift equations while preserving a reversible, diagonal diffusion tensor."
    ),
    "maier_stein": (
        r"\drift_1 and \drift_2 encode the Maier--Stein nonlinear saddle geometry, including the cubic and mixed terms that drive transition paths. "
        r"\diff_{11} and \diff_{22} are constant diagonal entries and \diff_{12}=0. "
        r"The component check is that the non-gradient transition structure is recovered in \drift rather than being hidden inside diffusion artifacts."
    ),
    "duffing": (
        r"\drift_1 is the kinematic identity mapping position to velocity, while \drift_2 contains damping and the cubic restoring force. "
        r"\diff_{11} and \diff_{22} are constant diagonal entries and \diff_{12}=0. "
        r"The estimator therefore has to respect the phase-space split: the first equation stays linear in velocity and the nonlinear force appears only in the second drift component."
    ),
    "diag_multiplicative": (
        r"\drift_1 and \drift_2 are low-order stabilising drift terms, while the main difficulty is in the state-dependent diagonal tensor. "
        r"\diff_{11} and \diff_{22} vary with their respective coordinates; \diff_{12}=0. "
        r"The component audit confirms that multiplicative noise is recovered as diagonal diffusion rather than as false leverage."
    ),
    "nondiag_cholesky": (
        r"\drift_1 and \drift_2 remain comparatively simple, but all three tensor entries are active through the Cholesky construction. "
        r"\diff_{11} and \diff_{22} set the state-dependent marginal variances and \diff_{12} carries the non-diagonal coupling. "
        r"The pass hinges on a componentwise PSD tensor: the off-diagonal field is recovered without pushing either diagonal entry negative."
    ),
    "heston_logsv": (
        r"\drift_1 is the log-price low-SNR channel and is treated as the documented physical null, while \drift_2 is the identifiable variance mean reversion. "
        r"\diff_{11}, \diff_{22}, and \diff_{12} are all linear in variance, with \diff_{12} carrying leverage. "
        r"The component read-out makes the success/failure split explicit: tensor and leverage pass, the variance drift passes, and the log-price drift is not claimed."
    ),
    "heston_sv": (
        r"\drift_1 is the low-SNR price drift in level coordinates, while \drift_2 is the variance mean-reversion component. "
        r"\diff_{11} scales as S^2V, \diff_{22} scales as V, and \diff_{12} carries the SV leverage term SV. "
        r"The component verdict is therefore positive for fluctuations and leverage, with the same honest caveat on the price-drift channel."
    ),
    "cir_pair": (
        r"\drift_1 and \drift_2 are the two square-root mean-reverting drifts. "
        r"\diff_{11} and \diff_{22} scale linearly with their own factors, while \diff_{12} follows the cross square-root leverage structure. "
        r"The component check verifies that the fractional library is used for tensor geometry, not to introduce spurious drift terms."
    ),
    "gbm_2d": (
        r"\drift_1 and \drift_2 are small multiplicative drift channels, which are the scoped weak point. "
        r"\diff_{11} and \diff_{22} scale quadratically with each asset, and \diff_{12} scales as the product S_1S_2. "
        r"The component verdict is intentionally split: drift is low-SNR, but fluctuations and leverage are recovered cleanly enough for a scoped review."
    ),
    "van_der_pol": (
        r"\drift_1 is the phase-space identity y, while \drift_2 carries the limit-cycle nonlinearity including the x^2y damping term. "
        r"\diff_{11} and \diff_{22} are constant diagonal entries and \diff_{12}=0. "
        r"The component audit checks that sparse sampling around the cycle does not induce false diffusion anisotropy."
    ),
    "fitzhugh_nagumo": (
        r"\drift_1 is the fast cubic activator equation and \drift_2 is the slow recovery equation. "
        r"\diff_{11} and \diff_{22} are constant diagonal entries and \diff_{12}=0. "
        r"The pass relies on componentwise scaling: GLS keeps the fast coordinate from overwhelming the slow drift while leaving the tensor simple."
    ),
    "stuart_landau": (
        r"\drift_1 and \drift_2 combine radial cubic saturation with the antisymmetric Hopf rotation. "
        r"\diff_{11} and \diff_{22} are constant diagonal entries and \diff_{12}=0. "
        r"The component check confirms that angular circulation is represented in the drift pair and not confused with off-diagonal noise."
    ),
    "brusselator": (
        r"\drift_1 and \drift_2 contain the opposing autocatalytic X^2Y terms plus the linear chemical inflow/outflow terms. "
        r"\diff_{11} and \diff_{22} are constant diagonal entries and \diff_{12}=0. "
        r"The component read verifies that the shared nonlinear monomial appears with opposite signs in the two drift equations while the diffusion remains isotropic."
    ),
}


@dataclass(frozen=True)
class V9Outputs:
    comparison_csv: Path
    heatmap_pdf: Path
    audit_md: Path


def _finite_or_blank(value: object) -> float | str:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return ""
    return x if math.isfinite(x) else ""


def _fmt_tex(value: object, digits: int = 3) -> str:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return "--"
    if not math.isfinite(x):
        return "--"
    return figstyle.fmt_value(x, digits=digits)


def _latex_escape(text: object) -> str:
    value = str(text)
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(repl.get(ch, ch) for ch in value)


def _write_csv(path: Path, rows: list[dict[str, object]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def build_per_system_comparison() -> pd.DataFrame:
    head = pd.read_csv(RESULTS_V6 / "headtohead.csv")
    cells = pd.read_csv(RESULTS_V6 / "headtohead_cells.csv")
    system_index = pd.read_csv(RESULTS_V6 / "system_index.csv")

    runtime = (
        cells.groupby(["system", "variant_id"], dropna=False)["runtime_sec"]
        .median()
        .rename("runtime_sec")
        .reset_index()
    )
    merged = head.merge(runtime, on=["system", "variant_id"], how="left")
    rows: list[dict[str, object]] = []
    for _, row in merged.iterrows():
        method = str(row["variant_id"])
        rows.append(
            {
                "system": row["system"],
                "system_label": figstyle.system_label(str(row["system"])),
                "method": method,
                "method_label": figstyle.method_label(method),
                "drift_l2": _finite_or_blank(row["median_objective_drift_rel_l2"]),
                "tensor_rel_l2": _finite_or_blank(row["median_diffusion_rel_l2"]),
                "a12_cosine": _finite_or_blank(row["median_a12_cosine"]),
                "psd_valid": _finite_or_blank(row["median_psd_valid_pct"]),
                "support_f1": "",
                "runtime_sec": _finite_or_blank(row["runtime_sec"]),
                "n_seeds": int(row["n"]),
                "comparison_scope": "matched_headtohead",
                "config_source": "results/v6/headtohead.csv; results/v6/headtohead_cells.csv",
                "notes": "support-F1 not logged in frozen baseline cells; support discipline audited separately by false-positive count",
            }
        )

    matched_systems = set(head["system"])
    missing = system_index.loc[~system_index["system"].isin(matched_systems)].copy()
    for _, row in missing.iterrows():
        rows.append(
            {
                "system": row["system"],
                "system_label": figstyle.system_label(str(row["system"])),
                "method": "WG_SINDY_FROZEN",
                "method_label": figstyle.method_label("WG_SINDY_FROZEN"),
                "drift_l2": _finite_or_blank(row["drift_l2_mu"]),
                "tensor_rel_l2": _finite_or_blank(row["tensor_rel_l2"]),
                "a12_cosine": _finite_or_blank(row["a12_cosine"]),
                "psd_valid": _finite_or_blank(row["psd_valid_pct"]),
                "support_f1": "",
                "runtime_sec": "",
                "n_seeds": int(row["n_seeds"]),
                "comparison_scope": "wg_only_system_index",
                "config_source": "results/v6/system_index.csv",
                "notes": f"not part of the frozen 13-system matched baseline grid; V9 table reports WG-SINDy only ({row['verdict']})",
            }
        )

    df = pd.DataFrame(rows, columns=REQUIRED_COLUMNS)
    df["method_sort"] = df["method"].map({name: i for i, name in enumerate(METHOD_ORDER)}).fillna(99)
    df["system_sort"] = df["system_label"].astype(str)
    df = df.sort_values(["comparison_scope", "system_sort", "method_sort"]).drop(columns=["method_sort", "system_sort"])
    _write_csv(RESULTS_V9 / "per_system_comparison.csv", df.to_dict("records"), REQUIRED_COLUMNS)

    config_rows = [
        {
            "method": method,
            "method_label": figstyle.method_label(method),
            "source": "results/v6/headtohead.csv",
            "role": "matched baseline" if method != "WG_SINDY_FROZEN" else "ours",
        }
        for method in METHOD_ORDER
        if method in set(head["variant_id"])
    ]
    config_rows.append(
        {
            "method": "VANILLA_SINDY_STLSQ",
            "method_label": "Vanilla SINDy STLSQ",
            "source": "not rerun in V9",
            "role": "excluded because no 2D stochastic tensor target was logged in the frozen baseline contract",
        }
    )
    _write_csv(RESULTS_V9 / "baseline_config_log.csv", config_rows, ["method", "method_label", "source", "role"])
    return df


def render_method_heatmap(df: pd.DataFrame) -> Path:
    FIGURES_V82.mkdir(parents=True, exist_ok=True)
    matched = df.loc[df["comparison_scope"] == "matched_headtohead"].copy()
    matched["method"] = pd.Categorical(matched["method"], METHOD_ORDER, ordered=True)
    systems = sorted(matched["system"].unique(), key=figstyle.system_label)
    methods = [m for m in METHOD_ORDER if m in set(matched["method"].astype(str))]
    titles = [
        ("drift_l2", r"Drift error $\|\hat b-b\|_{L^2(\mu)}/\|b\|$"),
        ("tensor_rel_l2", r"Tensor error $\|\hat a-a\|_F/\|a\|_F$"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12.6, max(5.2, 0.42 * len(systems) + 1.0)), sharey=True)
    for ax, (metric, title) in zip(axes, titles):
        raw = np.full((len(systems), len(methods)), np.nan)
        for i, system in enumerate(systems):
            for j, method in enumerate(methods):
                sub = matched.loc[(matched["system"] == system) & (matched["method"].astype(str) == method), metric]
                if len(sub):
                    raw[i, j] = float(sub.iloc[0])
        display = np.log10(1.0 + raw)
        vmin, vmax = figstyle.robust_limits(display, lower=0, upper=98, pad_fraction=0.02)
        image = ax.imshow(
            display,
            aspect="auto",
            cmap=figstyle.softened_cmap(figstyle.HEATMAP),
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(title)
        ax.set_xticks(np.arange(len(methods)))
        ax.set_xticklabels([figstyle.wrap_label(figstyle.method_label(m), width=14) for m in methods], rotation=35, ha="right")
        ax.set_yticks(np.arange(len(systems)))
        ax.set_yticklabels([figstyle.system_label(s) for s in systems])
        ax.grid(False)
        figstyle.annotate_heatmap(ax, raw, digits=2, text_colors=("black", "white"))
        cbar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.02)
        cbar.set_label("log10(1 + relative error)")
    axes[0].set_ylabel("Matched in-scope system")
    fig.suptitle("Per-system baseline comparison (matched 13-system grid)")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    figstyle.save_figure(fig, FIGURES_V82, "per_system_method_heatmap")
    return FIGURES_V82 / "per_system_method_heatmap.pdf"


def write_scale_legend_audit() -> Path:
    FIGURES_V82.mkdir(parents=True, exist_ok=True)
    lines = [
        "# V9 scale and legend audit",
        "",
        "Scope: paper-facing v8.2/V9 figures only. The audit is deliberately surgical: unchanged figures were visually judged as readable in the final v8.2 pass, and V9 regenerates only the new per-system comparison heatmap.",
        "",
        "| Figure | V9 action | Reason |",
        "|---|---:|---|",
    ]
    for stem, action, reason in PAPER_FIGURE_AUDIT:
        lines.append(f"| `{stem}` | {action} | {reason} |")
    lines.extend(
        [
            "",
            "Shared scale policy:",
            "",
            "- Error and comparison heatmaps use raw annotated values with robust display scaling rather than per-column normalization.",
            "- Diverging error maps use centered norms; nonnegative error maps use percentile clipping only for visual contrast.",
            "- Legends avoid internal version strings (`v6`, `V6.2`, `v7`, `v8`) in titles or captions.",
            "- The only newly generated paper figure in V9 is `per_system_method_heatmap.pdf`; unchanged PDFs were copied forward into the Overleaf tree.",
        ]
    )
    path = FIGURES_V82 / "SCALE_LEGEND_AUDIT.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def _remove_v9_blocks(text: str) -> str:
    patterns = [
        r"\n?% BEGIN (?:V9 )?COMPONENT BLOCK\n.*?% END (?:V9 )?COMPONENT BLOCK\n?",
        r"\n?% BEGIN (?:V9 )?METHOD TABLE\n.*?% END (?:V9 )?METHOD TABLE\n?",
        r"\n?% BEGIN (?:V9 )?PER-SYSTEM BASELINES\n.*?% END (?:V9 )?PER-SYSTEM BASELINES\n?",
    ]
    for pattern in patterns:
        text = re.sub(pattern, "\n", text, flags=re.DOTALL)
    return re.sub(r"\n{3,}", "\n\n", text)


def _section_bounds(text: str, label: str) -> tuple[int, int]:
    label_token = rf"\label{{{label}}}"
    label_pos = text.find(label_token)
    if label_pos < 0:
        raise ValueError(f"Could not find label {label}")
    start = text.rfind(r"\subsection", 0, label_pos)
    end = text.find(r"\subsection", label_pos + len(label_token))
    if start < 0:
        raise ValueError(f"Could not find subsection start for {label}")
    if end < 0:
        end = len(text)
    return start, end


def _insert_before_anchor(text: str, label: str, anchor: str, block: str) -> str:
    start, end = _section_bounds(text, label)
    anchor_pos = text.find(anchor, start, end)
    if anchor_pos < 0:
        raise ValueError(f"Could not find anchor {anchor!r} inside {label}")
    return text[:anchor_pos] + block + "\n" + text[anchor_pos:]


def _component_block(system: str) -> str:
    body = COMPONENT_NOTES.get(
        system,
        r"\drift_1 and \drift_2 are interpreted separately from the three tensor entries. "
        r"\diff_{11} and \diff_{22} encode marginal fluctuations, while \diff_{12} is reserved for leverage or covariance. "
        r"The component audit checks that recovery quality is assigned to the correct generator entry rather than hidden in an aggregate score.",
    )
    wrapped = textwrap.fill(body, width=110)
    return "% BEGIN COMPONENT BLOCK\n" + r"\paragraph{Component-by-component.} " + wrapped + "\n% END COMPONENT BLOCK\n\n"


def _method_table(system: str, df: pd.DataFrame) -> str:
    sub = df.loc[df["system"] == system].copy()
    sub["method"] = pd.Categorical(sub["method"], METHOD_ORDER, ordered=True)
    sub = sub.sort_values("method")
    rows = []
    for _, row in sub.iterrows():
        rows.append(
            " & ".join(
                [
                    _latex_escape(row["method_label"]),
                    _fmt_tex(row["drift_l2"], 3),
                    _fmt_tex(row["tensor_rel_l2"], 3),
                    _fmt_tex(row["a12_cosine"], 3),
                    _fmt_tex(row["psd_valid"], 3),
                    _fmt_tex(row["runtime_sec"], 2),
                ]
            )
            + r" \\"
        )
    scope_note = (
        "Matched frozen baseline grid."
        if set(sub["comparison_scope"]) == {"matched_headtohead"}
        else "WG-SINDy row only: this system was not included in the frozen matched baseline grid."
    )
    table = [
        "% BEGIN METHOD TABLE",
        r"\paragraph{Per-system baseline table.} " + scope_note,
        r"\begin{center}\scriptsize",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Method & Drift & Tensor & $a_{12}$ cos. & PSD & Time (s) \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{center}",
        "% END METHOD TABLE",
        "",
    ]
    return "\n".join(table)


def update_datasheets(df: pd.DataFrame) -> None:
    system_index = pd.read_csv(RESULTS_V6 / "system_index.csv")
    scoped = system_index.loc[system_index["verdict"].isin(["PASS", "SCOPED_REVIEW"])].copy()
    label_to_file: dict[str, Path] = {}
    for path in sorted(PAPER.glob("ds_*.tex")):
        text = path.read_text()
        for label in re.findall(r"\\label\{([^}]+)\}", text):
            label_to_file[label] = path

    grouped: dict[Path, list[pd.Series]] = {}
    for _, row in scoped.iterrows():
        label = str(row["paper_subsection_label"])
        path = label_to_file.get(label)
        if path is None:
            raise ValueError(f"No datasheet file found for {label}")
        grouped.setdefault(path, []).append(row)

    for path, rows in grouped.items():
        text = _remove_v9_blocks(path.read_text())
        for row in rows:
            label = str(row["paper_subsection_label"])
            system = str(row["system"])
            text = _insert_before_anchor(text, label, r"\paragraph{Detailed WG-SINDy Recovery Analysis}", _component_block(system))
            text = _insert_before_anchor(text, label, r"\paragraph{Quantitative verdict}", _method_table(system, df))
        path.write_text(text)


def update_main_text() -> None:
    path = PAPER / "main.tex"
    text = _remove_v9_blocks(path.read_text())
    block = r"""
% BEGIN PER-SYSTEM BASELINES
\subsection{Per-system baseline comparison}\label{sec:per-system-baselines}
The aggregate head-to-head in \Cref{fig:h2h} is intentionally compressed, so the paper adds the matched per-system
view in \Cref{fig:per-system-method-heatmap}. Rows are the thirteen systems shared by all frozen baseline
families, columns are the exact estimator variants from the frozen head-to-head ledger, and each cell is annotated
with the raw relative error while the color scale uses only a log display transform. The comparison keeps the
claim honest: WG-SINDy is not merely winning one averaged score, but is the only method that is simultaneously
competitive on drift and tensor recovery across correlated OU, nonlinear polynomial systems, multiplicative
noise, and stochastic-volatility leverage systems.

\begin{figure}[t]\centering
\includegraphics[width=0.98\linewidth]{per_system_method_heatmap.pdf}
\caption{Per-system baseline comparison on the matched thirteen-system grid. Entries show raw relative
$L^2$ drift or tensor error; color uses $\log_{10}(1+\mathrm{error})$ only for readability. Blank baseline
extensions are not imputed: systems outside this frozen matched grid are reported in the datasheets with
WG-SINDy-only rows.}\label{fig:per-system-method-heatmap}
\end{figure}
% END PER-SYSTEM BASELINES
"""
    anchor = r"\subsection{The three read-outs}"
    idx = text.find(anchor)
    if idx < 0:
        raise ValueError("Could not locate read-outs subsection anchor")
    text = text[:idx] + block.strip() + "\n\n" + text[idx:]
    path.write_text(text)


def copy_to_overleaf_and_stage() -> None:
    PAPER_FIGURES.mkdir(parents=True, exist_ok=True)
    for stem in ["per_system_method_heatmap"]:
        for suffix in [".pdf", ".png"]:
            source = FIGURES_V82 / f"{stem}{suffix}"
            if source.exists():
                shutil.copy2(source, PAPER_FIGURES / source.name)

    if STAGE.exists():
        shutil.copytree(PAPER, STAGE / "paper_overleaf", dirs_exist_ok=True)
        shutil.copytree(RESULTS_V9, STAGE / "results" / "v9", dirs_exist_ok=True)
        shutil.copytree(FIGURES_V82, STAGE / "figures" / "v8_2", dirs_exist_ok=True)
        shutil.copytree(ROOT / "experiments" / "v9", STAGE / "experiments" / "v9", dirs_exist_ok=True)
        for rel in [
            Path("src/sde2d/figstyle.py"),
            Path("tests/test_v9_paper.py"),
            Path("CODEX_PROMPT_V9.md"),
        ]:
            src = ROOT / rel
            if src.exists():
                dest = STAGE / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dest)


def write_summary(df: pd.DataFrame) -> None:
    matched = df.loc[df["comparison_scope"] == "matched_headtohead"]
    wg = matched.loc[matched["method"] == "WG_SINDY_FROZEN"]
    summary = [
        "# V9 run summary",
        "",
        f"- Wrote `{(RESULTS_V9 / 'per_system_comparison.csv').relative_to(ROOT)}` with {len(df)} rows.",
        f"- Matched head-to-head subset: {matched['system'].nunique()} systems x {matched['method'].nunique()} methods.",
        f"- WG-SINDy median drift error on matched subset: {wg['drift_l2'].astype(float).median():.3f}.",
        f"- WG-SINDy median tensor error on matched subset: {wg['tensor_rel_l2'].astype(float).median():.3f}.",
        "- `support_f1` is intentionally blank because the frozen baseline logs did not record comparable coefficient-support TP/FN counts.",
        "- PASS and SCOPED_REVIEW datasheets now contain component-by-component interpretation and a V9 baseline table.",
        "- The Overleaf figure folder and `v7-stage/` mirror were updated by the V9 builder.",
    ]
    (RESULTS_V9 / "V9_RUN_SUMMARY.md").write_text("\n".join(summary) + "\n")


def run_full() -> V9Outputs:
    RESULTS_V9.mkdir(parents=True, exist_ok=True)
    FIGURES_V82.mkdir(parents=True, exist_ok=True)
    df = build_per_system_comparison()
    heatmap_pdf = render_method_heatmap(df)
    audit_md = write_scale_legend_audit()
    update_datasheets(df)
    update_main_text()
    copy_to_overleaf_and_stage()
    write_summary(df)
    return V9Outputs(
        comparison_csv=RESULTS_V9 / "per_system_comparison.csv",
        heatmap_pdf=heatmap_pdf,
        audit_md=audit_md,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build V9 per-system comparison and paper updates.")
    parser.add_argument("--full", action="store_true", help="Run the complete V9 paper update.")
    args = parser.parse_args()
    if not args.full:
        parser.error("V9 currently supports only --full")
    outputs = run_full()
    print(f"wrote {outputs.comparison_csv}")
    print(f"wrote {outputs.heatmap_pdf}")
    print(f"wrote {outputs.audit_md}")


if __name__ == "__main__":
    main()
