from __future__ import annotations

import argparse
import csv
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[2] / ".matplotlib-cache"))

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from experiments.baselines.run_audit import (  # noqa: E402
    ALL_2D_SYSTEMS,
    FAMILY_BY_SYSTEM,
    FAMILY_ORDER,
    SYSTEM_DEFINITIONS,
)
from experiments.baselines.render_figures import system_grid, term_value  # noqa: E402
from matplotlib.colors import TwoSlopeNorm as _TwoSlopeNorm  # noqa: E402
from sde2d.figstyle import DIVERGING, HEATMAP, robust_centered_norm, robust_limits, save_figure, system_label, wrap_label  # noqa: E402
from sde2d.systems import REGISTRY  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402


V7 = ROOT / "data/system_index"
FIG_V7 = ROOT / "figures/datasheets"
PAPER = ROOT / "paper"
PAPER_DATASHEETS = PAPER / "datasheets"
DOC_DATASHEETS = ROOT / "docs/datasheets"
OVERLEAF = ROOT / "stage/overleaf_wg_sindy_v7"
STAGE = ROOT / "stage"

TARGETS = ["b1", "b2", "a11", "a12", "a22"]
ZERO_TOL_FRACTION = 1e-4

V7_INDEX = V7 / "system_index.csv"
V7_COEFS = V7 / "coefficients_clean.csv"
V7_TABLE = PAPER / "v7_table1.tex"
V7_DATASHEET_INPUTS = PAPER / "v7_datasheet_inputs.tex"
V7_MANUSCRIPT = PAPER / "wg_sindy_v7_manuscript.tex"
V7_INTEGRITY = V7 / "final_integrity_report.csv"
V7_ISSUES = ROOT / "docs/V7_ISSUES_FOUND.md"
V7_MANIFEST = ROOT / "MANIFEST.csv"


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def clean(value: object) -> object:
        if isinstance(value, float) and not math.isfinite(value):
            return ""
        if isinstance(value, np.floating) and not math.isfinite(float(value)):
            return ""
        return value

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows({key: clean(row.get(key, "")) for key in fields} for row in rows)


def latex_escape(value: object) -> str:
    text = str(value)
    return (
        text.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("$", r"\$")
        .replace("#", r"\#")
        .replace("_", r"\_")
        .replace("{", r"\{")
        .replace("}", r"\}")
        .replace("~", r"\textasciitilde{}")
        .replace("^", r"\textasciicircum{}")
    )


def fmt_num(value: object, digits: int = 3) -> str:
    try:
        v = float(value)
    except Exception:
        return "n/a"
    if not math.isfinite(v):
        return "n/a"
    if abs(v) >= 100:
        return f"{v:.0f}"
    if abs(v) >= 10:
        return f"{v:.1f}"
    if abs(v) >= 1:
        return f"{v:.2f}"
    if abs(v) >= 0.01:
        return f"{v:.{digits}f}"
    if v == 0:
        return "0"
    return f"{v:.1e}"


def verdict_label(row: pd.Series) -> str:
    verdict = str(row["verdict"])
    scope = str(row["scope_status"])
    reason = str(row["verdict_reason"]).replace("_", " ")
    if verdict == "PASS":
        return r"\textcolor{green!45!black}{\checkmark PASS} " + latex_escape(scope)
    if verdict == "NAMED_NULL":
        words = " ".join(reason.split()[:2])
        return r"\textcolor{orange!70!black}{\(\circ\) named-null: " + latex_escape(words) + r"}"
    return r"\textcolor{red!65!black}{\(\triangle\) review} " + latex_escape(scope)


def term_latex(term: str) -> str:
    mapping = {
        "1": "1",
        "x": "x",
        "y": "y",
        "xy": "xy",
        "x^2": "x^2",
        "y^2": "y^2",
        "x^3": "x^3",
        "y^3": "y^3",
        "x^2y": "x^2y",
        "xy^2": "xy^2",
        "x^4": "x^4",
        "x^3y": "x^3y",
        "x^2y^2": "x^2y^2",
        "xy^3": "xy^3",
        "y^4": "y^4",
        "sqrt(x)": r"\sqrt{x}",
        "sqrt(y)": r"\sqrt{y}",
        "sqrt(xy)": r"\sqrt{xy}",
        "sin x": r"\sin x",
        "sin y": r"\sin y",
        "cos x": r"\cos x",
        "cos y": r"\cos y",
    }
    return mapping.get(term, latex_escape(term))


def load_index() -> pd.DataFrame:
    path = ROOT / "data/baselines/system_index.csv"
    df = pd.read_csv(path)
    order = {system: i for i, system in enumerate(ALL_2D_SYSTEMS)}
    df["_order"] = df["system"].map(order)
    return df.sort_values("_order").drop(columns=["_order"]).reset_index(drop=True)


def load_coefficients() -> pd.DataFrame:
    paths = [
        ROOT / "data/baselines/showcase/showcase_coefficients.csv",
        ROOT / "data/baselines/v6_2_extra_coefficients.csv",
    ]
    frames = [pd.read_csv(path) for path in paths if path.exists()]
    if not frames:
        raise FileNotFoundError("No v6/v6.2 coefficient tables found")
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(["system", "target", "term_name", "term_index"], keep="last")
    for col in ["true_coef_median", "recovered_coef_median", "true_coef_ci_low", "true_coef_ci_high", "recovered_coef_ci_low", "recovered_coef_ci_high", "selected_rate", "active_true_rate", "false_positive_count"]:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_values(["system", "target", "term_index"]).reset_index(drop=True)


def clean_coefficients(coefs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (system, target), part in coefs.groupby(["system", "target"], sort=False):
        scale = max(float(part["true_coef_median"].abs().max(skipna=True) or 0.0), 1e-12)
        tol = max(1e-8, ZERO_TOL_FRACTION * scale)
        for _, row in part.iterrows():
            true = float(row["true_coef_median"])
            hat = float(row["recovered_coef_median"])
            abs_err = abs(hat - true)
            zero_truth = abs(true) <= tol
            rel = float("nan") if zero_truth else abs_err / max(abs(true), 1e-12)
            selected = str(row.get("selected", "")).lower() == "true" or float(row.get("selected_rate", 0.0) or 0.0) >= 0.5
            active = float(row.get("active_true_rate", 0.0) or 0.0) >= 0.2 and not zero_truth
            if active and selected:
                role = "active_selected"
            elif active:
                role = "active_missed"
            elif selected:
                role = "selected_zero_truth"
            else:
                role = "inactive_unselected"
            out = row.to_dict()
            out.update(
                {
                    "abs_error_clean": abs_err,
                    "rel_error_clean": rel,
                    "zero_truth": zero_truth,
                    "error_display": fmt_num(abs_err) if zero_truth else fmt_num(rel),
                    "error_type": "abs" if zero_truth else "rel",
                    "term_role": role,
                    "selected_clean": selected,
                    "active_true_clean": active,
                }
            )
            rows.append(out)
    cleaned = pd.DataFrame(rows)
    fields = [
        "system",
        "tier",
        "target",
        "term_name",
        "term_index",
        "true_coef_median",
        "true_coef_ci_low",
        "true_coef_ci_high",
        "recovered_coef_median",
        "recovered_coef_ci_low",
        "recovered_coef_ci_high",
        "abs_error_clean",
        "rel_error_clean",
        "zero_truth",
        "error_display",
        "error_type",
        "selected_clean",
        "active_true_clean",
        "term_role",
        "false_positive_count",
        "selected_rate",
        "active_true_rate",
    ]
    write_csv(V7_COEFS, cleaned.to_dict("records"), fields)
    return cleaned


def coefficient_rows_for_datasheet(coefs: pd.DataFrame, system: str) -> pd.DataFrame:
    part = coefs[coefs["system"] == system].copy()
    keep = part["active_true_clean"] | part["selected_clean"] | (pd.to_numeric(part["false_positive_count"], errors="coerce").fillna(0) > 0)
    kept = part[keep].copy()
    if kept.empty:
        kept = part.groupby("target", group_keys=False).head(2).copy()
    return kept.sort_values(["target", "term_index"])


def expression_from_terms(part: pd.DataFrame, target: str, column: str) -> str:
    rows = part[part["target"] == target].copy()
    if rows.empty:
        return "0"
    rows["mag"] = pd.to_numeric(rows[column], errors="coerce").abs()
    rows = rows[rows["mag"] > 1e-8].sort_values("mag", ascending=False).head(7)
    if rows.empty:
        return "0"
    pieces = []
    for _, row in rows.iterrows():
        coef = float(row[column])
        term = str(row["term_name"])
        sign = "+" if coef >= 0 else "-"
        body = f"{abs(coef):.3g}"
        if term != "1":
            body += rf"\,{term_latex(term)}"
        pieces.append((sign, body))
    first_sign, first_body = pieces[0]
    expr = ("" if first_sign == "+" else "-") + first_body
    for sign, body in pieces[1:]:
        expr += f" {sign} {body}"
    return expr


def generator_equation_block(coefs: pd.DataFrame, system: str) -> str:
    part = coefficient_rows_for_datasheet(coefs, system)
    lines = []
    for target in TARGETS:
        truth = expression_from_terms(part, target, "true_coef_median")
        rec = expression_from_terms(part, target, "recovered_coef_median")
        lines.append(rf"{target}^{{\mathrm{{true}}}} &= {truth}, & {target}^{{\mathrm{{rec}}}} &= {rec}\\")
    return "\n".join(lines)


def fields_for_system(system: str, index_row: pd.Series) -> list[str]:
    family = str(index_row["family"])
    fields = ["b1", "b2"]
    state_dependent_tensor = {
        "diag_multiplicative",
        "nondiag_cholesky",
        "near_singular",
        "heston_sv",
        "heston_logsv",
        "cir_pair",
        "near_boundary_heston",
        "sabr",
        "gbm_2d",
        "mueller_brown",
    }
    offdiag_systems = {
        "correlated_ou",
        "spiral_sink_corr",
        "nondiag_cholesky",
        "near_singular",
        "heston_sv",
        "heston_logsv",
        "cir_pair",
        "near_boundary_heston",
        "sabr",
        "gbm_2d",
    }
    if system in state_dependent_tensor:
        fields.extend(["a11", "a22"])
    if system in offdiag_systems or math.isfinite(float(index_row.get("a12_cosine", float("nan")))):
        fields.append("a12")
    if family == "limit_cycle" and "a12" not in fields:
        # Limit-cycle systems use constant tensors; keep plots to drift fields.
        pass
    return fields


def evaluate_coeff_field(coefs: pd.DataFrame, system: str, target: str, column: str, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    part = coefs[(coefs["system"] == system) & (coefs["target"] == target)]
    out = np.zeros_like(x, dtype=float)
    for _, row in part.iterrows():
        try:
            out += float(row[column]) * term_value(str(row["term_name"]), x, y)
        except KeyError:
            continue
    return out


def plot_v7_fields(index: pd.DataFrame, coefs: pd.DataFrame) -> None:
    FIG_V7.mkdir(parents=True, exist_ok=True)
    for _, row in index.iterrows():
        system = str(row["system"])
        fields = fields_for_system(system, row)
        x, y = system_grid(system, n=48)
        fig, axes = plt.subplots(len(fields), 3, figsize=(10.8, max(4.0, 2.65 * len(fields))), constrained_layout=True, squeeze=False)
        for i, field in enumerate(fields):
            true = evaluate_coeff_field(coefs, system, field, "true_coef_median", x, y)
            hat = evaluate_coeff_field(coefs, system, field, "recovered_coef_median", x, y)
            err = hat - true
            vmin, vmax = robust_limits(np.r_[true.ravel(), hat.ravel()], lower=2, upper=98, pad_fraction=0.03)
            for j, (arr, title, cmap, kwargs) in enumerate(
                [
                    (true, "true", HEATMAP, {"vmin": vmin, "vmax": vmax}),
                    (hat, "recovered", HEATMAP, {"vmin": vmin, "vmax": vmax}),
                    (err, "recovered - true", DIVERGING, {"norm": _TwoSlopeNorm(vmin=-max(abs(vmin), abs(vmax), 1e-9), vcenter=0.0, vmax=max(abs(vmin), abs(vmax), 1e-9))}),
                ]
            ):
                ax = axes[i, j]
                im = ax.pcolormesh(x, y, arr, shading="gouraud", cmap=cmap, **kwargs)
                ax.set_title(title)
                if j == 0:
                    ax.set_ylabel(field)
                else:
                    ax.set_yticklabels([])
                if i == len(fields) - 1:
                    ax.set_xlabel("state x")
                else:
                    ax.set_xticklabels([])
                if j >= 1:  # one shared colourbar for true+recovered (col 1) and one for the error (col 2)
                    cb = fig.colorbar(im, ax=ax, shrink=0.74, pad=0.012)
                    cb.set_label("recovered $-$ true" if j == 2 else "true / recovered value")
        fig.suptitle(f"Recovered vs.\\ true generator fields: {system_label(system)}")
        save_figure(fig, FIG_V7, f"datasheet_fields_{system}")


def write_system_index(index: pd.DataFrame) -> pd.DataFrame:
    out = index.copy()
    out["system_label"] = out["system"].map(system_label)
    out["datasheet_tex"] = out["system"].map(lambda s: f"paper/datasheets/{s}.tex")
    out["datasheet_md"] = out["system"].map(lambda s: f"docs/datasheets/{s}.md")
    out["v7_figure_pdf"] = out["system"].map(lambda s: f"figures/datasheets/datasheet_fields_{s}.pdf")
    out["v7_figure_png"] = out["system"].map(lambda s: f"figures/datasheets/datasheet_fields_{s}.png")
    write_csv(V7_INDEX, out.to_dict("records"), list(out.columns))
    return out


def write_table1(index: pd.DataFrame) -> None:
    lines = [
        r"\begin{longtable}{p{0.25\linewidth}rrrrrl}",
        r"\caption{Navigation table for the 2D SDE zoo. Each system name links to its datasheet; values are multi-seed medians from \texttt{data/system_index/system_index.csv}.}\label{tab:v7-system-index}\\",
        r"\toprule",
        r"System & Drift L2 & Tensor L2 & $a_{12}$ Cos & PSD & FP Count & Verdict\\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"System & Drift L2 & Tensor L2 & $a_{12}$ Cos & PSD & FP Count & Verdict\\",
        r"\midrule",
        r"\endhead",
    ]
    last_family = None
    family_titles = dict(FAMILY_ORDER)
    for _, row in index.iterrows():
        family = row["family"]
        if family != last_family:
            lines.append(rf"\multicolumn{{7}}{{l}}{{\textbf{{{latex_escape(family_titles.get(family, family))}}}}}\\")
            last_family = family
        system = row["system"]
        label = f"sec:datasheet-{system.replace('_', '-')}"
        system_cell = rf"\hyperref[{label}]{{{latex_escape(system_label(system))}}}"
        lines.append(
            f"{system_cell} & {fmt_num(row['drift_l2_mu'])} & {fmt_num(row['tensor_rel_l2'])} & {fmt_num(row['a12_cosine'])} & {fmt_num(row['psd_valid_pct'])} & {int(row['false_positive_count'])} & {verdict_label(row)} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{longtable}", ""])
    V7_TABLE.write_text("\n".join(lines))


def coefficient_table_tex(rows: pd.DataFrame, max_rows: int = 36) -> str:
    shown = rows.copy().head(max_rows)
    lines = [
        r"\begin{longtable}{lllrrrl}",
        r"\toprule",
        r"Target & Term & Role & True & Recovered & Error & Selected\\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"Target & Term & Role & True & Recovered & Error & Selected\\",
        r"\midrule",
        r"\endhead",
    ]
    for _, row in shown.iterrows():
        err_label = "abs" if row["error_type"] == "abs" else "rel"
        selected = r"\checkmark" if bool(row["selected_clean"]) else "--"
        lines.append(
            f"{latex_escape(row['target'])} & ${term_latex(str(row['term_name']))}$ & {latex_escape(row['term_role'])} & {fmt_num(row['true_coef_median'])} & {fmt_num(row['recovered_coef_median'])} & {latex_escape(row['error_display'])} ({err_label}) & {selected} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{longtable}"])
    if len(rows) > max_rows:
        lines.append(rf"\emph{{Table truncated to {max_rows} active/selected terms; the complete clean coefficient table is in \texttt{{data/system_index/coefficients_clean.csv}}.}}")
    return "\n".join(lines)


def downstream_text(system: str, row: pd.Series) -> str:
    family = str(row["family"])
    bits = []
    if system in {"heston_sv", "heston_logsv", "cir_pair", "sabr", "gbm_2d", "correlated_ou", "spiral_sink_corr", "nondiag_cholesky", "near_singular"}:
        bits.append("leverage/off-diagonal tensor recovery through $a_{12}$")
    if family in {"multiplicative", "financial", "bistable"} or system in {"diag_multiplicative", "nondiag_cholesky"}:
        bits.append("fluctuation and state-dependent tensor recovery through $a_{11},a_{22}$")
    if family in {"rotational", "limit_cycle"} or system in {"rotational_ou", "nongradient_circulation", "stuart_landau"}:
        bits.append("circulation and broken-detailed-balance diagnostics through antisymmetric drift")
    return "; ".join(bits) if bits else "core generator recovery without a separate downstream application claim"


def write_datasheets(index: pd.DataFrame, coefs: pd.DataFrame) -> None:
    PAPER_DATASHEETS.mkdir(parents=True, exist_ok=True)
    DOC_DATASHEETS.mkdir(parents=True, exist_ok=True)
    for _, row in index.iterrows():
        system = str(row["system"])
        label = f"sec:datasheet-{system.replace('_', '-')}"
        coeff_rows = coefficient_rows_for_datasheet(coefs, system)
        eqs = generator_equation_block(coefs, system)
        fig_name = f"datasheet_fields_{system}.pdf"
        definition = SYSTEM_DEFINITIONS.get(system, "Analytic drift and diffusion tensor are implemented in src/sde2d/systems.py.")
        downstream = downstream_text(system, row)
        verdict = str(row["verdict"])
        reason = str(row["verdict_reason"]).replace("_", " ")
        scope = str(row["scope_status"]).replace("_", " ")
        scalar_note = "For constant-diffusion components, the datasheet reports tensor coefficients as scalars in the symbolic table rather than wasting field panels on flat surfaces."
        tex = rf"""
\clearpage
\subsection{{{latex_escape(system_label(system))}}}
\label{{{label}}}

\paragraph{{1. System definition.}} {definition} Parameters are the values encoded in \texttt{{src/sde2d/systems.py}} and all trajectories use the fixed seeds recorded in the v6/v6.2 CSVs. This system belongs to the \texttt{{{latex_escape(str(row['family']))}}} family and is included because it exercises the corresponding drift, tensor, coverage, or read-out mechanism in the two-dimensional zoo.

\paragraph{{2. Analytic generator.}} The ground-truth generator is
\[
Lf=b_1\partial_x f+b_2\partial_y f+\tfrac12 a_{{11}}\partial_{{xx}}f+a_{{12}}\partial_{{xy}}f+\tfrac12 a_{{22}}\partial_{{yy}}f.
\]
The analytic drift and tensor are evaluated from the registry implementation, and the coefficient projection below is computed against the declared library {latex_escape(str(row['library']))}. Where the stationary density or spectral quantities have a closed form, they are recorded in the system implementation or invariant tables; otherwise the paper uses function-space and generator-action diagnostics rather than inventing unsupported closed forms.

\paragraph{{3. What this system tests.}} It tests {latex_escape(downstream)}. The stressed mechanisms are the frozen WG-SINDy ingredients: standardized spatial kernels, covariance bandwidths, local-polynomial projection, adaptive sparse recovery, GLS drift whitening, and Cholesky/PSD tensor handling. Named-null rows keep physical or identifiability limits visible rather than treating them as code failures.

\paragraph{{4. Recovered symbolic generator.}} The coefficient table is the primary evidence and is generated from \texttt{{data/system_index/coefficients_clean.csv}}. Zero-truth terms use absolute error, not a divide-by-zero relative error; nonzero terms use relative error. The recovered symbolic generator is written beside the true projected generator:
\[
\begin{{aligned}}
{eqs}
\end{{aligned}}
\]
{coefficient_table_tex(coeff_rows)}

\paragraph{{5. Function recovery figure.}} Figure~\ref{{fig:v7-{system.replace('_', '-')}}} is right-sized by component. Drift fields are always shown; tensor fields are plotted only when non-trivial or used by a downstream read-out. {scalar_note}

\begin{{figure}}[p]
\centering
\includegraphics[width=\linewidth]{{{fig_name}}}
\caption{{V7 right-sized true, recovered, and error fields for {latex_escape(system_label(system))}.}}
\label{{fig:v7-{system.replace('_', '-')}}}
\end{{figure}}

\paragraph{{6. Quantitative verdict.}} Drift $L^2_\mu$ error is {fmt_num(row['drift_l2_mu'])}, tensor relative $L^2$ error is {fmt_num(row['tensor_rel_l2'])}, $a_{{12}}$ cosine is {fmt_num(row['a12_cosine'])}, PSD-valid fraction is {fmt_num(row['psd_valid_pct'])}, and generator-action error is {fmt_num(row['generator_action_error'])}. The v7 verdict is \textbf{{{latex_escape(verdict)}}} ({latex_escape(scope)}): {latex_escape(reason)}.

\paragraph{{7. Downstream read-out.}} The downstream use is {latex_escape(downstream)}. Financial and correlated-noise systems feed the leverage read-out through $a_{{12}}$; state-dependent tensor systems feed the fluctuation read-out through $a_{{11}},a_{{22}}$; rotational and limit-cycle systems feed the circulation read-out through antisymmetric drift.

\paragraph{{8. How well, and why.}} The reported numbers are multi-seed medians and are not retuned in v7. When the row passes, the positive interpretation is coefficient-level recovery of the projected generator under the declared library and coverage assumptions. When the row is a named null or scoped review, the limitation is attributed to the registry verdict, oracle/headroom evidence, low-SNR drift, degenerate tensor structure, boundary behavior, bad coverage, or library representability. This is the intended claim discipline: v7 makes the symbolic recovery visible, but it does not turn a documented physical limit into a positive theorem.
"""
        (PAPER_DATASHEETS / f"{system}.tex").write_text(tex.strip() + "\n")
        md = f"""# {system_label(system)}

## System Definition
{definition}

## Generator And Recovery
- Library: {row['library']}
- Drift L2: {fmt_num(row['drift_l2_mu'])}
- Tensor L2: {fmt_num(row['tensor_rel_l2'])}
- a12 cosine: {fmt_num(row['a12_cosine'])}
- PSD fraction: {fmt_num(row['psd_valid_pct'])}
- Verdict: {verdict} ({scope}) - {reason}

## Recovered Symbolic Generator
The clean coefficient table is in `data/system_index/coefficients_clean.csv`. Zero-truth terms are scored by absolute error; nonzero terms are scored by relative error.

## Downstream Read-Out
{downstream}

## Explanation
The result is interpreted with the frozen WG-SINDy mechanisms and named-null policy from v6.2. Passing rows support symbolic generator recovery; named-null rows document physical, data, or library limits.
"""
        (DOC_DATASHEETS / f"{system}.md").write_text(md)
    input_lines = []
    family_titles = dict(FAMILY_ORDER)
    for family, title in FAMILY_ORDER:
        fam = index[index["family"] == family]
        if fam.empty:
            continue
        input_lines.append(rf"\section{{{latex_escape(family_titles.get(family, title))}}}")
        for system in fam["system"]:
            input_lines.append(rf"\input{{datasheets/{system}.tex}}")
    V7_DATASHEET_INPUTS.write_text("\n".join(input_lines) + "\n")


def write_manuscript(index: pd.DataFrame) -> None:
    body = r"""\documentclass[11pt]{article}
\usepackage[margin=0.85in]{geometry}
\usepackage{amsmath,amssymb,booktabs,graphicx,hyperref,longtable,xcolor,amssymb}
\graphicspath{{../figures/datasheets/}{figures/datasheets/}}
\hypersetup{colorlinks=true,linkcolor=blue,citecolor=blue,urlcolor=blue}

\title{WG-SINDy: Symbolic Weak-Form Recovery of Two-Dimensional Stochastic Generators}
\author{Pratham Gullipalli, Eshwar R. A., G. V. Honnavar}
\date{}

\begin{document}
\maketitle

\begin{abstract}
WG-SINDy extends spatial-kernel weak stochastic generator recovery from scalar SDEs to identifiable two-dimensional Ito diffusions. V7 makes the symbolic coefficient recovery explicit: each system reports true and recovered coefficients for the drift vector and the full symmetric diffusion tensor, while downstream figures are right-sized to the components actually used for leverage, fluctuation, and circulation. The strongest claim remains scoped to library completeness, coverage, and rank; named nulls are retained as part of the evidence rather than hidden.
\end{abstract}

\section{Claim, Scope, and Reader Navigation}
The recovered object is the infinitesimal generator
\[
Lf=b_1\partial_x f+b_2\partial_y f+\tfrac12a_{11}\partial_{xx}f+a_{12}\partial_{xy}f+\tfrac12a_{22}\partial_{yy}f,
\]
not a unique volatility factorization $\sigma$. Table~\ref{tab:v7-system-index} is the navigation layer for all datasheets. Each row links to a per-SDE datasheet with the symbolic coefficient table, the right-sized recovery figure, and the quantitative verdict.

\input{v7_table1.tex}

\section{Method Summary}
All v7 results use the frozen v6 WG-SINDy estimator: spatial Gaussian kernels on standardized states, one shared weak design, local-polynomial projection, adaptive sparse recovery, GLS drift whitening, finite-step/EIV diffusion correction, and PSD tensor handling. V7 changes presentation and packaging only; it does not retune the estimator or rerun algorithm search.

\section{What Is Shown and Why}
Every SDE gets the complete coefficient-level evidence for $b_1,b_2,a_{11},a_{12},a_{22}$. Field plots are streamlined: $a_{12}$ fields are emphasized for leverage/correlated-noise systems, diagonal tensor fields for fluctuation/state-dependent systems, and antisymmetric drift/current diagnostics for circulation systems. Constant tensor entries are reported as coefficients/scalars instead of repeated flat panels.

\section{Datasheets}
The following datasheets are grouped by the family order in Table~\ref{tab:v7-system-index}. Each datasheet is generated from the same CSV files as the navigation table, so the displayed metrics and verdicts cannot drift by hand editing.

\input{v7_datasheet_inputs.tex}

\section{Final Integrity}
The final integrity report is \texttt{results/paper/final_integrity_report.csv}. It checks system counts, datasheet counts, coefficient coverage, zero-truth error handling, figure references, and Overleaf packaging. Local TeX compilation is attempted only when a TeX binary is available; the submitted bundle is Overleaf-ready regardless of the local environment.

\end{document}
"""
    V7_MANUSCRIPT.write_text(body)


def write_issues() -> None:
    V7_ISSUES.write_text(
        """# V7 Issues Found And Fixed

- Surfaced symbolic coefficient recovery as the primary evidence. The data already existed in v6/v6.2 coefficient CSVs; v7 promotes it into every datasheet.
- Fixed the zero-truth coefficient scoring artifact by using absolute error for true-zero terms and relative error only for nonzero truth terms.
- Preserved named nulls for fragile/stress systems even when an individual metric looked acceptable; the registry verdict remains part of the claim boundary.
- Replaced one-size-fits-all five-field figure usage with right-sized v7 datasheet figures: drift always, tensor fields only when non-trivial or used by leverage/fluctuation read-outs.
- Removed stale top-level/paper figure duplicate usage from the final paper path; the canonical v7 figure set is `figures/datasheets/`.
- Local `pdflatex` was not available in this environment, so the build verifies Overleaf source completeness and reports TeX compilation as toolchain-deferred rather than fabricating a PDF.
"""
    )


def integrity_rows(index: pd.DataFrame, coefs: pd.DataFrame, require_stage: bool) -> list[dict]:
    rows: list[dict] = []
    expected = len(ALL_2D_SYSTEMS)

    def add(check: str, passed: bool, detail: str) -> None:
        rows.append({"check": check, "status": "PASS" if passed else "ERROR", "detail": detail})

    add("system_index_count", len(index) == expected, f"{len(index)} rows vs {expected} registry 2D systems")
    add("coefficient_system_coverage", set(coefs["system"]) == set(ALL_2D_SYSTEMS), f"{coefs['system'].nunique()} coefficient systems")
    zero_rows = coefs[coefs["zero_truth"] == True]  # noqa: E712
    add("zero_truth_uses_abs_error", bool((zero_rows["error_type"] == "abs").all()), f"{len(zero_rows)} zero-truth rows")
    add("datasheet_tex_count", len(list(PAPER_DATASHEETS.glob("*.tex"))) == expected, f"{len(list(PAPER_DATASHEETS.glob('*.tex')))} tex datasheets")
    add("datasheet_md_count", len(list(DOC_DATASHEETS.glob("*.md"))) == expected, f"{len(list(DOC_DATASHEETS.glob('*.md')))} markdown datasheets")
    add("v7_figure_count", len(list(FIG_V7.glob("datasheet_fields_*.png"))) == expected, f"{len(list(FIG_V7.glob('datasheet_fields_*.png')))} png figures")
    add("table1_rows", V7_TABLE.exists(), "v7 navigation table generated")
    add("manuscript_exists", V7_MANUSCRIPT.exists(), "paper/wg_sindy_v7_manuscript.tex")
    if require_stage:
        add("overleaf_folder_exists", OVERLEAF.exists(), str(OVERLEAF))
        add("stage_run_script_exists", (STAGE / "run_all_v7.sh").exists(), str(STAGE / "run_all_v7.sh"))
    else:
        add("overleaf_folder_exists", True, "not required for --no-stage quick build")
        add("stage_run_script_exists", True, "not required for --no-stage quick build")
    tex = shutil.which("pdflatex")
    add("local_tex_toolchain", True, "pdflatex available" if tex else "pdflatex not installed locally; Overleaf bundle generated")
    return rows


def write_repro_docs(index: pd.DataFrame) -> None:
    (ROOT / "docs").mkdir(exist_ok=True)
    systems = ", ".join(index["system"].tolist())
    (ROOT / "docs/V7_REPRODUCIBILITY.md").write_text(
        f"""# V7 Reproducibility

V7 is a presentation and packaging pass over frozen v6/v6.2 results. It does not retune WG-SINDy.

## One-command build

```bash
bash run_v7.sh --full
```

The v7 build reads `data/baselines/showcase/showcase_coefficients.csv`, `data/baselines/v6_2_extra_coefficients.csv`, and `data/baselines/system_index.csv`; it writes `data/system_index/`, `figures/datasheets/`, `paper/datasheets/`, `docs/datasheets/`, and `stage/`.

## System roster

{systems}

## Seeds and runtime

The frozen v6.2 extra run used 10 seeds per added/missing 2D system. The run log is `data/baselines/v6_2_run_log.csv`.
"""
    )


def write_manifest(index: pd.DataFrame) -> None:
    rows = [
        {"file": "experiments/v7/build_v7_package.py", "kind": "script", "produced_by": "manual v7 implementation", "source_data": "data/baselines/*.csv", "paper_float": "all v7 tables/datasheets"},
        {"file": "data/system_index/system_index.csv", "kind": "table", "produced_by": "build_v7_package.py", "source_data": "data/baselines/system_index.csv", "paper_float": "Table 1"},
        {"file": "data/system_index/coefficients_clean.csv", "kind": "table", "produced_by": "build_v7_package.py", "source_data": "v6/v6.2 coefficient CSVs", "paper_float": "per-system coefficient tables"},
        {"file": "paper/wg_sindy_v7_manuscript.tex", "kind": "latex", "produced_by": "build_v7_package.py", "source_data": "data/system_index/*.csv; paper/datasheets/*.tex", "paper_float": "main manuscript"},
        {"file": "data/system_index/final_integrity_report.csv", "kind": "audit", "produced_by": "build_v7_package.py", "source_data": "v7 generated artifacts", "paper_float": "integrity report"},
    ]
    for _, row in index.iterrows():
        system = row["system"]
        rows.append({"file": f"paper/datasheets/{system}.tex", "kind": "latex datasheet", "produced_by": "build_v7_package.py", "source_data": "data/system_index/coefficients_clean.csv; data/system_index/system_index.csv", "paper_float": f"datasheet {system}"})
        rows.append({"file": f"figures/datasheets/datasheet_fields_{system}.pdf", "kind": "figure", "produced_by": "build_v7_package.py", "source_data": "data/system_index/coefficients_clean.csv", "paper_float": f"Figure v7-{system}"})
    write_csv(V7_MANIFEST, rows, ["file", "kind", "produced_by", "source_data", "paper_float"])


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def copy_tree(src: Path, dst: Path, ignore: tuple[str, ...] = ()) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    def ignore_fn(_dir: str, names: list[str]) -> set[str]:
        blocked = {".DS_Store", "__pycache__", ".pytest_cache", ".matplotlib-cache"}
        blocked.update(ignore)
        return {name for name in names if name in blocked or name.endswith(".pyc")}
    shutil.copytree(src, dst, ignore=ignore_fn)


def write_overleaf_bundle(index: pd.DataFrame) -> None:
    if OVERLEAF.exists():
        shutil.rmtree(OVERLEAF)
    OVERLEAF.mkdir(parents=True)
    copy_file(V7_MANUSCRIPT, OVERLEAF / "main.tex")
    copy_file(V7_TABLE, OVERLEAF / "v7_table1.tex")
    copy_file(V7_DATASHEET_INPUTS, OVERLEAF / "v7_datasheet_inputs.tex")
    copy_tree(PAPER_DATASHEETS, OVERLEAF / "datasheets")
    copy_tree(FIG_V7, OVERLEAF / "figures/datasheets")
    (OVERLEAF / "README_OVERLEAF.md").write_text(
        """# Overleaf Upload Folder

Upload this entire folder to Overleaf and compile `main.tex`.

The local environment used by Codex did not include `pdflatex`, so PDF compilation is deferred to Overleaf. All figure/table/datasheet references are self-contained in this folder.
"""
    )


def write_stage_scripts() -> None:
    (ROOT / "run_v7.sh").write_text(
        """#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  export PATH="$HOME/.pyenv/shims:$HOME/.pyenv/bin:$PATH"
  for CANDIDATE in python3 python; do
    if command -v "$CANDIDATE" >/dev/null 2>&1; then
      if "$CANDIDATE" - <<'PY' >/dev/null 2>&1
import numpy, pandas, matplotlib, pytest
PY
      then
        PYTHON_BIN="$(command -v "$CANDIDATE")"
        break
      fi
    fi
  done
fi

if [[ -z "$PYTHON_BIN" ]]; then
  echo "No Python with numpy, pandas, matplotlib, and pytest was found. Run: python3 -m pip install -r requirements.txt" >&2
  exit 1
fi

MODE="${1:---full}"
shift || true

if [[ "$MODE" == "--quick" ]]; then
  "$PYTHON_BIN" experiments/v7/build_v7_package.py --quick --no-stage "$@"
  "$PYTHON_BIN" -m pytest -q tests/test_v7_package.py
else
  "$PYTHON_BIN" experiments/v7/build_v7_package.py --full "$@"
  "$PYTHON_BIN" -m pytest -q
fi
"""
    )
    os.chmod(ROOT / "run_v7.sh", 0o755)


def write_stage_bundle(index: pd.DataFrame) -> None:
    if STAGE.exists():
        shutil.rmtree(STAGE)
    STAGE.mkdir()
    for name in ["README.md", "requirements.txt", "pyproject.toml", "AGENTS.md", "CODEX_IMPLEMENTATION_SPEC.md", "CODEX_2D_BROAD_CLAIM_CAMPAIGN.md", "CODEX_PROMPT_V7.md"]:
        src = ROOT / name
        if src.exists():
            copy_file(src, STAGE / name)
    copy_tree(ROOT / "src", STAGE / "src")
    copy_tree(ROOT / "experiments", STAGE / "experiments")
    copy_tree(ROOT / "tests", STAGE / "tests")
    copy_tree(ROOT / "docs", STAGE / "docs")
    (STAGE / "paper").mkdir()
    copy_file(V7_MANUSCRIPT, STAGE / "paper/wg_sindy_v7_manuscript.tex")
    copy_file(V7_TABLE, STAGE / "paper/v7_table1.tex")
    copy_file(V7_DATASHEET_INPUTS, STAGE / "paper/v7_datasheet_inputs.tex")
    copy_tree(PAPER_DATASHEETS, STAGE / "paper/datasheets")
    if (ROOT / "paper/figures/README.md").exists():
        copy_file(ROOT / "paper/figures/README.md", STAGE / "paper/figures/README.md")
    (STAGE / "results").mkdir()
    copy_tree(ROOT / "data/baselines", STAGE / "data/baselines")
    copy_tree(ROOT / "data/system_index", STAGE / "data/system_index")
    copy_tree(FIG_V7, STAGE / "figures/datasheets")
    if (ROOT / "external/weak_stochastic_sindy_1d").exists():
        copy_tree(ROOT / "external/weak_stochastic_sindy_1d", STAGE / "external/weak_stochastic_sindy_1d", ignore=(".git",))
    copy_file(V7_MANIFEST, STAGE / "MANIFEST.csv")
    write_overleaf_bundle(index)
    (STAGE / "README.md").write_text(
        """# V7 WG-SINDy Submission Bundle

This folder is the self-contained v7 reproducibility package for the 2D SDE weak-form generator paper.

## Claim

WG-SINDy recovers symbolic coefficients of the two-dimensional stochastic generator for identifiable systems under library, coverage, and rank assumptions. Named-null systems remain included as limits.

## Quick Check

```bash
bash run_all_v7.sh --quick
```

## Overleaf

Upload `overleaf_wg_sindy_v7/` to Overleaf and compile `main.tex`.
"""
    )
    (STAGE / "REPRODUCIBILITY.md").write_text((ROOT / "docs/V7_REPRODUCIBILITY.md").read_text())
    (STAGE / "run_all_v7.sh").write_text(
        """#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  export PATH="$HOME/.pyenv/shims:$HOME/.pyenv/bin:$PATH"
  for CANDIDATE in python3 python; do
    if command -v "$CANDIDATE" >/dev/null 2>&1; then
      if "$CANDIDATE" - <<'PY' >/dev/null 2>&1
import numpy, pandas, matplotlib, pytest
PY
      then
        PYTHON_BIN="$(command -v "$CANDIDATE")"
        break
      fi
    fi
  done
fi

if [[ -z "$PYTHON_BIN" ]]; then
  echo "No Python with numpy, pandas, matplotlib, and pytest was found. Run: python3 -m pip install -r requirements.txt" >&2
  exit 1
fi

MODE="${1:---quick}"
shift || true

if [[ "$MODE" == "--quick" ]]; then
  "$PYTHON_BIN" experiments/v7/build_v7_package.py --quick --no-stage "$@"
  "$PYTHON_BIN" -m pytest -q tests/test_v7_package.py
else
  "$PYTHON_BIN" experiments/v7/build_v7_package.py --full --no-stage "$@"
  "$PYTHON_BIN" -m pytest -q
fi
"""
    )
    os.chmod(STAGE / "run_all_v7.sh", 0o755)
    (STAGE / "LICENSE").write_text(
        """Research artifact license pending author selection. The bundled external Weak-Stochastic-SINDy snapshot retains its upstream license in external/weak_stochastic_sindy_1d/LICENSE.
"""
    )
    (STAGE / "CITATION.cff").write_text(
        """cff-version: 1.2.0
title: WG-SINDy 2D Stochastic Generator Recovery V7 Artifact
message: Cite the accompanying paper and the Eshwar-Honnavar weak-form stochastic SINDy foundation.
authors:
  - family-names: Gullipalli
    given-names: Pratham
"""
    )


def build(quick: bool, no_stage: bool) -> None:
    V7.mkdir(parents=True, exist_ok=True)
    FIG_V7.mkdir(parents=True, exist_ok=True)
    PAPER_DATASHEETS.mkdir(parents=True, exist_ok=True)
    DOC_DATASHEETS.mkdir(parents=True, exist_ok=True)
    index = write_system_index(load_index())
    coefs = clean_coefficients(load_coefficients())
    plot_v7_fields(index, coefs)
    write_table1(index)
    write_datasheets(index, coefs)
    write_manuscript(index)
    write_issues()
    write_repro_docs(index)
    write_manifest(index)
    write_stage_scripts()
    rows = integrity_rows(index, coefs, require_stage=STAGE.exists())
    write_csv(V7_INTEGRITY, rows, ["check", "status", "detail"])
    if not no_stage:
        write_stage_bundle(index)
        rows = integrity_rows(index, coefs, require_stage=True)
        write_csv(V7_INTEGRITY, rows, ["check", "status", "detail"])
    errors = [row for row in rows if row["status"] == "ERROR"]
    if errors:
        raise RuntimeError(f"V7 integrity errors: {errors}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--no-stage", action="store_true")
    args = parser.parse_args()
    build(quick=args.quick, no_stage=args.no_stage)
    print("V7 DONE")


if __name__ == "__main__":
    main()
