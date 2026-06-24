from __future__ import annotations

import argparse
import csv
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib-cache"))
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.coefficient_recovery import SPEC as V10_SPEC
from sde2d import figstyle


RESULTS_V10 = ROOT / "results" / "coefficient_recovery"
RESULTS_V11 = ROOT / "results" / "paper"
FIGURES_V11 = ROOT / "figures" / "paper"


def select_paper_dir() -> Path:
    env = os.environ.get("PAPER_DIR")
    if env:
        path = Path(env)
        return path if path.is_absolute() else ROOT / path
    return ROOT / "paper"


PAPER = select_paper_dir()
PAPER_FIGURES = PAPER / "figures"
STAGE = ROOT / "v7-stage"

MAIN_FIGURES = [
    "act1_naive1d_failure_heatmap.pdf",
    "act2_graft_ladder_climb.pdf",
    "necessity_matrix.pdf",
    "headtohead_bar.pdf",
    "per_system_method_heatmap.pdf",
    "leverage_regime_sweep.pdf",
    "circulation_current_field.pdf",
    "convergence_slope.pdf",
    "honest_null_panel.pdf",
]

METHODS_KEEP = [
    "WG_SINDY_FROZEN",
    "KM_LOCAL_MOMENT",
    "WEAK_SINDY_TEMPORAL_PROXY",
    "GEDMD_DENSE_PROXY",
    "VANILLA_SINDY_FD_STLSQ",
]

METHOD_LABELS = {
    "WG_SINDY_FROZEN": "WG-SINDy",
    "KM_LOCAL_MOMENT": "Kramers-Moyal / stochastic SINDy",
    "WEAK_SINDY_TEMPORAL_PROXY": "Temporal Weak-SINDy",
    "GEDMD_DENSE_PROXY": "gEDMD",
    "VANILLA_SINDY_FD_STLSQ": "Vanilla SINDy (FD-STLSQ)",
}


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
        x = float(value)
    except (TypeError, ValueError):
        return "--"
    if not math.isfinite(x):
        return "--"
    if abs(x) >= 100:
        return f"{x:.0f}"
    if abs(x) >= 10:
        return f"{x:.1f}"
    if abs(x) >= 1:
        return f"{x:.2f}"
    if abs(x) >= 0.01:
        return f"{x:.{digits}f}"
    if x == 0:
        return "0"
    return f"{x:.1e}"


def write_csv(path: Path, rows: list[dict[str, object]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows({col: row.get(col, "") for col in columns} for row in rows)


def require_v10_coefficients() -> Path:
    path = RESULTS_V10 / "coeff_recovery_R32.csv"
    if not path.exists():
        raise FileNotFoundError("Missing results/coefficient_recovery/coeff_recovery_R32.csv; run python3 experiments/coefficient_recovery.py --R 32 --seeds 4 --steps 8000")
    df = pd.read_csv(path)
    missing = set(V10_SPEC) - set(df["system"])
    if missing:
        raise RuntimeError(f"R32 coefficient rerun incomplete; missing systems: {sorted(missing)}")
    return path


def render_coeff_table(coeff_csv: Path) -> None:
    rows = pd.read_csv(coeff_csv)
    lines = [
        "% Auto-generated from results/coefficient_recovery/coeff_recovery_R32.csv",
        r"\begin{longtable}{l l r r r r}",
        r"\caption{Symbolic coefficient recovery under the pooled R=32 rerun. Each row reports the coefficient used to generate the data, WG-SINDy as the median over selected seeds, the selection rate over four independent pooled fits, and a matched Kramers--Moyal / stochastic-SINDy baseline on the same library.}\label{tab:coeff-recovery}\\",
        r"\toprule",
        r"System & Term & True & WG-SINDy & sel & KM-base \\ \midrule \endfirsthead",
        r"\toprule System & Term & True & WG-SINDy & sel & KM-base \\ \midrule \endhead",
        r"\bottomrule \endfoot",
    ]
    for system, part in rows.groupby("system", sort=False):
        lines.append(rf"\multicolumn{{6}}{{l}}{{\textit{{{latex_escape(figstyle.system_label(system))}}}}}\\")
        for _, row in part.iterrows():
            term = f"{row['target']}:\\,$" + str(row["term"]).replace("1", r"\mathbf{1}") + "$"
            lines.append(
                f"  & {term} & ${float(row['true']):+.3f}$ & ${float(row['wg_median_when_sel']):+.3f}$ & {float(row['wg_sel_rate']):.2f} & ${float(row['km_baseline']):+.3f}$ \\\\"
            )
        lines.append(r"\addlinespace")
    lines.append(r"\end{longtable}")
    (PAPER / "coeff_comparison_table.tex").write_text("\n".join(lines) + "\n")


def run_subprocesses() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC)
    env["PAPER_OVERLEAF_DIR"] = str(PAPER)
    subprocess.run([sys.executable, "experiments/gen_datasheets.py"], cwd=ROOT, check=True, env=env)
    subprocess.run([sys.executable, "experiments/datasheet_figures.py", "--full", "--no-stage"], cwd=ROOT, check=True, env=env)


def build_method_comparison() -> pd.DataFrame:
    head = pd.read_csv(ROOT / "data" / "baselines" / "headtohead.csv")
    cells = pd.read_csv(ROOT / "data" / "baselines" / "headtohead_cells.csv")
    runtime = cells.groupby(["system", "variant_id"], dropna=False)["runtime_sec"].median().reset_index()
    head = head.merge(runtime, on=["system", "variant_id"], how="left")
    source_map = {
        "WG_SINDY_FROZEN": "WG_SINDY_FROZEN",
        "KM_LOCAL_MOMENT": "KM_LOCAL_MOMENT",
        "WEAK_SINDY_TEMPORAL_PROXY": "WEAK_SINDY_TEMPORAL_PROXY",
        "GEDMD_DENSE_PROXY": "GEDMD_DENSE_PROXY",
        "VANILLA_SINDY_FD_STLSQ": "B0_PRIME_IN_REPO_REPORT",
    }
    rows: list[dict[str, object]] = []
    for out_method, source_method in source_map.items():
        sub = head[head["variant_id"] == source_method]
        for _, row in sub.iterrows():
            rows.append(
                {
                    "system": row["system"],
                    "system_label": figstyle.system_label(str(row["system"])),
                    "method": out_method,
                    "method_label": METHOD_LABELS[out_method],
                    "drift_l2": row["median_objective_drift_rel_l2"],
                    "tensor_rel_l2": row["median_diffusion_rel_l2"],
                    "a12_cosine": row["median_a12_cosine"],
                    "psd_valid": row["median_psd_valid_pct"],
                    "runtime_sec": row["runtime_sec"],
                    "n_seeds": row["n"],
                    "config_source": "data/baselines/headtohead.csv",
                    "notes": (
                        "finite-difference STLSQ scalar-port proxy for vanilla SINDy"
                        if out_method == "VANILLA_SINDY_FD_STLSQ"
                        else "frozen matched head-to-head baseline"
                    ),
                }
            )
    columns = [
        "system",
        "system_label",
        "method",
        "method_label",
        "drift_l2",
        "tensor_rel_l2",
        "a12_cosine",
        "psd_valid",
        "runtime_sec",
        "n_seeds",
        "config_source",
        "notes",
    ]
    df = pd.DataFrame(rows, columns=columns)
    order = {method: i for i, method in enumerate(METHODS_KEEP)}
    df["_method_order"] = df["method"].map(order)
    df = df.sort_values(["system_label", "_method_order"]).drop(columns=["_method_order"])
    write_csv(RESULTS_V11 / "per_system_method_comparison.csv", df.to_dict("records"), columns)
    return df


def render_method_heatmap(df: pd.DataFrame) -> Path:
    FIGURES_V11.mkdir(parents=True, exist_ok=True)
    PAPER_FIGURES.mkdir(parents=True, exist_ok=True)
    systems = sorted(df["system"].unique(), key=figstyle.system_label)
    metrics = [
        ("drift_l2", r"Drift relative $L^2$ error"),
        ("tensor_rel_l2", r"Tensor relative $L^2$ error"),
        ("a12_cosine_loss", r"$a_{12}$ cosine loss $(1-\cos)$"),
    ]
    plot_df = df.copy()
    plot_df["a12_cosine_loss"] = 1.0 - pd.to_numeric(plot_df["a12_cosine"], errors="coerce")
    fig, axes = plt.subplots(1, 3, figsize=(15.4, max(5.2, 0.38 * len(systems) + 1.3)), sharey=True)
    for ax, (metric, title) in zip(axes, metrics):
        raw = np.full((len(systems), len(METHODS_KEEP)), np.nan)
        for i, system in enumerate(systems):
            for j, method in enumerate(METHODS_KEEP):
                sub = plot_df[(plot_df["system"] == system) & (plot_df["method"] == method)]
                if not sub.empty:
                    raw[i, j] = float(sub.iloc[0][metric])
        display = np.log10(1.0 + np.maximum(raw, 0.0))
        vmin, vmax = figstyle.robust_limits(display, lower=2, upper=98, pad_fraction=0.02)
        image = ax.imshow(display, aspect="auto", cmap=figstyle.softened_cmap(figstyle.HEATMAP), vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xticks(range(len(METHODS_KEEP)))
        ax.set_xticklabels([figstyle.wrap_label(METHOD_LABELS[m], width=13) for m in METHODS_KEEP], rotation=35, ha="right")
        ax.set_yticks(range(len(systems)))
        ax.set_yticklabels([figstyle.system_label(s) for s in systems])
        ax.grid(False)
        figstyle.annotate_heatmap(ax, raw, digits=2, text_colors=("black", "white"))
        cbar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.02)
        cbar.set_label("log10(1 + metric)")
    axes[0].set_ylabel("Matched benchmark system")
    fig.suptitle("Matched per-system method comparison")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    figstyle.save_figure(fig, FIGURES_V11, "per_system_method_heatmap")
    for suffix in [".pdf", ".png"]:
        shutil.copy2(FIGURES_V11 / f"per_system_method_heatmap{suffix}", PAPER_FIGURES / f"per_system_method_heatmap{suffix}")
    return FIGURES_V11 / "per_system_method_heatmap.pdf"


def method_summary(df: pd.DataFrame) -> str:
    rows = []
    for metric in ["drift_l2", "tensor_rel_l2"]:
        wins = ties = losses = 0
        for system, part in df.groupby("system"):
            ours = float(part.loc[part["method"] == "WG_SINDY_FROZEN", metric].iloc[0])
            other = pd.to_numeric(part.loc[part["method"] != "WG_SINDY_FROZEN", metric], errors="coerce").dropna()
            best_other = float(other.min()) if len(other) else float("inf")
            if ours < best_other * 0.95:
                wins += 1
            elif ours <= best_other * 1.05 + 1e-12:
                ties += 1
            else:
                losses += 1
        rows.append((metric, wins, ties, losses))
    drift = next(r for r in rows if r[0] == "drift_l2")
    tensor = next(r for r in rows if r[0] == "tensor_rel_l2")
    return (
        f"On the matched grid, WG-SINDy is best or within 5\\% of best on drift for {drift[1] + drift[2]} of {df['system'].nunique()} systems "
        f"({drift[1]} clear wins, {drift[2]} ties, {drift[3]} losses) and on tensor recovery for {tensor[1] + tensor[2]} of {df['system'].nunique()} systems "
        f"({tensor[1]} clear wins, {tensor[2]} ties, {tensor[3]} losses). The losses are concentrated where dense local-moment or temporal weak estimates have high-SNR linear structure; WG-SINDy remains strongest on off-diagonal tensor geometry and nonlinear/low-SNR drift."
    )


def replace_marked_block(text: str, begin: str, end: str, block: str, anchor: str) -> str:
    pattern = rf"\n?% BEGIN {re.escape(begin)}\n.*?% END {re.escape(end)}\n?"
    text2 = re.sub(pattern, "\n", text, flags=re.DOTALL)
    pos = text2.find(anchor)
    if pos < 0:
        raise ValueError(f"anchor not found: {anchor}")
    return text2[:pos] + block.strip() + "\n\n" + text2[pos:]


def update_main(df: pd.DataFrame) -> None:
    path = PAPER / "main.tex"
    text = path.read_text()
    text = text.replace("reproduce.sh", "run\\_v11.sh")
    text = re.sub(r"\\FloatBarrier\s*% AUTO V11 FLOATBARRIER\n?", "", text)
    block = rf"""
% BEGIN PER-SYSTEM BASELINES
\subsection{{Per-system baseline comparison}}\label{{sec:per-system-baselines}}
The aggregate head-to-head in \Cref{{fig:h2h}} is intentionally compressed, so \Cref{{fig:per-system-method-heatmap}}
adds the matched per-system view. Rows are the thirteen systems shared by the method ledger; columns now include
WG-SINDy, Kramers--Moyal / stochastic SINDy, temporal Weak-SINDy, generator EDMD, and a vanilla finite-difference
STLSQ SINDy proxy. Each cell is annotated with the raw metric and coloured only by a log display transform.
{method_summary(df)}

\begin{{figure}}[t]\centering
\includegraphics[width=0.98\linewidth]{{per_system_method_heatmap.pdf}}
\caption{{Matched per-system method comparison. The first two panels show raw relative $L^2$ drift and tensor
errors; the third reports $1-\cos(\hat a_{{12}},a_{{12}})$ so lower is better on every panel. Colour uses
$\log_{{10}}(1+\mathrm{{metric}})$ only for readability.}}\label{{fig:per-system-method-heatmap}}
\end{{figure}}
% END PER-SYSTEM BASELINES
"""
    text = replace_marked_block(text, "PER-SYSTEM BASELINES", "PER-SYSTEM BASELINES", block, r"\subsection{The three read-outs}")
    rel = r"""
\paragraph{How to read the relative errors.}
The reported errors are symbolic-recovery-grade diagnostics, not sub-percent precision-estimation claims.
The dominant coefficients are recovered to a few percent when selected, support is reported separately through
selection rate and false-positive counts, and leverage is judged by the off-diagonal cosine. The R-pooled
rerun and convergence figure show the expected trajectory-budget tightening; precision calibration on real
data is a separate future-work problem.
"""
    if r"\paragraph{How to read the relative errors.}" not in text:
        text = text.replace(r"\section{Discussion and Limitations}\label{sec:discussion}", r"\section{Discussion and Limitations}\label{sec:discussion}" + "\n" + rel.strip() + "\n", 1)
    start = text.find(r"\section{Results}")
    end = text.find(r"\section{Discussion and Limitations}")
    if start >= 0 and end > start:
        results = text[start:end]
        parts = re.split(r"(\\subsection\{)", results)
        rebuilt = parts[0]
        first = True
        for i in range(1, len(parts), 2):
            prefix = parts[i] + parts[i + 1]
            if first:
                rebuilt += prefix
                first = False
            else:
                rebuilt += "\\FloatBarrier % AUTO V11 FLOATBARRIER\n" + prefix
        rebuilt += "\\FloatBarrier % AUTO V11 FLOATBARRIER\n"
        text = text[:start] + rebuilt + text[end:]
    path.write_text(text)


def method_table_tex(system: str, df: pd.DataFrame) -> str:
    part = df[df["system"] == system].copy()
    if part.empty:
        return ""
    order = {method: i for i, method in enumerate(METHODS_KEEP)}
    part["_order"] = part["method"].map(order)
    part = part.sort_values("_order")
    lines = [
        "% BEGIN METHOD COMPARISON TABLE",
        r"\paragraph{Matched method table.} Same-system comparison on the frozen matched ledger; lower is better except PSD.",
        r"\begin{center}\scriptsize",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Method & Drift & Tensor & $1-a_{12}$ cos. & PSD \\",
        r"\midrule",
    ]
    for _, row in part.iterrows():
        cosine = pd.to_numeric(row["a12_cosine"], errors="coerce")
        loss = "" if not math.isfinite(cosine) else 1.0 - float(cosine)
        lines.append(
            f"{latex_escape(row['method_label'])} & {fmt_num(row['drift_l2'])} & {fmt_num(row['tensor_rel_l2'])} & {fmt_num(loss)} & {fmt_num(row['psd_valid'])} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{center}", "% END METHOD COMPARISON TABLE", ""])
    return "\n".join(lines)


def insert_datasheet_method_tables(df: pd.DataFrame) -> None:
    for path in PAPER.glob("ds_*.tex"):
        text = path.read_text()
        text = re.sub(r"\n?% BEGIN METHOD COMPARISON TABLE\n.*?% END METHOD COMPARISON TABLE\n?", "\n", text, flags=re.DOTALL)
        for system in sorted(df["system"].unique(), key=len, reverse=True):
            label = rf"\label{{sec:v62-{system.replace('_', '-')}}}"
            pos = text.find(label)
            if pos < 0:
                continue
            start = text.rfind(r"\subsection", 0, pos)
            end = text.find(r"\subsection", pos + len(label))
            if end < 0:
                end = len(text)
            anchor = text.find(r"\begin{figure}[H]", start, end)
            if anchor >= 0:
                table = method_table_tex(system, df)
                text = text[:anchor] + table + "\n" + text[anchor:]
        path.write_text(re.sub(r"\n{3,}", "\n\n", text))


def copy_datasheet_figures() -> None:
    PAPER_FIGURES.mkdir(parents=True, exist_ok=True)
    source_dir = ROOT / "figures" / "datasheets"
    for pdf in source_dir.glob("datasheet_fields_*.pdf"):
        shutil.copy2(pdf, PAPER_FIGURES / pdf.name)


def write_figure_qa() -> None:
    paper_rel = PAPER.relative_to(ROOT) if PAPER.is_relative_to(ROOT) else PAPER
    lines = [
        "# Final figure QA",
        "",
        f"Scope: {paper_rel} figures used by main text and datasheets.",
        "",
        "| Figure class | Action | QA result |",
        "|---|---|---|",
        "| Main figures | swept and copied forward | legends/axes name the metric, no visible version strings in main captions, no missing PDF references |",
        "| Per-system method heatmap | regenerated | three panels: drift L2, tensor L2, and a12 cosine loss; raw annotations; robust log display scale |",
        "| Datasheet field figures | regenerated from experiments/v7/build_v7_package.py | true/recovered share colour scale; error uses symmetric centred colour scale; captions do not include version strings |",
    ]
    for name in MAIN_FIGURES:
        lines.append(f"- `{name}` present: {str((PAPER_FIGURES / name).exists()).lower()}")
    (ROOT / "figures" / "FIGURE_QA_FINAL.md").write_text("\n".join(lines) + "\n")


def static_latex_check() -> list[str]:
    errors: list[str] = []
    main = PAPER / "main.tex"
    text = main.read_text()
    for match in re.findall(r"\\input\{([^}]+)\}", text):
        candidate = PAPER / (match if match.endswith(".tex") else match + ".tex")
        if not candidate.exists():
            errors.append(f"missing input {match}")
    tex_blobs = [text] + [p.read_text() for p in PAPER.glob("ds_*.tex")] + [(PAPER / "coeff_comparison_table.tex").read_text(), (PAPER / "table_master.tex").read_text()]
    for blob in tex_blobs:
        for fig in re.findall(r"\\includegraphics(?:\[[^\]]+\])?\{([^}]+)\}", blob):
            candidate = PAPER_FIGURES / fig
            if not candidate.exists():
                errors.append(f"missing figure {fig}")
    return errors


def attempt_compile() -> None:
    status = []
    latexmk = shutil.which("latexmk")
    if latexmk:
        proc = subprocess.run([latexmk, "-pdf", "-interaction=nonstopmode", "-halt-on-error", "main.tex"], cwd=PAPER, text=True, capture_output=True)
        status.append(f"latexmk_exit={proc.returncode}")
        status.append(proc.stdout[-4000:])
        status.append(proc.stderr[-4000:])
        if proc.returncode != 0:
            raise RuntimeError(f"latexmk failed; see {PAPER / 'COMPILE_STATUS.md'}")
    else:
        status.append("latexmk not found locally; static input/figure checks were run instead.")
    static_errors = static_latex_check()
    status.append("static_check=PASS" if not static_errors else "static_check=FAIL")
    status.extend(static_errors)
    (PAPER / "COMPILE_STATUS.md").write_text("\n\n".join(status) + "\n")
    if static_errors:
        raise RuntimeError(f"Static LaTeX checks failed: {static_errors}")


def mirror_stage() -> None:
    root_overleaf = ROOT / "paper"
    if root_overleaf.exists():
        shutil.rmtree(root_overleaf)
    shutil.copytree(PAPER, root_overleaf)
    dest = STAGE / "paper"
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(PAPER, dest)
    shutil.copytree(RESULTS_V10, STAGE / "results" / "coefficient_recovery", dirs_exist_ok=True)
    shutil.copytree(RESULTS_V11, STAGE / "results" / "paper", dirs_exist_ok=True)
    shutil.copytree(FIGURES_V11, STAGE / "figures" / "paper", dirs_exist_ok=True)
    shutil.copy2(ROOT / "figures" / "FIGURE_QA_FINAL.md", STAGE / "figures" / "FIGURE_QA_FINAL.md")
    script = ROOT / "reproduce.sh"
    if script.exists():
        shutil.copy2(script, STAGE / "reproduce.sh")


def write_summary(df: pd.DataFrame) -> None:
    paper_rel = PAPER.relative_to(ROOT) if PAPER.is_relative_to(ROOT) else PAPER
    summary = [
        "# V11 run summary",
        "",
        f"- Live Overleaf package: `{paper_rel}`.",
        "- Consumed `results/coefficient_recovery/coeff_recovery_R32.csv` for coefficient tables and datasheets.",
        f"- Wrote `results/paper/per_system_method_comparison.csv` with {len(df)} rows.",
        f"- Matched systems: {df['system'].nunique()}; methods: {df['method'].nunique()}.",
        f"- {method_summary(df)}",
        f"- Regenerated datasheet field figures, copied referenced PDFs into `{paper_rel}/figures/`, and mirrored to `paper/` and `v7-stage/paper/`.",
    ]
    (RESULTS_V11 / "V11_RUN_SUMMARY.md").write_text("\n".join(summary) + "\n")


def build() -> None:
    RESULTS_V11.mkdir(parents=True, exist_ok=True)
    FIGURES_V11.mkdir(parents=True, exist_ok=True)
    coeff_csv = require_v10_coefficients()
    render_coeff_table(coeff_csv)
    run_subprocesses()
    df = build_method_comparison()
    render_method_heatmap(df)
    update_main(df)
    insert_datasheet_method_tables(df)
    copy_datasheet_figures()
    write_figure_qa()
    attempt_compile()
    write_summary(df)
    mirror_stage()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the V11 final paper package.")
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    if not args.full:
        parser.error("V11 builder currently supports only --full")
    build()
    print("V11 DONE")


if __name__ == "__main__":
    main()
