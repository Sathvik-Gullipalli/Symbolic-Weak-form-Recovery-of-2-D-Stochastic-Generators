from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def paper_dir() -> Path:
    v4 = ROOT / "paper_overleaf_v4"
    return v4 if v4.exists() else ROOT / "paper_overleaf_pre_v3"


def test_v10_r32_coefficients_complete() -> None:
    path = ROOT / "results" / "coefficient_recovery" / "coeff_recovery_R32.csv"
    assert path.exists(), "run python3 experiments/coefficient_recovery.py --R 32 --seeds 4 --steps 8000"
    df = pd.read_csv(path)
    expected = {
        "correlated_ou",
        "coupled_ou",
        "rotational_ou",
        "spiral_sink_corr",
        "van_der_pol",
        "stuart_landau",
        "brusselator",
        "duffing",
        "maier_stein",
        "gradient_potential",
        "diag_multiplicative",
        "nondiag_cholesky",
    }
    assert set(df["system"]) == expected
    assert {"true", "wg_median_when_sel", "wg_sel_rate", "km_baseline", "R", "seeds"}.issubset(df.columns)
    assert set(pd.to_numeric(df["R"])) == {32}
    assert set(pd.to_numeric(df["seeds"])) == {4}
    assert pd.to_numeric(df["wg_sel_rate"]).between(0, 1).all()


def test_v11_method_comparison_and_heatmap() -> None:
    path = ROOT / "results" / "paper" / "per_system_method_comparison.csv"
    assert path.exists(), "run python3 experiments/build_paper.py --full"
    df = pd.read_csv(path)
    expected_methods = {
        "WG_SINDY_FROZEN",
        "KM_LOCAL_MOMENT",
        "WEAK_SINDY_TEMPORAL_PROXY",
        "GEDMD_DENSE_PROXY",
        "VANILLA_SINDY_FD_STLSQ",
    }
    assert set(df["method"]) == expected_methods
    assert df["system"].nunique() == 13
    assert not df.duplicated(["system", "method"]).any()
    assert (ROOT / "figures" / "paper" / "per_system_method_heatmap.pdf").exists()
    assert (paper_dir() / "figures" / "per_system_method_heatmap.pdf").exists()


def test_v11_paper_overleaf_package() -> None:
    paper = paper_dir()
    main = (paper / "main.tex").read_text()
    assert r"\input{coeff_comparison_table}" in main
    assert "VANILLA" not in main
    assert "finite-difference" in main
    assert r"\FloatBarrier % AUTO V11 FLOATBARRIER" in main
    assert r"\label{fig:per-system-method-heatmap}" in main
    assert (paper / "coeff_comparison_table.tex").read_text().startswith("% Auto-generated from results/v10")
    assert (paper / "COMPILE_STATUS.md").exists()
    assert "static_check=PASS" in (paper / "COMPILE_STATUS.md").read_text()

    datasheets = "\n".join(path.read_text() for path in paper.glob("ds_*.tex"))
    assert datasheets.count(r"\begin{figure}[H]") >= 27
    # The live 27-datasheet appendix excludes indep_ou; the heatmap still keeps
    # all 13 matched systems.
    assert datasheets.count("% BEGIN METHOD COMPARISON TABLE") >= 12

    for figure in [
        "act1_naive1d_failure_heatmap.pdf",
        "act2_graft_ladder_climb.pdf",
        "necessity_matrix.pdf",
        "headtohead_bar.pdf",
        "per_system_method_heatmap.pdf",
        "leverage_regime_sweep.pdf",
        "circulation_current_field.pdf",
        "convergence_slope.pdf",
        "honest_null_panel.pdf",
    ]:
        assert (paper / "figures" / figure).exists(), figure


def test_v11_figure_qa_and_stage_mirror() -> None:
    qa = ROOT / "figures" / "FIGURE_QA_FINAL.md"
    assert qa.exists()
    text = qa.read_text()
    assert "Per-system method heatmap" in text
    stage = ROOT / "stage"
    assert (stage / "paper" / "main.tex").exists()
    assert (stage / "paper" / "figures" / "per_system_method_heatmap.pdf").exists()
    assert (stage / "results" / "paper" / "per_system_method_comparison.csv").exists()
