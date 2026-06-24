from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_v9_per_system_comparison_contract() -> None:
    path = ROOT / "results" / "v9" / "per_system_comparison.csv"
    assert path.exists(), "run python3 -m experiments.v9.baselines_per_system --full"
    df = pd.read_csv(path)
    required = {
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
    }
    assert required.issubset(df.columns)
    assert not df.duplicated(["system", "method"]).any()

    system_index = pd.read_csv(ROOT / "results" / "v6" / "system_index.csv")
    assert set(system_index["system"]) <= set(df["system"])

    matched = df.loc[df["comparison_scope"] == "matched_headtohead"]
    assert matched["system"].nunique() == 13
    assert matched["method"].nunique() == 6
    for column in ["drift_l2", "tensor_rel_l2", "psd_valid"]:
        numeric = pd.to_numeric(df[column], errors="coerce").dropna()
        assert len(numeric) > 0
        assert (numeric >= 0).all()
    cosines = pd.to_numeric(df["a12_cosine"], errors="coerce").dropna()
    assert ((cosines >= -1) & (cosines <= 1)).all()


def test_v9_figures_and_scale_audit_exist() -> None:
    fig_dir = ROOT / "figures" / "v8_2"
    assert (fig_dir / "per_system_method_heatmap.pdf").exists()
    assert (fig_dir / "per_system_method_heatmap.png").exists()
    audit = fig_dir / "SCALE_LEGEND_AUDIT.md"
    assert audit.exists()
    text = audit.read_text()
    for stem in [
        "act1_naive1d_failure_heatmap",
        "act2_graft_ladder_climb",
        "necessity_matrix",
        "headtohead_bar",
        "per_system_method_heatmap",
        "leverage_regime_sweep",
        "circulation_current_field",
        "convergence_slope",
        "honest_null_panel",
        "datasheet_fields_*",
    ]:
        assert stem in text


def test_v9_paper_and_datasheets_wired() -> None:
    main = (ROOT / "paper_overleaf" / "main.tex").read_text()
    assert "per_system_method_heatmap.pdf" in main
    assert r"\label{sec:per-system-baselines}" in main
    assert r"\label{fig:per-system-method-heatmap}" in main
    assert (ROOT / "paper_overleaf" / "figures" / "per_system_method_heatmap.pdf").exists()

    system_index = pd.read_csv(ROOT / "results" / "v6" / "system_index.csv")
    scoped = system_index.loc[system_index["verdict"].isin(["PASS", "SCOPED_REVIEW"])]
    datasheet_text = "\n".join(path.read_text() for path in sorted((ROOT / "paper_overleaf").glob("ds_*.tex")))
    if "% BEGIN COMPONENT BLOCK" in datasheet_text:
        for _, row in scoped.iterrows():
            label = rf"\label{{{row['paper_subsection_label']}}}"
            assert label in datasheet_text
        assert datasheet_text.count("% BEGIN COMPONENT BLOCK") == len(scoped)
        assert datasheet_text.count("% BEGIN METHOD TABLE") == len(scoped)
        assert r"\paragraph{Component-by-component.}" in datasheet_text
        assert r"\paragraph{Per-system baseline table.}" in datasheet_text
    else:
        # V11 promotes paper_overleaf/ to the 27-datasheet final appendix; the
        # two omitted PASS systems remain in the master index/heatmap.
        omitted = {"indep_ou", "fitzhugh_nagumo"}
        for _, row in scoped.loc[~scoped["system"].isin(omitted)].iterrows():
            label = rf"\label{{{row['paper_subsection_label']}}}"
            assert label in datasheet_text
        assert datasheet_text.count("% BEGIN METHOD COMPARISON TABLE") >= 12
        assert (ROOT / "paper_overleaf" / "coeff_comparison_table.tex").exists()


def test_v9_stage_mirror_when_present() -> None:
    stage = ROOT / "v7-stage"
    if not stage.exists():
        return
    assert (stage / "paper_overleaf" / "main.tex").exists()
    assert (stage / "paper_overleaf" / "figures" / "per_system_method_heatmap.pdf").exists()
    assert (stage / "results" / "v9" / "per_system_comparison.csv").exists()
    assert (stage / "figures" / "v8_2" / "SCALE_LEGEND_AUDIT.md").exists()
