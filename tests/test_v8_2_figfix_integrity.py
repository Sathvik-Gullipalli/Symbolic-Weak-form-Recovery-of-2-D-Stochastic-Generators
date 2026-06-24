from __future__ import annotations

from pathlib import Path

import pandas as pd

from experiments.archive.v8_2.figfix_integrity import REPORT, run
from sde2d.systems import REGISTRY


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_2D = {key for key, truth in REGISTRY.items() if truth.dim == 2}


def test_counts_consistent() -> None:
    assert len(EXPECTED_2D) == 29
    for rel in ["results/v6/system_index.csv", "results/v7/system_index.csv"]:
        df = pd.read_csv(ROOT / rel)
        assert set(df["system"]) == EXPECTED_2D
        assert not df["system"].duplicated().any()

    assert len(list((ROOT / "figures/v6").glob("showcase_fields_*.pdf"))) == 29
    assert len(list((ROOT / "figures/v7").glob("datasheet_fields_*.pdf"))) == 29
    assert len(list((ROOT / "paper_overleaf/figures").glob("datasheet_fields_*.pdf"))) == 29
    assert len(list((ROOT / "paper/datasheets").glob("*.tex"))) == 29
    assert len(list((ROOT / "docs/datasheets").glob("*.md"))) == 29


def test_zero_truth_clean_table_is_the_datasheet_source() -> None:
    coefs = pd.read_csv(ROOT / "results/v7/coefficients_clean.csv")
    zero_rows = coefs[coefs["zero_truth"] == True]  # noqa: E712
    assert not zero_rows.empty
    assert (zero_rows["error_type"] == "abs").all()
    assert zero_rows["rel_error_clean"].isna().all()

    paper_text = (ROOT / "paper/wg_sindy_v7_manuscript.tex").read_text()
    assert "v7_datasheet_inputs.tex" in paper_text
    assert r"coefficients\_clean.csv" in (ROOT / "paper/datasheets/heston_logsv.tex").read_text()


def test_v8_2_integrity_report_empty() -> None:
    issues = run(fix=False)
    assert issues == []
    lines = REPORT.read_text().splitlines()
    assert lines == ["file,check,column,rows,detail"]


def test_final_figure_text_has_no_known_stale_strings() -> None:
    text = "\n".join(
        [
            (ROOT / "experiments/v6/render_v6_1_figures.py").read_text(),
            (ROOT / "paper/wg_sindy_v6_manuscript.tex").read_text(),
            (ROOT / "paper_overleaf/main.tex").read_text(),
        ]
    )
    stale = [
        "V6.2 broad zoo",
        "Per-column normalized positive degradation",
        "per-column normalized positive degradation",
        "within the v6 in-scope suite",
        "v6 in-scope necessity matrix",
    ]
    for token in stale:
        assert token not in text
