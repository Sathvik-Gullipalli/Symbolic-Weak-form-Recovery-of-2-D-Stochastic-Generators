from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

from sde2d.systems import REGISTRY


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SYSTEMS = {key for key, truth in REGISTRY.items() if truth.dim == 2}


def test_v7_system_index_and_datasheets_cover_registry() -> None:
    index_path = ROOT / "data/system_index/system_index.csv"
    assert index_path.exists(), "run bash run_v7.sh --full before v7 acceptance"
    index = pd.read_csv(index_path)

    assert set(index["system"]) == EXPECTED_SYSTEMS
    assert not index["system"].duplicated().any()

    tex_datasheets = sorted((ROOT / "paper/datasheets").glob("*.tex"))
    md_datasheets = sorted((ROOT / "docs/datasheets").glob("*.md"))
    assert {path.stem for path in tex_datasheets} == EXPECTED_SYSTEMS
    assert {path.stem for path in md_datasheets} == EXPECTED_SYSTEMS

    inputs = (ROOT / "paper/v7_datasheet_inputs.tex").read_text()
    assert inputs.count(r"\input{datasheets/") == len(EXPECTED_SYSTEMS)
    assert r"\section{" in inputs


def test_v7_coefficient_scoring_uses_absolute_error_for_zero_truth() -> None:
    coefs_path = ROOT / "data/system_index/coefficients_clean.csv"
    assert coefs_path.exists(), "run bash run_v7.sh --full before v7 acceptance"
    coefs = pd.read_csv(coefs_path)

    assert set(coefs["system"]) == EXPECTED_SYSTEMS
    assert {"b1", "b2", "a11", "a12", "a22"}.issubset(set(coefs["target"]))

    zero_rows = coefs[coefs["zero_truth"] == True]  # noqa: E712
    nonzero_rows = coefs[coefs["zero_truth"] == False]  # noqa: E712
    assert not zero_rows.empty
    assert (zero_rows["error_type"] == "abs").all()
    assert zero_rows["rel_error_clean"].isna().all()
    assert (nonzero_rows["error_type"] == "rel").all()


def test_v7_figures_and_overleaf_bundle_are_self_contained() -> None:
    pngs = sorted((ROOT / "figures/v7").glob("datasheet_fields_*.png"))
    pdfs = sorted((ROOT / "figures/v7").glob("datasheet_fields_*.pdf"))
    assert len(pngs) == len(EXPECTED_SYSTEMS)
    assert len(pdfs) == len(EXPECTED_SYSTEMS)
    assert {p.stem.replace("datasheet_fields_", "") for p in pngs} == EXPECTED_SYSTEMS

    overleaf = ROOT / "v7-stage/overleaf_wg_sindy_v7"
    if overleaf.exists():
        assert (overleaf / "main.tex").exists()
        assert len(list((overleaf / "datasheets").glob("*.tex"))) == len(EXPECTED_SYSTEMS)
        assert len(list((overleaf / "figures/v7").glob("datasheet_fields_*.pdf"))) == len(EXPECTED_SYSTEMS)
        main_text = (overleaf / "main.tex").read_text()
        assert r"\graphicspath{{../figures/v7/}{figures/v7/}}" in main_text


def test_v7_final_integrity_report_has_no_errors() -> None:
    report_path = ROOT / "data/system_index/final_integrity_report.csv"
    assert report_path.exists(), "run bash run_v7.sh --full before v7 acceptance"
    with report_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    errors = [row for row in rows if row["status"] == "ERROR"]
    assert not errors, errors


def test_v7_stage_bundle_entrypoints_exist_when_packaged() -> None:
    stage = ROOT / "v7-stage"
    if stage.exists():
        assert (stage / "run_all_v7.sh").exists()
        assert (stage / "MANIFEST.csv").exists()
        assert (stage / "README.md").exists()
        assert (stage / "REPRODUCIBILITY.md").exists()
