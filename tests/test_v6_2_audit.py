from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sde2d.systems import REGISTRY


ROOT = Path(__file__).resolve().parents[1]


def registry_2d_systems() -> set[str]:
    return {key for key, truth in REGISTRY.items() if truth.dim == 2}


def test_v62_new_canonical_truth_is_finite_and_psd():
    systems = [
        "van_der_pol",
        "fitzhugh_nagumo",
        "stuart_landau",
        "brusselator",
        "maier_stein",
        "duffing",
        "mueller_brown",
        "sabr",
        "gbm_2d",
        "two_factor_vasicek",
    ]
    for key in systems:
        system = REGISTRY[key].cls()
        traj = system.simulate(M=80, seed=202620)
        points = traj[::10]
        drift = system.true_drift(points)
        diffusion = system.true_diffusion(points)
        eigvals = np.linalg.eigvalsh(0.5 * (diffusion + diffusion.transpose(0, 2, 1)))
        assert traj.shape == (81, 2)
        assert np.isfinite(traj).all(), key
        assert np.isfinite(drift).all(), key
        assert np.isfinite(diffusion).all(), key
        assert np.max(np.abs(diffusion - diffusion.transpose(0, 2, 1))) < 1e-10, key
        assert eigvals.min() >= -1e-10, key


def test_v62_system_index_showcase_and_paper_counts_match_registry():
    index_path = ROOT / "data/baselines/system_index.csv"
    assert index_path.exists(), "run ./run_v6_2.sh full before acceptance"
    index = pd.read_csv(index_path)
    expected = registry_2d_systems()
    assert set(index["system"]) == expected
    assert not index["system"].duplicated().any()

    figure_systems = {p.stem.replace("showcase_fields_", "") for p in (ROOT / "figures/baselines").glob("showcase_fields_*.png")}
    assert figure_systems == expected

    segment_path = ROOT / "paper/v6_2_system_segments.tex"
    assert segment_path.exists()
    segment_text = segment_path.read_text()
    assert segment_text.count("\\subsection{") == len(expected)


def test_v62_data_integrity_report_has_no_errors():
    report_path = ROOT / "data/baselines/data_integrity_report.csv"
    assert report_path.exists(), "run ./run_v6_2.sh full before acceptance"
    report = pd.read_csv(report_path)
    errors = report[report["status"] == "ERROR"]
    assert errors.empty, errors.head().to_dict(orient="records")


def test_v62_new_systems_have_ten_seed_frozen_results():
    raw_path = ROOT / "data/baselines/v6_2_extra_summary_raw.csv"
    assert raw_path.exists(), "run ./run_v6_2.sh full before acceptance"
    raw = pd.read_csv(raw_path)
    for key in [
        "van_der_pol",
        "fitzhugh_nagumo",
        "stuart_landau",
        "brusselator",
        "maier_stein",
        "duffing",
        "mueller_brown",
        "sabr",
        "gbm_2d",
        "two_factor_vasicek",
    ]:
        part = raw[raw["system"] == key]
        assert part["seed"].nunique() >= 10, key
        assert set(part["variant_id"]) == {"WG_SINDY_FROZEN"}
