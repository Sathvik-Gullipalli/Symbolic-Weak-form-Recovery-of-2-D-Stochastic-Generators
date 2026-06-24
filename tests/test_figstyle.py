from __future__ import annotations

from sde2d.figstyle import LABELS, METHOD_LABELS, no_raw_key_text


def test_figstyle_human_labels_do_not_expose_raw_metric_keys() -> None:
    assert LABELS
    for visible in LABELS.values():
        assert no_raw_key_text(visible)


def test_v6_1_headtohead_has_exact_named_methods() -> None:
    expected = {
        "WG_SINDY_FROZEN",
        "B0_NAIVE_1D_PORT",
        "KM_LOCAL_MOMENT",
        "WEAK_SINDY_TEMPORAL_PROXY",
        "GEDMD_DENSE_PROXY",
    }
    assert expected.issubset(METHOD_LABELS)
