from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from experiments.v15.common import CHOSEN_CONFIG, RESULTS
from experiments.v15.v15_systems import CLUSTER, CONTROLS, DEEP_SYSTEMS, DOCUMENTED_NULLS

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    diagnosis = pd.read_csv(RESULTS / "diagnosis.csv") if (RESULTS / "diagnosis.csv").exists() else pd.DataFrame()
    candidates = pd.read_csv(RESULTS / "pilot_candidate_summary.csv") if (RESULTS / "pilot_candidate_summary.csv").exists() else pd.DataFrame()
    deep = pd.read_csv(RESULTS / "deep_confirm.csv") if (RESULTS / "deep_confirm.csv").exists() else pd.DataFrame()
    config_text = CHOSEN_CONFIG.read_text() if CHOSEN_CONFIG.exists() else "{}"
    try:
        config = json.loads(config_text)
    except json.JSONDecodeError:
        config = {}
    no_survivor = not config.get("grafts")
    lines = [
        "# V15 findings",
        "",
        "V15 screened intrinsic, default-off grafts over the low-SNR/scoped cluster plus controls. The frozen WG-SINDy defaults remain unchanged; V15 is reached only by explicit overrides from `experiments/v15`.",
        "",
        "## Chosen single config",
        "",
        "```json",
        config_text.strip(),
        "```",
        "",
        "## Pilot gate",
        "",
    ]
    if no_survivor:
        lines.append("- No graft survived the pilot gate, so V15 is a no-promotion result and the frozen WG-SINDy defaults stay frozen.")
    else:
        lines.append(f"- Surviving graft stack: {'+'.join(config.get('grafts', []))}.")
    if not candidates.empty:
        for _, row in candidates.iterrows():
            verdict = "kept" if bool(row["candidate"]) else "dropped"
            lines.append(
                f"- `{row['graft']}` {verdict}: cluster drift ratio {row['cluster_drift_ratio']:.3g}, "
                f"control max regression {row['control_max_regression']:.3g}, FP delta {int(row['fp_delta'])}."
            )
    lines.extend(
        [
            "",
        "## System roster",
        "",
        f"- Cluster: {', '.join(CLUSTER)}",
        f"- Controls: {', '.join(CONTROLS)}",
        f"- Documented nulls: {', '.join(DOCUMENTED_NULLS)}",
        f"- Deep driver used the current repo registry: {len(DEEP_SYSTEMS)} systems. The prompt says 34; the codebase currently exposes {len(DEEP_SYSTEMS)}.",
            "",
            "## Diagnosis",
            "",
        ]
    )
    if not diagnosis.empty:
        for _, row in diagnosis.iterrows():
            lines.append(f"- `{row['system']}`: {row['classification']} (frozen drift {row['drift_l2_frozen']:.3g}, oracle drift {row['drift_l2_oracle']:.3g}; frozen tensor {row['tensor_l2_frozen']:.3g}, oracle tensor {row['tensor_l2_oracle']:.3g}).")
    else:
        lines.append("- Diagnosis not run.")
    lines.extend(["", "## Stage 3" if no_survivor else "## Deep confirmation", ""])
    if not deep.empty:
        cluster = deep[deep["system"].isin(CLUSTER)]
        improved = cluster[cluster["drift_improvement"] > 0]
        regressed = deep[(deep["drift_improvement"] < -0.05) | (deep["v15_fp_median"] > deep["frozen_fp_median"])]
        if no_survivor:
            lines.append("- Since no graft survived, this stage records frozen-vs-frozen baseline identity, not an accepted V15 improvement.")
        lines.append(f"- Cluster systems with positive median drift improvement: {len(improved)} / {len(cluster)}.")
        lines.append(f"- Systems with >5% median drift regression or new median FP: {len(regressed)}.")
        for _, row in cluster.iterrows():
            lines.append(f"- `{row['system']}`: frozen drift {row['frozen_drift_median']:.3g} [{row['frozen_drift_ci_low']:.3g},{row['frozen_drift_ci_high']:.3g}], V15 {row['v15_drift_median']:.3g} [{row['v15_drift_ci_low']:.3g},{row['v15_drift_ci_high']:.3g}], improvement {row['drift_improvement']:.1%}.")
    else:
        if no_survivor:
            lines.append("- Stage 3 was skipped because the full pilot found no surviving graft to promote.")
        else:
            lines.append("- Deep confirmation not run.")
    lines.extend(
        [
            "",
            "## Honesty notes",
            "",
            "- SABR drift remains a martingale zero-truth null; tensor/leverage are the meaningful read-outs.",
            "- Heston log-price drift remains reported as low-SNR/null; V15 does not promote it unless oracle headroom and deep CIs support it.",
            "- `two_factor_vasicek` tensor is tiny in absolute scale, so relative tensor errors are interpreted with the threshold-artifact caveat.",
        ]
    )
    (ROOT / "docs" / "V15_FINDINGS.md").write_text("\n".join(lines) + "\n")
    print("wrote docs/V15_FINDINGS.md")


if __name__ == "__main__":
    main()
