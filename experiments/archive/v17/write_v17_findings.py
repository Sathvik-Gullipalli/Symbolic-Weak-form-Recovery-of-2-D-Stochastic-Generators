from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from experiments.v17.common import CHOSEN_CONFIG, RESULTS

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    cfg = json.loads(CHOSEN_CONFIG.read_text()) if CHOSEN_CONFIG.exists() else {"config_id": "missing"}
    diagnosis = pd.read_csv(RESULTS / "diagnosis.csv") if (RESULTS / "diagnosis.csv").exists() else pd.DataFrame()
    summary = pd.read_csv(RESULTS / "candidate_summary.csv") if (RESULTS / "candidate_summary.csv").exists() else pd.DataFrame()
    factorial = pd.read_csv(RESULTS / "factorial.csv") if (RESULTS / "factorial.csv").exists() else pd.DataFrame()
    lines = [
        "# V17 findings",
        "",
        "V17 implemented the parallel null-recovery campaign with cached trajectories, populated factorial columns, and default-off intrinsic graft flags.",
        "",
        "## Chosen config",
        "",
        "```json",
        json.dumps(cfg, indent=2),
        "```",
        "",
        "## Parallel/factorial audit",
        "",
    ]
    if not factorial.empty:
        lines.append(f"- factorial rows: {len(factorial)}")
        for col in ["coord", "lags", "moment", "rank", "domain", "selection", "library_atoms"]:
            vals = sorted(str(v) for v in factorial[col].dropna().unique())
            lines.append(f"- `{col}` values: {', '.join(vals)}")
    else:
        lines.append("- factorial not run.")
    lines.extend(["", "## Gate outcome", ""])
    if not summary.empty:
        kept = summary[summary["candidate"].astype(str).isin(["True", "true", "1"])]
        lines.append(f"- surviving candidates: {len(kept)}")
        if kept.empty:
            lines.append("- No v17 factor survived the target-improvement plus no-control-regression/no-FP gate; frozen WG-SINDy remains unpromoted.")
        for _, row in summary.sort_values("target_ratio").head(12).iterrows():
            lines.append(f"- `{row['config_id']}`: target ratio {row['target_ratio']:.3g}, control regression {row['control_regression']:.3g}, FP delta {int(row['fp_delta'])}, candidate={row['candidate']}.")
    else:
        lines.append("- candidate gate not run.")
    lines.extend(["", "## Diagnosis", ""])
    if not diagnosis.empty:
        for system, part in diagnosis.groupby("system"):
            notes = ", ".join(sorted(set(str(v) for v in part["notes"])))
            lines.append(f"- `{system}`: {notes}.")
    else:
        lines.append("- diagnosis not run.")
    lines.extend(
        [
            "",
            "## Honesty notes",
            "",
            "- NG8 library enrichment is implemented as an opt-in extension and is not part of the core polynomial claim unless explicitly enabled.",
            "- SABR drift is reported as N/A when the true drift norm is zero; tensor/leverage remain the meaningful read-outs.",
            "- Stage 3 deep confirmation is skipped when no pilot candidate survives, by design.",
        ]
    )
    out = ROOT / "docs" / "V17_FINDINGS.md"
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

