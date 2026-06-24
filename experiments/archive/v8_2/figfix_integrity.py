from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/v8_2"
REPORT = OUT / "integrity_report.csv"

NONFINITE_TOKENS = {"nan", "+nan", "-nan", "inf", "+inf", "-inf", "infinity", "+infinity", "-infinity"}


def paper_bound_csvs() -> list[Path]:
    paths: list[Path] = []
    paths.extend(sorted((ROOT / "results/v6").glob("*.csv")))
    paths.extend(sorted((ROOT / "results/v6/showcase").glob("*.csv")))
    paths.extend(sorted((ROOT / "results/v7").glob("*.csv")))
    return paths


def _is_nonfinite_token(value: object) -> bool:
    return str(value).strip().lower() in NONFINITE_TOKENS


def normalize_nonfinite_literals(paths: list[Path]) -> int:
    changed = 0
    for path in paths:
        with path.open(newline="") as f:
            reader = csv.reader(f)
            rows = list(reader)
        out_rows = []
        touched = False
        for row in rows:
            out = []
            for value in row:
                if _is_nonfinite_token(value):
                    out.append("")
                    touched = True
                else:
                    out.append(value)
            out_rows.append(out)
        if touched:
            with path.open("w", newline="") as f:
                writer = csv.writer(f, lineterminator="\n")
                writer.writerows(out_rows)
            changed += 1
    return changed


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.replace("", np.nan), errors="coerce").dropna()


def scan_integrity(paths: list[Path]) -> list[dict[str, object]]:
    issues: list[dict[str, object]] = []
    for path in paths:
        rel = str(path.relative_to(ROOT))
        try:
            df = pd.read_csv(path, keep_default_na=False, low_memory=False)
        except Exception as exc:  # pragma: no cover - defensive report path
            issues.append({"file": rel, "check": "read_csv", "column": "", "rows": "", "detail": str(exc)})
            continue

        for col in df.columns:
            literal = df[col].map(_is_nonfinite_token)
            if literal.any():
                issues.append({"file": rel, "check": "nonfinite_literal", "column": col, "rows": int(literal.sum()), "detail": "literal NaN/Inf token"})

        dup = df.duplicated()
        if dup.any():
            issues.append({"file": rel, "check": "duplicate_row", "column": "*", "rows": int(dup.sum()), "detail": "exact duplicate rows"})

        for col in df.columns:
            name = col.lower()
            values = _numeric(df[col])
            if values.empty:
                continue
            if "cosine" in name or name in {"rho_true", "rho_hat", "rho_tensor_median", "rho"}:
                bad = ~values.between(-1.0 - 1e-9, 1.0 + 1e-9)
                if bad.any():
                    issues.append({"file": rel, "check": "cosine_or_rho_bounds", "column": col, "rows": int(bad.sum()), "detail": f"range {values[bad].min():.6g}..{values[bad].max():.6g}"})
            if "psd" in name and any(token in name for token in ("pct", "rate", "valid", "fraction")):
                bad = ~values.between(0.0 - 1e-9, 1.0 + 1e-9)
                if bad.any():
                    issues.append({"file": rel, "check": "psd_bounds", "column": col, "rows": int(bad.sum()), "detail": f"range {values[bad].min():.6g}..{values[bad].max():.6g}"})
            if any(token in name for token in ("rel_l2", "rel-l2", "relative_l2", "tensor_rel_l2", "drift_l2")):
                bad = values < -1e-12
                if bad.any():
                    issues.append({"file": rel, "check": "negative_error", "column": col, "rows": int(bad.sum()), "detail": f"min {values[bad].min():.6g}"})
            if values.map(lambda x: not math.isfinite(float(x))).any():
                issues.append({"file": rel, "check": "numeric_nonfinite", "column": col, "rows": "", "detail": "parsed non-finite numeric value"})
    return issues


def write_report(issues: list[dict[str, object]]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fields = ["file", "check", "column", "rows", "detail"]
    with REPORT.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(issues)


def run(*, fix: bool = False) -> list[dict[str, object]]:
    paths = paper_bound_csvs()
    if fix:
        normalize_nonfinite_literals(paths)
    issues = scan_integrity(paths)
    write_report(issues)
    return issues


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fix", action="store_true", help="replace literal NaN/Inf tokens with blank cells before scanning")
    args = parser.parse_args()
    issues = run(fix=args.fix)
    print(f"integrity issues: {len(issues)}")
    if issues:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
