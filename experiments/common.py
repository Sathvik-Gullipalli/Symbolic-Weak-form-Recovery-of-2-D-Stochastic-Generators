from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def append_rows(path: str | Path, rows: list[dict]) -> None:
    if not rows:
        return
    p = ROOT / path
    p.parent.mkdir(parents=True, exist_ok=True)
    exists = p.exists()
    with p.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def write_rows(path: str | Path, rows: list[dict]) -> None:
    if not rows:
        return
    p = ROOT / path
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
