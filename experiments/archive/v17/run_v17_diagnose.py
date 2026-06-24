from __future__ import annotations

import argparse

from experiments.v17.common import RESULTS, V17_FIELDS, config_for_grafts_v17, write_csv
from experiments.v17.parallel import run_grid
from experiments.v17.v17_systems import CONTROLS, TARGETS


def classify(frozen: dict, oracle: dict) -> str:
    f_comp = float(frozen["composite"])
    o_comp = float(oracle["composite"])
    f_t = float(frozen["tensor_l2"])
    o_t = float(oracle["tensor_l2"])
    if o_t < 0.10 and f_t > 0.30:
        return "tensor-selection-curable"
    if o_comp < 0.75 * f_comp:
        return "selection-curable"
    if o_comp > 1.10 * f_comp:
        return "oracle-worse"
    return "information-limited"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--R", type=int, default=32)
    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()
    seeds = [17001 + i for i in range(args.seeds)]
    cfg = config_for_grafts_v17(config_id="frozen")
    rows = run_grid(TARGETS + CONTROLS, seeds, [cfg], R=args.R, steps=args.steps, max_workers=args.workers, oracle=True)
    by = {}
    for row in rows:
        key = (row["system"], row["seed"])
        by.setdefault(key, {})["oracle" if row["config_id"].endswith("_oracle") else "frozen"] = row
    out = []
    for (system, seed), pair in sorted(by.items()):
        if "frozen" not in pair or "oracle" not in pair:
            continue
        f, o = pair["frozen"], pair["oracle"]
        row = dict(f)
        row["notes"] = classify(f, o)
        row["oracle_composite"] = o["composite"]
        row["oracle_tensor_l2"] = o["tensor_l2"]
        out.append(row)
    fields = V17_FIELDS + ["oracle_composite", "oracle_tensor_l2"]
    write_csv(RESULTS / "diagnosis.csv", out, fields)
    print(f"wrote {RESULTS / 'diagnosis.csv'} rows={len(out)}", flush=True)


if __name__ == "__main__":
    main()

