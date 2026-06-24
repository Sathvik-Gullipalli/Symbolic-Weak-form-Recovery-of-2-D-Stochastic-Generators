from __future__ import annotations

import argparse
import math

import numpy as np

from experiments.v15.common import BASE_SEEDS, RESULTS, V15Config, fit_and_score, write_csv
from experiments.v15.v15_systems import CLUSTER, CONTROLS

FIELDS = [
    "system",
    "drift_l2_frozen",
    "drift_l2_oracle",
    "tensor_l2_frozen",
    "tensor_l2_oracle",
    "oracle_headroom_drift",
    "oracle_headroom_tensor",
    "classification",
    "R",
    "seeds",
    "steps",
]


def classify(frozen_drift: float, oracle_drift: float, frozen_tensor: float, oracle_tensor: float) -> str:
    tags = []
    if math.isfinite(oracle_drift) and oracle_drift <= 0.75 * max(frozen_drift, 1e-12):
        tags.append("selection-curable")
    if math.isfinite(oracle_tensor) and frozen_tensor >= 0.8 and oracle_tensor <= 0.75 * max(frozen_tensor, 1e-12):
        tags.append("threshold-artifact")
    if not tags and math.isfinite(oracle_drift) and oracle_drift >= 0.8 * max(frozen_drift, 1e-12):
        tags.append("snr-irreducible")
    return "+".join(tags) if tags else "mixed"


def main() -> None:
    parser = argparse.ArgumentParser(description="V15 Stage 0 oracle-headroom diagnosis.")
    parser.add_argument("--R", type=int, default=8)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--steps", type=int, default=900)
    args = parser.parse_args()
    systems = CLUSTER + CONTROLS
    seeds = BASE_SEEDS[: args.seeds]
    rows = []
    frozen = V15Config("frozen", (), {})
    for system in systems:
        f_rows = [fit_and_score(system, frozen, R=args.R, steps=args.steps, seed=seed) for seed in seeds]
        o_rows = [fit_and_score(system, frozen, R=args.R, steps=args.steps, seed=seed, oracle=True) for seed in seeds]
        fd = float(np.nanmedian([r["drift_l2"] for r in f_rows]))
        od = float(np.nanmedian([r["drift_l2"] for r in o_rows]))
        ft = float(np.nanmedian([r["tensor_l2"] for r in f_rows]))
        ot = float(np.nanmedian([r["tensor_l2"] for r in o_rows]))
        rows.append(
            {
                "system": system,
                "drift_l2_frozen": fd,
                "drift_l2_oracle": od,
                "tensor_l2_frozen": ft,
                "tensor_l2_oracle": ot,
                "oracle_headroom_drift": fd - od if math.isfinite(fd) and math.isfinite(od) else float("nan"),
                "oracle_headroom_tensor": ft - ot if math.isfinite(ft) and math.isfinite(ot) else float("nan"),
                "classification": classify(fd, od, ft, ot),
                "R": args.R,
                "seeds": len(seeds),
                "steps": args.steps,
            }
        )
        write_csv(RESULTS / "diagnosis.csv", rows, FIELDS)
        print("diagnosed", system, rows[-1]["classification"], flush=True)


if __name__ == "__main__":
    main()

