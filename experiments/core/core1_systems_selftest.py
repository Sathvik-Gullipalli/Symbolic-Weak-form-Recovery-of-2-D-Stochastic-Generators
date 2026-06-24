from __future__ import annotations

import argparse

import numpy as np

from experiments.common import write_rows
from sde2d.systems import REGISTRY


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    rows = []
    items = list(REGISTRY.items())
    if args.quick:
        items = items[:8]
    for name, truth in items:
        sys = truth.cls()
        x = sys.simulate(dt=0.01, M=200 if args.quick else 1000, seed=123)
        drift = sys.true_drift(x[:5])
        diff = sys.true_diffusion(x[:5])
        current = sys.true_current(x[:5])
        eig = np.linalg.eigvalsh(0.5 * (diff + np.swapaxes(diff, -1, -2)))
        ok = bool(np.all(np.isfinite(x)) and np.all(np.isfinite(drift)) and np.all(np.isfinite(diff)) and np.min(eig) >= -1e-8)
        rows.append({"system": name, "dim": truth.dim, "tier": truth.tier, "verdict": truth.verdict, "n_rows": len(x), "min_diffusion_eig": float(np.min(eig)), "has_current": current is not None, "status": "PASS" if ok else "FAIL"})
    write_rows("results/scalar_reproduction/core1_systems_selftest.csv", rows)
    print(f"wrote {len(rows)} self-test rows")


if __name__ == "__main__":
    main()
