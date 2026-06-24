from __future__ import annotations

import argparse

from experiments.v17.common import RESULTS, V17_FIELDS, config_for_grafts_v17, load_config, write_csv
from experiments.v17.parallel import run_grid
from experiments.v17.v17_systems import DEEP_SYSTEMS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--R", type=int, default=32)
    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--systems", nargs="*", default=None)
    args = parser.parse_args()
    chosen = load_config()
    if chosen.config_id == "frozen":
        write_csv(RESULTS / "deep_confirm.csv", [], V17_FIELDS)
        print("V17 no surviving config; skipped deep run.", flush=True)
        return
    seeds = [17201 + i for i in range(args.seeds)]
    systems = args.systems or DEEP_SYSTEMS
    rows = run_grid(systems, seeds, [config_for_grafts_v17(config_id="frozen", **(chosen.overrides or {})), chosen], R=args.R, steps=args.steps, max_workers=args.workers)
    write_csv(RESULTS / "deep_confirm.csv", rows, V17_FIELDS)
    print(f"wrote {RESULTS / 'deep_confirm.csv'} rows={len(rows)}", flush=True)


if __name__ == "__main__":
    main()
