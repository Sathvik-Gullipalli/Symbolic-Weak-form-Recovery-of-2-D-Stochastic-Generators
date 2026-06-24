from __future__ import annotations

import argparse
import itertools
import math

import numpy as np

from experiments.v17.common import RESULTS, V17_FIELDS, config_for_grafts_v17, ffloat, median_by, save_config, write_csv
from experiments.v17.parallel import run_grid
from experiments.v17.v17_systems import CONTROLS, TARGETS


def factorial_configs(include_ng8: bool = False, quick: bool = False):
    if quick:
        fast = {"regressor": "stlsq", "n_centers": 14, "local_poly_order": 1, "regression_kw": {"threshold": 0.04}}
        configs = [
            config_for_grafts_v17(config_id="frozen", **fast),
            config_for_grafts_v17(coord="lamperti", config_id="SMOKE_coord", **fast),
            config_for_grafts_v17(drift_lags=(1, 2, 4), config_id="SMOKE_lags", **fast),
            config_for_grafts_v17(moment="milstein", config_id="SMOKE_moment", **fast),
            config_for_grafts_v17(rank="auto", config_id="SMOKE_rank", **fast),
            config_for_grafts_v17(domain="positive_log", config_id="SMOKE_domain", **fast),
            config_for_grafts_v17(selection="noise_floor", config_id="SMOKE_selection", **fast),
            config_for_grafts_v17(coverage_mode="reweight", config_id="SMOKE_coverage", **fast),
        ]
        if include_ng8:
            configs.append(config_for_grafts_v17(library_atoms="poly+trig", config_id="SMOKE_ng8", **fast))
        return configs
    coords = ["none", "lamperti"]
    lags = [(1,), (1, 2, 4)] if quick else [(1,), (1, 2, 4), (1, 2, 4, 8)]
    moments = ["euler", "milstein"]
    ranks = ["full", "auto"]
    domains = ["euclidean", "positive_log"]
    sels = ["relative", "noise_floor"]
    libs = ["poly"]
    if include_ng8:
        libs += ["poly+trig", "poly+rational", "poly+rbf"]
    configs = [config_for_grafts_v17(config_id="frozen")]
    for coord, lag, moment, rank, domain, sel, lib in itertools.product(coords, lags, moments, ranks, domains, sels, libs):
        if (coord, lag, moment, rank, domain, sel, lib) == ("none", (1,), "euler", "full", "euclidean", "relative", "poly"):
            continue
        configs.append(
            config_for_grafts_v17(
                coord=coord,
                drift_lags=lag,
                moment=moment,
                rank=rank,
                domain=domain,
                selection=sel,
                library_atoms=lib,
            )
        )
    return configs


def control_regression(rows: list[dict], config_id: str) -> float:
    base = median_by([r for r in rows if r["config_id"] == "frozen" and r["system"] in CONTROLS], "system")
    cur = median_by([r for r in rows if r["config_id"] == config_id and r["system"] in CONTROLS], "system")
    vals = []
    for system, b in base.items():
        c = cur.get(system, float("nan"))
        if math.isfinite(b) and math.isfinite(c):
            vals.append(c / max(b, 1e-12) - 1.0)
    return float(max(vals)) if vals else 0.0


def select_config(rows: list[dict]) -> tuple:
    base = median_by([r for r in rows if r["config_id"] == "frozen" and r["system"] in TARGETS], "system")
    cfgs = sorted({r["config_id"] for r in rows if r["config_id"] != "frozen"})
    summary = []
    best = None
    for cfg in cfgs:
        cur = median_by([r for r in rows if r["config_id"] == cfg and r["system"] in TARGETS], "system")
        ratios = [cur[s] / max(base[s], 1e-12) for s in base if s in cur and math.isfinite(cur[s])]
        target_ratio = float(np.nanmedian(ratios)) if ratios else float("inf")
        reg = control_regression(rows, cfg)
        psd_min = min(ffloat(r["psd_pct"], 0.0) for r in rows if r["config_id"] == cfg)
        fp_delta = 0
        for r in rows:
            if r["config_id"] != cfg:
                continue
            matches = [b for b in rows if b["config_id"] == "frozen" and b["system"] == r["system"] and b["seed"] == r["seed"]]
            if matches:
                fp_delta += max(0, int(ffloat(r["n_fp"], 0)) - int(ffloat(matches[0]["n_fp"], 0)))
        keep = bool(target_ratio <= 0.95 and reg <= 0.05 and psd_min >= 0.999 and fp_delta == 0)
        row = {"config_id": cfg, "target_ratio": target_ratio, "control_regression": reg, "psd_min": psd_min, "fp_delta": fp_delta, "candidate": keep}
        summary.append(row)
        if keep and (best is None or target_ratio < best["target_ratio"]):
            best = row
    write_csv(RESULTS / "candidate_summary.csv", summary, ["config_id", "target_ratio", "control_regression", "psd_min", "fp_delta", "candidate"])
    if best is None:
        return config_for_grafts_v17(config_id="frozen"), summary
    template = next(r for r in rows if r["config_id"] == best["config_id"])
    overrides = {}
    if str(best["config_id"]).startswith("SMOKE_"):
        overrides = {"regressor": "stlsq", "n_centers": 14, "local_poly_order": 1, "regression_kw": {"threshold": 0.04}}
    cfg = config_for_grafts_v17(
        coord=template["coord"],
        drift_lags=tuple(int(v) for v in str(template["lags"]).split("-")),
        moment=template["moment"],
        rank=template["rank"],
        domain=template["domain"],
        selection=template["selection"],
        library_atoms=template["library_atoms"],
        coverage_mode=template["coverage_mode"],
        config_id=best["config_id"],
        **overrides,
    )
    return cfg, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--R", type=int, default=16)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--enable-ng8", action="store_true")
    args = parser.parse_args()
    configs = factorial_configs(include_ng8=args.enable_ng8, quick=args.quick)
    seeds = [17101 + i for i in range(args.seeds)]
    rows = run_grid(TARGETS + CONTROLS, seeds, configs, R=args.R, steps=args.steps, max_workers=args.workers)
    write_csv(RESULTS / "factorial.csv", rows, V17_FIELDS)
    chosen, summary = select_config(rows)
    if chosen.config_id == "frozen":
        ladder = [{"step": 0, "added_factor": "NONE", "config_id": "frozen", "target_ratio": 1.0, "control_regression": 0.0}]
        necessity = [{"factor": "NONE", "loo_delta": 0.0, "significant": False, "R": args.R, "stage": "pilot"}]
        reason = "no_factor_survived_gate"
    else:
        ladder = [{"step": 1, "added_factor": chosen.stack, "config_id": chosen.config_id, "target_ratio": min(s["target_ratio"] for s in summary if s["config_id"] == chosen.config_id), "control_regression": control_regression(rows, chosen.config_id)}]
        necessity = [{"factor": chosen.stack, "loo_delta": 1.0 - ladder[0]["target_ratio"], "significant": True, "R": args.R, "stage": "pilot"}]
        reason = "pilot_factor_survived_gate"
    write_csv(RESULTS / "graft_ladder.csv", ladder, ["step", "added_factor", "config_id", "target_ratio", "control_regression"])
    write_csv(RESULTS / "necessity_v17.csv", necessity, ["factor", "loo_delta", "significant", "R", "stage"])
    save_config(chosen, reason=reason, profile={"R": args.R, "steps": args.steps, "seeds": args.seeds, "configs": len(configs)})
    print(f"V17 chose {chosen.config_id} {chosen.stack}", flush=True)


if __name__ == "__main__":
    main()
