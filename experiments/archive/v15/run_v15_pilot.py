from __future__ import annotations

import argparse
import itertools
import math

import numpy as np

from experiments.v15.common import (
    BASE_SEEDS,
    METRIC_FIELDS,
    RESULTS,
    V15Config,
    config_for_grafts,
    fit_and_score,
    ffloat,
    median_by,
    ofat_configs,
    read_csv,
    save_config,
    write_csv,
)
from experiments.v15.v15_systems import CLUSTER, CONTROLS, PILOT_SYSTEMS


def run_cells(path, configs, systems, seeds, R, steps):
    rows = []
    for config in configs:
        for system in systems:
            for seed in seeds:
                row = fit_and_score(system, config, R=R, steps=steps, seed=seed)
                rows.append(row)
                write_csv(path, rows, METRIC_FIELDS)
            print("pilot", config.config_id, system, flush=True)
    return rows


def candidate_gate(rows):
    base = {(r["system"], r["seed"]): r for r in rows if r["config_id"] == "frozen"}
    candidates = []
    summary_rows = []
    for graft in ["G1", "G2", "G4", "G5", "G6", "G7", "G8", "G9"]:
        part = [r for r in rows if r["config_id"] == graft]
        cluster_ratios = []
        control_regressions = []
        fp_delta = 0
        psd_min = 1.0
        for row in part:
            b = base.get((row["system"], row["seed"]))
            if not b:
                continue
            b_d = ffloat(b["drift_l2"])
            r_d = ffloat(row["drift_l2"])
            b_t = ffloat(b["tensor_l2"])
            r_t = ffloat(row["tensor_l2"])
            if row["system"] in CLUSTER and math.isfinite(b_d) and b_d > 0 and math.isfinite(r_d):
                cluster_ratios.append(r_d / b_d)
            if row["system"] in CONTROLS:
                if math.isfinite(b_d) and math.isfinite(r_d):
                    control_regressions.append(r_d / max(b_d, 1e-12) - 1.0)
                if math.isfinite(b_t) and math.isfinite(r_t):
                    control_regressions.append(r_t / max(b_t, 1e-12) - 1.0)
            fp_delta += max(0, int(ffloat(row["n_fp"], 0)) - int(ffloat(b["n_fp"], 0)))
            psd_min = min(psd_min, ffloat(row["psd_pct"], 0.0))
        med_ratio = float(np.nanmedian(cluster_ratios)) if cluster_ratios else float("nan")
        max_reg = float(np.nanmax(control_regressions)) if control_regressions else 0.0
        keep = bool(math.isfinite(med_ratio) and med_ratio <= 0.95 and max_reg <= 0.05 and psd_min >= 0.999 and fp_delta == 0)
        summary_rows.append({"graft": graft, "cluster_drift_ratio": med_ratio, "control_max_regression": max_reg, "psd_min": psd_min, "fp_delta": fp_delta, "candidate": keep})
        if keep:
            candidates.append(graft)
    write_csv(RESULTS / "pilot_candidate_summary.csv", summary_rows, ["graft", "cluster_drift_ratio", "control_max_regression", "psd_min", "fp_delta", "candidate"])
    return candidates


def worst_cluster(rows, config_id):
    med = median_by([r for r in rows if r["config_id"] == config_id and r["system"] in CLUSTER], ["system"], "drift_l2")
    vals = [v for v in med.values() if math.isfinite(v)]
    return float(max(vals)) if vals else float("inf")


def control_max_regression(rows, config_id):
    base = median_by([r for r in rows if r["config_id"] == "frozen"], ["system"], "drift_l2")
    cur = median_by([r for r in rows if r["config_id"] == config_id], ["system"], "drift_l2")
    vals = []
    for system in CONTROLS:
        b = base.get((system,), float("nan"))
        c = cur.get((system,), float("nan"))
        if math.isfinite(b) and math.isfinite(c):
            vals.append(c / max(b, 1e-12) - 1.0)
    return float(max(vals)) if vals else 0.0


def run_stack(config, systems, seeds, R, steps, cache):
    if config.config_id in cache:
        return cache[config.config_id]
    rows = []
    for system in systems:
        for seed in seeds:
            rows.append(fit_and_score(system, config, R=R, steps=steps, seed=seed))
    cache[config.config_id] = rows
    return rows


def greedy_and_loo(candidates, base_rows, systems, seeds, R, steps):
    cache = {"frozen": [r for r in base_rows if r["config_id"] == "frozen"]}
    all_rows = list(base_rows)
    selected = []
    ladder = []
    current_worst = worst_cluster(all_rows, "frozen")
    step = 0
    while True:
        best = None
        for graft in candidates:
            if graft in selected:
                continue
            trial = tuple(selected + [graft])
            cfg = config_for_grafts(trial, "STACK_" + "_".join(trial))
            rows = run_stack(cfg, systems, seeds, R, steps, cache)
            trial_all = all_rows + rows
            w = worst_cluster(trial_all, cfg.config_id)
            reg = control_max_regression(trial_all, cfg.config_id)
            improve = (current_worst - w) / max(current_worst, 1e-12)
            if reg <= 0.05 and improve >= 0.03 and (best is None or w < best[0]):
                best = (w, graft, cfg, rows, reg, improve)
        if best is None:
            break
        current_worst, graft, cfg, rows, reg, improve = best
        selected.append(graft)
        all_rows.extend(rows)
        ladder.append({"step": step, "added_graft": graft, "stack": "+".join(selected), "worst_cluster_drift": current_worst, "control_max_regression": reg, "improvement": improve})
        step += 1
    final_stack = tuple(selected)
    write_csv(RESULTS / "graft_ladder.csv", ladder, ["step", "added_graft", "stack", "worst_cluster_drift", "control_max_regression", "improvement"])
    if not final_stack:
        write_csv(
            RESULTS / "necessity_v15.csv",
            [{"graft": "NONE", "loo_drift_delta": 0.0, "loo_tensor_delta": 0.0, "significant": False, "R": R, "stage": "pilot"}],
            ["graft", "loo_drift_delta", "loo_tensor_delta", "significant", "R", "stage"],
        )
        return ()
    loo = []
    final_cfg = config_for_grafts(final_stack, "STACK_FINAL")
    final_rows = run_stack(final_cfg, systems, seeds, R, steps, cache)
    all_rows.extend(final_rows)
    final_worst = worst_cluster(all_rows, final_cfg.config_id)
    for graft in final_stack:
        reduced = tuple(g for g in final_stack if g != graft)
        cfg = config_for_grafts(reduced, "LOO_MINUS_" + graft if reduced else "LOO_FROZEN")
        rows = run_stack(cfg, systems, seeds, R, steps, cache)
        trial_all = all_rows + rows
        w = worst_cluster(trial_all, cfg.config_id)
        delta = (w - final_worst) / max(final_worst, 1e-12)
        loo.append({"graft": graft, "loo_drift_delta": delta, "loo_tensor_delta": 0.0, "significant": bool(delta >= 0.02), "R": R, "stage": "pilot"})
    if not loo:
        loo.append({"graft": "NONE", "loo_drift_delta": 0.0, "loo_tensor_delta": 0.0, "significant": False, "R": R, "stage": "pilot"})
    write_csv(RESULTS / "necessity_v15.csv", loo, ["graft", "loo_drift_delta", "loo_tensor_delta", "significant", "R", "stage"])
    kept = tuple(row["graft"] for row in loo if row["graft"] != "NONE" and row["significant"])
    return kept or final_stack


def pilot_grid(stack, systems, seeds, R, steps):
    configs = []
    if not stack:
        configs = [config_for_grafts((), "GRID_frozen")]
    else:
        for z, conv, cap in itertools.product([1.0, 1.5, 2.0], [False, True], [1e3, 1e4]):
            configs.append(config_for_grafts(stack, f"GRID_z{z}_conv{int(conv)}_cap{int(cap)}", noise_floor_z=z, gls_converge=conv, gls_cond_cap=cap))
    rows = run_cells(RESULTS / "pilot_grid.csv", configs, systems, seeds, R, steps)
    best_cfg = None
    best_score = float("inf")
    for cfg in configs:
        w = worst_cluster(rows, cfg.config_id)
        reg = control_max_regression(rows + [r for r in read_csv(RESULTS / "pilot_ofat.csv")], cfg.config_id)
        score = w + max(0.0, reg) * 10.0
        if score < best_score:
            best_score = score
            best_cfg = cfg
    assert best_cfg is not None
    reason = "pilot_grid_worst_cluster_drift" if stack else "no_pilot_graft_survived_gate"
    save_config(best_cfg, reason=reason, profile={"pilot_R": R, "pilot_seeds": len(seeds), "pilot_steps": steps})
    return best_cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="V15 Stage 1/1.5/2 pilot screen.")
    parser.add_argument("--R", type=int, default=8)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--steps", type=int, default=900)
    args = parser.parse_args()
    seeds = BASE_SEEDS[: args.seeds]
    ofat = run_cells(RESULTS / "pilot_ofat.csv", ofat_configs(), PILOT_SYSTEMS, seeds, args.R, args.steps)
    candidates = candidate_gate(ofat)
    stack = greedy_and_loo(candidates, ofat, PILOT_SYSTEMS, seeds, args.R, args.steps)
    chosen = pilot_grid(stack, PILOT_SYSTEMS, seeds, args.R, args.steps)
    print("V15 pilot chose", chosen.config_id, chosen.stack, flush=True)


if __name__ == "__main__":
    main()
