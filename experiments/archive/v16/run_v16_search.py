import sys
from pathlib import Path
import json

from experiments.v16.common import (
    config_for_grafts_v16,
    fit_and_score,
    write_csv,
    RESULTS,
    CHOSEN_CONFIG,
)
from experiments.v16.v16_systems import CLUSTER

def run_stage_1():
    print("Stage 1: Full factorial on CLUSTER")
    rows = []
    seeds = [15001, 15002, 15003]
    R, steps = 16, 4000
    
    transforms = ["none"]
    lags_opts = [(1,), (1, 2, 4), (1, 2, 4, 8)]
    moments = ["euler"]
    gls_modes = ["diagonal", "full_tensor"]
    selections = ["relative", "noise_floor"]
    
    total = len(CLUSTER) * len(transforms) * len(lags_opts) * len(moments) * len(gls_modes) * len(selections) * len(seeds)
    print(f"Total configurations to run: {total}")
    
    out_csv = RESULTS / "factorial.csv"
    if out_csv.exists():
        out_csv.unlink()
        
    count = 0
    for system in CLUSTER:
        for seed in seeds:
            from experiments.v16.common import simulate_pool, append_csv
            config_base = config_for_grafts_v16()
            states, increments, traj_ids = simulate_pool(system, R, steps, seed, data_transform=config_base.data_transform, lags=(1, 2, 4, 8))
            precomputed_data = (states, increments, traj_ids)
            
            # Precompute projection using base config
            _, proj = fit_and_score(system, config_base, R=R, steps=steps, seed=seed, precomputed_data=precomputed_data, return_projection=True)
            
            for t in transforms:
                for l in lags_opts:
                    for m in moments:
                        for g in gls_modes:
                            for s in selections:
                                config = config_for_grafts_v16(
                                    coord_transform=t,
                                    drift_lags=l,
                                    moment_order=m,
                                    gls_mode=g,
                                    selection=s
                                )
                                count += 1
                                print(f"[{count}/{total}] Running {system} with {config.stack} seed={seed}")
                                res = fit_and_score(system, config, R=R, steps=steps, seed=seed, precomputed_data=precomputed_data, precomputed_projection=proj)
                                append_csv(out_csv, [res], ["system", "seed", "config_id", "stack", "R", "steps", "dt", "drift_l2", "tensor_l2", "drift_abs_l2", "coef_max_rel_err", "composite_drift", "a12_cos", "psd_pct", "n_fp", "status", "notes", "b1_l2", "b2_l2"])
                                rows.append(res)
    
    return rows

def run_stage_2(factorial_rows):
    print("Stage 2: Greedy forward select")
    import numpy as np
    
    configs = {}
    for row in factorial_rows:
        if row["stack"] not in configs:
            configs[row["stack"]] = []
        configs[row["stack"]].append(row)
    
    best_stack = "frozen"
    best_worst_drift = float('inf')
    best_config_id = "V16_frozen"
    
    for stack, items in configs.items():
        sys_drifts = {}
        for item in items:
            sys = item["system"]
            if sys not in sys_drifts:
                sys_drifts[sys] = []
            sys_drifts[sys].append(item["composite_drift"])
        
        worst_drift = max([np.median(vals) for vals in sys_drifts.values()])
        if worst_drift < best_worst_drift:
            best_worst_drift = worst_drift
            best_stack = stack
            best_config_id = items[0]["config_id"]
    
    print(f"Best stack: {best_stack} with worst drift: {best_worst_drift}")
    
    parts = best_stack.split("+")
    kwargs = {}
    for part in parts:
        if part.startswith("coord="): kwargs["coord_transform"] = part.split("=")[1]
        elif part.startswith("lags="): kwargs["drift_lags"] = eval(part.split("=")[1])
        elif part.startswith("moment="): kwargs["moment_order"] = part.split("=")[1]
        elif part.startswith("gls="): kwargs["gls_mode"] = part.split("=")[1]
        elif part.startswith("sel="): kwargs["selection"] = part.split("=")[1]
        
    config = config_for_grafts_v16(**kwargs)
    
    CHOSEN_CONFIG.write_text(json.dumps({
        "config_id": config.config_id,
        "grafts": list(config.grafts),
        "overrides": config.overrides,
        "data_transform": config.data_transform,
        "reason": "best worst-case cluster drift",
        "profile": {}
    }, indent=2))
    
    write_csv(
        RESULTS / "graft_ladder.csv",
        [{"step": 1, "config_id": config.config_id, "composite_drift": best_worst_drift, "added_factor": best_stack}],
        ["step", "config_id", "composite_drift", "added_factor"]
    )
    
    write_csv(
        RESULTS / "necessity_v16.csv",
        [],
        ["config_id", "dropped_factor", "composite_drift"]
    )

def main():
    rows = run_stage_1()
    run_stage_2(rows)

if __name__ == "__main__":
    main()
