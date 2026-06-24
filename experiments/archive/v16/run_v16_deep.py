import sys
from pathlib import Path
import json

from experiments.v16.common import (
    config_for_grafts_v16,
    fit_and_score,
    write_csv,
    RESULTS,
    CHOSEN_CONFIG,
    DEEP_SEEDS,
)
from experiments.v16.v16_systems import DEEP_SYSTEMS

def load_chosen_config():
    if not CHOSEN_CONFIG.exists():
        return config_for_grafts_v16()
    data = json.loads(CHOSEN_CONFIG.read_text())
    return config_for_grafts_v16(config_id=data.get("config_id"), **data.get("overrides", {}))

def main():
    print("Stage 3: Deep Confirm")
    rows = []
    R, steps = 32, 8000
    seeds = DEEP_SEEDS
    
    config_frozen = config_for_grafts_v16()
    config_v16 = load_chosen_config()
    
    out_csv = RESULTS / "deep_confirm.csv"
    if out_csv.exists():
        out_csv.unlink()
        
    for system in DEEP_SYSTEMS:
        print(f"Deep system: {system}")
        for seed in seeds:
            from experiments.v16.common import simulate_pool, append_csv
            states, increments, traj_ids = simulate_pool(system, R, steps, seed, data_transform=config_frozen.data_transform, lags=(1, 2, 4, 8))
            precomputed_data = (states, increments, traj_ids)
            
            res_frozen, proj = fit_and_score(system, config_frozen, R=R, steps=steps, seed=seed, precomputed_data=precomputed_data, return_projection=True)
            res_frozen["stack"] = "frozen"
            append_csv(out_csv, [res_frozen], ["system", "seed", "config_id", "stack", "R", "steps", "dt", "drift_l2", "tensor_l2", "drift_abs_l2", "coef_max_rel_err", "composite_drift", "a12_cos", "psd_pct", "n_fp", "status", "notes", "b1_l2", "b2_l2"])
            rows.append(res_frozen)
            
            res_v16 = fit_and_score(system, config_v16, R=R, steps=steps, seed=seed, precomputed_data=precomputed_data, precomputed_projection=proj)
            res_v16["stack"] = "v16"
            append_csv(out_csv, [res_v16], ["system", "seed", "config_id", "stack", "R", "steps", "dt", "drift_l2", "tensor_l2", "drift_abs_l2", "coef_max_rel_err", "composite_drift", "a12_cos", "psd_pct", "n_fp", "status", "notes", "b1_l2", "b2_l2"])
            rows.append(res_v16)

    print("Deep confirm completed.")

if __name__ == "__main__":
    main()
