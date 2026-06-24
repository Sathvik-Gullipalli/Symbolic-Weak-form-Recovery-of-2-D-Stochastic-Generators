import sys
import numpy as np
from pathlib import Path

from experiments.v16.common import (
    config_for_grafts_v16,
    fit_and_score,
    write_csv,
    RESULTS,
)
from experiments.v16.v16_systems import PILOT_SYSTEMS, CLUSTER, dt_for

def main():
    print("V16 Diagnosis: Running at realistic budgets")
    rows = []
    
    seeds = [15001, 15002, 15003, 15004, 15005]
    budgets = [(16, 4000), (32, 8000)]
    
    config_frozen = config_for_grafts_v16()
    
    out_csv = RESULTS / "diagnosis.csv"
    if out_csv.exists():
        out_csv.unlink()
        
    for system in PILOT_SYSTEMS:
        print(f"System: {system}")
        sys_rows = []
        for R, steps in budgets:
            print(f"  Budget: R={R}, steps={steps}")
            for seed in seeds:
                from experiments.v16.common import simulate_pool
                states, increments, traj_ids = simulate_pool(system, R, steps, seed, data_transform=config_frozen.data_transform)
                precomputed_data = (states, increments, traj_ids)
                
                frozen_res, proj = fit_and_score(system, config_frozen, R=R, steps=steps, seed=seed, oracle=False, precomputed_data=precomputed_data, return_projection=True)
                oracle_res = fit_and_score(system, config_frozen, R=R, steps=steps, seed=seed, oracle=True, precomputed_data=precomputed_data, precomputed_projection=proj)
                sys_rows.append({
                    "system": system,
                    "R": R,
                    "steps": steps,
                    "seed": seed,
                    "drift_rel_frozen": frozen_res["drift_l2"],
                    "drift_rel_oracle": oracle_res["drift_l2"],
                    "drift_abs_frozen": frozen_res["drift_abs_l2"],
                    "drift_abs_oracle": oracle_res["drift_abs_l2"],
                    "tensor_rel_frozen": frozen_res["tensor_l2"],
                    "tensor_rel_oracle": oracle_res["tensor_l2"],
                    "coef_maxrelerr_frozen": frozen_res["coef_max_rel_err"],
                    "coef_maxrelerr_oracle": oracle_res["coef_max_rel_err"],
                })
                
        for R, steps in budgets:
            budget_rows = [r for r in sys_rows if r["R"] == R and r["steps"] == steps]
            med_frozen_rel = float(np.median([r["drift_rel_frozen"] for r in budget_rows]))
            med_oracle_rel = float(np.median([r["drift_rel_oracle"] for r in budget_rows]))
            med_frozen_abs = float(np.median([r["drift_abs_frozen"] for r in budget_rows]))
            med_oracle_abs = float(np.median([r["drift_abs_oracle"] for r in budget_rows]))
            med_tensor_frozen = float(np.median([r["tensor_rel_frozen"] for r in budget_rows]))
            med_tensor_oracle = float(np.median([r["tensor_rel_oracle"] for r in budget_rows]))
            med_coef_frozen = float(np.median([r["coef_maxrelerr_frozen"] for r in budget_rows]))
            
            classification = ""
            if R == 32:
                if med_frozen_rel < 0.2:
                    classification = "pass"
                elif med_oracle_rel > 0.5:
                    classification = "genuinely-irreducible"
                elif med_tensor_oracle < 0.1 and med_tensor_frozen > 0.3:
                    classification = "selection-curable"
                elif med_frozen_abs < 0.05 and med_frozen_rel > 1.0:
                    classification = "metric-artifact"
                else:
                    b16 = [r for r in sys_rows if r["R"] == 16 and r["steps"] == 4000]
                    med_frozen_rel_16 = float(np.median([r["drift_rel_frozen"] for r in b16]))
                    if med_frozen_rel < med_frozen_rel_16 * 0.7:
                        classification = "budget-limited"
                    else:
                        classification = "genuinely-irreducible"
                        
            row = {
                "system": system,
                "R": R,
                "steps": steps,
                "drift_rel_frozen": med_frozen_rel,
                "drift_rel_oracle": med_oracle_rel,
                "drift_abs_frozen": med_frozen_abs,
                "drift_abs_oracle": med_oracle_abs,
                "tensor_rel_frozen": med_tensor_frozen,
                "tensor_rel_oracle": med_tensor_oracle,
                "coef_maxrelerr_frozen": med_coef_frozen,
                "classification": classification,
            }
            from experiments.v16.common import append_csv
            append_csv(
                out_csv,
                [row],
                [
                    "system", "R", "steps", 
                    "drift_rel_frozen", "drift_rel_oracle", 
                    "drift_abs_frozen", "drift_abs_oracle",
                    "tensor_rel_frozen", "tensor_rel_oracle", 
                    "coef_maxrelerr_frozen", "classification"
                ]
            )
            rows.append(row)

    print("Diagnosis complete.")

if __name__ == "__main__":
    main()
