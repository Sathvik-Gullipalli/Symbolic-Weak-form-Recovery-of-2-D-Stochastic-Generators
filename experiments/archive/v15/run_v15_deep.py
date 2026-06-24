from __future__ import annotations

import argparse
import math

import matplotlib.pyplot as plt
import numpy as np

from experiments.v15.common import (
    DEEP_SEEDS,
    FIGURES,
    METRIC_FIELDS,
    RESULTS,
    V15Config,
    bootstrap_ci,
    fit_and_score,
    ffloat,
    load_config,
    read_csv,
    write_csv,
)
from experiments.v15.v15_systems import CLUSTER, DEEP_SYSTEMS

DEEP_FIELDS = [
    "system",
    "frozen_drift_median",
    "frozen_drift_ci_low",
    "frozen_drift_ci_high",
    "v15_drift_median",
    "v15_drift_ci_low",
    "v15_drift_ci_high",
    "frozen_tensor_median",
    "v15_tensor_median",
    "frozen_a12_cos_median",
    "v15_a12_cos_median",
    "frozen_psd_median",
    "v15_psd_median",
    "frozen_fp_median",
    "v15_fp_median",
    "drift_improvement",
    "significant",
    "R",
    "seeds",
    "steps",
]


def summarize(system, frozen_rows, v15_rows, R, seeds, steps):
    fd, flo, fhi = bootstrap_ci([ffloat(r["drift_l2"]) for r in frozen_rows])
    vd, vlo, vhi = bootstrap_ci([ffloat(r["drift_l2"]) for r in v15_rows])
    ft, _, _ = bootstrap_ci([ffloat(r["tensor_l2"]) for r in frozen_rows])
    vt, _, _ = bootstrap_ci([ffloat(r["tensor_l2"]) for r in v15_rows])
    fcos, _, _ = bootstrap_ci([ffloat(r["a12_cos"]) for r in frozen_rows])
    vcos, _, _ = bootstrap_ci([ffloat(r["a12_cos"]) for r in v15_rows])
    fpsd, _, _ = bootstrap_ci([ffloat(r["psd_pct"]) for r in frozen_rows])
    vpsd, _, _ = bootstrap_ci([ffloat(r["psd_pct"]) for r in v15_rows])
    ffp, _, _ = bootstrap_ci([ffloat(r["n_fp"]) for r in frozen_rows])
    vfp, _, _ = bootstrap_ci([ffloat(r["n_fp"]) for r in v15_rows])
    improvement = (fd - vd) / max(fd, 1e-12) if math.isfinite(fd) and math.isfinite(vd) else float("nan")
    significant = bool(math.isfinite(flo) and math.isfinite(vhi) and vhi < flo)
    return {
        "system": system,
        "frozen_drift_median": fd,
        "frozen_drift_ci_low": flo,
        "frozen_drift_ci_high": fhi,
        "v15_drift_median": vd,
        "v15_drift_ci_low": vlo,
        "v15_drift_ci_high": vhi,
        "frozen_tensor_median": ft,
        "v15_tensor_median": vt,
        "frozen_a12_cos_median": fcos,
        "v15_a12_cos_median": vcos,
        "frozen_psd_median": fpsd,
        "v15_psd_median": vpsd,
        "frozen_fp_median": ffp,
        "v15_fp_median": vfp,
        "drift_improvement": improvement,
        "significant": significant,
        "R": R,
        "seeds": seeds,
        "steps": steps,
    }


def render_figures(rows, *, no_survivor: bool = False):
    FIGURES.mkdir(parents=True, exist_ok=True)
    cluster = [r for r in rows if r["system"] in CLUSTER]
    if not cluster:
        return
    labels = [r["system"] for r in cluster]
    frozen = [ffloat(r["frozen_drift_median"]) for r in cluster]
    v15 = [ffloat(r["v15_drift_median"]) for r in cluster]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar(x - 0.18, frozen, width=0.36, label="frozen")
    ax.bar(x + 0.18, v15, width=0.36, label="V15")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("drift relative L2")
    title = "V15 no-survivor baseline on target cluster" if no_survivor else "V15 deep confirmation on target cluster"
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "deep_cluster_drift.pdf")
    fig.savefig(FIGURES / "deep_cluster_drift.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="V15 Stage 3 deep confirm.")
    parser.add_argument("--R", type=int, default=32)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--steps", type=int, default=900)
    parser.add_argument("--systems", nargs="*", default=None)
    args = parser.parse_args()
    chosen = load_config()
    frozen = V15Config("frozen", (), {})
    no_survivor = not chosen.grafts
    systems = args.systems or DEEP_SYSTEMS
    seeds = DEEP_SEEDS[: args.seeds]
    cell_rows = []
    summaries = []
    for system in systems:
        f_rows = [fit_and_score(system, frozen, R=args.R, steps=args.steps, seed=seed, grid_n=11) for seed in seeds]
        if no_survivor:
            v_rows = [{**row, "config_id": chosen.config_id, "graft": chosen.config_id, "stack": chosen.stack} for row in f_rows]
        else:
            v_rows = [fit_and_score(system, chosen, R=args.R, steps=args.steps, seed=seed, grid_n=11) for seed in seeds]
        cell_rows.extend(f_rows)
        cell_rows.extend(v_rows)
        write_csv(RESULTS / "deep_cells.csv", cell_rows, METRIC_FIELDS)
        summaries.append(summarize(system, f_rows, v_rows, args.R, len(seeds), args.steps))
        write_csv(RESULTS / "deep_confirm.csv", summaries, DEEP_FIELDS)
        print("deep", system, flush=True)
    render_figures(summaries, no_survivor=no_survivor)
    necessity = read_csv(RESULTS / "necessity_v15.csv")
    for row in necessity:
        row["R"] = args.R
        row["stage"] = "no-pilot-survivor" if no_survivor else "deep-confirmed-from-single-winner"
    write_csv(RESULTS / "necessity_v15.csv", necessity, ["graft", "loo_drift_delta", "loo_tensor_delta", "significant", "R", "stage"])


if __name__ == "__main__":
    main()
