from __future__ import annotations

import concurrent.futures as cf
import os
from typing import Iterable

from experiments.v17.common import V17Config, fit_and_score_from_data, increments_for_lags, simulate_pool_cached


def _run_cell(args):
    system, seed, R, steps, configs, oracle = args
    max_lag = max(max(c.drift_lags) for c in configs)
    _, path, _ = simulate_pool_cached(system, R, steps, seed, max_lag=max_lag)
    rows = []
    by_lag = {}
    for cfg in configs:
        if cfg.drift_lags not in by_lag:
            by_lag[cfg.drift_lags] = increments_for_lags(path, cfg.drift_lags)
        states, inc, traj_ids = by_lag[cfg.drift_lags]
        rows.append(fit_and_score_from_data(system, cfg, states, inc, traj_ids, seed=seed))
        if oracle:
            rows.append(fit_and_score_from_data(system, cfg, states, inc, traj_ids, seed=seed, oracle=True))
    return rows


def run_grid(
    systems: Iterable[str],
    seeds: Iterable[int],
    configs: list[V17Config],
    *,
    R: int,
    steps: int,
    max_workers: int | None = None,
    oracle: bool = False,
) -> list[dict]:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    cells = [(system, seed, R, steps, configs, oracle) for system in systems for seed in seeds]
    workers = max_workers or max(1, min(len(cells), (os.cpu_count() or 2) - 1))
    print(f"v17 parallel: cpu_count={os.cpu_count()} workers={workers} cells={len(cells)} configs_per_cell={len(configs)}", flush=True)
    out: list[dict] = []
    if workers <= 1:
        for cell in cells:
            out.extend(_run_cell(cell))
        return out
    try:
        with cf.ProcessPoolExecutor(max_workers=workers) as ex:
            for rows in ex.map(_run_cell, cells, chunksize=1):
                out.extend(rows)
    except PermissionError as exc:
        print(f"v17 parallel: ProcessPool unavailable ({exc}); falling back to serial execution.", flush=True)
        for cell in cells:
            out.extend(_run_cell(cell))
    return out
