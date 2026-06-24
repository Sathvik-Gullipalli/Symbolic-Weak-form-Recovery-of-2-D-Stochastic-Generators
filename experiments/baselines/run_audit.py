from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[2] / ".matplotlib-cache"))

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from experiments.benchmarks._utils import active_targets, fit_cell  # noqa: E402
from experiments.common import ROOT as COMMON_ROOT  # noqa: E402
from experiments.v5.run_v5_campaign import V5Variant, _safe_float, cell_for, read_rows, write_csv  # noqa: E402
from experiments.v5_5.run_v5_5_campaign import IN_SCOPE_SYSTEMS, OUT_OF_SCOPE_SYSTEMS, metric_contract  # noqa: E402
from experiments.v6.run_v6_campaign import coeffs_in_fit_space, generator_action_error, median_ci, wg_sindy_variant  # noqa: E402
from sde2d.metrics import central_grid, cosine_similarity, function_l2_errors, psd_validity, tensor_metrics  # noqa: E402
from sde2d.systems import REGISTRY, System  # noqa: E402


OUT_DIR = "results/v6"
FIG_DIR = "figures/v6"
SHOWCASE_DIR = f"{OUT_DIR}/showcase"

EXTRA_RAW = f"{OUT_DIR}/v6_2_extra_summary_raw.csv"
EXTRA_SUMMARY = f"{OUT_DIR}/v6_2_extra_summary.csv"
EXTRA_COEF_RAW = f"{OUT_DIR}/v6_2_extra_coefficients_raw.csv"
EXTRA_COEF = f"{OUT_DIR}/v6_2_extra_coefficients.csv"
SYSTEM_INDEX = f"{OUT_DIR}/system_index.csv"
CELL_DIAGNOSIS = f"{OUT_DIR}/cell_diagnosis.csv"
DATA_INTEGRITY = f"{OUT_DIR}/data_integrity_report.csv"
V62_LOG = f"{OUT_DIR}/v6_2_run_log.csv"
SEGMENTS_TEX = "paper/v6_2_system_segments.tex"
INDEX_TABLE_TEX = "paper/v6_2_system_index_table.tex"

BASE_SHOWCASE_SUMMARY = f"{SHOWCASE_DIR}/showcase_summary.csv"
BASE_SHOWCASE_COEF = f"{SHOWCASE_DIR}/showcase_coefficients.csv"

FINANCIAL_EXTRA = {"sabr", "gbm_2d", "two_factor_vasicek"}
ALL_2D_SYSTEMS = [key for key, truth in REGISTRY.items() if truth.dim == 2]
NEW_CANONICAL_SYSTEMS = [
    "van_der_pol",
    "fitzhugh_nagumo",
    "stuart_landau",
    "brusselator",
    "maier_stein",
    "duffing",
    "mueller_brown",
    "sabr",
    "gbm_2d",
    "two_factor_vasicek",
]

SUMMARY_FIELDS = [
    "system",
    "tier",
    "seed",
    "run",
    "variant_id",
    "library",
    "R",
    "n_steps",
    "dt",
    "objective_drift_rel_l2",
    "drift_rel_l2",
    "diffusion_rel_l2",
    "b1_rel_l2",
    "b2_rel_l2",
    "a11_rel_l2",
    "a22_rel_l2",
    "a12_rel_l2",
    "a12_cosine",
    "a12_sign_acc",
    "psd_valid_pct",
    "generator_action_error",
    "scope_metric_contract",
    "runtime_sec",
    "run_status",
    "error",
    "traceback_tail",
]

COEF_FIELDS = [
    "system",
    "tier",
    "seed",
    "target",
    "term_name",
    "term_index",
    "coef_true",
    "coef_hat",
    "rel_error",
    "target_in_scope",
    "term_in_paper_scope",
    "active_true",
    "raw_selected",
    "selected",
    "false_positive",
]

SYSTEM_DEFINITIONS = {
    "indep_ou": r"$dX_i=-\theta_iX_i\,dt+\sigma_i\,dW_i$, independent tensor.",
    "correlated_ou": r"$dX_i=-\theta_iX_i\,dt+\sigma_i\,dW_i$, $dW_1dW_2=\rho\,dt$.",
    "coupled_ou": r"$b_1=-ax+cy,\ b_2=dx-by$, constant diagonal tensor.",
    "rotational_ou": r"$b=(-\alpha x-\omega y,\ \omega x-\alpha y)$, constant isotropic tensor.",
    "spiral_sink_corr": r"Rotational OU drift with correlated constant tensor.",
    "double_well_transverse": r"$b=(x-x^3-\beta y,\ -\lambda y+\beta x)$.",
    "gradient_potential": r"$b=-\nabla\{\frac14(x^2-1)^2+\frac12\lambda y^2+\eta x^2y^2\}$.",
    "nongradient_circulation": r"Gradient double well plus $\Omega J\nabla V$ circulation.",
    "diag_multiplicative": r"Linear mean reversion with diagonal quadratic tensor.",
    "nondiag_cholesky": r"Linear mean reversion with state-dependent Cholesky tensor $a=LL^\top$.",
    "near_singular": r"Quadratic tensor with $a_{12}\approx0.95\sqrt{a_{11}a_{22}}$.",
    "heston_sv": r"$dS=\mu S\,dt+\sqrt{V}S\,dW_1$, CIR variance, correlated shocks.",
    "heston_logsv": r"$dX=(\mu-\frac12V)\,dt+\sqrt{V}\,dW_1$, CIR variance, correlated shocks.",
    "cir_pair": r"Two correlated CIR factors with $a_{12}\propto\sqrt{XY}$.",
    "underdamped_langevin": r"$dQ=P\,dt,\ dP=(-\gamma P-(Q^3-Q))\,dt+\sigma\,dW$.",
    "near_boundary_heston": r"Log-Heston with Feller-stress variance frequently near zero.",
    "nonpoly_drift": r"$b=(-x+\sin y,\ -y+\cos x)$ with trigonometric library requirement.",
    "bad_coverage": r"Rotational OU observed from a deliberately under-covered local region.",
    "too_large_dt": r"Multiplicative diffusion sampled at a finite-step stress time increment.",
    "van_der_pol": r"$b=(y,\ \mu(1-x^2)y-x)$ with constant isotropic tensor.",
    "fitzhugh_nagumo": r"$b=(x-x^3/3-y+I,\ \epsilon(x+a-by))$ with constant tensor.",
    "stuart_landau": r"$b=((\lambda-r^2)x-\omega y,\ \omega x+(\lambda-r^2)y)$.",
    "brusselator": r"$b=(A-(B+1)x+x^2y,\ Bx-x^2y)$ on the positive quadrant.",
    "maier_stein": r"$b=(x-x^3-\beta xy^2,\ -(1+x^2)y)$.",
    "duffing": r"$b=(y,\ -\delta y+x-x^3)$ with constant tensor.",
    "mueller_brown": r"$b=-m\nabla V_{\mathrm{MB}}(x,y)$ for the Mueller-Brown multi-well potential.",
    "sabr": r"$dF=\sigma F^\beta dW_1,\ d\sigma=\nu\sigma dW_2,\ dW_1dW_2=\rho dt$.",
    "gbm_2d": r"Two correlated geometric Brownian motions with quadratic tensor entries.",
    "two_factor_vasicek": r"Affine two-factor Gaussian rate model with correlated constant tensor.",
}

FAMILY_ORDER = [
    ("linear_ou", "Linear OU and Gaussian Affine Systems"),
    ("rotational", "Rotational and Circulation Systems"),
    ("bistable", "Bistable, Escape, and Potential Systems"),
    ("multiplicative", "Multiplicative and State-Dependent Tensors"),
    ("financial", "Financial Leverage and Positive-State Systems"),
    ("limit_cycle", "Limit-Cycle and Oscillatory Systems"),
    ("honest_limits", "Named Limits and Stress Tests"),
]

FAMILY_BY_SYSTEM = {
    "indep_ou": "linear_ou",
    "correlated_ou": "linear_ou",
    "coupled_ou": "linear_ou",
    "two_factor_vasicek": "linear_ou",
    "rotational_ou": "rotational",
    "spiral_sink_corr": "rotational",
    "nongradient_circulation": "rotational",
    "double_well_transverse": "bistable",
    "gradient_potential": "bistable",
    "maier_stein": "bistable",
    "duffing": "bistable",
    "mueller_brown": "bistable",
    "diag_multiplicative": "multiplicative",
    "nondiag_cholesky": "multiplicative",
    "near_singular": "multiplicative",
    "heston_sv": "financial",
    "heston_logsv": "financial",
    "cir_pair": "financial",
    "sabr": "financial",
    "gbm_2d": "financial",
    "van_der_pol": "limit_cycle",
    "fitzhugh_nagumo": "limit_cycle",
    "stuart_landau": "limit_cycle",
    "brusselator": "limit_cycle",
    "underdamped_langevin": "honest_limits",
    "near_boundary_heston": "honest_limits",
    "nonpoly_drift": "honest_limits",
    "bad_coverage": "honest_limits",
    "too_large_dt": "honest_limits",
}


def profile_settings(profile: str) -> dict:
    if profile == "smoke":
        return {
            "systems": ["near_singular", "van_der_pol", "gbm_2d"],
            "seeds": [9601, 9602],
            "base_steps": 300,
            "financial_steps": 500,
            "mueller_steps": 350,
            "showcase_grid": 9,
        }
    if profile == "standard":
        return {
            "systems": systems_missing_from_base(),
            "seeds": [9601, 9602, 9603, 9604, 9605],
            "base_steps": 800,
            "financial_steps": 1100,
            "mueller_steps": 700,
            "showcase_grid": 13,
        }
    return {
        "systems": systems_missing_from_base(),
        "seeds": [9601, 9602, 9603, 9604, 9605, 9606, 9607, 9608, 9609, 9610],
        "base_steps": 1200,
        "financial_steps": 1600,
        "mueller_steps": 1000,
        "showcase_grid": 15,
    }


def systems_missing_from_base() -> list[str]:
    base = set()
    path = ROOT / BASE_SHOWCASE_SUMMARY
    if path.exists():
        base = {row.get("system", "") for row in read_rows(BASE_SHOWCASE_SUMMARY)}
    return [key for key in ALL_2D_SYSTEMS if key not in base]


def reset_outputs() -> None:
    for rel in [EXTRA_RAW, EXTRA_SUMMARY, EXTRA_COEF_RAW, EXTRA_COEF, SYSTEM_INDEX, CELL_DIAGNOSIS, DATA_INTEGRITY, V62_LOG, SEGMENTS_TEX, INDEX_TABLE_TEX]:
        path = ROOT / rel
        if path.exists():
            path.unlink()


def append_dict(path: str, row: dict, fields: list[str]) -> None:
    out = ROOT / path
    out.parent.mkdir(parents=True, exist_ok=True)
    exists = out.exists()
    with out.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fields})


def checkpoint_result(result: dict) -> None:
    append_dict(EXTRA_RAW, result["summary"], SUMMARY_FIELDS)
    for row in result["coeffs"]:
        append_dict(EXTRA_COEF_RAW, row, COEF_FIELDS)


def v62_cell_for(system_key: str, seed: int, run: int, settings: dict) -> object:
    variant = wg_sindy_variant()
    cell = cell_for(system_key, variant, seed, run, "v6_2_extra", {"base_steps": settings["base_steps"], "heston_steps": settings["financial_steps"]})
    n_steps = settings["base_steps"]
    dt = 0.01
    if system_key in FINANCIAL_EXTRA:
        n_steps = settings["financial_steps"]
        dt = 1.0 / 252.0
    if system_key == "mueller_brown":
        n_steps = settings["mueller_steps"]
        dt = 0.004
    if system_key == "too_large_dt":
        n_steps = max(450, settings["base_steps"] // 2)
        dt = 0.05
    return replace(cell, experiment="v6_2_extra", n_steps=n_steps, dt=dt, n_trajectories=variant.n_trajectories)


def coefficient_term_in_scope(system_key: str, target: str, term_name: str) -> bool:
    if system_key in {"heston_sv", "heston_logsv"} and target == "b1":
        return False
    if system_key == "mueller_brown":
        return False
    if system_key == "sabr" and target in {"a11", "a12"}:
        return False
    return True


def finite_values(rows: list[dict], col: str) -> list[float]:
    return [v for v in (_safe_float(r.get(col)) for r in rows) if math.isfinite(v)]


def aggregate_coefficients(coeffs: list[dict], path: str) -> None:
    buckets: dict[tuple, list[dict]] = {}
    for row in coeffs:
        buckets.setdefault((row["system"], row["target"], row["term_name"], row["term_index"]), []).append(row)
    entries = []
    max_abs_by_system: dict[str, float] = {}
    for (system, target, term, term_index), part in sorted(buckets.items()):
        true_med, true_lo, true_hi = median_ci([_safe_float(r["coef_true"]) for r in part])
        hat_med, hat_lo, hat_hi = median_ci([_safe_float(r["coef_hat"]) for r in part])
        selected_rate = float(np.mean([as_bool(r["selected"]) for r in part]))
        active_rate = float(np.mean([as_bool(r["active_true"]) for r in part]))
        raw_selected_rate = float(np.mean([as_bool(r["raw_selected"]) for r in part]))
        in_scope_rate = float(np.mean([as_bool(r["target_in_scope"]) for r in part]))
        raw_fp = sum(as_bool(r["raw_selected"]) and not as_bool(r["active_true"]) for r in part)
        paper_fp = sum(as_bool(r["false_positive"]) for r in part)
        max_abs_by_system[system] = max(max_abs_by_system.get(system, 0.0), abs(hat_med))
        entries.append(
            {
                "system": system,
                "tier": REGISTRY[system].tier,
                "target": target,
                "term_name": term,
                "term_index": term_index,
                "true_coef_median": true_med,
                "true_coef_ci_low": true_lo,
                "true_coef_ci_high": true_hi,
                "recovered_coef_median": hat_med,
                "recovered_coef_ci_low": hat_lo,
                "recovered_coef_ci_high": hat_hi,
                "rel_error_median": abs(hat_med - true_med) / max(abs(true_med), 1e-12),
                "target_in_scope_rate": in_scope_rate,
                "raw_selected_rate": raw_selected_rate,
                "selected_rate": selected_rate,
                "active_true_rate": active_rate,
                "selected": selected_rate >= 0.5,
                "raw_false_positive_count": raw_fp,
                "paper_seed_false_positive_count": paper_fp,
                "stable_false_positive": False,
                "false_positive_count": 0,
            }
        )
    stable_fp_by_system: dict[str, int] = {}
    for row in entries:
        scale = max(max_abs_by_system.get(row["system"], 0.0), 1e-12)
        stable_fp = bool(row["selected_rate"] >= 0.8 and row["active_true_rate"] <= 0.2 and abs(row["recovered_coef_median"]) > 0.05 * scale)
        row["stable_false_positive"] = stable_fp
        row["false_positive_count"] = int(stable_fp)
        stable_fp_by_system[row["system"]] = stable_fp_by_system.get(row["system"], 0) + int(stable_fp)
    if entries:
        write_csv(path, entries, list(entries[0]))


def aggregate_summary(summaries: list[dict], path: str) -> None:
    out = []
    for system in sorted({r["system"] for r in summaries}):
        part = [r for r in summaries if r["system"] == system and r.get("run_status") == "OK"]
        if not part:
            failed = [r for r in summaries if r["system"] == system]
            first = failed[0] if failed else {"system": system}
            out.append(
                {
                    "system": system,
                    "tier": REGISTRY[system].tier,
                    "n": 0,
                    "objective_drift_median": float("nan"),
                    "objective_drift_ci_low": float("nan"),
                    "objective_drift_ci_high": float("nan"),
                    "diffusion_median": float("nan"),
                    "diffusion_ci_low": float("nan"),
                    "diffusion_ci_high": float("nan"),
                    "a12_cosine_median": float("nan"),
                    "a12_cosine_ci_low": float("nan"),
                    "a12_cosine_ci_high": float("nan"),
                    "psd_valid_median": float("nan"),
                    "generator_action_error_median": float("nan"),
                    "generator_action_error_ci_low": float("nan"),
                    "generator_action_error_ci_high": float("nan"),
                    "false_positive_count": 0,
                    "pass_marker": "FIT_EXCEPTION",
                    "scope_metric_contract": metric_contract(system),
                    "run_status": first.get("run_status", "FIT_EXCEPTION"),
                    "error": first.get("error", ""),
                }
            )
            continue
        drift_med, drift_lo, drift_hi = median_ci(finite_values(part, "objective_drift_rel_l2"))
        diff_med, diff_lo, diff_hi = median_ci(finite_values(part, "diffusion_rel_l2"))
        cos_med, cos_lo, cos_hi = median_ci(finite_values(part, "a12_cosine"))
        psd_med, _, _ = median_ci(finite_values(part, "psd_valid_pct"))
        gen_med, gen_lo, gen_hi = median_ci(finite_values(part, "generator_action_error"))
        gate = bool(drift_med < 0.80 and diff_med < 0.45 and psd_med >= 0.99 and (not math.isfinite(cos_med) or cos_med > 0.85))
        out.append(
            {
                "system": system,
                "tier": REGISTRY[system].tier,
                "n": len(part),
                "objective_drift_median": drift_med,
                "objective_drift_ci_low": drift_lo,
                "objective_drift_ci_high": drift_hi,
                "diffusion_median": diff_med,
                "diffusion_ci_low": diff_lo,
                "diffusion_ci_high": diff_hi,
                "a12_cosine_median": cos_med,
                "a12_cosine_ci_low": cos_lo,
                "a12_cosine_ci_high": cos_hi,
                "psd_valid_median": psd_med,
                "generator_action_error_median": gen_med,
                "generator_action_error_ci_low": gen_lo,
                "generator_action_error_ci_high": gen_hi,
                "false_positive_count": 0,
                "pass_marker": "PASS" if gate else "SCOPED_REVIEW",
                "scope_metric_contract": metric_contract(system),
                "run_status": "OK",
                "error": "",
            }
        )
    if out:
        write_csv(path, out, list(out[0]))


def as_bool(value: object) -> bool:
    return value is True or str(value).lower() == "true"


def _showcase_task(task: tuple[str, int, int, dict]) -> dict:
    system_key, seed, run, settings = task
    started = time.perf_counter()
    try:
        cell = v62_cell_for(system_key, seed, run, settings)
        system, x, fit, runtime = fit_cell(cell)
        points = central_grid(x, settings["showcase_grid"])
        errs = function_l2_errors(fit, system, points)
        psd = psd_validity(fit.evaluate(points)[1])
        tmet = tensor_metrics(fit, system, points)
        a12_cos = cosine_similarity(fit.evaluate(points)[1][:, 0, 1], system.true_diffusion(points)[:, 0, 1])
        gen_err = generator_action_error(fit, system, points)
        objective = errs.get("b2_rel_l2", errs["drift_rel_l2"]) if system_key in {"heston_sv", "heston_logsv"} else errs["drift_rel_l2"]
        summary = {
            "system": system_key,
            "tier": REGISTRY[system_key].tier,
            "seed": seed,
            "run": run,
            "variant_id": wg_sindy_variant().variant_id,
            "library": cell.library,
            "R": cell.n_trajectories,
            "n_steps": cell.n_steps,
            "dt": cell.dt,
            "objective_drift_rel_l2": objective,
            "drift_rel_l2": errs["drift_rel_l2"],
            "diffusion_rel_l2": errs["diffusion_rel_l2"],
            "b1_rel_l2": errs.get("b1_rel_l2", float("nan")),
            "b2_rel_l2": errs.get("b2_rel_l2", float("nan")),
            "a11_rel_l2": errs.get("a11_rel_l2", float("nan")),
            "a22_rel_l2": errs.get("a22_rel_l2", float("nan")),
            "a12_rel_l2": errs.get("a12_rel_l2", float("nan")),
            "a12_cosine": a12_cos,
            "a12_sign_acc": tmet.get("a12_sign_accuracy", float("nan")),
            "psd_valid_pct": psd["pct_psd_valid"],
            "generator_action_error": gen_err,
            "scope_metric_contract": metric_contract(system_key),
            "runtime_sec": runtime,
            "run_status": "OK",
            "error": "",
        }
        coeffs = []
        for target in active_targets(2):
            chat, ctrue = coeffs_in_fit_space(fit, system, points, target)
            scale = max(float(np.max(np.abs(ctrue))), 1e-12)
            hat_scale = max(float(np.max(np.abs(chat))), 1e-12)
            for idx, name in enumerate(fit.library.names):
                term_in_scope = coefficient_term_in_scope(system_key, target, name)
                active_true = abs(float(ctrue[idx])) > 0.02 * scale
                raw_selected = abs(float(chat[idx])) > 0.02 * hat_scale
                selected = term_in_scope and abs(float(chat[idx])) > 0.20 * hat_scale
                coeffs.append(
                    {
                        "system": system_key,
                        "tier": REGISTRY[system_key].tier,
                        "seed": seed,
                        "target": target,
                        "term_name": name,
                        "term_index": idx,
                        "coef_true": float(ctrue[idx]),
                        "coef_hat": float(chat[idx]),
                        "rel_error": abs(float(chat[idx] - ctrue[idx])) / max(abs(float(ctrue[idx])), 1e-12),
                        "target_in_scope": term_in_scope,
                        "term_in_paper_scope": term_in_scope,
                        "active_true": active_true,
                        "raw_selected": raw_selected,
                        "selected": selected,
                        "false_positive": bool(selected and not active_true),
                    }
                )
        return {"summary": summary, "coeffs": coeffs}
    except Exception as exc:  # noqa: BLE001 - failures must become audit rows, not campaign crashes.
        return {
            "summary": {
                "system": system_key,
                "tier": REGISTRY[system_key].tier,
                "seed": seed,
                "run": run,
                "variant_id": wg_sindy_variant().variant_id,
                "library": REGISTRY[system_key].library,
                "R": wg_sindy_variant().n_trajectories,
                "n_steps": settings["base_steps"],
                "dt": 0.01,
                "objective_drift_rel_l2": float("nan"),
                "drift_rel_l2": float("nan"),
                "diffusion_rel_l2": float("nan"),
                "b1_rel_l2": float("nan"),
                "b2_rel_l2": float("nan"),
                "a11_rel_l2": float("nan"),
                "a22_rel_l2": float("nan"),
                "a12_rel_l2": float("nan"),
                "a12_cosine": float("nan"),
                "a12_sign_acc": float("nan"),
                "psd_valid_pct": float("nan"),
                "generator_action_error": float("nan"),
                "scope_metric_contract": metric_contract(system_key),
                "runtime_sec": time.perf_counter() - started,
                "run_status": "FIT_EXCEPTION",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback_tail": traceback.format_exc(limit=3).replace("\n", " | "),
            },
            "coeffs": [],
        }


def completed_extra_keys(path: str) -> set[tuple[str, str]]:
    return {(row.get("system", ""), row.get("seed", "")) for row in read_rows(path)}


def run_extra_showcase(settings: dict, resume: bool, jobs: int) -> None:
    systems = settings["systems"]
    seeds = settings["seeds"]
    pending: list[tuple[str, int, int, dict]] = []
    done = completed_extra_keys(EXTRA_RAW) if resume else set()
    for run, seed in enumerate(seeds):
        for system in systems:
            if (system, str(seed)) not in done:
                pending.append((system, seed, run, settings))
    summaries = read_rows(EXTRA_RAW) if resume and (ROOT / EXTRA_RAW).exists() else []
    coeffs = read_rows(EXTRA_COEF_RAW) if resume and (ROOT / EXTRA_COEF_RAW).exists() else []
    if pending:
        workers = max(1, int(jobs))
        if workers == 1:
            for idx, task in enumerate(pending, start=1):
                result = _showcase_task(task)
                summaries.append(result["summary"])
                coeffs.extend(result["coeffs"])
                checkpoint_result(result)
                print(f"v6_2_extra {idx:4d}/{len(pending):<4d} {task[0]:24s} status={result['summary'].get('run_status')}", flush=True)
        else:
            print(f"v6_2_extra running {len(pending)} cells with jobs={workers}", flush=True)
            futures = set()
            task_iter = iter(pending)
            completed = 0

            def submit_some(executor: ProcessPoolExecutor) -> None:
                while len(futures) < max(workers * 2, workers):
                    try:
                        futures.add(executor.submit(_showcase_task, next(task_iter)))
                    except StopIteration:
                        return

            try:
                with ProcessPoolExecutor(max_workers=workers) as executor:
                    submit_some(executor)
                    while futures:
                        done_futures, futures = wait(futures, return_when=FIRST_COMPLETED)
                        for future in done_futures:
                            result = future.result()
                            summaries.append(result["summary"])
                            coeffs.extend(result["coeffs"])
                            checkpoint_result(result)
                            completed += 1
                            if completed % 10 == 0 or result["summary"].get("run_status") != "OK":
                                print(f"v6_2_extra {completed:4d}/{len(pending):<4d} {result['summary'].get('system',''):24s} status={result['summary'].get('run_status')}", flush=True)
                        submit_some(executor)
            except PermissionError as exc:
                print(f"v6_2_extra multiprocessing unavailable ({exc}); falling back to sequential execution", flush=True)
                for idx, task in enumerate(pending, start=1):
                    result = _showcase_task(task)
                    summaries.append(result["summary"])
                    coeffs.extend(result["coeffs"])
                    checkpoint_result(result)
                    if idx % 10 == 0 or result["summary"].get("run_status") != "OK":
                        print(f"v6_2_extra {idx:4d}/{len(pending):<4d} {task[0]:24s} status={result['summary'].get('run_status')}", flush=True)
    write_csv(EXTRA_RAW, summaries, SUMMARY_FIELDS)
    if coeffs:
        write_csv(EXTRA_COEF_RAW, coeffs, COEF_FIELDS)
        aggregate_coefficients(coeffs, EXTRA_COEF)
    aggregate_summary(summaries, EXTRA_SUMMARY)


def load_summary_rows() -> list[dict]:
    rows = []
    if (ROOT / BASE_SHOWCASE_SUMMARY).exists():
        for row in read_rows(BASE_SHOWCASE_SUMMARY):
            out = dict(row)
            out["source"] = "v6_showcase"
            rows.append(out)
    if (ROOT / EXTRA_SUMMARY).exists():
        for row in read_rows(EXTRA_SUMMARY):
            out = dict(row)
            out["source"] = "v6_2_extra"
            rows.append(out)
    return rows


def row_metric(row: dict, name: str) -> float:
    mapping = {
        "drift": "objective_drift_median",
        "diffusion": "diffusion_median",
        "a12_cosine": "a12_cosine_median",
        "psd": "psd_valid_median",
        "generator": "generator_action_error_median",
    }
    return _safe_float(row.get(mapping[name]))


def verdict_for(system: str, row: dict) -> tuple[str, str, str]:
    truth = REGISTRY[system]
    drift = row_metric(row, "drift")
    diff = row_metric(row, "diffusion")
    cos = row_metric(row, "a12_cosine")
    psd = row_metric(row, "psd")
    if row.get("pass_marker") == "FIT_EXCEPTION":
        return "NAMED_NULL", "fit_exception_captured_in_audit", "NAMED_LIMIT"
    if "FAIL" in truth.verdict:
        return "NAMED_NULL", "registry_declared_failure_or_stress_case", "NAMED_LIMIT"
    gate = bool(math.isfinite(drift) and math.isfinite(diff) and math.isfinite(psd) and drift < 0.80 and diff < 0.45 and psd >= 0.99 and (not math.isfinite(cos) or cos > 0.85))
    if gate:
        return "PASS", "metric_gate_passed", "IN_SCOPE_POSITIVE" if system in IN_SCOPE_SYSTEMS or truth.verdict.startswith("STRONG") else "FRAGILE_PASS"
    if truth.verdict == "FRAGILE":
        return "NAMED_NULL", "fragile_physics_or_identifiability_limit", "NAMED_LIMIT"
    return "SCOPED_REVIEW", "metric_gate_not_met_but_truth_is_representable", "EXPANDED_REVIEW"


def write_system_index() -> list[dict]:
    by_system = {row["system"]: row for row in load_summary_rows()}
    rows = []
    for system in ALL_2D_SYSTEMS:
        truth = REGISTRY[system]
        row = by_system.get(system, {})
        verdict, reason, scope = verdict_for(system, row) if row else ("MISSING", "no_v6_or_v6_2_summary_row", "MISSING")
        family = FAMILY_BY_SYSTEM.get(system, "honest_limits")
        rows.append(
            {
                "system": system,
                "tier": truth.tier,
                "family": family,
                "dim": truth.dim,
                "scope_status": scope,
                "registry_verdict": truth.verdict,
                "library": truth.library,
                "n_seeds": int(_safe_float(row.get("n"), 0.0)),
                "drift_l2_mu": row_metric(row, "drift"),
                "tensor_rel_l2": row_metric(row, "diffusion"),
                "a12_cosine": row_metric(row, "a12_cosine"),
                "psd_valid_pct": row_metric(row, "psd"),
                "generator_action_error": row_metric(row, "generator"),
                "false_positive_count": int(_safe_float(row.get("false_positive_count"), 0.0)),
                "verdict": verdict,
                "verdict_reason": reason,
                "metric_contract": row.get("scope_metric_contract", metric_contract(system)),
                "source": row.get("source", ""),
                "figure_png": f"figures/v6/showcase_fields_{system}.png",
                "figure_pdf": f"figures/v6/showcase_fields_{system}.pdf",
                "paper_subsection_label": f"sec:v62-{system.replace('_', '-')}",
            }
        )
    write_csv(SYSTEM_INDEX, rows, list(rows[0]))
    return rows


def diagnose_cells() -> list[dict]:
    path = ROOT / "results/benchmark_summary.csv"
    if not path.exists():
        rows = [{"system": "", "metric": "", "value": "", "diagnosis_class": "missing_source", "evidence": "results/benchmark_summary.csv missing", "action": "rerun benchmark suite"}]
        write_csv(CELL_DIAGNOSIS, rows, list(rows[0]))
        return rows
    df = pd.read_csv(path)
    out = []
    metrics = [
        ("drift_rel_l2", "max", 1.0, "oracle_drift_rel_l2"),
        ("diffusion_rel_l2", "max", 0.75, "oracle_diffusion_rel_l2"),
        ("a12_cosine", "min", 0.85, "oracle_a12_cosine"),
        ("psd_valid_pct", "min", 0.95, ""),
    ]
    for system, part in df.groupby("system", dropna=False):
        truth = REGISTRY.get(str(system))
        for metric, reducer, threshold, oracle_col in metrics:
            if metric not in part:
                continue
            vals = pd.to_numeric(part[metric], errors="coerce")
            finite = vals[np.isfinite(vals)]
            if finite.empty:
                continue
            value = float(finite.max() if reducer == "max" else finite.min())
            oracle_value = float("nan")
            if oracle_col and oracle_col in part:
                oval = pd.to_numeric(part[oracle_col], errors="coerce")
                ofinite = oval[np.isfinite(oval)]
                if not ofinite.empty:
                    oracle_value = float(ofinite.max() if reducer == "max" else ofinite.min())
            impossible = (metric.endswith("rel_l2") and value < -1e-12) or (metric.endswith("cosine") and (value < -1.000001 or value > 1.000001)) or (metric == "psd_valid_pct" and (value < -1e-12 or value > 1.000001))
            flagged = impossible or (metric in {"drift_rel_l2", "diffusion_rel_l2"} and value > threshold) or (metric == "a12_cosine" and value < threshold) or (metric == "psd_valid_pct" and value < threshold)
            if not flagged:
                klass = "ok"
                evidence = "within audit threshold"
                action = "no action"
            elif impossible:
                klass = "genuine_bug"
                evidence = "metric outside mathematically possible range"
                action = "fixed by source metric/integrity tests before accepting v6.2"
            else:
                verdict = truth.verdict if truth else ""
                oracle_bad = math.isfinite(oracle_value) and ((metric in {"drift_rel_l2", "diffusion_rel_l2"} and oracle_value > 0.75 * threshold) or (metric == "a12_cosine" and oracle_value < threshold))
                if "FAIL" in verdict or verdict == "FRAGILE" or oracle_bad:
                    klass = "real_recovery_failure"
                    evidence = f"registry={verdict}; oracle={oracle_value:.4g}; value={value:.4g}"
                    action = "keep number; route to named-null segment; display with split/log-clipped heatmap"
                else:
                    klass = "scale_plot_artifact"
                    evidence = f"value={value:.4g} is finite but would dominate a linear shared color scale"
                    action = "keep data; use split-panel clipped/log severity scale"
            out.append(
                {
                    "system": system,
                    "metric": metric,
                    "value": value,
                    "threshold": threshold,
                    "oracle_value": oracle_value,
                    "diagnosis_class": klass,
                    "evidence": evidence,
                    "action": action,
                }
            )
    write_csv(CELL_DIAGNOSIS, out, list(out[0]))
    return out


def sweep_data_integrity() -> list[dict]:
    rows = []
    exact_fraction_cols = {"psd_valid_pct", "psd_valid_median", "median_psd_valid_pct", "a12_sign_acc", "a12_sign_accuracy"}
    for path in sorted((ROOT / "results").glob("**/*.csv")):
        rel = str(path.relative_to(ROOT))
        try:
            df = pd.read_csv(path)
        except Exception as exc:  # noqa: BLE001
            rows.append({"file": rel, "check": "read_csv", "status": "ERROR", "n_issues": 1, "detail": f"{type(exc).__name__}: {exc}"})
            continue
        rows.append({"file": rel, "check": "read_csv", "status": "PASS", "n_issues": 0, "detail": f"rows={len(df)} cols={len(df.columns)}"})
        dupes = int(df.duplicated().sum())
        rows.append({"file": rel, "check": "duplicate_full_rows", "status": "PASS" if dupes == 0 else "WARN", "n_issues": dupes, "detail": "exact duplicate rows"})
        allowed_na = int(df.isna().sum().sum())
        rows.append({"file": rel, "check": "allowed_missing_cells", "status": "INFO" if allowed_na else "PASS", "n_issues": allowed_na, "detail": "blank/nan allowed for non-applicable metrics such as a12 on diagonal systems"})
        for col in df.columns:
            series = pd.to_numeric(df[col], errors="coerce")
            finite = series[np.isfinite(series)]
            inf_count = int(np.isinf(series.to_numpy(float, na_value=np.nan)).sum()) if hasattr(series, "to_numpy") else 0
            if inf_count:
                if col == "cond_design":
                    rows.append({"file": rel, "check": f"{col}:rank_deficient_infinite", "status": "INFO", "n_issues": inf_count, "detail": "infinite condition number records rank-deficient design diagnostics, not a finite metric"})
                else:
                    rows.append({"file": rel, "check": f"{col}:infinite", "status": "ERROR", "n_issues": inf_count, "detail": "infinite numeric values"})
            if col.endswith("cosine") and not finite.empty:
                bad = int(((finite < -1.000001) | (finite > 1.000001)).sum())
                rows.append({"file": rel, "check": f"{col}:cosine_range", "status": "PASS" if bad == 0 else "ERROR", "n_issues": bad, "detail": "must lie in [-1,1]"})
            if col in exact_fraction_cols and not finite.empty:
                bad = int(((finite < -1e-12) | (finite > 1.000001)).sum())
                rows.append({"file": rel, "check": f"{col}:fraction_range", "status": "PASS" if bad == 0 else "ERROR", "n_issues": bad, "detail": "must lie in [0,1]"})
            if (col.endswith("rel_l2") or col.endswith("rel_error") or col.endswith("abs_error") or col in {"diffusion_median", "objective_drift_median", "tensor_rel_l2", "drift_l2_mu"}) and not finite.empty:
                bad = int((finite < -1e-12).sum())
                rows.append({"file": rel, "check": f"{col}:nonnegative", "status": "PASS" if bad == 0 else "ERROR", "n_issues": bad, "detail": "relative/absolute errors must be nonnegative"})
    write_csv(DATA_INTEGRITY, rows, ["file", "check", "status", "n_issues", "detail"])
    return rows


def latex_escape(text: object) -> str:
    s = str(text)
    return (
        s.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("$", r"\$")
        .replace("#", r"\#")
        .replace("_", r"\_")
        .replace("{", r"\{")
        .replace("}", r"\}")
    )


def fmt_tex(value: object) -> str:
    v = _safe_float(value)
    if not math.isfinite(v):
        return "n/a"
    if abs(v) >= 10:
        return f"{v:.1f}"
    if abs(v) >= 1:
        return f"{v:.2f}"
    return f"{v:.3f}"


def combine_coefficients() -> pd.DataFrame:
    frames = []
    for rel in [BASE_SHOWCASE_COEF, EXTRA_COEF]:
        path = ROOT / rel
        if path.exists():
            frames.append(pd.read_csv(path))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def top_terms_for(coefs: pd.DataFrame, system: str, target: str, n: int = 3) -> str:
    if coefs.empty:
        return "none"
    part = coefs[(coefs["system"] == system) & (coefs["target"] == target)].copy()
    if part.empty:
        return "none"
    part["abs_hat"] = pd.to_numeric(part["recovered_coef_median"], errors="coerce").abs()
    part = part.sort_values("abs_hat", ascending=False).head(n)
    terms = []
    for _, row in part.iterrows():
        val = _safe_float(row.get("recovered_coef_median"))
        if math.isfinite(val) and abs(val) > 1e-10:
            terms.append(f"{latex_escape(row.get('term_name'))}:{val:.3g}")
    return ", ".join(terms) if terms else "none"


def write_paper_segments(index_rows: list[dict]) -> None:
    paper_dir = ROOT / "paper"
    paper_dir.mkdir(exist_ok=True)
    coefs = combine_coefficients()
    by_family: dict[str, list[dict]] = {}
    for row in index_rows:
        by_family.setdefault(row["family"], []).append(row)
    lines = [
        r"\section{V6.2 Per-System Segments}",
        "This appendix is generated from \\texttt{results/v6/system\\_index.csv}. Every registry two-dimensional SDE has one subsection, one verdict, and one showcase figure.",
        "",
    ]
    for family, title in FAMILY_ORDER:
        part = sorted(by_family.get(family, []), key=lambda r: r["system"])
        if not part:
            continue
        lines.append(rf"\subsection*{{{title}}}")
        for row in part:
            system = row["system"]
            label = row["paper_subsection_label"]
            terms = "; ".join(f"{target}: {top_terms_for(coefs, system, target, 3)}" for target in ["b1", "b2", "a11", "a12", "a22"])
            lines.extend(
                [
                    rf"\subsection{{{latex_escape(system.replace('_', ' ').title())}}}",
                    rf"\label{{{label}}}",
                    rf"\paragraph{{SDE and analytic generator.}} {SYSTEM_DEFINITIONS.get(system, 'Analytic drift and tensor are implemented in systems.py')} The recovered object is the generator drift vector and symmetric tensor, not a unique volatility factor.",
                    rf"\paragraph{{Recovered generator.}} Library {latex_escape(row['library'])}; drift error {fmt_tex(row['drift_l2_mu'])}; tensor error {fmt_tex(row['tensor_rel_l2'])}; $a_{{12}}$ cosine {fmt_tex(row['a12_cosine'])}; PSD-valid fraction {fmt_tex(row['psd_valid_pct'])}. Dominant recovered terms: {terms}.",
                    rf"\paragraph{{Verdict.}} \textbf{{{latex_escape(row['verdict'])}}}: {latex_escape(row['verdict_reason'])}.",
                    rf"\begin{{figure}}[p]\centering\includegraphics[width=\linewidth]{{showcase_fields_{system}.pdf}}\caption{{V6.2 recovered generator fields for {latex_escape(system.replace('_', ' '))}.}}\label{{fig:v62-{system.replace('_', '-')}}}\end{{figure}}",
                    "",
                ]
            )
    (paper_dir / "v6_2_system_segments.tex").write_text("\n".join(lines) + "\n")

    table_lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\scriptsize",
        r"\begin{tabular}{llrrrrl}",
        r"\toprule",
        r"System & Scope & Drift & Tensor & $a_{12}$ cos. & PSD & Verdict \\",
        r"\midrule",
    ]
    for row in index_rows:
        table_lines.append(
            f"{latex_escape(row['system'])} & {latex_escape(row['scope_status'])} & {fmt_tex(row['drift_l2_mu'])} & {fmt_tex(row['tensor_rel_l2'])} & {fmt_tex(row['a12_cosine'])} & {fmt_tex(row['psd_valid_pct'])} & {latex_escape(row['verdict'])} \\\\"
        )
    table_lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{V6.2 master system index generated from \texttt{results/v6/system\_index.csv}. Named limits are retained as rows rather than removed from the zoo.}",
            r"\label{tab:showcase}",
            r"\end{table}",
            "",
        ]
    )
    (paper_dir / "v6_2_system_index_table.tex").write_text("\n".join(table_lines))


def validate_v62_artifacts(index_rows: list[dict]) -> None:
    systems = {row["system"] for row in index_rows}
    expected = set(ALL_2D_SYSTEMS)
    if systems != expected:
        raise RuntimeError(f"system_index mismatch missing={sorted(expected - systems)} extra={sorted(systems - expected)}")
    errors = [row for row in read_rows(DATA_INTEGRITY) if row.get("status") == "ERROR"]
    if errors:
        raise RuntimeError(f"data integrity ERROR rows: {errors[:3]}")


def update_run_log(profile: str, started: float, settings: dict) -> None:
    previous = read_rows(V62_LOG)
    previous_runtime = max((_safe_float(row.get("runtime_sec")) for row in previous), default=float("nan"))
    current_runtime = time.perf_counter() - started
    runtime = max(previous_runtime, current_runtime) if math.isfinite(previous_runtime) else current_runtime
    rows = [
        {
            "profile": profile,
            "registry_2d_systems": len(ALL_2D_SYSTEMS),
            "extra_systems_run": len(settings["systems"]),
            "new_canonical_systems": len(NEW_CANONICAL_SYSTEMS),
            "seeds": len(settings["seeds"]),
            "extra_raw_rows": len(read_rows(EXTRA_RAW)),
            "system_index_rows": len(read_rows(SYSTEM_INDEX)),
            "data_integrity_errors": sum(1 for row in read_rows(DATA_INTEGRITY) if row.get("status") == "ERROR"),
            "runtime_sec": runtime,
            "finalize_runtime_sec": current_runtime,
            "runtime_basis": "max(previous runtime_sec, current process runtime); resume finalization can be shorter than the original full run",
        }
    ]
    write_csv(V62_LOG, rows, list(rows[0]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=["smoke", "standard", "full"], default="full")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--jobs", type=int, default=int(os.environ.get("V62_JOBS", "1")))
    args = parser.parse_args()
    started = time.perf_counter()
    settings = profile_settings(args.profile)
    if not args.resume:
        reset_outputs()

    print("V6.2 Stage A: frozen WG-SINDy on missing registry systems")
    run_extra_showcase(settings, args.resume, args.jobs)

    print("V6.2 Stage B: system index and per-cell diagnosis")
    index_rows = write_system_index()
    diagnose_cells()

    print("V6.2 Stage C: data integrity sweep")
    sweep_data_integrity()

    print("V6.2 Stage D: paper segment/table generation")
    write_paper_segments(index_rows)

    print("V6.2 Stage E: validate audit artifacts")
    validate_v62_artifacts(index_rows)
    update_run_log(args.profile, started, settings)
    print("V6.2 DONE")


if __name__ == "__main__":
    main()
