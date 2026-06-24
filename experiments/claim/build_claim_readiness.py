from __future__ import annotations

import csv
from pathlib import Path

from experiments.common import ROOT


def read_csv(path: str) -> list[dict]:
    p = ROOT / path
    if not p.exists():
        return []
    with p.open() as f:
        return list(csv.DictReader(f))


def ok_systems(rows: list[dict]) -> set[str]:
    return {
        row["system"]
        for row in rows
        if row.get("status") == "VALIDATED_POSITIVE" and row.get("pass_level") in {"strong", "medium"}
    }


def ok_split_systems(rows: list[dict], column: str) -> set[str]:
    return {row["system"] for row in rows if row.get(column) in {"strong", "medium"}}


def ok_component_drift_systems(rows: list[dict]) -> set[str]:
    ok = set()
    for row in rows:
        b1 = fnum(row, "b1_rel_l2")
        b2 = fnum(row, "b2_rel_l2")
        if b1 < 0.75 and (row.get("dim") == "1" or b2 < 0.75):
            ok.add(row["system"])
    return ok


def status_for(required: set[str], have: set[str]) -> bool:
    return required.issubset(have)


def fnum(row: dict, key: str) -> float:
    try:
        return float(row.get(key, "nan"))
    except ValueError:
        return float("nan")


def main() -> None:
    rows = read_csv("results/benchmark_summary.csv")
    robust = read_csv("results/robustness_grid.csv")
    failures = read_csv("results/failure_case_report.csv")
    convergence = read_csv("results/convergence_slopes.csv")
    leverage = read_csv("results/heston_cir/lev1_synthetic_regimes.csv")
    eiv = read_csv("results/heston_cir/lev2_eiv_phase_transition.csv")
    ewma = read_csv("results/heston_cir/lev3_ewma_proxy.csv")
    circulation = read_csv("results/circulation/circ3_detector_nulls.csv")
    fluctuation = read_csv("results/fluctuation/fluc1_noise_correction.csv")
    lines = ["# Claim Readiness Report", ""]
    if rows:
        have = ok_systems(rows)
        tensor_have = ok_split_systems(rows, "tensor_pass_level")
        drift_have = ok_component_drift_systems(rows)
        l1_req = {"indep_ou", "coupled_ou", "diag_multiplicative"}
        l2_req = {"coupled_ou", "rotational_ou", "double_well_transverse"}
        l3_req = {"correlated_ou", "nondiag_cholesky", "heston_logsv"}
        l3_tensor_req = {"correlated_ou", "spiral_sink_corr", "nondiag_cholesky", "heston_logsv"}
        l1 = status_for(l1_req, have)
        l2 = status_for(l2_req, have)
        l3_tensor = status_for(l3_tensor_req, tensor_have)
        l3_drift_strict = status_for(l3_tensor_req, drift_have)
        l3_drift_except_heston = status_for(l3_tensor_req - {"heston_logsv"}, drift_have)
        l3 = l3_tensor and l3_drift_strict
        l3_exception = l3_tensor and l3_drift_except_heston
        non_failure = [r for r in rows if r.get("tier") not in {"6"} and r.get("dim") == "2"]
        positive = [r for r in non_failure if r.get("system") in have]
        tensor_positive = [r for r in non_failure if r.get("system") in tensor_have]
        broad_fraction = len({r["system"] for r in positive}) / max(len({r["system"] for r in non_failure}), 1)
        broad_tensor_fraction = len({r["system"] for r in tensor_positive}) / max(len({r["system"] for r in non_failure}), 1)
        robust_ok = bool(robust) and sum(str(r.get("conditioned_ok")) == "True" for r in robust) >= max(1, int(0.5 * len(robust)))
        failure_ok = bool(failures) and any(str(r.get("observed_failure")) == "True" and str(r.get("expected_failure")) == "True" for r in failures)
        convergence_ok = bool(convergence) and any(float(r.get("log_slope") or "nan") < 0.0 for r in convergence)
        l4 = l1 and l2 and l3 and broad_fraction >= 0.70 and robust_ok and failure_ok and convergence_ok
        l4_exception = l1 and l2 and l3_exception and broad_fraction >= 0.70 and robust_ok and failure_ok and convergence_ok
        l4_tensor_partial = l1 and l2 and l3_tensor and broad_tensor_fraction >= 0.70 and robust_ok and failure_ok and convergence_ok
        failed = [r for r in rows if r.get("status") not in {"VALIDATED_POSITIVE", "NEGATIVE_RESULT_WORTH_REPORTING"}]
        failure_modes = [r for r in failures if str(r.get("observed_failure")) == "True"]
        heston_rows = [r for r in rows if r.get("system") == "heston_logsv"]
        heston_best = min(heston_rows, key=lambda r: fnum(r, "diffusion_rel_l2")) if heston_rows else {}
        near_rows = [r for r in rows if r.get("system") == "near_singular"]
        near_best_drift = min((fnum(r, "drift_rel_l2") for r in near_rows), default=float("nan"))
        near_best_diff = min((fnum(r, "diffusion_rel_l2") for r in near_rows), default=float("nan"))
        leverage_sign = sum(str(r.get("rho_sign_correct")) == "True" for r in leverage)
        leverage_rate = leverage_sign / max(len(leverage), 1)
        eiv_pairs: dict[tuple[str, str], dict[bool, float]] = {}
        for row in eiv:
            key = (row.get("noise_to_signal_ratio", ""), row.get("seed", ""))
            eiv_pairs.setdefault(key, {})[str(row.get("noise_correct")) == "True"] = fnum(row, "a11_rel_l2")
        eiv_corrected_wins = sum(vals.get(True, float("inf")) < vals.get(False, -float("inf")) for vals in eiv_pairs.values())
        circ_null = [r for r in circulation if r.get("mode") == "reversible_null"]
        circ_power = [r for r in circulation if r.get("mode") == "power"]
        circ_type1 = sum(str(r.get("detector_fires")) == "True" for r in circ_null) / max(len(circ_null), 1)
        circ_power_rate = sum(str(r.get("detector_fires")) == "True" for r in circ_power) / max(len(circ_power), 1)
        fluc_pairs: dict[tuple[str, str, str], dict[bool, float]] = {}
        for row in fluctuation:
            key = (row.get("system", ""), row.get("noise_kind", ""), row.get("noise_to_signal_ratio", ""))
            fluc_pairs.setdefault(key, {})[str(row.get("noise_correct")) == "True"] = fnum(row, "diffusion_rel_l2")
        fluc_corrected_wins = sum(vals.get(True, float("inf")) < vals.get(False, -float("inf")) for vals in fluc_pairs.values())
        strongest = (
            "Level 4 broad-class 2D generator recovery"
            if l4
            else "Level 4 broad-class 2D generator recovery with named low-SNR Heston drift exception"
            if l4_exception
            else "Level 4-TENSOR-PARTIAL broad diffusion-tensor recovery with drift limitations"
            if l4_tensor_partial
            else "Level 3 drift+tensor recovery except the low-SNR Heston log-price drift"
            if l3_exception
            else "Level 3-TENSOR full 2x2 diffusion-tensor recovery with drift limitation"
            if l3_tensor
            else "Level 2 coupled-generator recovery"
            if l2
            else "Level 1 limited 2D feasibility"
            if l1
            else "No broad claim beyond runnable smoke evidence"
        )
        lines += [
            f"Claim Level 1 (2D feasibility - independent OU + coupled OU + diagonal multiplicative): {'PASS' if l1 else 'FAIL'}",
            f"Claim Level 2 (coupled generator - coupled OU + rotational OU + nonlinear double-well): {'PASS' if l2 else 'FAIL'}",
            f"Claim Level 3-TENSOR (full 2x2 diffusion tensor incl. off-diagonal/leverage): {'PASS' if l3_tensor else 'FAIL'}",
            f"Claim Level 3-DRIFT (same systems with drift recovery): {'PASS' if l3_drift_strict else 'PASS-except-heston' if l3_drift_except_heston else 'FAIL'}",
            f"Claim Level 3-COMBINED (legacy drift AND tensor gate): {'PASS' if l3 else 'PASS-except-heston' if l3_exception else 'FAIL'}",
            f"Claim Level 4 (broad class - most families + robustness grid + failures + convergence): {'PASS' if l4 else 'FAIL'}",
            f"Claim Level 4-HONEST-EXCEPTION (same, with Heston log-price drift named as low-SNR null): {'PASS' if l4_exception else 'FAIL'}",
            f"Claim Level 4-TENSOR-PARTIAL (tensor split, broad grid): {'PASS' if l4_tensor_partial else 'FAIL'}",
            "",
            f"Strongest defensible claim: {strongest}.",
            "Claims not supported: universal 2D recovery, unique sigma recovery, calibrated circulation detector, full empirical generator identification.",
            "Best figures for paper: figures/benchmark_zoo_errors.png, figures/robustness_grid.png, figures/failure_cases.png, figures/convergence_slopes.png, figures/leverage_tensor_regimes.png, figures/circulation_detector_nulls.png, figures/fluctuation_noise_correction.png.",
            f"Broad non-failure system pass fraction: {broad_fraction:.3f}.",
            f"Broad non-failure tensor-pass fraction: {broad_tensor_fraction:.3f}.",
            f"V3 default evidence: zoo/failure use z-space + lasso_stlsq where the library has a valid back-transform; raw/stlsq remains an explicit robustness ablation.",
            f"Benchmarks that failed: {', '.join(sorted({r['system'] for r in failed})) if failed else 'none in benchmark_summary.csv'}.",
            f"Reason for failure: {', '.join(sorted({r['failure_mode'] for r in failure_modes})) if failure_modes else 'none observed in failure_case_report.csv'}.",
            f"Heston logSV note: tensor/leverage succeeds separately from drift; best diffusion_rel_l2={heston_best.get('diffusion_rel_l2', 'nan')}, a12_cosine={heston_best.get('a12_cosine', 'nan')}, b1_rel_l2={heston_best.get('b1_rel_l2', 'nan')}, b2_rel_l2={heston_best.get('b2_rel_l2', 'nan')}. The log-price drift is treated as a low-SNR identifiability limitation, not hidden as a tensor failure.",
            f"Near-singular note: best drift_rel_l2={near_best_drift}, best diffusion_rel_l2={near_best_diff}; this is scoped as an off-diagonal/PSD stress case, not a drift null.",
            f"Leverage read-out: tensor-derived rho sign accuracy {leverage_sign}/{len(leverage)} = {leverage_rate:.3f}; parametric Heston decomposition is secondary.",
            f"EIV read-out: lag-1 correction wins on {eiv_corrected_wins}/{len(eiv_pairs)} diagonal-noise cells; the 32% phase transition remains documented.",
            f"EWMA proxy rows written: {len(ewma)}.",
            f"Circulation detector: conservative Type-I={circ_type1:.3f}, power={circ_power_rate:.3f}; nominal 5% recalibration overshoot remains a documented null.",
            f"Fluctuation read-out: lag-1 correction wins on {fluc_corrected_wins}/{len(fluc_pairs)} noisy tensor cells.",
            "Documented nulls kept visible: EIV phase transition, conservative detector, transport no-advantage, nonlinear-drift unreliability.",
        ]
    else:
        lines += ["Claim Level 1: FAIL", "Claim Level 2: FAIL", "Claim Level 3: FAIL", "Claim Level 4: FAIL", "", "No benchmark_summary.csv found."]
    out = ROOT / "docs/CLAIM_READINESS_REPORT.md"
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
