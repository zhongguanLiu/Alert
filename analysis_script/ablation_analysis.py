#!/usr/bin/env python3
"""Analyze ablation-study results across experiment variants."""

import csv
import json
import os
import sys
from pathlib import Path
from collections import defaultdict
import math

_SCRIPT_DIR = Path(__file__).resolve().parent
_PARENT_DIR = _SCRIPT_DIR.parent
if str(_PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(_PARENT_DIR))

import analysis_script.common as common

RESULT_BASE = common.result_root()
CSV_PATH = RESULT_BASE / "summary" / "per_run_metrics.csv"

# Variant name mapping from scenario_id keywords
VARIANT_MAP = {
    "no_cov_inflation": "(a) no_cov_inflation",
    "no_type_constraint": "(b) no_type_constraint",
    "single_model_ekf": "(c) single_model_ekf",
    "no_cusum_no_dir": "(d) no_cusum_no_dir",
    "no_drift_comp": "(e) no_drift_comp",
}

VARIANT_ORDER = [
    "Full pipeline (1.0 mm/s)",
    "(a) no_cov_inflation",
    "(b) no_type_constraint",
    "(c) single_model_ekf",
    "(d) no_cusum_no_dir",
    "(e) no_drift_comp",
]


def classify_variant(scenario_id: str) -> str:
    """Map scenario_id to variant label."""
    for key, label in VARIANT_MAP.items():
        if key in scenario_id:
            return label
    return "unknown"


def load_full_pipeline_from_json():
    """Load full pipeline runs 010-019 from paper_metrics.json files."""
    rows = []
    for i in range(10, 20):
        json_path = RESULT_BASE / f"sim_run_{i:03d}" / "paper_metrics.json"
        if not json_path.exists():
            print(f"WARNING: missing {json_path}", file=sys.stderr)
            continue

        with open(json_path) as f:
            data = json.load(f)

        # Extract per-object data
        rr_details = {d["object"]: d for d in data["R_r"]["details"]}
        t_resp_per_obj = {d["object"]: d for d in data["t_resp"]["per_object"]}
        eps_per_obj = data["epsilon_d"]["per_object"]

        for obj_name in ["model_01", "model_02"]:
            rr_d = rr_details.get(obj_name, {})
            detected = rr_d.get("matched", False)

            row = {
                "run_name": f"sim_run_{i:03d}",
                "variant": "Full pipeline (1.0 mm/s)",
                "controlled_object": obj_name,
                "detected": detected,
                "t_resp_s": None,
                "epsilon_d": None,
                "beta_d": None,
                "gt_disp_at_detection_mm": None,
            }

            if detected:
                t_obj = t_resp_per_obj.get(obj_name, {})
                e_obj = eps_per_obj.get(obj_name, {})
                row["t_resp_s"] = t_obj.get("t_resp")
                row["epsilon_d"] = e_obj.get("epsilon_d")
                # d_gt is in meters in JSON -> convert to mm
                d_gt_m = e_obj.get("d_gt_m")
                if d_gt_m is not None:
                    row["gt_disp_at_detection_mm"] = d_gt_m * 1000.0
                # beta_d per-object
                beta_obj = data["beta_d"]["per_object"].get(obj_name, {})
                row["beta_d"] = beta_obj.get("mean_bias")

            rows.append(row)
    return rows


def load_ablation_from_csv():
    """Load ablation runs from per_run_metrics.csv."""
    rows = []
    with open(CSV_PATH) as f:
        reader = csv.DictReader(f)
        for r in reader:
            variant = classify_variant(r["scenario_id"])
            t_resp = r["t_resp_s"].strip()
            detected = t_resp != ""

            row = {
                "run_name": r["run_name"],
                "variant": variant,
                "controlled_object": r["controlled_object"],
                "detected": detected,
                "t_resp_s": float(t_resp) if detected else None,
                "epsilon_d": float(r["epsilon_d"]) if detected and r["epsilon_d"].strip() else None,
                "beta_d": float(r["beta_d"]) if detected and r["beta_d"].strip() else None,
                "gt_disp_at_detection_mm": (
                    float(r["gt_disp_at_detection_mm"])
                    if detected and r["gt_disp_at_detection_mm"].strip()
                    else None
                ),
            }
            rows.append(row)
    return rows


def compute_stats(rows):
    """Compute aggregate statistics for a list of per-object rows."""
    n_total = len(rows)
    detected_rows = [r for r in rows if r["detected"]]
    n_detected = len(detected_rows)

    def safe_mean(vals):
        vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else None

    return {
        "n_total": n_total,
        "n_detected": n_detected,
        "R_r": n_detected / n_total if n_total > 0 else 0.0,
        "mean_t_resp": safe_mean([r["t_resp_s"] for r in detected_rows]),
        "mean_eps_d": safe_mean([r["epsilon_d"] for r in detected_rows]),
        "mean_abs_eps_d": safe_mean(
            [abs(r["epsilon_d"]) for r in detected_rows if r["epsilon_d"] is not None]
        ),
        "mean_d_first": safe_mean([r["gt_disp_at_detection_mm"] for r in detected_rows]),
        "mean_beta_d": safe_mean([r["beta_d"] for r in detected_rows]),
    }


def fmt(val, decimals=3, suffix=""):
    if val is None:
        return "  --  "
    return f"{val:.{decimals}f}{suffix}"


def main():
    try:
        result_base = common.latest_result_date_dir()
    except FileNotFoundError:
        result_base = common.result_root()
    csv_path = result_base / "summary" / "per_run_metrics.csv"

    # Load data
    global RESULT_BASE, CSV_PATH
    RESULT_BASE = result_base
    CSV_PATH = csv_path
    full_rows = load_full_pipeline_from_json()
    ablation_rows = load_ablation_from_csv()
    all_rows = full_rows + ablation_rows

    # Group by variant
    by_variant = defaultdict(list)
    for r in all_rows:
        by_variant[r["variant"]].append(r)

    # =========================================================================
    # Print per-variant, per-object breakdown
    # =========================================================================
    print("=" * 120)
    print("ABLATION STUDY -- PER-VARIANT, PER-OBJECT BREAKDOWN")
    print("=" * 120)

    for variant in VARIANT_ORDER:
        vrows = by_variant.get(variant, [])
        if not vrows:
            print(f"\n--- {variant}: NO DATA ---")
            continue

        m1 = [r for r in vrows if r["controlled_object"] == "model_01"]
        m2 = [r for r in vrows if r["controlled_object"] == "model_02"]
        combined = vrows

        s_m1 = compute_stats(m1)
        s_m2 = compute_stats(m2)
        s_all = compute_stats(combined)

        n_runs = max(len(m1), len(m2))

        print(f"\n{'─' * 120}")
        print(f"  {variant}   ({n_runs} runs, {s_all['n_total']} object-instances)")
        print(f"{'─' * 120}")

        print(f"  {'':30s} {'Det. rate':>12s} {'t_resp (s)':>12s} {'eps_d':>12s} "
              f"{'|eps_d|':>12s} {'d_first (mm)':>14s} {'beta_d':>12s}")

        for label, s, n in [("model_01", s_m1, len(m1)),
                            ("model_02", s_m2, len(m2)),
                            ("Combined", s_all, len(combined))]:
            det_str = f"{s['n_detected']}/{n}" if label != "Combined" else f"{s['n_detected']}/{s['n_total']}"
            pct = f" ({s['R_r']*100:.0f}%)"
            print(f"  {label:30s} {det_str + pct:>12s} {fmt(s['mean_t_resp']):>12s} "
                  f"{fmt(s['mean_eps_d'], 4):>12s} {fmt(s['mean_abs_eps_d'], 4):>12s} "
                  f"{fmt(s['mean_d_first'], 2):>14s} {fmt(s['mean_beta_d'], 4):>12s}")

    # =========================================================================
    # Summary comparison table
    # =========================================================================
    print("\n\n")
    print("=" * 140)
    print("SUMMARY COMPARISON TABLE")
    print("=" * 140)

    header = (
        f"{'Variant':36s} "
        f"{'R_r':>6s} "
        f"{'m01 det':>8s} "
        f"{'m02 det':>8s} "
        f"{'t_resp':>8s} "
        f"{'t_m01':>8s} "
        f"{'t_m02':>8s} "
        f"{'eps_d':>8s} "
        f"{'|eps_d|':>8s} "
        f"{'e_m01':>8s} "
        f"{'e_m02':>8s} "
        f"{'d_first':>8s} "
        f"{'d_m01':>8s} "
        f"{'d_m02':>8s} "
    )
    print(header)
    print("-" * 140)

    for variant in VARIANT_ORDER:
        vrows = by_variant.get(variant, [])
        if not vrows:
            continue
        m1 = [r for r in vrows if r["controlled_object"] == "model_01"]
        m2 = [r for r in vrows if r["controlled_object"] == "model_02"]
        s_m1 = compute_stats(m1)
        s_m2 = compute_stats(m2)
        s_all = compute_stats(vrows)

        n = max(len(m1), len(m2))
        line = (
            f"{variant:36s} "
            f"{s_all['R_r']:6.2f} "
            f"{s_m1['n_detected']:>2d}/{n:<2d}   "
            f"{s_m2['n_detected']:>2d}/{n:<2d}   "
            f"{fmt(s_all['mean_t_resp'], 2):>8s} "
            f"{fmt(s_m1['mean_t_resp'], 2):>8s} "
            f"{fmt(s_m2['mean_t_resp'], 2):>8s} "
            f"{fmt(s_all['mean_eps_d'], 4):>8s} "
            f"{fmt(s_all['mean_abs_eps_d'], 4):>8s} "
            f"{fmt(s_m1['mean_eps_d'], 4):>8s} "
            f"{fmt(s_m2['mean_eps_d'], 4):>8s} "
            f"{fmt(s_all['mean_d_first'], 2):>8s} "
            f"{fmt(s_m1['mean_d_first'], 2):>8s} "
            f"{fmt(s_m2['mean_d_first'], 2):>8s} "
        )
        print(line)

    print("-" * 140)

    # =========================================================================
    # Compact LaTeX-ready table
    # =========================================================================
    print("\n\n")
    print("=" * 100)
    print("COMPACT TABLE")
    print("=" * 100)
    print(f"{'Variant':36s} {'R_r':>6s} {'t_resp':>8s} {'|eps_d|':>8s} {'d_first':>10s} {'m01 det':>8s} {'m02 det':>8s}")
    print("-" * 100)
    for variant in VARIANT_ORDER:
        vrows = by_variant.get(variant, [])
        if not vrows:
            continue
        m1 = [r for r in vrows if r["controlled_object"] == "model_01"]
        m2 = [r for r in vrows if r["controlled_object"] == "model_02"]
        s_m1 = compute_stats(m1)
        s_m2 = compute_stats(m2)
        s_all = compute_stats(vrows)
        n = max(len(m1), len(m2))

        print(
            f"{variant:36s} "
            f"{s_all['R_r']:6.2f} "
            f"{fmt(s_all['mean_t_resp'], 2):>8s} "
            f"{fmt(s_all['mean_abs_eps_d'], 4):>8s} "
            f"{fmt(s_all['mean_d_first'], 2):>10s} "
            f"{s_m1['n_detected']:>2d}/{n:<2d}   "
            f"{s_m2['n_detected']:>2d}/{n:<2d}   "
        )
    print("-" * 100)

    # =========================================================================
    # Delta from full pipeline
    # =========================================================================
    print("\n\n")
    print("=" * 100)
    print("DELTA FROM FULL PIPELINE (positive = worse for t_resp/|eps_d|/d_first, negative = better)")
    print("=" * 100)

    ref_rows = by_variant.get("Full pipeline (1.0 mm/s)", [])
    if not ref_rows:
        print("No full pipeline data found!")
        return
    ref = compute_stats(ref_rows)

    print(f"{'Variant':36s} {'dR_r':>8s} {'dt_resp':>8s} {'d|eps_d|':>9s} {'dd_first':>10s}")
    print("-" * 100)
    for variant in VARIANT_ORDER[1:]:
        vrows = by_variant.get(variant, [])
        if not vrows:
            continue
        s = compute_stats(vrows)

        def delta(a, b):
            if a is None or b is None:
                return "  --  "
            d = a - b
            return f"{d:+.3f}"

        print(
            f"{variant:36s} "
            f"{delta(s['R_r'], ref['R_r']):>8s} "
            f"{delta(s['mean_t_resp'], ref['mean_t_resp']):>8s} "
            f"{delta(s['mean_abs_eps_d'], ref['mean_abs_eps_d']):>9s} "
            f"{delta(s['mean_d_first'], ref['mean_d_first']):>10s} "
        )
    print("-" * 100)


if __name__ == "__main__":
    main()
