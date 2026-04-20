#!/usr/bin/env python3
"""Summarize false-positive counts across experiment runs."""

import json
import pathlib
import statistics
import sys

_SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
_PARENT_DIR = _SCRIPT_DIR.parent
if str(_PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(_PARENT_DIR))

import analysis_script.common as common

# Main experiment velocity mapping (4 velocities x 10 repeats)
MAIN_VELOCITY_MAP = {}
for i in range(0, 10):
    MAIN_VELOCITY_MAP[i] = 0.5
for i in range(10, 20):
    MAIN_VELOCITY_MAP[i] = 1.0
for i in range(20, 30):
    MAIN_VELOCITY_MAP[i] = 2.0
for i in range(30, 40):
    MAIN_VELOCITY_MAP[i] = 5.0

# Ablation experiment variant mapping (5 variants x 10 repeats, all 1.0 mm/s)
ABLATION_VARIANT_MAP = {}
ABLATION_VARIANTS = [
    (0, 10, "no_cov_inflation"),
    (10, 20, "no_type_constraint"),
    (20, 30, "single_model_ekf"),
    (30, 40, "no_cusum_no_dir"),
    (40, 50, "no_drift_comp"),
]
for start, end, name in ABLATION_VARIANTS:
    for i in range(start, end):
        ABLATION_VARIANT_MAP[i] = name


def load_metrics(json_path):
    """Load paper_metrics.json and extract FP-related fields."""
    with open(json_path) as f:
        data = json.load(f)

    fc = data.get("F_c", {})
    pp = data.get("P_p", {})
    rr = data.get("R_r", {})

    return {
        "run_dir": data.get("run_dir", ""),
        # Track-level (F_c)
        "N_confirmed": fc.get("N_confirmed", 0),
        "N_false_tracks": fc.get("N_false", 0),
        "N_tp_tracks_fc": (fc.get("N_confirmed", 0) or 0) - (fc.get("N_false", 0) or 0),
        "F_c": fc.get("F_c"),
        # Zone-level (P_p)
        "N_zones": pp.get("N_zones", 0),
        "N_tp_zones": pp.get("N_tp_zones", 0),
        "N_fp_zones": (pp.get("N_zones", 0) or 0) - (pp.get("N_tp_zones", 0) or 0),
        "N_qualified": pp.get("N_qualified", 0),
        "N_tp_qualified": pp.get("N_tp", 0),
        "N_fp_qualified": (pp.get("N_qualified", 0) or 0) - (pp.get("N_tp", 0) or 0),
        "P_p": pp.get("P_p"),
        # R_r context
        "N_GT": rr.get("N_GT", 0),
        "N_matched": rr.get("N_matched", 0),
        "R_r": rr.get("R_r"),
    }


def extract_run_number(name):
    """Extract run number from directory name like 'sim_run_004' or '20260408_sim_run_004_xxxx'."""
    parts = name.split("_")
    for i, p in enumerate(parts):
        if p == "run" and i + 1 < len(parts):
            try:
                return int(parts[i + 1][:3])
            except ValueError:
                pass
    return None


def print_separator(char="=", width=120):
    print(char * width)


def print_group_stats(label, rows, fields):
    """Print summary stats for a group of rows."""
    if not rows:
        print(f"  {label}: NO DATA")
        return

    print(f"\n  {label} (n={len(rows)} runs)")
    print(f"  {'Metric':<30s} {'Mean':>8s} {'Std':>8s} {'Min':>6s} {'Med':>6s} {'Max':>6s} {'Sum':>6s}")
    print(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*6} {'-'*6} {'-'*6} {'-'*6}")

    for field, display_name in fields:
        vals = [r[field] for r in rows if r[field] is not None]
        if not vals:
            print(f"  {display_name:<30s} {'N/A':>8s}")
            continue
        mean = statistics.mean(vals)
        std = statistics.stdev(vals) if len(vals) > 1 else 0.0
        med = statistics.median(vals)
        print(f"  {display_name:<30s} {mean:8.2f} {std:8.2f} {min(vals):6.0f} {med:6.1f} {max(vals):6.0f} {sum(vals):6.0f}")


def print_per_run_table(rows, extra_col_name="Group", extra_col_key="group"):
    """Print per-run detail table."""
    header = (
        f"  {'Run':<16s} {extra_col_name:<20s} "
        f"{'N_conf':>6s} {'FP_trk':>6s} {'TP_trk':>6s} {'F_c':>6s} "
        f"{'N_zone':>6s} {'FP_zn':>5s} {'TP_zn':>5s} {'P_p':>6s} "
        f"{'N_GT':>4s} {'R_r':>5s}"
    )
    print(header)
    print(f"  {'-'*len(header)}")

    for r in rows:
        fc_str = f"{r['F_c']:.3f}" if r['F_c'] is not None else "N/A"
        pp_str = f"{r['P_p']:.3f}" if r['P_p'] is not None else "N/A"
        rr_str = f"{r['R_r']:.2f}" if r['R_r'] is not None else "N/A"
        print(
            f"  {r['run_name']:<16s} {r[extra_col_key]:<20s} "
            f"{r['N_confirmed']:6d} {r['N_false_tracks']:6d} {r['N_tp_tracks_fc']:6d} {fc_str:>6s} "
            f"{r['N_zones']:6d} {r['N_fp_zones']:5d} {r['N_tp_zones']:5d} {pp_str:>6s} "
            f"{r['N_GT']:4d} {rr_str:>5s}"
        )


def main():
    try:
        main_result_dir = common.latest_result_date_dir()
    except FileNotFoundError:
        main_result_dir = common.result_root()
    ablation_result_dir = common.result_root() / "external_runs"

    all_main_rows = []
    all_ablation_rows = []
    missing = []

    # -----------------------------------------------------------------------
    # Main experiment: result/20260408/sim_run_NNN/paper_metrics.json
    # -----------------------------------------------------------------------
    for run_idx in range(40):
        run_name = f"sim_run_{run_idx:03d}"
        json_path = main_result_dir / run_name / "paper_metrics.json"
        if not json_path.exists():
            missing.append(("main", run_name))
            continue
        row = load_metrics(json_path)
        row["run_name"] = run_name
        row["run_idx"] = run_idx
        row["velocity"] = MAIN_VELOCITY_MAP[run_idx]
        row["group"] = f"{MAIN_VELOCITY_MAP[run_idx]} mm/s"
        all_main_rows.append(row)

    # -----------------------------------------------------------------------
    # Ablation experiment: result/external_runs/20260408_sim_run_NNN_*/paper_metrics.json
    # -----------------------------------------------------------------------
    if not ablation_result_dir.is_dir():
        ablation_entries = []
    else:
        ablation_entries = sorted(ablation_result_dir.iterdir())
    for entry in ablation_entries:
        if not entry.is_dir():
            continue
        run_num = extract_run_number(entry.name)
        if run_num is None:
            continue
        json_path = entry / "paper_metrics.json"
        if not json_path.exists():
            missing.append(("ablation", entry.name))
            continue
        # Verify it is actually ablation data
        row = load_metrics(json_path)
        if "output_ablation" not in row.get("run_dir", ""):
            continue  # skip if not ablation
        row["run_name"] = f"sim_run_{run_num:03d}"
        row["run_idx"] = run_num
        row["variant"] = ABLATION_VARIANT_MAP.get(run_num, "unknown")
        row["group"] = row["variant"]
        all_ablation_rows.append(row)

    # -----------------------------------------------------------------------
    # Report
    # -----------------------------------------------------------------------
    print_separator("=")
    print("FALSE-POSITIVE BURDEN ANALYSIS — ALL EXPERIMENT RUNS")
    print_separator("=")

    if missing:
        print(f"\nWARNING: {len(missing)} missing paper_metrics.json files:")
        for exp, name in missing:
            print(f"  [{exp}] {name}")

    # -- Key fields for summary --
    fp_fields = [
        ("N_confirmed", "Confirmed tracks (N_confirmed)"),
        ("N_false_tracks", "FP tracks (N_false)"),
        ("N_tp_tracks_fc", "TP tracks"),
        ("N_zones", "Risk zones (N_zones)"),
        ("N_fp_zones", "FP zones"),
        ("N_tp_zones", "TP zones"),
        ("N_qualified", "Qualified tracks"),
        ("N_fp_qualified", "FP qualified tracks"),
        ("N_tp_qualified", "TP qualified tracks"),
    ]

    # =======================================================================
    # MAIN EXPERIMENT
    # =======================================================================
    print_separator("=")
    print("MAIN EXPERIMENT (40 runs, 4 velocities x 10 repeats)")
    print_separator("=")

    # Per-run table
    print("\nPer-run detail:")
    print_per_run_table(all_main_rows, "Velocity", "group")

    # Group by velocity
    for vel in [0.5, 1.0, 2.0, 5.0]:
        group = [r for r in all_main_rows if r["velocity"] == vel]
        print_group_stats(f"Velocity = {vel} mm/s", group, fp_fields)

    # Overall main
    print_group_stats("ALL MAIN (combined)", all_main_rows, fp_fields)

    # =======================================================================
    # ABLATION EXPERIMENT
    # =======================================================================
    print_separator("=")
    print("ABLATION EXPERIMENT (50 runs, 5 variants x 10 repeats, all 1.0 mm/s)")
    print_separator("=")

    # Per-run table
    print("\nPer-run detail:")
    print_per_run_table(all_ablation_rows, "Variant", "group")

    # Group by variant
    variant_order = ["no_cov_inflation", "no_type_constraint", "single_model_ekf",
                     "no_cusum_no_dir", "no_drift_comp"]
    for variant in variant_order:
        group = [r for r in all_ablation_rows if r.get("variant") == variant]
        print_group_stats(f"Variant: {variant}", group, fp_fields)

    # Overall ablation
    print_group_stats("ALL ABLATION (combined)", all_ablation_rows, fp_fields)

    # =======================================================================
    # CROSS-COMPARISON: Main 1.0 mm/s vs Ablation variants
    # =======================================================================
    print_separator("=")
    print("CROSS-COMPARISON: Main 1.0 mm/s (full system) vs Ablation variants")
    print_separator("=")
    main_1ms = [r for r in all_main_rows if r["velocity"] == 1.0]

    comparison_fields = [
        ("N_false_tracks", "FP tracks (mean)"),
        ("N_confirmed", "Confirmed tracks (mean)"),
        ("N_fp_zones", "FP zones (mean)"),
        ("N_zones", "Zones (mean)"),
    ]

    def group_means(rows, field):
        vals = [r[field] for r in rows if r[field] is not None]
        return statistics.mean(vals) if vals else float("nan")

    print(f"\n  {'Group':<25s}", end="")
    for _, label in comparison_fields:
        print(f" {label:>22s}", end="")
    print(f" {'n':>4s}")

    print(f"  {'-'*25}", end="")
    for _ in comparison_fields:
        print(f" {'-'*22}", end="")
    print(f" {'-'*4}")

    # Main full system at 1.0 mm/s
    print(f"  {'Full system (1.0mm/s)':<25s}", end="")
    for field, _ in comparison_fields:
        print(f" {group_means(main_1ms, field):22.2f}", end="")
    print(f" {len(main_1ms):4d}")

    # Each ablation variant
    for variant in variant_order:
        group = [r for r in all_ablation_rows if r.get("variant") == variant]
        print(f"  {variant:<25s}", end="")
        for field, _ in comparison_fields:
            print(f" {group_means(group, field):22.2f}", end="")
        print(f" {len(group):4d}")

    # =======================================================================
    # RAW FP COUNTS SUMMARY TABLE
    # =======================================================================
    print_separator("=")
    print("RAW FP COUNTS PER RUN (sorted by FP zone count, descending)")
    print_separator("=")

    all_rows = []
    for r in all_main_rows:
        r2 = dict(r)
        r2["experiment"] = f"main/{r['group']}"
        all_rows.append(r2)
    for r in all_ablation_rows:
        r2 = dict(r)
        r2["experiment"] = f"ablation/{r['variant']}"
        all_rows.append(r2)

    all_rows.sort(key=lambda r: r["N_fp_zones"], reverse=True)

    print(f"\n  {'Run':<16s} {'Experiment':<28s} {'FP_zones':>8s} {'Tot_zones':>9s} {'FP_tracks':>9s} {'Tot_conf':>8s} {'R_r':>5s}")
    print(f"  {'-'*16} {'-'*28} {'-'*8} {'-'*9} {'-'*9} {'-'*8} {'-'*5}")
    for r in all_rows:
        rr_str = f"{r['R_r']:.2f}" if r['R_r'] is not None else "N/A"
        print(
            f"  {r['run_name']:<16s} {r['experiment']:<28s} "
            f"{r['N_fp_zones']:8d} {r['N_zones']:9d} "
            f"{r['N_false_tracks']:9d} {r['N_confirmed']:8d} {rr_str:>5s}"
        )

    print(f"\n  Total runs processed: {len(all_rows)} (main={len(all_main_rows)}, ablation={len(all_ablation_rows)})")
    print()


if __name__ == "__main__":
    main()
