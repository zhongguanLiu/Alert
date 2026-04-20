#!/usr/bin/env python3
"""Compute minimum detectable displacement from multiple runs."""

import argparse
import csv
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import common
import compute_metrics


def _analysis_controlled_object_names(run_dir):
    controlled_names = common.get_analysis_controlled_object_names(run_dir)
    if controlled_names:
        return controlled_names
    controlled_name = common.get_analysis_controlled_object_name(run_dir)
    return [controlled_name] if controlled_name else [None]


def sweep_mdd(run_dirs, match_radius=common.MATCH_RADIUS):
    """Run metrics on each directory and compile MDD table."""
    rows = []

    for run_dir in sorted(run_dirs):
        run_dir = pathlib.Path(run_dir)
        controlled_names = _analysis_controlled_object_names(run_dir)
        velocities = [
            common.get_injection_velocity(run_dir, controlled_name)
            for controlled_name in controlled_names
        ]
        if not any(velocity is not None for velocity in velocities):
            print(f"[compute_mdd] Skipping {run_dir.name}: no scenario manifest")
            continue

        print(f"\n{'='*60}")
        print(f"[compute_mdd] Processing {run_dir.name} (objects={controlled_names})")
        print(f"{'='*60}")

        try:
            results = compute_metrics.run_metrics(run_dir, match_radius=match_radius)
        except Exception as e:
            print(f"[compute_mdd] Error processing {run_dir}: {e}")
            for controlled_name, velocity in zip(controlled_names, velocities):
                velocity_mmps = velocity * 1000.0 if velocity is not None else None
                rows.append({
                    "run_dir": run_dir.name,
                    "velocity_mmps": round(velocity_mmps, 2) if velocity_mmps is not None else None,
                    "controlled_object": controlled_name,
                    "detected": False,
                    "R_r": None,
                    "t_resp_s": None,
                    "gt_disp_at_detection_mm": None,
                    "error": str(e),
                })
            continue

        Rr = results.get("R_r", {}).get("R_r")
        t_resp_info = results.get("t_resp", {})
        rd = common.load_run_data(run_dir, load_clusters=False)
        gt_by_name = {obj.name: obj for obj in rd.gt_objects}

        for controlled_name, velocity in zip(controlled_names, velocities):
            velocity_mmps = velocity * 1000.0 if velocity is not None else None
            gt_disp_at_detection = None
            obj_t_resp = None

            for per_obj in t_resp_info.get("per_object", []):
                obj_name = per_obj.get("object")
                if controlled_name and obj_name != controlled_name:
                    continue
                obj_t_resp = per_obj.get("t_resp")
                t_det = per_obj.get("t_first_confirmed")
                if t_det is not None and obj_name:
                    obj = gt_by_name.get(obj_name)
                    if obj is not None:
                        d = common.gt_displacement_at_time(obj, t_det)
                        if d is not None:
                            gt_disp_at_detection = d
                break

            rows.append({
                "run_dir": run_dir.name,
                "velocity_mmps": round(velocity_mmps, 2) if velocity_mmps is not None else None,
                "controlled_object": controlled_name,
                "detected": obj_t_resp is not None,
                "R_r": Rr,
                "t_resp_s": round(obj_t_resp, 2) if obj_t_resp is not None else None,
                "gt_disp_at_detection_mm": round(gt_disp_at_detection * 1000, 2)
                    if gt_disp_at_detection is not None else None,
            })

    return rows


def print_mdd_table(rows):
    """Print the MDD sweep table."""
    def sort_key(row):
        velocity = row.get("velocity_mmps")
        return (
            velocity is None,
            float(velocity) if velocity is not None else float("inf"),
            row.get("run_dir") or "",
            row.get("controlled_object") or "",
        )

    print(f"\n{'='*80}")
    print("MDD Sweep Summary")
    print(f"{'='*80}")
    print(f"  {'Velocity (mm/s)':<18s} {'Detected?':<12s} {'R_r':<8s} "
          f"{'t_resp (s)':<12s} {'GT disp (mm)':<14s}")
    print(f"  {'-'*18} {'-'*12} {'-'*8} {'-'*12} {'-'*14}")

    for row in sorted(rows, key=sort_key):
        vel = f"{row['velocity_mmps']:.1f}" if row['velocity_mmps'] is not None else "N/A"
        det = "Yes" if row.get("detected") else "No"
        rr = f"{row['R_r']:.2f}" if row.get("R_r") is not None else "N/A"
        tr = f"{row['t_resp_s']:.1f}" if row.get("t_resp_s") is not None else "---"
        gd = f"{row['gt_disp_at_detection_mm']:.1f}" if row.get("gt_disp_at_detection_mm") is not None else "---"
        print(f"  {vel:<18s} {det:<12s} {rr:<8s} {tr:<12s} {gd:<14s}")

    # Determine MDD
    detected_rows = [r for r in rows if r.get("detected") and r.get("gt_disp_at_detection_mm") is not None]
    if detected_rows:
        mdd = min(r["gt_disp_at_detection_mm"] for r in detected_rows)
        print(f"\n  MDD = {mdd:.1f} mm (smallest GT displacement at confirmed detection)")
    else:
        print("\n  MDD = N/A (no confirmed detections across sweep)")


def main():
    parser = argparse.ArgumentParser(description="MDD sweep across multiple runs")
    parser.add_argument("--output-root", type=str, help="Root output directory")
    parser.add_argument("--run-dirs", nargs="+", type=str, help="Explicit list of run directories")
    parser.add_argument("--match-radius", type=float, default=common.MATCH_RADIUS)
    args = parser.parse_args()

    if args.run_dirs:
        run_dirs = [pathlib.Path(d) for d in args.run_dirs]
    elif args.output_root:
        run_dirs = common.find_all_runs(args.output_root)
    else:
        run_dirs = common.find_all_runs()

    if not run_dirs:
        print("[compute_mdd] No run directories found.")
        sys.exit(1)

    print(f"[compute_mdd] Found {len(run_dirs)} run(s)")
    rows = sweep_mdd(run_dirs, match_radius=args.match_radius)
    print_mdd_table(rows)

    # Write CSV
    out_path = common.result_root() / "mdd_summary.csv"
    fieldnames = ["run_dir", "velocity_mmps", "controlled_object", "detected", "R_r", "t_resp_s", "gt_disp_at_detection_mm"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[compute_mdd] CSV written to: {out_path}")


if __name__ == "__main__":
    main()
