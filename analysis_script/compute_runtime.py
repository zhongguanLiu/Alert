#!/usr/bin/env python3
"""Compute per-stage runtime statistics for a run."""

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import common


STAGE_KEYS = [
    ("stage_a_ms", "Stage A: Observation model & temporal fusion"),
    ("stage_b_ms", "Stage B: IMM information filter"),
    ("stage_c_ms", "Stage C: Evidence fusion cascade"),
    ("stage_d_ms", "Stage D: Drift comp. & risk aggregation"),
    ("total_ms",   "Total per cycle"),
]


WARMUP_CYCLES = 5  # Skip first N cycles (initialization overhead)


def compute_runtime_stats(run_dir):
    """Load stage_runtime.jsonl and compute statistics (excluding warmup)."""
    runtime_path = pathlib.Path(run_dir) / "runtime" / "stage_runtime.jsonl"
    records = common.load_jsonl(runtime_path)

    if not records:
        print(f"[compute_runtime] No runtime data found at {runtime_path}")
        return None

    total_loaded = len(records)
    if len(records) > WARMUP_CYCLES:
        records = records[WARMUP_CYCLES:]

    print(f"[compute_runtime] Run: {run_dir}")
    print(f"[compute_runtime] Loaded {total_loaded} records, using {len(records)} (skipped {WARMUP_CYCLES} warmup)")
    print()

    results = {"run_dir": str(run_dir), "n_cycles_total": total_loaded,
               "n_cycles_used": len(records), "warmup_skipped": WARMUP_CYCLES, "stages": {}}

    # Print table header
    print(f"  {'Stage':<50s} {'Mean':>8s} {'Std':>8s} {'Med':>8s} {'Max':>8s}")
    print(f"  {'-'*50} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    for key, label in STAGE_KEYS:
        values = [float(r.get(key, 0.0)) for r in records if key in r]
        if not values:
            print(f"  {label:<50s} {'N/A':>8s} {'N/A':>8s} {'N/A':>8s} {'N/A':>8s}")
            continue

        arr = np.array(values)
        stats = {
            "mean_ms": round(float(np.mean(arr)), 3),
            "std_ms": round(float(np.std(arr)), 3),
            "median_ms": round(float(np.median(arr)), 3),
            "max_ms": round(float(np.max(arr)), 3),
            "min_ms": round(float(np.min(arr)), 3),
            "n": len(values),
        }
        results["stages"][key] = stats

        print(f"  {label:<50s} {stats['mean_ms']:>7.1f} {stats['std_ms']:>7.1f} "
              f"{stats['median_ms']:>7.1f} {stats['max_ms']:>7.1f}")

    # Write output
    out_dir = common.result_dir_for_run(pathlib.Path(run_dir))
    output_path = out_dir / "runtime_summary.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[compute_runtime] Results written to: {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Compute per-stage runtime statistics")
    parser.add_argument("--run-dir", type=str, help="Path to sim_run_NNN directory")
    parser.add_argument("--latest", action="store_true", help="Auto-select latest run")
    parser.add_argument("--output-root", type=str, help="Override output root directory")
    args = parser.parse_args()

    run_dir = common.resolve_run_dir(
        run_dir=args.run_dir,
        output_root=args.output_root,
        latest=args.latest,
    )
    compute_runtime_stats(run_dir)


if __name__ == "__main__":
    main()
