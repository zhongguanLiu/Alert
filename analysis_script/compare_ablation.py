#!/usr/bin/env python3
"""Aggregate ablation results into a summary table."""

import argparse
import csv
import pathlib
import sys
from collections import defaultdict

_THIS_DIR = pathlib.Path(__file__).resolve().parent
_PARENT_DIR = _THIS_DIR.parent
if str(_PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(_PARENT_DIR))

import analysis_script.common as common
import analysis_script.compute_metrics as compute_metrics


VARIANT_LABEL_MAP = {
    "full_pipeline": "Full pipeline",
    "no_cov_inflation": "(a) w/o cov. inflation",
    "no_type_constraint": "(b) w/o type-constraint",
    "single_model_ekf": "(c) Single-model EKF",
    "no_cusum": "(d) w/o CUSUM & dir. acc.",
    "no_drift_compensation": "(e) w/o drift comp.",
}


def analysis_controlled_object_names(run_dir):
    names = common.get_analysis_controlled_object_names(pathlib.Path(run_dir))
    if names:
        return names
    name = common.get_analysis_controlled_object_name(pathlib.Path(run_dir))
    return [name] if name else [None]


def _per_object_t_resp(metrics, controlled_object):
    if controlled_object is None:
        return metrics.get("t_resp", {}).get("mean_t_resp")
    for row in metrics.get("t_resp", {}).get("per_object", []):
        if row.get("object") == controlled_object:
            return row.get("t_resp")
    return None


def _per_object_rr(metrics, controlled_object):
    rr = metrics.get("R_r", {})
    if controlled_object is None:
        n_gt = int(rr.get("N_GT") or 0)
        n_matched = int(rr.get("N_matched") or 0)
        return rr.get("R_r"), n_gt, n_matched
    for row in rr.get("details", []):
        if row.get("object") != controlled_object:
            continue
        matched = bool(row.get("matched"))
        return (1.0 if matched else 0.0), 1, (1 if matched else 0)
    return None, 0, 0


def _per_object_fc(metrics, controlled_object):
    if controlled_object is not None:
        return None, 0, 0
    fc = metrics.get("F_c", {})
    return fc.get("F_c"), int(fc.get("N_confirmed") or 0), int(fc.get("N_false") or 0)


def _per_object_pp(metrics, controlled_object):
    if controlled_object is not None:
        return None, 0, 0, None, 0, 0
    pp = metrics.get("P_p", {})
    return (
        pp.get("P_p"),
        int(pp.get("N_zones") or 0),
        int(pp.get("N_tp_zones") or 0),
        pp.get("P_p_track"),
        int(pp.get("N_qualified") or 0),
        int(pp.get("N_tp") or 0),
    )


def _per_object_beta(metrics, controlled_object):
    beta = metrics.get("beta_d", {})
    if controlled_object is None:
        value = beta.get("beta_d")
        samples = beta.get("N_samples") or 0
        return value, int(samples)
    info = beta.get("per_object", {}).get(controlled_object)
    if not isinstance(info, dict):
        return None, 0
    return info.get("mean_bias"), int(info.get("n_samples") or 0)


def _per_object_epsilon(metrics, controlled_object):
    epsilon = metrics.get("epsilon_d", {})
    if controlled_object is None:
        return epsilon.get("epsilon_d")
    info = epsilon.get("per_object", {}).get(controlled_object)
    if not isinstance(info, dict):
        return None
    return info.get("epsilon_d")


def identify_variant(run_dir):
    """Determine variant name from ablation manifest."""
    manifest = common.load_json(pathlib.Path(run_dir) / "meta" / "ablation_manifest.json")
    if not manifest:
        return "unknown"

    variant = manifest.get("variant", "")
    if variant and variant != "unknown":
        return variant

    eff = manifest.get("effective_runtime", {})
    if eff.get("disable_covariance_inflation") or eff.get("covariance_alpha_xi") == 1.0:
        return "no_cov_inflation"
    if not eff.get("imm_enable_type_constraint", True):
        return "no_type_constraint"
    if not eff.get("imm_enable_model_competition", True):
        return "single_model_ekf"
    if not eff.get("significance_enable_cusum", True):
        return "no_cusum"
    if not eff.get("background_bias_enable", True):
        return "no_drift_compensation"
    if not eff.get("directional_motion_enable", True):
        return "no_cusum"

    return "full_pipeline"


def gather_run_dirs(run_dirs=None, output_root=None, full_pipeline_root=None, ablation_root=None):
    gathered = []
    if run_dirs:
        return [pathlib.Path(item) for item in run_dirs]

    if full_pipeline_root:
        gathered.extend(common.find_all_runs(full_pipeline_root))
    if ablation_root:
        gathered.extend(common.find_all_runs(ablation_root))
    if not gathered and output_root:
        gathered.extend(common.find_all_runs(output_root))
    if not gathered and not any([output_root, full_pipeline_root, ablation_root]):
        gathered.extend(common.find_all_runs())

    # Stable unique order
    unique = []
    seen = set()
    for path in sorted(pathlib.Path(p) for p in gathered):
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)
    return unique


def collect_per_run_rows(run_dirs, match_radius=common.MATCH_RADIUS):
    rows = []
    for run_dir in run_dirs:
        run_dir = pathlib.Path(run_dir)
        variant = identify_variant(run_dir)
        label = VARIANT_LABEL_MAP.get(variant, variant)
        controlled_objects = analysis_controlled_object_names(run_dir)

        print(f"\n{'='*60}")
        print(f"[compare_ablation] {run_dir.name} -> variant: {label}")
        print(f"{'='*60}")

        try:
            results = compute_metrics.run_metrics(run_dir, match_radius=match_radius)
        except Exception as exc:
            print(f"[compare_ablation] Error: {exc}")
            for controlled_object in controlled_objects:
                rows.append(
                    {
                        "variant": variant,
                        "label": label,
                        "controlled_object": controlled_object,
                        "run_dir": run_dir.name,
                        "run_path": str(run_dir),
                        "metrics": None,
                        "t_resp_s": None,
                        "beta_d": None,
                        "N_beta_samples": 0,
                        "epsilon_d": None,
                        "error": str(exc),
                    }
                )
            continue

        for controlled_object in controlled_objects:
            rr_value, n_gt, n_matched = _per_object_rr(results, controlled_object)
            fc_value, n_confirmed, n_false = _per_object_fc(results, controlled_object)
            pp_value, n_zones, n_tp_zones, pp_track_value, n_qualified, n_tp = _per_object_pp(
                results, controlled_object
            )
            beta_d, n_beta_samples = _per_object_beta(results, controlled_object)
            rows.append(
                {
                    "variant": variant,
                    "label": label,
                    "controlled_object": controlled_object,
                    "run_dir": run_dir.name,
                    "run_path": str(run_dir),
                    "metrics": results,
                    "R_r": rr_value,
                    "N_GT": n_gt,
                    "N_matched": n_matched,
                    "F_c": fc_value,
                    "N_confirmed": n_confirmed,
                    "N_false": n_false,
                    "P_p": pp_value,
                    "N_zones": n_zones,
                    "N_tp_zones": n_tp_zones,
                    "P_p_track": pp_track_value,
                    "N_qualified": n_qualified,
                    "N_tp": n_tp,
                    "t_resp_s": _per_object_t_resp(results, controlled_object),
                    "beta_d": beta_d,
                    "N_beta_samples": n_beta_samples,
                    "epsilon_d": _per_object_epsilon(results, controlled_object),
                    "error": "",
                }
            )
    return rows


def aggregate_variant_rows(per_run_rows):
    groups = defaultdict(list)
    for row in per_run_rows:
        groups[(row["variant"], row.get("controlled_object"))].append(row)

    rows = []
    for (variant, controlled_object), items in groups.items():
        label = VARIANT_LABEL_MAP.get(variant, variant)
        total_gt = 0
        total_matched = 0
        # Legacy F_c counters
        total_confirmed = 0
        total_false = 0
        # Zone-level P_p counters (primary)
        total_zones = 0
        total_tp_zones = 0
        # Track-level P_p counters (legacy)
        total_qualified = 0
        total_tp = 0
        t_resp_values = []
        beta_weighted_sum = 0.0
        beta_sample_count = 0
        epsilon_d_values = []
        successful_runs = 0
        fc_values = []
        pp_values = []
        pp_track_values = []

        for item in items:
            metrics = item.get("metrics")
            if not metrics:
                continue
            successful_runs += 1
            total_gt += int(item.get("N_GT") or 0)
            total_matched += int(item.get("N_matched") or 0)

            total_confirmed += int(item.get("N_confirmed") or 0)
            total_false += int(item.get("N_false") or 0)

            total_zones += int(item.get("N_zones") or 0)
            total_tp_zones += int(item.get("N_tp_zones") or 0)

            total_qualified += int(item.get("N_qualified") or 0)
            total_tp += int(item.get("N_tp") or 0)

            if item.get("F_c") is not None:
                fc_values.append(float(item["F_c"]))
            if item.get("P_p") is not None:
                pp_values.append(float(item["P_p"]))
            if item.get("P_p_track") is not None:
                pp_track_values.append(float(item["P_p_track"]))

            if item.get("t_resp_s") is not None:
                t_resp_values.append(float(item["t_resp_s"]))

            if item.get("beta_d") is not None and item.get("N_beta_samples"):
                beta_weighted_sum += float(item["beta_d"]) * int(item["N_beta_samples"])
                beta_sample_count += int(item["N_beta_samples"])

            if item.get("epsilon_d") is not None:
                epsilon_d_values.append(float(item["epsilon_d"]))

        rows.append(
            {
                "variant": variant,
                "label": label,
                "controlled_object": controlled_object,
                "n_runs": len(items),
                "n_successful_runs": successful_runs,
                "R_r": (total_matched / total_gt) if total_gt > 0 else None,
                "F_c": (total_false / total_confirmed) if fc_values and total_confirmed > 0 else None,
                "P_p": (total_tp_zones / total_zones) if pp_values and total_zones > 0 else None,
                "P_p_track": (total_tp / total_qualified) if pp_track_values and total_qualified > 0 else None,
                "t_resp_s": (sum(t_resp_values) / len(t_resp_values)) if t_resp_values else None,
                "beta_d": (beta_weighted_sum / beta_sample_count) if beta_sample_count > 0 else None,
                "epsilon_d": (sum(epsilon_d_values) / len(epsilon_d_values)) if epsilon_d_values else None,
                "N_GT": total_gt,
                "N_matched": total_matched,
                "N_confirmed": total_confirmed,
                "N_false": total_false,
                "N_zones": total_zones,
                "N_tp_zones": total_tp_zones,
                "N_qualified": total_qualified,
                "N_tp": total_tp,
                "N_beta_samples": beta_sample_count,
                "N_epsilon_d_samples": len(epsilon_d_values),
            }
        )

    def sort_key(row):
        if row["variant"] == "full_pipeline":
            return (0, row.get("controlled_object") or "")
        return (1, row["variant"], row.get("controlled_object") or "")

    return sorted(rows, key=sort_key)


def print_ablation_table(rows):
    print(f"\n{'='*110}")
    print("Ablation Comparison Table  [R_r, P_p, t_resp, epsilon_d]")
    print(f"{'='*110}")
    # Header: show zone-level P_p (primary) and track-level P_p_track (legacy) side by side
    print(
        f"  {'Variant':<35s} {'Object':<20s} {'Runs':>4s} {'R_r':>6s} "
        f"{'N_z':>5s} {'P_p_z':>7s} {'P_p_trk':>8s} "
        f"{'t_resp':>8s} {'eps_d':>8s} {'(F_c)':>7s}"
    )
    print(
        f"  {'-'*35} {'-'*20} {'-'*4} {'-'*6} {'-'*5} {'-'*7} "
        f"{'-'*8} {'-'*8} {'-'*8} {'-'*7}"
    )

    for row in rows:
        rr   = f"{row['R_r']:.3f}"      if row.get("R_r")       is not None else "N/A"
        nz   = f"{row['N_zones']}"       if row.get("N_zones")   is not None else "N/A"
        pp_z = f"{row['P_p']:.3f}"      if row.get("P_p")        is not None else "N/A"
        pp_t = f"{row['P_p_track']:.4f}" if row.get("P_p_track") is not None else "N/A"
        tr   = f"{row['t_resp_s']:.1f}" if row.get("t_resp_s")  is not None else "N/A"
        ed   = f"{row['epsilon_d']:.3f}" if row.get("epsilon_d") is not None else "N/A"
        fc   = f"{row['F_c']:.3f}"      if row.get("F_c")        is not None else "N/A"
        print(
            f"  {row['label']:<35s} {(row.get('controlled_object') or 'all'):<20s} {row['n_runs']:>4d} "
            f"{rr:>6s} {nz:>5s} {pp_z:>7s} {pp_t:>8s} "
            f"{tr:>8s} {ed:>8s} {fc:>7s}"
        )


def write_outputs(per_run_rows, aggregated_rows):
    result_root = common.result_root()
    per_run_path = result_root / "ablation_per_run.csv"
    with open(per_run_path, "w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "variant",
                "label",
                "controlled_object",
                "run_dir",
                "run_path",
                "t_resp_s",
                "beta_d",
                "N_beta_samples",
                "epsilon_d",
                "error",
            ],
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(per_run_rows)

    summary_path = result_root / "ablation_comparison.csv"
    with open(summary_path, "w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "variant",
                "label",
                "controlled_object",
                "n_runs",
                "n_successful_runs",
                "R_r",
                # Zone-level P_p (primary summary metric)
                "P_p",
                "N_zones",
                "N_tp_zones",
                # Track-level P_p (legacy)
                "P_p_track",
                "N_qualified",
                "N_tp",
                # Other metrics
                "F_c",
                "t_resp_s",
                "epsilon_d",
                "beta_d",
                "N_GT",
                "N_matched",
                "N_confirmed",
                "N_false",
                "N_beta_samples",
                "N_epsilon_d_samples",
            ],
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(aggregated_rows)

    print(f"\n[compare_ablation] Per-run CSV written to: {per_run_path}")
    print(f"[compare_ablation] Aggregated CSV written to: {summary_path}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate ablation comparison table")
    parser.add_argument("--output-root", type=str, help="Legacy single root containing all runs")
    parser.add_argument("--ablation-root", type=str, help="Dedicated output root for ablation runs")
    parser.add_argument("--full-pipeline-root", type=str, help="Output root for full-pipeline control runs")
    parser.add_argument("--run-dirs", nargs="+", type=str, help="Explicit list of run directories")
    parser.add_argument("--match-radius", type=float, default=common.MATCH_RADIUS)
    args = parser.parse_args(argv)

    run_dirs = gather_run_dirs(
        run_dirs=args.run_dirs,
        output_root=args.output_root,
        full_pipeline_root=args.full_pipeline_root,
        ablation_root=args.ablation_root,
    )

    if not run_dirs:
        print("[compare_ablation] No run directories found.")
        sys.exit(1)

    print(f"[compare_ablation] Found {len(run_dirs)} run(s)")
    per_run_rows = collect_per_run_rows(run_dirs, match_radius=args.match_radius)
    aggregated_rows = aggregate_variant_rows(per_run_rows)
    print_ablation_table(aggregated_rows)
    write_outputs(per_run_rows, aggregated_rows)
    return aggregated_rows


if __name__ == "__main__":
    main()
