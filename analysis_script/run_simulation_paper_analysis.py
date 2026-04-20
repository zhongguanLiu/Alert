#!/usr/bin/env python3
"""Run batch analysis for simulation outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import sys
from collections import defaultdict

_THIS_DIR = pathlib.Path(__file__).resolve().parent
_PARENT_DIR = _THIS_DIR.parent
if str(_PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(_PARENT_DIR))

import analysis_script.common as common
import analysis_script.compute_metrics as compute_metrics
import analysis_script.compute_runtime as compute_runtime
import analysis_script.plot_sim_timeline as plot_sim_timeline


def summary_dir_for_runs(run_dirs: list[pathlib.Path]) -> pathlib.Path:
    parent_names = {run.parent.name for run in run_dirs}
    if len(parent_names) == 1:
        parent_name = next(iter(parent_names))
        if parent_name.isdigit() and len(parent_name) == 8:
            target = common.result_root() / parent_name / "summary"
            target.mkdir(parents=True, exist_ok=True)
            return target
    target = common.result_root() / "summary"
    target.mkdir(parents=True, exist_ok=True)
    return target


def _load_jsonl(path: pathlib.Path) -> list[dict]:
    rows = common.load_jsonl(path)
    return rows if rows else []


def _find_gt_object(run_dir: pathlib.Path, object_name: str | None):
    if not object_name:
        return None
    gt_objects = common.load_gt_objects(run_dir / "truth")
    for obj in gt_objects:
        if obj.name == object_name:
            return obj
    return None


def _estimate_motion_duration_sec(obj) -> float | None:
    if obj is None or len(obj.positions_t) < 2 or len(obj.positions_xyz) < 2:
        return None
    onset_time = obj.onset_time
    if onset_time is None:
        return None
    last_change_idx = None
    for idx in range(1, len(obj.positions_xyz)):
        prev = obj.positions_xyz[idx - 1]
        curr = obj.positions_xyz[idx]
        delta = math.sqrt(sum((a - b) ** 2 for a, b in zip(curr, prev)))
        if delta > 1.0e-9:
            last_change_idx = idx
    if last_change_idx is None:
        return None
    return obj.positions_t[last_change_idx] - onset_time


def _target_detection_details(run_dir: pathlib.Path, metrics: dict, controlled_name: str | None):
    target_entry = None
    for entry in metrics.get("t_resp", {}).get("per_object", []):
        if controlled_name and entry.get("object") == controlled_name:
            target_entry = entry
            break
    if target_entry is None:
        for entry in metrics.get("t_resp", {}).get("per_object", []):
            if entry.get("t_resp") is not None:
                target_entry = entry
                break
    if target_entry is None:
        return {"t_resp_s": None, "gt_disp_at_detection_mm": None, "detected": False}

    t_first = target_entry.get("t_first_confirmed")
    object_name = target_entry.get("object")
    gt_disp_mm = None
    if t_first is not None and object_name:
        obj = _find_gt_object(run_dir, object_name)
        gt_disp = common.gt_displacement_at_time(obj, t_first) if obj is not None else None
        if gt_disp is not None:
            gt_disp_mm = gt_disp * 1000.0

    return {
        "t_resp_s": target_entry.get("t_resp"),
        "gt_disp_at_detection_mm": gt_disp_mm,
        "detected": target_entry.get("t_resp") is not None,
    }


def _analysis_controlled_object_names(run_dir: pathlib.Path) -> list[str | None]:
    controlled_names = common.get_analysis_controlled_object_names(run_dir)
    if controlled_names:
        return controlled_names
    controlled_name = common.get_analysis_controlled_object_name(run_dir)
    return [controlled_name] if controlled_name else [None]


def compute_run_metrics(run_dir: pathlib.Path, match_radius: float, run_index: int | None = None) -> list[dict]:
    metrics = compute_metrics.run_metrics(run_dir, match_radius=match_radius)
    scenario = common.load_json(run_dir / "meta" / "scenario_manifest.json") or {}
    gt_objects = common.load_gt_objects(run_dir / "truth")
    gt_by_name = {obj.name: obj for obj in gt_objects}
    moving_gt_count = len([obj for obj in gt_objects if obj.classification == "moving"])
    rows = []

    for controlled_name in _analysis_controlled_object_names(run_dir):
        velocity_mps = common.get_injection_velocity(run_dir, controlled_name)
        velocity_mmps = velocity_mps * 1000.0 if velocity_mps is not None else None
        target_obj = gt_by_name.get(controlled_name) if controlled_name else None
        target_info = _target_detection_details(run_dir, metrics, controlled_name)
        valid_t_resp = [target_info["t_resp_s"]] if target_info.get("t_resp_s") is not None else []
        run_duration_s = None
        if target_obj is not None and target_obj.positions_t:
            run_duration_s = target_obj.positions_t[-1] - target_obj.positions_t[0]
        motion_duration_s = _estimate_motion_duration_sec(target_obj)

        run_name = f"sim_run_{run_index:03d}" if run_index is not None else run_dir.name

        rows.append(
            {
                "run_dir": run_dir,
                "run_name": run_name,
                "scenario_id": scenario.get("scenario_id"),
                "controlled_object": controlled_name,
                "velocity_mmps": velocity_mmps,
                "moving_gt_count": moving_gt_count,
                "run_duration_s": run_duration_s,
                "motion_duration_s": motion_duration_s,
                "metrics": metrics,
                "valid_t_resp_values": valid_t_resp,
                "target_detection": target_info,
            }
        )

    return rows


def _unique_run_rows(per_run_rows: list[dict]) -> list[dict]:
    unique_rows = []
    seen_runs = set()
    for row in per_run_rows:
        run_key = _row_run_key(row)
        if run_key in seen_runs:
            continue
        seen_runs.add(run_key)
        unique_rows.append(row)
    return unique_rows


def _row_run_key(row: dict):
    run_dir = row.get("run_dir")
    return str(run_dir) if run_dir is not None else row.get("run_name")


def aggregate_detection_table(per_run_rows: list[dict]) -> dict:
    total_gt = 0
    total_matched = 0
    total_confirmed = 0
    total_false = 0
    t_resp_values = []
    beta_weighted_sum = 0.0
    beta_sample_count = 0
    pp_values = []
    epsd_values = []

    for row in _unique_run_rows(per_run_rows):
        metrics = row["metrics"]
        rr = metrics.get("R_r", {})
        fc = metrics.get("F_c", {})
        bd = metrics.get("beta_d", {})
        pp = metrics.get("P_p", {})
        ed = metrics.get("epsilon_d", {})
        total_gt += int(rr.get("N_GT") or 0)
        total_matched += int(rr.get("N_matched") or 0)
        total_confirmed += int(fc.get("N_confirmed") or 0)
        total_false += int(fc.get("N_false") or 0)
        if bd.get("beta_d") is not None and bd.get("N_samples"):
            beta_weighted_sum += float(bd["beta_d"]) * int(bd["N_samples"])
            beta_sample_count += int(bd["N_samples"])
        if pp.get("P_p") is not None:
            pp_values.append(float(pp["P_p"]))
        if ed.get("epsilon_d") is not None:
            epsd_values.append(float(ed["epsilon_d"]))
    for row in per_run_rows:
        t_resp_values.extend(row.get("valid_t_resp_values", []))

    return {
        "method": "Ours",
        "R_r": total_matched / total_gt if total_gt > 0 else None,
        "P_p": _safe_mean(pp_values),
        "F_c": total_false / total_confirmed if total_confirmed > 0 else None,
        "t_resp_s": sum(t_resp_values) / len(t_resp_values) if t_resp_values else None,
        "epsilon_d": _safe_mean(epsd_values),
        "beta_d": beta_weighted_sum / beta_sample_count if beta_sample_count > 0 else None,
        "N_GT": total_gt,
        "N_matched": total_matched,
        "N_confirmed": total_confirmed,
        "N_false": total_false,
        "N_t_resp_samples": len(t_resp_values),
        "N_P_p_samples": len(pp_values),
        "N_epsilon_d_samples": len(epsd_values),
        "N_beta_samples": beta_sample_count,
    }


def compute_velocity_mdd_rows(per_run_rows: list[dict]) -> list[dict]:
    groups: dict[float, list[dict]] = defaultdict(list)
    for row in per_run_rows:
        velocity = row.get("velocity_mmps")
        if velocity is None:
            continue
        groups[round(float(velocity), 6)].append(row)

    rows = []
    for velocity in sorted(groups):
        run_group = groups[velocity]
        detected_rows = [
            row for row in run_group
            if row.get("target_detection", {}).get("detected")
            and row.get("target_detection", {}).get("gt_disp_at_detection_mm") is not None
        ]
        total_run_keys = {_row_run_key(row) for row in run_group}
        detected_run_keys = {_row_run_key(row) for row in detected_rows}

        # Aggregate R_r per velocity (micro-average over GT objects)
        vel_gt = sum(int(r["metrics"].get("R_r", {}).get("N_GT") or 0) for r in run_group)
        vel_matched = sum(int(r["metrics"].get("R_r", {}).get("N_matched") or 0) for r in run_group)

        # Aggregate P_p per velocity (mean over runs)
        pp_vals = [
            float(r["metrics"]["P_p"]["P_p"])
            for r in run_group
            if r["metrics"].get("P_p", {}).get("P_p") is not None
        ]

        # Aggregate epsilon_d per velocity (mean over runs)
        epsd_vals = [
            float(r["metrics"]["epsilon_d"]["epsilon_d"])
            for r in run_group
            if r["metrics"].get("epsilon_d", {}).get("epsilon_d") is not None
        ]

        rows.append(
            {
                "velocity_mmps": velocity,
                "detected": bool(detected_run_keys) and len(detected_run_keys) == len(total_run_keys),
                "detected_runs": len(detected_run_keys),
                "total_runs": len(total_run_keys),
                "mean_R_r": vel_matched / vel_gt if vel_gt > 0 else None,
                "mean_P_p": _safe_mean(pp_vals),
                "mean_t_resp_s": (
                    sum(float(row["target_detection"]["t_resp_s"]) for row in detected_rows)
                    / len(detected_rows)
                    if detected_rows
                    else None
                ),
                "mean_epsilon_d": _safe_mean(epsd_vals),
                "mean_gt_disp_at_detection_mm": (
                    sum(float(row["target_detection"]["gt_disp_at_detection_mm"]) for row in detected_rows)
                    / len(detected_rows)
                    if detected_rows
                    else None
                ),
            }
        )
    return rows


def aggregate_runtime_table(run_dirs: list[pathlib.Path]) -> dict:
    values = defaultdict(list)
    for run_dir in run_dirs:
        records = _load_jsonl(run_dir / "runtime" / "stage_runtime.jsonl")
        if len(records) > compute_runtime.WARMUP_CYCLES:
            records = records[compute_runtime.WARMUP_CYCLES:]
        for record in records:
            for key in ("stage_a_ms", "stage_b_ms", "stage_c_ms", "stage_d_ms", "total_ms"):
                if key not in record:
                    continue
                try:
                    value = float(record[key])
                except (TypeError, ValueError):
                    continue
                if math.isfinite(value):
                    values[key].append(value)

    return {
        key: (sum(items) / len(items) if items else None)
        for key, items in values.items()
    }


def summarize_scenarios(per_run_rows: list[dict]) -> dict:
    unique_run_rows = _unique_run_rows(per_run_rows)
    velocity_groups: dict[float, list[dict]] = defaultdict(list)
    velocity_run_keys: dict[float, set[str]] = defaultdict(set)
    for row in per_run_rows:
        velocity = row.get("velocity_mmps")
        if velocity is not None:
            velocity_key = round(float(velocity), 6)
            velocity_groups[velocity_key].append(row)
            velocity_run_keys[velocity_key].add(_row_run_key(row))

    return {
        "total_runs": len(unique_run_rows),
        "velocities_mmps": sorted(velocity_groups.keys()),
        "repeats_per_velocity": {
            f"{velocity:.3f}": len(velocity_run_keys[velocity])
            for velocity in sorted(velocity_groups)
        },
        "controlled_objects": sorted(
            {row["controlled_object"] for row in per_run_rows if row.get("controlled_object")}
        ),
        "moving_gt_count_per_run": sorted({row["moving_gt_count"] for row in unique_run_rows}),
        "mean_run_duration_s": _safe_mean(
            [row["run_duration_s"] for row in unique_run_rows if row.get("run_duration_s") is not None]
        ),
        "mean_motion_duration_s_by_velocity": {
            f"{velocity:.3f}": _safe_mean(
                [
                    row["motion_duration_s"]
                    for row in rows
                    if row.get("motion_duration_s") is not None
                ]
            )
            for velocity, rows in sorted(velocity_groups.items())
        },
    }


def _safe_mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def select_representative_run(per_run_rows: list[dict]) -> pathlib.Path:
    def sort_key(row: dict):
        velocity = row.get("velocity_mmps")
        delta = abs(velocity - 1.0) if velocity is not None else float("inf")
        return (delta, row["run_name"])

    return min(per_run_rows, key=sort_key)["run_dir"]


def run_representative_figures(run_dir: pathlib.Path) -> list[pathlib.Path]:
    outputs = []
    outputs.extend(plot_sim_timeline.run(run_dir))
    return outputs


def write_csv(path: pathlib.Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: pathlib.Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def write_markdown_report(
    summary_dir: pathlib.Path,
    scenario_summary: dict,
    detection_summary: dict,
    mdd_rows: list[dict],
    runtime_summary: dict,
    representative_run: pathlib.Path,
    figure_paths: list[pathlib.Path],
) -> pathlib.Path:
    path = summary_dir / "simulation_summary.md"
    lines = [
        "# Paper Front-Half Simulation Analysis",
        "",
        "## Scenario Summary",
        "",
        f"- Total runs: {scenario_summary['total_runs']}",
        f"- Velocities (mm/s): {', '.join(f'{v:g}' for v in scenario_summary['velocities_mmps'])}",
        f"- Repeats per velocity: {scenario_summary['repeats_per_velocity']}",
        f"- Controlled objects resolved for analysis: {scenario_summary['controlled_objects']}",
        f"- Moving GT objects per run: {scenario_summary['moving_gt_count_per_run']}",
        f"- Mean run duration (s): {_format_num(scenario_summary.get('mean_run_duration_s'))}",
        f"- Mean motion duration by velocity (s): {scenario_summary['mean_motion_duration_s_by_velocity']}",
        "",
        "## Ours: Simulation Detection Summary",
        "",
        f"- R_r: {_format_num(detection_summary.get('R_r'))}",
        f"- P_p: {_format_num(detection_summary.get('P_p'))}",
        f"- F_c: {_format_num(detection_summary.get('F_c'))}",
        f"- t_resp (s): {_format_num(detection_summary.get('t_resp_s'))}",
        f"- epsilon_d: {_format_num(detection_summary.get('epsilon_d'))}",
        f"- beta_d: {_format_num(detection_summary.get('beta_d'))}",
        "",
        "## MDD By Velocity",
        "",
        "| Velocity (mm/s) | Detected | Runs | R_r | P_p | Mean t_resp (s) | epsilon_d | Mean GT disp (mm) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in mdd_rows:
        lines.append(
            f"| {row['velocity_mmps']:.3f} | {'Yes' if row['detected'] else 'No'} "
            f"| {row['detected_runs']}/{row['total_runs']} | "
            f"{_format_num(row.get('mean_R_r'))} | "
            f"{_format_num(row.get('mean_P_p'))} | "
            f"{_format_num(row.get('mean_t_resp_s'))} | "
            f"{_format_num(row.get('mean_epsilon_d'))} | "
            f"{_format_num(row.get('mean_gt_disp_at_detection_mm'))} |"
        )

    lines.extend(
        [
            "",
            "## Runtime Summary",
            "",
            f"- Stage A (ms): {_format_num(runtime_summary.get('stage_a_ms'))}",
            f"- Stage B (ms): {_format_num(runtime_summary.get('stage_b_ms'))}",
            f"- Stage C (ms): {_format_num(runtime_summary.get('stage_c_ms'))}",
            f"- Stage D (ms): {_format_num(runtime_summary.get('stage_d_ms'))}",
            f"- Total (ms): {_format_num(runtime_summary.get('total_ms'))}",
            "",
            "## Representative Run",
            "",
            f"- Run: `{representative_run}`",
            f"- Figures: {[str(path) for path in figure_paths]}",
            "",
            "## Notes",
            "",
            "- This batch reads from `output/` only and writes all analysis artifacts under `analysis_script/result/`.",
            "- Baseline C2C/M3C2 rows are not included here because these 12 run directories contain the proposed-method recorder outputs, not baseline evaluation outputs.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _format_num(value) -> str:
    if value is None:
        return "---"
    return f"{float(value):.3f}"


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description="Run batch simulation analysis")
    parser.add_argument("--output-root", type=str, help="Root output directory")
    parser.add_argument("--run-dirs", nargs="+", type=str, help="Explicit run directories")
    parser.add_argument("--match-radius", type=float, default=common.MATCH_RADIUS)
    parser.add_argument("--skip-figures", action="store_true", help="Skip representative figures")
    args = parser.parse_args(argv)

    if args.run_dirs:
        run_dirs = [pathlib.Path(item) for item in args.run_dirs]
    else:
        run_dirs = common.find_all_runs(args.output_root)
    if not run_dirs:
        raise FileNotFoundError("No sim_run_* directories found for analysis.")

    run_dirs = sorted(run_dirs)

    # Assign globally sequential run indices across date directories.
    # E.g. 20260411/sim_run_000..004 and 20260412/sim_run_000..005
    # become sim_run_000..004 and sim_run_005..010 respectively.
    run_index_map: dict[pathlib.Path, int] = {}
    next_index = 0
    for run_dir in run_dirs:
        run_index_map[run_dir] = next_index
        next_index += 1

    summary_dir = summary_dir_for_runs(run_dirs)

    per_run_rows = []
    for run_dir in run_dirs:
        per_run_rows.extend(compute_run_metrics(run_dir, match_radius=args.match_radius, run_index=run_index_map[run_dir]))
    if not per_run_rows:
        raise FileNotFoundError("No analysis rows were produced for the selected runs.")
    detection_summary = aggregate_detection_table(per_run_rows)
    mdd_rows = compute_velocity_mdd_rows(per_run_rows)
    runtime_summary = aggregate_runtime_table(run_dirs)
    scenario_summary = summarize_scenarios(per_run_rows)
    representative_run = select_representative_run(per_run_rows)
    figure_paths = [] if args.skip_figures else run_representative_figures(representative_run)

    per_run_csv_rows = []
    for row in per_run_rows:
        metrics = row["metrics"]
        per_run_csv_rows.append(
            {
                "run_name": row["run_name"],
                "scenario_id": row.get("scenario_id"),
                "controlled_object": row.get("controlled_object"),
                "velocity_mmps": row.get("velocity_mmps"),
                "R_r": metrics.get("R_r", {}).get("R_r"),
                "P_p": metrics.get("P_p", {}).get("P_p"),
                "F_c": metrics.get("F_c", {}).get("F_c"),
                "t_resp_s": _safe_mean(row.get("valid_t_resp_values", [])),
                "epsilon_d": metrics.get("epsilon_d", {}).get("epsilon_d"),
                "beta_d": metrics.get("beta_d", {}).get("beta_d"),
                "gt_disp_at_detection_mm": row.get("target_detection", {}).get("gt_disp_at_detection_mm"),
            }
        )

    write_csv(
        summary_dir / "per_run_metrics.csv",
        per_run_csv_rows,
        [
            "run_name",
            "scenario_id",
            "controlled_object",
            "velocity_mmps",
            "R_r",
            "P_p",
            "F_c",
            "t_resp_s",
            "epsilon_d",
            "beta_d",
            "gt_disp_at_detection_mm",
        ],
    )
    write_json(summary_dir / "simulation_detection_summary.json", detection_summary)
    write_csv(
        summary_dir / "mdd_velocity_summary.csv",
        mdd_rows,
        [
            "velocity_mmps",
            "detected",
            "detected_runs",
            "total_runs",
            "mean_R_r",
            "mean_P_p",
            "mean_t_resp_s",
            "mean_epsilon_d",
            "mean_gt_disp_at_detection_mm",
        ],
    )
    write_json(summary_dir / "runtime_summary_table.json", runtime_summary)
    write_json(summary_dir / "scenario_design_summary.json", scenario_summary)
    report_path = write_markdown_report(
        summary_dir,
        scenario_summary,
        detection_summary,
        mdd_rows,
        runtime_summary,
        representative_run,
        figure_paths,
    )

    return {
        "summary_dir": summary_dir,
        "report_path": report_path,
        "representative_run": representative_run,
        "figure_paths": figure_paths,
    }


if __name__ == "__main__":
    main()
