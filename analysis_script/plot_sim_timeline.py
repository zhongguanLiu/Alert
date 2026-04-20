#!/usr/bin/env python3
"""Plot the simulation timeline figure for a run."""

import argparse
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np

# Ensure analysis_script is importable
_SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import common
import plot_common as pc


def _parse_args():
    parser = argparse.ArgumentParser(description="Plot 3-panel simulation timeline.")
    parser.add_argument("--run-dir", type=str, default=None,
                        help="Path to a specific sim_run_NNN directory.")
    parser.add_argument("--latest", action="store_true",
                        help="Auto-select the most recent run.")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Override output directory for figures.")
    parser.add_argument("--match-radius", type=float, default=0.8,
                        help="Spatial matching radius in meters.")
    parser.add_argument("--max-anchors-per-object", type=int, default=20,
                        help="Maximum number of anchor traces to plot per object.")
    return parser.parse_args()


def _build_gt_displacement_series(obj: common.GTObject) -> tuple:
    """Build displacement-vs-time arrays (in seconds and mm) for a GT object."""
    if not obj.positions_t or not obj.positions_xyz:
        return np.array([]), np.array([])
    ref = obj.positions_xyz[0]
    times = np.array(obj.positions_t)
    disps_mm = np.array([
        np.sqrt((p[0] - ref[0])**2 + (p[1] - ref[1])**2 + (p[2] - ref[2])**2) * 1000.0
        for p in obj.positions_xyz
    ])
    return times, disps_mm


def _find_best_track_per_object(
    track_events: list,
    track_ids: list,
) -> int:
    """Among matched tracks, select the one with the longest confirmed duration."""
    if not track_ids:
        return -1
    # Score by number of frame_status events in CONFIRMED state
    scores = {}
    for ev in track_events:
        if ev.get("event_type") != "frame_status":
            continue
        tid = ev.get("track_id")
        if tid not in track_ids:
            continue
        if ev.get("state") == pc.STATE_CONFIRMED:
            scores[tid] = scores.get(tid, 0) + 1
    if not scores:
        # Fall back to longest track overall
        for ev in track_events:
            if ev.get("event_type") != "frame_status":
                continue
            tid = ev.get("track_id")
            if tid in track_ids:
                scores[tid] = scores.get(tid, 0) + 1
    if not scores:
        return track_ids[0]
    return max(scores, key=scores.get)


def plot_timeline(run_dir: pathlib.Path, args):
    """Generate and save the 3-panel timeline figure."""
    pc.setup_plot_style()

    print(f"Loading data from: {run_dir}")
    rd = common.load_run_data(run_dir, load_clusters=False)

    # Load risk evidence (large file)
    risk_evidence_path = run_dir / "algorithm" / "risk_evidence.jsonl"
    print(f"Loading risk evidence: {risk_evidence_path}")
    risk_evidence = common.load_jsonl(risk_evidence_path)
    if risk_evidence is None:
        print("ERROR: risk_evidence.jsonl not found.")
        return

    # Build coordinate transform
    T_w_a = common.build_world_from_algorithm_transform(rd.alignment)

    # Identify moving objects
    moving_objects = [obj for obj in rd.gt_objects if obj.classification == "moving"]
    if not moving_objects:
        print("ERROR: No moving GT objects found.")
        return

    # Assign colors to objects
    obj_colors = {}
    color_pool = [pc.COLOR_MODEL_01, pc.COLOR_MODEL_02]
    for i, obj in enumerate(moving_objects):
        obj_colors[obj.name] = color_pool[i % len(color_pool)]

    # Get common t0 for relative time
    first_algo_t = None
    if risk_evidence:
        first_algo_t = common.record_time_sec(risk_evidence[0])
    if first_algo_t is None:
        first_algo_t = min(obj.positions_t[0] for obj in moving_objects if obj.positions_t)

    # -----------------------------------------------------------------------
    # Panel (a): GT Displacement
    # -----------------------------------------------------------------------
    print("Building GT displacement curves...")
    gt_series = {}
    for obj in moving_objects:
        times, disps = _build_gt_displacement_series(obj)
        gt_series[obj.name] = (times, disps)

    # -----------------------------------------------------------------------
    # Panel (b): Anchor risk scores matched to GT objects
    # -----------------------------------------------------------------------
    print("Matching anchors to GT objects (this may take a moment)...")
    anchor_risk_map = pc.match_anchors_to_gt(
        risk_evidence, rd.gt_objects, T_w_a, match_radius=args.match_radius
    )

    # -----------------------------------------------------------------------
    # Panel (c): Persistent track state
    # -----------------------------------------------------------------------
    print("Matching tracks to GT objects...")
    track_map = pc.match_tracks_to_gt(
        rd.track_events or [], rd.gt_objects, T_w_a, match_radius=args.match_radius
    )

    # Get first-confirmed times per object
    first_confirmed_times = {}
    for obj in moving_objects:
        tids = track_map.get(obj.name, [])
        first_confirmed_times[obj.name] = pc.get_first_confirmed_time(
            rd.track_events or [], tids
        )

    # Build per-track timeseries for the best track per object
    best_tracks = {}
    track_ts = {}
    for obj in moving_objects:
        tids = track_map.get(obj.name, [])
        if tids:
            best_tid = _find_best_track_per_object(rd.track_events or [], tids)
            best_tracks[obj.name] = best_tid
            ts_data = pc.build_track_timeseries(rd.track_events or [], [best_tid])
            track_ts[obj.name] = ts_data.get(best_tid, {"times": [], "states": [], "mean_risks": []})
        else:
            best_tracks[obj.name] = -1
            track_ts[obj.name] = {"times": [], "states": [], "mean_risks": []}

    # -----------------------------------------------------------------------
    # Create figure
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(7.16, 5.5), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1, 1], "hspace": 0.12})
    ax_gt, ax_risk, ax_track = axes

    # --- Panel (a): GT Displacement ---
    GT_COLOR = "#2ca02c"  # green for GT displacement
    for obj in moving_objects:
        times, disps = gt_series[obj.name]
        if len(times) == 0:
            continue
        rel_times = times - first_algo_t
        label = obj.name.replace("_", " ").replace("model ", "Object ")
        ax_gt.plot(rel_times, disps, color=GT_COLOR, linewidth=1.2, zorder=3,
                   label=label)

    # Mark motion onset per object (green dotted vertical line)
    for obj in moving_objects:
        if obj.onset_time is not None:
            onset_rel = obj.onset_time - first_algo_t
            ax_gt.axvline(onset_rel, color=GT_COLOR, linestyle=":", linewidth=0.7, alpha=0.6)

    # Mark first_confirmed per object (keep per-object color for t_resp arrows)
    for obj in moving_objects:
        color = obj_colors[obj.name]
        fc_t = first_confirmed_times.get(obj.name)
        if fc_t is not None:
            fc_rel = fc_t - first_algo_t
            fc_disp = common.gt_displacement_at_time(obj, fc_t)
            if fc_disp is not None:
                ax_gt.plot(fc_rel, fc_disp * 1000, marker="*", color=color,
                           markersize=8, zorder=5)
            # Vertical line for first detection across all panels
            for ax in axes:
                ax.axvline(fc_rel, color=color, linestyle="--", linewidth=0.6, alpha=0.5)

    ax_gt.set_ylabel("Displacement (mm)")
    # Object legend in lower-right corner of panel (a):
    # use per-object colors (matching the t_resp arrows and confirmation markers)
    _legend_handles = []
    for obj in moving_objects:
        label = obj.name.replace("_", " ").replace("model ", "Object ")
        _legend_handles.append(
            plt.Line2D([0], [0], color=obj_colors[obj.name], linewidth=1.2, label=label)
        )
    if _legend_handles:
        ax_gt.legend(handles=_legend_handles, loc="lower right",
                     fontsize=7, framealpha=0.9, edgecolor="#CCCCCC")
    # Panel label: bold (a) in upper-left corner inside axes
    ax_gt.text(0.01, 0.97, "(a)", transform=ax_gt.transAxes,
               fontsize=9, fontweight="bold", fontfamily="Times New Roman",
               va="top", ha="left")

    # --- Panel (b): Anchor Risk Scores ---
    for obj in moving_objects:
        color = obj_colors[obj.name]
        anchors = anchor_risk_map.get(obj.name, {})
        if not anchors:
            continue

        # Sort anchors by number of observations (descending), take top N
        sorted_aids = sorted(anchors.keys(), key=lambda a: len(anchors[a]), reverse=True)
        top_aids = sorted_aids[:args.max_anchors_per_object]

        # Plot individual anchor traces (semi-transparent)
        all_times = []
        all_scores = []
        for aid in top_aids:
            pts = anchors[aid]
            ts = np.array([p[0] - first_algo_t for p in pts])
            scores = np.array([p[1] for p in pts])
            ax_risk.plot(ts, scores, color=color, alpha=0.08, linewidth=0.5, zorder=1)
            all_times.extend(ts.tolist())
            all_scores.extend(scores.tolist())

        # Plot mean trend (binned)
        if all_times:
            all_times = np.array(all_times)
            all_scores = np.array(all_scores)
            # Bin by 1-second intervals
            t_min, t_max = all_times.min(), all_times.max()
            bin_edges = np.arange(t_min, t_max + 1.0, 1.0)
            if len(bin_edges) > 1:
                bin_centers = []
                bin_means = []
                for i in range(len(bin_edges) - 1):
                    mask = (all_times >= bin_edges[i]) & (all_times < bin_edges[i + 1])
                    if mask.any():
                        bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)
                        bin_means.append(all_scores[mask].mean())
                if bin_centers:
                    ax_risk.plot(bin_centers, bin_means, color=color, linewidth=1.5,
                                alpha=0.9, zorder=4)

    ax_risk.set_ylabel("Risk Score")
    ax_risk.set_ylim(-0.05, 1.05)
    # Panel label: bold (b) in upper-left corner inside axes
    ax_risk.text(0.01, 0.97, "(b)", transform=ax_risk.transAxes,
                 fontsize=9, fontweight="bold", fontfamily="Times New Roman",
                 va="top", ha="left")

    # --- Panel (c): Track State & Mean Risk ---
    for obj in moving_objects:
        color = obj_colors[obj.name]
        ts_data = track_ts.get(obj.name, {})
        times_raw = ts_data.get("times", [])
        states = ts_data.get("states", [])
        mean_risks = ts_data.get("mean_risks", [])

        if not times_raw:
            continue

        times = np.array(times_raw) - first_algo_t
        states = np.array(states)
        mean_risks = np.array(mean_risks)

        # Plot mean risk as line (no label)
        ax_track.plot(times, mean_risks, color=color, linewidth=1.0, zorder=3)

        # Color background by state
        for i in range(len(times) - 1):
            state = states[i]
            state_color = pc.STATE_COLORS.get(state, pc.COLOR_FADING)
            ax_track.axvspan(times[i], times[i + 1], alpha=0.15,
                             color=state_color, linewidth=0, zorder=1)

        # Mark first_confirmed
        fc_t = first_confirmed_times.get(obj.name)
        if fc_t is not None:
            fc_rel = fc_t - first_algo_t
            # Find mean_risk at that time
            idx = np.searchsorted(times, fc_rel)
            if 0 < idx < len(mean_risks):
                risk_val = mean_risks[min(idx, len(mean_risks) - 1)]
                ax_track.plot(fc_rel, risk_val, marker="*", color=color,
                              markersize=8, zorder=5)

    # Add text annotation for Confirmed state color (no legend)
    ax_track.text(0.99, 0.04, "Green region: Confirmed",
                  transform=ax_track.transAxes,
                  fontsize=7, fontfamily="Times New Roman",
                  color=pc.COLOR_CONFIRMED, ha="right", va="bottom",
                  bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                            edgecolor="#CCCCCC", alpha=0.85))

    ax_track.set_ylabel("Mean Risk")
    ax_track.set_ylim(-0.05, 1.05)
    ax_track.set_xlabel("Time (s)")
    # Panel label: bold (c) in upper-left corner inside axes
    ax_track.text(0.01, 0.97, "(c)", transform=ax_track.transAxes,
                  fontsize=9, fontweight="bold", fontfamily="Times New Roman",
                  va="top", ha="left")

    # Compute x-axis limit
    all_rel_times = []
    for obj in moving_objects:
        times, _ = gt_series[obj.name]
        if len(times) > 0:
            all_rel_times.extend((times - first_algo_t).tolist())
    if all_rel_times:
        ax_track.set_xlim(-1, max(all_rel_times) + 2)

    # Add t_resp annotations to panel (a)
    for obj in moving_objects:
        color = obj_colors[obj.name]
        fc_t = first_confirmed_times.get(obj.name)
        if fc_t is None or obj.onset_time is None:
            continue
        onset_rel = obj.onset_time - first_algo_t
        fc_rel = fc_t - first_algo_t
        t_resp = fc_rel - onset_rel
        # Draw double-headed arrow on panel (a)
        y_pos = 15 if obj.name == "model_01" else 30
        ax_gt.annotate("", xy=(fc_rel, y_pos), xytext=(onset_rel, y_pos),
                        arrowprops=dict(arrowstyle="<->", color=color, lw=0.8))
        ax_gt.text((onset_rel + fc_rel) / 2, y_pos + 3,
                   f"$t_{{resp}}$={t_resp:.1f}s",
                   ha="center", va="bottom", fontsize=6.5, color=color)

    fig.align_ylabels(axes)

    # Save
    out_dir = pathlib.Path(args.out_dir) if args.out_dir else common.result_dir_for_run(run_dir) / "figures"
    pc.save_figure(fig, out_dir, "sim_timeline")
    plt.close(fig)
    print("Done.")


def main():
    args = _parse_args()
    run_dir = common.resolve_run_dir(
        run_dir=args.run_dir,
        latest=args.latest,
    )
    plot_timeline(run_dir, args)


if __name__ == "__main__":
    main()
