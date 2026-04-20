#!/usr/bin/env python3
"""Provide shared plotting styles and helper functions."""

import math
import pathlib
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

try:
    from . import common
except ImportError:  # pragma: no cover - script-style fallback
    import common

# ---------------------------------------------------------------------------
# project Color Palette
# ---------------------------------------------------------------------------
PRIMARY_BLUE = "#00609C"
PRIMARY_RED = "#BA0C2F"
COLOR_MODEL_01 = PRIMARY_BLUE
COLOR_MODEL_02 = PRIMARY_RED
COLOR_CANDIDATE = "#FFB347"   # warm orange
COLOR_CONFIRMED = "#4CAF50"   # green
COLOR_FADING = "#9E9E9E"      # grey
COLOR_THRESHOLD = "#888888"
COLOR_GRID = "#E0E0E0"
COLOR_ONSET = "#666666"

# State integer codes from C++ publisher
STATE_CANDIDATE = 0
STATE_CONFIRMED = 1
STATE_FADING = 2

STATE_COLORS = {
    STATE_CANDIDATE: COLOR_CANDIDATE,
    STATE_CONFIRMED: COLOR_CONFIRMED,
    STATE_FADING: COLOR_FADING,
}
STATE_LABELS = {
    STATE_CANDIDATE: "Candidate",
    STATE_CONFIRMED: "Confirmed",
    STATE_FADING: "Fading",
}


# ---------------------------------------------------------------------------
# project Style Setup
# ---------------------------------------------------------------------------
def setup_plot_style():
    """Configure matplotlib for project journal figures."""
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "axes.linewidth": 0.6,
        "axes.grid": True,
        "grid.linewidth": 0.3,
        "grid.alpha": 0.5,
        "lines.linewidth": 1.0,
        "lines.markersize": 4,
        "patch.linewidth": 0.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "#CCCCCC",
        "text.usetex": False,
    })


@contextmanager
def apply_paper_style():
    """Temporarily apply the legacy plotting style expected by tests/scripts."""
    keys = [
        "font.size",
        "axes.grid",
        "grid.linestyle",
        "lines.linewidth",
        "figure.dpi",
    ]
    original = {key: mpl.rcParams.get(key) for key in keys}
    mpl.rcParams.update({
        "font.size": 10,
        "axes.grid": True,
        "grid.linestyle": "--",
        "lines.linewidth": 1.2,
        "figure.dpi": 150,
    })
    try:
        yield
    finally:
        mpl.rcParams.update(original)


def figure_dir_for_run(run_dir: pathlib.Path) -> pathlib.Path:
    """Return the figure directory for a run-specific analysis output."""
    target = common.result_dir_for_run(pathlib.Path(run_dir)) / "figures"
    target.mkdir(parents=True, exist_ok=True)
    return target


def summary_figure_dir() -> pathlib.Path:
    """Return the shared summary figure output directory."""
    target = common.result_root() / "summary" / "figures"
    target.mkdir(parents=True, exist_ok=True)
    return target


# ---------------------------------------------------------------------------
# Figure Save Helper
# ---------------------------------------------------------------------------
def save_figure(
    fig: plt.Figure,
    out_dir_or_path: pathlib.Path,
    name: Optional[str] = None,
    formats: Tuple[str, ...] = ("png", "pdf"),
):
    """Save a figure to either a single file path or a directory/name pair."""
    path_obj = pathlib.Path(out_dir_or_path)
    if name is None:
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(path_obj), format=path_obj.suffix.lstrip(".") or None)
        print(f"  Saved: {path_obj}")
        return

    path_obj.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        path = path_obj / f"{name}.{fmt}"
        fig.savefig(str(path), format=fmt)
        print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Spatial Matching: Anchors to GT Objects
# ---------------------------------------------------------------------------
def match_anchors_to_gt(
    risk_evidence_records: list,
    gt_objects: List[common.GTObject],
    T_w_a: Optional[np.ndarray],
    match_radius: float = 0.8,
) -> Dict[str, Dict[int, List[Tuple[float, float]]]]:
    """Match per-frame anchor evidences to GT objects by spatial proximity.

    Returns:
        dict mapping gt_object.name -> {anchor_id -> [(time_sec, risk_score), ...]}
    """
    result = {obj.name: {} for obj in gt_objects if obj.classification == "moving"}
    moving_objects = [obj for obj in gt_objects if obj.classification == "moving"]

    if not risk_evidence_records or not moving_objects:
        return result

    for rec in risk_evidence_records:
        t = common.record_time_sec(rec)
        if t is None:
            continue
        evidences = rec.get("evidences", [])
        for ev in evidences:
            if not ev.get("active") or not ev.get("comparable"):
                continue
            anchor_id = ev.get("id")
            risk_score = ev.get("risk_score", 0.0)
            pos = ev.get("position")
            if pos is None or anchor_id is None:
                continue

            # Transform anchor position to world frame
            if T_w_a is not None:
                world_pos = common.transform_point_to_world(pos, T_w_a)
            else:
                world_pos = (pos["x"], pos["y"], pos["z"])
            if world_pos is None:
                continue

            # Match to nearest GT object
            for obj in moving_objects:
                gt_pos = common.gt_position_at_time(obj, t)
                if gt_pos is None:
                    continue
                dist = common.distance_3d(world_pos, gt_pos)
                if dist < match_radius:
                    if anchor_id not in result[obj.name]:
                        result[obj.name][anchor_id] = []
                    result[obj.name][anchor_id].append((t, risk_score))

    return result


def match_anchors_displacement_to_gt(
    risk_evidence_records: list,
    gt_objects: List[common.GTObject],
    T_w_a: Optional[np.ndarray],
    match_radius: float = 0.8,
) -> Dict[str, List[Tuple[float, float]]]:
    """Extract per-frame mean displacement magnitude for anchors matched to each GT object.

    Returns:
        dict mapping gt_object.name -> [(time_sec, mean_displacement_mm), ...]
    """
    result = {obj.name: [] for obj in gt_objects if obj.classification == "moving"}
    moving_objects = [obj for obj in gt_objects if obj.classification == "moving"]

    if not risk_evidence_records or not moving_objects:
        return result

    for rec in risk_evidence_records:
        t = common.record_time_sec(rec)
        if t is None:
            continue
        evidences = rec.get("evidences", [])

        # Per-object accumulator for this frame
        obj_disps = {obj.name: [] for obj in moving_objects}

        for ev in evidences:
            if not ev.get("active") or not ev.get("comparable"):
                continue
            pos = ev.get("position")
            disp = ev.get("displacement")
            if pos is None or disp is None:
                continue

            if T_w_a is not None:
                world_pos = common.transform_point_to_world(pos, T_w_a)
            else:
                world_pos = (pos["x"], pos["y"], pos["z"])
            if world_pos is None:
                continue

            disp_mag = math.sqrt(disp["x"] ** 2 + disp["y"] ** 2 + disp["z"] ** 2) * 1000.0

            for obj in moving_objects:
                gt_pos = common.gt_position_at_time(obj, t)
                if gt_pos is None:
                    continue
                dist = common.distance_3d(world_pos, gt_pos)
                if dist < match_radius:
                    obj_disps[obj.name].append(disp_mag)

        for obj_name, disps in obj_disps.items():
            if disps:
                result[obj_name].append((t, np.mean(disps)))

    return result


# ---------------------------------------------------------------------------
# Spatial Matching: Tracks to GT Objects
# ---------------------------------------------------------------------------
def match_tracks_to_gt(
    track_events: list,
    gt_objects: List[common.GTObject],
    T_w_a: Optional[np.ndarray],
    match_radius: float = 0.8,
) -> Dict[str, List[int]]:
    """Match persistent track IDs to GT objects by spatial proximity at first_confirmed.

    Returns:
        dict mapping gt_object.name -> [track_id, ...]
    """
    result = {obj.name: [] for obj in gt_objects if obj.classification == "moving"}
    moving_objects = [obj for obj in gt_objects if obj.classification == "moving"]

    if not track_events or not moving_objects:
        return result

    # Collect unique track IDs with their first appearance center
    track_centers = {}
    for ev in track_events:
        tid = ev.get("track_id")
        if tid is None or tid in track_centers:
            continue
        center = ev.get("center")
        t = common.record_time_sec(ev)
        if center is not None and t is not None:
            track_centers[tid] = (center, t)

    for tid, (center, t) in track_centers.items():
        if T_w_a is not None:
            world_pos = common.transform_point_to_world(center, T_w_a)
        else:
            world_pos = (center["x"], center["y"], center["z"])
        if world_pos is None:
            continue

        best_obj = None
        best_dist = float("inf")
        for obj in moving_objects:
            gt_pos = common.gt_position_at_time(obj, t)
            if gt_pos is None:
                continue
            dist = common.distance_3d(world_pos, gt_pos)
            if dist < match_radius and dist < best_dist:
                best_dist = dist
                best_obj = obj.name

        if best_obj is not None:
            result[best_obj].append(tid)

    return result


def build_track_timeseries(
    track_events: list,
    track_ids: List[int],
) -> Dict[int, Dict[str, list]]:
    """Build per-track time-series of state and risk from frame_status events.

    Returns:
        dict mapping track_id -> {"times": [...], "states": [...], "mean_risks": [...], "peak_risks": [...]}
    """
    result = {}
    for tid in track_ids:
        result[tid] = {"times": [], "states": [], "mean_risks": [], "peak_risks": []}

    tid_set = set(track_ids)
    for ev in track_events:
        tid = ev.get("track_id")
        if tid not in tid_set:
            continue
        if ev.get("event_type") != "frame_status":
            continue
        t = common.record_time_sec(ev)
        if t is None:
            continue
        result[tid]["times"].append(t)
        result[tid]["states"].append(ev.get("state", -1))
        result[tid]["mean_risks"].append(ev.get("mean_risk", 0.0))
        result[tid]["peak_risks"].append(ev.get("peak_risk", 0.0))

    return result


def get_first_confirmed_time(
    track_events: list,
    track_ids: List[int],
) -> Optional[float]:
    """Get the earliest first_confirmed timestamp among a set of track IDs."""
    tid_set = set(track_ids)
    best_t = None
    for ev in track_events:
        if ev.get("event_type") != "first_confirmed":
            continue
        if ev.get("track_id") not in tid_set:
            continue
        t = common.record_time_sec(ev)
        if t is not None:
            if best_t is None or t < best_t:
                best_t = t
    return best_t
