#!/usr/bin/env python3
# Author: zgliu@cumt.edu.cn
# Affiliation: China University of Mining and Technology
# Open-source release date: 2026-04-20

import argparse
import csv
import json
import math
import os
import pathlib
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
pathlib.Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


GT_MOVING_THRESHOLD = 0.01
OUTLIER_MAX_ABS_POSITION = 1000.0
OUTLIER_MAX_NET_DISPLACEMENT = 5.0
MATCH_RADIUS = 0.6
TRUTH_BBOX_MARGIN = 0.2
DEFAULT_WORLD_FILE = (
    pathlib.Path(__file__).resolve().parents[2]
    / "gazebo_test"
    / "Mid360_simulation_plugin"
    / "livox_laser_simulation"
    / "worlds"
    / "tracked_mid360_fastlio_collapse_microdeform.world"
)

SUMMARY_HEADER = [
    "object_name",
    "classification",
    "gt_net_displacement",
    "gt_duration_sec",
    "gt_start_time",
    "gt_end_time",
    "gt_peak_displacement",
    "gt_peak_displacement_time",
    "evidence_status",
    "region_status",
    "motion_status",
    "first_evidence_time",
    "first_region_time",
    "first_motion_time",
    "evidence_delay_sec",
    "region_delay_sec",
    "motion_delay_sec",
    "peak_risk_score",
    "peak_region_risk",
    "peak_motion_distance",
    "summary_label",
    "notes",
]

OUTLIER_HEADER = [
    "object_name",
    "gt_net_displacement",
    "max_abs_position",
    "reason",
]


@dataclass
class TruthTrack:
    object_name: str
    time_sec: list
    x: list
    y: list
    z: list
    qx: list = None
    qy: list = None
    qz: list = None
    qw: list = None


@dataclass
class LinkTrack:
    scoped_link_name: str
    model_name: str
    link_name: str
    time_sec: list
    x: list
    y: list
    z: list
    qx: list = None
    qy: list = None
    qz: list = None
    qw: list = None


@dataclass(frozen=True)
class TruthBoxSpec:
    model_name: str
    size_x: float
    size_y: float
    size_z: float


@dataclass(frozen=True)
class AnalysisOutputs:
    output_dir: pathlib.Path
    summary_csv: pathlib.Path
    outlier_csv: pathlib.Path
    report_md: pathlib.Path
    gt_motion_timeline_png: pathlib.Path
    detection_stage_timeline_png: pathlib.Path
    spatial_overlay_png: pathlib.Path


def _to_numpy(values):
    return np.asarray(values, dtype=float)


def time_sec_from_dict(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, dict):
        if "sec" in value:
            return float(value["sec"])
        secs = float(value.get("secs", 0.0))
        nsecs = float(value.get("nsecs", 0.0))
        return secs + (nsecs / 1e9)
    return None


def layer_status(records):
    if records is None:
        return "missing"
    if len(records) == 0:
        return "empty"
    return "available"


def classify_truth_track(track, moving_threshold=GT_MOVING_THRESHOLD,
                         max_abs_position=OUTLIER_MAX_ABS_POSITION,
                         max_net_displacement=OUTLIER_MAX_NET_DISPLACEMENT):
    x = _to_numpy(track.x)
    y = _to_numpy(track.y)
    z = _to_numpy(track.z)

    if not (np.isfinite(x).all() and np.isfinite(y).all() and np.isfinite(z).all()):
        return "outlier"

    max_abs_position_value = float(
        max(np.max(np.abs(x)), np.max(np.abs(y)), np.max(np.abs(z)))
    )
    if max_abs_position_value > max_abs_position:
        return "outlier"

    net_displacement = compute_net_displacement(track)
    if net_displacement > max_net_displacement:
        return "outlier"
    if net_displacement < moving_threshold:
        return "static"
    return "moving"


def track_peak_displacement(track):
    x = _to_numpy(track.x)
    y = _to_numpy(track.y)
    z = _to_numpy(track.z)
    disp = np.sqrt((x - x[0]) ** 2 + (y - y[0]) ** 2 + (z - z[0]) ** 2)
    return float(np.max(disp))


def truth_track_metrics(track):
    x = _to_numpy(track.x)
    y = _to_numpy(track.y)
    z = _to_numpy(track.z)
    t = _to_numpy(track.time_sec)
    disp = np.sqrt((x - x[0]) ** 2 + (y - y[0]) ** 2 + (z - z[0]) ** 2)
    start_idx = int(np.argmax(disp >= GT_MOVING_THRESHOLD)) if np.any(disp >= GT_MOVING_THRESHOLD) else 0
    peak_idx = int(np.argmax(disp))
    max_abs_position = float(max(np.max(np.abs(x)), np.max(np.abs(y)), np.max(np.abs(z))))
    reasons = []
    if not (np.isfinite(x).all() and np.isfinite(y).all() and np.isfinite(z).all()):
        reasons.append("non_finite_position")
    if max_abs_position > OUTLIER_MAX_ABS_POSITION:
        reasons.append("abs_position_exceeds_threshold")
    if float(disp[-1]) > OUTLIER_MAX_NET_DISPLACEMENT:
        reasons.append("net_displacement_exceeds_threshold")

    return {
        "net_displacement": float(disp[-1]),
        "duration_sec": float(t[-1] - t[0]) if len(t) > 1 else 0.0,
        "start_time": float(t[start_idx]),
        "end_time": float(t[-1]),
        "peak_displacement": float(disp[peak_idx]),
        "peak_displacement_time": float(t[peak_idx]),
        "max_abs_position": max_abs_position,
        "outlier_reason": "|".join(reasons) if reasons else "",
    }


def select_bundle_motion_track(track, link_tracks):
    candidates = [track] + list(link_tracks)
    candidate_metrics = [truth_track_metrics(candidate) for candidate in candidates]
    best_index = max(
        range(len(candidates)),
        key=lambda index: (
            candidate_metrics[index]["peak_displacement"],
            candidate_metrics[index]["net_displacement"],
            candidate_metrics[index]["duration_sec"],
        ),
    )
    return candidates[best_index], candidate_metrics[best_index], candidate_metrics


def bundle_truth_metrics(track, link_tracks):
    representative_track, representative_metrics, candidate_metrics = select_bundle_motion_track(
        track, link_tracks
    )
    moving_candidate_metrics = [
        item for item in candidate_metrics if item["peak_displacement"] >= GT_MOVING_THRESHOLD
    ]
    timing_metrics = moving_candidate_metrics or candidate_metrics

    metrics = dict(representative_metrics)
    metrics["start_time"] = min(item["start_time"] for item in timing_metrics)
    metrics["end_time"] = max(item["end_time"] for item in timing_metrics)
    metrics["duration_sec"] = max(item["duration_sec"] for item in timing_metrics)
    metrics["net_displacement"] = max(
        item["net_displacement"] for item in candidate_metrics
    )
    metrics["peak_displacement"] = max(
        item["peak_displacement"] for item in candidate_metrics
    )
    metrics["max_abs_position"] = max(
        item["max_abs_position"] for item in candidate_metrics
    )
    metrics["outlier_reason"] = "|".join(
        reason for reason in (item["outlier_reason"] for item in candidate_metrics) if reason
    )
    metrics["representative_track_name"] = getattr(
        representative_track,
        "scoped_link_name",
        getattr(representative_track, "object_name", ""),
    )
    metrics["representative_track_index"] = candidate_metrics.index(representative_metrics)
    return metrics


def compute_net_displacement(track):
    x = _to_numpy(track.x)
    y = _to_numpy(track.y)
    z = _to_numpy(track.z)
    return float(
        math.sqrt(
            (x[-1] - x[0]) ** 2 +
            (y[-1] - y[0]) ** 2 +
            (z[-1] - z[0]) ** 2
        )
    )


def load_truth_track(csv_path):
    with pathlib.Path(csv_path).open() as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise ValueError(f"No truth rows found in {csv_path}")

    return TruthTrack(
        object_name=rows[0]["model_name"],
        time_sec=[float(row["recorded_time_sec"]) for row in rows],
        x=[float(row["position_x"]) for row in rows],
        y=[float(row["position_y"]) for row in rows],
        z=[float(row["position_z"]) for row in rows],
        qx=[float(row["orientation_x"]) for row in rows],
        qy=[float(row["orientation_y"]) for row in rows],
        qz=[float(row["orientation_z"]) for row in rows],
        qw=[float(row["orientation_w"]) for row in rows],
    )


def load_link_track(csv_path):
    with pathlib.Path(csv_path).open() as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise ValueError(f"No link truth rows found in {csv_path}")

    return LinkTrack(
        scoped_link_name=rows[0]["scoped_link_name"],
        model_name=rows[0]["model_name"],
        link_name=rows[0]["link_name"],
        time_sec=[float(row["recorded_time_sec"]) for row in rows],
        x=[float(row["position_x"]) for row in rows],
        y=[float(row["position_y"]) for row in rows],
        z=[float(row["position_z"]) for row in rows],
        qx=[float(row["orientation_x"]) for row in rows],
        qy=[float(row["orientation_y"]) for row in rows],
        qz=[float(row["orientation_z"]) for row in rows],
        qw=[float(row["orientation_w"]) for row in rows],
    )


def load_truth_tracks(truth_dir):
    tracks = []
    for csv_path in sorted(pathlib.Path(truth_dir).glob("*.csv")):
        try:
            tracks.append(load_truth_track(csv_path))
        except ValueError:
            continue
    return tracks


def load_link_tracks(truth_links_dir):
    tracks = []
    truth_links_dir = pathlib.Path(truth_links_dir)
    if not truth_links_dir.is_dir():
        return tracks
    for csv_path in sorted(truth_links_dir.glob("*.csv")):
        try:
            tracks.append(load_link_track(csv_path))
        except ValueError:
            continue
    return tracks


def load_jsonl_optional(path):
    path = pathlib.Path(path)
    if not path.exists():
        return None

    records = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def quaternion_to_rotation_matrix(quat):
    x = float(quat.get("x", 0.0))
    y = float(quat.get("y", 0.0))
    z = float(quat.get("z", 0.0))
    w = float(quat.get("w", 1.0))
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm == 0.0:
        return np.eye(3)
    x /= norm
    y /= norm
    z /= norm
    w /= norm
    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ])


def build_rigid_transform(pose_dict, source_frame="", target_frame="", metadata=None):
    position = pose_dict.get("position", {}) if isinstance(pose_dict, dict) else {}
    orientation = pose_dict.get("orientation", {}) if isinstance(pose_dict, dict) else {}
    return {
        "metadata": metadata,
        "source_frame": str(source_frame),
        "target_frame": str(target_frame),
        "translation": np.array([
            float(position.get("x", 0.0)),
            float(position.get("y", 0.0)),
            float(position.get("z", 0.0)),
        ]),
        "rotation": quaternion_to_rotation_matrix(orientation),
    }


def invert_rigid_transform(transform):
    rotation = np.asarray(transform["rotation"], dtype=float)
    translation = np.asarray(transform["translation"], dtype=float)
    inv_rotation = rotation.T
    inv_translation = -inv_rotation.dot(translation)
    return {
        "metadata": transform.get("metadata"),
        "source_frame": str(transform.get("target_frame", "")),
        "target_frame": str(transform.get("source_frame", "")),
        "translation": inv_translation,
        "rotation": inv_rotation,
    }


def load_alignment(run_dir):
    path = pathlib.Path(run_dir) / "meta" / "frame_alignment.json"
    if not path.exists():
        return None
    with path.open() as handle:
        metadata = json.load(handle)
    explicit_transform = metadata.get("world_from_algorithm_transform")
    if isinstance(explicit_transform, dict):
        explicit_pose = explicit_transform.get("pose")
        if isinstance(explicit_pose, dict):
            return build_rigid_transform(
                explicit_pose,
                source_frame=explicit_transform.get("source_frame", metadata.get("algorithm_frame", "")),
                target_frame=explicit_transform.get("target_frame", metadata.get("truth_frame", "")),
                metadata=metadata,
            )

    ego_pose = metadata.get("ego_initial_pose_world")
    if not ego_pose:
        return None
    return build_rigid_transform(
        ego_pose,
        source_frame=metadata.get("algorithm_frame", ""),
        target_frame=metadata.get("truth_frame", ""),
        metadata=metadata,
    )


def transform_point_with_transform(point_dict, transform):
    vec = np.array([
        float(point_dict.get("x", 0.0)),
        float(point_dict.get("y", 0.0)),
        float(point_dict.get("z", 0.0)),
    ])
    if transform is None:
        return None
    out = transform["rotation"].dot(vec) + transform["translation"]
    return {"x": float(out[0]), "y": float(out[1]), "z": float(out[2])}


def transform_vector_with_transform(vector_dict, transform):
    vec = np.array([
        float(vector_dict.get("x", 0.0)),
        float(vector_dict.get("y", 0.0)),
        float(vector_dict.get("z", 0.0)),
    ])
    if transform is None:
        return None
    out = transform["rotation"].dot(vec)
    return {"x": float(out[0]), "y": float(out[1]), "z": float(out[2])}


def transform_point_world(point_dict, alignment):
    return transform_point_with_transform(point_dict, alignment)


def transform_vector_world(vector_dict, alignment):
    return transform_vector_with_transform(vector_dict, alignment)


def resolve_world_file(world_file=None):
    if world_file is not None:
        world_file = pathlib.Path(world_file)
        return world_file if world_file.is_file() else None
    return DEFAULT_WORLD_FILE if DEFAULT_WORLD_FILE.is_file() else None


def load_truth_box_specs(world_file):
    world_file = resolve_world_file(world_file)
    if world_file is None:
        return {}
    root = ET.parse(world_file).getroot()
    world = root.find("world")
    if world is None:
        return {}

    specs = {}
    for model in world.findall("model"):
        model_name = str(model.attrib.get("name", "")).strip()
        if not model_name:
            continue
        size_node = model.find("./link/collision/geometry/box/size")
        if size_node is None or not (size_node.text or "").strip():
            continue
        try:
            sx, sy, sz = [float(part) for part in size_node.text.split()]
        except (TypeError, ValueError):
            continue
        specs[model_name] = TruthBoxSpec(
            model_name=model_name,
            size_x=sx,
            size_y=sy,
            size_z=sz,
        )
    return specs


def track_position_at_time(track, time_sec):
    t = _to_numpy(track.time_sec)
    idx = int(np.searchsorted(t, time_sec, side="left"))
    if idx <= 0:
        best = 0
    elif idx >= len(t):
        best = len(t) - 1
    else:
        best = idx if abs(t[idx] - time_sec) <= abs(t[idx - 1] - time_sec) else idx - 1
    return {
        "x": float(track.x[best]),
        "y": float(track.y[best]),
        "z": float(track.z[best]),
    }


def track_orientation_at_time(track, time_sec):
    qx = getattr(track, "qx", None)
    qy = getattr(track, "qy", None)
    qz = getattr(track, "qz", None)
    qw = getattr(track, "qw", None)
    if not qx or not qy or not qz or not qw:
        return {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}
    t = _to_numpy(track.time_sec)
    idx = int(np.searchsorted(t, time_sec, side="left"))
    if idx <= 0:
        best = 0
    elif idx >= len(t):
        best = len(t) - 1
    else:
        best = idx if abs(t[idx] - time_sec) <= abs(t[idx - 1] - time_sec) else idx - 1
    return {
        "x": float(qx[best]),
        "y": float(qy[best]),
        "z": float(qz[best]),
        "w": float(qw[best]),
    }


def truth_points_at_time(track, link_tracks, time_sec):
    points = [track_position_at_time(track, time_sec)]
    for link_track in link_tracks:
        points.append(track_position_at_time(link_track, time_sec))
    return points


def _bbox_corners(bmin, bmax):
    return [
        {"x": x, "y": y, "z": z}
        for x in (bmin["x"], bmax["x"])
        for y in (bmin["y"], bmax["y"])
        for z in (bmin["z"], bmax["z"])
    ]


def _aabb_from_points(points):
    xs = [float(point["x"]) for point in points]
    ys = [float(point["y"]) for point in points]
    zs = [float(point["z"]) for point in points]
    return {
        "min": {"x": min(xs), "y": min(ys), "z": min(zs)},
        "max": {"x": max(xs), "y": max(ys), "z": max(zs)},
    }


def transform_aabb_world(bmin_dict, bmax_dict, alignment):
    if alignment is None or bmin_dict is None or bmax_dict is None:
        return None
    corners_world = [
        transform_point_with_transform(corner, alignment)
        for corner in _bbox_corners(bmin_dict, bmax_dict)
    ]
    if any(point is None for point in corners_world):
        return None
    return _aabb_from_points(corners_world)


def build_truth_bbox_world(track, box_spec, time_sec, margin=TRUTH_BBOX_MARGIN):
    if box_spec is None:
        return None
    center = track_position_at_time(track, time_sec)
    orientation = track_orientation_at_time(track, time_sec)
    rotation = quaternion_to_rotation_matrix(orientation)
    half_extents = np.array([
        (float(box_spec.size_x) * 0.5) + float(margin),
        (float(box_spec.size_y) * 0.5) + float(margin),
        (float(box_spec.size_z) * 0.5) + float(margin),
    ])
    local_corners = []
    for sx in (-half_extents[0], half_extents[0]):
        for sy in (-half_extents[1], half_extents[1]):
            for sz in (-half_extents[2], half_extents[2]):
                local_corners.append(np.array([sx, sy, sz]))
    center_vec = np.array([center["x"], center["y"], center["z"]])
    world_points = []
    for corner in local_corners:
        world_corner = rotation.dot(corner) + center_vec
        world_points.append(
            {"x": float(world_corner[0]), "y": float(world_corner[1]), "z": float(world_corner[2])}
        )
    return _aabb_from_points(world_points)


def point_inside_aabb(point, aabb):
    if point is None or aabb is None:
        return False
    return (
        float(aabb["min"]["x"]) <= float(point["x"]) <= float(aabb["max"]["x"]) and
        float(aabb["min"]["y"]) <= float(point["y"]) <= float(aabb["max"]["y"]) and
        float(aabb["min"]["z"]) <= float(point["z"]) <= float(aabb["max"]["z"])
    )


def aabb_intersects(lhs, rhs):
    if lhs is None or rhs is None:
        return False
    return not (
        float(lhs["max"]["x"]) < float(rhs["min"]["x"]) or
        float(lhs["min"]["x"]) > float(rhs["max"]["x"]) or
        float(lhs["max"]["y"]) < float(rhs["min"]["y"]) or
        float(lhs["min"]["y"]) > float(rhs["max"]["y"]) or
        float(lhs["max"]["z"]) < float(rhs["min"]["z"]) or
        float(lhs["min"]["z"]) > float(rhs["max"]["z"])
    )


def classify_truth_bundle(track, link_tracks):
    classification = classify_truth_track(track)
    if classification == "outlier":
        return classification

    effective_peak = track_peak_displacement(track)
    for link_track in link_tracks:
        link_class = classify_truth_track(link_track)
        if link_class == "outlier":
            return "outlier"
        effective_peak = max(effective_peak, track_peak_displacement(link_track))

    if classification == "moving":
        return "moving"
    if effective_peak >= GT_MOVING_THRESHOLD:
        return "moving"
    return "static"


def distance_between_points(a, b):
    return math.sqrt(
        (float(a["x"]) - float(b["x"])) ** 2 +
        (float(a["y"]) - float(b["y"])) ** 2 +
        (float(a["z"]) - float(b["z"])) ** 2
    )


def record_time_sec(record):
    header = record.get("header", {})
    stamp = time_sec_from_dict(header.get("stamp"))
    if stamp is not None:
        return stamp
    return time_sec_from_dict(record.get("recorded_at"))


def persistent_region_is_confirmed(region):
    return bool(region.get("confirmed", False))


def persistent_track_presence(record):
    return any(persistent_region_is_confirmed(item) for item in record.get("regions", []))


def significant_region_presence(record):
    return any(bool(item.get("significant", False)) for item in record.get("regions", []))


def longest_presence_streak(records, presence_fn):
    best_count = 0
    best_duration = 0.0
    current_count = 0
    current_start = None
    current_end = None

    for record in records or []:
        t = record_time_sec(record)
        if t is None:
            continue
        if presence_fn(record):
            if current_count == 0:
                current_start = t
            current_count += 1
            current_end = t
            continue
        if current_count > 0:
            duration = float(current_end - current_start) if current_end is not None else 0.0
            if current_count > best_count or (current_count == best_count and duration > best_duration):
                best_count = current_count
                best_duration = duration
        current_count = 0
        current_start = None
        current_end = None

    if current_count > 0:
        duration = float(current_end - current_start) if current_end is not None else 0.0
        if current_count > best_count or (current_count == best_count and duration > best_duration):
            best_count = current_count
            best_duration = duration

    return {
        "count": best_count,
        "duration_sec": best_duration,
    }


def _matching_truth_points(truth_tracks, link_tracks_by_model, time_sec):
    points = []
    for track in truth_tracks:
        link_tracks = link_tracks_by_model.get(track.object_name, [])
        points.extend(truth_points_at_time(track, link_tracks, time_sec))
    return points


def build_persistent_risk_summary(persistent_records, region_records, truth_tracks, link_tracks_by_model,
                                  alignment, match_radius=MATCH_RADIUS):
    layer_status_value = layer_status(persistent_records)
    summary = {
        "layer_status": layer_status_value,
        "confirmed_track_count": 0,
        "first_confirmed_time": "",
        "max_confirmed_duration_sec": 0.0,
        "confirmed_coverage_hits": 0,
        "confirmed_presence_streak_count": 0,
        "confirmed_presence_streak_sec": 0.0,
        "significant_region_presence_streak_count": 0,
        "significant_region_presence_streak_sec": 0.0,
        "stability_judgment": "unavailable",
    }

    if persistent_records is None:
        return summary

    confirmed_track_times = {}
    first_confirmed_time = None
    coverage_hits = 0

    moving_truth_tracks = [
        track for track in truth_tracks
        if classify_truth_bundle(track, link_tracks_by_model.get(track.object_name, [])) == "moving"
    ]

    for record in persistent_records:
        t = record_time_sec(record)
        if t is None:
            continue
        confirmed_regions = [item for item in record.get("regions", []) if persistent_region_is_confirmed(item)]
        if not confirmed_regions:
            continue
        if first_confirmed_time is None:
            first_confirmed_time = t
        truth_points = _matching_truth_points(moving_truth_tracks, link_tracks_by_model, t) if alignment is not None else []
        for item in confirmed_regions:
            track_id = int(item.get("track_id", 0))
            span = confirmed_track_times.setdefault(track_id, {"first": t, "last": t})
            span["last"] = t
            span["first"] = min(span["first"], t)
            if alignment is None or not truth_points:
                continue
            center_world = transform_point_world(item.get("center", {}), alignment)
            if center_world is None:
                continue
            if any(distance_between_points(center_world, gt_point) <= match_radius for gt_point in truth_points):
                coverage_hits += 1

    if confirmed_track_times:
        summary["confirmed_track_count"] = len(confirmed_track_times)
        summary["first_confirmed_time"] = first_confirmed_time if first_confirmed_time is not None else ""
        summary["max_confirmed_duration_sec"] = max(
            float(span["last"] - span["first"]) for span in confirmed_track_times.values()
        )
        summary["confirmed_coverage_hits"] = coverage_hits

    confirmed_streak = longest_presence_streak(persistent_records, persistent_track_presence)
    significant_streak = longest_presence_streak(region_records, significant_region_presence)
    summary["confirmed_presence_streak_count"] = confirmed_streak["count"]
    summary["confirmed_presence_streak_sec"] = confirmed_streak["duration_sec"]
    summary["significant_region_presence_streak_count"] = significant_streak["count"]
    summary["significant_region_presence_streak_sec"] = significant_streak["duration_sec"]

    if layer_status_value != "available" or layer_status(region_records) != "available":
        summary["stability_judgment"] = "unavailable"
        return summary

    persistent_duration = confirmed_streak["duration_sec"]
    region_duration = significant_streak["duration_sec"]
    persistent_count = confirmed_streak["count"]
    region_count = significant_streak["count"]

    if persistent_count == 0 or region_count == 0:
        summary["stability_judgment"] = "unavailable"
    elif persistent_duration > region_duration * 1.25:
        summary["stability_judgment"] = "more_stable"
    elif region_duration > persistent_duration * 1.25:
        summary["stability_judgment"] = "less_stable"
    elif persistent_count > region_count:
        summary["stability_judgment"] = "more_stable"
    elif region_count > persistent_count:
        summary["stability_judgment"] = "less_stable"
    else:
        summary["stability_judgment"] = "similar"

    return summary


def evaluate_truth_object(track, link_tracks, alignment, evidence_records, region_records, motion_records,
                          truth_box_specs=None, match_radius=MATCH_RADIUS):
    metrics = bundle_truth_metrics(track, link_tracks)
    truth_box_spec = (truth_box_specs or {}).get(track.object_name)

    evidence_status = layer_status(evidence_records)
    region_status = layer_status(region_records)
    motion_status = layer_status(motion_records)

    summary = {
        "object_name": track.object_name,
        "classification": classify_truth_bundle(track, link_tracks),
        "gt_net_displacement": metrics["net_displacement"],
        "gt_duration_sec": metrics["duration_sec"],
        "gt_start_time": metrics["start_time"],
        "gt_end_time": metrics["end_time"],
        "gt_peak_displacement": metrics["peak_displacement"],
        "gt_peak_displacement_time": metrics["peak_displacement_time"],
        "evidence_status": evidence_status,
        "region_status": region_status,
        "motion_status": motion_status,
        "first_evidence_time": "",
        "first_region_time": "",
        "first_motion_time": "",
        "evidence_delay_sec": "",
        "region_delay_sec": "",
        "motion_delay_sec": "",
        "peak_risk_score": "",
        "peak_region_risk": "",
        "peak_motion_distance": "",
        "summary_label": "truth_only",
        "notes": "",
    }

    if summary["classification"] == "outlier":
        summary["summary_label"] = "outlier_excluded"
        summary["notes"] = metrics["outlier_reason"]
        return summary

    if alignment is None:
        summary["notes"] = "alignment_unavailable"
        summary["evidence_status"] = "alignment_unavailable" if evidence_status == "available" else evidence_status
        summary["region_status"] = "alignment_unavailable" if region_status == "available" else region_status
        summary["motion_status"] = "alignment_unavailable" if motion_status == "available" else motion_status
        return summary

    matched_evidence = False
    matched_region = False
    matched_motion = False
    peak_risk = None
    peak_region_risk = None
    peak_motion_distance = None

    if evidence_records is not None:
        for record in evidence_records:
            t = record_time_sec(record)
            if t is None:
                continue
            gt_points = truth_points_at_time(track, link_tracks, t)
            truth_bbox = build_truth_bbox_world(track, truth_box_spec, t)
            for item in record.get("evidences", []):
                if not item.get("active", False):
                    continue
                point_world = transform_point_world(item.get("position", {}), alignment)
                if point_world is None:
                    continue
                if (
                    (truth_bbox is not None and point_inside_aabb(point_world, truth_bbox)) or
                    any(distance_between_points(point_world, gt_point) <= match_radius for gt_point in gt_points)
                ):
                    matched_evidence = True
                    if summary["first_evidence_time"] == "":
                        summary["first_evidence_time"] = t
                        summary["evidence_delay_sec"] = t - metrics["start_time"]
                    peak_risk = max(float(item.get("risk_score", 0.0)), peak_risk or float(item.get("risk_score", 0.0)))

    if region_records is not None:
        for record in region_records:
            t = record_time_sec(record)
            if t is None:
                continue
            gt_points = truth_points_at_time(track, link_tracks, t)
            truth_bbox = build_truth_bbox_world(track, truth_box_spec, t)
            nearest = None
            nearest_dist = None
            for item in record.get("regions", []):
                point_world = transform_point_world(item.get("center", {}), alignment)
                if point_world is None:
                    continue
                bbox_world = transform_aabb_world(
                    item.get("bbox_min", {}),
                    item.get("bbox_max", {}),
                    alignment,
                )
                dist = min(distance_between_points(point_world, gt_point) for gt_point in gt_points)
                bbox_match = (
                    (truth_bbox is not None and point_inside_aabb(point_world, truth_bbox)) or
                    (truth_bbox is not None and bbox_world is not None and aabb_intersects(truth_bbox, bbox_world))
                )
                if bbox_match:
                    dist = min(dist, 0.0)
                if nearest_dist is None or dist < nearest_dist:
                    nearest = item
                    nearest_dist = dist
            if nearest is not None and nearest_dist is not None and nearest_dist <= match_radius:
                matched_region = True
                if summary["first_region_time"] == "":
                    summary["first_region_time"] = t
                    summary["region_delay_sec"] = t - metrics["start_time"]
                peak_region_risk = max(float(nearest.get("peak_risk", 0.0)),
                                       peak_region_risk or float(nearest.get("peak_risk", 0.0)))

    if motion_records is not None:
        for record in motion_records:
            t = record_time_sec(record)
            if t is None:
                continue
            gt_points = truth_points_at_time(track, link_tracks, t)
            truth_bbox = build_truth_bbox_world(track, truth_box_spec, t)
            nearest = None
            nearest_dist = None
            for item in record.get("motions", []):
                point_world = transform_point_world(item.get("new_center", {}), alignment)
                if point_world is None:
                    continue
                bbox_world = transform_aabb_world(
                    item.get("bbox_new_min", {}),
                    item.get("bbox_new_max", {}),
                    alignment,
                )
                dist = min(distance_between_points(point_world, gt_point) for gt_point in gt_points)
                bbox_match = (
                    (truth_bbox is not None and point_inside_aabb(point_world, truth_bbox)) or
                    (truth_bbox is not None and bbox_world is not None and aabb_intersects(truth_bbox, bbox_world))
                )
                if bbox_match:
                    dist = min(dist, 0.0)
                if nearest_dist is None or dist < nearest_dist:
                    nearest = item
                    nearest_dist = dist
            if nearest is not None and nearest_dist is not None and nearest_dist <= match_radius:
                matched_motion = True
                if summary["first_motion_time"] == "":
                    summary["first_motion_time"] = t
                    summary["motion_delay_sec"] = t - metrics["start_time"]
                peak_motion_distance = max(float(nearest.get("distance", 0.0)),
                                           peak_motion_distance or float(nearest.get("distance", 0.0)))

    if evidence_status == "available":
        summary["evidence_status"] = "matched" if matched_evidence else "not_detected"
    if region_status == "available":
        summary["region_status"] = "matched" if matched_region else "not_detected"
    if motion_status == "available":
        summary["motion_status"] = "matched" if matched_motion else "not_detected"

    if peak_risk is not None:
        summary["peak_risk_score"] = peak_risk
    if peak_region_risk is not None:
        summary["peak_region_risk"] = peak_region_risk
    if peak_motion_distance is not None:
        summary["peak_motion_distance"] = peak_motion_distance

    if matched_motion and not matched_region:
        summary["summary_label"] = "motion_without_region"
    elif matched_evidence and matched_region and matched_motion:
        summary["summary_label"] = "full_detection"
    elif matched_evidence and matched_region:
        summary["summary_label"] = "evidence_region"
    elif matched_evidence:
        summary["summary_label"] = "evidence_only"
    else:
        summary["summary_label"] = "truth_only"

    missing_layers = [name for name, status in [
        ("risk_evidence", evidence_status),
        ("risk_regions", region_status),
        ("structure_motions", motion_status),
    ] if status != "available"]
    if missing_layers:
        summary["notes"] = "missing_layers=" + ",".join(missing_layers)

    return summary


def write_csv(output_path, fieldnames, rows):
    with pathlib.Path(output_path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _plot_style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 10.0,
            "axes.labelsize": 10.5,
            "axes.titlesize": 11.0,
            "legend.frameon": False,
        }
    )


def plot_gt_motion_timeline(tracks, link_tracks_by_model, summary_rows, output_path):
    _plot_style()
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for track, summary in zip(tracks, summary_rows):
        if summary["classification"] != "moving":
            continue
        motion_track, _, _ = select_bundle_motion_track(
            track, link_tracks_by_model.get(track.object_name, [])
        )
        x = _to_numpy(motion_track.x)
        y = _to_numpy(motion_track.y)
        z = _to_numpy(motion_track.z)
        t0 = float(motion_track.time_sec[0])
        t = _to_numpy(motion_track.time_sec) - t0
        disp = np.sqrt((x - x[0]) ** 2 + (y - y[0]) ** 2 + (z - z[0]) ** 2)
        ax.plot(t, disp, linewidth=1.8, label=track.object_name)
    ax.set_title("GT Object Displacement Over Time")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Displacement [m]")
    ax.grid(True, linestyle="--", alpha=0.35)
    if ax.lines:
        ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def build_stage_series(records, key, value_key=None, active_key=None):
    if records is None:
        return np.array([]), np.array([]), np.array([])
    times = []
    counts = []
    peaks = []
    for record in records:
        t = record_time_sec(record)
        if t is None:
            continue
        items = record.get(key, [])
        if active_key is not None:
            items = [item for item in items if item.get(active_key, False)]
        times.append(t)
        counts.append(len(items))
        if value_key is None or not items:
            peaks.append(0.0)
        else:
            peaks.append(max(float(item.get(value_key, 0.0)) for item in items))
    return np.asarray(times, dtype=float), np.asarray(counts, dtype=float), np.asarray(peaks, dtype=float)


def plot_detection_stage_timeline(evidence_records, region_records, motion_records, output_path):
    _plot_style()
    fig, axes = plt.subplots(3, 1, figsize=(10, 8.5), sharex=True)
    stage_specs = [
        ("risk_evidence", evidence_records, "evidences", "risk_score", "active", "Evidence activity"),
        ("risk_regions", region_records, "regions", "peak_risk", None, "Region activity"),
        ("structure_motions", motion_records, "motions", "distance", None, "Structure-motion activity"),
    ]
    for ax, (name, records, key, value_key, active_key, title) in zip(axes, stage_specs):
        if records is None:
            ax.text(0.5, 0.5, f"{name}: missing", ha="center", va="center", transform=ax.transAxes)
            ax.set_ylabel("count")
            ax.grid(True, linestyle="--", alpha=0.25)
            continue
        times, counts, peaks = build_stage_series(records, key, value_key, active_key)
        if len(times) == 0:
            ax.text(0.5, 0.5, f"{name}: empty", ha="center", va="center", transform=ax.transAxes)
            ax.set_ylabel("count")
            ax.grid(True, linestyle="--", alpha=0.25)
            continue
        t_rel = times - times[0]
        ax.plot(t_rel, counts, color="#1f4e79", linewidth=1.8, label="count")
        ax.set_ylabel("count")
        ax.set_title(title)
        ax.grid(True, linestyle="--", alpha=0.25)
        twin = ax.twinx()
        twin.plot(t_rel, peaks, color="#d55e00", linewidth=1.4, linestyle="--", label="peak")
        twin.set_ylabel("peak")
    axes[-1].set_xlabel("Time [s]")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_spatial_overlay(tracks, link_tracks_by_model, summary_rows, alignment, region_records, motion_records,
                         output_path):
    _plot_style()
    fig, ax = plt.subplots(figsize=(8, 6))
    for track, summary in zip(tracks, summary_rows):
        if summary["classification"] != "moving":
            continue
        motion_track, _, _ = select_bundle_motion_track(
            track, link_tracks_by_model.get(track.object_name, [])
        )
        ax.plot(motion_track.x, motion_track.y, linewidth=1.6, label=track.object_name)
    if alignment is None:
        ax.text(0.5, 0.95, "Alignment unavailable: truth-only overlay",
                transform=ax.transAxes, ha="center", va="top")
    else:
        if region_records is not None:
            region_points = []
            for record in region_records:
                for item in record.get("regions", []):
                    pt = transform_point_world(item.get("center", {}), alignment)
                    if pt is not None:
                        region_points.append(pt)
            if region_points:
                ax.scatter([p["x"] for p in region_points], [p["y"] for p in region_points],
                           s=18, c="#009e73", alpha=0.65, label="risk_regions")
        if motion_records is not None:
            motion_starts = []
            motion_ends = []
            for record in motion_records:
                for item in record.get("motions", []):
                    start = transform_point_world(item.get("old_center", {}), alignment)
                    end = transform_point_world(item.get("new_center", {}), alignment)
                    if start is not None and end is not None:
                        motion_starts.append(start)
                        motion_ends.append(end)
            for start, end in zip(motion_starts, motion_ends):
                ax.annotate(
                    "",
                    xy=(end["x"], end["y"]),
                    xytext=(start["x"], start["y"]),
                    arrowprops={"arrowstyle": "->", "lw": 1.2, "color": "#d55e00"},
                )
    ax.set_title("Spatial Overlay in Gazebo World")
    ax.set_xlabel("world x [m]")
    ax.set_ylabel("world y [m]")
    ax.grid(True, linestyle="--", alpha=0.3)
    if ax.lines or ax.collections:
        ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def write_report(report_path, run_dir, summary_rows, outlier_rows, layer_statuses, alignment,
                 persistent_summary):
    lines = []
    lines.append(f"# Sim Run Analysis Report")
    lines.append("")
    lines.append(f"- run_dir: `{run_dir}`")
    lines.append(f"- alignment_available: `{alignment is not None}`")
    lines.append(f"- risk_evidence: `{layer_statuses['risk_evidence']}`")
    lines.append(f"- risk_regions: `{layer_statuses['risk_regions']}`")
    lines.append(f"- persistent_risk_regions: `{layer_statuses['persistent_risk_regions']}`")
    lines.append(f"- structure_motions: `{layer_statuses['structure_motions']}`")
    lines.append("")
    lines.append("## Moving Objects")
    moving_rows = [row for row in summary_rows if row["classification"] == "moving"]
    if not moving_rows:
        lines.append("- none")
    else:
        for row in moving_rows:
            lines.append(
                f"- `{row['object_name']}`: label=`{row['summary_label']}`, "
                f"gt_net_displacement={row['gt_net_displacement']:.6f} m, "
                f"evidence_status=`{row['evidence_status']}`, "
                f"region_status=`{row['region_status']}`, "
                f"motion_status=`{row['motion_status']}`"
            )
    lines.append("")
    lines.append("## Outliers")
    if not outlier_rows:
        lines.append("- none")
    else:
        for row in outlier_rows:
            lines.append(
                f"- `{row['object_name']}`: "
                f"net_displacement={row['gt_net_displacement']}, "
                f"max_abs_position={row['max_abs_position']}, "
                f"reason=`{row['reason']}`"
            )
    lines.append("")
    lines.append("## Persistent Risk")
    lines.append(f"- layer_status: `{persistent_summary['layer_status']}`")
    lines.append(f"- confirmed_track_count: `{persistent_summary['confirmed_track_count']}`")
    lines.append(f"- first_confirmed_time: `{persistent_summary['first_confirmed_time']}`")
    lines.append(
        f"- max_confirmed_duration_sec: `{persistent_summary['max_confirmed_duration_sec']}`"
    )
    lines.append(
        f"- confirmed_coverage_hits: `{persistent_summary['confirmed_coverage_hits']}`"
    )
    lines.append(
        f"- confirmed_presence_streak: count=`{persistent_summary['confirmed_presence_streak_count']}`, "
        f"duration_sec=`{persistent_summary['confirmed_presence_streak_sec']}`"
    )
    lines.append(
        f"- significant_region_presence_streak: "
        f"count=`{persistent_summary['significant_region_presence_streak_count']}`, "
        f"duration_sec=`{persistent_summary['significant_region_presence_streak_sec']}`"
    )
    lines.append(f"- stability_judgment: `{persistent_summary['stability_judgment']}`")
    with pathlib.Path(report_path).open("w") as handle:
        handle.write("\n".join(lines) + "\n")


def analyze_sim_run(run_dir, output_dir=None, world_file=None):
    run_dir = pathlib.Path(run_dir)
    truth_objects_dir = run_dir / "truth" / "objects"
    if not truth_objects_dir.is_dir():
        raise FileNotFoundError(f"Truth object directory missing: {truth_objects_dir}")

    output_dir = pathlib.Path(output_dir) if output_dir else run_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    evidence_records = load_jsonl_optional(run_dir / "algorithm" / "risk_evidence.jsonl")
    region_records = load_jsonl_optional(run_dir / "algorithm" / "risk_regions.jsonl")
    persistent_region_records = load_jsonl_optional(run_dir / "algorithm" / "persistent_risk_regions.jsonl")
    motion_records = load_jsonl_optional(run_dir / "algorithm" / "structure_motions.jsonl")
    alignment = load_alignment(run_dir)
    truth_box_specs = load_truth_box_specs(world_file)

    truth_tracks = load_truth_tracks(truth_objects_dir)
    link_tracks = load_link_tracks(run_dir / "truth" / "links")
    link_tracks_by_model = {}
    for link_track in link_tracks:
        link_tracks_by_model.setdefault(link_track.model_name, []).append(link_track)
    summary_rows = []
    outlier_rows = []
    for track in truth_tracks:
        summary = evaluate_truth_object(
            track,
            link_tracks_by_model.get(track.object_name, []),
            alignment=alignment,
            evidence_records=evidence_records,
            region_records=region_records,
            motion_records=motion_records,
            truth_box_specs=truth_box_specs,
        )
        summary_rows.append(summary)
        if summary["classification"] == "outlier":
            metrics = bundle_truth_metrics(track, link_tracks_by_model.get(track.object_name, []))
            outlier_rows.append(
                {
                    "object_name": track.object_name,
                    "gt_net_displacement": metrics["net_displacement"],
                    "max_abs_position": metrics["max_abs_position"],
                    "reason": metrics["outlier_reason"],
                }
            )

    moving_summary_rows = [row for row in summary_rows if row["classification"] == "moving"]
    persistent_summary = build_persistent_risk_summary(
        persistent_region_records,
        region_records,
        truth_tracks,
        link_tracks_by_model,
        alignment,
        match_radius=MATCH_RADIUS,
    )

    summary_csv = output_dir / "summary.csv"
    outlier_csv = output_dir / "outlier_objects.csv"
    report_md = output_dir / "report.md"
    gt_motion_timeline_png = output_dir / "gt_motion_timeline.png"
    detection_stage_timeline_png = output_dir / "detection_stage_timeline.png"
    spatial_overlay_png = output_dir / "spatial_overlay.png"

    write_csv(summary_csv, SUMMARY_HEADER, moving_summary_rows)
    write_csv(outlier_csv, OUTLIER_HEADER, outlier_rows)
    write_report(
        report_md,
        run_dir,
        moving_summary_rows,
        outlier_rows,
        {
            "risk_evidence": layer_status(evidence_records),
            "risk_regions": layer_status(region_records),
            "persistent_risk_regions": layer_status(persistent_region_records),
            "structure_motions": layer_status(motion_records),
        },
        alignment,
        persistent_summary,
    )
    plot_gt_motion_timeline(truth_tracks, link_tracks_by_model, summary_rows, gt_motion_timeline_png)
    plot_detection_stage_timeline(evidence_records, region_records, motion_records,
                                  detection_stage_timeline_png)
    plot_spatial_overlay(truth_tracks, link_tracks_by_model, summary_rows, alignment, region_records, motion_records,
                         spatial_overlay_png)

    return AnalysisOutputs(
        output_dir=output_dir,
        summary_csv=summary_csv,
        outlier_csv=outlier_csv,
        report_md=report_md,
        gt_motion_timeline_png=gt_motion_timeline_png,
        detection_stage_timeline_png=detection_stage_timeline_png,
        spatial_overlay_png=spatial_overlay_png,
    )


def resolve_latest_sim_run(output_root):
    output_root = pathlib.Path(output_root)
    candidates = sorted(output_root.glob("*/sim_run_*"))
    if not candidates:
        raise FileNotFoundError(f"No sim_run directories found under: {output_root}")
    return candidates[-1]


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze one recorded sim_run directory.")
    parser.add_argument(
        "--run-dir",
        type=pathlib.Path,
        default=None,
        help="Path to sim_run_XXX directory. Defaults to the latest run under --output-root.",
    )
    parser.add_argument(
        "--output-root",
        type=pathlib.Path,
        default=pathlib.Path.home() / ".ros" / "alert" / "output",
        help="Output root used when --run-dir is omitted.",
    )
    parser.add_argument(
        "--analysis-dir",
        type=pathlib.Path,
        default=None,
        help="Optional output directory for analysis files. Defaults to <run-dir>/analysis.",
    )
    parser.add_argument(
        "--world-file",
        type=pathlib.Path,
        default=None,
        help="Optional Gazebo world file used to derive truth bbox sizes.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_dir = args.run_dir if args.run_dir else resolve_latest_sim_run(args.output_root)
    outputs = analyze_sim_run(run_dir, args.analysis_dir, args.world_file)
    print(f"analysis_dir: {outputs.output_dir}")
    print(f"summary_csv: {outputs.summary_csv}")
    print(f"outlier_csv: {outputs.outlier_csv}")
    print(f"report_md: {outputs.report_md}")
    print(f"gt_motion_timeline_png: {outputs.gt_motion_timeline_png}")
    print(f"detection_stage_timeline_png: {outputs.detection_stage_timeline_png}")
    print(f"spatial_overlay_png: {outputs.spatial_overlay_png}")


if __name__ == "__main__":
    main()
