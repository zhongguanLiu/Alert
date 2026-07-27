#!/usr/bin/env python3

import argparse
import bisect
import csv
import importlib.util
import json
import math
import os
import pathlib
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np


try:
    from algorithm_frame_store import (
        AlgorithmFrameStoreError,
        CompressedFrameStore,
        ReplayableAlgorithmStream,
        ReplayableSequence,
        iter_algorithm_stream,
    )
except ImportError:  # source-tree execution and importlib-based unit tests
    _storage_module_path = pathlib.Path(__file__).with_name("algorithm_frame_store.py")
    _storage_spec = importlib.util.spec_from_file_location(
        "analyze_sim_run_algorithm_frame_store", _storage_module_path
    )
    _storage_module = importlib.util.module_from_spec(_storage_spec)
    _storage_spec.loader.exec_module(_storage_module)
    AlgorithmFrameStoreError = _storage_module.AlgorithmFrameStoreError
    CompressedFrameStore = _storage_module.CompressedFrameStore
    ReplayableAlgorithmStream = _storage_module.ReplayableAlgorithmStream
    ReplayableSequence = _storage_module.ReplayableSequence
    iter_algorithm_stream = _storage_module.iter_algorithm_stream

try:
    from formal_analysis_scope import (
        PHASE_NAMES,
        load_formal_analysis_scope,
        scoped_records,
        write_scope_json,
    )
except ImportError:  # source-tree execution and importlib-based unit tests
    _scope_module_path = pathlib.Path(__file__).with_name("formal_analysis_scope.py")
    _scope_spec = importlib.util.spec_from_file_location(
        "analyze_sim_run_formal_analysis_scope", _scope_module_path
    )
    _scope_module = importlib.util.module_from_spec(_scope_spec)
    _scope_spec.loader.exec_module(_scope_module)
    PHASE_NAMES = _scope_module.PHASE_NAMES
    load_formal_analysis_scope = _scope_module.load_formal_analysis_scope
    scoped_records = _scope_module.scoped_records
    write_scope_json = _scope_module.write_scope_json

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
pathlib.Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


GT_MOTION_EPSILON = 1.0e-6
GT_ANGULAR_MOTION_EPSILON_DEG = 1.0e-3
# Backward-compatible name used by older helpers. This is a truth-state
# tolerance, not an ALERT detection threshold.
GT_MOVING_THRESHOLD = GT_MOTION_EPSILON
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
    / "tracked_mid360_fastlio_collapse_microdeform_sim.world"
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
    "gt_peak_rotation_deg",
    "gt_peak_linear_speed_mps",
    "gt_peak_angular_speed_degps",
    "gt_peak_linear_acceleration_mps2",
    "gt_peak_angular_acceleration_degps2",
    "gt_surface_peak_linear_speed_mps",
    "gt_surface_peak_angular_speed_degps",
    "gt_surface_peak_linear_acceleration_mps2",
    "gt_surface_peak_angular_acceleration_degps2",
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

ANCHOR_TYPE_NAMES = {0: "PLANE", 1: "EDGE", 2: "BAND"}
ANCHOR_TYPE_CLASSIFIER_CONTRACT = {
    "features": {
        "linearity": "(lambda_max-lambda_mid)/lambda_max",
        "planarity": "(lambda_mid-lambda_min)/lambda_max",
        "scattering": "lambda_min/(lambda_min+lambda_mid+lambda_max)",
    },
    "decision_order": ["EDGE", "PLANE", "BAND"],
    "EDGE": "linearity>0.62 and linearity>planarity+0.10",
    "PLANE": "planarity>0.42 and scattering<0.20",
    "BAND": "otherwise_for_an_accepted_anchor",
    "diagnostics_only": [
        "shape_linearity",
        "shape_planarity",
        "shape_scattering",
        "type_stability",
    ],
}
OBJECT_ANCHOR_METRICS_HEADER = [
    "object_name",
    "object_id",
    "anchor_type",
    "classification",
    "evaluation_status",
    "lidar_observed",
    "hit_point_count",
    "hit_window_count",
    "visible_frame_count",
    "exposure_frame_count",
    "exposure_window_count",
    "lidar_visibility_rate",
    "hit_window_rate",
    "eligible_anchor_count",
    "mean_ref_quality",
    "mean_covariance_quality",
    "mean_type_stability",
    "mean_shape_linearity",
    "mean_shape_planarity",
    "mean_shape_scattering",
    "anchor_sample_count",
    "mean_anchor_count_per_frame",
    "observable_sample_count",
    "observable_rate",
    "comparable_sample_count",
    "comparable_rate",
    "matched_sample_count",
    "matched_rate",
    "loss_sample_count",
    "loss_rate",
    "association_consistent_sample_count",
    "association_mismatch_sample_count",
    "association_mixed_sample_count",
    "association_unavailable_sample_count",
    "association_consistency_rate",
    "significant_frame_count",
    "detected",
    "first_detection_time",
    "detection_delay_sec",
    "false_alarm_episodes",
    "outcome",
]
ANCHOR_TYPE_METRICS_HEADER = [
    "anchor_type",
    "tp",
    "fp",
    "fn",
    "tn",
    "not_evaluable",
    "eligible_anchor_count",
    "mean_ref_quality",
    "mean_covariance_quality",
    "mean_type_stability",
    "mean_shape_linearity",
    "mean_shape_planarity",
    "mean_shape_scattering",
    "anchor_sample_count",
    "observable_sample_count",
    "observable_rate",
    "comparable_sample_count",
    "comparable_rate",
    "matched_sample_count",
    "matched_rate",
    "loss_sample_count",
    "loss_rate",
    "association_consistent_sample_count",
    "association_mismatch_sample_count",
    "association_mixed_sample_count",
    "association_unavailable_sample_count",
    "association_consistency_rate",
    "precision",
    "recall",
    "f1",
    "false_alarm_episodes",
    "false_alarm_rate_per_min",
]
ANCHOR_VECTOR_METRICS_HEADER = [
    "object_name",
    "object_id",
    "classification",
    "reference_epoch",
    "anchor_id",
    "anchor_type",
    "time_sec",
    "reference_time_sec",
    "estimated_dx",
    "estimated_dy",
    "estimated_dz",
    "estimated_magnitude",
    "estimated_vx",
    "estimated_vy",
    "estimated_vz",
    "estimated_speed",
    "expected_dx",
    "expected_dy",
    "expected_dz",
    "expected_magnitude",
    "expected_vx",
    "expected_vy",
    "expected_vz",
    "expected_speed",
    "vector_error_norm",
    "magnitude_error_abs",
    "direction_error_deg",
    "velocity_vector_error_norm",
    "velocity_direction_error_deg",
    "valid",
    "invalid_reason",
]
ANCHOR_VECTOR_TYPE_METRICS_HEADER = [
    "anchor_type",
    "observation_count",
    "valid_observation_count",
    "invalid_observation_count",
    "anchor_count",
    "object_count",
    "vector_error_mean",
    "vector_error_rmse",
    "magnitude_error_mae",
    "magnitude_error_rmse",
    "direction_error_mean_deg",
    "direction_error_median_deg",
    "velocity_vector_error_mean",
    "velocity_vector_error_rmse",
    "velocity_direction_error_mean_deg",
    "per_anchor_macro_vector_error_mean",
    "per_anchor_macro_magnitude_error_mae",
    "per_anchor_macro_direction_error_mean_deg",
    "per_anchor_macro_velocity_vector_error_mean",
    "per_anchor_macro_velocity_direction_error_mean_deg",
    "object_type_macro_vector_error_mean",
    "object_type_macro_magnitude_error_mae",
    "object_type_macro_direction_error_mean_deg",
    "object_type_macro_velocity_vector_error_mean",
    "object_type_macro_velocity_direction_error_mean_deg",
]
ANCHOR_TYPE_INVENTORY_HEADER = [
    "anchor_type",
    "catalog_anchor_count",
    "observed_anchor_count",
    "observation_sample_count",
    "observable_sample_count",
    "comparable_sample_count",
    "matched_sample_count",
    "significant_sample_count",
    "significant_anchor_count",
    "object_id_valid_sample_count",
    "association_consistent_sample_count",
    "association_mismatch_sample_count",
    "association_mixed_sample_count",
    "association_unavailable_sample_count",
]
PERSISTENT_OBJECT_METRICS_HEADER = [
    "object_name",
    "object_id",
    "evaluation_status",
    "lidar_observed",
    "hit_point_count",
    "visible_frame_count",
    "exposure_frame_count",
    "lidar_visibility_rate",
    "gt_start_time",
    "gt_end_time",
    "preliminary_detected",
    "confirmed_detected",
    "geometric_preliminary_detected",
    "geometric_confirmed_detected",
    "identity_preliminary_detected",
    "identity_confirmed_detected",
    "first_candidate_time",
    "first_confirmed_time",
    "candidate_delay_sec",
    "confirmation_delay_sec",
    "candidate_to_confirmation_sec",
    "gt_peak_displacement_m",
    "gt_peak_rotation_deg",
    "gt_peak_linear_speed_mps",
    "gt_peak_angular_speed_degps",
    "gt_peak_linear_acceleration_mps2",
    "gt_peak_angular_acceleration_degps2",
    "gt_surface_peak_linear_speed_mps",
    "gt_surface_peak_angular_speed_degps",
    "gt_surface_peak_linear_acceleration_mps2",
    "gt_surface_peak_angular_acceleration_degps2",
    "gt_root_translation_at_candidate_m",
    "gt_root_rotation_at_candidate_deg",
    "gt_surface_displacement_min_at_candidate_m",
    "gt_surface_displacement_median_at_candidate_m",
    "gt_surface_displacement_max_at_candidate_m",
    "gt_root_translation_at_confirmation_m",
    "gt_root_rotation_at_confirmation_deg",
    "gt_surface_displacement_min_at_confirmation_m",
    "gt_surface_displacement_median_at_confirmation_m",
    "gt_surface_displacement_max_at_confirmation_m",
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
    vx: list = None
    vy: list = None
    vz: list = None
    wx: list = None
    wy: list = None
    wz: list = None


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
class TruthMotionPolicy:
    translation_deadband_m: float = GT_MOTION_EPSILON
    rotation_deadband_deg: float = GT_ANGULAR_MOTION_EPSILON_DEG
    linear_speed_deadband_mps: float = 0.0
    angular_speed_deadband_degps: float = 0.0
    sustained_motion_samples: int = 1


@dataclass(frozen=True)
class AnalysisOutputs:
    output_dir: pathlib.Path
    summary_csv: pathlib.Path
    outlier_csv: pathlib.Path
    report_md: pathlib.Path
    gt_motion_timeline_png: pathlib.Path
    detection_stage_timeline_png: pathlib.Path
    spatial_overlay_png: pathlib.Path
    object_anchor_metrics_csv: pathlib.Path
    anchor_type_metrics_csv: pathlib.Path
    anchor_vector_metrics_csv: pathlib.Path
    anchor_vector_type_metrics_csv: pathlib.Path
    persistent_object_metrics_csv: pathlib.Path
    alert_metrics_json: pathlib.Path
    analysis_scope_json: pathlib.Path = None
    run_metrics_json: pathlib.Path = None
    phase_metrics_csv: pathlib.Path = None
    failure_modes_csv: pathlib.Path = None
    anchor_type_inventory_csv: pathlib.Path = None


class DataQualityError(ValueError):
    def __init__(self, report):
        self.report = report
        errors = report.get("errors", []) if isinstance(report, dict) else []
        super().__init__("recorded run failed data-quality validation: " + ";".join(errors))


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
    if hasattr(records, "audit"):
        return (
            "available"
            if records.audit().get("included_record_count", 0) > 0
            else "empty"
        )
    if len(records) == 0:
        return "empty"
    return "available"


def load_truth_motion_policy(run_dir):
    path = pathlib.Path(run_dir) / "meta" / "run_info.json"
    payload = {}
    if path.is_file():
        try:
            run_info = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON at {path}:{exc.lineno}: {exc}") from exc
        payload = run_info.get("truth_motion_policy", {})
        if not isinstance(payload, dict):
            raise ValueError("truth_motion_policy must be a mapping")

    def positive_float(name, default, allow_zero=False):
        try:
            value = float(payload.get(name, default))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid truth motion policy field: {name}") from exc
        if not math.isfinite(value) or value < 0.0 or (not allow_zero and value == 0.0):
            raise ValueError(f"invalid truth motion policy field: {name}")
        return value

    try:
        sustained = int(payload.get("sustained_motion_samples", 1))
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid truth motion policy field: sustained_motion_samples") from exc
    if sustained <= 0:
        raise ValueError("sustained_motion_samples must be positive")
    return TruthMotionPolicy(
        translation_deadband_m=positive_float(
            "translation_deadband_m", GT_MOTION_EPSILON
        ),
        rotation_deadband_deg=positive_float(
            "rotation_deadband_deg", GT_ANGULAR_MOTION_EPSILON_DEG
        ),
        linear_speed_deadband_mps=positive_float(
            "linear_speed_deadband_mps", 0.0, allow_zero=True
        ),
        angular_speed_deadband_degps=positive_float(
            "angular_speed_deadband_degps", 0.0, allow_zero=True
        ),
        sustained_motion_samples=sustained,
    )


def classify_truth_track(track, moving_threshold=GT_MOVING_THRESHOLD,
                         angular_moving_threshold_deg=GT_ANGULAR_MOTION_EPSILON_DEG,
                         max_abs_position=OUTLIER_MAX_ABS_POSITION,
                         max_net_displacement=OUTLIER_MAX_NET_DISPLACEMENT,
                         policy=None):
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

    peak_displacement = track_peak_displacement(track)
    if peak_displacement > max_net_displacement:
        return "outlier"
    effective_policy = policy or TruthMotionPolicy(
        translation_deadband_m=float(moving_threshold),
        rotation_deadband_deg=float(angular_moving_threshold_deg),
    )
    return "moving" if truth_track_metrics(track, effective_policy)["moving"] else "static"


def track_peak_displacement(track):
    x = _to_numpy(track.x)
    y = _to_numpy(track.y)
    z = _to_numpy(track.z)
    disp = np.sqrt((x - x[0]) ** 2 + (y - y[0]) ** 2 + (z - z[0]) ** 2)
    return float(np.max(disp))


def track_rotation_angles_deg(track):
    qx = getattr(track, "qx", None)
    qy = getattr(track, "qy", None)
    qz = getattr(track, "qz", None)
    qw = getattr(track, "qw", None)
    if any(component is None or len(component) == 0 for component in (qx, qy, qz, qw)):
        return np.zeros(len(track.time_sec), dtype=float)
    quaternions = np.column_stack((qx, qy, qz, qw)).astype(float)
    norms = np.linalg.norm(quaternions, axis=1)
    valid = np.isfinite(quaternions).all(axis=1) & (norms > 1.0e-12)
    if not valid.all():
        return np.full(len(track.time_sec), np.inf, dtype=float)
    quaternions = quaternions / norms[:, None]
    dots = np.abs(quaternions.dot(quaternions[0]))
    dots = np.clip(dots, -1.0, 1.0)
    return np.degrees(2.0 * np.arccos(dots))


def track_peak_rotation_deg(track):
    angles = track_rotation_angles_deg(track)
    return float(np.max(angles)) if len(angles) else 0.0


def _finite_difference_vectors(values, time_values):
    values = np.asarray(values, dtype=float)
    time_values = np.asarray(time_values, dtype=float)
    if len(time_values) <= 1:
        return np.zeros_like(values)
    edge_order = 2 if len(time_values) >= 3 else 1
    return np.column_stack(
        [
            np.gradient(values[:, axis], time_values, edge_order=edge_order)
            for axis in range(values.shape[1])
        ]
    )


def _derived_angular_velocity(track):
    count = len(track.time_sec)
    result = np.zeros((count, 3), dtype=float)
    if count <= 1:
        return result
    components = (
        getattr(track, "qx", None),
        getattr(track, "qy", None),
        getattr(track, "qz", None),
        getattr(track, "qw", None),
    )
    if any(component is None or len(component) != count for component in components):
        return result
    quaternions = np.column_stack(components).astype(float)
    quaternions = np.asarray([_normalized_quaternion(row) for row in quaternions])
    interval_velocity = np.zeros((count - 1, 3), dtype=float)
    for index in range(count - 1):
        dt_value = float(track.time_sec[index + 1]) - float(track.time_sec[index])
        if dt_value <= 0.0:
            continue
        lower = quaternions[index]
        upper = quaternions[index + 1]
        if float(np.dot(lower, upper)) < 0.0:
            upper = -upper
        lower_inverse = np.array([-lower[0], -lower[1], -lower[2], lower[3]])
        delta = np.array(
            [
                lower_inverse[3] * upper[0]
                + lower_inverse[0] * upper[3]
                + lower_inverse[1] * upper[2]
                - lower_inverse[2] * upper[1],
                lower_inverse[3] * upper[1]
                - lower_inverse[0] * upper[2]
                + lower_inverse[1] * upper[3]
                + lower_inverse[2] * upper[0],
                lower_inverse[3] * upper[2]
                + lower_inverse[0] * upper[1]
                - lower_inverse[1] * upper[0]
                + lower_inverse[2] * upper[3],
                lower_inverse[3] * upper[3]
                - lower_inverse[0] * upper[0]
                - lower_inverse[1] * upper[1]
                - lower_inverse[2] * upper[2],
            ],
            dtype=float,
        )
        delta = _normalized_quaternion(delta)
        vector_norm = float(np.linalg.norm(delta[:3]))
        if vector_norm <= 1.0e-12:
            continue
        angle = 2.0 * math.atan2(vector_norm, max(0.0, float(delta[3])))
        axis_local = delta[:3] / vector_norm
        rotation_world = quaternion_to_rotation_matrix(
            {"x": lower[0], "y": lower[1], "z": lower[2], "w": lower[3]}
        )
        interval_velocity[index] = rotation_world.dot(axis_local * (angle / dt_value))
    result[0] = interval_velocity[0]
    result[-1] = interval_velocity[-1]
    if count > 2:
        result[1:-1] = 0.5 * (interval_velocity[:-1] + interval_velocity[1:])
    return result


def track_twist_arrays(track):
    count = len(track.time_sec)
    linear_components = (
        getattr(track, "vx", None),
        getattr(track, "vy", None),
        getattr(track, "vz", None),
    )
    if all(component is not None and len(component) == count for component in linear_components):
        linear = np.column_stack(linear_components).astype(float)
    else:
        positions = np.column_stack((track.x, track.y, track.z)).astype(float)
        linear = _finite_difference_vectors(positions, track.time_sec)

    angular_components = (
        getattr(track, "wx", None),
        getattr(track, "wy", None),
        getattr(track, "wz", None),
    )
    if all(component is not None and len(component) == count for component in angular_components):
        angular = np.column_stack(angular_components).astype(float)
    else:
        angular = _derived_angular_velocity(track)
    return linear, angular


def _first_sustained_true(mask, sample_count):
    sample_count = max(1, int(sample_count))
    run_length = 0
    for index, active in enumerate(mask):
        run_length = run_length + 1 if bool(active) else 0
        if run_length >= sample_count:
            return index - sample_count + 1
    return None


def truth_track_metrics(track, policy=None):
    policy = policy or TruthMotionPolicy()
    x = _to_numpy(track.x)
    y = _to_numpy(track.y)
    z = _to_numpy(track.z)
    t = _to_numpy(track.time_sec)
    disp = np.sqrt((x - x[0]) ** 2 + (y - y[0]) ** 2 + (z - z[0]) ** 2)
    rotation_deg = track_rotation_angles_deg(track)
    linear_velocity, angular_velocity = track_twist_arrays(track)
    linear_speed = np.linalg.norm(linear_velocity, axis=1)
    angular_speed_deg = np.degrees(np.linalg.norm(angular_velocity, axis=1))
    moving_samples = (
        (disp >= policy.translation_deadband_m)
        | (rotation_deg >= policy.rotation_deadband_deg)
        | (
            (policy.linear_speed_deadband_mps > 0.0)
            & (linear_speed >= policy.linear_speed_deadband_mps)
        )
        | (
            (policy.angular_speed_deadband_degps > 0.0)
            & (angular_speed_deg >= policy.angular_speed_deadband_degps)
        )
    )
    start_candidate = _first_sustained_true(
        moving_samples, policy.sustained_motion_samples
    )
    moving = start_candidate is not None
    start_idx = start_candidate if moving else 0
    peak_idx = int(np.argmax(disp))
    linear_acceleration = _finite_difference_vectors(linear_velocity, t)
    angular_acceleration = _finite_difference_vectors(angular_velocity, t)
    max_abs_position = float(max(np.max(np.abs(x)), np.max(np.abs(y)), np.max(np.abs(z))))
    reasons = []
    if not (np.isfinite(x).all() and np.isfinite(y).all() and np.isfinite(z).all()):
        reasons.append("non_finite_position")
    if max_abs_position > OUTLIER_MAX_ABS_POSITION:
        reasons.append("abs_position_exceeds_threshold")
    if float(np.max(disp)) > OUTLIER_MAX_NET_DISPLACEMENT:
        reasons.append("peak_displacement_exceeds_threshold")

    return {
        "net_displacement": float(disp[-1]),
        "duration_sec": float(t[-1] - t[0]) if len(t) > 1 else 0.0,
        "start_time": float(t[start_idx]),
        "end_time": float(t[-1]),
        "peak_displacement": float(disp[peak_idx]),
        "peak_displacement_time": float(t[peak_idx]),
        "peak_rotation_deg": float(np.max(rotation_deg)),
        "peak_linear_speed_mps": float(np.max(linear_speed)),
        "peak_angular_speed_degps": float(np.max(angular_speed_deg)),
        "peak_linear_acceleration_mps2": float(
            np.max(np.linalg.norm(linear_acceleration, axis=1))
        ),
        "peak_angular_acceleration_degps2": float(
            np.max(np.degrees(np.linalg.norm(angular_acceleration, axis=1)))
        ),
        "moving": moving,
        "max_abs_position": max_abs_position,
        "outlier_reason": "|".join(reasons) if reasons else "",
    }


def select_bundle_motion_track(track, link_tracks, policy=None):
    candidates = [track] + list(link_tracks)
    candidate_metrics = [
        truth_track_metrics(candidate, policy=policy) for candidate in candidates
    ]
    best_index = max(
        range(len(candidates)),
        key=lambda index: (
            candidate_metrics[index]["peak_displacement"],
            candidate_metrics[index]["net_displacement"],
            candidate_metrics[index]["duration_sec"],
        ),
    )
    return candidates[best_index], candidate_metrics[best_index], candidate_metrics


def bundle_truth_metrics(track, link_tracks, policy=None):
    representative_track, representative_metrics, candidate_metrics = select_bundle_motion_track(
        track, link_tracks, policy=policy
    )
    moving_candidate_metrics = [item for item in candidate_metrics if item["moving"]]
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
    metrics["peak_rotation_deg"] = max(
        item["peak_rotation_deg"] for item in candidate_metrics
    )
    root_metrics = candidate_metrics[0]
    surface_metrics = candidate_metrics[1:]
    for field in (
        "peak_linear_speed_mps",
        "peak_angular_speed_degps",
        "peak_linear_acceleration_mps2",
        "peak_angular_acceleration_degps2",
    ):
        metrics[field] = root_metrics[field]
        metrics["surface_" + field] = max(
            (item[field] for item in surface_metrics),
            default=root_metrics[field],
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

    def optional_column(name):
        if name not in rows[0] or any(row.get(name, "") == "" for row in rows):
            return None
        return [float(row[name]) for row in rows]

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
        vx=optional_column("linear_velocity_x"),
        vy=optional_column("linear_velocity_y"),
        vz=optional_column("linear_velocity_z"),
        wx=optional_column("angular_velocity_x"),
        wy=optional_column("angular_velocity_y"),
        wz=optional_column("angular_velocity_z"),
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
        except ValueError as exc:
            raise ValueError(f"Invalid truth track {csv_path}: {exc}") from exc
    return tracks


def load_link_tracks(truth_links_dir):
    tracks = []
    truth_links_dir = pathlib.Path(truth_links_dir)
    if not truth_links_dir.is_dir():
        return tracks
    for csv_path in sorted(truth_links_dir.glob("*.csv")):
        try:
            tracks.append(load_link_track(csv_path))
        except ValueError as exc:
            raise ValueError(f"Invalid link truth track {csv_path}: {exc}") from exc
    return tracks


def load_surface_truth_link_tracks(run_dir, truth_tracks, direct_link_tracks=None):
    run_dir = pathlib.Path(run_dir)
    catalog_path = run_dir / "truth" / "surface_truth_points.jsonl"
    if not catalog_path.is_file():
        return []
    records = load_jsonl_optional(catalog_path) or []
    roots = {track.object_name: track for track in truth_tracks}
    direct_by_scoped_name = {
        track.scoped_link_name: track for track in (direct_link_tracks or [])
    }
    tracks = []
    for record in records:
        model_name = str(record.get("model_name", "")).strip()
        parent_scoped_name = str(
            record.get("motion_parent_scoped_link_name", "")
        ).strip()
        motion_parent = direct_by_scoped_name.get(parent_scoped_name)
        source_track = motion_parent or roots.get(model_name)
        local_pose = record.get("local_pose", {})
        local_position = _point_array(local_pose.get("position"))
        if source_track is None or local_position is None:
            continue
        x_values = []
        y_values = []
        z_values = []
        qx_values = []
        qy_values = []
        qz_values = []
        qw_values = []
        for sample_time in source_track.time_sec:
            parent_position, parent_rotation = _track_pose_at_time(
                source_track, sample_time
            )
            world_position = parent_rotation.dot(local_position) + parent_position
            parent_orientation = track_orientation_at_time(
                source_track, sample_time
            )
            x_values.append(float(world_position[0]))
            y_values.append(float(world_position[1]))
            z_values.append(float(world_position[2]))
            qx_values.append(parent_orientation["x"])
            qy_values.append(parent_orientation["y"])
            qz_values.append(parent_orientation["z"])
            qw_values.append(parent_orientation["w"])
        tracks.append(
            LinkTrack(
                scoped_link_name=str(record.get("scoped_link_name", "")),
                model_name=model_name,
                link_name=str(record.get("link_name", "")),
                time_sec=list(source_track.time_sec),
                x=x_values,
                y=y_values,
                z=z_values,
                qx=qx_values,
                qy=qy_values,
                qz=qz_values,
                qw=qw_values,
            )
        )
    return tracks


def load_jsonl_optional(path):
    path = pathlib.Path(path)
    if not path.exists():
        return None

    records = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                records.append(
                    json.loads(
                        stripped,
                        parse_constant=lambda token: (_ for _ in ()).throw(
                            ValueError(f"non-finite JSON constant {token}")
                        ),
                    )
                )
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
    return records


def load_algorithm_stream_optional(run_dir, stream_name, required=False):
    return ReplayableAlgorithmStream(
        run_dir,
        stream_name,
        required=required,
    )


def validate_schema_v2_run(
    run_dir,
    validation_policy="strict",
    max_anchor_processing_drop_fraction=None,
    report_path=None,
    stream_consumers=None,
):
    run_dir = pathlib.Path(run_dir)
    run_info_path = run_dir / "meta" / "run_info.json"
    if not run_info_path.is_file():
        return None
    try:
        run_info = json.loads(run_info_path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON at {run_info_path}:{exc.lineno}: {exc}") from exc
    try:
        schema_version = int(
            run_info.get("algorithm_recording", {}).get("schema_version", 0)
        )
    except (AttributeError, TypeError, ValueError):
        schema_version = 0
    if schema_version < 2:
        return None

    validator_path = pathlib.Path(__file__).with_name("validate_recorded_run.py")
    spec = importlib.util.spec_from_file_location(
        "validate_recorded_run_for_analysis", validator_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report = module.validate_run(
        run_dir,
        write_report=True,
        policy=validation_policy,
        max_anchor_processing_drop_fraction=(
            max_anchor_processing_drop_fraction
        ),
        report_path=report_path,
        stream_consumers=stream_consumers,
    )
    if not report.get("valid_for_analysis", False):
        raise DataQualityError(report)
    return report


def iter_reconstructed_compact_anchor_records(catalog_records, observation_records):
    catalog = {}
    for record in catalog_records or []:
        anchor = record.get("anchor", {})
        if not isinstance(anchor, dict):
            continue
        try:
            epoch = int(record.get("reference_epoch", anchor.get("reference_epoch", 0)))
            anchor_id = int(record.get("anchor_id", anchor.get("id", 0)))
        except (TypeError, ValueError):
            continue
        catalog[(epoch, anchor_id)] = dict(anchor)

    observations = observation_records if observation_records is not None else ()
    for observation in observations:
        try:
            epoch = int(observation.get("reference_epoch", 0))
        except (TypeError, ValueError):
            epoch = 0
        record = {
            "header": observation.get("header", {}),
            "reference_epoch": epoch,
            "reference_initialized_at": observation.get("reference_initialized_at"),
            "recorded_at": observation.get("recorded_at"),
            "anchors": [],
        }
        for dynamic in observation.get("anchors", []):
            if not isinstance(dynamic, dict):
                continue
            try:
                anchor_id = int(dynamic.get("id", 0))
            except (TypeError, ValueError):
                continue
            anchor = dict(catalog.get((epoch, anchor_id), {}))
            anchor.update(dynamic)
            anchor.setdefault("id", anchor_id)
            anchor.setdefault("reference_epoch", epoch)
            record["anchors"].append(anchor)
        yield record


def reconstruct_compact_anchor_records(catalog_records, observation_records):
    return list(
        iter_reconstructed_compact_anchor_records(
            catalog_records,
            observation_records,
        )
    )


def load_anchor_state_records(run_dir):
    run_dir = pathlib.Path(run_dir)
    algorithm_dir = run_dir / "algorithm"
    legacy = load_jsonl_optional(algorithm_dir / "anchor_states.jsonl")
    if legacy is not None:
        return legacy
    catalog = load_jsonl_optional(algorithm_dir / "anchor_catalog.jsonl")
    observations = load_algorithm_stream_optional(
        run_dir,
        "anchor_observations",
        required=catalog is not None,
    )
    if catalog is None and not observations:
        return None
    if catalog is None:
        raise ValueError(
            "Compact anchor records require anchor_catalog.jsonl and the logical "
            "anchor_observations stream"
        )
    return ReplayableSequence(
        lambda: iter_reconstructed_compact_anchor_records(catalog, observations)
    )


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


def load_object_id_catalog(world_file):
    world_file = resolve_world_file(world_file)
    if world_file is None:
        return {}
    root = ET.parse(world_file).getroot()
    world = root.find("world")
    if world is None:
        return {}

    catalog = {}
    for model in world.findall("model"):
        model_name = str(model.attrib.get("name", "")).strip()
        if not model_name:
            continue
        for retro_node in model.findall(".//collision/laser_retro"):
            try:
                raw_value = float((retro_node.text or "").strip())
            except (TypeError, ValueError):
                continue
            object_id = int(round(raw_value))
            if abs(raw_value - object_id) > 0.25 or object_id <= 0 or object_id > 254:
                continue
            existing = catalog.get(object_id)
            if existing is not None and existing != model_name:
                raise ValueError(
                    f"laser_retro object ID {object_id} is shared by "
                    f"'{existing}' and '{model_name}'"
                )
            catalog[object_id] = model_name
    return catalog


def load_recorded_object_id_catalog(run_dir):
    path = pathlib.Path(run_dir) / "meta" / "object_id_catalog.json"
    if not path.is_file():
        return {}
    with path.open() as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("recorded object_id_catalog.json must contain a mapping")
    catalog = {}
    for raw_id, raw_name in payload.items():
        object_id = int(raw_id)
        model_name = str(raw_name).strip()
        if object_id <= 0 or object_id > 254 or not model_name:
            raise ValueError(f"invalid recorded object catalog entry: {raw_id!r}")
        catalog[object_id] = model_name
    return catalog


def load_alarm_operating_point(run_dir):
    path = pathlib.Path(run_dir) / "meta" / "config_snapshot.json"
    if not path.is_file():
        return {"status": "missing"}
    with path.open() as handle:
        payload = json.load(handle)
    parameters = payload.get("parameters", {}) if isinstance(payload, dict) else {}
    deform_monitor = parameters.get("deform_monitor", {}) if isinstance(parameters, dict) else {}
    if not isinstance(deform_monitor, dict):
        deform_monitor = {}
    significance = deform_monitor.get("significance", {})
    directional = deform_monitor.get("directional_motion", {})
    persistent = deform_monitor.get("persistent_risk", {})
    significance = significance if isinstance(significance, dict) else {}
    directional = directional if isinstance(directional, dict) else {}
    persistent = persistent if isinstance(persistent, dict) else {}
    return {
        "status": "available",
        "final_mean_risk_threshold": persistent.get("min_confirmed_mean_risk", ""),
        "min_confirmed_confidence": persistent.get("min_confirmed_confidence", ""),
        "min_hits_to_confirm": persistent.get("min_hits_to_confirm", ""),
        "min_hit_streak_to_confirm": persistent.get("min_hit_streak_to_confirm", ""),
        "min_confirmed_support_mass": persistent.get("min_confirmed_support_mass", ""),
        "min_confirmed_span": persistent.get("min_confirmed_span", ""),
        "cusum_h": significance.get("cusum_h", ""),
        "directional_tau_s": directional.get("tau_s", ""),
        "directional_tau_c": directional.get("tau_c", ""),
    }


def merge_object_id_catalogs(*catalogs):
    merged = {}
    for catalog in catalogs:
        for object_id, model_name in catalog.items():
            existing = merged.get(object_id)
            if existing is not None and existing != model_name:
                raise ValueError(
                    f"object ID {object_id} maps to both '{existing}' and '{model_name}'"
                )
            merged[object_id] = model_name
    names_to_ids = {}
    for object_id, model_name in merged.items():
        existing_id = names_to_ids.get(model_name)
        if existing_id is not None and existing_id != object_id:
            raise ValueError(
                f"object '{model_name}' maps to both IDs {existing_id} and {object_id}"
            )
        names_to_ids[model_name] = object_id
    return merged


def validate_object_association_inputs(anchor_records, truth_tracks, object_id_catalog):
    if anchor_records is None:
        return
    if hasattr(anchor_records, "audit"):
        if anchor_records.audit().get("included_record_count", 0) == 0:
            return
    elif len(anchor_records) == 0:
        return
    if not object_id_catalog:
        raise ValueError(
            "object ID catalog is empty while anchor_states data is available; "
            "assign nonzero unique laser_retro IDs or record object_id_catalog"
        )

    truth_names = {track.object_name for track in truth_tracks}
    catalog_names = set(object_id_catalog.values())
    missing_truth_names = sorted(truth_names - catalog_names)
    if missing_truth_names:
        raise ValueError(
            "truth objects missing from object ID catalog: " + ", ".join(missing_truth_names)
        )

    valid_ids = set()
    for record in anchor_records:
        for anchor in record.get("anchors", []):
            if not anchor.get("object_id_valid", False):
                continue
            object_id = int(anchor.get("object_id", 0))
            if object_id not in object_id_catalog:
                raise ValueError(f"anchor references unknown object ID {object_id}")
            valid_ids.add(object_id)
    if not valid_ids:
        raise ValueError(
            "anchor_states contains no valid object associations; verify laser_retro IDs "
            "reach the point-cloud intensity field"
        )


def _safe_metric_ratio(numerator, denominator):
    return float(numerator) / float(denominator) if denominator else ""


def _safe_f1(precision, recall):
    if precision == "" or recall == "":
        return 0.0 if recall == 0.0 else ""
    if precision + recall <= 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _count_rising_edges(samples):
    episodes = 0
    previously_active = False
    for _, active in sorted(samples):
        if active and not previously_active:
            episodes += 1
        previously_active = bool(active)
    return episodes


def _new_anchor_type_stats():
    return {
        "anchors": set(),
        "anchor_geometry": {},
        "significant_times": [],
        "sample_count": 0,
        "observable_count": 0,
        "comparable_count": 0,
        "matched_count": 0,
        "loss_count": 0,
        "association_counts": {0: 0, 1: 0, 2: 0, 3: 0},
    }


ANCHOR_GEOMETRY_FIELDS = (
    "ref_quality",
    "covariance_quality",
    "type_stability",
    "shape_linearity",
    "shape_planarity",
    "shape_scattering",
)


def _finite_anchor_geometry(anchor):
    values = {}
    for field in ANCHOR_GEOMETRY_FIELDS:
        if field not in anchor:
            continue
        try:
            value = float(anchor[field])
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            values[field] = value
    return values


def _mean_anchor_geometry(stats, selected_types):
    entries = []
    for type_id in selected_types:
        entries.extend(stats[type_id]["anchor_geometry"].values())
    result = {}
    for field in ANCHOR_GEOMETRY_FIELDS:
        values = [entry[field] for entry in entries if field in entry]
        result[f"mean_{field}"] = float(np.mean(values)) if values else ""
    return result


def build_object_hit_exposure(object_observation_records, object_id_catalog):
    """Aggregate compact monitoring-window LiDAR exposure by object ID."""
    exposure = {
        int(object_id): {
            "object_id": int(object_id),
            "object_name": object_name,
            "monitoring_window_count": 0,
            "hit_window_count": 0,
            "frame_count": 0,
            "visible_frame_count": 0,
            "hit_point_count": 0,
            "lidar_observed": False,
            "_windows": [],
            "exposure_status": (
                "MISSING_HIT_STATS"
                if object_observation_records is None else "AVAILABLE"
            ),
        }
        for object_id, object_name in object_id_catalog.items()
    }
    if object_observation_records is None:
        return exposure
    for record in object_observation_records:
        if int(record.get("phase", 0)) != 1:
            continue
        frame_count = max(0, int(record.get("frame_count", 0)))
        window_start = time_sec_from_dict(record.get("window_start"))
        window_end = time_sec_from_dict(record.get("window_end"))
        record_time = record_time_sec(record)
        if window_start is None:
            window_start = record_time
        if window_end is None:
            window_end = record_time
        objects_in_window = {}
        for item in record.get("objects", []):
            try:
                object_id = int(item.get("object_id", 0))
            except (TypeError, ValueError):
                continue
            if object_id not in exposure:
                continue
            point_count = max(0, int(item.get("point_count", 0)))
            visible_frames = max(0, int(item.get("visible_frame_count", 0)))
            objects_in_window[object_id] = (point_count, visible_frames)
        for object_id, stats in exposure.items():
            point_count, visible_frames = objects_in_window.get(object_id, (0, 0))
            stats["monitoring_window_count"] += 1
            stats["frame_count"] += frame_count
            stats["hit_point_count"] += point_count
            stats["visible_frame_count"] += visible_frames
            stats["_windows"].append({
                "start_time": window_start,
                "end_time": window_end,
                "frame_count": frame_count,
                "point_count": point_count,
                "visible_frame_count": visible_frames,
            })
            if point_count > 0:
                stats["hit_window_count"] += 1
    for stats in exposure.values():
        stats["lidar_observed"] = stats["hit_point_count"] > 0
        stats["exposure_frame_count"] = stats["frame_count"]
        stats["exposure_window_count"] = stats["monitoring_window_count"]
        stats["lidar_visibility_rate"] = _safe_metric_ratio(
            stats["visible_frame_count"], stats["frame_count"]
        )
        stats["hit_window_rate"] = _safe_metric_ratio(
            stats["hit_window_count"], stats["monitoring_window_count"]
        )
    return exposure


def _object_exposure_for_interval(exposure, start_time=None, end_time=None):
    if not exposure:
        return {
            "lidar_observed": False,
            "hit_point_count": 0,
            "hit_window_count": 0,
            "visible_frame_count": 0,
            "exposure_frame_count": 0,
            "exposure_window_count": 0,
            "lidar_visibility_rate": "",
            "hit_window_rate": "",
        }
    windows = exposure.get("_windows", [])
    if not windows or start_time is None or end_time is None:
        return exposure
    selected = []
    for window in windows:
        window_start = window.get("start_time")
        window_end = window.get("end_time")
        if window_start is None or window_end is None:
            continue
        if float(window_end) >= float(start_time) and float(window_start) <= float(end_time):
            selected.append(window)
    point_count = sum(int(window["point_count"]) for window in selected)
    frame_count = sum(int(window["frame_count"]) for window in selected)
    visible_frame_count = sum(
        int(window["visible_frame_count"]) for window in selected
    )
    hit_window_count = sum(int(window["point_count"]) > 0 for window in selected)
    return {
        "lidar_observed": point_count > 0,
        "hit_point_count": point_count,
        "hit_window_count": hit_window_count,
        "visible_frame_count": visible_frame_count,
        "exposure_frame_count": frame_count,
        "exposure_window_count": len(selected),
        "lidar_visibility_rate": _safe_metric_ratio(
            visible_frame_count, frame_count
        ),
        "hit_window_rate": _safe_metric_ratio(hit_window_count, len(selected)),
    }


def build_anchor_detection_metrics(anchor_records, truth_tracks, link_tracks_by_model,
                                   object_id_catalog, truth_policy=None,
                                   object_exposure=None,
                                   inventory_stats=None,
                                   significant_records=None,
                                   record_consumer=None,
                                   enforce_associations=False,
                                   return_record_count=False):
    records = (
        record
        for record in (anchor_records if anchor_records is not None else ())
        if record_time_sec(record) is not None
    )
    truth_by_name = {track.object_name: track for track in truth_tracks}
    if enforce_associations and anchor_records is not None and not object_id_catalog:
        raise ValueError(
            "object ID catalog is empty while anchor_states data is available; "
            "assign nonzero unique laser_retro IDs or record object_id_catalog"
        )
    object_stats = {}
    frame_flags = {}
    valid_ids = set()
    record_count = 0
    first_record_time = None
    last_record_time = None

    for record in records:
        if record_consumer is not None:
            record_consumer(record)
        if inventory_stats is not None:
            _update_anchor_type_inventory(inventory_stats, record)
        time_sec = record_time_sec(record)
        record_count += 1
        if first_record_time is None:
            first_record_time = time_sec
        last_record_time = time_sec
        epoch = int(record.get("reference_epoch", 0))
        frame_active = {}
        significant_anchors = []
        for anchor in record.get("anchors", []):
            if bool(anchor.get("significant", False)) and significant_records is not None:
                significant_anchors.append(anchor)
            if not anchor.get("object_id_valid", False):
                continue
            object_id = int(anchor.get("object_id", 0))
            anchor_type = int(anchor.get("anchor_type", -1))
            if object_id not in object_id_catalog:
                if enforce_associations:
                    raise ValueError(f"anchor references unknown object ID {object_id}")
                continue
            valid_ids.add(object_id)
            if anchor_type not in ANCHOR_TYPE_NAMES:
                continue
            stats = object_stats.setdefault(
                object_id,
                {type_id: _new_anchor_type_stats() for type_id in ANCHOR_TYPE_NAMES},
            )
            anchor_key = (epoch, int(anchor.get("id", -1)))
            type_stats = stats[anchor_type]
            type_stats["anchors"].add(anchor_key)
            geometry = _finite_anchor_geometry(anchor)
            if geometry:
                type_stats["anchor_geometry"].setdefault(anchor_key, geometry)
            type_stats["sample_count"] += 1
            observable = bool(anchor.get("observable", False))
            comparable = bool(anchor.get("comparable", False))
            try:
                obs_state = int(anchor.get("obs_state", -1))
            except (TypeError, ValueError):
                obs_state = -1
            type_stats["observable_count"] += int(observable)
            type_stats["comparable_count"] += int(comparable)
            type_stats["matched_count"] += int(
                observable and comparable and obs_state == 1
            )
            type_stats["loss_count"] += int(obs_state in (3, 4))
            try:
                association_state = int(anchor.get("object_association_state", 0))
            except (TypeError, ValueError):
                association_state = 0
            if association_state not in type_stats["association_counts"]:
                association_state = 0
            type_stats["association_counts"][association_state] += 1
            significant = bool(anchor.get("significant", False))
            if significant:
                type_stats["significant_times"].append(time_sec)
            frame_active[(object_id, anchor_type)] = (
                frame_active.get((object_id, anchor_type), False) or significant
            )
        if significant_anchors and significant_records is not None:
            significant_records.append(
                {
                    "header": record.get("header", {}),
                    "recorded_at": record.get("recorded_at"),
                    "reference_epoch": record.get("reference_epoch", 0),
                    "reference_initialized_at": record.get(
                        "reference_initialized_at"
                    ),
                    "anchors": significant_anchors,
                }
            )
        for object_id in object_id_catalog:
            all_active = False
            for type_id in ANCHOR_TYPE_NAMES:
                active = frame_active.get((object_id, type_id), False)
                frame_flags.setdefault((object_id, type_id), []).append((time_sec, active))
                all_active = all_active or active
            frame_flags.setdefault((object_id, "ALL"), []).append((time_sec, all_active))

    if enforce_associations and record_count:
        missing_truth_names = sorted(
            set(truth_by_name) - set(object_id_catalog.values())
        )
        if missing_truth_names:
            raise ValueError(
                "truth objects missing from object ID catalog: "
                + ", ".join(missing_truth_names)
            )
        if not valid_ids:
            raise ValueError(
                "anchor_states contains no valid object associations; verify laser_retro IDs "
                "reach the point-cloud intensity field"
            )

    detail_rows = []
    scopes = [("ALL", None)] + [
        (name, type_id) for type_id, name in ANCHOR_TYPE_NAMES.items()
    ]
    for object_id, object_name in sorted(object_id_catalog.items()):
        track = truth_by_name.get(object_name)
        if track is None:
            continue
        links = link_tracks_by_model.get(object_name, [])
        classification = classify_truth_bundle(track, links, policy=truth_policy)
        truth_metrics = bundle_truth_metrics(track, links, policy=truth_policy)
        stats = object_stats.get(
            object_id,
            {type_id: _new_anchor_type_stats() for type_id in ANCHOR_TYPE_NAMES},
        )
        exposure = (object_exposure or {}).get(object_id, {})
        exposure_available = object_exposure is not None
        if exposure_available and classification == "moving":
            exposure = _object_exposure_for_interval(
                exposure, truth_metrics["start_time"], truth_metrics["end_time"]
            )
        lidar_observed = (
            bool(exposure.get("lidar_observed", False))
            if exposure_available else True
        )
        for scope_name, type_id in scopes:
            selected = list(ANCHOR_TYPE_NAMES) if type_id is None else [type_id]
            anchors = set().union(*(stats[item]["anchors"] for item in selected))
            significant_times = sorted({
                time_sec
                for item in selected
                for time_sec in stats[item]["significant_times"]
            })
            if classification == "moving":
                significant_times = [
                    time_sec for time_sec in significant_times
                    if truth_metrics["start_time"] <= time_sec <= truth_metrics["end_time"]
                ]
            eligible = bool(anchors)
            sample_count = sum(stats[item]["sample_count"] for item in selected)
            observable_count = sum(stats[item]["observable_count"] for item in selected)
            comparable_count = sum(stats[item]["comparable_count"] for item in selected)
            matched_count = sum(stats[item]["matched_count"] for item in selected)
            loss_count = sum(stats[item]["loss_count"] for item in selected)
            association_counts = {
                state: sum(
                    stats[item]["association_counts"][state] for item in selected
                )
                for state in (0, 1, 2, 3)
            }
            detected = eligible and bool(significant_times)
            first_detection = significant_times[0] if detected else ""
            delay = (
                first_detection - truth_metrics["start_time"]
                if detected and classification == "moving" else ""
            )
            if classification == "outlier":
                evaluation_status = "TRUTH_OUTLIER"
            elif exposure_available and not lidar_observed:
                evaluation_status = "LIDAR_UNOBSERVED"
            elif not eligible:
                evaluation_status = (
                    "OBSERVED_WITHOUT_ANCHOR"
                    if type_id is None else "OBSERVED_WITHOUT_TYPE_ANCHOR"
                )
            elif detected:
                evaluation_status = "EVALUABLE_DETECTED"
            elif classification == "moving":
                evaluation_status = "ELIGIBLE_WITHOUT_EVIDENCE"
            else:
                evaluation_status = "EVALUABLE_NO_FALSE_EVIDENCE"
            if evaluation_status in {
                "TRUTH_OUTLIER", "LIDAR_UNOBSERVED", "OBSERVED_WITHOUT_ANCHOR",
                "OBSERVED_WITHOUT_TYPE_ANCHOR",
            }:
                outcome = "NOT_EVALUABLE"
            elif classification == "moving":
                outcome = "TP" if detected else "FN"
            else:
                outcome = "FP" if detected else "TN"
            false_episodes = (
                _count_rising_edges(frame_flags.get((object_id, scope_name if type_id is None else type_id), []))
                if eligible and classification == "static" else 0
            )
            geometry_summary = _mean_anchor_geometry(stats, selected)
            detail_rows.append({
                "object_name": object_name,
                "object_id": object_id,
                "anchor_type": scope_name,
                "classification": classification,
                "evaluation_status": evaluation_status,
                "lidar_observed": lidar_observed,
                "hit_point_count": int(exposure.get("hit_point_count", 0)),
                "hit_window_count": int(exposure.get("hit_window_count", 0)),
                "visible_frame_count": int(exposure.get("visible_frame_count", 0)),
                "exposure_frame_count": int(exposure.get("exposure_frame_count", 0)),
                "exposure_window_count": int(exposure.get("exposure_window_count", 0)),
                "lidar_visibility_rate": exposure.get("lidar_visibility_rate", ""),
                "hit_window_rate": exposure.get("hit_window_rate", ""),
                "eligible_anchor_count": len(anchors),
                **geometry_summary,
                "anchor_sample_count": sample_count,
                "mean_anchor_count_per_frame": (
                    sample_count / float(record_count) if record_count else ""
                ),
                "observable_sample_count": observable_count,
                "observable_rate": _safe_metric_ratio(observable_count, sample_count),
                "comparable_sample_count": comparable_count,
                "comparable_rate": _safe_metric_ratio(comparable_count, sample_count),
                "matched_sample_count": matched_count,
                "matched_rate": _safe_metric_ratio(matched_count, sample_count),
                "loss_sample_count": loss_count,
                "loss_rate": _safe_metric_ratio(loss_count, sample_count),
                "association_consistent_sample_count": association_counts[1],
                "association_mismatch_sample_count": association_counts[2],
                "association_mixed_sample_count": association_counts[3],
                "association_unavailable_sample_count": association_counts[0],
                "association_consistency_rate": _safe_metric_ratio(
                    association_counts[1], sample_count
                ),
                "significant_frame_count": len(significant_times),
                "detected": detected,
                "first_detection_time": first_detection,
                "detection_delay_sec": delay,
                "false_alarm_episodes": false_episodes,
                "outcome": outcome,
            })

    duration_sec = 0.0
    if record_count >= 2:
        duration_sec = max(0.0, last_record_time - first_record_time)
    aggregate_rows = []
    for scope_name, _ in scopes:
        rows = [row for row in detail_rows if row["anchor_type"] == scope_name]
        counts = {
            name.lower(): sum(row["outcome"] == name for row in rows)
            for name in ("TP", "FP", "FN", "TN")
        }
        not_evaluable = sum(row["outcome"] == "NOT_EVALUABLE" for row in rows)
        eligible_anchor_count = sum(int(row["eligible_anchor_count"]) for row in rows)
        geometry_summary = {}
        for field in ANCHOR_GEOMETRY_FIELDS:
            output_field = f"mean_{field}"
            weighted_values = [
                (float(row[output_field]), int(row["eligible_anchor_count"]))
                for row in rows
                if row[output_field] != "" and int(row["eligible_anchor_count"]) > 0
            ]
            weight_sum = sum(weight for _, weight in weighted_values)
            geometry_summary[output_field] = (
                sum(value * weight for value, weight in weighted_values) / weight_sum
                if weight_sum > 0 else ""
            )
        anchor_sample_count = sum(int(row["anchor_sample_count"]) for row in rows)
        observable_count = sum(int(row["observable_sample_count"]) for row in rows)
        comparable_count = sum(int(row["comparable_sample_count"]) for row in rows)
        matched_count = sum(int(row["matched_sample_count"]) for row in rows)
        loss_count = sum(int(row["loss_sample_count"]) for row in rows)
        association_counts = {
            state: sum(int(row[field]) for row in rows)
            for state, field in {
                0: "association_unavailable_sample_count",
                1: "association_consistent_sample_count",
                2: "association_mismatch_sample_count",
                3: "association_mixed_sample_count",
            }.items()
        }
        precision = _safe_metric_ratio(counts["tp"], counts["tp"] + counts["fp"])
        recall = _safe_metric_ratio(counts["tp"], counts["tp"] + counts["fn"])
        f1 = ""
        if precision != "" and recall != "" and (precision + recall) > 0.0:
            f1 = 2.0 * precision * recall / (precision + recall)
        false_episodes = sum(int(row["false_alarm_episodes"]) for row in rows)
        false_rate = false_episodes / (duration_sec / 60.0) if duration_sec > 0.0 else ""
        aggregate_rows.append({
            "anchor_type": scope_name,
            **counts,
            "not_evaluable": not_evaluable,
            "eligible_anchor_count": eligible_anchor_count,
            **geometry_summary,
            "anchor_sample_count": anchor_sample_count,
            "observable_sample_count": observable_count,
            "observable_rate": _safe_metric_ratio(observable_count, anchor_sample_count),
            "comparable_sample_count": comparable_count,
            "comparable_rate": _safe_metric_ratio(comparable_count, anchor_sample_count),
            "matched_sample_count": matched_count,
            "matched_rate": _safe_metric_ratio(matched_count, anchor_sample_count),
            "loss_sample_count": loss_count,
            "loss_rate": _safe_metric_ratio(loss_count, anchor_sample_count),
            "association_consistent_sample_count": association_counts[1],
            "association_mismatch_sample_count": association_counts[2],
            "association_mixed_sample_count": association_counts[3],
            "association_unavailable_sample_count": association_counts[0],
            "association_consistency_rate": _safe_metric_ratio(
                association_counts[1], anchor_sample_count
            ),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "false_alarm_episodes": false_episodes,
            "false_alarm_rate_per_min": false_rate,
        })
    if return_record_count:
        return detail_rows, aggregate_rows, record_count
    return detail_rows, aggregate_rows


def _initialize_anchor_type_inventory(catalog_records, formal_epoch=None):
    stats = {
        type_id: {
            "catalog_anchors": set(),
            "observed_anchors": set(),
            "significant_anchors": set(),
            "observation_sample_count": 0,
            "observable_sample_count": 0,
            "comparable_sample_count": 0,
            "matched_sample_count": 0,
            "significant_sample_count": 0,
            "object_id_valid_sample_count": 0,
            "association_counts": {0: 0, 1: 0, 2: 0, 3: 0},
        }
        for type_id in ANCHOR_TYPE_NAMES
    }
    for record in catalog_records or ():
        anchor = record.get("anchor", {})
        if not isinstance(anchor, dict):
            continue
        try:
            epoch = int(record.get("reference_epoch", anchor.get("reference_epoch", 0)))
            anchor_id = int(record.get("anchor_id", anchor.get("id", -1)))
            anchor_type = int(anchor.get("anchor_type", -1))
        except (TypeError, ValueError, OverflowError):
            continue
        if formal_epoch is not None and epoch != int(formal_epoch):
            continue
        if anchor_type in stats:
            stats[anchor_type]["catalog_anchors"].add((epoch, anchor_id))
    return stats


def _update_anchor_type_inventory(stats, record):
    try:
        epoch = int(record.get("reference_epoch", 0))
    except (TypeError, ValueError, OverflowError):
        epoch = 0
    for anchor in record.get("anchors", []):
        try:
            anchor_id = int(anchor.get("id", -1))
            anchor_type = int(anchor.get("anchor_type", -1))
        except (TypeError, ValueError, OverflowError):
            continue
        if anchor_type not in stats:
            continue
        item = stats[anchor_type]
        key = (epoch, anchor_id)
        item["observed_anchors"].add(key)
        item["observation_sample_count"] += 1
        observable = bool(anchor.get("observable", False))
        comparable = bool(anchor.get("comparable", False))
        try:
            obs_state = int(anchor.get("obs_state", -1))
        except (TypeError, ValueError, OverflowError):
            obs_state = -1
        item["observable_sample_count"] += int(observable)
        item["comparable_sample_count"] += int(comparable)
        item["matched_sample_count"] += int(
            observable and comparable and obs_state == 1
        )
        significant = bool(anchor.get("significant", False))
        item["significant_sample_count"] += int(significant)
        if significant:
            item["significant_anchors"].add(key)
        item["object_id_valid_sample_count"] += int(
            bool(anchor.get("object_id_valid", False))
        )
        try:
            association = int(anchor.get("object_association_state", 0))
        except (TypeError, ValueError, OverflowError):
            association = 0
        if association not in item["association_counts"]:
            association = 0
        item["association_counts"][association] += 1


def _finalize_anchor_type_inventory(stats):
    rows = []
    for type_id, type_name in ANCHOR_TYPE_NAMES.items():
        item = stats[type_id]
        rows.append(
            {
                "anchor_type": type_name,
                "catalog_anchor_count": len(item["catalog_anchors"]),
                "observed_anchor_count": len(item["observed_anchors"]),
                "observation_sample_count": item["observation_sample_count"],
                "observable_sample_count": item["observable_sample_count"],
                "comparable_sample_count": item["comparable_sample_count"],
                "matched_sample_count": item["matched_sample_count"],
                "significant_sample_count": item["significant_sample_count"],
                "significant_anchor_count": len(item["significant_anchors"]),
                "object_id_valid_sample_count": item["object_id_valid_sample_count"],
                "association_consistent_sample_count": item["association_counts"][1],
                "association_mismatch_sample_count": item["association_counts"][2],
                "association_mixed_sample_count": item["association_counts"][3],
                "association_unavailable_sample_count": item["association_counts"][0],
            }
        )
    return rows


def build_anchor_type_inventory(catalog_records, anchor_records, formal_epoch=None):
    """Count all anchors before object-association/evaluability filtering."""

    stats = _initialize_anchor_type_inventory(catalog_records, formal_epoch)
    for record in anchor_records or ():
        _update_anchor_type_inventory(stats, record)
    return _finalize_anchor_type_inventory(stats)


def _track_time_bracket(track, time_sec):
    t = track.time_sec
    if len(t) == 0:
        raise ValueError("truth track has no samples")
    idx = bisect.bisect_left(t, float(time_sec))
    if idx <= 0:
        return 0, 0, 0.0
    if idx >= len(t):
        last = len(t) - 1
        return last, last, 0.0
    lower = idx - 1
    upper = idx
    duration = float(t[upper] - t[lower])
    if duration <= 0.0:
        return upper, upper, 0.0
    alpha = (float(time_sec) - float(t[lower])) / duration
    return lower, upper, min(1.0, max(0.0, alpha))


def track_position_at_time(track, time_sec):
    lower, upper, alpha = _track_time_bracket(track, time_sec)
    position_lower = np.array(
        [track.x[lower], track.y[lower], track.z[lower]], dtype=float
    )
    position_upper = np.array(
        [track.x[upper], track.y[upper], track.z[upper]], dtype=float
    )
    position = ((1.0 - alpha) * position_lower) + (alpha * position_upper)
    return {
        "x": float(position[0]),
        "y": float(position[1]),
        "z": float(position[2]),
    }


def _normalized_quaternion(values):
    quaternion = np.asarray(values, dtype=float)
    norm = float(np.linalg.norm(quaternion))
    if not math.isfinite(norm) or norm <= 1.0e-12:
        return np.array([0.0, 0.0, 0.0, 1.0])
    return quaternion / norm


def _slerp_quaternion(lower, upper, alpha):
    lower = _normalized_quaternion(lower)
    upper = _normalized_quaternion(upper)
    dot = float(np.dot(lower, upper))
    if dot < 0.0:
        upper = -upper
        dot = -dot
    dot = min(1.0, max(-1.0, dot))
    if dot > 0.9995:
        return _normalized_quaternion(((1.0 - alpha) * lower) + (alpha * upper))
    angle = math.acos(dot)
    sine = math.sin(angle)
    if abs(sine) <= 1.0e-12:
        return lower
    return (
        (math.sin((1.0 - alpha) * angle) / sine) * lower
        + (math.sin(alpha * angle) / sine) * upper
    )


def track_orientation_at_time(track, time_sec):
    qx = getattr(track, "qx", None)
    qy = getattr(track, "qy", None)
    qz = getattr(track, "qz", None)
    qw = getattr(track, "qw", None)
    if any(component is None or len(component) == 0 for component in (qx, qy, qz, qw)):
        return {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}
    lower, upper, alpha = _track_time_bracket(track, time_sec)
    q_lower = [qx[lower], qy[lower], qz[lower], qw[lower]]
    q_upper = [qx[upper], qy[upper], qz[upper], qw[upper]]
    interpolated = _slerp_quaternion(q_lower, q_upper, alpha)
    return {
        "x": float(interpolated[0]),
        "y": float(interpolated[1]),
        "z": float(interpolated[2]),
        "w": float(interpolated[3]),
    }


def _point_array(point):
    if not isinstance(point, dict):
        return None
    try:
        result = np.array(
            [float(point["x"]), float(point["y"]), float(point["z"])],
            dtype=float,
        )
    except (KeyError, TypeError, ValueError):
        return None
    return result if np.isfinite(result).all() else None


def _track_pose_at_time(track, time_sec):
    position = _point_array(track_position_at_time(track, time_sec))
    rotation = quaternion_to_rotation_matrix(track_orientation_at_time(track, time_sec))
    return position, rotation


def _interpolate_vector_samples(samples, track, time_sec):
    lower, upper, alpha = _track_time_bracket(track, time_sec)
    lower_value = np.asarray(samples[lower], dtype=float)
    upper_value = np.asarray(samples[upper], dtype=float)
    return ((1.0 - alpha) * lower_value) + (alpha * upper_value)


_TRUTH_BOUNDARY_TOLERANCE_SEC = 0.15


def _cropped_sample_times(track, start_time, end_time):
    if not track.time_sec:
        raise ValueError("truth track has no samples")
    track_start = float(track.time_sec[0])
    track_end = float(track.time_sec[-1])
    if track_start > start_time + _TRUTH_BOUNDARY_TOLERANCE_SEC:
        raise ValueError(
            "truth track does not cover formal interval [{}, {}]".format(
                start_time, end_time
            )
        )
    if track_end < end_time - _TRUTH_BOUNDARY_TOLERANCE_SEC:
        raise ValueError(
            "truth track does not cover formal interval [{}, {}]".format(
                start_time, end_time
            )
        )
    # Clamp boundaries to available data when within tolerance.
    start_time = max(start_time, track_start)
    end_time = min(end_time, track_end)
    selected = [float(start_time)]
    selected.extend(
        float(value)
        for value in track.time_sec
        if start_time < float(value) < end_time
    )
    if end_time > start_time:
        selected.append(float(end_time))
    output = []
    for value in selected:
        if not output or abs(value - output[-1]) > 1.0e-9:
            output.append(value)
    return output


def _interpolate_scalar_samples(samples, track, sample_times):
    if samples is None:
        return None
    return [
        float(_interpolate_vector_samples(samples, track, sample_time))
        for sample_time in sample_times
    ]


def crop_track_to_interval(track, start_time, end_time):
    """Interpolate a rigid truth track exactly onto a closed time interval."""

    start_time = float(start_time)
    end_time = float(end_time)
    sample_times = _cropped_sample_times(track, start_time, end_time)
    positions = [track_position_at_time(track, value) for value in sample_times]
    orientations = [track_orientation_at_time(track, value) for value in sample_times]
    common = {
        "time_sec": sample_times,
        "x": [value["x"] for value in positions],
        "y": [value["y"] for value in positions],
        "z": [value["z"] for value in positions],
        "qx": [value["x"] for value in orientations],
        "qy": [value["y"] for value in orientations],
        "qz": [value["z"] for value in orientations],
        "qw": [value["w"] for value in orientations],
    }
    if isinstance(track, TruthTrack):
        return TruthTrack(
            object_name=track.object_name,
            vx=_interpolate_scalar_samples(track.vx, track, sample_times),
            vy=_interpolate_scalar_samples(track.vy, track, sample_times),
            vz=_interpolate_scalar_samples(track.vz, track, sample_times),
            wx=_interpolate_scalar_samples(track.wx, track, sample_times),
            wy=_interpolate_scalar_samples(track.wy, track, sample_times),
            wz=_interpolate_scalar_samples(track.wz, track, sample_times),
            **common,
        )
    if isinstance(track, LinkTrack):
        return LinkTrack(
            scoped_link_name=track.scoped_link_name,
            model_name=track.model_name,
            link_name=track.link_name,
            **common,
        )
    raise TypeError("unsupported truth track type: {}".format(type(track).__name__))


def crop_tracks_to_scope(tracks, scope):
    if scope is None:
        return tracks, []
    cropped = []
    audit = []
    for track in tracks:
        result = crop_track_to_interval(
            track, scope.trial_start, scope.trial_end
        )
        cropped.append(result)
        audit.append(
            {
                "track_name": getattr(
                    track, "object_name", getattr(track, "scoped_link_name", "")
                ),
                "input_sample_count": len(track.time_sec),
                "formal_sample_count": len(result.time_sec),
                "formal_first_time_sec": result.time_sec[0],
                "formal_last_time_sec": result.time_sec[-1],
            }
        )
    return cropped, audit


def rigid_point_truth_state(
    track,
    reference_point_world,
    reference_time_sec,
    time_sec,
    twist_samples=None,
):
    reference_point_world = np.asarray(reference_point_world, dtype=float)
    reference_position, reference_rotation = _track_pose_at_time(
        track, reference_time_sec
    )
    current_position, current_rotation = _track_pose_at_time(track, time_sec)
    point_object = reference_rotation.T.dot(
        reference_point_world - reference_position
    )
    current_radius_world = current_rotation.dot(point_object)
    point_world = current_position + current_radius_world
    linear_samples, angular_samples = (
        twist_samples if twist_samples is not None else track_twist_arrays(track)
    )
    linear_velocity = _interpolate_vector_samples(linear_samples, track, time_sec)
    angular_velocity = _interpolate_vector_samples(angular_samples, track, time_sec)
    point_velocity = linear_velocity + np.cross(
        angular_velocity, current_radius_world
    )
    return {
        "point_world": point_world,
        "displacement_world": point_world - reference_point_world,
        "velocity_world": point_velocity,
        "linear_velocity_world": linear_velocity,
        "angular_velocity_world_radps": angular_velocity,
        "radius_world": current_radius_world,
    }


def _track_contains_time(track, time_sec, tolerance=1.0e-6):
    if not track.time_sec:
        return False
    return (
        float(track.time_sec[0]) - tolerance
        <= float(time_sec)
        <= float(track.time_sec[-1]) + tolerance
    )


def _empty_anchor_vector_row(record, anchor, object_id, object_name, classification,
                             time_sec, reference_time_sec):
    disp_mean = anchor.get("disp_mean", [])
    estimated = None
    if isinstance(disp_mean, (list, tuple)) and len(disp_mean) == 3:
        try:
            candidate = np.asarray(disp_mean, dtype=float)
            if np.isfinite(candidate).all():
                estimated = candidate
        except (TypeError, ValueError):
            estimated = None
    vel_mean = anchor.get("vel_mean", [])
    estimated_velocity = None
    if isinstance(vel_mean, (list, tuple)) and len(vel_mean) == 3:
        try:
            candidate = np.asarray(vel_mean, dtype=float)
            if np.isfinite(candidate).all():
                estimated_velocity = candidate
        except (TypeError, ValueError):
            estimated_velocity = None
    return {
        "object_name": object_name,
        "object_id": object_id,
        "classification": classification,
        "reference_epoch": int(
            anchor.get("reference_epoch", record.get("reference_epoch", 0))
        ),
        "anchor_id": int(anchor.get("id", -1)),
        "anchor_type": ANCHOR_TYPE_NAMES.get(int(anchor.get("anchor_type", -1)), "UNKNOWN"),
        "time_sec": time_sec,
        "reference_time_sec": reference_time_sec if reference_time_sec is not None else "",
        "estimated_dx": float(estimated[0]) if estimated is not None else "",
        "estimated_dy": float(estimated[1]) if estimated is not None else "",
        "estimated_dz": float(estimated[2]) if estimated is not None else "",
        "estimated_magnitude": float(np.linalg.norm(estimated)) if estimated is not None else "",
        "estimated_vx": float(estimated_velocity[0]) if estimated_velocity is not None else "",
        "estimated_vy": float(estimated_velocity[1]) if estimated_velocity is not None else "",
        "estimated_vz": float(estimated_velocity[2]) if estimated_velocity is not None else "",
        "estimated_speed": (
            float(np.linalg.norm(estimated_velocity))
            if estimated_velocity is not None else ""
        ),
        "expected_dx": "",
        "expected_dy": "",
        "expected_dz": "",
        "expected_magnitude": "",
        "expected_vx": "",
        "expected_vy": "",
        "expected_vz": "",
        "expected_speed": "",
        "vector_error_norm": "",
        "magnitude_error_abs": "",
        "direction_error_deg": "",
        "velocity_vector_error_norm": "",
        "velocity_direction_error_deg": "",
        "valid": False,
        "invalid_reason": "",
    }, estimated, estimated_velocity


def _macro_mean(rows, unit_fields, value_field):
    units = {}
    for row in rows:
        if row.get(value_field, "") == "":
            continue
        try:
            value = float(row[value_field])
        except (TypeError, ValueError):
            continue
        if not math.isfinite(value):
            continue
        key = tuple(row.get(field) for field in unit_fields)
        units.setdefault(key, []).append(value)
    return (
        float(np.mean([np.mean(values) for values in units.values()]))
        if units else ""
    )


def summarize_anchor_vector_errors(rows, anchor_type):
    selected = [
        row for row in rows
        if (anchor_type == "ALL" or row.get("anchor_type") == anchor_type)
    ]
    valid_rows = [row for row in selected if bool(row.get("valid", False))]
    vector_errors = [float(row["vector_error_norm"]) for row in valid_rows]
    magnitude_errors = [float(row["magnitude_error_abs"]) for row in valid_rows]
    direction_errors = [
        float(row["direction_error_deg"]) for row in valid_rows
        if row.get("direction_error_deg", "") != ""
    ]
    velocity_errors = [
        float(row["velocity_vector_error_norm"]) for row in valid_rows
        if row.get("velocity_vector_error_norm", "") != ""
    ]
    velocity_direction_errors = [
        float(row["velocity_direction_error_deg"]) for row in valid_rows
        if row.get("velocity_direction_error_deg", "") != ""
    ]
    anchor_units = ("reference_epoch", "anchor_id")
    object_type_units = ("object_id", "anchor_type")
    return {
        "anchor_type": anchor_type,
        "observation_count": len(selected),
        "valid_observation_count": len(valid_rows),
        "invalid_observation_count": len(selected) - len(valid_rows),
        "anchor_count": len({
            (row.get("reference_epoch"), row.get("anchor_id")) for row in selected
        }),
        "object_count": len({row.get("object_id") for row in selected}),
        "vector_error_mean": float(np.mean(vector_errors)) if vector_errors else "",
        "vector_error_rmse": (
            float(math.sqrt(np.mean(np.square(vector_errors))))
            if vector_errors else ""
        ),
        "magnitude_error_mae": (
            float(np.mean(magnitude_errors)) if magnitude_errors else ""
        ),
        "magnitude_error_rmse": (
            float(math.sqrt(np.mean(np.square(magnitude_errors))))
            if magnitude_errors else ""
        ),
        "direction_error_mean_deg": (
            float(np.mean(direction_errors)) if direction_errors else ""
        ),
        "direction_error_median_deg": (
            float(np.median(direction_errors)) if direction_errors else ""
        ),
        "velocity_vector_error_mean": (
            float(np.mean(velocity_errors)) if velocity_errors else ""
        ),
        "velocity_vector_error_rmse": (
            float(math.sqrt(np.mean(np.square(velocity_errors))))
            if velocity_errors else ""
        ),
        "velocity_direction_error_mean_deg": (
            float(np.mean(velocity_direction_errors))
            if velocity_direction_errors else ""
        ),
        "per_anchor_macro_vector_error_mean": _macro_mean(
            valid_rows, anchor_units, "vector_error_norm"
        ),
        "per_anchor_macro_magnitude_error_mae": _macro_mean(
            valid_rows, anchor_units, "magnitude_error_abs"
        ),
        "per_anchor_macro_direction_error_mean_deg": _macro_mean(
            valid_rows, anchor_units, "direction_error_deg"
        ),
        "per_anchor_macro_velocity_vector_error_mean": _macro_mean(
            valid_rows, anchor_units, "velocity_vector_error_norm"
        ),
        "per_anchor_macro_velocity_direction_error_mean_deg": _macro_mean(
            valid_rows, anchor_units, "velocity_direction_error_deg"
        ),
        "object_type_macro_vector_error_mean": _macro_mean(
            valid_rows, object_type_units, "vector_error_norm"
        ),
        "object_type_macro_magnitude_error_mae": _macro_mean(
            valid_rows, object_type_units, "magnitude_error_abs"
        ),
        "object_type_macro_direction_error_mean_deg": _macro_mean(
            valid_rows, object_type_units, "direction_error_deg"
        ),
        "object_type_macro_velocity_vector_error_mean": _macro_mean(
            valid_rows, object_type_units, "velocity_vector_error_norm"
        ),
        "object_type_macro_velocity_direction_error_mean_deg": _macro_mean(
            valid_rows, object_type_units, "velocity_direction_error_deg"
        ),
    }


def build_anchor_vector_metrics(anchor_records, truth_tracks, link_tracks_by_model,
                                object_id_catalog, alignment, truth_policy=None):
    """Compare only emitted anchor displacement evidence against rigid-body truth."""
    truth_by_name = {track.object_name: track for track in truth_tracks}
    classification_by_name = {
        object_name: classify_truth_bundle(
            track,
            link_tracks_by_model.get(object_name, []),
            policy=truth_policy,
        )
        for object_name, track in truth_by_name.items()
    }
    twist_by_name = {
        object_name: track_twist_arrays(track)
        for object_name, track in truth_by_name.items()
    }
    detail_rows = []
    records = (
        record
        for record in (anchor_records if anchor_records is not None else ())
        if record_time_sec(record) is not None
    )

    for record in records:
        time_sec = record_time_sec(record)
        for anchor in record.get("anchors", []):
            if not bool(anchor.get("significant", False)):
                continue
            if not bool(anchor.get("object_id_valid", False)):
                continue
            object_id = int(anchor.get("object_id", 0))
            anchor_type = int(anchor.get("anchor_type", -1))
            if object_id not in object_id_catalog or anchor_type not in ANCHOR_TYPE_NAMES:
                continue
            object_name = object_id_catalog[object_id]
            track = truth_by_name.get(object_name)
            classification = (
                classification_by_name[object_name]
                if track is not None else "missing_truth"
            )
            reference_time_sec = time_sec_from_dict(anchor.get("reference_stamp"))
            if reference_time_sec is None:
                reference_time_sec = time_sec_from_dict(record.get("reference_initialized_at"))
            row, estimated, estimated_velocity = _empty_anchor_vector_row(
                record,
                anchor,
                object_id,
                object_name,
                classification,
                time_sec,
                reference_time_sec,
            )

            invalid_reason = ""
            if "object_association_state" in anchor:
                try:
                    association_state = int(anchor.get("object_association_state", 0))
                except (TypeError, ValueError):
                    association_state = 0
                association_reasons = {
                    0: "object_association_unavailable",
                    2: "object_association_mismatch",
                    3: "object_association_mixed",
                }
                invalid_reason = association_reasons.get(association_state, "")
            if not invalid_reason and alignment is None:
                invalid_reason = "alignment_unavailable"
            elif not invalid_reason and track is None:
                invalid_reason = "truth_track_missing"
            elif not invalid_reason and reference_time_sec is None:
                invalid_reason = "reference_time_missing"
            elif not invalid_reason and not _track_contains_time(track, reference_time_sec):
                invalid_reason = "reference_time_outside_truth_track"
            elif not invalid_reason and not _track_contains_time(track, time_sec):
                invalid_reason = "observation_time_outside_truth_track"
            elif not invalid_reason and estimated is None:
                invalid_reason = "estimated_displacement_invalid"

            reference_center = _point_array(anchor.get("ref_center"))
            if not invalid_reason and reference_center is None:
                invalid_reason = "reference_center_invalid"
            if invalid_reason:
                row["invalid_reason"] = invalid_reason
                detail_rows.append(row)
                continue

            alignment_rotation = np.asarray(alignment["rotation"], dtype=float)
            alignment_translation = np.asarray(alignment["translation"], dtype=float)
            reference_center_world = (
                alignment_rotation.dot(reference_center) + alignment_translation
            )
            truth_state = rigid_point_truth_state(
                track,
                reference_point_world=reference_center_world,
                reference_time_sec=reference_time_sec,
                time_sec=time_sec,
                twist_samples=twist_by_name[object_name],
            )
            expected_world = truth_state["displacement_world"]
            expected = alignment_rotation.T.dot(expected_world)
            expected_velocity = alignment_rotation.T.dot(
                truth_state["velocity_world"]
            )

            estimated_magnitude = float(np.linalg.norm(estimated))
            expected_magnitude = float(np.linalg.norm(expected))
            vector_error = float(np.linalg.norm(estimated - expected))
            magnitude_error = abs(estimated_magnitude - expected_magnitude)
            direction_error = ""
            if estimated_magnitude > 1.0e-12 and expected_magnitude > 1.0e-12:
                cosine = float(np.dot(estimated, expected)) / (
                    estimated_magnitude * expected_magnitude
                )
                cosine = min(1.0, max(-1.0, cosine))
                direction_error = (
                    0.0 if cosine >= 1.0 - 1.0e-12
                    else math.degrees(math.acos(cosine))
                )
            expected_speed = float(np.linalg.norm(expected_velocity))
            velocity_error = ""
            velocity_direction_error = ""
            if estimated_velocity is not None:
                estimated_speed = float(np.linalg.norm(estimated_velocity))
                velocity_error = float(
                    np.linalg.norm(estimated_velocity - expected_velocity)
                )
                if estimated_speed > 1.0e-12 and expected_speed > 1.0e-12:
                    velocity_cosine = float(
                        np.dot(estimated_velocity, expected_velocity)
                    ) / (estimated_speed * expected_speed)
                    velocity_cosine = min(1.0, max(-1.0, velocity_cosine))
                    velocity_direction_error = (
                        0.0
                        if velocity_cosine >= 1.0 - 1.0e-12
                        else math.degrees(math.acos(velocity_cosine))
                    )

            row.update({
                "expected_dx": float(expected[0]),
                "expected_dy": float(expected[1]),
                "expected_dz": float(expected[2]),
                "expected_magnitude": expected_magnitude,
                "expected_vx": float(expected_velocity[0]),
                "expected_vy": float(expected_velocity[1]),
                "expected_vz": float(expected_velocity[2]),
                "expected_speed": expected_speed,
                "vector_error_norm": vector_error,
                "magnitude_error_abs": magnitude_error,
                "direction_error_deg": direction_error,
                "velocity_vector_error_norm": velocity_error,
                "velocity_direction_error_deg": velocity_direction_error,
                "valid": True,
                "invalid_reason": "",
            })
            detail_rows.append(row)

    aggregate_rows = []
    scopes = [("ALL", None)] + [
        (name, type_id) for type_id, name in ANCHOR_TYPE_NAMES.items()
    ]
    for scope_name, type_id in scopes:
        aggregate_rows.append(summarize_anchor_vector_errors(detail_rows, scope_name))
    return detail_rows, aggregate_rows


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


def build_surface_truth_bbox_world(link_tracks, time_sec, margin=TRUTH_BBOX_MARGIN):
    surface_tracks = [
        track for track in link_tracks
        if str(getattr(track, "link_name", "")).startswith("ground_truth_")
    ]
    if len(surface_tracks) < 3:
        return None
    aabb = _aabb_from_points([
        track_position_at_time(track, time_sec) for track in surface_tracks
    ])
    for axis in ("x", "y", "z"):
        aabb["min"][axis] -= float(margin)
        aabb["max"][axis] += float(margin)
    return aabb


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


def truth_geometry_matches(point, candidate_bbox, truth_points, truth_bboxes,
                           match_radius=MATCH_RADIUS):
    if point is None:
        return False
    for truth_bbox in truth_bboxes:
        if truth_bbox is None:
            continue
        if point_inside_aabb(point, truth_bbox):
            return True
        if candidate_bbox is not None and aabb_intersects(candidate_bbox, truth_bbox):
            return True
    return any(
        distance_between_points(point, truth_point) <= match_radius
        for truth_point in truth_points
    )


def classify_truth_bundle(track, link_tracks, policy=None):
    classification = classify_truth_track(track, policy=policy)
    if classification == "outlier":
        return classification

    effective_peak = track_peak_displacement(track)
    link_motion_detected = False
    for link_track in link_tracks:
        link_class = classify_truth_track(link_track, policy=policy)
        if link_class == "outlier":
            return "outlier"
        if link_class == "moving":
            link_motion_detected = True
        effective_peak = max(effective_peak, track_peak_displacement(link_track))

    if classification == "moving" or link_motion_detected:
        return "moving"
    threshold = (
        policy.translation_deadband_m if policy is not None else GT_MOVING_THRESHOLD
    )
    if effective_peak >= threshold:
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


def temporal_occupancy_fraction(samples):
    indexed = sorted(
        (
            (float(time_sec), index, bool(active))
            for index, (time_sec, active) in enumerate(samples)
            if time_sec is not None and math.isfinite(float(time_sec))
        ),
        key=lambda item: (item[0], item[1]),
    )
    if len(indexed) < 2:
        return ""
    duration_sec = indexed[-1][0] - indexed[0][0]
    if duration_sec <= 0.0:
        return ""
    active_duration_sec = 0.0
    for current, following in zip(indexed[:-1], indexed[1:]):
        interval_sec = max(0.0, following[0] - current[0])
        if current[2]:
            active_duration_sec += interval_sec
    return active_duration_sec / duration_sec


def _matching_truth_points(truth_tracks, link_tracks_by_model, time_sec):
    points = []
    for track in truth_tracks:
        link_tracks = link_tracks_by_model.get(track.object_name, [])
        points.extend(truth_points_at_time(track, link_tracks, time_sec))
    return points


def rigid_truth_displacement_at_time(track, link_tracks, time_sec):
    baseline_time = float(track.time_sec[0])
    baseline_position, baseline_rotation = _track_pose_at_time(track, baseline_time)
    current_position, current_rotation = _track_pose_at_time(track, time_sec)
    root_translation = float(np.linalg.norm(current_position - baseline_position))
    relative_rotation = baseline_rotation.T.dot(current_rotation)
    rotation_cosine = min(1.0, max(-1.0, (float(np.trace(relative_rotation)) - 1.0) * 0.5))
    root_rotation_deg = math.degrees(math.acos(rotation_cosine))

    surface_displacements = []
    for link_track in link_tracks:
        if not str(getattr(link_track, "link_name", "")).startswith("ground_truth_"):
            continue
        if not (
            _track_contains_time(link_track, baseline_time)
            and _track_contains_time(link_track, time_sec)
        ):
            continue
        baseline = _point_array(track_position_at_time(link_track, baseline_time))
        current = _point_array(track_position_at_time(link_track, time_sec))
        surface_displacements.append(float(np.linalg.norm(current - baseline)))

    return {
        "gt_root_translation_at_confirmation_m": root_translation,
        "gt_root_rotation_at_confirmation_deg": root_rotation_deg,
        "gt_surface_displacement_min_at_confirmation_m": (
            float(min(surface_displacements)) if surface_displacements else ""
        ),
        "gt_surface_displacement_median_at_confirmation_m": (
            float(np.median(surface_displacements)) if surface_displacements else ""
        ),
        "gt_surface_displacement_max_at_confirmation_m": (
            float(max(surface_displacements)) if surface_displacements else ""
        ),
    }


def _persistent_region_truth_matches(item, time_sec, moving_truth_tracks,
                                     moving_truth_metrics, link_tracks_by_model,
                                     alignment, truth_box_specs, match_radius,
                                     object_id_catalog=None):
    association_fields_present = any(
        key in item
        for key in (
            "object_id",
            "object_id_valid",
            "object_id_confidence",
            "object_id_ambiguous",
        )
    )
    if association_fields_present:
        if bool(item.get("object_id_ambiguous", False)):
            return [], "ambiguous"
        if not bool(item.get("object_id_valid", False)):
            return [], "unattributed"
        try:
            object_id = int(item.get("object_id", 0))
        except (TypeError, ValueError):
            return [], "invalid_direct"
        object_name = (object_id_catalog or {}).get(object_id)
        if not object_name:
            return [], "invalid_direct"
        truth_metrics = moving_truth_metrics.get(object_name)
        if truth_metrics is None:
            return [], "direct"
        if truth_metrics["start_time"] <= time_sec <= truth_metrics["end_time"]:
            return [object_name], "direct"
        return [], "direct"

    if alignment is None:
        return [], "spatial_unavailable"
    center_world = transform_point_world(item.get("center", {}), alignment)
    if center_world is None:
        return [], "spatial_unavailable"
    bbox_world = transform_aabb_world(
        item.get("bbox_min", {}), item.get("bbox_max", {}), alignment
    )
    matched_names = []
    for moving_track in moving_truth_tracks:
        truth_metrics = moving_truth_metrics[moving_track.object_name]
        if not (truth_metrics["start_time"] <= time_sec <= truth_metrics["end_time"]):
            continue
        link_tracks = link_tracks_by_model.get(moving_track.object_name, [])
        truth_points = truth_points_at_time(moving_track, link_tracks, time_sec)
        truth_bbox = build_truth_bbox_world(
            moving_track,
            (truth_box_specs or {}).get(moving_track.object_name),
            time_sec,
        )
        surface_bbox = build_surface_truth_bbox_world(link_tracks, time_sec)
        if truth_geometry_matches(
            center_world,
            bbox_world,
            truth_points,
            [truth_bbox, surface_bbox],
            match_radius,
        ):
            matched_names.append(moving_track.object_name)
    return matched_names, "spatial_fallback"


def _persistent_region_geometry_matches(item, time_sec, moving_truth_tracks,
                                        moving_truth_metrics, link_tracks_by_model,
                                        alignment, truth_box_specs, match_radius):
    """Match an emitted region to rigid truth without consulting object labels."""
    if alignment is None:
        return [], False
    center_world = transform_point_world(item.get("center", {}), alignment)
    if center_world is None:
        return [], False
    bbox_world = transform_aabb_world(
        item.get("bbox_min", {}), item.get("bbox_max", {}), alignment
    )
    matched_names = []
    for moving_track in moving_truth_tracks:
        truth_metrics = moving_truth_metrics[moving_track.object_name]
        if not (truth_metrics["start_time"] <= time_sec <= truth_metrics["end_time"]):
            continue
        link_tracks = link_tracks_by_model.get(moving_track.object_name, [])
        truth_points = truth_points_at_time(moving_track, link_tracks, time_sec)
        truth_bbox = build_truth_bbox_world(
            moving_track,
            (truth_box_specs or {}).get(moving_track.object_name),
            time_sec,
        )
        surface_bbox = build_surface_truth_bbox_world(link_tracks, time_sec)
        if truth_geometry_matches(
            center_world,
            bbox_world,
            truth_points,
            [truth_bbox, surface_bbox],
            match_radius,
        ):
            matched_names.append(moving_track.object_name)
    return matched_names, True


def _persistent_region_identity_assessment(item, time_sec, moving_truth_metrics,
                                           object_id_catalog, geometry_matches,
                                           geometry_available):
    association_fields_present = any(
        key in item
        for key in (
            "object_id", "object_id_valid", "object_id_ambiguous",
            "observed_object_id", "observed_object_id_valid",
            "observed_object_id_ambiguous", "object_association_state",
        )
    )
    if not association_fields_present:
        return "unavailable", None
    if bool(item.get("object_id_ambiguous", False)) or bool(
        item.get("observed_object_id_ambiguous", False)
    ):
        return "ambiguous", None
    try:
        association_state = int(item.get("object_association_state", 0))
    except (TypeError, ValueError):
        association_state = 0
    if association_state == 3:
        return "ambiguous", None
    if not bool(item.get("object_id_valid", False)):
        return "unattributed", None
    try:
        object_id = int(item.get("object_id", 0))
    except (TypeError, ValueError):
        return "invalid", None
    object_name = (object_id_catalog or {}).get(object_id)
    if not object_name:
        return "invalid", None
    if association_state == 2:
        return "mismatch", object_name
    if bool(item.get("observed_object_id_valid", False)):
        try:
            observed_id = int(item.get("observed_object_id", 0))
        except (TypeError, ValueError):
            return "mismatch", object_name
        if observed_id != object_id:
            return "mismatch", object_name
    truth_metrics = moving_truth_metrics.get(object_name)
    if truth_metrics is None or not (
        truth_metrics["start_time"] <= time_sec <= truth_metrics["end_time"]
    ):
        return "mismatch", object_name
    if geometry_available and object_name not in geometry_matches:
        return "mismatch", object_name
    return "correct", object_name


def build_persistent_risk_summary(persistent_records, region_records, truth_tracks, link_tracks_by_model,
                                  alignment, match_radius=MATCH_RADIUS, truth_box_specs=None,
                                  object_id_catalog=None, truth_policy=None,
                                  object_exposure=None):
    layer_status_value = layer_status(persistent_records)
    summary = {
        "layer_status": layer_status_value,
        "candidate_track_count": 0,
        "true_candidate_track_count": "",
        "false_candidate_track_count": "",
        "preliminary_alert_precision": "",
        "preliminary_alert_f1": "",
        "preliminary_detected_moving_object_count": 0,
        "preliminary_alert_recall": "",
        "candidate_region_observations": 0,
        "false_candidate_region_observations": "",
        "false_candidate_tracks_per_min": "",
        "false_candidate_region_observations_per_min": "",
        "preliminary_false_alarm_time_fraction": "",
        "preliminary_false_alarm_frame_fraction": "",
        "first_candidate_time": "",
        "confirmed_track_count": 0,
        "true_confirmed_track_count": "",
        "false_confirmed_track_count": "",
        "final_alert_precision": "",
        "final_alert_f1": "",
        "moving_object_count": 0,
        "detected_moving_object_count": 0,
        "final_alert_recall": "",
        "confirmed_region_observations": 0,
        "false_confirmed_region_observations": "",
        "false_confirmed_tracks_per_min": "",
        "false_confirmed_region_observations_per_min": "",
        "false_alarm_time_fraction": "",
        "false_alarm_frame_fraction": "",
        "truth_matching_status": "unavailable",
        "evaluation_frame_count": 0,
        "evaluation_duration_sec": 0.0,
        "first_confirmed_time": "",
        "max_confirmed_duration_sec": 0.0,
        "confirmed_coverage_hits": 0,
        "direct_object_association_observations": 0,
        "spatial_fallback_observations": 0,
        "unattributed_object_association_observations": 0,
        "ambiguous_object_association_observations": 0,
        "invalid_direct_object_association_observations": 0,
        "object_association_coverage": "",
        "object_association_mode": "unavailable",
        "geometric_truth_matching_status": "unavailable",
        "geometric_true_candidate_track_count": "",
        "geometric_false_candidate_track_count": "",
        "preliminary_geometric_precision": "",
        "preliminary_geometric_recall": "",
        "identity_true_candidate_track_count": 0,
        "identity_false_candidate_track_count": 0,
        "preliminary_identity_precision": "",
        "preliminary_identity_recall": "",
        "geometric_true_confirmed_track_count": "",
        "geometric_false_confirmed_track_count": "",
        "geometric_detected_moving_object_count": "",
        "final_geometric_precision": "",
        "final_geometric_recall": "",
        "identity_true_confirmed_track_count": 0,
        "identity_false_confirmed_track_count": 0,
        "identity_detected_moving_object_count": 0,
        "final_identity_precision": "",
        "final_identity_recall": "",
        "geometrically_correct_unattributed_candidate_observations": 0,
        "geometrically_correct_unattributed_confirmed_observations": 0,
        "association_error_candidate_observations": 0,
        "association_error_confirmed_observations": 0,
        "candidate_association_error_rate": "",
        "confirmed_association_error_rate": "",
        "association_error_rate": "",
        "true_false_candidate_observations": "",
        "true_false_confirmed_observations": "",
        "candidate_fragmentation_count": "",
        "confirmed_fragmentation_count": "",
        "candidate_cross_object_merge_count": "",
        "confirmed_cross_object_merge_count": "",
        "evaluable_moving_object_count": 0,
        "confirmed_presence_streak_count": 0,
        "confirmed_presence_streak_sec": 0.0,
        "significant_region_presence_streak_count": 0,
        "significant_region_presence_streak_sec": 0.0,
        "stability_judgment": "unavailable",
        "per_moving_object": [],
    }

    if persistent_records is None:
        return summary

    candidate_track_times = {}
    candidate_track_matches = {}
    confirmed_track_times = {}
    confirmed_track_matches = {}
    moving_truth_metrics = {}
    first_candidate_time = None
    first_confirmed_time = None
    candidate_observations = 0
    matched_candidate_observations = 0
    coverage_hits = 0
    confirmed_observations = 0
    matched_observations = 0
    matched_moving_objects = set()
    preliminary_moving_objects = set()
    first_candidate_by_object = {}
    first_confirmed_by_object = {}
    first_geometric_candidate_by_object = {}
    first_geometric_confirmed_by_object = {}
    first_identity_candidate_by_object = {}
    first_identity_confirmed_by_object = {}
    preliminary_false_alarm_frames = 0
    false_alarm_frames = 0
    valid_record_times = []
    false_candidate_states = []
    false_confirmation_states = []
    association_mode_counts = {
        "direct": 0,
        "spatial_fallback": 0,
        "spatial_unavailable": 0,
        "unattributed": 0,
        "ambiguous": 0,
        "invalid_direct": 0,
    }
    candidate_geometry_matches = {}
    confirmed_geometry_matches = {}
    candidate_identity_matches = {}
    confirmed_identity_matches = {}
    candidate_identity_claims = set()
    confirmed_identity_claims = set()
    geometric_unattributed_candidates = 0
    geometric_unattributed_confirmed = 0
    association_error_candidates = 0
    association_error_confirmed = 0
    true_false_candidate_observations = 0
    true_false_confirmed_observations = 0

    moving_truth_tracks = [
        track for track in truth_tracks
        if classify_truth_bundle(
            track,
            link_tracks_by_model.get(track.object_name, []),
            policy=truth_policy,
        ) == "moving"
    ]
    for track in moving_truth_tracks:
        moving_truth_metrics[track.object_name] = bundle_truth_metrics(
            track,
            link_tracks_by_model.get(track.object_name, []),
            policy=truth_policy,
        )
    summary["moving_object_count"] = len(moving_truth_tracks)
    evaluable_moving_names = {
        track.object_name for track in moving_truth_tracks
        if object_exposure is None or bool(
            next(
                (
                    _object_exposure_for_interval(
                        stats,
                        moving_truth_metrics[track.object_name]["start_time"],
                        moving_truth_metrics[track.object_name]["end_time"],
                    ).get("lidar_observed", False)
                    for object_id, stats in object_exposure.items()
                    if (object_id_catalog or {}).get(object_id) == track.object_name
                ),
                False,
            )
        )
    }
    summary["evaluable_moving_object_count"] = len(evaluable_moving_names)

    indexed_records = []
    for index, record in enumerate(persistent_records):
        timestamp = record_time_sec(record)
        if timestamp is not None:
            indexed_records.append((timestamp, index, record))
    indexed_records.sort(key=lambda item: (item[0], item[1]))

    for t, _, record in indexed_records:
        valid_record_times.append(t)
        all_regions = record.get("regions", [])
        if all_regions and first_candidate_time is None:
            first_candidate_time = t
        frame_has_false_candidate = False
        frame_has_false_confirmation = False
        for item in all_regions:
            candidate_observations += 1
            track_id = int(item.get("track_id", 0))
            track_key = (int(record.get("reference_epoch", 0)), track_id)
            candidate_span = candidate_track_times.setdefault(
                track_key, {"first": t, "last": t}
            )
            candidate_span["last"] = t
            candidate_span["first"] = min(candidate_span["first"], t)
            candidate_track_matches.setdefault(track_key, False)
            candidate_geometry_matches.setdefault(track_key, set())
            candidate_identity_matches.setdefault(track_key, set())
            geometry_names, geometry_available = _persistent_region_geometry_matches(
                item,
                t,
                moving_truth_tracks,
                moving_truth_metrics,
                link_tracks_by_model,
                alignment,
                truth_box_specs,
                match_radius,
            )
            identity_status, identity_name = _persistent_region_identity_assessment(
                item,
                t,
                moving_truth_metrics,
                object_id_catalog,
                geometry_names,
                geometry_available,
            )
            candidate_geometry_matches[track_key].update(geometry_names)
            for object_name in geometry_names:
                first_geometric_candidate_by_object.setdefault(object_name, t)
            if identity_name is not None:
                candidate_identity_claims.add(track_key)
            if identity_status == "correct" and identity_name is not None:
                candidate_identity_matches[track_key].add(identity_name)
                first_identity_candidate_by_object.setdefault(identity_name, t)
            if geometry_names and identity_status in {
                "unavailable", "unattributed", "ambiguous", "invalid"
            }:
                geometric_unattributed_candidates += 1
            if identity_status == "mismatch":
                association_error_candidates += 1
            if geometry_available and not geometry_names:
                true_false_candidate_observations += 1
            matched_names, association_mode = _persistent_region_truth_matches(
                item,
                t,
                moving_truth_tracks,
                moving_truth_metrics,
                link_tracks_by_model,
                alignment,
                truth_box_specs,
                match_radius,
                object_id_catalog,
            )
            association_mode_counts[association_mode] += 1
            matched = bool(matched_names)
            if matched:
                matched_candidate_observations += 1
                candidate_track_matches[track_key] = True
                preliminary_moving_objects.update(matched_names)
                for object_name in matched_names:
                    first_candidate_by_object.setdefault(object_name, t)
            else:
                frame_has_false_candidate = True

            if not persistent_region_is_confirmed(item):
                continue
            confirmed_observations += 1
            if first_confirmed_time is None:
                first_confirmed_time = t
            confirmed_span = confirmed_track_times.setdefault(
                track_key, {"first": t, "last": t}
            )
            confirmed_span["last"] = t
            confirmed_span["first"] = min(confirmed_span["first"], t)
            confirmed_track_matches.setdefault(track_key, False)
            confirmed_geometry_matches.setdefault(track_key, set()).update(
                geometry_names
            )
            for object_name in geometry_names:
                first_geometric_confirmed_by_object.setdefault(object_name, t)
            confirmed_identity_matches.setdefault(track_key, set())
            if identity_name is not None:
                confirmed_identity_claims.add(track_key)
            if identity_status == "correct" and identity_name is not None:
                confirmed_identity_matches[track_key].add(identity_name)
                first_identity_confirmed_by_object.setdefault(identity_name, t)
            if geometry_names and identity_status in {
                "unavailable", "unattributed", "ambiguous", "invalid"
            }:
                geometric_unattributed_confirmed += 1
            if identity_status == "mismatch":
                association_error_confirmed += 1
            if geometry_available and not geometry_names:
                true_false_confirmed_observations += 1
            if matched:
                coverage_hits += 1
                matched_observations += 1
                confirmed_track_matches[track_key] = True
                matched_moving_objects.update(matched_names)
                for object_name in matched_names:
                    first_confirmed_by_object.setdefault(object_name, t)
            else:
                frame_has_false_confirmation = True
        if frame_has_false_candidate:
            preliminary_false_alarm_frames += 1
        if frame_has_false_confirmation:
            false_alarm_frames += 1
        false_candidate_states.append((t, frame_has_false_candidate))
        false_confirmation_states.append((t, frame_has_false_confirmation))

    if candidate_track_times:
        summary["candidate_track_count"] = len(candidate_track_times)
        summary["first_candidate_time"] = (
            first_candidate_time if first_candidate_time is not None else ""
        )
    summary["candidate_region_observations"] = candidate_observations
    direct_associations = association_mode_counts["direct"]
    summary["direct_object_association_observations"] = direct_associations
    summary["spatial_fallback_observations"] = association_mode_counts[
        "spatial_fallback"
    ]
    summary["unattributed_object_association_observations"] = (
        association_mode_counts["unattributed"]
        + association_mode_counts["spatial_unavailable"]
    )
    summary["ambiguous_object_association_observations"] = association_mode_counts[
        "ambiguous"
    ]
    summary["invalid_direct_object_association_observations"] = (
        association_mode_counts["invalid_direct"]
    )
    summary["object_association_coverage"] = _safe_metric_ratio(
        direct_associations, candidate_observations
    )
    if candidate_observations:
        if direct_associations == candidate_observations:
            summary["object_association_mode"] = "direct"
        elif association_mode_counts["spatial_fallback"] == candidate_observations:
            summary["object_association_mode"] = "legacy_spatial_fallback"
        else:
            summary["object_association_mode"] = "mixed_or_incomplete"
    if confirmed_track_times:
        summary["confirmed_track_count"] = len(confirmed_track_times)
        summary["first_confirmed_time"] = first_confirmed_time if first_confirmed_time is not None else ""
        summary["max_confirmed_duration_sec"] = max(
            float(span["last"] - span["first"]) for span in confirmed_track_times.values()
        )
        summary["confirmed_coverage_hits"] = coverage_hits
    summary["confirmed_region_observations"] = confirmed_observations

    duration_sec = (
        max(valid_record_times) - min(valid_record_times)
        if len(valid_record_times) >= 2 else 0.0
    )
    summary["evaluation_frame_count"] = len(valid_record_times)
    summary["evaluation_duration_sec"] = duration_sec
    truth_matching_available = association_mode_counts["spatial_unavailable"] == 0
    summary["truth_matching_status"] = (
        "available" if truth_matching_available else "unavailable_missing_frame_alignment"
    )
    geometry_available = alignment is not None
    summary["geometric_truth_matching_status"] = (
        "available" if geometry_available else "unavailable_missing_frame_alignment"
    )

    identity_candidate_true_tracks = sum(
        bool(matches) for matches in candidate_identity_matches.values()
    )
    identity_candidate_false_tracks = sum(
        track_key in candidate_identity_claims and not matches
        for track_key, matches in candidate_identity_matches.items()
    )
    identity_confirmed_true_tracks = sum(
        bool(matches) for matches in confirmed_identity_matches.values()
    )
    identity_confirmed_false_tracks = sum(
        track_key in confirmed_identity_claims and not matches
        for track_key, matches in confirmed_identity_matches.items()
    )
    candidate_identity_objects = set().union(
        *candidate_identity_matches.values()
    ) if candidate_identity_matches else set()
    confirmed_identity_objects = set().union(
        *confirmed_identity_matches.values()
    ) if confirmed_identity_matches else set()
    summary["identity_true_candidate_track_count"] = identity_candidate_true_tracks
    summary["identity_false_candidate_track_count"] = identity_candidate_false_tracks
    summary["preliminary_identity_precision"] = _safe_metric_ratio(
        identity_candidate_true_tracks,
        identity_candidate_true_tracks + identity_candidate_false_tracks,
    )
    summary["preliminary_identity_recall"] = _safe_metric_ratio(
        len(candidate_identity_objects & evaluable_moving_names),
        len(evaluable_moving_names),
    )
    summary["identity_true_confirmed_track_count"] = identity_confirmed_true_tracks
    summary["identity_false_confirmed_track_count"] = identity_confirmed_false_tracks
    summary["final_identity_precision"] = _safe_metric_ratio(
        identity_confirmed_true_tracks,
        identity_confirmed_true_tracks + identity_confirmed_false_tracks,
    )
    summary["final_identity_recall"] = _safe_metric_ratio(
        len(confirmed_identity_objects & evaluable_moving_names),
        len(evaluable_moving_names),
    )
    summary["identity_detected_moving_object_count"] = len(
        confirmed_identity_objects & evaluable_moving_names
    )
    summary["geometrically_correct_unattributed_candidate_observations"] = (
        geometric_unattributed_candidates
    )
    summary["geometrically_correct_unattributed_confirmed_observations"] = (
        geometric_unattributed_confirmed
    )
    summary["association_error_candidate_observations"] = association_error_candidates
    summary["association_error_confirmed_observations"] = association_error_confirmed
    summary["candidate_association_error_rate"] = _safe_metric_ratio(
        association_error_candidates, candidate_observations
    )
    summary["confirmed_association_error_rate"] = _safe_metric_ratio(
        association_error_confirmed, confirmed_observations
    )
    summary["association_error_rate"] = summary[
        "confirmed_association_error_rate"
    ]

    if geometry_available:
        geometric_candidate_true_tracks = sum(
            bool(matches) for matches in candidate_geometry_matches.values()
        )
        geometric_candidate_false_tracks = (
            len(candidate_geometry_matches) - geometric_candidate_true_tracks
        )
        geometric_confirmed_true_tracks = sum(
            bool(matches) for matches in confirmed_geometry_matches.values()
        )
        geometric_confirmed_false_tracks = (
            len(confirmed_geometry_matches) - geometric_confirmed_true_tracks
        )
        candidate_geometry_objects = set().union(
            *candidate_geometry_matches.values()
        ) if candidate_geometry_matches else set()
        confirmed_geometry_objects = set().union(
            *confirmed_geometry_matches.values()
        ) if confirmed_geometry_matches else set()
        summary["geometric_true_candidate_track_count"] = geometric_candidate_true_tracks
        summary["geometric_false_candidate_track_count"] = geometric_candidate_false_tracks
        summary["preliminary_geometric_precision"] = _safe_metric_ratio(
            geometric_candidate_true_tracks,
            geometric_candidate_true_tracks + geometric_candidate_false_tracks,
        )
        summary["preliminary_geometric_recall"] = _safe_metric_ratio(
            len(candidate_geometry_objects & evaluable_moving_names),
            len(evaluable_moving_names),
        )
        summary["geometric_true_confirmed_track_count"] = geometric_confirmed_true_tracks
        summary["geometric_false_confirmed_track_count"] = geometric_confirmed_false_tracks
        summary["final_geometric_precision"] = _safe_metric_ratio(
            geometric_confirmed_true_tracks,
            geometric_confirmed_true_tracks + geometric_confirmed_false_tracks,
        )
        summary["final_geometric_recall"] = _safe_metric_ratio(
            len(confirmed_geometry_objects & evaluable_moving_names),
            len(evaluable_moving_names),
        )
        summary["geometric_detected_moving_object_count"] = len(
            confirmed_geometry_objects & evaluable_moving_names
        )
        summary["true_false_candidate_observations"] = true_false_candidate_observations
        summary["true_false_confirmed_observations"] = true_false_confirmed_observations

        def fragmentation_count(track_matches):
            counts = {}
            for matches in track_matches.values():
                for object_name in matches:
                    counts[object_name] = counts.get(object_name, 0) + 1
            return sum(max(0, count - 1) for count in counts.values())

        summary["candidate_fragmentation_count"] = fragmentation_count(
            candidate_geometry_matches
        )
        summary["confirmed_fragmentation_count"] = fragmentation_count(
            confirmed_geometry_matches
        )
        summary["candidate_cross_object_merge_count"] = sum(
            len(matches) > 1 for matches in candidate_geometry_matches.values()
        )
        summary["confirmed_cross_object_merge_count"] = sum(
            len(matches) > 1 for matches in confirmed_geometry_matches.values()
        )

    if truth_matching_available:
        true_candidates = sum(bool(value) for value in candidate_track_matches.values())
        false_candidates = len(candidate_track_matches) - true_candidates
        false_candidate_observations = (
            candidate_observations - matched_candidate_observations
        )
        summary["true_candidate_track_count"] = true_candidates
        summary["false_candidate_track_count"] = false_candidates
        summary["preliminary_alert_precision"] = _safe_metric_ratio(
            true_candidates, true_candidates + false_candidates
        )
        summary["preliminary_detected_moving_object_count"] = len(
            preliminary_moving_objects
        )
        summary["preliminary_alert_recall"] = _safe_metric_ratio(
            len(preliminary_moving_objects), len(moving_truth_tracks)
        )
        summary["preliminary_alert_f1"] = _safe_f1(
            summary["preliminary_alert_precision"],
            summary["preliminary_alert_recall"],
        )
        summary["false_candidate_region_observations"] = false_candidate_observations
        true_tracks = sum(bool(value) for value in confirmed_track_matches.values())
        false_tracks = len(confirmed_track_matches) - true_tracks
        false_observations = confirmed_observations - matched_observations
        summary["true_confirmed_track_count"] = true_tracks
        summary["false_confirmed_track_count"] = false_tracks
        summary["final_alert_precision"] = _safe_metric_ratio(
            true_tracks, true_tracks + false_tracks
        )
        summary["false_confirmed_region_observations"] = false_observations
        summary["detected_moving_object_count"] = len(matched_moving_objects)
        summary["final_alert_recall"] = _safe_metric_ratio(
            len(matched_moving_objects), len(moving_truth_tracks)
        )
        summary["final_alert_f1"] = _safe_f1(
            summary["final_alert_precision"], summary["final_alert_recall"]
        )
        summary["false_alarm_frame_fraction"] = _safe_metric_ratio(
            false_alarm_frames, len(valid_record_times)
        )
        summary["preliminary_false_alarm_frame_fraction"] = _safe_metric_ratio(
            preliminary_false_alarm_frames, len(valid_record_times)
        )
        summary["false_alarm_time_fraction"] = temporal_occupancy_fraction(
            false_confirmation_states
        )
        summary["preliminary_false_alarm_time_fraction"] = (
            temporal_occupancy_fraction(false_candidate_states)
        )
        if duration_sec > 0.0:
            duration_min = duration_sec / 60.0
            summary["false_candidate_tracks_per_min"] = false_candidates / duration_min
            summary["false_candidate_region_observations_per_min"] = (
                false_candidate_observations / duration_min
            )
            summary["false_confirmed_tracks_per_min"] = false_tracks / duration_min
            summary["false_confirmed_region_observations_per_min"] = (
                false_observations / duration_min
            )

    per_moving_object = []
    for track in moving_truth_tracks:
        object_name = track.object_name
        truth_metrics = moving_truth_metrics[object_name]
        object_id = next(
            (
                object_id for object_id, catalog_name in (object_id_catalog or {}).items()
                if catalog_name == object_name
            ),
            None,
        )
        raw_exposure = (
            object_exposure.get(object_id, {})
            if object_exposure is not None and object_id is not None else {}
        )
        interval_exposure = (
            _object_exposure_for_interval(
                raw_exposure, truth_metrics["start_time"], truth_metrics["end_time"]
            )
            if object_exposure is not None else {}
        )
        lidar_observed = (
            bool(interval_exposure.get("lidar_observed", False))
            if object_exposure is not None else True
        )
        candidate_time = first_candidate_by_object.get(object_name)
        confirmed_time = first_confirmed_by_object.get(object_name)
        row = {
            "object_name": object_name,
            "object_id": object_id if object_id is not None else "",
            "evaluation_status": (
                "EVALUABLE" if lidar_observed else "LIDAR_UNOBSERVED"
            ),
            "lidar_observed": lidar_observed,
            "hit_point_count": int(interval_exposure.get("hit_point_count", 0)),
            "visible_frame_count": int(
                interval_exposure.get("visible_frame_count", 0)
            ),
            "exposure_frame_count": int(
                interval_exposure.get("exposure_frame_count", 0)
            ),
            "lidar_visibility_rate": interval_exposure.get(
                "lidar_visibility_rate", ""
            ),
            "gt_start_time": truth_metrics["start_time"],
            "gt_end_time": truth_metrics["end_time"],
            "preliminary_detected": candidate_time is not None,
            "confirmed_detected": confirmed_time is not None,
            "geometric_preliminary_detected": (
                object_name in first_geometric_candidate_by_object
            ),
            "geometric_confirmed_detected": (
                object_name in first_geometric_confirmed_by_object
            ),
            "identity_preliminary_detected": (
                object_name in first_identity_candidate_by_object
            ),
            "identity_confirmed_detected": (
                object_name in first_identity_confirmed_by_object
            ),
            "first_candidate_time": candidate_time if candidate_time is not None else "",
            "first_confirmed_time": confirmed_time if confirmed_time is not None else "",
            "candidate_delay_sec": (
                candidate_time - truth_metrics["start_time"]
                if candidate_time is not None else ""
            ),
            "confirmation_delay_sec": (
                confirmed_time - truth_metrics["start_time"]
                if confirmed_time is not None else ""
            ),
            "candidate_to_confirmation_sec": (
                confirmed_time - candidate_time
                if candidate_time is not None and confirmed_time is not None else ""
            ),
            "gt_peak_displacement_m": truth_metrics["peak_displacement"],
            "gt_peak_rotation_deg": truth_metrics["peak_rotation_deg"],
            "gt_peak_linear_speed_mps": truth_metrics["peak_linear_speed_mps"],
            "gt_peak_angular_speed_degps": truth_metrics[
                "peak_angular_speed_degps"
            ],
            "gt_peak_linear_acceleration_mps2": truth_metrics[
                "peak_linear_acceleration_mps2"
            ],
            "gt_peak_angular_acceleration_degps2": truth_metrics[
                "peak_angular_acceleration_degps2"
            ],
            "gt_surface_peak_linear_speed_mps": truth_metrics[
                "surface_peak_linear_speed_mps"
            ],
            "gt_surface_peak_angular_speed_degps": truth_metrics[
                "surface_peak_angular_speed_degps"
            ],
            "gt_surface_peak_linear_acceleration_mps2": truth_metrics[
                "surface_peak_linear_acceleration_mps2"
            ],
            "gt_surface_peak_angular_acceleration_degps2": truth_metrics[
                "surface_peak_angular_acceleration_degps2"
            ],
            "gt_root_translation_at_candidate_m": "",
            "gt_root_rotation_at_candidate_deg": "",
            "gt_surface_displacement_min_at_candidate_m": "",
            "gt_surface_displacement_median_at_candidate_m": "",
            "gt_surface_displacement_max_at_candidate_m": "",
            "gt_root_translation_at_confirmation_m": "",
            "gt_root_rotation_at_confirmation_deg": "",
            "gt_surface_displacement_min_at_confirmation_m": "",
            "gt_surface_displacement_median_at_confirmation_m": "",
            "gt_surface_displacement_max_at_confirmation_m": "",
        }
        if candidate_time is not None:
            candidate_values = rigid_truth_displacement_at_time(
                track,
                link_tracks_by_model.get(object_name, []),
                candidate_time,
            )
            row.update(
                {
                    key.replace("_at_confirmation_", "_at_candidate_"): value
                    for key, value in candidate_values.items()
                }
            )
        if confirmed_time is not None:
            row.update(rigid_truth_displacement_at_time(
                track,
                link_tracks_by_model.get(object_name, []),
                confirmed_time,
            ))
        per_moving_object.append(row)
    summary["per_moving_object"] = per_moving_object

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
                          truth_box_specs=None, match_radius=MATCH_RADIUS,
                          truth_policy=None):
    metrics = bundle_truth_metrics(track, link_tracks, policy=truth_policy)
    truth_box_spec = (truth_box_specs or {}).get(track.object_name)

    evidence_status = layer_status(evidence_records)
    region_status = layer_status(region_records)
    motion_status = layer_status(motion_records)

    summary = {
        "object_name": track.object_name,
        "classification": classify_truth_bundle(track, link_tracks, policy=truth_policy),
        "gt_net_displacement": metrics["net_displacement"],
        "gt_duration_sec": metrics["duration_sec"],
        "gt_start_time": metrics["start_time"],
        "gt_end_time": metrics["end_time"],
        "gt_peak_displacement": metrics["peak_displacement"],
        "gt_peak_displacement_time": metrics["peak_displacement_time"],
        "gt_peak_rotation_deg": metrics["peak_rotation_deg"],
        "gt_peak_linear_speed_mps": metrics["peak_linear_speed_mps"],
        "gt_peak_angular_speed_degps": metrics["peak_angular_speed_degps"],
        "gt_peak_linear_acceleration_mps2": metrics[
            "peak_linear_acceleration_mps2"
        ],
        "gt_peak_angular_acceleration_degps2": metrics[
            "peak_angular_acceleration_degps2"
        ],
        "gt_surface_peak_linear_speed_mps": metrics[
            "surface_peak_linear_speed_mps"
        ],
        "gt_surface_peak_angular_speed_degps": metrics[
            "surface_peak_angular_speed_degps"
        ],
        "gt_surface_peak_linear_acceleration_mps2": metrics[
            "surface_peak_linear_acceleration_mps2"
        ],
        "gt_surface_peak_angular_acceleration_degps2": metrics[
            "surface_peak_angular_acceleration_degps2"
        ],
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
            surface_bbox = build_surface_truth_bbox_world(link_tracks, t)
            for item in record.get("evidences", []):
                if not item.get("active", False):
                    continue
                point_world = transform_point_world(item.get("position", {}), alignment)
                if point_world is None:
                    continue
                if truth_geometry_matches(
                    point_world,
                    None,
                    gt_points,
                    [truth_bbox, surface_bbox],
                    match_radius,
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
            surface_bbox = build_surface_truth_bbox_world(link_tracks, t)
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
                bbox_match = truth_geometry_matches(
                    point_world,
                    bbox_world,
                    [],
                    [truth_bbox, surface_bbox],
                    match_radius,
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
            surface_bbox = build_surface_truth_bbox_world(link_tracks, t)
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
                bbox_match = truth_geometry_matches(
                    point_world,
                    bbox_world,
                    [],
                    [truth_bbox, surface_bbox],
                    match_radius,
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


PHASE_METRICS_HEADER = [
    "stream",
    "phase",
    "phase_start_time_sec",
    "phase_end_time_sec",
    "phase_duration_sec",
    "record_count",
]

FAILURE_MODES_HEADER = [
    "source",
    "object_name",
    "object_id",
    "anchor_type",
    "classification",
    "status",
    "reason",
    "sample_count",
]


def build_phase_metric_rows(scope, stream_audits):
    if scope is None:
        return []
    intervals = {row["phase"]: row for row in scope.phase_intervals()}
    rows = []
    for stream_name, audit in sorted(stream_audits.items()):
        counts = audit.get("phase_record_counts", {})
        for phase in PHASE_NAMES:
            interval = intervals[phase]
            rows.append(
                {
                    "stream": stream_name,
                    "phase": phase,
                    "phase_start_time_sec": interval["start_time_sec"],
                    "phase_end_time_sec": interval["end_time_sec"],
                    "phase_duration_sec": interval["duration_sec"],
                    "record_count": int(counts.get(phase, 0)),
                }
            )
    return rows


def build_failure_mode_rows(
    object_anchor_rows, anchor_vector_rows, persistent_object_rows
):
    """Build auditable raw status counts without defining final paper formulas."""

    rows = []
    for row in object_anchor_rows:
        status = str(row.get("evaluation_status", ""))
        outcome = str(row.get("outcome", ""))
        if outcome in {"TP", "TN"}:
            continue
        rows.append(
            {
                "source": "object_anchor_type",
                "object_name": row.get("object_name", ""),
                "object_id": row.get("object_id", ""),
                "anchor_type": row.get("anchor_type", ""),
                "classification": row.get("classification", ""),
                "status": status,
                "reason": outcome,
                "sample_count": row.get("anchor_sample_count", 0),
            }
        )

    invalid_vectors = {}
    for row in anchor_vector_rows:
        if bool(row.get("valid", False)):
            continue
        key = (
            str(row.get("object_name", "")),
            row.get("object_id", ""),
            str(row.get("anchor_type", "")),
            str(row.get("classification", "")),
            str(row.get("invalid_reason", "unspecified")),
        )
        invalid_vectors[key] = invalid_vectors.get(key, 0) + 1
    for key, count in sorted(invalid_vectors.items()):
        object_name, object_id, anchor_type, classification, reason = key
        rows.append(
            {
                "source": "anchor_vector",
                "object_name": object_name,
                "object_id": object_id,
                "anchor_type": anchor_type,
                "classification": classification,
                "status": "INVALID_VECTOR_OBSERVATION",
                "reason": reason,
                "sample_count": count,
            }
        )

    for row in persistent_object_rows:
        status = str(row.get("evaluation_status", ""))
        confirmed = bool(row.get("confirmed_detected", False))
        if confirmed and status not in {"LIDAR_UNOBSERVED", "TRUTH_OUTLIER"}:
            continue
        rows.append(
            {
                "source": "persistent_object_alert",
                "object_name": row.get("object_name", ""),
                "object_id": row.get("object_id", ""),
                "anchor_type": "ALL",
                "classification": "moving",
                "status": status or "NOT_CONFIRMED",
                "reason": "confirmed" if confirmed else "not_confirmed",
                "sample_count": 1,
            }
        )
    return rows


def load_protocol_controller_configuration(scope):
    if scope is None:
        return None
    with scope.protocol_path.open() as handle:
        protocol = json.load(handle)
    actual = protocol.get("actual", {}) if isinstance(protocol, dict) else {}
    configuration = actual.get("controller_configuration", {})
    return configuration if isinstance(configuration, dict) else {}


def load_protocol_timing_audit(scope):
    if scope is None:
        return {"status": "UNAVAILABLE_LEGACY_PROTOCOL"}
    with scope.protocol_path.open() as handle:
        protocol = json.load(handle)
    actual = protocol.get("actual", {}) if isinstance(protocol, dict) else {}
    audit = actual.get("timing_audit") if isinstance(actual, dict) else None
    if not isinstance(audit, dict):
        return {"status": "UNAVAILABLE_LEGACY_PROTOCOL"}
    return audit


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


def compact_active_evidence_records(records):
    """Retain only fields consumed by object matching and timeline plots."""

    if records is None:
        return None
    compact = []
    for record in records:
        compact_record = {
            "header": record.get("header", {}),
            "recorded_at": record.get("recorded_at"),
            "evidences": [
                {
                    "active": True,
                    "position": item.get("position", {}),
                    "risk_score": item.get("risk_score", 0.0),
                }
                for item in record.get("evidences", [])
                if bool(item.get("active", False))
            ],
        }
        if "reference_epoch" in record:
            compact_record["reference_epoch"] = record.get("reference_epoch")
        compact.append(compact_record)
    return compact


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
    lines.append(f"- candidate_track_count: `{persistent_summary['candidate_track_count']}`")
    lines.append(
        f"- preliminary_alert_precision: "
        f"`{persistent_summary['preliminary_alert_precision']}`"
    )
    lines.append(
        f"- preliminary_alert_f1: `{persistent_summary['preliminary_alert_f1']}`"
    )
    lines.append(
        f"- false_candidate_tracks_per_min: "
        f"`{persistent_summary['false_candidate_tracks_per_min']}`"
    )
    lines.append(f"- confirmed_track_count: `{persistent_summary['confirmed_track_count']}`")
    lines.append(
        f"- true_confirmed_track_count: `{persistent_summary['true_confirmed_track_count']}`"
    )
    lines.append(
        f"- false_confirmed_track_count: `{persistent_summary['false_confirmed_track_count']}`"
    )
    lines.append(
        f"- final_alert_precision: `{persistent_summary['final_alert_precision']}`"
    )
    lines.append(
        f"- final_alert_recall: `{persistent_summary['final_alert_recall']}`"
    )
    lines.append(f"- final_alert_f1: `{persistent_summary['final_alert_f1']}`")
    lines.append(
        f"- final_geometric_precision: "
        f"`{persistent_summary['final_geometric_precision']}`"
    )
    lines.append(
        f"- final_geometric_recall: `{persistent_summary['final_geometric_recall']}`"
    )
    lines.append(
        f"- final_identity_precision: `{persistent_summary['final_identity_precision']}`"
    )
    lines.append(
        f"- final_identity_recall: `{persistent_summary['final_identity_recall']}`"
    )
    lines.append(
        f"- association_error_rate: `{persistent_summary['association_error_rate']}`"
    )
    lines.append(
        f"- confirmed_fragmentation_count: "
        f"`{persistent_summary['confirmed_fragmentation_count']}`"
    )
    lines.append(
        f"- confirmed_cross_object_merge_count: "
        f"`{persistent_summary['confirmed_cross_object_merge_count']}`"
    )
    lines.append(
        f"- false_confirmed_tracks_per_min: "
        f"`{persistent_summary['false_confirmed_tracks_per_min']}`"
    )
    lines.append(
        f"- false_confirmed_region_observations_per_min: "
        f"`{persistent_summary['false_confirmed_region_observations_per_min']}`"
    )
    lines.append(
        f"- false_alarm_time_fraction: `{persistent_summary['false_alarm_time_fraction']}`"
    )
    lines.append(f"- first_confirmed_time: `{persistent_summary['first_confirmed_time']}`")
    lines.append(
        f"- max_confirmed_duration_sec: `{persistent_summary['max_confirmed_duration_sec']}`"
    )
    lines.append(
        f"- confirmed_coverage_hits: `{persistent_summary['confirmed_coverage_hits']}`"
    )
    for row in persistent_summary["per_moving_object"]:
        lines.append(
            f"- `{row['object_name']}`: preliminary=`{row['preliminary_detected']}`, "
            f"confirmed=`{row['confirmed_detected']}`, "
            f"candidate_delay_sec=`{row['candidate_delay_sec']}`, "
            f"confirmation_delay_sec=`{row['confirmation_delay_sec']}`, "
            f"gt_translation_at_confirmation_m="
            f"`{row['gt_root_translation_at_confirmation_m']}`, "
            f"gt_rotation_at_confirmation_deg="
            f"`{row['gt_root_rotation_at_confirmation_deg']}`"
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


def analyze_sim_run(
    run_dir,
    output_dir=None,
    world_file=None,
    validation_policy="strict",
    max_anchor_processing_drop_fraction=None,
    validation_report_path=None,
    anchor_record_consumer=None,
    validation_stream_consumers=None,
    validated_risk_evidence_records=None,
):
    run_dir = pathlib.Path(run_dir).resolve()
    validate_schema_v2_run(
        run_dir,
        validation_policy=validation_policy,
        max_anchor_processing_drop_fraction=(
            max_anchor_processing_drop_fraction
        ),
        report_path=validation_report_path,
        stream_consumers=validation_stream_consumers,
    )
    formal_scope = load_formal_analysis_scope(run_dir, required=False)
    truth_policy = load_truth_motion_policy(run_dir)
    truth_objects_dir = run_dir / "truth" / "objects"
    if not truth_objects_dir.is_dir():
        raise FileNotFoundError(f"Truth object directory missing: {truth_objects_dir}")

    output_dir = (
        pathlib.Path(output_dir)
        if output_dir
        else run_dir / "analysis" / "formal"
        if formal_scope is not None
        else run_dir / "analysis"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    evidence_records_raw = validated_risk_evidence_records
    if evidence_records_raw is None:
        evidence_records_raw = load_algorithm_stream_optional(
            run_dir,
            "risk_evidence",
            required=False,
        )
    region_records_raw = load_jsonl_optional(
        run_dir / "algorithm" / "risk_regions.jsonl"
    )
    persistent_region_records_raw = load_jsonl_optional(
        run_dir / "algorithm" / "persistent_risk_regions.jsonl"
    )
    motion_records_raw = load_jsonl_optional(
        run_dir / "algorithm" / "structure_motions.jsonl"
    )
    object_observation_records_raw = load_jsonl_optional(
        run_dir / "algorithm" / "object_observation_stats.jsonl"
    )
    anchor_records_raw = load_anchor_state_records(run_dir)
    anchor_catalog_records = load_jsonl_optional(
        run_dir / "algorithm" / "anchor_catalog.jsonl"
    )
    evidence_records = scoped_records(
        evidence_records_raw, formal_scope, "risk_evidence"
    )
    region_records = scoped_records(
        region_records_raw, formal_scope, "risk_regions"
    )
    persistent_region_records = scoped_records(
        persistent_region_records_raw,
        formal_scope,
        "persistent_risk_regions",
    )
    motion_records = scoped_records(
        motion_records_raw, formal_scope, "structure_motions"
    )
    object_observation_records = scoped_records(
        object_observation_records_raw,
        formal_scope,
        "object_observation_stats",
        predicate=lambda record: int(record.get("phase", 0)) == 1,
    )
    anchor_records = scoped_records(
        anchor_records_raw, formal_scope, "anchor_observations"
    )
    scoped_streams = {
        "anchor_observations": anchor_records,
        "risk_evidence": evidence_records,
        "risk_regions": region_records,
        "persistent_risk_regions": persistent_region_records,
        "structure_motions": motion_records,
        "object_observation_stats": object_observation_records,
    }
    alignment = load_alignment(run_dir)
    alarm_operating_point = load_alarm_operating_point(run_dir)
    truth_box_specs = load_truth_box_specs(world_file)
    object_id_catalog = merge_object_id_catalogs(
        load_object_id_catalog(world_file),
        load_recorded_object_id_catalog(run_dir),
    )
    object_exposure = build_object_hit_exposure(
        object_observation_records, object_id_catalog
    )

    truth_tracks_raw = load_truth_tracks(truth_objects_dir)
    truth_tracks, truth_track_audit = crop_tracks_to_scope(
        truth_tracks_raw, formal_scope
    )
    link_tracks_raw = load_link_tracks(run_dir / "truth" / "links")
    link_tracks, link_track_audit = crop_tracks_to_scope(
        link_tracks_raw, formal_scope
    )
    compact_surface_tracks = load_surface_truth_link_tracks(
        run_dir, truth_tracks, link_tracks
    )
    existing_scoped_names = {track.scoped_link_name for track in link_tracks}
    link_tracks.extend(
        track
        for track in compact_surface_tracks
        if track.scoped_link_name not in existing_scoped_names
    )
    link_tracks_by_model = {}
    for link_track in link_tracks:
        link_tracks_by_model.setdefault(link_track.model_name, []).append(link_track)
    anchor_inventory_stats = _initialize_anchor_type_inventory(
        anchor_catalog_records,
        formal_epoch=(
            formal_scope.formal_epoch if formal_scope is not None else None
        ),
    )
    significant_anchor_records = []
    object_anchor_rows, anchor_type_rows, anchor_record_count = build_anchor_detection_metrics(
        anchor_records,
        truth_tracks,
        link_tracks_by_model,
        object_id_catalog,
        truth_policy=truth_policy,
        object_exposure=object_exposure,
        inventory_stats=anchor_inventory_stats,
        significant_records=significant_anchor_records,
        record_consumer=anchor_record_consumer,
        enforce_associations=True,
        return_record_count=True,
    )
    anchor_type_inventory_rows = _finalize_anchor_type_inventory(
        anchor_inventory_stats
    )
    anchor_vector_rows, anchor_vector_type_rows = build_anchor_vector_metrics(
        significant_anchor_records,
        truth_tracks,
        link_tracks_by_model,
        object_id_catalog,
        alignment,
        truth_policy=truth_policy,
    )
    evaluation_evidence_records = compact_active_evidence_records(
        evidence_records
    )
    summary_rows = []
    outlier_rows = []
    for track in truth_tracks:
        summary = evaluate_truth_object(
            track,
            link_tracks_by_model.get(track.object_name, []),
            alignment=alignment,
            evidence_records=evaluation_evidence_records,
            region_records=region_records,
            motion_records=motion_records,
            truth_box_specs=truth_box_specs,
            truth_policy=truth_policy,
        )
        summary_rows.append(summary)
        if summary["classification"] == "outlier":
            metrics = bundle_truth_metrics(
                track,
                link_tracks_by_model.get(track.object_name, []),
                policy=truth_policy,
            )
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
        truth_box_specs=truth_box_specs,
        object_id_catalog=object_id_catalog,
        truth_policy=truth_policy,
        object_exposure=object_exposure,
    )

    summary_csv = output_dir / "summary.csv"
    outlier_csv = output_dir / "outlier_objects.csv"
    report_md = output_dir / "report.md"
    gt_motion_timeline_png = output_dir / "gt_motion_timeline.png"
    detection_stage_timeline_png = output_dir / "detection_stage_timeline.png"
    spatial_overlay_png = output_dir / "spatial_overlay.png"
    object_anchor_metrics_csv = output_dir / "object_anchor_metrics.csv"
    anchor_type_metrics_csv = output_dir / "anchor_type_metrics.csv"
    anchor_vector_metrics_csv = output_dir / "anchor_vector_metrics.csv"
    anchor_vector_type_metrics_csv = output_dir / "anchor_vector_type_metrics.csv"
    persistent_object_metrics_csv = output_dir / "persistent_object_metrics.csv"
    alert_metrics_json = output_dir / "alert_metrics.json"
    object_metrics_csv = output_dir / "object_metrics.csv"
    object_anchor_type_metrics_csv = output_dir / "object_anchor_type_metrics.csv"
    analysis_scope_json = output_dir / "analysis_scope.json"
    run_metrics_json = output_dir / "run_metrics.json"
    phase_metrics_csv = output_dir / "phase_metrics.csv"
    failure_modes_csv = output_dir / "failure_modes.csv"
    anchor_type_inventory_csv = output_dir / "anchor_type_inventory.csv"

    write_csv(summary_csv, SUMMARY_HEADER, moving_summary_rows)
    write_csv(outlier_csv, OUTLIER_HEADER, outlier_rows)
    write_csv(
        object_anchor_metrics_csv,
        OBJECT_ANCHOR_METRICS_HEADER,
        object_anchor_rows,
    )
    write_csv(
        anchor_type_metrics_csv,
        ANCHOR_TYPE_METRICS_HEADER,
        anchor_type_rows,
    )
    write_csv(
        anchor_vector_metrics_csv,
        ANCHOR_VECTOR_METRICS_HEADER,
        anchor_vector_rows,
    )
    write_csv(
        anchor_vector_type_metrics_csv,
        ANCHOR_VECTOR_TYPE_METRICS_HEADER,
        anchor_vector_type_rows,
    )
    write_csv(
        persistent_object_metrics_csv,
        PERSISTENT_OBJECT_METRICS_HEADER,
        persistent_summary["per_moving_object"],
    )
    write_csv(
        object_metrics_csv,
        PERSISTENT_OBJECT_METRICS_HEADER,
        persistent_summary["per_moving_object"],
    )
    write_csv(
        object_anchor_type_metrics_csv,
        OBJECT_ANCHOR_METRICS_HEADER,
        object_anchor_rows,
    )
    write_csv(
        anchor_type_inventory_csv,
        ANCHOR_TYPE_INVENTORY_HEADER,
        anchor_type_inventory_rows,
    )
    stream_audits = {
        name: records.audit()
        for name, records in scoped_streams.items()
        if formal_scope is not None and records is not None and hasattr(records, "audit")
    }
    phase_metric_rows = build_phase_metric_rows(formal_scope, stream_audits)
    failure_mode_rows = build_failure_mode_rows(
        object_anchor_rows,
        anchor_vector_rows,
        persistent_summary["per_moving_object"],
    )
    write_csv(phase_metrics_csv, PHASE_METRICS_HEADER, phase_metric_rows)
    write_csv(failure_modes_csv, FAILURE_MODES_HEADER, failure_mode_rows)
    scope_payload = formal_scope.to_dict() if formal_scope is not None else {
        "schema_version": 1,
        "scope_mode": "LEGACY_UNSCOPED",
    }
    if formal_scope is not None:
        write_scope_json(
            analysis_scope_json,
            formal_scope,
            stream_audits=stream_audits,
            extra={
                "truth_track_audit": truth_track_audit,
                "link_track_audit": link_track_audit,
            },
        )
    else:
        with analysis_scope_json.open("w") as handle:
            json.dump(scope_payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
    controller_configuration = load_protocol_controller_configuration(formal_scope)
    timing_audit = load_protocol_timing_audit(formal_scope)
    run_metrics_payload = {
        "schema_version": 1,
        "analysis_status": "INTERMEDIATE_METRICS_READY",
        "final_metric_formula_status": "PENDING_USER_REVIEW",
        "run_directory": str(run_dir),
        "analysis_directory": str(output_dir.resolve()),
        "analysis_scope": scope_payload,
        "stream_audits": stream_audits,
        "truth_track_count": len(truth_tracks),
        "link_track_count": len(link_tracks),
        "object_id_catalog_count": len(object_id_catalog),
        "object_anchor_type_row_count": len(object_anchor_rows),
        "anchor_type_inventory": anchor_type_inventory_rows,
        "anchor_vector_row_count": len(anchor_vector_rows),
        "persistent_object_row_count": len(
            persistent_summary["per_moving_object"]
        ),
        "failure_mode_row_count": len(failure_mode_rows),
        "controller_configuration": controller_configuration,
        "timing_audit": timing_audit,
    }
    with run_metrics_json.open("w") as handle:
        json.dump(
            run_metrics_payload,
            handle,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        handle.write("\n")
    with alert_metrics_json.open("w") as handle:
        json.dump(
            {
                "analysis_scope": scope_payload,
                "analysis_status": "INTERMEDIATE_METRICS_READY",
                "final_metric_formula_status": "PENDING_USER_REVIEW",
                "anchor_states_status": (
                    "available" if anchor_record_count else "empty"
                ),
                "object_id_catalog": {
                    str(object_id): object_name
                    for object_id, object_name in sorted(object_id_catalog.items())
                },
                "anchor_type_metrics": anchor_type_rows,
                "anchor_type_inventory": anchor_type_inventory_rows,
                "anchor_vector_type_metrics": anchor_vector_type_rows,
                "alarm_operating_point": alarm_operating_point,
                "metric_contract": {
                    "truth_motion_epsilon_m": truth_policy.translation_deadband_m,
                    "truth_angular_motion_epsilon_deg": truth_policy.rotation_deadband_deg,
                    "truth_linear_speed_deadband_mps": truth_policy.linear_speed_deadband_mps,
                    "truth_angular_speed_deadband_degps": truth_policy.angular_speed_deadband_degps,
                    "truth_sustained_motion_samples": truth_policy.sustained_motion_samples,
                    "anchor_types": ["PLANE", "EDGE", "BAND"],
                    "anchor_type_classifier": ANCHOR_TYPE_CLASSIFIER_CONTRACT,
                    "anchor_detection_unit": (
                        "object_x_anchor_type_any_significant_anchor"
                    ),
                    "missing_anchor_policy": "not_evaluable_not_false_negative",
                    "anchor_vector_truth": (
                        "rigid_displacement_at_anchor_reference_position"
                    ),
                    "final_alert_unit": "persistent_confirmed_track",
                    "final_geometry_match": (
                        "region_bbox_or_center_against_rigid_truth_extent"
                    ),
                    "final_identity_match": (
                        "propagated_object_id_consistent_with_geometry_and_truth_motion"
                    ),
                    "vector_averaging": [
                        "observation_micro", "per_anchor_macro",
                        "per_object_x_type_macro",
                    ],
                    "full_occlusion_policy": "exclude_from_evaluation",
                    "object_evaluability": (
                        "monitoring_window_object_hits_overlapping_truth_motion"
                    ),
                    "reference_reset_policy": "new_local_datum_per_epoch",
                    "scope_policy": (
                        "formal_protocol_time_and_epoch"
                        if formal_scope is not None else "legacy_unscoped"
                    ),
                    "final_metric_formula_status": "PENDING_USER_REVIEW",
                },
                "persistent_alert_metrics": persistent_summary,
            },
            handle,
            indent=2,
            sort_keys=True,
        )
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
            "anchor_states": "available" if anchor_record_count else "empty",
        },
        alignment,
        persistent_summary,
    )
    plot_gt_motion_timeline(truth_tracks, link_tracks_by_model, summary_rows, gt_motion_timeline_png)
    plot_detection_stage_timeline(evaluation_evidence_records, region_records, motion_records,
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
        object_anchor_metrics_csv=object_anchor_metrics_csv,
        anchor_type_metrics_csv=anchor_type_metrics_csv,
        anchor_vector_metrics_csv=anchor_vector_metrics_csv,
        anchor_vector_type_metrics_csv=anchor_vector_type_metrics_csv,
        persistent_object_metrics_csv=persistent_object_metrics_csv,
        alert_metrics_json=alert_metrics_json,
        analysis_scope_json=analysis_scope_json,
        run_metrics_json=run_metrics_json,
        phase_metrics_csv=phase_metrics_csv,
        failure_modes_csv=failure_modes_csv,
        anchor_type_inventory_csv=anchor_type_inventory_csv,
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
    parser.add_argument(
        "--validation-policy",
        choices=("strict", "formal_analysis_v2", "recording_v2"),
        default="strict",
    )
    parser.add_argument(
        "--max-anchor-processing-drop-fraction",
        type=float,
        default=None,
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_dir = args.run_dir if args.run_dir else resolve_latest_sim_run(args.output_root)
    outputs = analyze_sim_run(
        run_dir,
        output_dir=args.analysis_dir,
        world_file=args.world_file,
        validation_policy=args.validation_policy,
        max_anchor_processing_drop_fraction=(
            args.max_anchor_processing_drop_fraction
        ),
    )
    print(f"analysis_dir: {outputs.output_dir}")
    print(f"summary_csv: {outputs.summary_csv}")
    print(f"outlier_csv: {outputs.outlier_csv}")
    print(f"report_md: {outputs.report_md}")
    print(f"gt_motion_timeline_png: {outputs.gt_motion_timeline_png}")
    print(f"detection_stage_timeline_png: {outputs.detection_stage_timeline_png}")
    print(f"spatial_overlay_png: {outputs.spatial_overlay_png}")
    print(f"object_anchor_metrics_csv: {outputs.object_anchor_metrics_csv}")
    print(f"anchor_type_metrics_csv: {outputs.anchor_type_metrics_csv}")
    print(f"anchor_vector_metrics_csv: {outputs.anchor_vector_metrics_csv}")
    print(f"anchor_vector_type_metrics_csv: {outputs.anchor_vector_type_metrics_csv}")
    print(f"persistent_object_metrics_csv: {outputs.persistent_object_metrics_csv}")
    print(f"alert_metrics_json: {outputs.alert_metrics_json}")


if __name__ == "__main__":
    main()
