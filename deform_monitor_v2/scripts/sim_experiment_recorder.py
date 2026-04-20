#!/usr/bin/env python3
# Author: zgliu@cumt.edu.cn
# Affiliation: China University of Mining and Technology
# Open-source release date: 2026-04-20

import csv
import datetime as dt
import json
import math
import pathlib
import re
from typing import Any

try:
    import rospy
    import tf
    from gazebo_msgs.msg import LinkStates, ModelStates
    from nav_msgs.msg import Odometry
    from deform_monitor_v2.msg import (
        AnchorStates,
        MotionClusters,
        PersistentRiskRegions,
        RiskEvidenceArray,
        RiskRegions,
        StructureMotions,
    )
except ImportError:  # pragma: no cover - allows pure-Python helper tests
    rospy = None
    tf = None
    AnchorStates = None
    LinkStates = None
    ModelStates = None
    Odometry = None
    MotionClusters = None
    PersistentRiskRegions = None
    RiskEvidenceArray = None
    RiskRegions = None
    StructureMotions = None


TRUTH_OBJECT_HEADER = [
    "recorded_time_sec",
    "model_name",
    "frame_id",
    "position_x",
    "position_y",
    "position_z",
    "orientation_x",
    "orientation_y",
    "orientation_z",
    "orientation_w",
]

TRUTH_LINK_HEADER = [
    "recorded_time_sec",
    "scoped_link_name",
    "model_name",
    "link_name",
    "frame_id",
    "position_x",
    "position_y",
    "position_z",
    "orientation_x",
    "orientation_y",
    "orientation_z",
    "orientation_w",
]

EGO_INITIAL_POSE_HEADER = [
    "frame_id",
    "position_x",
    "position_y",
    "position_z",
    "orientation_x",
    "orientation_y",
    "orientation_z",
    "orientation_w",
]

RUN_DIR_PATTERN = re.compile(r"^sim_run_(\d{3})$")
SAFE_NAME_PATTERN = re.compile(r"[^A-Za-z0-9_.-]+")
DEFAULT_OUTPUT_ROOT = pathlib.Path.home() / ".ros" / "alert" / "output"
SENSOR_POSE_MAX_AGE_SEC = 0.25
PERSISTENT_STATE_NAMES = {
    0: "CANDIDATE",
    1: "CONFIRMED",
    2: "FADING",
}
REGION_TYPE_NAMES = {
    0: "NONE",
    1: "DISPLACEMENT_LIKE",
    2: "DISAPPEARANCE_LIKE",
    3: "MIXED",
}


def sanitize_name(name: Any) -> str:
    sanitized = SAFE_NAME_PATTERN.sub("_", str(name).strip()).strip("._")
    return sanitized or "unnamed"


def parse_scoped_link_name(scoped_name):
    parts = str(scoped_name).split("::", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return str(scoped_name), ""


def allocate_run_directory(day_dir):
    day_dir = pathlib.Path(day_dir)
    max_index = -1

    if day_dir.exists():
        for child in day_dir.iterdir():
            if not child.is_dir():
                continue
            match = RUN_DIR_PATTERN.match(child.name)
            if match is None:
                continue
            max_index = max(max_index, int(match.group(1)))

    return day_dir / ("sim_run_%03d" % (max_index + 1))


def point_to_dict(msg):
    return {
        "x": float(getattr(msg, "x", 0.0)),
        "y": float(getattr(msg, "y", 0.0)),
        "z": float(getattr(msg, "z", 0.0)),
    }


def quaternion_to_dict(msg):
    return {
        "x": float(getattr(msg, "x", 0.0)),
        "y": float(getattr(msg, "y", 0.0)),
        "z": float(getattr(msg, "z", 0.0)),
        "w": float(getattr(msg, "w", 1.0)),
    }


def pose_to_dict(pose):
    return {
        "position": point_to_dict(getattr(pose, "position", None)),
        "orientation": quaternion_to_dict(getattr(pose, "orientation", None)),
    }


def time_to_dict(value):
    if value is None:
        return None

    if hasattr(value, "secs") or hasattr(value, "nsecs"):
        secs = int(getattr(value, "secs", 0))
        nsecs = int(getattr(value, "nsecs", 0))
        return {
            "secs": secs,
            "nsecs": nsecs,
            "sec": secs + (nsecs / 1e9),
        }

    if hasattr(value, "to_sec"):
        sec = float(value.to_sec())
        secs = int(sec)
        nsecs = int(round((sec - secs) * 1e9))
        if nsecs >= 1000000000:
            secs += 1
            nsecs -= 1000000000
        if nsecs < 0:
            secs -= 1
            nsecs += 1000000000
        return {
            "secs": secs,
            "nsecs": nsecs,
            "sec": secs + (nsecs / 1e9),
        }

    return value


def common_record_time_sec_from_payload(payload):
    """
    Extract a float timestamp (seconds) from a serialised event payload dict.
    Tries header.stamp first, then recorded_at, returning None on failure.
    Mirrors the logic of common.record_time_sec() but operates on an in-memory
    dict rather than a JSONL-parsed record.
    """
    for key in ("header", "recorded_at"):
        ts = payload.get(key)
        if isinstance(ts, dict):
            stamp = ts.get("stamp") if key == "header" else ts
            if isinstance(stamp, dict):
                secs  = stamp.get("secs",  stamp.get("sec", 0))
                nsecs = stamp.get("nsecs", 0)
                try:
                    return float(secs) + float(nsecs) / 1e9
                except (TypeError, ValueError):
                    pass
    return None


def coerce_float(value, default=None):
    if value is None or value == "":
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(parsed):
        return default
    return parsed


def vector3_dict(x=0.0, y=0.0, z=0.0):
    return {
        "x": float(x),
        "y": float(y),
        "z": float(z),
    }


def vector_norm(vector):
    if not isinstance(vector, dict):
        return 0.0
    return math.sqrt(
        float(vector.get("x", 0.0)) ** 2
        + float(vector.get("y", 0.0)) ** 2
        + float(vector.get("z", 0.0)) ** 2
    )


def normalize_vector_dict(vector):
    norm = vector_norm(vector)
    if norm <= 1.0e-12:
        return vector3_dict()
    return {
        "x": float(vector.get("x", 0.0)) / norm,
        "y": float(vector.get("y", 0.0)) / norm,
        "z": float(vector.get("z", 0.0)) / norm,
    }


def copy_time_dict(value):
    if not isinstance(value, dict):
        return None
    return {
        "secs": int(value.get("secs", 0)),
        "nsecs": int(value.get("nsecs", 0)),
        "sec": float(value.get("sec", 0.0)),
    }


def pose_dict_is_finite(pose_dict):
    if not isinstance(pose_dict, dict):
        return False

    position = pose_dict.get("position")
    orientation = pose_dict.get("orientation")
    if not isinstance(position, dict) or not isinstance(orientation, dict):
        return False

    required_position_keys = ("x", "y", "z")
    required_orientation_keys = ("x", "y", "z", "w")
    if any(key not in position for key in required_position_keys):
        return False
    if any(key not in orientation for key in required_orientation_keys):
        return False

    try:
        values = (
            float(position["x"]),
            float(position["y"]),
            float(position["z"]),
            float(orientation["x"]),
            float(orientation["y"]),
            float(orientation["z"]),
            float(orientation["w"]),
        )
    except (TypeError, ValueError):
        return False

    return all(math.isfinite(value) for value in values)


def normalize_quaternion_tuple(quaternion):
    x, y, z, w = quaternion
    norm = math.sqrt((x * x) + (y * y) + (z * z) + (w * w))
    if not math.isfinite(norm) or norm <= 1.0e-12:
        return (0.0, 0.0, 0.0, 1.0)
    return (x / norm, y / norm, z / norm, w / norm)


def normalize_pose_dict(pose_dict):
    pose_dict = pose_dict or {}
    position = pose_dict.get("position", {}) if isinstance(pose_dict, dict) else {}
    orientation = pose_dict.get("orientation", {}) if isinstance(pose_dict, dict) else {}
    quaternion = normalize_quaternion_tuple(
        (
            float(orientation.get("x", 0.0)),
            float(orientation.get("y", 0.0)),
            float(orientation.get("z", 0.0)),
            float(orientation.get("w", 1.0)),
        )
    )
    return {
        "position": {
            "x": float(position.get("x", 0.0)),
            "y": float(position.get("y", 0.0)),
            "z": float(position.get("z", 0.0)),
        },
        "orientation": {
            "x": quaternion[0],
            "y": quaternion[1],
            "z": quaternion[2],
            "w": quaternion[3],
        },
    }


def quaternion_multiply(lhs, rhs):
    lx, ly, lz, lw = lhs
    rx, ry, rz, rw = rhs
    return (
        (lw * rx) + (lx * rw) + (ly * rz) - (lz * ry),
        (lw * ry) - (lx * rz) + (ly * rw) + (lz * rx),
        (lw * rz) + (lx * ry) - (ly * rx) + (lz * rw),
        (lw * rw) - (lx * rx) - (ly * ry) - (lz * rz),
    )


def quaternion_conjugate(quaternion):
    x, y, z, w = quaternion
    return (-x, -y, -z, w)


def rotate_point(point, quaternion):
    quaternion = normalize_quaternion_tuple(quaternion)
    rotated = quaternion_multiply(
        quaternion_multiply(quaternion, (point[0], point[1], point[2], 0.0)),
        quaternion_conjugate(quaternion),
    )
    return (rotated[0], rotated[1], rotated[2])


def compose_pose_dicts(base_pose, relative_pose):
    if not pose_dict_is_finite(base_pose) or not pose_dict_is_finite(relative_pose):
        raise ValueError("compose_pose_dicts requires finite pose dictionaries")

    base_position = base_pose["position"]
    base_orientation = base_pose["orientation"]
    relative_position = relative_pose["position"]
    relative_orientation = relative_pose["orientation"]

    base_quaternion = (
        float(base_orientation["x"]),
        float(base_orientation["y"]),
        float(base_orientation["z"]),
        float(base_orientation["w"]),
    )
    relative_quaternion = (
        float(relative_orientation["x"]),
        float(relative_orientation["y"]),
        float(relative_orientation["z"]),
        float(relative_orientation["w"]),
    )
    rotated_relative_position = rotate_point(
        (
            float(relative_position["x"]),
            float(relative_position["y"]),
            float(relative_position["z"]),
        ),
        base_quaternion,
    )
    composed_orientation = quaternion_multiply(base_quaternion, relative_quaternion)

    return {
        "position": {
            "x": float(base_position["x"]) + rotated_relative_position[0],
            "y": float(base_position["y"]) + rotated_relative_position[1],
            "z": float(base_position["z"]) + rotated_relative_position[2],
        },
        "orientation": {
            "x": composed_orientation[0],
            "y": composed_orientation[1],
            "z": composed_orientation[2],
            "w": composed_orientation[3],
        },
    }


def invert_pose_dict(pose_dict):
    if not pose_dict_is_finite(pose_dict):
        raise ValueError("invert_pose_dict requires a finite pose dictionary")

    orientation = pose_dict["orientation"]
    position = pose_dict["position"]
    inverse_orientation = quaternion_conjugate(
        normalize_quaternion_tuple(
            (
                float(orientation["x"]),
                float(orientation["y"]),
                float(orientation["z"]),
                float(orientation["w"]),
            )
        )
    )
    inverse_translation = rotate_point(
        (
            -float(position["x"]),
            -float(position["y"]),
            -float(position["z"]),
        ),
        inverse_orientation,
    )
    return {
        "position": {
            "x": inverse_translation[0],
            "y": inverse_translation[1],
            "z": inverse_translation[2],
        },
        "orientation": {
            "x": inverse_orientation[0],
            "y": inverse_orientation[1],
            "z": inverse_orientation[2],
            "w": inverse_orientation[3],
        },
    }


def derive_world_from_algorithm_pose(truth_reference_pose_world, algorithm_reference_pose_algorithm):
    if not pose_dict_is_finite(truth_reference_pose_world):
        raise ValueError("truth_reference_pose_world must be finite")
    if not pose_dict_is_finite(algorithm_reference_pose_algorithm):
        raise ValueError("algorithm_reference_pose_algorithm must be finite")

    algorithm_from_reference_pose = invert_pose_dict(algorithm_reference_pose_algorithm)
    return compose_pose_dicts(truth_reference_pose_world, algorithm_from_reference_pose)


def format_tum_line(timestamp_sec, position, orientation):
    return (
        f"{float(timestamp_sec):.9f} "
        f"{float(position['x']):.9f} {float(position['y']):.9f} {float(position['z']):.9f} "
        f"{float(orientation['x']):.9f} {float(orientation['y']):.9f} "
        f"{float(orientation['z']):.9f} {float(orientation['w']):.9f}\n"
    )


def write_tum_sample_pair(gt_path, odom_path, timestamp_sec, sensor_pose_world, odom_pose):
    if not pose_dict_is_finite(sensor_pose_world) or not pose_dict_is_finite(odom_pose):
        return False

    gt_path = pathlib.Path(gt_path)
    odom_path = pathlib.Path(odom_path)
    gt_path.parent.mkdir(parents=True, exist_ok=True)
    odom_path.parent.mkdir(parents=True, exist_ok=True)

    gt_line = format_tum_line(
        timestamp_sec=timestamp_sec,
        position=sensor_pose_world["position"],
        orientation=sensor_pose_world["orientation"],
    )
    odom_line = format_tum_line(
        timestamp_sec=timestamp_sec,
        position=odom_pose["position"],
        orientation=odom_pose["orientation"],
    )

    with gt_path.open("a") as gt_handle:
        gt_handle.write(gt_line)

    with odom_path.open("a") as odom_handle:
        odom_handle.write(odom_line)

    return True


def build_run_info_payload(
    run_dir,
    truth_frame,
    algorithm_frame,
    ego_model_name,
    model_states_topic,
    link_states_topic,
    risk_evidence_topic,
    risk_regions_topic,
    persistent_risk_regions_topic,
    structure_motions_topic,
    odometry_topic,
    sensor_scoped_link_name,
    gt_tum_filename,
    odom_tum_filename,
    ground_truth_odometry_topic="",
    sensor_frame_name="",
    clusters_topic="",
):
    enabled = bool(str(sensor_frame_name).strip() or str(sensor_scoped_link_name).strip())
    return {
        "created_at_iso": dt.datetime.now().isoformat(),
        "run_directory": str(run_dir),
        "truth_frame": str(truth_frame),
        "algorithm_frame": str(algorithm_frame),
        "ego_model_name": str(ego_model_name),
        "sensor_scoped_link_name": str(sensor_scoped_link_name),
        "sensor_frame_name": str(sensor_frame_name),
        "topics": {
            "model_states": str(model_states_topic),
            "link_states": str(link_states_topic),
            "ground_truth_odometry": str(ground_truth_odometry_topic),
            "clusters": str(clusters_topic),
            "risk_evidence": str(risk_evidence_topic),
            "risk_regions": str(risk_regions_topic),
            "persistent_risk_regions": str(persistent_risk_regions_topic),
            "structure_motions": str(structure_motions_topic),
            "odometry": str(odometry_topic),
        },
        "runtime_policy": {
            "transform_algorithm_outputs_at_runtime": False,
            "alignment_mode": "initial_ego_pose",
        },
        "trajectory_export": {
            "enabled": enabled,
            "gt_file": str(gt_tum_filename),
            "odom_file": str(odom_tum_filename),
            "timestamp_policy": "odometry_master_clock",
            "runtime_alignment_applied": False,
            "gt_pose_source": "ground_truth_odometry_plus_tf"
            if str(sensor_frame_name).strip()
            else "gazebo_link_states_cache",
        },
    }


def build_explicit_control_metadata(
    controlled_object="",
    command_frame="",
    linear_velocity=None,
    angular_velocity_deg=None,
    axis=None,
    start_delay_sec=None,
    duration_sec=None,
    scenario_id="",
):
    controlled_object = str(controlled_object).strip()
    command_frame = str(command_frame).strip()
    scenario_id = str(scenario_id).strip()
    linear_velocity = linear_velocity if isinstance(linear_velocity, dict) else vector3_dict()
    angular_velocity_deg = (
        angular_velocity_deg if isinstance(angular_velocity_deg, dict) else vector3_dict()
    )
    axis = axis if isinstance(axis, dict) else vector3_dict()

    if vector_norm(axis) <= 1.0e-12:
        if vector_norm(linear_velocity) > 1.0e-12:
            axis = normalize_vector_dict(linear_velocity)
        elif vector_norm(angular_velocity_deg) > 1.0e-12:
            axis = normalize_vector_dict(angular_velocity_deg)
        else:
            axis = vector3_dict()

    has_signal = any(
        [
            controlled_object,
            command_frame,
            vector_norm(linear_velocity) > 1.0e-12,
            vector_norm(angular_velocity_deg) > 1.0e-12,
            vector_norm(axis) > 1.0e-12,
            start_delay_sec is not None,
            duration_sec is not None,
            scenario_id,
        ]
    )
    if not has_signal:
        return None

    return {
        "controlled_object": controlled_object,
        "command_frame": command_frame,
        "velocity": {
            "linear_mps": linear_velocity,
            "angular_deg_per_sec": angular_velocity_deg,
        },
        "axis": normalize_vector_dict(axis),
        "start_delay_sec": coerce_float(start_delay_sec, None),
        "duration_sec": coerce_float(duration_sec, None),
        "scenario_id": scenario_id,
    }


_PARAM_MISSING = object()


def _namespace_has_required_keys(get_param, prefix, keys):
    try:
        for key in keys:
            if get_param(prefix + "/" + key, _PARAM_MISSING) is _PARAM_MISSING:
                return False
    except Exception:
        return False
    return True


def _looks_like_motion_controller_namespace(get_param, prefix):
    required_keys = (
        "model_name",
        "command_frame",
        "control_rate",
        "command_timeout",
        "start_delay",
        "duration",
        "scenario_id",
    )
    motion_keys = (
        "linear_x",
        "linear_y",
        "linear_z",
        "angular_x_deg",
        "angular_y_deg",
        "angular_z_deg",
    )
    if not _namespace_has_required_keys(get_param, prefix, required_keys):
        return False
    if not any(
        get_param(prefix + "/" + key, _PARAM_MISSING) is not _PARAM_MISSING
        for key in motion_keys
    ):
        return False

    return True


def discover_controlled_objects(get_param, get_param_names):
    discovered = []
    try:
        param_names = list(get_param_names())
    except Exception:
        return discovered

    controller_prefixes = set()
    for name in param_names:
        if not str(name).endswith("/model_name"):
            continue
        prefix = str(name).rsplit("/", 1)[0]
        if _looks_like_motion_controller_namespace(get_param, prefix):
            controller_prefixes.add(prefix)

    for prefix in sorted(controller_prefixes):
        model_name = str(get_param(prefix + "/model_name", "")).strip()
        if not model_name:
            continue

        linear_velocity = vector3_dict(
            coerce_float(get_param(prefix + "/linear_x", 0.0), 0.0),
            coerce_float(get_param(prefix + "/linear_y", 0.0), 0.0),
            coerce_float(get_param(prefix + "/linear_z", 0.0), 0.0),
        )
        angular_velocity_deg = vector3_dict(
            coerce_float(get_param(prefix + "/angular_x_deg", 0.0), 0.0),
            coerce_float(get_param(prefix + "/angular_y_deg", 0.0), 0.0),
            coerce_float(get_param(prefix + "/angular_z_deg", 0.0), 0.0),
        )
        axis = vector3_dict(
            coerce_float(get_param(prefix + "/axis_x", 0.0), 0.0),
            coerce_float(get_param(prefix + "/axis_y", 0.0), 0.0),
            coerce_float(get_param(prefix + "/axis_z", 0.0), 0.0),
        )
        if vector_norm(axis) <= 1.0e-12:
            if vector_norm(linear_velocity) > 1.0e-12:
                axis = normalize_vector_dict(linear_velocity)
            elif vector_norm(angular_velocity_deg) > 1.0e-12:
                axis = normalize_vector_dict(angular_velocity_deg)
            else:
                axis = vector3_dict()

        discovered.append(
            {
                "controller_namespace": prefix,
                "controlled_object": model_name,
                "command_frame": str(get_param(prefix + "/command_frame", "")).strip(),
                "velocity": {
                    "linear_mps": linear_velocity,
                    "angular_deg_per_sec": angular_velocity_deg,
                },
                "axis": axis,
                "start_delay_sec": coerce_float(get_param(prefix + "/start_delay", None), None),
                "duration_sec": coerce_float(get_param(prefix + "/duration", None), None),
                "scenario_id": str(get_param(prefix + "/scenario_id", "")).strip(),
            }
        )

    return discovered


def build_scenario_manifest_payload(
    run_dir,
    scenario_id="",
    explicit_control=None,
    discovered_controls=None,
):
    explicit_control = explicit_control if isinstance(explicit_control, dict) else None
    discovered_controls = discovered_controls if isinstance(discovered_controls, list) else []

    explicit_control = explicit_control if isinstance(explicit_control, dict) else None
    discovered_controls = discovered_controls if isinstance(discovered_controls, list) else []

    if discovered_controls:
        controls = discovered_controls
        source = "discovered"
    elif explicit_control is not None:
        controls = [explicit_control]
        source = "explicit"
    else:
        controls = []
        source = "empty"

    derived_scenario_id = str(scenario_id).strip()
    if not derived_scenario_id:
        for control in controls:
            candidate = str(control.get("scenario_id", "")).strip()
            if candidate:
                derived_scenario_id = candidate
                break

    return {
        "created_at_iso": dt.datetime.now().isoformat(),
        "run_directory": str(run_dir),
        "scenario_id": derived_scenario_id,
        "source": source,
        "controls": controls,
    }


def select_authoritative_discovered_controls(current_scenario_id, discovered_controls):
    current_scenario_id = str(current_scenario_id).strip()
    discovered_controls = discovered_controls if isinstance(discovered_controls, list) else []
    if not current_scenario_id:
        return []
    if len(discovered_controls) != 2:
        return []

    expected_namespaces = {
        "/model_01_motion": "model_01",
        "/model_02_motion": "model_02",
    }
    selected = {}
    for control in discovered_controls:
        if not isinstance(control, dict):
            return []
        controller_namespace = str(control.get("controller_namespace", "")).strip()
        controlled_object = str(control.get("controlled_object", "")).strip()
        expected_object = expected_namespaces.get(controller_namespace)
        if expected_object is None or controlled_object != expected_object:
            return []
        if str(control.get("scenario_id", "")).strip() != current_scenario_id:
            return []
        command_frame = str(control.get("command_frame", "")).strip()
        if not command_frame:
            return []
        start_delay_sec = control.get("start_delay_sec")
        duration_sec = control.get("duration_sec")
        try:
            if not math.isfinite(float(start_delay_sec)):
                return []
            if not math.isfinite(float(duration_sec)):
                return []
        except (TypeError, ValueError):
            return []
        selected[controlled_object] = control

    if set(selected) != set(expected_namespaces.values()):
        return []

    return [selected["model_01"], selected["model_02"]]


def build_config_snapshot_payload(run_dir, node_param_root, source_config_path, parameter_tree):
    return {
        "created_at_iso": dt.datetime.now().isoformat(),
        "run_directory": str(run_dir),
        "node_param_root": str(node_param_root),
        "source_config_path": str(source_config_path),
        "parameters": parameter_tree if isinstance(parameter_tree, dict) else {},
    }


def build_ablation_manifest_payload(
    run_dir, node_param_root, source_config_path, parameter_tree
):
    parameter_tree = parameter_tree if isinstance(parameter_tree, dict) else {}
    deform_monitor = parameter_tree.get("deform_monitor", {})
    if not isinstance(deform_monitor, dict):
        deform_monitor = {}

    covariance = deform_monitor.get("covariance", {})
    background_bias = deform_monitor.get("background_bias", {})
    imm = deform_monitor.get("imm", {})
    significance = deform_monitor.get("significance", {})
    directional_motion = deform_monitor.get("directional_motion", {})
    ablation = deform_monitor.get("ablation", {})

    if not isinstance(covariance, dict):
        covariance = {}
    if not isinstance(background_bias, dict):
        background_bias = {}
    if not isinstance(imm, dict):
        imm = {}
    if not isinstance(significance, dict):
        significance = {}
    if not isinstance(directional_motion, dict):
        directional_motion = {}
    if not isinstance(ablation, dict):
        ablation = {}

    switches = {
        "disable_covariance_inflation": bool(
            ablation.get("disable_covariance_inflation", False)
        ),
        "disable_type_constraint": bool(
            ablation.get("disable_type_constraint", False)
        ),
        "single_model_ekf": bool(ablation.get("single_model_ekf", False)),
        "disable_cusum": bool(ablation.get("disable_cusum", False)),
        "disable_directional_accumulation": bool(
            ablation.get("disable_directional_accumulation", False)
        ),
        "disable_drift_compensation": bool(
            ablation.get("disable_drift_compensation", False)
        ),
    }

    raw_alpha_xi = float(covariance.get("alpha_xi", 1.0))
    return {
        "created_at_iso": dt.datetime.now().isoformat(),
        "run_directory": str(run_dir),
        "node_param_root": str(node_param_root),
        "source_config_path": str(source_config_path),
        "variant": str(ablation.get("variant", "full_pipeline")),
        "parameter_tree_found": bool(parameter_tree),
        "switches": switches,
        "effective_runtime": {
            "covariance_alpha_xi": 1.0
            if switches["disable_covariance_inflation"]
            else raw_alpha_xi,
            "background_bias_enable": bool(background_bias.get("enable", True))
            and not switches["disable_drift_compensation"],
            "imm_enable_model_competition": bool(
                imm.get("enable_model_competition", True)
            )
            and not switches["single_model_ekf"],
            "imm_enable_type_constraint": bool(
                imm.get("enable_type_constraint", True)
            )
            and not switches["disable_type_constraint"],
            "significance_enable_cusum": bool(
                significance.get("enable_cusum", True)
            )
            and not switches["disable_cusum"],
            "directional_motion_enable": bool(directional_motion.get("enable", True))
            and not switches["disable_directional_accumulation"],
        },
    }


def serialize_anchor_state(msg):
    return {
        "id": int(getattr(msg, "id", 0)),
        "anchor_type": int(getattr(msg, "anchor_type", 0)),
        "center": point_to_dict(getattr(msg, "center", None)),
        "ref_center": point_to_dict(getattr(msg, "ref_center", None)),
        "disp_mean": list(getattr(msg, "disp_mean", [0.0, 0.0, 0.0])),
        "disp_norm": float(getattr(msg, "disp_norm", 0.0)),
        "cusum_score": float(getattr(msg, "cusum_score", 0.0)),
        "comparable": bool(getattr(msg, "comparable", False)),
        "significant": bool(getattr(msg, "significant", False)),
        "reacquired": bool(getattr(msg, "reacquired", False)),
        "detection_mode": int(getattr(msg, "detection_mode", 0)),
        "disappearance_score": float(getattr(msg, "disappearance_score", 0.0)),
    }


def serialize_anchor_states(msg):
    return {
        "header": serialize_header(msg.header),
        "anchors": [serialize_anchor_state(a) for a in msg.anchors],
    }


def serialize_structure_motion(msg):
    return {
        "id": int(getattr(msg, "id", 0)),
        "old_region_id": int(getattr(msg, "old_region_id", 0)),
        "new_region_id": int(getattr(msg, "new_region_id", 0)),
        "motion_type": int(getattr(msg, "motion_type", 0)),
        "old_center": point_to_dict(getattr(msg, "old_center", None)),
        "new_center": point_to_dict(getattr(msg, "new_center", None)),
        "bbox_old_min": point_to_dict(getattr(msg, "bbox_old_min", None)),
        "bbox_old_max": point_to_dict(getattr(msg, "bbox_old_max", None)),
        "bbox_new_min": point_to_dict(getattr(msg, "bbox_new_min", None)),
        "bbox_new_max": point_to_dict(getattr(msg, "bbox_new_max", None)),
        "motion": point_to_dict(getattr(msg, "motion", None)),
        "distance": float(getattr(msg, "distance", 0.0)),
        "match_cost": float(getattr(msg, "match_cost", 0.0)),
        "confidence": float(getattr(msg, "confidence", 0.0)),
        "support_old": int(getattr(msg, "support_old", 0)),
        "support_new": int(getattr(msg, "support_new", 0)),
        "significant": bool(getattr(msg, "significant", False)),
    }


def build_frame_alignment_metadata(
    ego_pose_world,
    truth_frame,
    algorithm_frame,
    truth_reference_frame="",
    truth_reference_pose_world=None,
    algorithm_reference_frame="",
    algorithm_reference_pose_algorithm=None,
):
    if isinstance(ego_pose_world, dict):
        serialized_ego_pose_world = ego_pose_world
    else:
        serialized_ego_pose_world = pose_to_dict(ego_pose_world)

    normalized_ego_pose_world = normalize_pose_dict(serialized_ego_pose_world)
    normalized_truth_reference_pose_world = (
        normalize_pose_dict(truth_reference_pose_world)
        if truth_reference_pose_world is not None
        else normalized_ego_pose_world
    )
    normalized_algorithm_reference_pose_algorithm = (
        normalize_pose_dict(algorithm_reference_pose_algorithm)
        if algorithm_reference_pose_algorithm is not None
        else {
            "position": {"x": 0.0, "y": 0.0, "z": 0.0},
            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        }
    )
    world_from_algorithm_pose = derive_world_from_algorithm_pose(
        normalized_truth_reference_pose_world,
        normalized_algorithm_reference_pose_algorithm,
    )
    algorithm_pose_truth = invert_pose_dict(world_from_algorithm_pose)

    return {
        "truth_frame": str(truth_frame),
        "algorithm_frame": str(algorithm_frame),
        "alignment_mode": "initial_ego_pose",
        "sim_only": True,
        "ego_initial_pose_world": serialized_ego_pose_world,
        "truth_reference_frame": str(truth_reference_frame).strip(),
        "truth_reference_pose_world": normalized_truth_reference_pose_world,
        "algorithm_reference_frame": str(algorithm_reference_frame).strip(),
        "algorithm_reference_pose_algorithm": normalized_algorithm_reference_pose_algorithm,
        "world_from_algorithm_transform": {
            "source_frame": str(algorithm_frame),
            "target_frame": str(truth_frame),
            "pose": world_from_algorithm_pose,
        },
        "algorithm_from_world_transform": {
            "source_frame": str(truth_frame),
            "target_frame": str(algorithm_frame),
            "pose": algorithm_pose_truth,
        },
        "notes": "Algorithm outputs remain in their native frame; alignment is performed offline.",
    }


def serialize_header(msg):
    header = getattr(msg, "header", None)
    if header is None:
        return {}
    return {
        "seq": int(getattr(header, "seq", 0)),
        "frame_id": str(getattr(header, "frame_id", "")),
        "stamp": time_to_dict(getattr(header, "stamp", None)),
    }


def serialize_motion_cluster(msg):
    return {
        "id": int(getattr(msg, "id", 0)),
        "anchor_ids": [int(value) for value in getattr(msg, "anchor_ids", [])],
        "center": point_to_dict(getattr(msg, "center", None)),
        "bbox_min": point_to_dict(getattr(msg, "bbox_min", None)),
        "bbox_max": point_to_dict(getattr(msg, "bbox_max", None)),
        "disp_mean": [float(value) for value in getattr(msg, "disp_mean", [])],
        "disp_cov": [float(value) for value in getattr(msg, "disp_cov", [])],
        "chi2_stat": float(getattr(msg, "chi2_stat", 0.0)),
        "disp_norm": float(getattr(msg, "disp_norm", 0.0)),
        "confidence": float(getattr(msg, "confidence", 0.0)),
        "support_count": int(getattr(msg, "support_count", 0)),
        "significant": bool(getattr(msg, "significant", False)),
    }


def serialize_motion_clusters(msg):
    clusters = getattr(msg, "clusters", [])
    return {
        "header": serialize_header(msg),
        "clusters": [serialize_motion_cluster(item) for item in clusters],
    }


def serialize_risk_evidence(msg):
    evidences = getattr(msg, "evidences", [])
    return {
        "header": serialize_header(msg),
        "evidences": [
            serialize_risk_evidence_entry(item)
            for item in evidences
            if getattr(item, "active", False)
        ],
    }


def serialize_risk_regions(msg):
    regions = getattr(msg, "regions", [])
    return {
        "header": serialize_header(msg),
        "regions": [serialize_risk_region_entry(item) for item in regions],
    }


def serialize_persistent_risk_region(msg):
    return {
        "track_id": int(getattr(msg, "track_id", 0)),
        "state": int(getattr(msg, "state", 0)),
        "region_type": int(getattr(msg, "region_type", 0)),
        "center": point_to_dict(getattr(msg, "center", None)),
        "bbox_min": point_to_dict(getattr(msg, "bbox_min", None)),
        "bbox_max": point_to_dict(getattr(msg, "bbox_max", None)),
        "mean_risk": float(getattr(msg, "mean_risk", 0.0)),
        "peak_risk": float(getattr(msg, "peak_risk", 0.0)),
        "confidence": float(getattr(msg, "confidence", 0.0)),
        "accumulated_risk": float(getattr(msg, "accumulated_risk", 0.0)),
        "support_mass": float(getattr(msg, "support_mass", 0.0)),
        "spatial_span": float(getattr(msg, "spatial_span", 0.0)),
        "hit_streak": int(getattr(msg, "hit_streak", 0)),
        "miss_streak": int(getattr(msg, "miss_streak", 0)),
        "age_frames": int(getattr(msg, "age_frames", 0)),
        "confirmed": bool(getattr(msg, "confirmed", False)),
    }


def serialize_persistent_risk_regions(msg):
    regions = getattr(msg, "regions", [])
    return {
        "header": serialize_header(msg),
        "regions": [serialize_persistent_risk_region(item) for item in regions],
    }


def serialize_structure_motions(msg):
    motions = getattr(msg, "motions", [])
    return {
        "header": serialize_header(msg),
        "motions": [serialize_structure_motion(item) for item in motions],
    }


def serialize_risk_evidence_entry(msg):
    return {
        "id": int(getattr(msg, "id", 0)),
        "anchor_type": int(getattr(msg, "anchor_type", 0)),
        "obs_state": int(getattr(msg, "obs_state", 0)),
        "mode": int(getattr(msg, "mode", 0)),
        "position": point_to_dict(getattr(msg, "position", None)),
        "displacement": point_to_dict(getattr(msg, "displacement", None)),
        "displacement_score": float(getattr(msg, "displacement_score", 0.0)),
        "disappearance_score": float(getattr(msg, "disappearance_score", 0.0)),
        "graph_score": float(getattr(msg, "graph_score", 0.0)),
        "confidence": float(getattr(msg, "confidence", 0.0)),
        "risk_score": float(getattr(msg, "risk_score", 0.0)),
        "graph_neighbor_count": int(getattr(msg, "graph_neighbor_count", 0)),
        "observable": bool(getattr(msg, "observable", False)),
        "comparable": bool(getattr(msg, "comparable", False)),
        "active": bool(getattr(msg, "active", False)),
    }


def serialize_risk_region_entry(msg):
    return {
        "id": int(getattr(msg, "id", 0)),
        "region_type": int(getattr(msg, "region_type", 0)),
        "center": point_to_dict(getattr(msg, "center", None)),
        "bbox_min": point_to_dict(getattr(msg, "bbox_min", None)),
        "bbox_max": point_to_dict(getattr(msg, "bbox_max", None)),
        "mean_risk": float(getattr(msg, "mean_risk", 0.0)),
        "peak_risk": float(getattr(msg, "peak_risk", 0.0)),
        "confidence": float(getattr(msg, "confidence", 0.0)),
        "voxel_count": int(getattr(msg, "voxel_count", 0)),
        "significant": bool(getattr(msg, "significant", False)),
    }


class SimExperimentRecorder:
    def __init__(self):
        if rospy is None or ModelStates is None or LinkStates is None:
            raise RuntimeError("ROS environment is not available for sim_experiment_recorder.py")

        self.output_root = pathlib.Path(
            rospy.get_param("~output_root", str(DEFAULT_OUTPUT_ROOT))
        ).expanduser()
        self.truth_frame = str(rospy.get_param("~truth_frame", "world")).strip() or "world"
        self.algorithm_frame = (
            str(rospy.get_param("~algorithm_frame", "camera_init")).strip() or "camera_init"
        )
        self.ego_model_name = str(rospy.get_param("~ego_model_name", "mid360_fastlio")).strip()
        self.model_states_topic = str(
            rospy.get_param("~model_states_topic", "/gazebo/model_states")
        ).strip()
        self.link_states_topic = str(
            rospy.get_param("~link_states_topic", "/gazebo/link_states")
        ).strip()
        self.ground_truth_odometry_topic = str(
            rospy.get_param("~ground_truth_odometry_topic", "/ground_truth/odom")
        ).strip()
        self.clusters_topic = str(
            rospy.get_param("~clusters_topic", "/deform/clusters")
        ).strip()
        self.risk_evidence_topic = str(
            rospy.get_param("~risk_evidence_topic", "/deform/risk_evidence")
        ).strip()
        self.risk_regions_topic = str(
            rospy.get_param("~risk_regions_topic", "/deform/risk_regions")
        ).strip()
        self.persistent_risk_regions_topic = str(
            rospy.get_param(
                "~persistent_risk_regions_topic", "/deform/persistent_risk_regions"
            )
        ).strip()
        self.structure_motions_topic = str(
            rospy.get_param("~structure_motions_topic", "/deform/structure_motions")
        ).strip()
        self.anchor_states_topic = str(
            rospy.get_param("~anchor_states_topic", "/deform/anchors")
        ).strip()
        self.odometry_topic = str(rospy.get_param("~odometry_topic", "/Odometry")).strip()
        self.sensor_scoped_link_name = str(
            rospy.get_param("~sensor_scoped_link_name", "")
        ).strip()
        self.sensor_frame_name = str(rospy.get_param("~sensor_frame_name", "")).strip()
        if not self.sensor_frame_name and self.sensor_scoped_link_name:
            _, inferred_sensor_frame_name = parse_scoped_link_name(self.sensor_scoped_link_name)
            self.sensor_frame_name = inferred_sensor_frame_name
        self.gt_tum_filename = str(
            rospy.get_param("~gt_tum_filename", "gt_sensor_world_tum.txt")
        ).strip()
        self.odom_tum_filename = str(
            rospy.get_param("~odom_tum_filename", "odom_raw_tum.txt")
        ).strip()
        self.deform_monitor_param_root = str(
            rospy.get_param("~deform_monitor_param_root", "/deform_monitor_v2")
        ).strip() or "/deform_monitor_v2"
        self.deform_monitor_config_path = str(
            rospy.get_param("~deform_monitor_config_path", "")
        ).strip()
        self.scenario_id = str(rospy.get_param("~scenario_id", "")).strip()
        self.controlled_object = str(rospy.get_param("~controlled_object", "")).strip()
        self.command_frame = str(rospy.get_param("~command_frame", "")).strip()
        self.linear_velocity = vector3_dict(
            coerce_float(rospy.get_param("~linear_velocity_x", 0.0), 0.0),
            coerce_float(rospy.get_param("~linear_velocity_y", 0.0), 0.0),
            coerce_float(rospy.get_param("~linear_velocity_z", 0.0), 0.0),
        )
        self.angular_velocity_deg = vector3_dict(
            coerce_float(rospy.get_param("~angular_velocity_x_deg", 0.0), 0.0),
            coerce_float(rospy.get_param("~angular_velocity_y_deg", 0.0), 0.0),
            coerce_float(rospy.get_param("~angular_velocity_z_deg", 0.0), 0.0),
        )
        self.control_axis = vector3_dict(
            coerce_float(rospy.get_param("~control_axis_x", 0.0), 0.0),
            coerce_float(rospy.get_param("~control_axis_y", 0.0), 0.0),
            coerce_float(rospy.get_param("~control_axis_z", 0.0), 0.0),
        )
        self.control_start_delay_sec = coerce_float(
            rospy.get_param("~control_start_delay_sec", None), None
        )
        self.control_duration_sec = coerce_float(
            rospy.get_param("~control_duration_sec", None), None
        )

        self.run_dir = self._create_run_directory()
        self.meta_dir = self.run_dir / "meta"
        self.truth_dir = self.run_dir / "truth"
        self.truth_objects_dir = self.truth_dir / "objects"
        self.truth_links_dir = self.truth_dir / "links"
        self.algorithm_dir = self.run_dir / "algorithm"
        self.trajectory_dir = self.run_dir / "trajectory"
        self._gt_tum_path = self.trajectory_dir / self.gt_tum_filename
        self._odom_tum_path = self.trajectory_dir / self.odom_tum_filename
        self._latest_sensor_pose_world = None
        self._latest_sensor_pose_stamp = None
        self._latest_truth_reference_pose_world = None
        self._latest_truth_reference_pose_stamp = None
        self._latest_truth_reference_frame = ""
        self._sensor_relative_pose_cache = {}
        listener_factory = getattr(tf, "TransformListener", None)
        self._tf_listener = listener_factory() if callable(listener_factory) else None

        self._object_files = {}
        self._link_files = {}
        self._algorithm_files = {}
        self._ego_initial_pose_written = False
        self._frame_alignment_written = False
        self._persistent_track_cache = {}
        # Tracks how many consecutive cluster frames each anchor_id has appeared in.
        # Resets to zero when an anchor disappears from all clusters for one frame.
        self._anchor_cluster_consecutive = {}
        # Cache of the latest serialized cluster payload, used to attach a
        # displacement estimate to first_confirmed persistent track events.
        self._latest_cluster_payload = None
        # Pending displacement-window entries: list of dicts, each describing
        # a first_confirmed event that still needs its post-detection frames
        # filled in before the window event is flushed to disk.
        # Schema per entry:
        #   track_id      : int
        #   region_center : dict {x,y,z} in algo frame
        #   confirmed_at  : float  (ROS time of first_confirmed, seconds)
        #   pre_frames    : list of {t_offset, recorded_at_sec, clusters_payload}
        #                   (populated retroactively from cluster history)
        #   post_frames   : list of {t_offset, recorded_at_sec, clusters_payload}
        #                   (populated from subsequent cluster callbacks)
        #   window_half   : int  (number of frames each side; default 3)
        self._disp_window_pending = []
        # Ring-buffer of recent cluster payloads for pre-detection back-fill.
        # Keeps the last WINDOW_HALF cluster messages so that when first_confirmed
        # fires we can immediately populate the negative-offset slots.
        self._DISP_WINDOW_HALF = 3
        self._cluster_history = []   # list of payload dicts, capped at WINDOW_HALF

        # Truth CSV throttling: write at most 10 Hz to reduce disk usage.
        self._TRUTH_WRITE_INTERVAL = 0.10  # seconds → 10 Hz
        self._last_model_states_write_time = 0.0
        self._last_link_states_write_time = 0.0

        self._ensure_directories()
        self._publish_runtime_output_dir_param()
        self._write_run_info()
        self._write_run_metadata()

        rospy.on_shutdown(self.close)
        self._subscribers = [
            rospy.Subscriber(
                self.model_states_topic,
                ModelStates,
                self._handle_model_states,
                queue_size=1,
            ),
            rospy.Subscriber(
                self.link_states_topic,
                LinkStates,
                self._handle_link_states,
                queue_size=1,
            ),
            rospy.Subscriber(
                self.ground_truth_odometry_topic,
                Odometry,
                self._handle_ground_truth_odometry,
                queue_size=10,
            ),
            rospy.Subscriber(
                self.clusters_topic,
                MotionClusters,
                self._handle_clusters,
                queue_size=10,
            ),
            rospy.Subscriber(
                self.odometry_topic,
                Odometry,
                self._handle_odometry,
                queue_size=1,
            ),
            rospy.Subscriber(
                self.risk_evidence_topic,
                RiskEvidenceArray,
                self._handle_risk_evidence,
                queue_size=10,
            ),
            rospy.Subscriber(
                self.risk_regions_topic,
                RiskRegions,
                self._handle_risk_regions,
                queue_size=10,
            ),
            rospy.Subscriber(
                self.persistent_risk_regions_topic,
                PersistentRiskRegions,
                self._handle_persistent_risk_regions,
                queue_size=10,
            ),
            rospy.Subscriber(
                self.structure_motions_topic,
                StructureMotions,
                self._handle_structure_motions,
                queue_size=10,
            ),
            rospy.Subscriber(
                self.anchor_states_topic,
                AnchorStates,
                self._handle_anchor_states,
                queue_size=10,
            ),
        ]

        rospy.loginfo("Sim experiment recorder writing to %s", self.run_dir)

    def _create_run_directory(self):
        day = dt.datetime.now().strftime("%Y%m%d")
        day_dir = self.output_root / day
        day_dir.mkdir(parents=True, exist_ok=True)
        run_dir = allocate_run_directory(day_dir)
        run_dir.mkdir(parents=True, exist_ok=False)
        return run_dir

    def _ensure_directories(self):
        self.meta_dir.mkdir(parents=True, exist_ok=True)
        self.truth_dir.mkdir(parents=True, exist_ok=True)
        self.truth_objects_dir.mkdir(parents=True, exist_ok=True)
        self.truth_links_dir.mkdir(parents=True, exist_ok=True)
        self.algorithm_dir.mkdir(parents=True, exist_ok=True)
        self.trajectory_dir.mkdir(parents=True, exist_ok=True)

    def _write_json(self, path, payload):
        with path.open("w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")

    def _write_json_if_changed(self, path, payload):
        serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        try:
            current = path.read_text()
        except FileNotFoundError:
            current = None
        except OSError:
            current = None

        if current == serialized:
            return False

        with path.open("w") as handle:
            handle.write(serialized)
        return True

    def _publish_runtime_output_dir_param(self):
        set_param = getattr(rospy, "set_param", None)
        if not callable(set_param):
            return
        set_param("/deform_monitor/runtime_output_dir", str(self.run_dir / "runtime"))

    def _write_run_info(self):
        payload = build_run_info_payload(
            run_dir=self.run_dir,
            truth_frame=self.truth_frame,
            algorithm_frame=self.algorithm_frame,
            ego_model_name=self.ego_model_name,
            model_states_topic=self.model_states_topic,
            link_states_topic=self.link_states_topic,
            clusters_topic=getattr(self, "clusters_topic", "/deform/clusters"),
            risk_evidence_topic=self.risk_evidence_topic,
            risk_regions_topic=self.risk_regions_topic,
            persistent_risk_regions_topic=self.persistent_risk_regions_topic,
            structure_motions_topic=self.structure_motions_topic,
            odometry_topic=self.odometry_topic,
            sensor_scoped_link_name=self.sensor_scoped_link_name,
            gt_tum_filename=self.gt_tum_filename,
            odom_tum_filename=self.odom_tum_filename,
            ground_truth_odometry_topic=self.ground_truth_odometry_topic,
            sensor_frame_name=self.sensor_frame_name,
        )
        self._write_json(self.meta_dir / "run_info.json", payload)

    def _read_node_parameter_tree(self):
        try:
            parameter_tree = rospy.get_param(self.deform_monitor_param_root, {})
        except Exception:
            parameter_tree = {}
        return parameter_tree if isinstance(parameter_tree, dict) else {}

    def _build_current_scenario_manifest_payload(self):
        explicit_control = build_explicit_control_metadata(
            controlled_object=getattr(self, "controlled_object", ""),
            command_frame=getattr(self, "command_frame", ""),
            linear_velocity=getattr(self, "linear_velocity", None),
            angular_velocity_deg=getattr(self, "angular_velocity_deg", None),
            axis=getattr(self, "control_axis", None),
            start_delay_sec=getattr(self, "control_start_delay_sec", None),
            duration_sec=getattr(self, "control_duration_sec", None),
            scenario_id=getattr(self, "scenario_id", ""),
        )
        discovered_controls = discover_controlled_objects(
            get_param=rospy.get_param,
            get_param_names=getattr(rospy, "get_param_names", lambda: []),
        )
        authoritative_controls = select_authoritative_discovered_controls(
            getattr(self, "scenario_id", ""),
            discovered_controls,
        )
        return build_scenario_manifest_payload(
            run_dir=self.run_dir,
            scenario_id=getattr(self, "scenario_id", ""),
            explicit_control=explicit_control,
            discovered_controls=authoritative_controls,
        )

    def _refresh_scenario_manifest_if_needed(self):
        manifest_path = self.meta_dir / "scenario_manifest.json"
        current_payload = self._build_current_scenario_manifest_payload()
        current_source = str(current_payload.get("source", ""))
        try:
            existing_payload = json.loads(manifest_path.read_text())
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            existing_payload = None

        if isinstance(existing_payload, dict):
            existing_source = str(existing_payload.get("source", ""))
            if existing_source == "discovered" and current_source != "discovered":
                return False
            current_without_timestamp = dict(current_payload)
            current_without_timestamp.pop("created_at_iso", None)
            existing_without_timestamp = dict(existing_payload)
            existing_without_timestamp.pop("created_at_iso", None)
            if existing_without_timestamp == current_without_timestamp:
                return False

        return self._write_json_if_changed(manifest_path, current_payload)

    def _ensure_algorithm_runtime_state(self):
        if not hasattr(self, "_anchor_cluster_consecutive"):
            self._anchor_cluster_consecutive = {}
        if not hasattr(self, "_latest_cluster_payload"):
            self._latest_cluster_payload = None
        if not hasattr(self, "_DISP_WINDOW_HALF"):
            self._DISP_WINDOW_HALF = 3
        if not hasattr(self, "_cluster_history"):
            self._cluster_history = []
        if not hasattr(self, "_disp_window_pending"):
            self._disp_window_pending = []

    def _write_run_metadata(self):
        parameter_tree = self._read_node_parameter_tree()
        self._write_json(
            self.meta_dir / "ablation_manifest.json",
            build_ablation_manifest_payload(
                run_dir=self.run_dir,
                node_param_root=self.deform_monitor_param_root,
                source_config_path=self.deform_monitor_config_path,
                parameter_tree=parameter_tree,
            ),
        )
        self._write_json(
            self.meta_dir / "config_snapshot.json",
            build_config_snapshot_payload(
                run_dir=self.run_dir,
                node_param_root=self.deform_monitor_param_root,
                source_config_path=self.deform_monitor_config_path,
                parameter_tree=parameter_tree,
            ),
        )
        self._refresh_scenario_manifest_if_needed()

    def _append_persistent_track_events(self, payload):
        track_cache = getattr(self, "_persistent_track_cache", None)
        if track_cache is None:
            track_cache = {}
            self._persistent_track_cache = track_cache

        header = payload.get("header", {}) if isinstance(payload, dict) else {}
        stamp = copy_time_dict(header.get("stamp")) or time_to_dict(rospy.Time.now())
        recorded_at = time_to_dict(rospy.Time.now())
        for region in payload.get("regions", []):
            track_id = int(region.get("track_id", 0))
            state = int(region.get("state", 0))
            confirmed = bool(region.get("confirmed", False))
            previous = track_cache.get(track_id)
            lifecycle = {
                "first_seen": copy_time_dict(stamp),
                "first_confirmed": copy_time_dict(stamp) if confirmed else None,
                "last_seen": copy_time_dict(stamp),
            }
            if previous is not None:
                lifecycle["first_seen"] = copy_time_dict(previous.get("first_seen")) or copy_time_dict(stamp)
                lifecycle["first_confirmed"] = copy_time_dict(previous.get("first_confirmed"))
                lifecycle["last_seen"] = copy_time_dict(stamp)
                if confirmed and lifecycle["first_confirmed"] is None:
                    lifecycle["first_confirmed"] = copy_time_dict(stamp)

            base_payload = {
                "track_id": track_id,
                "header": header,
                "stamp": copy_time_dict(stamp),
                "recorded_at": recorded_at,
                "state": state,
                "state_name": PERSISTENT_STATE_NAMES.get(state, "UNKNOWN"),
                "confirmed": confirmed,
                "region_type": int(region.get("region_type", 0)),
                "region_type_name": REGION_TYPE_NAMES.get(int(region.get("region_type", 0)), "UNKNOWN"),
                "center": region.get("center"),
                "bbox_min": region.get("bbox_min"),
                "bbox_max": region.get("bbox_max"),
                "mean_risk": float(region.get("mean_risk", 0.0)),
                "peak_risk": float(region.get("peak_risk", 0.0)),
                "confidence": float(region.get("confidence", 0.0)),
                "accumulated_risk": float(region.get("accumulated_risk", 0.0)),
                "support_mass": float(region.get("support_mass", 0.0)),
                "spatial_span": float(region.get("spatial_span", 0.0)),
                "hit_streak": int(region.get("hit_streak", 0)),
                "miss_streak": int(region.get("miss_streak", 0)),
                "age_frames": int(region.get("age_frames", 0)),
                "lifecycle": lifecycle,
            }

            if previous is None:
                created_payload = dict(base_payload)
                created_payload["event_type"] = "track_created"
                self._append_jsonl(
                    "persistent_track_events",
                    "persistent_track_events.jsonl",
                    created_payload,
                )

            if previous is not None and int(previous.get("state", state)) != state:
                transition_payload = dict(base_payload)
                transition_payload["event_type"] = "state_transition"
                transition_payload["from_state"] = int(previous.get("state", state))
                transition_payload["from_state_name"] = PERSISTENT_STATE_NAMES.get(
                    int(previous.get("state", state)), "UNKNOWN"
                )
                transition_payload["to_state"] = state
                transition_payload["to_state_name"] = base_payload["state_name"]
                self._append_jsonl(
                    "persistent_track_events",
                    "persistent_track_events.jsonl",
                    transition_payload,
                )

            if confirmed and (
                previous is None or previous.get("first_confirmed") is None
            ):
                confirmed_payload = dict(base_payload)
                confirmed_payload["event_type"] = "first_confirmed"
                # Attach the best matching cluster displacement estimate from the
                # current frame.  This enables epsilon_d computation without a
                # cross-join between persistent_track_events.jsonl and clusters.jsonl.
                disp_est = self._find_confirmed_displacement_estimate(
                    region_center=region.get("center"),
                    latest_cluster_payload=getattr(self, "_latest_cluster_payload", None),
                )
                if disp_est is not None:
                    confirmed_payload["confirmed_displacement_estimate"] = disp_est
                self._append_jsonl(
                    "persistent_track_events",
                    "persistent_track_events.jsonl",
                    confirmed_payload,
                )
                # Register a pending displacement-window entry.
                # Pre-frames are back-filled from cluster history (already collected).
                confirmed_at_sec = common_record_time_sec_from_payload(confirmed_payload)
                pre_frames = []
                history = list(self._cluster_history)  # snapshot (oldest first)
                for offset_idx, hist_payload in enumerate(history):
                    # offset relative to first_confirmed: -HALF, ..., -1
                    t_offset = offset_idx - len(history)
                    pre_frames.append({
                        "t_offset": t_offset,
                        "clusters_payload": hist_payload,
                    })
                self._disp_window_pending.append({
                    "track_id":      region.get("track_id"),
                    "region_center": region.get("center"),
                    "confirmed_at":  confirmed_at_sec,
                    "pre_frames":    pre_frames,
                    "post_frames":   [],
                    "window_half":   self._DISP_WINDOW_HALF,
                })

            frame_payload = dict(base_payload)
            frame_payload["event_type"] = "frame_status"
            # Skip unconfirmed frame_status events — they are not used by any
            # metric analysis and are the largest contributor to file size.
            if confirmed:
                self._append_jsonl(
                    "persistent_track_events",
                    "persistent_track_events.jsonl",
                    frame_payload,
                )

            track_cache[track_id] = {
                "state": state,
                "first_seen": lifecycle["first_seen"],
                "first_confirmed": lifecycle["first_confirmed"],
                "last_seen": lifecycle["last_seen"],
            }

    def _append_jsonl(self, key, filename, payload):
        handle = self._algorithm_files.get(key)
        if handle is None:
            handle = (self.algorithm_dir / filename).open("a")
            self._algorithm_files[key] = handle

        json.dump(payload, handle, sort_keys=True)
        handle.write("\n")
        handle.flush()

    def _alignment_ready(self):
        return self._frame_alignment_written

    def _warn_alignment_pending(self):
        rospy.logwarn_throttle(
            5.0,
            "sim_experiment_recorder is waiting for ego pose '%s' before recording algorithm outputs.",
            self.ego_model_name,
        )

    def _tracked_model_names(self, msg):
        tracked = []
        for name in getattr(msg, "name", []):
            if name in (self.ego_model_name, "ground_plane"):
                continue
            tracked.append(name)
        return tracked

    def _tracked_link_names(self, msg):
        tracked = []
        for scoped_name in getattr(msg, "name", []):
            model_name, _ = parse_scoped_link_name(scoped_name)
            if model_name in (self.ego_model_name, "ground_plane"):
                continue
            tracked.append(scoped_name)
        return tracked

    def _write_ego_initial_pose(self, pose):
        if self._ego_initial_pose_written:
            return

        csv_path = self.truth_dir / "ego_initial_pose_world.csv"
        with csv_path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(EGO_INITIAL_POSE_HEADER)
            writer.writerow(
                [
                    self.truth_frame,
                    pose.position.x,
                    pose.position.y,
                    pose.position.z,
                    pose.orientation.x,
                    pose.orientation.y,
                    pose.orientation.z,
                    pose.orientation.w,
                ]
            )
        self._ego_initial_pose_written = True

    def _object_writer(self, model_name):
        key = str(model_name)
        writer_entry = self._object_files.get(key)
        if writer_entry is not None:
            return writer_entry

        csv_path = self.truth_objects_dir / (sanitize_name(model_name) + ".csv")
        file_exists = csv_path.exists()
        handle = csv_path.open("a", newline="")
        writer = csv.writer(handle)
        if not file_exists:
            writer.writerow(TRUTH_OBJECT_HEADER)
            handle.flush()
        self._object_files[key] = (handle, writer)
        return self._object_files[key]

    def _link_writer(self, scoped_link_name):
        key = str(scoped_link_name)
        writer_entry = self._link_files.get(key)
        if writer_entry is not None:
            return writer_entry

        csv_path = self.truth_links_dir / (sanitize_name(scoped_link_name) + ".csv")
        file_exists = csv_path.exists()
        handle = csv_path.open("a", newline="")
        writer = csv.writer(handle)
        if not file_exists:
            writer.writerow(TRUTH_LINK_HEADER)
            handle.flush()
        self._link_files[key] = (handle, writer)
        return self._link_files[key]

    def _handle_model_states(self, msg):
        recorded_time_sec = rospy.Time.now().to_sec()
        self._refresh_scenario_manifest_if_needed()

        poses_by_name = dict(zip(getattr(msg, "name", []), getattr(msg, "pose", [])))

        ego_pose = poses_by_name.get(self.ego_model_name)
        if ego_pose is not None:
            self._write_ego_initial_pose(ego_pose)

        # Throttle truth/objects CSV writes to 10 Hz.
        if recorded_time_sec - self._last_model_states_write_time < self._TRUTH_WRITE_INTERVAL:
            return
        self._last_model_states_write_time = recorded_time_sec

        for model_name in self._tracked_model_names(msg):
            pose = poses_by_name.get(model_name)
            if pose is None:
                continue

            handle, writer = self._object_writer(model_name)
            writer.writerow(
                [
                    "%.9f" % float(recorded_time_sec),
                    model_name,
                    self.truth_frame,
                    pose.position.x,
                    pose.position.y,
                    pose.position.z,
                    pose.orientation.x,
                    pose.orientation.y,
                    pose.orientation.z,
                    pose.orientation.w,
                ]
            )
            handle.flush()

    def _handle_link_states(self, msg):
        recorded_time_sec = rospy.Time.now().to_sec()
        poses_by_name = dict(zip(getattr(msg, "name", []), getattr(msg, "pose", [])))

        sensor_pose = poses_by_name.get(self.sensor_scoped_link_name)
        if sensor_pose is not None:
            self._latest_sensor_pose_world = pose_to_dict(sensor_pose)
            self._latest_sensor_pose_stamp = recorded_time_sec

        # Throttle truth/links CSV writes to 10 Hz.
        if recorded_time_sec - self._last_link_states_write_time < self._TRUTH_WRITE_INTERVAL:
            return
        self._last_link_states_write_time = recorded_time_sec

        for scoped_link_name in self._tracked_link_names(msg):
            pose = poses_by_name.get(scoped_link_name)
            if pose is None:
                continue

            model_name, link_name = parse_scoped_link_name(scoped_link_name)
            handle, writer = self._link_writer(scoped_link_name)
            writer.writerow(
                [
                    "%.9f" % float(recorded_time_sec),
                    scoped_link_name,
                    model_name,
                    link_name,
                    self.truth_frame,
                    pose.position.x,
                    pose.position.y,
                    pose.position.z,
                    pose.orientation.x,
                    pose.orientation.y,
                    pose.orientation.z,
                    pose.orientation.w,
                ]
            )
            handle.flush()

    def _tf_lookup_time(self):
        time_ctor = getattr(rospy, "Time", None)
        if callable(time_ctor):
            try:
                return time_ctor(0)
            except TypeError:
                return None
        return None

    def _lookup_sensor_relative_pose(self, base_frame_id):
        base_frame_id = str(base_frame_id).strip()
        if not base_frame_id or not self.sensor_frame_name:
            return None

        cache = getattr(self, "_sensor_relative_pose_cache", None)
        if cache is None:
            cache = {}
            self._sensor_relative_pose_cache = cache

        cached_pose = cache.get(base_frame_id)
        if cached_pose is not None:
            return cached_pose

        if self._tf_listener is None:
            return None

        translation, rotation = self._tf_listener.lookupTransform(
            base_frame_id,
            self.sensor_frame_name,
            self._tf_lookup_time(),
        )
        relative_pose = {
            "position": {
                "x": float(translation[0]),
                "y": float(translation[1]),
                "z": float(translation[2]),
            },
            "orientation": {
                "x": float(rotation[0]),
                "y": float(rotation[1]),
                "z": float(rotation[2]),
                "w": float(rotation[3]),
            },
        }
        if pose_dict_is_finite(relative_pose):
            cache[base_frame_id] = relative_pose
            return relative_pose
        return None

    def _handle_ground_truth_odometry(self, msg):
        base_frame_id = str(getattr(msg, "child_frame_id", "")).strip()
        if not base_frame_id:
            return

        base_pose_world = pose_to_dict(getattr(getattr(msg, "pose", None), "pose", None))
        if not pose_dict_is_finite(base_pose_world):
            return

        self._latest_truth_reference_pose_world = base_pose_world
        self._latest_truth_reference_frame = base_frame_id

        try:
            relative_sensor_pose = self._lookup_sensor_relative_pose(base_frame_id)
        except Exception as exc:
            rospy.logwarn_throttle(
                5.0,
                "sim_experiment_recorder failed to look up sensor frame '%s' from base frame '%s': %s",
                self.sensor_frame_name,
                base_frame_id,
                exc,
            )
            return

        if relative_sensor_pose is None:
            return

        sensor_pose_world = compose_pose_dicts(base_pose_world, relative_sensor_pose)
        if not pose_dict_is_finite(sensor_pose_world):
            return

        stamp = time_to_dict(getattr(getattr(msg, "header", None), "stamp", None))
        self._latest_sensor_pose_world = sensor_pose_world
        if isinstance(stamp, dict) and "sec" in stamp:
            self._latest_sensor_pose_stamp = stamp["sec"]
            self._latest_truth_reference_pose_stamp = stamp["sec"]
        else:
            self._latest_sensor_pose_stamp = rospy.Time.now().to_sec()
            self._latest_truth_reference_pose_stamp = self._latest_sensor_pose_stamp

    def _try_write_frame_alignment_from_pose_pair(self, odom_pose, odom_child_frame_id, odom_stamp_sec):
        if getattr(self, "_frame_alignment_written", False):
            return True

        truth_reference_pose_world = getattr(self, "_latest_truth_reference_pose_world", None)
        truth_reference_pose_stamp = getattr(self, "_latest_truth_reference_pose_stamp", None)
        if truth_reference_pose_world is None or truth_reference_pose_stamp is None:
            return False

        try:
            truth_reference_pose_stamp_sec = float(truth_reference_pose_stamp)
            odom_stamp_sec = float(odom_stamp_sec)
        except (TypeError, ValueError):
            return False

        if abs(odom_stamp_sec - truth_reference_pose_stamp_sec) > SENSOR_POSE_MAX_AGE_SEC:
            return False

        metadata = build_frame_alignment_metadata(
            ego_pose_world=truth_reference_pose_world,
            truth_frame=getattr(self, "truth_frame", "world"),
            algorithm_frame=getattr(self, "algorithm_frame", "camera_init"),
            truth_reference_frame=getattr(self, "_latest_truth_reference_frame", ""),
            truth_reference_pose_world=truth_reference_pose_world,
            algorithm_reference_frame=odom_child_frame_id,
            algorithm_reference_pose_algorithm=odom_pose,
        )
        self._write_json(getattr(self, "meta_dir", pathlib.Path(".")) / "frame_alignment.json", metadata)
        self._frame_alignment_written = True
        return True

    def _handle_odometry(self, msg):
        stamp = time_to_dict(getattr(getattr(msg, "header", None), "stamp", None))
        if not isinstance(stamp, dict) or not stamp.get("sec"):
            rospy.logwarn_throttle(
                5.0,
                "sim_experiment_recorder skipped odometry export because the message stamp was invalid.",
            )
            return

        odom_pose_msg = getattr(getattr(msg, "pose", None), "pose", None)
        odom_pose = pose_to_dict(odom_pose_msg)
        if not pose_dict_is_finite(odom_pose):
            rospy.logwarn_throttle(
                5.0,
                "sim_experiment_recorder skipped odometry export because the odometry pose is invalid.",
            )
            return

        self._try_write_frame_alignment_from_pose_pair(
            odom_pose=odom_pose,
            odom_child_frame_id=str(getattr(msg, "child_frame_id", "")).strip(),
            odom_stamp_sec=stamp["sec"],
        )

        sensor_pose_world = self._latest_sensor_pose_world
        if sensor_pose_world is None:
            rospy.logwarn_throttle(
                5.0,
                "sim_experiment_recorder is waiting for a valid sensor pose cache before writing trajectory exports.",
            )
            return

        if not pose_dict_is_finite(sensor_pose_world):
            rospy.logwarn_throttle(
                5.0,
                "sim_experiment_recorder skipped odometry export because the cached sensor pose is invalid.",
            )
            return

        sensor_pose_stamp = self._latest_sensor_pose_stamp
        if sensor_pose_stamp is None:
            rospy.logwarn_throttle(
                5.0,
                "sim_experiment_recorder skipped odometry export because the cached sensor pose timestamp is missing.",
            )
            return

        try:
            odom_stamp_sec = float(stamp["sec"])
            sensor_pose_stamp_sec = float(sensor_pose_stamp)
        except (TypeError, ValueError):
            rospy.logwarn_throttle(
                5.0,
                "sim_experiment_recorder skipped odometry export because the cached sensor pose timestamp is invalid.",
            )
            return

        if abs(odom_stamp_sec - sensor_pose_stamp_sec) > SENSOR_POSE_MAX_AGE_SEC:
            rospy.logwarn_throttle(
                5.0,
                "sim_experiment_recorder skipped odometry export because the cached sensor pose timestamp is stale by more than %.3f s.",
                SENSOR_POSE_MAX_AGE_SEC,
            )
            return

        write_tum_sample_pair(
            gt_path=self._gt_tum_path,
            odom_path=self._odom_tum_path,
            timestamp_sec=stamp["sec"],
            sensor_pose_world=sensor_pose_world,
            odom_pose=odom_pose,
        )

    def _handle_risk_evidence(self, msg):
        if not self._alignment_ready():
            self._warn_alignment_pending()
            return
        payload = serialize_risk_evidence(msg)
        payload["recorded_at"] = time_to_dict(rospy.Time.now())
        self._append_jsonl("risk_evidence", "risk_evidence.jsonl", payload)

    def _extract_best_cluster_disp(self, region_center, clusters_payload):
        """
        Extract the displacement magnitude from the best-matching significant
        cluster in *clusters_payload* relative to *region_center*.

        Returns a dict {disp_norm_m, disp_mean, cluster_support_count,
        match_dist_m} or None if no suitable cluster is found.
        Reuses the same matching logic as _find_confirmed_displacement_estimate.
        """
        return self._find_confirmed_displacement_estimate(
            region_center=region_center,
            latest_cluster_payload=clusters_payload,
        )

    def _advance_disp_window_pending(self, new_cluster_payload):
        """
        Called on every new cluster payload.  For each pending window entry,
        append the new payload as the next post-detection frame.  When the
        window is complete (post_frames == window_half), flush the window event
        to persistent_track_events.jsonl and remove the entry.
        """
        still_pending = []
        for entry in self._disp_window_pending:
            half = entry["window_half"]
            if len(entry["post_frames"]) < half:
                t_offset = len(entry["post_frames"]) + 1  # +1, +2, +3
                entry["post_frames"].append({
                    "t_offset": t_offset,
                    "clusters_payload": new_cluster_payload,
                })

            if len(entry["post_frames"]) >= half:
                # Window complete — build and flush the event
                self._flush_disp_window(entry)
            else:
                still_pending.append(entry)
        self._disp_window_pending = still_pending

    def _flush_disp_window(self, entry):
        """
        Serialize the displacement window for one first_confirmed event and
        write a 'displacement_window' event to persistent_track_events.jsonl.

        The window contains cluster displacement estimates at t_offset =
        -HALF, ..., -1, 0 (first_confirmed), +1, ..., +HALF relative to the
        first_confirmed frame, extracted from the cached cluster payloads.
        Offsets are relative frame counts, NOT absolute times.

        The emitted event has event_type='displacement_window' and carries:
          track_id, confirmed_at, window_half,
          frames: list of {t_offset, disp_estimate}   (None if no cluster match)
        """
        region_center = entry["region_center"]
        frames_out = []

        # Pre-detection frames (t_offset < 0)
        for slot in entry["pre_frames"]:
            est = self._extract_best_cluster_disp(region_center, slot["clusters_payload"])
            frames_out.append({
                "t_offset": slot["t_offset"],
                "disp_estimate": est,
            })

        # t_offset = 0 (the first_confirmed frame itself, from latest cache at that moment)
        # The cluster payload at first_confirmed is the last element of pre_frames
        # if the history was full, or we re-use _latest_cluster_payload captured earlier.
        # We approximate it by taking the cluster payload that immediately preceded
        # first_confirmed (t_offset = -1 pre_frame + 1 post_frame gap is already covered).
        # For simplicity use the first post_frame's payload as t=0 proxy when pre is empty,
        # or the last pre_frame otherwise — but to avoid double-counting we mark t=0
        # explicitly using confirmed_displacement_estimate written in first_confirmed event.
        frames_out.append({
            "t_offset": 0,
            "disp_estimate": None,  # already captured in first_confirmed event
            "note": "see confirmed_displacement_estimate in first_confirmed event",
        })

        # Post-detection frames (t_offset > 0)
        for slot in entry["post_frames"]:
            est = self._extract_best_cluster_disp(region_center, slot["clusters_payload"])
            frames_out.append({
                "t_offset": slot["t_offset"],
                "disp_estimate": est,
            })

        window_event = {
            "event_type":   "displacement_window",
            "track_id":     entry["track_id"],
            "confirmed_at": entry["confirmed_at"],
            "window_half":  entry["window_half"],
            "frames":       frames_out,
            "recorded_at":  time_to_dict(rospy.Time.now()),
        }
        self._append_jsonl(
            "persistent_track_events",
            "persistent_track_events.jsonl",
            window_event,
        )

    def _find_confirmed_displacement_estimate(self, region_center, latest_cluster_payload):
        """Find the best matching significant cluster displacement for a newly confirmed region.

        Used to attach a displacement estimate to first_confirmed events so that
        analysis scripts can compute epsilon_d without needing a separate cross-join
        between persistent_track_events.jsonl and clusters.jsonl.
        """
        if not latest_cluster_payload or not isinstance(region_center, dict):
            return None
        try:
            cx = float(region_center.get("x", 0.0))
            cy = float(region_center.get("y", 0.0))
            cz = float(region_center.get("z", 0.0))
        except (TypeError, ValueError):
            return None

        _MATCH_RADIUS = 0.8  # metres — same threshold as analysis scripts
        best_dist = float("inf")
        best_cluster = None

        for cluster in latest_cluster_payload.get("clusters", []):
            if not cluster.get("significant", False):
                continue
            if int(cluster.get("support_count", 0)) < 5:
                continue
            c = cluster.get("center", {})
            if not isinstance(c, dict):
                continue
            try:
                dx = float(c.get("x", 0.0)) - cx
                dy = float(c.get("y", 0.0)) - cy
                dz = float(c.get("z", 0.0)) - cz
            except (TypeError, ValueError):
                continue
            dist = math.sqrt(dx * dx + dy * dy + dz * dz)
            if dist < best_dist and dist <= _MATCH_RADIUS:
                best_dist = dist
                best_cluster = cluster

        if best_cluster is None:
            return None

        return {
            "disp_norm_m": float(best_cluster.get("disp_norm", 0.0)),
            "disp_mean": list(best_cluster.get("disp_mean", [])),
            "cluster_support_count": int(best_cluster.get("support_count", 0)),
            "cluster_min_consecutive_active_frames": int(
                best_cluster.get("min_anchor_consecutive_active_frames", 0)
            ),
            "match_dist_m": round(best_dist, 4),
        }

    def _handle_clusters(self, msg):
        if not self._alignment_ready():
            self._warn_alignment_pending()
            return
        self._ensure_algorithm_runtime_state()

        # Update per-anchor consecutive-frame counters.
        # An anchor "resets" to 0 if it does not appear in this frame's clusters.
        seen_anchor_ids = set()
        for cluster in getattr(msg, "clusters", []):
            for aid in getattr(cluster, "anchor_ids", []):
                seen_anchor_ids.add(int(aid))

        for aid in seen_anchor_ids:
            self._anchor_cluster_consecutive[aid] = (
                self._anchor_cluster_consecutive.get(aid, 0) + 1
            )
        stale = [aid for aid in self._anchor_cluster_consecutive if aid not in seen_anchor_ids]
        for aid in stale:
            del self._anchor_cluster_consecutive[aid]

        payload = serialize_motion_clusters(msg)

        # Enrich each cluster with min_anchor_consecutive_active_frames so that
        # downstream analysis can filter out "flickering" clusters.
        for cluster_dict in payload.get("clusters", []):
            anchor_ids = cluster_dict.get("anchor_ids", [])
            if anchor_ids:
                min_consec = min(
                    self._anchor_cluster_consecutive.get(int(aid), 1)
                    for aid in anchor_ids
                )
            else:
                min_consec = 0
            cluster_dict["min_anchor_consecutive_active_frames"] = min_consec

        payload["recorded_at"] = time_to_dict(rospy.Time.now())
        self._latest_cluster_payload = payload  # cache for first_confirmed events

        # Maintain rolling history for pre-detection back-fill (ring buffer)
        self._cluster_history.append(payload)
        if len(self._cluster_history) > self._DISP_WINDOW_HALF:
            self._cluster_history.pop(0)

        # Advance any pending displacement-window entries
        self._advance_disp_window_pending(payload)

        # Only write significant clusters to disk — non-significant clusters are
        # never read by any metric analysis (compute_metrics.py always filters on
        # significant=True). Internal state (consecutive counters, history,
        # latest_cluster_payload) is maintained on the full set above.
        write_payload = dict(payload)
        write_payload["clusters"] = [
            c for c in payload.get("clusters", []) if c.get("significant", False)
        ]
        self._append_jsonl("clusters", "clusters.jsonl", write_payload)

    def _handle_risk_regions(self, msg):
        if not self._alignment_ready():
            self._warn_alignment_pending()
            return
        payload = serialize_risk_regions(msg)
        payload["recorded_at"] = time_to_dict(rospy.Time.now())
        self._append_jsonl("risk_regions", "risk_regions.jsonl", payload)

    def _handle_persistent_risk_regions(self, msg):
        if not self._alignment_ready():
            self._warn_alignment_pending()
            return
        self._ensure_algorithm_runtime_state()
        payload = serialize_persistent_risk_regions(msg)
        payload["recorded_at"] = time_to_dict(rospy.Time.now())
        self._append_jsonl(
            "persistent_risk_regions",
            "persistent_risk_regions.jsonl",
            payload,
        )
        self._append_persistent_track_events(payload)

    def _handle_structure_motions(self, msg):
        if not self._alignment_ready():
            self._warn_alignment_pending()
            return
        payload = serialize_structure_motions(msg)
        payload["recorded_at"] = time_to_dict(rospy.Time.now())
        self._append_jsonl("structure_motions", "structure_motions.jsonl", payload)

    def _handle_anchor_states(self, msg):
        if not self._alignment_ready():
            self._warn_alignment_pending()
            return
        payload = serialize_anchor_states(msg)
        payload["recorded_at"] = time_to_dict(rospy.Time.now())
        self._append_jsonl("anchor_states", "anchor_states.jsonl", payload)

    def close(self):
        for handle, _ in self._object_files.values():
            handle.close()
        self._object_files = {}

        for handle, _ in self._link_files.values():
            handle.close()
        self._link_files = {}

        for handle in self._algorithm_files.values():
            handle.close()
        self._algorithm_files = {}


def main():
    if rospy is None:
        raise RuntimeError("rospy is required to run sim_experiment_recorder.py")

    rospy.init_node("sim_experiment_recorder")
    SimExperimentRecorder()
    rospy.spin()


if __name__ == "__main__":
    main()
