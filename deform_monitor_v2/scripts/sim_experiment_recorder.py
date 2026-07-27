#!/usr/bin/env python3

import csv
import datetime as dt
import importlib.util
import json
import math
import os
import pathlib
import platform
import queue
import re
import socket
import threading
import time
import xml.etree.ElementTree as ET
from typing import Any


try:
    from algorithm_frame_store import (
        AlgorithmFrameStoreError,
        AsyncCompressedFrameStore,
        DATABASE_FILENAME as ALGORITHM_FRAME_DATABASE_FILENAME,
        HIGH_VOLUME_STREAMS,
        inspect_sqlite_store,
        iter_sqlite_stream,
        normalize_storage_backend,
        trim_sqlite_stream,
    )
except ImportError:  # source-tree execution and pure-Python helper tests
    _storage_module_path = pathlib.Path(__file__).with_name("algorithm_frame_store.py")
    _storage_spec = importlib.util.spec_from_file_location(
        "deform_monitor_algorithm_frame_store", _storage_module_path
    )
    _storage_module = importlib.util.module_from_spec(_storage_spec)
    _storage_spec.loader.exec_module(_storage_module)
    AlgorithmFrameStoreError = _storage_module.AlgorithmFrameStoreError
    AsyncCompressedFrameStore = _storage_module.AsyncCompressedFrameStore
    ALGORITHM_FRAME_DATABASE_FILENAME = _storage_module.DATABASE_FILENAME
    HIGH_VOLUME_STREAMS = _storage_module.HIGH_VOLUME_STREAMS
    inspect_sqlite_store = _storage_module.inspect_sqlite_store
    iter_sqlite_stream = _storage_module.iter_sqlite_stream
    normalize_storage_backend = _storage_module.normalize_storage_backend
    trim_sqlite_stream = _storage_module.trim_sqlite_stream

try:
    import rospy
    import tf
    from gazebo_msgs.msg import LinkStates, ModelStates
    from nav_msgs.msg import Odometry
    from deform_monitor_v2.msg import (
        AnchorStates,
        MotionClusters,
        ObjectObservationStats,
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
    ObjectObservationStats = None
    PersistentRiskRegions = None
    RiskEvidenceArray = None
    RiskRegions = None
    StructureMotions = None


TRUTH_OBJECT_HEADER = [
    "recorded_time_sec",
    "model_name",
    "frame_id",
    "twist_frame_id",
    "position_x",
    "position_y",
    "position_z",
    "orientation_x",
    "orientation_y",
    "orientation_z",
    "orientation_w",
    "linear_velocity_x",
    "linear_velocity_y",
    "linear_velocity_z",
    "angular_velocity_x",
    "angular_velocity_y",
    "angular_velocity_z",
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


def recording_callback(callback):
    """Track callbacks so shutdown can drain work that already entered."""

    def wrapped(self, *args, **kwargs):
        condition = self._ensure_close_state()
        thread_id = threading.get_ident()
        with condition:
            if getattr(self, "_closing", False) or getattr(self, "_closed", False):
                return None
            self._active_callbacks += 1
            self._callback_thread_counts[thread_id] = (
                self._callback_thread_counts.get(thread_id, 0) + 1
            )
        try:
            return callback(self, *args, **kwargs)
        finally:
            should_finalize = False
            with condition:
                thread_depth = self._callback_thread_counts.get(thread_id, 0) - 1
                if thread_depth > 0:
                    self._callback_thread_counts[thread_id] = thread_depth
                else:
                    self._callback_thread_counts.pop(thread_id, None)
                self._active_callbacks -= 1
                if self._active_callbacks == 0:
                    if (
                        self._closing
                        and self._deferred_close
                        and not self._finish_close_started
                    ):
                        self._finish_close_started = True
                        self._closed = True
                        should_finalize = True
                    condition.notify_all()
            if should_finalize:
                self._run_close_finalization()

    wrapped.__name__ = callback.__name__
    wrapped.__doc__ = callback.__doc__
    return wrapped
SAFE_NAME_PATTERN = re.compile(r"[^A-Za-z0-9_.-]+")
DEFAULT_OUTPUT_ROOT = pathlib.Path.home() / ".ros" / "alert" / "output"
SENSOR_POSE_MAX_AGE_SEC = 0.25
FRAME_COMMIT_STAMP_TOLERANCE_SEC = 1.0e-6
FRAME_COMMIT_STREAM_FILES = {
    "anchor_observations": "anchor_observations.jsonl",
    "clusters": "clusters.jsonl",
    "object_observation_stats": "object_observation_stats.jsonl",
    "persistent_risk_regions": "persistent_risk_regions.jsonl",
    "processing_stamps": "processing_stamps.jsonl",
    "risk_evidence": "risk_evidence.jsonl",
    "risk_regions": "risk_regions.jsonl",
    "structure_motions": "structure_motions.jsonl",
}
PERSISTENT_STATE_NAMES = {
    0: "CANDIDATE",
    1: "CONFIRMED",
    2: "FADING",
}
COMPRESSED_ALGORITHM_STREAMS = frozenset(HIGH_VOLUME_STREAMS)


def normalize_algorithm_storage_backend(value):
    try:
        return normalize_storage_backend(value or "jsonl")
    except AlgorithmFrameStoreError as exc:
        raise ValueError(str(exc)) from exc


def frame_commit_stamp_key(stamp_sec):
    if stamp_sec is None:
        return None
    try:
        stamp_sec = float(stamp_sec)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(stamp_sec):
        return None
    return round(stamp_sec / FRAME_COMMIT_STAMP_TOLERANCE_SEC)


def build_frame_commit_plan(records_by_stream):
    stamp_sets = {}
    for stream_name in FRAME_COMMIT_STREAM_FILES:
        records = records_by_stream.get(stream_name, [])
        eligible = [
            record
            for record in records
            if stream_name != "object_observation_stats"
            or int(record.get("phase", -1)) == 1
        ]
        stamps = {
            record.get("stamp_key")
            for record in eligible
            if record.get("stamp_key") is not None
        }
        if stamps:
            stamp_sets[stream_name] = stamps

    if len(stamp_sets) != len(FRAME_COMMIT_STREAM_FILES):
        return None
    common_stamps = set.intersection(*stamp_sets.values())
    if not common_stamps:
        return None

    last_complete_key = max(common_stamps)
    trim_counts = {}
    for stream_name, records in records_by_stream.items():
        if stream_name not in FRAME_COMMIT_STREAM_FILES:
            continue
        keep_count = len(records)
        while keep_count > 0:
            stamp_key = records[keep_count - 1].get("stamp_key")
            if stamp_key is None or stamp_key <= last_complete_key:
                break
            keep_count -= 1
        if keep_count < len(records):
            trim_counts[stream_name] = len(records) - keep_count

    return {
        "last_complete_stamp_key": last_complete_key,
        "last_complete_stamp_sec": (
            last_complete_key * FRAME_COMMIT_STAMP_TOLERANCE_SEC
        ),
        "trim_counts": trim_counts,
    }
REGION_TYPE_NAMES = {
    0: "NONE",
    1: "DISPLACEMENT_LIKE",
    2: "DISAPPEARANCE_LIKE",
    3: "MIXED",
}

RECORDING_MODES = {"debug", "formal"}
REQUIRED_FORMAL_RUN_FACTORS = (
    "scene_id",
    "moving_object_quantity",
    "scene_object_quantity",
    "platform_condition",
    "slam_pipeline",
    "point_cloud_setting",
    "repeat_index",
)
REQUIRED_FORMAL_OBJECT_ATTRIBUTES = (
    "shape",
    "size_class",
    "motion_profile",
    "motion_direction",
    "visibility_condition",
)

ANCHOR_STATIC_FIELDS = (
    "anchor_type",
    "object_id",
    "object_id_valid",
    "object_id_confidence",
    "object_id_support_count",
    "center",
    "normal",
    "edge_normal",
    "point_count",
    "ref_quality",
    "covariance_quality",
    "type_stability",
    "shape_linearity",
    "shape_planarity",
    "shape_scattering",
    "ref_center",
    "reference_epoch",
    "reference_stamp",
    "reference_origin",
)
ANCHOR_PROCESSING_STOP = object()
TRUTH_WRITE_STOP = object()


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


def normalize_positive_rate_hz(value, default, name):
    raw_value = default if value in (None, "") else value
    parsed = coerce_float(raw_value, None)
    if parsed is None or parsed <= 0.0:
        raise ValueError(f"{name} must be a finite positive frequency in Hz")
    return parsed


def normalize_string_prefixes(value, default=("ground_truth_",)):
    if value in (None, ""):
        value = default
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple)):
        raise ValueError("surface_truth_link_prefixes must be a string or list")
    prefixes = tuple(str(item).strip() for item in value if str(item).strip())
    if not prefixes:
        raise ValueError("surface_truth_link_prefixes must contain at least one prefix")
    return prefixes


def sdf_pose_to_dict(text):
    values = [float(value) for value in str(text or "0 0 0 0 0 0").split()]
    if len(values) != 6 or not all(math.isfinite(value) for value in values):
        raise ValueError("SDF pose must contain six finite values")
    x, y, z, roll, pitch, yaw = values
    half_roll = 0.5 * roll
    half_pitch = 0.5 * pitch
    half_yaw = 0.5 * yaw
    cr, sr = math.cos(half_roll), math.sin(half_roll)
    cp, sp = math.cos(half_pitch), math.sin(half_pitch)
    cy, sy = math.cos(half_yaw), math.sin(half_yaw)
    return {
        "position": {"x": x, "y": y, "z": z},
        "orientation": {
            "x": sr * cp * cy - cr * sp * sy,
            "y": cr * sp * cy + sr * cp * sy,
            "z": cr * cp * sy - sr * sp * cy,
            "w": cr * cp * cy + sr * sp * sy,
        },
    }


def load_static_surface_truth_catalog(
    world_file,
    motion_truth_drive_links,
    object_id_catalog,
    expected_count=0,
    max_local_radius_m=3.0,
    require_clean_world=False,
):
    """Load rigid surface landmarks from parent-link marker visuals."""

    world_path = pathlib.Path(world_file).expanduser().resolve()
    if not world_path.is_file():
        raise ValueError(f"surface truth world file is missing: {world_path}")
    try:
        world = ET.parse(world_path).getroot().find("world")
    except (ET.ParseError, OSError) as exc:
        raise ValueError(f"invalid surface truth world file: {world_path}") from exc
    if world is None:
        raise ValueError(f"surface truth world has no <world>: {world_path}")
    if require_clean_world and world.find("state") is not None:
        raise ValueError("formal surface truth world must not contain a saved <state>")

    models = {
        str(model.get("name", "")).strip(): model for model in world.findall("model")
    }
    if require_clean_world:
        # Reject physical truth links only if they carry collision geometry.
        # Collision-free fixed child links (no <collision> child) are pose-only
        # markers that do not affect LiDAR ray casting or rigid-body dynamics.
        physical_truth_links = [
            f"{model.get('name', '')}::{link.get('name', '')}"
            for model in world.findall("model")
            for link in model.findall("link")
            if str(link.get("name", "")).startswith("ground_truth_")
            and link.find("collision") is not None
        ]
        physical_truth_joints = [
            f"{model.get('name', '')}::{joint.get('name', '')}"
            for model in world.findall("model")
            for joint in model.findall("joint")
            if str(joint.get("name", "")).startswith("ground_truth_")
            and joint.get("type", "") not in ("fixed", "")
        ]
        if physical_truth_links or physical_truth_joints:
            raise ValueError(
                "formal surface truth must use marker visuals, not physical truth links"
            )

    object_ids_by_name = {
        str(model_name): int(object_id)
        for object_id, model_name in object_id_catalog.items()
    }
    records = []
    scoped_names = set()
    for model_name, parent_scoped_name in sorted(motion_truth_drive_links.items()):
        model = models.get(model_name)
        if model is None:
            raise ValueError(f"surface truth model missing from world: {model_name}")
        _, parent_link_name = parse_scoped_link_name(parent_scoped_name)
        parent_link = model.find(f"link[@name='{parent_link_name}']")
        if parent_link is None:
            raise ValueError(
                f"surface truth parent link missing: {parent_scoped_name}"
            )
        marker_visuals = sorted(
            (
                visual
                for visual in parent_link.findall("visual")
                if str(visual.get("name", "")).startswith(
                    "ground_truth_marker_v_"
                )
            ),
            key=lambda visual: str(visual.get("name", "")),
        )
        # Fallback: if no marker visuals, accept collision-free physical child
        # links named ground_truth_v_* (fixed joints, no collision geometry).
        use_physical_links = not marker_visuals
        if use_physical_links:
            marker_visuals = sorted(
                (
                    link
                    for link in model.findall("link")
                    if str(link.get("name", "")).startswith("ground_truth_v_")
                    and link.find("collision") is None
                ),
                key=lambda link: str(link.get("name", "")),
            )
        if not marker_visuals:
            raise ValueError(f"surface truth marker visuals missing: {model_name}")
        for visual in marker_visuals:
            marker_name = str(visual.get("name", "")).strip()
            if use_physical_links:
                link_name = marker_name  # already named ground_truth_v_*
            else:
                link_name = marker_name.replace("ground_truth_marker_", "ground_truth_", 1)
            scoped_name = f"{model_name}::{link_name}"
            if scoped_name in scoped_names:
                raise ValueError(f"duplicate surface truth point: {scoped_name}")
            local_pose = sdf_pose_to_dict(visual.findtext("pose"))
            position = local_pose["position"]
            local_radius = math.sqrt(
                position["x"] ** 2 + position["y"] ** 2 + position["z"] ** 2
            )
            if local_radius > float(max_local_radius_m):
                raise ValueError(
                    f"surface truth point exceeds local radius: {scoped_name}:"
                    f"{local_radius:.9g}>{float(max_local_radius_m):.9g}"
                )
            object_id = object_ids_by_name.get(model_name, 0)
            records.append(
                {
                    "schema_version": 2,
                    "catalog_source": "world_static_marker_visual",
                    "source_world_file": str(world_path),
                    "scoped_link_name": scoped_name,
                    "model_name": model_name,
                    "link_name": link_name,
                    "object_id": object_id,
                    "object_id_valid": object_id > 0,
                    "truth_frame": "world",
                    "object_local_frame": parent_scoped_name,
                    "motion_parent_scoped_link_name": parent_scoped_name,
                    "local_pose": local_pose,
                }
            )
            scoped_names.add(scoped_name)

    if expected_count and len(records) != int(expected_count):
        raise ValueError(
            f"surface truth point count: {len(records)}!={int(expected_count)}"
        )
    return records


def normalize_model_link_mapping(value):
    if value in (None, ""):
        return {}
    if not isinstance(value, dict):
        raise ValueError("motion_truth_drive_links must be a mapping")
    result = {}
    scoped_names = set()
    for raw_model_name, raw_scoped_name in value.items():
        model_name = str(raw_model_name).strip()
        scoped_name = str(raw_scoped_name).strip()
        parsed_model_name, link_name = parse_scoped_link_name(scoped_name)
        if not model_name or not scoped_name or not link_name:
            raise ValueError("motion_truth_drive_links contains an empty entry")
        if parsed_model_name != model_name:
            raise ValueError(
                "motion truth drive link model mismatch: {} != {}".format(
                    parsed_model_name, model_name
                )
            )
        if scoped_name in scoped_names:
            raise ValueError("motion_truth_drive_links must contain unique links")
        result[model_name] = scoped_name
        scoped_names.add(scoped_name)
    return result


def normalize_truth_motion_policy(value):
    defaults = {
        "translation_deadband_m": 0.001,
        "rotation_deadband_deg": 0.05,
        "linear_speed_deadband_mps": 0.0005,
        "angular_speed_deadband_degps": 0.01,
        "sustained_motion_samples": 2,
    }
    if value in (None, ""):
        value = {}
    if not isinstance(value, dict):
        raise ValueError("truth_motion_policy must be a mapping")
    result = {}
    for name, default in defaults.items():
        raw_value = value.get(name, default)
        if name == "sustained_motion_samples":
            try:
                parsed = int(raw_value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name} must be a positive integer") from exc
            if parsed <= 0:
                raise ValueError(f"{name} must be a positive integer")
        else:
            parsed = coerce_float(raw_value, None)
            allow_zero = name in {
                "linear_speed_deadband_mps", "angular_speed_deadband_degps"
            }
            if parsed is None or parsed < 0.0 or (parsed == 0.0 and not allow_zero):
                qualifier = "non-negative" if allow_zero else "positive"
                raise ValueError(f"{name} must be finite and {qualifier}")
        result[name] = parsed
    return result


def should_sample_stream(recorded_time_sec, last_write_time, rate_hz):
    now = coerce_float(recorded_time_sec, None)
    if now is None:
        return False
    rate_hz = normalize_positive_rate_hz(rate_hz, rate_hz, "rate_hz")
    if last_write_time is None:
        return True
    previous = coerce_float(last_write_time, None)
    if previous is None or now < previous:
        return True
    return (now - previous) + 1.0e-12 >= (1.0 / rate_hz)


def advance_sampling_clock(
    recorded_time_sec, next_sample_time_sec, last_observed_time_sec, rate_hz
):
    """Advance a phase-preserving sampler and return (sample, next, observed)."""
    now = coerce_float(recorded_time_sec, None)
    if now is None:
        return False, next_sample_time_sec, last_observed_time_sec
    rate_hz = normalize_positive_rate_hz(rate_hz, rate_hz, "rate_hz")
    period = 1.0 / rate_hz
    previous_observed = coerce_float(last_observed_time_sec, None)
    deadline = coerce_float(next_sample_time_sec, None)
    if previous_observed is not None and now < previous_observed:
        return True, now + period, now
    if deadline is None:
        return True, now + period, now
    if now + 1.0e-12 < deadline:
        return False, deadline, now
    elapsed_periods = int(math.floor((now - deadline + 1.0e-12) / period)) + 1
    return True, deadline + elapsed_periods * period, now


def valid_sim_recording_time(recorded_time_sec):
    value = coerce_float(recorded_time_sec, None)
    return value is not None and value > 0.0


def new_sampling_stats(configured_rate_hz):
    return {
        "configured_rate_hz": float(configured_rate_hz),
        "received_message_count": 0,
        "sampled_message_count": 0,
        "rows_written": 0,
        "first_sample_time_sec": None,
        "last_sample_time_sec": None,
    }


def update_sampling_stats(stats, timestamp_sec, sampled, rows_written=0):
    stats["received_message_count"] = int(stats.get("received_message_count", 0)) + 1
    if not sampled:
        return stats
    timestamp_sec = float(timestamp_sec)
    stats["sampled_message_count"] = int(stats.get("sampled_message_count", 0)) + 1
    stats["rows_written"] = int(stats.get("rows_written", 0)) + int(rows_written)
    if stats.get("first_sample_time_sec") is None:
        stats["first_sample_time_sec"] = timestamp_sec
    stats["last_sample_time_sec"] = timestamp_sec
    return stats


def finalize_sampling_stats(stats):
    payload = dict(stats)
    count = int(payload.get("sampled_message_count", 0))
    first = payload.get("first_sample_time_sec")
    last = payload.get("last_sample_time_sec")
    effective_rate = None
    if count > 1 and first is not None and last is not None and float(last) > float(first):
        effective_rate = (count - 1) / (float(last) - float(first))
    payload["effective_sample_rate_hz"] = effective_rate
    return payload


def new_truth_pipeline_stats(configured_rate_hz=None):
    return {
        "configured_rate_hz": (
            float(configured_rate_hz) if configured_rate_hz is not None else None
        ),
        "callback_count": 0,
        "sampled_batch_count": 0,
        "enqueued_batch_count": 0,
        "written_batch_count": 0,
        "enqueued_row_count": 0,
        "written_row_count": 0,
        "queue_full_error_count": 0,
        "write_error_count": 0,
        "max_queue_depth": 0,
        "max_callback_gap_sec": 0.0,
        "estimated_missed_sample_slots": 0,
        "last_callback_time_sec": None,
        "flush_count": 0,
    }


def update_truth_callback_stats(stats, timestamp_sec):
    timestamp_sec = float(timestamp_sec)
    previous = coerce_float(stats.get("last_callback_time_sec"), None)
    stats["callback_count"] = int(stats.get("callback_count", 0)) + 1
    if previous is not None and timestamp_sec >= previous:
        gap = timestamp_sec - previous
        stats["max_callback_gap_sec"] = max(
            float(stats.get("max_callback_gap_sec", 0.0)), gap
        )
        rate_hz = coerce_float(stats.get("configured_rate_hz"), None)
        if rate_hz is not None and rate_hz > 0.0:
            stats["estimated_missed_sample_slots"] = int(
                stats.get("estimated_missed_sample_slots", 0)
            ) + max(0, int(math.floor(gap * rate_hz + 1.0e-9)) - 1)
    stats["last_callback_time_sec"] = timestamp_sec
    return stats


def finalize_truth_pipeline_stats(stats):
    payload = dict(stats)
    enqueued_batches = int(payload.get("enqueued_batch_count", 0))
    written_batches = int(payload.get("written_batch_count", 0))
    enqueued_rows = int(payload.get("enqueued_row_count", 0))
    written_rows = int(payload.get("written_row_count", 0))
    payload["unwritten_batch_count"] = max(0, enqueued_batches - written_batches)
    payload["unwritten_row_count"] = max(0, enqueued_rows - written_rows)
    payload["lossless_after_enqueue"] = (
        payload["unwritten_batch_count"] == 0
        and payload["unwritten_row_count"] == 0
        and int(payload.get("queue_full_error_count", 0)) == 0
        and int(payload.get("write_error_count", 0)) == 0
    )
    return payload


def normalize_object_id_catalog(value):
    if value in (None, ""):
        return {}
    if not isinstance(value, dict):
        raise ValueError("object_id_catalog must be a mapping of ID to model name")
    catalog = {}
    names = set()
    for raw_id, raw_name in value.items():
        if isinstance(raw_id, bool):
            raise ValueError(f"invalid object ID: {raw_id!r}")
        if isinstance(raw_id, int):
            object_id = raw_id
        elif isinstance(raw_id, str) and raw_id.strip().isdigit():
            object_id = int(raw_id.strip())
        else:
            raise ValueError(f"invalid object ID: {raw_id!r}")
        model_name = str(raw_name).strip()
        if object_id <= 0 or object_id > 254:
            raise ValueError(f"object ID must be in [1, 254], got {object_id}")
        if not model_name:
            raise ValueError(f"object ID {object_id} has an empty model name")
        if object_id in catalog:
            raise ValueError(f"object ID {object_id} is assigned more than once")
        if model_name in names:
            raise ValueError(f"model name '{model_name}' has more than one object ID")
        catalog[object_id] = model_name
        names.add(model_name)
    return catalog


def normalize_recording_mode(value):
    mode = str(value or "debug").strip().lower()
    if mode not in RECORDING_MODES:
        raise ValueError(
            "recording_mode must be 'debug' or 'formal', got %r" % value
        )
    return mode


def _is_fully_occluded(value):
    normalized = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    return "occlu" in normalized and any(
        token in normalized for token in ("full", "complete", "total")
    )


def _required_nonnegative_integer(factors, name):
    value = factors.get(name)
    if isinstance(value, bool):
        raise ValueError(f"experiment factor '{name}' must be a non-negative integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"experiment factor '{name}' must be a non-negative integer"
        ) from exc
    if parsed < 0 or str(value).strip() not in {str(parsed), f"{parsed}.0"}:
        raise ValueError(f"experiment factor '{name}' must be a non-negative integer")
    return parsed


def validate_recording_configuration(
    recording_mode,
    scenario_id,
    launch_scenario_id,
    experiment_factors,
    object_metadata,
    object_id_catalog,
):
    mode = normalize_recording_mode(recording_mode)
    scenario_id = str(scenario_id or "").strip()
    launch_scenario_id = str(launch_scenario_id or "").strip()
    factors = normalize_experiment_factors(experiment_factors)
    metadata = normalize_object_metadata(object_metadata)
    catalog = normalize_object_id_catalog(object_id_catalog)

    if launch_scenario_id and scenario_id and launch_scenario_id != scenario_id:
        raise ValueError(
            "launch scenario_id %r does not match recorder scenario_id %r"
            % (launch_scenario_id, scenario_id)
        )
    if mode == "debug":
        return

    if not scenario_id:
        raise ValueError("formal recording requires a non-empty recorder scenario_id")
    if not launch_scenario_id:
        raise ValueError("formal recording requires a non-empty launch scenario_id")
    if not catalog:
        raise ValueError("formal recording requires a non-empty object_id_catalog")
    missing_factors = [
        name for name in REQUIRED_FORMAL_RUN_FACTORS if factors.get(name) in (None, "")
    ]
    if missing_factors:
        raise ValueError(
            "formal recording is missing required experiment_factors: "
            + ", ".join(missing_factors)
        )

    moving_quantity = _required_nonnegative_integer(
        factors, "moving_object_quantity"
    )
    scene_quantity = _required_nonnegative_integer(factors, "scene_object_quantity")
    _required_nonnegative_integer(factors, "repeat_index")
    if scene_quantity != len(catalog):
        raise ValueError(
            "scene_object_quantity does not match object_id_catalog: "
            f"{scene_quantity} != {len(catalog)}"
        )
    if moving_quantity != len(metadata):
        raise ValueError(
            "moving_object_quantity does not match object_metadata: "
            f"{moving_quantity} != {len(metadata)}"
        )

    catalog_names = set(catalog.values())
    for model_name, attributes in metadata.items():
        if model_name not in catalog_names:
            raise ValueError(
                f"object_metadata model '{model_name}' is not in object_id_catalog"
            )
        missing_attributes = [
            name
            for name in REQUIRED_FORMAL_OBJECT_ATTRIBUTES
            if attributes.get(name) in (None, "")
        ]
        if missing_attributes:
            raise ValueError(
                f"object_metadata for '{model_name}' is missing: "
                + ", ".join(missing_attributes)
            )
        if _is_fully_occluded(attributes.get("visibility_condition")):
            raise ValueError(
                f"fully occluded object '{model_name}' cannot be formally evaluated"
            )


def validate_control_scenario_ids(
    scenario_id, discovered_controls, evaluated_object_names=None
):
    scenario_id = str(scenario_id or "").strip()
    if not scenario_id:
        return
    allowed = None
    if evaluated_object_names is not None:
        allowed = {
            str(name).strip()
            for name in evaluated_object_names
            if str(name).strip()
        }
    for control in discovered_controls or []:
        if not isinstance(control, dict):
            continue
        model_name = str(control.get("controlled_object", "")).strip()
        if allowed is not None and model_name not in allowed:
            continue
        control_scenario_id = str(control.get("scenario_id", "")).strip()
        if control_scenario_id != scenario_id:
            raise ValueError(
                "controller scenario_id %r for '%s' does not match recorder scenario_id %r"
                % (control_scenario_id, model_name, scenario_id)
            )


def normalize_experiment_factors(value):
    if value in (None, ""):
        return {}
    if not isinstance(value, dict):
        raise ValueError("experiment_factors must be a mapping")

    factors = {}
    for raw_name, raw_value in value.items():
        if not isinstance(raw_name, str) or not raw_name.strip():
            raise ValueError("experiment factor names must be non-empty strings")
        name = raw_name.strip()
        if name in factors:
            raise ValueError(f"experiment factor '{name}' is assigned more than once")
        try:
            # The JSON round trip rejects ROS-only objects and creates an immutable
            # snapshot rather than retaining a reference to the parameter tree.
            factors[name] = json.loads(json.dumps(raw_value, allow_nan=False))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"experiment factor '{name}' must be JSON-serializable with finite values"
            ) from exc
    return factors


def normalize_object_metadata(value):
    if value in (None, ""):
        return {}
    if not isinstance(value, dict):
        raise ValueError("object_metadata must map model names to attribute mappings")
    metadata = {}
    for raw_model_name, raw_attributes in value.items():
        model_name = str(raw_model_name).strip()
        if not model_name:
            raise ValueError("object_metadata model names must be non-empty")
        if not isinstance(raw_attributes, dict):
            raise ValueError(
                f"object '{model_name}' metadata must be an attribute mapping"
            )
        if model_name in metadata:
            raise ValueError(f"object metadata for '{model_name}' is duplicated")
        metadata[model_name] = normalize_experiment_factors(raw_attributes)
    return metadata


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
    object_observation_stats_topic="",
    truth_object_rate_hz=10.0,
    truth_link_rate_hz=10.0,
    surface_truth_link_prefixes=("ground_truth_",),
    record_surface_truth_link_trajectories=False,
    motion_truth_drive_links=None,
    record_flush_interval_sec=1.0,
    record_flush_max_rows=100,
    anchor_subscriber_queue_size=64,
    anchor_processing_queue_size=64,
    anchor_processing_enqueue_timeout_sec=2.0,
    truth_subscriber_queue_size=64,
    truth_processing_queue_size=256,
    truth_processing_enqueue_timeout_sec=2.0,
    algorithm_storage_backend="jsonl",
    truth_motion_policy=None,
):
    enabled = bool(str(sensor_frame_name).strip() or str(sensor_scoped_link_name).strip())
    algorithm_storage_backend = normalize_algorithm_storage_backend(
        algorithm_storage_backend
    )
    algorithm_recording = {
        "schema_version": 3 if algorithm_storage_backend != "jsonl" else 2,
        "storage_backend": algorithm_storage_backend,
        "native_frame_only": True,
        "alignment_required_before_recording": False,
        "anchor_storage": "static_catalog_plus_dynamic_observations",
        "flush_interval_sec": float(record_flush_interval_sec),
        "flush_max_rows": int(record_flush_max_rows),
        "anchor_subscriber_queue_size": int(anchor_subscriber_queue_size),
        "anchor_processing_queue_size": int(anchor_processing_queue_size),
        "anchor_processing_enqueue_timeout_sec": float(
            anchor_processing_enqueue_timeout_sec
        ),
        "clean_shutdown_marker": "meta/run_complete.json",
    }
    if algorithm_storage_backend != "jsonl":
        algorithm_recording.update(
            {
                "database_file": "algorithm/{}".format(
                    ALGORITHM_FRAME_DATABASE_FILENAME
                ),
                "compressed_streams": list(HIGH_VOLUME_STREAMS),
                "codec": "zlib",
                "compression_level": 1,
                "canonical_json_version": 1,
            }
        )
    motion_truth_drive_links = normalize_model_link_mapping(
        motion_truth_drive_links
    )
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
            "object_observation_stats": str(object_observation_stats_topic),
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
        "truth_recording": {
            "object_state_source": "gazebo_model_states",
            "object_rate_hz": float(truth_object_rate_hz),
            "object_pose_frame": str(truth_frame),
            "object_twist_frame": str(truth_frame),
            "link_rate_hz": float(truth_link_rate_hz),
            "dynamic_link_policy": (
                "sensor_motion_drive_links_and_surface_catalog"
                if motion_truth_drive_links
                else "sensor_and_surface_truth_links"
                if record_surface_truth_link_trajectories
                else "sensor_only"
            ),
            "motion_truth_drive_links": motion_truth_drive_links,
            "surface_truth_points": {
                "file": "truth/surface_truth_points.jsonl",
                "storage_mode": (
                    "motion_parent_link_local_pose_once"
                    if motion_truth_drive_links
                    else "object_local_pose_once"
                ),
                "link_prefixes": list(surface_truth_link_prefixes),
            },
            "subscriber_queue_size": int(truth_subscriber_queue_size),
            "processing_queue_size": int(truth_processing_queue_size),
            "processing_enqueue_timeout_sec": float(
                truth_processing_enqueue_timeout_sec
            ),
            "write_mode": "asynchronous_batched",
        },
        "algorithm_recording": algorithm_recording,
        "truth_motion_policy": normalize_truth_motion_policy(truth_motion_policy),
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
    experiment_factors=None,
    object_metadata=None,
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
        "experiment_factors": normalize_experiment_factors(experiment_factors),
        "object_metadata": normalize_object_metadata(object_metadata),
    }


def select_authoritative_discovered_controls(
    current_scenario_id, discovered_controls, allowed_object_names=None
):
    current_scenario_id = str(current_scenario_id).strip()
    discovered_controls = discovered_controls if isinstance(discovered_controls, list) else []
    if not current_scenario_id:
        return []

    if allowed_object_names:
        allowed = {
            str(name).strip() for name in allowed_object_names if str(name).strip()
        }
        selected = {}
        for control in discovered_controls:
            if not isinstance(control, dict):
                continue
            controlled_object = str(control.get("controlled_object", "")).strip()
            if controlled_object not in allowed:
                continue
            if str(control.get("scenario_id", "")).strip() != current_scenario_id:
                return []
            if not str(control.get("command_frame", "")).strip():
                return []
            try:
                if not math.isfinite(float(control.get("start_delay_sec"))):
                    return []
                if not math.isfinite(float(control.get("duration_sec"))):
                    return []
            except (TypeError, ValueError):
                return []
            if controlled_object in selected:
                return []
            selected[controlled_object] = control
        return [selected[name] for name in sorted(selected)]

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


def _detect_cpu_model():
    try:
        for line in pathlib.Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith("model name") and ":" in line:
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def _detect_memory_total_bytes():
    try:
        for line in pathlib.Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) * 1024
    except (OSError, IndexError, TypeError, ValueError):
        pass
    return 0


def build_hardware_manifest_payload(
    hostname=None,
    platform_string=None,
    cpu_model=None,
    logical_cpu_count=None,
    memory_total_bytes=None,
):
    return {
        "created_at_iso": dt.datetime.now().isoformat(),
        "hostname": str(hostname if hostname is not None else socket.gethostname()),
        "platform": str(
            platform_string if platform_string is not None else platform.platform()
        ),
        "cpu_model": str(cpu_model if cpu_model is not None else _detect_cpu_model()),
        "logical_cpu_count": int(
            logical_cpu_count
            if logical_cpu_count is not None
            else (os.cpu_count() or 0)
        ),
        "memory_total_bytes": int(
            memory_total_bytes
            if memory_total_bytes is not None
            else _detect_memory_total_bytes()
        ),
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
    persistent_risk = deform_monitor.get("persistent_risk", {})
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
    if not isinstance(persistent_risk, dict):
        persistent_risk = {}
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
            "significance_cusum_h": float(significance.get("cusum_h", 0.0)),
            "directional_tau_s": float(directional_motion.get("tau_s", 0.0)),
            "directional_tau_c": float(directional_motion.get("tau_c", 0.0)),
            "persistent_min_confirmed_mean_risk": float(
                persistent_risk.get("min_confirmed_mean_risk", 0.0)
            ),
        },
    }


def serialize_anchor_static_state(msg):
    return {
        "id": int(getattr(msg, "id", 0)),
        "anchor_type": int(getattr(msg, "anchor_type", 0)),
        "object_id": int(getattr(msg, "object_id", 0)),
        "object_id_valid": bool(getattr(msg, "object_id_valid", False)),
        "object_id_confidence": float(
            getattr(msg, "object_id_confidence", 0.0)
        ),
        "object_id_support_count": int(
            getattr(msg, "object_id_support_count", 0)
        ),
        "center": point_to_dict(getattr(msg, "center", None)),
        "normal": point_to_dict(getattr(msg, "normal", None)),
        "edge_normal": point_to_dict(getattr(msg, "edge_normal", None)),
        "point_count": int(getattr(msg, "point_count", 0)),
        "ref_quality": float(getattr(msg, "ref_quality", 0.0)),
        "covariance_quality": float(getattr(msg, "covariance_quality", 0.0)),
        "type_stability": float(getattr(msg, "type_stability", 0.0)),
        "shape_linearity": float(getattr(msg, "shape_linearity", 0.0)),
        "shape_planarity": float(getattr(msg, "shape_planarity", 0.0)),
        "shape_scattering": float(getattr(msg, "shape_scattering", 0.0)),
        "ref_center": point_to_dict(getattr(msg, "ref_center", None)),
        "reference_epoch": int(getattr(msg, "reference_epoch", 0)),
        "reference_stamp": time_to_dict(getattr(msg, "reference_stamp", None)),
        "reference_origin": int(getattr(msg, "reference_origin", 0)),
    }


def serialize_anchor_dynamic_state(msg):
    dynamic = {
        "id": int(getattr(msg, "id", 0)),
        "observed_object_id": int(getattr(msg, "observed_object_id", 0)),
        "observed_object_id_valid": bool(
            getattr(msg, "observed_object_id_valid", False)
        ),
        "observed_object_id_confidence": float(
            getattr(msg, "observed_object_id_confidence", 0.0)
        ),
        "observed_object_id_support_count": int(
            getattr(msg, "observed_object_id_support_count", 0)
        ),
        "object_association_state": int(
            getattr(msg, "object_association_state", 0)
        ),
        "visible_count": int(getattr(msg, "visible_count", 0)),
        "observation_support_count": int(
            getattr(msg, "observation_support_count", 0)
        ),
        "edge_support_count": int(getattr(msg, "edge_support_count", 0)),
        "current_shape_linearity": float(
            getattr(msg, "current_shape_linearity", 0.0)
        ),
        "current_shape_planarity": float(
            getattr(msg, "current_shape_planarity", 0.0)
        ),
        "current_shape_scattering": float(
            getattr(msg, "current_shape_scattering", 0.0)
        ),
        "edge_direction_angle_deg": float(
            getattr(msg, "edge_direction_angle_deg", 180.0)
        ),
        "edge_geometry_stability": float(
            getattr(msg, "edge_geometry_stability", 0.0)
        ),
        "edge_geometry_valid": bool(getattr(msg, "edge_geometry_valid", False)),
        "scalar_count": int(getattr(msg, "scalar_count", 0)),
        "scalar_type": [int(value) for value in getattr(msg, "scalar_type", [])],
        "scalar_z": [float(value) for value in getattr(msg, "scalar_z", [])],
        "scalar_r": [float(value) for value in getattr(msg, "scalar_r", [])],
        "scalar_valid": [bool(value) for value in getattr(msg, "scalar_valid", [])],
        "matched_center": point_to_dict(getattr(msg, "matched_center", None)),
        "matched_delta": point_to_dict(getattr(msg, "matched_delta", None)),
        "predicted_center": point_to_dict(getattr(msg, "predicted_center", None)),
        "predicted_displacement": point_to_dict(
            getattr(msg, "predicted_displacement", None)
        ),
        "reacquisition_score": float(getattr(msg, "reacquisition_score", 0.0)),
        "reacquisition_innovation_norm": float(
            getattr(msg, "reacquisition_innovation_norm", 0.0)
        ),
        "disp_mean": [
            float(value) for value in getattr(msg, "disp_mean", [0.0, 0.0, 0.0])
        ],
        "disp_cov_diag": [
            float(value) for value in getattr(msg, "disp_cov_diag", [])
        ],
        "vel_mean": [float(value) for value in getattr(msg, "vel_mean", [])],
        "dof_obs": int(getattr(msg, "dof_obs", 0)),
        "chi2_stat": float(getattr(msg, "chi2_stat", 0.0)),
        "disp_norm": float(getattr(msg, "disp_norm", 0.0)),
        "disp_normal": float(getattr(msg, "disp_normal", 0.0)),
        "disp_edge": float(getattr(msg, "disp_edge", 0.0)),
        "cmp_score": float(getattr(msg, "cmp_score", 0.0)),
        "cusum_score": float(getattr(msg, "cusum_score", 0.0)),
        "directional_strength": float(getattr(msg, "directional_strength", 0.0)),
        "directional_coherence": float(getattr(msg, "directional_coherence", 0.0)),
        "directional_persistent": bool(
            getattr(msg, "directional_persistent", False)
        ),
        "instantaneous_displacement_evidence": bool(
            getattr(msg, "instantaneous_displacement_evidence", False)
        ),
        "persistent_candidate": bool(getattr(msg, "persistent_candidate", False)),
        "cluster_member": bool(getattr(msg, "cluster_member", False)),
        "graph_candidate": bool(getattr(msg, "graph_candidate", False)),
        "graph_neighbor_count": int(getattr(msg, "graph_neighbor_count", 0)),
        "graph_coherent_score": float(getattr(msg, "graph_coherent_score", 0.0)),
        "graph_temporal_score": float(getattr(msg, "graph_temporal_score", 0.0)),
        "graph_persistence_score": float(
            getattr(msg, "graph_persistence_score", 0.0)
        ),
        "weak_plane_candidate": bool(getattr(msg, "weak_plane_candidate", False)),
        "weak_plane_group_size": int(getattr(msg, "weak_plane_group_size", 0)),
        "weak_plane_current_support": int(
            getattr(msg, "weak_plane_current_support", 0)
        ),
        "weak_plane_temporal_frame_support": int(
            getattr(msg, "weak_plane_temporal_frame_support", 0)
        ),
        "weak_plane_streak": int(getattr(msg, "weak_plane_streak", 0)),
        "weak_plane_group_disp": float(getattr(msg, "weak_plane_group_disp", 0.0)),
        "weak_plane_mean_chi2": float(getattr(msg, "weak_plane_mean_chi2", 0.0)),
        "weak_plane_direction_consistency": float(
            getattr(msg, "weak_plane_direction_consistency", 0.0)
        ),
        "weak_plane_group_residual": float(
            getattr(msg, "weak_plane_group_residual", 0.0)
        ),
        "weak_plane_component_id": int(
            getattr(msg, "weak_plane_component_id", -1)
        ),
        "weak_plane_exterior_background_support": int(
            getattr(msg, "weak_plane_exterior_background_support", 0)
        ),
        "weak_plane_mixed_type_support": int(
            getattr(msg, "weak_plane_mixed_type_support", 0)
        ),
        "local_bg_count": int(getattr(msg, "local_bg_count", 0)),
        "local_contrast_score": float(getattr(msg, "local_contrast_score", 0.0)),
        "local_rel_norm": float(getattr(msg, "local_rel_norm", 0.0)),
        "local_rel_normal": float(getattr(msg, "local_rel_normal", 0.0)),
        "local_rel_edge": float(getattr(msg, "local_rel_edge", 0.0)),
        "plane_bg_count": int(getattr(msg, "plane_bg_count", 0)),
        "plane_contrast_score": float(getattr(msg, "plane_contrast_score", 0.0)),
        "plane_rel_norm": float(getattr(msg, "plane_rel_norm", 0.0)),
        "plane_rel_normal": float(getattr(msg, "plane_rel_normal", 0.0)),
        "plane_rel_edge": float(getattr(msg, "plane_rel_edge", 0.0)),
        "permanent_deformed": bool(getattr(msg, "permanent_deformed", False)),
        "permanent_displacement": [
            float(value) for value in getattr(msg, "permanent_displacement", [])
        ],
        "comparable": bool(getattr(msg, "comparable", False)),
        "observable": bool(getattr(msg, "observable", False)),
        "significant": bool(getattr(msg, "significant", False)),
        "reacquired": bool(getattr(msg, "reacquired", False)),
        "obs_state": int(getattr(msg, "obs_state", 0)),
        "detection_mode": int(getattr(msg, "detection_mode", 0)),
        "disappearance_score": float(getattr(msg, "disappearance_score", 0.0)),
    }
    return dynamic


def serialize_anchor_state(msg):
    anchor = serialize_anchor_static_state(msg)
    anchor.update(serialize_anchor_dynamic_state(msg))
    return anchor


def serialize_anchor_object_summary(msg):
    return {
        "object_id": int(getattr(msg, "object_id", 0)),
        "total_count": int(getattr(msg, "total_count", 0)),
        "comparable_count": int(getattr(msg, "comparable_count", 0)),
        "significant_count": int(getattr(msg, "significant_count", 0)),
        "plane_count": int(getattr(msg, "plane_count", 0)),
        "edge_count": int(getattr(msg, "edge_count", 0)),
        "band_count": int(getattr(msg, "band_count", 0)),
        "excluded_not_observable": int(
            getattr(msg, "excluded_not_observable", 0)
        ),
        "excluded_weak_or_missing": int(
            getattr(msg, "excluded_weak_or_missing", 0)
        ),
    }


def serialize_anchor_states(msg):
    return {
        "header": serialize_header(msg),
        "reference_epoch": int(getattr(msg, "reference_epoch", 0)),
        "reference_initialized_at": time_to_dict(
            getattr(msg, "reference_initialized_at", None)
        ),
        "total_anchor_count": int(getattr(msg, "total_anchor_count", 0)),
        "object_summaries": [
            serialize_anchor_object_summary(s)
            for s in getattr(msg, "object_summaries", [])
        ],
        "anchors": [serialize_anchor_state(a) for a in msg.anchors],
    }


def split_serialized_anchor_state(anchor):
    anchor = dict(anchor)
    anchor_id = int(anchor.get("id", 0))
    static = {"id": anchor_id}
    for field in ANCHOR_STATIC_FIELDS:
        if field in anchor:
            static[field] = anchor[field]

    dynamic = {"id": anchor_id}
    for field, value in anchor.items():
        if field == "id" or field in ANCHOR_STATIC_FIELDS:
            continue
        dynamic[field] = value
    return static, dynamic


def reconstruct_anchor_state_records(catalog_records, observation_records):
    catalog = {}
    for record in catalog_records:
        anchor = record.get("anchor", {})
        if not isinstance(anchor, dict):
            continue
        try:
            epoch = int(record.get("reference_epoch", anchor.get("reference_epoch", 0)))
            anchor_id = int(record.get("anchor_id", anchor.get("id", 0)))
        except (TypeError, ValueError):
            continue
        catalog[(epoch, anchor_id)] = dict(anchor)

    reconstructed = []
    for observation in observation_records:
        record = {
            "header": observation.get("header", {}),
            "reference_epoch": int(observation.get("reference_epoch", 0)),
            "reference_initialized_at": observation.get("reference_initialized_at"),
            "recorded_at": observation.get("recorded_at"),
            "anchors": [],
        }
        epoch = record["reference_epoch"]
        for dynamic in observation.get("anchors", []):
            if not isinstance(dynamic, dict):
                continue
            try:
                anchor_id = int(dynamic.get("id", 0))
            except (TypeError, ValueError):
                continue
            merged = dict(catalog.get((epoch, anchor_id), {}))
            merged.update(dynamic)
            merged.setdefault("id", anchor_id)
            merged.setdefault("reference_epoch", epoch)
            record["anchors"].append(merged)
        reconstructed.append(record)
    return reconstructed


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
    truth_reference_stamp_sec=None,
    algorithm_reference_stamp_sec=None,
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

    truth_stamp = coerce_float(truth_reference_stamp_sec, None)
    algorithm_stamp = coerce_float(algorithm_reference_stamp_sec, None)
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
        "truth_reference_stamp_sec": truth_stamp,
        "algorithm_reference_stamp_sec": algorithm_stamp,
        "pose_pair_delta_sec": (
            abs(algorithm_stamp - truth_stamp)
            if truth_stamp is not None and algorithm_stamp is not None
            else None
        ),
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
        "object_id": int(getattr(msg, "object_id", 0)),
        "object_id_valid": bool(getattr(msg, "object_id_valid", False)),
        "object_id_confidence": float(getattr(msg, "object_id_confidence", 0.0)),
        "object_id_ambiguous": bool(getattr(msg, "object_id_ambiguous", False)),
        "observed_object_id": int(getattr(msg, "observed_object_id", 0)),
        "observed_object_id_valid": bool(
            getattr(msg, "observed_object_id_valid", False)
        ),
        "observed_object_id_confidence": float(
            getattr(msg, "observed_object_id_confidence", 0.0)
        ),
        "observed_object_id_ambiguous": bool(
            getattr(msg, "observed_object_id_ambiguous", False)
        ),
        "object_association_state": int(getattr(msg, "object_association_state", 0)),
        "association_consistent_count": int(
            getattr(msg, "association_consistent_count", 0)
        ),
        "association_mismatch_count": int(
            getattr(msg, "association_mismatch_count", 0)
        ),
        "association_mixed_count": int(getattr(msg, "association_mixed_count", 0)),
        "association_unavailable_count": int(
            getattr(msg, "association_unavailable_count", 0)
        ),
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
        "evidences": [serialize_risk_evidence_entry(item) for item in evidences],
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
        "object_id": int(getattr(msg, "object_id", 0)),
        "object_id_valid": bool(getattr(msg, "object_id_valid", False)),
        "object_id_confidence": float(
            getattr(msg, "object_id_confidence", 0.0)
        ),
        "object_id_ambiguous": bool(
            getattr(msg, "object_id_ambiguous", False)
        ),
        "observed_object_id": int(getattr(msg, "observed_object_id", 0)),
        "observed_object_id_valid": bool(
            getattr(msg, "observed_object_id_valid", False)
        ),
        "observed_object_id_confidence": float(
            getattr(msg, "observed_object_id_confidence", 0.0)
        ),
        "observed_object_id_ambiguous": bool(
            getattr(msg, "observed_object_id_ambiguous", False)
        ),
        "object_association_state": int(getattr(msg, "object_association_state", 0)),
        "association_consistent_count": int(
            getattr(msg, "association_consistent_count", 0)
        ),
        "association_mismatch_count": int(
            getattr(msg, "association_mismatch_count", 0)
        ),
        "association_mixed_count": int(getattr(msg, "association_mixed_count", 0)),
        "association_unavailable_count": int(
            getattr(msg, "association_unavailable_count", 0)
        ),
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
        "reference_epoch": int(getattr(msg, "reference_epoch", 0)),
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
        "object_id": int(getattr(msg, "object_id", 0)),
        "object_id_valid": bool(getattr(msg, "object_id_valid", False)),
        "object_id_confidence": float(
            getattr(msg, "object_id_confidence", 0.0)
        ),
        "observed_object_id": int(getattr(msg, "observed_object_id", 0)),
        "observed_object_id_valid": bool(
            getattr(msg, "observed_object_id_valid", False)
        ),
        "observed_object_id_confidence": float(
            getattr(msg, "observed_object_id_confidence", 0.0)
        ),
        "observed_object_id_support_count": int(
            getattr(msg, "observed_object_id_support_count", 0)
        ),
        "object_association_state": int(getattr(msg, "object_association_state", 0)),
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
        "object_id": int(getattr(msg, "object_id", 0)),
        "object_id_valid": bool(getattr(msg, "object_id_valid", False)),
        "object_id_confidence": float(
            getattr(msg, "object_id_confidence", 0.0)
        ),
        "object_id_ambiguous": bool(
            getattr(msg, "object_id_ambiguous", False)
        ),
        "observed_object_id": int(getattr(msg, "observed_object_id", 0)),
        "observed_object_id_valid": bool(
            getattr(msg, "observed_object_id_valid", False)
        ),
        "observed_object_id_confidence": float(
            getattr(msg, "observed_object_id_confidence", 0.0)
        ),
        "observed_object_id_ambiguous": bool(
            getattr(msg, "observed_object_id_ambiguous", False)
        ),
        "object_association_state": int(getattr(msg, "object_association_state", 0)),
        "association_consistent_count": int(
            getattr(msg, "association_consistent_count", 0)
        ),
        "association_mismatch_count": int(
            getattr(msg, "association_mismatch_count", 0)
        ),
        "association_mixed_count": int(getattr(msg, "association_mixed_count", 0)),
        "association_unavailable_count": int(
            getattr(msg, "association_unavailable_count", 0)
        ),
        "center": point_to_dict(getattr(msg, "center", None)),
        "bbox_min": point_to_dict(getattr(msg, "bbox_min", None)),
        "bbox_max": point_to_dict(getattr(msg, "bbox_max", None)),
        "mean_risk": float(getattr(msg, "mean_risk", 0.0)),
        "peak_risk": float(getattr(msg, "peak_risk", 0.0)),
        "confidence": float(getattr(msg, "confidence", 0.0)),
        "voxel_count": int(getattr(msg, "voxel_count", 0)),
        "significant": bool(getattr(msg, "significant", False)),
    }


def serialize_object_observation_stats(msg):
    return {
        "header": serialize_header(msg),
        "phase": int(getattr(msg, "phase", 0)),
        "reference_epoch": int(getattr(msg, "reference_epoch", 0)),
        "window_start": time_to_dict(getattr(msg, "window_start", None)),
        "window_end": time_to_dict(getattr(msg, "window_end", None)),
        "frame_count": int(getattr(msg, "frame_count", 0)),
        "total_point_count": int(getattr(msg, "total_point_count", 0)),
        "valid_label_point_count": int(
            getattr(msg, "valid_label_point_count", 0)
        ),
        "invalid_label_point_count": int(
            getattr(msg, "invalid_label_point_count", 0)
        ),
        "objects": [
            {
                "object_id": int(getattr(item, "object_id", 0)),
                "point_count": int(getattr(item, "point_count", 0)),
                "visible_frame_count": int(
                    getattr(item, "visible_frame_count", 0)
                ),
            }
            for item in getattr(msg, "objects", [])
        ],
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
        self.truth_object_rate_hz = normalize_positive_rate_hz(
            rospy.get_param("~truth_object_rate_hz", 10.0),
            10.0,
            "truth_object_rate_hz",
        )
        self.truth_link_rate_hz = normalize_positive_rate_hz(
            rospy.get_param("~truth_link_rate_hz", 10.0),
            10.0,
            "truth_link_rate_hz",
        )
        self.surface_truth_link_prefixes = normalize_string_prefixes(
            rospy.get_param("~surface_truth_link_prefixes", ["ground_truth_"])
        )
        self.record_surface_truth_link_trajectories = bool(
            rospy.get_param("~record_surface_truth_link_trajectories", False)
        )
        self.motion_truth_drive_links = normalize_model_link_mapping(
            rospy.get_param("~motion_truth_drive_links", {})
        )
        self.surface_truth_world_file = str(
            rospy.get_param("~surface_truth_world_file", "")
        ).strip()
        self.surface_truth_expected_count = int(
            rospy.get_param("~surface_truth_expected_count", 0)
        )
        self.surface_truth_max_local_radius_m = float(
            rospy.get_param("~surface_truth_max_local_radius_m", 3.0)
        )
        self.surface_truth_require_clean_world = bool(
            rospy.get_param("~surface_truth_require_clean_world", False)
        )
        if self.surface_truth_expected_count < 0:
            raise ValueError("surface_truth_expected_count must be non-negative")
        if (
            not math.isfinite(self.surface_truth_max_local_radius_m)
            or self.surface_truth_max_local_radius_m <= 0.0
        ):
            raise ValueError(
                "surface_truth_max_local_radius_m must be finite and positive"
            )
        self.record_flush_interval_sec = normalize_positive_rate_hz(
            rospy.get_param("~record_flush_interval_sec", 1.0),
            1.0,
            "record_flush_interval_sec",
        )
        self.record_flush_max_rows = int(
            rospy.get_param("~record_flush_max_rows", 100)
        )
        if self.record_flush_max_rows <= 0:
            raise ValueError("record_flush_max_rows must be a positive integer")
        self.truth_subscriber_queue_size = int(
            rospy.get_param("~truth_subscriber_queue_size", 64)
        )
        if self.truth_subscriber_queue_size <= 0:
            raise ValueError("truth_subscriber_queue_size must be a positive integer")
        self.truth_processing_queue_size = int(
            rospy.get_param("~truth_processing_queue_size", 256)
        )
        if self.truth_processing_queue_size <= 0:
            raise ValueError("truth_processing_queue_size must be a positive integer")
        self.truth_processing_enqueue_timeout_sec = float(
            rospy.get_param("~truth_processing_enqueue_timeout_sec", 2.0)
        )
        if (
            not math.isfinite(self.truth_processing_enqueue_timeout_sec)
            or self.truth_processing_enqueue_timeout_sec <= 0.0
        ):
            raise ValueError(
                "truth_processing_enqueue_timeout_sec must be finite and positive"
            )
        self.anchor_subscriber_queue_size = int(
            rospy.get_param("~anchor_subscriber_queue_size", 64)
        )
        if self.anchor_subscriber_queue_size <= 0:
            raise ValueError("anchor_subscriber_queue_size must be a positive integer")
        self.anchor_processing_queue_size = int(
            rospy.get_param("~anchor_processing_queue_size", 64)
        )
        if self.anchor_processing_queue_size <= 0:
            raise ValueError("anchor_processing_queue_size must be a positive integer")
        self.anchor_processing_enqueue_timeout_sec = float(
            rospy.get_param("~anchor_processing_enqueue_timeout_sec", 2.0)
        )
        if (
            not math.isfinite(self.anchor_processing_enqueue_timeout_sec)
            or self.anchor_processing_enqueue_timeout_sec <= 0.0
        ):
            raise ValueError(
                "anchor_processing_enqueue_timeout_sec must be finite and positive"
            )
        self.truth_motion_policy = normalize_truth_motion_policy(
            rospy.get_param("~truth_motion_policy", {})
        )
        self.ground_truth_odometry_topic = str(
            rospy.get_param("~ground_truth_odometry_topic", "/ground_truth/odom")
        ).strip()
        self.clusters_topic = str(
            rospy.get_param("~clusters_topic", "/deform/clusters")
        ).strip()
        self.object_observation_stats_topic = str(
            rospy.get_param(
                "~object_observation_stats_topic", "/deform/object_observation_stats"
            )
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
        self.recording_mode = normalize_recording_mode(
            rospy.get_param("~recording_mode", "debug")
        )
        self.algorithm_storage_backend = normalize_algorithm_storage_backend(
            rospy.get_param("~algorithm_storage_backend", "jsonl")
        )
        self.scenario_id = str(rospy.get_param("~scenario_id", "")).strip()
        self.launch_scenario_id = str(
            rospy.get_param("~launch_scenario_id", "")
        ).strip()
        self.object_id_catalog = normalize_object_id_catalog(
            rospy.get_param("~object_id_catalog", {})
        )
        self.experiment_factors = normalize_experiment_factors(
            rospy.get_param("~experiment_factors", {})
        )
        self.object_metadata = normalize_object_metadata(
            rospy.get_param("~object_metadata", {})
        )
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

        validate_recording_configuration(
            recording_mode=self.recording_mode,
            scenario_id=self.scenario_id,
            launch_scenario_id=self.launch_scenario_id,
            experiment_factors=self.experiment_factors,
            object_metadata=self.object_metadata,
            object_id_catalog=self.object_id_catalog,
        )
        self._static_surface_truth_catalog = []
        if self.surface_truth_world_file:
            self._static_surface_truth_catalog = load_static_surface_truth_catalog(
                self.surface_truth_world_file,
                self.motion_truth_drive_links,
                self.object_id_catalog,
                expected_count=self.surface_truth_expected_count,
                max_local_radius_m=self.surface_truth_max_local_radius_m,
                require_clean_world=self.surface_truth_require_clean_world,
            )
        elif self.recording_mode == "formal" and self.motion_truth_drive_links:
            raise ValueError(
                "formal recording requires surface_truth_world_file for static landmarks"
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
        self._latest_model_pose_world = {}
        self._sensor_relative_pose_cache = {}
        listener_factory = getattr(tf, "TransformListener", None)
        self._tf_listener = listener_factory() if callable(listener_factory) else None

        self._object_files = {}
        self._link_files = {}
        self._algorithm_files = {}
        self._algorithm_frame_store = None
        self._algorithm_storage_summary = None
        self._recording_error = ""
        self._algorithm_pending_rows = {}
        self._algorithm_last_flush_monotonic = {}
        self._stream_stats = {}
        self._stream_last_seq = {}
        self._stream_sequences = {}
        self._frame_stream_records = {}
        self._anchor_catalog_cache = {}
        self._anchor_processing_queue = None
        self._anchor_processing_thread = None
        self._anchor_processing_error = None
        self._anchor_processing_stats = {
            "enqueued_message_count": 0,
            "processed_message_count": 0,
            "max_queue_depth": 0,
            "queue_full_error_count": 0,
            "processing_error_count": 0,
        }
        self._captured_surface_truth_links = {
            record["scoped_link_name"]
            for record in self._static_surface_truth_catalog
        }
        self._static_surface_truth_catalog_loaded = bool(
            self._static_surface_truth_catalog
        )
        self._surface_truth_catalog_handle = None
        self._callback_condition = threading.Condition()
        self._active_callbacks = 0
        self._callback_thread_counts = {}
        self._algorithm_write_lock = threading.RLock()
        self._closing = False
        self._closed = False
        self._deferred_close = False
        self._finish_close_started = False
        self._finish_close_done = False
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

        self._last_model_states_write_time = None
        self._last_link_states_write_time = None
        self._next_model_states_sample_time = None
        self._next_link_states_sample_time = None
        self._last_model_states_observed_time = None
        self._last_link_states_observed_time = None
        self._truth_sampling_stats = {
            "model_states": new_sampling_stats(self.truth_object_rate_hz),
            "link_states": new_sampling_stats(self.truth_link_rate_hz),
        }
        self._truth_write_queue = None
        self._truth_write_thread = None
        self._truth_write_error = None
        self._truth_pipeline_lock = threading.Lock()
        self._truth_pipeline_stats = {
            "model_states": new_truth_pipeline_stats(self.truth_object_rate_hz),
            "link_states": new_truth_pipeline_stats(self.truth_link_rate_hz),
            "surface_truth": new_truth_pipeline_stats(),
        }

        self._ensure_directories()
        if self.algorithm_storage_backend in ("sqlite_zlib", "dual"):
            self._algorithm_frame_store = AsyncCompressedFrameStore(
                self.algorithm_dir / ALGORITHM_FRAME_DATABASE_FILENAME
            )
        self._start_anchor_processing_worker()
        self._start_truth_write_worker()
        if self._static_surface_truth_catalog:
            self._enqueue_truth_rows(
                "surface_truth", self._static_surface_truth_catalog
            )
        self._publish_runtime_output_dir_param()
        self._write_run_info()
        self._write_run_metadata()

        rospy.on_shutdown(self.close)
        self._subscribers = [
            rospy.Subscriber(
                self.model_states_topic,
                ModelStates,
                self._handle_model_states,
                queue_size=self.truth_subscriber_queue_size,
            ),
            rospy.Subscriber(
                self.link_states_topic,
                LinkStates,
                self._handle_link_states,
                queue_size=self.truth_subscriber_queue_size,
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
                self.object_observation_stats_topic,
                ObjectObservationStats,
                self._handle_object_observation_stats,
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
                queue_size=self.anchor_subscriber_queue_size,
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
            object_observation_stats_topic=getattr(
                self,
                "object_observation_stats_topic",
                "/deform/object_observation_stats",
            ),
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
            truth_object_rate_hz=getattr(self, "truth_object_rate_hz", 10.0),
            truth_link_rate_hz=getattr(self, "truth_link_rate_hz", 10.0),
            surface_truth_link_prefixes=getattr(
                self, "surface_truth_link_prefixes", ("ground_truth_",)
            ),
            record_surface_truth_link_trajectories=getattr(
                self, "record_surface_truth_link_trajectories", False
            ),
            motion_truth_drive_links=getattr(
                self, "motion_truth_drive_links", {}
            ),
            record_flush_interval_sec=getattr(
                self, "record_flush_interval_sec", 1.0
            ),
            record_flush_max_rows=getattr(self, "record_flush_max_rows", 100),
            anchor_subscriber_queue_size=getattr(
                self, "anchor_subscriber_queue_size", 64
            ),
            anchor_processing_queue_size=getattr(
                self, "anchor_processing_queue_size", 64
            ),
            anchor_processing_enqueue_timeout_sec=getattr(
                self, "anchor_processing_enqueue_timeout_sec", 2.0
            ),
            truth_subscriber_queue_size=getattr(
                self, "truth_subscriber_queue_size", 64
            ),
            truth_processing_queue_size=getattr(
                self, "truth_processing_queue_size", 256
            ),
            truth_processing_enqueue_timeout_sec=getattr(
                self, "truth_processing_enqueue_timeout_sec", 2.0
            ),
            algorithm_storage_backend=getattr(
                self, "algorithm_storage_backend", "jsonl"
            ),
            truth_motion_policy=getattr(self, "truth_motion_policy", None),
        )
        payload["recording_mode"] = getattr(self, "recording_mode", "debug")
        payload["scenario_id"] = getattr(self, "scenario_id", "")
        surface_truth_info = payload["truth_recording"]["surface_truth_points"]
        if getattr(self, "_static_surface_truth_catalog_loaded", False):
            surface_truth_info.update(
                {
                    "catalog_source": "world_static_marker_visual",
                    "source_world_file": str(
                        pathlib.Path(self.surface_truth_world_file)
                        .expanduser()
                        .resolve()
                    ),
                    "expected_point_count": self.surface_truth_expected_count,
                    "max_local_radius_m": self.surface_truth_max_local_radius_m,
                }
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
        if getattr(self, "recording_mode", "debug") == "formal":
            validate_control_scenario_ids(
                getattr(self, "scenario_id", ""),
                discovered_controls,
                evaluated_object_names=getattr(self, "object_metadata", {}).keys(),
            )
        authoritative_controls = select_authoritative_discovered_controls(
            getattr(self, "scenario_id", ""),
            discovered_controls,
            allowed_object_names=getattr(self, "object_metadata", {}).keys(),
        )
        return build_scenario_manifest_payload(
            run_dir=self.run_dir,
            scenario_id=getattr(self, "scenario_id", ""),
            explicit_control=explicit_control,
            discovered_controls=authoritative_controls,
            experiment_factors=getattr(self, "experiment_factors", {}),
            object_metadata=getattr(self, "object_metadata", {}),
        )

    def _refresh_scenario_manifest_if_needed(self):
        manifest_path = self.meta_dir / "scenario_manifest.json"
        try:
            current_payload = self._build_current_scenario_manifest_payload()
        except ValueError as exc:
            self._configuration_error = str(exc)
            rospy.logfatal("Formal recording configuration failed: %s", exc)
            rospy.signal_shutdown(str(exc))
            return False
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
            self.meta_dir / "hardware_manifest.json",
            build_hardware_manifest_payload(),
        )
        self._write_json(
            self.meta_dir / "object_id_catalog.json",
            {
                str(object_id): model_name
                for object_id, model_name in sorted(
                    getattr(self, "object_id_catalog", {}).items()
                )
            },
        )
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
        reference_epoch = int(payload.get("reference_epoch", 0))
        stamp = copy_time_dict(header.get("stamp")) or time_to_dict(rospy.Time.now())
        recorded_at = time_to_dict(rospy.Time.now())
        for region in payload.get("regions", []):
            track_id = int(region.get("track_id", 0))
            track_key = (reference_epoch, track_id)
            state = int(region.get("state", 0))
            confirmed = bool(region.get("confirmed", False))
            previous = track_cache.get(track_key)
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
                "reference_epoch": reference_epoch,
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
                    "reference_epoch": reference_epoch,
                    "region_center": region.get("center"),
                    "confirmed_at":  confirmed_at_sec,
                    "pre_frames":    pre_frames,
                    "post_frames":   [],
                    "window_half":   self._DISP_WINDOW_HALF,
                })

            frame_payload = dict(base_payload)
            frame_payload["event_type"] = "frame_status"
            self._append_jsonl(
                "persistent_track_events",
                "persistent_track_events.jsonl",
                frame_payload,
            )

            track_cache[track_key] = {
                "state": state,
                "first_seen": lifecycle["first_seen"],
                "first_confirmed": lifecycle["first_confirmed"],
                "last_seen": lifecycle["last_seen"],
            }

    def _mark_recording_error(self, error):
        message = str(error)
        if not getattr(self, "_recording_error", ""):
            self._recording_error = message
        logfatal = getattr(rospy, "logfatal", None)
        if callable(logfatal):
            logfatal("Algorithm recording failed: %s", message)
        signal_shutdown = getattr(rospy, "signal_shutdown", None)
        if callable(signal_shutdown):
            signal_shutdown(message)

    def _record_sqlite_commit(self, key, filename, payload, result):
        lock = getattr(self, "_algorithm_write_lock", None)
        if lock is None:
            lock = threading.RLock()
            self._algorithm_write_lock = lock
        with lock:
            self._ensure_recording_runtime_state()
            self._update_stream_stats(key, filename, payload)
            if key in FRAME_COMMIT_STREAM_FILES:
                stamp_sec = float(result["stamp_secs"]) + (
                    float(result["stamp_nsecs"]) / 1.0e9
                )
                self._frame_stream_records.setdefault(key, []).append(
                    {
                        "stamp_sec": stamp_sec,
                        "stamp_key": frame_commit_stamp_key(stamp_sec),
                        "sequence": result.get("header_seq"),
                        "phase": int(payload.get("phase", -1)),
                        "start_offset": None,
                        "end_offset": None,
                        "frame_pk": result.get("frame_pk"),
                        "storage": "sqlite_zlib",
                    }
                )

    def _append_algorithm_payload(self, key, filename, payload):
        backend = normalize_algorithm_storage_backend(
            getattr(self, "algorithm_storage_backend", "jsonl")
        )
        if key not in COMPRESSED_ALGORITHM_STREAMS or backend == "jsonl":
            return self._append_jsonl(key, filename, payload)

        wrote_jsonl = False
        if backend == "dual":
            wrote_jsonl = self._append_jsonl(key, filename, payload)
            if not wrote_jsonl:
                return False

        store = getattr(self, "_algorithm_frame_store", None)
        if store is None:
            error = AlgorithmFrameStoreError(
                "Compressed algorithm storage backend is not initialized"
            )
            self._mark_recording_error(error)
            raise error

        try:
            on_commit = None
            if backend == "sqlite_zlib":
                stats_filename = "{}#{}".format(
                    ALGORITHM_FRAME_DATABASE_FILENAME,
                    key,
                )
                on_commit = lambda result: self._record_sqlite_commit(
                    key,
                    stats_filename,
                    payload,
                    result,
                )
            store.enqueue(key, payload, on_commit=on_commit)
        except AlgorithmFrameStoreError as exc:
            self._mark_recording_error(exc)
            raise
        return wrote_jsonl or True

    def _append_jsonl(self, key, filename, payload):
        lock = getattr(self, "_algorithm_write_lock", None)
        if lock is None:
            lock = threading.RLock()
            self._algorithm_write_lock = lock
        with lock:
            return self._append_jsonl_locked(key, filename, payload)

    def _append_jsonl_locked(self, key, filename, payload):
        self._ensure_recording_runtime_state()
        if getattr(self, "_closed", False):
            return False
        handle = self._algorithm_files.get(key)
        if handle is None:
            handle = (self.algorithm_dir / filename).open("a")
            self._algorithm_files[key] = handle
            self._algorithm_pending_rows[key] = 0
            self._algorithm_last_flush_monotonic[key] = time.monotonic()

        start_offset = handle.tell()
        try:
            json.dump(payload, handle, sort_keys=True)
            handle.write("\n")
        except Exception:
            stats = self._stream_stats.setdefault(key, self._new_stream_stats(filename))
            stats["write_error_count"] += 1
            raise

        end_offset = handle.tell()
        self._update_stream_stats(key, filename, payload)
        if key in FRAME_COMMIT_STREAM_FILES:
            header = payload.get("header", {}) if isinstance(payload, dict) else {}
            raw_seq = header.get("seq") if isinstance(header, dict) else None
            try:
                sequence = int(raw_seq) if raw_seq is not None else None
            except (TypeError, ValueError):
                sequence = None
            self._frame_stream_records.setdefault(key, []).append(
                {
                    "stamp_sec": common_record_time_sec_from_payload(payload),
                    "stamp_key": frame_commit_stamp_key(
                        common_record_time_sec_from_payload(payload)
                    ),
                    "sequence": sequence,
                    "phase": int(payload.get("phase", -1)),
                    "start_offset": start_offset,
                    "end_offset": end_offset,
                }
            )
        self._algorithm_pending_rows[key] += 1
        now_monotonic = time.monotonic()
        pending_limit = max(1, int(getattr(self, "record_flush_max_rows", 1)))
        interval = max(0.001, float(getattr(self, "record_flush_interval_sec", 1.0)))
        if (
            self._algorithm_pending_rows[key] >= pending_limit
            or now_monotonic - self._algorithm_last_flush_monotonic[key] >= interval
        ):
            self._flush_algorithm_stream(key)
        return True

    @staticmethod
    def _new_stream_stats(filename):
        return {
            "filename": str(filename),
            "row_count": 0,
            "first_stamp_sec": None,
            "last_stamp_sec": None,
            "duplicate_stamp_count": 0,
            "non_monotonic_stamp_count": 0,
            "invalid_stamp_count": 0,
            "sequence_gap_count": 0,
            "estimated_drop_count": 0,
            "drop_estimate_available": False,
            "inferred_sequence_stride": None,
            "irregular_sequence_delta_count": 0,
            "sequence_reset_or_reorder_count": 0,
            "write_error_count": 0,
            "flush_count": 0,
        }

    def _ensure_recording_runtime_state(self):
        if not hasattr(self, "_algorithm_files"):
            self._algorithm_files = {}
        if not hasattr(self, "_algorithm_pending_rows"):
            self._algorithm_pending_rows = {}
        if not hasattr(self, "_algorithm_last_flush_monotonic"):
            self._algorithm_last_flush_monotonic = {}
        if not hasattr(self, "_stream_stats"):
            self._stream_stats = {}
        if not hasattr(self, "_stream_last_seq"):
            self._stream_last_seq = {}
        if not hasattr(self, "_stream_sequences"):
            self._stream_sequences = {}
        if not hasattr(self, "_frame_stream_records"):
            self._frame_stream_records = {}
        if not hasattr(self, "_anchor_catalog_cache"):
            self._anchor_catalog_cache = {}

    def _rebuild_trimmed_stream_stats(self, key):
        records = self._frame_stream_records.get(key, [])
        previous_stats = self._stream_stats.get(key, {})
        stats = self._new_stream_stats(
            previous_stats.get("filename", FRAME_COMMIT_STREAM_FILES[key])
        )
        stats["flush_count"] = int(previous_stats.get("flush_count", 0))
        stats["write_error_count"] = int(
            previous_stats.get("write_error_count", 0)
        )
        previous_stamp = None
        sequences = []
        previous_sequence = None
        for record in records:
            stats["row_count"] += 1
            stamp = record.get("stamp_sec")
            if stamp is None:
                stats["invalid_stamp_count"] += 1
            else:
                stamp = float(stamp)
                if stats["first_stamp_sec"] is None:
                    stats["first_stamp_sec"] = stamp
                elif stamp == previous_stamp:
                    stats["duplicate_stamp_count"] += 1
                elif previous_stamp is not None and stamp < previous_stamp:
                    stats["non_monotonic_stamp_count"] += 1
                stats["last_stamp_sec"] = stamp
                previous_stamp = stamp

            sequence = record.get("sequence")
            if sequence is None:
                continue
            sequence = int(sequence)
            if previous_sequence is not None:
                if sequence > previous_sequence + 1:
                    stats["sequence_gap_count"] += 1
                elif sequence < previous_sequence:
                    stats["sequence_reset_or_reorder_count"] += 1
            sequences.append(sequence)
            previous_sequence = sequence

        self._stream_stats[key] = stats
        self._stream_sequences[key] = sequences
        if sequences:
            self._stream_last_seq[key] = sequences[-1]
        else:
            self._stream_last_seq.pop(key, None)

    def _trim_incomplete_frame_suffixes(self):
        summary = {
            "policy": "common_complete_frame_prefix",
            "last_complete_stamp_sec": None,
            "discarded_incomplete_suffix_rows": {},
        }
        plan = build_frame_commit_plan(
            getattr(self, "_frame_stream_records", {})
        )
        if plan is None:
            return summary

        storage_backend = normalize_algorithm_storage_backend(
            getattr(self, "algorithm_storage_backend", "jsonl")
        )
        database_path = getattr(self, "algorithm_dir", pathlib.Path(".")) / (
            ALGORITHM_FRAME_DATABASE_FILENAME
        )
        summary["last_complete_stamp_sec"] = plan["last_complete_stamp_sec"]
        for key, trim_count in sorted(plan["trim_counts"].items()):
            records = self._frame_stream_records.get(key, [])
            keep_count = len(records) - int(trim_count)
            trimmed_representation = False

            if (
                key in COMPRESSED_ALGORITHM_STREAMS
                and storage_backend in ("sqlite_zlib", "dual")
            ):
                try:
                    self._algorithm_storage_summary = trim_sqlite_stream(
                        database_path,
                        key,
                        keep_count,
                    )
                    trimmed_representation = True
                except AlgorithmFrameStoreError as exc:
                    self._mark_recording_error(exc)

            if (
                key not in COMPRESSED_ALGORITHM_STREAMS
                or storage_backend in ("jsonl", "dual")
            ):
                handle = self._algorithm_files.get(key)
                if handle is not None:
                    cutoff = records[keep_count]["start_offset"]
                    handle.flush()
                    handle.truncate(cutoff)
                    self._algorithm_pending_rows[key] = 0
                    trimmed_representation = True

            if trimmed_representation:
                self._frame_stream_records[key] = records[:keep_count]
                self._rebuild_trimmed_stream_stats(key)
                summary["discarded_incomplete_suffix_rows"][key] = int(trim_count)
        return summary

    def _update_stream_stats(self, key, filename, payload):
        stats = self._stream_stats.setdefault(key, self._new_stream_stats(filename))
        stats["row_count"] += 1
        stamp = common_record_time_sec_from_payload(payload)
        if stamp is None:
            stats["invalid_stamp_count"] += 1
        else:
            previous_stamp = stats["last_stamp_sec"]
            if previous_stamp is None:
                stats["first_stamp_sec"] = float(stamp)
            elif stamp == previous_stamp:
                stats["duplicate_stamp_count"] += 1
            elif stamp < previous_stamp:
                stats["non_monotonic_stamp_count"] += 1
            stats["last_stamp_sec"] = float(stamp)

        header = payload.get("header", {}) if isinstance(payload, dict) else {}
        raw_seq = header.get("seq") if isinstance(header, dict) else None
        try:
            sequence = int(raw_seq) if raw_seq is not None else None
        except (TypeError, ValueError):
            sequence = None
        if sequence is not None:
            previous_sequence = self._stream_last_seq.get(key)
            if previous_sequence is not None:
                if sequence > previous_sequence + 1:
                    stats["sequence_gap_count"] += 1
                elif sequence < previous_sequence:
                    stats["sequence_reset_or_reorder_count"] += 1
            self._stream_last_seq[key] = sequence
            self._stream_sequences.setdefault(key, []).append(sequence)

    def _finalize_drop_estimates(self):
        for key, sequences in getattr(self, "_stream_sequences", {}).items():
            stats = self._stream_stats.get(key)
            if stats is None:
                continue
            deltas = [
                right - left
                for left, right in zip(sequences, sequences[1:])
                if right > left
            ]
            if len(deltas) < 2:
                continue
            stride = min(deltas)
            if stride <= 0:
                continue
            estimated_drop_count = 0
            irregular_count = 0
            for delta in deltas:
                if delta <= stride:
                    continue
                quotient, remainder = divmod(delta, stride)
                if remainder == 0:
                    estimated_drop_count += max(0, quotient - 1)
                else:
                    irregular_count += 1
            stats["drop_estimate_available"] = True
            stats["inferred_sequence_stride"] = stride
            stats["estimated_drop_count"] = estimated_drop_count
            stats["irregular_sequence_delta_count"] = irregular_count
        for stats in getattr(self, "_stream_stats", {}).values():
            row_count = int(stats.get("row_count", 0))
            first_stamp = stats.get("first_stamp_sec")
            last_stamp = stats.get("last_stamp_sec")
            effective_rate = None
            if (
                row_count > 1
                and first_stamp is not None
                and last_stamp is not None
                and float(last_stamp) > float(first_stamp)
            ):
                effective_rate = (row_count - 1) / (
                    float(last_stamp) - float(first_stamp)
                )
            stats["effective_rate_hz"] = effective_rate

    def _flush_algorithm_stream(self, key):
        handle = getattr(self, "_algorithm_files", {}).get(key)
        if handle is None:
            return
        handle.flush()
        self._algorithm_pending_rows[key] = 0
        self._algorithm_last_flush_monotonic[key] = time.monotonic()
        stats = self._stream_stats.get(key)
        if stats is not None:
            stats["flush_count"] += 1

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
            if name == self.ego_model_name:
                continue
            tracked.append(name)
        return tracked

    def _is_surface_truth_link(self, scoped_name):
        _, link_name = parse_scoped_link_name(scoped_name)
        leaf_link_name = link_name.rsplit("::", 1)[-1]
        prefixes = getattr(self, "surface_truth_link_prefixes", ("ground_truth_",))
        return any(leaf_link_name.startswith(prefix) for prefix in prefixes)

    def _tracked_link_names(self, msg):
        tracked = []
        motion_drive_links = set(
            getattr(self, "motion_truth_drive_links", {}).values()
        )
        for scoped_name in getattr(msg, "name", []):
            if scoped_name == getattr(self, "sensor_scoped_link_name", ""):
                tracked.append(scoped_name)
            elif scoped_name in motion_drive_links:
                tracked.append(scoped_name)
            elif getattr(self, "record_surface_truth_link_trajectories", False):
                if self._is_surface_truth_link(scoped_name):
                    tracked.append(scoped_name)
        return tracked

    def _object_id_for_model(self, model_name):
        for object_id, catalog_model_name in getattr(self, "object_id_catalog", {}).items():
            if catalog_model_name == model_name:
                return int(object_id)
        return 0

    def _append_surface_truth_point(
        self, scoped_link_name, link_pose, link_time_sec, poses_by_name=None
    ):
        captured = getattr(self, "_captured_surface_truth_links", set())
        self._captured_surface_truth_links = captured
        if scoped_link_name in captured:
            return False

        model_name, link_name = parse_scoped_link_name(scoped_link_name)
        parent_scoped_name = getattr(self, "motion_truth_drive_links", {}).get(
            model_name, ""
        )
        parent_pose = None
        parent_time_sec = None
        if parent_scoped_name and isinstance(poses_by_name, dict):
            parent_message_pose = poses_by_name.get(parent_scoped_name)
            if parent_message_pose is not None:
                parent_pose = pose_to_dict(parent_message_pose)
                parent_time_sec = float(link_time_sec)
        if parent_pose is None:
            root_entry = getattr(self, "_latest_model_pose_world", {}).get(model_name)
            if not isinstance(root_entry, dict):
                return False
            parent_pose = root_entry.get("pose")
            parent_time_sec = coerce_float(
                root_entry.get("recorded_time_sec"), None
            )
            parent_scoped_name = model_name
        if parent_time_sec is None or not pose_dict_is_finite(parent_pose):
            return False

        link_pose_world = pose_to_dict(link_pose)
        if not pose_dict_is_finite(link_pose_world):
            return False
        local_pose = compose_pose_dicts(
            invert_pose_dict(parent_pose), link_pose_world
        )
        object_id = self._object_id_for_model(model_name)
        payload = {
            "schema_version": 1,
            "scoped_link_name": str(scoped_link_name),
            "model_name": model_name,
            "link_name": link_name,
            "object_id": object_id,
            "object_id_valid": object_id > 0,
            "truth_frame": self.truth_frame,
            "object_local_frame": parent_scoped_name,
            "motion_parent_scoped_link_name": parent_scoped_name,
            "motion_parent_pose_sample_time_sec": float(parent_time_sec),
            "link_pose_sample_time_sec": float(link_time_sec),
            "pose_pair_delta_sec": abs(
                float(link_time_sec) - float(parent_time_sec)
            ),
            "local_pose": local_pose,
            "motion_parent_pose_world_at_capture": parent_pose,
            "link_pose_world_at_capture": link_pose_world,
        }

        self._enqueue_truth_rows("surface_truth", [payload])
        captured.add(scoped_link_name)
        return True

    def _capture_surface_truth_points(self, msg, poses_by_name, recorded_time_sec):
        if getattr(self, "_static_surface_truth_catalog_loaded", False):
            return
        for scoped_link_name in getattr(msg, "name", []):
            if not self._is_surface_truth_link(scoped_link_name):
                continue
            pose = poses_by_name.get(scoped_link_name)
            if pose is not None:
                self._append_surface_truth_point(
                    scoped_link_name,
                    pose,
                    recorded_time_sec,
                    poses_by_name,
                )

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

    def _ensure_truth_pipeline_state(self):
        if not hasattr(self, "_truth_pipeline_lock"):
            self._truth_pipeline_lock = threading.Lock()
        if not hasattr(self, "_truth_pipeline_stats"):
            self._truth_pipeline_stats = {
                "model_states": new_truth_pipeline_stats(
                    getattr(self, "truth_object_rate_hz", 10.0)
                ),
                "link_states": new_truth_pipeline_stats(
                    getattr(self, "truth_link_rate_hz", 10.0)
                ),
                "surface_truth": new_truth_pipeline_stats(),
            }
        if not hasattr(self, "_truth_write_error"):
            self._truth_write_error = None

    def _note_truth_callback(self, stream_name, timestamp_sec):
        self._ensure_truth_pipeline_state()
        with self._truth_pipeline_lock:
            update_truth_callback_stats(
                self._truth_pipeline_stats[stream_name], timestamp_sec
            )

    def _flush_truth_files(self):
        for handle, _ in getattr(self, "_object_files", {}).values():
            handle.flush()
        for handle, _ in getattr(self, "_link_files", {}).values():
            handle.flush()
        surface_handle = getattr(self, "_surface_truth_catalog_handle", None)
        if surface_handle is not None:
            surface_handle.flush()
        self._ensure_truth_pipeline_state()
        with self._truth_pipeline_lock:
            for stats in self._truth_pipeline_stats.values():
                stats["flush_count"] = int(stats.get("flush_count", 0)) + 1

    def _write_truth_job(self, stream_name, rows):
        if stream_name == "model_states":
            for model_name, row in rows:
                _, writer = self._object_writer(model_name)
                writer.writerow(row)
        elif stream_name == "link_states":
            for scoped_link_name, row in rows:
                _, writer = self._link_writer(scoped_link_name)
                writer.writerow(row)
        elif stream_name == "surface_truth":
            handle = getattr(self, "_surface_truth_catalog_handle", None)
            if handle is None:
                catalog_path = self.truth_dir / "surface_truth_points.jsonl"
                handle = catalog_path.open("a")
                self._surface_truth_catalog_handle = handle
            for payload in rows:
                json.dump(payload, handle, sort_keys=True)
                handle.write("\n")
        else:
            raise ValueError("Unknown truth stream: %s" % stream_name)

        self._ensure_truth_pipeline_state()
        with self._truth_pipeline_lock:
            stats = self._truth_pipeline_stats[stream_name]
            stats["written_batch_count"] = int(
                stats.get("written_batch_count", 0)
            ) + 1
            stats["written_row_count"] = int(stats.get("written_row_count", 0)) + len(
                rows
            )

    def _start_truth_write_worker(self):
        worker = getattr(self, "_truth_write_thread", None)
        if worker is not None and worker.is_alive():
            return
        self._ensure_truth_pipeline_state()
        self._truth_write_queue = queue.Queue(
            maxsize=max(1, int(getattr(self, "truth_processing_queue_size", 256)))
        )
        self._truth_write_error = None
        worker = threading.Thread(
            target=self._run_truth_write_worker,
            name="sim-recorder-truth-writer",
            daemon=False,
        )
        self._truth_write_thread = worker
        worker.start()

    def _run_truth_write_worker(self):
        processing_queue = self._truth_write_queue
        pending_rows = 0
        last_flush = time.monotonic()
        flush_interval = max(
            0.001, float(getattr(self, "record_flush_interval_sec", 1.0))
        )
        flush_limit = max(1, int(getattr(self, "record_flush_max_rows", 100)))
        while True:
            try:
                item = processing_queue.get(timeout=min(0.25, flush_interval))
            except queue.Empty:
                if pending_rows and time.monotonic() - last_flush >= flush_interval:
                    self._flush_truth_files()
                    pending_rows = 0
                    last_flush = time.monotonic()
                continue
            try:
                if item is TRUTH_WRITE_STOP:
                    if pending_rows:
                        self._flush_truth_files()
                    return
                stream_name, rows = item
                self._write_truth_job(stream_name, rows)
                pending_rows += len(rows)
                if (
                    pending_rows >= flush_limit
                    or time.monotonic() - last_flush >= flush_interval
                ):
                    self._flush_truth_files()
                    pending_rows = 0
                    last_flush = time.monotonic()
            except BaseException as exc:
                first_error = self._truth_write_error is None
                self._truth_write_error = exc
                self._ensure_truth_pipeline_state()
                stream_name = item[0] if isinstance(item, tuple) else "model_states"
                with self._truth_pipeline_lock:
                    stats = self._truth_pipeline_stats.get(stream_name)
                    if stats is not None:
                        stats["write_error_count"] = int(
                            stats.get("write_error_count", 0)
                        ) + 1
                if first_error:
                    threading.Thread(
                        target=self._mark_recording_error,
                        args=(exc,),
                        name="sim-recorder-truth-error",
                        daemon=True,
                    ).start()
            finally:
                processing_queue.task_done()

    def _enqueue_truth_rows(self, stream_name, rows):
        self._ensure_truth_pipeline_state()
        rows = list(rows)
        processing_queue = getattr(self, "_truth_write_queue", None)
        worker = getattr(self, "_truth_write_thread", None)
        with self._truth_pipeline_lock:
            stats = self._truth_pipeline_stats[stream_name]
            stats["sampled_batch_count"] = int(
                stats.get("sampled_batch_count", 0)
            ) + 1

        # Pure-Python helper tests construct the recorder with __new__. Keep a
        # synchronous fallback for those isolated instances only.
        if processing_queue is None or worker is None:
            with self._truth_pipeline_lock:
                stats["enqueued_batch_count"] += 1
                stats["enqueued_row_count"] += len(rows)
            self._write_truth_job(stream_name, rows)
            self._flush_truth_files()
            return
        if not worker.is_alive():
            error = RuntimeError("Truth recording worker stopped unexpectedly")
            self._mark_recording_error(error)
            raise error
        try:
            processing_queue.put(
                (stream_name, rows),
                block=True,
                timeout=float(
                    getattr(self, "truth_processing_enqueue_timeout_sec", 2.0)
                ),
            )
        except queue.Full as exc:
            with self._truth_pipeline_lock:
                stats["queue_full_error_count"] += 1
            error = RuntimeError(
                "Truth processing queue remained full for {:.3f} s".format(
                    float(
                        getattr(self, "truth_processing_enqueue_timeout_sec", 2.0)
                    )
                )
            )
            self._mark_recording_error(error)
            raise error from exc
        with self._truth_pipeline_lock:
            stats["enqueued_batch_count"] += 1
            stats["enqueued_row_count"] += len(rows)
            stats["max_queue_depth"] = max(
                int(stats.get("max_queue_depth", 0)), processing_queue.qsize()
            )

    def _close_truth_write_worker(self):
        processing_queue = getattr(self, "_truth_write_queue", None)
        worker = getattr(self, "_truth_write_thread", None)
        if processing_queue is None or worker is None:
            return
        if threading.current_thread() is worker:
            raise RuntimeError("Truth recording worker cannot close itself")

        stop_enqueued = False
        while worker.is_alive() and not stop_enqueued:
            try:
                processing_queue.put(TRUTH_WRITE_STOP, block=True, timeout=0.1)
                stop_enqueued = True
            except queue.Full:
                continue
        worker.join()
        if not stop_enqueued:
            error = RuntimeError("Truth recording worker stopped before queue drain")
            self._mark_recording_error(error)
        self._truth_write_thread = None
        self._truth_write_queue = None

    @recording_callback
    def _handle_model_states(self, msg):
        recorded_time_sec = rospy.Time.now().to_sec()
        if not valid_sim_recording_time(recorded_time_sec):
            rospy.logwarn_throttle(
                5.0,
                "sim_experiment_recorder is waiting for a positive /clock before recording model truth.",
            )
            return
        self._note_truth_callback("model_states", recorded_time_sec)
        self._refresh_scenario_manifest_if_needed()

        poses_by_name = dict(zip(getattr(msg, "name", []), getattr(msg, "pose", [])))
        twists_by_name = dict(zip(getattr(msg, "name", []), getattr(msg, "twist", [])))
        latest_model_pose_world = getattr(self, "_latest_model_pose_world", {})
        self._latest_model_pose_world = latest_model_pose_world
        for model_name, pose in poses_by_name.items():
            latest_model_pose_world[model_name] = {
                "pose": pose_to_dict(pose),
                "recorded_time_sec": float(recorded_time_sec),
            }

        ego_pose = poses_by_name.get(self.ego_model_name)
        if ego_pose is not None:
            self._write_ego_initial_pose(ego_pose)

        should_sample, next_sample_time, last_observed_time = advance_sampling_clock(
            recorded_time_sec,
            getattr(self, "_next_model_states_sample_time", None),
            getattr(self, "_last_model_states_observed_time", None),
            getattr(self, "truth_object_rate_hz", 10.0),
        )
        self._next_model_states_sample_time = next_sample_time
        self._last_model_states_observed_time = last_observed_time
        sampling_stats = getattr(self, "_truth_sampling_stats", None)
        if sampling_stats is None:
            sampling_stats = {
                "model_states": new_sampling_stats(
                    getattr(self, "truth_object_rate_hz", 10.0)
                ),
                "link_states": new_sampling_stats(
                    getattr(self, "truth_link_rate_hz", 10.0)
                ),
            }
            self._truth_sampling_stats = sampling_stats
        tracked_model_names = self._tracked_model_names(msg) if should_sample else []
        update_sampling_stats(
            sampling_stats["model_states"],
            recorded_time_sec,
            should_sample,
            rows_written=sum(name in poses_by_name for name in tracked_model_names),
        )
        if not should_sample:
            return
        self._last_model_states_write_time = recorded_time_sec

        rows = []
        for model_name in tracked_model_names:
            pose = poses_by_name.get(model_name)
            if pose is None:
                continue
            twist = twists_by_name.get(model_name)
            linear = getattr(twist, "linear", None)
            angular = getattr(twist, "angular", None)

            rows.append(
                (
                    model_name,
                    [
                    "%.9f" % float(recorded_time_sec),
                    model_name,
                    self.truth_frame,
                    self.truth_frame,
                    pose.position.x,
                    pose.position.y,
                    pose.position.z,
                    pose.orientation.x,
                    pose.orientation.y,
                    pose.orientation.z,
                    pose.orientation.w,
                    getattr(linear, "x", ""),
                    getattr(linear, "y", ""),
                    getattr(linear, "z", ""),
                    getattr(angular, "x", ""),
                    getattr(angular, "y", ""),
                    getattr(angular, "z", ""),
                    ],
                )
            )
        self._enqueue_truth_rows("model_states", rows)

    @recording_callback
    def _handle_link_states(self, msg):
        recorded_time_sec = rospy.Time.now().to_sec()
        if not valid_sim_recording_time(recorded_time_sec):
            rospy.logwarn_throttle(
                5.0,
                "sim_experiment_recorder is waiting for a positive /clock before recording link truth.",
            )
            return
        self._note_truth_callback("link_states", recorded_time_sec)
        poses_by_name = dict(zip(getattr(msg, "name", []), getattr(msg, "pose", [])))

        sensor_pose = poses_by_name.get(self.sensor_scoped_link_name)
        if sensor_pose is not None:
            self._latest_sensor_pose_world = pose_to_dict(sensor_pose)
            self._latest_sensor_pose_stamp = recorded_time_sec

        self._capture_surface_truth_points(msg, poses_by_name, recorded_time_sec)

        should_sample, next_sample_time, last_observed_time = advance_sampling_clock(
            recorded_time_sec,
            getattr(self, "_next_link_states_sample_time", None),
            getattr(self, "_last_link_states_observed_time", None),
            getattr(self, "truth_link_rate_hz", 10.0),
        )
        self._next_link_states_sample_time = next_sample_time
        self._last_link_states_observed_time = last_observed_time
        sampling_stats = getattr(self, "_truth_sampling_stats", None)
        if sampling_stats is None:
            sampling_stats = {
                "model_states": new_sampling_stats(
                    getattr(self, "truth_object_rate_hz", 10.0)
                ),
                "link_states": new_sampling_stats(
                    getattr(self, "truth_link_rate_hz", 10.0)
                ),
            }
            self._truth_sampling_stats = sampling_stats
        tracked_link_names = self._tracked_link_names(msg) if should_sample else []
        update_sampling_stats(
            sampling_stats["link_states"],
            recorded_time_sec,
            should_sample,
            rows_written=sum(name in poses_by_name for name in tracked_link_names),
        )
        if not should_sample:
            return
        self._last_link_states_write_time = recorded_time_sec

        rows = []
        for scoped_link_name in tracked_link_names:
            pose = poses_by_name.get(scoped_link_name)
            if pose is None:
                continue

            model_name, link_name = parse_scoped_link_name(scoped_link_name)
            rows.append(
                (
                    scoped_link_name,
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
                    ],
                )
            )
        self._enqueue_truth_rows("link_states", rows)

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

    @recording_callback
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
            truth_reference_stamp_sec=truth_reference_pose_stamp_sec,
            algorithm_reference_stamp_sec=odom_stamp_sec,
        )
        self._write_json(getattr(self, "meta_dir", pathlib.Path(".")) / "frame_alignment.json", metadata)
        self._frame_alignment_written = True
        return True

    @recording_callback
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

    @recording_callback
    def _handle_risk_evidence(self, msg):
        payload = serialize_risk_evidence(msg)
        payload["recorded_at"] = time_to_dict(rospy.Time.now())
        self._append_algorithm_payload(
            "risk_evidence",
            "risk_evidence.jsonl",
            payload,
        )

    @recording_callback
    def _handle_object_observation_stats(self, msg):
        payload = serialize_object_observation_stats(msg)
        payload["recorded_at"] = time_to_dict(rospy.Time.now())
        self._append_jsonl(
            "object_observation_stats",
            "object_observation_stats.jsonl",
            payload,
        )

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
            "reference_epoch": entry.get("reference_epoch", 0),
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

    @recording_callback
    def _handle_clusters(self, msg):
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

        self._append_algorithm_payload("clusters", "clusters.jsonl", payload)

    @recording_callback
    def _handle_risk_regions(self, msg):
        payload = serialize_risk_regions(msg)
        payload["recorded_at"] = time_to_dict(rospy.Time.now())
        self._append_jsonl("risk_regions", "risk_regions.jsonl", payload)

    @recording_callback
    def _handle_persistent_risk_regions(self, msg):
        self._ensure_algorithm_runtime_state()
        payload = serialize_persistent_risk_regions(msg)
        payload["recorded_at"] = time_to_dict(rospy.Time.now())
        self._append_jsonl(
            "persistent_risk_regions",
            "persistent_risk_regions.jsonl",
            payload,
        )
        self._append_persistent_track_events(payload)

    @recording_callback
    def _handle_structure_motions(self, msg):
        payload = serialize_structure_motions(msg)
        payload["recorded_at"] = time_to_dict(rospy.Time.now())
        self._append_jsonl("structure_motions", "structure_motions.jsonl", payload)

    def _start_anchor_processing_worker(self):
        worker = getattr(self, "_anchor_processing_thread", None)
        if worker is not None and worker.is_alive():
            return
        processing_queue = queue.Queue(
            maxsize=max(1, int(getattr(self, "anchor_processing_queue_size", 64)))
        )
        self._anchor_processing_queue = processing_queue
        self._anchor_processing_error = None
        worker = threading.Thread(
            target=self._run_anchor_processing_worker,
            name="sim-recorder-anchor-processor",
            daemon=False,
        )
        self._anchor_processing_thread = worker
        worker.start()

    def _run_anchor_processing_worker(self):
        processing_queue = self._anchor_processing_queue
        while True:
            item = processing_queue.get()
            try:
                if item is ANCHOR_PROCESSING_STOP:
                    return
                msg, recorded_at = item
                try:
                    self._record_anchor_states(msg, recorded_at)
                    stats = getattr(self, "_anchor_processing_stats", None)
                    if stats is not None:
                        stats["processed_message_count"] += 1
                except BaseException as exc:
                    first_error = self._anchor_processing_error is None
                    self._anchor_processing_error = exc
                    stats = getattr(self, "_anchor_processing_stats", None)
                    if stats is not None:
                        stats["processing_error_count"] += 1
                    if first_error:
                        # signal_shutdown may invoke close() synchronously, so report
                        # from a different thread and let close() join this worker.
                        threading.Thread(
                            target=self._mark_recording_error,
                            args=(exc,),
                            name="sim-recorder-anchor-error",
                            daemon=True,
                        ).start()
            finally:
                processing_queue.task_done()

    def _enqueue_anchor_states(self, msg, recorded_at):
        processing_queue = getattr(self, "_anchor_processing_queue", None)
        worker = getattr(self, "_anchor_processing_thread", None)
        if processing_queue is None or worker is None:
            self._record_anchor_states(msg, recorded_at)
            return
        if not worker.is_alive():
            error = RuntimeError("Anchor recording worker stopped unexpectedly")
            self._mark_recording_error(error)
            raise error
        try:
            processing_queue.put(
                (msg, recorded_at),
                block=True,
                timeout=float(
                    getattr(self, "anchor_processing_enqueue_timeout_sec", 2.0)
                ),
            )
        except queue.Full as exc:
            stats = getattr(self, "_anchor_processing_stats", None)
            if stats is not None:
                stats["queue_full_error_count"] += 1
            error = RuntimeError(
                "Anchor processing queue remained full for {:.3f} s".format(
                    float(
                        getattr(self, "anchor_processing_enqueue_timeout_sec", 2.0)
                    )
                )
            )
            self._mark_recording_error(error)
            raise error from exc

        stats = getattr(self, "_anchor_processing_stats", None)
        if stats is not None:
            stats["enqueued_message_count"] += 1
            stats["max_queue_depth"] = max(
                int(stats["max_queue_depth"]), processing_queue.qsize()
            )

    @recording_callback
    def _handle_anchor_states(self, msg):
        self._enqueue_anchor_states(msg, time_to_dict(rospy.Time.now()))

    def _record_anchor_states(self, msg, recorded_at):
        self._ensure_recording_runtime_state()
        header = serialize_header(msg)
        epoch = int(getattr(msg, "reference_epoch", 0))
        reference_initialized_at = time_to_dict(
            getattr(msg, "reference_initialized_at", None)
        )
        dynamic_anchors = []
        for anchor in getattr(msg, "anchors", []):
            static = serialize_anchor_static_state(anchor)
            dynamic = serialize_anchor_dynamic_state(anchor)
            anchor_id = int(static.get("id", 0))
            key = (epoch, anchor_id)
            existing = self._anchor_catalog_cache.get(key)
            if existing is None:
                self._anchor_catalog_cache[key] = static
                self._append_jsonl(
                    "anchor_catalog",
                    "anchor_catalog.jsonl",
                    {
                        "schema_version": 2,
                        "reference_epoch": epoch,
                        "anchor_id": anchor_id,
                        "reference_initialized_at": reference_initialized_at,
                        "source_header": header,
                        "recorded_at": recorded_at,
                        "anchor": static,
                    },
                )
            elif existing != static:
                self._append_jsonl(
                    "anchor_catalog_conflicts",
                    "anchor_catalog_conflicts.jsonl",
                    {
                        "schema_version": 2,
                        "reference_epoch": epoch,
                        "anchor_id": anchor_id,
                        "source_header": header,
                        "recorded_at": recorded_at,
                        "first_anchor": existing,
                        "conflicting_anchor": static,
                    },
                )
            dynamic_anchors.append(dynamic)

        total_anchor_count = int(getattr(msg, "total_anchor_count", 0))
        object_summaries = [
            serialize_anchor_object_summary(s)
            for s in getattr(msg, "object_summaries", [])
        ]
        observation_payload = {
            "schema_version": 3,
            "header": header,
            "reference_epoch": epoch,
            "reference_initialized_at": reference_initialized_at,
            "recorded_at": recorded_at,
            "total_anchor_count": total_anchor_count,
            "object_summaries": object_summaries,
            "anchors": dynamic_anchors,
        }
        self._append_algorithm_payload(
            "anchor_observations",
            "anchor_observations.jsonl",
            observation_payload,
        )
        self._append_jsonl(
            "processing_stamps",
            "processing_stamps.jsonl",
            {
                "schema_version": 3,
                "header": header,
                "reference_epoch": epoch,
                "anchor_count": len(dynamic_anchors),
                "total_anchor_count": total_anchor_count,
                "object_summaries": object_summaries,
                "recorded_at": recorded_at,
            },
        )

    def _close_anchor_processing_worker(self):
        processing_queue = getattr(self, "_anchor_processing_queue", None)
        worker = getattr(self, "_anchor_processing_thread", None)
        if processing_queue is None or worker is None:
            return
        if threading.current_thread() is worker:
            raise RuntimeError("Anchor processing worker cannot close itself")

        stop_enqueued = False
        while worker.is_alive() and not stop_enqueued:
            try:
                processing_queue.put(
                    ANCHOR_PROCESSING_STOP,
                    block=True,
                    timeout=0.1,
                )
                stop_enqueued = True
            except queue.Full:
                continue
        worker.join()
        if not stop_enqueued:
            error = RuntimeError("Anchor recording worker stopped before queue drain")
            if not getattr(self, "_recording_error", ""):
                self._recording_error = str(error)
        self._anchor_processing_thread = None
        self._anchor_processing_queue = None

    def _ensure_close_state(self):
        condition = getattr(self, "_callback_condition", None)
        if condition is None:
            condition = threading.Condition()
            self._callback_condition = condition
        if not hasattr(self, "_active_callbacks"):
            self._active_callbacks = 0
        if not hasattr(self, "_callback_thread_counts"):
            self._callback_thread_counts = {}
        if not hasattr(self, "_closing"):
            self._closing = False
        if not hasattr(self, "_closed"):
            self._closed = False
        if not hasattr(self, "_deferred_close"):
            self._deferred_close = False
        if not hasattr(self, "_finish_close_started"):
            self._finish_close_started = False
        if not hasattr(self, "_finish_close_done"):
            self._finish_close_done = False
        return condition

    def _run_close_finalization(self):
        condition = self._ensure_close_state()
        try:
            self._finish_close()
        finally:
            with condition:
                self._finish_close_done = True
                condition.notify_all()

    def close(self):
        condition = self._ensure_close_state()
        thread_id = threading.get_ident()
        with condition:
            if self._finish_close_done:
                return
            first_closer = not self._closing
            self._closing = True
            called_from_callback = self._callback_thread_counts.get(thread_id, 0) > 0
            if called_from_callback:
                self._deferred_close = True

        if first_closer:
            subscribers = list(getattr(self, "_subscribers", []))
            self._subscribers = []
            for subscriber in subscribers:
                try:
                    subscriber.unregister()
                except Exception:
                    pass

        if called_from_callback:
            return

        with condition:
            while self._active_callbacks > 0:
                condition.wait()
            if self._finish_close_done:
                return
            if self._finish_close_started:
                while not self._finish_close_done:
                    condition.wait()
                return
            self._finish_close_started = True
            self._closed = True

        self._run_close_finalization()

    def _finish_close(self):
        try:
            self._close_truth_write_worker()
        except Exception as exc:
            self._mark_recording_error(exc)
        try:
            self._close_anchor_processing_worker()
        except Exception as exc:
            self._mark_recording_error(exc)

        storage_backend = normalize_algorithm_storage_backend(
            getattr(self, "algorithm_storage_backend", "jsonl")
        )
        frame_store = getattr(self, "_algorithm_frame_store", None)
        if frame_store is not None:
            try:
                self._algorithm_storage_summary = frame_store.close()
            except AlgorithmFrameStoreError as exc:
                self._mark_recording_error(exc)
            finally:
                self._algorithm_frame_store = None

        for handle, _ in getattr(self, "_object_files", {}).values():
            handle.flush()
            handle.close()
        self._object_files = {}

        for handle, _ in getattr(self, "_link_files", {}).values():
            handle.flush()
            handle.close()
        self._link_files = {}

        surface_truth_handle = getattr(self, "_surface_truth_catalog_handle", None)
        if surface_truth_handle is not None:
            surface_truth_handle.close()
            self._surface_truth_catalog_handle = None

        for key in list(getattr(self, "_algorithm_files", {})):
            self._flush_algorithm_stream(key)
        frame_commit_summary = self._trim_incomplete_frame_suffixes()
        self._finalize_drop_estimates()

        meta_dir = getattr(self, "meta_dir", None)
        if meta_dir is not None and pathlib.Path(meta_dir).is_dir():
            truth_write_error = getattr(self, "_truth_write_error", None)
            if truth_write_error is not None and not getattr(
                self, "_recording_error", ""
            ):
                self._recording_error = str(truth_write_error)
            recording_error = getattr(self, "_recording_error", "")
            configuration_error = getattr(self, "_configuration_error", "")
            storage_summary = getattr(self, "_algorithm_storage_summary", None)
            storage_integrity = (
                storage_summary.get("integrity_check")
                if isinstance(storage_summary, dict)
                else None
            )
            recording_integrity_valid = not bool(recording_error) and (
                storage_backend == "jsonl" or storage_integrity == "ok"
            )
            self._ensure_truth_pipeline_state()
            with self._truth_pipeline_lock:
                truth_pipeline = {
                    str(key): finalize_truth_pipeline_stats(value)
                    for key, value in sorted(self._truth_pipeline_stats.items())
                }
            truth_pipeline_valid = all(
                stats.get("lossless_after_enqueue", False)
                for stats in truth_pipeline.values()
            )
            recording_integrity_valid = (
                recording_integrity_valid and truth_pipeline_valid
            )
            completion_payload = {
                "schema_version": 3 if storage_backend != "jsonl" else 2,
                "clean_shutdown": not bool(configuration_error)
                and recording_integrity_valid,
                "configuration_valid": not bool(configuration_error),
                "configuration_error": configuration_error,
                "recording_integrity_valid": recording_integrity_valid,
                "recording_error": recording_error,
                "algorithm_storage": {
                    "backend": storage_backend,
                    "database_file": (
                        "algorithm/{}".format(ALGORITHM_FRAME_DATABASE_FILENAME)
                        if storage_backend != "jsonl"
                        else None
                    ),
                    "integrity_check": storage_integrity,
                    "streams": (
                        storage_summary.get("streams", {})
                        if isinstance(storage_summary, dict)
                        else {}
                    ),
                },
                "closed_at_iso": dt.datetime.now().isoformat(),
                "closed_at": time_to_dict(rospy.Time.now()) if rospy is not None else None,
                "pending_displacement_window_count": len(
                    getattr(self, "_disp_window_pending", [])
                ),
                "anchor_processing": {
                    **dict(getattr(self, "_anchor_processing_stats", {})),
                    "worker_error": (
                        str(getattr(self, "_anchor_processing_error", None))
                        if getattr(self, "_anchor_processing_error", None) is not None
                        else None
                    ),
                },
                "frame_commit": frame_commit_summary,
                "streams": {
                    str(key): dict(value)
                    for key, value in sorted(
                        getattr(self, "_stream_stats", {}).items()
                    )
                },
                "truth_sampling": {
                    str(key): finalize_sampling_stats(value)
                    for key, value in sorted(
                        getattr(self, "_truth_sampling_stats", {}).items()
                    )
                },
                "truth_pipeline": {
                    "subscriber_queue_size": int(
                        getattr(self, "truth_subscriber_queue_size", 1)
                    ),
                    "processing_queue_size": int(
                        getattr(self, "truth_processing_queue_size", 1)
                    ),
                    "processing_enqueue_timeout_sec": float(
                        getattr(self, "truth_processing_enqueue_timeout_sec", 2.0)
                    ),
                    "worker_error": (
                        str(truth_write_error) if truth_write_error is not None else None
                    ),
                    "lossless_after_enqueue": truth_pipeline_valid,
                    "streams": truth_pipeline,
                },
            }
            self._write_json(pathlib.Path(meta_dir) / "run_complete.json", completion_payload)

        for handle in getattr(self, "_algorithm_files", {}).values():
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
