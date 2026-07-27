#!/usr/bin/env python3

"""Strict integrity gate for schema-v2/v3 ALERT simulation recordings."""

import argparse
import csv
import importlib.util
import json
import math
import pathlib
import statistics

import numpy as np


try:
    from algorithm_frame_store import (
        AlgorithmFrameStoreError,
        CompressedFrameStore,
        HIGH_VOLUME_STREAMS,
        ReplayableAlgorithmStream,
        STREAM_ITEM_KEYS,
        compare_dual_storage,
        inspect_sqlite_store,
        iter_algorithm_stream,
        normalize_storage_backend,
    )
except ImportError:  # source-tree execution and importlib-based unit tests
    _storage_module_path = pathlib.Path(__file__).with_name("algorithm_frame_store.py")
    _storage_spec = importlib.util.spec_from_file_location(
        "validate_recorded_run_algorithm_frame_store", _storage_module_path
    )
    _storage_module = importlib.util.module_from_spec(_storage_spec)
    _storage_spec.loader.exec_module(_storage_module)
    AlgorithmFrameStoreError = _storage_module.AlgorithmFrameStoreError
    CompressedFrameStore = _storage_module.CompressedFrameStore
    HIGH_VOLUME_STREAMS = _storage_module.HIGH_VOLUME_STREAMS
    ReplayableAlgorithmStream = _storage_module.ReplayableAlgorithmStream
    STREAM_ITEM_KEYS = _storage_module.STREAM_ITEM_KEYS
    compare_dual_storage = _storage_module.compare_dual_storage
    inspect_sqlite_store = _storage_module.inspect_sqlite_store
    iter_algorithm_stream = _storage_module.iter_algorithm_stream
    normalize_storage_backend = _storage_module.normalize_storage_backend


STREAM_FILES = {
    "anchor_catalog": "anchor_catalog.jsonl",
    "anchor_observations": "anchor_observations.jsonl",
    "processing_stamps": "processing_stamps.jsonl",
    "object_observation_stats": "object_observation_stats.jsonl",
    "clusters": "clusters.jsonl",
    "risk_evidence": "risk_evidence.jsonl",
    "risk_regions": "risk_regions.jsonl",
    "persistent_risk_regions": "persistent_risk_regions.jsonl",
    "structure_motions": "structure_motions.jsonl",
}
FRAME_STREAMS = set(STREAM_FILES) - {"anchor_catalog"}
REQUIRED_TRUTH_COLUMNS = {
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
}
REQUIRED_LINK_TRUTH_COLUMNS = {
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
}
ALIGNMENT_MAX_DELTA_SEC = 0.25
STAMP_TOLERANCE_SEC = 1.0e-6
QUATERNION_REPAIR_TOLERANCE = 1.0e-3
CURRENT_MOTION_PROTOCOL_VERSION = "alert10obj_mixed_rotation_v3"
CURRENT_TRUTH_PROTOCOL_VERSION = "gazebo_drive_link_pose_static_surface_catalog_v3"
CURRENT_SURFACE_TRUTH_CONTRACT = {
    1: ("l_shape_target", 16),
    2: ("model_01", 8),
    3: ("model_02", 8),
    4: ("table", 12),
    5: ("table_marble", 8),
    6: ("person_walking", 10),
    7: ("table_clone", 12),
    8: ("cardboard_box", 8),
    9: ("lzg_bianwood", 8),
    10: ("bookshelf", 16),
}
CURRENT_DRIVE_LINK_CONTRACT = {
    "l_shape_target": "l_shape_target::link",
    "model_01": "model_01::link",
    "model_02": "model_02::link",
    "table": "table::link",
    "table_marble": "table_marble::link",
    "person_walking": "person_walking::link",
    "table_clone": "table_clone::link",
    "cardboard_box": "cardboard_box::link",
    "lzg_bianwood": "lzg_bianwood::link_0",
    "bookshelf": "bookshelf::link",
}
MOTION_DIRECTION_COSINE_MIN = 0.95
MOTION_TRANSLATION_ABS_TOLERANCE_M = 0.002
MOTION_ROTATION_ABS_TOLERANCE_RAD = 0.003
MOTION_RELATIVE_TOLERANCE = 0.15
MOTION_MAX_UNCOMMANDED_ROTATION_RAD = 0.005
MOTION_MIN_ROTATING_SURFACE_DISPLACEMENT_M = 0.001
VALIDATION_POLICY_STRICT = "strict"
VALIDATION_POLICY_FORMAL_ANALYSIS_V2 = "formal_analysis_v2"
VALIDATION_POLICY_RECORDING_V2 = "recording_v2"
VALIDATION_POLICIES = {
    VALIDATION_POLICY_STRICT,
    VALIDATION_POLICY_FORMAL_ANALYSIS_V2,
    VALIDATION_POLICY_RECORDING_V2,
}
DEFAULT_MAX_ANCHOR_PROCESSING_DROP_FRACTION = 0.10


class _NonFiniteJson(ValueError):
    pass


class ScannedCompressedStream:
    """Validation summary for one fully decoded high-volume stream."""

    def __init__(self, stream_name, stamps):
        self.stream_name = str(stream_name)
        self.stamps = tuple(stamps)
        self.stamp_keys = frozenset(
            round(stamp / STAMP_TOLERANCE_SEC) for stamp in stamps
        )

    def __len__(self):
        return len(self.stamps)


def _reject_nonfinite_constant(token):
    raise _NonFiniteJson(token)


def _contains_nonfinite(value):
    if isinstance(value, float):
        return not math.isfinite(value)
    if isinstance(value, dict):
        return any(_contains_nonfinite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_nonfinite(item) for item in value)
    return False


def _strict_json_loads(text):
    value = json.loads(text, parse_constant=_reject_nonfinite_constant)
    if _contains_nonfinite(value):
        raise _NonFiniteJson("overflow")
    return value


def time_sec(value):
    if isinstance(value, (int, float)):
        result = float(value)
        return result if math.isfinite(result) else None
    if not isinstance(value, dict):
        return None
    if "sec" in value:
        try:
            result = float(value["sec"])
        except (TypeError, ValueError):
            return None
        return result if math.isfinite(result) else None
    try:
        result = float(value.get("secs", 0)) + float(value.get("nsecs", 0)) / 1e9
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def record_time_sec(record):
    header = record.get("header", {}) if isinstance(record, dict) else {}
    stamp = header.get("stamp") if isinstance(header, dict) else None
    result = time_sec(stamp)
    return result if result is not None else time_sec(record.get("recorded_at"))


def _relative(path, run_dir):
    try:
        return str(pathlib.Path(path).relative_to(run_dir))
    except ValueError:
        return str(path)


def load_json_file(path, run_dir, errors, required=True):
    path = pathlib.Path(path)
    label = _relative(path, run_dir)
    if not path.is_file():
        if required:
            errors.append(f"missing_file:{label}")
        return None
    try:
        return _strict_json_loads(path.read_text())
    except _NonFiniteJson:
        errors.append(f"non_finite_json:{label}:1")
    except json.JSONDecodeError as exc:
        errors.append(f"malformed_json:{label}:{exc.lineno}")
    return None


def load_jsonl_stream(path, stream_name, run_dir, errors, repairs):
    path = pathlib.Path(path)
    if not path.is_file():
        errors.append(f"missing_stream:{stream_name}")
        return []

    records = []
    seen_exact = set()
    seen_by_stamp = {}
    last_stamp = None
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = _strict_json_loads(stripped)
            except _NonFiniteJson:
                errors.append(
                    f"non_finite_json:{_relative(path, run_dir)}:{line_number}"
                )
                continue
            except json.JSONDecodeError:
                errors.append(
                    f"malformed_json:{_relative(path, run_dir)}:{line_number}"
                )
                continue
            if not isinstance(record, dict):
                errors.append(
                    f"record_not_object:{_relative(path, run_dir)}:{line_number}"
                )
                continue

            stamp = record_time_sec(record)
            canonical = json.dumps(record, sort_keys=True, separators=(",", ":"))
            exact_key = (stamp, canonical)
            if stream_name in FRAME_STREAMS and exact_key in seen_exact:
                repairs.append(
                    f"identical_duplicate_removed:{stream_name}:{line_number}"
                )
                continue
            if stream_name in FRAME_STREAMS and stamp is not None:
                prior = seen_by_stamp.get(stamp)
                if prior is not None and prior != canonical:
                    errors.append(
                        f"conflicting_duplicate_timestamp:{stream_name}:{stamp:.9f}"
                    )
                    continue
                seen_by_stamp[stamp] = canonical
                if last_stamp is not None and stamp < last_stamp - STAMP_TOLERANCE_SEC:
                    errors.append(
                        f"non_monotonic_timestamp:{stream_name}:{line_number}:"
                        f"{stamp:.9f}<{last_stamp:.9f}"
                    )
                last_stamp = stamp
            if stream_name in FRAME_STREAMS and stamp is None:
                errors.append(f"missing_timestamp:{stream_name}:{line_number}")
            seen_exact.add(exact_key)
            records.append(record)
    return records


def _validate_record_object_ids(record, catalog, errors, unknown):
    for item in _walk_dicts(record):
        for field, valid_field in (
            ("object_id", "object_id_valid"),
            ("observed_object_id", "observed_object_id_valid"),
        ):
            if field not in item:
                continue
            should_check = bool(item.get(valid_field, False))
            if valid_field not in item and "point_count" in item:
                should_check = True
            if not should_check:
                continue
            try:
                object_id = int(item[field])
            except (TypeError, ValueError):
                errors.append(f"invalid_recorded_object_id:{item[field]}")
                continue
            if object_id not in catalog:
                unknown.add(object_id)


def load_compressed_stream(
    run_dir,
    stream_name,
    errors,
    catalog=None,
    anchor_catalog_keys=None,
    record_consumer=None,
):
    records = ReplayableAlgorithmStream(
        run_dir,
        stream_name,
        required=True,
    )
    last_stamp = None
    stamps = []
    unknown = set()
    try:
        for frame_number, record in enumerate(records, start=1):
            stamp = record_time_sec(record)
            if stamp is None:
                errors.append(f"missing_timestamp:{stream_name}:{frame_number}")
            elif last_stamp is not None and stamp < last_stamp - STAMP_TOLERANCE_SEC:
                errors.append(
                    f"non_monotonic_timestamp:{stream_name}:{frame_number}:"
                    f"{stamp:.9f}<{last_stamp:.9f}"
                )
            if stamp is not None:
                last_stamp = stamp
                stamps.append(stamp)
            if catalog is not None:
                _validate_record_object_ids(record, catalog, errors, unknown)
            if stream_name == "anchor_observations" and anchor_catalog_keys is not None:
                try:
                    epoch = int(record.get("reference_epoch", 0))
                except (TypeError, ValueError):
                    errors.append(
                        f"invalid_reference_epoch:anchor_observations:{frame_number}"
                    )
                    continue
                for anchor in record.get("anchors", []):
                    try:
                        key = (epoch, int(anchor.get("id", 0)))
                    except (TypeError, ValueError):
                        errors.append(
                            f"invalid_anchor_id:anchor_observations:{frame_number}"
                        )
                        continue
                    if key not in anchor_catalog_keys:
                        errors.append(
                            f"anchor_observation_without_catalog:{epoch}:{key[1]}"
                        )
            if record_consumer is not None:
                record_consumer(record)
    except AlgorithmFrameStoreError as exc:
        errors.append(f"algorithm_storage_error:{stream_name}:{exc}")
        return []
    for object_id in sorted(unknown):
        errors.append(f"unknown_object_id:{object_id}")
    return ScannedCompressedStream(stream_name, stamps)


def declared_storage_backend(run_info, errors):
    algorithm_recording = (
        run_info.get("algorithm_recording", {}) if isinstance(run_info, dict) else {}
    )
    raw_backend = algorithm_recording.get("storage_backend")
    if raw_backend is None:
        try:
            schema_version = int(algorithm_recording.get("schema_version", 0))
        except (TypeError, ValueError):
            schema_version = 0
        if schema_version >= 3:
            errors.append("schema_v3_storage_backend_missing")
        return "jsonl"
    try:
        return normalize_storage_backend(raw_backend)
    except AlgorithmFrameStoreError as exc:
        errors.append(f"invalid_algorithm_storage_backend:{exc}")
        return "jsonl"


def normalize_catalog(payload, errors):
    if not isinstance(payload, dict) or not payload:
        errors.append("object_id_catalog_missing_or_empty")
        return {}
    catalog = {}
    names = set()
    for raw_id, raw_name in payload.items():
        try:
            object_id = int(raw_id)
        except (TypeError, ValueError):
            errors.append(f"invalid_object_id:{raw_id}")
            continue
        model_name = str(raw_name).strip()
        if object_id <= 0 or object_id > 254:
            errors.append(f"object_id_out_of_range:{object_id}")
        elif not model_name:
            errors.append(f"empty_model_name:{object_id}")
        elif object_id in catalog or model_name in names:
            errors.append(f"non_bijective_object_catalog:{object_id}:{model_name}")
        else:
            catalog[object_id] = model_name
            names.add(model_name)
    return catalog


def _walk_dicts(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk_dicts(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_dicts(child)


def validate_object_ids(records_by_stream, catalog, errors):
    unknown = set()
    for records in records_by_stream.values():
        if isinstance(records, ScannedCompressedStream):
            continue
        for record in records:
            _validate_record_object_ids(record, catalog, errors, unknown)
    for object_id in sorted(unknown):
        errors.append(f"unknown_object_id:{object_id}")


def _validate_quaternion(values, label, errors, repairs):
    try:
        quaternion = [float(value) for value in values]
    except (TypeError, ValueError):
        errors.append(f"invalid_quaternion:{label}")
        return
    if not all(math.isfinite(value) for value in quaternion):
        errors.append(f"non_finite_quaternion:{label}")
        return
    norm = math.sqrt(sum(value * value for value in quaternion))
    difference = abs(norm - 1.0)
    if difference <= 1.0e-9:
        return
    if norm > 0.0 and difference <= QUATERNION_REPAIR_TOLERANCE:
        repairs.append(f"quaternion_renormalized:{label}")
    else:
        errors.append(f"invalid_quaternion_norm:{label}:{norm:.9g}")


def load_truth_tracks(run_dir, catalog, object_rate_hz, errors, repairs):
    truth_dir = run_dir / "truth" / "objects"
    tracks = {}
    if not truth_dir.is_dir():
        errors.append("missing_truth_objects_directory")
        return tracks
    for csv_path in sorted(truth_dir.glob("*.csv")):
        label = _relative(csv_path, run_dir)
        try:
            rows = list(csv.DictReader(csv_path.read_text().splitlines()))
        except csv.Error as exc:
            errors.append(f"malformed_csv:{label}:{exc}")
            continue
        if not rows:
            errors.append(f"empty_truth_track:{label}")
            continue
        missing_columns = REQUIRED_TRUTH_COLUMNS - set(rows[0])
        if missing_columns:
            errors.append(
                f"truth_schema_missing_columns:{label}:"
                + ",".join(sorted(missing_columns))
            )
            continue
        model_names = {str(row.get("model_name", "")).strip() for row in rows}
        if len(model_names) != 1 or "" in model_names:
            errors.append(f"truth_model_name_inconsistent:{label}")
            continue
        model_name = next(iter(model_names))
        stamps = []
        previous_row = None
        for line_number, row in enumerate(rows, start=2):
            numeric_fields = REQUIRED_TRUTH_COLUMNS - {
                "model_name", "frame_id", "twist_frame_id"
            }
            parsed = {}
            for field in numeric_fields:
                try:
                    parsed[field] = float(row[field])
                except (TypeError, ValueError):
                    errors.append(f"invalid_truth_number:{label}:{line_number}:{field}")
                    parsed[field] = math.nan
                if not math.isfinite(parsed[field]):
                    errors.append(f"non_finite_truth_number:{label}:{line_number}:{field}")
            stamp = parsed.get("recorded_time_sec")
            if math.isfinite(stamp):
                if stamps and stamp < stamps[-1] - STAMP_TOLERANCE_SEC:
                    errors.append(f"non_monotonic_truth_timestamp:{label}:{line_number}")
                if stamps and abs(stamp - stamps[-1]) <= STAMP_TOLERANCE_SEC:
                    if row == previous_row:
                        repairs.append(
                            f"identical_truth_duplicate_removed:{label}:{line_number}"
                        )
                    else:
                        errors.append(
                            f"conflicting_truth_timestamp:{label}:{line_number}"
                        )
                stamps.append(stamp)
            _validate_quaternion(
                [
                    row["orientation_x"],
                    row["orientation_y"],
                    row["orientation_z"],
                    row["orientation_w"],
                ],
                f"{label}:{line_number}",
                errors,
                repairs,
            )
            previous_row = row
        if len(stamps) < 2:
            errors.append(f"insufficient_truth_samples:{model_name}")
        elif object_rate_hz > 0.0:
            max_allowed_gap = max(0.5, 3.0 / object_rate_hz)
            max_gap = max(
                (right - left for left, right in zip(stamps, stamps[1:])),
                default=0.0,
            )
            if max_gap > max_allowed_gap + STAMP_TOLERANCE_SEC:
                errors.append(
                    f"truth_gap_too_large:{model_name}:{max_gap:.9f}>"
                    f"{max_allowed_gap:.9f}"
                )
        tracks[model_name] = stamps
    for model_name in catalog.values():
        if model_name not in tracks:
            errors.append(f"missing_truth_track:{model_name}")
    return tracks


def load_direct_link_truth_tracks(
    run_dir,
    link_rate_hz,
    errors,
    repairs,
    expected_track_names=None,
    catalog_requires_tracks=True,
):
    catalog_path = run_dir / "truth" / "surface_truth_points.jsonl"
    expected_scoped_names = set()
    if not catalog_path.is_file():
        errors.append("missing_file:truth/surface_truth_points.jsonl")
    else:
        for line_number, line in enumerate(catalog_path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                record = _strict_json_loads(line)
            except _NonFiniteJson:
                errors.append(
                    f"non_finite_json:truth/surface_truth_points.jsonl:{line_number}"
                )
                continue
            except json.JSONDecodeError:
                errors.append(
                    f"malformed_json:truth/surface_truth_points.jsonl:{line_number}"
                )
                continue
            scoped_name = str(record.get("scoped_link_name", "")).strip()
            if not scoped_name:
                errors.append(f"surface_truth_missing_scoped_name:{line_number}")
            elif scoped_name in expected_scoped_names:
                errors.append(f"surface_truth_duplicate_scoped_name:{scoped_name}")
            else:
                expected_scoped_names.add(scoped_name)

    truth_dir = run_dir / "truth" / "links"
    tracks = {}
    if not truth_dir.is_dir():
        errors.append("missing_truth_links_directory")
        csv_paths = []
    else:
        csv_paths = sorted(truth_dir.glob("*.csv"))

    for csv_path in csv_paths:
        label = _relative(csv_path, run_dir)
        try:
            rows = list(csv.DictReader(csv_path.read_text().splitlines()))
        except csv.Error as exc:
            errors.append(f"malformed_csv:{label}:{exc}")
            continue
        if not rows:
            errors.append(f"empty_link_truth_track:{label}")
            continue
        missing_columns = REQUIRED_LINK_TRUTH_COLUMNS - set(rows[0])
        if missing_columns:
            errors.append(
                f"link_truth_schema_missing_columns:{label}:"
                + ",".join(sorted(missing_columns))
            )
            continue
        scoped_names = {
            str(row.get("scoped_link_name", "")).strip() for row in rows
        }
        model_names = {str(row.get("model_name", "")).strip() for row in rows}
        link_names = {str(row.get("link_name", "")).strip() for row in rows}
        if len(scoped_names) != 1 or "" in scoped_names:
            errors.append(f"link_truth_scoped_name_inconsistent:{label}")
            continue
        if len(model_names) != 1 or "" in model_names:
            errors.append(f"link_truth_model_name_inconsistent:{label}")
            continue
        if len(link_names) != 1 or "" in link_names:
            errors.append(f"link_truth_link_name_inconsistent:{label}")
            continue
        scoped_name = next(iter(scoped_names))
        stamps = []
        previous_row = None
        numeric_fields = REQUIRED_LINK_TRUTH_COLUMNS - {
            "scoped_link_name",
            "model_name",
            "link_name",
            "frame_id",
        }
        for line_number, row in enumerate(rows, 2):
            parsed = {}
            for field in numeric_fields:
                try:
                    parsed[field] = float(row[field])
                except (TypeError, ValueError):
                    errors.append(
                        f"invalid_link_truth_number:{label}:{line_number}:{field}"
                    )
                    parsed[field] = math.nan
                if not math.isfinite(parsed[field]):
                    errors.append(
                        f"non_finite_link_truth_number:{label}:{line_number}:{field}"
                    )
            stamp = parsed.get("recorded_time_sec")
            if math.isfinite(stamp):
                if stamps and stamp < stamps[-1] - STAMP_TOLERANCE_SEC:
                    errors.append(
                        f"non_monotonic_link_truth_timestamp:{label}:{line_number}"
                    )
                if stamps and abs(stamp - stamps[-1]) <= STAMP_TOLERANCE_SEC:
                    if row == previous_row:
                        repairs.append(
                            f"identical_link_truth_duplicate_removed:{label}:{line_number}"
                        )
                    else:
                        errors.append(
                            f"conflicting_link_truth_timestamp:{label}:{line_number}"
                        )
                stamps.append(stamp)
            _validate_quaternion(
                [
                    row["orientation_x"],
                    row["orientation_y"],
                    row["orientation_z"],
                    row["orientation_w"],
                ],
                f"{label}:{line_number}",
                errors,
                repairs,
            )
            previous_row = row
        if len(stamps) < 2:
            errors.append(f"insufficient_link_truth_samples:{scoped_name}")
        elif link_rate_hz > 0.0:
            max_allowed_gap = max(0.5, 3.0 / link_rate_hz)
            max_gap = max(
                (right - left for left, right in zip(stamps, stamps[1:])),
                default=0.0,
            )
            if max_gap > max_allowed_gap + STAMP_TOLERANCE_SEC:
                errors.append(
                    f"link_truth_gap_too_large:{scoped_name}:{max_gap:.9f}>"
                    f"{max_allowed_gap:.9f}"
                )
        if scoped_name in tracks:
            errors.append(f"duplicate_link_truth_track:{scoped_name}")
        tracks[scoped_name] = stamps

    required_track_names = (
        set(expected_track_names)
        if expected_track_names is not None
        else expected_scoped_names
    )
    for scoped_name in sorted(required_track_names - set(tracks)):
        errors.append(f"missing_link_truth_track:{scoped_name}")
    if catalog_requires_tracks:
        for scoped_name in sorted(set(tracks) - expected_scoped_names):
            if scoped_name.rsplit("::", 1)[-1].startswith("ground_truth_"):
                errors.append(f"uncatalogued_link_truth_track:{scoped_name}")
    return tracks


def validate_static_surface_truth_catalog(
    run_dir,
    run_info,
    catalog,
    experiment_factors,
    errors,
    repairs,
):
    """Validate the v3 static marker catalog needed for surface reconstruction."""
    error_count_before = len(errors)
    required = (
        isinstance(experiment_factors, dict)
        and experiment_factors.get("truth_protocol_version")
        == CURRENT_TRUTH_PROTOCOL_VERSION
    )
    report = {
        "required": required,
        "status": "NOT_REQUIRED",
        "expected_point_count": sum(
            count for _, count in CURRENT_SURFACE_TRUTH_CONTRACT.values()
        ),
        "recorded_point_count": 0,
        "per_model_counts": {},
    }
    if not required:
        return report

    truth_recording = (
        run_info.get("truth_recording", {}) if isinstance(run_info, dict) else {}
    )
    truth_recording = truth_recording if isinstance(truth_recording, dict) else {}
    surface_info = truth_recording.get("surface_truth_points", {})
    surface_info = surface_info if isinstance(surface_info, dict) else {}
    if (
        truth_recording.get("dynamic_link_policy")
        != "sensor_motion_drive_links_and_surface_catalog"
    ):
        errors.append("surface_truth_catalog:dynamic_link_policy_mismatch")
    if surface_info.get("catalog_source") != "world_static_marker_visual":
        errors.append("surface_truth_catalog:catalog_source_mismatch")
    if surface_info.get("storage_mode") != "motion_parent_link_local_pose_once":
        errors.append("surface_truth_catalog:storage_mode_mismatch")

    expected_count = report["expected_point_count"]
    try:
        declared_count = int(surface_info.get("expected_point_count"))
    except (TypeError, ValueError):
        declared_count = -1
    if declared_count != expected_count:
        errors.append(
            f"surface_truth_catalog:declared_point_count:{declared_count}"
            f"!={expected_count}"
        )
    try:
        max_radius = float(surface_info.get("max_local_radius_m"))
    except (TypeError, ValueError):
        max_radius = math.nan
    if not math.isfinite(max_radius) or max_radius <= 0.0:
        errors.append("surface_truth_catalog:invalid_max_local_radius")

    relative_path = str(
        surface_info.get("file", "truth/surface_truth_points.jsonl")
    ).strip()
    if relative_path != "truth/surface_truth_points.jsonl":
        errors.append(f"surface_truth_catalog:file_contract:{relative_path}")
    path = pathlib.Path(run_dir) / "truth" / "surface_truth_points.jsonl"
    rows = []
    if not path.is_file():
        errors.append("surface_truth_catalog:missing_file")
    else:
        for line_number, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = _strict_json_loads(line)
            except (_NonFiniteJson, json.JSONDecodeError):
                errors.append(
                    f"surface_truth_catalog:invalid_json_row:{line_number}"
                )
                continue
            if not isinstance(row, dict):
                errors.append(
                    f"surface_truth_catalog:row_not_object:{line_number}"
                )
                continue
            rows.append((line_number, row))

    expected_by_model = {
        name: (object_id, count)
        for object_id, (name, count) in CURRENT_SURFACE_TRUTH_CONTRACT.items()
    }
    per_model_counts = {name: 0 for name in expected_by_model}
    seen_scoped_names = set()
    for line_number, row in rows:
        model_name = str(row.get("model_name", "")).strip()
        link_name = str(row.get("link_name", "")).strip()
        scoped_name = str(row.get("scoped_link_name", "")).strip()
        parent_name = str(
            row.get("motion_parent_scoped_link_name", "")
        ).strip()
        local_frame = str(row.get("object_local_frame", "")).strip()
        expected = expected_by_model.get(model_name)
        if expected is None:
            errors.append(
                f"surface_truth_catalog:unexpected_model:{line_number}:"
                f"{model_name}"
            )
            continue
        expected_object_id, _ = expected
        try:
            object_id = int(row.get("object_id"))
        except (TypeError, ValueError):
            object_id = -1
        if (
            not bool(row.get("object_id_valid", False))
            or object_id != expected_object_id
            or catalog.get(object_id) != model_name
        ):
            errors.append(
                f"surface_truth_catalog:object_identity:{line_number}:"
                f"{object_id}:{model_name}"
            )
        expected_parent = CURRENT_DRIVE_LINK_CONTRACT[model_name]
        if parent_name != expected_parent or local_frame != expected_parent:
            errors.append(
                f"surface_truth_catalog:parent_link:{line_number}:"
                f"{parent_name}:{local_frame}!={expected_parent}"
            )
        if row.get("catalog_source") != "world_static_marker_visual":
            errors.append(
                f"surface_truth_catalog:row_source:{line_number}"
            )
        if not link_name.startswith("ground_truth_v_"):
            errors.append(
                f"surface_truth_catalog:link_name:{line_number}:{link_name}"
            )
        if scoped_name != f"{model_name}::{link_name}":
            errors.append(
                f"surface_truth_catalog:scoped_name:{line_number}:"
                f"{scoped_name}"
            )
        elif scoped_name in seen_scoped_names:
            errors.append(
                f"surface_truth_catalog:duplicate_scoped_name:{scoped_name}"
            )
        else:
            seen_scoped_names.add(scoped_name)
        pose = row.get("local_pose", {})
        pose = pose if isinstance(pose, dict) else {}
        position = pose.get("position", {})
        position = position if isinstance(position, dict) else {}
        try:
            point = [float(position[axis]) for axis in ("x", "y", "z")]
        except (KeyError, TypeError, ValueError):
            point = []
        if not point or not all(math.isfinite(value) for value in point):
            errors.append(
                f"surface_truth_catalog:invalid_local_position:{line_number}"
            )
        elif math.sqrt(sum(value * value for value in point)) > max_radius:
            errors.append(
                f"surface_truth_catalog:local_radius:{line_number}"
            )
        orientation = pose.get("orientation", {})
        orientation = orientation if isinstance(orientation, dict) else {}
        _validate_quaternion(
            [orientation.get(axis) for axis in ("x", "y", "z", "w")],
            f"truth/surface_truth_points.jsonl:{line_number}",
            errors,
            repairs,
        )
        per_model_counts[model_name] += 1

    report["recorded_point_count"] = len(rows)
    report["per_model_counts"] = per_model_counts
    if len(rows) != expected_count:
        errors.append(
            f"surface_truth_catalog:point_count:{len(rows)}!={expected_count}"
        )
    for model_name, (_, expected_model_count) in expected_by_model.items():
        actual_count = per_model_counts.get(model_name, 0)
        if actual_count != expected_model_count:
            errors.append(
                f"surface_truth_catalog:model_point_count:{model_name}:"
                f"{actual_count}!={expected_model_count}"
            )
    report["status"] = "PASS" if len(errors) == error_count_before else "FAIL"
    return report


def _motion_report_json(path):
    try:
        payload = _strict_json_loads(path.read_text())
    except (OSError, json.JSONDecodeError, _NonFiniteJson):
        return None
    return payload if isinstance(payload, dict) else None


def _motion_vector(value, label, errors):
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        errors.append(f"invalid_controller_vector:{label}")
        return None
    try:
        vector = np.asarray([float(component) for component in value], dtype=float)
    except (TypeError, ValueError):
        errors.append(f"invalid_controller_vector:{label}")
        return None
    if not np.isfinite(vector).all():
        errors.append(f"non_finite_controller_vector:{label}")
        return None
    return vector


def _load_motion_link_pose_tracks(run_dir, errors):
    truth_links_dir = pathlib.Path(run_dir) / "truth" / "links"
    tracks = {}
    if not truth_links_dir.is_dir():
        errors.append("missing_truth_links_directory")
        return tracks
    for csv_path in sorted(truth_links_dir.glob("*.csv")):
        try:
            rows = list(csv.DictReader(csv_path.read_text().splitlines()))
        except (OSError, csv.Error) as exc:
            errors.append(f"motion_link_csv_error:{csv_path.name}:{exc}")
            continue
        if not rows:
            continue
        scoped_name = str(rows[0].get("scoped_link_name", "")).strip()
        model_name = str(rows[0].get("model_name", "")).strip()
        link_name = str(rows[0].get("link_name", "")).strip()
        if not scoped_name or not model_name or not link_name:
            continue
        samples = []
        for line_number, row in enumerate(rows, 2):
            try:
                quaternion = np.asarray(
                    [
                        float(row["orientation_x"]),
                        float(row["orientation_y"]),
                        float(row["orientation_z"]),
                        float(row["orientation_w"]),
                    ],
                    dtype=float,
                )
                quaternion_norm = float(np.linalg.norm(quaternion))
                sample = (
                    float(row["recorded_time_sec"]),
                    np.asarray(
                        [
                            float(row["position_x"]),
                            float(row["position_y"]),
                            float(row["position_z"]),
                        ],
                        dtype=float,
                    ),
                    quaternion / quaternion_norm,
                )
            except (KeyError, TypeError, ValueError, ZeroDivisionError):
                errors.append(
                    f"invalid_motion_link_sample:{csv_path.name}:{line_number}"
                )
                continue
            if (
                not math.isfinite(sample[0])
                or not np.isfinite(sample[1]).all()
                or not np.isfinite(sample[2]).all()
            ):
                errors.append(
                    f"non_finite_motion_link_sample:{csv_path.name}:{line_number}"
                )
                continue
            samples.append(sample)
        samples.sort(key=lambda item: item[0])
        if scoped_name in tracks:
            errors.append(f"duplicate_motion_link_track:{scoped_name}")
            continue
        tracks[scoped_name] = {
            "model_name": model_name,
            "link_name": link_name,
            "samples": samples,
        }
    return tracks


def _slerp_motion_quaternion(left, right, alpha):
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    dot = float(np.dot(left, right))
    if dot < 0.0:
        right = -right
        dot = -dot
    dot = min(1.0, max(-1.0, dot))
    if dot > 0.9995:
        result = left + alpha * (right - left)
        return result / np.linalg.norm(result)
    angle = math.acos(dot)
    sine = math.sin(angle)
    return (
        math.sin((1.0 - alpha) * angle) / sine * left
        + math.sin(alpha * angle) / sine * right
    )


def _interpolate_motion_pose(samples, target_time):
    if not samples or target_time < samples[0][0] or target_time > samples[-1][0]:
        return None
    for index, (stamp, position, quaternion) in enumerate(samples):
        if abs(stamp - target_time) <= STAMP_TOLERANCE_SEC:
            return position.copy(), quaternion.copy()
        if stamp > target_time:
            left_stamp, left_position, left_quaternion = samples[index - 1]
            interval = stamp - left_stamp
            if interval <= 0.0:
                return None
            alpha = (target_time - left_stamp) / interval
            return (
                (1.0 - alpha) * left_position + alpha * position,
                _slerp_motion_quaternion(left_quaternion, quaternion, alpha),
            )
    return samples[-1][1].copy(), samples[-1][2].copy()


def _motion_quaternion_multiply(left, right):
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    return np.asarray(
        [
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        ],
        dtype=float,
    )


def _motion_quaternion_rotation_vector(initial, final):
    relative = _motion_quaternion_multiply(
        final, np.asarray([-initial[0], -initial[1], -initial[2], initial[3]])
    )
    if relative[3] < 0.0:
        relative = -relative
    vector_norm = float(np.linalg.norm(relative[:3]))
    if vector_norm <= 1.0e-12:
        return np.zeros(3, dtype=float), 0.0
    angle = 2.0 * math.atan2(vector_norm, float(relative[3]))
    axis = relative[:3] / vector_norm
    return axis * angle, angle


def _motion_quaternion_rotation_matrix(quaternion):
    x, y, z, w = quaternion
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )


def _load_surface_catalog_local_points(run_dir, errors):
    path = pathlib.Path(run_dir) / "truth" / "surface_truth_points.jsonl"
    by_model = {}
    if not path.is_file():
        errors.append("missing_surface_truth_catalog")
        return by_model
    scoped_names = set()
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = _strict_json_loads(line)
            scoped_name = str(record.get("scoped_link_name", "")).strip()
            model_name = str(record.get("model_name", "")).strip()
            parent_scoped_name = str(
                record.get("motion_parent_scoped_link_name", "")
            ).strip()
            position = record.get("local_pose", {}).get("position", {})
            point = np.asarray(
                [float(position[axis]) for axis in ("x", "y", "z")], dtype=float
            )
        except (json.JSONDecodeError, _NonFiniteJson, KeyError, TypeError, ValueError):
            errors.append(f"invalid_surface_truth_catalog_row:{line_number}")
            continue
        if not scoped_name or not model_name or not np.isfinite(point).all():
            errors.append(f"invalid_surface_truth_catalog_row:{line_number}")
            continue
        if scoped_name in scoped_names:
            errors.append(f"duplicate_surface_truth_catalog_link:{scoped_name}")
            continue
        scoped_names.add(scoped_name)
        expected_parent = CURRENT_DRIVE_LINK_CONTRACT.get(model_name)
        if expected_parent is None:
            errors.append(
                f"unexpected_surface_truth_catalog_model:{line_number}:{model_name}"
            )
            continue
        if parent_scoped_name != expected_parent:
            errors.append(
                f"surface_truth_parent_link:{line_number}:{parent_scoped_name}"
                f"!={expected_parent}"
            )
            continue
        by_model.setdefault(model_name, []).append(point)
    return by_model


def _motion_direction_cosine(actual, expected):
    actual_norm = float(np.linalg.norm(actual))
    expected_norm = float(np.linalg.norm(expected))
    if actual_norm <= 1.0e-12 or expected_norm <= 1.0e-12:
        return None
    return float(np.dot(actual, expected) / (actual_norm * expected_norm))


def _translation_segment_report(
    samples,
    name,
    start_time,
    end_time,
    expected_velocity,
    prefix_errors=False,
):
    errors = []
    duration = float(end_time) - float(start_time)
    error_prefix = f"segment:{name}:" if prefix_errors else ""
    expected_velocity = np.asarray(expected_velocity, dtype=float)
    expected_translation = expected_velocity * max(0.0, duration)
    expected_translation_norm = float(np.linalg.norm(expected_translation))
    actual_translation = np.zeros(3, dtype=float)
    actual_translation_norm = 0.0
    direction_cosine = None
    translation_error = math.inf

    start_pose = None
    end_pose = None
    if not math.isfinite(duration) or duration <= 0.0:
        errors.append(f"{error_prefix}duration_invalid")
    else:
        start_pose = _interpolate_motion_pose(samples, float(start_time))
        end_pose = _interpolate_motion_pose(samples, float(end_time))
        if start_pose is None or end_pose is None:
            errors.append(f"{error_prefix}time_coverage_missing")
        else:
            actual_translation = end_pose[0] - start_pose[0]
            actual_translation_norm = float(np.linalg.norm(actual_translation))
            translation_error = float(
                np.linalg.norm(actual_translation - expected_translation)
            )
            direction_cosine = _motion_direction_cosine(
                actual_translation, expected_translation
            )
            tolerance = max(
                MOTION_TRANSLATION_ABS_TOLERANCE_M,
                MOTION_RELATIVE_TOLERANCE * expected_translation_norm,
            )
            if translation_error > tolerance:
                errors.append(
                    f"{error_prefix}translation_error:"
                    f"{translation_error:.9g}>{tolerance:.9g}"
                )
            if (
                expected_translation_norm > MOTION_TRANSLATION_ABS_TOLERANCE_M
                and (
                    direction_cosine is None
                    or direction_cosine < MOTION_DIRECTION_COSINE_MIN
                )
            ):
                errors.append(
                    f"{error_prefix}translation_direction_cosine:"
                    f"{direction_cosine}"
                )

    return {
        "name": str(name),
        "status": "PASS" if not errors else "FAIL",
        "errors": sorted(set(errors)),
        "start_time_sec": float(start_time),
        "end_time_sec": float(end_time),
        "duration_sec": duration,
        "expected_linear_velocity_world_mps": expected_velocity.tolist(),
        "expected_translation_world_m": expected_translation.tolist(),
        "expected_translation_m": expected_translation_norm,
        "actual_translation_world_m": (
            actual_translation.tolist()
            if start_pose is not None and end_pose is not None
            else None
        ),
        "actual_translation_m": (
            actual_translation_norm
            if start_pose is not None and end_pose is not None
            else None
        ),
        "translation_error_m": (
            translation_error if math.isfinite(translation_error) else None
        ),
        "translation_direction_cosine": direction_cosine,
        "actual_linear_speed_mps": (
            actual_translation_norm / duration
            if duration > 0.0 and start_pose is not None and end_pose is not None
            else None
        ),
    }, errors


def _motion_execution_report_path(run_dir):
    return pathlib.Path(run_dir) / "meta" / "motion_execution_validation.json"


def validate_current_motion_execution(run_dir, write_report=True):
    """Validate motion from drive-link poses and parent-local surface points."""

    run_dir = pathlib.Path(run_dir).resolve()
    errors = []
    scenario = _motion_report_json(run_dir / "meta" / "scenario_manifest.json")
    protocol = _motion_report_json(run_dir / "meta" / "experiment_protocol.json")
    if scenario is None:
        errors.append("missing_or_invalid_scenario_manifest")
        scenario = {}
    if protocol is None:
        errors.append("missing_or_invalid_experiment_protocol")
        protocol = {}
    factors = scenario.get("experiment_factors", {})
    factors = factors if isinstance(factors, dict) else {}
    object_metadata = scenario.get("object_metadata", {})
    object_metadata = object_metadata if isinstance(object_metadata, dict) else {}
    motion_protocol = str(factors.get("motion_protocol_version", "")).strip()
    truth_protocol = str(factors.get("truth_protocol_version", "")).strip()
    if motion_protocol != CURRENT_MOTION_PROTOCOL_VERSION:
        errors.append(
            f"motion_protocol_version:{motion_protocol}!={CURRENT_MOTION_PROTOCOL_VERSION}"
        )
    if truth_protocol != CURRENT_TRUTH_PROTOCOL_VERSION:
        errors.append(
            f"truth_protocol_version:{truth_protocol}!={CURRENT_TRUTH_PROTOCOL_VERSION}"
        )

    actual = protocol.get("actual", {}) if isinstance(protocol, dict) else {}
    actual = actual if isinstance(actual, dict) else {}
    controller = actual.get("controller_configuration", {})
    controller = controller if isinstance(controller, dict) else {}
    try:
        motion_start = float(actual["actual_motion_start_time"])
        hold_start = float(actual["hold_start_time"])
    except (KeyError, TypeError, ValueError):
        motion_start = math.nan
        hold_start = math.nan
        errors.append("motion_phase_boundaries_missing_or_invalid")
    duration = hold_start - motion_start
    if not math.isfinite(duration) or duration <= 0.0:
        errors.append("motion_duration_invalid")

    try:
        linear_speed_mps = float(controller["linear_speed_mm_s"]) / 1000.0
        angular_speed_rad_s = float(controller["angular_speed_rad_s"])
    except (KeyError, TypeError, ValueError):
        linear_speed_mps = math.nan
        angular_speed_rad_s = math.nan
        errors.append("controller_speed_parameters_missing_or_invalid")

    models = controller.get("models", [])
    models = models if isinstance(models, list) else []
    commands = {}
    for command in models:
        if not isinstance(command, dict):
            errors.append("controller_model_entry_not_mapping")
            continue
        try:
            object_id = int(command.get("id"))
        except (TypeError, ValueError):
            errors.append("controller_model_entry_invalid_id")
            continue
        if object_id in commands:
            errors.append(f"controller_model_duplicate_id:{object_id}")
        commands[object_id] = command

    track_errors = []
    tracks = _load_motion_link_pose_tracks(run_dir, track_errors)
    errors.extend(track_errors)
    track_errors = []
    surface_catalog = _load_surface_catalog_local_points(run_dir, track_errors)
    errors.extend(track_errors)

    object_reports = []
    for object_id, (expected_name, expected_link_count) in sorted(
        CURRENT_SURFACE_TRUTH_CONTRACT.items()
    ):
        object_errors = []
        command = commands.get(object_id)
        if command is None:
            object_errors.append("missing_controller_command")
            command = {}
        actual_name = str(command.get("model_name", "")).strip()
        if actual_name != expected_name:
            object_errors.append(
                f"controller_model_name:{actual_name}!={expected_name}"
            )
        frame = str(command.get("command_frame", "")).strip().lower()
        if frame != "world":
            object_errors.append(f"controller_frame:{frame}!=world")

        linear_direction = _motion_vector(
            command.get("linear_direction"),
            f"{object_id}:linear_direction",
            object_errors,
        )
        angular_axis = _motion_vector(
            command.get("angular_axis"),
            f"{object_id}:angular_axis",
            object_errors,
        )
        fixed_linear = _motion_vector(
            command.get("fixed_linear_mps"),
            f"{object_id}:fixed_linear_mps",
            object_errors,
        )
        expected_linear_velocity = np.zeros(3, dtype=float)
        expected_angular_velocity = np.zeros(3, dtype=float)
        if linear_direction is not None and math.isfinite(linear_speed_mps):
            expected_linear_velocity = linear_direction * linear_speed_mps
        fixed_linear_norm = (
            float(np.linalg.norm(fixed_linear)) if fixed_linear is not None else 0.0
        )
        if fixed_linear_norm > 0.0:
            expected_linear_velocity = fixed_linear
        if angular_axis is not None and math.isfinite(angular_speed_rad_s):
            expected_angular_velocity = angular_axis * angular_speed_rad_s

        expected_profile = (
            "constant_rotation"
            if float(np.linalg.norm(expected_angular_velocity)) > 0.0
            else "fixed_speed_translation"
            if fixed_linear_norm > 0.0
            else "constant_translation"
        )
        recorded_metadata = object_metadata.get(expected_name) or {}
        recorded_metadata = (
            recorded_metadata if isinstance(recorded_metadata, dict) else {}
        )
        recorded_profile = str(recorded_metadata.get("motion_profile", "")).strip()
        if recorded_profile != expected_profile:
            object_errors.append(
                f"motion_profile:{recorded_profile}!={expected_profile}"
            )

        expected_drive_scoped_name = CURRENT_DRIVE_LINK_CONTRACT[expected_name]
        drive_track = tracks.get(expected_drive_scoped_name)
        if drive_track is None:
            object_errors.append(
                f"missing_drive_link_track:{expected_drive_scoped_name}"
            )
        local_surface_points = surface_catalog.get(expected_name, [])
        if len(local_surface_points) != expected_link_count:
            object_errors.append(
                f"surface_catalog_count:{len(local_surface_points)}!={expected_link_count}"
            )
        initial_pose = None
        final_pose = None
        if math.isfinite(motion_start) and math.isfinite(hold_start):
            if drive_track is not None:
                initial_pose = _interpolate_motion_pose(
                    drive_track["samples"], motion_start
                )
                final_pose = _interpolate_motion_pose(
                    drive_track["samples"], hold_start
                )
                if initial_pose is None or final_pose is None:
                    object_errors.append(
                        f"motion_time_coverage:{expected_drive_scoped_name}"
                    )

        translation = np.zeros(3, dtype=float)
        rotation_vector = np.zeros(3, dtype=float)
        rotation_angle = 0.0
        displacements = []
        if initial_pose is not None and final_pose is not None:
            initial_position, initial_quaternion = initial_pose
            final_position, final_quaternion = final_pose
            translation = final_position - initial_position
            rotation_vector, rotation_angle = _motion_quaternion_rotation_vector(
                initial_quaternion, final_quaternion
            )
            initial_rotation = _motion_quaternion_rotation_matrix(
                initial_quaternion
            )
            final_rotation = _motion_quaternion_rotation_matrix(final_quaternion)
            for local_point in local_surface_points:
                initial_surface = initial_rotation.dot(local_point) + initial_position
                final_surface = final_rotation.dot(local_point) + final_position
                displacements.append(
                    float(np.linalg.norm(final_surface - initial_surface))
                )

        calculation_duration = (
            duration if math.isfinite(duration) and duration > 0.0 else 0.0
        )
        translation_segment_specs = []
        piecewise_translation = False
        if (
            float(np.linalg.norm(expected_angular_velocity)) <= 0.0
            and float(np.linalg.norm(expected_linear_velocity)) > 0.0
            and calculation_duration > 0.0
        ):
            translation_segment_specs = [
                (
                    "constant",
                    motion_start,
                    hold_start,
                    expected_linear_velocity,
                )
            ]
            has_person_switch = "person_walking_switch_time" in controller
            has_person_second_velocity = (
                "person_walking_second_linear_mps" in controller
            )
            if expected_name == "person_walking" and (
                has_person_switch or has_person_second_velocity
            ):
                piecewise_translation = True
                if not has_person_switch or not has_person_second_velocity:
                    object_errors.append(
                        "person_walking_piecewise_configuration_incomplete"
                    )
                    translation_segment_specs = []
                else:
                    try:
                        switch_offset = float(
                            controller["person_walking_switch_time"]
                        )
                    except (TypeError, ValueError):
                        switch_offset = math.nan
                    second_velocity = _motion_vector(
                        controller.get("person_walking_second_linear_mps"),
                        "person_walking_second_linear_mps",
                        object_errors,
                    )
                    if (
                        not math.isfinite(switch_offset)
                        or switch_offset <= 0.0
                        or switch_offset >= calculation_duration
                    ):
                        object_errors.append(
                            "person_walking_switch_time_outside_motion_interval"
                        )
                        translation_segment_specs = []
                    elif second_velocity is None:
                        translation_segment_specs = []
                    else:
                        switch_time = motion_start + switch_offset
                        translation_segment_specs = [
                            (
                                "outbound",
                                motion_start,
                                switch_time,
                                expected_linear_velocity,
                            ),
                            (
                                "return",
                                switch_time,
                                hold_start,
                                second_velocity,
                            ),
                        ]

        translation_segments = []
        for segment_name, segment_start, segment_end, segment_velocity in (
            translation_segment_specs
        ):
            if drive_track is None:
                break
            segment_report, segment_errors = _translation_segment_report(
                drive_track["samples"],
                segment_name,
                segment_start,
                segment_end,
                segment_velocity,
                prefix_errors=piecewise_translation,
            )
            translation_segments.append(segment_report)
            object_errors.extend(segment_errors)

        if translation_segment_specs:
            expected_translation = sum(
                (
                    np.asarray(segment_velocity, dtype=float)
                    * (float(segment_end) - float(segment_start))
                    for _, segment_start, segment_end, segment_velocity
                    in translation_segment_specs
                ),
                np.zeros(3, dtype=float),
            )
        else:
            expected_translation = np.zeros(3, dtype=float)
        expected_rotation = expected_angular_velocity * calculation_duration
        expected_translation_norm = float(np.linalg.norm(expected_translation))
        expected_rotation_angle = float(np.linalg.norm(expected_rotation))
        translation_error = float(np.linalg.norm(translation - expected_translation))
        rotation_error = abs(rotation_angle - expected_rotation_angle)
        translation_cosine = _motion_direction_cosine(
            translation, expected_translation
        )
        rotation_cosine = _motion_direction_cosine(
            rotation_vector, expected_rotation
        )

        if expected_rotation_angle > 0.0:
            tolerance = max(
                MOTION_ROTATION_ABS_TOLERANCE_RAD,
                MOTION_RELATIVE_TOLERANCE * expected_rotation_angle,
            )
            if rotation_error > tolerance:
                object_errors.append(
                    f"rotation_angle_error:{rotation_error:.9g}>{tolerance:.9g}"
                )
            if rotation_cosine is None or rotation_cosine < MOTION_DIRECTION_COSINE_MIN:
                object_errors.append(
                    f"rotation_axis_cosine:{rotation_cosine}"
            )
            if not displacements or max(displacements) < MOTION_MIN_ROTATING_SURFACE_DISPLACEMENT_M:
                object_errors.append("rotating_surface_displacement_below_minimum")
        elif translation_segment_specs:
            if rotation_angle > MOTION_MAX_UNCOMMANDED_ROTATION_RAD:
                object_errors.append(
                    f"uncommanded_rotation:{rotation_angle:.9g}>"
                    f"{MOTION_MAX_UNCOMMANDED_ROTATION_RAD:.9g}"
                )
        else:
            object_errors.append("controller_command_has_no_motion")

        expected_path_length = sum(
            float(np.linalg.norm(np.asarray(segment_velocity, dtype=float)))
            * (float(segment_end) - float(segment_start))
            for _, segment_start, segment_end, segment_velocity in (
                translation_segment_specs
            )
        )
        actual_path_values = [
            segment.get("actual_translation_m")
            for segment in translation_segments
        ]
        actual_path_length = (
            sum(float(value) for value in actual_path_values)
            if actual_path_values
            and all(value is not None for value in actual_path_values)
            else None
        )
        object_report = {
            "object_id": object_id,
            "object_name": expected_name,
            "motion_profile": expected_profile,
            "execution_profile": (
                "piecewise_fixed_speed_translation"
                if piecewise_translation
                else expected_profile
            ),
            "status": "PASS" if not object_errors else "FAIL",
            "errors": sorted(set(object_errors)),
            "expected_drive_link_scoped_name": expected_drive_scoped_name,
            "drive_link_track_recorded": drive_track is not None,
            "expected_surface_point_count": expected_link_count,
            "recorded_surface_catalog_count": len(local_surface_points),
            "evaluated_surface_point_count": len(displacements),
            "expected_linear_velocity_world_mps": expected_linear_velocity.tolist(),
            "expected_angular_velocity_world_rad_s": expected_angular_velocity.tolist(),
            "expected_translation_world_m": expected_translation.tolist(),
            "expected_rotation_vector_world_rad": expected_rotation.tolist(),
            "actual_translation_world_m": (
                translation.tolist() if expected_rotation_angle <= 0.0 else None
            ),
            "actual_rotation_vector_world_rad": rotation_vector.tolist(),
            "expected_translation_m": expected_translation_norm,
            "actual_translation_m": (
                float(np.linalg.norm(translation))
                if expected_rotation_angle <= 0.0 else None
            ),
            "translation_error_m": (
                translation_error if expected_rotation_angle <= 0.0 else None
            ),
            "translation_direction_cosine": translation_cosine,
            "motion_segments": translation_segments,
            "expected_path_length_m": (
                expected_path_length if translation_segment_specs else None
            ),
            "actual_path_length_m": actual_path_length,
            "expected_net_linear_speed_mps": (
                expected_translation_norm / duration
                if duration > 0.0 and translation_segment_specs else None
            ),
            "actual_net_linear_speed_mps": (
                float(np.linalg.norm(translation)) / duration
                if duration > 0.0 and expected_rotation_angle <= 0.0 else None
            ),
            "expected_rotation_angle_rad": expected_rotation_angle,
            "actual_rotation_angle_rad": rotation_angle,
            "rotation_angle_error_rad": rotation_error,
            "rotation_axis_cosine": rotation_cosine,
            "actual_linear_speed_mps": (
                actual_path_length / duration
                if duration > 0.0 and actual_path_length is not None
                else None
            ),
            "actual_angular_speed_rad_s": (
                rotation_angle / duration if duration > 0.0 else None
            ),
            "surface_displacement_min_m": min(displacements) if displacements else None,
            "surface_displacement_median_m": (
                statistics.median(displacements) if displacements else None
            ),
            "surface_displacement_max_m": max(displacements) if displacements else None,
        }
        object_reports.append(object_report)
        errors.extend(
            f"object:{expected_name}:{error}" for error in object_report["errors"]
        )

    report = {
        "schema_version": 2,
        "status": "PASS" if not errors else "FAIL",
        "valid_for_analysis": not errors,
        "motion_protocol_version": motion_protocol,
        "truth_protocol_version": truth_protocol,
        "motion_start_time_sec": motion_start if math.isfinite(motion_start) else None,
        "hold_start_time_sec": hold_start if math.isfinite(hold_start) else None,
        "motion_duration_sec": duration if math.isfinite(duration) else None,
        "expected_object_count": len(CURRENT_SURFACE_TRUTH_CONTRACT),
        "evaluated_object_count": sum(
            object_report["status"] == "PASS" for object_report in object_reports
        ),
        "errors": sorted(set(errors)),
        "tolerances": {
            "direction_cosine_min": MOTION_DIRECTION_COSINE_MIN,
            "translation_abs_m": MOTION_TRANSLATION_ABS_TOLERANCE_M,
            "rotation_abs_rad": MOTION_ROTATION_ABS_TOLERANCE_RAD,
            "relative": MOTION_RELATIVE_TOLERANCE,
            "max_uncommanded_rotation_rad": MOTION_MAX_UNCOMMANDED_ROTATION_RAD,
            "min_rotating_surface_displacement_m": (
                MOTION_MIN_ROTATING_SURFACE_DISPLACEMENT_M
            ),
        },
        "objects": object_reports,
    }
    if write_report:
        report_path = _motion_execution_report_path(run_dir)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
    return report


def validate_alignment(run_dir, errors, repairs):
    payload = load_json_file(run_dir / "meta" / "frame_alignment.json", run_dir, errors)
    if not isinstance(payload, dict):
        return
    truth_stamp = time_sec(payload.get("truth_reference_stamp_sec"))
    algorithm_stamp = time_sec(payload.get("algorithm_reference_stamp_sec"))
    delta = time_sec(payload.get("pose_pair_delta_sec"))
    if truth_stamp is None or algorithm_stamp is None or delta is None:
        errors.append("alignment_source_timestamps_missing")
    else:
        expected = abs(algorithm_stamp - truth_stamp)
        if abs(delta - expected) > STAMP_TOLERANCE_SEC:
            errors.append("alignment_pose_pair_delta_inconsistent")
        if expected > ALIGNMENT_MAX_DELTA_SEC:
            errors.append(f"alignment_pose_pair_delta_too_large:{expected:.9f}")
    pose = payload.get("world_from_algorithm_transform", {}).get("pose", {})
    orientation = pose.get("orientation", {}) if isinstance(pose, dict) else {}
    _validate_quaternion(
        [
            orientation.get("x"),
            orientation.get("y"),
            orientation.get("z"),
            orientation.get("w"),
        ],
        "meta/frame_alignment.json:world_from_algorithm",
        errors,
        repairs,
    )


def _stamp_keys(records, monitoring_only=False):
    if isinstance(records, ScannedCompressedStream):
        return set(records.stamp_keys)
    keys = set()
    for record in records:
        if monitoring_only and int(record.get("phase", -1)) != 1:
            continue
        stamp = record_time_sec(record)
        if stamp is not None:
            keys.add(round(stamp / STAMP_TOLERANCE_SEC))
    return keys


def _reference_reset_stamp_keys(processing_records):
    """Return synthetic output stamps emitted by a manual reference reset."""
    keys = set()
    ordered_records = sorted(
        (
            (record_time_sec(record), record)
            for record in processing_records
            if record_time_sec(record) is not None
        ),
        key=lambda item: item[0],
    )
    previous_epoch = None
    for stamp, record in ordered_records:
        try:
            reference_epoch = int(record.get("reference_epoch", 0))
            anchor_count = int(record.get("anchor_count", -1))
        except (TypeError, ValueError):
            continue
        if (
            previous_epoch is not None
            and reference_epoch > previous_epoch
            and anchor_count == 0
        ):
            keys.add(round(stamp / STAMP_TOLERANCE_SEC))
        previous_epoch = reference_epoch
    return keys


def validate_cross_stream_coverage(records_by_stream, errors):
    processing_records = records_by_stream.get("processing_stamps", [])
    processing = _stamp_keys(processing_records)
    if not processing:
        errors.append("processing_stamps_empty")
        return []
    reference_resets = _reference_reset_stamp_keys(processing_records)
    for stream_name in FRAME_STREAMS - {"processing_stamps"}:
        stream_stamps = _stamp_keys(
            records_by_stream.get(stream_name, []),
            monitoring_only=stream_name == "object_observation_stats",
        )
        missing = processing - stream_stamps
        if stream_name == "object_observation_stats":
            # ResetReference publishes one empty algorithm result immediately,
            # but observation statistics resume only after new LiDAR frames.
            missing -= reference_resets
        extra = stream_stamps - processing
        if missing:
            errors.append(
                f"processing_stamp_coverage_missing:{stream_name}:{len(missing)}"
            )
        if extra:
            errors.append(f"unexpected_processing_stamps:{stream_name}:{len(extra)}")
    return sorted(record_time_sec(record) for record in processing_records)


def validate_reference_epochs(records_by_stream, errors):
    catalog = {}
    for line_number, record in enumerate(records_by_stream.get("anchor_catalog", []), start=1):
        anchor = record.get("anchor", {})
        try:
            epoch = int(record.get("reference_epoch", anchor.get("reference_epoch", 0)))
            anchor_id = int(record.get("anchor_id", anchor.get("id", 0)))
        except (TypeError, ValueError):
            errors.append(f"invalid_anchor_catalog_key:{line_number}")
            continue
        key = (epoch, anchor_id)
        if key in catalog and catalog[key] != anchor:
            errors.append(f"conflicting_anchor_catalog_key:{epoch}:{anchor_id}")
        catalog[key] = anchor
    anchor_observations = records_by_stream.get("anchor_observations", [])
    if isinstance(anchor_observations, ScannedCompressedStream):
        return
    for line_number, record in enumerate(anchor_observations, start=1):
        try:
            epoch = int(record.get("reference_epoch", 0))
        except (TypeError, ValueError):
            errors.append(f"invalid_reference_epoch:anchor_observations:{line_number}")
            continue
        for anchor in record.get("anchors", []):
            try:
                key = (epoch, int(anchor.get("id", 0)))
            except (TypeError, ValueError):
                errors.append(f"invalid_anchor_id:anchor_observations:{line_number}")
                continue
            if key not in catalog:
                errors.append(f"anchor_observation_without_catalog:{epoch}:{key[1]}")


def validate_completion(
    run_dir,
    records_by_stream,
    errors,
    storage_backend="jsonl",
    storage_report=None,
):
    payload = load_json_file(run_dir / "meta" / "run_complete.json", run_dir, errors)
    if not isinstance(payload, dict):
        return None
    if not bool(payload.get("clean_shutdown", False)):
        errors.append("run_not_cleanly_closed")
    if storage_backend != "jsonl":
        if payload.get("recording_integrity_valid") is not True:
            errors.append("recording_integrity_invalid")
        reported_storage = payload.get("algorithm_storage")
        if not isinstance(reported_storage, dict):
            errors.append("run_complete_algorithm_storage_missing")
        else:
            if reported_storage.get("backend") != storage_backend:
                errors.append(
                    "run_complete_storage_backend_mismatch:{}!={}".format(
                        reported_storage.get("backend"), storage_backend
                    )
                )
            if reported_storage.get("integrity_check") != "ok":
                errors.append("run_complete_storage_integrity_not_ok")
            reported_streams = reported_storage.get("streams", {})
            if not isinstance(reported_streams, dict):
                errors.append("run_complete_storage_streams_missing")
                reported_streams = {}
            actual_streams = (
                storage_report.get("streams", {})
                if isinstance(storage_report, dict)
                else {}
            )
            for stream_name in HIGH_VOLUME_STREAMS:
                reported = reported_streams.get(stream_name, {})
                actual = actual_streams.get(stream_name, {})
                if not isinstance(reported, dict):
                    errors.append(
                        f"run_complete_storage_stream_missing:{stream_name}"
                    )
                    continue
                for field in (
                    "frame_count",
                    "item_count",
                    "raw_byte_count",
                    "compressed_byte_count",
                ):
                    try:
                        reported_value = int(reported.get(field, -1))
                        actual_value = int(actual.get(field, 0))
                    except (TypeError, ValueError):
                        reported_value = -1
                        actual_value = int(actual.get(field, 0) or 0)
                    if reported_value != actual_value:
                        errors.append(
                            "algorithm_storage_{}_mismatch:{}:{}!={}".format(
                                field,
                                stream_name,
                                reported_value,
                                actual_value,
                            )
                        )
                if int(actual.get("frame_count", 0) or 0) != len(
                    records_by_stream.get(stream_name, [])
                ):
                    errors.append(
                        f"decoded_storage_frame_count_mismatch:{stream_name}"
                    )
    stream_stats = payload.get("streams", {})
    if not isinstance(stream_stats, dict):
        errors.append("run_complete_stream_stats_missing")
        return payload
    for stream_name, records in records_by_stream.items():
        stats = stream_stats.get(stream_name)
        if not isinstance(stats, dict):
            errors.append(f"run_complete_stream_missing:{stream_name}")
            continue
        try:
            row_count = int(stats.get("row_count", -1))
        except (TypeError, ValueError):
            row_count = -1
        if row_count != len(records):
            errors.append(
                f"run_complete_row_count_mismatch:{stream_name}:"
                f"{row_count}!={len(records)}"
            )
        if bool(stats.get("drop_estimate_available", False)):
            try:
                drops = int(stats.get("estimated_drop_count", 0))
            except (TypeError, ValueError):
                drops = -1
            if drops != 0:
                errors.append(f"recorded_processing_drop:{stream_name}:{drops}")
            if int(stats.get("irregular_sequence_delta_count", 0)) != 0:
                errors.append(f"irregular_sequence_delta:{stream_name}")
    return payload


def build_anchor_processing_coverage_report(records_by_stream, completion):
    reference_stream = "clusters"
    reference_keys = _stamp_keys(records_by_stream.get(reference_stream, []))
    anchor_keys = _stamp_keys(records_by_stream.get("anchor_observations", []))
    processing_records = records_by_stream.get("processing_stamps", [])
    processing_keys = _stamp_keys(processing_records)
    reset_keys = _reference_reset_stamp_keys(processing_records)
    coherence_errors = []

    if not reference_keys:
        coherence_errors.append("reference_stream_empty")
    if anchor_keys != processing_keys:
        coherence_errors.append("anchor_and_processing_stamp_sets_differ")
    if anchor_keys - reference_keys:
        coherence_errors.append("anchor_stamps_outside_reference_stream")
    if processing_keys - reference_keys:
        coherence_errors.append("processing_stamps_outside_reference_stream")

    dependent_streams = sorted(
        FRAME_STREAMS - {"anchor_observations", "processing_stamps"}
    )
    for stream_name in dependent_streams:
        stream_keys = _stamp_keys(
            records_by_stream.get(stream_name, []),
            monitoring_only=stream_name == "object_observation_stats",
        )
        expected_keys = set(reference_keys)
        if stream_name == "object_observation_stats":
            expected_keys -= reset_keys
        if stream_keys != expected_keys:
            coherence_errors.append(f"dependent_stream_mismatch:{stream_name}")

    missing_keys = reference_keys - anchor_keys
    missing_count = len(missing_keys)
    reference_count = len(reference_keys)
    anchor_count = len(anchor_keys & reference_keys)
    coverage_fraction = (
        float(anchor_count) / float(reference_count)
        if reference_count > 0
        else None
    )
    drop_fraction = (
        float(missing_count) / float(reference_count)
        if reference_count > 0
        else None
    )

    stream_stats = (
        completion.get("streams", {}) if isinstance(completion, dict) else {}
    )
    stream_stats = stream_stats if isinstance(stream_stats, dict) else {}
    reported_drop_counts = {}
    for stream_name in sorted(FRAME_STREAMS):
        stats = stream_stats.get(stream_name, {})
        stats = stats if isinstance(stats, dict) else {}
        if bool(stats.get("drop_estimate_available", False)):
            try:
                reported_drop_counts[stream_name] = int(
                    stats.get("estimated_drop_count", -1)
                )
            except (TypeError, ValueError):
                reported_drop_counts[stream_name] = -1
        try:
            irregular_count = int(
                stats.get("irregular_sequence_delta_count", 0)
            )
        except (TypeError, ValueError):
            irregular_count = -1
        if irregular_count != 0:
            coherence_errors.append(f"irregular_sequence:{stream_name}")

    for stream_name in ("anchor_observations", "processing_stamps"):
        if reported_drop_counts.get(stream_name) != missing_count:
            coherence_errors.append(
                f"reported_drop_count_mismatch:{stream_name}"
            )
    for stream_name, drop_count in reported_drop_counts.items():
        if stream_name not in {"anchor_observations", "processing_stamps"}:
            if drop_count != 0:
                coherence_errors.append(
                    f"unexpected_reported_drop:{stream_name}"
                )

    accepted_error_codes = []
    if not coherence_errors and missing_count > 0:
        accepted_error_codes.extend(
            [
                f"recorded_processing_drop:anchor_observations:{missing_count}",
                f"recorded_processing_drop:processing_stamps:{missing_count}",
            ]
        )
        for stream_name in dependent_streams:
            accepted_error_codes.append(
                f"unexpected_processing_stamps:{stream_name}:{missing_count}"
            )

    return {
        "reference_stream": reference_stream,
        "reference_frame_count": reference_count,
        "anchor_observation_frame_count": anchor_count,
        "missing_anchor_frame_count": missing_count,
        "coverage_fraction": coverage_fraction,
        "drop_fraction": drop_fraction,
        "coherent": not coherence_errors,
        "coherence_errors": sorted(set(coherence_errors)),
        "reported_drop_counts": reported_drop_counts,
        "accepted_error_codes": sorted(set(accepted_error_codes)),
    }


def normalize_validation_policy(
    policy=VALIDATION_POLICY_STRICT,
    max_anchor_processing_drop_fraction=None,
):
    policy_payload = policy if isinstance(policy, dict) else {}
    policy_name = (
        str(policy_payload.get("name", VALIDATION_POLICY_STRICT)).strip()
        if isinstance(policy, dict)
        else str(policy or VALIDATION_POLICY_STRICT).strip()
    )
    if policy_name not in VALIDATION_POLICIES:
        raise ValueError(f"unsupported validation policy: {policy_name}")
    configured_fraction = max_anchor_processing_drop_fraction
    if configured_fraction is None and isinstance(policy, dict):
        configured_fraction = policy_payload.get(
            "max_anchor_processing_drop_fraction"
        )
    if configured_fraction is None:
        configured_fraction = DEFAULT_MAX_ANCHOR_PROCESSING_DROP_FRACTION
    try:
        configured_fraction = float(configured_fraction)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "max_anchor_processing_drop_fraction must be a finite number"
        ) from exc
    if (
        not math.isfinite(configured_fraction)
        or configured_fraction < 0.0
        or configured_fraction > 1.0
    ):
        raise ValueError(
            "max_anchor_processing_drop_fraction must be between 0 and 1"
        )
    return {
        "name": policy_name,
        "max_anchor_processing_drop_fraction": configured_fraction,
        "motion_execution_blocking": (
            policy_name != VALIDATION_POLICY_RECORDING_V2
        ),
    }


def classify_validation_findings(
    raw_errors,
    source_warnings,
    policy,
    processing_coverage,
):
    accepted = set()
    policy_name = policy["name"]
    drop_fraction = processing_coverage.get("drop_fraction")
    within_drop_limit = (
        processing_coverage.get("coherent", False)
        and drop_fraction is not None
        and drop_fraction
        <= float(policy["max_anchor_processing_drop_fraction"])
    )
    if policy_name != VALIDATION_POLICY_STRICT and within_drop_limit:
        allowed_gap_errors = set(
            processing_coverage.get("accepted_error_codes", [])
        )
        accepted.update(error for error in raw_errors if error in allowed_gap_errors)

    if policy_name == VALIDATION_POLICY_RECORDING_V2:
        diagnostic_prefixes = (
            "motion_execution:",
            "truth_gap_too_large:",
            "truth_starts_after_processing:",
            "truth_ends_before_processing:",
        )
        accepted.update(
            error for error in raw_errors if error.startswith(diagnostic_prefixes)
        )

    blocking = sorted(set(raw_errors) - accepted)
    accepted_warnings = sorted(accepted)
    warnings = sorted(set(source_warnings) | accepted)
    processing_coverage = dict(processing_coverage)
    processing_coverage["within_policy_limit"] = within_drop_limit
    processing_coverage["policy_max_drop_fraction"] = float(
        policy["max_anchor_processing_drop_fraction"]
    )
    return blocking, warnings, accepted_warnings, processing_coverage


def validate_truth_coverage(tracks, processing_stamps, errors):
    if not processing_stamps:
        return
    start = min(processing_stamps)
    end = max(processing_stamps)
    for model_name, stamps in tracks.items():
        if not stamps:
            continue
        if min(stamps) > start + STAMP_TOLERANCE_SEC:
            errors.append(f"truth_starts_after_processing:{model_name}")
        if max(stamps) < end - STAMP_TOLERANCE_SEC:
            errors.append(f"truth_ends_before_processing:{model_name}")


def validate_run(
    run_dir,
    write_report=True,
    policy=VALIDATION_POLICY_STRICT,
    max_anchor_processing_drop_fraction=None,
    report_path=None,
    stream_consumers=None,
):
    run_dir = pathlib.Path(run_dir).resolve()
    policy_config = normalize_validation_policy(
        policy,
        max_anchor_processing_drop_fraction=max_anchor_processing_drop_fraction,
    )
    errors = []
    warnings = []
    repairs = []
    run_info = load_json_file(run_dir / "meta" / "run_info.json", run_dir, errors)
    experiment_protocol = load_json_file(
        run_dir / "meta" / "experiment_protocol.json",
        run_dir,
        errors,
        required=False,
    )
    scenario_manifest = load_json_file(
        run_dir / "meta" / "scenario_manifest.json",
        run_dir,
        errors,
        required=False,
    )
    catalog_payload = load_json_file(
        run_dir / "meta" / "object_id_catalog.json", run_dir, errors
    )
    catalog = normalize_catalog(catalog_payload, errors)
    storage_backend = declared_storage_backend(run_info, errors)
    stream_consumers = stream_consumers or {}

    records_by_stream = {}
    for stream_name, filename in STREAM_FILES.items():
        if not (
            stream_name in HIGH_VOLUME_STREAMS
            and storage_backend in ("sqlite_zlib", "dual")
        ):
            records = load_jsonl_stream(
                run_dir / "algorithm" / filename,
                stream_name,
                run_dir,
                errors,
                repairs,
            )
            records_by_stream[stream_name] = records
            consumer = stream_consumers.get(stream_name)
            if consumer is not None:
                for record in records:
                    consumer(record)
    anchor_catalog_keys = set()
    for record in records_by_stream.get("anchor_catalog", []):
        anchor = record.get("anchor", {})
        try:
            anchor_catalog_keys.add(
                (
                    int(record.get("reference_epoch", anchor.get("reference_epoch", 0))),
                    int(record.get("anchor_id", anchor.get("id", 0))),
                )
            )
        except (TypeError, ValueError):
            continue
    for stream_name in HIGH_VOLUME_STREAMS:
        if storage_backend in ("sqlite_zlib", "dual"):
            records_by_stream[stream_name] = load_compressed_stream(
                run_dir,
                stream_name,
                errors,
                catalog=catalog,
                anchor_catalog_keys=anchor_catalog_keys,
                record_consumer=stream_consumers.get(stream_name),
            )

    storage_report = None
    if storage_backend in ("sqlite_zlib", "dual"):
        try:
            algorithm_recording = run_info.get("algorithm_recording", {})
            database_file = algorithm_recording.get(
                "database_file", "algorithm/algorithm_frames.sqlite3"
            )
            storage_report = inspect_sqlite_store(run_dir / database_file)
            if storage_report.get("integrity_check") != "ok":
                errors.append(
                    "algorithm_storage_integrity_failed:{}".format(
                        storage_report.get("integrity_check")
                    )
                )
        except (AlgorithmFrameStoreError, TypeError, ValueError) as exc:
            errors.append(f"algorithm_storage_inspection_failed:{exc}")
    if storage_backend == "dual":
        try:
            dual_report = compare_dual_storage(run_dir)
            if not dual_report.get("equivalent", False):
                errors.append("dual_algorithm_storage_mismatch")
        except AlgorithmFrameStoreError as exc:
            errors.append(f"dual_algorithm_storage_error:{exc}")

    conflicts_path = run_dir / "algorithm" / "anchor_catalog_conflicts.jsonl"
    if conflicts_path.is_file():
        conflict_records = load_jsonl_stream(
            conflicts_path,
            "anchor_catalog_conflicts",
            run_dir,
            errors,
            repairs,
        )
        if conflict_records:
            errors.append(
                f"anchor_catalog_conflicts_present:{len(conflict_records)}"
            )

    validate_object_ids(records_by_stream, catalog, errors)
    validate_reference_epochs(records_by_stream, errors)
    processing_stamps = validate_cross_stream_coverage(records_by_stream, errors)
    validate_alignment(run_dir, errors, repairs)
    object_rate_hz = 0.0
    if isinstance(run_info, dict):
        try:
            object_rate_hz = float(
                run_info.get("truth_recording", {}).get("object_rate_hz", 0.0)
            )
        except (TypeError, ValueError):
            object_rate_hz = 0.0
    if not math.isfinite(object_rate_hz) or object_rate_hz <= 0.0:
        errors.append("invalid_truth_object_rate_hz")
        object_rate_hz = 0.0
    tracks = load_truth_tracks(
        run_dir, catalog, object_rate_hz, errors, repairs
    )
    validate_truth_coverage(tracks, processing_stamps, errors)
    link_tracks = {}
    truth_recording = (
        run_info.get("truth_recording", {}) if isinstance(run_info, dict) else {}
    )
    experiment_factors = (
        scenario_manifest.get("experiment_factors", {})
        if isinstance(scenario_manifest, dict)
        else {}
    )
    experiment_factors = (
        experiment_factors if isinstance(experiment_factors, dict) else {}
    )
    surface_catalog_report = validate_static_surface_truth_catalog(
        run_dir,
        run_info,
        catalog,
        experiment_factors,
        errors,
        repairs,
    )
    dynamic_link_policy = truth_recording.get("dynamic_link_policy")
    direct_link_truth_enabled = dynamic_link_policy in {
        "sensor_and_surface_truth_links",
        "sensor_motion_drive_links_and_surface_catalog",
    }
    controller_configuration = (
        experiment_protocol.get("actual", {}).get("controller_configuration", {})
        if isinstance(experiment_protocol, dict)
        else {}
    )
    synchronized_models = (
        controller_configuration.get("models", [])
        if isinstance(controller_configuration, dict)
        else []
    )
    if synchronized_models and not direct_link_truth_enabled:
        errors.append("synchronized_motion_requires_direct_link_truth")
    if direct_link_truth_enabled:
        try:
            link_rate_hz = float(truth_recording.get("link_rate_hz", 0.0))
        except (TypeError, ValueError):
            link_rate_hz = 0.0
        if not math.isfinite(link_rate_hz) or link_rate_hz <= 0.0:
            errors.append("invalid_truth_link_rate_hz")
            link_rate_hz = 0.0
        expected_track_names = None
        catalog_requires_tracks = True
        if dynamic_link_policy == "sensor_motion_drive_links_and_surface_catalog":
            configured_drive_links = truth_recording.get(
                "motion_truth_drive_links", {}
            )
            configured_drive_links = (
                configured_drive_links
                if isinstance(configured_drive_links, dict)
                else {}
            )
            if configured_drive_links != CURRENT_DRIVE_LINK_CONTRACT:
                errors.append("motion_truth_drive_links_contract_mismatch")
            expected_track_names = set(configured_drive_links.values())
            catalog_requires_tracks = False
        link_tracks = load_direct_link_truth_tracks(
            run_dir,
            link_rate_hz,
            errors,
            repairs,
            expected_track_names=expected_track_names,
            catalog_requires_tracks=catalog_requires_tracks,
        )
        validate_truth_coverage(link_tracks, processing_stamps, errors)
    motion_execution_report = None
    if (
        experiment_factors.get("motion_protocol_version")
        == CURRENT_MOTION_PROTOCOL_VERSION
        or experiment_factors.get("truth_protocol_version")
        == CURRENT_TRUTH_PROTOCOL_VERSION
    ):
        motion_execution_report = validate_current_motion_execution(
            run_dir, write_report=write_report and report_path is None
        )
        errors.extend(
            "motion_execution:" + error
            for error in motion_execution_report.get("errors", [])
        )
    completion_payload = validate_completion(
        run_dir,
        records_by_stream,
        errors,
        storage_backend=storage_backend,
        storage_report=storage_report,
    )

    stream_report = {}
    for stream_name, records in records_by_stream.items():
        if isinstance(records, ScannedCompressedStream):
            stamps = list(records.stamps)
        else:
            stamps = [
                stamp
                for stamp in (record_time_sec(record) for record in records)
                if stamp is not None
            ]
        gaps = [right - left for left, right in zip(stamps, stamps[1:])]
        stream_report[stream_name] = {
            "row_count": len(records),
            "first_stamp_sec": min(stamps) if stamps else None,
            "last_stamp_sec": max(stamps) if stamps else None,
            "median_interval_sec": statistics.median(gaps) if gaps else None,
            "max_gap_sec": max(gaps) if gaps else None,
        }

    raw_errors = sorted(set(errors))
    source_warnings = sorted(set(warnings))
    repairs = sorted(set(repairs))
    processing_coverage = build_anchor_processing_coverage_report(
        records_by_stream,
        completion_payload,
    )
    errors, warnings, accepted_warnings, processing_coverage = (
        classify_validation_findings(
            raw_errors,
            source_warnings,
            policy_config,
            processing_coverage,
        )
    )
    status = "FAIL" if errors else ("WARN" if warnings or repairs else "PASS")
    report = {
        "schema_version": 2,
        "run_directory": str(run_dir),
        "status": status,
        "valid_for_analysis": not errors,
        "errors": errors,
        "blocking_errors": errors,
        "raw_errors": raw_errors,
        "warnings": warnings,
        "source_warnings": source_warnings,
        "accepted_warnings": accepted_warnings,
        "repairs": repairs,
        "validation_policy": policy_config,
        "processing_coverage": processing_coverage,
        "algorithm_storage": {
            "backend": storage_backend,
            "integrity_check": (
                storage_report.get("integrity_check")
                if isinstance(storage_report, dict)
                else None
            ),
        },
        "truth_link_tracks": {
            "required": direct_link_truth_enabled,
            "track_count": len(link_tracks),
            "sample_count": sum(len(stamps) for stamps in link_tracks.values()),
        },
        "surface_truth_catalog": surface_catalog_report,
        "motion_execution": motion_execution_report,
        "object_id_catalog": {str(key): value for key, value in sorted(catalog.items())},
        "streams": stream_report,
    }
    if write_report:
        destination = (
            pathlib.Path(report_path)
            if report_path is not None
            else run_dir / "analysis" / "data_quality_report.json"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=pathlib.Path)
    parser.add_argument("--no-write-report", action="store_true")
    parser.add_argument(
        "--policy",
        choices=sorted(VALIDATION_POLICIES),
        default=VALIDATION_POLICY_STRICT,
    )
    parser.add_argument(
        "--max-anchor-processing-drop-fraction",
        type=float,
        default=None,
    )
    args = parser.parse_args()
    report = validate_run(
        args.run_dir,
        write_report=not args.no_write_report,
        policy=args.policy,
        max_anchor_processing_drop_fraction=(
            args.max_anchor_processing_drop_fraction
        ),
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["valid_for_analysis"] else 1)


if __name__ == "__main__":
    main()
