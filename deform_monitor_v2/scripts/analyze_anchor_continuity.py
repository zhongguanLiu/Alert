#!/usr/bin/env python3

import argparse
import csv
import importlib.util
import json
import math
import pathlib
import statistics
from dataclasses import dataclass


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
        "analyze_anchor_continuity_algorithm_frame_store", _storage_module_path
    )
    _storage_module = importlib.util.module_from_spec(_storage_spec)
    _storage_spec.loader.exec_module(_storage_module)
    AlgorithmFrameStoreError = _storage_module.AlgorithmFrameStoreError
    CompressedFrameStore = _storage_module.CompressedFrameStore
    ReplayableAlgorithmStream = _storage_module.ReplayableAlgorithmStream
    ReplayableSequence = _storage_module.ReplayableSequence
    iter_algorithm_stream = _storage_module.iter_algorithm_stream

try:
    from formal_analysis_scope import load_formal_analysis_scope, scoped_records
except ImportError:  # source-tree execution and importlib-based unit tests
    _scope_module_path = pathlib.Path(__file__).with_name("formal_analysis_scope.py")
    _scope_spec = importlib.util.spec_from_file_location(
        "analyze_anchor_continuity_formal_scope", _scope_module_path
    )
    _scope_module = importlib.util.module_from_spec(_scope_spec)
    _scope_spec.loader.exec_module(_scope_module)
    load_formal_analysis_scope = _scope_module.load_formal_analysis_scope
    scoped_records = _scope_module.scoped_records


ANCHOR_TYPE_NAMES = {
    0: "PLANE",
    1: "EDGE",
    2: "BAND",
}

OBS_STATE_NAMES = {
    0: "NOT_OBSERVABLE",
    1: "OBSERVABLE_MATCHED",
    2: "OBSERVABLE_WEAK",
    3: "OBSERVABLE_MISSING",
    4: "OBSERVABLE_REPLACED",
}

REFERENCE_ORIGIN_NAMES = {
    0: "INITIAL",
    1: "INCREMENTAL",
}

EVENT_HEADER = [
    "reference_epoch",
    "anchor_id",
    "anchor_type",
    "reference_origin",
    "reference_stamp_sec",
    "loss_start_time_sec",
    "recovery_time_sec",
    "recovery_latency_sec",
    "loss_state",
    "last_gap_state",
    "gap_sample_count",
    "recovered",
    "reacquired_flag",
    "closure_reason",
    "reference_preserved",
    "max_ref_center_drift_m",
    "pre_loss_displacement_x_m",
    "pre_loss_displacement_y_m",
    "pre_loss_displacement_z_m",
    "pre_loss_displacement_m",
    "recovery_displacement_x_m",
    "recovery_displacement_y_m",
    "recovery_displacement_z_m",
    "recovery_displacement_m",
    "displacement_change_across_gap_m",
    "measurement_identity_error_m",
]

TYPE_HEADER = [
    "anchor_type",
    "anchor_count",
    "initial_anchor_count",
    "incremental_anchor_count",
    "sample_count",
    "matched_sample_count",
    "loss_event_count",
    "initial_loss_event_count",
    "incremental_loss_event_count",
    "recovered_event_count",
    "open_or_reset_event_count",
    "recovery_rate",
    "mean_recovery_latency_sec",
    "median_recovery_latency_sec",
    "p95_recovery_latency_sec",
    "datum_preservation_rate",
    "max_ref_center_drift_m",
    "max_measurement_identity_error_m",
]

REQUIRED_RECORD_FIELDS = (
    "reference_epoch",
    "reference_initialized_at",
)

REQUIRED_ANCHOR_FIELDS = (
    "obs_state",
    "observable",
    "matched_center",
    "matched_delta",
    "ref_center",
    "reference_epoch",
    "reference_stamp",
    "reference_origin",
)

REFERENCE_TOLERANCE_M = 1.0e-9
REFERENCE_TIME_TOLERANCE_SEC = 1.0e-9


@dataclass(frozen=True)
class ContinuityOutputs:
    output_dir: pathlib.Path
    events_csv: pathlib.Path
    by_type_csv: pathlib.Path
    audit_json: pathlib.Path
    report_md: pathlib.Path


def time_sec(value):
    if value is None:
        return None
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
        result = float(value.get("secs", 0.0)) + float(value.get("nsecs", 0.0)) / 1e9
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def record_time_sec(record):
    header = record.get("header", {})
    stamp = header.get("stamp") if isinstance(header, dict) else None
    result = time_sec(stamp)
    if result is not None:
        return result
    return time_sec(record.get("recorded_at"))


def vector_tuple(value):
    if not isinstance(value, dict):
        return None
    try:
        result = (
            float(value.get("x", 0.0)),
            float(value.get("y", 0.0)),
            float(value.get("z", 0.0)),
        )
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(component) for component in result):
        return None
    return result


def vector_distance(left, right):
    if left is None or right is None:
        return None
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))


def vector_norm(value):
    if value is None:
        return None
    return math.sqrt(sum(component ** 2 for component in value))


def measurement_identity_error(anchor):
    ref_center = vector_tuple(anchor.get("ref_center"))
    matched_center = vector_tuple(anchor.get("matched_center"))
    matched_delta = vector_tuple(anchor.get("matched_delta"))
    if ref_center is None or matched_center is None or matched_delta is None:
        return None
    expected_delta = tuple(
        matched - reference for matched, reference in zip(matched_center, ref_center)
    )
    return vector_distance(matched_delta, expected_delta)


def anchor_type_name(value):
    try:
        key = int(value)
    except (TypeError, ValueError):
        return "UNKNOWN"
    return ANCHOR_TYPE_NAMES.get(key, f"UNKNOWN_{key}")


def obs_state_name(value):
    if value == "ANCHOR_NOT_PUBLISHED":
        return value
    try:
        key = int(value)
    except (TypeError, ValueError):
        return "UNKNOWN"
    return OBS_STATE_NAMES.get(key, f"UNKNOWN_{key}")


def reference_origin_name(value):
    try:
        key = int(value)
    except (TypeError, ValueError):
        return "UNKNOWN"
    return REFERENCE_ORIGIN_NAMES.get(key, f"UNKNOWN_{key}")


def is_matched(anchor):
    try:
        obs_state = int(anchor.get("obs_state", -1))
    except (TypeError, ValueError):
        return False
    return obs_state == 1 and bool(anchor.get("observable", False)) and bool(
        anchor.get("comparable", False)
    )


def percentile(values, percentile_value):
    if not values:
        return ""
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile_value / 100.0
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _reference_metadata(anchor):
    return {
        "ref_center": vector_tuple(anchor.get("ref_center")),
        "reference_stamp_sec": time_sec(anchor.get("reference_stamp")),
        "reference_origin": anchor.get("reference_origin"),
        "anchor_type": anchor_type_name(anchor.get("anchor_type")),
    }


def _metadata_drift(anchor, baseline):
    current = _reference_metadata(anchor)
    center_drift = vector_distance(current["ref_center"], baseline["ref_center"])
    center_drift = center_drift if center_drift is not None else math.inf
    stamp_changed = (
        current["reference_stamp_sec"] is None
        or baseline["reference_stamp_sec"] is None
        or abs(current["reference_stamp_sec"] - baseline["reference_stamp_sec"])
        > REFERENCE_TIME_TOLERANCE_SEC
    )
    origin_changed = current["reference_origin"] != baseline["reference_origin"]
    type_changed = current["anchor_type"] != baseline["anchor_type"]
    preserved = (
        center_drift <= REFERENCE_TOLERANCE_M
        and not stamp_changed
        and not origin_changed
        and not type_changed
    )
    return preserved, center_drift


def _new_event(
    epoch, anchor, timestamp, loss_state, metadata_state, previous_anchor=None
):
    key = (epoch, int(anchor.get("id", -1)))
    baseline = metadata_state[key]["baseline"]
    pre_loss_delta = vector_tuple(
        (previous_anchor or anchor).get("matched_delta")
    )
    pre_loss_components = pre_loss_delta or ("", "", "")
    return {
        "reference_epoch": epoch,
        "anchor_id": key[1],
        "anchor_type": baseline["anchor_type"],
        "reference_origin": reference_origin_name(baseline["reference_origin"]),
        "reference_stamp_sec": baseline["reference_stamp_sec"],
        "loss_start_time_sec": timestamp,
        "recovery_time_sec": "",
        "recovery_latency_sec": "",
        "loss_state": obs_state_name(loss_state),
        "last_gap_state": obs_state_name(loss_state),
        "gap_sample_count": 1,
        "recovered": False,
        "reacquired_flag": False,
        "closure_reason": "END_OF_RUN",
        "reference_preserved": not metadata_state[key]["mutated"],
        "max_ref_center_drift_m": metadata_state[key]["max_center_drift_m"],
        "pre_loss_displacement_x_m": pre_loss_components[0],
        "pre_loss_displacement_y_m": pre_loss_components[1],
        "pre_loss_displacement_z_m": pre_loss_components[2],
        "pre_loss_displacement_m": (
            vector_norm(pre_loss_delta) if pre_loss_delta is not None else ""
        ),
        "recovery_displacement_x_m": "",
        "recovery_displacement_y_m": "",
        "recovery_displacement_z_m": "",
        "recovery_displacement_m": "",
        "displacement_change_across_gap_m": "",
        "measurement_identity_error_m": "",
    }


def _refresh_event_reference_status(event, metadata):
    event["reference_preserved"] = not metadata["mutated"]
    event["max_ref_center_drift_m"] = metadata["max_center_drift_m"]


def _build_type_rows(events, metadata_state, sample_stats):
    all_type_names = list(ANCHOR_TYPE_NAMES.values())
    unknown_names = sorted(
        {
            metadata["baseline"]["anchor_type"]
            for metadata in metadata_state.values()
            if metadata["baseline"]["anchor_type"] not in all_type_names
        }
    )
    rows = []
    for type_name in all_type_names + unknown_names:
        type_events = [event for event in events if event["anchor_type"] == type_name]
        recovered = [event for event in type_events if event["recovered"]]
        latencies = [event["recovery_latency_sec"] for event in recovered]
        anchor_metadata = [
            metadata
            for metadata in metadata_state.values()
            if metadata["baseline"]["anchor_type"] == type_name
        ]
        preserved_events = [event for event in type_events if event["reference_preserved"]]
        row_stats = sample_stats.get(type_name, {})
        rows.append(
            {
                "anchor_type": type_name,
                "anchor_count": len(anchor_metadata),
                "initial_anchor_count": sum(
                    1
                    for metadata in anchor_metadata
                    if metadata["baseline"]["reference_origin"] == 0
                ),
                "incremental_anchor_count": sum(
                    1
                    for metadata in anchor_metadata
                    if metadata["baseline"]["reference_origin"] == 1
                ),
                "sample_count": int(row_stats.get("sample_count", 0)),
                "matched_sample_count": int(row_stats.get("matched_sample_count", 0)),
                "loss_event_count": len(type_events),
                "initial_loss_event_count": sum(
                    1
                    for event in type_events
                    if event["reference_origin"] == "INITIAL"
                ),
                "incremental_loss_event_count": sum(
                    1
                    for event in type_events
                    if event["reference_origin"] == "INCREMENTAL"
                ),
                "recovered_event_count": len(recovered),
                "open_or_reset_event_count": len(type_events) - len(recovered),
                "recovery_rate": (
                    float(len(recovered)) / len(type_events) if type_events else ""
                ),
                "mean_recovery_latency_sec": (
                    statistics.fmean(latencies) if latencies else ""
                ),
                "median_recovery_latency_sec": (
                    statistics.median(latencies) if latencies else ""
                ),
                "p95_recovery_latency_sec": percentile(latencies, 95.0),
                "datum_preservation_rate": (
                    float(len(preserved_events)) / len(type_events) if type_events else ""
                ),
                "max_ref_center_drift_m": max(
                    (metadata["max_center_drift_m"] for metadata in anchor_metadata),
                    default=0.0,
                ),
                "max_measurement_identity_error_m": row_stats.get(
                    "max_measurement_identity_error_m", ""
                ),
            }
        )
    return rows


def analyze_records(records):
    metadata_state = {}
    observation_state = {}
    open_events = {}
    events = []
    sample_stats = {}
    epoch_sequence = []
    epoch_initialized_at = {}
    initialized_at_mutated_epochs = set()
    schema_missing_record_count = 0
    schema_missing_anchor_count = 0
    anchor_epoch_mismatch_count = 0
    initialized_stamp_missing_with_anchors_count = 0
    anchor_sample_count = 0
    reset_events = []
    previous_epoch = None
    last_timestamp = None
    previous_input_timestamp = None
    invalid_timestamp_count = 0
    record_count = 0

    for index, record in enumerate(records):
        timestamp = record_time_sec(record)
        if timestamp is None:
            invalid_timestamp_count += 1
            continue
        if (
            previous_input_timestamp is not None
            and timestamp < previous_input_timestamp
        ):
            raise ValueError(
                "anchor observation stream is non-monotonic at record {}: "
                "{} < {}".format(index + 1, timestamp, previous_input_timestamp)
            )
        previous_input_timestamp = timestamp
        record_count += 1
        last_timestamp = timestamp
        if any(field not in record for field in REQUIRED_RECORD_FIELDS):
            schema_missing_record_count += 1
        try:
            epoch = int(record.get("reference_epoch", 0))
        except (TypeError, ValueError):
            epoch = 0

        initialized_at = time_sec(record.get("reference_initialized_at"))
        if initialized_at is not None and initialized_at <= 0.0:
            initialized_at = None
        if epoch not in epoch_initialized_at:
            epoch_initialized_at[epoch] = initialized_at
        else:
            baseline_initialized_at = epoch_initialized_at[epoch]
            if baseline_initialized_at is None and initialized_at is not None:
                epoch_initialized_at[epoch] = initialized_at
            elif (
                baseline_initialized_at is not None
                and initialized_at is not None
                and abs(initialized_at - baseline_initialized_at)
                > REFERENCE_TIME_TOLERANCE_SEC
            ):
                initialized_at_mutated_epochs.add(epoch)

        if previous_epoch is None or epoch != previous_epoch:
            epoch_sequence.append(epoch)
        if previous_epoch is not None and epoch != previous_epoch:
            reset_events.append(
                {
                    "time_sec": timestamp,
                    "previous_reference_epoch": previous_epoch,
                    "new_reference_epoch": epoch,
                }
            )
            for key, event in list(open_events.items()):
                if key[0] != previous_epoch:
                    continue
                _refresh_event_reference_status(event, metadata_state[key])
                event["closure_reason"] = "REFERENCE_RESET"
                events.append(event)
                del open_events[key]
        previous_epoch = epoch

        anchors = record.get("anchors", [])
        if not isinstance(anchors, list):
            anchors = []
        if anchors and initialized_at is None:
            initialized_stamp_missing_with_anchors_count += 1
        seen_keys = set()
        for anchor in anchors:
            if not isinstance(anchor, dict):
                schema_missing_anchor_count += 1
                continue
            anchor_sample_count += 1
            if any(field not in anchor for field in REQUIRED_ANCHOR_FIELDS):
                schema_missing_anchor_count += 1
            try:
                anchor_id = int(anchor.get("id", -1))
                anchor_epoch = int(anchor.get("reference_epoch", epoch))
            except (TypeError, ValueError):
                continue
            if anchor_epoch != epoch:
                anchor_epoch_mismatch_count += 1
            key = (anchor_epoch, anchor_id)
            seen_keys.add(key)

            if key not in metadata_state:
                metadata_state[key] = {
                    "baseline": _reference_metadata(anchor),
                    "mutated": False,
                    "max_center_drift_m": 0.0,
                }
            else:
                preserved, center_drift = _metadata_drift(
                    anchor, metadata_state[key]["baseline"]
                )
                if math.isfinite(center_drift):
                    metadata_state[key]["max_center_drift_m"] = max(
                        metadata_state[key]["max_center_drift_m"], center_drift
                    )
                else:
                    metadata_state[key]["max_center_drift_m"] = math.inf
                if not preserved:
                    metadata_state[key]["mutated"] = True

            type_name = metadata_state[key]["baseline"]["anchor_type"]
            type_stats = sample_stats.setdefault(
                type_name,
                {
                    "sample_count": 0,
                    "matched_sample_count": 0,
                    "max_measurement_identity_error_m": "",
                },
            )
            type_stats["sample_count"] += 1

            matched = is_matched(anchor)
            if matched:
                type_stats["matched_sample_count"] += 1
                identity_error = measurement_identity_error(anchor)
                if identity_error is not None:
                    previous_max = type_stats["max_measurement_identity_error_m"]
                    type_stats["max_measurement_identity_error_m"] = max(
                        identity_error,
                        previous_max if previous_max != "" else 0.0,
                    )
            else:
                identity_error = None

            previous = observation_state.get(key)
            if matched and key in open_events:
                event = open_events.pop(key)
                recovery_delta = vector_tuple(anchor.get("matched_delta"))
                pre_loss_delta = (
                    event["pre_loss_displacement_x_m"],
                    event["pre_loss_displacement_y_m"],
                    event["pre_loss_displacement_z_m"],
                )
                if recovery_delta is not None:
                    event["recovery_displacement_x_m"] = recovery_delta[0]
                    event["recovery_displacement_y_m"] = recovery_delta[1]
                    event["recovery_displacement_z_m"] = recovery_delta[2]
                    event["recovery_displacement_m"] = vector_norm(recovery_delta)
                    if all(value != "" for value in pre_loss_delta):
                        event["displacement_change_across_gap_m"] = vector_distance(
                            recovery_delta, pre_loss_delta
                        )
                event["recovery_time_sec"] = timestamp
                event["recovery_latency_sec"] = max(
                    0.0, timestamp - event["loss_start_time_sec"]
                )
                event["recovered"] = True
                event["reacquired_flag"] = bool(anchor.get("reacquired", False))
                event["closure_reason"] = "RECOVERED"
                event["measurement_identity_error_m"] = (
                    identity_error if identity_error is not None else ""
                )
                _refresh_event_reference_status(event, metadata_state[key])
                events.append(event)
            elif not matched and previous is not None and previous["matched"]:
                open_events[key] = _new_event(
                    anchor_epoch,
                    anchor,
                    timestamp,
                    anchor.get("obs_state", -1),
                    metadata_state,
                    previous_anchor=previous["anchor"],
                )
            elif not matched and key in open_events:
                event = open_events[key]
                event["last_gap_state"] = obs_state_name(anchor.get("obs_state", -1))
                event["gap_sample_count"] += 1
                _refresh_event_reference_status(event, metadata_state[key])

            observation_state[key] = {
                "matched": matched,
                "timestamp": timestamp,
                "anchor": anchor,
            }

        known_current_epoch_keys = [
            key for key in observation_state if key[0] == epoch
        ]
        for key in known_current_epoch_keys:
            if key in seen_keys:
                continue
            previous = observation_state[key]
            if previous["matched"] and key not in open_events:
                open_events[key] = _new_event(
                    key[0],
                    previous["anchor"],
                    timestamp,
                    "ANCHOR_NOT_PUBLISHED",
                    metadata_state,
                    previous_anchor=previous["anchor"],
                )
            elif key in open_events:
                open_events[key]["last_gap_state"] = "ANCHOR_NOT_PUBLISHED"
                open_events[key]["gap_sample_count"] += 1
            observation_state[key]["matched"] = False
            observation_state[key]["timestamp"] = timestamp

    for key, event in open_events.items():
        _refresh_event_reference_status(event, metadata_state[key])
        if last_timestamp is not None:
            event["last_gap_time_sec"] = last_timestamp
        events.append(event)

    events.sort(
        key=lambda event: (
            event["loss_start_time_sec"],
            event["reference_epoch"],
            event["anchor_id"],
        )
    )
    by_type = _build_type_rows(events, metadata_state, sample_stats)

    datum_mutation_count = sum(
        1 for metadata in metadata_state.values() if metadata["mutated"]
    )
    max_ref_center_drift = max(
        (metadata["max_center_drift_m"] for metadata in metadata_state.values()),
        default=0.0,
    )
    lifecycle_schema_complete = (
        schema_missing_record_count == 0 and schema_missing_anchor_count == 0
    )
    audit = {
        "record_count": record_count,
        "invalid_timestamp_record_count": invalid_timestamp_count,
        "anchor_sample_count": anchor_sample_count,
        "distinct_anchor_count": len(metadata_state),
        "initial_anchor_count": sum(
            1
            for metadata in metadata_state.values()
            if metadata["baseline"]["reference_origin"] == 0
        ),
        "incremental_anchor_count": sum(
            1
            for metadata in metadata_state.values()
            if metadata["baseline"]["reference_origin"] == 1
        ),
        "reference_epochs": epoch_sequence,
        "reference_epoch_count": len(set(epoch_sequence)),
        "reference_epoch_change_count": len(reset_events),
        "reference_reset_events": reset_events,
        "continuous_single_epoch_run": len(reset_events) == 0,
        "reference_initialized_at_by_epoch": {
            str(epoch): value for epoch, value in sorted(epoch_initialized_at.items())
        },
        "reference_initialized_at_mutation_count": len(
            initialized_at_mutated_epochs
        ),
        "datum_mutation_count": datum_mutation_count,
        "max_ref_center_drift_m": max_ref_center_drift,
        "lifecycle_schema_complete": lifecycle_schema_complete,
        "schema_missing_record_count": schema_missing_record_count,
        "schema_missing_anchor_count": schema_missing_anchor_count,
        "anchor_epoch_mismatch_count": anchor_epoch_mismatch_count,
        "initialized_stamp_missing_with_anchors_count": (
            initialized_stamp_missing_with_anchors_count
        ),
        "continuity_evidence_valid": (
            lifecycle_schema_complete
            and invalid_timestamp_count == 0
            and datum_mutation_count == 0
            and not initialized_at_mutated_epochs
            and anchor_epoch_mismatch_count == 0
            and initialized_stamp_missing_with_anchors_count == 0
        ),
    }
    return {
        "events": events,
        "by_type": by_type,
        "audit": audit,
    }


def load_jsonl(path):
    records = []
    with pathlib.Path(path).open() as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                records.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
    return records


def iter_reconstructed_compact_anchor_records(catalog_records, observation_records):
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

    for observation in observation_records:
        try:
            epoch = int(observation.get("reference_epoch", 0))
        except (TypeError, ValueError):
            epoch = 0
        reconstructed = {
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
            reconstructed["anchors"].append(anchor)
        yield reconstructed


def reconstruct_compact_anchor_records(catalog_records, observation_records):
    return list(
        iter_reconstructed_compact_anchor_records(
            catalog_records,
            observation_records,
        )
    )


def load_anchor_state_records(run_dir, input_path=None):
    run_dir = pathlib.Path(run_dir)
    if input_path is not None:
        return load_jsonl(pathlib.Path(input_path))

    algorithm_dir = run_dir / "algorithm"
    legacy_path = algorithm_dir / "anchor_states.jsonl"
    if legacy_path.is_file():
        return load_jsonl(legacy_path)

    catalog_path = algorithm_dir / "anchor_catalog.jsonl"
    if not catalog_path.is_file():
        raise FileNotFoundError(
            "Anchor records require legacy anchor_states.jsonl or "
            "anchor_catalog.jsonl plus the logical anchor_observations stream"
        )
    observations = ReplayableAlgorithmStream(
        run_dir,
        "anchor_observations",
        required=True,
    )
    catalog = load_jsonl(catalog_path)
    return ReplayableSequence(
        lambda: iter_reconstructed_compact_anchor_records(catalog, observations)
    )


def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_report(path, run_dir, result):
    audit = result["audit"]
    lines = [
        "# Anchor Reference Continuity Audit",
        "",
        f"- Run: `{run_dir}`",
        f"- Lifecycle schema complete: `{audit['lifecycle_schema_complete']}`",
        f"- Continuity evidence valid: `{audit['continuity_evidence_valid']}`",
        f"- Reference epochs: `{audit['reference_epochs']}`",
        f"- Recorded reference resets: `{audit['reference_epoch_change_count']}`",
        f"- Reference datum mutations: `{audit['datum_mutation_count']}`",
        f"- Maximum reference-center drift: `{audit['max_ref_center_drift_m']:.9g} m`",
        "",
        "## Visibility Recovery By Anchor Type",
        "",
        "| Type | Anchors | Loss events | Recovered | Recovery rate | Mean latency (s) | Datum preservation |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in result["by_type"]:
        lines.append(
            "| {anchor_type} | {anchor_count} | {loss_event_count} | "
            "{recovered_event_count} | {recovery_rate} | "
            "{mean_recovery_latency_sec} | {datum_preservation_rate} |".format(**row)
        )
    lines.extend(
        [
            "",
            "A visibility recovery is counted only when the same `(reference_epoch, anchor_id)` "
            "returns to `OBSERVABLE_MATCHED`. An epoch change closes an open gap as "
            "`REFERENCE_RESET`; it is never counted as recovery.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def analyze_run(
    run_dir,
    input_path=None,
    output_dir=None,
    record_iterable=None,
):
    run_dir = pathlib.Path(run_dir).resolve()
    formal_scope = load_formal_analysis_scope(run_dir, required=False)
    input_path = pathlib.Path(input_path).resolve() if input_path is not None else None
    if input_path is not None and not input_path.is_file():
        raise FileNotFoundError(f"Anchor-state record not found: {input_path}")
    output_dir = (
        pathlib.Path(output_dir).resolve()
        if output_dir is not None
        else run_dir / "analysis" / "formal" / "anchor_continuity"
        if formal_scope is not None
        else run_dir / "analysis" / "anchor_continuity"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_records = (
        record_iterable
        if record_iterable is not None
        else load_anchor_state_records(run_dir, input_path)
    )
    records = scoped_records(raw_records, formal_scope, "anchor_observations")
    result = analyze_records(records)
    if formal_scope is not None:
        result["audit"].update(
            {
                "analysis_scope_mode": "FORMAL_PROTOCOL",
                "formal_reference_epoch": formal_scope.formal_epoch,
                "formal_trial_start_time_sec": formal_scope.trial_start,
                "formal_trial_end_time_sec": formal_scope.trial_end,
                "prescribed_pretrial_reference_reset_count": 1,
                "prescribed_pretrial_reference_reset_excluded": True,
                "runtime_reference_reset_count": result["audit"].get(
                    "reference_epoch_change_count", 0
                ),
                "scope_stream_audit": records.audit(),
            }
        )
    else:
        result["audit"]["analysis_scope_mode"] = "LEGACY_UNSCOPED"
    events_csv = output_dir / "anchor_continuity_events.csv"
    by_type_csv = output_dir / "anchor_continuity_by_type.csv"
    audit_json = output_dir / "anchor_reference_audit.json"
    report_md = output_dir / "anchor_continuity_report.md"
    write_csv(events_csv, EVENT_HEADER, result["events"])
    write_csv(by_type_csv, TYPE_HEADER, result["by_type"])
    with audit_json.open("w") as handle:
        json.dump(result["audit"], handle, indent=2, sort_keys=True)
        handle.write("\n")
    write_report(report_md, run_dir, result)
    return ContinuityOutputs(
        output_dir=output_dir,
        events_csv=events_csv,
        by_type_csv=by_type_csv,
        audit_json=audit_json,
        report_md=report_md,
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Audit anchor visibility recovery and verify that reference metadata remains "
            "frozen within each reference epoch."
        )
    )
    parser.add_argument("run_dir", type=pathlib.Path)
    parser.add_argument("--input", dest="input_path", type=pathlib.Path)
    parser.add_argument("--output-dir", type=pathlib.Path)
    args = parser.parse_args()

    outputs = analyze_run(args.run_dir, args.input_path, args.output_dir)
    print(f"events_csv: {outputs.events_csv}")
    print(f"by_type_csv: {outputs.by_type_csv}")
    print(f"audit_json: {outputs.audit_json}")
    print(f"report_md: {outputs.report_md}")


if __name__ == "__main__":
    main()
