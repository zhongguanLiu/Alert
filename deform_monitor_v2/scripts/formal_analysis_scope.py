#!/usr/bin/env python3

"""Formal experiment-protocol scope shared by ALERT analysis tools."""

import json
import math
import pathlib
from dataclasses import dataclass


PROTOCOL_RELATIVE_PATH = pathlib.Path("meta") / "experiment_protocol.json"
PHASE_REFERENCE_BUILD = "REFERENCE_BUILD"
PHASE_PRE_MOTION_STATIC = "PRE_MOTION_STATIC"
PHASE_MOTION = "MOTION"
PHASE_POST_MOTION_HOLD = "POST_MOTION_HOLD"
PHASE_NAMES = (
    PHASE_REFERENCE_BUILD,
    PHASE_PRE_MOTION_STATIC,
    PHASE_MOTION,
    PHASE_POST_MOTION_HOLD,
)


class FormalScopeError(ValueError):
    """Raised when a recorded formal protocol is missing or invalid."""


def _finite_number(value, label):
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise FormalScopeError("{} must be a finite number".format(label)) from exc
    if not math.isfinite(result):
        raise FormalScopeError("{} must be a finite number".format(label))
    return result


def time_sec(value):
    """Decode the time representations used by ROS JSON recorders."""

    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        result = float(value)
        return result if math.isfinite(result) else None
    if not isinstance(value, dict):
        return None
    if "sec" in value:
        try:
            result = float(value["sec"])
        except (TypeError, ValueError, OverflowError):
            return None
        return result if math.isfinite(result) else None
    if "secs" not in value:
        return None
    try:
        result = float(value["secs"]) + float(value.get("nsecs", 0)) * 1.0e-9
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def record_time_sec(record):
    if not isinstance(record, dict):
        return None
    header = record.get("header", {})
    if isinstance(header, dict):
        stamp = time_sec(header.get("stamp"))
        if stamp is not None:
            return stamp
    for key in ("stamp", "recorded_at", "recorded_time_sec", "time_sec"):
        stamp = time_sec(record.get(key))
        if stamp is not None:
            return stamp
    return None


@dataclass(frozen=True)
class FormalAnalysisScope:
    protocol_path: pathlib.Path
    protocol_schema_version: int
    scenario_id: str
    formal_epoch: int
    trial_start: float
    reference_ready: float
    motion_start: float
    motion_end: float
    trial_end: float

    @property
    def duration_sec(self):
        return self.trial_end - self.trial_start

    @property
    def motion_duration_sec(self):
        return self.motion_end - self.motion_start

    def contains_time(self, timestamp):
        try:
            timestamp = float(timestamp)
        except (TypeError, ValueError, OverflowError):
            return False
        return (
            math.isfinite(timestamp)
            and self.trial_start <= timestamp <= self.trial_end
        )

    def phase_at(self, timestamp):
        if not self.contains_time(timestamp):
            return None
        timestamp = float(timestamp)
        if timestamp < self.reference_ready:
            return PHASE_REFERENCE_BUILD
        if timestamp < self.motion_start:
            return PHASE_PRE_MOTION_STATIC
        if timestamp < self.motion_end:
            return PHASE_MOTION
        return PHASE_POST_MOTION_HOLD

    def phase_intervals(self):
        return [
            {
                "phase": PHASE_REFERENCE_BUILD,
                "start_time_sec": self.trial_start,
                "end_time_sec": self.reference_ready,
                "duration_sec": self.reference_ready - self.trial_start,
                "start_inclusive": True,
                "end_inclusive": False,
            },
            {
                "phase": PHASE_PRE_MOTION_STATIC,
                "start_time_sec": self.reference_ready,
                "end_time_sec": self.motion_start,
                "duration_sec": self.motion_start - self.reference_ready,
                "start_inclusive": True,
                "end_inclusive": False,
            },
            {
                "phase": PHASE_MOTION,
                "start_time_sec": self.motion_start,
                "end_time_sec": self.motion_end,
                "duration_sec": self.motion_end - self.motion_start,
                "start_inclusive": True,
                "end_inclusive": False,
            },
            {
                "phase": PHASE_POST_MOTION_HOLD,
                "start_time_sec": self.motion_end,
                "end_time_sec": self.trial_end,
                "duration_sec": self.trial_end - self.motion_end,
                "start_inclusive": True,
                "end_inclusive": True,
            },
        ]

    def record_in_scope(self, record, require_epoch_when_present=True):
        timestamp = record_time_sec(record)
        if not self.contains_time(timestamp):
            return False
        if not require_epoch_when_present or "reference_epoch" not in record:
            return True
        try:
            epoch = int(record.get("reference_epoch"))
        except (TypeError, ValueError, OverflowError):
            return False
        return epoch == self.formal_epoch

    def to_dict(self):
        return {
            "schema_version": 1,
            "scope_mode": "FORMAL_PROTOCOL",
            "protocol_path": str(self.protocol_path),
            "protocol_schema_version": self.protocol_schema_version,
            "scenario_id": self.scenario_id,
            "formal_epoch": self.formal_epoch,
            "trial_start_time_sec": self.trial_start,
            "reference_ready_time_sec": self.reference_ready,
            "motion_start_time_sec": self.motion_start,
            "motion_end_time_sec": self.motion_end,
            "trial_end_time_sec": self.trial_end,
            "duration_sec": self.duration_sec,
            "motion_duration_sec": self.motion_duration_sec,
            "time_interval_policy": "closed_trial_interval",
            "epoch_policy": "require_formal_epoch_when_stream_records_epoch",
            "phases": self.phase_intervals(),
        }


def load_formal_analysis_scope(run_dir, required=False):
    run_dir = pathlib.Path(run_dir).resolve()
    protocol_path = run_dir / PROTOCOL_RELATIVE_PATH
    if not protocol_path.is_file():
        if required:
            raise FormalScopeError(
                "formal experiment protocol is missing: {}".format(protocol_path)
            )
        return None
    try:
        protocol = json.loads(protocol_path.read_text())
    except json.JSONDecodeError as exc:
        raise FormalScopeError(
            "invalid experiment protocol JSON at {}:{}".format(
                protocol_path, exc.lineno
            )
        ) from exc
    if not isinstance(protocol, dict):
        raise FormalScopeError("experiment protocol root must be an object")
    if protocol.get("status") != "PASS":
        raise FormalScopeError(
            "experiment protocol status is not PASS: {!r}".format(
                protocol.get("status")
            )
        )
    if protocol.get("valid_for_analysis") is not True:
        raise FormalScopeError("experiment protocol is not valid_for_analysis")
    actual = protocol.get("actual")
    if not isinstance(actual, dict):
        raise FormalScopeError("experiment protocol actual section is missing")
    try:
        formal_epoch = int(actual.get("reference_epoch"))
    except (TypeError, ValueError, OverflowError) as exc:
        raise FormalScopeError("actual.reference_epoch must be an integer") from exc
    if formal_epoch <= 0:
        raise FormalScopeError("actual.reference_epoch must be greater than zero")

    scope = FormalAnalysisScope(
        protocol_path=protocol_path,
        protocol_schema_version=int(protocol.get("schema_version", 0)),
        scenario_id=str(protocol.get("scenario_id", "")),
        formal_epoch=formal_epoch,
        trial_start=_finite_number(
            actual.get("trial_start_time"), "actual.trial_start_time"
        ),
        reference_ready=_finite_number(
            actual.get("reference_initialized_at"),
            "actual.reference_initialized_at",
        ),
        motion_start=_finite_number(
            actual.get("actual_motion_start_time"),
            "actual.actual_motion_start_time",
        ),
        motion_end=_finite_number(
            actual.get("hold_start_time"), "actual.hold_start_time"
        ),
        trial_end=_finite_number(
            actual.get("trial_end_time"), "actual.trial_end_time"
        ),
    )
    ordered = (
        scope.trial_start
        <= scope.reference_ready
        <= scope.motion_start
        < scope.motion_end
        <= scope.trial_end
    )
    if not ordered or scope.duration_sec <= 0.0:
        raise FormalScopeError(
            "formal protocol timestamps are not ordered: "
            "trial_start <= reference_ready <= motion_start < motion_end <= trial_end"
        )
    return scope


class ScopedRecordSequence:
    """Replayable record view restricted to one formal protocol scope."""

    def __init__(self, records, scope, stream_name, predicate=None):
        self.records = records
        self.scope = scope
        self.stream_name = str(stream_name)
        self.predicate = predicate
        self._length = None
        self._audit_cache = None

    def __iter__(self):
        audit = {
            "stream": self.stream_name,
            "total_record_count": 0,
            "included_record_count": 0,
            "excluded_before_trial_count": 0,
            "excluded_after_trial_count": 0,
            "excluded_invalid_timestamp_count": 0,
            "excluded_epoch_count": 0,
            "excluded_predicate_count": 0,
            "included_epoch_record_count": 0,
            "included_epochless_record_count": 0,
            "first_included_time_sec": None,
            "last_included_time_sec": None,
            "phase_record_counts": {name: 0 for name in PHASE_NAMES},
        }
        for record in self.records:
            audit["total_record_count"] += 1
            timestamp = record_time_sec(record)
            if timestamp is None:
                audit["excluded_invalid_timestamp_count"] += 1
                continue
            if timestamp < self.scope.trial_start:
                audit["excluded_before_trial_count"] += 1
                continue
            if timestamp > self.scope.trial_end:
                audit["excluded_after_trial_count"] += 1
                continue
            has_epoch = "reference_epoch" in record
            if has_epoch:
                try:
                    epoch = int(record.get("reference_epoch"))
                except (TypeError, ValueError, OverflowError):
                    epoch = None
                if epoch != self.scope.formal_epoch:
                    audit["excluded_epoch_count"] += 1
                    continue
            if self.predicate is not None and not self.predicate(record):
                audit["excluded_predicate_count"] += 1
                continue
            if has_epoch:
                audit["included_epoch_record_count"] += 1
            else:
                audit["included_epochless_record_count"] += 1
            audit["included_record_count"] += 1
            if audit["first_included_time_sec"] is None:
                audit["first_included_time_sec"] = timestamp
            audit["last_included_time_sec"] = timestamp
            phase = self.scope.phase_at(timestamp)
            if phase is not None:
                audit["phase_record_counts"][phase] += 1
            yield record
        self._length = audit["included_record_count"]
        self._audit_cache = audit

    def __bool__(self):
        return next(iter(self), None) is not None

    def audit(self):
        if self._audit_cache is not None:
            return dict(self._audit_cache)
        result = {
            "stream": self.stream_name,
            "total_record_count": 0,
            "included_record_count": 0,
            "excluded_before_trial_count": 0,
            "excluded_after_trial_count": 0,
            "excluded_invalid_timestamp_count": 0,
            "excluded_epoch_count": 0,
            "excluded_predicate_count": 0,
            "included_epoch_record_count": 0,
            "included_epochless_record_count": 0,
            "first_included_time_sec": None,
            "last_included_time_sec": None,
            "phase_record_counts": {name: 0 for name in PHASE_NAMES},
        }
        for record in self.records:
            result["total_record_count"] += 1
            timestamp = record_time_sec(record)
            if timestamp is None:
                result["excluded_invalid_timestamp_count"] += 1
                continue
            if timestamp < self.scope.trial_start:
                result["excluded_before_trial_count"] += 1
                continue
            if timestamp > self.scope.trial_end:
                result["excluded_after_trial_count"] += 1
                continue
            has_epoch = "reference_epoch" in record
            if has_epoch:
                try:
                    epoch = int(record.get("reference_epoch"))
                except (TypeError, ValueError, OverflowError):
                    epoch = None
                if epoch != self.scope.formal_epoch:
                    result["excluded_epoch_count"] += 1
                    continue
            if self.predicate is not None and not self.predicate(record):
                result["excluded_predicate_count"] += 1
                continue
            if has_epoch:
                result["included_epoch_record_count"] += 1
            else:
                result["included_epochless_record_count"] += 1
            result["included_record_count"] += 1
            if result["first_included_time_sec"] is None:
                result["first_included_time_sec"] = timestamp
            result["last_included_time_sec"] = timestamp
            phase = self.scope.phase_at(timestamp)
            if phase is not None:
                result["phase_record_counts"][phase] += 1
        self._length = result["included_record_count"]
        self._audit_cache = result
        return result


def scoped_records(records, scope, stream_name, predicate=None):
    if records is None or scope is None:
        return records
    return ScopedRecordSequence(records, scope, stream_name, predicate=predicate)


def write_scope_json(path, scope, stream_audits=None, extra=None):
    payload = scope.to_dict()
    if stream_audits is not None:
        payload["stream_audits"] = stream_audits
    if extra:
        payload.update(extra)
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return path
