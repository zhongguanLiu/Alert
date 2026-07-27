#!/usr/bin/env python3

import argparse
import collections
import json
import math
import pathlib
import threading
import time

import yaml

try:
    import rospy
    import sensor_msgs.point_cloud2 as point_cloud2
    from deform_monitor_v2.msg import AnchorStates, ObjectObservationStats
    from livox_ros_driver.msg import CustomMsg
    from sensor_msgs.msg import PointCloud2
except ImportError:  # pragma: no cover - pure helper tests do not require ROS
    rospy = None
    point_cloud2 = None
    AnchorStates = None
    ObjectObservationStats = None
    CustomMsg = None
    PointCloud2 = None


ANCHOR_TYPE_NAMES = {0: "PLANE", 1: "EDGE", 2: "BAND"}
ASSOCIATION_STATE_NAMES = {
    0: "unavailable",
    1: "consistent",
    2: "mismatch",
    3: "mixed",
}


def normalize_object_id(value):
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0
    if not math.isfinite(numeric):
        return 0
    rounded = int(round(numeric))
    if rounded < 1 or rounded > 254 or abs(numeric - rounded) > 1.0e-6:
        return 0
    return rounded


def _field(item, name, default=None):
    if isinstance(item, dict):
        return item.get(name, default)
    return getattr(item, name, default)


def _confidence_summary(values):
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {"count": 0, "mean": None, "min": None, "max": None}
    return {
        "count": len(finite),
        "mean": sum(finite) / len(finite),
        "min": min(finite),
        "max": max(finite),
    }


class AssociationAccumulator:
    def __init__(self, catalog):
        self.catalog = {
            int(object_id): str(model_name)
            for object_id, model_name in sorted(catalog.items())
        }
        self.raw_counts = collections.Counter()
        self.fast_counts = collections.Counter()
        self.raw_invalid_count = 0
        self.fast_invalid_count = 0
        self.raw_message_count = 0
        self.fast_message_count = 0
        self.stats_message_count = 0
        self.anchor_message_count = 0
        self.latest_label_stats = {
            "total_point_count": 0,
            "valid_label_point_count": 0,
            "invalid_label_point_count": 0,
            "object_point_counts": {},
        }
        self.reference_epoch = 0
        self.latest_anchors = []
        self._lock = threading.Lock()

    def add_raw_reflectivities(self, values):
        with self._lock:
            self.raw_message_count += 1
            for value in values:
                object_id = normalize_object_id(value)
                if object_id:
                    self.raw_counts[object_id] += 1
                else:
                    self.raw_invalid_count += 1

    def add_fast_lio_intensities(self, values):
        with self._lock:
            self.fast_message_count += 1
            for value in values:
                object_id = normalize_object_id(value)
                if object_id:
                    self.fast_counts[object_id] += 1
                else:
                    self.fast_invalid_count += 1

    def add_observation_stats(
        self,
        total_point_count,
        valid_label_point_count,
        invalid_label_point_count,
        object_point_counts,
    ):
        with self._lock:
            self.stats_message_count += 1
            self.latest_label_stats = {
                "total_point_count": int(total_point_count),
                "valid_label_point_count": int(valid_label_point_count),
                "invalid_label_point_count": int(invalid_label_point_count),
                "object_point_counts": {
                    int(object_id): int(count)
                    for object_id, count in object_point_counts.items()
                },
            }

    def set_anchor_snapshot(self, reference_epoch, anchors):
        with self._lock:
            self.anchor_message_count += 1
            self.reference_epoch = int(reference_epoch)
            self.latest_anchors = list(anchors)

    def build_report(self, expected_visible_ids=None, expected_evaluable_ids=None):
        expected_visible_ids = {
            int(value) for value in (expected_visible_ids or set())
        }
        expected_evaluable_ids = {
            int(value) for value in (expected_evaluable_ids or set())
        }
        with self._lock:
            raw_counts = collections.Counter(self.raw_counts)
            fast_counts = collections.Counter(self.fast_counts)
            label_stats = dict(self.latest_label_stats)
            label_stats["object_point_counts"] = dict(
                self.latest_label_stats["object_point_counts"]
            )
            anchors = list(self.latest_anchors)
            topic_counts = {
                "livox_lidar": self.raw_message_count,
                "cloud_registered": self.fast_message_count,
                "object_observation_stats": self.stats_message_count,
                "anchors": self.anchor_message_count,
            }
            raw_invalid_count = self.raw_invalid_count
            fast_invalid_count = self.fast_invalid_count
            reference_epoch = self.reference_epoch

        per_object_anchors = collections.defaultdict(list)
        initial_count = 0
        incremental_count = 0
        invalid_anchor_id_count = 0
        for anchor in anchors:
            if int(_field(anchor, "reference_origin", 0)) == 1:
                incremental_count += 1
            else:
                initial_count += 1
            object_id = normalize_object_id(_field(anchor, "object_id", 0))
            if not bool(_field(anchor, "object_id_valid", False)) or not object_id:
                invalid_anchor_id_count += 1
                continue
            per_object_anchors[object_id].append(anchor)

        objects = {}
        for object_id, model_name in self.catalog.items():
            associated = per_object_anchors.get(object_id, [])
            type_counts = {name: 0 for name in ANCHOR_TYPE_NAMES.values()}
            association_counts = {name: 0 for name in ASSOCIATION_STATE_NAMES.values()}
            confidences = []
            observed_confidences = []
            for anchor in associated:
                type_name = ANCHOR_TYPE_NAMES.get(
                    int(_field(anchor, "anchor_type", -1)), "UNKNOWN"
                )
                type_counts[type_name] = type_counts.get(type_name, 0) + 1
                state_name = ASSOCIATION_STATE_NAMES.get(
                    int(_field(anchor, "object_association_state", 0)), "unavailable"
                )
                association_counts[state_name] += 1
                confidence = _field(anchor, "object_id_confidence", 0.0)
                try:
                    confidences.append(float(confidence))
                except (TypeError, ValueError):
                    pass
                observed_confidence = _field(
                    anchor, "observed_object_id_confidence", 0.0
                )
                try:
                    observed_confidences.append(float(observed_confidence))
                except (TypeError, ValueError):
                    pass
            objects[str(object_id)] = {
                "model_name": model_name,
                "raw_livox_point_count": int(raw_counts.get(object_id, 0)),
                "fast_lio_point_count": int(fast_counts.get(object_id, 0)),
                "alert_point_count": int(
                    label_stats["object_point_counts"].get(object_id, 0)
                ),
                "valid_associated_anchor_count": len(associated),
                "anchor_types": type_counts,
                "object_id_confidence": _confidence_summary(confidences),
                "observed_object_id_confidence": _confidence_summary(
                    observed_confidences
                ),
                "association": association_counts,
            }

        errors = []
        warnings = []
        for object_id in sorted(expected_visible_ids | expected_evaluable_ids):
            if object_id not in self.catalog:
                errors.append(f"expected_object_id_not_in_catalog:{object_id}")
        for object_id in sorted(expected_visible_ids):
            item = objects.get(str(object_id))
            if item is None:
                continue
            if item["raw_livox_point_count"] == 0:
                errors.append(f"visible_object_missing_raw_livox_points:{object_id}")
            if item["fast_lio_point_count"] == 0:
                errors.append(f"visible_object_missing_fast_lio_points:{object_id}")
            if item["alert_point_count"] == 0:
                errors.append(f"visible_object_missing_alert_points:{object_id}")
        for object_id in sorted(expected_evaluable_ids):
            item = objects.get(str(object_id))
            if item is None:
                continue
            if item["valid_associated_anchor_count"] == 0:
                errors.append(f"evaluable_object_missing_anchor_association:{object_id}")
            for anchor_type in ("EDGE", "BAND"):
                if item["anchor_types"].get(anchor_type, 0) == 0:
                    warnings.append(
                        f"low_anchor_type_count:{object_id}:{anchor_type}:0"
                    )

        for topic, count in topic_counts.items():
            if count == 0:
                warnings.append(f"no_messages_received:{topic}")

        total_labels = int(label_stats["total_point_count"])
        valid_labels = int(label_stats["valid_label_point_count"])
        report = {
            "status": "FAIL" if errors else "PASS",
            "errors": errors,
            "warnings": warnings,
            "catalog": {
                str(object_id): model_name
                for object_id, model_name in self.catalog.items()
            },
            "expected_visible_ids": sorted(expected_visible_ids),
            "expected_evaluable_ids": sorted(expected_evaluable_ids),
            "topics": {
                topic: {"message_count": count}
                for topic, count in topic_counts.items()
            },
            "raw_livox": {
                "valid_id_point_count": int(sum(raw_counts.values())),
                "invalid_id_point_count": int(raw_invalid_count),
            },
            "fast_lio": {
                "valid_id_point_count": int(sum(fast_counts.values())),
                "invalid_id_point_count": int(fast_invalid_count),
            },
            "labels": {
                "total_point_count": total_labels,
                "valid_point_count": valid_labels,
                "invalid_point_count": int(label_stats["invalid_label_point_count"]),
                "valid_ratio": (float(valid_labels) / total_labels) if total_labels else 0.0,
            },
            "anchors": {
                "reference_epoch": int(reference_epoch),
                "total_count": len(anchors),
                "initial_count": initial_count,
                "incremental_count": incremental_count,
                "invalid_object_id_count": invalid_anchor_id_count,
            },
            "objects": objects,
        }
        return report


def parse_id_set(value):
    if value is None or not str(value).strip():
        return set()
    parsed = set()
    for token in str(value).split(","):
        object_id = normalize_object_id(token.strip())
        if not object_id:
            raise ValueError(f"invalid object ID in list: {token!r}")
        parsed.add(object_id)
    return parsed


def load_catalog(path):
    with pathlib.Path(path).open() as handle:
        payload = yaml.safe_load(handle) or {}
    raw_catalog = payload.get("object_id_catalog", {})
    if not isinstance(raw_catalog, dict) or not raw_catalog:
        raise ValueError("configuration has no object_id_catalog mapping")
    catalog = {}
    for raw_id, raw_name in raw_catalog.items():
        object_id = normalize_object_id(raw_id)
        model_name = str(raw_name).strip()
        if not object_id or not model_name:
            raise ValueError(f"invalid catalog entry: {raw_id!r}: {raw_name!r}")
        if object_id in catalog or model_name in catalog.values():
            raise ValueError(f"duplicate catalog entry: {raw_id!r}: {raw_name!r}")
        catalog[object_id] = model_name
    return catalog


class RosAssociationCollector:
    def __init__(self, accumulator, args):
        self.accumulator = accumulator
        self.subscribers = [
            rospy.Subscriber(args.livox_topic, CustomMsg, self._handle_livox, queue_size=2),
            rospy.Subscriber(
                args.fast_lio_topic, PointCloud2, self._handle_fast_lio, queue_size=2
            ),
            rospy.Subscriber(
                args.stats_topic,
                ObjectObservationStats,
                self._handle_stats,
                queue_size=2,
            ),
            rospy.Subscriber(
                args.anchors_topic, AnchorStates, self._handle_anchors, queue_size=2
            ),
        ]

    def _handle_livox(self, msg):
        self.accumulator.add_raw_reflectivities(
            getattr(point, "reflectivity", 0) for point in getattr(msg, "points", [])
        )

    def _handle_fast_lio(self, msg):
        try:
            values = (
                row[0]
                for row in point_cloud2.read_points(
                    msg, field_names=("intensity",), skip_nans=True
                )
            )
            self.accumulator.add_fast_lio_intensities(values)
        except (KeyError, ValueError) as exc:
            rospy.logerr_throttle(2.0, "Cannot read PointCloud2 intensity: %s", exc)

    def _handle_stats(self, msg):
        self.accumulator.add_observation_stats(
            getattr(msg, "total_point_count", 0),
            getattr(msg, "valid_label_point_count", 0),
            getattr(msg, "invalid_label_point_count", 0),
            {
                int(getattr(item, "object_id", 0)): int(
                    getattr(item, "point_count", 0)
                )
                for item in getattr(msg, "objects", [])
            },
        )

    def _handle_anchors(self, msg):
        self.accumulator.set_anchor_snapshot(
            getattr(msg, "reference_epoch", 0), getattr(msg, "anchors", [])
        )


def parse_args():
    default_config = pathlib.Path(__file__).resolve().parents[1] / "config" / (
        "sim_experiment_recorder.yaml"
    )
    parser = argparse.ArgumentParser(
        description="Audit Gazebo object IDs across Livox, FAST-LIO, and ALERT."
    )
    parser.add_argument("--catalog-config", type=pathlib.Path, default=default_config)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--expected-visible-ids", default="")
    parser.add_argument("--expected-evaluable-ids", default="")
    parser.add_argument("--output-json", type=pathlib.Path, default=None)
    parser.add_argument("--livox-topic", default="/livox/lidar")
    parser.add_argument("--fast-lio-topic", default="/cloud_registered")
    parser.add_argument("--stats-topic", default="/deform/object_observation_stats")
    parser.add_argument("--anchors-topic", default="/deform/anchors")
    return parser.parse_args()


def print_summary(report):
    print(
        "association_check status=%s epoch=%s anchors=%s valid_label_ratio=%.6f"
        % (
            report["status"],
            report["anchors"]["reference_epoch"],
            report["anchors"]["total_count"],
            report["labels"]["valid_ratio"],
        )
    )
    for object_id, item in report["objects"].items():
        print(
            "id=%s model=%s raw=%d fast_lio=%d alert=%d anchors=%d "
            "types=P:%d/E:%d/B:%d assoc=C:%d/M:%d/X:%d confidence=%s"
            % (
                object_id,
                item["model_name"],
                item["raw_livox_point_count"],
                item["fast_lio_point_count"],
                item["alert_point_count"],
                item["valid_associated_anchor_count"],
                item["anchor_types"].get("PLANE", 0),
                item["anchor_types"].get("EDGE", 0),
                item["anchor_types"].get("BAND", 0),
                item["association"].get("consistent", 0),
                item["association"].get("mismatch", 0),
                item["association"].get("mixed", 0),
                item["object_id_confidence"].get("mean"),
            )
        )
    for warning in report["warnings"]:
        print("WARNING " + warning)
    for error in report["errors"]:
        print("ERROR " + error)


def main():
    args = parse_args()
    if rospy is None:
        raise RuntimeError("ROS Python packages are required")
    if args.duration <= 0.0:
        raise ValueError("--duration must be positive")
    visible_ids = parse_id_set(args.expected_visible_ids)
    evaluable_ids = parse_id_set(args.expected_evaluable_ids)
    accumulator = AssociationAccumulator(load_catalog(args.catalog_config))
    rospy.init_node("check_sim_object_associations", anonymous=True)
    RosAssociationCollector(accumulator, args)

    deadline = time.monotonic() + args.duration
    while not rospy.is_shutdown() and time.monotonic() < deadline:
        time.sleep(0.05)
    report = accumulator.build_report(visible_ids, evaluable_ids)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print_summary(report)
    print(rendered)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered + "\n")
    return 2 if report["status"] == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
