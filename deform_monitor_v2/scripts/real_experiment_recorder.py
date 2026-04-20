#!/usr/bin/env python3
# Author: zgliu@cumt.edu.cn
# Affiliation: China University of Mining and Technology
# Open-source release date: 2026-04-20
"""
real_experiment_recorder.py  —  ALERT recorder for real-world robot runs.

Records algorithm outputs from a live ALERT deployment to disk.
Much simpler than sim_experiment_recorder.py: no Gazebo truth data,
no TUM trajectory, no frame alignment, no ablation manifest.

Output tree:
  <output_root>/<YYYYMMDD>/real_run_NNN/
    meta/
      run_info.json         - run timestamps, topics
      config_snapshot.json  - full ROS parameter tree snapshot
    algorithm/
      clusters.jsonl
      risk_evidence.jsonl
      risk_regions.jsonl
      persistent_risk_regions.jsonl
      persistent_track_events.jsonl
      structure_motions.jsonl
    runtime/
      stage_runtime.jsonl   (written directly by the C++ node)
"""

import datetime as dt
import json
import math
import pathlib
import re
import sys

# ── Shared serialisation helpers from sim_experiment_recorder ─────────────────
# Both scripts live in the same scripts/ directory; add it to sys.path so
# Python can find the module regardless of the ROS install layout.
_SCRIPTS_DIR = pathlib.Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from sim_experiment_recorder import (
    PERSISTENT_STATE_NAMES,
    REGION_TYPE_NAMES,
    common_record_time_sec_from_payload,
    copy_time_dict,
    serialize_anchor_states,
    serialize_motion_clusters,
    serialize_persistent_risk_regions,
    serialize_risk_evidence,
    serialize_risk_regions,
    serialize_structure_motions,
    time_to_dict,
)

try:
    import rospy
    from deform_monitor_v2.msg import (
        AnchorStates,
        MotionClusters,
        PersistentRiskRegions,
        RiskEvidenceArray,
        RiskRegions,
        StructureMotions,
    )
except ImportError:  # pragma: no cover
    rospy = None
    AnchorStates = None
    MotionClusters = None
    PersistentRiskRegions = None
    RiskEvidenceArray = None
    RiskRegions = None
    StructureMotions = None

# ── Run-directory helpers ─────────────────────────────────────────────────────
_REAL_RUN_DIR_PATTERN = re.compile(r"^real_run_(\d{3})$")
DEFAULT_OUTPUT_ROOT = pathlib.Path.home() / ".ros" / "alert" / "real_output"


def _allocate_real_run_directory(day_dir: pathlib.Path) -> pathlib.Path:
    """Return the next real_run_NNN path (directory not yet created)."""
    day_dir = pathlib.Path(day_dir)
    max_index = -1
    if day_dir.exists():
        for child in day_dir.iterdir():
            if not child.is_dir():
                continue
            m = _REAL_RUN_DIR_PATTERN.match(child.name)
            if m:
                max_index = max(max_index, int(m.group(1)))
    return day_dir / ("real_run_%03d" % (max_index + 1))


# ── Recorder ──────────────────────────────────────────────────────────────────
class RealExperimentRecorder:
    def __init__(self):
        if rospy is None:
            raise RuntimeError("ROS environment is not available.")

        # ── Parameters ─────────────────────────────────────────────────────────
        self.output_root = pathlib.Path(
            rospy.get_param("~output_root", str(DEFAULT_OUTPUT_ROOT))
        ).expanduser()
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
        self.deform_monitor_param_root = (
            str(rospy.get_param("~deform_monitor_param_root", "/deform_monitor_v2")).strip()
            or "/deform_monitor_v2"
        )
        self.deform_monitor_config_path = str(
            rospy.get_param("~deform_monitor_config_path", "")
        ).strip()

        # ── Directories ─────────────────────────────────────────────────────────
        day = dt.datetime.now().strftime("%Y%m%d")
        day_dir = self.output_root / day
        day_dir.mkdir(parents=True, exist_ok=True)
        self.run_dir = _allocate_real_run_directory(day_dir)
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.meta_dir = self.run_dir / "meta"
        self.algorithm_dir = self.run_dir / "algorithm"
        self.runtime_dir = self.run_dir / "runtime"
        for d in (self.meta_dir, self.algorithm_dir, self.runtime_dir):
            d.mkdir(parents=True, exist_ok=True)

        # ── Internal state ──────────────────────────────────────────────────────
        self._algorithm_files = {}           # key → open file handle
        self._persistent_track_cache = {}    # track_id → lifecycle dict
        self._latest_cluster_payload = None  # most recent clusters payload
        self._DISP_WINDOW_HALF = 3
        self._cluster_history = []           # ring buffer for pre-detection back-fill
        self._disp_window_pending = []       # pending displacement-window entries
        self._anchor_cluster_consecutive = {}  # anchor_id → consecutive-frame count

        # ── Startup ─────────────────────────────────────────────────────────────
        self._publish_runtime_output_dir_param()
        self._write_run_info()
        self._write_config_snapshot()

        rospy.on_shutdown(self.close)
        self._subscribers = [
            rospy.Subscriber(
                self.clusters_topic, MotionClusters,
                self._handle_clusters, queue_size=10,
            ),
            rospy.Subscriber(
                self.risk_evidence_topic, RiskEvidenceArray,
                self._handle_risk_evidence, queue_size=10,
            ),
            rospy.Subscriber(
                self.risk_regions_topic, RiskRegions,
                self._handle_risk_regions, queue_size=10,
            ),
            rospy.Subscriber(
                self.persistent_risk_regions_topic, PersistentRiskRegions,
                self._handle_persistent_risk_regions, queue_size=10,
            ),
            rospy.Subscriber(
                self.structure_motions_topic, StructureMotions,
                self._handle_structure_motions, queue_size=10,
            ),
            rospy.Subscriber(
                self.anchor_states_topic, AnchorStates,
                self._handle_anchor_states, queue_size=10,
            ),
        ]

        rospy.loginfo("[real_experiment_recorder] writing to %s", self.run_dir)

    # ── Setup helpers ──────────────────────────────────────────────────────────

    def _publish_runtime_output_dir_param(self):
        """Tell the C++ node where to write stage_runtime.jsonl."""
        rospy.set_param("/deform_monitor/runtime_output_dir", str(self.runtime_dir))

    def _write_json(self, path, payload):
        with pathlib.Path(path).open("w") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
            fh.write("\n")

    def _write_run_info(self):
        self._write_json(self.meta_dir / "run_info.json", {
            "created_at_iso": dt.datetime.now().isoformat(),
            "run_directory": str(self.run_dir),
            "topics": {
                "clusters": self.clusters_topic,
                "risk_evidence": self.risk_evidence_topic,
                "risk_regions": self.risk_regions_topic,
                "persistent_risk_regions": self.persistent_risk_regions_topic,
                "structure_motions": self.structure_motions_topic,
                "anchor_states": self.anchor_states_topic,
            },
        })

    def _write_config_snapshot(self):
        try:
            parameter_tree = rospy.get_param(self.deform_monitor_param_root, {})
        except Exception:
            parameter_tree = {}
        self._write_json(self.meta_dir / "config_snapshot.json", {
            "created_at_iso": dt.datetime.now().isoformat(),
            "run_directory": str(self.run_dir),
            "node_param_root": self.deform_monitor_param_root,
            "source_config_path": self.deform_monitor_config_path,
            "parameters": parameter_tree if isinstance(parameter_tree, dict) else {},
        })

    def _append_jsonl(self, key, filename, payload):
        handle = self._algorithm_files.get(key)
        if handle is None:
            handle = (self.algorithm_dir / filename).open("a")
            self._algorithm_files[key] = handle
        json.dump(payload, handle, sort_keys=True)
        handle.write("\n")
        handle.flush()

    # ── Displacement-window helpers ────────────────────────────────────────────

    def _find_confirmed_displacement_estimate(self, region_center, latest_cluster_payload):
        """Find the best-matching significant cluster displacement for a confirmed region."""
        if not latest_cluster_payload or not isinstance(region_center, dict):
            return None
        try:
            cx = float(region_center.get("x", 0.0))
            cy = float(region_center.get("y", 0.0))
            cz = float(region_center.get("z", 0.0))
        except (TypeError, ValueError):
            return None
        _MATCH_RADIUS = 0.8  # metres
        best_dist, best_cluster = float("inf"), None
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
                best_dist, best_cluster = dist, cluster
        if best_cluster is None:
            return None
        return {
            "disp_norm_m": float(best_cluster.get("disp_norm", 0.0)),
            "disp_mean": list(best_cluster.get("disp_mean", [])),
            "cluster_support_count": int(best_cluster.get("support_count", 0)),
            "match_dist_m": round(best_dist, 4),
        }

    def _advance_disp_window_pending(self, new_cluster_payload):
        still_pending = []
        for entry in self._disp_window_pending:
            half = entry["window_half"]
            if len(entry["post_frames"]) < half:
                entry["post_frames"].append({
                    "t_offset": len(entry["post_frames"]) + 1,
                    "clusters_payload": new_cluster_payload,
                })
            if len(entry["post_frames"]) >= half:
                self._flush_disp_window(entry)
            else:
                still_pending.append(entry)
        self._disp_window_pending = still_pending

    def _flush_disp_window(self, entry):
        region_center = entry["region_center"]
        frames_out = []
        for slot in entry["pre_frames"]:
            est = self._find_confirmed_displacement_estimate(
                region_center, slot["clusters_payload"]
            )
            frames_out.append({"t_offset": slot["t_offset"], "disp_estimate": est})
        frames_out.append({
            "t_offset": 0,
            "disp_estimate": None,
            "note": "see confirmed_displacement_estimate in first_confirmed event",
        })
        for slot in entry["post_frames"]:
            est = self._find_confirmed_displacement_estimate(
                region_center, slot["clusters_payload"]
            )
            frames_out.append({"t_offset": slot["t_offset"], "disp_estimate": est})
        self._append_jsonl("persistent_track_events", "persistent_track_events.jsonl", {
            "event_type": "displacement_window",
            "track_id": entry["track_id"],
            "confirmed_at": entry["confirmed_at"],
            "window_half": entry["window_half"],
            "frames": frames_out,
            "recorded_at": time_to_dict(rospy.Time.now()),
        })

    # ── Persistent track events ────────────────────────────────────────────────

    def _append_persistent_track_events(self, payload):
        track_cache = self._persistent_track_cache
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
                lifecycle["first_seen"] = (
                    copy_time_dict(previous.get("first_seen")) or copy_time_dict(stamp)
                )
                lifecycle["first_confirmed"] = copy_time_dict(previous.get("first_confirmed"))
                lifecycle["last_seen"] = copy_time_dict(stamp)
                if confirmed and lifecycle["first_confirmed"] is None:
                    lifecycle["first_confirmed"] = copy_time_dict(stamp)

            base = {
                "track_id": track_id,
                "header": header,
                "stamp": copy_time_dict(stamp),
                "recorded_at": recorded_at,
                "state": state,
                "state_name": PERSISTENT_STATE_NAMES.get(state, "UNKNOWN"),
                "confirmed": confirmed,
                "region_type": int(region.get("region_type", 0)),
                "region_type_name": REGION_TYPE_NAMES.get(
                    int(region.get("region_type", 0)), "UNKNOWN"
                ),
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

            # track_created
            if previous is None:
                ev = dict(base)
                ev["event_type"] = "track_created"
                self._append_jsonl(
                    "persistent_track_events", "persistent_track_events.jsonl", ev
                )

            # state_transition
            if previous is not None and int(previous.get("state", state)) != state:
                ev = dict(base)
                ev["event_type"] = "state_transition"
                ev["from_state"] = int(previous.get("state", state))
                ev["from_state_name"] = PERSISTENT_STATE_NAMES.get(
                    int(previous.get("state", state)), "UNKNOWN"
                )
                ev["to_state"] = state
                ev["to_state_name"] = base["state_name"]
                self._append_jsonl(
                    "persistent_track_events", "persistent_track_events.jsonl", ev
                )

            # first_confirmed
            if confirmed and (previous is None or previous.get("first_confirmed") is None):
                ev = dict(base)
                ev["event_type"] = "first_confirmed"
                disp_est = self._find_confirmed_displacement_estimate(
                    region.get("center"), self._latest_cluster_payload
                )
                if disp_est is not None:
                    ev["confirmed_displacement_estimate"] = disp_est
                self._append_jsonl(
                    "persistent_track_events", "persistent_track_events.jsonl", ev
                )
                # Register displacement window
                confirmed_at_sec = common_record_time_sec_from_payload(ev)
                pre_frames = [
                    {"t_offset": i - len(self._cluster_history), "clusters_payload": p}
                    for i, p in enumerate(self._cluster_history)
                ]
                self._disp_window_pending.append({
                    "track_id": region.get("track_id"),
                    "region_center": region.get("center"),
                    "confirmed_at": confirmed_at_sec,
                    "pre_frames": pre_frames,
                    "post_frames": [],
                    "window_half": self._DISP_WINDOW_HALF,
                })

            # frame_status (confirmed frames only)
            if confirmed:
                ev = dict(base)
                ev["event_type"] = "frame_status"
                self._append_jsonl(
                    "persistent_track_events", "persistent_track_events.jsonl", ev
                )

            track_cache[track_id] = {
                "state": state,
                "first_seen": lifecycle["first_seen"],
                "first_confirmed": lifecycle["first_confirmed"],
                "last_seen": lifecycle["last_seen"],
            }

    # ── Topic callbacks ────────────────────────────────────────────────────────

    def _handle_clusters(self, msg):
        # Update per-anchor consecutive-frame counters
        seen_anchor_ids = set()
        for cluster in getattr(msg, "clusters", []):
            for aid in getattr(cluster, "anchor_ids", []):
                seen_anchor_ids.add(int(aid))
        for aid in seen_anchor_ids:
            self._anchor_cluster_consecutive[aid] = (
                self._anchor_cluster_consecutive.get(aid, 0) + 1
            )
        for aid in [a for a in self._anchor_cluster_consecutive if a not in seen_anchor_ids]:
            del self._anchor_cluster_consecutive[aid]

        payload = serialize_motion_clusters(msg)

        # Enrich with min_anchor_consecutive_active_frames (mirrors sim recorder)
        for cluster_dict in payload.get("clusters", []):
            anchor_ids = cluster_dict.get("anchor_ids", [])
            min_consec = (
                min(self._anchor_cluster_consecutive.get(int(a), 1) for a in anchor_ids)
                if anchor_ids else 0
            )
            cluster_dict["min_anchor_consecutive_active_frames"] = min_consec

        payload["recorded_at"] = time_to_dict(rospy.Time.now())
        self._latest_cluster_payload = payload
        self._cluster_history.append(payload)
        if len(self._cluster_history) > self._DISP_WINDOW_HALF:
            self._cluster_history.pop(0)
        self._advance_disp_window_pending(payload)

        # Write only significant clusters (non-significant never used by analysis)
        write_payload = dict(payload)
        write_payload["clusters"] = [
            c for c in payload.get("clusters", []) if c.get("significant", False)
        ]
        self._append_jsonl("clusters", "clusters.jsonl", write_payload)

    def _handle_risk_evidence(self, msg):
        payload = serialize_risk_evidence(msg)
        payload["recorded_at"] = time_to_dict(rospy.Time.now())
        self._append_jsonl("risk_evidence", "risk_evidence.jsonl", payload)

    def _handle_risk_regions(self, msg):
        payload = serialize_risk_regions(msg)
        payload["recorded_at"] = time_to_dict(rospy.Time.now())
        self._append_jsonl("risk_regions", "risk_regions.jsonl", payload)

    def _handle_persistent_risk_regions(self, msg):
        payload = serialize_persistent_risk_regions(msg)
        payload["recorded_at"] = time_to_dict(rospy.Time.now())
        self._append_jsonl(
            "persistent_risk_regions", "persistent_risk_regions.jsonl", payload
        )
        self._append_persistent_track_events(payload)

    def _handle_structure_motions(self, msg):
        payload = serialize_structure_motions(msg)
        payload["recorded_at"] = time_to_dict(rospy.Time.now())
        self._append_jsonl("structure_motions", "structure_motions.jsonl", payload)

    def _handle_anchor_states(self, msg):
        payload = serialize_anchor_states(msg)
        payload["recorded_at"] = time_to_dict(rospy.Time.now())
        self._append_jsonl("anchor_states", "anchor_states.jsonl", payload)

    def close(self):
        for handle in self._algorithm_files.values():
            handle.close()
        self._algorithm_files = {}


def main():
    if rospy is None:
        raise RuntimeError("rospy is required to run real_experiment_recorder.py")
    rospy.init_node("real_experiment_recorder")
    RealExperimentRecorder()
    rospy.spin()


if __name__ == "__main__":
    main()
