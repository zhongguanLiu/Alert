# Author: zgliu@cumt.edu.cn
# Affiliation: China University of Mining and Technology
# Open-source release date: 2026-04-20
"""Unit tests for sim_experiment_recorder.py.

The hard-coded rates, delays, velocities, and IDs in this file are synthetic
ROS parameter fixtures used to verify manifest writing and recorder behavior.
They are not ground-truth experiment logs or reported evaluation numbers.
"""

import importlib.util
import json
import pathlib
import xml.etree.ElementTree as ET
import sys
import tempfile
import types
import unittest
from types import SimpleNamespace


SCRIPT_PATH = (
    pathlib.Path(__file__).resolve().parents[1] / "scripts" / "sim_experiment_recorder.py"
)


ROS_IMPORT_ROOTS = {
    "rospy",
    "roslib",
    "std_msgs",
    "geometry_msgs",
    "sensor_msgs",
    "nav_msgs",
    "visualization_msgs",
    "tf",
    "tf2_ros",
    "tf2_msgs",
    "deform_monitor_v2",
}


class _StubModule(types.ModuleType):
    def __getattr__(self, name):
        value = type(name, (), {})
        setattr(self, name, value)
        return value


class _FakeSubscriber(SimpleNamespace):
    pass


class _FakeRospy:
    def __init__(self, params=None, now_sec=123.0):
        self.params = dict(params or {})
        self.now_sec = now_sec
        self.subscribers = []
        self.shutdown_callbacks = []
        self.logged_info = []
        self.logged_warnings = []
        self.Time = SimpleNamespace(
            now=lambda: SimpleNamespace(to_sec=lambda: self.now_sec)
        )

    def get_param(self, name, default=None):
        return self.params.get(name, default)

    def get_param_names(self):
        return sorted(self.params.keys())

    def on_shutdown(self, callback):
        self.shutdown_callbacks.append(callback)

    def Subscriber(self, topic, message_class, callback, queue_size=1):
        subscriber = _FakeSubscriber(
            topic=topic,
            message_class=message_class,
            callback=callback,
            queue_size=queue_size,
        )
        self.subscribers.append(subscriber)
        return subscriber

    def loginfo(self, *args):
        self.logged_info.append(args)

    def logwarn_throttle(self, *args):
        self.logged_warnings.append(args)


class _FakeTfListener:
    def __init__(self, transforms=None, error=None):
        self.transforms = dict(transforms or {})
        self.error = error
        self.lookups = []

    def lookupTransform(self, target_frame, source_frame, stamp):
        self.lookups.append((target_frame, source_frame, stamp))
        if self.error is not None:
            raise self.error
        return self.transforms[(target_frame, source_frame)]


def make_persistent_region(
    track_id=7,
    state=1,
    region_type=1,
    center=None,
    bbox_min=None,
    bbox_max=None,
    mean_risk=0.4,
    peak_risk=0.8,
    confidence=0.6,
    accumulated_risk=1.2,
    support_mass=3.0,
    spatial_span=0.7,
    hit_streak=4,
    miss_streak=1,
    age_frames=5,
    confirmed=True,
):
    return SimpleNamespace(
        track_id=track_id,
        state=state,
        region_type=region_type,
        center=SimpleNamespace(**(center or {"x": 1.0, "y": 2.0, "z": 3.0})),
        bbox_min=SimpleNamespace(**(bbox_min or {"x": 0.5, "y": 1.5, "z": 2.5})),
        bbox_max=SimpleNamespace(**(bbox_max or {"x": 1.5, "y": 2.5, "z": 3.5})),
        mean_risk=mean_risk,
        peak_risk=peak_risk,
        confidence=confidence,
        accumulated_risk=accumulated_risk,
        support_mass=support_mass,
        spatial_span=spatial_span,
        hit_streak=hit_streak,
        miss_streak=miss_streak,
        age_frames=age_frames,
        confirmed=confirmed,
    )


def make_motion_cluster(
    cluster_id=3,
    anchor_ids=None,
    center=None,
    bbox_min=None,
    bbox_max=None,
    disp_mean=None,
    disp_cov=None,
    chi2_stat=6.5,
    disp_norm=0.022,
    confidence=0.8,
    support_count=9,
    significant=True,
):
    return SimpleNamespace(
        id=cluster_id,
        anchor_ids=list(anchor_ids or [1, 2, 3]),
        center=SimpleNamespace(**(center or {"x": 1.0, "y": 2.0, "z": 3.0})),
        bbox_min=SimpleNamespace(**(bbox_min or {"x": 0.5, "y": 1.5, "z": 2.5})),
        bbox_max=SimpleNamespace(**(bbox_max or {"x": 1.5, "y": 2.5, "z": 3.5})),
        disp_mean=list(disp_mean or [0.01, 0.0, 0.0]),
        disp_cov=list(
            disp_cov
            or [1.0e-4, 0.0, 0.0, 0.0, 1.0e-4, 0.0, 0.0, 0.0, 1.0e-4]
        ),
        chi2_stat=chi2_stat,
        disp_norm=disp_norm,
        confidence=confidence,
        support_count=support_count,
        significant=significant,
    )


def _install_stub_module(module_name, added_modules, parent_attrs):
    parts = module_name.split(".")
    for index in range(1, len(parts) + 1):
        partial_name = ".".join(parts[:index])
        if partial_name in sys.modules:
            continue

        module = _StubModule(partial_name)
        if index < len(parts):
            module.__path__ = []
        sys.modules[partial_name] = module
        added_modules.append(partial_name)

        if index > 1:
            parent_name = ".".join(parts[: index - 1])
            parent_module = sys.modules[parent_name]
            attr_name = parts[index - 1]
            parent_key = (parent_name, attr_name)
            if parent_key not in parent_attrs:
                parent_attrs[parent_key] = getattr(parent_module, attr_name, None)
            setattr(parent_module, attr_name, module)


def _restore_stub_modules(added_modules, parent_attrs):
    for parent_name, attr_name in reversed(list(parent_attrs)):
        original_value = parent_attrs[(parent_name, attr_name)]
        parent_module = sys.modules.get(parent_name)
        if parent_module is None:
            continue
        if original_value is None:
            parent_module.__dict__.pop(attr_name, None)
        else:
            setattr(parent_module, attr_name, original_value)

    for module_name in reversed(added_modules):
        sys.modules.pop(module_name, None)


def load_module_if_exists():
    if not SCRIPT_PATH.exists():
        return None

    spec = importlib.util.spec_from_file_location("sim_experiment_recorder", SCRIPT_PATH)
    stubbed_module_names = set()
    while True:
        module = importlib.util.module_from_spec(spec)
        added_modules = []
        parent_attrs = {}
        try:
            for module_name in sorted(stubbed_module_names):
                _install_stub_module(module_name, added_modules, parent_attrs)
            spec.loader.exec_module(module)
            return module
        except ModuleNotFoundError as exc:
            missing_name = exc.name or ""
            if missing_name.split(".")[0] not in ROS_IMPORT_ROOTS:
                raise
            stubbed_module_names.add(missing_name)
        finally:
            _restore_stub_modules(added_modules, parent_attrs)


class SimExperimentRecorderHelperTests(unittest.TestCase):
    def _assert_launch_file_keeps_explicit_fallback_args(self, launch_path):
        root = ET.parse(launch_path).getroot()
        args = {element.attrib["name"]: element.attrib.get("default") for element in root.findall("arg")}

        self.assertEqual(args["controlled_object"], "")
        self.assertEqual(args["command_frame"], "")
        self.assertEqual(args["linear_velocity_x"], "0.0")
        self.assertEqual(args["linear_velocity_y"], "0.0")
        self.assertEqual(args["linear_velocity_z"], "0.0")
        self.assertEqual(args["angular_velocity_y_deg"], "0.0")
        self.assertEqual(args["control_start_delay_sec"], "")
        self.assertEqual(args["control_duration_sec"], "")

    def _make_manifest_recorder_fixture(self, module, temp_dir, params=None):
        fake_rospy = _FakeRospy(params=params or {}, now_sec=123.0)
        original_rospy = module.rospy
        module.rospy = fake_rospy

        recorder = module.SimExperimentRecorder.__new__(module.SimExperimentRecorder)
        recorder.run_dir = temp_dir
        recorder.meta_dir = temp_dir / "meta"
        recorder.meta_dir.mkdir(parents=True, exist_ok=True)
        recorder.truth_dir = temp_dir / "truth"
        recorder.truth_objects_dir = recorder.truth_dir / "objects"
        recorder.truth_links_dir = recorder.truth_dir / "links"
        recorder.algorithm_dir = temp_dir / "algorithm"
        recorder.trajectory_dir = temp_dir / "trajectory"
        recorder.ego_model_name = "mid360_fastlio"
        recorder.truth_frame = "world"
        recorder._object_files = {}
        recorder._link_files = {}
        recorder._algorithm_files = {}
        recorder._ego_initial_pose_written = False
        recorder._frame_alignment_written = False
        recorder._latest_sensor_pose_world = None
        recorder._latest_sensor_pose_stamp = None
        recorder._sensor_relative_pose_cache = {}
        recorder.deform_monitor_param_root = "/deform_monitor_v2"
        recorder.deform_monitor_config_path = "/tmp/deform_monitor_v2_sim.yaml"
        recorder.controlled_object = "obstacle_block_left_clone_clone"
        recorder.command_frame = "world"
        recorder.linear_velocity = {"x": 0.0, "y": 0.0, "z": 0.002}
        recorder.angular_velocity_deg = {"x": 0.0, "y": 0.0, "z": 0.0}
        recorder.control_axis = {"x": 0.0, "y": 0.0, "z": 1.0}
        recorder.control_start_delay_sec = 8.0
        recorder.control_duration_sec = 20.0
        recorder.scenario_id = "collapse_microdeform_case_01"

        return recorder, fake_rospy, original_rospy

    def _read_manifest(self, recorder):
        return json.loads((recorder.meta_dir / "scenario_manifest.json").read_text())

    def test_allocate_run_directory_increments_sim_run_indices(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        with tempfile.TemporaryDirectory(prefix="sim_experiment_day_") as temp_dir:
            day_dir = pathlib.Path(temp_dir)
            (day_dir / "sim_run_000").mkdir()
            (day_dir / "sim_run_001").mkdir()

            run_dir = module.allocate_run_directory(day_dir)

            self.assertEqual(run_dir, day_dir / "sim_run_002")

    def test_build_frame_alignment_metadata_marks_initial_ego_pose_sim_alignment(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        metadata = module.build_frame_alignment_metadata(
            ego_pose_world={"position": {"x": 1.0, "y": 2.0, "z": 3.0}},
            truth_frame="world",
            algorithm_frame="map",
        )

        self.assertEqual(metadata["truth_frame"], "world")
        self.assertEqual(metadata["algorithm_frame"], "map")
        self.assertEqual(metadata["alignment_mode"], "initial_ego_pose")
        self.assertEqual(metadata["sim_only"], True)

    def test_build_frame_alignment_metadata_includes_explicit_forward_and_inverse_transforms(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        metadata = module.build_frame_alignment_metadata(
            ego_pose_world={
                "position": {"x": 1.0, "y": 2.0, "z": 3.0},
                "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
            },
            truth_frame="world",
            algorithm_frame="camera_init",
        )

        self.assertEqual(
            metadata["world_from_algorithm_transform"]["source_frame"],
            "camera_init",
        )
        self.assertEqual(
            metadata["world_from_algorithm_transform"]["target_frame"],
            "world",
        )
        self.assertEqual(
            metadata["world_from_algorithm_transform"]["pose"]["position"],
            {"x": 1.0, "y": 2.0, "z": 3.0},
        )
        self.assertEqual(
            metadata["algorithm_from_world_transform"]["source_frame"],
            "world",
        )
        self.assertEqual(
            metadata["algorithm_from_world_transform"]["target_frame"],
            "camera_init",
        )
        self.assertEqual(
            metadata["algorithm_from_world_transform"]["pose"]["position"],
            {"x": -1.0, "y": -2.0, "z": -3.0},
        )

    def test_build_frame_alignment_metadata_uses_truth_and_algorithm_reference_pose_pair(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        metadata = module.build_frame_alignment_metadata(
            ego_pose_world={"position": {"x": 99.0, "y": 88.0, "z": 77.0}},
            truth_frame="world",
            algorithm_frame="camera_init",
            truth_reference_frame="base_footprint",
            truth_reference_pose_world={
                "position": {"x": 10.0, "y": 0.0, "z": 0.0},
                "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
            },
            algorithm_reference_frame="body",
            algorithm_reference_pose_algorithm={
                "position": {"x": 1.0, "y": 0.0, "z": 0.0},
                "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
            },
        )

        self.assertEqual(metadata["truth_reference_frame"], "base_footprint")
        self.assertEqual(metadata["algorithm_reference_frame"], "body")
        self.assertEqual(
            metadata["world_from_algorithm_transform"]["pose"]["position"],
            {"x": 9.0, "y": 0.0, "z": 0.0},
        )
        self.assertEqual(
            metadata["truth_reference_pose_world"]["position"],
            {"x": 10.0, "y": 0.0, "z": 0.0},
        )
        self.assertEqual(
            metadata["algorithm_reference_pose_algorithm"]["position"],
            {"x": 1.0, "y": 0.0, "z": 0.0},
        )

    def test_format_tum_line_writes_timestamp_position_and_quaternion(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        line = module.format_tum_line(
            timestamp_sec=12.5,
            position={"x": 1.0, "y": -2.0, "z": 3.25},
            orientation={"x": 0.1, "y": 0.2, "z": 0.3, "w": 0.9},
        )

        self.assertEqual(
            line,
            "12.500000000 1.000000000 -2.000000000 3.250000000 "
            "0.100000000 0.200000000 0.300000000 0.900000000\n",
        )

    def test_write_trajectory_sample_pair_uses_shared_odometry_timestamp(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        with tempfile.TemporaryDirectory(prefix="sim_experiment_recorder_") as temp_dir:
            temp_dir = pathlib.Path(temp_dir)
            gt_path = temp_dir / "truth" / "gt_sensor_world_tum.txt"
            odom_path = temp_dir / "algorithm" / "odom_raw_tum.txt"

            wrote_sample = module.write_tum_sample_pair(
                gt_path=gt_path,
                odom_path=odom_path,
                timestamp_sec=42.5,
                sensor_pose_world={
                    "position": {"x": 1.0, "y": 2.0, "z": 3.0},
                    "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                },
                odom_pose={
                    "position": {"x": 4.0, "y": 5.0, "z": 6.0},
                    "orientation": {"x": 0.1, "y": 0.2, "z": 0.3, "w": 0.9},
                },
            )

            self.assertTrue(wrote_sample)
            self.assertTrue(gt_path.exists())
            self.assertTrue(odom_path.exists())
            self.assertEqual(
                gt_path.read_text(),
                "42.500000000 1.000000000 2.000000000 3.000000000 "
                "0.000000000 0.000000000 0.000000000 1.000000000\n",
            )
            self.assertEqual(
                odom_path.read_text(),
                "42.500000000 4.000000000 5.000000000 6.000000000 "
                "0.100000000 0.200000000 0.300000000 0.900000000\n",
            )

    def test_write_tum_sample_pair_skips_non_finite_pose(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        with tempfile.TemporaryDirectory(prefix="sim_experiment_recorder_") as temp_dir:
            temp_dir = pathlib.Path(temp_dir)
            gt_path = temp_dir / "truth" / "gt_sensor_world_tum.txt"
            odom_path = temp_dir / "algorithm" / "odom_raw_tum.txt"

            wrote_sample = module.write_tum_sample_pair(
                gt_path=gt_path,
                odom_path=odom_path,
                timestamp_sec=42.5,
                sensor_pose_world={
                    "position": {"x": 1.0, "y": float("nan"), "z": 3.0},
                    "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                },
                odom_pose={
                    "position": {"x": 4.0, "y": 5.0, "z": 6.0},
                    "orientation": {"x": 0.1, "y": 0.2, "z": 0.3, "w": 0.9},
                },
            )

            self.assertFalse(wrote_sample)
            self.assertFalse(gt_path.exists())
            self.assertFalse(odom_path.exists())

    def test_pose_dict_is_finite_rejects_nan_components(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        pose = {
            "position": {"x": 1.0, "y": float("nan"), "z": 0.0},
            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        }

        self.assertFalse(module.pose_dict_is_finite(pose))

    def test_pose_dict_is_finite_returns_false_for_malformed_pose_dict(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        pose = {
            "position": None,
            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        }

        self.assertFalse(module.pose_dict_is_finite(pose))

    def test_pose_dict_is_finite_returns_false_for_missing_components(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        self.assertFalse(
            module.pose_dict_is_finite({"position": {}, "orientation": {}})
        )

    def test_pose_dict_is_finite_returns_false_for_non_numeric_components(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        pose = {
            "position": {"x": "bad", "y": 0.0, "z": 0.0},
            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        }

        self.assertFalse(module.pose_dict_is_finite(pose))

    def test_write_tum_sample_pair_returns_false_for_incomplete_pose_dict(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        with tempfile.TemporaryDirectory(prefix="sim_experiment_recorder_") as temp_dir:
            temp_dir = pathlib.Path(temp_dir)
            gt_path = temp_dir / "truth" / "gt_sensor_world_tum.txt"
            odom_path = temp_dir / "algorithm" / "odom_raw_tum.txt"

            wrote_sample = module.write_tum_sample_pair(
                gt_path=gt_path,
                odom_path=odom_path,
                timestamp_sec=42.5,
                sensor_pose_world={
                    "position": {},
                    "orientation": {},
                },
                odom_pose={
                    "position": {"x": 4.0, "y": 5.0, "z": 6.0},
                    "orientation": {"x": 0.1, "y": 0.2, "z": 0.3, "w": 0.9},
                },
            )

            self.assertFalse(wrote_sample)
            self.assertFalse(gt_path.exists())
            self.assertFalse(odom_path.exists())

    def test_write_tum_sample_pair_returns_false_for_non_numeric_sensor_pose(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        with tempfile.TemporaryDirectory(prefix="sim_experiment_recorder_") as temp_dir:
            temp_dir = pathlib.Path(temp_dir)
            gt_path = temp_dir / "truth" / "gt_sensor_world_tum.txt"
            odom_path = temp_dir / "algorithm" / "odom_raw_tum.txt"

            wrote_sample = module.write_tum_sample_pair(
                gt_path=gt_path,
                odom_path=odom_path,
                timestamp_sec=42.5,
                sensor_pose_world={
                    "position": {"x": "bad", "y": 2.0, "z": 3.0},
                    "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                },
                odom_pose={
                    "position": {"x": 4.0, "y": 5.0, "z": 6.0},
                    "orientation": {"x": 0.1, "y": 0.2, "z": 0.3, "w": 0.9},
                },
            )

            self.assertFalse(wrote_sample)
            self.assertFalse(gt_path.exists())
            self.assertFalse(odom_path.exists())

    def test_write_tum_sample_pair_returns_false_for_non_numeric_odom_pose(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        with tempfile.TemporaryDirectory(prefix="sim_experiment_recorder_") as temp_dir:
            temp_dir = pathlib.Path(temp_dir)
            gt_path = temp_dir / "truth" / "gt_sensor_world_tum.txt"
            odom_path = temp_dir / "algorithm" / "odom_raw_tum.txt"

            wrote_sample = module.write_tum_sample_pair(
                gt_path=gt_path,
                odom_path=odom_path,
                timestamp_sec=42.5,
                sensor_pose_world={
                    "position": {"x": 1.0, "y": 2.0, "z": 3.0},
                    "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                },
                odom_pose={
                    "position": {"x": 4.0, "y": "bad", "z": 6.0},
                    "orientation": {"x": 0.1, "y": 0.2, "z": 0.3, "w": 0.9},
                },
            )

            self.assertFalse(wrote_sample)
            self.assertFalse(gt_path.exists())
            self.assertFalse(odom_path.exists())

    def test_compose_pose_dicts_applies_sensor_offset_to_ground_truth_pose(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        composed_pose = module.compose_pose_dicts(
            base_pose={
                "position": {"x": 1.0, "y": 2.0, "z": 3.0},
                "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
            },
            relative_pose={
                "position": {"x": 0.0, "y": 0.0, "z": 0.23375},
                "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
            },
        )

        self.assertEqual(
            composed_pose,
            {
                "position": {"x": 1.0, "y": 2.0, "z": 3.23375},
                "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
            },
        )

    def test_ground_truth_odometry_updates_sensor_pose_cache_from_tf(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        fake_rospy = _FakeRospy()
        original_rospy = module.rospy
        module.rospy = fake_rospy
        try:
            recorder = module.SimExperimentRecorder.__new__(module.SimExperimentRecorder)
            recorder.sensor_frame_name = "livox"
            recorder._tf_listener = _FakeTfListener(
                transforms={
                    ("base_footprint", "livox"): (
                        (0.0, 0.0, 0.23375),
                        (0.0, 0.0, 0.0, 1.0),
                    )
                }
            )
            recorder._latest_sensor_pose_world = None
            recorder._latest_sensor_pose_stamp = None

            msg = SimpleNamespace(
                header=SimpleNamespace(stamp=SimpleNamespace(secs=42, nsecs=500000000)),
                child_frame_id="base_footprint",
                pose=SimpleNamespace(
                    pose=SimpleNamespace(
                        position=SimpleNamespace(x=1.0, y=2.0, z=3.0),
                        orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
                    )
                ),
            )

            recorder._handle_ground_truth_odometry(msg)

            self.assertEqual(
                recorder._latest_sensor_pose_world,
                {
                    "position": {"x": 1.0, "y": 2.0, "z": 3.23375},
                    "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                },
            )
            self.assertEqual(recorder._latest_sensor_pose_stamp, 42.5)
            self.assertEqual(
                recorder._tf_listener.lookups,
                [("base_footprint", "livox", None)],
            )
        finally:
            module.rospy = original_rospy

    def test_odometry_driven_export_waits_for_sensor_pose_then_writes_first_pair(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        fake_rospy = _FakeRospy()
        original_rospy = module.rospy
        module.rospy = fake_rospy
        try:
            with tempfile.TemporaryDirectory(prefix="sim_experiment_recorder_") as temp_dir:
                temp_dir = pathlib.Path(temp_dir)
                recorder = module.SimExperimentRecorder.__new__(module.SimExperimentRecorder)
                recorder._latest_sensor_pose_world = None
                recorder._latest_sensor_pose_stamp = None
                recorder._gt_tum_path = temp_dir / "trajectory" / "gt_sensor_world_tum.txt"
                recorder._odom_tum_path = temp_dir / "trajectory" / "odom_raw_tum.txt"

                odom_msg = SimpleNamespace(
                    header=SimpleNamespace(stamp=SimpleNamespace(secs=42, nsecs=500000000)),
                    pose=SimpleNamespace(
                        pose=SimpleNamespace(
                            position=SimpleNamespace(x=4.0, y=5.0, z=6.0),
                            orientation=SimpleNamespace(x=0.1, y=0.2, z=0.3, w=0.9),
                        )
                    ),
                )

                recorder._handle_odometry(odom_msg)

                self.assertFalse(recorder._gt_tum_path.exists())
                self.assertFalse(recorder._odom_tum_path.exists())
                self.assertEqual(len(fake_rospy.logged_warnings), 1)
                self.assertIn(
                    "waiting for a valid sensor pose cache",
                    fake_rospy.logged_warnings[0][1],
                )

                recorder._latest_sensor_pose_world = {
                    "position": {"x": 1.0, "y": 2.0, "z": 3.0},
                    "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                }
                recorder._latest_sensor_pose_stamp = 42.45

                recorder._handle_odometry(odom_msg)

                self.assertEqual(len(fake_rospy.logged_warnings), 1)
                self.assertEqual(
                    recorder._gt_tum_path.read_text(),
                    "42.500000000 1.000000000 2.000000000 3.000000000 "
                    "0.000000000 0.000000000 0.000000000 1.000000000\n",
                )
                self.assertEqual(
                    recorder._odom_tum_path.read_text(),
                    "42.500000000 4.000000000 5.000000000 6.000000000 "
                    "0.100000000 0.200000000 0.300000000 0.900000000\n",
                )
        finally:
            module.rospy = original_rospy

    def test_odometry_callback_writes_frame_alignment_from_initial_reference_pose_pair(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        fake_rospy = _FakeRospy()
        original_rospy = module.rospy
        module.rospy = fake_rospy
        try:
            recorder = module.SimExperimentRecorder.__new__(module.SimExperimentRecorder)
            recorder.truth_frame = "world"
            recorder.algorithm_frame = "camera_init"
            recorder.meta_dir = pathlib.Path("/tmp/unused_meta_dir")
            recorder._frame_alignment_written = False
            recorder._latest_truth_reference_pose_world = {
                "position": {"x": 10.0, "y": 0.0, "z": 0.0},
                "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
            }
            recorder._latest_truth_reference_pose_stamp = 42.45
            recorder._latest_truth_reference_frame = "base_footprint"
            recorder._latest_sensor_pose_world = None
            recorder._latest_sensor_pose_stamp = None

            captured = {}

            def fake_write_json(path, payload):
                captured["path"] = path
                captured["payload"] = payload

            recorder._write_json = fake_write_json

            odom_msg = SimpleNamespace(
                child_frame_id="body",
                header=SimpleNamespace(stamp=SimpleNamespace(secs=42, nsecs=500000000)),
                pose=SimpleNamespace(
                    pose=SimpleNamespace(
                        position=SimpleNamespace(x=1.0, y=0.0, z=0.0),
                        orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
                    )
                ),
            )

            recorder._handle_odometry(odom_msg)

            self.assertTrue(recorder._frame_alignment_written)
            self.assertEqual(captured["path"], recorder.meta_dir / "frame_alignment.json")
            self.assertEqual(
                captured["payload"]["world_from_algorithm_transform"]["pose"]["position"],
                {"x": 9.0, "y": 0.0, "z": 0.0},
            )
            self.assertEqual(captured["payload"]["truth_reference_frame"], "base_footprint")
            self.assertEqual(captured["payload"]["algorithm_reference_frame"], "body")
        finally:
            module.rospy = original_rospy

    def test_odometry_driven_export_skips_stale_sensor_pose_without_writing_files(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        fake_rospy = _FakeRospy()
        original_rospy = module.rospy
        module.rospy = fake_rospy
        try:
            with tempfile.TemporaryDirectory(prefix="sim_experiment_recorder_") as temp_dir:
                temp_dir = pathlib.Path(temp_dir)
                recorder = module.SimExperimentRecorder.__new__(module.SimExperimentRecorder)
                recorder._latest_sensor_pose_world = {
                    "position": {"x": 1.0, "y": 2.0, "z": 3.0},
                    "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                }
                recorder._latest_sensor_pose_stamp = 41.0
                recorder._gt_tum_path = temp_dir / "trajectory" / "gt_sensor_world_tum.txt"
                recorder._odom_tum_path = temp_dir / "trajectory" / "odom_raw_tum.txt"

                odom_msg = SimpleNamespace(
                    header=SimpleNamespace(stamp=SimpleNamespace(secs=42, nsecs=500000000)),
                    pose=SimpleNamespace(
                        pose=SimpleNamespace(
                            position=SimpleNamespace(x=4.0, y=5.0, z=6.0),
                            orientation=SimpleNamespace(x=0.1, y=0.2, z=0.3, w=0.9),
                        )
                    ),
                )

                recorder._handle_odometry(odom_msg)

                self.assertFalse(recorder._gt_tum_path.exists())
                self.assertFalse(recorder._odom_tum_path.exists())
                self.assertEqual(len(fake_rospy.logged_warnings), 1)
                self.assertIn(
                    "cached sensor pose timestamp is stale",
                    fake_rospy.logged_warnings[0][1],
                )
        finally:
            module.rospy = original_rospy

    def test_odometry_driven_export_skips_zero_stamp_without_writing_files(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        fake_rospy = _FakeRospy()
        original_rospy = module.rospy
        module.rospy = fake_rospy
        try:
            with tempfile.TemporaryDirectory(prefix="sim_experiment_recorder_") as temp_dir:
                temp_dir = pathlib.Path(temp_dir)
                recorder = module.SimExperimentRecorder.__new__(module.SimExperimentRecorder)
                recorder._latest_sensor_pose_world = {
                    "position": {"x": 1.0, "y": 2.0, "z": 3.0},
                    "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                }
                recorder._latest_sensor_pose_stamp = 7.0
                recorder._gt_tum_path = temp_dir / "trajectory" / "gt_sensor_world_tum.txt"
                recorder._odom_tum_path = temp_dir / "trajectory" / "odom_raw_tum.txt"

                odom_msg = SimpleNamespace(
                    header=SimpleNamespace(stamp=SimpleNamespace(secs=0, nsecs=0)),
                    pose=SimpleNamespace(
                        pose=SimpleNamespace(
                            position=SimpleNamespace(x=4.0, y=5.0, z=6.0),
                            orientation=SimpleNamespace(x=0.1, y=0.2, z=0.3, w=0.9),
                        )
                    ),
                )

                recorder._handle_odometry(odom_msg)

                self.assertFalse(recorder._gt_tum_path.exists())
                self.assertFalse(recorder._odom_tum_path.exists())
                self.assertEqual(len(fake_rospy.logged_warnings), 1)
                self.assertIn(
                    "message stamp was invalid",
                    fake_rospy.logged_warnings[0][1],
                )
        finally:
            module.rospy = original_rospy

    def test_module_imports_odometry_message_type_for_runtime_wiring(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        self.assertTrue(hasattr(module, "Odometry"))

    def test_serialize_persistent_risk_regions_includes_track_level_fields(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        msg = SimpleNamespace(
            header=SimpleNamespace(seq=9, stamp=SimpleNamespace(secs=12, nsecs=500000000), frame_id="camera_init"),
            regions=[make_persistent_region()],
        )

        payload = module.serialize_persistent_risk_regions(msg)

        self.assertEqual(payload["header"]["seq"], 9)
        self.assertEqual(payload["header"]["frame_id"], "camera_init")
        self.assertEqual(payload["regions"][0]["track_id"], 7)
        self.assertEqual(payload["regions"][0]["state"], 1)
        self.assertEqual(payload["regions"][0]["region_type"], 1)
        self.assertEqual(payload["regions"][0]["center"], {"x": 1.0, "y": 2.0, "z": 3.0})
        self.assertTrue(payload["regions"][0]["confirmed"])

    def test_handle_persistent_risk_regions_writes_empty_jsonl_after_alignment_ready(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        fake_rospy = _FakeRospy(now_sec=222.5)
        original_rospy = module.rospy
        module.rospy = fake_rospy
        try:
            with tempfile.TemporaryDirectory(prefix="sim_experiment_recorder_") as temp_dir:
                temp_dir = pathlib.Path(temp_dir)
                recorder = module.SimExperimentRecorder.__new__(module.SimExperimentRecorder)
                recorder.algorithm_dir = temp_dir / "algorithm"
                recorder.algorithm_dir.mkdir(parents=True)
                recorder._object_files = {}
                recorder._link_files = {}
                recorder._algorithm_files = {}
                recorder._frame_alignment_written = True

                msg = SimpleNamespace(
                    header=SimpleNamespace(seq=3, stamp=SimpleNamespace(secs=7, nsecs=0), frame_id="camera_init"),
                    regions=[],
                )

                recorder._handle_persistent_risk_regions(msg)

                jsonl_path = recorder.algorithm_dir / "persistent_risk_regions.jsonl"
                self.assertTrue(jsonl_path.exists())
                payload = json.loads(jsonl_path.read_text().strip())
                self.assertEqual(payload["header"]["seq"], 3)
                self.assertEqual(payload["regions"], [])
                self.assertEqual(payload["recorded_at"]["sec"], 222.5)
                recorder.close()
        finally:
            module.rospy = original_rospy

    def test_handle_persistent_risk_regions_writes_track_lifecycle_events(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        fake_rospy = _FakeRospy(now_sec=300.0)
        original_rospy = module.rospy
        module.rospy = fake_rospy
        try:
            with tempfile.TemporaryDirectory(prefix="sim_experiment_recorder_") as temp_dir:
                temp_dir = pathlib.Path(temp_dir)
                recorder = module.SimExperimentRecorder.__new__(module.SimExperimentRecorder)
                recorder.algorithm_dir = temp_dir / "algorithm"
                recorder.algorithm_dir.mkdir(parents=True)
                recorder._object_files = {}
                recorder._link_files = {}
                recorder._algorithm_files = {}
                recorder._frame_alignment_written = True
                recorder._persistent_track_cache = {}

                first_msg = SimpleNamespace(
                    header=SimpleNamespace(
                        seq=1,
                        stamp=SimpleNamespace(secs=10, nsecs=0),
                        frame_id="camera_init",
                    ),
                    regions=[make_persistent_region(track_id=7, state=0, confirmed=False)],
                )
                second_msg = SimpleNamespace(
                    header=SimpleNamespace(
                        seq=2,
                        stamp=SimpleNamespace(secs=11, nsecs=0),
                        frame_id="camera_init",
                    ),
                    regions=[make_persistent_region(track_id=7, state=1, confirmed=True)],
                )

                recorder._handle_persistent_risk_regions(first_msg)
                recorder._handle_persistent_risk_regions(second_msg)

                events_path = recorder.algorithm_dir / "persistent_track_events.jsonl"
                self.assertTrue(events_path.exists())
                events = [
                    json.loads(line)
                    for line in events_path.read_text().splitlines()
                    if line.strip()
                ]
                event_types = [event["event_type"] for event in events]
                self.assertIn("track_created", event_types)
                self.assertIn("frame_status", event_types)
                self.assertIn("state_transition", event_types)
                self.assertIn("first_confirmed", event_types)

                frame_events = [
                    event for event in events if event["event_type"] == "frame_status"
                ]
                self.assertEqual(len(frame_events), 2)
                self.assertEqual(frame_events[0]["lifecycle"]["first_seen"]["sec"], 10.0)
                self.assertIsNone(frame_events[0]["lifecycle"]["first_confirmed"])
                self.assertEqual(frame_events[1]["lifecycle"]["first_confirmed"]["sec"], 11.0)
                self.assertEqual(frame_events[1]["confirmed"], True)
                self.assertEqual(frame_events[1]["state_name"], "CONFIRMED")
                self.assertEqual(frame_events[1]["support_mass"], 3.0)
                recorder.close()
        finally:
            module.rospy = original_rospy

    def test_serialize_motion_clusters_preserves_cluster_level_displacement(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        msg = SimpleNamespace(
            header=SimpleNamespace(
                seq=5,
                stamp=SimpleNamespace(secs=13, nsecs=250000000),
                frame_id="camera_init",
            ),
            clusters=[make_motion_cluster()],
        )

        payload = module.serialize_motion_clusters(msg)

        self.assertEqual(payload["header"]["seq"], 5)
        self.assertEqual(payload["clusters"][0]["id"], 3)
        self.assertEqual(payload["clusters"][0]["anchor_ids"], [1, 2, 3])
        self.assertEqual(payload["clusters"][0]["disp_mean"], [0.01, 0.0, 0.0])
        self.assertEqual(payload["clusters"][0]["support_count"], 9)

    def test_handle_clusters_writes_cluster_jsonl_after_alignment_ready(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        fake_rospy = _FakeRospy(now_sec=250.0)
        original_rospy = module.rospy
        module.rospy = fake_rospy
        try:
            with tempfile.TemporaryDirectory(prefix="sim_experiment_recorder_") as temp_dir:
                temp_dir = pathlib.Path(temp_dir)
                recorder = module.SimExperimentRecorder.__new__(module.SimExperimentRecorder)
                recorder.algorithm_dir = temp_dir / "algorithm"
                recorder.algorithm_dir.mkdir(parents=True)
                recorder._object_files = {}
                recorder._link_files = {}
                recorder._algorithm_files = {}
                recorder._frame_alignment_written = True

                msg = SimpleNamespace(
                    header=SimpleNamespace(
                        seq=8,
                        stamp=SimpleNamespace(secs=20, nsecs=0),
                        frame_id="camera_init",
                    ),
                    clusters=[make_motion_cluster(cluster_id=11)],
                )

                recorder._handle_clusters(msg)

                clusters_path = recorder.algorithm_dir / "clusters.jsonl"
                self.assertTrue(clusters_path.exists())
                payload = json.loads(clusters_path.read_text().strip())
                self.assertEqual(payload["clusters"][0]["id"], 11)
                self.assertEqual(payload["recorded_at"]["sec"], 250.0)
                recorder.close()
        finally:
            module.rospy = original_rospy

    def test_recorder_initializes_trajectory_state_and_odometry_subscription(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        with tempfile.TemporaryDirectory(prefix="sim_experiment_recorder_") as temp_dir:
            temp_dir = pathlib.Path(temp_dir)
            fake_rospy = _FakeRospy(
                params={
                    "~output_root": str(temp_dir),
                    "~truth_frame": "world",
                    "~algorithm_frame": "camera_init",
                    "~ego_model_name": "mid360_fastlio",
                    "~model_states_topic": "/gazebo/model_states",
                    "~link_states_topic": "/gazebo/link_states",
                    "~ground_truth_odometry_topic": "/ground_truth/odom",
                    "~risk_evidence_topic": "/deform/risk_evidence",
                    "~risk_regions_topic": "/deform/risk_regions",
                    "~persistent_risk_regions_topic": "/deform/persistent_risk_regions",
                    "~structure_motions_topic": "/deform/structure_motions",
                    "~odometry_topic": "/Odometry",
                    "~sensor_scoped_link_name": "mid360_fastlio::mid360_link",
                    "~sensor_frame_name": "livox",
                    "~gt_tum_filename": "gt_sensor_world_tum.txt",
                    "~odom_tum_filename": "odom_raw_tum.txt",
                    "~scenario_id": "collapse_microdeform_case_01",
                    "~controlled_object": "obstacle_block_left_clone_clone",
                    "~command_frame": "world",
                    "~linear_velocity_x": "0.0",
                    "~linear_velocity_y": "0.0",
                    "~linear_velocity_z": "0.002",
                    "~control_axis_x": "0.0",
                    "~control_axis_y": "0.0",
                    "~control_axis_z": "1.0",
                    "~control_start_delay_sec": "8.0",
                    "~control_duration_sec": "20.0",
                }
            )
            original_rospy = module.rospy
            original_tf = getattr(module, "tf", None)
            original_model_states = module.ModelStates
            original_link_states = module.LinkStates
            original_persistent_risk_regions = module.PersistentRiskRegions
            original_motion_clusters = getattr(module, "MotionClusters", None)
            original_risk_evidence_array = module.RiskEvidenceArray
            original_risk_regions = module.RiskRegions
            original_structure_motions = module.StructureMotions
            module.rospy = fake_rospy
            module.tf = SimpleNamespace(TransformListener=lambda: object())
            module.ModelStates = object
            module.LinkStates = object
            module.PersistentRiskRegions = object
            module.MotionClusters = object
            module.RiskEvidenceArray = object
            module.RiskRegions = object
            module.StructureMotions = object
            try:
                recorder = module.SimExperimentRecorder()

                ablation_manifest_path = recorder.meta_dir / "ablation_manifest.json"
                config_snapshot_path = recorder.meta_dir / "config_snapshot.json"
                scenario_manifest_path = recorder.meta_dir / "scenario_manifest.json"

                self.assertEqual(recorder.ground_truth_odometry_topic, "/ground_truth/odom")
                self.assertEqual(recorder.odometry_topic, "/Odometry")
                self.assertEqual(
                    recorder.sensor_scoped_link_name, "mid360_fastlio::mid360_link"
                )
                self.assertEqual(
                    recorder.persistent_risk_regions_topic,
                    "/deform/persistent_risk_regions",
                )
                self.assertEqual(recorder.sensor_frame_name, "livox")
                self.assertEqual(recorder.gt_tum_filename, "gt_sensor_world_tum.txt")
                self.assertEqual(recorder.odom_tum_filename, "odom_raw_tum.txt")
                self.assertEqual(recorder.trajectory_dir, recorder.run_dir / "trajectory")
                self.assertEqual(recorder._gt_tum_path, recorder.trajectory_dir / "gt_sensor_world_tum.txt")
                self.assertEqual(recorder._odom_tum_path, recorder.trajectory_dir / "odom_raw_tum.txt")
                self.assertIsNone(recorder._latest_sensor_pose_world)
                self.assertIsNone(recorder._latest_sensor_pose_stamp)
                self.assertEqual(recorder._sensor_relative_pose_cache, {})
                self.assertIsNotNone(recorder._tf_listener)
                self.assertIn("/Odometry", [sub.topic for sub in fake_rospy.subscribers])
                self.assertIn("/ground_truth/odom", [sub.topic for sub in fake_rospy.subscribers])
                self.assertIn("/deform/persistent_risk_regions", [sub.topic for sub in fake_rospy.subscribers])
                self.assertTrue(
                    any(
                        sub.topic == "/Odometry"
                        and getattr(sub.callback, "__name__", "") == "_handle_odometry"
                        for sub in fake_rospy.subscribers
                    )
                )
                self.assertTrue(
                    any(
                        sub.topic == "/ground_truth/odom"
                        and getattr(sub.callback, "__name__", "") == "_handle_ground_truth_odometry"
                        for sub in fake_rospy.subscribers
                    )
                )
                self.assertTrue(
                    any(
                        sub.topic == "/deform/persistent_risk_regions"
                        and getattr(sub.callback, "__name__", "") == "_handle_persistent_risk_regions"
                        for sub in fake_rospy.subscribers
                    )
                )
                self.assertTrue(ablation_manifest_path.exists())
                self.assertTrue(config_snapshot_path.exists())
                self.assertTrue(scenario_manifest_path.exists())
            finally:
                module.rospy = original_rospy
                module.tf = original_tf
                module.ModelStates = original_model_states
                module.LinkStates = original_link_states
                module.PersistentRiskRegions = original_persistent_risk_regions
                module.MotionClusters = original_motion_clusters
                module.RiskEvidenceArray = original_risk_evidence_array
                module.RiskRegions = original_risk_regions
                module.StructureMotions = original_structure_motions

    def test_recorder_writes_ablation_manifest_and_config_snapshot(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        deform_monitor_params = {
            "deform_monitor": {
                "covariance": {"alpha_xi": 2.0},
                "background_bias": {"enable": True},
                "imm": {
                    "enable_model_competition": True,
                    "enable_type_constraint": True,
                },
                "significance": {"enable_cusum": True},
                "directional_motion": {"enable": True},
                "ablation": {
                    "variant": "single_model_ekf_no_drift",
                    "disable_covariance_inflation": True,
                    "disable_type_constraint": False,
                    "single_model_ekf": True,
                    "disable_cusum": False,
                    "disable_directional_accumulation": False,
                    "disable_drift_compensation": True,
                },
            }
        }

        with tempfile.TemporaryDirectory(prefix="sim_experiment_recorder_") as temp_dir:
            temp_dir = pathlib.Path(temp_dir)
            fake_rospy = _FakeRospy(
                params={
                    "~output_root": str(temp_dir),
                    "~truth_frame": "world",
                    "~algorithm_frame": "camera_init",
                    "~ego_model_name": "mid360_fastlio",
                    "~model_states_topic": "/gazebo/model_states",
                    "~link_states_topic": "/gazebo/link_states",
                    "~ground_truth_odometry_topic": "/ground_truth/odom",
                    "~risk_evidence_topic": "/deform/risk_evidence",
                    "~risk_regions_topic": "/deform/risk_regions",
                    "~persistent_risk_regions_topic": "/deform/persistent_risk_regions",
                    "~structure_motions_topic": "/deform/structure_motions",
                    "~odometry_topic": "/Odometry",
                    "~sensor_scoped_link_name": "mid360_fastlio::mid360_link",
                    "~sensor_frame_name": "livox",
                    "~gt_tum_filename": "gt_sensor_world_tum.txt",
                    "~odom_tum_filename": "odom_raw_tum.txt",
                    "~deform_monitor_param_root": "/deform_monitor_v2",
                    "~deform_monitor_config_path": "/tmp/deform_monitor_v2_sim.yaml",
                    "/deform_monitor_v2": deform_monitor_params,
                }
            )
            original_rospy = module.rospy
            original_tf = getattr(module, "tf", None)
            original_model_states = module.ModelStates
            original_link_states = module.LinkStates
            original_persistent_risk_regions = module.PersistentRiskRegions
            original_risk_evidence_array = module.RiskEvidenceArray
            original_risk_regions = module.RiskRegions
            original_structure_motions = module.StructureMotions
            module.rospy = fake_rospy
            module.tf = SimpleNamespace(TransformListener=lambda: object())
            module.ModelStates = object
            module.LinkStates = object
            module.PersistentRiskRegions = object
            module.RiskEvidenceArray = object
            module.RiskRegions = object
            module.StructureMotions = object
            try:
                recorder = module.SimExperimentRecorder()

                ablation_manifest = json.loads(
                    (recorder.meta_dir / "ablation_manifest.json").read_text()
                )
                config_snapshot = json.loads(
                    (recorder.meta_dir / "config_snapshot.json").read_text()
                )

                self.assertEqual(
                    ablation_manifest["variant"], "single_model_ekf_no_drift"
                )
                self.assertTrue(ablation_manifest["switches"]["single_model_ekf"])
                self.assertTrue(
                    ablation_manifest["switches"]["disable_covariance_inflation"]
                )
                self.assertTrue(
                    ablation_manifest["switches"]["disable_drift_compensation"]
                )
                self.assertEqual(
                    ablation_manifest["effective_runtime"]["covariance_alpha_xi"], 1.0
                )
                self.assertFalse(
                    ablation_manifest["effective_runtime"]["background_bias_enable"]
                )
                self.assertFalse(
                    ablation_manifest["effective_runtime"]["imm_enable_model_competition"]
                )
                self.assertEqual(
                    config_snapshot["source_config_path"],
                    "/tmp/deform_monitor_v2_sim.yaml",
                )
                self.assertEqual(
                    config_snapshot["node_param_root"], "/deform_monitor_v2"
                )
                self.assertEqual(
                    config_snapshot["parameters"]["deform_monitor"]["ablation"]["variant"],
                    "single_model_ekf_no_drift",
                )
            finally:
                module.rospy = original_rospy
                module.tf = original_tf
                module.ModelStates = original_model_states
                module.LinkStates = original_link_states
                module.PersistentRiskRegions = original_persistent_risk_regions
                module.RiskEvidenceArray = original_risk_evidence_array
                module.RiskRegions = original_risk_regions
                module.StructureMotions = original_structure_motions

    def test_build_scenario_manifest_falls_back_to_explicit_control_metadata(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        payload = module.build_scenario_manifest_payload(
            run_dir=pathlib.Path("/tmp/sim_run_000"),
            scenario_id="case_01",
            explicit_control={
                "controlled_object": "debris_block_02",
                "command_frame": "world",
                "velocity": {
                    "linear_mps": {"x": 0.0, "y": 0.0, "z": 0.002},
                    "angular_deg_per_sec": {"x": 0.0, "y": 0.0, "z": 0.0},
                },
                "axis": {"x": 0.0, "y": 0.0, "z": 1.0},
                "start_delay_sec": 8.0,
                "duration_sec": 20.0,
            },
            discovered_controls=[],
        )

        self.assertEqual(payload["scenario_id"], "case_01")
        self.assertEqual(len(payload["controls"]), 1)
        self.assertEqual(payload["controls"][0]["controlled_object"], "debris_block_02")
        self.assertEqual(payload["controls"][0]["command_frame"], "world")
        self.assertEqual(payload["controls"][0]["axis"], {"x": 0.0, "y": 0.0, "z": 1.0})
        self.assertEqual(payload["controls"][0]["start_delay_sec"], 8.0)
        self.assertEqual(payload["controls"][0]["duration_sec"], 20.0)
        self.assertEqual(payload["source"], "explicit")

    def test_build_scenario_manifest_prefers_discovered_control_metadata(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        payload = module.build_scenario_manifest_payload(
            run_dir=pathlib.Path("/tmp/sim_run_000"),
            scenario_id="explicit_case",
            explicit_control={
                "controlled_object": "explicit_object",
                "command_frame": "world",
                "velocity": {
                    "linear_mps": {"x": 0.0, "y": 0.0, "z": 0.002},
                    "angular_deg_per_sec": {"x": 0.0, "y": 0.0, "z": 0.0},
                },
                "axis": {"x": 0.0, "y": 0.0, "z": 1.0},
                "start_delay_sec": 8.0,
                "duration_sec": 20.0,
                "scenario_id": "explicit_case",
            },
            discovered_controls=[
                {
                    "controlled_object": "discovered_object",
                    "command_frame": "body",
                    "velocity": {
                        "linear_mps": {"x": 0.0, "y": 0.001, "z": 0.0},
                        "angular_deg_per_sec": {"x": 0.0, "y": 0.0, "z": 0.0},
                    },
                    "axis": {"x": 0.0, "y": 1.0, "z": 0.0},
                    "start_delay_sec": 5.0,
                    "duration_sec": 12.0,
                    "scenario_id": "",
                }
            ],
        )

        self.assertEqual(payload["source"], "discovered")
        self.assertEqual(payload["controls"][0]["controlled_object"], "discovered_object")
        self.assertEqual(payload["controls"][0]["duration_sec"], 12.0)
        self.assertEqual(payload["scenario_id"], "explicit_case")

    def test_write_run_metadata_initially_writes_fallback_control_metadata(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        with tempfile.TemporaryDirectory(prefix="sim_experiment_recorder_") as temp_dir:
            temp_dir = pathlib.Path(temp_dir)
            recorder, fake_rospy, original_rospy = self._make_manifest_recorder_fixture(
                module,
                temp_dir,
                params={
                    "/deform_monitor_v2": {},
                },
            )
            try:
                recorder._write_run_metadata()

                manifest = self._read_manifest(recorder)

                self.assertEqual(manifest["source"], "explicit")
                self.assertEqual(manifest["scenario_id"], "collapse_microdeform_case_01")
                self.assertEqual(len(manifest["controls"]), 1)
                self.assertEqual(
                    manifest["controls"][0]["controlled_object"],
                    "obstacle_block_left_clone_clone",
                )
                self.assertEqual(manifest["controls"][0]["command_frame"], "world")
            finally:
                module.rospy = original_rospy

    def test_refreshes_discovered_controls_after_motion_controller_params_appear(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        with tempfile.TemporaryDirectory(prefix="sim_experiment_recorder_") as temp_dir:
            temp_dir = pathlib.Path(temp_dir)
            recorder, fake_rospy, original_rospy = self._make_manifest_recorder_fixture(
                module,
                temp_dir,
                params={
                    "/deform_monitor_v2": {},
                },
            )
            try:
                recorder._write_run_metadata()
                initial_manifest = self._read_manifest(recorder)

                fake_rospy.params.update(
                    {
                        "/model_01_motion/model_name": "model_01",
                        "/model_01_motion/command_frame": "world",
                        "/model_01_motion/control_rate": 20.0,
                        "/model_01_motion/command_timeout": 1.5,
                        "/model_01_motion/linear_x": 0.0,
                        "/model_01_motion/linear_y": 0.0,
                        "/model_01_motion/linear_z": 0.015,
                        "/model_01_motion/angular_x_deg": 0.0,
                        "/model_01_motion/angular_y_deg": 0.0,
                        "/model_01_motion/angular_z_deg": 0.0,
                        "/model_01_motion/start_delay": 8.0,
                        "/model_01_motion/duration": 20.0,
                        "/model_01_motion/scenario_id": "collapse_microdeform_case_01",
                        "/model_02_motion/model_name": "model_02",
                        "/model_02_motion/command_frame": "world",
                        "/model_02_motion/control_rate": 10.0,
                        "/model_02_motion/command_timeout": 2.0,
                        "/model_02_motion/linear_x": 0.005,
                        "/model_02_motion/linear_y": 0.0,
                        "/model_02_motion/linear_z": 0.0,
                        "/model_02_motion/angular_x_deg": 0.0,
                        "/model_02_motion/angular_y_deg": 0.0,
                        "/model_02_motion/angular_z_deg": 0.0,
                        "/model_02_motion/start_delay": 12.5,
                        "/model_02_motion/duration": 35.0,
                        "/model_02_motion/scenario_id": "collapse_microdeform_case_01",
                    }
                )

                recorder._handle_model_states(SimpleNamespace(name=[], pose=[]))
                refreshed_manifest = self._read_manifest(recorder)

                self.assertEqual(initial_manifest["source"], "explicit")
                self.assertEqual(refreshed_manifest["source"], "discovered")
                self.assertEqual(
                    [control["controlled_object"] for control in refreshed_manifest["controls"]],
                    ["model_01", "model_02"],
                )
                self.assertEqual(
                    [control["controller_namespace"] for control in refreshed_manifest["controls"]],
                    ["/model_01_motion", "/model_02_motion"],
                )
            finally:
                module.rospy = original_rospy

    def test_refresh_keeps_discovered_manifest_after_controller_params_disappear(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        with tempfile.TemporaryDirectory(prefix="sim_experiment_recorder_") as temp_dir:
            temp_dir = pathlib.Path(temp_dir)
            recorder, fake_rospy, original_rospy = self._make_manifest_recorder_fixture(
                module,
                temp_dir,
                params={
                    "/deform_monitor_v2": {},
                },
            )
            try:
                recorder._write_run_metadata()

                fake_rospy.params.update(
                    {
                        "/model_01_motion/model_name": "model_01",
                        "/model_01_motion/command_frame": "world",
                        "/model_01_motion/control_rate": 20.0,
                        "/model_01_motion/command_timeout": 1.5,
                        "/model_01_motion/linear_x": 0.0,
                        "/model_01_motion/linear_y": 0.0,
                        "/model_01_motion/linear_z": 0.015,
                        "/model_01_motion/angular_x_deg": 0.0,
                        "/model_01_motion/angular_y_deg": 0.0,
                        "/model_01_motion/angular_z_deg": 0.0,
                        "/model_01_motion/start_delay": 8.0,
                        "/model_01_motion/duration": 20.0,
                        "/model_01_motion/scenario_id": "collapse_microdeform_case_01",
                        "/model_02_motion/model_name": "model_02",
                        "/model_02_motion/command_frame": "world",
                        "/model_02_motion/control_rate": 10.0,
                        "/model_02_motion/command_timeout": 2.0,
                        "/model_02_motion/linear_x": 0.005,
                        "/model_02_motion/linear_y": 0.0,
                        "/model_02_motion/linear_z": 0.0,
                        "/model_02_motion/angular_x_deg": 0.0,
                        "/model_02_motion/angular_y_deg": 0.0,
                        "/model_02_motion/angular_z_deg": 0.0,
                        "/model_02_motion/start_delay": 12.5,
                        "/model_02_motion/duration": 35.0,
                        "/model_02_motion/scenario_id": "collapse_microdeform_case_01",
                    }
                )

                recorder._handle_model_states(SimpleNamespace(name=[], pose=[]))
                promoted_manifest = self._read_manifest(recorder)

                fake_rospy.params.pop("/model_02_motion/model_name")
                fake_rospy.params.pop("/model_02_motion/command_frame")
                fake_rospy.params.pop("/model_02_motion/control_rate")
                fake_rospy.params.pop("/model_02_motion/command_timeout")
                fake_rospy.params.pop("/model_02_motion/linear_x")
                fake_rospy.params.pop("/model_02_motion/linear_y")
                fake_rospy.params.pop("/model_02_motion/linear_z")
                fake_rospy.params.pop("/model_02_motion/angular_x_deg")
                fake_rospy.params.pop("/model_02_motion/angular_y_deg")
                fake_rospy.params.pop("/model_02_motion/angular_z_deg")
                fake_rospy.params.pop("/model_02_motion/start_delay")
                fake_rospy.params.pop("/model_02_motion/duration")
                fake_rospy.params.pop("/model_02_motion/scenario_id")

                recorder._handle_model_states(SimpleNamespace(name=[], pose=[]))
                refreshed_manifest = self._read_manifest(recorder)

                self.assertEqual(promoted_manifest["source"], "discovered")
                self.assertEqual(refreshed_manifest["source"], "discovered")
                self.assertEqual(
                    [control["controlled_object"] for control in refreshed_manifest["controls"]],
                    ["model_01", "model_02"],
                )
            finally:
                module.rospy = original_rospy

    def test_refresh_keeps_explicit_manifest_for_stale_motion_params(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        with tempfile.TemporaryDirectory(prefix="sim_experiment_recorder_") as temp_dir:
            temp_dir = pathlib.Path(temp_dir)
            recorder, fake_rospy, original_rospy = self._make_manifest_recorder_fixture(
                module,
                temp_dir,
                params={
                    "/deform_monitor_v2": {},
                    "/model_01_motion/model_name": "model_01",
                    "/model_01_motion/command_frame": "world",
                    "/model_01_motion/control_rate": 20.0,
                    "/model_01_motion/command_timeout": 1.5,
                    "/model_01_motion/linear_x": 0.0,
                    "/model_01_motion/linear_y": 0.0,
                    "/model_01_motion/linear_z": 0.015,
                    "/model_01_motion/angular_x_deg": 0.0,
                    "/model_01_motion/angular_y_deg": 0.0,
                    "/model_01_motion/angular_z_deg": 0.0,
                    "/model_01_motion/start_delay": 8.0,
                    "/model_01_motion/duration": 20.0,
                    "/model_01_motion/scenario_id": "stale_case",
                    "/model_02_motion/model_name": "model_02",
                    "/model_02_motion/command_frame": "world",
                    "/model_02_motion/control_rate": 10.0,
                    "/model_02_motion/command_timeout": 2.0,
                    "/model_02_motion/linear_x": 0.005,
                    "/model_02_motion/linear_y": 0.0,
                    "/model_02_motion/linear_z": 0.0,
                    "/model_02_motion/angular_x_deg": 0.0,
                    "/model_02_motion/angular_y_deg": 0.0,
                    "/model_02_motion/angular_z_deg": 0.0,
                    "/model_02_motion/start_delay": 12.5,
                    "/model_02_motion/duration": 35.0,
                    "/model_02_motion/scenario_id": "stale_case",
                },
            )
            try:
                recorder._write_run_metadata()
                recorder._handle_model_states(SimpleNamespace(name=[], pose=[]))

                manifest = self._read_manifest(recorder)

                self.assertEqual(manifest["source"], "explicit")
                self.assertEqual(manifest["controls"][0]["controlled_object"], "obstacle_block_left_clone_clone")
            finally:
                module.rospy = original_rospy

    def test_refresh_keeps_explicit_manifest_for_single_controller(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        with tempfile.TemporaryDirectory(prefix="sim_experiment_recorder_") as temp_dir:
            temp_dir = pathlib.Path(temp_dir)
            recorder, fake_rospy, original_rospy = self._make_manifest_recorder_fixture(
                module,
                temp_dir,
                params={
                    "/deform_monitor_v2": {},
                    "/model_01_motion/model_name": "model_01",
                    "/model_01_motion/command_frame": "world",
                    "/model_01_motion/control_rate": 20.0,
                    "/model_01_motion/command_timeout": 1.5,
                    "/model_01_motion/linear_x": 0.0,
                    "/model_01_motion/linear_y": 0.0,
                    "/model_01_motion/linear_z": 0.015,
                    "/model_01_motion/angular_x_deg": 0.0,
                    "/model_01_motion/angular_y_deg": 0.0,
                    "/model_01_motion/angular_z_deg": 0.0,
                    "/model_01_motion/start_delay": 8.0,
                    "/model_01_motion/duration": 20.0,
                    "/model_01_motion/scenario_id": "collapse_microdeform_case_01",
                },
            )
            try:
                recorder._write_run_metadata()
                recorder._handle_model_states(SimpleNamespace(name=[], pose=[]))

                manifest = self._read_manifest(recorder)

                self.assertEqual(manifest["source"], "explicit")
                self.assertEqual(manifest["controls"][0]["controlled_object"], "obstacle_block_left_clone_clone")
            finally:
                module.rospy = original_rospy

    def test_refresh_keeps_explicit_manifest_for_extra_motion_namespace(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        with tempfile.TemporaryDirectory(prefix="sim_experiment_recorder_") as temp_dir:
            temp_dir = pathlib.Path(temp_dir)
            recorder, fake_rospy, original_rospy = self._make_manifest_recorder_fixture(
                module,
                temp_dir,
                params={
                    "/deform_monitor_v2": {},
                    "/model_01_motion/model_name": "model_01",
                    "/model_01_motion/command_frame": "world",
                    "/model_01_motion/control_rate": 20.0,
                    "/model_01_motion/command_timeout": 1.5,
                    "/model_01_motion/linear_x": 0.0,
                    "/model_01_motion/linear_y": 0.0,
                    "/model_01_motion/linear_z": 0.015,
                    "/model_01_motion/angular_x_deg": 0.0,
                    "/model_01_motion/angular_y_deg": 0.0,
                    "/model_01_motion/angular_z_deg": 0.0,
                    "/model_01_motion/start_delay": 8.0,
                    "/model_01_motion/duration": 20.0,
                    "/model_01_motion/scenario_id": "collapse_microdeform_case_01",
                    "/model_02_motion/model_name": "model_02",
                    "/model_02_motion/command_frame": "world",
                    "/model_02_motion/control_rate": 10.0,
                    "/model_02_motion/command_timeout": 2.0,
                    "/model_02_motion/linear_x": 0.005,
                    "/model_02_motion/linear_y": 0.0,
                    "/model_02_motion/linear_z": 0.0,
                    "/model_02_motion/angular_x_deg": 0.0,
                    "/model_02_motion/angular_y_deg": 0.0,
                    "/model_02_motion/angular_z_deg": 0.0,
                    "/model_02_motion/start_delay": 12.5,
                    "/model_02_motion/duration": 35.0,
                    "/model_02_motion/scenario_id": "collapse_microdeform_case_01",
                    "/spawn_mid360_fastlio_motion/model_name": "spawn_mid360_fastlio",
                    "/spawn_mid360_fastlio_motion/command_frame": "world",
                    "/spawn_mid360_fastlio_motion/control_rate": 15.0,
                    "/spawn_mid360_fastlio_motion/command_timeout": 1.0,
                    "/spawn_mid360_fastlio_motion/linear_x": 0.0,
                    "/spawn_mid360_fastlio_motion/linear_y": 0.0,
                    "/spawn_mid360_fastlio_motion/linear_z": 0.0,
                    "/spawn_mid360_fastlio_motion/start_delay": 0.0,
                    "/spawn_mid360_fastlio_motion/duration": 0.0,
                    "/spawn_mid360_fastlio_motion/scenario_id": "collapse_microdeform_case_01",
                },
            )
            try:
                recorder._write_run_metadata()
                recorder._handle_model_states(SimpleNamespace(name=[], pose=[]))

                manifest = self._read_manifest(recorder)

                self.assertEqual(manifest["source"], "explicit")
                self.assertEqual(manifest["controls"][0]["controlled_object"], "obstacle_block_left_clone_clone")
            finally:
                module.rospy = original_rospy

    def test_refresh_keeps_explicit_manifest_for_incomplete_discovered_metadata(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        with tempfile.TemporaryDirectory(prefix="sim_experiment_recorder_") as temp_dir:
            temp_dir = pathlib.Path(temp_dir)
            recorder, fake_rospy, original_rospy = self._make_manifest_recorder_fixture(
                module,
                temp_dir,
                params={
                    "/deform_monitor_v2": {},
                    "/model_01_motion/model_name": "model_01",
                    "/model_01_motion/command_frame": "world",
                    "/model_01_motion/control_rate": 20.0,
                    "/model_01_motion/command_timeout": 1.5,
                    "/model_01_motion/linear_x": 0.0,
                    "/model_01_motion/linear_y": 0.0,
                    "/model_01_motion/linear_z": 0.015,
                    "/model_01_motion/angular_x_deg": 0.0,
                    "/model_01_motion/angular_y_deg": 0.0,
                    "/model_01_motion/angular_z_deg": 0.0,
                    "/model_01_motion/start_delay": 8.0,
                    "/model_01_motion/duration": 20.0,
                    "/model_01_motion/scenario_id": "collapse_microdeform_case_01",
                    "/model_02_motion/model_name": "model_02",
                    "/model_02_motion/command_frame": "",
                    "/model_02_motion/control_rate": 10.0,
                    "/model_02_motion/command_timeout": 2.0,
                    "/model_02_motion/linear_x": 0.005,
                    "/model_02_motion/linear_y": 0.0,
                    "/model_02_motion/linear_z": 0.0,
                    "/model_02_motion/angular_x_deg": 0.0,
                    "/model_02_motion/angular_y_deg": 0.0,
                    "/model_02_motion/angular_z_deg": 0.0,
                    "/model_02_motion/start_delay": None,
                    "/model_02_motion/duration": 35.0,
                    "/model_02_motion/scenario_id": "collapse_microdeform_case_01",
                },
            )
            try:
                recorder._write_run_metadata()
                recorder._handle_model_states(SimpleNamespace(name=[], pose=[]))

                manifest = self._read_manifest(recorder)

                self.assertEqual(manifest["source"], "explicit")
                self.assertEqual(
                    manifest["controls"][0]["controlled_object"],
                    "obstacle_block_left_clone_clone",
                )
            finally:
                module.rospy = original_rospy

    def test_refresh_does_not_churn_on_unchanged_discovered_manifest(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        with tempfile.TemporaryDirectory(prefix="sim_experiment_recorder_") as temp_dir:
            temp_dir = pathlib.Path(temp_dir)
            recorder, fake_rospy, original_rospy = self._make_manifest_recorder_fixture(
                module,
                temp_dir,
                params={
                    "/deform_monitor_v2": {},
                },
            )
            try:
                recorder._write_run_metadata()

                fake_rospy.params.update(
                    {
                        "/model_01_motion/model_name": "model_01",
                        "/model_01_motion/command_frame": "world",
                        "/model_01_motion/control_rate": 20.0,
                        "/model_01_motion/command_timeout": 1.5,
                        "/model_01_motion/linear_x": 0.0,
                        "/model_01_motion/linear_y": 0.0,
                        "/model_01_motion/linear_z": 0.015,
                        "/model_01_motion/angular_x_deg": 0.0,
                        "/model_01_motion/angular_y_deg": 0.0,
                        "/model_01_motion/angular_z_deg": 0.0,
                        "/model_01_motion/start_delay": 8.0,
                        "/model_01_motion/duration": 20.0,
                        "/model_01_motion/scenario_id": "collapse_microdeform_case_01",
                        "/model_02_motion/model_name": "model_02",
                        "/model_02_motion/command_frame": "world",
                        "/model_02_motion/control_rate": 10.0,
                        "/model_02_motion/command_timeout": 2.0,
                        "/model_02_motion/linear_x": 0.005,
                        "/model_02_motion/linear_y": 0.0,
                        "/model_02_motion/linear_z": 0.0,
                        "/model_02_motion/angular_x_deg": 0.0,
                        "/model_02_motion/angular_y_deg": 0.0,
                        "/model_02_motion/angular_z_deg": 0.0,
                        "/model_02_motion/start_delay": 12.5,
                        "/model_02_motion/duration": 35.0,
                        "/model_02_motion/scenario_id": "collapse_microdeform_case_01",
                    }
                )

                first_refresh = recorder._refresh_scenario_manifest_if_needed()
                second_refresh = recorder._refresh_scenario_manifest_if_needed()

                self.assertTrue(first_refresh)
                self.assertFalse(second_refresh)
                manifest = self._read_manifest(recorder)
                self.assertEqual(manifest["source"], "discovered")
                self.assertEqual(
                    [control["controlled_object"] for control in manifest["controls"]],
                    ["model_01", "model_02"],
                )
            finally:
                module.rospy = original_rospy

    def test_discover_controlled_objects_extracts_motion_controller_metadata(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        params = {
            "/debris_controller/model_name": "debris_block_02",
            "/debris_controller/command_frame": "body",
            "/debris_controller/control_rate": 20.0,
            "/debris_controller/command_timeout": 1.5,
            "/debris_controller/linear_y": 0.001,
            "/debris_controller/start_delay": 5.0,
            "/debris_controller/duration": 12.0,
            "/debris_controller/scenario_id": "case_body_y",
        }

        controls = module.discover_controlled_objects(
            get_param=lambda name, default=None: params.get(name, default),
            get_param_names=lambda: sorted(params.keys()),
        )

        self.assertEqual(len(controls), 1)
        self.assertEqual(controls[0]["controlled_object"], "debris_block_02")
        self.assertEqual(controls[0]["command_frame"], "body")
        self.assertEqual(controls[0]["axis"], {"x": 0.0, "y": 1.0, "z": 0.0})
        self.assertEqual(controls[0]["start_delay_sec"], 5.0)
        self.assertEqual(controls[0]["duration_sec"], 12.0)
        self.assertEqual(controls[0]["scenario_id"], "case_body_y")

    def test_discover_controlled_objects_discovers_multi_motion_controller_namespaces(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        params = {
            "/model_01_motion/model_name": "model_01",
            "/model_01_motion/command_frame": "world",
            "/model_01_motion/control_rate": 20.0,
            "/model_01_motion/command_timeout": 1.5,
            "/model_01_motion/linear_x": 0.0,
            "/model_01_motion/linear_y": 0.0,
            "/model_01_motion/linear_z": 0.015,
            "/model_01_motion/angular_x_deg": 0.0,
            "/model_01_motion/angular_y_deg": 0.0,
            "/model_01_motion/angular_z_deg": 0.0,
            "/model_01_motion/start_delay": 8.0,
            "/model_01_motion/duration": 20.0,
            "/model_01_motion/scenario_id": "case_model_01",
            "/model_02_motion/model_name": "model_02",
            "/model_02_motion/command_frame": "world",
            "/model_02_motion/control_rate": 10.0,
            "/model_02_motion/command_timeout": 2.0,
            "/model_02_motion/linear_x": 0.005,
            "/model_02_motion/linear_y": 0.0,
            "/model_02_motion/linear_z": 0.0,
            "/model_02_motion/angular_x_deg": 0.0,
            "/model_02_motion/angular_y_deg": 0.0,
            "/model_02_motion/angular_z_deg": 0.0,
            "/model_02_motion/start_delay": 12.5,
            "/model_02_motion/duration": 35.0,
            "/model_02_motion/scenario_id": "case_model_02",
        }

        controls = module.discover_controlled_objects(
            get_param=lambda name, default=None: params.get(name, default),
            get_param_names=lambda: sorted(params.keys()),
        )

        controls_by_object = {control["controlled_object"]: control for control in controls}
        self.assertEqual(set(controls_by_object), {"model_01", "model_02"})
        self.assertEqual(controls_by_object["model_01"]["command_frame"], "world")
        self.assertEqual(controls_by_object["model_02"]["command_frame"], "world")
        self.assertEqual(controls_by_object["model_01"]["scenario_id"], "case_model_01")
        self.assertEqual(controls_by_object["model_02"]["scenario_id"], "case_model_02")
        self.assertEqual(
            controls_by_object["model_01"]["axis"],
            {"x": 0.0, "y": 0.0, "z": 1.0},
        )
        self.assertEqual(
            controls_by_object["model_02"]["axis"],
            {"x": 1.0, "y": 0.0, "z": 0.0},
        )
        self.assertEqual(controls_by_object["model_01"]["start_delay_sec"], 8.0)
        self.assertEqual(controls_by_object["model_02"]["start_delay_sec"], 12.5)
        self.assertEqual(controls_by_object["model_01"]["duration_sec"], 20.0)
        self.assertEqual(controls_by_object["model_02"]["duration_sec"], 35.0)

    def test_discover_controlled_objects_ignores_namespaces_without_motion_contract(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        params = {
            "/model_01_motion/model_name": "model_01",
            "/model_01_motion/command_frame": "world",
            "/model_01_motion/control_rate": 20.0,
            "/model_01_motion/command_timeout": 1.5,
            "/model_01_motion/linear_x": 0.0,
            "/model_01_motion/linear_y": 0.0,
            "/model_01_motion/linear_z": 0.015,
            "/model_01_motion/angular_x_deg": 0.0,
            "/model_01_motion/angular_y_deg": 0.0,
            "/model_01_motion/angular_z_deg": 0.0,
            "/model_01_motion/start_delay": 8.0,
            "/model_01_motion/duration": 20.0,
            "/model_01_motion/scenario_id": "case_model_01",
            "/spawn_mid360_fastlio/model_name": "spawn_mid360_fastlio",
            "/spawn_mid360_fastlio/command_frame": "world",
            "/spawn_mid360_fastlio/control_rate": 30.0,
            "/spawn_mid360_fastlio/command_timeout": 0.2,
            "/spawn_mid360_fastlio/start_delay": 0.0,
            "/spawn_mid360_fastlio/duration": 0.0,
            "/spawn_mid360_fastlio/scenario_id": "false_positive_case",
        }

        controls = module.discover_controlled_objects(
            get_param=lambda name, default=None: params.get(name, default),
            get_param_names=lambda: sorted(params.keys()),
        )

        self.assertEqual(len(controls), 1)
        self.assertEqual(controls[0]["controlled_object"], "model_01")
        self.assertEqual(controls[0]["controller_namespace"], "/model_01_motion")

    def test_discover_controlled_objects_rejects_namespace_missing_required_metadata(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        params = {
            "/missing_rate_motion/model_name": "model_missing_rate",
            "/missing_rate_motion/command_frame": "world",
            "/missing_rate_motion/linear_x": 0.01,
            "/missing_rate_motion/command_timeout": 1.0,
            "/missing_rate_motion/start_delay": 3.0,
            "/missing_rate_motion/duration": 9.0,
            "/missing_rate_motion/scenario_id": "case_missing_rate",
        }

        controls = module.discover_controlled_objects(
            get_param=lambda name, default=None: params.get(name, default),
            get_param_names=lambda: sorted(params.keys()),
        )

        self.assertEqual(controls, [])

    def test_sim_launch_keeps_explicit_fallback_args(self):
        self._assert_launch_file_keeps_explicit_fallback_args(
            pathlib.Path(__file__).resolve().parents[1] / "launch" / "deform_monitor_v2_sim.launch"
        )

    def test_sim_dynamic_launch_keeps_explicit_fallback_args(self):
        self._assert_launch_file_keeps_explicit_fallback_args(
            pathlib.Path(__file__).resolve().parents[1]
            / "launch"
            / "deform_monitor_v2_sim_dynamic.launch"
        )

    def test_ensure_directories_creates_trajectory_dir(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        with tempfile.TemporaryDirectory(prefix="sim_experiment_recorder_") as temp_dir:
            temp_dir = pathlib.Path(temp_dir)
            recorder = module.SimExperimentRecorder.__new__(module.SimExperimentRecorder)
            recorder.meta_dir = temp_dir / "meta"
            recorder.truth_dir = temp_dir / "truth"
            recorder.truth_objects_dir = recorder.truth_dir / "objects"
            recorder.truth_links_dir = recorder.truth_dir / "links"
            recorder.algorithm_dir = temp_dir / "algorithm"
            recorder.trajectory_dir = temp_dir / "trajectory"

            recorder._ensure_directories()

            self.assertTrue(recorder.meta_dir.exists())
            self.assertTrue(recorder.truth_dir.exists())
            self.assertTrue(recorder.truth_objects_dir.exists())
            self.assertTrue(recorder.truth_links_dir.exists())
            self.assertTrue(recorder.algorithm_dir.exists())
            self.assertTrue(recorder.trajectory_dir.exists())

    def test_write_run_info_uses_configured_trajectory_export_settings(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        with tempfile.TemporaryDirectory(prefix="sim_experiment_recorder_") as temp_dir:
            temp_dir = pathlib.Path(temp_dir)
            recorder = module.SimExperimentRecorder.__new__(module.SimExperimentRecorder)
            recorder.run_dir = temp_dir / "sim_run_000"
            recorder.meta_dir = recorder.run_dir / "meta"
            recorder.meta_dir.mkdir(parents=True)
            recorder.truth_frame = "world"
            recorder.algorithm_frame = "camera_init"
            recorder.ego_model_name = "mid360_fastlio"
            recorder.model_states_topic = "/gazebo/model_states"
            recorder.link_states_topic = "/gazebo/link_states"
            recorder.ground_truth_odometry_topic = "/ground_truth/odom"
            recorder.risk_evidence_topic = "/deform/risk_evidence"
            recorder.risk_regions_topic = "/deform/risk_regions"
            recorder.persistent_risk_regions_topic = "/deform/persistent_risk_regions"
            recorder.structure_motions_topic = "/deform/structure_motions"
            recorder.odometry_topic = "/Odometry"
            recorder.sensor_scoped_link_name = "mid360_fastlio::mid360_link"
            recorder.sensor_frame_name = "livox"
            recorder.gt_tum_filename = "gt_sensor_world_tum.txt"
            recorder.odom_tum_filename = "odom_raw_tum.txt"

            recorder._write_run_info()

            payload = json.loads((recorder.meta_dir / "run_info.json").read_text())

            self.assertEqual(payload["topics"]["odometry"], "/Odometry")
            self.assertEqual(payload["topics"]["ground_truth_odometry"], "/ground_truth/odom")
            self.assertEqual(
                payload["topics"]["persistent_risk_regions"],
                "/deform/persistent_risk_regions",
            )
            self.assertEqual(
                payload["sensor_scoped_link_name"], "mid360_fastlio::mid360_link"
            )
            self.assertEqual(payload["sensor_frame_name"], "livox")
            self.assertEqual(payload["trajectory_export"]["gt_file"], "gt_sensor_world_tum.txt")
            self.assertEqual(
                payload["trajectory_export"]["odom_file"], "odom_raw_tum.txt"
            )
            self.assertEqual(
                payload["trajectory_export"]["gt_pose_source"],
                "ground_truth_odometry_plus_tf",
            )

    def test_handle_link_states_updates_sensor_pose_cache_for_target_scoped_link(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        fake_rospy = _FakeRospy()
        original_rospy = module.rospy
        module.rospy = fake_rospy
        try:
            recorder = module.SimExperimentRecorder.__new__(module.SimExperimentRecorder)
            recorder.ego_model_name = "mid360_fastlio"
            recorder.sensor_scoped_link_name = "mid360_fastlio::mid360_link"
            recorder._latest_sensor_pose_world = None
            recorder._latest_sensor_pose_stamp = None
            recorder.truth_frame = "world"
            recorder._link_files = {}

            msg = SimpleNamespace(
                name=["mid360_fastlio::mid360_link"],
                pose=[
                    SimpleNamespace(
                        position=SimpleNamespace(x=1.0, y=2.0, z=3.0),
                        orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
                    )
                ],
            )

            recorder._handle_link_states(msg)

            self.assertEqual(
                recorder._latest_sensor_pose_world,
                {
                    "position": {"x": 1.0, "y": 2.0, "z": 3.0},
                    "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                },
            )
            self.assertEqual(recorder._latest_sensor_pose_stamp, fake_rospy.now_sec)
        finally:
            module.rospy = original_rospy

    def test_build_run_info_includes_trajectory_export_metadata(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        payload = module.build_run_info_payload(
            run_dir=pathlib.Path("/tmp/sim_run_000"),
            truth_frame="world",
            algorithm_frame="camera_init",
            ego_model_name="mid360_fastlio",
            model_states_topic="/gazebo/model_states",
            link_states_topic="/gazebo/link_states",
            risk_evidence_topic="/deform/risk_evidence",
            risk_regions_topic="/deform/risk_regions",
            persistent_risk_regions_topic="/deform/persistent_risk_regions",
            structure_motions_topic="/deform/structure_motions",
            odometry_topic="/Odometry",
            sensor_scoped_link_name="mid360_fastlio::mid360_link",
            gt_tum_filename="gt_sensor_world_tum.txt",
            odom_tum_filename="odom_raw_tum.txt",
            ground_truth_odometry_topic="/ground_truth/odom",
            sensor_frame_name="livox",
        )

        self.assertEqual(payload["topics"]["odometry"], "/Odometry")
        self.assertEqual(payload["topics"]["ground_truth_odometry"], "/ground_truth/odom")
        self.assertEqual(
            payload["topics"]["persistent_risk_regions"],
            "/deform/persistent_risk_regions",
        )
        self.assertEqual(payload["sensor_scoped_link_name"], "mid360_fastlio::mid360_link")
        self.assertEqual(payload["sensor_frame_name"], "livox")
        self.assertEqual(payload["trajectory_export"]["enabled"], True)
        self.assertEqual(payload["trajectory_export"]["gt_file"], "gt_sensor_world_tum.txt")
        self.assertEqual(payload["trajectory_export"]["odom_file"], "odom_raw_tum.txt")
        self.assertEqual(
            payload["trajectory_export"]["timestamp_policy"], "odometry_master_clock"
        )
        self.assertEqual(payload["trajectory_export"]["runtime_alignment_applied"], False)
        self.assertEqual(
            payload["trajectory_export"]["gt_pose_source"],
            "ground_truth_odometry_plus_tf",
        )

    def test_build_run_info_disables_trajectory_export_without_sensor_link(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        payload = module.build_run_info_payload(
            run_dir=pathlib.Path("/tmp/sim_run_000"),
            truth_frame="world",
            algorithm_frame="camera_init",
            ego_model_name="mid360_fastlio",
            model_states_topic="/gazebo/model_states",
            link_states_topic="/gazebo/link_states",
            risk_evidence_topic="/deform/risk_evidence",
            risk_regions_topic="/deform/risk_regions",
            persistent_risk_regions_topic="/deform/persistent_risk_regions",
            structure_motions_topic="/deform/structure_motions",
            odometry_topic="/Odometry",
            sensor_scoped_link_name="",
            gt_tum_filename="gt_sensor_world_tum.txt",
            odom_tum_filename="odom_raw_tum.txt",
        )

        self.assertEqual(payload["sensor_scoped_link_name"], "")
        self.assertFalse(payload["trajectory_export"]["enabled"])

    def test_serialize_structure_motion_preserves_nested_motion_and_bbox_fields(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        msg = SimpleNamespace(
            motion=SimpleNamespace(x=0.12, y=-0.34, z=0.56),
            bbox_new_max=SimpleNamespace(x=4.0, y=5.0, z=6.0),
        )

        serialized = module.serialize_structure_motion(msg)

        self.assertEqual(serialized["motion"]["x"], 0.12)
        self.assertEqual(serialized["bbox_new_max"]["z"], 6.0)

    def test_parse_scoped_link_name_extracts_model_and_link_names(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        model_name, link_name = module.parse_scoped_link_name("crate_model::base_link")

        self.assertEqual(model_name, "crate_model")
        self.assertEqual(link_name, "base_link")


if __name__ == "__main__":
    unittest.main()
