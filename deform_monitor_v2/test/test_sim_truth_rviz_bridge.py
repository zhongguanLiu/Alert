# Author: zgliu@cumt.edu.cn
# Affiliation: China University of Mining and Technology
# Open-source release date: 2026-04-20
"""Unit tests for sim_truth_rviz_bridge.py.

The numeric positions and thresholds in this file are hand-crafted geometry
fixtures for transform and marker-publishing checks. They do not encode
experimental outcomes and are not consumed by the runtime pipeline.
"""

import importlib.util
import pathlib
import sys
import types
import unittest
from types import SimpleNamespace


SCRIPT_PATH = (
    pathlib.Path(__file__).resolve().parents[1] / "scripts" / "sim_truth_rviz_bridge.py"
)


ROS_IMPORT_ROOTS = {
    "rospy",
    "roslib",
    "std_msgs",
    "geometry_msgs",
    "sensor_msgs",
    "nav_msgs",
    "visualization_msgs",
    "gazebo_msgs",
    "tf",
}


class _StubModule(types.ModuleType):
    def __getattr__(self, name):
        value = type(name, (), {})
        setattr(self, name, value)
        return value


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

    spec = importlib.util.spec_from_file_location("sim_truth_rviz_bridge", SCRIPT_PATH)
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


class _FakePublisher:
    def __init__(self):
        self.messages = []

    def publish(self, msg):
        self.messages.append(msg)


class _FakePoint:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = x
        self.y = y
        self.z = z


class _FakePose:
    def __init__(self):
        self.position = _FakePoint()
        self.orientation = SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0)


class _FakeVector3:
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0


class _FakeColor:
    def __init__(self):
        self.r = 0.0
        self.g = 0.0
        self.b = 0.0
        self.a = 0.0


class _FakeHeader:
    def __init__(self):
        self.frame_id = ""
        self.stamp = None


class _FakeMarker:
    ADD = 0
    DELETEALL = 3
    SPHERE = 2
    ARROW = 0
    TEXT_VIEW_FACING = 9

    def __init__(self):
        self.header = _FakeHeader()
        self.ns = ""
        self.id = 0
        self.type = 0
        self.action = 0
        self.pose = _FakePose()
        self.scale = _FakeVector3()
        self.color = _FakeColor()
        self.text = ""
        self.points = []


class _FakeMarkerArray:
    def __init__(self):
        self.markers = []


class SimTruthRvizBridgeTests(unittest.TestCase):
    def test_derive_world_from_algorithm_transform_uses_initial_pose_pair(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        transform = module.derive_world_from_algorithm_transform(
            truth_reference_pose_world={
                "position": {"x": 10.0, "y": 0.0, "z": 0.0},
                "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
            },
            algorithm_reference_pose_algorithm={
                "position": {"x": 1.0, "y": 0.0, "z": 0.0},
                "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
            },
            source_frame="camera_init",
            target_frame="world",
        )

        self.assertEqual(transform["source_frame"], "camera_init")
        self.assertEqual(transform["target_frame"], "world")
        self.assertEqual(transform["translation"].tolist(), [9.0, 0.0, 0.0])

    def test_handle_model_states_skips_publish_until_alignment_ready(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        module.Marker = _FakeMarker
        module.MarkerArray = _FakeMarkerArray
        module.Point = _FakePoint

        bridge = module.SimTruthRvizBridge.__new__(module.SimTruthRvizBridge)
        bridge.algorithm_frame = "camera_init"
        bridge.ego_model_name = "mid360_fastlio"
        bridge.moving_threshold_m = 0.01
        bridge.marker_pub = _FakePublisher()
        bridge._algorithm_from_world_transform = None
        bridge._initial_object_positions_world = {}
        bridge._current_stamp = None

        msg = SimpleNamespace(
            name=["mid360_fastlio", "moving_box"],
            pose=[
                SimpleNamespace(
                    position=SimpleNamespace(x=0.0, y=0.0, z=0.0),
                    orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
                ),
                SimpleNamespace(
                    position=SimpleNamespace(x=1.0, y=2.0, z=0.0),
                    orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
                ),
            ],
        )

        bridge._handle_model_states(msg)

        self.assertEqual(len(bridge.marker_pub.messages), 0)
        self.assertEqual(
            bridge._initial_object_positions_world["moving_box"],
            {"x": 1.0, "y": 2.0, "z": 0.0},
        )

    def test_handle_model_states_publishes_only_moving_truth_objects_in_algorithm_frame(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        module.Marker = _FakeMarker
        module.MarkerArray = _FakeMarkerArray
        module.Point = _FakePoint

        bridge = module.SimTruthRvizBridge.__new__(module.SimTruthRvizBridge)
        bridge.algorithm_frame = "camera_init"
        bridge.ego_model_name = "mid360_fastlio"
        bridge.moving_threshold_m = 0.01
        bridge.marker_pub = _FakePublisher()
        bridge._algorithm_from_world_transform = {
            "source_frame": "world",
            "target_frame": "camera_init",
            "rotation": module.np.eye(3),
            "translation": module.np.array([0.0, 0.0, 0.0]),
        }
        bridge._initial_object_positions_world = {
            "static_box": {"x": 0.0, "y": 0.0, "z": 0.0},
            "moving_box": {"x": 1.0, "y": 2.0, "z": 0.0},
        }
        bridge._current_stamp = 123.0

        msg = SimpleNamespace(
            name=["mid360_fastlio", "static_box", "moving_box"],
            pose=[
                SimpleNamespace(
                    position=SimpleNamespace(x=0.0, y=0.0, z=0.0),
                    orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
                ),
                SimpleNamespace(
                    position=SimpleNamespace(x=0.0, y=0.0, z=0.0),
                    orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
                ),
                SimpleNamespace(
                    position=SimpleNamespace(x=1.05, y=2.0, z=0.0),
                    orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
                ),
            ],
        )

        bridge._handle_model_states(msg)

        self.assertEqual(len(bridge.marker_pub.messages), 1)
        marker_array = bridge.marker_pub.messages[0]
        self.assertGreaterEqual(len(marker_array.markers), 3)
        current_marker = next(m for m in marker_array.markers if m.ns == "sim_truth_current")
        arrow_marker = next(m for m in marker_array.markers if m.ns == "sim_truth_motion")
        label_marker = next(m for m in marker_array.markers if m.ns == "sim_truth_labels")

        self.assertEqual(current_marker.header.frame_id, "camera_init")
        self.assertAlmostEqual(current_marker.pose.position.x, 1.05)
        self.assertAlmostEqual(current_marker.pose.position.y, 2.0)
        self.assertEqual(len(arrow_marker.points), 2)
        self.assertEqual(label_marker.text, "moving_box")


if __name__ == "__main__":
    unittest.main()
