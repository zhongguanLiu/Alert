from __future__ import annotations

import importlib.util
import pathlib
import sys
import types
import unittest


SCRIPT_PATH = (
    pathlib.Path(__file__).resolve().parents[1]
    / "scripts"
    / "work_patrol_cmd.py"
)

ROS_IMPORT_ROOTS = {
    "rospy",
    "geometry_msgs",
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

    spec = importlib.util.spec_from_file_location("work_patrol_cmd", SCRIPT_PATH)
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


class WorkPatrolCmdTests(unittest.TestCase):
    def test_default_patrol_mode_is_x_oscillate_with_yaw(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        self.assertEqual(module.DEFAULT_PATROL_MODE, "x_oscillate_with_yaw")

    def test_default_segment_duration_is_ten_seconds(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        self.assertAlmostEqual(module.DEFAULT_SEGMENT_DURATION_SEC, 10.0, places=9)

    def test_default_yaw_parameters_match_dynamic_patrol_profile(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        self.assertAlmostEqual(module.DEFAULT_YAW_AMPLITUDE_RAD_PER_SEC, 0.10, places=9)
        self.assertAlmostEqual(module.DEFAULT_YAW_PERIOD_SEC, 20.0, places=9)

    def test_angular_speed_is_linear_speed_over_radius(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        self.assertAlmostEqual(module.compute_angular_speed(0.3, 2.0), 0.15, places=9)

    def test_patrol_angular_z_is_negative_to_reverse_rotation_direction(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        self.assertAlmostEqual(module.compute_patrol_angular_z(0.2, 1.0), -0.2, places=9)

    def test_compute_angular_speed_rejects_non_positive_radius(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        with self.assertRaises(ValueError):
            module.compute_angular_speed(0.3, 0.0)

    def test_x_oscillate_velocity_alternates_direction_every_segment(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        self.assertAlmostEqual(
            module.compute_x_oscillate_linear_x(0.3, 10.0, 0.0),
            -0.3,
            places=9,
        )
        self.assertAlmostEqual(
            module.compute_x_oscillate_linear_x(0.3, 10.0, 9.999),
            -0.3,
            places=9,
        )
        self.assertAlmostEqual(
            module.compute_x_oscillate_linear_x(0.3, 10.0, 10.0),
            0.3,
            places=9,
        )
        self.assertAlmostEqual(
            module.compute_x_oscillate_linear_x(0.3, 10.0, 20.0),
            -0.3,
            places=9,
        )

    def test_x_oscillate_velocity_rejects_non_positive_segment_duration(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        with self.assertRaises(ValueError):
            module.compute_x_oscillate_linear_x(0.2, 0.0, 1.0)

    def test_x_oscillate_with_yaw_angular_velocity_follows_sine_profile(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        self.assertAlmostEqual(
            module.compute_x_oscillate_with_yaw_angular_z(0.10, 20.0, 0.0),
            0.0,
            places=9,
        )
        self.assertAlmostEqual(
            module.compute_x_oscillate_with_yaw_angular_z(0.10, 20.0, 5.0),
            0.10,
            places=9,
        )
        self.assertAlmostEqual(
            module.compute_x_oscillate_with_yaw_angular_z(0.10, 20.0, 10.0),
            0.0,
            places=9,
        )
        self.assertAlmostEqual(
            module.compute_x_oscillate_with_yaw_angular_z(0.10, 20.0, 15.0),
            -0.10,
            places=9,
        )

    def test_x_oscillate_with_yaw_rejects_non_positive_period(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        with self.assertRaises(ValueError):
            module.compute_x_oscillate_with_yaw_angular_z(0.10, 0.0, 1.0)


if __name__ == "__main__":
    unittest.main()
