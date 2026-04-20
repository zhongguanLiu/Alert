import importlib.util
import pathlib
import sys
import types
import unittest


SCRIPT_PATH = (
    pathlib.Path(__file__).resolve().parents[1]
    / "scripts"
    / "model_motion_controller.py"
)


ROS_IMPORT_ROOTS = {
    "rospy",
    "gazebo_msgs",
    "geometry_msgs",
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

    spec = importlib.util.spec_from_file_location("model_motion_controller", SCRIPT_PATH)
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


class _FakeVector:
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0


class _FakeTwist:
    def __init__(self):
        self.linear = _FakeVector()
        self.angular = _FakeVector()


class ModelMotionControllerTimingTests(unittest.TestCase):
    def test_timed_command_stays_idle_before_start_delay(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        module.Twist = _FakeTwist
        controller = module.ModelMotionController.__new__(module.ModelMotionController)
        controller.current_cmd = _FakeTwist()
        controller.current_cmd.linear.y = 0.001
        controller.last_cmd_time = 0.0
        controller.command_timeout = 0.0
        controller.start_delay = 8.0
        controller.duration = 20.0
        controller.motion_start_time = 100.0
        controller.lock = types.SimpleNamespace(__enter__=lambda self: self, __exit__=lambda self, exc_type, exc, tb: False)

        active_cmd = module.compute_scheduled_twist(
            controller.current_cmd,
            now_sec=105.0,
            motion_start_time=controller.motion_start_time,
            start_delay=controller.start_delay,
            duration=controller.duration,
            twist_factory=module.Twist,
        )

        self.assertEqual(active_cmd.linear.y, 0.0)

    def test_timed_command_stops_after_duration(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        module.Twist = _FakeTwist
        scheduled_cmd = _FakeTwist()
        scheduled_cmd.linear.y = 0.001

        active_cmd = module.compute_scheduled_twist(
            scheduled_cmd,
            now_sec=129.5,
            motion_start_time=100.0,
            start_delay=8.0,
            duration=20.0,
            twist_factory=module.Twist,
        )

        self.assertEqual(active_cmd.linear.y, 0.0)

    def test_timed_command_is_active_inside_motion_window(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        module.Twist = _FakeTwist
        scheduled_cmd = _FakeTwist()
        scheduled_cmd.linear.y = 0.001

        active_cmd = module.compute_scheduled_twist(
            scheduled_cmd,
            now_sec=118.0,
            motion_start_time=100.0,
            start_delay=8.0,
            duration=20.0,
            twist_factory=module.Twist,
        )

        self.assertEqual(active_cmd.linear.y, 0.001)

    def test_completion_announcement_stays_silent_before_deadline(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        controller = module.ModelMotionController.__new__(module.ModelMotionController)
        controller.motion_start_time = 100.0
        controller.start_delay = 8.0
        controller.duration = 20.0
        controller.completion_announced = False

        emitted = []
        announced = module.ModelMotionController.maybe_announce_completion(
            controller,
            now_sec=127.9,
            writer=emitted.append,
        )

        self.assertFalse(announced)
        self.assertEqual(emitted, [])
        self.assertFalse(controller.completion_announced)

    def test_completion_announcement_emits_red_banner_once_at_deadline(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        controller = module.ModelMotionController.__new__(module.ModelMotionController)
        controller.motion_start_time = 100.0
        controller.start_delay = 8.0
        controller.duration = 20.0
        controller.completion_announced = False

        emitted = []
        announced = module.ModelMotionController.maybe_announce_completion(
            controller,
            now_sec=128.0,
            writer=emitted.append,
        )

        self.assertTrue(announced)
        self.assertTrue(controller.completion_announced)
        self.assertEqual(
            emitted,
            [f"\033[31m{module.COMPLETION_MESSAGE}\033[0m"],
        )

        announced_again = module.ModelMotionController.maybe_announce_completion(
            controller,
            now_sec=140.0,
            writer=emitted.append,
        )

        self.assertFalse(announced_again)
        self.assertEqual(
            emitted,
            [f"\033[31m{module.COMPLETION_MESSAGE}\033[0m"],
        )

    def test_completion_announcement_is_disabled_for_infinite_duration(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        controller = module.ModelMotionController.__new__(module.ModelMotionController)
        controller.motion_start_time = 100.0
        controller.start_delay = 8.0
        controller.duration = 0.0
        controller.completion_announced = False

        emitted = []
        announced = module.ModelMotionController.maybe_announce_completion(
            controller,
            now_sec=500.0,
            writer=emitted.append,
        )

        self.assertFalse(announced)
        self.assertEqual(emitted, [])
        self.assertFalse(controller.completion_announced)


if __name__ == "__main__":
    unittest.main()
