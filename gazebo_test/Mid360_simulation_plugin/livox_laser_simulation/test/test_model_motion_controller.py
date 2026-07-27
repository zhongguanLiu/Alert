import importlib.util
import math
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


class _FakePose:
    def __init__(self):
        self.position = _FakeVector()
        self.orientation = types.SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0)


def _quaternion_multiply(left, right):
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    return (
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
        lw * rw - lx * rx - ly * ry - lz * rz,
    )


def _quaternion_from_euler(roll, pitch, yaw):
    cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
    cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


class ModelMotionControllerTimingTests(unittest.TestCase):
    def test_controller_has_no_wall_time_completion_or_hard_exit(self):
        source = SCRIPT_PATH.read_text()
        self.assertNotIn("time.sleep", source)
        self.assertNotIn("os._exit", source)
        self.assertNotIn("_start_completion_timer", source)
        self.assertIn('rospy.get_param("~motion_start_sim_time"', source)
        self.assertIn("/use_sim_time must be true", source)

    def test_controller_has_no_nonzero_physics_twist_mode(self):
        source = SCRIPT_PATH.read_text()

        self.assertNotIn('rospy.get_param("~set_twist"', source)
        self.assertNotIn("if self.set_twist", source)

    def test_controller_explicitly_clears_all_physics_velocity_components(self):
        module = load_module_if_exists()
        twist = _FakeTwist()
        twist.linear.x = 1.0
        twist.linear.y = 2.0
        twist.linear.z = 3.0
        twist.angular.x = 4.0
        twist.angular.y = 5.0
        twist.angular.z = 6.0

        module.clear_twist(twist)

        self.assertEqual(
            (
                twist.linear.x,
                twist.linear.y,
                twist.linear.z,
                twist.angular.x,
                twist.angular.y,
                twist.angular.z,
            ),
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        )

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

    def test_non_positive_duration_keeps_command_active_after_start_delay(self):
        module = load_module_if_exists()
        module.Twist = _FakeTwist
        scheduled_cmd = _FakeTwist()
        scheduled_cmd.angular.z = 0.2

        for duration in (0.0, -1.0):
            with self.subTest(duration=duration):
                active_cmd = module.compute_scheduled_twist(
                    scheduled_cmd,
                    now_sec=500.0,
                    motion_start_time=100.0,
                    start_delay=8.0,
                    duration=duration,
                    twist_factory=module.Twist,
                )
                self.assertEqual(active_cmd.angular.z, 0.2)

    def test_motion_start_time_prefers_explicit_sim_time_or_first_valid_clock(self):
        module = load_module_if_exists()
        self.assertEqual(module.resolve_motion_start_time(25.0, 100.0), 25.0)
        self.assertEqual(module.resolve_motion_start_time(-1.0, 100.0), 100.0)
        self.assertIsNone(module.resolve_motion_start_time(-1.0, 0.0))

    def test_sim_time_step_does_not_advance_during_pause_and_resets_after_jump(self):
        module = load_module_if_exists()
        dt, anchor = module.compute_sim_time_step(10.0, None, max_dt=0.2)
        self.assertEqual((dt, anchor), (0.0, 10.0))

        dt, anchor = module.compute_sim_time_step(10.0, anchor, max_dt=0.2)
        self.assertEqual((dt, anchor), (0.0, 10.0))

        dt, anchor = module.compute_sim_time_step(2.0, anchor, max_dt=0.2)
        self.assertEqual((dt, anchor), (0.0, 2.0))
        dt, anchor = module.compute_sim_time_step(2.1, anchor, max_dt=0.2)
        self.assertAlmostEqual(dt, 0.1)
        self.assertEqual(anchor, 2.1)

    def test_sim_time_step_never_discards_large_forward_progress(self):
        module = load_module_if_exists()

        dt, anchor = module.compute_sim_time_step(10.75, 10.0, max_dt=0.2)

        self.assertAlmostEqual(dt, 0.75)
        self.assertEqual(anchor, 10.75)

    def test_motion_interval_is_clipped_to_start_delay_and_duration(self):
        module = load_module_if_exists()

        self.assertAlmostEqual(
            module.compute_scheduled_interval_duration(
                107.9, 108.1, motion_start_time=100.0, start_delay=8.0, duration=20.0
            ),
            0.1,
        )
        self.assertAlmostEqual(
            module.compute_scheduled_interval_duration(
                127.9, 128.4, motion_start_time=100.0, start_delay=8.0, duration=20.0
            ),
            0.1,
        )
        self.assertEqual(
            module.compute_scheduled_interval_duration(
                128.0, 129.0, motion_start_time=100.0, start_delay=8.0, duration=20.0
            ),
            0.0,
        )

    def test_body_twist_integration_is_independent_of_control_tick_partition(self):
        module = load_module_if_exists()
        module.quaternion_multiply = _quaternion_multiply
        command = _FakeTwist()
        command.linear.x = 1.0
        command.angular.z = math.pi / 2.0

        one_step = module.ModelMotionController.__new__(module.ModelMotionController)
        one_step.command_frame = "body"
        one_step_pose = _FakePose()
        one_step.integrate_state(one_step_pose, command, 1.0)

        ten_steps = module.ModelMotionController.__new__(module.ModelMotionController)
        ten_steps.command_frame = "body"
        ten_step_pose = _FakePose()
        for _ in range(10):
            ten_steps.integrate_state(ten_step_pose, command, 0.1)

        self.assertAlmostEqual(one_step_pose.position.x, 2.0 / math.pi, places=9)
        self.assertAlmostEqual(one_step_pose.position.y, 2.0 / math.pi, places=9)
        self.assertAlmostEqual(one_step_pose.position.x, ten_step_pose.position.x, places=9)
        self.assertAlmostEqual(one_step_pose.position.y, ten_step_pose.position.y, places=9)
        self.assertAlmostEqual(one_step_pose.orientation.z, ten_step_pose.orientation.z, places=9)
        self.assertAlmostEqual(one_step_pose.orientation.w, ten_step_pose.orientation.w, places=9)

    def test_commanded_pose_is_initialized_from_gazebo_only_once(self):
        module = load_module_if_exists()
        controller = module.ModelMotionController.__new__(module.ModelMotionController)
        controller._target_pose = None
        response = types.SimpleNamespace(pose=_FakePose())
        read_count = []

        def read_model_state():
            read_count.append(True)
            return response

        controller._read_model_state = read_model_state

        self.assertTrue(controller._initialize_target_pose())
        response.pose.position.x = 100.0
        self.assertTrue(controller._initialize_target_pose())

        self.assertEqual(len(read_count), 1)
        self.assertEqual(controller._target_pose.position.x, 0.0)

    def test_command_changes_are_split_at_their_simulation_timestamps(self):
        module = load_module_if_exists()
        module.Twist = _FakeTwist
        initial = _FakeTwist()
        initial.linear.x = 1.0
        updated = _FakeTwist()
        updated.linear.x = 2.0

        segments, active, active_stamp, remaining = module.build_command_segments(
            initial_command=initial,
            initial_command_stamp=None,
            command_events=[(1.9, updated)],
            interval_start=1.0,
            interval_end=2.0,
            command_timeout=0.0,
            twist_factory=_FakeTwist,
        )

        self.assertEqual(
            [(start, end, command.linear.x) for start, end, command in segments],
            [(1.0, 1.9, 1.0), (1.9, 2.0, 2.0)],
        )
        self.assertEqual(active.linear.x, 2.0)
        self.assertEqual(active_stamp, 1.9)
        self.assertEqual(remaining, [])

    def test_command_timeout_splits_an_interval_instead_of_zeroing_it_all(self):
        module = load_module_if_exists()
        command = _FakeTwist()
        command.linear.y = 3.0

        segments, _, _, _ = module.build_command_segments(
            initial_command=command,
            initial_command_stamp=1.0,
            command_events=[],
            interval_start=1.4,
            interval_end=1.7,
            command_timeout=0.5,
            twist_factory=_FakeTwist,
        )

        self.assertEqual(len(segments), 2)
        self.assertAlmostEqual(segments[0][0], 1.4)
        self.assertAlmostEqual(segments[0][1], 1.5)
        self.assertEqual(segments[0][2].linear.y, 3.0)
        self.assertAlmostEqual(segments[1][0], 1.5)
        self.assertAlmostEqual(segments[1][1], 1.7)
        self.assertEqual(segments[1][2].linear.y, 0.0)

    def test_explicit_past_start_time_is_used_as_initial_integration_anchor(self):
        module = load_module_if_exists()

        self.assertEqual(
            module.resolve_initial_integration_anchor(
                now_sec=20.0,
                motion_start_time=10.0,
                configured_start_time=10.0,
            ),
            10.0,
        )
        self.assertIsNone(
            module.resolve_initial_integration_anchor(
                now_sec=20.0,
                motion_start_time=20.0,
                configured_start_time=-1.0,
            )
        )

    def test_backward_simulation_time_is_a_hard_schedule_error(self):
        module = load_module_if_exists()

        self.assertTrue(module.simulation_time_regressed(2.0, 10.0))
        self.assertFalse(module.simulation_time_regressed(10.0, 10.0))
        self.assertFalse(module.simulation_time_regressed(10.1, 10.0))

    def test_time_reset_finalization_never_writes_the_cached_future_pose(self):
        module = load_module_if_exists()
        controller = module.ModelMotionController.__new__(module.ModelMotionController)
        writes = []
        controller._write_zero_twist = lambda: writes.append(True)

        self.assertFalse(controller._finalize_motion(write_final_state=False))
        self.assertEqual(writes, [])
        self.assertTrue(controller._finalize_motion(write_final_state=True))
        self.assertEqual(writes, [True])

    def test_body_and_world_commands_preserve_translation_rotation_semantics(self):
        module = load_module_if_exists()
        module.quaternion_multiply = _quaternion_multiply
        module.quaternion_from_euler = _quaternion_from_euler
        yaw_90 = _quaternion_from_euler(0.0, 0.0, math.pi / 2.0)

        body_controller = module.ModelMotionController.__new__(module.ModelMotionController)
        body_controller.command_frame = "body"
        body_pose = _FakePose()
        (
            body_pose.orientation.x,
            body_pose.orientation.y,
            body_pose.orientation.z,
            body_pose.orientation.w,
        ) = yaw_90
        command = _FakeTwist()
        command.linear.x = 1.0
        command.angular.z = math.pi / 2.0
        linear_world, _ = body_controller.integrate_state(body_pose, command, 1.0)
        self.assertAlmostEqual(linear_world[0], 0.0, places=9)
        self.assertAlmostEqual(linear_world[1], 1.0, places=9)
        self.assertAlmostEqual(body_pose.position.x, -2.0 / math.pi, places=9)
        self.assertAlmostEqual(body_pose.position.y, 2.0 / math.pi, places=9)

        world_controller = module.ModelMotionController.__new__(module.ModelMotionController)
        world_controller.command_frame = "world"
        world_pose = _FakePose()
        (
            world_pose.orientation.x,
            world_pose.orientation.y,
            world_pose.orientation.z,
            world_pose.orientation.w,
        ) = yaw_90
        linear_world, _ = world_controller.integrate_state(world_pose, command, 1.0)
        self.assertEqual(linear_world, (1.0, 0.0, 0.0))
        self.assertAlmostEqual(world_pose.position.x, 1.0, places=9)
        self.assertAlmostEqual(abs(world_pose.orientation.w), 0.0, places=9)

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
