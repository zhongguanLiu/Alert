from __future__ import annotations

import pathlib
import unittest
import xml.etree.ElementTree as ET

import yaml


PACKAGE_ROOT = pathlib.Path(__file__).resolve().parents[1]
WORLD_PATH = (
    PACKAGE_ROOT
    / "worlds"
    / "tracked_mid360_fastlio_collapse_microdeform_sim.world"
)
CONFIG_PATH = PACKAGE_ROOT / "config" / "synchronized_multi_model_motion.yaml"
LAUNCH_PATH = PACKAGE_ROOT / "launch" / "synchronized_multi_model_motion.launch"
PLUGIN_PATH = (
    PACKAGE_ROOT / "src" / "synchronized_multi_model_motion_world_plugin.cpp"
)
CLI_PATH = PACKAGE_ROOT / "scripts" / "synchronized_motion_cli.py"

EXPECTED_CONTROLLED = {
    1: "l_shape_target",
    2: "model_01",
    3: "model_02",
    4: "table",
    5: "table_marble",
    6: "person_walking",
    7: "table_clone",
    8: "cardboard_box",
    9: "lzg_bianwood",
    10: "bookshelf",
}

EXPECTED_MOTION = {
    1: ([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]),
    2: ([0.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 0.0, 0.0]),
    3: ([1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
    4: ([0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
    5: ([0.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 0.0]),
    6: ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, -0.1, 0.0]),
    7: ([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]),
    8: ([1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
    9: ([0.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
    10: ([0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
}


def load_world():
    world = ET.parse(WORLD_PATH).getroot().find("world")
    if world is None:
        raise AssertionError("missing <world>")
    return world


def model_object_id(model):
    ids = {
        int(float(collision.findtext("laser_retro")))
        for collision in model.findall(".//collision")
        if collision.findtext("laser_retro") is not None
    }
    if not ids:
        return None
    if len(ids) != 1:
        raise AssertionError(f"{model.get('name')}: IDs={ids}")
    return ids.pop()


class SynchronizedMultiModelMotionTests(unittest.TestCase):
    def test_world_loads_exactly_one_idle_synchronization_plugin(self):
        plugins = load_world().findall("plugin")
        matching = [
            plugin
            for plugin in plugins
            if plugin.get("name") == "synchronized_multi_model_motion"
        ]
        self.assertEqual(len(matching), 1)
        self.assertEqual(
            matching[0].get("filename"),
            "libsynchronized_multi_model_motion_world_plugin.so",
        )
        self.assertEqual(
            matching[0].findtext("ros_namespace"),
            "/synchronized_model_motion",
        )

    def test_controlled_and_fixed_objects_have_disjoint_physics_ownership(self):
        models = {}
        for model in load_world().findall("model"):
            object_id = model_object_id(model)
            if object_id is not None:
                models[object_id] = model
        self.assertEqual(set(models), set(range(1, 26)))
        self.assertEqual(
            {object_id: models[object_id].get("name") for object_id in range(1, 11)},
            EXPECTED_CONTROLLED,
        )
        for object_id in range(1, 11):
            with self.subTest(object_id=object_id):
                self.assertEqual(models[object_id].findtext("static", "0"), "0")
        for object_id in range(11, 26):
            with self.subTest(object_id=object_id):
                self.assertEqual(models[object_id].findtext("static"), "1")

    def test_default_plan_matches_the_confirmed_per_object_motion_modes(self):
        config = yaml.safe_load(CONFIG_PATH.read_text())
        self.assertEqual(config["start_delay"], 10.0)
        self.assertEqual(config["duration"], 40.0)
        self.assertEqual(config["end_hold"], 10.0)
        self.assertEqual(config["linear_speed_mm_s"], 1.0)
        self.assertEqual(config["angular_speed_rad_s"], 0.0015)
        self.assertEqual(config["person_walking_switch_time"], 20.0)
        self.assertEqual(
            config["person_walking_second_linear_mps"], [0.0, 0.1, 0.0]
        )
        entries = config["models"]
        self.assertEqual(len(entries), 10)
        self.assertEqual(
            {entry["id"]: entry["model_name"] for entry in entries},
            EXPECTED_CONTROLLED,
        )
        for entry in entries:
            with self.subTest(object_id=entry["id"]):
                self.assertEqual(entry["command_frame"], "world")
                expected = EXPECTED_MOTION[entry["id"]]
                self.assertEqual(entry["linear_direction"], expected[0])
                self.assertEqual(entry["angular_axis"], expected[1])
                self.assertEqual(entry["fixed_linear_mps"], expected[2])

    def test_table_marble_high_local_x_end_moves_toward_positive_world_z(self):
        config = yaml.safe_load(CONFIG_PATH.read_text())
        table_marble = next(entry for entry in config["models"] if entry["id"] == 5)
        omega = table_marble["angular_axis"]
        high_end = [1.0, 0.0, 0.0]
        velocity_z = omega[0] * high_end[1] - omega[1] * high_end[0]

        self.assertEqual(omega, [0.0, -1.0, 0.0])
        self.assertGreater(velocity_z, 0.0)

    def test_speed_units_and_person_walking_two_phase_velocity_are_exact(self):
        entries = {
            entry["id"]: entry
            for entry in yaml.safe_load(CONFIG_PATH.read_text())["models"]
        }

        def effective_velocity(entry, linear_speed_mm_s, angular_speed_rad_s):
            linear = [
                direction * linear_speed_mm_s * 0.001 + fixed
                for direction, fixed in zip(
                    entry["linear_direction"], entry["fixed_linear_mps"]
                )
            ]
            angular = [
                axis * angular_speed_rad_s for axis in entry["angular_axis"]
            ]
            return linear, angular

        expected_linear_at_one_mm_s = {
            3: [0.001, 0.0, 0.0],
            4: [0.0, 0.001, 0.0],
            6: [0.0, -0.1, 0.0],
            8: [0.001, 0.0, 0.0],
            10: [0.0, 0.001, 0.0],
        }
        for object_id, expected in expected_linear_at_one_mm_s.items():
            with self.subTest(object_id=object_id):
                linear, _ = effective_velocity(entries[object_id], 1.0, 0.01)
                self.assertEqual(linear, expected)

        expected_angular = {
            1: [0.0, 0.0, 0.0015],
            2: [0.0, 0.0, -0.0015],
            5: [0.0, -0.0015, 0.0],
            7: [0.0, 0.0, 0.0015],
            9: [-0.0015, 0.0, 0.0],
        }
        for object_id, expected in expected_angular.items():
            with self.subTest(object_id=object_id):
                _, angular = effective_velocity(entries[object_id], 1.0, 0.0015)
                self.assertEqual(angular, expected)
                self.assertEqual(sum(abs(value) for value in angular), 0.0015)

        for experiment_speed in (0.0, 0.5, 1.0, 2.0, 5.0):
            with self.subTest(linear_speed_mm_s=experiment_speed):
                linear, angular = effective_velocity(
                    entries[6], experiment_speed, 1.0
                )
                self.assertEqual(linear, [0.0, -0.1, 0.0])
                self.assertEqual(angular, [0.0, 0.0, 0.0])
        self.assertEqual(
            yaml.safe_load(CONFIG_PATH.read_text())[
                "person_walking_second_linear_mps"
            ],
            [0.0, 0.1, 0.0],
        )

    def test_model_01_geometry_truth_corners_and_counter_rotation_are_aligned(self):
        world = load_world()
        model = world.find("model[@name='model_01']")
        self.assertIsNotNone(model)
        self.assertIsNone(world.find("state"))
        declared_pose = tuple(float(value) for value in model.findtext("pose").split())
        link = model.find("link[@name='link']")
        visual_size = tuple(
            float(value)
            for value in link.findtext("visual[@name='visual']/geometry/box/size").split()
        )
        collision_size = tuple(
            float(value)
            for value in link.findtext("collision/geometry/box/size").split()
        )
        self.assertEqual(visual_size, (2.53211, 0.676971, 2.31086))
        self.assertEqual(collision_size, visual_size)
        half = tuple(value * 0.5 for value in collision_size)
        self.assertAlmostEqual(declared_pose[2], half[2], places=9)

        expected_corners = {
            (sx * half[0], sy * half[1], sz * half[2])
            for sx in (-1.0, 1.0)
            for sy in (-1.0, 1.0)
            for sz in (-1.0, 1.0)
        }
        marker_corners = {
            tuple(float(value) for value in visual.findtext("pose").split()[:3])
            for visual in link.findall("visual")
            if visual.get("name", "").startswith("ground_truth_marker_v_")
        }
        truth_link_corners = {
            tuple(float(value) for value in truth_link.findtext("pose").split()[:3])
            for truth_link in model.findall("link")
            if truth_link.get("name", "").startswith("ground_truth_v_")
        }
        self.assertEqual(marker_corners, expected_corners)
        self.assertEqual(truth_link_corners, expected_corners)

        config = yaml.safe_load(CONFIG_PATH.read_text())
        l_shape = next(entry for entry in config["models"] if entry["id"] == 1)
        model_01 = next(entry for entry in config["models"] if entry["id"] == 2)
        self.assertEqual(l_shape["angular_axis"], [0.0, 0.0, 1.0])
        self.assertEqual(model_01["linear_direction"], [0.0, 0.0, 0.0])
        self.assertEqual(model_01["angular_axis"], [0.0, 0.0, -1.0])

    def test_plugin_delegates_integration_to_gazebo_without_pose_writes(self):
        source = PLUGIN_PATH.read_text()
        self.assertIn("ConnectWorldUpdateBegin", source)
        self.assertIn("SetLinearVel", source)
        self.assertIn("SetAngularVel", source)
        self.assertIn("SetKinematic(true)", source)
        self.assertNotIn("SetWorldPose", source)
        self.assertNotIn("SetModelState", source)
        self.assertNotIn("integrate", source.lower())

    def test_plugin_has_separate_speed_units_and_locked_person_velocity(self):
        source = PLUGIN_PATH.read_text()
        self.assertIn("State::Holding", source)
        self.assertIn('getParam("end_hold"', source)
        self.assertIn('getParam("linear_speed_mm_s"', source)
        self.assertIn('getParam("angular_speed_rad_s"', source)
        self.assertIn('getParam("person_walking_switch_time"', source)
        self.assertIn("personWalkingSecondLinear_", source)
        self.assertIn("_runningTime >= this->personWalkingSwitchTime_", source)
        self.assertIn("kMillimetersToMeters", source)
        self.assertIn("expectedWalkingVelocity(0.0, -0.1, 0.0)", source)
        self.assertIn("expectedSecondWalkingVelocity", source)
        self.assertNotIn("speedScale_", source)

    def test_launch_never_starts_object_or_robot_motion_automatically(self):
        launch = ET.parse(LAUNCH_PATH).getroot()
        self.assertEqual(launch.tag, "launch")
        self.assertEqual(launch.findall("node"), [])
        include = launch.find("include")
        self.assertIsNotNone(include)
        arguments = {
            argument.get("name"): argument.get("value")
            for argument in include.findall("arg")
        }
        self.assertEqual(arguments["spawn_robot"], "true")
        self.assertEqual(arguments["auto_drive"], "false")
        self.assertEqual(arguments["keyboard_teleop"], "false")
        launch_arguments = {
            argument.get("name"): argument.get("default")
            for argument in launch.findall("arg")
        }
        self.assertEqual(launch_arguments["linear_speed_mm_s"], "1.0")
        self.assertEqual(launch_arguments["angular_speed_rad_s"], "0.0015")
        self.assertNotIn("speed_scale", launch_arguments)
        source = LAUNCH_PATH.read_text()
        self.assertNotIn("/synchronized_model_motion/start", source)

    def test_robot_is_spawned_only_by_launch(self):
        embedded_robots = load_world().findall("model[@name='mid360_fastlio']")
        saved_robots = load_world().findall(
            "state/model[@name='mid360_fastlio']"
        )
        self.assertEqual(embedded_robots, [])
        self.assertEqual(saved_robots, [])

        launch = ET.parse(LAUNCH_PATH).getroot()
        include = launch.find("include")
        include_arguments = {
            argument.get("name"): argument.get("value")
            for argument in include.findall("arg")
        }
        self.assertEqual(include_arguments["spawn_robot"], "true")

        spawn_source = (PACKAGE_ROOT / "scripts" / "spawn_or_replace_model.py").read_text()
        self.assertIn("GetWorldProperties", spawn_source)
        self.assertIn("if model_name in world.model_names", spawn_source)
        self.assertIn("Remove it from the world file", spawn_source)
        self.assertNotIn("DeleteModel", spawn_source)

    def test_cli_supports_independent_linear_and_angular_experiment_speeds(self):
        source = CLI_PATH.read_text()
        self.assertIn('ACTION_NAMES = ("run", *SERVICE_NAMES)', source)
        self.assertIn('args.action == "run"', source)
        self.assertIn('f"{namespace}/linear_speed_mm_s"', source)
        self.assertIn('f"{namespace}/angular_speed_rad_s"', source)
        self.assertNotIn("--speed-scale", source)
        self.assertIn("def wait_for_plan(", source)
        self.assertIn("signal.signal(signal.SIGINT, request_stop)", source)
        self.assertIn('call_service(namespace, "stop"', source)
        self.assertIn('status.get("state") == "STOPPED"', source)
        self.assertIn('if state == "COMPLETED":', source)
        self.assertGreaterEqual(source.count("return 130"), 3)


if __name__ == "__main__":
    unittest.main()
