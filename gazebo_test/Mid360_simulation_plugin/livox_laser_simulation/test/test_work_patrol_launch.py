from __future__ import annotations

import pathlib
import unittest
import xml.etree.ElementTree as ET


MID360_LAUNCH_PATH = (
    pathlib.Path(__file__).resolve().parents[1]
    / "launch"
    / "mid360_fastlio.launch"
)
DEBRIS_LAUNCH_PATH = (
    pathlib.Path(__file__).resolve().parents[1]
    / "launch"
    / "debris_block_02_motion.launch"
)
FAST_LIO_ROOT = pathlib.Path(__file__).resolve().parents[4] / "FAST_LIO"
FAST_LIO_LAUNCH_PATH = FAST_LIO_ROOT / "launch" / "mapping_mid360.launch"
FAST_LIO_SOURCE_PATH = FAST_LIO_ROOT / "src" / "laserMapping.cpp"


class Mid360FastlioLaunchWorkPatrolTests(unittest.TestCase):
    def test_mid360_launch_does_not_expose_work_patrol_args_anymore(self):
        tree = ET.parse(MID360_LAUNCH_PATH)
        root = tree.getroot()
        arg_names = [elem.attrib.get("name") for elem in root.findall("arg")]
        self.assertNotIn("work_patrol", arg_names)
        self.assertNotIn("work_patrol_radius", arg_names)
        self.assertNotIn("work_patrol_linear_speed", arg_names)
        self.assertNotIn("work_patrol_start_delay", arg_names)

    def test_mid360_launch_spawns_robot_missing_from_default_world(self):
        root = ET.parse(MID360_LAUNCH_PATH).getroot()
        args = {elem.attrib.get("name"): elem.attrib for elem in root.findall("arg")}
        self.assertEqual(args["spawn_robot"].get("default"), "true")

        spawn_node = next(
            node
            for node in root.findall("node")
            if node.attrib.get("name") == "spawn_mid360_fastlio"
        )
        self.assertEqual(spawn_node.attrib.get("if"), "$(arg spawn_robot)")

    def test_mid360_launch_keeps_keyboard_gate_without_work_patrol_dependency(self):
        tree = ET.parse(MID360_LAUNCH_PATH)
        root = tree.getroot()

        keyboard_node = None
        for node in root.findall("node"):
            if node.attrib.get("name") == "mid360_keyboard_drive":
                keyboard_node = node

        self.assertIsNotNone(keyboard_node)
        self.assertEqual(
            keyboard_node.attrib.get("if"),
            "$(eval arg('keyboard_teleop') and not arg('auto_drive'))",
        )

    def test_mid360_launch_does_not_claim_to_throttle_gazebo_state_topics(self):
        tree = ET.parse(MID360_LAUNCH_PATH)
        root = tree.getroot()
        parameter_names = {
            element.attrib.get("name") for element in root.findall("param")
        }
        self.assertNotIn("gazebo/publish_period", parameter_names)

    def test_fast_lio_formal_launch_disables_detailed_runtime_logs(self):
        root = ET.parse(FAST_LIO_LAUNCH_PATH).getroot()
        args = {element.attrib["name"]: element.attrib for element in root.findall("arg")}
        params = {element.attrib["name"]: element.attrib for element in root.findall("param")}
        self.assertEqual(args["runtime_pos_log_enable"].get("default"), "false")
        self.assertEqual(
            args["runtime_performance_log_enable"].get("default"), "false"
        )
        self.assertEqual(
            params["runtime_pos_log_enable"].get("value"),
            "$(arg runtime_pos_log_enable)",
        )
        self.assertEqual(
            params["runtime_performance_log_enable"].get("value"),
            "$(arg runtime_performance_log_enable)",
        )

        source = FAST_LIO_SOURCE_PATH.read_text()
        self.assertIn('nh.param<bool>("runtime_performance_log_enable"', source)
        self.assertIn("if (runtime_performance_log)", source)


class DebrisMotionLaunchWorkPatrolTests(unittest.TestCase):
    def test_debris_launch_exposes_work_patrol_args(self):
        tree = ET.parse(DEBRIS_LAUNCH_PATH)
        root = tree.getroot()
        args = {elem.attrib.get("name"): elem.attrib for elem in root.findall("arg")}
        arg_names = list(args.keys())
        self.assertIn("enable_work_patrol", arg_names)
        self.assertIn("work_patrol_radius", arg_names)
        self.assertIn("work_patrol_linear_speed", arg_names)
        self.assertIn("work_patrol_start_delay", arg_names)
        self.assertEqual(args["work_patrol_radius"].get("default"), "1.0")

    def test_debris_launch_includes_work_patrol_node(self):
        tree = ET.parse(DEBRIS_LAUNCH_PATH)
        root = tree.getroot()
        args = {elem.attrib.get("name"): elem.attrib for elem in root.findall("arg")}

        patrol_node = None
        for node in root.findall("node"):
            if node.attrib.get("name") == "mid360_work_patrol":
                patrol_node = node
                break

        self.assertIsNotNone(patrol_node)
        self.assertEqual(patrol_node.attrib.get("type"), "work_patrol_cmd.py")
        self.assertEqual(patrol_node.attrib.get("if"), "$(arg enable_work_patrol)")

        patrol_params = {
            elem.attrib.get("name"): elem.attrib.get("value")
            for elem in patrol_node.findall("param")
        }
        self.assertEqual(patrol_params.get("radius"), "$(arg work_patrol_radius)")
        self.assertEqual(
            patrol_params.get("linear_speed"), "$(arg work_patrol_linear_speed)"
        )
        self.assertEqual(
            patrol_params.get("start_delay"), "$(arg work_patrol_start_delay)"
        )
        self.assertEqual(args["work_patrol_linear_speed"].get("default"), "0.2")

    def test_debris_launch_exposes_multi_control_args(self):
        tree = ET.parse(DEBRIS_LAUNCH_PATH)
        root = tree.getroot()
        args = {elem.attrib.get("name"): elem.attrib for elem in root.findall("arg")}

        self.assertEqual(args["control_mode"].get("default"), "single")
        for index in range(1, 5):
            prefix = f"model_{index:02d}"
            with self.subTest(model=prefix):
                self.assertEqual(args[f"{prefix}_name"].get("default"), prefix)
                self.assertEqual(
                    args[f"{prefix}_node_name"].get("default"),
                    f"{prefix}_motion",
                )
                self.assertEqual(
                    args[f"{prefix}_command_frame"].get("default"), "world"
                )
                self.assertEqual(args[f"{prefix}_enabled"].get("default"), "true")
                for component in ("x", "y", "z"):
                    self.assertEqual(
                        args[f"{prefix}_linear_{component}"].get("default"), "0.0"
                    )
                    self.assertEqual(
                        args[f"{prefix}_angular_{component}_deg"].get("default"),
                        "0.0",
                    )
                self.assertEqual(
                    args[f"{prefix}_start_delay"].get("default"),
                    "$(arg start_delay)",
                )
                self.assertEqual(
                    args[f"{prefix}_duration"].get("default"), "$(arg duration)"
                )
                self.assertEqual(
                    args[f"{prefix}_scenario_id"].get("default"),
                    "$(arg scenario_id)",
                )

    def test_debris_launch_includes_multi_motion_nodes(self):
        tree = ET.parse(DEBRIS_LAUNCH_PATH)
        root = tree.getroot()

        nodes = {node.attrib.get("name"): node for node in root.findall("node")}
        self.assertEqual(
            nodes["$(arg node_name)"].attrib.get("if"),
            "$(eval arg('control_mode') == 'single')",
        )

        for index in range(1, 5):
            prefix = f"model_{index:02d}"
            node = nodes[f"$(arg {prefix}_node_name)"]
            with self.subTest(model=prefix):
                self.assertEqual(node.attrib.get("type"), "model_motion_controller.py")
                self.assertEqual(
                    node.attrib.get("if"),
                    "$(eval arg('control_mode') == 'multi' and "
                    f"arg('{prefix}_enabled'))",
                )
                params = {
                    elem.attrib.get("name"): elem.attrib.get("value")
                    for elem in node.findall("param")
                }
                self.assertEqual(params["model_name"], f"$(arg {prefix}_name)")
                self.assertEqual(
                    params["command_frame"], f"$(arg {prefix}_command_frame)"
                )
                for component in ("x", "y", "z"):
                    self.assertEqual(
                        params[f"linear_{component}"],
                        f"$(arg {prefix}_linear_{component})",
                    )
                    self.assertEqual(
                        params[f"angular_{component}_deg"],
                        f"$(arg {prefix}_angular_{component}_deg)",
                    )
                self.assertEqual(
                    params["start_delay"], f"$(arg {prefix}_start_delay)"
                )
                self.assertEqual(params["duration"], f"$(arg {prefix}_duration)")
                self.assertEqual(
                    params["scenario_id"], f"$(arg {prefix}_scenario_id)"
                )


if __name__ == "__main__":
    unittest.main()
