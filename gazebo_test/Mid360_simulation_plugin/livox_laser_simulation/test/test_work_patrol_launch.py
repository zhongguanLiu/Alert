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


class Mid360FastlioLaunchWorkPatrolTests(unittest.TestCase):
    def test_mid360_launch_does_not_expose_work_patrol_args_anymore(self):
        tree = ET.parse(MID360_LAUNCH_PATH)
        root = tree.getroot()
        arg_names = [elem.attrib.get("name") for elem in root.findall("arg")]
        self.assertNotIn("work_patrol", arg_names)
        self.assertNotIn("work_patrol_radius", arg_names)
        self.assertNotIn("work_patrol_linear_speed", arg_names)
        self.assertNotIn("work_patrol_start_delay", arg_names)

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
        removed_arg_prefix = "model_" "03_"

        self.assertEqual(args["control_mode"].get("default"), "single")
        self.assertEqual(args["model_01_name"].get("default"), "model_01")
        self.assertEqual(args["model_02_name"].get("default"), "model_02")
        self.assertEqual(args["model_01_command_frame"].get("default"), "world")
        self.assertEqual(args["model_02_command_frame"].get("default"), "world")
        self.assertEqual(args["model_01_linear_y"].get("default"), "$(arg linear_y)")
        self.assertEqual(args["model_02_linear_x"].get("default"), "$(arg linear_y)")
        self.assertFalse(
            any(name.startswith(removed_arg_prefix) for name in args),
        )

    def test_debris_launch_includes_multi_motion_nodes(self):
        tree = ET.parse(DEBRIS_LAUNCH_PATH)
        root = tree.getroot()

        nodes = {node.attrib.get("name"): node for node in root.findall("node")}
        removed_node_name = "model_" "03_node_name"
        self.assertEqual(
            nodes["$(arg node_name)"].attrib.get("if"),
            "$(eval arg('control_mode') == 'single')",
        )

        model_01_node = nodes["$(arg model_01_node_name)"]
        model_02_node = nodes["$(arg model_02_node_name)"]
        self.assertNotIn("$(arg " + removed_node_name + ")", nodes)

        for node in (model_01_node, model_02_node):
            self.assertEqual(node.attrib.get("type"), "model_motion_controller.py")
            self.assertEqual(node.attrib.get("if"), "$(eval arg('control_mode') == 'multi')")

        model_01_params = {
            elem.attrib.get("name"): elem.attrib.get("value")
            for elem in model_01_node.findall("param")
        }
        model_02_params = {
            elem.attrib.get("name"): elem.attrib.get("value")
            for elem in model_02_node.findall("param")
        }

        self.assertEqual(model_01_params["model_name"], "$(arg model_01_name)")
        self.assertEqual(model_01_params["command_frame"], "$(arg model_01_command_frame)")
        self.assertEqual(model_01_params["linear_y"], "$(arg model_01_linear_y)")
        self.assertEqual(model_01_params["angular_y_deg"], "0.0")

        self.assertEqual(model_02_params["model_name"], "$(arg model_02_name)")
        self.assertEqual(model_02_params["command_frame"], "$(arg model_02_command_frame)")
        self.assertEqual(model_02_params["linear_x"], "$(arg model_02_linear_x)")
        self.assertEqual(model_02_params["angular_y_deg"], "0.0")


if __name__ == "__main__":
    unittest.main()
