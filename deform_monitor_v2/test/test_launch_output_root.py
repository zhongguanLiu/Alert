from __future__ import annotations

import pathlib
import unittest
import xml.etree.ElementTree as ET


LAUNCH_PATH = (
    pathlib.Path(__file__).resolve().parents[1]
    / "launch"
    / "deform_monitor_v2_sim.launch"
)
DYNAMIC_LAUNCH_PATH = LAUNCH_PATH.with_name("deform_monitor_v2_sim_dynamic.launch")


class DeformMonitorSimLaunchTests(unittest.TestCase):
    def test_launch_exposes_sim_experiment_output_root_arg_and_wires_recorder_param(self):
        tree = ET.parse(LAUNCH_PATH)
        root = tree.getroot()

        arg_names = [elem.attrib.get("name") for elem in root.findall("arg")]
        self.assertIn("sim_experiment_output_root", arg_names)

        recorder_node = None
        for node in root.findall("node"):
            if node.attrib.get("name") == "sim_experiment_recorder":
                recorder_node = node
                break

        self.assertIsNotNone(recorder_node)
        param_pairs = {
            elem.attrib.get("name"): elem.attrib.get("value")
            for elem in recorder_node.findall("param")
        }
        self.assertEqual(
            param_pairs.get("output_root"),
            "$(arg sim_experiment_output_root)",
        )

    def test_launch_exposes_final_alarm_risk_threshold_as_config_override(self):
        tree = ET.parse(LAUNCH_PATH)
        root = tree.getroot()

        args = {elem.attrib.get("name"): elem.attrib for elem in root.findall("arg")}
        self.assertIn("alarm_mean_risk_threshold", args)
        self.assertEqual(args["alarm_mean_risk_threshold"].get("default"), "0.55")

        monitor_node = next(
            node for node in root.findall("node")
            if node.attrib.get("name") == "deform_monitor_v2"
        )
        params = {
            elem.attrib.get("name"): elem.attrib.get("value")
            for elem in monitor_node.findall("param")
        }
        self.assertEqual(
            params.get("deform_monitor/persistent_risk/min_confirmed_mean_risk"),
            "$(arg alarm_mean_risk_threshold)",
        )

    def test_dynamic_launch_uses_the_same_final_alarm_threshold_override(self):
        tree = ET.parse(DYNAMIC_LAUNCH_PATH)
        root = tree.getroot()

        args = {elem.attrib.get("name"): elem.attrib for elem in root.findall("arg")}
        self.assertEqual(args["alarm_mean_risk_threshold"].get("default"), "0.55")
        monitor_node = next(
            node for node in root.findall("node")
            if node.attrib.get("name") == "deform_monitor_v2"
        )
        params = {
            elem.attrib.get("name"): elem.attrib.get("value")
            for elem in monitor_node.findall("param")
        }
        self.assertEqual(
            params.get("deform_monitor/persistent_risk/min_confirmed_mean_risk"),
            "$(arg alarm_mean_risk_threshold)",
        )


if __name__ == "__main__":
    unittest.main()
