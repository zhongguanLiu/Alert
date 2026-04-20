# Author: zgliu@cumt.edu.cn
# Affiliation: China University of Mining and Technology
# Open-source release date: 2026-04-20
from __future__ import annotations

import pathlib
import unittest
import xml.etree.ElementTree as ET


LAUNCH_PATH = (
    pathlib.Path(__file__).resolve().parents[1]
    / "launch"
    / "deform_monitor_v2_sim.launch"
)


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


if __name__ == "__main__":
    unittest.main()
