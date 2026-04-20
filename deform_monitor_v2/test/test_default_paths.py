# Author: zgliu@cumt.edu.cn
# Affiliation: China University of Mining and Technology
# Open-source release date: 2026-04-20
from __future__ import annotations

import importlib.util
import pathlib
import unittest
import xml.etree.ElementTree as ET

import yaml


SIM_LAUNCH_PATH = (
    pathlib.Path(__file__).resolve().parents[1]
    / "launch"
    / "deform_monitor_v2_sim.launch"
)
SIM_DYNAMIC_LAUNCH_PATH = (
    pathlib.Path(__file__).resolve().parents[1]
    / "launch"
    / "deform_monitor_v2_sim_dynamic.launch"
)
REAL_LAUNCH_PATH = (
    pathlib.Path(__file__).resolve().parents[1]
    / "launch"
    / "deform_monitor_v2_real.launch"
)
SIM_CONFIG_PATH = (
    pathlib.Path(__file__).resolve().parents[1]
    / "config"
    / "sim_experiment_recorder.yaml"
)
SIM_RECORDER_PATH = (
    pathlib.Path(__file__).resolve().parents[1]
    / "scripts"
    / "sim_experiment_recorder.py"
)
REAL_RECORDER_PATH = (
    pathlib.Path(__file__).resolve().parents[1]
    / "scripts"
    / "real_experiment_recorder.py"
)
ANALYZE_SIM_RUN_PATH = (
    pathlib.Path(__file__).resolve().parents[1]
    / "scripts"
    / "analyze_sim_run.py"
)


def _load_module(script_path: pathlib.Path, name: str):
    spec = importlib.util.spec_from_file_location(name, script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DefaultPathTests(unittest.TestCase):
    def test_sim_launch_defaults_to_standard_user_output_dir(self):
        tree = ET.parse(SIM_LAUNCH_PATH)
        root = tree.getroot()
        args = {elem.attrib.get("name"): elem.attrib.get("default") for elem in root.findall("arg")}
        self.assertEqual(args.get("sim_experiment_output_root"), "$(env HOME)/.ros/alert/output")

    def test_sim_dynamic_launch_defaults_to_standard_user_output_dir(self):
        tree = ET.parse(SIM_DYNAMIC_LAUNCH_PATH)
        root = tree.getroot()
        args = {elem.attrib.get("name"): elem.attrib.get("default") for elem in root.findall("arg")}
        self.assertEqual(args.get("sim_experiment_output_root"), "$(env HOME)/.ros/alert/output")

    def test_real_launch_defaults_to_standard_user_output_dir(self):
        tree = ET.parse(REAL_LAUNCH_PATH)
        root = tree.getroot()
        args = {elem.attrib.get("name"): elem.attrib.get("default") for elem in root.findall("arg")}
        self.assertEqual(args.get("real_experiment_output_root"), "$(env HOME)/.ros/alert/real_output")

    def test_sim_recorder_yaml_uses_expanduser_style_default(self):
        payload = yaml.safe_load(SIM_CONFIG_PATH.read_text())
        self.assertEqual(payload.get("output_root"), "~/.ros/alert/output")

    def test_sim_recorder_default_output_root_is_standard_user_dir(self):
        module = _load_module(SIM_RECORDER_PATH, "sim_experiment_recorder")
        self.assertEqual(
            module.DEFAULT_OUTPUT_ROOT,
            pathlib.Path.home() / ".ros" / "alert" / "output",
        )

    def test_real_recorder_default_output_root_is_standard_user_dir(self):
        module = _load_module(REAL_RECORDER_PATH, "real_experiment_recorder")
        self.assertEqual(
            module.DEFAULT_OUTPUT_ROOT,
            pathlib.Path.home() / ".ros" / "alert" / "real_output",
        )

    def test_analyze_sim_run_default_output_root_matches_runtime_default(self):
        module = _load_module(ANALYZE_SIM_RUN_PATH, "analyze_sim_run")
        parser = module.parse_args
        original_parse_args = None
        try:
            import sys
            argv = sys.argv
            sys.argv = [str(ANALYZE_SIM_RUN_PATH)]
            args = parser()
        finally:
            sys.argv = argv
        self.assertEqual(
            args.output_root,
            pathlib.Path.home() / ".ros" / "alert" / "output",
        )


if __name__ == "__main__":
    unittest.main()
