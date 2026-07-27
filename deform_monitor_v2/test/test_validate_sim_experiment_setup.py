import importlib.util
import pathlib
import tempfile
import unittest

import yaml


SCRIPT_PATH = (
    pathlib.Path(__file__).resolve().parents[1]
    / "scripts"
    / "validate_sim_experiment_setup.py"
)


def load_module_if_exists():
    if not SCRIPT_PATH.exists():
        return None
    spec = importlib.util.spec_from_file_location(
        "validate_sim_experiment_setup_under_test", SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_world(path, second_id=32, state_x=0.0, first_id=31):
    path.write_text(
        f"""<sdf version='1.6'>
  <world name='default'>
    <state world_name='default'>
      <model name='moving_panel'>
        <pose>{state_x} 0 0 0 0 0</pose>
        <link name='body'><pose>{state_x} 0 0 0 0 0</pose></link>
      </model>
      <model name='static_wedge'>
        <pose>2 0 0 0 0 0</pose>
        <link name='body'><pose>2 0 0 0 0 0</pose></link>
      </model>
    </state>
    <model name='moving_panel'>
      <pose>0 0 0 0 0 0</pose>
      <static>1</static>
      <link name='body'>
        <collision name='surface'><laser_retro>{first_id}</laser_retro></collision>
      </link>
      <link name='ground_truth_face_1'/>
      <link name='ground_truth_face_2'/>
      <link name='ground_truth_face_3'/>
      <joint name='truth_1_fixed' type='fixed'>
        <parent>body</parent><child>ground_truth_face_1</child>
      </joint>
      <joint name='truth_2_fixed' type='fixed'>
        <parent>body</parent><child>ground_truth_face_2</child>
      </joint>
      <joint name='truth_3_fixed' type='fixed'>
        <parent>body</parent><child>ground_truth_face_3</child>
      </joint>
    </model>
    <model name='static_wedge'>
      <pose>2 0 0 0 0 0</pose>
      <static>1</static>
      <link name='body'>
        <collision name='surface'><laser_retro>{second_id}</laser_retro></collision>
      </link>
    </model>
  </world>
</sdf>"""
    )


def recorder_config():
    return {
        "scenario_id": "fixed_world_case_01",
        "ego_model_name": "sensor_platform",
        "object_id_catalog": {31: "moving_panel", 32: "static_wedge"},
        "experiment_factors": {
            "scene_id": "collapse_world_v3",
            "moving_object_quantity": 1,
            "scene_object_quantity": 2,
            "platform_condition": "static",
            "slam_pipeline": "fast_lio",
            "point_cloud_setting": "nominal",
            "repeat_index": 1,
        },
        "object_metadata": {
            "moving_panel": {
                "shape": "rectangular_panel",
                "size_class": "large",
                "motion_profile": "accelerating_rotation",
                "motion_direction": "positive_yaw",
                "visibility_condition": "partial_occlusion",
            }
        },
    }


class ValidateSimExperimentSetupTests(unittest.TestCase):
    def test_valid_fixed_world_and_config_pass(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        with tempfile.TemporaryDirectory(prefix="setup_validation_") as temp_dir:
            root = pathlib.Path(temp_dir)
            world_path = root / "fixed.world"
            config_path = root / "recorder.yaml"
            write_world(world_path)
            config_path.write_text(yaml.safe_dump(recorder_config()))

            result = module.validate_setup(world_path, config_path)

            self.assertEqual(result["status"], "PASS")
            self.assertEqual(result["errors"], [])
            self.assertEqual(result["world_object_ids"], {31: "moving_panel", 32: "static_wedge"})

    def test_duplicate_id_full_occlusion_and_stale_state_fail(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        with tempfile.TemporaryDirectory(prefix="setup_validation_") as temp_dir:
            root = pathlib.Path(temp_dir)
            world_path = root / "invalid.world"
            config_path = root / "recorder.yaml"
            write_world(world_path, second_id=31, state_x=1000.0)
            config = recorder_config()
            config["object_metadata"]["moving_panel"][
                "visibility_condition"
            ] = "fully_occluded"
            config_path.write_text(yaml.safe_dump(config))

            result = module.validate_setup(world_path, config_path)

            self.assertEqual(result["status"], "FAIL")
            joined = ";".join(result["errors"])
            self.assertIn("object_id_reused", joined)
            self.assertIn("fully_occluded_object", joined)
            self.assertIn("state_pose_mismatch", joined)

    def test_invalid_laser_retro_is_reported_without_crashing(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        with tempfile.TemporaryDirectory(prefix="setup_validation_") as temp_dir:
            root = pathlib.Path(temp_dir)
            world_path = root / "invalid_retro.world"
            config_path = root / "recorder.yaml"
            write_world(world_path, first_id="not_an_object_id")
            config_path.write_text(yaml.safe_dump(recorder_config()))

            result = module.validate_setup(world_path, config_path)

            self.assertEqual(result["status"], "FAIL")
            self.assertTrue(
                any(
                    error.startswith("invalid_or_missing_laser_retro:moving_panel")
                    for error in result["errors"]
                )
            )

    def test_dynamic_environment_model_is_rejected(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        with tempfile.TemporaryDirectory(prefix="setup_validation_") as temp_dir:
            root = pathlib.Path(temp_dir)
            world_path = root / "dynamic.world"
            config_path = root / "recorder.yaml"
            write_world(world_path)
            payload = world_path.read_text().replace(
                "<model name='moving_panel'>\n      <pose>0 0 0 0 0 0</pose>\n      <static>1</static>",
                "<model name='moving_panel'>\n      <pose>0 0 0 0 0 0</pose>\n      <static>0</static>",
            )
            world_path.write_text(payload)
            config_path.write_text(yaml.safe_dump(recorder_config()))

            result = module.validate_setup(world_path, config_path)

            self.assertEqual(result["status"], "FAIL")
            self.assertIn(
                "environment_model_not_static:moving_panel",
                result["errors"],
            )

    def test_dynamic_visual_only_environment_model_is_rejected(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        with tempfile.TemporaryDirectory(prefix="setup_validation_") as temp_dir:
            root = pathlib.Path(temp_dir)
            world_path = root / "dynamic_visual.world"
            config_path = root / "recorder.yaml"
            write_world(world_path)
            payload = world_path.read_text().replace(
                "  </world>",
                """    <model name='visual_only_marker'>
      <static>0</static>
      <link name='marker'><visual name='marker_visual'/></link>
    </model>
  </world>""",
            )
            world_path.write_text(payload)
            config_path.write_text(yaml.safe_dump(recorder_config()))

            result = module.validate_setup(world_path, config_path)

            self.assertEqual(result["status"], "FAIL")
            self.assertIn(
                "environment_model_not_static:visual_only_marker",
                result["errors"],
            )

    def test_unresolved_environment_include_is_reported_for_strict_preflight(self):
        module = load_module_if_exists()
        with tempfile.TemporaryDirectory(prefix="setup_validation_") as temp_dir:
            root = pathlib.Path(temp_dir)
            world_path = root / "included.world"
            config_path = root / "recorder.yaml"
            write_world(world_path)
            payload = world_path.read_text().replace(
                "  </world>",
                "    <include><uri>model://unknown_scene_asset</uri></include>\n  </world>",
            )
            world_path.write_text(payload)
            config_path.write_text(yaml.safe_dump(recorder_config()))

            result = module.validate_setup(world_path, config_path)

            self.assertEqual(result["status"], "WARN")
            self.assertIn(
                "unverified_environment_include:model://unknown_scene_asset",
                result["warnings"],
            )

    def test_missing_model_pose_uses_sdf_zero_default(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        with tempfile.TemporaryDirectory(prefix="setup_validation_") as temp_dir:
            root = pathlib.Path(temp_dir)
            world_path = root / "default_pose.world"
            config_path = root / "recorder.yaml"
            write_world(world_path)
            payload = world_path.read_text()
            payload = payload.replace(
                "<model name='moving_panel'>\n      <pose>0 0 0 0 0 0</pose>",
                "<model name='moving_panel'>",
                1,
            )
            world_path.write_text(payload)
            config_path.write_text(yaml.safe_dump(recorder_config()))

            result = module.validate_setup(world_path, config_path)

            self.assertEqual(result["status"], "PASS")

    def test_surface_truth_links_must_be_fixed_and_collision_free(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        with tempfile.TemporaryDirectory(prefix="setup_validation_") as temp_dir:
            root = pathlib.Path(temp_dir)
            world_path = root / "invalid_truth.world"
            config_path = root / "recorder.yaml"
            write_world(world_path)
            payload = world_path.read_text()
            payload = payload.replace(
                "<link name='ground_truth_face_1'/>",
                "<link name='ground_truth_face_1'><collision name='bad'/></link>",
            )
            payload = payload.replace(
                "<joint name='truth_2_fixed' type='fixed'>\n"
                "        <parent>body</parent><child>ground_truth_face_2</child>\n"
                "      </joint>\n",
                "",
            )
            world_path.write_text(payload)
            config_path.write_text(yaml.safe_dump(recorder_config()))

            result = module.validate_setup(world_path, config_path)

            self.assertEqual(result["status"], "FAIL")
            joined = ";".join(result["errors"])
            self.assertIn("surface_truth_link_has_collision", joined)
            self.assertIn("surface_truth_link_not_fixed:moving_panel:ground_truth_face_2", joined)


if __name__ == "__main__":
    unittest.main()
