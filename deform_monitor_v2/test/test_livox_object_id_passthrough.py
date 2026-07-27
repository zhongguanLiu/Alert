import pathlib
import re
import unittest
import xml.etree.ElementTree as ET

import yaml


PLUGIN_SOURCE = (
    pathlib.Path(__file__).resolve().parents[2]
    / "gazebo_test"
    / "Mid360_simulation_plugin"
    / "livox_laser_simulation"
    / "src"
    / "livox_points_plugin.cpp"
)
RECORDER_CONFIG = pathlib.Path(__file__).resolve().parents[1] / "config" / (
    "sim_experiment_recorder.yaml"
)
WORLD_FILE = (
    pathlib.Path(__file__).resolve().parents[2]
    / "gazebo_test"
    / "Mid360_simulation_plugin"
    / "livox_laser_simulation"
    / "worlds"
    / "tracked_mid360_fastlio_collapse_microdeform_sim.world"
)

EXPECTED_OBJECT_ID_CATALOG = {
    "1": "l_shape_target",
    "2": "model_01",
    "3": "model_02",
    "4": "table",
    "5": "table_marble",
    "6": "person_walking",
    "7": "table_clone",
    "8": "cardboard_box",
    "9": "lzg_bianwood",
    "10": "bookshelf",
    "11": "frc2016_drawbridge",
    "12": "lzg_grey_wall",
    "13": "wooden_case_metal_peg",
    "14": "cardboard_box_0",
    "15": "bookshelf_0",
    "16": "lzg_three_zhuzi",
    "17": "cardboard_box_clone",
    "18": "cardboard_box_1",
    "19": "lzg_bianwood_clone",
    "20": "lzg_bianwood_clone_0",
    "21": "lzg_bianwood_clone_1",
    "22": "grey_wall",
    "23": "grey_wall_0",
    "24": "ground_plane",
    "25": "lzg_wall",
}


class LivoxObjectIdPassthroughTests(unittest.TestCase):
    def test_custom_message_reflectivity_uses_hit_laser_retro(self):
        source = PLUGIN_SOURCE.read_text()
        match = re.search(
            r"void LivoxPointsPlugin::PublishLivoxROSDriverCustomMsg\(.*?\n\}",
            source,
            flags=re.DOTALL,
        )
        self.assertIsNotNone(match)
        function_source = match.group(0)

        self.assertIn("auto intensity = rayShape->GetRetro(pair.first);", function_source)
        self.assertRegex(function_source, r"pt\.reflectivity\s*=\s*[^;]*intensity[^;]*;")
        self.assertNotRegex(function_source, r"pt\.reflectivity\s*=\s*100\s*;")

    def test_recorder_catalog_uses_string_keys_and_matches_the_world(self):
        configuration = yaml.safe_load(RECORDER_CONFIG.read_text())
        catalog = configuration.get("object_id_catalog", {})
        self.assertTrue(all(isinstance(key, str) for key in catalog))
        self.assertEqual(catalog, EXPECTED_OBJECT_ID_CATALOG)

        world = ET.parse(WORLD_FILE).getroot().find("world")
        self.assertIsNotNone(world)
        world_catalog = {}
        for model in world.findall("model"):
            model_name = model.attrib["name"]
            collision_ids = {
                int(float(collision.findtext("laser_retro")))
                for collision in model.findall(".//collision")
            }
            self.assertEqual(len(collision_ids), 1, model_name)
            object_id = collision_ids.pop()
            self.assertGreater(object_id, 0, model_name)
            self.assertNotIn(str(object_id), world_catalog)
            world_catalog[str(object_id)] = model_name
        self.assertEqual(world_catalog, catalog)


if __name__ == "__main__":
    unittest.main()
