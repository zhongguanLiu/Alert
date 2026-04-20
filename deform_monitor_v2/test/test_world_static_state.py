# Author: zgliu@cumt.edu.cn
# Affiliation: China University of Mining and Technology
# Open-source release date: 2026-04-20
"""Static world-file consistency tests.

The zero velocities, accelerations, and wrenches asserted here come from the
expected definition of static Gazebo models in the checked-in world file. These
assertions verify file consistency only; they are not reported experiment data.
"""

import pathlib
import unittest
import xml.etree.ElementTree as ET


WORLD_PATH = (
    pathlib.Path(__file__).resolve().parents[2]
    / "gazebo_test"
    / "Mid360_simulation_plugin"
    / "livox_laser_simulation"
    / "worlds"
    / "tracked_mid360_fastlio_collapse_microdeform.world"
)


TARGET_MODELS = [
    "debris_panel_01_clone",
    "debris_panel_01_clone_clone",
    "debris_panel_04_clone_0",
    "debris_panel_04_clone_1",
]


def parse_pose(text):
    return [float(part) for part in (text or "").split()]


def parse_vector6(text):
    values = [float(part) for part in (text or "").split()]
    return values if len(values) == 6 else None


class WorldStaticStateTests(unittest.TestCase):
    def test_target_panels_have_static_model_definitions(self):
        root = ET.parse(WORLD_PATH).getroot()
        world = root.find("world")
        self.assertIsNotNone(world)

        for model_name in TARGET_MODELS:
            model = next(
                (item for item in world.findall("model") if item.attrib.get("name") == model_name),
                None,
            )
            self.assertIsNotNone(model, model_name)
            self.assertEqual((model.findtext("static") or "").strip(), "1", model_name)

    def test_target_panels_state_pose_matches_static_definition_and_has_zero_dynamics(self):
        root = ET.parse(WORLD_PATH).getroot()
        world = root.find("world")
        state = world.find("state")
        self.assertIsNotNone(state)

        state_models = {
            item.attrib.get("name"): item for item in state.findall("model")
        }
        world_models = {
            item.attrib.get("name"): item for item in world.findall("model")
        }

        for model_name in TARGET_MODELS:
            state_model = state_models.get(model_name)
            world_model = world_models.get(model_name)
            self.assertIsNotNone(state_model, model_name)
            self.assertIsNotNone(world_model, model_name)

            state_pose = parse_pose(state_model.findtext("pose"))
            world_pose = parse_pose(world_model.findtext("pose"))
            self.assertEqual(len(state_pose), 6, model_name)
            self.assertEqual(len(world_pose), 6, model_name)
            for lhs, rhs in zip(state_pose, world_pose):
                self.assertAlmostEqual(lhs, rhs, places=6, msg=model_name)

            link = state_model.find("link")
            self.assertIsNotNone(link, model_name)
            link_pose = parse_pose(link.findtext("pose"))
            self.assertEqual(len(link_pose), 6, model_name)
            for lhs, rhs in zip(link_pose, world_pose):
                self.assertAlmostEqual(lhs, rhs, places=6, msg=model_name)

            velocity = parse_vector6(link.findtext("velocity"))
            acceleration = parse_vector6(link.findtext("acceleration"))
            wrench = parse_vector6(link.findtext("wrench"))
            self.assertEqual(velocity, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], model_name)
            self.assertEqual(acceleration, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], model_name)
            self.assertEqual(wrench, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], model_name)


if __name__ == "__main__":
    unittest.main()
