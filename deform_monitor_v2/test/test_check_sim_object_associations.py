import importlib.util
import pathlib
import unittest


SCRIPT_PATH = (
    pathlib.Path(__file__).resolve().parents[1]
    / "scripts"
    / "check_sim_object_associations.py"
)


def load_module_if_exists():
    if not SCRIPT_PATH.exists():
        return None
    spec = importlib.util.spec_from_file_location(
        "check_sim_object_associations_under_test", SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CheckSimObjectAssociationsTests(unittest.TestCase):
    def test_report_counts_each_pipeline_stage_and_anchor_type(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        accumulator = module.AssociationAccumulator({1: "model_01", 2: "model_02"})
        accumulator.add_raw_reflectivities([1, 1, 2, 0, 255])
        accumulator.add_fast_lio_intensities([1.0, 2.0, 0.0])
        accumulator.add_observation_stats(
            total_point_count=3,
            valid_label_point_count=2,
            invalid_label_point_count=1,
            object_point_counts={1: 1, 2: 1},
        )
        accumulator.set_anchor_snapshot(
            reference_epoch=7,
            anchors=[
                {
                    "object_id": 1,
                    "object_id_valid": True,
                    "anchor_type": 0,
                    "object_id_confidence": 0.9,
                    "object_association_state": 1,
                    "reference_origin": 0,
                },
                {
                    "object_id": 1,
                    "object_id_valid": True,
                    "anchor_type": 1,
                    "object_id_confidence": 0.7,
                    "object_association_state": 2,
                    "reference_origin": 1,
                },
                {
                    "object_id": 2,
                    "object_id_valid": True,
                    "anchor_type": 2,
                    "object_id_confidence": 0.8,
                    "object_association_state": 1,
                    "reference_origin": 0,
                },
            ],
        )

        report = accumulator.build_report(
            expected_visible_ids={1, 2}, expected_evaluable_ids={1, 2}
        )

        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["objects"]["1"]["raw_livox_point_count"], 2)
        self.assertEqual(report["objects"]["2"]["fast_lio_point_count"], 1)
        self.assertEqual(report["objects"]["1"]["anchor_types"]["PLANE"], 1)
        self.assertEqual(report["objects"]["1"]["anchor_types"]["EDGE"], 1)
        self.assertEqual(report["objects"]["2"]["anchor_types"]["BAND"], 1)
        self.assertEqual(report["objects"]["1"]["association"]["mismatch"], 1)
        self.assertAlmostEqual(
            report["objects"]["1"]["object_id_confidence"]["mean"], 0.8
        )
        self.assertEqual(report["anchors"]["reference_epoch"], 7)
        self.assertEqual(report["anchors"]["initial_count"], 2)
        self.assertEqual(report["anchors"]["incremental_count"], 1)
        self.assertEqual(report["labels"]["valid_point_count"], 2)
        self.assertAlmostEqual(report["labels"]["valid_ratio"], 2.0 / 3.0)

    def test_expected_visible_id_must_propagate_through_point_clouds_and_alert(self):
        module = load_module_if_exists()
        accumulator = module.AssociationAccumulator({1: "model_01"})
        accumulator.add_raw_reflectivities([1])
        accumulator.add_fast_lio_intensities([])
        accumulator.add_observation_stats(1, 0, 1, {})
        accumulator.set_anchor_snapshot(1, [])

        report = accumulator.build_report(
            expected_visible_ids={1}, expected_evaluable_ids={1}
        )

        self.assertEqual(report["status"], "FAIL")
        self.assertTrue(any("fast_lio" in error for error in report["errors"]))
        self.assertTrue(any("alert" in error for error in report["errors"]))
        self.assertTrue(any("anchor" in error for error in report["errors"]))

    def test_visible_object_without_anchors_is_not_failed_unless_declared_evaluable(self):
        module = load_module_if_exists()
        accumulator = module.AssociationAccumulator({2: "model_02"})
        accumulator.add_raw_reflectivities([2])
        accumulator.add_fast_lio_intensities([2.0])
        accumulator.add_observation_stats(1, 1, 0, {2: 1})
        accumulator.set_anchor_snapshot(1, [])

        report = accumulator.build_report(
            expected_visible_ids={2}, expected_evaluable_ids=set()
        )

        self.assertNotEqual(report["status"], "FAIL")
        self.assertEqual(report["errors"], [])


if __name__ == "__main__":
    unittest.main()
