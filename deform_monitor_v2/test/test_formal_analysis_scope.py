import importlib.util
import json
import pathlib
import tempfile
import unittest


SCRIPT_PATH = (
    pathlib.Path(__file__).resolve().parents[1]
    / "scripts"
    / "formal_analysis_scope.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location(
        "formal_analysis_scope_under_test", SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_protocol(run_dir, **overrides):
    actual = {
        "reference_epoch": 2,
        "trial_start_time": 10.0,
        "reference_initialized_at": 12.0,
        "actual_motion_start_time": 20.0,
        "hold_start_time": 50.0,
        "trial_end_time": 60.0,
    }
    actual.update(overrides.pop("actual", {}))
    payload = {
        "schema_version": 1,
        "scenario_id": "formal_case",
        "status": "PASS",
        "valid_for_analysis": True,
        "actual": actual,
    }
    payload.update(overrides)
    path = pathlib.Path(run_dir) / "meta" / "experiment_protocol.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(payload))


class FormalAnalysisScopeTests(unittest.TestCase):
    def test_loads_valid_protocol_and_classifies_non_overlapping_phases(self):
        module = load_module()
        with tempfile.TemporaryDirectory(prefix="formal_scope_") as temp_dir:
            write_protocol(temp_dir)
            scope = module.load_formal_analysis_scope(temp_dir, required=True)

        self.assertEqual(scope.formal_epoch, 2)
        self.assertEqual(scope.duration_sec, 50.0)
        self.assertEqual(scope.phase_at(10.0), module.PHASE_REFERENCE_BUILD)
        self.assertEqual(scope.phase_at(12.0), module.PHASE_PRE_MOTION_STATIC)
        self.assertEqual(scope.phase_at(20.0), module.PHASE_MOTION)
        self.assertEqual(scope.phase_at(50.0), module.PHASE_POST_MOTION_HOLD)
        self.assertEqual(scope.phase_at(60.0), module.PHASE_POST_MOTION_HOLD)
        self.assertIsNone(scope.phase_at(60.001))

    def test_rejects_failed_or_unordered_protocol(self):
        module = load_module()
        with tempfile.TemporaryDirectory(prefix="formal_scope_fail_") as temp_dir:
            write_protocol(temp_dir, status="FAIL")
            with self.assertRaisesRegex(module.FormalScopeError, "status"):
                module.load_formal_analysis_scope(temp_dir, required=True)

        with tempfile.TemporaryDirectory(prefix="formal_scope_order_") as temp_dir:
            write_protocol(
                temp_dir,
                actual={"hold_start_time": 19.0},
            )
            with self.assertRaisesRegex(module.FormalScopeError, "not ordered"):
                module.load_formal_analysis_scope(temp_dir, required=True)

    def test_scoped_records_filter_time_epoch_and_monitoring_phase(self):
        module = load_module()
        with tempfile.TemporaryDirectory(prefix="formal_scope_records_") as temp_dir:
            write_protocol(temp_dir)
            scope = module.load_formal_analysis_scope(temp_dir, required=True)
            records = [
                {"header": {"stamp": {"sec": 9.0}}, "reference_epoch": 1},
                {"header": {"stamp": {"sec": 11.0}}, "reference_epoch": 1},
                {
                    "header": {"stamp": {"secs": 20, "nsecs": 0}},
                    "reference_epoch": 2,
                    "phase": 1,
                },
                {
                    "header": {"stamp": {"sec": 30.0}},
                    "reference_epoch": 2,
                    "phase": 0,
                },
                {"header": {"stamp": {"sec": 50.0}}, "phase": 1},
                {"header": {"stamp": {"sec": 61.0}}, "reference_epoch": 2},
                {"header": {}, "reference_epoch": 2},
            ]
            scoped = module.scoped_records(
                records,
                scope,
                "test",
                predicate=lambda record: int(record.get("phase", 1)) == 1,
            )
            included = list(scoped)
            audit = scoped.audit()

        self.assertEqual(len(included), 2)
        self.assertEqual(audit["included_record_count"], 2)
        self.assertEqual(audit["excluded_before_trial_count"], 1)
        self.assertEqual(audit["excluded_after_trial_count"], 1)
        self.assertEqual(audit["excluded_epoch_count"], 1)
        self.assertEqual(audit["excluded_predicate_count"], 1)
        self.assertEqual(audit["excluded_invalid_timestamp_count"], 1)
        self.assertEqual(audit["included_epoch_record_count"], 1)
        self.assertEqual(audit["included_epochless_record_count"], 1)

    def test_completed_iteration_reuses_length_and_audit(self):
        module = load_module()

        class CountingRecords:
            def __init__(self, records):
                self.records = records
                self.iteration_count = 0

            def __iter__(self):
                self.iteration_count += 1
                return iter(self.records)

        with tempfile.TemporaryDirectory(prefix="formal_scope_single_pass_") as temp_dir:
            write_protocol(temp_dir)
            scope = module.load_formal_analysis_scope(temp_dir, required=True)
            source = CountingRecords(
                [
                    {"header": {"stamp": {"sec": 20.0}}, "reference_epoch": 2},
                    {"header": {"stamp": {"sec": 30.0}}, "reference_epoch": 2},
                ]
            )
            scoped = module.scoped_records(source, scope, "counted")

            self.assertEqual(len(list(scoped)), 2)
            self.assertEqual(scoped.audit()["included_record_count"], 2)

        self.assertEqual(source.iteration_count, 1)


if __name__ == "__main__":
    unittest.main()
