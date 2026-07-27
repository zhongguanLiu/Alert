import csv
import importlib.util
import json
import pathlib
import tempfile
import unittest


SCRIPT_PATH = (
    pathlib.Path(__file__).resolve().parents[1]
    / "scripts"
    / "analyze_anchor_continuity.py"
)


def load_module_if_exists():
    if not SCRIPT_PATH.exists():
        return None
    spec = importlib.util.spec_from_file_location("analyze_anchor_continuity", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def point(x, y=0.0, z=0.0):
    return {"x": x, "y": y, "z": z}


def anchor(
    anchor_id,
    anchor_type,
    obs_state,
    *,
    ref_x=1.0,
    matched_x=1.0,
    reference_epoch=0,
    reference_stamp=10.0,
    reference_origin=0,
    comparable=False,
    observable=True,
    reacquired=False,
):
    return {
        "id": anchor_id,
        "anchor_type": anchor_type,
        "ref_center": point(ref_x),
        "matched_center": point(matched_x),
        "matched_delta": point(matched_x - ref_x),
        "comparable": comparable,
        "observable": observable,
        "reacquired": reacquired,
        "obs_state": obs_state,
        "reference_epoch": reference_epoch,
        "reference_stamp": {"sec": reference_stamp},
        "reference_origin": reference_origin,
    }


def record(time_sec, anchors, reference_epoch=0, initialized_at=10.0):
    return {
        "header": {"stamp": {"sec": time_sec}},
        "reference_epoch": reference_epoch,
        "reference_initialized_at": {"sec": initialized_at},
        "anchors": anchors,
    }


class AnalyzeAnchorContinuityTests(unittest.TestCase):
    def test_rejects_non_monotonic_record_order_instead_of_buffering_and_sorting(self):
        module = load_module_if_exists()
        records = [
            record(21.0, []),
            record(20.0, []),
        ]

        with self.assertRaisesRegex(ValueError, "non-monotonic"):
            module.analyze_records(records)

    def test_loads_compact_anchor_catalog_and_observations(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        with tempfile.TemporaryDirectory(prefix="anchor_continuity_compact_") as temp_dir:
            run_dir = pathlib.Path(temp_dir)
            algorithm_dir = run_dir / "algorithm"
            algorithm_dir.mkdir()
            catalog = {
                "schema_version": 2,
                "reference_epoch": 4,
                "anchor_id": 7,
                "anchor": anchor(7, 0, 1, reference_epoch=4),
            }
            observation = {
                "schema_version": 2,
                "header": {"stamp": {"sec": 20.0}},
                "reference_epoch": 4,
                "reference_initialized_at": {"sec": 10.0},
                "anchors": [
                    {
                        "id": 7,
                        "obs_state": 3,
                        "observable": True,
                        "comparable": False,
                        "matched_center": point(1.0),
                        "matched_delta": point(0.0),
                    }
                ],
            }
            for filename, payload in (
                ("anchor_catalog.jsonl", catalog),
                ("anchor_observations.jsonl", observation),
            ):
                with (algorithm_dir / filename).open("w") as handle:
                    json.dump(payload, handle)
                    handle.write("\n")

            records = module.load_anchor_state_records(run_dir)
            self.assertIsInstance(records, module.ReplayableSequence)
            loaded_anchor = records[0]["anchors"][0]

        self.assertEqual(loaded_anchor["anchor_type"], 0)
        self.assertEqual(loaded_anchor["obs_state"], 3)
        self.assertEqual(loaded_anchor["reference_epoch"], 4)

    def test_loads_sqlite_anchor_observations_with_jsonl_catalog(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        with tempfile.TemporaryDirectory(prefix="anchor_continuity_sqlite_") as temp_dir:
            run_dir = pathlib.Path(temp_dir)
            algorithm_dir = run_dir / "algorithm"
            meta_dir = run_dir / "meta"
            algorithm_dir.mkdir()
            meta_dir.mkdir()
            (meta_dir / "run_info.json").write_text(
                json.dumps(
                    {
                        "algorithm_recording": {
                            "schema_version": 3,
                            "storage_backend": "sqlite_zlib",
                            "database_file": "algorithm/algorithm_frames.sqlite3",
                            "compressed_streams": [
                                "anchor_observations",
                                "risk_evidence",
                                "clusters",
                            ],
                        }
                    }
                )
            )
            catalog = {
                "reference_epoch": 4,
                "anchor_id": 7,
                "anchor": anchor(7, 0, 1, reference_epoch=4),
            }
            with (algorithm_dir / "anchor_catalog.jsonl").open("w") as handle:
                json.dump(catalog, handle)
                handle.write("\n")
            observation = {
                "header": {"seq": 20, "stamp": {"secs": 20, "nsecs": 0}},
                "reference_epoch": 4,
                "anchors": [
                    {
                        "id": 7,
                        "obs_state": 3,
                        "observable": True,
                        "comparable": False,
                    }
                ],
            }
            store = module.CompressedFrameStore(
                algorithm_dir / "algorithm_frames.sqlite3"
            )
            store.write_frame("anchor_observations", observation)
            store.close()

            records = module.load_anchor_state_records(run_dir)
            self.assertIsInstance(records, module.ReplayableSequence)
            loaded_anchor = records[0]["anchors"][0]

        self.assertEqual(loaded_anchor["anchor_type"], 0)
        self.assertEqual(loaded_anchor["obs_state"], 3)
        self.assertEqual(loaded_anchor["reference_epoch"], 4)

    def test_detects_visibility_gap_recovery_without_reference_reset(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        records = [
            record(
                20.0,
                [anchor(7, 0, 1, matched_x=1.02, comparable=True)],
            ),
            record(
                21.0,
                [anchor(7, 0, 3, observable=True)],
            ),
            record(
                22.0,
                [anchor(7, 0, 3, observable=True)],
            ),
            record(
                23.0,
                [
                    anchor(
                        7,
                        0,
                        1,
                        matched_x=1.05,
                        comparable=True,
                        reacquired=True,
                    )
                ],
            ),
        ]

        result = module.analyze_records(records)

        self.assertEqual(result["audit"]["reference_epoch_change_count"], 0)
        self.assertEqual(result["audit"]["datum_mutation_count"], 0)
        self.assertEqual(len(result["events"]), 1)
        event = result["events"][0]
        self.assertEqual(event["anchor_id"], 7)
        self.assertEqual(event["anchor_type"], "PLANE")
        self.assertEqual(event["loss_state"], "OBSERVABLE_MISSING")
        self.assertTrue(event["recovered"])
        self.assertTrue(event["reacquired_flag"])
        self.assertAlmostEqual(event["recovery_latency_sec"], 2.0)
        self.assertTrue(event["reference_preserved"])
        self.assertAlmostEqual(event["pre_loss_displacement_m"], 0.02)
        self.assertAlmostEqual(event["recovery_displacement_m"], 0.05)
        self.assertAlmostEqual(event["displacement_change_across_gap_m"], 0.03)
        self.assertAlmostEqual(event["measurement_identity_error_m"], 0.0)

        plane = next(
            row for row in result["by_type"] if row["anchor_type"] == "PLANE"
        )
        self.assertEqual(plane["loss_event_count"], 1)
        self.assertEqual(plane["initial_anchor_count"], 1)
        self.assertEqual(plane["incremental_anchor_count"], 0)
        self.assertEqual(plane["initial_loss_event_count"], 1)
        self.assertEqual(plane["incremental_loss_event_count"], 0)
        self.assertEqual(plane["recovered_event_count"], 1)
        self.assertAlmostEqual(plane["recovery_rate"], 1.0)
        self.assertAlmostEqual(plane["mean_recovery_latency_sec"], 2.0)
        self.assertAlmostEqual(plane["datum_preservation_rate"], 1.0)

    def test_type_summary_separates_incremental_anchor_births(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        records = [
            record(
                20.0,
                [
                    anchor(
                        10,
                        2,
                        1,
                        reference_stamp=20.0,
                        reference_origin=1,
                        comparable=True,
                    )
                ],
            ),
            record(
                21.0,
                [
                    anchor(
                        10,
                        2,
                        0,
                        reference_stamp=20.0,
                        reference_origin=1,
                        observable=False,
                    )
                ],
            ),
        ]

        result = module.analyze_records(records)
        band = next(
            row for row in result["by_type"] if row["anchor_type"] == "BAND"
        )

        self.assertEqual(band["initial_anchor_count"], 0)
        self.assertEqual(band["incremental_anchor_count"], 1)
        self.assertEqual(band["initial_loss_event_count"], 0)
        self.assertEqual(band["incremental_loss_event_count"], 1)

    def test_reports_open_gap_and_reference_datum_mutation(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        records = [
            record(
                20.0,
                [anchor(9, 1, 1, matched_x=1.01, comparable=True)],
            ),
            record(
                21.0,
                [anchor(9, 1, 2, ref_x=1.02)],
            ),
            record(
                22.0,
                [anchor(9, 1, 2, ref_x=1.02)],
            ),
        ]

        result = module.analyze_records(records)

        self.assertEqual(result["audit"]["datum_mutation_count"], 1)
        self.assertAlmostEqual(result["audit"]["max_ref_center_drift_m"], 0.02)
        self.assertEqual(len(result["events"]), 1)
        self.assertFalse(result["events"][0]["recovered"])
        self.assertEqual(result["events"][0]["recovery_latency_sec"], "")
        self.assertFalse(result["events"][0]["reference_preserved"])

        edge = next(
            row for row in result["by_type"] if row["anchor_type"] == "EDGE"
        )
        self.assertEqual(edge["loss_event_count"], 1)
        self.assertEqual(edge["recovered_event_count"], 0)
        self.assertAlmostEqual(edge["recovery_rate"], 0.0)

    def test_epoch_change_is_a_reset_not_a_visibility_recovery(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        records = [
            record(
                20.0,
                [anchor(3, 2, 1, reference_epoch=1, comparable=True)],
                reference_epoch=1,
                initialized_at=10.0,
            ),
            record(
                21.0,
                [anchor(3, 2, 3, reference_epoch=1)],
                reference_epoch=1,
                initialized_at=10.0,
            ),
            record(
                22.0,
                [
                    anchor(
                        3,
                        2,
                        1,
                        reference_epoch=2,
                        reference_stamp=22.0,
                        comparable=True,
                    )
                ],
                reference_epoch=2,
                initialized_at=22.0,
            ),
        ]

        result = module.analyze_records(records)

        self.assertEqual(result["audit"]["reference_epoch_change_count"], 1)
        self.assertEqual(result["audit"]["reference_epoch_count"], 2)
        self.assertEqual(len(result["events"]), 1)
        self.assertFalse(result["events"][0]["recovered"])
        self.assertEqual(result["events"][0]["closure_reason"], "REFERENCE_RESET")

    def test_empty_reset_record_does_not_mutate_initialized_timestamp(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        records = [
            record(
                20.0,
                [anchor(3, 2, 1, reference_epoch=0, comparable=True)],
                reference_epoch=0,
                initialized_at=10.0,
            ),
            record(21.0, [], reference_epoch=1, initialized_at=0.0),
            record(
                22.0,
                [
                    anchor(
                        3,
                        2,
                        1,
                        reference_epoch=1,
                        reference_stamp=22.0,
                        comparable=True,
                    )
                ],
                reference_epoch=1,
                initialized_at=22.0,
            ),
        ]

        result = module.analyze_records(records)

        self.assertEqual(result["audit"]["reference_epoch_change_count"], 1)
        self.assertEqual(
            result["audit"]["reference_initialized_at_mutation_count"], 0
        )
        self.assertEqual(
            result["audit"]["reference_initialized_at_by_epoch"]["1"], 22.0
        )

    def test_anchor_epoch_mismatch_invalidates_continuity_evidence(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        records = [
            record(
                20.0,
                [anchor(3, 2, 1, reference_epoch=4, comparable=True)],
                reference_epoch=3,
                initialized_at=10.0,
            )
        ]

        result = module.analyze_records(records)

        self.assertEqual(result["audit"]["anchor_epoch_mismatch_count"], 1)
        self.assertFalse(result["audit"]["continuity_evidence_valid"])

    def test_cli_writes_event_type_and_audit_outputs(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        with tempfile.TemporaryDirectory(prefix="anchor_continuity_") as temp_dir:
            run_dir = pathlib.Path(temp_dir) / "sim_run_000"
            input_path = run_dir / "algorithm" / "anchor_states.jsonl"
            input_path.parent.mkdir(parents=True)
            records = [
                record(20.0, [anchor(7, 0, 1, comparable=True)]),
                record(21.0, [anchor(7, 0, 3)]),
                record(
                    22.0,
                    [anchor(7, 0, 1, matched_x=1.03, comparable=True, reacquired=True)],
                ),
            ]
            with input_path.open("w") as handle:
                for item in records:
                    json.dump(item, handle)
                    handle.write("\n")

            outputs = module.analyze_run(run_dir)

            self.assertTrue(outputs.events_csv.is_file())
            self.assertTrue(outputs.by_type_csv.is_file())
            self.assertTrue(outputs.audit_json.is_file())
            with outputs.events_csv.open() as handle:
                event_rows = list(csv.DictReader(handle))
            self.assertEqual(event_rows[0]["anchor_type"], "PLANE")
            with outputs.audit_json.open() as handle:
                audit = json.load(handle)
            self.assertEqual(audit["reference_epoch_change_count"], 0)

    def test_analyze_run_accepts_an_already_decoded_record_iterable(self):
        module = load_module_if_exists()
        with tempfile.TemporaryDirectory(prefix="anchor_continuity_stream_") as temp_dir:
            run_dir = pathlib.Path(temp_dir) / "sim_run_000"
            run_dir.mkdir()
            records = iter(
                [
                    record(20.0, [anchor(7, 0, 1, comparable=True)]),
                    record(21.0, [anchor(7, 0, 3)]),
                ]
            )

            outputs = module.analyze_run(
                run_dir,
                output_dir=run_dir / "streamed_continuity",
                record_iterable=records,
            )

            with outputs.audit_json.open() as handle:
                audit = json.load(handle)
        self.assertEqual(audit["record_count"], 2)


if __name__ == "__main__":
    unittest.main()
