import importlib.util
import hashlib
import json
import pathlib
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import types
import unittest
import zlib


SCRIPT_PATH = (
    pathlib.Path(__file__).resolve().parents[1]
    / "scripts"
    / "algorithm_frame_store.py"
)
COMPARE_SCRIPT_PATH = SCRIPT_PATH.with_name("compare_algorithm_storage.py")


def load_module_if_exists():
    if not SCRIPT_PATH.exists():
        return None
    spec = importlib.util.spec_from_file_location(
        "algorithm_frame_store_under_test", SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_stamp(seq):
    return {
        "seq": seq,
        "frame_id": "camera_init",
        "stamp": {
            "secs": 100 + seq,
            "nsecs": 123456789,
            "sec": float(100 + seq) + 0.123456789,
        },
    }


def make_payload(stream_name, seq=1, item_count=3):
    item_key = {
        "anchor_observations": "anchors",
        "risk_evidence": "evidences",
        "clusters": "clusters",
    }[stream_name]
    items = []
    for index in range(item_count):
        items.append(
            {
                "id": index + 1,
                "active": index % 2 == 0,
                "significant": index % 3 == 0,
                "position": {
                    "x": 1.0 + index * 0.001,
                    "y": -2.5,
                    "z": 0.0,
                },
                "labels": [31, 32, 33],
                "diagnostic": None,
            }
        )
    return {
        "schema_version": 2,
        "header": make_stamp(seq),
        "reference_epoch": 4,
        "recorded_at": {
            "secs": 200 + seq,
            "nsecs": 987654321,
            "sec": float(200 + seq) + 0.987654321,
        },
        item_key: items,
    }


def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for record in records:
            json.dump(record, handle, sort_keys=True)
            handle.write("\n")


def write_run_info(run_dir, backend=None, schema_version=2):
    algorithm_recording = {"schema_version": schema_version}
    if backend is not None:
        algorithm_recording.update(
            {
                "storage_backend": backend,
                "database_file": "algorithm/algorithm_frames.sqlite3",
                "compressed_streams": [
                    "anchor_observations",
                    "risk_evidence",
                    "clusters",
                ],
            }
        )
    path = pathlib.Path(run_dir) / "meta" / "run_info.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"algorithm_recording": algorithm_recording},
            sort_keys=True,
        )
        + "\n"
    )


def write_dual_run(module, run_dir, sqlite_mutator=None):
    write_run_info(run_dir, backend="dual", schema_version=3)
    database_path = pathlib.Path(run_dir) / "algorithm" / "algorithm_frames.sqlite3"
    store = module.CompressedFrameStore(database_path)
    expected = {}
    for stream_name in module.HIGH_VOLUME_STREAMS:
        records = [
            make_payload(stream_name, seq=31, item_count=2),
            make_payload(stream_name, seq=32, item_count=3),
        ]
        expected[stream_name] = records
        write_jsonl(
            pathlib.Path(run_dir) / "algorithm" / f"{stream_name}.jsonl",
            records,
        )
        for record_index, record in enumerate(records):
            sqlite_record = json.loads(json.dumps(record))
            if sqlite_mutator is not None:
                sqlite_mutator(stream_name, record_index, sqlite_record)
            store.write_frame(stream_name, sqlite_record)
    store.close()
    return expected


class AlgorithmFrameStoreTests(unittest.TestCase):
    def test_inspection_rejects_tampered_storage_metadata(self):
        module = load_module_if_exists()
        with tempfile.TemporaryDirectory() as root:
            database_path = pathlib.Path(root) / "algorithm_frames.sqlite3"
            store = module.CompressedFrameStore(database_path)
            store.write_frame(
                "clusters",
                make_payload("clusters", seq=16, item_count=1),
            )
            store.close()
            with sqlite3.connect(str(database_path)) as connection:
                connection.execute(
                    """
                    UPDATE storage_metadata
                       SET metadata_value = 'lossy_codec'
                     WHERE metadata_key = 'codec'
                    """
                )

            with self.assertRaisesRegex(
                module.AlgorithmFrameStoreError,
                "metadata",
            ):
                module.inspect_sqlite_store(database_path)

    def test_all_supported_streams_round_trip_in_source_time_order(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        with tempfile.TemporaryDirectory() as root:
            database_path = pathlib.Path(root) / "algorithm_frames.sqlite3"
            store = module.CompressedFrameStore(database_path)
            expected_by_stream = {}
            for stream_name in module.HIGH_VOLUME_STREAMS:
                later = make_payload(stream_name, seq=2, item_count=2)
                earlier = make_payload(stream_name, seq=1, item_count=1)
                store.write_frame(stream_name, later)
                store.write_frame(stream_name, earlier)
                expected_by_stream[stream_name] = [earlier, later]
            store.close()

            for stream_name, expected in expected_by_stream.items():
                with self.subTest(stream_name=stream_name):
                    self.assertEqual(
                        list(module.iter_sqlite_stream(database_path, stream_name)),
                        expected,
                    )

    def test_round_trip_preserves_every_field_and_value(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        payload = make_payload("anchor_observations", seq=7, item_count=200)
        with tempfile.TemporaryDirectory() as root:
            database_path = pathlib.Path(root) / "algorithm_frames.sqlite3"
            store = module.CompressedFrameStore(database_path)
            store.write_frame("anchor_observations", payload)
            summary = store.close()

            decoded = list(
                module.iter_sqlite_stream(
                    database_path,
                    "anchor_observations",
                )
            )

        self.assertEqual(decoded, [payload])
        stream_summary = summary["streams"]["anchor_observations"]
        self.assertEqual(stream_summary["frame_count"], 1)
        self.assertEqual(stream_summary["item_count"], 200)
        self.assertLess(
            stream_summary["compressed_byte_count"],
            stream_summary["raw_byte_count"],
        )
        self.assertEqual(summary["integrity_check"], "ok")

    def test_duplicate_source_stamp_is_rejected_without_overwrite(self):
        module = load_module_if_exists()
        payload = make_payload("risk_evidence", seq=4, item_count=2)
        with tempfile.TemporaryDirectory() as root:
            database_path = pathlib.Path(root) / "algorithm_frames.sqlite3"
            store = module.CompressedFrameStore(database_path)
            store.write_frame("risk_evidence", payload)
            with self.assertRaisesRegex(
                module.AlgorithmFrameStoreError,
                "Duplicate",
            ):
                store.write_frame("risk_evidence", payload)
            store.close()

            self.assertEqual(
                list(module.iter_sqlite_stream(database_path, "risk_evidence")),
                [payload],
            )

    def test_writer_rejects_nonfinite_payload(self):
        module = load_module_if_exists()
        payload = make_payload("clusters", seq=5, item_count=1)
        payload["clusters"][0]["position"]["x"] = float("nan")
        with tempfile.TemporaryDirectory() as root:
            store = module.CompressedFrameStore(
                pathlib.Path(root) / "algorithm_frames.sqlite3"
            )
            with self.assertRaisesRegex(
                module.AlgorithmFrameStoreError,
                "finite JSON",
            ):
                store.write_frame("clusters", payload)
            summary = store.close()

        self.assertNotIn("clusters", summary["streams"])

    def test_reader_rejects_corrupt_compressed_payload(self):
        module = load_module_if_exists()
        payload = make_payload("anchor_observations", seq=6, item_count=2)
        with tempfile.TemporaryDirectory() as root:
            database_path = pathlib.Path(root) / "algorithm_frames.sqlite3"
            store = module.CompressedFrameStore(database_path)
            result = store.write_frame("anchor_observations", payload)
            store.close()

            with sqlite3.connect(str(database_path)) as connection:
                connection.execute(
                    "UPDATE stream_frames SET compressed_payload = ? WHERE frame_pk = ?",
                    (sqlite3.Binary(b"not-zlib"), result["frame_pk"]),
                )

            with self.assertRaises(module.AlgorithmFrameStoreError):
                list(
                    module.iter_sqlite_stream(
                        database_path,
                        "anchor_observations",
                    )
                )

    def test_reader_rejects_item_count_mismatch(self):
        module = load_module_if_exists()
        payload = make_payload("clusters", seq=8, item_count=3)
        with tempfile.TemporaryDirectory() as root:
            database_path = pathlib.Path(root) / "algorithm_frames.sqlite3"
            store = module.CompressedFrameStore(database_path)
            result = store.write_frame("clusters", payload)
            store.close()

            with sqlite3.connect(str(database_path)) as connection:
                connection.execute(
                    "UPDATE stream_frames SET item_count = ? WHERE frame_pk = ?",
                    (99, result["frame_pk"]),
                )

            with self.assertRaisesRegex(
                module.AlgorithmFrameStoreError,
                "Item count mismatch",
            ):
                list(module.iter_sqlite_stream(database_path, "clusters"))

    def test_reader_rejects_index_metadata_mismatch(self):
        module = load_module_if_exists()
        payload = make_payload("risk_evidence", seq=18, item_count=2)
        with tempfile.TemporaryDirectory() as root:
            database_path = pathlib.Path(root) / "algorithm_frames.sqlite3"
            store = module.CompressedFrameStore(database_path)
            result = store.write_frame("risk_evidence", payload)
            store.close()

            with sqlite3.connect(str(database_path)) as connection:
                connection.execute(
                    "UPDATE stream_frames SET stamp_nsecs = stamp_nsecs + 1 "
                    "WHERE frame_pk = ?",
                    (result["frame_pk"],),
                )

            with self.assertRaisesRegex(
                module.AlgorithmFrameStoreError,
                "Frame metadata mismatch.*stamp_nsecs",
            ):
                list(module.iter_sqlite_stream(database_path, "risk_evidence"))

    def test_close_is_idempotent(self):
        module = load_module_if_exists()
        with tempfile.TemporaryDirectory() as root:
            store = module.CompressedFrameStore(
                pathlib.Path(root) / "algorithm_frames.sqlite3"
            )
            store.write_frame(
                "clusters",
                make_payload("clusters", seq=9, item_count=1),
            )
            first = store.close()
            second = store.close()

        self.assertEqual(second, first)


class AsyncAlgorithmFrameStoreTests(unittest.TestCase):
    def test_encoding_runs_in_parallel_before_single_writer_commit(self):
        module = load_module_if_exists()
        original_encoder = module.canonical_json_bytes
        release_encoders = threading.Event()
        condition = threading.Condition()
        encoder_threads = set()

        def blocking_encoder(payload):
            with condition:
                encoder_threads.add(threading.current_thread().name)
                condition.notify_all()
            release_encoders.wait(timeout=5.0)
            return original_encoder(payload)

        with tempfile.TemporaryDirectory() as root:
            database_path = pathlib.Path(root) / "algorithm_frames.sqlite3"
            writer = module.AsyncCompressedFrameStore(database_path)
            module.canonical_json_bytes = blocking_encoder
            try:
                writer.enqueue(
                    "anchor_observations",
                    make_payload("anchor_observations", seq=101, item_count=20),
                )
                writer.enqueue(
                    "risk_evidence",
                    make_payload("risk_evidence", seq=102, item_count=20),
                )
                deadline = time.monotonic() + 2.0
                with condition:
                    while len(encoder_threads) < 2 and time.monotonic() < deadline:
                        condition.wait(timeout=deadline - time.monotonic())
                observed_threads = set(encoder_threads)
            finally:
                release_encoders.set()
                writer.close()
                module.canonical_json_bytes = original_encoder

            decoded_anchor = list(
                module.iter_sqlite_stream(database_path, "anchor_observations")
            )
            decoded_risk = list(
                module.iter_sqlite_stream(database_path, "risk_evidence")
            )

        self.assertGreaterEqual(len(observed_threads), 2)
        self.assertNotIn("algorithm-frame-store-writer", observed_threads)
        self.assertEqual(len(decoded_anchor), 1)
        self.assertEqual(len(decoded_risk), 1)

    def test_close_drains_every_enqueued_frame_before_returning(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        with tempfile.TemporaryDirectory() as root:
            database_path = pathlib.Path(root) / "algorithm_frames.sqlite3"
            writer = module.AsyncCompressedFrameStore(
                database_path,
                queue_capacity=2,
            )
            expected = []
            committed = []
            for seq in range(1, 21):
                payload = make_payload("clusters", seq=seq, item_count=3)
                expected.append(payload)
                writer.enqueue(
                    "clusters",
                    payload,
                    on_commit=lambda result: committed.append(result["header_seq"]),
                )

            summary = writer.close()
            decoded = list(module.iter_sqlite_stream(database_path, "clusters"))

        self.assertEqual(decoded, expected)
        self.assertEqual(committed, list(range(1, 21)))
        self.assertEqual(summary["streams"]["clusters"]["frame_count"], 20)

    def test_trim_sqlite_stream_keeps_only_committed_prefix(self):
        module = load_module_if_exists()
        with tempfile.TemporaryDirectory() as root:
            database_path = pathlib.Path(root) / "algorithm_frames.sqlite3"
            store = module.CompressedFrameStore(database_path)
            expected = []
            for seq in range(1, 6):
                payload = make_payload("risk_evidence", seq=seq, item_count=seq)
                store.write_frame("risk_evidence", payload)
                expected.append(payload)
            store.close()

            summary = module.trim_sqlite_stream(
                database_path,
                "risk_evidence",
                keep_count=3,
            )
            decoded = list(
                module.iter_sqlite_stream(database_path, "risk_evidence")
            )

        self.assertEqual(decoded, expected[:3])
        self.assertEqual(summary["integrity_check"], "ok")
        self.assertEqual(summary["streams"]["risk_evidence"]["frame_count"], 3)

    def test_duplicate_worker_failure_makes_close_fail(self):
        module = load_module_if_exists()
        payload = make_payload("clusters", seq=11, item_count=1)
        with tempfile.TemporaryDirectory() as root:
            writer = module.AsyncCompressedFrameStore(
                pathlib.Path(root) / "algorithm_frames.sqlite3"
            )
            writer.enqueue("clusters", payload)
            try:
                writer.enqueue("clusters", payload)
            except module.AlgorithmFrameStoreError:
                pass

            with self.assertRaises(module.AlgorithmFrameStoreError):
                writer.close()

    def test_queue_timeout_is_explicit_and_close_remains_failed(self):
        module = load_module_if_exists()
        callback_started = threading.Event()
        release_callback = threading.Event()

        def block_after_first_commit(_result):
            callback_started.set()
            release_callback.wait(timeout=5.0)

        with tempfile.TemporaryDirectory() as root:
            writer = module.AsyncCompressedFrameStore(
                pathlib.Path(root) / "algorithm_frames.sqlite3",
                queue_capacity=1,
                enqueue_timeout_sec=0.05,
            )
            writer.enqueue(
                "clusters",
                make_payload("clusters", seq=12, item_count=1),
                on_commit=block_after_first_commit,
            )
            self.assertTrue(callback_started.wait(timeout=2.0))
            writer.enqueue(
                "clusters",
                make_payload("clusters", seq=13, item_count=1),
            )
            with self.assertRaisesRegex(
                module.AlgorithmFrameStoreError,
                "queue remained full",
            ):
                writer.enqueue(
                    "clusters",
                    make_payload("clusters", seq=14, item_count=1),
                )
            release_callback.set()
            with self.assertRaises(module.AlgorithmFrameStoreError):
                writer.close()

    def test_enqueue_after_successful_close_is_rejected(self):
        module = load_module_if_exists()
        with tempfile.TemporaryDirectory() as root:
            writer = module.AsyncCompressedFrameStore(
                pathlib.Path(root) / "algorithm_frames.sqlite3"
            )
            writer.close()
            with self.assertRaisesRegex(
                module.AlgorithmFrameStoreError,
                "closed",
            ):
                writer.enqueue(
                    "clusters",
                    make_payload("clusters", seq=15, item_count=1),
                )


class AlgorithmStreamReaderTests(unittest.TestCase):
    def test_compare_cli_import_falls_back_from_catkin_relay_module(self):
        placeholder = types.ModuleType("algorithm_frame_store")
        previous = sys.modules.get("algorithm_frame_store")
        sys.modules["algorithm_frame_store"] = placeholder
        try:
            spec = importlib.util.spec_from_file_location(
                "compare_algorithm_storage_under_test",
                COMPARE_SCRIPT_PATH,
            )
            compare_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(compare_module)
        finally:
            if previous is None:
                sys.modules.pop("algorithm_frame_store", None)
            else:
                sys.modules["algorithm_frame_store"] = previous

        self.assertTrue(callable(compare_module.compare_dual_storage))
        self.assertTrue(issubclass(compare_module.AlgorithmFrameStoreError, RuntimeError))

    def test_required_storage_paths_follow_declared_backend(self):
        module = load_module_if_exists()
        with tempfile.TemporaryDirectory() as root:
            run_dir = pathlib.Path(root) / "sim_run_000"

            write_run_info(run_dir, backend="sqlite_zlib", schema_version=3)
            self.assertEqual(
                module.required_algorithm_storage_paths(run_dir, "clusters"),
                (pathlib.Path("algorithm/algorithm_frames.sqlite3"),),
            )

            write_run_info(run_dir, backend="dual", schema_version=3)
            self.assertEqual(
                module.required_algorithm_storage_paths(run_dir, "clusters"),
                (
                    pathlib.Path("algorithm/clusters.jsonl"),
                    pathlib.Path("algorithm/algorithm_frames.sqlite3"),
                ),
            )

    def test_replayable_stream_iterates_twice_without_caching_payload_list(self):
        module = load_module_if_exists()
        with tempfile.TemporaryDirectory() as root:
            run_dir = pathlib.Path(root) / "sim_run_000"
            expected = write_dual_run(module, run_dir)["clusters"]
            records = module.ReplayableAlgorithmStream(
                run_dir,
                "clusters",
                required=True,
                representation="sqlite_zlib",
            )

            self.assertEqual(len(records), 2)
            self.assertEqual(records[0], expected[0])
            self.assertEqual(list(records), expected)
            self.assertEqual(list(records), expected)
            self.assertFalse(hasattr(records, "_records"))

    def test_sqlite_stream_length_uses_row_count_without_decoding_payloads(self):
        module = load_module_if_exists()
        with tempfile.TemporaryDirectory() as root:
            run_dir = pathlib.Path(root) / "sim_run_000"
            write_run_info(run_dir, backend="sqlite_zlib", schema_version=3)
            database_path = run_dir / "algorithm" / "algorithm_frames.sqlite3"
            store = module.CompressedFrameStore(database_path)
            result = store.write_frame(
                "clusters",
                make_payload("clusters", seq=35, item_count=2),
            )
            store.close()
            with sqlite3.connect(str(database_path)) as connection:
                connection.execute(
                    "UPDATE stream_frames SET compressed_payload = ? WHERE frame_pk = ?",
                    (sqlite3.Binary(b"corrupt"), result["frame_pk"]),
                )

            records = module.ReplayableAlgorithmStream(
                run_dir,
                "clusters",
                required=True,
            )
            self.assertEqual(len(records), 1)
            with self.assertRaises(module.AlgorithmFrameStoreError):
                list(records)

    def test_compare_cli_writes_machine_readable_report(self):
        module = load_module_if_exists()
        with tempfile.TemporaryDirectory() as root:
            run_dir = pathlib.Path(root) / "sim_run_000"
            write_dual_run(module, run_dir)
            output_path = run_dir / "analysis" / "storage_equivalence.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(COMPARE_SCRIPT_PATH),
                    str(run_dir),
                    "--output",
                    str(output_path),
                ],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            report = json.loads(output_path.read_text())

        self.assertTrue(report["equivalent"])
    def test_schema_v2_run_without_backend_reads_legacy_jsonl(self):
        module = load_module_if_exists()
        payloads = [
            make_payload("clusters", seq=22, item_count=1),
            make_payload("clusters", seq=23, item_count=2),
        ]
        with tempfile.TemporaryDirectory() as root:
            run_dir = pathlib.Path(root) / "sim_run_000"
            write_run_info(run_dir, backend=None, schema_version=2)
            write_jsonl(run_dir / "algorithm" / "clusters.jsonl", payloads)
            decoded = list(
                module.iter_algorithm_stream(run_dir, "clusters", required=True)
            )
        self.assertEqual(decoded, payloads)

    def test_iter_algorithm_stream_honors_declared_sqlite_backend(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        payload = make_payload("risk_evidence", seq=21, item_count=4)
        with tempfile.TemporaryDirectory() as root:
            run_dir = pathlib.Path(root) / "sim_run_000"
            write_run_info(run_dir, backend="sqlite_zlib", schema_version=3)
            database_path = run_dir / "algorithm" / "algorithm_frames.sqlite3"
            store = module.CompressedFrameStore(database_path)
            store.write_frame("risk_evidence", payload)
            store.close()

            decoded = list(
                module.iter_algorithm_stream(
                    run_dir,
                    "risk_evidence",
                    required=True,
                )
            )

        self.assertEqual(decoded, [payload])

    def test_undeclared_simultaneous_representations_are_rejected(self):
        module = load_module_if_exists()
        payload = make_payload("clusters", seq=24, item_count=1)
        with tempfile.TemporaryDirectory() as root:
            run_dir = pathlib.Path(root) / "sim_run_000"
            write_run_info(run_dir, backend="sqlite_zlib", schema_version=3)
            write_jsonl(run_dir / "algorithm" / "clusters.jsonl", [payload])
            store = module.CompressedFrameStore(
                run_dir / "algorithm" / "algorithm_frames.sqlite3"
            )
            store.write_frame("clusters", payload)
            store.close()
            with self.assertRaisesRegex(
                module.AlgorithmFrameStoreError,
                "Ambiguous",
            ):
                list(module.iter_algorithm_stream(run_dir, "clusters", required=True))

    def test_dual_mode_can_read_each_representation_explicitly(self):
        module = load_module_if_exists()
        with tempfile.TemporaryDirectory() as root:
            run_dir = pathlib.Path(root) / "sim_run_000"
            expected = write_dual_run(module, run_dir)
            jsonl = list(
                module.iter_algorithm_stream(
                    run_dir,
                    "risk_evidence",
                    required=True,
                    representation="jsonl",
                )
            )
            sqlite_records = list(
                module.iter_algorithm_stream(
                    run_dir,
                    "risk_evidence",
                    required=True,
                    representation="sqlite_zlib",
                )
            )
        self.assertEqual(jsonl, expected["risk_evidence"])
        self.assertEqual(sqlite_records, expected["risk_evidence"])

    def test_compare_dual_storage_reports_exact_equivalence(self):
        module = load_module_if_exists()
        with tempfile.TemporaryDirectory() as root:
            run_dir = pathlib.Path(root) / "sim_run_000"
            write_dual_run(module, run_dir)
            report = module.compare_dual_storage(run_dir)

        self.assertTrue(report["equivalent"])
        self.assertEqual(report["database_integrity_check"], "ok")
        for stream_name in module.HIGH_VOLUME_STREAMS:
            self.assertEqual(report["streams"][stream_name]["frame_count"], 2)
            self.assertTrue(report["streams"][stream_name]["equivalent"])

    def test_compare_dual_storage_reports_type_sensitive_field_path(self):
        module = load_module_if_exists()

        def mutate(stream_name, record_index, payload):
            if stream_name == "risk_evidence" and record_index == 0:
                payload["evidences"][0]["active"] = 1

        with tempfile.TemporaryDirectory() as root:
            run_dir = pathlib.Path(root) / "sim_run_000"
            write_dual_run(module, run_dir, sqlite_mutator=mutate)
            report = module.compare_dual_storage(run_dir)

        self.assertFalse(report["equivalent"])
        mismatch = report["streams"]["risk_evidence"]["first_mismatch"]
        self.assertEqual(mismatch["path"], "$.evidences[0].active")
        self.assertEqual(mismatch["reason"], "type_mismatch")

    def test_reader_rejects_numeric_overflow_even_with_matching_hash(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        payload = make_payload("clusters", seq=3, item_count=1)
        with tempfile.TemporaryDirectory() as root:
            database_path = pathlib.Path(root) / "algorithm_frames.sqlite3"
            store = module.CompressedFrameStore(database_path)
            result = store.write_frame("clusters", payload)
            store.close()

            invalid_json = (
                b'{"clusters":[{"value":1e999}],"header":{"seq":3,'
                b'"stamp":{"nsecs":123456789,"secs":103}},'
                b'"recorded_at":{"nsecs":987654321,"secs":203},'
                b'"reference_epoch":4,"schema_version":2}'
            )
            compressed = zlib.compress(invalid_json, 1)
            with sqlite3.connect(str(database_path)) as connection:
                connection.execute(
                    """
                    UPDATE stream_frames
                       SET raw_byte_count = ?,
                           compressed_byte_count = ?,
                           payload_sha256 = ?,
                           compressed_payload = ?
                     WHERE frame_pk = ?
                    """,
                    (
                        len(invalid_json),
                        len(compressed),
                        hashlib.sha256(invalid_json).hexdigest(),
                        sqlite3.Binary(compressed),
                        result["frame_pk"],
                    ),
                )

            with self.assertRaisesRegex(
                module.AlgorithmFrameStoreError,
                "non-finite",
            ):
                list(module.iter_sqlite_stream(database_path, "clusters"))


if __name__ == "__main__":
    unittest.main()
