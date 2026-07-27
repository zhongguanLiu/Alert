import importlib.util
import csv
import json
import pathlib
import xml.etree.ElementTree as ET
import sys
import tempfile
import threading
import types
import unittest
from types import SimpleNamespace


SCRIPT_PATH = (
    pathlib.Path(__file__).resolve().parents[1] / "scripts" / "sim_experiment_recorder.py"
)


ROS_IMPORT_ROOTS = {
    "rospy",
    "roslib",
    "std_msgs",
    "geometry_msgs",
    "sensor_msgs",
    "nav_msgs",
    "visualization_msgs",
    "tf",
    "tf2_ros",
    "tf2_msgs",
    "deform_monitor_v2",
}


class _StubModule(types.ModuleType):
    def __getattr__(self, name):
        value = type(name, (), {})
        setattr(self, name, value)
        return value


class _FakeSubscriber(SimpleNamespace):
    pass


class _FakeRospy:
    def __init__(self, params=None, now_sec=123.0):
        self.params = dict(params or {})
        self.now_sec = now_sec
        self.subscribers = []
        self.shutdown_callbacks = []
        self.logged_info = []
        self.logged_warnings = []
        self.Time = SimpleNamespace(
            now=lambda: SimpleNamespace(to_sec=lambda: self.now_sec)
        )

    def get_param(self, name, default=None):
        return self.params.get(name, default)

    def get_param_names(self):
        return sorted(self.params.keys())

    def on_shutdown(self, callback):
        self.shutdown_callbacks.append(callback)

    def Subscriber(self, topic, message_class, callback, queue_size=1):
        subscriber = _FakeSubscriber(
            topic=topic,
            message_class=message_class,
            callback=callback,
            queue_size=queue_size,
        )
        self.subscribers.append(subscriber)
        return subscriber

    def loginfo(self, *args):
        self.logged_info.append(args)

    def logwarn_throttle(self, *args):
        self.logged_warnings.append(args)


class _FakeTfListener:
    def __init__(self, transforms=None, error=None):
        self.transforms = dict(transforms or {})
        self.error = error
        self.lookups = []

    def lookupTransform(self, target_frame, source_frame, stamp):
        self.lookups.append((target_frame, source_frame, stamp))
        if self.error is not None:
            raise self.error
        return self.transforms[(target_frame, source_frame)]


def make_persistent_region(
    track_id=7,
    state=1,
    region_type=1,
    center=None,
    bbox_min=None,
    bbox_max=None,
    mean_risk=0.4,
    peak_risk=0.8,
    confidence=0.6,
    accumulated_risk=1.2,
    support_mass=3.0,
    spatial_span=0.7,
    hit_streak=4,
    miss_streak=1,
    age_frames=5,
    confirmed=True,
):
    return SimpleNamespace(
        track_id=track_id,
        state=state,
        region_type=region_type,
        center=SimpleNamespace(**(center or {"x": 1.0, "y": 2.0, "z": 3.0})),
        bbox_min=SimpleNamespace(**(bbox_min or {"x": 0.5, "y": 1.5, "z": 2.5})),
        bbox_max=SimpleNamespace(**(bbox_max or {"x": 1.5, "y": 2.5, "z": 3.5})),
        mean_risk=mean_risk,
        peak_risk=peak_risk,
        confidence=confidence,
        accumulated_risk=accumulated_risk,
        support_mass=support_mass,
        spatial_span=spatial_span,
        hit_streak=hit_streak,
        miss_streak=miss_streak,
        age_frames=age_frames,
        confirmed=confirmed,
    )


def make_motion_cluster(
    cluster_id=3,
    anchor_ids=None,
    center=None,
    bbox_min=None,
    bbox_max=None,
    disp_mean=None,
    disp_cov=None,
    chi2_stat=6.5,
    disp_norm=0.022,
    confidence=0.8,
    support_count=9,
    significant=True,
):
    return SimpleNamespace(
        id=cluster_id,
        anchor_ids=list(anchor_ids or [1, 2, 3]),
        center=SimpleNamespace(**(center or {"x": 1.0, "y": 2.0, "z": 3.0})),
        bbox_min=SimpleNamespace(**(bbox_min or {"x": 0.5, "y": 1.5, "z": 2.5})),
        bbox_max=SimpleNamespace(**(bbox_max or {"x": 1.5, "y": 2.5, "z": 3.5})),
        disp_mean=list(disp_mean or [0.01, 0.0, 0.0]),
        disp_cov=list(
            disp_cov
            or [1.0e-4, 0.0, 0.0, 0.0, 1.0e-4, 0.0, 0.0, 0.0, 1.0e-4]
        ),
        chi2_stat=chi2_stat,
        disp_norm=disp_norm,
        confidence=confidence,
        support_count=support_count,
        significant=significant,
    )


def _install_stub_module(module_name, added_modules, parent_attrs):
    parts = module_name.split(".")
    for index in range(1, len(parts) + 1):
        partial_name = ".".join(parts[:index])
        if partial_name in sys.modules:
            continue

        module = _StubModule(partial_name)
        if index < len(parts):
            module.__path__ = []
        sys.modules[partial_name] = module
        added_modules.append(partial_name)

        if index > 1:
            parent_name = ".".join(parts[: index - 1])
            parent_module = sys.modules[parent_name]
            attr_name = parts[index - 1]
            parent_key = (parent_name, attr_name)
            if parent_key not in parent_attrs:
                parent_attrs[parent_key] = getattr(parent_module, attr_name, None)
            setattr(parent_module, attr_name, module)


def _restore_stub_modules(added_modules, parent_attrs):
    for parent_name, attr_name in reversed(list(parent_attrs)):
        original_value = parent_attrs[(parent_name, attr_name)]
        parent_module = sys.modules.get(parent_name)
        if parent_module is None:
            continue
        if original_value is None:
            parent_module.__dict__.pop(attr_name, None)
        else:
            setattr(parent_module, attr_name, original_value)

    for module_name in reversed(added_modules):
        sys.modules.pop(module_name, None)


def load_module_if_exists():
    if not SCRIPT_PATH.exists():
        return None

    spec = importlib.util.spec_from_file_location("sim_experiment_recorder", SCRIPT_PATH)
    stubbed_module_names = set()
    while True:
        module = importlib.util.module_from_spec(spec)
        added_modules = []
        parent_attrs = {}
        try:
            for module_name in sorted(stubbed_module_names):
                _install_stub_module(module_name, added_modules, parent_attrs)
            spec.loader.exec_module(module)
            return module
        except ModuleNotFoundError as exc:
            missing_name = exc.name or ""
            if missing_name.split(".")[0] not in ROS_IMPORT_ROOTS:
                raise
            stubbed_module_names.add(missing_name)
        finally:
            _restore_stub_modules(added_modules, parent_attrs)


class SimExperimentRecorderHelperTests(unittest.TestCase):
    def test_normalize_algorithm_storage_backend_accepts_only_declared_modes(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        self.assertEqual(module.normalize_algorithm_storage_backend(" JSONL "), "jsonl")
        self.assertEqual(
            module.normalize_algorithm_storage_backend("SQLITE_ZLIB"),
            "sqlite_zlib",
        )
        self.assertEqual(module.normalize_algorithm_storage_backend("dual"), "dual")
        with self.assertRaises(ValueError):
            module.normalize_algorithm_storage_backend("sampled")

    def test_sqlite_backend_routes_complete_risk_frame_without_jsonl(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        payload = {
            "schema_version": 2,
            "header": {
                "seq": 5,
                "frame_id": "camera_init",
                "stamp": {"secs": 12, "nsecs": 345000000, "sec": 12.345},
            },
            "reference_epoch": 2,
            "recorded_at": {"secs": 12, "nsecs": 400000000, "sec": 12.4},
            "evidences": [
                {
                    "anchor_id": 17,
                    "active": False,
                    "significant": True,
                    "risk_score": 0.25,
                }
            ],
        }
        with tempfile.TemporaryDirectory(prefix="sim_experiment_recorder_") as root:
            recorder = module.SimExperimentRecorder.__new__(module.SimExperimentRecorder)
            recorder.algorithm_dir = pathlib.Path(root) / "algorithm"
            recorder.algorithm_dir.mkdir()
            recorder.algorithm_storage_backend = "sqlite_zlib"
            recorder._algorithm_files = {}
            recorder._algorithm_frame_store = module.AsyncCompressedFrameStore(
                recorder.algorithm_dir / module.ALGORITHM_FRAME_DATABASE_FILENAME
            )
            recorder._closed = False
            recorder.record_flush_max_rows = 100
            recorder.record_flush_interval_sec = 60.0
            try:
                self.assertTrue(
                    recorder._append_algorithm_payload(
                        "risk_evidence",
                        "risk_evidence.jsonl",
                        payload,
                    )
                )
            finally:
                recorder._algorithm_frame_store.close()
            decoded = list(
                module.iter_sqlite_stream(
                    recorder.algorithm_dir
                    / module.ALGORITHM_FRAME_DATABASE_FILENAME,
                    "risk_evidence",
                )
            )

            self.assertFalse((recorder.algorithm_dir / "risk_evidence.jsonl").exists())
            self.assertEqual(decoded, [payload])
            self.assertEqual(recorder._stream_stats["risk_evidence"]["row_count"], 1)

    def test_risk_callback_uses_selected_sqlite_backend(self):
        module = load_module_if_exists()
        msg = SimpleNamespace(
            header=SimpleNamespace(
                seq=8,
                stamp=SimpleNamespace(secs=20, nsecs=0),
                frame_id="camera_init",
            ),
            reference_epoch=3,
            evidences=[SimpleNamespace(anchor_id=17, active=False, risk_score=0.2)],
        )
        fake_rospy = _FakeRospy(now_sec=20.1)
        original_rospy = module.rospy
        module.rospy = fake_rospy
        try:
            with tempfile.TemporaryDirectory(prefix="sim_experiment_recorder_") as root:
                recorder = module.SimExperimentRecorder.__new__(
                    module.SimExperimentRecorder
                )
                recorder.algorithm_dir = pathlib.Path(root) / "algorithm"
                recorder.algorithm_dir.mkdir()
                recorder.algorithm_storage_backend = "sqlite_zlib"
                recorder._algorithm_files = {}
                recorder._algorithm_frame_store = module.AsyncCompressedFrameStore(
                    recorder.algorithm_dir
                    / module.ALGORITHM_FRAME_DATABASE_FILENAME
                )
                recorder._closed = False
                recorder.record_flush_max_rows = 100
                recorder.record_flush_interval_sec = 60.0
                try:
                    recorder._handle_risk_evidence(msg)
                finally:
                    recorder._algorithm_frame_store.close()

                decoded = list(
                    module.iter_sqlite_stream(
                        recorder.algorithm_dir
                        / module.ALGORITHM_FRAME_DATABASE_FILENAME,
                        "risk_evidence",
                    )
                )
                self.assertFalse(
                    (recorder.algorithm_dir / "risk_evidence.jsonl").exists()
                )
                self.assertEqual(decoded[0]["header"]["seq"], 8)
                self.assertFalse(decoded[0]["evidences"][0]["active"])
        finally:
            module.rospy = original_rospy

    def test_recorder_initializes_sqlite_storage_before_subscribing(self):
        module = load_module_if_exists()
        fake_rospy = _FakeRospy(
            params={
                "~output_root": "",
                "~algorithm_storage_backend": "sqlite_zlib",
            }
        )
        original_rospy = module.rospy
        original_tf = module.tf
        original_model_states = module.ModelStates
        original_link_states = module.LinkStates
        recorder = None
        try:
            with tempfile.TemporaryDirectory(
                prefix="sim_experiment_recorder_"
            ) as root:
                fake_rospy.params["~output_root"] = root
                module.rospy = fake_rospy
                module.tf = SimpleNamespace(TransformListener=lambda: object())
                module.ModelStates = object
                module.LinkStates = object

                recorder = module.SimExperimentRecorder()
                run_info = json.loads(
                    (recorder.meta_dir / "run_info.json").read_text()
                )

                self.assertEqual(recorder.algorithm_storage_backend, "sqlite_zlib")
                self.assertIsNotNone(recorder._algorithm_frame_store)
                self.assertTrue(
                    (
                        recorder.algorithm_dir
                        / module.ALGORITHM_FRAME_DATABASE_FILENAME
                    ).is_file()
                )
                self.assertEqual(
                    run_info["algorithm_recording"]["schema_version"],
                    3,
                )
                self.assertEqual(
                    run_info["algorithm_recording"]["storage_backend"],
                    "sqlite_zlib",
                )
                self.assertGreater(len(fake_rospy.subscribers), 0)
                anchor_subscriber = next(
                    subscriber
                    for subscriber in fake_rospy.subscribers
                    if subscriber.topic == "/deform/anchors"
                )
                self.assertEqual(anchor_subscriber.queue_size, 64)
                model_subscriber = next(
                    subscriber
                    for subscriber in fake_rospy.subscribers
                    if subscriber.topic == "/gazebo/model_states"
                )
                link_subscriber = next(
                    subscriber
                    for subscriber in fake_rospy.subscribers
                    if subscriber.topic == "/gazebo/link_states"
                )
                self.assertEqual(model_subscriber.queue_size, 64)
                self.assertEqual(link_subscriber.queue_size, 64)
                self.assertEqual(
                    run_info["truth_recording"]["write_mode"],
                    "asynchronous_batched",
                )
                recorder.close()
                recorder = None
        finally:
            if recorder is not None:
                recorder.close()
            module.rospy = original_rospy
            module.tf = original_tf
            module.ModelStates = original_model_states
            module.LinkStates = original_link_states

    def test_sqlite_close_trims_only_incomplete_common_frame_suffix(self):
        module = load_module_if_exists()
        fake_rospy = _FakeRospy(now_sec=30.0)
        original_rospy = module.rospy
        module.rospy = fake_rospy
        try:
            with tempfile.TemporaryDirectory(
                prefix="sim_experiment_recorder_"
            ) as root:
                recorder = module.SimExperimentRecorder.__new__(
                    module.SimExperimentRecorder
                )
                recorder.algorithm_dir = pathlib.Path(root) / "algorithm"
                recorder.meta_dir = pathlib.Path(root) / "meta"
                recorder.algorithm_dir.mkdir()
                recorder.meta_dir.mkdir()
                recorder.algorithm_storage_backend = "sqlite_zlib"
                recorder._algorithm_files = {}
                recorder._object_files = {}
                recorder._link_files = {}
                recorder._algorithm_frame_store = module.AsyncCompressedFrameStore(
                    recorder.algorithm_dir
                    / module.ALGORITHM_FRAME_DATABASE_FILENAME
                )
                recorder._closed = False
                recorder.record_flush_max_rows = 100
                recorder.record_flush_interval_sec = 60.0

                def payload_for(stream_name, stamp):
                    payload = {
                        "schema_version": 2,
                        "header": {
                            "seq": stamp,
                            "stamp": {
                                "secs": stamp,
                                "nsecs": 0,
                                "sec": float(stamp),
                            },
                        },
                        "reference_epoch": 1,
                        "recorded_at": {
                            "secs": stamp,
                            "nsecs": 1000000,
                            "sec": float(stamp) + 0.001,
                        },
                    }
                    if stream_name == "anchor_observations":
                        payload["anchors"] = []
                    elif stream_name == "risk_evidence":
                        payload["evidences"] = []
                    elif stream_name == "clusters":
                        payload["clusters"] = []
                    if stream_name == "object_observation_stats":
                        payload["phase"] = 1
                    return payload

                def append(stream_name, stamp):
                    payload = payload_for(stream_name, stamp)
                    filename = module.FRAME_COMMIT_STREAM_FILES[stream_name]
                    if stream_name in module.COMPRESSED_ALGORITHM_STREAMS:
                        recorder._append_algorithm_payload(
                            stream_name,
                            filename,
                            payload,
                        )
                    else:
                        recorder._append_jsonl(stream_name, filename, payload)

                for stamp in (1, 2):
                    for stream_name in module.FRAME_COMMIT_STREAM_FILES:
                        append(stream_name, stamp)
                for stream_name in module.FRAME_COMMIT_STREAM_FILES:
                    if stream_name not in (
                        "processing_stamps",
                        "anchor_observations",
                    ):
                        append(stream_name, 3)

                recorder.close()

                database_path = (
                    recorder.algorithm_dir
                    / module.ALGORITHM_FRAME_DATABASE_FILENAME
                )
                for stream_name in module.COMPRESSED_ALGORITHM_STREAMS:
                    with self.subTest(stream_name=stream_name):
                        decoded = list(
                            module.iter_sqlite_stream(database_path, stream_name)
                        )
                        self.assertEqual(
                            [record["header"]["stamp"]["sec"] for record in decoded],
                            [1.0, 2.0],
                        )

                completion = json.loads(
                    (recorder.meta_dir / "run_complete.json").read_text()
                )
                self.assertEqual(
                    completion["algorithm_storage"]["integrity_check"],
                    "ok",
                )
                self.assertEqual(
                    completion["algorithm_storage"]["streams"]["clusters"][
                        "frame_count"
                    ],
                    2,
                )
                self.assertEqual(
                    completion["streams"]["clusters"]["row_count"],
                    2,
                )
        finally:
            module.rospy = original_rospy

    def test_sqlite_writer_failure_marks_run_incomplete(self):
        module = load_module_if_exists()
        fake_rospy = _FakeRospy(now_sec=40.0)
        original_rospy = module.rospy
        module.rospy = fake_rospy
        payload = {
            "schema_version": 2,
            "header": {
                "seq": 1,
                "stamp": {"secs": 1, "nsecs": 0, "sec": 1.0},
            },
            "reference_epoch": 1,
            "recorded_at": {"secs": 1, "nsecs": 1, "sec": 1.000000001},
            "clusters": [],
        }
        try:
            with tempfile.TemporaryDirectory(
                prefix="sim_experiment_recorder_"
            ) as root:
                recorder = module.SimExperimentRecorder.__new__(
                    module.SimExperimentRecorder
                )
                recorder.algorithm_dir = pathlib.Path(root) / "algorithm"
                recorder.meta_dir = pathlib.Path(root) / "meta"
                recorder.algorithm_dir.mkdir()
                recorder.meta_dir.mkdir()
                recorder.algorithm_storage_backend = "sqlite_zlib"
                recorder._algorithm_files = {}
                recorder._object_files = {}
                recorder._link_files = {}
                recorder._algorithm_frame_store = module.AsyncCompressedFrameStore(
                    recorder.algorithm_dir
                    / module.ALGORITHM_FRAME_DATABASE_FILENAME
                )
                recorder._closed = False
                recorder.record_flush_max_rows = 100
                recorder.record_flush_interval_sec = 60.0

                recorder._append_algorithm_payload(
                    "clusters", "clusters.jsonl", payload
                )
                try:
                    recorder._append_algorithm_payload(
                        "clusters", "clusters.jsonl", payload
                    )
                except module.AlgorithmFrameStoreError:
                    pass
                recorder.close()

                completion = json.loads(
                    (recorder.meta_dir / "run_complete.json").read_text()
                )
                self.assertFalse(completion["clean_shutdown"])
                self.assertFalse(completion["recording_integrity_valid"])
                self.assertTrue(completion["recording_error"])
        finally:
            module.rospy = original_rospy

    def test_dual_backend_writes_both_representations_but_counts_once(self):
        module = load_module_if_exists()
        payload = {
            "schema_version": 2,
            "header": {
                "seq": 2,
                "stamp": {"secs": 2, "nsecs": 0, "sec": 2.0},
            },
            "reference_epoch": 1,
            "recorded_at": {"secs": 2, "nsecs": 1, "sec": 2.000000001},
            "anchors": [{"id": 7, "significant": False}],
        }
        with tempfile.TemporaryDirectory(prefix="sim_experiment_recorder_") as root:
            recorder = module.SimExperimentRecorder.__new__(module.SimExperimentRecorder)
            recorder.algorithm_dir = pathlib.Path(root) / "algorithm"
            recorder.algorithm_dir.mkdir()
            recorder.algorithm_storage_backend = "dual"
            recorder._algorithm_files = {}
            recorder._algorithm_frame_store = module.AsyncCompressedFrameStore(
                recorder.algorithm_dir / module.ALGORITHM_FRAME_DATABASE_FILENAME
            )
            recorder._closed = False
            recorder.record_flush_max_rows = 100
            recorder.record_flush_interval_sec = 60.0
            try:
                recorder._append_algorithm_payload(
                    "anchor_observations",
                    "anchor_observations.jsonl",
                    payload,
                )
                recorder._algorithm_frame_store.close()
                for handle in recorder._algorithm_files.values():
                    handle.flush()
                jsonl_payload = json.loads(
                    (recorder.algorithm_dir / "anchor_observations.jsonl")
                    .read_text()
                    .strip()
                )
                sqlite_payload = list(
                    module.iter_sqlite_stream(
                        recorder.algorithm_dir
                        / module.ALGORITHM_FRAME_DATABASE_FILENAME,
                        "anchor_observations",
                    )
                )[0]
            finally:
                for handle in recorder._algorithm_files.values():
                    handle.close()

        self.assertEqual(jsonl_payload, payload)
        self.assertEqual(sqlite_payload, payload)
        self.assertEqual(recorder._stream_stats["anchor_observations"]["row_count"], 1)

    @staticmethod
    def _pose(x=0.0, y=0.0, z=0.0):
        return SimpleNamespace(
            position=SimpleNamespace(x=x, y=y, z=z),
            orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
        )

    @staticmethod
    def _twist(lx=0.0, ly=0.0, lz=0.0, ax=0.0, ay=0.0, az=0.0):
        return SimpleNamespace(
            linear=SimpleNamespace(x=lx, y=ly, z=lz),
            angular=SimpleNamespace(x=ax, y=ay, z=az),
        )

    def test_truth_object_schema_includes_rigid_body_twist(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        self.assertIn("twist_frame_id", module.TRUTH_OBJECT_HEADER)
        self.assertEqual(
            module.TRUTH_OBJECT_HEADER[-6:],
            [
                "linear_velocity_x",
                "linear_velocity_y",
                "linear_velocity_z",
                "angular_velocity_x",
                "angular_velocity_y",
                "angular_velocity_z",
            ],
        )

    def test_truth_recording_rate_is_positive_and_configurable(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        self.assertEqual(module.normalize_positive_rate_hz("5.0", 10.0, "rate"), 5.0)
        self.assertTrue(module.should_sample_stream(1.0, None, 10.0))
        self.assertFalse(module.should_sample_stream(1.05, 1.0, 10.0))
        self.assertTrue(module.should_sample_stream(1.10, 1.0, 10.0))
        self.assertTrue(module.should_sample_stream(0.01, 1.0, 10.0))
        with self.assertRaises(ValueError):
            module.normalize_positive_rate_hz(0.0, 10.0, "rate")

        policy = module.normalize_truth_motion_policy(
            {"translation_deadband_m": 0.002, "sustained_motion_samples": 3}
        )
        self.assertEqual(policy["translation_deadband_m"], 0.002)
        self.assertEqual(policy["sustained_motion_samples"], 3)
        zero_speed_policy = module.normalize_truth_motion_policy(
            {"linear_speed_deadband_mps": 0.0,
             "angular_speed_deadband_degps": 0.0}
        )
        self.assertEqual(zero_speed_policy["linear_speed_deadband_mps"], 0.0)
        self.assertEqual(zero_speed_policy["angular_speed_deadband_degps"], 0.0)

    def test_phase_preserving_sampler_approaches_ten_hz_from_twelve_point_five_hz(self):
        module = load_module_if_exists()
        next_sample = None
        last_observed = None
        sampled_times = []
        for index in range(126):
            stamp = 1.0 + 0.08 * index
            sampled, next_sample, last_observed = module.advance_sampling_clock(
                stamp, next_sample, last_observed, 10.0
            )
            if sampled:
                sampled_times.append(stamp)

        effective_rate = (len(sampled_times) - 1) / (
            sampled_times[-1] - sampled_times[0]
        )
        self.assertGreaterEqual(effective_rate, 9.5)
        self.assertLessEqual(effective_rate, 10.1)

    def test_phase_preserving_sampler_resets_after_clock_rollback(self):
        module = load_module_if_exists()
        sampled, next_sample, last_observed = module.advance_sampling_clock(
            10.0, None, None, 10.0
        )
        self.assertTrue(sampled)
        sampled, next_sample, last_observed = module.advance_sampling_clock(
            10.05, next_sample, last_observed, 10.0
        )
        self.assertFalse(sampled)
        sampled, next_sample, last_observed = module.advance_sampling_clock(
            1.0, next_sample, last_observed, 10.0
        )
        self.assertTrue(sampled)
        self.assertAlmostEqual(next_sample, 1.1)

    def test_sampling_statistics_report_configured_and_effective_rates(self):
        module = load_module_if_exists()
        stats = module.new_sampling_stats(configured_rate_hz=10.0)
        module.update_sampling_stats(stats, 1.0, sampled=True, rows_written=2)
        module.update_sampling_stats(stats, 1.05, sampled=False, rows_written=0)
        module.update_sampling_stats(stats, 1.1, sampled=True, rows_written=2)

        payload = module.finalize_sampling_stats(stats)

        self.assertEqual(payload["configured_rate_hz"], 10.0)
        self.assertEqual(payload["received_message_count"], 3)
        self.assertEqual(payload["sampled_message_count"], 2)
        self.assertEqual(payload["rows_written"], 4)
        self.assertAlmostEqual(payload["effective_sample_rate_hz"], 10.0)

    def test_truth_pipeline_reports_callback_gaps_and_lossless_drain(self):
        module = load_module_if_exists()
        stats = module.new_truth_pipeline_stats(configured_rate_hz=10.0)
        module.update_truth_callback_stats(stats, 1.0)
        module.update_truth_callback_stats(stats, 1.1)
        module.update_truth_callback_stats(stats, 1.5)
        stats["enqueued_batch_count"] = 2
        stats["written_batch_count"] = 2
        stats["enqueued_row_count"] = 20
        stats["written_row_count"] = 20

        payload = module.finalize_truth_pipeline_stats(stats)

        self.assertEqual(payload["callback_count"], 3)
        self.assertAlmostEqual(payload["max_callback_gap_sec"], 0.4)
        self.assertEqual(payload["estimated_missed_sample_slots"], 3)
        self.assertTrue(payload["lossless_after_enqueue"])

    def test_async_truth_writer_drains_rows_before_shutdown(self):
        module = load_module_if_exists()
        with tempfile.TemporaryDirectory(prefix="sim_experiment_recorder_") as root:
            recorder = module.SimExperimentRecorder.__new__(
                module.SimExperimentRecorder
            )
            recorder.truth_objects_dir = pathlib.Path(root) / "objects"
            recorder.truth_links_dir = pathlib.Path(root) / "links"
            recorder.truth_dir = pathlib.Path(root)
            recorder.truth_objects_dir.mkdir()
            recorder.truth_links_dir.mkdir()
            recorder._object_files = {}
            recorder._link_files = {}
            recorder._surface_truth_catalog_handle = None
            recorder.truth_object_rate_hz = 10.0
            recorder.truth_link_rate_hz = 10.0
            recorder.truth_processing_queue_size = 4
            recorder.truth_processing_enqueue_timeout_sec = 0.5
            recorder.record_flush_interval_sec = 60.0
            recorder.record_flush_max_rows = 100
            recorder._truth_write_queue = None
            recorder._truth_write_thread = None
            recorder._truth_write_error = None
            recorder._start_truth_write_worker()

            row = ["1.000000000", "model_01", "world", "world"] + [0.0] * 13
            recorder._enqueue_truth_rows("model_states", [("model_01", row)])
            recorder._close_truth_write_worker()

            rows = list(
                csv.DictReader(
                    (recorder.truth_objects_dir / "model_01.csv")
                    .read_text()
                    .splitlines()
                )
            )
            stats = module.finalize_truth_pipeline_stats(
                recorder._truth_pipeline_stats["model_states"]
            )
            self.assertEqual(len(rows), 1)
            self.assertEqual(stats["enqueued_row_count"], 1)
            self.assertEqual(stats["written_row_count"], 1)
            self.assertTrue(stats["lossless_after_enqueue"])
            for handle, _ in recorder._object_files.values():
                handle.close()

    def test_model_truth_records_ground_plane_and_gazebo_twist(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        with tempfile.TemporaryDirectory(prefix="sim_experiment_recorder_") as temp_dir:
            temp_dir = pathlib.Path(temp_dir)
            fake_rospy = _FakeRospy(now_sec=1.0)
            original_rospy = module.rospy
            module.rospy = fake_rospy
            try:
                recorder = module.SimExperimentRecorder.__new__(module.SimExperimentRecorder)
                recorder.ego_model_name = "mid360_fastlio"
                recorder.truth_frame = "world"
                recorder.truth_object_rate_hz = 10.0
                recorder.truth_objects_dir = temp_dir / "truth" / "objects"
                recorder.truth_objects_dir.mkdir(parents=True)
                recorder._object_files = {}
                recorder._link_files = {}
                recorder._algorithm_files = {}
                recorder._last_model_states_write_time = None
                recorder._latest_model_pose_world = {}
                recorder._ego_initial_pose_written = False
                recorder._refresh_scenario_manifest_if_needed = lambda: False

                msg = SimpleNamespace(
                    name=["ground_plane", "model_01"],
                    pose=[self._pose(), self._pose(1.0, 2.0, 3.0)],
                    twist=[self._twist(), self._twist(0.1, -0.2, 0.3, 0.4, -0.5, 0.6)],
                )
                recorder._handle_model_states(msg)

                self.assertTrue((recorder.truth_objects_dir / "ground_plane.csv").exists())
                rows = list(
                    csv.DictReader(
                        (recorder.truth_objects_dir / "model_01.csv")
                        .read_text()
                        .splitlines()
                    )
                )
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]["twist_frame_id"], "world")
                self.assertEqual(float(rows[0]["linear_velocity_x"]), 0.1)
                self.assertEqual(float(rows[0]["linear_velocity_y"]), -0.2)
                self.assertEqual(float(rows[0]["angular_velocity_z"]), 0.6)
                self.assertIn("model_01", recorder._latest_model_pose_world)
                recorder.close()
            finally:
                module.rospy = original_rospy

    def test_model_truth_waits_for_positive_simulation_clock(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        with tempfile.TemporaryDirectory(prefix="sim_experiment_recorder_") as temp_dir:
            temp_dir = pathlib.Path(temp_dir)
            fake_rospy = _FakeRospy(now_sec=0.0)
            original_rospy = module.rospy
            module.rospy = fake_rospy
            try:
                recorder = module.SimExperimentRecorder.__new__(module.SimExperimentRecorder)
                recorder.ego_model_name = "mid360_fastlio"
                recorder.truth_frame = "world"
                recorder.truth_object_rate_hz = 10.0
                recorder.truth_objects_dir = temp_dir / "truth" / "objects"
                recorder.truth_objects_dir.mkdir(parents=True)
                recorder._object_files = {}
                recorder._link_files = {}
                recorder._algorithm_files = {}
                recorder._last_model_states_write_time = None
                recorder._latest_model_pose_world = {}
                recorder._ego_initial_pose_written = False
                recorder._refresh_scenario_manifest_if_needed = lambda: False

                msg = SimpleNamespace(
                    name=["model_01"],
                    pose=[self._pose(1.0, 2.0, 3.0)],
                    twist=[self._twist()],
                )
                recorder._handle_model_states(msg)
                self.assertFalse(
                    (recorder.truth_objects_dir / "model_01.csv").exists()
                )
                self.assertEqual(recorder._latest_model_pose_world, {})

                fake_rospy.now_sec = 2219.49
                recorder._handle_model_states(msg)
                recorder.close()
                rows = list(
                    csv.DictReader(
                        (recorder.truth_objects_dir / "model_01.csv")
                        .read_text()
                        .splitlines()
                    )
                )
                self.assertEqual(len(rows), 1)
                self.assertEqual(float(rows[0]["recorded_time_sec"]), 2219.49)
                self.assertEqual(len(fake_rospy.logged_warnings), 1)
            finally:
                module.rospy = original_rospy

    def test_model_truth_rate_limits_rows_without_losing_latest_pose_cache(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        with tempfile.TemporaryDirectory(prefix="sim_experiment_recorder_") as temp_dir:
            temp_dir = pathlib.Path(temp_dir)
            fake_rospy = _FakeRospy(now_sec=1.0)
            original_rospy = module.rospy
            module.rospy = fake_rospy
            try:
                recorder = module.SimExperimentRecorder.__new__(module.SimExperimentRecorder)
                recorder.ego_model_name = "ego"
                recorder.truth_frame = "world"
                recorder.truth_object_rate_hz = 2.0
                recorder.truth_objects_dir = temp_dir / "objects"
                recorder.truth_objects_dir.mkdir()
                recorder._object_files = {}
                recorder._link_files = {}
                recorder._algorithm_files = {}
                recorder._last_model_states_write_time = None
                recorder._latest_model_pose_world = {}
                recorder._ego_initial_pose_written = False
                recorder._refresh_scenario_manifest_if_needed = lambda: False

                def emit(stamp, x):
                    fake_rospy.now_sec = stamp
                    recorder._handle_model_states(
                        SimpleNamespace(
                            name=["model_01"],
                            pose=[self._pose(x=x)],
                            twist=[self._twist(lx=x)],
                        )
                    )

                emit(1.0, 1.0)
                emit(1.2, 2.0)
                self.assertEqual(
                    recorder._latest_model_pose_world["model_01"]["pose"]["position"]["x"],
                    2.0,
                )
                emit(1.5, 3.0)

                recorder.close()
                rows = list(
                    csv.DictReader(
                        (recorder.truth_objects_dir / "model_01.csv")
                        .read_text()
                        .splitlines()
                    )
                )
                self.assertEqual([float(row["position_x"]) for row in rows], [1.0, 3.0])
            finally:
                module.rospy = original_rospy

    def test_surface_truth_link_is_saved_once_in_drive_link_local_coordinates(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        with tempfile.TemporaryDirectory(prefix="sim_experiment_recorder_") as temp_dir:
            temp_dir = pathlib.Path(temp_dir)
            fake_rospy = _FakeRospy(now_sec=2.02)
            original_rospy = module.rospy
            module.rospy = fake_rospy
            try:
                recorder = module.SimExperimentRecorder.__new__(module.SimExperimentRecorder)
                recorder.truth_dir = temp_dir / "truth"
                recorder.truth_links_dir = recorder.truth_dir / "links"
                recorder.truth_links_dir.mkdir(parents=True)
                recorder.truth_frame = "world"
                recorder.sensor_scoped_link_name = ""
                recorder.surface_truth_link_prefixes = ("ground_truth_",)
                recorder.record_surface_truth_link_trajectories = False
                recorder.motion_truth_drive_links = {
                    "model_01": "model_01::link"
                }
                recorder.truth_link_rate_hz = 10.0
                recorder._latest_sensor_pose_world = None
                recorder._latest_sensor_pose_stamp = None
                recorder._latest_model_pose_world = {
                    "model_01": {
                        "pose": module.pose_to_dict(self._pose(x=100.0)),
                        "recorded_time_sec": 2.0,
                    }
                }
                recorder._captured_surface_truth_links = set()
                recorder._surface_truth_catalog_handle = None
                recorder._link_files = {}
                recorder._object_files = {}
                recorder._algorithm_files = {}
                recorder._last_link_states_write_time = None

                msg = SimpleNamespace(
                    name=[
                        "model_01::link",
                        "model_01::ground_truth_face0_corner0",
                    ],
                    pose=[
                        self._pose(x=10.0),
                        self._pose(x=11.0, y=2.0, z=3.0),
                    ],
                )
                recorder._handle_link_states(msg)
                fake_rospy.now_sec = 2.12
                recorder._handle_link_states(msg)
                recorder.close()

                catalog_path = recorder.truth_dir / "surface_truth_points.jsonl"
                records = [json.loads(line) for line in catalog_path.read_text().splitlines()]
                self.assertEqual(len(records), 1)
                self.assertEqual(records[0]["model_name"], "model_01")
                self.assertEqual(records[0]["link_name"], "ground_truth_face0_corner0")
                self.assertEqual(
                    records[0]["motion_parent_scoped_link_name"],
                    "model_01::link",
                )
                self.assertAlmostEqual(records[0]["local_pose"]["position"]["x"], 1.0)
                self.assertAlmostEqual(records[0]["local_pose"]["position"]["y"], 2.0)
                self.assertAlmostEqual(records[0]["pose_pair_delta_sec"], 0.0)
                link_files = list(recorder.truth_links_dir.iterdir())
                self.assertEqual(len(link_files), 1)
                self.assertEqual(link_files[0].name, "model_01_link.csv")
            finally:
                module.rospy = original_rospy

    def test_static_surface_truth_catalog_is_loaded_from_parent_marker_visuals(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        with tempfile.TemporaryDirectory(prefix="static_surface_truth_") as temp_dir:
            world_path = pathlib.Path(temp_dir) / "clean.world"
            world_path.write_text(
                """<sdf version='1.6'>
  <world name='default'>
    <model name='model_01'>
      <link name='link'>
        <visual name='ground_truth_marker_v_000'>
          <pose>1 2 3 0 0 0</pose>
        </visual>
        <visual name='ground_truth_marker_v_001'>
          <pose>-1 0 0.5 0 0 0</pose>
        </visual>
      </link>
    </model>
  </world>
</sdf>"""
            )

            records = module.load_static_surface_truth_catalog(
                world_path,
                {"model_01": "model_01::link"},
                {2: "model_01"},
                expected_count=2,
                max_local_radius_m=4.0,
                require_clean_world=True,
            )

            self.assertEqual(len(records), 2)
            self.assertEqual(
                records[0]["catalog_source"], "world_static_marker_visual"
            )
            self.assertEqual(records[0]["scoped_link_name"], "model_01::ground_truth_v_000")
            self.assertEqual(records[0]["motion_parent_scoped_link_name"], "model_01::link")
            self.assertEqual(records[0]["object_id"], 2)
            self.assertEqual(
                records[0]["local_pose"]["position"],
                {"x": 1.0, "y": 2.0, "z": 3.0},
            )

    def test_static_surface_truth_catalog_rejects_saved_state(self):
        module = load_module_if_exists()
        with tempfile.TemporaryDirectory(prefix="static_surface_truth_") as temp_dir:
            world_path = pathlib.Path(temp_dir) / "state.world"
            world_path.write_text(
                """<sdf version='1.6'><world name='default'>
  <state world_name='default'/>
  <model name='model_01'><link name='link'>
    <visual name='ground_truth_marker_v_000'><pose>0 0 0 0 0 0</pose></visual>
  </link></model>
</world></sdf>"""
            )
            with self.assertRaisesRegex(ValueError, "saved <state>"):
                module.load_static_surface_truth_catalog(
                    world_path,
                    {"model_01": "model_01::link"},
                    {2: "model_01"},
                    expected_count=1,
                    require_clean_world=True,
                )

    def test_surface_truth_trajectory_mode_selects_landmarks_for_csv_recording(self):
        module = load_module_if_exists()
        recorder = module.SimExperimentRecorder.__new__(module.SimExperimentRecorder)
        recorder.sensor_scoped_link_name = "mid360::livox"
        recorder.surface_truth_link_prefixes = ("ground_truth_",)
        recorder.record_surface_truth_link_trajectories = True
        msg = SimpleNamespace(
            name=[
                "mid360::livox",
                "model_01::link",
                "model_01::ground_truth_v_000",
                "bookshelf::ground_truth_v_015",
            ]
        )

        selected = recorder._tracked_link_names(msg)

        self.assertEqual(
            selected,
            [
                "mid360::livox",
                "model_01::ground_truth_v_000",
                "bookshelf::ground_truth_v_015",
            ],
        )

    def test_drive_link_mode_selects_only_sensor_and_configured_drive_links(self):
        module = load_module_if_exists()
        recorder = module.SimExperimentRecorder.__new__(module.SimExperimentRecorder)
        recorder.sensor_scoped_link_name = "mid360::livox"
        recorder.surface_truth_link_prefixes = ("ground_truth_",)
        recorder.record_surface_truth_link_trajectories = False
        recorder.motion_truth_drive_links = {
            "model_01": "model_01::link",
            "bookshelf": "bookshelf::link",
        }
        msg = SimpleNamespace(
            name=[
                "mid360::livox",
                "model_01::link",
                "model_01::ground_truth_v_000",
                "bookshelf::link",
                "bookshelf::ground_truth_v_015",
            ]
        )

        selected = recorder._tracked_link_names(msg)

        self.assertEqual(
            selected,
            ["mid360::livox", "model_01::link", "bookshelf::link"],
        )

    def test_normalize_object_id_catalog_accepts_unique_ids_and_names(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        catalog = module.normalize_object_id_catalog(
            {"31": "moving_box", 32: "static_wall"}
        )

        self.assertEqual(catalog, {31: "moving_box", 32: "static_wall"})
        with self.assertRaises(ValueError):
            module.normalize_object_id_catalog({0: "invalid_zero"})
        with self.assertRaises(ValueError):
            module.normalize_object_id_catalog({31: "panel_a", "31": "panel_b"})
        for invalid_key in (True, 1.5, "1.5", "not_an_id"):
            with self.subTest(invalid_key=invalid_key):
                with self.assertRaises(ValueError):
                    module.normalize_object_id_catalog({invalid_key: "invalid"})

    def test_debug_recording_allows_incomplete_experiment_metadata(self):
        module = load_module_if_exists()
        module.validate_recording_configuration(
            recording_mode="debug",
            scenario_id="",
            launch_scenario_id="",
            experiment_factors={},
            object_metadata={},
            object_id_catalog={1: "model_01"},
        )

    def test_formal_recording_rejects_missing_required_metadata(self):
        module = load_module_if_exists()
        with self.assertRaisesRegex(ValueError, "scenario_id"):
            module.validate_recording_configuration(
                recording_mode="formal",
                scenario_id="",
                launch_scenario_id="",
                experiment_factors={},
                object_metadata={},
                object_id_catalog={1: "model_01"},
            )

    def test_formal_zero_motion_configuration_passes_strict_runtime_validation(self):
        module = load_module_if_exists()
        scenario_id = "collapse_microdeform_zero_motion_static_platform_r01"
        module.validate_recording_configuration(
            recording_mode="formal",
            scenario_id=scenario_id,
            launch_scenario_id=scenario_id,
            experiment_factors={
                "scene_id": "tracked_mid360_fastlio_collapse_microdeform",
                "moving_object_quantity": 0,
                "scene_object_quantity": 2,
                "platform_condition": "static",
                "slam_pipeline": "fast_lio",
                "point_cloud_setting": "mid360_sim_default",
                "repeat_index": 1,
            },
            object_metadata={},
            object_id_catalog={1: "model_01", 2: "model_02"},
        )

    def test_formal_recording_rejects_launch_or_controller_scenario_conflicts(self):
        module = load_module_if_exists()
        factors = {
            "scene_id": "scene",
            "moving_object_quantity": 0,
            "scene_object_quantity": 1,
            "platform_condition": "static",
            "slam_pipeline": "fast_lio",
            "point_cloud_setting": "mid360_sim_default",
            "repeat_index": 1,
        }
        with self.assertRaisesRegex(ValueError, "launch scenario_id"):
            module.validate_recording_configuration(
                recording_mode="formal",
                scenario_id="configured_case",
                launch_scenario_id="different_case",
                experiment_factors=factors,
                object_metadata={},
                object_id_catalog={1: "model_01"},
            )

        with self.assertRaisesRegex(ValueError, "controller scenario_id"):
            module.validate_control_scenario_ids(
                "configured_case",
                [
                    {
                        "controlled_object": "model_01",
                        "scenario_id": "different_case",
                    }
                ],
                evaluated_object_names={"model_01"},
            )

    def _assert_launch_file_keeps_explicit_fallback_args(self, launch_path):
        root = ET.parse(launch_path).getroot()
        args = {element.attrib["name"]: element.attrib.get("default") for element in root.findall("arg")}

        self.assertEqual(args["controlled_object"], "")
        self.assertEqual(args["command_frame"], "")
        self.assertEqual(args["linear_velocity_x"], "0.0")
        self.assertEqual(args["linear_velocity_y"], "0.0")
        self.assertEqual(args["linear_velocity_z"], "0.0")
        self.assertEqual(args["angular_velocity_y_deg"], "0.0")
        self.assertEqual(args["control_start_delay_sec"], "")
        self.assertEqual(args["control_duration_sec"], "")

        recorder_node = next(
            node
            for node in root.findall("node")
            if node.attrib.get("name") == "sim_experiment_recorder"
        )
        params = {
            element.attrib.get("name"): element.attrib.get("value")
            for element in recorder_node.findall("param")
        }
        self.assertNotIn("scenario_id", params)
        self.assertEqual(params["launch_scenario_id"], "$(arg scenario_id)")

    def _make_manifest_recorder_fixture(self, module, temp_dir, params=None):
        fake_rospy = _FakeRospy(params=params or {}, now_sec=123.0)
        original_rospy = module.rospy
        module.rospy = fake_rospy

        recorder = module.SimExperimentRecorder.__new__(module.SimExperimentRecorder)
        recorder.run_dir = temp_dir
        recorder.meta_dir = temp_dir / "meta"
        recorder.meta_dir.mkdir(parents=True, exist_ok=True)
        recorder.truth_dir = temp_dir / "truth"
        recorder.truth_objects_dir = recorder.truth_dir / "objects"
        recorder.truth_links_dir = recorder.truth_dir / "links"
        recorder.algorithm_dir = temp_dir / "algorithm"
        recorder.trajectory_dir = temp_dir / "trajectory"
        recorder.ego_model_name = "mid360_fastlio"
        recorder.truth_frame = "world"
        recorder._object_files = {}
        recorder._link_files = {}
        recorder._algorithm_files = {}
        recorder._ego_initial_pose_written = False
        recorder._frame_alignment_written = False
        recorder._latest_sensor_pose_world = None
        recorder._latest_sensor_pose_stamp = None
        recorder._sensor_relative_pose_cache = {}
        recorder.deform_monitor_param_root = "/deform_monitor_v2"
        recorder.deform_monitor_config_path = "/tmp/deform_monitor_v2_sim.yaml"
        recorder.controlled_object = "obstacle_block_left_clone_clone"
        recorder.command_frame = "world"
        recorder.linear_velocity = {"x": 0.0, "y": 0.0, "z": 0.002}
        recorder.angular_velocity_deg = {"x": 0.0, "y": 0.0, "z": 0.0}
        recorder.control_axis = {"x": 0.0, "y": 0.0, "z": 1.0}
        recorder.control_start_delay_sec = 8.0
        recorder.control_duration_sec = 20.0
        recorder.scenario_id = "collapse_microdeform_case_01"

        return recorder, fake_rospy, original_rospy

    def _read_manifest(self, recorder):
        return json.loads((recorder.meta_dir / "scenario_manifest.json").read_text())

    def test_allocate_run_directory_increments_sim_run_indices(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        with tempfile.TemporaryDirectory(prefix="sim_experiment_day_") as temp_dir:
            day_dir = pathlib.Path(temp_dir)
            (day_dir / "sim_run_000").mkdir()
            (day_dir / "sim_run_001").mkdir()

            run_dir = module.allocate_run_directory(day_dir)

            self.assertEqual(run_dir, day_dir / "sim_run_002")

    def test_build_frame_alignment_metadata_marks_initial_ego_pose_sim_alignment(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        metadata = module.build_frame_alignment_metadata(
            ego_pose_world={"position": {"x": 1.0, "y": 2.0, "z": 3.0}},
            truth_frame="world",
            algorithm_frame="map",
        )

        self.assertEqual(metadata["truth_frame"], "world")
        self.assertEqual(metadata["algorithm_frame"], "map")
        self.assertEqual(metadata["alignment_mode"], "initial_ego_pose")
        self.assertEqual(metadata["sim_only"], True)

    def test_build_frame_alignment_metadata_includes_explicit_forward_and_inverse_transforms(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        metadata = module.build_frame_alignment_metadata(
            ego_pose_world={
                "position": {"x": 1.0, "y": 2.0, "z": 3.0},
                "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
            },
            truth_frame="world",
            algorithm_frame="camera_init",
        )

        self.assertEqual(
            metadata["world_from_algorithm_transform"]["source_frame"],
            "camera_init",
        )
        self.assertEqual(
            metadata["world_from_algorithm_transform"]["target_frame"],
            "world",
        )
        self.assertEqual(
            metadata["world_from_algorithm_transform"]["pose"]["position"],
            {"x": 1.0, "y": 2.0, "z": 3.0},
        )
        self.assertEqual(
            metadata["algorithm_from_world_transform"]["source_frame"],
            "world",
        )
        self.assertEqual(
            metadata["algorithm_from_world_transform"]["target_frame"],
            "camera_init",
        )
        self.assertEqual(
            metadata["algorithm_from_world_transform"]["pose"]["position"],
            {"x": -1.0, "y": -2.0, "z": -3.0},
        )

    def test_build_frame_alignment_metadata_uses_truth_and_algorithm_reference_pose_pair(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        metadata = module.build_frame_alignment_metadata(
            ego_pose_world={"position": {"x": 99.0, "y": 88.0, "z": 77.0}},
            truth_frame="world",
            algorithm_frame="camera_init",
            truth_reference_frame="base_footprint",
            truth_reference_pose_world={
                "position": {"x": 10.0, "y": 0.0, "z": 0.0},
                "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
            },
            algorithm_reference_frame="body",
            algorithm_reference_pose_algorithm={
                "position": {"x": 1.0, "y": 0.0, "z": 0.0},
                "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
            },
        )

        self.assertEqual(metadata["truth_reference_frame"], "base_footprint")
        self.assertEqual(metadata["algorithm_reference_frame"], "body")
        self.assertEqual(
            metadata["world_from_algorithm_transform"]["pose"]["position"],
            {"x": 9.0, "y": 0.0, "z": 0.0},
        )
        self.assertEqual(
            metadata["truth_reference_pose_world"]["position"],
            {"x": 10.0, "y": 0.0, "z": 0.0},
        )
        self.assertEqual(
            metadata["algorithm_reference_pose_algorithm"]["position"],
            {"x": 1.0, "y": 0.0, "z": 0.0},
        )

    def test_format_tum_line_writes_timestamp_position_and_quaternion(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        line = module.format_tum_line(
            timestamp_sec=12.5,
            position={"x": 1.0, "y": -2.0, "z": 3.25},
            orientation={"x": 0.1, "y": 0.2, "z": 0.3, "w": 0.9},
        )

        self.assertEqual(
            line,
            "12.500000000 1.000000000 -2.000000000 3.250000000 "
            "0.100000000 0.200000000 0.300000000 0.900000000\n",
        )

    def test_write_trajectory_sample_pair_uses_shared_odometry_timestamp(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        with tempfile.TemporaryDirectory(prefix="sim_experiment_recorder_") as temp_dir:
            temp_dir = pathlib.Path(temp_dir)
            gt_path = temp_dir / "truth" / "gt_sensor_world_tum.txt"
            odom_path = temp_dir / "algorithm" / "odom_raw_tum.txt"

            wrote_sample = module.write_tum_sample_pair(
                gt_path=gt_path,
                odom_path=odom_path,
                timestamp_sec=42.5,
                sensor_pose_world={
                    "position": {"x": 1.0, "y": 2.0, "z": 3.0},
                    "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                },
                odom_pose={
                    "position": {"x": 4.0, "y": 5.0, "z": 6.0},
                    "orientation": {"x": 0.1, "y": 0.2, "z": 0.3, "w": 0.9},
                },
            )

            self.assertTrue(wrote_sample)
            self.assertTrue(gt_path.exists())
            self.assertTrue(odom_path.exists())
            self.assertEqual(
                gt_path.read_text(),
                "42.500000000 1.000000000 2.000000000 3.000000000 "
                "0.000000000 0.000000000 0.000000000 1.000000000\n",
            )
            self.assertEqual(
                odom_path.read_text(),
                "42.500000000 4.000000000 5.000000000 6.000000000 "
                "0.100000000 0.200000000 0.300000000 0.900000000\n",
            )

    def test_write_tum_sample_pair_skips_non_finite_pose(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        with tempfile.TemporaryDirectory(prefix="sim_experiment_recorder_") as temp_dir:
            temp_dir = pathlib.Path(temp_dir)
            gt_path = temp_dir / "truth" / "gt_sensor_world_tum.txt"
            odom_path = temp_dir / "algorithm" / "odom_raw_tum.txt"

            wrote_sample = module.write_tum_sample_pair(
                gt_path=gt_path,
                odom_path=odom_path,
                timestamp_sec=42.5,
                sensor_pose_world={
                    "position": {"x": 1.0, "y": float("nan"), "z": 3.0},
                    "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                },
                odom_pose={
                    "position": {"x": 4.0, "y": 5.0, "z": 6.0},
                    "orientation": {"x": 0.1, "y": 0.2, "z": 0.3, "w": 0.9},
                },
            )

            self.assertFalse(wrote_sample)
            self.assertFalse(gt_path.exists())
            self.assertFalse(odom_path.exists())

    def test_pose_dict_is_finite_rejects_nan_components(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        pose = {
            "position": {"x": 1.0, "y": float("nan"), "z": 0.0},
            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        }

        self.assertFalse(module.pose_dict_is_finite(pose))

    def test_pose_dict_is_finite_returns_false_for_malformed_pose_dict(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        pose = {
            "position": None,
            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        }

        self.assertFalse(module.pose_dict_is_finite(pose))

    def test_pose_dict_is_finite_returns_false_for_missing_components(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        self.assertFalse(
            module.pose_dict_is_finite({"position": {}, "orientation": {}})
        )

    def test_pose_dict_is_finite_returns_false_for_non_numeric_components(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        pose = {
            "position": {"x": "bad", "y": 0.0, "z": 0.0},
            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        }

        self.assertFalse(module.pose_dict_is_finite(pose))

    def test_write_tum_sample_pair_returns_false_for_incomplete_pose_dict(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        with tempfile.TemporaryDirectory(prefix="sim_experiment_recorder_") as temp_dir:
            temp_dir = pathlib.Path(temp_dir)
            gt_path = temp_dir / "truth" / "gt_sensor_world_tum.txt"
            odom_path = temp_dir / "algorithm" / "odom_raw_tum.txt"

            wrote_sample = module.write_tum_sample_pair(
                gt_path=gt_path,
                odom_path=odom_path,
                timestamp_sec=42.5,
                sensor_pose_world={
                    "position": {},
                    "orientation": {},
                },
                odom_pose={
                    "position": {"x": 4.0, "y": 5.0, "z": 6.0},
                    "orientation": {"x": 0.1, "y": 0.2, "z": 0.3, "w": 0.9},
                },
            )

            self.assertFalse(wrote_sample)
            self.assertFalse(gt_path.exists())
            self.assertFalse(odom_path.exists())

    def test_write_tum_sample_pair_returns_false_for_non_numeric_sensor_pose(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        with tempfile.TemporaryDirectory(prefix="sim_experiment_recorder_") as temp_dir:
            temp_dir = pathlib.Path(temp_dir)
            gt_path = temp_dir / "truth" / "gt_sensor_world_tum.txt"
            odom_path = temp_dir / "algorithm" / "odom_raw_tum.txt"

            wrote_sample = module.write_tum_sample_pair(
                gt_path=gt_path,
                odom_path=odom_path,
                timestamp_sec=42.5,
                sensor_pose_world={
                    "position": {"x": "bad", "y": 2.0, "z": 3.0},
                    "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                },
                odom_pose={
                    "position": {"x": 4.0, "y": 5.0, "z": 6.0},
                    "orientation": {"x": 0.1, "y": 0.2, "z": 0.3, "w": 0.9},
                },
            )

            self.assertFalse(wrote_sample)
            self.assertFalse(gt_path.exists())
            self.assertFalse(odom_path.exists())

    def test_write_tum_sample_pair_returns_false_for_non_numeric_odom_pose(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        with tempfile.TemporaryDirectory(prefix="sim_experiment_recorder_") as temp_dir:
            temp_dir = pathlib.Path(temp_dir)
            gt_path = temp_dir / "truth" / "gt_sensor_world_tum.txt"
            odom_path = temp_dir / "algorithm" / "odom_raw_tum.txt"

            wrote_sample = module.write_tum_sample_pair(
                gt_path=gt_path,
                odom_path=odom_path,
                timestamp_sec=42.5,
                sensor_pose_world={
                    "position": {"x": 1.0, "y": 2.0, "z": 3.0},
                    "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                },
                odom_pose={
                    "position": {"x": 4.0, "y": "bad", "z": 6.0},
                    "orientation": {"x": 0.1, "y": 0.2, "z": 0.3, "w": 0.9},
                },
            )

            self.assertFalse(wrote_sample)
            self.assertFalse(gt_path.exists())
            self.assertFalse(odom_path.exists())

    def test_compose_pose_dicts_applies_sensor_offset_to_ground_truth_pose(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        composed_pose = module.compose_pose_dicts(
            base_pose={
                "position": {"x": 1.0, "y": 2.0, "z": 3.0},
                "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
            },
            relative_pose={
                "position": {"x": 0.0, "y": 0.0, "z": 0.23375},
                "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
            },
        )

        self.assertEqual(
            composed_pose,
            {
                "position": {"x": 1.0, "y": 2.0, "z": 3.23375},
                "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
            },
        )

    def test_ground_truth_odometry_updates_sensor_pose_cache_from_tf(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        fake_rospy = _FakeRospy()
        original_rospy = module.rospy
        module.rospy = fake_rospy
        try:
            recorder = module.SimExperimentRecorder.__new__(module.SimExperimentRecorder)
            recorder.sensor_frame_name = "livox"
            recorder._tf_listener = _FakeTfListener(
                transforms={
                    ("base_footprint", "livox"): (
                        (0.0, 0.0, 0.23375),
                        (0.0, 0.0, 0.0, 1.0),
                    )
                }
            )
            recorder._latest_sensor_pose_world = None
            recorder._latest_sensor_pose_stamp = None

            msg = SimpleNamespace(
                header=SimpleNamespace(stamp=SimpleNamespace(secs=42, nsecs=500000000)),
                child_frame_id="base_footprint",
                pose=SimpleNamespace(
                    pose=SimpleNamespace(
                        position=SimpleNamespace(x=1.0, y=2.0, z=3.0),
                        orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
                    )
                ),
            )

            recorder._handle_ground_truth_odometry(msg)

            self.assertEqual(
                recorder._latest_sensor_pose_world,
                {
                    "position": {"x": 1.0, "y": 2.0, "z": 3.23375},
                    "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                },
            )
            self.assertEqual(recorder._latest_sensor_pose_stamp, 42.5)
            self.assertEqual(
                recorder._tf_listener.lookups,
                [("base_footprint", "livox", None)],
            )
        finally:
            module.rospy = original_rospy

    def test_odometry_driven_export_waits_for_sensor_pose_then_writes_first_pair(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        fake_rospy = _FakeRospy()
        original_rospy = module.rospy
        module.rospy = fake_rospy
        try:
            with tempfile.TemporaryDirectory(prefix="sim_experiment_recorder_") as temp_dir:
                temp_dir = pathlib.Path(temp_dir)
                recorder = module.SimExperimentRecorder.__new__(module.SimExperimentRecorder)
                recorder._latest_sensor_pose_world = None
                recorder._latest_sensor_pose_stamp = None
                recorder._gt_tum_path = temp_dir / "trajectory" / "gt_sensor_world_tum.txt"
                recorder._odom_tum_path = temp_dir / "trajectory" / "odom_raw_tum.txt"

                odom_msg = SimpleNamespace(
                    header=SimpleNamespace(stamp=SimpleNamespace(secs=42, nsecs=500000000)),
                    pose=SimpleNamespace(
                        pose=SimpleNamespace(
                            position=SimpleNamespace(x=4.0, y=5.0, z=6.0),
                            orientation=SimpleNamespace(x=0.1, y=0.2, z=0.3, w=0.9),
                        )
                    ),
                )

                recorder._handle_odometry(odom_msg)

                self.assertFalse(recorder._gt_tum_path.exists())
                self.assertFalse(recorder._odom_tum_path.exists())
                self.assertEqual(len(fake_rospy.logged_warnings), 1)
                self.assertIn(
                    "waiting for a valid sensor pose cache",
                    fake_rospy.logged_warnings[0][1],
                )

                recorder._latest_sensor_pose_world = {
                    "position": {"x": 1.0, "y": 2.0, "z": 3.0},
                    "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                }
                recorder._latest_sensor_pose_stamp = 42.45

                recorder._handle_odometry(odom_msg)

                self.assertEqual(len(fake_rospy.logged_warnings), 1)
                self.assertEqual(
                    recorder._gt_tum_path.read_text(),
                    "42.500000000 1.000000000 2.000000000 3.000000000 "
                    "0.000000000 0.000000000 0.000000000 1.000000000\n",
                )
                self.assertEqual(
                    recorder._odom_tum_path.read_text(),
                    "42.500000000 4.000000000 5.000000000 6.000000000 "
                    "0.100000000 0.200000000 0.300000000 0.900000000\n",
                )
        finally:
            module.rospy = original_rospy

    def test_odometry_callback_writes_frame_alignment_from_initial_reference_pose_pair(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        fake_rospy = _FakeRospy()
        original_rospy = module.rospy
        module.rospy = fake_rospy
        try:
            recorder = module.SimExperimentRecorder.__new__(module.SimExperimentRecorder)
            recorder.truth_frame = "world"
            recorder.algorithm_frame = "camera_init"
            recorder.meta_dir = pathlib.Path("/tmp/unused_meta_dir")
            recorder._frame_alignment_written = False
            recorder._latest_truth_reference_pose_world = {
                "position": {"x": 10.0, "y": 0.0, "z": 0.0},
                "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
            }
            recorder._latest_truth_reference_pose_stamp = 42.45
            recorder._latest_truth_reference_frame = "base_footprint"
            recorder._latest_sensor_pose_world = None
            recorder._latest_sensor_pose_stamp = None

            captured = {}

            def fake_write_json(path, payload):
                captured["path"] = path
                captured["payload"] = payload

            recorder._write_json = fake_write_json

            odom_msg = SimpleNamespace(
                child_frame_id="body",
                header=SimpleNamespace(stamp=SimpleNamespace(secs=42, nsecs=500000000)),
                pose=SimpleNamespace(
                    pose=SimpleNamespace(
                        position=SimpleNamespace(x=1.0, y=0.0, z=0.0),
                        orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
                    )
                ),
            )

            recorder._handle_odometry(odom_msg)

            self.assertTrue(recorder._frame_alignment_written)
            self.assertEqual(captured["path"], recorder.meta_dir / "frame_alignment.json")
            self.assertEqual(
                captured["payload"]["world_from_algorithm_transform"]["pose"]["position"],
                {"x": 9.0, "y": 0.0, "z": 0.0},
            )
            self.assertEqual(captured["payload"]["truth_reference_frame"], "base_footprint")
            self.assertEqual(captured["payload"]["algorithm_reference_frame"], "body")
        finally:
            module.rospy = original_rospy

    def test_odometry_driven_export_skips_stale_sensor_pose_without_writing_files(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        fake_rospy = _FakeRospy()
        original_rospy = module.rospy
        module.rospy = fake_rospy
        try:
            with tempfile.TemporaryDirectory(prefix="sim_experiment_recorder_") as temp_dir:
                temp_dir = pathlib.Path(temp_dir)
                recorder = module.SimExperimentRecorder.__new__(module.SimExperimentRecorder)
                recorder._latest_sensor_pose_world = {
                    "position": {"x": 1.0, "y": 2.0, "z": 3.0},
                    "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                }
                recorder._latest_sensor_pose_stamp = 41.0
                recorder._gt_tum_path = temp_dir / "trajectory" / "gt_sensor_world_tum.txt"
                recorder._odom_tum_path = temp_dir / "trajectory" / "odom_raw_tum.txt"

                odom_msg = SimpleNamespace(
                    header=SimpleNamespace(stamp=SimpleNamespace(secs=42, nsecs=500000000)),
                    pose=SimpleNamespace(
                        pose=SimpleNamespace(
                            position=SimpleNamespace(x=4.0, y=5.0, z=6.0),
                            orientation=SimpleNamespace(x=0.1, y=0.2, z=0.3, w=0.9),
                        )
                    ),
                )

                recorder._handle_odometry(odom_msg)

                self.assertFalse(recorder._gt_tum_path.exists())
                self.assertFalse(recorder._odom_tum_path.exists())
                self.assertEqual(len(fake_rospy.logged_warnings), 1)
                self.assertIn(
                    "cached sensor pose timestamp is stale",
                    fake_rospy.logged_warnings[0][1],
                )
        finally:
            module.rospy = original_rospy

    def test_odometry_driven_export_skips_zero_stamp_without_writing_files(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        fake_rospy = _FakeRospy()
        original_rospy = module.rospy
        module.rospy = fake_rospy
        try:
            with tempfile.TemporaryDirectory(prefix="sim_experiment_recorder_") as temp_dir:
                temp_dir = pathlib.Path(temp_dir)
                recorder = module.SimExperimentRecorder.__new__(module.SimExperimentRecorder)
                recorder._latest_sensor_pose_world = {
                    "position": {"x": 1.0, "y": 2.0, "z": 3.0},
                    "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                }
                recorder._latest_sensor_pose_stamp = 7.0
                recorder._gt_tum_path = temp_dir / "trajectory" / "gt_sensor_world_tum.txt"
                recorder._odom_tum_path = temp_dir / "trajectory" / "odom_raw_tum.txt"

                odom_msg = SimpleNamespace(
                    header=SimpleNamespace(stamp=SimpleNamespace(secs=0, nsecs=0)),
                    pose=SimpleNamespace(
                        pose=SimpleNamespace(
                            position=SimpleNamespace(x=4.0, y=5.0, z=6.0),
                            orientation=SimpleNamespace(x=0.1, y=0.2, z=0.3, w=0.9),
                        )
                    ),
                )

                recorder._handle_odometry(odom_msg)

                self.assertFalse(recorder._gt_tum_path.exists())
                self.assertFalse(recorder._odom_tum_path.exists())
                self.assertEqual(len(fake_rospy.logged_warnings), 1)
                self.assertIn(
                    "message stamp was invalid",
                    fake_rospy.logged_warnings[0][1],
                )
        finally:
            module.rospy = original_rospy

    def test_module_imports_odometry_message_type_for_runtime_wiring(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        self.assertTrue(hasattr(module, "Odometry"))

    def test_serialize_persistent_risk_regions_includes_track_level_fields(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        msg = SimpleNamespace(
            header=SimpleNamespace(seq=9, stamp=SimpleNamespace(secs=12, nsecs=500000000), frame_id="camera_init"),
            reference_epoch=4,
            regions=[make_persistent_region()],
        )

        payload = module.serialize_persistent_risk_regions(msg)

        self.assertEqual(payload["header"]["seq"], 9)
        self.assertEqual(payload["header"]["frame_id"], "camera_init")
        self.assertEqual(payload["reference_epoch"], 4)
        self.assertEqual(payload["regions"][0]["track_id"], 7)
        self.assertEqual(payload["regions"][0]["state"], 1)
        self.assertEqual(payload["regions"][0]["region_type"], 1)
        self.assertEqual(payload["regions"][0]["center"], {"x": 1.0, "y": 2.0, "z": 3.0})
        self.assertTrue(payload["regions"][0]["confirmed"])

    def test_serializes_object_association_through_alert_outputs(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        point = SimpleNamespace(x=1.0, y=2.0, z=3.0)
        evidence = SimpleNamespace(
            id=7,
            anchor_type=0,
            object_id=31,
            object_id_valid=True,
            object_id_confidence=0.96,
            observed_object_id=31,
            observed_object_id_valid=True,
            observed_object_id_confidence=0.91,
            observed_object_id_support_count=14,
            object_association_state=1,
            obs_state=1,
            mode=1,
            position=point,
            displacement=SimpleNamespace(x=0.01, y=0.0, z=0.0),
            displacement_score=0.8,
            disappearance_score=0.0,
            graph_score=0.7,
            confidence=0.9,
            risk_score=0.8,
            graph_neighbor_count=3,
            observable=True,
            comparable=True,
            active=True,
        )
        persistent = make_persistent_region()
        persistent.object_id = 31
        persistent.object_id_valid = True
        persistent.object_id_confidence = 0.94
        persistent.object_id_ambiguous = False
        persistent.observed_object_id = 31
        persistent.observed_object_id_valid = True
        persistent.observed_object_id_confidence = 0.90
        persistent.observed_object_id_ambiguous = False
        persistent.object_association_state = 1
        persistent.association_consistent_count = 8
        persistent.association_mismatch_count = 0
        persistent.association_mixed_count = 0
        persistent.association_unavailable_count = 2

        evidence_payload = module.serialize_risk_evidence_entry(evidence)
        persistent_payload = module.serialize_persistent_risk_region(persistent)

        self.assertEqual(evidence_payload["object_id"], 31)
        self.assertTrue(evidence_payload["object_id_valid"])
        self.assertEqual(evidence_payload["object_id_confidence"], 0.96)
        self.assertEqual(evidence_payload["observed_object_id"], 31)
        self.assertEqual(evidence_payload["observed_object_id_support_count"], 14)
        self.assertEqual(evidence_payload["object_association_state"], 1)
        self.assertEqual(persistent_payload["object_id"], 31)
        self.assertTrue(persistent_payload["object_id_valid"])
        self.assertEqual(persistent_payload["object_id_confidence"], 0.94)
        self.assertFalse(persistent_payload["object_id_ambiguous"])
        self.assertEqual(persistent_payload["observed_object_id"], 31)
        self.assertEqual(persistent_payload["association_consistent_count"], 8)

    def test_object_observation_stats_are_serialized_and_recorded_before_alignment(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        msg = SimpleNamespace(
            header=SimpleNamespace(
                seq=4,
                stamp=SimpleNamespace(secs=34, nsecs=0),
                frame_id="camera_init",
            ),
            phase=1,
            reference_epoch=4,
            window_start=SimpleNamespace(secs=30, nsecs=0),
            window_end=SimpleNamespace(secs=34, nsecs=0),
            frame_count=5,
            total_point_count=20,
            valid_label_point_count=15,
            invalid_label_point_count=5,
            objects=[
                SimpleNamespace(object_id=17, point_count=8, visible_frame_count=5),
                SimpleNamespace(object_id=23, point_count=7, visible_frame_count=5),
            ],
        )

        payload = module.serialize_object_observation_stats(msg)
        self.assertEqual(payload["reference_epoch"], 4)
        self.assertEqual(payload["window_start"]["sec"], 30.0)
        self.assertEqual(payload["objects"][0]["object_id"], 17)
        self.assertEqual(payload["objects"][1]["point_count"], 7)

        fake_rospy = _FakeRospy(now_sec=35.0)
        original_rospy = module.rospy
        module.rospy = fake_rospy
        try:
            with tempfile.TemporaryDirectory(prefix="sim_experiment_recorder_") as temp_dir:
                recorder = module.SimExperimentRecorder.__new__(module.SimExperimentRecorder)
                recorder.algorithm_dir = pathlib.Path(temp_dir)
                recorder._object_files = {}
                recorder._link_files = {}
                recorder._algorithm_files = {}
                recorder._frame_alignment_written = False
                recorder._handle_object_observation_stats(msg)
                recorder.close()
                records = [
                    json.loads(line)
                    for line in (pathlib.Path(temp_dir) / "object_observation_stats.jsonl")
                    .read_text()
                    .splitlines()
                ]
        finally:
            module.rospy = original_rospy

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["header"]["stamp"]["sec"], 34.0)

    def test_risk_evidence_is_recorded_before_alignment_and_keeps_inactive_rows(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        msg = SimpleNamespace(
            header=SimpleNamespace(
                seq=2,
                stamp=SimpleNamespace(secs=8, nsecs=0),
                frame_id="camera_init",
            ),
            evidences=[SimpleNamespace(anchor_id=17, active=False, risk_score=0.2)],
        )
        fake_rospy = _FakeRospy(now_sec=8.1)
        original_rospy = module.rospy
        module.rospy = fake_rospy
        try:
            with tempfile.TemporaryDirectory(prefix="sim_experiment_recorder_") as temp_dir:
                recorder = module.SimExperimentRecorder.__new__(module.SimExperimentRecorder)
                recorder.algorithm_dir = pathlib.Path(temp_dir)
                recorder._algorithm_files = {}
                recorder._frame_alignment_written = False
                recorder._handle_risk_evidence(msg)
                recorder.close()

                payload = json.loads(
                    (pathlib.Path(temp_dir) / "risk_evidence.jsonl").read_text().strip()
                )
        finally:
            module.rospy = original_rospy

        self.assertEqual(payload["header"]["stamp"]["sec"], 8.0)
        self.assertEqual(len(payload["evidences"]), 1)
        self.assertFalse(payload["evidences"][0]["active"])

    def test_compact_anchor_stream_reconstructs_static_and_dynamic_fields(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        point = lambda x, y, z: SimpleNamespace(x=x, y=y, z=z)
        anchor = SimpleNamespace(
            id=17,
            anchor_type=1,
            object_id=31,
            object_id_valid=True,
            object_id_confidence=0.95,
            object_id_support_count=40,
            center=point(1.0, 2.0, 3.0),
            normal=point(0.0, 0.0, 1.0),
            edge_normal=point(1.0, 0.0, 0.0),
            ref_center=point(1.0, 2.0, 3.0),
            matched_center=point(1.1, 2.0, 3.0),
            matched_delta=point(0.1, 0.0, 0.0),
            disp_mean=[0.1, 0.0, 0.0],
            disp_cov_diag=[1.0e-4] * 3,
            vel_mean=[0.02, 0.0, 0.0],
            observable=True,
            comparable=True,
            significant=True,
            obs_state=1,
            reference_epoch=4,
            reference_stamp=SimpleNamespace(secs=5, nsecs=0),
            reference_origin=0,
        )

        fake_rospy = _FakeRospy(now_sec=10.1)
        original_rospy = module.rospy
        module.rospy = fake_rospy
        try:
            with tempfile.TemporaryDirectory(prefix="sim_experiment_recorder_") as temp_dir:
                temp_dir = pathlib.Path(temp_dir)
                recorder = module.SimExperimentRecorder.__new__(module.SimExperimentRecorder)
                recorder.algorithm_dir = temp_dir / "algorithm"
                recorder.meta_dir = temp_dir / "meta"
                recorder.algorithm_dir.mkdir()
                recorder.meta_dir.mkdir()
                recorder._algorithm_files = {}
                recorder._frame_alignment_written = False
                recorder.record_flush_max_rows = 100
                recorder.record_flush_interval_sec = 60.0

                def emit(seq, stamp, dx):
                    anchor.matched_delta = point(dx, 0.0, 0.0)
                    anchor.disp_mean = [dx, 0.0, 0.0]
                    msg = SimpleNamespace(
                        header=SimpleNamespace(
                            seq=seq,
                            stamp=SimpleNamespace(secs=stamp, nsecs=0),
                            frame_id="camera_init",
                        ),
                        reference_epoch=4,
                        reference_initialized_at=SimpleNamespace(secs=5, nsecs=0),
                        anchors=[anchor],
                        total_anchor_count=30,
                        object_summaries=[
                            SimpleNamespace(
                                object_id=31,
                                total_count=30,
                                comparable_count=1,
                                significant_count=1,
                                plane_count=1,
                                edge_count=0,
                                band_count=0,
                                excluded_not_observable=12,
                                excluded_weak_or_missing=17,
                            )
                        ],
                    )
                    recorder._handle_anchor_states(msg)

                emit(1, 10, 0.1)
                emit(3, 11, 0.2)
                emit(7, 12, 0.3)
                recorder.close()

                catalog = [
                    json.loads(line)
                    for line in (recorder.algorithm_dir / "anchor_catalog.jsonl")
                    .read_text()
                    .splitlines()
                ]
                observations = [
                    json.loads(line)
                    for line in (recorder.algorithm_dir / "anchor_observations.jsonl")
                    .read_text()
                    .splitlines()
                ]
                reconstructed = module.reconstruct_anchor_state_records(
                    catalog, observations
                )
                completion = json.loads((recorder.meta_dir / "run_complete.json").read_text())
        finally:
            module.rospy = original_rospy

        self.assertEqual(len(catalog), 1)
        self.assertEqual(len(observations), 3)
        self.assertEqual(reconstructed[0]["anchors"][0]["anchor_type"], 1)
        self.assertEqual(reconstructed[2]["anchors"][0]["matched_delta"]["x"], 0.3)
        self.assertNotIn("state_cov", observations[0]["anchors"][0])
        self.assertEqual(observations[0]["total_anchor_count"], 30)
        self.assertEqual(
            observations[0]["object_summaries"],
            [
                {
                    "object_id": 31,
                    "total_count": 30,
                    "comparable_count": 1,
                    "significant_count": 1,
                    "plane_count": 1,
                    "edge_count": 0,
                    "band_count": 0,
                    "excluded_not_observable": 12,
                    "excluded_weak_or_missing": 17,
                }
            ],
        )
        self.assertEqual(completion["streams"]["anchor_catalog"]["row_count"], 1)
        self.assertEqual(completion["streams"]["anchor_observations"]["row_count"], 3)
        self.assertEqual(
            completion["streams"]["anchor_observations"]["estimated_drop_count"],
            1,
        )
        self.assertTrue(completion["clean_shutdown"])

    def test_direct_anchor_serialization_matches_compact_schema(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        point = lambda x, y, z: SimpleNamespace(x=x, y=y, z=z)
        anchor = SimpleNamespace(
            id=17,
            anchor_type=2,
            object_id=31,
            object_id_valid=True,
            object_id_confidence=0.95,
            object_id_support_count=40,
            center=point(1.0, 2.0, 3.0),
            normal=point(0.0, 0.0, 1.0),
            edge_normal=point(1.0, 0.0, 0.0),
            ref_center=point(1.0, 2.0, 3.0),
            matched_delta=point(0.1, 0.0, 0.0),
            disp_cov_diag=[2.0e-4, 2.0e-4, 2.0e-4],
            reference_epoch=4,
            reference_stamp=SimpleNamespace(secs=5, nsecs=0),
            reference_origin=0,
        )

        full = module.serialize_anchor_state(anchor)
        expected_static, expected_dynamic = module.split_serialized_anchor_state(full)

        self.assertEqual(module.serialize_anchor_static_state(anchor), expected_static)
        self.assertEqual(
            module.serialize_anchor_dynamic_state(anchor),
            expected_dynamic,
        )
        self.assertNotIn("state_cov", expected_dynamic)
        self.assertEqual(
            module.serialize_anchor_dynamic_state(anchor)["disp_cov_diag"],
            [2.0e-4, 2.0e-4, 2.0e-4],
        )

    def test_anchor_processing_worker_drains_accepted_messages_before_close(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        recorder = module.SimExperimentRecorder.__new__(module.SimExperimentRecorder)
        recorder.anchor_processing_queue_size = 4
        recorder.anchor_processing_enqueue_timeout_sec = 0.5
        recorder._anchor_processing_thread = None
        recorder._anchor_processing_queue = None
        recorder._anchor_processing_error = None
        recorder._anchor_processing_stats = {
            "enqueued_message_count": 0,
            "processed_message_count": 0,
            "max_queue_depth": 0,
            "queue_full_error_count": 0,
            "processing_error_count": 0,
        }
        processed = []
        recorder._record_anchor_states = lambda msg, recorded_at: processed.append(
            (msg, recorded_at)
        )

        recorder._start_anchor_processing_worker()
        recorder._enqueue_anchor_states("frame-1", {"sec": 1.0})
        recorder._enqueue_anchor_states("frame-2", {"sec": 2.0})
        recorder._close_anchor_processing_worker()

        self.assertEqual(
            processed,
            [("frame-1", {"sec": 1.0}), ("frame-2", {"sec": 2.0})],
        )
        self.assertEqual(
            recorder._anchor_processing_stats["processed_message_count"], 2
        )
        self.assertIsNone(recorder._anchor_processing_thread)
        self.assertIsNone(recorder._anchor_processing_queue)

    def test_algorithm_write_after_close_is_ignored(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        fake_rospy = _FakeRospy(now_sec=10.0)
        original_rospy = module.rospy
        module.rospy = fake_rospy
        try:
            with tempfile.TemporaryDirectory(prefix="sim_experiment_recorder_") as temp_dir:
                temp_dir = pathlib.Path(temp_dir)
                recorder = module.SimExperimentRecorder.__new__(module.SimExperimentRecorder)
                recorder.algorithm_dir = temp_dir / "algorithm"
                recorder.meta_dir = temp_dir / "meta"
                recorder.algorithm_dir.mkdir()
                recorder.meta_dir.mkdir()
                recorder._algorithm_files = {}
                recorder._closed = False
                recorder.record_flush_max_rows = 100
                recorder.record_flush_interval_sec = 60.0

                self.assertTrue(
                    recorder._append_jsonl(
                        "anchors", "anchors.jsonl", {"recorded_at_sec": 1.0}
                    )
                )
                recorder.close()
                self.assertFalse(
                    recorder._append_jsonl(
                        "anchors", "anchors.jsonl", {"recorded_at_sec": 2.0}
                    )
                )

                records = (
                    recorder.algorithm_dir / "anchors.jsonl"
                ).read_text().splitlines()
                self.assertEqual(len(records), 1)
        finally:
            module.rospy = original_rospy

    def test_close_called_inside_recording_callback_does_not_deadlock(self):
        module = load_module_if_exists()
        recorder = module.SimExperimentRecorder.__new__(module.SimExperimentRecorder)
        recorder._callback_condition = threading.Condition()
        recorder._active_callbacks = 0
        recorder._closed = False
        recorder._subscribers = []
        finished = []
        recorder._finish_close = lambda: finished.append(True)

        @module.recording_callback
        def callback(instance):
            instance.close()

        worker = threading.Thread(target=lambda: callback(recorder), daemon=True)
        worker.start()
        worker.join(timeout=0.5)

        self.assertFalse(worker.is_alive(), "close() waited for its own callback")
        self.assertEqual(finished, [True])
        self.assertTrue(recorder._closed)

    def test_callback_already_in_progress_can_finish_writing_during_close(self):
        module = load_module_if_exists()
        fake_rospy = _FakeRospy(now_sec=10.0)
        original_rospy = module.rospy
        module.rospy = fake_rospy
        try:
            with tempfile.TemporaryDirectory(prefix="sim_experiment_recorder_") as temp_dir:
                recorder = module.SimExperimentRecorder.__new__(module.SimExperimentRecorder)
                recorder.algorithm_dir = pathlib.Path(temp_dir) / "algorithm"
                recorder.algorithm_dir.mkdir()
                recorder._callback_condition = threading.Condition()
                recorder._active_callbacks = 0
                recorder._closed = False
                recorder._subscribers = []
                recorder._algorithm_files = {}
                recorder.record_flush_max_rows = 1
                recorder.record_flush_interval_sec = 60.0
                recorder._finish_close = lambda: None
                entered = threading.Event()
                release = threading.Event()
                write_results = []

                @module.recording_callback
                def callback(instance):
                    entered.set()
                    release.wait(timeout=1.0)
                    write_results.append(
                        instance._append_jsonl(
                            "anchors", "anchors.jsonl", {"recorded_at_sec": 1.0}
                        )
                    )

                worker = threading.Thread(target=lambda: callback(recorder), daemon=True)
                worker.start()
                self.assertTrue(entered.wait(timeout=0.5))
                closer = threading.Thread(target=recorder.close, daemon=True)
                closer.start()
                release.set()
                worker.join(timeout=0.5)
                closer.join(timeout=0.5)

                self.assertFalse(worker.is_alive())
                self.assertFalse(closer.is_alive())
                self.assertEqual(write_results, [True])
                self.assertTrue(recorder._closed)
                for handle in recorder._algorithm_files.values():
                    handle.close()
        finally:
            module.rospy = original_rospy

    def test_close_discards_only_incomplete_trailing_algorithm_frame(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        fake_rospy = _FakeRospy(now_sec=20.0)
        original_rospy = module.rospy
        module.rospy = fake_rospy
        try:
            with tempfile.TemporaryDirectory(prefix="sim_experiment_recorder_") as temp_dir:
                temp_dir = pathlib.Path(temp_dir)
                recorder = module.SimExperimentRecorder.__new__(module.SimExperimentRecorder)
                recorder.algorithm_dir = temp_dir / "algorithm"
                recorder.meta_dir = temp_dir / "meta"
                recorder.algorithm_dir.mkdir()
                recorder.meta_dir.mkdir()
                recorder._algorithm_files = {}
                recorder._closed = False
                recorder.record_flush_max_rows = 100
                recorder.record_flush_interval_sec = 60.0

                def append(stream_name, stamp):
                    payload = {
                        "header": {
                            "seq": int(stamp),
                            "stamp": {"sec": float(stamp), "secs": int(stamp), "nsecs": 0},
                        }
                    }
                    if stream_name == "object_observation_stats":
                        payload["phase"] = 1
                    recorder._append_jsonl(
                        stream_name,
                        module.FRAME_COMMIT_STREAM_FILES[stream_name],
                        payload,
                    )

                for stamp in (1, 2):
                    for stream_name in module.FRAME_COMMIT_STREAM_FILES:
                        append(stream_name, stamp)

                for stream_name in module.FRAME_COMMIT_STREAM_FILES:
                    if stream_name not in ("processing_stamps", "anchor_observations"):
                        append(stream_name, 3)

                recorder.close()

                for stream_name, filename in module.FRAME_COMMIT_STREAM_FILES.items():
                    records = [
                        json.loads(line)
                        for line in (recorder.algorithm_dir / filename).read_text().splitlines()
                    ]
                    self.assertEqual(
                        [record["header"]["stamp"]["sec"] for record in records],
                        [1.0, 2.0],
                        stream_name,
                    )

                completion = json.loads(
                    (recorder.meta_dir / "run_complete.json").read_text()
                )
                self.assertEqual(
                    completion["frame_commit"]["last_complete_stamp_sec"],
                    2.0,
                )
                self.assertEqual(
                    completion["frame_commit"]["discarded_incomplete_suffix_rows"],
                    {
                        stream_name: 1
                        for stream_name in module.FRAME_COMMIT_STREAM_FILES
                        if stream_name not in ("processing_stamps", "anchor_observations")
                    },
                )
                self.assertEqual(
                    completion["streams"]["clusters"]["row_count"],
                    2,
                )
        finally:
            module.rospy = original_rospy

    def test_frame_commit_does_not_hide_an_interior_missing_stream_row(self):
        module = load_module_if_exists()
        records_by_stream = {}
        for stream_name in module.FRAME_COMMIT_STREAM_FILES:
            stamps = (1.0, 3.0) if stream_name == "clusters" else (1.0, 2.0, 3.0)
            records_by_stream[stream_name] = [
                {
                    "stamp_key": module.frame_commit_stamp_key(stamp),
                    "phase": 1,
                    "start_offset": index,
                }
                for index, stamp in enumerate(stamps)
            ]

        plan = module.build_frame_commit_plan(records_by_stream)

        self.assertEqual(plan["last_complete_stamp_sec"], 3.0)
        self.assertEqual(plan["trim_counts"], {})

    def test_compact_anchor_stream_keeps_visible_count_dynamic(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        static, dynamic = module.split_serialized_anchor_state(
            {"id": 5, "anchor_type": 0, "visible_count": 21}
        )

        self.assertNotIn("visible_count", static)
        self.assertEqual(dynamic["visible_count"], 21)

    def test_serialize_anchor_states_preserves_reference_lifecycle_fields(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        point = lambda x, y, z: SimpleNamespace(x=x, y=y, z=z)
        anchor = SimpleNamespace(
            id=17,
            anchor_type=2,
            object_id=31,
            object_id_valid=True,
            object_id_confidence=0.96,
            object_id_support_count=48,
            center=point(1.0, 2.0, 3.0),
            normal=point(0.0, 0.0, 1.0),
            edge_normal=point(1.0, 0.0, 0.0),
            visible_count=20,
            point_count=65,
            ref_quality=0.91,
            covariance_quality=0.87,
            type_stability=0.84,
            shape_linearity=0.18,
            shape_planarity=0.73,
            shape_scattering=0.09,
            observation_support_count=11,
            edge_support_count=7,
            current_shape_linearity=0.81,
            current_shape_planarity=0.14,
            current_shape_scattering=0.05,
            edge_direction_angle_deg=8.0,
            edge_geometry_stability=0.78,
            edge_geometry_valid=True,
            scalar_count=2,
            scalar_type=[0, 1, 3],
            scalar_z=[0.004, 0.012, 0.0],
            scalar_r=[1.0e-4, 2.0e-4, 0.0],
            scalar_valid=[True, True, False],
            ref_center=point(1.0, 2.0, 3.0),
            matched_center=point(1.4, 2.1, 3.0),
            matched_delta=point(0.4, 0.1, 0.0),
            predicted_center=point(1.38, 2.08, 3.0),
            predicted_displacement=point(0.38, 0.08, 0.0),
            reacquisition_score=0.91,
            reacquisition_innovation_norm=0.022,
            disp_mean=[0.39, 0.11, 0.0],
            disp_cov_diag=[1.0e-4] * 3,
            vel_mean=[0.02, 0.01, 0.0],
            dof_obs=2,
            chi2_stat=12.5,
            disp_norm=0.405,
            disp_normal=0.0,
            disp_edge=0.39,
            cmp_score=0.92,
            cusum_score=5.0,
            directional_strength=0.06,
            directional_coherence=0.75,
            directional_persistent=True,
            instantaneous_displacement_evidence=True,
            persistent_candidate=True,
            cluster_member=True,
            graph_candidate=False,
            graph_neighbor_count=3,
            graph_coherent_score=0.82,
            graph_temporal_score=0.88,
            graph_persistence_score=1.2,
            local_bg_count=8,
            local_contrast_score=4.2,
            local_rel_norm=0.018,
            local_rel_normal=0.007,
            local_rel_edge=0.015,
            plane_bg_count=3,
            plane_contrast_score=2.8,
            plane_rel_norm=0.016,
            plane_rel_normal=0.006,
            plane_rel_edge=0.013,
            permanent_deformed=True,
            permanent_displacement=[0.39, 0.11, 0.0],
            comparable=True,
            observable=True,
            significant=True,
            reacquired=True,
            obs_state=1,
            detection_mode=1,
            disappearance_score=0.0,
            reference_epoch=4,
            reference_stamp=SimpleNamespace(secs=42, nsecs=500000000),
            reference_origin=0,
        )
        message = SimpleNamespace(
            header=SimpleNamespace(
                seq=3,
                frame_id="camera_init",
                stamp=SimpleNamespace(secs=50, nsecs=0),
            ),
            reference_epoch=4,
            reference_initialized_at=SimpleNamespace(secs=42, nsecs=500000000),
            anchors=[anchor],
            total_anchor_count=42,
            object_summaries=[
                SimpleNamespace(
                    object_id=2,
                    total_count=42,
                    comparable_count=17,
                    significant_count=5,
                    plane_count=10,
                    edge_count=4,
                    band_count=3,
                    excluded_not_observable=8,
                    excluded_weak_or_missing=17,
                )
            ],
        )

        serialized = module.serialize_anchor_states(message)

        self.assertEqual(serialized["total_anchor_count"], 42)
        self.assertEqual(
            serialized["object_summaries"],
            [
                {
                    "object_id": 2,
                    "total_count": 42,
                    "comparable_count": 17,
                    "significant_count": 5,
                    "plane_count": 10,
                    "edge_count": 4,
                    "band_count": 3,
                    "excluded_not_observable": 8,
                    "excluded_weak_or_missing": 17,
                }
            ],
        )
        self.assertEqual(serialized["reference_epoch"], 4)
        self.assertEqual(serialized["reference_initialized_at"]["sec"], 42.5)
        self.assertEqual(serialized["anchors"][0]["obs_state"], 1)
        self.assertTrue(serialized["anchors"][0]["observable"])
        self.assertEqual(
            serialized["anchors"][0]["matched_center"],
            {"x": 1.4, "y": 2.1, "z": 3.0},
        )
        self.assertEqual(
            serialized["anchors"][0]["matched_delta"],
            {"x": 0.4, "y": 0.1, "z": 0.0},
        )
        self.assertEqual(serialized["anchors"][0]["reference_epoch"], 4)
        self.assertEqual(serialized["anchors"][0]["reference_stamp"]["sec"], 42.5)
        self.assertEqual(serialized["anchors"][0]["reference_origin"], 0)
        self.assertEqual(serialized["anchors"][0]["object_id"], 31)
        self.assertTrue(serialized["anchors"][0]["object_id_valid"])
        self.assertEqual(serialized["anchors"][0]["object_id_confidence"], 0.96)
        self.assertEqual(serialized["anchors"][0]["object_id_support_count"], 48)
        self.assertEqual(
            serialized["anchors"][0]["normal"],
            {"x": 0.0, "y": 0.0, "z": 1.0},
        )
        self.assertEqual(
            serialized["anchors"][0]["edge_normal"],
            {"x": 1.0, "y": 0.0, "z": 0.0},
        )
        self.assertEqual(serialized["anchors"][0]["visible_count"], 20)
        self.assertEqual(serialized["anchors"][0]["point_count"], 65)
        self.assertEqual(serialized["anchors"][0]["ref_quality"], 0.91)
        self.assertEqual(serialized["anchors"][0]["covariance_quality"], 0.87)
        self.assertEqual(serialized["anchors"][0]["type_stability"], 0.84)
        self.assertEqual(serialized["anchors"][0]["shape_linearity"], 0.18)
        self.assertEqual(serialized["anchors"][0]["shape_planarity"], 0.73)
        self.assertEqual(serialized["anchors"][0]["shape_scattering"], 0.09)
        self.assertEqual(serialized["anchors"][0]["observation_support_count"], 11)
        self.assertEqual(serialized["anchors"][0]["edge_support_count"], 7)
        self.assertTrue(serialized["anchors"][0]["edge_geometry_valid"])
        self.assertEqual(serialized["anchors"][0]["scalar_z"], [0.004, 0.012, 0.0])
        self.assertEqual(
            serialized["anchors"][0]["predicted_center"],
            {"x": 1.38, "y": 2.08, "z": 3.0},
        )
        self.assertEqual(serialized["anchors"][0]["reacquisition_innovation_norm"], 0.022)
        self.assertEqual(serialized["anchors"][0]["disp_cov_diag"], [1.0e-4] * 3)
        self.assertEqual(serialized["anchors"][0]["vel_mean"], [0.02, 0.01, 0.0])
        self.assertNotIn("state_cov", serialized["anchors"][0])
        self.assertEqual(serialized["anchors"][0]["dof_obs"], 2)
        self.assertEqual(serialized["anchors"][0]["chi2_stat"], 12.5)
        self.assertEqual(serialized["anchors"][0]["disp_edge"], 0.39)
        self.assertEqual(serialized["anchors"][0]["cmp_score"], 0.92)
        self.assertEqual(serialized["anchors"][0]["directional_strength"], 0.06)
        self.assertEqual(serialized["anchors"][0]["directional_coherence"], 0.75)
        self.assertTrue(serialized["anchors"][0]["directional_persistent"])
        self.assertTrue(
            serialized["anchors"][0]["instantaneous_displacement_evidence"]
        )
        self.assertTrue(serialized["anchors"][0]["cluster_member"])
        self.assertTrue(serialized["anchors"][0]["permanent_deformed"])

    def test_handle_persistent_risk_regions_writes_empty_jsonl_after_alignment_ready(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        fake_rospy = _FakeRospy(now_sec=222.5)
        original_rospy = module.rospy
        module.rospy = fake_rospy
        try:
            with tempfile.TemporaryDirectory(prefix="sim_experiment_recorder_") as temp_dir:
                temp_dir = pathlib.Path(temp_dir)
                recorder = module.SimExperimentRecorder.__new__(module.SimExperimentRecorder)
                recorder.algorithm_dir = temp_dir / "algorithm"
                recorder.algorithm_dir.mkdir(parents=True)
                recorder._object_files = {}
                recorder._link_files = {}
                recorder._algorithm_files = {}
                recorder._frame_alignment_written = True

                msg = SimpleNamespace(
                    header=SimpleNamespace(seq=3, stamp=SimpleNamespace(secs=7, nsecs=0), frame_id="camera_init"),
                    regions=[],
                )

                recorder._handle_persistent_risk_regions(msg)

                jsonl_path = recorder.algorithm_dir / "persistent_risk_regions.jsonl"
                self.assertTrue(jsonl_path.exists())
                payload = json.loads(jsonl_path.read_text().strip())
                self.assertEqual(payload["header"]["seq"], 3)
                self.assertEqual(payload["regions"], [])
                self.assertEqual(payload["recorded_at"]["sec"], 222.5)
                recorder.close()
        finally:
            module.rospy = original_rospy

    def test_handle_persistent_risk_regions_writes_track_lifecycle_events(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        fake_rospy = _FakeRospy(now_sec=300.0)
        original_rospy = module.rospy
        module.rospy = fake_rospy
        try:
            with tempfile.TemporaryDirectory(prefix="sim_experiment_recorder_") as temp_dir:
                temp_dir = pathlib.Path(temp_dir)
                recorder = module.SimExperimentRecorder.__new__(module.SimExperimentRecorder)
                recorder.algorithm_dir = temp_dir / "algorithm"
                recorder.algorithm_dir.mkdir(parents=True)
                recorder._object_files = {}
                recorder._link_files = {}
                recorder._algorithm_files = {}
                recorder._frame_alignment_written = True
                recorder._persistent_track_cache = {}

                first_msg = SimpleNamespace(
                    header=SimpleNamespace(
                        seq=1,
                        stamp=SimpleNamespace(secs=10, nsecs=0),
                        frame_id="camera_init",
                    ),
                    reference_epoch=1,
                    regions=[make_persistent_region(track_id=7, state=0, confirmed=False)],
                )
                second_msg = SimpleNamespace(
                    header=SimpleNamespace(
                        seq=2,
                        stamp=SimpleNamespace(secs=11, nsecs=0),
                        frame_id="camera_init",
                    ),
                    reference_epoch=1,
                    regions=[make_persistent_region(track_id=7, state=1, confirmed=True)],
                )
                reset_msg = SimpleNamespace(
                    header=SimpleNamespace(
                        seq=3,
                        stamp=SimpleNamespace(secs=12, nsecs=0),
                        frame_id="camera_init",
                    ),
                    reference_epoch=2,
                    regions=[make_persistent_region(track_id=7, state=0, confirmed=False)],
                )

                recorder._handle_persistent_risk_regions(first_msg)
                recorder._handle_persistent_risk_regions(second_msg)
                recorder._handle_persistent_risk_regions(reset_msg)

                events_path = recorder.algorithm_dir / "persistent_track_events.jsonl"
                self.assertTrue(events_path.exists())
                events = [
                    json.loads(line)
                    for line in events_path.read_text().splitlines()
                    if line.strip()
                ]
                event_types = [event["event_type"] for event in events]
                self.assertEqual(event_types.count("track_created"), 2)
                self.assertIn("frame_status", event_types)
                self.assertIn("state_transition", event_types)
                self.assertIn("first_confirmed", event_types)

                frame_events = [
                    event for event in events if event["event_type"] == "frame_status"
                ]
                self.assertEqual(len(frame_events), 3)
                self.assertEqual(frame_events[0]["lifecycle"]["first_seen"]["sec"], 10.0)
                self.assertIsNone(frame_events[0]["lifecycle"]["first_confirmed"])
                self.assertEqual(frame_events[1]["lifecycle"]["first_confirmed"]["sec"], 11.0)
                self.assertEqual(frame_events[2]["reference_epoch"], 2)
                self.assertEqual(frame_events[2]["lifecycle"]["first_seen"]["sec"], 12.0)
                self.assertEqual(frame_events[1]["confirmed"], True)
                self.assertEqual(frame_events[1]["state_name"], "CONFIRMED")
                self.assertEqual(frame_events[1]["support_mass"], 3.0)
                recorder.close()
        finally:
            module.rospy = original_rospy

    def test_serialize_motion_clusters_preserves_cluster_level_displacement(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        msg = SimpleNamespace(
            header=SimpleNamespace(
                seq=5,
                stamp=SimpleNamespace(secs=13, nsecs=250000000),
                frame_id="camera_init",
            ),
            clusters=[make_motion_cluster()],
        )

        payload = module.serialize_motion_clusters(msg)

        self.assertEqual(payload["header"]["seq"], 5)
        self.assertEqual(payload["clusters"][0]["id"], 3)
        self.assertEqual(payload["clusters"][0]["anchor_ids"], [1, 2, 3])
        self.assertEqual(payload["clusters"][0]["disp_mean"], [0.01, 0.0, 0.0])
        self.assertEqual(payload["clusters"][0]["support_count"], 9)

    def test_handle_clusters_writes_cluster_jsonl_after_alignment_ready(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        fake_rospy = _FakeRospy(now_sec=250.0)
        original_rospy = module.rospy
        module.rospy = fake_rospy
        try:
            with tempfile.TemporaryDirectory(prefix="sim_experiment_recorder_") as temp_dir:
                temp_dir = pathlib.Path(temp_dir)
                recorder = module.SimExperimentRecorder.__new__(module.SimExperimentRecorder)
                recorder.algorithm_dir = temp_dir / "algorithm"
                recorder.algorithm_dir.mkdir(parents=True)
                recorder._object_files = {}
                recorder._link_files = {}
                recorder._algorithm_files = {}
                recorder._frame_alignment_written = True

                msg = SimpleNamespace(
                    header=SimpleNamespace(
                        seq=8,
                        stamp=SimpleNamespace(secs=20, nsecs=0),
                        frame_id="camera_init",
                    ),
                    clusters=[make_motion_cluster(cluster_id=11)],
                )

                recorder._handle_clusters(msg)

                clusters_path = recorder.algorithm_dir / "clusters.jsonl"
                self.assertTrue(clusters_path.exists())
                payload = json.loads(clusters_path.read_text().strip())
                self.assertEqual(payload["clusters"][0]["id"], 11)
                self.assertEqual(payload["recorded_at"]["sec"], 250.0)
                recorder.close()
        finally:
            module.rospy = original_rospy

    def test_recorder_initializes_trajectory_state_and_odometry_subscription(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        with tempfile.TemporaryDirectory(prefix="sim_experiment_recorder_") as temp_dir:
            temp_dir = pathlib.Path(temp_dir)
            fake_rospy = _FakeRospy(
                params={
                    "~output_root": str(temp_dir),
                    "~truth_frame": "world",
                    "~algorithm_frame": "camera_init",
                    "~ego_model_name": "mid360_fastlio",
                    "~model_states_topic": "/gazebo/model_states",
                    "~link_states_topic": "/gazebo/link_states",
                    "~ground_truth_odometry_topic": "/ground_truth/odom",
                    "~risk_evidence_topic": "/deform/risk_evidence",
                    "~risk_regions_topic": "/deform/risk_regions",
                    "~persistent_risk_regions_topic": "/deform/persistent_risk_regions",
                    "~structure_motions_topic": "/deform/structure_motions",
                    "~odometry_topic": "/Odometry",
                    "~sensor_scoped_link_name": "mid360_fastlio::mid360_link",
                    "~sensor_frame_name": "livox",
                    "~gt_tum_filename": "gt_sensor_world_tum.txt",
                    "~odom_tum_filename": "odom_raw_tum.txt",
                    "~scenario_id": "collapse_microdeform_case_01",
                    "~experiment_factors": {
                        "scene_id": "collapse_world_v3",
                        "visibility_condition": "partial_occlusion",
                        "repeat_index": 1,
                    },
                    "~object_metadata": {
                        "obstacle_block_left_clone_clone": {
                            "shape": "rectangular_panel",
                            "size_class": "large",
                            "visibility_condition": "partial_occlusion",
                        }
                    },
                    "~controlled_object": "obstacle_block_left_clone_clone",
                    "~command_frame": "world",
                    "~linear_velocity_x": "0.0",
                    "~linear_velocity_y": "0.0",
                    "~linear_velocity_z": "0.002",
                    "~control_axis_x": "0.0",
                    "~control_axis_y": "0.0",
                    "~control_axis_z": "1.0",
                    "~control_start_delay_sec": "8.0",
                    "~control_duration_sec": "20.0",
                }
            )
            original_rospy = module.rospy
            original_tf = getattr(module, "tf", None)
            original_model_states = module.ModelStates
            original_link_states = module.LinkStates
            original_persistent_risk_regions = module.PersistentRiskRegions
            original_motion_clusters = getattr(module, "MotionClusters", None)
            original_risk_evidence_array = module.RiskEvidenceArray
            original_risk_regions = module.RiskRegions
            original_structure_motions = module.StructureMotions
            module.rospy = fake_rospy
            module.tf = SimpleNamespace(TransformListener=lambda: object())
            module.ModelStates = object
            module.LinkStates = object
            module.PersistentRiskRegions = object
            module.MotionClusters = object
            module.RiskEvidenceArray = object
            module.RiskRegions = object
            module.StructureMotions = object
            recorder = None
            try:
                recorder = module.SimExperimentRecorder()

                ablation_manifest_path = recorder.meta_dir / "ablation_manifest.json"
                config_snapshot_path = recorder.meta_dir / "config_snapshot.json"
                scenario_manifest_path = recorder.meta_dir / "scenario_manifest.json"

                self.assertEqual(recorder.ground_truth_odometry_topic, "/ground_truth/odom")
                self.assertEqual(recorder.odometry_topic, "/Odometry")
                self.assertEqual(
                    recorder.sensor_scoped_link_name, "mid360_fastlio::mid360_link"
                )
                self.assertEqual(
                    recorder.persistent_risk_regions_topic,
                    "/deform/persistent_risk_regions",
                )
                self.assertEqual(recorder.sensor_frame_name, "livox")
                self.assertEqual(recorder.gt_tum_filename, "gt_sensor_world_tum.txt")
                self.assertEqual(recorder.odom_tum_filename, "odom_raw_tum.txt")
                self.assertEqual(recorder.trajectory_dir, recorder.run_dir / "trajectory")
                self.assertEqual(recorder._gt_tum_path, recorder.trajectory_dir / "gt_sensor_world_tum.txt")
                self.assertEqual(recorder._odom_tum_path, recorder.trajectory_dir / "odom_raw_tum.txt")
                self.assertIsNone(recorder._latest_sensor_pose_world)
                self.assertIsNone(recorder._latest_sensor_pose_stamp)
                self.assertEqual(recorder._sensor_relative_pose_cache, {})
                self.assertIsNotNone(recorder._tf_listener)
                self.assertIn("/Odometry", [sub.topic for sub in fake_rospy.subscribers])
                self.assertIn("/ground_truth/odom", [sub.topic for sub in fake_rospy.subscribers])
                self.assertIn("/deform/persistent_risk_regions", [sub.topic for sub in fake_rospy.subscribers])
                self.assertTrue(
                    any(
                        sub.topic == "/Odometry"
                        and getattr(sub.callback, "__name__", "") == "_handle_odometry"
                        for sub in fake_rospy.subscribers
                    )
                )
                self.assertTrue(
                    any(
                        sub.topic == "/ground_truth/odom"
                        and getattr(sub.callback, "__name__", "") == "_handle_ground_truth_odometry"
                        for sub in fake_rospy.subscribers
                    )
                )
                self.assertTrue(
                    any(
                        sub.topic == "/deform/persistent_risk_regions"
                        and getattr(sub.callback, "__name__", "") == "_handle_persistent_risk_regions"
                        for sub in fake_rospy.subscribers
                    )
                )
                self.assertTrue(ablation_manifest_path.exists())
                self.assertTrue(config_snapshot_path.exists())
                self.assertTrue(scenario_manifest_path.exists())
                scenario_manifest = json.loads(scenario_manifest_path.read_text())
                self.assertEqual(
                    scenario_manifest["experiment_factors"]["scene_id"],
                    "collapse_world_v3",
                )
                self.assertEqual(
                    scenario_manifest["object_metadata"]
                    ["obstacle_block_left_clone_clone"]["shape"],
                    "rectangular_panel",
                )
            finally:
                if recorder is not None:
                    recorder.close()
                module.rospy = original_rospy
                module.tf = original_tf
                module.ModelStates = original_model_states
                module.LinkStates = original_link_states
                module.PersistentRiskRegions = original_persistent_risk_regions
                module.MotionClusters = original_motion_clusters
                module.RiskEvidenceArray = original_risk_evidence_array
                module.RiskRegions = original_risk_regions
                module.StructureMotions = original_structure_motions

    def test_recorder_writes_ablation_manifest_and_config_snapshot(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        deform_monitor_params = {
            "deform_monitor": {
                "covariance": {"alpha_xi": 2.0},
                "background_bias": {"enable": True},
                "imm": {
                    "enable_model_competition": True,
                    "enable_type_constraint": True,
                },
                "significance": {"enable_cusum": True},
                "directional_motion": {"enable": True},
                "persistent_risk": {"min_confirmed_mean_risk": 0.65},
                "ablation": {
                    "variant": "single_model_ekf_no_drift",
                    "disable_covariance_inflation": True,
                    "disable_type_constraint": False,
                    "single_model_ekf": True,
                    "disable_cusum": False,
                    "disable_directional_accumulation": False,
                    "disable_drift_compensation": True,
                },
            }
        }

        with tempfile.TemporaryDirectory(prefix="sim_experiment_recorder_") as temp_dir:
            temp_dir = pathlib.Path(temp_dir)
            fake_rospy = _FakeRospy(
                params={
                    "~output_root": str(temp_dir),
                    "~truth_frame": "world",
                    "~algorithm_frame": "camera_init",
                    "~ego_model_name": "mid360_fastlio",
                    "~model_states_topic": "/gazebo/model_states",
                    "~link_states_topic": "/gazebo/link_states",
                    "~ground_truth_odometry_topic": "/ground_truth/odom",
                    "~risk_evidence_topic": "/deform/risk_evidence",
                    "~risk_regions_topic": "/deform/risk_regions",
                    "~persistent_risk_regions_topic": "/deform/persistent_risk_regions",
                    "~structure_motions_topic": "/deform/structure_motions",
                    "~odometry_topic": "/Odometry",
                    "~sensor_scoped_link_name": "mid360_fastlio::mid360_link",
                    "~sensor_frame_name": "livox",
                    "~gt_tum_filename": "gt_sensor_world_tum.txt",
                    "~odom_tum_filename": "odom_raw_tum.txt",
                    "~deform_monitor_param_root": "/deform_monitor_v2",
                    "~deform_monitor_config_path": "/tmp/deform_monitor_v2_sim.yaml",
                    "/deform_monitor_v2": deform_monitor_params,
                }
            )
            original_rospy = module.rospy
            original_tf = getattr(module, "tf", None)
            original_model_states = module.ModelStates
            original_link_states = module.LinkStates
            original_persistent_risk_regions = module.PersistentRiskRegions
            original_risk_evidence_array = module.RiskEvidenceArray
            original_risk_regions = module.RiskRegions
            original_structure_motions = module.StructureMotions
            module.rospy = fake_rospy
            module.tf = SimpleNamespace(TransformListener=lambda: object())
            module.ModelStates = object
            module.LinkStates = object
            module.PersistentRiskRegions = object
            module.RiskEvidenceArray = object
            module.RiskRegions = object
            module.StructureMotions = object
            recorder = None
            try:
                recorder = module.SimExperimentRecorder()

                ablation_manifest = json.loads(
                    (recorder.meta_dir / "ablation_manifest.json").read_text()
                )
                config_snapshot = json.loads(
                    (recorder.meta_dir / "config_snapshot.json").read_text()
                )

                self.assertEqual(
                    ablation_manifest["variant"], "single_model_ekf_no_drift"
                )
                self.assertTrue(ablation_manifest["switches"]["single_model_ekf"])
                self.assertTrue(
                    ablation_manifest["switches"]["disable_covariance_inflation"]
                )
                self.assertTrue(
                    ablation_manifest["switches"]["disable_drift_compensation"]
                )
                self.assertEqual(
                    ablation_manifest["effective_runtime"]["covariance_alpha_xi"], 1.0
                )
                self.assertFalse(
                    ablation_manifest["effective_runtime"]["background_bias_enable"]
                )
                self.assertFalse(
                    ablation_manifest["effective_runtime"]["imm_enable_model_competition"]
                )
                self.assertEqual(
                    ablation_manifest["effective_runtime"][
                        "persistent_min_confirmed_mean_risk"
                    ],
                    0.65,
                )
                self.assertEqual(
                    config_snapshot["source_config_path"],
                    "/tmp/deform_monitor_v2_sim.yaml",
                )
                self.assertEqual(
                    config_snapshot["node_param_root"], "/deform_monitor_v2"
                )
                self.assertEqual(
                    config_snapshot["parameters"]["deform_monitor"]["ablation"]["variant"],
                    "single_model_ekf_no_drift",
                )
            finally:
                if recorder is not None:
                    recorder.close()
                module.rospy = original_rospy
                module.tf = original_tf
                module.ModelStates = original_model_states
                module.LinkStates = original_link_states
                module.PersistentRiskRegions = original_persistent_risk_regions
                module.RiskEvidenceArray = original_risk_evidence_array
                module.RiskRegions = original_risk_regions
                module.StructureMotions = original_structure_motions

    def test_build_scenario_manifest_falls_back_to_explicit_control_metadata(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        payload = module.build_scenario_manifest_payload(
            run_dir=pathlib.Path("/tmp/sim_run_000"),
            scenario_id="case_01",
            explicit_control={
                "controlled_object": "debris_block_02",
                "command_frame": "world",
                "velocity": {
                    "linear_mps": {"x": 0.0, "y": 0.0, "z": 0.002},
                    "angular_deg_per_sec": {"x": 0.0, "y": 0.0, "z": 0.0},
                },
                "axis": {"x": 0.0, "y": 0.0, "z": 1.0},
                "start_delay_sec": 8.0,
                "duration_sec": 20.0,
            },
            discovered_controls=[],
        )

        self.assertEqual(payload["scenario_id"], "case_01")
        self.assertEqual(len(payload["controls"]), 1)
        self.assertEqual(payload["controls"][0]["controlled_object"], "debris_block_02")
        self.assertEqual(payload["controls"][0]["command_frame"], "world")
        self.assertEqual(payload["controls"][0]["axis"], {"x": 0.0, "y": 0.0, "z": 1.0})
        self.assertEqual(payload["controls"][0]["start_delay_sec"], 8.0)
        self.assertEqual(payload["controls"][0]["duration_sec"], 20.0)
        self.assertEqual(payload["source"], "explicit")

    def test_build_scenario_manifest_preserves_reproducible_experiment_factors(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        factors = {
            "scene_id": "collapse_world_v3",
            "shape_set": ["plane", "wedge", "curved_band"],
            "object_quantity": 6,
            "motion_profile": "accelerating_rotation",
            "visibility_condition": "partial_occlusion",
            "repeat_index": 2,
        }
        payload = module.build_scenario_manifest_payload(
            run_dir=pathlib.Path("/tmp/sim_run_000"),
            scenario_id="case_02",
            experiment_factors=factors,
            object_metadata={
                "moving_panel": {
                    "shape": "rectangular_panel",
                    "size_class": "large",
                    "motion_profile": "accelerating_rotation",
                    "visibility_condition": "partial_occlusion",
                }
            },
        )

        self.assertEqual(payload["experiment_factors"], factors)
        self.assertEqual(
            payload["object_metadata"]["moving_panel"]["shape"],
            "rectangular_panel",
        )

    def test_normalize_experiment_factors_rejects_non_finite_values(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        with self.assertRaisesRegex(ValueError, "JSON-serializable"):
            module.normalize_experiment_factors({"speed_mps": float("nan")})

    def test_normalize_object_metadata_requires_attribute_mappings(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        with self.assertRaisesRegex(ValueError, "attribute mapping"):
            module.normalize_object_metadata({"moving_panel": "large"})

    def test_build_scenario_manifest_prefers_discovered_control_metadata(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        payload = module.build_scenario_manifest_payload(
            run_dir=pathlib.Path("/tmp/sim_run_000"),
            scenario_id="explicit_case",
            explicit_control={
                "controlled_object": "explicit_object",
                "command_frame": "world",
                "velocity": {
                    "linear_mps": {"x": 0.0, "y": 0.0, "z": 0.002},
                    "angular_deg_per_sec": {"x": 0.0, "y": 0.0, "z": 0.0},
                },
                "axis": {"x": 0.0, "y": 0.0, "z": 1.0},
                "start_delay_sec": 8.0,
                "duration_sec": 20.0,
                "scenario_id": "explicit_case",
            },
            discovered_controls=[
                {
                    "controlled_object": "discovered_object",
                    "command_frame": "body",
                    "velocity": {
                        "linear_mps": {"x": 0.0, "y": 0.001, "z": 0.0},
                        "angular_deg_per_sec": {"x": 0.0, "y": 0.0, "z": 0.0},
                    },
                    "axis": {"x": 0.0, "y": 1.0, "z": 0.0},
                    "start_delay_sec": 5.0,
                    "duration_sec": 12.0,
                    "scenario_id": "",
                }
            ],
        )

        self.assertEqual(payload["source"], "discovered")
        self.assertEqual(payload["controls"][0]["controlled_object"], "discovered_object")
        self.assertEqual(payload["controls"][0]["duration_sec"], 12.0)
        self.assertEqual(payload["scenario_id"], "explicit_case")

    def test_authoritative_control_selection_supports_arbitrary_evaluated_objects(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        controls = [
            {
                "controller_namespace": "/wedge_motion",
                "controlled_object": "moving_wedge",
                "command_frame": "world",
                "start_delay_sec": 2.0,
                "duration_sec": 10.0,
                "scenario_id": "multi_shape_case",
            },
            {
                "controller_namespace": "/panel_motion",
                "controlled_object": "moving_panel",
                "command_frame": "world",
                "start_delay_sec": 4.0,
                "duration_sec": 12.0,
                "scenario_id": "multi_shape_case",
            },
            {
                "controller_namespace": "/sensor_platform_motion",
                "controlled_object": "sensor_platform",
                "command_frame": "world",
                "start_delay_sec": 0.0,
                "duration_sec": 30.0,
                "scenario_id": "multi_shape_case",
            },
        ]

        selected = module.select_authoritative_discovered_controls(
            "multi_shape_case",
            controls,
            allowed_object_names={"moving_panel", "moving_wedge"},
        )

        self.assertEqual(
            [item["controlled_object"] for item in selected],
            ["moving_panel", "moving_wedge"],
        )

    def test_write_run_metadata_initially_writes_fallback_control_metadata(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        with tempfile.TemporaryDirectory(prefix="sim_experiment_recorder_") as temp_dir:
            temp_dir = pathlib.Path(temp_dir)
            recorder, fake_rospy, original_rospy = self._make_manifest_recorder_fixture(
                module,
                temp_dir,
                params={
                    "/deform_monitor_v2": {},
                },
            )
            try:
                recorder._write_run_metadata()

                manifest = self._read_manifest(recorder)

                self.assertEqual(manifest["source"], "explicit")
                self.assertEqual(manifest["scenario_id"], "collapse_microdeform_case_01")
                self.assertEqual(len(manifest["controls"]), 1)
                self.assertEqual(
                    manifest["controls"][0]["controlled_object"],
                    "obstacle_block_left_clone_clone",
                )
                self.assertEqual(manifest["controls"][0]["command_frame"], "world")
            finally:
                module.rospy = original_rospy

    def test_refreshes_discovered_controls_after_motion_controller_params_appear(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        with tempfile.TemporaryDirectory(prefix="sim_experiment_recorder_") as temp_dir:
            temp_dir = pathlib.Path(temp_dir)
            recorder, fake_rospy, original_rospy = self._make_manifest_recorder_fixture(
                module,
                temp_dir,
                params={
                    "/deform_monitor_v2": {},
                },
            )
            try:
                recorder._write_run_metadata()
                initial_manifest = self._read_manifest(recorder)

                fake_rospy.params.update(
                    {
                        "/model_01_motion/model_name": "model_01",
                        "/model_01_motion/command_frame": "world",
                        "/model_01_motion/control_rate": 20.0,
                        "/model_01_motion/command_timeout": 1.5,
                        "/model_01_motion/linear_x": 0.0,
                        "/model_01_motion/linear_y": 0.0,
                        "/model_01_motion/linear_z": 0.015,
                        "/model_01_motion/angular_x_deg": 0.0,
                        "/model_01_motion/angular_y_deg": 0.0,
                        "/model_01_motion/angular_z_deg": 0.0,
                        "/model_01_motion/start_delay": 8.0,
                        "/model_01_motion/duration": 20.0,
                        "/model_01_motion/scenario_id": "collapse_microdeform_case_01",
                        "/model_02_motion/model_name": "model_02",
                        "/model_02_motion/command_frame": "world",
                        "/model_02_motion/control_rate": 10.0,
                        "/model_02_motion/command_timeout": 2.0,
                        "/model_02_motion/linear_x": 0.005,
                        "/model_02_motion/linear_y": 0.0,
                        "/model_02_motion/linear_z": 0.0,
                        "/model_02_motion/angular_x_deg": 0.0,
                        "/model_02_motion/angular_y_deg": 0.0,
                        "/model_02_motion/angular_z_deg": 0.0,
                        "/model_02_motion/start_delay": 12.5,
                        "/model_02_motion/duration": 35.0,
                        "/model_02_motion/scenario_id": "collapse_microdeform_case_01",
                    }
                )

                recorder._handle_model_states(SimpleNamespace(name=[], pose=[]))
                refreshed_manifest = self._read_manifest(recorder)

                self.assertEqual(initial_manifest["source"], "explicit")
                self.assertEqual(refreshed_manifest["source"], "discovered")
                self.assertEqual(
                    [control["controlled_object"] for control in refreshed_manifest["controls"]],
                    ["model_01", "model_02"],
                )
                self.assertEqual(
                    [control["controller_namespace"] for control in refreshed_manifest["controls"]],
                    ["/model_01_motion", "/model_02_motion"],
                )
            finally:
                module.rospy = original_rospy

    def test_refresh_keeps_discovered_manifest_after_controller_params_disappear(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        with tempfile.TemporaryDirectory(prefix="sim_experiment_recorder_") as temp_dir:
            temp_dir = pathlib.Path(temp_dir)
            recorder, fake_rospy, original_rospy = self._make_manifest_recorder_fixture(
                module,
                temp_dir,
                params={
                    "/deform_monitor_v2": {},
                },
            )
            try:
                recorder._write_run_metadata()

                fake_rospy.params.update(
                    {
                        "/model_01_motion/model_name": "model_01",
                        "/model_01_motion/command_frame": "world",
                        "/model_01_motion/control_rate": 20.0,
                        "/model_01_motion/command_timeout": 1.5,
                        "/model_01_motion/linear_x": 0.0,
                        "/model_01_motion/linear_y": 0.0,
                        "/model_01_motion/linear_z": 0.015,
                        "/model_01_motion/angular_x_deg": 0.0,
                        "/model_01_motion/angular_y_deg": 0.0,
                        "/model_01_motion/angular_z_deg": 0.0,
                        "/model_01_motion/start_delay": 8.0,
                        "/model_01_motion/duration": 20.0,
                        "/model_01_motion/scenario_id": "collapse_microdeform_case_01",
                        "/model_02_motion/model_name": "model_02",
                        "/model_02_motion/command_frame": "world",
                        "/model_02_motion/control_rate": 10.0,
                        "/model_02_motion/command_timeout": 2.0,
                        "/model_02_motion/linear_x": 0.005,
                        "/model_02_motion/linear_y": 0.0,
                        "/model_02_motion/linear_z": 0.0,
                        "/model_02_motion/angular_x_deg": 0.0,
                        "/model_02_motion/angular_y_deg": 0.0,
                        "/model_02_motion/angular_z_deg": 0.0,
                        "/model_02_motion/start_delay": 12.5,
                        "/model_02_motion/duration": 35.0,
                        "/model_02_motion/scenario_id": "collapse_microdeform_case_01",
                    }
                )

                recorder._handle_model_states(SimpleNamespace(name=[], pose=[]))
                promoted_manifest = self._read_manifest(recorder)

                fake_rospy.params.pop("/model_02_motion/model_name")
                fake_rospy.params.pop("/model_02_motion/command_frame")
                fake_rospy.params.pop("/model_02_motion/control_rate")
                fake_rospy.params.pop("/model_02_motion/command_timeout")
                fake_rospy.params.pop("/model_02_motion/linear_x")
                fake_rospy.params.pop("/model_02_motion/linear_y")
                fake_rospy.params.pop("/model_02_motion/linear_z")
                fake_rospy.params.pop("/model_02_motion/angular_x_deg")
                fake_rospy.params.pop("/model_02_motion/angular_y_deg")
                fake_rospy.params.pop("/model_02_motion/angular_z_deg")
                fake_rospy.params.pop("/model_02_motion/start_delay")
                fake_rospy.params.pop("/model_02_motion/duration")
                fake_rospy.params.pop("/model_02_motion/scenario_id")

                recorder._handle_model_states(SimpleNamespace(name=[], pose=[]))
                refreshed_manifest = self._read_manifest(recorder)

                self.assertEqual(promoted_manifest["source"], "discovered")
                self.assertEqual(refreshed_manifest["source"], "discovered")
                self.assertEqual(
                    [control["controlled_object"] for control in refreshed_manifest["controls"]],
                    ["model_01", "model_02"],
                )
            finally:
                module.rospy = original_rospy

    def test_refresh_keeps_explicit_manifest_for_stale_motion_params(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        with tempfile.TemporaryDirectory(prefix="sim_experiment_recorder_") as temp_dir:
            temp_dir = pathlib.Path(temp_dir)
            recorder, fake_rospy, original_rospy = self._make_manifest_recorder_fixture(
                module,
                temp_dir,
                params={
                    "/deform_monitor_v2": {},
                    "/model_01_motion/model_name": "model_01",
                    "/model_01_motion/command_frame": "world",
                    "/model_01_motion/control_rate": 20.0,
                    "/model_01_motion/command_timeout": 1.5,
                    "/model_01_motion/linear_x": 0.0,
                    "/model_01_motion/linear_y": 0.0,
                    "/model_01_motion/linear_z": 0.015,
                    "/model_01_motion/angular_x_deg": 0.0,
                    "/model_01_motion/angular_y_deg": 0.0,
                    "/model_01_motion/angular_z_deg": 0.0,
                    "/model_01_motion/start_delay": 8.0,
                    "/model_01_motion/duration": 20.0,
                    "/model_01_motion/scenario_id": "stale_case",
                    "/model_02_motion/model_name": "model_02",
                    "/model_02_motion/command_frame": "world",
                    "/model_02_motion/control_rate": 10.0,
                    "/model_02_motion/command_timeout": 2.0,
                    "/model_02_motion/linear_x": 0.005,
                    "/model_02_motion/linear_y": 0.0,
                    "/model_02_motion/linear_z": 0.0,
                    "/model_02_motion/angular_x_deg": 0.0,
                    "/model_02_motion/angular_y_deg": 0.0,
                    "/model_02_motion/angular_z_deg": 0.0,
                    "/model_02_motion/start_delay": 12.5,
                    "/model_02_motion/duration": 35.0,
                    "/model_02_motion/scenario_id": "stale_case",
                },
            )
            try:
                recorder._write_run_metadata()
                recorder._handle_model_states(SimpleNamespace(name=[], pose=[]))

                manifest = self._read_manifest(recorder)

                self.assertEqual(manifest["source"], "explicit")
                self.assertEqual(manifest["controls"][0]["controlled_object"], "obstacle_block_left_clone_clone")
            finally:
                module.rospy = original_rospy

    def test_refresh_keeps_explicit_manifest_for_single_controller(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        with tempfile.TemporaryDirectory(prefix="sim_experiment_recorder_") as temp_dir:
            temp_dir = pathlib.Path(temp_dir)
            recorder, fake_rospy, original_rospy = self._make_manifest_recorder_fixture(
                module,
                temp_dir,
                params={
                    "/deform_monitor_v2": {},
                    "/model_01_motion/model_name": "model_01",
                    "/model_01_motion/command_frame": "world",
                    "/model_01_motion/control_rate": 20.0,
                    "/model_01_motion/command_timeout": 1.5,
                    "/model_01_motion/linear_x": 0.0,
                    "/model_01_motion/linear_y": 0.0,
                    "/model_01_motion/linear_z": 0.015,
                    "/model_01_motion/angular_x_deg": 0.0,
                    "/model_01_motion/angular_y_deg": 0.0,
                    "/model_01_motion/angular_z_deg": 0.0,
                    "/model_01_motion/start_delay": 8.0,
                    "/model_01_motion/duration": 20.0,
                    "/model_01_motion/scenario_id": "collapse_microdeform_case_01",
                },
            )
            try:
                recorder._write_run_metadata()
                recorder._handle_model_states(SimpleNamespace(name=[], pose=[]))

                manifest = self._read_manifest(recorder)

                self.assertEqual(manifest["source"], "explicit")
                self.assertEqual(manifest["controls"][0]["controlled_object"], "obstacle_block_left_clone_clone")
            finally:
                module.rospy = original_rospy

    def test_refresh_keeps_explicit_manifest_for_extra_motion_namespace(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        with tempfile.TemporaryDirectory(prefix="sim_experiment_recorder_") as temp_dir:
            temp_dir = pathlib.Path(temp_dir)
            recorder, fake_rospy, original_rospy = self._make_manifest_recorder_fixture(
                module,
                temp_dir,
                params={
                    "/deform_monitor_v2": {},
                    "/model_01_motion/model_name": "model_01",
                    "/model_01_motion/command_frame": "world",
                    "/model_01_motion/control_rate": 20.0,
                    "/model_01_motion/command_timeout": 1.5,
                    "/model_01_motion/linear_x": 0.0,
                    "/model_01_motion/linear_y": 0.0,
                    "/model_01_motion/linear_z": 0.015,
                    "/model_01_motion/angular_x_deg": 0.0,
                    "/model_01_motion/angular_y_deg": 0.0,
                    "/model_01_motion/angular_z_deg": 0.0,
                    "/model_01_motion/start_delay": 8.0,
                    "/model_01_motion/duration": 20.0,
                    "/model_01_motion/scenario_id": "collapse_microdeform_case_01",
                    "/model_02_motion/model_name": "model_02",
                    "/model_02_motion/command_frame": "world",
                    "/model_02_motion/control_rate": 10.0,
                    "/model_02_motion/command_timeout": 2.0,
                    "/model_02_motion/linear_x": 0.005,
                    "/model_02_motion/linear_y": 0.0,
                    "/model_02_motion/linear_z": 0.0,
                    "/model_02_motion/angular_x_deg": 0.0,
                    "/model_02_motion/angular_y_deg": 0.0,
                    "/model_02_motion/angular_z_deg": 0.0,
                    "/model_02_motion/start_delay": 12.5,
                    "/model_02_motion/duration": 35.0,
                    "/model_02_motion/scenario_id": "collapse_microdeform_case_01",
                    "/spawn_mid360_fastlio_motion/model_name": "spawn_mid360_fastlio",
                    "/spawn_mid360_fastlio_motion/command_frame": "world",
                    "/spawn_mid360_fastlio_motion/control_rate": 15.0,
                    "/spawn_mid360_fastlio_motion/command_timeout": 1.0,
                    "/spawn_mid360_fastlio_motion/linear_x": 0.0,
                    "/spawn_mid360_fastlio_motion/linear_y": 0.0,
                    "/spawn_mid360_fastlio_motion/linear_z": 0.0,
                    "/spawn_mid360_fastlio_motion/start_delay": 0.0,
                    "/spawn_mid360_fastlio_motion/duration": 0.0,
                    "/spawn_mid360_fastlio_motion/scenario_id": "collapse_microdeform_case_01",
                },
            )
            try:
                recorder._write_run_metadata()
                recorder._handle_model_states(SimpleNamespace(name=[], pose=[]))

                manifest = self._read_manifest(recorder)

                self.assertEqual(manifest["source"], "explicit")
                self.assertEqual(manifest["controls"][0]["controlled_object"], "obstacle_block_left_clone_clone")
            finally:
                module.rospy = original_rospy

    def test_refresh_keeps_explicit_manifest_for_incomplete_discovered_metadata(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        with tempfile.TemporaryDirectory(prefix="sim_experiment_recorder_") as temp_dir:
            temp_dir = pathlib.Path(temp_dir)
            recorder, fake_rospy, original_rospy = self._make_manifest_recorder_fixture(
                module,
                temp_dir,
                params={
                    "/deform_monitor_v2": {},
                    "/model_01_motion/model_name": "model_01",
                    "/model_01_motion/command_frame": "world",
                    "/model_01_motion/control_rate": 20.0,
                    "/model_01_motion/command_timeout": 1.5,
                    "/model_01_motion/linear_x": 0.0,
                    "/model_01_motion/linear_y": 0.0,
                    "/model_01_motion/linear_z": 0.015,
                    "/model_01_motion/angular_x_deg": 0.0,
                    "/model_01_motion/angular_y_deg": 0.0,
                    "/model_01_motion/angular_z_deg": 0.0,
                    "/model_01_motion/start_delay": 8.0,
                    "/model_01_motion/duration": 20.0,
                    "/model_01_motion/scenario_id": "collapse_microdeform_case_01",
                    "/model_02_motion/model_name": "model_02",
                    "/model_02_motion/command_frame": "",
                    "/model_02_motion/control_rate": 10.0,
                    "/model_02_motion/command_timeout": 2.0,
                    "/model_02_motion/linear_x": 0.005,
                    "/model_02_motion/linear_y": 0.0,
                    "/model_02_motion/linear_z": 0.0,
                    "/model_02_motion/angular_x_deg": 0.0,
                    "/model_02_motion/angular_y_deg": 0.0,
                    "/model_02_motion/angular_z_deg": 0.0,
                    "/model_02_motion/start_delay": None,
                    "/model_02_motion/duration": 35.0,
                    "/model_02_motion/scenario_id": "collapse_microdeform_case_01",
                },
            )
            try:
                recorder._write_run_metadata()
                recorder._handle_model_states(SimpleNamespace(name=[], pose=[]))

                manifest = self._read_manifest(recorder)

                self.assertEqual(manifest["source"], "explicit")
                self.assertEqual(
                    manifest["controls"][0]["controlled_object"],
                    "obstacle_block_left_clone_clone",
                )
            finally:
                module.rospy = original_rospy

    def test_refresh_does_not_churn_on_unchanged_discovered_manifest(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        with tempfile.TemporaryDirectory(prefix="sim_experiment_recorder_") as temp_dir:
            temp_dir = pathlib.Path(temp_dir)
            recorder, fake_rospy, original_rospy = self._make_manifest_recorder_fixture(
                module,
                temp_dir,
                params={
                    "/deform_monitor_v2": {},
                },
            )
            try:
                recorder._write_run_metadata()

                fake_rospy.params.update(
                    {
                        "/model_01_motion/model_name": "model_01",
                        "/model_01_motion/command_frame": "world",
                        "/model_01_motion/control_rate": 20.0,
                        "/model_01_motion/command_timeout": 1.5,
                        "/model_01_motion/linear_x": 0.0,
                        "/model_01_motion/linear_y": 0.0,
                        "/model_01_motion/linear_z": 0.015,
                        "/model_01_motion/angular_x_deg": 0.0,
                        "/model_01_motion/angular_y_deg": 0.0,
                        "/model_01_motion/angular_z_deg": 0.0,
                        "/model_01_motion/start_delay": 8.0,
                        "/model_01_motion/duration": 20.0,
                        "/model_01_motion/scenario_id": "collapse_microdeform_case_01",
                        "/model_02_motion/model_name": "model_02",
                        "/model_02_motion/command_frame": "world",
                        "/model_02_motion/control_rate": 10.0,
                        "/model_02_motion/command_timeout": 2.0,
                        "/model_02_motion/linear_x": 0.005,
                        "/model_02_motion/linear_y": 0.0,
                        "/model_02_motion/linear_z": 0.0,
                        "/model_02_motion/angular_x_deg": 0.0,
                        "/model_02_motion/angular_y_deg": 0.0,
                        "/model_02_motion/angular_z_deg": 0.0,
                        "/model_02_motion/start_delay": 12.5,
                        "/model_02_motion/duration": 35.0,
                        "/model_02_motion/scenario_id": "collapse_microdeform_case_01",
                    }
                )

                first_refresh = recorder._refresh_scenario_manifest_if_needed()
                second_refresh = recorder._refresh_scenario_manifest_if_needed()

                self.assertTrue(first_refresh)
                self.assertFalse(second_refresh)
                manifest = self._read_manifest(recorder)
                self.assertEqual(manifest["source"], "discovered")
                self.assertEqual(
                    [control["controlled_object"] for control in manifest["controls"]],
                    ["model_01", "model_02"],
                )
            finally:
                module.rospy = original_rospy

    def test_discover_controlled_objects_extracts_motion_controller_metadata(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        params = {
            "/debris_controller/model_name": "debris_block_02",
            "/debris_controller/command_frame": "body",
            "/debris_controller/control_rate": 20.0,
            "/debris_controller/command_timeout": 1.5,
            "/debris_controller/linear_y": 0.001,
            "/debris_controller/start_delay": 5.0,
            "/debris_controller/duration": 12.0,
            "/debris_controller/scenario_id": "case_body_y",
        }

        controls = module.discover_controlled_objects(
            get_param=lambda name, default=None: params.get(name, default),
            get_param_names=lambda: sorted(params.keys()),
        )

        self.assertEqual(len(controls), 1)
        self.assertEqual(controls[0]["controlled_object"], "debris_block_02")
        self.assertEqual(controls[0]["command_frame"], "body")
        self.assertEqual(controls[0]["axis"], {"x": 0.0, "y": 1.0, "z": 0.0})
        self.assertEqual(controls[0]["start_delay_sec"], 5.0)
        self.assertEqual(controls[0]["duration_sec"], 12.0)
        self.assertEqual(controls[0]["scenario_id"], "case_body_y")

    def test_discover_controlled_objects_discovers_multi_motion_controller_namespaces(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        params = {
            "/model_01_motion/model_name": "model_01",
            "/model_01_motion/command_frame": "world",
            "/model_01_motion/control_rate": 20.0,
            "/model_01_motion/command_timeout": 1.5,
            "/model_01_motion/linear_x": 0.0,
            "/model_01_motion/linear_y": 0.0,
            "/model_01_motion/linear_z": 0.015,
            "/model_01_motion/angular_x_deg": 0.0,
            "/model_01_motion/angular_y_deg": 0.0,
            "/model_01_motion/angular_z_deg": 0.0,
            "/model_01_motion/start_delay": 8.0,
            "/model_01_motion/duration": 20.0,
            "/model_01_motion/scenario_id": "case_model_01",
            "/model_02_motion/model_name": "model_02",
            "/model_02_motion/command_frame": "world",
            "/model_02_motion/control_rate": 10.0,
            "/model_02_motion/command_timeout": 2.0,
            "/model_02_motion/linear_x": 0.005,
            "/model_02_motion/linear_y": 0.0,
            "/model_02_motion/linear_z": 0.0,
            "/model_02_motion/angular_x_deg": 0.0,
            "/model_02_motion/angular_y_deg": 0.0,
            "/model_02_motion/angular_z_deg": 0.0,
            "/model_02_motion/start_delay": 12.5,
            "/model_02_motion/duration": 35.0,
            "/model_02_motion/scenario_id": "case_model_02",
        }

        controls = module.discover_controlled_objects(
            get_param=lambda name, default=None: params.get(name, default),
            get_param_names=lambda: sorted(params.keys()),
        )

        controls_by_object = {control["controlled_object"]: control for control in controls}
        self.assertEqual(set(controls_by_object), {"model_01", "model_02"})
        self.assertEqual(controls_by_object["model_01"]["command_frame"], "world")
        self.assertEqual(controls_by_object["model_02"]["command_frame"], "world")
        self.assertEqual(controls_by_object["model_01"]["scenario_id"], "case_model_01")
        self.assertEqual(controls_by_object["model_02"]["scenario_id"], "case_model_02")
        self.assertEqual(
            controls_by_object["model_01"]["axis"],
            {"x": 0.0, "y": 0.0, "z": 1.0},
        )
        self.assertEqual(
            controls_by_object["model_02"]["axis"],
            {"x": 1.0, "y": 0.0, "z": 0.0},
        )
        self.assertEqual(controls_by_object["model_01"]["start_delay_sec"], 8.0)
        self.assertEqual(controls_by_object["model_02"]["start_delay_sec"], 12.5)
        self.assertEqual(controls_by_object["model_01"]["duration_sec"], 20.0)
        self.assertEqual(controls_by_object["model_02"]["duration_sec"], 35.0)

    def test_discover_controlled_objects_ignores_namespaces_without_motion_contract(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        params = {
            "/model_01_motion/model_name": "model_01",
            "/model_01_motion/command_frame": "world",
            "/model_01_motion/control_rate": 20.0,
            "/model_01_motion/command_timeout": 1.5,
            "/model_01_motion/linear_x": 0.0,
            "/model_01_motion/linear_y": 0.0,
            "/model_01_motion/linear_z": 0.015,
            "/model_01_motion/angular_x_deg": 0.0,
            "/model_01_motion/angular_y_deg": 0.0,
            "/model_01_motion/angular_z_deg": 0.0,
            "/model_01_motion/start_delay": 8.0,
            "/model_01_motion/duration": 20.0,
            "/model_01_motion/scenario_id": "case_model_01",
            "/spawn_mid360_fastlio/model_name": "spawn_mid360_fastlio",
            "/spawn_mid360_fastlio/command_frame": "world",
            "/spawn_mid360_fastlio/control_rate": 30.0,
            "/spawn_mid360_fastlio/command_timeout": 0.2,
            "/spawn_mid360_fastlio/start_delay": 0.0,
            "/spawn_mid360_fastlio/duration": 0.0,
            "/spawn_mid360_fastlio/scenario_id": "false_positive_case",
        }

        controls = module.discover_controlled_objects(
            get_param=lambda name, default=None: params.get(name, default),
            get_param_names=lambda: sorted(params.keys()),
        )

        self.assertEqual(len(controls), 1)
        self.assertEqual(controls[0]["controlled_object"], "model_01")
        self.assertEqual(controls[0]["controller_namespace"], "/model_01_motion")

    def test_discover_controlled_objects_rejects_namespace_missing_required_metadata(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        params = {
            "/missing_rate_motion/model_name": "model_missing_rate",
            "/missing_rate_motion/command_frame": "world",
            "/missing_rate_motion/linear_x": 0.01,
            "/missing_rate_motion/command_timeout": 1.0,
            "/missing_rate_motion/start_delay": 3.0,
            "/missing_rate_motion/duration": 9.0,
            "/missing_rate_motion/scenario_id": "case_missing_rate",
        }

        controls = module.discover_controlled_objects(
            get_param=lambda name, default=None: params.get(name, default),
            get_param_names=lambda: sorted(params.keys()),
        )

        self.assertEqual(controls, [])

    def test_sim_launch_keeps_explicit_fallback_args(self):
        self._assert_launch_file_keeps_explicit_fallback_args(
            pathlib.Path(__file__).resolve().parents[1] / "launch" / "deform_monitor_v2_sim.launch"
        )

    def test_sim_dynamic_launch_keeps_explicit_fallback_args(self):
        self._assert_launch_file_keeps_explicit_fallback_args(
            pathlib.Path(__file__).resolve().parents[1]
            / "launch"
            / "deform_monitor_v2_sim_dynamic.launch"
        )

    def test_ensure_directories_creates_trajectory_dir(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        with tempfile.TemporaryDirectory(prefix="sim_experiment_recorder_") as temp_dir:
            temp_dir = pathlib.Path(temp_dir)
            recorder = module.SimExperimentRecorder.__new__(module.SimExperimentRecorder)
            recorder.meta_dir = temp_dir / "meta"
            recorder.truth_dir = temp_dir / "truth"
            recorder.truth_objects_dir = recorder.truth_dir / "objects"
            recorder.truth_links_dir = recorder.truth_dir / "links"
            recorder.algorithm_dir = temp_dir / "algorithm"
            recorder.trajectory_dir = temp_dir / "trajectory"

            recorder._ensure_directories()

            self.assertTrue(recorder.meta_dir.exists())
            self.assertTrue(recorder.truth_dir.exists())
            self.assertTrue(recorder.truth_objects_dir.exists())
            self.assertTrue(recorder.truth_links_dir.exists())
            self.assertTrue(recorder.algorithm_dir.exists())
            self.assertTrue(recorder.trajectory_dir.exists())

    def test_write_run_info_uses_configured_trajectory_export_settings(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        with tempfile.TemporaryDirectory(prefix="sim_experiment_recorder_") as temp_dir:
            temp_dir = pathlib.Path(temp_dir)
            recorder = module.SimExperimentRecorder.__new__(module.SimExperimentRecorder)
            recorder.run_dir = temp_dir / "sim_run_000"
            recorder.meta_dir = recorder.run_dir / "meta"
            recorder.meta_dir.mkdir(parents=True)
            recorder.truth_frame = "world"
            recorder.algorithm_frame = "camera_init"
            recorder.ego_model_name = "mid360_fastlio"
            recorder.model_states_topic = "/gazebo/model_states"
            recorder.link_states_topic = "/gazebo/link_states"
            recorder.ground_truth_odometry_topic = "/ground_truth/odom"
            recorder.risk_evidence_topic = "/deform/risk_evidence"
            recorder.risk_regions_topic = "/deform/risk_regions"
            recorder.persistent_risk_regions_topic = "/deform/persistent_risk_regions"
            recorder.structure_motions_topic = "/deform/structure_motions"
            recorder.odometry_topic = "/Odometry"
            recorder.sensor_scoped_link_name = "mid360_fastlio::mid360_link"
            recorder.sensor_frame_name = "livox"
            recorder.gt_tum_filename = "gt_sensor_world_tum.txt"
            recorder.odom_tum_filename = "odom_raw_tum.txt"

            recorder._write_run_info()

            payload = json.loads((recorder.meta_dir / "run_info.json").read_text())

            self.assertEqual(payload["topics"]["odometry"], "/Odometry")
            self.assertEqual(payload["topics"]["ground_truth_odometry"], "/ground_truth/odom")
            self.assertEqual(
                payload["topics"]["persistent_risk_regions"],
                "/deform/persistent_risk_regions",
            )
            self.assertEqual(
                payload["sensor_scoped_link_name"], "mid360_fastlio::mid360_link"
            )
            self.assertEqual(payload["sensor_frame_name"], "livox")
            self.assertEqual(payload["trajectory_export"]["gt_file"], "gt_sensor_world_tum.txt")
            self.assertEqual(
                payload["trajectory_export"]["odom_file"], "odom_raw_tum.txt"
            )
            self.assertEqual(
                payload["trajectory_export"]["gt_pose_source"],
                "ground_truth_odometry_plus_tf",
            )

    def test_handle_link_states_updates_sensor_pose_cache_for_target_scoped_link(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        fake_rospy = _FakeRospy()
        original_rospy = module.rospy
        module.rospy = fake_rospy
        try:
            recorder = module.SimExperimentRecorder.__new__(module.SimExperimentRecorder)
            recorder.ego_model_name = "mid360_fastlio"
            recorder.sensor_scoped_link_name = "mid360_fastlio::mid360_link"
            recorder._latest_sensor_pose_world = None
            recorder._latest_sensor_pose_stamp = None
            recorder.truth_frame = "world"
            recorder._link_files = {}
            recorder._tracked_link_names = lambda _msg: []

            msg = SimpleNamespace(
                name=["mid360_fastlio::mid360_link"],
                pose=[
                    SimpleNamespace(
                        position=SimpleNamespace(x=1.0, y=2.0, z=3.0),
                        orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
                    )
                ],
            )

            recorder._handle_link_states(msg)

            self.assertEqual(
                recorder._latest_sensor_pose_world,
                {
                    "position": {"x": 1.0, "y": 2.0, "z": 3.0},
                    "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                },
            )
            self.assertEqual(recorder._latest_sensor_pose_stamp, fake_rospy.now_sec)
            recorder.close()
        finally:
            module.rospy = original_rospy

    def test_build_run_info_includes_trajectory_export_metadata(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        payload = module.build_run_info_payload(
            run_dir=pathlib.Path("/tmp/sim_run_000"),
            truth_frame="world",
            algorithm_frame="camera_init",
            ego_model_name="mid360_fastlio",
            model_states_topic="/gazebo/model_states",
            link_states_topic="/gazebo/link_states",
            risk_evidence_topic="/deform/risk_evidence",
            risk_regions_topic="/deform/risk_regions",
            persistent_risk_regions_topic="/deform/persistent_risk_regions",
            structure_motions_topic="/deform/structure_motions",
            odometry_topic="/Odometry",
            sensor_scoped_link_name="mid360_fastlio::mid360_link",
            gt_tum_filename="gt_sensor_world_tum.txt",
            odom_tum_filename="odom_raw_tum.txt",
            ground_truth_odometry_topic="/ground_truth/odom",
            sensor_frame_name="livox",
        )

        self.assertEqual(payload["topics"]["odometry"], "/Odometry")
        self.assertEqual(payload["topics"]["ground_truth_odometry"], "/ground_truth/odom")
        self.assertEqual(
            payload["topics"]["persistent_risk_regions"],
            "/deform/persistent_risk_regions",
        )
        self.assertEqual(payload["sensor_scoped_link_name"], "mid360_fastlio::mid360_link")
        self.assertEqual(payload["sensor_frame_name"], "livox")
        self.assertEqual(payload["trajectory_export"]["enabled"], True)
        self.assertEqual(payload["trajectory_export"]["gt_file"], "gt_sensor_world_tum.txt")
        self.assertEqual(payload["trajectory_export"]["odom_file"], "odom_raw_tum.txt")
        self.assertEqual(
            payload["trajectory_export"]["timestamp_policy"], "odometry_master_clock"
        )
        self.assertEqual(payload["trajectory_export"]["runtime_alignment_applied"], False)
        self.assertEqual(
            payload["trajectory_export"]["gt_pose_source"],
            "ground_truth_odometry_plus_tf",
        )

    def test_hardware_manifest_records_reproducible_platform_fields(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        payload = module.build_hardware_manifest_payload(
            hostname="rescue-compute-01",
            platform_string="Linux-5.15-test",
            cpu_model="Example CPU 8-Core",
            logical_cpu_count=16,
            memory_total_bytes=32 * 1024 ** 3,
        )

        self.assertEqual(payload["hostname"], "rescue-compute-01")
        self.assertEqual(payload["platform"], "Linux-5.15-test")
        self.assertEqual(payload["cpu_model"], "Example CPU 8-Core")
        self.assertEqual(payload["logical_cpu_count"], 16)
        self.assertEqual(payload["memory_total_bytes"], 32 * 1024 ** 3)

    def test_build_run_info_disables_trajectory_export_without_sensor_link(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        payload = module.build_run_info_payload(
            run_dir=pathlib.Path("/tmp/sim_run_000"),
            truth_frame="world",
            algorithm_frame="camera_init",
            ego_model_name="mid360_fastlio",
            model_states_topic="/gazebo/model_states",
            link_states_topic="/gazebo/link_states",
            risk_evidence_topic="/deform/risk_evidence",
            risk_regions_topic="/deform/risk_regions",
            persistent_risk_regions_topic="/deform/persistent_risk_regions",
            structure_motions_topic="/deform/structure_motions",
            odometry_topic="/Odometry",
            sensor_scoped_link_name="",
            gt_tum_filename="gt_sensor_world_tum.txt",
            odom_tum_filename="odom_raw_tum.txt",
        )

        self.assertEqual(payload["sensor_scoped_link_name"], "")
        self.assertFalse(payload["trajectory_export"]["enabled"])

    def test_serialize_structure_motion_preserves_nested_motion_and_bbox_fields(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        msg = SimpleNamespace(
            motion=SimpleNamespace(x=0.12, y=-0.34, z=0.56),
            bbox_new_max=SimpleNamespace(x=4.0, y=5.0, z=6.0),
        )

        serialized = module.serialize_structure_motion(msg)

        self.assertEqual(serialized["motion"]["x"], 0.12)
        self.assertEqual(serialized["bbox_new_max"]["z"], 6.0)

    def test_parse_scoped_link_name_extracts_model_and_link_names(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        model_name, link_name = module.parse_scoped_link_name("crate_model::base_link")

        self.assertEqual(model_name, "crate_model")
        self.assertEqual(link_name, "base_link")


if __name__ == "__main__":
    unittest.main()
