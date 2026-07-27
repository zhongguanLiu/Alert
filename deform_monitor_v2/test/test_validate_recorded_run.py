import csv
import importlib.util
import json
import math
import pathlib
import sqlite3
import tempfile
import unittest


SCRIPT_PATH = (
    pathlib.Path(__file__).resolve().parents[1]
    / "scripts"
    / "validate_recorded_run.py"
)


def load_module_if_exists():
    if not SCRIPT_PATH.exists():
        return None
    spec = importlib.util.spec_from_file_location(
        "validate_recorded_run_under_test", SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def header(seq, stamp):
    return {
        "seq": seq,
        "frame_id": "camera_init",
        "stamp": {"secs": int(stamp), "nsecs": 0, "sec": float(stamp)},
    }


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n")


def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for record in records:
            json.dump(record, handle, sort_keys=True)
            handle.write("\n")


def make_valid_run(root):
    run_dir = pathlib.Path(root) / "sim_run_000"
    meta_dir = run_dir / "meta"
    algorithm_dir = run_dir / "algorithm"
    truth_objects_dir = run_dir / "truth" / "objects"
    analysis_dir = run_dir / "analysis"
    for path in (meta_dir, algorithm_dir, truth_objects_dir, analysis_dir):
        path.mkdir(parents=True, exist_ok=True)

    write_json(
        meta_dir / "run_info.json",
        {
            "truth_frame": "world",
            "algorithm_frame": "camera_init",
            "truth_recording": {"object_rate_hz": 1.0, "link_rate_hz": 1.0},
            "algorithm_recording": {
                "schema_version": 2,
                "clean_shutdown_marker": "meta/run_complete.json",
            },
        },
    )
    write_json(meta_dir / "object_id_catalog.json", {"31": "model_01"})
    write_json(
        meta_dir / "frame_alignment.json",
        {
            "truth_frame": "world",
            "algorithm_frame": "camera_init",
            "truth_reference_stamp_sec": 0.0,
            "algorithm_reference_stamp_sec": 0.0,
            "pose_pair_delta_sec": 0.0,
            "world_from_algorithm_transform": {
                "pose": {
                    "position": {"x": 0.0, "y": 0.0, "z": 0.0},
                    "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                }
            },
        },
    )

    truth_path = truth_objects_dir / "model_01.csv"
    with truth_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "recorded_time_sec",
                "model_name",
                "frame_id",
                "twist_frame_id",
                "position_x",
                "position_y",
                "position_z",
                "orientation_x",
                "orientation_y",
                "orientation_z",
                "orientation_w",
                "linear_velocity_x",
                "linear_velocity_y",
                "linear_velocity_z",
                "angular_velocity_x",
                "angular_velocity_y",
                "angular_velocity_z",
            ]
        )
        for stamp in (0.0, 1.0, 2.0, 3.0):
            writer.writerow(
                [stamp, "model_01", "world", "world", stamp * 0.01, 0, 0,
                 0, 0, 0, 1, 0.01, 0, 0, 0, 0, 0]
            )

    stamps = (1.0, 2.0)
    catalog_records = [
        {
            "schema_version": 2,
            "reference_epoch": 1,
            "anchor_id": 7,
            "recorded_at": {"sec": 1.0},
            "anchor": {
                "id": 7,
                "anchor_type": 0,
                "object_id": 31,
                "object_id_valid": True,
                "ref_center": {"x": 0.0, "y": 0.0, "z": 0.0},
                "reference_epoch": 1,
                "reference_stamp": {"sec": 0.5},
                "reference_origin": 0,
            },
        }
    ]
    write_jsonl(algorithm_dir / "anchor_catalog.jsonl", catalog_records)

    frame_streams = {
        "anchor_observations": [],
        "processing_stamps": [],
        "object_observation_stats": [],
        "clusters": [],
        "risk_evidence": [],
        "risk_regions": [],
        "persistent_risk_regions": [],
        "structure_motions": [],
    }
    for index, stamp in enumerate(stamps, start=1):
        common = {
            "schema_version": 2,
            "header": header(index, stamp),
            "reference_epoch": 1,
            "recorded_at": {"sec": stamp + 0.01},
        }
        frame_streams["anchor_observations"].append(
            {
                **common,
                "reference_initialized_at": {"sec": 0.5},
                "anchors": [
                    {
                        "id": 7,
                        "observed_object_id": 31,
                        "observed_object_id_valid": True,
                        "object_association_state": 1,
                        "observable": True,
                        "comparable": True,
                        "significant": index == 2,
                    }
                ],
            }
        )
        frame_streams["processing_stamps"].append(
            {**common, "anchor_count": 1}
        )
        frame_streams["object_observation_stats"].append(
            {
                **common,
                "phase": 1,
                "window_start": {"sec": stamp - 0.4},
                "window_end": {"sec": stamp},
                "frame_count": 5,
                "total_point_count": 100,
                "valid_label_point_count": 90,
                "invalid_label_point_count": 10,
                "objects": [
                    {"object_id": 31, "point_count": 90, "visible_frame_count": 5}
                ],
            }
        )
        frame_streams["clusters"].append({**common, "clusters": []})
        frame_streams["risk_evidence"].append({**common, "evidences": []})
        frame_streams["risk_regions"].append({**common, "regions": []})
        frame_streams["persistent_risk_regions"].append(
            {**common, "regions": []}
        )
        frame_streams["structure_motions"].append({**common, "motions": []})

    for stream, records in frame_streams.items():
        write_jsonl(algorithm_dir / f"{stream}.jsonl", records)

    stream_rows = {"anchor_catalog": {"row_count": 1}}
    stream_rows.update(
        {stream: {"row_count": len(records)} for stream, records in frame_streams.items()}
    )
    write_json(
        meta_dir / "run_complete.json",
        {
            "schema_version": 2,
            "clean_shutdown": True,
            "streams": stream_rows,
        },
    )
    return run_dir


def enable_direct_link_truth(run_dir, write_track=True):
    run_info_path = run_dir / "meta" / "run_info.json"
    run_info = json.loads(run_info_path.read_text())
    run_info["truth_recording"]["dynamic_link_policy"] = (
        "sensor_and_surface_truth_links"
    )
    write_json(run_info_path, run_info)
    scoped_name = "model_01::ground_truth_v_000"
    write_jsonl(
        run_dir / "truth" / "surface_truth_points.jsonl",
        [{"scoped_link_name": scoped_name, "model_name": "model_01"}],
    )
    if not write_track:
        return scoped_name
    truth_links_dir = run_dir / "truth" / "links"
    truth_links_dir.mkdir(parents=True, exist_ok=True)
    with (truth_links_dir / "model_01__ground_truth_v_000.csv").open(
        "w", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "recorded_time_sec",
                "scoped_link_name",
                "model_name",
                "link_name",
                "frame_id",
                "position_x",
                "position_y",
                "position_z",
                "orientation_x",
                "orientation_y",
                "orientation_z",
                "orientation_w",
            ]
        )
        for stamp in (0.0, 1.0, 2.0, 3.0):
            writer.writerow(
                [
                    stamp,
                    scoped_name,
                    "model_01",
                    "ground_truth_v_000",
                    "world",
                    stamp * 0.01,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                ]
            )
    return scoped_name


def convert_high_volume_streams(module, run_dir, backend="sqlite_zlib"):
    stream_names = ("anchor_observations", "risk_evidence", "clusters")
    database_path = run_dir / "algorithm" / "algorithm_frames.sqlite3"
    store = module.CompressedFrameStore(database_path)
    records_by_stream = {}
    for stream_name in stream_names:
        jsonl_path = run_dir / "algorithm" / f"{stream_name}.jsonl"
        records = [
            json.loads(line)
            for line in jsonl_path.read_text().splitlines()
            if line.strip()
        ]
        records_by_stream[stream_name] = records
        for record in records:
            store.write_frame(stream_name, record)
    storage_summary = store.close()

    if backend == "sqlite_zlib":
        for stream_name in stream_names:
            (run_dir / "algorithm" / f"{stream_name}.jsonl").unlink()

    run_info_path = run_dir / "meta" / "run_info.json"
    run_info = json.loads(run_info_path.read_text())
    run_info["algorithm_recording"].update(
        {
            "schema_version": 3,
            "storage_backend": backend,
            "database_file": "algorithm/algorithm_frames.sqlite3",
            "compressed_streams": list(stream_names),
            "codec": "zlib",
            "compression_level": 1,
        }
    )
    write_json(run_info_path, run_info)

    completion_path = run_dir / "meta" / "run_complete.json"
    completion = json.loads(completion_path.read_text())
    completion.update(
        {
            "schema_version": 3,
            "clean_shutdown": True,
            "recording_integrity_valid": True,
            "recording_error": "",
            "algorithm_storage": {
                "backend": backend,
                "database_file": "algorithm/algorithm_frames.sqlite3",
                "integrity_check": storage_summary["integrity_check"],
                "streams": storage_summary["streams"],
            },
        }
    )
    write_json(completion_path, completion)
    return records_by_stream


MOTION_OBJECTS = {
    1: ("l_shape_target", 16, "rotation", (0.0, 0.0, 1.0)),
    2: ("model_01", 8, "translation", (0.0, -1.0, 0.0)),
    3: ("model_02", 8, "translation", (1.0, 0.0, 0.0)),
    4: ("table", 12, "translation", (0.0, 1.0, 0.0)),
    5: ("table_marble", 8, "rotation", (0.0, -1.0, 0.0)),
    6: ("person_walking", 10, "fixed", (0.0, -0.1, 0.0)),
    7: ("table_clone", 12, "rotation", (0.0, 0.0, 1.0)),
    8: ("cardboard_box", 8, "translation", (1.0, 0.0, 0.0)),
    9: ("lzg_bianwood", 8, "rotation", (-1.0, 0.0, 0.0)),
    10: ("bookshelf", 16, "translation", (0.0, 1.0, 0.0)),
}


def _rotate_test_point(point, axis, angle):
    x, y, z = point
    ax, ay, az = axis
    cosine = math.cos(angle)
    sine = math.sin(angle)
    dot = ax * x + ay * y + az * z
    cross = (ay * z - az * y, az * x - ax * z, ax * y - ay * x)
    return tuple(
        cosine * value
        + sine * cross[index]
        + (1.0 - cosine) * dot * axis[index]
        for index, value in enumerate(point)
    )


def _axis_angle_test_quaternion(axis, angle):
    half_angle = 0.5 * angle
    sine = math.sin(half_angle)
    return (
        axis[0] * sine,
        axis[1] * sine,
        axis[2] * sine,
        math.cos(half_angle),
    )


def make_current_motion_execution_run(
    root,
    frozen_model="",
    executed_axis_overrides=None,
    person_second_executed_velocity=None,
):
    run_dir = pathlib.Path(root) / "sim_run_motion"
    meta_dir = run_dir / "meta"
    links_dir = run_dir / "truth" / "links"
    meta_dir.mkdir(parents=True)
    links_dir.mkdir(parents=True)
    executed_axis_overrides = executed_axis_overrides or {}
    person_second_executed_velocity = (
        tuple(person_second_executed_velocity)
        if person_second_executed_velocity is not None
        else (0.0, 0.1, 0.0)
    )
    object_metadata = {}
    controller_models = []
    for object_id, (name, _, mode, command) in MOTION_OBJECTS.items():
        linear_direction = command if mode == "translation" else (0.0, 0.0, 0.0)
        angular_axis = command if mode == "rotation" else (0.0, 0.0, 0.0)
        fixed_linear = command if mode == "fixed" else (0.0, 0.0, 0.0)
        controller_models.append(
            {
                "id": object_id,
                "model_name": name,
                "command_frame": "world",
                "linear_direction": list(linear_direction),
                "angular_axis": list(angular_axis),
                "fixed_linear_mps": list(fixed_linear),
            }
        )
        object_metadata[name] = {
            "motion_profile": (
                "constant_rotation"
                if mode == "rotation"
                else "fixed_speed_translation"
                if mode == "fixed"
                else "constant_translation"
            )
        }
    write_json(
        meta_dir / "scenario_manifest.json",
        {
            "experiment_factors": {
                "motion_protocol_version": "alert10obj_mixed_rotation_v3",
                "truth_protocol_version": "gazebo_drive_link_pose_static_surface_catalog_v3",
            },
            "object_metadata": object_metadata,
        },
    )
    write_json(
        meta_dir / "experiment_protocol.json",
        {
            "actual": {
                "actual_motion_start_time": 10.0,
                "hold_start_time": 50.0,
                "controller_configuration": {
                    "linear_speed_mm_s": 1.0,
                    "angular_speed_rad_s": 0.0015,
                    "person_walking_switch_time": 20.0,
                    "person_walking_second_linear_mps": [0.0, 0.1, 0.0],
                    "models": controller_models,
                },
            }
        },
    )
    header = [
        "recorded_time_sec",
        "scoped_link_name",
        "model_name",
        "link_name",
        "frame_id",
        "position_x",
        "position_y",
        "position_z",
        "orientation_x",
        "orientation_y",
        "orientation_z",
        "orientation_w",
    ]
    surface_catalog = []
    drive_link_contract = {
        "l_shape_target": "link",
        "model_01": "link",
        "model_02": "link",
        "table": "link",
        "table_marble": "link",
        "person_walking": "link",
        "table_clone": "link",
        "cardboard_box": "link",
        "lzg_bianwood": "link_0",
        "bookshelf": "link",
    }
    write_json(
        meta_dir / "object_id_catalog.json",
        {str(object_id): values[0] for object_id, values in MOTION_OBJECTS.items()},
    )
    write_json(
        meta_dir / "run_info.json",
        {
            "truth_recording": {
                "dynamic_link_policy": "sensor_motion_drive_links_and_surface_catalog",
                "motion_truth_drive_links": {
                    name: f"{name}::{link_name}"
                    for name, link_name in drive_link_contract.items()
                },
                "surface_truth_points": {
                    "file": "truth/surface_truth_points.jsonl",
                    "storage_mode": "motion_parent_link_local_pose_once",
                    "catalog_source": "world_static_marker_visual",
                    "expected_point_count": 106,
                    "max_local_radius_m": 3.0,
                },
            }
        },
    )
    for object_id, (name, count, mode, command) in MOTION_OBJECTS.items():
        drive_link_name = drive_link_contract[name]
        drive_scoped_name = f"{name}::{drive_link_name}"
        initial_position = (1.0 + object_id * 0.2, -0.7, 0.4)
        middle_position = None
        final_position = initial_position
        final_quaternion = (0.0, 0.0, 0.0, 1.0)
        if name != frozen_model:
            if mode == "rotation":
                axis = executed_axis_overrides.get(name, command)
                final_quaternion = _axis_angle_test_quaternion(axis, 0.06)
            elif name == "person_walking":
                middle_position = tuple(
                    initial_position[component] + command[component] * 20.0
                    for component in range(3)
                )
                final_position = tuple(
                    middle_position[component]
                    + person_second_executed_velocity[component] * 20.0
                    for component in range(3)
                )
            else:
                velocity = command if mode == "fixed" else tuple(
                    value * 0.001 for value in command
                )
                final_position = tuple(
                    initial_position[component] + velocity[component] * 40.0
                    for component in range(3)
                )
        with (links_dir / f"{name}__{drive_link_name}.csv").open(
            "w", newline=""
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerow(
                [
                    10.0,
                    drive_scoped_name,
                    name,
                    drive_link_name,
                    "world",
                    *initial_position,
                    0,
                    0,
                    0,
                    1,
                ]
            )
            if middle_position is not None:
                writer.writerow(
                    [
                        30.0,
                        drive_scoped_name,
                        name,
                        drive_link_name,
                        "world",
                        *middle_position,
                        0,
                        0,
                        0,
                        1,
                    ]
                )
            writer.writerow(
                [
                    50.0,
                    drive_scoped_name,
                    name,
                    drive_link_name,
                    "world",
                    *final_position,
                    *final_quaternion,
                ]
            )
        for index in range(count):
            local_point = (
                0.3 + (index % 4) * 0.17,
                -0.2 + (index % 3) * 0.19,
                0.1 + (index % 5) * 0.13,
            )
            link_name = f"ground_truth_v_{index:03d}"
            scoped_name = f"{name}::{link_name}"
            surface_catalog.append(
                {
                    "schema_version": 2,
                    "catalog_source": "world_static_marker_visual",
                    "scoped_link_name": scoped_name,
                    "model_name": name,
                    "link_name": link_name,
                    "object_id": object_id,
                    "object_id_valid": True,
                    "object_local_frame": drive_scoped_name,
                    "motion_parent_scoped_link_name": drive_scoped_name,
                    "local_pose": {
                        "position": dict(zip(("x", "y", "z"), local_point)),
                        "orientation": {"x": 0, "y": 0, "z": 0, "w": 1},
                    },
                }
            )
    write_jsonl(run_dir / "truth" / "surface_truth_points.jsonl", surface_catalog)
    return run_dir


class ValidateRecordedRunTests(unittest.TestCase):
    def test_static_surface_truth_catalog_accepts_current_v3_contract(self):
        module = load_module_if_exists()
        with tempfile.TemporaryDirectory(prefix="surface_catalog_validation_") as temp_dir:
            run_dir = make_current_motion_execution_run(temp_dir)
            run_info = json.loads((run_dir / "meta" / "run_info.json").read_text())
            catalog_payload = json.loads(
                (run_dir / "meta" / "object_id_catalog.json").read_text()
            )
            errors = []
            repairs = []
            result = module.validate_static_surface_truth_catalog(
                run_dir,
                run_info,
                module.normalize_catalog(catalog_payload, errors),
                {"truth_protocol_version": module.CURRENT_TRUTH_PROTOCOL_VERSION},
                errors,
                repairs,
            )

        self.assertEqual(result["status"], "PASS", errors)
        self.assertEqual(result["recorded_point_count"], 106)
        self.assertEqual(errors, [])

    def test_static_surface_truth_catalog_rejects_missing_points(self):
        module = load_module_if_exists()
        with tempfile.TemporaryDirectory(prefix="surface_catalog_validation_") as temp_dir:
            run_dir = make_current_motion_execution_run(temp_dir)
            catalog_path = run_dir / "truth" / "surface_truth_points.jsonl"
            rows = [
                json.loads(line)
                for line in catalog_path.read_text().splitlines()
                if line.strip()
            ]
            write_jsonl(catalog_path, rows[:-1])
            run_info = json.loads((run_dir / "meta" / "run_info.json").read_text())
            catalog_payload = json.loads(
                (run_dir / "meta" / "object_id_catalog.json").read_text()
            )
            errors = []
            result = module.validate_static_surface_truth_catalog(
                run_dir,
                run_info,
                module.normalize_catalog(catalog_payload, errors),
                {"truth_protocol_version": module.CURRENT_TRUTH_PROTOCOL_VERSION},
                errors,
                [],
            )

        self.assertEqual(result["status"], "FAIL")
        self.assertIn("surface_truth_catalog:point_count:105!=106", errors)

    def test_current_motion_execution_gate_accepts_all_xyz_motion_modes(self):
        module = load_module_if_exists()
        with tempfile.TemporaryDirectory(prefix="motion_execution_validation_") as temp_dir:
            run_dir = make_current_motion_execution_run(temp_dir)
            result = module.validate_current_motion_execution(
                run_dir, write_report=True
            )

            saved = json.loads(
                (run_dir / "meta" / "motion_execution_validation.json").read_text()
            )

        self.assertEqual(result["status"], "PASS", result["errors"])
        self.assertEqual(result["evaluated_object_count"], 10)
        self.assertEqual(saved["status"], "PASS")
        by_name = {row["object_name"]: row for row in result["objects"]}
        self.assertAlmostEqual(
            by_name["lzg_bianwood"]["actual_angular_speed_rad_s"],
            0.0015,
            places=6,
        )
        self.assertLess(
            by_name["lzg_bianwood"]["actual_rotation_vector_world_rad"][0],
            0.0,
        )
        self.assertLess(
            by_name["table_marble"]["actual_rotation_vector_world_rad"][1],
            0.0,
        )
        self.assertAlmostEqual(
            by_name["bookshelf"]["actual_linear_speed_mps"],
            0.001,
            places=6,
        )
        self.assertAlmostEqual(
            by_name["person_walking"]["actual_linear_speed_mps"],
            0.1,
            places=6,
        )
        self.assertEqual(
            by_name["person_walking"]["execution_profile"],
            "piecewise_fixed_speed_translation",
        )
        self.assertAlmostEqual(
            by_name["person_walking"]["expected_translation_m"], 0.0
        )
        self.assertAlmostEqual(
            by_name["person_walking"]["actual_translation_m"], 0.0
        )
        self.assertAlmostEqual(
            by_name["person_walking"]["actual_path_length_m"], 4.0
        )
        self.assertEqual(
            [row["status"] for row in by_name["person_walking"]["motion_segments"]],
            ["PASS", "PASS"],
        )

    def test_piecewise_walker_cannot_pass_by_remaining_stationary(self):
        module = load_module_if_exists()
        with tempfile.TemporaryDirectory(prefix="motion_execution_validation_") as temp_dir:
            run_dir = make_current_motion_execution_run(
                temp_dir, frozen_model="person_walking"
            )
            result = module.validate_current_motion_execution(
                run_dir, write_report=False
            )

        self.assertEqual(result["status"], "FAIL")
        self.assertTrue(
            any(
                error.startswith(
                    "object:person_walking:segment:outbound:translation_error"
                )
                for error in result["errors"]
            )
        )

    def test_piecewise_walker_rejects_wrong_return_direction(self):
        module = load_module_if_exists()
        with tempfile.TemporaryDirectory(prefix="motion_execution_validation_") as temp_dir:
            run_dir = make_current_motion_execution_run(
                temp_dir,
                person_second_executed_velocity=(0.0, -0.1, 0.0),
            )
            result = module.validate_current_motion_execution(
                run_dir, write_report=False
            )

        self.assertEqual(result["status"], "FAIL")
        self.assertTrue(
            any(
                error.startswith(
                    "object:person_walking:segment:return:translation_direction_cosine"
                )
                for error in result["errors"]
            )
        )

    def test_current_motion_execution_gate_rejects_an_unmoved_object(self):
        module = load_module_if_exists()
        with tempfile.TemporaryDirectory(prefix="motion_execution_validation_") as temp_dir:
            run_dir = make_current_motion_execution_run(
                temp_dir, frozen_model="bookshelf"
            )
            result = module.validate_current_motion_execution(
                run_dir, write_report=False
            )

        self.assertEqual(result["status"], "FAIL")
        self.assertTrue(
            any(
                error.startswith("object:bookshelf:translation_error:")
                for error in result["errors"]
            )
        )

    def test_current_motion_execution_gate_rejects_a_wrong_rotation_axis(self):
        module = load_module_if_exists()
        with tempfile.TemporaryDirectory(prefix="motion_execution_validation_") as temp_dir:
            run_dir = make_current_motion_execution_run(
                temp_dir,
                executed_axis_overrides={"lzg_bianwood": (0.0, 0.0, 1.0)},
            )
            result = module.validate_current_motion_execution(
                run_dir, write_report=False
            )

        self.assertEqual(result["status"], "FAIL")
        self.assertTrue(
            any(
                error.startswith("object:lzg_bianwood:rotation_axis_cosine")
                for error in result["errors"]
            )
        )

    def test_current_motion_execution_gate_rejects_a_wrong_surface_parent(self):
        module = load_module_if_exists()
        with tempfile.TemporaryDirectory(prefix="motion_execution_validation_") as temp_dir:
            run_dir = make_current_motion_execution_run(temp_dir)
            catalog_path = run_dir / "truth" / "surface_truth_points.jsonl"
            records = [
                json.loads(line)
                for line in catalog_path.read_text().splitlines()
                if line.strip()
            ]
            records[0]["motion_parent_scoped_link_name"] = "l_shape_target"
            write_jsonl(catalog_path, records)

            result = module.validate_current_motion_execution(
                run_dir, write_report=False
            )

        self.assertEqual(result["status"], "FAIL")
        self.assertTrue(
            any(
                error.startswith("surface_truth_parent_link:1:")
                for error in result["errors"]
            )
        )

    def test_synchronized_motion_without_direct_link_truth_is_rejected(self):
        module = load_module_if_exists()
        with tempfile.TemporaryDirectory(prefix="recorded_run_validation_") as temp_dir:
            run_dir = make_valid_run(temp_dir)
            write_json(
                run_dir / "meta" / "experiment_protocol.json",
                {
                    "actual": {
                        "controller_configuration": {
                            "models": [{"id": 1, "model_name": "model_01"}]
                        }
                    }
                },
            )
            result = module.validate_run(run_dir)

        self.assertEqual(result["status"], "FAIL")
        self.assertIn(
            "synchronized_motion_requires_direct_link_truth", result["errors"]
        )

    def test_direct_surface_link_truth_is_required_and_validated(self):
        module = load_module_if_exists()
        with tempfile.TemporaryDirectory(prefix="recorded_run_validation_") as temp_dir:
            run_dir = make_valid_run(temp_dir)
            enable_direct_link_truth(run_dir)
            result = module.validate_run(run_dir)

        self.assertEqual(result["status"], "PASS", result["errors"])
        self.assertTrue(result["truth_link_tracks"]["required"])
        self.assertEqual(result["truth_link_tracks"]["track_count"], 1)
        self.assertEqual(result["truth_link_tracks"]["sample_count"], 4)

    def test_missing_declared_surface_link_truth_fails_validation(self):
        module = load_module_if_exists()
        with tempfile.TemporaryDirectory(prefix="recorded_run_validation_") as temp_dir:
            run_dir = make_valid_run(temp_dir)
            scoped_name = enable_direct_link_truth(run_dir, write_track=False)
            result = module.validate_run(run_dir)

        self.assertEqual(result["status"], "FAIL")
        self.assertIn("missing_truth_links_directory", result["errors"])
        self.assertIn(f"missing_link_truth_track:{scoped_name}", result["errors"])

    def test_reference_reset_processing_stamp_does_not_require_monitoring_stats(self):
        module = load_module_if_exists()
        processing = [
            {
                "header": header(1, 1.0),
                "reference_epoch": 0,
                "anchor_count": 5,
            },
            {
                "header": header(2, 2.0),
                "reference_epoch": 1,
                "anchor_count": 0,
            },
        ]
        ordinary_frames = [
            {"header": header(1, 1.0)},
            {"header": header(2, 2.0)},
        ]
        records = {
            "processing_stamps": processing,
            "object_observation_stats": [
                {"header": header(1, 1.0), "phase": 1}
            ],
        }
        for stream_name in module.FRAME_STREAMS - {
            "processing_stamps",
            "object_observation_stats",
        }:
            records[stream_name] = list(ordinary_frames)

        errors = []
        module.validate_cross_stream_coverage(records, errors)
        self.assertEqual(errors, [])

    def test_zero_anchor_monitoring_frame_in_same_epoch_is_not_exempt(self):
        module = load_module_if_exists()
        records = {
            "processing_stamps": [
                {
                    "header": header(1, 1.0),
                    "reference_epoch": 1,
                    "anchor_count": 5,
                },
                {
                    "header": header(2, 2.0),
                    "reference_epoch": 1,
                    "anchor_count": 0,
                }
            ],
            "object_observation_stats": [
                {"header": header(1, 1.0), "phase": 1}
            ],
        }
        for stream_name in module.FRAME_STREAMS - {
            "processing_stamps",
            "object_observation_stats",
        }:
            records[stream_name] = [
                {"header": header(1, 1.0)},
                {"header": header(2, 2.0)},
            ]

        errors = []
        module.validate_cross_stream_coverage(records, errors)
        self.assertIn(
            "processing_stamp_coverage_missing:object_observation_stats:1", errors
        )

    def test_formal_policy_accepts_only_coherent_anchor_gaps_within_limit(self):
        module = load_module_if_exists()
        all_records = [
            {"header": header(index, float(index))}
            for index in range(1, 11)
        ]
        recorded_records = [
            record for index, record in enumerate(all_records, start=1)
            if index != 5
        ]
        records = {
            "anchor_observations": list(recorded_records),
            "processing_stamps": [
                {**record, "anchor_count": 1}
                for record in recorded_records
            ],
        }
        for stream_name in module.FRAME_STREAMS - {
            "anchor_observations",
            "processing_stamps",
        }:
            records[stream_name] = [
                {**record, "phase": 1}
                if stream_name == "object_observation_stats"
                else dict(record)
                for record in all_records
            ]
        completion = {
            "streams": {
                stream_name: {
                    "drop_estimate_available": True,
                    "estimated_drop_count": (
                        1
                        if stream_name
                        in {"anchor_observations", "processing_stamps"}
                        else 0
                    ),
                    "irregular_sequence_delta_count": 0,
                }
                for stream_name in module.FRAME_STREAMS
            }
        }
        coverage = module.build_anchor_processing_coverage_report(
            records, completion
        )
        raw_errors = coverage["accepted_error_codes"]

        formal_policy = module.normalize_validation_policy(
            module.VALIDATION_POLICY_FORMAL_ANALYSIS_V2,
            max_anchor_processing_drop_fraction=0.10,
        )
        blocking, warnings, accepted, classified_coverage = (
            module.classify_validation_findings(
                raw_errors, [], formal_policy, coverage
            )
        )

        self.assertTrue(coverage["coherent"], coverage["coherence_errors"])
        self.assertAlmostEqual(coverage["coverage_fraction"], 0.9)
        self.assertEqual(blocking, [])
        self.assertEqual(warnings, accepted)
        self.assertEqual(sorted(raw_errors), accepted)
        self.assertTrue(classified_coverage["within_policy_limit"])

        tighter_policy = module.normalize_validation_policy(
            module.VALIDATION_POLICY_FORMAL_ANALYSIS_V2,
            max_anchor_processing_drop_fraction=0.09,
        )
        blocking, _, accepted, classified_coverage = (
            module.classify_validation_findings(
                raw_errors, [], tighter_policy, coverage
            )
        )
        self.assertEqual(sorted(raw_errors), blocking)
        self.assertEqual(accepted, [])
        self.assertFalse(classified_coverage["within_policy_limit"])

    def test_formal_policy_keeps_real_motion_failures_blocking(self):
        module = load_module_if_exists()
        coverage = {
            "coherent": True,
            "drop_fraction": 0.0,
            "accepted_error_codes": [],
        }
        raw_errors = [
            "motion_execution:object:bookshelf:translation_error:1>0.1"
        ]
        formal = module.normalize_validation_policy(
            module.VALIDATION_POLICY_FORMAL_ANALYSIS_V2
        )
        recording = module.normalize_validation_policy(
            module.VALIDATION_POLICY_RECORDING_V2
        )

        formal_blocking, _, formal_accepted, _ = (
            module.classify_validation_findings(
                raw_errors, [], formal, coverage
            )
        )
        recording_blocking, _, recording_accepted, _ = (
            module.classify_validation_findings(
                raw_errors, [], recording, coverage
            )
        )

        self.assertEqual(formal_blocking, raw_errors)
        self.assertEqual(formal_accepted, [])
        self.assertEqual(recording_blocking, [])
        self.assertEqual(recording_accepted, raw_errors)

    def test_valid_schema_v3_sqlite_run_passes_without_large_jsonl_files(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        with tempfile.TemporaryDirectory(prefix="recorded_run_validation_") as temp_dir:
            run_dir = make_valid_run(temp_dir)
            expected = convert_high_volume_streams(module, run_dir)
            consumed = {stream_name: [] for stream_name in expected}
            result = module.validate_run(
                run_dir,
                stream_consumers={
                    stream_name: rows.append
                    for stream_name, rows in consumed.items()
                },
            )

        self.assertEqual(result["status"], "PASS", result["errors"])
        self.assertTrue(result["valid_for_analysis"])
        for stream_name, records in expected.items():
            self.assertEqual(result["streams"][stream_name]["row_count"], len(records))
            self.assertEqual(len(consumed[stream_name]), len(records))

    def test_sqlite_payload_corruption_fails_validation(self):
        module = load_module_if_exists()
        with tempfile.TemporaryDirectory(prefix="recorded_run_validation_") as temp_dir:
            run_dir = make_valid_run(temp_dir)
            convert_high_volume_streams(module, run_dir)
            database_path = run_dir / "algorithm" / "algorithm_frames.sqlite3"
            with sqlite3.connect(str(database_path)) as connection:
                connection.execute(
                    """
                    UPDATE stream_frames
                       SET compressed_payload = ?
                     WHERE stream_name = 'clusters' AND frame_pk = (
                         SELECT MIN(frame_pk) FROM stream_frames
                          WHERE stream_name = 'clusters'
                     )
                    """,
                    (sqlite3.Binary(b"corrupt"),),
                )
            result = module.validate_run(run_dir)

        self.assertEqual(result["status"], "FAIL")
        self.assertTrue(
            any("algorithm_storage_error:clusters" in item for item in result["errors"])
        )

    def test_dual_payload_mismatch_fails_validation(self):
        module = load_module_if_exists()
        with tempfile.TemporaryDirectory(prefix="recorded_run_validation_") as temp_dir:
            run_dir = make_valid_run(temp_dir)
            convert_high_volume_streams(module, run_dir, backend="dual")
            risk_path = run_dir / "algorithm" / "risk_evidence.jsonl"
            records = [
                json.loads(line)
                for line in risk_path.read_text().splitlines()
                if line.strip()
            ]
            records[0]["evidences"] = [{"active": True, "risk_score": 99.0}]
            write_jsonl(risk_path, records)
            result = module.validate_run(run_dir)

        self.assertEqual(result["status"], "FAIL")
        self.assertIn("dual_algorithm_storage_mismatch", result["errors"])

    def test_run_complete_database_count_mismatch_fails_validation(self):
        module = load_module_if_exists()
        with tempfile.TemporaryDirectory(prefix="recorded_run_validation_") as temp_dir:
            run_dir = make_valid_run(temp_dir)
            convert_high_volume_streams(module, run_dir)
            completion_path = run_dir / "meta" / "run_complete.json"
            completion = json.loads(completion_path.read_text())
            completion["algorithm_storage"]["streams"]["clusters"][
                "frame_count"
            ] = 999
            write_json(completion_path, completion)
            result = module.validate_run(run_dir)

        self.assertEqual(result["status"], "FAIL")
        self.assertTrue(
            any(
                "algorithm_storage_frame_count_mismatch:clusters" in item
                for item in result["errors"]
            )
        )

    def test_recording_integrity_flag_false_fails_validation(self):
        module = load_module_if_exists()
        with tempfile.TemporaryDirectory(prefix="recorded_run_validation_") as temp_dir:
            run_dir = make_valid_run(temp_dir)
            convert_high_volume_streams(module, run_dir)
            completion_path = run_dir / "meta" / "run_complete.json"
            completion = json.loads(completion_path.read_text())
            completion["recording_integrity_valid"] = False
            completion["recording_error"] = "queue timeout"
            write_json(completion_path, completion)
            result = module.validate_run(run_dir)

        self.assertEqual(result["status"], "FAIL")
        self.assertIn("recording_integrity_invalid", result["errors"])

    def test_anchor_catalog_conflict_rows_fail_validation(self):
        module = load_module_if_exists()
        with tempfile.TemporaryDirectory(prefix="recorded_run_validation_") as temp_dir:
            run_dir = make_valid_run(temp_dir)
            write_jsonl(
                run_dir / "algorithm" / "anchor_catalog_conflicts.jsonl",
                [{"reference_epoch": 1, "anchor_id": 7}],
            )
            result = module.validate_run(run_dir)

        self.assertEqual(result["status"], "FAIL")
        self.assertIn("anchor_catalog_conflicts_present:1", result["errors"])

    def test_valid_complete_run_passes_and_writes_report(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        with tempfile.TemporaryDirectory(prefix="recorded_run_validation_") as temp_dir:
            run_dir = make_valid_run(temp_dir)
            result = module.validate_run(run_dir)

            self.assertEqual(result["status"], "PASS")
            self.assertEqual(result["errors"], [])
            self.assertTrue((run_dir / "analysis" / "data_quality_report.json").is_file())
            self.assertEqual(result["streams"]["processing_stamps"]["row_count"], 2)

    def test_stream_consumer_receives_each_cluster_record_during_validation(self):
        module = load_module_if_exists()
        with tempfile.TemporaryDirectory(prefix="recorded_run_consumer_") as temp_dir:
            run_dir = make_valid_run(temp_dir)
            received = []

            result = module.validate_run(
                run_dir,
                stream_consumers={"clusters": received.append},
            )

        self.assertEqual(result["status"], "PASS")
        self.assertEqual(len(received), 2)
        self.assertEqual(received[0]["reference_epoch"], 1)

    def test_malformed_nonfinite_and_conflicting_timestamp_fail_with_line_context(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        with tempfile.TemporaryDirectory(prefix="recorded_run_validation_") as temp_dir:
            run_dir = make_valid_run(temp_dir)
            clusters_path = run_dir / "algorithm" / "clusters.jsonl"
            with clusters_path.open("a") as handle:
                handle.write("{malformed\n")
            risk_path = run_dir / "algorithm" / "risk_evidence.jsonl"
            with risk_path.open("a") as handle:
                handle.write('{"header":{"stamp":{"sec":2.0}},"value":NaN}\n')
            observations = [
                json.loads(line)
                for line in (run_dir / "algorithm" / "anchor_observations.jsonl")
                .read_text()
                .splitlines()
            ]
            conflicting = dict(observations[-1])
            conflicting["anchors"] = []
            with (run_dir / "algorithm" / "anchor_observations.jsonl").open("a") as handle:
                json.dump(conflicting, handle)
                handle.write("\n")

            result = module.validate_run(run_dir)

        joined = ";".join(result["errors"])
        self.assertEqual(result["status"], "FAIL")
        self.assertIn("malformed_json:algorithm/clusters.jsonl:3", joined)
        self.assertIn("non_finite_json:algorithm/risk_evidence.jsonl:3", joined)
        self.assertIn("conflicting_duplicate_timestamp:anchor_observations", joined)

    def test_unknown_object_id_missing_truth_and_processing_stamps_fail(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        with tempfile.TemporaryDirectory(prefix="recorded_run_validation_") as temp_dir:
            run_dir = make_valid_run(temp_dir)
            catalog_path = run_dir / "algorithm" / "anchor_catalog.jsonl"
            record = json.loads(catalog_path.read_text())
            record["anchor"]["object_id"] = 99
            write_jsonl(catalog_path, [record])
            (run_dir / "truth" / "objects" / "model_01.csv").unlink()
            (run_dir / "algorithm" / "processing_stamps.jsonl").unlink()

            result = module.validate_run(run_dir)

        joined = ";".join(result["errors"])
        self.assertEqual(result["status"], "FAIL")
        self.assertIn("unknown_object_id:99", joined)
        self.assertIn("missing_truth_track:model_01", joined)
        self.assertIn("missing_stream:processing_stamps", joined)

    def test_identical_duplicate_and_small_quaternion_error_are_logged_repairs(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        with tempfile.TemporaryDirectory(prefix="recorded_run_validation_") as temp_dir:
            run_dir = make_valid_run(temp_dir)
            path = run_dir / "algorithm" / "risk_regions.jsonl"
            first_line = path.read_text().splitlines()[0]
            with path.open("a") as handle:
                handle.write(first_line + "\n")
            truth_path = run_dir / "truth" / "objects" / "model_01.csv"
            rows = list(csv.DictReader(truth_path.read_text().splitlines()))
            rows[1]["orientation_w"] = "1.0001"
            with truth_path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)

            result = module.validate_run(run_dir)

        self.assertNotEqual(result["status"], "FAIL")
        repairs = ";".join(result["repairs"])
        self.assertIn("identical_duplicate_removed:risk_regions", repairs)
        self.assertIn("quaternion_renormalized:truth/objects/model_01.csv", repairs)


if __name__ == "__main__":
    unittest.main()
