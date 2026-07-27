import csv
import importlib.util
import json
import math
import pathlib
import tempfile
import unittest


SCRIPT_PATH = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "analyze_sim_run.py"


def load_module_if_exists():
    if not SCRIPT_PATH.exists():
        return None

    spec = importlib.util.spec_from_file_location("analyze_sim_run", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_identity_alignment(meta_dir):
    meta_dir.mkdir(parents=True, exist_ok=True)
    with (meta_dir / "frame_alignment.json").open("w") as handle:
        json.dump(
            {
                "truth_frame": "world",
                "algorithm_frame": "camera_init",
                "alignment_mode": "initial_ego_pose",
                "sim_only": True,
                "ego_initial_pose_world": {
                    "position": {"x": 0.0, "y": 0.0, "z": 0.0},
                    "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                },
                "world_from_algorithm_transform": {
                    "source_frame": "camera_init",
                    "target_frame": "world",
                    "pose": {
                        "position": {"x": 0.0, "y": 0.0, "z": 0.0},
                        "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                    },
                },
                "algorithm_from_world_transform": {
                    "source_frame": "world",
                    "target_frame": "camera_init",
                    "pose": {
                        "position": {"x": 0.0, "y": 0.0, "z": 0.0},
                        "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                    },
                },
            },
            handle,
        )


def write_moving_truth_object(truth_objects_dir, object_name="moving_block", points=None):
    truth_objects_dir.mkdir(parents=True, exist_ok=True)
    points = points or [
        (0.0, 0.0, 0.0, 0.0),
        (1.0, 0.02, 0.0, 0.0),
    ]
    csv_path = truth_objects_dir / f"{object_name}.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "recorded_time_sec",
                "model_name",
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
        for row in points:
            time_sec, x, y, z = row
            writer.writerow([time_sec, object_name, "world", x, y, z, 0.0, 0.0, 0.0, 1.0])


def persistent_region_entry(track_id, center, confirmed=True, state=1, region_type=1):
    return {
        "track_id": track_id,
        "state": state,
        "region_type": region_type,
        "center": {"x": center[0], "y": center[1], "z": center[2]},
        "bbox_min": {"x": center[0] - 0.1, "y": center[1] - 0.1, "z": center[2] - 0.1},
        "bbox_max": {"x": center[0] + 0.1, "y": center[1] + 0.1, "z": center[2] + 0.1},
        "mean_risk": 0.4,
        "peak_risk": 0.8,
        "confidence": 0.6,
        "accumulated_risk": 1.2,
        "support_mass": 3.0,
        "spatial_span": 0.6,
        "hit_streak": 3,
        "miss_streak": 0,
        "age_frames": 4,
        "confirmed": confirmed,
    }


def write_jsonl_records(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for record in records:
            json.dump(record, handle)
            handle.write("\n")


def write_world_file(world_path, model_name, box_size):
    world_path.parent.mkdir(parents=True, exist_ok=True)
    world_path.write_text(
        f"""
<sdf version='1.7'>
  <world name='test_world'>
    <model name='{model_name}'>
      <static>1</static>
      <pose>0 0 0 0 0 0</pose>
      <link name='link'>
        <collision name='collision'>
          <geometry>
            <box>
              <size>{box_size[0]} {box_size[1]} {box_size[2]}</size>
            </box>
          </geometry>
        </collision>
      </link>
    </model>
  </world>
</sdf>
""".strip()
    )


class AnalyzeSimRunTests(unittest.TestCase):
    def test_anchor_type_inventory_keeps_unassociated_anchors(self):
        module = load_module_if_exists()
        catalog = [
            {
                "reference_epoch": 1,
                "anchor_id": 7,
                "anchor": {"id": 7, "anchor_type": 0},
            },
            {
                "reference_epoch": 1,
                "anchor_id": 8,
                "anchor": {"id": 8, "anchor_type": 1},
            },
            {
                "reference_epoch": 0,
                "anchor_id": 9,
                "anchor": {"id": 9, "anchor_type": 2},
            },
        ]
        records = [{
            "reference_epoch": 1,
            "anchors": [
                {
                    "id": 7,
                    "anchor_type": 0,
                    "observable": True,
                    "comparable": True,
                    "obs_state": 1,
                    "significant": True,
                    "object_id_valid": True,
                    "object_association_state": 1,
                },
                {
                    "id": 8,
                    "anchor_type": 1,
                    "observable": True,
                    "comparable": False,
                    "obs_state": 2,
                    "significant": False,
                    "object_id_valid": False,
                    "object_association_state": 0,
                },
            ],
        }]

        rows = module.build_anchor_type_inventory(catalog, records, formal_epoch=1)
        by_type = {row["anchor_type"]: row for row in rows}

        self.assertEqual(by_type["PLANE"]["catalog_anchor_count"], 1)
        self.assertEqual(by_type["PLANE"]["significant_anchor_count"], 1)
        self.assertEqual(by_type["EDGE"]["catalog_anchor_count"], 1)
        self.assertEqual(by_type["EDGE"]["object_id_valid_sample_count"], 0)
        self.assertEqual(
            by_type["EDGE"]["association_unavailable_sample_count"], 1
        )
        self.assertEqual(by_type["BAND"]["catalog_anchor_count"], 0)

    def test_safe_f1_distinguishes_complete_miss_from_no_positive_truth(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        self.assertEqual(module._safe_f1("", 0.0), 0.0)
        self.assertEqual(module._safe_f1("", ""), "")
        self.assertAlmostEqual(module._safe_f1(0.5, 1.0), 2.0 / 3.0)

    def test_truth_motion_policy_requires_sustained_motion_and_uses_twist(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        track = module.TruthTrack(
            object_name="panel",
            time_sec=[0.0, 1.0, 2.0, 3.0],
            x=[0.0, 0.002, 0.003, 0.004],
            y=[0.0] * 4,
            z=[0.0] * 4,
            qx=[0.0] * 4,
            qy=[0.0] * 4,
            qz=[0.0] * 4,
            qw=[1.0] * 4,
            vx=[0.0, 0.0, 0.002, 0.003],
            vy=[0.0] * 4,
            vz=[0.0] * 4,
            wx=[0.0] * 4,
            wy=[0.0] * 4,
            wz=[0.0] * 4,
        )
        policy = module.TruthMotionPolicy(
            translation_deadband_m=0.01,
            rotation_deadband_deg=0.05,
            linear_speed_deadband_mps=0.001,
            angular_speed_deadband_degps=0.01,
            sustained_motion_samples=2,
        )

        metrics = module.truth_track_metrics(track, policy=policy)

        self.assertTrue(metrics["moving"])
        self.assertEqual(metrics["start_time"], 2.0)
        self.assertAlmostEqual(metrics["peak_linear_speed_mps"], 0.003)
        self.assertGreater(metrics["peak_linear_acceleration_mps2"], 0.0)

    def test_load_truth_motion_policy_from_recorded_run_info(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        with tempfile.TemporaryDirectory(prefix="truth_policy_") as temp_dir:
            run_dir = pathlib.Path(temp_dir)
            meta_dir = run_dir / "meta"
            meta_dir.mkdir()
            (meta_dir / "run_info.json").write_text(
                json.dumps(
                    {
                        "truth_motion_policy": {
                            "translation_deadband_m": 0.002,
                            "rotation_deadband_deg": 0.1,
                            "linear_speed_deadband_mps": 0.0008,
                            "angular_speed_deadband_degps": 0.02,
                            "sustained_motion_samples": 3,
                        }
                    }
                )
            )

            policy = module.load_truth_motion_policy(run_dir)

        self.assertEqual(policy.translation_deadband_m, 0.002)
        self.assertEqual(policy.sustained_motion_samples, 3)

    def test_surface_truth_catalog_reconstructs_rotation_from_root_track(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        with tempfile.TemporaryDirectory(prefix="surface_truth_catalog_") as temp_dir:
            run_dir = pathlib.Path(temp_dir)
            truth_dir = run_dir / "truth"
            truth_dir.mkdir()
            root_half = math.sqrt(0.5)
            track = module.TruthTrack(
                object_name="panel",
                time_sec=[0.0, 1.0],
                x=[0.0, 0.0],
                y=[0.0, 0.0],
                z=[0.0, 0.0],
                qx=[0.0, 0.0],
                qy=[0.0, 0.0],
                qz=[0.0, root_half],
                qw=[1.0, root_half],
            )
            write_jsonl_records(
                truth_dir / "surface_truth_points.jsonl",
                [
                    {
                        "scoped_link_name": "panel::ground_truth_corner0",
                        "model_name": "panel",
                        "link_name": "ground_truth_corner0",
                        "local_pose": {
                            "position": {"x": 1.0, "y": 0.0, "z": 0.0},
                            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                        },
                    }
                ],
            )

            links = module.load_surface_truth_link_tracks(run_dir, [track])

        self.assertEqual(len(links), 1)
        self.assertAlmostEqual(links[0].x[0], 1.0)
        self.assertAlmostEqual(links[0].x[1], 0.0, places=7)
        self.assertAlmostEqual(links[0].y[1], 1.0, places=7)

    def test_surface_truth_catalog_prefers_rotating_drive_link_over_static_root(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        with tempfile.TemporaryDirectory(prefix="drive_link_surface_truth_") as temp_dir:
            run_dir = pathlib.Path(temp_dir)
            truth_dir = run_dir / "truth"
            truth_dir.mkdir()
            root = module.TruthTrack(
                object_name="panel",
                time_sec=[0.0, 1.0],
                x=[100.0, 100.0],
                y=[0.0, 0.0],
                z=[0.0, 0.0],
                qx=[0.0, 0.0],
                qy=[0.0, 0.0],
                qz=[0.0, 0.0],
                qw=[1.0, 1.0],
            )
            half = math.sqrt(0.5)
            drive = module.LinkTrack(
                scoped_link_name="panel::link",
                model_name="panel",
                link_name="link",
                time_sec=[0.0, 1.0],
                x=[10.0, 10.0],
                y=[20.0, 20.0],
                z=[0.0, 0.0],
                qx=[0.0, 0.0],
                qy=[0.0, 0.0],
                qz=[0.0, half],
                qw=[1.0, half],
            )
            write_jsonl_records(
                truth_dir / "surface_truth_points.jsonl",
                [
                    {
                        "scoped_link_name": "panel::ground_truth_corner0",
                        "model_name": "panel",
                        "link_name": "ground_truth_corner0",
                        "motion_parent_scoped_link_name": "panel::link",
                        "local_pose": {
                            "position": {"x": 1.0, "y": 0.0, "z": 0.0},
                            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                        },
                    }
                ],
            )

            links = module.load_surface_truth_link_tracks(
                run_dir, [root], [drive]
            )

        self.assertEqual(len(links), 1)
        self.assertAlmostEqual(links[0].x[0], 11.0)
        self.assertAlmostEqual(links[0].y[0], 20.0)
        self.assertAlmostEqual(links[0].x[1], 10.0, places=7)
        self.assertAlmostEqual(links[0].y[1], 21.0, places=7)
        self.assertEqual(module.classify_truth_bundle(root, [drive]), "moving")

    def test_rigid_point_velocity_uses_omega_cross_r_for_rotation(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        track = module.TruthTrack(
            object_name="panel",
            time_sec=[0.0, 1.0],
            x=[0.0, 0.0],
            y=[0.0, 0.0],
            z=[0.0, 0.0],
            qx=[0.0, 0.0],
            qy=[0.0, 0.0],
            qz=[0.0, 0.0],
            qw=[1.0, 1.0],
            vx=[0.0, 0.0],
            vy=[0.0, 0.0],
            vz=[0.0, 0.0],
            wx=[0.0, 0.0],
            wy=[0.0, 0.0],
            wz=[1.0, 1.0],
        )

        state = module.rigid_point_truth_state(
            track,
            reference_point_world=module.np.array([1.0, 0.0, 0.0]),
            reference_time_sec=0.0,
            time_sec=1.0,
        )

        self.assertAlmostEqual(state["velocity_world"][0], 0.0)
        self.assertAlmostEqual(state["velocity_world"][1], 1.0)
        self.assertAlmostEqual(state["velocity_world"][2], 0.0)

    def test_load_anchor_records_supports_compact_schema(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        with tempfile.TemporaryDirectory(prefix="sim_run_compact_anchor_") as temp_dir:
            run_dir = pathlib.Path(temp_dir)
            algorithm_dir = run_dir / "algorithm"
            algorithm_dir.mkdir()
            write_jsonl_records(
                algorithm_dir / "anchor_catalog.jsonl",
                [
                    {
                        "schema_version": 2,
                        "reference_epoch": 2,
                        "anchor_id": 9,
                        "anchor": {
                            "id": 9,
                            "anchor_type": 2,
                            "object_id": 31,
                            "object_id_valid": True,
                            "ref_center": {"x": 1.0, "y": 0.0, "z": 0.0},
                            "reference_epoch": 2,
                        },
                    }
                ],
            )
            write_jsonl_records(
                algorithm_dir / "anchor_observations.jsonl",
                [
                    {
                        "schema_version": 2,
                        "header": {"stamp": {"sec": 4.0}},
                        "reference_epoch": 2,
                        "reference_initialized_at": {"sec": 1.0},
                        "anchors": [
                            {
                                "id": 9,
                                "significant": True,
                                "matched_delta": {
                                    "x": 0.03,
                                    "y": 0.0,
                                    "z": 0.0,
                                },
                            }
                        ],
                    }
                ],
            )

            records = module.load_anchor_state_records(run_dir)
            self.assertIsInstance(records, module.ReplayableSequence)
            loaded_anchor = records[0]["anchors"][0]

        self.assertEqual(loaded_anchor["anchor_type"], 2)
        self.assertEqual(loaded_anchor["object_id"], 31)
        self.assertEqual(loaded_anchor["matched_delta"]["x"], 0.03)

    def test_load_anchor_records_supports_sqlite_observation_stream(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        with tempfile.TemporaryDirectory(prefix="sim_run_sqlite_anchor_") as temp_dir:
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
            write_jsonl_records(
                algorithm_dir / "anchor_catalog.jsonl",
                [
                    {
                        "reference_epoch": 2,
                        "anchor_id": 9,
                        "anchor": {
                            "id": 9,
                            "anchor_type": 2,
                            "object_id": 31,
                            "object_id_valid": True,
                        },
                    }
                ],
            )
            observation = {
                "schema_version": 2,
                "header": {"seq": 4, "stamp": {"secs": 4, "nsecs": 0}},
                "reference_epoch": 2,
                "anchors": [
                    {
                        "id": 9,
                        "significant": True,
                        "matched_delta": {"x": 0.03, "y": 0.0, "z": 0.0},
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

        self.assertEqual(loaded_anchor["anchor_type"], 2)
        self.assertEqual(loaded_anchor["object_id"], 31)
        self.assertEqual(loaded_anchor["matched_delta"]["x"], 0.03)

    def test_anchor_metrics_use_object_ids_and_report_each_anchor_type(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        moving = module.TruthTrack(
            object_name="moving_box",
            time_sec=[0.0, 1.0, 2.0],
            x=[0.0, 0.02, 0.04],
            y=[0.0, 0.0, 0.0],
            z=[0.0, 0.0, 0.0],
        )
        static = module.TruthTrack(
            object_name="static_wall",
            time_sec=[0.0, 1.0, 2.0],
            x=[0.0, 0.0, 0.0],
            y=[0.0, 0.0, 0.0],
            z=[0.0, 0.0, 0.0],
        )
        records = [
            {
                "header": {"stamp": {"sec": 1.0}},
                "reference_epoch": 1,
                "anchors": [
                    {"id": 1, "anchor_type": 0, "object_id": 31,
                     "object_id_valid": True, "observable": True,
                     "comparable": True, "obs_state": 1, "significant": False,
                     "ref_quality": 0.90, "covariance_quality": 0.80,
                     "type_stability": 0.70, "shape_linearity": 0.20,
                     "shape_planarity": 0.70, "shape_scattering": 0.10},
                    {"id": 4, "anchor_type": 0, "object_id": 31,
                     "object_id_valid": True, "observable": False,
                     "comparable": False, "obs_state": 3, "significant": False,
                     "ref_quality": 0.80, "covariance_quality": 0.70,
                     "type_stability": 0.60, "shape_linearity": 0.30,
                     "shape_planarity": 0.60, "shape_scattering": 0.12},
                    {"id": 2, "anchor_type": 1, "object_id": 31,
                     "object_id_valid": True, "observable": True,
                     "comparable": False, "obs_state": 2, "significant": False},
                    {"id": 3, "anchor_type": 0, "object_id": 32,
                     "object_id_valid": True, "significant": False},
                ],
            },
            {
                "header": {"stamp": {"sec": 2.0}},
                "reference_epoch": 1,
                "anchors": [
                    {"id": 1, "anchor_type": 0, "object_id": 31,
                     "object_id_valid": True, "observable": True,
                     "comparable": True, "obs_state": 1, "significant": True,
                     "ref_quality": 0.90, "covariance_quality": 0.80,
                     "type_stability": 0.70, "shape_linearity": 0.20,
                     "shape_planarity": 0.70, "shape_scattering": 0.10},
                    {"id": 4, "anchor_type": 0, "object_id": 31,
                     "object_id_valid": True, "observable": False,
                     "comparable": False, "obs_state": 4, "significant": True,
                     "ref_quality": 0.80, "covariance_quality": 0.70,
                     "type_stability": 0.60, "shape_linearity": 0.30,
                     "shape_planarity": 0.60, "shape_scattering": 0.12},
                    {"id": 2, "anchor_type": 1, "object_id": 31,
                     "object_id_valid": True, "observable": False,
                     "comparable": False, "obs_state": 0, "significant": False},
                    {"id": 3, "anchor_type": 0, "object_id": 32,
                     "object_id_valid": True, "significant": True},
                ],
            },
        ]

        object_rows, type_rows = module.build_anchor_detection_metrics(
            records,
            [moving, static],
            {},
            {31: "moving_box", 32: "static_wall"},
        )

        by_type = {row["anchor_type"]: row for row in type_rows}
        self.assertEqual(by_type["PLANE"]["tp"], 1)
        self.assertEqual(by_type["PLANE"]["fp"], 1)
        self.assertEqual(by_type["EDGE"]["fn"], 1)
        self.assertEqual(by_type["BAND"]["not_evaluable"], 2)
        moving_plane = next(
            row for row in object_rows
            if row["object_name"] == "moving_box" and row["anchor_type"] == "PLANE"
        )
        self.assertEqual(moving_plane["outcome"], "TP")
        self.assertEqual(moving_plane["first_detection_time"], 2.0)
        self.assertEqual(moving_plane["detection_delay_sec"], 1.0)
        self.assertEqual(moving_plane["significant_frame_count"], 1)
        self.assertEqual(moving_plane["anchor_sample_count"], 4)
        self.assertAlmostEqual(moving_plane["mean_anchor_count_per_frame"], 2.0)
        self.assertAlmostEqual(moving_plane["observable_rate"], 0.5)
        self.assertAlmostEqual(moving_plane["comparable_rate"], 0.5)
        self.assertAlmostEqual(moving_plane["matched_rate"], 0.5)
        self.assertAlmostEqual(moving_plane["loss_rate"], 0.5)
        self.assertAlmostEqual(moving_plane["mean_ref_quality"], 0.85)
        self.assertAlmostEqual(moving_plane["mean_covariance_quality"], 0.75)
        self.assertAlmostEqual(moving_plane["mean_type_stability"], 0.65)
        self.assertAlmostEqual(moving_plane["mean_shape_linearity"], 0.25)
        self.assertAlmostEqual(moving_plane["mean_shape_planarity"], 0.65)
        self.assertAlmostEqual(moving_plane["mean_shape_scattering"], 0.11)
        self.assertEqual(by_type["PLANE"]["anchor_sample_count"], 6)
        self.assertEqual(by_type["PLANE"]["observable_sample_count"], 2)
        self.assertEqual(by_type["PLANE"]["comparable_sample_count"], 2)
        self.assertEqual(by_type["PLANE"]["matched_sample_count"], 2)
        self.assertEqual(by_type["PLANE"]["loss_sample_count"], 2)

    def test_object_hit_exposure_controls_anchor_evaluability(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        moving = module.TruthTrack(
            object_name="moving_box",
            time_sec=[0.0, 1.0, 2.0],
            x=[0.0, 0.0, 0.02], y=[0.0, 0.0, 0.0], z=[0.0, 0.0, 0.0],
        )
        hidden = module.TruthTrack(
            object_name="hidden_box",
            time_sec=[0.0, 1.0, 2.0],
            x=[1.0, 1.0, 1.02], y=[0.0, 0.0, 0.0], z=[0.0, 0.0, 0.0],
        )
        hit_records = [
            {
                "header": {"stamp": {"sec": 0.0}}, "phase": 1,
                "window_start": {"sec": -1.0}, "window_end": {"sec": 0.0},
                "frame_count": 5,
                "objects": [{"object_id": 32, "point_count": 80,
                             "visible_frame_count": 4}],
            },
            {
                "header": {"stamp": {"sec": 2.0}}, "phase": 1,
                "window_start": {"sec": 1.0}, "window_end": {"sec": 2.0},
                "frame_count": 5,
                "objects": [{"object_id": 31, "point_count": 120,
                             "visible_frame_count": 4}],
            },
        ]
        anchor_records = [{
            "header": {"stamp": {"sec": 2.0}},
            "reference_epoch": 1,
            "anchors": [],
        }]

        exposure = module.build_object_hit_exposure(
            hit_records, {31: "moving_box", 32: "hidden_box"}
        )
        rows, type_rows = module.build_anchor_detection_metrics(
            anchor_records,
            [moving, hidden],
            {},
            {31: "moving_box", 32: "hidden_box"},
            object_exposure=exposure,
        )

        all_rows = {row["object_name"]: row for row in rows
                    if row["anchor_type"] == "ALL"}
        self.assertEqual(all_rows["moving_box"]["evaluation_status"],
                         "OBSERVED_WITHOUT_ANCHOR")
        self.assertEqual(all_rows["moving_box"]["outcome"], "NOT_EVALUABLE")
        self.assertEqual(all_rows["moving_box"]["hit_point_count"], 120)
        self.assertEqual(all_rows["moving_box"]["exposure_frame_count"], 5)
        self.assertEqual(all_rows["moving_box"]["exposure_window_count"], 1)
        self.assertAlmostEqual(all_rows["moving_box"]["lidar_visibility_rate"], 0.8)
        self.assertAlmostEqual(all_rows["moving_box"]["hit_window_rate"], 1.0)
        self.assertEqual(all_rows["hidden_box"]["evaluation_status"],
                         "LIDAR_UNOBSERVED")
        self.assertEqual(all_rows["hidden_box"]["outcome"], "NOT_EVALUABLE")
        self.assertEqual(all_rows["hidden_box"]["exposure_frame_count"], 5)
        self.assertEqual(all_rows["hidden_box"]["exposure_window_count"], 1)
        self.assertEqual(all_rows["hidden_box"]["lidar_visibility_rate"], 0.0)
        self.assertEqual(all_rows["hidden_box"]["hit_window_rate"], 0.0)
        self.assertEqual(next(row for row in type_rows
                              if row["anchor_type"] == "ALL")["not_evaluable"], 2)

    def test_anchor_association_audit_separates_consistent_mismatch_and_mixed(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        moving = module.TruthTrack(
            object_name="moving_box", time_sec=[0.0, 1.0],
            x=[0.0, 0.02], y=[0.0, 0.0], z=[0.0, 0.0],
        )
        anchors = []
        for anchor_id, state in enumerate((1, 2, 3, 0), start=1):
            anchors.append({
                "id": anchor_id, "anchor_type": 0, "object_id": 31,
                "object_id_valid": True, "object_association_state": state,
                "observable": True, "comparable": True, "obs_state": 1,
                "significant": state == 1,
            })
        records = [{"header": {"stamp": {"sec": 1.0}}, "anchors": anchors}]

        rows, _ = module.build_anchor_detection_metrics(
            records, [moving], {}, {31: "moving_box"}
        )

        plane = next(row for row in rows if row["anchor_type"] == "PLANE")
        self.assertEqual(plane["association_consistent_sample_count"], 1)
        self.assertEqual(plane["association_mismatch_sample_count"], 1)
        self.assertEqual(plane["association_mixed_sample_count"], 1)
        self.assertEqual(plane["association_unavailable_sample_count"], 1)
        self.assertAlmostEqual(plane["association_consistency_rate"], 0.25)

    def test_object_without_any_anchor_is_not_evaluable_in_all_scope(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        moving = module.TruthTrack(
            object_name="unobserved_block",
            time_sec=[0.0, 1.0],
            x=[0.0, 0.02],
            y=[0.0, 0.0],
            z=[0.0, 0.0],
        )
        records = [{
            "header": {"stamp": {"sec": 1.0}},
            "reference_epoch": 1,
            "anchors": [],
        }]

        object_rows, type_rows = module.build_anchor_detection_metrics(
            records, [moving], {}, {31: "unobserved_block"}
        )

        by_scope = {row["anchor_type"]: row for row in object_rows}
        self.assertEqual(by_scope["ALL"]["outcome"], "NOT_EVALUABLE")
        self.assertEqual(by_scope["PLANE"]["outcome"], "NOT_EVALUABLE")
        aggregate_all = next(row for row in type_rows if row["anchor_type"] == "ALL")
        self.assertEqual(aggregate_all["fn"], 0)
        self.assertEqual(aggregate_all["not_evaluable"], 1)

    def test_load_object_id_catalog_reads_nonzero_laser_retro(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        with tempfile.TemporaryDirectory(prefix="object_id_catalog_") as temp_dir:
            world_path = pathlib.Path(temp_dir) / "catalog.world"
            world_path.write_text(
                """
<sdf version='1.7'><world name='w'>
  <model name='moving_box'><link name='l'><collision name='c'>
    <laser_retro>31</laser_retro><geometry><box><size>1 1 1</size></box></geometry>
  </collision></link></model>
  <model name='static_wall'><link name='l'><collision name='c'>
    <laser_retro>32</laser_retro><geometry><box><size>1 1 1</size></box></geometry>
  </collision></link></model>
</world></sdf>
""".strip()
            )

            self.assertEqual(
                module.load_object_id_catalog(world_path),
                {31: "moving_box", 32: "static_wall"},
            )

    def test_alarm_operating_point_is_loaded_from_recorded_config(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        with tempfile.TemporaryDirectory(prefix="alarm_operating_point_") as temp_dir:
            run_dir = pathlib.Path(temp_dir) / "sim_run_000"
            meta_dir = run_dir / "meta"
            meta_dir.mkdir(parents=True)
            with (meta_dir / "config_snapshot.json").open("w") as handle:
                json.dump({
                    "parameters": {
                        "deform_monitor": {
                            "significance": {"cusum_h": 10.0},
                            "directional_motion": {"tau_s": 0.05, "tau_c": 0.65},
                            "persistent_risk": {
                                "min_confirmed_mean_risk": 0.65,
                                "min_hits_to_confirm": 5,
                                "min_hit_streak_to_confirm": 5,
                            },
                        }
                    }
                }, handle)

            operating_point = module.load_alarm_operating_point(run_dir)

            self.assertEqual(operating_point["final_mean_risk_threshold"], 0.65)
            self.assertEqual(operating_point["min_hits_to_confirm"], 5)
            self.assertEqual(operating_point["cusum_h"], 10.0)
            self.assertEqual(operating_point["directional_tau_s"], 0.05)

    def test_object_metrics_reject_available_anchor_data_without_catalog(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        moving = module.TruthTrack(
            object_name="moving_box",
            time_sec=[0.0, 1.0],
            x=[0.0, 0.02], y=[0.0, 0.0], z=[0.0, 0.0],
        )
        records = [{
            "header": {"stamp": {"sec": 1.0}},
            "anchors": [{
                "id": 1, "anchor_type": 0, "object_id": 0,
                "object_id_valid": False, "significant": False,
            }],
        }]

        with self.assertRaisesRegex(ValueError, "object ID catalog is empty"):
            module.validate_object_association_inputs(records, [moving], {})

    def test_persistent_summary_counts_false_confirmed_tracks_in_zero_deformation(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        static = module.TruthTrack(
            object_name="static_wall",
            time_sec=[0.0, 1.0],
            x=[0.0, 0.0],
            y=[0.0, 0.0],
            z=[0.0, 0.0],
        )
        persistent_records = [
            {
                "header": {"stamp": {"sec": 0.0}},
                "regions": [persistent_region_entry(7, (1.0, 0.0, 0.0), confirmed=True)],
            },
            {
                "header": {"stamp": {"sec": 1.0}},
                "regions": [persistent_region_entry(7, (1.0, 0.0, 0.0), confirmed=True)],
            },
        ]
        identity = {
            "rotation": module.np.eye(3),
            "translation": module.np.zeros(3),
        }

        summary = module.build_persistent_risk_summary(
            persistent_records,
            [],
            [static],
            {},
            identity,
        )

        self.assertEqual(summary["false_confirmed_track_count"], 1)
        self.assertEqual(summary["true_confirmed_track_count"], 0)
        self.assertEqual(summary["false_confirmed_region_observations"], 2)
        self.assertEqual(summary["false_confirmed_tracks_per_min"], 60.0)
        self.assertEqual(summary["false_alarm_time_fraction"], 1.0)

    def test_persistent_track_ids_are_scoped_by_reference_epoch(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        persistent_records = [
            {
                "header": {"stamp": {"sec": 0.0}},
                "reference_epoch": 1,
                "regions": [persistent_region_entry(0, (1.0, 0.0, 0.0), confirmed=True)],
            },
            {
                "header": {"stamp": {"sec": 1.0}},
                "reference_epoch": 2,
                "regions": [persistent_region_entry(0, (2.0, 0.0, 0.0), confirmed=True)],
            },
        ]
        identity = {
            "rotation": module.np.eye(3),
            "translation": module.np.zeros(3),
        }

        summary = module.build_persistent_risk_summary(
            persistent_records, [], [], {}, identity
        )

        self.assertEqual(summary["confirmed_track_count"], 2)
        self.assertEqual(summary["false_confirmed_track_count"], 2)

    def test_persistent_confirmation_before_truth_motion_is_false(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        moving = module.TruthTrack(
            object_name="moving_panel",
            time_sec=[0.0, 1.0, 2.0],
            x=[0.0, 0.0, 0.02],
            y=[0.0, 0.0, 0.0],
            z=[0.0, 0.0, 0.0],
        )
        persistent_records = [{
            "header": {"stamp": {"sec": 1.0}},
            "reference_epoch": 1,
            "regions": [persistent_region_entry(3, (0.0, 0.0, 0.0), confirmed=True)],
        }]
        identity = {"rotation": module.np.eye(3), "translation": module.np.zeros(3)}

        summary = module.build_persistent_risk_summary(
            persistent_records, [], [moving], {}, identity
        )

        self.assertEqual(summary["true_confirmed_track_count"], 0)
        self.assertEqual(summary["false_confirmed_track_count"], 1)
        self.assertEqual(summary["detected_moving_object_count"], 0)

    def test_persistent_surface_region_matches_large_truth_bbox(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        moving = module.TruthTrack(
            object_name="moving_panel",
            time_sec=[0.0, 1.0],
            x=[0.0, 0.02],
            y=[0.0, 0.0],
            z=[0.0, 0.0],
            qx=[0.0, 0.0], qy=[0.0, 0.0], qz=[0.0, 0.0], qw=[1.0, 1.0],
        )
        persistent_records = [{
            "header": {"stamp": {"sec": 1.0}},
            "reference_epoch": 1,
            "regions": [persistent_region_entry(4, (0.02, 1.5, 0.0), confirmed=True)],
        }]
        identity = {"rotation": module.np.eye(3), "translation": module.np.zeros(3)}
        box_specs = {
            "moving_panel": module.TruthBoxSpec("moving_panel", 0.2, 4.0, 2.0)
        }

        summary = module.build_persistent_risk_summary(
            persistent_records, [], [moving], {}, identity,
            truth_box_specs=box_specs,
        )

        self.assertEqual(summary["true_confirmed_track_count"], 1)
        self.assertEqual(summary["detected_moving_object_count"], 1)

    def test_persistent_summary_reports_preliminary_and_confirmed_object_response(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        moving = module.TruthTrack(
            object_name="moving_block",
            time_sec=[0.0, 1.0, 2.0],
            x=[0.0, 0.02, 0.04],
            y=[0.0, 0.0, 0.0],
            z=[0.0, 0.0, 0.0],
            qx=[0.0, 0.0, 0.0], qy=[0.0, 0.0, 0.0],
            qz=[0.0, 0.0, 0.0], qw=[1.0, 1.0, 1.0],
        )
        records = [
            {
                "header": {"stamp": {"sec": 1.0}},
                "reference_epoch": 1,
                "regions": [persistent_region_entry(
                    8, (0.02, 0.0, 0.0), confirmed=False, state=0
                )],
            },
            {
                "header": {"stamp": {"sec": 2.0}},
                "reference_epoch": 1,
                "regions": [persistent_region_entry(
                    8, (0.04, 0.0, 0.0), confirmed=True, state=1
                )],
            },
        ]
        identity = {"rotation": module.np.eye(3), "translation": module.np.zeros(3)}

        summary = module.build_persistent_risk_summary(
            records, [], [moving], {}, identity
        )

        self.assertEqual(summary["candidate_track_count"], 1)
        self.assertEqual(summary["true_candidate_track_count"], 1)
        self.assertAlmostEqual(summary["preliminary_alert_precision"], 1.0)
        self.assertEqual(summary["preliminary_detected_moving_object_count"], 1)
        self.assertAlmostEqual(summary["preliminary_alert_recall"], 1.0)
        self.assertAlmostEqual(summary["preliminary_alert_f1"], 1.0)
        per_object = summary["per_moving_object"][0]
        self.assertEqual(per_object["object_name"], "moving_block")
        self.assertTrue(per_object["preliminary_detected"])
        self.assertTrue(per_object["confirmed_detected"])
        self.assertAlmostEqual(per_object["first_candidate_time"], 1.0)
        self.assertAlmostEqual(per_object["first_confirmed_time"], 2.0)
        self.assertAlmostEqual(per_object["candidate_delay_sec"], 0.0)
        self.assertAlmostEqual(per_object["confirmation_delay_sec"], 1.0)
        self.assertAlmostEqual(per_object["candidate_to_confirmation_sec"], 1.0)
        self.assertAlmostEqual(
            per_object["gt_root_translation_at_confirmation_m"], 0.04
        )
        self.assertAlmostEqual(
            per_object["gt_root_rotation_at_confirmation_deg"], 0.0
        )

    def test_persistent_summary_prefers_direct_object_id_over_spatial_overlap(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        first = module.TruthTrack(
            object_name="first_block",
            time_sec=[0.0, 1.0],
            x=[0.0, 0.02],
            y=[0.0, 0.0],
            z=[0.0, 0.0],
        )
        second = module.TruthTrack(
            object_name="second_block",
            time_sec=[0.0, 1.0],
            x=[0.0, 0.02],
            y=[0.0, 0.0],
            z=[0.0, 0.0],
        )
        region = persistent_region_entry(
            5, (0.02, 0.0, 0.0), confirmed=True, state=1
        )
        region.update(
            {
                "object_id": 32,
                "object_id_valid": True,
                "object_id_confidence": 0.97,
                "object_id_ambiguous": False,
            }
        )
        records = [
            {
                "header": {"stamp": {"sec": 1.0}},
                "reference_epoch": 1,
                "regions": [region],
            }
        ]
        identity = {"rotation": module.np.eye(3), "translation": module.np.zeros(3)}

        summary = module.build_persistent_risk_summary(
            records,
            [],
            [first, second],
            {},
            identity,
            object_id_catalog={31: "first_block", 32: "second_block"},
        )

        by_name = {row["object_name"]: row for row in summary["per_moving_object"]}
        self.assertFalse(by_name["first_block"]["confirmed_detected"])
        self.assertTrue(by_name["second_block"]["confirmed_detected"])
        self.assertEqual(summary["direct_object_association_observations"], 1)
        self.assertEqual(summary["spatial_fallback_observations"], 0)
        self.assertAlmostEqual(summary["object_association_coverage"], 1.0)

    def test_persistent_direct_ids_are_evaluable_without_frame_alignment(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        moving = module.TruthTrack(
            object_name="moving_block",
            time_sec=[0.0, 1.0, 61.0],
            x=[0.0, 0.02, 0.04],
            y=[0.0, 0.0, 0.0],
            z=[0.0, 0.0, 0.0],
        )
        true_region = persistent_region_entry(
            5, (100.0, 100.0, 100.0), confirmed=True, state=1
        )
        true_region.update({
            "object_id": 31,
            "object_id_valid": True,
            "object_id_ambiguous": False,
        })
        false_region = persistent_region_entry(
            6, (-100.0, -100.0, -100.0), confirmed=True, state=1
        )
        false_region.update({
            "object_id": 32,
            "object_id_valid": True,
            "object_id_ambiguous": False,
        })
        records = [
            {
                "header": {"stamp": {"sec": 1.0}},
                "reference_epoch": 1,
                "regions": [true_region],
            },
            {
                "header": {"stamp": {"sec": 31.0}},
                "reference_epoch": 1,
                "regions": [false_region],
            },
            {
                "header": {"stamp": {"sec": 61.0}},
                "reference_epoch": 1,
                "regions": [],
            },
        ]

        summary = module.build_persistent_risk_summary(
            records,
            [],
            [moving],
            {},
            alignment=None,
            object_id_catalog={31: "moving_block", 32: "static_wall"},
        )

        self.assertEqual(summary["truth_matching_status"], "available")
        self.assertAlmostEqual(summary["evaluation_duration_sec"], 60.0)
        self.assertEqual(summary["true_confirmed_track_count"], 1)
        self.assertEqual(summary["false_confirmed_track_count"], 1)
        self.assertAlmostEqual(summary["final_alert_precision"], 0.5)
        self.assertAlmostEqual(summary["final_alert_recall"], 1.0)
        self.assertAlmostEqual(summary["final_alert_f1"], 2.0 / 3.0)
        self.assertAlmostEqual(summary["false_confirmed_tracks_per_min"], 1.0)
        self.assertAlmostEqual(summary["false_alarm_time_fraction"], 0.5)
        self.assertAlmostEqual(summary["false_alarm_frame_fraction"], 1.0 / 3.0)

    def test_persistent_summary_reports_geometry_and_identity_independently(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        first = module.TruthTrack(
            object_name="first_block", time_sec=[0.0, 1.0],
            x=[0.0, 0.02], y=[0.0, 0.0], z=[0.0, 0.0],
        )
        second = module.TruthTrack(
            object_name="second_block", time_sec=[0.0, 1.0],
            x=[3.0, 3.02], y=[0.0, 0.0], z=[0.0, 0.0],
        )
        correct_unattributed = persistent_region_entry(
            1, (0.02, 0.0, 0.0), confirmed=True
        )
        correct_unattributed.update({"object_id_valid": False,
                                     "object_id_ambiguous": False})
        wrong_identity = persistent_region_entry(
            2, (0.02, 0.0, 0.0), confirmed=True
        )
        wrong_identity.update({
            "object_id": 32, "object_id_valid": True,
            "object_id_ambiguous": False, "object_association_state": 2,
        })
        false_alert = persistent_region_entry(
            3, (20.0, 0.0, 0.0), confirmed=True
        )
        records = [{
            "header": {"stamp": {"sec": 1.0}}, "reference_epoch": 1,
            "regions": [correct_unattributed, wrong_identity, false_alert],
        }]
        identity = {"rotation": module.np.eye(3),
                    "translation": module.np.zeros(3)}

        summary = module.build_persistent_risk_summary(
            records, [], [first, second], {}, identity,
            object_id_catalog={31: "first_block", 32: "second_block"},
        )

        self.assertEqual(summary["geometrically_correct_unattributed_confirmed_observations"], 1)
        self.assertEqual(summary["association_error_confirmed_observations"], 1)
        self.assertEqual(summary["true_false_confirmed_observations"], 1)
        self.assertEqual(summary["geometric_true_confirmed_track_count"], 2)
        self.assertEqual(summary["geometric_false_confirmed_track_count"], 1)
        self.assertAlmostEqual(summary["final_geometric_precision"], 2.0 / 3.0)
        self.assertAlmostEqual(summary["final_geometric_recall"], 0.5)
        self.assertEqual(summary["identity_true_confirmed_track_count"], 0)
        self.assertEqual(summary["confirmed_fragmentation_count"], 1)
        self.assertEqual(summary["confirmed_cross_object_merge_count"], 0)

    def test_persistent_recall_excludes_lidar_unobserved_moving_objects(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        first = module.TruthTrack(
            object_name="first_block", time_sec=[0.0, 1.0],
            x=[0.0, 0.02], y=[0.0, 0.0], z=[0.0, 0.0],
        )
        second = module.TruthTrack(
            object_name="second_block", time_sec=[0.0, 1.0],
            x=[3.0, 3.02], y=[0.0, 0.0], z=[0.0, 0.0],
        )
        records = [{
            "header": {"stamp": {"sec": 1.0}}, "reference_epoch": 1,
            "regions": [persistent_region_entry(1, (0.02, 0.0, 0.0), True)],
        }]
        exposure = {
            31: {"lidar_observed": True, "hit_point_count": 100,
                 "visible_frame_count": 5, "exposure_frame_count": 5,
                 "_windows": [{"start_time": 0.0, "end_time": 1.0,
                               "frame_count": 5, "point_count": 100,
                               "visible_frame_count": 5}]},
            32: {"lidar_observed": False, "hit_point_count": 0,
                 "visible_frame_count": 0, "exposure_frame_count": 5,
                 "_windows": [{"start_time": 0.0, "end_time": 1.0,
                               "frame_count": 5, "point_count": 0,
                               "visible_frame_count": 0}]},
        }
        identity = {"rotation": module.np.eye(3),
                    "translation": module.np.zeros(3)}

        summary = module.build_persistent_risk_summary(
            records, [], [first, second], {}, identity,
            object_id_catalog={31: "first_block", 32: "second_block"},
            object_exposure=exposure,
        )

        self.assertEqual(summary["moving_object_count"], 2)
        self.assertEqual(summary["evaluable_moving_object_count"], 1)
        self.assertAlmostEqual(summary["final_geometric_recall"], 1.0)
        by_name = {row["object_name"]: row for row in summary["per_moving_object"]}
        self.assertEqual(by_name["first_block"]["evaluation_status"], "EVALUABLE")
        self.assertEqual(by_name["second_block"]["evaluation_status"],
                         "LIDAR_UNOBSERVED")

    def test_forward_and_inverse_rigid_transforms_round_trip_points(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        transform = {
            "source_frame": "camera_init",
            "target_frame": "world",
            "rotation": module.np.array(
                [
                    [0.0, -1.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            ),
            "translation": module.np.array([1.0, 2.0, 0.0]),
        }

        point_camera_init = {"x": 1.0, "y": 0.0, "z": 0.0}
        point_world = module.transform_point_with_transform(point_camera_init, transform)
        inverse_transform = module.invert_rigid_transform(transform)
        recovered = module.transform_point_with_transform(point_world, inverse_transform)

        self.assertEqual(point_world, {"x": 1.0, "y": 3.0, "z": 0.0})
        self.assertAlmostEqual(recovered["x"], point_camera_init["x"])
        self.assertAlmostEqual(recovered["y"], point_camera_init["y"])
        self.assertAlmostEqual(recovered["z"], point_camera_init["z"])

    def test_truth_pose_interpolation_is_linear_and_uses_quaternion_slerp(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        track = module.TruthTrack(
            object_name="rotating_panel",
            time_sec=[0.0, 2.0],
            x=[0.0, 2.0],
            y=[0.0, 0.0],
            z=[0.0, 0.0],
            qx=[0.0, 0.0],
            qy=[0.0, 0.0],
            qz=[0.0, 1.0],
            qw=[1.0, 0.0],
        )

        position = module.track_position_at_time(track, 1.0)
        orientation = module.track_orientation_at_time(track, 1.0)
        rotation = module.quaternion_to_rotation_matrix(orientation)
        rotated_x = rotation.dot(module.np.array([1.0, 0.0, 0.0]))

        self.assertAlmostEqual(position["x"], 1.0)
        self.assertAlmostEqual(position["y"], 0.0)
        self.assertAlmostEqual(rotated_x[0], 0.0, places=7)
        self.assertAlmostEqual(rotated_x[1], 1.0, places=7)

    def test_anchor_vector_metrics_validate_opposite_displacements_during_rotation(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        half_sqrt_two = 2.0 ** -0.5
        rotating = module.TruthTrack(
            object_name="rotating_panel",
            time_sec=[0.0, 1.0],
            x=[0.0, 0.0],
            y=[0.0, 0.0],
            z=[0.0, 0.0],
            qx=[0.0, 0.0],
            qy=[0.0, 0.0],
            qz=[0.0, half_sqrt_two],
            qw=[1.0, half_sqrt_two],
        )
        surface_link = module.LinkTrack(
            scoped_link_name="rotating_panel::ground_truth_face_0",
            model_name="rotating_panel",
            link_name="ground_truth_face_0",
            time_sec=[0.0, 1.0],
            x=[1.0, 0.0],
            y=[0.0, 1.0],
            z=[0.0, 0.0],
        )
        records = [{
            "header": {"stamp": {"sec": 1.0}},
            "reference_epoch": 3,
            "anchors": [
                {
                    "id": 10,
                    "anchor_type": 0,
                    "object_id": 31,
                    "object_id_valid": True,
                    "reference_epoch": 3,
                    "reference_stamp": {"sec": 0.0},
                    "ref_center": {"x": 1.0, "y": 0.0, "z": 0.0},
                    "disp_mean": [-1.0, 1.0, 0.0],
                    "significant": True,
                },
                {
                    "id": 11,
                    "anchor_type": 0,
                    "object_id": 31,
                    "object_id_valid": True,
                    "reference_epoch": 3,
                    "reference_stamp": {"sec": 0.0},
                    "ref_center": {"x": -1.0, "y": 0.0, "z": 0.0},
                    "disp_mean": [1.0, -1.0, 0.0],
                    "significant": True,
                },
                {
                    "id": 12,
                    "anchor_type": 1,
                    "object_id": 31,
                    "object_id_valid": True,
                    "reference_epoch": 3,
                    "reference_stamp": {"sec": 0.0},
                    "ref_center": {"x": 0.0, "y": 1.0, "z": 0.0},
                    "disp_mean": [99.0, 99.0, 99.0],
                    "significant": False,
                },
            ],
        }]
        identity = {
            "rotation": module.np.eye(3),
            "translation": module.np.zeros(3),
        }

        detail_rows, type_rows = module.build_anchor_vector_metrics(
            records,
            [rotating],
            {"rotating_panel": [surface_link]},
            {31: "rotating_panel"},
            identity,
        )

        self.assertEqual(len(detail_rows), 2)
        by_anchor = {row["anchor_id"]: row for row in detail_rows}
        self.assertAlmostEqual(by_anchor[10]["expected_dx"], -1.0)
        self.assertAlmostEqual(by_anchor[10]["expected_dy"], 1.0)
        self.assertAlmostEqual(by_anchor[11]["expected_dx"], 1.0)
        self.assertAlmostEqual(by_anchor[11]["expected_dy"], -1.0)
        self.assertAlmostEqual(by_anchor[10]["vector_error_norm"], 0.0, places=7)
        self.assertAlmostEqual(by_anchor[11]["direction_error_deg"], 0.0, places=7)
        plane = next(row for row in type_rows if row["anchor_type"] == "PLANE")
        edge = next(row for row in type_rows if row["anchor_type"] == "EDGE")
        self.assertEqual(plane["observation_count"], 2)
        self.assertEqual(plane["anchor_count"], 2)
        self.assertEqual(edge["observation_count"], 0)

    def test_anchor_vector_metrics_report_micro_and_macro_errors(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        rows = []
        for anchor_id, errors in ((10, [1.0, 1.0, 1.0]), (11, [5.0])):
            for index, error in enumerate(errors):
                rows.append({
                    "object_id": 31, "anchor_type": "PLANE",
                    "reference_epoch": 1, "anchor_id": anchor_id,
                    "time_sec": float(index), "valid": True,
                    "vector_error_norm": error, "magnitude_error_abs": error,
                    "direction_error_deg": error * 10.0,
                    "velocity_vector_error_norm": error * 2.0,
                    "velocity_direction_error_deg": error * 5.0,
                })

        summary = module.summarize_anchor_vector_errors(rows, "PLANE")

        self.assertAlmostEqual(summary["vector_error_mean"], 2.0)
        self.assertAlmostEqual(summary["per_anchor_macro_vector_error_mean"], 3.0)
        self.assertAlmostEqual(summary["object_type_macro_vector_error_mean"], 2.0)
        self.assertAlmostEqual(summary["per_anchor_macro_velocity_vector_error_mean"], 6.0)

    def test_anchor_vector_metrics_exclude_identity_mismatch_samples(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        moving = module.TruthTrack(
            object_name="moving_box", time_sec=[0.0, 1.0],
            x=[0.0, 0.02], y=[0.0, 0.0], z=[0.0, 0.0],
        )
        records = [{
            "header": {"stamp": {"sec": 1.0}}, "reference_epoch": 1,
            "anchors": [{
                "id": 10, "anchor_type": 0, "object_id": 31,
                "object_id_valid": True, "object_association_state": 2,
                "reference_stamp": {"sec": 0.0},
                "ref_center": {"x": 0.0, "y": 0.0, "z": 0.0},
                "disp_mean": [0.02, 0.0, 0.0], "significant": True,
            }],
        }]
        identity = {"rotation": module.np.eye(3),
                    "translation": module.np.zeros(3)}

        detail, summary = module.build_anchor_vector_metrics(
            records, [moving], {}, {31: "moving_box"}, identity
        )

        self.assertEqual(detail[0]["invalid_reason"], "object_association_mismatch")
        self.assertFalse(detail[0]["valid"])
        self.assertEqual(next(row for row in summary
                              if row["anchor_type"] == "PLANE")["valid_observation_count"], 0)

    def test_surface_truth_points_form_object_matching_extent_without_shape_model(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        moving = module.TruthTrack(
            object_name="moving_panel",
            time_sec=[0.0, 1.0],
            x=[10.0, 10.02],
            y=[0.0, 0.0],
            z=[0.0, 0.0],
            qx=[0.0, 0.0], qy=[0.0, 0.0], qz=[0.0, 0.0], qw=[1.0, 1.0],
        )
        surface_links = []
        for index, (x, y, z) in enumerate([
            (8.0, 2.0, -2.0),
            (12.0, 2.0, -2.0),
            (8.0, 2.0, 2.0),
            (12.0, 2.0, 2.0),
        ]):
            surface_links.append(module.LinkTrack(
                scoped_link_name=f"moving_panel::ground_truth_face_{index}",
                model_name="moving_panel",
                link_name=f"ground_truth_face_{index}",
                time_sec=[0.0, 1.0],
                x=[x, x + 0.02],
                y=[y, y],
                z=[z, z],
            ))
        persistent_records = [{
            "header": {"stamp": {"sec": 1.0}},
            "reference_epoch": 1,
            "regions": [persistent_region_entry(4, (10.02, 2.0, 0.0), confirmed=True)],
        }]
        identity = {"rotation": module.np.eye(3), "translation": module.np.zeros(3)}

        summary = module.build_persistent_risk_summary(
            persistent_records,
            [],
            [moving],
            {"moving_panel": surface_links},
            identity,
            truth_box_specs={},
        )

        self.assertEqual(summary["true_confirmed_track_count"], 1)
        self.assertEqual(summary["detected_moving_object_count"], 1)

    def test_load_alignment_prefers_explicit_world_from_algorithm_transform(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        with tempfile.TemporaryDirectory(prefix="sim_run_alignment_explicit_") as temp_dir:
            run_dir = pathlib.Path(temp_dir) / "sim_run_000"
            meta_dir = run_dir / "meta"
            meta_dir.mkdir(parents=True)
            with (meta_dir / "frame_alignment.json").open("w") as handle:
                json.dump(
                    {
                        "truth_frame": "world",
                        "algorithm_frame": "camera_init",
                        "alignment_mode": "initial_ego_pose",
                        "sim_only": True,
                        "ego_initial_pose_world": {
                            "position": {"x": 99.0, "y": 88.0, "z": 77.0},
                            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                        },
                        "world_from_algorithm_transform": {
                            "source_frame": "camera_init",
                            "target_frame": "world",
                            "pose": {
                                "position": {"x": 1.0, "y": 2.0, "z": 3.0},
                                "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                            },
                        },
                    },
                    handle,
                )

            alignment = module.load_alignment(run_dir)

            self.assertEqual(alignment["source_frame"], "camera_init")
            self.assertEqual(alignment["target_frame"], "world")
            self.assertEqual(alignment["translation"].tolist(), [1.0, 2.0, 3.0])

    def test_analyze_sim_run_matches_region_by_truth_bbox_when_point_distance_would_fail(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        with tempfile.TemporaryDirectory(prefix="sim_run_analysis_bbox_region_") as temp_dir:
            temp_dir = pathlib.Path(temp_dir)
            run_dir = temp_dir / "sim_run_000"
            truth_dir = run_dir / "truth" / "objects"
            meta_dir = run_dir / "meta"
            algorithm_dir = run_dir / "algorithm"
            world_path = temp_dir / "test.world"

            write_identity_alignment(meta_dir)
            write_moving_truth_object(
                truth_dir,
                object_name="moving_box",
                points=[
                    (0.0, 0.0, 0.0, 0.0),
                    (1.0, 0.03, 0.0, 0.0),
                ],
            )
            write_world_file(world_path, "moving_box", (2.0, 0.6, 0.6))
            algorithm_dir.mkdir(parents=True)
            write_jsonl_records(
                algorithm_dir / "risk_regions.jsonl",
                [
                    {
                        "header": {"seq": 1, "frame_id": "camera_init", "stamp": {"secs": 1, "nsecs": 0}},
                        "recorded_at": {"secs": 1, "nsecs": 0, "sec": 1.0},
                        "regions": [
                            {
                                "id": 1,
                                "region_type": 1,
                                "center": {"x": 0.90, "y": 0.0, "z": 0.0},
                                "bbox_min": {"x": 0.82, "y": -0.05, "z": -0.05},
                                "bbox_max": {"x": 0.98, "y": 0.05, "z": 0.05},
                                "mean_risk": 0.6,
                                "peak_risk": 0.8,
                                "confidence": 0.9,
                                "voxel_count": 10,
                                "significant": True,
                            }
                        ],
                    }
                ],
            )

            outputs = module.analyze_sim_run(run_dir, world_file=world_path)
            with outputs.summary_csv.open() as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual(rows[0]["object_name"], "moving_box")
            self.assertEqual(rows[0]["region_status"], "matched")

    def test_analyze_sim_run_matches_motion_by_truth_bbox_when_new_center_is_inside(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        with tempfile.TemporaryDirectory(prefix="sim_run_analysis_bbox_motion_") as temp_dir:
            temp_dir = pathlib.Path(temp_dir)
            run_dir = temp_dir / "sim_run_000"
            truth_dir = run_dir / "truth" / "objects"
            meta_dir = run_dir / "meta"
            algorithm_dir = run_dir / "algorithm"
            world_path = temp_dir / "test.world"

            write_identity_alignment(meta_dir)
            write_moving_truth_object(
                truth_dir,
                object_name="moving_box",
                points=[
                    (0.0, 0.0, 0.0, 0.0),
                    (1.0, 0.03, 0.0, 0.0),
                ],
            )
            write_world_file(world_path, "moving_box", (2.0, 0.6, 0.6))
            algorithm_dir.mkdir(parents=True)
            write_jsonl_records(
                algorithm_dir / "structure_motions.jsonl",
                [
                    {
                        "header": {"seq": 1, "frame_id": "camera_init", "stamp": {"secs": 1, "nsecs": 0}},
                        "recorded_at": {"secs": 1, "nsecs": 0, "sec": 1.0},
                        "motions": [
                            {
                                "id": 1,
                                "old_region_id": 0,
                                "new_region_id": 1,
                                "motion_type": 1,
                                "old_center": {"x": 0.70, "y": 0.0, "z": 0.0},
                                "new_center": {"x": 0.90, "y": 0.0, "z": 0.0},
                                "bbox_old_min": {"x": 0.65, "y": -0.05, "z": -0.05},
                                "bbox_old_max": {"x": 0.75, "y": 0.05, "z": 0.05},
                                "bbox_new_min": {"x": 0.82, "y": -0.05, "z": -0.05},
                                "bbox_new_max": {"x": 0.98, "y": 0.05, "z": 0.05},
                                "motion": {"x": 0.20, "y": 0.0, "z": 0.0},
                                "distance": 0.20,
                                "match_cost": 0.10,
                                "confidence": 0.90,
                                "support_old": 6,
                                "support_new": 6,
                                "significant": True,
                            }
                        ],
                    }
                ],
            )

            outputs = module.analyze_sim_run(run_dir, world_file=world_path)
            with outputs.summary_csv.open() as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual(rows[0]["object_name"], "moving_box")
            self.assertEqual(rows[0]["motion_status"], "matched")

    def test_classify_truth_track_labels_static_moving_and_outlier(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        static_track = module.TruthTrack(
            object_name="wall",
            time_sec=[0.0, 1.0],
            x=[0.0, 0.0],
            y=[0.0, 0.0],
            z=[0.0, 0.0],
        )
        moving_track = module.TruthTrack(
            object_name="block",
            time_sec=[0.0, 1.0],
            x=[0.0, 0.03],
            y=[0.0, 0.0],
            z=[0.0, 0.0],
        )
        outlier_track = module.TruthTrack(
            object_name="exploded_panel",
            time_sec=[0.0, 1.0],
            x=[0.0, 10000.0],
            y=[0.0, 0.0],
            z=[0.0, 0.0],
        )

        self.assertEqual(module.classify_truth_track(static_track), "static")
        self.assertEqual(module.classify_truth_track(moving_track), "moving")
        self.assertEqual(module.classify_truth_track(outlier_track), "outlier")

    def test_truth_motion_onset_is_first_physical_change_not_ten_mm_crossing(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        track = module.TruthTrack(
            object_name="slow_block",
            time_sec=[0.0, 1.0, 2.0, 3.0],
            x=[0.0, 0.002, 0.004, 0.012],
            y=[0.0, 0.0, 0.0, 0.0],
            z=[0.0, 0.0, 0.0, 0.0],
        )

        self.assertEqual(module.classify_truth_track(track), "moving")
        metrics = module.truth_track_metrics(track)
        self.assertEqual(metrics["start_time"], 1.0)

    def test_layer_status_marks_missing_empty_and_available(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        self.assertEqual(module.layer_status(None), "missing")
        self.assertEqual(module.layer_status([]), "empty")
        self.assertEqual(module.layer_status([{"dummy": 1}]), "available")

    def test_compact_risk_evidence_preserves_optional_epoch_semantics(self):
        module = load_module_if_exists()
        records = [
            {
                "header": {"stamp": {"sec": 1.0}},
                "evidences": [{"active": True, "risk_score": 2.0}],
            },
            {
                "header": {"stamp": {"sec": 2.0}},
                "reference_epoch": 4,
                "evidences": [{"active": False, "risk_score": 3.0}],
            },
        ]

        compact = module.compact_active_evidence_records(records)

        self.assertNotIn("reference_epoch", compact[0])
        self.assertEqual(compact[1]["reference_epoch"], 4)
        self.assertEqual(compact[1]["evidences"], [])

    def test_bundle_truth_metrics_uses_earliest_motion_onset_across_links(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        root_track = module.TruthTrack(
            object_name="armature",
            time_sec=[0.0, 1.0, 2.0],
            x=[0.0, 0.0, 0.0],
            y=[0.0, 0.0, 0.0],
            z=[0.0, 0.0, 0.0],
        )
        early_link = module.LinkTrack(
            scoped_link_name="armature::link_early",
            model_name="armature",
            link_name="link_early",
            time_sec=[0.0, 1.0, 2.0],
            x=[0.0, 0.02, 0.02],
            y=[0.0, 0.0, 0.0],
            z=[0.0, 0.0, 0.0],
        )
        late_link = module.LinkTrack(
            scoped_link_name="armature::link_late",
            model_name="armature",
            link_name="link_late",
            time_sec=[5.0, 6.0],
            x=[0.0, 0.08],
            y=[0.0, 0.0],
            z=[0.0, 0.0],
        )

        metrics = module.bundle_truth_metrics(root_track, [early_link, late_link])

        self.assertEqual(metrics["start_time"], 1.0)
        self.assertGreater(metrics["peak_displacement"], 0.02)
        self.assertGreater(metrics["net_displacement"], 0.0)

        alignment = {
            "rotation": module.np.eye(3),
            "translation": module.np.array([0.0, 0.0, 0.0]),
        }
        evidence_records = [
            {
                "header": {"seq": 1, "frame_id": "camera_init", "stamp": {"secs": 2, "nsecs": 0}},
                "recorded_at": {"secs": 2, "nsecs": 0, "sec": 2.0},
                "evidences": [
                    {
                        "active": True,
                        "position": {"x": 0.02, "y": 0.0, "z": 0.0},
                        "risk_score": 0.7,
                    }
                ],
            }
        ]

        summary = module.evaluate_truth_object(
            root_track,
            [early_link, late_link],
            alignment=alignment,
            evidence_records=evidence_records,
            region_records=[],
            motion_records=[],
        )

        self.assertEqual(summary["classification"], "moving")
        self.assertEqual(summary["gt_start_time"], 1.0)
        self.assertEqual(summary["evidence_delay_sec"], 1.0)

    def test_bundle_truth_metrics_keeps_root_twist_separate_from_surface_speed(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        root_track = module.TruthTrack(
            object_name="panel",
            time_sec=[0.0, 1.0, 2.0],
            x=[0.0, 0.002, 0.004],
            y=[0.0, 0.0, 0.0],
            z=[0.0, 0.0, 0.0],
            vx=[0.002, 0.002, 0.002],
            vy=[0.0, 0.0, 0.0],
            vz=[0.0, 0.0, 0.0],
        )
        surface_track = module.LinkTrack(
            scoped_link_name="panel::surface",
            model_name="panel",
            link_name="surface",
            time_sec=[0.0, 0.5, 2.0],
            x=[0.0, 0.05, 0.20],
            y=[0.0, 0.0, 0.0],
            z=[0.0, 0.0, 0.0],
        )

        metrics = module.bundle_truth_metrics(root_track, [surface_track])

        self.assertAlmostEqual(metrics["peak_linear_speed_mps"], 0.002)
        self.assertGreater(metrics["surface_peak_linear_speed_mps"], 0.05)

    def test_analyze_sim_run_creates_outputs_with_missing_layers(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        with tempfile.TemporaryDirectory(prefix="sim_run_analysis_") as temp_dir:
            run_dir = pathlib.Path(temp_dir) / "sim_run_000"
            truth_dir = run_dir / "truth" / "objects"
            meta_dir = run_dir / "meta"
            truth_dir.mkdir(parents=True)
            meta_dir.mkdir(parents=True)

            with (meta_dir / "frame_alignment.json").open("w") as handle:
                handle.write(
                    '{"truth_frame":"world","algorithm_frame":"camera_init","alignment_mode":"initial_ego_pose","sim_only":true}'
                )

            csv_path = truth_dir / "moving_block.csv"
            with csv_path.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    [
                        "recorded_time_sec",
                        "model_name",
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
                writer.writerow([0.0, "moving_block", "world", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
                writer.writerow([1.0, "moving_block", "world", 0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])

            outputs = module.analyze_sim_run(run_dir)

            self.assertTrue(outputs.summary_csv.is_file())
            self.assertTrue(outputs.outlier_csv.is_file())
            self.assertTrue(outputs.report_md.is_file())
            self.assertTrue(outputs.object_anchor_metrics_csv.is_file())
            self.assertTrue(outputs.anchor_type_metrics_csv.is_file())
            self.assertTrue(outputs.anchor_vector_metrics_csv.is_file())
            self.assertTrue(outputs.anchor_vector_type_metrics_csv.is_file())
            self.assertTrue(outputs.persistent_object_metrics_csv.is_file())
            self.assertTrue(outputs.alert_metrics_json.is_file())
            alert_metrics = json.loads(outputs.alert_metrics_json.read_text())
            self.assertEqual(
                alert_metrics["metric_contract"]["anchor_types"],
                ["PLANE", "EDGE", "BAND"],
            )
            self.assertEqual(
                alert_metrics["metric_contract"]["truth_motion_epsilon_m"],
                module.GT_MOTION_EPSILON,
            )

    def test_analyze_sim_run_reports_missing_persistent_risk_layer(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        with tempfile.TemporaryDirectory(prefix="sim_run_analysis_missing_persistent_") as temp_dir:
            run_dir = pathlib.Path(temp_dir) / "sim_run_000"
            truth_dir = run_dir / "truth" / "objects"
            meta_dir = run_dir / "meta"
            truth_dir.mkdir(parents=True)
            write_identity_alignment(meta_dir)
            write_moving_truth_object(truth_dir)

            outputs = module.analyze_sim_run(run_dir)
            report = outputs.report_md.read_text()

            self.assertIn("## Persistent Risk", report)
            self.assertIn("persistent_risk_regions: `missing`", report)
            self.assertIn("layer_status: `missing`", report)

    def test_analyze_sim_run_reports_empty_persistent_risk_layer(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        with tempfile.TemporaryDirectory(prefix="sim_run_analysis_empty_persistent_") as temp_dir:
            run_dir = pathlib.Path(temp_dir) / "sim_run_000"
            truth_dir = run_dir / "truth" / "objects"
            meta_dir = run_dir / "meta"
            algorithm_dir = run_dir / "algorithm"
            truth_dir.mkdir(parents=True)
            write_identity_alignment(meta_dir)
            write_moving_truth_object(truth_dir)
            algorithm_dir.mkdir(parents=True)
            (algorithm_dir / "persistent_risk_regions.jsonl").write_text("")

            outputs = module.analyze_sim_run(run_dir)
            report = outputs.report_md.read_text()

            self.assertIn("persistent_risk_regions: `empty`", report)
            self.assertIn("layer_status: `empty`", report)

    def test_analyze_sim_run_reports_persistent_confirmed_stats(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        with tempfile.TemporaryDirectory(prefix="sim_run_analysis_persistent_") as temp_dir:
            run_dir = pathlib.Path(temp_dir) / "sim_run_000"
            truth_dir = run_dir / "truth" / "objects"
            meta_dir = run_dir / "meta"
            algorithm_dir = run_dir / "algorithm"
            truth_dir.mkdir(parents=True)
            write_identity_alignment(meta_dir)
            write_moving_truth_object(
                truth_dir,
                points=[
                    (-1.0, 0.0, 0.0, 0.0),
                    (0.0, 0.02, 0.0, 0.0),
                    (1.0, 0.04, 0.0, 0.0),
                    (2.0, 0.06, 0.0, 0.0),
                ],
            )
            algorithm_dir.mkdir(parents=True)

            persistent_records = [
                {
                    "header": {
                        "seq": 1,
                        "frame_id": "camera_init",
                        "stamp": {"secs": 0, "nsecs": 0},
                    },
                    "recorded_at": {"secs": 0, "nsecs": 0, "sec": 0.0},
                    "regions": [persistent_region_entry(7, (0.0, 0.0, 0.0), confirmed=True)],
                },
                {
                    "header": {
                        "seq": 2,
                        "frame_id": "camera_init",
                        "stamp": {"secs": 1, "nsecs": 0},
                    },
                    "recorded_at": {"secs": 1, "nsecs": 0, "sec": 1.0},
                    "regions": [persistent_region_entry(7, (0.02, 0.0, 0.0), confirmed=True)],
                },
                {
                    "header": {
                        "seq": 3,
                        "frame_id": "camera_init",
                        "stamp": {"secs": 2, "nsecs": 0},
                    },
                    "recorded_at": {"secs": 2, "nsecs": 0, "sec": 2.0},
                    "regions": [persistent_region_entry(7, (0.04, 0.0, 0.0), confirmed=True)],
                },
            ]
            region_records = [
                {
                    "header": {
                        "seq": 1,
                        "frame_id": "camera_init",
                        "stamp": {"secs": 0, "nsecs": 0},
                    },
                    "recorded_at": {"secs": 0, "nsecs": 0, "sec": 0.0},
                    "regions": [
                        {
                            "id": 1,
                            "region_type": 1,
                            "center": {"x": 0.0, "y": 0.0, "z": 0.0},
                            "bbox_min": {"x": -0.1, "y": -0.1, "z": -0.1},
                            "bbox_max": {"x": 0.1, "y": 0.1, "z": 0.1},
                            "mean_risk": 0.3,
                            "peak_risk": 0.6,
                            "confidence": 0.5,
                            "voxel_count": 12,
                            "significant": True,
                        }
                    ],
                },
                {
                    "header": {
                        "seq": 2,
                        "frame_id": "camera_init",
                        "stamp": {"secs": 1, "nsecs": 0},
                    },
                    "recorded_at": {"secs": 1, "nsecs": 0, "sec": 1.0},
                    "regions": [],
                },
                {
                    "header": {
                        "seq": 3,
                        "frame_id": "camera_init",
                        "stamp": {"secs": 2, "nsecs": 0},
                    },
                    "recorded_at": {"secs": 2, "nsecs": 0, "sec": 2.0},
                    "regions": [],
                },
            ]
            write_jsonl_records(algorithm_dir / "persistent_risk_regions.jsonl", persistent_records)
            write_jsonl_records(algorithm_dir / "risk_regions.jsonl", region_records)

            outputs = module.analyze_sim_run(run_dir)
            report = outputs.report_md.read_text()

            self.assertIn("## Persistent Risk", report)
            self.assertIn("persistent_risk_regions: `available`", report)
            self.assertIn("confirmed_track_count: `1`", report)
            self.assertIn("first_confirmed_time: `0.0`", report)
            self.assertIn("max_confirmed_duration_sec: `2.0`", report)
            self.assertIn("confirmed_coverage_hits: `3`", report)
            self.assertIn("final_alert_recall: `1.0`", report)
            self.assertIn("stability_judgment: `more_stable`", report)

    def test_analyze_sim_run_uses_link_truth_when_model_root_is_static(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        with tempfile.TemporaryDirectory(prefix="sim_run_analysis_links_") as temp_dir:
            run_dir = pathlib.Path(temp_dir) / "sim_run_000"
            truth_objects_dir = run_dir / "truth" / "objects"
            truth_links_dir = run_dir / "truth" / "links"
            meta_dir = run_dir / "meta"
            truth_objects_dir.mkdir(parents=True)
            truth_links_dir.mkdir(parents=True)
            meta_dir.mkdir(parents=True)

            with (meta_dir / "frame_alignment.json").open("w") as handle:
                handle.write(
                    '{"truth_frame":"world","algorithm_frame":"camera_init","alignment_mode":"initial_ego_pose","sim_only":true}'
                )

            object_csv = truth_objects_dir / "gui_object.csv"
            with object_csv.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    [
                        "recorded_time_sec",
                        "model_name",
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
                writer.writerow([0.0, "gui_object", "world", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
                writer.writerow([1.0, "gui_object", "world", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])

            link_csv = truth_links_dir / "gui_object__base_link.csv"
            with link_csv.open("w", newline="") as handle:
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
                writer.writerow([0.0, "gui_object::base_link", "gui_object", "base_link", "world", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
                writer.writerow([1.0, "gui_object::base_link", "gui_object", "base_link", "world", 0.03, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])

            outputs = module.analyze_sim_run(run_dir)

            with outputs.summary_csv.open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["object_name"], "gui_object")
            self.assertEqual(rows[0]["classification"], "moving")
            self.assertGreater(float(rows[0]["gt_net_displacement"]), 0.0)
            self.assertGreater(float(rows[0]["gt_duration_sec"]), 0.0)
            self.assertGreater(float(rows[0]["gt_peak_displacement_time"]), 0.0)


if __name__ == "__main__":
    unittest.main()
