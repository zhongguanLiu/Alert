# Author: zgliu@cumt.edu.cn
# Affiliation: China University of Mining and Technology
# Open-source release date: 2026-04-20
"""Unit tests for analyze_sim_run.py.

All numeric literals in this file are synthetic fixture values chosen to
exercise coordinate transforms, bbox matching, and JSON/CSV parsing. They are
not experimental measurements and are not used by the runtime system.
"""

import csv
import importlib.util
import json
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
            rows = list(csv.DictReader(outputs.summary_csv.open()))

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
            rows = list(csv.DictReader(outputs.summary_csv.open()))

            self.assertEqual(rows[0]["object_name"], "moving_box")
            self.assertEqual(rows[0]["motion_status"], "matched")

    def test_classify_truth_track_labels_static_moving_and_outlier(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        static_track = module.TruthTrack(
            object_name="wall",
            time_sec=[0.0, 1.0],
            x=[0.0, 0.001],
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

    def test_layer_status_marks_missing_empty_and_available(self):
        module = load_module_if_exists()
        if module is None:
            self.fail(f"Missing implementation script: {SCRIPT_PATH}")

        self.assertEqual(module.layer_status(None), "missing")
        self.assertEqual(module.layer_status([]), "empty")
        self.assertEqual(module.layer_status([{"dummy": 1}]), "available")

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
                    (0.0, 0.0, 0.0, 0.0),
                    (1.0, 0.02, 0.0, 0.0),
                    (2.0, 0.04, 0.0, 0.0),
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
