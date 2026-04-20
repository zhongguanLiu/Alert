"""Test batch simulation analysis scripts."""

from __future__ import annotations

import csv
import contextlib
import io
import json
import pathlib
import tempfile
import unittest
from unittest import mock

import numpy as np

import analysis_script.common as common
import analysis_script.compare_ablation as compare_ablation
import analysis_script.compute_mdd as compute_mdd
import analysis_script.compute_metrics as compute_metrics
import analysis_script.run_simulation_paper_analysis as run_simulation_paper_analysis


class TemporaryResultRoot:
    def __init__(self, base_path: pathlib.Path):
        self.base_path = base_path
        self.new_root = self.base_path / "result"
        self._patcher = mock.patch(
            "analysis_script.common.RESULT_ROOT", new=self.new_root
        )

    def __enter__(self) -> pathlib.Path:
        self.new_root.mkdir(parents=True, exist_ok=True)
        self._patcher.start()
        return self.new_root

    def __exit__(self, exc_type, exc, exc_tb) -> None:
        self._patcher.stop()


def _write_object_csv(
    csv_path: pathlib.Path,
    rows: list[tuple[float, float, float, float]],
):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
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
        for t, x, y, z in rows:
            writer.writerow(
                [f"{t:.6f}", csv_path.stem, "world", x, y, z, 0.0, 0.0, 0.0, 1.0]
            )


class CommonFallbackTests(unittest.TestCase):
    def test_get_controlled_objects_returns_both_discovered_controls(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = pathlib.Path(tmp_dir) / "20260405" / "sim_run_000"
            meta_dir = run_dir / "meta"
            meta_dir.mkdir(parents=True)
            (meta_dir / "scenario_manifest.json").write_text(
                json.dumps(
                    {
                        "scenario_id": "sim_main_block02_ypos_1p0mmps_r02",
                        "controls": [
                            {
                                "controlled_object": "model_01",
                                "velocity": {
                                    "linear_mps": {"x": 0.0, "y": 0.001, "z": 0.0}
                                },
                                "start_delay_sec": 0.5,
                                "duration_sec": 2.0,
                            },
                            {
                                "controlled_object": "model_02",
                                "velocity": {
                                    "linear_mps": {"x": 0.0, "y": 0.002, "z": 0.0}
                                },
                                "start_delay_sec": 1.5,
                                "duration_sec": 3.0,
                            },
                        ],
                    }
                )
            )

            controls = common.get_controlled_objects(run_dir)
            self.assertEqual(
                [control.get("controlled_object") for control in controls],
                ["model_01", "model_02"],
            )
            self.assertEqual(common.get_controlled_object_name(run_dir), "model_01")

    def test_result_dir_for_run_uses_external_runs_for_non_default_dated_roots(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = pathlib.Path(tmp_dir)
            run_dir = tmp_path / "output_ablation" / "20260405" / "sim_run_000"
            run_dir.mkdir(parents=True)
            with TemporaryResultRoot(tmp_path) as result_root:
                target = common.result_dir_for_run(run_dir)
                self.assertTrue(target.exists())
                self.assertEqual(target.parent.name, "external_runs")
                self.assertIn("sim_run_000", target.name)

    def test_get_injection_velocity_falls_back_to_scenario_id_when_manifest_velocity_is_zero(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = pathlib.Path(tmp_dir) / "20260405" / "sim_run_000"
            meta_dir = run_dir / "meta"
            meta_dir.mkdir(parents=True)
            (meta_dir / "scenario_manifest.json").write_text(
                json.dumps(
                    {
                        "scenario_id": "sim_main_block02_ypos_1p0mmps_r02",
                        "controls": [
                            {
                                "controlled_object": "mid360_fastlio",
                                "velocity": {
                                    "linear_mps": {"x": 0.0, "y": 0.0, "z": 0.0}
                                },
                            }
                        ],
                    }
                )
            )

            self.assertAlmostEqual(
                common.get_injection_velocity(run_dir),
                0.001,
                places=9,
            )

    def test_get_injection_velocity_resolves_per_control_object(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = pathlib.Path(tmp_dir) / "20260405" / "sim_run_000"
            meta_dir = run_dir / "meta"
            meta_dir.mkdir(parents=True)
            (meta_dir / "scenario_manifest.json").write_text(
                json.dumps(
                    {
                        "scenario_id": "sim_main_block02_ypos_1p0mmps_r02",
                        "controls": [
                            {
                                "controlled_object": "model_01",
                                "velocity": {
                                    "linear_mps": {"x": 0.0, "y": 0.001, "z": 0.0}
                                },
                            },
                            {
                                "controlled_object": "model_02",
                                "velocity": {
                                    "linear_mps": {"x": 0.0, "y": 0.002, "z": 0.0}
                                },
                            },
                        ],
                    }
                )
            )

            self.assertAlmostEqual(common.get_injection_velocity(run_dir, "model_01"), 0.001)
            self.assertAlmostEqual(common.get_injection_velocity(run_dir, "model_02"), 0.002)

    def test_get_injection_velocity_missing_control_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = pathlib.Path(tmp_dir) / "20260405" / "sim_run_000"
            meta_dir = run_dir / "meta"
            meta_dir.mkdir(parents=True)
            (meta_dir / "scenario_manifest.json").write_text(
                json.dumps(
                    {
                        "scenario_id": "not_parseable",
                        "controls": [
                            {
                                "controlled_object": "model_01",
                                "velocity": {
                                    "linear_mps": {"x": 0.0, "y": 0.0, "z": 0.0}
                                },
                            }
                        ],
                    }
                )
            )

            self.assertIsNone(common.get_injection_velocity(run_dir, "missing_object"))

    def test_get_injection_velocity_falls_back_within_selected_control_only(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = pathlib.Path(tmp_dir) / "20260405" / "sim_run_000"
            meta_dir = run_dir / "meta"
            meta_dir.mkdir(parents=True)
            (meta_dir / "scenario_manifest.json").write_text(
                json.dumps(
                    {
                        "scenario_id": "not_parseable",
                        "controls": [
                            {
                                "controlled_object": "model_01",
                                "scenario_id": "sim_main_block02_ypos_0p5mmps_r01",
                                "velocity": {
                                    "linear_mps": {"x": 0.0, "y": 0.0, "z": 0.0}
                                },
                            },
                            {
                                "controlled_object": "model_02",
                                "scenario_id": "sim_main_block02_ypos_1p0mmps_r02",
                                "velocity": {
                                    "linear_mps": {"x": 0.0, "y": 0.0, "z": 0.0}
                                },
                            },
                        ],
                    }
                )
            )

            self.assertAlmostEqual(common.get_injection_velocity(run_dir, "model_02"), 0.001)
            self.assertAlmostEqual(common.get_injection_velocity(run_dir, "model_01"), 0.0005)

    def test_get_analysis_controlled_object_name_falls_back_to_unique_moving_gt_object(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = pathlib.Path(tmp_dir) / "20260405" / "sim_run_000"
            meta_dir = run_dir / "meta"
            meta_dir.mkdir(parents=True)
            (meta_dir / "scenario_manifest.json").write_text(
                json.dumps(
                    {
                        "scenario_id": "sim_main_block02_ypos_1p0mmps_r02",
                        "controls": [
                            {
                                "controlled_object": "mid360_fastlio",
                                "velocity": {
                                    "linear_mps": {"x": 0.0, "y": 0.0, "z": 0.0}
                                },
                            }
                        ],
                    }
                )
            )

            _write_object_csv(
                run_dir / "truth" / "objects" / "obstacle_block_left_clone_clone.csv",
                [
                    (0.0, 0.0, 0.0, 0.0),
                    (1.0, 0.0, 0.02, 0.0),
                ],
            )
            _write_object_csv(
                run_dir / "truth" / "objects" / "static_wall.csv",
                [
                    (0.0, 1.0, 1.0, 1.0),
                    (1.0, 1.0, 1.0, 1.0),
                ],
            )

            self.assertEqual(
                common.get_analysis_controlled_object_name(run_dir),
                "obstacle_block_left_clone_clone",
            )

    def test_get_analysis_controlled_object_names_filters_to_truth_backed_controls(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = pathlib.Path(tmp_dir) / "20260405" / "sim_run_000"
            meta_dir = run_dir / "meta"
            meta_dir.mkdir(parents=True)
            (meta_dir / "scenario_manifest.json").write_text(
                json.dumps(
                    {
                        "scenario_id": "sim_main_block02_ypos_1p0mmps_r02",
                        "controls": [
                            {
                                "controlled_object": "model_01",
                                "velocity": {
                                    "linear_mps": {"x": 0.0, "y": 0.001, "z": 0.0}
                                },
                            },
                            {
                                "controlled_object": "ego_platform",
                                "velocity": {
                                    "linear_mps": {"x": 0.0, "y": 0.002, "z": 0.0}
                                },
                            },
                        ],
                    }
                )
            )

            _write_object_csv(
                run_dir / "truth" / "objects" / "model_01.csv",
                [
                    (0.0, 0.0, 0.0, 0.0),
                    (1.0, 0.0, 0.02, 0.0),
                ],
            )
            _write_object_csv(
                run_dir / "truth" / "objects" / "static_wall.csv",
                [
                    (0.0, 1.0, 1.0, 1.0),
                    (1.0, 1.0, 1.0, 1.0),
                ],
            )

            self.assertEqual(
                common.get_analysis_controlled_object_names(run_dir),
                ["model_01"],
            )
            self.assertEqual(
                common.get_analysis_controlled_object_name(run_dir),
                "model_01",
            )

    def test_get_analysis_controlled_object_name_prefers_unique_moving_truth_backed_control(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = pathlib.Path(tmp_dir) / "20260405" / "sim_run_000"
            meta_dir = run_dir / "meta"
            meta_dir.mkdir(parents=True)
            (meta_dir / "scenario_manifest.json").write_text(
                json.dumps(
                    {
                        "scenario_id": "sim_main_block02_ypos_1p0mmps_r02",
                        "controls": [
                            {
                                "controlled_object": "model_01",
                                "velocity": {
                                    "linear_mps": {"x": 0.0, "y": 0.001, "z": 0.0}
                                },
                            },
                            {
                                "controlled_object": "model_02",
                                "velocity": {
                                    "linear_mps": {"x": 0.0, "y": 0.002, "z": 0.0}
                                },
                            },
                        ],
                    }
                )
            )

            _write_object_csv(
                run_dir / "truth" / "objects" / "model_01.csv",
                [
                    (0.0, 0.0, 0.0, 0.0),
                    (1.0, 0.0, 0.0, 0.0),
                ],
            )
            _write_object_csv(
                run_dir / "truth" / "objects" / "model_02.csv",
                [
                    (0.0, 0.0, 0.0, 0.0),
                    (1.0, 0.0, 0.02, 0.0),
                ],
            )

            self.assertEqual(
                [control.get("controlled_object") for control in common.get_controlled_objects(run_dir)],
                ["model_01", "model_02"],
            )
            self.assertEqual(
                common.get_analysis_controlled_object_names(run_dir),
                ["model_01", "model_02"],
            )
            self.assertEqual(
                common.get_analysis_controlled_object_name(run_dir),
                "model_02",
            )

    def test_get_scenario_timing_returns_per_control_object(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = pathlib.Path(tmp_dir) / "20260405" / "sim_run_000"
            meta_dir = run_dir / "meta"
            meta_dir.mkdir(parents=True)
            (meta_dir / "scenario_manifest.json").write_text(
                json.dumps(
                    {
                        "scenario_id": "sim_main_block02_ypos_1p0mmps_r02",
                        "controls": [
                            {
                                "controlled_object": "model_01",
                                "start_delay_sec": 0.5,
                                "duration_sec": 2.0,
                            },
                            {
                                "controlled_object": "model_02",
                                "start_delay_sec": 1.5,
                                "duration_sec": 3.0,
                            },
                        ],
                    }
                )
            )

            self.assertEqual(common.get_scenario_timing(run_dir, "model_02"), (1.5, 3.0))


class SimulationBatchScriptTests(unittest.TestCase):
    def _create_multi_control_run(
        self,
        base_path: pathlib.Path,
        date_name: str,
        run_name: str,
        scenario_id: str = "sim_main_block02_ypos_1p0mmps_r01",
    ) -> pathlib.Path:
        run_dir = base_path / date_name / run_name
        meta_dir = run_dir / "meta"
        meta_dir.mkdir(parents=True)
        (meta_dir / "scenario_manifest.json").write_text(
            json.dumps(
                {
                    "scenario_id": scenario_id,
                    "controls": [
                        {
                            "controlled_object": "model_01",
                            "velocity": {
                                "linear_mps": {"x": 0.0, "y": 0.001, "z": 0.0}
                            },
                        },
                        {
                            "controlled_object": "model_02",
                            "velocity": {
                                "linear_mps": {"x": 0.0, "y": 0.002, "z": 0.0}
                            },
                        },
                    ],
                }
            )
        )
        _write_object_csv(
            run_dir / "truth" / "objects" / "model_01.csv",
            [
                (0.0, 0.0, 0.0, 0.0),
                (1.0, 0.0, 0.02, 0.0),
            ],
        )
        _write_object_csv(
            run_dir / "truth" / "objects" / "model_02.csv",
            [
                (0.0, 0.0, 0.0, 0.0),
                (1.0, 0.0, 0.04, 0.0),
            ],
        )
        return run_dir

    def test_compute_run_metrics_expands_one_run_into_two_controlled_object_rows(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = self._create_multi_control_run(
                pathlib.Path(tmp_dir) / "output",
                "20260405",
                "sim_run_000",
            )
            fake_metrics = {
                "R_r": {"R_r": 1.0, "N_GT": 2, "N_matched": 2, "details": []},
                "F_c": {"F_c": 0.0, "N_confirmed": 2, "N_false": 0},
                "t_resp": {
                    "mean_t_resp": 0.25,
                    "per_object": [
                        {
                            "object": "model_01",
                            "t_resp": 0.1,
                            "t_first_confirmed": 1.0,
                        },
                        {
                            "object": "model_02",
                            "t_resp": 0.4,
                            "t_first_confirmed": 1.0,
                        },
                    ],
                },
                "beta_d": {"beta_d": 0.2, "N_samples": 6, "per_object": {}},
            }

            with mock.patch(
                "analysis_script.run_simulation_paper_analysis.compute_metrics.run_metrics",
                return_value=fake_metrics,
            ):
                rows = run_simulation_paper_analysis.compute_run_metrics(
                    run_dir,
                    match_radius=common.MATCH_RADIUS,
                )

            self.assertEqual(len(rows), 2)
            by_object = {row["controlled_object"]: row for row in rows}
            self.assertEqual(sorted(by_object), ["model_01", "model_02"])
            self.assertEqual(by_object["model_01"]["valid_t_resp_values"], [0.1])
            self.assertEqual(by_object["model_02"]["valid_t_resp_values"], [0.4])
            self.assertAlmostEqual(by_object["model_01"]["velocity_mmps"], 1.0, places=6)
            self.assertAlmostEqual(by_object["model_02"]["velocity_mmps"], 2.0, places=6)
            self.assertAlmostEqual(
                by_object["model_01"]["target_detection"]["gt_disp_at_detection_mm"],
                20.0,
                places=6,
            )
            self.assertAlmostEqual(
                by_object["model_02"]["target_detection"]["gt_disp_at_detection_mm"],
                40.0,
                places=6,
            )

    def test_sweep_mdd_writes_one_row_per_controlled_object(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = self._create_multi_control_run(
                pathlib.Path(tmp_dir) / "output",
                "20260405",
                "sim_run_000",
            )
            fake_metrics = {
                "R_r": {"R_r": 1.0, "N_GT": 2, "N_matched": 2, "details": []},
                "t_resp": {
                    "per_object": [
                        {
                            "object": "model_01",
                            "t_resp": 0.1,
                            "t_first_confirmed": 1.0,
                        },
                        {
                            "object": "model_02",
                            "t_resp": None,
                            "t_first_confirmed": None,
                        },
                    ]
                },
            }

            with mock.patch(
                "analysis_script.compute_mdd.compute_metrics.run_metrics",
                return_value=fake_metrics,
            ):
                rows = compute_mdd.sweep_mdd([run_dir], match_radius=common.MATCH_RADIUS)

            self.assertEqual(len(rows), 2)
            by_object = {row["controlled_object"]: row for row in rows}
            self.assertEqual(by_object["model_01"]["run_dir"], run_dir.name)
            self.assertEqual(by_object["model_02"]["run_dir"], run_dir.name)
            self.assertTrue(by_object["model_01"]["detected"])
            self.assertFalse(by_object["model_02"]["detected"])
            self.assertAlmostEqual(by_object["model_01"]["velocity_mmps"], 1.0, places=6)
            self.assertAlmostEqual(by_object["model_02"]["velocity_mmps"], 2.0, places=6)

    def test_print_mdd_table_handles_missing_velocity_rows(self):
        rows = [
            {
                "run_dir": "sim_run_000",
                "velocity_mmps": None,
                "controlled_object": "model_00",
                "detected": False,
                "R_r": None,
                "t_resp_s": None,
                "gt_disp_at_detection_mm": None,
            },
            {
                "run_dir": "sim_run_001",
                "velocity_mmps": 1.0,
                "controlled_object": "model_01",
                "detected": True,
                "R_r": 1.0,
                "t_resp_s": 0.1,
                "gt_disp_at_detection_mm": 10.0,
            },
        ]

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            compute_mdd.print_mdd_table(rows)

        output = buffer.getvalue()
        self.assertIn("N/A", output)
        self.assertIn("1.0", output)

    def test_compute_velocity_mdd_rows_counts_unique_runs_per_velocity(self):
        run_dir_0 = pathlib.Path("/tmp/run_0")
        run_dir_1 = pathlib.Path("/tmp/run_1")
        rows = [
            {
                "run_dir": run_dir_0,
                "run_name": "sim_run_000",
                "controlled_object": "model_01",
                "velocity_mmps": 1.0,
                "target_detection": {
                    "detected": True,
                    "t_resp_s": 0.1,
                    "gt_disp_at_detection_mm": 10.0,
                },
            },
            {
                "run_dir": run_dir_0,
                "run_name": "sim_run_000",
                "controlled_object": "model_02",
                "velocity_mmps": 1.0,
                "target_detection": {
                    "detected": False,
                    "t_resp_s": None,
                    "gt_disp_at_detection_mm": None,
                },
            },
            {
                "run_dir": run_dir_1,
                "run_name": "sim_run_001",
                "controlled_object": "model_03",
                "velocity_mmps": 1.0,
                "target_detection": {
                    "detected": True,
                    "t_resp_s": 0.3,
                    "gt_disp_at_detection_mm": 30.0,
                },
            },
        ]

        summary_rows = run_simulation_paper_analysis.compute_velocity_mdd_rows(rows)

        self.assertEqual(len(summary_rows), 1)
        summary = summary_rows[0]
        self.assertEqual(summary["total_runs"], 2)
        self.assertEqual(summary["detected_runs"], 2)

    def test_summarize_scenarios_counts_unique_runs_per_velocity(self):
        run_dir_0 = pathlib.Path("/tmp/run_0")
        run_dir_1 = pathlib.Path("/tmp/run_1")
        rows = [
            {
                "run_dir": run_dir_0,
                "run_name": "sim_run_000",
                "controlled_object": "model_01",
                "velocity_mmps": 1.0,
                "moving_gt_count": 2,
                "run_duration_s": 1.0,
                "motion_duration_s": 0.5,
            },
            {
                "run_dir": run_dir_0,
                "run_name": "sim_run_000",
                "controlled_object": "model_02",
                "velocity_mmps": 1.0,
                "moving_gt_count": 2,
                "run_duration_s": 1.0,
                "motion_duration_s": 0.6,
            },
            {
                "run_dir": run_dir_1,
                "run_name": "sim_run_001",
                "controlled_object": "model_03",
                "velocity_mmps": 2.0,
                "moving_gt_count": 1,
                "run_duration_s": 2.0,
                "motion_duration_s": 1.0,
            },
        ]

        summary = run_simulation_paper_analysis.summarize_scenarios(rows)

        self.assertEqual(summary["total_runs"], 2)
        self.assertEqual(summary["repeats_per_velocity"], {"1.000": 1, "2.000": 1})

    def test_batch_script_writes_summary_outputs_under_analysis_result_only(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = pathlib.Path(tmp_dir)
            output_root = tmp_path / "output" / "20260405"
            run_dir_0 = output_root / "sim_run_000"
            run_dir_1 = output_root / "sim_run_001"
            for run_dir, scenario_id in (
                (run_dir_0, "sim_main_block02_ypos_0p5mmps_r01"),
                (run_dir_1, "sim_main_block02_ypos_1p0mmps_r01"),
            ):
                meta_dir = run_dir / "meta"
                meta_dir.mkdir(parents=True)
                (meta_dir / "scenario_manifest.json").write_text(
                    json.dumps({"scenario_id": scenario_id, "controls": []})
                )

            with TemporaryResultRoot(tmp_path) as result_root:
                fake_metrics = [
                    {
                        "run_dir": str(run_dir_0),
                        "R_r": {"R_r": 1.0, "N_GT": 1, "N_matched": 1, "details": []},
                        "F_c": {"F_c": 0.0, "N_confirmed": 1, "N_false": 0},
                        "t_resp": {"mean_t_resp": 10.0, "per_object": []},
                        "beta_d": {"beta_d": 0.1, "N_samples": 4, "per_object": {}},
                    },
                    {
                        "run_dir": str(run_dir_1),
                        "R_r": {"R_r": 1.0, "N_GT": 1, "N_matched": 1, "details": []},
                        "F_c": {"F_c": 0.0, "N_confirmed": 1, "N_false": 0},
                        "t_resp": {"mean_t_resp": 8.0, "per_object": []},
                        "beta_d": {"beta_d": 0.2, "N_samples": 6, "per_object": {}},
                    },
                ]

                with mock.patch(
                    "analysis_script.run_simulation_paper_analysis.compute_metrics.run_metrics",
                    side_effect=fake_metrics,
                ), mock.patch(
                    "analysis_script.run_simulation_paper_analysis.aggregate_runtime_table",
                    return_value={
                        "stage_a_ms": 1.0,
                        "stage_b_ms": 2.0,
                        "stage_c_ms": 3.0,
                        "stage_d_ms": 4.0,
                        "total_ms": 10.0,
                    },
                ), mock.patch(
                    "analysis_script.run_simulation_paper_analysis.run_representative_figures",
                    return_value=[],
                ):
                    outputs = run_simulation_paper_analysis.main(
                        ["--output-root", str(output_root), "--skip-figures"]
                    )

                summary_dir = result_root / "20260405" / "summary"
                self.assertEqual(outputs["summary_dir"], summary_dir)
                self.assertTrue((summary_dir / "simulation_detection_summary.json").is_file())
                self.assertTrue((summary_dir / "mdd_velocity_summary.csv").is_file())
                self.assertTrue((summary_dir / "runtime_summary_table.json").is_file())
                with (summary_dir / "per_run_metrics.csv").open() as f:
                    rows = list(csv.DictReader(f))
                self.assertEqual(len(rows), 2)
                self.assertFalse((run_dir_0 / "evaluation").exists())
                self.assertFalse((run_dir_1 / "evaluation").exists())

    def test_batch_script_flattens_multi_control_rows_into_summary_csv(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = pathlib.Path(tmp_dir)
            output_root = tmp_path / "output" / "20260405"
            run_dir_0 = output_root / "sim_run_000"
            run_dir_1 = output_root / "sim_run_001"
            for run_dir in (run_dir_0, run_dir_1):
                meta_dir = run_dir / "meta"
                meta_dir.mkdir(parents=True)
                (meta_dir / "scenario_manifest.json").write_text(
                    json.dumps({"scenario_id": "sim_main_block02_ypos_1p0mmps_r01", "controls": []})
                )

            flattened_rows = {
                run_dir_0: [
                    {
                        "run_dir": run_dir_0,
                        "run_name": run_dir_0.name,
                        "scenario_id": "s0",
                        "controlled_object": "model_01",
                        "velocity_mmps": 1.0,
                        "moving_gt_count": 2,
                        "run_duration_s": 1.0,
                        "motion_duration_s": 1.0,
                        "metrics": {
                            "R_r": {"R_r": 1.0, "N_GT": 2, "N_matched": 2},
                            "F_c": {"F_c": 0.0, "N_confirmed": 1, "N_false": 0},
                            "beta_d": {"beta_d": 0.1, "N_samples": 4},
                        },
                        "valid_t_resp_values": [0.1],
                        "target_detection": {
                            "t_resp_s": 0.1,
                            "gt_disp_at_detection_mm": 10.0,
                            "detected": True,
                        },
                    },
                    {
                        "run_dir": run_dir_0,
                        "run_name": run_dir_0.name,
                        "scenario_id": "s0",
                        "controlled_object": "model_02",
                        "velocity_mmps": 2.0,
                        "moving_gt_count": 2,
                        "run_duration_s": 1.0,
                        "motion_duration_s": 1.0,
                        "metrics": {
                            "R_r": {"R_r": 1.0, "N_GT": 2, "N_matched": 2},
                            "F_c": {"F_c": 0.0, "N_confirmed": 1, "N_false": 0},
                            "beta_d": {"beta_d": 0.1, "N_samples": 4},
                        },
                        "valid_t_resp_values": [0.2],
                        "target_detection": {
                            "t_resp_s": 0.2,
                            "gt_disp_at_detection_mm": 20.0,
                            "detected": True,
                        },
                    },
                ],
                run_dir_1: [
                    {
                        "run_dir": run_dir_1,
                        "run_name": run_dir_1.name,
                        "scenario_id": "s1",
                        "controlled_object": "model_03",
                        "velocity_mmps": 3.0,
                        "moving_gt_count": 1,
                        "run_duration_s": 1.0,
                        "motion_duration_s": 1.0,
                        "metrics": {
                            "R_r": {"R_r": 1.0, "N_GT": 1, "N_matched": 1},
                            "F_c": {"F_c": 0.0, "N_confirmed": 1, "N_false": 0},
                            "beta_d": {"beta_d": 0.3, "N_samples": 5},
                        },
                        "valid_t_resp_values": [0.3],
                        "target_detection": {
                            "t_resp_s": 0.3,
                            "gt_disp_at_detection_mm": 30.0,
                            "detected": True,
                        },
                    }
                ],
            }

            with TemporaryResultRoot(tmp_path) as result_root:
                with mock.patch(
                    "analysis_script.run_simulation_paper_analysis.compute_run_metrics",
                    side_effect=lambda run_dir, match_radius: flattened_rows[run_dir],
                ), mock.patch(
                    "analysis_script.run_simulation_paper_analysis.aggregate_runtime_table",
                    return_value={
                        "stage_a_ms": 1.0,
                        "stage_b_ms": 2.0,
                        "stage_c_ms": 3.0,
                        "stage_d_ms": 4.0,
                        "total_ms": 10.0,
                    },
                ), mock.patch(
                    "analysis_script.run_simulation_paper_analysis.run_representative_figures",
                    return_value=[],
                ):
                    outputs = run_simulation_paper_analysis.main(
                        ["--output-root", str(output_root), "--skip-figures"]
                    )

                summary_dir = result_root / "20260405" / "summary"
                self.assertEqual(outputs["summary_dir"], summary_dir)
                with (summary_dir / "per_run_metrics.csv").open() as f:
                    rows = list(csv.DictReader(f))
                self.assertEqual(len(rows), 3)
                self.assertEqual(
                    [row["controlled_object"] for row in rows],
                    ["model_01", "model_02", "model_03"],
                )
                self.assertFalse((run_dir_0 / "evaluation").exists())
                self.assertFalse((run_dir_1 / "evaluation").exists())


class AblationAggregationTests(unittest.TestCase):
    def test_compare_ablation_keeps_controlled_object_dimension(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = pathlib.Path(tmp_dir)
            full_root = tmp_path / "output" / "20260405"
            run_dirs = [
                full_root / "sim_run_000",
                full_root / "sim_run_001",
            ]

            for run_dir in run_dirs:
                meta_dir = run_dir / "meta"
                meta_dir.mkdir(parents=True)
                (meta_dir / "ablation_manifest.json").write_text(
                    json.dumps({"variant": "full_pipeline"})
                )
                (meta_dir / "scenario_manifest.json").write_text(
                    json.dumps(
                        {
                            "controls": [
                                {"controlled_object": "model_01"},
                                {"controlled_object": "model_02"},
                            ]
                        }
                    )
                )
                _write_object_csv(
                    run_dir / "truth" / "objects" / "model_01.csv",
                    [
                        (0.0, 0.0, 0.0, 0.0),
                        (1.0, 0.0, 0.02, 0.0),
                    ],
                )
                _write_object_csv(
                    run_dir / "truth" / "objects" / "model_02.csv",
                    [
                        (0.0, 0.0, 0.0, 0.0),
                        (1.0, 0.0, 0.04, 0.0),
                    ],
                )

            fake_metrics = [
                {
                    "R_r": {
                        "R_r": 0.5,
                        "N_GT": 2,
                        "N_matched": 1,
                        "details": [
                            {"object": "model_01", "matched": True},
                            {"object": "model_02", "matched": False},
                        ],
                    },
                    "F_c": {"F_c": 0.0, "N_confirmed": 2, "N_false": 0},
                    "P_p": {"P_p": 0.5, "N_zones": 4, "N_tp_zones": 2, "N_qualified": 4, "N_tp": 2},
                    "t_resp": {
                        "mean_t_resp": 0.2,
                        "per_object": [
                            {"object": "model_01", "t_resp": 0.1},
                            {"object": "model_02", "t_resp": 0.3},
                        ],
                    },
                    "beta_d": {
                        "beta_d": 0.3,
                        "N_samples": 4,
                        "per_object": {
                            "model_01": {"mean_bias": 0.2, "n_samples": 2},
                            "model_02": {"mean_bias": 0.4, "n_samples": 2},
                        },
                    },
                    "epsilon_d": {
                        "epsilon_d": 0.1,
                        "N_samples": 2,
                        "per_object": {
                            "model_01": {"epsilon_d": 0.05},
                            "model_02": {"epsilon_d": 0.15},
                        },
                    },
                },
                {
                    "R_r": {
                        "R_r": 1.0,
                        "N_GT": 2,
                        "N_matched": 2,
                        "details": [
                            {"object": "model_01", "matched": True},
                            {"object": "model_02", "matched": True},
                        ],
                    },
                    "F_c": {"F_c": 0.25, "N_confirmed": 4, "N_false": 1},
                    "P_p": {"P_p": 0.75, "N_zones": 4, "N_tp_zones": 3, "N_qualified": 4, "N_tp": 3},
                    "t_resp": {
                        "mean_t_resp": 0.45,
                        "per_object": [
                            {"object": "model_01", "t_resp": 0.2},
                            {"object": "model_02", "t_resp": 0.7},
                        ],
                    },
                    "beta_d": {
                        "beta_d": 0.45,
                        "N_samples": 4,
                        "per_object": {
                            "model_01": {"mean_bias": 0.3, "n_samples": 1},
                            "model_02": {"mean_bias": 0.5, "n_samples": 3},
                        },
                    },
                    "epsilon_d": {
                        "epsilon_d": 0.2,
                        "N_samples": 2,
                        "per_object": {
                            "model_01": {"epsilon_d": 0.10},
                            "model_02": {"epsilon_d": 0.30},
                        },
                    },
                },
            ]

            with TemporaryResultRoot(tmp_path):
                with mock.patch(
                    "analysis_script.compare_ablation.compute_metrics.run_metrics",
                    side_effect=fake_metrics,
                ):
                    compare_ablation.main(
                        [
                            "--full-pipeline-root",
                            str(full_root),
                        ]
                    )

                out_path = tmp_path / "result" / "ablation_comparison.csv"
                self.assertTrue(out_path.is_file())
                with out_path.open() as f:
                    rows = list(csv.DictReader(f))

            self.assertEqual(len(rows), 2)
            by_object = {row["controlled_object"]: row for row in rows}
            self.assertEqual(sorted(by_object), ["model_01", "model_02"])
            self.assertEqual(by_object["model_01"]["variant"], "full_pipeline")
            self.assertEqual(by_object["model_02"]["variant"], "full_pipeline")
            self.assertAlmostEqual(float(by_object["model_01"]["R_r"]), 1.0, places=6)
            self.assertAlmostEqual(float(by_object["model_02"]["R_r"]), 0.5, places=6)
            self.assertNotEqual(by_object["model_01"]["R_r"], by_object["model_02"]["R_r"])
            self.assertEqual(by_object["model_01"]["N_GT"], "2")
            self.assertEqual(by_object["model_01"]["N_matched"], "2")
            self.assertEqual(by_object["model_02"]["N_GT"], "2")
            self.assertEqual(by_object["model_02"]["N_matched"], "1")
            self.assertEqual(by_object["model_01"]["F_c"], "")
            self.assertEqual(by_object["model_02"]["F_c"], "")
            self.assertEqual(by_object["model_01"]["P_p"], "")
            self.assertEqual(by_object["model_02"]["P_p"], "")
            self.assertEqual(by_object["model_01"]["P_p_track"], "")
            self.assertEqual(by_object["model_02"]["P_p_track"], "")
            self.assertAlmostEqual(float(by_object["model_01"]["t_resp_s"]), 0.15, places=6)
            self.assertAlmostEqual(float(by_object["model_02"]["t_resp_s"]), 0.5, places=6)
            self.assertAlmostEqual(float(by_object["model_01"]["beta_d"]), 0.2333333333, places=6)
            self.assertAlmostEqual(float(by_object["model_02"]["beta_d"]), 0.46, places=6)
            self.assertAlmostEqual(float(by_object["model_01"]["epsilon_d"]), 0.075, places=6)
            self.assertAlmostEqual(float(by_object["model_02"]["epsilon_d"]), 0.225, places=6)

    def test_compare_ablation_aggregates_variants_across_full_and_ablation_roots(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = pathlib.Path(tmp_dir)
            full_root = tmp_path / "output" / "20260405"
            ablation_root = tmp_path / "output_ablation" / "20260406"

            full_runs = [
                full_root / "sim_run_000",
                full_root / "sim_run_001",
            ]
            ablation_runs = [
                ablation_root / "sim_run_000",
                ablation_root / "sim_run_001",
            ]

            for run_dir, variant in (
                (full_runs[0], "full_pipeline"),
                (full_runs[1], "full_pipeline"),
                (ablation_runs[0], "no_cov_inflation"),
                (ablation_runs[1], "no_cov_inflation"),
            ):
                meta_dir = run_dir / "meta"
                meta_dir.mkdir(parents=True)
                (meta_dir / "ablation_manifest.json").write_text(
                    json.dumps({"variant": variant})
                )

            fake_metrics = [
                {
                    "R_r": {"R_r": 1.0, "N_GT": 1, "N_matched": 1},
                    "F_c": {"F_c": 0.0, "N_confirmed": 2, "N_false": 0},
                    "t_resp": {"mean_t_resp": 10.0},
                    "beta_d": {"beta_d": 0.1, "N_samples": 4},
                },
                {
                    "R_r": {"R_r": 0.0, "N_GT": 1, "N_matched": 0},
                    "F_c": {"F_c": 0.5, "N_confirmed": 2, "N_false": 1},
                    "t_resp": {"mean_t_resp": None},
                    "beta_d": {"beta_d": 0.2, "N_samples": 6},
                },
                {
                    "R_r": {"R_r": 1.0, "N_GT": 1, "N_matched": 1},
                    "F_c": {"F_c": 1.0, "N_confirmed": 1, "N_false": 1},
                    "t_resp": {"mean_t_resp": 20.0},
                    "beta_d": {"beta_d": 0.3, "N_samples": 5},
                },
                {
                    "R_r": {"R_r": 1.0, "N_GT": 1, "N_matched": 1},
                    "F_c": {"F_c": 0.0, "N_confirmed": 3, "N_false": 0},
                    "t_resp": {"mean_t_resp": 30.0},
                    "beta_d": {"beta_d": 0.5, "N_samples": 5},
                },
            ]

            with TemporaryResultRoot(tmp_path):
                with mock.patch(
                    "analysis_script.compare_ablation.compute_metrics.run_metrics",
                    side_effect=fake_metrics,
                ):
                    compare_ablation.main(
                        [
                            "--full-pipeline-root",
                            str(full_root),
                            "--ablation-root",
                            str(ablation_root),
                        ]
                    )

                out_path = tmp_path / "result" / "ablation_comparison.csv"
                self.assertTrue(out_path.is_file())
                with out_path.open() as f:
                    rows = list(csv.DictReader(f))
                self.assertEqual(len(rows), 2)

                by_variant = {row["variant"]: row for row in rows}
                self.assertIn("full_pipeline", by_variant)
                self.assertIn("no_cov_inflation", by_variant)

                full_row = by_variant["full_pipeline"]
                self.assertEqual(full_row["n_runs"], "2")
                self.assertAlmostEqual(float(full_row["R_r"]), 0.5, places=6)
                self.assertAlmostEqual(float(full_row["F_c"]), 0.25, places=6)
                self.assertAlmostEqual(float(full_row["t_resp_s"]), 10.0, places=6)
                self.assertAlmostEqual(float(full_row["beta_d"]), 0.16, places=6)

                ablation_row = by_variant["no_cov_inflation"]
                self.assertEqual(ablation_row["n_runs"], "2")
                self.assertAlmostEqual(float(ablation_row["R_r"]), 1.0, places=6)
                self.assertAlmostEqual(float(ablation_row["F_c"]), 0.25, places=6)
                self.assertAlmostEqual(float(ablation_row["t_resp_s"]), 25.0, places=6)
                self.assertAlmostEqual(float(ablation_row["beta_d"]), 0.4, places=6)


class DirectionConsistencyMetricTests(unittest.TestCase):
    def _moving_gt_object(self):
        return common.GTObject(
            name="moving_target",
            classification="moving",
            net_displacement=0.02,
            peak_displacement=0.02,
            onset_time=1.0,
            end_time=2.0,
            positions_t=[0.0, 1.0, 2.0],
            positions_xyz=[
                (0.0, 0.0, 0.0),
                (0.0, 0.01, 0.0),
                (0.0, 0.02, 0.0),
            ],
        )

    def test_rr_rejects_spatial_match_when_cluster_direction_opposes_gt(self):
        gt_obj = self._moving_gt_object()
        persistent_records = [
            {
                "header": {"stamp": {"sec": 2.0}},
                "regions": [
                    {
                        "confirmed": True,
                        "track_id": 7,
                        "center": {"x": 0.0, "y": 0.02, "z": 0.0},
                        "bbox_min": {"x": -0.05, "y": 0.0, "z": -0.05},
                        "bbox_max": {"x": 0.05, "y": 0.05, "z": 0.05},
                    }
                ],
            }
        ]
        cluster_records = [
            {
                "header": {"stamp": {"sec": 2.0}},
                "clusters": [
                    {
                        "significant": True,
                        "center": {"x": 0.0, "y": 0.02, "z": 0.0},
                        "disp_mean": [0.0, -0.02, 0.0],
                        "disp_norm": 0.02,
                    }
                ],
            }
        ]

        rr = compute_metrics.compute_Rr(
            [gt_obj],
            persistent_records,
            cluster_records,
            np.eye(4),
            common.MATCH_RADIUS,
        )
        self.assertEqual(rr["R_r"], 0.0)
        self.assertFalse(rr["details"][0]["matched"])

    def test_rr_accepts_spatial_match_when_cluster_direction_aligns_with_gt(self):
        gt_obj = self._moving_gt_object()
        persistent_records = [
            {
                "header": {"stamp": {"sec": 2.0}},
                "regions": [
                    {
                        "confirmed": True,
                        "track_id": 7,
                        "center": {"x": 0.0, "y": 0.02, "z": 0.0},
                        "bbox_min": {"x": -0.05, "y": 0.0, "z": -0.05},
                        "bbox_max": {"x": 0.05, "y": 0.05, "z": 0.05},
                    }
                ],
            }
        ]
        cluster_records = [
            {
                "header": {"stamp": {"sec": 2.0}},
                "clusters": [
                    {
                        "significant": True,
                        "center": {"x": 0.0, "y": 0.02, "z": 0.0},
                        "disp_mean": [0.0, 0.02, 0.0],
                        "disp_norm": 0.02,
                    }
                ],
            }
        ]

        rr = compute_metrics.compute_Rr(
            [gt_obj],
            persistent_records,
            cluster_records,
            np.eye(4),
            common.MATCH_RADIUS,
        )
        self.assertEqual(rr["R_r"], 1.0)
        self.assertTrue(rr["details"][0]["matched"])

    def test_beta_d_skips_directionally_inconsistent_clusters(self):
        gt_obj = self._moving_gt_object()
        cluster_records = [
            {
                "header": {"stamp": {"sec": 2.0}},
                "clusters": [
                    {
                        "significant": True,
                        "center": {"x": 0.0, "y": 0.02, "z": 0.0},
                        "disp_mean": [0.0, -0.02, 0.0],
                        "disp_norm": 0.02,
                    }
                ],
            }
        ]

        beta = compute_metrics.compute_beta_d(
            [gt_obj],
            cluster_records,
            np.eye(4),
            common.MATCH_RADIUS,
        )
        self.assertIsNone(beta["beta_d"])
        self.assertEqual(beta["N_samples"], 0)

    def test_run_metrics_uses_all_analysis_controlled_objects_for_beta_and_epsilon(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = pathlib.Path(tmp_dir) / "20260405" / "sim_run_000"
            run_dir.mkdir(parents=True)
            gt_obj_1 = common.GTObject(
                name="model_01",
                classification="moving",
                net_displacement=0.02,
                peak_displacement=0.02,
                onset_time=1.0,
                end_time=2.0,
                positions_t=[0.0, 1.0, 2.0],
                positions_xyz=[
                    (0.0, 0.0, 0.0),
                    (0.0, 0.01, 0.0),
                    (0.0, 0.02, 0.0),
                ],
            )
            gt_obj_2 = common.GTObject(
                name="model_02",
                classification="moving",
                net_displacement=0.03,
                peak_displacement=0.03,
                onset_time=1.0,
                end_time=2.0,
                positions_t=[0.0, 1.0, 2.0],
                positions_xyz=[
                    (0.0, 0.0, 0.0),
                    (0.0, 0.015, 0.0),
                    (0.0, 0.03, 0.0),
                ],
            )
            run_data = common.RunData(
                run_dir=run_dir,
                gt_objects=[gt_obj_1, gt_obj_2],
                alignment={},
                persistent_records=[],
                track_events=[],
                cluster_records=[],
            )

            with mock.patch(
                "analysis_script.compute_metrics.common.load_run_data",
                return_value=run_data,
            ), mock.patch(
                "analysis_script.compute_metrics.common.build_world_from_algorithm_transform",
                return_value=np.eye(4),
            ), mock.patch(
                "analysis_script.compute_metrics.common.get_analysis_controlled_object_names",
                return_value=["model_01", "model_02"],
            ), mock.patch(
                "analysis_script.compute_metrics.compute_Rr",
                return_value={
                    "R_r": 1.0,
                    "N_GT": 2,
                    "N_matched": 2,
                    "details": [
                        {
                            "object": "model_01",
                            "matched": True,
                            "first_match_time": 1.0,
                            "gt_onset": 1.0,
                        },
                        {
                            "object": "model_02",
                            "matched": True,
                            "first_match_time": 1.0,
                            "gt_onset": 1.0,
                        },
                    ],
                },
            ), mock.patch(
                "analysis_script.compute_metrics.compute_Fc",
                return_value={"F_c": 0.0, "N_confirmed": 0, "N_false": 0},
            ), mock.patch(
                "analysis_script.compute_metrics.compute_Pp",
                return_value={
                    "P_p": 1.0,
                    "N_tp": 1,
                    "N_qualified": 1,
                    "min_age_frames": 10,
                    "min_mean_risk": 0.6,
                },
            ), mock.patch(
                "analysis_script.compute_metrics.compute_t_resp",
                return_value={"mean_t_resp": 0.0, "per_object": []},
            ), mock.patch(
                "analysis_script.compute_metrics.compute_beta_d",
                return_value={"beta_d": 0.0, "N_samples": 0, "per_object": {}},
            ) as mock_beta, mock.patch(
                "analysis_script.compute_metrics.compute_epsilon_d",
                return_value={"epsilon_d": 0.0, "N_samples": 0, "per_object": {}},
            ) as mock_epsilon, mock.patch(
                "analysis_script.compute_metrics.common.result_dir_for_run",
                return_value=run_dir,
            ):
                compute_metrics.run_metrics(run_dir, match_radius=common.MATCH_RADIUS)

            self.assertEqual(
                [obj.name for obj in mock_beta.call_args.args[0]],
                ["model_01", "model_02"],
            )
            self.assertEqual(
                [obj.name for obj in mock_epsilon.call_args.args[0]],
                ["model_01", "model_02"],
            )

    def test_rr_rejects_when_nearest_cluster_is_directionally_wrong_even_if_farther_one_aligns(self):
        gt_obj = self._moving_gt_object()
        persistent_records = [
            {
                "header": {"stamp": {"sec": 2.0}},
                "regions": [
                    {
                        "confirmed": True,
                        "track_id": 7,
                        "center": {"x": 0.0, "y": 0.02, "z": 0.0},
                        "bbox_min": {"x": -0.05, "y": 0.0, "z": -0.05},
                        "bbox_max": {"x": 0.05, "y": 0.05, "z": 0.05},
                    }
                ],
            }
        ]
        cluster_records = [
            {
                "header": {"stamp": {"sec": 2.0}},
                "clusters": [
                    {
                        "significant": True,
                        "center": {"x": 0.0, "y": 0.021, "z": 0.0},
                        "disp_mean": [0.0, -0.02, 0.0],
                        "disp_norm": 0.02,
                    },
                    {
                        "significant": True,
                        "center": {"x": 0.0, "y": 0.50, "z": 0.0},
                        "disp_mean": [0.0, 0.02, 0.0],
                        "disp_norm": 0.02,
                    },
                ],
            }
        ]

        rr = compute_metrics.compute_Rr(
            [gt_obj],
            persistent_records,
            cluster_records,
            np.eye(4),
            common.MATCH_RADIUS,
        )
        self.assertEqual(rr["R_r"], 0.0)


if __name__ == "__main__":
    unittest.main()
