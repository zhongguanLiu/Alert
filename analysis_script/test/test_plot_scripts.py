"""Test shared plotting helpers and scripts."""

from __future__ import annotations

import atexit
import json
import os
import pathlib
import tempfile
import unittest
import warnings
from unittest import mock

_config_dir_obj = tempfile.TemporaryDirectory(prefix="matplotlib-config-")
atexit.register(_config_dir_obj.cleanup)
os.environ.setdefault("MPLCONFIGDIR", _config_dir_obj.name)
os.environ.setdefault("MPLBACKEND", "Agg")

warnings.filterwarnings(
    "error",
    message=".*More than 20 figures have been opened.*",
    category=RuntimeWarning,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import analysis_script.common as common
import analysis_script.plot_common as plot_common
import analysis_script.plot_runtime_profile as plot_runtime_profile
import analysis_script.plot_mdd_summary as plot_mdd_summary
import analysis_script.plot_ablation_summary as plot_ablation_summary
import analysis_script.plot_sim_spatial_overlay as plot_sim_spatial_overlay
import analysis_script.plot_sim_timeline as plot_sim_timeline


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


class PlotCommonTests(unittest.TestCase):
    def test_repo_path_helpers_are_relative_to_repository(self):
        repo_root = common.repo_root()
        analysis_root = common.analysis_root()
        self.assertEqual(analysis_root, repo_root / "analysis_script")
        self.assertEqual(common.DEFAULT_OUTPUT_ROOT, repo_root / "output")
        self.assertEqual(common.DEFAULT_REAL_OUTPUT_ROOT, repo_root / "real_output")

    def test_real_run_helpers_build_public_repo_paths(self):
        self.assertEqual(
            common.default_real_run_dir("real_run_001"),
            common.repo_root() / "real_output" / "real_run_001" / "algorithm",
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = pathlib.Path(tmp_dir)
            with TemporaryResultRoot(tmp_path) as result_root:
                figure_dir = common.real_figure_dir()
                self.assertEqual(figure_dir, result_root / "real_runs")
                self.assertTrue(figure_dir.exists())
                pdf_path, png_path = common.real_timeline_output_paths("real_run_001")
                self.assertEqual(pdf_path, figure_dir / "real_run_001_timeline.pdf")
                self.assertEqual(png_path, figure_dir / "real_run_001_timeline.png")

    def test_latest_result_date_dir_returns_latest_dated_directory(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = pathlib.Path(tmp_dir)
            with TemporaryResultRoot(tmp_path) as result_root:
                (result_root / "20260407").mkdir()
                (result_root / "20260408").mkdir()
                (result_root / "summary").mkdir()
                self.assertEqual(
                    common.latest_result_date_dir(),
                    result_root / "20260408",
                )

    def test_figure_dir_for_run_creates_figures_directory(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = pathlib.Path(tmp_dir)
            run_dir = tmp_path / "20260404" / "sim_run_000"
            run_dir.mkdir(parents=True)
            with TemporaryResultRoot(tmp_path) as result_root:
                figure_dir = plot_common.figure_dir_for_run(run_dir)
                self.assertTrue(figure_dir.exists())
                self.assertEqual(figure_dir.name, "figures")
                expected_parent = common.result_dir_for_run(run_dir)
                self.assertEqual(figure_dir.parent, expected_parent)

    def test_summary_figure_dir_creates_shared_output(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = pathlib.Path(tmp_dir)
            with TemporaryResultRoot(tmp_path) as result_root:
                summary = plot_common.summary_figure_dir()
                self.assertTrue(summary.exists())
                self.assertEqual(summary, result_root / "summary" / "figures")

    def test_save_figure_writes_file(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = pathlib.Path(tmp_dir) / "figure.png"
            fig = plt.figure()
            plt.plot([0, 1], [0, 1])
            plot_common.save_figure(fig, output_path)
            self.assertTrue(output_path.is_file())
            self.assertGreater(output_path.stat().st_size, 0)

    def test_apply_paper_style_updates_rcparams(self):
        keys = [
            "font.size",
            "axes.grid",
            "grid.linestyle",
            "lines.linewidth",
            "figure.dpi",
        ]
        original = {k: matplotlib.rcParams.get(k) for k in keys}
        with plot_common.apply_paper_style():
            self.assertEqual(matplotlib.rcParams.get("font.size"), 10)
            self.assertTrue(matplotlib.rcParams.get("axes.grid"))
            self.assertEqual(matplotlib.rcParams.get("grid.linestyle"), "--")
            self.assertEqual(matplotlib.rcParams.get("lines.linewidth"), 1.2)
            self.assertEqual(matplotlib.rcParams.get("figure.dpi"), 150)
        self.assertEqual(matplotlib.rcParams.get("font.size"), original["font.size"])
        self.assertEqual(matplotlib.rcParams.get("axes.grid"), original["axes.grid"])
        self.assertEqual(
            matplotlib.rcParams.get("grid.linestyle"), original["grid.linestyle"]
        )
        self.assertEqual(
            matplotlib.rcParams.get("lines.linewidth"), original["lines.linewidth"]
        )
        self.assertEqual(matplotlib.rcParams.get("figure.dpi"), original["figure.dpi"])

    def test_resolve_run_dir_handles_explicit_path(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = pathlib.Path(tmp_dir) / "20260404" / "sim_run_explicit"
            run_dir.mkdir(parents=True)
            resolved = common.resolve_run_dir(run_dir=str(run_dir))
            self.assertEqual(resolved, run_dir)

    def test_result_dir_for_run_handles_irregular_layout(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = pathlib.Path(tmp_dir)
            run_dir = tmp_path / "side_folder" / "custom_run"
            run_dir.mkdir(parents=True)
            with TemporaryResultRoot(tmp_path) as result_root:
                target = common.result_dir_for_run(run_dir)
                self.assertTrue(target.exists())
                self.assertEqual(target.parent.name, "external_runs")
                self.assertIn("custom_run", target.name)

    def test_build_world_transform_invalid_numeric_returns_none(self):
        alignment = {
            "world_from_algorithm_transform": {
                "pose": {
                    "position": {"x": "bad", "y": 0.0, "z": 0.0},
                    "orientation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
                }
            }
        }
        transform = common.build_world_from_algorithm_transform(alignment)
        self.assertIsNone(transform)

    def test_record_time_sec_invalid_numeric_returns_none(self):
        record = {"header": {"stamp": {"sec": "bad"}}}
        self.assertIsNone(common.record_time_sec(record))

    def test_record_time_sec_combines_sec_and_nanosecond_variants(self):
        rec_header_nsec = {"header": {"stamp": {"sec": 12, "nsec": 250000000}}}
        self.assertAlmostEqual(common.record_time_sec(rec_header_nsec), 12.25, places=9)

        rec_header_nanosec = {"header": {"stamp": {"secs": 8, "nanosec": 500000000}}}
        self.assertAlmostEqual(
            common.record_time_sec(rec_header_nanosec), 8.5, places=9
        )

        rec_top_level_stamp = {"stamp": {"sec": 4, "nsecs": 125000000}}
        self.assertAlmostEqual(common.record_time_sec(rec_top_level_stamp), 4.125, places=9)

    def test_bbox_contains_rejects_invalid_and_nonfinite_values(self):
        point = (0.0, 0.0, 0.0)
        self.assertFalse(
            common.bbox_contains(
                {"x": "bad", "y": -1.0, "z": -1.0},
                {"x": 1.0, "y": 1.0, "z": 1.0},
                point,
            )
        )
        self.assertFalse(
            common.bbox_contains(
                {"x": -1.0, "y": -1.0, "z": -1.0},
                {"x": float("inf"), "y": 1.0, "z": 1.0},
                point,
            )
        )

    def test_transform_point_to_world_rejects_nonfinite_values(self):
        import numpy as np

        T = np.eye(4)
        self.assertIsNone(common.transform_point_to_world({"x": "nan", "y": 0.0, "z": 0.0}, T))

        T_bad = np.eye(4)
        T_bad[0, 0] = float("inf")
        self.assertIsNone(common.transform_point_to_world({"x": 1.0, "y": 0.0, "z": 0.0}, T_bad))


class RuntimePlotScriptTests(unittest.TestCase):
    def test_runtime_plot_writes_figures(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = pathlib.Path(tmp_dir)
            run_dir = tmp_path / "20260404" / "sim_run_000"
            runtime_dir = run_dir / "runtime"
            runtime_dir.mkdir(parents=True)
            stage_records = [
                {
                    "stamp": 0.0,
                    "frame_index": 0,
                    "stage_a_ms": 1.0,
                    "stage_b_ms": 1.5,
                    "stage_c_ms": 2.0,
                    "stage_d_ms": 2.5,
                    "total_ms": 7.0,
                },
                {
                    "stamp": 0.1,
                    "frame_index": 1,
                    "stage_a_ms": 1.2,
                    "stage_b_ms": 1.4,
                    "stage_c_ms": 2.1,
                    "stage_d_ms": 2.3,
                    "total_ms": 7.0,
                },
            ]
            stage_file = runtime_dir / "stage_runtime.jsonl"
            with open(stage_file, "w") as f:
                for record in stage_records:
                    json.dump(record, f)
                    f.write("\n")
            with TemporaryResultRoot(tmp_path) as result_root:
                outputs = plot_runtime_profile.main(["--run-dir", str(run_dir)])
                figure_dir = common.result_dir_for_run(run_dir) / "figures"
                self.assertTrue(figure_dir.exists())
                expected_stage = figure_dir / "runtime_stage_means.png"
                expected_total = figure_dir / "runtime_total_line.png"
                self.assertTrue(expected_stage.exists())
                self.assertTrue(expected_total.exists())
        self.assertEqual(outputs, [expected_stage, expected_total])


    def test_runtime_plot_errors_on_missing_stage_data(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = pathlib.Path(tmp_dir)
            run_dir = tmp_path / "20260404" / "sim_run_sparse"
            runtime_dir = run_dir / "runtime"
            runtime_dir.mkdir(parents=True)
            stage_file = runtime_dir / "stage_runtime.jsonl"
            with open(stage_file, "w") as f:
                json.dump({"stamp": 0.0, "frame_index": 0, "total_ms": 5.0}, f)
                f.write("\n")
            with TemporaryResultRoot(tmp_path) as result_root:
                with self.assertRaises(ValueError) as ctx:
                    plot_runtime_profile.main(["--run-dir", str(run_dir)])
                self.assertIn("Stage A", str(ctx.exception))

    def test_runtime_plot_ignores_non_finite_values(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = pathlib.Path(tmp_dir)
            run_dir = tmp_path / "20260404" / "sim_run_nonfinite"
            runtime_dir = run_dir / "runtime"
            runtime_dir.mkdir(parents=True)
            stage_file = runtime_dir / "stage_runtime.jsonl"
            records = [
                {
                    "stamp": 0.0,
                    "frame_index": 0,
                    "stage_a_ms": float("nan"),
                    "stage_b_ms": 1.4,
                    "stage_c_ms": 2.0,
                    "stage_d_ms": 2.2,
                    "total_ms": 7.0,
                },
                {
                    "stamp": 0.1,
                    "frame_index": 1,
                    "stage_a_ms": 1.3,
                    "stage_b_ms": float("inf"),
                    "stage_c_ms": 2.1,
                    "stage_d_ms": 2.3,
                    "total_ms": float("nan"),
                },
            ]
            with open(stage_file, "w") as f:
                for record in records:
                    json.dump(record, f)
                    f.write("\n")
            with TemporaryResultRoot(tmp_path) as result_root:
                outputs = plot_runtime_profile.main(["--run-dir", str(run_dir)])
                figure_dir = common.result_dir_for_run(run_dir) / "figures"
                expected_stage = figure_dir / "runtime_stage_means.png"
                expected_total = figure_dir / "runtime_total_line.png"
                self.assertTrue(expected_stage.exists())
                self.assertTrue(expected_total.exists())
                self.assertEqual(outputs, [expected_stage, expected_total])

    def test_runtime_plot_handles_non_dict_json_lines(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = pathlib.Path(tmp_dir)
            run_dir = tmp_path / "20260404" / "sim_run_nondict"
            runtime_dir = run_dir / "runtime"
            runtime_dir.mkdir(parents=True)
            stage_file = runtime_dir / "stage_runtime.jsonl"
            records = [
                {
                    "stamp": 0.0,
                    "frame_index": 0,
                    "stage_a_ms": 1.1,
                    "stage_b_ms": 1.5,
                    "stage_c_ms": 2.2,
                    "stage_d_ms": 2.0,
                    "total_ms": 7.8,
                },
                {
                    "stamp": 0.1,
                    "frame_index": 1,
                    "stage_a_ms": 1.2,
                    "stage_b_ms": 1.4,
                    "stage_c_ms": 2.1,
                    "stage_d_ms": 2.1,
                    "total_ms": 7.6,
                },
            ]
            with open(stage_file, "w") as f:
                json.dump(records[0], f)
                f.write("\n")
                f.write("[]\n")
                json.dump(records[1], f)
                f.write("\n")
            with TemporaryResultRoot(tmp_path) as result_root:
                outputs = plot_runtime_profile.main(["--run-dir", str(run_dir)])
                figure_dir = common.result_dir_for_run(run_dir) / "figures"
                expected_stage = figure_dir / "runtime_stage_means.png"
                expected_total = figure_dir / "runtime_total_line.png"
                self.assertTrue(expected_stage.exists())
                self.assertTrue(expected_total.exists())
                self.assertEqual(outputs, [expected_stage, expected_total])


class MddPlotScriptTests(unittest.TestCase):
    def test_mdd_plot_writes_summary_figure(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = pathlib.Path(tmp_dir)
            with TemporaryResultRoot(tmp_path) as result_root:
                summary_csv = result_root / "mdd_summary.csv"
                summary_csv.write_text(
                    ",".join(
                        [
                            "run_dir",
                            "velocity_mmps",
                            "controlled_object",
                            "detected",
                            "R_r",
                            "t_resp_s",
                            "gt_disp_at_detection_mm",
                        ]
                    )
                    + "\n"
                    + "sim_run_000,1.0,object_a,True,1.0,12.0,10.0\n"
                    + "sim_run_000,1.2,object_b,False,0.9,13.5,11.2\n"
                )
                outputs, stats = plot_mdd_summary.main(
                    [
                        "--summary-csv",
                        str(summary_csv),
                        "--result-root",
                        str(result_root),
                    ]
                )
                figure_dir = result_root / "summary" / "figures"
                expected = figure_dir / "mdd_velocity_displacement.png"
                self.assertTrue(expected.exists())
                self.assertEqual(outputs, [expected])
                self.assertEqual(stats["total_rows"], 2)
                self.assertEqual(stats["used_rows"], 2)
                self.assertEqual(stats["skipped_invalid_values"], 0)

    def test_mdd_plot_requires_columns(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = pathlib.Path(tmp_dir)
            with TemporaryResultRoot(tmp_path) as result_root:
                summary_csv = result_root / "mdd_summary.csv"
                summary_csv.write_text("run_dir,gt_disp_at_detection_mm,detected\n")
                with self.assertRaises(ValueError) as ctx:
                    plot_mdd_summary.main(
                        [
                            "--summary-csv",
                            str(summary_csv),
                            "--result-root",
                            str(result_root),
                        ]
                    )
                self.assertIn("velocity_mmps", str(ctx.exception))

    def test_mdd_plot_errors_on_header_only(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = pathlib.Path(tmp_dir)
            with TemporaryResultRoot(tmp_path) as result_root:
                summary_csv = result_root / "mdd_summary.csv"
                summary_csv.write_text(
                    "run_dir,velocity_mmps,gt_disp_at_detection_mm,detected\n"
                )
                with self.assertRaises(ValueError) as ctx:
                    plot_mdd_summary.main(
                        [
                            "--summary-csv",
                            str(summary_csv),
                            "--result-root",
                            str(result_root),
                        ]
                    )
                self.assertIn("No valid MDD records", str(ctx.exception))

    def test_mdd_plot_skips_invalid_rows(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = pathlib.Path(tmp_dir)
            with TemporaryResultRoot(tmp_path) as result_root:
                summary_csv = result_root / "mdd_summary.csv"
                summary_csv.write_text(
                    "run_dir,velocity_mmps,gt_disp_at_detection_mm,detected\n"
                    "sim_run_000,nan,10,True\n"
                    "sim_run_000,1.1,15,False\n"
                )
                outputs, stats = plot_mdd_summary.main(
                    [
                        "--summary-csv",
                        str(summary_csv),
                        "--result-root",
                        str(result_root),
                    ]
                )
                figure_dir = result_root / "summary" / "figures"
                expected = figure_dir / "mdd_velocity_displacement.png"
                self.assertTrue(expected.exists())
                self.assertEqual(stats["total_rows"], 2)
                self.assertEqual(stats["used_rows"], 1)
                self.assertEqual(stats["skipped_invalid_values"], 1)

    def test_mdd_plot_handles_mixed_validity_rows(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = pathlib.Path(tmp_dir)
            with TemporaryResultRoot(tmp_path) as result_root:
                summary_csv = result_root / "mdd_summary.csv"
                summary_csv.write_text(
                    "run_dir,velocity_mmps,gt_disp_at_detection_mm,detected\n"
                    "sim_run_000,1.0,11.0,True\n"
                    "sim_run_000,1.5,NaN,False\n"
                    "sim_run_000,0.8,9.5,True\n"
                )
                outputs, stats = plot_mdd_summary.main(
                    [
                        "--summary-csv",
                        str(summary_csv),
                        "--result-root",
                        str(result_root),
                    ]
                )
                figure_dir = result_root / "summary" / "figures"
                expected = figure_dir / "mdd_velocity_displacement.png"
                self.assertTrue(expected.exists())
                self.assertEqual(stats["total_rows"], 3)
                self.assertEqual(stats["used_rows"], 2)
                self.assertEqual(stats["skipped_invalid_values"], 1)

    def test_mdd_run_respects_verbose_flag(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = pathlib.Path(tmp_dir)
            with TemporaryResultRoot(tmp_path) as result_root:
                summary_csv = result_root / "mdd_summary.csv"
                summary_csv.write_text(
                    "run_dir,velocity_mmps,gt_disp_at_detection_mm,detected\n"
                    "sim_run_000,1.0,11.0,True\n"
                    "sim_run_000,1.2,12.0,False\n"
                )
                with mock.patch("builtins.print") as mock_print:
                    plot_mdd_summary.run(
                        summary_csv=summary_csv,
                        result_root=result_root,
                    )
                    mock_print.assert_not_called()
                with mock.patch("builtins.print") as mock_print:
                    plot_mdd_summary.run(
                        summary_csv=summary_csv,
                        result_root=result_root,
                        verbose=True,
                    )
                    mock_print.assert_called()

    def test_mdd_plot_groups_series_by_controlled_object(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = pathlib.Path(tmp_dir)
            with TemporaryResultRoot(tmp_path) as result_root:
                summary_csv = result_root / "mdd_summary.csv"
                summary_csv.write_text(
                    "run_dir,velocity_mmps,controlled_object,detected,gt_disp_at_detection_mm\n"
                    "sim_run_000,1.0,model_01,True,10.0\n"
                    "sim_run_000,1.2,model_02,False,12.5\n"
                    "sim_run_001,1.4,model_01,True,15.0\n"
                )
                captured = {}

                def capture_save(fig, output_path):
                    captured["path"] = pathlib.Path(output_path)
                    captured["labels"] = fig.axes[0].get_legend_handles_labels()[1]
                    return pathlib.Path(output_path)

                with mock.patch(
                    "analysis_script.plot_mdd_summary.plot_common.save_figure",
                    side_effect=capture_save,
                ):
                    outputs, _stats = plot_mdd_summary.main(
                        [
                            "--summary-csv",
                            str(summary_csv),
                            "--result-root",
                            str(result_root),
                        ]
                    )

                self.assertEqual(outputs, [result_root / "summary" / "figures" / "mdd_velocity_displacement.png"])
                self.assertEqual(
                    captured["labels"],
                    [
                        "model_01 / Detected",
                        "model_02 / Not detected",
                    ],
                )


class AblationPlotScriptTests(unittest.TestCase):
    def test_ablation_plot_writes_summary_figures(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = pathlib.Path(tmp_dir)
            with TemporaryResultRoot(tmp_path) as result_root:
                ablation_csv = result_root / "ablation_comparison.csv"
                ablation_csv.write_text(
                    "variant,label,run_dir,R_r,F_c,t_resp_s,beta_d\n"
                    "full,Full pipeline,sim_run_000,1.0,0.98,12.3,0.41\n"
                    "no_beta,No beta,sim_run_001,,0.92,NaN,0.38\n"
                )
                outputs, stats = plot_ablation_summary.main(
                    [
                        "--ablation-csv",
                        str(ablation_csv),
                        "--result-root",
                        str(result_root),
                    ]
                )
                figure_dir = result_root / "summary" / "figures"
                expected_files = [
                    figure_dir / "ablation_R_r.png",
                    figure_dir / "ablation_F_c.png",
                    figure_dir / "ablation_t_resp_s.png",
                    figure_dir / "ablation_beta_d.png",
                ]
                for expected in expected_files:
                    self.assertTrue(expected.exists())
                self.assertEqual(outputs, expected_files)
                self.assertEqual(stats["total_rows"], 2)
                self.assertEqual(stats["rows_with_valid_metrics"], 2)
                self.assertEqual(stats["skipped_rows"], 0)

    def test_ablation_plot_handles_header_whitespace_and_blank_rows(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = pathlib.Path(tmp_dir)
            header = (
                "\ufeffvariant , label , run_dir , R_r , F_c , t_resp_s , beta_d\n"
            )
            with TemporaryResultRoot(tmp_path) as result_root:
                ablation_csv = result_root / "ablation_comparison.csv"
                ablation_csv.write_text(
                    header
                    + "full,Full pipeline,sim_run_000,1.0,0.98,12.3,0.41\n"
                    + "\n"
                    + "no_beta,No beta,sim_run_001,0.9,0.92,11.8,0.38\n"
                )
                outputs, stats = plot_ablation_summary.main(
                    [
                        "--ablation-csv",
                        str(ablation_csv),
                        "--result-root",
                        str(result_root),
                    ]
                )
                figure_dir = result_root / "summary" / "figures"
                expected_files = [
                    figure_dir / "ablation_R_r.png",
                    figure_dir / "ablation_F_c.png",
                    figure_dir / "ablation_t_resp_s.png",
                    figure_dir / "ablation_beta_d.png",
                ]
                for expected in expected_files:
                    self.assertTrue(expected.exists())
                self.assertEqual(outputs, expected_files)
                self.assertEqual(stats["total_rows"], 2)
                self.assertEqual(stats["skipped_rows"], 0)

    def test_ablation_skips_rows_with_empty_variant_and_label(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = pathlib.Path(tmp_dir)
            with TemporaryResultRoot(tmp_path) as result_root:
                ablation_csv = result_root / "ablation_comparison.csv"
                ablation_csv.write_text(
                    "variant,label,run_dir,R_r,F_c,t_resp_s,beta_d\n"
                    "full,Full pipeline,sim_run_000,1.0,0.98,12.3,0.41\n"
                    ",,sim_run_999,0.9,0.9,12.0,0.39\n"
                    "no_beta,No beta,sim_run_001,0.92,0.92,11.8,0.38\n"
                )
                outputs, stats = plot_ablation_summary.main(
                    [
                        "--ablation-csv",
                        str(ablation_csv),
                        "--result-root",
                        str(result_root),
                    ]
                )
                figure_dir = result_root / "summary" / "figures"
                expected_files = [
                    figure_dir / "ablation_R_r.png",
                    figure_dir / "ablation_F_c.png",
                    figure_dir / "ablation_t_resp_s.png",
                    figure_dir / "ablation_beta_d.png",
                ]
                for expected in expected_files:
                    self.assertTrue(expected.exists())
                self.assertEqual(outputs, expected_files)
                self.assertEqual(stats["total_rows"], 3)
                self.assertEqual(stats["rows_with_valid_metrics"], 2)
                self.assertEqual(stats["skipped_rows"], 1)

    def test_ablation_run_verbose_flag_controls_print(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = pathlib.Path(tmp_dir)
            with TemporaryResultRoot(tmp_path) as result_root:
                ablation_csv = result_root / "ablation_comparison.csv"
                ablation_csv.write_text(
                    "variant,label,run_dir,R_r,F_c,t_resp_s,beta_d\n"
                    "full,Full pipeline,sim_run_000,1.0,0.98,12.3,0.41\n"
                )
                with mock.patch("builtins.print") as mock_print:
                    plot_ablation_summary.run(
                        ablation_csv=ablation_csv, result_root=result_root
                    )
                    mock_print.assert_not_called()
                with mock.patch("builtins.print") as mock_print:
                    plot_ablation_summary.run(
                        ablation_csv=ablation_csv,
                        result_root=result_root,
                        verbose=True,
                    )
                    mock_print.assert_called()

    def test_ablation_plot_groups_series_by_controlled_object(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = pathlib.Path(tmp_dir)
            with TemporaryResultRoot(tmp_path) as result_root:
                ablation_csv = result_root / "ablation_comparison.csv"
                ablation_csv.write_text(
                    "variant,label,controlled_object,R_r,F_c,t_resp_s,beta_d\n"
                    "full,Full pipeline,model_01,1.0,0.98,12.3,0.41\n"
                    "full,Full pipeline,model_02,0.9,0.88,13.3,0.31\n"
                    "no_beta,No beta,model_01,0.8,0.78,14.3,0.21\n"
                    "no_beta,No beta,model_02,0.7,0.68,15.3,0.11\n"
                )
                captured = {}

                def capture_save(fig, output_path):
                    target = pathlib.Path(output_path)
                    if target.name == "ablation_R_r.png":
                        captured["legend_labels"] = fig.axes[0].get_legend_handles_labels()[1]
                    return target

                with mock.patch(
                    "analysis_script.plot_ablation_summary.plot_common.save_figure",
                    side_effect=capture_save,
                ):
                    outputs, _stats = plot_ablation_summary.main(
                        [
                            "--ablation-csv",
                            str(ablation_csv),
                            "--result-root",
                            str(result_root),
                        ]
                    )

                self.assertIn(result_root / "summary" / "figures" / "ablation_R_r.png", outputs)
                self.assertEqual(captured["legend_labels"], ["model_01", "model_02"])


class SimSpatialOverlayScriptTests(unittest.TestCase):
    def test_spatial_overlay_writes_figure(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = pathlib.Path(tmp_dir)
            run_dir = tmp_path / "20260404" / "sim_run_overlay"
            truth_objects_dir = run_dir / "truth" / "objects"
            algorithm_dir = run_dir / "algorithm"
            meta_dir = run_dir / "meta"
            truth_objects_dir.mkdir(parents=True)
            algorithm_dir.mkdir(parents=True)
            meta_dir.mkdir(parents=True)

            object_csv = truth_objects_dir / "debris_a.csv"
            object_csv.write_text(
                "recorded_time_sec,model_name,frame_id,position_x,position_y,position_z,"
                "orientation_x,orientation_y,orientation_z,orientation_w\n"
                "10.0,debris_a,world,1.0,2.0,0.2,0,0,0,1\n"
                "10.1,debris_a,world,1.1,2.2,0.2,0,0,0,1\n"
                "10.2,debris_a,world,1.25,2.35,0.2,0,0,0,1\n",
                encoding="utf-8",
            )

            alignment = {
                "world_from_algorithm_transform": {
                    "pose": {
                        "position": {"x": 0.0, "y": 0.0, "z": 0.0},
                        "orientation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
                    }
                }
            }
            (meta_dir / "frame_alignment.json").write_text(
                json.dumps(alignment), encoding="utf-8"
            )

            persistent_records = [
                {
                    "header": {"stamp": {"sec": 10.2}},
                    "regions": [
                        {
                            "track_id": 7,
                            "confirmed": True,
                            "center": {"x": 1.2, "y": 2.3, "z": 0.25},
                            "bbox_min": {"x": 1.0, "y": 2.1, "z": 0.1},
                            "bbox_max": {"x": 1.4, "y": 2.5, "z": 0.4},
                            "mean_risk": 0.72,
                        }
                    ],
                }
            ]
            with (algorithm_dir / "persistent_risk_regions.jsonl").open(
                "w", encoding="utf-8"
            ) as f:
                for record in persistent_records:
                    json.dump(record, f)
                    f.write("\n")

            clusters = [
                {
                    "header": {"stamp": {"sec": 10.2}},
                    "clusters": [
                        {
                            "id": 2,
                            "significant": True,
                            "center": {"x": 1.15, "y": 2.25, "z": 0.2},
                            "disp_mean": [0.18, 0.06, 0.0],
                            "disp_norm": 0.19,
                        }
                    ],
                }
            ]
            with (algorithm_dir / "clusters.jsonl").open("w", encoding="utf-8") as f:
                for record in clusters:
                    json.dump(record, f)
                    f.write("\n")

            evidence = [
                {
                    "header": {"stamp": {"sec": 10.2}},
                    "evidences": [
                        {
                            "id": 1,
                            "position": {"x": 1.1, "y": 2.1, "z": 0.2},
                            "risk_score": 0.25,
                        },
                        {
                            "id": 2,
                            "position": {"x": 1.3, "y": 2.4, "z": 0.2},
                            "risk_score": 0.85,
                        },
                    ],
                }
            ]
            with (algorithm_dir / "risk_evidence.jsonl").open("w", encoding="utf-8") as f:
                for record in evidence:
                    json.dump(record, f)
                    f.write("\n")

            with TemporaryResultRoot(tmp_path) as result_root:
                outputs = plot_sim_spatial_overlay.main(["--run-dir", str(run_dir)])
                figure_dir = common.result_dir_for_run(run_dir) / "figures"
                expected = figure_dir / "sim_spatial_overlay.png"
                self.assertTrue(expected.exists())
                self.assertEqual(outputs, [expected])

    def test_cluster_displacement_vector_rotates_to_world_xy(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = pathlib.Path(tmp_dir)
            run_dir = tmp_path / "20260404" / "sim_run_rot"
            truth_objects_dir = run_dir / "truth" / "objects"
            algorithm_dir = run_dir / "algorithm"
            meta_dir = run_dir / "meta"
            truth_objects_dir.mkdir(parents=True)
            algorithm_dir.mkdir(parents=True)
            meta_dir.mkdir(parents=True)

            (truth_objects_dir / "debris_b.csv").write_text(
                "recorded_time_sec,model_name,frame_id,position_x,position_y,position_z,"
                "orientation_x,orientation_y,orientation_z,orientation_w\n"
                "10.0,debris_b,world,0.0,0.0,0.2,0,0,0,1\n"
                "10.1,debris_b,world,0.1,0.0,0.2,0,0,0,1\n",
                encoding="utf-8",
            )

            # 90-degree yaw: algorithm +x should map to world +y.
            alignment = {
                "world_from_algorithm_transform": {
                    "pose": {
                        "position": {"x": 0.0, "y": 0.0, "z": 0.0},
                        "orientation": {
                            "w": 0.7071067811865476,
                            "x": 0.0,
                            "y": 0.0,
                            "z": 0.7071067811865476,
                        },
                    }
                }
            }
            (meta_dir / "frame_alignment.json").write_text(
                json.dumps(alignment), encoding="utf-8"
            )

            with (algorithm_dir / "persistent_risk_regions.jsonl").open(
                "w", encoding="utf-8"
            ) as f:
                json.dump({"header": {"stamp": {"sec": 10.1}}, "regions": []}, f)
                f.write("\n")
            with (algorithm_dir / "clusters.jsonl").open("w", encoding="utf-8") as f:
                json.dump(
                    {
                        "header": {"stamp": {"sec": 10.1}},
                        "clusters": [
                            {
                                "id": 1,
                                "significant": True,
                                "center": {"x": 0.0, "y": 0.0, "z": 0.2},
                                "disp_mean": [1.0, 0.0, 0.0],
                                "disp_norm": 1.0,
                            }
                        ],
                    },
                    f,
                )
                f.write("\n")
            with (algorithm_dir / "risk_evidence.jsonl").open("w", encoding="utf-8") as f:
                json.dump({"header": {"stamp": {"sec": 10.1}}, "evidences": []}, f)
                f.write("\n")

            transform = common.build_world_from_algorithm_transform(alignment)
            dx, dy = plot_sim_spatial_overlay._vector_to_world_xy([1.0, 0.0, 0.0], transform)
            self.assertAlmostEqual(dx, 0.0, places=5)
            self.assertAlmostEqual(dy, 1.0, places=5)

    def test_spatial_overlay_skips_malformed_alignment_and_timestamps(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = pathlib.Path(tmp_dir)
            run_dir = tmp_path / "20260404" / "sim_run_malformed"
            truth_objects_dir = run_dir / "truth" / "objects"
            algorithm_dir = run_dir / "algorithm"
            meta_dir = run_dir / "meta"
            truth_objects_dir.mkdir(parents=True)
            algorithm_dir.mkdir(parents=True)
            meta_dir.mkdir(parents=True)

            (truth_objects_dir / "debris_c.csv").write_text(
                "recorded_time_sec,model_name,frame_id,position_x,position_y,position_z,"
                "orientation_x,orientation_y,orientation_z,orientation_w\n"
                "10.0,debris_c,world,1.0,1.0,0.2,0,0,0,1\n"
                "10.1,debris_c,world,1.1,1.2,0.2,0,0,0,1\n",
                encoding="utf-8",
            )

            # Malformed numeric alignment fields should not crash.
            bad_alignment = {
                "world_from_algorithm_transform": {
                    "pose": {
                        "position": {"x": "oops", "y": 0.0, "z": 0.0},
                        "orientation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
                    }
                }
            }
            (meta_dir / "frame_alignment.json").write_text(
                json.dumps(bad_alignment), encoding="utf-8"
            )

            # Malformed timestamps should be tolerated and skipped.
            with (algorithm_dir / "persistent_risk_regions.jsonl").open(
                "w", encoding="utf-8"
            ) as f:
                json.dump(
                    {
                        "header": {"stamp": {"sec": "bad"}},
                        "regions": [
                            {
                                "confirmed": True,
                                "bbox_min": {"x": 0.9, "y": 0.9, "z": 0.1},
                                "bbox_max": {"x": 1.2, "y": 1.3, "z": 0.3},
                            }
                        ],
                    },
                    f,
                )
                f.write("\n")
            with (algorithm_dir / "clusters.jsonl").open("w", encoding="utf-8") as f:
                json.dump(
                    {
                        "header": {"stamp": {"sec": "bad"}},
                        "clusters": [
                            {
                                "id": 3,
                                "significant": True,
                                "center": {"x": 1.0, "y": 1.0, "z": 0.2},
                                "disp_mean": [0.2, 0.0, 0.0],
                                "disp_norm": 0.2,
                            }
                        ],
                    },
                    f,
                )
                f.write("\n")
            with (algorithm_dir / "risk_evidence.jsonl").open("w", encoding="utf-8") as f:
                json.dump(
                    {
                        "header": {"stamp": {"sec": "bad"}},
                        "evidences": [
                            {
                                "position": {"x": 1.0, "y": 1.1, "z": 0.2},
                                "risk_score": 0.5,
                            }
                        ],
                    },
                    f,
                )
                f.write("\n")

            with TemporaryResultRoot(tmp_path) as result_root:
                outputs = plot_sim_spatial_overlay.main(["--run-dir", str(run_dir)])
                figure_dir = common.result_dir_for_run(run_dir) / "figures"
                expected = figure_dir / "sim_spatial_overlay.png"
                self.assertTrue(expected.exists())
                self.assertEqual(outputs, [expected])

    def test_spatial_overlay_annotates_controlled_object_names(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = pathlib.Path(tmp_dir)
            run_dir = tmp_path / "20260404" / "sim_run_overlay_labels"
            truth_objects_dir = run_dir / "truth" / "objects"
            algorithm_dir = run_dir / "algorithm"
            truth_objects_dir.mkdir(parents=True)
            algorithm_dir.mkdir(parents=True)

            (truth_objects_dir / "model_01.csv").write_text(
                "recorded_time_sec,model_name,frame_id,position_x,position_y,position_z,"
                "orientation_x,orientation_y,orientation_z,orientation_w\n"
                "10.0,model_01,world,0.0,0.0,0.0,0,0,0,1\n"
                "10.1,model_01,world,0.1,0.0,0.0,0,0,0,1\n",
                encoding="utf-8",
            )
            with (algorithm_dir / "persistent_risk_regions.jsonl").open("w", encoding="utf-8") as f:
                json.dump({"header": {"stamp": {"sec": 10.1}}, "regions": []}, f)
                f.write("\n")
            with (algorithm_dir / "clusters.jsonl").open("w", encoding="utf-8") as f:
                json.dump({"header": {"stamp": {"sec": 10.1}}, "clusters": []}, f)
                f.write("\n")
            with (algorithm_dir / "risk_evidence.jsonl").open("w", encoding="utf-8") as f:
                json.dump({"header": {"stamp": {"sec": 10.1}}, "evidences": []}, f)
                f.write("\n")

            captured = {}

            def capture_save(fig, output_path):
                captured["title"] = fig.axes[0].get_title()
                return pathlib.Path(output_path)

            with TemporaryResultRoot(tmp_path):
                with mock.patch(
                    "analysis_script.plot_sim_spatial_overlay.common.get_analysis_controlled_object_names",
                    return_value=["model_01", "model_02"],
                ), mock.patch(
                    "analysis_script.plot_sim_spatial_overlay.plot_common.save_figure",
                    side_effect=capture_save,
                ):
                    plot_sim_spatial_overlay.main(["--run-dir", str(run_dir)])

            self.assertIn("model_01, model_02", captured["title"])

    def test_spatial_overlay_falls_back_to_singular_controlled_object_name(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = pathlib.Path(tmp_dir)
            run_dir = tmp_path / "20260404" / "sim_run_overlay_single_label"
            truth_objects_dir = run_dir / "truth" / "objects"
            algorithm_dir = run_dir / "algorithm"
            truth_objects_dir.mkdir(parents=True)
            algorithm_dir.mkdir(parents=True)

            (truth_objects_dir / "model_01.csv").write_text(
                "recorded_time_sec,model_name,frame_id,position_x,position_y,position_z,"
                "orientation_x,orientation_y,orientation_z,orientation_w\n"
                "10.0,model_01,world,0.0,0.0,0.0,0,0,0,1\n"
                "10.1,model_01,world,0.1,0.0,0.0,0,0,0,1\n",
                encoding="utf-8",
            )
            with (algorithm_dir / "persistent_risk_regions.jsonl").open("w", encoding="utf-8") as f:
                json.dump({"header": {"stamp": {"sec": 10.1}}, "regions": []}, f)
                f.write("\n")
            with (algorithm_dir / "clusters.jsonl").open("w", encoding="utf-8") as f:
                json.dump({"header": {"stamp": {"sec": 10.1}}, "clusters": []}, f)
                f.write("\n")
            with (algorithm_dir / "risk_evidence.jsonl").open("w", encoding="utf-8") as f:
                json.dump({"header": {"stamp": {"sec": 10.1}}, "evidences": []}, f)
                f.write("\n")

            captured = {}

            def capture_save(fig, output_path):
                captured["title"] = fig.axes[0].get_title()
                return pathlib.Path(output_path)

            with TemporaryResultRoot(tmp_path):
                with mock.patch(
                    "analysis_script.plot_sim_spatial_overlay.common.get_analysis_controlled_object_names",
                    return_value=[],
                ), mock.patch(
                    "analysis_script.plot_sim_spatial_overlay.common.get_analysis_controlled_object_name",
                    return_value="model_01",
                ), mock.patch(
                    "analysis_script.plot_sim_spatial_overlay.plot_common.save_figure",
                    side_effect=capture_save,
                ):
                    plot_sim_spatial_overlay.main(["--run-dir", str(run_dir)])

            self.assertIn("model_01", captured["title"])


class SimTimelineScriptTests(unittest.TestCase):
    def test_timeline_helpers_select_single_track_and_preserve_missing_risk(self):
        track_events = [
            {
                "event_type": "frame_status",
                "track_id": 11,
                "header": {"stamp": {"sec": 1, "nsec": 0}},
                "state_name": "CANDIDATE",
                "mean_risk": 0.20,
                "confirmed": False,
            },
            {
                "event_type": "first_confirmed",
                "track_id": 11,
                "header": {"stamp": {"sec": 1, "nsec": 50000000}},
                "state_name": "CONFIRMED",
                "mean_risk": 0.50,
                "confirmed": True,
            },
            {
                "event_type": "first_confirmed",
                "track_id": 22,
                "header": {"stamp": {"sec": 1, "nsec": 100000000}},
                "state_name": "CONFIRMED",
                "mean_risk": 0.95,
                "confirmed": True,
            },
            {
                "event_type": "frame_status",
                "track_id": 11,
                "header": {"stamp": {"sec": 1, "nsec": 400000000}},
                "state_name": "FADING",
                "mean_risk": 0.70,
                "confirmed": False,
            },
        ]
        persistent_records = [
            {
                "header": {"stamp": {"sec": 1, "nsec": 300000000}},
                "regions": [
                    {"track_id": 11, "state_name": "FADING"},
                    {"track_id": 22, "state_name": "LOST", "mean_risk": 0.99},
                ],
            }
        ]

        track_id = plot_sim_timeline._select_primary_track_id(
            track_events, persistent_records
        )
        self.assertEqual(track_id, 11)

        samples = plot_sim_timeline._collect_track_samples(
            track_events, persistent_records, track_id
        )
        times, states, ema_risks = plot_sim_timeline._state_and_risk_series(
            samples, alpha=0.5
        )

        self.assertEqual(times, [1.0, 1.05, 1.3, 1.4])
        self.assertEqual(states, [0.0, 1.0, 2.0, 2.0])
        self.assertAlmostEqual(ema_risks[2], ema_risks[1], places=9)
        self.assertLess(max(ema_risks), 0.8)

    def test_timeline_helpers_select_track_id_per_controlled_object(self):
        track_events = [
            {
                "track_id": 11,
                "object": "model_01",
                "header": {"stamp": {"sec": 1.0}},
                "state_name": "CONFIRMED",
                "mean_risk": 0.40,
                "confirmed": True,
            },
            {
                "track_id": 22,
                "object": "model_02",
                "header": {"stamp": {"sec": 1.1}},
                "state_name": "CONFIRMED",
                "mean_risk": 0.80,
                "confirmed": True,
            },
            {
                "track_id": 33,
                "header": {"stamp": {"sec": 0.8}},
                "state_name": "CONFIRMED",
                "mean_risk": 0.95,
                "confirmed": True,
            },
        ]
        persistent_records = [
            {
                "header": {"stamp": {"sec": 1.2}},
                "regions": [
                    {
                        "track_id": 11,
                        "object": "model_01",
                        "state_name": "FADING",
                    },
                    {
                        "track_id": 22,
                        "object": "model_02",
                        "state_name": "FADING",
                    },
                ],
            }
        ]

        self.assertEqual(
            plot_sim_timeline._select_track_id_for_object(
                track_events, persistent_records, "model_01"
            ),
            11,
        )
        self.assertEqual(
            plot_sim_timeline._select_track_id_for_object(
                track_events, persistent_records, "model_02"
            ),
            22,
        )

    def test_sim_timeline_writes_figure(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = pathlib.Path(tmp_dir)
            run_dir = tmp_path / "20260404" / "sim_run_timeline"
            truth_objects_dir = run_dir / "truth" / "objects"
            algorithm_dir = run_dir / "algorithm"
            truth_objects_dir.mkdir(parents=True)
            algorithm_dir.mkdir(parents=True)

            (truth_objects_dir / "debris_timeline.csv").write_text(
                "recorded_time_sec,model_name,frame_id,position_x,position_y,position_z,"
                "orientation_x,orientation_y,orientation_z,orientation_w\n"
                "10.0,debris_timeline,world,0.0,0.0,0.0,0,0,0,1\n"
                "10.1,debris_timeline,world,0.1,0.0,0.0,0,0,0,1\n"
                "10.2,debris_timeline,world,0.25,0.0,0.0,0,0,0,1\n"
                "10.3,debris_timeline,world,0.45,0.0,0.0,0,0,0,1\n",
                encoding="utf-8",
            )

            events = [
                {
                    "event_type": "frame_status",
                    "track_id": 7,
                    "header": {"stamp": {"sec": 10.0}},
                    "state": 0,
                    "state_name": "CANDIDATE",
                    "mean_risk": 0.20,
                    "confirmed": False,
                },
                {
                    "event_type": "state_transition",
                    "track_id": 7,
                    "header": {"stamp": {"sec": 10.1}},
                    "state": 1,
                    "state_name": "CONFIRMED",
                    "mean_risk": 0.55,
                    "confirmed": True,
                },
                {
                    "event_type": "frame_status",
                    "track_id": 7,
                    "header": {"stamp": {"sec": 10.2}},
                    "state": 1,
                    "state_name": "CONFIRMED",
                    "mean_risk": 0.75,
                    "confirmed": True,
                },
                {
                    "event_type": "frame_status",
                    "track_id": 7,
                    "header": {"stamp": {"sec": 10.3}},
                    "state": 2,
                    "state_name": "FADING",
                    "mean_risk": 0.40,
                    "confirmed": False,
                },
            ]
            with (algorithm_dir / "persistent_track_events.jsonl").open(
                "w", encoding="utf-8"
            ) as f:
                for event in events:
                    json.dump(event, f)
                    f.write("\n")

            with TemporaryResultRoot(tmp_path) as result_root:
                outputs = plot_sim_timeline.main(["--run-dir", str(run_dir)])
                self.assertEqual(len(outputs), 1)
                self.assertEqual(outputs[0].name, "sim_timeline_partial.png")
                self.assertTrue(outputs[0].exists())

    def test_sim_timeline_emits_one_figure_per_controlled_object(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = pathlib.Path(tmp_dir)
            run_dir = tmp_path / "20260404" / "sim_run_timeline_multi"
            truth_objects_dir = run_dir / "truth" / "objects"
            algorithm_dir = run_dir / "algorithm"
            meta_dir = run_dir / "meta"
            truth_objects_dir.mkdir(parents=True)
            algorithm_dir.mkdir(parents=True)
            meta_dir.mkdir(parents=True)

            (meta_dir / "scenario_manifest.json").write_text(
                json.dumps(
                    {
                        "controls": [
                            {"controlled_object": "model_01"},
                            {"controlled_object": "model_02"},
                        ]
                    }
                ),
                encoding="utf-8",
            )

            for object_name, end_y in (("model_01", 0.2), ("model_02", 0.4)):
                (truth_objects_dir / f"{object_name}.csv").write_text(
                    "recorded_time_sec,model_name,frame_id,position_x,position_y,position_z,"
                    "orientation_x,orientation_y,orientation_z,orientation_w\n"
                    f"10.0,{object_name},world,0.0,0.0,0.0,0,0,0,1\n"
                    f"10.1,{object_name},world,0.0,{end_y / 2:.3f},0.0,0,0,0,1\n"
                    f"10.2,{object_name},world,0.0,{end_y:.3f},0.0,0,0,0,1\n",
                    encoding="utf-8",
                )

            events = [
                {
                    "event_type": "frame_status",
                    "track_id": 11,
                    "object": "model_01",
                    "header": {"stamp": {"sec": 10.0}},
                    "state_name": "CANDIDATE",
                    "mean_risk": 0.20,
                    "confirmed": False,
                },
                {
                    "event_type": "state_transition",
                    "track_id": 11,
                    "object": "model_01",
                    "header": {"stamp": {"sec": 10.1}},
                    "state_name": "CONFIRMED",
                    "mean_risk": 0.55,
                    "confirmed": True,
                },
                {
                    "event_type": "frame_status",
                    "track_id": 22,
                    "object": "model_02",
                    "header": {"stamp": {"sec": 10.05}},
                    "state_name": "CANDIDATE",
                    "mean_risk": 0.30,
                    "confirmed": False,
                },
                {
                    "event_type": "state_transition",
                    "track_id": 22,
                    "object": "model_02",
                    "header": {"stamp": {"sec": 10.15}},
                    "state_name": "CONFIRMED",
                    "mean_risk": 0.75,
                    "confirmed": True,
                },
            ]
            with (algorithm_dir / "persistent_track_events.jsonl").open(
                "w", encoding="utf-8"
            ) as f:
                for event in events:
                    json.dump(event, f)
                    f.write("\n")

            with TemporaryResultRoot(tmp_path) as result_root:
                captured_titles = {}
                original_save_figure = plot_common.save_figure

                def capture_save(fig, output_path):
                    target = pathlib.Path(output_path)
                    captured_titles[target.name] = [
                        ax.get_title() for ax in fig.axes
                    ]
                    return original_save_figure(fig, target)

                with mock.patch(
                    "analysis_script.plot_sim_timeline.plot_common.save_figure",
                    side_effect=capture_save,
                ):
                    outputs = plot_sim_timeline.main(["--run-dir", str(run_dir)])
                expected_names = [
                    "sim_timeline_partial_model_01.png",
                    "sim_timeline_partial_model_02.png",
                ]
                self.assertEqual([path.name for path in outputs], expected_names)
                for path in outputs:
                    self.assertTrue(path.exists())
                self.assertIn(
                    "Persistent track state (track_id=11)",
                    captured_titles["sim_timeline_partial_model_01.png"],
                )
                self.assertIn(
                    "Persistent track state (track_id=22)",
                    captured_titles["sim_timeline_partial_model_02.png"],
                )
