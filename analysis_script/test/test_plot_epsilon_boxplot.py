"""Test the epsilon_d boxplot script."""

from __future__ import annotations

import atexit
import csv
import json
import os
import pathlib
import tempfile
import unittest

_config_dir_obj = tempfile.TemporaryDirectory(prefix="matplotlib-config-")
atexit.register(_config_dir_obj.cleanup)
os.environ.setdefault("MPLCONFIGDIR", _config_dir_obj.name)
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg")
from matplotlib.collections import PathCollection

import analysis_script.plot_epsilon_boxplot as plot_epsilon_boxplot


class PlotEpsilonBoxplotTests(unittest.TestCase):
    def test_collect_run_level_samples_deduplicates_per_run_rows(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = pathlib.Path(tmp_dir)
            csv_path = tmp_path / "per_run_metrics.csv"
            rows = [
                {
                    "run_name": "sim_run_000",
                    "scenario_id": "sim_main_block02_ypos_0p5mmps_r01",
                    "controlled_object": "model_01",
                    "velocity_mmps": "0.5",
                    "epsilon_d": "0.10",
                },
                {
                    "run_name": "sim_run_000",
                    "scenario_id": "sim_main_block02_ypos_0p5mmps_r01",
                    "controlled_object": "model_02",
                    "velocity_mmps": "0.5",
                    "epsilon_d": "0.10",
                },
                {
                    "run_name": "sim_run_001",
                    "scenario_id": "sim_main_block02_ypos_0p5mmps_r02",
                    "controlled_object": "model_01",
                    "velocity_mmps": "0.5",
                    "epsilon_d": "-0.20",
                },
                {
                    "run_name": "sim_run_001",
                    "scenario_id": "sim_main_block02_ypos_0p5mmps_r02",
                    "controlled_object": "model_02",
                    "velocity_mmps": "0.5",
                    "epsilon_d": "-0.20",
                },
                {
                    "run_name": "sim_run_002",
                    "scenario_id": "sim_main_block02_ypos_1p0mmps_r01",
                    "controlled_object": "model_01",
                    "velocity_mmps": "1.0",
                    "epsilon_d": "0.30",
                },
                {
                    "run_name": "sim_run_002",
                    "scenario_id": "sim_main_block02_ypos_1p0mmps_r01",
                    "controlled_object": "model_02",
                    "velocity_mmps": "1.0",
                    "epsilon_d": "0.30",
                },
            ]
            with csv_path.open("w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "run_name",
                        "scenario_id",
                        "controlled_object",
                        "velocity_mmps",
                        "epsilon_d",
                    ],
                )
                writer.writeheader()
                writer.writerows(rows)

            grouped = plot_epsilon_boxplot.collect_samples_from_summary_csv(
                csv_path, sample_level="run"
            )

            self.assertEqual(sorted(grouped.keys()), ["run_mean"])
            self.assertEqual(sorted(grouped["run_mean"].keys()), [0.5, 1.0])
            self.assertEqual(grouped["run_mean"][0.5], [0.10, -0.20])
            self.assertEqual(grouped["run_mean"][1.0], [0.30])

    def test_main_object_level_writes_figure_from_paper_metrics(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = pathlib.Path(tmp_dir)
            result_root = tmp_path / "result"
            result_root.mkdir()
            output_root = tmp_path / "output"
            run_dir = output_root / "20260415" / "sim_run_000"
            (run_dir / "meta").mkdir(parents=True)

            scenario_manifest = {
                "scenario_id": "sim_main_block02_ypos_0p5mmps_r01",
                "controls": [
                    {
                        "controlled_object": "model_01",
                        "scenario_id": "sim_main_block02_ypos_0p5mmps_r01",
                        "velocity": {"linear_mps": {"x": 0.0, "y": 0.0005, "z": 0.0}},
                    },
                    {
                        "controlled_object": "model_02",
                        "scenario_id": "sim_main_block01_xpos_0p5mmps_r01",
                        "velocity": {"linear_mps": {"x": 0.0005, "y": 0.0, "z": 0.0}},
                    },
                ],
            }
            (run_dir / "meta" / "scenario_manifest.json").write_text(
                json.dumps(scenario_manifest)
            )

            metrics_dir = result_root / "20260415" / "sim_run_000"
            metrics_dir.mkdir(parents=True)
            (metrics_dir / "paper_metrics.json").write_text(
                json.dumps(
                    {
                        "run_dir": str(run_dir),
                        "epsilon_d": {
                            "epsilon_d": 0.075,
                            "N_samples": 2,
                            "per_object": {
                                "model_01": {"epsilon_d": -0.10},
                                "model_02": {"epsilon_d": 0.25},
                            },
                        },
                    }
                )
            )

            out_dir = tmp_path / "figures"
            outputs = plot_epsilon_boxplot.main(
                [
                    "--paper-metrics-root",
                    str(result_root),
                    "--sample-level",
                    "object",
                    "--out-dir",
                    str(out_dir),
                ]
            )

            expected_png = out_dir / "epsilon_d_boxplot_by_velocity.png"
            expected_pdf = out_dir / "epsilon_d_boxplot_by_velocity.pdf"
            expected_stats = out_dir / "epsilon_d_boxplot_by_velocity_stats.csv"
            self.assertIn(expected_png, outputs)
            self.assertIn(expected_pdf, outputs)
            self.assertIn(expected_stats, outputs)
            self.assertTrue(expected_png.is_file())
            self.assertTrue(expected_pdf.is_file())
            self.assertTrue(expected_stats.is_file())

            with expected_stats.open(newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 2)
            self.assertEqual(
                sorted(row["controlled_object"] for row in rows),
                ["model_01", "model_02"],
            )
            self.assertIn("median_signed", rows[0])
            self.assertNotIn("mean_abs", rows[0])

    def test_build_boxplot_figure_adds_jittered_scatter_points(self):
        grouped = {
            "model_01": {0.5: [-0.20, -0.10, -0.05], 1.0: [-0.18, -0.12, -0.08]},
            "model_02": {0.5: [0.05, 0.12, 0.20], 1.0: [-0.02, 0.10, 0.24]},
        }

        fig, ax = plot_epsilon_boxplot.build_boxplot_figure(
            grouped, sample_level="object"
        )
        try:
            scatter_collections = [
                artist for artist in ax.collections if isinstance(artist, PathCollection)
            ]
            self.assertGreaterEqual(len(scatter_collections), 2)
        finally:
            import matplotlib.pyplot as plt

            plt.close(fig)


if __name__ == "__main__":
    unittest.main()
