"""Test public-repo-friendly path helpers and plotting compatibility."""

from __future__ import annotations

import atexit
import pathlib
import tempfile
import unittest
from unittest import mock

_config_dir_obj = tempfile.TemporaryDirectory(prefix="matplotlib-config-")
atexit.register(_config_dir_obj.cleanup)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import analysis_script.common as common
import analysis_script.plot_common as plot_common


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


class PublicRepoPathTests(unittest.TestCase):
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


class PlotCommonCompatibilityTests(unittest.TestCase):
    def test_figure_dir_for_run_creates_figures_directory(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = pathlib.Path(tmp_dir)
            run_dir = tmp_path / "20260404" / "sim_run_000"
            run_dir.mkdir(parents=True)
            with TemporaryResultRoot(tmp_path):
                figure_dir = plot_common.figure_dir_for_run(run_dir)
                self.assertTrue(figure_dir.exists())
                self.assertEqual(figure_dir.name, "figures")
                self.assertEqual(figure_dir.parent, common.result_dir_for_run(run_dir))

    def test_summary_figure_dir_creates_shared_output(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = pathlib.Path(tmp_dir)
            with TemporaryResultRoot(tmp_path) as result_root:
                summary = plot_common.summary_figure_dir()
                self.assertTrue(summary.exists())
                self.assertEqual(summary, result_root / "summary" / "figures")

    def test_save_figure_writes_single_file(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = pathlib.Path(tmp_dir) / "figure.png"
            fig = plt.figure()
            plt.plot([0, 1], [0, 1])
            plot_common.save_figure(fig, output_path)
            self.assertTrue(output_path.is_file())
            self.assertGreater(output_path.stat().st_size, 0)
            plt.close(fig)

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
        self.assertEqual(matplotlib.rcParams.get("grid.linestyle"), original["grid.linestyle"])
        self.assertEqual(matplotlib.rcParams.get("lines.linewidth"), original["lines.linewidth"])
        self.assertEqual(matplotlib.rcParams.get("figure.dpi"), original["figure.dpi"])
