"""Test the text-free real-run timeline plot."""

from __future__ import annotations

import atexit
import json
import os
import pathlib
import tempfile
import unittest
from unittest import mock

_config_dir_obj = tempfile.TemporaryDirectory(prefix="matplotlib-config-")
atexit.register(_config_dir_obj.cleanup)
os.environ.setdefault("MPLCONFIGDIR", _config_dir_obj.name)
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import analysis_script.common as common
import analysis_script.plot_run004_timeline as plot_run004_timeline
import analysis_script.plot_run004_timeline_noword as plot_run004_timeline_noword


class PlotRun004TimelineNoWordTests(unittest.TestCase):
    def test_default_paths_are_public_repo_relative(self):
        expected_data_dir = common.default_real_run_dir("real_run_000")
        expected_pdf, _ = common.real_timeline_output_paths("real_run_000")
        self.assertEqual(
            pathlib.Path(plot_run004_timeline.DATA_DIR),
            expected_data_dir,
        )
        self.assertEqual(
            pathlib.Path(plot_run004_timeline.OUT_DIR),
            common.real_figure_dir(),
        )
        self.assertEqual(
            pathlib.Path(plot_run004_timeline.OUT_FILE),
            expected_pdf,
        )
        self.assertEqual(
            pathlib.Path(plot_run004_timeline_noword.DATA_DIR),
            expected_data_dir,
        )
        self.assertEqual(
            pathlib.Path(plot_run004_timeline_noword.OUT_DIR),
            common.real_figure_dir(),
        )
        self.assertEqual(
            pathlib.Path(plot_run004_timeline_noword.OUT_FILE),
            common.real_figure_dir() / "real_run_000_timeline_noword.pdf",
        )

    def _write_jsonl(self, path: pathlib.Path, records) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for record in records:
                json.dump(record, f)
                f.write("\n")

    def test_main_creates_figure_without_text_artists(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = pathlib.Path(tmp_dir)
            data_dir = tmp_path / "algorithm"
            out_dir = tmp_path / "result"
            data_dir.mkdir()
            out_dir.mkdir()

            wall_clock_t0 = 1713495600
            cluster_frames = [
                {
                    "header": {"stamp": {"sec": 10.0}},
                    "recorded_at": {"sec": wall_clock_t0},
                    "clusters": [
                        {
                            "disp_mean": [0.010, 0.0, 0.0],
                            "support_count": 5,
                            "center": {"z": 0.05},
                        },
                        {
                            "disp_mean": [0.0, 0.012, 0.0],
                            "support_count": 3,
                            "center": {"z": 0.30},
                        },
                    ],
                },
                {
                    "header": {"stamp": {"sec": 10.5}},
                    "recorded_at": {"sec": wall_clock_t0 + 1},
                    "clusters": [
                        {
                            "disp_mean": [0.020, 0.0, 0.0],
                            "support_count": 6,
                            "center": {"z": 1.10},
                        }
                    ],
                },
            ]
            anchor_states = [
                {
                    "header": {"stamp": {"sec": 10.0}},
                    "recorded_at": {"sec": wall_clock_t0},
                    "anchors": [
                        {
                            "reacquired": True,
                            "significant": True,
                            "disp_norm": 0.080,
                        }
                    ],
                },
                {
                    "header": {"stamp": {"sec": 10.5}},
                    "recorded_at": {"sec": wall_clock_t0 + 1},
                    "anchors": [
                        {
                            "reacquired": True,
                            "significant": True,
                            "disp_norm": 0.120,
                        }
                    ],
                },
            ]
            risk_regions = [
                {"header": {"stamp": {"sec": 10.0}}, "regions": [{}, {}]},
                {"header": {"stamp": {"sec": 10.5}}, "regions": [{}, {}, {}, {}]},
            ]
            structure_motions = [
                {
                    "header": {"stamp": {"sec": 10.25}},
                    "motions": [
                        {
                            "motion": {"x": 0.050, "y": 0.0, "z": 0.0},
                            "confidence": 0.9,
                        }
                    ],
                }
            ]

            self._write_jsonl(data_dir / "clusters.jsonl", cluster_frames)
            self._write_jsonl(data_dir / "anchor_states.jsonl", anchor_states)
            self._write_jsonl(
                data_dir / "persistent_risk_regions.jsonl", risk_regions
            )
            self._write_jsonl(data_dir / "structure_motions.jsonl", structure_motions)

            output_path = out_dir / "run004_timeline_noword.pdf"
            with mock.patch.object(
                plot_run004_timeline_noword, "DATA_DIR", str(data_dir)
            ), mock.patch.object(
                plot_run004_timeline_noword, "OUT_DIR", str(out_dir)
            ), mock.patch.object(
                plot_run004_timeline_noword, "OUT_FILE", str(output_path)
            ):
                plot_run004_timeline_noword.main()

            self.assertTrue(output_path.exists())
            self.assertGreater(output_path.stat().st_size, 0)

            fig = plt.gcf()
            fig.canvas.draw()

            for ax in fig.axes:
                self.assertIsNone(ax.get_legend())
                self.assertEqual(ax.get_xlabel(), "")
                self.assertEqual(ax.get_ylabel(), "")
                self.assertEqual(ax.get_title(), "")
                self.assertEqual(list(ax.texts), [])

                for label in ax.get_xticklabels() + ax.get_yticklabels():
                    self.assertFalse(label.get_visible())

            plt.close(fig)


if __name__ == "__main__":
    unittest.main()
