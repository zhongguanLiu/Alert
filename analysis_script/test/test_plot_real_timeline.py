"""Test the real timeline plot script."""

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
import analysis_script.plot_real_timeline as plot_real_timeline


class PlotRealTimelineTests(unittest.TestCase):
    def test_default_paths_are_public_repo_relative(self):
        self.assertEqual(
            pathlib.Path(plot_real_timeline.DATA_DIR),
            common.default_real_run_dir("real_run_001"),
        )
        self.assertEqual(
            pathlib.Path(plot_real_timeline.OUT_DIR),
            common.real_figure_dir(),
        )
        pdf_path, png_path = common.real_timeline_output_paths("real_run_001")
        self.assertEqual(pathlib.Path(plot_real_timeline.OUT_FILE), pdf_path)
        self.assertEqual(pathlib.Path(plot_real_timeline.OUT_FILE_PNG), png_path)

    def _write_jsonl(self, path: pathlib.Path, records) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for record in records:
                json.dump(record, f)
                f.write("\n")

    def test_panel_a_legend_uses_two_side_by_side_vertical_columns(self):
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
                            "id": 1,
                            "disp_mean": [0.010, 0.0, 0.0],
                            "support_count": 5,
                            "center": {"x": 0.0, "y": 0.0, "z": 0.1},
                        }
                    ],
                },
                {
                    "header": {"stamp": {"sec": 10.5}},
                    "recorded_at": {"sec": wall_clock_t0 + 1},
                    "clusters": [
                        {
                            "id": 2,
                            "disp_mean": [0.020, 0.0, 0.0],
                            "support_count": 6,
                            "center": {"x": 0.0, "y": 0.0, "z": 0.1},
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
                {"header": {"stamp": {"sec": 10.5}}, "regions": [{}, {}, {}]},
            ]
            structure_motions = [
                {
                    "header": {"stamp": {"sec": 10.25}},
                    "motions": [
                        {
                            "id": 3,
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

            with mock.patch.object(plot_real_timeline, "DATA_DIR", str(data_dir)), \
                mock.patch.object(plot_real_timeline, "OUT_DIR", str(out_dir)), \
                mock.patch.object(
                    plot_real_timeline,
                    "OUT_FILE",
                    str(out_dir / "run001_timeline.pdf"),
                ), \
                mock.patch.object(
                    plot_real_timeline,
                    "OUT_FILE_PNG",
                    str(out_dir / "run001_timeline.png"),
                ):
                plot_real_timeline.main()

            fig = plt.gcf()
            fig.canvas.draw()

            legend_texts = [text for text in fig.texts if text.get_text()]
            self.assertEqual(
                [text.get_text() for text in legend_texts],
                ["MAX clusters", "MAX reacquired"],
            )
            self.assertEqual([text.get_rotation() for text in legend_texts], [90.0, 90.0])
            self.assertAlmostEqual(
                legend_texts[0].get_position()[1],
                legend_texts[1].get_position()[1],
                places=3,
            )
            self.assertLess(
                legend_texts[0].get_position()[0],
                legend_texts[1].get_position()[0],
            )

            plt.close(fig)


if __name__ == "__main__":
    unittest.main()
