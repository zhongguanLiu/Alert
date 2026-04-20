#!/usr/bin/env python3
"""Plot epsilon_d distributions by injected velocity."""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import statistics
import sys
from collections import defaultdict

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

_SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import common
import plot_common as pc


DEFAULT_SUMMARY_CSV = common.result_root() / "summary" / "per_run_metrics.csv"
DEFAULT_PAPER_METRICS_ROOT = common.result_root()
DEFAULT_FIGURE_NAME = "epsilon_d_boxplot_by_velocity"

RUN_SERIES_LABEL = "run_mean"


def _finite_float(value) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _series_color(name: str) -> str:
    if name == "model_01":
        return pc.COLOR_MODEL_01
    if name == "model_02":
        return pc.COLOR_MODEL_02
    return pc.PRIMARY_BLUE


def _series_display_name(name: str) -> str:
    if name == "model_01":
        return "Object 1"
    if name == "model_02":
        return "Object 2"
    if name == RUN_SERIES_LABEL:
        return "Run-level mean"
    return name.replace("_", " ")


def _scenario_id_for_run(run_dir: pathlib.Path) -> str:
    scenario = common.load_json(pathlib.Path(run_dir) / "meta" / "scenario_manifest.json") or {}
    return str(scenario.get("scenario_id") or "").strip()


def collect_samples_from_summary_csv(
    csv_path: pathlib.Path, sample_level: str = "run"
) -> dict[str, dict[float, list[float]]]:
    """Collect velocity-grouped epsilon_d samples from a summary CSV."""
    csv_path = pathlib.Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Summary CSV not found: {csv_path}")

    if sample_level != "run":
        raise ValueError(
            "Object-level epsilon_d cannot be reconstructed reliably from the "
            "summary CSV because it stores run-level epsilon_d per row."
        )

    grouped: dict[float, list[float]] = defaultdict(list)
    seen_runs: set[tuple[str, str]] = set()
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            velocity = _finite_float(row.get("velocity_mmps"))
            epsilon = _finite_float(row.get("epsilon_d"))
            if velocity is None or epsilon is None:
                continue
            run_key = (
                str(row.get("scenario_id") or "").strip(),
                str(row.get("run_name") or "").strip(),
            )
            if run_key in seen_runs:
                continue
            seen_runs.add(run_key)
            grouped[velocity].append(epsilon)
    if not grouped:
        raise ValueError(f"No valid epsilon_d samples found in {csv_path}")
    return {RUN_SERIES_LABEL: dict(grouped)}


def _iter_paper_metrics_paths(root: pathlib.Path):
    root = pathlib.Path(root)
    if not root.is_dir():
        return
    for path in sorted(root.rglob("paper_metrics.json")):
        if path.is_file():
            yield path


def collect_samples_from_paper_metrics(
    paper_metrics_root: pathlib.Path,
    sample_level: str = "object",
    scenario_prefix: str | None = None,
) -> dict[str, dict[float, list[float]]]:
    """Collect velocity-grouped epsilon_d samples from paper_metrics.json files."""
    paper_metrics_root = pathlib.Path(paper_metrics_root)
    if not paper_metrics_root.is_dir():
        raise FileNotFoundError(f"Paper metrics root not found: {paper_metrics_root}")

    grouped: dict[str, dict[float, list[float]]] = defaultdict(lambda: defaultdict(list))
    seen_keys: set[tuple[str, str]] = set()

    for metrics_path in _iter_paper_metrics_paths(paper_metrics_root):
        try:
            payload = json.loads(metrics_path.read_text())
        except json.JSONDecodeError:
            continue

        run_dir_str = payload.get("run_dir")
        if not run_dir_str:
            continue
        run_dir = pathlib.Path(run_dir_str)
        scenario_id = _scenario_id_for_run(run_dir)
        if scenario_prefix and not scenario_id.startswith(scenario_prefix):
            continue

        epsilon_block = payload.get("epsilon_d", {})

        if sample_level == "run":
            epsilon = _finite_float(epsilon_block.get("epsilon_d"))
            velocity_mps = common.get_injection_velocity(run_dir)
            velocity = velocity_mps * 1000.0 if velocity_mps is not None else None
            if velocity is None or epsilon is None:
                continue
            run_key = (str(run_dir.resolve()), RUN_SERIES_LABEL)
            if run_key in seen_keys:
                continue
            seen_keys.add(run_key)
            grouped[RUN_SERIES_LABEL][velocity].append(epsilon)
            continue

        if sample_level != "object":
            raise ValueError(f"Unsupported sample level: {sample_level}")

        per_object = epsilon_block.get("per_object", {})
        if not isinstance(per_object, dict):
            continue
        for object_name, info in per_object.items():
            if not isinstance(info, dict):
                continue
            epsilon = _finite_float(info.get("epsilon_d"))
            velocity_mps = common.get_injection_velocity(run_dir, object_name)
            if velocity_mps is None:
                velocity_mps = common.get_injection_velocity(run_dir)
            velocity = velocity_mps * 1000.0 if velocity_mps is not None else None
            if velocity is None or epsilon is None:
                continue
            sample_key = (str(run_dir.resolve()), str(object_name))
            if sample_key in seen_keys:
                continue
            seen_keys.add(sample_key)
            grouped[str(object_name)][velocity].append(epsilon)

    compact = {series: dict(per_vel) for series, per_vel in grouped.items() if per_vel}
    if not compact:
        raise ValueError(
            f"No valid epsilon_d samples found under metrics root: {paper_metrics_root}"
        )
    return compact


def compute_group_stats(
    grouped: dict[str, dict[float, list[float]]]
) -> list[dict[str, float | str]]:
    """Compute descriptive statistics for each series/velocity group."""
    rows: list[dict[str, float | str]] = []
    for series_name in sorted(grouped):
        for velocity in sorted(grouped[series_name]):
            samples = [float(v) for v in grouped[series_name][velocity] if math.isfinite(float(v))]
            if not samples:
                continue
            q1_signed, q3_signed = np.percentile(samples, [25, 75])
            n = len(samples)
            pos_count = sum(1 for v in samples if v > 0.0)
            neg_count = sum(1 for v in samples if v < 0.0)
            zero_count = n - pos_count - neg_count
            rows.append(
                {
                    "controlled_object": series_name,
                    "velocity_mmps": velocity,
                    "n_samples": n,
                    "mean_signed": statistics.mean(samples),
                    "median_signed": statistics.median(samples),
                    "q1_signed": float(q1_signed),
                    "q3_signed": float(q3_signed),
                    "positive_fraction": pos_count / n,
                    "negative_fraction": neg_count / n,
                    "zero_fraction": zero_count / n,
                }
            )
    return rows


def write_stats_csv(
    stats_rows: list[dict[str, float | str]], output_path: pathlib.Path
) -> pathlib.Path:
    """Write per-velocity descriptive statistics to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "controlled_object",
        "velocity_mmps",
        "n_samples",
        "mean_signed",
        "median_signed",
        "q1_signed",
        "q3_signed",
        "positive_fraction",
        "negative_fraction",
        "zero_fraction",
    ]
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in stats_rows:
            writer.writerow(row)
    return output_path


def _add_jittered_scatter(
    ax,
    positions: list[float],
    series_data: list[list[float]],
    color: str,
    jitter_half_width: float,
):
    """Overlay lightly jittered raw samples on top of each box."""
    rng = np.random.default_rng(0)
    for pos, samples in zip(positions, series_data):
        if not samples:
            continue
        jitter = rng.uniform(-jitter_half_width, jitter_half_width, size=len(samples))
        xs = np.full(len(samples), pos, dtype=float) + jitter
        ys = np.asarray(samples, dtype=float)
        ax.scatter(
            xs,
            ys,
            s=10,
            c=color,
            alpha=0.42,
            edgecolors="none",
            zorder=3,
        )


def build_boxplot_figure(
    grouped: dict[str, dict[float, list[float]]], sample_level: str
):
    """Build the epsilon_d boxplot figure and return (fig, ax)."""
    if not grouped:
        raise ValueError("No grouped epsilon_d samples available for plotting.")

    pc.setup_plot_style()

    series_names = sorted(grouped)
    velocities = sorted({vel for per_vel in grouped.values() for vel in per_vel})
    if not velocities:
        raise ValueError("No velocity groups available for plotting.")

    fig, ax = plt.subplots(1, 1, figsize=(6.2, 2.55))
    base_positions = np.arange(len(velocities), dtype=float) + 1.0

    if len(series_names) == 1:
        series_name = series_names[0]
        data = [grouped[series_name][v] for v in velocities]
        bp = ax.boxplot(
            data,
            positions=base_positions,
            widths=0.42,
            patch_artist=True,
            medianprops={"color": "black", "linewidth": 1.0},
            whiskerprops={"linewidth": 0.75, "color": "#555555"},
            capprops={"linewidth": 0.75, "color": "#555555"},
            boxprops={"linewidth": 0.75},
            flierprops={
                "marker": "o",
                "markersize": 2.2,
                "markerfacecolor": "#666666",
                "markeredgecolor": "#666666",
                "alpha": 0.45,
            },
        )
        color = _series_color(series_name)
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.82)
        _add_jittered_scatter(ax, list(base_positions), data, color, jitter_half_width=0.045)
    else:
        offsets = np.linspace(-0.16, 0.16, len(series_names))
        legend_handles = []
        for series_name, offset in zip(series_names, offsets):
            series_data = []
            positions = []
            for base_position, velocity in zip(base_positions, velocities):
                samples = grouped[series_name].get(velocity, [])
                if not samples:
                    continue
                series_data.append(samples)
                positions.append(base_position + offset)
            if not series_data:
                continue
            color = _series_color(series_name)
            bp = ax.boxplot(
                series_data,
                positions=positions,
                widths=0.24,
                patch_artist=True,
                medianprops={"color": "black", "linewidth": 1.0},
                whiskerprops={"linewidth": 0.75, "color": "#555555"},
                capprops={"linewidth": 0.75, "color": "#555555"},
                boxprops={"linewidth": 0.75},
                flierprops={
                    "marker": "o",
                    "markersize": 2.2,
                    "markerfacecolor": "#666666",
                    "markeredgecolor": "#666666",
                    "alpha": 0.45,
                },
            )
            for patch in bp["boxes"]:
                patch.set_facecolor(color)
                patch.set_alpha(0.82)
            _add_jittered_scatter(ax, positions, series_data, color, jitter_half_width=0.028)
            legend_handles.append(
                mpatches.Patch(color=color, alpha=0.85, label=_series_display_name(series_name))
            )
        if legend_handles:
            ax.legend(
                handles=legend_handles,
                loc="upper center",
                bbox_to_anchor=(0.5, 1.02),
                ncol=len(legend_handles),
                frameon=False,
                handlelength=1.0,
                handletextpad=0.4,
                columnspacing=0.8,
                borderaxespad=0.2,
            )

    ax.set_xticks(base_positions)
    ax.set_xticklabels([f"{v:.1f}" for v in velocities])
    ax.set_xlabel("Injection velocity (mm/s)")
    ax.set_ylabel(r"$\epsilon_d$")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.32, alpha=0.35)
    ax.tick_params(axis="both", which="major", length=2.5, width=0.5, pad=1.5)
    ax.margins(x=0.04)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.6)
    ax.spines["bottom"].set_linewidth(0.6)

    fig.tight_layout(pad=0.2)
    return fig, ax


def plot_boxplots(
    grouped: dict[str, dict[float, list[float]]],
    sample_level: str,
    out_dir: pathlib.Path,
    figure_name: str = DEFAULT_FIGURE_NAME,
) -> list[pathlib.Path]:
    """Render and save the epsilon_d boxplots."""
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, _ = build_boxplot_figure(grouped, sample_level)
    pc.save_figure(fig, out_dir, figure_name)
    plt.close(fig)

    return [
        out_dir / f"{figure_name}.png",
        out_dir / f"{figure_name}.pdf",
    ]


def _parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Plot epsilon_d boxplots grouped by injection velocity."
    )
    parser.add_argument(
        "--source",
        choices=("auto", "summary", "paper-metrics"),
        default="auto",
        help="Preferred data source for epsilon_d samples.",
    )
    parser.add_argument(
        "--summary-csv",
        type=str,
        default=str(DEFAULT_SUMMARY_CSV),
        help="Path to per_run_metrics.csv for run-level plotting.",
    )
    parser.add_argument(
        "--paper-metrics-root",
        type=str,
        default=str(DEFAULT_PAPER_METRICS_ROOT),
        help="Root directory containing paper_metrics.json files.",
    )
    parser.add_argument(
        "--sample-level",
        choices=("run", "object"),
        default="object",
        help="Sample granularity for the boxplot.",
    )
    parser.add_argument(
        "--scenario-prefix",
        type=str,
        default="sim_main_",
        help="Only use runs whose scenario_id starts with this prefix. Empty string disables filtering.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Directory for output figures/statistics CSV.",
    )
    return parser.parse_args(argv)


def _load_grouped_samples(args) -> dict[str, dict[float, list[float]]]:
    summary_csv = pathlib.Path(args.summary_csv)
    paper_metrics_root = pathlib.Path(args.paper_metrics_root)
    scenario_prefix = args.scenario_prefix or None

    if args.source in ("summary", "auto") and args.sample_level == "run" and summary_csv.is_file():
        return collect_samples_from_summary_csv(summary_csv, sample_level="run")

    if args.source == "summary":
        raise ValueError(
            "Summary CSV source only supports run-level epsilon_d. "
            "Use --source paper-metrics for object-level boxplots."
        )

    if args.source in ("paper-metrics", "auto"):
        return collect_samples_from_paper_metrics(
            paper_metrics_root,
            sample_level=args.sample_level,
            scenario_prefix=scenario_prefix,
        )

    raise ValueError("Unable to load epsilon_d samples with the requested settings.")


def main(argv: list[str] | None = None) -> list[pathlib.Path]:
    args = _parse_args(argv)
    grouped = _load_grouped_samples(args)

    out_dir = (
        pathlib.Path(args.out_dir)
        if args.out_dir
        else common.result_root() / "summary" / "figures"
    )
    figure_outputs = plot_boxplots(
        grouped,
        sample_level=args.sample_level,
        out_dir=out_dir,
        figure_name=DEFAULT_FIGURE_NAME,
    )
    stats_rows = compute_group_stats(grouped)
    stats_path = write_stats_csv(
        stats_rows, out_dir / f"{DEFAULT_FIGURE_NAME}_stats.csv"
    )
    outputs = [*figure_outputs, stats_path]
    for path in outputs:
        print(f"Saved: {path}")
    return outputs


if __name__ == "__main__":
    main()
