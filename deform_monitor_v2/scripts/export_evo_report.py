#!/usr/bin/env python3

import argparse
import datetime as dt
from dataclasses import dataclass
import os
import pathlib
import shlex
import shutil
import re
import subprocess
import tempfile


@dataclass(frozen=True)
class MetricSummary:
    max_value: float
    mean: float
    median: float
    min_value: float
    rmse: float
    sse: float
    std: float


@dataclass(frozen=True)
class TumSummary:
    sample_count: int
    start_time: float
    end_time: float
    duration_sec: float


@dataclass(frozen=True)
class EvoReportOutputs:
    output_dir: pathlib.Path
    report_txt: pathlib.Path
    commands_txt: pathlib.Path
    ape_stdout_txt: pathlib.Path
    rpe_stdout_txt: pathlib.Path
    traj_plot_png: pathlib.Path
    ape_plot_png: pathlib.Path
    rpe_plot_png: pathlib.Path


def load_tum_summary(tum_path):
    tum_path = pathlib.Path(tum_path)
    lines = []
    for line in tum_path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        lines.append(stripped)
    timestamps = [float(line.split()[0]) for line in lines]
    if not timestamps:
        raise ValueError(f"Empty TUM file: {tum_path}")
    return TumSummary(
        sample_count=len(timestamps),
        start_time=timestamps[0],
        end_time=timestamps[-1],
        duration_sec=timestamps[-1] - timestamps[0],
    )


def parse_metric_summary(stdout_text):
    value_by_name = {}
    number_pattern = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
    pattern = re.compile(rf"^\s*(max|mean|median|min|rmse|sse|std)\s+({number_pattern})\s*$", re.IGNORECASE)
    for line in stdout_text.splitlines():
        match = pattern.match(line)
        if match:
            key = match.group(1).lower()
            value_by_name[key] = float(match.group(2))

    missing = [name for name in ("max", "mean", "median", "min", "rmse", "sse", "std") if name not in value_by_name]
    if missing:
        raise ValueError(f"Missing metric fields: {', '.join(missing)}")

    return MetricSummary(
        max_value=value_by_name["max"],
        mean=value_by_name["mean"],
        median=value_by_name["median"],
        min_value=value_by_name["min"],
        rmse=value_by_name["rmse"],
        sse=value_by_name["sse"],
        std=value_by_name["std"],
    )


def _format_metric_block(title, summary, unit, sse_unit=None):
    sse_unit = sse_unit or unit
    return (
        f"{title}\n"
        f"最大值：{summary.max_value:.6f} {unit}\n"
        f"平均值：{summary.mean:.6f} {unit}\n"
        f"中位数：{summary.median:.6f} {unit}\n"
        f"最小值：{summary.min_value:.6f} {unit}\n"
        f"RMSE：{summary.rmse:.6f} {unit}\n"
        f"SSE：{summary.sse:.6f} {sse_unit}\n"
        f"STD：{summary.std:.6f} {unit}\n"
    )


def render_report_text(
    *,
    run_dir,
    gt_tum_path,
    odom_tum_path,
    tum_summary,
    ape_summary,
    rpe_summary,
    traj_plot_path,
    ape_plot_path,
    rpe_plot_path,
    traj_command,
    ape_command,
    rpe_command,
    generated_at,
    t_max_diff,
    alignment_mode,
    rpe_delta,
    notes,
):
    run_dir = pathlib.Path(run_dir)
    gt_tum_path = pathlib.Path(gt_tum_path)
    odom_tum_path = pathlib.Path(odom_tum_path)
    traj_plot_path = pathlib.Path(traj_plot_path)
    ape_plot_path = pathlib.Path(ape_plot_path)
    rpe_plot_path = pathlib.Path(rpe_plot_path)

    note_lines = "\n".join(f"- {note}" for note in notes) if notes else "- 无"

    return (
        "EVO 轨迹评测报告\n"
        "\n"
        "一、基本信息\n"
        f"运行目录：{run_dir}\n"
        f"生成时间：{generated_at}\n"
        f"t_max_diff：{t_max_diff:.6f} s\n"
        f"对齐模式：{alignment_mode}\n"
        f"RPE delta：{rpe_delta}\n"
        "\n"
        "二、输入轨迹\n"
        f"GT TUM：{gt_tum_path}\n"
        f"里程计 TUM：{odom_tum_path}\n"
        "\n"
        "三、轨迹概况\n"
        f"TUM 样本数：{tum_summary.sample_count}\n"
        f"时间跨度：{tum_summary.start_time:.9f} s -> {tum_summary.end_time:.9f} s\n"
        f"持续时间：{tum_summary.duration_sec:.9f} s\n"
        "\n"
        "四、APE 平移误差\n"
        + _format_metric_block("APE 平移误差统计", ape_summary, "m", sse_unit="m^2")
        + "\n"
        "五、RPE 平移误差\n"
        + _format_metric_block("RPE 平移误差统计", rpe_summary, "m", sse_unit="m^2")
        + "\n"
        "六、图像文件\n"
        f"轨迹叠加图：{traj_plot_path}\n"
        f"APE 图像：{ape_plot_path}\n"
        f"RPE 图像：{rpe_plot_path}\n"
        + "\n"
        "七、复现命令（发布产物路径）\n"
        "以下命令用于复现已发布产物，命令本身引用最终输出路径。\n"
        f"轨迹叠加命令：{traj_command}\n"
        f"APE 命令：{ape_command}\n"
        f"RPE 命令：{rpe_command}\n"
        "\n"
        "八、备注\n"
        f"{note_lines}\n"
    )


def run_evo_command(cmd):
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    completed = subprocess.run(
        cmd,
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    return completed.stdout


def build_ape_command(gt_tum_path, odom_tum_path, output_png, t_max_diff):
    return [
        "evo_ape",
        "tum",
        str(gt_tum_path),
        str(odom_tum_path),
        "--align_origin",
        "-r",
        "trans_part",
        "--t_max_diff",
        str(t_max_diff),
        "--save_plot",
        str(output_png),
    ]


def build_rpe_command(gt_tum_path, odom_tum_path, output_png, t_max_diff, delta):
    return [
        "evo_rpe",
        "tum",
        str(gt_tum_path),
        str(odom_tum_path),
        "--align_origin",
        "-r",
        "trans_part",
        "-u",
        "f",
        "-d",
        str(delta),
        "--t_max_diff",
        str(t_max_diff),
        "--save_plot",
        str(output_png),
    ]


def build_traj_command(gt_tum_path, odom_tum_path, output_png, t_max_diff):
    return [
        "evo_traj",
        "tum",
        str(gt_tum_path),
        str(odom_tum_path),
        "--ref",
        str(gt_tum_path),
        "--sync",
        "--align_origin",
        "--t_max_diff",
        str(t_max_diff),
        "--plot_mode",
        "xy",
        "--save_plot",
        str(output_png),
    ]


def _require_existing_file(path, label):
    path = pathlib.Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return path


def _command_to_text(cmd):
    return shlex.join(str(part) for part in cmd)


def _materialize_plot(expected_path, source_candidates):
    expected_path = pathlib.Path(expected_path)
    if expected_path.is_file():
        return expected_path

    for source_path in source_candidates:
        source_path = pathlib.Path(source_path)
        if source_path.is_file():
            shutil.copyfile(source_path, expected_path)
            return expected_path

    raise FileNotFoundError(
        f"Missing EVO plot output for {expected_path}: "
        + ", ".join(str(pathlib.Path(source)) for source in source_candidates)
    )


def _remove_path_if_exists(path):
    path = pathlib.Path(path)
    if path.exists() or path.is_symlink():
        path.unlink()


def _rpe_artifact_stem(rpe_delta):
    return f"rpe_trans_{rpe_delta}f"


def export_evo_report(run_dir, output_dir=None, t_max_diff=0.001, rpe_delta=1):
    run_dir = pathlib.Path(run_dir)
    output_dir = pathlib.Path(output_dir) if output_dir is not None else run_dir / "analysis" / "evo"

    gt_tum_path = _require_existing_file(run_dir / "trajectory" / "gt_sensor_world_tum.txt", "GT TUM file")
    odom_tum_path = _require_existing_file(run_dir / "trajectory" / "odom_raw_tum.txt", "odom TUM file")

    rpe_artifact_stem = _rpe_artifact_stem(rpe_delta)
    artifact_names = {
        "report.txt",
        "commands.txt",
        "ape_trans_stdout.txt",
        f"{rpe_artifact_stem}_stdout.txt",
        "traj_overlay_xy.png",
        "ape_trans_error.png",
        f"{rpe_artifact_stem}_error.png",
    }

    with tempfile.TemporaryDirectory(prefix="evo_report_stage_") as temp_dir:
        stage_output_dir = pathlib.Path(temp_dir)

        stage_traj_plot_png = stage_output_dir / "traj_overlay_xy.png"
        stage_ape_plot_png = stage_output_dir / "ape_trans_error.png"
        stage_rpe_plot_png = stage_output_dir / f"{rpe_artifact_stem}_error.png"
        stage_report_txt = stage_output_dir / "report.txt"
        stage_commands_txt = stage_output_dir / "commands.txt"
        stage_ape_stdout_txt = stage_output_dir / "ape_trans_stdout.txt"
        stage_rpe_stdout_txt = stage_output_dir / f"{rpe_artifact_stem}_stdout.txt"

        final_traj_plot_png = output_dir / "traj_overlay_xy.png"
        final_ape_plot_png = output_dir / "ape_trans_error.png"
        final_rpe_plot_png = output_dir / f"{rpe_artifact_stem}_error.png"
        final_report_txt = output_dir / "report.txt"
        final_commands_txt = output_dir / "commands.txt"
        final_ape_stdout_txt = output_dir / "ape_trans_stdout.txt"
        final_rpe_stdout_txt = output_dir / f"{rpe_artifact_stem}_stdout.txt"

        traj_command = build_traj_command(gt_tum_path, odom_tum_path, stage_traj_plot_png, t_max_diff)
        ape_command = build_ape_command(gt_tum_path, odom_tum_path, stage_ape_plot_png, t_max_diff)
        rpe_command = build_rpe_command(gt_tum_path, odom_tum_path, stage_rpe_plot_png, t_max_diff, rpe_delta)

        final_traj_command = build_traj_command(gt_tum_path, odom_tum_path, final_traj_plot_png, t_max_diff)
        final_ape_command = build_ape_command(gt_tum_path, odom_tum_path, final_ape_plot_png, t_max_diff)
        final_rpe_command = build_rpe_command(gt_tum_path, odom_tum_path, final_rpe_plot_png, t_max_diff, rpe_delta)

        run_evo_command(traj_command)
        ape_stdout = run_evo_command(ape_command)
        rpe_stdout = run_evo_command(rpe_command)

        _materialize_plot(
            stage_traj_plot_png,
            [stage_traj_plot_png.with_name(f"{stage_traj_plot_png.stem}_trajectories.png")],
        )
        _materialize_plot(
            stage_ape_plot_png,
            [
                stage_ape_plot_png.with_name(f"{stage_ape_plot_png.stem}_raw.png"),
                stage_ape_plot_png.with_name(f"{stage_ape_plot_png.stem}_map.png"),
            ],
        )
        _materialize_plot(
            stage_rpe_plot_png,
            [
                stage_rpe_plot_png.with_name(f"{stage_rpe_plot_png.stem}_raw.png"),
                stage_rpe_plot_png.with_name(f"{stage_rpe_plot_png.stem}_map.png"),
            ],
        )

        stage_ape_stdout_txt.write_text(ape_stdout)
        stage_rpe_stdout_txt.write_text(rpe_stdout)
        stage_commands_txt.write_text(
            "\n".join(
                [
                    "# 复现命令（发布产物路径）",
                    "# 以下命令引用最终输出路径，可直接用于复现已发布产物。",
                    _command_to_text(final_traj_command),
                    _command_to_text(final_ape_command),
                    _command_to_text(final_rpe_command),
                    "",
                ]
            )
        )

        tum_summary = load_tum_summary(gt_tum_path)
        ape_summary = parse_metric_summary(ape_stdout)
        rpe_summary = parse_metric_summary(rpe_stdout)
        generated_at = dt.datetime.now().isoformat(timespec="seconds")

        stage_report_txt.write_text(
            render_report_text(
                run_dir=run_dir,
                gt_tum_path=gt_tum_path,
                odom_tum_path=odom_tum_path,
                tum_summary=tum_summary,
                ape_summary=ape_summary,
                rpe_summary=rpe_summary,
                traj_plot_path=final_traj_plot_png,
                ape_plot_path=final_ape_plot_png,
                rpe_plot_path=final_rpe_plot_png,
                traj_command=_command_to_text(final_traj_command),
                ape_command=_command_to_text(final_ape_command),
                rpe_command=_command_to_text(final_rpe_command),
                generated_at=generated_at,
                t_max_diff=t_max_diff,
                alignment_mode="origin_alignment",
                rpe_delta=rpe_delta,
                notes=[],
            )
        )

        output_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(stage_report_txt, final_report_txt)
        shutil.copy2(stage_commands_txt, final_commands_txt)
        shutil.copy2(stage_ape_stdout_txt, final_ape_stdout_txt)
        shutil.copy2(stage_rpe_stdout_txt, final_rpe_stdout_txt)
        shutil.copy2(stage_traj_plot_png, final_traj_plot_png)
        shutil.copy2(stage_ape_plot_png, final_ape_plot_png)
        shutil.copy2(stage_rpe_plot_png, final_rpe_plot_png)

    for pattern in [f"{final_traj_plot_png.stem}*", f"{final_ape_plot_png.stem}*", "rpe_trans_*f*"]:
        for stale_path in output_dir.glob(pattern):
            if stale_path.name not in artifact_names:
                _remove_path_if_exists(stale_path)

    return EvoReportOutputs(
        output_dir=output_dir,
        report_txt=final_report_txt,
        commands_txt=final_commands_txt,
        ape_stdout_txt=final_ape_stdout_txt,
        rpe_stdout_txt=final_rpe_stdout_txt,
        traj_plot_png=final_traj_plot_png,
        ape_plot_png=final_ape_plot_png,
        rpe_plot_png=final_rpe_plot_png,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description="Export EVO report artifacts for a simulation run.")
    parser.add_argument("--run-dir", required=True, help="Path to sim_run_XXX directory.")
    parser.add_argument("--output-dir", default=None, help="Optional output directory. Defaults to run_dir/analysis/evo.")
    parser.add_argument("--t-max-diff", type=float, default=0.001, help="Maximum timestamp difference for evo sync.")
    parser.add_argument("--rpe-delta", type=int, default=1, help="RPE delta in frames.")
    args = parser.parse_args(argv)

    outputs = export_evo_report(
        run_dir=args.run_dir,
        output_dir=args.output_dir,
        t_max_diff=args.t_max_diff,
        rpe_delta=args.rpe_delta,
    )
    print(f"report_txt: {outputs.report_txt}")
    print(f"commands_txt: {outputs.commands_txt}")


if __name__ == "__main__":
    main()
