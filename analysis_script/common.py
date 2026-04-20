#!/usr/bin/env python3
"""Provide shared loaders and helpers for analysis scripts."""

import csv
import hashlib
import json
import math
import os
import pathlib
import re
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Import reusable functions from analyze_sim_run.py
# ---------------------------------------------------------------------------
_DEFORM_SCRIPTS_DIR = str(
    pathlib.Path(__file__).resolve().parents[0].parent
    / "deform_monitor_v2"
    / "scripts"
)
if _DEFORM_SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _DEFORM_SCRIPTS_DIR)

# Lazy import to avoid matplotlib side effects at import time
_asm = None


def _ensure_asm():
    """Lazy-import analyze_sim_run module."""
    global _asm
    if _asm is None:
        # Suppress matplotlib GUI init
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
        pathlib.Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
        import analyze_sim_run as asm
        _asm = asm
    return _asm


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GT_MOVING_THRESHOLD = 0.01  # m — minimum displacement to classify as "moving"
MATCH_RADIUS = 0.8          # m — spatial matching tolerance (accounts for object extent)
TRUTH_BBOX_MARGIN = 0.2     # m — margin for truth bbox construction
DIRECTION_COS_THRESHOLD = 0.5  # minimum cosine similarity for direction consistency

ANALYSIS_ROOT = pathlib.Path(__file__).resolve().parent
REPO_ROOT = ANALYSIS_ROOT.parent
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "output"
DEFAULT_REAL_OUTPUT_ROOT = REPO_ROOT / "real_output"
RESULT_ROOT = ANALYSIS_ROOT / "result"
_SCENARIO_VELOCITY_RE = re.compile(r"_(\d+)p(\d+)mmps(?:_|$)")


def repo_root() -> pathlib.Path:
    """Return the repository root that contains analysis_script."""
    return REPO_ROOT


def analysis_root() -> pathlib.Path:
    """Return the analysis_script directory."""
    return ANALYSIS_ROOT


def default_real_run_dir(run_name: str) -> pathlib.Path:
    """Return the default algorithm directory for a real-data run."""
    return DEFAULT_REAL_OUTPUT_ROOT / run_name / "algorithm"


def real_figure_dir() -> pathlib.Path:
    """Return the default output directory for real-run figures."""
    out = result_root() / "real_runs"
    out.mkdir(parents=True, exist_ok=True)
    return out


def real_timeline_output_paths(run_name: str) -> Tuple[pathlib.Path, pathlib.Path]:
    """Return default PDF and PNG output paths for a real-run timeline figure."""
    figure_dir = real_figure_dir()
    stem = f"{run_name}_timeline"
    return figure_dir / f"{stem}.pdf", figure_dir / f"{stem}.png"


def latest_result_date_dir() -> pathlib.Path:
    """Return the latest YYYYMMDD result directory under analysis_script/result."""
    root = result_root()
    candidates = [
        path for path in root.iterdir()
        if path.is_dir() and _looks_like_date(path.name)
    ]
    if not candidates:
        raise FileNotFoundError(f"No dated result directories found under {root}")
    return max(candidates, key=lambda path: path.name)


def _looks_like_date(value: str) -> bool:
    return len(value) == 8 and value.isdigit()


def _sanitize_segment(value: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9]+", "_", value or "")
    clean = clean.strip("_")
    return clean or "run"


def _fallback_result_dir(run_dir: pathlib.Path) -> pathlib.Path:
    run_hash = hashlib.sha1(str(run_dir.resolve()).encode()).hexdigest()[:8]
    parent_segment = _sanitize_segment(run_dir.parent.name)
    run_segment = _sanitize_segment(run_dir.name)
    return RESULT_ROOT / "external_runs" / f"{parent_segment}_{run_segment}_{run_hash}"


def result_dir_for_run(run_dir: pathlib.Path) -> pathlib.Path:
    """Return the result output directory for a given run.

    Preferred structure: analysis_script/result/<date>/<run_name>/ when
    run_dir lives under the standard <date>/sim_run_* layout. Otherwise
    fall back to an external_runs fingerprinted directory to avoid
    collisions and keep output organized.
    """
    run_dir = run_dir.resolve()
    parent_name = run_dir.parent.name
    default_output_root = DEFAULT_OUTPUT_ROOT.resolve()
    under_default_output = False
    try:
        run_dir.relative_to(default_output_root)
        under_default_output = True
    except ValueError:
        under_default_output = False

    if under_default_output and run_dir.name.startswith("sim_run_") and _looks_like_date(parent_name):
        date_name = parent_name
        run_name = run_dir.name
        out = RESULT_ROOT / date_name / run_name
    else:
        out = _fallback_result_dir(run_dir)

    out.mkdir(parents=True, exist_ok=True)
    return out


def result_root() -> pathlib.Path:
    """Return the top-level result directory."""
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    return RESULT_ROOT


# ---------------------------------------------------------------------------
# Run directory resolution
# ---------------------------------------------------------------------------
def resolve_run_dir(run_dir: Optional[str] = None,
                    output_root: Optional[str] = None,
                    latest: bool = False) -> pathlib.Path:
    """Resolve a run directory from various input forms.

    Args:
        run_dir: Explicit path to a sim_run_NNN directory.
        output_root: Root output directory (contains date subdirs).
        latest: If True, auto-select the most recent run.

    Returns:
        pathlib.Path to the resolved run directory.

    Raises:
        FileNotFoundError if no valid run directory can be found.
    """
    if run_dir:
        p = pathlib.Path(run_dir)
        if p.is_dir():
            return p
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    root = pathlib.Path(output_root) if output_root else DEFAULT_OUTPUT_ROOT
    if not root.is_dir():
        raise FileNotFoundError(f"Output root not found: {root}")

    if latest:
        return _find_latest_run(root)

    raise ValueError("Provide --run-dir or --latest")


def _find_latest_run(output_root: pathlib.Path) -> pathlib.Path:
    """Find the most recently modified sim_run_* directory."""
    candidates = []
    for date_dir in output_root.iterdir():
        if not date_dir.is_dir():
            continue
        for run_dir in date_dir.iterdir():
            if run_dir.is_dir() and run_dir.name.startswith("sim_run_"):
                candidates.append(run_dir)
    if not candidates:
        raise FileNotFoundError(f"No sim_run_* directories found under {output_root}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def find_all_runs(output_root: Optional[str] = None) -> List[pathlib.Path]:
    """Enumerate all sim_run_* directories under the output root, sorted by path."""
    root = pathlib.Path(output_root) if output_root else DEFAULT_OUTPUT_ROOT
    runs = []
    if not root.is_dir():
        return runs

    direct_runs = sorted(
        run_dir for run_dir in root.iterdir()
        if run_dir.is_dir() and run_dir.name.startswith("sim_run_")
    )
    if direct_runs:
        return direct_runs

    for date_dir in sorted(root.iterdir()):
        if not date_dir.is_dir():
            continue
        for run_dir in sorted(date_dir.iterdir()):
            if run_dir.is_dir() and run_dir.name.startswith("sim_run_"):
                runs.append(run_dir)
    return runs


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class GTObject:
    """Ground-truth object summary for metric computation."""
    name: str
    classification: str  # "moving", "static", "outlier"
    net_displacement: float
    peak_displacement: float
    onset_time: Optional[float]  # first time displacement > 0
    end_time: Optional[float]
    positions_t: List[float] = field(default_factory=list)
    positions_xyz: List[Tuple[float, float, float]] = field(default_factory=list)


@dataclass
class RunData:
    """All loaded data for a single run directory."""
    run_dir: pathlib.Path
    alignment: Optional[dict] = None
    gt_objects: List[GTObject] = field(default_factory=list)
    persistent_records: Optional[list] = None
    track_events: Optional[list] = None
    cluster_records: Optional[list] = None
    scenario: Optional[dict] = None
    ablation: Optional[dict] = None
    box_specs: Optional[dict] = None
    # Raw track/link data for spatial matching
    truth_tracks: list = field(default_factory=list)
    link_tracks_by_model: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_jsonl(path: pathlib.Path) -> Optional[list]:
    """Load a JSONL file, returning list of dicts or None if missing."""
    if not path.is_file():
        return None
    records = []
    with open(path, "r") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                # Truncated last line (process killed mid-write) — skip silently
                pass
    return records


def load_json(path: pathlib.Path) -> Optional[dict]:
    """Load a JSON file, returning dict or None if missing."""
    if not path.is_file():
        return None
    with open(path, "r") as f:
        return json.load(f)


def _compute_gt_displacement(positions_xyz, ref_xyz):
    """Compute Euclidean displacement from a reference position."""
    dx = positions_xyz[0] - ref_xyz[0]
    dy = positions_xyz[1] - ref_xyz[1]
    dz = positions_xyz[2] - ref_xyz[2]
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def load_gt_objects(truth_dir: pathlib.Path,
                    links_dir: Optional[pathlib.Path] = None,
                    moving_threshold: float = GT_MOVING_THRESHOLD) -> List[GTObject]:
    """Load ground-truth object trajectories and classify them."""
    objects = []
    obj_dir = truth_dir / "objects"
    if not obj_dir.is_dir():
        return objects

    for csv_path in sorted(obj_dir.glob("*.csv")):
        name = csv_path.stem
        times = []
        xs, ys, zs = [], [], []

        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    t = float(row["recorded_time_sec"])
                    x = float(row["position_x"])
                    y = float(row["position_y"])
                    z = float(row["position_z"])
                except (KeyError, ValueError):
                    continue
                # Outlier check
                if max(abs(x), abs(y), abs(z)) > 1000.0:
                    times, xs, ys, zs = [], [], [], []
                    break
                times.append(t)
                xs.append(x)
                ys.append(y)
                zs.append(z)

        if len(times) < 2:
            objects.append(GTObject(name=name, classification="outlier",
                                   net_displacement=0.0, peak_displacement=0.0,
                                   onset_time=None, end_time=None))
            continue

        ref_x, ref_y, ref_z = xs[0], ys[0], zs[0]
        displacements = []
        for x, y, z in zip(xs, ys, zs):
            d = math.sqrt((x - ref_x) ** 2 + (y - ref_y) ** 2 + (z - ref_z) ** 2)
            displacements.append(d)

        net_disp = displacements[-1] if displacements else 0.0
        peak_disp = max(displacements) if displacements else 0.0

        classification = "moving" if peak_disp >= moving_threshold else "static"

        # Find onset time: first time displacement exceeds a small fraction of moving_threshold
        onset_time = None
        onset_epsilon = moving_threshold * 0.1
        for i, d in enumerate(displacements):
            if d > onset_epsilon:
                onset_time = times[i]
                break

        obj = GTObject(
            name=name,
            classification=classification,
            net_displacement=net_disp,
            peak_displacement=peak_disp,
            onset_time=onset_time,
            end_time=times[-1] if times else None,
            positions_t=times,
            positions_xyz=list(zip(xs, ys, zs)),
        )
        objects.append(obj)

    return objects


def gt_displacement_at_time(obj: GTObject, t: float) -> Optional[float]:
    """Interpolate GT displacement magnitude at a given time."""
    if not obj.positions_t or not obj.positions_xyz:
        return None
    if t <= obj.positions_t[0]:
        return 0.0
    if t >= obj.positions_t[-1]:
        ref = obj.positions_xyz[0]
        last = obj.positions_xyz[-1]
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(last, ref)))

    # Binary search for bracketing interval
    idx = 0
    for i, ti in enumerate(obj.positions_t):
        if ti > t:
            idx = max(0, i - 1)
            break
    else:
        idx = len(obj.positions_t) - 2

    t0 = obj.positions_t[idx]
    t1 = obj.positions_t[idx + 1]
    alpha = (t - t0) / max(1e-9, t1 - t0)
    alpha = max(0.0, min(1.0, alpha))

    p0 = obj.positions_xyz[idx]
    p1 = obj.positions_xyz[idx + 1]
    px = p0[0] + alpha * (p1[0] - p0[0])
    py = p0[1] + alpha * (p1[1] - p0[1])
    pz = p0[2] + alpha * (p1[2] - p0[2])

    ref = obj.positions_xyz[0]
    return math.sqrt((px - ref[0]) ** 2 + (py - ref[1]) ** 2 + (pz - ref[2]) ** 2)


def gt_position_at_time(obj: GTObject, t: float) -> Optional[Tuple[float, float, float]]:
    """Interpolate GT world position at a given time."""
    if not obj.positions_t or not obj.positions_xyz:
        return None
    if t <= obj.positions_t[0]:
        return obj.positions_xyz[0]
    if t >= obj.positions_t[-1]:
        return obj.positions_xyz[-1]

    idx = 0
    for i, ti in enumerate(obj.positions_t):
        if ti > t:
            idx = max(0, i - 1)
            break
    else:
        idx = len(obj.positions_t) - 2

    t0 = obj.positions_t[idx]
    t1 = obj.positions_t[idx + 1]
    alpha = (t - t0) / max(1e-9, t1 - t0)
    alpha = max(0.0, min(1.0, alpha))

    p0 = obj.positions_xyz[idx]
    p1 = obj.positions_xyz[idx + 1]
    return (
        p0[0] + alpha * (p1[0] - p0[0]),
        p0[1] + alpha * (p1[1] - p0[1]),
        p0[2] + alpha * (p1[2] - p0[2]),
    )


def gt_displacement_vector_at_time(
    obj: GTObject, t: float
) -> Optional[Tuple[float, float, float]]:
    """Interpolate GT displacement vector from the initial position to time t."""
    current = gt_position_at_time(obj, t)
    if current is None or not obj.positions_xyz:
        return None
    ref = obj.positions_xyz[0]
    return (
        current[0] - ref[0],
        current[1] - ref[1],
        current[2] - ref[2],
    )


def vector_norm_3d(vector: Tuple[float, float, float]) -> float:
    return math.sqrt(vector[0] ** 2 + vector[1] ** 2 + vector[2] ** 2)


def cosine_similarity_3d(
    a: Tuple[float, float, float], b: Tuple[float, float, float]
) -> Optional[float]:
    norm_a = vector_norm_3d(a)
    norm_b = vector_norm_3d(b)
    if norm_a <= 1.0e-12 or norm_b <= 1.0e-12:
        return None
    return (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]) / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# Coordinate transforms
# ---------------------------------------------------------------------------
def build_world_from_algorithm_transform(alignment: dict):
    """Extract the 4x4 transform from algorithm frame to world frame."""
    if not alignment or not isinstance(alignment, dict):
        return None
    w2a = alignment.get("world_from_algorithm_transform", {})
    pose = w2a.get("pose", {}) if isinstance(w2a, dict) else {}
    pos = pose.get("position", {}) if isinstance(pose, dict) else {}
    ori = pose.get("orientation", {}) if isinstance(pose, dict) else {}

    def _safe_float(value):
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(numeric):
            return None
        return numeric

    qw = _safe_float(ori.get("w", 1.0))
    qx = _safe_float(ori.get("x", 0.0))
    qy = _safe_float(ori.get("y", 0.0))
    qz = _safe_float(ori.get("z", 0.0))
    px = _safe_float(pos.get("x", 0.0))
    py = _safe_float(pos.get("y", 0.0))
    pz = _safe_float(pos.get("z", 0.0))
    if None in {qw, qx, qy, qz, px, py, pz}:
        return None
    q_norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if not math.isfinite(q_norm) or q_norm <= 1.0e-12:
        return None
    qw /= q_norm
    qx /= q_norm
    qy /= q_norm
    qz /= q_norm

    try:
        # Quaternion to rotation matrix
        R = np.array([
            [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
            [2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
            [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)],
        ])
        t = np.array([px, py, pz])
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        return T
    except Exception:
        return None


def transform_point_to_world(point_dict: dict, T_w_a: np.ndarray) -> Optional[Tuple[float, float, float]]:
    """Transform a point from algorithm frame to world frame."""
    if T_w_a is None or point_dict is None:
        return None
    try:
        px = float(point_dict["x"])
        py = float(point_dict["y"])
        pz = float(point_dict["z"])
    except (KeyError, TypeError, ValueError):
        return None
    if not (math.isfinite(px) and math.isfinite(py) and math.isfinite(pz)):
        return None
    p = np.array([px, py, pz, 1.0])
    pw = T_w_a @ p
    xw = float(pw[0])
    yw = float(pw[1])
    zw = float(pw[2])
    if not (math.isfinite(xw) and math.isfinite(yw) and math.isfinite(zw)):
        return None
    return (xw, yw, zw)


# ---------------------------------------------------------------------------
# Spatial matching
# ---------------------------------------------------------------------------
def distance_3d(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)


def bbox_contains(bbox_min: dict, bbox_max: dict, point: Tuple[float, float, float],
                  margin: float = 0.0) -> bool:
    """Check if a point is inside an axis-aligned bounding box with optional margin."""
    try:
        min_x = float(bbox_min["x"])
        min_y = float(bbox_min["y"])
        min_z = float(bbox_min["z"])
        max_x = float(bbox_max["x"])
        max_y = float(bbox_max["y"])
        max_z = float(bbox_max["z"])
        px = float(point[0])
        py = float(point[1])
        pz = float(point[2])
    except (KeyError, TypeError, ValueError):
        return False

    values = [min_x, min_y, min_z, max_x, max_y, max_z, px, py, pz, margin]
    if not all(math.isfinite(v) for v in values):
        return False

    return (
        min_x - margin <= px <= max_x + margin
        and min_y - margin <= py <= max_y + margin
        and min_z - margin <= pz <= max_z + margin
    )


def record_time_sec(record: dict) -> Optional[float]:
    """Extract timestamp in seconds from a JSONL record."""
    if not isinstance(record, dict):
        return None

    def _safe_float(value):
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(numeric):
            return None
        return numeric

    def _first_valid(stamp_dict: dict, keys: Tuple[str, ...]) -> Optional[float]:
        for key in keys:
            if key in stamp_dict:
                value = _safe_float(stamp_dict.get(key))
                if value is not None:
                    return value
        return None

    def _stamp_to_sec(stamp_dict: dict) -> Optional[float]:
        if not isinstance(stamp_dict, dict):
            return None
        sec = _first_valid(stamp_dict, ("sec", "secs", "s"))
        nsec = _first_valid(
            stamp_dict,
            ("nsec", "nsecs", "nanosec", "nanosecs", "ns"),
        )
        if sec is None and nsec is None:
            return None
        if sec is None:
            return nsec * 1e-9
        if nsec is None:
            return sec
        return sec + nsec * 1e-9

    header = record.get("header", {})
    if isinstance(header, dict):
        stamp_sec = _stamp_to_sec(header.get("stamp", {}))
        if stamp_sec is not None:
            return stamp_sec

    stamp_sec = _stamp_to_sec(record.get("stamp", {}))
    if stamp_sec is not None:
        return stamp_sec

    recorded = record.get("recorded_at", {})
    if isinstance(recorded, dict):
        recorded_sec = _stamp_to_sec(recorded)
        if recorded_sec is not None:
            return recorded_sec
    return None


# ---------------------------------------------------------------------------
# Composite loader
# ---------------------------------------------------------------------------
def load_run_data(run_dir: pathlib.Path, load_clusters: bool = True) -> RunData:
    """Load all data needed for metric computation from a run directory."""
    rd = RunData(run_dir=run_dir)

    # Meta
    rd.alignment = load_json(run_dir / "meta" / "frame_alignment.json")
    rd.scenario = load_json(run_dir / "meta" / "scenario_manifest.json")
    rd.ablation = load_json(run_dir / "meta" / "ablation_manifest.json")

    # GT
    rd.gt_objects = load_gt_objects(run_dir / "truth")

    # Algorithm outputs
    rd.persistent_records = load_jsonl(run_dir / "algorithm" / "persistent_risk_regions.jsonl")
    rd.track_events = load_jsonl(run_dir / "algorithm" / "persistent_track_events.jsonl")
    if load_clusters:
        rd.cluster_records = load_jsonl(run_dir / "algorithm" / "clusters.jsonl")

    return rd


def get_ablation_variant(run_dir: pathlib.Path) -> str:
    """Read the ablation variant name from a run's manifest."""
    manifest = load_json(run_dir / "meta" / "ablation_manifest.json")
    if manifest:
        return manifest.get("variant", manifest.get("effective_runtime", {}).get("variant", "unknown"))
    return "unknown"


def get_controlled_objects(run_dir: pathlib.Path) -> List[dict]:
    """Return all controlled-object entries from the scenario manifest."""
    scenario = load_json(run_dir / "meta" / "scenario_manifest.json")
    if not scenario:
        return []
    controls = scenario.get("controls", [])
    if not isinstance(controls, list):
        return []
    return [control for control in controls if isinstance(control, dict)]


def _get_control_by_name(run_dir: pathlib.Path, controlled_object: Optional[str]) -> Optional[dict]:
    """Return the manifest control entry for a specific controlled object."""
    controls = get_controlled_objects(run_dir)
    if not controls:
        return None
    if controlled_object is None:
        return controls[0]
    for control in controls:
        if control.get("controlled_object") == controlled_object:
            return control
    return None


def _control_velocity(control: dict) -> Optional[float]:
    """Compute the magnitude of a control entry's linear velocity."""
    vel = control.get("velocity", {}).get("linear_mps", {})
    vx = float(vel.get("x", 0))
    vy = float(vel.get("y", 0))
    vz = float(vel.get("z", 0))
    speed = math.sqrt(vx * vx + vy * vy + vz * vz)
    return speed if speed > 1.0e-12 else None


def get_injection_velocity(run_dir: pathlib.Path, controlled_object: Optional[str] = None) -> Optional[float]:
    """Read the injection velocity magnitude from scenario manifest."""
    scenario = load_json(run_dir / "meta" / "scenario_manifest.json")
    if not scenario:
        return None

    control = _get_control_by_name(run_dir, controlled_object)
    if controlled_object is not None:
        if not control:
            return None
        speed = _control_velocity(control)
        if speed is not None:
            return speed
        parsed = parse_velocity_from_scenario_id(str(control.get("scenario_id", "")).strip())
        if parsed is not None:
            return parsed
        return None

    if control:
        speed = _control_velocity(control)
        if speed is not None:
            return speed
        parsed = parse_velocity_from_scenario_id(str(control.get("scenario_id", "")).strip())
        if parsed is not None:
            return parsed

    scenario_id = str(scenario.get("scenario_id", "")).strip()
    parsed = parse_velocity_from_scenario_id(scenario_id)
    if parsed is not None:
        return parsed
    return None


def get_controlled_object_name(run_dir: pathlib.Path) -> Optional[str]:
    """Read the controlled object name from scenario manifest."""
    control = _get_control_by_name(run_dir, None)
    return control.get("controlled_object") if control else None


def parse_velocity_from_scenario_id(scenario_id: str) -> Optional[float]:
    """Parse injected linear speed from scenario_id tokens such as `1p0mmps`."""
    text = str(scenario_id).strip()
    if not text:
        return None
    match = _SCENARIO_VELOCITY_RE.search(text)
    if not match:
        return None
    whole = match.group(1)
    frac = match.group(2)
    try:
        velocity_mmps = float(f"{whole}.{frac}")
    except ValueError:
        return None
    return velocity_mmps * 1.0e-3


def get_analysis_controlled_object_name(run_dir: pathlib.Path) -> Optional[str]:
    """Resolve the intended controlled object for analysis.

    Prefer the manifest entry when it matches recorded truth. If the manifest
    appears to point to the ego platform or another non-truth object, fall back
    to the unique moving GT object for the run.
    """
    run_dir = pathlib.Path(run_dir)
    analysis_names = get_analysis_controlled_object_names(run_dir)
    if analysis_names:
        gt_objects = load_gt_objects(run_dir / "truth")
        gt_by_name = {obj.name: obj for obj in gt_objects}
        moving_truth_backed = [
            name for name in analysis_names
            if gt_by_name.get(name) and gt_by_name[name].classification == "moving"
        ]
        if len(moving_truth_backed) == 1:
            return moving_truth_backed[0]
        if len(moving_truth_backed) > 1:
            return analysis_names[0]

        moving_objects = [obj.name for obj in gt_objects if obj.classification == "moving"]
        if len(moving_objects) == 1:
            return moving_objects[0]

        return analysis_names[0]

    gt_objects = load_gt_objects(run_dir / "truth")
    moving_objects = [obj.name for obj in gt_objects if obj.classification == "moving"]
    if len(moving_objects) == 1:
        return moving_objects[0]

    raw_name = get_controlled_object_name(run_dir)
    return raw_name if raw_name else None


def get_analysis_controlled_object_names(run_dir: pathlib.Path) -> List[str]:
    """Return manifest controlled-object names that have truth CSVs."""
    run_dir = pathlib.Path(run_dir)
    truth_dir = run_dir / "truth" / "objects"
    truth_names = {path.stem for path in truth_dir.glob("*.csv")} if truth_dir.is_dir() else set()
    return [
        control.get("controlled_object")
        for control in get_controlled_objects(run_dir)
        if control.get("controlled_object") in truth_names
    ]


def get_scenario_timing(run_dir: pathlib.Path, controlled_object: Optional[str] = None) -> Tuple[Optional[float], Optional[float]]:
    """Read start_delay_sec and duration_sec from scenario manifest."""
    scenario = load_json(run_dir / "meta" / "scenario_manifest.json")
    if not scenario:
        return None, None

    control = _get_control_by_name(run_dir, controlled_object)
    if not control:
        return None, None

    delay = control.get("start_delay_sec")
    duration = control.get("duration_sec")
    return (float(delay) if delay is not None else None,
            float(duration) if duration is not None else None)
