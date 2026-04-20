#!/usr/bin/env python3
"""Compute core metrics for a simulation run."""

import argparse
import json
import math
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import common


def _record_time_key(record: dict):
    t = common.record_time_sec(record)
    if t is None:
        return None
    return round(float(t), 6)


def _cluster_index(cluster_records):
    index = {}
    for record in cluster_records or []:
        key = _record_time_key(record)
        if key is None:
            continue
        index.setdefault(key, []).append(record)
    return index


def _cluster_disp_world(cluster, T_w_a):
    vector = cluster.get("disp_mean")
    if not isinstance(vector, list) or len(vector) < 3:
        return None
    try:
        dx = float(vector[0])
        dy = float(vector[1])
        dz = float(vector[2])
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(dx) and math.isfinite(dy) and math.isfinite(dz)):
        return None
    if T_w_a is None:
        return (dx, dy, dz)
    try:
        return (
            float(T_w_a[0, 0] * dx + T_w_a[0, 1] * dy + T_w_a[0, 2] * dz),
            float(T_w_a[1, 0] * dx + T_w_a[1, 1] * dy + T_w_a[1, 2] * dz),
            float(T_w_a[2, 0] * dx + T_w_a[2, 1] * dy + T_w_a[2, 2] * dz),
        )
    except Exception:
        return (dx, dy, dz)


def _has_directionally_consistent_cluster(
    cluster_records_index,
    timestamp_sec,
    obj,
    T_w_a,
    match_radius,
    direction_cos_threshold=common.DIRECTION_COS_THRESHOLD,
):
    if cluster_records_index is None:
        return False
    key = round(float(timestamp_sec), 6)
    records = cluster_records_index.get(key, [])
    if not records:
        return False

    gt_pos = common.gt_position_at_time(obj, timestamp_sec)
    gt_disp_vec = common.gt_displacement_vector_at_time(obj, timestamp_sec)
    if gt_pos is None or gt_disp_vec is None:
        return False

    best = None
    for record in records:
        for cluster in record.get("clusters", []):
            if not cluster.get("significant", False):
                continue
            center_world = common.transform_point_to_world(cluster.get("center"), T_w_a)
            if center_world is None:
                continue
            dist = common.distance_3d(center_world, gt_pos)
            if dist > match_radius:
                continue
            disp_world = _cluster_disp_world(cluster, T_w_a)
            if disp_world is None:
                continue
            cosine = common.cosine_similarity_3d(disp_world, gt_disp_vec)
            if cosine is None:
                continue
            if best is None or dist < best[0]:
                best = (dist, cosine)

    if best is None:
        return False
    return best[1] >= direction_cos_threshold


# ---------------------------------------------------------------------------
# Metric 1: R_r — Risk Region Recall (Eq. 22)
# ---------------------------------------------------------------------------
def compute_Rr(moving_objects, persistent_records, cluster_records, T_w_a, match_radius):
    """
    R_r = N_matched / N_GT

    For each GT moving object, check if at least one confirmed persistent
    risk region spatially overlaps it at any frame during the run.
    """
    N_GT = len(moving_objects)
    if N_GT == 0:
        return {"R_r": None, "N_GT": 0, "N_matched": 0, "details": []}

    matched_set = set()
    details = []
    cluster_records_index = _cluster_index(cluster_records)

    for obj in moving_objects:
        obj_matched = False
        first_match_time = None

        if persistent_records is None:
            details.append({"object": obj.name, "matched": False, "first_match_time": None})
            continue

        for record in persistent_records:
            t = common.record_time_sec(record)
            if t is None:
                continue
            # For t_resp: only consider frames after GT motion onset
            if obj.onset_time is not None and t < obj.onset_time:
                continue
            gt_pos = common.gt_position_at_time(obj, t)
            if gt_pos is None:
                continue

            for region in record.get("regions", []):
                if not region.get("confirmed", False):
                    continue
                center_algo = region.get("center")
                center_world = common.transform_point_to_world(center_algo, T_w_a)
                if center_world is None:
                    continue

                # Spatial match: center distance OR bbox containment
                dist = common.distance_3d(center_world, gt_pos)
                bbox_hit = common.bbox_contains(
                    region.get("bbox_min", {}), region.get("bbox_max", {}),
                    # Transform GT to algorithm frame for bbox check
                    # (bbox is in algorithm frame)
                    _world_to_algo_point(gt_pos, T_w_a),
                    margin=common.TRUTH_BBOX_MARGIN
                ) if T_w_a is not None else False

                if dist <= match_radius or bbox_hit:
                    if not _has_directionally_consistent_cluster(
                        cluster_records_index,
                        t,
                        obj,
                        T_w_a,
                        match_radius,
                    ):
                        continue
                    obj_matched = True
                    if first_match_time is None:
                        first_match_time = t
                    break  # found a match in this frame, move to next frame
            # Don't break outer loop — we want first_match_time even if already matched

        if obj_matched:
            matched_set.add(obj.name)
        details.append({
            "object": obj.name,
            "matched": obj_matched,
            "first_match_time": first_match_time,
            "gt_onset": obj.onset_time,
            "gt_net_displacement_mm": round(obj.net_displacement * 1000, 2),
        })

    N_matched = len(matched_set)
    R_r = N_matched / N_GT if N_GT > 0 else None

    return {"R_r": R_r, "N_GT": N_GT, "N_matched": N_matched, "details": details}


# ---------------------------------------------------------------------------
# Metric 2: F_c — False Confirmation Rate (Eq. 23)
# ---------------------------------------------------------------------------
def compute_Fc(moving_objects, persistent_records, track_events, cluster_records, T_w_a, match_radius):
    """
    F_c = N_false / N_confirmed

    Enumerate all distinct track_ids that ever reached confirmed state.
    For each, check across ALL frames whether it spatially overlaps any
    GT moving object at that frame's timestamp (frame-level matching).
    """
    # Step 1: Collect all confirmed track_ids
    confirmed_track_ids = set()
    if track_events:
        for event in track_events:
            if event.get("event_type") == "first_confirmed":
                tid = event.get("track_id")
                if tid is not None:
                    confirmed_track_ids.add(tid)

    # Fallback: scan persistent_risk_regions for any confirmed region
    if persistent_records:
        for record in persistent_records:
            for region in record.get("regions", []):
                if region.get("confirmed", False):
                    tid = region.get("track_id")
                    if tid is not None:
                        confirmed_track_ids.add(tid)

    N_confirmed = len(confirmed_track_ids)
    if N_confirmed == 0:
        return {"F_c": None, "N_confirmed": 0, "N_false": 0}

    # Step 2: For each confirmed track, check ALL frames for spatial overlap
    # with ANY GT moving object. A track is a true positive if it matches
    # at least once across its entire lifetime.
    matched_tracks = set()
    cluster_records_index = _cluster_index(cluster_records)

    if persistent_records and moving_objects:
        for record in persistent_records:
            t = common.record_time_sec(record)
            if t is None:
                continue
            for region in record.get("regions", []):
                if not region.get("confirmed", False):
                    continue
                tid = region.get("track_id")
                if tid is None or tid not in confirmed_track_ids or tid in matched_tracks:
                    continue

                center_world = common.transform_point_to_world(
                    region.get("center"), T_w_a)
                if center_world is None:
                    continue

                for obj in moving_objects:
                    gt_pos = common.gt_position_at_time(obj, t)
                    if gt_pos is None:
                        continue
                    if common.distance_3d(center_world, gt_pos) <= match_radius:
                        if not _has_directionally_consistent_cluster(
                            cluster_records_index,
                            t,
                            obj,
                            T_w_a,
                            match_radius,
                        ):
                            continue
                        matched_tracks.add(tid)
                        break

    N_false = N_confirmed - len(matched_tracks)

    F_c = N_false / N_confirmed if N_confirmed > 0 else None

    return {"F_c": F_c, "N_confirmed": N_confirmed, "N_false": N_false}


# ---------------------------------------------------------------------------
# Metric 2a: P_p — Operational Precision (redesigned replacement for F_c)
# ---------------------------------------------------------------------------
def _merge_tracks_into_zones(track_mean_centers_world, zone_merge_radius=1.5):
    """
    Greedy spatial clustering: merge qualified tracks whose mean world-frame
    centers are within zone_merge_radius of each other into a single risk zone.

    Parameters
    ----------
    track_mean_centers_world : dict  {track_id -> (x, y, z)}  world frame
    zone_merge_radius : float  [m]  merge distance (default 1.5 m)

    Returns
    -------
    zones : list of sets  each set contains track_ids belonging to that zone
    """
    import numpy as np
    tids = list(track_mean_centers_world.keys())
    centers = np.array([track_mean_centers_world[t] for t in tids])  # (N, 3)

    assigned = [False] * len(tids)
    zones = []

    for i in range(len(tids)):
        if assigned[i]:
            continue
        zone = {tids[i]}
        assigned[i] = True
        # Single-linkage: keep expanding as long as unassigned tracks are close
        # to ANY track already in this zone
        changed = True
        while changed:
            changed = False
            zone_centers = np.array([track_mean_centers_world[t] for t in zone])
            for j in range(len(tids)):
                if assigned[j]:
                    continue
                dists = np.linalg.norm(zone_centers - centers[j], axis=1)
                if dists.min() <= zone_merge_radius:
                    zone.add(tids[j])
                    assigned[j] = True
                    changed = True
        zones.append(zone)

    return zones


def compute_Pp(moving_objects, track_events, persistent_records, T_w_a, match_radius,
               min_age_frames=10, min_mean_risk=0.60, zone_merge_radius=1.5):
    """
    P_p = N_tp_zones / N_zones   (zone-level operational precision)

    A "qualified track" is a confirmed persistent risk track that has sustained
    age_frames >= min_age_frames AND mean_risk >= min_mean_risk simultaneously.
    Qualified tracks are then merged spatially into "risk zones" (greedy
    single-linkage, merge radius = zone_merge_radius).  A zone is a true
    positive (TP) if any of its member tracks spatially overlaps a GT moving
    object in the frames where it is qualified.

    Zone-level precision is the operationally relevant metric for disaster
    rescue: a rescue commander receives one alarm per zone and must decide
    whether to evacuate that spatial region.  Track-level counting inflates
    the denominator by the number of anchors covering large structural surfaces
    (e.g. a single wall may produce 20 qualified tracks) and therefore
    systematically underestimates precision.

    Returns both track-level (legacy) and zone-level values for reference.
    """
    import numpy as np

    # ------------------------------------------------------------------
    # Step 1: Collect qualified track_ids AND their mean world-frame centers
    # ------------------------------------------------------------------
    qualified_track_ids = set()
    track_center_accum = {}   # tid -> list of (x,y,z) in algo frame (converted later)

    if track_events:
        for event in track_events:
            if event.get("event_type") != "frame_status":
                continue
            if not event.get("confirmed", False):
                continue
            age = int(event.get("age_frames", 0))
            mean_risk = float(event.get("mean_risk", 0.0))
            if age >= min_age_frames and mean_risk >= min_mean_risk:
                tid = event.get("track_id")
                if tid is None:
                    continue
                qualified_track_ids.add(tid)
                c = event.get("center", {})
                if c:
                    track_center_accum.setdefault(tid, []).append(
                        (c.get("x", 0.0), c.get("y", 0.0), c.get("z", 0.0))
                    )

    N_qualified = len(qualified_track_ids)
    if N_qualified == 0:
        return {
            "P_p": None,
            "N_zones": 0,
            "N_tp_zones": 0,
            "N_qualified": 0,
            "N_tp": 0,
            "zone_merge_radius": zone_merge_radius,
            "min_age_frames": min_age_frames,
            "min_mean_risk": min_mean_risk,
        }

    # ------------------------------------------------------------------
    # Step 2: Compute mean world-frame center for each qualified track
    # ------------------------------------------------------------------
    track_mean_world = {}
    for tid, pts in track_center_accum.items():
        if tid not in qualified_track_ids:
            continue
        mean_algo = tuple(float(sum(v[i] for v in pts) / len(pts)) for i in range(3))
        cw = common.transform_point_to_world(
            {"x": mean_algo[0], "y": mean_algo[1], "z": mean_algo[2]}, T_w_a
        )
        if cw is not None:
            track_mean_world[tid] = cw

    # Tracks without center data are kept in qualified set but placed at origin
    # (they will form their own zone and are unlikely to match GT)
    for tid in qualified_track_ids:
        if tid not in track_mean_world:
            track_mean_world[tid] = (0.0, 0.0, 0.0)

    # ------------------------------------------------------------------
    # Step 3: Spatial matching — which qualified tracks are TPs?
    # ------------------------------------------------------------------
    matched_tracks = set()

    if persistent_records and moving_objects:
        for record in persistent_records:
            t = common.record_time_sec(record)
            if t is None:
                continue
            for region in record.get("regions", []):
                if not region.get("confirmed", False):
                    continue
                tid = region.get("track_id")
                if tid not in qualified_track_ids or tid in matched_tracks:
                    continue
                age = int(region.get("age_frames", 0))
                mr = float(region.get("mean_risk", 0.0))
                if age < min_age_frames or mr < min_mean_risk:
                    continue
                center_world = common.transform_point_to_world(region.get("center"), T_w_a)
                if center_world is None:
                    continue
                for obj in moving_objects:
                    gt_pos = common.gt_position_at_time(obj, t)
                    if gt_pos is None:
                        continue
                    if common.distance_3d(center_world, gt_pos) <= match_radius:
                        matched_tracks.add(tid)
                        break

    N_tp_tracks = len(matched_tracks)

    # ------------------------------------------------------------------
    # Step 4: Merge qualified tracks into risk zones (spatial clustering)
    # ------------------------------------------------------------------
    zones = _merge_tracks_into_zones(track_mean_world, zone_merge_radius)
    N_zones = len(zones)

    # A zone is TP if at least one member track is a TP
    N_tp_zones = sum(1 for zone in zones if zone & matched_tracks)

    P_p_zone  = N_tp_zones / N_zones  if N_zones  > 0 else None
    P_p_track = N_tp_tracks / N_qualified if N_qualified > 0 else None

    return {
        # Primary zone-level metric for summary reporting
        "P_p":          round(P_p_zone,  4) if P_p_zone  is not None else None,
        "N_zones":      N_zones,
        "N_tp_zones":   N_tp_zones,
        # Legacy (track-level) — kept for backward compatibility
        "P_p_track":    round(P_p_track, 4) if P_p_track is not None else None,
        "N_qualified":  N_qualified,
        "N_tp":         N_tp_tracks,
        # Parameters
        "zone_merge_radius": zone_merge_radius,
        "min_age_frames": min_age_frames,
        "min_mean_risk":  min_mean_risk,
    }


# ---------------------------------------------------------------------------
# Metric 3: t_resp — Detection Response Time (Eq. 24)
# ---------------------------------------------------------------------------
def compute_t_resp(Rr_details):
    """
    t_resp = t_first_confirmed - t_GT_onset

    Uses the per-object matching results from R_r computation.
    """
    results = []
    for detail in Rr_details:
        if not detail.get("matched"):
            results.append({
                "object": detail["object"],
                "t_resp": None,
                "reason": "not_detected",
            })
            continue

        first_match = detail.get("first_match_time")
        gt_onset = detail.get("gt_onset")
        if first_match is not None and gt_onset is not None:
            t_resp = first_match - gt_onset
            results.append({
                "object": detail["object"],
                "t_resp": round(t_resp, 3),
                "t_first_confirmed": first_match,
                "t_gt_onset": gt_onset,
            })
        else:
            results.append({
                "object": detail["object"],
                "t_resp": None,
                "reason": "missing_timestamps",
            })

    valid = [r["t_resp"] for r in results if r["t_resp"] is not None]
    mean_t_resp = sum(valid) / len(valid) if valid else None

    return {"mean_t_resp": mean_t_resp, "per_object": results}


# ---------------------------------------------------------------------------
# Metric 4: beta_d — Displacement Estimation Relative Bias (Eq. 25)
# ---------------------------------------------------------------------------
def compute_beta_d(moving_objects, cluster_records, T_w_a, match_radius):
    """
    beta_d = (1/N_T) * sum_t [ (||bar_u_C,t|| - d_GT,t) / d_GT,t ]

    For each frame, match significant clusters to GT objects spatially.
    Compute relative displacement bias per matched pair per frame.
    """
    if not cluster_records or not moving_objects:
        return {"beta_d": None, "N_samples": 0, "per_object": {}}

    per_object_biases = {obj.name: [] for obj in moving_objects}

    for record in cluster_records:
        t = common.record_time_sec(record)
        if t is None:
            continue

        clusters = record.get("clusters", [])
        for cluster in clusters:
            if not cluster.get("significant", False):
                continue

            disp_norm = float(cluster.get("disp_norm", 0.0))
            center_algo = cluster.get("center")
            center_world = common.transform_point_to_world(center_algo, T_w_a)
            if center_world is None:
                continue

            # Match to GT object
            for obj in moving_objects:
                gt_pos = common.gt_position_at_time(obj, t)
                if gt_pos is None:
                    continue
                dist = common.distance_3d(center_world, gt_pos)
                if dist > match_radius:
                    continue

                gt_disp_vec = common.gt_displacement_vector_at_time(obj, t)
                disp_world = _cluster_disp_world(cluster, T_w_a)
                if gt_disp_vec is None or disp_world is None:
                    continue
                cosine = common.cosine_similarity_3d(disp_world, gt_disp_vec)
                if cosine is None or cosine < common.DIRECTION_COS_THRESHOLD:
                    continue

                d_gt = common.gt_displacement_at_time(obj, t)
                if d_gt is None or d_gt < 1e-4:
                    continue  # Avoid division by near-zero

                bias = (disp_norm - d_gt) / d_gt
                per_object_biases[obj.name].append(bias)

    # Aggregate
    all_biases = []
    per_obj_summary = {}
    for name, biases in per_object_biases.items():
        if biases:
            mean_bias = sum(biases) / len(biases)
            per_obj_summary[name] = {
                "mean_bias": round(mean_bias, 4),
                "n_samples": len(biases),
            }
            all_biases.extend(biases)
        else:
            per_obj_summary[name] = {"mean_bias": None, "n_samples": 0}

    global_beta_d = sum(all_biases) / len(all_biases) if all_biases else None

    return {
        "beta_d": round(global_beta_d, 4) if global_beta_d is not None else None,
        "N_samples": len(all_biases),
        "per_object": per_obj_summary,
    }


# ---------------------------------------------------------------------------
# Metric 4a: epsilon_d — First-Detection Displacement Error (replaces beta_d)
# ---------------------------------------------------------------------------
def compute_epsilon_d(moving_objects, Rr_details, cluster_records, T_w_a, match_radius):
    """
    epsilon_d = (||hat_u_first_match|| - d_GT_t_first_match) / d_GT_t_first_match

    Displacement estimation error measured at the moment the GT object was FIRST
    correctly detected (first_match_time from compute_Rr). This timestamp is
    guaranteed to have a matching significant cluster since compute_Rr already
    verified directional consistency via _has_directionally_consistent_cluster().

    Unlike beta_d which averages over many frames (contaminated by background
    clusters and ambiguous matching), epsilon_d is a single, well-defined
    measurement per object per run — directly answering "how accurate was the
    displacement estimate at the first successful detection".
    """
    if not Rr_details or not moving_objects:
        return {"epsilon_d": None, "N_samples": 0, "per_object": {}}

    cluster_records_index = _cluster_index(cluster_records)
    obj_by_name = {obj.name: obj for obj in moving_objects}

    per_object = {}

    for detail in Rr_details:
        obj_name = detail.get("object")
        if not detail.get("matched"):
            per_object[obj_name] = None
            continue

        first_match_t = detail.get("first_match_time")
        if first_match_t is None:
            per_object[obj_name] = None
            continue

        obj = obj_by_name.get(obj_name)
        if obj is None:
            per_object[obj_name] = None
            continue

        gt_pos = common.gt_position_at_time(obj, first_match_t)
        d_gt = common.gt_displacement_at_time(obj, first_match_t)
        if gt_pos is None or d_gt is None or d_gt < 1.0e-4:
            per_object[obj_name] = None
            continue

        # Look up the best significant cluster at first_match_t.
        # This lookup should succeed because compute_Rr already verified
        # _has_directionally_consistent_cluster() returned True at this frame.
        key = round(float(first_match_t), 6)
        cluster_frame_records = cluster_records_index.get(key, [])

        best_disp_est = None
        best_dist = float("inf")

        for cluster_record in cluster_frame_records:
            for cluster in cluster_record.get("clusters", []):
                if not cluster.get("significant", False):
                    continue
                c_center = common.transform_point_to_world(cluster.get("center"), T_w_a)
                if c_center is None:
                    continue
                d = common.distance_3d(c_center, gt_pos)
                if d > match_radius:
                    continue
                if d < best_dist:
                    best_dist = d
                    best_disp_est = float(cluster.get("disp_norm", 0.0))

        if best_disp_est is None:
            per_object[obj_name] = None
            continue

        error = (best_disp_est - d_gt) / d_gt
        per_object[obj_name] = {
            "epsilon_d": round(error, 4),
            "t_first_match_s": round(first_match_t, 3),
            "disp_estimated_m": round(best_disp_est, 4),
            "d_gt_m": round(d_gt, 4),
        }

    valid_errors = [v["epsilon_d"] for v in per_object.values() if v is not None]
    mean_epsilon_d = sum(valid_errors) / len(valid_errors) if valid_errors else None

    return {
        "epsilon_d": round(mean_epsilon_d, 4) if mean_epsilon_d is not None else None,
        "N_samples": len(valid_errors),
        "per_object": per_object,
    }


# ---------------------------------------------------------------------------
# Helper: transform GT world point to algorithm frame (for bbox checks)
# ---------------------------------------------------------------------------
def _world_to_algo_point(world_xyz, T_w_a):
    """Approximate inverse transform for bbox containment check."""
    import numpy as np
    if T_w_a is None:
        return {"x": world_xyz[0], "y": world_xyz[1], "z": world_xyz[2]}
    try:
        T_inv = np.linalg.inv(T_w_a)
        p = T_inv @ [world_xyz[0], world_xyz[1], world_xyz[2], 1.0]
        return {"x": p[0], "y": p[1], "z": p[2]}
    except Exception:
        return {"x": world_xyz[0], "y": world_xyz[1], "z": world_xyz[2]}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_metrics(run_dir, match_radius=common.MATCH_RADIUS):
    """Compute all 4 metrics for a single run and return results dict."""
    import numpy as np

    rd = common.load_run_data(pathlib.Path(run_dir))
    T_w_a = common.build_world_from_algorithm_transform(rd.alignment)

    moving = [o for o in rd.gt_objects if o.classification == "moving"]

    analysis_controlled_names = common.get_analysis_controlled_object_names(pathlib.Path(run_dir))
    controlled_name = common.get_analysis_controlled_object_name(pathlib.Path(run_dir))

    print(f"[compute_metrics] Run: {rd.run_dir}")
    print(f"[compute_metrics] GT objects: {len(rd.gt_objects)} total, {len(moving)} moving")
    if analysis_controlled_names:
        print(f"[compute_metrics] Controlled objects: {analysis_controlled_names}")
    elif controlled_name:
        print(f"[compute_metrics] Controlled object: {controlled_name}")
    print(f"[compute_metrics] Persistent frames: {len(rd.persistent_records or [])}")
    print(f"[compute_metrics] Track events: {len(rd.track_events or [])}")
    print(f"[compute_metrics] Cluster frames: {len(rd.cluster_records or [])}")
    print()

    # Metric 1: R_r
    rr_result = compute_Rr(moving, rd.persistent_records, rd.cluster_records, T_w_a, match_radius)
    print(f"  R_r = {rr_result['R_r']}  (N_matched={rr_result['N_matched']}/{rr_result['N_GT']})")

    # Metric 2: F_c (legacy — kept for backward compat)
    fc_result = compute_Fc(
        moving,
        rd.persistent_records,
        rd.track_events,
        rd.cluster_records,
        T_w_a,
        match_radius,
    )
    print(f"  F_c = {fc_result['F_c']}  (N_false={fc_result['N_false']}/{fc_result['N_confirmed']}) [legacy]")

    # Metric 2a: P_p (Operational Precision — replaces F_c in paper)
    pp_result = compute_Pp(
        moving, rd.track_events, rd.persistent_records, T_w_a, match_radius
    )
    print(f"  P_p = {pp_result['P_p']}  (N_tp={pp_result['N_tp']}/{pp_result['N_qualified']}, "
          f"age>={pp_result['min_age_frames']}fr & risk>={pp_result['min_mean_risk']})")

    # Metric 3: t_resp
    t_resp_result = compute_t_resp(rr_result["details"])
    print(f"  t_resp (mean) = {t_resp_result['mean_t_resp']} s")
    for pr in t_resp_result["per_object"]:
        status = f"{pr['t_resp']} s" if pr.get("t_resp") is not None else pr.get("reason", "N/A")
        print(f"    {pr['object']}: {status}")

    # Metric 4: beta_d (legacy — kept for backward compat)
    beta_targets = moving
    if analysis_controlled_names:
        analysis_names_set = set(analysis_controlled_names)
        beta_targets = [obj for obj in moving if obj.name in analysis_names_set]
        if not beta_targets:
            beta_targets = moving
    elif controlled_name:
        beta_targets = [o for o in moving if o.name == controlled_name]
        if not beta_targets:
            beta_targets = moving  # fallback if controlled object not found in GT
    bd_result = compute_beta_d(beta_targets, rd.cluster_records, T_w_a, match_radius)
    print(f"  beta_d = {bd_result['beta_d']}  (N_samples={bd_result['N_samples']}) [legacy]")
    for name, info in bd_result["per_object"].items():
        print(f"    {name}: bias={info['mean_bias']}, n={info['n_samples']}")

    # Metric 4a: epsilon_d (First-Detection Displacement Error — replaces beta_d in paper)
    # Uses first_match_time from R_r (guaranteed to have a matching cluster already)
    ed_result = compute_epsilon_d(beta_targets, rr_result["details"], rd.cluster_records, T_w_a, match_radius)
    print(f"  epsilon_d = {ed_result['epsilon_d']}  (N_samples={ed_result['N_samples']})")
    for name, info in ed_result["per_object"].items():
        if info:
            print(f"    {name}: err={info['epsilon_d']}, est={info['disp_estimated_m']}m, gt={info['d_gt_m']}m @ t={info.get('t_first_match_s')}s")
        else:
            print(f"    {name}: no cluster match at first-detection time")

    # Assemble output
    results = {
        "run_dir": str(rd.run_dir),
        "R_r": rr_result,
        "F_c": fc_result,
        "P_p": pp_result,
        "t_resp": t_resp_result,
        "beta_d": bd_result,
        "epsilon_d": ed_result,
    }

    # Write output
    out_dir = common.result_dir_for_run(rd.run_dir)
    output_path = out_dir / "paper_metrics.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[compute_metrics] Results written to: {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Compute run metrics (R_r, P_p, t_resp, epsilon_d; F_c and beta_d kept for reference)")
    parser.add_argument("--run-dir", type=str, help="Path to sim_run_NNN directory")
    parser.add_argument("--latest", action="store_true", help="Auto-select latest run")
    parser.add_argument("--output-root", type=str, help="Override output root directory")
    parser.add_argument("--match-radius", type=float, default=common.MATCH_RADIUS,
                        help=f"Spatial matching radius in meters (default: {common.MATCH_RADIUS})")
    args = parser.parse_args()

    run_dir = common.resolve_run_dir(
        run_dir=args.run_dir,
        output_root=args.output_root,
        latest=args.latest,
    )
    run_metrics(run_dir, match_radius=args.match_radius)


if __name__ == "__main__":
    main()
