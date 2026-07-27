#!/usr/bin/env python3

"""Static preflight for a fixed Gazebo ALERT experiment setup."""

import argparse
import json
import math
import pathlib
import xml.etree.ElementTree as ET

import yaml


REQUIRED_RUN_FACTORS = (
    "scene_id",
    "moving_object_quantity",
    "scene_object_quantity",
    "platform_condition",
    "slam_pipeline",
    "point_cloud_setting",
    "repeat_index",
)
REQUIRED_OBJECT_ATTRIBUTES = (
    "shape",
    "size_class",
    "motion_profile",
    "motion_direction",
    "visibility_condition",
)


def is_fully_occluded(value):
    normalized = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    return "occlu" in normalized and any(
        token in normalized for token in ("full", "complete", "total")
    )


def parse_pose(text):
    try:
        values = [float(token) for token in str(text or "").split()]
    except ValueError:
        return None
    if len(values) != 6 or not all(math.isfinite(value) for value in values):
        return None
    return values


def normalize_catalog(value, errors):
    if not isinstance(value, dict) or not value:
        errors.append("object_id_catalog_missing")
        return {}
    catalog = {}
    names = set()
    for raw_id, raw_name in value.items():
        try:
            object_id = int(raw_id)
        except (TypeError, ValueError):
            errors.append(f"invalid_object_id:{raw_id}")
            continue
        model_name = str(raw_name).strip()
        if object_id <= 0 or object_id > 254:
            errors.append(f"object_id_out_of_range:{object_id}")
            continue
        if not model_name:
            errors.append(f"empty_model_name_for_id:{object_id}")
            continue
        if object_id in catalog:
            errors.append(f"duplicate_catalog_id:{object_id}")
            continue
        if model_name in names:
            errors.append(f"duplicate_catalog_model:{model_name}")
            continue
        catalog[object_id] = model_name
        names.add(model_name)
    return catalog


def load_recorder_config(path):
    with pathlib.Path(path).open() as handle:
        payload = yaml.safe_load(handle)
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError("recorder config must contain a YAML mapping")
    return payload


def parse_world(
    path,
    ego_model_name,
    errors,
    warnings,
    surface_truth_catalog_source="",
    allowed_dynamic_models=(),
):
    root = ET.parse(path).getroot()
    world = root.find("world") if root.tag != "world" else root
    if world is None:
        raise ValueError("Gazebo file does not contain an <sdf>/<world> element")
    model_elements = {
        str(model.attrib.get("name", "")).strip(): model
        for model in world.findall("model")
        if str(model.attrib.get("name", "")).strip()
    }
    for include in world.findall("include"):
        uri = str(include.findtext("uri") or "").strip() or "(missing_uri)"
        warnings.append(f"unverified_environment_include:{uri}")
    model_ids = {}
    id_owners = {}
    for model_name, model in model_elements.items():
        collisions = model.findall(".//collision")
        if model_name == ego_model_name:
            continue
        static_value = str(model.findtext("static") or "").strip().lower()
        if (
            static_value not in ("1", "true")
            and model_name not in set(allowed_dynamic_models)
        ):
            errors.append(f"environment_model_not_static:{model_name}")
        if not collisions:
            continue
        ids = []
        for collision in collisions:
            raw_value = collision.findtext("laser_retro")
            value = 0.0
            try:
                value = float(raw_value) if raw_value is not None else 0.0
                rounded = int(round(value))
            except (TypeError, ValueError):
                rounded = 0
            if rounded <= 0 or rounded > 254 or abs(value - rounded) > 1.0e-6:
                errors.append(
                    f"invalid_or_missing_laser_retro:{model_name}:"
                    f"{collision.attrib.get('name', '')}"
                )
                continue
            ids.append(rounded)
        distinct_ids = sorted(set(ids))
        if len(distinct_ids) > 1:
            errors.append(
                f"model_has_multiple_object_ids:{model_name}:"
                + ",".join(str(value) for value in distinct_ids)
            )
            continue
        if len(distinct_ids) != 1:
            continue
        object_id = distinct_ids[0]
        model_ids[model_name] = object_id
        previous_owner = id_owners.get(object_id)
        if previous_owner is not None and previous_owner != model_name:
            errors.append(
                f"object_id_reused:{object_id}:{previous_owner}:{model_name}"
            )
        else:
            id_owners[object_id] = model_name

    static_marker_catalog = (
        str(surface_truth_catalog_source).strip() == "world_static_marker_visual"
    )
    state = world.find("state")
    if static_marker_catalog and state is not None:
        errors.append("saved_state_snapshot_present")
    elif state is not None:
        state_models = {
            str(model.attrib.get("name", "")).strip(): model
            for model in state.findall("model")
        }
        for model_name, model in model_elements.items():
            state_model = state_models.get(model_name)
            if state_model is None:
                continue
            configured_pose = parse_pose(
                model.findtext("pose") or "0 0 0 0 0 0"
            )
            state_pose = parse_pose(state_model.findtext("pose"))
            if configured_pose is None or state_pose is None:
                errors.append(f"invalid_model_or_state_pose:{model_name}")
                continue
            max_error = max(
                abs(left - right) for left, right in zip(configured_pose, state_pose)
            )
            if max_error > 1.0e-6:
                errors.append(
                    f"state_pose_mismatch:{model_name}:max_abs_error={max_error:.9g}"
                )

    truth_link_counts = {}
    for model_name, model in model_elements.items():
        truth_links = {
            str(link.attrib.get("name", "")).strip(): link
            for link in model.findall(".//link")
            if str(link.attrib.get("name", "")).strip().startswith("ground_truth_")
        }
        marker_visuals = [
            visual
            for link in model.findall(".//link")
            for visual in link.findall("visual")
            if str(visual.attrib.get("name", "")).strip().startswith(
                "ground_truth_marker_v_"
            )
        ]
        truth_link_counts[model_name] = (
            len(marker_visuals) if static_marker_catalog else len(truth_links)
        )
        fixed_children = {
            str(joint.findtext("child") or "").strip().split("::")[-1]
            for joint in model.findall(".//joint")
            if str(joint.attrib.get("type", "")).strip().lower() == "fixed"
        }
        if static_marker_catalog and truth_links:
            errors.append(f"physical_surface_truth_links_present:{model_name}")
        if static_marker_catalog and any(
            str(joint.attrib.get("name", "")).strip().startswith("ground_truth_")
            for joint in model.findall(".//joint")
        ):
            errors.append(f"physical_surface_truth_joints_present:{model_name}")
        for link_name, link in truth_links.items():
            if static_marker_catalog:
                continue
            if link.find("collision") is not None:
                errors.append(
                    f"surface_truth_link_has_collision:{model_name}:{link_name}"
                )
            if link_name not in fixed_children:
                errors.append(
                    f"surface_truth_link_not_fixed:{model_name}:{link_name}"
                )
    return model_ids, truth_link_counts


def validate_setup(world_file, recorder_config):
    errors = []
    warnings = []
    config = load_recorder_config(recorder_config)
    catalog = normalize_catalog(config.get("object_id_catalog"), errors)
    catalog_by_name = {name: object_id for object_id, name in catalog.items()}
    experiment_factors = config.get("experiment_factors", {})
    if not isinstance(experiment_factors, dict):
        errors.append("experiment_factors_must_be_mapping")
        experiment_factors = {}
    object_metadata = config.get("object_metadata", {})
    if not isinstance(object_metadata, dict):
        errors.append("object_metadata_must_be_mapping")
        object_metadata = {}

    if not str(config.get("scenario_id", "")).strip():
        errors.append("scenario_id_missing")
    for factor in REQUIRED_RUN_FACTORS:
        if experiment_factors.get(factor) in (None, ""):
            errors.append(f"required_experiment_factor_missing:{factor}")

    for model_name, attributes in object_metadata.items():
        model_name = str(model_name).strip()
        if model_name not in catalog_by_name:
            errors.append(f"object_metadata_not_in_catalog:{model_name}")
        if not isinstance(attributes, dict):
            errors.append(f"object_metadata_not_mapping:{model_name}")
            continue
        for attribute in REQUIRED_OBJECT_ATTRIBUTES:
            if attributes.get(attribute) in (None, ""):
                errors.append(
                    f"required_object_attribute_missing:{model_name}:{attribute}"
                )
        if is_fully_occluded(attributes.get("visibility_condition")):
            errors.append(f"fully_occluded_object:{model_name}")

    moving_quantity = experiment_factors.get("moving_object_quantity")
    try:
        if int(moving_quantity) != len(object_metadata):
            errors.append("moving_object_quantity_does_not_match_object_metadata")
    except (TypeError, ValueError):
        pass
    scene_quantity = experiment_factors.get("scene_object_quantity")
    try:
        if int(scene_quantity) != len(catalog):
            errors.append("scene_object_quantity_does_not_match_catalog")
    except (TypeError, ValueError):
        pass

    model_ids, truth_link_counts = parse_world(
        world_file,
        str(config.get("ego_model_name", "")).strip(),
        errors,
        warnings,
        surface_truth_catalog_source=config.get(
            "surface_truth_catalog_source", ""
        ),
        allowed_dynamic_models=(config.get("motion_truth_drive_links") or {}).keys(),
    )
    world_catalog = {}
    for model_name, object_id in model_ids.items():
        expected_name = catalog.get(object_id)
        expected_id = catalog_by_name.get(model_name)
        if expected_name != model_name or expected_id != object_id:
            errors.append(
                f"world_catalog_mismatch:{model_name}:world_id={object_id}:"
                f"catalog_id={expected_id}"
            )
        if object_id not in world_catalog:
            world_catalog[object_id] = model_name
    for object_id, model_name in catalog.items():
        if model_name not in model_ids:
            warnings.append(
                f"catalog_model_not_inline_or_not_verifiable:{object_id}:{model_name}"
            )
    for model_name in object_metadata:
        if truth_link_counts.get(model_name, 0) < 3:
            warnings.append(
                f"surface_truth_points_lt_3_or_not_inline:{model_name}:"
                f"{truth_link_counts.get(model_name, 0)}"
            )

    expected_surface_count = config.get("surface_truth_expected_count")
    if expected_surface_count not in (None, ""):
        try:
            expected_surface_count = int(expected_surface_count)
        except (TypeError, ValueError):
            errors.append("invalid_surface_truth_expected_count")
        else:
            actual_surface_count = sum(
                truth_link_counts.get(model_name, 0)
                for model_name in object_metadata
            )
            if actual_surface_count != expected_surface_count:
                errors.append(
                    f"surface_truth_point_count:{actual_surface_count}"
                    f"!={expected_surface_count}"
                )

    status = "FAIL" if errors else ("WARN" if warnings else "PASS")
    return {
        "status": status,
        "world_file": str(pathlib.Path(world_file).resolve()),
        "recorder_config": str(pathlib.Path(recorder_config).resolve()),
        "errors": sorted(set(errors)),
        "warnings": sorted(set(warnings)),
        "world_object_ids": {
            object_id: model_name for object_id, model_name in sorted(world_catalog.items())
        },
        "configured_object_ids": {
            object_id: model_name for object_id, model_name in sorted(catalog.items())
        },
        "surface_truth_point_counts": truth_link_counts,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate a fixed Gazebo world and ALERT recorder config."
    )
    parser.add_argument("--world-file", type=pathlib.Path, required=True)
    parser.add_argument("--recorder-config", type=pathlib.Path, required=True)
    parser.add_argument("--output-json", type=pathlib.Path, default=None)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings (for example non-inline included models) as failure.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    result = validate_setup(args.world_file, args.recorder_config)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered + "\n")
    if result["status"] == "FAIL" or (args.strict and result["status"] == "WARN"):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
