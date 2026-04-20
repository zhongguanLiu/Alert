#!/usr/bin/env python3
# Author: zgliu@cumt.edu.cn
# Affiliation: China University of Mining and Technology
# Open-source release date: 2026-04-20

import math

try:
    import numpy as np
    import rospy
    from gazebo_msgs.msg import ModelStates
    from nav_msgs.msg import Odometry
    from geometry_msgs.msg import Point
    from visualization_msgs.msg import Marker, MarkerArray
except ImportError:  # pragma: no cover
    np = None
    rospy = None
    ModelStates = None
    Odometry = None
    Point = None
    Marker = None
    MarkerArray = None


SYNC_MAX_AGE_SEC = 0.25


def point_to_dict(msg):
    return {
        "x": float(getattr(msg, "x", 0.0)),
        "y": float(getattr(msg, "y", 0.0)),
        "z": float(getattr(msg, "z", 0.0)),
    }


def quaternion_to_dict(msg):
    return {
        "x": float(getattr(msg, "x", 0.0)),
        "y": float(getattr(msg, "y", 0.0)),
        "z": float(getattr(msg, "z", 0.0)),
        "w": float(getattr(msg, "w", 1.0)),
    }


def pose_to_dict(pose):
    return {
        "position": point_to_dict(getattr(pose, "position", None)),
        "orientation": quaternion_to_dict(getattr(pose, "orientation", None)),
    }


def pose_dict_is_finite(pose_dict):
    if not isinstance(pose_dict, dict):
        return False
    position = pose_dict.get("position")
    orientation = pose_dict.get("orientation")
    if not isinstance(position, dict) or not isinstance(orientation, dict):
        return False
    try:
        values = (
            float(position["x"]),
            float(position["y"]),
            float(position["z"]),
            float(orientation["x"]),
            float(orientation["y"]),
            float(orientation["z"]),
            float(orientation["w"]),
        )
    except (KeyError, TypeError, ValueError):
        return False
    return all(math.isfinite(v) for v in values)


def normalize_quaternion_tuple(quaternion):
    x, y, z, w = quaternion
    norm = math.sqrt((x * x) + (y * y) + (z * z) + (w * w))
    if not math.isfinite(norm) or norm <= 1.0e-12:
        return (0.0, 0.0, 0.0, 1.0)
    return (x / norm, y / norm, z / norm, w / norm)


def quaternion_multiply(lhs, rhs):
    lx, ly, lz, lw = lhs
    rx, ry, rz, rw = rhs
    return (
        (lw * rx) + (lx * rw) + (ly * rz) - (lz * ry),
        (lw * ry) - (lx * rz) + (ly * rw) + (lz * rx),
        (lw * rz) + (lx * ry) - (ly * rx) + (lz * rw),
        (lw * rw) - (lx * rx) - (ly * ry) - (lz * rz),
    )


def quaternion_conjugate(quaternion):
    x, y, z, w = quaternion
    return (-x, -y, -z, w)


def rotate_point(point, quaternion):
    quaternion = normalize_quaternion_tuple(quaternion)
    rotated = quaternion_multiply(
        quaternion_multiply(quaternion, (point[0], point[1], point[2], 0.0)),
        quaternion_conjugate(quaternion),
    )
    return (rotated[0], rotated[1], rotated[2])


def invert_pose_dict(pose_dict):
    if not pose_dict_is_finite(pose_dict):
        raise ValueError("invert_pose_dict requires a finite pose dictionary")
    orientation = pose_dict["orientation"]
    position = pose_dict["position"]
    inverse_orientation = quaternion_conjugate(
        normalize_quaternion_tuple(
            (
                float(orientation["x"]),
                float(orientation["y"]),
                float(orientation["z"]),
                float(orientation["w"]),
            )
        )
    )
    inverse_translation = rotate_point(
        (
            -float(position["x"]),
            -float(position["y"]),
            -float(position["z"]),
        ),
        inverse_orientation,
    )
    return {
        "position": {
            "x": inverse_translation[0],
            "y": inverse_translation[1],
            "z": inverse_translation[2],
        },
        "orientation": {
            "x": inverse_orientation[0],
            "y": inverse_orientation[1],
            "z": inverse_orientation[2],
            "w": inverse_orientation[3],
        },
    }


def compose_pose_dicts(base_pose, relative_pose):
    if not pose_dict_is_finite(base_pose) or not pose_dict_is_finite(relative_pose):
        raise ValueError("compose_pose_dicts requires finite pose dictionaries")
    base_position = base_pose["position"]
    base_orientation = base_pose["orientation"]
    relative_position = relative_pose["position"]
    relative_orientation = relative_pose["orientation"]
    base_quaternion = (
        float(base_orientation["x"]),
        float(base_orientation["y"]),
        float(base_orientation["z"]),
        float(base_orientation["w"]),
    )
    relative_quaternion = (
        float(relative_orientation["x"]),
        float(relative_orientation["y"]),
        float(relative_orientation["z"]),
        float(relative_orientation["w"]),
    )
    rotated_relative_position = rotate_point(
        (
            float(relative_position["x"]),
            float(relative_position["y"]),
            float(relative_position["z"]),
        ),
        base_quaternion,
    )
    composed_orientation = quaternion_multiply(base_quaternion, relative_quaternion)
    return {
        "position": {
            "x": float(base_position["x"]) + rotated_relative_position[0],
            "y": float(base_position["y"]) + rotated_relative_position[1],
            "z": float(base_position["z"]) + rotated_relative_position[2],
        },
        "orientation": {
            "x": composed_orientation[0],
            "y": composed_orientation[1],
            "z": composed_orientation[2],
            "w": composed_orientation[3],
        },
    }


def quaternion_to_rotation_matrix(quat):
    x = float(quat.get("x", 0.0))
    y = float(quat.get("y", 0.0))
    z = float(quat.get("z", 0.0))
    w = float(quat.get("w", 1.0))
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm == 0.0:
        return np.eye(3)
    x /= norm
    y /= norm
    z /= norm
    w /= norm
    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ])


def build_rigid_transform_from_pose(pose_dict, source_frame="", target_frame=""):
    position = pose_dict.get("position", {})
    orientation = pose_dict.get("orientation", {})
    return {
        "source_frame": str(source_frame),
        "target_frame": str(target_frame),
        "translation": np.array([
            float(position.get("x", 0.0)),
            float(position.get("y", 0.0)),
            float(position.get("z", 0.0)),
        ]),
        "rotation": quaternion_to_rotation_matrix(orientation),
    }


def invert_rigid_transform(transform):
    rotation = np.asarray(transform["rotation"], dtype=float)
    translation = np.asarray(transform["translation"], dtype=float)
    inv_rotation = rotation.T
    inv_translation = -inv_rotation.dot(translation)
    return {
        "source_frame": str(transform.get("target_frame", "")),
        "target_frame": str(transform.get("source_frame", "")),
        "translation": inv_translation,
        "rotation": inv_rotation,
    }


def transform_point_with_transform(point_dict, transform):
    vec = np.array([
        float(point_dict.get("x", 0.0)),
        float(point_dict.get("y", 0.0)),
        float(point_dict.get("z", 0.0)),
    ])
    out = transform["rotation"].dot(vec) + transform["translation"]
    return {"x": float(out[0]), "y": float(out[1]), "z": float(out[2])}


def derive_world_from_algorithm_transform(
    truth_reference_pose_world,
    algorithm_reference_pose_algorithm,
    source_frame="camera_init",
    target_frame="world",
):
    world_from_algorithm_pose = compose_pose_dicts(
        truth_reference_pose_world,
        invert_pose_dict(algorithm_reference_pose_algorithm),
    )
    return build_rigid_transform_from_pose(
        world_from_algorithm_pose,
        source_frame=source_frame,
        target_frame=target_frame,
    )


def distance_between_points(a, b):
    return math.sqrt(
        (float(a["x"]) - float(b["x"])) ** 2 +
        (float(a["y"]) - float(b["y"])) ** 2 +
        (float(a["z"]) - float(b["z"])) ** 2
    )


class SimTruthRvizBridge:
    def __init__(self):
        if rospy is None or np is None:
            raise RuntimeError("ROS environment is not available for sim_truth_rviz_bridge.py")

        self.truth_frame = str(rospy.get_param("~truth_frame", "world")).strip() or "world"
        self.algorithm_frame = (
            str(rospy.get_param("~algorithm_frame", "camera_init")).strip() or "camera_init"
        )
        self.ego_model_name = str(rospy.get_param("~ego_model_name", "mid360_fastlio")).strip()
        self.model_states_topic = str(
            rospy.get_param("~model_states_topic", "/gazebo/model_states")
        ).strip()
        self.ground_truth_odometry_topic = str(
            rospy.get_param("~ground_truth_odometry_topic", "/ground_truth/odom")
        ).strip()
        self.odometry_topic = str(rospy.get_param("~odometry_topic", "/Odometry")).strip()
        self.moving_threshold_m = float(rospy.get_param("~moving_threshold_m", 0.01))
        self.marker_topic = str(
            rospy.get_param("~marker_topic", "/sim_truth/moving_objects_markers")
        ).strip()
        self.marker_scale_m = float(rospy.get_param("~marker_scale_m", 0.18))
        self.arrow_scale_m = float(rospy.get_param("~arrow_scale_m", 0.05))
        self.text_scale_m = float(rospy.get_param("~text_scale_m", 0.20))

        self._latest_truth_reference_pose_world = None
        self._latest_truth_reference_pose_stamp = None
        self._latest_truth_reference_frame = ""
        self._world_from_algorithm_transform = None
        self._algorithm_from_world_transform = None
        self._initial_object_positions_world = {}
        self._current_stamp = None

        self.marker_pub = rospy.Publisher(self.marker_topic, MarkerArray, queue_size=10)
        self._subscribers = [
            rospy.Subscriber(
                self.model_states_topic,
                ModelStates,
                self._handle_model_states,
                queue_size=1,
            ),
            rospy.Subscriber(
                self.ground_truth_odometry_topic,
                Odometry,
                self._handle_ground_truth_odometry,
                queue_size=10,
            ),
            rospy.Subscriber(
                self.odometry_topic,
                Odometry,
                self._handle_odometry,
                queue_size=10,
            ),
        ]

    def _tracked_model_names(self, msg):
        tracked = []
        for name in getattr(msg, "name", []):
            if name in (self.ego_model_name, "ground_plane"):
                continue
            tracked.append(name)
        return tracked

    def _handle_ground_truth_odometry(self, msg):
        base_frame_id = str(getattr(msg, "child_frame_id", "")).strip()
        if not base_frame_id:
            return
        base_pose_world = pose_to_dict(getattr(getattr(msg, "pose", None), "pose", None))
        if not pose_dict_is_finite(base_pose_world):
            return
        self._latest_truth_reference_pose_world = base_pose_world
        self._latest_truth_reference_frame = base_frame_id
        header = getattr(msg, "header", None)
        stamp = getattr(header, "stamp", None)
        self._latest_truth_reference_pose_stamp = (
            stamp.to_sec() if hasattr(stamp, "to_sec") else None
        )
        self._current_stamp = stamp

    def _handle_odometry(self, msg):
        header = getattr(msg, "header", None)
        stamp = getattr(header, "stamp", None)
        odom_stamp_sec = stamp.to_sec() if hasattr(stamp, "to_sec") else None
        odom_pose = pose_to_dict(getattr(getattr(msg, "pose", None), "pose", None))
        if not pose_dict_is_finite(odom_pose):
            return
        self._current_stamp = stamp
        if self._world_from_algorithm_transform is not None:
            return
        truth_pose = self._latest_truth_reference_pose_world
        truth_stamp_sec = self._latest_truth_reference_pose_stamp
        if truth_pose is None or truth_stamp_sec is None or odom_stamp_sec is None:
            return
        if abs(float(odom_stamp_sec) - float(truth_stamp_sec)) > SYNC_MAX_AGE_SEC:
            return
        self._world_from_algorithm_transform = derive_world_from_algorithm_transform(
            truth_reference_pose_world=truth_pose,
            algorithm_reference_pose_algorithm=odom_pose,
            source_frame=self.algorithm_frame,
            target_frame=self.truth_frame,
        )
        self._algorithm_from_world_transform = invert_rigid_transform(
            self._world_from_algorithm_transform
        )

    def _handle_model_states(self, msg):
        poses_by_name = dict(zip(getattr(msg, "name", []), getattr(msg, "pose", [])))
        tracked_names = self._tracked_model_names(msg)
        for model_name in tracked_names:
            pose = poses_by_name.get(model_name)
            if pose is None:
                continue
            self._initial_object_positions_world.setdefault(
                model_name,
                point_to_dict(getattr(pose, "position", None)),
            )
        if self._algorithm_from_world_transform is None:
            return
        self.marker_pub.publish(self._build_marker_array(tracked_names, poses_by_name))

    def _marker_stamp(self):
        if self._current_stamp is not None:
            return self._current_stamp
        if rospy is not None and hasattr(rospy, "Time") and hasattr(rospy.Time, "now"):
            return rospy.Time.now()
        return None

    def _make_delete_all_marker(self):
        marker = Marker()
        marker.header.frame_id = self.algorithm_frame
        marker.header.stamp = self._marker_stamp()
        marker.action = Marker.DELETEALL
        return marker

    def _make_current_marker(self, marker_id, model_name, current_algorithm):
        marker_scale_m = float(getattr(self, "marker_scale_m", 0.18))
        marker = Marker()
        marker.header.frame_id = self.algorithm_frame
        marker.header.stamp = self._marker_stamp()
        marker.ns = "sim_truth_current"
        marker.id = marker_id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = current_algorithm["x"]
        marker.pose.position.y = current_algorithm["y"]
        marker.pose.position.z = current_algorithm["z"]
        marker.pose.orientation.w = 1.0
        marker.scale.x = marker_scale_m
        marker.scale.y = marker_scale_m
        marker.scale.z = marker_scale_m
        marker.color.r = 0.10
        marker.color.g = 0.85
        marker.color.b = 0.35
        marker.color.a = 0.90
        return marker

    def _make_motion_marker(self, marker_id, initial_algorithm, current_algorithm):
        arrow_scale_m = float(getattr(self, "arrow_scale_m", 0.05))
        marker = Marker()
        marker.header.frame_id = self.algorithm_frame
        marker.header.stamp = self._marker_stamp()
        marker.ns = "sim_truth_motion"
        marker.id = marker_id
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        marker.scale.x = arrow_scale_m * 0.35
        marker.scale.y = arrow_scale_m * 0.7
        marker.scale.z = arrow_scale_m
        marker.color.r = 0.98
        marker.color.g = 0.55
        marker.color.b = 0.08
        marker.color.a = 0.95
        marker.points = [
            Point(initial_algorithm["x"], initial_algorithm["y"], initial_algorithm["z"]),
            Point(current_algorithm["x"], current_algorithm["y"], current_algorithm["z"]),
        ]
        return marker

    def _make_label_marker(self, marker_id, model_name, current_algorithm):
        text_scale_m = float(getattr(self, "text_scale_m", 0.20))
        marker_scale_m = float(getattr(self, "marker_scale_m", 0.18))
        marker = Marker()
        marker.header.frame_id = self.algorithm_frame
        marker.header.stamp = self._marker_stamp()
        marker.ns = "sim_truth_labels"
        marker.id = marker_id
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        marker.pose.position.x = current_algorithm["x"]
        marker.pose.position.y = current_algorithm["y"]
        marker.pose.position.z = current_algorithm["z"] + (marker_scale_m * 1.1)
        marker.pose.orientation.w = 1.0
        marker.scale.z = text_scale_m
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        marker.color.a = 0.95
        marker.text = str(model_name)
        return marker

    def _build_marker_array(self, tracked_names, poses_by_name):
        marker_array = MarkerArray()
        marker_array.markers.append(self._make_delete_all_marker())
        marker_id = 0

        for model_name in tracked_names:
            pose = poses_by_name.get(model_name)
            if pose is None:
                continue
            initial_world = self._initial_object_positions_world.get(model_name)
            current_world = point_to_dict(getattr(pose, "position", None))
            if initial_world is None:
                continue
            displacement = distance_between_points(current_world, initial_world)
            if displacement < self.moving_threshold_m:
                continue
            current_algorithm = transform_point_with_transform(
                current_world,
                self._algorithm_from_world_transform,
            )
            initial_algorithm = transform_point_with_transform(
                initial_world,
                self._algorithm_from_world_transform,
            )
            marker_array.markers.append(
                self._make_current_marker(marker_id, model_name, current_algorithm)
            )
            marker_id += 1
            marker_array.markers.append(
                self._make_motion_marker(marker_id, initial_algorithm, current_algorithm)
            )
            marker_id += 1
            marker_array.markers.append(
                self._make_label_marker(marker_id, model_name, current_algorithm)
            )
            marker_id += 1
        return marker_array


def main():
    if rospy is None:
        raise RuntimeError("rospy is required to run sim_truth_rviz_bridge.py")
    rospy.init_node("sim_truth_rviz_bridge")
    SimTruthRvizBridge()
    rospy.spin()


if __name__ == "__main__":
    main()
