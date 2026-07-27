#!/usr/bin/env python3

import math

import rospy
from gazebo_msgs.srv import GetWorldProperties, SpawnModel
from geometry_msgs.msg import Pose


def main():
    rospy.init_node("spawn_or_replace_model")

    model_name = rospy.get_param("~model_name", "mid360_fastlio")
    model_param = rospy.get_param("~model_param", "robot_description")
    robot_namespace = rospy.get_param("~robot_namespace", "/")
    reference_frame = rospy.get_param("~reference_frame", "world")
    x = rospy.get_param("~x", 0.0)
    y = rospy.get_param("~y", 0.0)
    z = rospy.get_param("~z", 0.0)
    yaw = rospy.get_param("~yaw", 0.0)

    if rospy.has_param(model_param):
        model_xml = rospy.get_param(model_param)
    elif model_param.startswith("/") and rospy.has_param(model_param[1:]):
        model_xml = rospy.get_param(model_param[1:])
    else:
        rospy.logfatal("Model XML parameter not found: %s", model_param)
        raise SystemExit(1)

    rospy.wait_for_service("/gazebo/get_world_properties")
    rospy.wait_for_service("/gazebo/spawn_urdf_model")

    get_world_properties = rospy.ServiceProxy(
        "/gazebo/get_world_properties", GetWorldProperties
    )
    spawn_model = rospy.ServiceProxy("/gazebo/spawn_urdf_model", SpawnModel)

    try:
        world = get_world_properties()
    except rospy.ServiceException as exc:
        rospy.logfatal("Could not inspect Gazebo models before spawning %s: %s", model_name, exc)
        raise SystemExit(1)

    if model_name in world.model_names:
        rospy.logfatal(
            "Gazebo model %s is already present. Remove it from the world file "
            "instead of hot-replacing a model with active sensor plugins.",
            model_name,
        )
        raise SystemExit(1)

    pose = Pose()
    pose.position.x = x
    pose.position.y = y
    pose.position.z = z
    pose.orientation.z = math.sin(yaw * 0.5)
    pose.orientation.w = math.cos(yaw * 0.5)

    try:
        spawn_resp = spawn_model(model_name, model_xml, robot_namespace, pose, reference_frame)
    except rospy.ServiceException as exc:
        rospy.logfatal("SpawnModel service failed for %s: %s", model_name, exc)
        raise SystemExit(1)

    if not spawn_resp.success:
        rospy.logfatal("SpawnModel returned false for %s: %s", model_name, spawn_resp.status_message)
        raise SystemExit(1)

    rospy.loginfo("Spawned Gazebo model: %s", model_name)


if __name__ == "__main__":
    main()
