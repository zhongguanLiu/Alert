#!/usr/bin/env python3

import math

import rospy
from geometry_msgs.msg import Twist

DEFAULT_PATROL_MODE = "x_oscillate_with_yaw"
DEFAULT_SEGMENT_DURATION_SEC = 10.0
DEFAULT_YAW_AMPLITUDE_RAD_PER_SEC = 0.10
DEFAULT_YAW_PERIOD_SEC = 20.0


def compute_angular_speed(linear_speed, radius):
    radius = float(radius)
    if radius <= 0.0:
        raise ValueError("radius must be > 0")
    return float(linear_speed) / radius


def compute_patrol_angular_z(linear_speed, radius):
    return -compute_angular_speed(linear_speed, radius)


def compute_x_oscillate_linear_x(linear_speed, segment_duration, elapsed_time):
    segment_duration = float(segment_duration)
    if segment_duration <= 0.0:
        raise ValueError("segment_duration must be > 0")

    linear_speed = float(linear_speed)
    elapsed_time = max(0.0, float(elapsed_time))
    segment_index = int(math.floor(elapsed_time / segment_duration))
    return -linear_speed if (segment_index % 2) == 0 else linear_speed


def compute_x_oscillate_with_yaw_angular_z(yaw_amplitude, yaw_period, elapsed_time):
    yaw_period = float(yaw_period)
    if yaw_period <= 0.0:
        raise ValueError("yaw_period must be > 0")

    yaw_amplitude = float(yaw_amplitude)
    elapsed_time = max(0.0, float(elapsed_time))
    return yaw_amplitude * math.sin((2.0 * math.pi * elapsed_time) / yaw_period)


def main():
    rospy.init_node("mid360_work_patrol_cmd")

    patrol_mode = str(rospy.get_param("~patrol_mode", DEFAULT_PATROL_MODE)).strip() or DEFAULT_PATROL_MODE
    linear_speed = float(rospy.get_param("~linear_speed", 0.3))
    radius = float(rospy.get_param("~radius", 1.0))
    publish_rate = float(rospy.get_param("~publish_rate", 20.0))
    start_delay = float(rospy.get_param("~start_delay", 0.0))
    segment_duration = float(
        rospy.get_param("~segment_duration", DEFAULT_SEGMENT_DURATION_SEC)
    )
    yaw_amplitude = float(
        rospy.get_param("~yaw_amplitude", DEFAULT_YAW_AMPLITUDE_RAD_PER_SEC)
    )
    yaw_period = float(
        rospy.get_param("~yaw_period", DEFAULT_YAW_PERIOD_SEC)
    )

    pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
    rate = rospy.Rate(publish_rate)

    if start_delay > 0.0:
        rospy.sleep(start_delay)

    start_time = rospy.Time.now().to_sec()

    if patrol_mode == "circle":
        angular_speed = compute_patrol_angular_z(linear_speed, radius)
        rospy.loginfo(
            "Publishing work patrol cmd_vel (circle): linear=%.3f m/s radius=%.3f m angular=%.3f rad/s",
            linear_speed,
            radius,
            angular_speed,
        )
    elif patrol_mode == "x_oscillate":
        rospy.loginfo(
            "Publishing work patrol cmd_vel (x_oscillate): speed=%.3f m/s segment_duration=%.3f s",
            linear_speed,
            segment_duration,
        )
    elif patrol_mode == "x_oscillate_with_yaw":
        rospy.loginfo(
            "Publishing work patrol cmd_vel (x_oscillate_with_yaw): speed=%.3f m/s segment_duration=%.3f s yaw_amplitude=%.3f rad/s yaw_period=%.3f s",
            linear_speed,
            segment_duration,
            yaw_amplitude,
            yaw_period,
        )
    else:
        raise ValueError(f"unsupported patrol_mode: {patrol_mode}")

    while not rospy.is_shutdown():
        twist = Twist()
        if patrol_mode == "circle":
            twist.linear.x = linear_speed
            twist.angular.z = angular_speed
        elif patrol_mode == "x_oscillate":
            elapsed_time = rospy.Time.now().to_sec() - start_time
            twist.linear.x = compute_x_oscillate_linear_x(
                linear_speed,
                segment_duration,
                elapsed_time,
            )
        else:
            elapsed_time = rospy.Time.now().to_sec() - start_time
            twist.linear.x = compute_x_oscillate_linear_x(
                linear_speed,
                segment_duration,
                elapsed_time,
            )
            twist.angular.z = compute_x_oscillate_with_yaw_angular_z(
                yaw_amplitude,
                yaw_period,
                elapsed_time,
            )
        pub.publish(twist)
        rate.sleep()


if __name__ == "__main__":
    main()
