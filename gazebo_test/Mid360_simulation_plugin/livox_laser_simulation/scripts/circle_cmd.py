#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Twist


def main():
    rospy.init_node("mid360_circle_cmd")

    linear_speed = rospy.get_param("~linear_speed", 0.3)
    angular_speed = rospy.get_param("~angular_speed", 0.093)
    publish_rate = rospy.get_param("~publish_rate", 20.0)
    start_delay = rospy.get_param("~start_delay", 3.0)

    pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
    rate = rospy.Rate(publish_rate)

    if start_delay > 0.0:
        rospy.sleep(start_delay)

    cmd = Twist()
    cmd.linear.x = linear_speed
    cmd.angular.z = angular_speed

    rospy.loginfo("Publishing circle cmd_vel: linear=%.3f angular=%.3f", linear_speed, angular_speed)

    try:
        while not rospy.is_shutdown():
            pub.publish(cmd)
            rate.sleep()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()
