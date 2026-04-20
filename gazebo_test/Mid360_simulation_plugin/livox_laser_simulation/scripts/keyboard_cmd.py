#!/usr/bin/env python3

import select
import sys
import termios
import tty

import rospy
from geometry_msgs.msg import Twist


HELP = """
Mid360 Keyboard Teleop
----------------------
Arrow Up / W    : forward
Arrow Down / S  : backward
Arrow Left / A  : turn left
Arrow Right / D : turn right
Space           : stop
Q               : quit teleop
"""


def read_key(timeout):
    ready, _, _ = select.select([sys.stdin], [], [], timeout)
    if not ready:
        return ""

    first = sys.stdin.read(1)
    if first != "\x1b":
        return first

    ready, _, _ = select.select([sys.stdin], [], [], 0.01)
    if not ready:
        return first

    second = sys.stdin.read(1)
    if second != "[":
        return first + second

    ready, _, _ = select.select([sys.stdin], [], [], 0.01)
    if not ready:
        return first + second

    third = sys.stdin.read(1)
    return first + second + third


def build_cmd(key, linear_speed, angular_speed):
    cmd = Twist()

    if key in ("\x1b[A", "w", "W"):
        cmd.linear.x = linear_speed
    elif key in ("\x1b[B", "s", "S"):
        cmd.linear.x = -linear_speed
    elif key in ("\x1b[D", "a", "A"):
        cmd.angular.z = angular_speed
    elif key in ("\x1b[C", "d", "D"):
        cmd.angular.z = -angular_speed
    elif key == " ":
        pass
    else:
        return None

    return cmd


def main():
    rospy.init_node("mid360_keyboard_cmd")

    linear_speed = rospy.get_param("~linear_speed", 0.2)
    angular_speed = rospy.get_param("~angular_speed", 0.3)
    publish_rate = rospy.get_param("~publish_rate", 20.0)

    pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
    rate = rospy.Rate(publish_rate)

    print(HELP)
    print(
        "Current speed: linear={:.2f} m/s angular={:.2f} rad/s".format(
            linear_speed, angular_speed
        )
    )
    print("Focus this terminal, then press keys to drive the robot.\n")
    sys.stdout.flush()

    settings = termios.tcgetattr(sys.stdin)
    last_cmd = Twist()

    try:
        tty.setraw(sys.stdin.fileno())

        while not rospy.is_shutdown():
            key = read_key(1.0 / publish_rate)

            if key in ("\x03", "q", "Q"):
                break

            cmd = build_cmd(key, linear_speed, angular_speed)
            if cmd is not None:
                last_cmd = cmd

            pub.publish(last_cmd)
            rate.sleep()
    finally:
        pub.publish(Twist())
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)


if __name__ == "__main__":
    main()
