#!/usr/bin/env python3

import math
import os
import sys
import threading
import time

import rospy
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import GetModelState, GetPhysicsProperties, SetModelState
from geometry_msgs.msg import Twist
from tf.transformations import quaternion_from_euler, quaternion_multiply

COMPLETION_MESSAGE = "DONE"


def emit_completion_banner():
    msg = f"\033[31m{COMPLETION_MESSAGE}\033[0m\n"
    sys.stdout.write(msg)
    sys.stdout.flush()
    sys.stderr.write(msg)
    sys.stderr.flush()


def copy_twist(cmd, twist_factory):
    copied = twist_factory()
    copied.linear.x = cmd.linear.x
    copied.linear.y = cmd.linear.y
    copied.linear.z = cmd.linear.z
    copied.angular.x = cmd.angular.x
    copied.angular.y = cmd.angular.y
    copied.angular.z = cmd.angular.z
    return copied


def normalize_quaternion(qx, qy, qz, qw):
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm == 0.0:
        return 0.0, 0.0, 0.0, 1.0
    return qx / norm, qy / norm, qz / norm, qw / norm


def rotate_vector_by_quaternion(vector, quaternion):
    vx, vy, vz = vector
    qx, qy, qz, qw = quaternion
    rotated = quaternion_multiply(
        quaternion_multiply((qx, qy, qz, qw), (vx, vy, vz, 0.0)),
        (-qx, -qy, -qz, qw),
    )
    return rotated[0], rotated[1], rotated[2]


class ModelMotionController:
    def __init__(self):
        self.model_name = rospy.get_param("~model_name", "obstacle_block_left_clone_clone")
        self.state_reference_frame = rospy.get_param("~state_reference_frame", "world")
        self.command_frame = rospy.get_param("~command_frame", "body").strip().lower()
        self.control_rate = rospy.get_param("~control_rate", 50.0)
        self.command_timeout = rospy.get_param("~command_timeout", 0.0)
        self.set_twist = rospy.get_param("~set_twist", True)
        self.using_sim_time = rospy.get_param("/use_sim_time", False)
        self.max_dt = rospy.get_param("~max_dt", 0.2)
        self.start_delay = float(rospy.get_param("~start_delay", 2.0))
        self.duration = float(rospy.get_param("~duration", 60.0))
        self.scenario_id = str(rospy.get_param("~scenario_id", "")).strip()

        if self.command_frame not in ("body", "world"):
            rospy.logfatal("~command_frame must be 'body' or 'world', got: %s", self.command_frame)
            raise SystemExit(1)

        default_cmd = Twist()
        default_cmd.linear.x = rospy.get_param("~linear_x", 0.0)
        default_cmd.linear.y = rospy.get_param("~linear_y", 0.0)
        default_cmd.linear.z = rospy.get_param("~linear_z", 0.0)
        default_cmd.angular.x = math.radians(rospy.get_param("~angular_x_deg", 0.0))
        default_cmd.angular.y = math.radians(rospy.get_param("~angular_y_deg", 0.0))
        default_cmd.angular.z = math.radians(rospy.get_param("~angular_z_deg", 0.0))

        self.lock = threading.Lock()
        self.current_cmd = default_cmd
        self.last_cmd_wall = None

        rospy.loginfo(
            "Controlling model '%s' in %s frame at %.2f Hz  "
            "start_delay=%.1f s  duration=%.1f s  scenario=%s",
            self.model_name, self.command_frame, self.control_rate,
            self.start_delay, self.duration, self.scenario_id or "(empty)",
        )
        rospy.loginfo(
            "Initial cmd: linear=(%.6f, %.6f, %.6f) m/s",
            default_cmd.linear.x, default_cmd.linear.y, default_cmd.linear.z,
        )

        rospy.wait_for_service("/gazebo/get_model_state")
        rospy.wait_for_service("/gazebo/get_physics_properties")
        rospy.wait_for_service("/gazebo/set_model_state")
        self.get_model_state = rospy.ServiceProxy("/gazebo/get_model_state", GetModelState)
        self.get_physics_properties = rospy.ServiceProxy("/gazebo/get_physics_properties", GetPhysicsProperties)
        self.set_model_state = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)

        rospy.Subscriber("~cmd_vel", Twist, self._cmd_vel_cb, queue_size=1)

    def _cmd_vel_cb(self, msg):
        with self.lock:
            self.current_cmd = msg
            self.last_cmd_wall = time.monotonic()

    def _start_completion_timer(self):
        """Background thread: sleep (start_delay + duration) wall seconds,
        print the completion banner, wait 1 s, then hard-exit the process."""
        total = self.start_delay + self.duration

        def _run():
            time.sleep(total)
            emit_completion_banner()
            time.sleep(1.0)
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(0)

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        rospy.loginfo(
            "Completion timer started: %.1f s wall-time until exit "
            "(start_delay=%.1f + duration=%.1f).",
            total, self.start_delay, self.duration,
        )

    def integrate_state(self, pose, cmd, dt):
        orientation = (
            pose.orientation.x, pose.orientation.y,
            pose.orientation.z, pose.orientation.w,
        )
        linear = (cmd.linear.x, cmd.linear.y, cmd.linear.z)
        angular = (cmd.angular.x, cmd.angular.y, cmd.angular.z)

        if self.command_frame == "body":
            linear_world = rotate_vector_by_quaternion(linear, orientation)
            angular_world = rotate_vector_by_quaternion(angular, orientation)
            delta_q = quaternion_from_euler(angular[0] * dt, angular[1] * dt, angular[2] * dt)
            next_orientation = quaternion_multiply(orientation, delta_q)
        else:
            linear_world = linear
            angular_world = angular
            delta_q = quaternion_from_euler(angular[0] * dt, angular[1] * dt, angular[2] * dt)
            next_orientation = quaternion_multiply(delta_q, orientation)

        next_orientation = normalize_quaternion(*next_orientation)
        pose.position.x += linear_world[0] * dt
        pose.position.y += linear_world[1] * dt
        pose.position.z += linear_world[2] * dt
        pose.orientation.x = next_orientation[0]
        pose.orientation.y = next_orientation[1]
        pose.orientation.z = next_orientation[2]
        pose.orientation.w = next_orientation[3]
        return linear_world, angular_world

    def is_physics_paused(self):
        try:
            return self.get_physics_properties().pause
        except rospy.ServiceException:
            return False

    def run(self):
        self._start_completion_timer()

        control_period = 1.0 / self.control_rate
        next_tick = time.monotonic()
        last_sim_time = None

        while not rospy.is_shutdown():
            now_wall = time.monotonic()
            sleep_dur = next_tick - now_wall
            if sleep_dur > 0.0:
                time.sleep(sleep_dur)
            next_tick = max(next_tick + control_period, time.monotonic())

            now_sim = rospy.Time.now()
            if last_sim_time is None:
                last_sim_time = now_sim
                continue

            dt = (now_sim - last_sim_time).to_sec()
            last_sim_time = now_sim

            if dt <= 0.0:
                if self.using_sim_time and self.is_physics_paused():
                    rospy.logwarn_throttle(2.0, "Gazebo physics paused, waiting...")
                continue

            if dt > self.max_dt:
                rospy.logwarn_throttle(2.0, "Large dt=%.3f s, skipping.", dt)
                continue

            with self.lock:
                cmd = copy_twist(self.current_cmd, Twist)
                last_cmd_wall = self.last_cmd_wall

            if self.command_timeout > 0.0 and last_cmd_wall is not None:
                if time.monotonic() - last_cmd_wall > self.command_timeout:
                    cmd = Twist()

            if (cmd.linear.x == 0.0 and cmd.linear.y == 0.0 and cmd.linear.z == 0.0
                    and cmd.angular.x == 0.0 and cmd.angular.y == 0.0 and cmd.angular.z == 0.0):
                continue

            try:
                state_resp = self.get_model_state(self.model_name, self.state_reference_frame)
            except rospy.ServiceException as exc:
                rospy.logwarn_throttle(2.0, "GetModelState failed: %s", exc)
                continue

            if not state_resp.success:
                rospy.logwarn_throttle(2.0, "Model '%s' not available: %s",
                                       self.model_name, state_resp.status_message)
                continue

            model_state = ModelState()
            model_state.model_name = self.model_name
            model_state.reference_frame = self.state_reference_frame
            model_state.pose = state_resp.pose

            linear_world, angular_world = self.integrate_state(model_state.pose, cmd, dt)

            if self.set_twist:
                model_state.twist.linear.x = linear_world[0]
                model_state.twist.linear.y = linear_world[1]
                model_state.twist.linear.z = linear_world[2]
                model_state.twist.angular.x = angular_world[0]
                model_state.twist.angular.y = angular_world[1]
                model_state.twist.angular.z = angular_world[2]

            try:
                set_resp = self.set_model_state(model_state)
            except rospy.ServiceException as exc:
                rospy.logwarn_throttle(2.0, "SetModelState failed: %s", exc)
                continue

            if not set_resp.success:
                rospy.logwarn_throttle(2.0, "SetModelState returned false: %s",
                                       set_resp.status_message)


def main():
    rospy.init_node("model_motion_controller")
    controller = ModelMotionController()
    controller.run()


if __name__ == "__main__":
    main()
