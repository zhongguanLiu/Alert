#!/usr/bin/env python3

import copy
import math
import sys
import threading

import rospy
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import GetModelState, SetModelState
from geometry_msgs.msg import Twist
from tf.transformations import quaternion_multiply

COMPLETION_MESSAGE = "×××××××已完成××××××××"


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


def clear_twist(twist):
    twist.linear.x = 0.0
    twist.linear.y = 0.0
    twist.linear.z = 0.0
    twist.angular.x = 0.0
    twist.angular.y = 0.0
    twist.angular.z = 0.0


def resolve_motion_start_time(configured_start_time, first_valid_sim_time):
    try:
        configured = float(configured_start_time)
    except (TypeError, ValueError):
        configured = -1.0
    if math.isfinite(configured) and configured >= 0.0:
        return configured
    try:
        first_valid = float(first_valid_sim_time)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(first_valid) or first_valid <= 0.0:
        return None
    return first_valid


def compute_scheduled_twist(
    command,
    now_sec,
    motion_start_time,
    start_delay,
    duration,
    twist_factory,
):
    zero = twist_factory()
    try:
        now_sec = float(now_sec)
        motion_start_time = float(motion_start_time)
        start_delay = max(0.0, float(start_delay))
        duration = float(duration)
    except (TypeError, ValueError):
        return zero
    if not all(math.isfinite(value) for value in (now_sec, motion_start_time, start_delay, duration)):
        return zero

    elapsed = now_sec - motion_start_time
    if elapsed < start_delay:
        return zero
    if duration > 0.0 and elapsed >= start_delay + duration:
        return zero
    return copy_twist(command, twist_factory)


def compute_scheduled_interval_duration(
    interval_start,
    interval_end,
    motion_start_time,
    start_delay,
    duration,
):
    try:
        interval_start = float(interval_start)
        interval_end = float(interval_end)
        motion_start_time = float(motion_start_time)
        start_delay = max(0.0, float(start_delay))
        duration = float(duration)
    except (TypeError, ValueError):
        return 0.0
    if not all(
        math.isfinite(value)
        for value in (
            interval_start,
            interval_end,
            motion_start_time,
            start_delay,
            duration,
        )
    ):
        return 0.0
    if interval_end <= interval_start:
        return 0.0

    active_start = motion_start_time + start_delay
    active_end = math.inf if duration <= 0.0 else active_start + duration
    overlap_start = max(interval_start, active_start)
    overlap_end = min(interval_end, active_end)
    return max(0.0, overlap_end - overlap_start)


def build_command_segments(
    initial_command,
    initial_command_stamp,
    command_events,
    interval_start,
    interval_end,
    command_timeout,
    twist_factory,
):
    interval_start = float(interval_start)
    interval_end = float(interval_end)
    command_timeout = float(command_timeout)
    active_command = copy_twist(initial_command, twist_factory)
    active_stamp = initial_command_stamp
    valid_events = []
    for raw_stamp, command in command_events:
        try:
            stamp = float(raw_stamp)
        except (TypeError, ValueError):
            continue
        if math.isfinite(stamp):
            valid_events.append((stamp, copy_twist(command, twist_factory)))
    valid_events.sort(key=lambda item: item[0])

    remaining = []
    interval_events = []
    for stamp, command in valid_events:
        if stamp <= interval_start:
            active_command = command
            active_stamp = stamp
        elif stamp <= interval_end:
            interval_events.append((stamp, command))
        else:
            remaining.append((stamp, command))

    segments = []

    def append_segment(start, end, command, command_stamp):
        if end <= start:
            return
        if command_timeout <= 0.0 or command_stamp is None:
            segments.append((start, end, copy_twist(command, twist_factory)))
            return
        deadline = float(command_stamp) + command_timeout
        if deadline <= start:
            segments.append((start, end, twist_factory()))
        elif deadline < end:
            segments.append((start, deadline, copy_twist(command, twist_factory)))
            segments.append((deadline, end, twist_factory()))
        else:
            segments.append((start, end, copy_twist(command, twist_factory)))

    cursor = interval_start
    for stamp, command in interval_events:
        append_segment(cursor, stamp, active_command, active_stamp)
        cursor = stamp
        active_command = command
        active_stamp = stamp
    append_segment(cursor, interval_end, active_command, active_stamp)
    return segments, active_command, active_stamp, remaining


def resolve_initial_integration_anchor(
    now_sec, motion_start_time, configured_start_time
):
    try:
        now_sec = float(now_sec)
        motion_start_time = float(motion_start_time)
        configured_start_time = float(configured_start_time)
    except (TypeError, ValueError):
        return None
    if not all(
        math.isfinite(value)
        for value in (now_sec, motion_start_time, configured_start_time)
    ):
        return None
    if configured_start_time >= 0.0 and motion_start_time < now_sec:
        return motion_start_time
    return None


def simulation_time_regressed(now_sec, previous_sec):
    if previous_sec is None:
        return False
    try:
        now_sec = float(now_sec)
        previous_sec = float(previous_sec)
    except (TypeError, ValueError):
        return True
    if not math.isfinite(now_sec) or not math.isfinite(previous_sec):
        return True
    return now_sec < previous_sec


def compute_sim_time_step(now_sec, last_sim_time_sec, max_dt):
    try:
        now_sec = float(now_sec)
        max_dt = float(max_dt)
    except (TypeError, ValueError):
        return 0.0, last_sim_time_sec
    if not math.isfinite(now_sec) or not math.isfinite(max_dt) or max_dt <= 0.0:
        return 0.0, last_sim_time_sec
    if last_sim_time_sec is None:
        return 0.0, now_sec
    try:
        previous = float(last_sim_time_sec)
    except (TypeError, ValueError):
        return 0.0, now_sec
    if not math.isfinite(previous) or now_sec < previous:
        return 0.0, now_sec
    dt = now_sec - previous
    if dt <= 0.0:
        return 0.0, previous
    return dt, now_sec


def twist_is_zero(cmd):
    return all(
        value == 0.0
        for value in (
            cmd.linear.x,
            cmd.linear.y,
            cmd.linear.z,
            cmd.angular.x,
            cmd.angular.y,
            cmd.angular.z,
        )
    )


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


def quaternion_from_angular_velocity(angular, dt):
    rotation = tuple(float(value) * float(dt) for value in angular)
    angle = math.sqrt(sum(value * value for value in rotation))
    if angle < 1.0e-12:
        return normalize_quaternion(
            0.5 * rotation[0],
            0.5 * rotation[1],
            0.5 * rotation[2],
            1.0,
        )
    scale = math.sin(0.5 * angle) / angle
    return (
        rotation[0] * scale,
        rotation[1] * scale,
        rotation[2] * scale,
        math.cos(0.5 * angle),
    )


def cross_product(left, right):
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def integrate_body_translation(linear, angular, dt):
    omega_norm = math.sqrt(sum(value * value for value in angular))
    if omega_norm < 1.0e-12:
        return tuple(value * dt for value in linear)

    angle = omega_norm * dt
    omega_cross_linear = cross_product(angular, linear)
    omega_cross_twice = cross_product(angular, omega_cross_linear)
    first_scale = (1.0 - math.cos(angle)) / (omega_norm * omega_norm)
    second_scale = (angle - math.sin(angle)) / (omega_norm ** 3)
    return tuple(
        linear[index] * dt
        + omega_cross_linear[index] * first_scale
        + omega_cross_twice[index] * second_scale
        for index in range(3)
    )


class ModelMotionController:
    def __init__(self):
        self.model_name = rospy.get_param("~model_name", "obstacle_block_left_clone_clone")
        self.state_reference_frame = rospy.get_param("~state_reference_frame", "world")
        self.command_frame = rospy.get_param("~command_frame", "body").strip().lower()
        self.control_rate = rospy.get_param("~control_rate", 50.0)
        self.command_timeout = rospy.get_param("~command_timeout", 0.0)
        self.using_sim_time = rospy.get_param("/use_sim_time", False)
        self.max_dt = rospy.get_param("~max_dt", 0.2)
        self.start_delay = float(rospy.get_param("~start_delay", 2.0))
        self.duration = float(rospy.get_param("~duration", 60.0))
        self.scenario_id = str(rospy.get_param("~scenario_id", "")).strip()
        self.configured_motion_start_time = float(
            rospy.get_param("~motion_start_sim_time", -1.0)
        )
        self.motion_start_time = resolve_motion_start_time(
            self.configured_motion_start_time, None
        )
        self.completion_announced = False

        if self.control_rate <= 0.0:
            raise ValueError("~control_rate must be positive")
        if not self.using_sim_time:
            raise ValueError("/use_sim_time must be true for deterministic Gazebo motion")
        if self.max_dt <= 0.0:
            raise ValueError("~max_dt must be positive")
        if self.start_delay < 0.0:
            raise ValueError("~start_delay must be non-negative")

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
        self.last_cmd_time = None
        self._integration_command = copy_twist(default_cmd, Twist)
        self._integration_command_stamp = None
        self._command_events = []
        self._target_pose = None

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
        rospy.wait_for_service("/gazebo/set_model_state")
        self.get_model_state = rospy.ServiceProxy("/gazebo/get_model_state", GetModelState)
        self.set_model_state = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)

        rospy.Subscriber("~cmd_vel", Twist, self._cmd_vel_cb, queue_size=1)

    def _cmd_vel_cb(self, msg):
        with self.lock:
            received_at = rospy.Time.now().to_sec()
            self.current_cmd = copy_twist(msg, Twist)
            self.last_cmd_time = received_at
            self._command_events.append((received_at, copy_twist(msg, Twist)))

    def maybe_announce_completion(self, now_sec, writer=None):
        if self.completion_announced or self.duration <= 0.0:
            return False
        if self.motion_start_time is None:
            return False
        deadline = self.motion_start_time + self.start_delay + self.duration
        if float(now_sec) < deadline:
            return False
        rendered = f"\033[31m{COMPLETION_MESSAGE}\033[0m"
        if writer is None:
            emit_completion_banner()
        else:
            writer(rendered)
        self.completion_announced = True
        return True

    def integrate_state(self, pose, cmd, dt):
        orientation = (
            pose.orientation.x, pose.orientation.y,
            pose.orientation.z, pose.orientation.w,
        )
        linear = (cmd.linear.x, cmd.linear.y, cmd.linear.z)
        angular = (cmd.angular.x, cmd.angular.y, cmd.angular.z)
        delta_q = quaternion_from_angular_velocity(angular, dt)

        if self.command_frame == "body":
            linear_world = rotate_vector_by_quaternion(linear, orientation)
            angular_world = rotate_vector_by_quaternion(angular, orientation)
            body_translation = integrate_body_translation(linear, angular, dt)
            world_translation = rotate_vector_by_quaternion(
                body_translation, orientation
            )
            next_orientation = quaternion_multiply(orientation, delta_q)
        else:
            linear_world = linear
            angular_world = angular
            world_translation = tuple(value * dt for value in linear)
            next_orientation = quaternion_multiply(delta_q, orientation)

        next_orientation = normalize_quaternion(*next_orientation)
        pose.position.x += world_translation[0]
        pose.position.y += world_translation[1]
        pose.position.z += world_translation[2]
        pose.orientation.x = next_orientation[0]
        pose.orientation.y = next_orientation[1]
        pose.orientation.z = next_orientation[2]
        pose.orientation.w = next_orientation[3]
        return linear_world, angular_world

    def _read_model_state(self):
        try:
            response = self.get_model_state(self.model_name, self.state_reference_frame)
        except rospy.ServiceException as exc:
            rospy.logwarn_throttle(2.0, "GetModelState failed: %s", exc)
            return None
        if not response.success:
            rospy.logwarn_throttle(
                2.0,
                "Model '%s' not available: %s",
                self.model_name,
                response.status_message,
            )
            return None
        return response

    def _write_zero_twist(self):
        pose = getattr(self, "_target_pose", None)
        if pose is None:
            response = self._read_model_state()
            if response is None:
                return False
            pose = response.pose
        model_state = ModelState()
        model_state.model_name = self.model_name
        model_state.reference_frame = self.state_reference_frame
        model_state.pose = copy.deepcopy(pose)
        clear_twist(model_state.twist)
        try:
            result = self.set_model_state(model_state)
        except rospy.ServiceException as exc:
            rospy.logwarn("Failed to write final zero twist: %s", exc)
            return False
        return bool(result.success)

    def _initialize_target_pose(self):
        if getattr(self, "_target_pose", None) is not None:
            return True
        response = self._read_model_state()
        if response is None:
            return False
        self._target_pose = copy.deepcopy(response.pose)
        return True

    def _finalize_motion(self, write_final_state):
        if not write_final_state:
            return False
        self._write_zero_twist()
        return True

    def run(self):
        rate = rospy.Rate(self.control_rate, reset=True)
        last_sim_time = None
        write_final_state = True

        try:
            while not rospy.is_shutdown():
                now_sec = rospy.Time.now().to_sec()
                if simulation_time_regressed(now_sec, last_sim_time):
                    write_final_state = False
                    rospy.logfatal(
                        "Gazebo simulation time moved backwards from %.9f to %.9f; "
                        "stopping to preserve the deterministic motion contract.",
                        last_sim_time,
                        now_sec,
                    )
                    rospy.signal_shutdown("Gazebo simulation time moved backwards")
                    return
                if not self._initialize_target_pose():
                    rate.sleep()
                    continue
                if self.motion_start_time is None:
                    self.motion_start_time = resolve_motion_start_time(
                        self.configured_motion_start_time, now_sec
                    )
                    if self.motion_start_time is not None:
                        rospy.loginfo(
                            "Motion schedule anchored at simulation time %.9f s",
                            self.motion_start_time,
                        )

                if last_sim_time is None and self.motion_start_time is not None:
                    initial_anchor = resolve_initial_integration_anchor(
                        now_sec,
                        self.motion_start_time,
                        self.configured_motion_start_time,
                    )
                    if initial_anchor is not None:
                        last_sim_time = initial_anchor

                dt, last_sim_time = compute_sim_time_step(
                    now_sec, last_sim_time, self.max_dt
                )
                if dt <= 0.0:
                    if self.using_sim_time:
                        rospy.logwarn_throttle(
                            2.0, "Gazebo simulation time is paused or has jumped."
                        )
                    rate.sleep()
                    continue
                if dt > self.max_dt:
                    rospy.logwarn_throttle(
                        2.0,
                        "Controller loop advanced %.6f simulation seconds; "
                        "integrating the full interval without dropping motion.",
                        dt,
                    )

                with self.lock:
                    command_segments, active_command, active_stamp, remaining = (
                        build_command_segments(
                            initial_command=self._integration_command,
                            initial_command_stamp=self._integration_command_stamp,
                            command_events=self._command_events,
                            interval_start=now_sec - dt,
                            interval_end=now_sec,
                            command_timeout=self.command_timeout,
                            twist_factory=Twist,
                        )
                    )
                    self._integration_command = active_command
                    self._integration_command_stamp = active_stamp
                    self._command_events = remaining
                model_state = ModelState()
                model_state.model_name = self.model_name
                model_state.reference_frame = self.state_reference_frame
                for segment_start, segment_end, command in command_segments:
                    active_dt = compute_scheduled_interval_duration(
                        segment_start,
                        segment_end,
                        motion_start_time=self.motion_start_time,
                        start_delay=self.start_delay,
                        duration=self.duration,
                    )
                    if active_dt > 0.0:
                        self.integrate_state(self._target_pose, command, active_dt)
                model_state.pose = copy.deepcopy(self._target_pose)
                clear_twist(model_state.twist)

                try:
                    set_resp = self.set_model_state(model_state)
                except rospy.ServiceException as exc:
                    rospy.logwarn_throttle(2.0, "SetModelState failed: %s", exc)
                    rate.sleep()
                    continue
                if not set_resp.success:
                    rospy.logwarn_throttle(
                        2.0, "SetModelState returned false: %s", set_resp.status_message
                    )
                    rate.sleep()
                    continue

                if self.maybe_announce_completion(now_sec):
                    return
                rate.sleep()
        finally:
            self._finalize_motion(write_final_state)


def main():
    rospy.init_node("model_motion_controller")
    controller = ModelMotionController()
    controller.run()


if __name__ == "__main__":
    main()
