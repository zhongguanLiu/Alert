#!/usr/bin/env python3

import argparse
import json
import math
import queue
import signal
import sys
import threading
import time

import rospy
from std_msgs.msg import String
from std_srvs.srv import Trigger


SERVICE_NAMES = {
    "reload": "reload",
    "start": "start",
    "stop": "stop",
    "status": "get_status",
}
ACTION_NAMES = ("run", *SERVICE_NAMES)


def call_service(namespace, action, timeout):
    service_name = f"{namespace}/{SERVICE_NAMES[action]}"
    rospy.wait_for_service(service_name, timeout=timeout)
    return rospy.ServiceProxy(service_name, Trigger)()


def decoded_message(message):
    try:
        return json.loads(message)
    except (TypeError, ValueError):
        return message


def state_summary(status):
    state = status.get("state", "UNKNOWN")
    sim_time = status.get("sim_time")
    if isinstance(sim_time, (int, float)):
        return f"[{state}] simulation_time={sim_time:.3f} s"
    return f"[{state}] simulation_time=unavailable"


def wait_for_plan(namespace, service_timeout, interrupt_requested):
    status_messages = queue.Queue()

    def status_callback(message):
        status = decoded_message(message.data)
        if isinstance(status, dict):
            status_messages.put(status)

    subscriber = rospy.Subscriber(
        f"{namespace}/status", String, status_callback, queue_size=20
    )
    last_state = None
    try:
        while not interrupt_requested.is_set():
            try:
                status = status_messages.get(timeout=0.2)
            except queue.Empty:
                continue

            state = status.get("state")
            if state != last_state:
                print(state_summary(status), flush=True)
                last_state = state
            if state == "COMPLETED":
                print("Motion plan completed; all controlled objects remain stopped.")
                return 0
            if state == "STOPPED":
                print("Motion plan was stopped.")
                return 130
            if state == "ERROR":
                print(status.get("last_error", "motion plugin entered ERROR"), file=sys.stderr)
                return 1

        print("\nCtrl+C received; requesting one atomic stop...", flush=True)
        stop_response = call_service(namespace, "stop", service_timeout)
        if not stop_response.success:
            print(stop_response.message, file=sys.stderr)
            return 1

        deadline = time.monotonic() + max(2.0, service_timeout)
        while time.monotonic() < deadline:
            try:
                status = status_messages.get(timeout=0.2)
            except queue.Empty:
                continue
            if status.get("state") == "STOPPED":
                print(state_summary(status), flush=True)
                print("All ten object commands are now zero.")
                return 130

        print(
            "Stop was queued, but STOPPED was not observed before the timeout. "
            "If Gazebo is paused, unpause it for one update.",
            file=sys.stderr,
        )
        return 130
    finally:
        subscriber.unregister()


def main():
    parser = argparse.ArgumentParser(
        description="Control the atomic Gazebo motion plan for object IDs 1-10."
    )
    parser.add_argument(
        "action",
        choices=ACTION_NAMES,
        help="run atomically reloads the current parameters and queues start",
    )
    parser.add_argument(
        "--namespace", default="/synchronized_model_motion", help="plugin ROS namespace"
    )
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument(
        "--linear-speed-mm-s",
        type=float,
        help=(
            "experiment translation speed in millimetres per second; "
            "ID 6 keeps its configured two-phase 0.1 m/s walking speed"
        ),
    )
    parser.add_argument(
        "--angular-speed-rad-s",
        type=float,
        help="experiment rotation speed in radians per second",
    )
    args = parser.parse_args()

    namespace = "/" + args.namespace.strip("/")
    if args.timeout <= 0.0 or not math.isfinite(args.timeout):
        parser.error("--timeout must be a finite positive number")
    if args.linear_speed_mm_s is not None and (
        args.linear_speed_mm_s < 0.0 or not math.isfinite(args.linear_speed_mm_s)
    ):
        parser.error("--linear-speed-mm-s must be a finite non-negative number")
    if args.angular_speed_rad_s is not None and (
        args.angular_speed_rad_s < 0.0 or not math.isfinite(args.angular_speed_rad_s)
    ):
        parser.error("--angular-speed-rad-s must be a finite non-negative number")
    speed_override_requested = (
        args.linear_speed_mm_s is not None or args.angular_speed_rad_s is not None
    )
    if speed_override_requested and args.action not in {"reload", "run"}:
        parser.error("speed options are only valid with reload or run")

    rospy.init_node("synchronized_motion_cli", anonymous=True, disable_signals=True)
    if args.linear_speed_mm_s is not None:
        rospy.set_param(
            f"{namespace}/linear_speed_mm_s", args.linear_speed_mm_s
        )
    if args.angular_speed_rad_s is not None:
        rospy.set_param(
            f"{namespace}/angular_speed_rad_s", args.angular_speed_rad_s
        )

    try:
        if args.action == "run":
            interrupt_requested = threading.Event()

            def request_stop(_signum, _frame):
                interrupt_requested.set()

            previous_sigint_handler = signal.signal(signal.SIGINT, request_stop)
            reload_response = call_service(namespace, "reload", args.timeout)
            if not reload_response.success:
                print(reload_response.message, file=sys.stderr)
                return 1
            start_response = call_service(namespace, "start", args.timeout)
            result = {
                "reload": decoded_message(reload_response.message),
                "start": start_response.message,
            }
            print(json.dumps(result, indent=2, sort_keys=True))
            if not start_response.success:
                return 1
            print("Controller attached. Press Ctrl+C to stop all ten objects.")
            try:
                return wait_for_plan(namespace, args.timeout, interrupt_requested)
            finally:
                signal.signal(signal.SIGINT, previous_sigint_handler)

        response = call_service(namespace, args.action, args.timeout)
    except (rospy.ROSException, rospy.ServiceException) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    message = decoded_message(response.message)
    if isinstance(message, (dict, list)):
        print(json.dumps(message, indent=2, sort_keys=True))
    else:
        print(message)
    return 0 if response.success else 1


if __name__ == "__main__":
    sys.exit(main())
