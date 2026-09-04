#!/home/katallen/.springcontroller_venv/bin/python3
"""
wait_and_enable_torque.py

Only calls the torque-enable service once virtual_spring_node is confirmed
alive AND actually publishing torques, instead of blindly enabling after a
fixed delay. A bad springs config crashing virtual_spring_node on startup
before it ever published anything, combined with a timer-based auto-enable
firing anyway, would otherwise briefly put the arm into torque mode with
nobody actually commanding it before gen3_torque_control's own watchdog
catches the missing torque stream and disables it again. Mode-transition
windows are exactly what's fault-prone here, so this closes the gap at
the source rather than relying on the watchdog to clean up after it.

Also refuses to enable (unless --allow-danger-enable) if
virtual_spring_node's ~/safety_status reports the resting pose is already
inside danger_threshold or in outright collision. Enabling from an
already-marginal starting pose lets the self-collision scale_factor
oscillate right at the danger boundary, producing jerky commanded torque
that can trip the Kinova's own actuator-level current/following-error
protection -- not a bug in the clamp logic, the clamp reacting correctly
to an already-marginal starting pose. That's a deliberate, loud, opt-in
choice rather than something that just happens to work most of the time.

Usage:
  wait_and_enable_torque.py TORQUE_CONTROL_SERVICE TORQUE_TOPIC SAFETY_STATUS_TOPIC
      [--timeout-sec SEC] [--allow-danger-enable]

Exits nonzero and does NOT call the enable service if TORQUE_TOPIC never
publishes within the timeout, or if the resting pose is unsafe and
--allow-danger-enable wasn't passed.
"""
import argparse
import sys

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from std_srvs.srv import SetBool


def _wait_for_message(node: Node, topic: str, msg_type, timeout_sec: float):
    """Spin until one message arrives on topic, or return None on timeout."""
    received = []

    def _cb(msg) -> None:
        received.append(msg)

    sub = node.create_subscription(msg_type, topic, _cb, 10)
    deadline = node.get_clock().now() + rclpy.duration.Duration(seconds=timeout_sec)
    while not received and node.get_clock().now() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
    node.destroy_subscription(sub)
    return received[0] if received else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("torque_control_service")
    parser.add_argument("torque_topic")
    parser.add_argument("safety_status_topic")
    parser.add_argument("--timeout-sec", type=float, default=10.0)
    parser.add_argument(
        "--allow-danger-enable",
        default="false",
        help=(
            "'true'/'false' (default false). If true, enable even if "
            "~/safety_status reports DANGER/COLLISION for the resting pose "
            "(logs a loud warning instead of refusing). Takes a string "
            "value, not a bare flag, so it can be wired straight from a "
            "ROS 2 launch LaunchConfiguration."
        ),
    )
    args = parser.parse_args()
    args.allow_danger_enable = args.allow_danger_enable.strip().lower() in (
        "true", "1", "yes",
    )

    rclpy.init()
    node = Node("torque_enable_gate")
    logger = node.get_logger()

    torques_msg = _wait_for_message(node, args.torque_topic, JointState, args.timeout_sec)
    if torques_msg is None:
        logger.error(
            f"No message received on {args.torque_topic} within "
            f"{args.timeout_sec:.1f}s -- virtual_spring_node isn't "
            f"alive/publishing valid torques. NOT enabling torque control."
        )
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(1)
    logger.info(f"{args.torque_topic} is publishing.")

    safety_msg = _wait_for_message(
        node, args.safety_status_topic, String, args.timeout_sec
    )
    if safety_msg is None:
        logger.error(
            f"No message received on {args.safety_status_topic} within "
            f"{args.timeout_sec:.1f}s -- can't confirm the resting pose is "
            f"safe. NOT enabling torque control."
        )
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(1)

    is_unsafe = not safety_msg.data.startswith("SAFE")
    if is_unsafe:
        if args.allow_danger_enable:
            logger.warn(
                f"{args.safety_status_topic} reports '{safety_msg.data}' -- "
                f"enabling anyway (--allow-danger-enable). Expect the "
                f"self-collision clamp to fight the pose from the start."
            )
        else:
            logger.error(
                f"{args.safety_status_topic} reports '{safety_msg.data}' -- "
                f"resting pose is already inside danger_threshold or in "
                f"collision. NOT enabling torque control. Move the arm to "
                f"a clear pose first, or pass --allow-danger-enable "
                f"(enable_torque_control_allow_danger:=true in the launch "
                f"file) if this is deliberate."
            )
            node.destroy_node()
            rclpy.shutdown()
            sys.exit(1)
    else:
        logger.info(f"{args.safety_status_topic} reports '{safety_msg.data}'.")

    client = node.create_client(SetBool, args.torque_control_service)
    if not client.wait_for_service(timeout_sec=args.timeout_sec):
        logger.error(
            f"{args.torque_control_service} not available -- NOT enabling "
            f"torque control."
        )
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(1)

    future = client.call_async(SetBool.Request(data=True))
    rclpy.spin_until_future_complete(node, future, timeout_sec=args.timeout_sec)
    result = future.result()
    if result is None or not result.success:
        logger.error(f"Enable call failed: {result}")
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(1)

    logger.info(f"Torque control enabled: {result.message}")
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
