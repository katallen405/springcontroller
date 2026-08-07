#!/home/katallen/.springcontroller_venv/bin/python3
"""
wait_and_enable_torque.py

Only calls the torque-enable service once virtual_spring_node is confirmed
alive AND actually publishing torques, instead of blindly enabling after a
fixed delay. Confirmed live 2026-08-07: a bad springs config crashed
virtual_spring_node on startup before it ever published anything, and
gen3_spring.launch.py's old timer-based auto-enable fired anyway --
gen3_torque_control's own watchdog caught the missing torque stream and
disabled again ~200ms later, but the arm briefly entered torque mode with
nobody actually commanding it. Given mode-transition windows are exactly
what's shown to be fault-prone (see kinova_torque_control_node.py's
_enter_torque_mode fix), that's a real gap worth closing at the source
rather than relying on the watchdog to clean up after it.

Usage: wait_and_enable_torque.py <torque_control_service> <torque_topic> [timeout_sec]
Exits nonzero and does NOT call the enable service if torque_topic never
publishes within timeout_sec.
"""
import sys

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_srvs.srv import SetBool


def main() -> None:
    torque_control_service = sys.argv[1]
    torque_topic = sys.argv[2]
    timeout_sec = float(sys.argv[3]) if len(sys.argv) > 3 else 10.0

    rclpy.init()
    node = Node("torque_enable_gate")
    logger = node.get_logger()

    received = False

    def _cb(_msg: JointState) -> None:
        nonlocal received
        received = True

    node.create_subscription(JointState, torque_topic, _cb, 10)

    deadline = node.get_clock().now() + rclpy.duration.Duration(seconds=timeout_sec)
    while not received and node.get_clock().now() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)

    if not received:
        logger.error(
            f"No message received on {torque_topic} within {timeout_sec:.1f}s -- "
            f"virtual_spring_node isn't alive/publishing valid torques. "
            f"NOT enabling torque control."
        )
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(1)

    logger.info(f"{torque_topic} is publishing -- proceeding to enable torque control.")

    client = node.create_client(SetBool, torque_control_service)
    if not client.wait_for_service(timeout_sec=timeout_sec):
        logger.error(f"{torque_control_service} not available -- NOT enabling torque control.")
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(1)

    future = client.call_async(SetBool.Request(data=True))
    rclpy.spin_until_future_complete(node, future, timeout_sec=timeout_sec)
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
