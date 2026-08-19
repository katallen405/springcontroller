#!/home/katallen/.springcontroller_venv/bin/python3
"""
move_to_joint_angles.py

Collision-checked caller for gen3_torque_control's
~move_to_joint_angles topic (see kinova_torque_control_node.py). That topic
only knows how to move the arm -- it has no notion of the loaded
scene/self-collision model, so this script is where the safety gate lives:
queries virtual_spring_node's ~/check_collision_at service (the same
collision model virtual_spring_node itself uses, scene objects included) for
the target pose, and refuses to publish the move at all if the target would
be in outright collision.

Deliberately a separate always-run-this-instead-of-the-raw-topic script
rather than baking the check into kinova_torque_control_node itself: that
node has no URDF/pinocchio dependency today, and adding one just for this
would mean duplicating (and risking drift from) the collision model
virtual_spring_node already loads live -- see gen3_ros2_kortex_coexistence
and the move_to_joint_angles docstring for why this pair of nodes has to
stay this way (no kortex_bringup running alongside gen3_torque_control).

Usage:
  move_to_joint_angles.py J1 J2 J3 J4 J5 J6 J7   (radians)
      [--check-service SERVICE] [--move-topic TOPIC]
      [--timeout-sec SEC] [--strict]

--strict also refuses when the target is merely in the danger zone
(in_danger), not just outright in_collision. Default only refuses on
in_collision, matching "reject if it would cause a collision" -- the
in_danger case is reported as a loud warning, not a refusal, since it's not
actually a collision.

Exits nonzero and does NOT publish the move if the collision check itself
fails (service unavailable, wrong joint count, etc. -- fails closed) or if
the target is rejected.
"""
import argparse
import sys
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

from springcontroller_interfaces.srv import CheckCollisionAtAngles


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("joint_angles_rad", type=float, nargs=7, metavar="J")
    parser.add_argument(
        "--check-service", default="/virtual_spring_node/check_collision_at",
    )
    parser.add_argument(
        "--move-topic", default="/gen3_torque_control/move_to_joint_angles",
    )
    parser.add_argument("--timeout-sec", type=float, default=10.0)
    parser.add_argument(
        "--strict", action="store_true",
        help="Also refuse when the target is merely in the danger zone, not just in_collision.",
    )
    args = parser.parse_args()

    rclpy.init()
    node = Node("move_to_joint_angles_gate")
    logger = node.get_logger()

    client = node.create_client(CheckCollisionAtAngles, args.check_service)
    if not client.wait_for_service(timeout_sec=args.timeout_sec):
        logger.error(
            f"{args.check_service} not available -- can't verify the target "
            f"is collision-free. NOT moving (fail closed)."
        )
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(1)

    request = CheckCollisionAtAngles.Request(joint_angles_rad=args.joint_angles_rad)
    future = client.call_async(request)
    rclpy.spin_until_future_complete(node, future, timeout_sec=args.timeout_sec)
    result = future.result()
    if result is None:
        logger.error(f"{args.check_service} call failed or timed out. NOT moving (fail closed).")
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(1)

    logger.info(
        f"check_collision_at: in_collision={result.in_collision} "
        f"in_danger={result.in_danger} min_distance={result.min_distance:.4f}m "
        f"closest_pair=({result.link_a}, {result.link_b})"
    )

    if result.in_collision:
        logger.error(
            f"Target {args.joint_angles_rad} would be in self/scene collision "
            f"(closest pair: {result.link_a}, {result.link_b}, "
            f"min_distance={result.min_distance:.4f}m). NOT moving."
        )
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(1)

    if result.in_danger:
        if args.strict:
            logger.error(
                f"Target {args.joint_angles_rad} is in the danger zone "
                f"(min_distance={result.min_distance:.4f}m, closest pair: "
                f"{result.link_a}, {result.link_b}) and --strict was passed. NOT moving."
            )
            node.destroy_node()
            rclpy.shutdown()
            sys.exit(1)
        logger.warn(
            f"Target {args.joint_angles_rad} is in the danger zone "
            f"(min_distance={result.min_distance:.4f}m, closest pair: "
            f"{result.link_a}, {result.link_b}) but not in outright collision -- "
            f"moving anyway. Pass --strict to refuse this too."
        )

    publisher = node.create_publisher(Float64MultiArray, args.move_topic, 10)
    # Give DDS discovery a moment to match the publisher with
    # kinova_torque_control_node's subscriber before publishing once and
    # exiting -- otherwise a fresh publisher's first message can be sent
    # before the subscription-side connection is up and silently dropped.
    time.sleep(0.3)
    publisher.publish(Float64MultiArray(data=args.joint_angles_rad))
    logger.info(f"Published move target to {args.move_topic}: {args.joint_angles_rad}")

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
