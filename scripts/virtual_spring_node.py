#!/usr/bin/env python3
"""
virtual_spring_node.py

ROS 2 node that maintains a collection of virtual springs and publishes
the resulting joint torques on each JointState update from the arm.

Subscriptions
-------------
~/joint_states  (sensor_msgs/JointState)
    Current joint positions and velocities from the arm.

~/spring_targets  (geometry_msgs/PointStamped)  [one per spring, by topic suffix]
    Optionally update a spring's target at runtime. Topic name encodes the
    spring name, e.g. /virtual_spring_node/target/tip_spring.

Publications
------------
~/joint_torques  (sensor_msgs/JointState)
    Effort field carries the summed virtual-spring torques. Position and
    velocity fields are left empty.

Services
--------
~/enable   (std_srvs/SetBool)
    Enable or disable all springs at once.

Parameters
----------
urdf_path       : str   — path to the robot URDF
spring_configs  : list  — YAML list of spring definitions (see config/springs.yaml)
"""

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter

import numpy as np
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PointStamped
from std_srvs.srv import SetBool

from virtual_spring_ros2.virtual_spring import VirtualSpring, SpringCollection
from virtual_spring_ros2.urdf_arm_configuration import URDFArmConfiguration


class VirtualSpringNode(Node):

    def __init__(self):
        super().__init__("virtual_spring_node")

        # ── Parameters ──────────────────────────────────────────────────────
        self.declare_parameter("urdf_path", "")
        self.declare_parameter("publish_rate_hz", 100.0)

        urdf_path = self.get_parameter("urdf_path").get_parameter_value().string_value
        if not urdf_path:
            self.get_logger().fatal("Parameter 'urdf_path' must be set.")
            raise RuntimeError("urdf_path not set")

        # ── Kinematics ──────────────────────────────────────────────────────
        self._arm: URDFArmConfiguration | None = None
        self._urdf_path = urdf_path
        self._latest_q: np.ndarray | None = None
        self._latest_qdot: np.ndarray | None = None

        # ── Spring collection ────────────────────────────────────────────────
        self._springs = SpringCollection()
        self._load_springs_from_params()

        # ── Publishers ───────────────────────────────────────────────────────
        self._torque_pub = self.create_publisher(
            JointState, "~/joint_torques", 10
        )

        # ── Subscriptions ────────────────────────────────────────────────────
        self._js_sub = self.create_subscription(
            JointState, "~/joint_states", self._joint_state_cb, 10
        )

        # Per-spring target update subscriptions
        for spring in self._springs:
            topic = f"~/target/{spring.name}"
            self.create_subscription(
                PointStamped,
                topic,
                lambda msg, s=spring: self._target_cb(msg, s),
                10,
            )
            self.get_logger().info(f"Listening for target updates on {topic}")

        # ── Services ─────────────────────────────────────────────────────────
        self._enable_srv = self.create_service(
            SetBool, "~/enable", self._enable_cb
        )

        self.get_logger().info(
            f"VirtualSpringNode ready. {len(self._springs)} spring(s) loaded."
        )

    # ── Parameter-driven spring loading ─────────────────────────────────────

    def _load_springs_from_params(self) -> None:
        """
        Read spring definitions from ROS parameters.
        Each spring is a parameter namespace, e.g.:

            springs.tip_spring.link_name: "tool0"
            springs.tip_spring.local_point: [0.0, 0.0, 0.1]
            springs.tip_spring.target: [0.5, 0.0, 0.8]
            springs.tip_spring.stiffness: 120.0
            springs.tip_spring.damping: 8.0
            springs.tip_spring.rest_length: 0.0
        """
        # Declare a wildcard-style parameter to detect spring names
        self.declare_parameter("spring_names", [""])
        spring_names = (
            self.get_parameter("spring_names")
            .get_parameter_value()
            .string_array_value
        )

        for name in spring_names:
            if not name:
                continue
            prefix = f"springs.{name}"
            for p in [
                f"{prefix}.link_name",
                f"{prefix}.local_point",
                f"{prefix}.target",
                f"{prefix}.stiffness",
                f"{prefix}.damping",
                f"{prefix}.rest_length",
            ]:
                self.declare_parameter(p, Parameter.Type.NOT_SET)

            def _get(key, default=None):
                val = self.get_parameter(f"{prefix}.{key}")
                if val.type_ == Parameter.Type.NOT_SET:
                    return default
                return val.value

            spring = VirtualSpring(
                link_name=_get("link_name"),
                local_attachment_point=np.array(_get("local_point", [0, 0, 0])),
                target_world_point=np.array(_get("target", [0, 0, 0])),
                stiffness=float(_get("stiffness", 0.0)),
                damping=float(_get("damping", 0.0)),
                rest_length=float(_get("rest_length", 0.0)),
                name=name,
            )
            self._springs.add(spring)
            self.get_logger().info(f"Loaded spring: {spring}")

    # ── Callbacks ────────────────────────────────────────────────────────────

    def _joint_state_cb(self, msg: JointState) -> None:
        q = np.array(msg.position)
        qdot = np.array(msg.velocity) if msg.velocity else np.zeros(len(q))

        # Lazy-init arm configuration on first message
        if self._arm is None:
            try:
                self._arm = URDFArmConfiguration(self._urdf_path, q, qdot)
            except Exception as e:
                self.get_logger().error(f"Failed to init arm kinematics: {e}")
                return
        else:
            self._arm.update(q, qdot)

        # Compute and publish torques
        try:
            torques = self._springs.compute_total_torques(self._arm)
        except Exception as e:
            self.get_logger().error(f"Spring torque computation failed: {e}")
            return

        out = JointState()
        out.header.stamp = self.get_clock().now().to_msg()
        out.name = list(msg.name)
        out.effort = torques.tolist()
        self._torque_pub.publish(out)

    def _target_cb(self, msg: PointStamped, spring: VirtualSpring) -> None:
        p = msg.point
        spring.move_target(np.array([p.x, p.y, p.z]))
        self.get_logger().debug(
            f"Updated target for '{spring.name}': [{p.x:.3f}, {p.y:.3f}, {p.z:.3f}]"
        )

    def _enable_cb(
        self, request: SetBool.Request, response: SetBool.Response
    ) -> SetBool.Response:
        for spring in self._springs:
            spring.enabled = request.data
        state = "enabled" if request.data else "disabled"
        response.success = True
        response.message = f"All springs {state}."
        self.get_logger().info(response.message)
        return response


# ── Entry point ───────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = VirtualSpringNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
