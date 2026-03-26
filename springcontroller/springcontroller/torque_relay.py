#!/usr/bin/env python3
"""
torque_relay.py

Subscribes to /virtual_spring/joint_torques (sensor_msgs/JointState)
and republishes the effort field as a Float64MultiArray to
/forward_effort_controller/commands.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray


class TorqueRelay(Node):
    def __init__(self):
        super().__init__("torque_relay")

        self.declare_parameter("n_joints", 7)
        self._n_joints = self.get_parameter("n_joints").value

        self._pub = self.create_publisher(
            Float64MultiArray,
            "/forward_effort_controller/commands",
            10,
        )
        self._sub = self.create_subscription(
            JointState,
            "/virtual_spring/joint_torques",
            self._cb,
            10,
        )
        self.get_logger().info(
            f"Relaying /virtual_spring/joint_torques -> "
            f"/forward_effort_controller/commands ({self._n_joints} joints)"
        )

    def _cb(self, msg: JointState) -> None:
        effort = list(msg.effort[:self._n_joints])

        # Pad with zeros if fewer torques than expected
        if len(effort) < self._n_joints:
            effort += [0.0] * (self._n_joints - len(effort))

        out = Float64MultiArray()
        out.data = effort
        self._pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = TorqueRelay()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()