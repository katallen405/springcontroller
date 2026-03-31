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
        self.declare_parameter("joint_order", [""])  # robot-specific order
        self.declare_parameter("command_topic", "/forward_effort_controller/commands") # where to publish torques, should be specified in the config file

        self._joint_order = list(
            self.get_parameter("joint_order").get_parameter_value().string_array_value
        )
        if not self._joint_order or self._joint_order == [""]:
            self.get_logger().fatal("Parameter 'joint_order' must be set.")
            raise RuntimeError("joint_order not set")

        self.declare_parameter("torque_topic", "/virtual_spring/joint_torques")
        torque_topic = self.get_parameter("torque_topic").get_parameter_value().string_value

        command_topic = self.get_parameter("command_topic").get_parameter_value().string_value # where to publish torques

        self.springtorques_sub = self.create_subscription(JointState, torque_topic, self.springtorques_cb, 10)

        self.get_logger().info(
            f"Relaying /virtual_spring/joint_torques -> "
            f"/forward_effort_controller/commands "
            f"({len(self._joint_order)} joints: {self._joint_order})"
        )
        
        self.torque_pub = self.create_publisher(
            Float64MultiArray,
            command_topic,
            10,
        )
        

        self.get_logger().info(
            f"Relaying /virtual_spring/joint_torques -> "
            f"/forward_effort_controller/commands {self._joint_order} joints)"
        )


    def springtorques_cb(self, msg: JointState) -> None:
        name_to_effort = dict(zip(msg.name, msg.effort))
        effort = [name_to_effort.get(n, 0.0) for n in self._joint_order]
        out = Float64MultiArray()
        out.data = effort
        self.torque_pub.publish(out)


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
