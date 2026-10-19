#!/usr/bin/env python3
"""
hande_gripper_node.py

ROS 2 node for the Robotiq Hand-E on the UR3e. It talks to the gripper
through the Robotiq URCap's socket server on the robot controller (port
63352), so it works alongside External Control and the UR driver as they
are -- no tool-communication relay (/tmp/ttyUR) or gripper ros2_control
hardware needed.

The open/close service mirrors gen3_torque_control's gripper_command
(std_srvs/SetBool, data=true closes, data=false opens), and like Kortex's
it's fire-and-forget: the response comes back once the gripper has
acknowledged the move, not when the jaws stop. Watch ~/state for that.
To drive this node from the study UI, which calls
/gen3_torque_control/gripper_command, remap the service:

    ros2 run springcontroller hande_gripper_node --ros-args \
        -r ~/gripper_command:=/gen3_torque_control/gripper_command

Activation
----------
The gripper must be activated after every power-up, and activating it
moves the jaws. That's done on the first open/close command (or by
calling ~/activate, or at startup with activate_on_start:=true), never
silently in the background. Keep hands clear the first time.

Publications
------------
~/state   (std_msgs/String, JSON)
    Polled at state_rate Hz: connected, active, position (0-255, 0 is
    open), requested_position, object ("moving", "outer_grip",
    "inner_grip", "at_dest", or null until activated), and fault (0 is
    no fault).

Services
--------
~/gripper_command   (std_srvs/SetBool)
    true closes, false opens, at the configured speed and force.
    Activates the gripper first if needed.
~/activate          (std_srvs/Trigger)
    Activate (and clear faults). Moves the jaws.

Parameters
----------
robot_ip : str
    UR controller address (default 130.64.17.95).
port : int
    URCap gripper socket port (default 63352).
speed / force : int
    0-255, used for open/close (defaults 255 / 100).
state_rate : double
    Hz for polling and publishing ~/state (default 10.0).
activate_on_start : bool
    Activate as soon as the node connects (default false).
auto_calibrate : bool
    After activating, sweep the jaws fully closed and open to learn the
    real position range (default false). Fails if anything is between
    the jaws.
"""

from __future__ import annotations

import json

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import SetBool, Trigger

from springcontroller.robotiq_gripper import RobotiqGripper


OBJECT_NAMES = {
    RobotiqGripper.ObjectStatus.MOVING: "moving",
    RobotiqGripper.ObjectStatus.STOPPED_OUTER_OBJECT: "outer_grip",
    RobotiqGripper.ObjectStatus.STOPPED_INNER_OBJECT: "inner_grip",
    RobotiqGripper.ObjectStatus.AT_DEST: "at_dest",
}

# Seconds between reconnect attempts while the gripper socket is down.
RECONNECT_PERIOD = 2.0


class HandEGripperNode(Node):

    def __init__(self):
        super().__init__("hande_gripper")

        self.robot_ip = self.declare_parameter("robot_ip", "130.64.17.95").value
        self.port = self.declare_parameter("port", 63352).value
        self.speed = self.declare_parameter("speed", 255).value
        self.force = self.declare_parameter("force", 100).value
        state_rate = self.declare_parameter("state_rate", 10.0).value
        self.activate_on_start = self.declare_parameter("activate_on_start", False).value
        self.auto_calibrate = self.declare_parameter("auto_calibrate", False).value

        self.gripper: RobotiqGripper | None = None
        self.last_connect_attempt = None

        self.state_pub = self.create_publisher(String, "~/state", 10)
        self.create_service(SetBool, "~/gripper_command", self.on_gripper_command)
        self.create_service(Trigger, "~/activate", self.on_activate)
        self.create_timer(1.0 / state_rate, self.publish_state)

        self.try_connect()
        if self.gripper is not None and self.activate_on_start:
            ok, msg = self.activate()
            (self.get_logger().info if ok else self.get_logger().error)(msg)

    # -- connection ----------------------------------------------------------

    def try_connect(self) -> None:
        self.last_connect_attempt = self.get_clock().now()
        gripper = RobotiqGripper()
        try:
            gripper.connect(self.robot_ip, self.port)
            gripper.get_current_position()  # confirm the URCap answers
        except (OSError, ValueError, RuntimeError) as e:
            self.get_logger().warn(
                f"Can't reach the gripper at {self.robot_ip}:{self.port} ({e}); "
                f"retrying every {RECONNECT_PERIOD:.0f} s. Is the robot on, and "
                "the Robotiq URCap installed?",
                throttle_duration_sec=30.0,
            )
            if gripper.socket is not None:
                gripper.socket.close()
            return
        self.gripper = gripper
        self.get_logger().info(f"Connected to the gripper at {self.robot_ip}:{self.port}")

    def drop_connection(self, e: Exception) -> None:
        self.get_logger().error(f"Lost the gripper connection: {e}")
        try:
            self.gripper.disconnect()
        except OSError:
            pass
        self.gripper = None

    # -- gripper actions -----------------------------------------------------

    def activate(self) -> tuple[bool, str]:
        if self.gripper is None:
            return False, "Gripper not connected"
        try:
            self.gripper.activate(auto_calibrate=self.auto_calibrate)
        except (OSError, ValueError) as e:
            self.drop_connection(e)
            return False, f"Activation failed: {e}"
        except RuntimeError as e:  # calibration blocked, or lost power
            return False, f"Activation failed: {e}"
        lo, hi = self.gripper.get_min_position(), self.gripper.get_max_position()
        return True, f"Gripper activated (position range {lo}-{hi})"

    def on_activate(self, request, response):
        response.success, response.message = self.activate()
        return response

    def on_gripper_command(self, request, response):
        close = request.data
        word = "close" if close else "open"
        if self.gripper is None:
            response.success, response.message = False, f"Can't {word}: gripper not connected"
            return response
        try:
            if not self.gripper.is_active():
                self.get_logger().info("Gripper not active; activating (the jaws will move)")
                ok, msg = self.activate()
                if not ok:
                    response.success, response.message = False, msg
                    return response
            target = (self.gripper.get_closed_position() if close
                      else self.gripper.get_open_position())
            acked, _ = self.gripper.move(target, self.speed, self.force)
        except (OSError, ValueError, RuntimeError) as e:
            self.drop_connection(e)
            response.success, response.message = False, f"Can't {word}: {e}"
            return response
        response.success = acked
        response.message = (f"Gripper {word} command sent" if acked
                            else f"Gripper didn't acknowledge the {word} command")
        return response

    # -- state ---------------------------------------------------------------

    def publish_state(self) -> None:
        if self.gripper is None:
            elapsed = (self.get_clock().now() - self.last_connect_attempt).nanoseconds * 1e-9
            if elapsed >= RECONNECT_PERIOD:
                self.try_connect()
        state = {"connected": self.gripper is not None}
        if self.gripper is not None:
            g = self.gripper
            try:
                active = g.is_active()
                state.update(
                    active=active,
                    position=g.get_current_position(),
                    requested_position=g._get_var(g.PRE),
                    # OBJ reads 0 ("moving") until activation; don't report that
                    object=(OBJECT_NAMES[RobotiqGripper.ObjectStatus(g._get_var(g.OBJ))]
                            if active else None),
                    fault=g._get_var(g.FLT),
                )
            except (OSError, ValueError, RuntimeError) as e:
                self.drop_connection(e)
                state = {"connected": False}
        self.state_pub.publish(String(data=json.dumps(state)))

    def destroy_node(self):
        if self.gripper is not None:
            self.gripper.disconnect()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = HandEGripperNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
