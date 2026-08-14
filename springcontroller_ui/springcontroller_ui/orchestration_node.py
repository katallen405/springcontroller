#!/usr/bin/env python3
"""
orchestration_node.py

Backend node for the study control panel UI (springcontroller_ui). Wraps
the handful of operations that a browser-side roslibjs client can't safely
do as a plain 1:1 service passthrough over rosbridge: switching between
position control (ros2_kortex's joint_trajectory_controller) and torque
control (gen3_torque_control's kinova_torque_control_node) with the correct
disable-first ordering, a soft e-stop, moving to/capturing a saved "study
start" joint-angle preset, resolving forward kinematics for the spring
panel, and resetting springs back to a single anchor at the arm's current
tip position.

Everything else the UI needs (add_spring, remove_spring, direct torque-off,
direct position-off, live spring/status displays) is a plain 1:1
service/topic call the frontend makes straight to gen3_torque_control /
virtual_spring_node / controller_manager over rosbridge -- this node only
exists for the pieces that need sequencing, forward kinematics, or local
file state.

Concurrency note: several service callbacks here make blocking calls to
*other* services/actions and need to wait for their result without
deadlocking this node's own executor. This node must be spun with a
MultiThreadedExecutor (see main() below) and all service servers use a
ReentrantCallbackGroup, so that while one callback is busy-waiting on
`_wait_for_future`, other executor threads remain free to process the
response that future is waiting on.
"""
from __future__ import annotations

import json
import time
from typing import Optional

import numpy as np
import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy

from std_msgs.msg import String
from std_srvs.srv import SetBool, Trigger
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from controller_manager_msgs.srv import SwitchController, ListControllers
from rcl_interfaces.srv import GetParameters
from rcl_interfaces.msg import ParameterType

from springcontroller.urdf_arm_configuration import URDFArmConfiguration
from springcontroller_interfaces.srv import AddSpring, RemoveSpring
from springcontroller_ui_interfaces.srv import (
    EnableTorqueControl,
    GetLinkPose,
    ListLinkNames,
)

from springcontroller_ui.study_start_preset import load_preset, save_preset


class _Freshness:
    """Tracks the last time a cached value was received, so callers can ask
    'is this actually alive right now' instead of trusting a value that may
    be arbitrarily old (or never arrived) because the publishing process
    isn't running."""

    def __init__(self) -> None:
        self.value = None
        self._stamp: Optional[float] = None

    def update(self, value, now: float) -> None:
        self.value = value
        self._stamp = now

    def is_fresh(self, now: float, max_age_sec: float) -> bool:
        return self._stamp is not None and (now - self._stamp) <= max_age_sec


class StudyControlPanelNode(Node):
    def __init__(self) -> None:
        super().__init__("study_control_panel_node")

        # ------------------------------------------------------------
        # Parameters
        # ------------------------------------------------------------
        self.declare_parameter("torque_enable_service", "/gen3_torque_control/enable")
        self.declare_parameter("torque_status_topic", "/gen3_torque_control/status")
        self.declare_parameter("controller_manager_switch_service", "/controller_manager/switch_controller")
        self.declare_parameter("controller_manager_list_service", "/controller_manager/list_controllers")
        self.declare_parameter("position_controller_name", "joint_trajectory_controller")
        self.declare_parameter("follow_joint_trajectory_action", "/joint_trajectory_controller/follow_joint_trajectory")
        self.declare_parameter("add_spring_service", "/virtual_spring_node/add_spring")
        self.declare_parameter("remove_spring_service", "/virtual_spring_node/remove_spring")
        self.declare_parameter("spring_node_get_parameters_service", "/virtual_spring_node/get_parameters")
        self.declare_parameter("safety_status_topic", "/virtual_spring_node/safety_status")
        # /kinova/joint_states_lowlevel, not plain /joint_states: this is
        # the topic name gen3_spring.launch.py already remaps
        # virtual_spring_node's subscription to, so gen3_torque_node must be
        # launched with a matching `-r joint_states:=...` remap for that
        # wiring to work at all (see the terminal-reminder text in
        # web/index.html) -- this node listens on the same remapped name so
        # its FK/interlock state matches what virtual_spring_node itself
        # sees, not gen3_torque_node's unremapped default.
        self.declare_parameter("joint_states_topic", "/kinova/joint_states_lowlevel")
        self.declare_parameter("joint_names", [
            "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "joint_7",
        ])
        self.declare_parameter("urdf_path", "/home/katallen/sandbox/src/springcontroller/springcontroller/flat_urdf_files/gen3_kinova_flat.urdf")
        self.declare_parameter("locked_joint_names", [
            "robotiq_85_left_knuckle_joint",
            "robotiq_85_right_knuckle_joint",
            "robotiq_85_left_inner_knuckle_joint",
            "robotiq_85_right_inner_knuckle_joint",
            "robotiq_85_left_finger_tip_joint",
            "robotiq_85_right_finger_tip_joint",
        ])
        self.declare_parameter("tip_link_name", "end_effector_link")
        self.declare_parameter("reset_spring_name", "reset_tip_spring")
        self.declare_parameter("reset_spring_stiffness", 50.0)
        self.declare_parameter("reset_spring_damping", 5.0)
        self.declare_parameter("reset_spring_local_point", [0.0, 0.0, 0.1])
        self.declare_parameter("study_start_preset_path", "~/springcontroller_ui_study_start.yaml")
        self.declare_parameter("move_duration_sec", 8.0)
        self.declare_parameter("joint_state_freshness_sec", 1.0)
        self.declare_parameter("safety_status_freshness_sec", 3.0)
        self.declare_parameter("service_call_timeout_sec", 5.0)

        gp = lambda name: self.get_parameter(name).value  # noqa: E731
        self._torque_enable_service = gp("torque_enable_service")
        self._torque_status_topic = gp("torque_status_topic")
        self._switch_controller_service = gp("controller_manager_switch_service")
        self._list_controllers_service = gp("controller_manager_list_service")
        self._position_controller_name = gp("position_controller_name")
        self._follow_joint_trajectory_action = gp("follow_joint_trajectory_action")
        self._add_spring_service = gp("add_spring_service")
        self._remove_spring_service = gp("remove_spring_service")
        self._spring_node_get_parameters_service = gp("spring_node_get_parameters_service")
        self._safety_status_topic = gp("safety_status_topic")
        self._joint_states_topic = gp("joint_states_topic")
        self._joint_names = list(gp("joint_names"))
        urdf_path = gp("urdf_path")
        locked_joint_names = list(gp("locked_joint_names"))
        self._tip_link_name = gp("tip_link_name")
        self._reset_spring_name = gp("reset_spring_name")
        self._reset_spring_stiffness = float(gp("reset_spring_stiffness"))
        self._reset_spring_damping = float(gp("reset_spring_damping"))
        self._reset_spring_local_point = np.array(gp("reset_spring_local_point"), dtype=float)
        self._study_start_preset_path = gp("study_start_preset_path")
        self._move_duration_sec = float(gp("move_duration_sec"))
        self._joint_state_freshness_sec = float(gp("joint_state_freshness_sec"))
        self._safety_status_freshness_sec = float(gp("safety_status_freshness_sec"))
        self._service_call_timeout_sec = float(gp("service_call_timeout_sec"))

        # ------------------------------------------------------------
        # FK model (kinematics only -- no SRDF/collision needed here)
        # ------------------------------------------------------------
        self._arm = URDFArmConfiguration.from_urdf(
            urdf_path, srdf_path="", locked_joint_names=locked_joint_names,
        )
        self._joint_order: Optional[list[int]] = None  # built on first /joint_states msg

        # ------------------------------------------------------------
        # Cached live state
        # ------------------------------------------------------------
        self._joint_state_freshness = _Freshness()
        self._torque_status_freshness = _Freshness()
        self._safety_status_freshness = _Freshness()

        cb_group = ReentrantCallbackGroup()

        self.create_subscription(
            JointState, self._joint_states_topic, self._joint_state_cb, 10,
            callback_group=cb_group,
        )
        self.create_subscription(
            String, self._torque_status_topic, self._torque_status_cb, 10,
            callback_group=cb_group,
        )
        self.create_subscription(
            String, self._safety_status_topic, self._safety_status_cb, 10,
            callback_group=cb_group,
        )

        # ------------------------------------------------------------
        # Clients to other nodes' services/actions
        # ------------------------------------------------------------
        self._torque_enable_client = self.create_client(
            SetBool, self._torque_enable_service, callback_group=cb_group)
        self._switch_controller_client = self.create_client(
            SwitchController, self._switch_controller_service, callback_group=cb_group)
        self._list_controllers_client = self.create_client(
            ListControllers, self._list_controllers_service, callback_group=cb_group)
        self._add_spring_client = self.create_client(
            AddSpring, self._add_spring_service, callback_group=cb_group)
        self._remove_spring_client = self.create_client(
            RemoveSpring, self._remove_spring_service, callback_group=cb_group)
        self._spring_params_client = self.create_client(
            GetParameters, self._spring_node_get_parameters_service, callback_group=cb_group)
        self._trajectory_action_client = ActionClient(
            self, FollowJointTrajectory, self._follow_joint_trajectory_action,
            callback_group=cb_group)

        # ------------------------------------------------------------
        # Publisher: ~/study_start_status (latched -- late subscribers still
        # see the current defined/not-defined state without waiting for the
        # next capture).
        # ------------------------------------------------------------
        latched_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self._study_start_status_pub = self.create_publisher(
            String, "~/study_start_status", latched_qos)
        self._publish_study_start_status()

        # ------------------------------------------------------------
        # Services this node exposes
        # ------------------------------------------------------------
        self.create_service(Trigger, "~/enable_position_control",
                             self._enable_position_control_cb, callback_group=cb_group)
        self.create_service(EnableTorqueControl, "~/enable_torque_control",
                             self._enable_torque_control_cb, callback_group=cb_group)
        self.create_service(Trigger, "~/soft_estop",
                             self._soft_estop_cb, callback_group=cb_group)
        self.create_service(Trigger, "~/move_to_study_start",
                             self._move_to_study_start_cb, callback_group=cb_group)
        self.create_service(Trigger, "~/set_current_as_study_start",
                             self._set_current_as_study_start_cb, callback_group=cb_group)
        self.create_service(Trigger, "~/reset_springs",
                             self._reset_springs_cb, callback_group=cb_group)
        self.create_service(GetLinkPose, "~/get_link_pose",
                             self._get_link_pose_cb, callback_group=cb_group)
        self.create_service(ListLinkNames, "~/list_link_names",
                             self._list_link_names_cb, callback_group=cb_group)

        self.get_logger().info("study_control_panel_node ready.")

    # ------------------------------------------------------------
    # Small helpers
    # ------------------------------------------------------------

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9

    def _wait_for_future(self, future, timeout_sec: float):
        """Busy-wait for an rclpy Future without spinning this node
        ourselves -- safe under a MultiThreadedExecutor because other
        executor threads keep servicing the callbacks that will complete
        this future while the current thread just sleeps and polls."""
        start = time.monotonic()
        while not future.done():
            if time.monotonic() - start > timeout_sec:
                return None
            time.sleep(0.005)
        try:
            return future.result()
        except Exception:
            return None

    def _call_sync(self, client, request, timeout_sec: Optional[float] = None):
        """Call a service and block (see _wait_for_future) until it
        completes or times out. Returns None on unavailable/timeout/error --
        callers turn that into an 'is X running?' message."""
        timeout_sec = self._service_call_timeout_sec if timeout_sec is None else timeout_sec
        if not client.wait_for_service(timeout_sec=1.0):
            return None
        future = client.call_async(request)
        return self._wait_for_future(future, timeout_sec)

    # ------------------------------------------------------------
    # Subscription callbacks
    # ------------------------------------------------------------

    def _joint_state_cb(self, msg: JointState) -> None:
        now = self._now()
        self._joint_state_freshness.update(msg, now)

        if self._joint_order is None:
            pinocchio_names = self._arm.joint_names
            ros_names = list(msg.name)
            try:
                self._joint_order = [ros_names.index(n) for n in pinocchio_names]
            except ValueError as e:
                self.get_logger().error(
                    f"Joint name mismatch between {self._joint_states_topic} and "
                    f"the URDF: {e}. ROS names: {ros_names}, URDF names: {pinocchio_names}"
                )
                return

        if len(msg.position) != self._arm.n_dof:
            return
        pos = np.array(msg.position)
        vel = np.array(msg.velocity) if msg.velocity else np.zeros(self._arm.n_dof)
        self._arm.update_from_angles(pos[self._joint_order], vel[self._joint_order])

    def _torque_status_cb(self, msg: String) -> None:
        self._torque_status_freshness.update(msg.data, self._now())

    def _safety_status_cb(self, msg: String) -> None:
        self._safety_status_freshness.update(msg.data, self._now())

    # ------------------------------------------------------------
    # Publish helper
    # ------------------------------------------------------------

    def _publish_study_start_status(self) -> None:
        preset = load_preset(self._study_start_preset_path)
        payload = {
            "defined": preset is not None,
            "joint_angles": preset["joint_angles"] if preset else None,
            "path": self._study_start_preset_path,
            "n_joints": len(self._joint_names),
        }
        self._study_start_status_pub.publish(String(data=json.dumps(payload)))

    # ------------------------------------------------------------
    # Shared low-level actions used by the interlocked services
    # ------------------------------------------------------------

    def _set_torque(self, data: bool) -> tuple[bool, str]:
        req = SetBool.Request()
        req.data = data
        resp = self._call_sync(self._torque_enable_client, req)
        if resp is None:
            return False, f"{self._torque_enable_service} unavailable -- is gen3_torque_node running?"
        return resp.success, resp.message

    def _switch_controller(self, activate: list[str], deactivate: list[str]) -> tuple[bool, str]:
        req = SwitchController.Request()
        req.activate_controllers = activate
        req.deactivate_controllers = deactivate
        req.strictness = SwitchController.Request.STRICT
        resp = self._call_sync(self._switch_controller_client, req)
        if resp is None:
            return False, (
                f"{self._switch_controller_service} unavailable -- is the "
                "Kinova driver (gen3.launch.py) running?"
            )
        return bool(resp.ok), resp.message

    def _list_current_spring_names(self) -> Optional[list[str]]:
        """Live query of virtual_spring_node's spring_names/joint_spring_names
        parameters (not a cached topic value -- ~/springs_updated is only
        published on add/remove, so a cache could easily be stale/empty just
        because this node started after the last change). Returns None if
        the parameter service is unavailable (virtual_spring_node not
        running)."""
        req = GetParameters.Request()
        req.names = ["spring_names", "joint_spring_names"]
        resp = self._call_sync(self._spring_params_client, req, timeout_sec=2.0)
        if resp is None:
            return None
        names: list[str] = []
        for value in resp.values:
            if value.type == ParameterType.PARAMETER_STRING_ARRAY:
                names.extend(n for n in value.string_array_value if n)
        return names

    # ------------------------------------------------------------
    # Interlocked services
    # ------------------------------------------------------------

    def _enable_position_control_cb(self, request, response):
        ok, msg = self._set_torque(False)
        if not ok:
            response.success = False
            response.message = f"Torque disable failed: {msg}. NOT activating position controller."
            return response

        ok, msg = self._switch_controller([self._position_controller_name], [])
        if not ok:
            response.success = False
            response.message = (
                f"Torque control has been disabled. Activating "
                f"'{self._position_controller_name}' failed: {msg}"
            )
            return response

        response.success = True
        response.message = "Position control enabled."
        return response

    def _enable_torque_control_cb(self, request, response):
        now = self._now()
        if not self._safety_status_freshness.is_fresh(now, self._safety_status_freshness_sec):
            response.success = False
            response.message = (
                f"No recent {self._safety_status_topic} -- is virtual_spring_node running?"
            )
            return response

        status = self._safety_status_freshness.value or ""
        if not status.startswith("SAFE") and not request.allow_danger:
            response.success = False
            response.message = (
                f"safety_status reports '{status}' -- refusing to enable torque "
                "control (check 'allow danger' to override)."
            )
            return response

        ok, msg = self._switch_controller([], [self._position_controller_name])
        if not ok:
            response.success = False
            response.message = f"Deactivating '{self._position_controller_name}' failed: {msg}"
            return response

        ok, msg = self._set_torque(True)
        if not ok:
            response.success = False
            response.message = (
                f"Position controller deactivated, but torque enable failed: {msg}. "
                "Arm has no active controller -- press 'Turn on position control' to recover."
            )
            return response

        response.success = True
        response.message = "Torque control enabled."
        return response

    def _soft_estop_cb(self, request, response):
        torque_ok, torque_msg = self._set_torque(False)
        controller_ok, controller_msg = self._switch_controller([], [self._position_controller_name])

        response.success = torque_ok
        response.message = (
            f"Torque control: {'disabled' if torque_ok else 'FAILED - ' + torque_msg}. "
            f"Position controller: {'deactivated' if controller_ok else 'FAILED - ' + controller_msg}."
        )
        return response

    # ------------------------------------------------------------
    # Study-start preset
    # ------------------------------------------------------------

    def _controller_is_active(self, name: str) -> Optional[bool]:
        resp = self._call_sync(self._list_controllers_client, ListControllers.Request(), timeout_sec=2.0)
        if resp is None:
            return None
        for c in resp.controller:
            if c.name == name:
                return c.state == "active"
        return False

    def _move_to_study_start_cb(self, request, response):
        now = self._now()
        preset = load_preset(self._study_start_preset_path)
        if preset is None:
            response.success = False
            response.message = (
                "No study-start preset saved yet. Jog the arm to the desired "
                "pose, then use 'Set current position as study start'."
            )
            return response

        position_active = self._controller_is_active(self._position_controller_name)
        if position_active is None:
            response.success = False
            response.message = (
                f"{self._list_controllers_service} unavailable -- is the Kinova "
                "driver (gen3.launch.py) running?"
            )
            return response
        if not position_active:
            response.success = False
            response.message = "Position control is not active. Press 'Turn on position control' first."
            return response

        if not self._torque_status_freshness.is_fresh(now, self._joint_state_freshness_sec):
            response.success = False
            response.message = (
                f"No recent {self._torque_status_topic} -- cannot confirm torque "
                "control is off. Is gen3_torque_node running?"
            )
            return response
        if self._torque_status_freshness.value == "ENABLED":
            response.success = False
            response.message = "Torque control is enabled. Turn it off before moving via position control."
            return response

        if not self._trajectory_action_client.wait_for_server(timeout_sec=5.0):
            response.success = False
            response.message = (
                f"{self._follow_joint_trajectory_action} action server unavailable -- "
                "is the Kinova driver running and joint_trajectory_controller active?"
            )
            return response

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = JointTrajectory()
        goal.trajectory.joint_names = list(preset["joint_names"])
        point = JointTrajectoryPoint()
        point.positions = [float(a) for a in preset["joint_angles"]]
        duration_sec = self._move_duration_sec
        point.time_from_start = Duration(
            sec=int(duration_sec), nanosec=int((duration_sec % 1.0) * 1e9))
        goal.trajectory.points = [point]

        send_goal_future = self._trajectory_action_client.send_goal_async(goal)
        goal_handle = self._wait_for_future(send_goal_future, timeout_sec=5.0)
        if goal_handle is None or not goal_handle.accepted:
            response.success = False
            response.message = "Trajectory goal was rejected or timed out being sent."
            return response

        result_future = goal_handle.get_result_async()
        result_wrapper = self._wait_for_future(result_future, timeout_sec=duration_sec + 5.0)
        if result_wrapper is None:
            response.success = False
            response.message = "Timed out waiting for the move to complete."
            return response

        if result_wrapper.result.error_code != FollowJointTrajectory.Result.SUCCESSFUL:
            response.success = False
            response.message = f"Move failed: {result_wrapper.result.error_string}"
            return response

        response.success = True
        response.message = "Moved to study start."
        return response

    def _set_current_as_study_start_cb(self, request, response):
        now = self._now()
        if not self._joint_state_freshness.is_fresh(now, self._joint_state_freshness_sec):
            response.success = False
            response.message = f"No recent {self._joint_states_topic} -- is gen3_torque_node running?"
            return response

        msg: JointState = self._joint_state_freshness.value
        by_name = dict(zip(msg.name, msg.position))
        missing = [n for n in self._joint_names if n not in by_name]
        if missing:
            response.success = False
            response.message = f"Missing joints in {self._joint_states_topic}: {missing}"
            return response

        angles = [by_name[n] for n in self._joint_names]
        save_preset(self._study_start_preset_path, self._joint_names, angles)
        self._publish_study_start_status()

        response.success = True
        response.message = f"Saved current pose as study start ({len(angles)} joints)."
        return response

    # ------------------------------------------------------------
    # Reset springs
    # ------------------------------------------------------------

    def _reset_springs_cb(self, request, response):
        now = self._now()
        if not self._joint_state_freshness.is_fresh(now, self._joint_state_freshness_sec):
            response.success = False
            response.message = (
                f"No recent {self._joint_states_topic} -- cannot compute current "
                "tip position. Springs left unchanged."
            )
            return response

        names = self._list_current_spring_names()
        if names is None:
            response.success = False
            response.message = (
                f"{self._spring_node_get_parameters_service} unavailable -- is "
                "gen3_spring.launch.py (virtual_spring_node) running?"
            )
            return response

        removal_failures = []
        for name in names:
            req = RemoveSpring.Request()
            req.name = name
            resp = self._call_sync(self._remove_spring_client, req)
            if resp is None:
                removal_failures.append(f"{name} (service call failed/timed out)")
            elif not resp.success:
                removal_failures.append(f"{name} ({resp.message})")

        try:
            self._arm.validate_link_name(self._tip_link_name)
            T = self._arm.get_link_transform(self._tip_link_name)
        except Exception as e:
            response.success = False
            response.message = (
                f"Removed {len(names) - len(removal_failures)}/{len(names)} spring(s), "
                f"but FK for tip link '{self._tip_link_name}' failed: {e}. "
                "No anchor spring re-added."
            )
            return response

        world_point = T[:3, :3] @ self._reset_spring_local_point + T[:3, 3]

        add_req = AddSpring.Request()
        add_req.name = self._reset_spring_name
        add_req.link_name = self._tip_link_name
        add_req.stiffness = self._reset_spring_stiffness
        add_req.damping = self._reset_spring_damping
        add_req.rest_length = 0.0
        add_req.local_point = [float(v) for v in self._reset_spring_local_point]
        add_req.target = [float(v) for v in world_point]
        add_req.inner_radius = 0.0
        add_req.outer_radius = 0.0
        add_resp = self._call_sync(self._add_spring_client, add_req)

        parts = []
        if removal_failures:
            parts.append(f"Failed to remove: {removal_failures}.")
        else:
            parts.append(f"Removed {len(names)} existing spring(s).")

        if add_resp is None:
            parts.append(f"{self._add_spring_service} unavailable/timed out -- anchor spring NOT added.")
            response.success = False
        else:
            parts.append(
                f"Anchor spring '{self._reset_spring_name}': "
                f"{'added' if add_resp.success else 'FAILED - ' + add_resp.message}."
            )
            response.success = add_resp.success and not removal_failures

        response.message = " ".join(parts)
        return response

    # ------------------------------------------------------------
    # FK helpers for the spring panel
    # ------------------------------------------------------------

    def _get_link_pose_cb(self, request, response):
        now = self._now()
        if not self._joint_state_freshness.is_fresh(now, self._joint_state_freshness_sec):
            response.success = False
            response.message = f"No recent {self._joint_states_topic} -- is gen3_torque_node running?"
            response.point = [0.0, 0.0, 0.0]
            return response

        try:
            self._arm.validate_link_name(request.link_name)
        except ValueError as e:
            response.success = False
            response.message = str(e)
            response.point = [0.0, 0.0, 0.0]
            return response

        T = self._arm.get_link_transform(request.link_name)
        response.success = True
        response.message = "ok"
        response.point = [float(v) for v in T[:3, 3]]
        return response

    def _list_link_names_cb(self, request, response):
        response.success = True
        response.message = "ok"
        response.link_names = sorted(self._arm.link_names)
        return response


def main(args=None):
    rclpy.init(args=args)
    node = StudyControlPanelNode()
    executor = MultiThreadedExecutor(num_threads=8)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
