#!/usr/bin/env python3
"""
orchestration_node.py

Backend node for the study control panel UI (springcontroller_ui). Wraps
the handful of operations that a browser-side roslibjs client can't safely
do as a plain 1:1 service passthrough over rosbridge: enabling/disabling
torque control (gen3_torque_control's kinova_torque_control_node) with a
safety-status check, a soft e-stop that also aborts any in-flight move,
moving to/capturing a saved "study start" joint-angle preset (via
gen3_torque_control's move_to_joint_angles, collision-gated against
virtual_spring_node's ~/check_collision_at), resolving forward kinematics
for the spring panel, and resetting springs back to a single anchor at the
arm's current tip position.

Position moves never go through ros2_kortex/kortex_bringup -- that driver
and gen3_torque_control both open independent Kortex sessions and fight
over the arm-global servoing mode if run at the same time against real
hardware (see gen3_ros2_kortex_coexistence in project memory, 2026-08-19).
"Position control" in this file means "torque disabled, arm holding
position via SINGLE_LEVEL_SERVOING," not a separate ros2_control
controller.

Everything else the UI needs (add_spring, remove_spring, direct torque-off,
live spring/status displays) is a plain 1:1 service/topic call the frontend
makes straight to gen3_torque_control / virtual_spring_node over rosbridge
-- this node only exists for the pieces that need sequencing, forward
kinematics, or local file state.

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
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy

from std_msgs.msg import String, Float64MultiArray
from std_srvs.srv import SetBool, Trigger
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PointStamped
from rcl_interfaces.srv import GetParameters
from rcl_interfaces.msg import ParameterType

from springcontroller.urdf_arm_configuration import URDFArmConfiguration
from springcontroller_interfaces.srv import AddSpring, RemoveSpring, CheckCollisionAtAngles
from springcontroller_ui_interfaces.srv import (
    EnableTorqueControl,
    GetLinkPose,
    ListLinkNames,
    MoveToJointAngles,
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

    def stamp(self) -> Optional[float]:
        return self._stamp


class StudyControlPanelNode(Node):
    def __init__(self) -> None:
        super().__init__("study_control_panel_node")

        # ------------------------------------------------------------
        # Parameters
        # ------------------------------------------------------------
        self.declare_parameter("torque_enable_service", "/gen3_torque_control/enable")
        self.declare_parameter("torque_status_topic", "/gen3_torque_control/status")
        # Position moves go through gen3_torque_control's own
        # move_to_joint_angles/move_status/abort_move, not kortex_bringup's
        # joint_trajectory_controller -- the two drivers can't run
        # simultaneously against real hardware (both open independent
        # Kortex sessions and fight over the arm-global servoing mode; see
        # gen3_ros2_kortex_coexistence in project memory). check_collision_at
        # is the same pre-flight gate test/move_to_joint_angles.py uses.
        self.declare_parameter("move_to_joint_angles_topic", "/gen3_torque_control/move_to_joint_angles")
        self.declare_parameter("move_status_topic", "/gen3_torque_control/move_status")
        self.declare_parameter("abort_move_service", "/gen3_torque_control/abort_move")
        self.declare_parameter("check_collision_service", "/virtual_spring_node/check_collision_at")
        # How long to wait for a move_status update confirming a move
        # actually started, NOT how long to wait for it to finish -- a real
        # move can take many seconds to tens of seconds, far longer than
        # rosbridge's own service-call timeout tolerates (see
        # _move_to_joint_angles). Completion is tracked by the UI's live
        # move-status display instead of this call's return.
        self.declare_parameter("move_confirm_timeout_sec", 3.0)
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
        # Prefix for virtual_spring_node's live per-spring retarget topic
        # (~/target/<name>) -- combined with reset_spring_name to retarget
        # the tip anchor spring in place (see _move_tip_spring_to_current)
        # without removing/recreating it, unlike reset_springs which does.
        self.declare_parameter("spring_target_topic_prefix", "/virtual_spring_node/target")
        self.declare_parameter("study_start_preset_path", "~/springcontroller_ui_study_start.yaml")
        self.declare_parameter("joint_state_freshness_sec", 1.0)
        self.declare_parameter("safety_status_freshness_sec", 3.0)
        self.declare_parameter("service_call_timeout_sec", 5.0)

        gp = lambda name: self.get_parameter(name).value  # noqa: E731
        self._torque_enable_service = gp("torque_enable_service")
        self._torque_status_topic = gp("torque_status_topic")
        self._move_to_joint_angles_topic = gp("move_to_joint_angles_topic")
        self._move_status_topic = gp("move_status_topic")
        self._abort_move_service = gp("abort_move_service")
        self._check_collision_service = gp("check_collision_service")
        self._move_confirm_timeout_sec = float(gp("move_confirm_timeout_sec"))
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
        self._spring_target_topic_prefix = gp("spring_target_topic_prefix")
        self._study_start_preset_path = gp("study_start_preset_path")
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
        self._move_status_freshness = _Freshness()

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
        self.create_subscription(
            String, self._move_status_topic, self._move_status_cb, 10,
            callback_group=cb_group,
        )

        # TRANSIENT_LOCAL: matches virtual_spring_node's own QoS for this
        # topic (see its target_qos) -- current target is state, not a
        # stream, and every publisher on ~/target/<name> needs matching
        # durability or DDS refuses to deliver between them. A plain
        # default-QoS publisher here triggered a live "requesting
        # incompatible QoS ... DURABILITY" warning (2026-08-19).
        latched_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )

        # ------------------------------------------------------------
        # Clients to other nodes' services/actions
        # ------------------------------------------------------------
        self._torque_enable_client = self.create_client(
            SetBool, self._torque_enable_service, callback_group=cb_group)
        self._abort_move_client = self.create_client(
            Trigger, self._abort_move_service, callback_group=cb_group)
        self._check_collision_client = self.create_client(
            CheckCollisionAtAngles, self._check_collision_service, callback_group=cb_group)
        self._add_spring_client = self.create_client(
            AddSpring, self._add_spring_service, callback_group=cb_group)
        self._remove_spring_client = self.create_client(
            RemoveSpring, self._remove_spring_service, callback_group=cb_group)
        self._spring_params_client = self.create_client(
            GetParameters, self._spring_node_get_parameters_service, callback_group=cb_group)
        self._move_command_pub = self.create_publisher(
            Float64MultiArray, self._move_to_joint_angles_topic, 10)
        self._tip_spring_target_pub = self.create_publisher(
            PointStamped, f"{self._spring_target_topic_prefix}/{self._reset_spring_name}", latched_qos)

        # ------------------------------------------------------------
        # Publisher: ~/study_start_status (latched -- late subscribers still
        # see the current defined/not-defined state without waiting for the
        # next capture).
        # ------------------------------------------------------------
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
        self.create_service(MoveToJointAngles, "~/move_to_joint_angles",
                             self._move_to_joint_angles_cb, callback_group=cb_group)
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

    def _move_status_cb(self, msg: String) -> None:
        self._move_status_freshness.update(msg.data, self._now())

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

    def _abort_move(self) -> tuple[bool, str]:
        resp = self._call_sync(self._abort_move_client, Trigger.Request(), timeout_sec=2.0)
        if resp is None:
            return False, f"{self._abort_move_service} unavailable -- is gen3_torque_node running?"
        return resp.success, resp.message

    def _move_to_joint_angles(self, angles: list[float]) -> tuple[bool, str]:
        """
        Collision-checked move via gen3_torque_control's
        move_to_joint_angles topic -- mirrors test/move_to_joint_angles.py's
        logic (this node is itself a ROS client of both services, not a
        subprocess wrapper around that script). Refuses (no move published)
        if the target is in outright collision, or if the collision check
        itself is unreachable (fails closed); warns but proceeds if only in
        the danger zone.

        Only waits for confirmation the move *started*, not for it to
        finish -- a real PlayJointTrajectory move can take many seconds to
        tens of seconds, far longer than rosbridge's own service-call
        timeout tolerates (confirmed live 2026-08-19: move_to_study_start
        timed out through rosbridge waiting for full completion). Actual
        completion is tracked separately by the UI's live move-status
        display (gen3_torque_control/move_status), not this call's return.
        """
        check_resp = self._call_sync(
            self._check_collision_client,
            CheckCollisionAtAngles.Request(joint_angles_rad=angles),
            timeout_sec=5.0,
        )
        if check_resp is None:
            return False, (
                f"{self._check_collision_service} unavailable -- is "
                "virtual_spring_node running? NOT moving (fail closed)."
            )
        if check_resp.in_collision:
            return False, (
                f"Target would be in collision (closest pair: {check_resp.link_a}, "
                f"{check_resp.link_b}, min_distance={check_resp.min_distance:.4f}m). NOT moving."
            )
        danger_note = ""
        if check_resp.in_danger:
            danger_note = (
                f" (in danger zone, min_distance={check_resp.min_distance:.4f}m, "
                f"closest pair: {check_resp.link_a}, {check_resp.link_b} -- "
                "proceeding, not outright collision)"
            )

        issue_time = self._now()
        self._move_command_pub.publish(Float64MultiArray(data=[float(a) for a in angles]))

        # Busy-wait only for a move_status update newer than issue_time --
        # any status at all, "MOVING" included, just confirms
        # gen3_torque_node actually received and started the command.
        # Bounded by move_confirm_timeout_sec (a few seconds), not the
        # move's own duration.
        deadline = time.monotonic() + self._move_confirm_timeout_sec
        while time.monotonic() < deadline:
            stamp = self._move_status_freshness.stamp()
            if stamp is not None and stamp >= issue_time:
                break
            time.sleep(0.02)
        else:
            return False, (
                f"No {self._move_status_topic} update after publishing -- is "
                f"gen3_torque_node running?{danger_note}"
            )

        status = self._move_status_freshness.value
        if status == "MOVING":
            return True, f"Move started -- watch 'move status' for completion.{danger_note}"
        if status == "SUCCEEDED":
            return True, f"Move completed.{danger_note}"
        return False, f"Move did not start: status '{status}'.{danger_note}"

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
        """
        "Position control" no longer means a separate ros2_control
        controller (see move_to_joint_angles_topic's declare_parameter
        comment) -- disabling torque already leaves the arm holding
        position via SINGLE_LEVEL_SERVOING, and that's the only state
        move_to_joint_angles/move_to_study_start can run from. Kept as its
        own named service (rather than folding into the frontend's
        disableTorque path) since it's the interlocked, UI-facing entry
        point for "make it safe to jog/move the arm."
        """
        ok, msg = self._set_torque(False)
        response.success = ok
        response.message = msg if ok else f"Torque disable failed: {msg}."
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

        ok, msg = self._set_torque(True)
        if not ok:
            response.success = False
            response.message = f"Torque enable failed: {msg}."
            return response

        response.success = True
        response.message = "Torque control enabled."
        return response

    def _soft_estop_cb(self, request, response):
        # Abort first: an in-flight move_to_joint_angles holds
        # gen3_torque_control's _kortex_lock for its whole duration
        # (deliberately, so torque-enable can't race in mid-move -- see
        # move_to_joint_angles/kinova_torque_control_node), so _set_torque
        # below would otherwise queue up behind it for as long as that
        # node's own move_timeout_sec (default 30s) instead of taking
        # effect immediately.
        abort_ok, abort_msg = self._abort_move()
        torque_ok, torque_msg = self._set_torque(False)

        response.success = torque_ok and abort_ok
        response.message = (
            f"Abort move: {'ok' if abort_ok else 'FAILED - ' + abort_msg}. "
            f"Torque control: {'disabled' if torque_ok else 'FAILED - ' + torque_msg}."
        )
        return response

    # ------------------------------------------------------------
    # Study-start preset
    # ------------------------------------------------------------

    def _require_torque_disabled(self) -> Optional[str]:
        """Returns an error message if torque control isn't confirmed off,
        or None if it's safe to proceed (used by every move-issuing
        service)."""
        now = self._now()
        if not self._torque_status_freshness.is_fresh(now, self._joint_state_freshness_sec):
            return (
                f"No recent {self._torque_status_topic} -- cannot confirm torque "
                "control is off. Is gen3_torque_node running?"
            )
        if self._torque_status_freshness.value == "ENABLED":
            return "Torque control is enabled. Turn it off before moving."
        return None

    def _move_to_study_start_cb(self, request, response):
        preset = load_preset(self._study_start_preset_path)
        if preset is None:
            response.success = False
            response.message = (
                "No study-start preset saved yet. Jog the arm to the desired "
                "pose, then use 'Set current position as study start'."
            )
            return response

        err = self._require_torque_disabled()
        if err is not None:
            response.success = False
            response.message = err
            return response

        by_name = dict(zip(preset["joint_names"], preset["joint_angles"]))
        missing = [n for n in self._joint_names if n not in by_name]
        if missing:
            response.success = False
            response.message = f"Study-start preset is missing joints: {missing}"
            return response
        angles = [float(by_name[n]) for n in self._joint_names]

        response.success, response.message = self._move_to_joint_angles(angles)
        return response

    def _move_to_joint_angles_cb(self, request, response):
        angles = list(request.joint_angles_rad)
        if len(angles) != len(self._joint_names):
            response.success = False
            response.message = (
                f"Expected {len(self._joint_names)} joint_angles_rad, got {len(angles)}."
            )
            return response

        err = self._require_torque_disabled()
        if err is not None:
            response.success = False
            response.message = err
            return response

        response.success, response.message = self._move_to_joint_angles([float(a) for a in angles])
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

        # Also move the tip spring here -- otherwise it's left targeting
        # wherever it was before, and the next torque-enable immediately
        # pulls the arm away from the study-start pose just captured.
        spring_note = self._move_tip_spring_to_current()

        response.success = True
        response.message = f"Saved current pose as study start ({len(angles)} joints). {spring_note}"
        return response

    # ------------------------------------------------------------
    # Tip spring retargeting (shared by set_current_as_study_start and
    # reset_springs -- the former retargets in place, the latter removes
    # every spring first, but both want the same "anchor at the current tip
    # position" point)
    # ------------------------------------------------------------

    def _current_tip_world_point(self) -> np.ndarray:
        self._arm.validate_link_name(self._tip_link_name)
        T = self._arm.get_link_transform(self._tip_link_name)
        return T[:3, :3] @ self._reset_spring_local_point + T[:3, 3]

    def _move_tip_spring_to_current(self) -> str:
        """
        Ensure self._reset_spring_name exists and is targeted at the
        current tip position -- retargets in place via
        ~/target/<name> (spring_target_topic_prefix) if it already exists,
        otherwise creates it fresh via AddSpring (same params
        reset_springs uses). Returns a short status string for the caller
        to fold into its own response message; never raises.
        """
        try:
            world_point = self._current_tip_world_point()
        except Exception as e:
            return f"Could not move tip spring: FK for '{self._tip_link_name}' failed: {e}."

        names = self._list_current_spring_names()
        if names is None:
            return (
                f"Could not move tip spring: {self._spring_node_get_parameters_service} "
                "unavailable -- is virtual_spring_node running?"
            )

        if self._reset_spring_name in names:
            msg = PointStamped()
            msg.header.frame_id = "world"
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.point.x, msg.point.y, msg.point.z = [float(v) for v in world_point]
            self._tip_spring_target_pub.publish(msg)
            return f"Tip spring '{self._reset_spring_name}' retargeted to current position."

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
        if add_resp is None:
            return f"Could not move tip spring: {self._add_spring_service} unavailable/timed out."
        if not add_resp.success:
            return f"Could not move tip spring: {add_resp.message}"
        return f"Tip spring '{self._reset_spring_name}' created at current position."

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
            world_point = self._current_tip_world_point()
        except Exception as e:
            response.success = False
            response.message = (
                f"Removed {len(names) - len(removal_failures)}/{len(names)} spring(s), "
                f"but FK for tip link '{self._tip_link_name}' failed: {e}. "
                "No anchor spring re-added."
            )
            return response

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
