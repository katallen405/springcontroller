"""
trajectory_action.py
────────────────────
ROS glue between a control_msgs/FollowJointTrajectory action server and
trajectory_spring.TrajectoryFollower, for virtual_spring_node.

Threading: the action server runs in its own (reentrant) callback group on a
MultiThreadedExecutor, while the arm model is owned by the node's
/joint_states callback. So the action callbacks never touch the arm: they
queue a command (start / cancel / release) and wait. tick(), called from the
control callback once per cycle, carries the command out, advances the
follower and returns the tracking spring's torque.

Hand-over with the node's own Cartesian springs on the same link (e.g. a
tip-hold spring): while a trajectory runs they are faded out (stiffness and
damping scaled by 1 − mix) as the tracking spring fades in, and on release
they are re-anchored at the arm's current position before fading back, so
nothing snaps.
"""

from __future__ import annotations

import json
import threading
from typing import Optional

import numpy as np
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from std_msgs.msg import String
from std_srvs.srv import Trigger
from trajectory_msgs.msg import JointTrajectoryPoint

from springcontroller.trajectory_spring import (
    INVALID_GOAL, INVALID_JOINTS, PATH_TOLERANCE_VIOLATED, SUCCESSFUL, FollowerConfig,
    FollowerOutcome, JointTrajectoryReference, TrackingPoseSpring, TrajectoryFollower,
    wrap_angle,
)
from springcontroller.virtual_spring import VirtualSpring

DEFAULT_LINKS = ("end_effector_link", "j2s7s300_ee_link", "tool0")


def _seconds(d) -> float:
    return d.sec + d.nanosec * 1e-9


class _Command:
    def __init__(self, kind: str, **kw) -> None:
        self.kind = kind
        self.kw = kw
        self.done = threading.Event()
        self.ok = False
        self.message = ""


class TrajectoryActionGlue:

    def __init__(self, node, arm, springs) -> None:
        self._node = node
        self._arm = arm
        self._springs = springs          # the node's SpringCollection
        log = node.get_logger()

        p = self._declare
        link = p("trajectory_spring.link", "")
        if not link:
            link = next((n for n in DEFAULT_LINKS if n in arm.link_names), "")
        self.available = bool(link) and link in arm.link_names
        local_point = list(p("trajectory_spring.local_point", [0.0, 0.0, 0.0]))
        self.spring = TrackingPoseSpring(
            link or "none", local_point,
            stiffness=p("trajectory_spring.stiffness", 300.0),
            damping=p("trajectory_spring.damping", 30.0),
            rot_stiffness=p("trajectory_spring.rot_stiffness", 10.0),
            # See TrackingPoseSpring: above ~0.5 at 100 Hz the Gen2 wrist
            # oscillates.
            rot_damping=p("trajectory_spring.rot_damping", 0.2),
            null_stiffness=p("trajectory_spring.null_stiffness", 2.0),
            null_damping=p("trajectory_spring.null_damping", 0.5),
            max_force=p("trajectory_spring.max_force", 40.0),
            max_moment=p("trajectory_spring.max_moment", 5.0),
        )
        self.follower = TrajectoryFollower(self.spring, arm, FollowerConfig(
            pause_error_m=p("trajectory_spring.pause_error_m", 0.08),
            pause_error_rad=p("trajectory_spring.pause_error_rad", 0.5),
            pause_timeout_s=p("trajectory_spring.pause_timeout_sec", 10.0),
            goal_tol_rad=p("trajectory_spring.goal_tolerance_rad", 0.05),
            goal_time_tol_s=p("trajectory_spring.goal_time_tolerance_sec", 2.0),
            start_tol_rad=p("trajectory_spring.start_tolerance_rad", 0.05),
        ))
        self._max_ee_speed = p("trajectory_spring.max_ee_speed_mps", 0.25)
        self._crossfade_sec = max(1e-3, p("trajectory_spring.crossfade_sec", 1.0))
        action_name = p("trajectory_spring.action_name",
                        "/spring_trajectory_controller/follow_joint_trajectory")

        self._lock = threading.Lock()
        self._pending: Optional[_Command] = None
        self._goal_active = False
        self._outcome: Optional[FollowerOutcome] = None
        self._outcome_event = threading.Event()
        self._feedback = None
        self._mix = 0.0                  # tracking spring weight, 0..1
        self._mix_target = 0.0
        self._suspended: dict = {}       # VirtualSpring -> (stiffness, damping)
        self._last_tick: Optional[float] = None
        self._last_state_pub = -1.0

        self._state_pub = node.create_publisher(String, "~/trajectory/state", 10)
        if not self.available:
            log.warn("trajectory_spring: no end-effector link found "
                     f"(tried {list(DEFAULT_LINKS)}; set trajectory_spring.link) -- "
                     "FollowJointTrajectory server not started.")
            return
        group = ReentrantCallbackGroup()
        self._server = ActionServer(
            node, FollowJointTrajectory, action_name,
            execute_callback=self._execute, goal_callback=self._goal_cb,
            cancel_callback=lambda gh: CancelResponse.ACCEPT, callback_group=group)
        node.create_service(Trigger, "~/trajectory/release", self._release_cb,
                            callback_group=group)
        log.info(f"trajectory_spring: FollowJointTrajectory on {action_name}, "
                 f"spring on {link} at {local_point}.")

    def _declare(self, name, default):
        return self._node.declare_parameter(name, default).value

    # ── control loop side (node's /joint_states callback) ──

    def tick(self, now: float, ready: bool, not_ready_reason: str,
             springs_enabled: bool) -> np.ndarray:
        n = self._arm.n_dof
        if not self.available:
            return np.zeros(n)
        dt = 0.0 if self._last_tick is None else max(0.0, now - self._last_tick)
        self._last_tick = now
        outcome = None

        with self._lock:
            cmd, self._pending = self._pending, None
        if cmd is not None:
            outcome = self._run_command(cmd, now, ready, not_ready_reason)

        running = self.follower.state.value in ("RUNNING", "PAUSED")
        if running and not springs_enabled:
            outcome = self.follower.stop(
                PATH_TOLERANCE_VIOLATED,
                "Springs were disabled (collision clamp or ~/enable) mid-trajectory.")
        elif running:
            outcome = self.follower.tick(now) or outcome

        # Cross-fade between the tracking spring and the suspended springs.
        step = dt / self._crossfade_sec
        self._mix = (min(self._mix_target, self._mix + step) if self._mix < self._mix_target
                     else max(self._mix_target, self._mix - step))
        for spring, (k, b) in list(self._suspended.items()):
            spring.stiffness, spring.damping = k * (1.0 - self._mix), b * (1.0 - self._mix)
        if self._mix_target == 0.0 and self._mix == 0.0 and self.follower.active:
            self._finish_release()

        tau = np.zeros(n)
        if springs_enabled and self.follower.active and self._mix > 0.0:
            tau = self._mix * self.spring.compute_torques(self._arm)

        self._snapshot_feedback()
        if now - self._last_state_pub >= 0.1:
            self._publish_state()
            self._last_state_pub = now
        if outcome is not None:
            with self._lock:
                self._outcome = outcome
            self._outcome_event.set()
        return tau

    def _run_command(self, cmd: _Command, now, ready, not_ready_reason):
        outcome = None
        if cmd.kind == "start":
            if not ready:
                cmd.ok, cmd.message = False, not_ready_reason
            else:
                ref = cmd.kw["reference"]
                too_fast = self._peak_ee_speed(ref)
                if too_fast > self._max_ee_speed:
                    cmd.ok, cmd.message = False, (
                        f"Planned end-effector speed peaks at {too_fast:.2f} m/s, above "
                        f"trajectory_spring.max_ee_speed_mps={self._max_ee_speed:.2f}. "
                        "Plan with a lower velocity scaling.")
                else:
                    cmd.ok, cmd.message = self.follower.start(
                        ref, now, cmd.kw.get("goal_tol"), cmd.kw.get("goal_time_tol"))
                    if cmd.ok:
                        self._suspend_same_link_springs()
                        self._mix_target = 1.0
        elif cmd.kind == "cancel":
            outcome = self.follower.stop(SUCCESSFUL, "Trajectory cancelled.")
            cmd.ok, cmd.message = True, "Cancelled."
        elif cmd.kind == "release":
            if self.follower.state.value in ("RUNNING", "PAUSED"):
                cmd.ok, cmd.message = False, "A trajectory is running -- cancel it first."
            elif not self.follower.active:
                cmd.ok, cmd.message = True, "Nothing to release."
            else:
                self._retarget_suspended_springs()
                self._mix_target = 0.0
                cmd.ok, cmd.message = True, "Releasing: tracking spring fading out."
        cmd.done.set()
        return outcome

    def _peak_ee_speed(self, ref: JointTrajectoryReference) -> float:
        peak = 0.0
        for t in np.arange(0.0, ref.duration + 1e-9, 0.05):
            q, qd = ref.sample(t)
            _, _, J = self._arm.fk_and_jacobian_at(q, self.spring.link_name,
                                                   self.spring.local_attachment_point)
            peak = max(peak, float(np.linalg.norm(J[:3] @ qd)))
        return peak

    def _suspend_same_link_springs(self) -> None:
        for spring in self._springs:
            if (isinstance(spring, VirtualSpring) and spring.link_name == self.spring.link_name
                    and spring not in self._suspended):
                self._suspended[spring] = (spring.stiffness, spring.damping)

    def _retarget_suspended_springs(self) -> None:
        for spring in self._suspended:
            T = self._arm.get_link_transform(spring.link_name)
            spring.move_target((T @ np.append(spring.local_attachment_point, 1.0))[:3])
            self._node._publish_target(spring)

    def _finish_release(self) -> None:
        for spring, (k, b) in self._suspended.items():
            spring.stiffness, spring.damping = k, b
        self._suspended.clear()
        self.follower.release()

    def _snapshot_feedback(self) -> None:
        if self.follower.reference is None or not self.spring.has_anchor:
            return
        q = self.follower.current_angles()
        q_d = self.spring.anchor_joints
        with self._lock:
            self._feedback = (q_d.copy(), q, wrap_angle(q_d - q))

    def _publish_state(self) -> None:
        state = {"state": self.follower.state.value, "t": round(self.follower.t_traj, 3),
                 "mix": round(self._mix, 3)}
        if self.spring.last_state is not None:
            state["position_error_m"] = round(
                float(np.linalg.norm(self.spring.last_state.position_error)), 4)
            state["rotation_error_rad"] = round(
                float(np.linalg.norm(self.spring.last_state.rotation_error)), 4)
        self._state_pub.publish(String(data=json.dumps(state)))

    # ── action side (executor threads) ──

    def _post(self, cmd: _Command, timeout: float = 2.0) -> _Command:
        with self._lock:
            self._pending = cmd
        if not cmd.done.wait(timeout):
            with self._lock:
                if self._pending is cmd:
                    self._pending = None
            cmd.ok, cmd.message = False, ("Control loop not running (no /joint_states "
                                          f"for {timeout:.0f}s).")
        return cmd

    def _goal_cb(self, goal_request) -> GoalResponse:
        with self._lock:
            if self._goal_active:
                self._node.get_logger().warn(
                    "trajectory_spring: a trajectory is already running; rejecting the new goal.")
                return GoalResponse.REJECT
            self._goal_active = True
        return GoalResponse.ACCEPT

    def _release_cb(self, request, response):
        cmd = self._post(_Command("release"))
        response.success, response.message = cmd.ok, cmd.message
        return response

    def goal_to_reference(self, goal):
        """(reference, goal_tol, goal_time_tol, error_code, message)."""
        jt = goal.trajectory
        names = list(jt.joint_names)
        arm_names = self._arm.joint_names
        if sorted(names) != sorted(arm_names) or len(set(names)) != len(names):
            return None, None, None, INVALID_JOINTS, (
                f"Trajectory joints {names} must be exactly {arm_names}.")
        order = [names.index(n) for n in arm_names]
        if not jt.points:
            return None, None, None, INVALID_GOAL, "Trajectory has no points."
        if any(len(pt.positions) != len(names) for pt in jt.points):
            return None, None, None, INVALID_GOAL, "Every point needs every joint's position."
        has_vel = all(len(pt.velocities) == len(names) for pt in jt.points)
        try:
            ref = JointTrajectoryReference(
                [_seconds(pt.time_from_start) for pt in jt.points],
                [[pt.positions[k] for k in order] for pt in jt.points],
                [[pt.velocities[k] for k in order] for pt in jt.points] if has_vel else None,
                continuous_mask=self._arm.continuous_joint_mask())
        except ValueError as exc:
            return None, None, None, INVALID_GOAL, str(exc)
        goal_tol = np.full(len(arm_names), self.follower.cfg.goal_tol_rad)
        for tol in goal.goal_tolerance:
            if tol.name in arm_names and tol.position != 0.0:
                goal_tol[arm_names.index(tol.name)] = (np.inf if tol.position < 0
                                                       else tol.position)
        gtt = _seconds(goal.goal_time_tolerance)
        return ref, goal_tol, (gtt if gtt > 0 else None), SUCCESSFUL, ""

    def _execute(self, goal_handle):
        result = FollowJointTrajectory.Result()
        try:
            ref, goal_tol, gtt, code, message = self.goal_to_reference(goal_handle.request)
            if ref is None:
                return self._end(goal_handle, result, "abort", code, message)
            self._outcome_event.clear()
            with self._lock:
                self._outcome = None
            cmd = self._post(_Command("start", reference=ref, goal_tol=goal_tol,
                                      goal_time_tol=gtt))
            if not cmd.ok:
                return self._end(goal_handle, result, "abort", INVALID_GOAL, cmd.message)
            cancel_sent = False
            while not self._outcome_event.wait(0.1):
                if goal_handle.is_cancel_requested and not cancel_sent:
                    cancel_sent = True
                    self._post(_Command("cancel"))
                with self._lock:
                    fb, self._feedback = self._feedback, None
                if fb is not None:
                    goal_handle.publish_feedback(self._feedback_msg(*fb))
                if not self._node.context.ok():
                    return self._end(goal_handle, result, "abort",
                                     PATH_TOLERANCE_VIOLATED, "Shutting down.")
            with self._lock:
                outcome = self._outcome
            if goal_handle.is_cancel_requested:
                return self._end(goal_handle, result, "canceled", SUCCESSFUL, outcome.message)
            return self._end(goal_handle, result, "succeed" if outcome.succeeded else "abort",
                             outcome.error_code, outcome.message)
        finally:
            with self._lock:
                self._goal_active = False

    def _end(self, goal_handle, result, how, code, message):
        result.error_code = code
        result.error_string = message
        getattr(goal_handle, how)()
        if how != "succeed":
            self._node.get_logger().warn(f"trajectory_spring {how}: {message}")
        return result

    def _feedback_msg(self, desired, actual, error):
        fb = FollowJointTrajectory.Feedback()
        fb.header.stamp = self._node.get_clock().now().to_msg()
        fb.joint_names = list(self._arm.joint_names)
        fb.desired = JointTrajectoryPoint(positions=[float(x) for x in desired])
        fb.actual = JointTrajectoryPoint(positions=[float(x) for x in actual])
        fb.error = JointTrajectoryPoint(positions=[float(x) for x in error])
        return fb
