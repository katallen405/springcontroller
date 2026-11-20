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
    INVALID_GOAL, INVALID_JOINTS, PATH_TOLERANCE_VIOLATED, SUCCESSFUL, CartesianPathReference,
    FollowerConfig, FollowerOutcome, JointTrajectoryReference, TrackingPoseSpring,
    TrajectoryFollower, _rotation_from_quat, wrap_angle,
)
from springcontroller.virtual_spring import VirtualSpring

try:   # needs springcontroller_interfaces rebuilt with action/FollowAnchorPath
    from springcontroller_interfaces.action import FollowAnchorPath
except ImportError:                          # pragma: no cover - depends on the install
    FollowAnchorPath = None

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
            # Limits on how hard a spring pulls, not on distance: the allowed
            # error follows from the spring's strength (see FollowerConfig).
            pause_force_n=p("trajectory_spring.pause_force_n", 24.0),
            pause_moment_nm=p("trajectory_spring.pause_moment_nm", 5.0),
            pause_timeout_s=p("trajectory_spring.pause_timeout_sec", 10.0),
            goal_force_tol_n=p("trajectory_spring.goal_force_tolerance_n", 4.0),
            goal_moment_tol_nm=p("trajectory_spring.goal_moment_tolerance_nm", 1.2),
            goal_tol_rad=p("trajectory_spring.goal_tolerance_rad", 0.0),   # <= 0: joints unchecked
            goal_time_tol_s=p("trajectory_spring.goal_time_tolerance_sec", 2.0),
            start_tol_rad=p("trajectory_spring.start_tolerance_rad", 0.05),
        ))
        self._max_ee_speed = p("trajectory_spring.max_ee_speed_mps", 0.25)
        self._crossfade_sec = max(1e-3, p("trajectory_spring.crossfade_sec", 1.0))
        action_name = p("trajectory_spring.action_name",
                        "/spring_trajectory_controller/follow_joint_trajectory")
        anchor_action_name = p("trajectory_spring.anchor_action_name",
                               "/spring_trajectory_controller/follow_anchor_path")
        # The arm model's world frame: FollowAnchorPath goals are transformed
        # into it. The URDF root link (world for the Gen2/Gen3 flat URDFs).
        self._base_frame = p("trajectory_spring.base_frame", "world")
        self._tf_buffer = None
        self._last_state: Optional[dict] = None   # latest ~/trajectory/state, for feedback

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
        if FollowAnchorPath is None:
            log.warn("trajectory_spring: springcontroller_interfaces has no FollowAnchorPath "
                     "(rebuild it) -- anchor-path server not started.")
            return
        from tf2_ros import Buffer, TransformListener
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, node)
        self._anchor_server = ActionServer(
            node, FollowAnchorPath, anchor_action_name,
            execute_callback=self._execute_anchor, goal_callback=self._goal_cb,
            cancel_callback=lambda gh: CancelResponse.ACCEPT, callback_group=group)
        log.info(f"trajectory_spring: FollowAnchorPath on {anchor_action_name} "
                 f"(points in any TF frame, base frame '{self._base_frame}').")

    def _declare(self, name, default):
        # virtual_spring_node's config-file loader declares every key in the
        # YAML before this runs, so a config that sets trajectory_spring.*
        # has already declared it: read that value instead of re-declaring.
        if self._node.has_parameter(name):
            value = self._node.get_parameter(name).value
            if isinstance(default, float) and isinstance(value, int):
                value = float(value)   # "stiffness: 60" in YAML is an int
            return value
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
                # not_ready_reason names the actual cause (torque dropped,
                # collision clamp, ~/enable); a fixed "collision clamp or
                # ~/enable" message misled the first time torque dropped.
                "Stopped mid-trajectory: "
                + (not_ready_reason or "springs were disabled."))
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
                # Prepend the start before checking speeds: a goal with no
                # t=0 point is otherwise a single held point here, and its
                # first segment (from the arm to the plan) went unchecked.
                if isinstance(ref, CartesianPathReference):
                    ref.prepend_start(*self.follower.current_anchor_pose())
                    too_fast = ref.peak_speed()       # the anchor's own speed
                    joint_msg = ""                    # no joint plan to check
                else:
                    ref.prepend_start(self.follower.current_angles())
                    too_fast = self._peak_ee_speed(ref)
                    joint_msg = self._joint_speed_violation(ref)
                if too_fast > self._max_ee_speed:
                    cmd.ok, cmd.message = False, (
                        f"Planned end-effector speed peaks at {too_fast:.2f} m/s, above "
                        f"trajectory_spring.max_ee_speed_mps={self._max_ee_speed:.2f}. "
                        + ("Use longer times between points."
                           if isinstance(ref, CartesianPathReference)
                           else "Plan with a lower velocity scaling."))
                elif joint_msg:
                    cmd.ok, cmd.message = False, joint_msg
                else:
                    self._set_hold_stiffness()
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

    def _joint_speed_violation(self, ref: JointTrajectoryReference) -> str:
        """Empty if every joint stays within its URDF velocity limit, else why not.

        The end-effector speed check alone misses wrist motion: spinning
        joint 7 barely moves the end-effector point.
        """
        model = getattr(self._arm, "_model", None)
        limits = None if model is None else np.asarray(model.velocityLimit, dtype=float)
        if limits is None or len(limits) != ref.n_joints or not np.all(limits > 0):
            return ""
        peak = np.zeros(ref.n_joints)
        for t in np.arange(0.0, ref.duration + 1e-9, 0.05):
            peak = np.maximum(peak, np.abs(ref.sample(t)[1]))
        worst = int(np.argmax(peak / limits))
        if peak[worst] <= limits[worst]:
            return ""
        return (f"Planned speed of {self._arm.joint_names[worst]} peaks at "
                f"{peak[worst]:.2f} rad/s, above its limit of {limits[worst]:.2f} rad/s. "
                f"Plan a longer move (at least {ref.duration * peak[worst] / limits[worst]:.1f} s).")

    def _peak_ee_speed(self, ref: JointTrajectoryReference) -> float:
        peak = 0.0
        for t in np.arange(0.0, ref.duration + 1e-9, 0.05):
            q, qd = ref.sample(t)
            _, _, J = self._arm.fk_and_jacobian_at(q, self.spring.link_name,
                                                   self.spring.local_attachment_point)
            peak = max(peak, float(np.linalg.norm(J[:3] @ qd)))
        return peak

    def _same_link_springs(self) -> list:
        """The Cartesian springs on the tracking spring's link (e.g. the Gen2
        tip-hold spring): faded out while a trajectory runs, back after release."""
        return [s for s in self._springs
                if isinstance(s, VirtualSpring) and s.link_name == self.spring.link_name]

    def _set_hold_stiffness(self) -> None:
        """The end tolerance follows from the spring(s) that hold the gripper
        after release. Use their own stiffness -- the saved value if a
        previous trajectory already faded them out."""
        springs = self._same_link_springs()
        k = sum(self._suspended[s][0] if s in self._suspended else s.stiffness
                for s in springs)
        if springs and k > 0:
            self.follower.hold_stiffness = k
            self.follower.hold_label = "+".join(s.name for s in springs)
        else:                          # nothing else holds it: the tracking spring's own
            self.follower.hold_stiffness = None
            self.follower.hold_label = ""
        # Those springs have no orientation; the tracking spring's rot_stiffness applies.
        self.follower.hold_rot_stiffness = None

    def _suspend_same_link_springs(self) -> None:
        for spring in self._same_link_springs():
            if spring not in self._suspended:
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

    def _springs_report(self) -> dict:
        """The springs acting on the gripper right now (for ~/trajectory/state).

        tracking: the trajectory spring; weight is its cross-fade (0..1).
        same_link: Cartesian springs on the same link (e.g. tip_spring);
        stiffness is what it pulls with now (faded while a trajectory owns
        the gripper), stiffness_full what it returns to after release.
        """
        s = self.spring
        report = {"tracking": {
            "active": bool(self.follower.active and self._mix > 0.0),
            "weight": round(self._mix, 3),
            "stiffness": s.stiffness, "damping": s.damping,
            "rot_stiffness": s.rot_stiffness, "rot_damping": s.rot_damping,
            "max_force": s.max_force, "max_moment": s.max_moment,
            "null_stiffness": s.null_stiffness,
        }, "same_link": []}
        for v in self._same_link_springs():
            full = self._suspended[v][0] if v in self._suspended else v.stiffness
            report["same_link"].append({
                "name": v.name, "stiffness": round(v.stiffness, 3),
                "stiffness_full": full, "damping": round(v.damping, 3),
                "enabled": bool(getattr(v, "enabled", True)),
            })
        return report

    def _active_springs_text(self) -> str:
        """One sentence for a goal's result: which springs now hold the gripper."""
        s = self.spring
        parts = []
        if self.follower.active:
            parts.append(f"tracking spring ({s.stiffness:g} N/m, {s.rot_stiffness:g} N*m/rad) "
                         f"holding the gripper where it ended")
        for v in self._same_link_springs():
            if v in self._suspended:
                parts.append(f"{v.name} ({self._suspended[v][0]:g} N/m) faded out until "
                             f"~/trajectory/release")
            elif getattr(v, "enabled", True) and v.stiffness > 0:
                parts.append(f"{v.name} ({v.stiffness:g} N/m) active")
        others = [o.name for o in self._springs
                  if o not in self._same_link_springs()
                  and getattr(o, "enabled", True) and getattr(o, "stiffness", 0) > 0]
        if others:
            parts.append("also active: " + ", ".join(others))
        return "Active springs: " + ("; ".join(parts) if parts else "none") + "."

    def _publish_state(self) -> None:
        state = {"state": self.follower.state.value, "t": round(self.follower.t_traj, 3),
                 "mix": round(self._mix, 3)}
        if self.spring.last_state is not None:
            state["position_error_m"] = round(
                float(np.linalg.norm(self.spring.last_state.position_error)), 4)
            state["rotation_error_rad"] = round(
                float(np.linalg.norm(self.spring.last_state.rotation_error)), 4)
        state["springs"] = self._springs_report()
        with self._lock:
            self._last_state = state
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
        base_tol = self.follower.cfg.goal_tol_rad
        # <= 0: joints not checked unless the goal asks (the spring-force check decides).
        goal_tol = np.full(len(arm_names), base_tol if base_tol > 0 else np.inf)
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
            return self._run_goal(goal_handle, result, ref, goal_tol, gtt,
                                  self._joint_feedback_msg)
        finally:
            with self._lock:
                self._goal_active = False

    def _joint_feedback_msg(self):
        with self._lock:
            fb, self._feedback = self._feedback, None
        return None if fb is None else self._feedback_msg(*fb)

    def _run_goal(self, goal_handle, result, ref, goal_tol, gtt, make_feedback):
        """Start ref on the control loop, publish feedback every 0.1 s until
        it ends, handle cancel. Shared by both action servers."""
        self._outcome_event.clear()
        with self._lock:
            self._outcome = None
        cmd = self._post(_Command("start", reference=ref, goal_tol=goal_tol, goal_time_tol=gtt))
        if not cmd.ok:
            return self._end(goal_handle, result, "abort", INVALID_GOAL, cmd.message)
        cancel_sent = False
        while not self._outcome_event.wait(0.1):
            if goal_handle.is_cancel_requested and not cancel_sent:
                cancel_sent = True
                self._post(_Command("cancel"))
            fb = make_feedback()
            if fb is not None:
                goal_handle.publish_feedback(fb)
            if not self._node.context.ok():
                return self._end(goal_handle, result, "abort",
                                 PATH_TOLERANCE_VIOLATED, "Shutting down.")
        with self._lock:
            outcome = self._outcome
        if goal_handle.is_cancel_requested:
            return self._end(goal_handle, result, "canceled", SUCCESSFUL, outcome.message)
        return self._end(goal_handle, result, "succeed" if outcome.succeeded else "abort",
                         outcome.error_code, outcome.message)

    # ── FollowAnchorPath ──

    def anchor_goal_to_reference(self, goal):
        """(CartesianPathReference or None, error_code, message)."""
        n = len(goal.points)
        if n == 0:
            return None, INVALID_GOAL, "Goal has no points."
        if len(goal.times) != n:
            return None, INVALID_GOAL, f"{n} points but {len(goal.times)} times."
        frame = goal.header.frame_id or self._base_frame
        R_tf, t_tf = np.eye(3), np.zeros(3)
        if frame != self._base_frame:
            if self._tf_buffer is None:
                return None, INVALID_GOAL, "No TF listener; give points in the base frame."
            try:
                from rclpy.duration import Duration
                from rclpy.time import Time
                tf = self._tf_buffer.lookup_transform(self._base_frame, frame, Time(),
                                                      timeout=Duration(seconds=1.0))
            except Exception as exc:           # tf2 raises several exception types
                return None, INVALID_GOAL, (f"Can't transform '{frame}' to "
                                            f"'{self._base_frame}': {exc}")
            r, tr = tf.transform.rotation, tf.transform.translation
            R_tf = _rotation_from_quat([r.w, r.x, r.y, r.z])
            t_tf = np.array([tr.x, tr.y, tr.z])
        points = [R_tf @ np.array([pt.x, pt.y, pt.z]) + t_tf for pt in goal.points]
        times = [_seconds(d) for d in goal.times]
        constrained = [i for i in range(3) if goal.constrain_rotation[i]]
        rotation = None
        if constrained:
            o = goal.orientation
            if np.linalg.norm([o.w, o.x, o.y, o.z]) < 1e-9:
                return None, INVALID_GOAL, "orientation is not a valid quaternion."
            rotation = R_tf @ _rotation_from_quat([o.w, o.x, o.y, o.z])
        axes = np.array([R_tf[:, i] for i in constrained]).reshape(-1, 3)
        try:
            ref = CartesianPathReference(times, points, rotation=rotation, rotation_axes=axes)
        except ValueError as exc:
            return None, INVALID_GOAL, str(exc)
        return ref, SUCCESSFUL, ""

    def _execute_anchor(self, goal_handle):
        result = FollowAnchorPath.Result()
        try:
            ref, code, message = self.anchor_goal_to_reference(goal_handle.request)
            if ref is None:
                return self._end(goal_handle, result, "abort", code, message)
            return self._run_goal(goal_handle, result, ref, None, None,
                                  self._anchor_feedback_msg)
        finally:
            with self._lock:
                self._goal_active = False

    def _anchor_feedback_msg(self):
        with self._lock:
            state = self._last_state
        if state is None:
            return None
        fb = FollowAnchorPath.Feedback()
        fb.state = state["state"]
        fb.time_from_start = float(state["t"])
        fb.position_error_m = float(state.get("position_error_m", 0.0))
        fb.rotation_error_rad = float(state.get("rotation_error_rad", 0.0))
        fb.springs = json.dumps(state["springs"])
        return fb

    def _end(self, goal_handle, result, how, code, message):
        # FollowJointTrajectory feedback has no text field, but the result's
        # error_string is free text (also on success): say which springs now
        # hold the gripper. Live values are on ~/trajectory/state.
        springs_text = self._active_springs_text()
        message = f"{message} {springs_text}"
        if hasattr(result, "active_springs"):          # FollowAnchorPath
            result.active_springs = springs_text
        result.error_code = code
        result.error_string = message
        getattr(goal_handle, how)()
        if how == "succeed":
            self._node.get_logger().info(f"trajectory_spring succeeded: {message}")
        else:
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
