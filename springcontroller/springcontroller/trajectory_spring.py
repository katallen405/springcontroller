"""
trajectory_spring.py
────────────────────
Follow a planned joint trajectory compliantly: a 6-DOF end-effector spring
whose anchor slides along the plan.

Each control tick the follower samples the plan at its own trajectory time,
q_d(t) and q̇_d(t), and sets the anchor of a TrackingPoseSpring to
FK(q_d) with feed-forward twist J(q_d)·q̇_d. The spring pulls the end
effector toward that moving anchor:

    e_p = x_d − x                   F = K_p e_p + D_p (v_d − v)
    e_R = log(R_d Rᵀ)  (world)      M = K_r e_R + D_r (ω_d − ω)
    τ   = Jᵀ [F; M] + (I − Jᵀ J#ᵀ) (k_null wrap(q_d − q) − b_null q̇)

The nullspace term pulls the redundant 7th DOF toward the planned joint
configuration (so the elbow follows the plan) without adding any force at
the end effector. Gravity compensation is added by SpringCollection as for
every other spring.

If someone holds the arm back, the anchor waits: trajectory time stops
advancing while the tracking error is above the pause thresholds and
resumes when it drops back, so the arm never chases an anchor that ran
ahead. A pause that lasts too long aborts.

Nothing here imports ROS; virtual_spring_node wires it to a
FollowJointTrajectory action server.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

# control_msgs/action/FollowJointTrajectory Result.error_code values.
SUCCESSFUL = 0
INVALID_GOAL = -1
INVALID_JOINTS = -2
PATH_TOLERANCE_VIOLATED = -4
GOAL_TOLERANCE_VIOLATED = -5


def wrap_angle(a):
    """Wrap to (-π, π]."""
    return np.pi - np.mod(np.pi - np.asarray(a, dtype=float), 2.0 * np.pi)


def rotation_log(R: np.ndarray) -> np.ndarray:
    """Rotation vector (axis·angle) of a rotation matrix; robust near 0 and π."""
    cos = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    angle = float(np.arccos(cos))
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if angle < 1e-6:
        return 0.5 * w
    if np.pi - angle < 1e-4:
        # sin(angle) ~ 0: recover the axis from the symmetric part instead.
        B = (R + np.eye(3)) / 2.0
        axis = np.sqrt(np.clip(np.diag(B), 0.0, None))
        k = int(np.argmax(axis))
        axis = B[:, k] / max(axis[k], 1e-12)
        axis /= np.linalg.norm(axis)
        if np.dot(axis, w) < 0:
            axis = -axis
        return angle * axis
    return angle / (2.0 * np.sin(angle)) * w


# ── Reference ─────────────────────────────────────────────────────────────────

class JointTrajectoryReference:
    """A planned joint trajectory, sampled at arbitrary time.

    times: (N,) s from start, strictly increasing. positions: (N, n) rad.
    velocities: (N, n) rad/s or None (then linear interpolation). Columns of
    continuous joints are unwrapped so no step between points sweeps the
    long way round.
    """

    def __init__(self, times: Sequence[float], positions, velocities=None,
                 continuous_mask: Optional[Sequence[bool]] = None) -> None:
        self.times = np.asarray(times, dtype=float)
        self.positions = np.array(positions, dtype=float)
        self.velocities = None if velocities is None else np.array(velocities, dtype=float)
        if self.times.ndim != 1 or len(self.times) == 0:
            raise ValueError("Trajectory has no points.")
        if self.positions.shape[0] != len(self.times):
            raise ValueError("Every point needs positions.")
        if self.velocities is not None and self.velocities.shape != self.positions.shape:
            raise ValueError("Velocities must be given for every point and joint, or for none.")
        if np.any(np.diff(self.times) <= 0.0) or self.times[0] < 0.0:
            raise ValueError("time_from_start must be non-negative and strictly increasing.")
        if not np.all(np.isfinite(self.positions)) or (
                self.velocities is not None and not np.all(np.isfinite(self.velocities))):
            raise ValueError("Trajectory contains NaN or infinite values.")
        if continuous_mask is not None:
            for j, cont in enumerate(continuous_mask):
                if cont:
                    self.positions[:, j] = np.unwrap(self.positions[:, j])

    @property
    def duration(self) -> float:
        return float(self.times[-1])

    @property
    def n_joints(self) -> int:
        return self.positions.shape[1]

    def prepend_start(self, q_now: np.ndarray) -> None:
        """A plan whose first point is later than t=0 starts from q_now."""
        if self.times[0] <= 0.0:
            return
        self.times = np.concatenate([[0.0], self.times])
        self.positions = np.vstack([q_now, self.positions])
        if self.velocities is not None:
            self.velocities = np.vstack([np.zeros(self.n_joints), self.velocities])

    def sample(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        """(q_d, q̇_d) at time t; holds the end points outside [t0, T]."""
        if t <= self.times[0]:
            return self.positions[0].copy(), np.zeros(self.n_joints)
        if t >= self.times[-1]:
            return self.positions[-1].copy(), np.zeros(self.n_joints)
        k = int(np.searchsorted(self.times, t, side="right")) - 1
        t0, t1 = self.times[k], self.times[k + 1]
        h = t1 - t0
        s = (t - t0) / h
        p0, p1 = self.positions[k], self.positions[k + 1]
        if self.velocities is None:
            return p0 + (p1 - p0) * s, (p1 - p0) / h
        v0, v1 = self.velocities[k], self.velocities[k + 1]
        s2, s3 = s * s, s * s * s
        pos = ((2 * s3 - 3 * s2 + 1) * p0 + (s3 - 2 * s2 + s) * h * v0
               + (-2 * s3 + 3 * s2) * p1 + (s3 - s2) * h * v1)
        vel = ((6 * s2 - 6 * s) * p0 + (3 * s2 - 4 * s + 1) * h * v0
               + (-6 * s2 + 6 * s) * p1 + (3 * s2 - 2 * s) * h * v1) / h
        return pos, vel


# ── Spring ────────────────────────────────────────────────────────────────────

@dataclass
class TrackingSpringState:
    position: np.ndarray        # end-effector point, world
    position_error: np.ndarray  # x_d − x
    rotation_error: np.ndarray  # log(R_d Rᵀ), world
    force: np.ndarray           # [F; M] after saturation
    torques: np.ndarray


class TrackingPoseSpring:
    """6-DOF spring/damper from a point on a link to a moving anchor pose.

    Same compute_torques(arm) interface as VirtualSpring, so it sits in a
    SpringCollection and inherits spring_scale, the DANGER scaling and the
    COLLISION clamp. Produces zero torque until an anchor is set.

    rot_damping is limited by the control rate: the torque is held for a
    whole tick, so a damper on a low-inertia joint overshoots once
    b·Δt/I ≳ 2. On the Gen2 (100 Hz, wrist joint 7 carrying only the
    gripper) rot_damping = 1.0 N·m·s/rad drives joint 7 into a ~9 rad/s
    limit cycle in simulation; 0.05–0.2 tracks within ~1 mm.
    """

    def __init__(self, link_name: str, local_point: Sequence[float] = (0.0, 0.0, 0.0),
                 stiffness: float = 300.0, damping: float = 30.0,
                 rot_stiffness: float = 10.0, rot_damping: float = 0.2,
                 null_stiffness: float = 2.0, null_damping: float = 0.5,
                 max_force: float = 40.0, max_moment: float = 5.0,
                 enabled: bool = True, name: str = "trajectory_spring") -> None:
        self.link_name = link_name
        self.local_attachment_point = np.asarray(local_point, dtype=float)
        if self.local_attachment_point.shape != (3,):
            raise ValueError("local_point must have shape (3,)")
        for label, value in (("stiffness", stiffness), ("damping", damping),
                             ("rot_stiffness", rot_stiffness), ("rot_damping", rot_damping),
                             ("null_stiffness", null_stiffness), ("null_damping", null_damping),
                             ("max_force", max_force), ("max_moment", max_moment)):
            if value < 0:
                raise ValueError(f"{label} must be ≥ 0, got {value}")
        self.stiffness = stiffness
        self.damping = damping
        self.rot_stiffness = rot_stiffness
        self.rot_damping = rot_damping
        self.null_stiffness = null_stiffness
        self.null_damping = null_damping
        self.max_force = max_force
        self.max_moment = max_moment
        self.enabled = enabled
        self.name = name
        self.anchor_position: Optional[np.ndarray] = None
        self.anchor_rotation: Optional[np.ndarray] = None
        self.anchor_twist = np.zeros(6)
        self.anchor_joints: Optional[np.ndarray] = None
        self._last_state: Optional[TrackingSpringState] = None

    def set_anchor(self, position, rotation, twist=None, joints=None) -> None:
        self.anchor_position = np.asarray(position, dtype=float).copy()
        self.anchor_rotation = np.asarray(rotation, dtype=float).copy()
        self.anchor_twist = np.zeros(6) if twist is None else np.asarray(twist, dtype=float)
        self.anchor_joints = None if joints is None else np.asarray(joints, dtype=float).copy()

    def clear_anchor(self) -> None:
        self.anchor_position = self.anchor_rotation = self.anchor_joints = None
        self.anchor_twist = np.zeros(6)
        self._last_state = None

    @property
    def has_anchor(self) -> bool:
        return self.anchor_position is not None

    @property
    def last_state(self) -> Optional[TrackingSpringState]:
        return self._last_state

    def pose_error(self, arm) -> tuple[np.ndarray, np.ndarray]:
        """(x_d − x, log(R_d Rᵀ)) for the current arm state."""
        T = arm.get_link_transform(self.link_name)
        R = T[:3, :3]
        x = R @ self.local_attachment_point + T[:3, 3]
        return self.anchor_position - x, rotation_log(self.anchor_rotation @ R.T)

    def compute_torques(self, arm) -> np.ndarray:
        n = arm.n_dof
        if not self.enabled or not self.has_anchor:
            self._last_state = None
            return np.zeros(n)
        T = arm.get_link_transform(self.link_name)
        R = T[:3, :3]
        x = R @ self.local_attachment_point + T[:3, 3]
        J = arm.get_jacobian(self.link_name, self.local_attachment_point)
        qdot = np.asarray(arm.joint_velocities, dtype=float)
        twist = J @ qdot

        e_p = self.anchor_position - x
        e_R = rotation_log(self.anchor_rotation @ R.T)
        F = self.stiffness * e_p + self.damping * (self.anchor_twist[:3] - twist[:3])
        M = self.rot_stiffness * e_R + self.rot_damping * (self.anchor_twist[3:] - twist[3:])
        F = _saturate(F, self.max_force)
        M = _saturate(M, self.max_moment)
        wrench = np.concatenate([F, M])
        torques = J.T @ wrench

        if self.anchor_joints is not None and (self.null_stiffness > 0 or self.null_damping > 0):
            q = np.array([arm.get_joint_angle(i) for i in range(n)])
            tau0 = (self.null_stiffness * wrap_angle(self.anchor_joints - q)
                    - self.null_damping * qdot)
            # Damped pseudo-inverse; N = I − Jᵀ J#ᵀ keeps tau0 out of the
            # end-effector wrench.
            lam2 = 1e-4
            J_pinv = J.T @ np.linalg.inv(J @ J.T + lam2 * np.eye(6))
            torques = torques + (np.eye(n) - J.T @ J_pinv.T) @ tau0

        self._last_state = TrackingSpringState(x, e_p, e_R, wrench, torques)
        return torques

    def __repr__(self) -> str:
        return (f"TrackingPoseSpring(name={self.name!r}, link={self.link_name!r}, "
                f"k={self.stiffness}, b={self.damping}, k_rot={self.rot_stiffness}, "
                f"enabled={self.enabled})")


def _saturate(v: np.ndarray, limit: float) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    return v if norm <= limit or norm == 0.0 else v * (limit / norm)


# ── Follower ──────────────────────────────────────────────────────────────────

class FollowerState(enum.Enum):
    IDLE = "IDLE"          # no anchor; spring contributes nothing
    RUNNING = "RUNNING"    # anchor moving along the plan
    PAUSED = "PAUSED"      # tracking error too large: trajectory time frozen
    HOLDING = "HOLDING"    # plan finished (or stopped); anchor holds a fixed pose


@dataclass
class FollowerConfig:
    pause_error_m: float = 0.08
    pause_error_rad: float = 0.5
    resume_fraction: float = 0.6       # resume once errors < this × the pause thresholds
    pause_timeout_s: float = 10.0
    goal_tol_rad: float = 0.05         # per joint, at the end of the plan
    stopped_vel_rad_s: float = 0.02
    goal_time_tol_s: float = 2.0
    start_tol_rad: float = 0.05        # first point vs current pose


@dataclass
class FollowerOutcome:
    error_code: int
    succeeded: bool
    message: str


class TrajectoryFollower:
    """Moves a TrackingPoseSpring's anchor along a JointTrajectoryReference.

    Drive it with tick(now, arm) once per control cycle, before the springs'
    torques are computed. tick returns a FollowerOutcome once when the
    trajectory ends (success or failure), otherwise None.
    """

    def __init__(self, spring: TrackingPoseSpring, arm, config: FollowerConfig) -> None:
        self.spring = spring
        self.arm = arm                 # needs fk_and_jacobian_at, get_joint_angle, n_dof
        self.cfg = config
        self.state = FollowerState.IDLE
        self.reference: Optional[JointTrajectoryReference] = None
        self.t_traj = 0.0
        self._last_now: Optional[float] = None
        self._paused_since: Optional[float] = None
        self._ended_at: Optional[float] = None
        self._goal_time_tol = config.goal_time_tol_s
        self._goal_tol = np.full(arm.n_dof, config.goal_tol_rad)

    # ── commands ──

    def current_angles(self) -> np.ndarray:
        return np.array([self.arm.get_joint_angle(i) for i in range(self.arm.n_dof)])

    def start(self, reference: JointTrajectoryReference, now: float,
              goal_tol_rad: Optional[Sequence[float]] = None,
              goal_time_tol_s: Optional[float] = None) -> tuple[bool, str]:
        if self.state in (FollowerState.RUNNING, FollowerState.PAUSED):
            return False, "A trajectory is already running."
        if reference.n_joints != self.arm.n_dof:
            return False, f"Trajectory has {reference.n_joints} joints, arm has {self.arm.n_dof}."
        q_now = self.current_angles()
        reference.prepend_start(q_now)
        # Unwrapping may have shifted continuous joints by whole turns; FK
        # doesn't care, and errors are wrapped, so compare wrapped.
        start_err = np.abs(wrap_angle(reference.positions[0] - q_now))
        if np.max(start_err) > self.cfg.start_tol_rad:
            j = int(np.argmax(start_err))
            return False, (f"Trajectory starts {start_err[j]:.3f} rad from the current pose on "
                           f"joint {j + 1} (allowed {self.cfg.start_tol_rad:.3f}).")
        self.reference = reference
        self._goal_tol = (np.full(self.arm.n_dof, self.cfg.goal_tol_rad) if goal_tol_rad is None
                          else np.asarray(goal_tol_rad, dtype=float))
        self._goal_time_tol = (self.cfg.goal_time_tol_s if goal_time_tol_s is None
                               else goal_time_tol_s)
        self.t_traj = 0.0
        self._last_now = now
        self._paused_since = None
        self._ended_at = None
        self.state = FollowerState.RUNNING
        self._set_anchor_at(0.0, moving=True)
        return True, "Trajectory started."

    def hold_current(self) -> None:
        """Anchor at the arm's current pose, not moving -- after a stop, so
        nothing pulls the arm back toward where the plan would have been."""
        q = self.current_angles()
        p, R, _ = self.arm.fk_and_jacobian_at(q, self.spring.link_name,
                                              self.spring.local_attachment_point)
        self.spring.set_anchor(p, R, None, q)
        self.state = FollowerState.HOLDING

    def stop(self, error_code: int, message: str) -> Optional[FollowerOutcome]:
        """End a running trajectory (cancel, collision, ...) and hold where the arm is."""
        if self.state not in (FollowerState.RUNNING, FollowerState.PAUSED):
            return None
        self.hold_current()
        self.reference = None
        return FollowerOutcome(error_code, False, message)

    def release(self) -> None:
        """Drop the anchor entirely; the spring goes back to contributing nothing."""
        self.spring.clear_anchor()
        self.reference = None
        self.state = FollowerState.IDLE

    @property
    def active(self) -> bool:
        return self.state is not FollowerState.IDLE

    # ── per-tick ──

    def tick(self, now: float) -> Optional[FollowerOutcome]:
        if self.state not in (FollowerState.RUNNING, FollowerState.PAUSED):
            return None
        dt = max(0.0, now - (self._last_now if self._last_now is not None else now))
        self._last_now = now
        cfg = self.cfg

        e_p, e_R = self.spring.pose_error(self.arm)
        ep, eR = float(np.linalg.norm(e_p)), float(np.linalg.norm(e_R))
        if self.state is FollowerState.RUNNING:
            if ep > cfg.pause_error_m or eR > cfg.pause_error_rad:
                self.state = FollowerState.PAUSED
                self._paused_since = now
            else:
                self.t_traj += dt
        if self.state is FollowerState.PAUSED:
            if (ep < cfg.resume_fraction * cfg.pause_error_m
                    and eR < cfg.resume_fraction * cfg.pause_error_rad):
                self.state = FollowerState.RUNNING
                self._paused_since = None
            elif now - self._paused_since > cfg.pause_timeout_s:
                return self.stop(PATH_TOLERANCE_VIOLATED,
                                 f"Held off the path for more than {cfg.pause_timeout_s:.1f}s "
                                 f"(end-effector error {ep:.3f} m, {eR:.2f} rad).")

        ref = self.reference
        self._set_anchor_at(self.t_traj, moving=self.state is FollowerState.RUNNING)
        if self.t_traj < ref.duration:
            return None

        if self._ended_at is None:
            self._ended_at = now
        q = self.current_angles()
        qdot = np.asarray(self.arm.joint_velocities, dtype=float)
        joint_err = np.abs(wrap_angle(ref.positions[-1] - q))
        if np.all(joint_err <= self._goal_tol) and np.all(np.abs(qdot) <= cfg.stopped_vel_rad_s):
            self.state = FollowerState.HOLDING      # anchor stays on the final pose
            self.reference = None
            return FollowerOutcome(SUCCESSFUL, True, "Trajectory finished.")
        if now - self._ended_at > self._goal_time_tol:
            outcome = FollowerOutcome(
                GOAL_TOLERANCE_VIOLATED, False,
                f"Not within goal tolerance {self._goal_time_tol:.1f}s after the plan ended "
                f"(worst joint {int(np.argmax(joint_err)) + 1}: {np.max(joint_err):.3f} rad).")
            self.hold_current()
            self.reference = None
            return outcome
        return None

    def _set_anchor_at(self, t: float, moving: bool) -> None:
        q_d, qd_d = self.reference.sample(t)
        p, R, J = self.arm.fk_and_jacobian_at(q_d, self.spring.link_name,
                                              self.spring.local_attachment_point)
        twist = J @ qd_d if moving else np.zeros(6)
        self.spring.set_anchor(p, R, twist, q_d)
