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


def rotation_exp(w: np.ndarray) -> np.ndarray:
    """Rotation matrix of a rotation vector (inverse of rotation_log)."""
    w = np.asarray(w, dtype=float)
    angle = float(np.linalg.norm(w))
    if angle < 1e-12:
        return np.eye(3)
    k = w / angle
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


def _quat_from_rotation(R: np.ndarray) -> np.ndarray:
    """(w, x, y, z), w >= 0."""
    w = rotation_log(R)
    angle = float(np.linalg.norm(w))
    if angle < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    q = np.concatenate([[np.cos(angle / 2.0)], np.sin(angle / 2.0) * w / angle])
    return q if q[0] >= 0 else -q


def _rotation_from_quat(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float) / np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, av = a[0], a[1:]
    bw, bv = b[0], b[1:]
    return np.concatenate([[aw * bw - av @ bv], aw * bv + bw * av + np.cross(av, bv)])


def _twist_about(q: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Twist part of the swing-twist decomposition q = swing * twist about unit u."""
    t = np.concatenate([[q[0]], (q[1:] @ u) * u])
    n = float(np.linalg.norm(t))
    return np.array([1.0, 0.0, 0.0, 0.0]) if n < 1e-12 else t / n


def masked_rotation_error(E: np.ndarray, axes: Optional[np.ndarray]) -> np.ndarray:
    """Rotation vector of the error E = R_d Rᵀ (world), keeping only the
    rotation about the constrained axes.

    axes: None = all three constrained (plain log); else a (k, 3) array of
    orthonormal world unit vectors, k = 0..3. Exact for every k via the
    swing-twist decomposition, not just for small errors:
      k = 1 (axis c):        only the twist of E about c.
      k = 2 (free axis a):   E with its twist about a removed (the swing).
    """
    if axes is None:
        return rotation_log(E)
    axes = np.asarray(axes, dtype=float).reshape(-1, 3)
    k = axes.shape[0]
    if k == 0:
        return np.zeros(3)
    if k == 3:
        return rotation_log(E)
    q = _quat_from_rotation(E)
    if k == 1:
        return rotation_log(_rotation_from_quat(_twist_about(q, axes[0])))
    free = np.cross(axes[0], axes[1])
    free /= np.linalg.norm(free)
    twist = _twist_about(q, free)
    swing = _quat_mul(q, twist * np.array([1.0, -1.0, -1.0, -1.0]))
    return rotation_log(_rotation_from_quat(swing))


def rotation_projector(axes: Optional[np.ndarray]) -> np.ndarray:
    """3x3 projector onto the constrained rotation axes (identity for None)."""
    if axes is None:
        return np.eye(3)
    axes = np.asarray(axes, dtype=float).reshape(-1, 3)
    return sum((np.outer(a, a) for a in axes), np.zeros((3, 3)))


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
        self.continuous_mask = (None if continuous_mask is None
                                else np.asarray(continuous_mask, dtype=bool))
        if self.continuous_mask is not None:
            for j, cont in enumerate(self.continuous_mask):
                if cont:
                    self.positions[:, j] = np.unwrap(self.positions[:, j])

    @property
    def duration(self) -> float:
        return float(self.times[-1])

    @property
    def n_joints(self) -> int:
        return self.positions.shape[1]

    def prepend_start(self, q_now: np.ndarray) -> None:
        """A plan whose first point is later than t=0 starts from q_now.

        q_now usually comes from the arm model, which reports continuous
        joints wrapped into (-pi, pi], while the plan may use any turn (the
        Gen2 reports multi-turn angles, e.g. 7.73 rad). Each continuous joint
        of q_now is shifted by whole turns to the equivalent angle nearest
        the plan's first point; otherwise the plan sweeps a full turn.
        """
        if self.times[0] <= 0.0:
            return
        q_now = np.array(q_now, dtype=float)
        if self.continuous_mask is not None:
            first = self.positions[0]
            for j, cont in enumerate(self.continuous_mask):
                if cont:
                    q_now[j] = first[j] + wrap_angle(q_now[j] - first[j])
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


def _min_jerk(s: float) -> tuple[float, float]:
    """(position, d/ds) of the minimum-jerk blend on s in [0, 1]."""
    s = min(max(s, 0.0), 1.0)
    return 10 * s**3 - 15 * s**4 + 6 * s**5, 30 * s**2 - 60 * s**3 + 30 * s**4


MIN_JERK_PEAK = 1.875        # peak of d/ds of the minimum-jerk blend


class CartesianPathReference:
    """A path of the tracking spring's anchor point through world-frame
    points, with an optional orientation target.

    times: (N,) s from start, > 0, strictly increasing. points: (N, 3) world.
    Each segment (from the previous point, the first from where the anchor
    is at start) uses a minimum-jerk blend, so every waypoint is reached at
    rest. Orientation turns from the gripper's start orientation to
    `rotation` over the whole duration (also minimum-jerk); only the
    constrained axes (`rotation_axes`, world unit vectors, (k, 3)) matter,
    and with none the gripper's orientation is free.
    """

    def __init__(self, times: Sequence[float], points, rotation: Optional[np.ndarray] = None,
                 rotation_axes: Optional[np.ndarray] = None) -> None:
        self.times = np.asarray(times, dtype=float)
        self.points = np.array(points, dtype=float).reshape(-1, 3)
        if self.times.ndim != 1 or len(self.times) == 0:
            raise ValueError("Path has no points.")
        if len(self.times) != len(self.points):
            raise ValueError("Every point needs a time.")
        if self.times[0] <= 0.0 or np.any(np.diff(self.times) <= 0.0):
            raise ValueError("times must be > 0 and strictly increasing.")
        if not (np.all(np.isfinite(self.points)) and np.all(np.isfinite(self.times))):
            raise ValueError("Path contains NaN or infinite values.")
        self.rotation = None if rotation is None else np.asarray(rotation, dtype=float)
        self.rotation_axes = (np.zeros((0, 3)) if rotation_axes is None
                              else np.asarray(rotation_axes, dtype=float).reshape(-1, 3))
        if self.rotation_axes.shape[0] and self.rotation is None:
            raise ValueError("Constrained rotation axes need a target orientation.")
        self._start_rotation: Optional[np.ndarray] = None
        self._rot_delta = np.zeros(3)          # log(R_start^T R_target), start frame

    @property
    def duration(self) -> float:
        return float(self.times[-1])

    @property
    def started(self) -> bool:
        return self.times[0] == 0.0

    def prepend_start(self, point_now: np.ndarray, rotation_now: np.ndarray) -> None:
        """Start the path from where the anchor point and gripper are now."""
        if self.started:
            return
        self.times = np.concatenate([[0.0], self.times])
        self.points = np.vstack([np.asarray(point_now, dtype=float), self.points])
        self._start_rotation = np.asarray(rotation_now, dtype=float).copy()
        if self.rotation is not None:
            self._rot_delta = rotation_log(self._start_rotation.T @ self.rotation)

    def peak_speed(self) -> float:
        """Peak anchor speed (m/s) over the path; needs prepend_start first."""
        seg = np.linalg.norm(np.diff(self.points, axis=0), axis=1)
        return float(np.max(MIN_JERK_PEAK * seg / np.diff(self.times))) if len(seg) else 0.0

    def sample(self, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """(point, rotation, linear velocity, angular velocity) at time t (world)."""
        if not self.started:
            raise RuntimeError("prepend_start() before sampling.")
        T = self.duration
        # Orientation: minimum-jerk over the whole path, from the start orientation.
        b, db = _min_jerk(t / T)
        R0 = self._start_rotation
        R = R0 @ rotation_exp(b * self._rot_delta)
        omega = R0 @ (self._rot_delta * db / T) if 0.0 < t < T else np.zeros(3)
        if t <= 0.0:
            return self.points[0].copy(), R, np.zeros(3), np.zeros(3)
        if t >= T:
            return self.points[-1].copy(), R, np.zeros(3), np.zeros(3)
        k = int(np.searchsorted(self.times, t, side="right")) - 1
        t0, t1 = self.times[k], self.times[k + 1]
        h = t1 - t0
        s, ds = _min_jerk((t - t0) / h)
        p0, p1 = self.points[k], self.points[k + 1]
        return p0 + s * (p1 - p0), R, (p1 - p0) * ds / h, omega


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
        # Rotation axes the spring holds (world unit vectors, (k, 3)); None =
        # all three. Free axes get no rotational stiffness or damping.
        self.rotation_axes: Optional[np.ndarray] = None
        self._last_state: Optional[TrackingSpringState] = None

    def set_rotation_axes(self, axes: Optional[np.ndarray]) -> None:
        self.rotation_axes = None if axes is None else np.asarray(axes, dtype=float).reshape(-1, 3)

    def set_anchor(self, position, rotation, twist=None, joints=None) -> None:
        self.anchor_position = np.asarray(position, dtype=float).copy()
        self.anchor_rotation = np.asarray(rotation, dtype=float).copy()
        self.anchor_twist = np.zeros(6) if twist is None else np.asarray(twist, dtype=float)
        self.anchor_joints = None if joints is None else np.asarray(joints, dtype=float).copy()

    def clear_anchor(self) -> None:
        self.anchor_position = self.anchor_rotation = self.anchor_joints = None
        self.anchor_twist = np.zeros(6)
        self.rotation_axes = None
        self._last_state = None

    @property
    def has_anchor(self) -> bool:
        return self.anchor_position is not None

    @property
    def last_state(self) -> Optional[TrackingSpringState]:
        return self._last_state

    def pose_error(self, arm) -> tuple[np.ndarray, np.ndarray]:
        """(x_d − x, rotation error about the constrained axes) for the current arm state."""
        T = arm.get_link_transform(self.link_name)
        R = T[:3, :3]
        x = R @ self.local_attachment_point + T[:3, 3]
        return (self.anchor_position - x,
                masked_rotation_error(self.anchor_rotation @ R.T, self.rotation_axes))

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
        e_R = masked_rotation_error(self.anchor_rotation @ R.T, self.rotation_axes)
        P = rotation_projector(self.rotation_axes)     # free rotation axes: no damping either
        F = self.stiffness * e_p + self.damping * (self.anchor_twist[:3] - twist[:3])
        M = self.rot_stiffness * e_R + self.rot_damping * (P @ (self.anchor_twist[3:] - twist[3:]))
        F = _saturate(F, self.max_force)
        M = _saturate(M, self.max_moment)
        wrench = np.concatenate([F, M])
        torques = J.T @ wrench

        if self.anchor_joints is not None and (self.null_stiffness > 0 or self.null_damping > 0):
            q = np.array([arm.get_joint_angle(i) for i in range(n)])
            tau0 = (self.null_stiffness * wrap_angle(self.anchor_joints - q)
                    - self.null_damping * qdot)
            # Damped pseudo-inverse of the task the spring controls (position
            # + constrained rotation axes); N = I − Jtᵀ Jt#ᵀ keeps tau0 out of
            # that task, so free rotation axes are left to the posture term.
            Jt = np.vstack([J[:3], P @ J[3:]])
            lam2 = 1e-4
            J_pinv = Jt.T @ np.linalg.inv(Jt @ Jt.T + lam2 * np.eye(6))
            torques = torques + (np.eye(n) - Jt.T @ J_pinv.T) @ tau0

        self._last_state = TrackingSpringState(x, e_p, e_R, wrench, torques)
        return torques

    def __repr__(self) -> str:
        return (f"TrackingPoseSpring(name={self.name!r}, link={self.link_name!r}, "
                f"k={self.stiffness}, b={self.damping}, k_rot={self.rot_stiffness}, "
                f"enabled={self.enabled})")


def _saturate(v: np.ndarray, limit: float) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    return v if norm <= limit or norm == 0.0 else v * (limit / norm)


# ── Oscillation watchdog ──────────────────────────────────────────────────────

class OscillationWatchdog:
    """Spots the spring driving the arm into a sustained oscillation.

    The loop on the Gen2 has ~60 ms of delay (docs/gen2_stability_analysis.md),
    so stiff gains go unstable at a few Hz. Normal tracking makes the spring
    force change slowly; an oscillation makes it swing back and forth. Each
    update high-passes the force and moment (subtracting an exponential
    average with time constant hp_tau_s), and over the last window_s looks at
    the swing's RMS and how often its main component reverses sign. It trips
    when either swing is above its RMS limit AND reverses at min_hz..max_hz.

    One step, a slow pull, noise or a single bump don't trip it: they give no
    sustained reversals, or too small an RMS.
    """

    def __init__(self, force_rms_n: float = 2.5, moment_rms_nm: float = 0.6,
                 window_s: float = 1.0, hp_tau_s: float = 0.25,
                 min_hz: float = 1.5, max_hz: float = 20.0) -> None:
        self.force_rms_n = force_rms_n
        self.moment_rms_nm = moment_rms_nm
        self.window_s = window_s
        self.hp_tau_s = hp_tau_s
        self.min_hz = min_hz
        self.max_hz = max_hz
        self.reset()

    def reset(self) -> None:
        self._last_t: Optional[float] = None
        self._slow_f = self._slow_m = None
        self._buf: list = []                      # (t, hp force, hp moment)
        self.last = {"force_rms": 0.0, "moment_rms": 0.0, "freq_hz": 0.0}

    def _swing(self, arr: np.ndarray, span: float) -> tuple[float, float]:
        """(RMS of the vector swing, reversal frequency of its main component)."""
        rms = float(np.sqrt(np.mean(np.sum(arr * arr, axis=1))))
        if rms <= 1e-9:
            return 0.0, 0.0
        axis = np.linalg.svd(arr - arr.mean(axis=0), full_matrices=False)[2][0]
        proj = arr @ axis
        # Count sign changes with hysteresis, ignoring wobble near zero.
        band, sign, flips = 0.2 * rms, 0, 0
        for p in proj:
            s = 1 if p > band else (-1 if p < -band else 0)
            if s and sign and s != sign:
                flips += 1
            if s:
                sign = s
        return rms, flips / (2.0 * span)

    def update(self, t: float, force, moment) -> Optional[str]:
        """Feed one control tick. Returns why it tripped, or None."""
        f = np.asarray(force, dtype=float)
        m = np.asarray(moment, dtype=float)
        if self._last_t is None:
            self._slow_f, self._slow_m, self._last_t = f.copy(), m.copy(), t
            return None
        dt = max(0.0, t - self._last_t)
        self._last_t = t
        a = dt / (self.hp_tau_s + dt)
        self._slow_f = self._slow_f + a * (f - self._slow_f)
        self._slow_m = self._slow_m + a * (m - self._slow_m)
        self._buf.append((t, f - self._slow_f, m - self._slow_m))
        while self._buf and t - self._buf[0][0] > self.window_s:
            self._buf.pop(0)
        span = self._buf[-1][0] - self._buf[0][0]
        if span < 0.8 * self.window_s:
            return None
        hf = np.array([b[1] for b in self._buf])
        hm = np.array([b[2] for b in self._buf])
        f_rms, f_hz = self._swing(hf, span)
        m_rms, m_hz = self._swing(hm, span)
        self.last = {"force_rms": round(f_rms, 3), "moment_rms": round(m_rms, 3),
                     "freq_hz": round(f_hz if f_rms / self.force_rms_n >= m_rms / self.moment_rms_nm
                                      else m_hz, 2)}
        if f_rms > self.force_rms_n and self.min_hz <= f_hz <= self.max_hz:
            return f"spring force oscillating {f_rms:.1f} N RMS at about {f_hz:.1f} Hz"
        if m_rms > self.moment_rms_nm and self.min_hz <= m_hz <= self.max_hz:
            return f"spring moment oscillating {m_rms:.2f} N*m RMS at about {m_hz:.1f} Hz"
        return None


# ── Follower ──────────────────────────────────────────────────────────────────

class FollowerState(enum.Enum):
    IDLE = "IDLE"          # no anchor; spring contributes nothing
    RUNNING = "RUNNING"    # anchor moving along the plan
    PAUSED = "PAUSED"      # tracking error too large: trajectory time frozen
    HOLDING = "HOLDING"    # plan finished (or stopped); anchor holds a fixed pose


@dataclass
class FollowerConfig:
    # Pause (freeze trajectory time) when the tracking spring pulls harder
    # than this: the allowed lag follows from its strength (force / stiffness,
    # moment / rot_stiffness; unsaturated). Defaults = 8 cm / 0.5 rad at the
    # default 300 N/m / 10 N*m/rad.
    pause_force_n: float = 24.0
    pause_moment_nm: float = 5.0
    resume_fraction: float = 0.6       # resume once both are < this × the pause limits
    pause_timeout_s: float = 10.0
    # Done when the spring that holds the final pose afterwards pulls with at
    # most this force/moment -- see TrajectoryFollower.hold_stiffness. A
    # softer spring is automatically more forgiving.
    goal_force_tol_n: float = 4.0
    goal_moment_tol_nm: float = 1.2
    # Per-joint check on top of that, at the end of the plan; <= 0 = off. A
    # goal's own goal_tolerance entries still apply.
    goal_tol_rad: float = 0.0
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
        self._goal_tol = np.full(arm.n_dof, config.goal_tol_rad if config.goal_tol_rad > 0
                                 else np.inf)
        # Stiffness of the static spring(s) that hold the gripper after the
        # trajectory (e.g. the tip-hold spring); the end tolerance is
        # goal_force_tol_n / hold_stiffness. None = the tracking spring's own.
        self.hold_stiffness: Optional[float] = None
        self.hold_rot_stiffness: Optional[float] = None
        self.hold_label = ""           # for messages, e.g. "tip_spring"
        self._posture: Optional[np.ndarray] = None   # Cartesian paths: joints at start

    # ── commands ──

    def current_angles(self) -> np.ndarray:
        return np.array([self.arm.get_joint_angle(i) for i in range(self.arm.n_dof)])

    def current_anchor_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """(attachment point, link rotation) where the arm is now, world."""
        p, R, _ = self.arm.fk_and_jacobian_at(self.current_angles(), self.spring.link_name,
                                              self.spring.local_attachment_point)
        return np.asarray(p), np.asarray(R)

    def start(self, reference, now: float,
              goal_tol_rad: Optional[Sequence[float]] = None,
              goal_time_tol_s: Optional[float] = None) -> tuple[bool, str]:
        """Start a JointTrajectoryReference or a CartesianPathReference."""
        if self.state in (FollowerState.RUNNING, FollowerState.PAUSED):
            return False, "A trajectory is already running."
        q_now = self.current_angles()
        if isinstance(reference, CartesianPathReference):
            p_now, R_now = self.current_anchor_pose()
            reference.prepend_start(p_now, R_now)
            self._posture = q_now                 # posture term holds the start joints
            self.spring.set_rotation_axes(reference.rotation_axes)
            goal_tol_rad = np.full(self.arm.n_dof, np.inf)   # no joint goal to check
        else:
            if reference.n_joints != self.arm.n_dof:
                return False, (f"Trajectory has {reference.n_joints} joints, "
                               f"arm has {self.arm.n_dof}.")
            reference.prepend_start(q_now)
            # Unwrapping may have shifted continuous joints by whole turns; FK
            # doesn't care, and errors are wrapped, so compare wrapped.
            start_err = np.abs(wrap_angle(reference.positions[0] - q_now))
            if np.max(start_err) > self.cfg.start_tol_rad:
                j = int(np.argmax(start_err))
                return False, (f"Trajectory starts {start_err[j]:.3f} rad from the current pose "
                               f"on joint {j + 1} (allowed {self.cfg.start_tol_rad:.3f}).")
            self.spring.set_rotation_axes(None)  # joint plans hold the full orientation
        self.reference = reference
        default_tol = self.cfg.goal_tol_rad if self.cfg.goal_tol_rad > 0 else np.inf
        self._goal_tol = (np.full(self.arm.n_dof, default_tol) if goal_tol_rad is None
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
        # How hard the tracking spring is pulling (unsaturated).
        pull_f = self.spring.stiffness * ep
        pull_m = self.spring.rot_stiffness * eR
        if self.state is FollowerState.RUNNING:
            if pull_f > cfg.pause_force_n or pull_m > cfg.pause_moment_nm:
                self.state = FollowerState.PAUSED
                self._paused_since = now
            else:
                self.t_traj += dt
        if self.state is FollowerState.PAUSED:
            if (pull_f < cfg.resume_fraction * cfg.pause_force_n
                    and pull_m < cfg.resume_fraction * cfg.pause_moment_nm):
                self.state = FollowerState.RUNNING
                self._paused_since = None
            elif now - self._paused_since > cfg.pause_timeout_s:
                return self.stop(
                    PATH_TOLERANCE_VIOLATED,
                    f"Held off the path for more than {cfg.pause_timeout_s:.1f}s: spring "
                    f"pulling {pull_f:.1f} N ({ep * 100:.1f} cm off) and {pull_m:.2f} N*m "
                    f"({eR:.2f} rad off); resumes below "
                    f"{cfg.resume_fraction * cfg.pause_force_n:.1f} N and "
                    f"{cfg.resume_fraction * cfg.pause_moment_nm:.2f} N*m.")

        ref = self.reference
        self._set_anchor_at(self.t_traj, moving=self.state is FollowerState.RUNNING)
        if self.t_traj < ref.duration:
            return None

        if self._ended_at is None:
            self._ended_at = now
        q = self.current_angles()
        qdot = np.asarray(self.arm.joint_velocities, dtype=float)
        joint_err = (np.zeros(self.arm.n_dof) if isinstance(ref, CartesianPathReference)
                     else np.abs(wrap_angle(ref.positions[-1] - q)))
        # The anchor is now the final pose: how hard would the spring that
        # holds it from here on be pulling?
        k_hold = self.spring.stiffness if self.hold_stiffness is None else self.hold_stiffness
        kr_hold = (self.spring.rot_stiffness if self.hold_rot_stiffness is None
                   else self.hold_rot_stiffness)
        force = k_hold * ep
        moment = kr_hold * eR
        force_ok = force <= cfg.goal_force_tol_n
        moment_ok = moment <= cfg.goal_moment_tol_nm
        joints_ok = bool(np.all(joint_err <= self._goal_tol))
        stopped = bool(np.all(np.abs(qdot) <= cfg.stopped_vel_rad_s))
        if force_ok and moment_ok and joints_ok and stopped:
            self.state = FollowerState.HOLDING      # anchor stays on the final pose
            self.reference = None
            return FollowerOutcome(SUCCESSFUL, True, "Trajectory finished.")
        if now - self._ended_at > self._goal_time_tol:
            why = []
            if not force_ok:
                src = (f"{self.hold_label} at {k_hold:g} N/m" if self.hold_label
                       else f"{k_hold:g} N/m")
                tol_cm = cfg.goal_force_tol_n / k_hold * 100    # k_hold > 0 here: force > tol >= 0
                why.append(f"spring force {force:.2f} N > {cfg.goal_force_tol_n:.2f} N "
                           f"({ep * 100:.1f} cm off, {src}; tolerance {tol_cm:.1f} cm)")
            if not moment_ok:
                why.append(f"spring moment {moment:.2f} N*m > {cfg.goal_moment_tol_nm:.2f} N*m "
                           f"({eR:.2f} rad off)")
            if not joints_ok:
                j = int(np.argmax(joint_err - self._goal_tol))
                why.append(f"joint {j + 1} {joint_err[j]:.3f} rad off (goal tolerance "
                           f"{self._goal_tol[j]:.3f})")
            if not stopped:
                why.append("still moving")
            outcome = FollowerOutcome(
                GOAL_TOLERANCE_VIOLATED, False,
                f"Not within goal tolerance {self._goal_time_tol:.1f}s after the plan ended: "
                + "; ".join(why) + ".")
            self.hold_current()
            self.reference = None
            return outcome
        return None

    def _set_anchor_at(self, t: float, moving: bool) -> None:
        if isinstance(self.reference, CartesianPathReference):
            p, R, v, w = self.reference.sample(t)
            twist = np.concatenate([v, w]) if moving else np.zeros(6)
            self.spring.set_anchor(p, R, twist, self._posture)
            return
        q_d, qd_d = self.reference.sample(t)
        p, R, J = self.arm.fk_and_jacobian_at(q_d, self.spring.link_name,
                                              self.spring.local_attachment_point)
        twist = J @ qd_d if moving else np.zeros(6)
        self.spring.set_anchor(p, R, twist, q_d)
