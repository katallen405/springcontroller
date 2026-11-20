"""
test_trajectory_spring.py

TrackingPoseSpring / JointTrajectoryReference / TrajectoryFollower: unit
checks on the real Gen3 URDF, then closed-loop simulations with pinocchio
forward dynamics (pin.aba) -- the follower, spring and gravity compensation
driving a simulated arm along a planned trajectory.
"""

from pathlib import Path

import numpy as np
import pinocchio as pin
import pytest

from springcontroller.trajectory_spring import (
    GOAL_TOLERANCE_VIOLATED, PATH_TOLERANCE_VIOLATED, SUCCESSFUL, CartesianPathReference,
    FollowerConfig, FollowerState, JointTrajectoryReference, TrackingPoseSpring,
    TrajectoryFollower, masked_rotation_error, rotation_exp, rotation_log, wrap_angle,
)
from springcontroller.urdf_arm_configuration import URDFArmConfiguration

URDF = str(Path(__file__).resolve().parents[1] / "springcontroller" / "flat_urdf_files"
           / "gen3_kinova_flat.urdf")
LOCKED = [
    "robotiq_85_left_knuckle_joint", "robotiq_85_right_knuckle_joint",
    "robotiq_85_left_inner_knuckle_joint", "robotiq_85_right_inner_knuckle_joint",
    "robotiq_85_left_finger_tip_joint", "robotiq_85_right_finger_tip_joint",
]
LINK = "end_effector_link"
TIP = np.array([0.0, 0.0, 0.1])
Q0 = np.array([0.0, 0.3, 0.0, 1.8, 0.0, 1.0, 0.0])   # a bent, non-singular pose


def make_arm(q=Q0, qdot=None):
    arm = URDFArmConfiguration.from_urdf(URDF, locked_joint_names=LOCKED)
    arm.update_from_angles(np.asarray(q, float), None if qdot is None else np.asarray(qdot))
    return arm


def anchor_at(arm, spring, q):
    p, R, _ = arm.fk_and_jacobian_at(q, spring.link_name, spring.local_attachment_point)
    spring.set_anchor(p, R, None, q)


# ── math helpers ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("angle", [0.0, 1e-7, 0.3, 2.0, np.pi - 1e-6, np.pi])
def test_rotation_log_matches_pinocchio(angle):
    rng = np.random.default_rng(1)
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    R = pin.exp3(angle * axis)
    w = rotation_log(R)
    np.testing.assert_allclose(pin.exp3(w), R, atol=1e-6)
    assert np.linalg.norm(w) == pytest.approx(angle, abs=1e-6)


def test_wrap_angle():
    np.testing.assert_allclose(wrap_angle([0.0, 3 * np.pi, -3 * np.pi / 2]),
                               [0.0, np.pi, np.pi / 2])


# ── reference ─────────────────────────────────────────────────────────────────

def test_reference_hermite_and_hold():
    ref = JointTrajectoryReference([0.0, 1.0], [[0.0] * 2, [1.0] * 2],
                                   [[0.0] * 2, [0.5] * 2])
    q, v = ref.sample(1.0 - 1e-9)
    assert q[0] == pytest.approx(1.0) and v[0] == pytest.approx(0.5, abs=1e-6)
    q, v = ref.sample(3.0)
    np.testing.assert_allclose(q, [1.0, 1.0])
    np.testing.assert_allclose(v, [0.0, 0.0])


def test_reference_unwraps_continuous_columns():
    ref = JointTrajectoryReference([0.0, 1.0], [[3.0, 3.0], [-3.0, -3.0]],
                                   continuous_mask=[True, False])
    assert ref.positions[1, 0] == pytest.approx(2 * np.pi - 3.0)   # short way round
    assert ref.positions[1, 1] == pytest.approx(-3.0)              # bounded joint untouched


def test_prepend_start_takes_nearest_turn():
    # First Gen2 hardware goal: the arm reported multi-turn angles, the model
    # hands prepend_start wrapped ones. Joints 3, 5, 7 used to sweep a full turn.
    goal = [-1.192, 3.925, -4.164, 5.691, 7.728, 5.045, 5.850]
    arm_now = np.array([-1.192, 3.948, -4.163, 5.691, 7.728, 5.045, 5.650])
    mask = np.array([True, False, True, False, True, False, True])
    ref = JointTrajectoryReference([5.0], [goal], continuous_mask=mask)
    ref.prepend_start(np.where(mask, wrap_angle(arm_now), arm_now))   # as the model reports
    np.testing.assert_allclose(ref.positions[0], arm_now, atol=1e-9)
    for t in np.linspace(0.0, 5.0, 11):
        q, _ = ref.sample(t)
        assert np.max(np.abs(q - arm_now)) < 0.21


@pytest.mark.parametrize("times,pos,msg", [
    ([], np.zeros((0, 2)), "no points"),
    ([0.0, 0.0], np.zeros((2, 2)), "increasing"),
    ([0.0], [[np.nan, 0.0]], "NaN"),
])
def test_reference_validation(times, pos, msg):
    with pytest.raises(ValueError, match=msg):
        JointTrajectoryReference(times, pos)


# ── spring ────────────────────────────────────────────────────────────────────

def test_no_anchor_no_torque():
    spring = TrackingPoseSpring(LINK, TIP)
    np.testing.assert_array_equal(spring.compute_torques(make_arm()), np.zeros(7))


def test_zero_error_zero_torque():
    arm = make_arm()
    spring = TrackingPoseSpring(LINK, TIP)
    anchor_at(arm, spring, Q0)
    np.testing.assert_allclose(spring.compute_torques(arm), np.zeros(7), atol=1e-9)


def _descend(arm, spring, q, steps=40, eta=2e-3):
    """Follow the spring's torque as a gradient: error must shrink."""
    for _ in range(steps):
        arm.update_from_angles(q)
        q = q + eta * spring.compute_torques(arm)
    arm.update_from_angles(q)
    return q


def test_position_error_is_restored():
    arm = make_arm()
    spring = TrackingPoseSpring(LINK, TIP, rot_stiffness=0.0, null_stiffness=0.0)
    anchor_at(arm, spring, Q0)
    spring.anchor_position = spring.anchor_position + np.array([0.03, -0.02, 0.01])
    e0 = np.linalg.norm(spring.pose_error(arm)[0])
    _descend(arm, spring, Q0.copy())
    assert np.linalg.norm(spring.pose_error(arm)[0]) < 0.5 * e0


def test_rotation_error_is_restored():
    arm = make_arm()
    spring = TrackingPoseSpring(LINK, TIP, stiffness=0.0, null_stiffness=0.0, rot_stiffness=10.0)
    anchor_at(arm, spring, Q0)
    spring.anchor_rotation = pin.exp3(np.array([0.0, 0.0, 0.2])) @ spring.anchor_rotation
    e0 = np.linalg.norm(spring.pose_error(arm)[1])
    _descend(arm, spring, Q0.copy(), eta=2e-2)
    assert np.linalg.norm(spring.pose_error(arm)[1]) < 0.5 * e0


def test_nullspace_term_adds_no_end_effector_wrench():
    arm = make_arm()
    spring = TrackingPoseSpring(LINK, TIP, stiffness=0.0, damping=0.0, rot_stiffness=0.0,
                                rot_damping=0.0, null_stiffness=5.0)
    anchor_at(arm, spring, Q0)
    spring.anchor_joints = Q0 + np.array([0.3, 0.0, -0.3, 0.0, 0.2, 0.0, 0.0])
    tau = spring.compute_torques(arm)
    assert np.linalg.norm(tau) > 0.1
    J = arm.get_jacobian(LINK, TIP)
    wrench = np.linalg.solve(J @ J.T, J @ tau)     # wrench whose Jᵀ-image is tau's task part
    assert np.linalg.norm(wrench) < 1e-2 * np.linalg.norm(tau)


def test_force_saturates():
    arm = make_arm()
    spring = TrackingPoseSpring(LINK, TIP, stiffness=1000.0, max_force=20.0, null_stiffness=0.0)
    anchor_at(arm, spring, Q0)
    spring.anchor_position = spring.anchor_position + np.array([0.5, 0.0, 0.0])
    spring.compute_torques(arm)
    assert np.linalg.norm(spring.last_state.force[:3]) == pytest.approx(20.0)


def test_fk_at_planned_pose_leaves_live_state_alone():
    arm = make_arm()
    before = arm.get_link_transform(LINK).copy()
    J_before = arm.get_jacobian(LINK, TIP).copy()
    arm.fk_and_jacobian_at(Q0 + 0.5, LINK, TIP)
    np.testing.assert_array_equal(arm.get_link_transform(LINK), before)
    np.testing.assert_array_equal(arm.get_jacobian(LINK, TIP), J_before)


def test_fk_at_matches_live_fk():
    q = Q0 + 0.2
    arm = make_arm(q)
    p, R, J = arm.fk_and_jacobian_at(q, LINK, TIP)
    T = arm.get_link_transform(LINK)
    np.testing.assert_allclose(R, T[:3, :3], atol=1e-12)
    np.testing.assert_allclose(p, T[:3, :3] @ TIP + T[:3, 3], atol=1e-12)
    np.testing.assert_allclose(J, arm.get_jacobian(LINK, TIP), atol=1e-12)


# ── follower, closed loop ─────────────────────────────────────────────────────

def plan(q_start, dq, duration=3.0, n=31):
    """Smooth (min-jerk) joint-space plan from q_start to q_start + dq."""
    t = np.linspace(0.0, duration, n)
    s = t / duration
    blend = 10 * s**3 - 15 * s**4 + 6 * s**5
    dblend = (30 * s**2 - 60 * s**3 + 30 * s**4) / duration
    pos = q_start + np.outer(blend, dq)
    vel = np.outer(dblend, dq)
    return t, pos, vel


class Sim:
    """Forward dynamics at 1 kHz: gravity compensated, spring torque applied,
    a little joint friction, optional external torque (a hand)."""

    def __init__(self, q0, cfg=None, **spring_kw):
        self.arm = make_arm(q0, np.zeros(7))
        self.model = self.arm._model
        self.data = self.model.createData()
        self.q = np.array(q0, float)
        self.v = np.zeros(7)
        self.t = 0.0
        self.spring = TrackingPoseSpring(LINK, TIP, **spring_kw)
        self.follower = TrajectoryFollower(self.spring, self.arm, cfg or FollowerConfig())
        self.external = np.zeros(7)
        self.hand_force = np.zeros(3)      # world force on the tip, like a hand on the gripper
        self.outcomes = []
        self.max_error = 0.0

    def run(self, seconds, dt=1e-3):
        for _ in range(int(round(seconds / dt))):
            self.arm.update_from_angles(self.q, self.v)
            out = self.follower.tick(self.t)
            if out is not None:
                self.outcomes.append(out)
            hand = self.arm.get_jacobian(LINK, TIP)[:3, :].T @ self.hand_force
            tau = (self.spring.compute_torques(self.arm) + self.arm.get_gravity_torques()
                   - 0.5 * self.v + self.external + hand)
            if self.spring.last_state is not None and self.follower.state is FollowerState.RUNNING:
                self.max_error = max(self.max_error,
                                     np.linalg.norm(self.spring.last_state.position_error))
            a = pin.aba(self.model, self.data, self.arm._angles_to_q(self.q), self.v, tau)
            self.v = self.v + a * dt
            self.q = self.q + self.v * dt
            self.t += dt


def test_closed_loop_follows_plan_and_succeeds():
    dq = np.array([0.4, -0.2, 0.3, -0.3, 0.2, 0.3, 0.0])
    sim = Sim(Q0)
    ok, msg = sim.follower.start(JointTrajectoryReference(*plan(Q0, dq)), sim.t)
    assert ok, msg
    sim.run(6.0)
    assert sim.outcomes and sim.outcomes[0].error_code == SUCCESSFUL, sim.outcomes
    assert sim.max_error < 0.02                         # < 2 cm behind the anchor
    np.testing.assert_allclose(sim.q, Q0 + dq, atol=0.06)
    assert sim.follower.state is FollowerState.HOLDING


def test_held_back_pauses_then_resumes():
    dq = np.array([0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    sim = Sim(Q0)
    sim.follower.start(JointTrajectoryReference(*plan(Q0, dq)), sim.t)
    sim.run(0.5)
    # A hand on the gripper pulling against the direction of travel.
    sim.hand_force = -35.0 * _travel_direction(sim, dq)
    paused = False
    for _ in range(300):
        sim.run(0.01)
        paused |= sim.follower.state is FollowerState.PAUSED
    assert paused
    t_frozen = sim.follower.t_traj
    sim.run(0.2)
    if sim.follower.state is FollowerState.PAUSED:
        assert sim.follower.t_traj == pytest.approx(t_frozen)   # the anchor waits
    sim.hand_force = np.zeros(3)
    sim.run(6.0)
    assert sim.outcomes and sim.outcomes[0].error_code == SUCCESSFUL, sim.outcomes


def _travel_direction(sim, dq):
    """Unit vector from the tip's start to its planned end position."""
    p0, _, _ = sim.arm.fk_and_jacobian_at(Q0, LINK, TIP)
    p1, _, _ = sim.arm.fk_and_jacobian_at(Q0 + dq, LINK, TIP)
    d = p1 - p0
    return d / np.linalg.norm(d)


def test_long_hold_aborts():
    dq = np.array([0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    sim = Sim(Q0, cfg=FollowerConfig(pause_timeout_s=1.0))
    sim.follower.start(JointTrajectoryReference(*plan(Q0, dq)), sim.t)
    sim.run(0.5)
    sim.hand_force = -35.0 * _travel_direction(sim, dq)
    for _ in range(300):
        sim.run(0.01)
        if sim.outcomes:
            break
    assert sim.outcomes and sim.outcomes[0].error_code == PATH_TOLERANCE_VIOLATED
    assert sim.follower.state is FollowerState.HOLDING
    # Re-anchored where the arm is at the abort, not where the plan was heading.
    e_p, _ = sim.spring.pose_error(sim.arm)
    assert np.linalg.norm(e_p) < 0.01


def test_goal_time_tolerance_when_arm_cannot_reach():
    dq = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8])
    # With a spring this weak, any end error is a tiny pull, so the default
    # force/moment tolerances would call it done -- intended. Ask for tight ones.
    cfg = FollowerConfig(goal_time_tol_s=0.5, pause_force_n=1e6, pause_moment_nm=1e6,
                         goal_force_tol_n=1e-4, goal_moment_tol_nm=1e-4)
    # Too weak to beat the joint friction and finish within the tolerance.
    sim = Sim(Q0, cfg=cfg, stiffness=0.0, damping=0.0, rot_stiffness=0.01, rot_damping=0.0,
              null_stiffness=0.0, null_damping=0.0)
    sim.follower.start(JointTrajectoryReference(*plan(Q0, dq, duration=0.5)), sim.t)
    sim.run(2.0)
    assert sim.outcomes and sim.outcomes[0].error_code == GOAL_TOLERANCE_VIOLATED
    assert "spring moment" in sim.outcomes[0].message


def _end_with_hand(hold_stiffness, hand_n=3.0, goal_tol=None, cfg=None):
    """Short plan; a constant hand force keeps the gripper ~1 cm off the end
    (3 N on the 300 N/m tracking spring). Returns the outcome."""
    dq = np.array([0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    sim = Sim(Q0, cfg=cfg)
    sim.follower.hold_stiffness = hold_stiffness
    sim.follower.hold_label = "tip_spring"
    ok, msg = sim.follower.start(JointTrajectoryReference(*plan(Q0, dq, duration=1.0)), sim.t,
                                 goal_tol_rad=goal_tol)
    assert ok, msg
    sim.hand_force = np.array([0.0, 0.0, hand_n])
    sim.run(5.0)
    assert sim.outcomes, "no outcome"
    return sim.outcomes[0]


def test_end_tolerance_follows_hold_spring_stiffness():
    # Same ~1 cm end error: a 10 N/m holding spring pulls ~0.1 N (done); a
    # 1000 N/m one would pull ~10 N, over the 4 N default (abort).
    assert _end_with_hand(10.0).error_code == SUCCESSFUL
    out = _end_with_hand(1000.0)
    assert out.error_code == GOAL_TOLERANCE_VIOLATED
    assert "spring force" in out.message and "tip_spring at 1000 N/m" in out.message


def test_joints_unchecked_by_default_but_goal_tolerance_applies():
    assert _end_with_hand(10.0).error_code == SUCCESSFUL           # joints off, not checked
    out = _end_with_hand(10.0, goal_tol=np.full(7, 1e-5))          # the goal asks
    assert out.error_code == GOAL_TOLERANCE_VIOLATED and "joint" in out.message


def test_pause_is_a_force_limit():
    # At steady state the spring pulls as hard as the hand pushes, whatever
    # its stiffness: so the pause limit is "pushed harder than X N".
    dq = np.array([0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    for pause_n, expect_pause in [(24.0, False), (5.0, True)]:
        sim = Sim(Q0, cfg=FollowerConfig(pause_force_n=pause_n))
        sim.follower.start(JointTrajectoryReference(*plan(Q0, dq)), sim.t)
        sim.run(0.5)
        sim.hand_force = -10.0 * _travel_direction(sim, dq)
        paused = False
        for _ in range(150):
            sim.run(0.01)
            paused |= sim.follower.state is FollowerState.PAUSED
        assert paused == expect_pause, (pause_n, paused)


def test_start_rejected_far_from_plan():
    sim = Sim(Q0)
    t, pos, vel = plan(Q0 + np.array([0.3, 0, 0, 0, 0, 0, 0]), np.zeros(7))
    ok, msg = sim.follower.start(JointTrajectoryReference(t, pos, vel), 0.0)
    assert not ok and "joint 1" in msg
    assert sim.follower.state is FollowerState.IDLE and not sim.spring.has_anchor


def test_stop_holds_current_pose():
    dq = np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    sim = Sim(Q0)
    sim.follower.start(JointTrajectoryReference(*plan(Q0, dq)), sim.t)
    sim.run(1.0)
    out = sim.follower.stop(PATH_TOLERANCE_VIOLATED, "collision")
    assert out.error_code == PATH_TOLERANCE_VIOLATED and not out.succeeded
    q_stop = sim.q.copy()
    sim.run(1.0)
    np.testing.assert_allclose(sim.q, q_stop, atol=0.03)


# ── Cartesian anchor paths ────────────────────────────────────────────────────

X, Y, Z = np.eye(3)


def Rx(a):
    return rotation_exp(a * X)


def Rz(a):
    return rotation_exp(a * Z)


@pytest.mark.parametrize("w", [[0.3, -0.2, 0.5], [0.0, 0.0, 2.5], [1e-8, 0, 0]])
def test_rotation_exp_log_round_trip(w):
    np.testing.assert_allclose(rotation_log(rotation_exp(np.array(w))), w, atol=1e-9)


def test_masked_rotation_error_swing_twist():
    E = Rz(1.2) @ Rx(0.3)
    np.testing.assert_allclose(masked_rotation_error(E, None), rotation_log(E))
    np.testing.assert_allclose(masked_rotation_error(E, np.zeros((0, 3))), 0.0)
    # Only z constrained: the twist about z, nothing else.
    e = masked_rotation_error(E, Z[None])
    np.testing.assert_allclose(e[:2], 0.0, atol=1e-12)
    assert e[2] == pytest.approx(1.2, abs=1e-9)
    assert np.linalg.norm(masked_rotation_error(Rx(0.3), Z[None])) == pytest.approx(0.0, abs=1e-12)
    # z free: a large yaw is no error at all, the tilt is fully kept.
    assert np.linalg.norm(masked_rotation_error(Rz(2.0), np.array([X, Y]))) < 1e-9
    np.testing.assert_allclose(masked_rotation_error(Rx(0.3), np.array([X, Y])), [0.3, 0, 0],
                               atol=1e-9)
    assert np.linalg.norm(masked_rotation_error(E, np.array([X, Y]))) == pytest.approx(0.3, abs=1e-9)


def test_cartesian_path_sampling():
    p0, R0 = np.array([0.1, 0.0, 0.5]), np.eye(3)
    ref = CartesianPathReference([2.0, 4.0], [[0.3, 0.0, 0.5], [0.3, 0.2, 0.5]],
                                 rotation=Rz(0.4), rotation_axes=np.array([X, Y, Z]))
    ref.prepend_start(p0, R0)
    assert ref.duration == 4.0
    for t, p in [(0.0, p0), (2.0, [0.3, 0, 0.5]), (4.0, [0.3, 0.2, 0.5]), (9.0, [0.3, 0.2, 0.5])]:
        np.testing.assert_allclose(ref.sample(t)[0], p, atol=1e-12)
        np.testing.assert_allclose(ref.sample(t)[2], 0.0, atol=1e-12)   # at rest at waypoints
    np.testing.assert_allclose(ref.sample(1.0)[0], [0.2, 0, 0.5], atol=1e-12)   # halfway
    assert ref.peak_speed() == pytest.approx(1.875 * 0.2 / 2.0)
    np.testing.assert_allclose(ref.sample(4.0)[1], Rz(0.4), atol=1e-12)
    np.testing.assert_allclose(ref.sample(2.0)[1], Rz(0.2), atol=1e-9)      # halfway in time
    with pytest.raises(ValueError):
        CartesianPathReference([0.0], [[0, 0, 0]])                          # times must be > 0
    with pytest.raises(ValueError):
        CartesianPathReference([1.0], [[0, 0, 0]], rotation_axes=np.array([Z]))   # no target


def test_free_rotation_axis_gets_no_moment():
    arm = make_arm(Q0)
    T = arm.get_link_transform(LINK)
    x = T[:3, :3] @ TIP + T[:3, 3]
    spring = TrackingPoseSpring(LINK, TIP)
    spring.set_anchor(x, Rz(0.5) @ T[:3, :3])                 # yawed 0.5 rad about world z
    spring.compute_torques(arm)
    assert np.linalg.norm(spring.last_state.force[3:]) > 1.0      # all axes held: pulls back
    spring.set_rotation_axes(np.array([X, Y]))                    # z free
    spring.compute_torques(arm)
    assert np.linalg.norm(spring.last_state.force[3:]) < 1e-6


def _cartesian_sim(path_points, times, rotation=None, axes=None):
    sim = Sim(Q0)
    p0, _ = sim.follower.current_anchor_pose()
    ref = CartesianPathReference(times, [p0 + np.asarray(p) for p in path_points],
                                 rotation=rotation, rotation_axes=axes)
    ok, msg = sim.follower.start(ref, sim.t)
    assert ok, msg
    return sim, p0


def test_cartesian_position_only_path_closed_loop():
    sim, p0 = _cartesian_sim([[0.0, 0.0, 0.10]], [3.0])              # 10 cm straight up
    worst_off_line = 0.0
    for _ in range(300):
        sim.run(0.01)
        x, _ = sim.follower.current_anchor_pose()
        d = x - p0
        worst_off_line = max(worst_off_line, float(np.linalg.norm(d[:2])))
    sim.run(3.0)
    assert sim.outcomes and sim.outcomes[0].error_code == SUCCESSFUL, sim.outcomes
    x, _ = sim.follower.current_anchor_pose()
    np.testing.assert_allclose(x, p0 + [0, 0, 0.10], atol=0.01)
    assert worst_off_line < 0.01                                    # stays on the line
    assert sim.follower.state is FollowerState.HOLDING


def test_cartesian_orientation_target_closed_loop():
    sim = Sim(Q0)
    _, R0 = sim.follower.current_anchor_pose()
    R_target = Rz(0.3) @ R0
    p0, _ = sim.follower.current_anchor_pose()
    ref = CartesianPathReference([3.0], [p0], rotation=R_target, rotation_axes=np.eye(3))
    ok, msg = sim.follower.start(ref, sim.t)
    assert ok, msg
    sim.run(6.0)
    assert sim.outcomes and sim.outcomes[0].error_code == SUCCESSFUL, sim.outcomes
    _, R = sim.follower.current_anchor_pose()
    assert np.linalg.norm(rotation_log(R_target @ R.T)) < 0.05
