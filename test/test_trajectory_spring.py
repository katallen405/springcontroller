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
    GOAL_TOLERANCE_VIOLATED, PATH_TOLERANCE_VIOLATED, SUCCESSFUL, FollowerConfig,
    FollowerState, JointTrajectoryReference, TrackingPoseSpring, TrajectoryFollower,
    rotation_log, wrap_angle,
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
    cfg = FollowerConfig(goal_time_tol_s=0.5, pause_error_m=10.0, pause_error_rad=10.0)
    # Too weak to beat the joint friction and finish within the tolerance.
    sim = Sim(Q0, cfg=cfg, stiffness=0.0, damping=0.0, rot_stiffness=0.01, rot_damping=0.0,
              null_stiffness=0.0, null_damping=0.0)
    sim.follower.start(JointTrajectoryReference(*plan(Q0, dq, duration=0.5)), sim.t)
    sim.run(2.0)
    assert sim.outcomes and sim.outcomes[0].error_code == GOAL_TOLERANCE_VIOLATED


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
