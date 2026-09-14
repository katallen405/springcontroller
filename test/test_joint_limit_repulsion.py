"""
test_joint_limit_repulsion.py

Unit tests for URDFArmConfiguration.get_joint_limit_repulsion_torques() --
the per-joint proxy for self-collision risk, pushing joints 2/4/6 back
toward center as they near their own position limits (see
gen3_kinova_flat.urdf: those three are the only revolute, non-continuous
joints on this arm -- 1/3/5/7 are continuous and have no limit).

Uses the real Gen3 URDF (no collision geometry needed -- this field never
touches the collision model) so the limits under test are the same ones
get_gravity_torques()/get_jacobian() already trust, not a second,
hand-copied source of truth.
"""

from pathlib import Path

import numpy as np
import pytest

from springcontroller.urdf_arm_configuration import URDFArmConfiguration

URDF_PATH = str(
    Path(__file__).resolve().parents[1]
    / "springcontroller" / "flat_urdf_files" / "gen3_kinova_flat.urdf"
)

MARGIN = 0.175  # ~10 degrees
MAX_TORQUE = 2.0

# Same 6 gripper joints study_control_panel.yaml locks -- without this the
# model carries 13 DOF (arm + gripper) instead of the 7 every other test
# here assumes.
LOCKED_JOINT_NAMES = [
    "robotiq_85_left_knuckle_joint",
    "robotiq_85_right_knuckle_joint",
    "robotiq_85_left_inner_knuckle_joint",
    "robotiq_85_right_inner_knuckle_joint",
    "robotiq_85_left_finger_tip_joint",
    "robotiq_85_right_finger_tip_joint",
]

# joint_2, joint_4, joint_6 -- the only limited (non-continuous) joints on
# this arm -- at DOF indices 1, 3, 5 (joint_names order: 1..7).
LIMITED_DOF_INDICES = [1, 3, 5]
LIMITED_UPPER = {1: 2.24, 3: 2.57, 5: 2.09}


def make_arm(angles):
    arm = URDFArmConfiguration.from_urdf(URDF_PATH, locked_joint_names=LOCKED_JOINT_NAMES)
    arm.update_from_angles(np.asarray(angles, dtype=float))
    return arm


def test_zero_at_center_pose():
    arm = make_arm(np.zeros(7))
    torques = arm.get_joint_limit_repulsion_torques(MARGIN, MAX_TORQUE)
    assert torques.shape == (7,)
    np.testing.assert_array_equal(torques, np.zeros(7))


def test_continuous_joints_never_contribute():
    # Joints 1/3/5/7 have no real limit -- even at an extreme angle they
    # must stay exactly zero.
    angles = np.zeros(7)
    angles[[0, 2, 4, 6]] = 100.0  # continuous -- any angle is valid
    arm = make_arm(angles)
    torques = arm.get_joint_limit_repulsion_torques(MARGIN, MAX_TORQUE)
    np.testing.assert_array_equal(torques[[0, 2, 4, 6]], 0.0)


def test_zero_outside_margin():
    angles = np.zeros(7)
    angles[1] = LIMITED_UPPER[1] - MARGIN - 0.1  # just outside the margin
    arm = make_arm(angles)
    torques = arm.get_joint_limit_repulsion_torques(MARGIN, MAX_TORQUE)
    assert torques[1] == 0.0


@pytest.mark.parametrize("dof", LIMITED_DOF_INDICES)
def test_pushes_away_from_upper_limit(dof):
    # dist=0.01 is well inside MARGIN=0.175, so this checks direction and
    # near-saturation strength, not the exact ramp value (see
    # test_ramp_follows_quadratic_falloff/test_saturates_beyond_limit for
    # exact-magnitude coverage).
    angles = np.zeros(7)
    angles[dof] = LIMITED_UPPER[dof] - 0.01  # just inside the upper limit
    arm = make_arm(angles)
    torques = arm.get_joint_limit_repulsion_torques(MARGIN, MAX_TORQUE)
    assert torques[dof] < 0.0  # push back toward center (negative)
    assert abs(torques[dof]) > 0.8 * MAX_TORQUE


@pytest.mark.parametrize("dof", LIMITED_DOF_INDICES)
def test_pushes_away_from_lower_limit(dof):
    angles = np.zeros(7)
    angles[dof] = -LIMITED_UPPER[dof] + 0.01  # just inside the lower limit
    arm = make_arm(angles)
    torques = arm.get_joint_limit_repulsion_torques(MARGIN, MAX_TORQUE)
    assert torques[dof] > 0.0  # push back toward center (positive)
    assert torques[dof] > 0.8 * MAX_TORQUE


def test_ramp_follows_quadratic_falloff():
    dof = 1  # joint_2
    upper = LIMITED_UPPER[dof]

    dist_a, dist_b = 0.10, 0.05  # both inside MARGIN=0.175
    arm_a = make_arm(np.zeros(7))
    arm_a.update_from_angles(np.array([0, upper - dist_a, 0, 0, 0, 0, 0]))
    mag_a = abs(arm_a.get_joint_limit_repulsion_torques(MARGIN, MAX_TORQUE)[dof])

    arm_b = make_arm(np.zeros(7))
    arm_b.update_from_angles(np.array([0, upper - dist_b, 0, 0, 0, 0, 0]))
    mag_b = abs(arm_b.get_joint_limit_repulsion_torques(MARGIN, MAX_TORQUE)[dof])

    t_a = (MARGIN - dist_a) / MARGIN
    t_b = (MARGIN - dist_b) / MARGIN
    expected_ratio = (t_b / t_a) ** 2

    assert mag_a > 0.0
    assert mag_b > mag_a  # closer to the limit -> stronger
    np.testing.assert_allclose(mag_b / mag_a, expected_ratio, rtol=1e-6)


def test_saturates_beyond_limit():
    # Right at the limit and just past it (still a valid angle to *ask
    # about*, even though the real joint would never get there) must both
    # saturate at exactly max_torque_nm -- no runaway growth past the cap.
    dof = 1
    upper = LIMITED_UPPER[dof]
    arm_at = make_arm(np.array([0, upper, 0, 0, 0, 0, 0]))
    arm_past = make_arm(np.array([0, upper + 0.05, 0, 0, 0, 0, 0]))
    t_at = arm_at.get_joint_limit_repulsion_torques(MARGIN, MAX_TORQUE)[dof]
    t_past = arm_past.get_joint_limit_repulsion_torques(MARGIN, MAX_TORQUE)[dof]
    np.testing.assert_allclose(t_at, -MAX_TORQUE, atol=1e-6)
    np.testing.assert_allclose(t_past, -MAX_TORQUE, atol=1e-6)


def test_zero_margin_or_max_torque_returns_zero_vector():
    angles = np.zeros(7)
    angles[1] = LIMITED_UPPER[1] - 0.01
    arm = make_arm(angles)
    np.testing.assert_array_equal(
        arm.get_joint_limit_repulsion_torques(0.0, MAX_TORQUE), np.zeros(7)
    )
    np.testing.assert_array_equal(
        arm.get_joint_limit_repulsion_torques(MARGIN, 0.0), np.zeros(7)
    )


def test_multiple_joints_near_limits_independent():
    angles = np.zeros(7)
    angles[1] = LIMITED_UPPER[1]  # exactly at the limit -> saturated
    angles[3] = -LIMITED_UPPER[3]
    arm = make_arm(angles)
    torques = arm.get_joint_limit_repulsion_torques(MARGIN, MAX_TORQUE)
    np.testing.assert_allclose(torques[1], -MAX_TORQUE, atol=1e-6)
    np.testing.assert_allclose(torques[3], MAX_TORQUE, atol=1e-6)
    np.testing.assert_array_equal(torques[[0, 2, 4, 5, 6]], 0.0)
