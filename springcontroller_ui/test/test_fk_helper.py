"""
Offline regression test for the forward-kinematics helper the study control
panel's backend node relies on (springcontroller.urdf_arm_configuration.
URDFArmConfiguration, reused as-is -- not reimplemented here).

No ROS, no rclpy, no hardware -- just loads the checked-in flat URDF via
pinocchio and checks get_link_transform against reference values computed
once from that same URDF (see the git history of this file for how they
were derived). This is a "did anything change FK's behavior" regression
check, not an independent cross-validation against real hardware -- that
happens separately, on the real arm (see the plan's real-hardware
checklist).
"""
import os

import numpy as np
import pytest

from springcontroller.urdf_arm_configuration import URDFArmConfiguration

URDF_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "springcontroller", "flat_urdf_files", "gen3_kinova_flat.urdf",
)

LOCKED_JOINT_NAMES = [
    "robotiq_85_left_knuckle_joint",
    "robotiq_85_right_knuckle_joint",
    "robotiq_85_left_inner_knuckle_joint",
    "robotiq_85_right_inner_knuckle_joint",
    "robotiq_85_left_finger_tip_joint",
    "robotiq_85_right_finger_tip_joint",
]


@pytest.fixture
def arm():
    return URDFArmConfiguration.from_urdf(
        URDF_PATH, srdf_path="", locked_joint_names=LOCKED_JOINT_NAMES,
    )


def test_joint_order_matches_expected_names(arm):
    assert arm.joint_names == [
        "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "joint_7",
    ]
    assert arm.n_dof == 7


def test_end_effector_link_is_a_valid_frame(arm):
    arm.validate_link_name("end_effector_link")
    with pytest.raises(ValueError):
        arm.validate_link_name("not_a_real_link")


def test_fk_at_zero_configuration(arm):
    # Gen3 fully "unfolded" straight up -- translation should be close to
    # [0, 0, ~1.207] (the arm's ~1.187m reach plus world's +0.02 table-height
    # offset from base_link, see base_joint in gen3_kinova_flat.urdf).
    T = arm.get_link_transform("end_effector_link")
    np.testing.assert_allclose(T[:3, 3], [0.0, 0.0, 1.20738495], atol=1e-4)
    # NOTE: deliberately not asserting anything about the rotation block
    # here. At exactly the all-zero configuration this URDF/reduced-model
    # combination produces a degenerate (non-orthogonal, det=0) rotation for
    # end_effector_link -- reproducible, and pre-existing in
    # URDFArmConfiguration/the flat URDF (not introduced by this package,
    # and out of scope to fix here). It doesn't affect any current
    # production code path: virtual_spring_node only ever calls
    # update_from_angles with real measured joint angles, never
    # the exact zero configuration. See test_fk_at_nonzero_configuration
    # below for a proper-rotation regression check.


def test_fk_at_nonzero_configuration_is_a_valid_rotation(arm):
    angles = np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7])
    arm.update_from_angles(angles)
    T = arm.get_link_transform("end_effector_link")
    R = T[:3, :3]
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-9)
    assert abs(np.linalg.det(R) - 1.0) < 1e-9


def test_fk_at_nonzero_configuration(arm):
    angles = np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7])
    arm.update_from_angles(angles)
    T = arm.get_link_transform("end_effector_link")
    np.testing.assert_allclose(
        T[:3, 3], [-0.37604036, 0.1319885, 1.04789378], atol=1e-4)
    np.testing.assert_allclose(
        T[:3, :3],
        [[-0.37846897, 0.59389771, -0.7099625],
         [-0.81252407, 0.15422576, 0.56215571],
         [0.4433575, 0.78962011, 0.42418652]],
        atol=1e-4,
    )
