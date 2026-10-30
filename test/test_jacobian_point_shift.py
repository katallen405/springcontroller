"""
test_jacobian_point_shift.py

URDFArmConfiguration.get_jacobian(link, local_point) must be the Jacobian of
the attachment point (link origin + local_point, rigidly attached), not of
the link origin. Checked against central finite differences of FK on the
real Gen3 and Gen2 URDFs -- see docs/JACOBIAN_SIGN_NOTE.md for the sign bug
this guards against (the shift was `+=` where it must be `-=`).
"""

from pathlib import Path

import numpy as np
import pytest

from springcontroller.urdf_arm_configuration import URDFArmConfiguration

URDF_DIR = Path(__file__).resolve().parents[1] / "springcontroller" / "flat_urdf_files"

GEN3_LOCKED = [
    "robotiq_85_left_knuckle_joint",
    "robotiq_85_right_knuckle_joint",
    "robotiq_85_left_inner_knuckle_joint",
    "robotiq_85_right_inner_knuckle_joint",
    "robotiq_85_left_finger_tip_joint",
    "robotiq_85_right_finger_tip_joint",
]
GEN2_LOCKED = [
    "gripper_finger1_joint",
    "gripper_finger2_joint",
    "gripper_finger1_inner_knuckle_joint",
    "gripper_finger2_inner_knuckle_joint",
    "gripper_finger1_finger_tip_joint",
    "gripper_finger2_finger_tip_joint",
]

CASES = [
    ("gen3_kinova_flat.urdf", GEN3_LOCKED, "end_effector_link"),
    ("gen2_robotiq_flat.urdf", GEN2_LOCKED, "j2s7s300_ee_link"),
]
LOCAL_POINTS = [
    np.zeros(3),
    np.array([0.0, 0.0, 0.1]),       # gen3_springs.yaml tip offset
    np.array([0.03, -0.05, 0.12]),   # off-axis, exercises every skew term
]
EPS = 1e-6


def point_world(arm, angles, link, local_point):
    arm.update_from_angles(angles)
    T = arm.get_link_transform(link)
    return T[:3, :3] @ local_point + T[:3, 3]


def fd_point_jacobian(arm, angles, link, local_point):
    J = np.zeros((3, len(angles)))
    for i in range(len(angles)):
        d = np.zeros(len(angles))
        d[i] = EPS
        J[:, i] = (point_world(arm, angles + d, link, local_point)
                   - point_world(arm, angles - d, link, local_point)) / (2 * EPS)
    return J


@pytest.mark.parametrize("urdf,locked,link", CASES)
@pytest.mark.parametrize("local_point", LOCAL_POINTS, ids=["origin", "tip", "offaxis"])
def test_linear_jacobian_matches_finite_difference(urdf, locked, link, local_point):
    arm = URDFArmConfiguration.from_urdf(str(URDF_DIR / urdf), locked_joint_names=locked)
    rng = np.random.default_rng(0)
    for _ in range(5):
        angles = rng.uniform(-2.0, 2.0, 7)
        J_fd = fd_point_jacobian(arm, angles, link, local_point)
        arm.update_from_angles(angles)
        J = arm.get_jacobian(link, local_point)
        np.testing.assert_allclose(J[:3, :], J_fd, atol=1e-6)


@pytest.mark.parametrize("urdf,locked,link", CASES)
def test_angular_jacobian_unaffected_by_local_point(urdf, locked, link):
    arm = URDFArmConfiguration.from_urdf(str(URDF_DIR / urdf), locked_joint_names=locked)
    arm.update_from_angles(np.linspace(-1.0, 1.0, 7))
    J0 = arm.get_jacobian(link, np.zeros(3))
    J1 = arm.get_jacobian(link, np.array([0.03, -0.05, 0.12]))
    np.testing.assert_allclose(J1[3:, :], J0[3:, :])
