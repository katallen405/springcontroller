"""Held-object payload compensation: the holding torques, and the node's
~/set_payload ramp and checks called unbound on a stand-in `self` -- no rclpy
init, no node, no ROS graph."""

import types
import unittest.mock
from pathlib import Path

import numpy as np
import pytest

from springcontroller.urdf_arm_configuration import URDFArmConfiguration
from springcontroller.virtual_spring import GRAVITY_MPS2, payload_torques
from springcontroller.virtual_spring_node import VirtualSpringNode

URDFS = Path(__file__).resolve().parents[1] / "springcontroller" / "flat_urdf_files"
ARMS = {
    "gen2": ("gen2_robotiq_flat.urdf", "j2s7s300_ee_link",
             ["gripper_finger1_joint", "gripper_finger2_joint",
              "gripper_finger1_inner_knuckle_joint", "gripper_finger2_inner_knuckle_joint",
              "gripper_finger1_finger_tip_joint", "gripper_finger2_finger_tip_joint"]),
    "gen3": ("gen3_kinova_flat.urdf", "end_effector_link",
             ["robotiq_85_left_knuckle_joint", "robotiq_85_right_knuckle_joint",
              "robotiq_85_left_inner_knuckle_joint", "robotiq_85_right_inner_knuckle_joint",
              "robotiq_85_left_finger_tip_joint", "robotiq_85_right_finger_tip_joint"]),
}


@pytest.mark.parametrize("robot", ARMS)
def test_payload_torque_is_gradient_of_potential(robot):
    # Holding a mass up takes +dV/dq, V = m g z(q) of the point.
    urdf, link, locked = ARMS[robot]
    arm = URDFArmConfiguration.from_urdf(str(URDFS / urdf), locked_joint_names=locked)
    rng = np.random.default_rng(1)
    point, m, eps = np.array([0.0, 0.01, 0.05]), 0.2, 1e-6
    for _ in range(3):
        q = rng.uniform(-1.0, 1.0, arm.n_dof) + np.pi
        arm.update_from_angles(q, np.zeros(arm.n_dof))
        tau = payload_torques(arm, link, point, m)
        grad = np.zeros(arm.n_dof)
        for j in range(arm.n_dof):
            dq = np.zeros(arm.n_dof)
            dq[j] = eps
            zp = arm.fk_and_jacobian_at(q + dq, link, point)[0][2]
            zm = arm.fk_and_jacobian_at(q - dq, link, point)[0][2]
            grad[j] = m * GRAVITY_MPS2 * (zp - zm) / (2 * eps)
        np.testing.assert_allclose(tau, grad, atol=1e-5)


def test_zero_mass_is_zero():
    urdf, link, locked = ARMS["gen2"]
    arm = URDFArmConfiguration.from_urdf(str(URDFS / urdf), locked_joint_names=locked)
    arm.update_from_angles(np.full(arm.n_dof, np.pi), np.zeros(arm.n_dof))
    assert not payload_torques(arm, link, np.zeros(3), 0.0).any()


def make_self():
    urdf, link, locked = ARMS["gen2"]
    arm = URDFArmConfiguration.from_urdf(str(URDFS / urdf), locked_joint_names=locked)
    arm.update_from_angles(np.full(arm.n_dof, np.pi) + 0.3, np.zeros(arm.n_dof))
    fake = types.SimpleNamespace(
        _arm=arm, _max_payload_kg=1.0, _payload_last_t=None,
        _payload={"mass_now": 0.0, "mass_target": 0.0, "rate": 0.0, "link": "",
                  "point": [0.0, 0.0, 0.0]},
        _traj=types.SimpleNamespace(spring=types.SimpleNamespace(link_name=link)),
        get_logger=lambda: unittest.mock.Mock(),
    )
    fake._payload_link = lambda: VirtualSpringNode._payload_link(fake)
    return fake, link


def call(fake, mass, ramp=0.5, link="", point=(0.0, 0.0, 0.0)):
    req = types.SimpleNamespace(mass_kg=mass, ramp_sec=ramp, link_name=link, local_point=point)
    return VirtualSpringNode._set_payload_cb(fake, req, types.SimpleNamespace())


def test_set_payload_ramps_in_and_out():
    fake, link = make_self()
    full = payload_torques(fake._arm, link, np.zeros(3), 0.1)
    VirtualSpringNode._payload_torques(fake, 10.0)              # first tick sets the clock
    assert call(fake, 0.1, ramp=0.5).success
    half = VirtualSpringNode._payload_torques(fake, 10.25)      # halfway through the ramp
    np.testing.assert_allclose(half, 0.5 * full, atol=1e-9)
    done = VirtualSpringNode._payload_torques(fake, 10.6)       # past the end: exactly m
    np.testing.assert_allclose(done, full, atol=1e-12)
    assert fake._payload["mass_now"] == 0.1
    assert call(fake, 0.0, ramp=0.5).success                    # release
    VirtualSpringNode._payload_torques(fake, 11.2)
    assert fake._payload["mass_now"] == 0.0


@pytest.mark.parametrize("mass,link,expect", [
    (-0.1, "", "mass_kg must be"),
    (1.5, "", "mass_kg must be"),
    (0.1, "not_a_link", "Unknown link"),
])
def test_set_payload_rejects(mass, link, expect):
    fake, _ = make_self()
    res = call(fake, mass, link=link)
    assert not res.success and expect in res.message
    assert fake._payload["mass_target"] == 0.0
