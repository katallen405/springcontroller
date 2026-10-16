"""
Gating tests for VirtualSpringNode._compute_idle_amplification, called
unbound on a stand-in `self` -- no rclpy init, no node, no ROS graph.
"""

import os
import types
import unittest.mock
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pytest

from springcontroller.external_torque_observer import ExternalTorqueObserver
from springcontroller.urdf_arm_configuration import URDFArmConfiguration
from springcontroller.virtual_spring_node import VirtualSpringNode

URDF_PATH = str(
    Path(__file__).resolve().parents[1]
    / "springcontroller" / "flat_urdf_files" / "gen3_kinova_flat.urdf"
)
LOCKED_JOINT_NAMES = [
    "robotiq_85_left_knuckle_joint",
    "robotiq_85_right_knuckle_joint",
    "robotiq_85_left_inner_knuckle_joint",
    "robotiq_85_right_inner_knuckle_joint",
    "robotiq_85_left_finger_tip_joint",
    "robotiq_85_right_finger_tip_joint",
]
POSE = np.array([0.0, 0.4, 0.0, 1.8, 0.0, 0.9, 0.0])
compute = VirtualSpringNode._compute_idle_amplification


def make_self(**overrides):
    arm = URDFArmConfiguration.from_urdf(URDF_PATH, locked_joint_names=LOCKED_JOINT_NAMES)
    arm.update_from_angles(POSE)
    fake = types.SimpleNamespace(
        _arm=arm,
        _ext_observer=ExternalTorqueObserver(7, gain=20.0, bias_time_constant=0.0),
        _ext_effort_warned=False,
        _idle_amp_velocity_cutoff=1.5,
        _ext_torque_pub=unittest.mock.Mock(),
        _springs=[],
        _idle_amp_enabled=True,
        _last_torque_status="ENABLED",
        _spring_ramp_duration=0.0,
        _idle_amp_enabled_since=None,
        _idle_amp_alpha=0.5,
        _ext_deadband=np.full(7, 0.1),
        _ext_amp_max=np.full(7, 2.0),
        get_logger=lambda: unittest.mock.Mock(),
        get_clock=lambda: unittest.mock.Mock(),
    )
    for k, v in overrides.items():
        setattr(fake, k, v)
    return fake


def run(fake, push, steps=60, qdot=None, spring_scale=1.0):
    """Feed a static pose with τ_meas = g − push (a held push: momentum stays 0)."""
    qdot = np.zeros(7) if qdot is None else qdot
    g = fake._arm.get_gravity_torques()
    out = None
    for _ in range(steps):
        out = compute(fake, qdot, g - push, 0.01, spring_scale)
    return out


PUSH = np.array([0.0, 1.0, 0.0, -0.6, 0.0, 0.0, 0.0])


def test_active_amplifies_bias_corrected_push():
    fake = make_self()
    out = run(fake, PUSH)
    expected = 0.5 * (np.abs(PUSH) - 0.1).clip(min=0) * np.sign(PUSH)
    np.testing.assert_allclose(out, expected, atol=1e-3)
    assert fake._ext_torque_pub.publish.called


@pytest.mark.parametrize("override", [
    {"_idle_amp_enabled": False},
    {"_last_torque_status": "DISABLED"},
    {"_springs": [object()]},
])
def test_zero_when_gated_off_but_observer_still_runs(override):
    fake = make_self(**override)
    out = run(fake, PUSH)
    np.testing.assert_array_equal(out, np.zeros(7))
    np.testing.assert_allclose(fake._ext_observer.estimate, PUSH, atol=1e-3)


def test_zero_and_reset_above_velocity_cutoff():
    fake = make_self()
    run(fake, PUSH)
    out = compute(fake, np.full(7, 2.0), np.zeros(7), 0.01, 1.0)
    np.testing.assert_array_equal(out, np.zeros(7))
    np.testing.assert_array_equal(fake._ext_observer.residual, np.zeros(7))


def test_zero_and_reset_on_joint_state_gap():
    fake = make_self()
    run(fake, PUSH)
    g = fake._arm.get_gravity_torques()
    out = compute(fake, np.zeros(7), g - PUSH, 0.2, 1.0)
    np.testing.assert_array_equal(out, np.zeros(7))
    np.testing.assert_array_equal(fake._ext_observer.residual, np.zeros(7))


def test_missing_effort_is_inactive_and_warns_once():
    fake = make_self()
    logger = unittest.mock.Mock()
    fake.get_logger = lambda: logger
    for _ in range(3):
        out = compute(fake, np.zeros(7), None, 0.01, 1.0)
        np.testing.assert_array_equal(out, np.zeros(7))
    assert logger.warn.call_count == 1


def test_spring_scale_ramps_output():
    full = run(make_self(), PUSH, spring_scale=1.0)
    half = run(make_self(), PUSH, spring_scale=0.5)
    np.testing.assert_allclose(half, 0.5 * full, atol=1e-9)


def test_unavailable_observer_returns_zero():
    fake = make_self(_ext_observer=None)
    np.testing.assert_array_equal(
        compute(fake, np.zeros(7), np.zeros(7), 0.01, 1.0), np.zeros(7)
    )
