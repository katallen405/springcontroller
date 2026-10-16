"""
Unit tests for ExternalTorqueObserver / amplification_torque, using the
real Gen3 model with pinocchio forward dynamics (aba) as the plant.
"""

from pathlib import Path

import numpy as np
import pinocchio as pin
import pytest

from springcontroller.external_torque_observer import (
    ExternalTorqueObserver,
    amplification_torque,
)
from springcontroller.urdf_arm_configuration import URDFArmConfiguration

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
DT = 0.01
START = np.array([0.0, 0.4, 0.0, 1.8, 0.0, 0.9, 0.0])


def simulate(tau_ext_fn, seconds, observer, tau_bias=None, start=START):
    """
    Plant: q̈ = aba(q, q̇, τ_motor + τ_ext), with τ_motor = g(q) (pure gravity
    comp). The observer sees τ_measured = τ_motor + tau_bias (sensor offset).
    Sub-steps the plant at 1 ms; the observer runs at DT.
    """
    arm = URDFArmConfiguration.from_urdf(URDF_PATH, locked_joint_names=LOCKED_JOINT_NAMES)
    model = arm._model
    data = model.createData()
    q = arm._angles_to_q(start)
    v = np.zeros(model.nv)
    bias = np.zeros(model.nv) if tau_bias is None else tau_bias
    t = 0.0
    history = []
    substeps = 10
    for _ in range(int(round(seconds / DT))):
        arm.update(q, v)
        M, C, g = arm.get_dynamics()
        est = observer.step(M, C, g, v, g + bias, DT)
        history.append((t, est.copy(), tau_ext_fn(t)))
        for _ in range(substeps):
            g_now = pin.computeGeneralizedGravity(model, data, q)
            a = pin.aba(model, data, q, v, g_now + tau_ext_fn(t))
            v = v + a * (DT / substeps)
            q = pin.integrate(model, q, v * (DT / substeps))
            t += DT / substeps
    return history


def test_dynamics_shapes_and_symmetric_mass_matrix():
    arm = URDFArmConfiguration.from_urdf(URDF_PATH, locked_joint_names=LOCKED_JOINT_NAMES)
    arm.update_from_angles(START, np.full(7, 0.1))
    M, C, g = arm.get_dynamics()
    assert M.shape == (7, 7) and C.shape == (7, 7) and g.shape == (7,)
    np.testing.assert_allclose(M, M.T)
    assert np.all(np.linalg.eigvalsh(M) > 0)
    np.testing.assert_allclose(g, arm.get_gravity_torques())


def test_get_dynamics_does_not_disturb_kinematics():
    arm = URDFArmConfiguration.from_urdf(URDF_PATH, locked_joint_names=LOCKED_JOINT_NAMES)
    arm.update_from_angles(START, np.full(7, 0.3))
    T_before = arm.get_link_transform("bracelet_link").copy()
    J_before = arm.get_jacobian("bracelet_link", np.zeros(3)).copy()
    arm.get_dynamics()
    np.testing.assert_array_equal(arm.get_link_transform("bracelet_link"), T_before)
    np.testing.assert_array_equal(arm.get_jacobian("bracelet_link", np.zeros(3)), J_before)


def test_zero_external_torque_gives_zero_estimate():
    obs = ExternalTorqueObserver(7, gain=20.0, bias_time_constant=0.0)
    hist = simulate(lambda t: np.zeros(7), 1.0, obs)
    assert np.max(np.abs(hist[-1][1])) < 1e-3


def test_step_external_torque_is_tracked():
    # Sized so the undamped plant stays under ~2 rad/s, i.e. hand-guiding
    # speeds; at >10 rad/s the 100 Hz discretization error becomes visible.
    step = np.array([0.0, 0.5, 0.0, -0.25, 0.0, 0.05, 0.0])
    obs = ExternalTorqueObserver(7, gain=20.0, bias_time_constant=0.0)
    hist = simulate(lambda t: step if t >= 0.1 else np.zeros(7), 0.5, obs)
    # 3/K_o = 0.15 s after the step -> within ~5% (first-order response).
    est = next(e for t, e, _ in hist if t >= 0.1 + 3 / 20.0 + DT)
    np.testing.assert_allclose(est, step, atol=0.03)
    np.testing.assert_allclose(hist[-1][1], step, atol=0.01)


def test_constant_sensor_bias_absorbed_at_rest_and_clamped():
    bias = np.array([0.0, 0.7, 0.0, 0.0, 0.0, 0.0, -3.0])
    obs = ExternalTorqueObserver(
        7, gain=20.0, bias_max=1.0, rest_velocity=0.02, rest_time=0.5, bias_time_constant=0.5,
    )
    # Plant sees no external torque; the sensor offset makes r = −bias.
    hist = simulate(lambda t: np.zeros(7), 6.0, obs, tau_bias=bias)
    np.testing.assert_allclose(obs.residual, -bias, atol=1e-3)
    np.testing.assert_allclose(obs.bias, [0, -0.7, 0, 0, 0, 0, 1.0], atol=1e-3)
    est = hist[-1][1]
    assert abs(est[1]) < 1e-3          # fully absorbed
    assert est[6] == pytest.approx(2.0, abs=1e-3)  # clamp leaves the excess


def test_bias_not_learned_while_moving():
    obs = ExternalTorqueObserver(7, rest_velocity=0.02, rest_time=0.0, bias_time_constant=0.1)
    M = np.eye(7)
    C = np.zeros((7, 7))
    g = np.zeros(7)
    qdot = np.full(7, 0.5)
    for _ in range(200):
        obs.step(M, C, g, qdot, np.ones(7), DT)
    np.testing.assert_array_equal(obs.bias, np.zeros(7))


def test_reset_keeps_bias_and_zeroes_residual():
    obs = ExternalTorqueObserver(7, rest_time=0.0, bias_time_constant=0.05, bias_max=5.0)
    M, C, g = np.eye(7), np.zeros((7, 7)), np.zeros(7)
    for _ in range(300):
        obs.step(M, C, g, np.zeros(7), -np.ones(7), DT)
    learned = obs.bias
    assert np.all(learned > 0.9)
    obs.reset()
    np.testing.assert_array_equal(obs.residual, np.zeros(7))
    np.testing.assert_array_equal(obs.bias, learned)


def test_invalid_gain_rejected():
    with pytest.raises(ValueError):
        ExternalTorqueObserver(7, gain=0.0)


def test_amplification_deadband_alpha_and_clip():
    tau = np.array([0.2, -0.2, 1.5, -1.5, 10.0, -10.0, 0.0])
    out = amplification_torque(tau, alpha=0.5, deadband=0.5, max_torque=2.0)
    np.testing.assert_allclose(out, [0.0, 0.0, 0.5, -0.5, 2.0, -2.0, 0.0])


def test_amplification_is_continuous_at_deadband_edge():
    below = amplification_torque(np.array([0.5 - 1e-9]), 1.0, 0.5, 10.0)
    above = amplification_torque(np.array([0.5 + 1e-9]), 1.0, 0.5, 10.0)
    assert abs(above[0] - below[0]) < 1e-8
