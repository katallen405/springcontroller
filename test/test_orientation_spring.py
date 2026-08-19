"""
test_orientation_spring.py

Unit tests for OrientationSpring.
Uses a minimal stub arm so no real URDF is required.
"""

import pytest
import numpy as np
from springcontroller.virtual_spring import OrientationSpring, SpringCollection


# ── Minimal stub arm ──────────────────────────────────────────────────────────

class StubArm:
    """
    Trivial 3-DOF arm: identity transform, and a Jacobian whose rotational
    block (rows 3:6) is identity -- unlike test_virtual_spring.py's StubArm,
    which zeroes rows 3:6 (fine for VirtualSpring, useless for
    OrientationSpring since it only ever reads Jw).
    """

    def __init__(self, n_dof=3, translation_jacobian_nonzero=True):
        self._n = n_dof
        self._q = np.zeros(n_dof)
        self._qdot = np.zeros(n_dof)
        self._translation_jacobian_nonzero = translation_jacobian_nonzero

    @property
    def joint_positions(self):
        return self._q.copy()

    @property
    def joint_velocities(self):
        return self._qdot.copy()

    @property
    def n_dof(self):
        return self._n

    def get_link_transform(self, link_name):
        return np.eye(4)   # attachment point == local_point for stub

    def get_jacobian(self, link_name, local_point):
        J = np.zeros((6, self._n))
        if self._translation_jacobian_nonzero:
            J[:3, :min(3, self._n)] = np.eye(3)[:, :min(3, self._n)]
        J[3:, :min(3, self._n)] = np.eye(3)[:, :min(3, self._n)]
        return J


# ── Tests ─────────────────────────────────────────────────────────────────────

def make_spring(**kwargs):
    defaults = dict(
        link_name="link1",
        local_attachment_point=np.zeros(3),
        local_face_normal=np.array([0.0, 0.0, 1.0]),
        target_world_point=np.array([0.0, 0.0, 1.0]),  # face already points here
        stiffness=10.0,
    )
    defaults.update(kwargs)
    return OrientationSpring(**defaults)


def test_torques_shape():
    arm = StubArm(n_dof=3)
    spring = make_spring()
    torques = spring.compute_torques(arm)
    assert torques.shape == (3,)


def test_zero_angle_error_gives_zero_torque():
    arm = StubArm()
    # Identity transform, local_face_normal +z, target directly along +z
    # from the (origin) attachment point -- already aligned.
    spring = make_spring(
        local_face_normal=np.array([0.0, 0.0, 1.0]),
        target_world_point=np.array([0.0, 0.0, 5.0]),
        stiffness=50.0,
    )
    torques = spring.compute_torques(arm)
    np.testing.assert_allclose(torques, 0.0, atol=1e-10)
    assert abs(spring.last_state.extension) < 1e-10


def test_torque_proportional_to_stiffness():
    arm = StubArm()
    # Face points +z, target is off-axis -> nonzero angle error.
    s1 = make_spring(
        target_world_point=np.array([1.0, 0.0, 0.0]),
        stiffness=10.0,
    )
    s2 = make_spring(
        target_world_point=np.array([1.0, 0.0, 0.0]),
        stiffness=20.0,
    )
    t1 = s1.compute_torques(arm)
    t2 = s2.compute_torques(arm)
    np.testing.assert_allclose(t2, 2 * t1, rtol=1e-6)
    assert np.linalg.norm(t1) > 1e-6  # sanity: actually nonzero


def test_disabled_spring_gives_zero():
    arm = StubArm()
    spring = make_spring(
        target_world_point=np.array([1.0, 0.0, 0.0]),
        enabled=False,
    )
    torques = spring.compute_torques(arm)
    np.testing.assert_allclose(torques, 0.0, atol=1e-10)


def test_degenerate_target_at_attachment_point_no_nan():
    arm = StubArm()
    # Attachment point is the origin (per stub's identity transform); put
    # the target there too so dist == 0.
    spring = make_spring(target_world_point=np.zeros(3))
    torques = spring.compute_torques(arm)
    assert not np.any(np.isnan(torques))
    np.testing.assert_allclose(torques, 0.0, atol=1e-10)


def test_move_target_changes_torque():
    arm = StubArm()
    spring = make_spring(target_world_point=np.array([0.0, 0.0, 1.0]))
    t_aligned = spring.compute_torques(arm)
    np.testing.assert_allclose(t_aligned, 0.0, atol=1e-10)

    spring.move_target(np.array([1.0, 0.0, 0.0]))
    t_misaligned = spring.compute_torques(arm)
    assert np.linalg.norm(t_misaligned) > 1e-6


def test_only_rotational_jacobian_feeds_torque():
    """A spring computed against a stub whose Jv block is nonzero must still
    produce the same torque as one where Jv is zero -- confirming Jv is
    never referenced, only Jw."""
    arm_with_jv = StubArm(translation_jacobian_nonzero=True)
    arm_without_jv = StubArm(translation_jacobian_nonzero=False)
    spring_a = make_spring(target_world_point=np.array([1.0, 0.0, 0.0]))
    spring_b = make_spring(target_world_point=np.array([1.0, 0.0, 0.0]))

    t_a = spring_a.compute_torques(arm_with_jv)
    t_b = spring_b.compute_torques(arm_without_jv)
    np.testing.assert_allclose(t_a, t_b, rtol=1e-6)


def test_spring_collection_sums_torques():
    arm = StubArm()
    col = SpringCollection()
    col.add(make_spring(target_world_point=np.array([1.0, 0.0, 0.0]), stiffness=10.0, name="a"))
    col.add(make_spring(target_world_point=np.array([1.0, 0.0, 0.0]), stiffness=10.0, name="b"))
    total = col.compute_total_torques(arm, add_gravity_compensation=False)
    single = make_spring(target_world_point=np.array([1.0, 0.0, 0.0]), stiffness=10.0).compute_torques(arm)
    np.testing.assert_allclose(total, 2 * single, rtol=1e-6)


def test_invalid_stiffness_raises():
    with pytest.raises(ValueError):
        make_spring(stiffness=-1.0)


def test_invalid_attachment_shape_raises():
    with pytest.raises(ValueError):
        make_spring(local_attachment_point=np.array([0.0, 0.0]))


def test_invalid_face_normal_shape_raises():
    with pytest.raises(ValueError):
        make_spring(local_face_normal=np.array([0.0, 0.0]))


def test_zero_face_normal_raises():
    with pytest.raises(ValueError):
        make_spring(local_face_normal=np.zeros(3))


def test_invalid_target_shape_raises():
    with pytest.raises(ValueError):
        make_spring(target_world_point=np.array([0.0, 0.0]))
