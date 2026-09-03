"""
test_pose_spring.py

Unit tests for PoseSpring (renamed from OrientationSpring, 2026-09-02 --
now a position-and-orientation hybrid, see virtual_spring.py). Uses a
minimal stub arm so no real URDF is required.
"""

import pytest
import numpy as np
from springcontroller.virtual_spring import PoseSpring, SpringCollection


# ── Minimal stub arm ──────────────────────────────────────────────────────────

class StubArm:
    """
    Trivial 3-DOF arm: identity transform, and a Jacobian whose rotational
    block (rows 3:6) is identity -- unlike test_virtual_spring.py's StubArm,
    which zeroes rows 3:6 (fine for VirtualSpring, useless for PoseSpring
    since it only ever reads Jw for the moment itself).

    extra_rotation_only_dof, when True (and n_dof > 3), adds a 4th DOF
    whose Jacobian column is nonzero in Jw but zero in Jv -- an idealized
    wrist joint whose axis passes exactly through the attachment point, so
    it can rotate the point without translating it. Used by the
    position-radius tests below to give the null-space-safe torque
    component (tau_safe -- see PoseSpring.compute_torques) something real
    to work with; with only the original 3 DOF, Jv is square and
    invertible (trivial null space), so tau_safe is always exactly zero.
    """

    def __init__(self, n_dof=3, translation_jacobian_nonzero=True, extra_rotation_only_dof=False):
        self._n = n_dof
        self._q = np.zeros(n_dof)
        self._qdot = np.zeros(n_dof)
        self._translation_jacobian_nonzero = translation_jacobian_nonzero
        self._extra_rotation_only_dof = extra_rotation_only_dof

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
        m = min(3, self._n)
        if self._translation_jacobian_nonzero:
            J[:3, :m] = np.eye(3)[:, :m]
        J[3:, :m] = np.eye(3)[:, :m]
        if self._extra_rotation_only_dof and self._n > 3:
            J[3:, 3] = [0.0, 0.0, 1.0]   # rotates about z at the point; no translation
        return J


# ── Tests ─────────────────────────────────────────────────────────────────────

def make_spring(**kwargs):
    # position_center at the origin, matching the stub's fixed attachment
    # point (identity transform, local_attachment_point defaults to
    # zeros) -- so dist_from_center is always 0 and risky_scale is always
    # 1.0 unless a test deliberately moves position_center away. This
    # keeps every test below that doesn't care about the position-radius
    # gate seeing exactly the pre-gate behavior (full Jw.T @ m_total).
    defaults = dict(
        link_name="link1",
        local_attachment_point=np.zeros(3),
        local_face_normal=np.array([0.0, 0.0, 1.0]),
        target_world_point=np.array([0.0, 0.0, 1.0]),  # face already points here
        stiffness=10.0,
        position_center=np.zeros(3),
        position_radius=1.0,
    )
    defaults.update(kwargs)
    return PoseSpring(**defaults)


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


def test_only_rotational_jacobian_feeds_moment():
    """A spring computed against a stub whose Jv block is nonzero must
    still produce the same torque as one where Jv is zero -- confirming
    the moment itself (m_total) is still built from Jw alone, unchanged
    by PoseSpring's new position-radius gate. This holds regardless of Jv
    here specifically because both springs' attachment point coincides
    with position_center (dist_from_center == 0, so risky_scale == 1.0
    always) -- tau_safe + 1*tau_risky recombines to exactly Jw.T @
    m_total by construction, independent of how Jv split it. See
    test_risky_component_ramps_down_beyond_radius for a case where Jv
    actually changes the result."""
    arm_with_jv = StubArm(translation_jacobian_nonzero=True)
    arm_without_jv = StubArm(translation_jacobian_nonzero=False)
    spring_a = make_spring(target_world_point=np.array([1.0, 0.0, 0.0]))
    spring_b = make_spring(target_world_point=np.array([1.0, 0.0, 0.0]))

    t_a = spring_a.compute_torques(arm_with_jv)
    t_b = spring_b.compute_torques(arm_without_jv)
    np.testing.assert_allclose(t_a, t_b, rtol=1e-6)


def test_full_strength_within_position_radius():
    """Within position_radius (here: attachment point exactly at
    position_center), torque must exactly equal the raw Jw.T @ m_total --
    no suppression at all. This is the "full kinematic freedom near the
    goal" this design exists to guarantee, in contrast to a rejected
    earlier design that projected the risky component out unconditionally
    everywhere."""
    arm = StubArm(n_dof=4, extra_rotation_only_dof=True)
    spring = make_spring(
        target_world_point=np.array([1.0, 0.3, -0.2]),
        position_center=np.zeros(3),
        position_radius=1.0,
    )
    torques = spring.compute_torques(arm)
    J = arm.get_jacobian("link1", np.zeros(3))
    Jw = J[3:, :]
    expected = Jw.T @ spring.last_state.moment_world
    np.testing.assert_allclose(torques, expected, rtol=1e-6)
    assert np.linalg.norm(torques) > 1e-6  # sanity: not trivially zero


def test_risky_component_ramps_down_beyond_radius():
    """Attachment point far outside position_radius: the resulting torque
    must equal tau_safe + risky_scale * tau_risky (per PoseSpring.
    compute_torques' own formula) -- the null-space-safe component stays
    at full strength, only the component that would move the point gets
    scaled down."""
    arm = StubArm(n_dof=4, extra_rotation_only_dof=True)
    # local_face_normal deliberately NOT +z (the make_spring() default):
    # the moment's axis (n_current x n_desired) is always perpendicular to
    # n_current, so a +z face normal can never produce a moment with a
    # nonzero z-component -- and the stub's redundant wrist DOF only
    # responds to the z-component (see StubArm's docstring). A +x face
    # normal lets the axis have a real z-component here, so tau_safe is
    # actually nonzero, not vacuously zero.
    spring = make_spring(
        local_face_normal=np.array([1.0, 0.0, 0.0]),
        target_world_point=np.array([1.0, 0.3, -0.2]),
        position_center=np.array([10.0, 0.0, 0.0]),  # attachment (origin) is far outside
        position_radius=0.1,
    )
    torques = spring.compute_torques(arm)

    J = arm.get_jacobian("link1", np.zeros(3))
    Jv = J[:3, :]
    Jw = J[3:, :]
    m_total = spring.last_state.moment_world
    tau_orient = Jw.T @ m_total
    null_space_projector = np.eye(4) - np.linalg.pinv(Jv) @ Jv
    tau_safe = null_space_projector @ tau_orient
    tau_risky = tau_orient - tau_safe

    dist_from_center = np.linalg.norm(np.zeros(3) - spring.position_center)
    risky_scale = min(1.0, spring.position_radius / dist_from_center)
    expected = tau_safe + risky_scale * tau_risky

    np.testing.assert_allclose(torques, expected, rtol=1e-6)
    assert risky_scale < 0.05  # sanity: actually being suppressed here
    assert np.linalg.norm(tau_safe) > 1e-6  # sanity: the safe part is real, not vacuous


def test_position_stiffness_defaults_to_zero():
    spring = make_spring()
    assert spring.position_stiffness == 0.0


def test_position_stiffness_pulls_back_toward_center_beyond_radius():
    """With stiffness=0 (isolates the orientation moment away -- m_spring
    is exactly zero regardless of target/angle), a nonzero
    position_stiffness must produce a real Hooke's-law pull back toward
    position_center once the attachment point is outside position_radius:
    zero orientation stiffness/damping means torques would otherwise be
    all zero, so any nonzero result here can only be the new restoring
    force."""
    arm = StubArm(n_dof=3)  # Jv == identity here
    spring = make_spring(
        stiffness=0.0,
        position_center=np.array([2.0, 0.0, 0.0]),
        position_radius=0.5,
        position_stiffness=4.0,
    )
    torques = spring.compute_torques(arm)

    dist_from_center = 2.0
    direction_to_center = np.array([1.0, 0.0, 0.0])
    f_position = 4.0 * (dist_from_center - 0.5) * direction_to_center
    np.testing.assert_allclose(torques, f_position, rtol=1e-6)
    assert np.linalg.norm(torques) > 1e-6  # sanity: not trivially zero
    # last_state.position_force is this same restoring force pre-Jv.T
    # projection -- what ~/spring_forces' position_force_magnitude/
    # position_force report (see _publish_spring_forces).
    np.testing.assert_allclose(spring.last_state.position_force, f_position, rtol=1e-6)


def test_position_stiffness_zero_within_radius_dead_zone():
    """Inside position_radius, position_stiffness must contribute nothing
    at all -- a true dead zone, the participant is free to move anywhere
    in here, matching the "dead zone" this parameter is meant to bound."""
    arm = StubArm(n_dof=3)
    spring = make_spring(
        stiffness=0.0,
        position_center=np.array([0.05, 0.0, 0.0]),
        position_radius=0.5,
        position_stiffness=4.0,
    )
    torques = spring.compute_torques(arm)
    np.testing.assert_allclose(torques, 0.0, atol=1e-10)
    np.testing.assert_allclose(spring.last_state.position_force, 0.0, atol=1e-10)


def test_position_force_zero_when_position_stiffness_defaults_to_zero():
    arm = StubArm(n_dof=3)
    spring = make_spring(
        stiffness=0.0,
        position_center=np.array([2.0, 0.0, 0.0]),
        position_radius=0.5,
    )
    spring.compute_torques(arm)
    np.testing.assert_allclose(spring.last_state.position_force, 0.0, atol=1e-10)


def test_invalid_position_stiffness_negative_raises():
    with pytest.raises(ValueError):
        make_spring(position_stiffness=-1.0)


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


def test_invalid_position_center_shape_raises():
    with pytest.raises(ValueError):
        make_spring(position_center=np.array([0.0, 0.0]))


def test_invalid_position_radius_zero_raises():
    with pytest.raises(ValueError):
        make_spring(position_radius=0.0)


def test_invalid_position_radius_negative_raises():
    with pytest.raises(ValueError):
        make_spring(position_radius=-1.0)
