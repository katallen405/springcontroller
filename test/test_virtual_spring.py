"""
test_virtual_spring.py

Unit tests for VirtualSpring and URDFArmConfiguration.
Uses a minimal stub arm so no real URDF is required.
"""

import pytest
import numpy as np
from springcontroller.virtual_spring import VirtualSpring, SpringCollection


# ── Minimal stub arm ──────────────────────────────────────────────────────────

class StubArm:
    """Trivial 2-DOF arm: identity transforms, identity Jacobian."""

    def __init__(self, n_dof=2):
        self._n = n_dof
        self._q = np.zeros(n_dof)
        self._qdot = np.zeros(n_dof)

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
        # Translational rows: identity block; rotational rows: zero
        J = np.zeros((6, self._n))
        J[:3, :min(3, self._n)] = np.eye(3)[:, :min(3, self._n)]
        return J


# ── Tests ─────────────────────────────────────────────────────────────────────

def make_spring(**kwargs):
    defaults = dict(
        link_name="link1",
        local_attachment_point=np.zeros(3),
        target_world_point=np.array([1.0, 0.0, 0.0]),
        stiffness=10.0,
    )
    defaults.update(kwargs)
    return VirtualSpring(**defaults)


def test_torques_shape():
    arm = StubArm(n_dof=2)
    spring = make_spring()
    torques = spring.compute_torques(arm)
    assert torques.shape == (2,)


def test_zero_displacement_gives_zero_force():
    arm = StubArm()
    spring = make_spring(
        local_attachment_point=np.array([1.0, 0.0, 0.0]),
        target_world_point=np.array([1.0, 0.0, 0.0]),
        stiffness=50.0,
    )
    torques = spring.compute_torques(arm)
    np.testing.assert_allclose(torques, 0.0, atol=1e-10)


def test_force_proportional_to_stiffness():
    arm = StubArm()
    s1 = make_spring(stiffness=10.0)
    s2 = make_spring(stiffness=20.0)
    t1 = s1.compute_torques(arm)
    t2 = s2.compute_torques(arm)
    np.testing.assert_allclose(t2, 2 * t1, rtol=1e-6)


def test_disabled_spring_gives_zero():
    arm = StubArm()
    spring = make_spring(enabled=False)
    torques = spring.compute_torques(arm)
    np.testing.assert_allclose(torques, 0.0, atol=1e-10)


def test_rest_length_no_force_at_natural_length():
    arm = StubArm()
    # Attachment at origin, target at (1,0,0) → distance = 1.0 = rest_length
    spring = make_spring(
        target_world_point=np.array([1.0, 0.0, 0.0]),
        stiffness=50.0,
        rest_length=1.0,
    )
    torques = spring.compute_torques(arm)
    np.testing.assert_allclose(torques, 0.0, atol=1e-10)


def test_deadband_continuous_at_outer_radius_regardless_of_rest_length():
    # Regression for a live incident (2026-09-02): with rest_length=0 and a
    # deadband (inner_radius=0.1016, outer_radius=0.127), extension sitting
    # right at outer_radius made the commanded force jump ~5x depending on
    # which side of the boundary floating-point noise landed on. The force
    # magnitude just inside vs. just outside outer_radius must now match to
    # within the size of the probe step, however rest_length is configured.
    inner, outer = 0.1016, 0.127
    for rest_length in (0.0, inner, outer, 5.0):
        arm = StubArm()
        spring = make_spring(
            target_world_point=np.array([outer, 0.0, 0.0]),
            stiffness=50.0,
            rest_length=rest_length,
            inner_radius=inner,
            outer_radius=outer,
        )

        eps = 1e-9
        spring.target_world_point = np.array([outer - eps, 0.0, 0.0])
        torques_inside = spring.compute_torques(arm)

        spring.target_world_point = np.array([outer + eps, 0.0, 0.0])
        torques_outside = spring.compute_torques(arm)

        np.testing.assert_allclose(
            torques_outside, torques_inside, atol=1e-4,
            err_msg=f"discontinuous at outer_radius with rest_length={rest_length}",
        )


def test_deadband_beyond_outer_radius_ignores_rest_length():
    # Beyond outer_radius, the plain-spring region must be anchored at
    # inner_radius, not rest_length -- rest_length is irrelevant whenever a
    # deadband is configured (outer_radius > inner_radius).
    inner, outer = 0.1016, 0.127
    arm = StubArm()
    extension = 0.2
    s1 = make_spring(
        target_world_point=np.array([extension, 0.0, 0.0]),
        stiffness=50.0,
        rest_length=0.0,
        inner_radius=inner,
        outer_radius=outer,
    )
    s2 = make_spring(
        target_world_point=np.array([extension, 0.0, 0.0]),
        stiffness=50.0,
        rest_length=5.0,
        inner_radius=inner,
        outer_radius=outer,
    )
    t1 = s1.compute_torques(arm)
    t2 = s2.compute_torques(arm)
    np.testing.assert_allclose(t1, t2, rtol=1e-9)
    expected = 50.0 * (extension - inner)
    np.testing.assert_allclose(t1[0], expected, rtol=1e-9)


def test_potential_energy():
    arm = StubArm()
    spring = make_spring(stiffness=10.0, rest_length=0.0)
    # distance = 1.0, U = 0.5 * 10 * 1^2 = 5.0
    energy = spring.get_potential_energy(arm)
    assert abs(energy - 5.0) < 1e-9


def test_move_target():
    arm = StubArm()
    spring = make_spring(target_world_point=np.array([1.0, 0.0, 0.0]))
    spring.move_target(np.array([2.0, 0.0, 0.0]))
    np.testing.assert_allclose(spring.target_world_point, [2.0, 0.0, 0.0])


def test_spring_collection_sums_torques():
    arm = StubArm()
    col = SpringCollection()
    col.add(make_spring(stiffness=10.0, name="a"))
    col.add(make_spring(stiffness=10.0, name="b"))
    total = col.compute_total_torques(arm, add_gravity_compensation=False)
    single = make_spring(stiffness=10.0).compute_torques(arm)
    np.testing.assert_allclose(total, 2 * single, rtol=1e-6)


def test_invalid_stiffness_raises():
    with pytest.raises(ValueError):
        make_spring(stiffness=-1.0)


def test_invalid_attachment_shape_raises():
    with pytest.raises(ValueError):
        make_spring(local_attachment_point=np.array([0.0, 0.0]))  # wrong shape
