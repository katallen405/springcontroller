"""
test_repulsion.py

Unit tests for URDFArmConfiguration.get_repulsion_torques() -- the
low-strength repulsion field around scene collision objects (see the
"Low-strength repulsion field around collision objects" plan, 2026-08-19).

Uses two fixtures from test_press_to_pin.py's pattern:
  - simple_2dof_arm.urdf: visual-only, no <collision> elements -- exercises
    the "no collision model loaded" early-return.
  - repulsion_test_arm.urdf (new, springcontroller/flat_urdf_files): a
    single Z-axis joint with a 0.5m link extending along +X, carrying
    collision geometry. Environment objects in these tests are placed off
    the link's +Y face so witness-point geometry is predictable: the link's
    box spans world y:[-0.05, 0.05] at q=0 regardless of joint angle only
    because q is left at 0 throughout, so the robot-side witness point's
    (x, z) stay fixed across every gap variant below (only the environment
    object's y-position changes), which is what makes the ratio/equality
    assertions below exact rather than approximate.
"""

from pathlib import Path

import numpy as np
import pinocchio as pin
import pytest

from springcontroller.urdf_arm_configuration import URDFArmConfiguration

URDF_NO_COLLISION_PATH = str(
    Path(__file__).resolve().parents[1]
    / "springcontroller" / "flat_urdf_files" / "simple_2dof_arm.urdf"
)
URDF_PATH = str(
    Path(__file__).resolve().parents[1]
    / "springcontroller" / "flat_urdf_files" / "repulsion_test_arm.urdf"
)

DANGER = 0.02
CAUTION = 0.08
MAX_FORCE = 5.0


def make_arm(danger_threshold=DANGER):
    arm = URDFArmConfiguration.from_urdf(URDF_PATH, danger_threshold=danger_threshold)
    arm.update_from_angles(np.zeros(2))
    return arm


def add_object_at_gap(arm, gap, object_id="obstacle", side=1):
    """
    Place a box just off link_1's +Y face (side=1, world y=0.05 at q=0) or
    its mirror-image -Y face (side=-1, world y=-0.05), separated from it by
    `gap` metres (can be negative for interpenetration). x/z are chosen to
    sit fully inside link_1's box (x:[0,0.5], z:[-0.05,0.05]) so the
    separating axis is purely Y.
    """
    half_y = 0.03
    center_y = side * (0.05 + gap + half_y)
    pose = pin.SE3(np.eye(3), np.array([0.25, center_y, 0.0]))
    arm.add_environment_object(
        object_id, "box", [0.1, 2 * half_y, 0.1], pose,
    )
    # add_environment_object() rebuilds GeometryData (pair topology
    # changed), so distanceResults is unpopulated/stale until the next
    # distance query -- get_repulsion_torques() deliberately doesn't run
    # its own (see its docstring), so tests must, same as the real node
    # does via get_collision_status() every cycle.
    arm.get_collision_status()


# ── Tests ────────────────────────────────────────────────────────────────

def test_no_collision_model_returns_zero_vector():
    arm = URDFArmConfiguration.from_urdf(URDF_NO_COLLISION_PATH)
    arm.update_from_angles(np.zeros(2))
    torques = arm.get_repulsion_torques(CAUTION, MAX_FORCE)
    assert torques.shape == (2,)
    np.testing.assert_array_equal(torques, np.zeros(2))


def test_no_environment_objects_returns_zero_vector():
    arm = make_arm()
    torques = arm.get_repulsion_torques(CAUTION, MAX_FORCE)
    assert torques.shape == (2,)
    np.testing.assert_array_equal(torques, np.zeros(2))


def test_zero_outside_caution_threshold():
    arm = make_arm()
    add_object_at_gap(arm, gap=0.10)  # 10cm, beyond CAUTION=0.08
    torques = arm.get_repulsion_torques(CAUTION, MAX_FORCE)
    np.testing.assert_array_equal(torques, np.zeros(2))


def test_direction_flips_with_obstacle_side():
    # Which absolute sign corresponds to "away from a +Y obstacle" depends
    # on pin.getFrameJacobian's own sign convention, which this test
    # shouldn't need to reimplement by hand -- instead, check the
    # convention-independent property that actually matters: an obstacle on
    # the +Y side and its mirror image on the -Y side must push in opposite
    # generalized directions, with equal magnitude (mirror symmetry).
    arm_pos = make_arm()
    add_object_at_gap(arm_pos, gap=0.05, side=1)
    torques_pos = arm_pos.get_repulsion_torques(CAUTION, MAX_FORCE)

    arm_neg = make_arm()
    add_object_at_gap(arm_neg, gap=0.05, side=-1)
    torques_neg = arm_neg.get_repulsion_torques(CAUTION, MAX_FORCE)

    assert np.all(np.isfinite(torques_pos))
    assert torques_pos[0] != 0.0
    np.testing.assert_allclose(torques_neg, -torques_pos, rtol=1e-6)


def test_ramp_follows_quadratic_falloff():
    # Same witness-point geometry at every gap (only the obstacle's distance
    # changes -- see module docstring), so |torque| scales with the force
    # law's t^2 alone; comparing a ratio sidesteps needing the exact
    # Jacobian lever arm.
    arm = make_arm()
    gap_a, gap_b = 0.05, 0.03  # both inside (DANGER, CAUTION)
    add_object_at_gap(arm, gap=gap_a)
    mag_a = np.linalg.norm(arm.get_repulsion_torques(CAUTION, MAX_FORCE))
    arm.remove_environment_object("obstacle")
    add_object_at_gap(arm, gap=gap_b)
    mag_b = np.linalg.norm(arm.get_repulsion_torques(CAUTION, MAX_FORCE))

    t_a = (CAUTION - gap_a) / (CAUTION - DANGER)
    t_b = (CAUTION - gap_b) / (CAUTION - DANGER)
    expected_ratio = (t_b / t_a) ** 2

    assert mag_a > 0.0
    assert mag_b > mag_a  # closer -> stronger
    np.testing.assert_allclose(mag_b / mag_a, expected_ratio, rtol=1e-6)


def test_saturates_at_max_force_below_danger_threshold():
    # Two gaps, both < DANGER and both non-penetrating (so witness points
    # come from plain GJK, not EPA) -- the force law is a flat max_force_n
    # plateau there, so both should produce exactly the same torque vector.
    arm_a = make_arm()
    add_object_at_gap(arm_a, gap=0.010)
    torques_a = arm_a.get_repulsion_torques(CAUTION, MAX_FORCE)

    arm_b = make_arm()
    add_object_at_gap(arm_b, gap=0.005)
    torques_b = arm_b.get_repulsion_torques(CAUTION, MAX_FORCE)

    np.testing.assert_allclose(torques_a, torques_b, rtol=1e-6)
    assert np.linalg.norm(torques_a) > 0.0


def test_interpenetrating_pairs_contribute_nothing():
    # Regression test: confirmed live 2026-08-19 that coal's witness points
    # are not a reliable push-out direction once a pair actually overlaps
    # (a gripper link embedded in a cylinder obstacle got pushed further
    # in, not out) -- get_repulsion_torques() now skips any pair with
    # min_distance < 0 entirely rather than trust that direction. See
    # get_repulsion_torques's docstring.
    arm = make_arm()
    add_object_at_gap(arm, gap=-0.01)  # 1cm overlap
    torques = arm.get_repulsion_torques(CAUTION, MAX_FORCE)
    assert torques.shape == (2,)
    np.testing.assert_array_equal(torques, np.zeros(2))


def test_total_torque_clipped_to_cap_preserving_direction():
    # Two simultaneous near-max-force contacts on opposite links -- see
    # test_multiple_objects_sum_independently for the summing behavior this
    # builds on. Uncapped, their sum should exceed a small cap; capped, the
    # norm should land exactly at the cap with direction unchanged.
    arm_uncapped = make_arm()
    add_object_at_gap(arm_uncapped, gap=0.01, object_id="a", side=1)
    add_object_at_gap(arm_uncapped, gap=0.01, object_id="b", side=1)
    torques_uncapped = arm_uncapped.get_repulsion_torques(CAUTION, MAX_FORCE)
    uncapped_norm = np.linalg.norm(torques_uncapped)
    assert uncapped_norm > 0.5  # comfortably above the cap used below

    cap = 0.1
    arm_capped = make_arm()
    add_object_at_gap(arm_capped, gap=0.01, object_id="a", side=1)
    add_object_at_gap(arm_capped, gap=0.01, object_id="b", side=1)
    torques_capped = arm_capped.get_repulsion_torques(CAUTION, MAX_FORCE, cap)

    assert np.linalg.norm(torques_capped) == pytest.approx(cap, rel=1e-6)
    np.testing.assert_allclose(
        torques_capped / np.linalg.norm(torques_capped),
        torques_uncapped / uncapped_norm,
        rtol=1e-6,
    )


def test_cap_above_actual_norm_has_no_effect():
    arm = make_arm()
    add_object_at_gap(arm, gap=0.05)
    torques_uncapped = arm.get_repulsion_torques(CAUTION, MAX_FORCE)
    torques_with_high_cap = arm.get_repulsion_torques(CAUTION, MAX_FORCE, 1000.0)
    np.testing.assert_allclose(torques_with_high_cap, torques_uncapped, rtol=1e-6)


def test_multiple_objects_sum_independently():
    gap_symmetric = 0.05

    arm_a = make_arm()
    add_object_at_gap(arm_a, gap=gap_symmetric, object_id="a")
    torques_a = arm_a.get_repulsion_torques(CAUTION, MAX_FORCE)

    arm_b = make_arm()
    add_object_at_gap(arm_b, gap=0.03, object_id="b")
    torques_b = arm_b.get_repulsion_torques(CAUTION, MAX_FORCE)

    arm_both = make_arm()
    add_object_at_gap(arm_both, gap=gap_symmetric, object_id="a")
    add_object_at_gap(arm_both, gap=0.03, object_id="b")
    torques_both = arm_both.get_repulsion_torques(CAUTION, MAX_FORCE)

    np.testing.assert_allclose(torques_both, torques_a + torques_b, rtol=1e-6)
