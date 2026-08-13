"""
test_press_to_pin.py

Unit tests for the press_to_pin node's detection state machine and force
estimate, driven against the small `simple_2dof_arm.urdf` (the same rig
used by test/2Dspring_viz.py) instead of a trivial identity-Jacobian stub.
Using a real (if tiny) URDF means the Jacobian math running here is exactly
what press_to_pin uses on a real arm, including its known blind spots.

No rclpy spinning is used: PressToPin is constructed once per test (fast --
the URDF has 2 links) and then driven directly by calling its private
callbacks/methods, the same way a real /joint_states message would.
"""

import time
from pathlib import Path

import numpy as np
import pytest
import rclpy
import rclpy.client
from sensor_msgs.msg import JointState

import springcontroller.press_to_pin as ptp
from springcontroller.press_to_pin import PressToPin
from springcontroller_interfaces.srv import AddSpring

URDF_PATH = str(
    Path(__file__).resolve().parents[1]
    / "springcontroller" / "flat_urdf_files" / "simple_2dof_arm.urdf"
)

# A longer, 4-joint rig used only by the localization tests below, which
# need candidates 2+ joints apart (e.g. link_2 vs link_4) to exercise the
# non-adjacent-ambiguity path -- the 2-DOF rig above only ever has adjacent
# candidates.
URDF4_PATH = str(
    Path(__file__).resolve().parents[1]
    / "springcontroller" / "flat_urdf_files" / "simple_4dof_arm.urdf"
)

# Fast-but-real timing: hold/rearm still require actual elapsed wall-clock
# time (the state machine reads self.get_clock().now()), so keep them small
# rather than mocking the clock.
HOLD_TIME = 0.05
REARM_TIME = 0.05


# ── Analytic FK oracle, ported from test/2Dspring_viz.py::fk_2dof ─────────
# (not imported directly -- that module has matplotlib/rclpy.spin side
# effects at import/call time that don't belong in a unit test).

def fk_2dof(q, link_lengths):
    L1, L2 = link_lengths
    q1, q2 = q
    p0 = np.array([0.0, 0.0, 0.0])
    p1 = p0 + L1 * np.array([np.sin(q1), 0.0, -np.cos(q1)])
    p2 = p1 + L2 * np.array([np.sin(q1 + q2), 0.0, -np.cos(q1 + q2)])
    return p0, p1, p2


# ── Fixture ─────────────────────────────────────────────────────────────

@pytest.fixture
def node(monkeypatch):
    # Skip the 3s wait for /robot_description; fall back to urdf_path param.
    monkeypatch.setattr(ptp, "fetch_robot_description",
                         lambda node, timeout_sec=3.0: None)
    # Skip the 5s wait for a real add_spring service -- we replace the
    # client's call_async below instead of running a real service.
    monkeypatch.setattr(rclpy.client.Client, "wait_for_service",
                         lambda self, timeout_sec=None: True)

    rclpy.init(args=[
        "--ros-args",
        "-p", f"urdf_path:={URDF_PATH}",
        "-p", f"press.hold_time:={HOLD_TIME}",
        "-p", f"press.rearm_time:={REARM_TIME}",
    ])
    n = PressToPin()

    # Stand in for a real add_spring service that responds immediately, so
    # _pending_request clears synchronously and the state machine can keep
    # running on the next JointState callback (mirrors _add_spring_done
    # firing once the real future resolves).
    n.sent_requests = []

    def fake_call_async(req):
        n.sent_requests.append(req)
        future = rclpy.task.Future()
        future.set_result(
            AddSpring.Response(success=True, message="ok",
                                id=len(n.sent_requests))
        )
        return future

    n._add_spring_client.call_async = fake_call_async

    yield n

    n.destroy_node()
    rclpy.shutdown()


@pytest.fixture
def node4(monkeypatch):
    """Same setup as `node`, against the 4-joint rig (see URDF4_PATH)."""
    monkeypatch.setattr(ptp, "fetch_robot_description",
                         lambda node, timeout_sec=3.0: None)
    monkeypatch.setattr(rclpy.client.Client, "wait_for_service",
                         lambda self, timeout_sec=None: True)

    rclpy.init(args=[
        "--ros-args",
        "-p", f"urdf_path:={URDF4_PATH}",
        "-p", f"press.hold_time:={HOLD_TIME}",
        "-p", f"press.rearm_time:={REARM_TIME}",
    ])
    n = PressToPin()

    n.sent_requests = []

    def fake_call_async(req):
        n.sent_requests.append(req)
        future = rclpy.task.Future()
        future.set_result(
            AddSpring.Response(success=True, message="ok",
                                id=len(n.sent_requests))
        )
        return future

    n._add_spring_client.call_async = fake_call_async

    yield n

    n.destroy_node()
    rclpy.shutdown()


def send(n, effort, vel=(0.0, 0.0), pos=(0.0, 0.0)):
    msg = JointState()
    msg.name = ["joint_1", "joint_2"]
    msg.position = list(pos)
    msg.velocity = list(vel)
    msg.effort = list(effort)
    n._joint_state_cb(msg)


def establish_quiet_baseline(n, pos=(0.0, 0.0)):
    """
    Seed the commanded-torque cache (required before _joint_state_cb will
    do anything at all -- see the "waiting for a commanded-torque sample"
    guard) and prime the baseline via the first JointState message.
    """
    cmd = JointState()
    cmd.name = ["joint_1", "joint_2"]
    cmd.effort = [0.0, 0.0]
    n._commanded_cb(cmd)
    send(n, effort=[0.0, 0.0], pos=pos)


JOINT4_NAMES = ["joint_1", "joint_2", "joint_3", "joint_4"]


def send4(n, effort, vel=(0.0, 0.0, 0.0, 0.0), pos=(0.0, 0.0, 0.0, 0.0)):
    msg = JointState()
    msg.name = JOINT4_NAMES
    msg.position = list(pos)
    msg.velocity = list(vel)
    msg.effort = list(effort)
    n._joint_state_cb(msg)


def establish_quiet_baseline4(n, pos=(0.0, 0.0, 0.0, 0.0)):
    cmd = JointState()
    cmd.name = JOINT4_NAMES
    cmd.effort = [0.0, 0.0, 0.0, 0.0]
    n._commanded_cb(cmd)
    send4(n, effort=[0.0, 0.0, 0.0, 0.0], pos=pos)


# ── Tests ───────────────────────────────────────────────────────────────

def test_link_transform_matches_analytic_fk():
    """Sanity-check the URDF+pinocchio wiring against an independent FK."""
    from springcontroller.urdf_arm_configuration import URDFArmConfiguration

    arm = URDFArmConfiguration.from_urdf(URDF_PATH)
    q = np.array([0.4, -0.6])
    arm.update_from_angles(q)

    _, p1_expected, _ = fk_2dof(q, link_lengths=(0.5, 0.5))
    link2_origin = arm.get_link_transform("link_2")[:3, 3]

    np.testing.assert_allclose(link2_origin, p1_expected, atol=1e-9)


def test_press_latches_after_hold_time_and_sends_add_spring_request(node):
    establish_quiet_baseline(node)
    send(node, effort=[0.0, 2.5])          # joint_2 residual over threshold
    assert node.sent_requests == []        # not held long enough yet

    time.sleep(HOLD_TIME * 1.6)
    send(node, effort=[0.0, 2.5])          # same candidate, now past hold_time

    assert len(node.sent_requests) == 1
    req = node.sent_requests[0]
    assert req.link_name == "link_2"
    assert req.name == "pin_link_2_1"
    assert req.rest_length == 0.0
    np.testing.assert_allclose(req.local_point, [0.0, 0.0, 0.0])
    np.testing.assert_allclose(req.target, [0.0, 0.0, -0.5], atol=1e-9)
    assert 10.0 <= req.stiffness <= 300.0
    assert node._state == ptp.REFRACTORY


def test_localizes_most_distal_joint_when_both_over_threshold(node):
    establish_quiet_baseline(node)
    send(node, effort=[3.0, 3.0])          # both joints over threshold
    time.sleep(HOLD_TIME * 1.6)
    send(node, effort=[3.0, 3.0])

    assert len(node.sent_requests) == 1
    # joint_2 is the more distal joint, so link_2 -- not link_1 -- gets pinned
    assert node.sent_requests[0].link_name == "link_2"


def test_refractory_blocks_second_latch_until_rearm_time_of_quiet(node):
    establish_quiet_baseline(node)
    send(node, effort=[0.0, 2.5])
    time.sleep(HOLD_TIME * 1.6)
    send(node, effort=[0.0, 2.5])
    assert len(node.sent_requests) == 1

    # Still pressing -- REFRACTORY must ignore this, not create a 2nd spring.
    send(node, effort=[0.0, 2.5])
    assert len(node.sent_requests) == 1

    # Release below release_fraction*threshold and hold quiet for rearm_time.
    send(node, effort=[0.0, 0.0])
    time.sleep(REARM_TIME * 1.6)
    send(node, effort=[0.0, 0.0])
    assert node._state == ptp.ARMED

    send(node, effort=[0.0, 2.5])
    time.sleep(HOLD_TIME * 1.6)
    send(node, effort=[0.0, 2.5])
    assert len(node.sent_requests) == 2
    assert node.sent_requests[1].name == "pin_link_2_2"


def test_stiffness_clips_to_configured_bounds(node):
    # link_1's default contact frame sits exactly on joint_1's own axis, so
    # its translational Jacobian there is structurally zero in every
    # configuration (a point ON the rotation axis doesn't move when that
    # joint rotates) -- the force estimate for *any* press localized to
    # joint_1 comes out to |F|=0, clipping stiffness to stiffness.min.
    # This is a real blind spot of the "child BODY frame" contact-link
    # default for the most proximal joint, not a test artifact.
    node._latch(joint_idx=0, link="link_1", point=np.zeros(3),
                residual=np.array([50.0, 0.0]))
    assert node.sent_requests[-1].stiffness == pytest.approx(10.0)  # k_min

    # link_2's own Jacobian column (joint_2) is *also* zero at its frame
    # origin (same reasoning), so a big residual has to come in via joint_1
    # -- the only column that's nonzero there -- to push the estimate above
    # stiffness.max.
    node._latch(joint_idx=1, link="link_2", point=np.zeros(3),
                residual=np.array([50.0, 0.0]))
    assert node.sent_requests[-1].stiffness == pytest.approx(300.0)  # k_max


def test_force_estimate_underestimates_off_axis_press_on_link2(node):
    """
    Characterizes the docstring's "force estimate is an approximation"
    caveat with real numbers: _latch always evaluates the Jacobian at the
    contact link's frame *origin* (local_point=0). For link_2 that origin
    coincides with joint_2's own axis, so it's blind to joint_2's own
    lever arm. A real press away from that axis is therefore substantially
    underestimated.

    Ground truth: a 10N push in world +X applied 0.4m out along link_2
    (near its far end) produces residual torques of
    tau = Jv(link_2, [0,0,-0.4]).T @ [10,0,0] = [1.0, -4.0]
    (computed independently via pinocchio's own get_jacobian, not via
    press_to_pin's approximation).
    """
    residual = np.array([1.0, -4.0])
    node._latch(joint_idx=1, link="link_2", point=np.zeros(3),
                residual=residual)

    req = node.sent_requests[-1]
    recovered_force = req.stiffness / node._k_per_newton  # before clipping
    assert recovered_force < 5.0, (
        "expected the origin-Jacobian approximation to substantially "
        "underestimate a 10N off-axis press (recovered "
        f"{recovered_force:.2f}N)"
    )


def test_localize_contact_fixes_distal_press_misattributed_to_proximal_link(
    node4,
):
    """
    Regression test for the real, measured bug (9 of 11 labeled test presses
    near distal joints/links were misattributed to more proximal ones): the
    old "most distal joint over threshold" heuristic breaks down whenever a
    proximal joint's amplified (longer lever arm) share of a distal press
    clears threshold while the distal joint's own, smaller-but-real share
    does not.

    Ground truth: at the rig's zero configuration, a 20N (+world X) / 10N
    (+world Z) push applied at link_4's own origin produces residual torques
    of tau = Jv(link_4, [0,0,0]).T @ [20,0,10] = [24, 16, 8, 0] (computed
    independently via URDFArmConfiguration.get_jacobian, not through
    press_to_pin). All of joint_1/2/3's shares clear the default 1.5
    threshold; joint_4's own share is exactly zero (a point force at a
    joint's own axis produces no torque there, the same blind spot the
    2-DOF tests above document for joint_1/joint_2). The old heuristic would
    have picked link_3 -- the most distal joint that individually clears
    threshold -- which is wrong; the true contact is link_4.
    """
    from springcontroller.urdf_arm_configuration import URDFArmConfiguration

    arm = URDFArmConfiguration.from_urdf(URDF4_PATH)
    arm.update_from_angles(np.zeros(4))
    Jv4 = arm.get_jacobian("link_4", np.zeros(3))[:3, :]
    residual = Jv4.T @ np.array([20.0, 0.0, 10.0])
    assert residual[3] == pytest.approx(0.0)  # joint_4's own blind spot
    assert all(abs(residual[i]) > node4._thresholds[i] for i in (0, 1, 2)), (
        "test setup check: expected joints 1-3 to individually clear "
        "threshold, exercising the old heuristic's failure mode"
    )

    establish_quiet_baseline4(node4)
    send4(node4, effort=residual.tolist())
    time.sleep(HOLD_TIME * 1.6)
    send4(node4, effort=residual.tolist())

    assert len(node4.sent_requests) == 1
    assert node4.sent_requests[0].link_name == "link_4"


def test_ambiguous_non_adjacent_candidates_suppress_latch(node4, monkeypatch):
    """
    When two different, non-adjacent links fit the observed residual about
    equally well, _localize_contact should suppress the press rather than
    guess -- this is the ambiguity.margin behavior described in the module
    docstring. Real arm geometry rarely produces an exact, easy-to-construct
    tie between far-apart links (nearer links in between tend to fit at
    least as well as either endpoint, becoming the runner-up instead -- see
    the design notes for this test), so the Jacobians here are hand-crafted
    via monkeypatch rather than pulled from the real 4-DOF rig: link_2's
    only active column (joint_1) and link_4's active columns (joints 1-3,
    identity-like) both exactly reconstruct a target with only its joint_1
    component nonzero, while link_1 (fully degenerate, as always) and
    link_3 (active columns aligned only with a direction the target doesn't
    need) fit it poorly.
    """
    residual = np.array([10.0, 0.0, 0.0, 0.0])

    fake_jacobians = {
        "link_1": np.zeros((6, 4)),
        # Single active column (joint_1) -- exactly reconstructs the target
        # via Fx=10.
        "link_2": np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]),
        # Active columns (joint_1, joint_2) point only along Z -- the
        # target has no Z component, so this is a poor fit regardless of F.
        "link_3": np.array([
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]),
        # Active columns (joint_1/2/3) exactly span x/y/z -- exactly
        # reconstructs the target via F=[10,0,0], tying with link_2.
        "link_4": np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]),
    }
    monkeypatch.setattr(
        node4._arm, "get_jacobian",
        lambda link_name, local_point: fake_jacobians[link_name]
    )

    establish_quiet_baseline4(node4)
    send4(node4, effort=residual.tolist())
    time.sleep(HOLD_TIME * 1.6)
    send4(node4, effort=residual.tolist())

    assert node4.sent_requests == []
    assert node4._state == ptp.ARMED
