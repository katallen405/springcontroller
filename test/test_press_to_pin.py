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


def send(n, effort, vel=(0.0, 0.0), pos=(0.0, 0.0)):
    msg = JointState()
    msg.name = ["joint_1", "joint_2"]
    msg.position = list(pos)
    msg.velocity = list(vel)
    msg.effort = list(effort)
    n._joint_state_cb(msg)


def establish_quiet_baseline(n, pos=(0.0, 0.0)):
    """First JointState message only primes the baseline and returns."""
    send(n, effort=[0.0, 0.0], pos=pos)


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
    node._latch(joint_idx=0, residual=np.array([50.0, 0.0]))
    assert node.sent_requests[-1].stiffness == pytest.approx(10.0)  # k_min

    # link_2's own Jacobian column (joint_2) is *also* zero at its frame
    # origin (same reasoning), so a big residual has to come in via joint_1
    # -- the only column that's nonzero there -- to push the estimate above
    # stiffness.max.
    node._latch(joint_idx=1, residual=np.array([50.0, 0.0]))
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
    node._latch(joint_idx=1, residual=residual)

    req = node.sent_requests[-1]
    recovered_force = req.stiffness / node._k_per_newton  # before clipping
    assert recovered_force < 5.0, (
        "expected the origin-Jacobian approximation to substantially "
        "underestimate a 10N off-axis press (recovered "
        f"{recovered_force:.2f}N)"
    )
