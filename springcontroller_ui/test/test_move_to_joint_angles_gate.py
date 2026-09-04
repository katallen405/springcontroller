"""
Tests StudyControlPanelNode._move_to_joint_angles's collision gate and
move-start confirmation, without needing rosbridge, gen3_torque_control,
virtual_spring_node, or hardware.

`_call_sync` (the funnel point for the check_collision_at RPC) and
`_move_command_pub.publish` (the funnel point for issuing the move) are
monkeypatched -- `publish` fakes are used to simulate gen3_torque_control's
move_status response by writing directly to `_move_status_freshness`, since
the real thing happens on a background thread this test doesn't run.

_move_to_joint_angles only waits for confirmation a move *started*
(any move_status update at all), not for it to finish -- a full move can
take many seconds to tens of seconds, far longer than rosbridge's own
service-call timeout tolerates. Actual completion is tracked by the UI's
live move-status display instead.
"""
import time

import rclpy
import pytest

from springcontroller_ui.orchestration_node import StudyControlPanelNode


@pytest.fixture(scope="module", autouse=True)
def ros_context():
    rclpy.init()
    yield
    rclpy.shutdown()


@pytest.fixture
def node():
    n = StudyControlPanelNode()
    n._move_confirm_timeout_sec = 1.0  # keep the timeout test fast
    yield n
    n.destroy_node()


class _CheckResp:
    def __init__(self, in_collision, in_danger, min_distance=0.1, link_a="a", link_b="b"):
        self.in_collision = in_collision
        self.in_danger = in_danger
        self.min_distance = min_distance
        self.link_a = link_a
        self.link_b = link_b


ANGLES = [0.0] * 7


def test_refuses_when_check_service_unavailable(node):
    node._call_sync = lambda client, req, timeout_sec=None: None
    node._move_command_pub.publish = lambda msg: pytest.fail("must not publish -- fail closed")

    ok, msg = node._move_to_joint_angles(ANGLES)
    assert ok is False
    assert "unavailable" in msg
    assert "fail closed" in msg


def test_refuses_on_in_collision(node):
    node._call_sync = lambda client, req, timeout_sec=None: _CheckResp(in_collision=True, in_danger=True)
    node._move_command_pub.publish = lambda msg: pytest.fail("must not publish -- would be in collision")

    ok, msg = node._move_to_joint_angles(ANGLES)
    assert ok is False
    assert "collision" in msg


def test_proceeds_with_note_on_in_danger_only(node):
    node._call_sync = lambda client, req, timeout_sec=None: _CheckResp(
        in_collision=False, in_danger=True, min_distance=0.02,
    )

    def fake_publish(msg):
        node._move_status_freshness.update("MOVING", node._now())
    node._move_command_pub.publish = fake_publish

    ok, msg = node._move_to_joint_angles(ANGLES)
    assert ok is True
    assert "danger zone" in msg


def test_success_when_move_confirmed_started(node):
    node._call_sync = lambda client, req, timeout_sec=None: _CheckResp(in_collision=False, in_danger=False)

    def fake_publish(msg):
        node._move_status_freshness.update("MOVING", node._now())
    node._move_command_pub.publish = fake_publish

    ok, msg = node._move_to_joint_angles(ANGLES)
    assert ok is True
    assert "started" in msg


def test_success_when_move_completes_before_confirm_check(node):
    """A very fast move (or one that already finished by the time this
    node polls) reports SUCCEEDED directly instead of MOVING -- also a
    success, just phrased as completed rather than started."""
    node._call_sync = lambda client, req, timeout_sec=None: _CheckResp(in_collision=False, in_danger=False)

    def fake_publish(msg):
        node._move_status_freshness.update("SUCCEEDED", node._now())
    node._move_command_pub.publish = fake_publish

    ok, msg = node._move_to_joint_angles(ANGLES)
    assert ok is True
    assert "completed" in msg


def test_reports_other_terminal_status_as_failure_to_start(node):
    node._call_sync = lambda client, req, timeout_sec=None: _CheckResp(in_collision=False, in_danger=False)

    def fake_publish(msg):
        node._move_status_freshness.update("ABORTED", node._now())
    node._move_command_pub.publish = fake_publish

    ok, msg = node._move_to_joint_angles(ANGLES)
    assert ok is False
    assert "ABORTED" in msg


def test_times_out_if_move_status_never_updates(node):
    node._call_sync = lambda client, req, timeout_sec=None: _CheckResp(in_collision=False, in_danger=False)
    node._move_command_pub.publish = lambda msg: None  # never updates move_status_freshness

    start = time.monotonic()
    ok, msg = node._move_to_joint_angles(ANGLES)
    elapsed = time.monotonic() - start

    assert ok is False
    assert "No" in msg and "move_status" in msg
    assert elapsed < 3.0  # bounded by node._move_confirm_timeout_sec (1.0s), not hanging
