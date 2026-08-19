"""
Tests the exact sequencing contract of StudyControlPanelNode's interlocked
services (enable_position_control, enable_torque_control, soft_estop)
without needing rosbridge, gen3_torque_control, or hardware -- the node's
`_set_torque` / `_abort_move` helper methods (the only funnel points these
services call through) are monkeypatched to canned (ok, message) results,
and the service callback methods are invoked directly (bypassing rclpy's
service dispatch entirely, since we're testing the Python sequencing logic,
not IPC).
"""
import rclpy
import pytest
from std_srvs.srv import Trigger

from springcontroller_ui.orchestration_node import StudyControlPanelNode
from springcontroller_ui_interfaces.srv import EnableTorqueControl


@pytest.fixture(scope="module", autouse=True)
def ros_context():
    rclpy.init()
    yield
    rclpy.shutdown()


@pytest.fixture
def node():
    n = StudyControlPanelNode()
    yield n
    n.destroy_node()


def trigger_request():
    return Trigger.Request()


def trigger_response():
    return Trigger.Response()


# ---------------------------------------------------------------------
# enable_position_control: "position control" = torque disabled (arm holds
# position via SINGLE_LEVEL_SERVOING once idle) -- no separate controller
# to activate, since kortex_bringup never runs alongside gen3_torque_control.
# ---------------------------------------------------------------------

def test_enable_position_control_reports_torque_disable_failure(node):
    node._set_torque = lambda data: (False, "actuator fault")

    resp = node._enable_position_control_cb(trigger_request(), trigger_response())
    assert resp.success is False
    assert "actuator fault" in resp.message
    assert "Torque disable failed" in resp.message


def test_enable_position_control_success(node):
    node._set_torque = lambda data: (True, "Torque control disabled.")

    resp = node._enable_position_control_cb(trigger_request(), trigger_response())
    assert resp.success is True
    assert resp.message == "Torque control disabled."


# ---------------------------------------------------------------------
# enable_torque_control: safety gate first, then torque-on -- aborting
# immediately if the safety status is stale or unsafe (unless overridden).
# ---------------------------------------------------------------------

def test_enable_torque_control_aborts_if_safety_status_stale(node):
    req = EnableTorqueControl.Request(allow_danger=False)
    resp = EnableTorqueControl.Response()
    node._set_torque = lambda data: pytest.fail("must not be called")

    out = node._enable_torque_control_cb(req, resp)
    assert out.success is False
    assert "safety_status" in out.message
    assert "virtual_spring_node" in out.message


def test_enable_torque_control_refuses_when_unsafe(node):
    node._safety_status_freshness.update("DANGER: too close", node._now())
    node._set_torque = lambda data: pytest.fail("must not be called")

    req = EnableTorqueControl.Request(allow_danger=False)
    resp = node._enable_torque_control_cb(req, EnableTorqueControl.Response())
    assert resp.success is False
    assert "DANGER" in resp.message


def test_enable_torque_control_allow_danger_bypasses_safety_gate(node):
    node._safety_status_freshness.update("DANGER: too close", node._now())
    node._set_torque = lambda data: (True, "Torque control enabled.")

    req = EnableTorqueControl.Request(allow_danger=True)
    resp = node._enable_torque_control_cb(req, EnableTorqueControl.Response())
    assert resp.success is True


def test_enable_torque_control_reports_torque_call_failure(node):
    node._safety_status_freshness.update("SAFE", node._now())
    node._set_torque = lambda data: (False, "unavailable")

    req = EnableTorqueControl.Request(allow_danger=False)
    resp = node._enable_torque_control_cb(req, EnableTorqueControl.Response())
    assert resp.success is False
    assert "unavailable" in resp.message
    assert "Torque enable failed" in resp.message


def test_enable_torque_control_success(node):
    node._safety_status_freshness.update("SAFE", node._now())
    node._set_torque = lambda data: (True, "Torque control enabled.")

    req = EnableTorqueControl.Request(allow_danger=False)
    resp = node._enable_torque_control_cb(req, EnableTorqueControl.Response())
    assert resp.success is True
    assert resp.message == "Torque control enabled."


# ---------------------------------------------------------------------
# soft_estop: abort first, then torque-disable -- abort must run first
# since an in-flight move holds gen3_torque_control's Kortex lock for its
# whole duration, so torque-disable would otherwise queue up behind it
# instead of taking effect immediately. Both attempted regardless of the
# other's outcome; overall success requires both.
# ---------------------------------------------------------------------

def test_soft_estop_both_succeed(node):
    node._abort_move = lambda: (True, "Stop() called.")
    node._set_torque = lambda data: (True, "ok")

    resp = node._soft_estop_cb(trigger_request(), trigger_response())
    assert resp.success is True
    assert "Abort move: ok" in resp.message
    assert "disabled" in resp.message


def test_soft_estop_calls_abort_before_torque_disable(node):
    order = []
    node._abort_move = lambda: (order.append("abort"), (True, "ok"))[1]
    node._set_torque = lambda data: (order.append("torque"), (True, "ok"))[1]

    node._soft_estop_cb(trigger_request(), trigger_response())
    assert order == ["abort", "torque"]


def test_soft_estop_abort_fails_still_attempts_torque_disable(node):
    calls = []
    node._abort_move = lambda: (False, "unreachable")
    node._set_torque = lambda data: (calls.append(1), (True, "ok"))[1]

    resp = node._soft_estop_cb(trigger_request(), trigger_response())
    assert len(calls) == 1  # torque-disable was still attempted
    assert resp.success is False  # overall success requires both
    assert "FAILED - unreachable" in resp.message
    assert "disabled" in resp.message


def test_soft_estop_torque_disable_fails_overall_failure(node):
    node._abort_move = lambda: (True, "ok")
    node._set_torque = lambda data: (False, "actuator fault")

    resp = node._soft_estop_cb(trigger_request(), trigger_response())
    assert resp.success is False
    assert "Abort move: ok" in resp.message
    assert "FAILED - actuator fault" in resp.message
