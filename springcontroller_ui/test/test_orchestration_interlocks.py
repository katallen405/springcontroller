"""
Tests the exact sequencing contract of StudyControlPanelNode's interlocked
services (enable_position_control, enable_torque_control, soft_estop)
without needing rosbridge, controller_manager, or hardware -- the node's
`_set_torque` / `_switch_controller` helper methods (the only two funnel
points these services call through) are monkeypatched to canned (ok,
message) results, and the service callback methods are invoked directly
(bypassing rclpy's service dispatch entirely, since we're testing the
Python sequencing logic, not IPC).
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
# enable_position_control: torque-off must succeed before controller
# activation is even attempted.
# ---------------------------------------------------------------------

def test_enable_position_control_aborts_if_torque_disable_unavailable(node):
    node._set_torque = lambda data: (False, "unavailable -- is gen3_torque_node running?")
    node._switch_controller = lambda a, d: pytest.fail("must not be called")

    resp = node._enable_position_control_cb(trigger_request(), trigger_response())
    assert resp.success is False
    assert "NOT activating position controller" in resp.message


def test_enable_position_control_aborts_if_torque_disable_fails(node):
    node._set_torque = lambda data: (False, "actuator fault")
    node._switch_controller = lambda a, d: pytest.fail("must not be called")

    resp = node._enable_position_control_cb(trigger_request(), trigger_response())
    assert resp.success is False
    assert "actuator fault" in resp.message
    assert "NOT activating position controller" in resp.message


def test_enable_position_control_reports_partial_state_if_switch_fails(node):
    node._set_torque = lambda data: (True, "Torque control disabled.")
    node._switch_controller = lambda a, d: (False, "controller_manager not available")

    resp = node._enable_position_control_cb(trigger_request(), trigger_response())
    assert resp.success is False
    assert "controller_manager not available" in resp.message
    assert "joint_trajectory_controller" in resp.message


def test_enable_position_control_success(node):
    node._set_torque = lambda data: (True, "Torque control disabled.")
    node._switch_controller = lambda a, d: (True, "ok")

    resp = node._enable_position_control_cb(trigger_request(), trigger_response())
    assert resp.success is True
    assert resp.message == "Position control enabled."


# ---------------------------------------------------------------------
# enable_torque_control: safety gate first, then controller-off, then
# torque-on -- in that order, aborting immediately on any failure.
# ---------------------------------------------------------------------

def _enable_torque_request(allow_danger=False):
    req = EnableTorqueControl.Request()
    req.allow_danger = allow_danger
    return EnableTorqueControl.Response(), req


def test_enable_torque_control_aborts_if_safety_status_stale(node):
    req = EnableTorqueControl.Request(allow_danger=False)
    resp = EnableTorqueControl.Response()
    node._switch_controller = lambda a, d: pytest.fail("must not be called")
    node._set_torque = lambda data: pytest.fail("must not be called")

    out = node._enable_torque_control_cb(req, resp)
    assert out.success is False
    assert "safety_status" in out.message
    assert "virtual_spring_node" in out.message


def test_enable_torque_control_refuses_when_unsafe(node):
    node._safety_status_freshness.update("DANGER: too close", node._now())
    node._switch_controller = lambda a, d: pytest.fail("must not be called")
    node._set_torque = lambda data: pytest.fail("must not be called")

    req = EnableTorqueControl.Request(allow_danger=False)
    resp = node._enable_torque_control_cb(req, EnableTorqueControl.Response())
    assert resp.success is False
    assert "DANGER" in resp.message


def test_enable_torque_control_allow_danger_bypasses_safety_gate(node):
    node._safety_status_freshness.update("DANGER: too close", node._now())
    node._switch_controller = lambda a, d: (True, "ok")
    node._set_torque = lambda data: (True, "Torque control enabled.")

    req = EnableTorqueControl.Request(allow_danger=True)
    resp = node._enable_torque_control_cb(req, EnableTorqueControl.Response())
    assert resp.success is True


def test_enable_torque_control_aborts_if_switch_fails_before_torque_call(node):
    node._safety_status_freshness.update("SAFE", node._now())
    node._switch_controller = lambda a, d: (False, "controller_manager not available")
    node._set_torque = lambda data: pytest.fail("must not be called -- switch must be tried first")

    req = EnableTorqueControl.Request(allow_danger=False)
    resp = node._enable_torque_control_cb(req, EnableTorqueControl.Response())
    assert resp.success is False
    assert "controller_manager not available" in resp.message


def test_enable_torque_control_reports_partial_state_if_torque_call_fails(node):
    node._safety_status_freshness.update("SAFE", node._now())
    node._switch_controller = lambda a, d: (True, "ok")
    node._set_torque = lambda data: (False, "unavailable")

    req = EnableTorqueControl.Request(allow_danger=False)
    resp = node._enable_torque_control_cb(req, EnableTorqueControl.Response())
    assert resp.success is False
    assert "Position controller deactivated" in resp.message
    assert "recover" in resp.message


def test_enable_torque_control_success(node):
    node._safety_status_freshness.update("SAFE", node._now())
    node._switch_controller = lambda a, d: (True, "ok")
    node._set_torque = lambda data: (True, "Torque control enabled.")

    req = EnableTorqueControl.Request(allow_danger=False)
    resp = node._enable_torque_control_cb(req, EnableTorqueControl.Response())
    assert resp.success is True
    assert resp.message == "Torque control enabled."


# ---------------------------------------------------------------------
# soft_estop: best-effort, both attempted regardless of the other's
# outcome; overall success tracks torque-disable only.
# ---------------------------------------------------------------------

def test_soft_estop_both_succeed(node):
    node._set_torque = lambda data: (True, "ok")
    node._switch_controller = lambda a, d: (True, "ok")

    resp = node._soft_estop_cb(trigger_request(), trigger_response())
    assert resp.success is True
    assert "disabled" in resp.message
    assert "deactivated" in resp.message


def test_soft_estop_torque_fails_still_attempts_controller(node):
    calls = []
    node._set_torque = lambda data: (False, "unreachable")
    node._switch_controller = lambda a, d: (calls.append(1), (True, "ok"))[1]

    resp = node._soft_estop_cb(trigger_request(), trigger_response())
    assert len(calls) == 1  # switch_controller was still attempted
    assert resp.success is False  # tracks torque-disable outcome
    assert "FAILED - unreachable" in resp.message
    assert "deactivated" in resp.message


def test_soft_estop_controller_fails_success_still_tracks_torque(node):
    node._set_torque = lambda data: (True, "ok")
    node._switch_controller = lambda a, d: (False, "controller_manager not available")

    resp = node._soft_estop_cb(trigger_request(), trigger_response())
    assert resp.success is True  # torque-disable is the safety-critical half
    assert "disabled" in resp.message
    assert "FAILED - controller_manager not available" in resp.message
