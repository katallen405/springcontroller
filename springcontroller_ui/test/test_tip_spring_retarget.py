"""
Tests StudyControlPanelNode._move_tip_spring_to_current (retarget-in-place
if the tip spring already exists, create it fresh otherwise) and its
wiring into set_current_as_study_start, without needing rosbridge,
virtual_spring_node, or hardware.

_current_tip_world_point (FK), _list_current_spring_names, _call_sync (the
AddSpring funnel point), and _tip_spring_target_pub.publish are all
monkeypatched -- this is testing the sequencing/branching logic, not FK or
IPC.
"""
import numpy as np
import rclpy
import pytest
from std_srvs.srv import Trigger

from springcontroller_ui.orchestration_node import StudyControlPanelNode


@pytest.fixture(scope="module", autouse=True)
def ros_context():
    rclpy.init()
    yield
    rclpy.shutdown()


@pytest.fixture
def node():
    n = StudyControlPanelNode()
    n._current_tip_world_point = lambda: np.array([1.0, 2.0, 3.0])
    yield n
    n.destroy_node()


class _AddResp:
    def __init__(self, success, message="ok"):
        self.success = success
        self.message = message


def test_fk_failure_reports_and_skips_everything(node):
    node._current_tip_world_point = lambda: (_ for _ in ()).throw(RuntimeError("no frame"))
    node._list_current_spring_names = lambda: pytest.fail("must not be called")

    msg = node._move_tip_spring_to_current()
    assert "FK" in msg
    assert "no frame" in msg


def test_spring_names_unavailable(node):
    node._list_current_spring_names = lambda: None

    msg = node._move_tip_spring_to_current()
    assert "unavailable" in msg


def test_retargets_in_place_when_spring_exists(node):
    node._list_current_spring_names = lambda: [node._reset_spring_name, "other_spring"]
    published = []
    node._tip_spring_target_pub.publish = lambda m: published.append(m)
    node._add_spring_client = None  # must not be touched
    node._call_sync = lambda client, req, timeout_sec=None: pytest.fail("must not call AddSpring")

    msg = node._move_tip_spring_to_current()
    assert "retargeted" in msg
    assert len(published) == 1
    assert (published[0].point.x, published[0].point.y, published[0].point.z) == (1.0, 2.0, 3.0)


def test_creates_spring_when_missing(node):
    node._list_current_spring_names = lambda: ["other_spring"]
    node._tip_spring_target_pub.publish = lambda m: pytest.fail("must not publish -- spring doesn't exist yet")
    node._call_sync = lambda client, req, timeout_sec=None: _AddResp(success=True)

    msg = node._move_tip_spring_to_current()
    assert "created" in msg


def test_add_spring_unavailable(node):
    node._list_current_spring_names = lambda: []
    node._call_sync = lambda client, req, timeout_sec=None: None

    msg = node._move_tip_spring_to_current()
    assert "unavailable/timed out" in msg


def test_add_spring_failure_message_surfaced(node):
    node._list_current_spring_names = lambda: []
    node._call_sync = lambda client, req, timeout_sec=None: _AddResp(success=False, message="boom")

    msg = node._move_tip_spring_to_current()
    assert "boom" in msg


def test_set_current_as_study_start_folds_in_spring_note(node, tmp_path):
    node._study_start_preset_path = str(tmp_path / "preset.yaml")
    node._joint_state_freshness.update(
        type("FakeJointState", (), {
            "name": node._joint_names,
            "position": [0.1] * len(node._joint_names),
        })(),
        node._now(),
    )
    node._move_tip_spring_to_current = lambda: "Tip spring 'x' retargeted to current position."

    resp = node._set_current_as_study_start_cb(Trigger.Request(), Trigger.Response())
    assert resp.success is True
    assert "Saved current pose as study start" in resp.message
    assert "retargeted" in resp.message
