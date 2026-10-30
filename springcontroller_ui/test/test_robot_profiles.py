"""
The per-robot configs (config/study_control_panel*.yaml): each one starts
the node and publishes a profile the UI can wire itself from, the UR3e's
ros2_control torque switch sends the right controller lists, and robots
without a move interface refuse moves instead of publishing them. Same
approach as test_orchestration_interlocks.py: _call_sync is monkeypatched,
no ROS graph or hardware.
"""
import json
import os

import pytest
import rclpy
import yaml
from rclpy.parameter import Parameter
from std_srvs.srv import Trigger
from controller_manager_msgs.srv import SwitchController

from springcontroller_ui.orchestration_node import (
    StudyControlPanelNode, resolve_joint_indices,
)
from springcontroller_ui_interfaces.srv import MoveToJointAngles

PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
URDF_DIR = os.path.join(
    os.path.dirname(PKG_DIR), "springcontroller", "flat_urdf_files")
CONFIGS = {
    "gen3": "study_control_panel.yaml",
    "gen2": "study_control_panel_gen2.yaml",
    "ur3e": "study_control_panel_ur3e.yaml",
}


@pytest.fixture(scope="module", autouse=True)
def ros_context():
    rclpy.init()
    yield
    rclpy.shutdown()


def node_for(robot, tmp_path):
    with open(os.path.join(PKG_DIR, "config", CONFIGS[robot])) as f:
        params = yaml.safe_load(f)["study_control_panel_node"]["ros__parameters"]
    # Source-tree URDF (the configs point at the main checkout), no
    # /robot_description wait, and no writes outside tmp_path.
    params["urdf_path"] = os.path.join(URDF_DIR, os.path.basename(params["urdf_path"]))
    params["prefer_robot_description"] = False
    params["study_start_preset_path"] = str(tmp_path / "start.yaml")
    overrides = [Parameter(k, value=v) for k, v in params.items()]
    return StudyControlPanelNode(parameter_overrides=overrides)


@pytest.fixture(params=sorted(CONFIGS))
def any_robot_node(request, tmp_path):
    n = node_for(request.param, tmp_path)
    yield request.param, n
    n.destroy_node()


@pytest.fixture
def ur3e(tmp_path):
    n = node_for("ur3e", tmp_path)
    yield n
    n.destroy_node()


# ---------------------------------------------------------------------
# Every config starts, and its profile is self-consistent
# ---------------------------------------------------------------------

def test_profile_matches_config(any_robot_node):
    robot, node = any_robot_node
    p = node._robot_profile()
    json.dumps(p)  # must be publishable as-is
    assert p["robot"] == robot
    assert p["joint_names"] == node._joint_names
    assert node._arm.n_dof == len(p["joint_names"])
    assert p["features"]["torque"] is True
    assert p["torque_status_topic"]
    for preset in p["link_presets"].values():
        node._arm.validate_link_name(preset["link_name"])
    node._arm.validate_link_name(node._tip_link_name)


def test_ur3e_profile_points_at_its_own_torque_status(ur3e):
    p = ur3e._robot_profile()
    assert p["torque_status_topic"] == "/study_control_panel_node/torque_status"
    assert p["features"]["move"] is False
    assert p["features"]["clear_faults"] is False
    assert p["features"]["gripper"] is True
    assert p["features"]["gripper_requires_torque_off"] is False


def test_gripper_wiring_per_robot(any_robot_node):
    robot, node = any_robot_node
    p = node._robot_profile()
    expected = {
        # robot: (command service, activate service, needs torque off)
        "gen3": ("/gen3_torque_control/gripper_command", "", True),
        "gen2": ("/gen2_torque_control/gripper_command", "/gen2_torque_control/gripper_activate", False),
        "ur3e": ("/hande_gripper/gripper_command", "/hande_gripper/activate", False),
    }[robot]
    assert (p["gripper_command_service"], p["gripper_activate_service"],
            p["features"]["gripper_requires_torque_off"]) == expected


# ---------------------------------------------------------------------
# Joint names: exact, or the UR driver's tf_prefix'd form
# ---------------------------------------------------------------------

def test_resolve_joint_indices_exact_and_prefixed():
    msg = ["ur3e_shoulder_pan_joint", "ur3e_elbow_joint", "wrist_1_joint"]
    idx, missing = resolve_joint_indices(msg, ["wrist_1_joint", "shoulder_pan_joint", "elbow_joint"])
    assert (idx, missing) == ([2, 0, 1], [])


def test_resolve_joint_indices_reports_missing_and_ambiguous():
    msg = ["a_joint_1", "b_joint_1"]
    idx, missing = resolve_joint_indices(msg, ["joint_1", "joint_2"])
    assert missing == ["joint_1", "joint_2"]


# ---------------------------------------------------------------------
# UR3e torque on/off = controller switch
# ---------------------------------------------------------------------

def _capture_switch(node, ok=True, message=""):
    calls = []

    def fake(client, req, timeout_sec=None):
        calls.append(req)
        return SwitchController.Response(ok=ok, message=message)

    node._call_sync = fake
    return calls


def test_ur3e_torque_on_activates_effort_controller(ur3e):
    calls = _capture_switch(ur3e)
    ok, msg = ur3e._set_torque(True)
    assert ok, msg
    (req,) = calls
    assert list(req.activate_controllers) == ["forward_effort_controller"]
    assert list(req.deactivate_controllers) == ["scaled_joint_trajectory_controller"]
    assert req.strictness == SwitchController.Request.STRICT
    assert ur3e._torque_status_freshness.value == "ENABLED"


def test_ur3e_torque_off_restores_trajectory_controller(ur3e):
    calls = _capture_switch(ur3e)
    ok, _ = ur3e._set_torque(False)
    assert ok
    assert list(calls[0].activate_controllers) == ["scaled_joint_trajectory_controller"]
    assert list(calls[0].deactivate_controllers) == ["forward_effort_controller"]
    assert ur3e._torque_status_freshness.value == "DISABLED"


def test_ur3e_refused_switch_reports_and_keeps_status(ur3e):
    _capture_switch(ur3e, ok=False, message="interface claimed")
    ok, msg = ur3e._set_torque(True)
    assert not ok
    assert "interface claimed" in msg
    assert ur3e._torque_status_freshness.value is None


def test_ur3e_unreachable_controller_manager(ur3e):
    ur3e._call_sync = lambda client, req, timeout_sec=None: None
    ok, msg = ur3e._set_torque(True)
    assert not ok
    assert "/controller_manager/switch_controller unavailable" in msg


def test_ur3e_soft_estop_switches_torque_off_without_a_move_to_abort(ur3e):
    calls = _capture_switch(ur3e)
    resp = ur3e._soft_estop_cb(Trigger.Request(), Trigger.Response())
    assert resp.success, resp.message
    assert list(calls[0].deactivate_controllers) == ["forward_effort_controller"]


def test_ur3e_move_refused_without_publishing(ur3e):
    ur3e._torque_status_freshness.update("DISABLED", ur3e._now())
    ur3e._call_sync = lambda *a, **k: pytest.fail("no collision check or move expected")
    req = MoveToJointAngles.Request(joint_angles_rad=[0.0] * 6)
    resp = ur3e._move_to_joint_angles_cb(req, MoveToJointAngles.Response())
    assert not resp.success
    assert "no move-to-joint-angles interface" in resp.message
