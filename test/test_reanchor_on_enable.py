"""Hold springs re-anchor on every torque enable: VirtualSpringNode._torque_status_cb
called unbound on a stand-in `self` -- no rclpy init, no node, no ROS graph."""

import types
import unittest.mock

from springcontroller.virtual_spring_node import VirtualSpringNode


def make_self(reanchor=True):
    hold = types.SimpleNamespace(name="tip_spring", _hold_spring=True, _target_pending=False)
    fixed = types.SimpleNamespace(name="goal", _hold_spring=False, _target_pending=False)
    log = unittest.mock.Mock()
    fake = types.SimpleNamespace(
        _springs=[hold, fixed],
        _reanchor_hold_springs=reanchor,
        _last_torque_status=None,
        _torque_enabled_since=None,
        _ext_observer=None,
        _spring_ramp_duration=1.5,
        get_logger=lambda: log,
        get_clock=lambda: unittest.mock.Mock(),
    )
    fake._reanchor_hold_springs_for_enable = (
        lambda: VirtualSpringNode._reanchor_hold_springs_for_enable(fake))
    return fake, hold, fixed, log


def status(fake, data):
    VirtualSpringNode._torque_status_cb(fake, types.SimpleNamespace(data=data))


def test_hold_springs_reanchor_on_each_rising_edge_only():
    fake, hold, fixed, log = make_self()
    status(fake, "DISABLED")
    assert not hold._target_pending
    status(fake, "ENABLED")
    assert hold._target_pending and not fixed._target_pending
    assert "re-anchoring ['tip_spring']" in log.info.call_args[0][0]
    hold._target_pending = False                 # _joint_state_cb resolved it
    status(fake, "ENABLED")                      # still enabled: no new edge
    assert not hold._target_pending
    status(fake, "DISABLED")
    status(fake, "ENABLED")                      # next enable re-anchors again
    assert hold._target_pending and not fixed._target_pending


def test_reanchor_can_be_turned_off():
    fake, hold, fixed, log = make_self(reanchor=False)
    status(fake, "ENABLED")
    assert not hold._target_pending
    assert "re-anchoring" not in log.info.call_args[0][0]


def springs_self():
    """Stand-in for _set_springs_enabled: hold spring + trajectory glue."""
    fake, hold, fixed, log = make_self()
    fake._springs_on = True
    fake._springs_auto_disabled = False
    fake._springs_auto_disabled_since = None
    fake._springs_reenabled_since = None
    fake._publish_springs_enabled = lambda: None
    fake._traj = unittest.mock.Mock()
    return fake, hold, log


def test_springs_reenable_reanchors_hold_springs_and_trajectory_hold():
    fake, hold, log = springs_self()
    VirtualSpringNode._set_springs_enabled(fake, False, reason="teach")
    assert not hold._target_pending and not fake._traj.request_rehold.called
    VirtualSpringNode._set_springs_enabled(fake, True, reason="teach done")
    assert hold._target_pending                       # hold spring re-anchors
    fake._traj.request_rehold.assert_called_once()    # tracking hold too
    hold._target_pending = False
    VirtualSpringNode._set_springs_enabled(fake, True)  # already on: nothing to redo
    assert not hold._target_pending
    fake._traj.request_rehold.assert_called_once()


def test_glue_rehold_applies_only_when_holding():
    from springcontroller.trajectory_action import TrajectoryActionGlue
    glue = TrajectoryActionGlue.__new__(TrajectoryActionGlue)
    glue.follower = unittest.mock.Mock()
    glue._rehold = False
    glue.request_rehold()
    assert glue._rehold
    for state, expect in (("HOLDING", True), ("IDLE", False), ("RUNNING", False)):
        glue.follower.reset_mock()
        glue.follower.state.value = state
        glue.request_rehold()
        glue._apply_rehold()                          # what tick() calls first
        assert glue.follower.hold_current.called == expect, state
        assert not glue._rehold                       # consumed
        glue.follower.reset_mock()
        glue._apply_rehold()                          # nothing pending: no-op
        assert not glue.follower.hold_current.called
