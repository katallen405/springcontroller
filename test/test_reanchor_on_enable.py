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
