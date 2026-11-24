"""Live trajectory-spring gain changes via parameters: the glue's callback and
its control-loop side, on a stand-in glue -- no rclpy init, no node."""

import threading
import types
import unittest.mock

from springcontroller.trajectory_action import TrajectoryActionGlue
from springcontroller.trajectory_spring import OscillationWatchdog, TrackingPoseSpring


def make_glue():
    g = TrajectoryActionGlue.__new__(TrajectoryActionGlue)
    g._lock = threading.Lock()
    g._pending_gains = {}
    g._softened = None
    g._wd = OscillationWatchdog()
    g.spring = TrackingPoseSpring("ee", stiffness=60.0, damping=5.0, rot_stiffness=6.0,
                                  rot_damping=0.1)
    g._node = types.SimpleNamespace(get_logger=lambda: unittest.mock.Mock())
    return g


def param(name, value):
    return types.SimpleNamespace(name=name, value=value)


def test_gain_change_applies_on_next_tick():
    g = make_glue()
    res = g._on_set_parameters([param("trajectory_spring.stiffness", 80.0),
                                param("trajectory_spring.damping", 6)])
    assert res.successful
    assert g.spring.stiffness == 60.0                  # not yet: control loop applies it
    g._apply_pending_gains()
    assert g.spring.stiffness == 80.0 and g.spring.damping == 6.0


def test_rejects_negative_or_non_numbers():
    g = make_glue()
    for bad in (-1.0, "stiff", True):
        res = g._on_set_parameters([param("trajectory_spring.stiffness", bad)])
        assert not res.successful and "trajectory_spring.stiffness" in res.reason
    g._apply_pending_gains()
    assert g.spring.stiffness == 60.0


def test_other_parameters_are_left_alone():
    g = make_glue()
    res = g._on_set_parameters([param("spring_names", ["tip"]),
                                param("trajectory_spring.pause_force_n", 9.0)])
    assert res.successful and g._pending_gains == {}


def test_live_change_clears_a_watchdog_softening():
    g = make_glue()
    g._softened = {"stiffness": 300.0, "damping": 30.0, "rot_stiffness": 6.0,
                   "rot_damping": 0.1}
    g.spring.stiffness, g.spring.damping = 75.0, 7.5          # softened state
    g._on_set_parameters([param("trajectory_spring.stiffness", 60.0),
                          param("trajectory_spring.damping", 5.0)])
    g._apply_pending_gains()
    assert g._softened is None
    assert (g.spring.stiffness, g.spring.damping) == (60.0, 5.0)
    assert g.spring.rot_stiffness == 6.0                       # restored, not left soft
