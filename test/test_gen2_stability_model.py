"""The Gen2 stability model reproduces what the arm did (plain Python, ~1 min):
with the measured ~60 ms loop delay, the gains used in every working 2026-11
test are stable and the code's old defaults are unstable -- and without the
delay the model would (wrongly) call both stable."""

import importlib.util
import sys
from pathlib import Path

import pytest

PATH = (Path(__file__).resolve().parents[1] / "springcontroller" / "tools"
        / "gen2_stability_model.py")
spec = importlib.util.spec_from_file_location("gen2_stability_model", PATH)
model = importlib.util.module_from_spec(spec)
sys.modules["gen2_stability_model"] = model
spec.loader.exec_module(model)

ONSET = model.POSES["onset"]


def test_working_gains_stable_with_measured_delay():
    swing, _ = model.simulate(ONSET, model.gains(60, 5), delay_ticks=6)
    assert swing < 1.0


def test_old_defaults_unstable_with_measured_delay():
    swing, _ = model.simulate(ONSET, model.gains(300, 30), delay_ticks=6)
    assert swing > 1.0


@pytest.mark.parametrize("k,b", [(60, 5), (300, 30)])
def test_without_delay_both_look_stable(k, b):
    # Documents the earlier modelling error: no delay, no instability.
    swing, _ = model.simulate(ONSET, model.gains(k, b), delay_ticks=0)
    assert swing < 1.0


# ── the oscillation watchdog, closed loop on the delayed model ───────────────

def watched(soften=0.25):
    """Run the glue's real _watch() each tick on a stand-in glue whose
    follower holds the model's spring. Returns (hook, glue)."""
    import types
    import unittest.mock

    from springcontroller.trajectory_action import TrajectoryActionGlue
    from springcontroller.trajectory_spring import (
        FollowerConfig, FollowerState, OscillationWatchdog, TrajectoryFollower)

    glue = TrajectoryActionGlue.__new__(TrajectoryActionGlue)
    glue._wd = OscillationWatchdog()
    glue._wd_enabled, glue._wd_soften, glue._softened = True, soften, None
    glue._wd_grace, glue._wd_speed, glue._softened_at = 2.0, 4.0, 0.0
    glue.watchdog_trips, glue.on_unstable = 0, unittest.mock.Mock()
    glue._node = types.SimpleNamespace(get_logger=lambda: unittest.mock.Mock())
    glue.trip_times = []

    def hook(t, spring, arm):
        if not hasattr(glue, "follower"):
            glue.spring, glue._arm = spring, arm
            glue.follower = TrajectoryFollower(spring, arm, FollowerConfig())
            glue.follower.state = FollowerState.HOLDING
        before = glue.watchdog_trips
        glue._watch(t)
        if glue.watchdog_trips != before:
            glue.trip_times.append(t)
    return hook, glue


def test_watchdog_catches_the_old_defaults_and_softened_hold_settles():
    hook, glue = watched()
    swing, _ = model.simulate(ONSET, model.gains(300, 30), delay_ticks=6, seconds=8.0,
                              on_tick=hook)
    assert glue.trip_times and glue.trip_times[0] < 3.0, glue.trip_times   # caught early
    assert glue._softened == {"stiffness": 300, "damping": 30, "rot_stiffness": 6.0,
                              "rot_damping": 0.1}
    assert glue.spring.stiffness == 75 and glue.spring.damping == 7.5
    assert swing < 1.0                       # softened hold settles (last 3 s)
    glue.on_unstable.assert_not_called()     # no need to escalate


def test_watchdog_quiet_on_working_gains():
    hook, glue = watched()
    swing, _ = model.simulate(ONSET, model.gains(60, 5), delay_ticks=6, seconds=8.0,
                              on_tick=hook)
    assert swing < 1.0 and glue.watchdog_trips == 0


def test_watchdog_escalates_when_softening_does_not_help():
    hook, glue = watched(soften=1.0)          # "softening" that changes nothing
    model.simulate(ONSET, model.gains(300, 30), delay_ticks=6, seconds=8.0, on_tick=hook)
    assert glue.watchdog_trips >= 2
    glue.on_unstable.assert_called()          # asked the node to disable torque
    assert "disabling torque" in glue.on_unstable.call_args[0][0]
