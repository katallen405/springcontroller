"""Tests for analysis/reset_check.py's verdict logic on synthetic data."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "analysis"))
import reset_check as rc  # noqa: E402

DT = 0.01
POSE = np.array([1.0, 1.2, -1.2, 1.3, -2.7, -1.9, 1.0])
TZ = "America/New_York"


def _series(segments):
    """segments: [(duration_s, at_pose, commanded)] -> t, q, cmd."""
    t, q, cmd, t0 = [], [], [], 1000.0
    for dur, at_pose, c in segments:
        n = int(round(dur / DT))
        t += list(t0 + DT * np.arange(n))
        q += [POSE if at_pose else POSE + 0.5] * n
        cmd += [c] * n
        t0 += n * DT
    return np.array(t), np.array(q), np.array(cmd)


def test_reset_then_left_pose_before_logged_start_is_reset_ok():
    t, q, cmd = _series([(5, False, 0), (2, False, 1), (5, True, 0), (10, False, 0)])
    r = rc.check_period(t, q, cmd, start=1018.0, end=1021.0, poses=[POSE], tol_deg=1.15, tz=TZ)
    assert r["verdict"] == "reset OK"
    assert r["reset_reached_start_pose"] == "yes"


def test_reset_move_after_logged_start_flags_early_log():
    t, q, cmd = _series([(5, False, 0), (5, False, 0), (2, False, 1), (5, True, 0)])
    r = rc.check_period(t, q, cmd, start=1004.0, end=1016.0, poses=[POSE], tol_deg=1.15, tz=TZ)
    assert r["verdict"] == "logged start before reset"


def test_never_at_pose_is_no_reset_found():
    t, q, cmd = _series([(20, False, 0)])
    r = rc.check_period(t, q, cmd, start=1015.0, end=1019.0, poses=[POSE], tol_deg=1.15, tz=TZ)
    assert r["verdict"] == "no reset found"
    assert r["lookback_s"] == pytest.approx(15.0, abs=0.02)


def test_spring_condition_off_pose_at_torque_enable():
    t, q, cmd = _series([(2, False, 1), (3, True, 0), (10, False, 0)])
    events = [(1006.0, "springs", '["tip_spring"]'), (1007.0, "torque", "ENABLED")]
    r = rc.check_period(t, q, cmd, start=1012.0, end=1014.0, poses=[POSE], tol_deg=1.15,
                        tz=TZ, events=events, condition="Pose")
    assert r["verdict"] == "off start pose at torque enable"
    assert r["springs_at_torque_enable"] == '["tip_spring"]'
    # The same history is fine for KT, which doesn't use our torque control.
    r_kt = rc.check_period(t, q, cmd, start=1012.0, end=1014.0, poses=[POSE], tol_deg=1.15,
                           tz=TZ, events=events, condition="KT")
    assert r_kt["verdict"] == "reset OK"


def test_alt_pose_counts_as_start_pose():
    alt = POSE + 0.5
    t, q, cmd = _series([(5, False, 0)])
    r = rc.check_period(t, q, cmd, start=1003.0, end=1004.0, poses=[POSE, alt], tol_deg=1.15, tz=TZ)
    assert r["verdict"] == "at start pose"
