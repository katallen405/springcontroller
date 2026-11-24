"""Pure parts of tools/stability_step_test.py (plain Python; nothing here
starts rclpy or reaches the robot)."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

PATH = (Path(__file__).resolve().parents[1] / "springcontroller" / "tools"
        / "stability_step_test.py")
spec = importlib.util.spec_from_file_location("stability_step_test", PATH)
st = importlib.util.module_from_spec(spec)
sys.modules["stability_step_test"] = st
spec.loader.exec_module(st)


def test_no_ros_imported_at_module_level():
    assert "rclpy" not in vars(st)


def test_parse_ladder():
    assert st.parse_ladder("60:5,100:10") == [(60.0, 5.0), (100.0, 10.0)]
    for bad in ("0:5", "60:-1", "60"):
        with pytest.raises(ValueError):
            st.parse_ladder(bad)


def test_step_moves_out_and_back_within_speed_limit():
    moves = st.step_moves([0.1, 0.2, 0.4], 0.015, "xz")
    assert [m["label"] for m in moves] == ["+x", "+x back", "+z", "+z back"]
    assert np.allclose(moves[0]["target"], [0.115, 0.2, 0.4])
    assert np.allclose(moves[1]["target"], [0.1, 0.2, 0.4])
    assert np.allclose(moves[2]["target"], [0.1, 0.2, 0.415])
    assert all(st.peak_speed(m) <= 0.10 for m in moves)


def response(kind, t, step=0.015, t_end=0.3):
    """Gripper samples along x for a step from 0 to `step`."""
    x = np.where(t < t_end, step * t / t_end, step)
    s = np.clip(t - t_end, 0, None)
    if kind == "damped":
        x = x + np.where(t >= t_end, -0.004 * np.exp(-s / 0.1), 0)
    elif kind == "overshoot":
        x = x + np.where(t >= t_end, 0.006 * np.exp(-s / 0.4) * np.sin(2 * np.pi * 2 * s), 0)
    elif kind == "ringing":
        x = x + np.where(t >= t_end, 0.003 * np.sin(2 * np.pi * 3.5 * s), 0)
    return np.stack([x, np.zeros_like(t), np.zeros_like(t)], axis=1)


T = np.arange(0, 2.8, 0.01)


def test_damped_step_settles_quickly_without_overshoot():
    a = st.analyse_step(T, response("damped", T), [0, 0, 0], [0.015, 0, 0], 0.3)
    assert a["overshoot_pct"] == 0.0
    assert a["settle_s"] is not None and a["settle_s"] < 0.2
    assert a["ring_mm"] < 0.5


def test_overshoot_is_measured():
    a = st.analyse_step(T, response("overshoot", T), [0, 0, 0], [0.015, 0, 0], 0.3)
    assert 20 < a["overshoot_pct"] < 40          # ~5 mm of a 15 mm step
    assert a["settle_s"] is not None and a["settle_s"] > 0.3


def test_sustained_ringing_never_settles_and_its_frequency_is_found():
    a = st.analyse_step(T, response("ringing", T), [0, 0, 0], [0.015, 0, 0], 0.3)
    assert a["settle_s"] is None
    assert 1.8 < a["ring_mm"] < 2.4              # 3 mm amplitude -> 2.1 mm RMS
    assert abs(a["ring_hz"] - 3.5) <= 1.0        # 1 s window: 1 Hz bins


def test_step_back_is_measured_along_its_own_direction():
    fwd = response("overshoot", T)
    back = np.array([0.015, 0, 0]) - fwd          # mirror: stepping back to 0
    a = st.analyse_step(T, back, [0.015, 0, 0], [0, 0, 0], 0.3)
    assert 20 < a["overshoot_pct"] < 40


def test_cli_rejects_bad_arguments():
    with pytest.raises(SystemExit):
        st.main(["--step", "0.1"])
    with pytest.raises(SystemExit):
        st.main(["--axes", "xw"])
