"""
test_joint_kinematics.py

Unit tests for analysis/joint_kinematics.py's signal processing, using
synthetic signals with known derivatives -- no ROS, no bag files.
"""

import os
import sys
import warnings

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "analysis"))
import joint_kinematics as jk  # noqa: E402


DT = 0.01  # 100 Hz, the torque node's joint_states rate
ORDER = 3


def _window(window_s=0.15):
    return jk.sg_window_samples(window_s, DT, ORDER)


# ── Derivatives ───────────────────────────────────────────────────────────────

def test_sinusoid_derivatives_match_analytic():
    w = 2 * np.pi * 0.5  # 0.5 Hz, a plausible hand-guided motion
    t = np.arange(0, 6, DT)
    q = (0.4 * np.sin(w * t))[:, None]
    _, qd, qdd, qddd = jk.derivatives(q, DT, _window(), ORDER)

    # SG smoothing slightly attenuates each derivative (a fraction of a
    # percent at 0.5 Hz with a 150 ms window), so tolerances are 1-2% of
    # each signal's amplitude. SG edges are less accurate, hence interior.
    interior = slice(50, -50)
    ti = t[interior]
    np.testing.assert_allclose(qd[interior, 0], 0.4 * w * np.cos(w * ti), atol=0.01 * 0.4 * w)
    np.testing.assert_allclose(qdd[interior, 0], -0.4 * w**2 * np.sin(w * ti), atol=0.01 * 0.4 * w**2)
    np.testing.assert_allclose(qddd[interior, 0], -0.4 * w**3 * np.cos(w * ti), atol=0.02 * 0.4 * w**3)


def test_window_is_odd_and_longer_than_order():
    for window_s in (0.0, 0.02, 0.05, 0.15, 0.3):
        n = jk.sg_window_samples(window_s, DT, ORDER)
        assert n % 2 == 1
        assert n > ORDER


def test_too_few_samples_raises():
    with pytest.raises(ValueError):
        jk.derivatives(np.zeros((5, 1)), DT, 15, ORDER)


# ── Wrap / resampling ─────────────────────────────────────────────────────────

def test_unwrap_removes_wrap_spike():
    # Continuous joint rotating steadily through +pi, reported wrapped into
    # (-pi, pi] the way the torque node publishes it.
    t = np.arange(0, 2, DT)
    true_q = 3.0 + 0.5 * t
    wrapped = (np.angle(np.exp(1j * true_q)))[:, None]
    q = np.unwrap(wrapped, axis=0)
    _, _, qdd, _ = jk.derivatives(q, DT, _window(), ORDER)
    assert np.max(np.abs(qdd)) < 1e-6


def test_resample_uniform_handles_jitter_and_duplicates():
    rng = np.random.default_rng(0)
    t = np.arange(0, 1, DT) + rng.uniform(-0.002, 0.002, 100)
    t = np.concatenate([t, t[10:11]])  # duplicate stamp
    x = np.column_stack([2 * t, -t])
    grid, out, dt = jk.resample_uniform(t, x)
    assert np.allclose(np.diff(grid), dt)
    np.testing.assert_allclose(out[:, 0], 2 * grid, atol=1e-9)
    np.testing.assert_allclose(out[:, 1], -grid, atol=1e-9)


def test_resample_warns_on_gap():
    t = np.concatenate([np.arange(0, 1, DT), np.arange(2, 3, DT)])
    x = t[:, None]
    with pytest.warns(UserWarning, match="gap"):
        jk.resample_uniform(t, x)


# ── Path length / masks ───────────────────────────────────────────────────────

def test_path_length_per_column_and_norm():
    x = np.array([[0.0, 0.0], [3.0, 4.0], [3.0, 0.0]])
    np.testing.assert_allclose(jk.path_length(x, per_column=True), [3.0, 8.0])
    assert jk.path_length(x) == pytest.approx(5.0 + 4.0)


def test_path_length_mask_splits_moving_and_not():
    x = np.arange(6, dtype=float)[:, None]  # unit steps
    moving = np.array([True, True, True, False, False, False])
    assert jk.path_length(x, moving) == pytest.approx(2.0)
    assert jk.path_length(x, ~moving) == pytest.approx(2.0)
    # The one step straddling the boundary belongs to neither side.
    assert jk.path_length(x) == pytest.approx(5.0)


def test_moving_windows_and_mask():
    events = [(1.0, "MOVING"), (2.0, "IDLE"), (5.0, "MOVING")]
    windows = jk.moving_windows(events, last_t=6.0)
    assert windows == [(1.0, 2.0), (5.0, 6.0)]
    t = np.array([0.5, 1.5, 3.0, 5.5])
    assert list(jk.mask_in_windows(t, windows)) == [False, True, False, True]


# ── Periods / bag naming ──────────────────────────────────────────────────────

def test_to_epoch_uses_tz_for_naive_times():
    # 2026-09-04 is EDT (UTC-4).
    assert jk.to_epoch("2026-09-04T15:35:37", "America/New_York") == \
        jk.to_epoch("2026-09-04T19:35:37+00:00", "America/New_York")


def test_load_and_select_periods(tmp_path):
    f = tmp_path / "periods.csv"
    f.write_text(
        "participant_id,condition,side,start,end\n"
        "test1,pose,1,2026-09-04T15:35:37,2026-09-04T15:36:20\n"
        "test1,Pose,2,2026-09-04T15:40:00,2026-09-04T15:40:45\n"
        "test1,position,1,2026-09-04T15:50:00,2026-09-04T15:50:30\n"
        "test2,pose,1,2026-09-04T16:00:00,2026-09-04T16:00:30\n"
    )
    periods = jk.load_periods(str(f), "America/New_York")
    mine = jk.select_periods(periods, "test1", "pose")
    assert [p["side"] for p in mine] == ["1", "2"]
    assert mine[0]["end"] - mine[0]["start"] == pytest.approx(43.0)


def test_load_periods_corrected_format_skips_zero_length(tmp_path):
    f = tmp_path / "adjustment_times.csv"
    f.write_text(
        "Participant,Condition,Side,Adjustment Start,Adjustment End,Total Adjustment Time\n"
        "p1,Pose,1,2026-09-05T12:41:32,2026-09-05T12:42:48,76\n"
        "p10,Pose,2,2026-09-09T14:55:06,2026-09-09T14:55:06,0\n"
    )
    with pytest.warns(UserWarning, match="zero-length"):
        periods = jk.load_periods(str(f), "America/New_York")
    assert [(p["participant_id"], p["side"]) for p in periods] == [("p1", "1")]
    assert periods[0]["end"] - periods[0]["start"] == pytest.approx(76.0)


def test_select_periods_matches_p_prefixed_ids():
    periods = [
        {"participant_id": "p1", "condition": "Pose", "side": "1", "start": 0.0, "end": 1.0},
        {"participant_id": "p10", "condition": "Pose", "side": "1", "start": 0.0, "end": 1.0},
    ]
    # Bag folders are named 1..24; the times file uses p1..p24.
    assert len(jk.select_periods(periods, "1", "pose")) == 1
    assert len(jk.select_periods(periods, "10", "pose")) == 1
    assert jk.select_periods(periods, "2", "pose") == []


def test_load_periods_missing_column_raises(tmp_path):
    f = tmp_path / "bad.csv"
    f.write_text("who,condition,side,start,end\n")
    with pytest.raises(KeyError, match="participant_id"):
        jk.load_periods(str(f), "America/New_York")


def test_parse_bag_name():
    assert jk.parse_bag_name("/data/test4/pose_2026_09_04-15_35_37/") == ("test4", "pose")
    assert jk.parse_bag_name("/data/test4/KT_2026_09_04-15_35_37") == ("test4", "KT")


# ── End to end on a synthetic bag dict ────────────────────────────────────────

def test_analyze_period_on_synthetic_bag():
    t = np.arange(100.0, 110.0, DT)
    w = 2 * np.pi * 0.5
    q = np.zeros((t.size, 7))
    q[:, 0] = 0.2 * np.sin(w * (t - 100))
    qd = np.zeros_like(q)
    qd[:, 0] = 0.2 * w * np.cos(w * (t - 100))
    ee_p = np.column_stack([0.1 * (t - 100), np.zeros_like(t), np.zeros_like(t)])  # 0.1 m/s in x
    bag = {"t": t, "q": q, "qd": qd, "ee_t": t, "ee_p": ee_p, "move_status_events": []}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = jk.analyze_period(bag, 102.0, 106.0, windows=[], window_s=0.15, order=ORDER)

    assert r["duration_s"] == pytest.approx(4.0, abs=DT)
    assert r["frac_time_robot_moving"] == 0.0
    assert r["per_joint"]["peak_abs_acc"][0] == pytest.approx(0.2 * w**2, rel=0.05)
    assert np.all(r["per_joint"]["peak_abs_acc"][1:] == 0)
    assert r["per_joint"]["acc_vel_xcheck_r"][0] > 0.99
    assert r["ee_displacement_m"] == pytest.approx(0.4, abs=0.01)
    assert r["ee_path_length_m"] == pytest.approx(0.4, abs=0.01)
    # Two full periods of a 0.2 rad sinusoid -> 8 * 0.2 rad of travel.
    assert r["joint_path_length_rad"] == pytest.approx(1.6, rel=0.02)

    rows = jk.summary_rows("pose_x", {"participant_id": "test1", "condition": "pose", "side": "1"}, r)
    assert len(rows) == 7
    assert set(rows[0]) == set(jk.SUMMARY_FIELDS)
