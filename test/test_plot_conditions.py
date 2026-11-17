"""
test_plot_conditions.py

Unit tests for analysis/plot_conditions.py's significance brackets, on
synthetic tidy data (no Qualtrics export needed): every Holm-significant
pair gets a bracket, and stacked brackets never overlap.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "analysis"))
import plot_conditions as pc  # noqa: E402


N = 24


def _tidy(offsets, noise=0.15, seed=0, integer=False):
    """Long-format table like analyze.py's output: a per-participant base
    score plus a per-condition offset and a little noise, clipped to 1-5."""
    rng = np.random.default_rng(seed)
    base = rng.uniform(2.0, 3.0, N)
    rows = []
    for cond, off in offsets.items():
        vals = np.clip(base + off + rng.normal(0, noise, N), 1, 5)
        if integer:
            vals = np.round(vals)
        for pid, v in enumerate(vals):
            rows.append({"participant_id": f"p{pid:02d}", "condition": cond,
                         "fluency": v, "adapts": v, "helpfulness": v})
    return pd.DataFrame(rows)


# ── significant_pairs ─────────────────────────────────────────────────────────

def test_all_pairs_significant():
    df = _tidy({"KT": 0.0, "Position": 0.8, "Pose": 1.6})
    pairs = {(a, b) for a, b, _ in pc.significant_pairs(df, "helpfulness")}
    assert pairs == set(pc.PAIRS)


def test_helpfulness_like_pattern_includes_position_vs_pose():
    # KT ~ Position < Pose -- the pattern the paper's helpfulness table
    # reports, which the old KT-vs-Pose-only chart couldn't show.
    df = _tidy({"KT": 0.0, "Position": 0.05, "Pose": 1.0})
    pairs = {(a, b) for a, b, _ in pc.significant_pairs(df, "helpfulness")}
    assert ("Position", "Pose") in pairs
    assert ("KT", "Pose") in pairs
    assert ("KT", "Position") not in pairs


def test_no_brackets_when_omnibus_not_significant():
    df = _tidy({"KT": 0.0, "Position": 0.0, "Pose": 0.0}, noise=0.5)
    assert pc.significant_pairs(df, "helpfulness") == []


def test_stars_match_thresholds():
    assert pc.p_to_stars(0.0005) == "***"
    assert pc.p_to_stars(0.005) == "**"
    assert pc.p_to_stars(0.03) == "*"
    assert pc.p_to_stars(0.05) is None


# ── bracket_levels ────────────────────────────────────────────────────────────

def _overlaps(a, b):
    return a[0] <= b[1] and b[0] <= a[1]


@pytest.mark.parametrize("pairs", [
    [("KT", "Pose", "*")],
    [("Position", "Pose", "*"), ("KT", "Pose", "**")],
    [("KT", "Position", "*"), ("Position", "Pose", "*"), ("KT", "Pose", "***")],
    [("KT", "Position", "*"), ("KT", "Pose", "*")],
])
def test_overlapping_brackets_never_share_a_level(pairs):
    placed = pc.bracket_levels(pairs)
    assert len(placed) == len(pairs)
    for i, p in enumerate(placed):
        assert p[0] < p[1]
        for q in placed[i + 1:]:
            if _overlaps(p, q):
                assert p[3] != q[3]


def test_wide_bracket_stacks_above_narrow_one():
    placed = pc.bracket_levels([("KT", "Pose", "**"), ("Position", "Pose", "*")])
    levels = {(i1, i2): lvl for i1, i2, _, lvl in placed}
    assert levels[(1, 2)] == 0  # Position-Pose, innermost
    assert levels[(0, 2)] == 1  # KT-Pose, above it


def test_reversed_pair_order_is_normalized():
    (i1, i2, _, _), = pc.bracket_levels([("Pose", "KT", "*")])
    assert (i1, i2) == (0, 2)


# ── Rendering ─────────────────────────────────────────────────────────────────

def test_boxplot_draws_one_bracket_per_significant_pair(tmp_path, monkeypatch):
    calls = []
    real = pc.add_significance_bracket
    monkeypatch.setattr(pc, "add_significance_bracket",
                        lambda ax, x1, x2, y, stars, fs: (calls.append((x1, x2, y)),
                                                          real(ax, x1, x2, y, stars, fs)))
    df = _tidy({"KT": 0.0, "Position": 0.05, "Pose": 1.0})
    out = tmp_path / "helpfulness.png"
    pc.make_boxplot(df, "helpfulness", out)
    assert out.stat().st_size > 0
    spans = {(x1, x2) for x1, x2, _ in calls}
    assert spans == {(2, 3), (1, 3)}
    # Outer bracket drawn higher than the inner one.
    y = {(x1, x2): yy for x1, x2, yy in calls}
    assert y[(1, 3)] > y[(2, 3)]


def test_fluency_chart_renders_with_multiple_brackets(tmp_path):
    df = _tidy({"KT": 0.0, "Position": 1.0, "Pose": 2.0}, noise=0.3, integer=True)
    assert len(pc.significant_pairs(df, "fluency")) >= 2
    out = tmp_path / "fluency.png"
    pc.make_fluency_diverging(df, out)
    assert out.stat().st_size > 0
