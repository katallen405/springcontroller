#!/usr/bin/env python3
"""Plot total adjustment time by condition, from the table written by
adjustment_times.py -- the same per-condition-comparison shape as
compare_conditions.py's condition_comparison.png, adapted for a single
continuous measure instead of a stacked percentage breakdown: one
horizontal bar per condition (mean total adjustment time, each
participant's two sides summed first via anova_adjustment_time.py's
build_participant_condition_table), with the SEM as an error bar and each
participant's own value plotted as a jittered point so the spread behind
the mean stays visible.

Bar colors and condition order mirror CONDITION_COLORS/CONDITIONS in
../plot_conditions.py (Okabe-Ito,
one color per condition), which already compares this same study's
KT/Position/Pose conditions. Tick labels use DISPLAY_LABELS from
compare_conditions.py ("Adm."/"Pos.-only"/"Pose").

Stars the KT-vs-Pose comparison when it's significant, using the same
RM-ANOVA-gated, Holm-corrected post-hoc test as anova_adjustment_time.py
(see kt_pose_star()).

--exclude PID:CONDITION[:SIDE] (repeatable, e.g. p10:Pose) drops invalid
trials first; see anova_adjustment_time.py. The excluded participant's
other conditions are still plotted, but the star's test only uses
participants with all three conditions.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from anova_adjustment_time import add_exclude_argument, build_participant_condition_table
from compare_conditions import DISPLAY_LABELS, condition_sort_key
from stats_helpers import rm_anova_with_posthoc

CONDITION_COLORS = {
    "KT": "#0072B2",
    "Position": "#E69F00",
    "Pose": "#009E73",
}

STAR_PAIR = frozenset(("KT", "Pose"))


def p_to_stars(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return None


def kt_pose_star(per_condition):
    """Same RM-ANOVA-gated, Holm-corrected post-hoc procedure as
    anova_adjustment_time.py (see stats_helpers.rm_anova_with_posthoc),
    returning just the star string for the KT-vs-Pose comparison."""
    _, posthoc = rm_anova_with_posthoc(
        per_condition, dv="total_adjustment_time_sec", within="Condition", subject="Participant"
    )
    if posthoc is None:
        return None
    pair_rows = posthoc[posthoc.apply(lambda row: frozenset((row["A"], row["B"])) == STAR_PAIR, axis=1)]
    if pair_rows.empty:
        return None
    return p_to_stars(pair_rows.iloc[0]["p_corr"])


def plot_adjustment_time(per_condition, out_path, title=True, font_scale=1.0, seed=0):
    conditions = sorted(per_condition["Condition"].unique(), key=condition_sort_key)
    base_fontsize = plt.rcParams["font.size"] * font_scale
    rng = np.random.default_rng(seed)

    fig, ax = plt.subplots(figsize=(9, 0.9 * len(conditions) + 1.5))
    means = {}
    for y, cond in enumerate(conditions):
        vals = per_condition.loc[per_condition["Condition"] == cond, "total_adjustment_time_sec"].to_numpy()
        mean = vals.mean()
        sem = vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
        means[cond] = mean
        color = CONDITION_COLORS.get(cond, "tab:gray")
        ax.barh(y, mean, xerr=sem, height=0.6, color=color, ecolor="black", capsize=4)
        jitter = rng.uniform(-0.18, 0.18, size=len(vals))
        ax.scatter(vals, [y] * len(vals) + jitter, color="black", s=14, alpha=0.5, zorder=3)

    ax.set_yticks(range(len(conditions)))
    ax.set_yticklabels([DISPLAY_LABELS.get(c, c) for c in conditions], fontsize=base_fontsize)
    ax.invert_yaxis()
    ax.set_xlabel("Total adjustment time (s)", fontsize=base_fontsize)
    ax.tick_params(axis="x", labelsize=base_fontsize)
    if title:
        ax.set_title("Adjustment time by condition (mean ± SEM, points = participants)", fontsize=base_fontsize)

    stars = kt_pose_star(per_condition)
    if stars and "KT" in conditions and "Pose" in conditions:
        kt_y, pose_y = conditions.index("KT"), conditions.index("Pose")
        fig.canvas.draw()  # populate autoscaled xlim/scatter extents before placing the bracket
        xmax = ax.get_xlim()[1]
        bracket_x = xmax * 1.05
        tick = xmax * 0.02
        ax.plot([bracket_x, bracket_x], [kt_y, pose_y], color="black", linewidth=1, clip_on=False)
        ax.plot([bracket_x - tick, bracket_x], [kt_y, kt_y], color="black", linewidth=1, clip_on=False)
        ax.plot([bracket_x - tick, bracket_x], [pose_y, pose_y], color="black", linewidth=1, clip_on=False)
        ax.text(bracket_x + tick, (kt_y + pose_y) / 2, stars, ha="left", va="center", fontsize=base_fontsize)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("adjustment_times_csv", help="output of adjustment_times.py")
    ap.add_argument("--out", default=None, help="output PNG (default: <adjustment_times_csv's directory>/adjustment_time_comparison.png)")
    ap.add_argument("--no-title", action="store_true", help="omit the chart title (e.g. for a paper figure with its own caption)")
    ap.add_argument("--font-scale", type=float, default=1.0, help="multiplier on all chart text size (default: 1.0)")
    add_exclude_argument(ap)
    args = ap.parse_args()

    csv_path = Path(args.adjustment_times_csv)
    out_path = Path(args.out) if args.out else csv_path.parent / "adjustment_time_comparison.png"

    per_condition = build_participant_condition_table(csv_path, args.exclude)
    plot_adjustment_time(per_condition, out_path, title=not args.no_title, font_scale=args.font_scale)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
