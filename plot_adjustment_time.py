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
~/sandbox/src/springcontroller/analysis/plot_conditions.py (Okabe-Ito,
one color per condition), which already compares this same study's
KT/Position/Pose conditions.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from anova_adjustment_time import build_participant_condition_table
from compare_conditions import condition_sort_key

CONDITION_COLORS = {
    "KT": "#0072B2",
    "Position": "#E69F00",
    "Pose": "#009E73",
}


def plot_adjustment_time(per_condition, out_path, title=True, font_scale=1.0, seed=0):
    conditions = sorted(per_condition["Condition"].unique(), key=condition_sort_key)
    base_fontsize = plt.rcParams["font.size"] * font_scale
    rng = np.random.default_rng(seed)

    fig, ax = plt.subplots(figsize=(9, 0.9 * len(conditions) + 1.5))
    for y, cond in enumerate(conditions):
        vals = per_condition.loc[per_condition["Condition"] == cond, "total_adjustment_time_sec"].to_numpy()
        mean = vals.mean()
        sem = vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
        color = CONDITION_COLORS.get(cond, "tab:gray")
        ax.barh(y, mean, xerr=sem, height=0.6, color=color, ecolor="black", capsize=4)
        jitter = rng.uniform(-0.18, 0.18, size=len(vals))
        ax.scatter(vals, [y] * len(vals) + jitter, color="black", s=14, alpha=0.5, zorder=3)

    ax.set_yticks(range(len(conditions)))
    ax.set_yticklabels(conditions, fontsize=base_fontsize)
    ax.invert_yaxis()
    ax.set_xlabel("Total adjustment time (s)", fontsize=base_fontsize)
    ax.tick_params(axis="x", labelsize=base_fontsize)
    if title:
        ax.set_title("Adjustment time by condition (mean ± SEM, points = participants)", fontsize=base_fontsize)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("adjustment_times_csv", help="output of adjustment_times.py")
    ap.add_argument("--out", default=None, help="output PNG (default: <adjustment_times_csv's directory>/adjustment_time_comparison.png)")
    ap.add_argument("--no-title", action="store_true", help="omit the chart title (e.g. for a paper figure with its own caption)")
    ap.add_argument("--font-scale", type=float, default=1.0, help="multiplier on all chart text size (default: 1.0)")
    args = ap.parse_args()

    csv_path = Path(args.adjustment_times_csv)
    out_path = Path(args.out) if args.out else csv_path.parent / "adjustment_time_comparison.png"

    per_condition = build_participant_condition_table(csv_path)
    plot_adjustment_time(per_condition, out_path, title=not args.no_title, font_scale=args.font_scale)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
