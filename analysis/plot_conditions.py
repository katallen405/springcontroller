"""
plot_conditions.py

Reads the tidy scored CSV produced by analyze.py (one row per
participant x condition, columns: participant_id, condition, fluency,
adapts, helpfulness) and produces one chart per measure comparing the
three robot control conditions (KT / Position / Pose, shown on-chart as
"Adm." / "Pos.-only" / "Pose" -- see DISPLAY_LABELS): a diverging stacked
bar for fluency (a single 5-point item, centered on the neutral rating of
3) and a boxplot for adapts/helpfulness (each a mean of several items).
Each chart draws a significance bracket for every condition pair that's
significant (Holm-corrected paired t-test, gated on a significant omnibus
RM-ANOVA -- see significant_pairs()), stacked so brackets never overlap.
This used to annotate KT-vs-Pose only, which hid a significant
Position-vs-Pose helpfulness result that the paper's tables report.

Deliberately separate from analyze.py: re-plotting doesn't require
re-parsing the raw Qualtrics export, and this script's conventions mirror
the plotting scripts in
~/Dropbox/Research/SmartPlayground/VideoDataAnalysis/ (argparse CLI,
matplotlib Agg backend, a fixed colorblind-friendly palette, one output
file per chart).

Usage:
  python plot_conditions.py --input scored_data.csv --output charts/

Options:
  --input       Tidy CSV written by analyze.py (default: scored_data.csv)
  --output      Folder where chart files are saved (created if needed)
  --dpi         Resolution of output images (default: 150)
  --format      Output format: png, pdf, svg (default: png)
  --no-title    Omit the chart title (e.g. for a paper figure with its own caption)
  --font-scale  Multiplier on all chart text size (default: 1.0; e.g. 3 for a
                small multi-panel figure like a three-up subfigure row)
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import pandas as pd
from scipy.stats import ttest_rel
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multitest import multipletests

from analyze import wide_for_measure

CONDITIONS = ["KT", "Position", "Pose"]

# Fixed 3-color colorblind-friendly palette (Okabe-Ito subset), one color
# per condition in the order above.
CONDITION_COLORS = {
    "KT": "#0072B2",
    "Position": "#E69F00",
    "Pose": "#009E73",
}

# Display labels for chart axes/ticks only -- CONDITIONS/CONDITION_COLORS
# keys stay "KT"/"Position"/"Pose" since those match the values in the
# tidy CSV (and the filenames compare_conditions.py/plot_adjustment_time.py
# parse in posture/).
DISPLAY_LABELS = {
    "KT": "Baseline",
    "Position": "Pos.-only",
    "Pose": "Pose",
}

MEASURE_TITLES = {
    "fluency": "Fluency",
    "adapts": "PSI: Adapts to Human Behaviors",
    "helpfulness": "PSI: Helpfulness",
}

# Fluency (QID10/12/14) is a single 5-point item, not a mean of several --
# unlike adapts/helpfulness it's naturally a discrete Likert distribution,
# plotted as a diverging stacked bar rather than a boxplot. Colors are two
# Okabe-Ito hues (one per pole) flanking a neutral gray midpoint -- same
# colorblind-safe family as CONDITION_COLORS above, just used for a
# different encoding (response level, not condition).
FLUENCY_LEVELS = [1, 2, 3, 4, 5]
FLUENCY_LEVEL_LABELS = {
    1: "1 - Not at all fluent",
    2: "2 - Not very fluent",
    3: "3 - Acceptable",
    4: "4 - Good",
    5: "5 - Extremely fluent",
}
FLUENCY_LEVEL_COLORS = {
    1: "#0072B2",
    2: "#56B4E9",
    3: "#BBBBBB",
    4: "#E69F00",
    5: "#D55E00",
}

PAIRS = [("KT", "Position"), ("KT", "Pose"), ("Position", "Pose")]

# Vertical spacing between stacked boxplot brackets, in score units.
BRACKET_STEP = 0.35


def p_to_stars(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return None


def significant_pairs(df, measure):
    """Mirrors analyze.py's run_rm_anova_and_posthoc: an omnibus RM-ANOVA
    gate, then Holm-corrected paired t-tests across all 3 condition pairs.
    Returns [(cond_a, cond_b, stars), ...] for every pair whose corrected
    p-value is significant -- empty if the omnibus test isn't."""
    _, complete = wide_for_measure(df, measure)
    if len(complete) < 3:
        return []
    long = complete.reset_index().melt(id_vars="participant_id", value_vars=CONDITIONS,
                                        var_name="condition", value_name=measure)
    omnibus = AnovaRM(long, depvar=measure, subject="participant_id", within=["condition"]).fit()
    if omnibus.anova_table["Pr > F"].iloc[0] >= 0.05:
        return []
    pvals = [ttest_rel(complete[a], complete[b]).pvalue for a, b in PAIRS]
    _, corrected, _, _ = multipletests(pvals, method="holm")
    return [(a, b, p_to_stars(p)) for (a, b), p in zip(PAIRS, corrected) if p_to_stars(p)]


def bracket_levels(pairs):
    """
    Stacking level (0 = innermost) for each (cond_a, cond_b, stars) pair,
    so overlapping brackets never share a level. Narrow spans are placed
    first, so a bracket between neighbors sits inside one that spans all
    three conditions. Returns [(i1, i2, stars, level), ...] with i1 < i2
    as indices into CONDITIONS.
    """
    spans = sorted(
        ((min(CONDITIONS.index(a), CONDITIONS.index(b)),
          max(CONDITIONS.index(a), CONDITIONS.index(b)), stars) for a, b, stars in pairs),
        key=lambda s: (s[1] - s[0], s[0]),
    )
    placed = []
    for i1, i2, stars in spans:
        # Brackets sharing an endpoint overlap too (their end ticks would
        # land on the same box), hence <= rather than <.
        taken = {lvl for j1, j2, _, lvl in placed if j1 <= i2 and i1 <= j2}
        level = next(lvl for lvl in range(len(PAIRS) + 1) if lvl not in taken)
        placed.append((i1, i2, stars, level))
    return placed


def add_significance_bracket(ax, x1, x2, y, stars, fontsize):
    """Horizontal significance bracket between x1 and x2 at height y, star(s)
    centered above it. Expands the y-axis top so the bracket/star aren't
    clipped."""
    ymin, ymax = ax.get_ylim()
    tick = (ymax - ymin) * 0.02
    ax.plot([x1, x1, x2, x2], [y, y + tick, y + tick, y], color="black", linewidth=1, clip_on=False)
    ax.text((x1 + x2) / 2, y + tick, stars, ha="center", va="bottom", fontsize=fontsize)
    ax.set_ylim(ymin, max(ymax, y + tick * 4))


def make_boxplot(df, measure, out_path, dpi=150, title=True, font_scale=1.0):
    base_fontsize = plt.rcParams["font.size"] * font_scale
    data = [df.loc[df["condition"] == c, measure].dropna().values for c in CONDITIONS]

    fig, ax = plt.subplots(figsize=(6, 5))
    bp = ax.boxplot(data, labels=CONDITIONS, patch_artist=True, showmeans=True)
    for patch, cond in zip(bp["boxes"], CONDITIONS):
        patch.set_facecolor(CONDITION_COLORS[cond])
        patch.set_alpha(0.7)

    for i, values in enumerate(data, start=1):
        jitter = (pd.Series(range(len(values))).sub(len(values) / 2) * 0.02) if len(values) else []
        ax.scatter([i + j for j in jitter], values, color="black", alpha=0.5, s=15, zorder=3)

    if title:
        ax.set_title(MEASURE_TITLES.get(measure, measure), fontsize=base_fontsize)
    ax.set_ylabel("Score (1-5)", fontsize=base_fontsize)
    ax.set_xticklabels([DISPLAY_LABELS.get(c, c) for c in CONDITIONS], fontsize=base_fontsize)
    ax.tick_params(axis="y", labelsize=base_fontsize)
    ax.set_ylim(0.5, 5.5)

    placed = bracket_levels(significant_pairs(df, measure))
    if placed:
        # All brackets start above the tallest data point among the boxes
        # they could span, so stacked levels stay evenly spaced.
        top = max(data[i].max(initial=0) for i1, i2, _, _ in placed for i in range(i1, i2 + 1))
        for i1, i2, stars, level in placed:
            add_significance_bracket(ax, i1 + 1, i2 + 1, top + 0.15 + level * BRACKET_STEP,
                                     stars, base_fontsize)

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def make_fluency_diverging(df, out_path, dpi=150, title=True, font_scale=1.0):
    """Horizontal stacked bar of the fluency response distribution per
    condition, diverging around the midpoint rating of 3: the "3" segment
    straddles x=0 (half to each side), with 1/2 stacking further left and
    4/5 stacking further right -- so the whole stack is centered on that
    neutral rating rather than on zero frames/participants."""
    base_fontsize = plt.rcParams["font.size"] * font_scale
    measure = "fluency"

    pct_by_condition = {}
    for c in CONDITIONS:
        values = df.loc[df["condition"] == c, measure].dropna().astype(int)
        total = len(values)
        counts = values.value_counts()
        pct_by_condition[c] = {lvl: 100.0 * counts.get(lvl, 0) / total for lvl in FLUENCY_LEVELS}

    fig, ax = plt.subplots(figsize=(8, 0.9 * len(CONDITIONS) + 1.5))
    max_extent = 0.0
    for yi, c in enumerate(CONDITIONS):
        pct = pct_by_condition[c]
        half_mid = pct[3] / 2.0

        right_edge = half_mid
        for lvl in (4, 5):
            ax.barh(yi, pct[lvl], left=right_edge, height=0.6,
                    color=FLUENCY_LEVEL_COLORS[lvl],
                    label=FLUENCY_LEVEL_LABELS[lvl] if yi == 0 else None)
            right_edge += pct[lvl]

        ax.barh(yi, pct[3], left=-half_mid, height=0.6,
                color=FLUENCY_LEVEL_COLORS[3],
                label=FLUENCY_LEVEL_LABELS[3] if yi == 0 else None)

        left_edge = -half_mid
        for lvl in (2, 1):
            left_edge -= pct[lvl]
            ax.barh(yi, pct[lvl], left=left_edge, height=0.6,
                    color=FLUENCY_LEVEL_COLORS[lvl],
                    label=FLUENCY_LEVEL_LABELS[lvl] if yi == 0 else None)

        max_extent = max(max_extent, right_edge, -left_edge)

    ax.axvline(0, color="black", linewidth=0.8, linestyle="--", zorder=1)
    ax.set_yticks(range(len(CONDITIONS)))
    ax.set_yticklabels([DISPLAY_LABELS.get(c, c) for c in CONDITIONS], fontsize=base_fontsize)
    ax.invert_yaxis()

    xlim = max_extent * 1.15
    ax.set_xlim(-xlim, xlim)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{abs(x):.0f}"))
    ax.set_xlabel("% of participants (bars centered on rating 3)", fontsize=base_fontsize)
    ax.tick_params(axis="x", labelsize=base_fontsize)

    if title:
        ax.set_title(MEASURE_TITLES.get(measure, measure), fontsize=base_fontsize)

    tick = xlim * 0.025
    for y1, y2, stars, level in bracket_levels(significant_pairs(df, measure)):
        # Vertical brackets to the right of the bars; each stacking level
        # moves further right, leaving room for the previous level's stars
        # (which widen with font_scale).
        bracket_x = xlim * (1.05 + (0.06 + 0.06 * font_scale) * level)
        ax.plot([bracket_x, bracket_x], [y1, y2], color="black", linewidth=1, clip_on=False)
        ax.plot([bracket_x - tick, bracket_x], [y1, y1], color="black", linewidth=1, clip_on=False)
        ax.plot([bracket_x - tick, bracket_x], [y2, y2], color="black", linewidth=1, clip_on=False)
        ax.text(bracket_x + tick, (y1 + y2) / 2, stars, ha="left", va="center", fontsize=base_fontsize)

    # Place the legend below the (already-drawn) x-axis label rather than at
    # a fixed offset -- a fixed fraction of axes height doesn't scale with
    # font_scale, so at 2x+ font sizes a fixed offset let the legend collide
    # with the label text.
    fig.canvas.draw()
    xlabel_bbox_axes = ax.xaxis.label.get_window_extent(
        renderer=fig.canvas.get_renderer()
    ).transformed(ax.transAxes.inverted())
    legend_y = xlabel_bbox_axes.y0 - 0.04

    handles, labels = ax.get_legend_handles_labels()
    order = [FLUENCY_LEVEL_LABELS[lvl] for lvl in FLUENCY_LEVELS]
    by_label = dict(zip(labels, handles))
    ax.legend([by_label[l] for l in order], order, loc="upper center",
              bbox_to_anchor=(0.5, legend_y), ncol=3, frameon=False, fontsize=base_fontsize * 0.85)

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="scored_data.csv",
                         help="Tidy CSV written by analyze.py (default: scored_data.csv)")
    parser.add_argument("--output", default="charts",
                         help="Folder to save chart files into (default: charts)")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--format", default="png", choices=["png", "pdf", "svg"])
    parser.add_argument("--no-title", action="store_true", help="omit the chart title (e.g. for a paper figure with its own caption)")
    parser.add_argument("--font-scale", type=float, default=1.0, help="multiplier on all chart text size (default: 1.0)")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    fluency_path = out_dir / f"fluency_by_condition.{args.format}"
    make_fluency_diverging(df, fluency_path, dpi=args.dpi, title=not args.no_title, font_scale=args.font_scale)
    print(f"Wrote {fluency_path}")

    for measure in ["adapts", "helpfulness"]:
        out_path = out_dir / f"{measure}_by_condition.{args.format}"
        make_boxplot(df, measure, out_path, dpi=args.dpi, title=not args.no_title, font_scale=args.font_scale)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
