"""
plot_kinematics_conditions.py

Compares joint_kinematics.py's per-period metrics across the three robot
control conditions (KT / Position / Pose). Reads its summary.csv (one row
per bag x side x joint), reduces it to one value per participant x
condition, and draws one boxplot per measure in the same style as
plot_conditions.py (same condition colors, display labels, and
RM-ANOVA-gated Holm-corrected paired t-test brackets).

Reduction, per participant x condition:
  * per-period measures (duration, % time moving, path lengths) are
    averaged over the two sides;
  * per-joint measures (RMS jerk / acceleration) are first averaged over
    the 7 joints within each side, then over sides.
A participant missing one side (e.g. a zero-length period that
joint_kinematics.py skipped) uses the side that exists.

Kinematic measures are often right-skewed, so besides the brackets (which
match the paper's other figures) a stats CSV also reports the
nonparametric Friedman test per measure, to check the two agree.

Usage:
  ~/.springcontroller_venv/bin/python plot_kinematics_conditions.py \\
      --input ~/CQHRI_joint_kinematics/summary.csv \\
      --output ~/CQHRI_joint_kinematics/comparison \\
      --exclude p10:Pose

--exclude PID:CONDITION[:SIDE] (repeatable) drops invalid periods before
anything is averaged, e.g. p10:Pose (experimenter error: the arm wasn't
reset to the start pose before the condition began). The paired tests
use only participants with all three conditions, so excluding one
condition removes that participant from the tests. Their other
conditions still appear as dots.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare
from statsmodels.stats.anova import AnovaRM

from plot_conditions import (CONDITIONS, CONDITION_COLORS, DISPLAY_LABELS,
                             bracket_levels, significant_pairs)

# (column in the tidy table, chart title, y-axis label, scale applied to raw values)
MEASURES = [
    ("duration_s", "Adjustment duration", "seconds", 1.0),
    ("pct_time_moving", "Time arm was moving", "% of period", 100.0),
    ("joint_path_length_rad", "Joint-space path length", "rad (all joints)", 1.0),
    ("ee_path_length_m", "End-effector path length", "m", 1.0),
    ("ee_displacement_m", "End-effector net displacement", "m", 1.0),
    ("mean_rms_jerk", "RMS jerk (mean over joints)", "rad/s³", 1.0),
    ("mean_rms_acc", "RMS acceleration (mean over joints)", "rad/s²", 1.0),
]

PER_PERIOD = ["duration_s", "frac_time_robot_moving", "joint_path_length_rad",
              "ee_path_length_m", "ee_displacement_m"]
PER_JOINT = {"rms_jerk": "mean_rms_jerk", "rms_acc": "mean_rms_acc"}


def parse_exclusion(spec: str) -> tuple:
    parts = spec.split(":")
    if len(parts) not in (2, 3) or not all(parts):
        raise argparse.ArgumentTypeError(
            f"--exclude {spec!r}: expected PID:CONDITION or PID:CONDITION:SIDE")
    pid = parts[0].lower()
    pid = pid if pid.startswith("p") else f"p{pid}"
    return (pid, parts[1].lower(), parts[2] if len(parts) == 3 else None)


def apply_exclusions(summary: pd.DataFrame, exclusions: list) -> pd.DataFrame:
    """Drop summary rows matching any (pid, condition, side-or-None).
    An exclusion that matches nothing is an error, so a typo can't
    silently leave a bad period in."""
    pid = summary["participant_id"].astype(str).str.lower()
    pid = pid.where(pid.str.startswith("p"), "p" + pid)
    cond = summary["condition"].astype(str).str.lower()
    side = summary["side"].astype(str)
    drop = pd.Series(False, index=summary.index)
    for p, c, sd in exclusions:
        hit = (pid == p) & (cond == c) & ((side == sd) if sd else True)
        if not hit.any():
            raise SystemExit(f"--exclude {p}:{c}{':' + sd if sd else ''} matches no rows")
        sides = sorted(summary.loc[hit, "side"].astype(str).unique())
        print(f"Excluded {p} {c} side(s) {', '.join(sides)}")
        drop |= hit
    return summary[~drop]


def tidy_from_summary(summary: pd.DataFrame) -> pd.DataFrame:
    """One row per participant x condition, sides averaged."""
    keys = ["participant_id", "condition", "side"]
    periods = summary.groupby(keys)[PER_PERIOD].first()
    joints = summary.groupby(keys)[list(PER_JOINT)].mean().rename(columns=PER_JOINT)
    per_side = periods.join(joints).reset_index()
    per_side["pct_time_moving"] = per_side.pop("frac_time_robot_moving")
    tidy = per_side.groupby(["participant_id", "condition"]).mean(numeric_only=True).reset_index()
    tidy["n_sides"] = per_side.groupby(["participant_id", "condition"]).size().values
    tidy["condition"] = tidy["condition"].map(
        {c.lower(): c for c in CONDITIONS}.get).fillna(tidy["condition"])
    return tidy


def stats_row(tidy: pd.DataFrame, measure: str) -> dict:
    wide = tidy.pivot_table(index="participant_id", columns="condition",
                            values=measure).reindex(columns=CONDITIONS).dropna()
    row = {"measure": measure, "n_complete": len(wide)}
    for c in CONDITIONS:
        row[f"median_{c}"] = wide[c].median()
    if len(wide) >= 3:
        long = wide.reset_index().melt(id_vars="participant_id", var_name="condition",
                                       value_name=measure)
        row["rm_anova_p"] = AnovaRM(long, depvar=measure, subject="participant_id",
                                    within=["condition"]).fit().anova_table["Pr > F"].iloc[0]
        row["friedman_p"] = friedmanchisquare(*[wide[c] for c in CONDITIONS]).pvalue
    row["holm_sig_pairs"] = "; ".join(f"{a}-{b} {s}" for a, b, s in significant_pairs(tidy, measure))
    return row


def draw_box(ax, tidy, measure, title, ylabel, fontsize):
    data = [tidy.loc[tidy["condition"] == c, measure].dropna().values for c in CONDITIONS]
    # showfliers=False: every participant is already drawn as a dot below.
    bp = ax.boxplot(data, patch_artist=True, showmeans=True, showfliers=False, widths=0.6,
                    medianprops={"color": "black"},
                    meanprops={"marker": "D", "markerfacecolor": "white",
                               "markeredgecolor": "black", "markersize": 5})
    for patch, cond in zip(bp["boxes"], CONDITIONS):
        patch.set_facecolor(CONDITION_COLORS[cond])
        patch.set_alpha(0.7)
    rng = np.random.default_rng(0)
    for i, values in enumerate(data, start=1):
        ax.scatter(i + rng.uniform(-0.12, 0.12, len(values)), values,
                   color="black", alpha=0.5, s=12, zorder=3)
    ax.set_title(title, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize * 0.9)
    ax.set_xticks(range(1, len(CONDITIONS) + 1))
    ax.set_xticklabels([DISPLAY_LABELS.get(c, c) for c in CONDITIONS], fontsize=fontsize * 0.9)
    ax.spines[["top", "right"]].set_visible(False)

    placed = bracket_levels(significant_pairs(tidy, measure))
    if placed:
        lo = min(d.min(initial=np.inf) for d in data)
        hi = max(d.max(initial=-np.inf) for d in data)
        step = 0.08 * (hi - lo)
        for i1, i2, stars, level in placed:
            y = hi + step * (0.6 + level * 1.4)
            ax.plot([i1 + 1, i1 + 1, i2 + 1, i2 + 1], [y, y + step * 0.3, y + step * 0.3, y],
                    color="black", lw=1)
            ax.text((i1 + i2) / 2 + 1, y + step * 0.3, stars, ha="center", va="bottom",
                    fontsize=fontsize)
        ax.set_ylim(top=hi + step * (1.6 + max(p[3] for p in placed) * 1.4))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--input", required=True, help="joint_kinematics.py summary.csv")
    ap.add_argument("--output", required=True, help="Folder for charts + stats CSV")
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--format", default="png")
    ap.add_argument("--exclude", action="append", default=[], type=parse_exclusion,
                    metavar="PID:CONDITION[:SIDE]",
                    help="Drop an invalid period (repeatable), e.g. p10:Pose.")
    args = ap.parse_args()

    summary = apply_exclusions(pd.read_csv(Path(args.input).expanduser()), args.exclude)
    tidy = tidy_from_summary(summary)
    out = Path(args.output).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    for col, _, _, scale in MEASURES:
        tidy[col] = tidy[col] * scale
    tidy.to_csv(out / "kinematics_by_condition.csv", index=False)

    one_side = tidy[tidy["n_sides"] < 2]
    if len(one_side):
        print("Only one side available (not averaged):")
        for _, r in one_side.iterrows():
            print(f"  {r['participant_id']} {r['condition']}")

    fs = plt.rcParams["font.size"]
    for col, title, ylabel, _ in MEASURES:
        fig, ax = plt.subplots(figsize=(5, 4.5))
        draw_box(ax, tidy, col, title, ylabel, fs * 1.1)
        fig.tight_layout()
        fig.savefig(out / f"{col}.{args.format}", dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)

    ncols = 4
    nrows = int(np.ceil(len(MEASURES) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 4 * nrows))
    for ax, (col, title, ylabel, _) in zip(axes.flat, MEASURES):
        draw_box(ax, tidy, col, title, ylabel, fs)
    for ax in axes.flat[len(MEASURES):]:
        ax.axis("off")
    wide = tidy.pivot_table(index="participant_id", columns="condition",
                            values="duration_s").reindex(columns=CONDITIONS)
    n_all, n_complete = len(wide), len(wide.dropna())
    n_txt = f"n = {n_all}" if n_complete == n_all else f"n = {n_all}, {n_complete} in paired tests"
    excl = ("; excluded " + ", ".join(f"{p} {c}" + (f" side {sd}" if sd else "")
                                      for p, c, sd in args.exclude)) if args.exclude else ""
    fig.suptitle(f"Joint kinematics by condition ({n_txt}; sides averaged{excl}; "
                 "* Holm-corrected paired t-test after RM-ANOVA)", fontsize=fs * 1.2)
    fig.tight_layout()
    fig.savefig(out / f"overview.{args.format}", dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    stats = pd.DataFrame([stats_row(tidy, col) for col, *_ in MEASURES])
    stats.to_csv(out / "stats.csv", index=False)
    with pd.option_context("display.width", 200, "display.float_format", "{:.4g}".format):
        print(stats.to_string(index=False))
    print(f"\nWrote charts and stats to {out}")


if __name__ == "__main__":
    main()
