#!/usr/bin/env python3
"""Repeated-measures ANOVA of adjustment time across conditions, from the
table written by adjustment_times.py.

Same within-subjects design as anova_conditions.py (every participant did
all three conditions, each on two sides): builds one row per (Participant,
Condition) by summing that participant's two sides' Total Adjustment Time
for the condition, then runs a one-way repeated-measures ANOVA (condition
as the within-subject factor, participant as the subject) followed by
Holm-corrected post-hoc pairwise comparisons if the omnibus result is
significant -- see stats_helpers.rm_anova_with_posthoc.

Pass --latex to also write a LaTeX (booktabs) table of the omnibus result,
and of the post-hoc comparisons if they ran, for dropping into a paper.

--exclude PID:CONDITION[:SIDE] (repeatable) drops invalid trials before
the sides are summed, e.g. p10:Pose (experimenter error: the arm wasn't
reset to the start pose before the condition began, leaving nothing to
adjust). Parsing and matching come from ../exclusions.py, shared with
the other analysis scripts. A participant missing a condition is dropped
from the ANOVA by pingouin, so excluding one condition removes that
participant from the test.

Requires pingouin + pandas (see README's ANOVA section for install).
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

from stats_helpers import posthoc_to_latex, rm_anova_to_latex, rm_anova_with_posthoc, warn_if_unbalanced

# exclusions.py lives one level up, in analysis/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import exclusions as ex_mod  # noqa: E402

# plot_adjustment_time.py imports this name from here.
add_exclude_argument = ex_mod.add_argument


def apply_exclusions(df, exclusions):
    """Drop adjustment_times.csv rows matching any exclusion. One that
    matches nothing is an error, so a typo can't silently keep a bad trial."""
    keys = list(zip(df["Participant"], df["Condition"], df["Side"]))
    side = df["Side"].astype(str).str.strip()
    drop = pd.Series(False, index=df.index)
    for ex in exclusions:
        hit = pd.Series([ex_mod.matches(ex, *k) for k in keys], index=df.index)
        if not hit.any():
            raise SystemExit(f"--exclude {ex_mod.describe(ex)} matches no rows")
        print(f"excluded {ex_mod.describe(ex)} (side(s) {', '.join(sorted(side[hit].unique()))})")
        drop |= hit
    return df[~drop]


def build_participant_condition_table(adjustment_times_csv, exclusions=()):
    """One row per (Participant, Condition): that participant's two sides'
    Total Adjustment Time summed, after dropping any excluded trials."""
    df = apply_exclusions(pd.read_csv(adjustment_times_csv), exclusions)
    return (
        df.groupby(["Participant", "Condition"], as_index=False)["Total Adjustment Time"]
        .sum()
        .rename(columns={"Total Adjustment Time": "total_adjustment_time_sec"})
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("adjustment_times_csv", help="output of adjustment_times.py")
    ap.add_argument(
        "--alpha", type=float, default=0.05,
        help="significance threshold for running post-hoc pairwise comparisons "
        "after the omnibus ANOVA (default: 0.05)",
    )
    ap.add_argument("--latex", action="store_true", help="also write a LaTeX table of the results")
    ap.add_argument(
        "--latex-out", default=None,
        help="LaTeX output path (default: <adjustment_times_csv's directory>/adjustment_time_anova.tex)",
    )
    add_exclude_argument(ap)
    args = ap.parse_args()

    per_condition = build_participant_condition_table(args.adjustment_times_csv, args.exclude)

    warn_if_unbalanced(per_condition, "Condition", "Participant")

    n_participants = per_condition["Participant"].nunique()
    n_conditions = per_condition["Condition"].nunique()
    n_complete = (per_condition.groupby("Participant")["Condition"].nunique() == n_conditions).sum()
    print(f"{n_participants} participants x {n_conditions} conditions "
          f"(each participant's two sides summed per condition); "
          f"{n_complete} with all conditions go into the ANOVA\n")

    anova_result, posthoc_result = rm_anova_with_posthoc(
        per_condition, dv="total_adjustment_time_sec", within="Condition", subject="Participant", alpha=args.alpha
    )

    if args.latex:
        csv_path = Path(args.adjustment_times_csv)
        latex_out = Path(args.latex_out) if args.latex_out else csv_path.parent / "adjustment_time_anova.tex"
        tables = [rm_anova_to_latex(
            anova_result,
            caption="Repeated-measures ANOVA of total adjustment time by condition",
            label="tab:anova-adjustment-time",
        )]
        if posthoc_result is not None:
            tables.append(posthoc_to_latex(
                posthoc_result,
                caption="Post-hoc pairwise comparisons of total adjustment time (Holm-corrected)",
                label="tab:posthoc-adjustment-time",
            ))
        latex_out.write_text("\n\n".join(tables) + "\n")
        print(f"\nwrote {latex_out}")


if __name__ == "__main__":
    main()
