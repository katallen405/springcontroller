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

Requires pingouin + pandas (see README's ANOVA section for install).
"""
import argparse
from pathlib import Path

import pandas as pd

from stats_helpers import posthoc_to_latex, rm_anova_to_latex, rm_anova_with_posthoc, warn_if_unbalanced


def build_participant_condition_table(adjustment_times_csv):
    """One row per (Participant, Condition): that participant's two sides'
    Total Adjustment Time summed."""
    df = pd.read_csv(adjustment_times_csv)
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
    args = ap.parse_args()

    per_condition = build_participant_condition_table(args.adjustment_times_csv)

    warn_if_unbalanced(per_condition, "Condition", "Participant")

    n_participants = per_condition["Participant"].nunique()
    print(f"{n_participants} participants x {per_condition['Condition'].nunique()} conditions "
          f"(each participant's two sides summed per condition)\n")

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
