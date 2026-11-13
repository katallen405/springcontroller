#!/usr/bin/env python3
"""Shared repeated-measures ANOVA + post-hoc helper, used by
anova_conditions.py and anova_adjustment_time.py -- both fit a one-way
RM-ANOVA (condition as the within-subject factor, participant as the
subject) to a per-participant-per-condition table and only run post-hoc
pairwise comparisons when the omnibus result is significant.
"""
import pingouin as pg


def rm_anova_with_posthoc(df, dv, within, subject, alpha=0.05, posthoc_padjust="holm"):
    """Prints the omnibus RM-ANOVA table for `dv`, then Holm-corrected
    post-hoc pairwise comparisons (paired t-tests) if the omnibus result is
    significant -- p_GG_corr if rm_anova found sphericity violated, else
    p_unc. Returns (anova_result, posthoc_result_or_None)."""
    result = pg.rm_anova(data=df, dv=dv, within=within, subject=subject, detailed=True)
    print(result.to_string(index=False))

    omnibus = result.iloc[0]
    omnibus_p = omnibus["p_GG_corr"] if "p_GG_corr" in result.columns else omnibus["p_unc"]

    posthoc = None
    if omnibus_p < alpha:
        print(f"\nomnibus p={omnibus_p:.4f} < alpha={alpha} -- post-hoc pairwise comparisons (Holm-corrected):")
        posthoc = pg.pairwise_tests(data=df, dv=dv, within=within, subject=subject, padjust=posthoc_padjust)
        print(posthoc.to_string(index=False))
    return result, posthoc


def rm_anova_to_latex(anova_result, caption, label):
    """One-row LaTeX table (booktabs) for an rm_anova() result's omnibus
    effect: F, df1, df2, and p (Greenhouse-Geisser-corrected if rm_anova
    found sphericity violated, else uncorrected)."""
    effect, error = anova_result.iloc[0], anova_result.iloc[1]
    uses_gg = "p_GG_corr" in anova_result.columns
    p = effect["p_GG_corr"] if uses_gg else effect["p_unc"]
    p_header = r"$p_{GG}$" if uses_gg else r"$p$"
    return "\n".join([
        r"\begin{table}[h]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        rf"Source & $F$ & $df_1$ & $df_2$ & {p_header} \\",
        r"\midrule",
        rf"{effect['Source']} & {effect['F']:.2f} & {int(effect['DF'])} & {int(error['DF'])} & {p:.3f} \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])


def posthoc_to_latex(posthoc_result, caption, label):
    """LaTeX table (booktabs) for a pairwise_tests() result: one row per
    condition-pair contrast."""
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"A & B & $T$ & $p$ (unc.) & $p$ (Holm) & Hedges' $g$ \\",
        r"\midrule",
    ]
    for _, row in posthoc_result.iterrows():
        lines.append(
            rf"{row['A']} & {row['B']} & {row['T']:.2f} & {row['p_unc']:.3f} & {row['p_corr']:.3f} & {row['hedges']:.2f} \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def warn_if_unbalanced(df, condition_col, participant_col):
    """Prints a warning if some participant is missing a condition -- pingouin
    drops incomplete participants from that metric's test rather than
    erroring, so this is the only sign that happened."""
    counts = df.groupby(condition_col)[participant_col].nunique()
    n_participants = df[participant_col].nunique()
    unbalanced = counts[counts != n_participants]
    if not unbalanced.empty:
        print(
            f"WARNING: unbalanced design -- {n_participants} participant(s) total, but "
            f"{dict(unbalanced)} condition(s) don't have all of them. "
            "pingouin's rm_anova will drop incomplete participants."
        )
