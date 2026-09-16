#!/usr/bin/env python3
"""Batch-classify a directory of *_posture.csv files (extract_posture.py
output, one per participant/condition/etc.) and plot their time-in-zone
exposure side by side.

This reuses classify_ergonomics.py's own per-session classification
(load_posture_rows/shoulder_fallback_thresholds/summarize) rather than
reimplementing it, so a session's numbers here always match what
`classify_ergonomics.py <that_session>_posture.csv` would report on its
own -- thresholds stay session-relative per classify_ergonomics.py's
design (see its docstring), just computed once per file instead of once.

Does NOT write each session's *_zones.csv/*_episodes.csv/*_ergo_summary.csv
-- run classify_ergonomics.py directly on a session if you need those (e.g.
to spot-check a session that looks like an outlier here against its
overlay video).
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from classify_ergonomics import load_posture_rows, shoulder_fallback_thresholds, summarize

CATEGORY_COLORS = {
    "hunch": "tab:red",
    "lean": "tab:purple",
    "neutral": "tab:green",
    "unclassified": "tab:gray",
}


def session_label(csv_path):
    stem = csv_path.stem
    return stem[: -len("_posture")] if stem.endswith("_posture") else stem


def classify_session(csv_path, hunch_thresh_deg, lean_thresh_deg, shoulder_percentile, gap_tolerance, sustained_seconds):
    rows = load_posture_rows(csv_path)
    if not rows:
        return None
    dist_thresh, line_thresh = shoulder_fallback_thresholds(rows, shoulder_percentile)
    _, _, _, summary = summarize(
        rows, hunch_thresh_deg, lean_thresh_deg, dist_thresh, line_thresh,
        gap_tolerance, sustained_seconds,
    )
    return summary


def plot_comparison(sessions, summaries, out_path):
    n = len(sessions)
    categories = ["hunch", "lean", "neutral", "unclassified"]
    x = range(n)
    width = 0.2

    fig, ax = plt.subplots(figsize=(max(8, n * 0.9), 6))
    for i, cat in enumerate(categories):
        if cat == "unclassified":
            vals = [s["pct_unclassified_of_wallclock"] for s in summaries]
        else:
            vals = [s[f"pct_{cat}_of_classified"] for s in summaries]
        offsets = [xi + (i - (len(categories) - 1) / 2) * width for xi in x]
        ax.bar(offsets, vals, width=width, label=cat, color=CATEGORY_COLORS[cat])

    ax.set_xticks(list(x))
    ax.set_xticklabels(sessions, rotation=45, ha="right")
    ax.set_ylabel("% of time")
    ax.set_title(
        "Time-in-zone exposure by session\n"
        "hunch/lean/neutral are % of classified time; unclassified is % of wall-clock time (different denominator)"
    )
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("directory", help="directory containing *_posture.csv files")
    ap.add_argument("--pattern", default="*_posture.csv", help="glob pattern for input files (default: *_posture.csv)")
    ap.add_argument("--hunch-threshold-deg", type=float, default=20.0)
    ap.add_argument("--lean-threshold-deg", type=float, default=15.0)
    ap.add_argument("--shoulder-percentile", type=float, default=75.0)
    ap.add_argument("--gap-tolerance", type=float, default=1.0)
    ap.add_argument("--sustained-seconds", type=float, default=4.0)
    ap.add_argument("--out", default=None, help="output PNG (default: <directory>/exposure_comparison.png)")
    ap.add_argument("--out-csv", default=None, help="aggregated per-session summary CSV (default: <directory>/exposure_comparison_summary.csv)")
    args = ap.parse_args()

    directory = Path(args.directory)
    files = sorted(directory.glob(args.pattern))
    if not files:
        raise SystemExit(f"no files matching {args.pattern!r} in {directory}")

    out_path = Path(args.out) if args.out else directory / "exposure_comparison.png"
    out_csv = Path(args.out_csv) if args.out_csv else directory / "exposure_comparison_summary.csv"

    sessions, summaries, skipped = [], [], []
    for f in files:
        summary = classify_session(
            f, args.hunch_threshold_deg, args.lean_threshold_deg,
            args.shoulder_percentile, args.gap_tolerance, args.sustained_seconds,
        )
        if summary is None:
            skipped.append(f.name)
            continue
        sessions.append(session_label(f))
        summaries.append(summary)

    if not summaries:
        raise SystemExit(f"none of the {len(files)} matched file(s) had any rows")

    plot_comparison(sessions, summaries, out_path)

    fieldnames = ["session", "source_file"] + list(summaries[0].keys())
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for label, f, summary in zip(sessions, [f for f in files if f.name not in skipped], summaries):
            w.writerow({"session": label, "source_file": f.name, **summary})

    print(f"{len(sessions)} session(s) plotted ({len(skipped)} skipped: empty file)" if skipped else f"{len(sessions)} session(s) plotted")
    print(f"wrote {out_path}")
    print(f"wrote {out_csv}")


if __name__ == "__main__":
    main()
