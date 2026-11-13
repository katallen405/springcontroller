#!/usr/bin/env python3
"""Classify hunch/lean exposure from a *_posture.csv (output of
extract_posture.py) using simple threshold + time-in-zone rules.

This is deliberately NOT an adapted RULA-style composite score. These are
camera-derived deviation proxies, not goniometric angles (see
extract_posture.py's docstring), so a heavier clinical scoring model would
be pretending to more precision than the input actually has. Instead: pick
a zone per frame, group contiguous same-zone frames into episodes, and
report time-in-zone -- duration matters as much as peak magnitude in
ergonomics risk, which is the same reasoning behind static-posture/NIOSH
duration guidance.

Per-frame zone
--------------
Primary signal is the face camera's head pose (available whenever
face_detected is True):
  - hunch: head_pitch_dev_deg > --hunch-threshold-deg (default 20, an
    adapted reference point from the commonly-cited >20deg neck-flexion
    risk cutoff in ergonomics literature -- NOT a validated cutoff for this
    rig; revisit once there's ground truth to compare against).
  - lean:  abs(head_roll_dev_deg) > --lean-threshold-deg (default 15).

When the face camera misses (which is common -- see extract_posture.py's
measured 13-62% face-detection rates), this script falls back to the
shoulder camera's proxies rather than leaving the frame unclassified,
since shoulders_detected is far more complete. But shoulder_dist_dev_px is
pixels, not degrees, so it can't share the literature-derived cutoff --
it's classified against a SESSION-RELATIVE percentile instead (default:
above the 75th percentile of that session's own shoulder_dist_dev_px /
abs(shoulder_line_dev_deg) = in zone). Every row records which source
(face vs shoulder_fallback) produced its label, so downstream analysis can
weight or filter by confidence.

Frames with neither signal (person not at the desk, or both detectors
missed) get zone=None and are excluded from all percentage-time-in-zone
denominators -- never silently counted as neutral.

Always sanity-check a flagged "sustained" episode against the annotated
overlay videos from extract_posture.py before trusting it.
"""
import argparse
import bisect
import csv
import datetime as dt
from pathlib import Path

import numpy as np

ZONE_HUNCH = "hunch"
ZONE_LEAN = "lean"
ZONE_BOTH = "hunch+lean"
ZONE_NEUTRAL = "neutral"


def to_float(v):
    try:
        f = float(v)
        return f if not np.isnan(f) else None
    except (TypeError, ValueError):
        return None


def to_bool(v):
    return v == "True"


def load_posture_rows(csv_path):
    rows = list(csv.DictReader(open(csv_path)))
    for r in rows:
        r["t_sec"] = float(r["t_sec"])
    rows.sort(key=lambda r: r["t_sec"])
    return rows


def shoulder_fallback_thresholds(rows, percentile):
    """Session-relative thresholds for the shoulder fallback, computed only
    from frames where shoulders were actually detected. Returns (None, None)
    if there's no shoulder data at all in this session."""
    dists = [to_float(r["shoulder_dist_dev_px"]) for r in rows if to_bool(r["shoulders_detected"])]
    lines = [abs(to_float(r["shoulder_line_dev_deg"])) for r in rows if to_bool(r["shoulders_detected"])]
    dists = [d for d in dists if d is not None]
    lines = [l for l in lines if l is not None]
    dist_thresh = float(np.percentile(dists, percentile)) if dists else None
    line_thresh = float(np.percentile(lines, percentile)) if lines else None
    return dist_thresh, line_thresh


def classify_row(row, hunch_thresh_deg, lean_thresh_deg, shoulder_dist_thresh, shoulder_line_thresh):
    """Returns (zone, source). zone is one of hunch/lean/hunch+lean/neutral,
    or None if neither signal was available for this frame."""
    if to_bool(row["face_detected"]):
        pitch = to_float(row["head_pitch_dev_deg"])
        roll = to_float(row["head_roll_dev_deg"])
        if pitch is None or roll is None:
            return None, None
        hunch = pitch > hunch_thresh_deg
        lean = abs(roll) > lean_thresh_deg
        source = "face"
    elif to_bool(row["shoulders_detected"]) and shoulder_dist_thresh is not None:
        dist = to_float(row["shoulder_dist_dev_px"])
        line = to_float(row["shoulder_line_dev_deg"])
        if dist is None or line is None:
            return None, None
        hunch = dist > shoulder_dist_thresh
        lean = abs(line) > shoulder_line_thresh
        source = "shoulder_fallback"
    else:
        return None, None

    if hunch and lean:
        return ZONE_BOTH, source
    if hunch:
        return ZONE_HUNCH, source
    if lean:
        return ZONE_LEAN, source
    return ZONE_NEUTRAL, source


def frame_weights(t_sec):
    """Seconds each frame 'covers', for time-in-zone accounting: the gap to
    the next frame, with the last frame given the median gap (there's no
    next frame to measure to)."""
    if len(t_sec) < 2:
        return [0.0] * len(t_sec)
    gaps = [t_sec[i + 1] - t_sec[i] for i in range(len(t_sec) - 1)]
    median_gap = float(np.median(gaps))
    return gaps + [median_gap]


def find_episodes(t_sec, zones, sources, gap_tolerance):
    """Group contiguous same-zone frames into episodes, bridging gaps of
    zone=None up to gap_tolerance seconds (a momentary missed detection)
    without merging across a genuinely different zone."""
    episodes = []
    current = None
    last_real_t = None

    for t, zone, source in zip(t_sec, zones, sources):
        if zone is None:
            continue
        if (
            current is not None
            and zone == current["zone"]
            and last_real_t is not None
            and (t - last_real_t) <= gap_tolerance
        ):
            current["end_t"] = t
            current["sources"].append(source)
        else:
            if current is not None:
                episodes.append(current)
            current = {"zone": zone, "start_t": t, "end_t": t, "sources": [source]}
        last_real_t = t

    if current is not None:
        episodes.append(current)

    for ep in episodes:
        ep["duration"] = ep["end_t"] - ep["start_t"]
        # majority source across the episode's frames, for a quick confidence read
        ep["source"] = max(set(ep["sources"]), key=ep["sources"].count)
        del ep["sources"]
    return episodes


def summarize(rows_slice, hunch_thresh_deg, lean_thresh_deg, dist_thresh, line_thresh, gap_tolerance, sustained_seconds):
    """Per-frame classification + episode detection + exposure summary for
    one slice of posture rows (a whole session, or one trial's rows)."""
    t_sec = [r["t_sec"] for r in rows_slice]
    zones, sources = [], []
    for r in rows_slice:
        z, s = classify_row(r, hunch_thresh_deg, lean_thresh_deg, dist_thresh, line_thresh)
        zones.append(z)
        sources.append(s)

    weights = frame_weights(t_sec)
    total_wallclock = sum(weights)
    time_unclassified = sum(w for w, z in zip(weights, zones) if z is None)
    time_hunch = sum(w for w, z in zip(weights, zones) if z in (ZONE_HUNCH, ZONE_BOTH))
    time_lean = sum(w for w, z in zip(weights, zones) if z in (ZONE_LEAN, ZONE_BOTH))
    time_neutral = sum(w for w, z in zip(weights, zones) if z == ZONE_NEUTRAL)
    total_classified = total_wallclock - time_unclassified

    def pct(x, denom):
        return 100.0 * x / denom if denom > 0 else float("nan")

    episodes = find_episodes(t_sec, zones, sources, gap_tolerance)
    sustained_hunch = [e for e in episodes if e["zone"] in (ZONE_HUNCH, ZONE_BOTH) and e["duration"] >= sustained_seconds]
    sustained_lean = [e for e in episodes if e["zone"] in (ZONE_LEAN, ZONE_BOTH) and e["duration"] >= sustained_seconds]

    summary = {
        "n_frames": len(rows_slice),
        "duration_wallclock_sec": total_wallclock,
        "duration_classified_sec": total_classified,
        "pct_unclassified_of_wallclock": pct(time_unclassified, total_wallclock),
        "pct_hunch_of_classified": pct(time_hunch, total_classified),
        "pct_lean_of_classified": pct(time_lean, total_classified),
        "pct_neutral_of_classified": pct(time_neutral, total_classified),
        "n_sustained_hunch_episodes": len(sustained_hunch),
        "n_sustained_lean_episodes": len(sustained_lean),
        "total_sustained_hunch_sec": sum(e["duration"] for e in sustained_hunch),
        "total_sustained_lean_sec": sum(e["duration"] for e in sustained_lean),
    }
    return zones, sources, episodes, summary


def load_trials(event_log_path):
    """Pairs 'Trial start' / 'Trial end' rows into (condition, start, end)
    trials, matched in event order (not by condition text, since duplicate
    condition names across repeated trials are expected)."""
    rows = list(csv.DictReader(open(event_log_path)))
    trials = []
    pending = None
    for r in rows:
        event = r["event"].strip()
        if event == "Trial start":
            pending = {"condition": r["condition"], "start": r["timestamp"]}
        elif event == "Trial end" and pending is not None:
            trials.append({"condition": pending["condition"], "start": pending["start"], "end": r["timestamp"]})
            pending = None
    return trials


def rows_in_trial(rows, trial):
    start = dt.datetime.fromisoformat(trial["start"])
    end = dt.datetime.fromisoformat(trial["end"])
    return [r for r in rows if start <= dt.datetime.fromisoformat(r["timestamp_iso"]) <= end]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("posture_csv", help="output of extract_posture.py")
    ap.add_argument("--hunch-threshold-deg", type=float, default=20.0)
    ap.add_argument("--lean-threshold-deg", type=float, default=15.0)
    ap.add_argument(
        "--shoulder-percentile", type=float, default=75.0,
        help="session-relative percentile used as the shoulder-fallback zone cutoff",
    )
    ap.add_argument(
        "--gap-tolerance", type=float, default=1.0,
        help="max seconds of a momentary zone=None gap to bridge within one episode",
    )
    ap.add_argument("--sustained-seconds", type=float, default=4.0, help="episode duration to flag as sustained")
    ap.add_argument("--event-log", default=None, help="optional event_log.csv for a per-trial breakdown")
    ap.add_argument("--out-zones", default=None, help="per-frame zone labels (default: <input>_zones.csv)")
    ap.add_argument("--out-episodes", default=None, help="per-episode table (default: <input>_episodes.csv)")
    ap.add_argument("--out-summary", default=None, help="exposure summary, one row per session/trial (default: <input>_ergo_summary.csv)")
    args = ap.parse_args()

    posture_csv = Path(args.posture_csv)
    out_zones = Path(args.out_zones) if args.out_zones else posture_csv.with_name(posture_csv.stem + "_zones.csv")
    out_episodes = Path(args.out_episodes) if args.out_episodes else posture_csv.with_name(posture_csv.stem + "_episodes.csv")
    out_summary = Path(args.out_summary) if args.out_summary else posture_csv.with_name(posture_csv.stem + "_ergo_summary.csv")

    rows = load_posture_rows(posture_csv)
    dist_thresh, line_thresh = shoulder_fallback_thresholds(rows, args.shoulder_percentile)

    zones, sources, episodes, session_summary = summarize(
        rows, args.hunch_threshold_deg, args.lean_threshold_deg, dist_thresh, line_thresh,
        args.gap_tolerance, args.sustained_seconds,
    )

    with open(out_zones, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_sec", "timestamp_iso", "zone", "source"])
        for r, z, s in zip(rows, zones, sources):
            w.writerow([r["t_sec"], r["timestamp_iso"], z if z is not None else "", s if s is not None else ""])

    with open(out_episodes, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["zone", "source", "start_sec", "end_sec", "duration_sec", "sustained"])
        for e in episodes:
            w.writerow([e["zone"], e["source"], e["start_t"], e["end_t"], e["duration"], e["duration"] >= args.sustained_seconds])

    summary_rows = [{"trial_condition": "", "trial_start": "", "trial_end": "", **session_summary}]

    if args.event_log:
        trials = load_trials(args.event_log)
        for trial in trials:
            trial_rows = rows_in_trial(rows, trial)
            if not trial_rows:
                continue
            _, _, _, trial_summary = summarize(
                trial_rows, args.hunch_threshold_deg, args.lean_threshold_deg, dist_thresh, line_thresh,
                args.gap_tolerance, args.sustained_seconds,
            )
            summary_rows.append({
                "trial_condition": trial["condition"], "trial_start": trial["start"], "trial_end": trial["end"],
                **trial_summary,
            })

    fieldnames = list(summary_rows[0].keys())
    with open(out_summary, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in summary_rows:
            w.writerow(r)

    print(f"shoulder fallback thresholds (session-relative, {args.shoulder_percentile:.0f}th pct): "
          f"dist>{dist_thresh if dist_thresh is not None else float('nan'):.1f}px "
          f"line>{line_thresh if line_thresh is not None else float('nan'):.1f}deg")
    print(f"whole session: {session_summary['pct_hunch_of_classified']:.0f}% hunch, "
          f"{session_summary['pct_lean_of_classified']:.0f}% lean, "
          f"{session_summary['pct_neutral_of_classified']:.0f}% neutral "
          f"(of classified time; {session_summary['pct_unclassified_of_wallclock']:.0f}% of wall-clock time unclassified)")
    print(f"sustained episodes: {session_summary['n_sustained_hunch_episodes']} hunch "
          f"({session_summary['total_sustained_hunch_sec']:.1f}s total), "
          f"{session_summary['n_sustained_lean_episodes']} lean "
          f"({session_summary['total_sustained_lean_sec']:.1f}s total)")
    if args.event_log:
        print(f"{len(summary_rows) - 1} trial(s) matched from {args.event_log}")
    print(f"wrote {out_zones}")
    print(f"wrote {out_episodes}")
    print(f"wrote {out_summary}")


if __name__ == "__main__":
    main()
