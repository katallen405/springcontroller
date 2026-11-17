#!/usr/bin/env python3
"""Plot posture metrics extracted by extract_posture.py.

Produces a time-series PNG with:
  - shoulder_line_dev_deg (lateral-lean proxy, continuous)
  - shoulder_dist_dev_px  (forward-hunch proxy, continuous)
  - head_pitch_dev_deg / head_roll_dev_deg (sparse, only when a face was detected)

All series are already baseline-relative (see extract_posture.py docstring).

Pass --episodes-csv (the *_episodes.csv from classify_ergonomics.py) to
shade hunch/lean episodes across all three panels, so a flagged episode can
be visually cross-checked against --out-face-video/--out-video before
trusting it.
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ZONE_COLORS = {
    "hunch": ("tab:red", 0.12),
    "lean": ("tab:purple", 0.12),
    "hunch+lean": ("tab:brown", 0.15),
}


def to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def load_episodes(path):
    return list(csv.DictReader(open(path)))


def shade_episodes(ax, episodes):
    for ep in episodes:
        color, alpha = ZONE_COLORS.get(ep["zone"], (None, 0))
        if color is None:
            continue
        if ep["sustained"] == "True":
            alpha += 0.15
        ax.axvspan(float(ep["start_sec"]), float(ep["end_sec"]), color=color, alpha=alpha, linewidth=0)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv_path")
    ap.add_argument("--out", default=None)
    ap.add_argument("--episodes-csv", default=None, help="*_episodes.csv from classify_ergonomics.py, to shade hunch/lean zones")
    args = ap.parse_args()

    csv_path = Path(args.csv_path)
    out_path = Path(args.out) if args.out else csv_path.with_suffix(".png")

    rows = list(csv.DictReader(open(csv_path)))
    t = [to_float(r["t_sec"]) for r in rows]

    shoulder_lean = [to_float(r["shoulder_line_dev_deg"]) if r["shoulders_detected"] == "True" else float("nan") for r in rows]
    shoulder_dist = [to_float(r["shoulder_dist_dev_px"]) if r["shoulders_detected"] == "True" else float("nan") for r in rows]

    face_t = [tt for tt, r in zip(t, rows) if r["face_detected"] == "True"]
    face_pitch = [to_float(r["head_pitch_dev_deg"]) for r in rows if r["face_detected"] == "True"]
    face_roll = [to_float(r["head_roll_dev_deg"]) for r in rows if r["face_detected"] == "True"]

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    episodes = load_episodes(args.episodes_csv) if args.episodes_csv else []
    if episodes:
        for ax in axes:
            shade_episodes(ax, episodes)

    axes[0].plot(t, shoulder_lean, color="tab:blue")
    axes[0].axhline(0, color="gray", linewidth=0.8)
    axes[0].set_ylabel("shoulder lean\ndev (deg)")
    axes[0].set_title("Lateral lean proxy: shoulder-line tilt vs session baseline")

    axes[1].plot(t, shoulder_dist, color="tab:orange")
    axes[1].axhline(0, color="gray", linewidth=0.8)
    axes[1].set_ylabel("shoulder dist\ndev (px)")
    axes[1].set_title("Forward-hunch proxy: shoulder pixel spacing vs session baseline\n(under this camera's overhead mount, larger = leaning closer/lower)")

    axes[2].scatter(face_t, face_pitch, label="head pitch dev (+ = looking/leaning down)", color="tab:red", s=15)
    axes[2].scatter(face_t, face_roll, label="head roll dev (head tilt to a side)", color="tab:green", s=15)
    axes[2].axhline(0, color="gray", linewidth=0.8)
    axes[2].set_ylabel("head pose\ndev (deg)")
    axes[2].set_title("Face head-pose (sparse -- only frames where a face was actually detected)")
    axes[2].legend(loc="upper right", fontsize=8)
    axes[2].set_xlabel("time (s)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    print(f"wrote {out_path}")

    n = len(rows)
    n_face = sum(1 for r in rows if r["face_detected"] == "True")
    n_sh = sum(1 for r in rows if r["shoulders_detected"] == "True")
    print(f"{n} frames | face detected {n_face} ({100*n_face/n:.0f}%) | shoulders detected {n_sh} ({100*n_sh/n:.0f}%)")


if __name__ == "__main__":
    main()
