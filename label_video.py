#!/usr/bin/env python3
"""Extract a camera topic from a gen3_study rosbag2 and burn the wall-clock
timestamp onto every frame.

Why: the posture classifier's own extracted video has no on-screen clock, so
there's no way to look at a frame and know when it happened. This script
produces a companion video you *can* read a timestamp off of, so it can be
watched side by side with an event_log_N.csv (see
~/deidentified_study_data/fix_event_logs.py) to manually fill in the
`[MISSING - fill in from video]` rows -- or just to sanity-check any
timestamp in the log against what's actually on screen.

The burned-in timestamp is computed the same way as extract_posture.py's
`timestamp_iso` column (dt.datetime.fromtimestamp(t_ns / 1e9)), so a given
on-screen timestamp lines up with that same instant's row in the
`*_posture.csv` output -- they're reading the same clock.

Usage
-----
    source /opt/ros/kilted/setup.bash
    source ~/.posture_venv/bin/activate

    python3 label_video.py <path/to/bag_dir>
    python3 label_video.py <path/to/bag_dir> --topic /camera2/image_raw_throttled/compressed --out /tmp/task_view.mp4

With no --topic, it writes one labeled video per study camera that's actually
present in the bag (/camera and /camera2's image_raw_throttled/compressed
topics -- same defaults as extract_posture.py's --face-topic/--shoulder-topic).
A bag missing one camera (e.g. the loose-USB-cable bags noted in
posture_analysis/README.md) just gets skipped for that topic, not an error.
"""
import argparse
import datetime as dt
from pathlib import Path

import cv2

from extract_posture import open_reader, iter_images


def label_video(bag_path, topic, out_path, every=1, fps=15.0, font_scale=1.0):
    """Write an mp4 of `topic`'s frames from `bag_path`, each stamped with its
    capture time top-left. Returns the number of frames written."""
    writer = None
    n_written = 0
    for i, (t_ns, frame) in enumerate(iter_images(bag_path, topic)):
        if i % every != 0:
            continue
        h, w = frame.shape[:2]
        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

        # Same epoch conversion as extract_posture.py's timestamp_iso column,
        # so a timestamp burned in here means the same instant there.
        timestamp_iso = dt.datetime.fromtimestamp(t_ns / 1e9).isoformat(timespec="milliseconds")

        annotated = frame.copy()
        (tw, th), baseline = cv2.getTextSize(
            timestamp_iso, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2
        )
        cv2.rectangle(annotated, (5, 5), (15 + tw, 20 + th + baseline), (0, 0, 0), -1)
        cv2.putText(
            annotated, timestamp_iso, (10, 15 + th), cv2.FONT_HERSHEY_SIMPLEX,
            font_scale, (255, 255, 255), 2, cv2.LINE_AA,
        )
        writer.write(annotated)
        n_written += 1

    if writer:
        writer.release()
    return n_written


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("bag", help="path to a rosbag2 directory (containing metadata.yaml)")
    ap.add_argument(
        "--topic", action="append", dest="topics",
        help="camera topic to extract; repeatable. Default: both study cameras "
        "(/camera and /camera2's image_raw_throttled/compressed) that are present in the bag",
    )
    ap.add_argument(
        "--out", default=None,
        help="output mp4 path. Only valid with a single --topic; with the default "
        "(both cameras), use --out-dir instead",
    )
    ap.add_argument(
        "--out-dir", default=None,
        help="directory for output videos, named '<bag>_<camera>_labeled.mp4' "
        "(default: alongside the bag)",
    )
    ap.add_argument("--every", type=int, default=1, help="process every Nth frame")
    ap.add_argument("--fps", type=float, default=15.0, help="output video frame rate")
    ap.add_argument("--font-scale", type=float, default=1.0, help="timestamp text size")
    args = ap.parse_args()

    bag_path = Path(args.bag)
    if not (bag_path / "metadata.yaml").exists():
        raise SystemExit(f"{bag_path} doesn't look like a rosbag2 directory (no metadata.yaml)")

    topics = args.topics or [
        "/camera/image_raw_throttled/compressed",
        "/camera2/image_raw_throttled/compressed",
    ]
    if args.out and len(topics) != 1:
        raise SystemExit("--out requires exactly one --topic; use --out-dir for multiple topics")

    reader = open_reader(bag_path)
    available = {t.name for t in reader.get_all_topics_and_types()}

    out_dir = Path(args.out_dir) if args.out_dir else bag_path.parent

    for topic in topics:
        if topic not in available:
            print(f"skipping {topic}: not present in this bag (available: {sorted(available)})")
            continue
        if args.out:
            out_path = Path(args.out)
        else:
            camera_name = topic.strip("/").split("/")[0]
            out_path = out_dir / f"{bag_path.name}_{camera_name}_labeled.mp4"
        n = label_video(
            bag_path, topic, out_path,
            every=args.every, fps=args.fps, font_scale=args.font_scale,
        )
        print(f"{topic}: wrote {n} frames -> {out_path}")


if __name__ == "__main__":
    main()
