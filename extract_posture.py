#!/usr/bin/env python3
"""Extract posture (hunch / lateral lean) metrics from a gen3_study rosbag2,
using MediaPipe against both cameras.

This is upper-body only. The desk permanently occludes the participant's
hips/legs/feet in this camera view. When you run MediaPipe's full-body Pose
Landmarker on this view it still reports hip/knee/ankle landmarks with high
confidence -- it hallucinates a plausible lower body onto the desk/robot
because the model assumes a full person is visible. Those numbers are not
physically real; this script never uses hip/leg landmarks. See
sample_frame_annotated.jpg for an example (hips landed on the desk/robot arm,
nowhere near the person).

Two cameras, two jobs
----------------------
The two rig cameras look at the participant from very different angles, and
that turned out to matter more than either camera's height above the table:

- `/camera` (default --face-topic): mounted across the table. Its face
  detection rate, measured while the participant was actually present, was
  62% -- good enough to use as a real continuous head-pose signal. This
  script uses it for head_pitch_deg/head_roll_deg (via MediaPipe's Face
  Landmarker transformation matrix): pitch = looking/leaning down (hunch),
  roll = head tilted toward a shoulder (lateral lean).
- `/camera2` (default --shoulder-topic): mounted close and nearly overhead
  the workspace -- great for watching hands/task, but face detection there
  was only 13% under the same test (the steep angle mostly shows the crown
  of the head, not the face). This script uses it only for the shoulders
  (visible ~any time the participant is at the desk):
    - shoulder_line_deg: 2D tilt of the shoulder line from horizontal
      (secondary lateral-lean cross-check)
    - shoulder_dist_px: pixel distance between the shoulders (secondary
      forward-hunch cross-check -- increases as the torso comes closer
      to/lower under this camera)

Frames from the two cameras aren't synchronized, so each camera2 frame is
paired with the nearest-in-time camera1 result (within --max-pair-gap
seconds; unpaired frames get NaN face columns).

Because each camera's own mounting angle/position is fixed but unknown to
this script, absolute values are not physically meaningful on their own.
Instead it calibrates a per-recording, per-signal baseline from the first
few seconds after that signal is first detected (assumed roughly neutral
posture) and reports deviation from that baseline. Always sanity-check
against the annotated overlay video (--out-video) before trusting numbers --
these are proxies, not calibrated biomechanical angles.

If a bag only has one camera recording (check with `ros2 bag info`), the
other signal's columns are simply all NaN/not-detected -- this has happened
before due to a loose USB extension cable on the rig, not a code bug (see
project memory / README).
"""
import argparse
import bisect
import csv
import datetime as dt
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

import rosbag2_py
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import CompressedImage

MODELS_DIR = Path.home() / ".posture_venv" / "models"
FACE_MODEL = MODELS_DIR / "face_landmarker.task"
POSE_MODEL = MODELS_DIR / "pose_landmarker_full.task"

L_SHOULDER, R_SHOULDER = 11, 12


def open_reader(bag_path):
    storage_options = rosbag2_py.StorageOptions(uri=str(bag_path), storage_id="mcap")
    converter_options = rosbag2_py.ConverterOptions("", "")
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)
    return reader


def iter_images(bag_path, topic):
    reader = open_reader(bag_path)
    topic_types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    if topic not in topic_types:
        raise SystemExit(f"topic {topic} not found in bag; available: {sorted(topic_types)}")
    while reader.has_next():
        name, data, t_ns = reader.read_next()
        if name != topic:
            continue
        msg = deserialize_message(data, CompressedImage)
        frame = cv2.imdecode(np.frombuffer(msg.data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            continue
        yield t_ns, frame


def rotation_to_euler_deg(R):
    """pitch = tilt forward/down (+) or back (-); yaw = turn left/right;
    roll = tilt head toward a shoulder. Matches MediaPipe's face transform
    matrix convention (camera-facing, +x right, +y up, +z toward viewer)."""
    pitch = np.degrees(np.arctan2(-R[2, 1], R[2, 2]))
    yaw = np.degrees(np.arcsin(np.clip(R[2, 0], -1.0, 1.0)))
    roll = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    return pitch, yaw, roll


def shoulder_line_angle_deg(l_sh_px, r_sh_px):
    """Angle (deg) of the left->right shoulder line from horizontal. 0 = level."""
    dx, dy = r_sh_px[0] - l_sh_px[0], r_sh_px[1] - l_sh_px[1]
    return np.degrees(np.arctan2(dy, dx))


def shoulder_distance_px(l_sh_px, r_sh_px):
    return float(np.linalg.norm(r_sh_px - l_sh_px))


def angle_diff_deg(a, b):
    """Signed difference a-b in degrees, wrapped to [-180, 180]. Needed because
    shoulder-line angles can sit near the +-180 boundary depending on camera
    orientation, where naive subtraction produces spurious ~360deg jumps."""
    if np.isnan(a) or np.isnan(b):
        return float("nan")
    return (a - b + 180.0) % 360.0 - 180.0


def make_face_landmarker():
    opts = mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(FACE_MODEL)),
        running_mode=mp_vision.RunningMode.VIDEO,
        output_facial_transformation_matrixes=True,
        num_faces=1,
    )
    return mp_vision.FaceLandmarker.create_from_options(opts)


def make_pose_landmarker():
    opts = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(POSE_MODEL)),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=1,
    )
    return mp_vision.PoseLandmarker.create_from_options(opts)


def extract_face_rows(bag_path, topic, every, t0_ns):
    """Full pass over the face camera: returns a list of dicts sorted by
    t_ns, plus the (possibly updated) shared t0_ns."""
    face_lm = make_face_landmarker()
    rows = []
    last_t_ms = -1
    for i, (t_ns, frame) in enumerate(iter_images(bag_path, topic)):
        if i % every != 0:
            continue
        if t0_ns is None or t_ns < t0_ns:
            t0_ns = t_ns
        t_ms = t_ns // 1_000_000
        if t_ms <= last_t_ms:
            t_ms = last_t_ms + 1
        last_t_ms = t_ms

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = face_lm.detect_for_video(mp_image, t_ms)

        row = {"t_ns": t_ns, "frame_idx": i, "face_detected": False}
        if result.facial_transformation_matrixes:
            M = np.array(result.facial_transformation_matrixes[0])
            pitch, yaw, roll = rotation_to_euler_deg(M[:3, :3])
            row["face_detected"] = True
            row["head_pitch_deg"] = pitch
            row["head_yaw_deg"] = yaw
            row["head_roll_deg"] = roll
        rows.append(row)
    face_lm.close()
    return rows, t0_ns


def nearest_face_row(face_rows, face_t_ns_sorted, t_ns, max_gap_ns):
    if not face_rows:
        return None
    idx = bisect.bisect_left(face_t_ns_sorted, t_ns)
    candidates = [c for c in (idx - 1, idx) if 0 <= c < len(face_rows)]
    if not candidates:
        return None
    best = min(candidates, key=lambda c: abs(face_t_ns_sorted[c] - t_ns))
    if abs(face_t_ns_sorted[best] - t_ns) > max_gap_ns:
        return None
    return face_rows[best]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("bag", help="path to a rosbag2 directory (containing metadata.yaml)")
    ap.add_argument(
        "--face-topic", default="/camera/image_raw_throttled/compressed",
        help="across-table camera used for head pitch/roll (primary hunch/lean signal)",
    )
    ap.add_argument(
        "--shoulder-topic", default="/camera2/image_raw_throttled/compressed",
        help="overhead task camera used for shoulder-based proxies (secondary cross-check)",
    )
    ap.add_argument("--out-csv", default=None)
    ap.add_argument(
        "--out-video", default=None,
        help="optional annotated overlay video (mp4), built from the shoulder-topic frames",
    )
    ap.add_argument("--every", type=int, default=1, help="process every Nth frame of each camera")
    ap.add_argument(
        "--max-pair-gap", type=float, default=0.2,
        help="max seconds apart when pairing a shoulder-topic frame with a face-topic result",
    )
    ap.add_argument(
        "--baseline-seconds",
        type=float,
        default=3.0,
        help="seconds after each signal is first detected, assumed to be neutral posture, "
        "used to calibrate out each camera's fixed mounting tilt",
    )
    args = ap.parse_args()

    bag_path = Path(args.bag)
    out_csv = Path(args.out_csv) if args.out_csv else bag_path.parent / f"{bag_path.name}_posture.csv"
    max_gap_ns = int(args.max_pair_gap * 1e9)

    # Pass 1: face camera, fully processed up front so shoulder-frame pairing
    # below is a simple nearest-timestamp lookup.
    face_rows, t0_ns = extract_face_rows(bag_path, args.face_topic, args.every, None)
    face_t_ns_sorted = [r["t_ns"] for r in face_rows]

    # Pass 2: shoulder camera, streamed (so --out-video can be written
    # incrementally without holding every frame in memory).
    pose_lm = make_pose_landmarker()
    writer = None
    rows = []
    last_t_ms = -1
    n_seen = n_shoulders = 0

    for i, (t_ns, frame) in enumerate(iter_images(bag_path, args.shoulder_topic)):
        if i % args.every != 0:
            continue
        if t0_ns is None or t_ns < t0_ns:
            t0_ns = t_ns
        n_seen += 1
        h, w = frame.shape[:2]
        t_ms = t_ns // 1_000_000
        if t_ms <= last_t_ms:
            t_ms = last_t_ms + 1
        last_t_ms = t_ms

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        pose_result = pose_lm.detect_for_video(mp_image, t_ms)

        row = {
            "t_ns": t_ns,
            "frame_idx": i,
            "shoulders_detected": False,
            "face_detected": False,
        }

        face_row = nearest_face_row(face_rows, face_t_ns_sorted, t_ns, max_gap_ns)
        if face_row is not None:
            row["face_dt_sec"] = (face_row["t_ns"] - t_ns) / 1e9
            row["face_detected"] = face_row["face_detected"]
            if face_row["face_detected"]:
                row["head_pitch_deg"] = face_row["head_pitch_deg"]
                row["head_yaw_deg"] = face_row["head_yaw_deg"]
                row["head_roll_deg"] = face_row["head_roll_deg"]

        if pose_result.pose_landmarks:
            lm = pose_result.pose_landmarks[0]
            l_sh = np.array([lm[L_SHOULDER].x * w, lm[L_SHOULDER].y * h])
            r_sh = np.array([lm[R_SHOULDER].x * w, lm[R_SHOULDER].y * h])
            # Only trust shoulders (visible in this camera view). Do NOT use
            # hip/knee/ankle landmarks from this model -- see module docstring.
            n_shoulders += 1
            row["shoulders_detected"] = True
            row["shoulder_line_deg"] = shoulder_line_angle_deg(l_sh, r_sh)
            row["shoulder_dist_px"] = shoulder_distance_px(l_sh, r_sh)

        rows.append(row)

        if args.out_video:
            if writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(args.out_video, fourcc, 15, (w, h))
            annotated = frame.copy()
            if pose_result.pose_landmarks:
                lm = pose_result.pose_landmarks[0]
                for idx in (L_SHOULDER, R_SHOULDER):
                    cx, cy = int(lm[idx].x * w), int(lm[idx].y * h)
                    cv2.circle(annotated, (cx, cy), 5, (0, 255, 0), -1)
            text = (
                f"pitch={row.get('head_pitch_deg', float('nan')):.1f} "
                f"roll={row.get('head_roll_deg', float('nan')):.1f} "
                f"shoulder_ang={row.get('shoulder_line_deg', float('nan')):.1f} "
                f"shoulder_dist={row.get('shoulder_dist_px', float('nan')):.0f}"
            )
            cv2.putText(annotated, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            writer.write(annotated)

    if writer:
        writer.release()
    pose_lm.close()

    for r in rows:
        r["t_sec"] = (r["t_ns"] - t0_ns) / 1e9
        r["timestamp_iso"] = dt.datetime.fromtimestamp(r["t_ns"] / 1e9).isoformat()

    n_face_paired = sum(1 for r in rows if r["face_detected"])

    # --- baseline calibration ---
    # Anchor each signal's baseline window to when it was first detected, not
    # t=0, since the participant may not be seated at the desk yet when
    # recording starts.
    face_times = [r["t_sec"] for r in rows if r["face_detected"]]
    shoulder_times = [r["t_sec"] for r in rows if r["shoulders_detected"]]
    face_t0 = min(face_times) if face_times else None
    shoulder_t0 = min(shoulder_times) if shoulder_times else None

    baseline_rows = [
        r for r in rows
        if face_t0 is not None and face_t0 <= r["t_sec"] <= face_t0 + args.baseline_seconds
        and r["face_detected"]
    ]
    if baseline_rows:
        base_pitch = float(np.median([r["head_pitch_deg"] for r in baseline_rows]))
        base_roll = float(np.median([r["head_roll_deg"] for r in baseline_rows]))
    else:
        base_pitch = base_roll = float("nan")

    baseline_shoulder_rows = [
        r for r in rows
        if shoulder_t0 is not None and shoulder_t0 <= r["t_sec"] <= shoulder_t0 + args.baseline_seconds
        and r["shoulders_detected"]
    ]
    base_shoulder = (
        float(np.median([r["shoulder_line_deg"] for r in baseline_shoulder_rows]))
        if baseline_shoulder_rows
        else float("nan")
    )
    base_shoulder_dist = (
        float(np.median([r["shoulder_dist_px"] for r in baseline_shoulder_rows]))
        if baseline_shoulder_rows
        else float("nan")
    )

    for r in rows:
        r["head_pitch_dev_deg"] = angle_diff_deg(r.get("head_pitch_deg", float("nan")), base_pitch)
        r["head_roll_dev_deg"] = angle_diff_deg(r.get("head_roll_deg", float("nan")), base_roll)
        r["shoulder_line_dev_deg"] = angle_diff_deg(r.get("shoulder_line_deg", float("nan")), base_shoulder)
        r["shoulder_dist_dev_px"] = r.get("shoulder_dist_px", float("nan")) - base_shoulder_dist

    fieldnames = [
        "t_ns", "t_sec", "timestamp_iso", "frame_idx",
        "face_detected", "face_dt_sec", "head_pitch_deg", "head_yaw_deg", "head_roll_deg",
        "head_pitch_dev_deg", "head_roll_dev_deg",
        "shoulders_detected", "shoulder_line_deg", "shoulder_line_dev_deg",
        "shoulder_dist_px", "shoulder_dist_dev_px",
    ]
    with open(out_csv, "w", newline="") as f:
        w_csv = csv.DictWriter(f, fieldnames=fieldnames)
        w_csv.writeheader()
        for row in rows:
            w_csv.writerow(row)

    n_face_raw = sum(1 for r in face_rows if r["face_detected"])
    print(f"face camera ({args.face_topic}): {len(face_rows)} frames, "
          f"{n_face_raw} with a detected face ({100*n_face_raw/max(len(face_rows),1):.0f}%)")
    print(f"shoulder camera ({args.shoulder_topic}): {n_seen} frames, "
          f"{n_shoulders} with detected shoulders ({100*n_shoulders/max(n_seen,1):.0f}%), "
          f"{n_face_paired} paired with a detected face ({100*n_face_paired/max(n_seen,1):.0f}%)")
    print(f"baseline (first {args.baseline_seconds:.1f}s after each signal starts): "
          f"pitch={base_pitch:.1f} roll={base_roll:.1f} "
          f"shoulder_ang={base_shoulder:.1f} shoulder_dist={base_shoulder_dist:.1f}")
    print(f"wrote {out_csv}")
    if args.out_video:
        print(f"wrote {args.out_video}")


if __name__ == "__main__":
    main()
