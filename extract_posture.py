#!/usr/bin/env python3
"""Extract posture (hunch / lateral lean) metrics from the camera2 video topic
inside a gen3_study rosbag2, using MediaPipe.

This is upper-body only. The desk permanently occludes the participant's
hips/legs/feet in this camera view. When you run MediaPipe's full-body Pose
Landmarker on this view it still reports hip/knee/ankle landmarks with high
confidence -- it hallucinates a plausible lower body onto the desk/robot
because the model assumes a full person is visible. Those numbers are not
physically real; this script never uses hip/leg landmarks. See
sample_frame_annotated.jpg for an example (hips landed on the desk/robot arm,
nowhere near the person).

The camera is also mounted steep/overhead, looking down over the
participant's shoulder. Testing MediaPipe's Face Landmarker against frames
where a human can clearly see the participant's face still gave 0/5
detections (even at a relaxed confidence threshold) -- the face detector is
trained on roughly frontal/eye-level faces and doesn't recognize one
foreshortened this steeply from above. So face-based head pitch/roll is only
available in the rare moments the participant looks up toward the camera
(observed ~2% of frames in a 44s test clip); treat it as a bonus/spot-check
signal, not a continuous one. If you reposition the camera to the side, at
roughly seated head/shoulder height, 3/4 or profile view (rather than
overhead), face detection -- and therefore a real forward-flexion angle --
should become far more reliable.

Given that, the primary continuous signal here comes from the shoulders
(reliably visible ~100% of the time the participant is at the desk):
  - shoulder_line_deg: 2D tilt of the shoulder line from horizontal
    (lateral-lean proxy)
  - shoulder_dist_px: pixel distance between the shoulders (increases as the
    torso comes closer to/lower under the camera -- a rough forward-lean/
    hunch proxy under this specific overhead mount)
  - head_pitch_deg / head_roll_deg: from the Face Landmarker, only populated
    on frames where a face is actually detected

Because the camera's own mounting angle/position is fixed but unknown to this
script, absolute values are not physically meaningful on their own. Instead
it calibrates a per-recording baseline from the first few seconds (assumed
roughly neutral posture) and reports deviation from that baseline. Always
sanity-check against the annotated overlay video (--out-video) before
trusting numbers -- these are proxies, not calibrated biomechanical angles.
"""
import argparse
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
    these shoulder-line angles sit near the +-180 boundary under this camera's
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


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("bag", help="path to a rosbag2 directory (containing metadata.yaml)")
    ap.add_argument("--topic", default="/camera2/image_raw_throttled/compressed")
    ap.add_argument("--out-csv", default=None)
    ap.add_argument("--out-video", default=None, help="optional annotated overlay video (mp4)")
    ap.add_argument("--every", type=int, default=1, help="process every Nth frame")
    ap.add_argument(
        "--baseline-seconds",
        type=float,
        default=3.0,
        help="seconds at the start of the recording assumed to be neutral posture, "
        "used to calibrate out the camera's fixed mounting tilt",
    )
    args = ap.parse_args()

    bag_path = Path(args.bag)
    out_csv = Path(args.out_csv) if args.out_csv else bag_path.parent / f"{bag_path.name}_posture.csv"

    face_lm = make_face_landmarker()
    pose_lm = make_pose_landmarker()

    writer = None
    rows = []
    last_t_ms = -1
    n_seen = n_face = n_shoulders = 0
    t0_ns = None

    for i, (t_ns, frame) in enumerate(iter_images(bag_path, args.topic)):
        if i % args.every != 0:
            continue
        if t0_ns is None:
            t0_ns = t_ns
        n_seen += 1
        h, w = frame.shape[:2]
        t_ms = t_ns // 1_000_000
        if t_ms <= last_t_ms:
            t_ms = last_t_ms + 1
        last_t_ms = t_ms

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        face_result = face_lm.detect_for_video(mp_image, t_ms)
        pose_result = pose_lm.detect_for_video(mp_image, t_ms)

        row = {
            "t_ns": t_ns,
            "t_sec": (t_ns - t0_ns) / 1e9,
            "timestamp_iso": dt.datetime.fromtimestamp(t_ns / 1e9).isoformat(),
            "frame_idx": i,
            "face_detected": False,
            "shoulders_detected": False,
        }

        if face_result.facial_transformation_matrixes:
            n_face += 1
            M = np.array(face_result.facial_transformation_matrixes[0])
            pitch, yaw, roll = rotation_to_euler_deg(M[:3, :3])
            row["face_detected"] = True
            row["head_pitch_deg"] = pitch
            row["head_yaw_deg"] = yaw
            row["head_roll_deg"] = roll

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
            if face_result.face_landmarks:
                for p in face_result.face_landmarks[0][::5]:
                    cv2.circle(annotated, (int(p.x * w), int(p.y * h)), 1, (255, 200, 0), -1)
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
    face_lm.close()
    pose_lm.close()

    # --- baseline calibration ---
    # Anchor the baseline window to the first *detected* frame, not t=0, since
    # the participant may not be seated at the desk yet when recording starts.
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
        "face_detected", "head_pitch_deg", "head_yaw_deg", "head_roll_deg",
        "head_pitch_dev_deg", "head_roll_dev_deg",
        "shoulders_detected", "shoulder_line_deg", "shoulder_line_dev_deg",
        "shoulder_dist_px", "shoulder_dist_dev_px",
    ]
    with open(out_csv, "w", newline="") as f:
        w_csv = csv.DictWriter(f, fieldnames=fieldnames)
        w_csv.writeheader()
        for row in rows:
            w_csv.writerow(row)

    print(f"frames read: {n_seen}")
    print(f"  face detected:      {n_face} ({100*n_face/max(n_seen,1):.0f}%)")
    print(f"  shoulders detected: {n_shoulders} ({100*n_shoulders/max(n_seen,1):.0f}%)")
    print(f"baseline (first {args.baseline_seconds:.1f}s): pitch={base_pitch:.1f} "
          f"roll={base_roll:.1f} shoulder_ang={base_shoulder:.1f} "
          f"shoulder_dist={base_shoulder_dist:.1f}")
    print(f"wrote {out_csv}")
    if args.out_video:
        print(f"wrote {args.out_video}")


if __name__ == "__main__":
    main()
