#!/usr/bin/env python3
"""
joint_kinematics.py

Per-joint acceleration and jerk (change in acceleration) of the Gen3's 7
joints during each adjustment period, computed offline from the joint
positions recorded in a study rosbag, plus how far the arm actually moved
(joint-space path length, end-effector displacement/path length) -- split
into samples where the arm is moving at all (any joint faster than
--motion-threshold, whatever the cause) vs. sitting still.

Data sources (all already recorded by gen3_spring.launch.py's rosbag):
  /kinova/joint_states_lowlevel     sensor_msgs/JointState, ~100 Hz
                                    (gen3_torque_control's feedback_rate_hz).
                                    Positions are wrapped into (-pi, pi] by
                                    the torque node, so they're unwrapped
                                    here before differentiating -- otherwise
                                    a continuous joint crossing +-pi shows
                                    up as a huge acceleration spike.
  /gen3_torque_control/move_status  std_msgs/String; "MOVING" only while a
                                    move_to_joint_angles command is driving
                                    the arm. Reported separately as
                                    commanded_move; it does NOT feed
                                    robot_moving.

End-effector position is NOT read from /gen3_torque_control/ee_pose: that
topic copies Kortex's onboard tool pose, which stays frozen during torque
control and only jumps occasionally (3 updates in ~18k samples in a test
bag). Instead it's forward kinematics of the smoothed joint angles through
--urdf (default: the study's gen3_kinova_flat.urdf, same model as
virtual_spring_node) at --ee-frame (default end_effector_link), in base
coordinates. Position only, no EE rotation metric.

robot_moving: True wherever max_j |qd_j| (the SG-smoothed joint speed)
exceeds --motion-threshold (default 0.02 rad/s). At rest the smoothed speed
sits around 1e-4 rad/s, so the default is well above the noise floor while
still catching slow hand adjustments.

Differentiation: positions are resampled onto a uniform grid at the median
sample interval, then differentiated with a Savitzky-Golay filter (deriv=1,
2, 3). Plain finite differences amplify sensor noise badly by the third
derivative. The SG window (--sg-window-ms) is the main tuning knob; run a
few values and check the summary metrics are stable before trusting them.
As a sanity check, the arm's own reported velocity is also differentiated
once, and its correlation with the position-derived acceleration is
reported per joint (acc_vel_xcheck_r) -- it should be high whenever the
joint actually moves.

Path lengths use the SG-smoothed (deriv=0) signals rather than the raw
samples, since summing |delta| over raw 100 Hz samples accumulates sensor
noise even when the arm is still.

Adjustment periods come from a --periods CSV (see load_periods(); it also
accepts the video-corrected adjustment_times.csv headers). Simple format:
    participant_id,condition,side,start,end
    test1,pose,1,2026-09-04T15:35:37.5,2026-09-04T15:36:20
start/end are local wall-clock ISO times (interpreted in --tz); condition
matches the bag folder's prefix (KT / position / pose); side is free text.

Usage:
    python3 analysis/joint_kinematics.py <bag_dir_or_participant_dir>... \\
        --periods periods.csv --out OUT_DIR [--tz America/New_York] \\
        [--pad SECONDS] [--sg-window-ms MS] [--sg-order N] \\
        [--motion-threshold RAD_PER_S] [--urdf PATH] [--ee-frame LINK]

Example:
    python3 analysis/joint_kinematics.py ~/gen3_study_data/test4 \\
        --periods test4_periods.csv --out kinematics_test4
"""

from __future__ import annotations

import argparse
import csv as csv_module
import datetime
import os
import sys
import warnings
from zoneinfo import ZoneInfo

import numpy as np
from scipy.signal import savgol_filter

JOINT_STATES_TOPIC = "/kinova/joint_states_lowlevel"
MOVE_STATUS_TOPIC = "/gen3_torque_control/move_status"
JOINT_NAMES = [f"joint_{i}" for i in range(1, 8)]
GAP_WARN_FACTOR = 5.0
DEFAULT_MOTION_THRESHOLD = 0.02   # rad/s, max over joints of |qd|
DEFAULT_URDF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                            "springcontroller", "flat_urdf_files", "gen3_kinova_flat.urdf")
DEFAULT_EE_FRAME = "end_effector_link"


# ── Bag reading ───────────────────────────────────────────────────────────────

def read_bag(bag_path: str) -> dict:
    """
    Joint states and move_status from one bag. ROS imports
    live in here so the rest of this module (and its tests) import without
    rclpy. Only the two topics above are deserialized -- the bag's audio
    and camera topics are filtered out at the reader.

    Times are header stamps where set, falling back to the bag receive
    time when a stamp is zero.
    """
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import JointState
    from std_msgs.msg import String

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=bag_path, storage_id="mcap"),
        rosbag2_py.ConverterOptions("", ""),
    )
    reader.set_filter(rosbag2_py.StorageFilter(
        topics=[JOINT_STATES_TOPIC, MOVE_STATUS_TOPIC]))

    def stamp_or(header, t_ns):
        s = header.stamp.sec + header.stamp.nanosec / 1e9
        return s if s > 0 else t_ns / 1e9

    js_t, js_q, js_qd = [], [], []
    move_status_events = []
    while reader.has_next():
        topic, data, t_ns = reader.read_next()
        if topic == JOINT_STATES_TOPIC:
            msg = deserialize_message(data, JointState)
            order = [list(msg.name).index(n) for n in JOINT_NAMES]
            js_t.append(stamp_or(msg.header, t_ns))
            js_q.append([msg.position[i] for i in order])
            js_qd.append([msg.velocity[i] for i in order] if msg.velocity
                         else [np.nan] * len(JOINT_NAMES))
        elif topic == MOVE_STATUS_TOPIC:
            msg = deserialize_message(data, String)
            move_status_events.append((t_ns / 1e9, msg.data))

    return {
        "t": np.asarray(js_t), "q": np.asarray(js_q), "qd": np.asarray(js_qd),
        "move_status_events": move_status_events,
    }


def make_ee_fk(urdf_path: str, frame: str):
    """
    Returns fk(q) -> (N, 3) positions of `frame` for joint angles q (N, 7)
    ordered as JOINT_NAMES. Pinocchio represents continuous joints as
    (cos, sin) pairs, so angles are packed per joint, not copied raw. Any
    other joints in the URDF (the locked gripper) stay at neutral.
    """
    import pinocchio as pin
    model = pin.buildModelFromUrdf(os.path.expanduser(urdf_path))
    data = model.createData()
    if not model.existFrame(frame):
        raise ValueError(f"{urdf_path} has no frame {frame!r}")
    fid = model.getFrameId(frame)
    slots = []
    for name in JOINT_NAMES:
        if not model.existJointName(name):
            raise ValueError(f"{urdf_path} has no joint {name!r}")
        j = model.joints[model.getJointId(name)]
        slots.append((j.idx_q, j.nq))

    def fk(q: np.ndarray) -> np.ndarray:
        out = np.empty((q.shape[0], 3))
        cfg = pin.neutral(model)
        for i, qi in enumerate(q):
            for (k, nq), a in zip(slots, qi):
                if nq == 2:
                    cfg[k], cfg[k + 1] = np.cos(a), np.sin(a)
                else:
                    cfg[k] = a
            pin.framesForwardKinematics(model, data, cfg)
            out[i] = data.oMf[fid].translation
        return out
    return fk


def moving_windows(move_status_events, last_t: float):
    """[(start, end), ...] spans where move_status == MOVING. Same logic as
    test/extract_kt_manual_moves.py's moving_windows()."""
    windows = []
    start = None
    for t, status in move_status_events:
        if status == "MOVING" and start is None:
            start = t
        elif status != "MOVING" and start is not None:
            windows.append((start, t))
            start = None
    if start is not None:
        windows.append((start, last_t))
    return windows


def find_bags(paths: list[str]) -> list[str]:
    """Each path is either a bag folder (has metadata.yaml) or a folder of
    them (e.g. one participant's directory)."""
    bags = []
    for p in paths:
        p = os.path.expanduser(p)
        if os.path.isfile(os.path.join(p, "metadata.yaml")):
            bags.append(p)
            continue
        for name in sorted(os.listdir(p)):
            sub = os.path.join(p, name)
            if os.path.isfile(os.path.join(sub, "metadata.yaml")):
                bags.append(sub)
    return bags


def parse_bag_name(bag_path: str) -> tuple[str, str]:
    """(participant_id, condition) from <participant_id>/<condition>_<ts>/,
    the layout gen3_spring.launch.py's rosbag routing writes."""
    bag_path = os.path.normpath(bag_path)
    participant_id = os.path.basename(os.path.dirname(bag_path))
    condition = os.path.basename(bag_path).split("_")[0]
    return participant_id, condition


# ── Periods ───────────────────────────────────────────────────────────────────

def to_epoch(iso: str, tz: str) -> float:
    """Local wall-clock ISO time -> epoch seconds. An explicit offset in
    the string wins over --tz."""
    dt = datetime.datetime.fromisoformat(iso.strip())
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=ZoneInfo(tz))
    return dt.timestamp()


def load_periods(path: str, tz: str) -> list[dict]:
    """
    Two CSV layouts are accepted (header names matched case-insensitively):
      * simple:    participant_id, condition, side, start, end
      * corrected: Participant, Condition, Side, Adjustment Start,
                   Adjustment End[, Total Adjustment Time]
                   (the video-corrected adjustment_times.csv; the total
                   column is ignored, it is recomputed from start/end)
    Zero-length (end <= start) periods are skipped with a warning: a
    participant request to freeze at once leaves nothing to analyze.
    """
    aliases = {
        "participant_id": ("participant_id", "participant"),
        "condition": ("condition",),
        "side": ("side",),
        "start": ("start", "adjustment start"),
        "end": ("end", "adjustment end"),
    }
    periods = []
    with open(os.path.expanduser(path), newline="", encoding="utf-8-sig") as f:
        reader = csv_module.DictReader(f)
        header = {(h or "").strip().lower(): h for h in (reader.fieldnames or [])}
        col = {}
        for key, names in aliases.items():
            found = next((header[n] for n in names if n in header), None)
            if found is None:
                raise KeyError(f"periods file {path} has no {names[0]!r} column "
                               f"(headers: {list(reader.fieldnames or [])})")
            col[key] = found
        for row in reader:
            period = {
                "participant_id": row[col["participant_id"]].strip(),
                "condition": row[col["condition"]].strip(),
                "side": row[col["side"]].strip(),
                "start": to_epoch(row[col["start"]], tz),
                "end": to_epoch(row[col["end"]], tz),
            }
            if period["end"] <= period["start"]:
                warnings.warn(
                    f"skipping zero-length period: {period['participant_id']} "
                    f"{period['condition']} side {period['side']} "
                    f"({row[col['start']].strip()} -> {row[col['end']].strip()})")
                continue
            periods.append(period)
    return periods


def _norm_participant(pid: str) -> str:
    """'p7' and '7' name the same participant: the corrected times file
    uses p1..p24, the bag folders 1..24."""
    pid = pid.strip().lower()
    return pid[1:] if pid[:1] == "p" and pid[1:].isdigit() else pid


def select_periods(periods: list[dict], participant_id: str, condition: str) -> list[dict]:
    return [p for p in periods
            if _norm_participant(p["participant_id"]) == _norm_participant(participant_id)
            and p["condition"].lower() == condition.lower()]


# ── Signal processing ─────────────────────────────────────────────────────────

def resample_uniform(t: np.ndarray, x: np.ndarray, dt: float | None = None):
    """
    Linear-interpolate x (N, k) onto a uniform grid at the median sample
    interval (or dt). Warns about gaps longer than GAP_WARN_FACTOR x dt,
    since interpolating across one invents a straight-line motion.
    """
    order = np.argsort(t)
    t, x = t[order], x[order]
    keep = np.concatenate(([True], np.diff(t) > 0))
    t, x = t[keep], x[keep]
    if dt is None:
        dt = float(np.median(np.diff(t)))
    gaps = np.diff(t)
    n_gaps = int(np.sum(gaps > GAP_WARN_FACTOR * dt))
    if n_gaps:
        warnings.warn(f"{n_gaps} gap(s) > {GAP_WARN_FACTOR:g}x dt "
                      f"(longest {gaps.max():.3f} s) -- interpolated across")
    grid = np.arange(t[0], t[-1] + dt / 2, dt)
    out = np.column_stack([np.interp(grid, t, x[:, j]) for j in range(x.shape[1])])
    return grid, out, dt


def sg_window_samples(window_s: float, dt: float, order: int) -> int:
    """Odd window length in samples, at least order + 2 (savgol needs
    window > order, and odd)."""
    n = int(round(window_s / dt))
    n = max(n, order + 2)
    return n if n % 2 == 1 else n + 1


def derivatives(q: np.ndarray, dt: float, window: int, order: int):
    """Smoothed position, velocity, acceleration, jerk along axis 0."""
    if q.shape[0] < window:
        raise ValueError(f"period has {q.shape[0]} samples, fewer than the "
                         f"SG window ({window}); shorten --sg-window-ms")
    return tuple(
        savgol_filter(q, window, order, deriv=d, delta=dt, axis=0, mode="interp")
        for d in range(4)
    )


def path_length(x: np.ndarray, mask: np.ndarray | None = None, per_column: bool = False):
    """
    Sum of |step| along axis 0. per_column=True gives one value per column
    (|dq_j|); otherwise the Euclidean norm of each step across columns.
    mask (N,) keeps only steps whose two endpoints are both True.
    """
    steps = np.diff(x, axis=0)
    if mask is not None:
        steps = steps[mask[1:] & mask[:-1]]
    if per_column:
        return np.sum(np.abs(steps), axis=0)
    return float(np.sum(np.linalg.norm(steps, axis=1)))


def mask_in_windows(t: np.ndarray, windows) -> np.ndarray:
    mask = np.zeros(t.shape, dtype=bool)
    for a, b in windows:
        mask |= (t >= a) & (t <= b)
    return mask


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or np.std(a[ok]) == 0 or np.std(b[ok]) == 0:
        return float("nan")
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def rms(x: np.ndarray, axis=0):
    return np.sqrt(np.mean(np.square(x), axis=axis))


# ── Per-period analysis ───────────────────────────────────────────────────────

def analyze_period(bag: dict, start: float, end: float, windows,
                   window_s: float, order: int,
                   motion_threshold: float = DEFAULT_MOTION_THRESHOLD,
                   ee_fk=None) -> dict:
    """Everything computed for one adjustment period of one bag."""
    in_period = (bag["t"] >= start) & (bag["t"] <= end)
    if in_period.sum() < 3:
        raise ValueError("fewer than 3 joint-state samples in period "
                         "(check --tz and that the period falls inside this bag)")

    t_raw = bag["t"][in_period]
    q_raw = np.unwrap(bag["q"][in_period], axis=0)
    t, q, dt = resample_uniform(t_raw, q_raw)
    _, qd_reported, _ = resample_uniform(t_raw, bag["qd"][in_period], dt)

    window = sg_window_samples(window_s, dt, order)
    q_s, qd, qdd, qddd = derivatives(q, dt, window, order)
    qdd_from_vel = savgol_filter(qd_reported, window, min(order, window - 2),
                                 deriv=1, delta=dt, axis=0, mode="interp")

    moving = np.max(np.abs(qd), axis=1) > motion_threshold
    commanded = mask_in_windows(t, windows)
    result = {
        "t": t, "q": q_s, "qd": qd, "qdd": qdd, "qddd": qddd,
        "moving": moving, "commanded": commanded, "dt": dt, "window": window,
        "motion_threshold": motion_threshold,
        "duration_s": float(t[-1] - t[0]),
        "frac_time_robot_moving": float(moving.mean()),
        "frac_time_commanded_move": float(commanded.mean()),
        "per_joint": {
            "path_length_rad": path_length(q_s, per_column=True),
            "peak_abs_acc": np.max(np.abs(qdd), axis=0),
            "rms_acc": rms(qdd),
            "peak_abs_jerk": np.max(np.abs(qddd), axis=0),
            "rms_jerk": rms(qddd),
            "acc_vel_xcheck_r": np.array([safe_corr(qdd[:, j], qdd_from_vel[:, j])
                                          for j in range(q.shape[1])]),
        },
        "joint_path_length_rad": path_length(q_s),
        "joint_path_length_moving_rad": path_length(q_s, moving),
        "joint_path_length_not_moving_rad": path_length(q_s, ~moving),
    }

    if ee_fk is not None:
        # FK of the already-smoothed joint angles, on the same grid as
        # `moving`, so no separate smoothing or resampling is needed.
        ee_s = ee_fk(q_s)
        result.update({
            "ee": ee_s,
            "ee_displacement_m": float(np.linalg.norm(ee_s[-1] - ee_s[0])),
            "ee_path_length_m": path_length(ee_s),
            "ee_path_length_moving_m": path_length(ee_s, moving),
            "ee_path_length_not_moving_m": path_length(ee_s, ~moving),
        })
    else:
        result.update({k: float("nan") for k in (
            "ee_displacement_m", "ee_path_length_m",
            "ee_path_length_moving_m", "ee_path_length_not_moving_m")})
    return result


# ── Output ────────────────────────────────────────────────────────────────────

SUMMARY_FIELDS = [
    "bag", "participant_id", "condition", "side", "joint",
    "duration_s", "frac_time_robot_moving", "frac_time_commanded_move",
    "peak_abs_acc", "rms_acc", "peak_abs_jerk", "rms_jerk",
    "path_length_rad", "acc_vel_xcheck_r",
    "joint_path_length_rad", "joint_path_length_moving_rad",
    "joint_path_length_not_moving_rad",
    "ee_displacement_m", "ee_path_length_m",
    "ee_path_length_moving_m", "ee_path_length_not_moving_m",
    "sg_window_samples", "dt_s", "motion_threshold_rad_s",
]


def summary_rows(bag_name: str, period: dict, r: dict) -> list[dict]:
    """One row per joint; per-period metrics repeated on each row."""
    rows = []
    for j, name in enumerate(JOINT_NAMES):
        row = {
            "bag": bag_name, "participant_id": period["participant_id"],
            "condition": period["condition"], "side": period["side"],
            "joint": name, "duration_s": r["duration_s"],
            "frac_time_robot_moving": r["frac_time_robot_moving"],
            "frac_time_commanded_move": r["frac_time_commanded_move"],
            "sg_window_samples": r["window"], "dt_s": r["dt"],
            "motion_threshold_rad_s": r["motion_threshold"],
        }
        for k, v in r["per_joint"].items():
            row[k] = float(v[j])
        for k in SUMMARY_FIELDS:
            if k not in row:
                row[k] = r[k]
        rows.append(row)
    return rows


def write_timeseries(path: str, r: dict) -> None:
    header = ["t", "robot_moving", "commanded_move"]
    for sym in ("q", "qd", "qdd", "qddd"):
        header += [f"{sym}_{n}" for n in JOINT_NAMES]
    ee = r.get("ee")
    if ee is not None:
        header += ["ee_x", "ee_y", "ee_z"]
    with open(path, "w", newline="") as f:
        w = csv_module.writer(f)
        w.writerow(header)
        for i in range(len(r["t"])):
            w.writerow([f"{r['t'][i]:.4f}", int(r["moving"][i]), int(r["commanded"][i])]
                       + [f"{v:.6g}" for sym in ("q", "qd", "qdd", "qddd") for v in r[sym][i]]
                       + ([f"{v:.6g}" for v in ee[i]] if ee is not None else []))


def plot_period(path: str, title: str, r: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = r["t"] - r["t"][0]
    fig, axes = plt.subplots(len(JOINT_NAMES), 2, figsize=(11, 12), sharex=True)
    edges = np.diff(np.concatenate(([0], r["moving"].astype(int), [0])))
    spans = list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1))
    for j, name in enumerate(JOINT_NAMES):
        for col, (sym, label) in enumerate((("qdd", "acc (rad/s²)"), ("qddd", "jerk (rad/s³)"))):
            ax = axes[j, col]
            for a, b in spans:
                ax.axvspan(t[a], t[b], color="0.85", lw=0)
            ax.plot(t, r[sym][:, j], lw=0.8)
            ax.set_ylabel(f"{name}\n{label}", fontsize=8)
            ax.tick_params(labelsize=7)
    axes[-1, 0].set_xlabel("time in period (s)")
    axes[-1, 1].set_xlabel("time in period (s)")
    fig.suptitle(f"{title}  (grey = arm moving, |qd| > {r['motion_threshold']:g} rad/s)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("bags", nargs="+",
                        help="Bag folder(s), or folder(s) containing bag folders.")
    parser.add_argument("--periods", required=True, help="Adjustment periods CSV.")
    parser.add_argument("--out", required=True, help="Output directory.")
    parser.add_argument("--tz", default="America/New_York",
                        help="Time zone of the periods file's wall-clock times.")
    parser.add_argument("--pad", type=float, default=0.0,
                        help="Seconds to extend each period on both ends.")
    parser.add_argument("--sg-window-ms", type=float, default=150.0,
                        help="Savitzky-Golay window (default 150 ms, ~15 samples at 100 Hz).")
    parser.add_argument("--sg-order", type=int, default=3,
                        help="Savitzky-Golay polynomial order (>= 3 so jerk is defined).")
    parser.add_argument("--urdf", default=DEFAULT_URDF,
                        help="URDF for end-effector forward kinematics (default: the study's "
                             "gen3_kinova_flat.urdf).")
    parser.add_argument("--ee-frame", default=DEFAULT_EE_FRAME,
                        help=f"URDF frame used as the end effector (default {DEFAULT_EE_FRAME}).")
    parser.add_argument("--motion-threshold", type=float, default=DEFAULT_MOTION_THRESHOLD,
                        help="robot_moving is True where any joint's smoothed speed exceeds "
                             f"this (rad/s, default {DEFAULT_MOTION_THRESHOLD:g}).")
    args = parser.parse_args()
    if args.sg_order < 3:
        parser.error("--sg-order must be >= 3 for jerk")

    periods = load_periods(args.periods, args.tz)
    ee_fk = make_ee_fk(args.urdf, args.ee_frame)
    os.makedirs(os.path.join(args.out, "timeseries"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "plots"), exist_ok=True)

    rows = []
    skipped = []   # (label, reason) for everything not analyzed, listed at the end
    for bag_path in find_bags(args.bags):
        bag_name = os.path.basename(os.path.normpath(bag_path))
        participant_id, condition = parse_bag_name(bag_path)
        mine = select_periods(periods, participant_id, condition)
        if not mine:
            print(f"{participant_id}/{bag_name}: no periods in --periods file, skipped")
            skipped.append((f"{participant_id}/{bag_name}", "no periods in --periods file"))
            continue
        try:
            bag = read_bag(bag_path)
        except Exception as e:  # unreadable bag (missing .mcap, bad topic...): keep going
            reason = f"could not read bag ({type(e).__name__}: {e})"
            print(f"{participant_id}/{bag_name}: {reason}, skipped")
            skipped.append((f"{participant_id}/{bag_name}", reason))
            continue
        if bag["t"].size == 0:
            print(f"{participant_id}/{bag_name}: no {JOINT_STATES_TOPIC} messages, skipped")
            skipped.append((f"{participant_id}/{bag_name}", f"no {JOINT_STATES_TOPIC} messages"))
            continue
        windows = moving_windows(bag["move_status_events"], bag["t"][-1])
        for p in mine:
            start, end = p["start"] - args.pad, p["end"] + args.pad
            label = f"{participant_id}_{bag_name}_side{p['side']}"
            try:
                r = analyze_period(bag, start, end, windows,
                                   args.sg_window_ms / 1000.0, args.sg_order,
                                   args.motion_threshold, ee_fk)
            except ValueError as e:
                print(f"{label}: {e}, skipped")
                skipped.append((label, str(e)))
                continue
            rows += summary_rows(bag_name, p, r)
            write_timeseries(os.path.join(args.out, "timeseries", f"{label}.csv"), r)
            plot_period(os.path.join(args.out, "plots", f"{label}.png"), label, r)
            print(f"{label}: {r['duration_s']:.1f} s, "
                  f"joint path {r['joint_path_length_rad']:.3f} rad "
                  f"({r['joint_path_length_moving_rad']:.3f} while moving, "
                  f"{100 * r['frac_time_robot_moving']:.0f}% of time moving), "
                  f"EE disp {r['ee_displacement_m']:.3f} m, "
                  f"min acc x-check r {np.nanmin(r['per_joint']['acc_vel_xcheck_r']):.2f}")

    summary_path = os.path.join(args.out, "summary.csv")
    with open(summary_path, "w", newline="") as f:
        w = csv_module.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {summary_path}")
    if skipped:
        print(f"\n{len(skipped)} bag(s)/period(s) NOT analyzed:")
        for label, reason in skipped:
            print(f"  {label}: {reason}")
    return 0 if rows else 1


if __name__ == "__main__":
    sys.exit(main())
