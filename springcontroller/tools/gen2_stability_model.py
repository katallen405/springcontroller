#!/usr/bin/env python3
"""Gen2 trajectory-spring stability model: which gains are safe on the arm.

Rigid-body dynamics of the Gen2 (pinocchio, 0.2 ms steps) with exact gravity
compensation, viscous and Coulomb joint friction, and the two things the
real loop adds that matter:
  * a 100 Hz controller whose torque is held for each 10 ms tick, and
  * DELAY: the spring sees joint states delay_ticks x 10 ms old. On the arm
    the measured joint torque follows the command by 52-64 ms (2026-11-23
    bag, cross-correlation, joints 1/3/5), and the model's oscillation
    frequency matches the observed 3.7 Hz at about the same delay.
Without the delay the model calls everything stable -- which is where the
earlier simulations were wrong.

Calibrated against the arm: 60 N/m / 5 N*s/m stable (every 2026-11 test),
300 N/m / 30 N*s/m unstable at 3.7 Hz (2026-11-23 18:58).

Usage:
    gen2_stability_model.py map    [--delay-ms 60] [--pose q1,...,q7]
    gen2_stability_model.py check  --stiffness 100 --damping 5 [--delay-ms 60]

ROS-free; uses springcontroller's own TrackingPoseSpring and arm model.
"""

from __future__ import annotations

import argparse
import sys
from collections import deque
from pathlib import Path

import numpy as np
import pinocchio as pin

PKG = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PKG))
from springcontroller.trajectory_spring import TrackingPoseSpring  # noqa: E402
from springcontroller.urdf_arm_configuration import URDFArmConfiguration  # noqa: E402

URDF = PKG / "flat_urdf_files" / "gen2_robotiq_flat.urdf"
LOCKED = ["gripper_finger1_joint", "gripper_finger2_joint",
          "gripper_finger1_inner_knuckle_joint", "gripper_finger2_inner_knuckle_joint",
          "gripper_finger1_finger_tip_joint", "gripper_finger2_finger_tip_joint"]
LINK = "j2s7s300_ee_link"
# Recorded poses (2026-11-23 18:58 bag): where it went unstable (low,
# reaching toward the table) and where that goal started.
POSES = {
    "onset": [2.5508, 1.9209, 5.1615, 3.6745, 7.7070, 1.1949, 5.9853],
    "start": [2.9416, 2.3013, 4.9869, 3.6124, 7.5372, 1.3461, 5.9852],
}
MEASURED_DELAY_MS = 60
_ARM = None


def arm():
    global _ARM
    if _ARM is None:
        _ARM = URDFArmConfiguration.from_urdf(str(URDF), locked_joint_names=LOCKED)
    return _ARM


def simulate(q0, gains, delay_ticks=6, coulomb=0.3, viscous=0.3, vel_noise=0.0,
             seconds=6.0, ctrl_dt=0.01, sim_dt=0.0002, kick=0.02, rotation_axes=None, seed=0):
    """Release the arm 2 cm (kick) from a fixed anchor; returns (swing_cm over
    the last 3 s, dominant frequency Hz). inf swing = it blew up."""
    a = arm()
    model = a._model
    data = model.createData()
    rng = np.random.default_rng(seed)
    q, v = np.array(q0, float), np.zeros(a.n_dof)
    a.update_from_angles(q, v)
    T = a.get_link_transform(LINK)
    spring = TrackingPoseSpring(LINK, [0, 0, 0], **gains)
    spring.set_anchor(T[:3, 3] + [kick, 0.0, 0.0], T[:3, :3], None, q.copy())
    spring.set_rotation_axes(np.zeros((0, 3)) if rotation_axes is None else rotation_axes)
    hist = deque([(q.copy(), v.copy())] * (delay_ticks + 1), maxlen=delay_ticks + 1)
    n_sub = int(round(ctrl_dt / sim_dt))
    xs = []
    for _ in range(int(seconds / ctrl_dt)):
        hist.append((q.copy(), v.copy()))
        q_seen, v_seen = hist[0]
        if vel_noise:
            v_seen = v_seen + rng.normal(0.0, vel_noise, a.n_dof)
        a.update_from_angles(q_seen, v_seen)
        tau = spring.compute_torques(a)
        a.update_from_angles(q, v)
        xs.append(a.get_link_transform(LINK)[:3, 3].copy())
        for _ in range(n_sub):
            qq = a._angles_to_q(q)
            g = pin.computeGeneralizedGravity(model, data, qq)
            acc = pin.aba(model, data, qq, v, tau + g - viscous * v - coulomb * np.tanh(v / 0.01))
            v = v + acc * sim_dt
            q = q + v * sim_dt
            if not np.all(np.isfinite(q)) or np.abs(v).max() > 30:
                return float("inf"), float("nan")
    xs = np.array(xs[-300:])
    d = xs - xs.mean(axis=0)
    main = d @ np.linalg.svd(d, full_matrices=False)[2][0]
    f = np.fft.rfftfreq(len(main), d=ctrl_dt)
    P = np.abs(np.fft.rfft(main - main.mean())) ** 2
    return float(np.max(np.linalg.norm(d, axis=1))) * 100, float(f[np.argmax(P[1:]) + 1])


def gains(stiffness, damping, **kw):
    g = dict(stiffness=stiffness, damping=damping, rot_stiffness=6.0, rot_damping=0.1,
             max_force=40.0, max_moment=3.0, null_damping=0.1)
    g.update(kw)
    return g


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("map", help="stable/unstable table over stiffness x damping")
    c = sub.add_parser("check", help="one gain set at every recorded pose")
    for p in (m, c):
        p.add_argument("--delay-ms", type=int, default=MEASURED_DELAY_MS)
        p.add_argument("--pose", help="7 comma-separated joint angles (default: recorded poses)")
    c.add_argument("--stiffness", type=float, required=True)
    c.add_argument("--damping", type=float, required=True)
    a = ap.parse_args(argv)
    poses = ({"given": [float(v) for v in a.pose.split(",")]} if a.pose else POSES)
    ticks = int(round(a.delay_ms / 10))
    if a.cmd == "check":
        ok = True
        for name, q in poses.items():
            swing, f = simulate(q, gains(a.stiffness, a.damping), delay_ticks=ticks)
            bad = swing > 1.0
            ok &= not bad
            print(f"{name:6} {'UNSTABLE' if bad else 'stable':8} swing {min(swing, 99):5.2f} cm"
                  + (f" at {f:.1f} Hz" if bad and np.isfinite(f) else ""))
        return 0 if ok else 1
    K, B = [30, 60, 100, 150, 200, 300], [2, 5, 10, 20, 30]
    for name, q in poses.items():
        print(f"\n{name} pose, delay {a.delay_ms} ms -- rows stiffness N/m, columns damping N*s/m")
        print("        " + "".join(f"{b:>9}" for b in B))
        for k in K:
            row = []
            for b in B:
                swing, _ = simulate(q, gains(k, b), delay_ticks=ticks)
                row.append("UNSTABLE" if swing > 1.0 else "ok")
            print(f"{k:>6}  " + "".join(f"{r:>9}" for r in row), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
