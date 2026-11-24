#!/usr/bin/env python3
"""Step-response test of the Gen2 trajectory spring, walking up a gain ladder.

Usage (robot launched with gen2_spring.launch.py start_bridge:=true, torque
enabled, arm in an open pose clear of the table, hand on the E-STOP):
    ~/.springcontroller_venv/bin/python springcontroller/tools/stability_step_test.py
    ... --ladder 60:5,80:5,100:5 --step 0.015 --axes xyz
    ... --no-prompt                 # don't wait for Enter between levels

For each stiffness:damping level (default 60:5, 80:5, 100:5, 100:10, 150:5):
  1. checks it with the Gen2 stability model (tools/gen2_stability_model.py)
     at the arm's current pose and skips levels predicted unstable (unless
     --allow-predicted-unstable; the watchdog still guards them);
  2. sets the gains live (trajectory_spring.stiffness / damping);
  3. holds the current gripper point, then for each axis steps the anchor
     --step (1.5 cm) out in 0.3 s, watches 2.5 s, steps back, watches;
  4. from the TF gripper position at 100 Hz: overshoot (% of the step),
     settling time (within 2 mm), ringing left in the last second (mm RMS)
     and its frequency.
The ladder stops at the first oscillation-watchdog trip or aborted goal.
Gains are restored to what they were at the start, also on Ctrl-C.
Results print as a table and are saved as JSON (--out, default
~/gen2_bags/step_test_<time>.json), to compare with the model
(docs/gen2_stability_analysis.md).

Plain-Python helpers at the top are tested in test/test_stability_step_test.py.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

MIN_JERK_PEAK = 1.875


# ── plain-Python helpers ──────────────────────────────────────────────────────

def parse_ladder(text: str) -> list[tuple[float, float]]:
    levels = []
    for pair in text.split(","):
        k, b = (float(v) for v in pair.split(":"))
        if k <= 0 or b < 0:
            raise ValueError(f"bad level {pair!r}: stiffness > 0, damping >= 0")
        levels.append((k, b))
    return levels


def step_moves(start, step: float, axes: str, duration: float = 0.3) -> list[dict]:
    """Out-and-back steps along each axis from the gripper point `start`."""
    start = np.asarray(start, dtype=float)
    moves = []
    for ax in axes:
        e = np.zeros(3)
        e["xyz".index(ax)] = 1.0
        moves.append({"label": f"+{ax}", "target": (start + step * e).tolist(),
                      "from": start.tolist(), "duration": duration})
        moves.append({"label": f"+{ax} back", "target": start.tolist(),
                      "from": (start + step * e).tolist(), "duration": duration})
    return moves


def peak_speed(move: dict) -> float:
    d = np.linalg.norm(np.subtract(move["target"], move["from"]))
    return MIN_JERK_PEAK * d / move["duration"]


def analyse_step(t, pos, start, target, t_move_end: float, settle_mm: float = 2.0) -> dict:
    """Step response along the step direction, from gripper samples (t, xyz).

    overshoot_pct: furthest travel past the target, % of the step.
    settle_s: time after the move ended until it stays within settle_mm.
    ring_mm: RMS distance from target over the last second.
    ring_hz: dominant frequency of that last-second motion along the step."""
    t = np.asarray(t, float)
    pos = np.asarray(pos, float)
    start, target = np.asarray(start, float), np.asarray(target, float)
    d = target - start
    size = float(np.linalg.norm(d))
    u = d / size
    along = (pos - start) @ u                     # 0 at start, size at target
    after = t >= t_move_end
    overshoot = max(0.0, float(np.max(along[after]) - size)) / size * 100 if after.any() else 0.0
    err = np.linalg.norm(pos - target, axis=1) * 1000
    settle = None
    for i in np.where(after)[0]:
        if np.all(err[i:] <= settle_mm):
            settle = float(t[i] - t_move_end)
            break
    last = t >= t[-1] - 1.0
    ring = float(np.sqrt(np.mean(err[last] ** 2))) if last.any() else float("nan")
    x = along[last] - along[last].mean()
    hz = 0.0
    if len(x) > 8 and np.ptp(x) > 1e-5:
        dt = float(np.median(np.diff(t[last])))
        f = np.fft.rfftfreq(len(x), d=dt)
        hz = float(f[np.argmax(np.abs(np.fft.rfft(x))[1:]) + 1])
    return {"overshoot_pct": round(overshoot, 1), "settle_s": None if settle is None
            else round(settle, 2), "ring_mm": round(ring, 2), "ring_hz": round(hz, 1)}


# ── ROS runner ────────────────────────────────────────────────────────────────

class Stop(Exception):
    pass


def run(args) -> int:
    import rclpy
    from builtin_interfaces.msg import Duration
    from geometry_msgs.msg import Point
    from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
    from rcl_interfaces.srv import SetParameters
    from rclpy.action import ActionClient
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.node import Node
    from rclpy.time import Time
    from sensor_msgs.msg import JointState
    from std_msgs.msg import String
    from tf2_ros import Buffer, TransformListener
    from springcontroller_interfaces.action import FollowAnchorPath

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import gen2_stability_model as model                                  # noqa: E402

    class Runner(Node):
        def __init__(self):
            super().__init__("stability_step_test")
            self.tf = Buffer()
            self._tfl = TransformListener(self.tf, self)
            self.params = self.create_client(SetParameters, "/virtual_spring_node/set_parameters")
            self.action = ActionClient(self, FollowAnchorPath,
                                       "/spring_trajectory_controller/follow_anchor_path")
            self.state, self.q, self.torque, self.goal = None, None, None, None
            self.create_subscription(String, "/virtual_spring_node/trajectory/state",
                                     lambda m: setattr(self, "state", json.loads(m.data)), 10)
            self.create_subscription(JointState, args.joint_states,
                                     lambda m: setattr(self, "q", m), 10)
            self.create_subscription(String, "/gen2_torque_control/status",
                                     lambda m: setattr(self, "torque", m.data), 10)

        def log(self, text):
            print(f"[step_test] {text}", flush=True)
            self.get_logger().info(f"step_test: {text}")

        def wait(self, fut, timeout, what):
            end = time.monotonic() + timeout
            while not fut.done():
                if time.monotonic() > end:
                    raise Stop(f"No answer from {what} within {timeout:.0f} s.")
                time.sleep(0.01)
            return fut.result()

        def set_gains(self, k, b):
            if not self.params.wait_for_service(timeout_sec=3.0):
                raise Stop("virtual_spring_node parameters not reachable.")
            req = SetParameters.Request()
            for name, v in (("trajectory_spring.stiffness", k), ("trajectory_spring.damping", b)):
                req.parameters.append(Parameter(name=name, value=ParameterValue(
                    type=ParameterType.PARAMETER_DOUBLE, double_value=float(v))))
            res = self.wait(self.params.call_async(req), 3.0, "set_parameters")
            bad = [r.reason for r in res.results if not r.successful]
            if bad:
                raise Stop(f"Gains refused: {bad}")
            time.sleep(0.2)

        def gripper(self, at=None):
            tf = self.tf.lookup_transform(args.base_frame, args.link, at or Time())
            v = tf.transform.translation
            return np.array([v.x, v.y, v.z])

        def trips(self):
            return ((self.state or {}).get("watchdog") or {}).get("trips", 0)

        def move(self, target, duration):
            goal = FollowAnchorPath.Goal()
            goal.header.frame_id = args.base_frame
            goal.points = [Point(x=float(target[0]), y=float(target[1]), z=float(target[2]))]
            goal.times = [Duration(sec=int(duration), nanosec=int((duration % 1) * 1e9))]
            if not self.action.wait_for_server(timeout_sec=3.0):
                raise Stop("FollowAnchorPath server not available.")
            handle = self.wait(self.action.send_goal_async(goal), 3.0, "the action server")
            if not handle.accepted:
                raise Stop("Goal rejected.")
            self.goal = handle
            res = self.wait(handle.get_result_async(), duration + 30.0, "the trajectory").result
            self.goal = None
            return res

    rclpy.init()
    node = Runner()
    ex = MultiThreadedExecutor()
    ex.add_node(node)
    import threading
    threading.Thread(target=ex.spin, daemon=True).start()
    levels = parse_ladder(args.ladder)
    results, original = [], None
    try:
        end = time.monotonic() + 3.0
        while (node.state is None or node.q is None) and time.monotonic() < end:
            time.sleep(0.05)
        if node.torque != "ENABLED":
            raise Stop(f"Torque control is {node.torque or 'unknown'}; enable it first.")
        trk = ((node.state or {}).get("springs") or {}).get("tracking") or {}
        if not trk:
            raise Stop("No ~/trajectory/state from virtual_spring_node (springcontroller "
                       "rebuilt and running?).")
        if (node.state.get("watchdog") or {}).get("softened"):
            raise Stop("The oscillation watchdog has softened the spring; release the "
                       "trajectory hold or set the gains first, then rerun.")
        original = (trk["stiffness"], trk["damping"])
        node.log(f"current gains {original[0]:g} N/m / {original[1]:g} N*s/m; ladder {levels}")
        names = list(node.q.name)
        q_now = [node.q.position[names.index(f"j2s7s300_joint_{i}")] for i in range(1, 8)]
        start = node.gripper()
        moves = step_moves(start, args.step, args.axes)
        if max(peak_speed(m) for m in moves) > args.max_speed:
            raise Stop("--step too large for 0.3 s steps under the anchor speed limit.")
        for k, b in levels:
            swing, _ = model.simulate(q_now, model.gains(k, b),
                                      delay_ticks=int(round(args.delay_ms / 10)))
            predicted = "unstable" if swing > 1.0 else "stable"
            node.log(f"level {k:g}/{b:g}: model predicts {predicted} at this pose")
            if predicted == "unstable" and not args.allow_predicted_unstable:
                node.log("  skipped (predicted unstable; --allow-predicted-unstable to try)")
                results.append({"stiffness": k, "damping": b, "predicted": predicted,
                                "skipped": True})
                continue
            if not args.no_prompt and input(f"Run level {k:g}/{b:g}? [Enter = go, q = stop] "
                                            ).strip().lower() == "q":
                break
            node.set_gains(k, b)
            trips0 = node.trips()
            node.move(start, 2.0)                               # hold here first
            level = {"stiffness": k, "damping": b, "predicted": predicted, "steps": []}
            stop = None
            for mv in moves:
                t0 = node.get_clock().now()
                res = node.move(mv["target"], mv["duration"])
                time.sleep(args.watch)
                t_end = node.get_clock().now()
                ts, ps = [], []
                n = int((t_end - t0).nanoseconds / 1e7)
                for i in range(n):
                    when = Time(nanoseconds=t0.nanoseconds + i * 10_000_000)
                    try:
                        ps.append(node.gripper(when))
                        ts.append(i * 0.01)
                    except Exception:
                        continue
                a = analyse_step(ts, ps, mv["from"], mv["target"], mv["duration"])
                a.update(label=mv["label"], result=res.error_code)
                level["steps"].append(a)
                node.log(f"  {mv['label']:8}: overshoot {a['overshoot_pct']:5.1f} %, settle "
                         f"{a['settle_s']} s, ring {a['ring_mm']:.2f} mm @ {a['ring_hz']} Hz")
                if node.trips() > trips0:
                    stop = "oscillation watchdog tripped"
                elif res.error_code != 0:
                    stop = f"goal ended: {res.error_string}"
                if stop:
                    break
            level["stopped"] = stop
            results.append(level)
            if stop:
                node.log(f"stopping the ladder at {k:g}/{b:g}: {stop}")
                break
        return 0
    except (Stop, KeyboardInterrupt) as exc:
        if node.goal is not None:
            node.goal.cancel_goal_async()
            time.sleep(0.3)
        node.log(f"stopped: {'Ctrl-C' if isinstance(exc, KeyboardInterrupt) else exc}")
        return 1
    finally:
        if original is not None:
            try:
                node.set_gains(*original)
                node.log(f"gains restored to {original[0]:g} / {original[1]:g}")
            except Stop as exc:
                print(f"COULD NOT restore gains ({exc}): ros2 param set /virtual_spring_node "
                      f"trajectory_spring.stiffness {original[0]} (and damping {original[1]})")
        if results:
            out = Path(args.out or os.path.expanduser(
                f"~/gen2_bags/step_test_{time.strftime('%Y_%m_%d-%H_%M_%S')}.json"))
            out.write_text(json.dumps({"ladder": args.ladder, "step_m": args.step,
                                       "axes": args.axes, "levels": results}, indent=1))
            print(f"\nResults: {out}")
            for lv in results:
                if lv.get("skipped"):
                    print(f"  {lv['stiffness']:g}/{lv['damping']:g}: skipped (model: unstable)")
                    continue
                worst = max(lv["steps"], key=lambda s: s["ring_mm"]) if lv["steps"] else {}
                print(f"  {lv['stiffness']:g}/{lv['damping']:g}: max overshoot "
                      f"{max((s['overshoot_pct'] for s in lv['steps']), default=0):.0f} %, "
                      f"worst ringing {worst.get('ring_mm', 0):.2f} mm @ "
                      f"{worst.get('ring_hz', 0)} Hz"
                      + (f"  STOPPED: {lv['stopped']}" if lv.get("stopped") else ""))
        ex.shutdown()
        node.destroy_node()
        rclpy.shutdown()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ladder", default="60:5,80:5,100:5,100:10,150:5",
                    help="stiffness:damping levels, in order")
    ap.add_argument("--step", type=float, default=0.015, help="m (0.015)")
    ap.add_argument("--axes", default="xyz")
    ap.add_argument("--watch", type=float, default=2.5, help="s watched after each step")
    ap.add_argument("--delay-ms", type=int, default=60, help="loop delay for the model check")
    ap.add_argument("--allow-predicted-unstable", action="store_true")
    ap.add_argument("--no-prompt", action="store_true")
    ap.add_argument("--max-speed", type=float, default=0.10,
                    help="the node's trajectory_spring.max_ee_speed_mps")
    ap.add_argument("--base-frame", default="world")
    ap.add_argument("--link", default="j2s7s300_ee_link")
    ap.add_argument("--joint-states", default="/gen2/joint_states_lowlevel")
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    parse_ladder(args.ladder)
    if not set(args.axes) <= set("xyz") or not args.axes:
        ap.error("--axes: some of x, y, z")
    if not 0.002 <= args.step <= 0.03:
        ap.error("--step must be 2-30 mm")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
