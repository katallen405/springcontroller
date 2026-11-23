#!/usr/bin/env python3
"""Pick up a small object and put it down somewhere else, compliantly (Gen2).

Usage (ROS and the workspace sourced; robot launched with
gen2_spring.launch.py start_bridge:=true ..., torque enabled; hand on the
E-STOP):
    ~/.springcontroller_venv/bin/python springcontroller/tools/pick_place_test.py
    ... --pick X Y Z --place X Y Z      # skip teaching (printed by a taught run)
    ... --object-mass 0.08              # kg; default: weigh it after the lift

What it does:
  1. Checks torque is on and the gripper is activated (offers to activate:
     the jaws sweep fully open and closed), then opens the gripper.
  2. Teach (unless --pick/--place): springs off -- the arm floats on gravity
     compensation, so hold it -- you push the open gripper around the object
     and press Enter, then to the place spot and Enter; each records the
     gripper point (world -> j2s7s300_ee_link, the trajectory spring's
     attachment point). Springs back on, holding where the arm is.
  3. With FollowAnchorPath goals (position only, rotation free): above the
     pick spot (--approach, 8 cm) -> record the spring's vertical pull ->
     straight down -> check it got there -> close -> check it holds an
     object -> payload on (--object-mass) or lift and weigh it -> above the
     place spot -> down -> check -> open -> payload off -> lift.
  Every step is printed and logged to /rosout as "pick_place: ...", so
  tools/bag_replay.py lists the phases. Any failure or Ctrl-C stops: the
  current goal is cancelled, the gripper is left as it is (a held object is
  never dropped), the springs hold where the arm is.

The planning and weighing helpers at the top are plain Python (tested in
test/test_pick_place_plan.py); ROS is only imported to run.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from dataclasses import dataclass, field

import numpy as np

GRAVITY = 9.81
MIN_JERK_PEAK = 1.875        # peak/average speed of a minimum-jerk segment
TABLE_TOP_Z = -0.03          # Gen2 base frame (gen2_collision_scene.yaml)


# ── planning (plain Python) ───────────────────────────────────────────────────

@dataclass
class Step:
    kind: str                # move | baseline | close | payload | weigh | open | unload
    name: str
    point: list = field(default_factory=list)   # move target (base frame)
    duration: float = 0.0
    check_reach: bool = False


def segment_time(a, b, speed: float, minimum: float = 1.5) -> float:
    """Duration so a minimum-jerk move a->b peaks at no more than speed."""
    dist = float(np.linalg.norm(np.asarray(b, float) - np.asarray(a, float)))
    return max(minimum, MIN_JERK_PEAK * dist / speed)


def plan_moves(start, pick, place, approach: float = 0.08, speed: float = 0.05,
               descent: float = 0.02, weigh: bool = True) -> list[Step]:
    """The pick-and-place sequence as steps, from the gripper point `start`."""
    for name, p in (("pick", pick), ("place", place)):
        if len(p) != 3 or not np.all(np.isfinite(p)):
            raise ValueError(f"{name} must be three finite numbers")
        if p[2] < TABLE_TOP_Z + 0.02:
            raise ValueError(f"{name} z = {p[2]:.3f} m is within 2 cm of the table top "
                             f"(z = {TABLE_TOP_Z}); teach it higher")
    if approach <= 0 or speed <= 0 or descent <= 0:
        raise ValueError("approach, speed and descent speed must be > 0")
    up = np.array([0.0, 0.0, approach])
    pick, place, start = (np.asarray(v, float) for v in (pick, place, start))
    above_pick, above_place = pick + up, place + up
    steps = [
        Step("move", "to above the pick spot", above_pick.tolist(),
             segment_time(start, above_pick, speed)),
        Step("baseline", "record the spring's vertical pull before the grasp"),
        Step("move", "down to the pick spot", pick.tolist(),
             segment_time(above_pick, pick, descent), check_reach=True),
        Step("close", "close the gripper"),
    ]
    if not weigh:
        steps.append(Step("payload", "apply the given object mass"))
    steps.append(Step("move", "lift", above_pick.tolist(), segment_time(pick, above_pick, descent)))
    if weigh:
        steps.append(Step("weigh", "weigh the object and apply its mass"))
    steps += [
        Step("move", "to above the place spot", above_place.tolist(),
             segment_time(above_pick, above_place, speed)),
        Step("move", "down to the place spot", place.tolist(),
             segment_time(above_place, place, descent), check_reach=True),
        Step("open", "open the gripper"),
        Step("unload", "remove the object mass"),
        Step("move", "lift clear", above_place.tolist(), segment_time(place, above_place, descent)),
    ]
    return steps


def object_mass(fz_before: float, fz_after: float, max_kg: float = 1.0,
                min_kg: float = 0.01) -> tuple[float, str]:
    """Mass from the spring's vertical pull before and after picking up, at
    the same pose: the extra upward pull holds the object. (mass, note)."""
    m = (fz_after - fz_before) / GRAVITY
    if m < min_kg:
        return 0.0, (f"measured {m * 1000:.0f} g: too light to tell from friction; "
                     "using 0 (give --object-mass to set it)")
    if m > max_kg:
        return 0.0, (f"measured {m:.2f} kg, above the {max_kg} kg limit; using 0 "
                     "(check the grasp, or give --object-mass)")
    return m, f"measured {m * 1000:.0f} g"


# ── ROS runner ────────────────────────────────────────────────────────────────

class Stop(Exception):
    """Ends the sequence with a message; the arm holds where it is."""


def run(args) -> int:
    import rclpy
    from rclpy.action import ActionClient
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.node import Node
    from rclpy.time import Time
    from builtin_interfaces.msg import Duration
    from geometry_msgs.msg import Point
    from std_msgs.msg import String
    from std_srvs.srv import SetBool, Trigger
    from tf2_ros import Buffer, TransformListener
    from springcontroller_interfaces.action import FollowAnchorPath
    from springcontroller_interfaces.srv import SetPayload

    class Runner(Node):
        def __init__(self):
            super().__init__("pick_place_test")
            self.tf = Buffer()
            self._tfl = TransformListener(self.tf, self)
            sn = "/virtual_spring_node"
            self.enable = self.create_client(SetBool, f"{sn}/enable")
            self.release = self.create_client(Trigger, f"{sn}/trajectory/release")
            self.payload = self.create_client(SetPayload, f"{sn}/set_payload")
            self.grip = self.create_client(SetBool, "/gen2_torque_control/gripper_command")
            self.activate = self.create_client(Trigger, "/gen2_torque_control/gripper_activate")
            self.action = ActionClient(self, FollowAnchorPath,
                                       "/spring_trajectory_controller/follow_anchor_path")
            self.lock = threading.Lock()
            self.grip_status, self.grip_stamp = None, 0.0
            self.torque, self.fz = None, []
            self.create_subscription(String, "/gen2_torque_control/gripper_status",
                                     self._grip_cb, 10)
            self.create_subscription(String, "/gen2_torque_control/status",
                                     lambda m: setattr(self, "torque", m.data), 10)
            self.create_subscription(String, f"{sn}/trajectory/state", self._state_cb, 10)
            self.goal = None

        def _grip_cb(self, msg):
            with self.lock:
                self.grip_status, self.grip_stamp = json.loads(msg.data), time.monotonic()

        def _state_cb(self, msg):
            anchor = json.loads(msg.data).get("anchor") or {}
            if anchor.get("force") is not None:
                with self.lock:
                    self.fz.append((time.monotonic(), anchor["force"][2]))
                    self.fz = self.fz[-200:]

        # helpers
        def log(self, text):
            print(f"[pick_place] {text}", flush=True)
            self.get_logger().info(f"pick_place: {text}")

        def wait(self, fut, timeout, what):
            end = time.monotonic() + timeout
            while not fut.done():
                if time.monotonic() > end:
                    raise Stop(f"No answer from {what} within {timeout:.0f} s -- is it running?")
                time.sleep(0.02)
            return fut.result()

        def call(self, client, req, what, timeout=5.0):
            if not client.wait_for_service(timeout_sec=timeout):
                raise Stop(f"{what} is not available -- is the robot stack running "
                           "(and springcontroller_interfaces rebuilt)?")
            return self.wait(client.call_async(req), timeout, what)

        def gripper_point(self):
            t = self.tf.lookup_transform(args.base_frame, args.link, Time())
            v = t.transform.translation
            return np.array([v.x, v.y, v.z])

        def status_after(self, t_cmd, timeout, done_codes):
            end = time.monotonic() + timeout
            while time.monotonic() < end:
                with self.lock:
                    st, stamp = self.grip_status, self.grip_stamp
                if st and stamp > t_cmd + 0.3 and st.get("object_code") in done_codes:
                    return st
                time.sleep(0.05)
            return None

        def mean_fz(self, seconds=1.0):
            time.sleep(seconds)
            now = time.monotonic()
            with self.lock:
                vals = [f for t, f in self.fz if now - t <= seconds]
            return float(np.mean(vals)) if vals else None

        def set_payload(self, mass):
            req = SetPayload.Request(mass_kg=float(mass), link_name="", ramp_sec=0.5)
            res = self.call(self.payload, req, "set_payload")
            if not res.success:
                raise Stop(f"set_payload refused: {res.message}")
            self.log(res.message)

        def move(self, step):
            if not self.action.wait_for_server(timeout_sec=5.0):
                raise Stop("FollowAnchorPath server not available.")
            goal = FollowAnchorPath.Goal()
            goal.header.frame_id = args.base_frame
            goal.points = [Point(x=step.point[0], y=step.point[1], z=step.point[2])]
            goal.times = [Duration(sec=int(step.duration),
                                   nanosec=int((step.duration % 1) * 1e9))]
            handle = self.wait(self.action.send_goal_async(goal), 5.0, "the action server")
            if not handle.accepted:
                raise Stop(f"Goal '{step.name}' was rejected.")
            self.goal = handle
            res = self.wait(handle.get_result_async(), step.duration + 40.0, "the trajectory")
            self.goal = None
            r = res.result
            if r.error_code != 0:
                raise Stop(f"'{step.name}' ended with: {r.error_string}")
            self.log(f"{step.name}: done. {r.active_springs}")
            if step.check_reach:
                end = time.monotonic() + 5.0
                while True:
                    err = float(np.linalg.norm(self.gripper_point() - step.point))
                    if err <= args.reach_tol:
                        self.log(f"{step.name}: gripper within {err * 100:.1f} cm")
                        break
                    if time.monotonic() > end:
                        raise Stop(f"After '{step.name}' the gripper is {err * 100:.1f} cm from "
                                   f"the spot (allowed {args.reach_tol * 100:.1f} cm; "
                                   "--reach-tol to relax). Not grasping/releasing.")
                    time.sleep(0.1)

    rclpy.init()
    node = Runner()
    ex = MultiThreadedExecutor()
    ex.add_node(node)
    threading.Thread(target=ex.spin, daemon=True).start()
    held_mass = 0.0
    springs_off = False
    try:
        # 1. preflight
        end = time.monotonic() + 3.0
        while node.torque is None and time.monotonic() < end:
            time.sleep(0.05)
        if node.torque != "ENABLED":
            raise Stop(f"Torque control is {node.torque or 'unknown (no status)'}; enable it: "
                       "ros2 service call /gen2_torque_control/enable std_srvs/srv/SetBool "
                       "\"{data: true}\"")
        end = time.monotonic() + 3.0
        while node.grip_status is None and time.monotonic() < end:
            time.sleep(0.05)
        if not node.grip_status or not node.grip_status.get("connected", True):
            raise Stop("No gripper status -- is the Robotiq connected (gripper_port)?")
        if not node.grip_status.get("activated"):
            ans = input("Gripper not activated. Activate now? The jaws sweep fully open and "
                        "closed -- keep clear. [y/N] ")
            if ans.strip().lower() != "y":
                raise Stop("Gripper not activated.")
            node.call(node.activate, Trigger.Request(), "gripper_activate")
            end = time.monotonic() + 20.0
            while not (node.grip_status or {}).get("activated"):
                if time.monotonic() > end:
                    raise Stop("Gripper didn't finish activating within 20 s.")
                time.sleep(0.2)
            node.log("gripper activated")
        t_cmd = time.monotonic()
        node.call(node.grip, SetBool.Request(data=False), "gripper_command")
        node.status_after(t_cmd, args.grip_timeout, (1, 3))
        node.log("gripper open")

        # 2. teach
        if args.pick is None or args.place is None:
            node.call(node.release, Trigger.Request(), "trajectory/release")
            input("Teaching. Springs go OFF next: the arm floats and may drift slowly -- "
                  "hold it. Press Enter to turn springs off.")
            node.call(node.enable, SetBool.Request(data=False), "springs enable")
            springs_off = True
            input("Move the open gripper around the object, at grasp height. Press Enter "
                  "to record the PICK spot.")
            pick = node.gripper_point()
            node.log(f"pick spot {np.round(pick, 3).tolist()}")
            input("Move the gripper to where the object should go, at the same height above "
                  "the table as when picking. Press Enter to record the PLACE spot.")
            place = node.gripper_point()
            node.log(f"place spot {np.round(place, 3).tolist()}")
            input("Hold the arm where you want it to start. Press Enter to turn springs on "
                  "(they hold where the arm is).")
            node.call(node.enable, SetBool.Request(data=True), "springs enable")
            springs_off = False
            time.sleep(1.0)
            fmt = lambda v: " ".join(f"{c:.3f}" for c in v)  # noqa: E731
            print(f"\nTo repeat without teaching:\n  ... pick_place_test.py --pick {fmt(pick)} "
                  f"--place {fmt(place)}\n", flush=True)
        else:
            pick, place = np.array(args.pick), np.array(args.place)

        # 3. sequence
        steps = plan_moves(node.gripper_point(), pick, place, args.approach, args.speed,
                           args.descent_speed, weigh=args.object_mass is None)
        if input(f"{len(steps)} steps, pick {np.round(pick, 3).tolist()} -> place "
                 f"{np.round(place, 3).tolist()}. Start? [y/N] ").strip().lower() != "y":
            raise Stop("Not started.")
        fz_before = None
        for k, step in enumerate(steps, 1):
            node.log(f"step {k}/{len(steps)}: {step.name}")
            if step.kind == "move":
                node.move(step)
            elif step.kind == "baseline":
                fz_before = node.mean_fz()
                node.log(f"vertical pull before the grasp: "
                         f"{'n/a' if fz_before is None else f'{fz_before:.2f} N'}")
            elif step.kind == "close":
                t_cmd = time.monotonic()
                node.call(node.grip, SetBool.Request(data=True), "gripper_command")
                st = node.status_after(t_cmd, args.grip_timeout, (2, 3))
                if st is None:
                    raise Stop("Gripper didn't report a result within --grip-timeout.")
                if st["object_code"] == 3:
                    node.call(node.grip, SetBool.Request(data=False), "gripper_command")
                    node.move(Step("move", "lift clear (no object)",
                                   (np.asarray(pick) + [0, 0, args.approach]).tolist(),
                                   segment_time(pick, np.asarray(pick) + [0, 0, args.approach],
                                                args.descent_speed)))
                    raise Stop("The gripper closed without an object. Opened and lifted.")
                node.log(f"holding an object (gripper position {st.get('position')}/255)")
            elif step.kind == "payload":
                node.set_payload(args.object_mass)
                held_mass = args.object_mass
            elif step.kind == "weigh":
                fz_after = node.mean_fz()
                if fz_before is None or fz_after is None:
                    m, note = 0.0, "no spring force readings; using 0 (give --object-mass)"
                else:
                    m, note = object_mass(fz_before, fz_after, args.max_payload)
                node.log(f"object: {note}")
                if m > 0:
                    node.set_payload(m)
                    held_mass = m
            elif step.kind == "open":
                t_cmd = time.monotonic()
                node.call(node.grip, SetBool.Request(data=False), "gripper_command")
                if node.status_after(t_cmd, args.grip_timeout, (1, 3)) is None:
                    raise Stop("Gripper didn't report open within --grip-timeout.")
                node.log("released")
            elif step.kind == "unload":
                if held_mass > 0:
                    node.set_payload(0.0)
                    held_mass = 0.0
        node.log("done")
        return 0
    except (Stop, KeyboardInterrupt, ValueError) as exc:
        if node.goal is not None:
            node.goal.cancel_goal_async()
            time.sleep(0.5)
        why = "Ctrl-C" if isinstance(exc, KeyboardInterrupt) else str(exc)
        node.log(f"stopped: {why}")
        if springs_off:
            # Never leave the arm floating on gravity comp alone: springs
            # back on hold it where it is now (they re-anchor on re-enable).
            try:
                node.call(node.enable, SetBool.Request(data=True), "springs enable")
                print("Springs turned back on; they hold the arm where it is.", flush=True)
            except Stop as exc2:
                print(f"COULD NOT turn springs back on ({exc2}). The arm is on gravity "
                      "compensation only: hold it and run\n  ros2 service call "
                      "/virtual_spring_node/enable std_srvs/srv/SetBool \"{data: true}\"",
                      flush=True)
        print("The arm holds where it is; the gripper was left as it was.", flush=True)
        if held_mass > 0:
            print(f"Payload compensation is still on ({held_mass:.3f} kg). After releasing "
                  "the object: ros2 service call /virtual_spring_node/set_payload "
                  "springcontroller_interfaces/srv/SetPayload \"{mass_kg: 0.0}\"", flush=True)
        return 1
    finally:
        ex.shutdown()
        node.destroy_node()
        rclpy.shutdown()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pick", type=float, nargs=3, metavar=("X", "Y", "Z"))
    ap.add_argument("--place", type=float, nargs=3, metavar=("X", "Y", "Z"))
    ap.add_argument("--object-mass", type=float, help="kg; default: weigh it after the lift")
    ap.add_argument("--approach", type=float, default=0.08, help="m above each spot (0.08)")
    ap.add_argument("--speed", type=float, default=0.05, help="m/s between spots (0.05)")
    ap.add_argument("--descent-speed", type=float, default=0.02, help="m/s down/up (0.02)")
    ap.add_argument("--reach-tol", type=float, default=0.015, help="m (0.015)")
    ap.add_argument("--grip-timeout", type=float, default=5.0, help="s (5)")
    ap.add_argument("--max-payload", type=float, default=1.0,
                    help="kg; weighed masses above this are ignored (1.0)")
    ap.add_argument("--base-frame", default="world")
    ap.add_argument("--link", default="j2s7s300_ee_link",
                    help="the trajectory spring's link (its attachment point is the origin)")
    args = ap.parse_args(argv)
    if (args.pick is None) != (args.place is None):
        ap.error("give both --pick and --place, or neither (to teach them)")
    if args.object_mass is not None and not 0.0 < args.object_mass <= args.max_payload:
        ap.error(f"--object-mass must be in (0, {args.max_payload}]")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
