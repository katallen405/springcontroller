#!/usr/bin/env python3
"""
spring_viz.py

2-DOF arm visualizer that closes the control loop with the spring controller.

Publishes:
  /joint_states                  -- current joint positions and velocities

Subscribes to:
  /virtual_spring/joint_torques  -- torques from the spring controller

The simulator is purely kinematic: torque directly drives velocity,
scaled by a gain. Velocity is then integrated to get position.

  qdot = torque * velocity_gain
  q   += qdot * dt

Usage:
  python3 spring_viz.py [--velocity-gain 0.05] 

Springs are displayed but not simulated here -- the spring controller
handles all force math. This script just closes the loop and visualizes.
"""

import argparse
import threading
import time
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PointStamped
from std_msgs.msg import String
import json

# ---------------------------------------------------------------------------
# Forward kinematics (XZ plane, joints rotate around Y)
# ---------------------------------------------------------------------------

def fk_2dof(q, link_lengths):
    L1, L2 = link_lengths
    q1, q2 = q
    p0 = np.array([0.0, 0.0])
    p1 = p0 + L1 * np.array([np.sin(q1), -np.cos(q1)])
    p2 = p1 + L2 * np.array([np.sin(q1 + q2), -np.cos(q1 + q2)])
    return p0, p1, p2


def attachment_world_xz(q, link_lengths, frame, local_point):
    lx, ly, lz = local_point

    # Frames on joint_1/link_1 — attached at base, rotate with q[0]
    joint1_frames = {"joint_1", "link_1"}
    # Everything else is assumed to be on joint_2/end effector
    if frame in joint1_frames:
        origin = np.array([0.0, 0.0])
        angle = q[0]
    if "end_effector" in frame:
        origin = fk_2dof(q, link_lengths)[2]
        angle = q[0]+q[1]
    else:
        origin = fk_2dof(q, link_lengths)[1]
        angle = q[0] + q[1]

    along = np.array([ np.sin(angle), -np.cos(angle)])
    perp  = np.array([ np.cos(angle),  np.sin(angle)])
    return origin + lz * along + lx * perp


# ---------------------------------------------------------------------------
# ROS2 node
# ---------------------------------------------------------------------------

class ArmSimNode(Node):
    def __init__(self, velocity_gain: float):
        super().__init__("spring_viz")
        self.velocity_gain = velocity_gain

        self.lock  = threading.Lock()
        self.q     = np.zeros(2)
        self.qdot  = np.zeros(2)
        self._last_torque_time = time.monotonic()
        self._torque_timeout   = 0.5  # s — stop moving if no torques received

        self._js_pub = self.create_publisher(JointState, "/joint_states", 10)

        self.create_subscription(
            JointState,
            "/virtual_spring/joint_torques",
            self._torque_cb,
            10,
        )
        self.create_subscription(
            String,
            "/virtual_spring_node/springs_updated",
            self._springs_updated_cb,
            10,
        )

        self._springs_changed_callback = None # set by main after setup

        self.create_timer(0.01, self._step)      # integrate at 100 Hz
        self.create_timer(0.02, self._publish)   # publish at 50 Hz

    def _torque_cb(self, msg: JointState) -> None:
        if len(msg.effort) < 2:
            return
        with self.lock:
            self.qdot = np.array(msg.effort[:2]) * self.velocity_gain
            self._last_torque_time = time.monotonic()

    def _step(self) -> None:
        with self.lock:
            if time.monotonic() - self._last_torque_time > self._torque_timeout:
                self.qdot = np.zeros(2)
            self.q += self.qdot * 0.01

    def _publish(self) -> None:
        with self.lock:
            q    = self.q.copy()
            qdot = self.qdot.copy()
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name     = ["joint_1", "joint_2"]
        msg.position = q.tolist()
        msg.velocity = qdot.tolist()
        self._js_pub.publish(msg)

    def get_state(self):
        with self.lock:
            return self.q.copy(), self.qdot.copy()

    def watch_targets(self, springs: list) -> None:
        """Subscribe to target update topics and keep spring display in sync."""
        for s in springs:
            topic = f"/virtual_spring_node/target/{s['name']}"
            self.create_subscription(
                PointStamped,
                topic,
                lambda msg, spring=s: self._target_watch_cb(msg, spring),
                10,
            )

    def _target_watch_cb(self, msg: PointStamped, spring: dict) -> None:
        spring["target_xz"] = np.array([msg.point.x, msg.point.z])

    def read_springs_from_params(self) -> list:
        """Read spring definitions from the virtual_spring_node parameter server."""
        self.get_logger().info("Getting spring definitions from /virtual_spring_node")
        from rclpy.parameter import Parameter
        import subprocess, yaml

        # Use ros2 param dump to get all params from the controller node
        result = subprocess.run(
            ["ros2", "param", "dump", "/virtual_spring_node"],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            self.get_logger().warn("Could not read params from virtual_spring_node — using defaults")
            return None
        
        data = yaml.safe_load(result.stdout)
        ros_params = data.get("/virtual_spring_node", {}).get("ros__parameters", {})
        springs_dict = ros_params.get("springs", {})
        spring_names = ros_params.get("spring_names",{})
        springs = []
        for name in spring_names:
            print("getting spring name",name)
            p = springs_dict[name]
            print("for spring ", name," p=",p)
            if not p:
                print("no spring by name ", name)
                continue
            target = p.get("target", {})
            springs.append({
                "name":   name,
                "frame":  p.get("link_name", ""),
                "local":  p.get("local_point", {}),
                "target": target,
                "target_xz": np.array([target[0], target[2]]),
            })

        return springs if springs else None
    def _springs_updated_cb(self, msg:String) -> None:
        if self._springs_changed_callback is not None:
            spring_names = json.loads(msg.data)
            self._springs_changed_callback(spring_names)
            
# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_spring(spec):
    parts  = spec.split(":")
    frame  = parts[0]
    local  = list(map(float, parts[1].split(",")))
    target = list(map(float, parts[2].split(",")))
    return {"name": name, "frame": frame, "local": local, "target": target}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    # todo- fetch this from URDF?
    parser.add_argument("--link-lengths", nargs=2, type=float, default=[0.5, 0.5])
    parser.add_argument("--velocity-gain", type=float, default=0.05,
                        help="rad/s per N·m")
    parser.add_argument("--springs", nargs="*", default=[],
                        help="frame:lx,ly,lz:tx,ty,tz  (display only)")
    args = parser.parse_args()

    link_lengths = args.link_lengths

    # Start ROS2
    rclpy.init()
    node = ArmSimNode(velocity_gain=args.velocity_gain)
    # Try to read springs from the controller's parameter server
    springs = node.read_springs_from_params()
    if springs:
        for s in springs:
            print(f"Spring '{s['name']}': frame='{s['frame']}', local={s['local']}")
    
    if springs is None:
        print("Falling back to default springs")
        springs = [
            {"name": "tip_spring",   "frame": "end_effector", "local": [0.0, 0.0, 0.1], "target": [0.5, 0.0, 0.8]},
            {"name": "elbow_spring", "frame": "link_1",       "local": [0.0, 0.0, 0.2], "target": [0.2, 0.0, 0.5]},
        ]
        for s in springs:
            s["target_xz"] = np.array([s["target"][0], s["target"][2]])
    
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()
    node.watch_targets(springs)
    # --- Plot ---
    fig, ax = plt.subplots(figsize=(7, 7))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect("equal")
    ax.set_xlabel("X (m)", color="#aaaacc")
    ax.set_ylabel("Z (m)", color="#aaaacc")
    ax.set_title("2-DOF Arm — Closed-Loop Spring Sim", color="white", fontsize=13)
    ax.tick_params(colors="#aaaacc")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333355")
    ax.grid(True, color="#222244", linewidth=0.5)
    ax.axhline(0, color="#333355", linewidth=0.8)
    ax.axvline(0, color="#333355", linewidth=0.8)

    arm_line, = ax.plot([], [], "o-", color="#00d4ff", linewidth=3,
                        markersize=7, markerfacecolor="white", zorder=5)

    colors = ["#ff6b6b", "#ffd93d", "#6bcb77", "#a29bfe"]
    spring_artists = []
    for i, s in enumerate(springs):
        c = colors[i % len(colors)]
        att_dot,    = ax.plot([], [], "o",  color=c, markersize=10, zorder=6)
        tgt_dot,    = ax.plot([s["target_xz"][0]], [s["target_xz"][1]],
                              "*", color=c, markersize=14, zorder=6,
                              label=f"{s['name']} target")
        spring_line,= ax.plot([], [], "--", color=c, linewidth=1.5,
                              alpha=0.7, zorder=4)
        spring_artists.append({
            "att_dot": att_dot, "tgt_dot": tgt_dot,
            "spring_line": spring_line, "color": c, "arrow": None,
        })

    ax.legend(loc="upper right", facecolor="#1a1a2e", edgecolor="#333355",
              labelcolor="white", fontsize=9)

    info = ax.text(0.02, 0.97, "", transform=ax.transAxes, color="white",
                   fontsize=9, va="top", fontfamily="monospace")
    def on_springs_changed(spring_names):
        # Re-read full definitions from param server
        new_springs = node.read_springs_from_params()
        if new_springs is None:
            return
    
        # Remove artists for springs that no longer exist
        current_names = {s["name"] for s in springs}
        new_names = {s["name"] for s in new_springs}
    
        for name in current_names - new_names:
            idx = next(i for i, s in enumerate(springs) if s["name"] == name)
            sa = spring_artists.pop(idx)
            springs.pop(idx)
            sa["att_dot"].remove()
            sa["tgt_dot"].remove()
            sa["spring_line"].remove()
            if sa["arrow"] is not None:
                sa["arrow"].remove()
    
        # Add artists for new springs
        for s in new_springs:
            if s["name"] not in {sp["name"] for sp in springs}:
                springs.append(s)
                c = colors[len(spring_artists) % len(colors)]
                att_dot,     = ax.plot([], [], "o",  color=c, markersize=10, zorder=6)
                tgt_dot,     = ax.plot([s["target_xz"][0]], [s["target_xz"][1]],
                                       "*", color=c, markersize=14, zorder=6,
                                       label=f"{s['name']} target")
                spring_line, = ax.plot([], [], "--", color=c, linewidth=1.5,
                                       alpha=0.7, zorder=4)
                spring_artists.append({
                    "att_dot": att_dot, "tgt_dot": tgt_dot,
                    "spring_line": spring_line, "color": c, "arrow": None,
                })
            node.watch_targets([s for s in new_springs if s["name"] not in current_names])
    
        ax.legend(loc="upper right", facecolor="#1a1a2e", edgecolor="#333355",
              labelcolor="white", fontsize=9)

    node._springs_changed_callback = on_springs_changed
    
    def update(_frame):
        q, qdot = node.get_state()
        print("state q=", q, "qdot=", qdot)
        p0, p1, p2 = fk_2dof(q, link_lengths)
        arm_line.set_data([p0[0], p1[0], p2[0]], [p0[1], p1[1], p2[1]])

        for i, s in enumerate(springs):
            att = attachment_world_xz(q, link_lengths, s["frame"], s["local"])
            tgt = s["target_xz"]
            sa  = spring_artists[i]
            sa["att_dot"].set_data([att[0]], [att[1]])
            sa["tgt_dot"].set_data([tgt[0]], [tgt[1]]) 
            sa["spring_line"].set_data([att[0], tgt[0]], [att[1], tgt[1]])

            if sa["arrow"] is not None:
                sa["arrow"].remove()
                sa["arrow"] = None

            disp = tgt - att
            norm = np.linalg.norm(disp)
            if norm > 0.01:
                scale = min(norm, 0.3)
                sa["arrow"] = ax.annotate(
                    "", xy=att + disp / norm * scale, xytext=att,
                    arrowprops=dict(arrowstyle="->", color=sa["color"], lw=2.0),
                    zorder=7,
                )

        info.set_text(
            f"q1 = {np.degrees(q[0]):+7.2f}°   qdot1 = {np.degrees(qdot[0]):+6.2f}°/s\n"
            f"q2 = {np.degrees(q[1]):+7.2f}°   qdot2 = {np.degrees(qdot[1]):+6.2f}°/s"
        )
        return [arm_line, info] + [
            x for sa in spring_artists
            for x in [sa["att_dot"], sa["tgt_dot"], sa["spring_line"]]
        ]

    ani = FuncAnimation(fig, update, interval=50, blit=False)  # must keep reference
    plt.tight_layout()
    try:
        plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
